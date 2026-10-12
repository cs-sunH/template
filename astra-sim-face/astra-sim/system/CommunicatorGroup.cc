/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <stdexcept>

#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"

using namespace AstraSim;

namespace {
// Deep-copies a CollectiveImpl owned by the CollectiveImplLookup registry.
// The dynamic type must be preserved: direct and custom implementations carry
// a payload (direct_collective_window / filename) that a base-class copy would
// slice away, and Sys.cc reads it back through derived-type casts.
CollectiveImpl* clone_collective_impl(const CollectiveImpl* impl) {
    if (impl->type == CollectiveImplType::CustomCollectiveImpl) {
        return new CustomCollectiveImpl(
            impl->type,
            static_cast<const CustomCollectiveImpl*>(impl)->filename);
    }
    if (impl->type == CollectiveImplType::Direct ||
        impl->type == CollectiveImplType::OneDirect) {
        return new DirectCollectiveImpl(
            impl->type,
            static_cast<const DirectCollectiveImpl*>(impl)
                ->direct_collective_window);
    }
    return new CollectiveImpl(impl->type);
}
}  // namespace

bool AstraSim::same_communicator_definition(
    const std::vector<int>& lhs_ranks,
    const std::vector<int>& lhs_dimensions,
    const std::vector<int>& rhs_ranks,
    const std::vector<int>& rhs_dimensions) {
    return lhs_ranks == rhs_ranks && lhs_dimensions == rhs_dimensions;
}

int AstraSim::communicator_rank_position(
    const std::vector<int>& ordered_ranks, const int rank) noexcept {
    for (size_t i = 0; i < ordered_ranks.size(); ++i) {
        if (ordered_ranks[i] == rank) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

CommunicatorGroup::CommunicatorGroup(int comm_group_id,
                                     std::vector<int> involved_NPUs,
                                     Sys* generator,
                                     std::vector<int> dimension_sizes) {
    set_id(comm_group_id);
    this->involved_NPUs = involved_NPUs;
    this->generator = generator;
    this->dimension_sizes = dimension_sizes;

    // R7-D fix (2026-09-29): the rank list is trusted input from
    // comm_group.json / pg metadata. An out-of-range rank used to escape to
    // the backend's fluid route (remote crash) or, for a non-member rank, to
    // a RingTopology id_to_index operator[] that silently inserted a wrong
    // position; a duplicated rank silently collapsed two ring positions onto
    // one node (self-ring / deadlock). Both are always configuration errors.
    if (involved_NPUs.empty()) {
        throw std::invalid_argument(
            "Communicator group must list at least one rank");
    }
    {
        std::vector<int> seen = involved_NPUs;
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            throw std::invalid_argument(
                "Communicator group ranks contain duplicates");
        }
        for (int rank : involved_NPUs) {
            if (rank < 0 || rank >= generator->total_nodes) {
                throw std::invalid_argument(
                    "Communicator group rank " + std::to_string(rank) +
                    " is outside the cluster range [0, " +
                    std::to_string(generator->total_nodes) + ")");
            }
        }
    }

    if (!dimension_sizes.empty()) {
        // H4 (R7-A) fix (2026-09-29): the collective issue path walks one
        // queue level per group dimension, but the queue levels are built
        // from the PHYSICAL dimension count. A shaped group with more
        // dimensions than the physical topology read vLevels out of bounds
        // (QueueLevels::levels[level]) with no fail-loud interception
        // anywhere on the chain. Reject the shape at group construction.
        if (dimension_sizes.size() > generator->physical_dims.size()) {
            throw std::invalid_argument(
                "Communicator group declares " +
                std::to_string(dimension_sizes.size()) +
                " dimensions but the physical topology has only " +
                std::to_string(generator->physical_dims.size()));
        }
        int expected_group_size = 1;
        for (int size : dimension_sizes) {
            if (size <= 0) {
                throw std::invalid_argument(
                    "Communicator dimension sizes must be positive");
            }
            expected_group_size *= size;
        }
        if (expected_group_size != static_cast<int>(involved_NPUs.size())) {
            throw std::invalid_argument(
                "Communicator dimension sizes do not match rank-map size");
        }
    }
    // Communicator rank order is semantic: custom-collective ET file suffixes
    // and algorithm-rank translation both index the preserved input order.
    // Computing this position from a sorted copy selects a different ET file
    // for a valid permuted communicator such as [2, 0].
    pos_in_group =
        communicator_rank_position(this->involved_NPUs, generator->id);
}

CommunicatorGroup::~CommunicatorGroup() {
    for (auto cg : comm_plans) {
        CollectivePlan* cp = cg.second;
        delete cp;
    }
}

void CommunicatorGroup::set_id(int id) {
    // id 0 is reserved for the default comm group, which is all ranks.
    // CTRL+F "default communication group"
    // (2026-10-01) the last bare assert in this module, upgraded to a
    // Release-active guard: the existing protection lives in the callers
    // (parse_comm_group_id_strict at both construction points); a third
    // future construction site would otherwise be unprotected, and id<=0
    // here collides this group's stream-id band with the global counter.
    if (id <= 0) {
        Sys::sys_panic("CommunicatorGroup id must be positive, got " +
                       std::to_string(id));
    }
    // Every stream of this group draws its stream_id from the per-group band
    // starting at id * kStreamsPerIdBand, disjoint from the global Sys
    // counter for any realistic run. Two fail-loud bounds cap the id, and
    // the first is by far the tighter one:
    // 1. Sys::front_end_sim_send folds every COLLECTIVE tag with
    //    `tag % (RENDEZVOUS - COLLECTIVE) + COLLECTIVE` into the 5e8-wide
    //    window [COLLECTIVE, RENDEZVOUS). A group's stream ids span
    //    [id * kStreamsPerIdBand, (id + 1) * kStreamsPerIdBand); once the
    //    band crosses that window it wraps onto a lower band, so group id g
    //    and group id g + 500 (and, at id >= 500, the global stream counter)
    //    would emit identical folded tags on the same (src, dst) pair and
    //    their in-flight messages could not be told apart. This caps group
    //    ids at 499.
    // 2. The int narrowing below silently overflows at id >= 2148.
    constexpr int64_t kStreamsPerIdBand = 1000000;
    const int64_t band_base = static_cast<int64_t>(id) * kStreamsPerIdBand;
    if (band_base + kStreamsPerIdBand >
        static_cast<int64_t>(Sys::FrontEndSendRecvType::RENDEZVOUS -
                             Sys::FrontEndSendRecvType::COLLECTIVE)) {
        Sys::sys_panic(
            "communicator group id " + std::to_string(id) +
            " is too large: its stream-id band would wrap in the COLLECTIVE "
            "tag-folding window and collide with another group's tags "
            "(folding bound: group id <= 499, tighter than the int-overflow "
            "bound: group id <= 2147)");
    }
    if (band_base + (kStreamsPerIdBand - 1) >
        static_cast<int64_t>(std::numeric_limits<int>::max())) {
        Sys::sys_panic("communicator group id " + std::to_string(id) +
                       " is too large: its stream-id band would overflow the "
                       "int stream-id range");
    }
    this->id = id;
    this->num_streams = static_cast<int>(band_base);
}

bool CommunicatorGroup::matches_definition(
    const std::vector<int>& ranks,
    const std::vector<int>& dimensions) const {
    return same_communicator_definition(involved_NPUs, dimension_sizes, ranks,
                                        dimensions);
}

CollectivePlan* CommunicatorGroup::get_collective_plan(ComType comm_type, uint64_t workload_node_id) {
    // Upstream design boundary (kept as-is, identical in the sibling repos):
    // plans are cached per ComType only. workload_node_id participates in the
    // per-node custom-implementation lookup on the cache-miss paths below;
    // later calls for the same ComType reuse the first node's plan/impl, so
    // per-node custom implementations inside one communicator are decided by
    // its first collective of that ComType. See the known-limitation note
    // in CollectiveImplLookup.hh about custom algorithms x communicator
    // groups.
    if (comm_plans.find(comm_type) != comm_plans.end()) {
        return comm_plans[comm_type];
    }

    if (!dimension_sizes.empty()) {
        std::vector<CollectiveImpl*> collective_implementation =
            generator->collective_impl_lookup->get_collective_impl(
                comm_type, workload_node_id);
        if (collective_implementation.size() != dimension_sizes.size()) {
            throw std::runtime_error(
                "A shaped communicator requires one native collective "
                "implementation per communicator dimension");
        }
        for (auto* implementation : collective_implementation) {
            if (implementation->type != CollectiveImplType::Ring) {
                throw std::runtime_error(
                    "Shaped communicators currently support ring as the "
                    "per-dimension native collective implementation");
            }
        }

        LogicalTopology* logical_topology = new GeneralComplexTopology(
            generator->id, involved_NPUs, dimension_sizes,
            collective_implementation);
        std::vector<bool> dimensions_involved(dimension_sizes.size(), true);
        comm_plans[comm_type] = new CollectivePlan(
            logical_topology, collective_implementation, dimensions_involved,
            true, false);
        return comm_plans[comm_type];
    } else if (static_cast<uint64_t>(generator->total_nodes) == involved_NPUs.size()) {
        LogicalTopology* logical_topology =
            generator->get_logical_topology(comm_type);
        std::vector<CollectiveImpl*> collective_implementation =
            generator->collective_impl_lookup->get_collective_impl(comm_type, workload_node_id);
        if (collective_implementation.empty()) {
            throw std::runtime_error(
                "No collective implementation registered for a whole-cluster "
                "communicator");
        }
        // Whole-cluster plan: involve every dimension, sized from the actual
        // topology (the old hardcoded 10 under-sized topologies with more
        // than ten dimensions and over-sized all the rest).
        std::vector<bool> dimensions_involved(
            logical_topology->get_num_of_dimensions(), true);
        bool should_be_removed = false;
        comm_plans[comm_type] =
            new CollectivePlan(logical_topology, collective_implementation,
                               dimensions_involved, should_be_removed, false);
        return comm_plans[comm_type];
    } else {
        std::vector<CollectiveImpl*> collective_implementation =
            generator->collective_impl_lookup->get_collective_impl(comm_type, workload_node_id);
        if (collective_implementation.empty()) {
            throw std::runtime_error(
                "No collective implementation registered for a sub-cluster "
                "communicator");
        }
        if (collective_implementation.size() > 1) {
            // This means that everything fell through and we got a native collective that is multi-dimensional.
            // (Custom collective always assumes 1 dimension).
            // The current logic requires that, for a comm group that is smaller than the whole cluster,
            // we reduce everything to one dimension since the logical dimension no longer matches/matters.
            // Status note (2026-09-26 review, replaces the bare upstream
            // TODO): overriding with Ring here is the current, intentional
            // behavior for multi-dimensional native collectives on a
            // sub-cluster group; alternative policies (e.g. first dimension
            // in list) remain unexplored by design.
            collective_implementation = std::vector<CollectiveImpl*>{
                new CollectiveImpl(CollectiveImplType::Ring)};
        } else if (collective_implementation.size() == 1) {
            // The lookup registry owns and shares this implementation across
            // all comm groups, while the plan below deletes its
            // implementations on destruction (implementations_should_be_removed
            // = true). Handing the shared pointer to the plan double-frees once
            // two sub-cluster groups of the same ComType destruct and leaves
            // the registry dangling afterwards. Give the plan its own copy.
            collective_implementation = std::vector<CollectiveImpl*>{
                clone_collective_impl(collective_implementation[0])};
        }
        LogicalTopology* logical_topology = new RingTopology(
            RingTopology::Dimension::Local, generator->id, involved_NPUs);
        std::vector<bool> dimensions_involved(1, true);
        bool should_be_removed = true;
        comm_plans[comm_type] =
            new CollectivePlan(logical_topology, collective_implementation,
                               dimensions_involved, should_be_removed, true);
        return comm_plans[comm_type];
    }
    assert(false);
    return nullptr;
}

int CommunicatorGroup::get_position_in_group() {
    return pos_in_group;
}
