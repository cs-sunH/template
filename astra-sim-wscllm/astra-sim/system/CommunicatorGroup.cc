/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>

#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"

using namespace AstraSim;

namespace {
// Deep-copies a CollectiveImpl owned by the CollectiveImplLookup registry.
// The dynamic type must be preserved: direct implementations carry a payload
// (direct_collective_window) that a base-class copy would slice away, and
// Sys.cc reads it back through a derived-type cast.
CollectiveImpl* clone_collective_impl(const CollectiveImpl* impl) {
    if (impl->type == CollectiveImplType::Direct ||
        impl->type == CollectiveImplType::OneDirect) {
        return new DirectCollectiveImpl(
            impl->type,
            static_cast<const DirectCollectiveImpl*>(impl)
                ->direct_collective_window);
    }
    return new CollectiveImpl(impl->type);
}

// Collective stream-id window registry. The window of logical group id N is
// [N * STREAM_WINDOW, (N + 1) * STREAM_WINDOW): deterministic, therefore
// IDENTICAL on every rank's copy of the same logical group. This is a hard
// simulation invariant: the native algorithm layer (Ring/HalvingDoubling/...)
// sends and receives collective packets with tag = stream->stream_id and the
// network backend pairs them by (tag, src, dst, count, chunk) -- per-rank
// counters may only differ if every rank starts at the same base AND advances
// in lockstep (same graph, same splits), which the online commit boundary
// guarantees. DB1 2026-09-28: the intermediate per-object process-wide slot
// lease handed each rank's copy a different base (tags 520000000 vs
// 529000000 vs ... on the same 6-rank ring), every recv registered under a
// tag no sender ever used, and each rank's ring stalled at its first packet
// -- the "lost-wakeup dead end". Registry duties now:
//   * fail closed on group ids whose window would fold in the COLLECTIVE tag
//     remap (tag % 500000000 + 500000000): id must stay below
//     500000000 / STREAM_WINDOW;
//   * fail closed when a second, differently-owned live group claims the
//     same id (the C1 aliasing concern, answered without breaking the
//     cross-rank invariant);
//   * reference-count live copies per id so the identity record (and the
//     guard) lives exactly as long as the group does.
// The registry fingerprint is the ORDERED RANKS only. F1-CPP C1-2
// (2026-09-29): DB1 compared ranks AND dimensions, which mis-killed two
// legal sequences. (a) A pg-metadata group (Workload.cc constructs it with
// no dimensions) colliding with a same-id file group over the same
// membership is the SAME logical communicator -- dimensions are per-view
// plan-shape metadata that never reaches a stream id or a packet tag, so
// they cannot alias anything and must not panic. (b) A legal same-id
// redefinition is handled by the callers (release the old group BEFORE
// constructing the replacement), so the panic below only fires for a
// genuinely different membership that is still live -- the DB1 contract,
// unchanged. Same id + same ranks = same logical communicator ->
// reference-counted reuse. Group construction/destruction happens on the
// single simulation thread, so no synchronization is required.
constexpr int MAX_DERIVED_GROUP_ID =
    500000000 / CommunicatorGroup::STREAM_WINDOW - 1;

struct GroupWindowRegistration {
    std::vector<int> ranks;
    int live_copies;
};

std::map<int, GroupWindowRegistration> live_group_windows;
}  // namespace

int CommunicatorGroup::lease_stream_window(
    int id, const std::vector<int>& ranks,
    const std::vector<int>& /*dimensions*/) {
    // Both lease failures throw (2026-10-01, channel unification): the
    // callers construct CommunicatorGroup inside try blocks that translate
    // std::exception into critical + exit, so throwing keeps every group-id
    // validation on the same diagnostic channel as the constructor's checks
    // (and keeps these paths testable), instead of bypassing the catch with
    // a raw exit the way the former sys_panic did. release_stream_window
    // below keeps sys_panic on purpose: it also runs from the destructor,
    // where a throw would turn into std::terminate.
    if (id <= 0) {
        throw std::invalid_argument(
            "Communicator group id must be positive");
    }
    if (id > MAX_DERIVED_GROUP_ID) {
        throw std::invalid_argument(
            "communicator group id " + std::to_string(id) +
            " exceeds " + std::to_string(MAX_DERIVED_GROUP_ID) +
            "; its stream window would fold onto another group's collective "
            "tags in the COLLECTIVE remap (tag % 500000000 + 500000000)");
    }
    auto registration = live_group_windows.find(id);
    if (registration != live_group_windows.end()) {
        // Fingerprint = ordered ranks (see the registry comment above): a
        // dims-only difference is the same logical communicator seen
        // through another metadata view (pg metadata vs comm-group file),
        // while a different membership is a genuinely different group and
        // its collective tags would alias this window.
        if (registration->second.ranks != ranks) {
            throw std::runtime_error(
                "communicator group id " + std::to_string(id) +
                " is already live with a different membership; the shared "
                "stream window would alias the two groups' collective tags");
        }
        ++registration->second.live_copies;
    } else {
        live_group_windows.emplace(
            id, GroupWindowRegistration{ranks, 1});
    }
    return id * CommunicatorGroup::STREAM_WINDOW;
}

void CommunicatorGroup::release_stream_window(
    int id, const std::vector<int>& ranks, const std::vector<int>& /*dimensions*/) {
    auto registration = live_group_windows.find(id);
    if (registration == live_group_windows.end()) {
        Sys::sys_panic("communicator group id " + std::to_string(id) +
                       " released a stream window it never leased");
    }
    if (registration->second.ranks != ranks) {
        Sys::sys_panic("communicator group id " + std::to_string(id) +
                       " released a stream window under a different "
                       "membership than the lease");
    }
    if (--registration->second.live_copies == 0) {
        live_group_windows.erase(registration);
    }
}

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
    // Fail closed on malformed rank data here instead of at collective
    // runtime: an empty group builds a zero-node ring, and a duplicated
    // rank passes the dimension-product check below while double-counting
    // one node in every ring built from this group.
    if (involved_NPUs.empty()) {
        throw std::invalid_argument(
            "Communicator group must contain at least one rank");
    }
    std::vector<int> sorted_ranks = involved_NPUs;
    std::sort(sorted_ranks.begin(), sorted_ranks.end());
    if (std::adjacent_find(sorted_ranks.begin(), sorted_ranks.end()) !=
        sorted_ranks.end()) {
        throw std::invalid_argument(
            "Communicator group must not contain duplicate ranks");
    }
    this->involved_NPUs = involved_NPUs;
    this->generator = generator;
    this->dimension_sizes = dimension_sizes;

    if (!dimension_sizes.empty()) {
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
    // set_id runs last so a throwing validation above cannot leak the
    // stream-id slot (the destructor of a not-yet-constructed object never
    // runs). set_id itself can still throw through lease_stream_window's
    // validations, but every lease failure fires before the registry is
    // mutated, so the throw leaves the window registry untouched; the
    // workload-layer callers catch std::exception and translate it into
    // critical + exit.
    set_id(comm_group_id);
    // Communicator rank order is semantic: custom-collective ET file suffixes
    // and algorithm-rank translation both index the preserved input order.
}

CommunicatorGroup::~CommunicatorGroup() {
    for (auto cg : comm_plans) {
        CollectivePlan* cp = cg.second;
        delete cp;
    }
    if (stream_base_ != 0) {
        release_stream_window(this->id, this->involved_NPUs,
                              this->dimension_sizes);
        stream_base_ = 0;
    }
}

void CommunicatorGroup::set_id(int id) {
    // id 0 is reserved for the default comm group, which is all ranks.
    // CTRL+F "default communication group".  Checked instead of asserted:
    // the assert compiles out in release builds and a zero id would
    // silently zero the num_streams base below.
    if (id <= 0) {
        throw std::invalid_argument("Communicator group id must be positive");
    }
    this->id = id;
    // The window base is derived from the LOGICAL group id and is therefore
    // identical on every rank's copy of the group -- the cross-rank tag
    // invariant documented on the registry above. The lease still validates
    // the id range and the membership identity (fail closed), so the former
    // id*1000000 hazards (id >= 500 remap folding; two different live groups
    // sharing one id) stay impossible.
    stream_base_ =
        lease_stream_window(id, this->involved_NPUs, this->dimension_sizes);
    this->num_streams = stream_base_;
}

bool CommunicatorGroup::matches_definition(
    const std::vector<int>& ranks,
    const std::vector<int>& dimensions) const {
    return same_communicator_definition(involved_NPUs, dimension_sizes, ranks,
                                        dimensions);
}

CollectivePlan* CommunicatorGroup::get_collective_plan(ComType comm_type, uint64_t workload_node_id) {
    if (comm_plans.find(comm_type) != comm_plans.end()) {
        return comm_plans[comm_type];
    }

    // F9 follow-up (2026-10-01): gate EVERY plan path through the named
    // topology lookup FIRST. The whole-cluster branch already did; the
    // shaped and sub-cluster branches used to dive straight into
    // get_collective_impl, whose unconfigured-type error names neither the
    // collective type nor the missing '<type>-implementation' system key
    // (a numeric ComType and a generic lookup message). The lookup here is
    // a map hit for any configured type (topologies are pre-built at Sys
    // construction), so this only upgrades the diagnostic.
    generator->get_logical_topology(comm_type);

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
            // TODO: Revisit whether the choice to override with Ring (instead of e.g. first dimension in list)
            // was a good choice.
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
}

