/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <stdexcept>

#include "astra-sim/common/Logging.hh"

#include "astra-sim/system/CollectivePlan.hh"
#include "astra-sim/system/QueueLevels.hh"
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

    // involved_NPUs domain validation (sync-cpp-E5, 2026-09-30, ported from
    // the joint repo): the rank list is trusted input from comm_group.json /
    // pg metadata. An out-of-range rank used to escape to the backend's
    // fluid route (remote crash) and a duplicated rank silently collapsed
    // two ring positions onto one node (self-ring / hang). Both are always
    // configuration errors.
    // Fail closed, not throw (M2, face-LRU audit 2026-09-29): every
    // construction site sits on an issue path (the file loader inside the
    // Workload build, or the pytorch pg metadata path), so a thrown
    // invalid_argument either terminates without diagnostics or gets
    // swallowed by Sys::call_events and strands the issuing node.
    if (involved_NPUs.empty()) {
        LoggerFactory::get_logger("system::CommunicatorGroup")
            ->critical("######### Exiting because a communicator group must "
                       "list at least one rank #########");
        std::exit(1);
    }
    {
        std::vector<int> seen = involved_NPUs;
        std::sort(seen.begin(), seen.end());
        if (std::adjacent_find(seen.begin(), seen.end()) != seen.end()) {
            LoggerFactory::get_logger("system::CommunicatorGroup")
                ->critical("######### Exiting because communicator group "
                           "ranks contain duplicates #########");
            std::exit(1);
        }
        for (int rank : involved_NPUs) {
            if (rank < 0 || rank >= generator->total_nodes) {
                LoggerFactory::get_logger("system::CommunicatorGroup")
                    ->critical("######### Exiting because communicator group "
                               "rank {} is outside the cluster range [0, {}) "
                               "#########",
                               rank, generator->total_nodes);
                std::exit(1);
            }
        }
    }

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
            // Fail closed (M2): every construction site sits on an issue
            // path (file loader inside the Workload build, or the pytorch pg
            // metadata path), so a thrown invalid_argument either terminates
            // without diagnostics or gets swallowed by Sys::call_events and
            // strands the issuing node.
            LoggerFactory::get_logger("system::CommunicatorGroup")
                ->critical("######### Exiting because communicator "
                           "dimension sizes do not match rank-map size "
                           "({} != {}) #########",
                           expected_group_size, involved_NPUs.size());
            std::exit(1);
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
    // CTRL+F "default communication group". Fail closed (not assert): these
    // constructors run on issue paths where a thrown/compiled-out check
    // would either strand the issuing node or pass silently in Release.
    if (id <= 0) {
        LoggerFactory::get_logger("system::CommunicatorGroup")
            ->critical("######### Exiting because comm group id must be "
                       "positive (got {}) #########",
                       id);
        std::exit(1);
    }
    this->id = id;
    // Each comm group owns the disjoint stream-id block [id*1e6, (id+1)*1e6).
    // Collective tags are folded into [COLLECTIVE, RENDEZVOUS) via
    // `tag % 5e8`, so any block base at or beyond 5e8 (id >= 500) wraps into
    // another group's tag space and cross-group tag collisions deliver
    // streams to the wrong group; and past INT_MAX/1e6 the block itself
    // overflows. Refuse both instead of failing silently.
    if (id >= 500 ||
        id > std::numeric_limits<int>::max() / 1000000) {
        LoggerFactory::get_logger("system::CommunicatorGroup")
            ->critical("######### Exiting because comm group id {} exceeds "
                       "the 500-group tag-space bound (its stream-id block "
                       "would wrap or overflow) #########",
                       id);
        std::exit(1);
    }
    this->num_streams = id * 1000000;
}

bool CommunicatorGroup::matches_definition(
    const std::vector<int>& ranks,
    const std::vector<int>& dimensions) const {
    return same_communicator_definition(involved_NPUs, dimension_sizes, ranks,
                                        dimensions);
}

CollectivePlan* CommunicatorGroup::get_collective_plan(ComType comm_type, uint64_t workload_node_id) {
    // C1: resolve the implementation set for THIS workload node first and
    // cache per (ComType, implementation set).  The lookup gives
    // per-workload-node custom mappings priority, so caching per ComType
    // alone pinned every later same-type collective of this group to the
    // first node's algorithm.
    std::vector<CollectiveImpl*> looked_up_impls;
    try {
        looked_up_impls = generator->collective_impl_lookup->get_collective_impl(
            comm_type, workload_node_id);
    } catch (const std::exception& e) {
        // M2 (face-LRU audit, 2026-09-29): this runs on the collective issue
        // path -- static under Sys::call_events (which swallows the throw and
        // strands the issuing node) or online inside the deferred commit (and
        // an escape would unwind through proceed_to_next_vnet_baseline,
        // leaking the stream and its queue slot). A missing/unresolvable
        // implementation registration is a configuration fatality: report it
        // and exit.
        LoggerFactory::get_logger("system::CommunicatorGroup")
            ->critical("######### Exiting because the collective "
                       "implementation lookup failed (com_type={}, "
                       "workload_node_id={}): {} #########",
                       static_cast<int>(comm_type), workload_node_id, e.what());
        std::exit(1);
    }
    const CommunicatorGroup::PlanImplKey plan_key{
        looked_up_impls.begin(), looked_up_impls.end()};
    const auto cached =
        comm_plans.find({comm_type, plan_key});
    if (cached != comm_plans.end()) {
        return cached->second;
    }

    if (!dimension_sizes.empty()) {
        // H1 (face-LRU audit, 2026-09-29): Sys::generate_collective indexes
        // the physical queue levels with this shape's logical dimension
        // index, so a shaped communicator declaring more dimensions than the
        // physical system reads past vLevels->levels (UB: garbage queue_id,
        // silent map growth or crash). Same fail-closed discipline as the
        // whole-cluster path's GeneralComplexTopology implementation-count
        // check.
        const int physical_dims = generator->vLevels->num_levels();
        if (static_cast<int>(dimension_sizes.size()) > physical_dims) {
            LoggerFactory::get_logger("system::CommunicatorGroup")
                ->critical("######### Exiting because a shaped communicator "
                           "declares {} dimensions but the physical system "
                           "has only {} queue levels #########",
                           dimension_sizes.size(), physical_dims);
            std::exit(1);
        }
        std::vector<CollectiveImpl*> collective_implementation = looked_up_impls;
        if (collective_implementation.size() != dimension_sizes.size()) {
            LoggerFactory::get_logger("system::CommunicatorGroup")
                ->critical("######### Exiting because a shaped communicator "
                           "requires one native collective implementation per "
                           "communicator dimension ({} impls vs {} "
                           "dimensions) #########",
                           collective_implementation.size(),
                           dimension_sizes.size());
            std::exit(1);
        }
        for (auto* implementation : collective_implementation) {
            if (implementation->type != CollectiveImplType::Ring) {
                LoggerFactory::get_logger("system::CommunicatorGroup")
                    ->critical("######### Exiting because shaped "
                               "communicators currently support ring as the "
                               "per-dimension native collective "
                               "implementation #########");
                std::exit(1);
            }
        }

        LogicalTopology* logical_topology = new GeneralComplexTopology(
            generator->id, involved_NPUs, dimension_sizes,
            collective_implementation);
        std::vector<bool> dimensions_involved(dimension_sizes.size(), true);
        CollectivePlan* plan = new CollectivePlan(
            logical_topology, collective_implementation, dimensions_involved,
            true, false);
        comm_plans[{comm_type, plan_key}] = plan;
        return plan;
    } else if (static_cast<uint64_t>(generator->total_nodes) == involved_NPUs.size()) {
        LogicalTopology* logical_topology =
            generator->get_logical_topology(comm_type);
        std::vector<CollectiveImpl*> collective_implementation = looked_up_impls;
        if (collective_implementation.empty()) {
            LoggerFactory::get_logger("system::CommunicatorGroup")
                ->critical("######### Exiting because no collective "
                           "implementation is registered for a whole-cluster "
                           "communicator #########");
            std::exit(1);
        }
        // One involvement flag per logical dimension actually handed to
        // Sys::generate_collective, which indexes dimensions_involved with
        // dim_mapper[dim] over topology->get_num_of_dimensions(). The old
        // hard-coded 10 went out of bounds for topologies with more than 10
        // dimensions.
        std::vector<bool> dimensions_involved(
            logical_topology->get_num_of_dimensions(), true);
        bool should_be_removed = false;
        CollectivePlan* plan =
            new CollectivePlan(logical_topology, collective_implementation,
                               dimensions_involved, should_be_removed, false);
        comm_plans[{comm_type, plan_key}] = plan;
        return plan;
    } else {
        std::vector<CollectiveImpl*> collective_implementation = looked_up_impls;
        if (collective_implementation.empty()) {
            LoggerFactory::get_logger("system::CommunicatorGroup")
                ->critical("######### Exiting because no collective "
                           "implementation is registered for a sub-cluster "
                           "communicator #########");
            std::exit(1);
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
            if (collective_implementation[0]->type ==
                CollectiveImplType::DoubleBinaryTree) {
                // The plan below is hard-wired to a one-dimensional
                // RingTopology, and Sys::generate_collective_phase hands that
                // topology to the collective algorithm unchecked. A
                // double-binary-tree algorithm would reinterpret the ring as
                // a BinaryTree (type-confused memory access), so fail closed
                // here instead; only a whole-cluster or shaped communicator
                // can serve double-binary-tree.
                LoggerFactory::get_logger("system::CommunicatorGroup")
                    ->critical("######### Exiting because sub-cluster "
                               "communicators currently support only "
                               "ring-compatible native collective "
                               "implementations; double-binary-tree requires "
                               "a whole-cluster communicator #########");
                std::exit(1);
            }
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
        CollectivePlan* plan =
            new CollectivePlan(logical_topology, collective_implementation,
                               dimensions_involved, should_be_removed, true);
        comm_plans[{comm_type, plan_key}] = plan;
        return plan;
    }
    assert(false);
    return nullptr;
}

int CommunicatorGroup::get_position_in_group() {
    return pos_in_group;
}
