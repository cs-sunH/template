/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/CommunicatorGroup.hh"

#include <algorithm>
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
    assert(id > 0);
    this->id = id;
    // Each comm group owns the disjoint stream-id block [id*1e6, (id+1)*1e6).
    // Refuse absurdly large ids instead of wrapping around via signed
    // overflow once id*1e6 exceeds INT_MAX.
    if (id > std::numeric_limits<int>::max() / 1000000) {
        throw std::invalid_argument(
            "comm group id too large: its stream-id block would overflow");
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
        // One involvement flag per logical dimension actually handed to
        // Sys::generate_collective, which indexes dimensions_involved with
        // dim_mapper[dim] over topology->get_num_of_dimensions(). The old
        // hard-coded 10 went out of bounds for topologies with more than 10
        // dimensions.
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
            if (collective_implementation[0]->type ==
                CollectiveImplType::DoubleBinaryTree) {
                // The plan below is hard-wired to a one-dimensional
                // RingTopology, and Sys::generate_collective_phase hands that
                // topology to the collective algorithm unchecked. A
                // double-binary-tree algorithm would reinterpret the ring as
                // a BinaryTree (type-confused memory access), so fail closed
                // here instead; only a whole-cluster or shaped communicator
                // can serve double-binary-tree.
                throw std::runtime_error(
                    "Sub-cluster communicators currently support only "
                    "ring-compatible native collective implementations; "
                    "double-binary-tree requires a whole-cluster "
                    "communicator");
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
