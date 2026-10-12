/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"

#include <algorithm>
#include <cstdlib>
#include <iterator>
#include <stdexcept>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/astraccl/CollectiveImpl.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/DoubleBinaryTreeTopology.hh"
#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"

using namespace std;
using namespace AstraSim;

GeneralComplexTopology::GeneralComplexTopology(
    int id,
    std::vector<int> dimension_size,
    std::vector<CollectiveImpl*> collective_impl) {
    int offset = 1;
    // An empty implementation list is tolerated here: Sys unconditionally
    // builds one topology per collective type from the BYPASS_ALL_CUSTOM
    // lookup, and a collective type with no native configuration simply gets
    // an empty (dummy) topology. Requesting such a collective fails later
    // with a proper diagnostic inside CollectiveImplLookup::get_collective_impl
    // instead of terminating during Sys construction.
    if (collective_impl.size() > dimension_size.size()) {
        throw std::runtime_error(
            "GeneralComplexTopology requires at most one collective "
            "implementation per topology dimension");
    }
    // Same positivity contract as the rank-map constructor below: a zero or
    // negative dimension reaches the ring arithmetic as a modulo-by-zero or
    // an empty ring instead of failing here.
    for (int size : dimension_size) {
        if (size <= 0) {
            throw std::runtime_error(
                "GeneralComplexTopology requires every topology dimension "
                "size to be positive");
        }
    }
    // Clamp before the subtraction: on the tolerated empty implementation
    // list, size() - 1 would wrap around (the value is never read -- the
    // loop below does not execute -- but the wrap stays a latent trap).
    const uint64_t last_dim =
        collective_impl.empty() ? 0 : collective_impl.size() - 1;
    for (uint64_t dim = 0; dim < collective_impl.size(); dim++) {
        if (collective_impl[dim]->type == CollectiveImplType::Ring ||
            collective_impl[dim]->type == CollectiveImplType::Direct ||
            collective_impl[dim]->type == CollectiveImplType::HalvingDoubling ||
            // While executing a collective according a Chakra ET representation
            // does not need information on the logical topology, The system
            // layer's logic of defining and invoking "collective phase" objects
            // (which in turn executes the individual collective algorithm
            // implementation) does rely on the existence (not the values) of
            // the logical topology. (Refer to functions involving
            // 'Sys.cc::generate_collective') Therefore, we fill in the logical
            // topology with a dummy, default value.
            collective_impl[dim]->type == CollectiveImplType::CustomCollectiveImpl) {
            RingTopology* ring = new RingTopology(
                RingTopology::Dimension::NA, id, dimension_size[dim],
                (id % (offset * dimension_size[dim])) / offset, offset);
            dimension_topology.push_back(ring);
        } else if (collective_impl[dim]->type == CollectiveImplType::OneRing ||
                   collective_impl[dim]->type ==
                       CollectiveImplType::OneDirect ||
                   collective_impl[dim]->type ==
                       CollectiveImplType::OneHalvingDoubling) {
            int total_npus = 1;
            for (int d : dimension_size) {
                total_npus *= d;
            }
            RingTopology* ring =
                new RingTopology(RingTopology::Dimension::NA, id, total_npus,
                                 id % total_npus, 1);
            dimension_topology.push_back(ring);
            return;
        } else if (collective_impl[dim]->type ==
                   CollectiveImplType::DoubleBinaryTree) {
            if (dim == last_dim) {
                DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
                    dimension_size[dim], id % offset, offset);
                dimension_topology.push_back(DBT);
            } else {
                DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
                    dimension_size[dim],
                    (id - (id % (offset * dimension_size[dim]))) +
                        (id % offset),
                    offset);
                dimension_topology.push_back(DBT);
            }
        }
        offset *= dimension_size[dim];
    }
}

GeneralComplexTopology::GeneralComplexTopology(
    int id,
    std::vector<int> rank_map,
    std::vector<int> dimension_size,
    std::vector<CollectiveImpl*> collective_impl) {
    // rank_map is a compact row-major embedding: dimension 0 changes fastest.
    // Build the X/Y slice containing this physical rank for each dimension.
    if (rank_map.empty()) {
        throw std::runtime_error(
            "GeneralComplexTopology requires a non-empty rank map");
    }
    if (collective_impl.size() != dimension_size.size()) {
        throw std::runtime_error(
            "GeneralComplexTopology requires one collective implementation "
            "per topology dimension when constructing from a rank map");
    }

    int expected_ranks = 1;
    for (int size : dimension_size) {
        if (size <= 0) {
            throw std::runtime_error(
                "GeneralComplexTopology requires every topology dimension "
                "size to be positive");
        }
        expected_ranks *= size;
    }
    if (static_cast<int>(rank_map.size()) != expected_ranks) {
        throw std::runtime_error(
            "GeneralComplexTopology requires the rank map size to equal the "
            "product of the topology dimension sizes");
    }

    auto rank_position = std::find(rank_map.begin(), rank_map.end(), id);
    if (rank_position == rank_map.end()) {
        throw std::runtime_error(
            "GeneralComplexTopology requires the local rank to be present in "
            "the rank map");
    }
    int compact_id = std::distance(rank_map.begin(), rank_position);
    int offset = 1;

    for (uint64_t dim = 0; dim < dimension_size.size(); dim++) {
        if (collective_impl[dim]->type != CollectiveImplType::Ring) {
            throw std::runtime_error(
                "GeneralComplexTopology requires the Ring collective "
                "implementation for every topology dimension when "
                "constructing from a rank map");
        }

        int coordinate = (compact_id / offset) % dimension_size[dim];
        int slice_base = compact_id - coordinate * offset;
        std::vector<int> dimension_ranks;
        dimension_ranks.reserve(dimension_size[dim]);
        for (int index = 0; index < dimension_size[dim]; index++) {
            dimension_ranks.push_back(rank_map[slice_base + index * offset]);
        }

        RingTopology::Dimension ring_dimension = RingTopology::Dimension::NA;
        if (dim == 0) {
            ring_dimension = RingTopology::Dimension::Horizontal;
        } else if (dim == 1) {
            ring_dimension = RingTopology::Dimension::Vertical;
        }
        dimension_topology.push_back(
            new RingTopology(ring_dimension, id, dimension_ranks));
        offset *= dimension_size[dim];
    }
}

GeneralComplexTopology::~GeneralComplexTopology() {
    for (uint64_t i = 0; i < dimension_topology.size(); i++) {
        delete dimension_topology[i];
    }
}

int GeneralComplexTopology::get_num_of_dimensions() {
    return dimension_topology.size();
}

int GeneralComplexTopology::get_num_of_nodes_in_dimension(int dimension) {
    if (static_cast<uint64_t>(dimension) >= dimension_topology.size()) {
        LoggerFactory::get_logger("system::topology::GeneralComplexTopology")
            ->critical("dim: {} requested! but max dim is {}", dimension,
                       dimension_topology.size() - 1);
        std::exit(1);
    }
    return dimension_topology[dimension]->get_num_of_nodes_in_dimension(0);
}

BasicLogicalTopology* GeneralComplexTopology::get_basic_topology_at_dimension(
    int dimension, ComType type) {
    // 与上方 get_num_of_nodes_in_dimension 同款 fail-closed 守卫
    // （2026-10-01 对齐）：本访问器此前完全没有守卫，越界维度解引用越过
    // vector 末端（UB），而不是带着维度数字在手地失败。
    if (static_cast<uint64_t>(dimension) >= dimension_topology.size()) {
        LoggerFactory::get_logger("system::topology::GeneralComplexTopology")
            ->critical("dim: {} requested! but max dim is {}", dimension,
                       dimension_topology.size() - 1);
        std::exit(1);
    }
    return dimension_topology[dimension]->get_basic_topology_at_dimension(0,
                                                                          type);
}
