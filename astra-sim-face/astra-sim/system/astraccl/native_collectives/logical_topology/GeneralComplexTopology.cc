/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <iterator>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
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
    uint64_t last_dim = collective_impl.size() - 1;
    if (collective_impl.size() > dimension_size.size()) {
        Sys::sys_panic(
            "GeneralComplexTopology: collective implementation count (" +
            std::to_string(collective_impl.size()) +
            ") exceeds dimension count (" + std::to_string(dimension_size.size()) +
            ")");
    }
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
                    id, dimension_size[dim], id % offset, offset);
                dimension_topology.push_back(DBT);
            } else {
                DoubleBinaryTreeTopology* DBT = new DoubleBinaryTreeTopology(
                    id, dimension_size[dim],
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
        Sys::sys_panic("GeneralComplexTopology: rank_map is empty (size 0)");
    }
    if (collective_impl.size() != dimension_size.size()) {
        Sys::sys_panic(
            "GeneralComplexTopology: collective implementation count (" +
            std::to_string(collective_impl.size()) +
            ") does not match dimension count (" +
            std::to_string(dimension_size.size()) + ")");
    }

    int expected_ranks = 1;
    for (int size : dimension_size) {
        if (size <= 0) {
            Sys::sys_panic("GeneralComplexTopology: non-positive dimension "
                           "size " +
                           std::to_string(size));
        }
        expected_ranks *= size;
    }
    if (static_cast<int>(rank_map.size()) != expected_ranks) {
        Sys::sys_panic(
            "GeneralComplexTopology: rank_map size (" +
            std::to_string(rank_map.size()) +
            ") does not match expected rank count (" +
            std::to_string(expected_ranks) + ")");
    }

    auto rank_position = std::find(rank_map.begin(), rank_map.end(), id);
    if (rank_position == rank_map.end()) {
        Sys::sys_panic("GeneralComplexTopology: id " + std::to_string(id) +
                       " not found in rank_map of size " +
                       std::to_string(rank_map.size()));
    }
    int compact_id = std::distance(rank_map.begin(), rank_position);
    int offset = 1;

    for (uint64_t dim = 0; dim < dimension_size.size(); dim++) {
        if (collective_impl[dim]->type != CollectiveImplType::Ring) {
            Sys::sys_panic(
                "GeneralComplexTopology: unsupported collective implementation "
                "type " +
                std::to_string(
                    static_cast<int>(collective_impl[dim]->type)) +
                " in rank-map constructor (only Ring is supported)");
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
    }
    assert(static_cast<uint64_t>(dimension) < dimension_topology.size());
    return dimension_topology[dimension]->get_num_of_nodes_in_dimension(0);
}

BasicLogicalTopology* GeneralComplexTopology::get_basic_topology_at_dimension(
    int dimension, ComType type) {
    return dimension_topology[dimension]->get_basic_topology_at_dimension(0,
                                                                          type);
}
