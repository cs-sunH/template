/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/GeneralComplexTopology.hh"

#include <algorithm>
#include <cassert>
#include <iostream>
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
    if (collective_impl.empty()) {
        throw std::runtime_error(
            "GeneralComplexTopology requires at least one collective "
            "implementation");
    }
    if (collective_impl.size() > dimension_size.size()) {
        throw std::runtime_error(
            "GeneralComplexTopology requires at most one collective "
            "implementation per topology dimension");
    }
    uint64_t last_dim = collective_impl.size() - 1;
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
    assert(!rank_map.empty());
    assert(collective_impl.size() == dimension_size.size());

    int expected_ranks = 1;
    for (int size : dimension_size) {
        assert(size > 0);
        expected_ranks *= size;
    }
    assert(static_cast<int>(rank_map.size()) == expected_ranks);

    auto rank_position = std::find(rank_map.begin(), rank_map.end(), id);
    assert(rank_position != rank_map.end());
    int compact_id = std::distance(rank_map.begin(), rank_position);
    int offset = 1;

    for (uint64_t dim = 0; dim < dimension_size.size(); dim++) {
        assert(collective_impl[dim]->type == CollectiveImplType::Ring);

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
