/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"
#include "astra-sim/common/Logging.hh"

#include <cassert>
#include <cstdlib>
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace AstraSim;

RingTopology::RingTopology(Dimension dimension, int id, std::vector<int> NPUs) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    // Fail closed instead of asserting: the default release build compiles
    // the assert out, and a wrong rank list would otherwise surface only as
    // a broken ring at collective runtime.
    if (NPUs.empty()) {
        throw std::invalid_argument(
            "RingTopology requires a non-empty NPU list");
    }
    this->id = id;
    this->total_nodes_in_ring = NPUs.size();
    this->dimension = dimension;
    this->offset = -1;
    this->index_in_ring = -1;
    for (int i = 0; i < total_nodes_in_ring; i++) {
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_ring = i;
        }
    }

    LoggerFactory::get_logger("system::topology::RingTopology")
        ->info("rank-mapped ring, id: {}, dimension: {} total nodes in ring: {} "
               "index in ring: {}",
               id, name, total_nodes_in_ring, index_in_ring);

    if (index_in_ring < 0) {
        throw std::invalid_argument(
            "RingTopology requires the local rank to be present in the NPU "
            "list");
    }
}
RingTopology::RingTopology(Dimension dimension,
                           int id,
                           int total_nodes_in_ring,
                           int index_in_ring,
                           int offset) {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (total_nodes_in_ring <= 0 || offset <= 0) {
        throw std::invalid_argument(
            "RingTopology requires a positive ring size and offset");
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->info("ring of node 0, id: {} dimension: {} total nodes in ring: "
                   "{} index in ring: {} offset: {}",
                   id, name, total_nodes_in_ring, index_in_ring, offset);
    }
    this->id = id;
    this->total_nodes_in_ring = total_nodes_in_ring;
    this->index_in_ring = index_in_ring;
    this->dimension = dimension;
    this->offset = offset;

    id_to_index[id] = index_in_ring;
    index_to_id[index_in_ring] = id;
    int tmp = id;
    for (int i = 0; i < total_nodes_in_ring - 1; i++) {
        tmp = get_receiver_homogeneous(tmp, RingTopology::Direction::Clockwise,
                                       offset);
    }
}

int RingTopology::get_receiver_homogeneous(int node_id,
                                           Direction direction,
                                           int offset) {
    // The only caller (the RingTopology constructor) always walks the ring
    // clockwise, so the former Anticlockwise branch was dead code and was
    // removed. The direction parameter is kept for signature compatibility.
    (void)direction;
    const auto node_it = id_to_index.find(node_id);
    if (node_it == id_to_index.end()) {
        // Fail closed (2026-09-27): the old assert vanished under NDEBUG and
        // operator[] then silently inserted index 0 for the missing key,
        // corrupting the ring. Log and terminate instead.
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->critical("at dim: {} at id: {}: node id {} is not in the ring "
                       "id_to_index map",
                       name, id, node_id);
        std::exit(EXIT_FAILURE);
    }
    int index = node_it->second;
    int receiver = node_id + offset;
    if (index == total_nodes_in_ring - 1) {
        receiver -= (total_nodes_in_ring * offset);
        index = 0;
    } else {
        index++;
    }
    if (receiver < 0) {
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                       "id: {}, offset: {}, index_in_ring {} receiver {}",
                       name, id, name, index, node_id, offset,
                       index_in_ring, receiver);
        std::exit(EXIT_FAILURE);
    }
    assert(receiver >= 0);
    id_to_index[receiver] = index;
    index_to_id[index] = receiver;
    return receiver;
}

int RingTopology::get_receiver(int node_id, Direction direction) {
    const auto node_it = id_to_index.find(node_id);
    if (node_it == id_to_index.end()) {
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->critical("at dim: {} at id: {}: node id {} is not in the ring "
                       "id_to_index map",
                       name, id, node_id);
        std::exit(EXIT_FAILURE);
    }
    int index = node_it->second;
    if (direction == RingTopology::Direction::Clockwise) {
        index++;
        if (index == total_nodes_in_ring) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_ring - 1;
        }
        return index_to_id[index];
    }
}

int RingTopology::get_sender(int node_id, Direction direction) {
    const auto node_it = id_to_index.find(node_id);
    if (node_it == id_to_index.end()) {
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->critical("at dim: {} at id: {}: node id {} is not in the ring "
                       "id_to_index map",
                       name, id, node_id);
        std::exit(EXIT_FAILURE);
    }
    int index = node_it->second;
    if (direction == RingTopology::Direction::Anticlockwise) {
        index++;
        if (index == total_nodes_in_ring) {
            index = 0;
        }
        return index_to_id[index];
    } else {
        index--;
        if (index < 0) {
            index = total_nodes_in_ring - 1;
        }
        return index_to_id[index];
    }
}

int RingTopology::get_index_in_ring() {
    return index_in_ring;
}

RingTopology::Dimension RingTopology::get_dimension() {
    return dimension;
}

int RingTopology::get_num_of_nodes_in_dimension(int dimension) {
    return get_nodes_in_ring();
}

int RingTopology::get_nodes_in_ring() {
    return total_nodes_in_ring;
}
