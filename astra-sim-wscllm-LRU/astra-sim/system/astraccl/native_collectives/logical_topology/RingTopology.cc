/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"
#include "astra-sim/common/Logging.hh"

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
    // a broken ring at collective runtime. A ring of fewer than two nodes
    // has no neighbor to hand a packet to (the ring algorithms derive their
    // stream counts from nodes_in_ring - 1), so it is rejected as well.
    // DELIBERATE TIGHTENING vs the upstream/mother repos, which accept a
    // size-1 degenerate ring: this repository intentionally keeps the >= 2
    // requirement (a single-rank sub-cluster communicator is unreachable in
    // this repo's context -- Sys::generate_collective skips 1-node
    // dimensions -- and this repo's ring algorithms have no self-loop
    // get_receiver/get_sender semantics). Registered here as an intentional
    // fork; do not "re-align" it back to the mother's non-empty-only check.
    if (NPUs.size() < 2) {
        throw std::invalid_argument(
            "RingTopology requires at least two NPUs in the ring");
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
        ->info("custom ring, id: {}, dimension: {} total nodes in ring: {} "
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

// The constructors already throw on malformed input, so the accessors fail
// the same way instead of relying on asserts (compiled out in release) that
// would let an unknown id silently default-insert a wrong mapping.
int RingTopology::checked_index(int node_id) const {
    auto it = id_to_index.find(node_id);
    if (it == id_to_index.end()) {
        throw std::invalid_argument(
            "RingTopology: node id " + std::to_string(node_id) +
            " is not part of ring '" + name + "'");
    }
    return it->second;
}

int RingTopology::get_receiver_homogeneous(int node_id,
                                           Direction direction,
                                           int offset) {
    // The only caller (the RingTopology constructor) always walks the ring
    // clockwise, so the former Anticlockwise branch was dead code and was
    // removed (aligned with the wscllm mother repo). The direction
    // parameter is kept for signature compatibility.
    (void)direction;
    int index = checked_index(node_id);
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
        throw std::invalid_argument(
            "RingTopology produced a negative receiver id " +
            std::to_string(receiver));
    }
    id_to_index[receiver] = index;
    index_to_id[index] = receiver;
    return receiver;
}

int RingTopology::get_receiver(int node_id, Direction direction) {
    int index = checked_index(node_id);
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
    int index = checked_index(node_id);
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
