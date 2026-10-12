/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/astraccl/native_collectives/logical_topology/RingTopology.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"

#include <cassert>
#include <cstdlib>

using namespace std;
using namespace AstraSim;

RingTopology::RingTopology(Dimension dimension, int id, std::vector<int> NPUs)
    : BasicLogicalTopology() {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    this->id = id;
    this->total_nodes_in_ring = NPUs.size();
    this->dimension = dimension;
    this->index_in_ring = -1;
    for (int i = 0; i < total_nodes_in_ring; i++) {
        // R6-T3/R7-D fix (2026-09-29): a duplicated rank used to silently
        // overwrite the first occurrence in id_to_index (two ring positions
        // collapsing onto one node -> self-ring / hang), and this constructor
        // is the one place a non-member rank surfaces -- the assert below is
        // compiled out of release builds, so the non-member rank proceeded
        // with index_in_ring == -1 and mis-routed every message. Both are
        // always configuration errors: fail loud in every build type.
        if (id_to_index.find(NPUs[i]) != id_to_index.end()) {
            LoggerFactory::get_logger("system::topology::RingTopology")
                ->critical("custom ring, id: {}: rank {} appears more than "
                           "once in the ring membership",
                           id, NPUs[i]);
            Sys::sys_panic("RingTopology: duplicated rank " +
                           std::to_string(NPUs[i]) + " in ring membership");
        }
        id_to_index[NPUs[i]] = i;
        index_to_id[i] = NPUs[i];
        if (id == NPUs[i]) {
            index_in_ring = i;
        }
    }

    LoggerFactory::get_logger("system::topology::RingTopology")
        ->info("custom ring, id: {}, dimension: {} total nodes in ring: {} "
               "index in ring: {} total nodes in ring {}",
               id, name, total_nodes_in_ring, index_in_ring,
               total_nodes_in_ring);

    if (index_in_ring < 0) {
        Sys::sys_panic("RingTopology: issuing rank " + std::to_string(id) +
                       " is not a member of the ring (membership does not "
                       "contain it); a non-member rank cannot route in this "
                       "ring");
    }
    assert(index_in_ring >= 0);
}
RingTopology::RingTopology(Dimension dimension,
                           int id,
                           int total_nodes_in_ring,
                           int index_in_ring,
                           int offset)
    : BasicLogicalTopology() {
    name = "local";
    if (dimension == Dimension::Vertical) {
        name = "vertical";
    } else if (dimension == Dimension::Horizontal) {
        name = "horizontal";
    }
    if (id == 0) {
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->info("ring of node 0, id: {} dimension: {} total nodes in ring: "
                   "{} index in ring: {} offset: {} total nodes in ring: {}",
                   id, name, total_nodes_in_ring, index_in_ring, offset,
                   total_nodes_in_ring);
    }
    this->id = id;
    this->total_nodes_in_ring = total_nodes_in_ring;
    this->index_in_ring = index_in_ring;
    this->dimension = dimension;

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
    // removed. The direction parameter is kept for signature compatibility
    // but a non-Clockwise value now fails loud (2026-09-30, mirroring the
    // joint repo) instead of being silently treated as Clockwise.
    if (direction != Direction::Clockwise) {
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->critical(
                "get_receiver_homogeneous: non-Clockwise direction is not "
                "supported (node {})", node_id);
        std::exit(EXIT_FAILURE);
    }
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
    int receiver = node_id + offset;
    if (index == total_nodes_in_ring - 1) {
        receiver -= (total_nodes_in_ring * offset);
        index = 0;
    } else {
        index++;
    }
    if (receiver < 0) {
        // (2026-10-01) Release-active fail-closed, matching the 2026-09-30
        // batch's upgrade of the sibling topology guards: the old assert
        // compiled out of Release and the negative receiver was inserted
        // into both index maps. Unreachable via the constructor's in-bounds
        // stepping; this is the belt for future callers.
        LoggerFactory::get_logger("system::topology::RingTopology")
            ->critical("at dim: {} at id: {} dimension: {} index: {}, node "
                       "id: {}, offset: {}, index_in_ring {} receiver {}",
                       name, id, name, index, node_id, offset,
                       index_in_ring, receiver);
        std::exit(EXIT_FAILURE);
    }
    id_to_index[receiver] = index;
    index_to_id[index] = receiver;
    return receiver;
}

int RingTopology::get_receiver(int node_id, Direction direction) {
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
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
    assert(id_to_index.find(node_id) != id_to_index.end());
    int index = id_to_index[node_id];
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
