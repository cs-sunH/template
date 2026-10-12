/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/QueueLevelHandler.hh"

using namespace AstraSim;

// (2026-09-28 C1: the BackendType parameter is gone with the unreachable
// upstream Garnet backend -- the only thing it ever selected was the former
// `backend != Garnet` clause below, which was always true for the only
// constructible value.)
QueueLevelHandler::QueueLevelHandler(int level, int start, int end) {
    for (int i = start; i <= end; i++) {
        queues.push_back(i);
    }
    allocator = 0;
    first_allocator = 0;
    last_allocator = queues.size() / 2;
    this->level = level;
}

std::pair<int, RingTopology::Direction> QueueLevelHandler::get_next_queue_id() {
    RingTopology::Direction dir;
    // (The former `backend != Garnet || level > 0` disjunct was always true
    // -- the Garnet backend is unreachable in this repository.)
    if (queues.size() > 1 && allocator >= (queues.size() / 2)) {
        dir = RingTopology::Direction::Anticlockwise;
    } else {
        dir = RingTopology::Direction::Clockwise;
    }
    if (queues.size() == 0) {
        return std::make_pair(-1, dir);
    }
    int tmp = queues[allocator++];
    if (allocator >= queues.size()) {
        allocator = 0;
    }
    return std::make_pair(tmp, dir);
}

std::pair<int, RingTopology::Direction> QueueLevelHandler::
    get_next_queue_id_first() {
    RingTopology::Direction dir;
    dir = RingTopology::Direction::Clockwise;
    if (queues.size() == 0) {
        return std::make_pair(-1, dir);
    }
    int tmp = queues[first_allocator++];
    if (first_allocator >= queues.size() / 2) {
        first_allocator = 0;
    }
    return std::make_pair(tmp, dir);
}

std::pair<int, RingTopology::Direction> QueueLevelHandler::
    get_next_queue_id_last() {
    RingTopology::Direction dir;
    dir = RingTopology::Direction::Anticlockwise;
    if (queues.size() == 0) {
        return std::make_pair(-1, dir);
    }
    int tmp = queues[last_allocator++];
    if (last_allocator >= queues.size()) {
        last_allocator = queues.size() / 2;
    }
    return std::make_pair(tmp, dir);
}
