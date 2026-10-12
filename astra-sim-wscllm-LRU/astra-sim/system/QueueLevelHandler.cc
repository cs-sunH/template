/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/QueueLevelHandler.hh"

using namespace AstraSim;

// (The BackendType parameter is gone with the unreachable upstream Garnet
// backend -- the only thing it ever selected was the former
// `backend != Garnet` clause below, which was always true for the only
// constructible value. The level parameter is gone too (2026-10-01): the
// handler never read it back -- level identity is carried by the index into
// QueueLevels::levels.)
QueueLevelHandler::QueueLevelHandler(int start, int end) {
    for (int i = start; i <= end; i++) {
        queues.push_back(i);
    }
    allocator = 0;
    first_allocator = 0;
    // The first/last allocators split the level's queues into two DISJOINT
    // halves for the issue #137 deadlock avoidance (Sys routes the first
    // phase through get_next_queue_id_first and the second phase through
    // get_next_queue_id_last). With an odd queue count the split is
    // floor(N/2) / ceil(N/2) rather than two equal halves -- still a full,
    // disjoint partition, so the deadlock-avoidance premise (the two phases
    // never share a queue) holds; only the "half" naming in the Sys comment
    // is approximate.
    last_allocator = queues.size() / 2;
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
