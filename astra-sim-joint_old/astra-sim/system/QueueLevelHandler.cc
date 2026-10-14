/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/QueueLevelHandler.hh"

using namespace AstraSim;

QueueLevelHandler::QueueLevelHandler(int level,
                                     int start,
                                     int end,
                                     AstraNetworkAPI::BackendType backend) {
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
    // is approximate. CAVEAT (2026-10-01 audit): a level with exactly ONE
    // queue degenerates the split -- both allocators resolve to that same
    // queue 0, so the "disjoint halves" premise does NOT hold there; the
    // issue #137 argument requires queues_per_dim >= 2 for every level
    // that carries first/last phased collectives (the Sys ctor validates
    // entries >= 1; single-queue dimensions ride one shared queue).
    last_allocator = queues.size() / 2;
    this->level = level;
    this->backend = backend;
}

std::pair<int, RingTopology::Direction> QueueLevelHandler::get_next_queue_id() {
    RingTopology::Direction dir;
    if ((backend != AstraNetworkAPI::BackendType::Garnet || level > 0) &&
        queues.size() > 1 && allocator >= (queues.size() / 2)) {
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
