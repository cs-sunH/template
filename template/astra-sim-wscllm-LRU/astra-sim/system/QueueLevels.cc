/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/QueueLevels.hh"

#include <cassert>
#include <stdexcept>

using namespace AstraSim;

QueueLevels::QueueLevels(std::vector<int> lv, int offset) {
    int start = offset;
    for (auto& i : lv) {
        // The level index itself is implicit: levels[i] IS level i (the
        // former QueueLevelHandler ctor level parameter was write-only).
        QueueLevelHandler tmp(start, start + i - 1);
        levels.push_back(tmp);
        start += i;
    }
}

// Level bounds are checked before indexing: a negative or past-the-end level
// used to index `levels` unchecked (UB). Assert in debug, fail closed with an
// exception in release (Sys::call_events fails closed on any escaping
// exception, so the run stops with a diagnosis instead of reading garbage).
void QueueLevels::check_level(int level) const {
    assert(level >= 0 &&
           level < static_cast<int>(levels.size()) && "queue level out of range");
    if (level < 0 || level >= static_cast<int>(levels.size())) {
        throw std::out_of_range("QueueLevels: queue level " +
                                std::to_string(level) +
                                " out of range (levels=" +
                                std::to_string(levels.size()) + ")");
    }
}

std::pair<int, RingTopology::Direction> QueueLevels::get_next_queue_at_level(
    int level) {
    check_level(level);
    return levels[level].get_next_queue_id();
}

std::pair<int, RingTopology::Direction> QueueLevels::
    get_next_queue_at_level_first(int level) {
    check_level(level);
    return levels[level].get_next_queue_id_first();
}

std::pair<int, RingTopology::Direction> QueueLevels::
    get_next_queue_at_level_last(int level) {
    check_level(level);
    return levels[level].get_next_queue_id_last();
}
