/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/QueueLevels.hh"

#include "astra-sim/common/Logging.hh"

using namespace AstraSim;

namespace {

// Fail closed on an out-of-range level the same way the other
// system-layer bounds checks do (GeneralComplexTopology/
// get_num_of_nodes_in_dimension): the former bare levels[level] read was
// silent UB -- a communicator whose dimension count exceeds the system's
// queue dimensions reached these accessors with an out-of-range level and
// read the vector out of bounds.
QueueLevelHandler& level_or_die(std::vector<QueueLevelHandler>& levels,
                                int level) {
    if (level < 0 || static_cast<size_t>(level) >= levels.size()) {
        LoggerFactory::get_logger("system::QueueLevels")
            ->critical("level: {} requested! but max level is {}", level,
                       levels.size() - 1);
        std::exit(1);
    }
    return levels[level];
}

}  // namespace

QueueLevels::QueueLevels(std::vector<int> lv, int offset) {
    int start = offset;
    int l = 0;
    for (auto& i : lv) {
        QueueLevelHandler tmp(l++, start, start + i - 1);
        levels.push_back(tmp);
        start += i;
    }
}

std::pair<int, RingTopology::Direction> QueueLevels::get_next_queue_at_level(
    int level) {
    return level_or_die(levels, level).get_next_queue_id();
}

std::pair<int, RingTopology::Direction> QueueLevels::
    get_next_queue_at_level_first(int level) {
    return level_or_die(levels, level).get_next_queue_id_first();
}

std::pair<int, RingTopology::Direction> QueueLevels::
    get_next_queue_at_level_last(int level) {
    return level_or_die(levels, level).get_next_queue_id_last();
}
