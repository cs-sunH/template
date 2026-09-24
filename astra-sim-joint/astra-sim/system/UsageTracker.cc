/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/UsageTracker.hh"

#include <stdexcept>

#include "astra-sim/system/Sys.hh"

using namespace AstraSim;

UsageTracker::UsageTracker(int levels, bool retain_history)
    : levels(levels),
      current_level(0),
      last_tick(0),
      retain_history_(retain_history) {}

void UsageTracker::increase_usage() {
    if (current_level < levels - 1) {
        if (retain_history_) {
            Usage u(current_level, last_tick, Sys::boostedTick());
            usage.push_back(u);
        }
        current_level++;
        if (retain_history_) {
            last_tick = Sys::boostedTick();
        }
    }
}

void UsageTracker::decrease_usage() {
    if (current_level > 0) {
        if (retain_history_) {
            Usage u(current_level, last_tick, Sys::boostedTick());
            usage.push_back(u);
        }
        current_level--;
        if (retain_history_) {
            last_tick = Sys::boostedTick();
        }
    }
}

void UsageTracker::set_usage(int level) {
    if (current_level != level) {
        if (retain_history_) {
            Usage u(current_level, last_tick, Sys::boostedTick());
            usage.push_back(u);
        }
        current_level = level;
        if (retain_history_) {
            last_tick = Sys::boostedTick();
        }
    }
}

std::list<std::pair<uint64_t, double>> UsageTracker::report_percentage(
    uint64_t cycles) {
    if (!retain_history_) {
        throw std::logic_error(
            "UsageTracker history was disabled for this online scheduler");
    }
    decrease_usage();
    increase_usage();
    Tick total_activity_possible = (this->levels - 1) * cycles;
    std::list<Usage>::iterator usage_pointer = this->usage.begin();
    Tick current_activity = 0;
    Tick period_start = 0;
    Tick period_end = cycles;
    std::list<std::pair<uint64_t, double>> result;
    while (usage_pointer != this->usage.end()) {
        Usage current_usage = *usage_pointer;
        uint64_t begin =
            std::max(static_cast<uint64_t>(period_start), current_usage.start);
        uint64_t end =
            std::min(static_cast<uint64_t>(period_end), current_usage.end);
        assert(begin <= end);
        current_activity += ((end - begin) * current_usage.level);
        if (current_usage.end >= period_end) {
            result.push_back(std::make_pair(
                (uint64_t)period_end,
                (((double)current_activity) / total_activity_possible) * 100));
            period_start += cycles;
            period_end += cycles;
            current_activity = 0;
        } else {
            std::advance(usage_pointer, 1);
        }
    }
    return result;
}
