/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/UsageTracker.hh"

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
