/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/system/UsageTracker.hh"

using namespace AstraSim;

UsageTracker::UsageTracker(int levels) : levels(levels), current_level(0) {}

void UsageTracker::increase_usage() {
    if (current_level < levels - 1) {
        current_level++;
    }
}

void UsageTracker::decrease_usage() {
    if (current_level > 0) {
        current_level--;
    }
}

void UsageTracker::set_usage(int level) {
    current_level = level;
}
