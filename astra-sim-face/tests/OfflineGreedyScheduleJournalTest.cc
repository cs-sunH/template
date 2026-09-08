/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "astra-sim/system/scheduling/OfflineGreedy.hh"

using namespace AstraSim;

int main() {
    OfflineGreedyScheduleJournal::reset();

    const OfflineGreedyScheduleKey group_a{17, 42};
    const OfflineGreedyScheduleKey group_b{18, 42};
    const std::vector<int> schedule_a{0, 2, 1};
    const std::vector<int> schedule_b{1, 0};

    // Equal communicator-local stream ids occupy independent namespaces.
    OfflineGreedyScheduleJournal::publish(group_a, schedule_a, 64, 2);
    OfflineGreedyScheduleJournal::publish(group_b, schedule_b, 32, 3);
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 2);

    std::vector<int> observed_schedule;
    uint64_t observed_chunk_size = 0;
    assert(OfflineGreedyScheduleJournal::consume(
        group_a, 2, observed_schedule, observed_chunk_size));
    assert(observed_schedule == schedule_a);
    assert(observed_chunk_size == 64);
    // Producer + one peer is the complete two-rank communicator.
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 1);
    assert(!OfflineGreedyScheduleJournal::consume(
        group_a, 2, observed_schedule, observed_chunk_size));

    assert(OfflineGreedyScheduleJournal::consume(
        group_b, 3, observed_schedule, observed_chunk_size));
    assert(observed_schedule == schedule_b);
    assert(observed_chunk_size == 32);
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 1);
    assert(OfflineGreedyScheduleJournal::consume(
        group_b, 3, observed_schedule, observed_chunk_size));
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 0);

    // A one-rank group has already consumed its result at publication time.
    OfflineGreedyScheduleJournal::publish({19, 7}, {3}, 16, 1);
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 0);

    // A live key must never silently change communicator cardinality.
    OfflineGreedyScheduleJournal::publish({20, 9}, {0}, 8, 3);
    bool rejected_mismatch = false;
    try {
        OfflineGreedyScheduleJournal::consume(
            {20, 9}, 2, observed_schedule, observed_chunk_size);
    } catch (const std::runtime_error&) {
        rejected_mismatch = true;
    }
    assert(rejected_mismatch);
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 1);

    OfflineGreedyScheduleJournal::reset();
    assert(OfflineGreedyScheduleJournal::pending_entry_count() == 0);
    return 0;
}
