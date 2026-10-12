/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <cstdint>
#include <cstdio>
#include <stdexcept>
#include <vector>

#include "astra-sim/system/scheduling/OfflineGreedy.hh"

using namespace AstraSim;

namespace {

bool g_ok = true;

// Runtime check, independent of NDEBUG: the default CMAKE_BUILD_TYPE is
// Release (-DNDEBUG), which would compile bare assert() into nothing and let
// the whole fixture pass vacuously (review 2026-09-26 infra-1).
void expect(const bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr,
                     "[OfflineGreedyScheduleJournalTest] FAIL: %s\n", what);
        g_ok = false;
    }
}

}  // namespace

int main() {
    const OfflineGreedyScheduleKey group_a{17, 42};
    const OfflineGreedyScheduleKey group_b{18, 42};
    const std::vector<int> schedule_a{0, 2, 1};
    const std::vector<int> schedule_b{1, 0};

    // Equal communicator-local stream ids occupy independent namespaces.
    OfflineGreedyScheduleJournal::publish(group_a, schedule_a, 64, 2);
    OfflineGreedyScheduleJournal::publish(group_b, schedule_b, 32, 3);

    std::vector<int> observed_schedule;
    uint64_t observed_chunk_size = 0;
    expect(OfflineGreedyScheduleJournal::consume(
               group_a, 2, observed_schedule, observed_chunk_size),
           "group_a consumes with the full consumer count");
    expect(observed_schedule == schedule_a, "group_a schedule matches");
    expect(observed_chunk_size == 64, "group_a chunk size matches");
    expect(!OfflineGreedyScheduleJournal::consume(
               group_a, 2, observed_schedule, observed_chunk_size),
           "group_a cannot consume the same entry twice");
    // Producer + one peer is the complete two-rank communicator: the group_a
    // entry was erased once fully consumed, which is observable as a fresh
    // publish of the same key succeeding instead of being rejected.
    bool republished_group_a = true;
    try {
        OfflineGreedyScheduleJournal::publish(group_a, schedule_a, 64, 2);
    } catch (const std::runtime_error&) {
        republished_group_a = false;
    }
    expect(republished_group_a,
           "fully consumed group_a entry is erased (republish succeeds)");

    expect(OfflineGreedyScheduleJournal::consume(
               group_b, 3, observed_schedule, observed_chunk_size),
           "group_b first consumption succeeds");
    expect(observed_schedule == schedule_b, "group_b schedule matches");
    expect(observed_chunk_size == 32, "group_b chunk size matches");
    // The group_b entry is still live with one consumer left: republishing it
    // must be rejected as an attempt to replace a live schedule.
    bool group_b_still_live = false;
    try {
        OfflineGreedyScheduleJournal::publish(group_b, schedule_b, 32, 3);
    } catch (const std::runtime_error&) {
        group_b_still_live = true;
    }
    expect(group_b_still_live,
           "group_b entry still live after one of three consumptions");
    expect(OfflineGreedyScheduleJournal::consume(
               group_b, 3, observed_schedule, observed_chunk_size),
           "group_b second consumption succeeds");

    // A one-rank group has already consumed its result at publication time:
    // nothing is stored, so a consume attempt finds no entry.
    OfflineGreedyScheduleJournal::publish({19, 7}, {3}, 16, 1);
    expect(!OfflineGreedyScheduleJournal::consume(
               {19, 7}, 1, observed_schedule, observed_chunk_size),
           "one-rank group publishes fully consumed");

    // A live key must never silently change communicator cardinality.
    OfflineGreedyScheduleJournal::publish({20, 9}, {0}, 8, 3);
    bool rejected_mismatch = false;
    try {
        OfflineGreedyScheduleJournal::consume(
            {20, 9}, 2, observed_schedule, observed_chunk_size);
    } catch (const std::runtime_error&) {
        rejected_mismatch = true;
    }
    expect(rejected_mismatch,
           "consumer-count mismatch is rejected (runtime_error)");
    // The rejected consumption left the entry untouched and pending: a
    // consume with the correct consumer count still succeeds.
    expect(OfflineGreedyScheduleJournal::consume(
               {20, 9}, 3, observed_schedule, observed_chunk_size),
           "rejected consumption leaves the entry pending");
    if (!g_ok) {
        std::fprintf(stderr,
                     "[OfflineGreedyScheduleJournalTest] FAIL: see messages "
                     "above\n");
        return 1;
    }
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
