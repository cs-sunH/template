/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <vector>

#include "astra-sim/system/scheduling/OfflineGreedy.hh"

using namespace AstraSim;

// The aggregation build uses -O3 -DNDEBUG (both Dockerfile stages compile
// Release), which compiles the bare assert() calls this file used before
// into no-ops: every check silently vanished, main() always returned 0 and
// a broken journal implementation still PASSed ctest. TEST_CHECK keeps
// every check fail-closed under NDEBUG -- same pattern as
// cli_online_test.cc's CLI_CHECK / ingress_idle_fixture.cc's FIXTURE_CHECK:
// one stderr line, then abort.
#define TEST_CHECK(cond)                                                     \
    ((cond) ? (void)0                                                        \
            : (std::fprintf(stderr,                                           \
                            "[offline_greedy_schedule_journal_test] "         \
                            "CHECK FAILED: %s (%s:%d)\n",                    \
                            #cond, __FILE__, __LINE__), std::abort()))

int main() {
    // The journal is a process-local static map; a fresh process starts
    // empty (the former reset()/pending_entry_count() probes were removed
    // as production-dead, so liveness is observed through consume()).

    const OfflineGreedyScheduleKey group_a{17, 42};
    const OfflineGreedyScheduleKey group_b{18, 42};
    const std::vector<int> schedule_a{0, 2, 1};
    const std::vector<int> schedule_b{1, 0};
    std::vector<int> observed_schedule;
    uint64_t observed_chunk_size = 0;

    // Equal communicator-local stream ids occupy independent namespaces:
    // each key resolves to its own schedule and own consumer budget.
    OfflineGreedyScheduleJournal::publish(group_a, schedule_a, 64, 2);
    OfflineGreedyScheduleJournal::publish(group_b, schedule_b, 32, 3);

    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        group_a, 2, observed_schedule, observed_chunk_size));
    TEST_CHECK(observed_schedule == schedule_a);
    TEST_CHECK(observed_chunk_size == 64);
    // Producer + one peer is the complete two-rank communicator: the entry
    // is reclaimed at the last consume, so a repeat consume misses.
    TEST_CHECK(!OfflineGreedyScheduleJournal::consume(
        group_a, 2, observed_schedule, observed_chunk_size));

    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        group_b, 3, observed_schedule, observed_chunk_size));
    TEST_CHECK(observed_schedule == schedule_b);
    TEST_CHECK(observed_chunk_size == 32);
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        group_b, 3, observed_schedule, observed_chunk_size));
    TEST_CHECK(!OfflineGreedyScheduleJournal::consume(
        group_b, 3, observed_schedule, observed_chunk_size));

    // A one-rank group has already consumed its result at publication time.
    OfflineGreedyScheduleJournal::publish({19, 7}, {3}, 16, 1);
    TEST_CHECK(!OfflineGreedyScheduleJournal::consume(
        {19, 7}, 1, observed_schedule, observed_chunk_size));

    // A live key must never silently change communicator cardinality.
    OfflineGreedyScheduleJournal::publish({20, 9}, {0}, 8, 3);
    bool rejected_mismatch = false;
    try {
        OfflineGreedyScheduleJournal::consume(
            {20, 9}, 2, observed_schedule, observed_chunk_size);
    } catch (const std::runtime_error&) {
        rejected_mismatch = true;
    }
    TEST_CHECK(rejected_mismatch);
    // The mismatched consume must not have consumed the live entry: the
    // full three-consumer cycle still completes exactly.
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        {20, 9}, 3, observed_schedule, observed_chunk_size));
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        {20, 9}, 3, observed_schedule, observed_chunk_size));
    TEST_CHECK(!OfflineGreedyScheduleJournal::consume(
        {20, 9}, 3, observed_schedule, observed_chunk_size));

    // M-6 divergence tripwire: within one communicator namespace, published
    // chunk ids must never move backwards. A strictly smaller id arriving
    // after a larger one means two ranks counted different streams for the
    // same logical collective -- the rendezvous can never complete, which
    // used to hang the run silently.
    OfflineGreedyScheduleJournal::publish({30, 100}, {0, 1}, 8, 2);
    bool rejected_divergence = false;
    try {
        OfflineGreedyScheduleJournal::publish({30, 99}, {1, 0}, 8, 2);
    } catch (const std::runtime_error&) {
        rejected_divergence = true;
    }
    TEST_CHECK(rejected_divergence);
    // The rejected publish left the journal untouched: the original entry
    // still completes its exact two-consumer cycle.
    const std::vector<int> expected_replay{0, 1};
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        {30, 100}, 2, observed_schedule, observed_chunk_size));
    TEST_CHECK(observed_schedule == expected_replay);
    TEST_CHECK(!OfflineGreedyScheduleJournal::consume(
        {30, 100}, 2, observed_schedule, observed_chunk_size));
    // The namespace high-water mark survives the rejected publish: a repeat
    // of the smaller id is still rejected, the same id republishes legally
    // once its entry drained, and a larger id publishes normally.
    rejected_divergence = false;
    try {
        OfflineGreedyScheduleJournal::publish({30, 99}, {1}, 8, 2);
    } catch (const std::runtime_error&) {
        rejected_divergence = true;
    }
    TEST_CHECK(rejected_divergence);
    OfflineGreedyScheduleJournal::publish({30, 100}, {2}, 8, 2);
    // producer + one peer completes the two-consumer cycle exactly
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        {30, 100}, 2, observed_schedule, observed_chunk_size));
    TEST_CHECK(!OfflineGreedyScheduleJournal::consume(
        {30, 100}, 2, observed_schedule, observed_chunk_size));
    OfflineGreedyScheduleJournal::publish({30, 101}, {3}, 8, 2);
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        {30, 101}, 2, observed_schedule, observed_chunk_size));
    // Divergence stays namespace-local: a smaller id in a different
    // namespace is unaffected by namespace 30's high-water mark.
    OfflineGreedyScheduleJournal::publish({31, 5}, {0}, 8, 2);
    TEST_CHECK(OfflineGreedyScheduleJournal::consume(
        {31, 5}, 2, observed_schedule, observed_chunk_size));
    return 0;
}
