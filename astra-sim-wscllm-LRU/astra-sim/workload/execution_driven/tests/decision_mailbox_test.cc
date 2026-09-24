/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

decision_mailbox_test.cc -- phase-1 step 1-6 DecisionMailbox fixture.

Exercises the decision-event aggregation machinery (方案 §4 步骤 1-6) as a
standalone test -- no network simulation, no baseline artifacts touched:

  Part A  Order + drain: events drain in insertion order; drain clears the
          pending set (new dedup epoch); has_decision_work toggles; seqs are
          consecutive; StateDelta assembly (build_state_delta).
  Part B  Same-epoch dedup: same identity (reason/request_id/stage/
          generation) pushes in one epoch collapse to one event; distinct
          identities stay; the same identity may legitimately return in a
          later epoch.
  Part C  Counters: event_count / delivery_count / coalescing_ratio /
          tick_end_without_decision_count / no_decision_python_callback_count
          / finalize_pending has_decision_work.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
(build with cmake --build build/astra_analytical/build_congestion_aware -j).
Run (from template/astra-sim-wscllm-LRU):
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AstraSim::ExecutionDriven;

namespace {

bool g_ok = true;

void expect(bool cond, const char* what) {
    if (!cond) {
        std::fprintf(stderr, "[decision_mailbox_test] FAIL: %s\n", what);
        g_ok = false;
    }
}

DecisionEvent arrival(const std::string& request_id) {
    DecisionEvent ev;
    ev.reason = DecisionReason::ARRIVAL;
    ev.request_id = request_id;
    return ev;
}

DecisionEvent stage_done(const std::string& request_id,
                         const std::string& stage, uint64_t generation) {
    DecisionEvent ev;
    ev.reason = DecisionReason::DECODE_COMPLETION;
    ev.request_id = request_id;
    ev.stage = stage;
    ev.generation = generation;
    ev.payload.watch_member_count = 42;
    return ev;
}

// ---------------------------------------------------------------- Part A --
// Order, drain, seq, StateDelta assembly.
void test_order_drain_and_delta() {
    DecisionMailbox mailbox;
    expect(!mailbox.has_decision_work(), "A: no work initially");

    mailbox.push(arrival("req-1"));
    mailbox.push(stage_done("req-1", "decode", 1));
    expect(mailbox.has_decision_work(), "A: work after pushes");

    const auto events = mailbox.drain();
    expect(events.size() == 2, "A: two events delivered");
    expect(events[0].seq == 1 && events[0].reason == DecisionReason::ARRIVAL &&
               events[0].request_id == "req-1",
           "A: first event seq 1, ARRIVAL, insertion order");
    expect(events[1].seq == 2 && events[1].stage == "decode" &&
               events[1].generation == 1 &&
               events[1].payload.watch_member_count == 42,
           "A: second event seq 2, decode completion, payload kept");
    expect(!mailbox.has_decision_work(), "A: drain clears the pending set");

    // build_state_delta carries tick + delivery_sequence + the events.
    StateDelta delta = build_state_delta(events, /*tick=*/1234,
                                         /*delivery_sequence=*/7);
    expect(delta.tick == 1234 && delta.delivery_sequence == 7 &&
               delta.events.size() == 2,
           "A: StateDelta carries epoch fields");
    const auto second = build_state_delta({}, 99, 8);
    expect(second.events.empty() && second.delivery_sequence == 8,
           "A: empty delta is valid (no-work epochs are separate)");

    // The same identity is accepted again in the next epoch (dedup scope is
    // one epoch).
    mailbox.push(arrival("req-1"));
    expect(mailbox.drain().size() == 1, "A: identity returns in later epoch");
}

// ---------------------------------------------------------------- Part B --
// Same-epoch dedup.
void test_dedup() {
    DecisionMailbox mailbox;
    mailbox.push(arrival("req-1"));
    mailbox.push(arrival("req-1"));  // duplicate identity in the same epoch
    mailbox.push(stage_done("req-1", "decode", 1));
    mailbox.push(stage_done("req-1", "decode", 1));  // duplicate
    mailbox.push(stage_done("req-1", "decode", 2));  // distinct generation
    mailbox.push(stage_done("req-2", "decode", 1));  // distinct request

    const auto events = mailbox.drain();
    expect(events.size() == 4, "B: duplicates collapse, distinct stay");
    expect(events[0].request_id == "req-1" &&
               events[0].reason == DecisionReason::ARRIVAL,
           "B: arrival kept once");
    size_t decode_1_count = 0;
    size_t decode_2_count = 0;
    size_t req2_count = 0;
    for (const auto& ev : events) {
        if (ev.request_id == "req-2") {
            ++req2_count;
        }
        if (ev.request_id == "req-1" && ev.stage == "decode") {
            if (ev.generation == 1) {
                ++decode_1_count;
            } else if (ev.generation == 2) {
                ++decode_2_count;
            }
        }
    }
    expect(decode_1_count == 1, "B: same identity+generation kept once");
    expect(decode_2_count == 1, "B: different generation is distinct");
    expect(req2_count == 1, "B: different request is distinct");

    // Dedup does not disturb insertion order of the survivors.
    expect(events[0].request_id == "req-1" &&
               events[0].reason == DecisionReason::ARRIVAL,
           "B: order preserved after dedup");
}

// ---------------------------------------------------------------- Part C --
// Counters and finalize flag.
void test_counters() {
    DecisionMailbox mailbox;
    expect(mailbox.event_count() == 0 && mailbox.delivery_count() == 0 &&
               mailbox.coalescing_ratio() == 0.0,
           "C: counters start at zero");

    mailbox.push(arrival("req-1"));
    mailbox.push(arrival("req-1"));  // dedup: not counted
    mailbox.push(stage_done("req-1", "decode", 1));
    expect(mailbox.event_count() == 2, "C: event_count counts accepted only");

    mailbox.drain();
    expect(mailbox.delivery_count() == 1, "C: delivery_count counts epochs");
    expect(mailbox.coalescing_ratio() == 2.0,
           "C: 2 events batched into 1 delivery epoch -> ratio 2.0");

    // A drain with no events (finalize-only epoch) is not a delivery.
    mailbox.set_finalize_pending(true);
    expect(mailbox.drain().empty(), "C: finalize-only drain has no events");
    expect(mailbox.delivery_count() == 1,
           "C: empty drain is not a delivery epoch");

    // Gate counters: no-work ticks increment; the Python-error counter is
    // only incremented by a buggy entry (phase-1 acceptance: must stay 0).
    mailbox.count_tick_end_without_decision();
    mailbox.count_tick_end_without_decision();
    mailbox.count_no_decision_python_callback();
    expect(mailbox.tick_end_without_decision_count() == 2,
           "C: no-work ticks counted");
    expect(mailbox.no_decision_python_callback_count() == 1,
           "C: python-error counter incremented explicitly");

    // finalize_pending makes has_decision_work() true with no events; the
    // drain consumes it (a finalize-only epoch delivers an empty delta).
    expect(!mailbox.has_decision_work(), "C: no work with empty mailbox");
    mailbox.set_finalize_pending(true);
    expect(mailbox.has_decision_work(), "C: finalize_pending is decision work");
    expect(mailbox.drain().empty(), "C: finalize-only drain has no events");
    expect(!mailbox.has_decision_work(), "C: drain consumed the finalize");
}

// ---------------------------------------------------------------- Part D --
// Completed-node accounting: compact production mode versus exact audit mode.
void test_completed_fact_accumulator() {
    const int success = static_cast<int>(NodeTerminalStatus::Success);
    const int skipped = static_cast<int>(NodeTerminalStatus::Skipped);

    CompletedFactAccumulator compact;
    // Make the fixture independent of an inherited audit-mode environment.
    compact.set_exact_mode(false);
    compact.record(/*rank=*/3, /*node_id=*/101, "req-compact", "prefill",
                   /*generation=*/4, /*tick=*/500, success);
    compact.record(/*rank=*/3, /*node_id=*/102, "req-compact", "prefill",
                   /*generation=*/4, /*tick=*/501, skipped);
    compact.record(/*rank=*/3, /*node_id=*/103, "req-compact", "prefill",
                   /*generation=*/4, /*tick=*/502, /*invalid=*/99);
    expect(compact.size() == 0 && compact.drain().empty(),
           "D: production accumulator drains completed_nodes as []");
    const CompletedFactCounters& compact_counts = compact.counters();
    expect(compact_counts.total == 3 && compact_counts.success == 1 &&
               compact_counts.skipped == 1 && compact_counts.other == 1,
           "D: compact mode keeps O(1) success/skipped/invalid counters");

    bool compact_mode_change_rejected = false;
    try {
        compact.set_exact_mode(true);
    } catch (const std::logic_error&) {
        compact_mode_change_rejected = true;
    }
    expect(compact_mode_change_rejected,
           "D: mode cannot change after the first terminal");

    CompletedFactAccumulator exact;
    exact.set_exact_mode(true);
    exact.record(/*rank=*/7, /*node_id=*/7001, "req-exact", "decode",
                 /*generation=*/8, /*tick=*/9001, success);
    const std::vector<CompletedNodeFact> first_exact = exact.drain();
    expect(first_exact.size() == 1 && first_exact[0].rank == 7 &&
               first_exact[0].node_id == 7001 &&
               first_exact[0].request_id == "req-exact" &&
               first_exact[0].stage == "decode" &&
               first_exact[0].generation == 8 && first_exact[0].tick == 9001 &&
               first_exact[0].terminal_status == success,
           "D: exact mode preserves the frozen legacy terminal fields");
    expect(exact.size() == 0 && exact.counters().total == 1 &&
               exact.counters().success == 1,
           "D: drain clears only exact records, not run-lifetime counters");

    exact.record(/*rank=*/8, /*node_id=*/7002, "req-exact", "decode",
                 /*generation=*/9, /*tick=*/9002, skipped);
    const std::vector<CompletedNodeFact> second_exact = exact.drain();
    expect(second_exact.size() == 1 && second_exact[0].node_id == 7002 &&
               second_exact[0].terminal_status == skipped &&
               exact.counters().total == 2 && exact.counters().success == 1 &&
               exact.counters().skipped == 1 && exact.counters().other == 0,
           "D: later exact drains retain cumulative terminal accounting");
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
    test_order_drain_and_delta();
    test_dedup();
    test_counters();
    test_completed_fact_accumulator();

    if (!g_ok) {
        std::fprintf(stderr,
                     "[decision_mailbox_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[decision_mailbox_test] ALL PASS: order/drain/StateDelta, "
                "same-epoch dedup, counters, completed-node accounting\n");
    return 0;
}
