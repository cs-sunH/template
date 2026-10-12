/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

decision_mailbox_test.cc -- phase-1 step 1-6 DecisionMailbox fixture.

Exercises the decision-event aggregation machinery (方案 §4 步骤 1-6) as a
standalone test -- no network simulation, no baseline artifacts touched:

  Part A  Order + drain: events drain in insertion order; drain clears the
          pending set (new dedup epoch); has_decision_work toggles; seqs are
          consecutive; StateDelta assembly (build_state_delta_v1).
  Part B  Same-epoch dedup: same identity (reason/request_id/stage/
          generation) pushes in one epoch collapse to one event; distinct
          identities stay; the same identity may legitimately return in a
          later epoch.
  Part C  Counters: event_count / delivery_count / coalescing_ratio /
          tick_end_without_decision_count / has_decision_work.
  Part D  Completed-node accounting: compact production mode versus exact
          audit mode.
  Part E  Same-tick epoch order (策略规格 §6, review 2026-09-26 M9-1): a
          completion-class deposit and a real ingress arrival sharing one
          tick drain as [completion, ARRIVAL] through the real
          EventQueue -> arrival_cb -> mailbox -> drain chain.

Build: the CMake target AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
(build with cmake --build build/astra_analytical/build_congestion_aware -j).
Run (from template/astra-sim-face):
  build/astra_analytical/build_congestion_aware/bin/\
      AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest
Exit code 0 on ALL PASS.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "common/EventQueue.h"

#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

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

    // build_state_delta_v1 carries tick + delivery_sequence + the events.
    StateDelta delta = build_state_delta_v1(events, /*tick=*/1234,
                                            /*delivery_sequence=*/7,
                                            /*deferred_from_tick=*/0, {}, {});
    expect(delta.tick == 1234 && delta.delivery_sequence == 7 &&
               delta.events.size() == 2,
           "A: StateDelta carries epoch fields");
    const auto second = build_state_delta_v1({}, 99, 8, 0, {}, {});
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

    // A drain with no pending events is not a delivery epoch.
    // (M1-3/M11-3, 2026-09-26: the former finalize-pending scaffolding --
    // set_finalize_pending and the finalize-only epoch it produced -- is
    // deleted as production-unreachable; the real commit-ack finalize is
    // send_commit_ack -> ack file -> Python on_commit_ack.)
    expect(mailbox.drain().empty(), "C: empty drain has no events");
    expect(mailbox.delivery_count() == 1,
           "C: empty drain is not a delivery epoch");

    // Gate counters: no-work ticks increment. (The former
    // count_no_decision_python_callback increment hook and its always-zero
    // counter were removed -- see DecisionMailbox.hh.)
    mailbox.count_tick_end_without_decision();
    mailbox.count_tick_end_without_decision();
    expect(mailbox.tick_end_without_decision_count() == 2,
           "C: no-work ticks counted");

    expect(!mailbox.has_decision_work(), "C: no work with empty mailbox");
}

// ---------------------------------------------------------------- Part D --
// Completed-node accounting: compact production mode versus exact audit mode.
void test_completed_fact_accumulator() {
    const int success = static_cast<int>(NodeTerminalStatus::Success);
    const int skipped = static_cast<int>(NodeTerminalStatus::Skipped);

    // The exact audit mode is the production env-gated construction channel
    // (ASTRA_SIM_ONLINE_COMPLETED_NODES=exact, read once by the
    // constructor). Scope the env around the exact accumulator's
    // construction so the fixture is deterministic and independent of an
    // inherited environment; the compact accumulator is then constructed
    // with the env unset (the production default).
    ::setenv("ASTRA_SIM_ONLINE_COMPLETED_NODES", "exact", /*overwrite=*/1);
    CompletedFactAccumulator exact;
    ::unsetenv("ASTRA_SIM_ONLINE_COMPLETED_NODES");
    CompletedFactAccumulator compact;
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

// ---------------------------------------------------------------- Part E --
// Same-tick epoch order: completions before arrivals (策略规格 §6: "同一
// 时刻的事件先处理完成类、再处理到达类、最后启动新迭代"; review 2026-09-26
// M9-1 coverage gap -- no fixture guarded this invariant).
//
// The production epoch is formed once per tick at the tick-end gate
// (main_online.cc ed_driver_tick_end) by draining the mailbox in DEPOSIT
// order. Deposit order is the EventQueue's same-tick EventList firing
// order: the completion side deposits through the watch-fire hook, the
// arrival side through RequestIngress::arrival_cb; nothing downstream
// reorders by reason (DecisionMailbox::push preserves insertion order, Part
// A). A same-tick ordering regression anywhere on that chain would
// therefore silently flip the epoch order. This fixture drives the real
// machinery end to end: a completion-class deposit (the watch-hook shape)
// and a real future-alarm arrival share tick 1000, and the drained epoch --
// and the StateDelta built from it -- must list the completion first.
void completion_deposit(void* v) {
    auto* mailbox = static_cast<DecisionMailbox*>(v);
    DecisionEvent ev;
    ev.reason = DecisionReason::REQUEST_COMPLETE;
    ev.request_id = "req-complete";
    mailbox->push(std::move(ev));
}

void test_same_tick_completion_before_arrival() {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    ingress.bind(&eq, &mailbox, &svc);

    // Completion-class deposit armed BEFORE the arrival alarm: this is the
    // shared-tick arm order the contract pins (the completion deposit must
    // precede the arrival deposit within the same epoch).
    eq.schedule_event(1000, completion_deposit, &mailbox);

    // Real arrival path: a commit-leg future arrival targeting the SAME
    // tick (RequestIngress::schedule_future_arrival arms the alarm;
    // arrival_cb fires at 1000 and deposits the ARRIVAL event).
    RequestEnvelope env;
    env.session_id = "s-arrival";
    env.turn_index = 0;
    env.request_id = "req-arrival";
    env.prefill_length = 100;
    env.decode_length = 50;
    env.arrival_world_ns = 1000;
    ingress.schedule_future_arrival(env);

    while (!eq.finished()) {
        eq.proceed();
    }
    expect(eq.get_current_time() == 1000,
           "E: both events fired in the shared tick 1000");

    expect(mailbox.has_decision_work(), "E: the shared epoch has work");
    const auto events = mailbox.drain();
    expect(events.size() == 2,
           "E: completion and arrival co-occur in one epoch");
    expect(events[0].reason == DecisionReason::REQUEST_COMPLETE &&
               events[0].request_id == "req-complete",
           "E: completion-class deposits first in the same-tick epoch");
    expect(events[1].reason == DecisionReason::ARRIVAL &&
               events[1].request_id == "req-arrival",
           "E: arrival-class deposits second in the same-tick epoch");

    const StateDelta delta =
        build_state_delta_v1(events, /*tick=*/1000, /*delivery_sequence=*/1,
                             /*deferred_from_tick=*/0, {}, {});
    expect(delta.tick == 1000 && delta.events.size() == 2 &&
               delta.events[0].reason == DecisionReason::REQUEST_COMPLETE &&
               delta.events[1].reason == DecisionReason::ARRIVAL,
           "E: delta order is completion-before-arrival");

    std::printf("[decision_mailbox_test] PASS: same-tick epoch drains "
                "completion before arrival (spec §6 order)\n");
}

}  // namespace

int main(int /*argc*/, char* /*argv*/[]) {
    test_order_drain_and_delta();
    test_dedup();
    test_counters();
    test_completed_fact_accumulator();
    test_same_tick_completion_before_arrival();

    if (!g_ok) {
        std::fprintf(stderr,
                     "[decision_mailbox_test] FAIL: see messages above\n");
        return 1;
    }
    std::printf("[decision_mailbox_test] ALL PASS: order/drain/StateDelta, "
                "same-epoch dedup, counters, completed-node accounting, "
                "same-tick completion-before-arrival epoch order\n");
    return 0;
}
