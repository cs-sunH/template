/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ingress_idle_fixture.cc -- phase-1 step 1-2 IDLE lifecycle fixture.

Scenarios (方案 §4 步骤 1-2 操作 6, amended step 1-10):
  1. No requests at startup: the service stays IDLE (no exit, no busy wait);
     after the input is closed it drains straight through
     IDLE -> DRAINING -> FINISHED.
  2. Two requests injected at ticks T1/T2 while the loop is blocked in
     wait_for_work(): IDLE -> ACTIVE, the arrival alarms fire at exactly
     T1/T2 (asserted; no dependency on any rank being idle). Step 1-10: with
     the input still open the fully drained service returns to IDLE
     (五态迁移 IDLE -> ACTIVE -> IDLE), then the input closes ->
     DRAINING -> FINISHED. Fixture requests have no graph nodes, so the
     arrival hook completes them immediately (drain).
  3. Close-at-startup with a preloaded command queue (主控裁决 2026-08-15,
     steps-1-6 偏差①): two Submits AND the Close are queued BEFORE the loop
     starts. Drain-first means the commands are processed (accepted=2,
     completed=2, alarms at exactly 1000/2000) and only then FINISHED --
     nothing is silently dropped by a finished()-before-drain check.
  4. Phase-7 §10.7: EOF terminal command (合同② EOF vs 显式 close 三态
     区分). An EndOfFile command drains the input straight through
     IDLE -> DRAINING -> FINISHED exactly like an explicit close, but the
     lifecycle audit records close_source=EOF (svc.input_close_reason() ==
     InputCloseReason::EndOfFile).
  5. Phase-7 §10.7: overflow audit. A tiny-capacity ingress (2) accepts
     three Submits: the third is rejected (overflow_count == 1) and the
     peak command occupancy is 2 -- the bounded-queue backpressure contract
     with an observable rejection counter.
  6. Phase-7 §10.7: Error terminal command = fail-closed abort (合同②:
     异常/错误 → fail-closed abort, 进程退出非 0, 不做静默降级). In a forked
     child, draining an Error command must abort (SIGABRT); the parent
     asserts the child died by signal (NOT by a clean exit -- the old
     silent-close behavior would have exited 0).
  8. Close-vs-Submit race: close and one producer contend on the same ingress
     mutex. The Submit is either accepted before close and fully drained, or
     rejected after close; every later Submit is rejected and no command is
     left behind at the finished exit.

Expected state-transition log across all scenarios: 2 (scenario 1), 4
(scenario 2), 2 (scenario 3), 2 (scenario 4; the log records the NEW state
of each transition, initial IDLE is not an entry).

Build: registered in the CMake build (M20) as target
  AstraSim_Analytical_Congestion_Aware_IngressIdleTest in
  astra-sim/network_frontend/analytical/CMakeLists.txt (same shared-source +
  execution_driven recipe as the sibling fixtures; the old manual g++ line's
  RequestIngress/DecisionMailbox/ServiceCoordinator and backend event-queue
  sources come from the execution_driven glob and the linked
  Analytical_Congestion_Aware static library, and the -pthread of that line
  is carried by the target's pthread link). Configure per README §2 (the
  build/astra_analytical aggregation with
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON), then:
    cmake --build build/astra_analytical/build_congestion_aware \
          --target AstraSim_Analytical_Congestion_Aware_IngressIdleTest -j
    build/astra_analytical/build_congestion_aware/bin/\
AstraSim_Analytical_Congestion_Aware_IngressIdleTest
  (the binary is emitted to <build-tree>/bin/ via the targets'
  RUNTIME_OUTPUT_DIRECTORY ../bin; run it with no arguments)
*******************************************************************************/

#include <sys/wait.h>
#include <unistd.h>

#include <cassert>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <csignal>
#include <limits>
#include <thread>
#include <vector>

#include "astra-sim/workload/execution_driven/RequestIngress.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include "common/EventQueue.h"

using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

struct Fixture {
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress;
    std::vector<EventTime> arrival_ticks;
    std::vector<ServiceState> scenario1_transitions;
    std::vector<ServiceState> scenario2_transitions;
    std::vector<ServiceState> scenario3_transitions;

    // Step 1-6: the ingress now deposits ARRIVAL into the mailbox (the
    // fixture never drains it -- the mailbox is a pure sink here, and the
    // fixture has no tick-end gate).
    Fixture() { ingress.bind(&eq, &mailbox, &svc); }
};

void run_loop(Fixture& f) {
    // same shape as the online main loop: drain FIRST, then check
    // finished() (drain-before-finished per 主控裁决 2026-08-15 -- commands
    // submitted before a Close are never dropped).
    while (true) {
        f.ingress.drain_commands();
        if (f.svc.finished() &&
            f.ingress.pending_command_count() == 0) {
            break;
        }
        if (f.eq.finished()) {
            f.svc.wait_for_work();
        } else {
            f.eq.proceed();
        }
    }
}

void scenario1_no_requests(Fixture& f) {
    // Start with no requests and nothing closed: the service must stay IDLE
    // (the main loop blocks in wait_for_work, neither exiting nor busy
    // waiting). The producer thread observes IDLE before closing.
    std::thread producer([&f]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        assert(f.svc.state() == ServiceState::IDLE);
        assert(f.svc.active_request_count() == 0);
        f.ingress.mark_input_closed();
    });

    run_loop(f);
    producer.join();

    assert(f.svc.state() == ServiceState::FINISHED);
    assert(f.svc.transition_log().size() == 2);
    assert(f.svc.transition_log()[0] == ServiceState::DRAINING);
    assert(f.svc.transition_log()[1] == ServiceState::FINISHED);
    f.scenario1_transitions = f.svc.transition_log();

    std::printf("[fixture] scenario 1 PASS: IDLE -> DRAINING -> FINISHED "
                "(no busy wait, no premature exit)\n");
}

void scenario3_close_at_startup_with_preloaded_queue(Fixture& f) {
    // 主控裁决 2026-08-15: the drain-before-finished loop must NOT drop
    // commands that were submitted before the Close took effect. Both
    // Submits and the Close are queued BEFORE the loop starts; the old
    // while(!svc.finished()) shape exited immediately (finished() true at
    // entry because the input was already closed), silently discarding the
    // two requests.
    assert(f.svc.state() == ServiceState::IDLE);

    f.ingress.set_arrival_hook([&f](const RequestEnvelope& env) {
        f.arrival_ticks.push_back(f.eq.get_current_time());
        f.svc.on_request_completed();
    });

    IngressCommand c1;
    c1.kind = IngressCommandKind::Submit;
    c1.envelope.session_id = "s0";
    c1.envelope.turn_index = 0;
    c1.envelope.request_id = "s0_r0";
    c1.envelope.prefill_length = 100;
    c1.envelope.decode_length = 50;
    c1.envelope.arrival_world_ns = 1000;
    assert(f.ingress.enqueue_command(c1));

    IngressCommand c2;
    c2.kind = IngressCommandKind::Submit;
    c2.envelope.session_id = "s0";
    c2.envelope.turn_index = 1;
    c2.envelope.request_id = "s0_r1";
    c2.envelope.prefill_length = 80;
    c2.envelope.decode_length = 40;
    c2.envelope.arrival_world_ns = 2000;
    assert(f.ingress.enqueue_command(c2));

    // close-input-at-startup: the Close is queued behind the Submits
    f.ingress.mark_input_closed();

    run_loop(f);

    assert(f.svc.state() == ServiceState::FINISHED);
    assert(f.svc.accepted_request_count() == 2);
    assert(f.svc.completed_request_count() == 2);
    assert(f.arrival_ticks == std::vector<EventTime>({1000, 2000}));
    // the log records the NEW state of each transition (initial IDLE is not
    // a log entry). mark_input_closed() at startup moves IDLE -> DRAINING
    // immediately (no request accepted yet), and the preloaded requests run
    // to completion while DRAINING -- "不再接受新 request,等待已接受 request
    // 全部完成后才结束" per 合同②/总体方案 §5.5 -- then FINISHED.
    assert(f.svc.transition_log().size() == 2);
    assert(f.svc.transition_log()[0] == ServiceState::DRAINING);
    assert(f.svc.transition_log()[1] == ServiceState::FINISHED);
    f.scenario3_transitions = f.svc.transition_log();

    std::printf("[fixture] scenario 3 PASS: close-at-startup + preloaded "
                "2 requests processed (accepted=2, completed=2), no drop\n");
}

void scenario4_eof_command(Fixture& f) {
    // Phase-7 §10.7 (合同② 三态区分): an EndOfFile terminal command must
    // drain the input exactly like a close (IDLE -> DRAINING -> FINISHED),
    // and the lifecycle audit must record the source as EOF -- distinct
    // from an explicit close. The old skeleton collapsed every terminal
    // command into one mark_input_closed(); §10.7 completes the semantics.
    assert(f.svc.state() == ServiceState::IDLE);

    IngressCommand eof;
    eof.kind = IngressCommandKind::EndOfFile;
    assert(f.ingress.enqueue_command(eof));
    IngressCommand after_eof;
    after_eof.kind = IngressCommandKind::Submit;
    after_eof.envelope.request_id = "must_be_rejected_after_eof";
    assert(!f.ingress.enqueue_command(after_eof));

    run_loop(f);

    assert(f.svc.state() == ServiceState::FINISHED);
    assert(f.svc.transition_log().size() == 2);
    assert(f.svc.transition_log()[0] == ServiceState::DRAINING);
    assert(f.svc.transition_log()[1] == ServiceState::FINISHED);
    assert(f.svc.input_closed());
    assert(f.svc.input_close_reason() == InputCloseReason::EndOfFile);

    std::printf("[fixture] scenario 4 PASS: EndOfFile terminal command -> "
                "IDLE -> DRAINING -> FINISHED, close_source=EOF\n");
}

void scenario5_overflow_audit() {
    // Phase-7 §10.7: bounded-queue overflow audit. A capacity-2 ingress
    // rejects the third Submit (backpressure contract unchanged) and
    // counts the rejection; the peak occupancy is observable.
    EventQueue eq;
    DecisionMailbox mailbox;
    ServiceCoordinator svc;
    RequestIngress ingress(/*capacity=*/2);
    ingress.bind(&eq, &mailbox, &svc);

    for (int i = 0; i < 3; ++i) {
        IngressCommand cmd;
        cmd.kind = IngressCommandKind::Submit;
        cmd.envelope.session_id = "s0";
        cmd.envelope.turn_index = 0;
        cmd.envelope.request_id = "s0_r" + std::to_string(i);
        cmd.envelope.prefill_length = 100;
        cmd.envelope.decode_length = 50;
        cmd.envelope.arrival_world_ns = 1000 * (i + 1);
        const bool accepted = ingress.enqueue_command(cmd);
        assert(accepted == (i < 2));
    }
    assert(ingress.overflow_count() == 1);
    assert(ingress.peak_command_occupancy() == 2);
    assert(ingress.pending_command_count() == 2);

    std::printf("[fixture] scenario 5 PASS: overflow audit "
                "(overflow_count=1, peak_commands=2)\n");
}

void scenario6_error_command_aborts() {
    // Phase-7 §10.7 (合同②): an Error terminal command must fail-closed
    // abort (process exits non-zero via SIGABRT) -- NEVER a silent close.
    // The old skeleton collapsed Error into mark_input_closed() (silent
    // degradation); §10.7 restores the contract. Forked child so the abort
    // does not kill the fixture; the parent asserts death-by-signal.
    const pid_t pid = fork();
    if (pid == 0) {
        // child: drain an Error command -> abort() (fail-closed)
        Fixture f;
        IngressCommand err;
        err.kind = IngressCommandKind::Error;
        if (!f.ingress.enqueue_command(err)) {
            _exit(42);  // never reached; distinguishable from SIGABRT
        }
        f.ingress.drain_commands();  // must abort
        _exit(43);                   // silent-close regression would land here
    }
    assert(pid > 0);
    int status = 0;
    assert(waitpid(pid, &status, 0) == pid);
    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT) {
        std::printf("[fixture] scenario 6 PASS: Error command aborts "
                    "fail-closed (SIGABRT, not a silent close)\n");
    } else {
        std::fprintf(stderr,
                     "[fixture] FAIL: scenario 6 expected SIGABRT, got "
                     "status=%d\n",
                     status);
        std::abort();
    }
}

void scenario7_transition_log_is_bounded() {
    ServiceCoordinator svc;
    uint64_t hook_count = 0;
    svc.set_transition_hook(
        [&hook_count](ServiceState, ServiceState) { ++hook_count; });

    constexpr uint64_t cycles =
        ServiceCoordinator::kTransitionLogCapacity / 2 + 16;
    for (uint64_t i = 0; i < cycles; ++i) {
        svc.on_command_accepted();
        svc.on_alarm_scheduled();
        svc.on_request_arrived();
        svc.on_request_completed();
    }
    svc.mark_input_closed();

    const uint64_t transitions = cycles * 2 + 2;
    assert(svc.finished());
    assert(svc.transition_log().size() ==
           ServiceCoordinator::kTransitionLogCapacity);
    assert(svc.transition_log_dropped() ==
           transitions - ServiceCoordinator::kTransitionLogCapacity);
    assert(hook_count == transitions);
    for (std::size_t i = 0; i < svc.transition_log().size(); ++i) {
        assert(svc.transition_log()[i] ==
               (i % 2 == 0 ? ServiceState::ACTIVE : ServiceState::IDLE));
    }

    std::printf("[fixture] scenario 7 PASS: transition memory is capped at "
                "%zu entries while hook observed all %llu transitions\n",
                ServiceCoordinator::kTransitionLogCapacity,
                static_cast<unsigned long long>(transitions));
}

void scenario8_close_submit_race_is_linearized() {
    constexpr int kRounds = 128;
    int accepted_before_close = 0;
    for (int round = 0; round < kRounds; ++round) {
        Fixture f;
        f.ingress.set_arrival_hook(
            [&f](const RequestEnvelope&) { f.svc.on_request_completed(); });

        std::atomic<int> ready{0};
        std::atomic<bool> go{false};
        std::atomic<bool> accepted{false};
        std::thread producer([&]() {
            IngressCommand cmd;
            cmd.kind = IngressCommandKind::Submit;
            cmd.envelope.session_id = "race";
            cmd.envelope.turn_index = 0;
            cmd.envelope.request_id = "race_" + std::to_string(round);
            cmd.envelope.prefill_length = 1;
            cmd.envelope.decode_length = 1;
            cmd.envelope.arrival_world_ns = 1000;
            ready.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            accepted.store(f.ingress.enqueue_command(std::move(cmd)));
        });
        std::thread closer([&]() {
            ready.fetch_add(1);
            while (!go.load()) {
                std::this_thread::yield();
            }
            f.ingress.mark_input_closed();
        });
        while (ready.load() != 2) {
            std::this_thread::yield();
        }
        go.store(true);
        producer.join();
        closer.join();

        // This attempt is strictly after mark_input_closed() returned.
        IngressCommand after_close;
        after_close.kind = IngressCommandKind::Submit;
        after_close.envelope.request_id = "post_close";
        assert(!f.ingress.enqueue_command(std::move(after_close)));

        run_loop(f);
        const uint64_t expected = accepted.load() ? 1 : 0;
        accepted_before_close += static_cast<int>(expected);
        assert(f.ingress.pending_command_count() == 0);
        assert(f.svc.accepted_request_count() == expected);
        assert(f.svc.completed_request_count() == expected);
        assert(f.svc.finished());
    }
    std::printf("[fixture] scenario 8 PASS: %d close-vs-submit races "
                "linearized (%d accepted-before-close, remainder rejected)\n",
                kRounds, accepted_before_close);
}

void scenario2_inject_at_ticks(Fixture& f) {
    // request-neutral startup: no requests yet
    assert(f.svc.state() == ServiceState::IDLE);

    // fixture requests have no graph nodes: arrival hook completes them
    // immediately (immediate drain); arrival ticks are recorded to assert
    // the alarm fired at exactly the injected tick T
    f.ingress.set_arrival_hook([&f](const RequestEnvelope& env) {
        f.arrival_ticks.push_back(f.eq.get_current_time());
        f.svc.on_request_completed();
    });

    std::thread producer([&f]() {
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        // the loop must be blocked in wait_for_work, still IDLE
        assert(f.svc.state() == ServiceState::IDLE);

        IngressCommand c1;
        c1.kind = IngressCommandKind::Submit;
        c1.envelope.session_id = "s0";
        c1.envelope.turn_index = 0;
        c1.envelope.request_id = "s0_r0";
        c1.envelope.prefill_length = 100;
        c1.envelope.decode_length = 50;
        c1.envelope.arrival_world_ns = 1000;
        assert(f.ingress.enqueue_command(c1));

        IngressCommand c2;
        c2.kind = IngressCommandKind::Submit;
        c2.envelope.session_id = "s0";
        c2.envelope.turn_index = 1;
        c2.envelope.request_id = "s0_r1";
        c2.envelope.prefill_length = 80;
        c2.envelope.decode_length = 40;
        c2.envelope.arrival_world_ns = 2000;
        assert(f.ingress.enqueue_command(c2));

        std::this_thread::sleep_for(std::chrono::milliseconds(150));
        // step 1-10: with the input still open, both requests have drained
        // long ago (alarms at 1000/2000) and the service is back to IDLE --
        // the five-state migration IDLE -> ACTIVE -> IDLE is observable
        // before the close.
        assert(f.svc.state() == ServiceState::IDLE);
        f.ingress.mark_input_closed();
    });

    run_loop(f);
    producer.join();

    assert(f.svc.state() == ServiceState::FINISHED);
    assert(f.svc.accepted_request_count() == 2);
    assert(f.svc.completed_request_count() == 2);
    // alarms fired at exactly the injected ticks, from an otherwise empty
    // queue: no dependency on any rank being idle
    assert(f.arrival_ticks == std::vector<EventTime>({1000, 2000}));
    assert(f.svc.transition_log().size() == 4);
    assert(f.svc.transition_log()[0] == ServiceState::ACTIVE);
    assert(f.svc.transition_log()[1] == ServiceState::IDLE);
    assert(f.svc.transition_log()[2] == ServiceState::DRAINING);
    assert(f.svc.transition_log()[3] == ServiceState::FINISHED);
    f.scenario2_transitions = f.svc.transition_log();

    std::printf("[fixture] scenario 2 PASS: IDLE -> ACTIVE -> IDLE -> "
                "DRAINING -> FINISHED; alarms at exactly 1000/2000\n");
}

}  // namespace

namespace {

// FP1 (2026-09-01, sync-A16 batch P; contract §2.4/E9/E26): the checked
// deadline helper is a pure function over its inputs -- every failure mode
// leaves out_deadline untouched, and the near-max now() exercises the
// pre-addition bound (the addition itself would overflow time_point).
void scenario9_checked_deadline_bounds() {
    using clk = std::chrono::steady_clock;
    clk::time_point out;
    const clk::time_point now = clk::now();

    // normal value succeeds and produces a deadline after now
    out = clk::time_point::min();
    assert(ServiceCoordinator::checked_wait_deadline(1.5, now, out));
    assert(out > now);
    // tiny-but-legal sub-second values arm at least one tick
    assert(ServiceCoordinator::checked_wait_deadline(0.001, now, out));
    // 0 / negative / non-finite fail (helper keeps the 0-fails contract;
    // the CLI layer treats 0 as "off" and never calls it with 0)
    for (const double bad : {0.0, -1.0, -0.0,
                             std::numeric_limits<double>::infinity(),
                             -std::numeric_limits<double>::infinity(),
                             std::numeric_limits<double>::quiet_NaN()}) {
        clk::time_point sentinel = now + std::chrono::hours(1);
        assert(!ServiceCoordinator::checked_wait_deadline(bad, now, sentinel));
        assert(sentinel == now + std::chrono::hours(1));
    }
    // above duration::max() in the tick domain fails (1e9 s parses at the
    // CLI but a value beyond the clock's representable range must not)
    assert(!ServiceCoordinator::checked_wait_deadline(9.3e9, now, out));
    assert(!ServiceCoordinator::checked_wait_deadline(1e308, now, out));
    // a sub-tick positive value converts to a zero duration -> reject
    assert(!ServiceCoordinator::checked_wait_deadline(1e-12, now, out));
    // near time_point::max now(): the pre-addition bound fires (the add
    // would overflow) even for a tiny timeout, and out stays untouched
    const clk::time_point near_max =
        clk::time_point::max() - std::chrono::hours(1);
    clk::time_point sentinel2 = now;
    assert(!ServiceCoordinator::checked_wait_deadline(3600.0 * 24.0 * 3.0,
                                                      near_max, sentinel2));
    assert(sentinel2 == now);

    // wait_for_work_until semantics on a live coordinator: a timeout
    // consumes nothing (a racing signal stays pending), a wake consumes
    // wakeup_pending_ exactly like wait_for_work.
    ServiceCoordinator svc;
    const auto deadline = clk::now() + std::chrono::milliseconds(50);
    assert(!svc.wait_for_work_until(deadline));  // input open, no signal
    svc.signal_work();                           // pending, not yet consumed
    const bool woke = svc.wait_for_work_until(clk::now() +
                                              std::chrono::seconds(30));
    assert(woke);
    // the consumed pending flag is observable: a second bounded wait with
    // no new signal and the input still open must time out again
    assert(!svc.wait_for_work_until(clk::now() +
                                    std::chrono::milliseconds(20)));

    std::printf("[fixture] scenario 9 PASS: checked_wait_deadline bounds "
                "(0/neg/non-finite/over-duration/sub-tick/near-max) + "
                "wait_for_work_until consume-on-wake-only semantics\n");
}

}  // namespace

int main() {
    {
        Fixture f;
        scenario1_no_requests(f);
    }
    {
        Fixture f;
        scenario2_inject_at_ticks(f);
    }
    {
        Fixture f;
        scenario3_close_at_startup_with_preloaded_queue(f);
    }
    {
        Fixture f;
        scenario4_eof_command(f);
    }
    scenario5_overflow_audit();
    scenario6_error_command_aborts();
    scenario7_transition_log_is_bounded();
    scenario8_close_submit_race_is_linearized();
    scenario9_checked_deadline_bounds();
    std::printf("[fixture] ALL PASS: 10 state transitions logged "
                "(2 + 4 + 2 + 2) + overflow audit + fail-closed abort + "
                "bounded transition audit + close/submit linearization + "
                "FP1 checked-deadline bounds\n");
    return 0;
}
