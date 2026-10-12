/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

terminal_tail_drain_test.cc -- online shutdown-tail regression.

The logical REQUEST_COMPLETE watch can fire before the final physical tail of
the same train: its shared end-barrier is already committed and will later
produce real CompletionObserver terminal callbacks.  The former main-loop
order broke immediately at ServiceCoordinator::finished(), leaving that tail
queued and making the run-end terminal audit smaller than committed nodes.

This fixture drives the real EventQueue deferred-drain semantics and the real
CompletionObserver -> CompletedFactAccumulator counter path.  It proves both
the RED legacy behavior (one logical terminal observed, one tail terminal
still queued) and the GREEN policy (after logical service finish, drain the
main/deferred queues to quiescence before exit).
*******************************************************************************/

#include <cstdio>
#include <cstdint>

#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/DecisionMailbox.hh"
#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"
#include <astra-network-analytical/common/EventQueue.h>

using AstraSim::ExecutionDriven::CompletedFactAccumulator;
using AstraSim::ExecutionDriven::CompletionObserver;
using AstraSim::ExecutionDriven::NodeTerminalStatus;
using AstraSim::ExecutionDriven::ServiceCoordinator;
using NetworkAnalytical::EventQueue;

namespace {

constexpr uint64_t kCommittedNodes = 2;

struct RunContext {
    EventQueue event_queue;
    ServiceCoordinator service;
    CompletedFactAccumulator terminals;
    uint64_t deferred_issue_passes = 0;
    uint64_t tail_terminal_callbacks = 0;
    bool unexpected_empty_queue = false;
};

void count_terminal(void* opaque, int rank, uint64_t node_id,
                    const char* request_id, const char* stage,
                    uint64_t generation, uint64_t tick, int terminal_status) {
    auto* const ctx = static_cast<RunContext*>(opaque);
    ctx->terminals.record(rank, node_id, request_id, stage, generation, tick,
                          terminal_status);
}

void tail_terminal(void* opaque) {
    auto* const ctx = static_cast<RunContext*>(opaque);
    ++ctx->tail_terminal_callbacks;
    CompletionObserver::instance().record_node_terminal(
        /*rank=*/0, /*node_id=*/2, "tail_request", "decode",
        /*generation=*/1, ctx->event_queue.get_current_time(),
        NodeTerminalStatus::Success);
}

void deferred_issue_tail(void* opaque) {
    auto* const ctx = static_cast<RunContext*>(opaque);
    ++ctx->deferred_issue_passes;
    // This is the post-watch deferred issue pass: it puts the already
    // committed end-barrier onto the future physical EventQueue.
    ctx->event_queue.schedule_event(ctx->event_queue.get_current_time() + 1,
                                    tail_terminal, ctx);
}

void logical_request_complete(void* opaque) {
    auto* const ctx = static_cast<RunContext*>(opaque);
    CompletionObserver::instance().record_node_terminal(
        /*rank=*/0, /*node_id=*/1, "tail_request", "decode",
        /*generation=*/1, ctx->event_queue.get_current_time(),
        NodeTerminalStatus::Success);
    // The production callback schedules the ready-set drain as deferred work.
    // It then commits REQUEST_COMPLETE, so the logical service is FINISHED
    // while the physical tail remains scheduled for the next tick.
    ctx->event_queue.schedule_event_deferred(deferred_issue_tail, ctx);
    ctx->service.on_request_completed();
}

struct RunResult {
    uint64_t terminal_total = 0;
    uint64_t tail_terminal_callbacks = 0;
    uint64_t deferred_issue_passes = 0;
    bool service_finished = false;
    bool main_queue_empty = false;
    bool deferred_queue_empty = false;
    bool unexpected_empty_queue = false;
};

RunResult run_shutdown_loop(const bool drain_physical_tail) {
    RunContext ctx;
    CompletionObserver::instance().set_hook(count_terminal, &ctx);

    // One close-input request.  Its logical completion at tick 100 is valid,
    // but the final committed barrier terminal is deliberately at tick 101.
    ctx.service.on_command_accepted();
    ctx.service.on_alarm_scheduled();
    ctx.service.on_request_arrived();
    ctx.service.mark_input_closed();
    ctx.event_queue.schedule_event(/*tick=*/100, logical_request_complete,
                                   &ctx);

    while (true) {
        const bool service_finished = ctx.service.finished();
        if (!drain_physical_tail && service_finished) {
            // Pre-fix main_online.cc behavior: the queued tail is abandoned.
            break;
        }
        if (ctx.event_queue.finished()) {
            if (ctx.event_queue.has_deferred_work()) {
                // Same wakeup mechanism main_online.cc uses when the main map
                // is empty but a deferred issue pass still needs a proceed().
                ctx.event_queue.schedule_event(ctx.event_queue.get_current_time() + 1,
                                               [](void*) {}, nullptr);
            } else if (service_finished) {
                // Fixed shutdown gate: logical finish plus physical
                // EventQueue/deferred quiescence is the only exit condition.
                break;
            } else {
                ctx.unexpected_empty_queue = true;
                break;
            }
        } else {
            ctx.event_queue.proceed();
        }
    }

    CompletionObserver::instance().set_hook(nullptr, nullptr);
    const auto& counts = ctx.terminals.counters();
    return RunResult{counts.total,
                     ctx.tail_terminal_callbacks,
                     ctx.deferred_issue_passes,
                     ctx.service.finished(),
                     ctx.event_queue.finished(),
                     !ctx.event_queue.has_deferred_work(),
                     ctx.unexpected_empty_queue};
}

bool expect(const bool condition, const char* what) {
    if (condition) {
        return true;
    }
    std::fprintf(stderr, "[terminal_tail_drain_test] FAIL: %s\n", what);
    return false;
}

}  // namespace

int main() {
    bool ok = true;

    const RunResult legacy = run_shutdown_loop(/*drain_physical_tail=*/false);
    ok &= expect(legacy.service_finished,
                 "legacy reproduction reaches logical service finish");
    ok &= expect(legacy.terminal_total + 1 == kCommittedNodes,
                 "legacy reproduction leaves exactly one committed tail without "
                 "a CompletionObserver terminal callback");
    ok &= expect(legacy.tail_terminal_callbacks == 0,
                 "legacy reproduction never executes the tail terminal");
    ok &= expect(!legacy.main_queue_empty,
                 "legacy reproduction exits while the physical EventQueue is non-empty");
    ok &= expect(legacy.deferred_queue_empty,
                 "legacy deferred issue pass ran and exposed the future tail");
    ok &= expect(legacy.deferred_issue_passes == 1,
                 "legacy reproduction ran exactly one deferred issue pass");
    ok &= expect(!legacy.unexpected_empty_queue,
                 "legacy reproduction did not take an unrelated empty-queue path");

    const RunResult fixed = run_shutdown_loop(/*drain_physical_tail=*/true);
    ok &= expect(fixed.service_finished,
                 "fixed shutdown reaches logical service finish");
    ok &= expect(fixed.terminal_total == kCommittedNodes,
                 "fixed shutdown observes every committed terminal callback");
    ok &= expect(fixed.tail_terminal_callbacks == 1,
                 "fixed shutdown executes the queued end-barrier terminal");
    ok &= expect(fixed.main_queue_empty && fixed.deferred_queue_empty,
                 "fixed shutdown exits only after physical queues are quiescent");
    ok &= expect(fixed.deferred_issue_passes == 1,
                 "fixed shutdown preserves the deferred issue-pass lifecycle");
    ok &= expect(!fixed.unexpected_empty_queue,
                 "fixed shutdown did not take an unrelated empty-queue path");

    if (!ok) {
        return 1;
    }
    std::printf("[terminal_tail_drain_test] PASS: legacy terminal gap "
                "reproduced; fixed shutdown drains the real tail callback "
                "(%llu/%llu terminals)\n",
                static_cast<unsigned long long>(fixed.terminal_total),
                static_cast<unsigned long long>(kCommittedNodes));
    return 0;
}
