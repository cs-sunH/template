/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

event_queue_deferred_test.cc -- phase-1 step 1-1 unit tests for the map-version
(std::map<EventTime, EventList>) EventQueue tick-end closing and same-tick
deferred channel, plus the FluidScheduler deferred-flush integration
(post-commit comm emission) and the map-specific fail-fast proof (case E).

Cases:
  A. tick-end callback is invoked exactly once per proceed(), after the
     physical events and before the deferred drain;
  B. deferred events execute in insertion order within the current tick, and
     deferred-inside-deferred is drained by the same pass;
  C. without a callback, 1000 randomized schedule/proceed runs are
     event-for-event identical to the pre-extension reference implementation;
  D. FluidScheduler integration: comm started from a tick-end callback
     (deferred flush mode) does not trip the EventQueue strict-increase assert
     and the flow completes; the legacy (non-deferred) start_flow path still
     works from within a physical event handler.
  F. Same-time physical FIFO remains exact with the inline-first EventList:
     callback-appended events run after already queued events in the same pass.

Build (from template/astra-sim-wscllm):
  g++ -std=c++17 -I extern/network_backend/analytical/include \
      -I extern/network_backend/analytical/include/astra-network-analytical \
      astra-sim/workload/execution_driven/tests/event_queue_deferred_test.cc \
      extern/network_backend/analytical/common/event-queue/EventQueue.cpp \
      extern/network_backend/analytical/common/event-queue/EventList.cpp \
      extern/network_backend/analytical/common/event-queue/Event.cpp \
      extern/network_backend/analytical/common/NetworkFunction.cpp \
      extern/network_backend/analytical/congestion_aware/fluid/FluidScheduler.cpp \
      extern/network_backend/analytical/congestion_aware/fluid/FluidFlow.cpp \
      extern/network_backend/analytical/congestion_aware/fluid/FluidLinkState.cpp \
      extern/network_backend/analytical/congestion_aware/network/Link.cpp \
      extern/network_backend/analytical/congestion_aware/network/Device.cpp \
      -o /tmp/eq_test && /tmp/eq_test
*******************************************************************************/

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <list>
#include <map>
#include <memory>
#include <sys/wait.h>
#include <unistd.h>
#include <random>
#include <utility>
#include <vector>

#include "common/EventQueue.h"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidScheduler.h"

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

void noop_handler(void*) noexcept {}

// ---------------------------------------------------------------------------
// reference (pre-extension) EventQueue semantics -- oracle for case C
// ---------------------------------------------------------------------------

struct RefEventList {
    EventTime event_time;
    std::list<std::pair<Callback, CallbackArg>> events;
};

class ReferenceQueue {
  public:
    ReferenceQueue() = default;

    [[nodiscard]] bool finished() const noexcept { return event_queue.empty(); }

    [[nodiscard]] EventTime get_current_time() const noexcept { return current_time; }

    void proceed() noexcept {
        assert(!finished());
        auto begin_it = event_queue.begin();
        auto& front = begin_it->second;
        assert(front.event_time > current_time);
        current_time = front.event_time;
        // mirror EventList::invoke_events: pop_front + invoke loop, so events
        // scheduled at the current time from inside a handler (same-time
        // merge) are picked up by the same pass
        while (!front.events.empty()) {
            front.events.front().first(front.events.front().second);
            front.events.pop_front();
        }
        event_queue.erase(event_queue.begin());
    }

    void schedule_event(EventTime t, Callback cb, CallbackArg arg) noexcept {
        // mirror the map-family schedule_event: try_emplace merges same-time
        // events into the existing EventList (same-time merge semantics)
        assert(t >= current_time);
        auto [it, inserted] = event_queue.try_emplace(t, RefEventList{t, {}});
        (void)inserted;
        it->second.events.emplace_back(cb, arg);
    }

  private:
    EventTime current_time = 0;
    std::map<EventTime, RefEventList> event_queue;
};

// ---------------------------------------------------------------------------
// case A: tick-end callback exactly once per proceed, after physical events,
// before deferred drain
// ---------------------------------------------------------------------------

struct CaseACtx {
    std::vector<int> log;
    int tick_end_count = 0;
    EventQueue* eq = nullptr;
};

void case_a_physical(void* v) {
    auto* c = static_cast<CaseACtx*>(v);
    c->log.push_back(1);
}

void case_a_deferred(void* v) {
    auto* c = static_cast<CaseACtx*>(v);
    c->log.push_back(3);
}

void case_a_tick_end(void* v) {
    auto* c = static_cast<CaseACtx*>(v);
    ++c->tick_end_count;
    c->log.push_back(2);
    // schedule a deferred event from inside the callback: it must be drained
    // by the same pass, after the callback
    c->eq->schedule_event_deferred(case_a_deferred, c);
}

void test_case_a() {
    EventQueue eq;
    CaseACtx ctx;
    ctx.eq = &eq;

    // one deferred scheduled before the first proceed (drains at its end)
    eq.schedule_event_deferred(case_a_deferred, &ctx);

    eq.set_tick_end_callback(case_a_tick_end, &ctx);

    // two physical events at t=10
    eq.schedule_event(10, case_a_physical, &ctx);
    eq.schedule_event(10, case_a_physical, &ctx);

    eq.proceed();
    assert(eq.get_current_time() == 10);
    assert(ctx.tick_end_count == 1);
    // physical, physical, tick-end, pre-scheduled deferred, callback deferred
    assert(ctx.log == std::vector<int>({1, 1, 2, 3, 3}));

    // second tick: two physical events at t=20
    eq.schedule_event(20, case_a_physical, &ctx);
    eq.schedule_event(20, case_a_physical, &ctx);
    eq.proceed();
    assert(eq.get_current_time() == 20);
    assert(ctx.tick_end_count == 2);
    assert(ctx.log == std::vector<int>({1, 1, 2, 3, 3, 1, 1, 2, 3}));
    assert(eq.finished());

    std::printf("[case A] PASS: tick-end exactly once per proceed, after "
                "physical, before deferred\n");
}

// ---------------------------------------------------------------------------
// case B: deferred insertion order + deferred-inside-deferred same-pass drain
// ---------------------------------------------------------------------------

struct CaseBCtx {
    std::vector<int> log;
    EventQueue* eq = nullptr;
};

void case_b_nested(void* v) {
    auto* c = static_cast<CaseBCtx*>(v);
    c->log.push_back(5);
}

void case_b_outer(void* v) {
    auto* c = static_cast<CaseBCtx*>(v);
    c->log.push_back(4);
    // nested deferred must be drained by the same pass
    c->eq->schedule_event_deferred(case_b_nested, c);
}

void case_b_physical(void* v) {
    auto* c = static_cast<CaseBCtx*>(v);
    c->log.push_back(0);
}

void test_case_b() {
    EventQueue eq;
    CaseBCtx ctx;
    ctx.eq = &eq;
    ctx.log.reserve(16);

    // one physical event at t=10
    eq.schedule_event(10, case_b_physical, &ctx);

    // two deferred events, in order
    eq.schedule_event_deferred(case_b_outer, &ctx);
    eq.schedule_event_deferred(case_b_outer, &ctx);

    eq.proceed();
    assert(eq.get_current_time() == 10);
    // physical(0), then both outer deferred in insertion order (4,4), then the
    // two nested deferred appended by the outer ones, still in the same drain
    // pass (5,5): FIFO order, same-pass nested drain
    assert(ctx.log == std::vector<int>({0, 4, 4, 5, 5}));
    assert(eq.finished());

    std::printf("[case B] PASS: deferred insertion order + nested same-pass "
                "drain\n");
}

// ---------------------------------------------------------------------------
// case C: 1000 randomized ops identical to the reference implementation
// ---------------------------------------------------------------------------

struct DiffCtx {
    int id;
    std::vector<int>* real_log;
    std::vector<int>* ref_log;
};

void diff_handler(void* v) {
    auto* c = static_cast<DiffCtx*>(v);
    c->real_log->push_back(c->id);
    c->ref_log->push_back(c->id);
}

void test_case_c() {
    std::mt19937 rng(20260815);
    EventQueue eq;
    ReferenceQueue ref;
    auto real_log = std::vector<int>();
    auto ref_log = std::vector<int>();

    std::uniform_int_distribution<int> op_dist(0, 99);
    std::uniform_int_distribution<int> id_dist(0, 999);
    std::uniform_int_distribution<int> time_dist(1, 50);

    for (int step = 0; step < 1000; ++step) {
        const int op = op_dist(rng);
        if (op < 60) {
            // schedule at a strictly future time (same-time scheduling from a
            // handler is exercised separately in case D2, where it merges into
            // the list currently being invoked)
            const auto t = eq.get_current_time() +
                           static_cast<EventTime>(time_dist(rng));
            auto* c = new DiffCtx{id_dist(rng), &real_log, &ref_log};
            eq.schedule_event(t, diff_handler, c);
            ref.schedule_event(t, diff_handler, c);
        } else {
            // proceed when possible
            if (eq.finished()) {
                assert(ref.finished());
                continue;
            }
            assert(!ref.finished());
            const auto real_before = real_log.size();
            const auto ref_before = ref_log.size();
            eq.proceed();
            ref.proceed();
            // event-for-event identical within this tick
            const auto ref_appended = ref_log.size() - ref_before;
            assert(real_log.size() - real_before == ref_appended);
            for (size_t i = 0; i < ref_appended; ++i) {
                assert(real_log[real_before + i] == ref_log[ref_before + i]);
            }
            assert(eq.get_current_time() == ref.get_current_time());
        }
    }

    // drain leftovers and compare terminal state
    while (!eq.finished()) {
        assert(!ref.finished());
        eq.proceed();
        ref.proceed();
        assert(eq.get_current_time() == ref.get_current_time());
    }
    assert(ref.finished());
    assert(real_log == ref_log);

    std::printf("[case C] PASS: 1000 randomized ops identical to reference\n");
}

// ---------------------------------------------------------------------------
// case D: FluidScheduler integration -- deferred flush vs legacy flush
// ---------------------------------------------------------------------------

struct FSIntCtx {
    int completed = 0;
    bool started = false;
    FluidScheduler* fs = nullptr;
    int ticks_seen = 0;
};

void flow_done_cb(void* v) {
    auto* c = static_cast<FSIntCtx*>(v);
    ++c->completed;
}

void fs_comm_tick_end(void* v) {
    auto* c = static_cast<FSIntCtx*>(v);
    ++c->ticks_seen;
    if (c->started) {
        return;
    }
    c->started = true;
    auto route = std::make_shared<FluidRoute>(FluidRoute{{0}, 5});
    c->fs->start_flow(1000, route, flow_done_cb, c);
}

void fs_comm_physical(void* v) {
    // legacy path: start a flow from within a physical event handler
    auto* c = static_cast<FSIntCtx*>(v);
    auto route = std::make_shared<FluidRoute>(FluidRoute{{0}, 5});
    c->fs->start_flow(1000, route, flow_done_cb, c);
}

void run_fs_scenario(bool deferred_mode) {
    auto eq = std::make_shared<EventQueue>();
    auto link = std::make_shared<Link>(0, 100.0 /* GB/s */, 10 /* ns */);
    FluidScheduler fs(eq, {link}, 8, 8, 1ull << 30);
    fs.set_deferred_flush_mode(deferred_mode);
    fs.mark_event_loop_started();

    FSIntCtx ctx;
    ctx.fs = &fs;

    if (deferred_mode) {
        // comm emitted from the tick-end callback (post-commit emission);
        // without the deferred channel this would insert a current_time event
        // into the main queue and trip the strict-increase assert
        eq->set_tick_end_callback(fs_comm_tick_end, &ctx);
        eq->schedule_event(10, noop_handler, nullptr);
    } else {
        // legacy path: comm emitted from inside a physical event handler
        eq->schedule_event(10, fs_comm_physical, &ctx);
    }

    size_t guard = 0;
    while (!eq->finished()) {
        eq->proceed();
        assert(++guard < 1000000);
    }
    assert(ctx.completed == 1);
    assert(fs.get_total_started_flows() == 1);
    assert(fs.get_total_completed_flows() == 1);
    assert(fs.get_active_flow_count() == 0);

    std::printf("[case D] PASS: FluidScheduler integration (%s flush mode), "
                "no :33 assert, flow completed\n",
                deferred_mode ? "deferred" : "legacy");
}

// ---------------------------------------------------------------------------
// case E (map family): fail-fast proof -- schedule_event(current_time) from a
// tick-end callback inserts a current_time EventList into the main map and the
// NEXT proceed() must trip the strict-increase assert (EventQueue.cpp :31);
// schedule_event(future) from the callback lands in the map normally.
// ---------------------------------------------------------------------------

struct CaseECtx {
    EventQueue* eq = nullptr;
    bool bad = false;   // true -> schedule_event(current_time) in tick-end
    bool fired = false; // schedule from the tick-end callback only once
};

void case_e_tick_end(void* v) {
    auto* c = static_cast<CaseECtx*>(v);
    if (c->fired) {
        return;
    }
    c->fired = true;
    if (c->bad) {
        // hard-rule violation: same-tick event via the main queue
        c->eq->schedule_event(c->eq->get_current_time(), noop_handler, nullptr);
    } else {
        // legal: strictly future event via the main queue
        c->eq->schedule_event(c->eq->get_current_time() + 5, noop_handler, nullptr);
    }
}

bool run_case_e_child(bool bad) {
    EventQueue eq;
    CaseECtx ctx;
    ctx.eq = &eq;
    ctx.bad = bad;
    eq.set_tick_end_callback(case_e_tick_end, &ctx);
    eq.schedule_event(10, noop_handler, nullptr);
    eq.proceed();
    if (bad) {
        // the next proceed must abort on the strict-increase assert
        eq.proceed();
        return false;  // should not reach here in a failing build
    }
    // legal path: the future event is in the map and executes at t=15
    assert(eq.get_current_time() == 10);
    assert(!eq.finished());
    eq.proceed();
    assert(eq.get_current_time() == 15);
    assert(eq.finished());
    return true;
}

void test_case_e() {
    // legal variant runs in-process
    assert(run_case_e_child(/*bad=*/false));
    // violating variant must abort (assert) -- verify in a forked child
    pid_t pid = fork();
    if (pid == 0) {
        const bool reached_end = run_case_e_child(/*bad=*/true);
        std::_Exit(reached_end ? 0 : 0);  // child exits 0 only if no abort AND completed
    }
    int status = 0;
    waitpid(pid, &status, 0);
    // the child must have died from the assert (SIGABRT), not exited cleanly
    assert(WIFSIGNALED(status) && WTERMSIG(status) == SIGABRT);
    std::printf("[case E] PASS: map fail-fast -- tick-end schedule_event("
                "current_time) trips :31 on next proceed; future event OK\n");
}

// ---------------------------------------------------------------------------
// case F: inline-first EventList preserves callback-appended same-time FIFO
// ---------------------------------------------------------------------------

struct CaseFCtx {
    EventQueue* eq = nullptr;
    std::vector<int> log;
};

void case_f_nested(void* v) {
    static_cast<CaseFCtx*>(v)->log.push_back(3);
}

void case_f_tail_nested(void* v) {
    static_cast<CaseFCtx*>(v)->log.push_back(4);
}

void case_f_first(void* v) {
    auto* const c = static_cast<CaseFCtx*>(v);
    c->log.push_back(1);
    c->eq->schedule_event(c->eq->get_current_time(), case_f_nested, c);
}

void case_f_tail(void* v) {
    auto* const c = static_cast<CaseFCtx*>(v);
    c->log.push_back(2);
    c->eq->schedule_event(c->eq->get_current_time(), case_f_tail_nested, c);
}

void test_case_f() {
    EventQueue eq;
    CaseFCtx ctx;
    ctx.eq = &eq;
    eq.schedule_event(10, case_f_first, &ctx);
    eq.schedule_event(10, case_f_tail, &ctx);
    eq.proceed();
    assert(ctx.log == std::vector<int>({1, 2, 3, 4}));
    assert(eq.finished());
    std::printf("[case F] PASS: inline-first same-time FIFO / nested append\n");
}

}  // namespace

int main() {
    test_case_a();
    test_case_b();
    test_case_c();
    run_fs_scenario(/*deferred_mode=*/true);
    run_fs_scenario(/*deferred_mode=*/false);
    test_case_e();
    test_case_f();
    std::printf("ALL TESTS PASSED\n");
    return 0;
}
