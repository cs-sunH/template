/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
alarm_cancellation_test.cc -- end-to-end outer alarm cancellation regression
(R1, frozen plan §3.3 / §6.2-1).

Runtime coverage for the five-layer cancellable-alarm chain:
Sys timestamp bucket -> CommonNetworkApi opt-in registry -> analytical
backend EventQueue (EventList/QueuedEvent).  The fixture reuses the
LocalHbmTest approach: real configs, a real CongestionAwareNetworkApi
frontend, a real backend EventQueue, and a real Sys; only the payloads are
test-local CallData objects.

Cases:
  A. cancelling the LAST internal event of a timestamp bucket physically
     removes the outer alarm from the backend EventQueue (the main queue is
     empty afterwards -- not a callback-turned-no-op shell);
  B. several internal events share one outer alarm per bucket: the alarm
     survives cancellations while the bucket is non-empty (cancellable and
     legacy register_event occupants alike) and is cascaded only when the
     bucket empties; a surviving event keeps its original trigger tick and
     fires exactly once (equivalence red-line: cancellation must not disturb
     live events);
  C. repeated cancellation is idempotent: a consumed handle safely returns
     false, payload cleanup runs exactly once per event, and a fresh bucket
     is never re-erased by a stale handle;
  D. a legacy/fake backend that cannot cancel keeps its alarm resident after
     the Sys bucket is gone; the stale guards (missing bucket in
     Sys::call_events, retired Sys in Sys::handleEvent) drop the leftover
     alarm with zero side effects -- both while the Sys is alive and after
     its deletion.

Build: cmake target AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest.
Run: build/astra_analytical/build_congestion_aware/bin/\
     AstraSim_Analytical_Congestion_Aware_AlarmCancellationTest
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "congestion_aware/CongestionAwareNetworkApi.hh"
#include <astra-network-analytical/common/EventQueue.h>
#include <astra-network-analytical/common/NetworkParser.h>
#include <astra-network-analytical/congestion_aware/Helper.h>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <sys/wait.h>
#include <unistd.h>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace AstraSimAnalytical;
using namespace AstraSimAnalyticalCongestionAware;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

// ---- assertion helpers (same style as local_hbm_model_test.cc) ----

void expect_true(const bool cond, const char* const what) {
    std::printf("%-62s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        std::exit(1);
    }
}

// ---- test-local payload: delivered by the normal callback path (the
// consumer owns it after Sys pops the event) or reclaimed exactly once by
// the cancellation callback (Sys::cancel_event path) ----

struct Payload final : public CallData {
    Payload(int* deliveries_in, int* cleanups_in)
        : deliveries(deliveries_in), cleanups(cleanups_in) {}

    int* deliveries = nullptr;
    int* cleanups = nullptr;
};

Payload* make_payload(int* deliveries, int* cleanups) {
    return new Payload{deliveries, cleanups};
}

void reclaim_payload(CallData* data) {
    auto* const payload = static_cast<Payload*>(data);
    ++*payload->cleanups;
    delete payload;
}

class CountingCallable final : public Callable {
  public:
    void call(EventType, CallData* data) override {
        ++*deliveries;
        delete data;  // the invoked consumer takes over the payload
    }

    int* deliveries = nullptr;
};

// ---- fixture: real configs + real congestion-aware frontend + real
// backend EventQueue + real Sys (contention off: no HBM model events) ----

const char* kSystemJson = R"({
  "scheduling-policy": "LIFO",
  "endpoint-delay": 10,
  "active-chunks-per-dimension": 1,
  "preferred-dataset-splits": 6,
  "all-reduce-implementation": ["ring", "ring"],
  "all-gather-implementation": ["ring", "ring"],
  "reduce-scatter-implementation": ["ring", "ring"],
  "all-to-all-implementation": ["ring", "ring"],
  "collective-optimization": "localBWAware",
  "roofline-enabled": 1,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 0,
  "peak-perf": 261.12,
  "local-mem-bw": 1640.0,
  "local-mem-latency": 100
}
)";

const char* kNetworkYaml = R"(topology: [ Line, Line ]
npus_count: [ 4, 1 ]
bandwidth: [ 400.0, 400.0 ]
latency: [ 5, 5 ]
)";

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    assert(out.is_open());
    out << content;
}

struct AnalyticalFixture {
    std::shared_ptr<EventQueue> event_queue;
    // Declared before `sys` dies: Sys::cancel_event re-enters the API while
    // the Workload destructor cancels pending model events, so the API must
    // outlive the Sys (the fixture struct is destroyed only after main()
    // has deleted the Sys explicitly).
    std::unique_ptr<CongestionAwareNetworkApi> network_api;
    Sys* sys = nullptr;
};

AnalyticalFixture make_analytical_fixture(const std::string& config_dir) {
    AstraSim::LoggerFactory::init("empty", "off");

    AnalyticalFixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    const auto network_parser =
        NetworkParser(config_dir + "/network.yml");
    const auto topology = construct_topology(network_parser);
    CongestionAwareNetworkApi::set_event_queue(fixture.event_queue);
    CongestionAwareNetworkApi::set_topology(topology);
    const auto fluid_scheduler = std::make_shared<FluidScheduler>(
        fixture.event_queue, topology->get_directed_links(),
        network_parser.get_fluid_max_active_flows(),
        network_parser.get_fluid_max_route_memberships(),
        network_parser.get_progress_report_event_interval());
    fluid_scheduler->set_deferred_flush_mode(true);
    CongestionAwareNetworkApi::set_fluid_scheduler(fluid_scheduler);

    fixture.network_api = std::make_unique<CongestionAwareNetworkApi>(0);
    auto graph_source = std::make_shared<NodeStoreGraphSource>();
    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim;
    for (size_t i = 0; i < npus_count_per_dim.size(); ++i) {
        queues_per_dim.push_back(1);
    }
    fixture.sys = new Sys(0, config_dir + "/workload", config_dir +
                          "/comm_group.json", config_dir + "/system.json",
                          fixture.network_api.get(),
                          npus_count_per_dim, queues_per_dim, 1.0, 1.0,
                          false, ExecutionDriven::ExecutionMode::Online,
                          graph_source);
    return fixture;
}

// ---- legacy/fake backend: implements only the mandatory AstraNetworkAPI
// surface (which now includes the pure-virtual get_BW_at_dimension), so
// sim_schedule_cancellable/sim_cancel_event keep the base-class
// legacy behavior (schedule normally, report an invalid handle, never
// cancel).  Its alarms land in an ordinary, non-cancellable backend queue. ----

class FakeLegacyNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeLegacyNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    // AstraNetworkAPI 纯虚面:桩无链路带宽语义,返回有限占位值即可。
    double get_BW_at_dimension(int /*dim*/) override { return 1.0; }

    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(const timespec_t delta, void (*fun_ptr)(void*),
                      void* fun_arg) override {
        timespec_t now = sim_get_time();
        queue_->schedule_event(now.time_val + delta.time_val, fun_ptr,
                               fun_arg);
    }

    timespec_t sim_get_time() override {
        timespec_t now;
        now.time_res = NS;
        now.time_val = queue_->get_current_time();
        return now;
    }

  private:
    std::shared_ptr<EventQueue> queue_;
};

// ===========================================================================
// Case A: cancelling the last internal event physically removes the outer
// alarm from the backend EventQueue.
// ===========================================================================

void test_last_event_removal_is_physical(AnalyticalFixture& fixture) {
    std::printf("[case A] last-event cancellation is physical\n");
    int deliveries = 0;
    int cleanups = 0;
    CountingCallable callable;
    callable.deliveries = &deliveries;

    auto handle = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        100, reclaim_payload);
    expect_true(handle.valid(), "A: cancellable registration returns a handle");
    expect_true(fixture.sys->pending_events == 1,
                "A: one pending internal event");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "A: outer alarm physically resident in backend queue");

    expect_true(fixture.sys->cancel_event(handle),
                "A: cancelling the only bucket event succeeds");
    expect_true(!handle.valid(), "A: successful cancellation consumes handle");
    expect_true(cleanups == 1, "A: payload reclaimed exactly once");
    expect_true(deliveries == 0, "A: cancelled callback never delivered");
    expect_true(fixture.sys->pending_events == 0,
                "A: no pending internal events remain");
    // The core physical-removal assertion: the main backend queue is empty,
    // so a later proceed() cannot even observe the alarm.
    expect_true(fixture.event_queue->scheduled_event_count() == 0,
                "A: outer alarm physically removed from backend queue");
    expect_true(fixture.event_queue->finished(),
                "A: backend main queue fully drained by the cancellation");
}

// ===========================================================================
// Case B: one outer alarm per timestamp bucket; cascaded cancellation only
// when the bucket empties; surviving events keep their trigger tick.
// ===========================================================================

void test_bucket_scoped_cascade(AnalyticalFixture& fixture) {
    std::printf("[case B] bucket-scoped outer-alarm cascade\n");
    // B1: two cancellable events in one bucket.  The shared outer alarm must
    // survive the first cancellation and disappear with the second one.
    int deliveries = 0;
    int cleanups = 0;
    CountingCallable callable;
    callable.deliveries = &deliveries;

    auto first = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        150, reclaim_payload);
    auto second = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        150, reclaim_payload);
    expect_true(fixture.sys->pending_events == 2,
                "B1: two internal events pending");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "B1: one bucket -> exactly one outer alarm");

    expect_true(fixture.sys->cancel_event(first),
                "B1: first event cancels");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "B1: outer alarm survives while the bucket is non-empty");
    expect_true(fixture.sys->pending_events == 1,
                "B1: one internal event still pending");

    expect_true(fixture.sys->cancel_event(second),
                "B1: second (last) event cancels");
    expect_true(fixture.event_queue->scheduled_event_count() == 0,
                "B1: emptying the bucket cascades the outer alarm away");
    expect_true(cleanups == 2, "B1: both payloads reclaimed exactly once");
    expect_true(fixture.event_queue->finished(),
                "B1: backend main queue empty after the cascade");

    // B2: mixed bucket -- one cancellable plus one legacy register_event
    // occupant.  The outer alarm must stay until the legacy event actually
    // fires, and that survivor must keep its original trigger tick.
    deliveries = 0;
    cleanups = 0;
    auto cancellable = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        200, reclaim_payload);
    fixture.sys->register_event(&callable, EventType::General,
                                make_payload(&deliveries, &cleanups), 200);
    expect_true(fixture.sys->pending_events == 2,
                "B2: mixed bucket holds two internal events");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "B2: mixed bucket still has exactly one outer alarm");

    expect_true(fixture.sys->cancel_event(cancellable),
                "B2: cancellable occupant removed");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "B2: legacy occupant keeps the outer alarm resident");
    expect_true(fixture.sys->pending_events == 1,
                "B2: legacy occupant still pending");

    fixture.event_queue->proceed();
    expect_true(deliveries == 1 && cleanups == 1,
                "B2: survivor delivered once, cancelled one reclaimed once");
    expect_true(fixture.event_queue->get_current_time() == 200,
                "B2: survivor keeps its original trigger tick (200)");
    expect_true(fixture.sys->pending_events == 0,
                "B2: no pending internal events after the tick");
    expect_true(fixture.event_queue->finished(),
                "B2: backend queue drained by the normal delivery");
}

// ===========================================================================
// Case C: repeated cancellation is idempotent.
// ===========================================================================

void test_repeat_cancellation_is_idempotent(AnalyticalFixture& fixture) {
    std::printf("[case C] repeated cancellation idempotence\n");
    int deliveries = 0;
    int cleanups = 0;
    CountingCallable callable;
    callable.deliveries = &deliveries;

    // An empty default handle can never cancel anything.
    SystemEventHandle invalid;
    expect_true(!fixture.sys->cancel_event(invalid),
                "C: invalid handle cancellation safely returns false");

    auto first = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        300, reclaim_payload);
    auto second = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        300, reclaim_payload);

    expect_true(fixture.sys->cancel_event(first), "C: first cancel succeeds");
    expect_true(!fixture.sys->cancel_event(first),
                "C: re-cancelling a consumed handle returns false");
    expect_true(cleanups == 1, "C: exactly-once cleanup after double cancel");
    expect_true(fixture.sys->pending_events == 1,
                "C: second event untouched by the stale handle");

    expect_true(fixture.sys->cancel_event(second),
                "C: second cancel succeeds");
    expect_true(!fixture.sys->cancel_event(first) &&
                    !fixture.sys->cancel_event(second),
                "C: stale handles never re-erase a fresh bucket");
    expect_true(cleanups == 2 && deliveries == 0,
                "C: two cleanups total, zero deliveries");
    expect_true(fixture.event_queue->scheduled_event_count() == 0,
                "C: backend queue empty after idempotent cancels");
}

// ===========================================================================
// Case D: legacy/fake backend fallback -- the alarm cannot be removed, and
// the stale guards must drop it without side effects.
// ===========================================================================

void test_legacy_backend_fallback(const std::string& config_dir) {
    std::printf("[case D] legacy backend fallback + stale guards\n");
    // An independent backend queue: the fake API never touches the shared
    // analytical frontend statics.
    auto legacy_queue = std::make_shared<EventQueue>();
    auto legacy_api = std::make_unique<FakeLegacyNetworkApi>(legacy_queue);
    auto graph_source = std::make_shared<NodeStoreGraphSource>();
    const auto network_parser = NetworkParser(config_dir + "/network.yml");
    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim;
    for (size_t i = 0; i < npus_count_per_dim.size(); ++i) {
        queues_per_dim.push_back(1);
    }
    auto* const legacy_sys =
        new Sys(1, config_dir + "/workload", config_dir + "/comm_group.json",
                config_dir + "/system.json", legacy_api.get(),
                npus_count_per_dim, queues_per_dim, 1.0, 1.0, false,
                ExecutionDriven::ExecutionMode::Online, graph_source);

    int deliveries = 0;
    int cleanups = 0;
    CountingCallable callable;
    callable.deliveries = &deliveries;

    // D1: cancellation removes the Sys bucket and reclaims the payload, but
    // the legacy backend keeps its alarm; draining it must be a no-op.
    auto stale = legacy_sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        50, reclaim_payload);
    expect_true(stale.valid(), "D1: Sys-level handle is still returned");
    expect_true(legacy_queue->scheduled_event_count() == 1,
                "D1: legacy backend queued the outer alarm");
    expect_true(legacy_sys->pending_events == 1,
                "D1: internal event pending");

    expect_true(legacy_sys->cancel_event(stale),
                "D1: Sys event cancellation succeeds");
    expect_true(cleanups == 1, "D1: payload reclaimed via cleanup callback");
    expect_true(legacy_sys->pending_events == 0,
                "D1: Sys bucket emptied by the cancellation");
    expect_true(legacy_queue->scheduled_event_count() == 1,
                "D1: legacy backend alarm stays resident (cannot cancel)");

    legacy_queue->proceed();  // fires Sys::handleEvent -> call_events
    expect_true(deliveries == 0,
                "D1: stale guard blocks the wrong callback (Sys alive)");
    expect_true(legacy_queue->finished(),
                "D1: stale alarm drained harmlessly");

    // D2: the same leftover-alarm drain after the owning Sys is deleted.
    auto retired = legacy_sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        60, reclaim_payload);
    expect_true(legacy_sys->cancel_event(retired),
                "D2: second event cancelled before retirement");
    expect_true(legacy_queue->scheduled_event_count() == 1,
                "D2: one more resident legacy alarm before retirement");

    delete legacy_sys;  // all_sys[1] = nullptr; the alarm payload is a
                        // BasicEventHandlerData that handleEvent itself owns
    legacy_queue->proceed();
    expect_true(deliveries == 0 && cleanups == 2,
                "D2: retired-Sys guard drops the alarm with no side effects");
    expect_true(legacy_queue->finished(),
                "D2: backend queue drained after Sys retirement");
}

// ===========================================================================
// Case E (P2-10 fix, 2026-09-28): same-tick sim_schedule from a tick-end
// (non-invoke) context routes through the same-tick deferred drain, and the
// cancellable variant (which must stay resident in the main map) takes the
// +1 ns strict-future lower bound. Pre-fix, a delta==0 schedule_event from
// the tick-end callback inserted a current_time EventList into the main map
// and the next proceed() tripped the Release-always strict-increase abort
// (latent: unreachable under the factory configuration). The risky sequence
// runs in a forked child: pre-fix the child fails/aborts; post-fix it
// completes with the deferred callback drained inside the same proceed().
// ===========================================================================

struct SameTickRecorder {
    int invoke_context_deliveries = 0;
    int deferred_drain_deliveries = 0;
};

void record_invoke_delivery(void* arg) {
    ++static_cast<SameTickRecorder*>(arg)->invoke_context_deliveries;
}

void record_deferred_delivery(void* arg) {
    ++static_cast<SameTickRecorder*>(arg)->deferred_drain_deliveries;
}

int cancellation_cleanups = 0;

void count_cancellation_cleanup(void*) { ++cancellation_cleanups; }

struct TickEndCtx {
    CommonNetworkApi* api = nullptr;
    SameTickRecorder* rec = nullptr;
    int visits = 0;
};

void same_tick_tick_end(void* arg) {
    auto* const ctx = static_cast<TickEndCtx*>(arg);
    ++ctx->visits;
    if (ctx->visits != 2) {
        return;  // visit 1 (t0+100): nothing to prove here
    }
    // t0+200 tick-end: the tick-end callback is a NON-invoke context, so a
    // delta==0 schedule must NOT touch the main map (EventQueue hard rule).
    ctx->api->sim_schedule(timespec_t{NS, 0}, record_deferred_delivery,
                           ctx->rec);
    // The cancellable variant stays map-resident: it takes the +1 ns lower
    // bound and stays cancellable from here.
    auto handle = ctx->api->sim_schedule_cancellable(
        timespec_t{NS, 0}, record_deferred_delivery, ctx->rec,
        &count_cancellation_cleanup);
    expect_true(handle.valid(),
                "E: same-tick cancellable from tick-end returns a handle");
    expect_true(ctx->api->sim_cancel_event(handle),
                "E: the +1ns same-tick cancellable cancels cleanly");
}

void noop_same_tick_callback(void*) {}

void test_same_tick_schedule_from_tick_end(AnalyticalFixture& fixture) {
    std::printf("[case E] same-tick sim_schedule from tick-end context\n");
    const pid_t pid = ::fork();
    if (pid < 0) {
        expect_true(false, "E: fork failed");
        return;
    }
    if (pid == 0) {
        if (std::freopen("/dev/null", "w", stderr) == nullptr) {
            // POSIX forked-child discipline: _exit skips this copy's
            // atexit/static destructors (the mkdtemp cleanup belongs to the
            // parent, which still lives in the shared config dir).
            ::_exit(1);
        }
        SameTickRecorder rec;
        TickEndCtx ctx;
        ctx.api = fixture.network_api.get();
        ctx.rec = &rec;
        fixture.event_queue->set_tick_end_callback(&same_tick_tick_end,
                                                   &ctx);

        // Cases A-C already advanced the shared queue's clock; all Case E
        // arithmetic is RELATIVE to the clock on entry.
        const EventTime t0 = fixture.event_queue->get_current_time();

        // Seed one tick ahead (plain future schedule from outside any
        // proceed).
        fixture.network_api->sim_schedule(timespec_t{NS, 100},
                                          &record_invoke_delivery, &rec);
        fixture.event_queue->proceed();  // t0+100: record_invoke fires
        fixture.network_api->sim_schedule(timespec_t{NS, 100},
                                          &noop_same_tick_callback, nullptr);
        fixture.event_queue->proceed();  // t0+200: tick-end visit 2 (risky)

        expect_true(rec.invoke_context_deliveries == 1,
                    "E: invoke-context delta==0 keeps the same-pass merge");
        expect_true(rec.deferred_drain_deliveries == 1,
                    "E: tick-end delta==0 routed via the deferred drain "
                    "and drained inside the same proceed()");
        expect_true(fixture.event_queue->get_current_time() == t0 + 200,
                    "E: time ordering intact (current time t0+200)");
        expect_true(fixture.event_queue->scheduled_event_count() == 0 &&
                        fixture.event_queue->finished(),
                    "E: no main-map residue (cancelled cancellable + "
                    "deferred insert never touched the map)");
        expect_true(!fixture.event_queue->has_deferred_work(),
                    "E: deferred queue fully drained");
        expect_true(cancellation_cleanups == 1,
                    "E: cancellation cleanup ran exactly once");
        // POSIX forked-child discipline: flush stdio streams, then raw
        // _exit -- std::exit would run this copy's atexit/static
        // destructors (spdlog etc.), which is exactly where a forked
        // child's exit status can go wrong without any test failure.
        std::fflush(nullptr);
        ::_exit(0);
    }
    int status = 0;
    const pid_t waited = ::waitpid(pid, &status, 0);
    char verdict[160];
    std::snprintf(verdict, sizeof(verdict),
                  "E: same-tick tick-end schedule survives (deferred "
                  "routing, no strict-increase abort) [waited=%d "
                  "exited=%d sig=%d code=%d]",
                  static_cast<int>(waited == pid), WIFEXITED(status),
                  WIFSIGNALED(status) ? WTERMSIG(status) : 0,
                  WIFEXITED(status) ? WEXITSTATUS(status) : status);
    expect_true(waited == pid && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                verdict);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string tmp_template = "/tmp/alarm_cancel_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    assert(made != nullptr);
    // P3 fix (2026-09-28): expect_true exits(1) on the first failure, which
    // used to skip the remove_all at the end of main and leak the mkdtemp
    // dir. Static lifetime + atexit give every exit path the same cleanup:
    // static storage outlives the atexit handler, and the forked Case E
    // child exits via POSIX _exit, so the handler runs in the parent only.
    static const std::string config_dir(made);
    std::atexit(
        []() {
            std::error_code ec;
            std::filesystem::remove_all(config_dir, ec);
        });
    write_text(config_dir + "/system.json", kSystemJson);
    write_text(config_dir + "/network.yml", kNetworkYaml);
    write_text(config_dir + "/comm_group.json", "{}");

    // ---- cases A-C on the real analytical cancellation chain ----
    AnalyticalFixture fixture = make_analytical_fixture(config_dir);
    expect_true(fixture.sys->pending_events == 0 &&
                    fixture.event_queue->scheduled_event_count() == 0,
                "fixture: fresh Sys leaves both queues empty");

    test_last_event_removal_is_physical(fixture);
    test_bucket_scoped_cascade(fixture);
    test_repeat_cancellation_is_idempotent(fixture);

    // ---- case D on a legacy/fake backend; the analytical Sys must retire
    // first because Sys::boostedTick() reads the clock of all_sys[0] ----
    delete fixture.sys;
    fixture.sys = nullptr;
    test_legacy_backend_fallback(config_dir);

    // ---- case E (P2-10): same-tick sim_schedule from a tick-end context
    // (network api + event queue only; no Sys needed) ----
    test_same_tick_schedule_from_tick_end(fixture);

    std::printf("ALL PASS\n");
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    return 0;
}
