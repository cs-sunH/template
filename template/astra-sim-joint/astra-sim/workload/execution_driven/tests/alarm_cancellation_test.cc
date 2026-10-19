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
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace Analytical;
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

// R4-NF1 port (2026-09-30, from the face repo): an expect_true failure
// leaves via std::exit, skipping the run-end cleanup below; the config dir
// is removed by an atexit handler instead, so a failing run leaves no /tmp
// residue.
static std::string g_config_dir_for_cleanup;

static void cleanup_config_dir() {
    if (!g_config_dir_for_cleanup.empty()) {
        std::error_code ec;
        std::filesystem::remove_all(g_config_dir_for_cleanup, ec);
        g_config_dir_for_cleanup.clear();
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
  "replay-only": 0,
  "track-local-mem": 0,
  "trace-enabled": 0,
  "hbm-bandwidth-contention": 0,
  "peak-perf": 261.12,
  "local-mem-bw": 1640.0,
  "local-mem-latency": 100,
  "remote-mem-bw": 1000.0,
  "remote-mem-latency": 100
}
)";

const char* kNetworkYaml = R"(topology: [ Line, Line ]
npus_count: [ 4, 1 ]
bandwidth: [ 400.0, 400.0 ]
latency: [ 5, 5 ]
)";

const char* kRemoteMemoryJson = R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "remote-mem-bw": 1000.0,
  "remote-mem-latency": 100,
  "npu-ids": [0, 1, 2, 3]
}
)";

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    // NDEBUG-proof guard (2026-10-01 audit): this test's discipline forbids
    // bare assert() -- Release builds strip it and the failure mode would
    // become a confusing downstream symptom.
    expect_true(out.is_open(), "fixture file opens for writing");
    out << content;
}

struct AnalyticalFixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<AnalyticalRemoteMemory> memory_api;
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

    fixture.memory_api = std::make_unique<AnalyticalRemoteMemory>(
        config_dir + "/remote_memory.json");

    fixture.network_api = std::make_unique<CongestionAwareNetworkApi>(0);
    auto graph_source = std::make_shared<NodeStoreGraphSource>();
    const auto npus_count_per_dim = network_parser.get_npus_counts_per_dim();
    std::vector<int> queues_per_dim;
    for (size_t i = 0; i < npus_count_per_dim.size(); ++i) {
        queues_per_dim.push_back(1);
    }
    fixture.sys = new Sys(0, config_dir + "/workload", config_dir +
                          "/comm_group.json", config_dir + "/system.json",
                          fixture.memory_api.get(), fixture.network_api.get(),
                          npus_count_per_dim, queues_per_dim, 1.0, 1.0,
                          false, ExecutionDriven::ExecutionMode::Online,
                          graph_source);
    return fixture;
}

// ---- [C1 step 3, 2026-10-17] FakeLegacyNetworkApi (the non-cancellable
// backend used by retired Case D) is deleted with the case: the legacy
// fallback it exercised is now a register-time fail-closed panic. ----

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
// Case B: per-event backend residency under the C1 direct mount; cascaded
// cancellation is per registration; surviving events keep their trigger tick.
// [C1 step 1, 2026-10-17] Cancellable registrations each own one backend
// QueuedEvent (no shared bucket alarm), so same-tick counts shift 1 -> N;
// the cascade contract becomes "each cancellation physically removes its own
// backend event".
// ===========================================================================

void test_bucket_scoped_cascade(AnalyticalFixture& fixture) {
    std::printf("[case B] per-event backend residency cascade\n");
    // B1: two cancellable events at one tick.  Each registration holds its
    // own backend event; removing one leaves the other physically resident.
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
    expect_true(fixture.event_queue->scheduled_event_count() == 2,
                "B1: direct mount -> one backend event per registration");

    expect_true(fixture.sys->cancel_event(first),
                "B1: first event cancels");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "B1: sibling backend event survives the cancellation");
    expect_true(fixture.sys->pending_events == 1,
                "B1: one internal event still pending");

    expect_true(fixture.sys->cancel_event(second),
                "B1: second (last) event cancels");
    expect_true(fixture.event_queue->scheduled_event_count() == 0,
                "B1: cancelling the last event empties the backend queue");
    expect_true(cleanups == 2, "B1: both payloads reclaimed exactly once");
    expect_true(fixture.event_queue->finished(),
                "B1: backend main queue empty after the cascade");

    // B2: mixed tick -- one cancellable (direct mount) plus one legacy
    // register_event occupant (bucket alarm channel).  The legacy alarm
    // must stay until it actually fires, and that survivor must keep its
    // original trigger tick.
    deliveries = 0;
    cleanups = 0;
    auto cancellable = fixture.sys->register_event_cancellable(
        &callable, EventType::General, make_payload(&deliveries, &cleanups),
        200, reclaim_payload);
    fixture.sys->register_event(&callable, EventType::General,
                                make_payload(&deliveries, &cleanups), 200);
    expect_true(fixture.sys->pending_events == 2,
                "B2: mixed tick holds two internal events");
    expect_true(fixture.event_queue->scheduled_event_count() == 2,
                "B2: one direct event plus one legacy bucket alarm");

    expect_true(fixture.sys->cancel_event(cancellable),
                "B2: cancellable occupant removed");
    expect_true(fixture.event_queue->scheduled_event_count() == 1,
                "B2: legacy occupant keeps its bucket alarm resident");
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
// [C1 step 3, 2026-10-17] Case D (legacy/fake backend fallback + stale
// guards) is RETIRED with the legacy fallback itself: a backend whose
// sim_schedule_cancellable cannot cancel now makes register_event_
// cancellable fail closed (sys_panic "backend does not support
// cancellable schedule") instead of silently posting an uncancellable
// alarm. The five Sys-level fake fixtures gained real cancellable
// delegation, so this file no longer needs a legacy-backend case.
// ===========================================================================

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    std::atexit(cleanup_config_dir);
    const std::string tmp_template = "/tmp/alarm_cancel_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    expect_true(made != nullptr, "fixture mkdtemp succeeds");
    const std::string config_dir(made);
    g_config_dir_for_cleanup = config_dir;
    write_text(config_dir + "/system.json", kSystemJson);
    write_text(config_dir + "/network.yml", kNetworkYaml);
    write_text(config_dir + "/remote_memory.json", kRemoteMemoryJson);
    write_text(config_dir + "/comm_group.json", "{}");

    // ---- cases A-C on the real analytical cancellation chain ----
    AnalyticalFixture fixture = make_analytical_fixture(config_dir);
    expect_true(fixture.sys->pending_events == 0 &&
                    fixture.event_queue->scheduled_event_count() == 0,
                "fixture: fresh Sys leaves both queues empty");

    test_last_event_removal_is_physical(fixture);
    test_bucket_scoped_cascade(fixture);
    test_repeat_cancellation_is_idempotent(fixture);
    // [C1 step 3, 2026-10-17] Case D retired with the legacy fallback; the
    // analytical fixture teardown below is all that remains.

    std::printf("ALL PASS\n");
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    return 0;
}
