/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
local_hbm_die_memory_test.cc -- LocalHbmBandwidthModel DIE_COMPUTE jobs
(joint rollout stage 1, contract §7.4; hermetic direct-model fixture,
ported from the face reference and adapted to this repository's Sys ctor
(remote-memory API parameter) and mandatory scheduling-policy key).

The legacy HbmNwayTest needs materialized fixture ET artifacts, so this
regression drives the model directly: a Sys in online mode over a
NodeStoreGraphSource (no .et files), local-mem-bw 6 GB/s /
local-mem-latency 100 ns, and a real analytical EventQueue behind a fake
backend. Two DIE_COMPUTE jobs plus one COMM_READ (created the real way, by
issuing a COMM_SEND node through Workload::issue, so the HBM join entry
exists) all start at tick 0:

  DIE A   service 30 units, actual 21 bytes (eta 0.7; NO latency phase)
  DIE B   service 60 units, actual 30 bytes (eta 0.5; NO latency phase)
  COMM_READ 120 bytes (eta 1; legacy 100 ns latency phase first)

Equal split of the 6 B/ns scalar among the bandwidth users:
  [0,10)    2 die users @ 3 B/ns: A drains 30 units -> sink A fires at 10
            (the comm job sits in its latency phase, not a bandwidth user)
  [10,15)   B alone @ 6 B/ns: 30 units left -> sink B fires at 15
  [100,120) the comm byte drain alone @ 6 B/ns -> hbm_done at 120 (the
            network side never arrives, so the send node never terminals)

Assertions: exact sink ticks (memory completions are NOT outer COMP
completions), equal-share service accounting, actual-vs-service kept in
separate columns (die_service_units_served == 90, die_actual_bytes_served
== 51, comm_read bytes untouched by the die accounting), malformed-request
guards (eta >= 1 / non-positive service) fail closed, and a second fixture
checks fractional DIE drain boundaries within one integer callback tick.

Build: cmake target AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect_true(const bool cond, const char* const what) {
    std::printf("%-74s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        g_ok = false;
    }
}

// Terminal collector: proves the DIE jobs never masquerade as outer COMP
// completions (only the send node could ever terminal here, and it cannot
// -- its network side never fires under the fake backend).
std::map<uint64_t, uint64_t> g_terminals;

void collecting_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
                     const char* /*request_id*/, const char* /*stage*/,
                     uint64_t /*generation*/, uint64_t /*tick*/,
                     int /*terminal_status*/) {
    ++g_terminals[node_id];
}

// No-op remote-memory API: this repository's Sys ctor requires a
// non-null AstraRemoteMemoryAPI (it calls set_sys on it); the die memory
// fixture never issues remote transactions.
class TestRemoteMemoryApi final : public AstraSim::AstraRemoteMemoryAPI {
  public:
    void set_sys(int, AstraSim::Sys*) override {}
    void issue(uint64_t, AstraSim::WorkloadLayerHandlerData*) override {}
};

class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;  // never completes the network side of the send node
    }

    int sim_recv(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
    }

    void sim_schedule(const timespec_t delta, void (*fun_ptr)(void*),
                      void* fun_arg) override {
        const timespec_t now = sim_get_time();
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

struct SinkData final : public CallData {
    explicit SinkData(int id_in) : id(id_in) {}
    int id = 0;
};

class CountingSink final : public Callable {
  public:
    LocalHbmBandwidthModel* model = nullptr;
    std::map<int, uint64_t> fire_ticks;
    std::map<int, uint64_t> fires;
    std::map<int, uint64_t> fire_versions;
    std::vector<int> fire_order;

    void call(EventType, CallData* data) override {
        auto* payload = static_cast<SinkData*>(data);
        fire_order.push_back(payload->id);
        if (fires.find(payload->id) == fires.end()) {
            fire_ticks[payload->id] = Sys::boostedTick();
        }
        fire_versions[payload->id] =
            model == nullptr ? 0 : model->resource_version();
        ++fires[payload->id];
        delete payload;
    }
};

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CompletionObserver::instance().set_hook(collecting_hook, nullptr);

    const std::string tmp_template = "/tmp/die_hbm_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr, "[local_hbm_die_memory_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    {
        std::ofstream out(config_dir + "/system.json");
        out << R"({
  "scheduling-policy": "FIFO",
  "hbm-bandwidth-contention": 1,
  "peak-perf": 1,
  "local-mem-bw": 6,
  "local-mem-latency": 100
})";
    }
    {
        std::ofstream out(config_dir + "/comm_group.json");
        out << "{}";
    }

    auto event_queue = std::make_shared<EventQueue>();
    FakeQueueNetworkApi api(event_queue);
    TestRemoteMemoryApi remote_memory;
    auto source = std::make_shared<NodeStoreGraphSource>();
    auto sys = std::make_unique<Sys>(
        0, "unused", config_dir + "/comm_group.json",
        config_dir + "/system.json", &remote_memory, &api,
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
    Workload& workload = *sys->workload;
    LocalHbmBandwidthModel* const model =
        workload.local_hbm_bandwidth_model.get();
    if (model == nullptr) {
        std::fprintf(stderr,
                     "[local_hbm_die_memory_test] FAIL: no HBM model\n");
        return 1;
    }

    std::vector<std::pair<bool, std::pair<Tick, uint64_t>>> mutations;
    std::map<Tick, uint64_t> published_version_at_tick;
    const auto observer_id = model->add_resource_mutation_observer(
        [&mutations, &published_version_at_tick](const bool before,
                                                const Tick tick,
                                                const uint64_t version) {
            mutations.push_back({before, {tick, version}});
            if (!before) published_version_at_tick[tick] = version;
        });
    expect_true(model->resource_version() == 0,
                "HBM resource version starts at zero for an unchanged rank");

    CountingSink sink;
    sink.model = model;
    // Two direct DIE jobs at tick 0 (service > actual: eta < 1).
    model->issue_die_memory(30.0, 21.0, &sink, new SinkData(1));
    model->issue_die_memory(60.0, 30.0, &sink, new SinkData(2));

    // One COMM_READ created the real way: issue a COMM_SEND node through
    // Workload::issue so its HBM join entry exists (bytes 120, src rank 0).
    OnlineNode send_node;
    send_node.global_id = 31;
    send_node.rank = 0;
    send_node.kind = NodeKind::CommSend;
    send_node.node_type = 5;  // ChakraProtoMsg::COMM_SEND_NODE
    send_node.name = "send";
    send_node.request_id = "r1";
    send_node.stage = "prefill";
    send_node.generation = 0;
    send_node.comm.bytes = 120;
    send_node.comm.src = 0;
    send_node.comm.dst = 0;
    send_node.comm.tag = 0;
    send_node.comm.hbm_charge = true;
    source->store().add_node(send_node);
    workload.issue_dep_free_nodes();

    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    // ---- Completion ticks (exact equal-share arithmetic) --------------
    expect_true(sink.fires[1] == 1 && sink.fire_ticks[1] == 10,
                "DIE A completes at 10 (2 die users @ 3 B/ns, 30 units; the "
                "comm job is still in its latency phase)");
    expect_true(sink.fires[2] == 1 && sink.fire_ticks[2] == 15,
                "DIE B completes at 15 (alone @ 6 B/ns, 30 units left)");
    expect_true(sink.fires.size() == 2,
                "exactly two sink fires total (one per die job)");
    expect_true(sink.fire_versions[1] ==
                        published_version_at_tick[sink.fire_ticks[1]] &&
                    sink.fire_versions[2] ==
                        published_version_at_tick[sink.fire_ticks[2]],
                "HBM membership versions publish before DIE completion sinks");

    // ---- The send node never terminals (network side never arrived) ----
    expect_true(g_terminals.count(31) == 0,
                "internal die/comm HBM jobs never complete the outer node "
                "alone (send node has no terminal)");

    // ---- Side-band: service vs actual kept separate --------------------
    expect_true(std::abs(model->die_service_units_served() - 90.0) < 1e-6,
                "die_service_units_served == 90 (drained quantity)");
    expect_true(std::abs(model->die_actual_bytes_served() - 51.0) < 1e-6,
                "die_actual_bytes_served == 51 (21 + 30, service * eta)");
    expect_true(std::abs(model->comm_read_bytes_served() - 120.0) < 1e-6,
                "comm_read_bytes_served == 120 (untouched by die units)");
    expect_true(std::abs(model->compute_bytes_served() - 0.0) < 1e-6,
                "no COMP roofline bytes mixed into the die accounting");
    expect_true(std::abs(model->hbm_busy_ns() - 35.0) < 1e-6,
                "hbm_busy_ns == 35 (15 ns die drain + 20 ns comm drain)");
    expect_true(model->peak_concurrent_jobs() == 3,
                "peak concurrency 3 (2 die jobs + 1 comm read)");
    expect_true(model->redistribution_events() == 4,
                "redistribution events 4 (2 joins + 2 survivor splits)");
    bool mutation_order_valid = !mutations.empty() && mutations.size() % 2 == 0;
    uint64_t expected_version = 0;
    for (size_t i = 0; i + 1 < mutations.size(); i += 2) {
        const auto& before = mutations[i];
        const auto& after = mutations[i + 1];
        mutation_order_valid = mutation_order_valid && before.first &&
            !after.first && before.second.first == after.second.first &&
            before.second.second == expected_version &&
            after.second.second == expected_version + 1;
        expected_version = after.second.second;
    }
    expect_true(mutation_order_valid && model->resource_version() == expected_version,
                "each issue/share transition publishes before its paired after-version");
    model->remove_resource_mutation_observer(observer_id);

    // ---- Malformed-request guards (fail closed) -------------------------
    bool threw_eta = false;
    try {
        model->issue_die_memory(10.0, 10.0, &sink, new SinkData(3));
    } catch (const std::runtime_error&) {
        threw_eta = true;
    }
    expect_true(threw_eta, "eta >= 1 request (service <= actual) refused");

    bool threw_zero = false;
    try {
        model->issue_die_memory(0.0, 0.0, &sink, new SinkData(4));
    } catch (const std::runtime_error&) {
        threw_zero = true;
    }
    expect_true(threw_zero, "zero-service request refused");

    // One real HBM model callback at the next integer tick internally
    // crosses two fractional drain boundaries (start+10.3 and start+10.9).
    // The legacy transfer's integer 10ns latency ends at start+10. Its bytes
    // must not be served in the pre-latency step that reaches that boundary;
    // after it, the transfer joins the same fluid stream and receives only
    // its equal share.
    // Keep this fixture on the main EventQueue because Sys::boostedTick()
    // reads rank 0's clock, and the earlier secondary-queue fixture has
    // already finished.
    {
        const uint64_t saved_latency = sys->local_mem_latency;
        sys->local_mem_latency = 10;
        const Tick start_tick = event_queue->get_current_time();
        const uint64_t version_before = model->resource_version();
        const double service_before = model->die_service_units_served();
        const double actual_before = model->die_actual_bytes_served();
        const double comm_before = model->comm_read_bytes_served();
        const double busy_before = model->hbm_busy_ns();
        const double shared_before = model->hbm_shared_ns();

        struct MutationSample {
            bool before = false;
            Tick tick = 0;
            uint64_t version = 0;
        };
        std::vector<MutationSample> samples;
        const auto fractional_observer = model->add_resource_mutation_observer(
            [&samples](const bool before, const Tick tick,
                       const uint64_t version) {
                samples.push_back(MutationSample{before, tick, version});
            });
        CountingSink fractional_sink;
        fractional_sink.model = model;
        auto legacy_fires = std::make_shared<uint64_t>(0);

        model->issue_die_memory(30.6, 21.42, &fractional_sink,
                                new SinkData(5));
        model->issue_die_memory(32.4, 16.2, &fractional_sink,
                                new SinkData(6));
        model->issue_transfer(LocalHbmBandwidthModel::JobKind::COMM_READ,
            100, 0, 900, [legacy_fires] { ++*legacy_fires; });

        // The first live transition is the legacy latency boundary at +10.
        // FakeQueueNetworkApi cannot remove an already-posted outer alarm
        // when Sys cancels its inner transition entry, so drain stale alarms
        // until this model publishes the latency-boundary membership change.
        while (model->resource_version() != version_before + 4 &&
               !event_queue->finished()) {
            event_queue->proceed();
        }
        expect_true(event_queue->get_current_time() == start_tick + 10,
                    "fractional fixture first callback is the 10ns latency boundary");
        expect_true(fractional_sink.fire_order.empty() && *legacy_fires == 0,
                    "latency-ending callback completes neither DIE nor COMM_READ job");
        expect_true(std::abs(model->die_service_units_served() - service_before -
                             60.0) < 1e-8 &&
                    std::abs(model->comm_read_bytes_served() - comm_before) < 1e-8,
                    "pre-latency interval serves only the two captured DIE users");
        expect_true(model->resource_version() == version_before + 4,
                    "latency expiry publishes its membership version before later drain boundaries");

        // The next EventQueue event is tick +11. During its single
        // advance_to(), A drains at +10.3, then B drains at +10.9; the legacy
        // job joins at +10 and receives the captured equal share from there.
        while (fractional_sink.fire_order.size() < 2 &&
               !event_queue->finished()) {
            event_queue->proceed();
        }
        expect_true(event_queue->get_current_time() == start_tick + 11,
                    "both fractional DIE boundaries are integrated by the next integer tick");
        expect_true(fractional_sink.fire_order == std::vector<int>{5, 6} &&
                    fractional_sink.fires[5] == 1 &&
                    fractional_sink.fires[6] == 1 &&
                    fractional_sink.fire_ticks[5] == start_tick + 11 &&
                    fractional_sink.fire_ticks[6] == start_tick + 11,
                    "different-service DIE jobs complete in insertion order at tick +11");
        expect_true(fractional_sink.fire_versions[5] == version_before + 6 &&
                    fractional_sink.fire_versions[6] == version_before + 6,
                    "DIE sinks observe both membership and completion versions already published");
        expect_true(*legacy_fires == 0 && model->has_active_jobs(),
                    "latency transfer remains active after the DIE completion batch");

        const auto fractional_snapshot = model->job_snapshot();
        expect_true(fractional_snapshot.size() == 1 &&
                    fractional_snapshot[0].kind == static_cast<int>(
                        LocalHbmBandwidthModel::JobKind::COMM_READ) &&
                    std::abs(fractional_snapshot[0].remaining_bytes - 97.0) < 1e-8 &&
                    fractional_snapshot[0].memory_latency_ns == 0.0,
                    "legacy transfer receives exactly 3B after its 10ns latency");
        expect_true(std::abs(model->die_service_units_served() - service_before -
                             63.0) < 1e-8 &&
                    std::abs(model->die_actual_bytes_served() - actual_before -
                             37.62) < 1e-8 &&
                    std::abs(model->comm_read_bytes_served() - comm_before -
                             3.0) < 1e-8,
                    "fractional boundaries preserve exact DIE service, actual bytes, and legacy bytes");
        expect_true(std::abs(model->hbm_busy_ns() - busy_before - 11.0) < 1e-8 &&
                    std::abs(model->hbm_shared_ns() - shared_before - 10.9) < 1e-8,
                    "fractional boundary integration preserves busy/shared durations");

        bool exact_mutations = samples.size() == 12;
        for (size_t pair = 0; pair < 6 && exact_mutations; ++pair) {
            const Tick expected_tick = pair < 3
                ? start_tick
                : (pair == 3 ? start_tick + 10 : start_tick + 11);
            const uint64_t expected_version = version_before + pair;
            const MutationSample& before = samples[2 * pair];
            const MutationSample& after = samples[2 * pair + 1];
            exact_mutations = before.before && !after.before &&
                before.tick == expected_tick && after.tick == expected_tick &&
                before.version == expected_version &&
                after.version == expected_version + 1;
        }
        expect_true(exact_mutations &&
                    model->resource_version() == version_before + 6,
                    "fixture publishes three issue, two membership, and one completion transition");

        model->remove_resource_mutation_observer(fractional_observer);
        sys->local_mem_latency = saved_latency;

    // ---- cancel_die_jobs_of: teardown drops a pending job silently -----
    {
        auto event_queue2 = std::make_shared<EventQueue>();
        FakeQueueNetworkApi api2(event_queue2);
        TestRemoteMemoryApi remote_memory2;
        auto source2 = std::make_shared<NodeStoreGraphSource>();
        auto sys2 = std::make_unique<Sys>(
            0, "unused", config_dir + "/comm_group.json",
            config_dir + "/system.json", &remote_memory2, &api2,
            std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
            ExecutionMode::Online, source2);
        LocalHbmBandwidthModel* const model2 =
            sys2->workload->local_hbm_bandwidth_model.get();
        CountingSink sink2;
        model2->issue_die_memory(30.0, 21.0, &sink2, new SinkData(9));
        model2->cancel_die_jobs_of(&sink2);
        expect_true(!model2->has_active_jobs(),
                    "cancel_die_jobs_of drops the pending job");
        while (!event_queue2->finished()) {
            event_queue2->proceed();
        }
        expect_true(sink2.fires.empty(),
                    "cancelled job never fires its sink");
        expect_true(std::abs(model2->die_service_units_served()) < 1e-9,
                    "cancelled job served nothing");

        // Pipeline endpoint HBM completion uses the same COMM_READ/WRITE
        // latency and shared-bus accounting, but delivers a typed closure
        // rather than routing through a physical Workload node lookup.
        std::map<int, uint64_t> transfer_fires;
        std::map<int, uint64_t> transfer_ticks;
        const uint64_t transfer_start = event_queue2->get_current_time();
        model2->issue_transfer(LocalHbmBandwidthModel::JobKind::COMM_READ,
            60, 0, 0, [&] {
                ++transfer_fires[0];
                transfer_ticks[0] = event_queue2->get_current_time();
            });
        model2->issue_transfer(LocalHbmBandwidthModel::JobKind::COMM_WRITE,
            60, 0, 77, [&] {
                ++transfer_fires[1];
                transfer_ticks[1] = event_queue2->get_current_time();
            });
        const auto transfer_snapshot = model2->job_snapshot();
        expect_true(transfer_snapshot.size() == 2 &&
                    transfer_snapshot[0].sys_id == 0 &&
                    transfer_snapshot[0].node_id == 0 &&
                    transfer_snapshot[0].kind == static_cast<int>(
                        LocalHbmBandwidthModel::JobKind::COMM_READ) &&
                    transfer_snapshot[0].remaining_bytes == 60.0 &&
                    transfer_snapshot[1].node_id == 77 &&
                    transfer_snapshot[1].kind == static_cast<int>(
                        LocalHbmBandwidthModel::JobKind::COMM_WRITE),
                    "transfer HBM snapshot preserves rank, mapped ID, kind, and remainder");
        while (!event_queue2->finished()) {
            event_queue2->proceed();
        }
        expect_true(transfer_fires[0] == 1 && transfer_fires[1] == 1 &&
                    transfer_ticks[0] == transfer_start + 120 &&
                    transfer_ticks[1] == transfer_start + 120,
                    "two transfer closures complete exactly once after shared 100ns latency and 60B/N drain");
        expect_true(std::abs(model2->comm_read_bytes_served() - 60.0) < 1e-6 &&
                    std::abs(model2->comm_write_bytes_served() - 60.0) < 1e-6,
                    "transfer jobs use the legacy COMM_READ/COMM_WRITE accounting");
    }

    }

    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    if (!g_ok) {
        std::fprintf(stderr,
                     "[local_hbm_die_memory_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[local_hbm_die_memory_test] ALL PASS\n");
    return 0;
}
