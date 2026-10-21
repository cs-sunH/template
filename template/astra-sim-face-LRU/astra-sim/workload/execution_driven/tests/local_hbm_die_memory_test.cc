/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
local_hbm_die_memory_test.cc -- LocalHbmBandwidthModel DIE_COMPUTE jobs
(stage B, contract §7.4; hermetic direct-model fixture).

The legacy local_hbm_bandwidth_model_test.cc needs materialized fixture ET
artifacts (make_local_hbm_fixture_et.py), so this regression drives the
model directly: a Sys in online mode over a NodeStoreGraphSource (no .et
files), local-mem-bw 6 GB/s / local-mem-latency 100 ns, and a real
analytical EventQueue behind a fake backend. Two DIE_COMPUTE jobs plus one
COMM_READ (created the real way, by issuing a COMM_SEND node through
Workload::issue, so the HBM join entry exists) all start at tick 0:

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
== 51, comm_read bytes untouched by the die accounting), and the
malformed-request guards (eta >= 1 / non-positive service) fail closed.

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
    std::map<int, uint64_t> fire_ticks;
    std::map<int, uint64_t> fires;

    void call(EventType, CallData* data) override {
        auto* payload = static_cast<SinkData*>(data);
        if (fires.find(payload->id) == fires.end()) {
            fire_ticks[payload->id] = Sys::boostedTick();
        }
        ++fires[payload->id];
        delete payload;
    }
};

}  // namespace

// face-LRU adaptation: this repository's Sys constructor requires a non-null
// remote memory backend (AstraRemoteMemoryAPI). The die fixtures never issue
// remote MEM nodes, so a minimal no-op backend suffices (an issue call would
// be a fixture defect and fails closed).
class FakeRemoteMemoryApi final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int, Sys*) override {}
    void issue(uint64_t, WorkloadLayerHandlerData*) override {
        std::fprintf(stderr,
                     "[local_hbm_die_memory_test] FakeRemoteMemoryApi::issue "
                     "called -- the fixture never issues remote MEM nodes\n");
        std::exit(EXIT_FAILURE);
    }
};

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CompletionObserver::instance().set_hook(collecting_hook, nullptr);

    const std::string tmp_template = "/tmp/die_hbm_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr,
                     "[local_hbm_die_memory_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    {
        std::ofstream out(config_dir + "/system.json");
        // face-LRU adaptation: this repo's LocalHbmBandwidthModel constructor
        // also requires a positive peak-perf (the legacy COMPUTE job drains
        // FLOPs at that rate). The value never enters the DIE/COMM arithmetic
        // below (no COMPUTE job is issued), so the tick expectations are
        // unchanged.
        out << R"({
  "hbm-bandwidth-contention": 1,
  "local-mem-bw": 6,
  "local-mem-latency": 100,
  "peak-perf": 1.0
})";
    }
    {
        std::ofstream out(config_dir + "/comm_group.json");
        out << "{}";
    }

    auto event_queue = std::make_shared<EventQueue>();
    FakeQueueNetworkApi api(event_queue);
    FakeRemoteMemoryApi remote_api;
    auto source = std::make_shared<NodeStoreGraphSource>();
    auto sys = std::make_unique<Sys>(
        0, "unused", config_dir + "/comm_group.json",
        config_dir + "/system.json", &remote_api, &api, std::vector<int>{1},
        std::vector<int>{1}, 1.0, 1.0, false, ExecutionMode::Online, source);
    Workload& workload = *sys->workload;
    LocalHbmBandwidthModel* const model =
        workload.local_hbm_bandwidth_model.get();
    if (model == nullptr) {
        std::fprintf(stderr,
                     "[local_hbm_die_memory_test] FAIL: no HBM model\n");
        return 1;
    }

    CountingSink sink;
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

    // ---- cancel_die_jobs_of: teardown drops a pending job silently -----
    {
        auto event_queue2 = std::make_shared<EventQueue>();
        FakeQueueNetworkApi api2(event_queue2);
        FakeRemoteMemoryApi remote_api2;
        auto source2 = std::make_shared<NodeStoreGraphSource>();
        auto sys2 = std::make_unique<Sys>(
            0, "unused", config_dir + "/comm_group.json",
            config_dir + "/system.json", &remote_api2, &api2,
            std::vector<int>{1},
            std::vector<int>{1}, 1.0, 1.0, false, ExecutionMode::Online,
            source2);
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
