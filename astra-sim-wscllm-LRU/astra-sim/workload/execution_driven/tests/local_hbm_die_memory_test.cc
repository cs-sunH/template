/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
local_hbm_die_memory_test.cc -- LocalHbmBandwidthModel DIE_COMPUTE jobs
(stage B, contract §7.4; hermetic direct-model fixture for this LRU
repository).

Fixture: a Sys in online mode over a NodeStoreGraphSource (no .et files),
local-mem-bw 6 GB/s / local-mem-latency 100 ns, hbm-kv-restore-bandwidth-
sharing on (the LRU restore-DMA path), and a real analytical EventQueue
behind a fake backend. The stub AstraRemoteMemoryAPI keeps the pool nodes'
port legs pending forever, so a pool/comm node never terminals on its HBM
leg alone (exactly the join semantics under test).

Scenario 1 (die + die + comm_read, all at tick 0):
  DIE A   service 30 units, actual 21 bytes (eta 0.7; NO latency phase)
  DIE B   service 60 units, actual 30 bytes (eta 0.5; NO latency phase)
  COMM_READ 120 bytes (eta 1; legacy 100 ns latency phase first)
Equal split of the 6 B/ns scalar among the bandwidth users:
  [0,10)    2 die users @ 3 B/ns: A drains 30 units -> sink A fires at 10
  [10,15)   B alone @ 6 B/ns: 30 units left -> sink B fires at 15
  [100,120) comm byte drain alone @ 6 B/ns -> hbm_done at 120 (the network
            side never arrives, so the send node never terminals)

Scenario 2 (this repository's N-way: die + RESTORE + POOL_READ +
COMM_WRITE, all created the REAL way through Workload::issue so the
endpoint join entries exist, all at tick 0):
  DIE       service 30 units, actual 21 bytes (no latency phase)
  RESTORE   24 bytes (latency 100; issued by a local_hbm_kv_restore node)
  POOL_READ 36 bytes (latency 100; issued by an hbm_access_mode=1 mem node)
  COMM_WRITE 12 bytes (latency 100; issued by a COMM_RECV node)
  [0,5)     die alone @ 6: sink at 5
  [100,106) 3 legacy users @ 2: comm_write done at 106 (join leg only)
  [106,110) restore + pool @ 3: restore done at 110 (whole-node terminal)
  [110,112) pool alone @ 6: hbm leg done at 112 (port leg pending)
Asserts the die job competes in the same equal-split arbiter as the
RESTORE/POOL/comm kinds WITHOUT changing their byte accounting (per-kind
byte columns untouched, die service/actual in separate columns), and the
restore node's whole-node terminal tick.

Scenario 3 (full-runtime contention, die backend): a REAL die COMP node
(issued through Workload::issue -> issue_die_comp -> DieModelAdapter ->
DieExecution) races a REAL restore node (issue_local_hbm_kv_restore) under
compute-cost-model=die with the default die point. The restore (100 MB)
streams for ~64 us after its latency phase while the die node's first
memory request arrives at ~1.1 us, so the two overlap in the N-way arbiter
by construction; asserts single terminals, span >= isolated estimate +
latency (equal share can only delay), byte conservation of the restore
column, die service units > 0, and hbm_shared_ns > 0 (overlap proven).

Also: malformed-request guards (eta >= 1 / non-positive service) fail
closed; cancel_die_jobs_of drops a pending job silently.

Build: cmake target AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "DieModel.hh"
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
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
// completions (only nodes whose every join leg fired could terminal here,
// and the fake backend never completes a network/port leg).
std::map<uint64_t, uint64_t> g_terminals;
std::map<uint64_t, uint64_t> g_first_tick;

void collecting_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
                     const char* /*request_id*/, const char* /*stage*/,
                     uint64_t /*generation*/, uint64_t tick,
                     int /*terminal_status*/) {
    if (g_first_tick.find(node_id) == g_first_tick.end()) {
        g_first_tick[node_id] = tick;
    }
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
        return 0;  // never completes the network side of the recv node
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

// Stub remote memory: the pool nodes' port legs never complete (the join
// latch keeps them pending, which is the point of the scenario).
class StubRemoteMemory final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int /*id*/, Sys* /*sys*/) override {}
    void issue(uint64_t /*tensor_size*/,
               WorkloadLayerHandlerData* /*wlhd*/) override {}
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

struct Fixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    StubRemoteMemory remote_mem;
    std::shared_ptr<NodeStoreGraphSource> source =
        std::make_shared<NodeStoreGraphSource>();
    std::unique_ptr<Sys> sys;

    Fixture(const std::string& config_dir) {
        event_queue = std::make_shared<EventQueue>();
        network_api = std::make_unique<FakeQueueNetworkApi>(event_queue);
        sys = std::make_unique<Sys>(
            0, "unused", config_dir + "/comm_group.json",
            config_dir + "/system.json", &remote_mem, network_api.get(),
            std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
            ExecutionMode::Online, source);
    }

    Workload* workload() const {
        return sys->workload;
    }

    Sys* sys_ptr() const {
        return sys.get();
    }

    // Build a node in the store and hand the read view to Workload::issue
    // (the REAL production path: endpoint joins / restore dispatch are
    // created exactly as in a production run).
    uint64_t issue_node(OnlineNode node) {
        node.rank = 0;
        node.global_id = 0;  // store assigns a fresh id
        const uint64_t id = source->store().add_node(std::move(node));
        const auto* view = source->lookup_ptr(id);
        workload()->issue(*view);
        return id;
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
        // Same official-template fixture shape as the sibling
        // local_hbm_bandwidth_model_test: roofline-enabled + peak-perf keep
        // the LocalHbmBandwidthModel constructor's positive-divisor guard
        // happy (this fixture drives the model directly; no COMP node ever
        // reaches the legacy roofline path).
        out << R"({
  "roofline-enabled": 1,
  "peak-perf": 1000,
  "hbm-bandwidth-contention": 1,
  "hbm-kv-restore-bandwidth-sharing": 1,
  "local-mem-bw": 6,
  "local-mem-latency": 100
})";
    }
    {
        std::ofstream out(config_dir + "/comm_group.json");
        out << "{}";
    }

    // ------------------------------------------------------------------
    // Scenario 1: two direct DIE jobs + one real COMM_READ at tick 0.
    // ------------------------------------------------------------------
    uint64_t send_node_id = 0;
    {
        Fixture fixture(config_dir);
        Workload& workload = *fixture.workload();
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
        // Workload::issue so its HBM join entry exists (bytes 120, src 0).
        OnlineNode send_node;
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
        send_node_id = fixture.issue_node(std::move(send_node));

        while (!fixture.event_queue->finished()) {
            fixture.event_queue->proceed();
        }

        // ---- Completion ticks (exact equal-share arithmetic) ------------
        expect_true(sink.fires[1] == 1 && sink.fire_ticks[1] == 10,
                    "S1: DIE A completes at 10 (2 die users @ 3 B/ns, 30 "
                    "units; the comm job is still in its latency phase)");
        expect_true(sink.fires[2] == 1 && sink.fire_ticks[2] == 15,
                    "S1: DIE B completes at 15 (alone @ 6 B/ns, 30 units "
                    "left)");
        expect_true(sink.fires.size() == 2,
                    "S1: exactly two sink fires total (one per die job)");

        // ---- The send node never terminals (network side never arrived) --
        expect_true(g_terminals.count(send_node_id) == 0,
                    "S1: internal die/comm HBM jobs never complete the outer "
                    "node alone (send node has no terminal)");

        // ---- Side-band: service vs actual kept separate -----------------
        expect_true(std::abs(model->die_service_units_served() - 90.0) < 1e-6,
                    "S1: die_service_units_served == 90 (drained quantity)");
        expect_true(std::abs(model->die_actual_bytes_served() - 51.0) < 1e-6,
                    "S1: die_actual_bytes_served == 51 (21 + 30, service * "
                    "eta)");
        expect_true(std::abs(model->comm_read_bytes_served() - 120.0) < 1e-6,
                    "S1: comm_read_bytes_served == 120 (untouched by die "
                    "units)");
        expect_true(std::abs(model->restore_bytes_served() - 0.0) < 1e-9 &&
                        std::abs(model->pool_read_bytes_served() - 0.0) <
                            1e-9 &&
                        std::abs(model->pool_write_bytes_served() - 0.0) <
                            1e-9,
                    "S1: restore/pool columns untouched (no such jobs)");
        expect_true(std::abs(model->compute_bytes_served() - 0.0) < 1e-6,
                    "S1: no COMP roofline bytes mixed into the die "
                    "accounting");
        expect_true(std::abs(model->hbm_busy_ns() - 35.0) < 1e-6,
                    "S1: hbm_busy_ns == 35 (15 ns die drain + 20 ns comm "
                    "drain)");
        expect_true(model->peak_concurrent_jobs() == 3,
                    "S1: peak concurrency 3 (2 die jobs + 1 comm read)");

        // ---- Malformed-request guards (fail closed) ---------------------
        bool threw_eta = false;
        try {
            model->issue_die_memory(10.0, 10.0, &sink, new SinkData(3));
        } catch (const std::runtime_error&) {
            threw_eta = true;
        }
        expect_true(threw_eta,
                    "S1: eta >= 1 request (service <= actual) refused");

        bool threw_zero = false;
        try {
            model->issue_die_memory(0.0, 0.0, &sink, new SinkData(4));
        } catch (const std::runtime_error&) {
            threw_zero = true;
        }
        expect_true(threw_zero, "S1: zero-service request refused");
    }

    // ------------------------------------------------------------------
    // Scenario 2: DIE + RESTORE + POOL_READ + COMM_WRITE (the LRU N-way).
    // ------------------------------------------------------------------
    {
        g_terminals.clear();
        Fixture fixture(config_dir);
        Workload& workload = *fixture.workload();
        LocalHbmBandwidthModel* const model =
            workload.local_hbm_bandwidth_model.get();

        CountingSink sink;
        // The die job first (direct model entry, no latency phase).
        model->issue_die_memory(30.0, 21.0, &sink, new SinkData(11));

        // A local-HBM KV restore node (MEM_LOAD + is_local_hbm_kv_restore,
        // 24 bytes) -- the real issue path under
        // hbm-kv-restore-bandwidth-sharing.
        OnlineNode restore_node;
        restore_node.kind = NodeKind::MemLoad;
        restore_node.node_type = 2;  // ChakraProtoMsg::MEM_LOAD_NODE
        restore_node.name = "kv_restore";
        restore_node.request_id = "r1";
        restore_node.stage = "prefill";
        restore_node.generation = 0;
        restore_node.is_local_hbm_kv_restore = true;
        restore_node.compute.tensor_size = 24;
        const uint64_t restore_id = fixture.issue_node(std::move(restore_node));

        // A pool-read endpoint (MEM_STORE + hbm_access_mode=1, 36 bytes).
        OnlineNode pool_node;
        pool_node.kind = NodeKind::MemStore;
        pool_node.node_type = 3;  // ChakraProtoMsg::MEM_STORE_NODE
        pool_node.name = "pool_read";
        pool_node.request_id = "r1";
        pool_node.stage = "prefill";
        pool_node.generation = 0;
        pool_node.hbm_access_mode = 1;
        pool_node.compute.tensor_size = 36;
        const uint64_t pool_id = fixture.issue_node(std::move(pool_node));

        // A comm-write endpoint (COMM_RECV, 12 bytes).
        OnlineNode recv_node;
        recv_node.kind = NodeKind::CommRecv;
        recv_node.node_type = 6;  // ChakraProtoMsg::COMM_RECV_NODE
        recv_node.name = "recv";
        recv_node.request_id = "r1";
        recv_node.stage = "prefill";
        recv_node.generation = 0;
        recv_node.comm.bytes = 12;
        recv_node.comm.src = 0;
        recv_node.comm.dst = 0;
        recv_node.comm.tag = 0;
        recv_node.comm.hbm_charge = true;
        const uint64_t recv_id = fixture.issue_node(std::move(recv_node));

        while (!fixture.event_queue->finished()) {
            fixture.event_queue->proceed();
        }

        // ---- Completion ticks -------------------------------------------
        expect_true(sink.fires[11] == 1 && sink.fire_ticks[11] == 5,
                    "S2: die sink fires at 5 (alone @ 6 B/ns; the three "
                    "legacy jobs are in their latency phase)");
        expect_true(g_terminals.count(restore_id) == 1,
                    "S2: restore node reaches its single-leg terminal "
                    "exactly once");
        expect_true(g_terminals.count(pool_id) == 0,
                    "S2: pool node never terminals (port leg pending; the "
                    "HBM leg alone must not complete the node)");
        expect_true(g_terminals.count(recv_id) == 0,
                    "S2: recv node never terminals (network leg pending)");

        // ---- Side-band columns: every kind keeps its own accounting -----
        expect_true(std::abs(model->die_service_units_served() - 30.0) <
                        1e-6,
                    "S2: die_service_units_served == 30");
        expect_true(std::abs(model->die_actual_bytes_served() - 21.0) < 1e-6,
                    "S2: die_actual_bytes_served == 21 (service * eta)");
        expect_true(std::abs(model->restore_bytes_served() - 24.0) < 1e-6,
                    "S2: restore_bytes_served == 24 (real bytes, no die "
                    "pollution)");
        expect_true(std::abs(model->pool_read_bytes_served() - 36.0) < 1e-6,
                    "S2: pool_read_bytes_served == 36 (real bytes)");
        expect_true(std::abs(model->comm_write_bytes_served() - 12.0) < 1e-6,
                    "S2: comm_write_bytes_served == 12 (real bytes)");
        expect_true(std::abs(model->comm_read_bytes_served() - 0.0) < 1e-9,
                    "S2: comm_read column untouched in this scenario");
        expect_true(std::abs(model->hbm_busy_ns() - 17.0) < 1e-6,
                    "S2: hbm_busy_ns == 17 (5 + 6 + 4 + 2)");
        expect_true(model->peak_concurrent_jobs() == 4,
                    "S2: peak concurrency 4 (die + restore + pool + comm)");
    }

    // ------------------------------------------------------------------
    // Scenario 3 (full runtime): die backend COMP node vs restore DMA.
    // ------------------------------------------------------------------
    {
        g_terminals.clear();
        const std::string die_config_dir = config_dir + "_die";
        {
            std::error_code mk_ec;
            std::filesystem::create_directories(die_config_dir, mk_ec);
        }
        {
            std::ofstream out(die_config_dir + "/comm_group.json");
            out << "{}";
        }
        {
            // The default die point (contract section 3.5), same as the
            // die_model_adapter_test fixture.
            std::ofstream out(die_config_dir + "/resolved_die_model.json");
            out << R"({
  "schema_version": 1,
  "profile": "mesh_die_v1",
  "die": {
    "sram-mb-per-core": 0.75,
    "core-type": "large",
    "compute-area-mm2": 800,
    "hbm-per-side": 4,
    "hbm-sides": 1,
    "noc-width-bits": 512
  }
})";
        }
        {
            // No peak-perf/local-mem-bw keys: both are filled FROM the
            // library (1.49205e15 FLOP/s, 1.64e12 byte/s).
            std::ofstream out(die_config_dir + "/system.json");
            out << R"({
  "compute-cost-model": "die",
  "die-model-profile": "mesh_die_v1",
  "local-mem-latency": 100,
  "hbm-bandwidth-contention": 1,
  "hbm-kv-restore-bandwidth-sharing": 1,
  "trace-enabled": 0
})";
        }
        Fixture fixture(die_config_dir);
        Workload& workload = *fixture.workload();
        LocalHbmBandwidthModel* const model =
            workload.local_hbm_bandwidth_model.get();
        Sys& sys = *fixture.sys_ptr();

        // Restore node: 100 MB DMA (the real issue path; serial graph
        // semantics unchanged, hbm_dma single slot).
        OnlineNode restore_node;
        restore_node.kind = NodeKind::MemLoad;
        restore_node.node_type = 2;
        restore_node.name = "kv_restore";
        restore_node.request_id = "r1";
        restore_node.stage = "prefill";
        restore_node.generation = 0;
        restore_node.is_local_hbm_kv_restore = true;
        restore_node.compute.tensor_size = 104857600u;
        const uint64_t restore_id =
            fixture.issue_node(std::move(restore_node));

        // Die COMP node (norm op) through the real dispatch chain.
        nlohmann::json span;
        span["request_id"] = "r1";
        span["q_tokens"] = 8u;
        span["kv_tokens"] = 128u;
        span["q_start_pos"] = 0u;
        span["kv_start_pos"] = 0u;
        nlohmann::json spec;
        spec["schema_version"] = 1u;
        spec["op_kind"] = "norm";
        spec["phase"] = "prefill";
        spec["dtype"] = "bf16";
        spec["model_dims"] = {{"hidden", 4096u},
                              {"ffn", 14336u},
                              {"vocab", 152064u},
                              {"mlp_variant", "swiglu"},
                              {"norm_type", "rmsnorm"},
                              {"position_encoding", "rope"},
                              {"num_q_heads", 32u},
                              {"num_kv_heads", 8u},
                              {"head_dim", 128u}};
        spec["tp_slice"] = {{"tensor_parallel", 1u},
                            {"tensor_parallel_rank", 0u},
                            {"q_heads_local", 32u},
                            {"kv_heads_local", 8u},
                            {"ffn_local", 14336u},
                            {"vocab_local", 152064u}};
        spec["spans"] = nlohmann::json::array({span});
        spec["layer_count"] = 1u;
        spec["weight_passes"] = 1u;
        spec["kv_access"] = {{"identity", "local"}};
        spec["fusion_policy"] = "auto";
        OnlineNode die_node;
        die_node.kind = NodeKind::Compute;
        die_node.node_type = 4;
        die_node.name = "norm_node";
        die_node.request_id = "r1";
        die_node.stage = "prefill";
        die_node.generation = 0;
        die_node.compute.die_spec =
            DieModel::parse_die_op_spec(spec);
        const uint64_t die_id = fixture.issue_node(std::move(die_node));

        while (!fixture.event_queue->finished()) {
            fixture.event_queue->proceed();
        }

        // ---- Both nodes reach exactly one terminal ----------------------
        expect_true(g_terminals.count(restore_id) == 1 &&
                    g_terminals.count(die_id) == 1,
                    "S3: restore node and die COMP node each terminal "
                    "exactly once");
        // ---- Contention can only delay: span >= isolated estimate -------
        const auto plan =
            DieModel::compile(*DieModel::parse_die_op_spec(spec),
                              sys.die_runtime->hardware);
        const DieModel::CostResult alone = DieModel::estimate_alone(*plan);
        expect_true(g_first_tick.count(die_id) == 1 &&
                    g_first_tick.at(die_id) >=
                        alone.total_ns_suggested + 100,
                    "S3: die span >= isolated estimate + one-shot latency "
                    "(equal share only delays)");
        // ---- Byte conservation of the restore column --------------------
        expect_true(std::abs(model->restore_bytes_served() -
                             104857600.0) < 1e-6,
                    "S3: restore_bytes_served == 100 MB (untouched by die "
                    "service units)");
        expect_true(model->die_service_units_served() > 0.0 &&
                        model->die_actual_bytes_served() > 0.0,
                    "S3: die side-band records service AND actual bytes");
        // ---- Dynamic overlap proof --------------------------------------
        expect_true(model->hbm_shared_ns() > 0.0,
                    "S3: hbm_shared_ns > 0 (die memory leg and restore DMA "
                    "actually shared the bus)");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        !workload.die_model_adapter()
                             ->has_active_die_executions(),
                    "S3: GPU slot released and adapter drained at end");
    }

    // ------------------------------------------------------------------
    // cancel_die_jobs_of: teardown drops a pending job silently.
    // ------------------------------------------------------------------
    {
        Fixture fixture(config_dir);
        LocalHbmBandwidthModel* const model =
            fixture.workload()->local_hbm_bandwidth_model.get();
        CountingSink sink2;
        model->issue_die_memory(30.0, 21.0, &sink2, new SinkData(9));
        model->cancel_die_jobs_of(&sink2);
        expect_true(!model->has_active_jobs(),
                    "cancel_die_jobs_of drops the pending job");
        while (!fixture.event_queue->finished()) {
            fixture.event_queue->proceed();
        }
        expect_true(sink2.fires.empty(),
                    "cancelled job never fires its sink");
        expect_true(std::abs(model->die_service_units_served()) < 1e-9,
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
