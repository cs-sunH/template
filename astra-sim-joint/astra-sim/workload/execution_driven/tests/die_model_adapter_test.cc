/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
die_model_adapter_test.cc -- die backend dispatch regression (joint rollout
stage 1, contract §7.3; hermetic online fixture, no sh_test_mesh artifacts;
ported from the face reference and adapted to this repository: Sys ctor takes
a remote-memory API parameter, the system config requires scheduling-policy,
and the Workload-layer die guards use this repository's fail_closed (abort)
convention -- the missing-spec case is probed in a forked child and the
adapter's own throw is asserted directly).

Fixture: a real Sys in online mode over a NodeStoreGraphSource, a fake
AstraNetworkAPI backed by a real analytical EventQueue (time actually
advances), the default die point of contract §3.5 (0.75 MB large / 800 mm2 /
HBM 4x1 / 512 bit, resolved_die_model.json written next to the system
config), and CompletionObserver terminal collection. LoggerFactory is never
initialized (null sinks, same shape as statistics_online_compaction_test);
the forked probes run BEFORE any logger exists in the parent, so each child
builds its own async pool (fork-safety note there).

Cases:
  A. Sys backend resolution: compute-cost-model=die loads the die runtime,
     fills peak-perf/local-mem-bw FROM the library (1.49205e15 FLOP/s /
     1.64e12 byte/s), keeps the HBM contention model on, and rejects the
     roofline/replay conflicts (die + roofline-enabled=1 and die +
     replay-only=1 each exit fail-closed in a forked child).
  B. A real COMP node carrying a die_op_spec (norm op) dispatches through
     the DieModelAdapter: the node finishes through finish_general_node
     EXACTLY ONCE (one Success terminal), the GPU comp slot is released,
     and tics_gpu_ops equals the node's issue -> completion span.
  C. A control_marker node keeps the legacy micro-job semantics verbatim:
     with the HBM fluid model it drains a (1 op, 1 byte) COMPUTE job whose
     latency phase is local-mem-latency (100 ns) plus one sub-ns drain
     transition tick -- terminal at exactly issue + 101, no die plan, no
     launch charge.
  D. A real COMP node WITHOUT a die_op_spec fails closed at the Workload
     layer (abort in a forked child, never a silent roofline/replay
     fallback), the DieModelAdapter's own guard throws naming die_op_spec,
     and a die_op_spec carrying a non-zero runtime_ns is rejected by the
     GraphBatch parser.

Build: cmake target AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"
#include "astra-sim/workload/execution_driven/PipelineKernelContract.hh"
#include "DieModel.hh"
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
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
    std::printf("%-70s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        g_ok = false;
    }
}

struct CompletionRecord {
    uint64_t tick = 0;
    uint64_t fires = 0;
    int status = -1;
};

std::map<uint64_t, CompletionRecord> g_terminals;

void collecting_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
                     const char* /*request_id*/, const char* /*stage*/,
                     uint64_t /*generation*/, uint64_t tick,
                     int terminal_status) {
    auto& record = g_terminals[node_id];
    record.tick = tick;
    record.status = terminal_status;
    ++record.fires;
}

// No-op remote-memory API (this repository's Sys ctor requires one).
class TestRemoteMemoryApi final : public AstraSim::AstraRemoteMemoryAPI {
  public:
    void set_sys(int, AstraSim::Sys*) override {}
    void issue(uint64_t, AstraSim::WorkloadLayerHandlerData*) override {}
};

// Fake backend: schedules into a real EventQueue and reads the clock from
// it (the legacy cancellable path -- sim_schedule_cancellable keeps the
// base-class behavior, Sys-level bucket cancellation still works).
class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*)(void*), void*) override {
        return 0;
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

const char* kDieNodeJson = R"({
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

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr,
                     "[die_model_adapter_test] cannot write fixture file %s\n",
                     path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out << content;
}

nlohmann::json norm_spec_json() {
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
    return spec;
}

OnlineNode make_compute_node(uint64_t id, const char* name) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = name;
    node.is_cpu_op = false;
    node.is_timer_op = false;
    node.request_id = "r1";
    node.stage = "prefill";
    node.generation = 0;
    return node;
}

struct Fixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    TestRemoteMemoryApi remote_memory;
    std::unique_ptr<Sys> sys;
    NodeStoreGraphSource* source = nullptr;
};

Fixture make_fixture(const std::string& config_dir,
                     const std::string& system_file) {
    Fixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    fixture.network_api =
        std::make_unique<FakeQueueNetworkApi>(fixture.event_queue);
    auto source = std::make_shared<NodeStoreGraphSource>();
    fixture.source = source.get();
    fixture.sys = std::make_unique<Sys>(
        0, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, &fixture.remote_memory,
        fixture.network_api.get(), std::vector<int>{1}, std::vector<int>{1},
        1.0, 1.0, false, ExecutionMode::Online, source);
    // The adapter's online guard requires a run-frozen PipelineKernelContract
    // before the first DIE node (in production the first valid GraphBatch
    // freezes it via the bridge; this hermetic fixture bypasses the batch
    // path, so freeze the same minimal contract here -- the identical
    // pattern die_execution_lifecycle_test's fixture adopted when the
    // guard landed). Without it case B's die node dispatch aborts with
    // the adapter's "before the first valid GraphBatch" throw.
    fixture.sys->workload->set_pipeline_kernel_contract(
        PipelineKernelContract::parse(nlohmann::json{
            {"projection_abi_version", 1u},
            {"descriptor_version", 1u},
            {"pipeline_protocol_version", 1u},
            {"model_version", "v7-three-actions-20261014"},
            {"timer_whitelist_version", "timer-whitelist-v1-20261005"},
            {"window_tile_limit", 1u},
            {"window_local_error_ppm", 10000u},
            {"window_refine_margin_ppm", 20000u},
            {"prediction_config_sha256",
             "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}}));
    return fixture;
}

void drain_queue(const std::shared_ptr<EventQueue>& queue) {
    while (!queue->finished()) {
        queue->proceed();
    }
}

std::string make_config_dir() {
    const std::string tmp_template = "/tmp/die_adapter_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr,
                     "[die_model_adapter_test] mkdtemp(%s) failed\n",
                     tmp_template.c_str());
        std::exit(EXIT_FAILURE);
    }
    return std::string(made);
}

// Child-probe helper for the fail-closed Sys configurations (sys_panic
// exits(1)) and the abort-based Workload-layer guards (fail_closed aborts):
// a clean EXIT_SUCCESS child means the configuration was NOT rejected.
// Accepts exit(1) or SIGABRT -- both are deliberate fail-closed channels
// here, unlike a crash probe. Runs before any parent-side logger exists:
// each child builds its own async pool (same fork-safety reason as
// statistics_online_compaction_test).
void expect_child_rejected(const char* what,
                           const std::function<void()>& child_body) {
    const pid_t child = fork();
    if (child == 0) {
        child_body();
        _exit(EXIT_SUCCESS);
    }
    int status = 0;
    const pid_t waited = waitpid(child, &status, 0);
    const bool rejected =
        waited == child &&
        ((WIFEXITED(status) && WEXITSTATUS(status) != EXIT_SUCCESS) ||
         WIFSIGNALED(status));
    expect_true(rejected, what);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const std::string config_dir = make_config_dir();
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    write_text(config_dir + "/die_system.json",
               R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "die-model-profile": "mesh_die_v1",
  "local-mem-latency": 100,
  "trace-enabled": 0
})");
    write_text(config_dir + "/conflict_roofline.json",
               R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "roofline-enabled": 1
})");
    write_text(config_dir + "/conflict_replay.json",
               R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "replay-only": 1
})");

    // ---- Case A (fail-closed halves): the conflict matrix, forked -------
    // (Both this and the Case-D Workload-half probe below must run BEFORE
    // any parent-side Sys/fixture exists: the Workload constructor creates
    // the process's first async logger pool, and forking after that risks
    // inheriting a queue mutex held by the (non-copied) worker thread --
    // the same fork-safety discipline as statistics_online_compaction_test.)
    expect_child_rejected("A: die + roofline-enabled=1 exits fail-closed", [&] {
        auto queue = std::make_shared<EventQueue>();
        FakeQueueNetworkApi api(queue);
        TestRemoteMemoryApi remote_memory;
        auto source = std::make_shared<NodeStoreGraphSource>();
        const Sys sys(0, "unused", "empty",
                      config_dir + "/conflict_roofline.json", &remote_memory,
                      &api, std::vector<int>{1}, std::vector<int>{1}, 1.0,
                      1.0, false, ExecutionMode::Online, source);
        (void)sys;
    });
    expect_child_rejected("A: die + replay-only=1 exits fail-closed", [&] {
        auto queue = std::make_shared<EventQueue>();
        FakeQueueNetworkApi api(queue);
        TestRemoteMemoryApi remote_memory;
        auto source = std::make_shared<NodeStoreGraphSource>();
        const Sys sys(0, "unused", "empty",
                      config_dir + "/conflict_replay.json", &remote_memory,
                      &api, std::vector<int>{1}, std::vector<int>{1}, 1.0,
                      1.0, false, ExecutionMode::Online, source);
        (void)sys;
    });
    // ---- Case D (Workload half), early fork: missing die_op_spec aborts --
    // (this repository's fail-closed convention for issue-path guards; the
    // whole fixture is built INSIDE the forked child so the abort never
    // touches this process)
    expect_child_rejected(
        "D: real COMP without die_op_spec fails closed at the "
        "Workload layer (no silent roofline/replay fallback)",
        [&] {
            Fixture guard_fixture = make_fixture(config_dir, "die_system.json");
            OnlineNode orphan = make_compute_node(13, "real_comp");
            orphan.compute.num_ops = 1000;
            orphan.compute.tensor_size = 2000;
            guard_fixture.source->store().add_node(orphan);
            guard_fixture.sys->workload->issue_dep_free_nodes();
        });

    // Parent-side logging starts here (null sinks).
    CompletionObserver::instance().set_hook(collecting_hook, nullptr);

    // ---- Case A (resolution) + B + C + D on one fixture -----------------
    {
        Fixture fixture = make_fixture(config_dir, "die_system.json");
        Sys& sys = *fixture.sys;
        expect_true(sys.compute_cost_model == ComputeCostModel::Die,
                    "A: compute-cost-model=die resolves to the Die backend");
        expect_true(sys.die_runtime != nullptr,
                    "A: die runtime (hardware + plan cache) installed");
        expect_true(!sys.roofline_enabled && sys.roofline == nullptr,
                    "A: roofline stays off under the die backend");
        expect_true(sys.workload->local_hbm_bandwidth_model != nullptr,
                    "A: HBM fluid model stays on (contention default)");
        const double expect_peak = 1225.0 * 1218.0 * 1e9;  // FLOP/s
        const double expect_bw = 4.0 * 410e9;              // byte/s
        expect_true(std::abs(sys.peak_perf - expect_peak) <= 1e-6 * expect_peak,
                    "A: peak_perf filled from the library (PE_peak_die)");
        expect_true(std::abs(sys.local_mem_bw - expect_bw) <= 1e-6 * expect_bw,
                    "A: local-mem-bw filled from the library");
        expect_true(sys.die_model_config_path ==
                        config_dir + "/resolved_die_model.json",
                    "A: die-model-config defaults next to the system.json");

        // No peak-perf/local-mem-bw keys were written above: the fills came
        // from the library, proving the missing-scalar path.
        Workload& workload = *sys.workload;

        // ---- Case B: die_spec node -> adapter -> one completion ---------
        OnlineNode node = make_compute_node(11, "attn_norm");
        node.compute.num_ops = 123;
        node.compute.tensor_size = 456;
        node.compute.die_spec = DieModel::parse_die_op_spec(norm_spec_json());
        fixture.source->store().add_node(node);
        const DieModelAdapter::IssueTiming timing_before_first_issue =
            DieModelAdapter::cumulative_issue_timing();
        workload.issue_dep_free_nodes();
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 1,
                    "B: GPU comp slot occupied while the die node runs");
        drain_queue(fixture.event_queue);

        const auto it = g_terminals.find(11);
        expect_true(it != g_terminals.end() && it->second.fires == 1 &&
                    it->second.status ==
                        static_cast<int>(NodeTerminalStatus::Success),
                    "B: die node finishes through finish_general_node "
                    "exactly once (Success)");
        expect_true(it != g_terminals.end() && it->second.tick >= 101,
                    "B: completion tick includes the one-shot "
                    "local_mem_latency (>= 100 latency + 1)");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 0,
                    "B: GPU comp slot released after completion");
        expect_true(it != g_terminals.end() &&
                    workload.hw_resource->tics_gpu_ops == it->second.tick,
                    "B: tics_gpu_ops equals the issue->completion span");
        expect_true(workload.die_model_adapter() != nullptr &&
                    !workload.die_model_adapter()->has_active_die_executions(),
                    "B: adapter drains after the node completes");
        const DieModelAdapter::IssueTiming timing_after_first_issue =
            DieModelAdapter::cumulative_issue_timing();
        expect_true(
            timing_after_first_issue.calls == timing_before_first_issue.calls + 1 &&
                timing_after_first_issue.cache_get_calls ==
                    timing_before_first_issue.cache_get_calls + 1 &&
                timing_after_first_issue.cache_misses ==
                    timing_before_first_issue.cache_misses + 1 &&
                timing_after_first_issue.cache_hits ==
                    timing_before_first_issue.cache_hits &&
                timing_after_first_issue.compile_calls ==
                    timing_before_first_issue.compile_calls + 1 &&
                timing_after_first_issue.cache_put_calls ==
                    timing_before_first_issue.cache_put_calls + 1 &&
                timing_after_first_issue.setup_calls ==
                    timing_before_first_issue.setup_calls + 1 &&
                timing_after_first_issue.start_calls ==
                    timing_before_first_issue.start_calls + 1,
            "B: first DIE dispatch records cache miss, compile, setup, start");

        // Utilization aggregation (contract §7.6): the die completion fills
        // compute/memory/vector on the compact online state, and the exact
        // accumulator receives value * duration == plan quantity * 1e9 /
        // physical peak (the elapsed cancels), verifiable bit-for-bit
        // against the independently compiled plan.
        {
            const DieModel::Hardware& dhw = sys.die_runtime->hardware;
            const auto plan =
                DieModel::compile(*node.compute.die_spec, dhw);
            const auto roofline = workload.stats
                                      ->calculate_roofline_utilization_in_window(
                                          0, it->second.tick);
            const double expect_compute =
                plan->flops() / dhw.pe_peak_die * 1e9;
            const double expect_memory =
                (plan->matrix_bytes() + plan->vector_bytes()) /
                dhw.hbm_bandwidth_bytes * 1e9;
            const double expect_vector =
                plan->eqops() / dhw.vec_peak_die * 1e9;
            expect_true(std::abs(roofline.compute_utilization_weighted_sum -
                                 expect_compute) <=
                            1e-6 * std::max(1.0, expect_compute),
                        "B: compute utilization weighted sum == plan FLOP / "
                        "PE_peak (die caliber)");
            expect_true(std::abs(roofline.memory_utilization_weighted_sum -
                                 expect_memory) <=
                            1e-6 * std::max(1.0, expect_memory),
                        "B: memory utilization weighted sum == actual bytes "
                        "/ B_mem (die caliber)");
            expect_true(std::abs(roofline.vector_utilization_weighted_sum -
                                 expect_vector) <=
                            1e-6 * std::max(1.0, expect_vector),
                        "B: vector utilization weighted sum == eqops / V_peak "
                        "(die-only field aggregates)");
        }

        // Reissue the exact spec through a second real DIE node. This keeps
        // the fixture physical and verifies the warm-cache branch without
        // placing a wall-time threshold on either path.
        OnlineNode repeated = make_compute_node(15, "attn_norm_repeat");
        repeated.compute.die_spec =
            DieModel::parse_die_op_spec(norm_spec_json());
        fixture.source->store().add_node(repeated);
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        const auto repeated_it = g_terminals.find(15);
        const DieModelAdapter::IssueTiming timing_after_reuse =
            DieModelAdapter::cumulative_issue_timing();
        expect_true(repeated_it != g_terminals.end() &&
                        repeated_it->second.fires == 1 &&
                        timing_after_reuse.calls ==
                            timing_before_first_issue.calls + 2 &&
                        timing_after_reuse.cache_get_calls ==
                            timing_before_first_issue.cache_get_calls + 2 &&
                        timing_after_reuse.cache_misses ==
                            timing_before_first_issue.cache_misses + 1 &&
                        timing_after_reuse.cache_hits ==
                            timing_before_first_issue.cache_hits + 1 &&
                        timing_after_reuse.compile_calls ==
                            timing_before_first_issue.compile_calls + 1 &&
                        timing_after_reuse.cache_put_calls ==
                            timing_before_first_issue.cache_put_calls + 1 &&
                        timing_after_reuse.setup_calls ==
                            timing_before_first_issue.setup_calls + 2 &&
                        timing_after_reuse.start_calls ==
                            timing_before_first_issue.start_calls + 2,
                    "B: repeated same-spec DIE dispatch records warm cache hit");

        // ---- Case C: control_marker keeps the legacy micro-job ----------
        // The marker drains a (1 op, 1 byte) COMPUTE job: latency phase
        // 100 ns then one sub-ns byte-drain transition tick (ceil to 1 ns;
        // the single FLOP drains within that same tick at the die-filled
        // peak).
        g_terminals.clear();
        OnlineNode marker = make_compute_node(12, "first_token");
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        marker.compute.op_role = OpRole::ControlMarker;
        fixture.source->store().add_node(marker);
        const uint64_t marker_issue_tick = Sys::boostedTick();
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        const auto marker_it = g_terminals.find(12);
        expect_true(marker_it != g_terminals.end() &&
                    marker_it->second.fires == 1 &&
                    marker_it->second.tick == marker_issue_tick + 101,
                    "C: control_marker completes at issue + 100 + 1 "
                    "(legacy micro-job, no die plan)");

        // ---- Case D (adapter half): the adapter's own guard throws ------
        {
            OnlineNode orphan = make_compute_node(14, "real_comp");
            orphan.compute.num_ops = 1000;
            orphan.compute.tensor_size = 2000;
            bool threw = false;
            try {
                workload.die_model_adapter()->issue_die(orphan);
            } catch (const std::runtime_error& e) {
                threw = std::string(e.what()).find("die_op_spec") !=
                        std::string::npos;
            }
            expect_true(threw,
                        "D: DieModelAdapter refuses a spec-less node "
                        "(throw names die_op_spec)");
        }
    }

    // ---- Case D (parser half): die_op_spec + runtime_ns != 0 ------------
    {
        nlohmann::json resp;
        resp["schema_version"] = 3u;
        resp["source_delivery_sequence"] = 1u;
        resp["kernel_contract"] = {
            {"projection_abi_version", 1u},
            {"descriptor_version", 1u},
            {"pipeline_protocol_version", 1u},
            {"model_version", "v7-three-actions-20261014"},
            {"timer_whitelist_version", "timer-whitelist-v1-20261005"},
            {"window_tile_limit", 1u},
            {"window_local_error_ppm", 10000u},
            {"window_refine_margin_ppm", 20000u},
            {"prediction_config_sha256", std::string(64, '0')}};
        nlohmann::json node_json;
        node_json["rank"] = 0;
        node_json["id"] = 1u;
        node_json["type"] = 4u;
        node_json["name"] = "n";
        node_json["is_cpu_op"] = false;
        node_json["is_timer_op"] = false;
        node_json["inputs_values"] = "";
        node_json["request_id"] = "r1";
        node_json["stage"] = "prefill";
        node_json["generation"] = 0u;
        node_json["compute"] = {{"num_ops", 1u},
                                {"tensor_size", 1u},
                                {"runtime_ns", 55u},
                                {"die_op_spec", norm_spec_json()}};
        node_json["comm"] = {
            {"bytes", 0u}, {"src", 0}, {"dst", 0}, {"tag", 0u}};
        node_json["coll"] = {{"comm_type", 0u},
                             {"bytes", 0u},
                             {"priority", 0u},
                             {"pg_name", ""}};
        node_json["mem"] = {{"tensor_size", 0u},
                            {"is_local_hbm_kv_restore", false},
                            {"hbm_access_mode", 0}};
        resp["nodes"] = nlohmann::json::array({node_json});
        bool rejected = false;
        try {
            parse_graph_batch(resp, 1);
        } catch (const ParseError& e) {
            rejected = std::string(e.what()).find("runtime_ns == 0") !=
                       std::string::npos;
        }
        expect_true(rejected,
                    "D: parser rejects die_op_spec with runtime_ns != 0");
    }

    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    if (!g_ok) {
        std::fprintf(stderr, "[die_model_adapter_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[die_model_adapter_test] ALL PASS\n");
    return 0;
}
