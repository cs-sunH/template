/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
die_execution_lifecycle_test.cc -- DieExecution end-to-end lifecycle
regression (stage B; hermetic online fixture, no sh_test_mesh artifacts).

Same fixture shape as die_model_adapter_test.cc (online Sys over a
NodeStoreGraphSource, fake backend on a real EventQueue, default die point
of contract §3.5). The Timer/Memory mixed plan is a real compiled `norm`
plan driven to Done.

The contention-off oracle is an in-test full-speed driver of the same plan
(same shape as the stage A3 execution parity driver: Timer -> advance(ns),
Memory -> advance(max(1, ceil(service_units * 1e9 / B))) + complete), so the
expected machine clock is computed independently of the runtime under test.

Cases:
  A. contention off + local_mem_latency 100: the wall span equals the
     oracle machine elapsed + exactly one 100 ns latency charge (the
     one-shot first-access TIMER); the terminal fires exactly once; the GPU
     comp slot is occupied during and released after the run; tics_gpu_ops
     equals the span; the adapter's drain set empties.
  B. contention off + local_mem_latency 0: wall span == oracle machine
     elapsed exactly (no latency charge when the configured latency is 0).
  C. contention on: mid-flight the adapter reports an active execution and
     the GPU slot stays occupied; after the drain both clear; the span is
     at least estimate_alone + latency; the die side-band counters recorded
     service units AND actual bytes as separate quantities.
  D. marker under contention off: the legacy 1 ns event micro-job (issue
     tick + 1).
  E. teardown: an in-flight execution (big plan, partially pumped) is torn
     down with its Sys without crash, without any terminal firing, and the
     stale outer alarms drain harmlessly afterwards.

Build: cmake target
      AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/AstraRemoteMemoryAPI.hh"
#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
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
#include <json/json.hpp>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;

namespace {

bool g_ok = true;

void expect_true(const bool cond, const char* const what) {
    std::printf("%-72s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        g_ok = false;
    }
}

std::map<uint64_t, uint64_t> g_terminal_fires;
std::map<uint64_t, uint64_t> g_first_tick;

void tick_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
               const char* /*request_id*/, const char* /*stage*/,
               uint64_t /*generation*/, uint64_t tick, int status) {
    if (status != static_cast<int>(NodeTerminalStatus::Success)) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] node %llu terminal "
                     "status %d\n",
                     static_cast<unsigned long long>(node_id), status);
        g_ok = false;
    }
    if (g_first_tick.find(node_id) == g_first_tick.end()) {
        g_first_tick[node_id] = tick;
    }
    ++g_terminal_fires[node_id];
}

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

// Stub remote memory (this repository's Sys requires the backend; no MEM
// node is issued in this fixture, so the stub is never called).
class StubRemoteMemory final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int /*id*/, Sys* /*sys*/) override {}
    void issue(uint64_t /*tensor_size*/,
               WorkloadLayerHandlerData* /*wlhd*/) override {}
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
                     "[die_execution_lifecycle_test] cannot write %s\n",
                     path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out << content;
}

std::string repo_vendor_coarse_hardware_path() {
    std::filesystem::path source(__FILE__);
    if (source.is_relative()) source = std::filesystem::absolute(source);
    for (unsigned parent = 0; parent < 5; ++parent) {
        source = source.parent_path();
    }
    return (source / "extern/die_model/coarse/configs/example_hardware.json")
        .string();
}

void write_coarse_system_config(const std::string& config_dir,
                                const bool slow_hbm = false,
                                const uint64_t local_mem_latency = 100) {
    std::ifstream hardware_in(repo_vendor_coarse_hardware_path());
    if (!hardware_in.is_open()) {
        throw std::runtime_error("cannot open coarse DIE example hardware");
    }
    nlohmann::json hardware;
    hardware_in >> hardware;
    if (slow_hbm) {
        hardware["performance_profile"]["eta_hbm_matrix"] = 1e-6;
    }
    const std::string hardware_path = config_dir + "/coarse_hardware.json";
    write_text(hardware_path, hardware.dump(2));
    write_text(config_dir + "/coarse_system.json", nlohmann::json{
        {"compute-cost-model", "die"},
        {"die-model-profile", "coarse_die_v2"},
        {"die-model-config", hardware_path},
        {"hbm-bandwidth-contention", 1},
        {"hbm-kv-restore-bandwidth-sharing", 1},
        {"local-mem-latency", local_mem_latency},
        {"trace-enabled", 0}}.dump(2));
}

long double exact_number(const nlohmann::json& value) {
    if (value.is_string()) {
        const std::string exact = value.get<std::string>();
        size_t consumed = 0;
        const long double parsed = std::stold(exact, &consumed);
        if (consumed != exact.size()) {
            throw std::runtime_error("invalid exact decimal counter");
        }
        return parsed;
    }
    return value.get<long double>();
}

long double sum_hbm_matrix(const nlohmann::json& matrix) {
    long double sum = 0.0L;
    for (const auto& row : matrix) {
        for (const auto& direction : row) sum += exact_number(direction);
    }
    return sum;
}

void no_op_event(void*) {}

struct TeardownBeforeRuntimeContext {
    Workload* workload = nullptr;
    CoarseDieRuntime* bridge = nullptr;
    uint64_t marker_handle = 0;
    uint64_t sibling_handle = 0;
    uint64_t sampled_tick = 0;
    bool both_finished = false;
};

void teardown_before_runtime_callback(void* raw) {
    auto* context = static_cast<TeardownBeforeRuntimeContext*>(raw);
    context->sampled_tick = Sys::boostedTick();
    const auto marker = context->bridge->query_progress_at(
        context->marker_handle, context->sampled_tick);
    const auto sibling = context->bridge->query_progress_at(
        context->sibling_handle, context->sampled_tick);
    context->both_finished =
        marker.value("state", std::string()) == "finished" &&
        sibling.value("state", std::string()) == "finished" &&
        sum_hbm_matrix(marker.at("hbm_served_bytes")) == 1.0L &&
        sum_hbm_matrix(sibling.at("hbm_served_bytes")) == 1.0L;
    context->workload->local_hbm_bandwidth_model.reset();
}

// Norm spec; layer_count lets the caller scale the plan for long runs.
nlohmann::json spec_json(uint64_t layers) {
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
    spec["layer_count"] = layers;
    spec["weight_passes"] = 1u;
    spec["kv_access"] = {{"identity", "local"}};
    spec["fusion_policy"] = "auto";
    return spec;
}

std::shared_ptr<const DieModel::DieOpSpec> parse_spec(uint64_t layers) {
    return DieModel::parse_die_op_spec(spec_json(layers));
}

OnlineNode make_die_node(uint64_t id, uint64_t layers) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = "norm_node";
    node.request_id = "r1";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.die_spec = parse_spec(layers);
    return node;
}

// Independent full-speed oracle for the same plan: the exact drive policy
// DieExecution uses with hbm-bandwidth-contention=0 (Timer -> advance,
// Memory -> advance(max(1, ceil(service * 1e9 / B))) + complete).
uint64_t oracle_full_speed_elapsed(const std::shared_ptr<const DieModel::DieOpSpec>& spec,
                                   const DieModel::Hardware& hw) {
    auto plan = DieModel::compile(*spec, hw);
    DieModel::Execution machine(plan);
    while (!machine.finished()) {
        const DieModel::ExecutionAction action = machine.next_action();
        if (action.kind == DieModel::ExecutionAction::Kind::Timer) {
            machine.advance_time(std::max<uint64_t>(action.ns, 1));
        } else if (action.kind ==
                   DieModel::ExecutionAction::Kind::MemoryRequest) {
            const double ns_double =
                std::ceil(action.service_units * 1e9 / hw.hbm_bandwidth_bytes);
            machine.advance_time(std::max<uint64_t>(
                static_cast<uint64_t>(ns_double), 1));
            machine.complete_memory(action.stage_id);
        } else {
            break;
        }
    }
    return machine.elapsed_ns();
}

struct Fixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    StubRemoteMemory remote_mem;
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
        config_dir + "/" + system_file, &fixture.remote_mem,
        fixture.network_api.get(),
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
    return fixture;
}

void drain_queue(const std::shared_ptr<EventQueue>& queue) {
    while (!queue->finished()) {
        queue->proceed();
    }
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CompletionObserver::instance().set_hook(tick_hook, nullptr);

    const std::string tmp_template = "/tmp/die_exec_test_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    write_text(config_dir + "/die_no_contention.json",
               R"({
  "compute-cost-model": "die",
  "hbm-bandwidth-contention": 0,
  "local-mem-latency": 100
})");
    write_text(config_dir + "/die_no_contention_no_latency.json",
               R"({
  "compute-cost-model": "die",
  "hbm-bandwidth-contention": 0
})");
    write_text(config_dir + "/die_contention.json",
               R"({
  "compute-cost-model": "die",
  "local-mem-latency": 100
})");

    const DieModel::Hardware hw =
        DieModel::make_hardware(nlohmann::json::parse(kDieNodeJson));
    const auto one_layer_spec = parse_spec(1);
    const uint64_t oracle_one_layer =
        oracle_full_speed_elapsed(one_layer_spec, hw);
    const auto reference_plan = DieModel::compile(*one_layer_spec, hw);
    const DieModel::CostResult reference =
        DieModel::estimate_alone(*reference_plan);

    // ---- Case A: contention off, latency 100, norm plan -----------------
    {
        Fixture fixture = make_fixture(config_dir, "die_no_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(21, 1));
        workload.issue_dep_free_nodes();
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 1,
                    "A: GPU comp slot occupied at issue");
        drain_queue(fixture.event_queue);
        const uint64_t span = g_first_tick.at(21);  // issued at tick 0
        expect_true(g_terminal_fires.at(21) == 1,
                    "A: node terminal fires exactly once");
        expect_true(span >= reference.total_ns_suggested + 100,
                    "A: span >= isolated estimate + one-shot latency");
        expect_true(span == oracle_one_layer + 100,
                    "A: span == oracle machine elapsed + exactly one 100 ns "
                    "local_mem_latency charge");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 0,
                    "A: GPU comp slot released after Done");
        expect_true(
            !workload.die_model_adapter()->has_active_die_executions(),
            "A: adapter drains after Done");
    }

    // ---- Case B: contention off, no latency ------------------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture =
            make_fixture(config_dir, "die_no_contention_no_latency.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(22, 1));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(22) == 1,
                    "B: node terminal fires exactly once");
        expect_true(g_first_tick.at(22) == oracle_one_layer,
                    "B: span == oracle machine elapsed exactly (latency 0 "
                    "-> no latency charge)");
    }

    // ---- Case C: contention on ------------------------------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(23, 1));
        workload.issue_dep_free_nodes();
        // Mid-flight: pump a few events, the norm plan is far from done.
        for (int i = 0; i < 5 && !fixture.event_queue->finished(); ++i) {
            fixture.event_queue->proceed();
        }
        expect_true(
            workload.die_model_adapter()->has_active_die_executions(),
            "C: adapter reports an active execution mid-flight");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 1,
                    "C: GPU comp slot still occupied mid-flight");
        drain_queue(fixture.event_queue);
        const uint64_t span = g_first_tick.at(23);
        expect_true(g_terminal_fires.at(23) == 1,
                    "C: node terminal fires exactly once under contention");
        expect_true(span >= reference.total_ns_suggested + 100,
                    "C: span >= isolated estimate + latency (equal share "
                    "can only delay)");
        expect_true(
            !workload.die_model_adapter()->has_active_die_executions(),
            "C: adapter drains after Done under contention");
        expect_true(
            workload.local_hbm_bandwidth_model != nullptr &&
                workload.local_hbm_bandwidth_model
                        ->die_service_units_served() > 0.0 &&
                workload.local_hbm_bandwidth_model
                        ->die_actual_bytes_served() > 0.0,
            "C: die side-band records service AND actual bytes (separate "
            "columns)");
    }

    // ---- Case D: marker micro-job, 1 ns event arm ------------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_no_contention.json");
        Workload& workload = *fixture.sys->workload;
        OnlineNode marker;
        marker.global_id = 24;
        marker.rank = 0;
        marker.kind = NodeKind::Compute;
        marker.node_type = 4;
        marker.name = "first_token";
        marker.request_id = "r1";
        marker.stage = "prefill";
        marker.generation = 0;
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        const uint64_t issue_tick = Sys::boostedTick();  // == 0
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(24) == 1 &&
                    g_first_tick.at(24) == issue_tick + 1,
                    "D: marker without the HBM model completes on the 1 ns "
                    "event (legacy micro-job)");
    }

    // ---- Case E: teardown of an in-flight execution ----------------------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(25, 8));  // big plan
        workload.issue_dep_free_nodes();
        for (int i = 0; i < 8 && !fixture.event_queue->finished(); ++i) {
            fixture.event_queue->proceed();
        }
        expect_true(g_terminal_fires.count(25) == 0,
                    "E: big node still in flight after partial pumping");
        // Tear the whole Sys down mid-flight (Workload destructor ->
        // adapter shutdown -> every DieExecution cancels its pending event
        // and drops its pending HBM die job). No completion may fire.
        fixture.sys.reset();
        // Stale outer alarms (the legacy fake backend cannot cancel
        // physically) must drain through the retired-Sys guard harmlessly.
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(25) == 0,
                    "E: teardown fires no completion for the in-flight "
                    "node (exactly-zero terminals)");
    }

    // ---- Case F: coarse control marker is a delayed DIE endpoint ----------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        write_coarse_system_config(config_dir);
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        bool scalar_comp_rejected = false;
        try {
            workload.local_hbm_bandwidth_model->issue_compute(1, 1, nullptr);
        } catch (const std::runtime_error& error) {
            scalar_comp_rejected =
                std::string(error.what()).find("typed plan") != std::string::npos;
        }
        expect_true(scalar_comp_rejected,
                    "F: ordinary scalar COMPUTE remains rejected in coarse mode");

        OnlineNode marker;
        marker.global_id = 26;
        marker.rank = 0;
        marker.kind = NodeKind::Compute;
        marker.node_type = 4;
        marker.name = "coarse_first_token";
        marker.request_id = "r1";
        marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();

        const auto before = bridge.export_snapshot(Sys::boostedTick());
        uint64_t endpoint_count = 0;
        bool endpoint_contract = false;
        for (const auto& execution :
             before.at("runtime_snapshot").at("executions")) {
            const auto& endpoint =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint.value("owner_kind", std::string()) != "die") continue;
            ++endpoint_count;
            endpoint_contract = endpoint.value("physical_bytes", uint64_t{0}) == 1 &&
                endpoint.value("initial_delay_ns", uint64_t{0}) == 100 &&
                execution.at("plan").at("summary").value(
                    "initial_delay_ns", uint64_t{0}) == 100;
        }
        expect_true(endpoint_count == 1 && endpoint_contract &&
                        g_terminal_fires.count(26) == 0 &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1,
                    "F: one-byte DIE endpoint waits its 100 ns startup delay");
        drain_queue(fixture.event_queue);
        const auto counters = bridge.counters();
        const auto& categories = counters.at("served_by_category");
        const long double die_bytes = sum_hbm_matrix(
            categories.at("die").at("hbm_served_bytes_by_row_direction_exact"));
        const long double comp_bytes = sum_hbm_matrix(
            categories.at("comp").at("hbm_served_bytes_by_row_direction_exact"));
        const long double comm_bytes =
            sum_hbm_matrix(categories.at("comm_send")
                               .at("hbm_served_bytes_by_row_direction_exact")) +
            sum_hbm_matrix(categories.at("comm_recv")
                               .at("hbm_served_bytes_by_row_direction_exact"));
        expect_true(g_terminal_fires[26] == 1 && g_first_tick[26] >= 101 &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        die_bytes == 1.0L && comp_bytes == 0.0L &&
                        comm_bytes == 0.0L &&
                        workload.local_hbm_bandwidth_model
                                ->compute_bytes_served() == 1.0 &&
                        workload.local_hbm_bandwidth_model
                                ->restore_bytes_served() == 0.0 &&
                        workload.local_hbm_bandwidth_model
                                ->pool_read_bytes_served() == 0.0 &&
                        workload.local_hbm_bandwidth_model
                                ->pool_write_bytes_served() == 0.0,
                    "F: one marker byte is DIE-owned; COMP/COMM/RESTORE/POOL ledgers stay isolated");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[26] == 1,
                    "F: completed coarse marker teardown has no duplicate terminal");
    }

    // ---- Case G: live marker prefix is read-only and survives cancel ------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        write_coarse_system_config(config_dir, true);
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 27;
        marker.rank = 0;
        marker.kind = NodeKind::Compute;
        marker.node_type = 4;
        marker.name = "coarse_partial_marker";
        marker.request_id = "r1";
        marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        fixture.event_queue->schedule_event(101, no_op_event, nullptr);
        fixture.event_queue->proceed();
        fixture.event_queue->proceed();
        const uint64_t now = Sys::boostedTick();
        const auto snapshot = bridge.export_snapshot(now);
        uint64_t handle = 0;
        long double expected_prefix = 0.0L;
        for (const auto& execution :
             snapshot.at("runtime_snapshot").at("executions")) {
            const auto& endpoint =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint.value("owner_kind", std::string()) != "die") continue;
            handle = execution.at("handle").get<uint64_t>();
            expected_prefix = sum_hbm_matrix(execution.at("hbm_served"));
        }
        const auto progress_before = bridge.query_progress(handle);
        const auto counters_before = bridge.counters();
        const double live_bytes =
            workload.local_hbm_bandwidth_model->compute_bytes_served();
        const auto sampled = bridge.query_progress_at(handle, now);
        const auto progress_after = bridge.query_progress(handle);
        const auto counters_after = bridge.counters();
        const bool reversal_rejected = [&] {
            try {
                (void)bridge.query_progress_at(handle, now - 1);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        }();
        const bool stale_rejected = [&] {
            try {
                (void)bridge.query_progress_at(0, now);
            } catch (const std::out_of_range&) {
                return true;
            }
            return false;
        }();
        expect_true(now == 101 && handle != 0 && expected_prefix > 0.0L &&
                        expected_prefix < 1.0L && live_bytes > 0.0 &&
                        std::abs(static_cast<long double>(live_bytes) -
                                 expected_prefix) < 1e-18L &&
                        sampled == progress_after &&
                        progress_before == progress_after &&
                        counters_before == counters_after && reversal_rejected &&
                        stale_rejected && bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        g_terminal_fires.count(27) == 0,
                    "G: live prefix getter is exact/read-only; stale and reverse samples fail closed");
        bridge.cancel(handle, now, true);
        const auto cancelled = bridge.counters();
        const auto& die = cancelled.at("served_by_category").at("die");
        const long double die_prefix = sum_hbm_matrix(
            die.at("hbm_served_bytes_by_row_direction_exact"));
        const auto& comm_send = cancelled.at("served_by_category").at("comm_send");
        expect_true(std::abs(die_prefix - expected_prefix) < 1e-18L &&
                        std::abs(static_cast<long double>(
                                     workload.local_hbm_bandwidth_model
                                         ->compute_bytes_served()) -
                                 expected_prefix) < 1e-18L &&
                        sum_hbm_matrix(comm_send.at(
                            "hbm_served_bytes_by_row_direction_exact")) == 0.0L &&
                        !bridge.has_active_executions() &&
                        cancelled.at("endpoint_registry_size") == 0 &&
                        cancelled.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(27) == 0,
                    "G: cancellation retains the exact partial DIE/compatibility prefix without graph completion");
        bool retired_handle_rejected = false;
        try {
            (void)bridge.query_progress_at(handle, now);
        } catch (const std::out_of_range&) {
            retired_handle_rejected = true;
        }
        expect_true(retired_handle_rejected,
                    "G: cancelled endpoint handle is stale after retirement");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(27) == 0,
                    "G: cancelled marker teardown does not publish a terminal");
    }

    // ---- Case H: partial-marker teardown retains its exact DIE prefix ----
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        write_coarse_system_config(config_dir, true, 1);
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 28;
        marker.rank = 0;
        marker.kind = NodeKind::Compute;
        marker.node_type = 4;
        marker.name = "coarse_teardown_marker";
        marker.request_id = "r1";
        marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        fixture.event_queue->schedule_event(101, no_op_event, nullptr);
        fixture.event_queue->proceed();
        fixture.event_queue->proceed();
        const uint64_t now = Sys::boostedTick();
        const double expected_prefix =
            workload.local_hbm_bandwidth_model->compute_bytes_served();
        expect_true(bridge.has_active_executions() &&
                        now == 101 && expected_prefix > 0.0 &&
                        expected_prefix < 1.0 &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        g_terminal_fires.count(28) == 0,
                    "H: partially served marker is live before HBM facade teardown");
        workload.local_hbm_bandwidth_model.reset();
        const auto retired = bridge.counters();
        const long double die_prefix = sum_hbm_matrix(
            retired.at("served_by_category").at("die")
                .at("hbm_served_bytes_by_row_direction_exact"));
        expect_true(!bridge.has_active_executions() &&
                        retired.at("endpoint_registry_size") == 0 &&
                        retired.at("transaction_registry_size") == 0 &&
                        std::abs(die_prefix - expected_prefix) < 1e-18L &&
                        g_terminal_fires.count(28) == 0,
                    "H: teardown retires the marker without losing its prefix or calling the graph");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(28) == 0,
                    "H: stale teardown alarms do not complete the marker");
    }

    // ---- Case I: completed markers await callbacks during teardown -------
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        write_coarse_system_config(config_dir);
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        const uint64_t issue_tick = Sys::boostedTick();
        const uint64_t terminal_tick = issue_tick + 101;
        TeardownBeforeRuntimeContext teardown{&workload, &bridge};
        // This callback is inserted before the Runtime's terminal event.
        // It tears down the HBM facade inside the same-tick EventList, while
        // the marker callback is still waiting behind it.
        fixture.event_queue->schedule_event(
            terminal_tick, teardown_before_runtime_callback, &teardown);
        OnlineNode marker;
        marker.global_id = 29;
        marker.rank = 0;
        marker.kind = NodeKind::Compute;
        marker.node_type = 4;
        marker.name = "coarse_finished_marker";
        marker.request_id = "r1";
        marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1;
        marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        const auto before = bridge.export_snapshot(issue_tick);
        uint64_t marker_handle = 0;
        nlohmann::json sibling_endpoint;
        for (const auto& execution :
             before.at("runtime_snapshot").at("executions")) {
            const auto& endpoint =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint.value("owner_kind", std::string()) == "die") {
                marker_handle = execution.at("handle").get<uint64_t>();
                sibling_endpoint = endpoint;
            }
        }
        bool sibling_callback = false;
        sibling_endpoint["endpoint_id"] = 9001;
        sibling_endpoint["transaction_id"] = 9001;
        coarse_die::Owner sibling_owner;
        sibling_owner.request_id = 9001;
        sibling_owner.node_id = 9001;
        sibling_owner.execution_id = 9001;
        sibling_owner.generation = 1;
        const uint64_t sibling_handle = bridge.submit_endpoint(
            sibling_endpoint, sibling_owner,
            [&sibling_callback](uint64_t, const nlohmann::json&) {
                sibling_callback = true;
            });
        teardown.marker_handle = marker_handle;
        teardown.sibling_handle = sibling_handle;
        fixture.event_queue->proceed();
        fixture.event_queue->proceed();
        expect_true(teardown.both_finished &&
                        teardown.sampled_tick == terminal_tick &&
                        workload.local_hbm_bandwidth_model == nullptr &&
                        bridge.has_active_executions() && !sibling_callback &&
                        g_terminal_fires.count(29) == 0,
                    "I: teardown runs between full service and queued marker/sibling callbacks");
        const auto marker_retired = bridge.counters();
        const long double marker_die_bytes = sum_hbm_matrix(
            marker_retired.at("served_by_category").at("die")
                .at("hbm_served_bytes_by_row_direction_exact"));
        const bool marker_handle_stale = [&] {
            try {
                (void)bridge.query_progress_at(marker_handle, terminal_tick);
            } catch (const std::out_of_range&) {
                return true;
            }
            return false;
        }();
        const bool sibling_still_terminal =
            bridge.query_progress_at(sibling_handle, terminal_tick)
                    .value("state", std::string()) == "finished";
        expect_true(bridge.has_active_executions() &&
                        marker_retired.at("endpoint_registry_size") == 1 &&
                        marker_retired.at("transaction_registry_size") == 1 &&
                        marker_die_bytes == 2.0L && marker_handle_stale &&
                        sibling_still_terminal && !sibling_callback &&
                        g_terminal_fires.count(29) == 0,
                    "I: marker teardown retains both prefixes without dispatching its terminal sibling");
        bridge.cancel(sibling_handle, terminal_tick, false);
        const auto retired = bridge.counters();
        const bool sibling_handle_stale_after_retire = [&] {
            try {
                (void)bridge.query_progress_at(sibling_handle, terminal_tick);
            } catch (const std::out_of_range&) {
                return true;
            }
            return false;
        }();
        expect_true(!bridge.has_active_executions() &&
                        retired.at("endpoint_registry_size") == 0 &&
                        retired.at("transaction_registry_size") == 0 &&
                        sibling_handle_stale_after_retire && !sibling_callback &&
                        g_terminal_fires.count(29) == 0,
                    "I: sibling retires independently without publishing either callback");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(29) == 0 && !sibling_callback,
                    "I: stale teardown alarms do not publish completed markers");
    }

    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    if (!g_ok) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[die_execution_lifecycle_test] ALL PASS\n");
    return 0;
}
