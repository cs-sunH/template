/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
die_execution_lifecycle_test.cc -- DieExecution end-to-end lifecycle
regression (joint rollout stage 1; hermetic online fixture, no sh_test_mesh
artifacts; ported from the face reference and adapted to this repository's
Sys ctor (remote-memory API parameter) and mandatory scheduling-policy key).

Same fixture shape as die_model_adapter_test.cc (online Sys over a
NodeStoreGraphSource, fake backend on a real EventQueue, default die point
of contract §3.5). The Timer/Memory mixed plan is a real compiled `norm`
plan driven to Done.

The contention-off oracle is an in-test full-speed driver of the same plan
(same shape as the execution parity driver: Timer -> advance(ns),
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
  T1. immediate completion with finish timestamp 0: a zero-work endpoint is
     terminal at submit time, its finish_ns wire field is exactly 0, and the
     deferred notification fires exactly once on the next Sys tick.
  T2. callback re-entry guard: a 1 ns endpoint submitted from inside a
     completion callback is not notified in the same dispatch round and
     fires exactly once on the following tick.
  T3. absolute capacity update through the production bridge: non-finite,
     negative and out-of-range rates are rejected as invalid_argument before
     any live state moves; halving HBM0 retimes the completion to the new
     finish tick (the previously armed completion event is invalidated and
     re-armed), still firing exactly one terminal.

Build: cmake target
      AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/PipelineKernelContract.hh"
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
#include <limits>
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

// No-op remote-memory API (this repository's Sys ctor requires one).
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

    // [C1 step 3, 2026-10-17] Delegate the cancellable pair to the shared
    // real EventQueue so Sys direct mounting works under this fake backend
    // (legacy invalid-handle fallback is retired, register now fails
    // closed without these).
    [[nodiscard]] CancellableScheduleHandle sim_schedule_cancellable(
        timespec_t delta, void (*fun_ptr)(void*), void* fun_arg,
        ScheduleCancellationCallback cancellation_cleanup) override {
        const timespec_t now = sim_get_time();
        return make_cancellable_schedule_handle(
            queue_->schedule_event_cancellable(
                now.time_val + delta.time_val, fun_ptr, fun_arg,
                cancellation_cleanup));
    }

    [[nodiscard]] bool sim_cancel_event(
        CancellableScheduleHandle& handle) override {
        if (!owns_cancellable_schedule_handle(handle)) {
            handle.reset();
            return false;
        }
        const bool cancelled =
            queue_->cancel_event(cancellable_schedule_event_handle(handle));
        handle.reset();
        return cancelled;
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
    "mesh-nx": 23, "mesh-ny": 23,
    "compute-area-mm2": 800,
    "hbm-per-side": 4,
    "hbm-sides": 1,
    "noc-width-bits": 512
  }
})";

nlohmann::json kernel_contract_wire() {
    return {
        {"projection_abi_version", 1u},
        {"descriptor_version", 1u},
        {"pipeline_protocol_version", 1u},
        {"model_version", "v8-two-actions-20261015"},
        {"timer_whitelist_version", "timer-whitelist-v1-20261005"},
        {"prediction_config_sha256",
         "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}};
}

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
    fixture.sys->workload->set_pipeline_kernel_contract(
        PipelineKernelContract::parse(kernel_contract_wire()));
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
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "hbm-bandwidth-contention": 0,
  "local-mem-latency": 100
})");
    write_text(config_dir + "/die_no_contention_no_latency.json",
               R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "hbm-bandwidth-contention": 0
})");
    write_text(config_dir + "/die_contention.json",
               R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 100
})");
    std::filesystem::path repo(__FILE__);
    for (unsigned i = 0; i < 5; ++i) repo = repo.parent_path();
    std::ifstream coarse_hardware_in(
        repo / "extern/die_model/coarse/configs/example_hardware.json");
    nlohmann::json coarse_hardware;
    coarse_hardware_in >> coarse_hardware;
    write_text(config_dir + "/coarse_hardware.json", coarse_hardware.dump());
    write_text(config_dir + "/coarse_system_latency.json", nlohmann::json{
        {"scheduling-policy", "FIFO"},
        {"compute-cost-model", "die"},
        {"die-model-profile", "coarse_die_v2"},
        {"die-model-config", config_dir + "/coarse_hardware.json"},
        {"local-mem-latency", 100}}.dump());

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
        expect_true(workload.hw_resource->tics_gpu_ops == span,
                    "A: tics_gpu_ops equals the wall span");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 0,
                    "A: GPU comp slot released after Done");
        expect_true(
            !workload.die_model_adapter()->has_active_die_executions(),
            "A: adapter drains after Done");
    }

    // ---- Case F: online run contract installed, real DieExecution exact --
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "die_no_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_die_node(26, 1));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);

        const nlohmann::json quality =
            workload.die_model_adapter()->quality_json();
        expect_true(g_terminal_fires.at(26) == 1,
                    "F: online DIE terminal fires once under the frozen run "
                    "contract");
        expect_true(g_first_tick.at(26) == oracle_one_layer + 100,
                    "F: the frozen contract keeps the exact mixed-plan "
                    "runtime");
        expect_true(quality.at("completed_execution_count") == 1,
                    "F: completed online DIE execution is accounted");
        expect_true(quality.at("logical_event_count").get<uint64_t>() > 0 &&
                        quality.at("physical_step_count").get<uint64_t>() > 0,
                    "F: mixed plan discloses live DIE event counters under "
                    "the frozen contract");
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
        std::printf("E: resetting Sys with a live DieExecution\n");
        fixture.sys.reset();
        std::printf("E: draining stale outer alarms\n");
        // Stale outer alarms (the legacy fake backend cannot cancel
        // physically) must drain through the retired-Sys guard harmlessly.
        drain_queue(fixture.event_queue);
        std::printf("E: stale alarms drained\n");
        expect_true(g_terminal_fires.count(25) == 0,
                    "E: teardown fires no completion for the in-flight "
                    "node (exactly-zero terminals)");
    }

    // ---- Case T1: zero-work endpoint, finish timestamp 0, one notice -----
    // A zero-byte endpoint compiles to a zero-work plan that is already
    // terminal when submit returns; issued at tick 0 its terminal timestamp
    // is exactly 0, so terminal detection must rely on the optional's
    // presence, never on the timestamp's numeric truth. The deferred
    // completion must fire exactly once on the next Sys tick.
    {
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        const auto& performance =
            bridge.hardware().input.at("performance_profile");
        uint64_t notified = 0;
        uint64_t notify_tick = 0;
        uint64_t finish_wire = 1;  // sentinel; the callback must overwrite 0
        bool state_wire = false;
        bool terminal_wire = false;
        const coarse_die::Owner owner{33, 33, 33, 1};
        const nlohmann::json zero_work_endpoint{
            {"endpoint_id", 3301},
            {"transaction_id", 3302},
            {"owner_kind", "die"},
            {"direction", "read"},
            {"physical_bytes", 0},
            {"path_class", "hbm_to_edge"},
            {"eta_hbm", performance.at("eta_hbm_matrix")},
            {"route", {{"internal_route", false}}}};
        const uint64_t handle = bridge.submit_endpoint(
            zero_work_endpoint, owner,
            [&](uint64_t, const coarse_die::TerminalSummary& progress) {
                ++notified;
                notify_tick = Sys::boostedTick();
                state_wire = progress.terminal && !progress.cancelled;
                terminal_wire = progress.terminal;
                finish_wire = progress.finish_ns;
            });
        expect_true(handle != 0 && notified == 0 &&
                        bridge.has_active_executions(),
                    "T1: submit-time terminal stays deferred without an "
                    "immediate callback");
        drain_queue(fixture.event_queue);
        expect_true(notified == 1 && notify_tick == 1 && state_wire &&
                        terminal_wire && finish_wire == 0 &&
                        !bridge.has_active_executions(),
                    "T1: timestamp-0 terminal notifies exactly once on the "
                    "next tick");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(notified == 1,
                    "T1: teardown after the timestamp-0 notice stays silent");
    }

    // ---- Case T2: callback re-entry defers nested submits one tick -------
    // A handle submitted from inside a completion callback must not be
    // notified inside the same dispatch round: the ready snapshot and the
    // dispatch guard keep it out of this ring, and its own 1 ns flight
    // notifies exactly once on the following tick.
    {
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        const auto& performance =
            bridge.hardware().input.at("performance_profile");
        uint64_t outer_notified = 0;
        uint64_t inner_notified = 0;
        uint64_t outer_tick = 0;
        uint64_t inner_tick = 0;
        bool inner_submitted = false;
        bool inner_deferred_in_round = false;
        const coarse_die::Owner owner{34, 34, 34, 1};
        const auto one_ns_endpoint =
            [&performance](const uint64_t endpoint_id,
                           const uint64_t transaction_id) {
                return nlohmann::json{
                    {"endpoint_id", endpoint_id},
                    {"transaction_id", transaction_id},
                    {"owner_kind", "die"},
                    {"direction", "read"},
                    {"physical_bytes", 1},
                    {"path_class", "hbm_to_edge"},
                    {"eta_hbm", performance.at("eta_hbm_matrix")},
                    {"route", {{"internal_route", false}}}};
            };
        const uint64_t outer = bridge.submit_endpoint(
            one_ns_endpoint(3401, 3402), owner,
            [&](uint64_t, const coarse_die::TerminalSummary&) {
                ++outer_notified;
                outer_tick = Sys::boostedTick();
                const uint64_t inner = bridge.submit_endpoint(
                    one_ns_endpoint(3403, 3404), owner,
                    [&](uint64_t, const coarse_die::TerminalSummary&) {
                        ++inner_notified;
                        inner_tick = Sys::boostedTick();
                    });
                inner_submitted = inner != 0;
                inner_deferred_in_round =
                    inner_notified == 0 && bridge.has_active_executions();
            });
        expect_true(outer != 0, "T2: outer 1 ns endpoint accepted");
        drain_queue(fixture.event_queue);
        expect_true(outer_notified == 1 && inner_submitted &&
                        inner_deferred_in_round && inner_notified == 1 &&
                        inner_tick == outer_tick + 1,
                    "T2: nested submit notifies once on the tick after its "
                    "callback round");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(outer_notified == 1 && inner_notified == 1,
                    "T2: teardown publishes no duplicate terminals");
    }

    // ---- Case T3: absolute capacity update through the production bridge ---
    // A read endpoint on the local HBM edge path serves half its bytes on
    // HBM0, so HBM0 is the binding resource. Running the same endpoint at the
    // full rate and at half the rate must terminate at the retimed tick: if
    // the stale completion event were not invalidated and re-armed, the old
    // full-rate event would fire early and both runs would share a tick.
    const auto run_edge_endpoint =
        [&config_dir](const bool slow_down, bool& rejected_rates) -> uint64_t {
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        const auto& performance =
            bridge.hardware().input.at("performance_profile");
        const double hbm0_peak = bridge.hardware().rates[coarse_die::HBM0];
        uint64_t notified = 0;
        uint64_t notify_tick = 0;
        const coarse_die::Owner owner{35, 35, 35, 1};
        const nlohmann::json endpoint{
            {"endpoint_id", 3501},
            {"transaction_id", 3502},
            {"owner_kind", "die"},
            {"direction", "read"},
            {"physical_bytes", 1000000},
            {"path_class", "hbm_to_edge"},
            {"eta_hbm", performance.at("eta_hbm_matrix")},
            {"route", {{"internal_route", false}}}};
        const uint64_t handle = bridge.submit_endpoint(
            endpoint, owner,
            [&](uint64_t, const coarse_die::TerminalSummary&) {
                ++notified;
                notify_tick = Sys::boostedTick();
            });
        if (slow_down) {
            const uint64_t now = Sys::boostedTick();
            bool negative = false;
            bool not_a_number = false;
            bool infinite = false;
            bool out_of_range = false;
            try {
                bridge.update_capacity(coarse_die::HBM0, -1.0, now);
            } catch (const std::invalid_argument&) {
                negative = true;
            }
            try {
                bridge.update_capacity(coarse_die::HBM0, std::nan(""), now);
            } catch (const std::invalid_argument&) {
                not_a_number = true;
            }
            try {
                bridge.update_capacity(
                    coarse_die::HBM0,
                    std::numeric_limits<double>::infinity(), now);
            } catch (const std::invalid_argument&) {
                infinite = true;
            }
            try {
                bridge.update_capacity(coarse_die::resource_count, 1.0, now);
            } catch (const std::invalid_argument&) {
                out_of_range = true;
            }
            rejected_rates =
                negative && not_a_number && infinite && out_of_range;
            bridge.update_capacity(coarse_die::HBM0, hbm0_peak / 2.0, now);
        }
        drain_queue(fixture.event_queue);
        expect_true(handle != 0 && notified == 1,
                    slow_down ? "T3: slowed endpoint terminates exactly once"
                              : "T3: full-rate endpoint terminates exactly once");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(notified == 1,
                    "T3: bridge teardown publishes no extra terminal");
        return notify_tick;
    };
    bool rejected_rates = false;
    const uint64_t full_rate_tick =
        run_edge_endpoint(false, rejected_rates);
    const uint64_t half_rate_tick =
        run_edge_endpoint(true, rejected_rates);
    expect_true(rejected_rates,
                "T3: non-finite, negative and out-of-range rates are rejected "
                "as invalid_argument without touching live state");
    expect_true(half_rate_tick > full_rate_tick &&
                    half_rate_tick >= (3 * full_rate_tick) / 2,
                "T3: half HBM0 capacity retimes the completion (stale event "
                "invalidated and re-armed)");

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
