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
  M. partial-prefix teardown keeps physical bytes without graph completion.
  N. terminal-tick teardown keeps two finished markers without dispatching
     their ready callbacks.
  E. teardown: an in-flight execution (big plan, partially pumped) is torn
     down with its Sys without crash, without any terminal firing, and the
     stale outer alarms drain harmlessly afterwards.
  T1. immediate completion with finish timestamp 0: a zero-work endpoint is
     terminal at submit time, its finish_ns wire field is exactly 0, and the
     deferred notification fires exactly once on the next Sys tick.
  T2. callback re-entry guard: a 1 ns endpoint submitted from inside a
     completion callback is not notified in the same dispatch round and
     fires exactly once on the following tick.
  J. actual FACE-emitted typed FFN v2 workspace admission (see main).
  Q. incomplete coarse prediction boundaries (helper in this file).

Build: cmake target
      AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"
#include "DieModel.hh"
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <set>
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
std::map<uint64_t, int> g_terminal_status;
std::set<uint64_t> g_expected_skipped;

void tick_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
               const char* /*request_id*/, const char* /*stage*/,
               uint64_t /*generation*/, uint64_t tick, int status) {
    const bool expected_skip =
        status == static_cast<int>(NodeTerminalStatus::Skipped) &&
        g_expected_skipped.erase(node_id) != 0;
    if (status != static_cast<int>(NodeTerminalStatus::Success) &&
        !expected_skip) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] node %llu terminal "
                     "status %d\n",
                     static_cast<unsigned long long>(node_id), status);
        g_ok = false;
    }
    if (g_first_tick.find(node_id) == g_first_tick.end()) {
        g_first_tick[node_id] = tick;
    }
    g_terminal_status[node_id] = status;
    ++g_terminal_fires[node_id];
}

class FakeQueueNetworkApi final : public AstraNetworkAPI {
  public:
    explicit FakeQueueNetworkApi(std::shared_ptr<EventQueue> queue)
        : AstraNetworkAPI(0), queue_(std::move(queue)) {}

    void complete_network_callbacks(bool enabled) {
        complete_network_callbacks_ = enabled;
    }

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*callback)(void*), void* data) override {
        if (complete_network_callbacks_ && callback != nullptr) {
            queue_->schedule_event(queue_->get_current_time() + 1,
                                   callback, data);
        }
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
    bool complete_network_callbacks_ = false;
};

// face-LRU adaptation: this repository's Sys constructor requires a non-null
// remote memory backend (AstraRemoteMemoryAPI). The die fixtures never issue
// remote MEM nodes, so a minimal no-op backend suffices (an issue call would
// be a fixture defect and fails closed).
class FakeRemoteMemoryApi final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int, Sys*) override {}
    void issue(uint64_t, WorkloadLayerHandlerData*) override {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] "
                     "FakeRemoteMemoryApi::issue called -- the fixture never "
                     "issues remote MEM nodes\n");
        std::exit(EXIT_FAILURE);
    }
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
    std::unique_ptr<FakeRemoteMemoryApi> remote_api;
    std::unique_ptr<Sys> sys;
    NodeStoreGraphSource* source = nullptr;
};

Fixture make_fixture(const std::string& config_dir,
                     const std::string& system_file) {
    Fixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    fixture.network_api =
        std::make_unique<FakeQueueNetworkApi>(fixture.event_queue);
    fixture.remote_api = std::make_unique<FakeRemoteMemoryApi>();
    auto source = std::make_shared<NodeStoreGraphSource>();
    fixture.source = source.get();
    fixture.sys = std::make_unique<Sys>(
        0, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, fixture.remote_api.get(),
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

long double exact_json_number(const nlohmann::json& value) {
    if (value.is_string()) return std::stold(value.get<std::string>());
    return value.get<long double>();
}

long double progress_read_bytes(const nlohmann::json& progress) {
    const auto& matrix = progress.at("hbm_served_bytes_exact");
    long double read_bytes = 0.0L;
    for (const auto& row : matrix) read_bytes += exact_json_number(row.at(0));
    return read_bytes;
}

void no_op_event(void*) {}

struct HbmTeardownEventContext {
    Workload* workload = nullptr;
    CoarseDieRuntime* bridge = nullptr;
    std::vector<uint64_t> handles;
    std::vector<nlohmann::json> samples;
    bool fired = false;
};

void teardown_hbm_model_event(void* data) {
    auto* context = static_cast<HbmTeardownEventContext*>(data);
    const uint64_t now_ns = Sys::boostedTick();
    if (context->bridge != nullptr) {
        for (const uint64_t handle : context->handles) {
            context->samples.push_back(
                context->bridge->query_progress_at(handle, now_ns));
        }
    }
    context->workload->local_hbm_bandwidth_model.reset();
    context->fired = true;
}

void test_incomplete_coarse_prediction_boundaries(
    const std::string& config_dir, const std::string& system_file) {
    for (const char* field : {"prediction_status", "alone_estimate_status"}) {
        Fixture fixture = make_fixture(config_dir, system_file);
        Workload& workload = *fixture.sys->workload;
        auto& adapter = *workload.die_model_adapter();
        auto& bridge = *fixture.sys->coarse_die_runtime;
        auto& ledger = *fixture.sys->workspace_admission;
        // Make the Sys clock newer than the untouched Runtime clock. A
        // rejected submit must not silently settle the Runtime to this tick.
        const uint64_t arrival = fixture.event_queue->get_current_time() + 17;
        fixture.event_queue->schedule_event(arrival, [](void*) {}, nullptr);
        while (fixture.event_queue->get_current_time() < arrival) {
            fixture.event_queue->proceed();
        }
        const nlohmann::json task{
            {"schema", "coarse_die_task_v2"}, {"op_kind", "ffn_region"},
            {"phase", "prefill"},
            {"tensor_shapes", {{"m", 64}, {"hidden", 128}, {"ffn", 256},
                               {"mlp_variant", "swiglu"}}},
            {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"},
            {"layer_instances", 1}, {"weight_passes", 1}};
        OnlineNode comp;
        comp.global_id = 950; comp.rank = 0; comp.generation = 7;
        comp.kind = NodeKind::Compute; comp.node_type = 4;
        comp.name = "incomplete_coarse_prediction";
        comp.request_id = "availability"; comp.stage = "prefill";
        comp.compute.op_role = OpRole::Compute;
        comp.compute.die_task_json =
            std::make_shared<const nlohmann::json>(task);
        fixture.source->store().add_node(comp);
        const auto* stored = fixture.source->lookup_ptr(comp.global_id);
        expect_true(stored != nullptr,
                    "Q: availability fixture has a dependency-free typed COMP");
        if (stored == nullptr) continue;

        // This is a legal R1 task on both the old SDK and a new SDK. Do not
        // reduce production layer counts to conceal unsupported R>1 timing.
        const auto prepared = adapter.prepare_coarse_plan(*stored);
        const uint64_t peak = prepared.workspace_bytes;
        expect_true(peak > 0 && prepared.task != nullptr &&
                        *prepared.task == task,
                    "Q: legal R1 prepare preserves the task and positive workspace");
        if (peak == 0) continue;
        auto unavailable = std::make_shared<coarse_die::Plan>(*prepared.plan);
        unavailable->summary["prediction_status"] = "ok_assumed";
        unavailable->summary["alone_estimate_status"] = "ok_assumed";
        unavailable->summary[field] = "not_available";
        unavailable->fingerprint.clear();
        unavailable->fingerprint = coarse_die::plan_to_json(*unavailable)
                                        .at("fingerprint").get<std::string>();
        auto bypass = prepared;
        bypass.plan = unavailable;

        // Prove the only defect is prediction completeness. The public
        // diagnostic Runtime must still accept this canonical Plan.
        coarse_die::Runtime diagnostic(bridge.hardware());
        const coarse_die::Owner owner{1, comp.global_id, 7, 1};
        const uint64_t diagnostic_handle =
            diagnostic.submit(unavailable, owner, 0);
        expect_true(diagnostic_handle != 0 &&
                        coarse_die::plan_from_json(
                            coarse_die::plan_to_json(*unavailable))->fingerprint ==
                            unavailable->fingerprint,
                    "Q: each unavailable marker has a valid canonical Plan/hash");
        const WorkspaceCapacityRank capacity{
            fixture.sys->id, ledger.workspace_revision(), peak + 4096, 0, 0, 0};
        ledger.validate_capacity_snapshot(capacity);
        ledger.install_capacity_snapshot_unchecked(capacity);
        const auto runtime_before = bridge.counters();
        const uint64_t sys_tick_before = Sys::boostedTick();
        const auto terminals_before = g_terminal_fires;
        const auto terminal_status_before = g_terminal_status;
        std::vector<uint64_t> free_before;
        fixture.source->store().fill_free_node_snapshot(free_before);
        uint64_t callbacks = 0;
        const auto unchanged_execution = [&]() {
            std::vector<uint64_t> free_after;
            fixture.source->store().fill_free_node_snapshot(free_after);
            return bridge.counters() == runtime_before &&
                Sys::boostedTick() == sys_tick_before &&
                !bridge.has_active_executions() &&
                !adapter.has_active_die_executions() &&
                callbacks == 0 && g_terminal_fires == terminals_before &&
                g_terminal_status == terminal_status_before &&
                free_after == free_before &&
                workload.hw_resource->num_in_flight_gpu_comp_ops == 0;
        };
        const auto availability_error = [&](const std::exception& error) {
            const std::string reason = error.what();
            return reason.find("prediction_not_available:") != std::string::npos &&
                reason.find(field) != std::string::npos;
        };

        bool submit_rejected = false;
        uint64_t unexpected_handle = 0;
        try {
            unexpected_handle = bridge.submit_plan(
                unavailable, owner,
                [&](uint64_t, const coarse_die::TerminalSummary&) {
                    ++callbacks;
                });
        } catch (const std::exception& error) {
            submit_rejected = availability_error(error);
        }
        expect_true(submit_rejected && unchanged_execution() &&
                        runtime_before.at("now_ns").get<uint64_t>() < sys_tick_before,
                    "Q: submit rejects the exact unavailable marker without clock/service/callback changes");
        if (unexpected_handle != 0) {
            bridge.cancel(unexpected_handle, Sys::boostedTick(), false);
        }

        const uint64_t revision_before_acquire = ledger.workspace_revision();
        bool acquire_rejected = false;
        try {
            auto accidental = adapter.try_acquire_workspace(bypass);
            if (accidental.lease.active()) accidental.lease.release();
        } catch (const std::exception& error) {
            acquire_rejected = availability_error(error);
        }
        expect_true(acquire_rejected && unchanged_execution() &&
                        ledger.active_lease_count() == 0 &&
                        ledger.active_workspace_bytes() == 0 &&
                        ledger.workspace_revision() == revision_before_acquire,
                    "Q: manual unavailable Prepared Plan is rejected before workspace mutation");

        // A pre-existing valid token belongs to the caller. The const-ref
        // validator may reject the Plan but must leave that token untouched.
        const WorkspaceCapacityRank refreshed{
            fixture.sys->id, ledger.workspace_revision(), peak + 4096, 0, 0, 0};
        ledger.validate_capacity_snapshot(refreshed);
        ledger.install_capacity_snapshot_unchecked(refreshed);
        auto held = adapter.try_acquire_workspace(prepared);
        expect_true(held.status == WorkspaceAdmission::AcquireStatus::Acquired,
                    "Q: valid R1 Prepared Plan still acquires its canonical lease");
        if (held.status != WorkspaceAdmission::AcquireStatus::Acquired) continue;
        const uint64_t held_revision = ledger.workspace_revision();
        bool validation_rejected = false;
        try {
            adapter.validate_prepared_coarse_plan(*stored, bypass, held.lease);
        } catch (const std::exception& error) {
            validation_rejected = availability_error(error);
        }
        expect_true(validation_rejected && unchanged_execution() &&
                        held.lease.active() && ledger.owns_lease(held.lease, peak) &&
                        ledger.active_lease_count() == 1 &&
                        ledger.active_workspace_bytes() == peak &&
                        ledger.workspace_revision() == held_revision,
                    "Q: validation rejects before take/occupy and preserves the caller's live lease");
        held.lease.release();
        expect_true(ledger.active_lease_count() == 0 &&
                        ledger.active_workspace_bytes() == 0,
                    "Q: caller explicitly releases the preserved lease exactly once");

        const WorkspaceCapacityRank issue_capacity{
            fixture.sys->id, ledger.workspace_revision(), peak + 4096, 0, 0, 0};
        ledger.validate_capacity_snapshot(issue_capacity);
        ledger.install_capacity_snapshot_unchecked(issue_capacity);
        auto issued = adapter.try_acquire_workspace(prepared);
        expect_true(issued.status == WorkspaceAdmission::AcquireStatus::Acquired,
                    "Q: actual adapter issue fixture owns a canonical token");
        bool issue_rejected = false;
        if (issued.status == WorkspaceAdmission::AcquireStatus::Acquired) {
            try {
                adapter.issue_prepared_coarse_plan(*stored, bypass,
                                               std::move(issued.lease));
            } catch (const std::exception& error) {
                issue_rejected = availability_error(error);
            }
        }
        expect_true(issue_rejected && unchanged_execution() &&
                        ledger.active_lease_count() == 0 &&
                        ledger.active_workspace_bytes() == 0,
                    "Q: actual adapter issue rejects manual bypass before graph/GPU/runtime changes");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(callbacks == 0 && g_terminal_fires == terminals_before,
                    "Q: rejected predictions leave no deferred completion on teardown");
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

    std::filesystem::path repo(__FILE__);
    for (unsigned i = 0; i < 5; ++i) repo = repo.parent_path();
    std::ifstream coarse_hardware_in(
        repo / "extern/die_model/coarse/configs/example_hardware.json");
    nlohmann::json coarse_hardware;
    coarse_hardware_in >> coarse_hardware;
    write_text(config_dir + "/coarse_hardware.json", coarse_hardware.dump());
    write_text(config_dir + "/coarse_system_latency.json", nlohmann::json{
        {"compute-cost-model", "die"},
        {"die-model-profile", "coarse_die_v2"},
        {"die-model-config", config_dir + "/coarse_hardware.json"},
        {"hbm-bandwidth-contention", 1},
        {"local-mem-latency", 100}}.dump());

    const DieModel::Hardware hw =
        DieModel::make_hardware(nlohmann::json::parse(kDieNodeJson));
    const auto one_layer_spec = parse_spec(1);
    const uint64_t oracle_one_layer =
        oracle_full_speed_elapsed(one_layer_spec, hw);
    const auto reference_plan = DieModel::compile(*one_layer_spec, hw);
    const DieModel::CostResult reference =
        DieModel::estimate_alone(*reference_plan);

    // ---- Case J: actual FACE-emitted typed FFN v2 workspace admission ----
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        g_terminal_status.clear();
        g_expected_skipped.clear();
        Fixture fixture =
            make_fixture(config_dir, "coarse_system_latency.json");
        Workload& workload = *fixture.sys->workload;

        const auto fixture_path =
            repo / "astra-sim/workload/execution_driven/tests/testdata/"
                  "face_emitted_ffn_region.json";
        std::ifstream emitted_in(fixture_path);
        nlohmann::json emitted_fixture;
        emitted_in >> emitted_fixture;
        const auto& provenance = emitted_fixture.at("fixture_provenance");
        nlohmann::json emitted_node = emitted_fixture.at("emitted_node");
        const std::string emitted_task =
            emitted_node.at("compute").at("die_op_spec").dump();
        expect_true(
            provenance.at("source_sha256") ==
                    "97549bf7ea41c11174d58d6f9eb16304e057819b78156882db1b0a59b5fd9548" &&
                provenance.at("generator") ==
                    "GraphBatchBuilder._emit_train_body -> transformer_pass_aggregated" &&
                provenance.at("source_rank") == 2 &&
                provenance.at("source_node_id") == 7 &&
                emitted_node.at("name") ==
                    "batch_train_i1_1_all_layers_ffn_region" &&
                emitted_node.at("compute").at("die_op_spec").at("op_kind") ==
                    "ffn_region",
            "J: fixture retains the actual emitted FACE FFN task and source provenance");

        // Keep the compiler payload exact while placing this rank-2 emitter
        // record into the one-rank native lifecycle fixture.
        emitted_node["rank"] = 0;
        nlohmann::json response{
            {"schema_version", 2},
            {"batch_id", 0},
            {"source_delivery_sequence", 0},
            {"nodes", nlohmann::json::array({emitted_node})},
            {"workspace_capacity_snapshot",
             {{"protocol_version", 1},
              {"ranks",
               nlohmann::json::array(
                   {{{"rank", 0},
                     {"workspace_revision", 0},
                     {"capacity_bytes", 2000},
                     {"model_weight_bytes", 1000},
                     {"resident_kv_bytes", 745},
                     {"reserved_request_bytes", 0}}})}}}};
        const std::vector<WorkspaceRankState> request_state{{0, 0, 0}};
        const auto parsed = parse_graph_batch(response, 1, 2, request_state);
        OnlineNode task = parsed.nodes.at(0).node;
        task.global_id = 80;
        expect_true(parsed.schema_version == 2 &&
                        task.compute.die_task_json != nullptr &&
                        task.compute.die_spec != nullptr &&
                        task.compute.die_task_json->dump() == emitted_task &&
                        task.compute.die_task_json->at("op_kind") == "ffn_region" &&
                        task.compute.runtime_ns == 0,
                    "J: real v2 parser preserves raw FFN JSON and diagnostic DieSpec");

        WorkspaceAdmission& ledger = *fixture.sys->workspace_admission;
        const WorkspaceCapacityRank low_headroom{
            0, 0, 2000, 1000, 745, 0};  // 255 B headroom < 256 B Plan.
        ledger.validate_capacity_snapshot(low_headroom);
        ledger.install_capacity_snapshot_unchecked(low_headroom);
        fixture.source->store().add_node(task);
        const auto candidate_is_free = [&](const uint64_t node_id) {
            bool found = false;
            fixture.source->for_each_dep_free([&](const auto& candidate) {
                found = found || candidate.global_id == node_id;
            });
            return found;
        };
        workload.issue_dep_free_nodes();
        expect_true(candidate_is_free(80),
                    "J: insufficient KV headroom keeps FFN dependency-free");
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 0,
                    "J: blocked FFN does not occupy a GPU slot");
        expect_true(ledger.active_workspace_bytes() == 0 &&
                        ledger.active_lease_count() == 0 &&
                        ledger.workspace_revision() == 0,
                    "J: blocked FFN does not acquire a workspace lease");

        const WorkspaceCapacityRank released_kv_headroom{
            0, 0, 2000, 1000, 744, 0};  // 256 B exact fit.
        ledger.validate_capacity_snapshot(released_kv_headroom);
        expect_true(ledger.capacity_snapshot_changes(released_kv_headroom),
                    "J: one released KV byte changes the rank capacity view");
        ledger.install_capacity_snapshot_unchecked(released_kv_headroom);
        workload.issue_dep_free_nodes();
        expect_true(!candidate_is_free(80) &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        ledger.active_workspace_bytes() == 256 &&
                        ledger.active_lease_count() == 1 &&
                        ledger.workspace_revision() == 1,
                    "J: capacity retry acquires the canonical 256 B Plan lease");
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[80] == 1 &&
                        g_terminal_status[80] ==
                            static_cast<int>(NodeTerminalStatus::Success) &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        ledger.active_workspace_bytes() == 0 &&
                        ledger.active_lease_count() == 0 &&
                        ledger.workspace_revision() == 2,
                    "J: finish releases the FFN lease exactly once");

        OnlineNode cancelled = task;
        cancelled.global_id = 81;
        cancelled.generation = 2;
        fixture.source->store().add_node(cancelled);
        workload.issue_dep_free_nodes();
        expect_true(workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        ledger.active_workspace_bytes() == 256 &&
                        ledger.active_lease_count() == 1 &&
                        ledger.workspace_revision() == 3,
                    "J: second FFN issue owns an independent cancellation lease");
        g_expected_skipped.insert(81);
        workload.die_model_adapter()->cancel_coarse_node(
            81, Sys::boostedTick());
        expect_true(g_terminal_fires[81] == 1 &&
                        g_terminal_status[81] ==
                            static_cast<int>(NodeTerminalStatus::Skipped) &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        ledger.active_workspace_bytes() == 0 &&
                        ledger.active_lease_count() == 0 &&
                        ledger.workspace_revision() == 4 &&
                        workload.die_cancelled_comp_count() == 1,
                    "J: cancel returns the FFN lease once and terminalizes as skipped");
    }

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

    // ---- Case F: coarse control marker and real COMM share Runtime -------
    // FACE-LRU keeps the single GPU Compute gate intact. This production-path
    // overlap uses the independent COMM slot and exercises Workload::call
    // while preserving its Pool/RESTORE join behavior.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        fixture.network_api->complete_network_callbacks(true);
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        bool scalar_comp_rejected = false;
        try {
            workload.local_hbm_bandwidth_model->issue_compute(1, 1, nullptr);
        } catch (const std::runtime_error& error) {
            scalar_comp_rejected =
                std::string(error.what()).find("typed plan") !=
                std::string::npos;
        }
        expect_true(scalar_comp_rejected,
                    "F: ordinary scalar COMP remains forbidden with coarse DIE");

        OnlineNode marker;
        marker.global_id = 26; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "coarse_first_token";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        OnlineNode comm;
        comm.global_id = 27; comm.rank = 0; comm.kind = NodeKind::CommSend;
        comm.node_type = 5; comm.name = "coarse_comm_with_marker";
        comm.request_id = "r1"; comm.stage = "prefill";
        comm.comm.bytes = 1ULL << 20; comm.comm.src = 0; comm.comm.dst = 0;
        comm.comm.tag = 0; comm.comm.hbm_charge = true;
        fixture.source->store().add_node(comm);

        workload.issue_dep_free_nodes();
        const auto snapshot = bridge.export_snapshot(Sys::boostedTick());
        uint64_t die_endpoints = 0;
        uint64_t comm_endpoints = 0;
        bool marker_shape_ok = false;
        bool comm_keeps_default_wire = false;
        for (const auto& execution :
             snapshot.at("runtime_snapshot").at("executions")) {
            const auto& endpoint =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint.value("owner_kind", std::string()) == "die") {
                ++die_endpoints;
                marker_shape_ok =
                    endpoint.value("direction", std::string()) == "read" &&
                    endpoint.value("physical_bytes", uint64_t{0}) == 1 &&
                    endpoint.value("initial_delay_ns", uint64_t{0}) == 100;
            } else if (endpoint.value("owner_kind", std::string()) == "comm") {
                ++comm_endpoints;
                comm_keeps_default_wire =
                    endpoint.value("physical_bytes", uint64_t{0}) ==
                        (1ULL << 20) &&
                    !endpoint.contains("initial_delay_ns");
            }
        }
        expect_true(snapshot.at("runtime_snapshot").at("executions").size() == 2 &&
                        die_endpoints == 1 && comm_endpoints == 1 &&
                        marker_shape_ok && comm_keeps_default_wire &&
                        bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        workload.hw_resource->num_in_flight_gpu_comm_ops == 1 &&
                        g_terminal_fires.count(26) == 0 &&
                        g_terminal_fires.count(27) == 0,
                    "F: normal gates submit one delayed DIE byte and a COMM endpoint to shared Runtime");
        drain_queue(fixture.event_queue);
        const auto counters = bridge.counters();
        const auto& categories = counters.at("served_by_category");
        const auto& typed_comp = categories.at("comp");
        expect_true(g_terminal_fires[26] == 1 && g_terminal_fires[27] == 1 &&
                        g_first_tick[26] == 101 &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        workload.hw_resource->num_in_flight_gpu_comm_ops == 0 &&
                        std::abs(workload.local_hbm_bandwidth_model
                                     ->compute_bytes_served() - 1.0) < 1e-12 &&
                        typed_comp.at("int4_ops").get<double>() == 0.0 &&
                        typed_comp.at("vec_eqops").get<double>() == 0.0 &&
                        progress_read_bytes(nlohmann::json{
                            {"hbm_served_bytes_exact",
                             categories.at("die")
                                 .at("hbm_served_bytes_by_row_direction_exact")}}) ==
                            1.0L &&
                        progress_read_bytes(nlohmann::json{
                            {"hbm_served_bytes_exact",
                             categories.at("comm_send")
                                 .at("hbm_served_bytes_by_row_direction_exact")}}) ==
                            static_cast<long double>(1ULL << 20),
                    "F: marker and COMM complete once; 1-byte compatibility count stays separate from COMM");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[26] == 1 && g_terminal_fires[27] == 1,
                    "F: completed marker/COMM teardown publishes no duplicate terminals");
    }

    // ---- Case G: live marker prefix, read-only sampling, cancellation ----
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        nlohmann::json slow_hardware = coarse_hardware;
        slow_hardware["performance_profile"]["eta_hbm_matrix"] = 1e-6;
        write_text(config_dir + "/coarse_hardware_slow.json",
                   slow_hardware.dump());
        write_text(config_dir + "/coarse_system_slow.json", nlohmann::json{
            {"compute-cost-model", "die"},
            {"die-model-profile", "coarse_die_v2"},
            {"die-model-config", config_dir + "/coarse_hardware_slow.json"},
            {"hbm-bandwidth-contention", 1},
            {"local-mem-latency", 100}}.dump());
        Fixture fixture = make_fixture(config_dir, "coarse_system_slow.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 28; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "coarse_partial_marker";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        fixture.event_queue->schedule_event(101, no_op_event, nullptr);
        fixture.event_queue->proceed();
        fixture.event_queue->proceed();
        const uint64_t now = Sys::boostedTick();
        const auto envelope = bridge.export_snapshot(now);
        uint64_t marker_handle = 0;
        for (const auto& execution :
             envelope.at("runtime_snapshot").at("executions")) {
            const auto& endpoint =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint.value("owner_kind", std::string()) == "die") {
                marker_handle = execution.at("handle").get<uint64_t>();
            }
        }
        const auto live_before = bridge.query_progress(marker_handle);
        const auto counters_before = bridge.counters();
        const auto sampled = bridge.query_progress_at(marker_handle, now);
        const long double expected_prefix = progress_read_bytes(sampled);
        const double getter_before =
            workload.local_hbm_bandwidth_model->compute_bytes_served();
        const auto live_after = bridge.query_progress(marker_handle);
        const auto counters_after = bridge.counters();
        const bool reversed_time_rejected = [&] {
            try {
                (void)bridge.query_progress_at(marker_handle, now - 1);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        }();
        const bool unknown_handle_rejected = [&] {
            try {
                (void)bridge.query_progress_at(0, now);
            } catch (const std::out_of_range&) {
                return true;
            }
            return false;
        }();
        expect_true(now == 101 && marker_handle != 0 && expected_prefix > 0.0L &&
                        expected_prefix < 1.0L &&
                        std::abs(getter_before -
                                 static_cast<double>(expected_prefix)) < 1e-18 &&
                        live_before == live_after && counters_before == counters_after &&
                        bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        g_terminal_fires.count(28) == 0 &&
                        reversed_time_rejected && unknown_handle_rejected,
                    "G: live getter sees a strict prefix without moving Runtime, slot, or terminal state");
        bridge.cancel(marker_handle, now, true);
        const auto cancelled = bridge.counters();
        const long double cancelled_prefix = progress_read_bytes(nlohmann::json{
            {"hbm_served_bytes_exact",
             cancelled.at("served_by_category").at("die")
                 .at("hbm_served_bytes_by_row_direction_exact")}});
        const double getter_cancelled =
            workload.local_hbm_bandwidth_model->compute_bytes_served();
        expect_true(std::abs(cancelled_prefix - expected_prefix) < 1e-18L &&
                        std::abs(getter_cancelled - getter_before) < 1e-18 &&
                        !bridge.has_active_executions() &&
                        cancelled.at("endpoint_registry_size") == 0 &&
                        cancelled.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(28) == 0,
                    "G: cancellation closes the same physical prefix once and suppresses graph completion");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(28) == 0,
                    "G: cancelled marker teardown publishes no terminal");
    }

    // ---- Case H: HBM facade teardown retires a delayed marker cleanly ----
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        fixture.sys->local_mem_latency = 1'000'000;
        OnlineNode marker;
        marker.global_id = 29; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "coarse_teardown_marker";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        expect_true(bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        g_terminal_fires.count(29) == 0,
                    "H: marker is pending before explicit HBM teardown");
        workload.local_hbm_bandwidth_model.reset();
        const auto after_teardown = bridge.counters();
        expect_true(!bridge.has_active_executions() &&
                        after_teardown.at("endpoint_registry_size") == 0 &&
                        after_teardown.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(29) == 0,
                    "H: HBM teardown cancels and retires the marker without callback");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(29) == 0,
                    "H: retired delayed marker alarm cannot terminalize later");
    }

    // M: partial teardown retains the physical marker prefix and removes its
    // callback without terminalizing the graph node.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_slow.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 30; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "partial_teardown_marker";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        HbmTeardownEventContext teardown{&workload, &bridge};
        fixture.event_queue->schedule_event(
            101, teardown_hbm_model_event, &teardown);
        workload.issue_dep_free_nodes();
        const auto before_teardown = bridge.export_snapshot(0);
        for (const auto& execution :
             before_teardown.at("runtime_snapshot").at("executions")) {
            if (execution.at("plan").at("task").at("endpoint")
                    .value("owner_kind", std::string()) == "die") {
                teardown.handles.push_back(
                    execution.at("handle").get<uint64_t>());
            }
        }
        const auto projected = teardown.handles.empty()
            ? nlohmann::json::object()
            : bridge.query_progress_at(teardown.handles.front(), 101);
        const long double expected_prefix = projected.is_object()
            ? progress_read_bytes(projected) : 0.0L;
        while (!teardown.fired && !fixture.event_queue->finished()) {
            fixture.event_queue->proceed();
        }
        const auto counters = bridge.counters();
        const auto& die = counters.at("served_by_category").at("die");
        const long double retained_prefix = exact_json_number(
            die.at("hbm_served_bytes_by_row_direction_exact")[0][0]);
        expect_true(teardown.fired && Sys::boostedTick() == 101 &&
                        expected_prefix > 0.0L && expected_prefix < 1.0L &&
                        teardown.samples.size() == 1 &&
                        std::abs(progress_read_bytes(teardown.samples.front()) -
                                 expected_prefix) < 1e-18L &&
                        std::abs(retained_prefix - expected_prefix) < 1e-18L &&
                        !bridge.has_active_executions() &&
                        counters.at("endpoint_registry_size") == 0 &&
                        counters.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(30) == 0,
                    "M: partial teardown retains sampled DIE bytes and suppresses the marker callback");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(30) == 0,
                    "M: partial marker teardown remains silent after stale alarms drain");
    }

    // N: the tick-1 no-op was registered before the two endpoints. At that
    // tick the endpoints are already physically finished, but Runtime has
    // not dispatched their callbacks yet. Teardown must stay silent for both.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        fixture.sys->local_mem_latency = 0;
        OnlineNode first;
        first.global_id = 31; first.rank = 0; first.kind = NodeKind::Compute;
        first.node_type = 4; first.name = "finished_teardown_marker_a";
        first.request_id = "r1"; first.stage = "prefill";
        first.compute.op_role = OpRole::ControlMarker;
        first.compute.num_ops = 1; first.compute.tensor_size = 1;
        fixture.source->store().add_node(first);
        OnlineNode second = first;
        second.global_id = 32;
        second.name = "finished_teardown_marker_b";
        fixture.source->store().add_node(second);
        HbmTeardownEventContext teardown{&workload, &bridge};
        fixture.event_queue->schedule_event(
            1, teardown_hbm_model_event, &teardown);
        workload.issue_dep_free_nodes();
        // The production single-GPU gate serializes these graph nodes. Direct
        // Workload dispatch here only creates a second ready Runtime callback
        // so destructor notification behavior is tested at shared tick 1.
        const auto* second_view = fixture.source->lookup_ptr(32);
        if (second_view != nullptr) workload.issue(*second_view);
        const auto before_teardown = bridge.export_snapshot(0);
        for (const auto& execution :
             before_teardown.at("runtime_snapshot").at("executions")) {
            if (execution.at("plan").at("task").at("endpoint")
                    .value("owner_kind", std::string()) == "die") {
                teardown.handles.push_back(
                    execution.at("handle").get<uint64_t>());
            }
        }
        fixture.event_queue->proceed();
        const auto counters = bridge.counters();
        const auto& die = counters.at("served_by_category").at("die");
        const long double retained_bytes = exact_json_number(
            die.at("hbm_served_bytes_by_row_direction_exact")[0][0]);
        const bool sampled_both_finished = teardown.samples.size() == 2 &&
            std::all_of(teardown.samples.begin(), teardown.samples.end(),
                [](const nlohmann::json& sample) {
                    return sample.at("state") == "finished" &&
                           sample.at("terminal").get<bool>() &&
                           std::abs(progress_read_bytes(sample) - 1.0L) < 1e-18L;
                });
        expect_true(teardown.fired && Sys::boostedTick() == 1 &&
                        teardown.handles.size() == 2 &&
                        sampled_both_finished &&
                        std::abs(retained_bytes - 2.0L) < 1e-18L &&
                        !bridge.has_active_executions() &&
                        counters.at("endpoint_registry_size") == 0 &&
                        counters.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(31) == 0 &&
                        g_terminal_fires.count(32) == 0,
                    "N: teardown retains both finished marker bytes without dispatching either ready callback");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(31) == 0 &&
                        g_terminal_fires.count(32) == 0,
                    "N: terminal-tick marker teardown stays silent after stale alarms drain");
    }

    // ---- Case I: teardown of an in-flight execution ----------------------
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
                    "I: big node still in flight after partial pumping");
        // Tear the whole Sys down mid-flight (Workload destructor ->
        // adapter shutdown -> every DieExecution cancels its pending event
        // and drops its pending HBM die job). No completion may fire.
        fixture.sys.reset();
        // Stale outer alarms (the legacy fake backend cannot cancel
        // physically) must drain through the retired-Sys guard harmlessly.
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(25) == 0,
                    "I: teardown fires no completion for the in-flight "
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

    test_incomplete_coarse_prediction_boundaries(config_dir, "coarse_system_latency.json");

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
