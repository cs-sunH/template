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
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"
#include "DieModel.hh"
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <unistd.h>

#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
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
std::map<uint64_t, int> g_terminal_status;

void tick_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
               const char* /*request_id*/, const char* /*stage*/,
               uint64_t /*generation*/, uint64_t tick, int status) {
    if (status != static_cast<int>(NodeTerminalStatus::Success) &&
        !((node_id == 100 || node_id == 101) &&
          status == static_cast<int>(NodeTerminalStatus::Skipped))) {
        std::fprintf(stderr,
                     "[die_execution_lifecycle_test] node %llu terminal "
                     "status %d\n",
                     static_cast<unsigned long long>(node_id), status);
        g_ok = false;
    }
    g_terminal_status[node_id] = status;
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

// Exact FACE emitter evidence from
// work/face-workspace-port-20261014/python-astra-sim-face/emitter-evidence/actual-face-emitter.json,
// nodes[1].compute.die_op_spec (source task retained verbatim; only GraphBatch rank/id are
// adapted below for this one-rank Sys fixture).
nlohmann::json face_emitter_ffn_spec_json() {
    static const char* const raw_spec = R"json({
  "schema_version": 1,
  "op_kind": "ffn_region",
  "phase": "decode",
  "dtype": "bf16",
  "model_dims": {
    "hidden": 64,
    "ffn": 128,
    "vocab": 256,
    "mlp_variant": "gelu",
    "norm_type": "layernorm",
    "position_encoding": "none",
    "num_q_heads": 8,
    "num_kv_heads": 8,
    "head_dim": 8
  },
  "tp_slice": {
    "tensor_parallel": 2,
    "tensor_parallel_rank": 0,
    "q_heads_local": 4,
    "kv_heads_local": 4,
    "ffn_local": 64,
    "vocab_local": 128
  },
  "spans": [
    {
      "request_id": "batch_train_i1_1#s0",
      "q_tokens": 1,
      "kv_tokens": 1,
      "q_start_pos": 0,
      "kv_start_pos": 0
    },
    {
      "request_id": "batch_train_i1_1#s1",
      "q_tokens": 1,
      "kv_tokens": 2,
      "q_start_pos": 1,
      "kv_start_pos": 0
    }
  ],
  "layer_count": 2,
  "weight_passes": 2,
  "kv_access": {"identity": "local"},
  "fusion_policy": "auto",
  "provenance": "transformer_pass_aggregated",
  "config_hash": "3321c0cd5543414ad55b4601497027130d0e0a32994a7e52cae3716e13424cda"
})json";
    return nlohmann::json::parse(raw_spec);
}

OnlineNode parse_face_emitter_ffn_node() {
    const nlohmann::json raw_spec = face_emitter_ffn_spec_json();
    nlohmann::json node{
        {"rank", 0},
        {"id", 7},
        {"name", "batch_train_i1_1_all_layers_ffn_region"},
        {"type", 4},
        {"is_cpu_op", false},
        {"is_timer_op", false},
        {"inputs_values", ""},
        {"request_id", "batch_train_i1_1"},
        {"stage", "decode"},
        {"generation", 1},
        {"compute", {{"num_ops", 67584u},
                      {"tensor_size", 68608u},
                      {"runtime_ns", 0u},
                      {"die_op_spec", raw_spec}}},
        {"comm", {{"bytes", 0u}, {"src", 0}, {"dst", 0}, {"tag", 0u}}},
        {"coll", {{"comm_type", 0u}, {"bytes", 0u}, {"priority", 0u},
                   {"pg_name", ""}, {"involved_dim", nlohmann::json::array()}}}};
    nlohmann::json response;
    response["schema_version"] = 1u;
    response["source_delivery_sequence"] = 1u;
    response["nodes"] = nlohmann::json::array({node});

    // This repo's real legacy GraphBatch parser retains the raw die_op_spec
    // and parses its schema_version=1 diagnostic form independently.
    const ParsedGraphBatch parsed = parse_graph_batch(response, 1, 1);
    if (parsed.nodes.size() != 1 ||
        parsed.nodes.front().node.compute.die_task_json == nullptr ||
        parsed.nodes.front().node.compute.die_spec == nullptr ||
        *parsed.nodes.front().node.compute.die_task_json != raw_spec) {
        throw std::runtime_error(
            "FACE emitter compatibility fixture did not parse both typed views");
    }
    OnlineNode result = parsed.nodes.front().node;
    // NodeStore IDs are external GraphSource metadata, independent of the
    // producer's per-rank JSON ID.
    result.global_id = 907;
    return result;
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
                     const std::string& system_file, const int rank = 0) {
    Fixture fixture;
    fixture.event_queue = std::make_shared<EventQueue>();
    fixture.network_api =
        std::make_unique<FakeQueueNetworkApi>(fixture.event_queue);
    auto source = std::make_shared<NodeStoreGraphSource>();
    fixture.source = source.get();
    fixture.sys = std::make_unique<Sys>(
        rank, config_dir + "/workload", config_dir + "/comm_group.json",
        config_dir + "/" + system_file, &fixture.remote_mem,
        fixture.network_api.get(),
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
    return fixture;
}

std::vector<nlohmann::json> capture_metric_records(
    const std::string& manifest_path, const std::vector<Sys*>& systems,
    const uint64_t end_tick, const int restore_issue_rank = -1,
    const uint64_t restore_issue_bytes = 0) {
    FILE* const capture = std::tmpfile();
    if (capture == nullptr) {
        throw std::runtime_error("cannot create MetricCollector capture file");
    }
    const int saved_stdout = ::dup(STDOUT_FILENO);
    std::fflush(stdout);
    if (saved_stdout < 0 ||
        ::dup2(::fileno(capture), STDOUT_FILENO) < 0) {
        if (saved_stdout >= 0) ::close(saved_stdout);
        std::fclose(capture);
        throw std::runtime_error("cannot redirect stdout for metrics");
    }
    try {
        MetricCollector::instance().initialize(manifest_path, "summary");
        if (restore_issue_rank >= 0 && restore_issue_bytes > 0) {
            MetricCollector::instance().on_local_hbm_restore_issue(
                restore_issue_rank, restore_issue_bytes);
        }
        MetricCollector::instance().finalize(systems, end_tick);
    } catch (...) {
        std::fflush(stdout);
        ::dup2(saved_stdout, STDOUT_FILENO);
        ::close(saved_stdout);
        std::fclose(capture);
        throw;
    }
    std::fflush(stdout);
    ::dup2(saved_stdout, STDOUT_FILENO);
    ::close(saved_stdout);
    std::rewind(capture);
    std::string output;
    char buffer[4096];
    while (const size_t n = std::fread(buffer, 1, sizeof(buffer), capture)) {
        output.append(buffer, n);
    }
    std::fclose(capture);

    std::vector<nlohmann::json> records;
    std::istringstream lines(output);
    std::string line;
    while (std::getline(lines, line)) {
        if (line.rfind("[METRIC] ", 0) == 0) {
            records.push_back(nlohmann::json::parse(line.substr(9)));
        }
    }
    return records;
}

void install_workspace_snapshot(Sys* const sys, const uint64_t capacity_bytes,
                                const uint64_t resident_kv_bytes) {
    if (sys == nullptr || sys->workspace_admission == nullptr) {
        throw std::runtime_error(
            "coarse lifecycle fixture has no workspace ledger");
    }
    WorkspaceAdmission& ledger = *sys->workspace_admission;
    const WorkspaceCapacityRank snapshot{
        sys->id, ledger.workspace_revision(), capacity_bytes, 0,
        resident_kv_bytes, 0};
    ledger.validate_capacity_snapshot(snapshot);
    ledger.install_capacity_snapshot_unchecked(snapshot);
}

void drain_queue(const std::shared_ptr<EventQueue>& queue) {
    while (!queue->finished()) {
        queue->proceed();
    }
}

void test_prepared_coarse_preflight(const std::string& config_dir) {
    g_first_tick.clear();
    g_terminal_fires.clear();
    g_terminal_status.clear();
    Fixture fixture = make_fixture(config_dir, "coarse_system.json");
    const nlohmann::json task{{"op_kind", "ffn_region"},
        {"tensor_shapes", {{"m", 64}, {"hidden", 128},
                            {"ffn", 256}, {"mlp_variant", "swiglu"}}},
        {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"}};
    OnlineNode comp;
    comp.global_id = 900; comp.rank = 0; comp.generation = 7;
    comp.kind = NodeKind::Compute; comp.node_type = 4;
    comp.name = "coarse_preflight";
    comp.request_id = "r-preflight"; comp.stage = "prefill";
    comp.compute.die_task_json =
        std::make_shared<const nlohmann::json>(task);
    fixture.source->store().add_node(comp);
    const NodeView* const stored = fixture.source->lookup_ptr(comp.global_id);
    if (stored == nullptr) {
        expect_true(false, "P: preflight fixture node is addressable");
        return;
    }

    DieModelAdapter& adapter = *fixture.sys->workload->die_model_adapter();
    const auto prepared = adapter.prepare_coarse_plan(*stored);
    const uint64_t canonical_bytes =
        coarse_die::hbm_workspace_peak_bytes(*prepared.plan);
    expect_true(canonical_bytes > 1,
                "P: preflight fixture Plan has a positive workspace peak");
    if (canonical_bytes <= 1) return;

    install_workspace_snapshot(
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0);
    auto wrong_bytes_prepared = prepared;
    wrong_bytes_prepared.workspace_bytes = canonical_bytes - 1;
    const uint64_t revision_before_wrong_acquire =
        fixture.sys->workspace_admission->workspace_revision();
    bool wrong_prepared_bytes_rejected = false;
    try {
        (void)adapter.try_acquire_workspace(wrong_bytes_prepared);
    } catch (const std::runtime_error&) {
        wrong_prepared_bytes_rejected = true;
    }
    expect_true(wrong_prepared_bytes_rejected &&
                    fixture.sys->workspace_admission->active_lease_count() == 0 &&
                    fixture.sys->workspace_admission->active_workspace_bytes() == 0 &&
                    fixture.sys->workspace_admission->workspace_revision() ==
                        revision_before_wrong_acquire,
                "P: noncanonical Prepared bytes fail before ledger mutation");

    WorkspaceAdmission& local_ledger = *fixture.sys->workspace_admission;
    auto local = local_ledger.try_acquire(
        comp.global_id, comp.generation, canonical_bytes);
    WorkspaceAdmission foreign_ledger(fixture.sys->id);
    const WorkspaceCapacityRank foreign_capacity{
        fixture.sys->id, foreign_ledger.workspace_revision(),
        std::numeric_limits<uint64_t>::max(), 0, 0, 0};
    foreign_ledger.validate_capacity_snapshot(foreign_capacity);
    foreign_ledger.install_capacity_snapshot_unchecked(foreign_capacity);
    auto foreign = foreign_ledger.try_acquire(
        comp.global_id, comp.generation, canonical_bytes);
    const bool both_acquired =
        local.status == WorkspaceAdmission::AcquireStatus::Acquired &&
        foreign.status == WorkspaceAdmission::AcquireStatus::Acquired;
    expect_true(both_acquired,
                "P: independent same-rank ledgers admit the same fixture key");
    if (!both_acquired) return;
    const auto& local_key = local.lease.key();
    const auto& foreign_key = foreign.lease.key();
    const bool full_key_collision =
        local_key.rank == foreign_key.rank &&
        local_key.node_id == foreign_key.node_id &&
        local_key.generation == foreign_key.generation &&
        local_key.lease_id == foreign_key.lease_id;
    bool foreign_rejected = false;
    try {
        fixture.sys->workload->issue_prepared_coarse(
            *stored, prepared, std::move(foreign.lease));
    } catch (const std::runtime_error&) {
        foreign_rejected = true;
    }
    std::vector<uint64_t> free_ids;
    fixture.source->store().fill_free_node_snapshot(free_ids);
    expect_true(full_key_collision && foreign_rejected &&
                    local_ledger.active_lease_count() == 1 &&
                    local_ledger.active_workspace_bytes() == canonical_bytes &&
                    foreign_ledger.active_lease_count() == 0 &&
                    foreign_ledger.active_workspace_bytes() == 0 &&
                    std::find(free_ids.begin(), free_ids.end(), comp.global_id) !=
                        free_ids.end() &&
                    fixture.sys->workload->hw_resource
                            ->num_in_flight_gpu_comp_ops == 0 &&
                    g_terminal_fires.count(comp.global_id) == 0,
                "P: full-key collision from a foreign same-rank ledger is rejected before take/occupy");
    local.lease.release();

    install_workspace_snapshot(
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0);
    auto short_lease = local_ledger.try_acquire(
        comp.global_id, comp.generation, canonical_bytes - 1);
    auto forged_bytes_prepared = prepared;
    forged_bytes_prepared.workspace_bytes = canonical_bytes - 1;
    bool forged_bytes_rejected = false;
    if (short_lease.status == WorkspaceAdmission::AcquireStatus::Acquired) {
        try {
            fixture.sys->workload->issue_prepared_coarse(
                *stored, forged_bytes_prepared, std::move(short_lease.lease));
        } catch (const std::runtime_error&) {
            forged_bytes_rejected = true;
        }
    }
    free_ids.clear();
    fixture.source->store().fill_free_node_snapshot(free_ids);
    expect_true(forged_bytes_rejected &&
                    local_ledger.active_lease_count() == 0 &&
                    local_ledger.active_workspace_bytes() == 0 &&
                    std::find(free_ids.begin(), free_ids.end(), comp.global_id) !=
                        free_ids.end() &&
                    fixture.sys->workload->hw_resource
                            ->num_in_flight_gpu_comp_ops == 0 &&
                    g_terminal_fires.count(comp.global_id) == 0,
                "P: mismatched Plan/Prepared/lease byte count fails before take/occupy");

    install_workspace_snapshot(
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0);
    auto stale_task_lease = adapter.try_acquire_workspace(prepared);
    bool stale_task_rejected = false;
    if (stale_task_lease.status ==
        WorkspaceAdmission::AcquireStatus::Acquired) {
        NodeView stale_node = *stored;
        nlohmann::json replacement = *stored->compute.die_task_json;
        replacement["tensor_shapes"]["m"] = 65;
        stale_node.compute.die_task_json =
            std::make_shared<const nlohmann::json>(replacement);
        try {
            fixture.sys->workload->issue_prepared_coarse(
                stale_node, prepared, std::move(stale_task_lease.lease));
        } catch (const std::runtime_error&) {
            stale_task_rejected = true;
        }
    }
    free_ids.clear();
    fixture.source->store().fill_free_node_snapshot(free_ids);
    expect_true(stale_task_rejected &&
                    local_ledger.active_lease_count() == 0 &&
                    local_ledger.active_workspace_bytes() == 0 &&
                    std::find(free_ids.begin(), free_ids.end(), comp.global_id) !=
                        free_ids.end() &&
                    fixture.sys->workload->hw_resource
                            ->num_in_flight_gpu_comp_ops == 0 &&
                    g_terminal_fires.count(comp.global_id) == 0,
                "P: stale task payload with the same node/generation is rejected before take/occupy");

    fixture.sys.reset();
    drain_queue(fixture.event_queue);
}

void test_legacy_parsed_coarse_lifecycle(const std::string& config_dir) {
    g_first_tick.clear();
    g_terminal_fires.clear();
    g_terminal_status.clear();
    Fixture fixture = make_fixture(config_dir, "coarse_system.json");
    const nlohmann::json raw_task = face_emitter_ffn_spec_json();
    OnlineNode comp = parse_face_emitter_ffn_node();
    fixture.source->store().add_node(comp);
    const NodeView* const stored = fixture.source->lookup_ptr(comp.global_id);
    if (stored == nullptr) {
        expect_true(false, "O: parsed FACE emitter node is addressable in NodeStore");
        return;
    }

    expect_true(stored->compute.die_task_json != nullptr &&
                    stored->compute.die_spec != nullptr &&
                    *stored->compute.die_task_json == raw_task &&
                    stored->compute.die_spec->op_kind ==
                        DieModel::OpKind::FfnRegion,
                "O: real ParsedGraphBatch retains raw old FFN task and parsed legacy diagnostic");

    install_workspace_snapshot(
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0);
    DieModelAdapter& adapter = *fixture.sys->workload->die_model_adapter();
    auto prepared = adapter.prepare_coarse_plan(*stored);
    const auto prepared_plan = prepared.plan;
    const std::string prepared_fingerprint = prepared_plan->fingerprint;
    const uint64_t canonical_bytes =
        coarse_die::hbm_workspace_peak_bytes(*prepared_plan);
    expect_true(prepared.task != nullptr && *prepared.task == raw_task &&
                    prepared_fingerprint.size() == 64 &&
                    canonical_bytes == 256,
                "O: coarse prepare compiles the unchanged emitter task to its canonical 256-byte Plan");

    auto admission = adapter.try_acquire_workspace(prepared);
    const bool acquired = admission.status ==
        WorkspaceAdmission::AcquireStatus::Acquired;
    expect_true(acquired, "O: prepared canonical Plan acquires its workspace lease");
    if (!acquired) return;

    fixture.sys->workload->issue_prepared_coarse(
        *stored, std::move(prepared), std::move(admission.lease));
    expect_true(fixture.sys->workload->hw_resource
                            ->num_in_flight_gpu_comp_ops == 1 &&
                    fixture.sys->workspace_admission->active_lease_count() == 1 &&
                    fixture.sys->workspace_admission->active_workspace_bytes() ==
                        canonical_bytes,
                "O: prepared issue occupies the GPU and holds the exact lease");

    // At the current tick this exports the live Runtime identity without
    // advancing simulated service; the accepted Runtime Plan must have the
    // same canonical fingerprint as the exact Prepared Plan passed to issue.
    const nlohmann::json live =
        fixture.sys->coarse_die_runtime->export_snapshot(Sys::boostedTick());
    const auto& executions = live.at("runtime_snapshot").at("executions");
    const bool same_plan_executed = executions.size() == 1 &&
        executions.front().at("plan").at("fingerprint") ==
            prepared_fingerprint;
    expect_true(same_plan_executed,
                "O: Runtime accepted and runs the same canonical Plan prepared before acquire");

    drain_queue(fixture.event_queue);
    const auto served = fixture.sys->coarse_die_runtime->counters_at(
        Sys::boostedTick()).at("served_by_category").at("comp");
    expect_true(g_terminal_fires[comp.global_id] == 1 &&
                    g_terminal_status[comp.global_id] ==
                        static_cast<int>(NodeTerminalStatus::Success),
                "O: parsed coarse COMP publishes one successful graph terminal");
    expect_true(fixture.sys->workload->hw_resource
                            ->num_in_flight_gpu_comp_ops == 0 &&
                    fixture.sys->workspace_admission->active_lease_count() == 0 &&
                    fixture.sys->workspace_admission->active_workspace_bytes() == 0 &&
                    !adapter.has_active_die_executions() &&
                    !fixture.sys->coarse_die_runtime->has_active_executions(),
                "O: finish releases GPU occupancy, its workspace lease, and Runtime ownership");
    expect_true(served.at("int4_ops").get<double>() > 0.0 &&
                    served.at("vec_eqops").get<double>() > 0.0,
                "O: Runtime served typed FFN work from the submitted Plan");
    fixture.sys.reset();
    drain_queue(fixture.event_queue);
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
                [&](uint64_t, const nlohmann::json&) { ++callbacks; });
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
                    "Q: actual Workload preflight fixture owns a canonical token");
        bool issue_rejected = false;
        if (issued.status == WorkspaceAdmission::AcquireStatus::Acquired) {
            try {
                workload.issue_prepared_coarse(*stored, bypass,
                                               std::move(issued.lease));
            } catch (const std::exception& error) {
                issue_rejected = availability_error(error);
            }
        }
        expect_true(issue_rejected && unchanged_execution() &&
                        ledger.active_lease_count() == 0 &&
                        ledger.active_workspace_bytes() == 0,
                    "Q: actual Workload issue rejects manual bypass before graph/GPU/runtime changes");
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

    // ---- Case K: coarse workspace blocks, leases, cancels and retries ----
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        g_terminal_status.clear();
        write_coarse_system_config(config_dir);
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        Fixture idle_rank = make_fixture(config_dir, "coarse_system.json", 1);
        auto& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        const std::string metric_manifest_path =
            config_dir + "/coarse_metric_manifest.json";
        write_text(metric_manifest_path, nlohmann::json{
            {"schema_version", 1},
            {"repo_variant", "coarse-metric-fixture"},
            {"run_mode", "service"},
            {"run_id", "coarse-metric-fixture"},
            {"requests", nlohmann::json::array()},
            {"node_events_by_rank", nlohmann::json::object()},
            {"planner_memory_peaks", nlohmann::json::array()},
            {"memory_actions", nlohmann::json::array()}}.dump());

        const auto zero_records = capture_metric_records(
            metric_manifest_path,
            {fixture.sys.get(), idle_rank.sys.get()}, 0);
        std::map<int, nlohmann::json> zero_rank_compute;
        std::map<int, nlohmann::json> zero_local_hbm;
        for (const auto& record : zero_records) {
            const std::string type = record.value("type", std::string());
            if (type == "rank_compute") {
                zero_rank_compute[record.at("rank").get<int>()] = record;
            } else if (type == "local_hbm") {
                zero_local_hbm[record.at("rank").get<int>()] = record;
            }
        }
        bool zero_rank_counters = zero_rank_compute.size() == 2 &&
            zero_local_hbm.size() == 2;
        for (const auto& [rank, record] : zero_rank_compute) {
            const auto& coarse = record.at("coarse_die_runtime");
            zero_rank_counters = zero_rank_counters &&
                record.at("utilization_window_ns").get<uint64_t>() == 0 &&
                coarse.at("utilization_window_ns").get<uint64_t>() == 0 &&
                record.at("die_cancelled_comp_count").get<uint64_t>() == 0 &&
                exact_number(coarse.at("served_by_category").at("comp")
                                 .at("int4_ops_exact_decimal")) == 0.0L &&
                exact_number(coarse.at("served_by_category").at("comp")
                                 .at("vec_eqops_exact_decimal")) == 0.0L;
            const auto& hbm = zero_local_hbm.at(rank);
            zero_rank_counters = zero_rank_counters &&
                hbm.at("utilization_window_ns").get<uint64_t>() == 0 &&
                hbm.at("hbm_shared_definition") ==
                    hbm.at("shared_definition") &&
                hbm.at("local_hbm_restore_bytes_issued").get<uint64_t>() == 0 &&
                hbm.at("dram_bw_util").is_null() &&
                exact_number(hbm.at("served_by_category").at("comp")
                                 .at("int4_ops_exact_decimal")) == 0.0L &&
                sum_hbm_matrix(
                    hbm.at("served_by_category").at("comp")
                        .at("hbm_served_bytes_by_row_direction_exact")) ==
                    0.0L;
        }
        expect_true(zero_rank_counters,
                    "K: zero coarse typed/category counters emit rank-scoped zero-window records");

        nlohmann::json task{{"op_kind", "ffn_region"},
            {"tensor_shapes", {{"m", 64}, {"hidden", 128},
                               {"ffn", 256}, {"mlp_variant", "swiglu"}}},
            {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"}};
        const auto plan = bridge.compile(task);
        const uint64_t workspace_bytes =
            coarse_die::hbm_workspace_peak_bytes(*plan);
        expect_true(workspace_bytes > 0,
                    "K: fixture Plan has positive transient workspace");
        if (workspace_bytes > 0) {
            OnlineNode comp;
            comp.global_id = 100;
            comp.rank = 0;
            comp.kind = NodeKind::Compute;
            comp.node_type = 4;
            comp.name = "coarse_workspace_cancel";
            comp.request_id = "r1";
            comp.stage = "prefill";
            comp.compute.die_task_json =
                std::make_shared<const nlohmann::json>(task);

            // Separate pressure fixture: two independent positive-workspace
            // Plans remain ready and blocked in one complete free-node pass.
            // Add them in reverse id order to verify the snapshot's ascending
            // traversal and that the diagnostic keeps the last observation.
            Fixture pressure_fixture =
                make_fixture(config_dir, "coarse_system.json", 2);
            auto& pressure_workload = *pressure_fixture.sys->workload;
            const auto pressure_plan =
                pressure_fixture.sys->coarse_die_runtime->compile(task);
            const uint64_t pressure_workspace_bytes =
                coarse_die::hbm_workspace_peak_bytes(*pressure_plan);
            OnlineNode pressure_last = comp;
            pressure_last.global_id = 902;
            pressure_last.name = "coarse_workspace_pressure_last";
            OnlineNode pressure_first = comp;
            pressure_first.global_id = 901;
            pressure_first.name = "coarse_workspace_pressure_first";
            pressure_fixture.source->store().add_node(pressure_last);
            pressure_fixture.source->store().add_node(pressure_first);
            install_workspace_snapshot(pressure_fixture.sys.get(),
                                       pressure_workspace_bytes + 1, 2);
            pressure_workload.issue_dep_free_nodes();
            std::vector<uint64_t> pressure_free_ids;
            pressure_fixture.source->store().fill_free_node_snapshot(
                pressure_free_ids);
            const auto pressure_diagnostic =
                pressure_workload.blocked_workspace_plan_diagnostic();
            const OnlineNode* const last_pressure_node =
                pressure_fixture.source->store().node_ptr(902);
            expect_true(pressure_workspace_bytes == workspace_bytes &&
                            pressure_free_ids ==
                                std::vector<uint64_t>{901, 902} &&
                            pressure_diagnostic.has_blocked_plan &&
                            pressure_diagnostic.blocked_plan_count == 2 &&
                            pressure_diagnostic.node_id == 902 &&
                            last_pressure_node != nullptr &&
                            pressure_diagnostic.generation ==
                                last_pressure_node->generation &&
                            pressure_diagnostic.workspace_bytes ==
                                pressure_workspace_bytes &&
                            pressure_fixture.sys->workspace_admission
                                    ->active_lease_count() == 0 &&
                            pressure_workload.hw_resource
                                    ->num_in_flight_gpu_comp_ops == 0 &&
                            pressure_fixture.event_queue->finished() &&
                            pressure_fixture.event_queue
                                    ->scheduled_event_count() == 0 &&
                            !pressure_fixture.event_queue->has_deferred_work(),
                        "K: one issue pass records both independent blocked Plans and the ascending last observation without GPU, lease, or queue work");
            pressure_fixture.event_queue->schedule_event(
                101, no_op_event, nullptr);
            pressure_fixture.event_queue->proceed();
            pressure_free_ids.clear();
            pressure_fixture.source->store().fill_free_node_snapshot(
                pressure_free_ids);
            expect_true(pressure_fixture.event_queue->finished() &&
                            pressure_fixture.event_queue
                                    ->scheduled_event_count() == 0 &&
                            !pressure_fixture.event_queue->has_deferred_work() &&
                            pressure_free_ids ==
                                std::vector<uint64_t>{901, 902} &&
                            pressure_workload.hw_resource
                                    ->num_in_flight_gpu_comp_ops == 0 &&
                            pressure_fixture.sys->workspace_admission
                                    ->active_lease_count() == 0 &&
                            pressure_workload
                                    .blocked_workspace_plan_diagnostic()
                                    .blocked_plan_count == 2 &&
                            pressure_workload
                                    .blocked_workspace_plan_diagnostic()
                                    .node_id == 902,
                        "K: real queue drains and both independent Plans remain blocked with last-pass evidence intact");

            fixture.source->store().add_node(comp);

            OnlineNode child;
            child.global_id = 101;
            child.rank = 0;
            child.kind = NodeKind::Invalid;
            child.node_type = 0;
            child.name = "after_workspace_cancel";
            child.request_id = "r1";
            child.stage = "prefill";
            fixture.source->store().add_node(child);
            fixture.source->store().add_dependency(100, 101, DepKind::Data);

            // KV leaves one byte less than the prepared Plan needs. The
            // blocked node must remain in the free set without GPU occupancy.
            install_workspace_snapshot(fixture.sys.get(),
                                       workspace_bytes + 1, 2);
            workload.issue_dep_free_nodes();
            std::vector<uint64_t> free_ids;
            fixture.source->store().fill_free_node_snapshot(free_ids);
            const auto blocked_plan =
                workload.blocked_workspace_plan_diagnostic();
            const auto blocked_workspace_state =
                fixture.sys->workspace_admission->state();
            const auto blocked_capacity =
                fixture.sys->workspace_admission->capacity_snapshot_view();
            expect_true(fixture.sys->workspace_admission->active_lease_count() == 0 &&
                            workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                            std::find(free_ids.begin(), free_ids.end(), 100) !=
                                free_ids.end() &&
                            g_terminal_fires.count(100) == 0,
                        "K: blocked Plan stays free and consumes no GPU slot or lease");
            expect_true(blocked_plan.has_blocked_plan &&
                            blocked_plan.blocked_plan_count == 1 &&
                            blocked_plan.node_id == 100 &&
                            blocked_plan.generation == comp.generation &&
                            blocked_plan.workspace_bytes == workspace_bytes &&
                            fixture.sys->workspace_admission
                                    ->has_capacity_snapshot() &&
                            blocked_workspace_state.workspace_revision == 0 &&
                            blocked_workspace_state.active_workspace_bytes == 0 &&
                            blocked_capacity.workspace_revision == 0 &&
                            blocked_capacity.capacity_bytes == workspace_bytes + 1 &&
                            blocked_capacity.model_weight_bytes == 0 &&
                            blocked_capacity.resident_kv_bytes == 2 &&
                            blocked_capacity.reserved_request_bytes == 0,
                        "K: diagnostic captures the last ready blocked Plan and authoritative rank capacity view");

            // Reuse Case H's harmless queued event as a liveness-pressure
            // probe: after it drains there is no physical or deferred work
            // to wake a rank whose only ready COMP is workspace-blocked.
            fixture.event_queue->schedule_event(101, no_op_event, nullptr);
            expect_true(!fixture.event_queue->finished() &&
                            fixture.event_queue->scheduled_event_count() == 1,
                        "K: the no-op probe starts as real queued work");
            fixture.event_queue->proceed();
            free_ids.clear();
            fixture.source->store().fill_free_node_snapshot(free_ids);
            expect_true(fixture.event_queue->finished() &&
                            !fixture.event_queue->has_deferred_work() &&
                            fixture.event_queue->scheduled_event_count() == 0 &&
                            std::find(free_ids.begin(), free_ids.end(), 100) !=
                                free_ids.end() &&
                            workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                            blocked_plan.has_blocked_plan &&
                            workload.blocked_workspace_plan_diagnostic()
                                    .workspace_bytes == workspace_bytes,
                        "K: ready Plan remains blocked with GPU free after the real queue drains");

            install_workspace_snapshot(fixture.sys.get(),
                                       workspace_bytes + 1, 1);
            workload.issue_dep_free_nodes();
            free_ids.clear();
            fixture.source->store().fill_free_node_snapshot(free_ids);
            expect_true(fixture.sys->workspace_admission->active_lease_count() == 1 &&
                            fixture.sys->workspace_admission
                                    ->active_workspace_bytes() == workspace_bytes &&
                            workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                            !workload.blocked_workspace_plan_diagnostic()
                                 .has_blocked_plan &&
                            std::find(free_ids.begin(), free_ids.end(), 100) ==
                                free_ids.end(),
                        "K: exact capacity acquires the lease before take/occupy");

            // Startup waiting is an event with no typed service. Wait for
            // real compute before testing partial cancellation and metrics.
            auto before_cancel = bridge.counters_at(Sys::boostedTick());
            for (unsigned events = 0; events < 32 &&
                    !fixture.event_queue->finished() &&
                    g_terminal_fires.count(100) == 0; ++events) {
                fixture.event_queue->proceed();
                before_cancel = bridge.counters_at(Sys::boostedTick());
                const auto& typed = before_cancel.at("served_by_category").at("comp");
                if (typed.at("int4_ops").get<double>() > 0.0 ||
                    typed.at("vec_eqops").get<double>() > 0.0) break;
            }
            const auto& comp_served =
                before_cancel.at("served_by_category").at("comp");
            bool saw_served_work =
                comp_served.at("int4_ops").get<double>() > 0.0 ||
                comp_served.at("vec_eqops").get<double>() > 0.0;
            for (const auto& row :
                 comp_served.at("hbm_served_bytes_by_row_direction")) {
                for (const auto& direction : row) {
                    saw_served_work =
                        saw_served_work || direction.get<double>() > 0.0;
                }
            }
            expect_true(Sys::boostedTick() > 0 && saw_served_work &&
                            g_terminal_fires.count(100) == 0,
                        "K: cancellation begins after real typed service");
            workload.die_model_adapter()->cancel_coarse_node(
                100, Sys::boostedTick());

            const auto* stats =
                fixture.source->store().mutable_online_statistics(100);
            std::vector<uint64_t> free_after_cancel;
            fixture.source->store().fill_free_node_snapshot(
                free_after_cancel);
            expect_true(g_terminal_fires[100] == 1 &&
                            g_terminal_status[100] == static_cast<int>(
                                NodeTerminalStatus::Skipped) &&
                            stats != nullptr &&
                            stats->compute_utilization.has_value() &&
                            stats->memory_utilization.has_value() &&
                            stats->vector_utilization.has_value() &&
                            workload.die_cancelled_comp_count() == 1 &&
                            std::find(free_after_cancel.begin(),
                                      free_after_cancel.end(), 101) !=
                                free_after_cancel.end() &&
                            fixture.sys->workspace_admission
                                    ->active_workspace_bytes() == 0 &&
                            fixture.sys->workspace_admission
                                    ->active_lease_count() == 0 &&
                            workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                            fixture.sys->workspace_admission
                                    ->workspace_revision() == 2,
                        "K: live cancellation records typed utilization, releases once, terminalizes Skipped, and frees its dependent node");
            workload.issue_dep_free_nodes();
            expect_true(g_terminal_fires[101] == 1 &&
                            g_terminal_status[101] == static_cast<int>(
                                NodeTerminalStatus::Skipped),
                        "K: cancelled COMP releases its dependent node normally");

            const uint64_t metric_window_end = Sys::boostedTick();
            const auto served_records = capture_metric_records(
                metric_manifest_path,
                {fixture.sys.get(), idle_rank.sys.get()},
                metric_window_end, 0, 1234);
            std::map<int, nlohmann::json> served_rank_compute;
            std::map<int, nlohmann::json> served_local_hbm;
            for (const auto& record : served_records) {
                const std::string type = record.value("type", std::string());
                if (type == "rank_compute") {
                    served_rank_compute[record.at("rank").get<int>()] = record;
                } else if (type == "local_hbm") {
                    served_local_hbm[record.at("rank").get<int>()] = record;
                }
            }
            bool served_metrics_valid = metric_window_end > 0 &&
                served_rank_compute.size() == 2 &&
                served_local_hbm.size() == 2;
            if (served_metrics_valid) {
                const auto& active_rank = served_rank_compute.at(0);
                const auto& active_coarse =
                    active_rank.at("coarse_die_runtime");
                const auto& active_categories =
                    active_coarse.at("served_by_category");
                const auto& active_hbm = served_local_hbm.at(0);
                const auto& idle_rank_record = served_rank_compute.at(1);
                const auto& idle_coarse =
                    idle_rank_record.at("coarse_die_runtime");
                const auto& idle_hbm = served_local_hbm.at(1);
                served_metrics_valid =
                    active_rank.at("die_cancelled_comp_count").get<uint64_t>() == 1 &&
                    active_rank.at("utilization_window_ns").get<uint64_t>() ==
                        metric_window_end &&
                    active_coarse.at("utilization_window_ns").get<uint64_t>() ==
                        metric_window_end &&
                    active_hbm.at("utilization_window_ns").get<uint64_t>() ==
                        metric_window_end &&
                    active_hbm.at("hbm_shared_definition") ==
                        active_hbm.at("shared_definition") &&
                    active_hbm.at("local_hbm_restore_bytes_issued")
                            .get<uint64_t>() == 1234 &&
                    active_rank.at("local_hbm_restore_bytes")
                            .get<uint64_t>() == 1234 &&
                    exact_number(active_categories.at("comp")
                                     .at("int4_ops_exact_decimal")) > 0.0L &&
                    active_hbm.at("served_by_category") == active_categories &&
                    idle_rank_record.at("die_cancelled_comp_count")
                            .get<uint64_t>() == 0 &&
                    exact_number(idle_coarse.at("served_by_category")
                                     .at("comp").at("int4_ops_exact_decimal")) ==
                        0.0L &&
                    exact_number(idle_hbm.at("served_by_category")
                                     .at("comp").at("int4_ops_exact_decimal")) ==
                        0.0L &&
                    idle_hbm.at("local_hbm_restore_bytes_issued")
                            .get<uint64_t>() == 0;
            }
            expect_true(served_metrics_valid,
                        "K: two-rank metrics preserve typed serving, category totals, windows, and one live cancellation");

            // A Plan larger than capacity after weights is rejected before
            // graph take, GPU occupancy, or lease acquisition.
            install_workspace_snapshot(fixture.sys.get(),
                                       workspace_bytes - 1, 0);
            comp.global_id = 102;
            comp.name = "coarse_workspace_impossible";
            fixture.source->store().add_node(comp);
            bool impossible_rejected = false;
            try {
                workload.issue_dep_free_nodes();
            } catch (const std::runtime_error& exc) {
                impossible_rejected =
                    std::string(exc.what()).find("exceeds empty-instance") !=
                    std::string::npos;
            }
            free_ids.clear();
            fixture.source->store().fill_free_node_snapshot(free_ids);
            expect_true(impossible_rejected &&
                            std::find(free_ids.begin(), free_ids.end(), 102) !=
                                free_ids.end() &&
                            fixture.sys->workspace_admission
                                    ->active_lease_count() == 0 &&
                            fixture.sys->workspace_admission
                                    ->active_workspace_bytes() == 0 &&
                            workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                            g_terminal_fires.count(102) == 0,
                        "K: impossible Plan fails before take/occupy and leaves no lease");
        }
        idle_rank.sys.reset();
        drain_queue(idle_rank.event_queue);
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(100) == 1,
                    "K: teardown after cancellation does not duplicate terminals");
    }

    // ---- Case L: legacy shared-HBM metric projection stays stable --------
    {
        Fixture fixture = make_fixture(config_dir, "die_contention.json");
        const std::string manifest_path = config_dir + "/legacy_metric_manifest.json";
        write_text(manifest_path, nlohmann::json{
            {"schema_version", 1},
            {"repo_variant", "legacy-metric-fixture"},
            {"run_mode", "service"},
            {"run_id", "legacy-metric-fixture"},
            {"requests", nlohmann::json::array()},
            {"node_events_by_rank", nlohmann::json::object()},
            {"planner_memory_peaks", nlohmann::json::array()},
            {"memory_actions", nlohmann::json::array()}}.dump());
        const auto records = capture_metric_records(
            manifest_path, {fixture.sys.get()}, 0);
        nlohmann::json rank_record;
        nlohmann::json hbm_record;
        for (const auto& record : records) {
            const std::string type = record.value("type", std::string());
            if (type == "rank_compute") rank_record = record;
            if (type == "local_hbm") hbm_record = record;
        }
        const std::vector<std::string> legacy_hbm_fields = {
            "schema", "type", "source", "repo_variant", "run_id", "rank",
            "model", "utilization_window_ns", "utilization_window",
            "hbm_busy_ns", "hbm_busy_definition", "hbm_shared_ns",
            "hbm_shared_definition", "compute_bytes_served",
            "restore_bytes_served", "comm_read_bytes_served",
            "comm_write_bytes_served", "pool_read_bytes_served",
            "pool_write_bytes_served", "bytes_served_definition",
            "peak_concurrent_jobs", "peak_concurrent_jobs_definition",
            "redistribution_events", "redistribution_events_definition",
            "local_hbm_restore_bytes_issued", "local_hbm_bw_bytes_per_second",
            "local_hbm_bw_source", "hbm_busy_util",
            "hbm_shared_fraction_of_window", "dram_bw_util",
            "dram_bw_util_note"};
        bool old_fields_retained = !rank_record.is_null() &&
            !hbm_record.is_null() &&
            rank_record.value("die_cancelled_comp_count", uint64_t{0}) == 0 &&
            !rank_record.contains("coarse_die_runtime") &&
            hbm_record.value("source", std::string()) ==
                "shared_hbm_runtime_model" &&
            hbm_record.value("hbm_shared_definition", std::string()) ==
                "subset of hbm_busy_ns during which two or more jobs shared "
                "the rank HBM bandwidth (equal split)" &&
            hbm_record.value("hbm_busy_ns", -1.0) == 0.0 &&
            hbm_record.value("compute_bytes_served", -1.0) == 0.0 &&
            hbm_record.value("local_hbm_restore_bytes_issued", uint64_t{1}) == 0 &&
            rank_record.value("local_hbm_restore_bytes", uint64_t{1}) ==
                hbm_record.value("local_hbm_restore_bytes_issued", uint64_t{2}) &&
            hbm_record.at("dram_bw_util").is_null() &&
            hbm_record.value("dram_bw_util_note", std::string()) ==
                "undefined: zero local-mem-bw/window" &&
            !hbm_record.contains("dram_bw_util_numerator_bytes") &&
            !hbm_record.contains("dram_bw_util_denominator");
        for (const auto& field : legacy_hbm_fields) {
            old_fields_retained = old_fields_retained &&
                                  hbm_record.contains(field);
        }
        expect_true(old_fields_retained,
                    "L: non-coarse local-HBM fields and values match the preserved LRU baseline");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
    }

    test_prepared_coarse_preflight(config_dir);
    test_legacy_parsed_coarse_lifecycle(config_dir);

    write_coarse_system_config(config_dir);
    test_incomplete_coarse_prediction_boundaries(config_dir, "coarse_system.json");

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
