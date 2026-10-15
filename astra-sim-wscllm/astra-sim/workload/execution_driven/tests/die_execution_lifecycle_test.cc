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
     comp slot is occupied during and released after the run (this repo
     removed the write-only tics_gpu_ops accumulator); the adapter's drain
     set empties.
  B. contention off + local_mem_latency 0: wall span == oracle machine
     elapsed exactly (no latency charge when the configured latency is 0).
  C. contention on: mid-flight the adapter reports an active execution and
     the GPU slot stays occupied; after the drain both clear; the span is
     at least estimate_alone + latency; the die side-band counters recorded
     service units AND actual bytes as separate quantities.
  D. marker under contention off: the legacy 1 ns event micro-job (issue
     tick + 1).
  M. partial marker HBM teardown samples and retains its physical prefix
     without notifying or terminalizing the graph node.
  N. teardown at a shared terminal tick samples two finished marker
     endpoints, retires them without dispatching either ready callback.
  E. teardown: an in-flight execution (big plan, partially pumped) is torn
     down with its Sys without crash, without any terminal firing, and the
     stale outer alarms drain harmlessly afterwards.

Build: cmake target
      AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest.
Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/MetricCollector.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/ParsedGraphBatch.hh"
#include "DieModel.hh"
#include "coarse_die/Plan.hh"
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
#include <limits>
#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
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
        status != static_cast<int>(NodeTerminalStatus::Skipped)) {
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

    void complete_network_callbacks(bool enabled) {
        complete_network_callbacks_ = enabled;
    }

    int sim_send(void*, uint64_t, int, int, int, sim_request*,
                 void (*callback)(void*), void* data) override {
        if (complete_network_callbacks_)
            queue_->schedule_event(queue_->get_current_time() + 1, callback, data);
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

    // AstraNetworkAPI pure-virtual surface: the stub has no link-bandwidth
    // semantics, a finite placeholder satisfies it.
    double get_BW_at_dimension(int /*dim*/) override {
        return 1.0;
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
                          {"ffn", 11008u},
                          {"vocab", 32000u},
                          {"mlp_variant", "swiglu"},
                          {"norm_type", "rmsnorm"},
                          {"position_encoding", "rope"},
                          {"num_q_heads", 32u},
                          {"num_kv_heads", 32u},
                          {"head_dim", 128u}};
    spec["tp_slice"] = {{"tensor_parallel", 1u},
                        {"tensor_parallel_rank", 0u},
                        {"q_heads_local", 32u},
                        {"kv_heads_local", 32u},
                        {"ffn_local", 11008u},
                        {"vocab_local", 32000u}};
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
        config_dir + "/" + system_file, fixture.network_api.get(),
        std::vector<int>{1}, std::vector<int>{1}, 1.0, 1.0, false,
        ExecutionMode::Online, source);
    return fixture;
}

void install_workspace_snapshot(Sys* const sys, const uint64_t capacity_bytes,
                                const uint64_t model_weight_bytes,
                                const uint64_t resident_kv_bytes,
                                const uint64_t reserved_request_bytes = 0) {
    if (sys == nullptr || sys->workspace_admission == nullptr) {
        throw std::runtime_error(
            "coarse lifecycle fixture has no workspace ledger");
    }
    WorkspaceAdmission& ledger = *sys->workspace_admission;
    const WorkspaceCapacityRank snapshot{
        sys->id, ledger.workspace_revision(), capacity_bytes,
        model_weight_bytes, resident_kv_bytes, reserved_request_bytes};
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
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0, 0);
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
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0, 0);
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
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0, 0);
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
        fixture.sys.get(), std::numeric_limits<uint64_t>::max(), 0, 0);
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

void no_op_event(void*) {}

long double marker_read_bytes(const nlohmann::json& progress) {
    long double bytes = 0.0L;
    for (const auto& row : progress.at("hbm_served_bytes_exact")) {
        bytes += std::stold(row.at(0).get<std::string>());
    }
    return bytes;
}

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

std::vector<nlohmann::json> capture_metrics(Sys* sys, uint64_t end_tick) {
    FILE* file = std::tmpfile();
    if (file == nullptr) throw std::runtime_error("cannot capture metrics");
    const int saved = dup(STDOUT_FILENO);
    std::fflush(stdout);
    if (saved < 0 || dup2(fileno(file), STDOUT_FILENO) < 0)
        throw std::runtime_error("cannot redirect metrics");
    try {
        MetricCollector::instance().finalize({sys}, end_tick);
    } catch (...) {
        dup2(saved, STDOUT_FILENO);
        close(saved);
        std::fclose(file);
        throw;
    }
    std::fflush(stdout);
    dup2(saved, STDOUT_FILENO);
    close(saved);
    std::rewind(file);
    std::string text;
    char buffer[4096];
    while (const auto n = std::fread(buffer, 1, sizeof(buffer), file))
        text.append(buffer, n);
    std::fclose(file);
    std::vector<nlohmann::json> records;
    std::istringstream lines(text);
    std::string line;
    while (std::getline(lines, line))
        if (line.rfind("[METRIC] ", 0) == 0)
            records.push_back(nlohmann::json::parse(line.substr(9)));
    return records;
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
    ::setenv("SH_WORKSPACE_ADMISSION", "auto", 1);
    LoggerFactory::init("empty", "off");
    // Keep diagnostics visible while the metrics fixture captures stdout.
    for (const auto* name : {"statistics", "system", "HardwareResource"}) {
        LoggerFactory::get_logger(name)->sinks().push_back(
            std::make_shared<spdlog::sinks::stderr_color_sink_mt>());
    }
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

    // F: typed COMP and a real COMM_SEND share one coarse Runtime. The
    // fake network never completes; cancellation/teardown must retain the
    // delivered prefix and must not publish an outer node completion.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        std::filesystem::path repo(__FILE__);
        for (unsigned i = 0; i < 5; ++i) repo = repo.parent_path();
        std::ifstream hardware_file(repo / "extern/die_model/coarse/configs/example_hardware.json");
        nlohmann::json hardware;
        hardware_file >> hardware;
        write_text(config_dir + "/coarse_hardware.json", hardware.dump());
        write_text(config_dir + "/coarse_system.json", nlohmann::json{
            {"compute-cost-model", "die"}, {"die-model-profile", "coarse_die_v2"},
            {"die-model-config", config_dir + "/coarse_hardware.json"},
            {"hbm-bandwidth-contention", 1}}.dump());
        write_text(config_dir + "/coarse_system_latency.json", nlohmann::json{
            {"compute-cost-model", "die"}, {"die-model-profile", "coarse_die_v2"},
            {"die-model-config", config_dir + "/coarse_hardware.json"},
            {"hbm-bandwidth-contention", 1},
            {"local-mem-latency", 100}}.dump());
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        // Observe the cancellation prefix with the existing microbenchmark
        // history policy; service finalization still rejects live GPU nodes.
        fixture.sys->workload->stats->configure_online_history_preservation(true);
        auto& bridge = *fixture.sys->coarse_die_runtime;
        nlohmann::json task{{"op_kind", "gemm"},
            {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
            {"storage_dtype", "bf16"}};
        const auto plan = bridge.compile(task);
        install_workspace_snapshot(fixture.sys.get(),
                                   std::numeric_limits<uint64_t>::max(), 0, 0);
        OnlineNode comp;
        comp.global_id = 40; comp.rank = 0; comp.kind = NodeKind::Compute;
        comp.node_type = 4; comp.name = "coarse_comp";
        comp.request_id = "r1"; comp.stage = "prefill";
        comp.compute.die_task_json = std::make_shared<const nlohmann::json>(task);
        fixture.source->store().add_node(comp);
        OnlineNode comm;
        comm.global_id = 41; comm.rank = 0; comm.kind = NodeKind::CommSend;
        comm.node_type = 5; comm.name = "coarse_send";
        comm.request_id = "r1"; comm.stage = "prefill";
        comm.comm.bytes = 1ULL << 30; comm.comm.src = 0; comm.comm.dst = 0;
        comm.comm.tag = 0; comm.comm.hbm_charge = true;
        fixture.source->store().add_node(comm);
        fixture.sys->workload->issue_dep_free_nodes();
        expect_true(fixture.sys->workspace_admission->active_lease_count() == 1 &&
                    fixture.sys->workspace_admission->active_workspace_bytes() ==
                        coarse_die::hbm_workspace_peak_bytes(*plan),
                    "F: pre-issue admission holds exactly the compiled Plan workspace");
        nlohmann::json partial;
        for (unsigned i = 0; i < 32 && !fixture.event_queue->finished(); ++i) {
            fixture.event_queue->proceed();
            partial = bridge.counters_at(Sys::boostedTick());
            const double ops = partial["served_by_category"]["comp"]["int4_ops"].get<double>();
            if (ops > 0 && ops < plan->summary["executed_int4_ops"].get<double>()) break;
        }
        const double delivered = partial["served_by_category"]["comp"]["int4_ops"].get<double>();
        expect_true(delivered > 0 && delivered < plan->summary["executed_int4_ops"].get<double>(),
                    "F: real typed COMP has positive partial INT4 service");
        expect_true(partial["served_by_category"]["comm_send"]["hbm_served_bytes_by_row_direction"][0][0].get<double>() > 0 &&
                    partial["hbm_shared_ns"].get<double>() > 0,
                    "F: COMP and declared COMM source read share local HBM");
        fixture.sys->workload->die_model_adapter()->shutdown();
        expect_true(fixture.sys->workspace_admission->active_lease_count() == 0 &&
                    fixture.sys->workspace_admission->active_workspace_bytes() == 0,
                    "F: shutdown cancels live Runtime work before releasing its lease");
        const auto cancelled = bridge.counters_at(Sys::boostedTick());
        expect_true(cancelled["served_by_category"]["comp"]["int4_ops"].get<double>() == delivered &&
                    g_terminal_fires.count(40) == 0 && g_terminal_fires.count(41) == 0,
                    "F: teardown cancellation preserves delivered work without graph terminals");
        write_text(config_dir + "/coarse_manifest.json", nlohmann::json{
            {"schema_version", 1}, {"repo_variant", "coarse-lifecycle-fixture"},
            {"run_mode", "microbenchmark"}, {"run_id", "coarse-lifecycle-fixture"},
            {"requests", nlohmann::json::array()},
            {"node_events_by_rank", nlohmann::json::object()},
            {"planner_memory_peaks", nlohmann::json::array()},
            {"memory_actions", nlohmann::json::array()}}.dump());
        MetricCollector::instance().initialize(config_dir + "/coarse_manifest.json", "summary");
        const auto records = capture_metrics(fixture.sys.get(), Sys::boostedTick());
        nlohmann::json rank_record, hbm_record;
        for (const auto& record : records) {
            if (record.value("type", std::string()) == "rank_compute") rank_record = record;
            if (record.value("type", std::string()) == "local_hbm") hbm_record = record;
        }
        expect_true(!rank_record.is_null() && !hbm_record.is_null(),
                    "F: MetricCollector emits coarse rank and HBM records");
        if (!rank_record.is_null() && !hbm_record.is_null()) {
            const auto& ledger = rank_record["coarse_die_runtime"];
            expect_true(ledger["served_by_category"] == cancelled["served_by_category"] &&
                        ledger["accounting_history_complete"] == true,
                        "F: rank metrics retain cancelled partial typed service");
            const auto& rows = cancelled["hbm_served_bytes_by_row_direction"];
            const double total = rows[0][0].get<double>() + rows[0][1].get<double>() +
                                 rows[1][0].get<double>() + rows[1][1].get<double>();
            const auto& raw = cancelled["hardware_raw"]["hbm_row_peak_Bps"];
            const double denominator = (raw[0].get<double>() + raw[1].get<double>()) *
                                       (static_cast<double>(Sys::boostedTick()) / 1e9);
            expect_true(std::abs(hbm_record["dram_bw_util_numerator_bytes"].get<double>() - total) < 1e-6 &&
                        std::abs(hbm_record["dram_bw_util"].get<double>() - total / denominator) < 1e-12 &&
                        hbm_record["compute_bytes_served"].get<double>() == 0,
                        "F: total HBM numerator includes DIE and COMM once over raw peaks");
        }
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(40) == 0 && g_terminal_fires.count(41) == 0,
                    "F: COMP plus COMM teardown drains stale alarms without callbacks");
    }

    // G: the normal service path joins a completed network callback with
    // local HBM, retires both nodes once, and finalizes the lifetime ledger.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        fixture.network_api->complete_network_callbacks(true);
        auto& bridge = *fixture.sys->coarse_die_runtime;
        nlohmann::json task{{"op_kind", "gemm"},
            {"tensor_shapes", {{"m", 128}, {"n", 128}, {"k", 128}}},
            {"storage_dtype", "bf16"}};
        const auto plan = bridge.compile(task);
        install_workspace_snapshot(fixture.sys.get(),
                                   std::numeric_limits<uint64_t>::max(), 0, 0);
        OnlineNode comp;
        comp.global_id = 50; comp.rank = 0; comp.kind = NodeKind::Compute;
        comp.node_type = 4; comp.name = "coarse_complete";
        comp.request_id = "r1"; comp.stage = "prefill";
        comp.compute.die_task_json = std::make_shared<const nlohmann::json>(task);
        fixture.source->store().add_node(comp);
        OnlineNode comm;
        comm.global_id = 51; comm.rank = 0; comm.kind = NodeKind::CommSend;
        comm.node_type = 5; comm.name = "coarse_complete_send";
        comm.request_id = "r1"; comm.stage = "prefill";
        comm.comm.bytes = 1ULL << 20; comm.comm.src = 0; comm.comm.dst = 0;
        comm.comm.tag = 0; comm.comm.hbm_charge = true;
        fixture.source->store().add_node(comm);
        fixture.sys->workload->issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        const auto completed = bridge.counters_at(Sys::boostedTick());
        expect_true(g_terminal_fires[50] == 1 && g_terminal_fires[51] == 1 &&
                    fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                    fixture.sys->workspace_admission->active_workspace_bytes() == 0 &&
                    fixture.sys->workspace_admission->active_lease_count() == 0 &&
                    !fixture.sys->workload->die_model_adapter()->has_active_die_executions(),
                    "G: typed COMP and COMM complete once and release GPU + workspace");
        const auto& categories = completed["served_by_category"];
        expect_true(categories["comp"]["int4_ops"].get<double>() ==
                        plan->summary["executed_int4_ops"].get<double>() &&
                    categories["comm_send"]["hbm_served_bytes_by_row_direction"][0][0].get<double>() +
                        categories["comm_send"]["hbm_served_bytes_by_row_direction"][1][0].get<double>() ==
                        static_cast<double>(comm.comm.bytes),
                    "G: retired COMP and COMM conserve their typed served work");
        nlohmann::json manifest;
        std::ifstream(config_dir + "/coarse_manifest.json") >> manifest;
        manifest["run_mode"] = "service";
        write_text(config_dir + "/coarse_manifest.json", manifest.dump());
        MetricCollector::instance().initialize(config_dir + "/coarse_manifest.json", "summary");
        const auto records = capture_metrics(fixture.sys.get(), Sys::boostedTick());
        bool matched = false;
        for (const auto& record : records)
            if (record.value("type", std::string()) == "rank_compute")
                matched = record["coarse_die_runtime"]["served_by_category"] == categories;
        expect_true(matched, "G: normal service finalization retains retired coarse counters");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[50] == 1 && g_terminal_fires[51] == 1,
                    "G: completed teardown publishes no duplicate terminals");
    }

    // I: a unit-level direct Workload dispatch proves a control marker can
    // share the canonical Runtime with a positive-workspace typed COMP. This
    // deliberately exercises the shared-runtime boundary, not the production
    // single-GPU availability gate (which serializes Compute nodes). The
    // marker owns one DIE endpoint byte, never typed COMP Plan work or COMM.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        auto& workload = *fixture.sys->workload;
        auto& bridge = *fixture.sys->coarse_die_runtime;
        nlohmann::json task{{"op_kind", "ffn_region"},
            {"tensor_shapes", {{"m", 64}, {"hidden", 128},
                               {"ffn", 256}, {"mlp_variant", "swiglu"}}},
            {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"}};
        const auto plan = bridge.compile(task);
        const uint64_t workspace_bytes =
            coarse_die::hbm_workspace_peak_bytes(*plan);
        install_workspace_snapshot(fixture.sys.get(),
                                   std::numeric_limits<uint64_t>::max(), 0, 0);
        bool scalar_comp_rejected = false;
        try {
            workload.local_hbm_bandwidth_model->issue_compute(1, 1, nullptr);
        } catch (const std::runtime_error& error) {
            scalar_comp_rejected =
                std::string(error.what()).find("typed plan") != std::string::npos;
        }
        expect_true(scalar_comp_rejected,
                    "I: ordinary scalar COMP remains forbidden by coarse HBM guard");
        OnlineNode comp;
        comp.global_id = 80; comp.rank = 0; comp.kind = NodeKind::Compute;
        comp.node_type = 4; comp.name = "coarse_workspace_comp";
        comp.request_id = "r1"; comp.stage = "prefill";
        comp.compute.die_task_json = std::make_shared<const nlohmann::json>(task);
        fixture.source->store().add_node(comp);
        OnlineNode marker;
        marker.global_id = 81; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "coarse_first_token";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        // The production single-GPU COMP availability gate serializes these
        // graph nodes. Exercise their shared Runtime boundary directly here:
        // the typed positive-workspace COMP has already been issued, while
        // the normal ControlMarker Workload issue path submits its local
        // endpoint concurrently.
        workload.issue_dep_free_nodes();
        const auto* marker_view = fixture.source->lookup_ptr(marker.global_id);
        if (marker_view != nullptr) workload.issue(*marker_view);
        expect_true(workspace_bytes > 0 &&
                        fixture.sys->workspace_admission->active_lease_count() == 1 &&
                        fixture.sys->workspace_admission->active_workspace_bytes() ==
                            workspace_bytes &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 2 &&
                        bridge.has_active_executions(),
                    "I: positive-workspace typed COMP and marker are concurrently live");
        drain_queue(fixture.event_queue);
        const auto counters = bridge.counters();
        const auto& die = counters.at("served_by_category").at("die");
        const auto& comp_category = counters.at("served_by_category").at("comp");
        const auto& comm_send = counters.at("served_by_category").at("comm_send");
        expect_true(g_terminal_fires[80] == 1 && g_terminal_fires[81] == 1 &&
                        g_first_tick[81] >= 101 &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        fixture.sys->workspace_admission->active_lease_count() == 0 &&
                        fixture.sys->workspace_admission->active_workspace_bytes() == 0 &&
                        !workload.die_model_adapter()->has_active_die_executions(),
                    "I: typed COMP and delayed marker terminalize once and release GPU/workspace");
        expect_true(die.at("hbm_served_bytes_by_row_direction_exact")[0][0] == "1" &&
                        die.at("hbm_served_bytes_by_row_direction_exact")[0][1] == "0" &&
                        die.at("hbm_served_bytes_by_row_direction_exact")[1][0] == "0" &&
                        die.at("hbm_served_bytes_by_row_direction_exact")[1][1] == "0" &&
                        die.at("int4_ops").get<double>() == 0.0 &&
                        die.at("vec_eqops").get<double>() == 0.0 &&
                        comp_category.at("int4_ops").get<double>() ==
                            plan->summary.at("executed_int4_ops").get<double>() &&
                        comm_send.at("hbm_served_bytes_by_row_direction_exact")[0][0] == "0" &&
                        comm_send.at("hbm_served_bytes_by_row_direction_exact")[1][0] == "0" &&
                        counters.at("served_by_category").at("comm_recv")
                            .at("hbm_served_bytes_by_row_direction_exact")[0][1] == "0" &&
                        workload.local_hbm_bandwidth_model->compute_bytes_served() == 1.0,
                    "I: exactly one physical marker byte is die-owned; compatibility compute bytes stay 1 and COMM/typed ops stay clean");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[80] == 1 && g_terminal_fires[81] == 1,
                    "I: completed coarse marker teardown publishes no duplicate terminal");
    }

    // L: production issue_dep_free_nodes keeps the single GPU Compute gate
    // intact while allowing a control marker and a real COMM HBM endpoint to
    // be live together in the same Runtime under their independent gates.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        fixture.network_api->complete_network_callbacks(true);
        auto& workload = *fixture.sys->workload;
        auto& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 84; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "coarse_marker_with_comm";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        OnlineNode comm;
        comm.global_id = 85; comm.rank = 0; comm.kind = NodeKind::CommSend;
        comm.node_type = 5; comm.name = "coarse_comm_with_marker";
        comm.request_id = "r1"; comm.stage = "prefill";
        comm.comm.bytes = 1; comm.comm.src = 0; comm.comm.dst = 0;
        comm.comm.tag = 0; comm.comm.hbm_charge = true;
        fixture.source->store().add_node(comm);
        workload.issue_dep_free_nodes();
        const auto snapshot = bridge.export_snapshot(Sys::boostedTick());
        uint64_t die_endpoints = 0;
        uint64_t comm_endpoints = 0;
        for (const auto& execution :
             snapshot.at("runtime_snapshot").at("executions")) {
            const auto& endpoint =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint.value("owner_kind", std::string()) == "die")
                ++die_endpoints;
            if (endpoint.value("owner_kind", std::string()) == "comm")
                ++comm_endpoints;
        }
        expect_true(die_endpoints == 1 && comm_endpoints == 1 &&
                        bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        workload.hw_resource->num_in_flight_gpu_comm_ops == 1 &&
                        g_terminal_fires.count(84) == 0 &&
                        g_terminal_fires.count(85) == 0,
                    "L: normal availability-gated issue shares Runtime between delayed marker and real COMM endpoint");
        drain_queue(fixture.event_queue);
        const auto counters = bridge.counters();
        const auto& die = counters.at("served_by_category").at("die");
        const auto& comm_send = counters.at("served_by_category").at("comm_send");
        expect_true(g_terminal_fires[84] == 1 && g_terminal_fires[85] == 1 &&
                        g_first_tick[84] >= 101 &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 0 &&
                        workload.hw_resource->num_in_flight_gpu_comm_ops == 0 &&
                        die.at("hbm_served_bytes_by_row_direction_exact")[0][0] == "1" &&
                        comm_send.at("hbm_served_bytes_by_row_direction_exact")[0][0] == "1" &&
                        workload.local_hbm_bandwidth_model->compute_bytes_served() == 1.0,
                    "L: marker and COMM endpoint both complete once with isolated DIE/COMM byte accounting");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires[84] == 1 && g_terminal_fires[85] == 1,
                    "L: completed marker/COMM teardown does not duplicate terminals");
    }

    // J: destroying the HBM facade while a delayed marker endpoint is live
    // retires the Runtime handle, drops handler data, and suppresses callbacks.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        auto& workload = *fixture.sys->workload;
        auto& bridge = *fixture.sys->coarse_die_runtime;
        // Make the endpoint remain in its startup delay while teardown runs.
        fixture.sys->local_mem_latency = 1'000'000;
        OnlineNode marker;
        marker.global_id = 82; marker.rank = 0; marker.kind = NodeKind::Compute;
        marker.node_type = 4; marker.name = "coarse_teardown_marker";
        marker.request_id = "r1"; marker.stage = "prefill";
        marker.compute.op_role = OpRole::ControlMarker;
        marker.compute.num_ops = 1; marker.compute.tensor_size = 1;
        fixture.source->store().add_node(marker);
        workload.issue_dep_free_nodes();
        expect_true(bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        g_terminal_fires.count(82) == 0,
                    "J: marker endpoint is live and nonterminal before teardown");
        workload.local_hbm_bandwidth_model.reset();
        expect_true(!bridge.has_active_executions() &&
                        bridge.counters().at("endpoint_registry_size") == 0 &&
                        bridge.counters().at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(82) == 0,
                    "J: HBM teardown cancels and retires delayed endpoint without callback");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(82) == 0,
                    "J: stale teardown alarms drain without a marker terminal");
    }

    // K: a deliberately low declared HBM efficiency keeps the exact 1-byte
    // marker in flight long enough to cancel after readiness. Both the DIE
    // category and the legacy compute_bytes_served side-band must retain the
    // same partial physical prefix from this Runtime execution.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        std::ifstream hardware_in(config_dir + "/coarse_hardware.json");
        nlohmann::json slow_hardware;
        hardware_in >> slow_hardware;
        slow_hardware["performance_profile"]["eta_hbm_matrix"] = 1e-6;
        write_text(config_dir + "/coarse_hardware_slow.json",
                   slow_hardware.dump());
        write_text(config_dir + "/coarse_system_slow.json", nlohmann::json{
            {"compute-cost-model", "die"}, {"die-model-profile", "coarse_die_v2"},
            {"die-model-config", config_dir + "/coarse_hardware_slow.json"},
            {"hbm-bandwidth-contention", 1},
            {"local-mem-latency", 100}}.dump());
        Fixture fixture = make_fixture(config_dir, "coarse_system_slow.json");
        auto& workload = *fixture.sys->workload;
        auto& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 83; marker.rank = 0; marker.kind = NodeKind::Compute;
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
        uint64_t endpoint_handle = 0;
        long double expected_partial_bytes = 0.0L;
        for (const auto& execution :
             envelope.at("runtime_snapshot").at("executions")) {
            const auto& endpoint_json =
                execution.at("plan").at("task").at("endpoint");
            if (endpoint_json.value("owner_kind", std::string()) != "die")
                continue;
            endpoint_handle = execution.at("handle").get<uint64_t>();
            const auto& served = execution.at("hbm_served");
            for (const auto& row : served) {
                for (const auto& direction : row) {
                    expected_partial_bytes +=
                        std::stold(direction.get<std::string>());
                }
            }
        }
        expect_true(now == 101 && endpoint_handle != 0 &&
                        expected_partial_bytes > 0.0L &&
                        expected_partial_bytes < 1.0L &&
                        g_terminal_fires.count(83) == 0,
                    "K: endpoint has a positive strict-prefix read after its delay");
        const auto live_progress_before =
            bridge.query_progress(endpoint_handle);
        const auto live_counters_before = bridge.counters();
        const double getter_live_bytes =
            workload.local_hbm_bandwidth_model->compute_bytes_served();
        const auto live_progress_after =
            bridge.query_progress(endpoint_handle);
        const auto live_counters_after = bridge.counters();
        const bool live_time_reversal_rejected = [&] {
            try {
                (void)bridge.query_progress_at(endpoint_handle, now - 1);
            } catch (const std::invalid_argument&) {
                return true;
            }
            return false;
        }();
        const bool stale_handle_rejected = [&] {
            try {
                (void)bridge.query_progress_at(0, now);
            } catch (const std::out_of_range&) {
                return true;
            }
            return false;
        }();
        expect_true(std::abs(static_cast<long double>(getter_live_bytes) -
                             expected_partial_bytes) < 1e-18L &&
                        live_progress_before == live_progress_after &&
                        live_counters_before == live_counters_after &&
                        bridge.has_active_executions() &&
                        workload.hw_resource->num_in_flight_gpu_comp_ops == 1 &&
                        g_terminal_fires.count(83) == 0 &&
                        live_time_reversal_rejected && stale_handle_rejected,
                    "K: const getter samples live marker bytes without mutating Runtime, slots, terminals, or fail-closed checks");
        bridge.cancel(endpoint_handle, now, true);
        const auto cancelled = bridge.counters();
        const auto& die = cancelled.at("served_by_category").at("die");
        const long double die_read = std::stold(
            die.at("hbm_served_bytes_by_row_direction_exact")[0][0]
                .get<std::string>());
        const auto& comm = cancelled.at("served_by_category").at("comm_send");
        expect_true(std::abs(die_read - expected_partial_bytes) < 1e-18L &&
                        std::abs(static_cast<long double>(
                                     workload.local_hbm_bandwidth_model
                                         ->compute_bytes_served()) -
                                 getter_live_bytes) < 1e-18L &&
                        comm.at("hbm_served_bytes_by_row_direction_exact")[0][0] == "0" &&
                        !bridge.has_active_executions() &&
                        cancelled.at("endpoint_registry_size") == 0 &&
                        cancelled.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(83) == 0,
                    "K: partial cancellation preserves the same die/compatibility bytes, leaves COMM clean, and suppresses graph completion");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(83) == 0,
                    "K: cancelled marker teardown does not publish a terminal");
    }

    // M: destroying the HBM facade after a partial marker read preserves the
    // exact physical prefix while silently retiring its Runtime callback.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_slow.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        OnlineNode marker;
        marker.global_id = 86; marker.rank = 0; marker.kind = NodeKind::Compute;
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
            ? marker_read_bytes(projected) : 0.0L;
        while (!teardown.fired && !fixture.event_queue->finished()) {
            fixture.event_queue->proceed();
        }
        const auto counters = bridge.counters();
        const auto& die = counters.at("served_by_category").at("die");
        const long double retained_prefix = std::stold(
            die.at("hbm_served_bytes_by_row_direction_exact")[0][0]
                .get<std::string>());
        expect_true(teardown.fired && Sys::boostedTick() == 101 &&
                        expected_prefix > 0.0L && expected_prefix < 1.0L &&
                        teardown.samples.size() == 1 &&
                        std::abs(marker_read_bytes(teardown.samples.front()) -
                                 expected_prefix) < 1e-18L &&
                        std::abs(retained_prefix - expected_prefix) < 1e-18L &&
                        !bridge.has_active_executions() &&
                        counters.at("endpoint_registry_size") == 0 &&
                        counters.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(86) == 0,
                    "M: partial teardown retains the sampled DIE prefix and suppresses marker callbacks");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(86) == 0,
                    "M: partial marker teardown remains nonterminal after stale alarms drain");
    }

    // N: the first event at the physical terminal tick tears down the HBM
    // facade before Runtime's own event can dispatch. Both marker callbacks
    // are ready, so notify=true here would terminalize both graph nodes.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system_latency.json");
        Workload& workload = *fixture.sys->workload;
        CoarseDieRuntime& bridge = *fixture.sys->coarse_die_runtime;
        fixture.sys->local_mem_latency = 0;
        OnlineNode first;
        first.global_id = 87; first.rank = 0; first.kind = NodeKind::Compute;
        first.node_type = 4; first.name = "finished_teardown_marker_a";
        first.request_id = "r1"; first.stage = "prefill";
        first.compute.op_role = OpRole::ControlMarker;
        first.compute.num_ops = 1; first.compute.tensor_size = 1;
        fixture.source->store().add_node(first);
        OnlineNode second = first;
        second.global_id = 88;
        second.name = "finished_teardown_marker_b";
        fixture.source->store().add_node(second);
        HbmTeardownEventContext teardown{&workload, &bridge};
        fixture.event_queue->schedule_event(
            1, teardown_hbm_model_event, &teardown);
        workload.issue_dep_free_nodes();
        // The production single-GPU gate serializes these graph nodes. Direct
        // Workload dispatch here only creates a second ready Runtime callback
        // so destructor notification behavior is tested at shared tick 1.
        const auto* second_view = fixture.source->lookup_ptr(88);
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
        const long double retained_bytes = std::stold(
            die.at("hbm_served_bytes_by_row_direction_exact")[0][0]
                .get<std::string>());
        const bool sampled_both_finished = teardown.samples.size() == 2 &&
            std::all_of(teardown.samples.begin(), teardown.samples.end(),
                [](const nlohmann::json& sample) {
                    return sample.at("state") == "finished" &&
                           sample.at("terminal").get<bool>() &&
                           std::abs(marker_read_bytes(sample) - 1.0L) < 1e-18L;
                });
        expect_true(teardown.fired && Sys::boostedTick() == 1 &&
                        teardown.handles.size() == 2 &&
                        sampled_both_finished &&
                        std::abs(retained_bytes - 2.0L) < 1e-18L &&
                        !bridge.has_active_executions() &&
                        counters.at("endpoint_registry_size") == 0 &&
                        counters.at("transaction_registry_size") == 0 &&
                        g_terminal_fires.count(87) == 0 &&
                        g_terminal_fires.count(88) == 0,
                    "N: teardown retains two physically finished bytes without dispatching either ready callback");
        fixture.sys.reset();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.count(87) == 0 &&
                        g_terminal_fires.count(88) == 0,
                    "N: terminal-tick marker teardown stays silent after stale alarms drain");
    }

    // H: a real coarse COMP is prepared before NodeStore take/HardwareResource
    // occupy, stays free while Python's base usage blocks it, then submits
    // with the same exact workspace reservation after a capacity refresh.
    // A live cancellation terminalizes as Skipped and releases the lease
    // exactly once before dependency completion.
    {
        g_first_tick.clear();
        g_terminal_fires.clear();
        g_terminal_status.clear();
        Fixture fixture = make_fixture(config_dir, "coarse_system.json");
        nlohmann::json task{{"op_kind", "ffn_region"},
            {"tensor_shapes", {{"m", 64}, {"hidden", 128},
                               {"ffn", 256}, {"mlp_variant", "swiglu"}}},
            {"fusion_policy", "materialized"}, {"storage_dtype", "bf16"}};
        const auto plan = fixture.sys->coarse_die_runtime->compile(task);
        const uint64_t workspace_bytes =
            coarse_die::hbm_workspace_peak_bytes(*plan);
        expect_true(workspace_bytes > 0 &&
                        workspace_bytes < std::numeric_limits<uint64_t>::max(),
                    "H: fixture Plan has a positive exact workspace peak");
        if (workspace_bytes == 0 ||
            workspace_bytes == std::numeric_limits<uint64_t>::max()) {
            fixture.sys.reset();
        } else {
            OnlineNode comp;
            comp.global_id = 70; comp.rank = 0; comp.kind = NodeKind::Compute;
            comp.node_type = 4; comp.name = "coarse_workspace_gate";
            comp.request_id = "r1"; comp.stage = "prefill";
            comp.compute.die_task_json =
                std::make_shared<const nlohmann::json>(task);
            fixture.source->store().add_node(comp);
            OnlineNode child;
            child.global_id = 72; child.rank = 0;
            child.kind = NodeKind::Invalid; child.node_type = 0;
            child.name = "after_cancelled_coarse_comp";
            child.request_id = "r1"; child.stage = "prefill";
            fixture.source->store().add_node(child);
            fixture.source->store().add_dependency(70, 72, DepKind::Control);

            const uint64_t capacity_bytes = workspace_bytes + 1;
            install_workspace_snapshot(fixture.sys.get(), capacity_bytes,
                                       0, 2);
            fixture.sys->workload->issue_dep_free_nodes();
            std::vector<uint64_t> free_ids;
            fixture.source->store().fill_free_node_snapshot(free_ids);
            expect_true(fixture.sys->workspace_admission->active_lease_count() == 0 &&
                            fixture.sys->workload->hw_resource
                                    ->num_in_flight_gpu_comp_ops == 0 &&
                            std::find(free_ids.begin(), free_ids.end(), 70) !=
                                free_ids.end() &&
                            g_terminal_fires.count(70) == 0,
                        "H: blocked Plan remains free and does not occupy GPU or publish terminal");

            install_workspace_snapshot(fixture.sys.get(), capacity_bytes,
                                       0, 1);
            fixture.sys->workload->issue_dep_free_nodes();
            free_ids.clear();
            fixture.source->store().fill_free_node_snapshot(free_ids);
            expect_true(fixture.sys->workspace_admission->active_lease_count() == 1 &&
                            fixture.sys->workspace_admission
                                    ->active_workspace_bytes() == workspace_bytes &&
                            fixture.sys->workload->hw_resource
                                    ->num_in_flight_gpu_comp_ops == 1 &&
                            std::find(free_ids.begin(), free_ids.end(), 70) ==
                                free_ids.end(),
                        "H: capacity refresh admits exact workspace before take/occupy");

            // Let at least one modeled Runtime event advance before live
            // cancellation so the callback must report real served work over
            // a positive elapsed-time denominator.
            fixture.event_queue->proceed();
            const auto before_cancel =
                fixture.sys->coarse_die_runtime->counters_at(
                    Sys::boostedTick());
            const auto& comp_served =
                before_cancel.at("served_by_category").at("comp");
            bool saw_served_work =
                comp_served.at("int4_ops").get<double>() > 0.0 ||
                comp_served.at("vec_eqops").get<double>() > 0.0;
            const auto& comp_hbm =
                comp_served.at("hbm_served_bytes_by_row_direction");
            for (size_t row = 0; row < comp_hbm.size(); ++row) {
                for (size_t direction = 0;
                     direction < comp_hbm[row].size(); ++direction) {
                    saw_served_work =
                        saw_served_work ||
                        comp_hbm[row][direction].get<double>() > 0.0;
                }
            }
            expect_true(Sys::boostedTick() > 0 && saw_served_work &&
                            g_terminal_fires.count(70) == 0,
                        "H: cancellation fixture has typed served work and a positive elapsed window");
            fixture.sys->workload->die_model_adapter()->cancel_coarse_node(
                70, Sys::boostedTick());
            const auto* cancelled_stats =
                fixture.source->store().mutable_online_statistics(70);
            std::vector<uint64_t> free_after_cancel;
            fixture.source->store().fill_free_node_snapshot(
                free_after_cancel);
            expect_true(g_terminal_fires[70] == 1 &&
                            g_terminal_status[70] == static_cast<int>(
                                NodeTerminalStatus::Skipped) &&
                            cancelled_stats != nullptr &&
                            cancelled_stats->compute_utilization.has_value() &&
                            cancelled_stats->memory_utilization.has_value() &&
                            cancelled_stats->vector_utilization.has_value() &&
                            fixture.sys->workload
                                    ->die_cancelled_comp_count() == 1 &&
                            std::find(free_after_cancel.begin(),
                                      free_after_cancel.end(), 72) !=
                                free_after_cancel.end() &&
                            fixture.sys->workspace_admission
                                    ->active_workspace_bytes() == 0 &&
                            fixture.sys->workspace_admission
                                    ->active_lease_count() == 0 &&
                            fixture.sys->workload->hw_resource
                                    ->num_in_flight_gpu_comp_ops == 0 &&
                            fixture.sys->workspace_admission
                                    ->workspace_revision() == 2,
                        "H: live cancellation records served utilization, releases workspace once, marks cancellation, terminalizes Skipped, and frees its dependent node");
            fixture.sys->workload->issue_dep_free_nodes();
            expect_true(g_terminal_fires[72] == 1 &&
                            g_terminal_status[72] == static_cast<int>(
                                NodeTerminalStatus::Skipped),
                        "H: cancelled COMP releases its real graph dependency for normal successor issue");

            nlohmann::json manifest;
            std::ifstream(config_dir + "/coarse_manifest.json") >> manifest;
            manifest["run_mode"] = "service";
            write_text(config_dir + "/coarse_manifest.json",
                       manifest.dump());
            MetricCollector::instance().initialize(
                config_dir + "/coarse_manifest.json", "summary");
            const auto cancelled_records =
                capture_metrics(fixture.sys.get(), Sys::boostedTick());
            bool cancellation_count_exported = false;
            for (const auto& record : cancelled_records) {
                if (record.value("type", std::string()) == "rank_compute") {
                    cancellation_count_exported =
                        record.value("die_cancelled_comp_count", uint64_t{0}) ==
                        1;
                }
            }
            expect_true(cancellation_count_exported,
                        "H: rank service report exports the live cancelled COMP count");

            install_workspace_snapshot(fixture.sys.get(),
                                       workspace_bytes - 1, 0, 0);
            comp.global_id = 71;
            comp.name = "coarse_workspace_impossible";
            fixture.source->store().add_node(comp);
            bool impossible_rejected = false;
            try {
                fixture.sys->workload->issue_dep_free_nodes();
            } catch (const std::runtime_error& exc) {
                impossible_rejected =
                    std::string(exc.what()).find("exceeds empty-instance") !=
                    std::string::npos;
            }
            free_ids.clear();
            fixture.source->store().fill_free_node_snapshot(free_ids);
            expect_true(impossible_rejected &&
                            std::find(free_ids.begin(), free_ids.end(), 71) !=
                                free_ids.end() &&
                            fixture.sys->workspace_admission
                                    ->active_lease_count() == 0 &&
                            fixture.sys->workspace_admission
                                    ->active_workspace_bytes() == 0 &&
                            fixture.sys->workload->hw_resource
                                    ->num_in_flight_gpu_comp_ops == 0 &&
                            g_terminal_fires.count(71) == 0,
                        "H: impossible Plan fails before take/occupy and leaves no lease");
            fixture.sys.reset();
            drain_queue(fixture.event_queue);
        }
    }

    test_prepared_coarse_preflight(config_dir);
    test_legacy_parsed_coarse_lifecycle(config_dir);

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
