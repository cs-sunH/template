/******************************************************************************
This source code is licensed under the MIT license found in the root directory
of this source tree.
*******************************************************************************/

/**
hbm_tail_component_test.cc -- v7 E2 closed-loop component fixture
(joint改造方案v7 §5/§8/§9).

Reproduces, on the REAL production mechanisms (online Sys + Workload +
LocalHbmBandwidthModel + DieModelAdapter/DieExecution over the analytical
EventQueue), the decode-burst shape the v7 attribution measured in the
first-1s settle tail: a batch of same-shape die nodes issued together, each
draining its internal memory stages through the rank's single HBM arbiter
at the equal share, then a post-burst drain with no further adapter entry
(the corpse-observer regime). The fixture is the A/B differential vehicle
for the v7 execution-engine restructure AND a semantic regression:

  - closed form: with M identical plans issued at tick 0, the first HBM
    completion boundary lands exactly at first_memory_earliest_tick +
    ceil(S_min * M / B) (equal split of B bytes/ns across M streamers);
  - conservation: die_service_units_served == M * (sum of the plan's
    service units); die_actual_bytes_served == the same sum scaled by the
    plan's eta triple (relative 1e-9 -- float association across substeps
    is not identical to a single multiply; exactness across engine
    versions is enforced by the byte-identical canonical output below);
  - lifecycle: every node terminals exactly once, the model drains to
    zero live jobs, and resource_observer_count() returns to its baseline
    after settle (a finished DieExecution must not keep its no-op HBM
    mutation observer registered -- the v7 tail root cause);
  - observer dispatch snapshot semantics (add during a pass is invisible
    to the pass, removal during a pass keeps the entry visible to that
    pass and hides it from later passes);
  - canonical summary: one deterministic line every consumer of this
    fixture diffs byte-for-byte between engine versions (counts, ticks,
    counters; %.17g for doubles).

Timing line (prefixed [TIME]) is printed separately and is NOT part of the
canonical output. Build: cmake target
AstraSim_Analytical_Congestion_Aware_HbmTailComponentTest. Exit 0 = ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/CallData.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/DieExecution.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/WatchRegistry.hh"
#include <DieModel.hh>
#include <astra-network-analytical/common/EventQueue.h>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
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

std::map<uint64_t, uint64_t> g_terminals;
std::map<uint64_t, uint64_t> g_first_tick;

void tick_hook(void* /*ctx*/, int /*rank*/, uint64_t node_id,
               const char* /*request_id*/, const char* /*stage*/,
               uint64_t /*generation*/, uint64_t tick, int status) {
    if (status != static_cast<int>(NodeTerminalStatus::Success)) {
        g_ok = false;
    }
    if (g_first_tick.find(node_id) == g_first_tick.end()) {
        g_first_tick[node_id] = tick;
    }
    ++g_terminals[node_id];
}

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

nlohmann::json window_contract_wire(uint32_t window_limit) {
    return {
        {"projection_abi_version", 1u},
        {"descriptor_version", 1u},
        {"pipeline_protocol_version", 1u},
        {"model_version", "v7-three-actions-20261014"},
        {"timer_whitelist_version", "timer-whitelist-v1-20261005"},
        {"window_tile_limit", window_limit},
        {"window_local_error_ppm", 0u},
        {"window_refine_margin_ppm", 20000u},
        {"prediction_config_sha256",
         "0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"}};
}

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr, "[hbm_tail_component_test] cannot write %s\n",
                     path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out << content;
}

// Production-caliber HBM scalar from the canonical hardware config
// (face_case5_config_c: 1640 GB/s) so per-event costs land in the same
// regime the v7 attribution measured.
constexpr double kBandwidthBytesPerSec = 1640.0e9;

struct PlanShape {
    // Action walk of one compiled plan (independent of the engine under
    // test): the first memory request's service units / eta inputs and the
    // accumulated per-node service totals.
    uint64_t first_timer_ns = 0;         // issue -> first MemoryRequest delay
    uint64_t first_memory_service = 0;   // service units of the first leg
    double first_memory_actual = 0.0;    // actual bytes of the first leg
    double total_service_units = 0.0;    // sum over all memory legs
    double total_actual_bytes = 0.0;     // sum over all memory legs
    uint64_t memory_legs = 0;
    uint64_t final_elapsed = 0;          // isolated (full-speed) elapsed ns
};

PlanShape walk_plan(const DieModel::DieOpSpec& spec,
                    const DieModel::Hardware& hw,
                    const uint64_t layer_count) {
    (void)layer_count;
    auto plan = DieModel::compile(spec, hw);
    DieModel::Execution machine(plan);
    PlanShape shape;
    bool seen_memory = false;
    while (!machine.finished()) {
        const DieModel::ExecutionAction action = machine.next_action();
        if (action.kind == DieModel::ExecutionAction::Kind::Timer) {
            if (!seen_memory) {
                shape.first_timer_ns += std::max<uint64_t>(action.ns, 1);
            }
            machine.advance_time(std::max<uint64_t>(action.ns, 1));
        } else if (action.kind ==
                   DieModel::ExecutionAction::Kind::MemoryRequest) {
            if (!seen_memory) {
                seen_memory = true;
                shape.first_memory_service = static_cast<uint64_t>(
                    action.service_units);
                shape.first_memory_actual = action.actual_bytes;
            }
            shape.total_service_units += action.service_units;
            shape.total_actual_bytes += action.actual_bytes;
            ++shape.memory_legs;
            const double ns_double = std::ceil(
                action.service_units * 1e9 / hw.hbm_bandwidth_bytes);
            machine.advance_time(std::max<uint64_t>(
                static_cast<uint64_t>(ns_double), 1));
            machine.complete_memory(action.stage_id);
        } else {
            break;
        }
    }
    shape.final_elapsed = machine.elapsed_ns();
    return shape;
}

nlohmann::json spec_json(uint64_t layers, uint64_t q_tokens,
                         uint64_t kv_tokens) {
    nlohmann::json span;
    span["request_id"] = "r1";
    span["q_tokens"] = q_tokens;
    span["kv_tokens"] = kv_tokens;
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

// v8 K2 component: production decode shape (32 layers, q=1, kv=512) --
// same span structure the online graph authors for decode nodes.
nlohmann::json decode_spec_json(uint64_t layers, uint64_t q_tokens,
                                uint64_t kv_tokens) {
    nlohmann::json spec = spec_json(layers, q_tokens, kv_tokens);
    spec["phase"] = "decode";
    return spec;
}

struct Fixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    TestRemoteMemoryApi remote_memory;
    std::unique_ptr<Sys> sys;
    NodeStoreGraphSource* source = nullptr;
};

Fixture make_fixture(const std::string& config_dir,
                     const std::string& system_file,
                     uint32_t window_limit = 1) {
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
        PipelineKernelContract::parse(window_contract_wire(window_limit)));
    return fixture;
}

// Engine-version-agnostic diagnostics: the v7 WORK engine exposes
// transition_event_count()/resource_observer_count(); older engines do
// not. The canonical A/B diff line carries only fields both engines
// produce; these counters print on a separate non-diffed line.
template <typename T>
auto transition_count_or_zero(const T* model, int)
    -> decltype(model->transition_event_count()) {
    return model->transition_event_count();
}
template <typename T>
uint64_t transition_count_or_zero(const T*, long) {
    return 0;
}
template <typename T>
auto observer_count_or_zero(const T* model, int)
    -> decltype(model->resource_observer_count()) {
    return model->resource_observer_count();
}
template <typename T>
size_t observer_count_or_zero(const T*, long) {
    return 0;
}
// v8 reclamation accessors: storage size and tombstone share. Engines
// without them (pre-v8) report "absent" and every assertion that depends on
// REAL counts is skipped there -- a zero stub would fake the observation.
template <typename T>
auto observer_storage_or_absent(const T* model, int)
    -> decltype(model->resource_observer_storage_size()) {
    return model->resource_observer_storage_size();
}
template <typename T>
size_t observer_storage_or_absent(const T*, long) {
    return static_cast<size_t>(-1);
}
template <typename T>
auto observer_tombstone_or_absent(const T* model, int)
    -> decltype(model->resource_observer_tombstone_count()) {
    return model->resource_observer_tombstone_count();
}
template <typename T>
size_t observer_tombstone_or_absent(const T*, long) {
    return static_cast<size_t>(-1);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CompletionObserver::instance().set_hook(tick_hook, nullptr);

    const std::string tmp_template = "/tmp/hbm_tail_component_XXXXXX";
    std::vector<char> buffer(tmp_template.begin(), tmp_template.end());
    buffer.push_back('\0');
    const char* const made = mkdtemp(buffer.data());
    if (made == nullptr) {
        std::fprintf(stderr, "[hbm_tail_component_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 100
})");

    const DieModel::Hardware hw =
        DieModel::make_hardware(nlohmann::json::parse(kDieNodeJson));

    // ---- Plan shape (independent walk of the SAME spec/hardware) --------
    const uint64_t layers = 2;
    const auto spec = DieModel::parse_die_op_spec(
        spec_json(layers, 8, 128));
    const PlanShape shape = walk_plan(*spec, hw, layers);
    std::printf("[shape] first_timer=%llu first_service=%llu legs=%llu "
                "total_service=%.17g total_actual=%.17g isolated=%llu\n",
                static_cast<unsigned long long>(shape.first_timer_ns),
                static_cast<unsigned long long>(shape.first_memory_service),
                static_cast<unsigned long long>(shape.memory_legs),
                shape.total_service_units, shape.total_actual_bytes,
                static_cast<unsigned long long>(shape.final_elapsed));

    // ---- Burst component: serial die-node chain over the real hook -----
    // Production serializes real COMP nodes through the single GPU-comp
    // gate (HardwareResource is_available), so the v7 tail shape is a
    // serial chain of die nodes on the rank, each draining its memory legs
    // alone at the full rate while COMPLETED DieExecutions accumulate in
    // the adapter (the corpse-observer regime the attribution measured).
    // The completion hook wiring below mirrors main_online.cc: the terminal
    // callback schedules the same-tick deferred per-rank issue pass that
    // advances the chain.
    const uint64_t M = 512;
    const auto t_start = std::chrono::steady_clock::now();

    Fixture fixture = make_fixture(config_dir, "die_contention.json");
    Workload& workload = *fixture.sys->workload;
    auto* model = workload.local_hbm_bandwidth_model.get();
    const size_t observer_baseline = observer_count_or_zero(model, 0);
    std::vector<uint64_t> mutation_ticks;
    const auto recorder_id = model->add_resource_mutation_observer(
        [&](bool, Tick tick, uint64_t) { mutation_ticks.push_back(tick); });

    WatchRegistry watch_registry;
    OnlineCompletionHookContext online_hook_ctx;
    online_hook_ctx.registry = &watch_registry;
    // Combined terminal hook: the production online hook (watch facts +
    // deferred issue pass) PLUS this fixture's terminal recorder.
    struct HookPair {
        OnlineCompletionHookContext* online_ctx;
    } hook_pair{&online_hook_ctx};
    std::vector<Workload*> workloads{&workload};
    online_hook_ctx.configure_issue_passes(
        fixture.event_queue.get(), workloads.size(),
        [](void* opaque, const int rank) {
            auto* const online_workloads =
                static_cast<std::vector<Workload*>*>(opaque);
            (*online_workloads)[rank]->issue_dep_free_nodes();
        },
        &workloads);
    CompletionObserver::instance().set_hook(
        [](void* ctx, const int rank, const uint64_t node_id,
           const char* const request_id, const char* const stage,
           const uint64_t generation, const uint64_t tick,
           const int terminal_status) {
            auto* const pair = static_cast<HookPair*>(ctx);
            online_completion_hook(pair->online_ctx, rank, node_id,
                                   request_id, stage, generation, tick,
                                   terminal_status);
            tick_hook(nullptr, rank, node_id, request_id, stage, generation,
                      tick, terminal_status);
        },
        &hook_pair);

    for (uint64_t id = 0; id < M; ++id) {
        OnlineNode node;
        node.global_id = 100 + id;
        node.rank = 0;
        node.kind = NodeKind::Compute;
        node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
        node.name = "norm_node";
        node.request_id = "r1";
        node.stage = "prefill";
        node.generation = 0;
        node.compute.die_spec = spec;
        fixture.source->store().add_node(node);
    }
    const uint64_t local_mem_latency =
        static_cast<uint64_t>(fixture.sys->local_mem_latency);
    workload.issue_dep_free_nodes();
    while (!fixture.event_queue->finished()) {
        fixture.event_queue->proceed();
    }
    // Expected per-node span: leading plan timers + one-shot local memory
    // latency + each memory leg alone at the FULL rate (serial chain, N=1).
    // The independent walk below recomputes the same arithmetic from the
    // immutable spec/hardware; +/- one tick per leg for the double
    // association of ceil(S/(B/1)) vs ceil(S*1e9/B).
    const uint64_t expected_span = shape.final_elapsed + local_mem_latency;
    const uint64_t first_memory_tick =
        shape.first_timer_ns + local_mem_latency;
    const uint64_t expected_first_boundary =
        first_memory_tick +
        static_cast<uint64_t>(std::ceil(
            static_cast<double>(shape.first_memory_service) /
            (kBandwidthBytesPerSec / 1e9)));
    const auto query_snapshot = model->job_snapshot();
    const uint64_t transitions = transition_count_or_zero(model, 0);
    const double served = model->die_service_units_served();
    const double actual = model->die_actual_bytes_served();
    const uint64_t peak = model->peak_concurrent_jobs();
    const uint64_t redist = model->redistribution_events();
    const double busy = model->hbm_busy_ns();
    model->remove_resource_mutation_observer(recorder_id);
    const size_t observers_after = observer_count_or_zero(model, 0);
    const uint64_t settle_tick = fixture.event_queue->get_current_time();
    const auto t_end = std::chrono::steady_clock::now();
    const uint64_t wall_ns = static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(t_end - t_start)
            .count());
    CompletionObserver::instance().set_hook(tick_hook, nullptr);

    // ---- Closed form: serial spans and first full-rate boundary ---------
    const uint64_t earliest_first_completion = [&] {
        uint64_t best = ~0ull;
        for (const auto& [node_id, tick] : g_first_tick) {
            (void)node_id;
            best = std::min(best, tick);
        }
        return best;
    }();
    uint64_t first_boundary_tick = ~0ull;
    for (const uint64_t tick : mutation_ticks) {
        if (tick > first_memory_tick && tick < first_boundary_tick) {
            first_boundary_tick = tick;
        }
    }
    expect_true(first_boundary_tick != ~0ull,
                "closed form: a post-issue mutation boundary exists");
    expect_true(first_boundary_tick + 1 >= expected_first_boundary &&
                first_boundary_tick <= expected_first_boundary + 1,
                "closed form: first boundary == ceil(S/B) (1-tick "
                "double-association tolerance)");
    const uint64_t span_tol = shape.memory_legs + 2;
    expect_true(earliest_first_completion + span_tol >= expected_span &&
                earliest_first_completion <= expected_span + span_tol,
                "closed form: node 0 span == isolated full-speed elapsed + "
                "latency (per-leg ceil tolerance)");
    const uint64_t expected_settle = static_cast<uint64_t>(M) *
                                     expected_span;
    expect_true(settle_tick + M * span_tol >= expected_settle &&
                settle_tick <= expected_settle + M * span_tol,
                "closed form: settle == M serial spans");

    // ---- Lifecycle / accounting -----------------------------------------
    expect_true(g_terminals.size() == M,
                "every burst node terminal exactly once (count)");
    bool all_once = true;
    for (const auto& [node_id, fires] : g_terminals) {
        (void)node_id;
        if (fires != 1) all_once = false;
    }
    expect_true(all_once, "every burst node terminal exactly once (fires)");
    expect_true(query_snapshot.empty(),
                "model drains to zero live jobs at settle");
    if (observer_baseline != 0 || observers_after != 0) {
        // Newer engines expose the count; only they can assert the v7
        // corpse-observer lifecycle (baseline == after == 0 there).
        expect_true(observer_baseline == 0 && observers_after == 0,
                    "observer registrations return to baseline after settle");
    }
    expect_true(peak >= 1 && peak <= 8,
                "serial chain keeps concurrent die jobs bounded small");
    const double expected_served = static_cast<double>(M) *
                                   shape.total_service_units;
    const double rel_service =
        std::fabs(served - expected_served) / expected_served;
    expect_true(rel_service <= 1e-9,
                "service conservation: die_service_units_served == M*plan");
    const double expected_actual = static_cast<double>(M) *
                                   shape.total_actual_bytes;
    const double rel_actual =
        std::fabs(actual - expected_actual) / expected_actual;
    expect_true(rel_actual <= 1e-9,
                "actual conservation: die_actual_bytes_served == M*plan");
    expect_true(busy > 0.0 && busy <= static_cast<double>(settle_tick) + 1.0,
                "busy accounting within the settle span");
    expect_true(redist == 0,
                "serial same-shape chain: no share reshaping events");

    // ---- Canonical summary (byte-diff target across engine versions) ----
    std::printf("[CANONICAL] M=%llu layers=%llu settle_tick=%llu "
                "peak=%llu redist=%llu "
                "served=%.17g actual=%.17g busy=%.17g "
                "earliest=%llu\n",
                static_cast<unsigned long long>(M),
                static_cast<unsigned long long>(layers),
                static_cast<unsigned long long>(settle_tick),
                static_cast<unsigned long long>(peak),
                static_cast<unsigned long long>(redist),
                served, actual, busy,
                static_cast<unsigned long long>(earliest_first_completion));
    std::printf("[DIAG] transitions=%llu observers_before=%zu "
                "observers_after=%zu\n",
                static_cast<unsigned long long>(transitions),
                observer_baseline, observers_after);
    for (const auto& [node_id, tick] : g_first_tick) {
        if (node_id < 100 + 8 || node_id >= 100 + M - 8) {
            std::printf("[CANONICAL_NODE] %llu %llu\n",
                        static_cast<unsigned long long>(node_id),
                        static_cast<unsigned long long>(tick));
        }
    }
    std::printf("[TIME] wall_ns=%llu transitions=%llu "
                "ns_per_transition=%.1f\n",
                static_cast<unsigned long long>(wall_ns),
                static_cast<unsigned long long>(transitions),
                static_cast<double>(wall_ns) /
                    static_cast<double>(transitions == 0 ? 1 : transitions));

    // ---- v8 K2 production-shape component (completion-chain hotspot) -----
    // The v8 PROFILE attribution measured the settle tail's ~98us
    // completion callback entirely inside DieExecution::drive()'s Timer
    // branch after each memory completion (transfer/retire/watch exits at
    // zero). This arm reproduces that regime with the REAL production K2
    // contract (window_tile_limit=2) and production-shaped decode specs:
    // every memory completion re-enters the Timer branch with hbm_idle
    // (serial chain, the completed job already detached), which is exactly
    // the tail's dominant path.
    {
        const uint64_t M2 = 128;
        const uint64_t layers2 = 32;
        const auto decode_spec = DieModel::parse_die_op_spec(
            decode_spec_json(layers2, 512, 512));
        const PlanShape shape2 = walk_plan(*decode_spec, hw, layers2);
        std::printf("[shape_k2] legs=%llu total_service=%.17g "
                    "total_actual=%.17g isolated=%llu\n",
                    static_cast<unsigned long long>(shape2.memory_legs),
                    shape2.total_service_units,
                    shape2.total_actual_bytes,
                    static_cast<unsigned long long>(shape2.final_elapsed));

        Fixture k2 = make_fixture(config_dir, "die_contention.json", 2);
        Workload& k2_workload = *k2.sys->workload;
        auto* k2_model = k2_workload.local_hbm_bandwidth_model.get();
        const size_t k2_observer_baseline =
            observer_count_or_zero(k2_model, 0);
        std::vector<uint64_t> k2_mutation_ticks;
        const auto k2_recorder_id = k2_model->add_resource_mutation_observer(
            [&](bool, Tick tick, uint64_t) {
                k2_mutation_ticks.push_back(tick);
            });

        WatchRegistry k2_watch_registry;
        OnlineCompletionHookContext k2_hook_ctx;
        k2_hook_ctx.registry = &k2_watch_registry;
        struct K2HookPair {
            OnlineCompletionHookContext* online_ctx;
        } k2_hook_pair{&k2_hook_ctx};
        std::vector<Workload*> k2_workloads{&k2_workload};
        k2_hook_ctx.configure_issue_passes(
            k2.event_queue.get(), k2_workloads.size(),
            [](void* opaque, const int rank) {
                auto* const ws =
                    static_cast<std::vector<Workload*>*>(opaque);
                (*ws)[rank]->issue_dep_free_nodes();
            },
            &k2_workloads);
        CompletionObserver::instance().set_hook(
            [](void* ctx, const int rank, const uint64_t node_id,
               const char* const request_id, const char* const stage,
               const uint64_t generation, const uint64_t tick,
               const int terminal_status) {
                auto* const pair = static_cast<K2HookPair*>(ctx);
                online_completion_hook(pair->online_ctx, rank, node_id,
                                       request_id, stage, generation, tick,
                                       terminal_status);
                tick_hook(nullptr, rank, node_id, request_id, stage,
                          generation, tick, terminal_status);
            },
            &k2_hook_pair);

        for (uint64_t id = 0; id < M2; ++id) {
            OnlineNode node;
            node.global_id = 5000 + id;
            node.rank = 0;
            node.kind = NodeKind::Compute;
            node.node_type = 4;
            node.name = "decode_node";
            node.request_id = "r1";
            node.stage = "decode";
            node.generation = 0;
            node.compute.die_spec = decode_spec;
            k2.source->store().add_node(node);
        }
        const auto k2_start = std::chrono::steady_clock::now();
        k2_workload.issue_dep_free_nodes();
        while (!k2.event_queue->finished()) {
            k2.event_queue->proceed();
        }
        const auto k2_end = std::chrono::steady_clock::now();
        const uint64_t k2_wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                k2_end - k2_start)
                .count());
        const uint64_t k2_transitions =
            transition_count_or_zero(k2_model, 0);
        const double k2_served = k2_model->die_service_units_served();
        const double k2_actual = k2_model->die_actual_bytes_served();
        const uint64_t k2_peak = k2_model->peak_concurrent_jobs();
        const uint64_t k2_settle_tick =
            k2.event_queue->get_current_time();
        k2_model->remove_resource_mutation_observer(k2_recorder_id);
        const size_t k2_observers_after =
            observer_count_or_zero(k2_model, 0);
        CompletionObserver::instance().set_hook(tick_hook, nullptr);

        expect_true(g_terminals.size() == M + M2,
                    "K2 arm: every node terminal exactly once (count)");
        const double k2_expected_served =
            static_cast<double>(M2) * shape2.total_service_units;
        const double k2_rel =
            std::fabs(k2_served - k2_expected_served) / k2_expected_served;
        expect_true(k2_rel <= 1e-9,
                    "K2 arm: service conservation holds");
        expect_true(k2_model->job_snapshot().empty(),
                    "K2 arm: model drains to zero live jobs");
        expect_true(k2_observer_baseline == 0 && k2_observers_after == 0,
                    "K2 arm: observer registrations return to baseline");
        expect_true(k2_peak >= 1 && k2_peak <= 8,
                    "K2 arm: serial chain keeps concurrency bounded");

        std::printf("[CANONICAL_K2] M=%llu layers=%llu settle_tick=%llu "
                    "peak=%llu served=%.17g actual=%.17g busy=%.17g\n",
                    static_cast<unsigned long long>(M2),
                    static_cast<unsigned long long>(layers2),
                    static_cast<unsigned long long>(k2_settle_tick),
                    static_cast<unsigned long long>(k2_peak),
                    k2_served, k2_actual, k2_model->hbm_busy_ns());
        std::printf("[DIAG_K2] transitions=%llu legs=%llu "
                    "observers_after=%zu\n",
                    static_cast<unsigned long long>(k2_transitions),
                    static_cast<unsigned long long>(shape2.memory_legs),
                    k2_observers_after);
        std::printf("[TIME_K2] wall_ns=%llu transitions=%llu "
                    "ns_per_transition=%.1f\n",
                    static_cast<unsigned long long>(k2_wall_ns),
                    static_cast<unsigned long long>(k2_transitions),
                    static_cast<double>(k2_wall_ns) /
                        static_cast<double>(
                            k2_transitions == 0 ? 1 : k2_transitions));
    }

    // ---- v9 big-frontier preview component (transactional preview) ------
    // The v8 PROFILE label correction (v9 §1) located the tail hotspot in
    // preview_consecutive_internal_timers, whose per-call cost was the copy
    // of the four scheduler frontier vectors. This arm drives REAL
    // production-shaped plans whose machines reach thousands-entry ready
    // frontiers through the legal action protocol (qk prefill with the
    // first-1s trace's real prefill volume: 21184 instances, ~7k heap
    // frontier max, ~14k previews per node), so the Timer-branch preview
    // dominates the arm's wall time on the copy engine and the A/B diff
    // isolates the transactional restructure under the full DieExecution +
    // HBM + K2 contract.
    {
        const uint64_t M3 = 4;
        const uint64_t layers3 = 32;
        nlohmann::json big_spec_json = spec_json(layers3, 61550, 61550);
        big_spec_json["op_kind"] = "qk";
        const auto big_spec =
            DieModel::parse_die_op_spec(big_spec_json);
        const PlanShape shape3 = walk_plan(*big_spec, hw, layers3);
        std::printf("[shape_v9] legs=%llu total_service=%.17g "
                    "total_actual=%.17g isolated=%llu\n",
                    static_cast<unsigned long long>(shape3.memory_legs),
                    shape3.total_service_units,
                    shape3.total_actual_bytes,
                    static_cast<unsigned long long>(shape3.final_elapsed));

        Fixture v9 = make_fixture(config_dir, "die_contention.json", 2);
        Workload& v9_workload = *v9.sys->workload;
        auto* v9_model = v9_workload.local_hbm_bandwidth_model.get();
        const size_t v9_observer_baseline =
            observer_count_or_zero(v9_model, 0);
        std::vector<uint64_t> v9_mutation_ticks;
        const auto v9_recorder_id = v9_model->add_resource_mutation_observer(
            [&](bool, Tick tick, uint64_t) {
                v9_mutation_ticks.push_back(tick);
            });

        WatchRegistry v9_watch_registry;
        OnlineCompletionHookContext v9_hook_ctx;
        v9_hook_ctx.registry = &v9_watch_registry;
        struct V9HookPair {
            OnlineCompletionHookContext* online_ctx;
        } v9_hook_pair{&v9_hook_ctx};
        std::vector<Workload*> v9_workloads{&v9_workload};
        v9_hook_ctx.configure_issue_passes(
            v9.event_queue.get(), v9_workloads.size(),
            [](void* opaque, const int rank) {
                auto* const ws =
                    static_cast<std::vector<Workload*>*>(opaque);
                (*ws)[rank]->issue_dep_free_nodes();
            },
            &v9_workloads);
        CompletionObserver::instance().set_hook(
            [](void* ctx, const int rank, const uint64_t node_id,
               const char* const request_id, const char* const stage,
               const uint64_t generation, const uint64_t tick,
               const int terminal_status) {
                auto* const pair = static_cast<V9HookPair*>(ctx);
                online_completion_hook(pair->online_ctx, rank, node_id,
                                       request_id, stage, generation, tick,
                                       terminal_status);
                tick_hook(nullptr, rank, node_id, request_id, stage,
                          generation, tick, terminal_status);
            },
            &v9_hook_pair);

        for (uint64_t id = 0; id < M3; ++id) {
            OnlineNode node;
            node.global_id = 9000 + id;
            node.rank = 0;
            node.kind = NodeKind::Compute;
            node.node_type = 4;
            node.name = "qk_prefill_node";
            node.request_id = "r1";
            node.stage = "prefill";
            node.generation = 0;
            node.compute.die_spec = big_spec;
            v9.source->store().add_node(node);
        }
        const auto v9_start = std::chrono::steady_clock::now();
        v9_workload.issue_dep_free_nodes();
        while (!v9.event_queue->finished()) {
            v9.event_queue->proceed();
        }
        const auto v9_end = std::chrono::steady_clock::now();
        const uint64_t v9_wall_ns = static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                v9_end - v9_start)
                .count());
        const uint64_t v9_transitions =
            transition_count_or_zero(v9_model, 0);
        const double v9_served = v9_model->die_service_units_served();
        const double v9_actual = v9_model->die_actual_bytes_served();
        const uint64_t v9_peak = v9_model->peak_concurrent_jobs();
        const uint64_t v9_settle_tick =
            v9.event_queue->get_current_time();
        v9_model->remove_resource_mutation_observer(v9_recorder_id);
        const size_t v9_observers_after =
            observer_count_or_zero(v9_model, 0);
        CompletionObserver::instance().set_hook(tick_hook, nullptr);

        // The K2 arm asserted the M+M2 prefix above; this arm owns the
        // 9000+ node range and every one of its nodes must terminal once.
        for (uint64_t id = 0; id < M3; ++id) {
            const auto it = g_terminals.find(9000 + id);
            expect_true(it != g_terminals.end() && it->second == 1,
                        "V9 arm: every node terminal exactly once");
        }
        const double v9_expected_served =
            static_cast<double>(M3) * shape3.total_service_units;
        const double v9_rel =
            std::fabs(v9_served - v9_expected_served) / v9_expected_served;
        expect_true(v9_rel <= 1e-9,
                    "V9 arm: service conservation holds");
        expect_true(v9_model->job_snapshot().empty(),
                    "V9 arm: model drains to zero live jobs");
        expect_true(v9_observer_baseline == 0 && v9_observers_after == 0,
                    "V9 arm: observer registrations return to baseline");

        std::printf("[CANONICAL_V9] M=%llu layers=%llu settle_tick=%llu "
                    "peak=%llu served=%.17g actual=%.17g busy=%.17g\n",
                    static_cast<unsigned long long>(M3),
                    static_cast<unsigned long long>(layers3),
                    static_cast<unsigned long long>(v9_settle_tick),
                    static_cast<unsigned long long>(v9_peak),
                    v9_served, v9_actual, v9_model->hbm_busy_ns());
        std::printf("[DIAG_V9] transitions=%llu legs=%llu "
                    "observers_after=%zu\n",
                    static_cast<unsigned long long>(v9_transitions),
                    static_cast<unsigned long long>(shape3.memory_legs),
                    v9_observers_after);
        std::printf("[TIME_V9] wall_ns=%llu transitions=%llu "
                    "ns_per_transition=%.1f\n",
                    static_cast<unsigned long long>(v9_wall_ns),
                    static_cast<unsigned long long>(v9_transitions),
                    static_cast<double>(v9_wall_ns) /
                        static_cast<double>(
                            v9_transitions == 0 ? 1 : v9_transitions));
    }

    // ---- Observer dispatch snapshot semantics ---------------------------
    {
        struct CountingCallData : public CallData {};
        struct CountingSink final : public Callable {
            int calls = 0;
            void call(EventType, CallData* data) override {
                ++calls;
                delete static_cast<CountingCallData*>(data);
            }
        };
        // Case 1 -- removal during a pass keeps the entry visible to the
        // removing pass and hides it from later passes (the historical
        // per-notify copy semantics).
        Fixture of = make_fixture(config_dir, "die_contention.json");
        auto* om = of.sys->workload->local_hbm_bandwidth_model.get();
        std::vector<int> a_calls, b_calls, c_calls;
        LocalHbmBandwidthModel::MutationObserverId id_c = 0;
        om->add_resource_mutation_observer(
            [&](bool, Tick, uint64_t) { a_calls.push_back(1); });
        om->add_resource_mutation_observer([&](bool, Tick, uint64_t) {
            b_calls.push_back(1);
            if (id_c != 0) {
                om->remove_resource_mutation_observer(id_c);
            }
        });
        id_c = om->add_resource_mutation_observer(
            [&](bool, Tick, uint64_t) { c_calls.push_back(1); });
        CountingSink sink;
        om->issue_die_memory(1000.0, 500.0, &sink, new CountingCallData);
        // Two passes inside issue (before + after): pass 1 (before) visits
        // A, B, C -- during pass 1 B removes C, so C is still visited
        // (snapshot); pass 2 (after) visits A, B only. Net A=2, B=2, C=1.
        expect_true(a_calls.size() >= 2, "observer A: both issue passes");
        expect_true(b_calls.size() >= 2, "observer B: both issue passes");
        expect_true(c_calls.size() == 1,
                    "observer C: removed mid-pass, still visited by the "
                    "removing pass and skipped by every later one");
        while (!of.event_queue->finished()) {
            of.event_queue->proceed();
        }
        expect_true(sink.calls == 1, "probe sink completes exactly once");

        // Case 2 -- an observer added during a pass is invisible to that
        // pass pair and visited by later passes.
        Fixture af = make_fixture(config_dir, "die_contention.json");
        auto* am = af.sys->workload->local_hbm_bandwidth_model.get();
        std::vector<int> a2, d2;
        LocalHbmBandwidthModel::MutationObserverId id_d = 0;
        am->add_resource_mutation_observer([&](bool, Tick, uint64_t) {
            a2.push_back(1);
            if (id_d == 0) {
                id_d = am->add_resource_mutation_observer(
                    [&](bool, Tick, uint64_t) { d2.push_back(1); });
            }
        });
        CountingSink sink2;
        am->issue_die_memory(1000.0, 500.0, &sink2, new CountingCallData);
        // Pass 1 (notify_before) visits A only (D appended mid-pass is
        // beyond the bound); pass 2 (publish) snapshots AFTER the append,
        // so it visits D exactly once here (the historical copy behaves
        // identically).
        expect_true(a2.size() == 2 && d2.size() == 1,
                    "observer D: invisible to the pass that added it, "
                    "visited by the next");
        while (!af.event_queue->finished()) {
            af.event_queue->proceed();
        }
        expect_true(d2.size() >= 2, "observer D visited by later passes");
    }

    // ---- v8 observer reclamation (tombstone lifecycle) -------------------
    // Case 3 -- the destruction proof: an observer removed during a pass
    // holds the LAST reference to its captured state; after the outermost
    // pass ends, that capture must be DESTROYED (observable via the
    // capture's own destructor -- callback counters alone cannot show it)
    // and the deque storage must return to the live count.
    {
        struct Capture {
            int* destroyed;
            explicit Capture(int* d) : destroyed(d) {}
            ~Capture() { ++*destroyed; }
        };
        struct CountingCallData : public CallData {};
        struct CountingSink final : public Callable {
            int calls = 0;
            void call(EventType, CallData* data) override {
                ++calls;
                delete static_cast<CountingCallData*>(data);
            }
        };
        Fixture vf = make_fixture(config_dir, "die_contention.json");
        auto* vm = vf.sys->workload->local_hbm_bandwidth_model.get();
        int destroyed = 0;
        LocalHbmBandwidthModel::MutationObserverId id_victim = 0;
        // The lambda holds the ONLY reference: the capture destructs exactly
        // when the deque entry is reclaimed (init-capture, no local copy).
        id_victim = vm->add_resource_mutation_observer(
            [held = std::make_shared<Capture>(&destroyed)](
                bool, Tick, uint64_t) {});
        vm->add_resource_mutation_observer([&](bool, Tick, uint64_t) {
            if (id_victim != 0) {
                vm->remove_resource_mutation_observer(id_victim);
                id_victim = 0;
            }
        });
        CountingSink v_sink;
        vm->issue_die_memory(1000.0, 500.0, &v_sink, new CountingCallData);
        expect_true(destroyed == 1,
                    "reclamation: removed observer's captured state is "
                    "destroyed once the outermost pass ends (last reference "
                    "released)");
        const size_t v_live = observer_count_or_zero(vm, 0);
        const size_t v_storage = observer_storage_or_absent(vm, 0);
        const size_t v_tomb = observer_tombstone_or_absent(vm, 0);
        if (v_storage != static_cast<size_t>(-1)) {
            expect_true(v_live == 1 && v_storage == v_live && v_tomb == 0,
                        "reclamation: deque storage == live count and zero "
                        "tombstones after the pass");
        }
        while (!vf.event_queue->finished()) {
            vf.event_queue->proceed();
        }
        expect_true(v_sink.calls == 1, "reclamation probe sink completes");

        // Case 4 -- nested passes take independent snapshots: an observer
        // appended during the outer pass is invisible to that pass's
        // remaining entries, but the nested dispatch (which starts after
        // the append) and the next top-level pass both visit it. Exact call
        // counts are identical to the historical copy semantics.
        Fixture nf = make_fixture(config_dir, "die_contention.json");
        auto* nm = nf.sys->workload->local_hbm_bandwidth_model.get();
        std::map<char, int> n_calls;
        bool n_armed = true;
        LocalHbmBandwidthModel::MutationObserverId id_added = 0;
        CountingSink n_sink_inner;
        nm->add_resource_mutation_observer([&](bool, Tick, uint64_t) {
            ++n_calls['a'];
            if (n_armed) {
                n_armed = false;
                id_added = nm->add_resource_mutation_observer(
                    [&](bool, Tick, uint64_t) { ++n_calls['d']; });
                // Nested mutation dispatch from inside the pass: two nested
                // passes (before/publish), each an independent snapshot.
                nm->issue_die_memory(2000.0, 1000.0, &n_sink_inner,
                                     new CountingCallData);
            }
        });
        nm->add_resource_mutation_observer(
            [&](bool, Tick, uint64_t) { ++n_calls['z']; });
        CountingSink n_sink_outer;
        nm->issue_die_memory(1000.0, 500.0, &n_sink_outer,
                             new CountingCallData);
        // a: outer pass1 + nested before + nested publish + outer pass2 = 4.
        // z: same sequence = 4. d: appended during outer pass1 -> invisible
        // to the outer pass remainder, visited by both nested passes and
        // outer pass2 = 3.
        expect_true(n_calls['a'] == 4, "nested pass: trigger observer calls");
        expect_true(n_calls['z'] == 4, "nested pass: sibling observer calls");
        expect_true(n_calls['d'] == 3,
                    "nested pass: observer appended mid-pass is invisible to "
                    "the appending pass, visible to nested and later passes");
        const size_t n_live = observer_count_or_zero(nm, 0);
        const size_t n_storage = observer_storage_or_absent(nm, 0);
        const size_t n_tomb = observer_tombstone_or_absent(nm, 0);
        if (n_storage != static_cast<size_t>(-1)) {
            expect_true(n_live == 3 && n_storage == n_live && n_tomb == 0,
                        "nested pass: storage compacted at the true "
                        "outermost boundary");
        }
        while (!nf.event_queue->finished()) {
            nf.event_queue->proceed();
        }
        expect_true(n_sink_outer.calls == 1 && n_sink_inner.calls == 1,
                    "nested pass: both die jobs complete exactly once");

        // Case 5 -- exception exit: a throwing observer unwinds the pass;
        // the reclamation state stays legally consumable (a removal that
        // happened before the throw is reclaimed at the unwinding boundary)
        // and the model accepts subsequent legal calls.
        Fixture xf = make_fixture(config_dir, "die_contention.json");
        auto* xm = xf.sys->workload->local_hbm_bandwidth_model.get();
        int x_destroyed = 0;
        LocalHbmBandwidthModel::MutationObserverId id_xvictim = 0;
        id_xvictim = xm->add_resource_mutation_observer(
            [held = std::make_shared<Capture>(&x_destroyed)](
                bool, Tick, uint64_t) {});
        bool x_throw_armed = true;
        xm->add_resource_mutation_observer([&](bool, Tick, uint64_t) {
            if (!x_throw_armed) {
                return;
            }
            x_throw_armed = false;
            if (id_xvictim != 0) {
                xm->remove_resource_mutation_observer(id_xvictim);
                id_xvictim = 0;
            }
            throw std::runtime_error("observer probe exception");
        });
        bool x_caught = false;
        {
            CountingSink x_aborted_sink;
            CallData* const probe = new CountingCallData;
            try {
                xm->issue_die_memory(1000.0, 500.0, &x_aborted_sink, probe);
                // Unreachable on the throwing engine; if a future engine
                // stops propagating, the probe would be owned by the job.
            } catch (const std::runtime_error&) {
                // The throw happened before any job was created, so the
                // probe was never stored: reclaiming it here is exact.
                delete probe;
                x_caught = true;
            }
            expect_true(x_aborted_sink.calls == 0,
                        "exception exit: aborted issue fired no callback");
        }
        expect_true(x_caught,
                    "exception exit: observer exception propagates out of "
                    "the dispatch");
        expect_true(x_destroyed == 1,
                    "exception exit: pre-throw removal is reclaimed at the "
                    "unwinding pass boundary");
        CountingSink x_sink;
        xm->issue_die_memory(3000.0, 1500.0, &x_sink, new CountingCallData);
        const size_t x_live = observer_count_or_zero(xm, 0);
        const size_t x_storage = observer_storage_or_absent(xm, 0);
        const size_t x_tomb = observer_tombstone_or_absent(xm, 0);
        if (x_storage != static_cast<size_t>(-1)) {
            expect_true(x_live == 1 && x_storage == x_live && x_tomb == 0,
                        "exception exit: tombstones drained, subsequent "
                        "calls legally served");
        }
        while (!xf.event_queue->finished()) {
            xf.event_queue->proceed();
        }
        expect_true(x_sink.calls == 1,
                    "exception exit: subsequent die job completes exactly "
                    "once");
    }

    if (g_ok) {
        std::printf("[hbm_tail_component_test] ALL PASS\n");
        return 0;
    }
    std::printf("[hbm_tail_component_test] FAILURES PRESENT\n");
    return 1;
}
