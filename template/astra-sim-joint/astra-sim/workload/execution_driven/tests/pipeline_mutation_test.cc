/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/**
pipeline_mutation_test.cc -- real resource-mutation semantics on the die
backend's exact timer path (the window-algebra half of the former D1 suite
was retired with the K>1 timer-window approximation on 2026-10-16; what
remains is the resource-version machinery the approximation used to hook
into, now proven against the always-exact arm).

  Part 2 (hermetic Sys + EventQueue + LocalHbmBandwidthModel, same shape as
  die_execution_lifecycle_test.cc): a real DieExecution driving a
  launch-timer-only LocalRegion plan (4 uniform waves x 1000 ns) under a
  frozen K=8 contract. A test-injected COMM_READ HBM job triggers the real
  rank mutation observer mid-arm and exactly at the due tick, in BOTH
  physical insertion orders (M10): mutation before the due callback keeps
  the cookie and publishes once at the original callback position; the
  callback first leaves nothing to invalidate. Mid-arm mutation settles the
  old state to now and never steals service from the exact arm (the
  terminal tick is unchanged vs the no-mutation baseline), while the
  injected physical HBM job keeps draining naturally.

  Part 3 (real backends): a FluidScheduler (NoC link versions, M07) and an
  AnalyticalRemoteMemory port (remote-port versions) prove that every real
  membership/rate change pairs one before with one after observer event and
  strictly advances the resource version, and that observers never re-enter
  the backends (flows/transactions complete naturally, exactly once).

Build: cmake target JointPipelineMutationTest (full stack: AstraSim +
Analytical_Congestion_Aware). Exit code 0 on ALL PASS.
*/

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/PipelineKernelContract.hh"
#include "DieModel.hh"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidScheduler.h"
#include <astra-network-analytical/common/EventQueue.h>
#include <remote_memory_backend/analytical/AnalyticalRemoteMemory.hh>

#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;
using Analytical::AnalyticalRemoteMemory;

namespace {

bool g_ok = true;

void expect_true(const bool cond, const char* const what) {
    std::printf("%-76s -> %s\n", what, cond ? "PASS" : "FAIL");
    if (!cond) {
        g_ok = false;
    }
}

nlohmann::json contract_wire() {
    return nlohmann::json{
        {"projection_abi_version", 1u},
        {"descriptor_version", 1u},
        {"pipeline_protocol_version", 1u},
        {"model_version", PipelineKernelContract::kModelVersion},
        {"timer_whitelist_version",
         PipelineKernelContract::kTimerWhitelistVersion},
        {"prediction_config_sha256", std::string(64, 'a')}};
}

// ---------------------------------------------------------------------------
// Part 2 fixture: hermetic Sys over the real EventQueue + HBM model.
// ---------------------------------------------------------------------------

std::map<uint64_t, uint64_t> g_terminal_fires;
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
    ++g_terminal_fires[node_id];
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

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out.is_open()) {
        std::fprintf(stderr,
                     "[pipeline_mutation_test] cannot write %s\n",
                     path.c_str());
        std::exit(EXIT_FAILURE);
    }
    out << content;
}

/// A LocalRegion spec is the zero-work identity region: its compiled plan
/// is exactly one launch TIMER stage with layer_count*weight_passes uniform
/// waves of launch_us_per_macro_group (1 us -> 1000 ns each). No memory
/// leg, no HBM job of its own: every HBM mutation in Part 2 comes from the
/// test injection alone.
nlohmann::json launch_region_spec_json(uint64_t layers) {
    nlohmann::json span;
    span["request_id"] = "r1";
    span["q_tokens"] = 1u;
    span["kv_tokens"] = 1u;
    span["q_start_pos"] = 0u;
    span["kv_start_pos"] = 0u;
    nlohmann::json spec;
    spec["schema_version"] = 1u;
    spec["op_kind"] = "local_region";
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

OnlineNode make_launch_region_node(uint64_t id, uint64_t layers) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = "local_region";
    node.request_id = "r1";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.die_spec =
        DieModel::parse_die_op_spec(launch_region_spec_json(layers));
    return node;
}

struct SysFixture {
    std::shared_ptr<EventQueue> event_queue;
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    TestRemoteMemoryApi remote_memory;
    std::unique_ptr<Sys> sys;
    NodeStoreGraphSource* source = nullptr;
};

SysFixture make_sys_fixture(const std::string& config_dir,
                            const std::string& system_file) {
    SysFixture fixture;
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
        PipelineKernelContract::parse(contract_wire()));
    return fixture;
}

void drain_queue(const std::shared_ptr<EventQueue>& queue) {
    while (!queue->finished()) {
        queue->proceed();
    }
}

struct InjectionContext {
    Workload* workload = nullptr;
    LocalHbmBandwidthModel* model = nullptr;
    uint64_t node_id = 0;
    uint64_t bytes = 0;
    uint64_t transfers_completed = 0;
    bool has_active_at_injection = false;
    uint64_t terminals_at_injection = 0;
};

void inject_hbm_comm_read(void* raw) {
    std::unique_ptr<std::pair<InjectionContext*,
                              std::shared_ptr<void>>> guard(
        static_cast<std::pair<InjectionContext*, std::shared_ptr<void>>*>(
            raw));
    InjectionContext* context = guard->first;
    // Ordering discriminator, captured BEFORE the mutation pair: the die
    // execution state at the moment the same-tick injection runs.
    context->has_active_at_injection =
        context->workload->die_model_adapter()
            ->has_active_die_executions();
    context->terminals_at_injection =
        g_terminal_fires.count(context->node_id)
            ? g_terminal_fires[context->node_id]
            : 0;
    // The real rank mutation: an independent COMM_READ transfer joins the
    // HBM bus. issue_transfer publishes the before/after observer pair.
    InjectionContext* unowned = context;
    context->model->issue_transfer(
        LocalHbmBandwidthModel::JobKind::COMM_READ, context->bytes, 0,
        0x7fffffff, [unowned]() { ++unowned->transfers_completed; });
}

// ---------------------------------------------------------------------------
// Part 2: DieExecution × HBM mutation.
// ---------------------------------------------------------------------------

void test_hbm_mutation_mid_arm_settles_and_keeps_physical_callback() {
    g_terminal_fires.clear();
    g_first_tick.clear();
    const std::string config_dir = "/tmp/pipeline_mutation_test_hbm";
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    std::filesystem::create_directories(config_dir, ec);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    // No hbm-bandwidth-contention key: under the die backend the derived
    // local-mem-bw fill restores the contention default (an explicit key
    // with no local-mem-bw trips the auto-disable and skips the restore).
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 0
})");

    constexpr uint64_t kWaves = 4;
    // launch_us_per_macro_group == 1.0 us and launch waves are slot-free:
    // every wave ends at tick 1000, so the coalesced exact arm is due at
    // 1000 regardless of the wave count.
    constexpr uint64_t kDueTick = 1000;

    // Baseline: no mutation, the coalesced exact arm completes at 4000.
    {
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        fixture.source->store().add_node(make_launch_region_node(31, kWaves));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(31) == 1 &&
                        g_first_tick.at(31) == kDueTick,
                    "HBM: baseline exact arm completes at the coalesced "
                    "stretch tick");
    }

    // Mid-arm mutation at 1500: settle + keep the physical callback,
    // the terminal stays at 4000, and the injected physical job drains.
    {
        g_terminal_fires.clear();
        g_first_tick.clear();
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        InjectionContext context;
        context.workload = &workload;
        context.model = workload.local_hbm_bandwidth_model.get();
        context.node_id = 32;
        context.bytes = 1000;
        fixture.source->store().add_node(make_launch_region_node(32, kWaves));
        workload.issue_dep_free_nodes();
        auto* payload =
            new std::pair<InjectionContext*, std::shared_ptr<void>>(
                &context, std::shared_ptr<void>());
        fixture.event_queue->schedule_event(500, inject_hbm_comm_read,
                                            payload);
        drain_queue(fixture.event_queue);
        expect_true(g_terminal_fires.at(32) == 1 &&
                        g_first_tick.at(32) == kDueTick,
                    "HBM: mid-arm mutation never steals service from the "
                    "exact arm (terminal unchanged)");
        expect_true(context.has_active_at_injection,
                    "HBM: the mutation ran while the exact arm was still "
                    "active mid-flight");
        expect_true(context.transfers_completed == 1,
                    "HBM: the injected physical HBM job drained naturally "
                    "alongside the exact arm");
        const nlohmann::json quality =
            workload.die_model_adapter()->quality_json();
        expect_true(quality.at("completed_execution_count") == 1 &&
                        quality.at("logical_event_count").get<uint64_t>() > 0,
                    "HBM: the uniform launch template reports live DIE event "
                    "counters (no fake averaged approximation)");
    }

    std::filesystem::remove_all(config_dir, ec);
}

void test_m10_due_tick_both_insertion_orders() {
    g_terminal_fires.clear();
    g_first_tick.clear();
    const std::string config_dir = "/tmp/pipeline_mutation_test_m10";
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    std::filesystem::create_directories(config_dir, ec);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    // No hbm-bandwidth-contention key: under the die backend the derived
    // local-mem-bw fill restores the contention default (an explicit key
    // with no local-mem-bw trips the auto-disable and skips the restore).
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 0
})");

    constexpr uint64_t kWaves = 4;
    constexpr uint64_t kDueTick = 1000;  // launch waves all end at 1 us

    // Order A (mutation physically first at the due tick): the injection
    // event is scheduled BEFORE the node issue, so it sits ahead of the
    // Sys alarm for tick 4000 in the queue's insertion order.
    {
        g_terminal_fires.clear();
        g_first_tick.clear();
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        InjectionContext context;
        context.workload = &workload;
        context.model = workload.local_hbm_bandwidth_model.get();
        context.node_id = 33;
        context.bytes = 1000;
        auto* payload =
            new std::pair<InjectionContext*, std::shared_ptr<void>>(
                &context, std::shared_ptr<void>());
        fixture.event_queue->schedule_event(kDueTick, inject_hbm_comm_read,
                                            payload);
        fixture.source->store().add_node(make_launch_region_node(33, kWaves));
        workload.issue_dep_free_nodes();
        drain_queue(fixture.event_queue);
        expect_true(context.has_active_at_injection &&
                        context.terminals_at_injection == 0,
                    "M10-A: the due-tick mutation ran before the physical "
                    "timer callback (execution still active, no terminal)");
        expect_true(g_terminal_fires.at(33) == 1 &&
                        g_first_tick.at(33) == kDueTick,
                    "M10-A: the kept cookie published exactly once at its "
                    "original callback position");
        expect_true(context.transfers_completed == 1,
                    "M10-A: the mutation's physical job drained naturally");
    }

    // Order B (callback physically first): the injection is scheduled
    // AFTER the node issue, i.e. behind the Sys alarm for tick 4000.
    {
        g_terminal_fires.clear();
        g_first_tick.clear();
        SysFixture fixture =
            make_sys_fixture(config_dir, "die_contention.json");
        Workload& workload = *fixture.sys->workload;
        InjectionContext context;
        context.workload = &workload;
        context.model = workload.local_hbm_bandwidth_model.get();
        context.node_id = 34;
        context.bytes = 1000;
        fixture.source->store().add_node(make_launch_region_node(34, kWaves));
        workload.issue_dep_free_nodes();
        auto* payload =
            new std::pair<InjectionContext*, std::shared_ptr<void>>(
                &context, std::shared_ptr<void>());
        fixture.event_queue->schedule_event(kDueTick, inject_hbm_comm_read,
                                            payload);
        drain_queue(fixture.event_queue);
        expect_true(!context.has_active_at_injection &&
                        context.terminals_at_injection == 1,
                    "M10-B: the physical timer callback ran before the "
                    "same-tick mutation (execution retired, terminal "
                    "already fired)");
        expect_true(g_terminal_fires.at(34) == 1 &&
                        g_first_tick.at(34) == kDueTick,
                    "M10-B: the callback-first order completes exactly once "
                    "at the same tick");
        expect_true(context.transfers_completed == 1,
                    "M10-B: the later mutation still drains its physical "
                    "job naturally");
    }

    std::filesystem::remove_all(config_dir, ec);
}

// ---------------------------------------------------------------------------
// Part 3: backend resource-version observer semantics.
// ---------------------------------------------------------------------------

void no_op_callback(void*) {}

void test_noc_link_resource_mutation_observer() {
    auto event_queue = std::make_shared<EventQueue>();
    std::vector<std::shared_ptr<const Link>> links;
    links.push_back(std::make_shared<Link>(0, 1.0, 0.0));
    links.push_back(std::make_shared<Link>(1, 0.5, 0.0));
    FluidScheduler scheduler(event_queue, std::move(links), 8, 16, 1024);

    auto shared_route = std::make_shared<FluidRoute>();
    shared_route->link_ids = {0, 1};
    shared_route->propagation_latency_ns = 7;

    struct ObserverLog {
        uint64_t before_events = 0;
        uint64_t after_events = 0;
        std::vector<uint64_t> versions_at_before;
        std::vector<uint64_t> versions_at_after;
    } log;

    // The observer only records; it must never re-enter the scheduler
    // (the §17.2 red line the retired window host used to exercise from
    // this same callback).
    const uint64_t observer_id = scheduler.add_link_resource_mutation_observer(
        [&scheduler, &log](const bool before_change,
                           const EventTime tick,
                           const std::vector<LinkId>& /*links*/) {
            (void)tick;
            if (before_change) {
                ++log.before_events;
                log.versions_at_before.push_back(
                    scheduler.link_resource_version(0));
            } else {
                ++log.after_events;
                log.versions_at_after.push_back(
                    scheduler.link_resource_version(0));
            }
        });

    // Short background flow starts first: one membership change on each
    // traversed link -> version 1, one before/after pair.
    scheduler.start_flow(10, shared_route, no_op_callback, nullptr);
    scheduler.flush_pending_starts();
    expect_true(log.before_events == 1 && log.after_events == 1 &&
                    log.versions_at_after.back() == 1,
                "NoC: the first membership change publishes exactly one "
                "version bump");

    // A second flow joins the same links mid-flight: another real mutation.
    scheduler.mark_event_loop_started();
    struct DelayedJoin {
        FluidScheduler* scheduler;
        std::shared_ptr<FluidRoute> route;
    };
    auto* join = new DelayedJoin{&scheduler, shared_route};
    event_queue->schedule_event(
        2,
        [](void* raw) {
            std::unique_ptr<DelayedJoin> delayed(
                static_cast<DelayedJoin*>(raw));
            delayed->scheduler->start_flow(6, delayed->route, no_op_callback,
                                           nullptr);
        },
        join);
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    expect_true(log.before_events >= 2 && log.after_events == log.before_events,
                "NoC: every real membership/rate change pairs one before "
                "with one after event");
    bool versions_advance = true;
    for (std::size_t i = 1; i < log.versions_at_after.size(); ++i) {
        if (log.versions_at_after[i] <= log.versions_at_after[i - 1]) {
            versions_advance = false;
        }
    }
    expect_true(versions_advance &&
                    !log.versions_at_after.empty() &&
                    log.versions_at_after.front() == 1,
                "NoC: link versions strictly advance across joins and "
                "completions");
    expect_true(scheduler.get_active_flow_count() == 0 &&
                    scheduler.get_total_completed_flows() == 2,
                "NoC: both real flows completed naturally (the observer "
                "never cancelled a physical flow)");
    scheduler.remove_link_resource_mutation_observer(observer_id);
}

void test_remote_port_resource_mutation_observer() {
    const std::string config_dir = "/tmp/pipeline_mutation_test_port";
    std::error_code ec;
    std::filesystem::remove_all(config_dir, ec);
    std::filesystem::create_directories(config_dir, ec);
    write_text(config_dir + "/comm_group.json", "{}");
    write_text(config_dir + "/resolved_die_model.json", kDieNodeJson);
    write_text(config_dir + "/remote_memory.json", R"({
  "memory-type": "PER_NPU_MEMORY_EXPANSION",
  "npu-ids": [0],
  "remote-mem-bw": 6.0,
  "remote-mem-latency": 100
})");
    // No hbm-bandwidth-contention key: under the die backend the derived
    // local-mem-bw fill restores the contention default (an explicit key
    // with no local-mem-bw trips the auto-disable and skips the restore).
    write_text(config_dir + "/die_contention.json", R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "local-mem-latency": 0
})");

    AnalyticalRemoteMemory remote_memory(config_dir + "/remote_memory.json");

    g_terminal_fires.clear();
    g_first_tick.clear();
    SysFixture fixture =
        make_sys_fixture(config_dir, "die_contention.json");
    // Bind the real port backend as this Sys' remote memory (the Sys ctor
    // already called set_sys on the placeholder API; bind the real one and
    // register it for the rank explicitly).
    fixture.sys->remote_mem = &remote_memory;
    remote_memory.set_sys(0, fixture.sys.get());
    Workload& workload = *fixture.sys->workload;

    struct PortLog {
        uint64_t before_events = 0;
        uint64_t after_events = 0;
        std::vector<uint64_t> versions;
    } log;
    // The observer only records; it must never re-enter the port backend.
    const uint64_t observer_id =
        remote_memory.add_port_resource_mutation_observer(
            [&log](const bool before_change,
                   std::size_t /*port_index*/,
                   double /*time_ns*/,
                   uint64_t version) {
                if (before_change) {
                    ++log.before_events;
                } else {
                    ++log.after_events;
                    log.versions.push_back(version);
                }
            });

    // Two real MEM_LOAD nodes on rank 0: their port transactions are the
    // real mutations (insert + streaming-flip + completion settlement).
    OnlineNode mem_a;
    mem_a.global_id = 41;
    mem_a.rank = 0;
    mem_a.kind = NodeKind::MemLoad;
    mem_a.node_type = 2;  // ChakraProtoMsg::MEM_LOAD_NODE
    mem_a.name = "mem_a";
    mem_a.request_id = "r1";
    mem_a.stage = "prefill";
    mem_a.mem.tensor_size = 600;
    OnlineNode mem_b = mem_a;
    mem_b.global_id = 42;
    mem_b.name = "mem_b";
    fixture.source->store().add_node(mem_a);
    fixture.source->store().add_node(mem_b);
    workload.issue_dep_free_nodes();
    drain_queue(fixture.event_queue);

    expect_true(log.before_events >= 2 &&
                    log.after_events == log.before_events &&
                    !log.versions.empty() && log.versions.front() == 1,
                "port: every real port membership change paired a "
                "before/after event with an advancing version");
    expect_true(g_terminal_fires.at(41) == 1 &&
                    g_terminal_fires.at(42) == 1,
                "port: both physical MEM transactions completed exactly "
                "once (the observer never cancelled a transaction)");
    expect_true(remote_memory.is_drained(),
                "port: the backend drained fully (no fabricated "
                "cancellations)");
    remote_memory.remove_port_resource_mutation_observer(observer_id);

    std::filesystem::remove_all(config_dir, ec);
}

}  // namespace

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    CompletionObserver::instance().set_hook(tick_hook, nullptr);

    test_hbm_mutation_mid_arm_settles_and_keeps_physical_callback();
    test_m10_due_tick_both_insertion_orders();
    test_noc_link_resource_mutation_observer();
    test_remote_port_resource_mutation_observer();

    if (!g_ok) {
        std::fprintf(stderr, "[pipeline_mutation_test] FAIL (see above)\n");
        return 1;
    }
    std::printf("[pipeline_mutation_test] ALL PASS\n");
    return 0;
}
