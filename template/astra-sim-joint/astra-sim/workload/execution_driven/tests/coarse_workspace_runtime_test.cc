/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root of this source tree.
*******************************************************************************/

// Coarse workspace/runtime seam: typed COMP Plans share the rank Runtime
// with endpoint transfers, receive capacity before admission, and release the
// exact lease on completion or cancellation.

#include <astra-network-analytical/common/EventQueue.h>

#include <coarse_die/Execution.hh>
#include <coarse_die/Plan.hh>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"
#include "astra-sim/workload/execution_driven/CompletionObserver.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"

using namespace AstraSim;
using namespace AstraSim::ExecutionDriven;
using namespace NetworkAnalytical;
using json = nlohmann::json;

namespace {

bool g_ok = true;
uint64_t g_terminal_count = 0;
std::vector<uint64_t> g_comp_terminal_ticks;

std::string sample_evidence_directory() {
    const char* const path = std::getenv("COARSE_DIE_SAMPLE_EVIDENCE_DIR");
    return path == nullptr ? std::string() : std::string(path);
}

void write_sample_evidence(const std::string& filename, const json& sample) {
    const std::string directory = sample_evidence_directory();
    if (directory.empty()) return;
    std::filesystem::create_directories(directory);
    const auto path = std::filesystem::path(directory) / filename;
    std::ofstream out(path);
    if (!out) {
        throw std::runtime_error("cannot write sample evidence " + path.string());
    }
    out << sample.dump(2) << '\n';
}

void expect(bool condition, const char* message) {
    std::printf("%-78s %s\n", message, condition ? "PASS" : "FAIL");
    if (!condition) g_ok = false;
}

void terminal_hook(void*, int, uint64_t, const char*, const char*, uint64_t,
                   uint64_t, int) {
    ++g_terminal_count;
    g_comp_terminal_ticks.push_back(Sys::boostedTick());
}

class TestRemoteMemoryApi final : public AstraRemoteMemoryAPI {
  public:
    void set_sys(int, Sys*) override {}
    void issue(uint64_t, WorkloadLayerHandlerData*) override {}
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
    void sim_schedule(timespec_t delta, void (*callback)(void*),
                      void* argument) override {
        const timespec_t now = sim_get_time();
        queue_->schedule_event(now.time_val + delta.time_val,
                               callback, argument);
    }
    // Cancellable arm matching the C1 direct-mount contract: coarse
    // wakeups register cancellable Sys events, and Sys fails closed on
    // backends without a real handle. Route through the same queue
    // cancellable API the analytical backend uses.
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

struct Fixture {
    std::shared_ptr<EventQueue> event_queue = std::make_shared<EventQueue>();
    std::unique_ptr<FakeQueueNetworkApi> network_api;
    TestRemoteMemoryApi remote_memory;
    std::shared_ptr<NodeStoreGraphSource> source =
        std::make_shared<NodeStoreGraphSource>();
    std::unique_ptr<Sys> sys;

    Fixture(const std::string& config_dir, const std::string& system_file) {
        network_api = std::make_unique<FakeQueueNetworkApi>(event_queue);
        sys = std::make_unique<Sys>(
            0, config_dir + "/workload", config_dir + "/comm_group.json",
            config_dir + "/" + system_file, &remote_memory,
            network_api.get(), std::vector<int>{1}, std::vector<int>{1},
            1.0, 1.0, false, ExecutionMode::Online, source);
    }
};

json endpoint_json(uint64_t endpoint_id, uint64_t transaction_id,
                   uint64_t physical_bytes) {
    return {
        {"endpoint_id", endpoint_id},
        {"transaction_id", transaction_id},
        {"owner_kind", "restore"},
        {"direction", "write"},
        {"physical_bytes", physical_bytes},
        {"eta_hbm", 0.7},
        {"path_class", "edge_to_hbm"},
        {"route", {{"internal_route", false}}},
    };
}

struct SampleLiveState {
    json counters;
    json progress;
    json plan_cache;
    std::size_t queue_events = 0;
    std::size_t leases = 0;
    uint64_t workspace_bytes = 0;
    uint64_t callbacks = 0;
    uint64_t in_flight_comp = 0;
    bool has_active_executions = false;
    bool has_active_hbm_jobs = false;
};

struct SampleProbe {
    Fixture* fixture = nullptr;
    WorkspaceAdmission* ledger = nullptr;
    uint64_t handle = 0;
    uint64_t* callback_count = nullptr;
    bool called = false;
    bool succeeded = false;
    bool wrong_time_rejected = false;
    std::string error;
    uint64_t sampled_tick = 0;
    SampleLiveState before;
    SampleLiveState after_wrong_time;
    SampleLiveState after_sample;
    json sample;
    json workload_sample;
    bool workload_sample_succeeded = false;
    std::string workload_sample_error;
    bool comp_owner_tamper_rejected = false;
    bool endpoint_identity_tamper_rejected = false;
    SampleLiveState after_catalog_tampers;
    json live_progress_during_sample;
    bool live_handle_active_during_sample = false;
    uint64_t callback_count_during_sample = 0;
};

SampleLiveState observe_sample_live_state(SampleProbe& probe) {
    SampleLiveState state;
    const auto& runtime = *probe.fixture->sys->coarse_die_runtime;
    state.counters = runtime.counters();
    state.progress = runtime.query_progress(probe.handle);
    state.plan_cache = runtime.plan_cache_counters();
    state.queue_events = probe.fixture->event_queue->scheduled_event_count();
    if (probe.ledger != nullptr) {
        state.leases = probe.ledger->active_lease_count();
        state.workspace_bytes = probe.ledger->active_workspace_bytes();
    }
    state.callbacks = probe.callback_count == nullptr ? 0 : *probe.callback_count;
    state.in_flight_comp = probe.fixture->sys->workload->hw_resource
                               ->num_in_flight_gpu_comp_ops;
    state.has_active_hbm_jobs =
        probe.fixture->sys->workload->local_hbm_bandwidth_model
            ->has_active_jobs();
    state.has_active_executions = runtime.has_active_executions();
    return state;
}

bool same_sample_live_state(const SampleLiveState& left,
                            const SampleLiveState& right) {
    return left.counters == right.counters &&
           left.progress == right.progress &&
           left.plan_cache == right.plan_cache &&
           left.queue_events == right.queue_events &&
           left.leases == right.leases &&
           left.workspace_bytes == right.workspace_bytes &&
           left.callbacks == right.callbacks &&
           left.in_flight_comp == right.in_flight_comp &&
           left.has_active_executions == right.has_active_executions &&
           left.has_active_hbm_jobs == right.has_active_hbm_jobs;
}

void sample_runtime_on_queue_tick(void* arg) {
    auto& probe = *static_cast<SampleProbe*>(arg);
    probe.called = true;
    try {
        const auto& runtime = *probe.fixture->sys->coarse_die_runtime;
        probe.sampled_tick = Sys::boostedTick();
        probe.before = observe_sample_live_state(probe);
        try {
            static_cast<void>(runtime.sample_snapshot_at(probe.sampled_tick + 1));
        } catch (const std::exception&) {
            probe.wrong_time_rejected = true;
        }
        probe.after_wrong_time = observe_sample_live_state(probe);
        probe.sample = runtime.sample_snapshot_at(probe.sampled_tick);
        try {
            probe.workload_sample =
                probe.fixture->sys->workload->sample_coarse_local_state_at(
                    probe.sampled_tick);
            probe.workload_sample_succeeded = true;
        } catch (const std::exception& error) {
            probe.workload_sample_error = error.what();
        }
        if (probe.workload_sample_succeeded) {
            probe.after_catalog_tampers = observe_sample_live_state(probe);
            const uint64_t comp_handle =
                probe.workload_sample.at("comp_bindings").at(0)
                    .at("handle").get<uint64_t>();
            json forged_comp_owner = probe.sample;
            bool comp_execution_changed = false;
            bool comp_callback_changed = false;
            for (auto& execution :
                 forged_comp_owner["runtime_snapshot"]["executions"]) {
                if (execution.at("handle").get<uint64_t>() != comp_handle) continue;
                const uint64_t request_id =
                    execution.at("owner").at("request_id").get<uint64_t>();
                execution["owner"]["request_id"] = request_id ^ uint64_t{1};
                comp_execution_changed = true;
            }
            for (auto& binding : forged_comp_owner["completion_bindings"]) {
                if (binding.at("handle").get<uint64_t>() != comp_handle) continue;
                const uint64_t request_id =
                    binding.at("owner").at("request_id").get<uint64_t>();
                binding["owner"]["request_id"] = request_id ^ uint64_t{1};
                comp_callback_changed = true;
            }
            if (comp_execution_changed && comp_callback_changed) {
                try {
                    static_cast<void>(probe.fixture->sys->workload
                        ->die_model_adapter()->coarse_snapshot_bindings(
                            forged_comp_owner));
                } catch (const std::exception&) {
                    probe.comp_owner_tamper_rejected = true;
                }
            }

            const uint64_t endpoint_handle =
                probe.workload_sample.at("endpoint_bindings").at(0)
                    .at("handle").get<uint64_t>();
            json forged_endpoint = probe.sample;
            bool endpoint_changed = false;
            for (auto& execution :
                 forged_endpoint["runtime_snapshot"]["executions"]) {
                if (execution.at("handle").get<uint64_t>() != endpoint_handle) continue;
                auto& transaction_id =
                    execution["plan"]["task"]["endpoint"]["transaction_id"];
                const uint64_t current = transaction_id.get<uint64_t>();
                transaction_id = current ^ uint64_t{1};
                endpoint_changed = true;
            }
            if (endpoint_changed) {
                try {
                    static_cast<void>(probe.fixture->sys->workload
                        ->local_hbm_bandwidth_model->coarse_snapshot_bindings(
                            forged_endpoint));
                } catch (const std::exception&) {
                    probe.endpoint_identity_tamper_rejected = true;
                }
            }
            probe.after_catalog_tampers = observe_sample_live_state(probe);
        }
        probe.live_progress_during_sample = runtime.query_progress(probe.handle);
        probe.live_handle_active_during_sample = runtime.has_active_executions();
        probe.callback_count_during_sample = probe.callback_count == nullptr
            ? 0 : *probe.callback_count;
        probe.after_sample = observe_sample_live_state(probe);
        probe.succeeded = true;
    } catch (const std::exception& error) {
        probe.error = error.what();
    } catch (...) {
        probe.error = "unknown sample callback failure";
    }
}

long double sampled_hbm_bytes(const json& sample) {
    long double total = 0.0L;
    const auto& rows = sample.at("runtime_snapshot").at("stats").at("hbm_served");
    for (const auto& row : rows) {
        for (const auto& amount : row) {
            total += std::stold(amount.get<std::string>());
        }
    }
    return total;
}

json sampled_binding(const json& sample, uint64_t handle) {
    for (const auto& binding : sample.at("completion_bindings")) {
        if (binding.at("handle").get<uint64_t>() == handle) return binding;
    }
    return json::object();
}

json sampled_execution(const json& sample, uint64_t handle) {
    for (const auto& execution :
         sample.at("runtime_snapshot").at("executions")) {
        if (execution.at("handle").get<uint64_t>() == handle) return execution;
    }
    return json::object();
}

void write_text(const std::string& path, const std::string& content) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("cannot write test input " + path);
    out << content;
}

json task_json() {
    return {
        {"schema", "coarse_die_task_v2"},
        {"op_kind", "attention_region"},
        {"phase", "prefill"},
        {"tensor_shapes", {{"m", 64}, {"q_heads", 1},
                            {"kv_heads", 1}, {"head_dim", 32}}},
        {"spans", json::array({{{"q_tokens", 64}, {"kv_tokens", 64},
                                 {"q_start_pos", 0}, {"kv_start_pos", 0}}})},
        // Force the materialized mapping so the selected typed Plan has a
        // concrete HBM scratch reservation for the admission seam.
        {"fusion_policy", "materialized"},
    };
}

OnlineNode coarse_node(const uint64_t id, const json& task) {
    OnlineNode node;
    node.global_id = id;
    node.rank = 0;
    node.kind = NodeKind::Compute;
    node.node_type = 4;  // ChakraProtoMsg::COMP_NODE
    node.name = "typed_coarse_comp";
    node.request_id = "coarse-request";
    node.stage = "prefill";
    node.generation = 0;
    node.compute.op_role = OpRole::Compute;
    node.compute.runtime_ns = 0;
    node.compute.die_task_json =
        std::make_shared<const json>(task);
    return node;
}

void install_capacity(WorkspaceAdmission& ledger, uint64_t capacity,
                      uint64_t revision = 0) {
    WorkspaceCapacityRank snapshot;
    snapshot.rank = ledger.rank();
    snapshot.workspace_revision = revision;
    snapshot.capacity_bytes = capacity;
    snapshot.model_weight_bytes = 100;
    ledger.validate_capacity_snapshot(snapshot);
    ledger.install_capacity_snapshot_unchecked(snapshot);
}

void test_incomplete_coarse_prediction_boundaries(
    const std::string& config_dir) {
    for (const char* field : {"prediction_status", "alone_estimate_status"}) {
        Fixture fixture(config_dir, "coarse_system.json");
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
        nlohmann::json task = task_json();
        task["layer_instances"] = 1;
        task["weight_passes"] = 1;
        OnlineNode comp = coarse_node(950, task);
        comp.generation = 7;
        fixture.source->store().add_node(comp);
        const auto stored = fixture.source->lookup(comp.global_id);
        expect(stored.has_value(),
                    "Q: availability fixture has a dependency-free typed COMP");
        if (!stored.has_value()) continue;

        // This is a legal R1 task on both the old SDK and a new SDK. Do not
        // reduce production layer counts to conceal unsupported R>1 timing.
        const auto prepared = adapter.prepare_coarse_plan(*stored);
        const uint64_t peak = prepared.workspace_bytes;
        expect(peak > 0 && prepared.task != nullptr &&
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
        expect(diagnostic_handle != 0 &&
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
        const uint64_t terminals_before = g_terminal_count;
        const auto terminal_ticks_before = g_comp_terminal_ticks;
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
                callbacks == 0 && g_terminal_count == terminals_before &&
                g_comp_terminal_ticks == terminal_ticks_before &&
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
        expect(submit_rejected && unchanged_execution() &&
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
        expect(acquire_rejected && unchanged_execution() &&
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
        expect(held.status == WorkspaceAdmission::AcquireStatus::Acquired,
                    "Q: valid R1 Prepared Plan still acquires its canonical lease");
        if (held.status != WorkspaceAdmission::AcquireStatus::Acquired) continue;
        const uint64_t held_revision = ledger.workspace_revision();
        bool validation_rejected = false;
        try {
            adapter.validate_prepared_coarse_plan(*stored, bypass, held.lease);
        } catch (const std::exception& error) {
            validation_rejected = availability_error(error);
        }
        expect(validation_rejected && unchanged_execution() &&
                        held.lease.active() && ledger.owns_lease(held.lease, peak) &&
                        ledger.active_lease_count() == 1 &&
                        ledger.active_workspace_bytes() == peak &&
                        ledger.workspace_revision() == held_revision,
                    "Q: validation rejects before take/occupy and preserves the caller's live lease");
        held.lease.release();
        expect(ledger.active_lease_count() == 0 &&
                        ledger.active_workspace_bytes() == 0,
                    "Q: caller explicitly releases the preserved lease exactly once");

        const WorkspaceCapacityRank issue_capacity{
            fixture.sys->id, ledger.workspace_revision(), peak + 4096, 0, 0, 0};
        ledger.validate_capacity_snapshot(issue_capacity);
        ledger.install_capacity_snapshot_unchecked(issue_capacity);
        auto issued = adapter.try_acquire_workspace(prepared);
        expect(issued.status == WorkspaceAdmission::AcquireStatus::Acquired,
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
        expect(issue_rejected && unchanged_execution() &&
                        ledger.active_lease_count() == 0 &&
                        ledger.active_workspace_bytes() == 0,
                    "Q: actual Workload issue rejects manual bypass before graph/GPU/runtime changes");
        fixture.sys.reset();
        while (!fixture.event_queue->finished()) fixture.event_queue->proceed();
        expect(callbacks == 0 && g_terminal_count == terminals_before,
                    "Q: rejected predictions leave no deferred completion on teardown");
    }
}

void test_acquire_block_retry_shared_endpoint_and_release(
    const std::string& config_dir) {
    Fixture fixture(config_dir, "coarse_system.json");
    expect(fixture.sys->coarse_die_runtime != nullptr &&
               fixture.sys->workspace_admission != nullptr &&
               fixture.sys->die_runtime == nullptr,
           "coarse profile creates one independent rank Runtime and ledger");
    if (fixture.sys->coarse_die_runtime == nullptr ||
        fixture.sys->workspace_admission == nullptr) return;

    const json task = task_json();
    const auto plan = fixture.sys->coarse_die_runtime->compile(task);
    const uint64_t peak = coarse_die::hbm_workspace_peak_bytes(*plan);
    expect(peak > 0, "typed attention Plan has an explicit workspace peak");
    if (peak == 0) return;
    const uint64_t capacity = 100 + peak;
    install_capacity(*fixture.sys->workspace_admission, capacity);

    auto held = fixture.sys->workspace_admission->try_acquire(999, 1, peak);
    expect(held.status == WorkspaceAdmission::AcquireStatus::Acquired &&
               fixture.sys->workspace_admission->active_workspace_bytes() == peak,
           "workspace lease is acquired against the installed scalar capacity");

    fixture.source->store().add_node(coarse_node(41, task));
    fixture.sys->workload->issue_dep_free_nodes();
    expect(fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 0 &&
               fixture.source->store().resolve_free_nodes() ==
                   std::vector<uint64_t>({41}),
           "a competing ready Plan stays free and unoccupied while capacity is held");

    held.lease.release();
    expect(fixture.sys->workspace_admission->active_workspace_bytes() == 0,
           "explicit cancellation/release returns the exact byte hold");

    uint64_t sampled_endpoint_completions = 0;
    SampleProbe sample_probe;
    sample_probe.fixture = &fixture;
    sample_probe.ledger = fixture.sys->workspace_admission.get();
    sample_probe.callback_count = &sampled_endpoint_completions;
    // Register the observer first. If CoarseDieRuntime also has a wakeup at
    // tick 1, EventQueue invokes this sampler first from that same EventList.
    fixture.event_queue->schedule_event(1, sample_runtime_on_queue_tick,
                                        &sample_probe);
    fixture.sys->workload->issue_dep_free_nodes();
    expect(fixture.sys->workspace_admission->active_lease_count() == 1 &&
               fixture.sys->workspace_admission->active_workspace_bytes() == peak &&
               fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 1,
           "the same typed Plan acquires workspace before COMP issue");

    bool endpoint_done = false;
    uint64_t endpoint_terminal_tick = 0;
    fixture.sys->workload->local_hbm_bandwidth_model->issue_transfer(
        LocalHbmBandwidthModel::JobKind::COMM_READ, 1000000000ULL, 0, 900,
        [&endpoint_done, &sampled_endpoint_completions,
         &endpoint_terminal_tick] {
            endpoint_done = true;
            endpoint_terminal_tick = Sys::boostedTick();
            ++sampled_endpoint_completions;
        });
    const json initial_local_sample =
        fixture.sys->coarse_die_runtime->sample_snapshot_at(
            Sys::boostedTick());
    for (const auto& binding : initial_local_sample.at("completion_bindings")) {
        if (binding.at("owner").at("node_id").get<uint64_t>() == 900) {
            sample_probe.handle = binding.at("handle").get<uint64_t>();
            break;
        }
    }
    expect(sample_probe.handle != 0,
           "local-HBM registry exposes its real Runtime endpoint handle");
    expect(fixture.sys->coarse_die_runtime->has_active_executions() &&
               fixture.sys->workload->die_model_adapter()
                   ->has_active_die_executions(),
           "Plan and registered local-HBM endpoint share the Runtime");

    std::size_t event_steps = 0;
    while (!sample_probe.called && !fixture.event_queue->finished() &&
           event_steps < 1000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(sample_probe.called && sample_probe.succeeded &&
               sample_probe.error.empty(),
           "Sys-tick sampler returns a complete local snapshot");
    if (sample_probe.succeeded) {
        expect(sample_probe.sampled_tick == 1 &&
                   sample_probe.sample.at("sampled_at_ns") == 1 &&
                   sample_probe.sample.at("schema") ==
                       "coarse_die_local_sample_v1",
               "the sample is bound to the scheduled active Sys tick");
        expect(sample_probe.sample.at("scope") == "rank_local_runtime_only" &&
                   sample_probe.sample.at("callbacks_dispatched") == false &&
                   sample_probe.sample.at("production_world_import_supported") == false,
               "the snapshot discloses its local-only and callback-free boundary");
        expect(sample_probe.comp_owner_tamper_rejected &&
                   sample_probe.endpoint_identity_tamper_rejected &&
                   same_sample_live_state(sample_probe.before,
                                          sample_probe.after_catalog_tampers),
               "COMP/endpoint catalogs reject copied identity tampering without changing live state");
        expect(sampled_hbm_bytes(sample_probe.sample) > 0.0L,
               "sample copy integrates positive actual HBM service");
        const json endpoint_binding = sampled_binding(
            sample_probe.sample, sample_probe.handle);
        expect(!endpoint_binding.empty() &&
                   !endpoint_binding.at("terminal_at_sample").get<bool>(),
               "the long endpoint remains live in the partial sample");
        expect(sample_probe.sample.at("live_runtime_now_ns") == 0,
               "sampling advances only its Runtime copy to the Sys tick");

        expect(sample_probe.workload_sample_succeeded,
               sample_probe.workload_sample_error.empty()
                   ? "Workload returns the joined Runtime/owner sample"
                   : sample_probe.workload_sample_error.c_str());
        if (sample_probe.workload_sample_succeeded) {
            const json& aggregate = sample_probe.workload_sample;
            expect(aggregate.at("schema") ==
                       "coarse_die_workload_local_sample_v1" &&
                       aggregate.at("scope") ==
                           "rank_local_runtime_and_owners" &&
                       aggregate.at("production_world_import_supported") == false,
                   "Workload sample identifies its rank-local owner-only scope");
            expect(aggregate.at("local_runtime") == sample_probe.sample,
                   "Workload aggregation preserves the exact underlying Runtime sample");
            expect(aggregate.at("comp_bindings").size() == 1 &&
                       aggregate.at("endpoint_bindings").size() == 1 &&
                       aggregate.at("local_endpoint_joins").empty(),
                   "owner catalogs bind exactly one COMP and one local transfer endpoint");

            const json comp = aggregate.at("comp_bindings").at(0);
            const json lease = comp.at("workspace_lease");
            expect(comp.at("rank") == 0 && comp.at("node_id") == 41 &&
                       comp.at("generation") == 0 &&
                       comp.at("request_id") == "coarse-request" &&
                       comp.at("stage") == "prefill" &&
                       !comp.at("terminal_at_sample").get<bool>() &&
                       lease.at("rank") == 0 && lease.at("node_id") == 41 &&
                       lease.at("generation") == 0 &&
                       lease.at("bytes") == peak,
                   "COMP catalog retains the actual node, request, stage and exact Plan lease");
            const json workspace = aggregate.at("workspace");
            const json capacity_view = workspace.at("capacity_snapshot");
            expect(workspace.at("rank") == 0 &&
                       workspace.at("active_workspace_bytes") == peak &&
                       workspace.at("active_lease_count") == 1 &&
                       workspace.at("has_capacity_snapshot") == true &&
                       capacity_view.at("capacity_bytes") == capacity &&
                       capacity_view.at("model_weight_bytes") == 100,
                   "workspace sample conserves the live lease against its capacity snapshot");

            const json endpoint = aggregate.at("endpoint_bindings").at(0);
            expect(endpoint.at("domain_rank") == 0 &&
                       endpoint.at("completion_rank") == 0 &&
                       endpoint.at("node_id") == 900 &&
                       endpoint.at("job_kind") == "comm_read" &&
                       endpoint.at("transfer_sink") == true &&
                       !endpoint.at("terminal_at_sample").get<bool>(),
                   "local HBM catalog retains its real rank, node, kind and callback sink");
            const json endpoint_execution = sampled_execution(
                aggregate.at("local_runtime"),
                endpoint.at("handle").get<uint64_t>());
            const json endpoint_callback = sampled_binding(
                aggregate.at("local_runtime"),
                endpoint.at("handle").get<uint64_t>());
            const json raw_endpoint =
                endpoint_execution.at("plan").at("task").at("endpoint");
            const json owner = endpoint_execution.at("owner");
            expect(raw_endpoint == endpoint.at("endpoint") &&
                       raw_endpoint.at("owner_kind") == "comm" &&
                       raw_endpoint.at("direction") == "read" &&
                       raw_endpoint.at("path_class") == "hbm_to_edge" &&
                       raw_endpoint.at("physical_bytes") == 1000000000ULL &&
                       raw_endpoint.at("eta_hbm") ==
                           fixture.sys->coarse_die_runtime->hardware()
                               .input.at("performance_profile")
                               .at("eta_hbm_matrix") &&
                       raw_endpoint.at("route").at("internal_route") == false,
                   "endpoint catalog preserves the complete Runtime endpoint request");
            expect(owner == endpoint.at("owner") &&
                       owner == endpoint_callback.at("owner") &&
                       endpoint.at("terminal_at_sample") ==
                           endpoint_callback.at("terminal_at_sample") &&
                       endpoint.at("finish_ns_at_sample") ==
                           endpoint_callback.at("finish_ns_at_sample"),
                   "endpoint owner and callback terminal metadata match the raw Runtime binding");
        }

        bool raw_import_parity = false;
        try {
            const auto restored = coarse_die::Runtime::import_snapshot(
                fixture.sys->coarse_die_runtime->hardware(),
                sample_probe.sample.at("runtime_snapshot"));
            raw_import_parity =
                restored->export_snapshot() ==
                    sample_probe.sample.at("runtime_snapshot") &&
                !restored->query_progress(sample_probe.handle)
                     .at("terminal").get<bool>();
        } catch (const std::exception&) {
            raw_import_parity = false;
        }
        expect(raw_import_parity,
               "sampled raw Runtime v3 snapshot imports and re-exports exactly");

        expect(sample_probe.wrong_time_rejected &&
                   same_sample_live_state(sample_probe.before,
                                          sample_probe.after_wrong_time),
               "wrong-tick sample rejection leaves Runtime, lease, cache and queue unchanged");
        expect(same_sample_live_state(sample_probe.before,
                                      sample_probe.after_sample),
               "valid sampling leaves live progress, accounting, lease, cache and queue unchanged");
        expect(sample_probe.before.leases == 1 &&
                   sample_probe.before.workspace_bytes == peak &&
                   sample_probe.before.in_flight_comp == 1 &&
                   sample_probe.before.has_active_hbm_jobs &&
                   sample_probe.callback_count_during_sample == 0,
               "sampling preserves the active COMP lease and pending transfer callback");
        if (sample_probe.workload_sample_succeeded) {
            write_sample_evidence("partial-workload-sample.json",
                                  sample_probe.workload_sample);
        }
    }
    expect(sampled_endpoint_completions == 0,
           "partial sampling never invokes a real endpoint completion callback");

    while (fixture.sys->workspace_admission->active_lease_count() != 0 &&
           !fixture.event_queue->finished() && event_steps < 100000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(fixture.sys->workspace_admission->active_lease_count() == 0 &&
               fixture.sys->workspace_admission->active_workspace_bytes() == 0,
           "normal typed Plan completion releases its one workspace lease");
    expect(!endpoint_done && sampled_endpoint_completions == 0 &&
               fixture.sys->coarse_die_runtime->has_active_executions(),
           "long endpoint remains active after the COMP lease releases");

    while (!fixture.event_queue->finished() && event_steps < 200000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(endpoint_done && sampled_endpoint_completions == 1 &&
               !fixture.sys->coarse_die_runtime->has_active_executions() &&
               g_terminal_count == 1,
           "shared Runtime drains endpoint and emits exactly one COMP terminal");

    const uint64_t drained_sample_tick = Sys::boostedTick();
    const json drained_sample =
        fixture.sys->workload->sample_coarse_local_state_at(
            drained_sample_tick);
    expect(drained_sample.at("local_runtime").at("sampled_at_ns") ==
                   drained_sample_tick &&
               drained_sample.at("comp_bindings").empty() &&
               drained_sample.at("endpoint_bindings").empty() &&
               drained_sample.at("workspace").at("active_lease_count") == 0 &&
               drained_sample.at("workspace").at("active_workspace_bytes") == 0,
           "queue-drained Workload sample has no live callbacks or workspace holds");
    write_sample_evidence("drained-workload-sample.json", drained_sample);
    write_sample_evidence("terminal-ticks.json", {
        {"schema", "coarse_die_terminal_ticks_v1"},
        {"comp_terminal_ticks_ns", g_comp_terminal_ticks},
        {"local_hbm_transfer_terminal_tick_ns", endpoint_terminal_tick},
    });
}

void test_adapter_cancellation_releases_lease_without_terminal(
    const std::string& config_dir) {
    Fixture fixture(config_dir, "coarse_system.json");
    const json task = task_json();
    const auto plan = fixture.sys->coarse_die_runtime->compile(task);
    const uint64_t peak = coarse_die::hbm_workspace_peak_bytes(*plan);
    install_capacity(*fixture.sys->workspace_admission, 100 + peak);
    fixture.source->store().add_node(coarse_node(42, task));
    fixture.sys->workload->issue_dep_free_nodes();
    const uint64_t terminals_before = g_terminal_count;
    expect(fixture.sys->workspace_admission->active_lease_count() == 1 &&
               fixture.sys->coarse_die_runtime->has_active_executions(),
           "cancellation fixture starts with a live Plan and lease");

    fixture.sys->workload->die_model_adapter()->shutdown();
    expect(fixture.sys->workspace_admission->active_lease_count() == 0 &&
               fixture.sys->workspace_admission->active_workspace_bytes() == 0 &&
               !fixture.sys->coarse_die_runtime->has_active_executions(),
           "adapter cancellation destroys Runtime execution and releases lease");
    expect(g_terminal_count == terminals_before,
           "cancellation never synthesizes an outer COMP completion");
}

void test_invalid_prepared_workspace_is_rejected_before_issue(
    const std::string& config_dir, const uint64_t node_id,
    const bool foreign_ledger, const bool wrong_lease_bytes,
    const bool wrong_prepared_bytes, const bool stale_node_task,
    const char* description) {
    Fixture fixture(config_dir, "coarse_system.json");
    const json task = task_json();
    auto& runtime = *fixture.sys->coarse_die_runtime;
    auto& admission = *fixture.sys->workspace_admission;
    const auto prepared_node = coarse_node(node_id, task);
    fixture.source->store().add_node(prepared_node);
    const auto node = fixture.source->lookup(node_id);
    expect(node.has_value(), "negative workspace case has a live ready NodeView");
    if (!node.has_value()) return;

    auto prepared = fixture.sys->workload->die_model_adapter()
                        ->prepare_coarse_plan(*node);
    const uint64_t peak = prepared.workspace_bytes;
    expect(peak > 0, "negative workspace case prepares a nonzero canonical peak");
    if (peak == 0) return;
    install_capacity(admission, 100 + peak + 1);

    WorkspaceAdmission foreign(0);
    WorkspaceAdmission* lease_owner = &admission;
    if (foreign_ledger) {
        install_capacity(foreign, 100 + peak + 1);
        lease_owner = &foreign;
    }
    const uint64_t lease_bytes = wrong_lease_bytes ? peak + 1 : peak;
    auto acquired = lease_owner->try_acquire(node_id, node->generation,
                                              lease_bytes);
    expect(acquired.status == WorkspaceAdmission::AcquireStatus::Acquired,
           "negative workspace case acquires its deliberately adversarial lease");
    if (acquired.status != WorkspaceAdmission::AcquireStatus::Acquired) return;

    const auto& key = acquired.lease.key();
    expect(key.rank == 0 && key.node_id == node_id &&
               key.generation == node->generation,
           "adversarial lease matches the expected rank/node/generation identity");
    if (foreign_ledger) {
        auto same_identity = admission.try_acquire(
            node_id, node->generation, peak);
        expect(same_identity.status == WorkspaceAdmission::AcquireStatus::Acquired,
               "authoritative ledger can issue the same visible lease identity");
        if (same_identity.status == WorkspaceAdmission::AcquireStatus::Acquired) {
            const auto& authoritative_key = same_identity.lease.key();
            expect(authoritative_key.rank == key.rank &&
                       authoritative_key.node_id == key.node_id &&
                       authoritative_key.generation == key.generation &&
                       authoritative_key.lease_id == key.lease_id,
                   "foreign token collides on every public LeaseKey field");
            same_identity.lease.release();
        }
    }
    if (wrong_prepared_bytes) prepared.workspace_bytes = peak + 1;
    auto submitted_node = *node;
    if (stale_node_task) {
        json substituted_task = task;
        substituted_task["tensor_shapes"]["head_dim"] = 64;
        submitted_node.compute.die_task_json =
            std::make_shared<const json>(std::move(substituted_task));
    }

    const uint64_t terminals_before = g_terminal_count;
    bool rejected = false;
    try {
        fixture.sys->workload->issue_prepared_coarse(
            submitted_node, std::move(prepared), std::move(acquired.lease));
    } catch (const std::exception&) {
        rejected = true;
    }
    expect(rejected, description);
    expect(fixture.source->store().resolve_free_nodes() ==
                   std::vector<uint64_t>({node_id}),
           "invalid prepared workspace leaves the COMP in the ready set");
    const auto still_stored = fixture.source->lookup(node_id);
    expect(still_stored.has_value() &&
               still_stored->compute.die_task_json != nullptr &&
               *still_stored->compute.die_task_json == task,
           "preflight rejection leaves the source NodeStore payload untouched");
    expect(fixture.sys->workload->hw_resource->num_in_flight_gpu_comp_ops == 0 &&
               !runtime.has_active_executions() &&
               !fixture.sys->workload->die_model_adapter()
                    ->has_active_die_executions() &&
               g_terminal_count == terminals_before,
           "invalid workspace fails before Runtime submission, COMP occupancy or terminal");
    expect(admission.active_lease_count() == 0 &&
               admission.active_workspace_bytes() == 0 &&
               foreign.active_lease_count() == 0 &&
               foreign.active_workspace_bytes() == 0,
           "by-value lease destruction releases the exact authoritative or foreign hold");
    // Keep even a failing negative path safe: a mistakenly accepted foreign
    // token must not outlive its stack-local ledger during Fixture teardown.
    fixture.sys->workload->die_model_adapter()->shutdown();
}

void test_workspace_lease_ownership_and_exact_bytes(
    const std::string& config_dir) {
    test_invalid_prepared_workspace_is_rejected_before_issue(
        config_dir, 51, true, false, false, false,
        "same-rank foreign-ledger lease is rejected before COMP issue");
    test_invalid_prepared_workspace_is_rejected_before_issue(
        config_dir, 52, false, true, false, false,
        "authoritative lease with noncanonical bytes is rejected before COMP issue");
    test_invalid_prepared_workspace_is_rejected_before_issue(
        config_dir, 53, false, false, true, false,
        "prepared workspace bytes differing from the Plan peak are rejected before COMP issue");
    test_invalid_prepared_workspace_is_rejected_before_issue(
        config_dir, 54, false, false, false, true,
        "stale typed task payload is rejected before COMP issue");
}

void test_terminal_sample_keeps_live_callback_pending(
    const std::string& config_dir) {
    Fixture fixture(config_dir, "coarse_system.json");
    auto& runtime = *fixture.sys->coarse_die_runtime;
    const json tiny_endpoint = endpoint_json(7101, 7102, 1);
    const coarse_die::Owner owner{7103, 7104, 7105, 1};

    // This direct user callback deliberately has no LocalHbm/Adapter owner
    // registry, so this case exercises only the raw Runtime snapshot contract.
    // Predict this one-byte endpoint's real Runtime terminal tick with an
    // isolated instance. Insert the sampler before the live submission so
    // EventQueue's same-tick FIFO observes the terminal copy before the
    // CoarseDieRuntime wakeup dispatches its bound callback.
    coarse_die::Runtime reference(runtime.hardware());
    const uint64_t reference_handle =
        reference.submit_endpoint(tiny_endpoint, owner, 0);
    const auto terminal_tick = reference.next_event();
    expect(terminal_tick.has_value(),
           "tiny endpoint has one finite native terminal boundary");
    if (!terminal_tick.has_value()) return;
    reference.advance_to(*terminal_tick);
    expect(reference.query_progress(reference_handle).at("terminal").get<bool>(),
           "reference Runtime confirms the scheduled endpoint tick is terminal");

    uint64_t callback_count = 0;
    json callback_progress;
    SampleProbe probe;
    probe.fixture = &fixture;
    probe.callback_count = &callback_count;
    fixture.event_queue->schedule_event(*terminal_tick,
                                        sample_runtime_on_queue_tick, &probe);
    const uint64_t live_handle = runtime.submit_endpoint(
        tiny_endpoint, owner,
        [&callback_count, &callback_progress](
            const uint64_t, const json& progress) {
            ++callback_count;
            callback_progress = progress;
        });
    probe.handle = live_handle;
    expect(live_handle == reference_handle,
           "live and isolated endpoint runs use the same native handle sequence");

    std::size_t event_steps = 0;
    while (!probe.called && !fixture.event_queue->finished() &&
           event_steps < 1000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(probe.called && probe.succeeded && probe.error.empty(),
           "same-tick terminal sampler runs before the live Runtime callback");
    expect(!probe.workload_sample_succeeded &&
               probe.workload_sample_error.find(
                   "unbound Runtime callback or owner") != std::string::npos,
           "Workload aggregation fails closed for a direct endpoint without an owner registry");
    if (probe.succeeded) {
        write_sample_evidence("terminal-local-runtime-sample.json",
                              probe.sample);
        const json binding = sampled_binding(probe.sample, live_handle);
        expect(probe.sampled_tick == *terminal_tick &&
                   binding.at("terminal_at_sample").get<bool>() &&
                   binding.at("state_at_sample") == "finished" &&
                   binding.at("finish_ns_at_sample") == *terminal_tick,
               "sample copy reaches terminal and retains its completion binding");
        expect(!probe.live_progress_during_sample.at("terminal").get<bool>() &&
                   probe.live_handle_active_during_sample &&
                   probe.callback_count_during_sample == 0,
               "live handle and callback remain pending while sample copy is terminal");
        expect(sampled_hbm_bytes(probe.sample) > 0.0L,
               "terminal local sample retains the endpoint's served bytes");

        bool raw_import_parity = false;
        try {
            const auto restored = coarse_die::Runtime::import_snapshot(
                runtime.hardware(), probe.sample.at("runtime_snapshot"));
            raw_import_parity =
                restored->export_snapshot() ==
                    probe.sample.at("runtime_snapshot") &&
                restored->query_progress(live_handle)
                    .at("terminal").get<bool>();
        } catch (const std::exception&) {
            raw_import_parity = false;
        }
        expect(raw_import_parity,
               "terminal sampled raw Runtime v3 snapshot imports with its handle intact");
        expect(probe.wrong_time_rejected &&
                   same_sample_live_state(probe.before,
                                          probe.after_wrong_time),
               "wrong-tick rejection does not mutate the live terminal binding or queue");
        expect(same_sample_live_state(probe.before, probe.after_sample),
               "terminal sampling itself leaves all live endpoint state unchanged");
    }

    while (!fixture.event_queue->finished() && event_steps < 1000) {
        fixture.event_queue->proceed();
        ++event_steps;
    }
    expect(callback_count == 1 &&
               callback_progress.at("terminal").get<bool>() &&
               callback_progress.at("finish_ns") == *terminal_tick &&
               !runtime.has_active_executions(),
           "the real queue dispatches exactly one terminal and retires its handle");
}

}  // namespace

int main() {
    const std::string tmp_template = "/tmp/coarse_workspace_runtime_XXXXXX";
    std::vector<char> directory(tmp_template.begin(), tmp_template.end());
    directory.push_back('\0');
    const char* made = mkdtemp(directory.data());
    if (made == nullptr) {
        std::fprintf(stderr, "[coarse_workspace_runtime_test] mkdtemp failed\n");
        return 1;
    }
    const std::string config_dir(made);
    try {
        std::filesystem::copy_file(
            std::string(COARSE_DIE_SOURCE_DIR) + "/configs/example_hardware.json",
            config_dir + "/coarse_hardware.json");
        write_text(config_dir + "/comm_group.json", "{}");
        write_text(config_dir + "/coarse_system.json",
            std::string(R"({
  "scheduling-policy": "FIFO",
  "compute-cost-model": "die",
  "die-model-profile": "coarse_die_v2",
  "die-model-config": ")") + config_dir + R"(/coarse_hardware.json",
  "local-mem-latency": 20
})");
        CompletionObserver::instance().set_hook(terminal_hook, nullptr);
        test_incomplete_coarse_prediction_boundaries(config_dir);
        test_acquire_block_retry_shared_endpoint_and_release(config_dir);
        test_workspace_lease_ownership_and_exact_bytes(config_dir);
        test_terminal_sample_keeps_live_callback_pending(config_dir);
        test_adapter_cancellation_releases_lease_without_terminal(config_dir);
        CompletionObserver::instance().set_hook(nullptr, nullptr);
    } catch (const std::exception& error) {
        std::fprintf(stderr,
                     "[coarse_workspace_runtime_test] exception: %s\n",
                     error.what());
        g_ok = false;
    }
    std::filesystem::remove_all(config_dir);
    if (g_ok) {
        std::printf("[coarse_workspace_runtime_test] ALL PASS\n");
        return 0;
    }
    return 1;
}
