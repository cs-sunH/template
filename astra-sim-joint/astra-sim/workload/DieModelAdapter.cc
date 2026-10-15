/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/CoarseDiePrediction.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <exception>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>

#include "DieModel.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/DieExecution.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/Workload.hh"

namespace AstraSim {
namespace {

class ScopedNanosecondTimer final {
  public:
    explicit ScopedNanosecondTimer(uint64_t& total_ns)
        : total_ns_(total_ns), start_(std::chrono::steady_clock::now()) {}
    ~ScopedNanosecondTimer() {
        total_ns_ += static_cast<uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start_)
                .count());
    }

  private:
    uint64_t& total_ns_;
    std::chrono::steady_clock::time_point start_;
};

uint64_t add_quality_counter(const uint64_t left, const uint64_t right,
                             const char* const label) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error(std::string("DieModelAdapter ") + label +
                                 " counter overflows uint64");
    }
    return left + right;
}

uint64_t stable_request_id(const std::string& value) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash;
}

}  // namespace

DieModelAdapter::DieModelAdapter(Workload* workload)
    : workload_(workload), sys_(workload != nullptr ? workload->sys : nullptr) {
    if (workload_ == nullptr || sys_ == nullptr ||
        (sys_->die_runtime == nullptr &&
         sys_->coarse_die_runtime == nullptr)) {
        throw std::runtime_error(
            "DieModelAdapter requires a Workload whose Sys loaded the die "
            "runtime (compute-cost-model=die with a valid die-model-config)");
    }
}

DieModelAdapter::~DieModelAdapter() {
    shutdown();
}

void DieModelAdapter::issue_die(const ExecutionDriven::NodeView& node) {
    IssueTiming& timing = mutable_cumulative_issue_timing();
    ++timing.calls;
    ScopedNanosecondTimer issue_timer(timing.total_issue_ns);
    purge_finished_executions();
    if (sys_->coarse_die_runtime != nullptr) {
        throw std::runtime_error(
            "coarse_die_v2 COMP must prepare a Plan, acquire its workspace "
            "lease, and submit that same Plan");
    }
    if (node.compute.die_spec == nullptr) {
        // Contract §2/§7.3: a real COMP node without a die_op_spec must fail
        // closed -- never a silent roofline/replay fallback.
        throw std::runtime_error(
            "die backend requires die_op_spec on real COMP node " +
            std::to_string(node.global_id) +
            " (op_role=compute without a spec is a generator/regression "
            "defect; refusing to fall back to roofline or replay)");
    }
    const auto pipeline_contract = workload_->pipeline_kernel_contract();
    if (workload_->requires_pipeline_kernel_contract() &&
        pipeline_contract == nullptr) {
        throw std::runtime_error(
            "DieModelAdapter refuses an online DIE node before the first valid "
            "GraphBatch freezes its PipelineKernelContract");
    }

    Sys::DieModelRuntime& runtime = *sys_->die_runtime;
    std::string key;
    ++timing.plan_key_calls;
    {
        ScopedNanosecondTimer key_timer(timing.plan_key_ns);
        key = DieModel::plan_cache_key(*node.compute.die_spec,
                                       runtime.hardware);
    }

    ++timing.cache_get_calls;
    std::shared_ptr<const DieModel::DiePlan> plan;
    {
        ScopedNanosecondTimer cache_get_timer(timing.cache_get_ns);
        plan = runtime.plan_cache->get(key);
    }
    if (plan == nullptr) {
        ++timing.cache_misses;
        // compile() is a pure function (DieModelError on any mapping
        // violation); the immutable result enters the LRU cache, the
        // mutable execution handle never does (design doc §8.5).  The cache
        // is process-level shared across every Sys of the same resolved
        // hardware (see Sys.hh), so a rank reuses plans other ranks
        // compiled for the identical (spec, hardware) key.
        ++timing.compile_calls;
        {
            ScopedNanosecondTimer compile_timer(timing.compile_ns);
            plan = DieModel::compile(*node.compute.die_spec,
                                     runtime.hardware);
        }
        ++timing.cache_put_calls;
        {
            ScopedNanosecondTimer cache_put_timer(timing.cache_put_ns);
            runtime.plan_cache->put(key, plan);
        }
    } else {
        ++timing.cache_hits;
    }

    ++timing.setup_calls;
    DieExecution* raw = nullptr;
    {
        ScopedNanosecondTimer setup_timer(timing.setup_ns);
        WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
        wlhd->sys_id = sys_->id;
        wlhd->workload = workload_;
        wlhd->node_id = node.global_id;
        auto execution = std::make_unique<DieExecution>(
            sys_, workload_, this, node.global_id, wlhd, plan,
            pipeline_contract);
        raw = execution.get();
        active_.push_back(std::move(execution));
    }
    // start() may complete the node synchronously (a zero-work plan reports
    // Done on the first action); the finished execution stays in active_
    // until the next purge.
    ++timing.start_calls;
    {
        ScopedNanosecondTimer start_timer(timing.start_ns);
        raw->start();
    }
}

DieModelAdapter::PreparedCoarsePlan DieModelAdapter::prepare_coarse_plan(
    const ExecutionDriven::NodeView& node) {
    purge_finished_executions();
    if (sys_->coarse_die_runtime == nullptr ||
        sys_->workspace_admission == nullptr) {
        throw std::runtime_error(
            "coarse workspace preparation requires a coarse_die_v2 rank");
    }
    if (node.node_type != ChakraNodeType::COMP_NODE || node.is_timer_op ||
        node.is_cpu_op || node.compute.op_role ==
                               ExecutionDriven::OpRole::ControlMarker ||
        node.compute.die_task_json == nullptr) {
        throw std::runtime_error(
            "workspace preparation requires a real GPU coarse COMP node " +
            std::to_string(node.global_id));
    }
    auto plan = sys_->coarse_die_runtime->compile(*node.compute.die_task_json);
    if (!plan) {
        throw std::runtime_error("coarse_die_v2 compiler returned a null Plan");
    }
    coarse_die::require_complete_prediction(*plan);
    const uint64_t workspace_bytes =
        coarse_die::hbm_workspace_peak_bytes(*plan);
    return PreparedCoarsePlan{std::move(plan), workspace_bytes,
                              node.global_id, node.generation,
                              node.compute.die_task_json};
}

WorkspaceAdmission::AcquireResult DieModelAdapter::try_acquire_workspace(
    const PreparedCoarsePlan& prepared) {
    if (sys_ == nullptr || sys_->coarse_die_runtime == nullptr ||
        sys_->workspace_admission == nullptr || !prepared.plan ||
        !prepared.task) {
        throw std::runtime_error(
            "workspace acquisition requires a prepared coarse Plan");
    }
    coarse_die::require_complete_prediction(*prepared.plan);
    const uint64_t canonical_workspace_bytes =
        coarse_die::hbm_workspace_peak_bytes(*prepared.plan);
    if (sys_->workspace_admission->rank() != sys_->id ||
        prepared.workspace_bytes != canonical_workspace_bytes ||
        prepared.plan->fingerprint.empty() ||
        prepared.plan->hardware_fingerprint !=
            sys_->coarse_die_runtime->hardware().fingerprint) {
        throw std::runtime_error(
            "prepared coarse Plan workspace or hardware identity is not canonical");
    }
    return sys_->workspace_admission->try_acquire(
        prepared.node_id, prepared.generation, prepared.workspace_bytes);
}

void DieModelAdapter::validate_prepared_coarse_plan(
    const ExecutionDriven::NodeView& node,
    const PreparedCoarsePlan& prepared,
    const WorkspaceAdmission::Lease& lease) const {
    if (sys_ == nullptr || sys_->coarse_die_runtime == nullptr ||
        sys_->workspace_admission == nullptr || !prepared.plan ||
        !prepared.task || !lease.active()) {
        throw std::runtime_error(
            "coarse issue requires a live prepared Plan and workspace lease");
    }
    if (node.node_type != ChakraNodeType::COMP_NODE || node.is_timer_op ||
        node.is_cpu_op ||
        node.compute.op_role != ExecutionDriven::OpRole::Compute ||
        node.compute.die_task_json == nullptr || node.compute.die_spec != nullptr) {
        throw std::runtime_error(
            "prepared coarse issue requires a typed GPU COMP node");
    }
    coarse_die::require_complete_prediction(*prepared.plan);
    const uint64_t canonical_workspace_bytes =
        coarse_die::hbm_workspace_peak_bytes(*prepared.plan);
    if (sys_->workspace_admission->rank() != sys_->id ||
        prepared.node_id != node.global_id ||
        prepared.generation != node.generation ||
        *prepared.task != *node.compute.die_task_json ||
        prepared.workspace_bytes != canonical_workspace_bytes ||
        prepared.plan->fingerprint.empty() ||
        prepared.plan->hardware_fingerprint !=
            sys_->coarse_die_runtime->hardware().fingerprint ||
        !sys_->workspace_admission->owns_lease(
            lease, canonical_workspace_bytes)) {
        throw std::runtime_error(
            "prepared Plan/workspace lease identity or canonical workspace "
            "does not match COMP node and rank ledger");
    }
    const auto& key = lease.key();
    if (key.rank != sys_->id || key.node_id != node.global_id ||
        key.generation != node.generation) {
        throw std::runtime_error(
            "workspace lease key differs from the prepared COMP identity");
    }
}

void DieModelAdapter::issue_prepared_coarse_plan(
    const ExecutionDriven::NodeView& node,
    PreparedCoarsePlan prepared,
    WorkspaceAdmission::Lease lease) {
    validate_prepared_coarse_plan(node, prepared, lease);
    purge_finished_executions();

    coarse_die::Owner owner;
    owner.request_id = stable_request_id(node.request_id);
    owner.node_id = node.global_id;
    owner.execution_id = node.generation;
    owner.generation = std::max<uint64_t>(node.generation, 1);
    const uint64_t node_id = node.global_id;
    const uint64_t handle = sys_->coarse_die_runtime->submit_plan(
        prepared.plan, owner,
        [this](const uint64_t done_handle,
               const nlohmann::json& progress) {
            auto found = coarse_active_.find(done_handle);
            if (found == coarse_active_.end()) {
                throw std::runtime_error(
                    "coarse completion has no active Plan/workspace record "
                    "for handle " + std::to_string(done_handle));
            }
            CoarseActiveRecord active = std::move(found->second);
            coarse_active_.erase(found);
            active.lease.release();

            const auto state = progress.value("state", std::string());
            if (state != "finished") {
                throw std::runtime_error(
                    "coarse COMP completion has invalid terminal state " +
                    state);
            }
            const uint64_t elapsed_ns =
                progress.value("elapsed_ns", uint64_t{0});
            const double elapsed_s = static_cast<double>(elapsed_ns) / 1e9;
            const auto& derived =
                sys_->coarse_die_runtime->hardware_summary();
            const auto& hw_summary = derived.at("hardware_summary");
            const double int4_peak =
                hw_summary.at("peak_int4_ops_per_s").get<double>();
            const double vec_peak =
                hw_summary.at("peak_vec_fp32_eq_ops_per_s").get<double>();
            const double hbm_peak =
                hw_summary.at("hbm_row_peak_Bps")[0].get<double>() +
                hw_summary.at("hbm_row_peak_Bps")[1].get<double>();
            const double int4_ops =
                progress.at("served").at("int4_ops").get<double>();
            const double vec_ops =
                progress.at("served").at("vec_eqops").get<double>();
            const auto& matrix = progress.at("hbm_served_bytes");
            if (!matrix.is_array() || matrix.size() != 2 ||
                !matrix[0].is_array() || matrix[0].size() != 2 ||
                !matrix[1].is_array() || matrix[1].size() != 2) {
                throw std::runtime_error(
                    "coarse_die_v2 completion is missing its 2x2 HBM "
                    "served matrix");
            }
            const double hbm_bytes =
                matrix[0][0].get<double>() + matrix[0][1].get<double>() +
                matrix[1][0].get<double>() + matrix[1][1].get<double>();
            if (!(int4_peak > 0.0) || !std::isfinite(int4_peak) ||
                !(vec_peak > 0.0) || !std::isfinite(vec_peak) ||
                !(hbm_peak > 0.0) || !std::isfinite(hbm_peak) ||
                !std::isfinite(int4_ops) || int4_ops < 0.0 ||
                !std::isfinite(vec_ops) || vec_ops < 0.0 ||
                !std::isfinite(hbm_bytes) || hbm_bytes < 0.0) {
                throw std::runtime_error(
                    "coarse_die_v2 completion is missing finite typed plan "
                    "work or hardware peaks");
            }
            const bool any_served = int4_ops != 0.0 || vec_ops != 0.0 ||
                                    hbm_bytes != 0.0;
            if (elapsed_ns == 0 && any_served) {
                throw std::runtime_error(
                    "coarse_die_v2 completion reports served work with a "
                    "zero elapsed-time denominator");
            }
            const double compute_utilization = elapsed_ns == 0 ? 0.0
                : int4_ops / (int4_peak * elapsed_s);
            const double vector_utilization = elapsed_ns == 0 ? 0.0
                : vec_ops / (vec_peak * elapsed_s);
            const double memory_utilization = elapsed_ns == 0 ? 0.0
                : hbm_bytes / (hbm_peak * elapsed_s);
            if (!std::isfinite(compute_utilization) ||
                !std::isfinite(vector_utilization) ||
                !std::isfinite(memory_utilization)) {
                throw std::runtime_error(
                    "coarse_die_v2 completion utilization is non-finite");
            }
            workload_->record_die_utilizations(
                active.node_id, compute_utilization, memory_utilization,
                vector_utilization);
            workload_->finish_general_node(active.node_id, EventType::General);
        });
    try {
        const auto inserted = coarse_active_.try_emplace(handle);
        if (!inserted.second) {
            throw std::runtime_error(
                "coarse Runtime reused an active execution handle");
        }
        inserted.first->second.node_id = node_id;
        inserted.first->second.generation = node.generation;
        inserted.first->second.plan = std::move(prepared.plan);
        inserted.first->second.lease = std::move(lease);
    } catch (...) {
        try {
            sys_->coarse_die_runtime->cancel(
                handle, Sys::boostedTick(), /*notify=*/false);
        } catch (...) {
            std::throw_with_nested(std::runtime_error(
                "coarse Plan submitted but adapter registration failed and "
                "Runtime cancellation could not be confirmed"));
        }
        throw;
    }
}

nlohmann::json DieModelAdapter::coarse_snapshot_bindings(
    const nlohmann::json& local_sample) const {
    using Json = nlohmann::json;
    const auto reject = [](const std::string& reason) {
        throw std::runtime_error("coarse DIE sample binding: " + reason);
    };
    if (workload_ == nullptr || sys_ == nullptr ||
        sys_->coarse_die_runtime == nullptr ||
        sys_->workspace_admission == nullptr) {
        reject("requires a live coarse_die_v2 Runtime and workspace ledger");
    }
    if (!local_sample.is_object()) reject("sample envelope must be an object");
    const auto member = [&](const Json& object, const char* name) -> const Json& {
        if (!object.is_object() || !object.contains(name)) {
            reject(std::string("sample is missing ") + name);
        }
        return object.at(name);
    };
    const auto uint64_value = [&](const Json& value, const char* name) {
        if (value.is_number_unsigned()) return value.get<uint64_t>();
        if (value.is_number_integer()) {
            const int64_t signed_value = value.get<int64_t>();
            if (signed_value >= 0) return static_cast<uint64_t>(signed_value);
        }
        reject(std::string(name) + " must be an exact nonnegative integer");
        return uint64_t{0};
    };
    const auto string_value = [&](const Json& value, const char* name) {
        if (!value.is_string()) reject(std::string(name) + " must be a string");
        return value.get<std::string>();
    };
    const auto bool_value = [&](const Json& value, const char* name) {
        if (!value.is_boolean()) reject(std::string(name) + " must be a boolean");
        return value.get<bool>();
    };

    if (string_value(member(local_sample, "schema"), "schema") !=
        "coarse_die_local_sample_v1") {
        reject("unsupported local sample schema");
    }
    if (sys_->id < 0 || uint64_value(member(local_sample, "rank"), "rank") !=
                            static_cast<uint64_t>(sys_->id)) {
        reject("sample rank differs from this Sys");
    }
    const uint64_t sample_tick =
        uint64_value(member(local_sample, "sampled_at_ns"), "sampled_at_ns");
    if (sample_tick != Sys::boostedTick()) {
        reject("sample clock differs from the active Sys decision tick");
    }
    const std::string hardware_fingerprint =
        sys_->coarse_die_runtime->hardware().fingerprint;
    if (hardware_fingerprint.empty() ||
        string_value(member(local_sample, "hardware_fingerprint"),
                     "hardware_fingerprint") != hardware_fingerprint) {
        reject("sample hardware fingerprint differs from this Runtime");
    }

    const Json& runtime_snapshot = member(local_sample, "runtime_snapshot");
    if (!runtime_snapshot.is_object() ||
        string_value(member(runtime_snapshot, "schema"),
                     "runtime_snapshot.schema") != "coarse_die_snapshot_v3" ||
        uint64_value(member(runtime_snapshot, "snapshot_version"),
                     "runtime_snapshot.snapshot_version") != 3) {
        reject("Runtime snapshot must use coarse_die_snapshot_v3");
    }
    if (string_value(member(runtime_snapshot, "hardware_fingerprint"),
                     "runtime_snapshot.hardware_fingerprint") !=
            hardware_fingerprint ||
        uint64_value(member(runtime_snapshot, "now_ns"),
                     "runtime_snapshot.now_ns") != sample_tick) {
        reject("Runtime snapshot hardware or clock differs from the sample");
    }
    const Json& executions = member(runtime_snapshot, "executions");
    if (!executions.is_array()) reject("Runtime executions must be an array");
    std::map<uint64_t, const Json*> executions_by_handle;
    for (const auto& execution : executions) {
        const uint64_t handle = uint64_value(
            member(execution, "handle"), "Runtime execution handle");
        if (!executions_by_handle.emplace(handle, &execution).second) {
            reject("Runtime snapshot contains a duplicate execution handle");
        }
    }

    const Json& callbacks = member(local_sample, "completion_bindings");
    if (!callbacks.is_array()) reject("completion_bindings must be an array");
    std::map<uint64_t, const Json*> callbacks_by_handle;
    for (const auto& callback : callbacks) {
        const uint64_t handle = uint64_value(
            member(callback, "handle"), "completion binding handle");
        if (!callbacks_by_handle.emplace(handle, &callback).second) {
            reject("sample contains a duplicate completion binding handle");
        }
    }

    if (workload_->graph_source_ == nullptr) {
        reject("Workload has no graph source for live COMP metadata");
    }
    Json result = Json::array();
    for (const auto& entry : coarse_active_) {
        const uint64_t handle = entry.first;
        const CoarseActiveRecord& active = entry.second;
        if (!active.plan || !active.lease.active()) {
            reject("adapter has a missing Plan or inactive workspace lease");
        }
        const uint64_t workspace_bytes =
            coarse_die::hbm_workspace_peak_bytes(*active.plan);
        if (!sys_->workspace_admission->owns_lease(active.lease,
                                                    workspace_bytes)) {
            reject("adapter workspace lease is not owned by this rank ledger");
        }
        const auto& lease_key = active.lease.key();
        if (lease_key.rank != sys_->id || lease_key.node_id != active.node_id ||
            lease_key.generation != active.generation) {
            reject("workspace lease key differs from its adapter record");
        }

        const ExecutionDriven::NodeView* node =
            workload_->graph_source_->lookup_ptr(active.node_id);
        std::optional<ExecutionDriven::NodeView> owned_node;
        if (node == nullptr) {
            owned_node = workload_->graph_source_->lookup(active.node_id);
            if (!owned_node.has_value()) {
                reject("active COMP node is absent from the Workload graph");
            }
            node = &*owned_node;
        }
        if (node->global_id != active.node_id || node->rank != sys_->id ||
            node->generation != active.generation ||
            node->node_type != ChakraNodeType::COMP_NODE || node->is_cpu_op ||
            node->is_timer_op ||
            node->compute.op_role != ExecutionDriven::OpRole::Compute ||
            node->compute.die_task_json == nullptr ||
            node->compute.die_spec != nullptr) {
            reject("adapter record no longer identifies the same typed COMP node");
        }

        if (active.plan->fingerprint.empty() ||
            active.plan->hardware_fingerprint != hardware_fingerprint) {
            reject("adapter Plan identity differs from the sampled hardware");
        }
        coarse_die::Owner expected_owner;
        expected_owner.request_id = stable_request_id(node->request_id);
        expected_owner.node_id = node->global_id;
        expected_owner.execution_id = node->generation;
        expected_owner.generation = std::max<uint64_t>(node->generation, 1);

        const auto execution_found = executions_by_handle.find(handle);
        const auto callback_found = callbacks_by_handle.find(handle);
        if (execution_found == executions_by_handle.end() ||
            callback_found == callbacks_by_handle.end()) {
            reject("Runtime execution or live callback is missing an adapter handle");
        }
        const Json& sampled_execution = *execution_found->second;
        const Json& sampled_callback = *callback_found->second;
        const Json& sampled_plan = member(sampled_execution, "plan");
        const Json& sampled_execution_owner = member(sampled_execution, "owner");
        const Json& sampled_callback_owner = member(sampled_callback, "owner");
        const auto owner_matches = [&](const Json& owner) {
            return uint64_value(member(owner, "request_id"), "owner.request_id") ==
                       expected_owner.request_id &&
                   uint64_value(member(owner, "node_id"), "owner.node_id") ==
                       expected_owner.node_id &&
                   uint64_value(member(owner, "execution_id"), "owner.execution_id") ==
                       expected_owner.execution_id &&
                   uint64_value(member(owner, "generation"), "owner.generation") ==
                       expected_owner.generation &&
                   string_value(member(owner, "key"), "owner.key") ==
                       expected_owner.key();
        };
        if (string_value(member(sampled_plan, "fingerprint"),
                         "Runtime Plan fingerprint") != active.plan->fingerprint ||
            string_value(member(sampled_plan, "hardware_fingerprint"),
                         "Runtime Plan hardware fingerprint") != hardware_fingerprint ||
            !owner_matches(sampled_execution_owner) ||
            !owner_matches(sampled_callback_owner)) {
            reject("Runtime execution or callback metadata differs from live COMP identity");
        }

        const bool terminal = bool_value(
            member(sampled_callback, "terminal_at_sample"),
            "callback terminal_at_sample");
        const std::string callback_state = string_value(
            member(sampled_callback, "state_at_sample"), "callback state_at_sample");
        const Json& callback_finish = member(
            sampled_callback, "finish_ns_at_sample");
        const bool runtime_terminal = bool_value(
            member(sampled_execution, "terminal"), "Runtime execution terminal");
        const bool runtime_cancelled = bool_value(
            member(sampled_execution, "cancelled"), "Runtime execution cancelled");
        const Json& runtime_finish = member(sampled_execution, "finish_ns");
        uint64_t finish_ns = 0;
        if (terminal) {
            if (callback_state != "finished" && callback_state != "cancelled") {
                reject("terminal callback state is not finished or cancelled");
            }
            if ((callback_state == "cancelled") != runtime_cancelled ||
                !runtime_terminal || callback_finish.is_null() ||
                runtime_finish.is_null()) {
                reject("terminal callback metadata differs from Runtime execution");
            }
            finish_ns = uint64_value(callback_finish,
                                     "callback finish_ns_at_sample");
            if (finish_ns != uint64_value(runtime_finish,
                                          "Runtime execution finish_ns") ||
                finish_ns > sample_tick) {
                reject("terminal finish time differs or lies after the sample");
            }
        } else if (callback_state == "finished" || callback_state == "cancelled" ||
                   runtime_terminal || runtime_cancelled ||
                   !callback_finish.is_null() || !runtime_finish.is_null()) {
            reject("nonterminal callback metadata differs from Runtime execution");
        }

        result.push_back({
            {"handle", handle},
            {"rank", sys_->id},
            {"node_id", node->global_id},
            {"generation", node->generation},
            {"request_id", node->request_id},
            {"stage", node->stage},
            {"plan_fingerprint", active.plan->fingerprint},
            {"owner", sampled_callback_owner},
            {"workspace_lease", {
                {"rank", lease_key.rank},
                {"node_id", lease_key.node_id},
                {"generation", lease_key.generation},
                {"lease_id", lease_key.lease_id},
                {"bytes", workspace_bytes}}},
            {"terminal_at_sample", terminal},
            {"finish_ns_at_sample", terminal ? Json(finish_ns) : Json(nullptr)},
        });
    }
    return result;
}

DieModelAdapter::IssueTiming&
DieModelAdapter::mutable_cumulative_issue_timing() {
    static IssueTiming timing;
    return timing;
}

const DieModelAdapter::IssueTiming& DieModelAdapter::cumulative_issue_timing() {
    return mutable_cumulative_issue_timing();
}

bool DieModelAdapter::has_active_die_executions() {
    purge_finished_executions();
    return !active_.empty() || !coarse_active_.empty();
}

nlohmann::json DieModelAdapter::window_quality_json() const {
    uint64_t windows = completed_window_count_;
    uint64_t refinements = completed_refinement_count_;
    uint64_t prefix_bound = completed_local_prefix_bound_max_;
    uint64_t logical_events = completed_logical_event_count_;
    uint64_t physical_steps = completed_physical_step_count_;
    uint64_t active_count = 0;
    nlohmann::json active = nlohmann::json::array();
    std::string unsupported = last_unsupported_reason_;
    for (const auto& execution : active_) {
        if (execution == nullptr || execution->finished()) continue;
        const nlohmann::json quality = execution->window_quality_json();
        windows = add_quality_counter(windows,
            quality.value("window_count", uint64_t{0}), "window");
        refinements = add_quality_counter(refinements,
            quality.value("refinement_count", uint64_t{0}), "refinement");
        prefix_bound = std::max(prefix_bound,
            quality.value("local_prefix_bound_max", uint64_t{0}));
        logical_events = add_quality_counter(logical_events,
            quality.value("logical_event_count", uint64_t{0}), "logical event");
        physical_steps = add_quality_counter(physical_steps,
            quality.value("physical_step_count", uint64_t{0}), "physical step");
        ++active_count;
        const nlohmann::json reason = quality.value(
            "unsupported_reason", nlohmann::json(nullptr));
        if (!reason.is_null()) unsupported = reason.get<std::string>();
        active.push_back({{"node_id", execution->node_id()},
                          {"quality", quality}});
    }
    nlohmann::json result{{"approximation_profile",
                           "typed_die_internal_timer_window_v1"},
        {"window_count", windows}, {"refinement_count", refinements},
        {"local_prefix_bound_max", prefix_bound},
        {"logical_event_count", logical_events},
        {"physical_step_count", physical_steps},
        {"bound_kind", "local_prefix_service_ticks"},
        {"fully_exact", windows == 0},
        {"unsupported_reason", unsupported.empty()
             ? nlohmann::json(nullptr) : nlohmann::json(unsupported)},
        {"completed_execution_count", completed_execution_count_},
        {"active_execution_count", active_count},
        {"active_executions", std::move(active)}};
    const auto contract = workload_ == nullptr
        ? nullptr : workload_->pipeline_kernel_contract();
    if (contract != nullptr) {
        result["window_contract"] = {
            {"model_version", contract->model_version()},
            {"timer_whitelist_version", contract->timer_whitelist_version()},
            {"window_tile_limit", contract->window_tile_limit()},
            {"window_local_error_ppm", contract->window_local_error_ppm()},
            {"window_refine_margin_ppm", contract->window_refine_margin_ppm()},
            {"prediction_config_sha256", contract->prediction_config_sha256()}};
    } else {
        result["window_contract"] = nullptr;
    }
    return result;
}

void DieModelAdapter::note_execution_finished(
    const DieExecution* execution) {
    if (execution == nullptr) {
        throw std::runtime_error(
            "DieModelAdapter cannot record a null DIE execution");
    }
    const nlohmann::json quality = execution->window_quality_json();
    completed_window_count_ = add_quality_counter(
        completed_window_count_, quality.value("window_count", uint64_t{0}),
        "window");
    completed_refinement_count_ = add_quality_counter(
        completed_refinement_count_,
        quality.value("refinement_count", uint64_t{0}), "refinement");
    completed_logical_event_count_ = add_quality_counter(
        completed_logical_event_count_,
        quality.value("logical_event_count", uint64_t{0}), "logical event");
    completed_physical_step_count_ = add_quality_counter(
        completed_physical_step_count_,
        quality.value("physical_step_count", uint64_t{0}), "physical step");
    completed_local_prefix_bound_max_ = std::max(
        completed_local_prefix_bound_max_,
        quality.value("local_prefix_bound_max", uint64_t{0}));
    ++completed_execution_count_;
    const nlohmann::json reason = quality.value(
        "unsupported_reason", nlohmann::json(nullptr));
    if (!reason.is_null()) last_unsupported_reason_ = reason.get<std::string>();
}

void DieModelAdapter::purge_finished_executions() {
    // A DieExecution completes inside its own event callback (or the HBM
    // model's transition callback), so it cannot be destroyed synchronously
    // there; retired entries are inert (no pending events, wlhd freed) and
    // leave at the next adapter entry point. The window is bounded by the
    // in-flight node count.
    active_.erase(std::remove_if(active_.begin(), active_.end(),
                                 [](const std::unique_ptr<DieExecution>& e) {
                                     return e->finished();
                                 }),
                  active_.end());
}

void DieModelAdapter::shutdown() {
    // Each ~DieExecution cancels its pending event and drops its pending
    // HBM die job (payload reclaimed by the model); no completion fires.
    active_.clear();
    if (sys_ != nullptr && sys_->coarse_die_runtime != nullptr) {
        for (const auto& entry : coarse_active_) {
            try {
                sys_->coarse_die_runtime->cancel(
                    entry.first, Sys::boostedTick(), /*notify=*/false);
            } catch (...) {
                std::terminate();
            }
        }
    }
    coarse_active_.clear();
}

}  // namespace AstraSim
