/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieModelAdapter.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
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
}

DieModelAdapter::DieModelAdapter(Workload* workload)
    : workload_(workload), sys_(workload != nullptr ? workload->sys : nullptr) {
    if (workload_ == nullptr || sys_ == nullptr ||
        (sys_->die_runtime == nullptr && sys_->coarse_die_runtime == nullptr)) {
        throw std::runtime_error(
            "DieModelAdapter requires a Workload whose Sys loaded a die "
            "runtime (compute-cost-model=die with a valid profile descriptor)");
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
    const auto pipeline_contract = workload_->pipeline_kernel_contract();
    if (sys_->coarse_die_runtime != nullptr) {
        if (workload_->requires_pipeline_kernel_contract() &&
            pipeline_contract == nullptr) {
            throw std::runtime_error(
                "DieModelAdapter refuses an online DIE node before the first "
                "valid GraphBatch freezes its PipelineKernelContract");
        }
        if (pipeline_contract != nullptr) {
            throw std::runtime_error(
                "coarse_die_v2 cannot execute a JOINT PipelineKernelContract: "
                "there is no lossless conversion from the frozen pipeline "
                "timing contract to a coarse_die_v2 plan");
        }
        if (node.compute.die_task_json == nullptr) {
            throw std::runtime_error(
                "coarse_die_v2 requires the preserved typed die task JSON on "
                "real COMP node " + std::to_string(node.global_id));
        }
        const auto plan =
            sys_->coarse_die_runtime->compile(*node.compute.die_task_json);
        coarse_die::Owner owner;
        owner.request_id = stable_request_id(node.request_id);
        owner.node_id = node.global_id;
        owner.execution_id = node.generation;
        owner.generation = std::max<uint64_t>(node.generation, 1);
        const uint64_t node_id = node.global_id;
        const uint64_t handle = sys_->coarse_die_runtime->submit_plan(
            plan, owner,
            [this, node_id](const uint64_t done_handle,
                                  const nlohmann::json& progress) {
                coarse_active_.erase(done_handle);
                const auto state = progress.value("state", std::string());
                if (state == "cancelled") {
                    // Workload teardown removes callbacks before canceling.
                    // A live cancellation is surfaced through the existing
                    // graph cancellation path and cannot be turned into a
                    // successful COMP completion here.
                    return;
                }
                const uint64_t elapsed_ns =
                    progress.value("elapsed_ns", uint64_t{0});
                const double elapsed_s =
                    static_cast<double>(std::max<uint64_t>(elapsed_ns, 1)) /
                    1e9;
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
                const double hbm_bytes =
                    progress.at("hbm_served_bytes")[0][0].get<double>() +
                    progress.at("hbm_served_bytes")[0][1].get<double>() +
                    progress.at("hbm_served_bytes")[1][0].get<double>() +
                    progress.at("hbm_served_bytes")[1][1].get<double>();
                if (!(int4_peak > 0.0) || !(vec_peak > 0.0) ||
                    !(hbm_peak > 0.0) || !std::isfinite(int4_ops) ||
                    !std::isfinite(vec_ops) || !std::isfinite(hbm_bytes)) {
                    throw std::runtime_error(
                        "coarse_die_v2 completion is missing finite typed "
                        "plan work or hardware peaks");
                }
                workload_->record_die_utilizations(
                    node_id, int4_ops / (int4_peak * elapsed_s),
                    hbm_bytes / (hbm_peak * elapsed_s),
                    vec_ops / (vec_peak * elapsed_s));
                workload_->finish_general_node(node_id, EventType::General);
            });
        coarse_active_.emplace(handle, node_id);
        return;
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
    if (workload_->requires_pipeline_kernel_contract() &&
        pipeline_contract == nullptr) {
        throw std::runtime_error(
            "DieModelAdapter refuses an online DIE node before the first "
            "valid GraphBatch freezes its PipelineKernelContract");
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
    std::shared_ptr<const DieModel::DiePlan> plan =
        nullptr;
    {
        ScopedNanosecondTimer cache_get_timer(timing.cache_get_ns);
        plan = runtime.plan_cache->get(key);
    }
    if (plan == nullptr) {
        ++timing.cache_misses;
        // compile() is a pure function (DieModelError on any mapping
        // violation); the immutable result enters the LRU cache, the
        // mutable execution handle never does (design doc §8.5).  Stage E2:
        // the cache is process-level shared across every Sys of the same
        // resolved hardware (see Sys.hh), so a rank reuses plans other
        // ranks compiled for the identical (spec, hardware) key.
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
        const uint64_t now = Sys::boostedTick();
        while (!coarse_active_.empty()) {
            const uint64_t handle = coarse_active_.begin()->first;
            sys_->coarse_die_runtime->cancel(handle, now, false);
            coarse_active_.erase(handle);
        }
    }
}

}  // namespace AstraSim
