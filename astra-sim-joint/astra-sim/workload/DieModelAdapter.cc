/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieModelAdapter.hh"

#include <algorithm>
#include <chrono>
#include <limits>
#include <stdexcept>
#include <utility>

#include "DieModel.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/DieExecution.hh"
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

}  // namespace

DieModelAdapter::DieModelAdapter(Workload* workload)
    : workload_(workload), sys_(workload != nullptr ? workload->sys : nullptr) {
    if (workload_ == nullptr || sys_ == nullptr ||
        sys_->die_runtime == nullptr) {
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
    return !active_.empty();
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
}

}  // namespace AstraSim
