/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieModelAdapter.hh"

#include <algorithm>
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
        sys_->die_runtime == nullptr && sys_->coarse_die_runtime == nullptr) {
        throw std::runtime_error(
            "DieModelAdapter requires a Workload whose Sys loaded the die "
            "runtime (compute-cost-model=die with a valid die-model-config)");
    }
}

DieModelAdapter::~DieModelAdapter() {
    shutdown();
}

void DieModelAdapter::issue_die(const ExecutionDriven::NodeView& node) {
    purge_finished_executions();
    if (sys_->coarse_die_runtime != nullptr) {
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
                workload_->finish_generic_node(node_id);
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

    Sys::DieModelRuntime& runtime = *sys_->die_runtime;
    const std::string key =
        DieModel::plan_cache_key(*node.compute.die_spec, runtime.hardware);
    std::shared_ptr<const DieModel::DiePlan> plan =
        runtime.plan_cache->get(key);
    if (plan == nullptr) {
        // compile() is a pure function (DieModelError on any mapping
        // violation); the immutable result enters the LRU cache, the
        // mutable execution handle never does (design doc §8.5).  Stage E2:
        // the cache is process-level shared across every Sys of the same
        // resolved hardware (see Sys.hh), so a rank reuses plans other
        // ranks compiled for the identical (spec, hardware) key.
        plan = DieModel::compile(*node.compute.die_spec, runtime.hardware);
        runtime.plan_cache->put(key, plan);
    }

    WorkloadLayerHandlerData* wlhd = new WorkloadLayerHandlerData;
    wlhd->node_id = node.global_id;
    auto execution = std::make_unique<DieExecution>(
        sys_, workload_, this, node.global_id, wlhd, plan);
    DieExecution* const raw = execution.get();
    active_.push_back(std::move(execution));
    // start() may complete the node synchronously (a zero-work plan reports
    // Done on the first action); the finished execution stays in active_
    // until the next purge.
    raw->start();
}

bool DieModelAdapter::has_active_die_executions() {
    purge_finished_executions();
    return !active_.empty() || !coarse_active_.empty();
}

void DieModelAdapter::dump_census_to_stderr(const char* prefix) const {
    // DeadEnd-census diagnostic only (2026-10-11, ported from WL): read-only
    // walk over the live-execution list (retired-but-unpurged entries
    // included; their census lines carry finished=1). Deliberately does NOT
    // purge -- the census must not mutate any state.
    for (const auto& execution : active_) {
        execution->dump_census_to_stderr(prefix);
    }
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
