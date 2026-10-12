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
#include "astra-sim/workload/Workload.hh"

namespace AstraSim {

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
    return !active_.empty();
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
}

}  // namespace AstraSim
