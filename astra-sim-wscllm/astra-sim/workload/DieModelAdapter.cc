/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieModelAdapter.hh"

#include <algorithm>
#include <cmath>
#include <exception>
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
    purge_finished_executions();
    if (sys_->coarse_die_runtime != nullptr) {
        throw std::runtime_error(
            "coarse_die_v2 COMP must use prepare_coarse_plan, acquire a "
            "workspace lease, and submit that same Plan before issue");
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

DieModelAdapter::PreparedCoarsePlan DieModelAdapter::prepare_coarse_plan(
    const ExecutionDriven::NodeView& node) {
    purge_finished_executions();
    if (sys_->coarse_die_runtime == nullptr ||
        sys_->workspace_admission == nullptr) {
        throw std::runtime_error(
            "coarse workspace preparation requires a coarse_die_v2 rank");
    }
    if (node.node_type != ChakraProtoMsg::COMP_NODE || node.is_timer_op ||
        node.is_cpu_op ||
        node.compute.op_role == ExecutionDriven::OpRole::ControlMarker ||
        node.compute.die_task_json == nullptr) {
        throw std::runtime_error(
            "workspace preparation requires a real GPU coarse COMP node " +
            std::to_string(node.global_id));
    }
    auto plan = sys_->coarse_die_runtime->compile(*node.compute.die_task_json);
    if (!plan) {
        throw std::runtime_error("coarse_die_v2 compiler returned a null Plan");
    }
    const uint64_t workspace_bytes =
        coarse_die::hbm_workspace_peak_bytes(*plan);
    return PreparedCoarsePlan{std::move(plan), workspace_bytes,
                              node.global_id, node.generation};
}

WorkspaceAdmission::AcquireResult DieModelAdapter::try_acquire_workspace(
    const PreparedCoarsePlan& prepared) {
    if (sys_->workspace_admission == nullptr || !prepared.plan) {
        throw std::runtime_error(
            "workspace acquisition requires a prepared coarse Plan");
    }
    return sys_->workspace_admission->try_acquire(
        prepared.node_id, prepared.generation, prepared.workspace_bytes);
}

void DieModelAdapter::issue_prepared_coarse_plan(
    const ExecutionDriven::NodeView& node,
    PreparedCoarsePlan prepared,
    WorkspaceAdmission::Lease lease) {
    purge_finished_executions();
    if (sys_->coarse_die_runtime == nullptr ||
        sys_->workspace_admission == nullptr || !prepared.plan ||
        !lease.active()) {
        throw std::runtime_error(
            "coarse issue requires a live prepared Plan and workspace lease");
    }
    const auto& key = lease.key();
    if (prepared.node_id != node.global_id ||
        prepared.generation != node.generation || key.rank != sys_->id ||
        key.node_id != node.global_id || key.generation != node.generation) {
        throw std::runtime_error(
            "prepared Plan/workspace lease identity does not match COMP node");
    }

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
            // Workspace capacity returns before Workload terminalization; the
            // observer's deferred issue pass therefore sees the released
            // lease in this same tick.
            active.lease.release();

            const auto state = progress.value("state", std::string());
            const bool cancelled = state == "cancelled";
            if (!cancelled && state != "finished") {
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
            const double hbm_bytes =
                progress.at("hbm_served_bytes")[0][0].get<double>() +
                progress.at("hbm_served_bytes")[0][1].get<double>() +
                progress.at("hbm_served_bytes")[1][0].get<double>() +
                progress.at("hbm_served_bytes")[1][1].get<double>();
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
            const bool any_served_work = int4_ops != 0.0 || vec_ops != 0.0 ||
                                         hbm_bytes != 0.0;
            if (elapsed_ns == 0 && any_served_work) {
                throw std::runtime_error(
                    "coarse_die_v2 completion reports served work with a "
                    "zero elapsed-time denominator");
            }
            const double compute_utilization =
                elapsed_ns == 0 ? 0.0 : int4_ops / (int4_peak * elapsed_s);
            const double memory_utilization =
                elapsed_ns == 0 ? 0.0 : hbm_bytes / (hbm_peak * elapsed_s);
            const double vector_utilization =
                elapsed_ns == 0 ? 0.0 : vec_ops / (vec_peak * elapsed_s);
            if (!std::isfinite(compute_utilization) ||
                !std::isfinite(memory_utilization) ||
                !std::isfinite(vector_utilization)) {
                throw std::runtime_error(
                    "coarse_die_v2 completion utilization is non-finite");
            }
            workload_->record_die_utilizations(
                active.node_id, compute_utilization, memory_utilization,
                vector_utilization);
            if (cancelled) {
                // Cancellation still reports exactly the typed work served
                // before cancel. The live graph terminal remains Skipped,
                // never a successful COMP completion.
                workload_->finish_cancelled_coarse_node(active.node_id);
                return;
            }
            workload_->finish_generic_node(active.node_id);
        });
    try {
        // Allocate the map node before transferring lease ownership. If the
        // allocation throws, the local token remains held until the accepted
        // Runtime handle has been cancelled below.
        const auto inserted = coarse_active_.try_emplace(handle);
        if (!inserted.second) {
            throw std::runtime_error(
                "coarse Runtime reused an active execution handle");
        }
        inserted.first->second.node_id = node_id;
        inserted.first->second.generation = node.generation;
        inserted.first->second.plan = prepared.plan;
        inserted.first->second.lease = std::move(lease);
    } catch (...) {
        // Runtime accepted the Plan, but the adapter could not establish its
        // unique completion/lease owner. Cancel and retire before the local
        // RAII token is allowed to release capacity; this is fatal cleanup,
        // never a dispatch retry.
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

void DieModelAdapter::cancel_coarse_node(const uint64_t node_id,
                                         const uint64_t now_ns) {
    for (const auto& entry : coarse_active_) {
        if (entry.second.node_id == node_id) {
            sys_->coarse_die_runtime->cancel(entry.first, now_ns,
                                             /*notify=*/true);
            return;
        }
    }
    throw std::runtime_error(
        "cannot cancel unknown coarse COMP node " + std::to_string(node_id));
}

bool DieModelAdapter::has_active_die_executions() {
    purge_finished_executions();
    return !active_.empty() || !coarse_active_.empty();
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
            auto found = coarse_active_.begin();
            const uint64_t handle = found->first;
            CoarseActiveRecord active = std::move(found->second);
            coarse_active_.erase(found);
            sys_->coarse_die_runtime->cancel(handle, now, false);
            active.lease.release();
        }
    }
}

}  // namespace AstraSim
