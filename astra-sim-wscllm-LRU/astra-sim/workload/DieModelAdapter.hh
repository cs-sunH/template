/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DieModelAdapter.hh -- per-Workload (per-rank) bridge from a die-backend COMP
NodeView to a DieExecution (die backend, stage B; design doc §8.1/§8.3).

Responsibilities (one adapter per Workload, owned by the Workload):

  - issue_die(NodeView): compile-or-reuse the immutable DiePlan through the
    Sys-owned DieModelRuntime (Hardware + LRU PlanCache keyed with
    DieModel::plan_cache_key; design doc §8.5 "执行句柄不入缓存"), create the
    DieExecution with its completion handler data, and start it;
  - track the set of live (not yet completed) DieExecutions so the Workload's
    is_finished / drain checks include them (design doc §6.3 lifecycle red
    line: a service must not exit while an outer node is still executing);
  - retire finished executions lazily (a DieExecution completes inside its
    own event callback, so it may not be destroyed synchronously there) and
    tear every execution down on Workload destruction (no leak, no double
    completion, no dangling HBM jobs).
*/

#ifndef __DIE_MODEL_ADAPTER_HH__
#define __DIE_MODEL_ADAPTER_HH__

#include <cstdint>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include "astra-sim/workload/execution_driven/GraphSource.hh"
#include "astra-sim/workload/WorkspaceAdmission.hh"

namespace coarse_die {
struct Plan;
}

namespace AstraSim {

class Sys;
class Workload;
class DieExecution;

class DieModelAdapter {
  public:
    struct PreparedCoarsePlan {
        std::shared_ptr<const coarse_die::Plan> plan;
        uint64_t workspace_bytes = 0;
        uint64_t node_id = 0;
        uint64_t generation = 0;
    };

    explicit DieModelAdapter(Workload* workload);
    ~DieModelAdapter();

    // Dispatch one die-backend real COMP node (die_spec != nullptr; the
    // Workload layer guarantees the dispatch order of contract §7.3 before
    // calling). Throws std::runtime_error when the node carries no
    // die_op_spec (fail-closed, never a silent roofline fallback).
    void issue_die(const ExecutionDriven::NodeView& node);

    /// Compile/cache before graph take/COMP occupy. The returned immutable
    /// Plan is passed unchanged through admission and Runtime submission.
    PreparedCoarsePlan prepare_coarse_plan(
        const ExecutionDriven::NodeView& node);
    WorkspaceAdmission::AcquireResult try_acquire_workspace(
        const PreparedCoarsePlan& prepared);
    void issue_prepared_coarse_plan(
        const ExecutionDriven::NodeView& node,
        PreparedCoarsePlan prepared,
        WorkspaceAdmission::Lease lease);
    /// Explicit live cancellation terminalizes the graph node as Skipped;
    /// shutdown uses notify=false and does not issue a retry pass.
    void cancel_coarse_node(uint64_t node_id, uint64_t now_ns);

    // Whether any outer node still has a live DieExecution. Purges retired
    // entries first (they linger only until the next adapter entry point).
    bool has_active_die_executions();

    // DeadEnd-census diagnostic only. Read-only walk over live legacy
    // DieExecution objects; coarse active Plans are represented by the
    // Runtime's typed counters and workspace ledger.
    void dump_census_to_stderr(const char* prefix) const;

    // Workload teardown: cancel every pending event, drop the pending HBM
    // die jobs, and destroy all executions. No completion callback fires.
    void shutdown();

  private:
    friend class DieExecution;
    // DieExecution completion hook: marks the execution retired (the object
    // itself stays alive until the next purge -- it cannot destroy itself
    // from inside its own completion callback).
    void note_execution_finished(const DieExecution* execution);
    void purge_finished_executions();

    Workload* workload_;
    Sys* sys_;
    std::vector<std::unique_ptr<DieExecution>> active_;
    struct CoarseActiveRecord {
        uint64_t node_id = 0;
        uint64_t generation = 0;
        std::shared_ptr<const coarse_die::Plan> plan;
        WorkspaceAdmission::Lease lease;
    };
    std::map<uint64_t, CoarseActiveRecord> coarse_active_;
};

}  // namespace AstraSim

#endif /* __DIE_MODEL_ADAPTER_HH__ */
