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
#include <string>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {

class Sys;
class Workload;
class DieExecution;

class DieModelAdapter {
  public:
    // Cumulative host-side timings for DIE dispatch. Child phases are nested
    // inside total_issue_ns and must not be added to it. The simulator drives
    // Workloads on one event thread, so these process-wide totals need no lock.
    struct IssueTiming {
        uint64_t calls = 0;
        uint64_t total_issue_ns = 0;
        uint64_t plan_key_calls = 0;
        uint64_t plan_key_ns = 0;
        uint64_t cache_get_ns = 0;
        uint64_t cache_get_calls = 0;
        uint64_t cache_hits = 0;
        uint64_t cache_misses = 0;
        uint64_t compile_ns = 0;
        uint64_t compile_calls = 0;
        uint64_t cache_put_ns = 0;
        uint64_t cache_put_calls = 0;
        uint64_t setup_ns = 0;
        uint64_t setup_calls = 0;
        uint64_t start_ns = 0;
        uint64_t start_calls = 0;
    };

    explicit DieModelAdapter(Workload* workload);
    ~DieModelAdapter();

    /// Read cumulative adapter dispatch diagnostics across all Workloads.
    static const IssueTiming& cumulative_issue_timing();

    // Dispatch one die-backend real COMP node (legacy die_spec or preserved
    // coarse typed task JSON; the Workload layer guarantees contract §7.3
    // ordering before calling). A node without the selected backend's task
    // payload fails closed; it never silently falls back to roofline.
    void issue_die(const ExecutionDriven::NodeView& node);

    // Whether any outer node still has a live DieExecution. Purges retired
    // entries first (they linger only until the next adapter entry point).
    bool has_active_die_executions();

    /// Runtime quality disclosure from actual DIE timer arms. Completed
    /// executions are folded into bounded aggregate counters; only active
    /// executions are inspected on each snapshot.
    nlohmann::json window_quality_json() const;

    // Workload teardown: cancel every pending event, drop the pending HBM
    // die jobs, and destroy all executions. No completion callback fires.
    void shutdown();

  private:
    static IssueTiming& mutable_cumulative_issue_timing();

    friend class DieExecution;
    // DieExecution completion hook: marks the execution retired (the object
    // itself stays alive until the next purge -- it cannot destroy itself
    // from inside its own completion callback).
    void note_execution_finished(const DieExecution* execution);
    void purge_finished_executions();

    Workload* workload_;
    Sys* sys_;
    std::vector<std::unique_ptr<DieExecution>> active_;
    std::map<uint64_t, uint64_t> coarse_active_;
    uint64_t completed_execution_count_ = 0;
    uint64_t completed_window_count_ = 0;
    uint64_t completed_refinement_count_ = 0;
    uint64_t completed_local_prefix_bound_max_ = 0;
    uint64_t completed_logical_event_count_ = 0;
    uint64_t completed_physical_step_count_ = 0;
    std::string last_unsupported_reason_;
};

}  // namespace AstraSim

#endif /* __DIE_MODEL_ADAPTER_HH__ */
