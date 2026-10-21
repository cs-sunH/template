/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DieExecution.hh -- Sys-clock-driven driver of one DieModel::Execution (die
backend, stage B; design doc §6.3/§6.4, contract §4 action protocol).

One DieExecution per outer COMP node under the die compute-cost-model
backend. It owns:

  - the immutable plan (shared_ptr, produced through the Sys plan cache by
    the DieModelAdapter) and the mutable DieModel::Execution state machine;
  - the drive loop: Timer actions become cancellable Sys events,
    MemoryRequest actions are handed to the rank's LocalHbmBandwidthModel
    (contract §7.4) or, when hbm-bandwidth-contention=0, self-drained at the
    full memory rate with an explicit "未模拟双方争用" trace note.  Stage E2:
    a pure stretch of consecutive Timer actions is coalesced into ONE Sys
    event while the HBM model reports no active job (design doc §8.5's
    unchanged-external-competition guard); memory-request boundaries are
    never crossed by a coalesced stretch;
  - the one-shot local_mem_latency TIMER inserted before the node's FIRST
    memory request (contract §7.4: no latency phase inside the HBM job, at
    most one charge per outer node);
  - the completion: exactly one Workload::finish_generic_node, one GPU
    occupancy interval (tics_gpu_ops += end - start), and the utilization
    fill of the node's statistics state (contract §7.6 caliber).

Red lines (design doc §6.3): an internal HBM job NEVER masquerades as an
outer COMP completion (no early dependency release / GPU-slot release); the
outer node completes exactly once, when the machine reports Done.
*/

#ifndef __DIE_EXECUTION_HH__
#define __DIE_EXECUTION_HH__

#include <cstdint>
#include <memory>

#include "DieTypes.hh"
#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"
#include "astra-sim/workload/execution_driven/GraphSource.hh"

namespace AstraSim {

class Sys;
class Workload;
class WorkloadLayerHandlerData;
class DieModelAdapter;

class DieExecution final : public Callable {
  public:
    DieExecution(Sys* sys,
                 Workload* workload,
                 DieModelAdapter* adapter,
                 uint64_t node_id,
                 WorkloadLayerHandlerData* wlhd,
                 std::shared_ptr<const DieModel::DiePlan> plan);
    ~DieExecution() override;

    // Non-copyable mutable run state (one execution per outer node).
    DieExecution(const DieExecution&) = delete;
    DieExecution& operator=(const DieExecution&) = delete;

    /// Kick the drive loop at issue time. Must be called exactly once,
    /// right after construction.
    void start();

    /// Callable: Timer/Memory event callbacks (own cancellable events and
    /// the LocalHbmBandwidthModel DIE_COMPUTE sink).
    void call(EventType type, CallData* data) override;

    bool finished() const { return finished_; }
    uint64_t node_id() const { return node_id_; }
    /// Internal machine clock (ns) accumulated so far.
    uint64_t machine_elapsed_ns() const { return machine_.elapsed_ns(); }

  private:
    // Payload of every event this execution schedules or hands to the HBM
    // model; owned by the recipient (the model / the cancellation callback /
    // this execution once the callback fires).
    struct DieEventData final : public CallData {
        enum class Kind { Timer, Memory };
        DieEventData(Kind kind_in, uint64_t stage_id_in, uint64_t generation_in)
            : kind(kind_in), stage_id(stage_id_in), generation(generation_in) {}
        Kind kind;
        uint64_t stage_id;   // Memory only
        uint64_t generation;  // stale-callback guard
    };
    static void destroy_event_data(CallData* data);

    /// The action loop; returns with a pending callback outstanding or the
    /// execution finished.
    void drive();
    /// Terminal path: exactly-once node completion through the Workload's
    /// generic exit (never from an internal memory callback).
    void complete();
    void register_timer_or_self_driven_memory(uint64_t ns, DieEventData* data);
    void fill_utilizations(uint64_t elapsed_ns);

    Sys* sys_;
    Workload* workload_;
    DieModelAdapter* adapter_;
    uint64_t node_id_;
    WorkloadLayerHandlerData* wlhd_;
    std::shared_ptr<const DieModel::DiePlan> plan_;
    DieModel::Execution machine_;

    Tick issue_tick_;
    Tick last_synced_tick_;
    SystemEventHandle pending_event_;
    uint64_t event_generation_ = 0;
    // Stage E2 timer coalescing: machine-clock nanoseconds already consumed
    // synchronously by self-advancing a coalesced timer stretch (see drive).
    // The merged Sys event fires exactly at last_synced_tick_ + this value,
    // so the next call() advances the machine by the (zero) remainder only.
    uint64_t self_advanced_ns_ = 0;
    bool latency_charged_ = false;
    // True while the pending event is the one-shot local_mem_latency TIMER
    // (external dead time: call() re-anchors the sync point without
    // advancing the machine clock).
    bool latency_timer_pending_ = false;
    bool finished_ = false;
};

}  // namespace AstraSim

#endif /* __DIE_EXECUTION_HH__ */
