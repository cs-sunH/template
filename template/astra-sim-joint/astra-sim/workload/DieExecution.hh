/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

/*
DieExecution.hh -- Sys-clock-driven driver of one DieModel::Execution (die
backend, joint rollout stage 1; design doc §6.3/§6.4, contract §4 action
protocol; ported from the face reference implementation, adapted to this
repository's Workload::finish_general_node(node_id, event) completion exit).

One DieExecution per outer COMP node under the die compute-cost-model
backend. It owns:

  - the immutable plan (shared_ptr, produced through the Sys plan cache by
    the DieModelAdapter) and the mutable DieModel::Execution state machine;
  - the drive loop: Timer actions become cancellable Sys events,
    MemoryRequest actions are handed to the rank's LocalHbmBandwidthModel
    (contract §7.4) or, when hbm-bandwidth-contention=0, self-drained at the
    full memory rate with an explicit "未模拟双方争用" trace note.  A pure
    stretch of consecutive Timer actions is coalesced into ONE Sys event
    while the HBM model reports no active job (design doc §8.5's
    unchanged-external-competition guard); memory-request boundaries are
    never crossed by a coalesced stretch;
  - the one-shot local_mem_latency TIMER inserted before the node's FIRST
    memory request (contract §7.4: no latency phase inside the HBM job, at
    most one charge per outer node);
  - the completion: exactly one Workload::finish_general_node, one GPU
    occupancy interval (tics_gpu_ops += end - start), and the utilization
    fill of the node's statistics state (contract §7.6 caliber).

Red lines (design doc §6.3): an internal HBM job NEVER masquerades as an
outer COMP completion (no early dependency release / GPU-slot release); the
outer node completes exactly once, when the machine reports Done.
*/

#ifndef __DIE_EXECUTION_HH__
#define __DIE_EXECUTION_HH__

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

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

    // Diagnostic-only dispatch label (v7 event attribution); see Callable.
    [[nodiscard]] const char* profile_dispatch_label() const noexcept override {
        return "die_timer_memory";
    }


    bool finished() const { return finished_; }
    uint64_t node_id() const { return node_id_; }
    /// Internal machine clock (ns) accumulated so far.
    uint64_t machine_elapsed_ns() const { return machine_.elapsed_ns(); }
    nlohmann::json quality_json() const;

  private:
    // Payload of every event this execution schedules or hands to the HBM
    // model; owned by the recipient (the model / the cancellation callback /
    // this execution once the callback fires).
    struct DieEventData final : public CallData {
        enum class Kind { Timer, Memory };
        enum class TimerPurpose { InternalService, MemoryLatency, SelfDrivenMemory };
        DieEventData(Kind kind_in, uint64_t stage_id_in, uint64_t generation_in,
                     TimerPurpose timer_purpose_in = TimerPurpose::InternalService)
            : kind(kind_in), stage_id(stage_id_in), generation(generation_in),
              timer_purpose(timer_purpose_in) {}
        Kind kind;
        uint64_t stage_id;   // Memory only
        uint64_t generation;  // stale-callback guard
        TimerPurpose timer_purpose;
        // Owning execution (pooling stage A): set for every Timer payload
        // handed out by acquire_event_data and never cleared through
        // freelist reuse, so the static destroy_event_data can route
        // cancellations back to the owner.  Memory payloads keep it null
        // and stay plain-deleted (stage B is out of scope).
        DieExecution* owner = nullptr;
    };
    struct InternalTimerArm {
        DieModel::InternalActionToken action_token;
        uint64_t template_position = 0;
        uint64_t stage_generation = 0;
        uint64_t original_service_ns = 0;
        uint64_t external_delay_ns = 0;
        uint64_t logical_action_count = 1;
        Tick external_start_tick = 0;
        uint64_t service_already_applied = 0;
        uint64_t rounding_residual = 0;
        uint64_t logical_start_service_tick = 0;
        // Scalar since batch 3 (was a 1-element vector); write-only today.
        // A future reader must first restore/justify the vector semantics.
        uint64_t resource_version = 0;
        bool partial_api_eligible = false;
        bool due_at_mutation = false;
        bool active = false;
    };
    static void destroy_event_data(CallData* data);
    /// Timer-payload freelist (pooling stage A).  The simulation is
    /// single-threaded, so no lock is needed.  DieEventData's whole state
    /// is its four payload fields plus owner (the CallData base has no
    /// members and no virtual functions), so whole-struct reassignment on
    /// acquire leaves no residue.  Memory payloads are NOT pooled (stage B
    /// is out of scope).
    DieEventData* acquire_event_data(
        DieEventData::Kind kind, uint64_t stage_id, uint64_t generation,
        DieEventData::TimerPurpose timer_purpose =
            DieEventData::TimerPurpose::InternalService);
    void release_event_data(DieEventData* data);

    /// The action loop; returns with a pending callback outstanding or the
    /// execution finished.  `now` is the caller's Sys clock reading (P1: the
    /// clock is constant within one callback, so re-reading it here would
    /// return the identical value).
    void drive(const Tick now);
    void on_hbm_resource_mutation(bool before_change, Tick tick,
                                  uint64_t version);
    void settle_internal_timer_to(Tick tick);
    void register_internal_timer(
        const Tick now, uint64_t original_service_ns,
        uint64_t external_delay_ns, uint64_t logical_action_count = 1);
    void register_external_timer(uint64_t delay_ns,
                                 DieEventData::TimerPurpose purpose,
                                 uint64_t stage_id = 0);
    /// Terminal path: exactly-once node completion through the Workload's
    /// generic exit (never from an internal memory callback).
    void complete();
    void fill_utilizations(uint64_t elapsed_ns);

    Sys* sys_;
    Workload* workload_;
    DieModelAdapter* adapter_;
    uint64_t node_id_;
    WorkloadLayerHandlerData* wlhd_;
    std::shared_ptr<const DieModel::DiePlan> plan_;
    DieModel::Execution machine_;
    uint64_t logical_timer_event_count_ = 0;
    uint64_t physical_timer_callback_count_ = 0;
    bool quality_recorded_ = false;

    Tick issue_tick_;
    Tick last_synced_tick_;
    SystemEventHandle pending_event_;
    uint64_t event_generation_ = 0;
    InternalTimerArm timer_arm_;
    /// Timer payload freelist (pooling stage A); see acquire_event_data.
    std::vector<DieEventData*> timer_event_data_freelist_;
    uint64_t hbm_observer_id_ = 0;
    bool memory_latency_pending_ = false;
    bool latency_charged_ = false;
    bool finished_ = false;
};

}  // namespace AstraSim

#endif /* __DIE_EXECUTION_HH__ */
