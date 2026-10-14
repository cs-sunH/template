/******************************************************************************
This source code is licensed under the MIT license found in the root directory
of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <map>
#include <utility>
#include <vector>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class Workload;
class WorkloadLayerHandlerData;

/**
 * Per-NPU fluid HBM model with N-way strict equal-split bandwidth sharing.
 *
 * Every HBM user on a rank competes in one fluid model. Job kinds:
 *   - COMPUTE     : inference COMP node (bytes = tensor_size; ops progress
 *                   at peak_perf in parallel, preserving Roofline's
 *                   max(compute time, memory time) semantics),
 *   - RESTORE     : KV-restore DMA write (multiple concurrent restores are
 *                   a supported state and share the pool),
 *   - COMM_READ   : NoC p2p send-side endpoint HBM read (bytes = comm size),
 *   - COMM_WRITE  : NoC p2p recv-side endpoint HBM write (bytes = comm size),
 *   - POOL_READ   : off-chip pool endpoint HBM read (bytes = tensor_size),
 *   - POOL_WRITE  : off-chip pool endpoint HBM write (bytes = tensor_size),
 *   - DIE_COMPUTE : die-backend internal memory stage (joint rollout stage
 *                   1, contract §7.4): one job per MemoryRequest of a
 *                   DieExecution. The job drains SERVICE UNITS
 *                   (actual_bytes/eta) at the equal-share rate -- the access
 *                   inefficiency is spent as service time, never by
 *                   inflating actual bytes -- and has NO latency phase and
 *                   NO FLOP leg (the DieExecution charges local_mem_latency
 *                   itself, once per outer node, and drains compute through
 *                   the plan's PE/VEC stages). Completion fires the sink
 *                   Callable directly (the sink is the DieExecution, not
 *                   the Workload), so an internal die job never masquerades
 *                   as an outer COMP completion.
 *
 * All outstanding jobs strictly share the rank's single configured HBM
 * bandwidth: each active byte streamer receives full_rate/N. When any job
 * consumes its remaining HBM bytes it completes immediately and the
 * survivors are re-split (full_rate/N') at the same event -- no reservation,
 * no work conservation loss. Each legacy job pays sys->local_mem_latency
 * once at start (the pre-existing convention) -- except DIE_COMPUTE jobs,
 * which never enter the latency phase (see above).
 *
 * Reads and writes share one bus and one aggregate bandwidth scalar
 * (system key "local-mem-bw"); there is no peak/sustained distinction.
 *
 * v7 execution-engine restructure (joint改造方案v7 §6 E2, 2026-10-12).
 * The service arithmetic (equal split, substep boundaries, per-job float
 * operations in issue order, side-band accumulation order, residue clamp
 * set, observer pass boundaries) is preserved operation-for-operation; what
 * changed is the STATE INDEXING around it:
 *   - mutation observers live in a stable deque with tombstones and
 *     pass-serial bookkeeping instead of a per-notify vector copy (the old
 *     copy implemented "snapshot at pass start" by paying O(observers) per
 *     notification; the deque reproduces the same snapshot semantics --
 *     entries appended after a pass started are not visited by it, entries
 *     removed during a pass are still visited by that pass and skipped by
 *     every later one -- at O(1) dispatch overhead);
 *   - jobs occupy stable slots with dead-slot reclamation, and three
 *     index-sorted phase lists (latency-waiting / bandwidth-streaming /
 *     compute) plus a completion-candidate set and a live drain counter
 *     replace the per-event full-vector scans in advance_to, the completion
 *     detach, the join gate, and schedule_next_transition. Every per-job
 *     floating-point operation still executes, in the same jobs order, with
 *     the same operands; only the iteration that FINDS the affected jobs is
 *     indexed.
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    using MutationObserverId = uint64_t;
    // Called at a resource mutation boundary. before_change=true observes
    // the old version after the old interval has been integrated; the false
    // call observes the newly published version after membership/rates have
    // changed and before any job completion callback is invoked.
    using ResourceMutationObserver =
        std::function<void(bool before_change, Tick tick, uint64_t version)>;
    enum class JobKind {
        COMPUTE = 0,
        RESTORE,
        COMM_READ,
        COMM_WRITE,
        POOL_READ,
        POOL_WRITE,
        DIE_COMPUTE,
        CONTROL_MARKER,
        KIND_COUNT,
    };
    static constexpr size_t kJobKindCount =
        static_cast<size_t>(JobKind::KIND_COUNT);

    LocalHbmBandwidthModel(Sys* sys, Workload* workload);
    ~LocalHbmBandwidthModel() override;

    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    /// Keep the legacy micro-marker completion path while accounting its
    /// one-byte HBM endpoint through coarse_die_v2's shared Runtime.
    void issue_control_marker(WorkloadLayerHandlerData* wlhd);
    void issue_restore(uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_comm_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_comm_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    /// Issue a network-correlated COMM_READ/COMM_WRITE transfer whose
    /// completion is owned by an independent executor rather than a physical
    /// Workload node. The callback runs exactly once after the same legacy
    /// latency + shared-bandwidth path completes. The callback object owns
    /// any typed cookie it captures; destruction before completion releases
    /// that capture without invoking it.
    void issue_transfer(JobKind kind,
                        uint64_t bytes,
                        int sys_id,
                        uint64_t node_id,
                        std::function<void()> completion);
    void issue_pool_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_pool_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    // Die backend (joint rollout stage 1, contract §7.4): one die internal
    // memory request. The job drains `service_units` at its equal share of
    // the physical bandwidth; the actual-byte advance is served_service *
    // eta (side-band only). `actual_bytes` is carried for bookkeeping/trace,
    // never drained and never written back into any tensor_size/D2D
    // payload. Completion fires sink->call(EventType::General, data); the
    // sink owns `data` from that callback on (and this model owns it until
    // then, including destruction-time cleanup).
    void issue_die_memory(double service_units,
                          double actual_bytes,
                          Callable* sink,
                          CallData* data);
    void call(EventType type, CallData* data) override;

    // Diagnostic-only dispatch label (v7 event attribution); see Callable.
    [[nodiscard]] const char* profile_dispatch_label() const noexcept override {
        return "hbm_transition";
    }


    bool has_active_jobs() const;

    // Die backend teardown helper: drop every pending DIE_COMPUTE job of
    // `sink` WITHOUT firing its callback (the owning DieExecution is going
    // away -- e.g. Workload destruction). The job payloads are reclaimed
    // here. Legacy COMPUTE/RESTORE/COMM_*/POOL_* jobs are untouched (their
    // completions route through the Workload, which outlives this call).
    void cancel_die_jobs_of(Callable* sink);

    // Read-only side-band counters (implementation doc sec.9.4).  They are
    // accumulated inside advance_to() from the already-computed step_ns and
    // per-user served bytes; they never feed back into transition
    // scheduling, sharing rules, or rounding.  Steps below the numerical
    // time tolerance (the residue-clamp branch) move no measurable time or
    // bytes and are excluded.
    double hbm_busy_ns() const {
        return this->hbm_busy_ns_;
    }
    double hbm_shared_ns() const {
        return this->hbm_shared_ns_;
    }
    double compute_bytes_served() const;
    double restore_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::RESTORE)] +
               coarse_restore_served_;
    }
    double comm_read_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::COMM_READ)] +
               coarse_comm_read_served_;
    }
    double comm_write_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::COMM_WRITE)] +
               coarse_comm_write_served_;
    }
    double pool_read_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::POOL_READ)] +
               coarse_pool_read_served_;
    }
    double pool_write_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::POOL_WRITE)] +
               coarse_pool_write_served_;
    }
    // Die backend side-band (joint rollout stage 1; kept SEPARATE from
    // actual bytes by contract §7.4/§6.2: service units are the drained
    // quantity; actual bytes are service * eta and are never fed back into
    // any byte-based consumer. The per-kind bytes_served_by_kind_ array
    // never accumulates DIE_COMPUTE -- these two dedicated counters are the
    // die disclosure).
    double die_service_units_served() const {
        return this->die_service_units_served_;
    }
    double die_actual_bytes_served() const {
        return this->die_actual_bytes_served_;
    }
    // Peak number of simultaneously active jobs (any phase) on this rank.
    uint64_t peak_concurrent_jobs() const {
        return this->peak_concurrent_jobs_;
    }
    // Number of equal-split reallocations: every job issue that joined
    // already-active jobs plus every completion BATCH that left survivors.
    // The unified batch rule (see call() in LocalHbmBandwidthModel.cc)
    // counts a multi-job completion batch ONCE, not once per completed job
    // -- the disclosure in MetricCollector.cc follows the same batch-level
    // caliber.
    uint64_t redistribution_events() const {
        return this->redistribution_events_;
    }

    // v7 diagnostics: physical transition-callback count and the current
    // mutation-observer registration count (live entries; tombstones
    // pending compaction are excluded). Measurement only.
    [[nodiscard]] uint64_t transition_event_count() const noexcept {
        return transition_event_count_;
    }
    [[nodiscard]] size_t resource_observer_count() const noexcept;
    // v8 reclamation diagnostics (real counts, no stubs): the deque length
    // (live + tombstones pending compaction) and the tombstone share. After
    // the outermost dispatch pass ends, storage == live: tombstoned entries
    // and their captured state have been reclaimed at that boundary.
    [[nodiscard]] size_t resource_observer_storage_size() const noexcept {
        return resource_observers_.size();
    }
    [[nodiscard]] size_t resource_observer_tombstone_count() const noexcept;

    // -------------------------------------------------------------------
    // W1 (joint work ledger, 2026-09-29): per-rank job snapshot for
    // the causal work ledger -- the execution-truth remaining
    // latency/bytes/ops §5.1a requires ("每 job 剩余 latency/bytes/ops、
    // 服务阶段、compute 与 memory 各自完成状态"). The query integrates to
    // the current simulation tick before reading the static point. Each Job
    // holds its wlhd until the completion callback hands it back, so
    // wlhd->sys_id/node_id give
    // the (rank, node_id) identity. remaining_* are integrated up to
    // last_update_tick (exported as as_of_tick by the caller) -- the
    // aggregate-bus N-way arbitration state is the job list itself
    // (streamers = jobs past latency with remaining_bytes > 0).
    // DIE_COMPUTE jobs are INTERNAL memory stages of a still-running outer
    // COMP node (wlhd == nullptr, no ledger node identity) and are EXCLUDED
    // from the snapshot: the outer node is already in the ledger through
    // its own lifecycle, and its remaining work is driven by the DiePlan
    // (phase-2 projection owns plan-level projection, design doc §9.6).
    // -------------------------------------------------------------------
    struct JobSnapshotEntry {
        int kind = 0;             // JobKind raw value
        int sys_id = -1;          // issuing rank (from wlhd)
        uint64_t node_id = 0;     // workload node id (from wlhd)
        uint64_t job_id = 0;      // backend issue sequence (real identity)
        bool transfer_sink = false;  // issued via issue_transfer (independent
                                     // completion sink, no Workload join)
        // Original amounts stay alongside the integrated residual.  A job
        // can first appear in a delta after service has already begun, so
        // the bridge must be able to publish total and remaining together.
        double total_ops = 0.0;
        double total_bytes = 0.0;
        double total_latency_ns = 0.0;
        double remaining_ops = 0.0;
        double remaining_bytes = 0.0;
        double memory_latency_ns = 0.0;
        Tick start_tick = 0;
    };

    [[nodiscard]] std::vector<JobSnapshotEntry> job_snapshot();

    /// Jobs' remaining_* are integrated through this tick (the model's own
    /// fluid clock; the caller exports it as the frame's as_of_tick).
    [[nodiscard]] Tick jobs_as_of_tick() const {
        return last_update_tick;
    }

    /// Rank-local resource membership version used by bounded prediction
    /// windows. Continuous service only changes with the clock; every issue,
    /// active-streamer transition, cancellation, and completion membership
    /// change publishes a new version before user callbacks can re-enter.
    [[nodiscard]] uint64_t resource_version() const noexcept {
        return resource_version_;
    }
    MutationObserverId add_resource_mutation_observer(
        ResourceMutationObserver observer);
    void remove_resource_mutation_observer(MutationObserverId observer_id) noexcept;

    /// Rank aggregate bandwidth (bytes/ns) -- the physical capacity scalar
    /// for the ledger's capacity column (read from Sys). (A former
    /// peak_ops_per_ns companion had no reader and was removed with the
    /// 2026-10-01 dead-code audit.)
    [[nodiscard]] double bandwidth_bytes_per_ns() const;

  private:
    struct CoarseJob {
        WorkloadLayerHandlerData* wlhd = nullptr;
        JobKind kind = JobKind::COMM_READ;
        std::function<void()> transfer_completion;
    };

    struct Job {
        JobKind kind;
        WorkloadLayerHandlerData* wlhd;
        double total_ops;
        double total_bytes;
        double total_latency_ns;
        double remaining_ops;
        double remaining_bytes;
        double memory_latency_ns;
        Tick start_tick;
        // DIE_COMPUTE only (contract §7.4): the drained quantity is service
        // units (remaining_service), and served actual bytes are tracked as
        // service * eta. The sink receives the completion directly; the
        // Workload layer is not involved in a die job's completion (wlhd
        // stays null for DIE_COMPUTE jobs).
        double remaining_service = 0.0;
        double service_eta = 0.0;
        Callable* die_sink = nullptr;
        CallData* die_data = nullptr;
        std::function<void()> transfer_completion;
        int transfer_sys_id = -1;
        uint64_t transfer_node_id = 0;
        // Backend job identity: a per-model monotone issue sequence
        // assigned at issue time. It is the rank backend's real job id
        // (not an export-side counter) and backs the v2 ledger's
        // exogenous-background row identity (execdoc 14.3, 2026-10-07
        // ruling) for transfer jobs injected without a NodeStore node.
        uint64_t issue_seq = 0;
        // v7 stable-slot index (see jobs_/slot_alive_ below).
        size_t slot = 0;
        // v7 completion-candidate membership flag (see completion_candidates_).
        bool in_candidates = false;
    };

    class TransitionData : public CallData {
      public:
        explicit TransitionData(uint64_t generation)
            : generation(generation) {}

        uint64_t generation;
    };

    static constexpr size_t kind_index(JobKind kind) {
        return static_cast<size_t>(kind);
    }

    void issue_job(JobKind kind,
                   uint64_t num_ops,
                   uint64_t tensor_size,
                   WorkloadLayerHandlerData* wlhd,
                   std::function<void()> transfer_completion = {},
                   int transfer_sys_id = -1,
                   uint64_t transfer_node_id = 0);
    void advance_to(Tick now);
    void notify_resource_mutation_before(Tick now);
    void publish_resource_mutation(Tick now);
    // Shared observer dispatch (v7): stable-deque iteration with the same
    // snapshot semantics the old per-notify vector copy implemented. The
    // v8 reclamation fix drives tombstone compaction from the pass guard:
    // the OUTERMOST pass end (depth back to zero, including the
    // exception-unwind path) is the release boundary for removed
    // observers' captured state.
    class ObserverPassGuard {
      public:
        explicit ObserverPassGuard(LocalHbmBandwidthModel& model) noexcept;
        ~ObserverPassGuard();
        ObserverPassGuard(const ObserverPassGuard&) = delete;
        ObserverPassGuard& operator=(const ObserverPassGuard&) = delete;

      private:
        LocalHbmBandwidthModel& model_;
    };
    void dispatch_mutation_observers(bool before_change, Tick now,
                                     uint64_t version);
    void compact_observers();
    void cancel_scheduled_transition();
    void schedule_next_transition();
    static void destroy_transition_data(CallData* data);
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);
    // The drained bandwidth quantity of a job: service units for
    // DIE_COMPUTE, bytes for every legacy kind.
    static double remaining_drain(const Job& job);
    // 4.2 zombie fix (same as the wscllm model): a fully served job
    // contributes neither bandwidth nor concurrency -- detach completed
    // jobs BEFORE any membership accounting (peak/redistribution),
    // delivering their completions from the caller.
    std::vector<Job> detach_completed_jobs();
    // v7 phase-index maintenance (see the class comment): index-sorted
    // membership lists over stable job slots. insert_phase_member keeps the
    // list sorted by slot so every per-job apply loop below iterates in the
    // original issue order.
    void phase_insert(std::vector<size_t>& members, size_t slot);
    bool phase_erase(std::vector<size_t>& members, size_t slot);
    void register_job_phase(Job& job);
    void unregister_job_phase(size_t slot);
    void mark_completion_candidate(Job& job);
    // Dead-slot reclamation: called only at safe points (outside advance_to
    // and outside any observer pass); rebuilds the slot array and every
    // index preserving order.
    void maybe_compact_slots();
    // Shared per-job completion body (elapsed accounting + callback).
    void on_job_complete(const Job& job, Tick now);

    Sys* sys;
    Workload* workload;
    // v7 stable-slot storage: jobs_[slot] is valid iff slot_alive_[slot].
    // Detach marks the slot dead instead of erasing (erasing would shift
    // every later slot and invalidate the phase lists); dead slots are
    // compacted in bulk at safe points while their order is preserved, so
    // every per-job iteration below keeps the original issue order.
    std::vector<Job> jobs_;
    std::map<uint64_t, CoarseJob> coarse_jobs_;
    double coarse_comm_read_served_ = 0.0;
    double coarse_comm_write_served_ = 0.0;
    double coarse_restore_served_ = 0.0;
    double coarse_pool_read_served_ = 0.0;
    double coarse_pool_write_served_ = 0.0;
    double coarse_control_marker_bytes_served_ = 0.0;
    std::vector<uint8_t> slot_alive_;
    size_t live_job_count_ = 0;
    size_t dead_slot_count_ = 0;
    // Phase lists (index-sorted by slot): jobs with memory_latency_ns >
    // kEpsilon; jobs past latency with remaining drain > kEpsilon; every
    // live COMPUTE job (its ops leg drains in parallel with any phase).
    std::vector<size_t> latency_members_;
    std::vector<size_t> stream_members_;
    std::vector<size_t> compute_members_;
    // Live jobs whose remaining drain exceeds the tolerance (the v7
    // has_bandwidth_user counter: an exact O(1) replacement of the old
    // any-of scan over the job vector).
    size_t drain_pending_count_ = 0;
    // Jobs whose tracked quantities landed at/below a boundary since the
    // last detach -- the completion-candidate set. Only these jobs can have
    // crossed complete() since the last detach, so call()/issue paths check
    // candidates instead of rescanning the vector.
    std::vector<size_t> completion_candidates_;
    Tick last_update_tick;
    uint64_t event_generation;
    // Backend job identity sequence (see Job::issue_seq): assigned at issue
    // time, monotone for this rank's model lifetime.
    uint64_t next_issue_seq_ = 1;
    SystemEventHandle scheduled_transition_event;
    struct ObserverEntry {
        MutationObserverId id = 0;  // 0 = tombstone (dead_pass set)
        uint64_t dead_pass = 0;     // pass serial during which it was removed
        ResourceMutationObserver callback;
    };
    // v7 stable observer storage (see the class comment). Entries append at
    // the back; removals tombstone; compaction (dead-slot erase) runs after
    // the outermost pass and only there.
    std::deque<ObserverEntry> resource_observers_;
    size_t observer_pass_depth_ = 0;
    uint64_t observer_pass_serial_ = 0;
    bool observers_compaction_pending_ = false;
    MutationObserverId next_observer_id_ = 1;
    uint64_t resource_version_ = 0;
    Tick next_transition_tick_ = 0;
    bool has_next_transition_tick_ = false;
    bool next_transition_changes_membership_ = false;

    // Side-band observation counters (see the public getters).  Doubles keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double hbm_shared_ns_ = 0.0;
    std::array<double, kJobKindCount> bytes_served_by_kind_ = {};
    // Die backend side-band (see the public getters); DIE_COMPUTE never
    // enters bytes_served_by_kind_.
    double die_service_units_served_ = 0.0;
    double die_actual_bytes_served_ = 0.0;
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
    uint64_t transition_event_count_ = 0;

#ifdef JOINT_V7_EVENT_PROFILE
    // Diagnostic-only sub-attribution of the transition callback (v7 §4.1
    // Q4: which stage of call() carries the cost, and how large the live
    // job set is). Compiled out in production builds.
  public:
    struct TransitionSubTimers {
        uint64_t advance_ns = 0;
        uint64_t detach_ns = 0;
        uint64_t complete_ns = 0;
        uint64_t schedule_ns = 0;
        uint64_t calls = 0;
        size_t last_live_jobs = 0;
        size_t max_live_jobs = 0;
        size_t last_observers = 0;
        size_t max_observers = 0;
    };
    [[nodiscard]] const TransitionSubTimers& profile_transition_timers()
        const noexcept {
        return profile_transition_timers_;
    }

  private:
    TransitionSubTimers profile_transition_timers_;
#endif
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
