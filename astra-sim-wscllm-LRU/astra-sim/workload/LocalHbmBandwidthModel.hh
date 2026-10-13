/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <map>
#include <array>
#include <cstddef>
#include <cstdint>
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
 *   - RESTORE     : KV-restore DMA write (serial graph semantics unchanged),
 *   - COMM_READ   : NoC p2p send-side endpoint HBM read (bytes = comm size),
 *   - COMM_WRITE  : NoC p2p recv-side endpoint HBM write (bytes = comm size),
 *   - POOL_READ   : off-chip pool endpoint HBM read (bytes = tensor_size),
 *   - POOL_WRITE  : off-chip pool endpoint HBM write (bytes = tensor_size),
 *   - DIE_COMPUTE : die-backend internal memory stage (stage B, contract
 *                   §7.4): one job per MemoryRequest of a DieExecution.
 *                   The job drains SERVICE UNITS (actual_bytes/eta) at the
 *                   equal-share rate -- the access inefficiency is spent as
 *                   service time, never by inflating actual bytes -- and
 *                   has NO latency phase and NO FLOP leg (the DieExecution
 *                   charges local_mem_latency itself, once per outer node,
 *                   and drains compute through the plan's PE/VEC stages).
 *                   Completion fires the sink Callable directly (the sink
 *                   is the DieExecution, not the Workload).
 *
 * All outstanding jobs strictly share the rank's single configured HBM
 * bandwidth: each active byte streamer receives full_rate/N. When any job
 * consumes its remaining HBM bytes it completes immediately and the
 * survivors are re-split (full_rate/N') at the same event -- no reservation,
 * no work conservation loss. Each job pays sys->local_mem_latency once at
 * start (the pre-existing convention) -- except DIE_COMPUTE jobs, which
 * never enter the latency phase (see above).
 *
 * Reads and writes share one bus and one aggregate bandwidth scalar
 * (system key "local-mem-bw"); there is no peak/sustained distinction.
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    enum class JobKind {
        COMPUTE = 0,
        RESTORE,
        COMM_READ,
        COMM_WRITE,
        POOL_READ,
        POOL_WRITE,
        DIE_COMPUTE,
        KIND_COUNT,
    };
    static constexpr size_t kJobKindCount =
        static_cast<size_t>(JobKind::KIND_COUNT);

    LocalHbmBandwidthModel(Sys* sys, Workload* workload);
    ~LocalHbmBandwidthModel() override;

    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_restore(uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_comm_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_comm_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    void issue_pool_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_pool_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    // Stage B (contract §7.4): one die internal memory request. The job
    // drains `service_units` at its equal share of the physical bandwidth;
    // the actual-byte advance is served_service * eta (side-band only).
    // `actual_bytes` is carried for bookkeeping/trace, never drained and
    // never written back into any tensor_size/D2D/pool payload. Completion
    // fires sink->call(EventType::General, data); the sink owns `data`
    // from that callback on (and this model owns it until then, including
    // destruction-time cleanup). The die job enters the SAME N-way
    // equal-split arbitration as COMPUTE/RESTORE/COMM_*/POOL_* -- the
    // legacy kinds' sharing semantics are byte-for-byte unchanged.
    void issue_die_memory(double service_units,
                          double actual_bytes,
                          Callable* sink,
                          CallData* data);
    void call(EventType type, CallData* data) override;

    bool has_active_jobs() const;
    void record_coarse_die_hbm_bytes(double read_bytes, double write_bytes) {
        coarse_die_read_served_ += read_bytes;
        coarse_die_write_served_ += write_bytes;
    }
    double coarse_die_read_bytes_served() const { return coarse_die_read_served_; }
    double coarse_die_write_bytes_served() const { return coarse_die_write_served_; }
    double coarse_restore_bytes_served() const { return coarse_restore_served_; }
    double coarse_pool_read_bytes_served() const { return coarse_pool_read_served_; }
    double coarse_pool_write_bytes_served() const { return coarse_pool_write_served_; }


    // DeadEnd-census diagnostic only (2026-10-11): one stderr line per
    // in-flight job plus the transition-prediction state (validity and
    // scheduled tick). Purely read-only; never called on the hot path.
    void dump_census_to_stderr(const char* prefix) const;

    // Stage B teardown helper: drop every pending DIE_COMPUTE job of `sink`
    // WITHOUT firing its callback (the owning DieExecution is going away --
    // e.g. Workload destruction). The job payloads are reclaimed here. Old
    // COMPUTE/RESTORE/COMM_*/POOL_* jobs are untouched (their completions
    // route through the Workload, which outlives this call).
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
    double compute_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::COMPUTE)];
    }
    double restore_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::RESTORE)];
    }
    double comm_read_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::COMM_READ)];
    }
    double comm_write_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::COMM_WRITE)];
    }
    double pool_read_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::POOL_READ)];
    }
    double pool_write_bytes_served() const {
        return this->bytes_served_by_kind_[kind_index(JobKind::POOL_WRITE)];
    }
    // Stage B die side-band (kept SEPARATE from actual bytes by contract
    // §7.4/§6.2: service units are the drained quantity; actual bytes are
    // service * eta and are never fed back into any byte-based consumer).
    // Deliberately NOT part of bytes_served_by_kind_: mixing service units
    // into the per-kind byte counters would pollute the six legacy HBM
    // metrics.
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
    // Number of real equal-share switches (wl-metrics-C3): every event that
    // actually changes the full_rate/N split among streaming jobs counts.
    //   - a zero-latency job issued while at least one byte-streaming job
    //     remains counts once at issue (it joins the bus on the next step);
    //   - jobs whose fixed latency expires onto a bus with at least one
    //     streamer count once each at that expiry transition (the first
    //     streamer onto an idle bus starts no re-split);
    //   - each completion batch that leaves at least one byte-streaming
    //     survivor counts ONCE regardless of how many jobs completed
    //     together (batch-level rule).
    uint64_t redistribution_events() const {
        return this->redistribution_events_;
    }

  private:
    struct Job {
        JobKind kind;
        WorkloadLayerHandlerData* wlhd;
        double remaining_ops;
        double remaining_bytes;
        double memory_latency_ns;
        // DIE_COMPUTE only (contract §7.4): the drained quantity is service
        // units (remaining_service), and served actual bytes are tracked as
        // service * eta. The sink receives the completion directly; the
        // Workload layer is not involved in a die job's completion.
        double remaining_service = 0.0;
        double service_eta = 0.0;
        Callable* die_sink = nullptr;
        CallData* die_data = nullptr;
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
                   WorkloadLayerHandlerData* wlhd);
    void advance_to(Tick now);
    // Retire every complete() job: erase from `jobs`, apply the batch-level
    // redistribution count, and fire the completion callbacks. Shared by the
    // transition callback and by issue_job, which must retire jobs that
    // advance_to() completed at exactly the issue tick instead of deferring
    // them to the next transition (M23).
    void retire_completed_jobs();
    void cancel_scheduled_transition();
    void schedule_next_transition();
    static void destroy_transition_data(CallData* data);
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);
    // The drained bandwidth quantity of a job: service units for
    // DIE_COMPUTE, bytes for every legacy kind. Unifies the six
    // classification points of advance_to()/schedule_next_transition() so
    // a die job takes part in the same N-way equal split without touching
    // the legacy kinds' arithmetic (for them the predicate is exactly the
    // old remaining_bytes read).
    static double remaining_drain(const Job& job);

    Sys* sys;
    Workload* workload;
    // Active jobs in issue order. The old two-slot (compute/restore)
    // representation and its "second same-kind job throws" guards were
    // removed with the N-way generalization; HardwareResource's single
    // hbm_dma slot still structurally guarantees at most one in-flight
    // restore per rank.
    std::vector<Job> jobs;
    double coarse_comm_read_served_ = 0.0;
    double coarse_comm_write_served_ = 0.0;
    double coarse_restore_served_ = 0.0;
    double coarse_pool_read_served_ = 0.0;
    double coarse_pool_write_served_ = 0.0;
    double coarse_die_read_served_ = 0.0;
    double coarse_die_write_served_ = 0.0;
    std::map<uint64_t, WorkloadLayerHandlerData*> coarse_jobs_;

    Tick last_update_tick;
    uint64_t event_generation;
    SystemEventHandle scheduled_transition_event;

    // Side-band observation counters (see the public getters).  They keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double hbm_shared_ns_ = 0.0;
    std::array<double, kJobKindCount> bytes_served_by_kind_ = {};
    // DIE_COMPUTE side-band (stage B): service units and actual bytes are
    // tracked SEPARATELY from bytes_served_by_kind_ (see the getters above).
    double die_service_units_served_ = 0.0;
    double die_actual_bytes_served_ = 0.0;
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
