/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <cstdint>
#include <map>
#include <vector>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class Workload;
class WorkloadLayerHandlerData;

/**
 * Per-rank (per-chiplet) fluid model of the shared local HBM bandwidth.
 *
 * Generalization of the sh_2.0 two-user (inference COMP + KV-restore DMA)
 * model to N concurrent users. Every workload (=rank=chiplet) owns exactly
 * one instance when "hbm-bandwidth-contention" is enabled in the system
 * configuration (Sys auto-disables the flag when local-mem-bw <= 0).
 *
 * Users of the single HBM bandwidth scalar (read/write share one bus, no
 * peak/sustained distinction, no direction split):
 *   COMP       roofline compute traffic: bytes = tensor_size (read+write
 *              merged into one charge), FLOPs drain at peak_perf in
 *              parallel; the job completes when BOTH drains finish (single
 *              user keeps Roofline's max(compute, memory) semantics).
 *   COMM_READ  p2p send endpoint: the sender's rank reads `comm bytes`
 *              out of its local HBM (created by Workload::issue_send_comm);
 *              since 2026-09-28 also the ALL_REDUCE endpoint buffer, and
 *              since 2026-09-29 ALL FOUR collective types charge their
 *              endpoint buffers identically here (ALL_REDUCE/ALL_TO_ALL/
 *              ALL_GATHER/REDUCE_SCATTER, created by
 *              Workload::issue_coll_comm, bytes = coll size, read+write
 *              merged with the same convention as COMP; the node completes
 *              at the join of the collective and this job) -- so
 *              comm_read_bytes_served carries p2p send plus collective
 *              endpoint traffic and is NOT a pure p2p-read figure.
 *   COMM_WRITE p2p recv endpoint: the receiver's rank writes `comm bytes`
 *              into its local HBM (created by Workload::issue_recv_comm).
 *   DIE_COMPUTE die-backend internal memory stage (contract §7.4): one
 *              job per MemoryRequest of a DieExecution. The job drains
 *              SERVICE UNITS (actual_bytes/eta) at the equal-share rate --
 *              the access inefficiency is spent as service time, never by
 *              inflating actual bytes -- and has NO latency phase and NO
 *              FLOP leg (the DieExecution charges local_mem_latency
 *              itself, once per outer node, and drains compute through
 *              the plan's PE/VEC stages). Completion fires the sink
 *              Callable directly (the sink is the DieExecution, not the
 *              Workload).
 *
 * Multi-hop NoC routes never touch intermediate chiplets' HBM (router
 * pass-through); the network backend only charges endpoints, and this model
 * mirrors that: only the send-source and recv-destination ranks create jobs.
 *
 * Arbitration: with N active bandwidth users each receives exactly
 * full_rate/N; the instant any user finishes, the rate is redistributed
 * among the survivors (event-driven, no fixed quantum). The bandwidth is
 * the configured local-mem-bw verbatim. local_mem_latency is charged once
 * per job at its start (sh_2.0 convention: a latency-only phase precedes
 * the byte drain and does not consume bandwidth) -- except DIE_COMPUTE
 * jobs, which never enter the latency phase (see above).
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    enum class JobKind { COMP, COMM_READ, COMM_WRITE, DIE_COMPUTE };

    LocalHbmBandwidthModel(Sys* sys, Workload* workload);
    ~LocalHbmBandwidthModel() override;

    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_comm_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_comm_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    // Die backend (contract §7.4): one die internal memory request. The
    // job drains `service_units` at its equal share of the physical
    // bandwidth; the actual-byte advance is served_service * eta
    // (side-band only). `actual_bytes` is carried for bookkeeping/trace,
    // never drained and never written back into any tensor_size/D2D
    // payload. Completion fires sink->call(EventType::General, data); the
    // sink owns `data` from that callback on (and this model owns it until
    // then, including destruction-time cleanup).
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


    // Die-backend teardown helper: drop every pending DIE_COMPUTE job of
    // `sink` WITHOUT firing its callback (the owning DieExecution is going
    // away -- e.g. Workload destruction). The job payloads are reclaimed
    // here. Old COMP/COMM_* jobs are untouched (their completions route
    // through the Workload, which outlives this call).
    void cancel_die_jobs_of(Callable* sink);

    // Read-only side-band counters.  They are accumulated inside
    // advance_to() from the already-computed step_ns and per-user served
    // bytes; they never feed back into transition scheduling, sharing
    // rules, or rounding.  Steps below the numerical time tolerance (the
    // residue-clamp branch) move no measurable time or bytes and are
    // excluded.
    double hbm_busy_ns() const {
        return this->hbm_busy_ns_;
    }
    double compute_bytes_served() const {
        return this->compute_bytes_served_;
    }
    double comm_read_bytes_served() const {
        return this->comm_read_bytes_served_;
    }
    double comm_write_bytes_served() const {
        return this->comm_write_bytes_served_;
    }
    // Die side-band (kept SEPARATE from actual bytes by contract
    // §7.4/§6.2: service units are the drained quantity; actual bytes are
    // service * eta and are never fed back into any byte-based consumer).
    double die_service_units_served() const {
        return this->die_service_units_served_;
    }
    double die_actual_bytes_served() const {
        return this->die_actual_bytes_served_;
    }
    // Largest number of simultaneously active jobs observed at a
    // membership change (issue/completion).
    uint64_t peak_concurrent_jobs() const {
        return this->peak_concurrent_jobs_;
    }
    // Count of equal-share redistributions: every membership change that
    // leaves at least one active bandwidth user (a job joining a non-empty
    // set, or a completion leaving survivors behind) recomputes the
    // full_rate/N split and is counted once.
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
        // DIE_COMPUTE only (contract §7.4): the drained quantity is
        // service units (remaining_service), and served actual bytes are
        // tracked as service * eta. The sink receives the completion
        // directly; the Workload layer is not involved in a die job's
        // completion.
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

    void issue_job(JobKind kind,
                   uint64_t num_ops,
                   uint64_t bytes,
                   WorkloadLayerHandlerData* wlhd);
    void advance_to(Tick now);
    // Single completion dispatcher for every detached (completed) job:
    // legacy kinds -> Workload::on_local_hbm_job_complete; DIE_COMPUTE ->
    // the owning DieExecution sink (contract §7.4).
    void deliver_job_completion(const Job& job);
    void cancel_scheduled_transition();
    void schedule_next_transition();
    static void destroy_transition_data(CallData* data);
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);
    // The drained bandwidth quantity of a job: service units for
    // DIE_COMPUTE, bytes for every legacy kind.
    static double remaining_drain(const Job& job);
    // 4.2 zombie fix: a fully served job contributes neither bandwidth nor
    // concurrency -- detach completed jobs BEFORE any membership
    // accounting (peak/redistribution), delivering their completions from
    // the caller.
    std::vector<Job> detach_completed_jobs();
    // Definition-aligned redistribution gate: only a live job with
    // outstanding bytes reshapes shares.
    static bool has_bandwidth_user(const std::vector<Job>& jobs);
    void note_membership_change();

    Sys* sys;
    Workload* workload;
    std::vector<Job> jobs;
    double coarse_comm_read_served_ = 0.0;
    double coarse_comm_write_served_ = 0.0;
    double coarse_restore_served_ = 0.0;
    double coarse_pool_read_served_ = 0.0;
    double coarse_pool_write_served_ = 0.0;
    double coarse_die_read_served_ = 0.0;
    double coarse_die_write_served_ = 0.0;
    std::map<uint64_t, WorkloadLayerHandlerData*> coarse_jobs_;
  // active jobs, arbitrary cardinality
    Tick last_update_tick;
    uint64_t event_generation;
    SystemEventHandle scheduled_transition_event;

    // Side-band observation counters (see the public getters).  Doubles keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double compute_bytes_served_ = 0.0;
    double comm_read_bytes_served_ = 0.0;
    double comm_write_bytes_served_ = 0.0;
    double die_service_units_served_ = 0.0;
    double die_actual_bytes_served_ = 0.0;
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
