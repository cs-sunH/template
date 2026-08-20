/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <cstdint>
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
 *              out of its local HBM (created by Workload::issue_send_comm).
 *   COMM_WRITE p2p recv endpoint: the receiver's rank writes `comm bytes`
 *              into its local HBM (created by Workload::issue_recv_comm).
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
 * the byte drain and does not consume bandwidth).
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    enum class JobKind { COMP, COMM_READ, COMM_WRITE };

    LocalHbmBandwidthModel(Sys* sys, Workload* workload);

    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_comm_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_comm_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    void call(EventType type, CallData* data) override;

    bool has_active_jobs() const;

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
        Tick start_tick;
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
    void schedule_next_transition();
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);
    void note_membership_change();

    Sys* sys;
    Workload* workload;
    std::vector<Job> jobs;  // active jobs, arbitrary cardinality (the
                            // sh_2.0 "at most one job per kind" limit was
                            // removed with the generalization)
    Tick last_update_tick;
    uint64_t event_generation;

    // Side-band observation counters (see the public getters).  Doubles keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double compute_bytes_served_ = 0.0;
    double comm_read_bytes_served_ = 0.0;
    double comm_write_bytes_served_ = 0.0;
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
