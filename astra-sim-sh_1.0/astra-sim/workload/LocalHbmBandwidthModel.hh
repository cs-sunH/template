/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <cstdint>
#include <deque>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class Workload;
class WorkloadLayerHandlerData;

/**
 * Per-NPU multi-user fluid HBM bandwidth model (sh_1.0 generalization of the
 * sh_2.0 two-user LocalHbmBandwidthModel).
 *
 * A rank's local HBM is a single shared bus described by one scalar
 * (`local-mem-bw`): read and write traffic share the same total bandwidth and
 * no peak/sustained distinction is modeled. Every HBM job of the rank
 * competes for that one bus; while N jobs have outstanding HBM bytes, each
 * receives exactly full_rate / N; the instant any job drains its bytes, the
 * survivors are reallocated the freed share (event-driven transitions).
 *
 * Job categories (all competing on the same bus):
 *  - COMP       : roofline compute node, bytes = tensor_size (combined
 *                 read+write accounting), ops = num_ops draining at peak_perf
 *                 in parallel; the job completes when BOTH the byte drain and
 *                 the op drain finish, preserving Roofline's
 *                 max(compute, memory) semantics for a single user.
 *  - COMM_READ  : NoC p2p sender-side endpoint HBM read (bytes = comm bytes).
 *  - COMM_WRITE : NoC p2p receiver-side endpoint HBM write (bytes = comm bytes).
 *  - POOL_READ  : edge-rank local endpoint of a remote-pool MEM_LOAD.
 *  - POOL_WRITE: edge-rank local endpoint of a remote-pool MEM_STORE.
 *
 * `local-mem-latency` is charged once per job at its start (sh_2.0
 * convention); during that latency phase the job does not yet consume
 * bandwidth. NoC multi-hop pass-through traffic, SerDes<->NoC direct
 * pass-through on edge chiplets, the port FIFO itself, and TP collective
 * communication are NOT modeled here (each byte stream is charged exactly
 * once, at its data endpoint).
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    enum class JobKind {
        COMP = 0,
        COMM_READ,
        COMM_WRITE,
        POOL_READ,
        POOL_WRITE,
    };

    LocalHbmBandwidthModel(Sys* sys, Workload* workload);

    // One HBM job per node endpoint; the completion of the job invokes
    // wlhd->workload->call(EventType::General, wlhd) exactly once (the same
    // contract as the sh_2.0 model). The caller owns the wlhd.
    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_comm_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_comm_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    void issue_pool_read(uint64_t bytes,
                         WorkloadLayerHandlerData* wlhd);
    void issue_pool_write(uint64_t bytes,
                          WorkloadLayerHandlerData* wlhd);
    void call(EventType type, CallData* data) override;

    bool has_active_jobs() const;

    // Read-only side-band observation counters (doc: 请求实例映射与KV冷热
    // 管理机制.md §2.5). Accumulated inside advance_to() from the
    // already-computed step_ns / per-user rate; they never feed back into
    // transition scheduling, sharing rules, or rounding.
    double hbm_busy_ns() const {
        return this->hbm_busy_ns_;
    }
    double compute_bytes_served() const {
        return this->served_bytes_[0];
    }
    double comm_read_bytes_served() const {
        return this->served_bytes_[1];
    }
    double comm_write_bytes_served() const {
        return this->served_bytes_[2];
    }
    double pool_read_bytes_served() const {
        return this->served_bytes_[3];
    }
    double pool_write_bytes_served() const {
        return this->served_bytes_[4];
    }
    // Max number of simultaneously active jobs (any category).
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
    // True HBM utilization over [0, window_ns]: busy_ns / window.
    double hbm_utilization(Tick window_ns) const;

  private:
    struct Job {
        JobKind kind;
        WorkloadLayerHandlerData* wlhd;
        double remaining_ops;      // COMP only; drains at peak_perf
        double remaining_bytes;    // drains at the shared rate
        double memory_latency_ns;  // once per job, drains with wall time
        Tick start_tick;
    };

    class TransitionData : public CallData {
      public:
        explicit TransitionData(uint64_t generation)
            : generation(generation) {}

        uint64_t generation;
    };

    void issue(JobKind kind,
               double bytes,
               double ops,
               WorkloadLayerHandlerData* wlhd);
    void advance_to(Tick now);
    void schedule_next_transition();
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);

    Sys* sys;
    Workload* workload;
    // deque: stable element pointers across push/pop (advance_to keeps
    // pointers into the jobs while reordering never happens).
    std::deque<Job> jobs;
    Tick last_update_tick;
    uint64_t event_generation;

    // Side-band observation counters (see the public getters). Doubles keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double served_bytes_[5] = {0.0, 0.0, 0.0, 0.0, 0.0};
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
