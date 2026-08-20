/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <array>
#include <cstdint>
#include <optional>
#include <vector>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class Workload;
class WorkloadLayerHandlerData;

/**
 * Per-NPU fluid HBM model with N-way equal-split bandwidth sharing.
 *
 * Every HBM user on a rank runs as a job in this model. While N jobs have
 * outstanding HBM bytes, each receives exactly full_rate / N of the rank's
 * configured HBM bandwidth (strict equal split; reads and writes share one
 * bus and one total bandwidth, no peak/sustained distinction). As soon as
 * any job consumes its remaining HBM bytes the survivors are re-split
 * immediately (event-driven reallocation). Job kinds:
 *
 *   COMPUTE     inference COMP: bytes = tensor_size, ops drained at peak
 *               perf in parallel (dual constraint, Roofline
 *               max(compute, memory) semantics preserved)
 *   RESTORE     KV-restore DMA write (serial semantics unchanged)
 *   COMM_READ   NoC p2p send endpoint: sender reads comm bytes from HBM
 *   COMM_WRITE  NoC p2p recv endpoint: receiver writes comm bytes to HBM
 *   POOL_READ   off-chip pool store endpoint: edge rank reads from HBM
 *   POOL_WRITE  off-chip pool load endpoint: edge rank writes to HBM
 *
 * local_mem_latency is charged once per job at issue (existing convention).
 * Multi-hop NoC traffic through a rank never enters this model (only data
 * endpoints are charged); HardwareResource guarantees at most one in-flight
 * COMP and serializes restore DMAs through the single hbm_dma slot, so the
 * model itself no longer rejects same-kind concurrency.
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    enum class JobKind : int {
        COMPUTE = 0,
        RESTORE = 1,
        COMM_READ = 2,
        COMM_WRITE = 3,
        POOL_READ = 4,
        POOL_WRITE = 5,
    };
    static constexpr int kJobKindCount = 6;

    LocalHbmBandwidthModel(Sys* sys, Workload* workload);

    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_restore(uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_comm_read(uint64_t bytes, WorkloadLayerHandlerData* wlhd);
    void issue_comm_write(uint64_t bytes, WorkloadLayerHandlerData* wlhd);
    void issue_pool_read(uint64_t bytes, WorkloadLayerHandlerData* wlhd);
    void issue_pool_write(uint64_t bytes, WorkloadLayerHandlerData* wlhd);
    void call(EventType type, CallData* data) override;

    bool has_active_jobs() const;
    size_t active_job_count() const {
        return jobs.size();
    }

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
        return this->kind_bytes_served_[static_cast<int>(JobKind::COMPUTE)];
    }
    double restore_bytes_served() const {
        return this->kind_bytes_served_[static_cast<int>(JobKind::RESTORE)];
    }
    double comm_read_bytes_served() const {
        return this->kind_bytes_served_[static_cast<int>(JobKind::COMM_READ)];
    }
    double comm_write_bytes_served() const {
        return this->kind_bytes_served_[static_cast<int>(JobKind::COMM_WRITE)];
    }
    double pool_read_bytes_served() const {
        return this->kind_bytes_served_[static_cast<int>(JobKind::POOL_READ)];
    }
    double pool_write_bytes_served() const {
        return this->kind_bytes_served_[static_cast<int>(JobKind::POOL_WRITE)];
    }
    // Peak number of simultaneously active jobs (all kinds, incl. the
    // memory-latency-only phase) and the count of equal-share
    // redistributions: every membership change that leaves at least one
    // active bandwidth user (a job joining a non-empty set, or a completion
    // leaving survivors behind) recomputes the full_rate/N split and is
    // counted once (中-4④, unified with face).
    uint64_t peak_concurrent_jobs() const {
        return this->peak_concurrent_jobs_;
    }
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

    void issue_job(Job&& job);
    void advance_to(Tick now);
    void schedule_next_transition();
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);

    Sys* sys;
    Workload* workload;
    std::vector<Job> jobs;  // insertion order; completion preserves it
    Tick last_update_tick;
    uint64_t event_generation;

    // Side-band observation counters (see the public getters).  Doubles keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double hbm_shared_ns_ = 0.0;
    std::array<double, kJobKindCount> kind_bytes_served_ = {};
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
