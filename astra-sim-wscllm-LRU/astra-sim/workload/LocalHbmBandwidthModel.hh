/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

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
 *   - POOL_WRITE  : off-chip pool endpoint HBM write (bytes = tensor_size).
 *
 * All outstanding jobs strictly share the rank's single configured HBM
 * bandwidth: each active byte streamer receives full_rate/N. When any job
 * consumes its remaining HBM bytes it completes immediately and the
 * survivors are re-split (full_rate/N') at the same event -- no reservation,
 * no work conservation loss. Each job pays sys->local_mem_latency once at
 * start (the pre-existing convention).
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
    void call(EventType type, CallData* data) override;

    bool has_active_jobs() const;

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
    // Peak number of simultaneously active jobs (any phase) on this rank.
    uint64_t peak_concurrent_jobs() const {
        return this->peak_concurrent_jobs_;
    }
    // Number of equal-split reallocations: every job issue that joined
    // already-active jobs plus every job completion that left survivors
    // (each changes full_rate/N for the remaining streams).
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

    static constexpr size_t kind_index(JobKind kind) {
        return static_cast<size_t>(kind);
    }

    void issue_job(JobKind kind,
                   uint64_t num_ops,
                   uint64_t tensor_size,
                   WorkloadLayerHandlerData* wlhd);
    void advance_to(Tick now);
    void cancel_scheduled_transition();
    void schedule_next_transition();
    static void destroy_transition_data(CallData* data);
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);

    Sys* sys;
    Workload* workload;
    // Active jobs in issue order. The old two-slot (compute/restore)
    // representation and its "second same-kind job throws" guards were
    // removed with the N-way generalization; HardwareResource's single
    // hbm_dma slot still structurally guarantees at most one in-flight
    // restore per rank.
    std::vector<Job> jobs;
    Tick last_update_tick;
    uint64_t event_generation;
    SystemEventHandle scheduled_transition_event;

    // Side-band observation counters (see the public getters).  They keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double hbm_shared_ns_ = 0.0;
    std::array<double, kJobKindCount> bytes_served_by_kind_ = {};
    uint64_t peak_concurrent_jobs_ = 0;
    uint64_t redistribution_events_ = 0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
