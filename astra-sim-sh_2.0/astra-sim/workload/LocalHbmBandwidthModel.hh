/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef __LOCAL_HBM_BANDWIDTH_MODEL_HH__
#define __LOCAL_HBM_BANDWIDTH_MODEL_HH__

#include <cstdint>
#include <optional>

#include "astra-sim/system/Callable.hh"
#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;
class Workload;
class WorkloadLayerHandlerData;

/**
 * Per-NPU fluid HBM model for inference/restore overlap.
 *
 * At most one inference COMP node and one KV-restore DMA node can be active on
 * a rank.  If both have outstanding HBM bytes, each receives exactly half of
 * that rank's configured HBM bandwidth.  As soon as either side consumes its
 * remaining HBM bytes, the other immediately returns to full bandwidth.
 * Compute FLOPs continue at peak rate in parallel, preserving Roofline's
 * max(compute time, memory time) semantics.
 */
class LocalHbmBandwidthModel : public Callable {
  public:
    LocalHbmBandwidthModel(Sys* sys, Workload* workload);

    void issue_compute(uint64_t num_ops,
                       uint64_t tensor_size,
                       WorkloadLayerHandlerData* wlhd);
    void issue_restore(uint64_t tensor_size,
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
        return this->compute_bytes_served_;
    }
    double restore_bytes_served() const {
        return this->restore_bytes_served_;
    }

  private:
    enum class JobKind { COMPUTE, RESTORE };

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

    void advance_to(Tick now);
    void schedule_next_transition();
    static bool memory_done(const Job& job);
    static bool complete(const Job& job);

    Sys* sys;
    Workload* workload;
    std::optional<Job> compute_job;
    std::optional<Job> restore_job;
    Tick last_update_tick;
    uint64_t event_generation;

    // Side-band observation counters (see the public getters).  Doubles keep
    // the exact same values advance_to() already computed; nothing reads
    // them while scheduling the next transition.
    double hbm_busy_ns_ = 0.0;
    double hbm_shared_ns_ = 0.0;
    double compute_bytes_served_ = 0.0;
    double restore_bytes_served_ = 0.0;
};

}  // namespace AstraSim

#endif /* __LOCAL_HBM_BANDWIDTH_MODEL_HH__ */
