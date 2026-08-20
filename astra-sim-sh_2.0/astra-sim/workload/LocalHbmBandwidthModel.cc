/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/LocalHbmBandwidthModel.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"

using namespace AstraSim;

namespace {
constexpr double kEpsilon = 1e-9;
}

LocalHbmBandwidthModel::LocalHbmBandwidthModel(Sys* sys, Workload* workload)
    : sys(sys),
      workload(workload),
      last_update_tick(Sys::boostedTick()),
      event_generation(0) {
    if (sys == nullptr || workload == nullptr) {
        throw std::invalid_argument("local HBM model requires Sys and Workload");
    }
    if (sys->local_mem_bw <= 0 || sys->peak_perf <= 0) {
        throw std::invalid_argument(
            "local HBM sharing requires positive bandwidth and peak compute");
    }
}

bool LocalHbmBandwidthModel::has_active_jobs() const {
    return !jobs.empty();
}

bool LocalHbmBandwidthModel::memory_done(const Job& job) {
    return job.memory_latency_ns <= kEpsilon &&
        job.remaining_bytes <= kEpsilon;
}

bool LocalHbmBandwidthModel::complete(const Job& job) {
    return memory_done(job) && job.remaining_ops <= kEpsilon;
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        throw std::runtime_error("local HBM model time moved backwards");
    }
    const double elapsed_ns = static_cast<double>(now - last_update_tick);
    if (elapsed_ns <= 0) {
        return;
    }

    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (Job& job : jobs) {
        if (job.kind == JobKind::COMPUTE) {
            job.remaining_ops = std::max(
                0.0,
                job.remaining_ops - compute_rate_ops_per_ns * elapsed_ns);
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    double remaining_ns = elapsed_ns;
    while (remaining_ns > kEpsilon) {
        std::vector<Job*> bandwidth_users;
        double step_ns = remaining_ns;
        for (Job& job : jobs) {
            if (job.memory_latency_ns > kEpsilon) {
                step_ns = std::min(step_ns, job.memory_latency_ns);
            } else if (job.remaining_bytes > kEpsilon) {
                bandwidth_users.push_back(&job);
            }
        }

        const double per_user_rate = bandwidth_users.empty()
            ? 0.0
            : full_rate_bytes_per_ns /
                static_cast<double>(bandwidth_users.size());
        if (per_user_rate > 0) {
            for (const Job* job : bandwidth_users) {
                step_ns = std::min(
                    step_ns, job->remaining_bytes / per_user_rate);
            }
        }

        if (step_ns <= kEpsilon) {
            // A subtraction at the previous transition can leave a byte
            // residue whose service time is below the numerical time
            // tolerance, while the byte value itself is still above the
            // quantity tolerance.  Breaking here would preserve that residue
            // forever and create one transition per nanosecond.  Clamp every
            // quantity that can complete within the time tolerance, then
            // recompute rates without consuming measurable simulation time.
            bool clamped_residue = false;
            for (Job& job : jobs) {
                if (job.memory_latency_ns > 0 &&
                    job.memory_latency_ns <= kEpsilon) {
                    job.memory_latency_ns = 0;
                    clamped_residue = true;
                }
                if (job.remaining_bytes > 0 &&
                    job.remaining_bytes <= kEpsilon) {
                    job.remaining_bytes = 0;
                    clamped_residue = true;
                }
            }
            if (per_user_rate > 0) {
                for (Job* job : bandwidth_users) {
                    if (job->remaining_bytes / per_user_rate <= kEpsilon) {
                        job->remaining_bytes = 0;
                        clamped_residue = true;
                    }
                }
            }
            if (!clamped_residue) {
                throw std::runtime_error(
                    "local HBM model made no progress at a transition");
            }
            continue;
        }

        for (Job& job : jobs) {
            if (job.memory_latency_ns > kEpsilon) {
                job.memory_latency_ns =
                    std::max(0.0, job.memory_latency_ns - step_ns);
            }
        }
        for (Job* job : bandwidth_users) {
            job->remaining_bytes = std::max(
                0.0, job->remaining_bytes - per_user_rate * step_ns);
        }
        // Read-only side-band accumulation (doc sec.9.4): reuse the step_ns
        // and per_user_rate already computed above.  These counters must not
        // change any value used for the next transition.
        if (!bandwidth_users.empty()) {
            this->hbm_busy_ns_ += step_ns;
            if (bandwidth_users.size() > 1) {
                this->hbm_shared_ns_ += step_ns;
            }
            for (const Job* job : bandwidth_users) {
                this->kind_bytes_served_[static_cast<int>(job->kind)] +=
                    per_user_rate * step_ns;
            }
        }
        remaining_ns -= step_ns;
    }
    last_update_tick = now;
}

void LocalHbmBandwidthModel::schedule_next_transition() {
    ++event_generation;
    if (!has_active_jobs()) {
        return;
    }

    std::vector<const Job*> bandwidth_users;
    double next_ns = std::numeric_limits<double>::infinity();
    for (const Job& job : jobs) {
        if (job.memory_latency_ns > kEpsilon) {
            next_ns = std::min(next_ns, job.memory_latency_ns);
        } else if (job.remaining_bytes > kEpsilon) {
            bandwidth_users.push_back(&job);
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    const double per_user_rate = bandwidth_users.empty()
        ? 0.0
        : full_rate_bytes_per_ns /
            static_cast<double>(bandwidth_users.size());
    if (per_user_rate > 0) {
        for (const Job* job : bandwidth_users) {
            next_ns = std::min(next_ns, job->remaining_bytes / per_user_rate);
        }
    }

    for (const Job& job : jobs) {
        if (job.kind == JobKind::COMPUTE && memory_done(job) &&
            job.remaining_ops > kEpsilon) {
            next_ns = std::min(
                next_ns, job.remaining_ops / (sys->peak_perf / 1e9));
        }
    }

    if (!std::isfinite(next_ns)) {
        throw std::runtime_error("local HBM model has no schedulable transition");
    }
    Tick delay = static_cast<Tick>(std::ceil(next_ns));
    delay = std::max<Tick>(1, delay);
    sys->register_event(
        this,
        EventType::General,
        new TransitionData(event_generation),
        delay);
}

void LocalHbmBandwidthModel::issue_job(Job&& job) {
    advance_to(Sys::boostedTick());
    const bool joins_active_set = !jobs.empty();
    const Tick start_tick = Sys::boostedTick();
    jobs.push_back(std::move(job));
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = jobs.size();
    }
    if (joins_active_set) {
        // The new job dilutes the equal split of the survivors.
        ++redistribution_events_;
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::issue_compute(
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(Job{
        JobKind::COMPUTE,
        wlhd,
        static_cast<double>(num_ops),
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
}

void LocalHbmBandwidthModel::issue_restore(
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(Job{
        JobKind::RESTORE,
        wlhd,
        0.0,
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
}

void LocalHbmBandwidthModel::issue_comm_read(
    uint64_t bytes, WorkloadLayerHandlerData* wlhd) {
    issue_job(Job{
        JobKind::COMM_READ,
        wlhd,
        0.0,
        static_cast<double>(bytes),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
}

void LocalHbmBandwidthModel::issue_comm_write(
    uint64_t bytes, WorkloadLayerHandlerData* wlhd) {
    issue_job(Job{
        JobKind::COMM_WRITE,
        wlhd,
        0.0,
        static_cast<double>(bytes),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
}

void LocalHbmBandwidthModel::issue_pool_read(
    uint64_t bytes, WorkloadLayerHandlerData* wlhd) {
    issue_job(Job{
        JobKind::POOL_READ,
        wlhd,
        0.0,
        static_cast<double>(bytes),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
}

void LocalHbmBandwidthModel::issue_pool_write(
    uint64_t bytes, WorkloadLayerHandlerData* wlhd) {
    issue_job(Job{
        JobKind::POOL_WRITE,
        wlhd,
        0.0,
        static_cast<double>(bytes),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
}

void LocalHbmBandwidthModel::call(EventType, CallData* data) {
    TransitionData* transition = static_cast<TransitionData*>(data);
    const uint64_t generation = transition->generation;
    delete transition;
    if (generation != event_generation) {
        return;
    }

    const Tick now = Sys::boostedTick();
    advance_to(now);
    std::vector<Job> completed;
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (complete(*it)) {
            completed.push_back(std::move(*it));
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
    if (!completed.empty() && !jobs.empty()) {
        // Survivors are re-split from full_rate/N to full_rate/(N-k).
        ++redistribution_events_;
    }

    for (const Job& job : completed) {
        const uint64_t elapsed = now - job.start_tick;
        // 中-4② unified binary rule (sh_3.0 semantics): every non-COMP job
        // here (COMM_READ/WRITE, RESTORE, POOL_READ/WRITE) occupies local
        // HBM bandwidth for its elapsed time -- a comm endpoint's stretched
        // duration is HBM occupancy, not network transfer time, so the
        // legacy comm tics bucket no longer receives it.
        if (job.kind == JobKind::COMPUTE) {
            workload->hw_resource->tics_gpu_ops += elapsed;
        } else {
            workload->hw_resource->tics_hbm_dma_ops += elapsed;
        }
        job.wlhd->workload->call(EventType::General, job.wlhd);
    }

    // A completion callback may immediately issue the next job (compute,
    // restore, comm or pool), which schedules a transition and advances the
    // generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}
