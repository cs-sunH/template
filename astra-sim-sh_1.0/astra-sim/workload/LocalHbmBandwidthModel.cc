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
    if (sys->local_mem_bw <= 0) {
        throw std::invalid_argument(
            "local HBM sharing requires a positive local-mem-bw (the "
            "hbm-bandwidth-contention flag is auto-disabled for "
            "local-mem-bw <= 0)");
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
    return memory_done(job) &&
        (job.kind != JobKind::COMP || job.remaining_ops <= kEpsilon);
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        throw std::runtime_error("local HBM model time moved backwards");
    }
    const double elapsed_ns = static_cast<double>(now - last_update_tick);
    if (elapsed_ns <= 0) {
        return;
    }

    // COMP FLOPs drain at peak rate in parallel with any HBM sharing,
    // preserving Roofline's max(compute, memory) semantics.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (Job& job : jobs) {
        if (job.kind == JobKind::COMP && job.remaining_ops > kEpsilon) {
            job.remaining_ops = std::max(
                0.0, job.remaining_ops - compute_rate_ops_per_ns * elapsed_ns);
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    double remaining_ns = elapsed_ns;
    while (remaining_ns > kEpsilon) {
        std::vector<Job*> all_jobs;
        all_jobs.reserve(jobs.size());
        for (Job& job : jobs) {
            all_jobs.push_back(&job);
        }

        std::vector<Job*> bandwidth_users;
        double step_ns = remaining_ns;
        for (Job* job : all_jobs) {
            if (job->memory_latency_ns > kEpsilon) {
                step_ns = std::min(step_ns, job->memory_latency_ns);
            } else if (job->remaining_bytes > kEpsilon) {
                bandwidth_users.push_back(job);
            }
        }

        const double per_user_rate = bandwidth_users.empty()
            ? 0.0
            : full_rate_bytes_per_ns /
                static_cast<double>(bandwidth_users.size());
        if (per_user_rate > 0) {
            for (const Job* job : bandwidth_users) {
                step_ns = std::min(step_ns, job->remaining_bytes / per_user_rate);
            }
        }

        if (step_ns <= kEpsilon) {
            // A subtraction at the previous transition can leave a byte
            // residue whose service time is below the numerical time
            // tolerance, while the byte value itself is still above the
            // quantity tolerance. Breaking here would preserve that residue
            // forever and create one transition per nanosecond. Clamp every
            // quantity that can complete within the time tolerance, then
            // recompute rates without consuming measurable simulation time.
            bool clamped_residue = false;
            for (Job* job : all_jobs) {
                if (job->memory_latency_ns > 0 &&
                    job->memory_latency_ns <= kEpsilon) {
                    job->memory_latency_ns = 0;
                    clamped_residue = true;
                }
                if (job->remaining_bytes > 0 &&
                    job->remaining_bytes <= kEpsilon) {
                    job->remaining_bytes = 0;
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

        for (Job* job : all_jobs) {
            if (job->memory_latency_ns > kEpsilon) {
                job->memory_latency_ns =
                    std::max(0.0, job->memory_latency_ns - step_ns);
            }
        }
        for (Job* job : bandwidth_users) {
            job->remaining_bytes = std::max(
                0.0, job->remaining_bytes - per_user_rate * step_ns);
        }
        // Read-only side-band accumulation: reuse the step_ns and
        // per_user_rate already computed above. These counters must not
        // change any value used for the next transition.
        if (!bandwidth_users.empty()) {
            this->hbm_busy_ns_ += step_ns;
            for (const Job* job : bandwidth_users) {
                this->served_bytes_[static_cast<size_t>(job->kind)] +=
                    per_user_rate * step_ns;
            }
        }
        remaining_ns -= step_ns;
    }
    last_update_tick = now;
}

void LocalHbmBandwidthModel::schedule_next_transition() {
    ++event_generation;
    if (jobs.empty()) {
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
        if (job.kind == JobKind::COMP && memory_done(job) &&
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

void LocalHbmBandwidthModel::issue(JobKind kind,
                                   double bytes,
                                   double ops,
                                   WorkloadLayerHandlerData* wlhd) {
    advance_to(Sys::boostedTick());
    // Generalized model: any number of concurrent jobs per category is
    // allowed (the sh_2.0 two-user hard cap and its duplicate-job exception
    // are removed); N active jobs each receive full_rate / N.
    if (!jobs.empty()) {
        // Joining a non-empty set reshapes everyone's equal share.
        ++redistribution_events_;
    }
    jobs.push_back(Job{
        kind,
        wlhd,
        ops,
        bytes,
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = jobs.size();
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::issue_compute(
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue(JobKind::COMP,
          static_cast<double>(tensor_size),
          static_cast<double>(num_ops),
          wlhd);
}

void LocalHbmBandwidthModel::issue_comm_read(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue(JobKind::COMM_READ, static_cast<double>(bytes), 0.0, wlhd);
}

void LocalHbmBandwidthModel::issue_comm_write(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue(JobKind::COMM_WRITE, static_cast<double>(bytes), 0.0, wlhd);
}

void LocalHbmBandwidthModel::issue_pool_read(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue(JobKind::POOL_READ, static_cast<double>(bytes), 0.0, wlhd);
}

void LocalHbmBandwidthModel::issue_pool_write(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue(JobKind::POOL_WRITE, static_cast<double>(bytes), 0.0, wlhd);
}

double LocalHbmBandwidthModel::hbm_utilization(Tick window_ns) const {
    if (window_ns == 0) {
        return 0.0;
    }
    return hbm_busy_ns_ / static_cast<double>(window_ns);
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
            completed.push_back(*it);
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
    // 中-4④ unified batch-level rule (face semantics): a completion batch
    // that leaves at least one survivor (any membership state, not only
    // actively-draining ones) reallocates the survivors' shares
    // (full_rate / N -> full_rate / (N - k)) and counts ONCE per batch.
    // The old draining_after>0 guard differed from !jobs.empty() only when
    // every survivor was in the latency phase (rare edge, +1 by definition).
    if (!completed.empty() && !jobs.empty()) {
        ++redistribution_events_;
    }

    for (const Job& job : completed) {
        const uint64_t elapsed = now - job.start_tick;
        if (job.kind == JobKind::COMP) {
            // Same accounting as the legacy register_event path (issue-time
            // tics_gpu_ops += runtime).
            workload->hw_resource->tics_gpu_ops += elapsed;
        } else {
            // 中-4② unified rule (sh_3.0 binary classification): non-COMP
            // endpoint jobs occupy local HBM bandwidth for their elapsed
            // time; account it on tics_hbm_dma_ops.
            workload->hw_resource->tics_hbm_dma_ops += elapsed;
        }
        job.wlhd->workload->call(EventType::General, job.wlhd);
    }

    // A completion callback may immediately issue the next job, which
    // schedules a transition and advances the generation. Do not invalidate
    // that event by scheduling the same state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}
