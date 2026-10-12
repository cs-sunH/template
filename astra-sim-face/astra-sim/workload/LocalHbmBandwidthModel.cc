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
      event_generation(0),
      scheduled_transition_event() {
    if (sys == nullptr || workload == nullptr) {
        throw std::invalid_argument("local HBM model requires Sys and Workload");
    }
    if (sys->local_mem_bw <= 0) {
        throw std::invalid_argument(
            "local HBM sharing requires a positive local-mem-bw");
    }
}

LocalHbmBandwidthModel::~LocalHbmBandwidthModel() {
    cancel_scheduled_transition();
    // (2026-09-30, R2 audit) A throw out of schedule_next_transition after a
    // job was pushed (or an early run termination) used to leak every
    // in-flight job's WorkloadLayerHandlerData -- the normal path drains all
    // jobs before destruction, so this only runs on the abnormal paths.
    for (const Job& job : jobs) {
        delete job.wlhd;
        // DIE_COMPUTE payloads have no WorkloadLayerHandlerData; the model
        // owned them from issue_die_memory until here (the sink is gone or
        // going away with the Workload -- never call it from a destructor).
        delete job.die_data;
    }
    jobs.clear();
}

bool LocalHbmBandwidthModel::has_active_jobs() const {
    return !jobs.empty();
}

double LocalHbmBandwidthModel::remaining_drain(const Job& job) {
    return job.kind == JobKind::DIE_COMPUTE ? job.remaining_service
                                            : job.remaining_bytes;
}

bool LocalHbmBandwidthModel::memory_done(const Job& job) {
    if (job.kind == JobKind::DIE_COMPUTE) {
        // No latency phase (contract §7.4: the DieExecution charges
        // local_mem_latency itself as a one-shot TIMER before the first
        // memory request); the service stream is the only memory quantity.
        return job.remaining_service <= kEpsilon;
    }
    return job.memory_latency_ns <= kEpsilon &&
        job.remaining_bytes <= kEpsilon;
}

bool LocalHbmBandwidthModel::complete(const Job& job) {
    return memory_done(job) &&
        (job.kind != JobKind::COMP || job.remaining_ops <= kEpsilon);
}

void LocalHbmBandwidthModel::note_membership_change() {
    peak_concurrent_jobs_ = std::max<uint64_t>(peak_concurrent_jobs_,
                                               jobs.size());
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        throw std::runtime_error("local HBM model time moved backwards");
    }
    const double elapsed_ns = static_cast<double>(now - last_update_tick);
    if (elapsed_ns <= 0) {
        return;
    }

    // FLOPs drain at peak rate in parallel with the (shared) HBM bytes;
    // only the byte stream is subject to the equal-share arbitration.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    if (compute_rate_ops_per_ns > 0) {
        for (Job& job : jobs) {
            if (job.kind == JobKind::COMP) {
                job.remaining_ops = std::max(
                    0.0,
                    job.remaining_ops -
                        compute_rate_ops_per_ns * elapsed_ns);
            }
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
            } else if (remaining_drain(job) > kEpsilon) {
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
                    step_ns, remaining_drain(*job) / per_user_rate);
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
                if (job.remaining_service > 0 &&
                    job.remaining_service <= kEpsilon) {
                    job.remaining_service = 0;
                    clamped_residue = true;
                }
            }
            if (per_user_rate > 0) {
                for (Job* job : bandwidth_users) {
                    if (remaining_drain(*job) / per_user_rate <= kEpsilon) {
                        if (job->kind == JobKind::DIE_COMPUTE) {
                            job->remaining_service = 0;
                        } else {
                            job->remaining_bytes = 0;
                        }
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
            const double served = per_user_rate * step_ns;
            if (job->kind == JobKind::DIE_COMPUTE) {
                // Contract §7.4: service units are drained; the actual-byte
                // advance is served * eta and is bookkeeping only.
                job->remaining_service =
                    std::max(0.0, job->remaining_service - served);
            } else {
                job->remaining_bytes =
                    std::max(0.0, job->remaining_bytes - served);
            }
        }
        // Read-only side-band accumulation: reuse the step_ns and
        // per_user_rate already computed above.  These counters must not
        // change any value used for the next transition.
        if (!bandwidth_users.empty()) {
            this->hbm_busy_ns_ += step_ns;
            for (const Job* job : bandwidth_users) {
                switch (job->kind) {
                case JobKind::COMP:
                    this->compute_bytes_served_ += per_user_rate * step_ns;
                    break;
                case JobKind::COMM_READ:
                    this->comm_read_bytes_served_ +=
                        per_user_rate * step_ns;
                    break;
                case JobKind::COMM_WRITE:
                    this->comm_write_bytes_served_ +=
                        per_user_rate * step_ns;
                    break;
                case JobKind::DIE_COMPUTE:
                    this->die_service_units_served_ +=
                        per_user_rate * step_ns;
                    this->die_actual_bytes_served_ +=
                        per_user_rate * step_ns * job->service_eta;
                    break;
                }
            }
        }
        remaining_ns -= step_ns;
    }
    last_update_tick = now;
}

void LocalHbmBandwidthModel::harvest_completed_jobs(Tick now) {
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
        // At least one survivor remains: its share immediately widens from
        // full_rate/N to full_rate/(N-k) -- one redistribution event.
        ++redistribution_events_;
    }

    for (const Job& job : completed) {
        const uint64_t elapsed = now - job.start_tick;
        if (job.kind == JobKind::COMP) {
            workload->hw_resource->tics_gpu_ops += elapsed;
        }
        if (job.kind == JobKind::DIE_COMPUTE) {
            // Contract §7.4/§6.3: the sink is the DieExecution (an internal
            // memory stage of a still-running outer node -- NOT an outer
            // COMP completion). The DieExecution accounts the outer node's
            // GPU occupancy itself at its own completion; this callback only
            // hands back the memory port.
            CallData* const data = job.die_data;
            job.die_sink->call(EventType::General, data);
            continue;
        }
        workload->on_local_hbm_job_complete(job.wlhd, job.kind);
    }
}

void LocalHbmBandwidthModel::cancel_scheduled_transition() {
    if (scheduled_transition_event.valid()) {
        static_cast<void>(sys->cancel_event(scheduled_transition_event));
    }
    scheduled_transition_event.reset();
}

void LocalHbmBandwidthModel::destroy_transition_data(CallData* data) {
    delete static_cast<TransitionData*>(data);
}

void LocalHbmBandwidthModel::schedule_next_transition() {
    // Every reallocation replaces the old prediction. The old callback would
    // only hit its generation guard, so delete its payload/list node now.
    cancel_scheduled_transition();
    ++event_generation;
    if (!has_active_jobs()) {
        return;
    }

    std::vector<const Job*> bandwidth_users;
    double next_ns = std::numeric_limits<double>::infinity();
    for (const Job& job : jobs) {
        if (job.memory_latency_ns > kEpsilon) {
            next_ns = std::min(next_ns, job.memory_latency_ns);
        } else if (remaining_drain(job) > kEpsilon) {
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
            next_ns = std::min(next_ns, remaining_drain(*job) / per_user_rate);
        }
    }

    // A COMP job whose byte stream is already done but whose FLOPs are not
    // still constrains the next transition through the (unshared) compute
    // drain.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (const Job& job : jobs) {
        if (job.kind == JobKind::COMP && memory_done(job) &&
            job.remaining_ops > kEpsilon) {
            if (compute_rate_ops_per_ns <= 0) {
                throw std::runtime_error(
                    "local HBM model COMP job cannot drain without peak-perf");
            }
            next_ns = std::min(
                next_ns, job.remaining_ops / compute_rate_ops_per_ns);
        }
    }

    if (!std::isfinite(next_ns)) {
        throw std::runtime_error("local HBM model has no schedulable transition");
    }
    // Upper bound before the uint64 cast (2026-09-30): a tiny-but-positive
    // local-mem-bw gives per_user_rate ~ 0 and a finite-but-astronomical
    // next_ns; casting that to Tick is UB. Mirror of the R5-Q5 guard on the
    // closed-form COMP path in Workload (1e18 ns ~= 31 years of sim time).
    if (next_ns >= 1e18) {
        throw std::runtime_error(
            "local HBM model transition delay exceeds 1e18 ns "
            "(check local-mem-bw / job sizes)");
    }
    Tick delay = static_cast<Tick>(std::ceil(next_ns));
    delay = std::max<Tick>(1, delay);
    scheduled_transition_event = sys->register_event_cancellable(
        this,
        EventType::General,
        new TransitionData(event_generation),
        delay,
        destroy_transition_data);
}

void LocalHbmBandwidthModel::issue_job(JobKind kind,
                                       uint64_t num_ops,
                                       uint64_t bytes,
                                       WorkloadLayerHandlerData* wlhd) {
    const Tick now = Sys::boostedTick();
    advance_to(now);
    // The advance above can drive an already-active job to completion at
    // this very tick -- the tick its still-pending transition event was
    // scheduled for, which the schedule_next_transition() below cancels
    // before that event can fire. Harvest those completions here with the
    // same predicate and callbacks as the transition dispatch: a drained
    // job left in `jobs` would defer its completion to a later transition,
    // inflating tics_gpu_ops and peak_concurrent_jobs and delaying the
    // node past the physical bandwidth-exhaustion tick.
    harvest_completed_jobs(now);
    if (bytes == 0) {
        // Zero-byte endpoints never create an HBM job (the Workload layer
        // already guards this); reaching here is a wiring bug, not a data
        // property: fail closed instead of silently stalling the node.
        throw std::runtime_error(
            "local HBM model refused a zero-byte job");
    }
    if (kind == JobKind::COMP && sys->peak_perf <= 0) {
        throw std::runtime_error(
            "local HBM model COMP job requires a positive peak-perf");
    }
    if (!jobs.empty()) {
        // Joining a non-empty set reshapes everyone's equal share.
        ++redistribution_events_;
    }
    jobs.push_back(Job{
        kind,
        wlhd,
        static_cast<double>(num_ops),
        static_cast<double>(bytes),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
    note_membership_change();
    schedule_next_transition();
}

void LocalHbmBandwidthModel::issue_compute(
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMP, num_ops, tensor_size, wlhd);
}

void LocalHbmBandwidthModel::issue_comm_read(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMM_READ, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_comm_write(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMM_WRITE, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_die_memory(double service_units,
                                              double actual_bytes,
                                              Callable* sink,
                                              CallData* data) {
    // Contract §7.4: no latency phase, no FLOP leg; the drained quantity is
    // service_units. eta is derived from the served amounts
    // (actual/service) so the side-band actual-byte counter matches the
    // request triple the DieExecution received from the plan.
    if (!(service_units > 0.0) || !(actual_bytes >= 0.0) ||
        sink == nullptr || data == nullptr) {
        delete data;
        throw std::runtime_error(
            "local HBM model refused a malformed die memory request");
    }
    if (actual_bytes > 0.0 && service_units <= actual_bytes) {
        // eta <= 1 would mean the access is at least as efficient as an
        // ideal eta=1 stream -- the die calibration etas are all < 1.
        delete data;
        throw std::runtime_error(
            "local HBM model refused a die memory request with eta >= 1 "
            "(service_units must exceed actual_bytes)");
    }
    const double eta = actual_bytes / service_units;

    const Tick now = Sys::boostedTick();
    advance_to(now);
    harvest_completed_jobs(now);
    if (!jobs.empty()) {
        // Joining a non-empty set reshapes everyone's equal share.
        ++redistribution_events_;
    }
    Job job{};
    job.kind = JobKind::DIE_COMPUTE;
    job.wlhd = nullptr;
    job.remaining_ops = 0.0;
    job.remaining_bytes = 0.0;
    job.memory_latency_ns = 0.0;
    job.start_tick = now;
    job.remaining_service = service_units;
    job.service_eta = eta;
    job.die_sink = sink;
    job.die_data = data;
    jobs.push_back(std::move(job));
    note_membership_change();
    schedule_next_transition();
}

void LocalHbmBandwidthModel::cancel_die_jobs_of(Callable* sink) {
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (it->kind == JobKind::DIE_COMPUTE && it->die_sink == sink) {
            delete it->die_data;
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
    // A wiped job may have been the next transition's reason; re-arm the
    // prediction for whatever remains (a no-op when nothing is left).
    schedule_next_transition();
}

void LocalHbmBandwidthModel::call(EventType, CallData* data) {
    TransitionData* transition = static_cast<TransitionData*>(data);
    const uint64_t generation = transition->generation;
    delete transition;
    if (generation != event_generation) {
        return;
    }
    // Sys popped this callback before invoking us, so it is no longer
    // cancellable. Forget the consumed handle before a completion callback can
    // install the next prediction. A defensive stale-generation callback must
    // leave a newer handle untouched.
    scheduled_transition_event.reset();

    const Tick now = Sys::boostedTick();
    advance_to(now);
    harvest_completed_jobs(now);

    // A completion callback may immediately issue the next job (e.g. the
    // freed comm slot lets a queued SEND issue inside
    // issue_dep_free_nodes), which schedules a transition and advances the
    // generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}
