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
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"

using namespace AstraSim;

namespace {
constexpr double kEpsilon = 1e-9;

std::pair<double, double> coarse_hbm_read_write_bytes(
    const nlohmann::json& progress) {
    const nlohmann::json matrix = progress.value(
        "hbm_served_bytes_exact",
        progress.value("hbm_served_bytes", nlohmann::json::array()));
    if (!matrix.is_array() || matrix.size() != 2) {
        throw std::runtime_error(
            "coarse_die_v2 progress is missing the 2x2 HBM served matrix");
    }
    long double read_bytes = 0.0L;
    long double write_bytes = 0.0L;
    for (const auto& row : matrix) {
        if (!row.is_array() || row.size() != 2) {
            throw std::runtime_error("coarse_die_v2 HBM served row is malformed");
        }
        for (size_t direction = 0; direction < 2; ++direction) {
            const auto& value = row[direction];
            long double amount = 0.0L;
            if (value.is_string()) {
                const std::string exact = value.get<std::string>();
                size_t consumed = 0;
                amount = std::stold(exact, &consumed);
                if (consumed != exact.size()) {
                    throw std::runtime_error("invalid exact HBM served byte");
                }
            } else if (value.is_number()) {
                amount = value.get<long double>();
            } else {
                throw std::runtime_error("invalid HBM served byte type");
            }
            if (!std::isfinite(amount) || amount < 0.0L) {
                throw std::runtime_error("invalid HBM served byte value");
            }
            (direction == 0 ? read_bytes : write_bytes) += amount;
        }
    }
    return {static_cast<double>(read_bytes), static_cast<double>(write_bytes)};
}

uint64_t coarse_endpoint_identity(const std::string& value) {
    uint64_t hash = 14695981039346656037ull;
    for (unsigned char byte : value) {
        hash ^= byte;
        hash *= 1099511628211ull;
    }
    return hash == 0 ? 1 : hash;
}

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
    if (sys->local_mem_bw <= 0 && sys->coarse_die_runtime == nullptr) {
        throw std::invalid_argument(
            "local HBM sharing requires a positive local-mem-bw");
    }
}

double LocalHbmBandwidthModel::compute_bytes_served() const {
    double served = compute_bytes_served_;
    if (sys == nullptr || sys->coarse_die_runtime == nullptr) {
        return served;
    }

    const uint64_t now_ns = Sys::boostedTick();
    for (const auto& entry : coarse_jobs_) {
        if (entry.second.kind != JobKind::CONTROL_MARKER) continue;
        const auto progress =
            sys->coarse_die_runtime->query_progress_at(entry.first, now_ns);
        const auto [read_bytes, ignored_write_bytes] =
            coarse_hbm_read_write_bytes(progress);
        (void)ignored_write_bytes;
        served += read_bytes;
    }
    return served;
}

LocalHbmBandwidthModel::~LocalHbmBandwidthModel() {
    cancel_scheduled_transition();
    if (sys != nullptr && sys->coarse_die_runtime != nullptr) {
        while (!coarse_jobs_.empty()) {
            const uint64_t handle = coarse_jobs_.begin()->first;
            const CoarseJob job = coarse_jobs_.begin()->second;
            if (job.kind == JobKind::CONTROL_MARKER) {
                // Sample the physical prefix without dispatching any Runtime
                // completion callback. At a terminal tick, callbacks for
                // this and other executions may already be ready, so notify
                // would complete graph nodes while their HBM owner is dying.
                const uint64_t now_ns = Sys::boostedTick();
                const auto progress =
                    sys->coarse_die_runtime->query_progress_at(handle, now_ns);
                const auto [read_bytes, ignored_write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                (void)ignored_write_bytes;
                compute_bytes_served_ += read_bytes;
                coarse_jobs_.erase(handle);
                sys->coarse_die_runtime->cancel(handle, now_ns, false);
                if (job.wlhd != nullptr) {
                    workload->abandon_local_hbm_join_entry(job.wlhd->node_id);
                    delete job.wlhd;
                }
            } else {
                sys->coarse_die_runtime->cancel(handle, Sys::boostedTick(),
                                                 false);
                if (job.wlhd != nullptr) {
                    workload->abandon_local_hbm_join_entry(job.wlhd->node_id);
                    delete job.wlhd;
                }
                coarse_jobs_.erase(handle);
            }
        }
    }
    // Teardown drain: normally every job leaves `jobs` only through
    // workload->on_local_hbm_job_complete (which deletes the wlhd and
    // completes the node), so a healthy run end finds the vector empty. A
    // fatal exit or a test harness dismantling Sys mid-flight strands jobs
    // here; their completion callbacks must NOT be invoked from the
    // destructor (the Workload completion path can re-enter the dying model
    // and the event loop is already gone). Ownership terminates here
    // instead: delete each residual payload and drop the pending
    // HbmCommJoin entry of its joined comm node (that node can never
    // complete through a join whose endpoint job will never fire). No
    // simulation or metric effect -- only the abnormal teardown path is
    // sanitizer-clean.
    for (const Job& job : jobs) {
        if (job.wlhd != nullptr) {
            workload->abandon_local_hbm_join_entry(job.wlhd->node_id);
            delete job.wlhd;
        }
        // DIE_COMPUTE payloads have no WorkloadLayerHandlerData; the model
        // owned them from issue_die_memory until here (the sink is gone or
        // going away with the Workload -- never call it from a destructor).
        delete job.die_data;
    }
    jobs.clear();
}

bool LocalHbmBandwidthModel::has_active_jobs() const {
    return !jobs.empty() || !coarse_jobs_.empty();
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

bool LocalHbmBandwidthModel::has_bandwidth_user(
    const std::vector<Job>& jobs) {
    // A job in its latency phase (memory_latency_ns > 0) or a fully served
    // zombie (remaining_bytes == 0, not yet detached) contributes no
    // bandwidth demand: an equal-share recomputation over such a set
    // reshapes nobody's share and must not count as a redistribution.
    for (const Job& job : jobs) {
        if (remaining_drain(job) > kEpsilon) {
            return true;
        }
    }
    return false;
}

std::vector<LocalHbmBandwidthModel::Job>
LocalHbmBandwidthModel::detach_completed_jobs() {
    // 4.2 zombie fix (2026-09-29): jobs fully served by advance_to() but
    // not yet detached (their transition event still queued later in the
    // same tick) used to stay in `jobs` while a same-tick issue_job ran --
    // inflating peak_concurrent_jobs_ and counting a phantom join against
    // redistribution_events_. Both counters are live-membership metrics;
    // detach first, account after.
    std::vector<Job> completed;
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (complete(*it)) {
            completed.push_back(std::move(*it));
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
    return completed;
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
                // Contract §7.4: service units are drained; the
                // actual-byte advance is served * eta and is bookkeeping
                // only.
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
                case JobKind::CONTROL_MARKER:
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
    if (sys->coarse_die_runtime != nullptr) {
        if (kind == JobKind::COMP || kind == JobKind::DIE_COMPUTE) {
            throw std::runtime_error(
                "coarse_die_v2 COMP traffic must be owned by a typed plan, "
                "not the legacy scalar HBM queue");
        }
        if (kind != JobKind::COMM_READ && kind != JobKind::COMM_WRITE &&
            kind != JobKind::CONTROL_MARKER) {
            throw std::runtime_error(
                "coarse_die_v2 local endpoint has an unsupported job kind");
        }
        if (bytes == 0 || wlhd == nullptr) {
            throw std::runtime_error(
                "coarse_die_v2 local endpoint requires nonzero bytes and "
                "handler data");
        }
        const bool control_marker = kind == JobKind::CONTROL_MARKER;
        const bool read = kind == JobKind::COMM_READ || control_marker;
        const std::string identity = std::to_string(sys->id) + ":" +
            std::to_string(wlhd->node_id) + ":" + std::to_string(static_cast<int>(kind));
        coarse_die::Owner owner;
        owner.request_id = coarse_endpoint_identity(std::to_string(wlhd->node_id));
        owner.node_id = wlhd->node_id;
        owner.execution_id = wlhd->node_id;
        owner.generation = 1;
        const auto& performance =
            sys->coarse_die_runtime->hardware().input.at("performance_profile");
        nlohmann::json endpoint = {
            {"endpoint_id", coarse_endpoint_identity(identity)},
            {"transaction_id", coarse_endpoint_identity(std::to_string(wlhd->node_id))},
            {"owner_kind", control_marker ? "die" : "comm"},
            {"direction", read ? "read" : "write"},
            {"physical_bytes", bytes},
            {"path_class", read ? "hbm_to_edge" : "edge_to_hbm"},
            {"eta_hbm", performance.at("eta_hbm_matrix")},
            {"route", {{"internal_route", false}}},
        };
        if (control_marker) {
            endpoint["initial_delay_ns"] = sys->local_mem_latency;
        }
        const uint64_t handle = sys->coarse_die_runtime->submit_endpoint(
            endpoint, owner,
            [this, wlhd, kind](const uint64_t completed_handle,
                               const nlohmann::json& progress) {
                coarse_jobs_.erase(completed_handle);
                const bool cancelled =
                    progress.value("state", std::string()) == "cancelled";
                if (kind == JobKind::CONTROL_MARKER) {
                    const auto [read_bytes, ignored_write_bytes] =
                        coarse_hbm_read_write_bytes(progress);
                    (void)ignored_write_bytes;
                    // Keep the old COMP byte side-band equal to the physical
                    // marker prefix served by this exact Runtime execution,
                    // including a cancelled prefix. Typed ops remain zero.
                    compute_bytes_served_ += read_bytes;
                }
                if (cancelled) {
                    if (kind == JobKind::CONTROL_MARKER) {
                        workload->abandon_local_hbm_join_entry(wlhd->node_id);
                        delete wlhd;
                    }
                    return;
                }
                const auto [read_bytes, write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                if (kind == JobKind::COMM_READ) coarse_comm_read_served_ += read_bytes;
                else if (kind == JobKind::COMM_WRITE) {
                    coarse_comm_write_served_ += write_bytes;
                }
                workload->on_local_hbm_job_complete(wlhd, kind);
            });
        if (!coarse_jobs_.emplace(handle, CoarseJob{wlhd, kind}).second) {
            sys->coarse_die_runtime->cancel(handle, Sys::boostedTick(), false);
            throw std::runtime_error("coarse_die_v2 endpoint handle collision");
        }
        return;
    }
    advance_to(Sys::boostedTick());
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
    {
        // Same-tick zombies out first (see detach_completed_jobs); their
        // completion callbacks fire here -- same tick, before this new
        // job's membership is accounted.
        const std::vector<Job> zombies = detach_completed_jobs();
        for (const Job& job : zombies) {
            deliver_job_completion(job);
        }
    }
    // Definition-aligned counting (MetricCollector local_hbm record): a
    // join counts only when at least one remaining LIVE bandwidth user
    // exists whose share the join reshapes.
    if (!jobs.empty() && has_bandwidth_user(jobs)) {
        ++redistribution_events_;
    }
    jobs.push_back(Job{
        kind,
        wlhd,
        static_cast<double>(num_ops),
        static_cast<double>(bytes),
        static_cast<double>(sys->local_mem_latency),
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

void LocalHbmBandwidthModel::issue_control_marker(
    WorkloadLayerHandlerData* wlhd) {
    if (sys->coarse_die_runtime == nullptr) {
        issue_compute(1, 1, wlhd);
        return;
    }
    issue_job(JobKind::CONTROL_MARKER, 0, 1, wlhd);
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

void LocalHbmBandwidthModel::deliver_job_completion(const Job& job) {
    // Single completion dispatcher for every detached job (issue_job's
    // same-tick zombie path and the transition callback): legacy kinds
    // route through the Workload's join-aware completion; DIE_COMPUTE jobs
    // hand the memory port straight back to the owning DieExecution
    // (contract §7.4/§6.3 -- an internal memory stage is NOT an outer COMP
    // completion; the DieExecution accounts the outer node's GPU occupancy
    // itself at its own completion).
    if (job.kind == JobKind::DIE_COMPUTE) {
        CallData* const data = job.die_data;
        job.die_sink->call(EventType::General, data);
        return;
    }
    workload->on_local_hbm_job_complete(job.wlhd, job.kind);
}

void LocalHbmBandwidthModel::issue_die_memory(double service_units,
                                              double actual_bytes,
                                              Callable* sink,
                                              CallData* data) {
    // Contract §7.4: no latency phase, no FLOP leg; the drained quantity
    // is service_units. eta is derived from the served amounts
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

    advance_to(Sys::boostedTick());
    {
        // Same-tick zombies out first (same discipline as issue_job);
        // DIE completions deliver through the sink, legacy through the
        // Workload.
        const std::vector<Job> zombies = detach_completed_jobs();
        for (const Job& job : zombies) {
            deliver_job_completion(job);
        }
    }
    // Definition-aligned counting, same gate as issue_job: a join counts
    // only when at least one remaining LIVE bandwidth user exists whose
    // share the join reshapes.
    if (!jobs.empty() && has_bandwidth_user(jobs)) {
        ++redistribution_events_;
    }
    Job job{};
    job.kind = JobKind::DIE_COMPUTE;
    job.wlhd = nullptr;
    job.remaining_ops = 0.0;
    job.remaining_bytes = 0.0;
    job.memory_latency_ns = 0.0;
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
    const std::vector<Job> completed = detach_completed_jobs();
    if (!completed.empty() && has_bandwidth_user(jobs)) {
        // At least one live bandwidth-user survivor remains: its share
        // immediately widens from full_rate/N to full_rate/(N-k) -- one
        // redistribution event (definition: "at least one remaining
        // bandwidth user").
        ++redistribution_events_;
    }

    for (const Job& job : completed) {
        deliver_job_completion(job);
    }

    // A completion callback may immediately issue the next job (e.g. the
    // freed comm slot lets a queued SEND issue inside
    // issue_dep_free_nodes), which schedules a transition and advances the
    // generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}
