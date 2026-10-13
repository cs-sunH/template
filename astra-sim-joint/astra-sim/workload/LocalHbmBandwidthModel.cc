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
#include "astra-sim/workload/FailClosed.hh"
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
            throw std::runtime_error(
                "coarse_die_v2 HBM served matrix row is malformed");
        }
        auto as_number = [](const nlohmann::json& value) -> long double {
            if (value.is_string()) {
                const std::string exact = value.get<std::string>();
                size_t consumed = 0;
                const long double parsed = std::stold(exact, &consumed);
                if (consumed != exact.size() || !std::isfinite(parsed) ||
                    parsed < 0.0L) {
                    throw std::runtime_error(
                        "coarse_die_v2 HBM served byte string is invalid");
                }
                return parsed;
            }
            if (value.is_number()) {
                const long double parsed = value.get<long double>();
                if (!std::isfinite(parsed) || parsed < 0.0L) {
                    throw std::runtime_error(
                        "coarse_die_v2 HBM served byte value is invalid");
                }
                return parsed;
            }
            throw std::runtime_error(
                "coarse_die_v2 HBM served byte is neither number nor string");
        };
        read_bytes += as_number(row[0]);
        write_bytes += as_number(row[1]);
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
    if (sys->coarse_die_runtime == nullptr &&
        (!std::isfinite(sys->local_mem_bw) ||
         !std::isfinite(sys->peak_perf) || sys->local_mem_bw <= 0 ||
         sys->peak_perf <= 0)) {
        throw std::invalid_argument(
            "local HBM sharing requires positive finite bandwidth and peak "
            "compute");
    }
}

LocalHbmBandwidthModel::~LocalHbmBandwidthModel() {
    cancel_scheduled_transition();
    if (sys != nullptr && sys->coarse_die_runtime != nullptr) {
        for (const auto& entry : coarse_jobs_) {
            sys->coarse_die_runtime->cancel(entry.first,
                                             Sys::boostedTick(), false);
            if (entry.second != nullptr) {
                workload->                delete entry.second;
            }
        }
        coarse_jobs_.clear();
    }

    // A normal completion transfers each cookie to Workload::call(), but an
    // early shutdown can destroy the model while jobs are still queued. The
    // model owns those outstanding cookies until that callback, so release
    // them after cancelling the only event that could have delivered one.
    for (Job& job : jobs) {
        delete job.wlhd;
        job.wlhd = nullptr;
        // DIE_COMPUTE payloads have no WorkloadLayerHandlerData; the model
        // owned them from issue_die_memory until here (the sink is gone or
        // going away with the Workload -- never call it from a destructor).
        delete job.die_data;
        job.die_data = nullptr;
    }
    jobs.clear();
}

bool LocalHbmBandwidthModel::has_active_jobs() const {
    return !jobs.empty() || !coarse_jobs_.empty();
}

LocalHbmBandwidthModel::MutationObserverId
LocalHbmBandwidthModel::add_resource_mutation_observer(
    ResourceMutationObserver observer) {
    if (!observer || next_observer_id_ == 0 ||
        next_observer_id_ == std::numeric_limits<MutationObserverId>::max()) {
        throw std::invalid_argument("invalid or exhausted HBM mutation observer");
    }
    const MutationObserverId id = next_observer_id_++;
    resource_observers_.push_back(ObserverEntry{id, std::move(observer)});
    return id;
}

void LocalHbmBandwidthModel::remove_resource_mutation_observer(
    const MutationObserverId observer_id) noexcept {
    resource_observers_.erase(
        std::remove_if(resource_observers_.begin(), resource_observers_.end(),
            [observer_id](const ObserverEntry& entry) {
                return entry.id == observer_id;
            }),
        resource_observers_.end());
}

void LocalHbmBandwidthModel::notify_resource_mutation_before(const Tick now) {
    const auto observers = resource_observers_;
    for (const ObserverEntry& entry : observers) {
        entry.callback(true, now, resource_version_);
    }
}

void LocalHbmBandwidthModel::publish_resource_mutation(const Tick now) {
    if (resource_version_ == std::numeric_limits<uint64_t>::max()) {
        fail_closed("local HBM resource version overflows uint64");
    }
    ++resource_version_;
    const auto observers = resource_observers_;
    for (const ObserverEntry& entry : observers) {
        entry.callback(false, now, resource_version_);
    }
}

std::vector<size_t> LocalHbmBandwidthModel::active_streamer_membership() const {
    std::vector<size_t> active;
    active.reserve(jobs.size());
    for (size_t i = 0; i < jobs.size(); ++i) {
        const Job& job = jobs[i];
        if (job.memory_latency_ns <= kEpsilon &&
            remaining_drain(job) > kEpsilon) {
            active.push_back(i);
        }
    }
    return active;
}

double LocalHbmBandwidthModel::remaining_drain(const Job& job) {
    return job.kind == JobKind::DIE_COMPUTE ? job.remaining_service
                                            : job.remaining_bytes;
}

std::vector<LocalHbmBandwidthModel::JobSnapshotEntry>
LocalHbmBandwidthModel::job_snapshot() {
    // W1 (joint work ledger): per-job execution-truth remainder. Issue order
    // is the container order and the identity is (sys_id, node_id, kind) --
    // a node issues at most one job per HBM leg (mark_issued consumes the
    // node from the free set), so the export needs no extra keying.
    // A tick-end callback can be triggered by an unrelated event before this
    // model's own completion alarm.  Integrate that interval here so the
    // exported residual/as_of_tick is the actual decision boundary rather
    // than the last HBM transition.
    advance_to(Sys::boostedTick());
    std::vector<JobSnapshotEntry> snapshot;
    snapshot.reserve(jobs.size());
    for (const Job& job : jobs) {
        if (job.kind == JobKind::DIE_COMPUTE) {
            // Internal memory stage of a still-running outer die COMP node:
            // no wlhd, no ledger node identity -- excluded by contract (see
            // the job_snapshot note in the header).
            continue;
        }
        JobSnapshotEntry entry;
        entry.kind = static_cast<int>(job.kind);
        entry.sys_id = job.wlhd != nullptr ? job.wlhd->sys_id
                                           : job.transfer_sys_id;
        entry.node_id = job.wlhd != nullptr ? job.wlhd->node_id
                                            : job.transfer_node_id;
        entry.job_id = job.issue_seq;
        entry.transfer_sink = job.transfer_completion != nullptr;
        entry.total_ops = job.total_ops;
        entry.total_bytes = job.total_bytes;
        entry.total_latency_ns = job.total_latency_ns;
        entry.remaining_ops = job.remaining_ops;
        entry.remaining_bytes = job.remaining_bytes;
        entry.memory_latency_ns = job.memory_latency_ns;
        entry.start_tick = job.start_tick;
        snapshot.push_back(entry);
    }
    return snapshot;
}

double LocalHbmBandwidthModel::bandwidth_bytes_per_ns() const {
    return sys->local_mem_bw / 1e9;
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
    return memory_done(job) && job.remaining_ops <= kEpsilon;
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        fail_closed("local HBM model time moved backwards");
    }
    const double elapsed_ns = static_cast<double>(now - last_update_tick);
    if (elapsed_ns <= 0) {
        return;
    }
    const std::vector<size_t> members_before = active_streamer_membership();
    const bool predicted_boundary = has_next_transition_tick_ &&
        next_transition_changes_membership_ && now >= next_transition_tick_;
    if (predicted_boundary) {
        notify_resource_mutation_before(now);
    }

    // Compute FLOPs always progress at peak rate in parallel with the HBM
    // fluid (Roofline max(compute, memory) semantics), for every COMPUTE
    // job regardless of how many jobs share the bus.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (Job& job : jobs) {
        if (job.kind == JobKind::COMPUTE) {
            job.remaining_ops = std::max(
                0.0,
                job.remaining_ops -
                    compute_rate_ops_per_ns * elapsed_ns);
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    double remaining_ns = elapsed_ns;
    std::vector<Job*> bandwidth_users;
    while (remaining_ns > kEpsilon) {
        // Reuse this call-local scratch vector across fractional boundaries.
        // Clear and refill it before latency is updated so each substep keeps
        // the same captured bandwidth membership as the original interval.
        bandwidth_users.clear();
        double step_ns = remaining_ns;
        for (Job& job : jobs) {
            if (job.memory_latency_ns > kEpsilon) {
                step_ns = std::min(step_ns, job.memory_latency_ns);
            } else if (remaining_drain(job) > kEpsilon) {
                bandwidth_users.push_back(&job);
            }
        }

        // N-way strict equal split: every byte-streaming job receives
        // full_rate / N where N is the number of concurrent streamers.
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
                fail_closed("local HBM model made no progress at a transition");
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
        // Read-only side-band accumulation (doc sec.9.4): reuse the step_ns
        // and per_user_rate already computed above.  These counters must not
        // change any value used for the next transition.
        if (!bandwidth_users.empty()) {
            this->hbm_busy_ns_ += step_ns;
            if (bandwidth_users.size() > 1) {
                this->hbm_shared_ns_ += step_ns;
            }
            for (const Job* job : bandwidth_users) {
                if (job->kind == JobKind::DIE_COMPUTE) {
                    // Die side-band (kept separate from the per-kind byte
                    // array): service units drained + actual bytes advanced
                    // (= service * eta). Neither ever feeds a byte-based
                    // consumer.
                    this->die_service_units_served_ +=
                        per_user_rate * step_ns;
                    this->die_actual_bytes_served_ +=
                        per_user_rate * step_ns * job->service_eta;
                    continue;
                }
                this->bytes_served_by_kind_[kind_index(job->kind)] +=
                    per_user_rate * step_ns;
            }
        }
        remaining_ns -= step_ns;
    }
    last_update_tick = now;
    const std::vector<size_t> members_after = active_streamer_membership();
    if (members_before != members_after) {
        if (!predicted_boundary) {
            // This can occur when a read-only snapshot advances over a
            // transition whose event callback has not run yet. The model
            // still publishes the version before any completion callback;
            // callers must preserve the pending same-tick callback itself.
            notify_resource_mutation_before(now);
        }
        publish_resource_mutation(now);
    }
}

void LocalHbmBandwidthModel::cancel_scheduled_transition() {
    if (scheduled_transition_event.valid()) {
        static_cast<void>(sys->cancel_event(scheduled_transition_event));
    }
    scheduled_transition_event.reset();
    has_next_transition_tick_ = false;
    next_transition_changes_membership_ = false;
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

    size_t bandwidth_user_count = 0;
    double next_ns = std::numeric_limits<double>::infinity();
    double next_membership_ns = std::numeric_limits<double>::infinity();
    for (const Job& job : jobs) {
        if (job.memory_latency_ns > kEpsilon) {
            next_ns = std::min(next_ns, job.memory_latency_ns);
            if (remaining_drain(job) > kEpsilon) {
                next_membership_ns =
                    std::min(next_membership_ns, job.memory_latency_ns);
            }
        } else if (remaining_drain(job) > kEpsilon) {
            ++bandwidth_user_count;
        }
    }

    const double full_rate_bytes_per_ns = sys->local_mem_bw / 1e9;
    const double per_user_rate = bandwidth_user_count == 0
        ? 0.0
        : full_rate_bytes_per_ns /
            static_cast<double>(bandwidth_user_count);
    if (per_user_rate > 0) {
        // The first pass above counts eligible jobs while preserving the
        // latency/terminal checks. Rescan in original job order to apply the
        // same per-user division and minimum updates without allocating a
        // pointer vector; this remains O(number of jobs).
        for (const Job& job : jobs) {
            if (job.memory_latency_ns > kEpsilon) {
                continue;
            }
            const double drain = remaining_drain(job);
            if (!(drain > kEpsilon)) {
                continue;
            }
            const double drain_ns = drain / per_user_rate;
            next_ns = std::min(next_ns, drain_ns);
            next_membership_ns = std::min(next_membership_ns, drain_ns);
        }
    }

    // A COMPUTE job whose HBM side is already done may still be streaming
    // FLOPs; its ops completion is a transition of its own.
    const double compute_rate_ops_per_ns = sys->peak_perf / 1e9;
    for (const Job& job : jobs) {
        if (job.kind == JobKind::COMPUTE && memory_done(job) &&
            job.remaining_ops > kEpsilon) {
            next_ns = std::min(
                next_ns, job.remaining_ops / compute_rate_ops_per_ns);
        }
    }

    if (!std::isfinite(next_ns)) {
        fail_closed("local HBM model has no schedulable transition");
    }
    const double delay_ns = std::ceil(next_ns);
    if (!std::isfinite(delay_ns) ||
        delay_ns >= static_cast<double>(std::numeric_limits<Tick>::max())) {
        fail_closed("local HBM transition exceeds the representable Tick range");
    }
    Tick delay = static_cast<Tick>(delay_ns);
    delay = std::max<Tick>(1, delay);
    const Tick now = Sys::boostedTick();
    if (delay > std::numeric_limits<Tick>::max() - now) {
        fail_closed("local HBM transition tick overflows Tick");
    }
    has_next_transition_tick_ = true;
    next_transition_tick_ = now + delay;
    next_transition_changes_membership_ =
        std::isfinite(next_membership_ns) &&
        next_membership_ns <= next_ns + kEpsilon;
    scheduled_transition_event = sys->register_event_cancellable(
        this,
        EventType::General,
        new TransitionData(event_generation),
        delay,
        destroy_transition_data);
}

void LocalHbmBandwidthModel::issue_job(
    JobKind kind,
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd,
    std::function<void()> transfer_completion,
    int transfer_sys_id,
    uint64_t transfer_node_id) {
    if (sys->coarse_die_runtime != nullptr) {
        if (kind == JobKind::COMPUTE || kind == JobKind::DIE_COMPUTE) {
            throw std::runtime_error(
                "coarse_die_v2 COMP traffic must be owned by a typed plan, "
                "not the legacy scalar HBM queue");
        }
        if (tensor_size == 0) {
            throw std::runtime_error(
                "coarse_die_v2 local endpoint requires nonzero bytes");
        }
        const bool read = kind == JobKind::COMM_READ || kind == JobKind::POOL_READ;
        const bool is_comm = kind == JobKind::COMM_READ || kind == JobKind::COMM_WRITE;
        const bool is_pool = kind == JobKind::POOL_READ || kind == JobKind::POOL_WRITE;
        const int owner_sys = transfer_completion ? transfer_sys_id : sys->id;
        const uint64_t owner_node = transfer_completion ? transfer_node_id :
            (wlhd != nullptr ? wlhd->node_id : 0);
        if (owner_node == 0 && wlhd == nullptr && !transfer_completion) {
            throw std::runtime_error("coarse_die_v2 endpoint has no node owner");
        }
        const std::string identity = std::to_string(owner_sys) + ":" +
            std::to_string(owner_node) + ":" + std::to_string(static_cast<int>(kind));
        coarse_die::Owner owner;
        owner.request_id = coarse_endpoint_identity(std::to_string(owner_node));
        owner.node_id = owner_node;
        owner.execution_id = owner_node;
        owner.generation = 1;
        const auto& performance =
            sys->coarse_die_runtime->hardware().input.at("performance_profile");
        const uint64_t endpoint_id = coarse_endpoint_identity(identity);
        const uint64_t transaction_id = coarse_endpoint_identity(std::to_string(owner_node));
        nlohmann::json endpoint = {
            {"endpoint_id", endpoint_id},
            {"transaction_id", transaction_id},
            {"owner_kind", is_comm ? "comm" : (is_pool ? "pool" : "restore")},
            {"direction", read ? "read" : "write"},
            {"physical_bytes", tensor_size},
            {"path_class", read ? "hbm_to_edge" : "edge_to_hbm"},
            {"eta_hbm", performance.at("eta_hbm_matrix")},
            {"route", {{"internal_route", false}}},
        };
        const uint64_t handle = sys->coarse_die_runtime->submit_endpoint(
            endpoint, owner,
            [this, kind, wlhd, transfer_completion = std::move(transfer_completion)](
                const uint64_t completed_handle, const nlohmann::json& progress) mutable {
                coarse_jobs_.erase(completed_handle);
                if (progress.value("state", std::string()) == "cancelled") return;
                const auto [read_bytes, write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                if (kind == JobKind::COMM_READ) coarse_comm_read_served_ += read_bytes;
                else if (kind == JobKind::COMM_WRITE) coarse_comm_write_served_ += write_bytes;
                else if (kind == JobKind::RESTORE) coarse_restore_served_ += write_bytes;
                else if (kind == JobKind::POOL_READ) coarse_pool_read_served_ += read_bytes;
                else if (kind == JobKind::POOL_WRITE) coarse_pool_write_served_ += write_bytes;
                if (transfer_completion) transfer_completion();
                else if (wlhd != nullptr) workload->call(EventType::General, wlhd);
            });
        if (!coarse_jobs_.emplace(handle, wlhd).second) {
            sys->coarse_die_runtime->cancel(handle, Sys::boostedTick(), false);
            throw std::runtime_error("coarse_die_v2 endpoint handle collision");
        }
        return;
    }
    advance_to(Sys::boostedTick());
    if (tensor_size == 0) {
        // Zero-byte endpoints never create an HBM job (the Workload layer
        // already guards this); reaching here is a wiring bug, not a data
        // property: fail closed instead of silently stalling the node.
        fail_closed("local HBM model refused a zero-byte job");
    }
    {
        // Same-tick zombies out first (4.2 zombie fix, same as the wscllm
        // model): a job fully served by advance_to() but not yet detached
        // (its transition event still queued later in the same tick) used
        // to stay in `jobs` while a same-tick issue_job ran -- inflating
        // peak_concurrent_jobs_ and counting a phantom join against
        // redistribution_events_. Their completion callbacks fire here --
        // same tick, before this new job's membership is accounted.
        const std::vector<Job> zombies = detach_completed_jobs();
        for (const Job& job : zombies) {
            on_job_complete(job, Sys::boostedTick());
        }
    }
    // Definition-aligned counting: a join counts only when at least one
    // remaining LIVE bandwidth user exists whose share the join reshapes.
    const bool joins_active_jobs =
        !jobs.empty() && has_bandwidth_user(jobs);
    notify_resource_mutation_before(Sys::boostedTick());
    jobs.push_back(Job{
        kind,
        wlhd,
        static_cast<double>(num_ops),
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        static_cast<double>(num_ops),
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
        0.0,
        0.0,
        nullptr,
        nullptr,
        std::move(transfer_completion),
        transfer_sys_id,
        transfer_node_id,
        next_issue_seq_,
    });
    ++next_issue_seq_;
    publish_resource_mutation(Sys::boostedTick());
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(jobs.size());
    }
    if (joins_active_jobs) {
        // Every new streamer immediately dilutes the equal split for the
        // already-active ones -- one redistribution per joining job.
        ++redistribution_events_;
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::issue_compute(
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::COMPUTE, num_ops, tensor_size, wlhd);
}

void LocalHbmBandwidthModel::issue_restore(
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::RESTORE, 0, tensor_size, wlhd);
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

void LocalHbmBandwidthModel::issue_transfer(
    JobKind kind,
    uint64_t bytes,
    int sys_id,
    uint64_t node_id,
    std::function<void()> completion) {
    if ((kind != JobKind::COMM_READ && kind != JobKind::COMM_WRITE) ||
        !completion) {
        throw std::invalid_argument(
            "local HBM transfer requires COMM_READ/COMM_WRITE and a completion");
    }
    issue_job(kind, 0, bytes, nullptr, std::move(completion), sys_id, node_id);
}

void LocalHbmBandwidthModel::issue_pool_read(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::POOL_READ, 0, bytes, wlhd);
}

void LocalHbmBandwidthModel::issue_pool_write(
    uint64_t bytes,
    WorkloadLayerHandlerData* wlhd) {
    issue_job(JobKind::POOL_WRITE, 0, bytes, wlhd);
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
    {
        // Same-tick zombie detach as issue_job: a job fully served by
        // advance_to() but not yet detached must not inflate the membership
        // counters against this new DIE job.
        const std::vector<Job> zombies = detach_completed_jobs();
        for (const Job& job : zombies) {
            on_job_complete(job, now);
        }
    }
    const bool joins_active_jobs =
        !jobs.empty() && has_bandwidth_user(jobs);
    notify_resource_mutation_before(now);
    Job job{};
    job.kind = JobKind::DIE_COMPUTE;
    job.wlhd = nullptr;
    job.total_ops = 0.0;
    job.total_bytes = 0.0;
    job.total_latency_ns = 0.0;
    job.remaining_ops = 0.0;
    job.remaining_bytes = 0.0;
    job.memory_latency_ns = 0.0;
    job.start_tick = now;
    job.remaining_service = service_units;
    job.service_eta = eta;
    job.die_sink = sink;
    job.die_data = data;
    jobs.push_back(std::move(job));
    publish_resource_mutation(now);
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(jobs.size());
    }
    if (joins_active_jobs) {
        ++redistribution_events_;
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::cancel_die_jobs_of(Callable* sink) {
    const Tick now = Sys::boostedTick();
    // Account for the old share before removing this sink. Completed jobs
    // are detached before any user callback, matching the normal transition
    // path; callbacks for the sink being destroyed are reclaimed below.
    advance_to(now);
    const std::vector<Job> completed = detach_completed_jobs();
    const auto matching = std::find_if(jobs.begin(), jobs.end(),
        [sink](const Job& job) {
            return job.kind == JobKind::DIE_COMPUTE && job.die_sink == sink;
        });
    if (matching != jobs.end()) {
        notify_resource_mutation_before(now);
    }
    bool removed = false;
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (it->kind == JobKind::DIE_COMPUTE && it->die_sink == sink) {
            delete it->die_data;
            it->die_data = nullptr;
            it = jobs.erase(it);
            removed = true;
        } else {
            ++it;
        }
    }
    if (removed) {
        publish_resource_mutation(now);
    }
    for (const Job& job : completed) {
        if (job.kind == JobKind::DIE_COMPUTE && job.die_sink == sink) {
            delete job.die_data;
            continue;
        }
        on_job_complete(job, now);
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
    const bool has_completed = std::any_of(jobs.begin(), jobs.end(),
        [](const Job& job) { return complete(job); });
    if (has_completed) {
        notify_resource_mutation_before(now);
    }
    std::vector<Job> completed;
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (complete(*it)) {
            completed.push_back(std::move(*it));
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
    if (!completed.empty()) {
        publish_resource_mutation(now);
    }

    // 中-4④ unified batch-level rule (face semantics): the whole completion
    // batch leaves the survivors' shares re-split (full_rate/N ->
    // full_rate/(N-k)); the batch counts ONCE regardless of how many jobs
    // completed together (previously each completed job counted separately,
    // diverging from the other four repos).
    if (!completed.empty() && has_bandwidth_user(jobs)) {
        ++redistribution_events_;
    }

    for (const Job& job : completed) {
        on_job_complete(job, now);
    }

    // A completion callback may immediately issue the next job (compute,
    // restore, comm, or pool), which schedules a transition and advances
    // the generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}

bool LocalHbmBandwidthModel::has_bandwidth_user(
    const std::vector<Job>& jobs) {
    // A job in its latency phase (memory_latency_ns > 0) or a fully served
    // zombie (drained quantity == 0, not yet detached) contributes no
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
    // 4.2 zombie fix (same as the wscllm model): see issue_job.
    const bool has_completed = std::any_of(jobs.begin(), jobs.end(),
        [](const Job& job) { return complete(job); });
    if (has_completed) {
        notify_resource_mutation_before(Sys::boostedTick());
    }
    std::vector<Job> completed;
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (complete(*it)) {
            completed.push_back(std::move(*it));
            it = jobs.erase(it);
        } else {
            ++it;
        }
    }
    if (!completed.empty()) {
        publish_resource_mutation(Sys::boostedTick());
    }
    return completed;
}

void LocalHbmBandwidthModel::on_job_complete(const Job& job,
                                             Tick now) {
    const uint64_t elapsed = now - job.start_tick;
    if (job.kind == JobKind::DIE_COMPUTE) {
        // Contract §7.4/§6.3: the sink is the DieExecution (an internal
        // memory stage of a still-running outer node -- NOT an outer COMP
        // completion). The DieExecution accounts the outer node's GPU
        // occupancy itself at its own completion; this callback only hands
        // back the memory port. No tics, no Workload layer, no join.
        CallData* const data = job.die_data;
        job.die_sink->call(EventType::General, data);
        return;
    }
    if (job.transfer_completion) {
        job.transfer_completion();
        return;
    }
    // RESTORE / COMM_* / POOL_* jobs are HBM transfers and do not count
    // toward the GPU busy-tick accumulator (the old sibling
    // tics_hbm_dma_ops counter was write-only and was removed).
    if (job.kind == JobKind::COMPUTE) {
        workload->hw_resource->tics_gpu_ops += elapsed;
    }
    job.wlhd->workload->call(EventType::General, job.wlhd);
}
