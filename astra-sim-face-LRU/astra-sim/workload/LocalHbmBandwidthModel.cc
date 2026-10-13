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

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/CoarseDieRuntime.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"

#include <cstdlib>

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
        (sys->local_mem_bw <= 0 || sys->peak_perf <= 0)) {
        throw std::invalid_argument(
            "local HBM sharing requires positive bandwidth and peak compute");
    }
}

LocalHbmBandwidthModel::~LocalHbmBandwidthModel() {
    cancel_scheduled_transition();
    if (sys != nullptr && sys->coarse_die_runtime != nullptr) {
        for (const auto& entry : coarse_jobs_) {
            sys->coarse_die_runtime->cancel(entry.first,
                                             Sys::boostedTick(), false);
            if (entry.second != nullptr) {
                delete entry.second;
            }
        }
        coarse_jobs_.clear();
    }

    // Abnormal teardown: reclaim what the model still owns. DIE_COMPUTE
    // payloads have no WorkloadLayerHandlerData; the model owned them from
    // issue_die_memory until here (the sink is gone or going away with the
    // Workload -- never call it from a destructor). The legacy wlhd carriers
    // are freed the same way this destructor always did.
    for (const Job& job : jobs) {
        if (job.die_data != nullptr) {
            delete job.die_data;
        } else {
            delete job.wlhd;
        }
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
    return memory_done(job) && job.remaining_ops <= kEpsilon;
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        // Invariant breach on a path whose callers run under
        // Sys::call_events (which swallows std::exception): fail closed
        // with exit, never throw (see issue_job).
        LoggerFactory::get_logger("workload")
            ->critical("local HBM model time moved backwards (sys.id={})",
                       sys->id);
        std::exit(EXIT_FAILURE);
    }
    const double elapsed_ns = static_cast<double>(now - last_update_tick);
    if (elapsed_ns <= 0) {
        return;
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
                LoggerFactory::get_logger("workload")
                    ->critical("local HBM model made no progress at a "
                               "transition (sys.id={})",
                               sys->id);
                std::exit(EXIT_FAILURE);
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
                    // Contract §7.4/§6.2: die traffic is accounted in
                    // service units (drained) and service*eta (actual),
                    // NEVER mixed into the byte-domain per-kind array.
                    this->die_service_units_served_ +=
                        per_user_rate * step_ns;
                    this->die_actual_bytes_served_ +=
                        per_user_rate * step_ns * job->service_eta;
                } else {
                    this->bytes_served_by_kind_[kind_index(job->kind)] +=
                        per_user_rate * step_ns;
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

    // Invariant: every job in `jobs` must always contribute a finite
    // deadline here. A job that advance_to() has already fully served but
    // that only call() harvests (completed-awaiting-harvest) contributes
    // the 0 deadline, so the max(1,...) clamp below arms its completion
    // delivery at the earliest possible Tick. Skipping it would combine
    // fatally with cancel_scheduled_transition() above: the cancelled
    // prediction was exactly that job's (same-tick) delivery, so its
    // completion would be pushed behind the next job's whole service time
    // -- the node keeps its HW slot, its HBM endpoint join stays shut, and
    // a delayed COMPUTE harvest inflates tics_gpu_ops. Same shape as the
    // remote backend's FluidCompleteAwaitingCallback deadline
    // (callback_tick) with the clamp-at-1 rule there: Tick 1 delivers the
    // job at the first observable Tick after this one.
    for (const Job& job : jobs) {
        if (complete(job)) {
            next_ns = 0;
            break;
        }
    }

    if (!std::isfinite(next_ns)) {
        LoggerFactory::get_logger("workload")
            ->critical("local HBM model has no schedulable transition "
                       "(sys.id={})",
                       sys->id);
        std::exit(EXIT_FAILURE);
    }
    // Upper bound before the Tick cast (fix (2026-10-01), ported from the
    // face repo): a tiny-but-positive local-mem-bw gives per_user_rate ~ 0
    // and a finite-but-astronomical next_ns; casting that to Tick is UB.
    // Mirror of the guard on the closed-form compute path in Workload
    // (1e18 ns ~= 31 years of sim time).
    if (next_ns >= 1e18) {
        LoggerFactory::get_logger("workload")
            ->critical("local HBM model transition delay exceeds 1e18 ns "
                       "(sys.id={}, check local-mem-bw / job sizes)",
                       sys->id);
        std::exit(EXIT_FAILURE);
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

void LocalHbmBandwidthModel::issue_job(
    JobKind kind,
    uint64_t num_ops,
    uint64_t tensor_size,
    WorkloadLayerHandlerData* wlhd) {
    if (sys->coarse_die_runtime != nullptr) {
        if (kind == JobKind::COMPUTE || kind == JobKind::DIE_COMPUTE) {
            throw std::runtime_error(
                "coarse_die_v2 COMP traffic must be owned by a typed plan, "
                "not the legacy scalar HBM queue");
        }
        if (tensor_size == 0 || wlhd == nullptr) {
            throw std::runtime_error(
                "coarse_die_v2 local endpoint requires nonzero bytes and "
                "handler data");
        }
        const bool read = kind == JobKind::COMM_READ || kind == JobKind::POOL_READ;
        const bool is_comm = kind == JobKind::COMM_READ || kind == JobKind::COMM_WRITE;
        const bool is_pool = kind == JobKind::POOL_READ || kind == JobKind::POOL_WRITE;
        const uint64_t node_id = wlhd->node_id;
        const std::string identity = std::to_string(sys->id) + ":" +
            std::to_string(node_id) + ":" + std::to_string(static_cast<int>(kind));
        coarse_die::Owner owner;
        owner.request_id = coarse_endpoint_identity(std::to_string(node_id));
        owner.node_id = node_id;
        owner.execution_id = node_id;
        owner.generation = 1;
        const auto& performance =
            sys->coarse_die_runtime->hardware().input.at("performance_profile");
        const uint64_t endpoint_id = coarse_endpoint_identity(identity);
        const uint64_t transaction_id = coarse_endpoint_identity(std::to_string(node_id));
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
            [this, wlhd, kind](const uint64_t completed_handle,
                               const nlohmann::json& progress) {
                coarse_jobs_.erase(completed_handle);
                if (progress.value("state", std::string()) == "cancelled") return;
                const auto [read_bytes, write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                if (kind == JobKind::COMM_READ) coarse_comm_read_served_ += read_bytes;
                else if (kind == JobKind::COMM_WRITE) coarse_comm_write_served_ += write_bytes;
                else if (kind == JobKind::RESTORE) coarse_restore_served_ += write_bytes;
                else if (kind == JobKind::POOL_READ) coarse_pool_read_served_ += read_bytes;
                else if (kind == JobKind::POOL_WRITE) coarse_pool_write_served_ += write_bytes;
                workload->call(EventType::General, wlhd);
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
        // property. Fail closed with exit instead of throw: the issuing
        // node is already occupied, and static-path callers run under
        // Sys::call_events, which swallows std::exception -- a throw here
        // would leave the node occupied forever (static_all_done never
        // fires).
        LoggerFactory::get_logger("workload")
            ->critical("local HBM model refused a zero-byte job "
                       "(sys.id={}, kind={}, bytes=0)",
                       sys->id, static_cast<int>(kind));
        std::exit(EXIT_FAILURE);
    }
    // A job still owing drain quantity (HBM bytes, or service units for a
    // DIE_COMPUTE job) is a bandwidth user (imminently, after its fixed
    // latency): a join dilutes the equal split only for those.  Jobs whose
    // HBM side already finished (pure-compute COMPUTE stragglers) use no
    // bandwidth, so joining them changes nobody's rate.
    bool joins_bandwidth_user = false;
    for (const Job& job : jobs) {
        if (remaining_drain(job) > kEpsilon) {
            joins_bandwidth_user = true;
            break;
        }
    }
    jobs.push_back(Job{
        kind,
        wlhd,
        static_cast<double>(num_ops),
        static_cast<double>(tensor_size),
        static_cast<double>(sys->local_mem_latency),
        Sys::boostedTick(),
    });
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(jobs.size());
    }
    if (joins_bandwidth_user) {
        // The new streamer immediately dilutes the equal split for the
        // remaining bandwidth users -- one redistribution per such join.
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
    // Joining a set that still has any bandwidth user reshapes everyone's
    // equal share (same predicate as issue_job; DIE jobs with pending
    // service ARE bandwidth users).
    bool survivor_bandwidth_user = false;
    for (const Job& job : jobs) {
        if (remaining_drain(job) > kEpsilon) {
            survivor_bandwidth_user = true;
            break;
        }
    }
    if (survivor_bandwidth_user) {
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
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(jobs.size());
    }
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

void LocalHbmBandwidthModel::dump_census_to_stderr(const char* prefix) const {
    // DeadEnd-census diagnostic only (2026-10-11, ported from WL):
    // read-only dump of the in-flight job list and the transition
    // prediction. Never called on the hot path.
    std::fprintf(stderr,
                 "%s hbm_jobs=%zu transition_valid=%d transition_time_ns=%llu "
                 "generation=%llu last_update_tick=%llu\n",
                 prefix, jobs.size(),
                 scheduled_transition_event.valid() ? 1 : 0,
                 static_cast<unsigned long long>(
                     scheduled_transition_event.valid()
                         ? scheduled_transition_event.event_time()
                         : 0),
                 static_cast<unsigned long long>(event_generation),
                 static_cast<unsigned long long>(last_update_tick));
    for (size_t index = 0; index < jobs.size(); ++index) {
        const Job& job = jobs[index];
        std::fprintf(stderr,
                     "%s hbm_job[%zu] kind=%d remaining_ops=%.3f "
                     "remaining_bytes=%.3f remaining_service=%.3f "
                     "memory_latency_ns=%.3f\n",
                     prefix, index, static_cast<int>(job.kind),
                     job.remaining_ops, job.remaining_bytes,
                     job.remaining_service, job.memory_latency_ns);
    }
}

void LocalHbmBandwidthModel::call(EventType, CallData* data) {
    TransitionData* transition = static_cast<TransitionData*>(data);
    const uint64_t generation = transition->generation;
    delete transition;
    if (generation != event_generation) {
        // DeadEnd-census diagnostic only (2026-10-11, ported from WL):
        // count stale transition callbacks (superseded predictions the
        // backend could not physically remove). Zero output on the healthy
        // path.
        std::fprintf(stderr,
                     "[HbmGenDiag] stale transition callback dropped: "
                     "time_ns=%llu cb_gen=%llu current_gen=%llu "
                     "active_jobs=%zu transition_valid=%d\n",
                     static_cast<unsigned long long>(Sys::boostedTick()),
                     static_cast<unsigned long long>(generation),
                     static_cast<unsigned long long>(event_generation),
                     jobs.size(), scheduled_transition_event.valid() ? 1 : 0);
        return;
    }
    // Sys popped this callback before invoking us, so it is no longer
    // cancellable. Forget the consumed handle before a completion callback can
    // install the next prediction. A defensive stale-generation callback must
    // leave a newer handle untouched.
    scheduled_transition_event.reset();

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

    // 中-4④ unified batch-level rule (face semantics): the whole completion
    // batch leaves the survivors' shares re-split (full_rate/N ->
    // full_rate/(N-k)); the batch counts ONCE regardless of how many jobs
    // completed together (previously each completed job counted separately,
    // diverging from the other four repos).  A batch whose survivors all
    // finished their HBM bytes (pure-compute COMPUTE stragglers) changes no
    // stream's rate and does not count.  Stage B: a survivor DIE_COMPUTE
    // job still owing service units is a bandwidth user for this predicate
    // (remaining_drain), exactly like a legacy job still owing bytes.
    if (!completed.empty()) {
        bool survivor_bandwidth_user = false;
        for (const Job& job : jobs) {
            if (remaining_drain(job) > kEpsilon) {
                survivor_bandwidth_user = true;
                break;
            }
        }
        if (survivor_bandwidth_user) {
            ++redistribution_events_;
        }
    }

    for (const Job& job : completed) {
        if (job.kind == JobKind::COMPUTE) {
            workload->hw_resource->tics_gpu_ops +=
                now - job.start_tick;
        }
        if (job.kind == JobKind::DIE_COMPUTE) {
            // Contract §7.4/§6.3: the sink is the DieExecution (an internal
            // memory stage of a still-running outer node -- NOT an outer
            // COMP completion, and never a restore/pool endpoint join). The
            // DieExecution accounts the outer node's GPU occupancy itself at
            // its own completion; this callback only hands back the memory
            // port, bypassing the Workload layer entirely.
            CallData* const data = job.die_data;
            job.die_sink->call(EventType::General, data);
            continue;
        }
        job.wlhd->workload->call(EventType::General, job.wlhd);
    }

    // A completion callback may immediately issue the next job (compute,
    // restore, comm, or pool), which schedules a transition and advances
    // the generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}
