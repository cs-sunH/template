/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/LocalHbmBandwidthModel.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>
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
        // M25: this ctor sits inside the Workload/Sys construction chain,
        // which has no try/catch -- an escaping exception kills the process
        // via std::terminate with no diagnostics. Fail closed the same way
        // the Sys/Workload config guards do (critical log + exit), matching
        // the closed-form restore path's own zero-rate guard.
        Sys::sys_panic(
            "local HBM bandwidth sharing requires positive local-mem-bw and "
            "peak-perf in the system configuration (local_mem_bw=" +
            std::to_string(sys->local_mem_bw) + ", peak_perf=" +
            std::to_string(sys->peak_perf) + ")");
    }
}

double LocalHbmBandwidthModel::compute_bytes_served() const {
    double served = this->bytes_served_by_kind_[kind_index(JobKind::COMPUTE)] +
        coarse_control_marker_bytes_served_;
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
                // Sample this handle without advancing the live event loop,
                // then retire only this execution. A callback dispatch here
                // could publish an already-finished marker (or a sibling
                // endpoint) while Workload is being torn down.
                const uint64_t now_ns = Sys::boostedTick();
                const auto progress =
                    sys->coarse_die_runtime->query_progress_at(handle, now_ns);
                const auto [read_bytes, ignored_write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                (void)ignored_write_bytes;
                coarse_control_marker_bytes_served_ += read_bytes;
                sys->coarse_die_runtime->cancel(handle, now_ns, false);
                if (job.wlhd != nullptr) {
                    workload->abandon_hbm_endpoint_join(job.wlhd->node_id);
                    delete job.wlhd;
                }
                coarse_jobs_.erase(handle);
            } else {
                sys->coarse_die_runtime->cancel(handle, Sys::boostedTick(),
                                                 false);
                if (job.wlhd != nullptr) {
                    workload->abandon_hbm_endpoint_join(job.wlhd->node_id);
                    delete job.wlhd;
                }
                coarse_jobs_.erase(handle);
            }
        }
    }

    // Teardown drain（2026-10-01 对齐）：正常路径每个作业只经
    // workload->on_local_hbm_job_complete 离开 jobs（删 wlhd 并完成节
    // 点），健康 run 末该向量为空。fatal 退出或测试 harness 中途拆除
    // Sys 会把作业滞留在此；其完成回调绝不能从析构器里调用
    // （Workload 完成路径可能重入垂死的模型、事件循环也已不在）。
    // 所有权在此终结：逐个删除残余载荷，并丢弃其Joined 通信节点挂起的
    // Hbm 端点 join 表项（该节点的 join 端点作业永远不会 fire，节点再
    // 无从经 join 完成）。零仿真/指标影响——只有异常拆卸路径需要
    // sanitizer 干净。DIE_COMPUTE 载荷无 WorkloadLayerHandlerData：由
    // 本模型自 issue_die_memory 起持有到此处（sink 已随 Workload 消亡
    // 或正在消亡——绝不从析构器回调）。
    for (const Job& job : jobs) {
        if (job.wlhd != nullptr) {
            workload->abandon_hbm_endpoint_join(job.wlhd->node_id);
            delete job.wlhd;
        }
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
    return memory_done(job) && job.remaining_ops <= kEpsilon;
}

void LocalHbmBandwidthModel::advance_to(Tick now) {
    if (now < last_update_tick) {
        // P17: this can fire from inside the transition callback. Fail
        // closed with an explicit exit instead of an exception: the panic
        // keeps the diagnosis scoped to the model, and a live-jobs hang with
        // no future transition must never be reachable.
        Sys::sys_panic("local HBM model time moved backwards");
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
        // For a DIE_COMPUTE member the streamed quantity is its remaining
        // SERVICE units (remaining_drain); the equal-split rule itself is
        // identical for every kind.
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
                // P17/low: fail closed explicitly -- a silent hang with
                // active jobs and no schedulable transition must be
                // unreachable.
                Sys::sys_panic(
                    "local HBM model made no progress at a transition");
            }
            continue;
        }

        // Latency-expiry joiners of this step (wl-metrics-C3): every job
        // whose fixed latency ends at this step boundary starts streaming
        // afterwards and really re-splits the bus. When at least one streamer
        // was on the bus during the step, each joiner dilutes it once; the
        // first streamer onto an idle bus starts no re-split.
        if (sys->local_mem_latency > kEpsilon) {
            size_t latency_joiners = 0;
            for (const Job& job : jobs) {
                if (job.memory_latency_ns > kEpsilon &&
                    job.memory_latency_ns <= step_ns + kEpsilon) {
                    latency_joiners++;
                }
            }
            if (latency_joiners > 0) {
                this->redistribution_events_ +=
                    bandwidth_users.empty()
                        ? latency_joiners - 1
                        : latency_joiners;
            }
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
                    // Stage B (contract §7.4/§6.2): the die side-band keeps
                    // service units and actual bytes in SEPARATE counters;
                    // per-kind byte metrics never observe a die job.
                    this->die_service_units_served_ += per_user_rate * step_ns;
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

    if (!std::isfinite(next_ns)) {
        // P17: panic (fail-closed) rather than an exception -- a rank with
        // active jobs must never be left without a schedulable transition.
        Sys::sys_panic("local HBM model has no schedulable transition");
    }
    // Upper bound before the Tick cast (fix 2026-10-01, ported from the face
    // repo's R5-Q5): a tiny-but-positive local-mem-bw / peak-perf gives a
    // per-user rate ~ 0 and a finite-but-astronomical next_ns; casting that
    // to Tick is UB. Mirror of the guard on the closed-form compute path in
    // Workload (1e18 ns ~= 31 years of sim time).
    if (next_ns >= 1e18) {
        Sys::sys_panic(
            "local HBM model transition delay exceeds 1e18 ns (check "
            "local-mem-bw / peak-perf and job sizes)");
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
        const bool control_marker = kind == JobKind::CONTROL_MARKER;
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
            {"owner_kind", control_marker ? "die" :
                (is_comm ? "comm" : (is_pool ? "pool" : "restore"))},
            {"direction", (read || control_marker) ? "read" : "write"},
            {"physical_bytes", tensor_size},
            {"path_class", (read || control_marker) ? "hbm_to_edge" : "edge_to_hbm"},
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
                    // Keep marker bytes visible through the legacy COMP
                    // compatibility getter without marking them as typed
                    // Compute work or communication traffic.
                    coarse_control_marker_bytes_served_ += read_bytes;
                }
                if (cancelled) {
                    if (kind == JobKind::CONTROL_MARKER && wlhd != nullptr) {
                        workload->abandon_hbm_endpoint_join(wlhd->node_id);
                        delete wlhd;
                    }
                    return;
                }
                const auto [read_bytes, write_bytes] =
                    coarse_hbm_read_write_bytes(progress);
                if (kind == JobKind::COMM_READ) coarse_comm_read_served_ += read_bytes;
                else if (kind == JobKind::COMM_WRITE) coarse_comm_write_served_ += write_bytes;
                else if (kind == JobKind::RESTORE) coarse_restore_served_ += write_bytes;
                else if (kind == JobKind::POOL_READ) coarse_pool_read_served_ += read_bytes;
                else if (kind == JobKind::POOL_WRITE) coarse_pool_write_served_ += write_bytes;
                workload->call(EventType::General, wlhd);
            });
        if (!coarse_jobs_.emplace(handle, CoarseJob{wlhd, kind}).second) {
            sys->coarse_die_runtime->cancel(handle, Sys::boostedTick(), false);
            throw std::runtime_error("coarse_die_v2 endpoint handle collision");
        }
        return;
    }
    const Tick now = Sys::boostedTick();
    advance_to(now);
    if (tensor_size == 0) {
        // Zero-byte endpoints never create an HBM job (the Workload layer
        // already guards this); reaching here is a wiring bug, not a data
        // property: fail closed instead of silently stalling the node.
        Sys::sys_panic("local HBM model refused a zero-byte job");
    }
    // M23: advance_to() can bring jobs to completion at exactly this tick
    // (their transition event may still be sitting in this tick's queue
    // behind the event that issued this job). Retire them NOW -- tics,
    // batch redistribution count, and completion callbacks -- instead of
    // letting the push below cancel that event and defer the whole
    // retirement to the next transition point.
    retire_completed_jobs();
    // A job still owing HBM bytes is a bandwidth user (imminently, after its
    // fixed latency): a join dilutes the equal split only for those.  Jobs
    // whose HBM side already finished (pure-compute COMPUTE stragglers) use
    // no bandwidth, so joining them changes nobody's rate.  A pending
    // DIE_COMPUTE job is a streamer of service units (remaining_drain).
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
    });
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(jobs.size());
    }
    if (joins_bandwidth_user && sys->local_mem_latency <= kEpsilon) {
        // A zero-latency joiner starts streaming on the very next step, so
        // the equal split really changes at issue time -- one redistribution
        // per such join.  A joiner still inside its fixed latency changes no
        // share yet; its dilution is counted at the latency-expiry
        // transition inside advance_to().
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

void LocalHbmBandwidthModel::issue_control_marker(
    WorkloadLayerHandlerData* wlhd) {
    if (sys->coarse_die_runtime == nullptr) {
        issue_compute(1, 1, wlhd);
        return;
    }
    issue_job(JobKind::CONTROL_MARKER, 0, 1, wlhd);
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

// H6 定位注记(2026-09-30 审查):POOL_WRITE 路径当前无发射方——本仓
// Python 发射端只产 hbm_access_mode=1(链 B 直连池读);保留该路径是
// sh_2.0 契约面(引擎侧 M-19 preflight 的 mode 域 {0,1}——2026-10-01 自
// {0,1,2} 收紧,协议面 mode=2 已 fail-closed、本腿仅模型层与测试夹具可达、
// hbm_busy 六类指标、joint 仓同款调用点共同引用),删除会造成家族分叉。
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
    // M23 counterpart: retire whatever advance_to completed at exactly this
    // tick before joining (issue_job's invariant; a die arrival must not
    // silently defer same-tick retirements either).
    retire_completed_jobs();
    // A die job has no latency phase, so it starts streaming on the very
    // next step: the equal split really changes at issue time when any
    // current job is still streaming (the same zero-latency-joiner rule as
    // issue_job's redistribution count).
    bool joins_bandwidth_user = false;
    for (const Job& job : jobs) {
        if (remaining_drain(job) > kEpsilon) {
            joins_bandwidth_user = true;
            break;
        }
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
    if (jobs.size() > peak_concurrent_jobs_) {
        peak_concurrent_jobs_ = static_cast<uint64_t>(jobs.size());
    }
    if (joins_bandwidth_user) {
        ++redistribution_events_;
    }
    schedule_next_transition();
}

void LocalHbmBandwidthModel::cancel_die_jobs_of(Callable* sink) {
    for (auto it = jobs.begin(); it != jobs.end();) {
        if (it->kind == JobKind::DIE_COMPUTE && it->die_sink == sink) {
            delete it->die_data;
            it->die_data = nullptr;
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
        // DeadEnd-census diagnostic only (2026-10-11): count stale
        // transition callbacks (superseded predictions the backend could
        // not physically remove). Zero output on the healthy path.
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
    retire_completed_jobs();

    // A completion callback may immediately issue the next job (compute,
    // restore, comm, or pool), which schedules a transition and advances
    // the generation.  Do not invalidate that event by scheduling the same
    // state a second time.
    if (event_generation == generation) {
        schedule_next_transition();
    }
}

void LocalHbmBandwidthModel::dump_census_to_stderr(const char* prefix) const {
    // DeadEnd-census diagnostic only (2026-10-11): read-only dump of the
    // in-flight job list and the transition prediction. Never called on the
    // hot path.
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

void LocalHbmBandwidthModel::retire_completed_jobs() {
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
    // stream's rate and does not count.
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
        if (job.kind == JobKind::DIE_COMPUTE) {
            // Contract §7.4/§6.3: the sink is the DieExecution (an internal
            // memory stage of a still-running outer node -- NOT an outer
            // COMP completion). The DieExecution accounts the outer node's
            // occupancy itself at its own completion; this callback only
            // hands back the memory port. It must never route through
            // Workload::call (no endpoint-join latch, no terminal record).
            CallData* const data = job.die_data;
            job.die_sink->call(EventType::General, data);
            continue;
        }
        job.wlhd->workload->call(EventType::General, job.wlhd);
    }
}
