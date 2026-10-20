/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieExecution.hh"

#include <algorithm>
#include <cmath>
#include <cstdio>

#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/Workload.hh"

namespace AstraSim {

DieExecution::DieExecution(
    Sys* sys, Workload* workload, DieModelAdapter* adapter, uint64_t node_id,
    WorkloadLayerHandlerData* wlhd,
    std::shared_ptr<const DieModel::DiePlan> plan)
    : sys_(sys),
      workload_(workload),
      adapter_(adapter),
      node_id_(node_id),
      wlhd_(wlhd),
      plan_(std::move(plan)),
      machine_(plan_),
      issue_tick_(Sys::boostedTick()),
      last_synced_tick_(issue_tick_),
      pending_event_() {
    if (sys_ == nullptr || workload_ == nullptr || adapter_ == nullptr ||
        wlhd_ == nullptr || !plan_->valid()) {
        throw std::runtime_error(
            "DieExecution requires Sys/Workload/adapter/handler data and a "
            "valid DiePlan");
    }
}

DieExecution::~DieExecution() {
    // Teardown safety (design doc §6.3: 成功完成、异常退出和终止均需回收阶段
    // 事件): cancel the pending cancellable event and drop any still-served
    // HBM die job (its payload is reclaimed by the model). No completion
    // callback fires from here.
    if (pending_event_.valid()) {
        static_cast<void>(sys_->cancel_event(pending_event_));
    }
    if (!finished_ && workload_ != nullptr &&
        workload_->local_hbm_bandwidth_model != nullptr) {
        workload_->local_hbm_bandwidth_model->cancel_die_jobs_of(this);
    }
    delete wlhd_;
    wlhd_ = nullptr;
}

void DieExecution::destroy_event_data(CallData* data) {
    delete static_cast<DieEventData*>(data);
}

void DieExecution::start() {
    drive();
}

void DieExecution::register_timer_or_self_driven_memory(uint64_t ns,
                                                         DieEventData* data) {
    if (ns < 1) {
        // register_event_cancellable requires delta >= 1; the machine
        // already guarantees Timer >= 1 and the self-driven service drain
        // clamps below, this is the belt-and-braces guard.
        delete data;
        throw std::runtime_error(
            "DieExecution refused to register a zero-delay event");
    }
    ++event_generation_;
    data->generation = event_generation_;
    pending_event_ = sys_->register_event_cancellable(
        this, EventType::General, data, ns, &destroy_event_data);
}

void DieExecution::drive() {
    // The "workload" logger is process-global and immutable for the
    // whole run (sinks are attached on first fetch), so fetch it once
    // instead of paying the spdlog registry lookup on every drive().
    static const auto logger = LoggerFactory::get_logger("workload");
    while (!machine_.finished()) {
        const DieModel::ExecutionAction action = machine_.next_action();
        switch (action.kind) {
            case DieModel::ExecutionAction::Kind::Timer: {
                // Stage E2 timer coalescing (design doc §8.5 "重复 wave
                // 惰性展开…只有证明区间内外部竞争集合不变时才能压缩",
                // applied to a PURE timer stretch): a run of consecutive
                // internal Timer actions is externally observable only
                // through the HBM arbiter, which this execution touches
                // exclusively at MemoryRequest boundaries (never crossed
                // here).  While the rank's HBM model reports no active job
                // at this instant -- no non-DIE user and no pending DIE
                // request, the guard prescribed for exactly this
                // compression -- the whole stretch is self-advanced through
                // the machine and delivered to Sys as ONE event: the
                // machine's internal schedule is deterministic, its clock
                // re-syncs to the same wall tick the stretch would have
                // ended on, and no memory-sharing decision can change
                // inside the synchronous loop (Sys state only changes
                // through the event loop).  Without the model
                // (hbm-bandwidth-contention=0) there is no arbiter at all
                // and the guard is trivially satisfied.
                uint64_t merged_ns = 0;
                uint64_t timer_ns = action.ns;
                while (timer_ns > 0) {
                    if (workload_->local_hbm_bandwidth_model != nullptr &&
                        workload_->local_hbm_bandwidth_model
                            ->has_active_jobs()) {
                        break;  // external HBM activity: do not compress
                    }
                    machine_.advance_time(timer_ns);
                    self_advanced_ns_ += timer_ns;
                    merged_ns += timer_ns;
                    timer_ns = 0;
                    if (!machine_.finished()) {
                        const DieModel::ExecutionAction next =
                            machine_.next_action();
                        if (next.kind ==
                            DieModel::ExecutionAction::Kind::Timer) {
                            timer_ns = next.ns;
                        }
                        // MemoryRequest/Done: stop; the merged event's
                        // callback re-enters drive(), which re-derives the
                        // parked report (actions are idempotent).
                    }
                }
                if (merged_ns == 0) {
                    if (sys_->trace_enabled) {
                        logger->debug(
                            "die_exec node={} timer ns={} machine_elapsed={}",
                            node_id_, action.ns, machine_.elapsed_ns());
                    }
                    register_timer_or_self_driven_memory(
                        std::max<uint64_t>(action.ns, 1),
                        new DieEventData(DieEventData::Kind::Timer, 0, 0));
                    return;
                }
                if (sys_->trace_enabled) {
                    logger->debug(
                        "die_exec node={} coalesced_timer ns={} "
                        "machine_elapsed={} (pure timer stretch, no active "
                        "HBM jobs)",
                        node_id_, merged_ns, machine_.elapsed_ns());
                }
                register_timer_or_self_driven_memory(
                    std::max<uint64_t>(merged_ns, 1),
                    new DieEventData(DieEventData::Kind::Timer, 0, 0));
                return;
            }
            case DieModel::ExecutionAction::Kind::MemoryRequest: {
                // One-shot first-access latency (contract §7.4): charged as
                // a TIMER before the node's FIRST memory request, at most
                // once per outer node, never inside the HBM job. The
                // outstanding request re-reports idempotently when the
                // latency timer fires.
                if (!latency_charged_ && sys_->local_mem_latency > 0) {
                    latency_charged_ = true;
                    latency_timer_pending_ = true;
                    if (sys_->trace_enabled) {
                        logger->debug(
                            "die_exec node={} one-shot local_mem_latency "
                            "ns={}",
                            node_id_, sys_->local_mem_latency);
                    }
                    register_timer_or_self_driven_memory(
                        sys_->local_mem_latency,
                        new DieEventData(DieEventData::Kind::Timer, 0, 0));
                    return;
                }
                const DieModel::DieStage& stage =
                    plan_->stages().at(action.stage_id);
                if (sys_->trace_enabled) {
                    logger->debug(
                        "die_exec node={} memory_request stage={} label={} "
                        "actual_bytes={} service_units={}",
                        node_id_, action.stage_id, stage.label,
                        action.actual_bytes, action.service_units);
                }
                if (workload_->local_hbm_bandwidth_model != nullptr) {
                    // Normal production mode (hbm-bandwidth-contention=1):
                    // the rank's single HBM arbiter serves the request at
                    // its equal share; completion arrives through the sink
                    // callback below.
                    workload_->local_hbm_bandwidth_model->issue_die_memory(
                        action.service_units, action.actual_bytes, this,
                        new DieEventData(DieEventData::Kind::Memory,
                                         action.stage_id, event_generation_));
                    return;
                }
                // Contention off (contract §7.5): the memory leg self-drains
                // at the FULL rate. 先乘后除 (stage A3 report §6) keeps the
                // per-link rounding to one ceil.
                if (!(sys_->local_mem_bw > 0.0)) {
                    throw std::runtime_error(
                        "DieExecution cannot self-drive a memory leg "
                        "without a positive local-mem-bw");
                }
                const double ns_double = std::ceil(
                    action.service_units * 1e9 / sys_->local_mem_bw);
                if (!std::isfinite(ns_double) || ns_double >= 1e18) {
                    throw std::runtime_error(
                        "DieExecution self-driven memory service exceeds "
                        "1e18 ns (check local-mem-bw and the plan's "
                        "service_units)");
                }
                if (sys_->trace_enabled) {
                    logger->debug(
                        "die_exec node={} self-driven memory stage={} ns={} "
                        "(hbm-bandwidth-contention=0: 双方争用未模拟 -- this "
                        "leg runs at the full rate, comm keeps its legacy "
                        "switch semantics)",
                        node_id_, action.stage_id,
                        static_cast<uint64_t>(ns_double));
                }
                register_timer_or_self_driven_memory(
                    std::max<uint64_t>(
                        static_cast<uint64_t>(ns_double), 1),
                    new DieEventData(DieEventData::Kind::Memory,
                                     action.stage_id, 0));
                return;
            }
            case DieModel::ExecutionAction::Kind::Done: {
                complete();
                return;
            }
        }
    }
    // Defensive: the machine reports finished() without a Done action pass
    // (e.g. an empty plan). Complete through the same exactly-once exit.
    complete();
}

void DieExecution::call(EventType, CallData* data_ptr) {
    auto* data = static_cast<DieEventData*>(data_ptr);
    const DieEventData::Kind kind = data->kind;
    const uint64_t stage_id = data->stage_id;
    const uint64_t generation = data->generation;
    delete data;

    if (finished_ || generation != event_generation_) {
        // Stale callback (a cancelled event that the backend could not
        // physically remove, or a post-completion re-fire): drop it.
        // DeadEnd-census diagnostic only (2026-10-11): count the drop.
        // Zero output on the healthy path.
        std::fprintf(stderr,
                     "[DieGenDiag] stale die callback dropped: node_id=%llu "
                     "time_ns=%llu cb_gen=%llu current_gen=%llu "
                     "finished=%d pending_event_valid=%d\n",
                     static_cast<unsigned long long>(node_id_),
                     static_cast<unsigned long long>(Sys::boostedTick()),
                     static_cast<unsigned long long>(generation),
                     static_cast<unsigned long long>(event_generation_),
                     finished_ ? 1 : 0,
                     pending_event_.valid() ? 1 : 0);
        return;
    }

    // Sync the machine's internal clock to the Sys clock (the callback's
    // tick IS the completion moment of whatever was outstanding).
    const Tick now = Sys::boostedTick();
    if (now < last_synced_tick_) {
        throw std::runtime_error("DieExecution clock moved backwards");
    }
    if (latency_timer_pending_) {
        // The one-shot local_mem_latency charge is EXTERNAL dead time
        // (contract §7.4): the machine's clock must NOT advance through
        // it -- the plan's stages are relative to the machine clock, so
        // advancing would merely shift the internal schedule and the
        // latency would cancel out of the node's completion span. Only
        // re-anchor the wall-clock sync point; the machine stays one
        // latency behind the wall for the rest of the node.
        latency_timer_pending_ = false;
        last_synced_tick_ = now;
    } else {
        // Stage E2 coalescing: self_advanced_ns_ of this wall delta was
        // already consumed synchronously by the merged timer stretch; the
        // merged event fires exactly at last_synced_tick_ +
        // self_advanced_ns_, so the remainder (zero for a merged event, the
        // full delta otherwise) is the new machine time.
        const uint64_t wall_delta =
            static_cast<uint64_t>(now - last_synced_tick_);
        machine_.advance_time(
            wall_delta >= self_advanced_ns_ ? wall_delta - self_advanced_ns_
                                            : 0);
        self_advanced_ns_ = 0;
        last_synced_tick_ = now;
    }
    pending_event_.reset();

    if (kind == DieEventData::Kind::Memory) {
        machine_.complete_memory(stage_id);
    }
    drive();
}

void DieExecution::fill_utilizations(uint64_t elapsed_ns) {
    // Contract §7.6 caliber (design doc §8.5): compute = plan FLOP /
    // (PE_peak_die * elapsed); memory = actual bytes / (B_mem * elapsed);
    // vector = eqops / (V_peak * elapsed). The peaks are the die library's
    // physical peaks, and the elapsed window is the node's actual issue ->
    // completion span (contention makes utilization fall below the isolated
    // estimate, never above the physical bound by construction).
    const DieModel::Hardware& hw = plan_->hardware();
    const double seconds =
        static_cast<double>(std::max<uint64_t>(elapsed_ns, 1)) / 1e9;
    const double compute_utilization = plan_->flops() / (hw.pe_peak_die * seconds);
    const double memory_utilization =
        (plan_->matrix_bytes() + plan_->vector_bytes()) /
        (hw.hbm_bandwidth_bytes * seconds);
    const double vector_utilization =
        plan_->eqops() / (hw.vec_peak_die * seconds);
    workload_->record_die_utilizations(node_id_, compute_utilization,
                                       memory_utilization,
                                       vector_utilization);
}

void DieExecution::dump_census_to_stderr(const char* prefix) const {
    // DeadEnd-census diagnostic only (2026-10-11): read-only dump of the
    // execution's outstanding-timer state. Never called on the hot path.
    std::fprintf(stderr,
                 "%s die_exec node_id=%llu finished=%d "
                 "pending_event_valid=%d pending_event_time_ns=%llu "
                 "generation=%llu machine_elapsed_ns=%llu "
                 "latency_timer_pending=%d\n",
                 prefix, static_cast<unsigned long long>(node_id_),
                 finished_ ? 1 : 0, pending_event_.valid() ? 1 : 0,
                 static_cast<unsigned long long>(
                     pending_event_.valid() ? pending_event_.event_time()
                                            : 0),
                 static_cast<unsigned long long>(event_generation_),
                 static_cast<unsigned long long>(machine_.elapsed_ns()),
                 latency_timer_pending_ ? 1 : 0);
}

void DieExecution::complete() {
    if (finished_) {
        throw std::runtime_error(
            "DieExecution completion fired twice for node " +
            std::to_string(node_id_));
    }
    finished_ = true;
    if (pending_event_.valid()) {
        static_cast<void>(sys_->cancel_event(pending_event_));
    }
    const Tick end_tick = Sys::boostedTick();
    const uint64_t elapsed_ns =
        end_tick >= issue_tick_ ? end_tick - issue_tick_ : 0;
    if (sys_->trace_enabled) {
        LoggerFactory::get_logger("workload")->debug(
            "die_exec node={} done machine_elapsed={} wall_elapsed={}",
            node_id_, machine_.elapsed_ns(), elapsed_ns);
    }
    // GPU occupancy (design doc §6.3) needs no extra accounting here: this
    // repository's HardwareResource tracks occupancy through the standard
    // occupy/release pair -- Workload::issue occupied the node and the
    // finish_generic_node call below releases it, so the whole issue ->
    // completion span is already the node's occupancy interval.
    fill_utilizations(elapsed_ns);
    // Exactly one generic completion. The internal HBM jobs of this node
    // never reached this path -- only the machine's Done does.
    workload_->finish_generic_node(node_id_);
    delete wlhd_;
    wlhd_ = nullptr;
}

}  // namespace AstraSim
