/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieExecution.hh"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <limits>


#include "astra-sim/common/Logging.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/system/WorkloadLayerHandlerData.hh"
#include "astra-sim/workload/DieModelAdapter.hh"
#include "astra-sim/workload/HardwareResource.hh"
#include "astra-sim/workload/LocalHbmBandwidthModel.hh"
#include "astra-sim/workload/Workload.hh"

namespace AstraSim {
namespace {

uint64_t checked_counter_add(const uint64_t left, const uint64_t right,
                             const char* const label) {
    if (right > std::numeric_limits<uint64_t>::max() - left) {
        throw std::runtime_error(std::string("DieExecution ") + label +
                                 " counter overflows uint64");
    }
    return left + right;
}

}  // namespace

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
    if (workload_->local_hbm_bandwidth_model != nullptr) {
        hbm_observer_id_ =
            workload_->local_hbm_bandwidth_model->add_resource_mutation_observer(
                [this](const bool before_change, const Tick tick,
                       const uint64_t version) {
                    on_hbm_resource_mutation(before_change, tick, version);
                });
    }
}

nlohmann::json DieExecution::quality_json() const {
    // Precise disclosure (window mechanism retired 2026-10-16): the die
    // timer path is exact by construction, so only the event counters and
    // the (currently always-null) unsupported reason are reported.
    return nlohmann::json{
        {"logical_event_count", logical_timer_event_count_},
        {"physical_step_count", physical_timer_callback_count_},
        {"unsupported_reason", nullptr}};
}

DieExecution::~DieExecution() {
    // Teardown safety (design doc §6.3: 成功完成、异常退出和终止均需回收阶段
    // 事件): cancel the pending cancellable event and drop any still-served
    // HBM die job (its payload is reclaimed by the model). No completion
    // callback fires from here.
    if (pending_event_.valid()) {
        static_cast<void>(sys_->cancel_event(pending_event_));
    }
    if (hbm_observer_id_ != 0 && workload_ != nullptr &&
        workload_->local_hbm_bandwidth_model != nullptr) {
        workload_->local_hbm_bandwidth_model->remove_resource_mutation_observer(
            hbm_observer_id_);
        hbm_observer_id_ = 0;
    }
    if (!finished_ && workload_ != nullptr &&
        workload_->local_hbm_bandwidth_model != nullptr) {
        workload_->local_hbm_bandwidth_model->cancel_die_jobs_of(this);
    }
    delete wlhd_;
    wlhd_ = nullptr;
    // Drain the Timer-payload freelist (pooling stage A): any payload the
    // cancel above routed home is reclaimed here.  Member destruction runs
    // after this body, so the freelist is still alive at the cancel point.
    for (DieEventData* data : timer_event_data_freelist_) {
        delete data;
    }
    timer_event_data_freelist_.clear();
}

void DieExecution::destroy_event_data(CallData* data) {
    auto* payload = static_cast<DieEventData*>(data);
    if (payload == nullptr) {
        return;
    }
    if (payload->kind == DieEventData::Kind::Timer &&
        payload->owner != nullptr) {
        // Cancellation sink (pooling stage A): route the Timer payload back
        // to its owning execution's freelist through the owner
        // back-pointer.  No destructor call and no owner clear -- reuse
        // relies on acquire_event_data's whole-struct reassignment.
        payload->owner->release_event_data(payload);
        return;
    }
    delete payload;
}

DieExecution::DieEventData* DieExecution::acquire_event_data(
    const DieEventData::Kind kind, const uint64_t stage_id,
    const uint64_t generation,
    const DieEventData::TimerPurpose timer_purpose) {
    DieEventData* data = nullptr;
    if (timer_event_data_freelist_.empty()) {
        data = new DieEventData(kind, stage_id, generation, timer_purpose);
    } else {
        data = timer_event_data_freelist_.back();
        timer_event_data_freelist_.pop_back();
        data->kind = kind;
        data->stage_id = stage_id;
        data->generation = generation;
        data->timer_purpose = timer_purpose;
    }
    data->owner = this;
    return data;
}

void DieExecution::release_event_data(DieEventData* data) {
    // Freelist return; see acquire_event_data for the no-destructor,
    // no-owner-clear reuse contract.
    timer_event_data_freelist_.push_back(data);
}

void DieExecution::register_external_timer(
    const uint64_t delay_ns, const DieEventData::TimerPurpose purpose,
    const uint64_t stage_id) {
    if (delay_ns < 1) {
        throw std::runtime_error(
            "DieExecution refused to register a zero-delay external timer");
    }
    timer_arm_ = InternalTimerArm{};
    memory_latency_pending_ =
        purpose == DieEventData::TimerPurpose::MemoryLatency;
    ++event_generation_;
    pending_event_ = sys_->register_event_cancellable(
        this, EventType::General,
        acquire_event_data(DieEventData::Kind::Timer, stage_id,
                           event_generation_, purpose),
        delay_ns, &destroy_event_data);
}

void DieExecution::register_internal_timer(
    const Tick now, const uint64_t original_service_ns,
    const uint64_t external_delay_ns, const uint64_t logical_action_count) {
    if (original_service_ns == 0 || external_delay_ns == 0) {
        throw std::runtime_error(
            "DieExecution refused to arm a zero-service internal timer");
    }
    InternalTimerArm arm;
    arm.original_service_ns = original_service_ns;
    arm.external_delay_ns = external_delay_ns;
    arm.logical_action_count = logical_action_count;
    arm.external_start_tick = now;
    arm.logical_start_service_tick = machine_.elapsed_ns();
    arm.stage_generation = machine_.current_frontier_generation();
    if (machine_.partial_api_probe(original_service_ns)) {
        // Probe hit: describe_internal_slice would judge this arm
        // partial-API eligible; run the original describe path verbatim.
        const DieModel::InternalSliceInfo frontier =
            machine_.describe_internal_slice();
        arm.action_token = frontier.action_token;
        arm.partial_api_eligible = frontier.partial_service_supported &&
            frontier.action_token.valid && frontier.has_next_boundary &&
            frontier.next_boundary_ns >= frontier.now_ns &&
            frontier.next_boundary_ns - frontier.now_ns >= original_service_ns;
        if (arm.partial_api_eligible && !frontier.active_stages.empty()) {
            arm.template_position = frontier.active_stages.front().wave_index;
        }
    } else {
        // Probe miss: describe would reject eligibility; settle takes the
        // plain advance_time path.  action_token stays invalid and
        // template_position is a write-only sentinel.
        arm.action_token = {};
        arm.partial_api_eligible = false;
        arm.template_position = UINT64_MAX;
    }
    arm.resource_version =
        workload_->local_hbm_bandwidth_model == nullptr
            ? 0
            : workload_->local_hbm_bandwidth_model->resource_version();
    arm.active = true;
    timer_arm_ = std::move(arm);

    ++event_generation_;
    pending_event_ = sys_->register_event_cancellable(
        this, EventType::General,
        acquire_event_data(DieEventData::Kind::Timer, 0, event_generation_,
                           DieEventData::TimerPurpose::InternalService),
        external_delay_ns, &destroy_event_data);
}

void DieExecution::settle_internal_timer_to(const Tick tick) {
    if (!timer_arm_.active || tick < timer_arm_.external_start_tick) {
        return;
    }
    const uint64_t elapsed = static_cast<uint64_t>(
        tick - timer_arm_.external_start_tick);
    uint64_t target = 0;
    if (elapsed >= timer_arm_.external_delay_ns) {
        // Closed form (batch 3): at or past due, bounded == external_delay_ns,
        // so the 128-bit numerator is original_service_ns * external_delay_ns,
        // which divides out to exactly original_service_ns with a zero
        // residual.  Skips the __int128 mul/div/mod (the __udivti3 libcall).
        target = timer_arm_.original_service_ns;
        timer_arm_.rounding_residual = 0;
    } else {
        // Pre-due: bounded == elapsed here; keep the original 128-bit path.
        const uint64_t bounded =
            std::min(elapsed, timer_arm_.external_delay_ns);
        const unsigned __int128 numerator =
            static_cast<unsigned __int128>(timer_arm_.original_service_ns) * bounded;
        target = static_cast<uint64_t>(
            numerator / timer_arm_.external_delay_ns);
        timer_arm_.rounding_residual = static_cast<uint64_t>(
            numerator % timer_arm_.external_delay_ns);
    }
    const uint64_t clamped_target =
        std::min(target, timer_arm_.original_service_ns);
    if (clamped_target < timer_arm_.service_already_applied) {
        throw std::runtime_error(
            "DieExecution timer service mapping moved backwards");
    }
    const uint64_t delta = clamped_target - timer_arm_.service_already_applied;
    if (delta != 0) {
        if (timer_arm_.partial_api_eligible) {
            machine_.advance_internal_service(timer_arm_.action_token, delta);
        } else {
            machine_.advance_time(delta);
        }
        timer_arm_.service_already_applied = clamped_target;
    }
    if (tick > last_synced_tick_) {
        last_synced_tick_ = tick;
    }
}

void DieExecution::on_hbm_resource_mutation(const bool before_change,
                                           const Tick tick,
                                           const uint64_t version) {
    if (finished_) return;
    if (before_change) {
        const auto memory_stage = machine_.pending_memory_stage_id();
        if (memory_stage.has_value()) {
            if (memory_latency_pending_) return;
            if (tick < last_synced_tick_) {
                throw std::runtime_error(
                    "DieExecution memory wait clock moved backwards");
            }
            const uint64_t delta = static_cast<uint64_t>(tick - last_synced_tick_);
            if (delta != 0) {
                machine_.advance_external_wait(delta, *memory_stage);
                last_synced_tick_ = tick;
            }
            return;
        }
        if (!timer_arm_.active) return;
        if (timer_arm_.external_delay_ns >
            std::numeric_limits<Tick>::max() - timer_arm_.external_start_tick) {
            throw std::runtime_error("DieExecution timer deadline overflows Tick");
        }
        const Tick due_tick = timer_arm_.external_start_tick +
                              timer_arm_.external_delay_ns;
        if (tick >= due_tick) {
            // The physical timer callback already has its original same-tick
            // place in Sys' FIFO. Settle its complete service, but preserve
            // that cookie and callback position for publication.
            timer_arm_.due_at_mutation = true;
            settle_internal_timer_to(tick);
            return;
        }
        settle_internal_timer_to(tick);
        // Exact timer service does not depend on this resource membership.
        // Keep the original physical callback in the Sys FIFO so a same-tick
        // peer cannot be reordered by cancel/reinsert.
        return;
    }

    if (timer_arm_.active) {
        timer_arm_.resource_version = version;
    }
}

void DieExecution::start() {
    // start() runs exactly once per execution at issue time (no enclosing
    // callback owns a clock reading yet), so take one reading here; the
    // identical value would be re-derived inside drive()'s Timer branch.
    drive(Sys::boostedTick());
}

void DieExecution::drive(const Tick now) {
    auto logger =
        sys_->trace_enabled ? LoggerFactory::get_logger("workload") : nullptr;
    // Pooling-stage-A invariant: the timer register paths are called only
    // from this drive loop, and drive() is entered only while
    // pending_event_ is invalid (start() right after construction;
    // call() after pending_event_.reset()), so at most ONE Timer payload
    // of this execution is ever in flight in the Sys queue.  Breaking
    // this rule (e.g. adding a drive() call at the tail of complete())
    // requires re-auditing the in-flight payload count before keeping
    // the freelist.
    while (!machine_.finished()) {
        const DieModel::ExecutionAction action = [this]() {
            const DieModel::ExecutionAction a = machine_.next_action();
            return a;
        }();
        switch (action.kind) {
            case DieModel::ExecutionAction::Kind::Timer: {
                // Keep the authoritative DIE cursor at its current service
                // frontier. The timer callback (or a resource-mutation
                // interruption) commits only the service that has elapsed.
                if (sys_->trace_enabled) {
                    logger->debug(
                        "die_exec node={} timer ns={} machine_elapsed={}",
                        node_id_, action.ns, machine_.elapsed_ns());
                }
                const bool hbm_idle =
                    workload_->local_hbm_bandwidth_model == nullptr ||
                    !workload_->local_hbm_bandwidth_model->has_active_jobs();
                uint64_t exact_timer_delay = action.ns;
                uint64_t logical_action_count = 1;

                // The exact contiguous-Timer preview path (the only timer
                // merge path since the K>1 window retirement): it stops at
                // the real next memory/done boundary and advances only the
                // original DIE clock when that one physical callback fires.
                if (hbm_idle) {
                    const DieModel::TimerStretchPreview stretch =
                        machine_.preview_consecutive_internal_timers();
                    if (stretch.timer_action_count != 0 &&
                        stretch.external_delay_ns != 0) {
                        exact_timer_delay = stretch.external_delay_ns;
                        logical_action_count = stretch.timer_action_count;
                    }
                }
                register_internal_timer(now, exact_timer_delay,
                                        exact_timer_delay,
                                        logical_action_count);
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
                    if (sys_->trace_enabled) {
                        logger->debug(
                            "die_exec node={} one-shot local_mem_latency "
                            "ns={}",
                            node_id_, sys_->local_mem_latency);
                    }
                    timer_arm_ = InternalTimerArm{};
                    register_external_timer(
                        sys_->local_mem_latency,
                        DieEventData::TimerPurpose::MemoryLatency);
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
                // at the FULL rate. 先乘后除 keeps the per-link rounding to
                // one ceil.
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
                register_external_timer(
                    std::max<uint64_t>(static_cast<uint64_t>(ns_double), 1),
                    DieEventData::TimerPurpose::SelfDrivenMemory,
                    action.stage_id);
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
    const DieEventData::TimerPurpose timer_purpose = data->timer_purpose;
    if (kind == DieEventData::Kind::Timer) {
        // Pooling stage A: Timer payloads return to the freelist; Memory
        // payloads keep the plain delete (stage B is out of scope).
        release_event_data(data);
    } else {
        delete data;
    }

    if (finished_ || generation != event_generation_) {
        // A cancelled non-due arm may still reach the callback through a
        // backend that cannot physically retract it. It owns no service.
        return;
    }

    const Tick now = Sys::boostedTick();
    if (now < last_synced_tick_) {
        throw std::runtime_error("DieExecution clock moved backwards");
    }
    pending_event_.reset();

    if (kind == DieEventData::Kind::Memory ||
        timer_purpose == DieEventData::TimerPurpose::SelfDrivenMemory) {
        const uint64_t wait_ns = static_cast<uint64_t>(now - last_synced_tick_);
        if (wait_ns != 0) {
            const auto memory_stage = machine_.pending_memory_stage_id();
            if (!memory_stage.has_value() || *memory_stage != stage_id) {
                throw std::runtime_error(
                    "DieExecution external wait lost its outstanding memory stage");
            }
            {
                machine_.advance_external_wait(wait_ns, stage_id);
            }
        }
        {
            machine_.complete_memory(stage_id);
        }
        last_synced_tick_ = now;
    } else if (timer_purpose == DieEventData::TimerPurpose::MemoryLatency) {
        // The first-access latency is external dead time and does not advance
        // the DIE service cursor.
        memory_latency_pending_ = false;
        last_synced_tick_ = now;
    } else {
        const uint64_t completed_logical_actions = timer_arm_.active
            ? timer_arm_.logical_action_count : 1;
        if (timer_arm_.active) {
            settle_internal_timer_to(now);
            timer_arm_ = InternalTimerArm{};
        } else {
            const uint64_t service_ns = static_cast<uint64_t>(now - last_synced_tick_);
            if (service_ns != 0) {
                machine_.advance_time(service_ns);
            }
            last_synced_tick_ = now;
        }
        logical_timer_event_count_ = checked_counter_add(
            logical_timer_event_count_, completed_logical_actions,
            "logical timer event");
        physical_timer_callback_count_ = checked_counter_add(
            physical_timer_callback_count_, 1, "physical timer callback");
    }
    drive(now);
}

void DieExecution::fill_utilizations(uint64_t elapsed_ns) {
    // Contract §7.6 caliber (design doc §8.5): compute = plan FLOP /
    // (PE_peak_die * node elapsed); memory = actual bytes / (B_mem * node
    // elapsed); vector = eq-ops / (V_peak * node elapsed). The peaks are the
    // die library's physical peaks, and the elapsed window is the node's
    // actual issue -> completion span (contention makes utilization fall
    // below the isolated estimate, never above the physical bound by
    // construction).
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

void DieExecution::complete() {
    if (finished_) {
        throw std::runtime_error(
            "DieExecution completion fired twice for node " +
            std::to_string(node_id_));
    }
    finished_ = true;
    // v7 attribution fix: drop the HBM mutation observer NOW instead of at
    // destruction. After finished_ the observer callback is a proven no-op
    // (its first line returns on finished_), and the adapter keeps finished
    // DieExecution objects alive (and their observers registered) until its
    // next entry point -- which never comes during a natural settle tail,
    // so the model's per-mutation observer fan-out grew with every retired
    // node. Removing here only skips no-op invocations; the destructor
    // removal below stays as the teardown backstop.
    if (hbm_observer_id_ != 0 && workload_ != nullptr &&
        workload_->local_hbm_bandwidth_model != nullptr) {
        workload_->local_hbm_bandwidth_model->remove_resource_mutation_observer(
            hbm_observer_id_);
        hbm_observer_id_ = 0;
    }
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
    // One GPU occupancy interval per outer node (design doc §6.3): the whole
    // issue -> completion span, mirroring the LocalHbm COMPUTE accounting.
    workload_->hw_resource->tics_gpu_ops += elapsed_ns;
    fill_utilizations(elapsed_ns);
    if (!quality_recorded_) {
        adapter_->note_execution_finished(this);
        quality_recorded_ = true;
    }
    // Exactly one generic completion (this repository's shared terminal body
    // takes the triggering event alongside the node id). The internal HBM
    // jobs of this node never reached this path -- only the machine's Done
    // does.
    workload_->finish_general_node(node_id_, EventType::General);
    delete wlhd_;
    wlhd_ = nullptr;
}

}  // namespace AstraSim
