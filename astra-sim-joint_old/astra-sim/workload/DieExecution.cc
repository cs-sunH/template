/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "astra-sim/workload/DieExecution.hh"

#include <algorithm>
#include <cmath>
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
    std::shared_ptr<const DieModel::DiePlan> plan,
    std::shared_ptr<const ExecutionDriven::PipelineKernelContract>
        pipeline_contract)
    : sys_(sys),
      workload_(workload),
      adapter_(adapter),
      node_id_(node_id),
      wlhd_(wlhd),
      plan_(std::move(plan)),
      machine_(plan_),
      pipeline_contract_(std::move(pipeline_contract)),
      issue_tick_(Sys::boostedTick()),
      last_synced_tick_(issue_tick_),
      pending_event_() {
    if (sys_ == nullptr || workload_ == nullptr || adapter_ == nullptr ||
        wlhd_ == nullptr || !plan_->valid()) {
        throw std::runtime_error(
            "DieExecution requires Sys/Workload/adapter/handler data and a "
            "valid DiePlan");
    }
    if (pipeline_contract_ != nullptr) {
        effective_window_tile_limit_ =
            pipeline_contract_->window_tile_limit();
        window_controller_ = std::make_unique<
            ExecutionDriven::PipelineWindowController>(
                effective_window_tile_limit_);
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

nlohmann::json DieExecution::window_quality_json() const {
    const uint64_t windows = window_controller_ == nullptr
        ? 0 : window_controller_->window_count();
    const uint64_t refinements = checked_counter_add(
        window_controller_ == nullptr ? 0 : window_controller_->refinement_count(),
        uncommitted_refinement_count_, "refinement");
    const uint64_t prefix_bound = window_controller_ == nullptr
        ? 0 : window_controller_->local_prefix_bound_max();
    nlohmann::json result = pipeline_contract_ != nullptr &&
            window_controller_ != nullptr
        ? window_controller_->counters_json(*pipeline_contract_,
                                             effective_window_tile_limit_)
        : nlohmann::json::object();
    result["approximation_profile"] = pipeline_contract_ == nullptr
        ? "exact_legacy_die_timer"
        : "typed_die_internal_timer_window_v1";
    result["window_count"] = windows;
    result["refinement_count"] = refinements;
    result["local_prefix_bound_max"] = prefix_bound;
    result["logical_event_count"] = logical_timer_event_count_;
    result["physical_step_count"] = physical_timer_callback_count_;
    result["bound_kind"] = "local_prefix_service_ticks";
    result["fully_exact"] = windows == 0;
    result["unsupported_reason"] = last_template_exact_reason_.empty()
        ? nlohmann::json(nullptr)
        : nlohmann::json(last_template_exact_reason_);
    result["last_window_plan"] = last_window_plan_;
    return result;
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
}

void DieExecution::destroy_event_data(CallData* data) {
    delete static_cast<DieEventData*>(data);
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
        new DieEventData(DieEventData::Kind::Timer, stage_id,
                         event_generation_, purpose),
        delay_ns, &destroy_event_data);
}

void DieExecution::register_internal_timer(
    const uint64_t original_service_ns, const uint64_t external_delay_ns,
    const ExecutionDriven::PipelineWindowPlan* window_plan,
    const DieModel::InternalServiceWindowToken* service_window,
    const uint64_t logical_action_count,
    const ExecutionDriven::PipelineWindowPlan* exact_feedback_plan) {
    if (original_service_ns == 0 || external_delay_ns == 0) {
        throw std::runtime_error(
            "DieExecution refused to arm a zero-service internal timer");
    }
    const DieModel::InternalSliceInfo frontier =
        machine_.describe_internal_slice();
    InternalTimerArm arm;
    arm.original_service_ns = original_service_ns;
    arm.external_delay_ns = external_delay_ns;
    arm.logical_action_count = logical_action_count;
    arm.external_start_tick = Sys::boostedTick();
    arm.logical_start_service_tick = machine_.elapsed_ns();
    arm.window_generation = window_generation_;
    arm.stage_generation = frontier.action_token.frontier_generation;
    arm.action_token = frontier.action_token;
    arm.partial_api_eligible = frontier.partial_service_supported &&
        frontier.action_token.valid && frontier.has_next_boundary &&
        frontier.next_boundary_ns >= frontier.now_ns &&
        frontier.next_boundary_ns - frontier.now_ns >= original_service_ns;
    if (arm.partial_api_eligible && !frontier.active_stages.empty()) {
        arm.template_position = frontier.active_stages.front().wave_index;
    }
    if ((window_plan == nullptr) != (service_window == nullptr)) {
        throw std::runtime_error(
            "DieExecution window plan and stable token must be supplied together");
    }
    if (window_plan != nullptr) {
        if (!window_plan->accepted || window_plan->exact ||
            window_plan->selected_count <= 1 ||
            !service_window->valid ||
            window_plan->original_total_ticks != original_service_ns ||
            service_window->original_service_ticks.size() !=
                window_plan->selected_count) {
            throw std::runtime_error(
                "DieExecution refused an invalid approximate timer window");
        }
        uint64_t assigned_total = 0;
        for (const auto value : window_plan->assigned_service_ticks) {
            if (value > std::numeric_limits<uint64_t>::max() - assigned_total) {
                throw std::runtime_error(
                    "DieExecution assigned timer window overflows uint64");
            }
            assigned_total += value;
        }
        if (assigned_total != external_delay_ns) {
            throw std::runtime_error(
                "DieExecution external timer differs from committed window");
        }
        arm.window_plan = *window_plan;
        arm.service_window = *service_window;
        arm.partial_api_eligible = true;
        arm.uses_service_window = true;
        last_window_plan_ = window_plan->to_json(
            *pipeline_contract_, effective_window_tile_limit_);
        last_window_plan_["committed"] = true;
    }
    if (exact_feedback_plan != nullptr) {
        if (!exact_feedback_plan->accepted || !exact_feedback_plan->exact ||
            exact_feedback_plan->selected_count != 1 ||
            window_plan != nullptr || window_controller_ == nullptr) {
            throw std::runtime_error(
                "DieExecution refused an invalid exact fallback observation");
        }
        arm.exact_feedback_plan = *exact_feedback_plan;
        arm.has_exact_feedback_plan = true;
        last_window_plan_ = exact_feedback_plan->to_json(
            *pipeline_contract_, effective_window_tile_limit_);
        last_window_plan_["committed"] = false;
    }
    arm.resource_version_vector.push_back(
        workload_->local_hbm_bandwidth_model == nullptr
            ? 0
            : workload_->local_hbm_bandwidth_model->resource_version());
    arm.active = true;
    timer_arm_ = std::move(arm);

    ++event_generation_;
    pending_event_ = sys_->register_event_cancellable(
        this, EventType::General,
        new DieEventData(DieEventData::Kind::Timer, 0, event_generation_,
                         DieEventData::TimerPurpose::InternalService),
        external_delay_ns, &destroy_event_data);
}

void DieExecution::settle_internal_timer_to(const Tick tick) {
    if (!timer_arm_.active || tick < timer_arm_.external_start_tick) {
        return;
    }
    const uint64_t elapsed = static_cast<uint64_t>(
        tick - timer_arm_.external_start_tick);
    const uint64_t bounded = std::min(elapsed, timer_arm_.external_delay_ns);
    const unsigned __int128 numerator =
        static_cast<unsigned __int128>(timer_arm_.original_service_ns) * bounded;
    const uint64_t target = timer_arm_.uses_service_window
        ? ExecutionDriven::PipelineKernel::original_service_at_external_tick(
              timer_arm_.window_plan, bounded)
        : (timer_arm_.external_delay_ns == 0
            ? timer_arm_.original_service_ns
            : static_cast<uint64_t>(numerator /
                                    timer_arm_.external_delay_ns));
    const uint64_t clamped_target =
        std::min(target, timer_arm_.original_service_ns);
    timer_arm_.rounding_residual = timer_arm_.uses_service_window
        ? 0
        : (timer_arm_.external_delay_ns == 0
            ? 0
            : static_cast<uint64_t>(numerator %
                                    timer_arm_.external_delay_ns));
    if (clamped_target < timer_arm_.service_already_applied) {
        throw std::runtime_error(
            "DieExecution timer service mapping moved backwards");
    }
    const uint64_t delta = clamped_target - timer_arm_.service_already_applied;
    if (delta != 0) {
        if (timer_arm_.uses_service_window) {
            machine_.advance_internal_service_window(
                timer_arm_.service_window, delta);
        } else if (timer_arm_.partial_api_eligible) {
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

void DieExecution::record_window_completion() {
    if (!timer_arm_.uses_service_window || timer_arm_.window_plan_recorded) return;
    if (timer_arm_.service_already_applied !=
        timer_arm_.original_service_ns) {
        throw std::runtime_error(
            "DieExecution window callback reached its endpoint before original service");
    }
    if (!window_controller_) {
        throw std::runtime_error(
            "DieExecution accepted a window without its controller");
    }
    window_controller_->record(timer_arm_.window_plan, true);
    timer_arm_.window_plan_recorded = true;
}

void DieExecution::record_exact_feedback() {
    if (!timer_arm_.has_exact_feedback_plan ||
        timer_arm_.exact_feedback_plan_recorded) {
        return;
    }
    if (window_controller_ == nullptr) {
        throw std::runtime_error(
            "DieExecution exact fallback lost its window controller");
    }
    window_controller_->record(timer_arm_.exact_feedback_plan, true);
    timer_arm_.exact_feedback_plan_recorded = true;
    last_window_plan_["committed"] = true;
}

void DieExecution::rearm_interrupted_internal_timer(const Tick tick) {
    if (!timer_interrupted_for_mutation_ || !timer_arm_.active ||
        timer_arm_.due_at_mutation) {
        timer_interrupted_for_mutation_ = false;
        return;
    }
    const uint64_t remaining = timer_arm_.original_service_ns -
                               timer_arm_.service_already_applied;
    if (remaining == 0) {
        throw std::runtime_error(
            "DieExecution internal timer completed before its due tick");
    }
    const DieModel::InternalSliceInfo frontier =
        machine_.describe_internal_slice();
    if (timer_arm_.uses_service_window &&
        timer_arm_.service_window.completed_slices >
            timer_arm_.logical_action_count) {
        throw std::runtime_error(
            "DieExecution interrupted window cursor exceeds logical action count");
    }
    const uint64_t remaining_logical_actions = timer_arm_.uses_service_window
        ? timer_arm_.logical_action_count -
              timer_arm_.service_window.completed_slices
        : timer_arm_.logical_action_count;
    if (remaining_logical_actions == 0) {
        throw std::runtime_error(
            "DieExecution interrupted timer has no remaining logical action");
    }
    timer_arm_.action_token = frontier.action_token;
    timer_arm_.stage_generation = frontier.action_token.frontier_generation;
    timer_arm_.partial_api_eligible = frontier.partial_service_supported &&
        frontier.action_token.valid && frontier.has_next_boundary &&
        frontier.next_boundary_ns >= frontier.now_ns &&
        frontier.next_boundary_ns - frontier.now_ns >= remaining;
    timer_arm_.original_service_ns = remaining;
    timer_arm_.external_delay_ns = remaining;
    timer_arm_.logical_action_count = remaining_logical_actions;
    timer_arm_.external_start_tick = tick;
    timer_arm_.service_already_applied = 0;
    timer_arm_.rounding_residual = 0;
    timer_arm_.logical_start_service_tick = machine_.elapsed_ns();
    timer_arm_.window_generation = window_generation_;
    timer_arm_.uses_service_window = false;
    timer_arm_.window_plan_recorded = false;
    timer_arm_.has_exact_feedback_plan = false;
    timer_arm_.exact_feedback_plan_recorded = false;
    timer_arm_.window_plan = ExecutionDriven::PipelineWindowPlan{};
    timer_arm_.exact_feedback_plan = ExecutionDriven::PipelineWindowPlan{};
    timer_arm_.service_window = DieModel::InternalServiceWindowToken{};
    timer_arm_.due_at_mutation = false;
    timer_arm_.resource_version_vector.clear();
    timer_arm_.resource_version_vector.push_back(
        workload_->local_hbm_bandwidth_model == nullptr
            ? 0
            : workload_->local_hbm_bandwidth_model->resource_version());
    timer_arm_.active = true;
    timer_interrupted_for_mutation_ = false;
    ++event_generation_;
    pending_event_ = sys_->register_event_cancellable(
        this, EventType::General,
        new DieEventData(DieEventData::Kind::Timer, 0, event_generation_,
                         DieEventData::TimerPurpose::InternalService),
        remaining, &destroy_event_data);
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
            record_window_completion();
            record_exact_feedback();
            if (window_controller_ != nullptr) {
                window_controller_->invalidate();
            }
            if (window_generation_ == std::numeric_limits<uint64_t>::max()) {
                throw std::runtime_error(
                    "DieExecution window generation overflows uint64");
            }
            ++window_generation_;
            return;
        }
        settle_internal_timer_to(tick);
        if (timer_arm_.uses_service_window &&
            timer_arm_.service_already_applied != 0 &&
            !timer_arm_.window_plan_recorded &&
            window_controller_ != nullptr) {
            window_controller_->record_partial(timer_arm_.window_plan);
        }
        if (window_controller_ != nullptr) {
            window_controller_->invalidate();
        }
        if (!timer_arm_.uses_service_window) {
            // Exact K=1 service does not depend on this resource membership.
            // Keep the original physical callback in the Sys FIFO so a
            // same-tick peer cannot be reordered by cancel/reinsert.
            if (timer_arm_.has_exact_feedback_plan) {
                timer_arm_.has_exact_feedback_plan = false;
                timer_arm_.exact_feedback_plan_recorded = false;
                last_template_exact_reason_ =
                    "resource_changed_before_exact_fallback_callback";
            }
            if (window_generation_ == std::numeric_limits<uint64_t>::max()) {
                throw std::runtime_error(
                    "DieExecution window generation overflows uint64");
            }
            ++window_generation_;
            return;
        }
        const bool cancelled = pending_event_.valid() &&
                               sys_->cancel_event(pending_event_);
        if (!cancelled) {
            throw std::runtime_error(
                "DieExecution could not cancel a not-yet-due timer arm");
        }
        timer_interrupted_for_mutation_ = true;
        if (window_generation_ == std::numeric_limits<uint64_t>::max()) {
            throw std::runtime_error(
                "DieExecution window generation overflows uint64");
        }
        ++window_generation_;
        return;
    }

    if (timer_arm_.active && !timer_arm_.due_at_mutation) {
        timer_arm_.window_generation = window_generation_;
        if (timer_arm_.resource_version_vector.empty()) {
            timer_arm_.resource_version_vector.push_back(version);
        } else {
            timer_arm_.resource_version_vector[0] = version;
        }
    } else if (timer_arm_.active && timer_arm_.due_at_mutation) {
        timer_arm_.window_generation = window_generation_;
        if (timer_arm_.resource_version_vector.empty()) {
            timer_arm_.resource_version_vector.push_back(version);
        } else {
            timer_arm_.resource_version_vector[0] = version;
        }
    }
    rearm_interrupted_internal_timer(tick);
}

void DieExecution::start() {
    drive();
}

void DieExecution::drive() {
    auto logger = LoggerFactory::get_logger("workload");
    while (!machine_.finished()) {
        const DieModel::ExecutionAction action = machine_.next_action();
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
                const ExecutionDriven::PipelineWindowPlan* exact_feedback =
                    nullptr;
                ExecutionDriven::PipelineWindowPlan exact_feedback_value;
                uint64_t exact_timer_delay = action.ns;
                uint64_t logical_action_count = 1;
                last_window_plan_ = nullptr;

                if (hbm_idle && pipeline_contract_ != nullptr &&
                    window_controller_ != nullptr &&
                    window_controller_->current_limit() > 1) {
                    const uint32_t current_limit =
                        window_controller_->current_limit();
                    const DieModel::InternalTimerWindowInfo info =
                        machine_.query_internal_timer_window(current_limit);
                    if (!info.supported) {
                        last_template_exact_reason_ = info.reason.empty()
                            ? "native_timer_window_unsupported"
                            : info.reason;
                    } else {
                        std::vector<uint64_t> original;
                        std::vector<uint64_t> hardware_minimum;
                        original.reserve(info.slices.size());
                        hardware_minimum.reserve(info.slices.size());
                        uint64_t local_span = 0;
                        for (const auto& slice : info.slices) {
                            if (slice.original_service_ns == 0 ||
                                slice.original_service_ns >
                                    std::numeric_limits<uint64_t>::max() -
                                        local_span) {
                                throw std::runtime_error(
                                    "DieExecution immutable timer window has an invalid service span");
                            }
                            local_span += slice.original_service_ns;
                            original.push_back(slice.original_service_ns);
                            hardware_minimum.push_back(slice.hardware_min_ns);
                        }
                        const ExecutionDriven::PipelineWindowPlan plan =
                            window_controller_->plan(
                                *pipeline_contract_,
                                effective_window_tile_limit_, original,
                                hardware_minimum, local_span);
                        last_window_plan_ = plan.to_json(
                            *pipeline_contract_,
                            effective_window_tile_limit_);
                        last_window_plan_["committed"] = false;
                        if (plan.accepted && !plan.exact &&
                            plan.selected_count > 1) {
                            if (plan.selected_count >
                                info.token.original_service_ticks.size()) {
                                throw std::runtime_error(
                                    "DieExecution window plan exceeds its proved service token");
                            }
                            DieModel::InternalServiceWindowToken service_window =
                                info.token;
                            service_window.original_service_ticks.resize(
                                plan.selected_count);
                            uint64_t assigned_total = 0;
                            for (const uint64_t assigned :
                                 plan.assigned_service_ticks) {
                                if (assigned > std::numeric_limits<uint64_t>::max() -
                                                   assigned_total) {
                                    throw std::runtime_error(
                                        "DieExecution assigned window span overflows uint64");
                                }
                                assigned_total += assigned;
                            }
                            last_template_exact_reason_.clear();
                            if (sys_->trace_enabled) {
                                logger->debug(
                                    "die_exec node={} timer_window K={} original={} assigned={} prefix_bound={}",
                                    node_id_, plan.selected_count,
                                    plan.original_total_ticks, assigned_total,
                                    plan.max_prefix_delta_ticks);
                            }
                            register_internal_timer(
                                plan.original_total_ticks, assigned_total,
                                &plan, &service_window,
                                plan.selected_count);
                            return;
                        }
                        if (plan.accepted && plan.exact &&
                            plan.selected_count == 1) {
                            exact_feedback_value = plan;
                            exact_feedback = &exact_feedback_value;
                            last_template_exact_reason_ =
                                plan.unsupported_reason.empty()
                                    ? "exact_single_slice_fallback"
                                    : plan.unsupported_reason;
                        } else {
                            last_template_exact_reason_ =
                                plan.unsupported_reason.empty()
                                    ? "window_not_committed"
                                    : plan.unsupported_reason;
                            uncommitted_refinement_count_ = checked_counter_add(
                                uncommitted_refinement_count_,
                                plan.refinement_count, "uncommitted refinement");
                        }
                    }
                } else if (!hbm_idle) {
                    last_template_exact_reason_ = "hbm_membership_active";
                } else if (pipeline_contract_ == nullptr ||
                           window_controller_ == nullptr) {
                    last_template_exact_reason_ =
                        "missing_run_contract_exact_only";
                } else {
                    last_template_exact_reason_ = "contract_limit_one";
                }

                // Preserve the existing exact contiguous-Timer preview path
                // on ineligible or K=1 service. It stops at the real next
                // memory/done boundary and advances only the original DIE
                // clock when that one physical callback fires.
                if (hbm_idle) {
                    const DieModel::TimerStretchPreview stretch =
                        machine_.preview_consecutive_internal_timers();
                    if (stretch.timer_action_count != 0 &&
                        stretch.external_delay_ns != 0) {
                        exact_timer_delay = stretch.external_delay_ns;
                        logical_action_count = stretch.timer_action_count;
                    }
                }
                register_internal_timer(exact_timer_delay, exact_timer_delay,
                                        nullptr, nullptr, logical_action_count,
                                        exact_feedback);
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
    delete data;

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
            machine_.advance_external_wait(wait_ns, stage_id);
        }
        machine_.complete_memory(stage_id);
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
            record_window_completion();
            record_exact_feedback();
            timer_arm_ = InternalTimerArm{};
        } else {
            const uint64_t service_ns = static_cast<uint64_t>(now - last_synced_tick_);
            if (service_ns != 0) machine_.advance_time(service_ns);
            last_synced_tick_ = now;
        }
        logical_timer_event_count_ = checked_counter_add(
            logical_timer_event_count_, completed_logical_actions,
            "logical timer event");
        physical_timer_callback_count_ = checked_counter_add(
            physical_timer_callback_count_, 1, "physical timer callback");
        timer_interrupted_for_mutation_ = false;
    }
    drive();
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
