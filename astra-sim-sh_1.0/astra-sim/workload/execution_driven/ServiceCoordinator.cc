/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ServiceCoordinator -- execution-driven mechanism layer (sh_1.0 port; blueprint wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 3): IDLE/ACTIVE/DRAINING/FINISHED
lifecycle; final end authority; no-busy-wait work channel. Step 1-10:
fully-drained ACTIVE returns to IDLE while the input stays open (合同②
目标 5 五态迁移), plus an optional transition hook for the lifecycle log.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"

#include <cassert>

namespace AstraSim {
namespace ExecutionDriven {

void ServiceCoordinator::on_command_accepted() {
    std::lock_guard<std::mutex> lock(mtx_);
    ++accepted_request_count_;
    if (state_ == ServiceState::IDLE) {
        set_state(ServiceState::ACTIVE);
    }
}

void ServiceCoordinator::on_request_arrived() {
    std::lock_guard<std::mutex> lock(mtx_);
    assert(pending_alarm_count_ > 0);
    --pending_alarm_count_;
    ++active_request_count_;
    if (state_ == ServiceState::IDLE) {
        set_state(ServiceState::ACTIVE);
    }
}

void ServiceCoordinator::on_request_completed() {
    std::lock_guard<std::mutex> lock(mtx_);
    assert(active_request_count_ > 0);
    --active_request_count_;
    ++completed_request_count_;
    maybe_finish();
}

void ServiceCoordinator::on_alarm_scheduled() {
    std::lock_guard<std::mutex> lock(mtx_);
    ++pending_alarm_count_;
}

void ServiceCoordinator::on_fence_scheduled() {
    std::lock_guard<std::mutex> lock(mtx_);
    ++pending_fence_count_;
}

void ServiceCoordinator::on_fence_resolved() {
    std::lock_guard<std::mutex> lock(mtx_);
    assert(pending_fence_count_ > 0);
    --pending_fence_count_;
    maybe_finish();
}

void ServiceCoordinator::mark_input_closed(const InputCloseReason reason) {
    std::lock_guard<std::mutex> lock(mtx_);
    // Phase-7 §10.7: record the close source BEFORE the state transitions
    // (the hook below may read it back on the same thread only via the
    // state queries; the audit queries input_close_reason()/input_closed()
    // after the run).
    input_close_reason_ = reason;
    input_open_ = false;
    if (state_ == ServiceState::IDLE) {
        // no request was ever accepted: close straight into draining
        set_state(ServiceState::DRAINING);
    } else if (state_ == ServiceState::ACTIVE) {
        set_state(ServiceState::DRAINING);
    }
    maybe_finish();
    wakeup_pending_ = true;
    work_cv_.notify_one();
}

InputCloseReason ServiceCoordinator::input_close_reason() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return input_close_reason_;
}

bool ServiceCoordinator::input_closed() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return !input_open_;
}

bool ServiceCoordinator::finished() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return finished_locked();
}

ServiceState ServiceCoordinator::state() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return state_;
}

void ServiceCoordinator::wait_for_work() {
    std::unique_lock<std::mutex> lock(mtx_);
    work_cv_.wait(lock, [this] {
        return wakeup_pending_ || !input_open_ || finished_locked();
    });
    wakeup_pending_ = false;
}

void ServiceCoordinator::signal_work() {
    std::lock_guard<std::mutex> lock(mtx_);
    wakeup_pending_ = true;
    work_cv_.notify_one();
}

const std::vector<ServiceState>& ServiceCoordinator::transition_log() const {
    return transition_log_;
}

void ServiceCoordinator::set_transition_hook(StateTransitionHook hook) {
    std::lock_guard<std::mutex> lock(mtx_);
    transition_hook_ = std::move(hook);
}

uint64_t ServiceCoordinator::accepted_request_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return accepted_request_count_;
}

uint64_t ServiceCoordinator::completed_request_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return completed_request_count_;
}

uint64_t ServiceCoordinator::active_request_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return active_request_count_;
}

uint64_t ServiceCoordinator::pending_alarm_count() const {
    std::lock_guard<std::mutex> lock(mtx_);
    return pending_alarm_count_;
}

void ServiceCoordinator::set_state(const ServiceState next) {
    // caller holds mtx_
    const ServiceState prev = state_;
    state_ = next;
    transition_log_.push_back(next);
    if (transition_hook_) {
        // Contract: the hook must not call back into the coordinator (it is
        // invoked with mtx_ held); the online entry uses it to print the
        // lifecycle transition log with wall timestamps (step 1-10).
        transition_hook_(prev, next);
    }
}

void ServiceCoordinator::maybe_finish() {
    // caller holds mtx_
    if (state_ == ServiceState::FINISHED) {
        return;
    }
    if (finished_locked()) {
        set_state(ServiceState::FINISHED);
        wakeup_pending_ = true;
        work_cv_.notify_one();
        return;
    }
    // Step 1-10 (合同② 目标 5 五态迁移): the input is still open and every
    // accepted/active request has drained (active==0, no pending alarm or
    // fence) -- the service returns to IDLE and keeps waiting for the next
    // external injection (IDLE --ACTIVE--> ... --complete--> IDLE). The
    // official runners close the input at startup, so this branch is
    // unreachable there (input_open_ is false); it is the IDLE fixture's
    // contract.
    if (input_open_ && active_request_count_ == 0 &&
        pending_alarm_count_ == 0 && pending_fence_count_ == 0 &&
        state_ == ServiceState::ACTIVE) {
        set_state(ServiceState::IDLE);
    }
}

bool ServiceCoordinator::finished_locked() const {
    return !input_open_ && active_request_count_ == 0 &&
           pending_alarm_count_ == 0 && pending_fence_count_ == 0;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
