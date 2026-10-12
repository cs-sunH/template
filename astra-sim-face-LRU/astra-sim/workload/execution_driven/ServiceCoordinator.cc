/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ServiceCoordinator -- execution-driven mechanism layer (wscllm phase 1).
Implementation (方案 §4 步骤 1-2 操作 3): IDLE/ACTIVE/DRAINING/FINISHED
lifecycle; final end authority; no-busy-wait work channel. Step 1-10:
fully-drained ACTIVE returns to IDLE while the input stays open (合同②
目标 5 五态迁移), plus an optional transition hook for the lifecycle log.
*******************************************************************************/

#include "astra-sim/workload/execution_driven/ServiceCoordinator.hh"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>

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
    // E3 fix (2026-09-26): the invariant used to be assert-only, i.e. a
    // no-op under NDEBUG. An arrival with no pending alarm would wrap the
    // uint64_t counter below to ~0, and finished_locked() would then never
    // observe the drained state (FINISHED unreachable). Fail closed with a
    // diagnostic instead, matching the repo-wide coordinator/ingress abort
    // contract. (Keep the assert for Debug builds' early break.)
    assert(pending_alarm_count_ > 0);
    if (pending_alarm_count_ == 0) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/coordinator) arrival with "
                     "no pending alarm (pending_alarm_count_ == 0)\n");
        std::abort();
    }
    --pending_alarm_count_;
    ++active_request_count_;
    if (state_ == ServiceState::IDLE) {
        set_state(ServiceState::ACTIVE);
    }
}

void ServiceCoordinator::on_request_completed() {
    std::lock_guard<std::mutex> lock(mtx_);
    // E3 fix (2026-09-26): same NDEBUG-proofing as on_request_arrived -- a
    // completion with no active request would wrap active_request_count_
    // below to ~0 and wedge the lifecycle short of FINISHED.
    assert(active_request_count_ > 0);
    if (active_request_count_ == 0) {
        std::fprintf(stderr,
                     "[Error] (execution_driven/coordinator) completion with "
                     "no active request (active_request_count_ == 0)\n");
        std::abort();
    }
    --active_request_count_;
    ++completed_request_count_;
    maybe_finish();
}

void ServiceCoordinator::on_alarm_scheduled() {
    std::lock_guard<std::mutex> lock(mtx_);
    ++pending_alarm_count_;
}

void ServiceCoordinator::mark_input_closed(const InputCloseReason reason) {
    std::lock_guard<std::mutex> lock(mtx_);
    // Phase-7 §10.7: record the close source BEFORE the state transitions
    // (the hook below may read it back on the same thread only via the
    // state queries; the audit queries input_close_reason()/input_closed()
    // after the run). (2026-10-01) Keep the FIRST reason on double close:
    // an explicit --close-input followed by the CSV EOF command used to
    // overwrite the audit attribution from "explicit" to "eof".
    if (input_open_) {
        input_close_reason_ = reason;
    }
    input_open_ = false;
    if (state_ == ServiceState::IDLE) {
        // no request was ever accepted: close straight into draining
        set_state(ServiceState::DRAINING);
    } else if (state_ == ServiceState::ACTIVE) {
        set_state(ServiceState::DRAINING);
    }
    // M4 fix (2026-09-29): do NOT evaluate maybe_finish() here. At a
    // producer-linearized close the bounded ingress queue routinely holds
    // every Submit (the --close-input startup path queues the whole turn-0
    // calendar before the first drain), so all three counters are still
    // zero and the old immediate evaluation stamped FINISHED for a service
    // that had not even accepted its first request -- the state_ label then
    // sat at FINISHED for the whole run and the lifecycle transition log
    // went silent. The FINISHED evaluation moves to note_input_drained(),
    // called by the ingress drain once the closed queue is empty.
    finish_after_drain_ = true;
    wakeup_pending_ = true;
    work_cv_.notify_one();
}

void ServiceCoordinator::note_input_drained() {
    std::lock_guard<std::mutex> lock(mtx_);
    if (input_open_ || !finish_after_drain_) {
        return;  // input still open, or nothing deferred (already drained)
    }
    finish_after_drain_ = false;
    maybe_finish();
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

bool ServiceCoordinator::checked_wait_deadline(
    const double timeout_s, const std::chrono::steady_clock::time_point now,
    std::chrono::steady_clock::time_point& out_deadline) {
    // FP1 (2026-09-01, sync-A16 batch P; contract E9/E26 §2.4): the frozen
    // five-step order -- bound-check in the tick domain BEFORE any
    // float->integer conversion, convert once, bound-check the addition
    // BEFORE it happens, add once. The helper keeps the 0-fails contract
    // (the CLI layer treats 0 as "off" and never calls it with 0; a
    // negative or non-finite timeout is a programming error and fails the
    // same way).
    if (!std::isfinite(timeout_s) || timeout_s <= 0.0) {
        return false;
    }
    // The upper bound must live in the seconds domain: the tick count of
    // steady_clock::duration::max() is ns ticks, not seconds, so comparing
    // a seconds value against the raw count would let any timeout of
    // roughly (max/1e9, max) seconds pass and overflow the float->int
    // conversion below. Scale the tick count by the tick period; '>=' also
    // rejects the exact boundary, where the double multiply could round up
    // past duration::max().
    const double duration_max_s =
        static_cast<double>(
            std::chrono::steady_clock::duration::max().count()) *
        static_cast<double>(
            std::chrono::steady_clock::duration::period::num) /
        static_cast<double>(
            std::chrono::steady_clock::duration::period::den);
    if (timeout_s >= duration_max_s) {
        return false;  // could not be represented as a duration at all
    }
    const std::chrono::steady_clock::duration timeout =
        std::chrono::duration_cast<std::chrono::steady_clock::duration>(
            std::chrono::duration<double>(timeout_s));
    if (timeout <= std::chrono::steady_clock::duration::zero()) {
        return false;  // sub-tick: parsed fine, but cannot arm even one tick
    }
    const auto epoch_max =
        std::chrono::steady_clock::time_point::max().time_since_epoch();
    if (now.time_since_epoch() > epoch_max - timeout) {
        return false;  // the addition itself would overflow time_point
    }
    out_deadline = now + timeout;
    return true;
}

bool ServiceCoordinator::wait_for_work_until(
    const std::chrono::steady_clock::time_point deadline) {
    // FP1 (2026-09-01, sync-A16 batch P): absolute-deadline variant of the
    // P0-2 wall-clock parking watchdog. Identical predicate and
    // wakeup_pending_ consumption as wait_for_work; the deadline arrives
    // precomputed from checked_wait_deadline, so this function takes
    // now()/does no arithmetic of its own (the old duration entry
    // recomputed now()+timeout internally -- E9). wait_until with a
    // predicate returns false only on timeout (spurious wake-ups
    // re-evaluate the predicate and keep waiting), so a false return is a
    // genuine liveness failure of every producer: the caller turns it into
    // a fail-closed abort with the full parking diagnostics instead of the
    // old silent forever-wait (the 41-minute futex wedge).
    std::unique_lock<std::mutex> lock(mtx_);
    const bool woke = work_cv_.wait_until(lock, deadline, [this] {
        return wakeup_pending_ || !input_open_ || finished_locked();
    });
    if (woke) {
        wakeup_pending_ = false;
    }
    return woke;
}

void ServiceCoordinator::signal_work() {
    std::lock_guard<std::mutex> lock(mtx_);
    wakeup_pending_ = true;
    work_cv_.notify_one();
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
    // accepted/active request has drained (active==0, no pending alarm) --
    // the service returns to IDLE and keeps waiting for the next
    // external injection (IDLE --ACTIVE--> ... --complete--> IDLE). The
    // official runners close the input at startup, so this branch is
    // unreachable there (input_open_ is false); it is the IDLE fixture's
    // contract.
    if (input_open_ && active_request_count_ == 0 &&
        pending_alarm_count_ == 0 && state_ == ServiceState::ACTIVE) {
        set_state(ServiceState::IDLE);
    }
}

bool ServiceCoordinator::finished_locked() const {
    return !input_open_ && active_request_count_ == 0 &&
           pending_alarm_count_ == 0;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
