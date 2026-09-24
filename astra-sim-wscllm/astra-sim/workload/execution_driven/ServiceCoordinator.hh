/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.

ServiceCoordinator -- execution-driven mechanism layer (wscllm phase 1).

Request-neutral lifecycle state machine (方案 §4 步骤 1-2 操作 3):

    IDLE     -- initial state; input open, no request accepted yet.
                Request-neutral default: without --request-queue-csv the
                service stays IDLE (no pre-loaded queue is read). Step 1-10:
                with the input still open, a fully drained ACTIVE returns to
                IDLE (五态迁移: IDLE -> ACTIVE -> IDLE -> DRAINING -> FINISHED).
    ACTIVE   -- at least one request accepted / arrival alarm pending.
    DRAINING -- input closed, pending alarms/requests still draining.
    FINISHED -- input closed && active==0 && no pending alarm.

Final end authority belongs exclusively to the ServiceCoordinator
(总体方案 §5.5): EventQueue empty != FINISHED, and in online mode the
Workload::call sim_notify_finished()/is_finished side effect is disabled
(Workload.cc:625-634), so a temporarily empty graph can never end the online
simulation and drop later injections.

wait_for_work() blocks (no busy wait) until signal_work() is raised or the
input is closed; the main loop must guarantee that after wait_for_work
returns, either the EventQueue has an alarm or the service is FINISHED,
otherwise the loop would spin forever (方案 §4 步骤 1-2 操作 5).
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_SERVICECOORDINATOR_HH
#define EXECUTION_DRIVEN_SERVICECOORDINATOR_HH

#include <chrono>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <vector>

namespace AstraSim {
namespace ExecutionDriven {

enum class ServiceState { IDLE, ACTIVE, DRAINING, FINISHED };

/// Phase-7 §10.7: input-close source (合同② "EOF vs 显式 close" 的审计落地).
/// The producer command queue distinguishes submit/close/EOF/error; this
/// enum records WHICH terminal command (or the --close-input CLI / direct
/// close path) closed the input, so a run-end audit can tell a natural EOF
/// from an explicit close. An Error command aborts in the ingress drain and
/// never reaches the coordinator (no Error reason exists).
enum class InputCloseReason { ExplicitClose, EndOfFile };

class ServiceCoordinator {
  public:
    // Fixture-only in-memory prefix. Production observers must use the hook,
    // which still receives every transition. Keeping a fixed prefix prevents
    // a long-lived open-input service from retaining one entry per request.
    static constexpr std::size_t kTransitionLogCapacity = 64;

    // Step 1-10: transition observer (fixture/status logging). Invoked from
    // set_state under the internal mutex; the hook MUST NOT call back into
    // the coordinator (documented contract, same as the state queries).
    using StateTransitionHook =
        std::function<void(ServiceState prev, ServiceState next)>;

    ServiceCoordinator() = default;

    // ---- sim-thread lifecycle accounting -------------------------------
    /// One submit command was drained from the ingress (accepted_request_count++
    /// and IDLE -> ACTIVE on first acceptance).
    void on_command_accepted();
    /// An arrival alarm fired (arrival_cb): pending_alarm_count--, active_count++.
    void on_request_arrived();
    /// A request finished (no-node fixture drain / step-1-3 CompletionObserver):
    /// completed_request_count++, active_count--, maybe FINISHED.
    void on_request_completed();
    /// An arrival alarm was queued on the EventQueue (future tick).
    void on_alarm_scheduled();
    /// Input closed (thread-safe; may be called from the injector/bridge
    /// thread): IDLE/DRAINING transition + maybe FINISHED. Phase-7 §10.7:
    /// `reason` records HOW the input was closed (explicit close command /
    /// CLI / EOF terminal command) for the run-end lifecycle audit
    /// (合同② EOF vs 显式 close 三态区分).
    void mark_input_closed(
        InputCloseReason reason = InputCloseReason::ExplicitClose);
    /// Phase-7 §10.7: how the input was closed (valid once the input is
    /// closed; ExplicitClose is the default for the direct/CLI close path).
    [[nodiscard]] InputCloseReason input_close_reason() const;
    /// Phase-7 §10.7: has the input been closed? (state query for the
    /// run-end audit; distinguishes "EOF reached but input still open" from
    /// a closed input.)
    [[nodiscard]] bool input_closed() const;

    // ---- main loop -------------------------------------------------------
    [[nodiscard]] bool finished() const;            // thread-safe
    [[nodiscard]] ServiceState state() const;       // thread-safe
    /// Block until signal_work() is raised or the input is closed.
    void wait_for_work();
    /// FP1 (2026-09-01, sync-A16 batch P; contract E9/E26): checked
    /// wall-clock deadline computation for the parking watchdog. Pure
    /// function over its inputs -- never touches the coordinator state --
    /// and the ONLY place the timeout -> duration -> deadline arithmetic
    /// happens (the old wait_for_work_for(duration) entry recomputed
    /// now()+timeout internally, so a caller-supplied overflow survived
    /// two separate checks). Frozen order:
    ///   1. reject non-finite or <= 0 seconds (the CLI layer already
    ///      accepted 0 = off and never calls the helper for it);
    ///   2. reject seconds above steady_clock::duration::max() in the tick
    ///      domain (checked on the double BEFORE any float->int conversion);
    ///   3. convert; reject a sub-tick zero duration;
    ///   4. reject when now > time_point::max() - timeout (pre-addition
    ///      bound; the subtraction is safe because timeout is proved
    ///      non-negative and representable);
    ///   5. compute deadline = now + timeout exactly once.
    /// On failure returns false and leaves out_deadline untouched.
    static bool checked_wait_deadline(
        double timeout_s, std::chrono::steady_clock::time_point now,
        std::chrono::steady_clock::time_point& out_deadline);
    /// FP1 (2026-09-01, sync-A16 batch P): absolute-deadline parking wait.
    /// Same predicate as wait_for_work, but bounded by the caller's
    /// precomputed steady_clock (WALL clock, never the simulation clock --
    /// real-time traces space two turn arrivals hours apart, so simulation
    /// time says nothing about liveness) deadline. Returns whether the
    /// predicate fired (true = woken by signal_work()/input-close/finish;
    /// false = timed out). On a wake-up it consumes wakeup_pending_
    /// exactly like wait_for_work; on a timeout nothing is consumed (a
    /// signal racing the timeout stays pending for the next wait). The
    /// coordinator takes now()/does no arithmetic of its own. The original
    /// no-timeout wait_for_work() is untouched (IDLE fixture contract).
    bool wait_for_work_until(std::chrono::steady_clock::time_point deadline);
    /// Raise the work signal (thread-safe; ingress enqueue / close path).
    void signal_work();

    /// Bounded ordered prefix of state transitions (fixture assertion data).
    [[nodiscard]] const std::vector<ServiceState>& transition_log() const;
    /// Number of transitions omitted after the bounded prefix filled.
    [[nodiscard]] uint64_t transition_log_dropped() const;
    /// Step 1-10: install the transition observer (IDLE/ACTIVE/DRAINING/
    /// FINISHED 迁移日志与时间). Overwrites any previous hook.
    void set_transition_hook(StateTransitionHook hook);

    // ---- counters (fixture/status assertions) ---------------------------
    [[nodiscard]] uint64_t accepted_request_count() const;
    [[nodiscard]] uint64_t completed_request_count() const;
    [[nodiscard]] uint64_t active_request_count() const;
    [[nodiscard]] uint64_t pending_alarm_count() const;

  private:
    void set_state(ServiceState next);   // under mtx_: bounded log + hook
    void maybe_finish();                 // under mtx_: FINISHED when drained
    [[nodiscard]] bool finished_locked() const;

    mutable std::mutex mtx_;
    std::condition_variable work_cv_;
    bool wakeup_pending_ = false;
    ServiceState state_ = ServiceState::IDLE;
    bool input_open_ = true;
    // Phase-7 §10.7: how the input was closed (valid once input_open_ is
    // false). Guarded by mtx_.
    InputCloseReason input_close_reason_ = InputCloseReason::ExplicitClose;
    uint64_t accepted_request_count_ = 0;
    uint64_t completed_request_count_ = 0;
    uint64_t active_request_count_ = 0;
    uint64_t pending_alarm_count_ = 0;
    std::vector<ServiceState> transition_log_;
    uint64_t transition_log_dropped_ = 0;
    StateTransitionHook transition_hook_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_SERVICECOORDINATOR_HH
