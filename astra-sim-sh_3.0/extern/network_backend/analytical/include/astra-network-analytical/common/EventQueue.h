/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/EventList.h"
#include "common/Type.h"
#include <list>
#include <map>

namespace NetworkAnalytical {

/**
 * EventQueue manages scheduled EventLists.
 */
class EventQueue {
  public:
    /**
     * Constructor.
     */
    EventQueue() noexcept;

    /**
     * Get current event time of the event queue.
     *
     * @return current event time
     */
    [[nodiscard]] EventTime get_current_time() const noexcept;

    /**
     * Check all registered events are invoked.
     * i.e., check if the event queue is empty.
     *
     * @return true if the event queue is empty, false otherwise
     */
    [[nodiscard]] bool finished() const noexcept;

    /**
     * Proceed the event queue.
     * i.e., first update the current event time to the next registered event
     * time, and then invoke all events registered at the current updated event
     * time.
     */
    void proceed() noexcept;

    /**
     * Schedule an event with a given event time.
     *
     * @param event_time time of event
     * @param callback callback function pointer
     * @param callback_arg argument of the callback function
     */
    void schedule_event(EventTime event_time, Callback callback, CallbackArg callback_arg) noexcept;

    /**
     * Set the tick-end callback (phase-1 execution-driven extension, map
     * version; semantics identical to the list-version blueprint).
     *
     * When set, the callback is invoked exactly once per proceed(), after the
     * current tick's physical EventList has been erased from the map and
     * before the same-tick deferred queue drains. When not set (nullptr, the
     * default) the queue behaves byte-for-byte like the pre-extension
     * implementation.
     *
     * Hard rule (see EventQueue.cpp proceed()): from inside the tick-end
     * callback or a deferred handler, same-tick events MUST go through
     * schedule_event_deferred(); schedule_event(current_time, ...) would
     * try_emplace a current_time EventList into the main map queue and trip
     * the strict-increase assert on the next proceed().
     *
     * @param callback tick-end callback function pointer (nullptr = disabled)
     * @param arg argument passed to the tick-end callback
     */
    void set_tick_end_callback(Callback callback, CallbackArg arg) noexcept;

    /**
     * Schedule an event for same-tick post-commit execution (phase-1
     * execution-driven extension).
     *
     * The event is executed by the deferred drain at the end of the current
     * proceed() (after the tick-end callback), in insertion order. Events
     * scheduled while the drain is already running are executed by the same
     * drain pass. Deferred events belong to the current tick: they must not
     * be used for future times (use schedule_event for those).
     *
     * @param callback callback function pointer
     * @param callback_arg argument of the callback function
     */
    void schedule_event_deferred(Callback callback, CallbackArg arg) noexcept;

    /**
     * Whether the same-tick deferred queue still holds unexecuted events
     * (phase-1 execution-driven extension, map version; defect-C fix
     * 2026-08-16, synced from face-defectfix2-done -- semantics identical
     * to the list-version blueprint's accessor).
     *
     * The deferred queue (a std::list<EventList>, independent of the main
     * std::map<EventTime, EventList>) only drains inside proceed(); a
     * deferred event scheduled outside a proceed context (or left over
     * after one) is INVISIBLE to finished() -- the main map can be empty
     * while deferred work still pends. The online main loop queries this
     * so it can force the next decision boundary (schedule_event(T+1,
     * wakeup)) and let a proceed() drain it, instead of blocking on a
     * wait that would never be woken. Informational for the static binary
     * (nothing schedules deferred there).
     *
     * @return true if deferred_queue_ is non-empty
     */
    [[nodiscard]] bool has_deferred_work() const noexcept;

    /**
     * Whether the current execution context is the main-queue invoke pass
     * (phase-1 execution-driven extension).
     *
     * True only while proceed() invokes the current tick's physical EventList;
     * false in the tick-end callback and in the deferred drain. A same-tick
     * schedule_event(current_time, ...) is legal ONLY in the invoke context
     * (it merges into the EventList currently being invoked); from any other
     * context it must go through schedule_event_deferred(). This accessor
     * lets shared front-end code (e.g. CommonNetworkApi::sim_recv's immediate
     * completion path) pick the correct route without knowing the execution
     * mode. The static binary never executes outside the invoke context, so
     * the flag is informational there -- zero behavior change.
     *
     * @return true if inside invoke_events() of the main queue
     */
    [[nodiscard]] bool in_invoke_context() const noexcept;

  private:
    /// true while the main queue's current EventList is being invoked
    bool in_invoke_ = false;

    /// current time of the event queue
    EventTime current_time;

    /// EventLists indexed by event time.
    std::map<EventTime, EventList> event_queue;

    /// tick-end callback (nullptr = disabled; zero behavior change)
    Callback tick_end_cb_ = nullptr;

    /// argument passed to the tick-end callback
    CallbackArg tick_end_arg_ = nullptr;

    /// same-tick post-commit deferred event lists
    std::list<EventList> deferred_queue_;
};

}  // namespace NetworkAnalytical
