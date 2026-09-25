/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/EventList.h"
#include "common/Type.h"
#include <cstddef>
#include <deque>
#include <map>
#include <utility>

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
     * Schedule an event with an explicit cancellation cleanup contract.
     *
     * If cancel_event(handle) succeeds before the queue pops the event, the
     * EventList node is erased and cancellation_callback(callback_arg) runs
     * synchronously.  Once the event is popped for callback invocation,
     * cancellation safely returns false; the callback owns callback_arg.
     */
    [[nodiscard]] EventHandle schedule_event_cancellable(
        EventTime event_time,
        Callback callback,
        CallbackArg callback_arg,
        EventCancellationCallback cancellation_callback) noexcept;

    /// Cancel one pending main-queue event.  The handle is consumed whether
    /// cancellation succeeds or the event was already popped/removed.
    [[nodiscard]] bool cancel_event(EventHandle& handle) noexcept;

    /// Number of physical main-queue callbacks currently resident. Deferred
    /// post-commit callbacks are intentionally excluded.
    [[nodiscard]] size_t scheduled_event_count() const noexcept;

    /**
     * Set the tick-end callback (phase-1 execution-driven extension, map-family).
     *
     * When set, the callback is invoked exactly once per proceed(), after the
     * current tick's physical EventList has been erased from the map and before
     * the same-tick deferred queue drains. When not set (nullptr, the default)
     * the queue behaves byte-for-byte like the pre-extension implementation.
     *
     * Hard rule (see EventQueue.cpp proceed()): from inside the tick-end
     * callback or a deferred handler, same-tick events MUST go through
     * schedule_event_deferred(); schedule_event(current_time, ...) would
     * try_emplace a current_time EventList into the main map and trip the
     * strict-increase assert (EventQueue.cpp :48) on the next proceed().
     */
    void set_tick_end_callback(Callback callback, CallbackArg arg) noexcept;

    /**
     * Schedule an event for same-tick post-commit execution. The event is
     * executed by the deferred drain at the end of the current proceed() (or
     * of the next one if called outside a proceed context), in insertion
     * order. Deferred events belong to the current tick.
     */
    void schedule_event_deferred(Callback callback, CallbackArg arg) noexcept;

    /**
     * Whether the same-tick deferred queue still holds unexecuted events
     * (phase-1 execution-driven extension, map family; defect-C fix
     * 2026-08-16, 同步自 face 0049ef5 / face-defectfix2-done).
     *
     * The deferred queue (deferred_queue_, container-independent of the
     * main std::map) only drains inside proceed(); a deferred event
     * scheduled outside a proceed context (or left over after one) is
     * INVISIBLE to finished() -- the main map can be empty while deferred
     * work still pends. The online main loop queries this so it can force
     * the next decision boundary (schedule_event(T+1, wakeup)) and let a
     * proceed() drain it, instead of blocking on a wait that would never
     * be woken. Informational for the static binary (nothing schedules
     * deferred there).
     *
     * @return true if deferred_queue_ is non-empty
     */
    [[nodiscard]] bool has_deferred_work() const noexcept;

    /**
     * Whether the current execution context is the main-queue invoke pass.
     * True only while proceed() invokes the current tick's physical EventList;
     * false in the tick-end callback and in the deferred drain. A same-tick
     * schedule_event(current_time, ...) is legal ONLY in the invoke context
     * (it merges into the EventList currently being invoked via try_emplace);
     * from any other context it must go through schedule_event_deferred().
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

    /// Same-tick post-commit FIFO. A deque of callback pairs avoids allocating
    /// one EventList list node plus one Event node for every terminal
    /// completion. Nested scheduling remains FIFO and drains in the same pass.
    std::deque<std::pair<Callback, CallbackArg>> deferred_queue_;
};

}  // namespace NetworkAnalytical
