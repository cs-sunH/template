/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include <cstdint>

namespace NetworkAnalytical {

/// Cancellable-event lifecycle states (inline in QueuedEvent; the former
/// heap-allocated EventControl block is gone). The cancellation cleanup
/// runs only on the Pending -> Cancelled edge, exactly once per payload.
enum class EventState {
    Pending,
    Invoking,
    Delivered,
    Cancelled,
};

/**
 * A non-owning value identity for one pending EventQueue event.
 *
 * The handle is a plain (serial, tick) pair. The serial comes from the
 * queue's monotonic source and is never reused, so a stale copy of a
 * consumed handle can never match a later event; serial 0 is reserved as
 * the null identity and never matches anything. Popping an event before
 * invoking its callback removes the node that carries the serial, so a
 * re-entrant cancellation of the in-flight callback safely returns false
 * and a callback can never delete the argument it is executing with.
 * EventQueue::cancel_event() removes a still-pending event from its
 * EventList and runs the supplied cancellation cleanup immediately.
 */
class EventHandle {
  public:
    EventHandle() noexcept = default;

    /// Non-null means the holder has not consumed the identity yet.
    /// Whether the event itself is still queued is the queue's business:
    /// a valid() handle whose event already fired simply fails to cancel.
    [[nodiscard]] bool valid() const noexcept;
    void reset() noexcept;

  private:
    EventHandle(uint64_t serial, EventTime event_time) noexcept;

    uint64_t serial_ = 0;
    EventTime event_time_ = 0;

    friend class QueuedEvent;
    friend class EventList;
    friend class EventQueue;
};

/// Called exactly once if a cancellable pending event is removed before its
/// normal callback invocation.  The normal callback owns its argument after
/// it starts, so cleanup is never called for a popped/invoking event.
using EventCancellationCallback = void (*)(CallbackArg);

/**
 * Event is a wrapper for a callback function and its argument.
 */
class Event {
  public:
    /**
     * Constructor.
     *
     * @param callback function pointer
     * @param callback_arg argument of the callback function
     */
    Event(Callback callback, CallbackArg callback_arg) noexcept;

    /**
     * Invoke the callback function.
     */
    void invoke_event() noexcept;

  private:
    Callback callback;
    CallbackArg callback_arg;
};

/**
 * EventQueue-internal event with a cancellation-aware payload contract.
 *
 * This intentionally remains distinct from Event: CallbackTracker stores
 * Event values by copy and retains the historical copy/value semantics.
 */
class QueuedEvent {
  public:
    /// Normal EventQueue events retain the legacy no-extra-allocation path
    /// (serial 0: never visible to a handle, never cancellable).
    QueuedEvent(Callback callback, CallbackArg callback_arg) noexcept;

    /// Cancellable events carry their identity and the four-state machine
    /// inline; no control block is allocated.
    QueuedEvent(Callback callback,
                CallbackArg callback_arg,
                EventCancellationCallback cancellation_callback,
                uint64_t serial) noexcept;
    ~QueuedEvent() noexcept;

    QueuedEvent(const QueuedEvent&) = delete;
    QueuedEvent& operator=(const QueuedEvent&) = delete;
    QueuedEvent(QueuedEvent&& other) noexcept;
    QueuedEvent& operator=(QueuedEvent&& other) noexcept;

    void invoke_event() noexcept;
    [[nodiscard]] EventHandle get_handle() const noexcept;
    [[nodiscard]] bool matches(const EventHandle& handle) const noexcept;
    [[nodiscard]] bool cancel_event() noexcept;

  private:
    // Identity + state of a cancellable event live inline. serial 0 marks a
    // plain event; the move operations void the source serial so a
    // moved-from node runs no teardown cleanup a second time.
    uint64_t serial_ = 0;
    EventState state_ = EventState::Pending;
    Callback callback_ = nullptr;
    CallbackArg callback_arg_ = nullptr;
    EventCancellationCallback cancellation_callback_ = nullptr;
};

}  // namespace NetworkAnalytical
