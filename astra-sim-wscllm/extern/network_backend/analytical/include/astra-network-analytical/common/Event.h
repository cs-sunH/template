/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include <memory>

namespace NetworkAnalytical {

class EventControl;

/**
 * A non-owning identity for one pending EventQueue event.
 *
 * Handles are deliberately weak: popping an event before invoking its
 * callback makes its handle non-cancellable, so a callback can never delete
 * the argument it is currently executing with.  EventQueue::cancel_event()
 * removes a still-pending event from its EventList and runs the supplied
 * cancellation cleanup immediately.
 */
class EventHandle {
  public:
    EventHandle() noexcept = default;

    [[nodiscard]] bool valid() const noexcept;
    void reset() noexcept;

  private:
    EventHandle(std::weak_ptr<EventControl> control,
                EventTime event_time) noexcept;

    std::weak_ptr<EventControl> control_;
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
    /// Normal EventQueue events retain the legacy no-extra-allocation path.
    QueuedEvent(Callback callback, CallbackArg callback_arg) noexcept;

    /// Cancellable events allocate a small control block for their weak handle.
    QueuedEvent(Callback callback,
                CallbackArg callback_arg,
                EventCancellationCallback cancellation_callback) noexcept;
    ~QueuedEvent() noexcept;

    QueuedEvent(const QueuedEvent&) = delete;
    QueuedEvent& operator=(const QueuedEvent&) = delete;
    QueuedEvent(QueuedEvent&&) noexcept = default;
    QueuedEvent& operator=(QueuedEvent&&) noexcept = default;

    void invoke_event() noexcept;
    [[nodiscard]] EventHandle get_handle() const noexcept;
    [[nodiscard]] bool matches(const EventHandle& handle) const noexcept;
    [[nodiscard]] bool cancel_event() noexcept;

  private:
    // Normal events use these two legacy fields directly. Cancellable events
    // keep their payload in control_ so an EventHandle can outlive a list-node
    // move without ever owning the callback argument.
    Callback callback_ = nullptr;
    CallbackArg callback_arg_ = nullptr;
    std::shared_ptr<EventControl> control_;
};

}  // namespace NetworkAnalytical
