/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Type.h"
#include <cstdint>

namespace NetworkAnalytical {

/**
 * A non-owning identity for one pending EventQueue event.
 *
 * [B3, 2026-10-17] The handle is a (serial, event_time) pair instead of a
 * weak_ptr to an out-of-line control block: the queue assigns every
 * cancellable event a serial that is unique for the queue's lifetime, and
 * cancel_event() relocates the node by time bucket + serial scan. A stale
 * handle can therefore never match a reused slot (serials are never
 * recycled) and no pointer in the handle can dangle -- which is what the
 * old weak_ptr guaranteed, at one heap allocation per event.
 *
 * Handles are deliberately non-cancellable once popped: an event is removed
 * from its EventList before its callback runs, so a re-entrant cancel finds
 * no resident node and returns false -- a callback can never delete the
 * argument it is currently executing with.  EventQueue::cancel_event()
 * removes a still-pending event from its EventList and runs the supplied
 * cancellation cleanup immediately.
 */
class EventHandle {
  public:
    EventHandle() noexcept = default;

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
    /// (serial_ stays 0: they are invisible to the cancellation scan).
    QueuedEvent(Callback callback, CallbackArg callback_arg) noexcept;

    /// Cancellable events inline the former control-block payload -- zero
    /// allocation beyond the queue storage the event already occupies (B3,
    /// 2026-10-17). serial must be nonzero and unique per queue lifetime.
    QueuedEvent(Callback callback,
                CallbackArg callback_arg,
                EventCancellationCallback cancellation_callback,
                uint64_t serial) noexcept;
    ~QueuedEvent() noexcept;

    QueuedEvent(const QueuedEvent&) = delete;
    QueuedEvent& operator=(const QueuedEvent&) = delete;
    // Custom moves clear the source payload: a moved-from node must be
    // inert at teardown (no second cleanup run) and never deliver twice.
    QueuedEvent(QueuedEvent&& other) noexcept;
    QueuedEvent& operator=(QueuedEvent&& other) noexcept;

    void invoke_event() noexcept;
    [[nodiscard]] EventHandle get_handle() const noexcept;
    [[nodiscard]] bool matches(const EventHandle& handle) const noexcept;
    [[nodiscard]] bool cancel_event() noexcept;

    /// Identity of the callback this event will invoke (nullptr once the
    /// payload has been cancelled or delivered). Diagnostic classification
    /// only; never gates dispatch.
    [[nodiscard]] Callback get_callback() const noexcept;

  private:
    // Both flavors keep callback_/callback_arg_ inline: invoke pops them to
    // locals before calling, so the callback owns its argument from the
    // first instruction. cancellation_callback_ != nullptr marks a
    // cancellable payload that has not been consumed yet.
    Callback callback_ = nullptr;
    CallbackArg callback_arg_ = nullptr;
    EventCancellationCallback cancellation_callback_ = nullptr;
    uint64_t serial_ = 0;
};

}  // namespace NetworkAnalytical
