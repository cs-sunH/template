/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/Event.h"
#include <cassert>
#include <utility>

using namespace NetworkAnalytical;

EventHandle::EventHandle(const uint64_t serial,
                         const EventTime event_time) noexcept
    : serial_(serial),
      event_time_(event_time) {}

bool EventHandle::valid() const noexcept {
    return serial_ != 0;
}

void EventHandle::reset() noexcept {
    serial_ = 0;
    event_time_ = 0;
}

Event::Event(const Callback callback, const CallbackArg callback_arg) noexcept
    : callback(callback),
      callback_arg(callback_arg) {
    assert(callback != nullptr);
}

void Event::invoke_event() noexcept {
    assert(callback != nullptr);
    (*callback)(callback_arg);
}

QueuedEvent::QueuedEvent(const Callback callback,
                         const CallbackArg callback_arg) noexcept
    : callback_(callback),
      callback_arg_(callback_arg) {
    assert(callback != nullptr);
}

QueuedEvent::QueuedEvent(
    const Callback callback,
    const CallbackArg callback_arg,
    const EventCancellationCallback cancellation_callback,
    const uint64_t serial) noexcept
    : serial_(serial),
      state_(EventState::Pending),
      callback_(callback),
      callback_arg_(callback_arg),
      cancellation_callback_(cancellation_callback) {
    assert(callback != nullptr);
    assert(serial != 0);  // 0 is reserved for plain (non-cancellable) events
}

QueuedEvent::QueuedEvent(QueuedEvent&& other) noexcept
    : serial_(other.serial_),
      state_(other.state_),
      callback_(other.callback_),
      callback_arg_(other.callback_arg_),
      cancellation_callback_(other.cancellation_callback_) {
    // Steal the payload and void the source identity, mirroring the null
    // control the old default shared_ptr move left behind: a moved-from
    // node must not run teardown cleanup a second time.
    other.serial_ = 0;
    other.callback_ = nullptr;
    other.callback_arg_ = nullptr;
    other.cancellation_callback_ = nullptr;
}

QueuedEvent& QueuedEvent::operator=(QueuedEvent&& other) noexcept {
    if (this != &other) {
        // Same contract as the old default shared_ptr move: the displaced
        // payload is dropped silently (no cancellation cleanup on overwrite).
        serial_ = other.serial_;
        state_ = other.state_;
        callback_ = other.callback_;
        callback_arg_ = other.callback_arg_;
        cancellation_callback_ = other.cancellation_callback_;
        other.serial_ = 0;
        other.callback_ = nullptr;
        other.callback_arg_ = nullptr;
        other.cancellation_callback_ = nullptr;
    }
    return *this;
}

QueuedEvent::~QueuedEvent() noexcept {
    // A queue/list teardown must not strand an owned cancellable payload.
    // Ordinary EventQueue entries retain their legacy non-owning semantics.
    if (serial_ != 0) {
        static_cast<void>(cancel_event());
    }
}

void QueuedEvent::invoke_event() noexcept {
    if (serial_ == 0) {
        assert(callback_ != nullptr);
        const Callback callback = callback_;
        const CallbackArg callback_arg = callback_arg_;
        callback_ = nullptr;
        callback_arg_ = nullptr;
        (*callback)(callback_arg);
        return;
    }

    assert(state_ == EventState::Pending);
    assert(callback_ != nullptr);

    // Pop ownership before invocation. A re-entrant cancellation observes
    // non-Pending state (and the node carrying this serial has already left
    // its EventList) and cannot free callback_arg while this frame uses it.
    const Callback callback = callback_;
    const CallbackArg callback_arg = callback_arg_;
    callback_ = nullptr;
    callback_arg_ = nullptr;
    cancellation_callback_ = nullptr;
    state_ = EventState::Invoking;
    (*callback)(callback_arg);
    state_ = EventState::Delivered;
}

EventHandle QueuedEvent::get_handle() const noexcept {
    if (serial_ == 0) {
        return EventHandle();
    }
    return EventHandle(serial_, 0);
}

bool QueuedEvent::matches(const EventHandle& handle) const noexcept {
    return serial_ != 0 && serial_ == handle.serial_;
}

bool QueuedEvent::cancel_event() noexcept {
    if (serial_ == 0 || state_ != EventState::Pending) {
        return false;
    }

    const EventCancellationCallback cancellation_callback =
        cancellation_callback_;
    const CallbackArg callback_arg = callback_arg_;
    callback_ = nullptr;
    callback_arg_ = nullptr;
    cancellation_callback_ = nullptr;
    state_ = EventState::Cancelled;
    if (cancellation_callback != nullptr) {
        cancellation_callback(callback_arg);
    }

    return true;
}
