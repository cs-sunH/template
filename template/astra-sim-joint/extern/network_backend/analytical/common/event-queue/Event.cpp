/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/Event.h"
#include <cassert>
#include <utility>

using namespace NetworkAnalytical;

// [B3, 2026-10-17] The out-of-line EventControl block (one make_shared per
// cancellable schedule, 61.75M/round under the joint online workload) is
// gone: its payload now lives inline in QueuedEvent and the handle identity
// is the queue-assigned serial. The former Invoking/Delivered states are
// structurally unreachable -- an invoked node is popped from its EventList
// before the callback runs, so no handle scan can observe it mid-invoke and
// a re-entrant cancel simply finds nothing.

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
    : callback_(callback),
      callback_arg_(callback_arg),
      cancellation_callback_(cancellation_callback),
      serial_(serial) {
    assert(callback != nullptr);
    assert(serial != 0);
}

QueuedEvent::~QueuedEvent() noexcept {
    // A queue/list teardown must not strand an owned cancellable payload.
    // Ordinary EventQueue entries retain their legacy non-owning semantics,
    // and a moved-from source is inert (its cleanup was cleared on move).
    if (cancellation_callback_ != nullptr) {
        static_cast<void>(cancel_event());
    }
}

QueuedEvent::QueuedEvent(QueuedEvent&& other) noexcept
    : callback_(other.callback_),
      callback_arg_(other.callback_arg_),
      cancellation_callback_(other.cancellation_callback_),
      serial_(other.serial_) {
    other.callback_ = nullptr;
    other.callback_arg_ = nullptr;
    other.cancellation_callback_ = nullptr;
    // other.serial_ is left as-is: only a node resident in an EventList is
    // reachable by a handle scan, and a moved-from node never is again.
}

QueuedEvent& QueuedEvent::operator=(QueuedEvent&& other) noexcept {
    if (this != &other) {
        callback_ = other.callback_;
        callback_arg_ = other.callback_arg_;
        cancellation_callback_ = other.cancellation_callback_;
        serial_ = other.serial_;
        other.callback_ = nullptr;
        other.callback_arg_ = nullptr;
        other.cancellation_callback_ = nullptr;
    }
    return *this;
}

void QueuedEvent::invoke_event() noexcept {
    assert(callback_ != nullptr);

    // Pop ownership before invocation. The node is already out of its
    // EventList, so a re-entrant cancellation cannot free callback_arg while
    // this stack frame uses it (four-layer contract, unchanged).
    const Callback callback = callback_;
    const CallbackArg callback_arg = callback_arg_;
    callback_ = nullptr;
    callback_arg_ = nullptr;
    cancellation_callback_ = nullptr;
    (*callback)(callback_arg);
}

EventHandle QueuedEvent::get_handle() const noexcept {
    if (serial_ == 0) {
        return EventHandle();
    }
    return EventHandle(serial_, 0);
}

Callback QueuedEvent::get_callback() const noexcept {
    return callback_;
}

bool QueuedEvent::matches(const EventHandle& handle) const noexcept {
    return serial_ != 0 && handle.serial_ == serial_;
}

bool QueuedEvent::cancel_event() noexcept {
    if (serial_ == 0 || callback_ == nullptr) {
        return false;
    }

    const EventCancellationCallback cancellation_callback =
        cancellation_callback_;
    const CallbackArg callback_arg = callback_arg_;
    callback_ = nullptr;
    callback_arg_ = nullptr;
    cancellation_callback_ = nullptr;
    if (cancellation_callback != nullptr) {
        cancellation_callback(callback_arg);
    }

    return true;
}
