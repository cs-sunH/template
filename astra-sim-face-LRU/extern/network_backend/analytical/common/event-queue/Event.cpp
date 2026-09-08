/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/Event.h"
#include <cassert>
#include <utility>

using namespace NetworkAnalytical;

namespace NetworkAnalytical {

enum class EventState {
    Pending,
    Invoking,
    Delivered,
    Cancelled,
};

class EventControl {
  public:
    EventControl(const Callback callback,
                 const CallbackArg callback_arg,
                 const EventCancellationCallback cancellation_callback) noexcept
        : callback(callback),
          callback_arg(callback_arg),
          cancellation_callback(cancellation_callback),
          state(EventState::Pending) {}

    Callback callback;
    CallbackArg callback_arg;
    EventCancellationCallback cancellation_callback;
    EventState state;
};

}  // namespace NetworkAnalytical

EventHandle::EventHandle(std::weak_ptr<EventControl> control,
                         const EventTime event_time) noexcept
    : control_(std::move(control)),
      event_time_(event_time) {}

bool EventHandle::valid() const noexcept {
    return !control_.expired();
}

void EventHandle::reset() noexcept {
    control_.reset();
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

std::pair<Callback, CallbackArg> Event::get_handler_arg() const noexcept {
    assert(callback != nullptr);
    return {callback, callback_arg};
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
    const EventCancellationCallback cancellation_callback) noexcept
    : control_(std::make_shared<EventControl>(callback, callback_arg,
                                              cancellation_callback)) {
    assert(callback != nullptr);
}

QueuedEvent::~QueuedEvent() noexcept {
    // A queue/list teardown must not strand an owned cancellable payload.
    // Ordinary EventQueue entries retain their legacy non-owning semantics.
    if (control_ != nullptr) {
        static_cast<void>(cancel_event());
    }
}

void QueuedEvent::invoke_event() noexcept {
    if (control_ == nullptr) {
        assert(callback_ != nullptr);
        const Callback callback = callback_;
        const CallbackArg callback_arg = callback_arg_;
        callback_ = nullptr;
        callback_arg_ = nullptr;
        (*callback)(callback_arg);
        return;
    }

    assert(control_->state == EventState::Pending);
    assert(control_->callback != nullptr);

    // Pop ownership before invocation. A re-entrant cancellation observes
    // Invoking and cannot free callback_arg while this stack frame uses it.
    const Callback callback = control_->callback;
    const CallbackArg callback_arg = control_->callback_arg;
    control_->callback = nullptr;
    control_->callback_arg = nullptr;
    control_->cancellation_callback = nullptr;
    control_->state = EventState::Invoking;
    (*callback)(callback_arg);
    control_->state = EventState::Delivered;
}

EventHandle QueuedEvent::get_handle() const noexcept {
    if (control_ == nullptr) {
        return EventHandle();
    }
    return EventHandle(control_, 0);
}

bool QueuedEvent::matches(const EventHandle& handle) const noexcept {
    return control_ != nullptr && handle.control_.lock() == control_;
}

bool QueuedEvent::cancel_event() noexcept {
    if (control_ == nullptr || control_->state != EventState::Pending) {
        return false;
    }

    const EventCancellationCallback cancellation_callback =
        control_->cancellation_callback;
    const CallbackArg callback_arg = control_->callback_arg;
    control_->callback = nullptr;
    control_->callback_arg = nullptr;
    control_->cancellation_callback = nullptr;
    control_->state = EventState::Cancelled;
    if (cancellation_callback != nullptr) {
        cancellation_callback(callback_arg);
    }

    return true;
}
