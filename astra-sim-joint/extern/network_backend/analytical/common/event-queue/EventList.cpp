/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventList.h"
#include <cassert>
#include <utility>

#ifdef JOINT_V7_EVENT_PROFILE
#include "common/EventProfile.h"
using namespace NetworkAnalytical::EventProfile;
#endif

using namespace NetworkAnalytical;

EventList::EventList(const EventTime event_time) noexcept : event_time(event_time) {

}

EventTime EventList::get_event_time() const noexcept {
    return event_time;
}

void EventList::add_event(const Callback callback,
                          const CallbackArg callback_arg) noexcept {
    assert(callback != nullptr);

    // Preserve the legacy fast path: ordinary events have no handle and no
    // cancellation cleanup, so they do not allocate a control block.
    if (!first_event.has_value() && overflow_events.empty()) {
        first_event.emplace(callback, callback_arg);
    } else {
        overflow_events.emplace_back(callback, callback_arg);
    }
}

EventHandle EventList::add_cancellable_event(
    const Callback callback,
    const CallbackArg callback_arg,
    const EventCancellationCallback cancellation_callback) noexcept {
    assert(callback != nullptr);

    // Keep the first event inline. If an event is appended by a callback while
    // older overflow events remain, it must go to the tail to preserve FIFO.
    if (!first_event.has_value() && overflow_events.empty()) {
        first_event.emplace(callback, callback_arg,
                            cancellation_callback);
        auto handle = first_event->get_handle();
        handle.event_time_ = event_time;
        return handle;
    } else {
        overflow_events.emplace_back(callback, callback_arg,
                                     cancellation_callback);
        auto handle = overflow_events.back().get_handle();
        handle.event_time_ = event_time;
        return handle;
    }
}

bool EventList::cancel_event(const EventHandle& handle) noexcept {
    if (first_event.has_value() && first_event->matches(handle)) {
        if (!first_event->cancel_event()) {
            return false;
        }
        first_event.reset();
        return true;
    }

    for (auto event_it = overflow_events.begin();
         event_it != overflow_events.end(); ++event_it) {
        if (!event_it->matches(handle)) {
            continue;
        }
        if (!event_it->cancel_event()) {
            return false;
        }
        overflow_events.erase(event_it);
        return true;
    }
    return false;
}

bool EventList::empty() const noexcept {
    return !first_event.has_value() && overflow_events.empty();
}

size_t EventList::size() const noexcept {
    return (first_event.has_value() ? 1U : 0U) + overflow_events.size();
}

void EventList::invoke_events() noexcept {
    // Pop/copy before invoking: the callback may append another same-time
    // event to this EventList. The loop observes it in exact FIFO order.
    while (first_event.has_value() || !overflow_events.empty()) {
#ifdef JOINT_V7_EVENT_PROFILE
        // Diagnostic-only classification bracket (v7 attribution): one label
        // lookup + two clock reads per event. Compiled out in production.
        const Callback profile_cb = first_event.has_value()
            ? first_event->get_callback()
            : overflow_events.front().get_callback();
        const char* profile_label = classify(profile_cb);
        Slot* profile_slot = profile_label != nullptr
            ? find_slot(state().main, profile_label)
            : nullptr;
        const uint64_t profile_t0 = now_ns();
#endif
        if (first_event.has_value()) {
            QueuedEvent event = std::move(*first_event);
            first_event.reset();
            event.invoke_event();
        } else {
            QueuedEvent event = std::move(overflow_events.front());
            overflow_events.pop_front();
            event.invoke_event();
        }
#ifdef JOINT_V7_EVENT_PROFILE
        if (profile_label != nullptr) {
            close_slot(profile_slot, nullptr, profile_t0);
        } else {
            state().unclassified_main.ns += now_ns() - profile_t0;
            state().unclassified_main.calls += 1;
        }
#endif
    }
}
