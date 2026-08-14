/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventQueue.h"
#include <cassert>

using namespace NetworkAnalytical;

EventQueue::EventQueue() noexcept : current_time(0), event_queue() {}

EventTime EventQueue::get_current_time() const noexcept {
    return current_time;
}

bool EventQueue::finished() const noexcept {
    // check whether event queue is empty
    return event_queue.empty();
}

void EventQueue::proceed() noexcept {
    // to proceed, next event should exist
    assert(!finished());

    // proceed to the next event time
    auto current_event_list_it = event_queue.begin();
    auto& current_event_list = current_event_list_it->second;

    // check the validity and update current time
    assert(current_event_list.get_event_time() > current_time);
    current_time = current_event_list.get_event_time();

    // invoke events
    current_event_list.invoke_events();

    // drop processed event list
    event_queue.erase(current_event_list_it);
}

void EventQueue::schedule_event(const EventTime event_time,
                                const Callback callback,
                                const CallbackArg callback_arg) noexcept {
    // time should be at least larger than current time
    assert(event_time >= current_time);

    auto event_list_it = event_queue.try_emplace(event_time, event_time).first;
    event_list_it->second.add_event(callback, callback_arg);
}
