/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventQueue.h"
#include <cassert>

using namespace NetworkAnalytical;

EventQueue::EventQueue() noexcept : current_time(0) {
    // create empty event queue
    event_queue = std::list<EventList>();
}

EventTime EventQueue::get_current_time() const noexcept {
    return current_time;
}

bool EventQueue::finished() const noexcept {
    // check whether event queue is empty
    return event_queue.empty();
}

bool EventQueue::has_deferred_work() const noexcept {
    // defect-C fix (2026-08-16): the same-tick deferred queue drains only
    // inside proceed(); deferred events pended outside a proceed context are
    // invisible to finished() and would otherwise have no execution path.
    return !deferred_queue_.empty();
}

void EventQueue::proceed() noexcept {
    // to proceed, next event should exist
    assert(!finished());

    // proceed to the next event time
    auto& current_event_list = event_queue.front();

    // check the validity and update current time
    assert(current_event_list.get_event_time() > current_time);
    current_time = current_event_list.get_event_time();

    // invoke events. in_invoke_ marks the only context in which a same-tick
    // schedule_event(current_time, ...) is legal (it merges into the EventList
    // currently being invoked); it is cleared before the tick-end callback and
    // the deferred drain so shared front-end code can route same-tick events
    // to schedule_event_deferred from those contexts.
    in_invoke_ = true;
    current_event_list.invoke_events();
    in_invoke_ = false;

    // drop processed event list
    event_queue.pop_front();

    // NEW (phase-1 execution-driven): tick-end closing.
    // The callback runs only after the pop above: at this point the main queue
    // no longer holds an EventList at current_time, so any
    // schedule_event(current_time, ...) issued from inside the callback (or
    // from the deferred drain below) would create a NEW EventList at
    // current_time, and the next proceed() would immediately trip the
    // strict-increase assert at the top of this function. Same-tick events
    // MUST use schedule_event_deferred(); only future events may use
    // schedule_event. (If the callback were invoked before the pop, such an
    // event would instead be merged into the already-invoked current list by
    // schedule_event's same-time merge and silently dropped with the pop --
    // that ordering is forbidden.)
    if (tick_end_cb_ != nullptr) {
        tick_end_cb_(tick_end_arg_);
    }

    // NEW: drain same-tick deferred events (post-commit produced nodes/events).
    // The same hard rule applies here: no schedule_event(current_time, ...)
    // from within a deferred handler. Deferred handlers may append further
    // deferred events; the while loop picks them up in the same drain pass.
    while (!deferred_queue_.empty()) {
        EventList deferred = std::move(deferred_queue_.front());
        deferred_queue_.pop_front();
        deferred.invoke_events();
    }
}

void EventQueue::set_tick_end_callback(const Callback callback, const CallbackArg arg) noexcept {
    tick_end_cb_ = callback;
    tick_end_arg_ = arg;
}

bool EventQueue::in_invoke_context() const noexcept {
    return in_invoke_;
}

void EventQueue::schedule_event_deferred(const Callback callback, const CallbackArg arg) noexcept {
    // deferred events belong to the current tick: construct an EventList at
    // current_time; it is executed by the deferred drain of the current
    // proceed() (or of the next one if called outside a proceed context).
    deferred_queue_.emplace_back(current_time).add_event(callback, arg);
}

void EventQueue::schedule_event(const EventTime event_time,
                                const Callback callback,
                                const CallbackArg callback_arg) noexcept {
    // time should be at least larger than current time
    assert(event_time >= current_time);

    // find the entry to insert event
    auto event_list_it = event_queue.begin();
    while (event_list_it != event_queue.end() && event_list_it->get_event_time() < event_time) {
        event_list_it++;
    }

    // There can be three scenarios:
    // (1) event list matching with event_time is found
    // (2) there's no event list matching with event_time
    //   (2-1) the event_time requested is
    //   larger than the largest event time scheduled
    //   (2-2) the event_time requested is
    //   smaller than the largest event time scheduled
    // for both (2-1) or (2-2), a new event should be created
    if (event_list_it == event_queue.end() || event_time < event_list_it->get_event_time()) {
        // insert new event_list
        event_list_it = event_queue.insert(event_list_it, EventList(event_time));
    }

    // now, whether (1) or (2), the entry to insert the event is found
    // add event to event_list
    event_list_it->add_event(callback, callback_arg);
}
