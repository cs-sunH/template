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

bool EventQueue::has_deferred_work() const noexcept {
    // defect-C fix (2026-08-16, synced from face-defectfix2-done; map
    // version): the same-tick deferred queue (a std::list<EventList>,
    // independent of the main map) drains only inside proceed(); deferred
    // events pended outside a proceed context are invisible to finished()
    // and would otherwise have no execution path.
    return !deferred_queue_.empty();
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

    // invoke events. in_invoke_ marks the only context in which a same-tick
    // schedule_event(current_time, ...) is legal (map version: schedule_event
    // try_emplaces into the EventList still held in the map and the merge
    // executes within this invoke pass); it is cleared before the tick-end
    // callback and the deferred drain so shared front-end code can route
    // same-tick events to schedule_event_deferred from those contexts.
    in_invoke_ = true;
    current_event_list.invoke_events();
    in_invoke_ = false;

    // drop processed event list (map version: erase by iterator; the
    // reference above must not be touched afterwards)
    event_queue.erase(current_event_list_it);

    // NEW (phase-1 execution-driven): tick-end closing.
    // The callback runs only after the erase above: at this point the main
    // queue no longer holds an EventList at current_time, so any
    // schedule_event(current_time, ...) issued from inside the callback (or
    // from the deferred drain below) would try_emplace a NEW EventList at
    // current_time, and the next proceed() would immediately trip the
    // strict-increase assert at the top of this function. Same-tick events
    // MUST use schedule_event_deferred(); only future events may use
    // schedule_event. (If the callback were invoked before the erase, such an
    // event would instead be merged into the already-invoked current list by
    // schedule_event's try_emplace merge and silently dropped with the erase
    // -- that ordering is forbidden.)
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

    auto event_list_it = event_queue.try_emplace(event_time, event_time).first;
    event_list_it->second.add_event(callback, callback_arg);
}
