/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventQueue.h"
#include <cassert>
#include <cstdlib>
#include <iostream>

using namespace NetworkAnalytical;

EventQueue::EventQueue() noexcept : current_time(0), event_queue() {}

EventTime EventQueue::get_current_time() const noexcept {
    return current_time;
}

bool EventQueue::finished() const noexcept {
    // check whether event queue is empty
    return event_queue.empty();
}

size_t EventQueue::scheduled_event_count() const noexcept {
    size_t count = 0;
    for (const auto& [event_time, events] : event_queue) {
        (void)event_time;
        count += events.size();
    }
    return count;
}

bool EventQueue::has_deferred_work() const noexcept {
    // defect-C fix (2026-08-16, 同步自 face 0049ef5): the same-tick
    // deferred queue drains only inside proceed(); deferred events pended
    // outside a proceed context are invisible to finished() (which sees
    // only the main map) and would otherwise have no execution path.
    return !deferred_queue_.empty();
}

void EventQueue::proceed() noexcept {
    // to proceed, next event should exist
    assert(!finished());

    // proceed to the next event time
    auto current_event_list_it = event_queue.begin();
    auto& current_event_list = current_event_list_it->second;

    // check the validity and update current time. Fail closed in every
    // build type (the former assert is compiled out of Release): a
    // non-increasing event time here means a same-tick schedule_event()
    // escaped an invoke/tick-end/deferred context -- see the tick-end
    // comment below -- and would silently rewind simulation time.
    if (current_event_list.get_event_time() <= current_time) {
        std::cerr << "[Error] (network/analytical) event queue time must "
                  << "strictly increase: "
                  << current_event_list.get_event_time()
                  << " <= current_time " << current_time << std::endl;
        std::exit(-1);
    }
    current_time = current_event_list.get_event_time();

    // invoke events. in_invoke_ marks the only context in which a same-tick
    // schedule_event(current_time, ...) is legal (it merges into the EventList
    // currently being invoked via try_emplace); it is cleared before the
    // tick-end callback and the deferred drain so shared front-end code can
    // route same-tick events to schedule_event_deferred from those contexts.
    in_invoke_ = true;
    current_event_list.invoke_events();
    in_invoke_ = false;

    // drop processed event list
    event_queue.erase(current_event_list_it);

    // NEW (phase-1 execution-driven, map family): tick-end closing.
    // The callback runs only after the erase above: at this point the map no
    // longer holds an EventList at current_time, so any
    // schedule_event(current_time, ...) issued from inside the callback (or
    // from the deferred drain below) would try_emplace a NEW EventList at
    // current_time, and the next proceed() would immediately trip the
    // strict-increase fail-closed guard at the top of this function.
    // Same-tick events MUST use schedule_event_deferred(); only future
    // events may use schedule_event. (If the callback were invoked before
    // the erase, such an
    // event would instead be merged into the already-invoked current list by
    // schedule_event's same-time try_emplace and silently dropped with the
    // erase -- that ordering is forbidden.)
    // Note: the erase above invalidated current_event_list; it must not be
    // touched again past this point.
    if (tick_end_cb_ != nullptr) {
        tick_end_cb_(tick_end_arg_);
    }

    // NEW: drain same-tick deferred events (post-commit produced nodes/events).
    // The same hard rule applies here: no schedule_event(current_time, ...)
    // from within a deferred handler. Deferred handlers may append further
    // deferred events; the while loop picks them up in the same drain pass.
    while (!deferred_queue_.empty()) {
        // Copy before invocation: a nested callback may grow/reallocate the
        // deque, while the local pair remains stable.
        const auto deferred = deferred_queue_.front();
        deferred_queue_.pop_front();
        deferred.first(deferred.second);
    }
}

void EventQueue::set_tick_end_callback(const Callback callback, const CallbackArg callback_arg) noexcept {
    tick_end_cb_ = callback;
    tick_end_arg_ = callback_arg;
}

bool EventQueue::in_invoke_context() const noexcept {
    return in_invoke_;
}

void EventQueue::schedule_event_deferred(const Callback callback, const CallbackArg callback_arg) noexcept {
    assert(callback != nullptr);
    // Deferred events belong to the current tick. The compact pair FIFO is
    // executed by the current proceed() drain (or the next one when scheduled
    // outside a proceed context), without a per-event list-node allocation.
    deferred_queue_.emplace_back(callback, callback_arg);
}

void EventQueue::schedule_event(const EventTime event_time,
                                const Callback callback,
                                const CallbackArg callback_arg) noexcept {
    // time should be at least larger than current time. Fail closed in
    // every build type (the former assert is compiled out of Release).
    if (event_time < current_time) {
        std::cerr << "[Error] (network/analytical) schedule_event time "
                  << event_time << " is earlier than current_time "
                  << current_time << std::endl;
        std::exit(-1);
    }

    auto event_list_it = event_queue.try_emplace(event_time, event_time).first;
    event_list_it->second.add_event(callback, callback_arg);
}

EventHandle EventQueue::schedule_event_cancellable(
    const EventTime event_time,
    const Callback callback,
    const CallbackArg callback_arg,
    const EventCancellationCallback cancellation_callback) noexcept {
    // Fail closed in every build type (the former assert is compiled out
    // of Release).
    if (event_time < current_time) {
        std::cerr << "[Error] (network/analytical) "
                  << "schedule_event_cancellable time " << event_time
                  << " is earlier than current_time " << current_time
                  << std::endl;
        std::exit(-1);
    }
    assert(callback != nullptr);

    auto event_list_it = event_queue.try_emplace(event_time, event_time).first;
    return event_list_it->second.add_cancellable_event(
        callback, callback_arg, cancellation_callback);
}

bool EventQueue::cancel_event(EventHandle& handle) noexcept {
    if (!handle.valid()) {
        handle.reset();
        return false;
    }

    const auto event_list_it = event_queue.find(handle.event_time_);
    if (event_list_it == event_queue.end()) {
        handle.reset();
        return false;
    }

    const bool cancelled = event_list_it->second.cancel_event(handle);
    handle.reset();
    if (cancelled && event_list_it->second.empty() &&
        !(in_invoke_ && event_list_it->first == current_time)) {
        // Do not erase the EventList currently being invoked: proceed() owns
        // its iterator until the pass ends. A future empty bucket has no
        // payload or list node left, so drop it immediately.
        event_queue.erase(event_list_it);
    }
    return cancelled;
}
