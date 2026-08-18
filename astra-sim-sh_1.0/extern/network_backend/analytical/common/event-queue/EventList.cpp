/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventList.h"
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <chrono>

using namespace NetworkAnalytical;

EventList::EventList(const EventTime event_time) noexcept : event_time(event_time) {
    assert(event_time >= 0);

    // create an empty event list
    events = std::list<Event>();
}

EventTime EventList::get_event_time() const noexcept {
    return event_time;
}

void EventList::add_event(const Callback callback, const CallbackArg callback_arg) noexcept {
    assert(callback != nullptr);

    // add the event to the event list
    events.emplace_back(callback, callback_arg);
}

void EventList::invoke_events() noexcept {
    // invoke all events in the event list
    uint64_t n = 0;
    auto loop_t0 = std::chrono::steady_clock::now();
    bool announced = false;
    while (!events.empty()) {
        if (std::getenv("SH10_DEBUG_ISSUE") && !announced) {
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - loop_t0).count();
            if (dt > 5000) {
                announced = true;
                std::fprintf(stderr,
                             "[dbg-stall] entering possibly-stuck cb=%p "
                             "t=%llu remaining=%zu\n",
                             events.front().invoke_event_ptr(),
                             (unsigned long long)event_time,
                             events.size());
            }
        }
        if (std::getenv("SH10_DEBUG_ISSUE")) {
            std::fprintf(stderr, "[dbg-enter] cb=%p t=%llu\n",
                         events.front().invoke_event_ptr(),
                         (unsigned long long)event_time);
        }
        if (std::getenv("SH10_DEBUG_ISSUE")) {
            auto t0 = std::chrono::steady_clock::now();
            events.front().invoke_event();
            auto dt = std::chrono::duration_cast<std::chrono::milliseconds>(
                std::chrono::steady_clock::now() - t0).count();
            if (dt > 2000) {
                std::fprintf(stderr,
                             "[dbg-slowcb] ms=%lld t=%llu cb=%p\n",
                             (long long)dt,
                             (unsigned long long)event_time,
                             (void*)events.front().invoke_event_ptr());
            }
        } else {
            events.front().invoke_event();
        }
        events.pop_front();
        if (std::getenv("SH10_DEBUG_ISSUE") && (++n & 0xFFFFF) == 0) {
            std::fprintf(stderr, "[dbg-inv] n=%llu t=%llu sz=%zu\n",
                         (unsigned long long)n,
                         (unsigned long long)event_time,
                         events.size());
        }
    }
}
