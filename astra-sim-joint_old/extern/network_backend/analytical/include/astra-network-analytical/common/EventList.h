/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/Event.h"
#include "common/Type.h"
#include <cstddef>
#include <list>
#include <optional>

namespace NetworkAnalytical {

/**
 * EventList encapsulates a number of Events along with its event time.
 */
class EventList {
  public:
    /**
     * Constructor.
     *
     * @param event_time event time of the event list
     */
    explicit EventList(EventTime event_time) noexcept;

    /**
     * Get the registered event time.
     *
     * @return event time
     */
    [[nodiscard]] EventTime get_event_time() const noexcept;

    /**
     * Register an event into the event list.
     *
     * @param callback callback function pointer
     * @param callback_arg argument of the callback function
     */
    void add_event(Callback callback, CallbackArg callback_arg) noexcept;

    /// Add an event whose argument is reclaimed by cancellation_callback if
    /// this list removes it before invocation.
    [[nodiscard]] EventHandle add_cancellable_event(
        Callback callback,
        CallbackArg callback_arg,
        EventCancellationCallback cancellation_callback) noexcept;

    /// Remove a pending event by handle.  The current event is popped before
    /// callback invocation, so attempting to cancel it while re-entrant is a
    /// safe false return rather than a use-after-free.
    [[nodiscard]] bool cancel_event(const EventHandle& handle) noexcept;

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] size_t size() const noexcept;

    /**
     * Invoke all events in the event list.
     */
    void invoke_events() noexcept;

  private:
    /// event time of the event list
    EventTime event_time;

    /// The overwhelmingly common case is one event per timestamp. Keep that
    /// first event inline so it needs no list-node allocation; only same-time
    /// collisions spill into the FIFO below.
    std::optional<QueuedEvent> first_event;
    std::list<QueuedEvent> overflow_events;
};

}  // namespace NetworkAnalytical
