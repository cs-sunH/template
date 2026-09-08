/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

// Regression coverage for immediate cancellation of superseded analytical
// events.  The first case is intentionally EventQueue-only: it nails payload
// reclamation, FIFO after a middle-node erase, and safe self-cancellation once
// the currently invoking event has been popped.  The second case drives a
// long flow plus recursively injected short flows.  Every completed short flow
// creates a new earlier wakeup than the long flow's prediction; stale alarms
// must therefore be removed instead of accumulating in the main queue.

#include "common/EventQueue.h"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidScheduler.h"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

[[noreturn]] void fail(const char* const message) {
    std::cerr << "Stale event cancellation test failure: " << message
              << std::endl;
    std::exit(EXIT_FAILURE);
}

void require(const bool condition, const char* const message) {
    if (!condition) {
        fail(message);
    }
}

struct OrderedPayload {
    int value;
    std::vector<int>* order;
    int* cleanup_count;
};

void deliver_ordered(void* const raw) noexcept {
    std::unique_ptr<OrderedPayload> payload(
        static_cast<OrderedPayload*>(raw));
    payload->order->push_back(payload->value);
}

void reclaim_ordered(void* const raw) noexcept {
    std::unique_ptr<OrderedPayload> payload(
        static_cast<OrderedPayload*>(raw));
    ++*payload->cleanup_count;
}

struct SelfCancellationPayload {
    EventQueue* queue;
    EventHandle* handle;
    std::vector<int>* order;
    bool* cancellation_result;
};

void deliver_and_cancel_self(void* const raw) noexcept {
    std::unique_ptr<SelfCancellationPayload> payload(
        static_cast<SelfCancellationPayload*>(raw));
    *payload->cancellation_result =
        payload->queue->cancel_event(*payload->handle);
    payload->order->push_back(1);
}

void reclaim_self(void* const raw) noexcept {
    delete static_cast<SelfCancellationPayload*>(raw);
}

void test_event_queue_cancellation() {
    EventQueue queue;
    std::vector<int> order;
    int cleanup_count = 0;
    bool cancellation_of_popped_event = true;

    EventHandle first_handle;
    first_handle = queue.schedule_event_cancellable(
        10,
        deliver_and_cancel_self,
        new SelfCancellationPayload{&queue, &first_handle, &order,
                                    &cancellation_of_popped_event},
        reclaim_self);
    EventHandle removed_handle = queue.schedule_event_cancellable(
        10, deliver_ordered,
        new OrderedPayload{2, &order, &cleanup_count}, reclaim_ordered);
    queue.schedule_event(10, deliver_ordered,
                         new OrderedPayload{3, &order, &cleanup_count});

    require(queue.scheduled_event_count() == 3,
            "three callbacks are resident before cancellation");
    require(queue.cancel_event(removed_handle),
            "pending middle event cancels successfully");
    require(!removed_handle.valid(), "successful cancellation consumes handle");
    require(cleanup_count == 1,
            "cancellation cleanup runs synchronously, before proceed");
    require(queue.scheduled_event_count() == 2,
            "cancellation removes the physical EventList node");

    queue.proceed();
    require(!cancellation_of_popped_event,
            "cancelling an already-popped callback safely returns false");
    require(order == std::vector<int>({1, 3}),
            "remaining same-tick callbacks retain FIFO order");
    require(queue.finished(), "no queue work remains after the two deliveries");

    EventHandle singleton_handle = queue.schedule_event_cancellable(
        20, deliver_ordered,
        new OrderedPayload{4, &order, &cleanup_count}, reclaim_ordered);
    require(queue.cancel_event(singleton_handle),
            "a singleton future EventList cancels successfully");
    require(cleanup_count == 2,
            "singleton cancellation also reclaims its payload immediately");
    require(queue.scheduled_event_count() == 0 && queue.finished(),
            "cancelling the last node erases the now-empty timestamp bucket");
}

constexpr uint64_t kMouseCount = 64;
constexpr ChunkSize kElephantBytes = 10000;
// Link accepts binary GB/s (2^30 B/s). This value is exactly 1 B/ns after
// the backend conversion, keeping the expected integer completion ticks clear.
constexpr double kOneBytePerNsGbps =
    1'000'000'000.0 / static_cast<double>(1ULL << 30);

struct FluidScenario;

struct CompletionContext {
    FluidScenario* scenario;
    bool is_elephant;
};

struct FluidScenario {
    std::shared_ptr<EventQueue> event_queue;
    FluidScheduler* scheduler = nullptr;
    std::shared_ptr<const FluidRoute> route;
    uint64_t mice_issued = 0;
    std::vector<EventTime> mouse_completion_times;
    EventTime elephant_completion_time = 0;
    bool elephant_completed = false;
    size_t peak_resident_events = 0;
};

void observe_residency(FluidScenario& scenario) noexcept {
    scenario.peak_resident_events = std::max(
        scenario.peak_resident_events,
        scenario.event_queue->scheduled_event_count());
}

void flow_completed(void* const raw) noexcept {
    std::unique_ptr<CompletionContext> context(
        static_cast<CompletionContext*>(raw));
    auto& scenario = *context->scenario;
    const auto now = scenario.event_queue->get_current_time();
    if (context->is_elephant) {
        scenario.elephant_completed = true;
        scenario.elephant_completion_time = now;
        observe_residency(scenario);
        return;
    }

    scenario.mouse_completion_times.push_back(now);
    if (scenario.mice_issued < kMouseCount) {
        ++scenario.mice_issued;
        scenario.scheduler->start_flow(
            1, scenario.route, flow_completed,
            new CompletionContext{&scenario, false});
    }
    observe_residency(scenario);
}

void seed_first_mouse(void* const raw) noexcept {
    auto& scenario = *static_cast<FluidScenario*>(raw);
    require(scenario.mice_issued == 0,
            "the seed callback starts exactly one first mouse");
    scenario.mice_issued = 1;
    scenario.scheduler->start_flow(
        1, scenario.route, flow_completed,
        new CompletionContext{&scenario, false});
    observe_residency(scenario);
}

void test_fluid_wakeup_replacement() {
    auto event_queue = std::make_shared<EventQueue>();
    std::vector<std::shared_ptr<const Link>> links;
    links.push_back(std::make_shared<Link>(0, kOneBytePerNsGbps, 0.0));
    FluidScheduler scheduler(event_queue, links, kMouseCount + 1,
                             kMouseCount + 1, 1'000'000);

    auto mutable_route = std::make_shared<FluidRoute>();
    mutable_route->link_ids = {0};
    mutable_route->propagation_latency_ns = 0;

    FluidScenario scenario;
    scenario.event_queue = event_queue;
    scenario.scheduler = &scheduler;
    scenario.route = mutable_route;

    // Install the elephant directly at t=0, then add its first competing mouse
    // at t=1 from a physical callback.  Subsequent mouse completions re-enter
    // start_flow at their same tick, so each flush replaces a far elephant
    // prediction with one short-flow wakeup.
    scheduler.start_flow(kElephantBytes, scenario.route, flow_completed,
                         new CompletionContext{&scenario, true});
    scheduler.flush_pending_starts();
    scheduler.mark_event_loop_started();
    event_queue->schedule_event(1, seed_first_mouse, &scenario);
    observe_residency(scenario);

    uint64_t guard = 0;
    while (!event_queue->finished()) {
        observe_residency(scenario);
        event_queue->proceed();
        observe_residency(scenario);
        require(++guard < 100000,
                "repeated wakeup replacement keeps the event loop draining");
    }

    require(scenario.mice_issued == kMouseCount &&
                scenario.mouse_completion_times.size() == kMouseCount,
            "every recursively injected mouse completes exactly once");
    for (uint64_t index = 0; index < kMouseCount; ++index) {
        require(scenario.mouse_completion_times[index] == 3 + 2 * index,
                "mouse completion tick preserves the fluid timing contract");
    }
    if (!scenario.elephant_completed ||
        scenario.elephant_completion_time !=
            static_cast<EventTime>(kElephantBytes + kMouseCount)) {
        std::cerr << "elephant expected=" << kElephantBytes + kMouseCount
                  << " actual=" << scenario.elephant_completion_time
                  << " completed=" << scenario.elephant_completed << std::endl;
        fail("elephant completion tick remains unchanged after wakeup replacement");
    }
    require(scheduler.get_total_started_flows() == kMouseCount + 1 &&
                scheduler.get_total_completed_flows() == kMouseCount + 1 &&
                scheduler.get_active_flow_count() == 0 &&
                scheduler.get_active_route_memberships() == 0,
            "fluid scheduler delivers every flow once and leaves no active state");
    require(scenario.peak_resident_events <= 3,
            "superseded wakeups stay bounded instead of accumulating");
}

}  // namespace

int main() {
    test_event_queue_cancellation();
    test_fluid_wakeup_replacement();
    std::cout << "Stale event cancellation regression: PASS" << std::endl;
    return EXIT_SUCCESS;
}
