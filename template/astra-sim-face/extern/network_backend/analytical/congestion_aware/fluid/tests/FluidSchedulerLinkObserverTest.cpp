/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "common/EventQueue.h"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidScheduler.h"
#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <vector>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

namespace {

struct Record {
    uint64_t bucket;
    LinkId link_id;
    uint64_t bytes;
};

[[noreturn]] void fail(const char* const message) {
    std::cerr << "FluidScheduler link observer test failure: " << message << std::endl;
    std::exit(EXIT_FAILURE);
}

void require(const bool condition, const char* const message) {
    if (!condition) {
        fail(message);
    }
}

bool same_records(const std::vector<Record>& lhs, const std::vector<Record>& rhs) {
    if (lhs.size() != rhs.size()) {
        return false;
    }
    for (size_t index = 0; index < lhs.size(); ++index) {
        if (lhs[index].bucket != rhs[index].bucket ||
            lhs[index].link_id != rhs[index].link_id ||
            lhs[index].bytes != rhs[index].bytes) {
            return false;
        }
    }
    return true;
}

bool append_record(void* const context, const uint64_t bucket,
                   const LinkId link_id, const uint64_t bytes) {
    static_cast<std::vector<Record>*>(context)->push_back({bucket, link_id, bytes});
    return true;
}

/// Deliberately retains the original row-vector integration semantics so this
/// fixture compares the streaming observer against the historical result.
struct LegacyObserver {
    explicit LegacyObserver(const size_t link_count, const uint64_t bucket_ns)
        : bucket_ns(bucket_ns),
          carry(link_count, 0.0L),
          total_bytes(link_count, 0),
          active_ns(link_count, 0),
          bucket_bytes(link_count) {}

    void integrate(const EventTime now, const std::vector<long double>& rates) {
        require(rates.size() == carry.size(), "legacy rate count");
        EventTime segment_start = last_tick;
        while (segment_start < now) {
            const auto bucket = static_cast<uint64_t>(segment_start / bucket_ns);
            const auto bucket_end = static_cast<EventTime>(bucket + 1) * bucket_ns;
            const auto segment_end = now < bucket_end ? now : bucket_end;
            require(segment_end > segment_start, "legacy integration segment");
            const auto dt = segment_end - segment_start;
            for (size_t link = 0; link < rates.size(); ++link) {
                if (rates[link] <= 0.0L) {
                    continue;
                }
                carry[link] += rates[link] * static_cast<long double>(dt);
                const auto whole = static_cast<uint64_t>(carry[link]);
                if (whole > 0) {
                    auto& row = bucket_bytes[link];
                    if (row.size() <= bucket) {
                        row.resize(bucket + 1, 0);
                    }
                    row[bucket] += whole;
                    total_bytes[link] += whole;
                    carry[link] -= static_cast<long double>(whole);
                }
                active_ns[link] += dt;
            }
            segment_start = segment_end;
        }
        last_tick = now;
    }

    std::vector<Record> records() const {
        std::vector<Record> result;
        for (size_t link = 0; link < bucket_bytes.size(); ++link) {
            for (size_t bucket = 0; bucket < bucket_bytes[link].size(); ++bucket) {
                if (bucket_bytes[link][bucket] != 0) {
                    result.push_back(
                        {static_cast<uint64_t>(bucket), static_cast<LinkId>(link),
                         bucket_bytes[link][bucket]});
                }
            }
        }
        std::sort(result.begin(), result.end(), [](const Record& lhs, const Record& rhs) {
            return lhs.bucket != rhs.bucket ? lhs.bucket < rhs.bucket
                                             : lhs.link_id < rhs.link_id;
        });
        return result;
    }

    uint64_t bucket_ns;
    EventTime last_tick = 0;
    std::vector<long double> carry;
    std::vector<uint64_t> total_bytes;
    std::vector<uint64_t> active_ns;
    std::vector<std::vector<uint64_t>> bucket_bytes;
};

void no_op(void*) {}

struct DelayedStart {
    FluidScheduler* scheduler;
    std::shared_ptr<const FluidRoute> route;
    uint64_t bytes;
};

void start_delayed_flow(void* const context) {
    std::unique_ptr<DelayedStart> delayed(static_cast<DelayedStart*>(context));
    delayed->scheduler->start_flow(delayed->bytes, delayed->route, no_op, nullptr);
}

struct CompletionCounter {
    uint64_t count = 0;
};

void count_completion(void* const context) {
    ++static_cast<CompletionCounter*>(context)->count;
}

struct ReentrantTailCompletion {
    FluidScheduler* scheduler;
    std::shared_ptr<const FluidRoute> zero_tail_route;
    CompletionCounter* completions;
    bool started = false;
};

void complete_tail_and_start_one_flow(void* const context) {
    auto* const completion = static_cast<ReentrantTailCompletion*>(context);
    ++completion->completions->count;
    if (!completion->started) {
        completion->started = true;
        completion->scheduler->start_flow(1, completion->zero_tail_route,
                                          count_completion,
                                          completion->completions);
    }
}

struct TailRehashBurst {
    FluidScheduler* scheduler;
    std::shared_ptr<const FluidRoute> route;
    CompletionCounter* completions;
};

void start_tail_rehash_burst(void* const context) {
    std::unique_ptr<TailRehashBurst> burst(static_cast<TailRehashBurst*>(context));
    constexpr uint64_t kBurstFlows = 192;
    for (uint64_t index = 0; index < kBurstFlows; ++index) {
        burst->scheduler->start_flow(1, burst->route, count_completion,
                                     burst->completions);
    }
}

std::vector<std::shared_ptr<const Link>> two_links() {
    std::vector<std::shared_ptr<const Link>> links;
    links.push_back(std::make_shared<Link>(0, 1.0, 0.0));
    links.push_back(std::make_shared<Link>(1, 0.5, 0.0));
    return links;
}

void run_tail_rehash_lifetime_test() {
    constexpr uint64_t kBurstFlows = 192;
    auto event_queue = std::make_shared<EventQueue>();
    std::vector<std::shared_ptr<const Link>> links;
    links.push_back(std::make_shared<Link>(0, 1.0, 0.0));
    FluidScheduler scheduler(event_queue, links, 512, 512, 1024);

    auto long_tail_route = std::make_shared<FluidRoute>();
    long_tail_route->link_ids = {0};
    long_tail_route->propagation_latency_ns = 1000;
    auto burst_route = std::make_shared<FluidRoute>();
    burst_route->link_ids = {0};
    burst_route->propagation_latency_ns = 0;

    CompletionCounter completions;
    ReentrantTailCompletion primary_completion{
        &scheduler, burst_route, &completions};
    scheduler.start_flow(1, long_tail_route, complete_tail_and_start_one_flow,
                         &primary_completion);
    scheduler.flush_pending_starts();
    scheduler.mark_event_loop_started();
    // The first flow has entered a long propagation tail before this burst
    // starts.  192 inserts force unordered_map rehashes on ordinary load
    // factors; the tail callback still holds the original FluidFlow*.
    event_queue->schedule_event(
        2, start_tail_rehash_burst,
        new TailRehashBurst{&scheduler, burst_route, &completions});
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    require(primary_completion.started && completions.count == kBurstFlows + 2,
            "tail callback survives rehash, erases before callback reentry, and completes every flow once");
}

void run_equivalence_test() {
    auto event_queue = std::make_shared<EventQueue>();
    FluidScheduler scheduler(event_queue, two_links(), 16, 32, 1024);

    auto first_route = std::make_shared<FluidRoute>();
    first_route->link_ids = {0, 1};
    first_route->propagation_latency_ns = 0;
    auto second_route = std::make_shared<FluidRoute>();
    second_route->link_ids = {0};
    second_route->propagation_latency_ns = 0;

    scheduler.enable_link_observer(5);
    scheduler.start_flow(10, first_route, no_op, nullptr);
    scheduler.flush_pending_starts();
    scheduler.mark_event_loop_started();
    event_queue->schedule_event(
        3, start_delayed_flow,
        new DelayedStart{&scheduler, second_route, 4});
    while (!event_queue->finished()) {
        event_queue->proceed();
    }
    // Flow completion is implied by the queue draining plus the observer
    // record equivalence checked below.

    LegacyObserver legacy(2, 5);
    // SI identity: 1 GB/s = 1 B/ns, matching bw_GBps_to_Bpns and the two
    // Link capacities above (1.0 and 0.5 B/ns).
    const auto full_rate = 1.0L;
    legacy.integrate(3, {full_rate / 2.0L, full_rate / 2.0L});
    legacy.integrate(11, {full_rate, full_rate / 2.0L});
    legacy.integrate(20, {full_rate / 2.0L, full_rate / 2.0L});

    std::vector<Record> observed;
    scheduler.link_observer_visit_buckets(append_record, &observed);
    const auto expected = legacy.records();
    const std::vector<Record> hand_checked{
        {0, 0, 3}, {0, 1, 2}, {1, 0, 5}, {1, 1, 3},
        {2, 0, 3}, {2, 1, 2}, {3, 0, 3}, {3, 1, 3}};
    require(same_records(observed, expected), "streaming/legacy record equivalence");
    require(same_records(observed, hand_checked), "fractional carry bucket values");
    require(scheduler.link_observer_window_ns() == 20, "last observer window");

    const auto& totals = scheduler.link_observer_totals();
    require(totals.size() == 2, "equivalence total count");
    require(totals[0].total_bytes == legacy.total_bytes[0] &&
                totals[1].total_bytes == legacy.total_bytes[1],
            "streaming/legacy totals");
    require(totals[0].active_ns == legacy.active_ns[0] &&
                totals[1].active_ns == legacy.active_ns[1],
            "streaming/legacy active time");

    // Spool/residency internals were previously observed through the (now
    // removed) link_observer_storage() snapshot; the replay below is the
    // remaining behavioral oracle for the anonymous-spool design.
    scheduler.link_observer_release();
}

struct CountingVisitor {
    uint64_t count = 0;
    uint64_t total_bytes = 0;
};

bool count_record(void* const context, const uint64_t bucket,
                  const LinkId link_id, const uint64_t bytes) {
    auto& visitor = *static_cast<CountingVisitor*>(context);
    require(bucket == visitor.count, "long timeline bucket order");
    require(link_id == 0 && bytes > 0, "long timeline record value");
    ++visitor.count;
    visitor.total_bytes += bytes;
    return true;
}

void run_bounded_storage_test() {
    constexpr uint64_t kBuckets = 16384;
    auto event_queue = std::make_shared<EventQueue>();
    std::vector<std::shared_ptr<const Link>> links;
    links.push_back(std::make_shared<Link>(0, 1.0, 0.0));
    FluidScheduler scheduler(event_queue, links, 4, 4, 1024);

    auto route = std::make_shared<FluidRoute>();
    route->link_ids = {0};
    route->propagation_latency_ns = 0;
    scheduler.enable_link_observer(1);
    scheduler.start_flow(kBuckets, route, no_op, nullptr);
    scheduler.flush_pending_starts();
    scheduler.mark_event_loop_started();
    while (!event_queue->finished()) {
        event_queue->proceed();
    }

    // A long timeline must close and spool many buckets (instead of
    // materializing an all-run bucket array) and replay them in strict
    // bucket order with totals matching link_observer_totals().
    CountingVisitor visitor;
    scheduler.link_observer_visit_buckets(count_record, &visitor);
    require(visitor.count > kBuckets / 2,
            "long timeline reaches many closed buckets and replays them");
    require(visitor.total_bytes == scheduler.link_observer_totals()[0].total_bytes,
            "long timeline replay total");

    scheduler.link_observer_release();
}

}  // namespace

int main() {
    run_tail_rehash_lifetime_test();
    run_equivalence_test();
    run_bounded_storage_test();
    return EXIT_SUCCESS;
}
