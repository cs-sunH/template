/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#include "congestion_aware/fluid/FluidScheduler.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>

using namespace NetworkAnalytical;
using namespace NetworkAnalyticalCongestionAware;

bool FluidScheduler::CompletionLater::operator()(const CompletionEntry& lhs,
                                                  const CompletionEntry& rhs) const noexcept {
    if (lhs.predicted_finish_time != rhs.predicted_finish_time) {
        return lhs.predicted_finish_time > rhs.predicted_finish_time;
    }
    return lhs.flow_id > rhs.flow_id;
}

FluidScheduler::FluidScheduler(std::shared_ptr<EventQueue> event_queue,
                               const std::vector<std::shared_ptr<const Link>>& directed_links,
                               const uint64_t max_active_flows,
                               const uint64_t max_route_memberships,
                               const uint64_t progress_report_event_interval) noexcept
    : event_queue(std::move(event_queue)),
      wakeup_generation(0),
      scheduled_wakeup_time(std::nullopt),
      scheduled_wakeup_event(),
      flush_scheduled(false),
      event_loop_started(false),
      deferred_flush_mode(false),
      current_dirty_epoch(0),
      next_flow_id(0),
      active_flow_count(0),
      active_route_memberships(0),
      total_started_flows(0),
      total_completed_flows(0),
      max_active_flows(max_active_flows),
      max_route_memberships(max_route_memberships),
      progress_report_event_interval(progress_report_event_interval),
      scheduler_event_count(0),
      dirty_batch_count(0),
      total_dirty_flows(0),
      max_dirty_flows(0),
      peak_completion_heap_size(0),
      wall_start_time(std::chrono::steady_clock::now()) {
    assert(this->event_queue != nullptr);
    assert(max_active_flows > 0);
    assert(max_route_memberships > 0);
    assert(progress_report_event_interval > 0);

    link_states.reserve(directed_links.size());
    for (size_t index = 0; index < directed_links.size(); ++index) {
        const auto& link = directed_links[index];
        if (link == nullptr || link->get_id() != index || link->get_bandwidth_Bpns() <= 0) {
            std::cerr << "[Error] (network/analytical/congestion_aware) invalid directed LinkId table"
                      << std::endl;
            std::exit(-1);
        }
        link_states.push_back({link->get_id(), link->get_bandwidth_Bpns(), {}});
    }
}

FluidScheduler::~FluidScheduler() noexcept {
    cancel_scheduled_wakeup();
    link_observer_release();
}

void FluidScheduler::start_flow(const ChunkSize bytes,
                                std::shared_ptr<const FluidRoute> route,
                                const Callback callback,
                                const CallbackArg callback_arg) noexcept {
    if (bytes == 0 || route == nullptr || route->link_ids.empty() || callback == nullptr) {
        std::cerr << "[Error] (network/analytical/congestion_aware) invalid fluid flow" << std::endl;
        std::exit(-1);
    }
    if (next_flow_id == std::numeric_limits<FlowId>::max()) {
        std::cerr << "[Error] (network/analytical/congestion_aware) FluidFlow ID space exhausted"
                  << std::endl;
        std::exit(-1);
    }

    pending_starts.push_back({next_flow_id++, bytes, std::move(route), callback, callback_arg});
    if (event_loop_started && !flush_scheduled) {
        flush_scheduled = true;
        if (deferred_flush_mode) {
            // Online/deferred mode: never insert a current_time event into the
            // main queue from a tick-end/deferred context (EventQueue :33
            // strict-increase assert). Post-commit comm emission lands here and
            // is executed by the same-tick deferred drain.
            event_queue->schedule_event_deferred(flush_callback, this);
        } else {
            // Static mode (pre-extension behavior, byte-for-byte preserved):
            // start_flow is only reachable from inside an invoke_events pass,
            // where the current_time alarm merges into the EventList currently
            // being invoked and executes within the same pass.
            event_queue->schedule_event(event_queue->get_current_time(), flush_callback, this);
        }
    }
}

void FluidScheduler::set_deferred_flush_mode(const bool enabled) noexcept {
    deferred_flush_mode = enabled;
}

void FluidScheduler::flush_callback(void* const context) noexcept {
    static_cast<FluidScheduler*>(context)->flush_pending_starts();
}

void FluidScheduler::service_wakeup_callback(void* const context) noexcept {
    auto* const wakeup = static_cast<WakeupContext*>(context);
    auto* const scheduler = wakeup->scheduler;
    const auto generation = wakeup->generation;
    delete wakeup;
    scheduler->handle_service_wakeup(generation);
}

void FluidScheduler::cancel_wakeup_callback(void* const context) noexcept {
    delete static_cast<WakeupContext*>(context);
}

void FluidScheduler::tail_arrival_callback(void* const context) noexcept {
    // A std::unordered_map rehash invalidates iterators but keeps references
    // and pointers to its elements valid.  The flow stays resident from
    // schedule_tail_arrival() until handle_tail_arrival() erases it below, so
    // this callback needs no per-tail heap context.
    auto* const flow = static_cast<FluidFlow*>(context);
    auto* const scheduler = flow->tail_scheduler;
    if (scheduler == nullptr) {
        std::cerr << "[Error] (network/analytical/congestion_aware) invalid fluid tail callback"
                  << std::endl;
        std::exit(-1);
    }
    scheduler->handle_tail_arrival(flow->flow_id);
}

void FluidScheduler::mark_event_loop_started() noexcept {
    event_loop_started = true;
}

void FluidScheduler::begin_dirty_batch() noexcept {
    ++current_dirty_epoch;
    if (current_dirty_epoch == 0) {
        for (auto& [flow_id, flow] : flows_by_id) {
            flow.dirty_epoch = 0;
        }
        current_dirty_epoch = 1;
    }
    dirty_flow_ids.clear();
}

void FluidScheduler::mark_dirty(const FlowId flow_id) noexcept {
    const auto found = flows_by_id.find(flow_id);
    if (found == flows_by_id.end() || found->second.state != FluidFlowState::Active) {
        return;
    }
    auto& flow = found->second;
    if (flow.dirty_epoch != current_dirty_epoch) {
        flow.dirty_epoch = current_dirty_epoch;
        dirty_flow_ids.push_back(flow_id);
    }
}

void FluidScheduler::advance_dirty_flows(const EventTime now) noexcept {
    // WP6 link observer (CPP_SPEC §D): integrate the just-ended constant
    // interval with the pre-change rates/memberships. advance_dirty_flows
    // is the single choke point both mutation paths pass through before
    // touching any rate or membership, so the observer always sees the
    // interval [last_tick, now) under the rates that were actually in
    // force during it. No-op (single branch) when the observer is off.
    if (link_observer_.enabled) {
        link_observer_integrate(now);
    }
    std::sort(dirty_flow_ids.begin(), dirty_flow_ids.end());
    for (const auto flow_id : dirty_flow_ids) {
        auto& flow = flows_by_id.at(flow_id);
        if (flow.state != FluidFlowState::Active) {
            continue;
        }
        assert(now >= flow.last_rate_update_time);
        const auto elapsed = now - flow.last_rate_update_time;
        flow.remaining_bytes -= static_cast<long double>(elapsed) *
                                static_cast<long double>(flow.current_rate_Bpns);
        if (flow.remaining_bytes < 0.0L) {
            const auto rounding_tolerance =
                std::max(1.0e-9L, static_cast<long double>(flow.current_rate_Bpns) * 1.000001L);
            if (-flow.remaining_bytes > rounding_tolerance) {
                std::cerr << "[Error] (network/analytical/congestion_aware) fluid progress became negative"
                          << std::endl;
                std::exit(-1);
            }
            flow.remaining_bytes = 0.0L;
        }
        flow.last_rate_update_time = now;
    }
}

EventTime FluidScheduler::checked_add_time(const EventTime lhs, const EventTime rhs) const noexcept {
    if (rhs > std::numeric_limits<EventTime>::max() - lhs) {
        std::cerr << "[Error] (network/analytical/congestion_aware) fluid event time overflow" << std::endl;
        std::exit(-1);
    }
    return lhs + rhs;
}

EventTime FluidScheduler::predicted_finish_time(const FluidFlow& flow, const EventTime now) const noexcept {
    if (flow.current_rate_Bpns <= 0 || !std::isfinite(flow.current_rate_Bpns)) {
        std::cerr << "[Error] (network/analytical/congestion_aware) invalid fluid rate" << std::endl;
        std::exit(-1);
    }
    const auto exact_delta = flow.remaining_bytes /
                             static_cast<long double>(flow.current_rate_Bpns);
    if (!std::isfinite(exact_delta) ||
        exact_delta > static_cast<long double>(std::numeric_limits<EventTime>::max())) {
        std::cerr << "[Error] (network/analytical/congestion_aware) fluid service time overflow" << std::endl;
        std::exit(-1);
    }
    const auto rounded = static_cast<EventTime>(std::ceil(std::max(0.0L, exact_delta)));
    return checked_add_time(now, std::max<EventTime>(1, rounded));
}

void FluidScheduler::recalculate_dirty_rates(const EventTime now) noexcept {
    // advance_dirty_flows() sorts the existing dirty prefix. A start batch
    // then appends strictly increasing, never-reused flow ids, all greater
    // than every existing id; a completion batch appends nothing. Preserve
    // that precondition instead of sorting the same batch twice.
    assert(std::is_sorted(dirty_flow_ids.begin(), dirty_flow_ids.end()));
    for (const auto flow_id : dirty_flow_ids) {
        auto& flow = flows_by_id.at(flow_id);
        if (flow.state != FluidFlowState::Active) {
            continue;
        }

        auto new_rate = std::numeric_limits<Bandwidth>::infinity();
        for (const auto link_id : flow.route->link_ids) {
            if (link_id >= link_states.size() || link_states[link_id].active_flows.empty()) {
                std::cerr << "[Error] (network/analytical/congestion_aware) invalid fluid link membership"
                          << std::endl;
                std::exit(-1);
            }
            const auto& link = link_states[link_id];
            const auto share = link.capacity_Bpns /
                               static_cast<Bandwidth>(link.active_flows.size());
            new_rate = std::min(new_rate, share);
        }

        if (link_observer_.enabled) {
            link_observer_adjust_flow_rate(flow, flow.current_rate_Bpns, new_rate);
        }
        flow.current_rate_Bpns = new_rate;
        ++flow.rate_version;
        completion_heap.push({predicted_finish_time(flow, now), flow.flow_id, flow.rate_version});
    }

    ++dirty_batch_count;
    total_dirty_flows += dirty_flow_ids.size();
    max_dirty_flows = std::max<uint64_t>(max_dirty_flows, dirty_flow_ids.size());
    peak_completion_heap_size = std::max(peak_completion_heap_size, completion_heap.size());
}

void FluidScheduler::check_resource_limits(const uint64_t new_flows,
                                           const uint64_t new_memberships) const noexcept {
    const auto flow_overflow = new_flows > max_active_flows - std::min(active_flow_count, max_active_flows);
    const auto membership_overflow =
        new_memberships > max_route_memberships - std::min(active_route_memberships, max_route_memberships);
    if (!flow_overflow && !membership_overflow) {
        return;
    }

    size_t busiest_link = 0;
    for (size_t i = 1; i < link_states.size(); ++i) {
        if (link_states[i].active_flows.size() > link_states[busiest_link].active_flows.size()) {
            busiest_link = i;
        }
    }
    std::cerr << "[Error] (network/analytical/congestion_aware) fluid resource limit exceeded at "
              << event_queue->get_current_time() << " ns: active_flows=" << active_flow_count
              << ", pending_flows=" << new_flows << ", route_memberships=" << active_route_memberships
              << ", pending_memberships=" << new_memberships;
    if (!link_states.empty()) {
        std::cerr << ", busiest_link=" << busiest_link
                  << " (" << link_states[busiest_link].active_flows.size() << " flows)";
    }
    std::cerr << std::endl;
    std::exit(-1);
}

void FluidScheduler::flush_pending_starts() noexcept {
    flush_scheduled = false;
    if (pending_starts.empty()) {
        return;
    }

    uint64_t new_memberships = 0;
    for (const auto& pending : pending_starts) {
        if (pending.route->link_ids.size() > std::numeric_limits<uint64_t>::max() - new_memberships) {
            std::cerr << "[Error] (network/analytical/congestion_aware) route membership count overflow"
                      << std::endl;
            std::exit(-1);
        }
        new_memberships += pending.route->link_ids.size();
    }
    check_resource_limits(pending_starts.size(), new_memberships);

    const auto now = event_queue->get_current_time();
    begin_dirty_batch();
    for (const auto& pending : pending_starts) {
        for (const auto link_id : pending.route->link_ids) {
            if (link_id >= link_states.size()) {
                std::cerr << "[Error] (network/analytical/congestion_aware) invalid LinkId in fluid route"
                          << std::endl;
                std::exit(-1);
            }
            for (const auto& active : link_states[link_id].active_flows) {
                mark_dirty(active.flow_id);
            }
        }
    }
    advance_dirty_flows(now);

    for (const auto& pending : pending_starts) {
        FluidFlow flow{pending.flow_id,
                       pending.bytes,
                       static_cast<long double>(pending.bytes),
                       0.0,
                       now,
                       0,
                       0,
                       pending.route,
                       {},
                       FluidFlowState::Active,
                       pending.callback,
                       pending.callback_arg};
        flow.memberships.reserve(flow.route->link_ids.size());
        const auto inserted = flows_by_id.emplace(flow.flow_id, std::move(flow));
        assert(inserted.second);
        auto& stored = inserted.first->second;

        for (const auto link_id : stored.route->link_ids) {
            auto& active = link_states[link_id].active_flows;
            if (link_observer_.enabled && active.empty()) {
                link_observer_activate_link(link_id);
            }
            const auto membership_index = static_cast<uint32_t>(stored.memberships.size());
            const auto active_index = static_cast<uint32_t>(active.size());
            active.push_back({stored.flow_id, membership_index});
            stored.memberships.push_back({link_id, active_index});
        }
        mark_dirty(stored.flow_id);
    }

    active_flow_count += pending_starts.size();
    active_route_memberships += new_memberships;
    total_started_flows += pending_starts.size();
    pending_starts.clear();

    recalculate_dirty_rates(now);
    maybe_rebuild_completion_heap();
    schedule_next_wakeup();
    note_scheduler_event();
}

void FluidScheduler::clean_completion_heap() noexcept {
    while (!completion_heap.empty()) {
        const auto& entry = completion_heap.top();
        const auto found = flows_by_id.find(entry.flow_id);
        if (found != flows_by_id.end() && found->second.state == FluidFlowState::Active &&
            found->second.rate_version == entry.rate_version) {
            break;
        }
        completion_heap.pop();
    }
}

void FluidScheduler::maybe_rebuild_completion_heap() noexcept {
    const auto rebuild_threshold = active_flow_count >
                                           (std::numeric_limits<uint64_t>::max() - 1024) / 4
                                       ? std::numeric_limits<uint64_t>::max()
                                       : 4 * active_flow_count + 1024;
    if (completion_heap.size() <= rebuild_threshold) {
        return;
    }

    decltype(completion_heap) rebuilt;
    while (!completion_heap.empty()) {
        const auto entry = completion_heap.top();
        completion_heap.pop();

        const auto found = flows_by_id.find(entry.flow_id);
        if (found != flows_by_id.end() && found->second.state == FluidFlowState::Active &&
            found->second.rate_version == entry.rate_version) {
            rebuilt.push(entry);
        }
    }

    assert(rebuilt.size() == active_flow_count);
    completion_heap.swap(rebuilt);
    peak_completion_heap_size = std::max(peak_completion_heap_size, completion_heap.size());
}

void FluidScheduler::cancel_scheduled_wakeup() noexcept {
    if (scheduled_wakeup_event.valid()) {
        static_cast<void>(event_queue->cancel_event(scheduled_wakeup_event));
    }
    scheduled_wakeup_event.reset();
}

void FluidScheduler::schedule_next_wakeup() noexcept {
    clean_completion_heap();
    if (completion_heap.empty()) {
        cancel_scheduled_wakeup();
        scheduled_wakeup_time.reset();
        return;
    }

    const auto next_time = completion_heap.top().predicted_finish_time;
    if (scheduled_wakeup_time.has_value() && next_time >= scheduled_wakeup_time.value()) {
        return;
    }

    // The superseded alarm would only fail the generation guard.  Remove its
    // EventList node now and reclaim WakeupContext immediately; the new
    // earlier prediction is the sole live wakeup representation.
    cancel_scheduled_wakeup();
    ++wakeup_generation;
    scheduled_wakeup_time = next_time;
    auto* const context = new WakeupContext{this, wakeup_generation};
    scheduled_wakeup_event = event_queue->schedule_event_cancellable(
        next_time, service_wakeup_callback, context, cancel_wakeup_callback);
}

void FluidScheduler::remove_memberships(FluidFlow& flow) noexcept {
    for (uint32_t membership_index = 0; membership_index < flow.memberships.size(); ++membership_index) {
        const auto membership = flow.memberships[membership_index];
        auto& active = link_states[membership.link_id].active_flows;
        assert(membership.index_in_link_active_flows < active.size());
        assert(active[membership.index_in_link_active_flows].flow_id == flow.flow_id);

        const auto removed_index = membership.index_in_link_active_flows;
        const auto moved = active.back();
        if (removed_index + 1 != active.size()) {
            active[removed_index] = moved;
            auto& moved_flow = flows_by_id.at(moved.flow_id);
            moved_flow.memberships[moved.membership_index_in_flow].index_in_link_active_flows = removed_index;
        }
        active.pop_back();
        if (link_observer_.enabled && active.empty()) {
            link_observer_deactivate_link(membership.link_id);
        }
    }
}

void FluidScheduler::schedule_tail_arrival(FluidFlow& flow, const EventTime now) noexcept {
    const auto arrival_time = checked_add_time(now, flow.route->propagation_latency_ns);
    if (flow.tail_scheduler != nullptr) {
        std::cerr << "[Error] (network/analytical/congestion_aware) duplicate fluid tail arrival"
                  << std::endl;
        std::exit(-1);
    }
    flow.tail_scheduler = this;
    event_queue->schedule_event(arrival_time, tail_arrival_callback, &flow);
}

void FluidScheduler::handle_service_wakeup(const uint64_t generation) noexcept {
    if (generation != wakeup_generation) {
        return;
    }
    // This wakeup was popped before its callback began; it cannot be
    // cancelled any more.  Forget the weak handle before callbacks below
    // potentially schedule the next prediction.
    scheduled_wakeup_event.reset();
    scheduled_wakeup_time.reset();
    const auto now = event_queue->get_current_time();
    clean_completion_heap();

    auto completed_ids = std::vector<FlowId>();
    while (!completion_heap.empty() && completion_heap.top().predicted_finish_time <= now) {
        const auto entry = completion_heap.top();
        completion_heap.pop();
        const auto found = flows_by_id.find(entry.flow_id);
        if (found != flows_by_id.end() && found->second.state == FluidFlowState::Active &&
            found->second.rate_version == entry.rate_version) {
            completed_ids.push_back(entry.flow_id);
        }
        clean_completion_heap();
    }

    if (completed_ids.empty()) {
        schedule_next_wakeup();
        note_scheduler_event();
        return;
    }
    std::sort(completed_ids.begin(), completed_ids.end());

    begin_dirty_batch();
    for (const auto flow_id : completed_ids) {
        const auto& flow = flows_by_id.at(flow_id);
        for (const auto link_id : flow.route->link_ids) {
            for (const auto& active : link_states[link_id].active_flows) {
                mark_dirty(active.flow_id);
            }
        }
    }
    advance_dirty_flows(now);

    for (const auto flow_id : completed_ids) {
        auto& flow = flows_by_id.at(flow_id);
        flow.remaining_bytes = 0.0L;
        if (link_observer_.enabled) {
            link_observer_adjust_flow_rate(flow, flow.current_rate_Bpns, 0.0);
        }
        remove_memberships(flow);
        active_route_memberships -= flow.memberships.size();
        --active_flow_count;
        flow.state = FluidFlowState::PropagatingTail;
        ++flow.rate_version;
        schedule_tail_arrival(flow, now);
    }

    recalculate_dirty_rates(now);
    maybe_rebuild_completion_heap();
    schedule_next_wakeup();
    note_scheduler_event();
}

void FluidScheduler::handle_tail_arrival(const FlowId flow_id) noexcept {
    const auto found = flows_by_id.find(flow_id);
    if (found == flows_by_id.end() || found->second.state != FluidFlowState::PropagatingTail ||
        found->second.tail_scheduler != this) {
        std::cerr << "[Error] (network/analytical/congestion_aware) invalid fluid tail arrival" << std::endl;
        std::exit(-1);
    }

    found->second.tail_scheduler = nullptr;
    found->second.state = FluidFlowState::Completed;
    const auto callback = found->second.completion_callback;
    const auto callback_arg = found->second.completion_arg;
    flows_by_id.erase(found);
    ++total_completed_flows;
    note_scheduler_event();
    callback(callback_arg);
}

void FluidScheduler::note_scheduler_event() noexcept {
    ++scheduler_event_count;
    if (scheduler_event_count % progress_report_event_interval == 0) {
        report_progress();
    }
}

void FluidScheduler::report_progress() const noexcept {
    auto congestion = std::vector<std::pair<size_t, size_t>>();
    congestion.reserve(link_states.size());
    for (size_t i = 0; i < link_states.size(); ++i) {
        congestion.emplace_back(link_states[i].active_flows.size(), i);
    }
    std::sort(congestion.begin(), congestion.end(), std::greater<>());
    const auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - wall_start_time).count();
    const auto dirty_average = dirty_batch_count == 0
                                   ? 0.0
                                   : static_cast<double>(total_dirty_flows) /
                                         static_cast<double>(dirty_batch_count);
    std::cerr << "[FluidProgress] time_ns=" << event_queue->get_current_time()
              << " wall_seconds=" << elapsed << " active_flows=" << active_flow_count
              << " completed_flows=" << total_completed_flows
              << " route_memberships=" << active_route_memberships
              << " heap=" << completion_heap.size() << " peak_heap=" << peak_completion_heap_size
              << " dirty_average=" << dirty_average << " dirty_max=" << max_dirty_flows;
    for (size_t i = 0; i < std::min<size_t>(3, congestion.size()); ++i) {
        std::cerr << " congested_link_" << i << '=' << congestion[i].second << ':' << congestion[i].first;
    }
    std::cerr << std::endl;
}

uint64_t FluidScheduler::get_active_flow_count() const noexcept {
    return active_flow_count;
}

uint64_t FluidScheduler::get_active_route_memberships() const noexcept {
    return active_route_memberships;
}

// ---------------------------------------------------------------------------
// WP6 NoC link observer (read-only side-band; CPP_SPEC §D).
// ---------------------------------------------------------------------------

void FluidScheduler::enable_link_observer(const uint64_t link_bucket_ns) noexcept {
    if (link_bucket_ns == 0) {
        link_observer_fail("bucket must be positive");
    }
    auto& observer = link_observer_;
    if (observer.enabled || observer.spool != nullptr) {
        link_observer_fail("enabled more than once");
    }
    observer.spool = std::tmpfile();
    if (observer.spool == nullptr) {
        link_observer_fail_io("creating anonymous spool");
    }

    observer.enabled = true;
    observer.bucket_ns = link_bucket_ns;
    observer.last_tick = event_queue->get_current_time();
    const auto link_count_value = link_states.size();
    observer.carry.assign(link_count_value, 0.0L);
    observer.rate_sum.assign(link_count_value, 0.0L);
    observer.total_bytes.assign(link_count_value, 0);
    observer.active_ns.assign(link_count_value, 0);
    observer.current_bucket_bytes.assign(link_count_value, 0);
    observer.active_links.clear();
    observer.active_links.reserve(link_count_value);
    observer.current_bucket_links.clear();
    observer.current_bucket_links.reserve(link_count_value);
    observer.current_bucket = 0;
    observer.current_bucket_open = false;
    observer.spool_record_count = 0;
    observer.finalized = false;
    observer.totals_scratch.clear();

    // This one-time initialization preserves the exact pre-enable rate sum.
    // Subsequent membership/rate changes update the same sums incrementally.
    for (const auto& [flow_id, flow] : flows_by_id) {
        (void)flow_id;
        if (flow.state != FluidFlowState::Active) {
            continue;
        }
        for (const auto& membership : flow.memberships) {
            if (membership.link_id >= observer.rate_sum.size()) {
                link_observer_fail("invalid link membership while enabling");
            }
            observer.rate_sum[membership.link_id] +=
                static_cast<long double>(flow.current_rate_Bpns);
        }
    }
    for (size_t link = 0; link < link_states.size(); ++link) {
        if (!link_states[link].active_flows.empty()) {
            observer.active_links.push_back(static_cast<LinkId>(link));
        }
    }
}

bool FluidScheduler::link_observer_enabled() const noexcept {
    return link_observer_.enabled;
}

uint64_t FluidScheduler::link_observer_bucket_ns() const noexcept {
    return link_observer_.bucket_ns;
}

EventTime FluidScheduler::link_observer_window_ns() const noexcept {
    return link_observer_.last_tick;
}

void FluidScheduler::link_observer_fail(const char* const reason) const noexcept {
    std::cerr << "[Error] (network/analytical/congestion_aware) link observer " << reason
              << std::endl;
    std::exit(-1);
}

void FluidScheduler::link_observer_fail_io(const char* const operation) const noexcept {
    std::cerr << "[Error] (network/analytical/congestion_aware) link observer I/O failure while "
              << operation << std::endl;
    std::exit(-1);
}

void FluidScheduler::link_observer_write_u64(const uint64_t value) noexcept {
    auto* const spool = link_observer_.spool;
    if (spool == nullptr) {
        link_observer_fail("spool is unavailable");
    }
    if (std::fwrite(&value, 1, sizeof(value), spool) != sizeof(value)) {
        link_observer_fail_io("writing spool");
    }
}

void FluidScheduler::link_observer_activate_link(const LinkId link_id) noexcept {
    auto& observer = link_observer_;
    if (link_id >= observer.rate_sum.size()) {
        link_observer_fail("activating an invalid link");
    }
    const auto position = std::lower_bound(observer.active_links.begin(),
                                           observer.active_links.end(), link_id);
    if (position == observer.active_links.end() || *position != link_id) {
        observer.active_links.insert(position, link_id);
    }
}

void FluidScheduler::link_observer_deactivate_link(const LinkId link_id) noexcept {
    auto& observer = link_observer_;
    if (link_id >= observer.rate_sum.size()) {
        link_observer_fail("deactivating an invalid link");
    }
    const auto position = std::lower_bound(observer.active_links.begin(),
                                           observer.active_links.end(), link_id);
    if (position == observer.active_links.end() || *position != link_id) {
        link_observer_fail("deactivating an untracked link");
    }
    observer.active_links.erase(position);
    // The last membership has just left; discard any floating-point residue
    // from subtracting its old rate before the membership mutation.
    observer.rate_sum[link_id] = 0.0L;
}

void FluidScheduler::link_observer_adjust_flow_rate(const FluidFlow& flow,
                                                     const Bandwidth old_rate,
                                                     const Bandwidth new_rate) noexcept {
    if (old_rate == new_rate) {
        return;
    }
    auto& observer = link_observer_;
    for (const auto& membership : flow.memberships) {
        if (membership.link_id >= observer.rate_sum.size()) {
            link_observer_fail("adjusting an invalid link rate");
        }
        auto& rate_sum = observer.rate_sum[membership.link_id];
        rate_sum += static_cast<long double>(new_rate) - static_cast<long double>(old_rate);
        if (!std::isfinite(rate_sum)) {
            link_observer_fail("rate sum became non-finite");
        }
    }
}

void FluidScheduler::link_observer_flush_current_bucket() noexcept {
    auto& observer = link_observer_;
    if (!observer.current_bucket_open) {
        return;
    }

    // active_links is sorted, but a link can first produce a whole byte only
    // after a later rate-change segment. Sort the compact touched list so the
    // spool retains the legacy ascending-LinkId tie order.
    std::sort(observer.current_bucket_links.begin(), observer.current_bucket_links.end());
    for (const auto link_id : observer.current_bucket_links) {
        if (link_id >= observer.current_bucket_bytes.size()) {
            link_observer_fail("current bucket contains an invalid link");
        }
        const auto bytes = observer.current_bucket_bytes[link_id];
        if (bytes == 0) {
            link_observer_fail("current bucket contains an empty record");
        }
        if (observer.spool_record_count == std::numeric_limits<uint64_t>::max()) {
            link_observer_fail("spool record count overflow");
        }
        link_observer_write_u64(observer.current_bucket);
        link_observer_write_u64(static_cast<uint64_t>(link_id));
        link_observer_write_u64(bytes);
        ++observer.spool_record_count;
        observer.current_bucket_bytes[link_id] = 0;
    }
    observer.current_bucket_links.clear();
    observer.current_bucket_open = false;
}

void FluidScheduler::link_observer_finish() noexcept {
    auto& observer = link_observer_;
    if (observer.finalized) {
        return;
    }
    link_observer_flush_current_bucket();
    if (observer.spool == nullptr || std::fflush(observer.spool) != 0) {
        link_observer_fail_io("flushing spool");
    }
    observer.finalized = true;
}

void FluidScheduler::link_observer_visit_buckets(const LinkObserverBucketVisitor visitor,
                                                  void* const context) noexcept {
    if (!link_observer_.enabled) {
        return;
    }
    if (visitor == nullptr) {
        link_observer_fail("bucket visitor is null");
    }
    link_observer_finish();

    auto* const spool = link_observer_.spool;
    if (spool == nullptr) {
        link_observer_fail("spool is unavailable during replay");
    }
    std::clearerr(spool);
    if (std::fseek(spool, 0, SEEK_SET) != 0) {
        link_observer_fail_io("seeking spool for replay");
    }

    bool have_previous = false;
    uint64_t previous_bucket = 0;
    uint64_t previous_link = 0;
    while (true) {
        uint64_t bucket = 0;
        const auto first_bytes = std::fread(&bucket, 1, sizeof(bucket), spool);
        if (first_bytes == 0) {
            if (std::feof(spool) != 0) {
                break;
            }
            link_observer_fail_io("reading spool");
        }
        if (first_bytes != sizeof(bucket)) {
            link_observer_fail_io("reading a partial spool record");
        }

        uint64_t raw_link_id = 0;
        uint64_t bytes = 0;
        if (std::fread(&raw_link_id, 1, sizeof(raw_link_id), spool) != sizeof(raw_link_id) ||
            std::fread(&bytes, 1, sizeof(bytes), spool) != sizeof(bytes)) {
            link_observer_fail_io("reading a partial spool record");
        }
        if (raw_link_id >= link_states.size() || bytes == 0) {
            link_observer_fail("spool record is corrupt");
        }
        if (have_previous &&
            (bucket < previous_bucket ||
             (bucket == previous_bucket && raw_link_id <= previous_link))) {
            link_observer_fail("spool record order is corrupt");
        }
        if (!visitor(context, bucket, static_cast<LinkId>(raw_link_id), bytes)) {
            link_observer_fail("bucket visitor rejected a record");
        }
        have_previous = true;
        previous_bucket = bucket;
        previous_link = raw_link_id;
    }
    if (std::ferror(spool) != 0) {
        link_observer_fail_io("reading spool");
    }
}

const std::vector<FluidScheduler::LinkObserverTotals>& FluidScheduler::link_observer_totals() const noexcept {
    // Materialized into scratch storage the caller can iterate while
    // emitting; the scheduler is single-threaded (event loop owner).
    auto& totals = link_observer_.totals_scratch;
    totals.assign(link_states.size(), LinkObserverTotals{0, 0});
    for (size_t link = 0; link < link_states.size() && link < link_observer_.total_bytes.size(); ++link) {
        totals[link].total_bytes = link_observer_.total_bytes[link];
        totals[link].active_ns = link_observer_.active_ns[link];
    }
    return totals;
}

FluidScheduler::LinkObserverStorage FluidScheduler::link_observer_storage() const noexcept {
    const auto& observer = link_observer_;
    return {observer.current_bucket_links.size(), observer.current_bucket_links.capacity(),
            observer.active_links.size(), observer.spool_record_count,
            observer.spool != nullptr};
}

void FluidScheduler::link_observer_release() noexcept {
    auto& observer = link_observer_;
    if (observer.spool != nullptr) {
        if (std::fclose(observer.spool) != 0) {
            link_observer_fail_io("closing spool");
        }
        observer.spool = nullptr;
    }
    observer.carry.clear();
    observer.carry.shrink_to_fit();
    observer.rate_sum.clear();
    observer.rate_sum.shrink_to_fit();
    observer.total_bytes.clear();
    observer.total_bytes.shrink_to_fit();
    observer.active_ns.clear();
    observer.active_ns.shrink_to_fit();
    observer.active_links.clear();
    observer.active_links.shrink_to_fit();
    observer.current_bucket_bytes.clear();
    observer.current_bucket_bytes.shrink_to_fit();
    observer.current_bucket_links.clear();
    observer.current_bucket_links.shrink_to_fit();
    observer.totals_scratch.clear();
    observer.totals_scratch.shrink_to_fit();
    observer.bucket_ns = 0;
    observer.last_tick = 0;
    observer.current_bucket = 0;
    observer.current_bucket_open = false;
    observer.spool_record_count = 0;
    observer.finalized = false;
    observer.enabled = false;
}

void FluidScheduler::link_observer_integrate(const EventTime now) noexcept {
    auto& observer = link_observer_;
    if (observer.finalized) {
        link_observer_fail("integrated after finalization");
    }
    if (now <= observer.last_tick) {
        return;  // same-tick mutation batch: nothing elapsed
    }

    bool any_active_rate = false;
    for (const auto link_id : observer.active_links) {
        if (link_id >= observer.rate_sum.size()) {
            link_observer_fail("active link is invalid");
        }
        const auto rate = observer.rate_sum[link_id];
        if (!std::isfinite(rate)) {
            link_observer_fail("rate sum became non-finite");
        }
        if (rate > 0.0L) {
            any_active_rate = true;
            break;
        }
    }
    if (!any_active_rate) {
        // An idle time gap has no bucket records. Flush the one previously
        // open bucket if it closed, then jump the watermark without walking
        // every empty bucket.
        if (observer.current_bucket_open &&
            now / observer.bucket_ns > observer.current_bucket) {
            link_observer_flush_current_bucket();
        }
        observer.last_tick = now;
        return;
    }

    // Walk the interval, splitting it at bucket boundaries so long constant
    // segments still land in the right buckets (periodic coverage without
    // registering any timer event).
    EventTime segment_start = observer.last_tick;
    while (segment_start < now) {
        const auto bucket = static_cast<uint64_t>(segment_start / observer.bucket_ns);
        if (!observer.current_bucket_open) {
            observer.current_bucket = bucket;
            observer.current_bucket_open = true;
        } else if (bucket != observer.current_bucket) {
            if (bucket < observer.current_bucket) {
                link_observer_fail("bucket order regressed");
            }
            link_observer_flush_current_bucket();
            observer.current_bucket = bucket;
            observer.current_bucket_open = true;
        }
        auto bucket_end = std::numeric_limits<EventTime>::max();
        if (bucket < std::numeric_limits<EventTime>::max() / observer.bucket_ns) {
            bucket_end = static_cast<EventTime>(bucket + 1) * observer.bucket_ns;
        }
        auto segment_end = now < bucket_end ? now : bucket_end;
        if (segment_end <= segment_start) {
            link_observer_fail("encountered a zero-length integration segment");
        }
        const auto dt = segment_end - segment_start;
        for (const auto link_id : observer.active_links) {
            const auto link = static_cast<size_t>(link_id);
            const auto rate = observer.rate_sum[link];
            if (rate <= 0.0L) {
                continue;
            }
            observer.carry[link] += rate * static_cast<long double>(dt);
            if (!std::isfinite(observer.carry[link]) || observer.carry[link] < 0.0L ||
                observer.carry[link] >
                    static_cast<long double>(std::numeric_limits<uint64_t>::max())) {
                link_observer_fail("byte carry overflow");
            }
            const auto whole = static_cast<uint64_t>(observer.carry[link]);
            if (whole > 0) {
                if (whole > std::numeric_limits<uint64_t>::max() -
                                observer.current_bucket_bytes[link] ||
                    whole > std::numeric_limits<uint64_t>::max() - observer.total_bytes[link]) {
                    link_observer_fail("byte total overflow");
                }
                if (observer.current_bucket_bytes[link] == 0) {
                    observer.current_bucket_links.push_back(link_id);
                }
                observer.current_bucket_bytes[link] += whole;
                observer.total_bytes[link] += whole;
                observer.carry[link] -= static_cast<long double>(whole);
            }
            if (dt > std::numeric_limits<uint64_t>::max() - observer.active_ns[link]) {
                link_observer_fail("active-time overflow");
            }
            observer.active_ns[link] += dt;
        }
        segment_start = segment_end;
    }
    observer.last_tick = now;
}

uint64_t FluidScheduler::get_total_started_flows() const noexcept {
    return total_started_flows;
}

uint64_t FluidScheduler::get_total_completed_flows() const noexcept {
    return total_completed_flows;
}
