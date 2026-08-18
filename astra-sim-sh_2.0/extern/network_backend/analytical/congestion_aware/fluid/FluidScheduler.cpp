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
      link_state_epoch_(0),
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

void FluidScheduler::flush_pending_starts_deferred() noexcept {
    if (flush_scheduled) {
        return;
    }
    flush_scheduled = true;
    event_queue->schedule_event_deferred(flush_callback, this);
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

void FluidScheduler::tail_arrival_callback(void* const context) noexcept {
    auto* const tail = static_cast<TailContext*>(context);
    auto* const scheduler = tail->scheduler;
    const auto flow_id = tail->flow_id;
    delete tail;
    scheduler->handle_tail_arrival(flow_id);
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
    std::sort(dirty_flow_ids.begin(), dirty_flow_ids.end());
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
    ++link_state_epoch_;  // Phase-7 §10.2: membership added to link_states

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

void FluidScheduler::schedule_next_wakeup() noexcept {
    clean_completion_heap();
    if (completion_heap.empty()) {
        scheduled_wakeup_time.reset();
        return;
    }

    const auto next_time = completion_heap.top().predicted_finish_time;
    if (scheduled_wakeup_time.has_value() && next_time >= scheduled_wakeup_time.value()) {
        return;
    }

    ++wakeup_generation;
    scheduled_wakeup_time = next_time;
    auto* const context = new WakeupContext{this, wakeup_generation};
    event_queue->schedule_event(next_time, service_wakeup_callback, context);
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
    }
}

void FluidScheduler::schedule_tail_arrival(FluidFlow& flow, const EventTime now) noexcept {
    const auto arrival_time = checked_add_time(now, flow.route->propagation_latency_ns);
    auto* const context = new TailContext{this, flow.flow_id};
    event_queue->schedule_event(arrival_time, tail_arrival_callback, context);
}

void FluidScheduler::handle_service_wakeup(const uint64_t generation) noexcept {
    if (generation != wakeup_generation) {
        return;
    }
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
        remove_memberships(flow);
        active_route_memberships -= flow.memberships.size();
        --active_flow_count;
        flow.state = FluidFlowState::PropagatingTail;
        ++flow.rate_version;
        schedule_tail_arrival(flow, now);
    }
    ++link_state_epoch_;  // Phase-7 §10.2: memberships removed from link_states

    recalculate_dirty_rates(now);
    maybe_rebuild_completion_heap();
    schedule_next_wakeup();
    note_scheduler_event();
}

void FluidScheduler::handle_tail_arrival(const FlowId flow_id) noexcept {
    const auto found = flows_by_id.find(flow_id);
    if (found == flows_by_id.end() || found->second.state != FluidFlowState::PropagatingTail) {
        std::cerr << "[Error] (network/analytical/congestion_aware) invalid fluid tail arrival" << std::endl;
        std::exit(-1);
    }

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

uint64_t FluidScheduler::link_state_epoch() const noexcept {
    return link_state_epoch_;
}

size_t FluidScheduler::link_count() const noexcept {
    return link_states.size();
}

std::optional<LinkCongestionSnapshot> FluidScheduler::link_congestion_snapshot(
    const LinkId link_id, const uint64_t expected_tick,
    const uint64_t expected_epoch) const noexcept {
    // Phase-7 §10.2 expired-handle semantics: a snapshot is only valid for the
    // (tick, epoch) it was taken at; stale tick or stale epoch is rejected.
    if (expected_tick != event_queue->get_current_time() ||
        expected_epoch != link_state_epoch_) {
        return std::nullopt;
    }
    if (link_id >= link_states.size()) {
        return std::nullopt;
    }
    const auto& link = link_states[link_id];
    long double remaining_bytes = 0.0L;
    for (const auto& active : link.active_flows) {
        const auto found = flows_by_id.find(active.flow_id);
        if (found == flows_by_id.end() ||
            found->second.state != FluidFlowState::Active) {
            continue;  // defensive: link membership must be self-consistent
        }
        // Each active flow contributes its full outstanding bytes (the fluid
        // model transfers the whole flow over every link of its route).
        remaining_bytes += found->second.remaining_bytes;
    }
    return LinkCongestionSnapshot{link_id, remaining_bytes,
                                  static_cast<uint64_t>(link.active_flows.size()),
                                  expected_tick, expected_epoch};
}

uint64_t FluidScheduler::get_active_route_memberships() const noexcept {
    return active_route_memberships;
}

uint64_t FluidScheduler::get_total_started_flows() const noexcept {
    return total_started_flows;
}

uint64_t FluidScheduler::get_total_completed_flows() const noexcept {
    return total_completed_flows;
}

size_t FluidScheduler::get_completion_heap_size() const noexcept {
    return completion_heap.size();
}
