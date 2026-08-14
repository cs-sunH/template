/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/EventQueue.h"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidFlow.h"
#include "congestion_aware/fluid/FluidLinkState.h"
#include <chrono>
#include <optional>
#include <queue>
#include <unordered_map>

namespace NetworkAnalyticalCongestionAware {

class FluidScheduler {
  public:
    FluidScheduler(std::shared_ptr<NetworkAnalytical::EventQueue> event_queue,
                   const std::vector<std::shared_ptr<const Link>>& directed_links,
                   uint64_t max_active_flows,
                   uint64_t max_route_memberships,
                   uint64_t progress_report_event_interval) noexcept;

    void start_flow(NetworkAnalytical::ChunkSize bytes,
                    std::shared_ptr<const FluidRoute> route,
                    NetworkAnalytical::Callback callback,
                    NetworkAnalytical::CallbackArg callback_arg) noexcept;

    void flush_pending_starts() noexcept;
    void mark_event_loop_started() noexcept;
    void handle_service_wakeup(uint64_t generation) noexcept;
    void handle_tail_arrival(FlowId flow_id) noexcept;

    [[nodiscard]] uint64_t get_active_flow_count() const noexcept;
    [[nodiscard]] uint64_t get_active_route_memberships() const noexcept;
    [[nodiscard]] uint64_t get_total_started_flows() const noexcept;
    [[nodiscard]] uint64_t get_total_completed_flows() const noexcept;
    [[nodiscard]] size_t get_completion_heap_size() const noexcept;

  private:
    struct PendingFlowStart {
        FlowId flow_id;
        NetworkAnalytical::ChunkSize bytes;
        std::shared_ptr<const FluidRoute> route;
        NetworkAnalytical::Callback callback;
        NetworkAnalytical::CallbackArg callback_arg;
    };

    struct CompletionEntry {
        NetworkAnalytical::EventTime predicted_finish_time;
        FlowId flow_id;
        uint64_t rate_version;
    };

    struct CompletionLater {
        bool operator()(const CompletionEntry& lhs, const CompletionEntry& rhs) const noexcept;
    };

    struct WakeupContext {
        FluidScheduler* scheduler;
        uint64_t generation;
    };

    struct TailContext {
        FluidScheduler* scheduler;
        FlowId flow_id;
    };

    static void flush_callback(void* context) noexcept;
    static void service_wakeup_callback(void* context) noexcept;
    static void tail_arrival_callback(void* context) noexcept;

    void begin_dirty_batch() noexcept;
    void mark_dirty(FlowId flow_id) noexcept;
    void advance_dirty_flows(NetworkAnalytical::EventTime now) noexcept;
    void recalculate_dirty_rates(NetworkAnalytical::EventTime now) noexcept;
    void remove_memberships(FluidFlow& flow) noexcept;
    void clean_completion_heap() noexcept;
    void maybe_rebuild_completion_heap() noexcept;
    void schedule_next_wakeup() noexcept;
    void schedule_tail_arrival(FluidFlow& flow, NetworkAnalytical::EventTime now) noexcept;
    void check_resource_limits(uint64_t new_flows, uint64_t new_memberships) const noexcept;
    void note_scheduler_event() noexcept;
    void report_progress() const noexcept;
    [[nodiscard]] NetworkAnalytical::EventTime
    predicted_finish_time(const FluidFlow& flow, NetworkAnalytical::EventTime now) const noexcept;
    [[nodiscard]] NetworkAnalytical::EventTime checked_add_time(
        NetworkAnalytical::EventTime lhs, NetworkAnalytical::EventTime rhs) const noexcept;

    std::shared_ptr<NetworkAnalytical::EventQueue> event_queue;
    std::unordered_map<FlowId, FluidFlow> flows_by_id;
    std::vector<FluidLinkState> link_states;
    std::vector<PendingFlowStart> pending_starts;

    std::priority_queue<CompletionEntry, std::vector<CompletionEntry>, CompletionLater> completion_heap;
    uint64_t wakeup_generation;
    std::optional<NetworkAnalytical::EventTime> scheduled_wakeup_time;
    bool flush_scheduled;
    bool event_loop_started;

    uint64_t current_dirty_epoch;
    std::vector<FlowId> dirty_flow_ids;
    FlowId next_flow_id;

    uint64_t active_flow_count;
    uint64_t active_route_memberships;
    uint64_t total_started_flows;
    uint64_t total_completed_flows;
    uint64_t max_active_flows;
    uint64_t max_route_memberships;

    uint64_t progress_report_event_interval;
    uint64_t scheduler_event_count;
    uint64_t dirty_batch_count;
    uint64_t total_dirty_flows;
    uint64_t max_dirty_flows;
    size_t peak_completion_heap_size;
    std::chrono::steady_clock::time_point wall_start_time;
};

}  // namespace NetworkAnalyticalCongestionAware
