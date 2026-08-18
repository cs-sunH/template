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

/// Phase-7 §10.2 route-congestion snapshot (audit/explanation input only;
/// the wscllm strategy never consumes it -- red-line decision inputs stay
/// Python queue/KV ledgers and the static route).
///
/// remaining_bytes sums the outstanding bytes of the flows currently
/// traversing the link (each active flow contributes its full remaining
/// bytes -- the fluid model transfers the whole flow over every link of its
/// route). It is the accounting value as of the last flow advance (the
/// scheduler advances flows lazily on membership changes, not on every
/// tick), i.e. a stable monotone-decreasing bookkeeping value rather than a
/// per-nanosecond estimate. active_flow_count is exact at the snapshot
/// instant. tick/epoch are the freshness guards: a snapshot is only valid
/// for the (tick, epoch) it was taken at; any later use must be rejected.
struct LinkCongestionSnapshot {
    LinkId link_id;
    long double remaining_bytes;
    uint64_t active_flow_count;
    NetworkAnalytical::EventTime tick;
    uint64_t epoch;
};

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

    /**
     * Set whether start_flow's flush alarm must go through the same-tick
     * deferred channel (online/deferred mode) instead of
     * schedule_event(current_time, ...).
     *
     * Audit (phase-1 step 1-1): the plain schedule_event(current_time) alarm
     * is only safe while the current tick's EventList is being invoked (the
     * same-time merge keeps it inside the same invoke_events pass). A comm
     * node emitted from a tick-end callback or the deferred drain would insert
     * a current_time EventList into the main queue and trip the EventQueue
     * strict-increase assert on the next proceed(). Online drivers must enable
     * deferred mode; the static path stays in the legacy mode (default), which
     * is byte-for-byte the pre-extension behavior.
     */
    void set_deferred_flush_mode(bool enabled) noexcept;

    /**
     * Schedule a flush of all pending flow starts via the same-tick deferred
     * channel (schedule_event_deferred). Post-commit communication emission in
     * online mode must go through this entry (or start_flow with deferred mode
     * enabled); it never inserts a current_time event into the main queue.
     */
    void flush_pending_starts_deferred() noexcept;

    void mark_event_loop_started() noexcept;
    void handle_service_wakeup(uint64_t generation) noexcept;
    void handle_tail_arrival(FlowId flow_id) noexcept;

    [[nodiscard]] uint64_t get_active_flow_count() const noexcept;
    [[nodiscard]] uint64_t get_active_route_memberships() const noexcept;
    [[nodiscard]] uint64_t get_total_started_flows() const noexcept;
    [[nodiscard]] uint64_t get_total_completed_flows() const noexcept;
    [[nodiscard]] size_t get_completion_heap_size() const noexcept;

    /**
     * Phase-7 §10.2: per-link route-congestion snapshot accessor.
     *
     * Pure query -- no flow is advanced and no state is touched. Expired-handle
     * semantics: the caller must pass the tick it believes is current and the
     * link-state epoch it last observed; when either does not match the
     * scheduler's current event time / epoch, or the link id is unknown, the
     * snapshot is rejected and nullopt is returned (a stale snapshot must
     * never be consumed as fresh congestion).
     *
     * The accessor is a shared mechanism (like the list-mode EventQueue):
     * reusable by face/sh_1.0 online backends as audit/explanation input.
     */
    [[nodiscard]] std::optional<LinkCongestionSnapshot> link_congestion_snapshot(
        LinkId link_id, uint64_t expected_tick, uint64_t expected_epoch) const noexcept;

    /// Phase-7 §10.2: current link-state epoch. Incremented on every
    /// membership change that touches link_states (flow starts / completions).
    /// Audit/explanation input; a snapshot's epoch is only valid while it
    /// equals this value.
    [[nodiscard]] uint64_t link_state_epoch() const noexcept;

    /// Phase-7 §10.2: number of directed links tracked by the scheduler.
    [[nodiscard]] size_t link_count() const noexcept;

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
    bool deferred_flush_mode;

    uint64_t current_dirty_epoch;
    std::vector<FlowId> dirty_flow_ids;
    FlowId next_flow_id;

    uint64_t active_flow_count;
    uint64_t active_route_memberships;
    uint64_t total_started_flows;
    uint64_t total_completed_flows;
    uint64_t max_active_flows;
    uint64_t max_route_memberships;
    uint64_t link_state_epoch_;

    uint64_t progress_report_event_interval;
    uint64_t scheduler_event_count;
    uint64_t dirty_batch_count;
    uint64_t total_dirty_flows;
    uint64_t max_dirty_flows;
    size_t peak_completion_heap_size;
    std::chrono::steady_clock::time_point wall_start_time;
};

}  // namespace NetworkAnalyticalCongestionAware
