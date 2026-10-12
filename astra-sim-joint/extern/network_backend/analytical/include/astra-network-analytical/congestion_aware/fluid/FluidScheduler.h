/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#pragma once

#include "common/EventQueue.h"
#include "common/WallClock.h"
#include "congestion_aware/Link.h"
#include "congestion_aware/fluid/FluidFlow.h"
#include "congestion_aware/fluid/FluidLinkState.h"
#include <chrono>
#include <cstddef>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <queue>
#include <unordered_map>
#include <utility>
#include <vector>

namespace NetworkAnalyticalCongestionAware {

class FluidScheduler {
  public:
    FluidScheduler(std::shared_ptr<NetworkAnalytical::EventQueue> event_queue,
                   const std::vector<std::shared_ptr<const Link>>& directed_links,
                   uint64_t max_active_flows,
                   uint64_t max_route_memberships,
                   uint64_t progress_report_event_interval) noexcept;

    ~FluidScheduler() noexcept;

    void start_flow(NetworkAnalytical::ChunkSize bytes,
                    std::shared_ptr<const FluidRoute> route,
                    NetworkAnalytical::Callback callback,
                    NetworkAnalytical::CallbackArg callback_arg,
                    int src_rank = -1, int dst_rank = -1,
                    std::optional<LogicalFlowKey> logical_key = std::nullopt) noexcept;

    /// Invoked at the existing pending-start bind point, after the ordinary
    /// FlowId has been assigned and inserted into the active table.
    void set_on_flow_bound(
        std::function<void(const LogicalFlowKey&, FlowId)> callback) {
        on_flow_bound_ = std::move(callback);
    }

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

    // (flush_pending_starts_deferred was removed with the 2026-10-01
    // dead-code audit: it had no caller -- start_flow's deferred arm
    // schedules the shared flush_callback directly, which is the ONLY
    // post-commit emission channel in online mode.)

    void mark_event_loop_started() noexcept;
    void handle_service_wakeup(uint64_t generation) noexcept;
    void handle_tail_arrival(FlowId flow_id) noexcept;

    /// Per-directed-link version for D1 window invalidation. It changes only
    /// when active membership or a member's allocated flow rate changes.
    [[nodiscard]] uint64_t link_resource_version(LinkId link_id) const;

    /// Observe an actual batch membership/rate transition. The before call
    /// follows integration under old rates and receives the conservative set
    /// of links that may change; the after call receives the exact changed
    /// set after rates and versions have been committed. Observers must not
    /// re-enter this scheduler, start flows, or publish workload completions.
    using LinkResourceMutationObserver = std::function<void(
        bool before_change, NetworkAnalytical::EventTime tick,
        const std::vector<LinkId>& affected_links)>;
    [[nodiscard]] uint64_t add_link_resource_mutation_observer(
        LinkResourceMutationObserver observer);
    void remove_link_resource_mutation_observer(uint64_t observer_id) noexcept;

    [[nodiscard]] uint64_t get_active_flow_count() const noexcept;
    [[nodiscard]] uint64_t get_active_route_memberships() const noexcept;
    [[nodiscard]] uint64_t get_total_started_flows() const noexcept;
    [[nodiscard]] uint64_t get_total_completed_flows() const noexcept;

    /// Phase-7 §10.2: number of directed links tracked by the scheduler.
    [[nodiscard]] size_t link_count() const noexcept;

    /// -------------------------------------------------------------------
    /// WP6 NoC link observer (SLO pipeline B2, /tmp/slo_wps/plans/
    /// CPP_SPEC.md §D): read-only side-band integration of the fluid bytes
    /// each directed link carried. The observer hooks the scheduler's
    /// existing rate/membership change points (advance_dirty_flows entry,
    /// BEFORE any mutation), integrates bytes_delta = rate x dt per link
    /// into fixed time buckets, and never registers a simulation event,
    /// never advances a flow, and never touches scheduling state.
    ///
    /// Zero work until enable_link_observer() is called; the online main
    /// gates that on MetricCollector::enabled() (metrics != off) AND env
    /// ASTRA_LINK_OBSERVER != 0, passing the slo_sampling link bucket.
    /// Fractional byte accumulation keeps a per-link carry so the bucket
    /// sums reconcile with the integer transfer byte counts.
    struct LinkObserverTotals {
        uint64_t total_bytes;  ///< whole bytes integrated over the window
        uint64_t active_ns;    ///< time with at least one active flow
        /// M1 (2026-09-23 acceptance audit): sum over active time of the
        /// per-link active-flow count, i.e. time-weighted flow-seconds.
        /// window average flow count = flow_active_ns / active_ns; this is
        /// the aggregate-throughput-blind denominator the SH-side pricing
        /// divisors and AIMD need (aggregate throughput cannot recover the
        /// flow count: a saturated link reports ~capacity regardless of N,
        /// and a downstream-capped flow would be misread as contention).
        uint64_t flow_active_ns;
    };

    /// Finalization callback for one non-empty (bucket, link) contribution.
    /// Calls arrive in strictly increasing bucket order and ascending LinkId
    /// order within each bucket. Returning false is treated as a fail-closed
    /// output failure.
    using LinkObserverBucketVisitor = bool (*)(void*, uint64_t, LinkId, uint64_t);

    /// Bounded-residency accounting exposed for the focused observer test.
    /// spool_record_count is on-disk logical history, never resident rows.
    struct LinkObserverStorage {
        size_t resident_bucket_entries;
        size_t resident_bucket_capacity;
        size_t active_link_count;
        uint64_t spool_record_count;
        bool spool_open;
    };

    void enable_link_observer(uint64_t link_bucket_ns) noexcept;


    /// Bucket length actually in effect (echoed into every link record).
    [[nodiscard]] uint64_t link_observer_bucket_ns() const noexcept;

    /// Last integrated tick == observer window end (bytes past the last
    /// scheduler event are not attributed).
    [[nodiscard]] NetworkAnalytical::EventTime link_observer_window_ns() const noexcept;

    /// Close the last partial bucket, then replay each non-empty contribution
    /// from the anonymous spool. No all-run bucket array is materialized.
    void link_observer_visit_buckets(LinkObserverBucketVisitor visitor,
                                     void* context) noexcept;

    [[nodiscard]] const std::vector<LinkObserverTotals>&
    link_observer_totals() const noexcept;

    [[nodiscard]] LinkObserverStorage link_observer_storage() const noexcept;

    /// Free the integration arrays after the records were emitted (RSS
    /// discipline; the observer stays disabled afterwards).
    void link_observer_release() noexcept;

    /// -------------------------------------------------------------------
    /// W1 (joint work ledger, 2026-09-29): per-flow / per-link state
    /// snapshot for the causal work ledger. The query integrates live flows
    /// to the online tick-end static point before copying them, so
    /// remaining_bytes and as_of_tick describe the actual decision boundary.
    /// -------------------------------------------------------------------
    struct FlowSnapshotEntry {
        FlowId flow_id;
        int src_rank = -1;
        int dst_rank = -1;
        uint64_t total_bytes;
        long double remaining_bytes;
        long double current_rate_Bpns;
        NetworkAnalytical::EventTime last_rate_update_time;
        std::vector<LinkId> route;
        int state;  // FluidFlowState raw value (0 Active, 1 tail, 2 done)
        std::optional<LogicalFlowKey> logical_key;
    };

    struct PendingFlowSnapshotEntry {
        FlowId assigned_flow_id;
        int src_rank = -1;
        int dst_rank = -1;
        uint64_t total_bytes = 0;
        std::vector<LinkId> route;
        std::optional<LogicalFlowKey> logical_key;
    };

    struct LinkSnapshotEntry {
        LinkId link_id;
        NetworkAnalytical::Bandwidth capacity_Bpns;
        uint32_t active_flow_count;
    };

    /// Live flows (Active / PropagatingTail; completed flows are erased from
    /// flows_by_id at their completion), ascending flow_id.
    [[nodiscard]] std::vector<FlowSnapshotEntry> flow_snapshot() noexcept;

    /// Starts assigned an id but not yet bound by flush_pending_starts().
    /// This separate view preserves flow_snapshot()'s active-flow meaning.
    [[nodiscard]] std::vector<PendingFlowSnapshotEntry>
    pending_flow_snapshot() const;

    /// All directed links with their capacity and active-flow count.
    [[nodiscard]] std::vector<LinkSnapshotEntry> link_snapshot() const noexcept;

    /// Current event-queue tick of the frame produced by flow_snapshot().
    /// Tails can retain an older last_rate_update_time after leaving link
    /// arbitration, so the frame watermark comes from the boundary itself.
    [[nodiscard]] NetworkAnalytical::EventTime
    flow_snapshot_as_of_tick() const noexcept;

    /// -------------------------------------------------------------------
    /// Additive wall-clock instrumentation (measurement only; the brackets
    /// never gate scheduling logic). Buckets accumulate total ns + call
    /// count; the counters complement total_started_flows /
    /// total_completed_flows. Mutable solely because report_progress() is
    /// const, matching the wall_start_time idiom.
    /// -------------------------------------------------------------------
    struct SchedulerWallStats {
        NetworkAnalytical::WallClockBucket flush_pending_starts;
        /// advance_dirty_flows() + recalculate_dirty_rates() combined
        /// (each call increments calls by one).
        NetworkAnalytical::WallClockBucket dirty_advance_recalc;
        NetworkAnalytical::WallClockBucket service_wakeup;
        /// Whole handle_tail_arrival(), INCLUDING the completion callback.
        NetworkAnalytical::WallClockBucket tail_arrival;
        /// The completion callback invoked at the end of handle_tail_arrival()
        /// alone (subset of tail_arrival) -- the prime suspect for the
        /// per-flow wall constant.
        NetworkAnalytical::WallClockBucket completion_callback;
        uint64_t completion_heap_stale_pops = 0;
        uint64_t rebuild_calls = 0;
        uint64_t rebuilds_performed = 0;
    };

    [[nodiscard]] const SchedulerWallStats& wall_clock_stats() const noexcept {
        return wall_stats_;
    }

  private:
    struct PendingFlowStart {
        FlowId flow_id;
        NetworkAnalytical::ChunkSize bytes;
        std::shared_ptr<const FluidRoute> route;
        NetworkAnalytical::Callback callback;
        NetworkAnalytical::CallbackArg callback_arg;
        int src_rank = -1;
        int dst_rank = -1;
        std::optional<LogicalFlowKey> logical_key;
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

    static void flush_callback(void* context) noexcept;
    static void service_wakeup_callback(void* context) noexcept;
    static void cancel_wakeup_callback(void* context) noexcept;
    static void tail_arrival_callback(void* context) noexcept;

    /// WP6 link observer: integrate [last_tick, now) with the CURRENT
    /// (pre-change) rates and memberships. Called at the top of
    /// advance_dirty_flows -- the single choke point both mutation paths
    /// (flush_pending_starts / handle_service_wakeup) pass through before
    /// touching any rate or membership.
    void link_observer_integrate(NetworkAnalytical::EventTime now) noexcept;
    void link_observer_adjust_flow_rate(const FluidFlow& flow,
                                        NetworkAnalytical::Bandwidth old_rate,
                                        NetworkAnalytical::Bandwidth new_rate) noexcept;
    void link_observer_activate_link(LinkId link_id) noexcept;
    void link_observer_deactivate_link(LinkId link_id) noexcept;
    void link_observer_flush_current_bucket() noexcept;
    void link_observer_finish() noexcept;
    void link_observer_write_u64(uint64_t value) noexcept;
    [[noreturn]] void link_observer_fail(const char* reason) const noexcept;
    [[noreturn]] void link_observer_fail_io(const char* operation) const noexcept;

    void begin_dirty_batch() noexcept;
    void mark_dirty(FlowId flow_id) noexcept;
    void advance_live_flows_to_current_time() noexcept;
    void advance_dirty_flows(NetworkAnalytical::EventTime now) noexcept;
    void recalculate_dirty_rates(NetworkAnalytical::EventTime now) noexcept;
    [[nodiscard]] std::vector<LinkId> dirty_route_links(
        const std::vector<LinkId>& membership_links) const;
    [[nodiscard]] std::vector<LinkId> changed_rate_links(
        const std::vector<std::pair<FlowId, NetworkAnalytical::Bandwidth>>&
            old_rates,
        const std::vector<FlowId>& excluded_flows = {}) const;
    void bump_link_resource_versions(
        const std::vector<LinkId>& link_ids) noexcept;
    void notify_link_resource_mutation(
        bool before_change, NetworkAnalytical::EventTime tick,
        const std::vector<LinkId>& affected_links) noexcept;
    void remove_memberships(FluidFlow& flow) noexcept;
    void clean_completion_heap() noexcept;
    void maybe_rebuild_completion_heap() noexcept;
    void cancel_scheduled_wakeup() noexcept;
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
    std::vector<uint64_t> link_resource_versions_;
    std::vector<PendingFlowStart> pending_starts;
    std::function<void(const LogicalFlowKey&, FlowId)> on_flow_bound_;
    std::map<uint64_t, LinkResourceMutationObserver>
        link_resource_mutation_observers_;
    uint64_t next_link_resource_observer_id_ = 1;

    std::priority_queue<CompletionEntry, std::vector<CompletionEntry>, CompletionLater> completion_heap;
    uint64_t wakeup_generation;
    std::optional<NetworkAnalytical::EventTime> scheduled_wakeup_time;
    NetworkAnalytical::EventHandle scheduled_wakeup_event;
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

    uint64_t progress_report_event_interval;
    uint64_t scheduler_event_count;
    uint64_t dirty_batch_count;
    uint64_t total_dirty_flows;
    uint64_t max_dirty_flows;
    size_t peak_completion_heap_size;
    std::chrono::steady_clock::time_point wall_start_time;

    /// instrumentation buckets (see the public SchedulerWallStats block)
    mutable SchedulerWallStats wall_stats_;

    /// WP6 link observer state (empty/disabled until enabled by the online
    /// main; see the public block above). Only per-link state and the current
    /// bucket remain resident; closed bucket records live in an anonymous
    /// tmpfile() until the ordered end-of-run replay.
    struct LinkObserverState {
        bool enabled = false;
        uint64_t bucket_ns = 0;
        EventTime last_tick = 0;
        std::vector<long double> carry;                // per-link fractional bytes
        std::vector<long double> rate_sum;             // maintained per-link Bpns sum
        std::vector<uint64_t> total_bytes;             // per-link whole bytes
        std::vector<uint64_t> active_ns;               // per-link active time
        // M1: per-link time-weighted flow count integral (dt * active
        // flow count per integration segment).
        std::vector<uint64_t> flow_active_ns;
        std::vector<LinkId> active_links;              // sorted links with memberships
        std::vector<uint64_t> current_bucket_bytes;    // one bucket, indexed by link
        std::vector<LinkId> current_bucket_links;      // non-zero rows in that bucket
        uint64_t current_bucket = 0;
        bool current_bucket_open = false;
        std::FILE* spool = nullptr;                    // tmpfile(), unlinked on open
        uint64_t spool_record_count = 0;
        bool finalized = false;
        mutable std::vector<LinkObserverTotals> totals_scratch;  // query result
    };
    LinkObserverState link_observer_;
};

}  // namespace NetworkAnalyticalCongestionAware
