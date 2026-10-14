/******************************************************************************
Bounded, descriptor-driven remote-read execution over real Sys endpoints.
*******************************************************************************/

#ifndef EXECUTION_DRIVEN_REMOTE_READ_EXECUTOR_HH
#define EXECUTION_DRIVEN_REMOTE_READ_EXECUTOR_HH

#include <cstdint>
#include <array>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/workload/execution_driven/GraphBatchCommitter.hh"
#include "astra-sim/workload/execution_driven/NodeStore.hh"
#include "astra-sim/workload/execution_driven/PipelineDescriptor.hh"
#include "astra-sim/workload/execution_driven/TileTransportAdapter.hh"

namespace AstraSim {

class Workload;

namespace ExecutionDriven {

/// Owns mutable per-descriptor run cursors. Descriptor and tile data remain
/// immutable and compact; only one current endpoint per affine run is held.
class RemoteReadExecutor final {
  public:
    struct ProducerMetrics {
        uint64_t resource_row_capture_ns = 0;
        uint64_t resource_row_capture_calls = 0;
        uint64_t row_helper_visits = 0;
        uint64_t diagnostic_row_helper_visits = 0;
        uint64_t authoritative_snapshot_calls = 0;
        uint64_t blocked_diagnostics_calls = 0;
        uint64_t cursor_patch_capture_ns = 0;
        uint64_t cursor_patch_capture_calls = 0;
        uint64_t cursor_run_captures = 0;
        uint64_t transition_publish_ns = 0;
        uint64_t transition_publish_calls = 0;
        uint64_t actual_issue_ns = 0;
        uint64_t actual_issue_calls = 0;
        uint64_t endpoint_index_candidate_examinations = 0;
        uint64_t endpoint_index_entries_built = 0;
        uint64_t endpoint_index_sort_comparisons = 0;
        uint64_t endpoint_index_tree_build_operations = 0;
        uint64_t endpoint_index_block_search_steps = 0;
        uint64_t endpoint_index_blocks_examined = 0;
        uint64_t current_cursor_candidates_examined = 0;
        uint64_t terminal_watcher_index_entries_built = 0;
        uint64_t terminal_watcher_index_insert_steps = 0;
        uint64_t terminal_watcher_index_query_node_visits = 0;
        uint64_t terminal_watcher_index_candidate_examinations = 0;
        uint64_t anchor_resource_queries = 0;
        uint64_t anchor_coordinate_rejections = 0;
        uint64_t anchor_dynamic_preflights = 0;
        uint64_t virtual_physical_cursor_skips = 0;
        uint64_t slot_patch_rows = 0;
        uint64_t credit_patch_rows = 0;
        uint64_t obligation_patch_rows = 0;
        uint64_t obligation_completed_count_reads = 0;
        uint64_t cursor_run_patch_rows = 0;
        uint64_t target_delta_patch_rows = 0;
        uint64_t target_claim_shard_elements = 0;
        uint64_t target_consume_added_shards = 0;
        uint64_t target_delta_operations = 0;
        uint64_t target_full_row_json_builds = 0;
    };

    struct ReadyEndpoint {
        uint64_t store_id = 0;
        NodeView resource_node;
    };

    using ScheduleIssuePass = std::function<void(int rank)>;
    using PipelineTransitionPublisher = std::function<void(nlohmann::json)>;
    using DescriptorCommitPublisher = std::function<void(nlohmann::json)>;

    struct WatchMemberProgression {
        int rank = -1;
        uint64_t first_store_id = 0;
        uint64_t count = 0;
        uint64_t stride = 0;
    };
    using WatchReferenceCounter =
        std::function<uint64_t(const std::vector<WatchMemberProgression>&)>;

    struct ResourceMutation {
        bool before = false;
        std::string resource_kind;
        std::string stable_resource_key;
        uint64_t version = 0;
        uint64_t tick = 0;
    };
    using ResourceMutationObserver =
        std::function<void(const ResourceMutation&)>;
    using ResourceMutationObserverToken = uint64_t;

    RemoteReadExecutor(
        std::vector<std::shared_ptr<NodeStoreGraphSource>> graph_sources,
        std::vector<Workload*> workloads,
        TileTransportAdapter* transport,
        ScheduleIssuePass schedule_issue_pass);
    ~RemoteReadExecutor();
    RemoteReadExecutor(const RemoteReadExecutor&) = delete;
    RemoteReadExecutor& operator=(const RemoteReadExecutor&) = delete;

    /// Pure prepare hook for GraphBatchCommitter Phase A. All endpoint and
    /// anchor IDs are resolved against the future affine spans, but no
    /// registry/cursor/store state is changed.
    std::shared_ptr<void> prepare_batch(
        const GraphBatch& batch,
        const GraphBatchCommitter& committer) const;

    /// Commit the prepared plan after graph, edges, watches and alarms have
    /// been installed, but before any rank issue pass can see its roots.
    void commit_batch(std::shared_ptr<void> prepared,
                      const GraphBatch& batch,
                      const GraphBatchCommitter& committer);

    /// Current compact virtual frontier for one rank, in logical store-ID
    /// order. Each run contributes at most one candidate; no tile array is
    /// materialized.
    std::vector<ReadyEndpoint> ready_virtual_endpoints(int rank) const;

    /// Translate a structural physical anchor to the synthetic COMM/COMP
    /// view whose real hardware availability gate it represents. False means
    /// this anchor is not the current ready operation of its descriptor.
    bool resource_view_for_anchor(const NodeView& anchor,
                                  NodeView* resource_view) const;

    // F (dynamic ordered current physical anchor candidate set, 2026-10-11).
    // The issue pass no longer enumerates every ready anchor from the
    // NodeStore free set (97% were coordinate-rejected future anchors);
    // instead the executor maintains, per rank, the ordered set of still
    // registered physical anchors whose run cursor currently points at them.
    // Membership is updated inside the cursor/binding lifecycle
    // (commit_batch registration, set_current_cursor / erase_current_cursor,
    // retirement) and stores plain store ids -- never pointers -- so a
    // callback that deletes the owning source simply makes the id
    // unfindable. The updates themselves never fire callbacks.
    /// Smallest current anchor id strictly greater than after_id
    /// (lower_bound(after_id + 1) semantics; after_id is the walk's last
    /// visited id, initially 0). False when none remains.
    bool next_current_anchor_id(uint64_t rank, uint64_t after_id,
                                uint64_t* out) const;
    /// Current-anchor-set size of one rank (counters/diagnostics).
    size_t current_anchor_count(uint64_t rank) const;
    /// Membership probe for tests/diagnostics.
    bool is_current_anchor_id(uint64_t rank, uint64_t id) const;

    /// Dispatch an already-gated physical anchor. Workload has taken and
    /// occupied the zero-cost structural node; completion returns through its
    /// single shared terminal path.
    bool dispatch_anchor(const NodeView& anchor);

    /// Recheck and issue one virtual candidate selected from a stable ready
    /// snapshot. The transport owns the actual COMM/COMP gate until both real
    /// endpoint legs have completed.
    bool dispatch_virtual_endpoint(int rank, uint64_t store_id);

    /// Install the single run-global C4 transition publisher. Ordinary
    /// endpoint/marker lifecycle rows and pipeline cursor rows are emitted
    /// synchronously from the actual issue/completion/commit callbacks.
    void set_ledger_publishers(
        LedgerNodeTransitionPublisher node_publisher,
        PipelineTransitionPublisher pipeline_publisher,
        DescriptorCommitPublisher descriptor_publisher);

    /// Query the authoritative stage-watch reverse index before retiring a
    /// settled descriptor. The ranges are compact NodeStore-ID progressions;
    /// a missing provider prevents retirement rather than assuming no pins.
    void set_watch_reference_counter(WatchReferenceCounter counter);
    /// Recheck settled descriptors after WatchRegistry removes fired watches.
    void notify_watch_registry_changed();

    /// Register a D1 observer for executor-owned control resources. The
    /// observer is held only while its lifetime token remains alive and can
    /// be explicitly removed before teardown. It receives paired before /
    /// after events around each real state mutation.
    ResourceMutationObserverToken add_resource_mutation_observer(
        std::weak_ptr<void> lifetime,
        ResourceMutationObserver observer);
    void remove_resource_mutation_observer(
        ResourceMutationObserverToken token);
    std::vector<std::pair<std::string, uint64_t>>
    resource_version_vector() const;

    /// Bind a descriptor's cursor-only existing_service_gate to the original
    /// stage-watch fire. Settlement remains stricter and also waits for every
    /// endpoint run and owned physical callback to close.
    void notify_service_gate(const std::string& request_id,
                             const std::string& stage,
                             uint64_t generation);

    /// Observe a completed ordinary body node registered as the terminal
    /// consumer of one or more target-side descriptor frontiers. This is a
    /// deferred readiness fact only; it never starts a transfer inline.
    void notify_node_completed(int rank, uint64_t store_id);

    /// Administrative request close (main-agent ruling 2026-10-06): at
    /// request completion the staging lease retires -- unconsumed slots are
    /// released, never-issued tail endpoints retire with the lease (their
    /// consume watchers can no longer block settle), while every already
    /// issued data/ACK/HBM physical leg keeps draining naturally before
    /// settle. Mirrors the old engine's REQUEST_COMPLETE behavior where the
    /// tail tiles' legs completed as ordinary graph nodes with nobody left
    /// waiting on their consumption.
    void notify_request_closed(const std::string& request_id);

    /// Terminal-gate consume-obligation retirement (main-agent ruling
    /// 2026-10-07, rounds 5+9 synthesis). Semantic split from
    /// notify_request_closed: DURING close processing the consume_pending
    /// balance is untouchable (in-flight deferred deliveries still need it
    /// to pay); but at the settle/retire TERMINAL gate a still-live
    /// consume balance whose obligations can no longer receive ANY
    /// delivery is dead bookkeeping and retires with the lease (round 5:
    /// a never-firing consume watcher must not block settle/retire).
    /// Four-condition guard, all checked here or guaranteed by the
    /// caller's provably-no-future-events contract (the same contract as
    /// TileTransportAdapter::retire_stranded_transfers):
    ///   (i) the descriptor's request is already closed;
    ///  (ii) the event queue is finished and (iv) no deferred work
    ///       remains -- caller-guaranteed;
    /// (iii) no transport leg is active anywhere (a live leg could still
    ///       deliver a terminal that pays a consume obligation).
    /// The retirement is auditable, never silent: each descriptor records
    /// its administratively retired count (certificate + settle-gate
    /// diagnostics carry it) and settles/retires through the ordinary
    /// transitions. Affected ranks get a coalesced issue pass (round-18
    /// missed-wake precedent).
    void retire_dead_consume_obligations();

    /// Per-descriptor audit summary of the terminal-gate retirement:
    /// [{segment_id, retired_count}] for every descriptor with a nonzero
    /// administratively retired consume-obligation count (committed AND
    /// already-retired descriptors included); null when nothing retired,
    /// so unaffected tapes keep byte-identical certificates.
    nlohmann::json administratively_retired_consume_summary() const;

    /// Current bounded execution state used by diagnostics and legacy tests.
    nlohmann::json authoritative_snapshot(uint64_t now_tick) const;
    /// Final changed cursor-run rows for the next sparse protocol-v4 frame.
    /// This serializes only rows touched by callbacks since the last drain.
    nlohmann::json take_cursor_patches(uint64_t now_tick);
    [[nodiscard]] const ProducerMetrics& producer_metrics() const {
        return producer_metrics_;
    }
    /// Final-state diagnostics (fixture audit sidecar only): for every
    /// active, non-exhausted run cursor report its current position and
    /// the first unresolved dependency (store id, kind, completion
    /// state) or the blocking protocol gate. One capture adjudicates
    /// cross-batch dependency deadlocks (wire cycle vs mapping gap).
    nlohmann::json blocked_endpoints_diagnostics() const;
    /// Resolve an active endpoint's immutable C4 selector/context from its
    /// NodeStore-domain id. Missing means the resource is not pipeline-owned.
    std::optional<nlohmann::json> ledger_endpoint_binding(
        int rank, uint64_t store_id) const;
    uint64_t active_descriptor_count() const {
        return active_descriptors_.size();
    }
    uint64_t active_cursor_count() const;
    bool is_drained() const;

  private:
    friend class RemoteReadExecutorTestProbe;

    /// Count ordinary-terminal IDs that resolve to physical nodes in the
    /// candidate batch. Probe the requested affine IDs in the exact batch
    /// index instead of rescanning every parsed node for each terminal run.
    /// `membership_probes` is optional test evidence; production passes null.
    static uint64_t count_ordinary_terminal_candidate_matches(
        const std::unordered_map<uint64_t, const ParsedNode*>& nodes_by_id,
        uint64_t first_json_id, uint64_t count, uint64_t stride,
        uint64_t* membership_probes);

    struct RunCursor {
        const PipelineDescriptor* descriptor = nullptr;
        const PipelineEndpointRun* run = nullptr;
        uint64_t first_store_id = 0;
        uint64_t initial_parent_store_id = 0;
        bool has_initial_parent = false;
        uint64_t tile_offset = 0;
        size_t shard_index = 0;
        size_t leg_index = 0;
        uint64_t in_flight_count = 0;
        uint64_t completed_prefix = 0;
        // Exact count of completed endpoints in completed_prefix plus the
        // disjoint completed_out_of_order ranges. Request-close cancellation
        // is deliberately excluded.
        uint64_t completed_count = 0;
        // Completed intervals remain compact when a short shard leaves a
        // long shared tile axis. Live callbacks add singleton intervals;
        // inactive suffixes are recorded as one affine cursor range.
        std::map<uint64_t, uint64_t> completed_out_of_order;
        std::map<uint64_t, uint64_t> active_store_ids;
        uint64_t cancelled_endpoint_count = 0;
        bool exhausted = false;
    };

    struct ResolvedDependencyRun {
        const PipelineEndpointDependencyRun* run = nullptr;
        uint64_t first_store_id = 0;
    };

    struct DependencyIntervalEntry {
        uint64_t first_tile = 0;
        uint64_t last_tile = 0;
        size_t dependency_run_index = 0;
        size_t local_shard_index = 0;
    };

    struct DependencyIntervalIndex {
        std::vector<DependencyIntervalEntry> entries;
        // Entries are ordered by first_tile. This max-end segment tree reports
        // every interval containing one group-local tile without expanding
        // any represented tile positions.
        size_t leaf_base = 0;
        std::vector<uint64_t> subtree_max_last_tile;
    };

    using DependencyIntervalKey =
        std::tuple<int, uint64_t, PipelineEndpointLeg, uint64_t>;
    using DependencyIntervalIndexes =
        std::map<DependencyIntervalKey, DependencyIntervalIndex>;

    struct DependencyIndexMatch {
        size_t dependency_run_index = 0;
        size_t local_shard_index = 0;
    };

    struct ResolvedConsumeRun {
        const PipelineConsumeCompletionRun* run = nullptr;
        uint64_t first_store_id = 0;
    };

    struct HistoricalRunProgress {
        const PipelineEndpointRun* run = nullptr;
        uint64_t first_store_id = 0;
        uint64_t completed_prefix = 0;
        uint64_t cancelled_endpoint_count = 0;
    };

    struct RuntimeDescriptor {
        std::shared_ptr<const PipelineDescriptor> descriptor;
        std::vector<RunCursor> runs;
        std::vector<HistoricalRunProgress> historical_runs;
        std::vector<ResolvedDependencyRun> dependency_runs;
        DependencyIntervalIndexes dependency_interval_indexes;
        std::vector<ResolvedConsumeRun> consume_runs;
        uint64_t consume_pending_count = 0;
        // Audit ledger for the terminal-gate consume-obligation retirement
        // (see retire_dead_consume_obligations): how many consume
        // obligations were administratively retired with the lease.
        uint64_t administratively_retired_consume_obligations = 0;
        uint64_t cursor_version = 0;
        uint64_t commit_tick = 0;
        uint64_t logical_id_base = 0;
        uint64_t total_endpoint_count = 0;
        uint64_t outstanding_count = 0;
        std::vector<WatchMemberProgression> watch_member_ranges;
        std::vector<std::pair<int, uint64_t>> physical_anchor_store_ids;
        bool service_done = false;
        bool settle_done = false;
        bool request_closed = false;
        bool retired = false;
        bool queued_for_watch_retirement = false;
    };

    struct TerminalWatcher {
        RuntimeDescriptor* owner = nullptr;
        const PipelineConsumeCompletionRun* consume_run = nullptr;
        size_t consume_run_index = 0;
        int wake_rank = -1;
        uint64_t group_ordinal = 0;
        PipelineEndpointDependencyKind kind =
            PipelineEndpointDependencyKind::OrdinaryNodeTerminal;
        bool is_consume_completion = false;
        uint64_t first_store_id = 0;
        uint64_t tile_count = 0;
        uint64_t tile_stride = 0;
        std::vector<uint64_t> shard_offsets;
        // Signature 6 (2026-10-07): per-watcher consume-obligation ledger.
        // total_obligations = registered consume mappings; preaccounted =
        // mappings whose terminal was ALREADY finished at commit time (the
        // owner-global pending count excludes them, yet a late re-delivery
        // of such a terminal is legal -- the committer wakes watchers for
        // pre-completed parents). delivered tracks notifications so a true
        // over-delivery still fails closed.
        uint64_t total_obligations = 0;
        uint64_t preaccounted = 0;
        uint64_t delivered = 0;
    };

    struct TerminalWatcherInterval {
        uint64_t first_store_id = 0;
        uint64_t last_store_id = 0;
        size_t watcher_index = 0;
        size_t shard_index = 0;
    };

    struct TerminalWatcherIntervalNode {
        explicit TerminalWatcherIntervalNode(
            TerminalWatcherInterval value)
            : interval(value),
              subtree_min_first_store_id(this->interval.first_store_id),
              subtree_max_last_store_id(this->interval.last_store_id) {}

        TerminalWatcherInterval interval;
        std::unique_ptr<TerminalWatcherIntervalNode> left;
        std::unique_ptr<TerminalWatcherIntervalNode> right;
        uint64_t subtree_min_first_store_id = 0;
        uint64_t subtree_max_last_store_id = 0;
        int height = 1;
    };

    struct TerminalWatcherProgressionIndex {
        std::unique_ptr<TerminalWatcherIntervalNode> root;
    };

    struct TerminalWatcherMatch {
        size_t watcher_index = 0;
        uint64_t completed_obligations = 0;
    };

    struct ConsumeMutationSnapshot {
        RuntimeDescriptor* owner = nullptr;
        uint64_t cursor_version_start = 0;
    };

    struct SlotState {
        bool occupied = false;
        uint64_t owner_tile = 0;
        uint64_t generation = 0;
        bool consumed = false;
        std::string owner_segment_id;
        uint64_t owner_pipeline_generation = 0;
        uint64_t owner_stage_epoch = 0;
        uint64_t owner_staging_allocation_id = 0;
        uint64_t owner_group_ordinal = 0;
        uint64_t owner_local_tile = 0;
        uint64_t remaining_target_count = 0;
        std::set<uint64_t> target_shards_to_consume;
        std::set<uint64_t> consumed_target_shards;
    };

    using TargetSlotStableKey =
        std::tuple<std::string, std::string, int,
                   std::optional<uint64_t>, size_t>;

    struct PendingTargetSlotDelta {
        uint64_t before_slot_generation = 0;
        uint64_t predecessor_consumed_count = 0;
        bool before_present = false;
        std::vector<nlohmann::json> operations;
    };

    struct BufferDomainState {
        std::array<SlotState, 2> target_slots{};
        std::map<std::pair<int, uint64_t>, std::array<SlotState, 2>>
            source_slots;
        std::map<std::pair<int, uint64_t>, uint64_t> outstanding_release_acks;
        // Lag-2 release watermark per source slot ((rank, shard) -> per
        // slot index): the highest old-tile ordinal whose reuse a completed
        // release ACK has already authorized. A slot still occupied by an
        // owner at or below this watermark is a stale residual (the ACK
        // completed before that owner's claim, so the reserve-time capture
        // could not see it) and must not block the next data leg.
        std::map<std::pair<int, uint64_t>, std::array<uint64_t, 2>>
            source_slot_released_through{};
    };

    // Compact protocol provenance survives quiescent release of the heavy
    // occupancy rows. A same-request continuation may legally recreate the
    // domain and must retain slot generations and source release watermarks.
    struct BufferDomainProvenance {
        int target_rank = -1;
        std::array<uint64_t, 2> target_slot_generations{};
        std::map<std::pair<int, uint64_t>, std::array<uint64_t, 2>>
            source_slot_generations;
        std::map<std::pair<int, uint64_t>, std::array<uint64_t, 2>>
            source_slot_released_through;
    };

    struct AnchorBinding {
        RuntimeDescriptor* owner = nullptr;
        size_t run_index = 0;
        uint64_t tile_ordinal = 0;
        size_t shard_index = 0;
        size_t leg_index = 0;
        PipelineAnchorRole role = PipelineAnchorRole::Start;
    };

    struct EndpointProgressionIndexEntry {
        RuntimeDescriptor* owner = nullptr;
        size_t run_index = 0;
        size_t shard_index = 0;
        size_t leg_index = 0;
        uint64_t first_store_id = 0;
        uint64_t last_store_id = 0;
        uint64_t tile_stride = 0;
        uint64_t tile_count = 0;
        uint64_t insertion_order = 0;
    };

    struct EndpointProgressionIndexBlock {
        uint64_t first_store_id = 0;
        uint64_t last_store_id = 0;
        std::vector<EndpointProgressionIndexEntry> entries;
        // A max-endpoint segment tree supports point and interval queries
        // without walking unrelated earlier descriptor history.
        size_t leaf_base = 0;
        std::vector<uint64_t> max_last_store_id_tree;
    };

    struct EndpointProgressionIndex {
        std::vector<EndpointProgressionIndexBlock> blocks;
    };

    struct CurrentCursorBinding {
        RuntimeDescriptor* owner = nullptr;
        size_t run_index = 0;
    };

    struct PendingCreditRowCapture {
        std::string stable_key;
        nlohmann::json before;
    };

    struct EndpointPosition {
        RuntimeDescriptor* owner = nullptr;
        size_t run_index = 0;
        uint64_t tile_offset = 0;
        size_t shard_index = 0;
        size_t leg_index = 0;
        uint64_t slot_generation = 0;
        bool releases_source_slot = false;
        uint64_t released_slot_tile = 0;
        uint64_t released_slot_generation = 0;
        std::string released_slot_segment_id;
        uint64_t released_slot_pipeline_generation = 0;
        uint64_t released_slot_stage_epoch = 0;
        uint64_t released_slot_staging_allocation_id = 0;
    };

    struct PreparedPlan;

    struct ResourceObserverEntry {
        std::weak_ptr<void> lifetime;
        ResourceMutationObserver observer;
    };

    static uint64_t position_ordinal(const RunCursor& cursor,
                                     uint64_t tile_offset,
                                     size_t shard_index,
                                     size_t leg_index);
    static void build_dependency_interval_indexes(
        const std::vector<ResolvedDependencyRun>& dependency_runs,
        DependencyIntervalIndexes* indexes);
    static bool query_dependency_interval_node(
        const DependencyIntervalIndex& index, size_t tree_node,
        size_t first_entry, size_t end_entry, uint64_t tile_ordinal,
        std::vector<DependencyIndexMatch>* matches,
        uint64_t* visited_nodes);
    static std::vector<DependencyIndexMatch> matching_dependency_runs(
        const RuntimeDescriptor& owner, int rank, uint64_t group_ordinal,
        PipelineEndpointLeg endpoint, uint64_t tile_ordinal,
        uint64_t shard_ordinal, uint64_t* visited_nodes = nullptr);
    static bool has_matching_dependency_run(
        const RuntimeDescriptor& owner, int rank, uint64_t group_ordinal,
        PipelineEndpointLeg endpoint, uint64_t tile_ordinal,
        uint64_t shard_ordinal, uint64_t* visited_nodes = nullptr);
    static bool sort_and_detect_duplicate_dependency_parents(
        std::vector<uint64_t>* parent_store_ids);
    bool current_store_id(const RunCursor& cursor,
                         uint64_t* store_id) const;
    static bool current_position_active(const RunCursor& cursor);
    static void mark_cursor_position_completed(RunCursor& cursor);
    static void mark_cursor_range_completed(RunCursor& cursor,
                                            uint64_t first_ordinal,
                                            uint64_t count);
    bool endpoint_dependencies_ready(const RuntimeDescriptor& owner,
                                     const RunCursor& cursor) const;
    bool terminal_completed(int rank, uint64_t store_id,
                            PipelineEndpointDependencyKind kind) const;
    void install_terminal_watcher(int rank, TerminalWatcher watcher);
    static size_t terminal_watcher_kind_index(
        PipelineEndpointDependencyKind kind);
    static int terminal_watcher_node_height(
        const std::unique_ptr<TerminalWatcherIntervalNode>& node);
    static void update_terminal_watcher_node(
        TerminalWatcherIntervalNode& node);
    static void rotate_terminal_watcher_left(
        std::unique_ptr<TerminalWatcherIntervalNode>& root);
    static void rotate_terminal_watcher_right(
        std::unique_ptr<TerminalWatcherIntervalNode>& root);
    static void insert_terminal_watcher_interval(
        std::unique_ptr<TerminalWatcherIntervalNode>& root,
        TerminalWatcherInterval interval,
        uint64_t* insert_steps);
    static void query_terminal_watcher_intervals(
        const TerminalWatcherIntervalNode* root, uint64_t store_id,
        std::vector<TerminalWatcherInterval>* intervals,
        uint64_t* node_visits);
    std::vector<TerminalWatcherMatch> collect_terminal_watcher_matches(
        int rank, uint64_t store_id,
        PipelineEndpointDependencyKind kind);
    void notify_terminal_completed(
        int rank, uint64_t store_id,
        PipelineEndpointDependencyKind kind);
    void publish_consume_done_transition(
        const ConsumeMutationSnapshot& snapshot,
        const EndpointPosition* endpoint_position, uint64_t store_id);
    std::optional<EndpointPosition> locate_endpoint(int rank,
                                                     uint64_t store_id) const;
    bool endpoint_position_completed(const EndpointPosition& position) const;
    bool protocol_ready(const RuntimeDescriptor& owner,
                        const RunCursor& cursor) const;
    bool reserve_slot_for_issue(RuntimeDescriptor& owner,
                                const RunCursor& cursor,
                                EndpointPosition* position,
                                bool* newly_reserved);
    void complete_protocol_leg(RuntimeDescriptor& owner,
                               const EndpointPosition& position);
    uint64_t source_released_through(const BufferDomainState& domain,
                                     int rank, uint64_t shard_ordinal,
                                     size_t slot_index) const;
    static bool slot_matches(const SlotState& slot,
                             const RuntimeDescriptor& owner,
                             const RunCursor& cursor,
                             uint64_t global_tile);
    /// Whether the occupied slot's foreign-segment owner can no longer need
    /// its contents: the owning descriptor settled (every physical leg
    /// drained) or retired. In-flight foreign owners return false and keep
    /// blocking a cross-segment claim until their own lag-2 release.
    bool foreign_segment_drained(const SlotState& slot) const;
    void retain_buffer_domain_provenance(
        const std::string& domain_id, const BufferDomainState& domain,
        int target_rank);
    void restore_buffer_domain_provenance(
        const std::string& domain_id, BufferDomainState& domain,
        int target_rank);
    void restore_source_slot_provenance(
        const std::string& domain_id,
        const std::pair<int, uint64_t>& source_key,
        std::array<SlotState, 2>& slots);
    static void claim_slot(SlotState& slot, RuntimeDescriptor& owner,
                           const RunCursor& cursor, uint64_t global_tile);
    static void clear_slot(SlotState& slot);
    void release_target_slot_for_consume(const TerminalWatcher& watcher,
                                         int terminal_rank,
                                         uint64_t terminal_store_id);
    std::optional<EndpointPosition> locate_anchor_endpoint(
        const PipelineDescriptor& descriptor,
        const PipelineAnchor& anchor) const;
    NodeView make_resource_view(const RuntimeDescriptor& owner,
                                const RunCursor& cursor,
                                uint64_t store_id) const;
    std::vector<uint64_t> endpoint_parent_store_ids(
        const RuntimeDescriptor& owner,
        const EndpointPosition& position) const;
    TileTransportAdapter::TransferSpec make_transfer_spec(
        const RuntimeDescriptor& owner,
        const RunCursor& cursor,
        uint64_t store_id,
        const NodeView& resource_view) const;
    bool begin_transfer(RuntimeDescriptor* owner,
                        size_t run_index,
                        uint64_t store_id,
                        const NodeView& resource_view,
                        bool physical_anchor);
    void complete_transfer(RuntimeDescriptor* owner,
                           size_t run_index,
                           EndpointPosition position,
                           uint64_t store_id,
                           bool physical_anchor,
                           NodeView resource_view);
    void update_settle_state(RuntimeDescriptor& owner);
    void try_retire_state(RuntimeDescriptor& owner);
    void flush_deferred_terminal_followups();
    nlohmann::json slot_state_json(const std::string& domain_id) const;
    nlohmann::json credit_state_json(const std::string& domain_id) const;
    nlohmann::json obligation_state_json(
        const RuntimeDescriptor& owner) const;
    nlohmann::json obligation_state_json_for_domain(
        const std::string& domain_id,
        const RuntimeDescriptor* additional_owner = nullptr,
        const RuntimeDescriptor* excluded_owner = nullptr) const;
    nlohmann::json slot_row_json(const std::string& domain_id,
                                 const std::string& role, int rank,
                                 uint64_t shard_ordinal,
                                 size_t slot_index,
                                 const SlotState& slot) const;
    nlohmann::json credit_row_json(const std::string& domain_id, int rank,
                                   uint64_t shard_ordinal) const;
    nlohmann::json resource_state_row_json(
        const std::string& resource_kind,
        const std::string& stable_key) const;
    nlohmann::json obligation_row_json(const RuntimeDescriptor& owner,
                                       size_t run_index) const;
    nlohmann::json cursor_run_json(const RuntimeDescriptor& owner,
                                   size_t run_index) const;
    void record_resource_row_patch(const std::string& resource_kind,
                                   nlohmann::json before,
                                   nlohmann::json after);
    void capture_credit_row_before(const std::string& stable_key);
    void capture_source_credit_row_before(const std::string& domain_id,
                                         int rank,
                                         uint64_t shard_ordinal);
    void flush_pending_slot_row_patches();
    void append_target_slot_operation(const std::string& stable_key,
                                      nlohmann::json operation);
    void record_target_slot_claim(const std::string& stable_key,
                                  const SlotState& slot);
    void record_target_slot_consume(const std::string& stable_key,
                                    uint64_t slot_generation,
                                    int terminal_rank,
                                    uint64_t terminal_store_id,
                                    size_t consume_run_index,
                                    const uint64_t* added_shards,
                                    size_t added_shard_count,
                                    bool cleared);
    void record_target_slot_request_close_clear(
        const std::string& stable_key, uint64_t slot_generation);
    void record_target_slot_erase(const std::string& stable_key,
                                  uint64_t slot_generation);
    void flush_pending_credit_row_patches();
    void flush_pending_obligation_row_patches();
    void record_obligation_row_patch(nlohmann::json before,
                                     nlohmann::json after);
    void count_row_helper_visit() const;
    void mark_cursor_dirty(RuntimeDescriptor& owner,
                           std::optional<size_t> run_index = std::nullopt);
    void index_batch_descriptors(
        const std::vector<RuntimeDescriptor*>& owners);
    std::vector<const EndpointProgressionIndexEntry*>
    overlapping_endpoint_progressions(int rank, uint64_t first_store_id,
                                      uint64_t last_store_id) const;
    void set_current_cursor(RuntimeDescriptor& owner, size_t run_index);
    void erase_current_cursor(RuntimeDescriptor& owner, size_t run_index);
    // F (2026-10-11) current-anchor-set maintenance helpers. Both are pure
    // index bookkeeping (no callbacks) and are safe from inside any cursor
    // lifecycle transition.
    /// Register-time static binding validation: the fail-closed chain of
    /// resource_view_for_anchor minus its dynamic (coordinate/preflight)
    /// checks, executed once per anchor so that future anchors -- which the
    /// issue pass no longer enumerates -- still cannot carry an illegal
    /// descriptor binding silently.
    void validate_anchor_binding_static(int rank, uint64_t store_id,
                                        const AnchorBinding& binding) const;
    void note_current_cursor_anchor(int rank, uint64_t store_id);
    void drop_current_cursor_anchor(int rank, uint64_t store_id);
    uint64_t resource_mutation_before(const std::string& resource_kind,
                                      const std::string& stable_key,
                                      int target_rank_hint = -1);
    void resource_mutation_after(const std::string& resource_kind,
                                 const std::string& stable_key,
                                 uint64_t previous_version,
                                 bool allow_target_noop = false);
    void notify_resource_mutation(const ResourceMutation& mutation);
    std::string slot_resource_key(const std::string& domain_id,
                                  const std::string& role,
                                  int rank, uint64_t shard_ordinal,
                                  size_t slot_index) const;
    std::string credit_resource_key(const std::string& domain_id,
                                    int rank,
                                    uint64_t shard_ordinal) const;
    void publish_pipeline_transition(
        RuntimeDescriptor& owner, const std::string& transition_kind,
        const EndpointPosition* position, uint64_t store_id,
        const std::string& resource_leg, const nlohmann::json& amount,
        uint64_t cursor_version_start);
    static void advance(RunCursor& cursor);
    [[noreturn]] static void fail_closed(const std::string& message);

    std::vector<std::shared_ptr<NodeStoreGraphSource>> graph_sources_;
    std::vector<Workload*> workloads_;
    TileTransportAdapter* transport_ = nullptr;
    ScheduleIssuePass schedule_issue_pass_;
    std::vector<std::unique_ptr<RuntimeDescriptor>> descriptors_;
    std::vector<RuntimeDescriptor*> active_descriptors_;
    std::vector<RuntimeDescriptor*> pending_watch_retirements_;
    bool defer_terminal_followups_ = false;
    std::set<RuntimeDescriptor*> deferred_terminal_settle_owners_;
    std::set<int> deferred_terminal_wake_ranks_;
    std::vector<ConsumeMutationSnapshot> deferred_consume_mutations_;
    std::vector<std::unordered_map<uint64_t, AnchorBinding>>
        physical_anchors_by_rank_;
    // F (2026-10-11): rank -> ordered store ids of registered physical
    // anchors whose run cursor currently points at them (the run's
    // current_store_id equals the anchor id). Kept in lockstep with
    // current_cursors_by_store_id_ and physical_anchors_by_rank_; ids only.
    std::unordered_map<uint64_t, std::set<uint64_t>>
        current_anchor_ids_by_rank_;
    std::map<int, EndpointProgressionIndex>
        endpoint_progressions_by_rank_;
    std::map<std::pair<int, uint64_t>, CurrentCursorBinding>
        current_cursors_by_store_id_;
    std::unordered_map<std::string, uint64_t>
        active_owner_count_by_domain_;
    uint64_t next_endpoint_index_order_ = 0;
    std::vector<std::vector<TerminalWatcher>> terminal_watchers_by_rank_;
    std::vector<std::array<TerminalWatcherProgressionIndex, 2>>
        terminal_watcher_indexes_by_rank_;
    std::unordered_map<std::string, BufferDomainState> buffer_domains_;
    std::unordered_map<std::string, int> buffer_domain_target_ranks_;
    std::unordered_map<std::string, BufferDomainProvenance>
        buffer_domain_provenance_;
    std::unordered_map<std::string, RuntimeDescriptor*> descriptor_by_identity_;
    LedgerNodeTransitionPublisher node_transition_publisher_;
    PipelineTransitionPublisher pipeline_transition_publisher_;
    DescriptorCommitPublisher descriptor_commit_publisher_;
    WatchReferenceCounter watch_reference_counter_;
    ResourceMutationObserverToken next_resource_observer_token_ = 1;
    std::map<ResourceMutationObserverToken, ResourceObserverEntry>
        resource_mutation_observers_;
    std::map<std::pair<std::string, std::string>, uint64_t>
        resource_versions_;
    std::map<std::pair<std::string, std::string>, nlohmann::json>
        resource_mutation_before_rows_;
    std::map<TargetSlotStableKey, PendingTargetSlotDelta>
        pending_target_slot_deltas_;
    std::map<std::tuple<std::string, int, uint64_t>, PendingCreditRowCapture>
        pending_credit_row_before_;
    std::vector<nlohmann::json> pending_slot_patches_;
    std::vector<nlohmann::json> pending_credit_patches_;
    std::vector<nlohmann::json> pending_obligation_patches_;
    std::map<RuntimeDescriptor*, std::set<size_t>> dirty_cursor_runs_;
    std::set<RuntimeDescriptor*> dirty_cursor_segments_;
    std::set<RuntimeDescriptor*> retired_cursor_segments_;
    mutable ProducerMetrics producer_metrics_;
    mutable bool diagnostic_row_capture_active_ = false;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_REMOTE_READ_EXECUTOR_HH
