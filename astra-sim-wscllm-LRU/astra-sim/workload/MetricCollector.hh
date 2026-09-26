/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef ASTRASIM_WORKLOAD_METRIC_COLLECTOR_HH
#define ASTRASIM_WORKLOAD_METRIC_COLLECTOR_HH

#include <cstdint>
#include <cstdio>
#include <functional>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/system/Common.hh"

namespace AstraSim {

class Sys;

// MetricCollector is a side-band, read-only observer of the simulation. It
// loads a metrics manifest (see doc "四仓库实验指标采集详细修改方案" sec.4),
// maps sparse (rank, node_id) events to request/iteration boundaries, and
// computes/prints metrics once after the event loop finishes.
//
// Hard constraints (doc sec.5.1 and sec.13):
// - It must never call Sys::register_event().
// - It must never modify nodes, the dependency resolver, or hardware
//   resources.
// - It must not log per event inside the event loop.
// - When metrics are disabled, Workload only pays a single enabled() branch.
class MetricCollector {
  public:
    static MetricCollector& instance();

    // The collector owns an anonymous online memory-anchor spool when that
    // mode is active.  The explicit destructor keeps process teardown and
    // test/reinitialization lifecycles from leaking its file descriptor.
    ~MetricCollector();

    // Load the manifest and set the detail level ("off", "summary", "full").
    // Must be called once, early during startup. With detail "off" the
    // collector stays disabled and performs no work at all.
    void initialize(const std::string& manifest_path,
                    const std::string& detail_level);

    bool enabled() const {
        return this->enabled_;
    }

    // Online service statistics can retire terminal node records.  The one
    // exception is an enabled microbenchmark: its arbitrary iteration window
    // queries still require the complete per-node history.
    [[nodiscard]] bool preserve_online_operator_history() const {
        return this->enabled_ && this->run_mode_ == "microbenchmark";
    }

    // Sparse boundary events. Only nodes listed in the manifest (static) or
    // dynamically registered (online) ever match; all other nodes cost two
    // failed hash lookups. R2 one-shot semantics (2026-08-29): a matched
    // node bucket is applied in full (a bucket can carry several events,
    // e.g. a code-4 + code-7 watch tail) and then ERASED immediately; an
    // empty rank bucket is erased too. Each (rank, node) edge fires at most
    // once (take_node consumes the free set; store ids are never reused;
    // registration precedes the issue pass), so the second call for the
    // same edge is a silent miss. The erase touches ONLY these routing
    // tables: every finalize consumer (memory_anchor_ticks_ / spool,
    // transfer maxima, requests_, request_boundaries_, first_token_ticks_,
    // iterations_) reads derived storage filled by apply_event, never the
    // routing tables.
    void on_node_issue(int rank, uint64_t node_id, Tick tick);
    void on_node_complete(int rank, uint64_t node_id, Tick tick);

    // ---------------------------------------------------------------------
    // Phase-7 §10.3: online dynamic anchor registration (plan step 1-8
    // contract ⑤/⑥, implemented at phase 7).
    //
    // The static manifest's sparse node events only match the offline
    // per-rank node-id sequence. The online dynamic graph emits per-rank
    // node ids in the online emission order, which interleaves requests
    // across stages, so most boundary nodes do not match the static table
    // (measured: 708/1177 requests got a completion anchor). The online
    // driver therefore registers the anchors dynamically, BEFORE the nodes
    // are issued, from the same data it commits:
    //   - online_register_request: one call per request-queue CSV data row
    //     at load time (turn-0 rows carry an ABSOLUTE arrival; turn>0 rows
    //     arrive AFTER_REQUEST, parent = previous same-session row);
    //   - online_register_ranks: the watch member sets (prefill/decode);
    //   - online_register_node_anchor: for every (rank, request, stage)
    //     group the FIRST node is the start anchor (issue event) and the
    //     LAST is the end anchor (complete event); transfer nodes of the
    //     KV routes (node names contain "kv") register the memory anchor
    //     (code 7) on the last transfer node of each (request, rank) --
    //     the offline doc-sec.4.3 semantics.
    // Registration is idempotent: re-registering an existing (request,
    // rank, node, event) is a no-op, and re-registering a request updates
    // only missing fields (the online CSV data is authoritative for
    // arrival/session fields; rank sets merge). No registered anchor ever
    // makes the static path observable: all three entry points no-op when
    // metrics are disabled, and the static main never calls them.
    // Phase-7 §10.3: the online driver runs the collector against the live
    // dynamic graph. The manifest's static node-event table is keyed by the
    // OFFLINE ET's global node-id space, which overlaps the online per-rank
    // id space (measured: online ids 0..7733 vs static ids 1..8000+), so
    // static events would fire on unrelated online nodes and corrupt every
    // request state. The online main therefore clears the static tables
    // right after initialize() and lets the dynamic anchors fully own the
    // event tables. The offline static main never calls this; the static
    // [METRIC] bytes are untouched.
    void clear_static_node_events();

    void online_register_request(int64_t queue_index,
                                 const std::string& request_id,
                                 const std::string& session_id,
                                 int64_t turn_index,
                                 bool absolute_arrival,
                                 Tick arrival_value_ns,
                                 int64_t arrival_parent_queue_index,
                                 Tick arrival_interval_ns);
    void online_register_ranks(const std::string& request_id, bool prefill,
                               const std::vector<int>& ranks);
    // Returns true when the (rank, node) HAS an anchor on some edge after
    // the call -- including an idempotent duplicate hit -- so the online
    // driver can set the OnlineNode fast-path flags; false for disabled /
    // unknown-request / unknown-kind (no routing entry exists, matching
    // the Workload-side enabled() gate).
    [[nodiscard]] bool online_register_node_anchor(int rank, uint64_t node_id,
                                                   const std::string& request_id,
                                                   const std::string& kind,
                                                   bool transfer_anchor);

    // Side-band accumulation of executed GPU compute node FLOPs and local
    // tensor bytes (values already read by Workload::issue_comp).
    void on_compute_issue(int rank,
                          uint64_t num_ops,
                          uint64_t local_tensor_bytes);

    // Side-band local-HBM restore byte accounting. Workload's local-HBM
    // KV-restore issue path calls this when metrics are enabled (the -LRU
    // repos wire it; face keeps no such caller); the accumulator feeds the
    // rank_compute/local_hbm records' restore-bytes fields at finalize.
    void on_local_hbm_restore_issue(int rank, uint64_t bytes);

    // Compute and print all metric records. Called after the event loop,
    // before systems are deleted. No-op when disabled.
    void finalize(const std::vector<Sys*>& systems, Tick sim_end_tick);

    // ---------------------------------------------------------------------
    // SLO pipeline B2 observers (WP6/WP8/WP9; /tmp/slo_wps/plans/
    // CPP_SPEC.md). WP6: observers outside the workload layer (the
    // congestion-aware FluidScheduler link observer) emit their [METRIC]
    // records through the same single-write channel. Read-only; callers
    // must check enabled() first (records are never emitted in off mode).
    void emit_observer_record(const std::string& json_line) const;

    // A2 (2026-08-28): release the completed request's dynamic anchor
    // routing entries (issue_events_ / complete_events_ per (rank, node))
    // after its REQUEST_COMPLETE commit. Safe because every anchored node
    // of the request has fired both hooks before the decode watch could
    // fire, and the REQUEST_COMPLETE delta fact arrives at a strictly later
    // epoch; store ids are never reused, so the entries cannot be
    // re-referenced. The requests_ record itself stays (finalize emits one
    // [METRIC] request row per manifest request from it). No-op when
    // metrics are disabled or the request has no anchors.
    void online_release_request_anchors(const std::string& request_id);

    // C5 (2026-08-28): flush the buffered [METRIC] emit channel. Records
    // accumulate in an in-memory buffer (flushed at 1 MiB to bound memory)
    // and land in one ::write per flush instead of one syscall per record;
    // content and order are unchanged. finalize() flushes at its end; the
    // link-observer emission path (which runs after finalize) must call
    // this once when done.
    void flush_emit_buffer() const;

    [[nodiscard]] const std::string& metric_repo_variant() const {
        return this->repo_variant_;
    }

    [[nodiscard]] const std::string& metric_run_id() const {
        return this->run_id_;
    }

    // WP8/WP6 sampling anchors loaded from the manifest ``slo_sampling``
    // node (CPP_SPEC §A). A missing node or null values fall back to the
    // documented provisional anchor (5,000,000 ns) with provisional=true;
    // batch B4 replaces them with derived values. Every related record
    // echoes the period actually used.
    [[nodiscard]] uint64_t slo_link_bucket_ns() const {
        return this->link_bucket_ns_;
    }

    [[nodiscard]] bool slo_sampling_provisional() const {
        return this->slo_sampling_provisional_;
    }

  private:
    MetricCollector() = default;

    // Manifest event codes (doc sec.4.3). Code 8 (FIRST_TOKEN_COMPLETE,
    // WP9 /tmp/slo_wps/plans/WP9_CONTRACT.md §1) is the first-token marker
    // complete edge: subject is the request, multiple ranks take the min.
    enum class EventCode : uint8_t {
        PREFILL_START_ISSUE = 1,
        PREFILL_END_COMPLETE = 2,
        DECODE_START_ISSUE = 3,
        COMPLETION_COMPLETE = 4,
        ITERATION_START_ISSUE = 5,
        ITERATION_END_COMPLETE = 6,
        MEMORY_ANCHOR_COMPLETE = 7,
        FIRST_TOKEN_COMPLETE = 8,
    };

    struct NodeMetricEvent {
        uint8_t event_code;
        int64_t subject_id;
    };

    struct ArrivalSpec {
        enum class Kind { ABSOLUTE, AFTER_REQUEST } kind = Kind::ABSOLUTE;
        Tick value_ns = 0;
        int64_t parent_queue_index = -1;
        Tick interval_ns = 0;
    };

    // Doc sec.5.3. Min-start / max-end ticks are tracked incrementally;
    // a boundary is authoritative only once every expected rank reported.
    struct RequestMetricState {
        int64_t queue_index = -1;
        std::string request_id;
        std::string session_id;
        int64_t turn_index = 0;
        ArrivalSpec arrival;
        int64_t prefill_instance = -1;
        int64_t decode_instance = -1;
        std::set<int> prefill_ranks;
        std::set<int> decode_ranks;
        std::optional<Tick> prefill_start;
        std::optional<Tick> prefill_end;
        std::optional<Tick> decode_start;
        std::optional<Tick> completion;
        std::set<int> prefill_start_ranks_seen;
        std::set<int> prefill_end_ranks_seen;
        std::set<int> decode_start_ranks_seen;
        std::set<int> completion_ranks_seen;
        // Resolved at finalize time (doc sec.3.1: after_request arrivals use
        // the parent's actual completion, not a planner prediction).
        std::optional<Tick> resolved_arrival;
        // WP9 (CPP_SPEC §C + WP9_CONTRACT §6 2026-08-27 ruling):
        // manifest-request decode length, used only by the first-token
        // checks (a completed decode_length==1 request reports the
        // informational first_token_completion_skew_ns =
        // completion_ns - first_token_ns; the former equality invariant was
        // relaxed -- code 8 aggregates the min tick across TP ranks while
        // completion aggregates the max). Optional: synthetic online
        // manifests may omit it, in which case the skew field is not
        // evaluable and the request record notes
        // manifest_decode_length_missing.
        std::optional<int64_t> decode_length;
    };

    struct IterationMetricState {
        int64_t benchmark_point_id = -1;
        std::optional<Tick> iteration_start;
        std::optional<Tick> iteration_end;
        std::set<int> start_ranks_seen;
        std::set<int> end_ranks_seen;
    };

    struct MemoryAnchorTick {
        int64_t subject_id;
        int rank;
        uint64_t node_id;
        Tick tick;
    };

    // Online transfer-anchor replay only needs the last observed tick for
    // planner ledger actions.  Keep the key deliberately separate from the
    // raw spool record: the latter must preserve every event and its original
    // order for full-detail memory_anchor output.
    struct TransferAnchorKey {
        int64_t subject_id;
        int rank;

        bool operator==(const TransferAnchorKey& other) const noexcept {
            return this->subject_id == other.subject_id &&
                this->rank == other.rank;
        }
    };

    struct TransferAnchorKeyHash {
        size_t operator()(const TransferAnchorKey& key) const noexcept {
            const size_t subject_hash = std::hash<int64_t>{}(key.subject_id);
            const size_t rank_hash = std::hash<int>{}(key.rank);
            return subject_hash ^
                (rank_hash + 0x9e3779b9u + (subject_hash << 6) +
                 (subject_hash >> 2));
        }
    };

    // 128-bit accumulators: 64-bit integers risk overflowing total FLOPs of
    // long workloads (doc sec.5.3). Serialized as decimal strings.
    struct RankMetricState {
        unsigned __int128 total_num_ops = 0;
        unsigned __int128 total_local_bytes = 0;
        uint64_t local_hbm_restore_bytes = 0;
    };

    // One planner memory ledger delta (manifest ``memory_actions``, doc
    // sec.7.1/7.8). Anchors are resolved to actual ASTRA ticks at finalize.
    struct MemoryAction {
        int64_t sequence_index = 0;
        std::string anchor_kind;
        std::string anchor_quality;
        int64_t trigger_queue_index = -1;  // -1 = no triggering request
        int rank = -1;
        int64_t weight_delta_bytes = 0;
        int64_t resident_kv_delta_bytes = 0;
        int64_t reserved_kv_delta_bytes = 0;
        std::string cause;
    };

    // Observed request stage boundaries, stored during the finalize request
    // pass so memory action anchors can resolve them afterwards.
    struct RequestBoundaryInfo {
        std::optional<Tick> prefill_start;
        std::optional<Tick> decode_start;
        std::optional<Tick> completion;
        bool prefill_done = false;
        bool decode_done = false;
        bool completion_done = false;
    };

    // Aggregate result of the doc sec.7.8 capacity-time replay.
    struct MemoryReplayTotals {
        uint64_t actions_total = 0;
        uint64_t actions_replayed = 0;
        uint64_t actions_unresolved = 0;
        uint64_t peaks_count = 0;
        uint64_t final_state_mismatches = 0;
        uint64_t transfer_anchor_request_level_fallback = 0;
        std::map<std::string, uint64_t> unresolved_by_reason;
    };

    // WP8 (CPP_SPEC §B): per-rank bucketed watermark sample series, filled
    // during the doc sec.7.8 delta replay. The step function is sampled at
    // every ledger delta (event-driven) and at every watermark bucket
    // boundary (periodic bottom line); bucket peaks are sparse (buckets
    // that stayed zero are omitted at emit time). Areas use independent
    // 128-bit accumulators so the timeavg cross-check against the direct
    // replay integral is a real comparison, not a tautology.
    struct WatermarkSeries {
        // bucket index -> (resident peak bytes, committed peak bytes).
        std::map<uint64_t, std::pair<uint64_t, uint64_t>> bucket_peaks;
        // Buckets with actual occupancy activity (a delta applied inside
        // the bucket or a nonzero segment covered by it). Coordinator
        // ruling 2026-08-26: bucket records are emitted only for these,
        // plus the first and last bucket of the window (always).
        std::set<uint64_t> changed_buckets;
        uint64_t resident_peak = 0;
        uint64_t committed_peak = 0;
        unsigned __int128 resident_area = 0;
        unsigned __int128 committed_area = 0;
        // Same integrals from the pre-existing direct delta loop, kept for
        // the <=1% timeavg cross-check (A-class criterion). Both sides are
        // kept: the direct loop and the watermark walk clamp the same step
        // function at zero, so each pair must agree exactly.
        unsigned __int128 direct_resident_area = 0;
        unsigned __int128 direct_committed_area = 0;
        uint64_t capacity_violations = 0;
        uint64_t sample_count = 0;
        bool capacity_known = false;
        int64_t capacity_bytes = 0;
        bool replayed = false;  // rank appeared in the delta replay loop
    };

    // Top-level ``microbenchmark`` manifest extension (doc sec.8): one point
    // per ET/run, so a run's per-rank num_ops/byte totals over the point's
    // active ranks are exactly that iteration's numerators.
    struct MicrobenchPointSpec {
        int64_t benchmark_point_id = -1;
        int64_t tp_degree = 0;
        std::vector<int> active_ranks;
        nlohmann::json raw;  // full point descriptor, passed through
    };

    void load_manifest(const std::string& manifest_path);
    void reset_for_initialize();
    void ensure_online_memory_anchor_spool();
    void close_memory_anchor_spool_or_die();
    void append_online_memory_anchor(const MemoryAnchorTick& anchor);
    void emit_online_memory_anchor_spool_records();
    void emit_memory_anchor_record(const MemoryAnchorTick& anchor) const;
    void rebuild_online_transfer_anchor_interest();
    void apply_event(const NodeMetricEvent& event, int rank, uint64_t node_id,
                     Tick tick);
    std::optional<Tick> resolve_arrival(size_t request_index);
    void emit_record(const std::string& json_line) const;
    void check_consistency(bool condition, const std::string& message);
    // Doc sec.7.8: anchor every memory action to an actual ASTRA tick, replay
    // the per-rank ledger over [0, sim_end_tick], and emit the
    // capacity-time integral records. Also passes the planner peaks through
    // (full detail only). Runs entirely at finalize time. fallback_ranks
    // (the simulated Sys ids) extends the WP8 watermark universe to ranks
    // with no planner ledger rows (online synthetic manifests carry none).
    MemoryReplayTotals emit_memory_records(Tick sim_end_tick,
                                           bool full_detail,
                                           const std::vector<int>& fallback_ranks);
    // WP8 (CPP_SPEC §B): aggregate the per-rank watermark series to
    // instance-level bucket peaks and emit the hbm_watermark /
    // hbm_watermark_summary records (bucket detail full-only; the summary
    // record emits in every non-off mode).
    void emit_watermark_records(const std::map<int, WatermarkSeries>& series_by_rank,
                                Tick sim_end_tick,
                                bool full_detail);

    bool enabled_ = false;
    std::string detail_level_ = "off";
    // D3 (2026-08-28): set by clear_static_node_events (online mode only).
    // Online synthetic manifests carry no planner ledger, so the
    // hbm_watermark bucket records are the documented all-zero series (the
    // authoritative WP8 data source is the ledger.jsonl replay, see
    // slo_tools/hbm_watermark.py) and metrics_postprocess does not consume
    // them -- emit_watermark_records skips the bucket lines (the per-rank
    // summary records stay).
    bool online_mode_ = false;
    // A2 (2026-08-28): request_id -> anchored (rank, node) pairs, filled by
    // online_register_node_anchor and drained by
    // online_release_request_anchors at REQUEST_COMPLETE.
    std::unordered_map<std::string,
                       std::vector<std::pair<int, uint64_t>>>
        online_anchor_nodes_;
    // C5 (2026-08-28): buffered [METRIC] emit channel (mutable: emit_record
    // is const; flushed at 1 MiB and by flush_emit_buffer).
    mutable std::string emit_buffer_;

    // Manifest metadata.
    int schema_version_ = 0;
    std::string repo_variant_ = "unknown";
    std::string run_mode_ = "service";
    std::string run_id_;
    std::optional<int64_t> manifest_npus_count_;
    std::optional<std::string> trace_digest_;
    std::optional<std::string> request_mapping_digest_;
    std::optional<std::string> kv_event_digest_;

    // Manifest ``slo_sampling`` node (CPP_SPEC §A): watermark/link-observer
    // sampling anchors. Defaults are the documented provisional anchor
    // (coordinator ruling 2026-08-26: 5,000,000 ns -- 1 ms measured 31,895
    // bucket records on the S3 2s window and would overflow the 60s-window
    // log budget); load_manifest overwrites them when the node carries real
    // values and clears the provisional flag accordingly.
    uint64_t watermark_period_ns_ = 5000000;
    uint64_t link_bucket_ns_ = 5000000;
    bool slo_sampling_provisional_ = true;
    bool slo_sampling_from_manifest_ = false;

    std::vector<RequestMetricState> requests_;
    std::unordered_map<int64_t, size_t> request_index_by_queue_index_;
    // Phase-7 §10.3: request_id -> requests_ index, populated by the online
    // dynamic registration path (the online driver registers node/rank
    // anchors by request_id, which is what the committed nodes carry).
    std::unordered_map<std::string, size_t> request_index_by_request_id_;
    std::map<int64_t, IterationMetricState> iterations_;
    // Static ET runs keep the legacy in-memory history.  Online service runs
    // stream raw records to memory_anchor_spool_ and retain only the bounded
    // planner-action index below, so terminal-node count cannot grow RAM.
    std::vector<MemoryAnchorTick> memory_anchor_ticks_;
    std::FILE* memory_anchor_spool_ = nullptr;
    uint64_t memory_anchor_spool_record_count_ = 0;
    std::unordered_set<TransferAnchorKey, TransferAnchorKeyHash>
        online_transfer_anchor_keys_needed_;
    std::unordered_set<int64_t> online_transfer_anchor_subjects_needed_;
    std::unordered_map<TransferAnchorKey, Tick, TransferAnchorKeyHash>
        online_transfer_anchor_ticks_;
    std::unordered_map<int64_t, Tick> online_transfer_anchor_by_subject_;
    // WP9 (WP9_CONTRACT §1): subject (queue_index) -> min observed code-8
    // first-token complete tick across ranks and re-registrations.
    std::unordered_map<int64_t, Tick> first_token_ticks_;
    std::vector<MemoryAction> memory_actions_;
    // Raw manifest ``planner_memory_peaks`` payload, passed through with
    // scope/projection annotations at finalize (doc sec.7.7/7.8).
    std::vector<nlohmann::json> planner_memory_peaks_;
    // queue_index -> observed stage boundaries (filled during finalize).
    std::unordered_map<int64_t, RequestBoundaryInfo> request_boundaries_;
    // Present only for run_mode=microbenchmark manifests (doc sec.8).
    std::optional<MicrobenchPointSpec> microbench_point_;

    // Doc sec.5.4: sparse per-rank lookup from node id to boundary events.
    std::unordered_map<int, std::unordered_map<uint64_t, std::vector<NodeMetricEvent>>>
        issue_events_;
    std::unordered_map<int, std::unordered_map<uint64_t, std::vector<NodeMetricEvent>>>
        complete_events_;

    std::unordered_map<int, RankMetricState> rank_states_;

    uint64_t dropped_events_ = 0;
    std::vector<std::string> consistency_violations_;
    bool finalized_ = false;

    // Test-only friend keeps storage-bound assertions out of the production
    // API while allowing the regression fixture to prove that online raw
    // anchor history is spooled rather than retained in a vector.
    friend struct MetricCollectorTestAccess;
};

}  // namespace AstraSim

#endif /* ASTRASIM_WORKLOAD_METRIC_COLLECTOR_HH */
