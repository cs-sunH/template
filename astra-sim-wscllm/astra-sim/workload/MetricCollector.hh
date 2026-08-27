/******************************************************************************
This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
*******************************************************************************/

#ifndef ASTRASIM_WORKLOAD_METRIC_COLLECTOR_HH
#define ASTRASIM_WORKLOAD_METRIC_COLLECTOR_HH

#include <cstdint>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include <json/json.hpp>

#include "astra-sim/common/Common.hh"

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

    // Load the manifest and set the detail level ("off", "summary", "full").
    // Must be called once, early during startup. With detail "off" the
    // collector stays disabled and performs no work at all.
    void initialize(const std::string& manifest_path,
                    const std::string& detail_level);

    bool enabled() const {
        return this->enabled_;
    }

    // Sparse boundary events. Only nodes listed in the manifest ever match;
    // all other nodes cost two failed hash lookups.
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
    void online_register_node_anchor(int rank, uint64_t node_id,
                                     const std::string& request_id,
                                     const std::string& kind,
                                     bool transfer_anchor);

    // Side-band accumulation of executed GPU compute node FLOPs and local
    // tensor bytes (values already read by Workload::issue_comp).
    void on_compute_issue(int rank,
                          uint64_t num_ops,
                          uint64_t local_tensor_bytes);

    // Reserved for repos with a local HBM restore model (e.g. SH2). In this
    // repo no caller exists; the accumulator is kept for interface parity.
    void on_local_hbm_restore_issue(int rank, uint64_t bytes);

    // Compute and print all metric records. Called after the event loop,
    // before systems are deleted. No-op when disabled.
    void finalize(const std::vector<Sys*>& systems, Tick sim_end_tick);

    // Phase-0 performance counter framework (plan step 0-7; 总改造计划 §6
    // 阶段 0): named per-category counters for the stage-6/7 accounting.
    // Default OFF and never serialized into any [METRIC] record.  Offline
    // static-ET runs never exercise these mechanisms, so every counter stays
    // 0 and [METRIC] output is byte-identical to a build without this
    // framework; stage 6/7 may opt in per category without touching the
    // static baseline bytes.
    struct PerformanceCounters {
      uint64_t callback_count = 0;        // Python decision callbacks
      uint64_t poll_count = 0;            // completion polling hits
      uint64_t bridge_bytes = 0;          // Python<->C++ bridge payload bytes
      uint64_t gil_contention_ns = 0;     // GIL wait time in the bridge
      uint64_t log_bytes = 0;             // decision/audit log bytes written
      uint64_t node_completion_count = 0; // node completion events observed
      uint64_t reader_window_events = 0;  // ingress reader window events
      uint64_t backpressure_events = 0;   // backpressure/watermark hits
      uint64_t batch_count = 0;           // decision batches committed
    };

    void enable_counters() {
      this->counters_enabled_ = true;
    }

    bool counters_enabled() const {
      return this->counters_enabled_;
    }

    const PerformanceCounters& counters() const {
        return this->counters_;
    }

    // ---------------------------------------------------------------------
    // SLO pipeline B2 observers (WP6/WP8/WP9; /tmp/slo_wps/plans/
    // CPP_SPEC.md). WP6: observers outside the workload layer (the
    // congestion-aware FluidScheduler link observer) emit their [METRIC]
    // records through the same single-write channel. Read-only; callers
    // must check enabled() first (records are never emitted in off mode).
    void emit_observer_record(const std::string& json_line) const;

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
    [[nodiscard]] uint64_t slo_watermark_period_ns() const {
        return this->watermark_period_ns_;
    }

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
        // Same integral from the pre-existing direct delta loop, kept for
        // the <=1% timeavg cross-check (A-class criterion).
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
    // Phase-0 counters (plan step 0-7): opt-in, default off, never emitted.
    bool counters_enabled_ = false;
    PerformanceCounters counters_;

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
    std::vector<MemoryAnchorTick> memory_anchor_ticks_;
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
};

}  // namespace AstraSim

#endif /* ASTRASIM_WORKLOAD_METRIC_COLLECTOR_HH */
