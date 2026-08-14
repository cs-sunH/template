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

  private:
    MetricCollector() = default;

    // Manifest event codes (doc sec.4.3).
    enum class EventCode : uint8_t {
        PREFILL_START_ISSUE = 1,
        PREFILL_END_COMPLETE = 2,
        DECODE_START_ISSUE = 3,
        COMPLETION_COMPLETE = 4,
        ITERATION_START_ISSUE = 5,
        ITERATION_END_COMPLETE = 6,
        MEMORY_ANCHOR_COMPLETE = 7,
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
    // (full detail only). Runs entirely at finalize time.
    MemoryReplayTotals emit_memory_records(Tick sim_end_tick,
                                           bool full_detail);

    bool enabled_ = false;
    std::string detail_level_ = "off";

    // Manifest metadata.
    int schema_version_ = 0;
    std::string repo_variant_ = "unknown";
    std::string run_mode_ = "service";
    std::string run_id_;
    std::optional<int64_t> manifest_npus_count_;
    std::optional<std::string> trace_digest_;
    std::optional<std::string> request_mapping_digest_;
    std::optional<std::string> kv_event_digest_;
    // Doc sec.9.2 (WSC-LLM): the real KV policy is read from the manifest
    // (generator config), never parsed from the scenario banner.
    std::optional<std::string> kv_policy_;

    std::vector<RequestMetricState> requests_;
    std::unordered_map<int64_t, size_t> request_index_by_queue_index_;
    std::map<int64_t, IterationMetricState> iterations_;
    std::vector<MemoryAnchorTick> memory_anchor_ticks_;
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
