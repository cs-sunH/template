#ifndef ASTRASIM_WORKLOAD_STATISTICS_HH
#define ASTRASIM_WORKLOAD_STATISTICS_HH

#include "astra-sim/system/Common.hh"
#include "astra-sim/common/Logging.hh"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <unordered_map>
#include <unordered_set>

// et_feeder.h stays: the ChakraProtoMsg::NodeType typedef below is part of
// this header's surface (Workload/MetricCollector include it for the type).
#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"
#include "astra-sim/workload/execution_driven/GraphSource.hh"

typedef ChakraProtoMsg::NodeType ChakraNodeType;

typedef uint64_t NodeId;

namespace AstraSim {
class Workload;
class Statistics {
  public:
    class OperatorStatistics {
      public:
        static const Tick INVALID_TICK = UINT64_MAX;
        enum class OperatorType { CPU, GPU, COMM, REMOTE_MEM, INVALID };
        // Dispatching on the NodeView fields (kind / is_cpu_op):
        // MemLoad/MemStore -> REMOTE_MEM, Compute -> CPU/GPU by is_cpu_op,
        // CommSend/CommRecv/CommCollective -> COMM, else INVALID.
        static OperatorType get_operator_type(
            const ExecutionDriven::NodeView& node);
        OperatorStatistics(Tick start_time, OperatorType type)
            : start_time(start_time),
              end_time(INVALID_TICK),
              type(type) {}
        OperatorStatistics()
            : start_time(INVALID_TICK),
              end_time(INVALID_TICK),
              type(OperatorType::INVALID) {}

        Tick start_time;
        Tick end_time;
        OperatorType type;

        // compute node
        std::optional<double> memory_utilization;
        std::optional<double> compute_utilization;
        // (2026-10-01 清除：operation_intensity/is_memory_bound 写后不读
        // ——compact online 聚合吸收 roofline 求和后二者再无读者，随其
        // 写点一并移除。)
    };

  public:
    Statistics(Workload* workload);

    OperatorStatistics& get_operator_statistics(NodeId node_id);

    // (The const get_operator_statistics overload was removed -- its only
    // historical callers all bound a non-const Statistics lvalue.)

    // NodeView-keyed recorders (step 1-8), keyed by NodeView::global_id.
    void record_start(const ExecutionDriven::NodeView& node, Tick start_time);

    void record_end(const ExecutionDriven::NodeView& node, Tick end_time);

    // Compact online-service path.  Workload keeps this state on the live
    // NodeStore record and calls these O(1) hooks directly; unlike the legacy
    // NodeView overloads, they never insert/find/erase operator_statistics.
    void record_online_service_start(
        const ExecutionDriven::NodeView& node,
        ExecutionDriven::OnlineStatisticsState& state,
        Tick start_time);
    void complete_online_service_operator(
        const ExecutionDriven::NodeView& node,
        ExecutionDriven::OnlineStatisticsState& state,
        Tick end_time);

    // Read-only accessors for side-band metric computation (doc sec.5.7).
    // Valid after post_processing() has run. They only scan existing
    // operator statistics and never influence simulation execution.

    // Wall time of this rank: max operator end_time.
    Tick get_wall_time() const;

    // Merged union of the intervals of the given operator type, clipped to
    // [window_start, window_end].
    Tick calculate_type_time_in_window(
        OperatorStatistics::OperatorType type,
        Tick window_start,
        Tick window_end) const;

    // Duration-weighted sums of the roofline per-node utilization fields over
    // CPU/GPU operator statistics, clipped to [window_start, window_end].
    // Same weighting formula as the removed legacy extract_utilizations()
    // ("Average compute utilization": utilization * duration accumulated
    // over CPU/GPU nodes, with a 1ns total guard), only windowed and
    // unnormalized so callers can aggregate across ranks before dividing
    // (doc sec.8.7). Read-only.
    struct WindowedRooflineUtilization {
        double compute_utilization_weighted_sum = 0;
        double memory_utilization_weighted_sum = 0;
        // Die-backend-only field (stage B, contract §7.6): eq-ops / (vector
        // peak * elapsed), aggregated with the same duration weighting and
        // the same exact accumulator as the two sums above. Zero for every
        // legacy backend (no producer ever set the field).
        double vector_utilization_weighted_sum = 0;
        // Includes the same 1ns guard as the removed extract_utilizations().
        Tick total_comp_time = 1ul;
    };
    WindowedRooflineUtilization calculate_roofline_utilization_in_window(
        Tick window_start,
        Tick window_end) const;

    // Workload fixes this policy before issuing the first online node.  It
    // lets the service path fail closed on a live GPU even when no terminal
    // record has yet activated aggregate compaction, while microbenchmarks
    // retain their legacy arbitrary-window history queries.
    void configure_online_history_preservation(bool preserve_history) {
        online_mode_configured_ = true;
        online_history_preserved_ = preserve_history;
    }

    [[nodiscard]] bool online_history_preserved() const {
        return online_history_preserved_;
    }

    // Read-only diagnostic for online RSS tests.  In compact service mode it
    // is the current in-flight/tail window, not the executed-node total.
    [[nodiscard]] size_t retained_online_operator_count() const {
        return operator_statistics.size();
    }

    ~Statistics() {
        // G2-15: clear() keeps the hash buckets allocated; swap-with-empty
        // (same pattern as Statistics.cc's online_gpu_active_nodes_ reset)
        // releases the reserved memory along with the elements.
        std::unordered_map<NodeId, OperatorStatistics>().swap(
            operator_statistics);
    }

    // Legacy finalization scans operator_statistics. Compact online service
    // intentionally does not retain that per-node history, so fail closed
    // before a legacy consumer can overwrite/report invalid aggregates.
    // This preflight has no statistics-side effects.
    void ensure_legacy_post_processing_supported() const;

    void post_processing();

    // (The report(logger)/report() pair was removed with its zero callers;
    // see the note in Statistics.cc.)

  private:
    // Canonical online roofline sum: each contribution is first rounded as
    // the legacy double expression (utilization * duration), decomposed into
    // its IEEE-754 integer significand, and accumulated exactly in 2304 fixed
    // bits.  One round-to-nearest-even happens only at query time.  This is
    // independent of completion order and unordered_map iteration order.
    struct OnlineExactDoubleSum {
        static constexpr size_t kBitCount = 2304;
        static constexpr size_t kLimbCount = kBitCount / 64;
        std::array<uint64_t, kLimbCount> limbs{};

        void add(double rounded_value);
        [[nodiscard]] double value() const;
    };

    // (The legacy finalize aggregation chain -- extract_type_time,
    // extract_utilizations, extract_comp_comm_overlap and their six
    // write-only members type_time / comp_comm_overlap /
    // compute_bound_percentage_ / average_compute_utilization_ /
    // average_memory_utilization_ / average_operation_intensity_ -- was
    // removed (2026-10-01): a whole-repo grep found zero readers outside
    // the chain itself, and post_processing's only remaining callers are
    // the fail-closed preflight test and the history-preserving
    // microbenchmark fixture, neither of which reads those aggregates.)
    Tick _calculateTotalRuntimeFromIntervals(
        const std::vector<std::pair<Tick, Tick>>& intervals) const;

    // NodeView-only GPU accounting. The active-count integral closes a
    // fully-completed connected segment in O(1). A successful online service
    // run cannot finalize with live GPU nodes; querying such an incomplete
    // run fails closed instead of retaining an unbounded completed-interval
    // history behind one long-lived node.
    void note_online_gpu_start(NodeId node_id, Tick start_time);
    void note_online_gpu_end(NodeId node_id, Tick end_time);
    void integrate_online_gpu_active_to(Tick tick);
    Tick online_gpu_completed_busy() const;
    void add_online_roofline_contribution(
        OperatorStatistics::OperatorType type,
        Tick start_time,
        Tick end_time,
        const std::optional<double>& compute_utilization,
        const std::optional<double>& memory_utilization,
        const std::optional<double>& vector_utilization =
            std::nullopt);
    void add_online_total_comp_time(Tick duration);
    void require_online_compacted_full_window(Tick window_start,
                                              Tick window_end,
                                              Tick newest_retired_end,
                                              const char* consumer) const;

    // (The six legacy finalize aggregates -- type_time, comp_comm_overlap,
    // compute_bound_percentage_, average_compute_utilization_,
    // average_memory_utilization_, average_operation_intensity_ -- were
    // removed with the extract_* chain that was their only writer; nothing
    // ever read them. See the note near _calculateTotalRuntimeFromIntervals.)
    // Fix (2026-08-28): zero-initialized + maintained as a running max in
    // record_end. Online mode reads get_wall_time() at MetricCollector
    // finalize -- BEFORE Workload's post_processing() ever computes it -- so
    // the member used to be an uninitialized read and the rank_compute
    // records emitted garbage wall_time_ns values (varying per run). With the
    // running max the finalize-time value is the true "max operator end_time"
    // (identical to post_processing()'s own recompute, which stays as is).
    Tick wall_time = 0;
    Workload* workload;
    std::unordered_map<NodeId, OperatorStatistics> operator_statistics;
    // A3-lite (2026-08-28): the start_times multimap was removed -- a
    // whole-tree grep found zero readers (only the insertions and the
    // destructor clear), so it was pure per-node dead weight. The
    // operator_statistics map stays complete for static ET and online
    // microbenchmark runs, whose legacy finalize consumers scan it directly.
    // Online service runs instead retire terminal records into compact,
    // completion-order aggregates; only their full-prefix GPU/roofline
    // queries are supported after compaction.

    // Online service-only compact aggregates.  They are populated exclusively
    // by complete_online_service_operator(); static and online microbenchmark
    // runs keep the legacy unordered_map scan unchanged.
    bool online_compaction_active_ = false;
    // Set at the first direct compact-service record, including a live node.
    // It keeps legacy finalization from silently treating an empty map as a
    // completed service run before a terminal record activates compaction.
    bool online_compact_statistics_recorded_ = false;
    bool online_mode_configured_ = false;
    bool online_history_preserved_ = false;
    OnlineExactDoubleSum online_compute_utilization_weighted_sum_;
    OnlineExactDoubleSum online_memory_utilization_weighted_sum_;
    // Stage B (die backend, contract §7.6): vector (eq-op) utilization
    // aggregate, same exact-accumulator discipline as the two above.
    OnlineExactDoubleSum online_vector_utilization_weighted_sum_;
    Tick online_total_comp_time_ = 1ul;
    Tick online_roofline_newest_retired_end_ = 0;

    std::unordered_set<NodeId> online_gpu_active_nodes_;
    uint64_t online_gpu_active_count_ = 0;
    Tick online_gpu_active_last_tick_ = 0;
    Tick online_gpu_active_union_ns_ = 0;
    Tick online_gpu_segment_union_begin_ = 0;
    bool online_gpu_segment_open_ = false;
    Tick online_gpu_closed_completed_busy_ns_ = 0;
    Tick online_gpu_newest_retired_end_ = 0;
};

}  // namespace AstraSim

#endif /*ASTRASIM_WORKLOAD_STATISTICS_HH*/
