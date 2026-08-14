#ifndef ASTRASIM_WORKLOAD_STATISTICS_HH
#define ASTRASIM_WORKLOAD_STATISTICS_HH

#include "astra-sim/common/Common.hh"
#include "astra-sim/common/Logging.hh"
#include <map>
#include <memory>
#include <optional>
#include <unordered_map>

#include "extern/graph_frontend/chakra/src/feeder_v3/et_feeder.h"

typedef ChakraProtoMsg::NodeType ChakraNodeType;

typedef uint64_t NodeId;

namespace AstraSim {
class Workload;
class LocalMemoryTracker;
class Statistics {
  public:
    class OperatorStatistics {
      public:
        static const Tick INVALID_TICK = UINT64_MAX;
        enum class OperatorType { CPU, GPU, COMM, REMOTE_MEM, REPLAY, INVALID };
        static OperatorType get_operator_type(
            const std::shared_ptr<Chakra::ETFeederNode> node);
        OperatorStatistics(NodeId node_id,
                           Tick start_time,
                           Tick end_time,
                           OperatorType type)
            : node_id(node_id),
              start_time(start_time),
              end_time(end_time),
              type(type) {}
        OperatorStatistics(NodeId node_id, Tick start_time, OperatorType type)
            : node_id(node_id),
              start_time(start_time),
              end_time(INVALID_TICK),
              type(type) {}
        OperatorStatistics()
            : node_id(UINT64_MAX),
              start_time(INVALID_TICK),
              end_time(INVALID_TICK),
              type(OperatorType::INVALID) {}

        NodeId node_id;
        Tick start_time;
        Tick end_time;
        OperatorType type;

        // compute node
        std::optional<double> memory_utilization;
        std::optional<double> compute_utilization;
        std::optional<double> operation_intensity;
        std::optional<bool> is_memory_bound;

        // communication node
        std::optional<uint64_t> comm_size;  // Size of communication in bytes
        std::optional<double>
            network_bandwidth;  // Achieved bandwidth in bytes/ns

        // remote memory node

        // replay node
    };

  public:
    Statistics(Workload* workload);

    OperatorStatistics& get_operator_statistics(NodeId node_id);

    const OperatorStatistics& get_operator_statistics(NodeId node_id) const;

    const std::unordered_map<NodeId, OperatorStatistics>&
    get_operator_statistics() const;

    void record_start(std::shared_ptr<Chakra::ETFeederNode> node,
                      Tick start_time);

    void record_end(std::shared_ptr<Chakra::ETFeederNode> node, Tick end_time);

    // Read-only accessors for side-band metric computation (doc sec.5.7).
    // Valid after post_processing() has run. They only scan existing
    // operator statistics and never influence simulation execution.

    // Wall time of this rank: max operator end_time.
    Tick get_wall_time() const;

    // Merged union of all intervals of the given operator type.
    Tick get_type_time(OperatorStatistics::OperatorType type) const;

    // Merged union of the intervals of the given operator type, clipped to
    // [window_start, window_end].
    Tick calculate_type_time_in_window(
        OperatorStatistics::OperatorType type,
        Tick window_start,
        Tick window_end) const;

    // Duration-weighted sums of the roofline per-node utilization fields over
    // CPU/GPU operator statistics, clipped to [window_start, window_end].
    // Same weighting formula as extract_utilizations() ("Average compute
    // utilization"), only windowed and unnormalized so callers can aggregate
    // across ranks before dividing (doc sec.8.7). Read-only.
    struct WindowedRooflineUtilization {
        double compute_utilization_weighted_sum = 0;
        double memory_utilization_weighted_sum = 0;
        // Includes the same 1ns guard as extract_utilizations().
        Tick total_comp_time = 1ul;
    };
    WindowedRooflineUtilization calculate_roofline_utilization_in_window(
        Tick window_start,
        Tick window_end) const;

    ~Statistics() {
        operator_statistics.clear();
    }

    void post_processing();

    void report(std::shared_ptr<spdlog::logger> logger) const;

    void report() const;

  private:
    void extract_type_time();
    Tick _calculateTotalRuntimeFromIntervals(
        const std::vector<std::pair<Tick, Tick>>& intervals) const;
    void extract_utilizations();
    void extract_comp_comm_overlap();

    std::unordered_map<OperatorStatistics::OperatorType, Tick> type_time;
    Tick wall_time;
    Tick comp_comm_overlap;
    double compute_bound_percentage_;
    double average_compute_utilization_;
    double average_memory_utilization_;
    double average_operation_intensity_;
    Workload* workload;
    std::unordered_map<NodeId, OperatorStatistics> operator_statistics;
    std::multimap<Tick, NodeId> start_times;
};

}  // namespace AstraSim

#endif /*ASTRASIM_WORKLOAD_STATISTICS_HH*/
