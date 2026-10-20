#include "astra-sim/workload/Statistics.hh"
#include "astra-sim/system/Sys.hh"
#include "astra-sim/workload/Workload.hh"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace AstraSim;

namespace {

double rounded_online_weighted_utilization(double utilization,
                                           Tick duration,
                                           const char* field) {
    if (!std::isfinite(utilization) || utilization < 0.0) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online {} utilization must be finite and non-negative; "
                       "got {}",
                       field, utilization);
        std::exit(EXIT_FAILURE);
    }
    // Preserve the old expression's per-node IEEE-754 double rounding before
    // the canonical accumulator consumes the term.
    const double rounded = utilization * static_cast<double>(duration);
    if (!std::isfinite(rounded) || rounded < 0.0) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online {} weighted utilization is not finite/non-negative "
                       "for duration {}",
                       field, duration);
        std::exit(EXIT_FAILURE);
    }
    return rounded;
}

void checked_add_online_tick(Tick& total, Tick value, const char* consumer) {
    if (value > std::numeric_limits<Tick>::max() - total) {
        LoggerFactory::get_logger("statistics")
            ->critical("{} overflow while accumulating online compute time",
                       consumer);
        std::exit(EXIT_FAILURE);
    }
    total += value;
}

}  // namespace

Statistics::Statistics() = default;

Statistics::OperatorStatistics& Statistics::get_operator_statistics(
    NodeId node_id) {
    return operator_statistics.at(node_id);
}

// ---------------------------------------------------------- NodeView (step 1-8)

void Statistics::record_start(const ExecutionDriven::NodeView& node,
                              Tick start_time) {
    const NodeId& node_id = node.global_id;
    if (online_mode_configured_ &&
        operator_statistics.find(node_id) != operator_statistics.end()) {
        LoggerFactory::get_logger("statistics")
            ->critical("Duplicate online statistics start for node {}", node_id);
        std::exit(EXIT_FAILURE);
    }
    const auto type = OperatorStatistics::get_operator_type(node);
    operator_statistics[node_id] = OperatorStatistics(start_time, type);
    if (type == OperatorStatistics::OperatorType::GPU) {
        note_online_gpu_start(node_id, start_time);
    }
}

void Statistics::record_end(const ExecutionDriven::NodeView& node,
                            Tick end_time) {
    const NodeId& node_id = node.global_id;
    auto& stat = this->get_operator_statistics(node_id);
    if (online_mode_configured_ &&
        stat.end_time != OperatorStatistics::INVALID_TICK) {
        LoggerFactory::get_logger("statistics")
            ->critical("Duplicate online statistics end for node {}", node_id);
        std::exit(EXIT_FAILURE);
    }
    const bool first_completion =
        stat.end_time == OperatorStatistics::INVALID_TICK;
    stat.end_time = end_time;
    if (first_completion &&
        stat.type == OperatorStatistics::OperatorType::GPU) {
        note_online_gpu_end(node_id, end_time);
    }
    // Fix (2026-08-28): running max so get_wall_time() is valid at
    // finalize time (online finalize reads it; see Statistics.hh).
    if (end_time > this->wall_time) {
        this->wall_time = end_time;
    }
}

void Statistics::record_online_service_start(
    const ExecutionDriven::NodeView& node,
    ExecutionDriven::OnlineStatisticsState& state,
    Tick start_time) {
    if (!online_mode_configured_ || online_history_preserved_) {
        LoggerFactory::get_logger("statistics")
            ->critical("Compact online statistics start used without compact "
                       "online-service configuration for node {}",
                       node.global_id);
        std::exit(EXIT_FAILURE);
    }
    if (state.started || state.completed) {
        LoggerFactory::get_logger("statistics")
            ->critical("Duplicate compact online statistics start for node {}",
                       node.global_id);
        std::exit(EXIT_FAILURE);
    }

    const auto type = OperatorStatistics::get_operator_type(node);
    state.start_time = start_time;
    state.started = true;
    state.is_gpu = type == OperatorStatistics::OperatorType::GPU;
    if (state.is_gpu) {
        note_online_gpu_start(node.global_id, start_time);
    }
}

void Statistics::complete_online_service_operator(
    const ExecutionDriven::NodeView& node,
    ExecutionDriven::OnlineStatisticsState& state,
    Tick end_time) {
    if (!online_mode_configured_ || online_history_preserved_) {
        LoggerFactory::get_logger("statistics")
            ->critical("Compact online statistics completion used without "
                       "compact online-service configuration for node {}",
                       node.global_id);
        std::exit(EXIT_FAILURE);
    }
    if (!state.started || state.completed ||
        state.start_time == ExecutionDriven::OnlineStatisticsState::kInvalidTick) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online statistics completion without one live start for "
                       "node {}",
                       node.global_id);
        std::exit(EXIT_FAILURE);
    }
    if (end_time < state.start_time) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online statistics end regressed for node {} from {} to {}",
                       node.global_id, state.start_time, end_time);
        std::exit(EXIT_FAILURE);
    }

    const auto type = OperatorStatistics::get_operator_type(node);
    const bool is_gpu = type == OperatorStatistics::OperatorType::GPU;
    if (state.is_gpu != is_gpu) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online statistics type changed between start/end for "
                       "node {}",
                       node.global_id);
        std::exit(EXIT_FAILURE);
    }
    if (is_gpu) {
        note_online_gpu_end(node.global_id, end_time);
    }

    // Compact-mode completion folds the terminal duration immediately into
    // the streaming aggregators below; no per-node end-time snapshot is
    // retained (the former write-only end_time member is gone).
    state.completed = true;
    if (end_time > wall_time) {
        wall_time = end_time;
    }
    online_compaction_active_ = true;
    add_online_roofline_contribution(type, state.start_time, end_time,
                                     state.compute_utilization,
                                     state.memory_utilization,
                                     state.vector_utilization);
    if (is_gpu && end_time > state.start_time) {
        online_gpu_newest_retired_end_ =
            std::max(online_gpu_newest_retired_end_, end_time);
    }
}

Tick Statistics::get_wall_time() const {
    return this->wall_time;
}

Tick Statistics::calculate_type_time_in_window(
    OperatorStatistics::OperatorType type,
    Tick window_start,
    Tick window_end) const {
    // Service Statistics can be queried before the first terminal record has
    // enabled aggregate compaction.  Do not let that zero-retirement state
    // silently report a completed-only GPU total while a GPU is still live.
    // Microbenchmarks explicitly preserve node history and retain the legacy
    // clipped-window semantics instead.
    if (type == OperatorStatistics::OperatorType::GPU &&
        !online_history_preserved_ &&
        (online_gpu_active_count_ != 0 || online_gpu_segment_open_ ||
         !online_gpu_active_nodes_.empty())) {
        return online_gpu_completed_busy();
    }
    if (online_compaction_active_) {
        if (type != OperatorStatistics::OperatorType::GPU) {
            LoggerFactory::get_logger("statistics")
                ->critical("Online compact Statistics cannot answer type {} "
                           "window queries", static_cast<int>(type));
            std::exit(EXIT_FAILURE);
        }
        require_online_compacted_full_window(
            window_start, window_end, online_gpu_newest_retired_end_,
            "calculate_type_time_in_window");
        return online_gpu_completed_busy();
    }
    std::vector<std::pair<Tick, Tick>> clipped_intervals;
    for (const auto& [node_id, stat] : operator_statistics) {
        if (stat.type != type) {
            continue;
        }
        if (stat.end_time == OperatorStatistics::INVALID_TICK ||
            stat.end_time <= stat.start_time) {
            continue;
        }
        const Tick clipped_start = std::max(stat.start_time, window_start);
        const Tick clipped_end = std::min(stat.end_time, window_end);
        if (clipped_end > clipped_start) {
            clipped_intervals.push_back({clipped_start, clipped_end});
        }
    }
    return _calculateTotalRuntimeFromIntervals(clipped_intervals);
}

Statistics::WindowedRooflineUtilization
Statistics::calculate_roofline_utilization_in_window(
    Tick window_start,
    Tick window_end) const {
    if (online_compaction_active_) {
        require_online_compacted_full_window(
            window_start, window_end, online_roofline_newest_retired_end_,
            "calculate_roofline_utilization_in_window");
        WindowedRooflineUtilization result;
        result.compute_utilization_weighted_sum =
            online_compute_utilization_weighted_sum_.value();
        result.memory_utilization_weighted_sum =
            online_memory_utilization_weighted_sum_.value();
        result.vector_utilization_weighted_sum =
            online_vector_utilization_weighted_sum_.value();
        result.total_comp_time = online_total_comp_time_;
        return result;
    }
    if (online_mode_configured_) {
        // History-preserving online microbenchmarks retain arbitrary windows,
        // but their queried roofline sum is still the same canonical exact
        // accumulator as compact service mode.  Iteration order therefore
        // cannot change the final IEEE-754 bits.
        WindowedRooflineUtilization result;
        OnlineExactDoubleSum compute_sum;
        OnlineExactDoubleSum memory_sum;
        for (const auto& [node_id, stat] : operator_statistics) {
            (void)node_id;
            if (stat.type != OperatorStatistics::OperatorType::CPU &&
                stat.type != OperatorStatistics::OperatorType::GPU) {
                continue;
            }
            if (stat.end_time == OperatorStatistics::INVALID_TICK ||
                stat.end_time <= stat.start_time) {
                continue;
            }
            const Tick clipped_start = std::max(stat.start_time, window_start);
            const Tick clipped_end = std::min(stat.end_time, window_end);
            if (clipped_end <= clipped_start) {
                continue;
            }
            const Tick duration = clipped_end - clipped_start;
            if (stat.compute_utilization.has_value()) {
                compute_sum.add(rounded_online_weighted_utilization(
                    stat.compute_utilization.value(), duration, "compute"));
            }
            if (stat.memory_utilization.has_value()) {
                memory_sum.add(rounded_online_weighted_utilization(
                    stat.memory_utilization.value(), duration, "memory"));
            }
            checked_add_online_tick(result.total_comp_time, duration,
                                    "online history roofline query");
        }
        result.compute_utilization_weighted_sum = compute_sum.value();
        result.memory_utilization_weighted_sum = memory_sum.value();
        return result;
    }
    // Unreachable: Workload configures every Statistics it constructs before
    // issuing the first node (and the regression tests configure theirs the
    // same way), so online_mode_configured_ is always true by query time.
    // The former legacy double-precision tail was dead from there.
    LoggerFactory::get_logger("statistics")
        ->critical("roofline window query on a Statistics whose online mode "
                   "was never configured");
    std::exit(EXIT_FAILURE);
}

void Statistics::OnlineExactDoubleSum::add(double rounded_value) {
    if (!std::isfinite(rounded_value) || rounded_value < 0.0) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online exact roofline accumulator accepts only finite "
                       "non-negative rounded doubles");
        std::exit(EXIT_FAILURE);
    }
    if (rounded_value == 0.0) {
        return;  // +0 and -0 are both an exact no-op.
    }

    uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(rounded_value),
                  "IEEE-754 double must be 64 bits");
    std::memcpy(&bits, &rounded_value, sizeof(bits));
    if ((bits >> 63) != 0) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online exact roofline accumulator received a negative "
                       "rounded double");
        std::exit(EXIT_FAILURE);
    }

    const uint64_t exponent = (bits >> 52) & 0x7ffu;
    const uint64_t fraction = bits & ((uint64_t{1} << 52) - 1);
    uint64_t significand = 0;
    size_t shift = 0;  // units are 2^-1074, the minimum IEEE-754 subnormal.
    if (exponent == 0) {
        significand = fraction;
    } else {
        significand = (uint64_t{1} << 52) | fraction;
        shift = static_cast<size_t>(exponent - 1);
    }
    if (significand == 0) {
        return;
    }

    auto add_word = [this](size_t index, uint64_t word) {
        if (word == 0) {
            return;
        }
        if (index >= kLimbCount) {
            LoggerFactory::get_logger("statistics")
                ->critical("Online exact roofline accumulator overflow");
            std::exit(EXIT_FAILURE);
        }
        const uint64_t before = limbs[index];
        limbs[index] += word;
        if (limbs[index] >= before) {
            return;
        }
        for (++index; index < kLimbCount; ++index) {
            ++limbs[index];
            if (limbs[index] != 0) {
                return;
            }
        }
        LoggerFactory::get_logger("statistics")
            ->critical("Online exact roofline accumulator overflow");
        std::exit(EXIT_FAILURE);
    };

    const size_t limb_index = shift / 64;
    const unsigned int bit_offset = static_cast<unsigned int>(shift % 64);
    add_word(limb_index, significand << bit_offset);
    if (bit_offset != 0) {
        add_word(limb_index + 1, significand >> (64 - bit_offset));
    }
}

double Statistics::OnlineExactDoubleSum::value() const {
    size_t high_limb = kLimbCount;
    while (high_limb != 0 && limbs[high_limb - 1] == 0) {
        --high_limb;
    }
    if (high_limb == 0) {
        return 0.0;
    }
    --high_limb;
    const unsigned int high_bit =
        63u - static_cast<unsigned int>(__builtin_clzll(limbs[high_limb]));
    const size_t highest = high_limb * 64 + high_bit;

    uint64_t result_bits = 0;
    if (highest <= 52) {
        // The integer units directly encode either a subnormal or the first
        // normal binade (exponent field 1), both exactly representable.
        result_bits = limbs[0];
    } else {
        const size_t shift = highest - 52;
        const size_t word = shift / 64;
        const unsigned int offset = static_cast<unsigned int>(shift % 64);
        uint64_t significand = limbs[word] >> offset;
        if (offset != 0 && word + 1 < kLimbCount) {
            significand |= limbs[word + 1] << (64 - offset);
        }

        const size_t half_bit = shift - 1;
        const bool at_least_half =
            ((limbs[half_bit / 64] >> (half_bit % 64)) & 1u) != 0;
        bool below_half_nonzero = false;
        const size_t whole_below = half_bit / 64;
        for (size_t i = 0; i < whole_below; ++i) {
            if (limbs[i] != 0) {
                below_half_nonzero = true;
                break;
            }
        }
        if (!below_half_nonzero) {
            const unsigned int partial_bits =
                static_cast<unsigned int>(half_bit % 64);
            if (partial_bits != 0 &&
                (limbs[whole_below] &
                 ((uint64_t{1} << partial_bits) - 1)) != 0) {
                below_half_nonzero = true;
            }
        }
        if (at_least_half && (below_half_nonzero || (significand & 1u) != 0)) {
            ++significand;  // round-to-nearest, ties-to-even
        }

        uint64_t exponent = static_cast<uint64_t>(highest - 51);
        if (significand == (uint64_t{1} << 53)) {
            significand >>= 1;
            ++exponent;
        }
        if (exponent >= 0x7ffu) {
            LoggerFactory::get_logger("statistics")
                ->critical("Online exact roofline sum rounds beyond finite "
                           "IEEE-754 double range");
            std::exit(EXIT_FAILURE);
        }
        result_bits = (exponent << 52) |
                      (significand & ((uint64_t{1} << 52) - 1));
    }

    double result = 0.0;
    std::memcpy(&result, &result_bits, sizeof(result));
    return result;
}

void Statistics::add_online_total_comp_time(Tick duration) {
    checked_add_online_tick(online_total_comp_time_, duration,
                            "compact online roofline aggregation");
}

void Statistics::add_online_roofline_contribution(
    OperatorStatistics::OperatorType type,
    Tick start_time,
    Tick end_time,
    const std::optional<double>& compute_utilization,
    const std::optional<double>& memory_utilization,
    const std::optional<double>& vector_utilization) {
    if ((type != OperatorStatistics::OperatorType::CPU &&
         type != OperatorStatistics::OperatorType::GPU) ||
        end_time <= start_time) {
        return;
    }
    const Tick duration = end_time - start_time;
    if (compute_utilization.has_value()) {
        online_compute_utilization_weighted_sum_.add(
            rounded_online_weighted_utilization(compute_utilization.value(),
                                                duration, "compute"));
    }
    if (memory_utilization.has_value()) {
        online_memory_utilization_weighted_sum_.add(
            rounded_online_weighted_utilization(memory_utilization.value(),
                                                duration, "memory"));
    }
    if (vector_utilization.has_value()) {
        online_vector_utilization_weighted_sum_.add(
            rounded_online_weighted_utilization(vector_utilization.value(),
                                                duration, "vector"));
    }
    add_online_total_comp_time(duration);
    online_roofline_newest_retired_end_ =
        std::max(online_roofline_newest_retired_end_, end_time);
}

void Statistics::integrate_online_gpu_active_to(Tick tick) {
    if (tick < online_gpu_active_last_tick_) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online GPU timing regressed from {} to {}",
                       online_gpu_active_last_tick_, tick);
        std::exit(EXIT_FAILURE);
    }
    if (online_gpu_active_count_ > 0) {
        checked_add_online_tick(online_gpu_active_union_ns_,
                                tick - online_gpu_active_last_tick_,
                                "online GPU active-count union");
    }
    online_gpu_active_last_tick_ = tick;
}

void Statistics::note_online_gpu_start(NodeId node_id, Tick start_time) {
    if (online_gpu_active_nodes_.find(node_id) !=
        online_gpu_active_nodes_.end()) {
        LoggerFactory::get_logger("statistics")
            ->critical("Duplicate online GPU start for node {}", node_id);
        std::exit(EXIT_FAILURE);
    }
    if (online_gpu_active_count_ == std::numeric_limits<uint64_t>::max()) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online GPU active-count overflow for node {}", node_id);
        std::exit(EXIT_FAILURE);
    }
    integrate_online_gpu_active_to(start_time);
    const auto [it, inserted] = online_gpu_active_nodes_.emplace(node_id);
    (void)it;
    if (!inserted) {
        LoggerFactory::get_logger("statistics")
            ->critical("Duplicate online GPU start for node {}", node_id);
        std::exit(EXIT_FAILURE);
    }
    if (online_gpu_active_count_ == 0) {
        online_gpu_segment_open_ = true;
        online_gpu_segment_union_begin_ = online_gpu_active_union_ns_;
    }
    ++online_gpu_active_count_;
}

void Statistics::note_online_gpu_end(NodeId node_id, Tick end_time) {
    const auto active_it = online_gpu_active_nodes_.find(node_id);
    if (active_it == online_gpu_active_nodes_.end()) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online GPU end without a live start for node {}",
                       node_id);
        std::exit(EXIT_FAILURE);
    }
    if (online_gpu_active_count_ == 0) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online GPU active-count underflow for node {}",
                       node_id);
        std::exit(EXIT_FAILURE);
    }
    // Validate the id/count before advancing the integral, then mutate.  A
    // duplicate/end-without-start therefore fails closed without accepting a
    // partial second terminal transition.
    integrate_online_gpu_active_to(end_time);
    online_gpu_active_nodes_.erase(active_it);
    --online_gpu_active_count_;
    if (online_gpu_active_count_ == 0) {
        if (!online_gpu_segment_open_) {
            LoggerFactory::get_logger("statistics")
                ->critical("Online GPU segment closed without an open segment");
            std::exit(EXIT_FAILURE);
        }
        // A zero active count proves the entire connected segment is
        // complete.  The active-count integral is therefore the exact union
        // of its intervals, including overlap and same-tick endpoints.
        checked_add_online_tick(
            online_gpu_closed_completed_busy_ns_,
            online_gpu_active_union_ns_ - online_gpu_segment_union_begin_,
            "online GPU completed union");
        online_gpu_segment_open_ = false;
        // Erase does not shrink an unordered_set's bucket array. Release a
        // large historical concurrency peak at the idle boundary while
        // retaining small tables to avoid allocator churn on ordinary runs.
        constexpr size_t kRetainedGpuBuckets = 1024;
        if (online_gpu_active_nodes_.bucket_count() >
            kRetainedGpuBuckets) {
            std::unordered_set<NodeId>().swap(online_gpu_active_nodes_);
        }
    }
}

Tick Statistics::online_gpu_completed_busy() const {
    if (online_gpu_active_count_ != 0 || online_gpu_segment_open_ ||
        !online_gpu_active_nodes_.empty()) {
        LoggerFactory::get_logger("statistics")
            ->critical("Online compact Statistics finalized with {} live GPU "
                       "nodes; successful service completion requires every "
                       "issued node to reach a terminal state",
                       online_gpu_active_count_);
        std::exit(EXIT_FAILURE);
    }
    return online_gpu_closed_completed_busy_ns_;
}

void Statistics::require_online_compacted_full_window(
    Tick window_start, Tick window_end, Tick newest_retired_end,
    const char* consumer) const {
    if (window_start == 0 && window_end >= newest_retired_end) {
        return;
    }
    LoggerFactory::get_logger("statistics")
        ->critical("{} cannot query compacted online Statistics window "
                   "[{}, {}] with retired data through {}", consumer,
                   window_start, window_end, newest_retired_end);
    std::exit(EXIT_FAILURE);
}

Statistics::OperatorStatistics::OperatorType Statistics::OperatorStatistics::
    get_operator_type(const ExecutionDriven::NodeView& node) {
    using ExecutionDriven::NodeKind;
    // Initialize before any switch arm so a future NodeKind can never
    // return an uninitialized enum (UB) in -DNDEBUG builds.
    Statistics::OperatorStatistics::OperatorType stat_node_type =
        Statistics::OperatorStatistics::OperatorType::INVALID;
    switch (node.kind) {
    case NodeKind::MemLoad:
    case NodeKind::MemStore:
        stat_node_type =
            Statistics::OperatorStatistics::OperatorType::REMOTE_MEM;
        break;
    case NodeKind::Compute:
        stat_node_type =
            node.is_cpu_op
                ? Statistics::OperatorStatistics::OperatorType::CPU
                : Statistics::OperatorStatistics::OperatorType::GPU;
        break;
    case NodeKind::CommCollective:
    case NodeKind::CommSend:
    case NodeKind::CommRecv:
        stat_node_type = Statistics::OperatorStatistics::OperatorType::COMM;
        break;
    case NodeKind::Invalid:
    case NodeKind::Metadata:
        stat_node_type = Statistics::OperatorStatistics::OperatorType::INVALID;
        break;
    default:
        LoggerFactory::get_logger("statistics")
            ->critical("Invalid node kind, node.id={}, node.kind={}",
                       node.global_id, static_cast<int>(node.kind));
        // Fail closed (NDEBUG no longer has an assert to catch this).
        std::exit(EXIT_FAILURE);
    }
    return stat_node_type;
}

Tick Statistics::_calculateTotalRuntimeFromIntervals(
    const std::vector<std::pair<Tick, Tick>>& intervals) const {
    if (intervals.empty()) {
        return 0;
    }
    std::vector<std::pair<Tick, Tick>> sorted_intervals = intervals;
    sort(sorted_intervals.begin(), sorted_intervals.end());

    Tick total_runtime = 0;
    Tick merged_start = sorted_intervals[0].first;
    Tick merged_end = sorted_intervals[0].second;

    for (const auto& [start, end] : sorted_intervals) {
        if (start <= merged_end) {
            merged_end = std::max(merged_end, end);
        } else {
            total_runtime += merged_end - merged_start;
            merged_start = start;
            merged_end = end;
        }
    }
    total_runtime += merged_end - merged_start;
    return total_runtime;
}
