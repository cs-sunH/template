/******************************************************************************
Checked algebra for compact remote-read ranges. All tile queries are lazy;
memory use is O(shards), never O(total virtual tiles).
******************************************************************************/

#ifndef EXECUTION_DRIVEN_PIPELINE_KERNEL_HH
#define EXECUTION_DRIVEN_PIPELINE_KERNEL_HH

#include <cstdint>
#include <string>
#include <vector>

#include "astra-sim/workload/execution_driven/PipelineDescriptor.hh"

namespace AstraSim {
namespace ExecutionDriven {

/// Auditable result of assigning integer service across a bounded repeated
/// window. A plan contains only current, already-known reference slices; it
/// never predicts a completion or publishes work.
struct PipelineWindowPlan final {
    uint32_t requested_limit = 1;
    uint32_t tier_limit = 1;
    uint32_t selected_count = 0;
    uint32_t refinement_count = 0;
    uint64_t window_generation = 0;
    uint64_t original_total_ticks = 0;
    uint64_t local_span_ticks = 0;
    uint64_t max_prefix_delta_ticks = 0;
    bool accepted = false;
    bool exact = false;
    std::string unsupported_reason;
    std::vector<uint64_t> original_service_ticks;
    std::vector<uint64_t> assigned_service_ticks;

    nlohmann::json to_json(const PipelineKernelContract& contract,
                           uint32_t effective_limit) const;
};

/// Per-executor precision controller and disclosure counters. Mutation and
/// structural boundaries call invalidate(); due-event cookies remain owned
/// by the caller and are never cancelled by this value object.
class PipelineWindowController final {
  public:
    explicit PipelineWindowController(uint32_t configured_limit);

    PipelineWindowPlan plan(const PipelineKernelContract& contract,
                            uint32_t effective_limit,
                            const std::vector<uint64_t>& original_service_ticks,
                            const std::vector<uint64_t>& hardware_minimum_ticks,
                            uint64_t local_span_ticks) const;
    void record(const PipelineWindowPlan& plan, bool committed);
    /// Account an externally committed prefix of a window that was cut by a
    /// real mutation. It counts delivered approximate work but never raises
    /// the stable-window tier.
    void record_partial(const PipelineWindowPlan& plan);
    void invalidate();

    uint32_t current_limit() const { return current_limit_; }
    uint32_t stable_windows() const { return stable_windows_; }
    uint64_t generation() const { return generation_; }
    uint64_t window_count() const { return window_count_; }
    uint64_t refinement_count() const { return refinement_count_; }
    uint64_t local_prefix_bound_max() const {
        return local_prefix_bound_max_;
    }
    nlohmann::json counters_json(const PipelineKernelContract& contract,
                                 uint32_t effective_limit) const;

  private:
    uint32_t configured_limit_;
    uint32_t current_limit_;
    uint32_t stable_windows_ = 0;
    uint64_t generation_ = 0;
    uint64_t window_count_ = 0;
    uint64_t refinement_count_ = 0;
    uint64_t local_prefix_bound_max_ = 0;
};

class PipelineKernel final {
  public:
    /// local_tile_ordinal is zero-based; returned global ordinal is one-based.
    static PipelineTile tile(const PipelineDescriptor& descriptor,
                             uint64_t group_ordinal,
                             uint64_t local_tile_ordinal);

    /// Sum data bytes for the whole group without enumerating its tiles.
    static uint64_t total_data_bytes(const PipelineDescriptor& descriptor,
                                     uint64_t group_ordinal);

    /// Sum the first count tiles without enumerating all preceding tiles.
    static uint64_t prefix_data_bytes(const PipelineDescriptor& descriptor,
                                      uint64_t group_ordinal,
                                      uint64_t count);

    /// Construct a bounded precision plan. Hardware minima are per logical
    /// slice; an empty vector means no additional floor beyond the exact
    /// service already present in the input. effective_limit must be one of
    /// 1/2/4/8 and no greater than the immutable run contract.
    static PipelineWindowPlan plan_window(
        const PipelineKernelContract& contract,
        uint32_t effective_limit,
        uint32_t preferred_limit,
        const std::vector<uint64_t>& original_service_ticks,
        const std::vector<uint64_t>& hardware_minimum_ticks,
        uint64_t local_span_ticks);

    /// Map elapsed external ticks to the exact original-service prefix of a
    /// committed plan, piecewise at each assigned tile boundary.
    static uint64_t original_service_at_external_tick(
        const PipelineWindowPlan& plan, uint64_t external_elapsed_ticks);
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_PIPELINE_KERNEL_HH
