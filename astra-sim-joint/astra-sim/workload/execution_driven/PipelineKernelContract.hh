/******************************************************************************
Run-scoped prediction/runtime ABI contract carried by every GraphBatch
(D4, three-action retirement 2026-10-14: the former tail of
PipelineDescriptor.hh, kept verbatim as the die compute backend's timer
window contract; the descriptor (tile pipeline) wire it used to live next
to is retired with the remote-read mechanism).
******************************************************************************/

#ifndef EXECUTION_DRIVEN_PIPELINE_KERNEL_CONTRACT_HH
#define EXECUTION_DRIVEN_PIPELINE_KERNEL_CONTRACT_HH

#include <cstdint>
#include <memory>
#include <string>

#include <json/json.hpp>

namespace AstraSim {
namespace ExecutionDriven {

/// Run-scoped prediction/runtime ABI contract carried by every GraphBatch.
/// The prediction config itself remains owned by Python; this object freezes
/// the parsed values and canonical-config digest shared by every C++ consumer.
class PipelineKernelContract final {
  public:
    static constexpr uint32_t kProjectionAbiVersion = 1;
    static constexpr uint32_t kDescriptorVersion = 1;
    static constexpr uint32_t kPipelineProtocolVersion = 1;
    static constexpr const char* kModelVersion =
        "v7-three-actions-20261014";
    static constexpr const char* kTimerWhitelistVersion =
        "timer-whitelist-v1-20261005";
    static constexpr uint32_t kStableWindowCount = 8;
    static constexpr uint64_t kAbsolutePrefixBudgetTicks = 1;
    static constexpr const char* kAverageRemainderRule =
        "prefix_window_quotient_remainder_earliest_ticks";

    static std::shared_ptr<const PipelineKernelContract> parse(
        const nlohmann::json& wire);

    /// Recheck all invariants. Throws std::invalid_argument on wire/runtime
    /// contract mismatch.
    void validate() const;

    uint32_t projection_abi_version() const { return projection_abi_version_; }
    uint32_t descriptor_version() const { return descriptor_version_; }
    uint32_t pipeline_protocol_version() const {
        return pipeline_protocol_version_;
    }
    const std::string& model_version() const { return model_version_; }
    const std::string& timer_whitelist_version() const {
        return timer_whitelist_version_;
    }
    uint32_t window_tile_limit() const { return window_tile_limit_; }
    uint32_t window_local_error_ppm() const {
        return window_local_error_ppm_;
    }
    uint32_t window_refine_margin_ppm() const {
        return window_refine_margin_ppm_;
    }
    const std::string& prediction_config_sha256() const {
        return prediction_config_sha256_;
    }
    static constexpr uint32_t stable_window_count() {
        return kStableWindowCount;
    }
    static constexpr uint64_t absolute_prefix_budget_ticks() {
        return kAbsolutePrefixBudgetTicks;
    }
    static constexpr const char* average_remainder_rule() {
        return kAverageRemainderRule;
    }

    bool operator==(const PipelineKernelContract& other) const;
    bool operator!=(const PipelineKernelContract& other) const {
        return !(*this == other);
    }

  private:
    PipelineKernelContract(uint32_t projection_abi_version,
                           uint32_t descriptor_version,
                           uint32_t pipeline_protocol_version,
                           std::string model_version,
                           std::string timer_whitelist_version,
                           uint32_t window_tile_limit,
                           uint32_t window_local_error_ppm,
                           uint32_t window_refine_margin_ppm,
                           std::string prediction_config_sha256);

    uint32_t projection_abi_version_;
    uint32_t descriptor_version_;
    uint32_t pipeline_protocol_version_;
    std::string model_version_;
    std::string timer_whitelist_version_;
    uint32_t window_tile_limit_;
    uint32_t window_local_error_ppm_;
    uint32_t window_refine_margin_ppm_;
    std::string prediction_config_sha256_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_PIPELINE_KERNEL_CONTRACT_HH
