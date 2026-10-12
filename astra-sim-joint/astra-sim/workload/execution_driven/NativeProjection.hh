/******************************************************************************
Native projection API shared by the C++ simulator and joint_projection C ABI.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_NATIVE_PROJECTION_HH
#define EXECUTION_DRIVEN_NATIVE_PROJECTION_HH

#include <cstdint>
#include <list>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>

#include <json/json.hpp>

namespace AstraSim {
namespace ExecutionDriven {

struct NativeProjectionInput;
class PipelineKernelContract;

class NativeProjectionContext final {
  public:
    static std::shared_ptr<const NativeProjectionContext> parse(
        const nlohmann::json& snapshot_envelope);

    const NativeProjectionInput& input() const { return *input_; }
    const PipelineKernelContract& contract() const {
        if (contract_ == nullptr) {
            throw std::logic_error(
                "NativeProjectionContext has no frozen kernel_contract");
        }
        return *contract_;
    }
    const PipelineKernelContract* contract_ptr() const {
        return contract_.get();
    }
    bool has_kernel_contract() const { return contract_ != nullptr; }
    const nlohmann::json& kernel_contract() const { return kernel_contract_; }

    /// Background-reference memoization over the immutable snapshot.
    ///
    /// The reference world of a project() call is a pure function of the
    /// context input plus the reference-relevant parts of the per-call
    /// private overlay and the run options.  Within one context (one
    /// scheduling decision) candidates that share those inputs execute the
    /// reference world once; later calls reuse the recorded completion
    /// times.  This is a memo table of a deterministic function, not shared
    /// mutable candidate state: entries are only published after a run
    /// completed (any status), lookups never observe partial results, and
    /// failed builds publish nothing.  Bounded LRU; the table dies with the
    /// context.  Single-threaded per context (the production ABI is called
    /// from one scheduler thread).
    struct ReferenceRunResult final {
        std::string status;
        // work_id -> completion ns (nullopt = not finished within budget).
        std::map<std::string, std::optional<uint64_t>> completion_by_work;
    };
    const ReferenceRunResult* lookup_reference_run(uint64_t key) const;
    const ReferenceRunResult* store_reference_run(
        uint64_t key, ReferenceRunResult result) const;
    uint64_t reference_cache_hits() const { return reference_cache_hits_; }
    uint64_t reference_cache_misses() const {
        return reference_cache_misses_;
    }

  private:
    NativeProjectionContext(std::shared_ptr<const NativeProjectionInput> input,
                            std::shared_ptr<const PipelineKernelContract> contract,
                            nlohmann::json kernel_contract);
    std::shared_ptr<const NativeProjectionInput> input_;
    std::shared_ptr<const PipelineKernelContract> contract_;
    nlohmann::json kernel_contract_;

    static constexpr size_t kReferenceCacheCapacity = 32;
    mutable std::list<std::pair<uint64_t, ReferenceRunResult>>
        reference_cache_;
    mutable std::map<uint64_t,
        std::list<std::pair<uint64_t, ReferenceRunResult>>::iterator>
        reference_cache_index_;
    mutable uint64_t reference_cache_hits_ = 0;
    mutable uint64_t reference_cache_misses_ = 0;
};

/// Mutable, stateful view of the frozen snapshot. This is the native backing
/// for the legacy ProjectionWorld constructor/step/run facade; it is separate
/// from query calls, which always construct isolated worlds.
class NativeProjectionWorldSession final {
  public:
    explicit NativeProjectionWorldSession(
        std::shared_ptr<const NativeProjectionContext> context);
    ~NativeProjectionWorldSession();
    NativeProjectionWorldSession(NativeProjectionWorldSession&&) noexcept;
    NativeProjectionWorldSession& operator=(NativeProjectionWorldSession&&) noexcept;
    NativeProjectionWorldSession(const NativeProjectionWorldSession&) = delete;
    NativeProjectionWorldSession& operator=(const NativeProjectionWorldSession&) = delete;

    nlohmann::json operation(const nlohmann::json& request);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

/// Immutable per-call controls. Precision parameters are bounded by the
/// run-scoped PipelineKernelContract; effective_window_tile_limit may tighten
/// a particular query but can never alter the frozen contract.
struct NativeProjectionOptions final {
    uint64_t projection_event_budget = 262144;
    uint64_t closure_resource_budget = 512;
    bool has_horizon = false;
    double horizon_ns = 0.0;
    bool include_background_reference = true;
    bool analytic_segments = true;
    bool drain_background = true;
    uint32_t effective_window_tile_limit = 0;  // zero selects run maximum

    static NativeProjectionOptions parse(
        const nlohmann::json& wire,
        const nlohmann::json& kernel_contract);
};

/// Projection context is immutable. Each call builds a fresh mutable world;
/// candidates therefore cannot contaminate later candidates from this context.
class NativeProjection final {
  public:
    static nlohmann::json project(
        const NativeProjectionContext& context,
        const nlohmann::json& candidate_work_items,
        const NativeProjectionOptions& options);
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_NATIVE_PROJECTION_HH
