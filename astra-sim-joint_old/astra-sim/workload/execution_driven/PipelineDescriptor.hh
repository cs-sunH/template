/******************************************************************************
Joint remote-read pipeline descriptor (wire v1).

The parsed descriptor is immutable. Runtime progress belongs to a separate
PipelineCursor and must never be written back into this value.
******************************************************************************/

#ifndef EXECUTION_DRIVEN_PIPELINE_DESCRIPTOR_HH
#define EXECUTION_DRIVEN_PIPELINE_DESCRIPTOR_HH

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <json/json.hpp>

namespace AstraSim {
namespace ExecutionDriven {

enum class PipelineStage { Prefill, Decode };
enum class PipelineStreamKind { PrefillReadWindow, DecodeRemoteCredit };
enum class PipelineTileUnit { SourceTokens, StreamBytes };
enum class PipelineAnchorRole { Start, Frontier, Service, Settle };
enum class PipelineAnchorBinding {
    PhysicalEndpointSlot,
    DescriptorCursorSelector
};
enum class PipelineSelectorKind {
    StageStart,
    TileEndpointLeg,
    TileEndpointTerminal,
    InternalMarkerDone,
    ExistingServiceGate,
    AllObligationsSettled
};
enum class PipelineEndpointLeg {
    DataSend,
    DataRecv,
    FillAckSend,
    FillAckRecv,
    ReleaseAckSend,
    ReleaseAckRecv,
    BodyMarker
};
enum class PipelineEndpointDependencyKind {
    OrdinaryNodeTerminal,
    VirtualEndpointTerminal
};

/// Run-scoped prediction/runtime ABI contract carried by every GraphBatch.
/// The prediction config itself remains owned by Python; this object freezes
/// the parsed values and canonical-config digest shared by every C++ consumer.
class PipelineKernelContract final {
  public:
    static constexpr uint32_t kProjectionAbiVersion = 1;
    static constexpr uint32_t kDescriptorVersion = 1;
    static constexpr uint32_t kPipelineProtocolVersion = 1;
    static constexpr const char* kModelVersion =
        "v6-compact-window-20261005";
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

struct PipelineIdentity final {
    std::string request_id;
    PipelineStage stage = PipelineStage::Prefill;
    uint64_t generation = 0;
    uint64_t stage_epoch = 0;
    uint64_t segment_ordinal = 0;
    int rank = 0;
    uint64_t staging_allocation_id = 0;
    std::string segment_id;
    std::string buffer_domain_id;
};

struct PipelineTileAxis final {
    PipelineTileUnit unit = PipelineTileUnit::SourceTokens;
    uint64_t total_units = 0;
    uint64_t units_per_tile = 0;
    uint64_t tile_count = 0;
    uint64_t tail_units = 0;
    uint64_t global_tile_first = 0;  // one-based
};

struct PipelineShardRule final {
    uint64_t shard_ordinal = 0;
    int source_rank = 0;
    int target_rank = 0;
    uint64_t total_data_bytes = 0;
    uint64_t bytes_per_unit = 0;
    std::vector<int> noc_path;
};

struct PipelineGroup final {
    uint64_t group_ordinal = 0;
    uint64_t pass_or_window_ordinal = 0;
    uint64_t source_ordinal = 0;
    std::string source_key;
    uint64_t layer_start = 0;
    uint64_t layer_end = 0;
    PipelineTileAxis tile_axis;
    std::vector<PipelineShardRule> shards;  // original builder order
};

struct PipelineProtocol final {
    uint32_t buffer_slots = 2;
    uint32_t release_lag_tiles = 2;
    uint32_t initial_tiles_without_release = 2;
    uint64_t fill_ack_bytes = 1;
    uint64_t release_ack_bytes = 1;
    bool hbm_charge = true;
};

struct PipelineDecodePredictionProfile final {
    std::string profile_id;
    double compute_ns_total = 0.0;
    uint64_t hbm_bytes_total = 0;
    std::string resource_key;
};

struct PipelinePredictionProfile final {
    PipelineDecodePredictionProfile decode_consume;
};

struct PipelineExecutionProfile final {
    // Execution uses the original COMM/endpoint HBM path and the original
    // body nodes; it does not consume projected profile durations.
    bool ordinary_comm_send_recv = true;
    bool charged_endpoint_hbm = true;
};

struct PipelineLogicalIdRange final {
    int rank = 0;
    uint64_t first_json_id = 0;
    uint64_t count = 0;
    uint64_t stride = 1;
};

struct PipelineTagAllocationRun final {
    std::string allocator_scope_id;
    uint64_t emission_ordinal = 0;
    uint32_t first_tag = 0;
    uint32_t count = 0;
    uint64_t group_ordinal = 0;
    uint64_t first_tile = 0;   // one-based within group
    uint64_t tile_count = 0;
    uint64_t first_shard = 0;
    uint64_t shard_count = 0;
};

struct PipelineEndpointLegRun final {
    PipelineEndpointLeg endpoint = PipelineEndpointLeg::DataSend;
    uint64_t json_offset = 0;
    bool has_tag_fields = false;
    uint64_t tag_emission_ordinal = 0;
    uint32_t tag_offset = 0;
    uint64_t tag_tile_stride = 0;
    uint64_t tag_shard_stride = 0;
};

struct PipelineLegacyContext final {
    std::string request_id;
    /// Original ordinary-node stage spelling: prefill, decode, or completion.
    std::string stage;
    uint64_t generation = 0;
};

/// Affine mapping for an ordered endpoint-role range. Tile ordinals are
/// one-based within the group; shard_ordinals preserves the original shard
/// order and the formulas use its local index, not the numeric shard id.
struct PipelineEndpointRun final {
    int rank = 0;
    uint64_t group_ordinal = 0;
    uint64_t first_tile = 1;
    uint64_t tile_count = 0;
    std::vector<uint64_t> shard_ordinals;
    uint64_t first_json_id = 0;
    uint64_t tile_stride = 0;
    uint64_t shard_stride = 0;
    bool has_initial_parent_json_id = false;
    uint64_t initial_parent_json_id = 0;
    PipelineLegacyContext legacy_context;
    std::vector<PipelineEndpointLegRun> legs;
};

/// Complete compact parent-edge pattern for endpoint nodes. Dependency
/// IDs use the local shard index in builder order, not numeric shard ID.
struct PipelineEndpointDependencyRun final {
    int rank = 0;
    uint64_t group_ordinal = 0;
    uint64_t first_tile = 1;
    uint64_t tile_count = 0;
    std::vector<uint64_t> shard_ordinals;
    PipelineEndpointLeg endpoint = PipelineEndpointLeg::DataSend;
    PipelineEndpointDependencyKind dependency_kind =
        PipelineEndpointDependencyKind::OrdinaryNodeTerminal;
    uint64_t first_dependency_json_id = 0;
    uint64_t dependency_tile_stride = 0;
    uint64_t dependency_shard_stride = 0;
};

/// Compact original consumption-completion terminal mapping used by the
/// two-slot release cursor. The ID formula matches endpoint dependencies.
struct PipelineConsumeCompletionRun final {
    int rank = 0;
    uint64_t group_ordinal = 0;
    uint64_t first_tile = 1;
    uint64_t tile_count = 0;
    std::vector<uint64_t> shard_ordinals;
    PipelineEndpointDependencyKind dependency_kind =
        PipelineEndpointDependencyKind::OrdinaryNodeTerminal;
    uint64_t first_dependency_json_id = 0;
    uint64_t dependency_tile_stride = 0;
    uint64_t dependency_shard_stride = 0;
};

/// Compact service-side workload projection. Each chain describes two affine
/// leg-ID sequences; it never allocates a per-tile array.
struct PipelineProjectionShare final {
    std::string rule;
    uint64_t total_units = 0;
    uint64_t total_tiles = 0;
    uint64_t share_index_first = 0;
    uint64_t tile_bytes = 0;
};

struct PipelineProjectionFill final {
    std::string leg_id_prefix;
    uint64_t first_index = 0;
    uint64_t fill_order_first = 0;
    uint64_t fill_order_stride = 1;
    bool has_completion_order_first = false;
    int64_t completion_order_first = 0;
    int64_t completion_order_stride = 0;
    std::vector<std::string> route;
    uint64_t bytes_per_tile = 0;
    uint64_t latency_ns = 0;
    std::vector<std::string> start_deps;
    std::string gate_id;
    uint32_t gate_capacity = 0;
};

struct PipelineProjectionConsume final {
    std::string leg_id_prefix;
    uint64_t first_index = 0;
    uint64_t consume_order_first = 0;
    uint64_t consume_order_stride = 1;
    bool has_completion_order_first = false;
    int64_t completion_order_first = 0;
    int64_t completion_order_stride = 0;
    std::string kind;
    std::vector<std::string> route;
    bool has_group = false;
    std::string group;
    uint64_t latency_first_ns = 0;
    uint64_t latency_rest_ns = 0;
    std::vector<std::string> consume_deps;
    std::vector<std::string> consume_start_deps;
    std::string gate_id;
    bool has_compute_share = false;
    PipelineProjectionShare compute_share;
    bool has_hbm_share = false;
    PipelineProjectionShare hbm_share;
};

struct PipelineProjectionChain final {
    uint64_t group_ordinal = 0;
    int rank = 0;
    uint64_t tile_count = 0;
    PipelineProjectionFill fill;
    PipelineProjectionConsume consume;
};

struct PipelineProjection final {
    uint32_t version = 1;
    std::vector<PipelineProjectionChain> chains;
};

struct PipelineAnchorSelector final {
    PipelineSelectorKind kind = PipelineSelectorKind::StageStart;
    uint64_t group_ordinal = 0;
    uint64_t tile_ordinal = 0;
    uint64_t shard_ordinal = 0;
    PipelineEndpointLeg endpoint = PipelineEndpointLeg::DataSend;
    uint64_t tag_allocation_emission_ordinal = 0;
    uint32_t tag_offset = 0;
    std::string existing_gate;
};

struct PipelineAnchor final {
    PipelineAnchorRole role = PipelineAnchorRole::Start;
    int rank = 0;
    PipelineAnchorBinding binding =
        PipelineAnchorBinding::DescriptorCursorSelector;
    bool has_node_id = false;
    uint64_t node_id = 0;
    PipelineAnchorSelector selector;
};

struct PipelineShardTile final {
    uint64_t shard_ordinal = 0;
    int source_rank = 0;
    int target_rank = 0;
    uint64_t data_bytes = 0;
    bool emit_endpoint = false;
};

struct PipelineTile final {
    uint64_t global_tile_ordinal = 0;  // one-based
    uint64_t units = 0;
    uint64_t data_bytes_total = 0;
    std::vector<PipelineShardTile> shards;
};

class PipelineDescriptor final {
  public:
    /// Strict parser. Rejects unknown keys, unknown versions, overflow,
    /// algebra inconsistencies, and invalid anchor/tag/ID references.
    static std::shared_ptr<const PipelineDescriptor> parse(
        const nlohmann::json& wire);

    const PipelineIdentity& identity() const { return identity_; }
    PipelineStreamKind stream_kind() const { return stream_kind_; }
    const std::vector<PipelineGroup>& groups() const { return groups_; }
    const PipelineProtocol& protocol() const { return protocol_; }
    const PipelinePredictionProfile& prediction_profile() const {
        return prediction_profile_;
    }
    const PipelineExecutionProfile& execution_profile() const {
        return execution_profile_;
    }
    const std::vector<PipelineLogicalIdRange>& logical_id_ranges() const {
        return logical_id_ranges_;
    }
    const std::vector<PipelineTagAllocationRun>& tag_allocation_runs() const {
        return tag_allocation_runs_;
    }
    const std::vector<PipelineEndpointRun>& endpoint_runs() const {
        return endpoint_runs_;
    }
    const std::vector<PipelineEndpointDependencyRun>& endpoint_dependency_runs() const {
        return endpoint_dependency_runs_;
    }
    const std::vector<PipelineConsumeCompletionRun>& consume_completion_runs() const {
        return consume_completion_runs_;
    }
    const PipelineProjection& projection() const { return projection_; }
    const std::vector<PipelineAnchor>& anchors() const { return anchors_; }
    /// Strictly parsed original semantic wire retained for idempotent
    /// cross-batch identity comparison (JSON object key order is immaterial).
    const nlohmann::json& wire() const { return wire_; }
    bool operator==(const PipelineDescriptor& other) const {
        return wire_ == other.wire_;
    }
    bool operator!=(const PipelineDescriptor& other) const {
        return !(*this == other);
    }

    uint64_t endpoint_json_id(const PipelineEndpointRun& run,
                              const PipelineEndpointLegRun& leg,
                              uint64_t tile_ordinal,
                              uint64_t shard_ordinal) const;
    uint32_t endpoint_tag(const PipelineEndpointRun& run,
                         const PipelineEndpointLegRun& leg,
                         uint64_t tile_ordinal,
                         uint64_t shard_ordinal) const;

  private:
    PipelineDescriptor(PipelineIdentity identity,
                       PipelineStreamKind stream_kind,
                       std::vector<PipelineGroup> groups,
                       PipelineProtocol protocol,
                       PipelinePredictionProfile prediction_profile,
                       PipelineExecutionProfile execution_profile,
                       std::vector<PipelineLogicalIdRange> logical_id_ranges,
                       std::vector<PipelineTagAllocationRun> tag_runs,
                       std::vector<PipelineEndpointRun> endpoint_runs,
                       std::vector<PipelineEndpointDependencyRun> endpoint_dependency_runs,
                       std::vector<PipelineConsumeCompletionRun> consume_completion_runs,
                       PipelineProjection projection,
                       std::vector<PipelineAnchor> anchors,
                       nlohmann::json wire);

    PipelineIdentity identity_;
    PipelineStreamKind stream_kind_;
    std::vector<PipelineGroup> groups_;
    PipelineProtocol protocol_;
    PipelinePredictionProfile prediction_profile_;
    PipelineExecutionProfile execution_profile_;
    std::vector<PipelineLogicalIdRange> logical_id_ranges_;
    std::vector<PipelineTagAllocationRun> tag_allocation_runs_;
    std::vector<PipelineEndpointRun> endpoint_runs_;
    std::vector<PipelineEndpointDependencyRun> endpoint_dependency_runs_;
    std::vector<PipelineConsumeCompletionRun> consume_completion_runs_;
    PipelineProjection projection_;
    std::vector<PipelineAnchor> anchors_;
    nlohmann::json wire_;
};

}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_PIPELINE_DESCRIPTOR_HH
