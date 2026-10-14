// PipelineDescriptor.hh -- native port of joint/pipeline_descriptor.py.
//
// Immutable, fully validated descriptor with the same field surface as the
// Python dataclasses.  from_wire reproduces every geometry/identity check of
// PipelineDescriptor.from_wire; to_wire reproduces the canonical wire form
// (used for immutable-descriptor redeclaration comparison).  Selected-tile
// algebra (endpoint_id / selected_shard_tile / tile facts) matches the lazy
// Python methods exactly.
#ifndef JOINT_LEDGER_PIPELINE_DESCRIPTOR_HH
#define JOINT_LEDGER_PIPELINE_DESCRIPTOR_HH

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "LedgerTypes.hh"

namespace jointledger {

struct PipelineIdentity {
    std::string request_id;
    std::string stage;               // "prefill" | "decode"
    std::uint64_t generation = 0;
    std::uint64_t stage_epoch = 0;
    std::uint64_t segment_ordinal = 0;
    std::uint64_t rank = 0;
    std::uint64_t staging_allocation_id = 0;
    std::string segment_id;
    std::string buffer_domain_id;
};

struct PipelineTileAxis {
    std::string unit;                // "source_tokens" | "stream_bytes"
    std::uint64_t total_units = 0;
    std::uint64_t units_per_tile = 0;
    std::uint64_t tile_count = 0;
    std::uint64_t tail_units = 0;
    std::uint64_t global_tile_first = 0;
};

struct PipelineShardRule {
    std::uint64_t shard_ordinal = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t target_rank = 0;
    std::uint64_t total_data_bytes = 0;
    std::uint64_t bytes_per_unit = 0;
    std::vector<std::uint64_t> noc_path;
};

struct PipelineGroup {
    std::uint64_t group_ordinal = 0;
    std::uint64_t pass_or_window_ordinal = 0;
    std::uint64_t source_ordinal = 0;
    std::string source_key;
    std::uint64_t layer_start = 0;
    std::uint64_t layer_end = 0;
    PipelineTileAxis tile_axis;
    std::vector<PipelineShardRule> shards;
};

struct PipelineProtocol {
    std::uint64_t buffer_slots = 2;
    std::uint64_t release_lag_tiles = 2;
    std::uint64_t initial_tiles_without_release = 2;
    std::uint64_t fill_ack_bytes = 1;
    std::uint64_t release_ack_bytes = 1;
    bool hbm_charge = true;
    std::string target_rank_order;
    std::string source_rank_order;
    std::string tile_branching;
};

struct PipelineLegacyContext {
    std::string request_id;
    std::string stage;
    std::uint64_t generation = 0;
};

struct PipelineEndpointLegRun {
    std::string endpoint;
    std::uint64_t json_offset = 0;
    std::optional<std::uint64_t> tag_emission_ordinal;
    std::optional<std::uint64_t> tag_offset;
    std::optional<std::uint64_t> tag_tile_stride;
    std::optional<std::uint64_t> tag_shard_stride;
};

struct PipelineEndpointRun {
    std::uint64_t rank = 0;
    std::uint64_t group_ordinal = 0;
    std::uint64_t first_tile = 0;
    std::uint64_t tile_count = 0;
    std::vector<std::uint64_t> shard_ordinals;
    std::uint64_t first_json_id = 0;
    std::uint64_t tile_stride = 0;
    std::uint64_t shard_stride = 0;
    std::optional<std::uint64_t> initial_parent_json_id;
    PipelineLegacyContext legacy_context;
    std::vector<PipelineEndpointLegRun> legs;
};

struct PipelineConsumeCompletionRun {
    std::uint64_t rank = 0;
    std::uint64_t group_ordinal = 0;
    std::uint64_t first_tile = 0;
    std::uint64_t tile_count = 0;
    std::vector<std::uint64_t> shard_ordinals;
    std::string dependency_kind;
    std::uint64_t first_dependency_json_id = 0;
    std::uint64_t dependency_tile_stride = 0;
    std::uint64_t dependency_shard_stride = 0;
};

struct PipelineDescriptor;
using DescriptorPtr = std::shared_ptr<const PipelineDescriptor>;

struct PipelineShardTile {
    std::uint64_t shard_ordinal = 0;
    std::uint64_t source_rank = 0;
    std::uint64_t target_rank = 0;
    std::uint64_t data_bytes = 0;
    bool emit_endpoint = false;
};

struct PipelineDescriptor {
    PipelineIdentity identity;
    std::string stream_kind;      // prefill_read_window | decode_remote_credit
    std::vector<PipelineGroup> groups;
    PipelineProtocol protocol;
    std::vector<PipelineEndpointRun> endpoint_runs;
    std::vector<PipelineConsumeCompletionRun> consume_completion_runs;
    std::uint64_t descriptor_version = 1;

    // Port-unit additions (joint/pipeline_descriptor.py port, N2): canonical
    // wire sections for descriptor areas the ledger consumers do not read as
    // typed structs.  from_wire fully validates these sections (including all
    // cross-validation against groups/tag runs/endpoint runs) and stores the
    // canonical form; to_wire() emits them verbatim, which reproduces the
    // Python to_wire() shape key for key.  Newly added fields only -- the
    // frozen consumer surface above is unchanged.
    Json prediction_profile_wire;
    Json execution_profile_wire;
    Json logical_id_ranges_wire;
    Json tag_allocation_runs_wire;
    Json endpoint_dependency_runs_wire;
    Json projection_wire;
    Json anchors_wire;

    // ---- queries (1:1 with the Python lazy tile algebra) ----
    // endpoint_id(run_index, tile_ordinal, shard_ordinal, leg_index):
    // first_json_id + json_offset + affine offsets; throws on bad geometry.
    std::uint64_t endpoint_id(std::size_t run_index, std::uint64_t tile_ordinal,
                              std::uint64_t shard_ordinal,
                              std::size_t leg_index) const;
    // selected_shard_tile(group_ordinal, tile_ordinal (0-based input exactly
    // like Python callers pass tile-1), shard_ordinal) ->
    // (global_tile_ordinal, PipelineShardTile).  Throws on range errors.
    std::pair<std::uint64_t, PipelineShardTile> selected_shard_tile(
        std::uint64_t group_ordinal, std::uint64_t tile_ordinal,
        std::uint64_t shard_ordinal) const;
    const PipelineGroup* group_by_ordinal(std::uint64_t group_ordinal) const;

    // Canonical wire form (pipeline_descriptor.to_wire) -- identity equality
    // and immutable-redeclaration comparison use this.
    Json to_wire() const;

    // ---- construction ----
    // Full validation; throws LedgerError on any violation (the Python
    // PipelineDescriptorError surface folds into LedgerError here).
    static DescriptorPtr from_wire(const Json& wire);
};

// Python dict equality of to_wire() forms (ordered comparison by keys).
bool descriptor_wire_equal(const PipelineDescriptor& a,
                           const PipelineDescriptor& b);

}  // namespace jointledger

#endif  // JOINT_LEDGER_PIPELINE_DESCRIPTOR_HH
