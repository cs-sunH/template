/******************************************************************************
Joint remote-read pipeline descriptor wire parser.
******************************************************************************/

#include "astra-sim/workload/execution_driven/PipelineDescriptor.hh"

#include <algorithm>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <utility>

namespace AstraSim {
namespace ExecutionDriven {
namespace {

using Json = nlohmann::json;

[[noreturn]] void fail(const std::string& path, const std::string& reason) {
    throw std::invalid_argument("pipeline descriptor " + path + ": " + reason);
}

void require_object(const Json& value, const std::string& path) {
    if (!value.is_object()) fail(path, "expected object");
}

void require_exact_keys(const Json& value, const std::string& path,
                        const std::set<std::string>& required,
                        const std::set<std::string>& optional = {}) {
    require_object(value, path);
    for (const auto& key : required) {
        if (value.find(key) == value.end()) fail(path, "missing key '" + key + "'");
    }
    for (auto it = value.begin(); it != value.end(); ++it) {
        if (required.count(it.key()) == 0 && optional.count(it.key()) == 0) {
            fail(path, "unknown key '" + it.key() + "'");
        }
    }
}

uint64_t get_u64(const Json& value, const std::string& path) {
    if (value.is_number_unsigned()) return value.get<uint64_t>();
    if (!value.is_number_integer()) fail(path, "expected unsigned integer");
    const int64_t signed_value = value.get<int64_t>();
    if (signed_value < 0) fail(path, "must be nonnegative");
    return static_cast<uint64_t>(signed_value);
}

uint64_t get_u64_key(const Json& object, const char* key,
                     const std::string& path) {
    return get_u64(object.at(key), path + "." + key);
}

uint64_t get_optional_u64_nullable(const Json& object, const char* key,
                                   const std::string& path, bool* present) {
    const auto& value = object.at(key);
    if (value.is_null()) {
        *present = false;
        return 0;
    }
    *present = true;
    return get_u64(value, path + "." + key);
}

uint32_t get_u32(const Json& value, const std::string& path) {
    const uint64_t parsed = get_u64(value, path);
    if (parsed > std::numeric_limits<uint32_t>::max()) {
        fail(path, "exceeds uint32");
    }
    return static_cast<uint32_t>(parsed);
}

int get_rank(const Json& value, const std::string& path) {
    const uint64_t parsed = get_u64(value, path);
    if (parsed > static_cast<uint64_t>(std::numeric_limits<int>::max())) {
        fail(path, "exceeds rank domain");
    }
    return static_cast<int>(parsed);
}

std::string get_string(const Json& value, const std::string& path) {
    if (!value.is_string()) fail(path, "expected string");
    return value.get<std::string>();
}

std::string get_nonempty_string(const Json& value, const std::string& path) {
    auto parsed = get_string(value, path);
    if (parsed.empty()) fail(path, "must not be empty");
    return parsed;
}

double get_finite_nonnegative(const Json& value, const std::string& path) {
    if (!value.is_number()) fail(path, "expected number");
    const double parsed = value.get<double>();
    if (!std::isfinite(parsed) || parsed < 0.0) {
        fail(path, "must be finite and nonnegative");
    }
    return parsed;
}

bool get_bool(const Json& value, const std::string& path) {
    if (!value.is_boolean()) fail(path, "expected boolean");
    return value.get<bool>();
}

const Json& required(const Json& value, const char* key,
                     const std::string& path) {
    const auto it = value.find(key);
    if (it == value.end()) fail(path, "missing key '" + std::string(key) + "'");
    return *it;
}

void require_array(const Json& value, const std::string& path,
                   bool nonempty = false) {
    if (!value.is_array()) fail(path, "expected array");
    if (nonempty && value.empty()) fail(path, "must not be empty");
}

bool checked_add(uint64_t a, uint64_t b, uint64_t* out) {
    if (b > std::numeric_limits<uint64_t>::max() - a) return false;
    *out = a + b;
    return true;
}

bool checked_mul(uint64_t a, uint64_t b, uint64_t* out) {
    if (a != 0 && b > std::numeric_limits<uint64_t>::max() / a) return false;
    *out = a * b;
    return true;
}

int64_t get_i64(const Json& value, const std::string& path) {
    if (value.is_number_unsigned()) {
        const uint64_t parsed = value.get<uint64_t>();
        if (parsed > static_cast<uint64_t>(std::numeric_limits<int64_t>::max())) {
            fail(path, "exceeds int64");
        }
        return static_cast<int64_t>(parsed);
    }
    if (!value.is_number_integer()) fail(path, "expected signed int64");
    return value.get<int64_t>();
}

std::vector<std::string> parse_nonempty_string_array(
    const Json& value, const std::string& path, bool require_nonempty = false) {
    require_array(value, path, require_nonempty);
    std::vector<std::string> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        result.push_back(get_string(value[i], path + "[" + std::to_string(i) + "]"));
    }
    return result;
}

template <typename Enum>
Enum parse_enum(const Json& value, const std::string& path,
                std::initializer_list<std::pair<std::string_view, Enum>> values) {
    const auto name = get_string(value, path);
    for (const auto& item : values) {
        if (name == item.first) return item.second;
    }
    fail(path, "unknown value '" + name + "'");
}

PipelineStage parse_stage(const Json& value, const std::string& path) {
    return parse_enum<PipelineStage>(
        value, path, {{"prefill", PipelineStage::Prefill},
                      {"decode", PipelineStage::Decode}});
}

PipelineStreamKind parse_stream_kind(const Json& value,
                                     const std::string& path) {
    return parse_enum<PipelineStreamKind>(
        value, path, {{"prefill_read_window", PipelineStreamKind::PrefillReadWindow},
                      {"decode_remote_credit", PipelineStreamKind::DecodeRemoteCredit}});
}

PipelineTileUnit parse_tile_unit(const Json& value, const std::string& path) {
    return parse_enum<PipelineTileUnit>(
        value, path, {{"source_tokens", PipelineTileUnit::SourceTokens},
                      {"stream_bytes", PipelineTileUnit::StreamBytes}});
}

PipelineAnchorRole parse_anchor_role(const Json& value,
                                     const std::string& path) {
    return parse_enum<PipelineAnchorRole>(
        value, path, {{"start", PipelineAnchorRole::Start},
                      {"frontier", PipelineAnchorRole::Frontier},
                      {"service", PipelineAnchorRole::Service},
                      {"settle", PipelineAnchorRole::Settle}});
}

PipelineAnchorBinding parse_anchor_binding(const Json& value,
                                           const std::string& path) {
    return parse_enum<PipelineAnchorBinding>(
        value, path,
        {{"physical_endpoint_slot", PipelineAnchorBinding::PhysicalEndpointSlot},
         {"descriptor_cursor_selector", PipelineAnchorBinding::DescriptorCursorSelector}});
}

PipelineSelectorKind parse_selector_kind(const Json& value,
                                         const std::string& path) {
    return parse_enum<PipelineSelectorKind>(
        value, path,
        {{"stage_start", PipelineSelectorKind::StageStart},
         {"tile_endpoint_leg", PipelineSelectorKind::TileEndpointLeg},
         {"tile_endpoint_terminal", PipelineSelectorKind::TileEndpointTerminal},
         {"internal_marker_done", PipelineSelectorKind::InternalMarkerDone},
         {"existing_service_gate", PipelineSelectorKind::ExistingServiceGate},
         {"all_obligations_settled", PipelineSelectorKind::AllObligationsSettled}});
}

PipelineEndpointLeg parse_endpoint(const Json& value, const std::string& path) {
    return parse_enum<PipelineEndpointLeg>(
        value, path,
        {{"data_send", PipelineEndpointLeg::DataSend},
         {"data_recv", PipelineEndpointLeg::DataRecv},
         {"fill_ack_send", PipelineEndpointLeg::FillAckSend},
         {"fill_ack_recv", PipelineEndpointLeg::FillAckRecv},
         {"release_ack_send", PipelineEndpointLeg::ReleaseAckSend},
         {"release_ack_recv", PipelineEndpointLeg::ReleaseAckRecv},
         {"body_marker", PipelineEndpointLeg::BodyMarker}});
}

PipelineEndpointDependencyKind parse_dependency_kind(
    const Json& value, const std::string& path) {
    return parse_enum<PipelineEndpointDependencyKind>(
        value, path,
        {{"ordinary_node_terminal",
          PipelineEndpointDependencyKind::OrdinaryNodeTerminal},
         {"virtual_endpoint_terminal",
          PipelineEndpointDependencyKind::VirtualEndpointTerminal}});
}

PipelineIdentity parse_identity(const Json& value) {
    const std::string path = "identity";
    require_exact_keys(value, path,
                       {"request_id", "stage", "generation", "stage_epoch",
                        "segment_ordinal", "rank", "staging_allocation_id",
                        "segment_id", "buffer_domain_id"});
    PipelineIdentity result;
    result.request_id = get_nonempty_string(value.at("request_id"), path + ".request_id");
    result.stage = parse_stage(value.at("stage"), path + ".stage");
    result.generation = get_u64_key(value, "generation", path);
    result.stage_epoch = get_u64_key(value, "stage_epoch", path);
    result.segment_ordinal = get_u64_key(value, "segment_ordinal", path);
    result.rank = get_rank(value.at("rank"), path + ".rank");
    result.staging_allocation_id =
        get_u64_key(value, "staging_allocation_id", path);
    result.segment_id = get_nonempty_string(value.at("segment_id"), path + ".segment_id");
    result.buffer_domain_id =
        get_nonempty_string(value.at("buffer_domain_id"), path + ".buffer_domain_id");
    return result;
}

PipelineLegacyContext parse_legacy_context(const Json& value,
                                           const std::string& path) {
    require_exact_keys(value, path, {"request_id", "stage", "generation"});
    PipelineLegacyContext result;
    result.request_id = get_nonempty_string(value.at("request_id"), path + ".request_id");
    result.stage = get_string(value.at("stage"), path + ".stage");
    if (result.stage != "prefill" && result.stage != "decode" &&
        result.stage != "completion") {
        fail(path + ".stage", "must preserve ordinary-node prefill/decode/completion stage");
    }
    result.generation = get_u64_key(value, "generation", path);
    const uint64_t expected_generation = result.stage == "prefill" ? 0 : 1;
    if (result.generation != expected_generation) {
        fail(path + ".generation", "does not match the original node stage generation");
    }
    return result;
}

PipelineGroup parse_group(const Json& value, size_t index) {
    const std::string path = "groups[" + std::to_string(index) + "]";
    require_exact_keys(value, path,
                       {"group_ordinal", "pass_or_window_ordinal",
                        "source_ordinal", "source_key", "layer_start",
                        "layer_end", "tile_axis", "shards"});
    PipelineGroup result;
    result.group_ordinal = get_u64_key(value, "group_ordinal", path);
    result.pass_or_window_ordinal =
        get_u64_key(value, "pass_or_window_ordinal", path);
    result.source_ordinal = get_u64_key(value, "source_ordinal", path);
    result.source_key = get_nonempty_string(value.at("source_key"), path + ".source_key");
    result.layer_start = get_u64_key(value, "layer_start", path);
    result.layer_end = get_u64_key(value, "layer_end", path);
    if (result.layer_end < result.layer_start) {
        fail(path + ".layer_end", "must be >= layer_start");
    }

    const auto& axis = value.at("tile_axis");
    const std::string axis_path = path + ".tile_axis";
    require_exact_keys(axis, axis_path,
                       {"unit", "total_units", "units_per_tile", "tile_count",
                        "tail_units", "global_tile_first"});
    result.tile_axis.unit = parse_tile_unit(axis.at("unit"), axis_path + ".unit");
    result.tile_axis.total_units = get_u64_key(axis, "total_units", axis_path);
    result.tile_axis.units_per_tile = get_u64_key(axis, "units_per_tile", axis_path);
    result.tile_axis.tile_count = get_u64_key(axis, "tile_count", axis_path);
    result.tile_axis.tail_units = get_u64_key(axis, "tail_units", axis_path);
    result.tile_axis.global_tile_first =
        get_u64_key(axis, "global_tile_first", axis_path);
    if (result.tile_axis.total_units == 0 || result.tile_axis.units_per_tile == 0 ||
        result.tile_axis.tile_count == 0 || result.tile_axis.global_tile_first == 0) {
        fail(axis_path, "unit counts and global_tile_first must be positive");
    }
    const uint64_t expected_count =
        result.tile_axis.total_units / result.tile_axis.units_per_tile +
        (result.tile_axis.total_units % result.tile_axis.units_per_tile != 0 ? 1 : 0);
    if (result.tile_axis.tile_count != expected_count) {
        fail(axis_path + ".tile_count", "does not match ceil(total_units/units_per_tile)");
    }
    const uint64_t expected_tail =
        result.tile_axis.total_units -
        result.tile_axis.units_per_tile * (result.tile_axis.tile_count - 1);
    if (result.tile_axis.tail_units != expected_tail) {
        fail(axis_path + ".tail_units", "does not match the final tile size");
    }
    uint64_t tile_end = 0;
    if (!checked_add(result.tile_axis.global_tile_first,
                     result.tile_axis.tile_count - 1, &tile_end)) {
        fail(axis_path, "global tile range overflows uint64");
    }
    (void)tile_end;

    const auto& shards = value.at("shards");
    const std::string shards_path = path + ".shards";
    require_array(shards, shards_path, true);
    std::set<uint64_t> shard_ids;
    uint64_t max_shard_bytes = 0;
    for (size_t i = 0; i < shards.size(); ++i) {
        const auto& shard = shards[i];
        const std::string shard_path = shards_path + "[" + std::to_string(i) + "]";
        require_exact_keys(shard, shard_path,
                           {"shard_ordinal", "source_rank", "target_rank",
                            "total_data_bytes", "bytes_per_unit", "noc_path"});
        PipelineShardRule parsed;
        parsed.shard_ordinal = get_u64_key(shard, "shard_ordinal", shard_path);
        parsed.source_rank = get_rank(shard.at("source_rank"), shard_path + ".source_rank");
        parsed.target_rank = get_rank(shard.at("target_rank"), shard_path + ".target_rank");
        parsed.total_data_bytes = get_u64_key(shard, "total_data_bytes", shard_path);
        parsed.bytes_per_unit = get_u64_key(shard, "bytes_per_unit", shard_path);
        if (!shard_ids.insert(parsed.shard_ordinal).second) {
            fail(shard_path + ".shard_ordinal", "duplicate shard ordinal");
        }
        if (parsed.shard_ordinal != i) {
            fail(shard_path + ".shard_ordinal",
                 "must be a zero-based position in original shard order");
        }
        max_shard_bytes = std::max(max_shard_bytes, parsed.total_data_bytes);
        const auto& noc_path = shard.at("noc_path");
        require_array(noc_path, shard_path + ".noc_path");
        parsed.noc_path.reserve(noc_path.size());
        for (size_t p = 0; p < noc_path.size(); ++p) {
            parsed.noc_path.push_back(
                get_rank(noc_path[p], shard_path + ".noc_path[" + std::to_string(p) + "]"));
        }
        if (!parsed.noc_path.empty() &&
            (parsed.noc_path.front() != parsed.source_rank ||
             parsed.noc_path.back() != parsed.target_rank)) {
            fail(shard_path + ".noc_path", "must begin/end at source_rank/target_rank");
        }
        if (result.tile_axis.unit == PipelineTileUnit::SourceTokens) {
            if (parsed.bytes_per_unit == 0) {
                fail(shard_path + ".bytes_per_unit", "must be positive for source_tokens");
            }
            uint64_t expected_bytes = 0;
            if (!checked_mul(result.tile_axis.total_units, parsed.bytes_per_unit,
                             &expected_bytes) || expected_bytes != parsed.total_data_bytes) {
                fail(shard_path + ".total_data_bytes",
                     "must equal total_units * bytes_per_unit for source_tokens");
            }
        } else {
            if (parsed.bytes_per_unit != 1) {
                fail(shard_path + ".bytes_per_unit",
                     "must be 1 when tile_axis.unit is stream_bytes");
            }
            if (parsed.total_data_bytes > result.tile_axis.total_units) {
                fail(shard_path + ".total_data_bytes",
                     "exceeds stream_bytes tile axis total_units");
            }
        }
        result.shards.push_back(std::move(parsed));
    }
    if (result.tile_axis.unit == PipelineTileUnit::StreamBytes &&
        max_shard_bytes != result.tile_axis.total_units) {
        fail(axis_path + ".total_units",
             "stream_bytes axis must equal the maximum shard byte range");
    }
    return result;
}

PipelineProtocol parse_protocol(const Json& value) {
    const std::string path = "protocol";
    require_exact_keys(value, path,
                       {"buffer_slots", "release_lag_tiles",
                        "initial_tiles_without_release", "fill_ack_bytes",
                        "release_ack_bytes", "hbm_charge", "target_rank_order",
                        "source_rank_order", "tile_branching"});
    PipelineProtocol result;
    result.buffer_slots = get_u32(value.at("buffer_slots"), path + ".buffer_slots");
    result.release_lag_tiles =
        get_u32(value.at("release_lag_tiles"), path + ".release_lag_tiles");
    result.initial_tiles_without_release =
        get_u32(value.at("initial_tiles_without_release"),
                path + ".initial_tiles_without_release");
    result.fill_ack_bytes = get_u64_key(value, "fill_ack_bytes", path);
    result.release_ack_bytes = get_u64_key(value, "release_ack_bytes", path);
    result.hbm_charge = get_bool(value.at("hbm_charge"), path + ".hbm_charge");
    if (get_string(value.at("target_rank_order"), path + ".target_rank_order") !=
        "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1") {
        fail(path + ".target_rank_order", "unsupported ordering rule");
    }
    if (get_string(value.at("source_rank_order"), path + ".source_rank_order") !=
        "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i") {
        fail(path + ".source_rank_order", "unsupported ordering rule");
    }
    if (get_string(value.at("tile_branching"), path + ".tile_branching") !=
        "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated") {
        fail(path + ".tile_branching", "unsupported tile branch rule");
    }
    if (result.buffer_slots != 2 || result.release_lag_tiles != 2 ||
        result.initial_tiles_without_release != 2 || result.fill_ack_bytes != 1 ||
        result.release_ack_bytes != 1 || !result.hbm_charge) {
        fail(path, "unsupported buffer/ACK/HBM protocol parameters");
    }
    return result;
}

PipelinePredictionProfile parse_prediction_profile(const Json& value) {
    const std::string path = "prediction_profile";
    require_exact_keys(value, path, {"prefill_consume", "decode_consume"});
    if (get_string(value.at("prefill_consume"), path + ".prefill_consume") !=
        "zero_duration_no_extra_consume") {
        fail(path + ".prefill_consume", "unsupported prefill consume rule");
    }
    const auto& decode = value.at("decode_consume");
    require_exact_keys(decode, path + ".decode_consume",
                       {"kind", "profile_id", "compute_ns_total",
                        "hbm_bytes_total", "resource_key", "share_rule"});
    if (get_string(decode.at("kind"), path + ".decode_consume.kind") !=
        "profiled_consume") {
        fail(path + ".decode_consume.kind", "unsupported profile kind");
    }
    if (get_string(decode.at("share_rule"), path + ".decode_consume.share_rule") !=
        "existing_action_work_graph_byte_share_vector") {
        fail(path + ".decode_consume.share_rule", "unsupported profile share rule");
    }
    PipelinePredictionProfile result;
    result.decode_consume.profile_id = get_nonempty_string(
        decode.at("profile_id"), path + ".decode_consume.profile_id");
    result.decode_consume.compute_ns_total = get_finite_nonnegative(
        decode.at("compute_ns_total"), path + ".decode_consume.compute_ns_total");
    result.decode_consume.hbm_bytes_total =
        get_u64_key(decode, "hbm_bytes_total", path + ".decode_consume");
    result.decode_consume.resource_key = get_nonempty_string(
        decode.at("resource_key"), path + ".decode_consume.resource_key");
    return result;
}

PipelineExecutionProfile parse_execution_profile(const Json& value) {
    const std::string path = "execution_profile";
    require_exact_keys(value, path,
                       {"data_legs", "endpoint_hbm_rule", "prefill_consume",
                        "decode_consume"});
    const std::pair<const char*, const char*> expected[] = {
        {"data_legs", "ordinary_comm_send_recv_path"},
        {"endpoint_hbm_rule", "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint"},
        {"prefill_consume", "existing_body_or_marker_only"},
        {"decode_consume", "existing_body_nodes_only"}};
    for (const auto& item : expected) {
        if (get_string(value.at(item.first), path + "." + item.first) != item.second) {
            fail(path + "." + item.first, "unsupported execution rule");
        }
    }
    return PipelineExecutionProfile{};
}

struct IdProgression final {
    int rank;
    uint64_t first;
    uint64_t stride;
    uint64_t count;
    std::string label;
};

uint64_t gcd_u64(uint64_t a, uint64_t b);
uint64_t mod_inverse_u64(uint64_t a, uint64_t mod);
bool progressions_intersect(const IdProgression& a, const IdProgression& b);

std::vector<PipelineLogicalIdRange> parse_id_ranges(const Json& value) {
    const std::string path = "logical_id_ranges";
    require_array(value, path);
    std::vector<PipelineLogicalIdRange> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& item = value[i];
        const std::string item_path = path + "[" + std::to_string(i) + "]";
        require_exact_keys(item, item_path,
                           {"rank", "first_json_id", "count", "stride", "kind"});
        if (get_string(item.at("kind"), item_path + ".kind") !=
            "virtual_endpoint_hole") {
            fail(item_path + ".kind", "unsupported logical ID range kind");
        }
        PipelineLogicalIdRange range;
        range.rank = get_rank(item.at("rank"), item_path + ".rank");
        range.first_json_id = get_u64_key(item, "first_json_id", item_path);
        range.count = get_u64_key(item, "count", item_path);
        range.stride = get_u64_key(item, "stride", item_path);
        if (range.count == 0 || range.stride == 0) {
            fail(item_path, "range count and stride must be positive");
        }
        const unsigned __int128 last =
            static_cast<unsigned __int128>(range.first_json_id) +
            static_cast<unsigned __int128>(range.stride) * (range.count - 1);
        if (last > std::numeric_limits<uint64_t>::max()) {
            fail(item_path, "range exceeds JSON uint64 ID domain");
        }
        if (!result.empty()) {
            const auto& previous = result.back();
            if (range.rank < previous.rank ||
                (range.rank == previous.rank &&
                 range.first_json_id < previous.first_json_id)) {
                fail(item_path, "ranges must be ordered by rank and first_json_id");
            }
        }
        result.push_back(range);
    }
    for (size_t i = 0; i < result.size(); ++i) {
        for (size_t j = i + 1; j < result.size(); ++j) {
            if (result[j].rank != result[i].rank) break;
            if (progressions_intersect(
                    {result[i].rank, result[i].first_json_id, result[i].stride,
                     result[i].count, "logical_id_ranges[" + std::to_string(i) + "]"},
                    {result[j].rank, result[j].first_json_id, result[j].stride,
                     result[j].count, "logical_id_ranges[" + std::to_string(j) + "]"})) {
                fail(path, "virtual ID progressions overlap for rank " +
                               std::to_string(result[i].rank));
            }
        }
    }
    return result;
}

std::vector<PipelineTagAllocationRun> parse_tag_runs(const Json& value) {
    const std::string path = "tag_allocation_runs";
    require_array(value, path);
    std::vector<PipelineTagAllocationRun> result;
    result.reserve(value.size());
    std::map<std::string, uint64_t> last_emission;
    std::map<std::string, uint64_t> last_tag_end;
    std::set<uint64_t> unique_emissions;
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& item = value[i];
        const std::string item_path = path + "[" + std::to_string(i) + "]";
        require_exact_keys(item, item_path,
                           {"allocator_scope_id", "emission_ordinal", "first_tag",
                            "count", "group_ordinal", "first_tile", "tile_count",
                            "first_shard", "shard_count"});
        PipelineTagAllocationRun run;
        run.allocator_scope_id = get_nonempty_string(
            item.at("allocator_scope_id"), item_path + ".allocator_scope_id");
        run.emission_ordinal = get_u64_key(item, "emission_ordinal", item_path);
        if (!unique_emissions.insert(run.emission_ordinal).second) {
            fail(item_path + ".emission_ordinal",
                 "must be globally unique because endpoint selectors address it directly");
        }
        run.first_tag = get_u32(item.at("first_tag"), item_path + ".first_tag");
        run.count = get_u32(item.at("count"), item_path + ".count");
        run.group_ordinal = get_u64_key(item, "group_ordinal", item_path);
        run.first_tile = get_u64_key(item, "first_tile", item_path);
        run.tile_count = get_u64_key(item, "tile_count", item_path);
        run.first_shard = get_u64_key(item, "first_shard", item_path);
        run.shard_count = get_u64_key(item, "shard_count", item_path);
        if (run.first_tag == 0 || run.count == 0 || run.first_tile == 0 ||
            run.tile_count == 0 || run.shard_count == 0) {
            fail(item_path, "tag/tile/shard ranges must be positive");
        }
        const uint64_t tag_end = static_cast<uint64_t>(run.first_tag) + run.count;
        if (tag_end > static_cast<uint64_t>(std::numeric_limits<uint32_t>::max()) + 1) {
            fail(item_path, "tag range exceeds uint32 allocator domain");
        }
        uint64_t tile_end = 0;
        uint64_t shard_end = 0;
        if (!checked_add(run.first_tile, run.tile_count, &tile_end) ||
            !checked_add(run.first_shard, run.shard_count, &shard_end)) {
            fail(item_path, "tile/shard range overflows uint64");
        }
        (void)tile_end;
        (void)shard_end;
        auto emission_it = last_emission.find(run.allocator_scope_id);
        if (emission_it != last_emission.end() &&
            run.emission_ordinal <= emission_it->second) {
            fail(item_path + ".emission_ordinal",
                 "must increase within allocator scope");
        }
        auto tag_it = last_tag_end.find(run.allocator_scope_id);
        if (tag_it != last_tag_end.end() && run.first_tag < tag_it->second) {
            fail(item_path + ".first_tag", "overlaps/reverses allocator tag order");
        }
        last_emission[run.allocator_scope_id] = run.emission_ordinal;
        last_tag_end[run.allocator_scope_id] = tag_end;
        result.push_back(std::move(run));
    }
    return result;
}

std::vector<PipelineEndpointRun> parse_endpoint_runs(const Json& value) {
    const std::string path = "endpoint_runs";
    require_array(value, path);
    std::vector<PipelineEndpointRun> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& item = value[i];
        const std::string item_path = path + "[" + std::to_string(i) + "]";
        require_exact_keys(item, item_path,
            {"rank", "group_ordinal", "first_tile", "tile_count",
             "shard_ordinals", "first_json_id", "tile_stride",
             "shard_stride", "initial_parent_json_id", "legacy_context", "legs"});
        PipelineEndpointRun run;
        run.rank = get_rank(item.at("rank"), item_path + ".rank");
        run.group_ordinal = get_u64_key(item, "group_ordinal", item_path);
        run.first_tile = get_u64_key(item, "first_tile", item_path);
        run.tile_count = get_u64_key(item, "tile_count", item_path);
        run.first_json_id = get_u64_key(item, "first_json_id", item_path);
        run.tile_stride = get_u64_key(item, "tile_stride", item_path);
        run.shard_stride = get_u64_key(item, "shard_stride", item_path);
        run.initial_parent_json_id = get_optional_u64_nullable(
            item, "initial_parent_json_id", item_path,
            &run.has_initial_parent_json_id);
        run.legacy_context = parse_legacy_context(
            item.at("legacy_context"), item_path + ".legacy_context");
        if (run.first_tile == 0 || run.tile_count == 0 ||
            run.tile_stride == 0 ||
            run.shard_stride == 0) {
            fail(item_path, "tile/id ranges and affine strides must be positive");
        }
        // Array order is the legacy emitter's physical issue order. It is
        // intentionally not normalized by rank/group/tile: tail side-branch
        // endpoints can be emitted before the tile-1 main chain, including
        // across source/pass boundaries. Alias and coverage validation is
        // performed over the affine ranges below without reordering them.
        const auto& shard_ids = item.at("shard_ordinals");
        require_array(shard_ids, item_path + ".shard_ordinals", true);
        std::set<uint64_t> unique_shards;
        for (size_t s = 0; s < shard_ids.size(); ++s) {
            auto shard = get_u64(shard_ids[s], item_path + ".shard_ordinals[" + std::to_string(s) + "]");
            if (!unique_shards.insert(shard).second) {
                fail(item_path + ".shard_ordinals", "duplicate shard ordinal");
            }
            run.shard_ordinals.push_back(shard);
        }
        const auto& leg_array = item.at("legs");
        require_array(leg_array, item_path + ".legs", true);
        std::set<PipelineEndpointLeg> unique_legs;
        for (size_t l = 0; l < leg_array.size(); ++l) {
            const auto& leg = leg_array[l];
            const auto leg_path = item_path + ".legs[" + std::to_string(l) + "]";
            require_exact_keys(leg, leg_path,
                {"endpoint", "json_offset", "tag_emission_ordinal", "tag_offset",
                 "tag_tile_stride", "tag_shard_stride"});
            PipelineEndpointLegRun parsed;
            parsed.endpoint = parse_endpoint(leg.at("endpoint"), leg_path + ".endpoint");
            parsed.json_offset = get_u64_key(leg, "json_offset", leg_path);
            bool e = false, o = false, ts = false, ss = false;
            parsed.tag_emission_ordinal = get_optional_u64_nullable(
                leg, "tag_emission_ordinal", leg_path, &e);
            parsed.tag_offset = leg.at("tag_offset").is_null()
                ? 0 : get_u32(leg.at("tag_offset"), leg_path + ".tag_offset");
            o = !leg.at("tag_offset").is_null();
            parsed.tag_tile_stride = get_optional_u64_nullable(
                leg, "tag_tile_stride", leg_path, &ts);
            parsed.tag_shard_stride = get_optional_u64_nullable(
                leg, "tag_shard_stride", leg_path, &ss);
            const bool all_tags = e && o && ts && ss;
            const bool no_tags = !e && !o && !ts && !ss;
            const bool marker = parsed.endpoint == PipelineEndpointLeg::BodyMarker;
            if ((marker && !no_tags) || (!marker && !all_tags)) {
                fail(leg_path, "body_marker tag fields must all be null; communication legs need all four tag fields");
            }
            parsed.has_tag_fields = all_tags;
            if (all_tags && (parsed.tag_tile_stride == 0 ||
                             parsed.tag_shard_stride == 0)) {
                fail(leg_path, "tag affine strides must be positive");
            }
            if (!unique_legs.insert(parsed.endpoint).second) {
                fail(item_path + ".legs", "duplicate endpoint kind in a run");
            }
            run.legs.push_back(parsed);
        }
        result.push_back(std::move(run));
    }
    return result;
}

std::vector<uint64_t> parse_run_shards(const Json& value,
                                       const std::string& path) {
    require_array(value, path, true);
    std::vector<uint64_t> result;
    result.reserve(value.size());
    std::set<uint64_t> unique;
    for (size_t i = 0; i < value.size(); ++i) {
        const auto shard = get_u64(value[i], path + "[" + std::to_string(i) + "]");
        if (!unique.insert(shard).second) fail(path, "duplicate shard ordinal");
        result.push_back(shard);
    }
    return result;
}

std::vector<PipelineEndpointDependencyRun> parse_endpoint_dependency_runs(
    const Json& value) {
    const std::string path = "endpoint_dependency_runs";
    require_array(value, path);
    std::vector<PipelineEndpointDependencyRun> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& item = value[i];
        const std::string item_path = path + "[" + std::to_string(i) + "]";
        require_exact_keys(item, item_path,
            {"rank", "group_ordinal", "first_tile", "tile_count",
             "shard_ordinals", "endpoint", "dependency_kind",
             "first_dependency_json_id", "dependency_tile_stride",
             "dependency_shard_stride"});
        PipelineEndpointDependencyRun run;
        run.rank = get_rank(item.at("rank"), item_path + ".rank");
        run.group_ordinal = get_u64_key(item, "group_ordinal", item_path);
        run.first_tile = get_u64_key(item, "first_tile", item_path);
        run.tile_count = get_u64_key(item, "tile_count", item_path);
        run.shard_ordinals = parse_run_shards(
            item.at("shard_ordinals"), item_path + ".shard_ordinals");
        run.endpoint = parse_endpoint(item.at("endpoint"), item_path + ".endpoint");
        run.dependency_kind = parse_dependency_kind(
            item.at("dependency_kind"), item_path + ".dependency_kind");
        run.first_dependency_json_id =
            get_u64_key(item, "first_dependency_json_id", item_path);
        run.dependency_tile_stride =
            get_u64_key(item, "dependency_tile_stride", item_path);
        run.dependency_shard_stride =
            get_u64_key(item, "dependency_shard_stride", item_path);
        result.push_back(std::move(run));
    }
    return result;
}

std::vector<PipelineConsumeCompletionRun> parse_consume_completion_runs(
    const Json& value) {
    const std::string path = "consume_completion_runs";
    require_array(value, path);
    std::vector<PipelineConsumeCompletionRun> result;
    result.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& item = value[i];
        const std::string item_path = path + "[" + std::to_string(i) + "]";
        require_exact_keys(item, item_path,
            {"rank", "group_ordinal", "first_tile", "tile_count",
             "shard_ordinals", "dependency_kind",
             "first_dependency_json_id", "dependency_tile_stride",
             "dependency_shard_stride"});
        PipelineConsumeCompletionRun run;
        run.rank = get_rank(item.at("rank"), item_path + ".rank");
        run.group_ordinal = get_u64_key(item, "group_ordinal", item_path);
        run.first_tile = get_u64_key(item, "first_tile", item_path);
        run.tile_count = get_u64_key(item, "tile_count", item_path);
        run.shard_ordinals = parse_run_shards(
            item.at("shard_ordinals"), item_path + ".shard_ordinals");
        run.dependency_kind = parse_dependency_kind(
            item.at("dependency_kind"), item_path + ".dependency_kind");
        run.first_dependency_json_id =
            get_u64_key(item, "first_dependency_json_id", item_path);
        run.dependency_tile_stride =
            get_u64_key(item, "dependency_tile_stride", item_path);
        run.dependency_shard_stride =
            get_u64_key(item, "dependency_shard_stride", item_path);
        result.push_back(std::move(run));
    }
    return result;
}

uint64_t gcd_u64(uint64_t a, uint64_t b) {
    while (b != 0) {
        const uint64_t r = a % b;
        a = b;
        b = r;
    }
    return a;
}

uint64_t mod_inverse_u64(uint64_t a, uint64_t mod) {
    // Called only for gcd(a, mod)==1, mod>1. Signed 128-bit intermediates
    // cover the extended-Euclid coefficient products for uint64 moduli.
    __int128 t = 0, next_t = 1;
    __int128 r = mod, next_r = a % mod;
    while (next_r != 0) {
        const __int128 q = r / next_r;
        const __int128 old_t = t;
        t = next_t;
        next_t = old_t - q * next_t;
        const __int128 old_r = r;
        r = next_r;
        next_r = old_r - q * next_r;
    }
    if (r != 1) fail("endpoint_runs", "internal modular inverse failure");
    t %= static_cast<__int128>(mod);
    if (t < 0) t += mod;
    return static_cast<uint64_t>(t);
}

unsigned __int128 progression_intersection_count(const IdProgression& a,
                                                const IdProgression& b) {
    const unsigned __int128 a_last =
        static_cast<unsigned __int128>(a.first) +
        static_cast<unsigned __int128>(a.stride) * (a.count - 1);
    const unsigned __int128 b_last =
        static_cast<unsigned __int128>(b.first) +
        static_cast<unsigned __int128>(b.stride) * (b.count - 1);
    const unsigned __int128 low = std::max(a.first, b.first);
    const unsigned __int128 high = std::min(a_last, b_last);
    if (low > high) return 0;
    const uint64_t g = gcd_u64(a.stride, b.stride);
    const __int128 delta = static_cast<__int128>(b.first) -
                           static_cast<__int128>(a.first);
    if (delta % static_cast<__int128>(g) != 0) return 0;
    const uint64_t mod = b.stride / g;
    uint64_t k = 0;
    if (mod > 1) {
        __int128 reduced = (delta / static_cast<__int128>(g)) %
                           static_cast<__int128>(mod);
        if (reduced < 0) reduced += mod;
        const uint64_t inv = static_cast<uint64_t>(
            mod_inverse_u64(a.stride / g, mod));
        k = static_cast<uint64_t>(
            (static_cast<unsigned __int128>(reduced) * inv) % mod);
    }
    unsigned __int128 first = static_cast<unsigned __int128>(a.first) +
        static_cast<unsigned __int128>(a.stride) * k;
    const unsigned __int128 period =
        static_cast<unsigned __int128>(a.stride / g) * b.stride;
    if (first < low) {
        const unsigned __int128 gap = low - first;
        const unsigned __int128 steps =
            gap / period + (gap % period != 0 ? 1 : 0);
        first += steps * period;
    }
    if (first > high) return 0;
    return (high - first) / period + 1;
}

bool progressions_intersect(const IdProgression& a,
                            const IdProgression& b) {
    return progression_intersection_count(a, b) != 0;
}

uint64_t active_tile_count(const PipelineGroup& group,
                           const PipelineShardRule& shard);

void validate_endpoint_ranges(
    const PipelineDescriptor& descriptor,
    const std::vector<IdProgression>& progressions) {
    for (size_t i = 0; i < progressions.size(); ++i) {
        for (size_t j = i + 1; j < progressions.size(); ++j) {
            if (progressions[i].rank == progressions[j].rank &&
                progressions_intersect(progressions[i], progressions[j])) {
                fail("endpoint_runs", "affine endpoint IDs overlap: " +
                     progressions[i].label + " and " + progressions[j].label);
            }
        }
    }

    for (const auto& progression : progressions) {
        unsigned __int128 covered = 0;
        for (const auto& hole : descriptor.logical_id_ranges()) {
            if (hole.rank != progression.rank) continue;
            covered += progression_intersection_count(
                progression,
                {hole.rank, hole.first_json_id, hole.stride, hole.count,
                 "logical_id_ranges"});
            if (covered > progression.count) break;
        }
        if (covered <= progression.count) {
            for (const auto& anchor : descriptor.anchors()) {
                if (anchor.binding != PipelineAnchorBinding::PhysicalEndpointSlot ||
                    (anchor.selector.kind != PipelineSelectorKind::TileEndpointLeg &&
                     anchor.selector.kind != PipelineSelectorKind::TileEndpointTerminal &&
                     anchor.selector.kind != PipelineSelectorKind::InternalMarkerDone) ||
                    anchor.rank != progression.rank || anchor.node_id < progression.first) continue;
                const uint64_t delta = anchor.node_id - progression.first;
                if (delta % progression.stride == 0 &&
                    delta / progression.stride < progression.count) {
                    ++covered;
                    if (covered > progression.count) break;
                }
            }
        }
        if (covered != progression.count) {
            fail("endpoint_runs", progression.label +
                 " endpoint IDs must map one-to-one onto virtual AP holes or physical anchors");
        }
    }
}

std::vector<IdProgression> validate_endpoint_runs(
    const PipelineDescriptor& descriptor) {
    std::vector<IdProgression> progressions;
    for (const auto& run : descriptor.endpoint_runs()) {
        const auto group_it = std::find_if(
            descriptor.groups().begin(), descriptor.groups().end(),
            [&](const auto& group) { return group.group_ordinal == run.group_ordinal; });
        if (group_it == descriptor.groups().end()) {
            fail("endpoint_runs", "run references an unknown group");
        }
        if (run.first_tile == 0 || run.first_tile > group_it->tile_axis.tile_count ||
            run.tile_count > group_it->tile_axis.tile_count - run.first_tile + 1) {
            fail("endpoint_runs", "run tile range is outside its group");
        }
        uint64_t last_tile = 0;
        if (!checked_add(run.first_tile, run.tile_count - 1, &last_tile)) {
            fail("endpoint_runs", "run tile range overflows uint64");
        }

        std::vector<size_t> shard_positions;
        shard_positions.reserve(run.shard_ordinals.size());
        size_t previous_position = 0;
        bool have_previous = false;
        for (const auto shard_ordinal : run.shard_ordinals) {
            const auto shard_it = std::find_if(
                group_it->shards.begin(), group_it->shards.end(),
                [&](const auto& shard) { return shard.shard_ordinal == shard_ordinal; });
            if (shard_it == group_it->shards.end()) {
                fail("endpoint_runs", "run references an unknown shard");
            }
            const size_t position = static_cast<size_t>(
                std::distance(group_it->shards.begin(), shard_it));
            if (have_previous && position <= previous_position) {
                fail("endpoint_runs", "shard_ordinals must preserve builder order");
            }
            previous_position = position;
            have_previous = true;
            shard_positions.push_back(position);
        }

        for (const auto& leg : run.legs) {
            if (leg.endpoint == PipelineEndpointLeg::BodyMarker) {
                if (run.shard_ordinals.size() != 1) {
                    fail("endpoint_runs",
                         "body_marker run must name one canonical target shard");
                }
                uint64_t canonical_target_shard =
                    std::numeric_limits<uint64_t>::max();
                // A multi-rank group can have lower-ordinal target shards
                // owned by another rank. The marker is canonical within this
                // descriptor's owner rank, not across the whole group.
                for (const auto& shard : group_it->shards) {
                    if (shard.target_rank == descriptor.identity().rank) {
                        canonical_target_shard = std::min(
                            canonical_target_shard, shard.shard_ordinal);
                    }
                }
                if (canonical_target_shard ==
                        std::numeric_limits<uint64_t>::max() ||
                    run.shard_ordinals.front() != canonical_target_shard) {
                    fail("endpoint_runs",
                         "body_marker must map the canonical target shard");
                }
            }
            for (size_t local_shard = 0; local_shard < run.shard_ordinals.size();
                 ++local_shard) {
                const auto& shard = group_it->shards[shard_positions[local_shard]];
                if (leg.endpoint != PipelineEndpointLeg::BodyMarker &&
                    last_tile > active_tile_count(*group_it, shard)) {
                    fail("endpoint_runs",
                         "COMM endpoint run extends beyond the shard's active tile count");
                }
                const int expected_rank =
                    leg.endpoint == PipelineEndpointLeg::BodyMarker
                        ? descriptor.identity().rank
                        : ((leg.endpoint == PipelineEndpointLeg::DataSend ||
                            leg.endpoint == PipelineEndpointLeg::FillAckRecv ||
                            leg.endpoint == PipelineEndpointLeg::ReleaseAckRecv)
                               ? shard.source_rank : shard.target_rank);
                if (run.rank != expected_rank) {
                    fail("endpoint_runs", "leg rank disagrees with its shard endpoint");
                }

                const uint64_t shard_ordinal = run.shard_ordinals[local_shard];
                const uint64_t first_id = descriptor.endpoint_json_id(
                    run, leg, run.first_tile, shard_ordinal);
                const uint64_t last_id = descriptor.endpoint_json_id(
                    run, leg, last_tile, shard_ordinal);
                if (last_id < first_id) {
                    fail("endpoint_runs", "endpoint ID affine range wrapped");
                }
                progressions.push_back({run.rank, first_id, run.tile_stride,
                                        run.tile_count,
                                        "group=" + std::to_string(run.group_ordinal) +
                                        ",endpoint=" + std::to_string(
                                            static_cast<int>(leg.endpoint)) +
                                        ",shard=" + std::to_string(shard_ordinal)});

                if (leg.has_tag_fields) {
                    // Positive affine strides make the terminal corner the
                    // largest tag offset. Checking both endpoint corners
                    // also proves the emission/rank-domain lookup succeeds.
                    const auto first_tag = descriptor.endpoint_tag(
                        run, leg, run.first_tile, shard_ordinal);
                    const auto last_tag = descriptor.endpoint_tag(
                        run, leg, last_tile, shard_ordinal);
                    if (last_tag < first_tag) {
                        fail("endpoint_runs", "endpoint tag affine range wrapped");
                    }
                }
            }
        }
    }

    validate_endpoint_ranges(descriptor, progressions);

    for (const auto& anchor : descriptor.anchors()) {
        if (anchor.binding != PipelineAnchorBinding::PhysicalEndpointSlot ||
            (anchor.selector.kind != PipelineSelectorKind::TileEndpointLeg &&
             anchor.selector.kind != PipelineSelectorKind::TileEndpointTerminal &&
             anchor.selector.kind != PipelineSelectorKind::InternalMarkerDone)) {
            continue;
        }
        const PipelineEndpointRun* matching_run = nullptr;
        const PipelineEndpointLegRun* matching_leg = nullptr;
        size_t matches = 0;
        for (const auto& run : descriptor.endpoint_runs()) {
            if (run.rank != anchor.rank ||
                run.group_ordinal != anchor.selector.group_ordinal ||
                anchor.selector.tile_ordinal < run.first_tile ||
                anchor.selector.tile_ordinal - run.first_tile >= run.tile_count ||
                std::find(run.shard_ordinals.begin(), run.shard_ordinals.end(),
                          anchor.selector.shard_ordinal) == run.shard_ordinals.end()) {
                continue;
            }
            for (const auto& leg : run.legs) {
                if (leg.endpoint == anchor.selector.endpoint) {
                    ++matches;
                    matching_run = &run;
                    matching_leg = &leg;
                }
            }
        }
        if (matches != 1 || matching_run == nullptr || matching_leg == nullptr) {
            fail("anchors", "physical endpoint selector must resolve to exactly one endpoint run leg");
        }
        const auto resolved_id = descriptor.endpoint_json_id(
            *matching_run, *matching_leg, anchor.selector.tile_ordinal,
            anchor.selector.shard_ordinal);
        if (resolved_id != anchor.node_id) {
            fail("anchors", "physical endpoint node_id disagrees with endpoint_runs affine mapping");
        }
        if (matching_leg->has_tag_fields) {
            if (matching_leg->tag_emission_ordinal !=
                anchor.selector.tag_allocation_emission_ordinal) {
                fail("anchors", "physical endpoint tag emission ordinal disagrees with endpoint_runs");
            }
            const auto resolved_tag = descriptor.endpoint_tag(
                *matching_run, *matching_leg, anchor.selector.tile_ordinal,
                anchor.selector.shard_ordinal);
            const auto tag_run = std::find_if(
                descriptor.tag_allocation_runs().begin(),
                descriptor.tag_allocation_runs().end(),
                [&](const auto& candidate) {
                    return candidate.emission_ordinal ==
                           anchor.selector.tag_allocation_emission_ordinal;
                });
            if (tag_run == descriptor.tag_allocation_runs().end() ||
                static_cast<uint64_t>(tag_run->first_tag) +
                    anchor.selector.tag_offset != resolved_tag) {
                fail("anchors", "physical endpoint tag offset disagrees with endpoint_runs affine mapping");
            }
        }
    }
    return progressions;
}

const PipelineGroup* find_group(const PipelineDescriptor& descriptor,
                               uint64_t group_ordinal) {
    const auto it = std::find_if(
        descriptor.groups().begin(), descriptor.groups().end(),
        [&](const auto& group) { return group.group_ordinal == group_ordinal; });
    return it == descriptor.groups().end() ? nullptr : &*it;
}

size_t shard_position(const PipelineGroup& group, uint64_t shard_ordinal,
                      const std::string& path) {
    const auto it = std::find_if(group.shards.begin(), group.shards.end(),
        [&](const auto& shard) { return shard.shard_ordinal == shard_ordinal; });
    if (it == group.shards.end()) fail(path, "references unknown shard ordinal");
    return static_cast<size_t>(std::distance(group.shards.begin(), it));
}

uint64_t active_tile_count(const PipelineGroup& group,
                           const PipelineShardRule& shard) {
    if (shard.total_data_bytes == 0) return 0;
    if (group.tile_axis.unit == PipelineTileUnit::SourceTokens) {
        return group.tile_axis.tile_count;
    }
    const uint64_t tile_bytes = group.tile_axis.units_per_tile;
    return shard.total_data_bytes / tile_bytes +
        (shard.total_data_bytes % tile_bytes != 0 ? 1 : 0);
}

uint64_t progression_last(uint64_t first, uint64_t count, uint64_t stride,
                          const std::string& path) {
    if (count == 0) fail(path, "tile_count must be positive");
    const unsigned __int128 last = static_cast<unsigned __int128>(first) +
        static_cast<unsigned __int128>(count - 1) * stride;
    if (last > std::numeric_limits<uint64_t>::max()) {
        fail(path, "affine progression exceeds uint64");
    }
    return static_cast<uint64_t>(last);
}

uint64_t dependency_id(const uint64_t first, const uint64_t tile_stride,
                       const uint64_t tile_delta, const uint64_t shard_stride,
                       const uint64_t local_shard, const std::string& path) {
    const unsigned __int128 value = static_cast<unsigned __int128>(first) +
        static_cast<unsigned __int128>(tile_stride) * tile_delta +
        static_cast<unsigned __int128>(shard_stride) * local_shard;
    if (value > std::numeric_limits<uint64_t>::max()) {
        fail(path, "affine dependency ID exceeds uint64");
    }
    return static_cast<uint64_t>(value);
}

void validate_run_shards(const PipelineGroup& group,
                         const std::vector<uint64_t>& shards,
                         const std::string& path) {
    size_t previous = 0;
    bool have_previous = false;
    for (const auto shard : shards) {
        const auto position = shard_position(group, shard, path);
        if (have_previous && position <= previous) {
            fail(path, "shard_ordinals must preserve builder order");
        }
        previous = position;
        have_previous = true;
    }
}

void validate_endpoint_child_span(const PipelineDescriptor& descriptor,
                                  int rank, uint64_t group_ordinal,
                                  uint64_t first_tile, uint64_t tile_count,
                                  const std::vector<uint64_t>& shards,
                                  PipelineEndpointLeg endpoint,
                                  const std::string& path) {
    const auto* group = find_group(descriptor, group_ordinal);
    if (group == nullptr) fail(path + ".group_ordinal", "references unknown group");
    const uint64_t last_tile = progression_last(first_tile, tile_count, 1, path);
    if (first_tile == 0 || last_tile > group->tile_axis.tile_count) {
        fail(path, "tile range is outside its group");
    }
    validate_run_shards(*group, shards, path + ".shard_ordinals");

    int expected_rank = -1;
    for (const auto shard_ordinal : shards) {
        const auto& shard = group->shards[shard_position(*group, shard_ordinal, path)];
        if (endpoint != PipelineEndpointLeg::BodyMarker &&
            last_tile > active_tile_count(*group, shard)) {
            fail(path,
                 "COMM endpoint dependency extends beyond the shard's active tile count");
        }
        const int shard_rank = endpoint == PipelineEndpointLeg::BodyMarker
            ? shard.target_rank
            : ((endpoint == PipelineEndpointLeg::DataSend ||
                endpoint == PipelineEndpointLeg::FillAckRecv ||
                endpoint == PipelineEndpointLeg::ReleaseAckRecv)
                   ? shard.source_rank : shard.target_rank);
        if (expected_rank < 0) expected_rank = shard_rank;
        if (rank != shard_rank) {
            fail(path + ".rank", "does not match the child endpoint physical rank");
        }
    }
    (void)expected_rank;

    // Check interval coverage by structural endpoint runs. The loop advances
    // by run boundaries, never by virtual tile count.
    for (const auto shard : shards) {
        uint64_t cursor = first_tile;
        while (cursor <= last_tile) {
            const PipelineEndpointRun* match = nullptr;
            const PipelineEndpointLegRun* match_leg = nullptr;
            uint64_t next_tile = cursor;
            size_t matches = 0;
            for (const auto& run : descriptor.endpoint_runs()) {
                if (run.rank != rank || run.group_ordinal != group_ordinal ||
                    cursor < run.first_tile ||
                    cursor - run.first_tile >= run.tile_count ||
                    std::find(run.shard_ordinals.begin(), run.shard_ordinals.end(),
                              shard) == run.shard_ordinals.end()) {
                    continue;
                }
                const auto leg = std::find_if(run.legs.begin(), run.legs.end(),
                    [&](const auto& candidate) { return candidate.endpoint == endpoint; });
                if (leg == run.legs.end()) continue;
                ++matches;
                match = &run;
                match_leg = &*leg;
                next_tile = std::min(last_tile, run.first_tile + run.tile_count - 1);
            }
            if (matches != 1 || match == nullptr || match_leg == nullptr) {
                fail(path, "child selector is not covered by exactly one endpoint run");
            }
            if (next_tile == std::numeric_limits<uint64_t>::max()) break;
            cursor = next_tile + 1;
        }
    }
}

void validate_endpoint_dependency_runs(
    const PipelineDescriptor& descriptor,
    const std::vector<PipelineEndpointDependencyRun>& runs) {
    std::set<std::tuple<int, uint64_t, uint64_t, uint64_t,
                        PipelineEndpointLeg, PipelineEndpointDependencyKind,
                        uint64_t, uint64_t, uint64_t, std::vector<uint64_t>>> seen;
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];
        const std::string path = "endpoint_dependency_runs[" + std::to_string(i) + "]";
        validate_endpoint_child_span(descriptor, run.rank, run.group_ordinal,
            run.first_tile, run.tile_count, run.shard_ordinals, run.endpoint, path);
        const auto* group = find_group(descriptor, run.group_ordinal);
        (void)group;
        (void)progression_last(run.first_dependency_json_id, run.tile_count,
                               run.dependency_tile_stride,
                               path + ".first_dependency_json_id");
        const unsigned __int128 max_id =
            static_cast<unsigned __int128>(run.first_dependency_json_id) +
            static_cast<unsigned __int128>(run.dependency_tile_stride) *
                (run.tile_count - 1) +
            static_cast<unsigned __int128>(run.dependency_shard_stride) *
                (run.shard_ordinals.size() - 1);
        if (max_id > std::numeric_limits<uint64_t>::max()) {
            fail(path, "tile/shard dependency affine range exceeds uint64");
        }
        const auto key = std::make_tuple(
            run.rank, run.group_ordinal, run.first_tile, run.tile_count,
            run.endpoint, run.dependency_kind, run.first_dependency_json_id,
            run.dependency_tile_stride, run.dependency_shard_stride,
            run.shard_ordinals);
        if (!seen.insert(key).second) fail(path, "duplicate dependency run");
    }

    // initial_parent_json_id is retained for builder compatibility, but must
    // mirror the first encoded parent for that endpoint's first tile/shard.
    for (size_t i = 0; i < descriptor.endpoint_runs().size(); ++i) {
        const auto& endpoint_run = descriptor.endpoint_runs()[i];
        if (!endpoint_run.has_initial_parent_json_id) continue;
        if (endpoint_run.shard_ordinals.empty()) {
            fail("endpoint_runs[" + std::to_string(i) + "]",
                 "initial parent requires a first shard");
        }
        bool found = false;
        for (const auto& leg : endpoint_run.legs) {
            for (const auto& dependency : runs) {
                if (dependency.rank != endpoint_run.rank ||
                    dependency.group_ordinal != endpoint_run.group_ordinal ||
                    dependency.endpoint != leg.endpoint ||
                    endpoint_run.first_tile < dependency.first_tile ||
                    endpoint_run.first_tile - dependency.first_tile >= dependency.tile_count) {
                    continue;
                }
                const auto shard_it = std::find(dependency.shard_ordinals.begin(),
                    dependency.shard_ordinals.end(), endpoint_run.shard_ordinals.front());
                if (shard_it == dependency.shard_ordinals.end()) continue;
                const auto local_shard = static_cast<uint64_t>(
                    std::distance(dependency.shard_ordinals.begin(), shard_it));
                const auto id = dependency_id(
                    dependency.first_dependency_json_id,
                    dependency.dependency_tile_stride,
                    endpoint_run.first_tile - dependency.first_tile,
                    dependency.dependency_shard_stride, local_shard,
                    "endpoint_dependency_runs");
                if (id != endpoint_run.initial_parent_json_id) {
                    fail("endpoint_runs[" + std::to_string(i) + "]",
                         "initial_parent_json_id disagrees with its encoded first parent");
                }
                found = true;
                break;
            }
            if (found) break;
        }
        if (!found) {
            fail("endpoint_runs[" + std::to_string(i) + "]",
                 "initial_parent_json_id has no matching endpoint dependency rule");
        }
    }
}

void validate_consume_completion_runs(
    const PipelineDescriptor& descriptor,
    const std::vector<PipelineConsumeCompletionRun>& runs) {
    struct CoveredTileSpan final {
        int rank;
        uint64_t group;
        uint64_t first;
        uint64_t last;
        uint64_t shard;
    };
    std::vector<CoveredTileSpan> covered;
    for (size_t i = 0; i < runs.size(); ++i) {
        const auto& run = runs[i];
        const std::string path = "consume_completion_runs[" + std::to_string(i) + "]";
        const auto* group = find_group(descriptor, run.group_ordinal);
        if (group == nullptr) fail(path + ".group_ordinal", "references unknown group");
        if (run.rank != descriptor.identity().rank) {
            fail(path + ".rank", "consume completion rank must equal descriptor owner rank");
        }
        const uint64_t last_tile = progression_last(run.first_tile, run.tile_count, 1, path);
        if (run.first_tile == 0 || last_tile > group->tile_axis.tile_count) {
            fail(path, "tile range is outside its group");
        }
        validate_run_shards(*group, run.shard_ordinals, path + ".shard_ordinals");
        for (const auto shard_ordinal : run.shard_ordinals) {
            const auto& shard = group->shards[
                shard_position(*group, shard_ordinal, path)];
            if (shard.target_rank != run.rank) {
                fail(path + ".rank", "completion shard must be owned by target rank");
            }
            if (last_tile > active_tile_count(*group, shard)) {
                fail(path,
                     "completion run extends beyond the shard's active tile count");
            }
        }
        const unsigned __int128 max_id =
            static_cast<unsigned __int128>(run.first_dependency_json_id) +
            static_cast<unsigned __int128>(run.dependency_tile_stride) *
                (run.tile_count - 1) +
            static_cast<unsigned __int128>(run.dependency_shard_stride) *
                (run.shard_ordinals.size() - 1);
        if (max_id > std::numeric_limits<uint64_t>::max()) {
            fail(path, "tile/shard completion affine range exceeds uint64");
        }
        for (const auto shard : run.shard_ordinals) {
            for (const auto& previous : covered) {
                if (previous.rank == run.rank && previous.group == run.group_ordinal &&
                    previous.shard == shard && run.first_tile <= previous.last &&
                    previous.first <= last_tile) {
                    fail(path, "completion tile/shard selector is duplicated");
                }
            }
        covered.push_back({run.rank, run.group_ordinal, run.first_tile,
                           last_tile, shard});
        }

        if (run.dependency_kind !=
            PipelineEndpointDependencyKind::VirtualEndpointTerminal) {
            continue;
        }
        // A virtual consume completion is specifically the body's marker
        // terminal. The builder may emit one marker per target rank rather
        // than one per shard; in that case its canonical shard is the
        // minimum target shard ordinal on the descriptor owner rank and each
        // local shard's consumer points to that same terminal with
        // dependency_shard_stride==0.
        uint64_t canonical_target_shard = std::numeric_limits<uint64_t>::max();
        for (const auto& candidate : group->shards) {
            if (candidate.target_rank == run.rank) {
                canonical_target_shard =
                    std::min(canonical_target_shard, candidate.shard_ordinal);
            }
        }
        if (canonical_target_shard == std::numeric_limits<uint64_t>::max()) {
            fail(path, "virtual completion group has no target shard");
        }
        for (size_t local_shard = 0; local_shard < run.shard_ordinals.size();
             ++local_shard) {
            const auto shard = run.shard_ordinals[local_shard];
            uint64_t cursor = run.first_tile;
            while (cursor <= last_tile) {
                const PipelineEndpointRun* marker_run = nullptr;
                const PipelineEndpointLegRun* marker_leg = nullptr;
                size_t matches = 0;
                uint64_t segment_end = cursor;
                for (const auto& endpoint_run : descriptor.endpoint_runs()) {
                    if (endpoint_run.rank != run.rank ||
                        endpoint_run.group_ordinal != run.group_ordinal ||
                        cursor < endpoint_run.first_tile ||
                        cursor - endpoint_run.first_tile >= endpoint_run.tile_count ||
                        std::none_of(endpoint_run.legs.begin(),
                            endpoint_run.legs.end(), [](const auto& candidate) {
                                return candidate.endpoint ==
                                       PipelineEndpointLeg::BodyMarker;
                            })) {
                        continue;
                    }
                    const auto leg = std::find_if(endpoint_run.legs.begin(),
                        endpoint_run.legs.end(), [](const auto& candidate) {
                            return candidate.endpoint == PipelineEndpointLeg::BodyMarker;
                    });
                    if (leg == endpoint_run.legs.end()) continue;
                    const bool direct_shard = std::find(
                        endpoint_run.shard_ordinals.begin(),
                        endpoint_run.shard_ordinals.end(), shard) !=
                        endpoint_run.shard_ordinals.end();
                    const bool canonical_broadcast =
                        run.dependency_shard_stride == 0 &&
                        endpoint_run.shard_ordinals.size() == 1 &&
                        endpoint_run.shard_ordinals.front() ==
                            canonical_target_shard;
                    if (!direct_shard && !canonical_broadcast) continue;
                    ++matches;
                    marker_run = &endpoint_run;
                    marker_leg = &*leg;
                    segment_end = std::min(last_tile,
                        endpoint_run.first_tile + endpoint_run.tile_count - 1);
                }
                if (matches != 1 || marker_run == nullptr || marker_leg == nullptr) {
                    fail(path, "virtual completion does not resolve to one body_marker run");
                }
                const auto completion_id = dependency_id(
                    run.first_dependency_json_id, run.dependency_tile_stride,
                    cursor - run.first_tile, run.dependency_shard_stride,
                    local_shard, path);
                const auto marker_id = descriptor.endpoint_json_id(
                    *marker_run, *marker_leg, cursor,
                    marker_run->shard_ordinals.front());
                if (completion_id != marker_id ||
                    (segment_end > cursor &&
                     run.dependency_tile_stride != marker_run->tile_stride)) {
                    fail(path, "virtual completion affine mapping disagrees with body_marker IDs");
                }
                if (segment_end == std::numeric_limits<uint64_t>::max()) break;
                cursor = segment_end + 1;
            }
        }
    }

    // Execution descriptors must identify the release authority for every
    // target-side tile, including tails that have no separate communication
    // endpoint. Prediction-only descriptors intentionally have no execution
    // endpoint map and therefore keep both run arrays empty.
    if (!descriptor.endpoint_runs().empty()) {
        for (const auto& group : descriptor.groups()) {
            for (size_t shard_index = 0; shard_index < group.shards.size();
                 ++shard_index) {
                const auto& shard = group.shards[shard_index];
                if (shard.target_rank != descriptor.identity().rank) continue;
                std::vector<std::pair<uint64_t, uint64_t>> intervals;
                for (const auto& span : covered) {
                    if (span.rank == descriptor.identity().rank &&
                        span.group == group.group_ordinal &&
                        span.shard == shard.shard_ordinal) {
                        intervals.emplace_back(span.first, span.last);
                    }
                }
                const uint64_t expected_active_tiles =
                    active_tile_count(group, shard);
                if (expected_active_tiles == 0) {
                    if (!intervals.empty()) {
                        fail("consume_completion_runs",
                             "empty shard must not have completion work");
                    }
                    continue;
                }
                std::sort(intervals.begin(), intervals.end());
                unsigned __int128 next_tile = 1;
                for (const auto& interval : intervals) {
                    if (interval.first != next_tile) {
                        fail("consume_completion_runs",
                             "completion runs have a gap or overlap for group " +
                             std::to_string(group.group_ordinal) + ", shard " +
                             std::to_string(shard.shard_ordinal));
                    }
                    next_tile = static_cast<unsigned __int128>(interval.second) + 1;
                }
                if (next_tile !=
                    static_cast<unsigned __int128>(expected_active_tiles) + 1) {
                    fail("consume_completion_runs",
                         "completion runs do not cover every active target tile for group " +
                         std::to_string(group.group_ordinal) + ", shard " +
                         std::to_string(shard.shard_ordinal));
                }
            }
        }
    }
}

std::vector<PipelineAnchor> parse_anchors(const Json& value) {
    const std::string path = "anchors";
    require_array(value, path);
    std::vector<PipelineAnchor> result;
    result.reserve(value.size());
    std::map<PipelineAnchorRole, size_t> role_counts;
    std::set<std::pair<int, uint64_t>> physical_slots;
    std::set<std::string> selector_keys;
    for (size_t i = 0; i < value.size(); ++i) {
        const auto& item = value[i];
        const std::string item_path = path + "[" + std::to_string(i) + "]";
        require_exact_keys(item, item_path,
                           {"role", "rank", "binding", "node_id", "selector"});
        PipelineAnchor anchor;
        anchor.role = parse_anchor_role(item.at("role"), item_path + ".role");
        const auto role_count = ++role_counts[anchor.role];
        anchor.rank = get_rank(item.at("rank"), item_path + ".rank");
        anchor.binding = parse_anchor_binding(item.at("binding"), item_path + ".binding");
        if ((anchor.role == PipelineAnchorRole::Service ||
             anchor.role == PipelineAnchorRole::Settle) && role_count != 1) {
            fail(item_path + ".role", "service and settle anchors must be unique");
        }
        if ((anchor.role == PipelineAnchorRole::Service ||
             anchor.role == PipelineAnchorRole::Settle) &&
            anchor.binding != PipelineAnchorBinding::DescriptorCursorSelector) {
            fail(item_path + ".binding", "service and settle anchors must be cursor-only");
        }
        if ((anchor.role == PipelineAnchorRole::Start ||
             anchor.role == PipelineAnchorRole::Frontier) &&
            anchor.binding != PipelineAnchorBinding::PhysicalEndpointSlot) {
            fail(item_path + ".binding", "start and frontier anchors must bind physical endpoint slots");
        }
        const auto& node_id = item.at("node_id");
        if (!node_id.is_null()) {
            anchor.node_id = get_u64(node_id, item_path + ".node_id");
            anchor.has_node_id = true;
        }
        const auto& selector = item.at("selector");
        const std::string selector_path = item_path + ".selector";
        if (!selector_keys.insert(selector.dump()).second) {
            fail(selector_path, "anchor selector must be unique");
        }
        require_exact_keys(selector, selector_path, {"kind"},
                           {"group_ordinal", "tile_ordinal", "shard_ordinal",
                            "endpoint", "existing_gate",
                            "tag_allocation_emission_ordinal", "tag_offset"});
        anchor.selector.kind =
            parse_selector_kind(selector.at("kind"), selector_path + ".kind");
        if (selector.contains("group_ordinal")) {
            anchor.selector.group_ordinal =
                get_u64_key(selector, "group_ordinal", selector_path);
        }
        if (selector.contains("tile_ordinal")) {
            anchor.selector.tile_ordinal =
                get_u64_key(selector, "tile_ordinal", selector_path);
        }
        if (selector.contains("shard_ordinal")) {
            anchor.selector.shard_ordinal =
                get_u64_key(selector, "shard_ordinal", selector_path);
        }
        if (selector.contains("endpoint")) {
            anchor.selector.endpoint =
                parse_endpoint(selector.at("endpoint"), selector_path + ".endpoint");
        }
        if (selector.contains("tag_allocation_emission_ordinal")) {
            anchor.selector.tag_allocation_emission_ordinal =
                get_u64_key(selector, "tag_allocation_emission_ordinal", selector_path);
        }
        if (selector.contains("tag_offset")) {
            anchor.selector.tag_offset =
                get_u32(selector.at("tag_offset"), selector_path + ".tag_offset");
        }
        if (selector.contains("existing_gate")) {
            anchor.selector.existing_gate = get_nonempty_string(
                selector.at("existing_gate"), selector_path + ".existing_gate");
        }

        const bool marker_selector =
            anchor.selector.kind == PipelineSelectorKind::InternalMarkerDone;
        const bool endpoint_selector =
            anchor.selector.kind == PipelineSelectorKind::TileEndpointLeg ||
            anchor.selector.kind == PipelineSelectorKind::TileEndpointTerminal ||
            marker_selector;
        if (anchor.binding == PipelineAnchorBinding::PhysicalEndpointSlot) {
            if (!anchor.has_node_id) {
                fail(item_path + ".node_id", "physical binding requires node_id");
            }
            if (!physical_slots.emplace(anchor.rank, anchor.node_id).second) {
                fail(item_path + ".node_id", "physical rank/node_id slot must be unique");
            }
            if (endpoint_selector) {
                if (!selector.contains("group_ordinal") ||
                    !selector.contains("tile_ordinal") ||
                    !selector.contains("shard_ordinal") ||
                    !selector.contains("endpoint")) {
                    fail(item_path, "physical endpoint anchor needs a complete endpoint selector");
                }
                const bool marker = anchor.selector.endpoint == PipelineEndpointLeg::BodyMarker;
                if (marker_selector) {
                    if (anchor.role != PipelineAnchorRole::Frontier) {
                        fail(item_path + ".role", "internal marker selector is frontier-only");
                    }
                    if (!marker) {
                        fail(selector_path + ".endpoint", "internal_marker_done must select body_marker");
                    }
                    if (selector.contains("tag_allocation_emission_ordinal") ||
                        selector.contains("tag_offset") ||
                        selector.contains("existing_gate")) {
                        fail(selector_path,
                             "internal_marker_done forbids tag and existing_gate fields");
                    }
                } else {
                    if (marker) {
                        fail(selector_path + ".kind",
                             "body_marker requires internal_marker_done selector kind");
                    }
                    if (!selector.contains("tag_allocation_emission_ordinal") ||
                        !selector.contains("tag_offset")) {
                        fail(item_path, "communication endpoint anchor must reference its tag allocation");
                    }
                }
                if (anchor.selector.tile_ordinal == 0) {
                    fail(selector_path + ".tile_ordinal", "endpoint tile ordinal is one-based");
                }
            } else if (anchor.selector.kind == PipelineSelectorKind::ExistingServiceGate) {
                if (anchor.selector.existing_gate.empty()) {
                    fail(selector_path + ".existing_gate",
                         "physical service gate binding requires an existing gate name");
                }
            } else {
                fail(item_path, "physical binding requires endpoint or existing gate selector");
            }
            if ((anchor.role == PipelineAnchorRole::Start ||
                 anchor.role == PipelineAnchorRole::Frontier) &&
                !endpoint_selector) {
                fail(item_path + ".binding",
                     "start/frontier anchors must bind original endpoint slots");
            }
        } else {
            if (anchor.has_node_id) {
                fail(item_path + ".node_id", "cursor selector must not bind a physical node");
            }
            if (endpoint_selector ||
                anchor.selector.kind == PipelineSelectorKind::StageStart) {
                fail(selector_path + ".kind", "cursor selector kind is incompatible with binding");
            }
            if (anchor.selector.kind == PipelineSelectorKind::ExistingServiceGate &&
                anchor.selector.existing_gate.empty()) {
                fail(selector_path + ".existing_gate", "service gate selector requires a gate name");
            }
        }
        result.push_back(std::move(anchor));
    }
    if (!result.empty() &&
        (role_counts[PipelineAnchorRole::Start] == 0 ||
         role_counts[PipelineAnchorRole::Frontier] == 0 ||
         role_counts[PipelineAnchorRole::Service] != 1 ||
         role_counts[PipelineAnchorRole::Settle] != 1)) {
        fail(path, "non-empty anchors need start/frontier anchors and exactly one service/settle anchor");
    }
    return result;
}

void parse_completion_order(const Json& wire, const std::string& path,
                            uint64_t tile_count, bool* has_first,
                            int64_t* first, int64_t* stride) {
    const auto& first_wire = wire.at("completion_order_first");
    *has_first = !first_wire.is_null();
    *first = *has_first ? get_i64(first_wire, path + ".completion_order_first") : 0;
    *stride = get_i64(wire.at("completion_order_stride"),
                       path + ".completion_order_stride");
    if (!*has_first) {
        if (*stride != 0) {
            fail(path, "null completion_order_first requires zero stride");
        }
        return;
    }
    const __int128 last = static_cast<__int128>(*first) +
        static_cast<__int128>(*stride) * static_cast<__int128>(tile_count - 1);
    if (last < std::numeric_limits<int64_t>::min() ||
        last > std::numeric_limits<int64_t>::max()) {
        fail(path, "completion-order progression overflows int64");
    }
}

PipelineProjectionShare parse_projection_share(const Json& wire,
                                                const std::string& path) {
    require_exact_keys(wire, path,
                       {"rule", "total_units", "total_tiles",
                        "share_index_first", "tile_bytes"});
    PipelineProjectionShare result;
    result.rule = get_nonempty_string(wire.at("rule"), path + ".rule");
    if (result.rule != "quotient_remainder_earliest_global_indices") {
        fail(path + ".rule", "unsupported byte-share rule");
    }
    result.total_units = get_u64_key(wire, "total_units", path);
    result.total_tiles = get_u64_key(wire, "total_tiles", path);
    result.share_index_first = get_u64_key(wire, "share_index_first", path);
    result.tile_bytes = get_u64_key(wire, "tile_bytes", path);
    uint64_t share_end = 0;
    if (result.total_tiles == 0 || result.tile_bytes == 0 ||
        !checked_add(result.share_index_first, 1, &share_end) ||
        share_end > result.total_tiles) {
        fail(path, "share tile range is empty or outside total_tiles");
    }
    return result;
}

PipelineProjectionFill parse_projection_fill(const Json& wire,
                                              const std::string& path,
                                              uint64_t tile_count) {
    require_exact_keys(wire, path,
        {"leg_id_prefix", "first_index", "fill_order_first",
         "fill_order_stride", "completion_order_first",
         "completion_order_stride", "route", "bytes_per_tile",
         "latency_ns", "start_deps", "gate_id", "gate_capacity"});
    PipelineProjectionFill result;
    result.leg_id_prefix = get_nonempty_string(wire.at("leg_id_prefix"), path + ".leg_id_prefix");
    result.first_index = get_u64_key(wire, "first_index", path);
    result.fill_order_first = get_u64_key(wire, "fill_order_first", path);
    result.fill_order_stride = get_u64_key(wire, "fill_order_stride", path);
    if (result.fill_order_stride == 0) fail(path + ".fill_order_stride", "must be positive");
    parse_completion_order(wire, path, tile_count,
        &result.has_completion_order_first, &result.completion_order_first,
        &result.completion_order_stride);
    result.route = parse_nonempty_string_array(wire.at("route"), path + ".route", true);
    result.bytes_per_tile = get_u64_key(wire, "bytes_per_tile", path);
    if (result.bytes_per_tile == 0) fail(path + ".bytes_per_tile", "must be positive");
    result.latency_ns = get_u64_key(wire, "latency_ns", path);
    result.start_deps = parse_nonempty_string_array(wire.at("start_deps"), path + ".start_deps");
    result.gate_id = get_nonempty_string(wire.at("gate_id"), path + ".gate_id");
    result.gate_capacity = get_u32(wire.at("gate_capacity"), path + ".gate_capacity");
    if (result.gate_capacity == 0) fail(path + ".gate_capacity", "must be positive");
    uint64_t ignored = 0, step = 0;
    if (!checked_add(result.first_index, tile_count, &ignored) ||
        !checked_mul(result.fill_order_stride, tile_count - 1, &step) ||
        !checked_add(result.fill_order_first, step, &ignored)) {
        fail(path, "fill affine ID/order range overflows uint64");
    }
    return result;
}

bool parse_nullable_projection_share(const Json& wire,
                                     const std::string& path,
                                     PipelineProjectionShare* output) {
    if (wire.is_null()) return false;
    *output = parse_projection_share(wire, path);
    return true;
}

PipelineProjectionConsume parse_projection_consume(const Json& wire,
                                                    const std::string& path,
                                                    uint64_t tile_count) {
    require_exact_keys(wire, path,
        {"leg_id_prefix", "first_index", "consume_order_first",
         "consume_order_stride", "completion_order_first",
         "completion_order_stride", "kind", "route", "group",
         "latency_first_ns", "latency_rest_ns", "consume_deps",
         "consume_start_deps", "gate_id", "compute_share", "hbm_share"});
    PipelineProjectionConsume result;
    result.leg_id_prefix = get_nonempty_string(wire.at("leg_id_prefix"), path + ".leg_id_prefix");
    result.first_index = get_u64_key(wire, "first_index", path);
    result.consume_order_first = get_u64_key(wire, "consume_order_first", path);
    result.consume_order_stride = get_u64_key(wire, "consume_order_stride", path);
    if (result.consume_order_stride == 0) fail(path + ".consume_order_stride", "must be positive");
    parse_completion_order(wire, path, tile_count,
        &result.has_completion_order_first, &result.completion_order_first,
        &result.completion_order_stride);
    result.kind = get_string(wire.at("kind"), path + ".kind");
    if (result.kind != "fixed" && result.kind != "compute") {
        fail(path + ".kind", "must be fixed or compute");
    }
    result.route = parse_nonempty_string_array(wire.at("route"), path + ".route");
    if (!wire.at("group").is_null()) {
        result.has_group = true;
        result.group = get_nonempty_string(wire.at("group"), path + ".group");
    }
    result.latency_first_ns = get_u64_key(wire, "latency_first_ns", path);
    result.latency_rest_ns = get_u64_key(wire, "latency_rest_ns", path);
    result.consume_deps = parse_nonempty_string_array(wire.at("consume_deps"), path + ".consume_deps");
    result.consume_start_deps = parse_nonempty_string_array(wire.at("consume_start_deps"), path + ".consume_start_deps");
    result.gate_id = get_nonempty_string(wire.at("gate_id"), path + ".gate_id");
    result.has_compute_share = parse_nullable_projection_share(
        wire.at("compute_share"), path + ".compute_share", &result.compute_share);
    result.has_hbm_share = parse_nullable_projection_share(
        wire.at("hbm_share"), path + ".hbm_share", &result.hbm_share);
    if (result.kind == "fixed") {
        if (!result.route.empty() || result.has_group || result.latency_first_ns != 0 ||
            result.latency_rest_ns != 0 || result.has_compute_share || result.has_hbm_share) {
            fail(path, "fixed consume must have empty route, null group/shares, and zero latency");
        }
    } else {
        if (result.route.empty() || !result.has_group || result.group != "decode" ||
            !result.has_compute_share || !result.has_hbm_share) {
            fail(path, "compute consume requires route, decode group, and both shares");
        }
        uint64_t end = 0;
        if (result.compute_share.total_tiles != result.hbm_share.total_tiles ||
            result.compute_share.share_index_first != result.hbm_share.share_index_first ||
            result.compute_share.tile_bytes != result.hbm_share.tile_bytes ||
            result.compute_share.total_tiles != result.hbm_share.total_tiles ||
            !checked_add(result.compute_share.share_index_first, tile_count, &end) ||
            end > result.compute_share.total_tiles) {
            fail(path, "compute/HBM share ranges do not align with this chain");
        }
    }
    uint64_t ignored = 0, step = 0;
    if (!checked_add(result.first_index, tile_count, &ignored) ||
        !checked_mul(result.consume_order_stride, tile_count - 1, &step) ||
        !checked_add(result.consume_order_first, step, &ignored)) {
        fail(path, "consume affine ID/order range overflows uint64");
    }
    return result;
}

struct U64Progression final { uint64_t first, stride, count; };

__int128 extended_gcd(__int128 a, __int128 b, __int128* x, __int128* y) {
    if (b == 0) { *x = 1; *y = 0; return a; }
    __int128 next_x = 0, next_y = 0;
    const __int128 gcd = extended_gcd(b, a % b, &next_x, &next_y);
    *x = next_y;
    *y = next_x - (a / b) * next_y;
    return gcd;
}

bool progressions_intersect(const U64Progression& a,
                            const U64Progression& b) {
    if (a.count == 0 || b.count == 0) return false;
    const uint64_t a_last = a.first + a.stride * (a.count - 1);
    const uint64_t b_last = b.first + b.stride * (b.count - 1);
    const uint64_t low = std::max(a.first, b.first);
    const uint64_t high = std::min(a_last, b_last);
    if (low > high) return false;
    const uint64_t gcd = std::gcd(a.stride, b.stride);
    const __int128 delta = static_cast<__int128>(b.first) - a.first;
    if (delta % gcd != 0) return false;
    const uint64_t modulus = b.stride / gcd;
    uint64_t k = 0;
    if (modulus != 1) {
        __int128 inverse = 0, ignored = 0;
        extended_gcd(a.stride / gcd, modulus, &inverse, &ignored);
        __int128 residue = ((delta / gcd) * inverse) % modulus;
        if (residue < 0) residue += modulus;
        k = static_cast<uint64_t>(residue);
    }
    const unsigned __int128 period =
        static_cast<unsigned __int128>(a.stride / gcd) * b.stride;
    unsigned __int128 first =
        static_cast<unsigned __int128>(a.first) +
        static_cast<unsigned __int128>(a.stride) * k;
    first %= period;
    if (first < low) {
        const unsigned __int128 distance = low - first;
        first += ((distance + period - 1) / period) * period;
    }
    return first <= high;
}

PipelineProjection parse_projection(const Json& wire,
                                    const std::vector<PipelineGroup>& groups,
                                    const PipelineProtocol& protocol) {
    require_exact_keys(wire, "projection", {"version", "chains"});
    PipelineProjection result;
    result.version = get_u32(wire.at("version"), "projection.version");
    if (result.version != 1) fail("projection.version", "unsupported version");
    const auto& chain_array = wire.at("chains");
    require_array(chain_array, "projection.chains");
    result.chains.reserve(chain_array.size());
    std::map<uint64_t, const PipelineGroup*> group_by_id;
    for (const auto& group : groups) group_by_id.emplace(group.group_ordinal, &group);
    std::set<std::pair<uint64_t, int>> seen_chains;
    std::map<std::string, uint32_t> gate_capacities;
    std::map<std::string, std::vector<std::pair<uint64_t, uint64_t>>> id_ranges;
    std::vector<U64Progression> registration_orders;
    std::map<int, std::vector<std::tuple<uint64_t, uint64_t, uint64_t>>> rank_chains;

    for (size_t i = 0; i < chain_array.size(); ++i) {
        const std::string path = "projection.chains[" + std::to_string(i) + "]";
        const auto& item = chain_array[i];
        require_exact_keys(item, path,
                           {"group_ordinal", "rank", "tile_count", "fill", "consume"});
        PipelineProjectionChain chain;
        chain.group_ordinal = get_u64_key(item, "group_ordinal", path);
        chain.rank = get_rank(item.at("rank"), path + ".rank");
        chain.tile_count = get_u64_key(item, "tile_count", path);
        if (chain.tile_count == 0) fail(path + ".tile_count", "must be positive");
        const auto group_it = group_by_id.find(chain.group_ordinal);
        if (group_it == group_by_id.end()) fail(path, "unknown group_ordinal");
        const auto& group = *group_it->second;
        if (chain.tile_count != group.tile_axis.tile_count) {
            fail(path + ".tile_count", "must match group tile_count");
        }
        if (!seen_chains.emplace(chain.group_ordinal, chain.rank).second) {
            fail(path, "duplicate group/rank chain");
        }
        bool rank_has_target_shard = false;
        for (const auto& shard : group.shards) {
            if (shard.target_rank == chain.rank && shard.total_data_bytes > 0) {
                rank_has_target_shard = true;
            }
        }
        if (!rank_has_target_shard) fail(path, "rank has no target shard");
        chain.fill = parse_projection_fill(item.at("fill"), path + ".fill", chain.tile_count);
        chain.consume = parse_projection_consume(item.at("consume"), path + ".consume", chain.tile_count);
        if (chain.fill.first_index != chain.consume.first_index) {
            fail(path, "fill and consume index ranges must align");
        }
        if (chain.fill.gate_capacity != protocol.buffer_slots) {
            fail(path + ".fill.gate_capacity", "disagrees with buffer_slots");
        }
        if (chain.fill.gate_id != chain.consume.gate_id) {
            fail(path, "fill and consume gate IDs must match");
        }
        auto [capacity_it, inserted] = gate_capacities.emplace(
            chain.fill.gate_id, chain.fill.gate_capacity);
        if (!inserted && capacity_it->second != chain.fill.gate_capacity) {
            fail(path, "projection gate ID has conflicting capacities");
        }
        uint64_t fill_end = 0, consume_end = 0;
        if (!checked_add(chain.fill.first_index, chain.tile_count, &fill_end) ||
            !checked_add(chain.consume.first_index, chain.tile_count, &consume_end)) {
            fail(path, "projection leg ID range overflows uint64");
        }
        id_ranges[chain.fill.leg_id_prefix].emplace_back(chain.fill.first_index, fill_end);
        id_ranges[chain.consume.leg_id_prefix].emplace_back(chain.consume.first_index, consume_end);
        registration_orders.push_back({chain.fill.fill_order_first,
                                       chain.fill.fill_order_stride, chain.tile_count});
        registration_orders.push_back({chain.consume.consume_order_first,
                                       chain.consume.consume_order_stride, chain.tile_count});
        rank_chains[chain.rank].emplace_back(chain.group_ordinal,
            chain.fill.first_index, chain.tile_count);
        result.chains.push_back(std::move(chain));
    }
    for (auto& entry : id_ranges) {
        auto& ranges = entry.second;
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i) {
            if (ranges[i].first < ranges[i - 1].second) {
                fail("projection.chains", "leg ID ranges overlap for prefix '" + entry.first + "'");
            }
        }
    }
    auto check_orders = [&](const std::vector<U64Progression>& orders,
                            const std::string& label) {
        for (size_t i = 0; i < orders.size(); ++i) {
            for (size_t j = i + 1; j < orders.size(); ++j) {
                if (progressions_intersect(orders[i], orders[j])) {
                    fail("projection.chains", label + " progressions overlap");
                }
            }
        }
    };
    check_orders(registration_orders, "registration-order");
    for (auto& entry : rank_chains) {
        auto& chains = entry.second;
        std::sort(chains.begin(), chains.end());
        if (std::get<1>(chains.front()) != 0) {
            fail("projection.chains", "each rank's first chain must start at index zero");
        }
        uint64_t previous_end = 0;
        bool first = true;
        for (const auto& chain : chains) {
            const uint64_t start = std::get<1>(chain);
            const uint64_t count = std::get<2>(chain);
            if (!first && start != previous_end) {
                fail("projection.chains", "rank group chains are not contiguous");
            }
            if (!checked_add(start, count, &previous_end)) {
                fail("projection.chains", "rank chain range overflows uint64");
            }
            first = false;
        }
    }
    return result;
}

}  // namespace

PipelineKernelContract::PipelineKernelContract(
    uint32_t projection_abi_version, uint32_t descriptor_version,
    uint32_t pipeline_protocol_version, std::string model_version,
    std::string timer_whitelist_version, uint32_t window_tile_limit,
    uint32_t window_local_error_ppm, uint32_t window_refine_margin_ppm,
    std::string prediction_config_sha256)
    : projection_abi_version_(projection_abi_version),
      descriptor_version_(descriptor_version),
      pipeline_protocol_version_(pipeline_protocol_version),
      model_version_(std::move(model_version)),
      timer_whitelist_version_(std::move(timer_whitelist_version)),
      window_tile_limit_(window_tile_limit),
      window_local_error_ppm_(window_local_error_ppm),
      window_refine_margin_ppm_(window_refine_margin_ppm),
      prediction_config_sha256_(std::move(prediction_config_sha256)) {}

std::shared_ptr<const PipelineKernelContract> PipelineKernelContract::parse(
    const nlohmann::json& wire) {
    const std::string path = "kernel_contract";
    require_exact_keys(wire, path,
                       {"projection_abi_version", "descriptor_version",
                        "pipeline_protocol_version", "model_version",
                        "timer_whitelist_version", "window_tile_limit",
                        "window_local_error_ppm", "window_refine_margin_ppm",
                        "prediction_config_sha256"});
    auto result = std::shared_ptr<const PipelineKernelContract>(
        new PipelineKernelContract(
            get_u32(wire.at("projection_abi_version"), path + ".projection_abi_version"),
            get_u32(wire.at("descriptor_version"), path + ".descriptor_version"),
            get_u32(wire.at("pipeline_protocol_version"), path + ".pipeline_protocol_version"),
            get_nonempty_string(wire.at("model_version"), path + ".model_version"),
            get_nonempty_string(wire.at("timer_whitelist_version"),
                                path + ".timer_whitelist_version"),
            get_u32(wire.at("window_tile_limit"), path + ".window_tile_limit"),
            get_u32(wire.at("window_local_error_ppm"), path + ".window_local_error_ppm"),
            get_u32(wire.at("window_refine_margin_ppm"), path + ".window_refine_margin_ppm"),
            get_nonempty_string(wire.at("prediction_config_sha256"),
                                path + ".prediction_config_sha256")));
    result->validate();
    return result;
}

void PipelineKernelContract::validate() const {
    if (projection_abi_version_ != kProjectionAbiVersion ||
        descriptor_version_ != kDescriptorVersion ||
        pipeline_protocol_version_ != kPipelineProtocolVersion ||
        model_version_ != kModelVersion ||
        timer_whitelist_version_ != kTimerWhitelistVersion) {
        fail("kernel_contract", "unsupported ABI/model/protocol/whitelist version");
    }
    if (window_tile_limit_ != 1 && window_tile_limit_ != 2 &&
        window_tile_limit_ != 4 && window_tile_limit_ != 8) {
        fail("kernel_contract.window_tile_limit", "must be one of 1, 2, 4, 8");
    }
    if (window_local_error_ppm_ > 10000) {
        fail("kernel_contract.window_local_error_ppm", "must be in [0, 10000]");
    }
    if (window_refine_margin_ppm_ > 1000000) {
        fail("kernel_contract.window_refine_margin_ppm", "must be in [0, 1000000]");
    }
    if (prediction_config_sha256_.size() != 64 ||
        !std::all_of(prediction_config_sha256_.begin(), prediction_config_sha256_.end(),
                     [](unsigned char c) {
                         return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
                     })) {
        fail("kernel_contract.prediction_config_sha256",
             "must be 64 lowercase hexadecimal characters");
    }
}

bool PipelineKernelContract::operator==(
    const PipelineKernelContract& other) const {
    return projection_abi_version_ == other.projection_abi_version_ &&
           descriptor_version_ == other.descriptor_version_ &&
           pipeline_protocol_version_ == other.pipeline_protocol_version_ &&
           model_version_ == other.model_version_ &&
           timer_whitelist_version_ == other.timer_whitelist_version_ &&
           window_tile_limit_ == other.window_tile_limit_ &&
           window_local_error_ppm_ == other.window_local_error_ppm_ &&
           window_refine_margin_ppm_ == other.window_refine_margin_ppm_ &&
           prediction_config_sha256_ == other.prediction_config_sha256_;
}

PipelineDescriptor::PipelineDescriptor(
    PipelineIdentity identity, PipelineStreamKind stream_kind,
    std::vector<PipelineGroup> groups, PipelineProtocol protocol,
    PipelinePredictionProfile prediction_profile,
    PipelineExecutionProfile execution_profile,
    std::vector<PipelineLogicalIdRange> logical_id_ranges,
    std::vector<PipelineTagAllocationRun> tag_runs,
    std::vector<PipelineEndpointRun> endpoint_runs,
    std::vector<PipelineEndpointDependencyRun> endpoint_dependency_runs,
    std::vector<PipelineConsumeCompletionRun> consume_completion_runs,
    PipelineProjection projection,
    std::vector<PipelineAnchor> anchors, nlohmann::json wire)
    : identity_(std::move(identity)),
      stream_kind_(stream_kind),
      groups_(std::move(groups)),
      protocol_(protocol),
      prediction_profile_(std::move(prediction_profile)),
      execution_profile_(execution_profile),
      logical_id_ranges_(std::move(logical_id_ranges)),
      tag_allocation_runs_(std::move(tag_runs)),
      endpoint_runs_(std::move(endpoint_runs)),
      endpoint_dependency_runs_(std::move(endpoint_dependency_runs)),
      consume_completion_runs_(std::move(consume_completion_runs)),
      projection_(std::move(projection)),
      anchors_(std::move(anchors)), wire_(std::move(wire)) {}

uint64_t PipelineDescriptor::endpoint_json_id(
    const PipelineEndpointRun& run, const PipelineEndpointLegRun& leg,
    uint64_t tile_ordinal, uint64_t shard_ordinal) const {
    if (tile_ordinal < run.first_tile ||
        tile_ordinal - run.first_tile >= run.tile_count) {
        fail("endpoint_json_id", "tile ordinal is outside endpoint run");
    }
    auto shard_it = std::find(run.shard_ordinals.begin(),
                              run.shard_ordinals.end(), shard_ordinal);
    if (shard_it == run.shard_ordinals.end()) {
        fail("endpoint_json_id", "shard ordinal is outside endpoint run");
    }
    if (std::none_of(run.legs.begin(), run.legs.end(), [&](const auto& item) {
            return item.endpoint == leg.endpoint;
        })) {
        fail("endpoint_json_id", "leg is not registered in endpoint run");
    }
    const uint64_t tile_delta = tile_ordinal - run.first_tile;
    const uint64_t shard_index =
        static_cast<uint64_t>(std::distance(run.shard_ordinals.begin(), shard_it));
    uint64_t id = run.first_json_id;
    uint64_t term = 0;
    if (!checked_mul(tile_delta, run.tile_stride, &term) ||
        !checked_add(id, term, &id) ||
        !checked_mul(shard_index, run.shard_stride, &term) ||
        !checked_add(id, term, &id) ||
        !checked_add(id, leg.json_offset, &id)) {
        fail("endpoint_json_id", "affine JSON ID calculation overflows uint64");
    }
    return id;
}

uint32_t PipelineDescriptor::endpoint_tag(
    const PipelineEndpointRun& run, const PipelineEndpointLegRun& leg,
    uint64_t tile_ordinal, uint64_t shard_ordinal) const {
    if (!leg.has_tag_fields) {
        fail("endpoint_tag", "body_marker endpoint has no tag");
    }
    if (tile_ordinal < run.first_tile ||
        tile_ordinal - run.first_tile >= run.tile_count) {
        fail("endpoint_tag", "tile ordinal is outside endpoint run");
    }
    auto shard_it = std::find(run.shard_ordinals.begin(),
                              run.shard_ordinals.end(), shard_ordinal);
    if (shard_it == run.shard_ordinals.end()) {
        fail("endpoint_tag", "shard ordinal is outside endpoint run");
    }
    const auto tag_run = std::find_if(
        tag_allocation_runs_.begin(), tag_allocation_runs_.end(),
        [&](const auto& item) {
            return item.emission_ordinal == leg.tag_emission_ordinal &&
                   item.group_ordinal == run.group_ordinal;
        });
    if (tag_run == tag_allocation_runs_.end()) {
        fail("endpoint_tag", "unknown tag-allocation emission ordinal");
    }
    const uint64_t tile_delta = tile_ordinal - run.first_tile;
    const uint64_t shard_index =
        static_cast<uint64_t>(std::distance(run.shard_ordinals.begin(), shard_it));
    const uint64_t shard_end = tag_run->first_shard + tag_run->shard_count;
    if (tile_ordinal < tag_run->first_tile ||
        tile_ordinal - tag_run->first_tile >= tag_run->tile_count ||
        shard_ordinal < tag_run->first_shard || shard_ordinal >= shard_end) {
        fail("endpoint_tag", "endpoint is outside its tag-allocation run");
    }
    uint64_t offset = leg.tag_offset;
    uint64_t term = 0;
    if (!checked_mul(tile_delta, leg.tag_tile_stride, &term) ||
        !checked_add(offset, term, &offset) ||
        !checked_mul(shard_index, leg.tag_shard_stride, &term) ||
        !checked_add(offset, term, &offset) || offset >= tag_run->count) {
        fail("endpoint_tag", "affine tag offset is outside its allocation run");
    }
    const uint64_t tag = static_cast<uint64_t>(tag_run->first_tag) + offset;
    if (tag == 0 || tag > std::numeric_limits<uint32_t>::max()) {
        fail("endpoint_tag", "tag exceeds uint32 allocator domain");
    }
    return static_cast<uint32_t>(tag);
}

std::shared_ptr<const PipelineDescriptor> PipelineDescriptor::parse(
    const nlohmann::json& wire) {
    const std::string path = "pipeline descriptor";
    require_exact_keys(wire, path,
                       {"descriptor_version", "identity", "stream_kind", "groups",
                       "protocol", "prediction_profile", "execution_profile",
                        "logical_id_ranges", "tag_allocation_runs",
                        "endpoint_runs", "endpoint_dependency_runs",
                        "consume_completion_runs", "projection", "anchors"});
    if (get_u32(wire.at("descriptor_version"), path + ".descriptor_version") !=
        PipelineKernelContract::kDescriptorVersion) {
        fail(path + ".descriptor_version", "unsupported descriptor version");
    }
    PipelineIdentity identity = parse_identity(wire.at("identity"));
    const auto stream_kind = parse_stream_kind(wire.at("stream_kind"), path + ".stream_kind");
    if ((identity.stage == PipelineStage::Prefill) !=
        (stream_kind == PipelineStreamKind::PrefillReadWindow)) {
        fail(path + ".stream_kind", "does not match identity.stage");
    }
    const auto& groups_wire = wire.at("groups");
    require_array(groups_wire, path + ".groups", true);
    std::vector<PipelineGroup> groups;
    groups.reserve(groups_wire.size());
    std::set<uint64_t> group_ids;
    for (size_t i = 0; i < groups_wire.size(); ++i) {
        auto group = parse_group(groups_wire[i], i);
        if (!group_ids.insert(group.group_ordinal).second) {
            fail(path + ".groups", "duplicate group_ordinal");
        }
        if (!groups.empty() && group.group_ordinal <= groups.back().group_ordinal) {
            fail(path + ".groups", "group_ordinal must preserve strict builder order");
        }
        groups.push_back(std::move(group));
    }
    auto protocol = parse_protocol(wire.at("protocol"));
    auto prediction_profile = parse_prediction_profile(wire.at("prediction_profile"));
    auto execution_profile = parse_execution_profile(wire.at("execution_profile"));
    auto id_ranges = parse_id_ranges(wire.at("logical_id_ranges"));
    auto tag_runs = parse_tag_runs(wire.at("tag_allocation_runs"));
    auto endpoint_runs = parse_endpoint_runs(wire.at("endpoint_runs"));
    auto endpoint_dependency_runs =
        parse_endpoint_dependency_runs(wire.at("endpoint_dependency_runs"));
    auto consume_completion_runs =
        parse_consume_completion_runs(wire.at("consume_completion_runs"));
    auto projection = parse_projection(wire.at("projection"), groups, protocol);
    for (const auto& chain : projection.chains) {
        if (chain.rank != identity.rank) {
            fail("projection.chains", "chain rank must equal descriptor owner rank");
        }
    }
    auto anchors = parse_anchors(wire.at("anchors"));
    const bool projection_only = endpoint_runs.empty() && tag_runs.empty() &&
        id_ranges.empty() && !projection.chains.empty();
    if (projection_only && !anchors.empty()) {
        fail("anchors", "projection-only descriptors must not synthesize execution anchors");
    }
    if (!projection_only && anchors.empty()) {
        fail("anchors", "execution descriptors require original start/frontier and service/settle anchors");
    }

    std::set<uint64_t> groups_by_id;
    for (const auto& group : groups) groups_by_id.insert(group.group_ordinal);
    for (const auto& run : tag_runs) {
        if (groups_by_id.count(run.group_ordinal) == 0) {
            fail("tag_allocation_runs", "run references unknown group_ordinal");
        }
        const auto it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) {
            return g.group_ordinal == run.group_ordinal;
        });
        uint64_t end = 0;
        uint64_t shard_end = 0;
        if (!checked_add(run.first_tile, run.tile_count, &end) ||
            !checked_add(run.first_shard, run.shard_count, &shard_end) ||
            run.first_tile < 1 || end - 1 > it->tile_axis.tile_count ||
            shard_end > it->shards.size()) {
            fail("tag_allocation_runs", "run tile/shard range is outside its group");
        }
    }
    for (const auto& anchor : anchors) {
        if (anchor.binding != PipelineAnchorBinding::PhysicalEndpointSlot) continue;
        for (const auto& range : id_ranges) {
            if (range.rank != anchor.rank ||
                anchor.node_id < range.first_json_id) {
                continue;
            }
            const uint64_t delta = anchor.node_id - range.first_json_id;
            if (delta % range.stride == 0 && delta / range.stride < range.count) {
                fail("anchors", "physical anchor node_id is inside a virtual ID hole");
            }
        }
        if (anchor.selector.kind == PipelineSelectorKind::ExistingServiceGate) {
            continue;
        }
        const auto group_it = std::find_if(groups.begin(), groups.end(), [&](const auto& g) {
            return g.group_ordinal == anchor.selector.group_ordinal;
        });
        if (group_it == groups.end()) {
            fail("anchors", "physical selector references unknown group");
        }
        if (anchor.selector.tile_ordinal > group_it->tile_axis.tile_count) {
            fail("anchors", "physical selector tile is outside group");
        }
        const auto shard_it = std::find_if(
            group_it->shards.begin(), group_it->shards.end(),
            [&](const auto& shard) {
                return shard.shard_ordinal == anchor.selector.shard_ordinal;
            });
        if (shard_it == group_it->shards.end()) {
            fail("anchors", "physical selector references unknown shard");
        }
        const auto leg = anchor.selector.endpoint;
        const int expected_rank =
            (leg == PipelineEndpointLeg::DataSend ||
             leg == PipelineEndpointLeg::FillAckRecv ||
             leg == PipelineEndpointLeg::ReleaseAckRecv)
                ? shard_it->source_rank
                : shard_it->target_rank;
        if (leg != PipelineEndpointLeg::BodyMarker && anchor.rank != expected_rank) {
            fail("anchors", "physical endpoint anchor rank disagrees with shard endpoint");
        }
        if (anchor.selector.kind == PipelineSelectorKind::InternalMarkerDone) {
            // Marker selectors map to the original, untagged body marker run.
            // Their operation charge remains in the executor; the structural
            // COMP anchor itself is zero-cost.
            continue;
        }
        const auto run_it = std::find_if(tag_runs.begin(), tag_runs.end(), [&](const auto& run) {
            return run.emission_ordinal ==
                       anchor.selector.tag_allocation_emission_ordinal &&
                   run.group_ordinal == anchor.selector.group_ordinal &&
                   anchor.selector.tile_ordinal >= run.first_tile &&
                   anchor.selector.tile_ordinal - run.first_tile < run.tile_count &&
                   anchor.selector.shard_ordinal >= run.first_shard &&
                   anchor.selector.shard_ordinal - run.first_shard < run.shard_count;
        });
        if (run_it == tag_runs.end() || anchor.selector.tag_offset >= run_it->count) {
            fail("anchors", "physical endpoint tag selector is outside its allocated run");
        }
    }

    auto descriptor = std::shared_ptr<const PipelineDescriptor>(
        new PipelineDescriptor(std::move(identity), stream_kind, std::move(groups),
                               protocol, std::move(prediction_profile),
                               execution_profile, std::move(id_ranges),
                               std::move(tag_runs), std::move(endpoint_runs),
                               std::move(endpoint_dependency_runs),
                               std::move(consume_completion_runs),
                               std::move(projection), std::move(anchors), wire));
    (void)validate_endpoint_runs(*descriptor);
    validate_endpoint_dependency_runs(
        *descriptor, descriptor->endpoint_dependency_runs());
    validate_consume_completion_runs(
        *descriptor, descriptor->consume_completion_runs());
    return descriptor;
}

}  // namespace ExecutionDriven
}  // namespace AstraSim
