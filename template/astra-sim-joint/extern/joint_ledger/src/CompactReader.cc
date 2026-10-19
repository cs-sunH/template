// CompactReader.cc -- workledger-compact-v1 frame decoder + validator.
//
// One sequential walk: magic/header/pool, then the five sections in wire
// order (descriptors/nodes/runs/cursors/resources).  Field layout mirrors
// the encoder in CompactWire.cc; the validations mirror the v4 row
// validators in RowParsers.cc with identity-derived fields reconstructed
// instead of compared (the reconstruction IS the compared value of the
// legacy accepted domain).
#include "joint_ledger/CompactReader.hh"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "joint_ledger/CompactWire.hh"
#include "joint_ledger/LedgerTypes.hh"

namespace jointledger {

// Cross-TU helpers of RowParsers.cc (external linkage, rowport namespace).
namespace rowport {
bool python_equal(const Json& left, const Json& right);
}

namespace {

[[noreturn]] void fail(const std::string& what) {
    throw LedgerError("compact frame: " + what);
}

// read_rank over the compact varint domain (the legacy reader validates the
// wire JSON scalar; the varint is unsigned by construction).
std::uint64_t read_rank_json_wrap(std::uint64_t value);

// ------------------------------------------------------------------ views --

struct DescriptorView {
    const std::unordered_map<std::string, DescriptorPtr>* declared = nullptr;
    const std::unordered_map<std::string, DescriptorPtr>* retained = nullptr;
    const DescriptorPtr* find(const std::string& segment) const {
        if (declared != nullptr) {
            auto it = declared->find(segment);
            if (it != declared->end()) return &it->second;
        }
        if (retained != nullptr) {
            auto it = retained->find(segment);
            if (it != retained->end()) return &it->second;
        }
        return nullptr;
    }
};

// ------------------------------------------------------------ resolution --

// resolve a compact (group, tile, shard, endpoint, rank) selector to exactly
// one descriptor endpoint (parse_one_pipeline_run's typed-rank path).
struct Resolution {
    std::uint64_t run_index = 0;
    std::uint64_t leg_index = 0;
    std::uint64_t json_id = 0;
    std::uint64_t group = 0;
    std::uint64_t tile = 0;
    std::uint64_t shard = 0;
};

Resolution resolve(const PipelineDescriptor& descriptor,
                   const SelectorIndex& index,
                   const EndpointSelector& selector,
                   std::uint64_t rank, const std::string& path) {
    const auto found = index.selector_intervals.find(std::make_tuple(
        selector.group_ordinal, rank, selector.shard_ordinal,
        selector.endpoint));
    if (found != index.selector_intervals.end()) {
        const std::vector<std::uint64_t>& starts = found->second.first;
        const auto position = std::upper_bound(starts.begin(), starts.end(),
                                               selector.tile_ordinal);
        if (position != starts.begin()) {
            const std::size_t at =
                static_cast<std::size_t>(position - starts.begin()) - 1;
            const auto& row = found->second.second[at];
            if (row[0] <= selector.tile_ordinal
                    && selector.tile_ordinal < row[1]) {
                Resolution out;
                out.run_index = row[2];
                out.leg_index = row[3];
                out.json_id = descriptor.endpoint_id(
                    static_cast<std::size_t>(out.run_index),
                    selector.tile_ordinal, selector.shard_ordinal,
                    static_cast<std::size_t>(out.leg_index));
                out.group = selector.group_ordinal;
                out.tile = selector.tile_ordinal;
                out.shard = selector.shard_ordinal;
                return out;
            }
        }
    }
    fail(path + " endpoint selector must map to exactly one descriptor "
               "endpoint");
}

// expected_endpoint_amount (RowParsers.cc) reconstructed for the compact
// path: {ops,bytes,latency_ns} of a validated endpoint quantity.
WireAmount endpoint_amount(const PipelineDescriptor& descriptor,
                           std::size_t run_index, std::uint64_t ordinal,
                           const PipelineShardTile& shard_tile) {
    const PipelineEndpointRun& run = descriptor.endpoint_runs[run_index];
    const std::uint64_t per_tile = run.shard_ordinals.size()
        * run.legs.size();
    const std::uint64_t within = ordinal % per_tile;
    const std::size_t leg_index = static_cast<std::size_t>(
        within % run.legs.size());
    const std::string& leg = run.legs[leg_index].endpoint;
    WireAmount out;
    if (leg == "body_marker") {
        out.ops = WireNum::of_uint(1);
        out.bytes = WireNum::of_uint(1);
        return out;
    }
    std::uint64_t amount = shard_tile.data_bytes;
    if (leg == "fill_ack_send" || leg == "fill_ack_recv") {
        amount = descriptor.protocol.fill_ack_bytes;
    } else if (leg == "release_ack_send" || leg == "release_ack_recv") {
        amount = descriptor.protocol.release_ack_bytes;
    }
    out.bytes = WireNum::of_uint(amount);
    return out;
}

WireNum read_num(WireCursor& cursor, const std::vector<std::string>& pool,
                 const char* what) {
    const std::uint8_t tag = cursor.u8(what);
    switch (tag) {
        case 0:
            return WireNum::null_num();
        case 1:
            return WireNum::of_uint(cursor.var(what));
        case 2:
            return WireNum::of_double(cursor.f64(what));
        default:
            fail(std::string(what) + " has an unknown number tag");
    }
}

// --------------------------------------------------------------- decode ---

class FrameReader {
public:
    FrameReader(const char* data, std::size_t length,
                const RetainedRegistry& retained, bool allow_retained_replay,
                std::uint64_t expected_delivery, std::uint64_t expected_tick)
        : retained_(retained),
          allow_retained_replay_(allow_retained_replay),
          expected_delivery_(expected_delivery),
          expected_tick_(expected_tick) {
        if (!is_compact_frame(data, length)) {
            fail("segment does not start with the compact magic");
        }
        const std::size_t whole = compact_frame_length(data, length);
        const std::uint8_t version = static_cast<std::uint8_t>(data[4]);
        if (version != kCompactWireVersion) {
            fail("unsupported compact wire version");
        }
        const std::uint8_t flags = static_cast<std::uint8_t>(data[5]);
        if (flags & ~0x01) fail("unknown frame flags");
        full_ = (flags & 0x01) != 0;
        WireCursor header(data + 10, whole - 10);
        delivery_ = header.var("delivery_sequence");
        tick_ = header.var("tick");
        base_watermark_ = header.var("base_watermark");
        seq_start_ = header.var("seq_start");
        seq_end_ = header.var("seq_end");
        event_count_ = header.var("event_count");
        base_version_ = header.var("base_state_version");
        result_version_ = header.var("result_state_version");
        {
            const std::uint64_t epoch_len = header.var("epoch length");
            header.bytes(static_cast<std::size_t>(epoch_len), epoch_,
                         "epoch");
            if (epoch_.empty()) fail("epoch must be nonempty");
            const std::uint64_t schema_len = header.var("schema name length");
            std::string schema;
            header.bytes(static_cast<std::size_t>(schema_len), schema,
                         "schema name");
            if (schema != kCompactSchemaName) {
                fail("unexpected compact schema identity");
            }
        }
        header.load_pool(pool_);
        cursor_ = header;   // sections continue on the same cursor
    }

    ParsedEnvelope read_all();

private:
    // ---- section helpers ----
    std::uint64_t section(std::uint8_t expected) {
        const std::uint8_t tag = cursor_.u8("section tag");
        if (tag != expected) fail("unexpected section order");
        return cursor_.var("section count");
    }
    const std::string& str(std::uint64_t ref, const char* what) {
        return cursor_.pool_str(pool_, ref, what);
    }
    const DescriptorPtr* find_descriptor(const std::string& segment) const {
        if (declared_.count(segment)) return &declared_.at(segment);
        if (retained_.descriptors != nullptr) {
            const auto it = retained_.descriptors->find(segment);
            if (it != retained_.descriptors->end()) return &it->second;
        }
        return nullptr;
    }
    const SelectorIndex* selector_index_for(const std::string& segment) {
        const auto declared = selector_additions_.find(segment);
        if (declared != selector_additions_.end()) return &declared->second;
        if (retained_.selector_indexes != nullptr) {
            const auto it = retained_.selector_indexes->find(segment);
            if (it != retained_.selector_indexes->end()) return &it->second;
        }
        return nullptr;
    }

    void read_descriptors();
    void read_nodes();
    void read_runs();
    void read_cursors();
    void read_resources();

    // run record readers
    SlotPatch read_slot_patch(const std::string& path,
                              const PipelineDescriptor& descriptor);
    CreditPatch read_credit_patch(const std::string& path,
                                  const PipelineDescriptor& descriptor);
    ObligationPatch read_obligation_patch(const std::string& path,
                                          const PipelineDescriptor& descriptor);
    void validate_slot_row(const SlotRow& row, const std::string& path,
                           const PipelineDescriptor*& owner_descriptor);
    CursorRun read_cursor_run(const std::string& path,
                              const PipelineDescriptor& descriptor,
                              std::uint64_t run_index);
    ResourceRow read_resource_row(const std::string& path);

    // ---- parse inputs/state ----
    const RetainedRegistry& retained_;
    bool allow_retained_replay_ = false;
    std::uint64_t expected_delivery_ = 0;
    std::uint64_t expected_tick_ = 0;
    WireCursor cursor_{nullptr, 0};
    std::vector<std::string> pool_;
    bool full_ = false;
    std::string epoch_;
    std::uint64_t delivery_ = 0;
    std::uint64_t tick_ = 0;
    std::uint64_t base_watermark_ = 0;
    std::uint64_t seq_start_ = 0;
    std::uint64_t seq_end_ = 0;
    std::uint64_t event_count_ = 0;
    std::uint64_t base_version_ = 0;
    std::uint64_t result_version_ = 0;
    std::vector<std::pair<DescriptorPtr, Json>> commit_list_;
    std::unordered_map<std::string, DescriptorPtr> declared_;
    std::unordered_set<std::string> seen_segments_;
    std::unordered_map<std::string, SelectorIndex> selector_additions_;
    std::unordered_map<std::string, DependencyIndex> dependency_additions_;
    std::vector<std::pair<std::uint64_t, NodeEvent>> node_events_;
    bool have_previous_node_ = false;
    std::uint64_t previous_node_seq_ = 0;
    std::vector<PipelineRun> runs_;
    bool have_previous_run_ = false;
    std::uint64_t previous_run_seq_ = 0;
    std::vector<CursorPatch> cursor_patches_;
    bool have_previous_cursor_segment_ = false;
    std::string previous_cursor_segment_;
    std::vector<ResourceRow> resources_;
    std::optional<std::tuple<std::string, std::uint64_t, std::uint64_t,
                             std::string, std::string>> previous_resource_;
};

void FrameReader::read_descriptors() {
    const std::uint64_t count = section(1);
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::string path =
            "descriptor_commits[" + std::to_string(i) + "]";
        const std::uint64_t size = cursor_.var("descriptor length");
        const std::string bytes = cursor_.raw_bytes(
            static_cast<std::size_t>(size), "descriptor declaration");
        Json element;
        try {
            element = Json::parse(bytes);
        } catch (const Json::parse_error& error) {
            fail(path + " is not valid JSON: "
                 + std::string(error.what()));
        }
        const Json& commit = read_mapping(element, path);
        const std::vector<const char*> key_list = {
            "identity", "descriptor", "commit_tick", "cursor_version",
            "service_done", "settle_done", "retired"};
        require_keys(commit, path, key_list, key_list);
        DescriptorPtr descriptor =
            PipelineDescriptor::from_wire(commit.at("descriptor"));
        const PipelineIdentity& ident = descriptor->identity;
        const Json& identity = read_mapping(commit.at("identity"),
                                            path + ".identity");
        const std::vector<const char*> identity_keys = {
            "request_id", "stage", "generation", "stage_epoch",
            "segment_ordinal", "rank", "staging_allocation_id", "segment_id",
            "buffer_domain_id"};
        require_keys(identity, path + ".identity", identity_keys,
                     identity_keys);
        if (!rowport::python_equal(identity,
                                   descriptor->to_wire().at("identity"))) {
            fail(path + ".identity differs from immutable descriptor");
        }
        if (seen_segments_.count(ident.segment_id) != 0) {
            fail("duplicate descriptor declaration '" + ident.segment_id
                 + "'");
        }
        seen_segments_.insert(ident.segment_id);
        read_uint(commit.at("commit_tick"), path + ".commit_tick");
        read_uint(commit.at("cursor_version"), path + ".cursor_version");
        read_bool(commit.at("service_done"), path + ".service_done");
        read_bool(commit.at("settle_done"), path + ".settle_done");
        read_bool(commit.at("retired"), path + ".retired");
        commit_list_.emplace_back(std::move(descriptor), std::move(element));
    }
    // Registry merge (EnvelopeReader finalize_commits).
    for (const auto& entry : commit_list_) {
        const std::string& segment_id = entry.first->identity.segment_id;
        const DescriptorPtr* prior = nullptr;
        if (retained_.descriptors != nullptr) {
            const auto found = retained_.descriptors->find(segment_id);
            if (found != retained_.descriptors->end()) prior = &found->second;
        }
        if (prior != nullptr
                && !descriptor_wire_equal(**prior, *entry.first)) {
            fail("immutable descriptor changed after declaration");
        }
        if (prior != nullptr && !allow_retained_replay_) {
            fail("descriptor segment was declared more than once");
        }
        declared_[segment_id] = entry.first;
    }
    for (const auto& entry : declared_) {
        const bool has_retained_index =
            retained_.selector_indexes != nullptr
            && retained_.selector_indexes->find(entry.first)
                != retained_.selector_indexes->end();
        if (!has_retained_index) {
            selector_additions_.emplace(entry.first,
                                        build_selector_index(*entry.second));
        }
    }
    for (const auto& entry : declared_) {
        const bool has_retained_descriptor =
            retained_.descriptors != nullptr
            && retained_.descriptors->find(entry.first)
                != retained_.descriptors->end();
        if (!has_retained_descriptor) {
            dependency_additions_.emplace(
                entry.first, build_dependency_index(*entry.second));
        }
    }
}

void FrameReader::read_nodes() {
    static const char* const kAllowedPayloadKeys[] = {
        "delta_served", "state", "request_id", "stage", "generation", "rank",
        "node_id", "operation", "leg", "num_ops", "bytes", "runtime_ns",
        "tensor_size", "total_bytes", "dependencies", "parents",
        "service_unit", "route", "flow_id", "port_index", "issue_seq",
        "total", "total_latency_ns", "release_conditions", "remaining_ops",
        "remaining_bytes", "remaining_latency_ns"};
    static const char* const kRemainingKeys[] = {"remaining_ops",
                                                 "remaining_bytes",
                                                 "remaining_latency_ns"};
    const std::uint64_t count = section(2);
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::string path = "node_transitions["
            + std::to_string(i) + "]";
        const std::uint8_t kind_byte = cursor_.u8("node kind");
        if (kind_byte > static_cast<std::uint8_t>(NodeKind::Release)) {
            fail(path + ".kind is unknown");
        }
        const std::string kind = node_kind_name(
            static_cast<NodeKind>(kind_byte));
        const std::uint64_t seq = cursor_.var("node logical_seq");
        const std::uint64_t node_tick = cursor_.var("node tick");
        const std::uint64_t work_ref = cursor_.var("node work_id ref");
        const std::string work_id = str(work_ref, "node work_id");
        const std::uint64_t payload_len = cursor_.var("node payload length");
        const std::string payload_bytes = cursor_.raw_bytes(
            static_cast<std::size_t>(payload_len), "node payload");
        if (seq == 0) {
            fail(path + " logical sequence/tick is invalid");
        }
        if (node_tick > tick_) {
            fail(path + " logical sequence/tick is invalid");
        }
        if (work_id.empty()) {
            fail(path + " requires a work_id");
        }
        Json payload;
        try {
            payload = Json::parse(payload_bytes);
        } catch (const Json::parse_error& error) {
            fail(path + ".payload is not valid JSON: "
                 + std::string(error.what()));
        }
        if (!payload.is_object()) {
            fail(path + ".payload must be a string-keyed object");
        }
        for (auto member = payload.begin(); member != payload.end();
             ++member) {
            bool allowed = false;
            for (const char* key : kAllowedPayloadKeys) {
                if (member.key() == key) {
                    allowed = true;
                    break;
                }
            }
            if (!allowed) {
                fail(path + ".payload has unknown key '" + member.key()
                     + "'");
            }
        }
        const std::string kind_text = kind;
        if (kind_text != "commit" && kind_text != "progress") {
            for (const char* key : kRemainingKeys) {
                if (payload.find(key) != payload.end()) {
                    fail("remaining_* fields are only valid on commit/"
                         "progress events");
                }
            }
        }
        NodeEvent typed;
        typed.kind = kind;
        typed.work_id = work_id;
        typed.tick = node_tick;
        typed.producer = "cpp_online";
        typed.local_seq = seq;
        if (const auto delta_field = payload.find("delta_served");
                delta_field != payload.end() && !delta_field->is_null()) {
            if (!delta_field->is_object()) {
                fail("unknown work-amount keys");
            }
            static const char* const dims[] = {"ops", "bytes",
                                               "latency_ns"};
            WorkAmount delta;
            std::optional<double> WorkAmount::*const members[3] = {
                &WorkAmount::ops, &WorkAmount::bytes,
                &WorkAmount::latency_ns};
            for (const auto& item : delta_field->items()) {
                bool known = false;
                for (const char* dim : dims) {
                    if (item.key() == dim) known = true;
                }
                if (!known) fail("unknown work-amount keys");
            }
            for (int d = 0; d < 3; ++d) {
                const auto member = delta_field->find(dims[d]);
                if (member == delta_field->end()
                        || member->is_null()) continue;
                if (member->is_boolean()
                        || !(member->is_number_integer()
                             || member->is_number_unsigned()
                             || member->is_number_float())) {
                    fail("work amount must be a finite number");
                }
                if (member->is_number_float()
                        && !std::isfinite(member->get<double>())) {
                    fail("work amount must be a finite number");
                }
                const double number = member->get<double>();
                if (number < -1e-6) {
                    fail(std::string("work amount '") + dims[d]
                         + "' must be non-negative");
                }
                delta.*members[d] = number;
            }
            typed.delta_served = delta;
        }
        typed.payload = std::move(payload);
        if (have_previous_node_ && seq <= previous_node_seq_) {
            fail("node_transitions must preserve unique logical sequence "
                 "order");
        }
        have_previous_node_ = true;
        previous_node_seq_ = seq;
        node_events_.emplace_back(seq, std::move(typed));
    }
}

void FrameReader::validate_slot_row(const SlotRow& row,
                                    const std::string& path,
                                    const PipelineDescriptor*& owner_out) {
    owner_out = nullptr;
    if (!row.occupied) {
        if (row.consumed || !row.target_shards_to_consume.empty()
                || !row.consumed_target_shards.empty()) {
            fail(path + " empty slot cannot retain consume state");
        }
    }
    if (row.role_source
            && (!row.target_shards_to_consume.empty()
                || !row.consumed_target_shards.empty() || row.consumed)) {
        fail(path + " source slot cannot carry target consume state");
    }
    if (row.occupied && !row.role_source
            && row.target_shards_to_consume.empty()) {
        fail(path + " occupied target slot needs consumption authority");
    }
    const auto sorted_unique = [](const std::vector<std::uint64_t>& values) {
        for (std::size_t i = 1; i < values.size(); ++i) {
            if (values[i - 1] >= values[i]) return false;
        }
        return true;
    };
    if (!sorted_unique(row.target_shards_to_consume)
            || !sorted_unique(row.consumed_target_shards)) {
        fail(path + " target shard arrays must be sorted unique");
    }
    for (const std::uint64_t value : row.consumed_target_shards) {
        if (!std::binary_search(row.target_shards_to_consume.begin(),
                                row.target_shards_to_consume.end(), value)) {
            fail(path + ".consumed_target_shards is not a subset");
        }
    }
    if (row.consumed != !row.consumed_target_shards.empty()) {
        fail(path + ".consumed must match consumed target shards");
    }
    if (row.role_source != row.has_shard) {
        fail(path + ".shard_ordinal nullability disagrees with role");
    }
    if (row.slot_index >= 2) {
        fail(path + ".slot_index is outside its stable key domain");
    }
    if (row.occupied && !row.owner) {
        fail(path + " occupied slot lacks its owner");
    }
    if (!row.occupied && row.owner) {
        fail(path + " empty slot cannot carry an owner");
    }
    if (row.owner) {
        const std::string& segment = row.owner->segment_id;
        const DescriptorPtr* found = find_descriptor(segment);
        if (found == nullptr) {
            fail(path + ".owner references unknown segment '" + segment
                 + "'");
        }
        const PipelineDescriptor& descriptor = **found;
        const PipelineIdentity& ident = descriptor.identity;
        if (row.domain != ident.buffer_domain_id
                || row.owner->pipeline_generation != ident.generation
                || row.owner->stage_epoch != ident.stage_epoch
                || row.owner->staging_allocation_id
                       != ident.staging_allocation_id) {
            fail(path + ".owner disagrees with immutable descriptor "
                       "identity");
        }
        const SelectorIndex* index = selector_index_for(segment);
        if (index == nullptr) {
            fail(path + ".owner has no selector index");
        }
        const auto group_entry = index->groups_by_ordinal.find(
            row.owner->group_ordinal);
        if (group_entry == index->groups_by_ordinal.end()) {
            fail(path + ".owner points outside descriptor group/tile range");
        }
        const PipelineGroup& group =
            descriptor.groups[group_entry->second];
        if (row.owner->tile_ordinal < 1
                || row.owner->tile_ordinal > group.tile_axis.tile_count) {
            fail(path + ".owner points outside descriptor group/tile range");
        }
        const std::uint64_t global_tile_ordinal =
            group.tile_axis.global_tile_first + row.owner->tile_ordinal - 1;
        if (row.slot_index != (global_tile_ordinal - 1) % 2) {
            fail(path + ".slot_index disagrees with two-slot tile parity");
        }
        if (row.role_source) {
            // Interval-index run resolution (the JSON validator's
            // descriptor_run_at): the (group, rank) tile interval that
            // contains the owner tile names the owning run.
            std::optional<std::uint64_t> run_index;
            const auto found_runs = index->run_intervals.find(
                std::make_tuple(row.owner->group_ordinal, row.rank));
            if (found_runs != index->run_intervals.end()) {
                const std::vector<std::uint64_t>& starts =
                    found_runs->second.first;
                const auto position = std::upper_bound(
                    starts.begin(), starts.end(), row.owner->tile_ordinal);
                if (position != starts.begin()) {
                    const std::size_t at = static_cast<std::size_t>(
                        position - starts.begin()) - 1;
                    const auto& interval = found_runs->second.second[at];
                    if (interval[0] <= row.owner->tile_ordinal
                            && row.owner->tile_ordinal < interval[1]) {
                        run_index = interval[2];
                    }
                }
            }
            const PipelineEndpointRun* owner_run = run_index.has_value()
                ? &descriptor.endpoint_runs[
                      static_cast<std::size_t>(*run_index)]
                : nullptr;
            if (owner_run == nullptr
                    || std::find(owner_run->shard_ordinals.begin(),
                                 owner_run->shard_ordinals.end(),
                                 row.shard)
                        == owner_run->shard_ordinals.end()) {
                fail(path + " source shard is absent from the owner's "
                           "endpoint run");
            }
        } else {
            if (row.rank != ident.rank) {
                fail(path + " target slot rank disagrees with descriptor "
                           "target rank");
            }
            for (const std::uint64_t target_shard_ordinal :
                 row.target_shards_to_consume) {
                if (target_shard_ordinal >= group.shards.size()) {
                    fail(path + " target consumption shards disagree with "
                               "descriptor tile");
                }
                const std::pair<std::uint64_t, PipelineShardTile> selected =
                    descriptor.selected_shard_tile(row.owner->group_ordinal,
                                                   row.owner->tile_ordinal
                                                       - 1,
                                                   target_shard_ordinal);
                if (!selected.second.emit_endpoint
                        || selected.second.target_rank != ident.rank) {
                    fail(path + " target consumption shards disagree with "
                               "descriptor tile");
                }
            }
        }
        owner_out = &descriptor;
    }
}

SlotPatch FrameReader::read_slot_patch(const std::string& path,
                                       const PipelineDescriptor& descriptor) {
    SlotPatch patch;
    const std::uint8_t form = cursor_.u8("slot patch form");
    if (form == 1) {
        patch.is_target_delta = true;
        const std::uint64_t domain_ref = cursor_.var("domain ref");
        patch.key.domain = str(domain_ref, "target delta domain");
        patch.key.rank = read_rank_json_wrap(cursor_.var("rank"));
        patch.key.slot_index = cursor_.var("slot_index");
        patch.key.role_source = false;
        patch.key.shard = -1;
        if (patch.key.slot_index >= 2) {
            fail(path + ".key.slot_index is outside its stable key domain");
        }
        patch.before_slot_generation =
            cursor_.var("before_slot_generation");
        patch.predecessor_logical_seq =
            cursor_.var("predecessor_logical_seq");
        patch.predecessor_consumed_count =
            cursor_.var("predecessor_consumed_count");
        const std::uint64_t op_count = cursor_.var("operation count");
        if (op_count == 0) {
            fail(path + ".operations must be nonempty");
        }
        patch.operations.reserve(static_cast<std::size_t>(op_count));
        for (std::uint64_t i = 0; i < op_count; ++i) {
            const std::string op_path = path + ".operations["
                + std::to_string(i) + "]";
            TargetOp op;
            const std::uint8_t op_kind = cursor_.u8("op kind");
            if (op_kind
                    > static_cast<std::uint8_t>(
                        TargetOp::Op::Erase)) {
                fail(op_path + ".op is unknown");
            }
            op.op = static_cast<TargetOp::Op>(op_kind);
            op.slot_generation = cursor_.var("slot_generation");
            switch (op.op) {
                case TargetOp::Op::Initialize:
                case TargetOp::Op::RequestCloseClear:
                case TargetOp::Op::Erase:
                    break;
                case TargetOp::Op::Claim: {
                    SlotOwner owner;
                    const std::uint64_t owner_ref = cursor_.var("owner ref");
                    owner.segment_id = str(owner_ref, "claim owner segment");
                    owner.group_ordinal = cursor_.var("owner group_ordinal");
                    owner.tile_ordinal = cursor_.var("owner tile_ordinal");
                    const DescriptorPtr* claim_descriptor =
                        find_descriptor(owner.segment_id);
                    if (claim_descriptor == nullptr) {
                        fail(op_path + ".owner references unknown segment '"
                             + owner.segment_id + "'");
                    }
                    owner.pipeline_generation =
                        (*claim_descriptor)->identity.generation;
                    owner.stage_epoch =
                        (*claim_descriptor)->identity.stage_epoch;
                    owner.staging_allocation_id =
                        (*claim_descriptor)->identity.staging_allocation_id;
                    const std::uint64_t shard_count =
                        cursor_.var("target_shards count");
                    for (std::uint64_t s = 0; s < shard_count; ++s) {
                        op.target_shards.push_back(
                            cursor_.var("target_shard"));
                    }
                    const auto sorted_unique = [](
                            const std::vector<std::uint64_t>& values) {
                        for (std::size_t k = 1; k < values.size(); ++k) {
                            if (values[k - 1] >= values[k]) return false;
                        }
                        return true;
                    };
                    if (!sorted_unique(op.target_shards)) {
                        fail(op_path
                             + ".target_shards must be sorted unique");
                    }
                    op.owner = std::move(owner);
                    break;
                }
                case TargetOp::Op::Consume: {
                    op.terminal_rank = read_rank_json_wrap(
                        cursor_.var("terminal_rank"));
                    op.terminal_store_id =
                        cursor_.var("terminal_store_id");
                    op.consume_run_index = cursor_.var("consume_run_index");
                    const std::uint64_t shard_count =
                        cursor_.var("add_shards count");
                    for (std::uint64_t s = 0; s < shard_count; ++s) {
                        op.add_shards.push_back(cursor_.var("add_shard"));
                    }
                    for (std::size_t k = 1; k < op.add_shards.size(); ++k) {
                        if (op.add_shards[k - 1] >= op.add_shards[k]) {
                            fail(op_path
                                 + ".add_shards must be sorted unique");
                        }
                    }
                    op.cleared = cursor_.u8("cleared flag") != 0;
                    break;
                }
            }
            patch.operations.push_back(std::move(op));
        }
    } else if (form == 0) {
        patch.is_target_delta = false;
        patch.key.domain = str(cursor_.var("domain ref"), "slot domain");
        patch.key.rank = read_rank_json_wrap(cursor_.var("rank"));
        const std::uint8_t has_shard = cursor_.u8("shard presence");
        patch.key.shard = has_shard
            ? static_cast<std::int64_t>(cursor_.var("shard")) : -1;
        patch.key.slot_index = cursor_.var("slot_index");
        patch.key.role_source = true;
        if (patch.key.slot_index >= 2) {
            fail(path + ".slot_index is outside its stable key domain");
        }
        const auto read_row = [&](std::optional<SlotRow>& target) {
            const std::uint8_t present = cursor_.u8("row presence");
            if (!present) return;
            SlotRow row;
            row.domain = patch.key.domain;
            row.role_source = true;
            row.rank = patch.key.rank;
            row.has_shard = patch.key.shard >= 0;
            row.shard = patch.key.shard >= 0
                ? static_cast<std::uint64_t>(patch.key.shard) : 0;
            row.slot_index = patch.key.slot_index;
            row.occupied = cursor_.u8("occupied flag") != 0;
            if (row.occupied) {
                SlotOwner owner;
                const std::uint64_t owner_ref = cursor_.var("owner ref");
                owner.segment_id = str(owner_ref, "slot owner segment");
                owner.group_ordinal = cursor_.var("owner group_ordinal");
                owner.tile_ordinal = cursor_.var("owner tile_ordinal");
                // generation/epoch/allocation are identity-derived: the
                // owner's descriptor carries them (validated below).
                const DescriptorPtr* owner_descriptor =
                    find_descriptor(owner.segment_id);
                if (owner_descriptor == nullptr) {
                    fail(path + ".owner references unknown segment '"
                         + owner.segment_id + "'");
                }
                owner.pipeline_generation =
                    (*owner_descriptor)->identity.generation;
                owner.stage_epoch = (*owner_descriptor)->identity.stage_epoch;
                owner.staging_allocation_id =
                    (*owner_descriptor)->identity.staging_allocation_id;
                row.owner = std::move(owner);
            }
            row.consumed = cursor_.u8("consumed flag") != 0;
            const std::uint64_t targets = cursor_.var("targets count");
            for (std::uint64_t t = 0; t < targets; ++t) {
                row.target_shards_to_consume.push_back(
                    cursor_.var("target shard"));
            }
            const std::uint64_t consumed = cursor_.var("consumed count");
            for (std::uint64_t t = 0; t < consumed; ++t) {
                row.consumed_target_shards.push_back(
                    cursor_.var("consumed shard"));
            }
            const PipelineDescriptor* owner_descriptor = nullptr;
            validate_slot_row(row, path + ".row", owner_descriptor);
            (void)owner_descriptor;
            target = std::move(row);
        };
        read_row(patch.before);
        read_row(patch.after);
        if (!patch.before.has_value() && !patch.after.has_value()) {
            fail(path + " cannot have both before and after null");
        }
        if (patch.before.has_value() && patch.after.has_value()
                && patch.before->typed_eq(*patch.after)) {
            fail(path + " is a no-op patch");
        }
    } else {
        fail(path + " has an unknown patch form");
    }
    return patch;
}

CreditPatch FrameReader::read_credit_patch(
        const std::string& path,
        const PipelineDescriptor& descriptor) {
    (void)descriptor;
    CreditPatch patch;
    patch.key.domain = str(cursor_.var("domain ref"), "credit domain");
    patch.key.source_rank = read_rank_json_wrap(
        cursor_.var("source_rank"));
    patch.key.shard = cursor_.var("shard_ordinal");
    const auto read_row = [&](std::optional<CreditRow>& target) {
        const std::uint8_t present = cursor_.u8("row presence");
        if (!present) return;
        CreditRow row;
        row.domain = patch.key.domain;
        row.source_rank = patch.key.source_rank;
        row.shard = patch.key.shard;
        row.capacity = 2;
        row.available = cursor_.var("available");
        row.outstanding_ack_count = cursor_.var("outstanding_ack_count");
        if (row.capacity != 2 || row.available > row.capacity) {
            fail(path + " requires capacity=2 and available in [0,2]");
        }
        target = row;
    };
    read_row(patch.before);
    read_row(patch.after);
    if (!patch.before.has_value() && !patch.after.has_value()) {
        fail(path + " cannot have both before and after null");
    }
    if (patch.before.has_value() && patch.after.has_value()
            && patch.before->typed_eq(*patch.after)) {
        fail(path + " is a no-op patch");
    }
    return patch;
}

ObligationPatch FrameReader::read_obligation_patch(
        const std::string& path,
        const PipelineDescriptor& descriptor) {
    ObligationPatch patch;
    const std::uint64_t segment_ref = cursor_.var("segment ref");
    const std::string segment = str(segment_ref, "obligation segment");
    const DescriptorPtr* found = find_descriptor(segment);
    if (found == nullptr) {
        fail(path + " references unknown segment");
    }
    const std::uint64_t run_index = cursor_.var("run_index");
    if (run_index >= descriptor.endpoint_runs.size()) {
        fail(path + " run index exceeds descriptor runs");
    }
    patch.obligation_id = segment + "/run/" + std::to_string(run_index);
    const auto read_row = [&](std::optional<ObligationRow>& target) {
        const std::uint8_t present = cursor_.u8("row presence");
        if (!present) return;
        ObligationRow row;
        row.obligation_id = patch.obligation_id;
        row.segment_id = segment;
        row.rank = descriptor.endpoint_runs[
            static_cast<std::size_t>(run_index)].rank;
        row.first_store_id = cursor_.var("first_store_id");
        row.remaining_endpoint_count =
            cursor_.var("remaining_endpoint_count");
        const std::uint8_t state = cursor_.u8("obligation state");
        if (state > static_cast<std::uint8_t>(ObligationState::Active)) {
            fail(path + " state must be pending or active");
        }
        row.state = static_cast<ObligationState>(state);
        if (row.remaining_endpoint_count == 0) {
            fail(path + " must be pending/active with positive remaining "
                       "count");
        }
        target = row;
    };
    read_row(patch.before);
    read_row(patch.after);
    if (!patch.before.has_value() && !patch.after.has_value()) {
        fail(path + " cannot have both before and after null");
    }
    if (patch.before.has_value() && patch.after.has_value()
            && patch.before->typed_eq(*patch.after)) {
        fail(path + " is a no-op patch");
    }
    return patch;
}

void FrameReader::read_runs() {
    const std::uint64_t count = section(3);
    runs_.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::string path = "pipeline_transition_runs["
            + std::to_string(i) + "]";
        PipelineRun run;
        run.first_logical_seq = cursor_.var("first_logical_seq");
        if (run.first_logical_seq == 0) {
            fail(path + ".first_logical_seq must start at 1");
        }
        const std::uint8_t kind_byte = cursor_.u8("transition kind");
        if (kind_byte
                > static_cast<std::uint8_t>(
                    TransitionKind::DescriptorRetire)) {
            fail(path + ".transition_kind is unknown");
        }
        run.transition_kind = static_cast<TransitionKind>(kind_byte);
        const std::uint64_t segment_ref = cursor_.var("segment ref");
        run.descriptor_segment_id = str(segment_ref, "run segment");
        const DescriptorPtr* descriptor_ptr =
            find_descriptor(run.descriptor_segment_id);
        if (descriptor_ptr == nullptr) {
            fail(path + " references undeclared descriptor '"
                 + run.descriptor_segment_id + "'");
        }
        const PipelineDescriptor& descriptor = **descriptor_ptr;
        const SelectorIndex* index =
            selector_index_for(run.descriptor_segment_id);
        if (index == nullptr) {
            fail(path + " has no selector index");
        }
        const std::uint8_t flags = cursor_.u8("run flags");
        if (flags & ~0x03) fail(path + " has unknown flags");
        run.have_selector = (flags & 0x01) != 0;
        const bool has_leg = (flags & 0x02) != 0;
        Resolution resolution;
        if (has_leg) {
            const std::uint8_t leg_byte = cursor_.u8("resource leg");
            if (leg_byte > static_cast<std::uint8_t>(
                    ResourceLeg::HbmCommWrite)) {
                fail(path + ".resource_leg is unknown");
            }
            run.resource_leg = static_cast<ResourceLeg>(leg_byte);
        }
        if (run.have_selector) {
            run.endpoint_selector.group_ordinal =
                cursor_.var("selector group_ordinal");
            run.endpoint_selector.tile_ordinal =
                cursor_.var("selector tile_ordinal");
            run.endpoint_selector.shard_ordinal =
                cursor_.var("selector shard_ordinal");
            const std::uint64_t endpoint_ref = cursor_.var("endpoint ref");
            run.endpoint_selector.endpoint = str(endpoint_ref,
                                                 "selector endpoint");
            run.selector_rank = read_rank_json_wrap(cursor_.var("rank"));
            resolution = resolve(descriptor, *index, run.endpoint_selector,
                                 run.selector_rank, path);
            const PipelineEndpointRun& endpoint_run =
                descriptor.endpoint_runs[
                    static_cast<std::size_t>(resolution.run_index)];
            if (run.selector_rank != endpoint_run.rank) {
                fail(path + " affine selector axes disagree");
            }
            run.has_endpoint_run_index = true;
            run.endpoint_run_index = resolution.run_index;
            run.has_endpoint_leg_index = true;
            run.endpoint_leg_index = resolution.leg_index;
            run.has_endpoint_store_id = true;
            run.endpoint_store_id = resolution.json_id + 1;
            const auto shard_position = std::find(
                endpoint_run.shard_ordinals.begin(),
                endpoint_run.shard_ordinals.end(), resolution.shard);
            if (shard_position == endpoint_run.shard_ordinals.end()) {
                fail(path + " affine selector axes disagree");
            }
            run.has_endpoint_ordinal = true;
            run.endpoint_ordinal =
                (resolution.tile - endpoint_run.first_tile)
                    * endpoint_run.shard_ordinals.size()
                    * endpoint_run.legs.size()
                + static_cast<std::uint64_t>(shard_position
                    - endpoint_run.shard_ordinals.begin())
                    * endpoint_run.legs.size()
                + resolution.leg_index;
            const std::pair<std::uint64_t, PipelineShardTile> selected =
                descriptor.selected_shard_tile(resolution.group,
                                               resolution.tile - 1,
                                               resolution.shard);
            run.has_selected_global_tile_ordinal = true;
            run.selected_global_tile_ordinal = selected.first;
            run.has_selected_shard_tile = true;
            run.selected_shard_tile.shard_ordinal =
                selected.second.shard_ordinal;
            run.selected_shard_tile.source_rank = selected.second.source_rank;
            run.selected_shard_tile.target_rank = selected.second.target_rank;
            run.selected_shard_tile.data_bytes = selected.second.data_bytes;
            run.selected_shard_tile.emit_endpoint =
                selected.second.emit_endpoint;
            if (run.resource_leg != ResourceLeg::None) {
                // The reconstructed quantity IS the descriptor endpoint
                // amount (RowParsers compares the wire value against it);
                // the marker/kind agreement is still validated.
                const WireAmount expected = endpoint_amount(
                    descriptor,
                    static_cast<std::size_t>(resolution.run_index),
                    run.endpoint_ordinal, selected.second);
                (void)expected;
                const std::string& endpoint_name =
                    descriptor.endpoint_runs[
                        static_cast<std::size_t>(resolution.run_index)]
                        .legs[static_cast<std::size_t>(
                            resolution.leg_index)].endpoint;
                const bool leg_is_marker =
                    run.resource_leg == ResourceLeg::BodyMarker;
                if (leg_is_marker != (endpoint_name == "body_marker")) {
                    fail(path
                         + ".resource_leg disagrees with descriptor endpoint "
                            "kind");
                }
            }
        } else {
            // descriptor-level rows must not bind resources (wire omits the
            // quantities; the leg presence itself is the binding).
            if (has_leg) {
                const bool descriptor_level =
                    run.transition_kind == TransitionKind::DescriptorCommit
                    || run.transition_kind == TransitionKind::ServiceDone
                    || run.transition_kind == TransitionKind::RequestClose
                    || run.transition_kind == TransitionKind::SettleDone
                    || run.transition_kind
                           == TransitionKind::DescriptorRetire
                    || (run.transition_kind
                            == TransitionKind::ConsumeDone
                        && !run.have_selector);
                if (descriptor_level) {
                    fail(path + " descriptor-level transition cannot bind an "
                                "endpoint or resource");
                }
                fail(path + " resource quantities require an endpoint "
                            "selector");
            }
        }
        run.tick_first = cursor_.var("tick_first");
        run.cursor_version_start = cursor_.var("cursor_version_start");
        run.cursor_version_end = cursor_.var("cursor_version_end");
        if (run.cursor_version_end < run.cursor_version_start
                || run.cursor_version_end - run.cursor_version_start > 1) {
            fail(path + " cursor version transition must be stable or "
                        "increment by one");
        }
        const std::uint64_t slot_count = cursor_.var("slot patch count");
        bool have_previous_slot = false;
        SlotKey previous_slot;
        for (std::uint64_t s = 0; s < slot_count; ++s) {
            const std::string slot_path = path + ".slot_patches["
                + std::to_string(s) + "]";
            SlotPatch patch = read_slot_patch(slot_path, descriptor);
            if (have_previous_slot
                    && !(previous_slot < patch.key)) {
                fail(path + ".slot_patches must be sorted and "
                            "duplicate-free");
            }
            have_previous_slot = true;
            previous_slot = patch.key;
            if (patch.key.domain
                    != descriptor.identity.buffer_domain_id) {
                fail(slot_path + " crosses the transition buffer domain");
            }
            run.slot_patches.push_back(std::move(patch));
        }
        const std::uint64_t credit_count = cursor_.var("credit patch count");
        bool have_previous_credit = false;
        CreditKey previous_credit;
        for (std::uint64_t c = 0; c < credit_count; ++c) {
            const std::string credit_path = path + ".credit_patches["
                + std::to_string(c) + "]";
            CreditPatch patch = read_credit_patch(credit_path, descriptor);
            if (have_previous_credit
                    && !(previous_credit < patch.key)) {
                fail(path + ".credit_patches must be sorted and "
                            "duplicate-free");
            }
            have_previous_credit = true;
            previous_credit = patch.key;
            if (patch.key.domain
                    != descriptor.identity.buffer_domain_id) {
                fail(credit_path + " crosses the transition buffer domain");
            }
            run.credit_patches.push_back(std::move(patch));
        }
        const std::uint64_t obligation_count =
            cursor_.var("obligation patch count");
        bool have_previous_obligation = false;
        std::string previous_obligation;
        for (std::uint64_t o = 0; o < obligation_count; ++o) {
            const std::string obligation_path =
                path + ".obligation_patches["
                + std::to_string(o) + "]";
            ObligationPatch patch = read_obligation_patch(obligation_path,
                                                          descriptor);
            if (have_previous_obligation
                    && !(previous_obligation < patch.obligation_id)) {
                fail(path + ".obligation_patches must be sorted and "
                            "duplicate-free");
            }
            have_previous_obligation = true;
            previous_obligation = patch.obligation_id;
            run.obligation_patches.push_back(std::move(patch));
        }
        // Transition-kind binding constraints (parse_one_pipeline_run tail).
        const bool descriptor_level =
            run.transition_kind == TransitionKind::DescriptorCommit
            || run.transition_kind == TransitionKind::ServiceDone
            || run.transition_kind == TransitionKind::RequestClose
            || run.transition_kind == TransitionKind::SettleDone
            || run.transition_kind == TransitionKind::DescriptorRetire
            || (run.transition_kind == TransitionKind::ConsumeDone
                && !run.have_selector);
        if (descriptor_level
                && (run.have_selector
                    || run.resource_leg != ResourceLeg::None)) {
            fail(path + " descriptor-level transition cannot bind an "
                        "endpoint or resource");
        }
        if (run.transition_kind == TransitionKind::EndpointDone
                && !run.have_selector) {
            fail(path + " endpoint_done transition requires a concrete "
                        "selector");
        }
        if (run.transition_kind != TransitionKind::DescriptorCommit
                && run.transition_kind != TransitionKind::ServiceDone
                && run.transition_kind != TransitionKind::RequestClose
                && run.transition_kind != TransitionKind::SettleDone
                && run.transition_kind != TransitionKind::DescriptorRetire
                && !run.have_selector
                && run.transition_kind != TransitionKind::ConsumeDone) {
            fail(path + " endpoint transition requires a concrete selector");
        }
        if (run.resource_leg == ResourceLeg::None) {
            // quantities are reconstructed; nothing to cross-check here
        } else if (!run.have_selector) {
            fail(path + " resource quantities require a resource_leg "
                        "bound endpoint");
        }
        if (have_previous_run_ && run.first_logical_seq <= previous_run_seq_) {
            fail("pipeline_transition_runs must preserve logical sequence "
                 "order");
        }
        have_previous_run_ = true;
        previous_run_seq_ = run.first_logical_seq;
        runs_.push_back(std::move(run));
    }
}

CursorRun FrameReader::read_cursor_run(const std::string& path,
                                       const PipelineDescriptor& descriptor,
                                       std::uint64_t run_index) {
    const PipelineEndpointRun& expected =
        descriptor.endpoint_runs.at(static_cast<std::size_t>(run_index));
    CursorRun run;
    run.rank = expected.rank;
    run.group_ordinal = expected.group_ordinal;
    run.first_tile = expected.first_tile;
    run.tile_count = expected.tile_count;
    run.completed_endpoint_count = cursor_.var("completed_endpoint_count");
    run.cancelled_endpoint_count = cursor_.var("cancelled_endpoint_count");
    run.tile_offset = cursor_.var("tile_offset");
    run.shard_index = cursor_.var("shard_index");
    run.leg_index = cursor_.var("leg_index");
    const std::uint64_t shard_count = expected.shard_ordinals.size();
    const std::uint64_t leg_count = expected.legs.size();
    const std::uint64_t total_count = run.tile_count * shard_count
        * leg_count;
    if (total_count == 0 || run.tile_offset > run.tile_count) {
        fail(path + " has an invalid endpoint cursor extent");
    }
    if (run.tile_offset == run.tile_count) {
        if (run.shard_index != 0 || run.leg_index != 0) {
            fail(path + " exhausted cursor position must have null next "
                        "store");
        }
    } else if (run.shard_index >= shard_count
               || run.leg_index >= leg_count) {
        fail(path + " cursor position exceeds run axes");
    }
    run.has_initial_parent = expected.initial_parent_json_id.has_value();
    run.initial_parent_store_id =
        expected.initial_parent_json_id
            ? *expected.initial_parent_json_id + 1 : 0;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> completed_ranges;
    if (run.completed_endpoint_count != 0) {
        completed_ranges.emplace_back(
            std::uint64_t{0}, run.completed_endpoint_count);
    }
    std::uint64_t out_of_order_count = 0;
    std::optional<std::uint64_t> last_end;
    const std::uint64_t ooo_count =
        cursor_.var("completed_out_of_order count");
    for (std::uint64_t i = 0; i < ooo_count; ++i) {
        const std::string p = path + ".completed_out_of_order["
            + std::to_string(i) + "]";
        const std::uint64_t begin = cursor_.var("first_ordinal");
        const std::uint64_t end = cursor_.var("end_ordinal_exclusive");
        if (begin >= end || begin < run.completed_endpoint_count
                || end > total_count
                || (last_end.has_value() && begin < *last_end)) {
            fail(p + " is empty, overlaps, or precedes completed prefix");
        }
        last_end = end;
        out_of_order_count += end - begin;
        completed_ranges.emplace_back(begin, end);
        run.completed_out_of_order.emplace_back(begin, end);
    }
    run.remaining_endpoint_count = total_count
        - run.completed_endpoint_count - out_of_order_count
        - run.cancelled_endpoint_count;
    const std::uint64_t active_count = cursor_.var("active count");
    std::set<std::uint64_t> seen_store_ids;
    std::set<std::uint64_t> seen_transfer_ids;
    for (std::uint64_t i = 0; i < active_count; ++i) {
        const std::string p = path + ".active_endpoints["
            + std::to_string(i) + "]";
        const std::uint64_t ordinal = cursor_.var("active ordinal");
        const std::uint64_t transfer_id = cursor_.var("transfer_id");
        const std::uint8_t flow_bound = cursor_.u8("flow bound flag");
        const std::uint64_t flow_id =
            flow_bound ? cursor_.var("flow_id") : 0;
        const bool network_done = cursor_.u8("network_done flag") != 0;
        const bool hbm_done = cursor_.u8("hbm_done flag") != 0;
        if (ordinal >= total_count || shard_count * leg_count == 0) {
            fail(p + " endpoint position is outside its run");
        }
        const std::uint64_t tile_offset = ordinal / (shard_count
                                                     * leg_count);
        const std::uint64_t within_tile = ordinal
            % (shard_count * leg_count);
        const std::uint64_t shard_position = within_tile / leg_count;
        const std::uint64_t leg_position = within_tile % leg_count;
        const std::uint64_t active_tile = expected.first_tile + tile_offset;
        const std::uint64_t active_shard =
            expected.shard_ordinals[static_cast<std::size_t>(
                shard_position)];
        const std::string endpoint = expected.legs[
            static_cast<std::size_t>(leg_position)].endpoint;
        const std::uint64_t store_id = descriptor.endpoint_id(
            static_cast<std::size_t>(run_index), active_tile, active_shard,
            static_cast<std::size_t>(leg_position)) + 1;
        if (!seen_store_ids.insert(store_id).second) {
            fail(path + " duplicates active store_id");
        }
        if (!seen_transfer_ids.insert(transfer_id).second) {
            fail(path + " duplicates active transfer_id");
        }
        const auto contains = [&completed_ranges](
                std::uint64_t value) {
            const auto position = std::upper_bound(
                completed_ranges.begin(), completed_ranges.end(),
                std::make_pair(value, U64_MAX));
            if (position == completed_ranges.begin()) return false;
            const auto& candidate = *(position - 1);
            return value < candidate.second;
        };
        if (contains(ordinal)) {
            fail(p + " is simultaneously complete and active");
        }
        if (endpoint != "body_marker") {
            const std::pair<std::uint64_t, PipelineShardTile> selected =
                descriptor.selected_shard_tile(expected.group_ordinal,
                                               active_tile - 1,
                                               active_shard);
            if (!selected.second.emit_endpoint) {
                fail(p + " maps to an inactive stream-bytes endpoint");
            }
        }
        ActiveEndpoint typed;
        typed.ordinal = ordinal;
        typed.store_id = store_id;
        typed.tile_ordinal = active_tile;
        typed.shard_ordinal = active_shard;
        typed.endpoint = endpoint;
        typed.transfer_id = transfer_id;
        typed.flow_bound = flow_bound != 0;
        typed.flow_id = flow_id;
        typed.network_done = network_done;
        typed.hbm_done = hbm_done;
        typed.owner = SlotOwner{
            descriptor.identity.segment_id, descriptor.identity.generation,
            descriptor.identity.stage_epoch,
            descriptor.identity.staging_allocation_id,
            expected.group_ordinal, active_tile};
        run.active_endpoints.push_back(std::move(typed));
    }
    run.in_flight = !run.active_endpoints.empty();
    if (run.tile_offset < run.tile_count) {
        const std::uint64_t tile_ordinal = expected.first_tile
            + run.tile_offset;
        const std::uint64_t shard =
            expected.shard_ordinals[static_cast<std::size_t>(
                run.shard_index)];
        const PipelineEndpointLegRun& leg = expected.legs[
            static_cast<std::size_t>(run.leg_index)];
        if (leg.endpoint != "body_marker") {
            const std::pair<std::uint64_t, PipelineShardTile> selected =
                descriptor.selected_shard_tile(expected.group_ordinal,
                                               tile_ordinal - 1, shard);
            if (!selected.second.emit_endpoint) {
                fail(path + " cursor position points to an inactive "
                            "endpoint");
            }
        }
        run.has_next_store = true;
        run.next_store_id = descriptor.endpoint_id(
            static_cast<std::size_t>(run_index), tile_ordinal, shard,
            static_cast<std::size_t>(run.leg_index)) + 1;
    }
    return run;
}

void FrameReader::read_cursors() {
    const std::uint64_t count = section(4);
    cursor_patches_.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::string path = "cursor_patches["
            + std::to_string(i) + "]";
        CursorPatch patch;
        const std::uint64_t segment_ref = cursor_.var("segment ref");
        patch.segment_id = str(segment_ref, "cursor segment");
        const DescriptorPtr* found = find_descriptor(patch.segment_id);
        if (found == nullptr) {
            fail(path + " references unknown descriptor segment");
        }
        const PipelineDescriptor& descriptor = **found;
        patch.cursor_version = cursor_.var("cursor_version");
        patch.retired = cursor_.u8("retired flag") != 0;
        const std::uint64_t run_count = cursor_.var("run count");
        std::optional<std::uint64_t> previous_index;
        for (std::uint64_t r = 0; r < run_count; ++r) {
            const std::string q = path + ".runs["
                + std::to_string(r) + "]";
            const std::uint64_t ordinal = cursor_.var("run_index");
            if (ordinal >= descriptor.endpoint_runs.size()) {
                fail(q + ".run_index exceeds descriptor runs");
            }
            const std::uint8_t present = cursor_.u8("run presence");
            if (present) {
                patch.runs.emplace_back(
                    ordinal, read_cursor_run(q + ".value", descriptor,
                                             ordinal));
            } else {
                patch.runs.emplace_back(ordinal, std::nullopt);
            }
            if (!present && !patch.retired) {
                fail(q + ".value may be null only during descriptor "
                         "retirement");
            }
            if (present && patch.retired) {
                fail(q + ".value must be null on descriptor retirement");
            }
            if (previous_index.has_value() && ordinal <= *previous_index) {
                fail(path + ".runs must be sorted and duplicate-free");
            }
            previous_index = ordinal;
        }
        if (patch.retired) {
            const std::uint64_t total = descriptor.endpoint_runs.size();
            bool tombstoned = patch.runs.size() == total;
            if (tombstoned && !patch.runs.empty()) {
                tombstoned = patch.runs.front().first == 0
                    && patch.runs.back().first == total - 1;
            }
            if (tombstoned) {
                for (const auto& entry : patch.runs) {
                    if (entry.second.has_value()) tombstoned = false;
                }
            }
            if (!tombstoned) {
                fail(path + " retirement must tombstone every descriptor "
                            "cursor run");
            }
        }
        if (have_previous_cursor_segment_
                && patch.segment_id <= previous_cursor_segment_) {
            fail("cursor_patches must be sorted and duplicate-free");
        }
        have_previous_cursor_segment_ = true;
        previous_cursor_segment_ = patch.segment_id;
        cursor_patches_.push_back(std::move(patch));
    }
}

ResourceRow FrameReader::read_resource_row(const std::string& p) {
    ResourceRow row;
    const std::uint8_t kind_byte = cursor_.u8("resource kind");
    if (kind_byte > static_cast<std::uint8_t>(ResourceKind::PoolJob)) {
        fail(p + ".resource_type is unknown");
    }
    row.kind = static_cast<ResourceKind>(kind_byte);
    const std::uint64_t request_ref = cursor_.var("request_id ref");
    row.request_id = str(request_ref, "resource request_id");
    const std::uint64_t stage_ref = cursor_.var("stage ref");
    row.stage = str(stage_ref, "resource stage");
    row.generation = cursor_.var("generation");
    row.rank = read_rank_json_wrap(cursor_.var("rank"));
    row.node_id = cursor_.var("node_id");
    const std::uint64_t operation_ref = cursor_.var("operation ref");
    row.operation = str(operation_ref, "resource operation");
    const std::uint64_t leg_ref = cursor_.var("leg ref");
    row.leg = str(leg_ref, "resource leg");
    const std::uint8_t state_index = cursor_.u8("resource state");
    row.state = ResourceRow::state_name(row.kind, state_index);
    row.total.ops = read_num(cursor_, pool_, "total ops");
    row.total.bytes = read_num(cursor_, pool_, "total bytes");
    row.total.latency_ns = read_num(cursor_, pool_, "total latency_ns");
    row.remaining.ops = read_num(cursor_, pool_, "remaining ops");
    row.remaining.bytes = read_num(cursor_, pool_, "remaining bytes");
    row.remaining.latency_ns = read_num(cursor_, pool_,
                                        "remaining latency_ns");
    const std::uint64_t route_count = cursor_.var("route count");
    for (std::uint64_t r = 0; r < route_count; ++r) {
        row.route.push_back(cursor_.var("route hop"));
    }
    row.as_of_tick = cursor_.var("as_of_tick");
    const std::uint8_t issued = cursor_.u8("issued presence");
    row.has_issued_tick = issued != 0;
    if (row.has_issued_tick) row.issued_tick = cursor_.var("issued_tick");
    const std::uint8_t bound = cursor_.u8("bound presence");
    row.bound = bound != 0;
    if (row.bound) {
        const std::uint64_t segment_ref = cursor_.var("segment ref");
        row.descriptor_segment_id = str(segment_ref,
                                        "resource descriptor segment");
        row.has_endpoint_selector = true;
        row.endpoint_selector.group_ordinal =
            cursor_.var("selector group_ordinal");
        row.endpoint_selector.tile_ordinal =
            cursor_.var("selector tile_ordinal");
        row.endpoint_selector.shard_ordinal =
            cursor_.var("selector shard_ordinal");
        const std::uint64_t endpoint_ref = cursor_.var("endpoint ref");
        row.endpoint_selector.endpoint = str(endpoint_ref,
                                             "selector endpoint");
    }
    switch (row.kind) {
        case ResourceKind::NocFlow:
            row.flow_id = cursor_.var("flow_id");
            row.current_rate_Bpns = read_num(cursor_, pool_,
                                             "current_rate_Bpns");
            row.last_rate_update_time = read_num(cursor_, pool_,
                                                 "last_rate_update_time");
            break;
        case ResourceKind::HbmJob: {
            const std::uint64_t kind_ref = cursor_.var("job kind ref");
            row.job_kind = str(kind_ref, "hbm job kind");
            row.start_tick = cursor_.var("start_tick");
            break;
        }
        case ResourceKind::PoolJob:
            row.port_index = cursor_.var("port_index");
            row.issue_seq = cursor_.var("issue_seq");
            row.latency_ready_ns = read_num(cursor_, pool_,
                                            "latency_ready_ns");
            break;
    }
    return row;
}

void FrameReader::read_resources() {
    const std::uint64_t count = section(5);
    resources_.reserve(static_cast<std::size_t>(count));
    for (std::uint64_t i = 0; i < count; ++i) {
        const std::string path = "active_resources["
            + std::to_string(i) + "]";
        ResourceRow row = read_resource_row(path);
        const PipelineDescriptor* descriptor = nullptr;
        if (row.bound) {
            const DescriptorPtr* found =
                find_descriptor(row.descriptor_segment_id);
            if (found == nullptr) {
                fail(path + " references unknown descriptor");
            }
            descriptor = found->get();
            const SelectorIndex* index =
                selector_index_for(row.descriptor_segment_id);
            if (index == nullptr) {
                fail(path + " has no selector index");
            }
            const Resolution resolution = resolve(
                *descriptor, *index, row.endpoint_selector, row.rank, path);
            const PipelineEndpointRun& run =
                descriptor->endpoint_runs[
                    static_cast<std::size_t>(resolution.run_index)];
            const PipelineLegacyContext& context = run.legacy_context;
            if (row.node_id != resolution.json_id + 1
                    || row.rank != run.rank) {
                fail(path + " pipeline NodeStore mapping is inconsistent");
            }
            if (row.request_id != context.request_id
                    || row.stage != context.stage
                    || row.generation != context.generation) {
                fail(path + " pipeline context disagrees with endpoint "
                            "legacy_context");
            }
            const std::string& leg = row.leg;
            if (row.kind == ResourceKind::NocFlow
                    && leg.rfind("noc:", 0) != 0) {
                fail(path + " NoC resource leg name is invalid");
            }
            if (row.kind == ResourceKind::HbmJob
                    && leg.rfind("hbm:", 0) != 0) {
                fail(path + " HBM resource leg name is invalid");
            }
            if (row.kind == ResourceKind::PoolJob
                    && leg.rfind("pool:", 0) != 0) {
                fail(path + " pool resource leg name is invalid");
            }
        } else {
            const bool unattributed_noc =
                row.kind == ResourceKind::NocFlow
                && row.request_id == "noc" && row.stage.empty()
                && row.generation == 0;
            const bool external_background_hbm =
                row.kind == ResourceKind::HbmJob
                && row.request_id == "external_background"
                && row.stage.empty() && row.generation == 0;
            if (!(unattributed_noc || external_background_hbm)
                    && row.stage != "prefill" && row.stage != "decode"
                    && row.stage != "completion") {
                fail(path + ".stage is invalid");
            }
        }
        if (row.bound && row.stage != "prefill" && row.stage != "decode"
                && row.stage != "completion") {
            fail(path + ".stage is invalid");
        }
        // Reconstructed canonical identity (validated form of the legacy
        // wire's work_id comparison).
        row.work_id = leg_work_id(
            canonical_work_id(row.request_id, row.stage, row.generation,
                              row.rank, row.node_id),
            row.leg);
        if (row.kind != ResourceKind::NocFlow && !row.route.empty()) {
            fail(path + " non-NoC route must be empty");
        }
        if (row.kind != ResourceKind::HbmJob) {
            if (!row.total.ops.is_null()
                    || !row.total.latency_ns.is_null()
                    || !row.remaining.ops.is_null()
                    || !row.remaining.latency_ns.is_null()) {
                fail(path + " total/remaining for this kind may define "
                            "bytes only");
            }
        }
        if (row.kind == ResourceKind::NocFlow) {
            if (row.leg != "noc:" + std::to_string(row.flow_id)) {
                fail(path + ".leg must identify its original flow_id");
            }
            if (!row.bound && !(row.request_id == "noc"
                    && row.stage.empty() && row.generation == 0
                    && row.node_id == row.flow_id)) {
                fail(path + " unattributed NoC identity must preserve "
                            "noc/empty-stage/flow-id form");
            }
            if (!row.current_rate_Bpns.is_null()) {
                if (row.current_rate_Bpns.as_double() < 0
                        || !std::isfinite(row.current_rate_Bpns.as_double())) {
                    fail(path + ".current_rate_Bpns must be finite "
                                "non-negative");
                }
            }
            if (!row.last_rate_update_time.is_null()) {
                if (row.last_rate_update_time.as_double() < 0
                        || !std::isfinite(
                               row.last_rate_update_time.as_double())) {
                    fail(path
                         + ".last_rate_update_time must be finite "
                           "non-negative");
                }
            }
            if (row.state == "Completed"
                    && !(row.remaining.bytes.is_null()
                         || (row.remaining.bytes.is_int()
                             && row.remaining.bytes.integer == 0))) {
                fail(path + " completed NoC tail must have zero remaining "
                            "bytes");
            }
        } else if (row.kind == ResourceKind::HbmJob) {
            if (row.request_id == "external_background"
                    && row.stage.empty() && row.generation == 0) {
                if (row.job_kind != "COMM_READ"
                        && row.job_kind != "COMM_WRITE") {
                    fail(path + " external background HBM job kind must be "
                                "a transfer-sink COMM_READ/COMM_WRITE");
                }
                const std::string& leg = row.leg;
                const std::size_t hash = leg.find('#');
                const std::string leg_rank = leg.substr(0, hash);
                const std::string leg_seq = hash == std::string::npos
                    ? std::string() : leg.substr(hash + 1);
                const std::size_t colon = leg_rank.find(':');
                const std::string head = leg_rank.substr(0, colon);
                const std::string tail = colon == std::string::npos
                    ? std::string() : leg_rank.substr(colon + 1);
                bool seq_ok = false;
                {
                    std::uint64_t seq_value = 0;
                    bool digits = !leg_seq.empty()
                        && leg_seq.find_first_not_of("0123456789")
                            == std::string::npos;
                    if (digits && leg_seq.size() <= 20) {
                        try {
                            seq_value = std::stoull(leg_seq);
                            seq_ok = seq_value >= 1;
                        } catch (...) {
                            seq_ok = true;
                        }
                    } else if (digits) {
                        seq_ok = true;
                    }
                }
                std::uint64_t rank_value = 0;
                bool rank_ok = false;
                {
                    bool digits = !tail.empty()
                        && tail.find_first_not_of("0123456789")
                            == std::string::npos;
                    if (digits && tail.size() <= 20) {
                        try {
                            rank_value = std::stoull(tail);
                            rank_ok = rank_value == row.rank;
                        } catch (...) {
                            rank_ok = false;
                        }
                    }
                }
                const bool leg_ok =
                    leg.rfind("hbm_external:", 0) == 0 && seq_ok
                    && head == "hbm_external"
                    && !tail.empty()
                    && tail.find_first_not_of("0123456789")
                        == std::string::npos
                    && rank_ok;
                if (!leg_ok) {
                    fail(path + ".leg must be the literal external "
                                "background form");
                }
            } else if (row.leg != "hbm:" + row.job_kind) {
                fail(path + ".leg must identify its HBM job kind");
            }
        } else {
            if (row.leg != "pool:" + std::to_string(row.port_index) + ":"
                           + std::to_string(row.issue_seq)) {
                fail(path + ".leg must identify its original pool job");
            }
            if (!row.latency_ready_ns.is_null()
                    && (row.latency_ready_ns.as_double() < 0
                        || !std::isfinite(
                               row.latency_ready_ns.as_double()))) {
                fail(path + ".latency_ready_ns must be finite "
                            "non-negative");
            }
        }
        const WireNum* total_values[3] = {&row.total.ops, &row.total.bytes,
                                          &row.total.latency_ns};
        const WireNum* remaining_values[3] = {&row.remaining.ops,
                                              &row.remaining.bytes,
                                              &row.remaining.latency_ns};
        for (int d = 0; d < 3; ++d) {
            if (total_values[d]->is_null()
                    != remaining_values[d]->is_null()) {
                fail(path + " changes a dimension's applicability");
            }
            if (!total_values[d]->is_null()
                    && remaining_values[d]->as_double()
                           > total_values[d]->as_double()) {
                fail(path + " remaining exceeds total");
            }
        }
        // Sorted-unique walk over the legacy resource sort key.
        const std::tuple<std::string, std::uint64_t, std::uint64_t,
                         std::string, std::string> sort_key = std::make_tuple(
            row.bound ? row.descriptor_segment_id : std::string(), row.rank,
            row.node_id, row.leg, std::string(resource_kind_name(row.kind)));
        if (previous_resource_.has_value()
                && !(sort_key > previous_resource_.value())) {
            fail("active_resources must be sorted and duplicate-free");
        }
        previous_resource_ = sort_key;
        resources_.push_back(std::move(row));
    }
}

ParsedEnvelope FrameReader::read_all() {
    read_descriptors();
    read_nodes();
    read_runs();
    read_cursors();
    read_resources();
    if (!cursor_.exhausted()) {
        fail("trailing bytes after the last section");
    }

    // ---- header validation (validate_header_fields parity) ----
    if (delivery_ != expected_delivery_) {
        fail("work_ledger delivery_sequence " + std::to_string(delivery_)
             + " != delta delivery_sequence "
             + std::to_string(expected_delivery_));
    }
    if (tick_ != expected_tick_) {
        fail("work_ledger tick " + std::to_string(tick_)
             + " != delta tick " + std::to_string(expected_tick_));
    }
    const bool seq_start_ok =
        base_watermark_ < U64_MAX && seq_start_ == base_watermark_ + 1;
    if (event_count_ == 0) {
        if (!seq_start_ok || seq_end_ != base_watermark_) {
            fail("empty delta sequence interval is invalid");
        }
    } else {
        const bool interval_ok = seq_start_ok && seq_end_ >= seq_start_
            && seq_end_ - seq_start_ + 1 == event_count_;
        if (!interval_ok) {
            fail("logical sequence interval/count is inconsistent");
        }
    }
    const bool version_ok =
        result_version_ == base_version_
        || (base_version_ < U64_MAX
            && result_version_ == base_version_ + 1);
    if ((full_ && base_watermark_ != 0) || !version_ok) {
        fail("full watermark or state-version transition is invalid");
    }
    if (node_events_.size() + runs_.size() != event_count_) {
        fail("logical event count differs from typed transition count");
    }

    for (auto& entry : node_events_) {
        entry.second.delivery_sequence = delivery_;
    }
    ParsedEnvelope out;
    out.header.replay_token =
        "cpp_online/" + epoch_ + "/" + std::to_string(delivery_);
    out.header.epoch = epoch_;
    out.header.delivery_sequence = delivery_;
    out.header.full = full_;
    out.header.tick = tick_;
    out.header.base_watermark = base_watermark_;
    out.header.seq_start = seq_start_;
    out.header.seq_end = seq_end_;
    out.header.event_count = event_count_;
    out.header.base_version = base_version_;
    out.header.result_version = result_version_;
    out.descriptors = std::move(commit_list_);
    out.selector_index_additions = std::move(selector_additions_);
    out.dependency_index_additions = std::move(dependency_additions_);
    out.node_events = std::move(node_events_);
    out.pipeline_runs = std::move(runs_);
    out.cursor_patches = std::move(cursor_patches_);
    out.resources = std::move(resources_);
    return out;
}

std::uint64_t read_rank_json_wrap(std::uint64_t value) {
    if (value > I32_MAX) {
        fail("rank exceeds its typed domain");
    }
    return value;
}

}  // namespace

ParsedEnvelope parse_envelope_compact(const char* data, std::size_t length,
                                     const RetainedRegistry& retained,
                                     bool allow_retained_replay,
                                     std::uint64_t expected_delivery,
                                     std::uint64_t expected_tick) {
    FrameReader reader(data, length, retained, allow_retained_replay,
                       expected_delivery, expected_tick);
    return reader.read_all();
}

}  // namespace jointledger
