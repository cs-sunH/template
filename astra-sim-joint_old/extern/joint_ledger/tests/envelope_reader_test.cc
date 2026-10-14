// envelope_reader_test.cc -- Python-parity tests for the v4 envelope typed
// reader (EnvelopeReader.cc + RowParsers.cc).
//
// Cases are grouped so that everything which needs PipelineDescriptor
// (src/PipelineDescriptor.cc, port unit N2) lives in
// run_descriptor_dependent_tests(): when that unit is still a stub the
// probe fails closed and the group is reported as SKIPPED_DEPENDENCY, while
// the header/node/SAX/canonical group always runs.
//
// The descriptor wire fixture is the golden minimal descriptor from
// sh_test_mesh/workload/llama2_7b_inference/joint/test_pipeline_descriptor.py
// _wire() (identity pp1/726571/decode/1/2/1/0), so every derived number in
// the descriptor group (endpoint IDs 51..58, tile bytes 2/2/1, store IDs,
// dependency intervals 52..56 and 55..59) matches the Python algebra.
#include <array>
#include <cstdio>
#include <cstdint>
#include <functional>
#include <string>
#include <vector>

#include "joint_ledger/LedgerTypes.hh"
#include "joint_ledger/PipelineRows.hh"

// Internal row-parse API (defined in RowParsers.cc with external linkage);
// the view structs repeat the identical definition used there.
namespace jointledger {
namespace rowport {

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

struct SelectorView {
    const std::unordered_map<std::string, SelectorIndex>* declared = nullptr;
    const std::unordered_map<std::string, SelectorIndex>* retained = nullptr;
    const SelectorIndex* find(const std::string& segment) const {
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

bool python_equal(const Json& left, const Json& right);
std::string canonical_json_dumps(const Json& value);
std::string sha256_hex(const std::string& bytes);
Json parse_v4_slot_patches(Json value, const std::string& path,
                           const DescriptorView& descriptors,
                           const SelectorView& selector_indexes);
Json parse_credit_patches(Json value, const std::string& path);
Json parse_obligation_patches(Json value, const std::string& path,
                              const DescriptorView& descriptors);

}  // namespace rowport
}  // namespace jointledger

using jointledger::Json;
using jointledger::LedgerError;
using jointledger::ParsedEnvelope;
using jointledger::RetainedRegistry;
using jointledger::parse_envelope_v4;

namespace {

int g_checks = 0;
int g_failures = 0;

void check(bool condition, const char* expression) {
    ++g_checks;
    if (!condition) {
        ++g_failures;
        std::printf("FAIL: %s\n", expression);
    }
}

void check_throws(const std::function<void()>& action,
                  const std::string& needle, const char* name) {
    ++g_checks;
    try {
        action();
        ++g_failures;
        std::printf("FAIL: %s (no LedgerError raised)\n", name);
    } catch (const LedgerError& error) {
        const std::string message = error.what();
        if (message.find(needle) == std::string::npos) {
            ++g_failures;
            std::printf("FAIL: %s (message '%s' lacks '%s')\n", name,
                        message.c_str(), needle.c_str());
        }
    }
}

Json u(std::uint64_t value) { return Json(value); }

// ------------------------------------------------------------ fixtures ----

const char* const kEpoch = "e1";
constexpr std::uint64_t kDelivery = 1;
constexpr std::uint64_t kTick = 10;

Json empty_amount() {
    return Json{{"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}};
}

Json base_header() {
    return Json{
        {"producer", "cpp_online"}, {"schema_version", u(4)},
        {"protocol_version", u(4)}, {"bridge_protocol", "bridge-v4-20261011"},
        {"work_ledger_schema", "workledger-v4-20261011"},
        {"epoch", kEpoch}, {"delivery_sequence", u(kDelivery)},
        {"base_watermark", u(0)}, {"logical_seq_start", u(1)},
        {"logical_seq_end", u(0)}, {"logical_event_count", u(0)},
        {"replay_token", "cpp_online/e1/1"}, {"descriptor_version", u(1)},
        {"base_state_version", u(0)}, {"result_state_version", u(0)},
        {"snapshot_kind", "full"}, {"tick", u(kTick)},
        {"descriptor_commits", Json::array()},
        {"node_transitions", Json::array()},
        {"pipeline_transition_runs", Json::array()},
        {"cursor_patches", Json::array()},
        {"active_resources", Json::array()}};
}

Json patched(Json envelope, const Json& patch) {
    for (auto member = patch.begin(); member != patch.end(); ++member) {
        envelope[member.key()] = member.value();
    }
    return envelope;
}

Json envelope_with_events(std::uint64_t count) {
    Json header = base_header();
    header["logical_event_count"] = u(count);
    header["logical_seq_start"] = u(1);
    header["logical_seq_end"] = u(count == 0 ? 0 : count);
    header["snapshot_kind"] = "delta";
    return header;
}

ParsedEnvelope parse_text(const std::string& text,
                          std::uint64_t delivery = kDelivery,
                          std::uint64_t tick = kTick) {
    RetainedRegistry registry;  // empty registry (both maps null)
    return parse_envelope_v4(text.data(), text.size(), registry, false,
                             delivery, tick);
}

ParsedEnvelope parse_ok(const Json& envelope, std::uint64_t delivery = kDelivery,
                        std::uint64_t tick = kTick) {
    return parse_text(envelope.dump(), delivery, tick);
}

void parse_fail(const Json& envelope, const std::string& needle,
                std::uint64_t delivery = kDelivery,
                std::uint64_t tick = kTick) {
    const std::string text = envelope.dump();
    check_throws([&] { parse_text(text, delivery, tick); }, needle,
                 "parse_fail");
}

Json node_row(std::uint64_t seq, std::uint64_t tick,
              const std::string& kind, const std::string& work_id,
              Json payload) {
    return Json{{"logical_seq", u(seq)}, {"tick", u(tick)}, {"kind", kind},
                {"work_id", work_id}, {"payload", std::move(payload)}};
}

// --------------------------------------------- descriptor-dependent -------

const char* const kSegment = "pp1/726571/decode/1/2/1/0";
const char* const kDomain = "pb1/726571/1/9";

const char* kDescriptorWire = R"({"descriptor_version":1,"identity":{"request_id":"req","stage":"decode","generation":1,"stage_epoch":2,"segment_ordinal":0,"rank":1,"staging_allocation_id":9,"segment_id":"pp1/726571/decode/1/2/1/0","buffer_domain_id":"pb1/726571/1/9"},"stream_kind":"decode_remote_credit","groups":[{"group_ordinal":0,"pass_or_window_ordinal":0,"source_ordinal":0,"source_key":"source-0","layer_start":0,"layer_end":1,"tile_axis":{"unit":"stream_bytes","total_units":5,"units_per_tile":2,"tile_count":3,"tail_units":1,"global_tile_first":1},"shards":[{"shard_ordinal":0,"source_rank":0,"target_rank":1,"total_data_bytes":5,"bytes_per_unit":1,"noc_path":[0,1]}]}],"protocol":{"buffer_slots":2,"release_lag_tiles":2,"initial_tiles_without_release":2,"fill_ack_bytes":1,"release_ack_bytes":1,"hbm_charge":true,"target_rank_order":"release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1","source_rank_order":"release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i","tile_branching":"tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"},"prediction_profile":{"prefill_consume":"zero_duration_no_extra_consume","decode_consume":{"kind":"profiled_consume","profile_id":"decode-service-profile-v1","compute_ns_total":240.0,"hbm_bytes_total":5,"resource_key":"hbm:1","share_rule":"existing_action_work_graph_byte_share_vector"}},"execution_profile":{"data_legs":"ordinary_comm_send_recv_path","endpoint_hbm_rule":"one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint","prefill_consume":"existing_body_or_marker_only","decode_consume":"existing_body_nodes_only"},"logical_id_ranges":[{"rank":0,"first_json_id":53,"count":2,"stride":2,"kind":"virtual_endpoint_hole"},{"rank":1,"first_json_id":56,"count":2,"stride":2,"kind":"virtual_endpoint_hole"}],"tag_allocation_runs":[{"allocator_scope_id":"builder-0","emission_ordinal":7,"first_tag":101,"count":3,"group_ordinal":0,"first_tile":1,"first_shard":0,"tile_count":3,"shard_count":1}],"endpoint_runs":[{"rank":0,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"first_json_id":51,"tile_stride":2,"shard_stride":1,"initial_parent_json_id":50,"legacy_context":{"request_id":"req","stage":"decode","generation":1},"legs":[{"endpoint":"data_send","json_offset":0,"tag_emission_ordinal":7,"tag_offset":0,"tag_tile_stride":1,"tag_shard_stride":1}]},{"rank":1,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"first_json_id":54,"tile_stride":2,"shard_stride":1,"initial_parent_json_id":53,"legacy_context":{"request_id":"req","stage":"decode","generation":1},"legs":[{"endpoint":"data_recv","json_offset":0,"tag_emission_ordinal":7,"tag_offset":0,"tag_tile_stride":1,"tag_shard_stride":1}]}],"endpoint_dependency_runs":[{"rank":0,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"endpoint":"data_send","dependency_kind":"ordinary_node_terminal","first_dependency_json_id":50,"dependency_tile_stride":0,"dependency_shard_stride":0}],"consume_completion_runs":[{"rank":1,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"dependency_kind":"ordinary_node_terminal","first_dependency_json_id":70,"dependency_tile_stride":0,"dependency_shard_stride":0}],"projection":{"version":1,"chains":[{"group_ordinal":0,"rank":1,"tile_count":3,"fill":{"leg_id_prefix":"work#dcstream#R1#F","first_index":0,"fill_order_first":0,"fill_order_stride":1,"completion_order_first":null,"completion_order_stride":0,"route":["link:0"],"bytes_per_tile":2,"latency_ns":3,"start_deps":["work#start"],"gate_id":"work#gate:dcstream:R1","gate_capacity":2},"consume":{"leg_id_prefix":"work#dcstream#R1#C","first_index":0,"consume_order_first":3,"consume_order_stride":1,"completion_order_first":null,"completion_order_stride":0,"kind":"compute","route":["hbm:1"],"group":"decode","latency_first_ns":7,"latency_rest_ns":0,"consume_deps":["work#consume-start"],"consume_start_deps":[],"gate_id":"work#gate:dcstream:R1","compute_share":{"rule":"quotient_remainder_earliest_global_indices","total_units":8,"total_tiles":3,"share_index_first":0,"tile_bytes":2},"hbm_share":{"rule":"quotient_remainder_earliest_global_indices","total_units":5,"total_tiles":3,"share_index_first":0,"tile_bytes":2}}}]},"anchors":[{"role":"start","rank":0,"binding":"physical_endpoint_slot","node_id":51,"selector":{"kind":"tile_endpoint_leg","group_ordinal":0,"tile_ordinal":1,"shard_ordinal":0,"endpoint":"data_send","tag_allocation_emission_ordinal":7,"tag_offset":0}},{"role":"frontier","rank":1,"binding":"physical_endpoint_slot","node_id":54,"selector":{"kind":"tile_endpoint_terminal","group_ordinal":0,"tile_ordinal":1,"shard_ordinal":0,"endpoint":"data_recv","tag_allocation_emission_ordinal":7,"tag_offset":0}},{"role":"service","rank":1,"binding":"descriptor_cursor_selector","node_id":null,"selector":{"kind":"existing_service_gate","existing_gate":"decode_service_gate"}},{"role":"settle","rank":1,"binding":"descriptor_cursor_selector","node_id":null,"selector":{"kind":"all_obligations_settled"}}]})";

Json descriptor_wire() { return Json::parse(kDescriptorWire); }

Json descriptor_commit_entry() {
    Json wire = descriptor_wire();
    return Json{{"identity", wire["identity"]}, {"descriptor", wire},
                {"commit_tick", u(3)}, {"cursor_version", u(0)},
                {"service_done", false}, {"settle_done", false},
                {"retired", false}};
}

Json selector(std::uint64_t group, std::uint64_t tile, std::uint64_t shard,
              const std::string& endpoint) {
    return Json{{"group_ordinal", u(group)}, {"tile_ordinal", u(tile)},
                {"shard_ordinal", u(shard)}, {"endpoint", endpoint}};
}

Json run_row(const Json& patch) {
    Json row = Json{
        {"first_logical_seq", u(1)}, {"pattern_id", "p"},
        {"repeat_count", u(1)}, {"tile_first", nullptr},
        {"tile_stride", nullptr}, {"rank_first", nullptr},
        {"rank_stride", nullptr}, {"shard_first", nullptr},
        {"shard_stride", nullptr}, {"transition_kind", "service_done"},
        {"descriptor_segment_id", kSegment}, {"endpoint_selector", nullptr},
        {"store_id_first", nullptr}, {"store_id_stride", nullptr},
        {"legacy_context", nullptr}, {"tick_first", u(5)},
        {"tick_stride", u(0)}, {"resource_leg", nullptr},
        {"resource_quantity_prefix", Json::array({empty_amount(),
                                                  empty_amount()})},
        {"cursor_version_start", u(0)}, {"cursor_version_end", u(0)},
        {"slot_patches", Json::array()}, {"credit_patches", Json::array()},
        {"obligation_patches", Json::array()}};
    for (auto member = patch.begin(); member != patch.end(); ++member) {
        row[member.key()] = member.value();
    }
    return row;
}

Json endpoint_run_patch(std::uint64_t tile, std::uint64_t rank,
                        std::uint64_t shard, std::uint64_t store,
                        const Json& selector_value,
                        const std::string& transition = "tile_issue") {
    return Json{{"tile_first", u(tile)}, {"tile_stride", u(0)},
                {"rank_first", u(rank)}, {"rank_stride", u(0)},
                {"shard_first", u(shard)}, {"shard_stride", u(0)},
                {"transition_kind", transition},
                {"endpoint_selector", selector_value},
                {"store_id_first", u(store)}, {"store_id_stride", u(0)},
                {"legacy_context", Json{{"request_id", "req"},
                                        {"stage", "decode"},
                                        {"generation", u(1)}}}};
}

// Envelope with one declared descriptor and the given row sections.
Json descriptor_envelope(const Json& runs, const Json& cursor = Json::array(),
                         const Json& resources = Json::array(),
                         const Json& header_patch = Json::object()) {
    Json header = patched(base_header(), header_patch);
    header["descriptor_commits"] = Json::array({descriptor_commit_entry()});
    header["pipeline_transition_runs"] = runs;
    header["cursor_patches"] = cursor;
    header["active_resources"] = resources;
    return header;
}

Json slot_owner() {
    return Json{{"segment_id", kSegment}, {"pipeline_generation", u(1)},
                {"stage_epoch", u(2)}, {"staging_allocation_id", u(9)},
                {"group_ordinal", u(0)}, {"tile_ordinal", u(1)}};
}

Json source_slot_row(std::uint64_t tile) {
    const std::uint64_t global_tile = tile;  // global_tile_first = 1
    return Json{{"buffer_domain_id", kDomain}, {"role", "source"},
                {"rank", u(0)}, {"shard_ordinal", u(0)},
                {"slot_index", u((global_tile - 1) % 2)}, {"occupied", true},
                {"owner", slot_owner()}, {"consumed", false},
                {"target_shards_to_consume", Json::array()},
                {"consumed_target_shards", Json::array()}};
}

Json cursor_run_value(std::uint64_t run_index) {
    const std::uint64_t first_json_id = run_index == 0 ? 51 : 54;
    const std::uint64_t rank = run_index == 0 ? 0 : 1;
    return Json{
        {"rank", u(rank)}, {"group_ordinal", u(0)}, {"first_tile", u(1)},
        {"tile_count", u(3)}, {"completed_endpoint_count", u(0)},
        {"cancelled_endpoint_count", u(0)}, {"remaining_endpoint_count", u(3)},
        {"in_flight", false}, {"next_store_id", u(first_json_id + 1)},
        {"tile_offset", u(0)}, {"shard_index", u(0)}, {"leg_index", u(0)},
        {"initial_parent_store_id", u(first_json_id)},
        {"completed_out_of_order", Json::array()},
        {"active_endpoints", Json::array()}};
}

Json cursor_patch(std::uint64_t run_index, bool retired = false) {
    Json runs = Json::array();
    if (retired) {
        runs.push_back(Json{{"run_index", u(0)}, {"value", nullptr}});
        runs.push_back(Json{{"run_index", u(1)}, {"value", nullptr}});
    } else {
        runs.push_back(Json{{"run_index", u(run_index)},
                            {"value", cursor_run_value(run_index)}});
    }
    return Json{{"segment_id", kSegment}, {"cursor_version", u(1)},
                {"retired", retired}, {"runs", runs}};
}

Json noc_flow_resource(const Json& patch) {
    Json row = Json{
        {"resource_type", "noc_flow"}, {"work_id", "noc#:0:r0n7@noc:7"},
        {"request_id", "noc"}, {"stage", ""}, {"generation", u(0)},
        {"rank", u(0)}, {"node_id", u(7)}, {"operation", "comm"},
        {"leg", "noc:7"}, {"state", "Active"},
        {"total", Json{{"ops", nullptr}, {"bytes", u(10)},
                       {"latency_ns", nullptr}}},
        {"remaining", Json{{"ops", nullptr}, {"bytes", u(4)},
                           {"latency_ns", nullptr}}},
        {"progress_source", "authoritative"}, {"service_unit", "bytes"},
        {"route", Json::array({u(1), u(2)})}, {"as_of_tick", u(5)},
        {"issued_tick", u(3)}, {"descriptor_segment_id", nullptr},
        {"endpoint_selector", nullptr}, {"flow_id", u(7)},
        {"current_rate_Bpns", 1.5}, {"last_rate_update_time", 2.25}};
    for (auto member = patch.begin(); member != patch.end(); ++member) {
        row[member.key()] = member.value();
    }
    return row;
}

// ------------------------------------------------------- test groups ------

void run_header_tests() {
    // 1. minimal full envelope parses with every header field surfaced.
    {
        const ParsedEnvelope parsed = parse_ok(base_header());
        check(parsed.header.full, "full flag");
        check(parsed.header.epoch == "e1", "epoch");
        check(parsed.header.delivery_sequence == 1, "delivery");
        check(parsed.header.tick == 10, "tick");
        check(parsed.header.base_watermark == 0, "watermark");
        check(parsed.header.seq_start == 1 && parsed.header.seq_end == 0,
              "seq interval");
        check(parsed.header.event_count == 0, "event count");
        check(parsed.header.replay_token == "cpp_online/e1/1",
              "replay token");
        check(parsed.descriptors.empty() && parsed.node_events.empty()
                  && parsed.pipeline_runs.empty()
                  && parsed.cursor_patches.empty() && parsed.resources.empty(),
              "empty sections");
        check(parsed.selector_index_additions.empty()
                  && parsed.dependency_index_additions.empty(),
              "no index additions");
    }
    // 2. delta with count 0 (seq_end == watermark) is valid.
    {
        Json header = base_header();
        header["snapshot_kind"] = "delta";
        const ParsedEnvelope parsed = parse_ok(header);
        check(!parsed.header.full, "delta flag");
    }
    // 3-8: identity family.
    parse_fail(patched(base_header(), Json{{"schema_version", u(5)}}),
               "unsupported v4 producer/schema/protocol");
    parse_fail(patched(base_header(), Json{{"protocol_version", u(3)}}),
               "unsupported v4 producer/schema/protocol");
    parse_fail(patched(base_header(), Json{{"producer", "py_online"}}),
               "unsupported v4 producer/schema/protocol");
    parse_fail(patched(base_header(),
                       Json{{"bridge_protocol", "bridge-v3-20261009"}}),
               "unsupported v4 bridge or work-ledger schema");
    parse_fail(patched(base_header(),
                       Json{{"work_ledger_schema", "workledger-v3-20261009"}}),
               "unsupported v4 bridge or work-ledger schema");
    parse_fail(patched(base_header(), Json{{"descriptor_version", u(2)}}),
               "unsupported descriptor_version");
    // 9: replay token mismatch.
    parse_fail(patched(base_header(), Json{{"replay_token", "cpp_online/e2/1"}}),
               "replay_token does not match");
    // 10/11: key set.  A missing tick hits the outer-frame cross-check
    // first (the coordinator validates the payload tick before the ledger
    // key set, like work_ledger.py _validate_payload_envelope).
    {
        Json missing = base_header();
        missing.erase("epoch");
        parse_fail(missing, "pipeline ledger envelope keys invalid");
        parse_fail(patched(base_header(), Json{{"surprise", u(1)}}),
                   "pipeline ledger envelope keys invalid");
        Json no_tick = base_header();
        no_tick.erase("tick");
        parse_fail(no_tick, "work_ledger tick -1 != delta tick");
    }
    // 12-14: optional_sha256.
    {
        Json with_digest = base_header();
        const std::string digest = jointledger::rowport::sha256_hex(
            jointledger::rowport::canonical_json_dumps(base_header()));
        with_digest["optional_sha256"] = digest;
        const ParsedEnvelope parsed = parse_ok(with_digest);
        check(parsed.header.event_count == 0, "digest envelope parses");

        Json wrong = base_header();
        wrong["optional_sha256"] = std::string(64, '0');
        parse_fail(wrong, "optional_sha256 does not match");

        Json malformed = base_header();
        malformed["optional_sha256"] = std::string(63, 'a');
        parse_fail(malformed, "64 lowercase hexadecimal");

        Json uppercase = base_header();
        uppercase["optional_sha256"] =
            jointledger::rowport::sha256_hex("x");
        uppercase["optional_sha256"] = "ABCD";
        parse_fail(uppercase, "64 lowercase hexadecimal");
    }
    // 15-17: sequence interval consistency.
    {
        Json header = base_header();
        header["logical_seq_start"] = u(2);  // watermark is 0
        parse_fail(header, "empty delta sequence interval is invalid");
    }
    {
        Json header = base_header();
        header["logical_seq_end"] = u(1);
        parse_fail(header, "empty delta sequence interval is invalid");
    }
    {
        Json header = envelope_with_events(2);
        header["logical_seq_end"] = u(3);  // 1..3 covers 3, count says 2
        parse_fail(header, "logical sequence interval/count is inconsistent");
    }
    // 18: full snapshot with nonzero watermark.
    {
        Json header = envelope_with_events(1);
        header["snapshot_kind"] = "full";
        header["base_watermark"] = u(4);
        header["logical_seq_start"] = u(5);
        header["logical_seq_end"] = u(5);
        parse_fail(header, "full watermark or state-version transition");
    }
    // 19: result version must be base or base+1.
    {
        Json header = base_header();
        header["result_state_version"] = u(2);
        parse_fail(header, "full watermark or state-version transition");
    }
    // 20/21: scalar typing (bool/float rejected like Python).
    {
        Json header = base_header();
        header["delivery_sequence"] = true;
        parse_fail(header, "must be uint");
    }
    {
        Json header = base_header();
        header["tick"] = 10.5;
        parse_fail(header, "must be uint");
    }
    // 22/23: outer-frame cross-check.
    parse_fail(base_header(), "delta delivery_sequence", 2, 10);
    parse_fail(base_header(), "delta tick", 1, 11);
    // 24/25: string enums.
    parse_fail(patched(base_header(), Json{{"epoch", ""}}),
               "epoch must be a non-empty string");
    parse_fail(patched(base_header(), Json{{"snapshot_kind", "incremental"}}),
               "snapshot_kind must be full or delta");
}

void run_node_tests() {
    const Json payload = Json{
        {"request_id", "r"}, {"stage", "decode"}, {"generation", u(1)},
        {"rank", u(0)}, {"node_id", u(5)},
        {"delta_served", Json{{"ops", u(3)}, {"bytes", nullptr},
                              {"latency_ns", nullptr}}}};
    // 26: commit event parses into a NodeEvent with payload kept verbatim.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(1, 8, "commit", "r#decode:1:r0n5",
                                  payload)});
        const ParsedEnvelope parsed = parse_ok(header);
        check(parsed.node_events.size() == 1, "one node event");
        const auto& event = parsed.node_events[0];
        check(event.first == 1, "logical seq");
        check(event.second.kind == "commit", "kind");
        check(event.second.work_id == "r#decode:1:r0n5", "work id");
        check(event.second.tick == 8, "event tick");
        check(event.second.producer == "cpp_online", "producer");
        check(event.second.delivery_sequence == 1, "delivery seq");
        check(event.second.local_seq == 1, "local seq");
        check(event.second.delta_served.has_value()
                  && event.second.delta_served->ops == 3.0
                  && !event.second.delta_served->bytes
                  && !event.second.delta_served->latency_ns,
              "delta served parsed");
        check(event.second.payload.contains("delta_served"),
              "payload keeps delta_served");
        check(event.second.payload["request_id"] == "r", "payload body");
    }
    // 27: progress with float delta.
    {
        Json progress_payload = Json{{"leg", "hbm:COMM_READ"}};
        progress_payload["delta_served"] =
            Json{{"ops", nullptr}, {"bytes", 2.5}, {"latency_ns", nullptr}};
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array(
            {node_row(1, 9, "progress", "w", progress_payload)});
        const ParsedEnvelope parsed = parse_ok(header);
        check(parsed.node_events[0].second.delta_served->bytes == 2.5,
              "float delta");
    }
    // 28/29: row key set.
    {
        Json row = node_row(1, 5, "issue", "w", Json::object());
        row.erase("tick");
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array({row});
        parse_fail(header, "keys invalid");
    }
    {
        Json row = node_row(1, 5, "issue", "w", Json::object());
        row["extra"] = u(1);
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array({row});
        parse_fail(header, "keys invalid");
    }
    // 30: kind enum.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(1, 5, "revoke", "w", Json::object())});
        parse_fail(header, ".kind is unknown");
    }
    // 31: payload whitelist.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array(
            {node_row(1, 5, "issue", "w", Json{{"bogus", u(1)}})});
        parse_fail(header, ".payload has unknown keys");
    }
    // 32: seq must start at 1.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(0, 5, "issue", "w", Json::object())});
        parse_fail(header, "logical sequence/tick is invalid");
    }
    // 33: event tick must not exceed the envelope tick.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(1, 11, "issue", "w", Json::object())});
        parse_fail(header, "logical sequence/tick is invalid");
    }
    // 34: strictly increasing sequence.
    {
        Json header = envelope_with_events(2);
        header["logical_seq_end"] = u(2);
        header["node_transitions"] = Json::array(
            {node_row(1, 5, "issue", "w", Json::object()),
             node_row(1, 6, "ack", "w", Json::object())});
        parse_fail(header, "must preserve unique logical sequence order");
    }
    // 35: remaining_* only on commit/progress.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array(
            {node_row(1, 5, "issue", "w", Json{{"remaining_bytes", u(2)}})});
        parse_fail(header,
                   "remaining_* fields are only valid on commit/progress");
    }
    // 36: delta_served key whitelist.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array(
            {node_row(1, 5, "progress", "w",
                      Json{{"delta_served", Json{{"ops", u(1)},
                                                 {"bogus", u(2)}}}})});
        parse_fail(header, "unknown work-amount keys");
    }
    // 37: negative delta_served.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array(
            {node_row(1, 5, "progress", "w",
                      Json{{"delta_served", Json{{"ops", u(0)},
                                                 {"bytes", -2.0}}}})});
        parse_fail(header, "must be non-negative");
    }
    // 38: bool inside delta_served.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] = Json::array(
            {node_row(1, 5, "progress", "w",
                      Json{{"delta_served", Json{{"ops", true}}}})});
        parse_fail(header, "is not a valid typed node event");
    }
    // 39: payload must be an object.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(1, 5, "issue", "w", Json::array())});
        parse_fail(header, ".payload must be a string-keyed object");
    }
    // 40: the count-vs-rows assertion belongs to the stage core: one node
    // row with a consistent interval parses even though nothing else
    // declares the events.
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(1, 5, "issue", "w", Json::object())});
        const ParsedEnvelope parsed = parse_ok(header);
        check(parsed.node_events.size() == 1, "single node row accepted");
    }
}

void run_sax_structure_tests() {
    // 41: malformed JSON reports a byte offset.
    check_throws([] { parse_text("{\"producer\":\"cpp_online\","); },
                 "byte", "truncated json");
    // 42: top level must be an object.
    check_throws([] { parse_text("[1,2,3]"); }, "must be an object",
                 "array envelope");
    check_throws([] { parse_text("7"); }, "must be an object",
                 "scalar envelope");
    // 43: sections must be arrays.
    {
        Json header = base_header();
        header["node_transitions"] = u(5);
        parse_fail(header, "node_transitions must be an array");
    }
    // 44: duplicate scalar header keys keep last-wins (raw JSON text; the
    // node rows precede both tick keys, so the deferred check sees the
    // final value).
    {
        Json header = envelope_with_events(1);
        header["node_transitions"] =
            Json::array({node_row(1, 9, "issue", "w", Json::object())});
        std::string raw = header.dump();
        const std::size_t tick_pos = raw.find("\"tick\":10");
        raw.insert(tick_pos, "\"tick\":4,");
        const ParsedEnvelope parsed = parse_text(raw);
        check(parsed.header.tick == 10, "duplicate tick last-wins");
    }
    // 45: sections may arrive before the header scalars; the tick check is
    // applied once the header completes.
    {
        const std::string tail =
            "\"producer\":\"cpp_online\",\"schema_version\":4,"
            "\"protocol_version\":4,\"bridge_protocol\":"
            "\"bridge-v4-20261011\",\"work_ledger_schema\":"
            "\"workledger-v4-20261011\",\"epoch\":\"e1\","
            "\"delivery_sequence\":1,\"base_watermark\":0,"
            "\"logical_seq_start\":1,\"logical_seq_end\":1,"
            "\"logical_event_count\":1,\"replay_token\":"
            "\"cpp_online/e1/1\",\"descriptor_version\":1,"
            "\"base_state_version\":0,\"result_state_version\":1,"
            "\"snapshot_kind\":\"delta\",\"active_resources\":[],"
            "\"cursor_patches\":[],\"descriptor_commits\":[],"
            "\"pipeline_transition_runs\":[]";
        std::string ok = "{\"node_transitions\":[{\"logical_seq\":1,"
                         "\"tick\":10,\"kind\":\"issue\",\"work_id\":\"w\","
                         "\"payload\":{}}]," + tail + ",\"tick\":10}";
        const ParsedEnvelope parsed = parse_text(ok);
        check(parsed.node_events.size() == 1, "sections before header");
        std::string bad = "{\"node_transitions\":[{\"logical_seq\":1,"
                          "\"tick\":11,\"kind\":\"issue\",\"work_id\":\"w\","
                          "\"payload\":{}}]," + tail + ",\"tick\":10}";
        check_throws([&] { parse_text(bad); },
                     "logical sequence/tick is invalid", "late tick check");
    }
    // Pending-section path: cursor_patches/active_resources alphabetically
    // precede descriptor_commits in any nlohmann dump, so the envelope
    // fixtures of the descriptor group exercise the buffering implicitly.
}

void run_canonical_tests() {
    using jointledger::canonical_envelope_bytes;
    using jointledger::rowport::canonical_json_dumps;
    // 46: key sorting, separators, nesting, empty containers.
    check(canonical_json_dumps(Json::parse(
              R"({"b":1,"a":[1,2],"c":{},"d":[]})"))
          == R"({"a":[1,2],"b":1,"c":{},"d":[]})", "canonical basic");
    check(canonical_json_dumps(Json::parse("null")) == "null",
          "canonical null");
    check(canonical_json_dumps(Json::parse("true")) == "true",
          "canonical bool");
    // 47: Python repr(float) battery.
    struct FloatCase { const char* in; const char* out; };
    const FloatCase cases[] = {
        {"1.0", "1.0"}, {"0.1", "0.1"}, {"1e30", "1e+30"},
        {"1.5e-7", "1.5e-07"}, {"123456789.125", "123456789.125"},
        {"9007199254740992.0", "9007199254740992.0"},
        {"0.5", "0.5"}, {"1e15", "1000000000000000.0"}, {"1e16", "1e+16"},
        {"-0.0", "-0.0"}, {"1e-4", "0.0001"}, {"1.25e-5", "1.25e-05"},
        {"3.0e2", "300.0"}, {"5e-324", "5e-324"},
        {"1.7976931348623157e308", "1.7976931348623157e+308"},
        {"2.5", "2.5"}, {"100.0", "100.0"}, {"0.25", "0.25"}};
    for (const FloatCase& item : cases) {
        const std::string text = std::string(R"({"v":)") + item.in + "}";
        const std::string expected =
            std::string(R"({"v":)") + item.out + "}";
        const std::string actual =
            canonical_json_dumps(Json::parse(text));
        check(actual == expected, item.in);
    }
    // 48: strings -- escapes, control characters and raw non-ASCII.
    check(canonical_json_dumps(Json::parse(
              "{\"k\":\"a\\\"b\\\\c\\nd\\u001f\"}"))
              == "{\"k\":\"a\\\"b\\\\c\\nd\\u001f\"}", "escapes");
    check(canonical_json_dumps(Json::parse(
              "{\"k\":\"\\u4e2d\\u6587\"}")) == "{\"k\":\"中文\"}",
          "raw unicode");
    // 49: int/float distinction survives.
    check(canonical_json_dumps(Json::parse(R"({"i":5,"f":5.0})"))
          == R"({"f":5.0,"i":5})", "int vs float");
    check(canonical_json_dumps(Json::parse("9007199254740993"))
          == "9007199254740993", "big int stays int");
    // sha256 sanity (FIPS 180-4 vector) and the byte-offset error path.
    check(jointledger::rowport::sha256_hex("abc")
              == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb"
                 "410ff61f20015ad", "sha256 abc");
    check_throws([] { canonical_envelope_bytes("{", 1); }, "byte",
                 "canonical parse error");
}

void run_descriptor_dependent_tests() {
    using jointledger::rowport::parse_credit_patches;
    using jointledger::rowport::parse_obligation_patches;
    using jointledger::rowport::parse_v4_slot_patches;

    // 50: descriptor commit parses; indexes are derived additions.
    {
        const ParsedEnvelope parsed =
            parse_ok(descriptor_envelope(Json::array()));
        check(parsed.descriptors.size() == 1, "one descriptor");
        check(parsed.descriptors[0].first->identity.segment_id == kSegment,
              "descriptor identity");
        check(parsed.descriptors[0].second["commit_tick"] == u(3),
              "commit entry raw");
        check(parsed.selector_index_additions.count(kSegment) == 1,
              "selector index addition");
        check(parsed.dependency_index_additions.count(kSegment) == 1,
              "dependency index addition");
        const auto& selector_index =
            parsed.selector_index_additions.at(kSegment);
        check(selector_index.run_intervals.count(
                  std::make_tuple(std::uint64_t{0}, std::uint64_t{1})) == 1,
              "run interval for rank 1");
        const auto& rows = selector_index.run_intervals.at(
            std::make_tuple(std::uint64_t{0}, std::uint64_t{1}));
        check(rows.first == std::vector<std::uint64_t>{1}
                  && rows.second
                      == std::vector<std::array<std::uint64_t, 3>>{
                          {1, 4, 1}},
              "run interval rows");
        const auto& dependency =
            parsed.dependency_index_additions.at(kSegment);
        check(dependency.intervals_by_prefix.count("req#decode:1:r0n") == 1
                  && dependency.intervals_by_prefix.count(
                         "req#decode:1:r1n") == 1,
              "dependency prefixes");
        const auto& prefix_rows =
            dependency.intervals_by_prefix.at("req#decode:1:r0n");
        check(prefix_rows.size() == 1
                  && prefix_rows[0][0] == 52 && prefix_rows[0][1] == 56
                  && prefix_rows[0][4] == 2 && prefix_rows[0][5] == 0,
              "dependency interval 52..56");
        const auto& rank_rows = dependency.intervals_by_rank.at(1);
        check(rank_rows[0][0] == 55 && rank_rows[0][1] == 59,
              "dependency interval 55..59");
    }
    // 51: duplicate declaration inside one delivery.
    {
        Json header = base_header();
        Json commits = Json::array({descriptor_commit_entry(),
                                    descriptor_commit_entry()});
        header["descriptor_commits"] = commits;
        parse_fail(header, "duplicate descriptor declaration");
    }
    // 52: identity must equal the descriptor's own identity.
    {
        Json entry = descriptor_commit_entry();
        entry["identity"]["generation"] = u(7);
        Json header = base_header();
        header["descriptor_commits"] = Json::array({entry});
        parse_fail(header, "identity differs from immutable descriptor");
    }
    // 53: commit key set.
    {
        Json entry = descriptor_commit_entry();
        entry["surprise"] = u(1);
        Json header = base_header();
        header["descriptor_commits"] = Json::array({entry});
        parse_fail(header, "keys invalid");
    }
    // 54: endpoint-bound run resolves the selector and geometry.
    {
        const Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        const ParsedEnvelope parsed =
            parse_ok(descriptor_envelope(Json::array({row})));
        check(parsed.pipeline_runs.size() == 1, "one run");
        const Json& parsed_row = parsed.pipeline_runs[0];
        check(parsed_row["endpoint_run_index"] == u(0), "run index");
        check(parsed_row["endpoint_leg_index"] == u(0), "leg index");
        check(parsed_row["endpoint_ordinal"] == u(0), "endpoint ordinal");
        check(parsed_row["endpoint_store_id"] == u(52), "store id");
        check(parsed_row["_selected_global_tile_ordinal"] == u(1),
              "selected tile");
        check(parsed_row["_selected_shard_tile"]["data_bytes"] == u(2)
                  && parsed_row["_selected_shard_tile"]["emit_endpoint"]
                      == true,
              "selected shard tile");
        check(parsed_row["transition_kind"] == "tile_issue", "transition");
    }
    // 55: endpoint fields must be all null or all present.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"));
        patch["legacy_context"] = nullptr;
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "all null or all present");
    }
    // 56: strides must be zero while repeat_count=1.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"));
        patch["tile_stride"] = u(1);
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "must be 0 while repeat_count=1");
    }
    // 57: store_id_first == endpoint JSON ID + 1.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 53,
                                        selector(0, 1, 0, "data_send"));
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "store_id_first must equal endpoint JSON ID + 1");
    }
    // 58: legacy context must match the endpoint run.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"));
        patch["legacy_context"]["generation"] = u(2);
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "legacy_context disagrees with endpoint run");
    }
    // 59: descriptor-level transition cannot bind an endpoint.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"),
                                        "service_done");
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "descriptor-level transition cannot bind");
    }
    // 60: endpoint_done requires a selector.
    parse_fail(descriptor_envelope(
                   Json::array({run_row(Json{{"transition_kind",
                                             "endpoint_done"}})})),
               "endpoint_done transition requires a concrete selector");
    // 61: endpoint transitions require a selector.
    parse_fail(descriptor_envelope(
                   Json::array({run_row(Json{{"transition_kind",
                                             "data_ready"}})})),
               "endpoint transition requires a concrete selector");
    // 62: consume_done without a selector is descriptor-level and valid.
    {
        const ParsedEnvelope parsed = parse_ok(descriptor_envelope(
            Json::array({run_row(Json{{"transition_kind", "consume_done"}})})));
        check(parsed.pipeline_runs[0]["endpoint_selector"].is_null(),
              "consume_done without selector");
    }
    // 63: resource-bounded endpoint quantity matches the descriptor tile.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"));
        patch["resource_leg"] = "endpoint";
        patch["resource_quantity_prefix"] = Json::array(
            {Json{{"ops", nullptr}, {"bytes", u(0)}, {"latency_ns", nullptr}},
             Json{{"ops", nullptr}, {"bytes", u(2)}, {"latency_ns", nullptr}}});
        const ParsedEnvelope parsed =
            parse_ok(descriptor_envelope(Json::array({run_row(patch)})));
        check(parsed.pipeline_runs.size() == 1, "resource leg accepted");
    }
    // 64: wrong amount rejected.
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"));
        patch["resource_leg"] = "endpoint";
        patch["resource_quantity_prefix"] = Json::array(
            {Json{{"ops", nullptr}, {"bytes", u(0)}, {"latency_ns", nullptr}},
             Json{{"ops", nullptr}, {"bytes", u(3)}, {"latency_ns", nullptr}}});
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "resource quantity does not match descriptor endpoint");
    }
    // 65: baseline must be integer zero (float 0.0 rejected).
    {
        Json patch = endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"));
        patch["resource_leg"] = "endpoint";
        patch["resource_quantity_prefix"] = Json::array(
            {Json{{"ops", nullptr}, {"bytes", 0.0}, {"latency_ns", nullptr}},
             Json{{"ops", nullptr}, {"bytes", u(2)}, {"latency_ns", nullptr}}});
        parse_fail(descriptor_envelope(Json::array({run_row(patch)})),
                   "baseline must be integer zero");
    }
    // 66: quantities require a resource leg.
    {
        Json row = run_row(Json{});
        row["resource_quantity_prefix"] = Json::array(
            {Json{{"ops", nullptr}, {"bytes", u(0)}, {"latency_ns", nullptr}},
             Json{{"ops", nullptr}, {"bytes", u(2)}, {"latency_ns", nullptr}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "resource quantities require a resource_leg");
    }
    // 67: nullability must not change.
    {
        Json row = run_row(Json{});
        row["resource_quantity_prefix"] = Json::array(
            {Json{{"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}},
             Json{{"ops", nullptr}, {"bytes", u(2)}, {"latency_ns", nullptr}}});
        row["resource_leg"] = nullptr;
        parse_fail(descriptor_envelope(Json::array({row})),
                   "resource prefix changes bytes nullability");
    }
    // 68: cursor version and tick stride gates.
    {
        Json row = run_row(Json{{"cursor_version_end", u(2)}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "cursor version transition");
    }
    {
        Json row = run_row(Json{{"tick_stride", u(1)}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "tick_stride must be 0");
    }
    // 69-71: target_delta slot patches (no descriptor needed for the ops
    // themselves: empty views).
    const jointledger::rowport::DescriptorView no_descriptors{};
    const jointledger::rowport::SelectorView no_indexes{};
    {
        Json patches = Json::array({Json{
            {"form", "target_delta"},
            {"key", Json{{"buffer_domain_id", kDomain}, {"role", "target"},
                         {"rank", u(1)}, {"shard_ordinal", nullptr},
                         {"slot_index", u(0)}}},
            {"before_slot_generation", u(0)},
            {"predecessor_logical_seq", u(0)},
            {"predecessor_consumed_count", u(0)},
            {"operations", Json::array({Json{{"op", "initialize"},
                                             {"slot_generation", u(1)}}})}}});
        const Json parsed = parse_v4_slot_patches(patches, "slot_patches",
                                                  no_descriptors,
                                                  no_indexes);
        check(parsed[0]["form"] == "target_delta", "target_delta form");
        check(parsed[0]["key"][0] == kDomain && parsed[0]["key"][1] == "target"
                  && parsed[0]["key"][4] == u(0), "target_delta key array");
    }
    {
        Json patches = Json::array({Json{
            {"form", "target_delta"},
            {"key", Json{{"buffer_domain_id", kDomain}, {"role", "target"},
                         {"rank", u(1)}, {"shard_ordinal", nullptr},
                         {"slot_index", u(0)}}},
            {"before_slot_generation", u(0)},
            {"predecessor_logical_seq", u(0)},
            {"predecessor_consumed_count", u(0)},
            {"operations", Json::array(
                {Json{{"op", "claim"}, {"slot_generation", u(1)},
                      {"owner", slot_owner()},
                      {"target_shards", Json::array({u(0)})}},
                 Json{{"op", "consume"}, {"slot_generation", u(1)},
                      {"cause", Json{{"terminal_rank", u(1)},
                                     {"terminal_store_id", u(70)},
                                     {"consume_run_index", u(0)}}},
                      {"add_shards", Json::array({u(0), u(1)})},
                      {"cleared", false}}})}}});
        const Json parsed = parse_v4_slot_patches(patches, "slot_patches",
                                                  no_descriptors,
                                                  no_indexes);
        check(parsed[0]["operations"][0]["op"] == "claim"
                  && parsed[0]["operations"][0]["target_shards"].size() == 1,
              "claim op");
        check(parsed[0]["operations"][1]["cause"]["terminal_store_id"]
                  == u(70), "consume op");
    }
    {
        auto bad_ops = [&](Json operations) {
            Json patches = Json::array({Json{
                {"form", "target_delta"},
                {"key", Json{{"buffer_domain_id", kDomain},
                             {"role", "target"}, {"rank", u(1)},
                             {"shard_ordinal", nullptr}, {"slot_index", u(0)}}},
                {"before_slot_generation", u(0)},
                {"predecessor_logical_seq", u(0)},
                {"predecessor_consumed_count", u(0)},
                {"operations", operations}}});
            return parse_v4_slot_patches(patches, "slot_patches",
                                         no_descriptors, no_indexes);
        };
        check_throws([&] { bad_ops(Json::array({Json{{"op", "rename"},
                                                    {"slot_generation",
                                                     u(1)}}})); },
                     "op is unknown", "unknown op");
        check_throws([&] { bad_ops(Json::array()); },
                     "operations must be nonempty", "empty operations");
        check_throws([&] {
            bad_ops(Json::array({Json{{"op", "claim"},
                                      {"slot_generation", u(1)},
                                      {"owner", slot_owner()},
                                      {"target_shards",
                                       Json::array({u(1), u(0)})}}})); },
                     "target_shards must be sorted unique", "unsorted shards");
        check_throws([&] {
            Json one = Json{
                {"form", "target_delta"},
                {"key", Json{{"buffer_domain_id", kDomain},
                             {"role", "target"}, {"rank", u(1)},
                             {"shard_ordinal", nullptr},
                             {"slot_index", u(0)}}},
                {"before_slot_generation", u(0)},
                {"predecessor_logical_seq", u(0)},
                {"predecessor_consumed_count", u(0)},
                {"operations", Json::array({Json{{"op", "initialize"},
                                                 {"slot_generation",
                                                  u(1)}}})}};
            parse_v4_slot_patches(Json::array({one, one}),
                                  "slot_patches", no_descriptors,
                                  no_indexes); },
                     "sorted and duplicate-free", "duplicate slot patch");
    }
    // 72/73/74/75: source slot patches inside a run (descriptor-bound).
    {
        Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        row["slot_patches"] = Json::array({Json{
            {"before", nullptr}, {"after", source_slot_row(1)}}});
        const ParsedEnvelope parsed =
            parse_ok(descriptor_envelope(Json::array({row})));
        const Json& patch = parsed.pipeline_runs[0]["slot_patches"][0];
        check(patch[0] == "source", "source patch tag");
        check(patch[1][2] == u(0) && patch[1][4] == u(0), "source key");
        check(patch[3]["owner"]["segment_id"] == kSegment, "owner row");
    }
    {
        Json bad_slot = source_slot_row(1);
        bad_slot["slot_index"] = u(1);  // global tile 1 needs slot 0
        Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        row["slot_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", bad_slot}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "two-slot tile parity");
    }
    {
        Json bad_slot = source_slot_row(1);
        bad_slot["owner"]["stage_epoch"] = u(9);
        Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        row["slot_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", bad_slot}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "owner disagrees with immutable descriptor identity");
    }
    {
        Json target_row = Json{{"buffer_domain_id", kDomain},
                               {"role", "target"}, {"rank", u(1)},
                               {"shard_ordinal", nullptr},
                               {"slot_index", u(0)}, {"occupied", true},
                               {"owner", slot_owner()},
                               {"consumed", false},
                               {"target_shards_to_consume", Json::array({u(0)})},
                               {"consumed_target_shards", Json::array()}};
        Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        row["slot_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", target_row}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "v4 target slots must use target_delta form");
    }
    // 76: credit patches.
    {
        const Json credit_row = Json{{"buffer_domain_id", kDomain},
                                     {"source_rank", u(0)},
                                     {"shard_ordinal", u(0)},
                                     {"capacity", u(2)}, {"available", u(1)},
                                     {"outstanding_ack_count", u(0)}};
        const Json parsed = parse_credit_patches(
            Json::array({Json{{"before", nullptr}, {"after", credit_row}}}),
            "credit_patches");
        check(parsed[0][0][0] == kDomain && parsed[0][0][1] == u(0),
              "credit key");
        check(parsed[0][1].is_null() && parsed[0][2]["capacity"] == u(2),
              "credit before/after");
        check_throws([&] {
            parse_credit_patches(
                Json::array({Json{{"before", nullptr},
                                  {"after", Json{{"buffer_domain_id", kDomain},
                                                 {"source_rank", u(0)},
                                                 {"shard_ordinal", u(0)},
                                                 {"capacity", u(3)},
                                                 {"available", u(1)},
                                                 {"outstanding_ack_count",
                                                  u(0)}}}}}),
                "credit_patches"); }, "capacity=2", "capacity 3");
        check_throws([&] {
            parse_credit_patches(
                Json::array({Json{{"before", credit_row},
                                  {"after", credit_row}}}),
                "credit_patches"); }, "no-op patch", "no-op credit");
        check_throws([&] {
            parse_credit_patches(
                Json::array({Json{{"before", nullptr},
                                  {"after", Json{{"buffer_domain_id", kDomain},
                                                 {"source_rank", u(0)},
                                                 {"shard_ordinal", u(1)},
                                                 {"capacity", u(2)},
                                                 {"available", u(1)},
                                                 {"outstanding_ack_count",
                                                  u(0)}}}},
                             Json{{"before", nullptr},
                                  {"after", credit_row}}}),
                "credit_patches"); }, "sorted and duplicate-free",
                "unsorted credits");
    }
    // 77: obligation patches.
    {
        const Json obligation = Json{{"obligation_id", "o1"},
                                     {"segment_id", kSegment},
                                     {"rank", u(1)},
                                     {"first_store_id", u(70)},
                                     {"remaining_endpoint_count", u(2)},
                                     {"state", "pending"}};
        check_throws([&] {
            parse_obligation_patches(
                Json::array({Json{{"before", nullptr},
                                  {"after", obligation}}}),
                "obligation_patches", no_descriptors); },
                     "references unknown segment", "obligation no registry");
        Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        Json bad = obligation;
        bad["segment_id"] = "pp1/none";
        row["obligation_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", bad}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "references unknown segment");
        Json bad_state = obligation;
        bad_state["state"] = "done";
        row["obligation_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", bad_state}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "pending/active with positive remaining count");
        Json zero = obligation;
        zero["remaining_endpoint_count"] = u(0);
        row["obligation_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", zero}}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "pending/active with positive remaining count");
        row["obligation_patches"] =
            Json::array({Json{{"before", nullptr}, {"after", obligation}}});
        const ParsedEnvelope parsed_env =
            parse_ok(descriptor_envelope(Json::array({row})));
        check(parsed_env.pipeline_runs[0]["obligation_patches"][0][0]
                  == "o1", "obligation key");
        check(parsed_env.pipeline_runs[0]["obligation_patches"][0][2]
                  ["remaining_endpoint_count"] == u(2),
              "obligation after row");
    }
    // 78-84: cursor patches.
    {
        const ParsedEnvelope parsed = parse_ok(
            descriptor_envelope(Json::array(), Json::array({cursor_patch(1)})));
        check(parsed.cursor_patches.size() == 1, "one cursor patch");
        const Json& patch = parsed.cursor_patches[0];
        check(patch["segment_id"] == kSegment && patch["retired"] == false,
              "cursor patch header");
        check(patch["runs"][0][0] == u(1), "cursor run index");
        check(patch["runs"][0][1]["next_store_id"] == u(55),
              "cursor next store");
    }
    {
        Json patch = cursor_patch(1);
        patch["runs"][0]["value"]["next_store_id"] = u(56);
        parse_fail(descriptor_envelope(Json::array(), Json::array({patch})),
                   "next_store_id disagrees with cursor position");
    }
    {
        const ParsedEnvelope parsed = parse_ok(descriptor_envelope(
            Json::array(), Json::array({cursor_patch(0, true)})));
        check(parsed.cursor_patches[0]["runs"].size() == 2
                  && parsed.cursor_patches[0]["retired"] == true,
              "retired tombstones");
    }
    {
        Json patch = cursor_patch(0, true);
        patch["runs"][0]["value"] = cursor_run_value(0);
        parse_fail(descriptor_envelope(Json::array(), Json::array({patch})),
                   "value must be null on descriptor retirement");
    }
    {
        Json patch = cursor_patch(0, true);
        std::swap(patch["runs"][0], patch["runs"][1]);
        parse_fail(descriptor_envelope(Json::array(), Json::array({patch})),
                   "runs must be sorted and duplicate-free");
    }
    {
        Json patch = cursor_patch(1);
        patch["runs"][0]["value"]["completed_endpoint_count"] = u(1);
        patch["runs"][0]["value"]["completed_out_of_order"] = Json::array(
            {Json{{"first_ordinal", u(0)}, {"end_ordinal_exclusive", u(1)}}});
        patch["runs"][0]["value"]["remaining_endpoint_count"] = u(1);
        parse_fail(descriptor_envelope(Json::array(), Json::array({patch})),
                   "precedes completed prefix");
    }
    {
        Json patch = cursor_patch(1);
        patch["runs"][0]["value"]["completed_endpoint_count"] = u(1);
        patch["runs"][0]["value"]["completed_out_of_order"] = Json::array(
            {Json{{"first_ordinal", u(1)}, {"end_ordinal_exclusive", u(2)}},
             Json{{"first_ordinal", u(1)}, {"end_ordinal_exclusive", u(2)}}});
        patch["runs"][0]["value"]["remaining_endpoint_count"] = u(0);
        parse_fail(descriptor_envelope(Json::array(), Json::array({patch})),
                   "overlaps");
    }
    {
        Json patch = cursor_patch(1);
        patch["runs"][0]["value"]["completed_endpoint_count"] = u(1);
        patch["runs"][0]["value"]["remaining_endpoint_count"] = u(1);
        patch["runs"][0]["value"]["completed_out_of_order"] = Json::array(
            {Json{{"first_ordinal", u(1)}, {"end_ordinal_exclusive", u(3)}}});
        parse_fail(descriptor_envelope(Json::array(), Json::array({patch})),
                   "do not cover its endpoint run");
    }
    {
        // Active endpoints: store id must equal endpoint ID + 1 and the
        // owner must match the descriptor endpoint (run 1, tile 1 ->
        // endpoint 54 -> store 55; cursor advanced to tile 2 -> 57).
        Json value = cursor_run_value(1);
        value["tile_offset"] = u(1);
        value["next_store_id"] = u(57);
        value["in_flight"] = true;
        Json endpoint = Json{
            {"store_id", u(55)}, {"tile_ordinal", u(1)},
            {"shard_ordinal", u(0)}, {"endpoint", "data_recv"},
            {"transfer_id", u(9)}, {"flow_id", u(4)},
            {"network_done", true}, {"hbm_done", false},
            {"owner", Json{{"segment_id", kSegment},
                           {"pipeline_generation", u(1)},
                           {"stage_epoch", u(2)},
                           {"staging_allocation_id", u(9)},
                           {"group_ordinal", u(0)},
                           {"tile_ordinal", u(1)}}}};
        auto patch_with = [&](const Json& run_value) {
            return Json{{"segment_id", kSegment}, {"cursor_version", u(2)},
                        {"retired", false},
                        {"runs", Json::array({Json{{"run_index", u(1)},
                                                   {"value", run_value}}})}};
        };
        {
            Json complete = value;
            complete["completed_endpoint_count"] = u(1);
            complete["remaining_endpoint_count"] = u(2);
            complete["active_endpoints"] = Json::array({endpoint});
            parse_fail(descriptor_envelope(
                           Json::array(), Json::array({patch_with(complete)})),
                       "simultaneously complete and active");
        }
        {
            Json wrong_store = value;
            Json bad = endpoint;
            bad["store_id"] = u(99);
            wrong_store["active_endpoints"] = Json::array({bad});
            parse_fail(descriptor_envelope(
                           Json::array(),
                           Json::array({patch_with(wrong_store)})),
                       "store_id disagrees with endpoint ID + 1");
        }
        {
            Json wrong_owner = value;
            Json bad = endpoint;
            bad["owner"]["group_ordinal"] = u(3);
            wrong_owner["active_endpoints"] = Json::array({bad});
            parse_fail(descriptor_envelope(
                           Json::array(),
                           Json::array({patch_with(wrong_owner)})),
                       "owner disagrees with its descriptor endpoint");
        }
    }
    {
        // Unsorted cursor patches (segment ids must strictly increase).
        parse_fail(descriptor_envelope(
                       Json::array(),
                       Json::array({cursor_patch(1), cursor_patch(0)})),
                   "cursor_patches must be sorted");
    }
    // 85-94: active_resources.
    {
        const ParsedEnvelope parsed = parse_ok(descriptor_envelope(
            Json::array(), Json::array(), Json::array({noc_flow_resource(
                                              Json::object())})));
        check(parsed.resources.size() == 1, "unattributed noc accepted");
    }
    {
        parse_fail(descriptor_envelope(
                       Json::array(), Json::array(),
                       Json::array({noc_flow_resource(
                           Json{{"node_id", u(8)},
                                {"work_id", "noc#:0:r0n8@noc:7"}})})),
                   "unattributed NoC identity must preserve");
    }
    {
        const Json bound = Json{
            {"resource_type", "noc_flow"},
            {"work_id", "req#decode:1:r1n55@noc:9"},
            {"request_id", "req"}, {"stage", "decode"}, {"generation", u(1)},
            {"rank", u(1)}, {"node_id", u(55)}, {"operation", "comm"},
            {"leg", "noc:9"}, {"state", "Active"},
            {"total", Json{{"ops", nullptr}, {"bytes", u(10)},
                           {"latency_ns", nullptr}}},
            {"remaining", Json{{"ops", nullptr}, {"bytes", u(0)},
                               {"latency_ns", nullptr}}},
            {"progress_source", "authoritative"}, {"service_unit", "bytes"},
            {"route", Json::array()}, {"as_of_tick", u(5)},
            {"issued_tick", u(3)}, {"descriptor_segment_id", kSegment},
            {"endpoint_selector", selector(0, 1, 0, "data_recv")},
            {"flow_id", u(9)}, {"current_rate_Bpns", 1.0},
            {"last_rate_update_time", 0.5}};
        const ParsedEnvelope parsed = parse_ok(
            descriptor_envelope(Json::array(), Json::array(),
                                Json::array({bound})));
        check(parsed.resources.size() == 1, "bound noc accepted");
        Json wrong_node = bound;
        wrong_node["node_id"] = u(54);
        wrong_node["work_id"] = "req#decode:1:r1n54@noc:9";
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({wrong_node})),
                   "NodeStore mapping is inconsistent");
        Json bad_state = bound;
        bad_state["state"] = "Streaming";
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({bad_state})),
                   ".state is invalid for noc_flow");
    }
    {
        Json non_canonical = noc_flow_resource(Json::object());
        non_canonical["work_id"] = "other#x:0:r0n7@noc:7";
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({non_canonical})),
                   "work_id is not canonical");
    }
    {
        Json over = noc_flow_resource(Json::object());
        over["total"] = Json{{"ops", nullptr}, {"bytes", u(3)},
                             {"latency_ns", nullptr}};
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({over})),
                   "remaining bytes exceeds total");
    }
    {
        const Json external = Json{
            {"resource_type", "hbm_job"},
            {"work_id", "external_background#:0:r3n9@hbm_external:3#5"},
            {"request_id", "external_background"}, {"stage", ""},
            {"generation", u(0)}, {"rank", u(3)}, {"node_id", u(9)},
            {"operation", "comm"}, {"leg", "hbm_external:3#5"},
            {"state", "issued"},
            {"total", Json{{"ops", nullptr}, {"bytes", u(8)},
                           {"latency_ns", nullptr}}},
            {"remaining", Json{{"ops", nullptr}, {"bytes", u(8)},
                               {"latency_ns", nullptr}}},
            {"progress_source", "authoritative"}, {"service_unit", "bytes"},
            {"route", Json::array()}, {"as_of_tick", u(6)},
            {"issued_tick", u(4)}, {"descriptor_segment_id", nullptr},
            {"endpoint_selector", nullptr}, {"kind", "COMM_READ"},
            {"start_tick", u(4)}};
        const ParsedEnvelope parsed = parse_ok(descriptor_envelope(
            Json::array(), Json::array(), Json::array({external})));
        check(parsed.resources.size() == 1, "external background hbm");
        Json wrong_kind = external;
        wrong_kind["kind"] = "COMM_OTHER";
        wrong_kind["leg"] = "hbm_external:3#5";
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({wrong_kind})),
                   "transfer-sink COMM_READ/COMM_WRITE");
        Json wrong_rank = external;
        wrong_rank["leg"] = "hbm_external:4#5";
        wrong_rank["work_id"] = "external_background#:0:r3n9@hbm_external:4#5";
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({wrong_rank})),
                   "external background form");
    }
    {
        const Json pool = Json{
            {"resource_type", "pool_job"},
            {"work_id", "req#decode:1:r0n1@pool:0:17"},
            {"request_id", "req"}, {"stage", "decode"}, {"generation", u(1)},
            {"rank", u(0)}, {"node_id", u(1)}, {"operation", "pool_read"},
            {"leg", "pool:0:17"}, {"state", "LatencyWaiting"},
            {"total", Json{{"ops", nullptr}, {"bytes", u(4)},
                           {"latency_ns", nullptr}}},
            {"remaining", Json{{"ops", nullptr}, {"bytes", u(4)},
                               {"latency_ns", nullptr}}},
            {"progress_source", "authoritative"}, {"service_unit", "bytes"},
            {"route", Json::array()}, {"as_of_tick", u(6)},
            {"issued_tick", u(4)}, {"descriptor_segment_id", nullptr},
            {"endpoint_selector", nullptr}, {"port_index", u(0)},
            {"issue_seq", u(17)}, {"latency_ready_ns", 2.0}};
        const ParsedEnvelope parsed = parse_ok(descriptor_envelope(
            Json::array(), Json::array(), Json::array({pool})));
        check(parsed.resources.size() == 1, "pool job accepted");
        Json mismatched = pool;
        mismatched["leg"] = "pool:1:17";
        mismatched["work_id"] = "req#decode:1:r0n1@pool:1:17";
        parse_fail(descriptor_envelope(Json::array(), Json::array(),
                                       Json::array({mismatched})),
                   "leg must identify its original pool job");
    }
    {
        // resources must be sorted by (segment, rank, node, leg, type);
        // both rows are individually valid unattributed NoC flows.
        parse_fail(descriptor_envelope(
                       Json::array(), Json::array(),
                       Json::array({noc_flow_resource(
                                        Json{{"node_id", u(8)},
                                             {"flow_id", u(8)},
                                             {"leg", "noc:8"},
                                             {"work_id",
                                              "noc#:0:r0n8@noc:8"}}),
                                    noc_flow_resource(Json::object())})),
                   "active_resources must be sorted");
    }
    // 95: runs must preserve logical sequence order.
    {
        Json second = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        second["first_logical_seq"] = u(1);
        Json header = descriptor_envelope(Json::array({second}));
        header["pipeline_transition_runs"] = Json::array(
            {run_row(endpoint_run_patch(1, 0, 0, 52,
                                        selector(0, 1, 0, "data_send"))),
             second});
        header["logical_event_count"] = u(2);
        header["logical_seq_end"] = u(2);
        header["snapshot_kind"] = "delta";
        parse_fail(header, "must preserve logical sequence order");
    }
    // 96: runs reference only declared/retained descriptors.
    {
        Json row = run_row(Json{{"descriptor_segment_id", "pp1/unknown"}});
        parse_fail(descriptor_envelope(Json::array({row})),
                   "references undeclared descriptor");
    }
}

bool descriptor_backend_ready() {
    try {
        const auto descriptor =
            jointledger::PipelineDescriptor::from_wire(
                Json::parse(kDescriptorWire));
        return descriptor != nullptr;
    } catch (...) {
        return false;
    }
}

// 97: retained-registry branches of _parse_descriptor_commits.
void run_retained_registry_tests() {
    const auto descriptor =
        jointledger::PipelineDescriptor::from_wire(
            Json::parse(kDescriptorWire));
    std::unordered_map<std::string, jointledger::DescriptorPtr> retained;
    retained.emplace(kSegment, descriptor);
    jointledger::SelectorIndex retained_index =
        jointledger::build_selector_index(*descriptor);
    std::unordered_map<std::string, jointledger::SelectorIndex>
        retained_indexes;
    retained_indexes.emplace(kSegment, std::move(retained_index));
    RetainedRegistry registry{&retained, &retained_indexes};

    Json header = base_header();
    header["descriptor_commits"] = Json::array({descriptor_commit_entry()});
    const std::string text = header.dump();
    {
        // Identical redeclaration with allow_retained_replay: accepted, no
        // new selector index (retained one wins), no dependency addition.
        const ParsedEnvelope parsed = parse_envelope_v4(
            text.data(), text.size(), registry, true, kDelivery, kTick);
        check(parsed.descriptors.size() == 1, "retained replay accepted");
        check(parsed.selector_index_additions.count(kSegment) == 0,
              "no duplicate selector addition");
        check(parsed.dependency_index_additions.count(kSegment) == 0,
              "no dependency addition for retained descriptor");
    }
    {
        check_throws([&] { parse_envelope_v4(text.data(), text.size(),
                                             registry, false, kDelivery,
                                             kTick); },
                     "descriptor segment was declared more than once",
                     "redeclaration without replay");
    }
    {
        Json changed = descriptor_commit_entry();
        // Keep the canonical identity/protocol encodings intact but change
        // a free-form prediction field, so from_wire still accepts the wire
        // while the descriptor differs from the retained one.
        changed["descriptor"]["prediction_profile"]["decode_consume"]
            ["compute_ns_total"] = 241.0;
        Json other = base_header();
        other["descriptor_commits"] = Json::array({changed});
        const std::string changed_text = other.dump();
        check_throws([&] {
            parse_envelope_v4(changed_text.data(), changed_text.size(),
                              registry, true, kDelivery, kTick);
        }, "immutable descriptor changed after declaration",
           "redeclaration with changed wire");
    }
    {
        // Retained descriptor without a retained selector index still
        // yields a selector addition, and rows may reference it.
        RetainedRegistry indexless{&retained, nullptr};
        Json row = run_row(endpoint_run_patch(
            1, 0, 0, 52, selector(0, 1, 0, "data_send")));
        Json with_run = base_header();
        with_run["pipeline_transition_runs"] = Json::array({row});
        const ParsedEnvelope parsed = parse_envelope_v4(
            with_run.dump().data(), with_run.dump().size(), indexless,
            false, kDelivery, kTick);
        check(parsed.pipeline_runs.size() == 1,
              "retained descriptor resolves endpoint rows");
    }
}

}  // namespace

int main() {
    run_header_tests();
    run_node_tests();
    run_sax_structure_tests();
    run_canonical_tests();
    if (descriptor_backend_ready()) {
        run_descriptor_dependent_tests();
        run_retained_registry_tests();
    } else {
        std::printf("SKIPPED_DEPENDENCY: descriptor-dependent cases "
                    "(PipelineDescriptor::from_wire unavailable)\n");
    }
    std::printf("%d checks, %d failures\n", g_checks, g_failures);
    return g_failures == 0 ? 0 : 1;
}
