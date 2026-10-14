// pipeline_descriptor_port_test.cc -- port of the joint/test_pipeline_descriptor.py
// coverage (plus representative bad-input families from pipeline_ledger.py
// descriptor consumption) against the C++ PipelineDescriptor port.
#include <cstdio>
#include <string>

#include "joint_ledger/PipelineDescriptor.hh"

namespace jl = jointledger;
using jl::Json;

static int g_checks = 0;
static int g_failures = 0;

#define CHECK(cond)                                                      \
    do {                                                                 \
        ++g_checks;                                                      \
        if (!(cond)) {                                                   \
            ++g_failures;                                                \
            std::printf("FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        }                                                                \
    } while (0)

static void expect_accept(const Json& wire) {
    ++g_checks;
    try {
        jl::PipelineDescriptor::from_wire(wire);
        return;
    } catch (const jl::LedgerError& error) {
        ++g_failures;
        std::printf("FAIL expected acceptance, got LedgerError: %s\n",
                    error.what());
    }
}

static void expect_reject(const Json& wire, const char* needle) {
    ++g_checks;
    try {
        jl::PipelineDescriptor::from_wire(wire);
    } catch (const jl::LedgerError& error) {
        const std::string message = error.what();
        if (message.find(needle) == std::string::npos) {
            ++g_failures;
            std::printf(
                "FAIL rejection message mismatch: expected '%s' in '%s'\n",
                needle, message.c_str());
        }
        return;
    }
    ++g_failures;
    std::printf("FAIL expected LedgerError containing '%s'\n", needle);
}

// Port of the Python test_pipeline_descriptor._wire() fixture.
static Json make_wire() {
    return Json::parse(R"({
        "descriptor_version": 1,
        "identity": {
            "request_id": "req",
            "stage": "decode",
            "generation": 1,
            "stage_epoch": 2,
            "segment_ordinal": 0,
            "rank": 1,
            "staging_allocation_id": 9,
            "segment_id": "pp1/726571/decode/1/2/1/0",
            "buffer_domain_id": "pb1/726571/1/9"
        },
        "stream_kind": "decode_remote_credit",
        "groups": [{
            "group_ordinal": 0,
            "pass_or_window_ordinal": 0,
            "source_ordinal": 0,
            "source_key": "source-0",
            "layer_start": 0,
            "layer_end": 1,
            "tile_axis": {
                "unit": "stream_bytes",
                "total_units": 5,
                "units_per_tile": 2,
                "tile_count": 3,
                "tail_units": 1,
                "global_tile_first": 1
            },
            "shards": [{
                "shard_ordinal": 0,
                "source_rank": 0,
                "target_rank": 1,
                "total_data_bytes": 5,
                "bytes_per_unit": 1,
                "noc_path": [0, 1]
            }]
        }],
        "protocol": {
            "buffer_slots": 2,
            "release_lag_tiles": 2,
            "initial_tiles_without_release": 2,
            "fill_ack_bytes": 1,
            "release_ack_bytes": 1,
            "hbm_charge": true,
            "target_rank_order": "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1",
            "source_rank_order": "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i",
            "tile_branching": "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"
        },
        "prediction_profile": {
            "prefill_consume": "zero_duration_no_extra_consume",
            "decode_consume": {
                "kind": "profiled_consume",
                "profile_id": "decode-service-profile-v1",
                "compute_ns_total": 240.0,
                "hbm_bytes_total": 5,
                "resource_key": "hbm:1",
                "share_rule": "existing_action_work_graph_byte_share_vector"
            }
        },
        "execution_profile": {
            "data_legs": "ordinary_comm_send_recv_path",
            "endpoint_hbm_rule": "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
            "prefill_consume": "existing_body_or_marker_only",
            "decode_consume": "existing_body_nodes_only"
        },
        "logical_id_ranges": [
            {"rank": 0, "first_json_id": 53, "count": 2, "stride": 2,
             "kind": "virtual_endpoint_hole"},
            {"rank": 1, "first_json_id": 56, "count": 2, "stride": 2,
             "kind": "virtual_endpoint_hole"}
        ],
        "tag_allocation_runs": [{
            "allocator_scope_id": "builder-0",
            "emission_ordinal": 7,
            "first_tag": 101,
            "count": 3,
            "group_ordinal": 0,
            "first_tile": 1,
            "first_shard": 0,
            "tile_count": 3,
            "shard_count": 1
        }],
        "endpoint_runs": [
            {
                "rank": 0,
                "group_ordinal": 0,
                "first_tile": 1,
                "tile_count": 3,
                "shard_ordinals": [0],
                "first_json_id": 51,
                "tile_stride": 2,
                "shard_stride": 1,
                "initial_parent_json_id": 50,
                "legacy_context": {
                    "request_id": "req", "stage": "decode",
                    "generation": 1
                },
                "legs": [{
                    "endpoint": "data_send",
                    "json_offset": 0,
                    "tag_emission_ordinal": 7,
                    "tag_offset": 0,
                    "tag_tile_stride": 1,
                    "tag_shard_stride": 1
                }]
            },
            {
                "rank": 1,
                "group_ordinal": 0,
                "first_tile": 1,
                "tile_count": 3,
                "shard_ordinals": [0],
                "first_json_id": 54,
                "tile_stride": 2,
                "shard_stride": 1,
                "initial_parent_json_id": 53,
                "legacy_context": {
                    "request_id": "req", "stage": "decode",
                    "generation": 1
                },
                "legs": [{
                    "endpoint": "data_recv",
                    "json_offset": 0,
                    "tag_emission_ordinal": 7,
                    "tag_offset": 0,
                    "tag_tile_stride": 1,
                    "tag_shard_stride": 1
                }]
            }
        ],
        "endpoint_dependency_runs": [{
            "rank": 0,
            "group_ordinal": 0,
            "first_tile": 1,
            "tile_count": 3,
            "shard_ordinals": [0],
            "endpoint": "data_send",
            "dependency_kind": "ordinary_node_terminal",
            "first_dependency_json_id": 50,
            "dependency_tile_stride": 0,
            "dependency_shard_stride": 0
        }],
        "consume_completion_runs": [{
            "rank": 1,
            "group_ordinal": 0,
            "first_tile": 1,
            "tile_count": 3,
            "shard_ordinals": [0],
            "dependency_kind": "ordinary_node_terminal",
            "first_dependency_json_id": 70,
            "dependency_tile_stride": 0,
            "dependency_shard_stride": 0
        }],
        "projection": {
            "version": 1,
            "chains": [{
                "group_ordinal": 0,
                "rank": 1,
                "tile_count": 3,
                "fill": {
                    "leg_id_prefix": "work#dcstream#R1#F",
                    "first_index": 0,
                    "fill_order_first": 0,
                    "fill_order_stride": 1,
                    "completion_order_first": null,
                    "completion_order_stride": 0,
                    "route": ["link:0"],
                    "bytes_per_tile": 2,
                    "latency_ns": 3,
                    "start_deps": ["work#start"],
                    "gate_id": "work#gate:dcstream:R1",
                    "gate_capacity": 2
                },
                "consume": {
                    "leg_id_prefix": "work#dcstream#R1#C",
                    "first_index": 0,
                    "consume_order_first": 3,
                    "consume_order_stride": 1,
                    "completion_order_first": null,
                    "completion_order_stride": 0,
                    "kind": "compute",
                    "route": ["hbm:1"],
                    "group": "decode",
                    "latency_first_ns": 7,
                    "latency_rest_ns": 0,
                    "consume_deps": ["work#consume-start"],
                    "consume_start_deps": [],
                    "gate_id": "work#gate:dcstream:R1",
                    "compute_share": {
                        "rule": "quotient_remainder_earliest_global_indices",
                        "total_units": 8,
                        "total_tiles": 3,
                        "share_index_first": 0,
                        "tile_bytes": 2
                    },
                    "hbm_share": {
                        "rule": "quotient_remainder_earliest_global_indices",
                        "total_units": 5,
                        "total_tiles": 3,
                        "share_index_first": 0,
                        "tile_bytes": 2
                    }
                }
            }]
        },
        "anchors": [
            {
                "role": "start",
                "rank": 0,
                "binding": "physical_endpoint_slot",
                "node_id": 51,
                "selector": {
                    "kind": "tile_endpoint_leg",
                    "group_ordinal": 0,
                    "tile_ordinal": 1,
                    "shard_ordinal": 0,
                    "endpoint": "data_send",
                    "tag_allocation_emission_ordinal": 7,
                    "tag_offset": 0
                }
            },
            {
                "role": "frontier",
                "rank": 1,
                "binding": "physical_endpoint_slot",
                "node_id": 54,
                "selector": {
                    "kind": "tile_endpoint_terminal",
                    "group_ordinal": 0,
                    "tile_ordinal": 1,
                    "shard_ordinal": 0,
                    "endpoint": "data_recv",
                    "tag_allocation_emission_ordinal": 7,
                    "tag_offset": 0
                }
            },
            {
                "role": "service",
                "rank": 1,
                "binding": "descriptor_cursor_selector",
                "node_id": null,
                "selector": {"kind": "existing_service_gate",
                             "existing_gate": "decode_service_gate"}
            },
            {
                "role": "settle",
                "rank": 1,
                "binding": "descriptor_cursor_selector",
                "node_id": null,
                "selector": {"kind": "all_obligations_settled"}
            }
        ]
    })");
}

// Port of the Python _descriptor_with_shard_count helper.
static Json make_shard_count_wire(int shard_count, const char* unit) {
    Json wire = make_wire();
    wire["groups"][0]["tile_axis"]["unit"] = unit;
    Json shards = Json::array();
    for (int index = 0; index < shard_count; ++index) {
        const int source_rank = index == 0 ? 0 : index + 1;
        const int bytes_per_unit =
            std::string(unit) == "source_tokens" ? index + 1 : 1;
        Json shard;
        shard["shard_ordinal"] = index;
        shard["source_rank"] = source_rank;
        shard["target_rank"] = 1;
        shard["total_data_bytes"] = 5 * bytes_per_unit;
        shard["bytes_per_unit"] = bytes_per_unit;
        shard["noc_path"] = Json::array({source_rank, 1});
        shards.push_back(std::move(shard));
    }
    wire["groups"][0]["shards"] = std::move(shards);
    Json ordinals = Json::array();
    for (int index = 0; index < shard_count; ++index) ordinals.push_back(index);
    wire["consume_completion_runs"][0]["shard_ordinals"] = std::move(ordinals);
    wire["consume_completion_runs"][0]["dependency_shard_stride"] = 1;
    return wire;
}

// Standard wire plus a second, shorter stream shard (total 2 bytes) whose own
// consume-completion run covers exactly its single active tile.
static Json make_two_shard_wire() {
    Json wire = make_wire();
    wire["groups"][0]["shards"] = Json::array({
        Json{
            {"shard_ordinal", 0}, {"source_rank", 0}, {"target_rank", 1},
            {"total_data_bytes", 5}, {"bytes_per_unit", 1},
            {"noc_path", Json::array({0, 1})}},
        Json{
            {"shard_ordinal", 1}, {"source_rank", 2}, {"target_rank", 1},
            {"total_data_bytes", 2}, {"bytes_per_unit", 1},
            {"noc_path", Json::array({2, 1})}},
    });
    Json completion = Json::array({
        Json{
            {"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
            {"tile_count", 3}, {"shard_ordinals", Json::array({0})},
            {"dependency_kind", "ordinary_node_terminal"},
            {"first_dependency_json_id", 70},
            {"dependency_tile_stride", 0}, {"dependency_shard_stride", 0}},
        Json{
            {"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
            {"tile_count", 1}, {"shard_ordinals", Json::array({1})},
            {"dependency_kind", "ordinary_node_terminal"},
            {"first_dependency_json_id", 73},
            {"dependency_tile_stride", 0}, {"dependency_shard_stride", 0}},
    });
    wire["consume_completion_runs"] = std::move(completion);
    return wire;
}

// ------------------------------------------------------------- wire shape --

static void test_round_trip_matches_input_wire() {
    Json wire = make_wire();
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(wire);
    CHECK(descriptor->to_wire() == wire);
    const Json again = jl::PipelineDescriptor::from_wire(descriptor->to_wire())->to_wire();
    CHECK(again == wire);
    CHECK(jl::descriptor_wire_equal(*descriptor,
                                    *jl::PipelineDescriptor::from_wire(wire)));
    CHECK(descriptor->descriptor_version == 1);
    CHECK(descriptor->identity.request_id == "req");
    CHECK(descriptor->identity.rank == 1);
    CHECK(descriptor->identity.segment_id == "pp1/726571/decode/1/2/1/0");
    CHECK(descriptor->identity.buffer_domain_id == "pb1/726571/1/9");
    CHECK(descriptor->stream_kind == "decode_remote_credit");
}

static void test_from_wire_rejects_non_object() {
    Json wire = Json::array();
    expect_reject(wire, "must be an object");
}

static void test_from_wire_rejects_unknown_and_missing_top_keys() {
    Json wire = make_wire();
    wire["future_section"] = 1;
    expect_reject(wire, "keys invalid");
    Json missing = make_wire();
    missing.erase("anchors");
    expect_reject(missing, "keys invalid");
}

// ------------------------------------------------------------ identity --

static void test_identity_hex_encoding_non_ascii() {
    Json wire = make_wire();
    wire["identity"]["request_id"] = "r\xc3\xa9q";  // "réq" UTF-8
    wire["identity"]["segment_id"] = "pp1/72c3a971/decode/1/2/1/0";
    wire["identity"]["buffer_domain_id"] = "pb1/72c3a971/1/9";
    expect_accept(wire);
}

static void test_identity_segment_tamper_rejected() {
    Json wire = make_wire();
    wire["identity"]["segment_id"] = "pp1/726571/decode/9/2/1/0";
    expect_reject(wire, "identity IDs do not match canonical encoding");
}

static void test_identity_allocation_mismatch_rejected() {
    Json wire = make_wire();
    wire["identity"]["staging_allocation_id"] = 8;
    expect_reject(wire, "identity IDs do not match canonical encoding");
}

static void test_identity_bad_stage_and_rank_rejected() {
    Json wire = make_wire();
    wire["identity"]["stage"] = "invert";
    expect_reject(wire, "identity.stage");
    Json rank_wire = make_wire();
    rank_wire["identity"]["rank"] = 2147483648ULL;  // I32_MAX + 1
    expect_reject(rank_wire, "identity.rank");
}

static void test_prefill_stage_and_stream_kind_accepted() {
    Json wire = make_wire();
    wire["identity"]["stage"] = "prefill";
    wire["identity"]["segment_id"] = "pp1/726571/prefill/1/2/1/0";
    wire["stream_kind"] = "prefill_read_window";
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(wire);
    CHECK(descriptor->stream_kind == "prefill_read_window");
    CHECK(descriptor->to_wire() == wire);
}

static void test_stream_kind_stage_mismatch_rejected() {
    Json wire = make_wire();
    wire["stream_kind"] = "prefill_read_window";
    expect_reject(wire, "stream_kind does not match identity.stage");
    Json unknown = make_wire();
    unknown["stream_kind"] = "decode_remote_credit_x";
    expect_reject(unknown, "stream_kind");
}

// -------------------------------------------------------------- groups --

static void test_groups_strictly_increasing_rejected() {
    Json wire = make_wire();
    Json duplicate = wire["groups"][0];
    wire["groups"].push_back(std::move(duplicate));  // same ordinal 0 twice
    expect_reject(wire, "strictly increasing group_ordinal");
}

static void test_tile_axis_count_tail_identity_rejected() {
    Json wire = make_wire();
    wire["groups"][0]["tile_axis"]["tile_count"] = 4;
    expect_reject(wire, "inconsistent count/tail");
    Json tail = make_wire();
    tail["groups"][0]["tile_axis"]["tail_units"] = 2;
    expect_reject(tail, "inconsistent count/tail");
}

static void test_tile_axis_derived_fields_parsed() {
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(make_wire());
    const jl::PipelineTileAxis& axis = descriptor->groups[0].tile_axis;
    CHECK(axis.total_units == 5);
    CHECK(axis.units_per_tile == 2);
    CHECK(axis.tile_count == 3);   // ceil(5 / 2)
    CHECK(axis.tail_units == 1);   // 5 - (3 - 1) * 2
    CHECK(axis.global_tile_first == 1);
    CHECK(axis.unit == "stream_bytes");
}

static void test_tile_axis_type_lattice_rejected() {
    Json as_float = make_wire();
    as_float["groups"][0]["tile_axis"]["tile_count"] = 3.0;
    expect_reject(as_float, "tile_count");
    Json as_bool = make_wire();
    as_bool["groups"][0]["tile_axis"]["tile_count"] = true;
    expect_reject(as_bool, "tile_count");
    Json negative = make_wire();
    negative["groups"][0]["layer_start"] = -1;
    expect_reject(negative, "layer_start");
}

static void test_shard_stream_rules_rejected() {
    Json bpu = make_wire();
    bpu["groups"][0]["shards"][0]["bytes_per_unit"] = 2;
    expect_reject(bpu, "invalid stream_bytes extent/bytes_per_unit");
    Json tokens = make_wire();
    tokens["groups"][0]["tile_axis"]["unit"] = "source_tokens";
    tokens["groups"][0]["shards"][0]["total_data_bytes"] = 4;  // 5 * 1 != 4
    expect_reject(tokens, "total_data_bytes must equal total_units*bytes_per_unit");
}

static void test_shard_noc_path_endpoints_rejected() {
    Json wire = make_wire();
    wire["groups"][0]["shards"][0]["noc_path"] = Json::array({1, 1});
    expect_reject(wire, "noc_path must begin/end at shard ranks");
}

static void test_group_layer_range_rejected() {
    Json wire = make_wire();
    wire["groups"][0]["layer_end"] = 0;
    expect_reject(wire, "layer_end");
    Json empty_range = make_wire();
    empty_range["groups"][0]["layer_start"] = 1;  // layer_end == layer_start
    expect_reject(empty_range, "layer range must be nonempty");
}

// ------------------------------------------------------------ protocol --

static void test_protocol_frozen_contract_rejected() {
    Json slots = make_wire();
    slots["protocol"]["buffer_slots"] = 3;
    expect_reject(slots, "frozen two-slot lag-2 contract");
    Json charge = make_wire();
    charge["protocol"]["hbm_charge"] = false;
    expect_reject(charge, "frozen two-slot lag-2 contract");
    Json branching = make_wire();
    branching["protocol"]["tile_branching"] = "something_else";
    expect_reject(branching, "frozen two-slot lag-2 contract");
    Json ack = make_wire();
    ack["protocol"]["fill_ack_bytes"] = 0;
    expect_reject(ack, "must be positive");
}

// ------------------------------------------------- prediction/execution --

static void test_prediction_profile_validation() {
    Json negative = make_wire();
    negative["prediction_profile"]["decode_consume"]["compute_ns_total"] = -1.0;
    expect_reject(negative, "compute_ns_total");
    Json as_bool = make_wire();
    as_bool["prediction_profile"]["decode_consume"]["compute_ns_total"] = true;
    expect_reject(as_bool, "compute_ns_total");
    Json prefill = make_wire();
    prefill["prediction_profile"]["prefill_consume"] = "sometimes";
    expect_reject(prefill, "prefill_consume is fixed");
    Json share = make_wire();
    share["prediction_profile"]["decode_consume"]["share_rule"] = "unknown";
    expect_reject(share, "share_rule");
    Json empty_id = make_wire();
    empty_id["prediction_profile"]["decode_consume"]["profile_id"] = "";
    expect_reject(empty_id, "profile_id");
}

static void test_prediction_profile_int_compute_accepted() {
    // Python stores float(compute); 240 and 240.0 compare equal as dict values.
    Json wire = make_wire();
    wire["prediction_profile"]["decode_consume"]["compute_ns_total"] = 240;
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(wire);
    CHECK(descriptor->to_wire() == make_wire());
    CHECK(jl::descriptor_wire_equal(*descriptor,
                                    *jl::PipelineDescriptor::from_wire(make_wire())));
}

static void test_execution_profile_frozen_rejected() {
    Json wire = make_wire();
    wire["execution_profile"]["data_legs"] = "fancy_legs";
    expect_reject(wire, "must preserve original COMM/HBM/body semantics");
}

// ------------------------------------------------------- logical id ranges --

static void test_id_ranges_overlap_rejected() {
    Json wire = make_wire();
    wire["logical_id_ranges"] = Json::array({
        Json{{"rank", 0}, {"first_json_id", 10}, {"count", 3}, {"stride", 3},
             {"kind", "virtual_endpoint_hole"}},
        Json{{"rank", 0}, {"first_json_id", 14}, {"count", 2}, {"stride", 2},
             {"kind", "virtual_endpoint_hole"}},
    });
    expect_reject(wire, "arithmetic progressions overlap");
}

static void test_id_ranges_interleaved_disjoint_accepted() {
    // Port of test_hole_progressions_may_overlap_as_intervals_when_ids_do_not:
    // intervals [10,19] and [12,21] overlap but the IDs never do.
    Json wire = make_wire();
    wire["logical_id_ranges"].push_back(Json{
        {"rank", 2}, {"first_json_id", 10}, {"count", 3}, {"stride", 3},
        {"kind", "virtual_endpoint_hole"}});
    wire["logical_id_ranges"].push_back(Json{
        {"rank", 2}, {"first_json_id", 12}, {"count", 3}, {"stride", 3},
        {"kind", "virtual_endpoint_hole"}});
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(wire);
    CHECK(descriptor->to_wire() == wire);
}

static void test_id_ranges_order_rejected() {
    Json rank_order = make_wire();
    rank_order["logical_id_ranges"] = Json::array({
        Json{{"rank", 1}, {"first_json_id", 56}, {"count", 2}, {"stride", 2},
             {"kind", "virtual_endpoint_hole"}},
        Json{{"rank", 0}, {"first_json_id", 53}, {"count", 2}, {"stride", 2},
             {"kind", "virtual_endpoint_hole"}},
    });
    expect_reject(rank_order, "rank ordered");
    Json first_order = make_wire();
    first_order["logical_id_ranges"] = Json::array({
        Json{{"rank", 0}, {"first_json_id", 56}, {"count", 2}, {"stride", 2},
             {"kind", "virtual_endpoint_hole"}},
        Json{{"rank", 0}, {"first_json_id", 53}, {"count", 2}, {"stride", 2},
             {"kind", "virtual_endpoint_hole"}},
    });
    expect_reject(first_order, "ordered by first_json_id per rank");
}

static void test_id_range_terminal_overflow_rejected() {
    Json wire = make_wire();
    wire["logical_id_ranges"] = Json::array({
        Json{{"rank", 0}, {"first_json_id", 18446744073709551614ULL},
             {"count", 2}, {"stride", 2}, {"kind", "virtual_endpoint_hole"}},
    });
    expect_reject(wire, "terminal ID exceeds uint64");
}

// ------------------------------------------------------------ tag runs --

static void test_tag_run_rules_rejected() {
    Json group = make_wire();
    group["tag_allocation_runs"][0]["group_ordinal"] = 7;
    expect_reject(group, "tag run references unknown group");
    Json tiles = make_wire();
    tiles["tag_allocation_runs"][0]["tile_count"] = 4;
    expect_reject(tiles, "tag run tile range exceeds group");
    Json emission = make_wire();
    Json later = make_wire()["tag_allocation_runs"][0];
    later["emission_ordinal"] = 6;  // out of emission order for the scope
    emission["tag_allocation_runs"].push_back(std::move(later));
    expect_reject(emission, "must preserve allocator emission/tag order");
}

// -------------------------------------------------------- endpoint runs --

static void test_endpoint_run_stride_rules_rejected() {
    Json shard_stride = make_wire();
    shard_stride["endpoint_runs"][0]["legs"][0]["json_offset"] = 1;
    expect_reject(shard_stride, "shard_stride overlaps leg IDs");
    // Grow the second shard to a full-extent same-source shard so the only
    // violated rule is the tile stride: within-tile extent = 1*2 + 0 = 2.
    Json tile_stride = make_two_shard_wire();
    tile_stride["groups"][0]["shards"][1]["total_data_bytes"] = 5;
    tile_stride["groups"][0]["shards"][1]["source_rank"] = 0;
    tile_stride["groups"][0]["shards"][1]["noc_path"] = Json::array({0, 1});
    tile_stride["consume_completion_runs"][1]["tile_count"] = 3;
    tile_stride["endpoint_runs"][0]["shard_ordinals"] = Json::array({0, 1});
    tile_stride["endpoint_runs"][0]["shard_stride"] = 2;
    tile_stride["endpoint_runs"][0]["tile_stride"] = 2;
    expect_reject(tile_stride, "tile_stride overlaps tile IDs");
}

static void test_endpoint_run_rank_route_rejected() {
    Json wire = make_wire();
    wire["endpoint_runs"][0]["rank"] = 1;  // data_send belongs to source rank 0
    expect_reject(wire, "rank disagrees with shard route");
}

static void test_endpoint_run_geometry_rejected() {
    Json tiles = make_wire();
    tiles["endpoint_runs"][0]["tile_count"] = 4;
    expect_reject(tiles, "endpoint run tile range exceeds group");
    Json shards = make_wire();
    shards["endpoint_runs"][0]["shard_ordinals"] = Json::array({1});
    expect_reject(shards, "endpoint run shard is outside group");
    Json group = make_wire();
    group["endpoint_runs"][0]["group_ordinal"] = 3;
    expect_reject(group, "endpoint run references unknown group");
}

static void test_endpoint_run_active_tile_cap_rejected() {
    Json wire = make_two_shard_wire();
    // Shard 1 only has bytes for one tile, so a COMM run spanning tiles 1..3
    // across both shards exceeds its active cap.
    wire["endpoint_runs"][0]["shard_ordinals"] = Json::array({0, 1});
    expect_reject(wire, "active tile cap");
}

static void test_endpoint_run_id_extent_overflow_rejected() {
    Json wire = make_wire();
    wire["endpoint_runs"][0]["first_json_id"] = 18446744073709551615ULL;
    expect_reject(wire, "uint64 overflow in endpoint run ID extent");
}

static void test_body_marker_with_tags_rejected() {
    Json wire = make_wire();
    wire["endpoint_runs"][0]["legs"][0]["endpoint"] = "body_marker";
    expect_reject(wire, "body_marker tag fields must be null");
}

static void test_unknown_endpoint_run_field_rejected() {
    Json wire = make_wire();
    wire["endpoint_runs"][0]["future_tiles"] = 99;
    expect_reject(wire, "keys invalid");
}

static void test_legacy_context_generation_rejected() {
    Json wire = make_wire();
    wire["endpoint_runs"][0]["legacy_context"]["stage"] = "prefill";
    expect_reject(wire, "generation disagrees with stage");
}

static void test_leg_uniqueness_rejected() {
    Json wire = make_wire();
    Json duplicate = wire["endpoint_runs"][0]["legs"][0];
    duplicate["json_offset"] = 1;
    wire["endpoint_runs"][0]["legs"].push_back(std::move(duplicate));
    expect_reject(wire, "endpoint kinds must be unique");
    Json offsets = make_wire();
    Json second = offsets["endpoint_runs"][0]["legs"][0];
    second["endpoint"] = "data_recv";
    second["json_offset"] = 0;  // same offset, different endpoint
    offsets["endpoint_runs"][0]["legs"].push_back(std::move(second));
    expect_reject(offsets, "json_offset values must be unique");
}

// ------------------------------------------------------ dependency runs --

static void test_dependency_run_strictness_rejected() {
    Json wire = make_wire();
    wire["endpoint_dependency_runs"][0]["future_tiles"] = 99;
    expect_reject(wire, "keys invalid");
    Json duplicate = make_wire();
    Json again = duplicate["endpoint_dependency_runs"][0];
    duplicate["endpoint_dependency_runs"].push_back(std::move(again));
    expect_reject(duplicate, "duplicates an earlier run");
}

static void test_dependency_run_needs_matching_endpoint_run_rejected() {
    Json wire = make_wire();
    wire["endpoint_runs"][0]["tile_count"] = 2;  // emits only tiles 1..2
    expect_reject(wire, "no matching emitted endpoint run");
}

static void test_consume_completion_coverage_rejected() {
    Json gap = make_wire();
    gap["consume_completion_runs"][0]["first_tile"] = 2;
    gap["consume_completion_runs"][0]["tile_count"] = 2;
    expect_reject(gap, "gap or overlap");
    Json rank = make_wire();
    rank["consume_completion_runs"][0]["rank"] = 0;
    expect_reject(rank, "must equal the descriptor buffer owner rank");
    Json missing = make_wire();
    missing["consume_completion_runs"] = Json::array();
    expect_reject(missing, "do not cover every active target tile");
}

static void test_projection_only_descriptor_rules() {
    // Port of test_projection_only_descriptor_allows_empty_physical_terminals.
    Json wire = make_wire();
    wire["logical_id_ranges"] = Json::array();
    wire["tag_allocation_runs"] = Json::array();
    wire["endpoint_runs"] = Json::array();
    wire["endpoint_dependency_runs"] = Json::array();
    wire["consume_completion_runs"] = Json::array();
    wire["anchors"] = Json::array();
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(wire);
    CHECK(descriptor->endpoint_runs.empty());
    CHECK(descriptor->consume_completion_runs.empty());
    CHECK(descriptor->to_wire() == wire);

    Json with_completions = wire;
    with_completions["consume_completion_runs"] = make_wire()["consume_completion_runs"];
    expect_reject(with_completions,
                  "projection-only descriptor must not include consume completion");
    Json with_anchors = wire;
    with_anchors["anchors"] = make_wire()["anchors"];
    expect_reject(with_anchors,
                  "projection-only descriptor must not invent execution anchors");
}

// ---------------------------------------------------------- projection --

static void test_projection_chain_rules_rejected() {
    Json noncontiguous = make_wire();
    noncontiguous["projection"]["chains"][0]["fill"]["first_index"] = 1;
    noncontiguous["projection"]["chains"][0]["consume"]["first_index"] = 1;
    expect_reject(noncontiguous, "first chain must start at index zero");
    Json misaligned = make_wire();
    misaligned["projection"]["chains"][0]["fill"]["first_index"] = 1;
    expect_reject(misaligned, "fill and consume index ranges must align");
    Json registration = make_wire();
    registration["projection"]["chains"][0]["consume"]["consume_order_first"] = 2;
    expect_reject(registration, "registration-order ranges overlap");
    Json gate = make_wire();
    gate["projection"]["chains"][0]["fill"]["gate_capacity"] = 3;
    expect_reject(gate, "gate capacity disagrees");
    Json gate_ids = make_wire();
    gate_ids["projection"]["chains"][0]["consume"]["gate_id"] = "work#other";
    expect_reject(gate_ids, "gate IDs must match");
    Json group = make_wire();
    group["projection"]["chains"][0]["group_ordinal"] = 5;
    expect_reject(group, "references unknown group");
    Json tile_count = make_wire();
    tile_count["projection"]["chains"][0]["tile_count"] = 2;
    expect_reject(tile_count, "tile_count must match its group");
}

static void test_projection_share_rules_rejected() {
    Json bytes = make_wire();
    bytes["projection"]["chains"][0]["consume"]["hbm_share"]["tile_bytes"] = 3;
    expect_reject(bytes, "same global byte positions");
    Json range = make_wire();
    range["projection"]["chains"][0]["consume"]["hbm_share"]["share_index_first"] = 1;
    expect_reject(range, "tile range is inconsistent with the chain");
    Json fixed = make_wire();
    fixed["projection"]["chains"][0]["consume"]["kind"] = "fixed";
    expect_reject(fixed, "fixed consume must have empty route");
    Json compute = make_wire();
    compute["projection"]["chains"][0]["consume"]["compute_share"] = nullptr;
    expect_reject(compute,
                  "compute consume requires separate compute_share and hbm_share");
}

static void test_projection_completion_order_rules_rejected() {
    Json wire = make_wire();
    wire["projection"]["chains"][0]["fill"]["completion_order_first"] = nullptr;
    wire["projection"]["chains"][0]["fill"]["completion_order_stride"] = 1;
    expect_reject(wire, "null completion_order_first requires zero stride");
    Json overflow = make_wire();
    overflow["projection"]["chains"][0]["fill"]["completion_order_first"] =
        9223372036854775807LL;
    overflow["projection"]["chains"][0]["fill"]["completion_order_stride"] =
        9223372036854775807LL;
    expect_reject(overflow, "overflows int64");
}

// ------------------------------------------------------------- anchors --

static void test_anchor_rules_rejected() {
    Json missing_role = make_wire();
    missing_role["anchors"].erase(3);  // drop settle
    expect_reject(missing_role, "exactly one settle anchor");
    Json duplicate_selector = make_wire();
    Json duplicate = make_wire()["anchors"][3];
    duplicate_selector["anchors"].push_back(std::move(duplicate));
    expect_reject(duplicate_selector, "selectors must be unique per rank");
    Json cursor_node = make_wire();
    cursor_node["anchors"][3]["node_id"] = 5;
    expect_reject(cursor_node, "cursor selector must not carry node_id");
    Json node_mismatch = make_wire();
    node_mismatch["anchors"][0]["node_id"] = 52;
    expect_reject(node_mismatch, "node_id disagrees with endpoint ID mapping");
    Json service_kind = make_wire();
    service_kind["anchors"][2]["selector"]["kind"] = "all_obligations_settled";
    expect_reject(service_kind, "must select existing_service_gate");
    Json physical_role = make_wire();
    physical_role["anchors"][0]["role"] = "service";
    expect_reject(physical_role, "only start/frontier anchors");
    Json zero_tile = make_wire();
    zero_tile["anchors"][0]["selector"]["tile_ordinal"] = 0;
    expect_reject(zero_tile, "complete one-based selector");
}

// --------------------------------------------------- endpoint id targets --

static void test_endpoint_id_targets_rejected() {
    Json hole_missing = make_wire();
    hole_missing["logical_id_ranges"] = Json::array({
        Json{{"rank", 1}, {"first_json_id", 56}, {"count", 2}, {"stride", 2},
             {"kind", "virtual_endpoint_hole"}},
    });
    expect_reject(hole_missing, "maps an ID neither to a virtual hole nor");
    Json anchor_missing = make_wire();
    anchor_missing["anchors"].erase(0);  // drop the rank-0 start anchor
    expect_reject(anchor_missing, "maps an ID neither to a virtual hole nor");
    Json alias = make_wire();
    alias["logical_id_ranges"][0]["first_json_id"] = 51;
    expect_reject(alias, "aliases a virtual ID hole");
}

// ------------------------------------------------- endpoint id algebra --

static void test_endpoint_id_algebra() {
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(make_wire());
    CHECK(descriptor->endpoint_id(0, 1, 0, 0) == 51);
    CHECK(descriptor->endpoint_id(0, 2, 0, 0) == 53);
    CHECK(descriptor->endpoint_id(1, 1, 0, 0) == 54);
    CHECK(descriptor->endpoint_id(1, 3, 0, 0) == 58);
}

static void test_endpoint_id_rejections() {
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(make_wire());
    ++g_checks;
    try {
        descriptor->endpoint_id(2, 1, 0, 0);
        ++g_failures;
        std::printf("FAIL endpoint_id run out of range accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("outside descriptor")
              != std::string::npos);
    }
    ++g_checks;
    try {
        descriptor->endpoint_id(0, 4, 0, 0);
        ++g_failures;
        std::printf("FAIL endpoint_id tile out of range accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("tile is outside")
              != std::string::npos);
    }
    ++g_checks;
    try {
        descriptor->endpoint_id(0, 1, 5, 0);
        ++g_failures;
        std::printf("FAIL endpoint_id shard out of range accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("shard is outside")
              != std::string::npos);
    }
    ++g_checks;
    try {
        descriptor->endpoint_id(0, 1, 0, 1);
        ++g_failures;
        std::printf("FAIL endpoint_id leg out of range accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("leg is outside")
              != std::string::npos);
    }
}

// ------------------------------------------------- selected shard tile --

static void test_selected_shard_tile_standard() {
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(make_wire());
    auto first = descriptor->selected_shard_tile(0, 0, 0);
    CHECK(first.first == 1);
    CHECK(first.second.data_bytes == 2);
    CHECK(first.second.emit_endpoint);
    CHECK(first.second.source_rank == 0 && first.second.target_rank == 1);
    auto middle = descriptor->selected_shard_tile(0, 1, 0);
    CHECK(middle.first == 2);
    CHECK(middle.second.data_bytes == 2);
    auto tail = descriptor->selected_shard_tile(0, 2, 0);
    CHECK(tail.first == 3);
    CHECK(tail.second.data_bytes == 1);  // tail units = 1, cap 5 - offset 4
}

static void test_selected_shard_tile_matches_tails_both_units() {
    // Port of test_selected_shard_tile_matches_full_tiles_across_groups_and_tails.
    for (const char* unit : {"stream_bytes", "source_tokens"}) {
        jl::DescriptorPtr descriptor =
            jl::PipelineDescriptor::from_wire(make_shard_count_wire(3, unit));
        const jl::PipelineTileAxis& axis = descriptor->groups[0].tile_axis;
        const std::uint64_t units_per_tile = axis.units_per_tile;
        const std::uint64_t tail_units = axis.tail_units;
        for (std::uint64_t local = 0; local < axis.tile_count; ++local) {
            const std::uint64_t units =
                local + 1 == axis.tile_count ? tail_units : units_per_tile;
            for (std::uint64_t shard = 0; shard < 3; ++shard) {
                auto selected = descriptor->selected_shard_tile(0, local, shard);
                CHECK(selected.first == axis.global_tile_first + local);
                const std::uint64_t expected_bytes =
                    std::string(unit) == "source_tokens"
                        ? units * (shard + 1)
                        : (std::min<std::uint64_t>(
                               5, local * units_per_tile + units)
                           - local * units_per_tile);
                CHECK(selected.second.data_bytes == expected_bytes);
                CHECK(selected.second.emit_endpoint == (expected_bytes > 0));
                CHECK(selected.second.shard_ordinal == shard);
            }
        }
        // Python: expected_tail_bytes 1 (stream_bytes) / 3 (source_tokens).
        auto tail = descriptor->selected_shard_tile(0, 2, 2);
        CHECK(tail.second.data_bytes ==
              (std::string(unit) == "stream_bytes" ? 1 : 3));
    }
}

static void test_selected_shard_tile_second_group() {
    jl::DescriptorPtr base =
        jl::PipelineDescriptor::from_wire(make_shard_count_wire(3, "stream_bytes"));
    // Port of the dataclasses.replace() multi-group construction.
    jl::PipelineDescriptor multi = *base;
    jl::PipelineGroup second = base->groups[0];
    second.group_ordinal = 9;
    second.tile_axis.global_tile_first = 101;
    multi.groups.push_back(second);
    for (std::uint64_t local = 0; local < second.tile_axis.tile_count; ++local) {
        auto selected = multi.selected_shard_tile(9, local, 2);
        CHECK(selected.first == 101 + local);
    }
    CHECK(multi.selected_shard_tile(0, 2, 2).first == 3);
}

static void test_selected_shard_tile_max_zero_semantics() {
    // A short stream shard past its extent produces zero bytes (max(0, ...)).
    jl::DescriptorPtr descriptor =
        jl::PipelineDescriptor::from_wire(make_two_shard_wire());
    auto active = descriptor->selected_shard_tile(0, 1, 0);
    CHECK(active.second.data_bytes == 2);
    auto exhausted = descriptor->selected_shard_tile(0, 1, 1);
    CHECK(exhausted.second.data_bytes == 0);
    CHECK(!exhausted.second.emit_endpoint);
    auto tail_zero = descriptor->selected_shard_tile(0, 2, 1);
    CHECK(tail_zero.second.data_bytes == 0);
    CHECK(!tail_zero.second.emit_endpoint);
    auto tail = descriptor->selected_shard_tile(0, 2, 0);
    CHECK(tail.second.data_bytes == 1);
}

static void test_selected_shard_tile_rejections() {
    jl::DescriptorPtr descriptor =
        jl::PipelineDescriptor::from_wire(make_shard_count_wire(2, "stream_bytes"));
    ++g_checks;
    try {
        descriptor->selected_shard_tile(99, 0, 0);
        ++g_failures;
        std::printf("FAIL selected_shard_tile unknown group accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("unknown group")
              != std::string::npos);
    }
    ++g_checks;
    try {
        descriptor->selected_shard_tile(0, 3, 0);
        ++g_failures;
        std::printf("FAIL selected_shard_tile tile out of range accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("outside group")
              != std::string::npos);
    }
    ++g_checks;
    try {
        descriptor->selected_shard_tile(0, 0, 2);
        ++g_failures;
        std::printf("FAIL selected_shard_tile shard out of range accepted\n");
    } catch (const jl::LedgerError& error) {
        CHECK(std::string(error.what()).find("shard index")
              != std::string::npos);
    }
}

// ------------------------------------------------------ descriptor equal --

static void test_descriptor_wire_equal() {
    Json wire = make_wire();
    jl::DescriptorPtr a = jl::PipelineDescriptor::from_wire(wire);
    jl::DescriptorPtr b = jl::PipelineDescriptor::from_wire(wire);
    CHECK(jl::descriptor_wire_equal(*a, *b));
    Json mutated = make_wire();
    mutated["tag_allocation_runs"][0]["first_tag"] = 202;
    jl::DescriptorPtr c = jl::PipelineDescriptor::from_wire(mutated);
    CHECK(!jl::descriptor_wire_equal(*a, *c));
}

static void test_group_by_ordinal_query() {
    jl::DescriptorPtr descriptor = jl::PipelineDescriptor::from_wire(make_wire());
    const jl::PipelineGroup* group = descriptor->group_by_ordinal(0);
    CHECK(group != nullptr);
    CHECK(group->source_key == "source-0");
    CHECK(group->layer_start == 0 && group->layer_end == 1);
    CHECK(group->shards.size() == 1);
    CHECK(group->shards[0].noc_path == std::vector<std::uint64_t>({0, 1}));
    CHECK(descriptor->group_by_ordinal(42) == nullptr);
}

int main() {
    test_round_trip_matches_input_wire();
    test_from_wire_rejects_non_object();
    test_from_wire_rejects_unknown_and_missing_top_keys();
    test_identity_hex_encoding_non_ascii();
    test_identity_segment_tamper_rejected();
    test_identity_allocation_mismatch_rejected();
    test_identity_bad_stage_and_rank_rejected();
    test_prefill_stage_and_stream_kind_accepted();
    test_stream_kind_stage_mismatch_rejected();
    test_groups_strictly_increasing_rejected();
    test_tile_axis_count_tail_identity_rejected();
    test_tile_axis_derived_fields_parsed();
    test_tile_axis_type_lattice_rejected();
    test_shard_stream_rules_rejected();
    test_shard_noc_path_endpoints_rejected();
    test_group_layer_range_rejected();
    test_protocol_frozen_contract_rejected();
    test_prediction_profile_validation();
    test_prediction_profile_int_compute_accepted();
    test_execution_profile_frozen_rejected();
    test_id_ranges_overlap_rejected();
    test_id_ranges_interleaved_disjoint_accepted();
    test_id_ranges_order_rejected();
    test_id_range_terminal_overflow_rejected();
    test_tag_run_rules_rejected();
    test_endpoint_run_stride_rules_rejected();
    test_endpoint_run_rank_route_rejected();
    test_endpoint_run_geometry_rejected();
    test_endpoint_run_active_tile_cap_rejected();
    test_endpoint_run_id_extent_overflow_rejected();
    test_body_marker_with_tags_rejected();
    test_unknown_endpoint_run_field_rejected();
    test_legacy_context_generation_rejected();
    test_leg_uniqueness_rejected();
    test_dependency_run_strictness_rejected();
    test_dependency_run_needs_matching_endpoint_run_rejected();
    test_consume_completion_coverage_rejected();
    test_projection_only_descriptor_rules();
    test_projection_chain_rules_rejected();
    test_projection_share_rules_rejected();
    test_projection_completion_order_rules_rejected();
    test_anchor_rules_rejected();
    test_endpoint_id_targets_rejected();
    test_endpoint_id_algebra();
    test_endpoint_id_rejections();
    test_selected_shard_tile_standard();
    test_selected_shard_tile_matches_tails_both_units();
    test_selected_shard_tile_second_group();
    test_selected_shard_tile_max_zero_semantics();
    test_selected_shard_tile_rejections();
    test_descriptor_wire_equal();
    test_group_by_ordinal_query();

    if (g_failures != 0) {
        std::printf("pipeline_descriptor_port_test FAILED (%d/%d checks)\n",
                    g_failures, g_checks);
        return 1;
    }
    std::printf("pipeline_descriptor_port_test PASS (%d checks)\n", g_checks);
    return 0;
}
