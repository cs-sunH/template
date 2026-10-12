#include "astra-sim/workload/execution_driven/PipelineKernel.hh"
#include "pipeline_descriptor_test_support.hh"

#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>

using AstraSim::ExecutionDriven::PipelineDescriptor;
using AstraSim::ExecutionDriven::PipelineKernel;
using AstraSim::ExecutionDriven::PipelineKernelContract;
using AstraSim::ExecutionDriven::PipelineSelectorKind;
using AstraSim::ExecutionDriven::PipelineEndpointLeg;
using AstraSim::ExecutionDriven::TestSupport::descriptor_wire;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (!condition) {
        std::cerr << "[pipeline_descriptor_test] FAIL: " << message << '\n';
        ++failures;
    }
}

template <typename Fn>
void expect_throw(Fn&& fn, const std::string& message) {
    try {
        fn();
        expect(false, message + " (did not throw)");
    } catch (const std::exception&) {
    }
}

nlohmann::json contract_wire(uint32_t tile_limit = 1) {
    return nlohmann::json{{"projection_abi_version", 1},
        {"descriptor_version", 1}, {"pipeline_protocol_version", 1},
        {"model_version", PipelineKernelContract::kModelVersion},
        {"timer_whitelist_version", PipelineKernelContract::kTimerWhitelistVersion},
        {"window_tile_limit", tile_limit}, {"window_local_error_ppm", 10000},
        {"window_refine_margin_ppm", 20000},
        {"prediction_config_sha256", std::string(64, 'a')}};
}

void test_valid_descriptor_and_algebra() {
    const auto wire = descriptor_wire();
    const auto descriptor = PipelineDescriptor::parse(wire);
    expect(descriptor->identity().request_id == "req-A", "identity parsed");
    expect(descriptor->groups().size() == 1, "one compact group retained");
    const auto t0 = PipelineKernel::tile(*descriptor, 0, 0);
    const auto t1 = PipelineKernel::tile(*descriptor, 0, 1);
    const auto t2 = PipelineKernel::tile(*descriptor, 0, 2);
    expect(t0.global_tile_ordinal == 1 && t0.units == 2 &&
               t0.data_bytes_total == 128 && t0.shards[0].emit_endpoint,
           "first tile algebra");
    expect(t1.global_tile_ordinal == 2 && t1.units == 2 &&
               t1.data_bytes_total == 128,
           "middle tile algebra");
    expect(t2.global_tile_ordinal == 3 && t2.units == 1 &&
               t2.data_bytes_total == 64,
           "tail tile algebra");
    expect(PipelineKernel::total_data_bytes(*descriptor, 0) == 320,
           "total is shard sum");
    expect(PipelineKernel::prefix_data_bytes(*descriptor, 0, 2) == 256,
           "prefix is exact without tile materialization");
    expect(PipelineKernel::prefix_data_bytes(*descriptor, 0, 0) == 0,
           "empty prefix");
    expect(descriptor->endpoint_runs().size() == 2,
           "typed endpoint affine runs retained");
    expect(descriptor->endpoint_dependency_runs().size() == 1 &&
               descriptor->consume_completion_runs().size() == 1,
           "typed endpoint parent/completion affine rules retained");
    expect(descriptor->projection().version == 1 &&
               descriptor->projection().chains.size() == 1 &&
               descriptor->projection().chains[0].fill.bytes_per_tile == 64,
           "typed compact projection chain retained");
    const auto& source_run = descriptor->endpoint_runs()[0];
    const auto& source_leg = source_run.legs[0];
    expect(descriptor->endpoint_json_id(source_run, source_leg, 3, 0) == 53 &&
               descriptor->endpoint_tag(source_run, source_leg, 3, 0) == 103,
           "endpoint ID/tag affine algebra");
    expect(descriptor->endpoint_runs()[1].has_initial_parent_json_id &&
               descriptor->endpoint_runs()[1].initial_parent_json_id == 0,
           "zero-valued external JSON parent ID is retained");

    auto interleaved = descriptor_wire();
    std::swap(interleaved["endpoint_runs"][0], interleaved["endpoint_runs"][1]);
    const auto interleaved_descriptor = PipelineDescriptor::parse(interleaved);
    expect(interleaved_descriptor->endpoint_runs()[0].rank == 1 &&
               interleaved_descriptor->endpoint_runs()[1].rank == 0,
           "endpoint-run array preserves physical emitter order");
}

void test_stream_bytes_union() {
    auto wire = descriptor_wire();
    auto& group = wire["groups"][0];
    group["tile_axis"]["unit"] = "stream_bytes";
    group["tile_axis"]["total_units"] = 5;
    group["tile_axis"]["units_per_tile"] = 2;
    group["tile_axis"]["tile_count"] = 3;
    group["tile_axis"]["tail_units"] = 1;
    group["shards"][0]["total_data_bytes"] = 5;
    group["shards"][0]["bytes_per_unit"] = 1;
    group["shards"].push_back({{"shard_ordinal", 1}, {"source_rank", 0},
        {"target_rank", 2}, {"total_data_bytes", 3}, {"bytes_per_unit", 1},
        {"noc_path", nlohmann::json::array({0, 2})}});
    const auto descriptor = PipelineDescriptor::parse(wire);
    const auto t0 = PipelineKernel::tile(*descriptor, 0, 0);
    const auto t1 = PipelineKernel::tile(*descriptor, 0, 1);
    const auto t2 = PipelineKernel::tile(*descriptor, 0, 2);
    expect(t0.data_bytes_total == 4 && t0.shards[0].data_bytes == 2 &&
               t0.shards[1].data_bytes == 2,
           "stream-byte first union tile");
    expect(t1.data_bytes_total == 3 && t1.shards[1].data_bytes == 1,
           "short shard tail is preserved");
    expect(t2.data_bytes_total == 1 && !t2.shards[1].emit_endpoint,
           "empty shard tail does not create endpoint");
    expect(PipelineKernel::prefix_data_bytes(*descriptor, 0, 2) == 7,
           "stream prefix sums per-shard prefixes");
}

void test_uneven_stream_shard_active_ranges() {
    auto wire = descriptor_wire();
    auto& group = wire["groups"][0];
    group["tile_axis"]["unit"] = "stream_bytes";
    group["tile_axis"]["total_units"] = 5;
    group["tile_axis"]["units_per_tile"] = 2;
    group["tile_axis"]["tile_count"] = 3;
    group["tile_axis"]["tail_units"] = 1;
    group["shards"][0]["total_data_bytes"] = 5;
    group["shards"][0]["bytes_per_unit"] = 1;
    group["shards"].push_back({
        {"shard_ordinal", 1}, {"source_rank", 0}, {"target_rank", 1},
        {"total_data_bytes", 3}, {"bytes_per_unit", 1},
        {"noc_path", nlohmann::json::array({0, 1})}});
    group["shards"].push_back({
        {"shard_ordinal", 2}, {"source_rank", 0}, {"target_rank", 1},
        {"total_data_bytes", 0}, {"bytes_per_unit", 1},
        {"noc_path", nlohmann::json::array({0, 1})}});
    wire["consume_completion_runs"].push_back({
        {"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
        {"tile_count", 2}, {"shard_ordinals", nlohmann::json::array({1})},
        {"dependency_kind", "ordinary_node_terminal"},
        {"first_dependency_json_id", 80}, {"dependency_tile_stride", 1},
        {"dependency_shard_stride", 0}});

    const auto descriptor = PipelineDescriptor::parse(wire);
    const auto short_tail = PipelineKernel::tile(*descriptor, 0, 1);
    const auto past_short_tail = PipelineKernel::tile(*descriptor, 0, 2);
    expect(short_tail.shards[1].data_bytes == 1 &&
               short_tail.shards[1].emit_endpoint,
           "uneven byte shard keeps its one-byte tail endpoint");
    expect(past_short_tail.shards[1].data_bytes == 0 &&
               !past_short_tail.shards[1].emit_endpoint &&
               past_short_tail.shards[2].data_bytes == 0,
           "short and empty shards have no endpoint past their active count");

    auto inactive_comm = wire;
    inactive_comm["endpoint_runs"][1]["shard_ordinals"] =
        nlohmann::json::array({1});
    expect_throw([&] { (void)PipelineDescriptor::parse(inactive_comm); },
                 "COMM run beyond short shard active count rejected");

    auto inactive_consume = wire;
    inactive_consume["consume_completion_runs"][1]["tile_count"] = 3;
    expect_throw([&] { (void)PipelineDescriptor::parse(inactive_consume); },
                 "consume completion beyond short shard active count rejected");

    auto consume_gap = wire;
    consume_gap["consume_completion_runs"][1]["first_tile"] = 2;
    consume_gap["consume_completion_runs"][1]["tile_count"] = 1;
    consume_gap["consume_completion_runs"][1]["first_dependency_json_id"] = 81;
    expect_throw([&] { (void)PipelineDescriptor::parse(consume_gap); },
                 "active shard completion gap rejected");

    auto consume_overlap = wire;
    auto duplicate_completion = consume_overlap["consume_completion_runs"][1];
    duplicate_completion["tile_count"] = 1;
    consume_overlap["consume_completion_runs"].push_back(duplicate_completion);
    expect_throw([&] { (void)PipelineDescriptor::parse(consume_overlap); },
                 "active shard completion overlap rejected");

    auto marker_after_cap = wire;
    marker_after_cap["groups"][0]["shards"][0]["total_data_bytes"] = 3;
    marker_after_cap["groups"][0]["shards"][1]["total_data_bytes"] = 5;
    marker_after_cap["endpoint_runs"][0]["tile_count"] = 2;
    marker_after_cap["endpoint_runs"][1]["tile_count"] = 2;
    marker_after_cap["endpoint_dependency_runs"][0]["tile_count"] = 2;
    marker_after_cap["consume_completion_runs"][0]["tile_count"] = 2;
    marker_after_cap["consume_completion_runs"][1]["tile_count"] = 3;
    marker_after_cap["logical_id_ranges"].push_back({
        {"rank", 1}, {"first_json_id", 90}, {"count", 1}, {"stride", 1},
        {"kind", "virtual_endpoint_hole"}});
    marker_after_cap["endpoint_runs"].push_back({
        {"rank", 1}, {"group_ordinal", 0}, {"first_tile", 3},
        {"tile_count", 1}, {"shard_ordinals", nlohmann::json::array({0})},
        {"first_json_id", 90}, {"tile_stride", 1}, {"shard_stride", 1},
        {"initial_parent_json_id", nullptr},
        {"legacy_context", {{"request_id", "req-A"}, {"stage", "decode"},
                             {"generation", 1}}},
        {"legs", nlohmann::json::array({
            {{"endpoint", "body_marker"}, {"json_offset", 0},
             {"tag_emission_ordinal", nullptr}, {"tag_offset", nullptr},
             {"tag_tile_stride", nullptr}, {"tag_shard_stride", nullptr}}})}});
    expect(PipelineDescriptor::parse(marker_after_cap)->endpoint_runs().size() == 3,
           "real canonical body marker may remain after its shard data cap");
}

void test_rejections() {
    auto wire = descriptor_wire();
    wire["unknown"] = true;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "unknown descriptor field rejected");

    wire = descriptor_wire();
    wire["groups"][0]["tile_axis"]["tile_count"] = 4;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "inconsistent tile count rejected");

    wire = descriptor_wire();
    wire["anchors"][0]["node_id"] = 52;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "physical anchor inside virtual hole rejected");

    wire = descriptor_wire();
    wire["anchors"][0]["rank"] = 1;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "source endpoint anchor rank mismatch rejected");

    wire = descriptor_wire();
    wire["anchors"][0]["selector"]["tag_offset"] = 7;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "tag selector outside allocated run rejected");

    wire = descriptor_wire();
    wire["logical_id_ranges"][0]["count"] = 1;
    wire["logical_id_ranges"][1]["first_json_id"] = 56;
    wire["logical_id_ranges"][1]["count"] = 1;
    wire["anchors"].push_back({
        {"role", "start"}, {"rank", 0}, {"binding", "physical_endpoint_slot"},
        {"node_id", 53}, {"selector", {{"kind", "tile_endpoint_leg"},
            {"group_ordinal", 0}, {"tile_ordinal", 3}, {"shard_ordinal", 0},
            {"endpoint", "data_send"}, {"tag_allocation_emission_ordinal", 7},
            {"tag_offset", 2}}}});
    wire["anchors"].push_back({
        {"role", "frontier"}, {"rank", 1}, {"binding", "physical_endpoint_slot"},
        {"node_id", 55}, {"selector", {{"kind", "tile_endpoint_terminal"},
            {"group_ordinal", 0}, {"tile_ordinal", 2}, {"shard_ordinal", 0},
            {"endpoint", "data_recv"}, {"tag_allocation_emission_ordinal", 7},
            {"tag_offset", 1}}}});
    const auto multi_anchor_descriptor = PipelineDescriptor::parse(wire);
    expect(multi_anchor_descriptor->anchors().size() == 6,
           "multiple physical start/frontier anchors are accepted");

    auto duplicate_slot = wire;
    duplicate_slot["anchors"][4]["node_id"] = 51;
    expect_throw([&] { (void)PipelineDescriptor::parse(duplicate_slot); },
                 "duplicate physical anchor rank/node_id rejected");

    auto duplicate_selector = wire;
    duplicate_selector["anchors"][4]["selector"] = duplicate_selector["anchors"][0]["selector"];
    expect_throw([&] { (void)PipelineDescriptor::parse(duplicate_selector); },
                 "duplicate full anchor selector rejected");

    auto duplicate_service = wire;
    duplicate_service["anchors"].push_back(duplicate_service["anchors"][2]);
    expect_throw([&] { (void)PipelineDescriptor::parse(duplicate_service); },
                 "service anchor remains unique");

    auto physical_service = wire;
    physical_service["anchors"][2]["binding"] = "physical_endpoint_slot";
    physical_service["anchors"][2]["node_id"] = 51;
    expect_throw([&] { (void)PipelineDescriptor::parse(physical_service); },
                 "service anchor remains cursor-only");

    wire = descriptor_wire();
    wire["endpoint_runs"][0]["first_json_id"] = 0;
    wire["anchors"][0]["node_id"] = 0;
    wire["logical_id_ranges"][0]["first_json_id"] = 1;
    const auto zero_id_descriptor = PipelineDescriptor::parse(wire);
    expect(zero_id_descriptor->endpoint_json_id(
               zero_id_descriptor->endpoint_runs()[0],
               zero_id_descriptor->endpoint_runs()[0].legs[0], 1, 0) == 0,
           "zero-valued external endpoint JSON ID is accepted");

    wire = descriptor_wire();
    wire["endpoint_runs"].push_back(wire["endpoint_runs"][0]);
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "overlapping affine endpoint IDs rejected");

    wire = descriptor_wire();
    wire["endpoint_dependency_runs"][0]["endpoint"] = "data_send";
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "endpoint dependency child rank mismatch rejected");

    wire = descriptor_wire();
    wire["endpoint_runs"][1]["initial_parent_json_id"] = 9;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "initial parent must mirror the dependency run");

    wire = descriptor_wire();
    wire.erase("consume_completion_runs");
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "consume completion map is mandatory");

    expect_throw([&] { (void)PipelineKernel::tile(
                         *PipelineDescriptor::parse(descriptor_wire()), 0, 3); },
                 "out-of-range tile rejected");
}

void test_sparse_logical_id_ranges() {
    auto wire = descriptor_wire();
    wire["groups"][0]["tile_axis"]["total_units"] = 9;
    wire["groups"][0]["tile_axis"]["tile_count"] = 5;
    wire["groups"][0]["tile_axis"]["tail_units"] = 1;
    wire["groups"][0]["shards"][0]["total_data_bytes"] = 576;
    wire["projection"]["chains"][0]["tile_count"] = 5;
    wire["projection"]["chains"][0]["consume"]["consume_order_first"] = 5;
    wire["endpoint_runs"][0]["tile_count"] = 5;
    wire["endpoint_runs"][0]["tile_stride"] = 2;
    wire["consume_completion_runs"][0]["tile_count"] = 5;
    wire["tag_allocation_runs"][0]["tile_count"] = 5;
    wire["logical_id_ranges"] = nlohmann::json::array({
        {{"rank", 0}, {"first_json_id", 53}, {"count", 2}, {"stride", 4},
         {"kind", "virtual_endpoint_hole"}},
        {{"rank", 0}, {"first_json_id", 55}, {"count", 2}, {"stride", 4},
         {"kind", "virtual_endpoint_hole"}},
        {{"rank", 1}, {"first_json_id", 55}, {"count", 2}, {"stride", 1},
         {"kind", "virtual_endpoint_hole"}}});
    const auto descriptor = PipelineDescriptor::parse(wire);
    expect(descriptor->logical_id_ranges()[0].stride == 4 &&
               descriptor->logical_id_ranges()[1].stride == 4,
           "sparse logical APs with overlapping bounds but disjoint members are accepted");

    wire["logical_id_ranges"][1]["first_json_id"] = 57;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "intersecting sparse AP holes are rejected");

    wire = descriptor_wire();
    wire["groups"][0]["tile_axis"]["total_units"] = 9;
    wire["groups"][0]["tile_axis"]["tile_count"] = 5;
    wire["groups"][0]["tile_axis"]["tail_units"] = 1;
    wire["groups"][0]["shards"][0]["total_data_bytes"] = 576;
    wire["projection"]["chains"][0]["tile_count"] = 5;
    wire["projection"]["chains"][0]["consume"]["consume_order_first"] = 5;
    wire["endpoint_runs"][0]["tile_count"] = 5;
    wire["endpoint_runs"][0]["tile_stride"] = 2;
    wire["tag_allocation_runs"][0]["tile_count"] = 5;
    wire["logical_id_ranges"][0]["first_json_id"] = 53;
    wire["logical_id_ranges"][0]["count"] = 2;
    wire["logical_id_ranges"][0]["stride"] = 4;
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "endpoint AP coverage rejects an unmapped virtual endpoint ID");
}

void test_projection_only_anchor_shape() {
    auto wire = descriptor_wire();
    wire["logical_id_ranges"] = nlohmann::json::array();
    wire["tag_allocation_runs"] = nlohmann::json::array();
    wire["endpoint_runs"] = nlohmann::json::array();
    wire["endpoint_dependency_runs"] = nlohmann::json::array();
    wire["consume_completion_runs"] = nlohmann::json::array();
    wire["anchors"] = nlohmann::json::array();
    const auto descriptor = PipelineDescriptor::parse(wire);
    expect(descriptor->anchors().empty() &&
               !descriptor->projection().chains.empty(),
           "projection-only descriptor uses no synthetic execution anchors");

    wire["anchors"] = descriptor_wire()["anchors"];
    expect_throw([&] { (void)PipelineDescriptor::parse(wire); },
                 "projection-only descriptor rejects synthetic cursor anchors");
}

void test_internal_marker_frontier_selector() {
    auto wire = descriptor_wire();
    auto& marker_leg = wire["endpoint_runs"][1]["legs"][0];
    marker_leg["endpoint"] = "body_marker";
    marker_leg["tag_emission_ordinal"] = nullptr;
    marker_leg["tag_offset"] = nullptr;
    marker_leg["tag_tile_stride"] = nullptr;
    marker_leg["tag_shard_stride"] = nullptr;
    wire["endpoint_dependency_runs"][0]["endpoint"] = "body_marker";
    wire["anchors"][1]["selector"] = {
        {"kind", "internal_marker_done"},
        {"group_ordinal", 0}, {"tile_ordinal", 1},
        {"shard_ordinal", 0}, {"endpoint", "body_marker"}};

    const auto descriptor = PipelineDescriptor::parse(wire);
    expect(descriptor->anchors()[1].selector.kind ==
               PipelineSelectorKind::InternalMarkerDone &&
               descriptor->endpoint_runs()[1].legs[0].endpoint ==
                   PipelineEndpointLeg::BodyMarker &&
               !descriptor->endpoint_runs()[1].legs[0].has_tag_fields,
           "internal marker frontier resolves to the original untagged marker run");

    auto with_tag = wire;
    with_tag["anchors"][1]["selector"]["tag_offset"] = nullptr;
    expect_throw([&] { (void)PipelineDescriptor::parse(with_tag); },
                 "internal marker selector rejects even null tag fields");

    auto wrong_kind = wire;
    wrong_kind["anchors"][1]["selector"]["kind"] = "tile_endpoint_terminal";
    expect_throw([&] { (void)PipelineDescriptor::parse(wrong_kind); },
                 "body marker cannot use a communication endpoint selector");

    auto wrong_role = wire;
    wrong_role["anchors"][1]["role"] = "start";
    expect_throw([&] { (void)PipelineDescriptor::parse(wrong_role); },
                 "internal marker selector is frontier-only");

    auto existing_gate = wire;
    existing_gate["anchors"][1]["selector"]["existing_gate"] = "gate-x";
    expect_throw([&] { (void)PipelineDescriptor::parse(existing_gate); },
                 "internal marker selector rejects existing_gate");
}

void test_shared_marker_completion_for_target_shards() {
    auto wire = descriptor_wire();
    wire["groups"][0]["shards"].push_back({
        {"shard_ordinal", 1}, {"source_rank", 0}, {"target_rank", 1},
        {"total_data_bytes", 320}, {"bytes_per_unit", 64},
        {"noc_path", nlohmann::json::array({0, 1})}});

    // The COMM receive is emitted per shard; the body marker is emitted once
    // per tile/rank and names the canonical target shard only.
    auto marker_run = wire["endpoint_runs"][1];
    wire["endpoint_runs"][1]["shard_ordinals"] = {0, 1};
    wire["endpoint_runs"][1]["tile_stride"] = 2;
    wire["endpoint_runs"][1]["shard_stride"] = 1;
    wire["endpoint_runs"][1]["legs"][0]["tag_tile_stride"] = 2;
    wire["endpoint_runs"][1]["initial_parent_json_id"] = 0;

    marker_run["first_json_id"] = 60;
    marker_run["tile_stride"] = 1;
    marker_run["shard_stride"] = 1;
    marker_run["shard_ordinals"] = {0};
    marker_run["initial_parent_json_id"] = 54;
    marker_run["legs"] = nlohmann::json::array({nlohmann::json{
        {"endpoint", "body_marker"}, {"json_offset", 0},
        {"tag_emission_ordinal", nullptr}, {"tag_offset", nullptr},
        {"tag_tile_stride", nullptr}, {"tag_shard_stride", nullptr}}});
    wire["endpoint_runs"].push_back(marker_run);

    wire["logical_id_ranges"] = nlohmann::json::array({
        {{"rank", 0}, {"first_json_id", 52}, {"count", 2},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}},
        {{"rank", 1}, {"first_json_id", 55}, {"count", 5},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}},
        {{"rank", 1}, {"first_json_id", 61}, {"count", 2},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}}});
    wire["tag_allocation_runs"][0]["shard_count"] = 2;
    wire["endpoint_dependency_runs"] = nlohmann::json::array({
        {{"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
         {"tile_count", 3}, {"shard_ordinals", {0, 1}},
         {"endpoint", "data_recv"},
         {"dependency_kind", "ordinary_node_terminal"},
         {"first_dependency_json_id", 0},
         {"dependency_tile_stride", 0}, {"dependency_shard_stride", 0}},
        {{"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
         {"tile_count", 3}, {"shard_ordinals", {0}},
         {"endpoint", "body_marker"},
         {"dependency_kind", "virtual_endpoint_terminal"},
         {"first_dependency_json_id", 54},
         {"dependency_tile_stride", 2}, {"dependency_shard_stride", 1}}});
    wire["consume_completion_runs"] = nlohmann::json::array({
        {{"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
         {"tile_count", 3}, {"shard_ordinals", {0, 1}},
         {"dependency_kind", "virtual_endpoint_terminal"},
         {"first_dependency_json_id", 60},
         {"dependency_tile_stride", 1}, {"dependency_shard_stride", 0}}});
    wire["anchors"].push_back({
        {"role", "frontier"}, {"rank", 1},
        {"binding", "physical_endpoint_slot"}, {"node_id", 60},
        {"selector", {{"kind", "internal_marker_done"},
                       {"group_ordinal", 0}, {"tile_ordinal", 1},
                       {"shard_ordinal", 0}, {"endpoint", "body_marker"}}}});

    const auto descriptor = PipelineDescriptor::parse(wire);
    expect(descriptor->endpoint_runs().size() == 3 &&
               descriptor->consume_completion_runs().size() == 1 &&
               descriptor->consume_completion_runs()[0].shard_ordinals.size() == 2,
           "two target shards share one canonical marker completion authority");

    auto bad_alias = wire;
    bad_alias["consume_completion_runs"][0]["dependency_shard_stride"] = 1;
    expect_throw([&] { (void)PipelineDescriptor::parse(bad_alias); },
                 "shared marker completion requires zero shard stride");

    auto noncanonical_marker = wire;
    noncanonical_marker["endpoint_runs"][2]["shard_ordinals"] = {1};
    expect_throw([&] { (void)PipelineDescriptor::parse(noncanonical_marker); },
                 "body marker run rejects a noncanonical target shard");
}

void test_marker_canonical_shard_is_owner_local() {
    auto wire = descriptor_wire();
    wire["identity"]["rank"] = 3;
    wire["identity"]["segment_id"] = "pp1/7265712d41/decode/4/0/3/0";
    wire["identity"]["buffer_domain_id"] = "pb1/7265712d41/3/17";
    wire["groups"][0]["shards"][0]["target_rank"] = 2;
    wire["groups"][0]["shards"][0]["noc_path"] = {0, 2};
    wire["groups"][0]["shards"].push_back({
        {"shard_ordinal", 1}, {"source_rank", 0}, {"target_rank", 3},
        {"total_data_bytes", 320}, {"bytes_per_unit", 64},
        {"noc_path", nlohmann::json::array({0, 3})}});
    wire["projection"]["chains"][0]["rank"] = 3;

    // Keep the existing shard-0 source/target pair on ranks 0 and 2.
    wire["endpoint_runs"][1]["rank"] = 2;
    wire["endpoint_runs"][1]["legacy_context"]["request_id"] = "req-A";
    wire["endpoint_dependency_runs"][0]["rank"] = 2;

    // Add the owner-rank shard (ordinal 1). The source emits before receive;
    // it has a distinct tag allocation and affine endpoint IDs.
    auto source_shard_1 = wire["endpoint_runs"][0];
    source_shard_1["shard_ordinals"] = {1};
    source_shard_1["first_json_id"] = 80;
    source_shard_1["legs"][0]["tag_emission_ordinal"] = 8;
    wire["endpoint_runs"].push_back(source_shard_1);

    auto recv_shard_1 = wire["endpoint_runs"][1];
    recv_shard_1["rank"] = 3;
    recv_shard_1["shard_ordinals"] = {1};
    recv_shard_1["first_json_id"] = 70;
    recv_shard_1["legs"][0]["tag_emission_ordinal"] = 8;
    recv_shard_1["initial_parent_json_id"] = 0;
    wire["endpoint_runs"].push_back(recv_shard_1);

    auto marker_run = recv_shard_1;
    marker_run["first_json_id"] = 60;
    marker_run["initial_parent_json_id"] = 70;
    marker_run["legs"] = nlohmann::json::array({nlohmann::json{
        {"endpoint", "body_marker"}, {"json_offset", 0},
        {"tag_emission_ordinal", nullptr}, {"tag_offset", nullptr},
        {"tag_tile_stride", nullptr}, {"tag_shard_stride", nullptr}}});
    wire["endpoint_runs"].push_back(marker_run);

    wire["logical_id_ranges"] = nlohmann::json::array({
        {{"rank", 0}, {"first_json_id", 52}, {"count", 2},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}},
        {{"rank", 0}, {"first_json_id", 80}, {"count", 3},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}},
        {{"rank", 2}, {"first_json_id", 55}, {"count", 2},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}},
        {{"rank", 3}, {"first_json_id", 61}, {"count", 2},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}},
        {{"rank", 3}, {"first_json_id", 71}, {"count", 2},
         {"stride", 1}, {"kind", "virtual_endpoint_hole"}}});
    wire["tag_allocation_runs"][0]["count"] = 3;
    wire["tag_allocation_runs"].push_back({
        {"allocator_scope_id", "builder-0"}, {"emission_ordinal", 8},
        {"first_tag", 104}, {"count", 3}, {"group_ordinal", 0},
        {"first_tile", 1}, {"tile_count", 3}, {"first_shard", 1},
        {"shard_count", 1}});

    wire["endpoint_dependency_runs"][0]["shard_ordinals"] = {0};
    wire["endpoint_dependency_runs"].push_back({
        {"rank", 3}, {"group_ordinal", 0}, {"first_tile", 1},
        {"tile_count", 3}, {"shard_ordinals", {1}},
        {"endpoint", "data_recv"},
        {"dependency_kind", "ordinary_node_terminal"},
        {"first_dependency_json_id", 0}, {"dependency_tile_stride", 0},
        {"dependency_shard_stride", 0}});
    wire["endpoint_dependency_runs"].push_back({
        {"rank", 3}, {"group_ordinal", 0}, {"first_tile", 1},
        {"tile_count", 3}, {"shard_ordinals", {1}},
        {"endpoint", "body_marker"},
        {"dependency_kind", "ordinary_node_terminal"},
        {"first_dependency_json_id", 70}, {"dependency_tile_stride", 1},
        {"dependency_shard_stride", 0}});
    wire["consume_completion_runs"][0]["rank"] = 3;
    wire["consume_completion_runs"][0]["shard_ordinals"] = {1};
    wire["consume_completion_runs"][0]["first_dependency_json_id"] = 60;

    wire["anchors"][1]["rank"] = 2;
    wire["anchors"][2]["rank"] = 3;
    wire["anchors"][3]["rank"] = 3;
    wire["anchors"].push_back({
        {"role", "frontier"}, {"rank", 3},
        {"binding", "physical_endpoint_slot"}, {"node_id", 70},
        {"selector", {{"kind", "tile_endpoint_terminal"},
                       {"group_ordinal", 0}, {"tile_ordinal", 1},
                       {"shard_ordinal", 1}, {"endpoint", "data_recv"},
                       {"tag_allocation_emission_ordinal", 8}, {"tag_offset", 0}}}});
    wire["anchors"].push_back({
        {"role", "frontier"}, {"rank", 3},
        {"binding", "physical_endpoint_slot"}, {"node_id", 60},
        {"selector", {{"kind", "internal_marker_done"},
                       {"group_ordinal", 0}, {"tile_ordinal", 1},
                       {"shard_ordinal", 1}, {"endpoint", "body_marker"}}}});

    const auto descriptor = PipelineDescriptor::parse(wire);
    expect(descriptor->identity().rank == 3 &&
               descriptor->endpoint_runs().back().shard_ordinals.front() == 1,
           "owner-rank marker may use ordinal 1 when lower target ordinal belongs to another rank");

    auto wrong_owner_shard = wire;
    wrong_owner_shard["endpoint_runs"].back()["shard_ordinals"] = {0};
    expect_throw([&] { (void)PipelineDescriptor::parse(wrong_owner_shard); },
                 "marker cannot select a lower ordinal owned by a different target rank");
}

void test_shared_active_count_vectors(const char* path) {
    std::ifstream input(path);
    expect(static_cast<bool>(input), "shared active-count vector file opens");
    if (!input) return;
    nlohmann::json document;
    try {
        input >> document;
    } catch (const std::exception& error) {
        expect(false, std::string("shared vector JSON parses: ") + error.what());
        return;
    }
    expect(document.is_object() && document.value("schema_version", 0) == 1 &&
               document.contains("vectors") && document["vectors"].is_array(),
           "shared active-count vector schema is valid");
    if (!document.contains("vectors") || !document["vectors"].is_array()) return;
    for (const auto& vector : document["vectors"]) {
        const std::string name = vector.value("name", "<unnamed>");
        const bool should_accept = vector.value("expect", "") == "accept";
        const std::string category = vector.value("reason_category", "");
        try {
            (void)PipelineDescriptor::parse(vector.at("wire"));
            expect(should_accept, "shared vector " + name + " acceptance agrees");
        } catch (const std::exception& error) {
            const std::string message = error.what();
            expect(!should_accept, "shared vector " + name + " rejection agrees");
            if (should_accept) continue;
            if (category == "active_count") {
                expect(message.find("active tile count") != std::string::npos,
                       "shared vector " + name + " rejects for active-count rule");
            } else if (category == "coverage_gap") {
                expect(message.find("gap or overlap") != std::string::npos,
                       "shared vector " + name + " rejects for coverage gap");
            } else if (category == "coverage_overlap") {
                expect(message.find("selector is duplicated") != std::string::npos,
                       "shared vector " + name + " rejects for coverage overlap");
            }
        }
    }
}

void test_kernel_contract() {
    const auto a = PipelineKernelContract::parse(contract_wire());
    const auto b = PipelineKernelContract::parse(contract_wire());
    const auto c = PipelineKernelContract::parse(contract_wire(8));
    expect(*a == *b && *a != *c, "typed run contract equality compares all fields");
    expect(a->window_tile_limit() == 1 &&
               PipelineKernelContract::stable_window_count() == 8 &&
               PipelineKernelContract::absolute_prefix_budget_ticks() == 1,
           "parsed tunables and fixed constants exposed");
    auto bad = contract_wire(3);
    expect_throw([&] { (void)PipelineKernelContract::parse(bad); },
                 "unsupported precision limit rejected");
    bad = contract_wire();
    bad["extra"] = 1;
    expect_throw([&] { (void)PipelineKernelContract::parse(bad); },
                 "unknown kernel contract field rejected");
}

}  // namespace

int main(int argc, char** argv) {
    test_valid_descriptor_and_algebra();
    test_stream_bytes_union();
    test_uneven_stream_shard_active_ranges();
    test_rejections();
    test_sparse_logical_id_ranges();
    test_projection_only_anchor_shape();
    test_internal_marker_frontier_selector();
    test_shared_marker_completion_for_target_shards();
    test_marker_canonical_shard_is_owner_local();
    test_kernel_contract();
    if (argc == 2) test_shared_active_count_vectors(argv[1]);
    else if (argc > 2) expect(false, "expected at most one shared vector path");
    if (failures != 0) {
        std::cerr << "[pipeline_descriptor_test] " << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "[pipeline_descriptor_test] PASS\n";
    return 0;
}
