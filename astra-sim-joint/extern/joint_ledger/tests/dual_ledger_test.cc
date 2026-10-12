// dual_ledger_test.cc -- unit-level port of the pipeline ledger stage-core
// scenarios from joint/test_pipeline_ledger_v2.py (descriptor lifecycle,
// cursor/treap progression, v4 target-slot claim/consume chain, credit
// balance, obligations, request_close/retirement, gates/replay, atomicity).
//
// Most cases drive the full chain: wire envelope JSON -> parse_envelope_v4
// -> PipelineLedgerCore::prepare (coordinated WorkLedger transaction) ->
// commit; a handful construct ParsedEnvelope values directly to exercise
// paths the parser cannot produce.
#include <cstdio>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "joint_ledger/LedgerTypes.hh"
#include "joint_ledger/PipelineDescriptor.hh"
#include "joint_ledger/PipelineRows.hh"
#include "joint_ledger/PipelineStage.hh"
#include "joint_ledger/WorkLedgerCore.hh"
#include "json/json.hpp"

namespace jointledger {
// Defined in PipelineStage.cc (no header declaration: orchestrator/test hook
// binding this stage core to the WorkLedgerCore owning its transactions).
void bind_work_ledger(PipelineLedgerCore& pipeline, WorkLedgerCore& work);
}

using jointledger::DescriptorPtr;
using jointledger::Json;
using jointledger::LedgerError;
using jointledger::ParsedEnvelope;
using jointledger::PipelineDescriptor;
using jointledger::PipelineLedgerCore;
using jointledger::PipelineReceipt;
using jointledger::RetainedRegistry;
using jointledger::WorkLedgerCore;

static int failures = 0;

#define CHECK(cond) do { \
    if (!(cond)) { \
        std::fprintf(stderr, "FAIL %s:%d %s\n", __FILE__, __LINE__, #cond); \
        ++failures; \
    } \
} while (0)

#define CHECK_THROW_STATEMENT(stmt, fragment) do { \
    bool threw_ = false; \
    try { stmt; } catch (const LedgerError& error_) { \
        threw_ = true; \
        const std::string what_ = error_.what(); \
        if (what_.find(fragment) == std::string::npos) { \
            std::fprintf(stderr, \
                "FAIL %s:%d wrong message: '%s' (want '%s')\n", \
                __FILE__, __LINE__, what_.c_str(), fragment); \
            ++failures; \
        } \
    } \
    if (!threw_) { \
        std::fprintf(stderr, "FAIL %s:%d expected LedgerError('%s')\n", \
                     __FILE__, __LINE__, fragment); \
        ++failures; \
    } \
} while (0)

// ---------------------------------------------------------------- fixture --

// joint/test_pipeline_descriptor.py::_wire(), byte-for-byte semantics.
static const char* kDescriptorWire = R"JSON({
  "descriptor_version": 1,
  "identity": {
    "request_id": "req", "stage": "decode", "generation": 1,
    "stage_epoch": 2, "segment_ordinal": 0, "rank": 1,
    "staging_allocation_id": 9, "segment_id": "pp1/726571/decode/1/2/1/0",
    "buffer_domain_id": "pb1/726571/1/9"
  },
  "stream_kind": "decode_remote_credit",
  "groups": [{
    "group_ordinal": 0, "pass_or_window_ordinal": 0, "source_ordinal": 0,
    "source_key": "source-0", "layer_start": 0, "layer_end": 1,
    "tile_axis": {
      "unit": "stream_bytes", "total_units": 5, "units_per_tile": 2,
      "tile_count": 3, "tail_units": 1, "global_tile_first": 1
    },
    "shards": [{
      "shard_ordinal": 0, "source_rank": 0, "target_rank": 1,
      "total_data_bytes": 5, "bytes_per_unit": 1, "noc_path": [0, 1]
    }]
  }],
  "protocol": {
    "buffer_slots": 2, "release_lag_tiles": 2,
    "initial_tiles_without_release": 2, "fill_ack_bytes": 1,
    "release_ack_bytes": 1, "hbm_charge": true,
    "target_rank_order": "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1",
    "source_rank_order": "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i",
    "tile_branching": "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"
  },
  "prediction_profile": {
    "prefill_consume": "zero_duration_no_extra_consume",
    "decode_consume": {
      "kind": "profiled_consume", "profile_id": "decode-service-profile-v1",
      "compute_ns_total": 240.0, "hbm_bytes_total": 5,
      "resource_key": "hbm:1", "share_rule": "existing_action_work_graph_byte_share_vector"
    }
  },
  "execution_profile": {
    "data_legs": "ordinary_comm_send_recv_path",
    "endpoint_hbm_rule": "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
    "prefill_consume": "existing_body_or_marker_only",
    "decode_consume": "existing_body_nodes_only"
  },
  "logical_id_ranges": [
    {"rank": 0, "first_json_id": 53, "count": 2, "stride": 2, "kind": "virtual_endpoint_hole"},
    {"rank": 1, "first_json_id": 56, "count": 2, "stride": 2, "kind": "virtual_endpoint_hole"}
  ],
  "tag_allocation_runs": [{
    "allocator_scope_id": "builder-0", "emission_ordinal": 7,
    "first_tag": 101, "count": 3, "group_ordinal": 0, "first_tile": 1,
    "first_shard": 0, "tile_count": 3, "shard_count": 1
  }],
  "endpoint_runs": [
    {
      "rank": 0, "group_ordinal": 0, "first_tile": 1, "tile_count": 3,
      "shard_ordinals": [0], "first_json_id": 51, "tile_stride": 2,
      "shard_stride": 1, "initial_parent_json_id": 50,
      "legacy_context": {"request_id": "req", "stage": "decode", "generation": 1},
      "legs": [{
        "endpoint": "data_send", "json_offset": 0, "tag_emission_ordinal": 7,
        "tag_offset": 0, "tag_tile_stride": 1, "tag_shard_stride": 1
      }]
    },
    {
      "rank": 1, "group_ordinal": 0, "first_tile": 1, "tile_count": 3,
      "shard_ordinals": [0], "first_json_id": 54, "tile_stride": 2,
      "shard_stride": 1, "initial_parent_json_id": 53,
      "legacy_context": {"request_id": "req", "stage": "decode", "generation": 1},
      "legs": [{
        "endpoint": "data_recv", "json_offset": 0, "tag_emission_ordinal": 7,
        "tag_offset": 0, "tag_tile_stride": 1, "tag_shard_stride": 1
      }]
    }
  ],
  "endpoint_dependency_runs": [{
    "rank": 0, "group_ordinal": 0, "first_tile": 1, "tile_count": 3,
    "shard_ordinals": [0], "endpoint": "data_send",
    "dependency_kind": "ordinary_node_terminal",
    "first_dependency_json_id": 50, "dependency_tile_stride": 0,
    "dependency_shard_stride": 0
  }],
  "consume_completion_runs": [{
    "rank": 1, "group_ordinal": 0, "first_tile": 1, "tile_count": 3,
    "shard_ordinals": [0], "dependency_kind": "ordinary_node_terminal",
    "first_dependency_json_id": 70, "dependency_tile_stride": 0,
    "dependency_shard_stride": 0
  }],
  "projection": {
    "version": 1,
    "chains": [{
      "group_ordinal": 0, "rank": 1, "tile_count": 3,
      "fill": {
        "leg_id_prefix": "work#dcstream#R1#F", "first_index": 0,
        "fill_order_first": 0, "fill_order_stride": 1,
        "completion_order_first": null, "completion_order_stride": 0,
        "route": ["link:0"], "bytes_per_tile": 2, "latency_ns": 3,
        "start_deps": ["work#start"], "gate_id": "work#gate:dcstream:R1",
        "gate_capacity": 2
      },
      "consume": {
        "leg_id_prefix": "work#dcstream#R1#C", "first_index": 0,
        "consume_order_first": 3, "consume_order_stride": 1,
        "completion_order_first": null, "completion_order_stride": 0,
        "kind": "compute", "route": ["hbm:1"], "group": "decode",
        "latency_first_ns": 7, "latency_rest_ns": 0,
        "consume_deps": ["work#consume-start"], "consume_start_deps": [],
        "gate_id": "work#gate:dcstream:R1",
        "compute_share": {
          "rule": "quotient_remainder_earliest_global_indices",
          "total_units": 8, "total_tiles": 3, "share_index_first": 0,
          "tile_bytes": 2
        },
        "hbm_share": {
          "rule": "quotient_remainder_earliest_global_indices",
          "total_units": 5, "total_tiles": 3, "share_index_first": 0,
          "tile_bytes": 2
        }
      }
    }]
  },
  "anchors": [
    {
      "role": "start", "rank": 0, "binding": "physical_endpoint_slot",
      "node_id": 51,
      "selector": {
        "kind": "tile_endpoint_leg", "group_ordinal": 0, "tile_ordinal": 1,
        "shard_ordinal": 0, "endpoint": "data_send",
        "tag_allocation_emission_ordinal": 7, "tag_offset": 0
      }
    },
    {
      "role": "frontier", "rank": 1, "binding": "physical_endpoint_slot",
      "node_id": 54,
      "selector": {
        "kind": "tile_endpoint_terminal", "group_ordinal": 0,
        "tile_ordinal": 1, "shard_ordinal": 0, "endpoint": "data_recv",
        "tag_allocation_emission_ordinal": 7, "tag_offset": 0
      }
    },
    {
      "role": "service", "rank": 1, "binding": "descriptor_cursor_selector",
      "node_id": null,
      "selector": {"kind": "existing_service_gate", "existing_gate": "decode_service_gate"}
    },
    {
      "role": "settle", "rank": 1, "binding": "descriptor_cursor_selector",
      "node_id": null,
      "selector": {"kind": "all_obligations_settled"}
    }
  ]
})JSON";

static DescriptorPtr make_descriptor() {
    return PipelineDescriptor::from_wire(Json::parse(kDescriptorWire));
}

static DescriptorPtr make_descriptor_with_shards(int shard_count,
                                                 bool shared_terminal) {
    Json wire = Json::parse(kDescriptorWire);
    Json shards = Json::array();
    for (int shard = 0; shard < shard_count; ++shard) {
Json row = wire["groups"][0]["shards"][0];
        row["shard_ordinal"] = shard;
        shards.push_back(row);
    }
    wire["groups"][0]["shards"] = shards;
    wire["tag_allocation_runs"][0]["count"] = 3 * shard_count;
    wire["tag_allocation_runs"][0]["shard_count"] = shard_count;
    Json& source_run = wire["endpoint_runs"][0];
    Json& target_run = wire["endpoint_runs"][1];
    target_run["shard_ordinals"] = Json::array();
    for (int shard = 0; shard < shard_count; ++shard) {
        target_run["shard_ordinals"].push_back(shard);
    }
    source_run["legs"][0]["tag_tile_stride"] = shard_count;
    target_run["tile_stride"] = shard_count;
    target_run["shard_stride"] = 1;
    target_run["legs"][0]["tag_tile_stride"] = shard_count;
    target_run["legs"][0]["tag_shard_stride"] = 1;
    Json id_ranges = Json::array();
    id_ranges.push_back(wire["logical_id_ranges"][0]);
    id_ranges.push_back(Json::object({
        {"rank", 1}, {"first_json_id", 55},
        {"count", 3 * shard_count - 1}, {"stride", 1},
        {"kind", "virtual_endpoint_hole"}}));
    wire["logical_id_ranges"] = id_ranges;
    Json consume = wire["consume_completion_runs"][0];
    consume["shard_ordinals"] = target_run["shard_ordinals"];
    consume["first_dependency_json_id"] = 1000;
    consume["dependency_shard_stride"] = shared_terminal ? 0 : 1;
    wire["consume_completion_runs"][0] = consume;
    return PipelineDescriptor::from_wire(wire);
}

// _initial_cursor_run (pipeline_ledger.py 1215-1265).
static Json initial_cursor_row(const PipelineDescriptor& descriptor,
                               std::uint64_t run_index) {
    const auto& run = descriptor.endpoint_runs[run_index];
    const std::size_t shards = run.shard_ordinals.size();
    const std::size_t legs = run.legs.size();
    const std::uint64_t total =
        run.tile_count * shards * legs;
    std::uint64_t prefix = 0;
    bool any_active = false;
    for (std::size_t si = 0; si < shards; ++si) {
        const auto shard = descriptor.selected_shard_tile(
            run.group_ordinal, run.first_tile - 1, run.shard_ordinals[si]);
        for (std::size_t li = 0; li < legs; ++li) {
            const bool active =
                run.legs[li].endpoint == "body_marker"
                || (shard.second.emit_endpoint && shard.second.data_bytes != 0);
            if (active) any_active = true;
            if (prefix == si * legs + li && !active) prefix += 1;
        }
    }
    if (!any_active) prefix = total;
    const std::uint64_t position = prefix;
    Json row = Json::object();
    row["rank"] = run.rank;
    row["group_ordinal"] = run.group_ordinal;
    row["first_tile"] = run.first_tile;
    row["tile_count"] = run.tile_count;
    row["completed_endpoint_count"] = prefix;
    row["cancelled_endpoint_count"] = 0;
    row["remaining_endpoint_count"] = total - prefix;
    row["in_flight"] = false;
    if (position == total) {
        row["next_store_id"] = nullptr;
        row["tile_offset"] = run.tile_count;
        row["shard_index"] = 0;
        row["leg_index"] = 0;
    } else {
        const std::uint64_t per_tile = shards * legs;
        const std::uint64_t tile_offset = position / per_tile;
        const std::uint64_t shard_index =
            (position % per_tile) / legs;
        const std::uint64_t leg_index = (position % per_tile) % legs;
        row["next_store_id"] = descriptor.endpoint_id(
            run_index, run.first_tile + tile_offset,
            run.shard_ordinals[shard_index], leg_index) + 1;
        row["tile_offset"] = tile_offset;
        row["shard_index"] = shard_index;
        row["leg_index"] = leg_index;
    }
    row["initial_parent_store_id"] = run.initial_parent_json_id
        ? Json(*run.initial_parent_json_id + 1) : Json(nullptr);
    row["completed_out_of_order"] = Json::array();
    row["active_endpoints"] = Json::array();
    return row;
}

static Json descriptor_commit_entry(const PipelineDescriptor& descriptor) {
    Json entry = Json::object();
    entry["identity"] = descriptor.to_wire()["identity"];
    entry["descriptor"] = descriptor.to_wire();
    entry["commit_tick"] = 0;
    entry["cursor_version"] = 0;
    entry["service_done"] = false;
    entry["settle_done"] = false;
    entry["retired"] = false;
    return entry;
}

static Json empty_envelope(const std::string& kind, std::uint64_t delivery,
                           std::uint64_t base_watermark,
                           std::uint64_t base_version,
                           std::uint64_t tick = 0) {
    Json envelope = Json::object();
    envelope["producer"] = "cpp_online";
    envelope["schema_version"] = 4;
    envelope["protocol_version"] = 4;
    envelope["bridge_protocol"] = "bridge-v4-20261011";
    envelope["work_ledger_schema"] = "workledger-v4-20261011";
    envelope["epoch"] = "test-run";
    envelope["delivery_sequence"] = delivery;
    envelope["base_watermark"] = base_watermark;
    envelope["logical_seq_start"] = base_watermark + 1;
    envelope["logical_seq_end"] = base_watermark;
    envelope["logical_event_count"] = 0;
    envelope["replay_token"] =
        "cpp_online/test-run/" + std::to_string(delivery);
    envelope["descriptor_version"] = 1;
    envelope["base_state_version"] = base_version;
    envelope["result_state_version"] = base_version;
    envelope["snapshot_kind"] = kind;
    envelope["tick"] = tick;
    envelope["descriptor_commits"] = Json::array();
    envelope["node_transitions"] = Json::array();
    envelope["pipeline_transition_runs"] = Json::array();
    envelope["cursor_patches"] = Json::array();
    envelope["active_resources"] = Json::array();
    return envelope;
}

// test_pipeline_ledger_v2.py::_pipeline_run.
static Json base_pipeline_run(const std::string& segment,
                              std::uint64_t seq,
                              const std::string& transition,
                              std::uint64_t tick,
                              std::uint64_t cv_start = 0,
                              std::uint64_t cv_end = 0) {
    Json run = Json::object();
    run["first_logical_seq"] = seq;
    run["pattern_id"] = segment + "/" + transition + "/" + std::to_string(seq);
    run["repeat_count"] = 1;
    run["tile_first"] = nullptr;
    run["tile_stride"] = nullptr;
    run["rank_first"] = nullptr;
    run["rank_stride"] = nullptr;
    run["shard_first"] = nullptr;
    run["shard_stride"] = nullptr;
    run["transition_kind"] = transition;
    run["descriptor_segment_id"] = segment;
    run["endpoint_selector"] = nullptr;
    run["store_id_first"] = nullptr;
    run["store_id_stride"] = nullptr;
    run["legacy_context"] = nullptr;
    run["tick_first"] = tick;
    run["tick_stride"] = 0;
    run["resource_leg"] = nullptr;
    run["resource_quantity_prefix"] = Json::array(
        {Json{{"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}},
Json{{"ops", nullptr}, {"bytes", nullptr}, {"latency_ns", nullptr}}});
    run["cursor_version_start"] = cv_start;
    run["cursor_version_end"] = cv_end;
    run["slot_patches"] = Json::array();
    run["credit_patches"] = Json::array();
    run["obligation_patches"] = Json::array();
    return run;
}

static void bind_selector(Json& run, const PipelineDescriptor& descriptor,
                          std::uint64_t run_index, std::uint64_t tile,
                          std::uint64_t shard, const std::string& endpoint) {
    const auto& endpoint_run = descriptor.endpoint_runs[run_index];
    run["endpoint_selector"] = Json::object({
        {"group_ordinal", endpoint_run.group_ordinal},
        {"tile_ordinal", tile},
        {"shard_ordinal", shard},
        {"endpoint", endpoint}});
    run["tile_first"] = tile;
    run["rank_first"] = endpoint_run.rank;
    run["shard_first"] = shard;
    run["tile_stride"] = 0;
    run["rank_stride"] = 0;
    run["shard_stride"] = 0;
    run["store_id_stride"] = 0;
    run["store_id_first"] =
        descriptor.endpoint_id(run_index, tile, shard, 0) + 1;
    run["legacy_context"] = Json::object({
        {"request_id", endpoint_run.legacy_context.request_id},
        {"stage", endpoint_run.legacy_context.stage},
        {"generation", endpoint_run.legacy_context.generation}});
}

static Json target_delta(const std::string& domain, std::uint64_t rank,
                         std::uint64_t slot_index,
                         std::uint64_t before_generation,
                         std::uint64_t predecessor_seq,
                         std::uint64_t predecessor_count,
Json operations) {
    Json delta = Json::object();
    delta["form"] = "target_delta";
    delta["key"] = Json::object({
        {"buffer_domain_id", domain},
        {"role", "target"},
        {"rank", rank},
        {"shard_ordinal", nullptr},
        {"slot_index", slot_index}});
    delta["before_slot_generation"] = before_generation;
    delta["predecessor_logical_seq"] = predecessor_seq;
    delta["predecessor_consumed_count"] = predecessor_count;
    delta["operations"] = std::move(operations);
    return delta;
}

static Json owner_row(const PipelineDescriptor& descriptor,
                      std::uint64_t group_ordinal, std::uint64_t tile) {
    return Json::object({
        {"segment_id", descriptor.identity.segment_id},
        {"pipeline_generation", descriptor.identity.generation},
        {"stage_epoch", descriptor.identity.stage_epoch},
        {"staging_allocation_id", descriptor.identity.staging_allocation_id},
        {"group_ordinal", group_ordinal},
        {"tile_ordinal", tile}});
}

static std::vector<std::uint64_t> claimable_shards(
    const PipelineDescriptor& descriptor, std::uint64_t group_ordinal,
    std::uint64_t tile) {
    const auto* group = descriptor.group_by_ordinal(group_ordinal);
    std::vector<std::uint64_t> out;
    for (std::uint64_t shard = 0; shard < group->shards.size(); ++shard) {
        const auto tile_shard = descriptor.selected_shard_tile(
            group_ordinal, tile - 1, shard);
        if (tile_shard.second.emit_endpoint
                && tile_shard.second.target_rank
                    == descriptor.identity.rank) {
            out.push_back(shard);
        }
    }
    return out;
}

// ---------------------------------------------------------------- harness --

struct LedgerHarness {
    WorkLedgerCore work;
    PipelineLedgerCore pipeline;

    LedgerHarness() { jointledger::bind_work_ledger(pipeline, work); }

    ParsedEnvelope parse(const Json& envelope) const {
        RetainedRegistry retained;
        retained.descriptors = &pipeline.descriptors();
        retained.selector_indexes = &pipeline.selector_indexes();
        const bool allow_retained_replay =
            static_cast<std::int64_t>(
                envelope.value("delivery_sequence", 0ull))
            == pipeline.delivery_sequence()
            && pipeline.delivery_sequence() >= 0;
        const std::string text = envelope.dump();
        return parse_envelope_v4(text.data(), text.size(), retained,
                                 allow_retained_replay,
                                 envelope.value("delivery_sequence", 0ull),
                                 envelope.value("tick", 0ull));
    }

    // prepare + commit of a wire envelope (the DualLedger flow).
    PipelineReceipt apply(const Json& envelope) {
        ParsedEnvelope parsed = parse(envelope);
        const std::string text = envelope.dump();
        auto tx = work.begin_protocol_v3_transaction(
            parsed.header.delivery_sequence, parsed.header.tick,
            parsed.header.full ? "full" : "delta");
        const int slot = pipeline.prepare(parsed, tx, text);
        work.tx_commit(tx);
        return pipeline.commit_prepared(slot);
    }

    // prepare only; returns the slot (caller must commit or discard).
    int stage(const Json& envelope, WorkLedgerCore::Transaction& tx) {
        ParsedEnvelope parsed = parse(envelope);
        const std::string text = envelope.dump();
        tx = work.begin_protocol_v3_transaction(
            parsed.header.delivery_sequence, parsed.header.tick,
            parsed.header.full ? "full" : "delta");
        return pipeline.prepare(parsed, tx, text);
    }

    // Idempotent-replay path: production short-circuits replays in the outer
    // frame before any WorkLedger transaction begins (the work core's
    // monotonic delivery gate rejects a repeat sequence by design), so the
    // pipeline replay branch is driven here with a monotonic transaction
    // handle; the pipeline gate returns before the transaction is touched.
    PipelineReceipt apply_replay(const Json& envelope) {
        ParsedEnvelope parsed = parse(envelope);
        const std::string text = envelope.dump();
        auto tx = work.begin_protocol_v3_transaction(
            parsed.header.delivery_sequence + 1, parsed.header.tick, "delta");
        const int slot = pipeline.prepare(parsed, tx, text);
        return pipeline.commit_prepared(slot);
    }
};

// The v4 full frame: descriptor declaration + target-slot initializes +
// initial cursor patches + the initial obligation rows.
static Json full_with_descriptor(const DescriptorPtr& descriptor,
const Json& node_events = Json::array()) {
    const std::string segment = descriptor->identity.segment_id;
    const std::string domain = descriptor->identity.buffer_domain_id;
    Json run = base_pipeline_run(segment, 1, "descriptor_commit", 0);
    Json obligation_patches = Json::array();
    Json cursor_rows = Json::array();
    for (std::size_t run_index = 0;
         run_index < descriptor->endpoint_runs.size(); ++run_index) {
        const auto& endpoint_run = descriptor->endpoint_runs[run_index];
const Json cursor = initial_cursor_row(*descriptor, run_index);
        cursor_rows.push_back(Json::object({
            {"run_index", run_index},
            {"value", cursor}}));
        const std::uint64_t remaining =
            cursor["remaining_endpoint_count"];
        if (remaining > 0) {
Json row = Json::object({
                {"obligation_id", segment + "/run/" + std::to_string(run_index)},
                {"segment_id", segment},
                {"rank", endpoint_run.rank},
                {"first_store_id", cursor["next_store_id"]},
                {"remaining_endpoint_count", remaining},
                {"state", "pending"}});
            obligation_patches.push_back(Json::object({
                {"before", nullptr}, {"after", row}}));
        }
        run["slot_patches"].push_back(target_delta(
            domain, descriptor->identity.rank, run_index, 0, 0, 0,
Json::array({Json::object({
                {"op", "initialize"}, {"slot_generation", 0}})})));
    }
    run["obligation_patches"] = obligation_patches;
    const std::uint64_t count = 1 + node_events.size();
    Json envelope = empty_envelope("full", 0, 0, 0);
    envelope["logical_seq_end"] = count;
    envelope["logical_event_count"] = count;
    envelope["result_state_version"] = 1;
    envelope["descriptor_commits"] = Json::array(
        {descriptor_commit_entry(*descriptor)});
    envelope["node_transitions"] = node_events;
    envelope["pipeline_transition_runs"] = Json::array({run});
    envelope["cursor_patches"] = Json::array({Json::object({
        {"segment_id", segment},
        {"cursor_version", 0},
        {"retired", false},
        {"runs", cursor_rows}})});
    return envelope;
}

// ------------------------------------------------------------------ cases --

static void case_gates_and_replay() {
    // 1: empty full accepted; scalars/receipt/snapshot.
    {
        LedgerHarness harness;
const Json envelope = empty_envelope("full", 0, 0, 0, 5);
        const PipelineReceipt receipt = harness.apply(envelope);
CHECK(receipt.snapshot_kind == "full");

CHECK(receipt.delivery_sequence == 0);

CHECK(receipt.event_watermark == 0);

CHECK(receipt.state_version == 0);   // empty: no visible change

CHECK(!receipt.replayed);

CHECK(harness.pipeline.initialized());

CHECK(harness.pipeline.epoch() == "test-run");

CHECK(harness.pipeline.delivery_sequence() == 0);

CHECK(harness.pipeline.last_tick() == 5);

const Json snapshot = harness.pipeline.snapshot_state_json();

CHECK(snapshot["epoch"] == "test-run");

CHECK(snapshot["delivery_sequence"] == 0);

CHECK(snapshot["event_watermark"] == 0);

CHECK(snapshot["state_version"] == 0);

CHECK(snapshot["tick"] == 5);

CHECK(snapshot["descriptors"].empty());

CHECK(snapshot["active_resources"].empty());

const Json pricing = harness.pipeline.pricing_json();

CHECK(pricing["state_version"] == 0);

CHECK(pricing["tick"] == 5);
    }
    // 2: empty delta after full advances watermark/version.
    {
        LedgerHarness harness;
CHECK(harness.apply(empty_envelope("full", 0, 0, 0)).state_version == 0);

const Json delta = empty_envelope("delta", 1, 0, 0, 7);
        const PipelineReceipt receipt = harness.apply(delta);
CHECK(receipt.snapshot_kind == "delta");

CHECK(receipt.event_watermark == 0);

CHECK(receipt.state_version == 0);

CHECK(harness.pipeline.watermark() == 0);

CHECK(harness.pipeline.last_tick() == 7);
    }
    // 3: identical replay is a zero-change accept.
    {
        LedgerHarness harness;
const Json envelope = empty_envelope("full", 0, 0, 0);

harness.apply(envelope);

const Json before = harness.pipeline.snapshot_state_json();
        const PipelineReceipt receipt = harness.apply_replay(envelope);
CHECK(receipt.replayed);

CHECK(harness.pipeline.replay_count() == 1);

CHECK(harness.pipeline.snapshot_state_json() == before);
    }
    // 4: same delivery with different bytes fails closed.
    {
        LedgerHarness harness;
const Json envelope = empty_envelope("full", 0, 0, 0);

harness.apply(envelope);

Json mutated = envelope;
        mutated["tick"] = 9;   // same delivery sequence, different content
CHECK_THROW_STATEMENT(harness.apply_replay(mutated),
                              "typed-different canonical content");
CHECK(harness.pipeline.delivery_sequence() == 0);
    }
    // 5: delivery gap.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0));

CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("delta", 2, 0, 1)),
            "delivery_sequence gap or stale replay");
    }
    // 6: stale delivery.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0));

harness.apply(empty_envelope("delta", 1, 0, 0));

CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("delta", 0, 0, 0)),
            "v3 delivery sequence is not monotonic");   // begin gate first
    }
    // 7: delta before full.
    {
        LedgerHarness harness;
CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("delta", 0, 0, 0)),
            "delta received before initial full snapshot");
CHECK(!harness.pipeline.initialized());
    }
    // 8: full with nonzero base watermark.
    {
        LedgerHarness harness;
CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("full", 0, 3, 0)),
            "full watermark or state-version transition is invalid");
    }
    // 9: second full after initialization.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0));

CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("full", 1, 0, 1)),
            "v3 full snapshot requires a pristine WorkLedger");
    }
    // 10: epoch change mid-run.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0));

Json delta = empty_envelope("delta", 1, 0, 1);
        delta["epoch"] = "other-run";
        delta["replay_token"] = "cpp_online/other-run/1";
CHECK_THROW_STATEMENT(harness.apply(delta), "v3 epoch changed during run");
    }
    // 11: base watermark/state-version mismatch.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0));

CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("delta", 1, 5, 1)),
            "delta base watermark/state version mismatch");
CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("delta", 1, 0, 9)),
            "delta base watermark/state version mismatch");
    }
    // 12: tick regression.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0, 10));

CHECK_THROW_STATEMENT(

harness.apply(empty_envelope("delta", 1, 0, 0, 4)),
            "v3 envelope tick regressed");   // begin gate first
    }
}

static void case_merge_walk() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    // 13: node and pipeline transitions sharing a logical sequence.
    {
        LedgerHarness harness;
Json envelope = full_with_descriptor(descriptor);
        // seq 1 is the descriptor_commit row; add a node event also at 1.
Json node = Json::object({
            {"logical_seq", 1}, {"tick", 0}, {"kind", "commit"},
            {"work_id", "req#decode:1:r1n80"}, {"payload", Json::object()}});
        envelope["node_transitions"] = Json::array({node});
        envelope["logical_seq_end"] = 2;
        envelope["logical_event_count"] = 2;
CHECK_THROW_STATEMENT(

harness.apply(envelope),
            "node and pipeline transitions share a logical sequence");
    }
    // 14: logical event count differs from typed transition count.
    {
        LedgerHarness harness;
Json envelope = full_with_descriptor(descriptor);
        envelope["logical_seq_end"] = 2;
        envelope["logical_event_count"] = 2;
        // still exactly one typed row: prepare's count check fires
CHECK_THROW_STATEMENT(

harness.apply(envelope),
            "logical event count differs from typed transition count");
    }
    // 15: node/pipeline interleave accepted; work staged between rows.
    {
        LedgerHarness harness;
Json node = Json::object({
            {"logical_seq", 2}, {"tick", 0}, {"kind", "commit"},
            {"work_id", "req#decode:1:r1n80"}, {"payload", Json::object()}});
Json envelope = full_with_descriptor(descriptor, Json::array({node}));
        const PipelineReceipt receipt = harness.apply(envelope);
CHECK(receipt.logical_event_count == 2);

CHECK(harness.work.work("req#decode:1:r1n80") != nullptr);

CHECK(harness.work.active_work_count() == 1);
    }
    // 16: node event without a bound WorkLedger core.
    {
        PipelineLedgerCore unbound;
        LedgerHarness source;
Json node = Json::object({
            {"logical_seq", 2}, {"tick", 0}, {"kind", "commit"},
            {"work_id", "req#decode:1:r1n80"}, {"payload", Json::object()}});
const Json envelope =
            full_with_descriptor(descriptor, Json::array({node}));
        ParsedEnvelope parsed = source.parse(envelope);
        WorkLedgerCore::Transaction tx;   // unbound default handle
CHECK_THROW_STATEMENT(
            unbound.prepare(parsed, tx, envelope.dump()),
            "v3 node transitions require a coordinated WorkLedger COW transaction");
    }
}

// Delivery builder declared early: later cases reuse it below.
static Json issue_target_tile1(const DescriptorPtr& descriptor,
                               std::uint64_t delivery,
                               std::uint64_t watermark,
                               std::uint64_t base_version);

static void case_descriptor_lifecycle() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    // 17: descriptor full accepted; snapshot carries registry state.
    {
        LedgerHarness harness;
        const PipelineReceipt receipt = harness.apply(
            full_with_descriptor(descriptor));
CHECK(receipt.logical_event_count == 1);

CHECK(harness.pipeline.descriptors().count(segment) == 1);

const Json snapshot = harness.pipeline.snapshot_state_json();

CHECK(snapshot["descriptors"].size() == 1);

CHECK(snapshot["descriptors"][segment] == descriptor->to_wire());

CHECK(snapshot["descriptor_state"][segment]["cursor_version"] == 0);

CHECK(snapshot["descriptor_state"][segment]["retired"] == false);

CHECK(snapshot["authoritative_cursors"].size() == 1);

CHECK(snapshot["authoritative_cursors"][0]["segment_id"] == segment);

CHECK(snapshot["authoritative_cursors"][0]["runs"].size() == 2);

CHECK(snapshot["obligations"].size() == 2);

CHECK(snapshot["obligations"][0]["state"] == "pending");

CHECK(snapshot["slots"].size() == 2);   // target initializes

CHECK(snapshot["slots"][0]["occupied"] == false);

CHECK(snapshot["credits"].empty());
    }
    // 18: redeclared descriptor in a later delivery rejected.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["descriptor_commits"] =
Json::array({descriptor_commit_entry(*descriptor)});

CHECK_THROW_STATEMENT(harness.apply(delta),
                              "descriptor segment was declared more than once");
    }
    // 19: declaration without a descriptor_commit transition.
    {
        LedgerHarness harness;
Json envelope = full_with_descriptor(descriptor);
        envelope["pipeline_transition_runs"] = Json::array();
        envelope["logical_event_count"] = 0;
        envelope["logical_seq_end"] = 0;   // empty interval == watermark
CHECK_THROW_STATEMENT(harness.apply(envelope),
                              "every descriptor declaration needs exactly one descriptor_commit");
    }
    // 20: endpoint transition before descriptor_commit.
    {
        LedgerHarness harness;
Json envelope = full_with_descriptor(descriptor);

Json run = base_pipeline_run(segment, 1, "service_done", 0);
        envelope["pipeline_transition_runs"][0] = run;
CHECK_THROW_STATEMENT(

harness.apply(envelope),
            "pipeline transition precedes descriptor_commit");
    }
    // 21: missing initial cursor patch.
    {
        LedgerHarness harness;
Json envelope = full_with_descriptor(descriptor);
        envelope["cursor_patches"] = Json::array();
CHECK_THROW_STATEMENT(

harness.apply(envelope),
            "cursor patches do not exactly cover initialized, retired, or changed cursors");
    }
    // 22: new descriptor patch marked retired.
    {
        LedgerHarness harness;
Json envelope = full_with_descriptor(descriptor);
        envelope["cursor_patches"][0]["retired"] = true;
        for (Json& run : envelope["cursor_patches"][0]["runs"]) {
            run["value"] = nullptr;
        }
        // retired tombstones are the legal wire shape; the stage core
        // rejects a new descriptor that starts retired.
CHECK_THROW_STATEMENT(

harness.apply(envelope),
            "new descriptor cannot start retired");
    }
    // 23: metadata-only cursor patch has no sequenced version change.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        // stable cursor version (0 -> 0) makes the patch required only via
        // the dirty run; the empty run list then trips the metadata gate.
        delta["pipeline_transition_runs"][0]["cursor_version_end"] = 0;
        delta["cursor_patches"][0]["cursor_version"] = 0;
        delta["cursor_patches"][0]["runs"] = Json::array();
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "metadata-only cursor patch has no sequenced version change");
    }
    // 24: cursor pre-version mismatch.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
Json run = base_pipeline_run(segment, 2, "service_done", 0,
                                     /*cv_start=*/7, /*cv_end=*/7);
        delta["pipeline_transition_runs"] = Json::array({run});
CHECK_THROW_STATEMENT(harness.apply(delta),
                              "pipeline transition cursor pre-version mismatch");
    }
}

// Delivery-1 for the base descriptor: tile_issue of run1 tile 1 (target).
static Json issue_target_tile1(const DescriptorPtr& descriptor,
                               std::uint64_t delivery,
                               std::uint64_t watermark,
                               std::uint64_t base_version) {
    const std::string segment = descriptor->identity.segment_id;
    const std::string domain = descriptor->identity.buffer_domain_id;
    Json delta = empty_envelope("delta", delivery, watermark, base_version);
    delta["logical_seq_start"] = watermark + 1;
    delta["logical_seq_end"] = watermark + 1;
    delta["logical_event_count"] = 1;
    delta["result_state_version"] = base_version + 1;
    Json run = base_pipeline_run(segment, watermark + 1, "tile_issue", 0,
                                 /*cv_start=*/0, /*cv_end=*/1);
    bind_selector(run, *descriptor, 1, 1, 0, "data_recv");
    const Json owner = owner_row(*descriptor, 0, 1);
    run["slot_patches"] = Json::array({target_delta(
        domain, descriptor->identity.rank,
        /*slot_index=*/0, /*before_generation=*/0,
        /*predecessor_seq=*/1, /*predecessor_count=*/0,
Json::array({Json::object({
            {"op", "claim"},
            {"slot_generation", 1},
            {"owner", owner},
            {"target_shards", Json::array({0})}})}))});
    // obligation run/1: pending(3, first 55) -> active(3, first 55)
    Json before = Json::object({
        {"obligation_id", segment + "/run/1"},
        {"segment_id", segment},
        {"rank", 1},
        {"first_store_id", 55},
        {"remaining_endpoint_count", 3},
        {"state", "pending"}});
    Json after = before;
    after["state"] = "active";
    run["obligation_patches"] = Json::array({Json::object({
        {"before", before}, {"after", after}})});
    // final cursor row for run 1 (remaining keeps the unresolved total)
    Json cursor = initial_cursor_row(*descriptor, 1);
    cursor["completed_endpoint_count"] = 0;
    cursor["in_flight"] = true;
    cursor["tile_offset"] = 1;
    cursor["next_store_id"] = 57;
    Json active = Json::object({
        {"store_id", 55},
        {"tile_ordinal", 1},
        {"shard_ordinal", 0},
        {"endpoint", "data_recv"},
        {"transfer_id", 1000},
        {"flow_id", nullptr},
        {"network_done", false},
        {"hbm_done", false},
        {"owner", owner}});
    cursor["active_endpoints"] = Json::array({active});
    delta["pipeline_transition_runs"] = Json::array({run});
    delta["cursor_patches"] = Json::array({Json::object({
        {"segment_id", segment},
        {"cursor_version", 1},
        {"retired", false},
        {"runs", Json::array({Json::object({
            {"run_index", 1}, {"value", cursor}})})}})});
    return delta;
}

static void case_cursor_simulation() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    // 25: issue at a non-cursor position.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        bind_selector(delta["pipeline_transition_runs"][0], *descriptor, 1,
                      /*tile=*/2, 0, "data_recv");   // position mismatch
Json& run = delta["pipeline_transition_runs"][0];
        run["slot_patches"][0]["key"]["slot_index"] = 1;
        run["slot_patches"][0]["operations"][0]["owner"]["tile_ordinal"] = 2;
Json& value = delta["cursor_patches"][0]["runs"][0]["value"];
        value["tile_offset"] = 2;
        value["next_store_id"] = 59;
        value["active_endpoints"][0]["tile_ordinal"] = 2;
        value["active_endpoints"][0]["store_id"] = 57;
        value["active_endpoints"][0]["owner"]["tile_ordinal"] = 2;
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "endpoint issue does not match the next legal cursor position");
    }
    // 26: completion without an active issue.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
Json run = base_pipeline_run(segment, 2, "endpoint_done", 0, 0, 1);
        bind_selector(run, *descriptor, 1, 1, 0, "data_recv");
        delta["pipeline_transition_runs"] = Json::array({run});
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "endpoint completion has no active issue");
    }
    // 27: issue, then out-of-order completion merges adjacent ranges.
    {
        LedgerHarness harness;
        const DescriptorPtr wide = make_descriptor_with_shards(2, true);
        const std::string wide_segment = wide->identity.segment_id;
harness.apply(full_with_descriptor(wide));
        // Two issues at positions 0 and 1 (run 1, tile 1, shards 0/1).
Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_start"] = 2;
        delta["logical_seq_end"] = 3;
        delta["logical_event_count"] = 2;
        delta["result_state_version"] = 2;
Json issue0 = base_pipeline_run(wide_segment, 2, "tile_issue", 0, 0, 1);
        bind_selector(issue0, *wide, 1, 1, 0, "data_recv");
Json issue1 = base_pipeline_run(wide_segment, 3, "tile_issue", 0, 1, 2);
        bind_selector(issue1, *wide, 1, 1, 1, "data_recv");
        const std::string domain = wide->identity.buffer_domain_id;
const Json owner = owner_row(*wide, 0, 1);
        // tile 1 maps to slot 0 (two-slot parity) claiming both shards
        issue0["slot_patches"].push_back(target_delta(
            domain, wide->identity.rank, 0, 0, 1, 0,
Json::array({Json::object({
                {"op", "claim"}, {"slot_generation", 1},
                {"owner", owner},
                {"target_shards", Json::array({0, 1})}})})));
        // obligation run/1: 6 pending -> 6 active, first store 55
Json before = Json::object({
            {"obligation_id", wide_segment + "/run/1"},
            {"segment_id", wide_segment}, {"rank", 1},
            {"first_store_id", 55},
            {"remaining_endpoint_count", 6}, {"state", "pending"}});
Json after = before;
        after["state"] = "active";
        issue0["obligation_patches"] = Json::array({Json::object({
            {"before", before}, {"after", after}})});
        delta["pipeline_transition_runs"] = Json::array({issue0, issue1});
Json cursor = initial_cursor_row(*wide, 1);
        cursor["in_flight"] = true;
        cursor["tile_offset"] = 1;
        cursor["shard_index"] = 0;
        cursor["leg_index"] = 0;
        cursor["next_store_id"] = 57;
Json active0 = Json::object({
            {"store_id", 55}, {"tile_ordinal", 1}, {"shard_ordinal", 0},
            {"endpoint", "data_recv"}, {"transfer_id", 2000},
            {"flow_id", nullptr}, {"network_done", false},
            {"hbm_done", false}, {"owner", owner}});
Json active1 = active0;
        active1["store_id"] = 56;
        active1["shard_ordinal"] = 1;
        active1["transfer_id"] = 2001;
        cursor["active_endpoints"] = Json::array({active0, active1});
        delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", wide_segment},
            {"cursor_version", 2},
            {"retired", false},
            {"runs", Json::array({Json::object({
                {"run_index", 1}, {"value", cursor}})})}})});
harness.apply(delta);
        // Next delivery: complete ordinal 1 (out of order), then ordinal 0
        // (prefix merge); the exported ooo ranges must normalize to empty.
Json complete = empty_envelope("delta", 2, 3, 2);
        complete["logical_seq_start"] = 4;
        complete["logical_seq_end"] = 5;
        complete["logical_event_count"] = 2;
        complete["result_state_version"] = 3;
Json done1 = base_pipeline_run(wide_segment, 4, "endpoint_done", 0, 2, 3);
        bind_selector(done1, *wide, 1, 1, 1, "data_recv");
Json done0 = base_pipeline_run(wide_segment, 5, "endpoint_done", 0, 3, 4);
        bind_selector(done0, *wide, 1, 1, 0, "data_recv");
Json obl_before = Json::object({
            {"obligation_id", wide_segment + "/run/1"},
            {"segment_id", wide_segment}, {"rank", 1},
            {"first_store_id", 55},
            {"remaining_endpoint_count", 6}, {"state", "active"}});
Json obl_after = obl_before;
        obl_after["remaining_endpoint_count"] = 5;   // one active remains
        done1["obligation_patches"] = Json::array({Json::object({
            {"before", obl_before}, {"after", obl_after}})});
Json obl0_after = Json::object({
            {"obligation_id", wide_segment + "/run/1"},
            {"segment_id", wide_segment}, {"rank", 1},
            {"first_store_id", 57},
            {"remaining_endpoint_count", 4}, {"state", "pending"}});
        done0["obligation_patches"] = Json::array({Json::object({
            {"before", obl_after}, {"after", obl0_after}})});
        complete["pipeline_transition_runs"] = Json::array({done1, done0});
Json final_cursor = cursor;
        final_cursor["completed_endpoint_count"] = 2;
        final_cursor["remaining_endpoint_count"] = 4;
        final_cursor["in_flight"] = false;
        final_cursor["active_endpoints"] = Json::array();
        final_cursor["completed_out_of_order"] = Json::array();  // merged
        complete["cursor_patches"] = Json::array({Json::object({
            {"segment_id", wide_segment},
            {"cursor_version", 4},
            {"retired", false},
            {"runs", Json::array({Json::object({
                {"run_index", 1}, {"value", final_cursor}})})}})});
        const PipelineReceipt receipt = harness.apply(complete);
CHECK(receipt.state_version == 3);
    }
    // 28: redundant cursor row republish.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);

Json initial = initial_cursor_row(*descriptor, 1);
        delta["cursor_patches"][0]["runs"][0]["value"] = initial;
        // note: initial row equals the accepted row -> redundant republish
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "cursor run patch redundantly republishes an unchanged row");
    }
    // 29: cursor counts disagree with sequenced progress.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);

Json& value29 = delta["cursor_patches"][0]["runs"][0]["value"];
        value29["tile_offset"] = 2;
        value29["next_store_id"] = 59;
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "cursor next position disagrees with sequenced endpoint issues");
    }
    // 30: touched run without its final patch.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        delta["cursor_patches"][0]["runs"] = Json::array();
        // the dirty run has no final row, so the sequenced active-endpoint
        // aggregate diverges from the (empty) patch
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "active endpoint aggregate differs from final cursor rows");
    }
    // 31: service_done before cursor issue exhaustion.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
Json run = base_pipeline_run(segment, 2, "service_done", 0, 0, 1);
        delta["pipeline_transition_runs"] = Json::array({run});
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "service_done precedes cursor issue exhaustion");
    }
}

static void case_slots_credits_obligations() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    const std::string domain = descriptor->identity.buffer_domain_id;
    // 32: full v4 target claim chain accepted.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

const Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        const PipelineReceipt receipt = harness.apply(delta);
CHECK(receipt.state_version == 2);

const Json snapshot = harness.pipeline.snapshot_state_json();

CHECK(snapshot["slots"].size() == 2);
        bool found_claim = false;
        for (const Json& slot : snapshot["slots"]) {
            if (slot["slot_index"] == 0 && slot["occupied"] == true) {
                found_claim = true;
CHECK(slot["target_shards_to_consume"] == Json::array({0}));

CHECK(slot["consumed_target_shards"].empty());
            }
        }
CHECK(found_claim);

CHECK(snapshot["obligations"].size() == 2);

CHECK(snapshot["obligations"][1]["state"] == "active");

const Json cursor = snapshot["authoritative_cursors"][0]["runs"][1];

CHECK(cursor["in_flight"] == true);

CHECK(cursor["next_store_id"] == 57);

CHECK(cursor["active_endpoints"].size() == 1);
    }
    // 33: source issue creates its credit row with exact balance.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
Json run = base_pipeline_run(segment, 2, "tile_issue", 0, 0, 1);
        bind_selector(run, *descriptor, 0, 1, 0, "data_send");
const Json owner = owner_row(*descriptor, 0, 1);

Json slot = Json::object({
            {"buffer_domain_id", domain},
            {"role", "source"},
            {"rank", 0},
            {"shard_ordinal", 0},
            {"slot_index", 0},
            {"occupied", true},
            {"owner", owner},
            {"consumed", false},
            {"target_shards_to_consume", Json::array()},
            {"consumed_target_shards", Json::array()}});
        run["slot_patches"] = Json::array({Json::object({
            {"before", nullptr}, {"after", slot}})});
Json credit = Json::object({
            {"buffer_domain_id", domain},
            {"source_rank", 0},
            {"shard_ordinal", 0},
            {"capacity", 2},
            {"available", 1},
            {"outstanding_ack_count", 0}});
        run["credit_patches"] = Json::array({Json::object({
            {"before", nullptr}, {"after", credit}})});
Json before = Json::object({
            {"obligation_id", segment + "/run/0"},
            {"segment_id", segment}, {"rank", 0},
            {"first_store_id", 52},
            {"remaining_endpoint_count", 3}, {"state", "pending"}});
Json after = before;
        after["state"] = "active";
        run["obligation_patches"] = Json::array({Json::object({
            {"before", before}, {"after", after}})});
Json cursor = initial_cursor_row(*descriptor, 0);
        cursor["in_flight"] = true;
        cursor["tile_offset"] = 1;
        cursor["next_store_id"] = 54;
Json active = Json::object({
            {"store_id", 52}, {"tile_ordinal", 1}, {"shard_ordinal", 0},
            {"endpoint", "data_send"}, {"transfer_id", 3000},
            {"flow_id", nullptr}, {"network_done", false},
            {"hbm_done", false}, {"owner", owner}});
        cursor["active_endpoints"] = Json::array({active});
        delta["pipeline_transition_runs"] = Json::array({run});
        delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 1},
            {"retired", false},
            {"runs", Json::array({Json::object({
                {"run_index", 0}, {"value", cursor}})})}})});
harness.apply(delta);

const Json snapshot = harness.pipeline.snapshot_state_json();

CHECK(snapshot["credits"].size() == 1);

CHECK(snapshot["credits"][0]["available"] == 1);
    }
    // 34: credit balance mismatch rejected.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
Json run = base_pipeline_run(segment, 2, "tile_issue", 0, 0, 1);
        bind_selector(run, *descriptor, 0, 1, 0, "data_send");
const Json owner = owner_row(*descriptor, 0, 1);

Json slot = Json::object({
            {"buffer_domain_id", domain}, {"role", "source"}, {"rank", 0},
            {"shard_ordinal", 0}, {"slot_index", 0}, {"occupied", true},
            {"owner", owner}, {"consumed", false},
            {"target_shards_to_consume", Json::array()},
            {"consumed_target_shards", Json::array()}});
        run["slot_patches"] = Json::array({Json::object({
            {"before", nullptr}, {"after", slot}})});
Json credit = Json::object({
            {"buffer_domain_id", domain}, {"source_rank", 0},
            {"shard_ordinal", 0}, {"capacity", 2},
            {"available", 2},   // should be 1: one occupied source slot
            {"outstanding_ack_count", 0}});
        run["credit_patches"] = Json::array({Json::object({
            {"before", nullptr}, {"after", credit}})});
Json before = Json::object({
            {"obligation_id", segment + "/run/0"},
            {"segment_id", segment}, {"rank", 0}, {"first_store_id", 52},
            {"remaining_endpoint_count", 3}, {"state", "pending"}});
Json after = before;
        after["state"] = "active";
        run["obligation_patches"] = Json::array({Json::object({
            {"before", before}, {"after", after}})});
Json cursor = initial_cursor_row(*descriptor, 0);
        cursor["in_flight"] = true;
        cursor["tile_offset"] = 1;
        cursor["next_store_id"] = 54;
Json active = Json::object({
            {"store_id", 52}, {"tile_ordinal", 1}, {"shard_ordinal", 0},
            {"endpoint", "data_send"}, {"transfer_id", 3100},
            {"flow_id", nullptr}, {"network_done", false},
            {"hbm_done", false}, {"owner", owner}});
        cursor["active_endpoints"] = Json::array({active});
        delta["pipeline_transition_runs"] = Json::array({run});
        delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 1},
            {"retired", false},
            {"runs", Json::array({Json::object({
                {"run_index", 0}, {"value", cursor}})})}})});
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "source credit balance does not match occupied slots");
    }
    // 35: occupied slot cleared outside a legal release.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

harness.apply(issue_target_tile1(descriptor, 1, 1, 1));

Json delta = empty_envelope("delta", 2, 2, 2);
        delta["logical_seq_start"] = 3;
        delta["logical_seq_end"] = 4;
        delta["logical_event_count"] = 2;
        delta["result_state_version"] = 3;
        // Claim a SOURCE slot first (run 0, tile 1), then try clearing it
        // on an unrelated consume_start transition for run 1.
Json claim = base_pipeline_run(segment, 3, "tile_issue", 0, 1, 2);
        bind_selector(claim, *descriptor, 0, 1, 0, "data_send");
const Json owner = owner_row(*descriptor, 0, 1);

Json claimed = Json::object({
            {"buffer_domain_id", domain}, {"role", "source"}, {"rank", 0},
            {"shard_ordinal", 0}, {"slot_index", 0},
            {"occupied", true}, {"owner", owner}, {"consumed", false},
            {"target_shards_to_consume", Json::array()},
            {"consumed_target_shards", Json::array()}});
        claim["slot_patches"] = Json::array({Json::object({
            {"before", nullptr}, {"after", claimed}})});
Json credit = Json::object({
            {"buffer_domain_id", domain}, {"source_rank", 0},
            {"shard_ordinal", 0}, {"capacity", 2}, {"available", 1},
            {"outstanding_ack_count", 0}});
        claim["credit_patches"] = Json::array({Json::object({
            {"before", nullptr}, {"after", credit}})});
Json obl0_before = Json::object({
            {"obligation_id", segment + "/run/0"},
            {"segment_id", segment}, {"rank", 0}, {"first_store_id", 52},
            {"remaining_endpoint_count", 3}, {"state", "pending"}});
Json obl0_after = obl0_before;
        obl0_after["state"] = "active";
        claim["obligation_patches"] = Json::array({Json::object({
            {"before", obl0_before}, {"after", obl0_after}})});
Json cursor0 = initial_cursor_row(*descriptor, 0);
        cursor0["in_flight"] = true;
        cursor0["tile_offset"] = 1;
        cursor0["next_store_id"] = 54;
Json active0 = Json::object({
            {"store_id", 52}, {"tile_ordinal", 1}, {"shard_ordinal", 0},
            {"endpoint", "data_send"}, {"transfer_id", 3200},
            {"flow_id", nullptr}, {"network_done", false},
            {"hbm_done", false}, {"owner", owner}});
        cursor0["active_endpoints"] = Json::array({active0});
Json clear = base_pipeline_run(segment, 4, "consume_start", 0, 2, 3);
        bind_selector(clear, *descriptor, 1, 2, 0, "data_recv");
        clear["slot_patches"] = Json::array({Json::object({
            {"before", claimed}, {"after", nullptr}})});
        delta["pipeline_transition_runs"] = Json::array({claim, clear});
        delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 3},
            {"retired", false},
            {"runs", Json::array({Json::object({
                {"run_index", 0}, {"value", cursor0}})})}})});
        // consume_start may not clear an occupied source slot
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "occupied pipeline slot was cleared before its legal release");
    }
    // 36: claim with the wrong descriptor membership.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        delta["pipeline_transition_runs"][0]["slot_patches"][0]
            ["operations"][0]["target_shards"] = Json::array();
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "target claim does not declare the exact descriptor membership");
    }
    // 37: target consume lacking terminal proof.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

harness.apply(issue_target_tile1(descriptor, 1, 1, 1));

Json delta = empty_envelope("delta", 2, 2, 2);
        delta["logical_seq_start"] = 3;
        delta["logical_seq_end"] = 3;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 3;
Json run = base_pipeline_run(segment, 3, "consume_done", 0, 1, 2);
        (void)0;
        run["slot_patches"] = Json::array({target_delta(
            domain, 1, 0,
            /*before_generation=*/1, /*predecessor_seq=*/2,
            /*predecessor_count=*/0,
Json::array({Json::object({
                {"op", "consume"}, {"slot_generation", 1},
                {"cause", Json::object({
                    {"terminal_rank", 1},
                    {"terminal_store_id", 71},
                    {"consume_run_index", 0}})},
                {"add_shards", Json::array({0})},
                {"cleared", true}})}))});
        delta["pipeline_transition_runs"] = Json::array({run});
        // No node completion event and no endpoint_done preceded the consume:
        // the cause lacks prior terminal proof.
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "target consume cause lacks prior terminal proof");
    }
    // 38: lineage mismatch on the second target delta.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

harness.apply(issue_target_tile1(descriptor, 1, 1, 1));

Json delta = issue_target_tile1(descriptor, 2, 2, 2);
        bind_selector(delta["pipeline_transition_runs"][0], *descriptor, 1,
                      /*tile=*/2, 0, "data_recv");   // position 1: legal
Json& run = delta["pipeline_transition_runs"][0];
        run["slot_patches"][0]["key"]["slot_index"] = 1;
        run["slot_patches"][0]["operations"][0]["owner"]["tile_ordinal"] = 2;
        // slot 1 lineage is (0, 1, 0); declare a stale generation instead
        run["slot_patches"][0]["before_slot_generation"] = 5;
        delta["pipeline_transition_runs"][0]["obligation_patches"] =
Json::array();   // active row already matches progress
        delta["cursor_patches"][0]["runs"] = Json::array();
        delta["pipeline_transition_runs"][0]["cursor_version_start"] = 1;
        delta["pipeline_transition_runs"][0]["cursor_version_end"] = 1;
        delta["cursor_patches"][0]["cursor_version"] = 1;
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "target delta predecessor lineage differs from accepted state");
    }
    // 39: obligation patch disagreeing with derived progress.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        delta["pipeline_transition_runs"][0]["obligation_patches"][0]["after"]
            ["remaining_endpoint_count"] = 1;
CHECK_THROW_STATEMENT(

harness.apply(delta),
            "obligation patch does not match endpoint completion/close progress");
    }
}

static void case_close_retire_resources() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    const std::string domain = descriptor->identity.buffer_domain_id;
    // 40: request_close repeated.
    // 41: request_close -> settle_done -> descriptor_retire lifecycle.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));        harness.apply(issue_target_tile1(descriptor, 1, 1, 1));

        // completion + data_ready + descriptor-level consume_done in one
        // delivery, with the terminal node completion event interleaved.
Json commit_node = Json::object({
            {"logical_seq", 3}, {"tick", 0}, {"kind", "commit"},
            {"work_id", "req#decode:1:r1n71"}, {"payload", Json::object()}});
Json node = Json::object({
            {"logical_seq", 4}, {"tick", 0}, {"kind", "completion"},
            {"work_id", "req#decode:1:r1n71"},
            {"payload", Json::object({{"rank", 1}, {"node_id", 71}})}});
Json delta = empty_envelope("delta", 2, 2, 2);
        delta["logical_seq_start"] = 3;
        delta["logical_seq_end"] = 7;
        delta["logical_event_count"] = 5;
        delta["result_state_version"] = 3;
Json done = base_pipeline_run(segment, 5, "endpoint_done", 0, 1, 2);
        bind_selector(done, *descriptor, 1, 1, 0, "data_recv");
Json ready = base_pipeline_run(segment, 6, "data_ready", 0, 2, 2);
        bind_selector(ready, *descriptor, 1, 1, 0, "data_recv");
Json consume = base_pipeline_run(segment, 7, "consume_done", 0, 2, 2);
        consume["slot_patches"] = Json::array({target_delta(
            domain, 1, 0,
            /*before_generation=*/1, /*predecessor_seq=*/2,
            /*predecessor_count=*/0,
Json::array({Json::object({
                {"op", "consume"}, {"slot_generation", 1},
                {"cause", Json::object({
                    {"terminal_rank", 1},
                    {"terminal_store_id", 71},
                    {"consume_run_index", 0}})},
                {"add_shards", Json::array({0})},
                {"cleared", true}})}))});
        delta["node_transitions"] = Json::array({commit_node, node});
        // obligation: active(3, first 55) -> pending(2, first 57)
Json obl_before = Json::object({
            {"obligation_id", segment + "/run/1"},
            {"segment_id", segment}, {"rank", 1}, {"first_store_id", 55},
            {"remaining_endpoint_count", 3}, {"state", "active"}});
Json obl_after = obl_before;
        obl_after["first_store_id"] = 57;
        obl_after["remaining_endpoint_count"] = 2;
        obl_after["state"] = "pending";
        done["obligation_patches"] = Json::array({Json::object({
            {"before", obl_before}, {"after", obl_after}})});
        delta["pipeline_transition_runs"] =
Json::array({done, ready, consume});

Json cursor = initial_cursor_row(*descriptor, 1);
        cursor["completed_endpoint_count"] = 1;
        cursor["remaining_endpoint_count"] = 2;
        cursor["in_flight"] = false;
        cursor["tile_offset"] = 1;
        cursor["next_store_id"] = 57;
        delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 2},
            {"retired", false},
            {"runs", Json::array({Json::object({
                {"run_index", 1}, {"value", cursor}})})}})});
harness.apply(delta);

const Json mid = harness.pipeline.snapshot_state_json();
        bool slot_empty = false;
        for (const Json& slot : mid["slots"]) {
            if (slot["slot_index"] == 0) {
                slot_empty = slot["occupied"] == false;
            }
        }
CHECK(slot_empty);   // consume cleared the target row

        // request_close cancels the unissued tail and tombstones obligations.
Json close_delta = empty_envelope("delta", 3, 7, 3);
        close_delta["logical_seq_start"] = 8;
        close_delta["logical_seq_end"] = 8;
        close_delta["logical_event_count"] = 1;
        close_delta["result_state_version"] = 4;
Json close = base_pipeline_run(segment, 8, "request_close", 0, 2, 3);

Json obl1_before = Json::object({
            {"obligation_id", segment + "/run/0"},
            {"segment_id", segment}, {"rank", 0}, {"first_store_id", 52},
            {"remaining_endpoint_count", 3}, {"state", "pending"}});
Json obl1_after2 = Json::object({
            {"obligation_id", segment + "/run/1"},
            {"segment_id", segment}, {"rank", 1}, {"first_store_id", 57},
            {"remaining_endpoint_count", 2}, {"state", "pending"}});
        close["obligation_patches"] = Json::array({
Json::object({{"before", obl1_before}, {"after", nullptr}}),

Json::object({{"before", obl1_after2}, {"after", nullptr}})});

Json cursor0 = initial_cursor_row(*descriptor, 0);
        cursor0["cancelled_endpoint_count"] = 3;
        cursor0["remaining_endpoint_count"] = 0;
        cursor0["tile_offset"] = 3;
        cursor0["next_store_id"] = nullptr;
Json cursor1 = cursor;
        cursor1["cancelled_endpoint_count"] = 2;
        cursor1["remaining_endpoint_count"] = 0;
        cursor1["tile_offset"] = 3;
        cursor1["next_store_id"] = nullptr;
        close_delta["pipeline_transition_runs"] = Json::array({close});
        close_delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 3},
            {"retired", false},
            {"runs", Json::array({
Json::object({{"run_index", 0}, {"value", cursor0}}),

Json::object({{"run_index", 1}, {"value", cursor1}})})}})});        harness.apply(close_delta);

const Json closed = harness.pipeline.snapshot_state_json();

CHECK(closed["obligations"].empty());

        // request_close repeated -> rejected.
        {
Json repeat = empty_envelope("delta", 4, 8, 4);
            repeat["logical_seq_start"] = 9;
            repeat["logical_seq_end"] = 9;
            repeat["logical_event_count"] = 1;
            repeat["result_state_version"] = 5;
Json again = base_pipeline_run(segment, 9, "request_close", 0, 3, 4);
            repeat["pipeline_transition_runs"] = Json::array({again});
CHECK_THROW_STATEMENT(harness.apply(repeat),
                                  "request_close transition repeated");
        }

        // settle_done then descriptor_retire with tombstones and erases.
Json settle = empty_envelope("delta", 4, 8, 4);
        settle["logical_seq_start"] = 9;
        settle["logical_seq_end"] = 9;
        settle["logical_event_count"] = 1;
        settle["result_state_version"] = 5;
        settle["pipeline_transition_runs"] = Json::array({
            base_pipeline_run(segment, 9, "settle_done", 0, 3, 4)});
        settle["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 4},
            {"retired", false}, {"runs", Json::array()}})});
harness.apply(settle);

Json retire = empty_envelope("delta", 5, 9, 5);
        retire["logical_seq_start"] = 10;
        retire["logical_seq_end"] = 10;
        retire["logical_event_count"] = 1;
        retire["result_state_version"] = 6;
Json retire_run = base_pipeline_run(segment, 10, "descriptor_retire",
                                            0, 4, 5);
        retire_run["slot_patches"] = Json::array({
            target_delta(domain, 1, 0, 1, 7, 0,
Json::array({Json::object({
                             {"op", "erase"}, {"slot_generation", 1}})})),
            target_delta(domain, 1, 1, 0, 1, 0,
Json::array({Json::object({
                             {"op", "erase"}, {"slot_generation", 0}})}))});
        retire["pipeline_transition_runs"] = Json::array({retire_run});
        retire["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 5},
            {"retired", true},
            {"runs", Json::array({
Json::object({{"run_index", 0}, {"value", nullptr}}),

Json::object({{"run_index", 1}, {"value", nullptr}})})}})});        harness.apply(retire);

const Json retired = harness.pipeline.snapshot_state_json();

CHECK(retired["descriptors"].empty());

CHECK(retired["slots"].empty());

CHECK(retired["authoritative_cursors"].empty());

CHECK(retired["obligations"].empty());
        // the retained registry still keeps the immutable descriptor
CHECK(harness.pipeline.descriptors().count(segment) == 1);
    }
    // 42: descriptor_retire before service/settle.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 1;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["pipeline_transition_runs"] = Json::array({
            base_pipeline_run(segment, 2, "descriptor_retire", 0, 0, 1)});
CHECK_THROW_STATEMENT(            harness.apply(delta),
            "descriptor retirement precedes service and settle");
    }
    // 43: settle_done before service_done.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 1;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["pipeline_transition_runs"] = Json::array({
            base_pipeline_run(segment, 2, "settle_done", 0, 0, 1)});
CHECK_THROW_STATEMENT(            harness.apply(delta),
            "settle_done precedes service_done/request_close");
    }
    // 44: cursor tombstones without a retire transition.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = empty_envelope("delta", 1, 1, 1);
        delta["logical_seq_end"] = 2;
        delta["logical_event_count"] = 1;
        delta["result_state_version"] = 2;
Json run = base_pipeline_run(segment, 2, "service_done", 0, 0, 1);
        delta["pipeline_transition_runs"] = Json::array({run});
        delta["cursor_patches"] = Json::array({Json::object({
            {"segment_id", segment}, {"cursor_version", 1},
            {"retired", true},
            {"runs", Json::array({
Json::object({{"run_index", 0}, {"value", nullptr}}),

Json::object({{"run_index", 1}, {"value", nullptr}})})}})});
        // parse accepts retirement tombstones only after validation; the
        // lifecycle marker check rejects them at staging.
        try {
harness.apply(delta);

CHECK(false);
        } catch (const LedgerError&) {
CHECK(true);
        }
    }
    // 45+46: resource rows accepted/replaced; as_of_tick must match.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0, 3));

Json delta = empty_envelope("delta", 1, 0, 0, 4);
        delta["result_state_version"] = 1;
Json noc = Json::object({
            {"resource_type", "noc_flow"},
            {"work_id", "noc#:0:r0n7@noc:7"},
            {"request_id", "noc"}, {"stage", ""}, {"generation", 0},
            {"rank", 0}, {"node_id", 7}, {"operation", "xfer"},
            {"leg", "noc:7"}, {"state", "Active"},
            {"total", Json{{"ops", nullptr}, {"bytes", 10},
                           {"latency_ns", nullptr}}},
            {"remaining", Json{{"ops", nullptr}, {"bytes", 10},
                               {"latency_ns", nullptr}}},
            {"progress_source", "authoritative"}, {"service_unit", "bytes"},
            {"route", Json::array({0, 1})}, {"as_of_tick", 4},
            {"issued_tick", nullptr}, {"descriptor_segment_id", nullptr},
            {"endpoint_selector", nullptr},
            {"flow_id", 7}, {"current_rate_Bpns", 1.5},
            {"last_rate_update_time", 0.0}});
        delta["active_resources"] = Json::array({noc});
harness.apply(delta);

Json pricing = harness.pipeline.pricing_json();

CHECK(pricing["active_resources"].size() == 1);

CHECK(pricing["active_resources"][0]["flow_id"] == 7);
        // replacement with a smaller remaining set
Json next = delta;
        next["delivery_sequence"] = 2;
        next["replay_token"] = "cpp_online/test-run/2";
        next["base_watermark"] = 0;
        next["base_state_version"] = 1;
        next["result_state_version"] = 2;
        next["tick"] = 5;
        next["active_resources"][0]["as_of_tick"] = 5;
        next["active_resources"][0]["remaining"]["bytes"] = 4;
harness.apply(next);

CHECK(harness.pipeline.pricing_json()["active_resources"][0]
                       ["remaining"]["bytes"] == 4);
        // as_of_tick off the envelope tick is rejected.
Json bad = next;
        bad["delivery_sequence"] = 3;
        bad["replay_token"] = "cpp_online/test-run/3";
        bad["base_state_version"] = 2;
        bad["result_state_version"] = 3;
        bad["active_resources"][0]["as_of_tick"] = 99;
CHECK_THROW_STATEMENT(            harness.apply(bad),
            "active resource frame must match envelope tick");
    }
}

static void case_dependency_and_atomicity() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    // 47: dependency pin created on commit, released on completion.
    {
        LedgerHarness harness;
        // ordinary node work on rank 1 depending on the pipeline endpoint
        // node id 55 (data_recv tile 1 -> endpoint id 54 -> store 55).
Json commit = Json::object({
            {"logical_seq", 2}, {"tick", 0}, {"kind", "commit"},
            {"work_id", "req#decode:1:r1n80"},
            {"payload", Json::object({
                {"dependencies", Json::array({"req#decode:1:r1n55"})}})}});
Json envelope = full_with_descriptor(
            descriptor, Json::array({commit}));
harness.apply(envelope);

        auto build_close_settle = [&]() -> Json {
Json chain = Json::array();

Json close = empty_envelope("delta", 1, 2, 1);
            close["logical_seq_start"] = 3;
            close["logical_seq_end"] = 3;
            close["logical_event_count"] = 1;
            close["result_state_version"] = 2;
Json close_run =
                base_pipeline_run(segment, 3, "request_close", 0, 0, 1);
            close_run["obligation_patches"] = Json::array({
Json::object({
                    {"before", Json::object({
                        {"obligation_id", segment + "/run/0"},
                        {"segment_id", segment}, {"rank", 0},
                        {"first_store_id", 52},
                        {"remaining_endpoint_count", 3},
                        {"state", "pending"}})},
                    {"after", nullptr}}),
Json::object({
                    {"before", Json::object({
                        {"obligation_id", segment + "/run/1"},
                        {"segment_id", segment}, {"rank", 1},
                        {"first_store_id", 55},
                        {"remaining_endpoint_count", 3},
                        {"state", "pending"}})},
                    {"after", nullptr}})});
Json cursor0 = initial_cursor_row(*descriptor, 0);
            cursor0["cancelled_endpoint_count"] = 3;
            cursor0["remaining_endpoint_count"] = 0;
            cursor0["tile_offset"] = 3;
            cursor0["next_store_id"] = nullptr;
Json cursor1 = initial_cursor_row(*descriptor, 1);
            cursor1["cancelled_endpoint_count"] = 3;
            cursor1["remaining_endpoint_count"] = 0;
            cursor1["tile_offset"] = 3;
            cursor1["next_store_id"] = nullptr;
            close["pipeline_transition_runs"] = Json::array({close_run});
            close["cursor_patches"] = Json::array({Json::object({
                {"segment_id", segment}, {"cursor_version", 1},
                {"retired", false},
                {"runs", Json::array({
Json::object({{"run_index", 0}, {"value", cursor0}}),

Json::object({{"run_index", 1}, {"value", cursor1}})})}})});
            chain.push_back(close);
Json settle = empty_envelope("delta", 2, 3, 2);
            settle["logical_seq_start"] = 4;
            settle["logical_seq_end"] = 4;
            settle["logical_event_count"] = 1;
            settle["result_state_version"] = 3;
            settle["pipeline_transition_runs"] = Json::array({
                base_pipeline_run(segment, 4, "settle_done", 0, 1, 2)});
            settle["cursor_patches"] = Json::array({Json::object({
                {"segment_id", segment}, {"cursor_version", 2},
                {"retired", false}, {"runs", Json::array()}})});
            chain.push_back(settle);
            return chain;
        };        for (const Json& step : build_close_settle()) {
harness.apply(step);
        }

        // A retire attempt now fails: the committed work still waits on the
        // descriptor's endpoint, so the dependency pin is unresolved.
        auto retire_envelope = [&](std::uint64_t seq_base,
const Json& extra_nodes) {

Json retire = empty_envelope("delta", 3, 4, 3);
            retire["node_transitions"] = extra_nodes;
            retire["logical_seq_start"] = seq_base;
            retire["logical_seq_end"] = seq_base + extra_nodes.size();
            retire["logical_event_count"] = 1 + extra_nodes.size();
            retire["result_state_version"] = 4;
Json retire_run = base_pipeline_run(
                segment, seq_base + extra_nodes.size(), "descriptor_retire", 0,
                2, 3);
            retire_run["slot_patches"] = Json::array({
                target_delta(descriptor->identity.buffer_domain_id, 1, 0,
                             0, 1, 0,
Json::array({Json::object({
                                 {"op", "erase"},
                                 {"slot_generation", 0}})})),
                target_delta(descriptor->identity.buffer_domain_id, 1, 1,
                             0, 1, 0,
Json::array({Json::object({
                                 {"op", "erase"},
                                 {"slot_generation", 0}})}))});
            retire["pipeline_transition_runs"] = Json::array({retire_run});
            retire["cursor_patches"] = Json::array({Json::object({
                {"segment_id", segment}, {"cursor_version", 3},
                {"retired", true},
                {"runs", Json::array({
Json::object({{"run_index", 0}, {"value", nullptr}}),

Json::object({{"run_index", 1}, {"value", nullptr}})})}})});
            return retire;
        };
CHECK_THROW_STATEMENT(            harness.apply(retire_envelope(5, Json::array())),
            "descriptor retirement retains an unfinished ordinary dependency pin");
const Json before =
            harness.pipeline.snapshot_state_json();
        // Completing the dependent work releases the pin in the same delivery;
        // the interleaved retire then succeeds.
Json completion = Json::object({
            {"logical_seq", 5}, {"tick", 0}, {"kind", "completion"},
            {"work_id", "req#decode:1:r1n80"}, {"payload", Json::object()}});
const Json released =
            retire_envelope(5, Json::array({completion}));
harness.apply(released);

const Json after = harness.pipeline.snapshot_state_json();

CHECK(after["descriptors"].empty());

CHECK(after != before);

CHECK(harness.pipeline.snapshot_state_json()["slots"].empty());
    }
    // 48: dependency referencing an uncommitted descriptor.
    {
        LedgerHarness harness;
Json commit = Json::object({
            {"logical_seq", 2}, {"tick", 0}, {"kind", "commit"},
            {"work_id", "req#decode:1:r1n80"},
            {"payload", Json::object({
                {"dependencies",
Json::array({"other#decode:1:r1n55"})}})}});
        const DescriptorPtr descriptor_b = make_descriptor();
Json envelope = full_with_descriptor(
            descriptor_b, Json::array({commit}));
        // The dependency prefix "other#decode:1:r1n" is not in any index, so
        // dependency_reference returns none and the commit is accepted.
        const PipelineReceipt receipt = harness.apply(envelope);
CHECK(receipt.state_version == 1);
    }
    // 49: atomicity: a failed prepare leaves the accepted state unchanged.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

const Json before = harness.pipeline.snapshot_state_json();

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        delta["pipeline_transition_runs"][0]["slot_patches"][0]
            ["operations"][0]["target_shards"] = Json::array();
        try {
harness.apply(delta);

CHECK(false);
        } catch (const LedgerError&) {
CHECK(true);
        }
CHECK(harness.pipeline.snapshot_state_json() == before);
        // A subsequent valid delivery still applies.
harness.apply(issue_target_tile1(descriptor, 1, 1, 1));

CHECK(harness.pipeline.snapshot_state_json() != before);
    }
    // 50: discard_prepared frees the slot; the next delivery works.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

const Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        ParsedEnvelope parsed = harness.parse(delta);
        const std::string text = delta.dump();
        auto tx = harness.work.begin_protocol_v3_transaction(
            parsed.header.delivery_sequence, parsed.header.tick, "delta");
        const int slot = harness.pipeline.prepare(parsed, tx, text);
        harness.pipeline.discard_prepared(slot);
CHECK(harness.pipeline.delivery_sequence() == 0);
        // unknown slot commit/discard are rejected
CHECK_THROW_STATEMENT(harness.pipeline.commit_prepared(slot),
                              "unknown prepared delivery slot");
harness.apply(delta);

CHECK(harness.pipeline.delivery_sequence() == 1);
    }
}

// Empty-envelope full chain (integration repro shape): a 19-key v4 frame
// with every section empty; zero visible change keeps result_state_version
// at base, and both ledger loads advance exactly once.
static void case_empty_full_chain() {
    const char* full_json =
        R"JSON({"producer":"cpp_online","schema_version":4,"protocol_version":4,"bridge_protocol":"bridge-v4-20261011","work_ledger_schema":"workledger-v4-20261011","epoch":"e1","delivery_sequence":0,"base_watermark":0,"logical_seq_start":1,"logical_seq_end":0,"logical_event_count":0,"replay_token":"cpp_online/e1/0","descriptor_version":1,"base_state_version":0,"result_state_version":0,"snapshot_kind":"full","tick":5,"descriptor_commits":[],"node_transitions":[],"pipeline_transition_runs":[],"cursor_patches":[],"active_resources":[]})JSON";
    const char* delta_json =
        R"JSON({"producer":"cpp_online","schema_version":4,"protocol_version":4,"bridge_protocol":"bridge-v4-20261011","work_ledger_schema":"workledger-v4-20261011","epoch":"e1","delivery_sequence":1,"base_watermark":0,"logical_seq_start":1,"logical_seq_end":0,"logical_event_count":0,"replay_token":"cpp_online/e1/1","descriptor_version":1,"base_state_version":0,"result_state_version":0,"snapshot_kind":"delta","tick":6,"descriptor_commits":[],"node_transitions":[],"pipeline_transition_runs":[],"cursor_patches":[],"active_resources":[]})JSON";
    // The exact integration sequence: parse -> begin -> prepare -> work
    // commit -> pipeline commit, for the empty full frame then the empty
    // delta frame.
    {
        LedgerHarness harness;
        const std::string full_text(full_json);
        ParsedEnvelope parsed = parse_envelope_v4(
            full_text.data(), full_text.size(), RetainedRegistry{}, false, 0, 5);
        auto tx = harness.work.begin_protocol_v3_transaction(
            parsed.header.delivery_sequence, parsed.header.tick, "full");
        const int slot =
            harness.pipeline.prepare(parsed, tx, full_text);
        CHECK(slot == 0);
        harness.work.tx_commit(tx);
        const PipelineReceipt r0 = harness.pipeline.commit_prepared(slot);
        CHECK(r0.snapshot_kind == "full");
        CHECK(r0.state_version == 0);   // zero visible change
        CHECK(!r0.replayed);

        const std::string delta_text(delta_json);
        RetainedRegistry retained;
        retained.descriptors = &harness.pipeline.descriptors();
        retained.selector_indexes = &harness.pipeline.selector_indexes();
        ParsedEnvelope parsed2 = parse_envelope_v4(
            delta_text.data(), delta_text.size(), retained, false, 1, 6);
        auto tx2 = harness.work.begin_protocol_v3_transaction(
            parsed2.header.delivery_sequence, parsed2.header.tick, "delta");
        const int slot2 = harness.pipeline.prepare(parsed2, tx2, delta_text);
        harness.work.tx_commit(tx2);
        const PipelineReceipt r1 = harness.pipeline.commit_prepared(slot2);
        CHECK(r1.snapshot_kind == "delta");
        CHECK(r1.state_version == 0);

        CHECK(harness.work.full_load_count() == 1);
        CHECK(harness.work.delta_load_count() == 1);
        CHECK(harness.pipeline.delivery_sequence() == 1);
        CHECK(harness.pipeline.watermark() == 0);
        CHECK(harness.pipeline.last_tick() == 6);
        CHECK(harness.pipeline.pricing_json()["active_resources"].empty());
        const Json snapshot = harness.pipeline.snapshot_state_json();
        CHECK(snapshot["epoch"] == "e1");
        CHECK(snapshot["descriptors"].empty());
        CHECK(snapshot["authoritative_cursors"].empty());
    }
    // The same empty frames through the harness's wire-JSON path.
    {
        LedgerHarness harness;
        harness.apply(Json::parse(std::string(full_json)));
        harness.apply(Json::parse(std::string(delta_json)));
        CHECK(harness.work.full_load_count() == 1);
        CHECK(harness.work.delta_load_count() == 1);
    }
}

// Direct-construction cases exercising parser-independent paths.
static void case_direct_construction() {
    const DescriptorPtr descriptor = make_descriptor();
    const std::string segment = descriptor->identity.segment_id;
    // 51: replay gate compares the raw segment bytes of THIS delivery.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));
        ParsedEnvelope parsed = harness.parse(
            full_with_descriptor(descriptor));
        WorkLedgerCore::Transaction tx;
        // Same parsed DTO, different raw bytes -> fail closed.
CHECK_THROW_STATEMENT(
            harness.pipeline.prepare(parsed, tx, std::string("{\"x\":1}")),
            "typed-different canonical content");
    }
    // 52: malformed source slot patch quad rejected by the row readers.
    {
        LedgerHarness harness;
harness.apply(full_with_descriptor(descriptor));

Json delta = issue_target_tile1(descriptor, 1, 1, 1);
        ParsedEnvelope parsed = harness.parse(delta);
        // Corrupt the parsed transition's slot-patch form directly.
        parsed.pipeline_runs[0]["slot_patches"][0] =
Json::array({"source", Json::array({1, 2, 3}), nullptr, nullptr});
        auto tx = harness.work.begin_protocol_v3_transaction(
            parsed.header.delivery_sequence, parsed.header.tick, "delta");
CHECK_THROW_STATEMENT(
            harness.pipeline.prepare(parsed, tx, delta.dump()),
            "slot key must have five fields");
    }
    // 53: an empty full is idempotent-zero-change even after prepare+discard.
    {
        LedgerHarness harness;
harness.apply(empty_envelope("full", 0, 0, 0));

const Json delta = empty_envelope("delta", 1, 0, 0);
        ParsedEnvelope parsed = harness.parse(delta);
        auto tx = harness.work.begin_protocol_v3_transaction(
            1, 0, "delta");
        const int slot = harness.pipeline.prepare(parsed, tx, delta.dump());
        harness.pipeline.discard_prepared(slot);
        // The discarded delivery never happened: sequence 1 is still next.
        const PipelineReceipt receipt = harness.apply(delta);
CHECK(!receipt.replayed);

CHECK(harness.pipeline.delivery_sequence() == 1);
    }
}

static void run_case(const char* name, void (*fn)()) {
    try {
        fn();
    } catch (const LedgerError& error) {
        std::fprintf(stderr, "FAIL case %s: uncaught LedgerError: %s\n",
                     name, error.what());
        ++failures;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL case %s: uncaught exception: %s\n",
                     name, error.what());
        ++failures;
    }
}

int main() {
    run_case("gates_and_replay", case_gates_and_replay);
    run_case("merge_walk", case_merge_walk);
    run_case("descriptor_lifecycle", case_descriptor_lifecycle);
    run_case("cursor_simulation", case_cursor_simulation);
    run_case("slots_credits_obligations", case_slots_credits_obligations);
    run_case("close_retire_resources", case_close_retire_resources);
    run_case("dependency_and_atomicity", case_dependency_and_atomicity);
    run_case("empty_full_chain", case_empty_full_chain);
    run_case("direct_construction", case_direct_construction);
    if (failures == 0) {
        std::printf("dual_ledger_test PASS\n");
        return 0;
    }
    std::printf("dual_ledger_test FAIL (%d)\n", failures);
    return 1;
}
