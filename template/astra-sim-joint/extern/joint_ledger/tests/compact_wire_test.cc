// compact_wire_test.cc -- workledger-compact-v1 encode/decode round trip and
// the structural scan of a compact request member (v6 round).
//
// 1. encode_compact_frame is deterministic (identical logical input -> byte
//    identical output, the idempotent-replay gate contract);
// 2. parse_envelope_compact validates and reconstructs the typed envelope of
//    a synthetic-but-legal frame and rejects corrupted variants;
// 3. scan_request splices a compact member out of a request file body.
#include "joint_ledger/CompactReader.hh"
#include "joint_ledger/CompactWire.hh"
#include "joint_ledger/JsonScanner.hh"
#include "joint_ledger/PipelineRows.hh"

#include <cstdio>
#include <cstring>
#include <string>

using namespace jointledger;

namespace {

int failures = 0;

void check(bool ok, const char* what) {
    if (!ok) {
        std::printf("compact_wire_test FAIL: %s\n", what);
        ++failures;
    }
}

const char* const kSegment = "pp1/726571/decode/1/2/1/0";
const char* const kDomain = "pb1/726571/1/9";

const char* kDescriptorWire = R"({"descriptor_version":1,"identity":{"request_id":"req","stage":"decode","generation":1,"stage_epoch":2,"segment_ordinal":0,"rank":1,"staging_allocation_id":9,"segment_id":"pp1/726571/decode/1/2/1/0","buffer_domain_id":"pb1/726571/1/9"},"stream_kind":"decode_remote_credit","groups":[{"group_ordinal":0,"pass_or_window_ordinal":0,"source_ordinal":0,"source_key":"source-0","layer_start":0,"layer_end":1,"tile_axis":{"unit":"stream_bytes","total_units":5,"units_per_tile":2,"tile_count":3,"tail_units":1,"global_tile_first":1},"shards":[{"shard_ordinal":0,"source_rank":0,"target_rank":1,"total_data_bytes":5,"bytes_per_unit":1,"noc_path":[0,1]}]}],"protocol":{"buffer_slots":2,"release_lag_tiles":2,"initial_tiles_without_release":2,"fill_ack_bytes":1,"release_ack_bytes":1,"hbm_charge":true,"target_rank_order":"release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1","source_rank_order":"release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i","tile_branching":"tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"},"prediction_profile":{"prefill_consume":"zero_duration_no_extra_consume","decode_consume":{"kind":"profiled_consume","profile_id":"decode-service-profile-v1","compute_ns_total":240.0,"hbm_bytes_total":5,"resource_key":"hbm:1","share_rule":"existing_action_work_graph_byte_share_vector"}},"execution_profile":{"data_legs":"ordinary_comm_send_recv_path","endpoint_hbm_rule":"one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint","prefill_consume":"existing_body_or_marker_only","decode_consume":"existing_body_nodes_only"},"logical_id_ranges":[{"rank":0,"first_json_id":53,"count":2,"stride":2,"kind":"virtual_endpoint_hole"},{"rank":1,"first_json_id":56,"count":2,"stride":2,"kind":"virtual_endpoint_hole"}],"tag_allocation_runs":[{"allocator_scope_id":"builder-0","emission_ordinal":7,"first_tag":101,"count":3,"group_ordinal":0,"first_tile":1,"first_shard":0,"tile_count":3,"shard_count":1}],"endpoint_runs":[{"rank":0,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"first_json_id":51,"tile_stride":2,"shard_stride":1,"initial_parent_json_id":50,"legacy_context":{"request_id":"req","stage":"decode","generation":1},"legs":[{"endpoint":"data_send","json_offset":0,"tag_emission_ordinal":7,"tag_offset":0,"tag_tile_stride":1,"tag_shard_stride":1}]},{"rank":1,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"first_json_id":54,"tile_stride":2,"shard_stride":1,"initial_parent_json_id":53,"legacy_context":{"request_id":"req","stage":"decode","generation":1},"legs":[{"endpoint":"data_recv","json_offset":0,"tag_emission_ordinal":7,"tag_offset":0,"tag_tile_stride":1,"tag_shard_stride":1}]}],"endpoint_dependency_runs":[{"rank":0,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"endpoint":"data_send","dependency_kind":"ordinary_node_terminal","first_dependency_json_id":50,"dependency_tile_stride":0,"dependency_shard_stride":0}],"consume_completion_runs":[{"rank":1,"group_ordinal":0,"first_tile":1,"tile_count":3,"shard_ordinals":[0],"dependency_kind":"ordinary_node_terminal","first_dependency_json_id":70,"dependency_tile_stride":0,"dependency_shard_stride":0}],"projection":{"version":1,"chains":[{"group_ordinal":0,"rank":1,"tile_count":3,"fill":{"leg_id_prefix":"work#dcstream#R1#F","first_index":0,"fill_order_first":0,"fill_order_stride":1,"completion_order_first":null,"completion_order_stride":0,"route":["link:0"],"bytes_per_tile":2,"latency_ns":3,"start_deps":["work#start"],"gate_id":"work#gate:dcstream:R1","gate_capacity":2},"consume":{"leg_id_prefix":"work#dcstream#R1#C","first_index":0,"consume_order_first":3,"consume_order_stride":1,"completion_order_first":null,"completion_order_stride":0,"kind":"compute","route":["hbm:1"],"group":"decode","latency_first_ns":7,"latency_rest_ns":0,"consume_deps":["work#consume-start"],"consume_start_deps":[],"gate_id":"work#gate:dcstream:R1","compute_share":{"rule":"quotient_remainder_earliest_global_indices","total_units":8,"total_tiles":3,"share_index_first":0,"tile_bytes":2},"hbm_share":{"rule":"quotient_remainder_earliest_global_indices","total_units":5,"total_tiles":3,"share_index_first":0,"tile_bytes":2}}}]},"anchors":[{"role":"start","rank":0,"binding":"physical_endpoint_slot","node_id":51,"selector":{"kind":"tile_endpoint_leg","group_ordinal":0,"tile_ordinal":1,"shard_ordinal":0,"endpoint":"data_send","tag_allocation_emission_ordinal":7,"tag_offset":0}},{"role":"frontier","rank":1,"binding":"physical_endpoint_slot","node_id":54,"selector":{"kind":"tile_endpoint_terminal","group_ordinal":0,"tile_ordinal":1,"shard_ordinal":0,"endpoint":"data_recv","tag_allocation_emission_ordinal":7,"tag_offset":0}},{"role":"service","rank":1,"binding":"descriptor_cursor_selector","node_id":null,"selector":{"kind":"existing_service_gate","existing_gate":"decode_service_gate"}},{"role":"settle","rank":1,"binding":"descriptor_cursor_selector","node_id":null,"selector":{"kind":"all_obligations_settled"}}]})";

Json descriptor_wire() { return Json::parse(kDescriptorWire); }

DescriptorPtr make_descriptor() {
    return PipelineDescriptor::from_wire(descriptor_wire());
}

Json descriptor_commit_row(const DescriptorPtr& descriptor) {
    Json commit = Json::object();
    commit["identity"] = descriptor->to_wire().at("identity");
    commit["descriptor"] = descriptor->to_wire();
    commit["commit_tick"] = 7;
    commit["cursor_version"] = 0;
    commit["service_done"] = false;
    commit["settle_done"] = false;
    commit["retired"] = false;
    return commit;
}

CompactFrameInput sample_frame(const DescriptorPtr& descriptor) {
    CompactFrameInput frame;
    frame.full = true;
    frame.epoch = "compact-test-epoch";
    frame.delivery_sequence = 0;
    frame.tick = 9;
    frame.base_watermark = 0;
    frame.seq_start = 1;
    frame.seq_end = 2;
    frame.event_count = 2;
    frame.base_state_version = 0;
    frame.result_state_version = 1;
    frame.descriptor_commits_json.push_back(
        descriptor_commit_row(descriptor).dump());

    CompactNodeRecord node;
    node.kind = NodeKind::Commit;
    node.logical_seq = 1;
    node.tick = 5;
    node.work_id = "req#decode:1:r1n56";
    node.payload = Json{{"request_id", "req"}, {"rank", 1},
                        {"node_id", 56}};
    frame.nodes.push_back(std::move(node));

    // Endpoint run index 1 = rank-1 data_recv run (first_json_id 54, tile
    // stride 2, one shard, one leg): the selector resolves tile 1/shard 0.
    PipelineRun run;
    run.first_logical_seq = 2;
    run.transition_kind = TransitionKind::DescriptorCommit;
    run.descriptor_segment_id = kSegment;
    run.tick_first = 6;
    run.cursor_version_start = 0;
    run.cursor_version_end = 0;
    SlotPatch target;
    target.is_target_delta = true;
    target.key.domain = kDomain;
    target.key.role_source = false;
    target.key.rank = 1;   // descriptor identity rank
    target.key.shard = -1;
    target.key.slot_index = 0;
    target.before_slot_generation = 0;
    TargetOp init;
    init.op = TargetOp::Op::Initialize;
    init.slot_generation = 0;
    target.operations.push_back(init);
    run.slot_patches.push_back(std::move(target));
    CreditPatch credit;
    credit.key.domain = kDomain;
    credit.key.source_rank = 0;
    credit.key.shard = 0;
    CreditRow credit_row;
    credit_row.domain = kDomain;
    credit_row.source_rank = 0;
    credit_row.shard = 0;
    credit_row.capacity = 2;
    credit_row.available = 2;
    credit_row.outstanding_ack_count = 0;
    credit.after = credit_row;
    run.credit_patches.push_back(std::move(credit));
    ObligationPatch obligation;
    obligation.obligation_id = std::string(kSegment) + "/run/1";
    ObligationRow obligation_row;
    obligation_row.obligation_id = obligation.obligation_id;
    obligation_row.segment_id = kSegment;
    obligation_row.rank = 1;
    obligation_row.first_store_id = 55;
    obligation_row.remaining_endpoint_count = 3;
    obligation_row.state = ObligationState::Pending;
    obligation.after = obligation_row;
    run.obligation_patches.push_back(std::move(obligation));
    frame.runs.push_back(std::move(run));

    CursorPatch cursor;
    cursor.segment_id = kSegment;
    cursor.cursor_version = 0;
    cursor.retired = false;
    CursorRun cursor_run;
    cursor_run.rank = 1;
    cursor_run.group_ordinal = 0;
    cursor_run.first_tile = 1;
    cursor_run.tile_count = 3;
    cursor_run.completed_endpoint_count = 0;
    cursor_run.cancelled_endpoint_count = 0;
    cursor_run.remaining_endpoint_count = 3;
    cursor_run.tile_offset = 0;
    cursor_run.shard_index = 0;
    cursor_run.leg_index = 0;
    cursor_run.has_next_store = true;
    cursor_run.next_store_id = 55;
    cursor.runs.emplace_back(1, std::move(cursor_run));
    frame.cursors.push_back(std::move(cursor));

    ResourceRow resource;
    resource.kind = ResourceKind::NocFlow;
    resource.request_id = "noc";
    resource.stage = "";
    resource.generation = 0;
    resource.rank = 0;
    resource.node_id = 11;
    resource.operation = "comm_send";
    resource.leg = "noc:11";
    resource.state = "Active";
    resource.total.bytes = WireNum::of_uint(27);
    resource.remaining.bytes = WireNum::of_uint(27);
    resource.route.push_back(0);
    resource.as_of_tick = 9;
    resource.flow_id = 11;
    resource.current_rate_Bpns = WireNum::of_double(4050.0);
    resource.last_rate_update_time = WireNum::of_uint(8);
    frame.resources.push_back(std::move(resource));
    return frame;
}

}  // namespace

int main() {
    const auto descriptor = make_descriptor();

    // 1: deterministic encoding.
    std::string first;
    std::string second;
    encode_compact_frame(sample_frame(descriptor), first);
    encode_compact_frame(sample_frame(descriptor), second);
    check(first == second, "encoding is deterministic");
    check(is_compact_frame(first.data(), first.size()), "magic present");

    // 2: full parse of the synthetic frame.
    RetainedRegistry retained;
    bool threw = false;
    ParsedEnvelope parsed;
    try {
        parsed = parse_envelope_compact(first.data(), first.size(),
                                        retained, false, 0, 9);
    } catch (const LedgerError& error) {
        threw = true;
        std::printf("unexpected parse failure: %s\n", error.what());
    }
    check(!threw, "sample frame parses");
    if (!threw) {
        check(parsed.header.full && parsed.header.epoch == "compact-test-epoch"
                  && parsed.header.event_count == 2
                  && parsed.header.result_version == 1,
              "header reconstructs");
        check(parsed.descriptors.size() == 1
                  && parsed.pipeline_runs.size() == 1
                  && parsed.cursor_patches.size() == 1
                  && parsed.resources.size() == 1
                  && parsed.node_events.size() == 1,
              "sections reconstruct");
        const PipelineRun& run = parsed.pipeline_runs[0];
        check(run.transition_kind == TransitionKind::DescriptorCommit
                  && run.slot_patches.size() == 1
                  && run.slot_patches[0].is_target_delta
                  && run.credit_patches.size() == 1
                  && run.obligation_patches.size() == 1,
              "run patches reconstruct");
        // identity reconstruction of the obligation row
        const ObligationRow& row = *run.obligation_patches[0].after;
        check(row.obligation_id == std::string(kSegment) + "/run/1"
                  && row.rank == 1 && row.first_store_id == 55,
              "obligation identity reconstructs from segment/run");
        // resource identity reconstruction
        check(parsed.resources[0].work_id == "noc#:0:r0n11@noc:11",
              "resource canonical work_id reconstructs");
        check(parsed.resources[0].current_rate_Bpns.is_float()
                  && parsed.resources[0].current_rate_Bpns.as_double()
                      == 4050.0,
              "float/int distinction survives the wire");
        // cursor reconstruction of derivable fields
        const CursorRun& cursor_run =
            *parsed.cursor_patches[0].runs[0].second;
        check(cursor_run.remaining_endpoint_count == 3
                  && cursor_run.in_flight == false
                  && cursor_run.has_next_store
                  && cursor_run.next_store_id == 55
                  && cursor_run.has_initial_parent
                  && cursor_run.initial_parent_store_id == 54,
              "cursor derivable fields reconstruct");
    }

    // 3: corrupted variants fail closed.
    const auto fails = [&](const std::string& mutated,
                           const char* what) {
        bool rejected = false;
        try {
            ParsedEnvelope out = parse_envelope_compact(
                mutated.data(), mutated.size(), retained, false, 0, 9);
            (void)out;
        } catch (const LedgerError&) {
            rejected = true;
        }
        check(rejected, what);
    };
    {
        std::string truncated = first.substr(0, first.size() - 1);
        fails(truncated, "truncated frame rejected");
    }
    {
        std::string bad_delivery = first;
        bad_delivery[10] = bad_delivery[10] + 1;   // delivery varint
        fails(bad_delivery, "outer delivery mismatch rejected");
    }
    {
        std::string bad_schema = first;
        const std::size_t at = bad_schema.find("workledger-compact-v1");
        check(at != std::string::npos, "schema literal present");
        bad_schema[at] = 'X';
        fails(bad_schema, "schema identity mismatch rejected");
    }

    // 4: structural scan of a request carrying the compact member.
    {
        const std::string request =
            "{\"delivery_sequence\":0,\"tick\":9,\"work_ledger\":"
            + first + ",\"reasons\":[\"ARRIVAL\"]}";
        const ScanResult result = scan_request(request.data(),
                                               request.size());
        const std::string outer = request.substr(0, result.member_start)
            + request.substr(result.member_end);
        check(outer == "{\"delivery_sequence\":0,\"tick\":9,"
                       "\"reasons\":[\"ARRIVAL\"]}",
              "outer splice stays valid JSON");
        const std::string segment =
            request.substr(result.value_start,
                           result.value_end - result.value_start);
        check(segment == first, "scan extracts the exact frame bytes");
    }

    if (failures == 0) {
        std::printf("compact_wire_test PASS\n");
        return 0;
    }
    std::printf("compact_wire_test FAIL (%d)\n", failures);
    return 1;
}
