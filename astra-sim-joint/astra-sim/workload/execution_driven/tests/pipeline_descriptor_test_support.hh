#ifndef EXECUTION_DRIVEN_TESTS_PIPELINE_DESCRIPTOR_TEST_SUPPORT_HH
#define EXECUTION_DRIVEN_TESTS_PIPELINE_DESCRIPTOR_TEST_SUPPORT_HH

#include <json/json.hpp>

namespace AstraSim {
namespace ExecutionDriven {
namespace TestSupport {

inline nlohmann::json descriptor_wire() {
    return nlohmann::json{
        {"descriptor_version", 1},
        {"identity", {{"request_id", "req-A"}, {"stage", "decode"},
                       {"generation", 4}, {"stage_epoch", 0},
                       {"segment_ordinal", 0}, {"rank", 1},
                       {"staging_allocation_id", 17},
                       {"segment_id", "pp1/7265712d41/decode/4/0/1/0"},
                       {"buffer_domain_id", "pb1/7265712d41/1/17"}}},
        {"stream_kind", "decode_remote_credit"},
        {"groups", nlohmann::json::array({
            {{"group_ordinal", 0}, {"pass_or_window_ordinal", 1},
             {"source_ordinal", 0}, {"source_key", "source-0"},
             {"layer_start", 0}, {"layer_end", 1},
             {"tile_axis", {{"unit", "source_tokens"}, {"total_units", 5},
                            {"units_per_tile", 2}, {"tile_count", 3},
                            {"tail_units", 1}, {"global_tile_first", 1}}},
             {"shards", nlohmann::json::array({
                 {{"shard_ordinal", 0}, {"source_rank", 0}, {"target_rank", 1},
                  {"total_data_bytes", 320}, {"bytes_per_unit", 64},
                  {"noc_path", nlohmann::json::array({0, 1})}}})}}
        })},
        {"protocol", {{"buffer_slots", 2}, {"release_lag_tiles", 2},
                      {"initial_tiles_without_release", 2},
                      {"fill_ack_bytes", 1}, {"release_ack_bytes", 1},
                      {"hbm_charge", true},
                      {"target_rank_order", "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1"},
                      {"source_rank_order", "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i"},
                      {"tile_branching", "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated"}}},
        {"prediction_profile", {{"prefill_consume", "zero_duration_no_extra_consume"},
            {"decode_consume", {{"kind", "profiled_consume"},
                                {"profile_id", "decode-service-profile-v1"},
                                {"compute_ns_total", 240.0},
                                {"hbm_bytes_total", 320},
                                {"resource_key", "hbm:1"},
                                {"share_rule", "existing_action_work_graph_byte_share_vector"}}}}},
        {"execution_profile", {{"data_legs", "ordinary_comm_send_recv_path"},
                               {"endpoint_hbm_rule", "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint"},
                               {"prefill_consume", "existing_body_or_marker_only"},
                               {"decode_consume", "existing_body_nodes_only"}}},
        {"logical_id_ranges", nlohmann::json::array({
            {{"rank", 0}, {"first_json_id", 52}, {"count", 2}, {"stride", 1},
             {"kind", "virtual_endpoint_hole"}},
            {{"rank", 1}, {"first_json_id", 55}, {"count", 2}, {"stride", 1},
             {"kind", "virtual_endpoint_hole"}}})},
        {"tag_allocation_runs", nlohmann::json::array({
            {{"allocator_scope_id", "builder-0"}, {"emission_ordinal", 7},
             {"first_tag", 101}, {"count", 7}, {"group_ordinal", 0},
             {"first_tile", 1}, {"first_shard", 0}, {"tile_count", 3},
             {"shard_count", 1}}})},
        {"endpoint_runs", nlohmann::json::array({
            {{"rank", 0}, {"group_ordinal", 0}, {"first_tile", 1},
             {"tile_count", 3}, {"shard_ordinals", nlohmann::json::array({0})},
             {"first_json_id", 51}, {"tile_stride", 1}, {"shard_stride", 1},
             {"initial_parent_json_id", nullptr},
             {"legacy_context", {{"request_id", "req-A"},
                                  {"stage", "decode"}, {"generation", 1}}},
             {"legs", nlohmann::json::array({
                 {{"endpoint", "data_send"}, {"json_offset", 0},
                  {"tag_emission_ordinal", 7}, {"tag_offset", 0},
                  {"tag_tile_stride", 1}, {"tag_shard_stride", 1}}})}},
            {{"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
             {"tile_count", 3}, {"shard_ordinals", nlohmann::json::array({0})},
             {"first_json_id", 54}, {"tile_stride", 1}, {"shard_stride", 1},
             {"initial_parent_json_id", 0},
             {"legacy_context", {{"request_id", "req-A"},
                                  {"stage", "decode"}, {"generation", 1}}},
             {"legs", nlohmann::json::array({
                 {{"endpoint", "data_recv"}, {"json_offset", 0},
                  {"tag_emission_ordinal", 7}, {"tag_offset", 0},
                  {"tag_tile_stride", 1}, {"tag_shard_stride", 1}}})}}
        })},
        {"endpoint_dependency_runs", nlohmann::json::array({
            {{"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
             {"tile_count", 3}, {"shard_ordinals", nlohmann::json::array({0})},
             {"endpoint", "data_recv"},
             {"dependency_kind", "ordinary_node_terminal"},
             {"first_dependency_json_id", 0},
             {"dependency_tile_stride", 0}, {"dependency_shard_stride", 0}}
        })},
        {"consume_completion_runs", nlohmann::json::array({
            {{"rank", 1}, {"group_ordinal", 0}, {"first_tile", 1},
             {"tile_count", 3}, {"shard_ordinals", nlohmann::json::array({0})},
             {"dependency_kind", "ordinary_node_terminal"},
             {"first_dependency_json_id", 70},
             {"dependency_tile_stride", 1}, {"dependency_shard_stride", 0}}
        })},
        {"projection", {{"version", 1}, {"chains", nlohmann::json::array({
            {{"group_ordinal", 0}, {"rank", 1}, {"tile_count", 3},
             {"fill", {{"leg_id_prefix", "fill:r1:"}, {"first_index", 0},
                       {"fill_order_first", 0}, {"fill_order_stride", 1},
                       {"completion_order_first", nullptr}, {"completion_order_stride", 0},
                       {"route", nlohmann::json::array({"hbm:0"})},
                       {"bytes_per_tile", 64}, {"latency_ns", 0},
                       {"start_deps", nlohmann::json::array()},
                       {"gate_id", "credit:0"}, {"gate_capacity", 2}}},
             {"consume", {{"leg_id_prefix", "consume:r1:"}, {"first_index", 0},
                           {"consume_order_first", 3}, {"consume_order_stride", 1},
                           {"completion_order_first", nullptr}, {"completion_order_stride", 0},
                           {"kind", "fixed"}, {"route", nlohmann::json::array()},
                           {"group", nullptr}, {"latency_first_ns", 0},
                           {"latency_rest_ns", 0}, {"consume_deps", nlohmann::json::array()},
                           {"consume_start_deps", nlohmann::json::array()},
                           {"gate_id", "credit:0"}, {"compute_share", nullptr},
                           {"hbm_share", nullptr}}}}
        })}}},
        {"anchors", nlohmann::json::array({
            {{"role", "start"}, {"rank", 0}, {"binding", "physical_endpoint_slot"},
             {"node_id", 51}, {"selector", {{"kind", "tile_endpoint_leg"},
                {"group_ordinal", 0}, {"tile_ordinal", 1}, {"shard_ordinal", 0},
                {"endpoint", "data_send"}, {"tag_allocation_emission_ordinal", 7},
                {"tag_offset", 0}}}},
            {{"role", "frontier"}, {"rank", 1}, {"binding", "physical_endpoint_slot"},
             {"node_id", 54}, {"selector", {{"kind", "tile_endpoint_terminal"},
                {"group_ordinal", 0}, {"tile_ordinal", 1}, {"shard_ordinal", 0},
                {"endpoint", "data_recv"}, {"tag_allocation_emission_ordinal", 7},
                {"tag_offset", 0}}}},
            {{"role", "service"}, {"rank", 1}, {"binding", "descriptor_cursor_selector"},
             {"node_id", nullptr}, {"selector", {{"kind", "existing_service_gate"},
                {"existing_gate", "decode_service_gate"}}}},
            {{"role", "settle"}, {"rank", 1}, {"binding", "descriptor_cursor_selector"},
             {"node_id", nullptr}, {"selector", {{"kind", "all_obligations_settled"}}}}
        })}
    };
}


}  // namespace TestSupport
}  // namespace ExecutionDriven
}  // namespace AstraSim

#endif  // EXECUTION_DRIVEN_TESTS_PIPELINE_DESCRIPTOR_TEST_SUPPORT_HH
