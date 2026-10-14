import copy
from collections import UserDict
from collections.abc import Mapping as _TestMapping
from dataclasses import replace
from types import MappingProxyType
import unittest

from joint import pipeline_descriptor as pipeline_descriptor_module
from joint.pipeline_descriptor import (
    PipelineDescriptor,
    PipelineDescriptorError,
    PipelineLogicalIdRange,
    PipelineShardRule,
    _endpoint_owner_rank,
    _validate_id_ranges,
)
from joint.resource_projection import (
    ResourceProjectionError,
    TASK_KIND_FIXED,
    TaskSpec,
    WorkItem,
)


def _wire():
    return {
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
            "buffer_domain_id": "pb1/726571/1/9",
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
                "global_tile_first": 1,
            },
            "shards": [{
                "shard_ordinal": 0,
                "source_rank": 0,
                "target_rank": 1,
                "total_data_bytes": 5,
                "bytes_per_unit": 1,
                "noc_path": [0, 1],
            }],
        }],
        "protocol": {
            "buffer_slots": 2,
            "release_lag_tiles": 2,
            "initial_tiles_without_release": 2,
            "fill_ack_bytes": 1,
            "release_ack_bytes": 1,
            "hbm_charge": True,
            "target_rank_order": "release_ack_send_i_if_required_then_data_recv_i_then_fill_ack_send_i_then_data_recv_i_plus_1",
            "source_rank_order": "release_ack_recv_i_if_required_then_data_send_i_then_fill_ack_recv_i",
            "tile_branching": "tile1_head_main_chain_tile2_immediate_side_branch_tile3_plus_release_gated",
        },
        "prediction_profile": {
            "prefill_consume": "zero_duration_no_extra_consume",
            "decode_consume": {
                "kind": "profiled_consume",
                "profile_id": "decode-service-profile-v1",
                "compute_ns_total": 240.0,
                "hbm_bytes_total": 5,
                "resource_key": "hbm:1",
                "share_rule": "existing_action_work_graph_byte_share_vector",
            },
        },
        "execution_profile": {
            "data_legs": "ordinary_comm_send_recv_path",
            "endpoint_hbm_rule": "one_original_COMM_READ_or_COMM_WRITE_job_per_charged_endpoint",
            "prefill_consume": "existing_body_or_marker_only",
            "decode_consume": "existing_body_nodes_only",
        },
        "logical_id_ranges": [
            {"rank": 0, "first_json_id": 53, "count": 2, "stride": 2,
             "kind": "virtual_endpoint_hole"},
            {"rank": 1, "first_json_id": 56, "count": 2, "stride": 2,
             "kind": "virtual_endpoint_hole"},
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
            "shard_count": 1,
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
                    "generation": 1,
                },
                "legs": [{
                    "endpoint": "data_send",
                    "json_offset": 0,
                    "tag_emission_ordinal": 7,
                    "tag_offset": 0,
                    "tag_tile_stride": 1,
                    "tag_shard_stride": 1,
                }],
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
                    "generation": 1,
                },
                "legs": [{
                    "endpoint": "data_recv",
                    "json_offset": 0,
                    "tag_emission_ordinal": 7,
                    "tag_offset": 0,
                    "tag_tile_stride": 1,
                    "tag_shard_stride": 1,
                }],
            },
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
            "dependency_shard_stride": 0,
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
            "dependency_shard_stride": 0,
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
                    "completion_order_first": None,
                    "completion_order_stride": 0,
                    "route": ["link:0"],
                    "bytes_per_tile": 2,
                    "latency_ns": 3,
                    "start_deps": ["work#start"],
                    "gate_id": "work#gate:dcstream:R1",
                    "gate_capacity": 2,
                },
                "consume": {
                    "leg_id_prefix": "work#dcstream#R1#C",
                    "first_index": 0,
                    "consume_order_first": 3,
                    "consume_order_stride": 1,
                    "completion_order_first": None,
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
                        "tile_bytes": 2,
                    },
                    "hbm_share": {
                        "rule": "quotient_remainder_earliest_global_indices",
                        "total_units": 5,
                        "total_tiles": 3,
                        "share_index_first": 0,
                        "tile_bytes": 2,
                    },
                },
            }],
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
                    "tag_offset": 0,
                },
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
                    "tag_offset": 0,
                },
            },
            {
                "role": "service",
                "rank": 1,
                "binding": "descriptor_cursor_selector",
                "node_id": None,
                "selector": {"kind": "existing_service_gate", "existing_gate": "decode_service_gate"},
            },
            {
                "role": "settle",
                "rank": 1,
                "binding": "descriptor_cursor_selector",
                "node_id": None,
                "selector": {"kind": "all_obligations_settled"},
            },
        ],
    }


def _descriptor_with_shard_count(shard_count, *, unit="stream_bytes"):
    wire = copy.deepcopy(_wire())
    group = wire["groups"][0]
    group["tile_axis"]["unit"] = unit
    group["shards"] = []
    for index in range(shard_count):
        source_rank = 0 if index == 0 else index + 1
        bytes_per_unit = index + 1 if unit == "source_tokens" else 1
        group["shards"].append({
            "shard_ordinal": index,
            "source_rank": source_rank,
            "target_rank": 1,
            "total_data_bytes": 5 * bytes_per_unit,
            "bytes_per_unit": bytes_per_unit,
            "noc_path": [source_rank, 1],
        })
    # The descriptor validator requires a consume-completion proof for every
    # active target shard. Give each added shard its own ordinary terminal ID
    # using the same affine shard stride as the declared shard order.
    consume = wire["consume_completion_runs"][0]
    consume["shard_ordinals"] = list(range(shard_count))
    consume["dependency_shard_stride"] = 1
    return PipelineDescriptor.from_wire(wire)


class PipelineDescriptorTest(unittest.TestCase):
    def test_from_wire_accepts_standard_and_custom_mapping_forms(self):
        class CustomMapping(_TestMapping):
            def __init__(self, values):
                self._values = dict(values)

            def __getitem__(self, key):
                return self._values[key]

            def __iter__(self):
                return iter(self._values)

            def __len__(self):
                return len(self._values)

        def map_tree(value, factory):
            if isinstance(value, dict):
                return factory({key: map_tree(item, factory)
                                for key, item in value.items()})
            if isinstance(value, list):
                return [map_tree(item, factory) for item in value]
            if isinstance(value, tuple):
                return tuple(map_tree(item, factory) for item in value)
            return value

        expected = PipelineDescriptor.from_wire(_wire())
        factories = (
            ("dict", dict),
            ("UserDict", UserDict),
            ("MappingProxyType", MappingProxyType),
            ("custom Mapping", CustomMapping),
        )
        for name, factory in factories:
            with self.subTest(mapping_type=name):
                wrapped = map_tree(_wire(), factory)
                self.assertEqual(
                    PipelineDescriptor.from_wire(wrapped), expected)
                self.assertEqual(
                    PipelineDescriptor.from_plan(wrapped), expected)

        with self.assertRaisesRegex(PipelineDescriptorError, "must be an object"):
            PipelineDescriptor.from_wire([])

    def test_endpoint_id_and_tag_runs_are_resolved_lazily(self):
        descriptor = PipelineDescriptor.from_wire(_wire())
        self.assertEqual(descriptor.endpoint_id(0, 1, 0, 0), 51)
        self.assertEqual(descriptor.endpoint_id(1, 3, 0, 0), 58)
        self.assertEqual(descriptor.endpoint_tag(0, 3, 0, 0), 103)
        self.assertEqual(descriptor.tile(0, 2).shards[0].data_bytes, 1)
        self.assertEqual(descriptor.prefix(0, 2), 4)
        share = descriptor.projection.chains[0].consume.compute_share
        self.assertEqual([share.at_global_index(i) for i in range(3)], [3, 3, 2])
        self.assertEqual(descriptor.to_wire(), _wire())

    def test_selected_shard_tile_matches_full_tiles_across_groups_and_tails(self):
        for unit in ("stream_bytes", "source_tokens"):
            descriptor = _descriptor_with_shard_count(3, unit=unit)
            group = descriptor.groups[0]
            for local_tile in range(group.tile_axis.tile_count):
                full = descriptor.tile(group.group_ordinal, local_tile)
                for shard_index, expected in enumerate(full.shards):
                    global_tile, selected = descriptor.selected_shard_tile(
                        group.group_ordinal, local_tile, shard_index)
                    with self.subTest(unit=unit, local_tile=local_tile,
                                      shard_index=shard_index):
                        self.assertEqual(global_tile, full.global_tile_ordinal)
                        self.assertEqual(selected, expected)
                        self.assertEqual((global_tile - 1) % 2,
                                         (full.global_tile_ordinal - 1) % 2)
                        released_tile = (
                            global_tile
                            - descriptor.protocol.release_lag_tiles)
                        if released_tile > 0:
                            self.assertEqual(
                                (global_tile - 1) % descriptor.protocol.buffer_slots,
                                (released_tile - 1)
                                % descriptor.protocol.buffer_slots)
            expected_tail_bytes = 1 if unit == "stream_bytes" else 3
            self.assertEqual(
                descriptor.selected_shard_tile(0, 2, 2)[1].data_bytes,
                expected_tail_bytes)

            # A second independently numbered group checks lookup by the
            # caller's group ordinal and global-tile offset.
            second_axis = replace(
                group.tile_axis, global_tile_first=101)
            second_group = replace(
                group, group_ordinal=9, tile_axis=second_axis)
            multi_group = replace(
                descriptor, groups=(group, second_group))
            for local_tile in range(second_axis.tile_count):
                full = multi_group.tile(9, local_tile)
                global_tile, selected = multi_group.selected_shard_tile(
                    9, local_tile, 2)
                self.assertEqual(global_tile, 101 + local_tile)
                self.assertEqual(selected, full.shards[2])
                self.assertEqual((global_tile - 1) % 2,
                                 (full.global_tile_ordinal - 1) % 2)

    def test_selected_shard_tile_builds_one_shard_at_all_scales(self):
        original = pipeline_descriptor_module._make_pipeline_shard_tile
        for shard_count in (8, 32, 128):
            descriptor = _descriptor_with_shard_count(shard_count)
            built = []

            def counted_builder(shard, data_bytes):
                built.append(shard.shard_ordinal)
                return original(shard, data_bytes)

            pipeline_descriptor_module._make_pipeline_shard_tile = counted_builder
            try:
                full = descriptor.tile(0, 2)
                self.assertEqual(len(built), shard_count)
                built.clear()
                global_tile, selected = descriptor.selected_shard_tile(
                    0, 2, shard_count - 1)
                self.assertEqual(len(built), 1)
                self.assertEqual(built, [shard_count - 1])
                self.assertEqual(global_tile, full.global_tile_ordinal)
                self.assertEqual(selected, full.shards[-1])
            finally:
                pipeline_descriptor_module._make_pipeline_shard_tile = original

    def test_selected_shard_tile_rejects_bad_indices(self):
        descriptor = _descriptor_with_shard_count(2)
        with self.assertRaisesRegex(PipelineDescriptorError, "unknown group"):
            descriptor.selected_shard_tile(99, 0, 0)
        with self.assertRaisesRegex(PipelineDescriptorError, "outside group"):
            descriptor.selected_shard_tile(0, 3, 0)
        with self.assertRaisesRegex(PipelineDescriptorError, "local_tile_ordinal"):
            descriptor.selected_shard_tile(0, True, 0)
        with self.assertRaisesRegex(PipelineDescriptorError, "group_shard_index"):
            descriptor.selected_shard_tile(0, 0, True)
        with self.assertRaisesRegex(PipelineDescriptorError, "group_shard_index"):
            descriptor.selected_shard_tile(0, 0, -1)
        with self.assertRaisesRegex(PipelineDescriptorError, "shard index"):
            descriptor.selected_shard_tile(0, 0, 2)

    def test_interleaved_holes_use_sparse_arithmetic_ranges(self):
        descriptor = PipelineDescriptor.from_wire(_wire())
        self.assertEqual(
            [(row.first_json_id, row.count, row.stride)
             for row in descriptor.logical_id_ranges],
            [(53, 2, 2), (56, 2, 2)])

    def test_ack_endpoints_belong_to_the_receiver_rank(self):
        shard = PipelineShardRule(
            shard_ordinal=0, source_rank=0, target_rank=1,
            total_data_bytes=5, bytes_per_unit=1, noc_path=(0, 1))
        self.assertEqual(_endpoint_owner_rank(shard, "data_send"), 0)
        self.assertEqual(_endpoint_owner_rank(shard, "data_recv"), 1)
        self.assertEqual(_endpoint_owner_rank(shard, "release_ack_send"), 1)
        self.assertEqual(_endpoint_owner_rank(shard, "release_ack_recv"), 0)
        self.assertEqual(_endpoint_owner_rank(shard, "fill_ack_send"), 1)
        self.assertEqual(_endpoint_owner_rank(shard, "fill_ack_recv"), 0)
        self.assertEqual(_endpoint_owner_rank(shard, "body_marker"), 1)

    def test_projection_only_descriptor_allows_empty_physical_terminals(self):
        wire = copy.deepcopy(_wire())
        wire["logical_id_ranges"] = []
        wire["tag_allocation_runs"] = []
        wire["endpoint_runs"] = []
        wire["endpoint_dependency_runs"] = []
        wire["consume_completion_runs"] = []
        wire["anchors"] = []
        descriptor = PipelineDescriptor.from_wire(wire)
        self.assertEqual(descriptor.endpoint_runs, ())
        self.assertEqual(descriptor.consume_completion_runs, ())

    def test_hole_progressions_may_overlap_as_intervals_when_ids_do_not(self):
        _validate_id_ranges((
            PipelineLogicalIdRange(0, 10, 3, 4, "virtual_endpoint_hole"),
            PipelineLogicalIdRange(0, 12, 3, 4, "virtual_endpoint_hole"),
        ))
        with self.assertRaises(PipelineDescriptorError):
            _validate_id_ranges((
                PipelineLogicalIdRange(0, 10, 3, 4, "virtual_endpoint_hole"),
                PipelineLogicalIdRange(0, 14, 2, 2, "virtual_endpoint_hole"),
            ))

    def test_fill_and_consume_registration_progressions_must_not_overlap(self):
        wire = _wire()
        wire["projection"]["chains"][0]["consume"][
            "consume_order_first"] = 2
        with self.assertRaises(PipelineDescriptorError):
            PipelineDescriptor.from_wire(wire)

    def test_work_item_merges_ordinary_and_virtual_registration_ordinals(self):
        descriptor = PipelineDescriptor.from_wire(_wire())
        terminal = descriptor.projection_terminal_ids()
        self.assertEqual(terminal, ("work#dcstream#R1#C2",))
        accepted = WorkItem(
            "work",
            (TaskSpec("work#ordinary", kind=TASK_KIND_FIXED,
                      duration_ns=1, registration_ordinal=6),),
            is_candidate=True,
            service_leg_ids=terminal,
            pipeline_descriptors=(descriptor,))
        self.assertEqual(accepted.service_leg_ids, terminal)
        with self.assertRaises(ResourceProjectionError):
            WorkItem(
                "work",
                (TaskSpec("work#colliding", kind=TASK_KIND_FIXED,
                          duration_ns=1, registration_ordinal=2),),
                is_candidate=True,
                pipeline_descriptors=(descriptor,))

    def test_body_marker_must_not_reserve_a_tag(self):
        wire = _wire()
        wire["endpoint_runs"][0]["legs"][0] = {
            "endpoint": "body_marker",
            "json_offset": 0,
            "tag_emission_ordinal": 7,
            "tag_offset": 0,
            "tag_tile_stride": 1,
            "tag_shard_stride": 1,
        }
        with self.assertRaises(PipelineDescriptorError):
            PipelineDescriptor.from_wire(wire)

    def test_unknown_endpoint_run_fields_fail_closed(self):
        wire = copy.deepcopy(_wire())
        wire["endpoint_runs"][0]["future_tiles"] = 99
        with self.assertRaises(PipelineDescriptorError):
            PipelineDescriptor.from_wire(wire)

    def test_dependency_run_wire_keys_and_extent_are_strict(self):
        wire = copy.deepcopy(_wire())
        wire["endpoint_dependency_runs"][0]["future_tiles"] = 99
        with self.assertRaisesRegex(PipelineDescriptorError, "keys invalid"):
            PipelineDescriptor.from_wire(wire)
        wire = copy.deepcopy(_wire())
        wire["consume_completion_runs"][0]["first_tile"] = 2
        wire["consume_completion_runs"][0]["tile_count"] = 2
        with self.assertRaisesRegex(PipelineDescriptorError, "gap|cover every"):
            PipelineDescriptor.from_wire(wire)

    def test_projection_chain_rejects_noncontiguous_rank_indices(self):
        wire = copy.deepcopy(_wire())
        wire["projection"]["chains"][0]["fill"]["first_index"] = 1
        with self.assertRaises(PipelineDescriptorError):
            PipelineDescriptor.from_wire(wire)


if __name__ == "__main__":
    unittest.main()
