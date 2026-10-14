import unittest
from types import SimpleNamespace

from joint.compact_pipeline import PhysicalPipelineDraft
from joint.pipeline_identity import PipelineIdentityAllocator, PipelineIdentitySeed


def _draft(tile_count):
    identity = PipelineIdentityAllocator().preview_batch((
        PipelineIdentitySeed("range", "decode", 1, 4, 1, 9),))[0]
    draft = PhysicalPipelineDraft(
        identity=identity, stream_kind="decode_remote_credit")
    shard = SimpleNamespace(
        source_rank=0, target_rank=1,
        noc_path=(0, 1))
    group = draft.add_group_range(
        pass_or_window_ordinal=0, source_ordinal=0, source_key="source",
        layer_start=0, layer_end=2, global_tile_first=1,
        tile_count=tile_count, axis_unit="stream_bytes",
        total_units=tile_count, units_per_tile=1, tail_units=1,
        owner_shards=(shard,), shard_total_bytes=(tile_count,))
    draft.add_tag_allocation_reference(
        allocator_scope_id="graph-batch-transfer-tags-v1",
        emission_ordinal=0, first_tag=1, count=2 * tile_count,
        group_ordinal=group, first_tile=1, tile_count=tile_count,
        first_shard=0, shard_count=1)
    tag_fields = {
        "tag_emission_ordinal": 0,
        "tag_offset": 0,
        "tag_tile_stride": 2,
        "tag_shard_stride": 1,
    }
    source_run = {
        "rank": 0, "group_ordinal": group,
        "first_tile": 1, "tile_count": tile_count,
        "shard_ordinals": [0], "first_json_id": 0,
        "tile_stride": 2, "shard_stride": 2,
        "initial_parent_json_id": None,
        "legacy_context": {"request_id": "range", "stage": "decode",
                           "generation": 1},
        "legs": [
            {"endpoint": "data_send", "json_offset": 0,
             **tag_fields},
            {"endpoint": "fill_ack_recv", "json_offset": 1,
             **{**tag_fields, "tag_offset": 1}},
        ],
    }
    target_run = {
        **source_run,
        "rank": 1,
        "legs": [
            {"endpoint": "data_recv", "json_offset": 0,
             **tag_fields},
            {"endpoint": "fill_ack_send", "json_offset": 1,
             **{**tag_fields, "tag_offset": 1}},
        ],
    }
    draft.add_endpoint_run(
        source_run,
        anchor_exceptions=((1, 0, "data_send", "start"),))
    draft.add_endpoint_run(
        target_run,
        anchor_exceptions=((tile_count, 0, "fill_ack_send", "frontier"),))
    draft.add_consume_completion_run({
        "rank": 1, "group_ordinal": group,
        "first_tile": 1, "tile_count": tile_count,
        "shard_ordinals": [0],
        "dependency_kind": "virtual_endpoint_terminal",
        "first_dependency_json_id": 0,
        "dependency_tile_stride": 2, "dependency_shard_stride": 0,
    })
    return draft.finish().to_wire()


class CompactPhysicalRangeTests(unittest.TestCase):
    def test_wire_size_is_structural_not_per_tile(self):
        small = _draft(3)
        large = _draft(100_000)
        self.assertEqual(len(small["groups"]), len(large["groups"]))
        self.assertEqual(len(small["endpoint_runs"]),
                         len(large["endpoint_runs"]))
        self.assertEqual(len(small["logical_id_ranges"]),
                         len(large["logical_id_ranges"]))
        self.assertEqual(len(small["consume_completion_runs"]),
                         len(large["consume_completion_runs"]))
        self.assertEqual(large["groups"][0]["tile_axis"]["tile_count"],
                         100_000)
        self.assertEqual(large["tag_allocation_runs"][0]["count"],
                         200_000)


if __name__ == "__main__":
    unittest.main()
