import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _path in (_JOINT_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from joint.native_projection import _task_wire
from joint.pipeline_identity import (
    PipelineIdentityAllocator,
    PipelineIdentitySeed,
)
from joint.resource_projection import ResourceProjectionError, TaskSpec


class TaskRegistrationOrdinalTest(unittest.TestCase):
    def test_uint64_registration_is_separate_from_completion_order(self):
        task = TaskSpec("leg", order=-1, registration_ordinal=0)
        wire = _task_wire(task)
        self.assertEqual(wire["order"], -1)
        self.assertEqual(wire["registration_ordinal"], 0)

    def test_registration_ordinal_rejects_bool_and_out_of_range_values(self):
        for invalid in (True, -1, 1 << 64):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ResourceProjectionError):
                    TaskSpec("leg", registration_ordinal=invalid)

    def test_pipeline_identity_preview_and_atomic_commit(self):
        allocator = PipelineIdentityAllocator()
        seed = PipelineIdentitySeed(
            request_id="req", stage="decode", generation=1,
            stage_epoch=3, rank=2, staging_allocation_id=9)
        first_preview = allocator.preview_batch((seed, seed))
        second_preview = allocator.preview_batch((seed, seed))
        self.assertEqual(first_preview, second_preview)
        self.assertEqual(
            [item.segment_ordinal for item in first_preview], [0, 1])
        self.assertEqual(
            first_preview[0].segment_id, "pp1/726571/decode/1/3/2/0")
        self.assertEqual(
            first_preview[0].buffer_domain_id, "pb1/726571/2/9")
        committed = allocator.commit_batch((seed, seed))
        self.assertEqual(committed, first_preview)
        self.assertEqual(allocator.next_ordinal(seed), 2)
        self.assertEqual(allocator.preview_batch((seed,))[0].segment_ordinal, 2)

    def test_pipeline_identity_preview_does_not_consume_and_p_d_share_domain(self):
        allocator = PipelineIdentityAllocator()
        prefill = PipelineIdentitySeed(
            "req", "prefill", 0, 0, 1, 7)
        decode = PipelineIdentitySeed(
            "req", "decode", 1, 1, 1, 7)
        self.assertEqual(allocator.preview_batch((prefill,))[0].segment_ordinal, 0)
        self.assertEqual(allocator.preview_batch((decode,))[0].segment_ordinal, 0)
        p_identity = allocator.commit_batch((prefill,))[0]
        d_identity = allocator.commit_batch((decode,))[0]
        self.assertEqual(p_identity.buffer_domain_id, d_identity.buffer_domain_id)
        self.assertNotEqual(p_identity.segment_id, d_identity.segment_id)

    def test_identity_state_restore_merges_committed_descriptor_floor(self):
        allocator = PipelineIdentityAllocator()
        seed = PipelineIdentitySeed(
            "req", "decode", 1, 3, 2, 9)
        identity = allocator.commit_batch((seed,))[0]
        stale_state = {"version": 1, "next_segment_ordinals": []}
        restored = PipelineIdentityAllocator()
        restored.restore_state(stale_state,
                               committed_identities=(identity,))
        self.assertEqual(restored.next_ordinal(seed), 1)
        self.assertEqual(
            restored.preview_batch((seed,))[0].segment_ordinal, 1)
        state = allocator.snapshot_state()
        clone = PipelineIdentityAllocator()
        clone.restore_state(state)
        self.assertEqual(clone.snapshot_state(), state)


if __name__ == "__main__":
    unittest.main()
