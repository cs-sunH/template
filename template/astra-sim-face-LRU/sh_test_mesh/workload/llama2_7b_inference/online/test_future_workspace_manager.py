"""Future workspace is immutable fit policy, separate from KV and active leases."""
import unittest
from online.test_workspace_capacity_manager import _manager, _complete_local_session
from session_kv_manager import SessionKVCacheManager


def manager_with_margin(margin):
    base = _manager()
    values = {rank: margin for rank in base.rank_ids}
    return SessionKVCacheManager(
        base.topology, base.model, strict_invariants=True,
        prospective_workspace_bytes_by_rank=values), values


class FutureWorkspaceManagerTests(unittest.TestCase):
    def test_immutable_floor_max_active_and_unchanged_kv_snapshot(self):
        manager, original = manager_with_margin(300)
        before = manager.hbm_snapshots()
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 460)
        original[2] = 0
        self.assertEqual(manager.prospective_workspace_bytes_by_rank[2], 300)
        with self.assertRaises(TypeError):
            manager.prospective_workspace_bytes_by_rank[2] = 0
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(snapshot['protocol_version'], 2)
        self.assertTrue(all(row['prospective_workspace_bytes'] == 300
                            and row['resident_kv_bytes'] == 0
                            and row['reserved_request_bytes'] == 0
                            for row in snapshot['ranks']))
        revisions = {rank: 1 for rank in manager.rank_ids}
        active = {rank: 400 for rank in manager.rank_ids}
        manager.set_external_workspace_view(revisions, active)
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 360)
        manager.set_external_workspace_view(
            {rank: 2 for rank in manager.rank_ids},
            {rank: 0 for rank in manager.rank_ids})
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 460)
        self.assertEqual(manager.hbm_snapshots(), before)
        self.assertEqual(manager.events_since(0), [])

    def test_permanent_floor_is_infeasible_before_evicting_a_cold_session(self):
        manager, _ = manager_with_margin(300)
        _complete_local_session(manager)
        before = manager.hbm_snapshots()
        cursor = len(manager.events_since(0))
        with self.assertRaisesRegex(ValueError, 'cannot fit an empty instance'):
            manager.ensure_physical_fit(0, (461, 461), 3, 'oversized')
        self.assertEqual(manager.hbm_snapshots(), before)
        events = manager.events_since(cursor)
        self.assertTrue(any(event.event_type == 'deep_gap' for event in events))
        self.assertFalse(any(event.event_type in ('evict_delete', 'remote_store')
                             for event in events))

    def test_transient_excess_over_floor_blocks_without_reclaim_then_retries(self):
        manager, _ = manager_with_margin(300)
        _complete_local_session(manager)
        before = manager.hbm_snapshots()
        manager.set_external_workspace_view(
            {rank: 1 for rank in manager.rank_ids},
            {rank: 600 if rank in (2, 3) else 0 for rank in manager.rank_ids})
        result = manager.ensure_physical_fit(0, (200, 200), 3, 'retry')
        self.assertFalse(result.admitted)
        self.assertTrue(result.blocked_by_workspace)
        self.assertFalse(result.evictions)
        self.assertEqual(manager.hbm_snapshots(), before)
        manager.set_external_workspace_view(
            {rank: 2 for rank in manager.rank_ids},
            {rank: 0 for rank in manager.rank_ids})
        self.assertTrue(manager.ensure_physical_fit(0, (200, 200), 4, 'retry').admitted)

    def test_invalid_fixed_floor_and_explicit_zero_protocol(self):
        base = _manager()
        ranks = base.rank_ids
        valid = {rank: 0 for rank in ranks}
        cases = [None, {rank: 0 for rank in ranks[:-1]}]
        for value in (True, 1.0, -1, 1 << 64, 761):
            malformed = dict(valid)
            malformed[2] = value
            cases.append(malformed)
        for values in cases:
            if values is None:
                values = []
            with self.subTest(values=values):
                with self.assertRaises(ValueError):
                    SessionKVCacheManager(
                        base.topology, base.model,
                        prospective_workspace_bytes_by_rank=values)
        zero, _ = manager_with_margin(0)
        self.assertEqual(zero.workspace_capacity_snapshot()['protocol_version'], 2)
        self.assertEqual(base.workspace_capacity_snapshot()['protocol_version'], 1)
        self.assertEqual(base.hbm_snapshots(), zero.hbm_snapshots())


if __name__ == '__main__':
    unittest.main()
