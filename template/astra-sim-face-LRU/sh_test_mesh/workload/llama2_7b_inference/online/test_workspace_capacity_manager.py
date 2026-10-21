"""Focused tests for the external C++ workspace view in the FACE KV manager."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path


MANAGER_DIR = Path(__file__).resolve().parents[1]
if str(MANAGER_DIR) not in sys.path:
    sys.path.insert(0, str(MANAGER_DIR))

from face_scheduler import FaceHardware, FaceInstanceSpec, FaceModel, build_instances  # noqa: E402
from session_kv_manager import SessionKVCacheManager  # noqa: E402


_UINT64_MAX = (1 << 64) - 1


def _manager() -> SessionKVCacheManager:
    hardware = FaceHardware(2, 2, 1000, 1.0, 2.0, 1.0, 0, 0)
    # Reverse instance order so the bridge must expose its own sorted full-rank
    # table instead of leaking topology iteration order.
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins1", "1", (2, 3)),
            FaceInstanceSpec("ins0", "2", (0, 1)),
        ),
    )
    model = FaceModel(4, 4, 4, 2, 4, 1, "gelu")
    return SessionKVCacheManager(topology, model, strict_invariants=True)


def _view(manager, *, revision: int, active_bytes: int,
          target_ranks: tuple[int, ...] = (2, 3)):
    revisions = {rank: 0 for rank in manager.rank_ids}
    workspace = {rank: 0 for rank in manager.rank_ids}
    for rank in target_ranks:
        revisions[rank] = revision
        workspace[rank] = active_bytes
    return revisions, workspace


def _complete_local_session(manager, session_id: str = "cold") -> None:
    prepared = manager.prepare_history(
        session_id, 0, 0, 0, f"{session_id}-prefill",
        required_context_tokens=10,
    )
    assert not prepared.admission_blocked
    grown = manager.grow_prefill(session_id, 10, 1, f"{session_id}-prefill")
    assert grown.admitted
    manager.mark_complete(session_id, 2, f"{session_id}-prefill")


class WorkspaceCapacityManagerTests(unittest.TestCase):
    def test_full_rank_atomic_exact_uint64_monotonic_workspace_view(self) -> None:
        manager = _manager()
        self.assertEqual(manager.rank_ids, (0, 1, 2, 3))
        original_snapshot = manager.workspace_capacity_snapshot()
        original_fit = tuple(
            manager.workspace_fit_remaining_bytes(rank) for rank in manager.rank_ids
        )

        revisions = {rank: 0 for rank in manager.rank_ids}
        workspace = {rank: 0 for rank in manager.rank_ids}
        invalid_views = []

        missing_rank = dict(revisions)
        del missing_rank[2]
        invalid_views.append((missing_rank, workspace, "exactly match topology"))
        bool_rank = {True: 0, 0: 0, 2: 0, 3: 0}
        invalid_views.append((bool_rank, workspace, "exact uint64"))
        bool_revision = dict(revisions)
        bool_revision[2] = True
        invalid_views.append((bool_revision, workspace, "exact uint64"))
        float_revision = dict(revisions)
        float_revision[2] = 1.0
        invalid_views.append((float_revision, workspace, "exact uint64"))
        bool_bytes = dict(workspace)
        bool_bytes[2] = False
        invalid_views.append((revisions, bool_bytes, "exact uint64"))
        float_bytes = dict(workspace)
        float_bytes[2] = 1.0
        invalid_views.append((revisions, float_bytes, "exact uint64"))
        overflow_revision = dict(revisions)
        overflow_revision[2] = _UINT64_MAX + 1
        invalid_views.append((overflow_revision, workspace, "exact uint64"))
        overflow_bytes = dict(workspace)
        overflow_bytes[2] = _UINT64_MAX + 1
        invalid_views.append((revisions, overflow_bytes, "exact uint64"))

        for bad_revisions, bad_bytes, message in invalid_views:
            with self.subTest(message=message, revisions=bad_revisions,
                              workspace=bad_bytes):
                with self.assertRaisesRegex(ValueError, message):
                    manager.set_external_workspace_view(bad_revisions, bad_bytes)
                self.assertEqual(manager.workspace_capacity_snapshot(), original_snapshot)
                self.assertEqual(
                    tuple(manager.workspace_fit_remaining_bytes(rank)
                          for rank in manager.rank_ids),
                    original_fit,
                )

        first_revisions, first_bytes = _view(
            manager, revision=5, active_bytes=400
        )
        self.assertEqual(
            manager.set_external_workspace_view(first_revisions, first_bytes),
            (2, 3),
        )
        installed_snapshot = manager.workspace_capacity_snapshot()
        installed_fit = tuple(
            manager.workspace_fit_remaining_bytes(rank) for rank in manager.rank_ids
        )

        stale_revisions = dict(first_revisions)
        stale_revisions[2] = 4
        with self.assertRaisesRegex(ValueError, "stale workspace revision"):
            manager.set_external_workspace_view(stale_revisions, first_bytes)
        same_revision_bytes = dict(first_bytes)
        same_revision_bytes[2] = 399
        with self.assertRaisesRegex(ValueError, "without revision advance"):
            manager.set_external_workspace_view(first_revisions, same_revision_bytes)

        # Rank 2's candidate is valid, but rank 3 exceeds physical capacity.
        # Neither rank may be installed before every rank passes validation.
        capacity_revisions = dict(first_revisions)
        capacity_revisions[2] = 6
        capacity_revisions[3] = 6
        capacity_bytes = dict(first_bytes)
        capacity_bytes[2] = 10
        capacity_bytes[3] = 1000
        with self.assertRaisesRegex(ValueError, "exceeds Python HBM capacity"):
            manager.set_external_workspace_view(capacity_revisions, capacity_bytes)
        self.assertEqual(manager.workspace_capacity_snapshot(), installed_snapshot)
        self.assertEqual(
            tuple(manager.workspace_fit_remaining_bytes(rank)
                  for rank in manager.rank_ids),
            installed_fit,
        )

        # An absolute release is valid: bytes decrease while the revision rises.
        released_revisions, released_bytes = _view(
            manager, revision=6, active_bytes=0
        )
        self.assertEqual(
            manager.set_external_workspace_view(released_revisions, released_bytes),
            (2, 3),
        )
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 760)

    def test_workspace_fit_block_release_retry_and_policy_specific_reclaim(self) -> None:
        manager = _manager()
        _complete_local_session(manager)
        target_ranks = manager.topology.instance(0).ranks
        self.assertEqual(target_ranks, (2, 3))
        before_events = manager.events_since(0)
        revisions, workspace = _view(
            manager, revision=1, active_bytes=500, target_ranks=target_ranks
        )
        self.assertEqual(
            manager.set_external_workspace_view(revisions, workspace), target_ranks
        )
        # Model weights (240B), resident KV (160B), and workspace (500B) all
        # coexist inside the 1000B rank capacity. The existing KV snapshot
        # remains KV-only; a separate getter exposes the actual fit headroom.
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 100)
        self.assertEqual(manager.workspace_fit_remaining_bytes(3), 100)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 600)
        self.assertEqual(manager.hbm_snapshots(0)[0].used_bytes, 400)
        self.assertEqual(manager.events_since(0), before_events)

        capacity = manager.workspace_capacity_snapshot()
        rank_rows = capacity["ranks"]
        self.assertEqual([row["rank"] for row in rank_rows], list(manager.rank_ids))
        self.assertEqual(rank_rows[2]["workspace_revision"], 1)
        self.assertEqual(rank_rows[2]["resident_kv_bytes"], 160)
        self.assertEqual(rank_rows[2]["reserved_request_bytes"], 0)
        self.assertTrue(all("E" not in row for row in rank_rows))

        fits = manager.ensure_physical_fit(0, (80, 80), 3, "small")
        self.assertTrue(fits.admitted)
        self.assertFalse(fits.blocked_by_workspace)
        self.assertEqual(fits.evictions, ())

        blocked = manager.ensure_physical_fit(0, (300, 300), 4, "blocked")
        self.assertFalse(blocked.admitted)
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertEqual(blocked.insufficient_ranks, target_ranks)
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 100)
        self.assertEqual(manager.session_snapshot("cold").shard_bytes, (160, 160))

        admission = manager.prepare_history(
            "new", 0, 0, 4, "blocked-prefill", required_context_tokens=20
        )
        self.assertTrue(admission.admission_blocked)
        self.assertTrue(admission.blocked_by_workspace)
        self.assertEqual(admission.insufficient_ranks, target_ranks)
        self.assertIsNone(manager.session_snapshot("new"))

        released_revisions, released_bytes = _view(
            manager, revision=2, active_bytes=0, target_ranks=target_ranks
        )
        self.assertEqual(
            manager.set_external_workspace_view(released_revisions, released_bytes),
            target_ranks,
        )
        retry_admission = manager.prepare_history(
            "new", 0, 0, 5, "retry-prefill", required_context_tokens=20
        )
        self.assertFalse(retry_admission.admission_blocked)
        self.assertFalse(retry_admission.blocked_by_workspace)
        growth = manager.grow_prefill("new", 20, 6, "retry-prefill")
        self.assertTrue(growth.admitted)

        retry = manager.ensure_physical_fit(0, (400, 400), 7, "retry-reclaim")
        self.assertTrue(retry.admitted)
        self.assertFalse(retry.blocked_by_workspace)
        self.assertEqual(
            [record.victim_session_id for record in retry.evictions], ["cold"]
        )

        cold = manager.session_snapshot("cold")
        if hasattr(cold, "location"):
            # FACE-LRU stores a completed victim and retains its KV remotely.
            self.assertEqual(cold.location, "remote_memory")
            self.assertEqual(cold.remote_bytes, 320)
            self.assertEqual(retry.evictions[0].transfer.kind, "remote_store")
        else:
            # FACE keeps its existing whole-session deletion/recompute policy.
            self.assertEqual(cold.state, "EVICTED")
        self.assertEqual(
            tuple(s.resident_kv_bytes for s in manager.hbm_snapshots(0)),
            (320, 320),
        )

        # A smaller request fits on an otherwise empty rank under this live
        # workspace. Reclaim exactly the first whole-session victim and stop.
        prefix_manager = _manager()
        _complete_local_session(prefix_manager)
        prefix_revisions, prefix_workspace = _view(
            prefix_manager, revision=1, active_bytes=500
        )
        prefix_manager.set_external_workspace_view(
            prefix_revisions, prefix_workspace
        )
        prefix_fit = prefix_manager.ensure_physical_fit(
            0, (250, 250), 3, "workspace-prefix"
        )
        self.assertTrue(prefix_fit.admitted)
        self.assertFalse(prefix_fit.blocked_by_workspace)
        self.assertEqual(
            [record.victim_session_id for record in prefix_fit.evictions],
            ["cold"],
        )

    def test_capacity_snapshot_reports_reserved_bytes_where_supported(self) -> None:
        manager = _manager()
        if not hasattr(manager, "reserve_request_capacity"):
            snapshot = manager.workspace_capacity_snapshot()
            self.assertEqual(
                [row["reserved_request_bytes"] for row in snapshot["ranks"]],
                [0, 0, 0, 0],
            )
            return

        reservation = manager.reserve_request_capacity(
            "reserve", "session", 0, (17, 19), 0
        )
        self.assertTrue(reservation.admitted)
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(
            [row["reserved_request_bytes"] for row in snapshot["ranks"]],
            [0, 0, 17, 19],
        )
        revisions, workspace = _view(manager, revision=1, active_bytes=0)
        workspace[2] = 743
        workspace[3] = 741
        self.assertEqual(
            manager.set_external_workspace_view(revisions, workspace), (2, 3)
        )
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 0)
        self.assertEqual(manager.workspace_fit_remaining_bytes(3), 0)
        blocked = manager.ensure_physical_fit(0, (1, 1), 1, "reserved-block")
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertEqual(blocked.evictions, ())

        revisions[2] = 2
        workspace[2] = 744
        with self.assertRaisesRegex(ValueError, "exceeds Python HBM capacity"):
            manager.set_external_workspace_view(revisions, workspace)
        self.assertEqual(manager.workspace_fit_remaining_bytes(2), 0)


if __name__ == "__main__":
    unittest.main()
