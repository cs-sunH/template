#!/usr/bin/env python3
"""Focused Python tests for the LRU manager's external workspace view."""

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

from metrics_integration import MemoryActionRecorder  # noqa: E402
from metrics_schema import MemoryMetricsObserver  # noqa: E402
import session_kv_manager as manager_module  # noqa: E402
from session_kv_manager import (  # noqa: E402
    LOCAL_HBM,
    REMOTE_MEMORY,
    REMOTE_RESTORE,
    SessionKVCacheManager,
    kv_cache_shard_bytes_for_tokens,
    set_metrics_observer,
)
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    build_instances,
)


def _model() -> WscLlmModel:
    # Same pinned compact model as test_tiered_eviction_sequence: 320 B for
    # ten tokens per rank and 480 B of weight per rank.
    return WscLlmModel(4, 4, 4, 2, 4, 1, "gelu")


def _topology(capacity_bytes: int):
    hardware = WscLlmHardware(
        mesh_rows=1,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    return build_instances(
        hardware,
        (
            WscLlmInstanceSpec("d", "1", (0,), DECODE_ROLE),
            WscLlmInstanceSpec("p", "2", (1,), PREFILL_ROLE),
        ),
    )


def _manager(
    extra_capacity: int = 320,
    *,
    strict: bool = True,
    prospective_workspace_bytes_by_rank=None,
):
    return SessionKVCacheManager(
        _topology(480 + extra_capacity),
        _model(),
        strict_invariants=strict,
        prospective_workspace_bytes_by_rank=prospective_workspace_bytes_by_rank,
    )


def _seed_local(manager, session_id: str, completion_ns: int, tokens: int = 10):
    prepared = manager.prepare_history(
        session_id, 0, 0, completion_ns, f"{session_id}0",
        required_context_tokens=tokens,
    )
    assert not prepared.admission_blocked
    grown = manager.grow_prefill(
        session_id, tokens, completion_ns, f"{session_id}0"
    )
    assert grown.admitted
    manager.mark_complete(session_id, completion_ns, f"{session_id}0")


class WorkspaceAdmissionManagerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.addCleanup(set_metrics_observer, None)

    def test_absolute_revisions_and_capacity_snapshot(self) -> None:
        manager = _manager()
        self.assertEqual(manager.rank_ids, (0, 1))
        initial = manager.workspace_capacity_snapshot()
        self.assertEqual(initial["protocol_version"], 1)
        self.assertEqual(
            [row["workspace_revision"] for row in initial["ranks"]], [0, 0]
        )
        self.assertEqual(
            [row["model_weight_bytes"] for row in initial["ranks"]], [480, 480]
        )

        self.assertEqual(
            manager.set_external_workspace_view({0: 4, 1: 7}, {0: 64, 1: 0}),
            (0, 1),
        )
        self.assertEqual(
            manager.set_external_workspace_view({0: 4, 1: 7}, {0: 64, 1: 0}),
            (),
        )
        # A revision-only advance is still a capacity-epoch change.
        self.assertEqual(
            manager.set_external_workspace_view({0: 5, 1: 7}, {0: 64, 1: 0}),
            (0,),
        )
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(
            [row["workspace_revision"] for row in snapshot["ranks"]], [5, 7]
        )
        self.assertEqual(
            [row["reserved_request_bytes"] for row in snapshot["ranks"]], [0, 0]
        )

    def test_prospective_workspace_is_full_rank_exact_uint64_and_immutable(self) -> None:
        margins = {0: 64, 1: 32}
        manager = _manager(prospective_workspace_bytes_by_rank=margins)
        margins[0] = 0  # Constructor owns a defensive copy.
        self.assertEqual(
            dict(manager.prospective_workspace_bytes_by_rank),
            {0: 64, 1: 32},
        )
        with self.assertRaises(TypeError):
            manager.prospective_workspace_bytes_by_rank[0] = 0

        with self.assertRaisesRegex(ValueError, "exactly match topology ranks"):
            _manager(prospective_workspace_bytes_by_rank={0: 64})
        with self.assertRaisesRegex(ValueError, "exact uint64"):
            _manager(prospective_workspace_bytes_by_rank={0: True, 1: 0})
        with self.assertRaisesRegex(ValueError, "exact uint64"):
            _manager(
                prospective_workspace_bytes_by_rank={
                    0: manager_module._UINT64_MAX + 1,
                    1: 0,
                }
            )
        with self.assertRaisesRegex(ValueError, "exceeds empty Python HBM"):
            _manager(prospective_workspace_bytes_by_rank={0: 321, 1: 0})

        # Weight + floor may exactly equal u64/capacity without wrapping.
        max_u64 = manager_module._UINT64_MAX
        boundary = _manager(
            extra_capacity=max_u64 - 480,
            prospective_workspace_bytes_by_rank={0: max_u64 - 480, 1: 0},
        )
        self.assertEqual(boundary._rank_states[0].capacity_bytes, max_u64)
        self.assertEqual(boundary._fit_remaining(0), (0,))

    def test_fit_uses_max_of_actual_and_prospective_without_snapshot_double_count(self) -> None:
        manager = _manager(
            extra_capacity=70,
            prospective_workspace_bytes_by_rank={0: 50, 1: 0},
        )
        self.assertEqual(manager._fit_remaining(0), (20,))
        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 40, 1: 0})
        # The prospective 50 B floor wins; the actual 40 B lease is not added.
        self.assertEqual(manager._fit_remaining(0), (20,))
        manager.set_external_workspace_view({0: 2, 1: 2}, {0: 60, 1: 0})
        # Once A > E, the actual lease is the margin, still not E + A.
        self.assertEqual(manager._fit_remaining(0), (10,))
        self.assertEqual(manager._remaining(0), (70,))
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(snapshot["ranks"][0]["capacity_bytes"], 550)
        self.assertNotIn("prospective_workspace_bytes", snapshot["ranks"][0])

    def test_permanent_floor_overflow_is_error_but_larger_active_hold_is_retryable(self) -> None:
        manager = _manager(
            prospective_workspace_bytes_by_rank={0: 64, 1: 0}
        )
        self.assertEqual(manager._fit_remaining(0), (256,))
        with self.assertRaisesRegex(
            ValueError, "cannot fit an empty instance after prospective workspace"
        ):
            manager.ensure_physical_fit(0, (257,), 1, "permanent-overflow")
        self.assertEqual(manager.deep_gap_events, 0)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 0, 1: 0})

        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 128, 1: 0})
        blocked = manager.ensure_physical_fit(0, (250,), 2, "temporary-block")
        self.assertFalse(blocked.admitted)
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.deep_gap_events, 0)
        manager.set_external_workspace_view({0: 2, 1: 2}, {0: 0, 1: 0})
        retried = manager.ensure_physical_fit(0, (250,), 3, "temporary-block")
        self.assertTrue(retried.admitted)

    def test_reservation_extension_checks_workspace_before_mutating(self) -> None:
        manager = _manager(
            prospective_workspace_bytes_by_rank={0: 64, 1: 0}
        )
        reserved = manager.reserve_request_capacity("r0", "s0", 0, (250,), 1)
        self.assertTrue(reserved.admitted)
        with self.assertRaisesRegex(
            ValueError, "reservation extension exceeds Python HBM capacity"
        ):
            manager.extend_request_capacity("r0", (7,), 2)
        self.assertEqual(manager._reservations["r0"].shard_bytes, (250,))
        self.assertEqual(manager.hbm_snapshots(0)[0].reserved_request_bytes, 250)
        manager._check_invariants()

    def test_prospective_floor_preserves_protected_lru_and_continues_in_order(self) -> None:
        manager = _manager(
            extra_capacity=640 + 64,
            prospective_workspace_bytes_by_rank={0: 64, 1: 0},
        )
        _seed_local(manager, "old", 10)
        _seed_local(manager, "new", 20)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 64)

        # Keeping the oldest session non-reclaimable leaves only 320 B after
        # the cold candidate is evicted; E leaves 256 B, short of 350 B.
        # Do not perform a futile store or count a deep gap.
        blocked = manager.ensure_physical_fit(
            0, (350,), 30, "retry", protected_sessions=("old",)
        )
        self.assertFalse(blocked.admitted)
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.deep_gap_events, 0)
        self.assertEqual(manager.session_snapshot("old").location, LOCAL_HBM)
        self.assertEqual(manager.session_snapshot("new").location, LOCAL_HBM)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 0, 1: 0})

        # Once the protected residency is no longer protected, E still needs
        # both whole-session victims: the first eviction clears bare KV
        # shortfall, while the second is required to clear the E-adjusted gap.
        retried = manager.ensure_physical_fit(0, (350,), 31, "retry")
        self.assertTrue(retried.admitted)
        self.assertEqual(
            [transfer.session_id for transfer in retried.evictions],
            ["old", "new"],
        )
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 640, 1: 0})

    def test_invalid_full_rank_update_is_atomic(self) -> None:
        manager = _manager()
        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 20, 1: 0})
        before = manager.workspace_capacity_snapshot()
        held_before = dict(manager._external_workspace_bytes_by_rank)

        # Rank 0 is a valid staged update; rank 1 is stale. Neither staged
        # value may become visible when full-table validation fails.
        with self.assertRaisesRegex(ValueError, "stale workspace revision"):
            manager.set_external_workspace_view(
                {0: 2, 1: 0}, {0: 21, 1: 0}
            )
        self.assertEqual(manager.workspace_capacity_snapshot(), before)
        self.assertEqual(manager._external_workspace_bytes_by_rank, held_before)

        # The same atomic guarantee applies when the late rank exceeds HBM.
        with self.assertRaisesRegex(ValueError, "exceeds Python HBM capacity"):
            manager.set_external_workspace_view(
                {0: 2, 1: 2}, {0: 21, 1: 10_000}
            )
        self.assertEqual(manager.workspace_capacity_snapshot(), before)
        self.assertEqual(manager._external_workspace_bytes_by_rank, held_before)

        with self.assertRaisesRegex(ValueError, "exactly match topology ranks"):
            manager.set_external_workspace_view({0: 2}, {0: 21})
        with self.assertRaisesRegex(ValueError, "exact uint64"):
            manager.set_external_workspace_view({0: True, 1: 1}, {0: 20, 1: 0})

    def test_capacity_snapshot_validates_uint64_fields_and_combined_headroom(self) -> None:
        manager = _manager()
        max_u64 = manager_module._UINT64_MAX
        state = manager._rank_states[0]
        state.capacity_bytes = max_u64
        state.model_weight_bytes = max_u64 - 1
        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 1, 1: 0})

        # Base usage plus workspace reaches the uint64 limit exactly.
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(snapshot["ranks"][0]["capacity_bytes"], max_u64)
        self.assertEqual(snapshot["ranks"][0]["model_weight_bytes"], max_u64 - 1)

        # Every emitted numeric field is checked, and sums are bounded before
        # the JSON-facing snapshot is constructed.
        state.reserved_request_bytes = 1
        with self.assertRaisesRegex(ValueError, "combined bytes exceed uint64"):
            manager.workspace_capacity_snapshot()
        state.reserved_request_bytes = 0
        state.capacity_bytes = max_u64 + 1
        with self.assertRaisesRegex(ValueError, "capacity bytes.*uint64"):
            manager.workspace_capacity_snapshot()

    def test_workspace_hold_reclaims_when_physical_headroom_alone_fits(self) -> None:
        # 128 B remains after two cold sessions, enough for a 100 B request.
        # A 64 B workspace hold makes the combined fit short, so the existing
        # whole-session LRU path must reclaim the oldest session.
        manager = _manager(extra_capacity=640 + 128)
        _seed_local(manager, "old", 10)
        _seed_local(manager, "new", 20)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 128)
        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 64, 1: 0})

        fit = manager.ensure_physical_fit(0, (100,), 30, "next")
        self.assertTrue(fit.admitted)
        self.assertFalse(fit.blocked_by_workspace)
        self.assertEqual([transfer.session_id for transfer in fit.evictions], ["old"])
        self.assertEqual(fit.evictions[0].kind, "remote_store")
        self.assertEqual(manager.session_snapshot("old").location, REMOTE_MEMORY)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 320, 1: 0})
        manager._check_invariants()

    def test_workspace_fit_continues_lru_after_physical_shortfall_clears(self) -> None:
        # The first 320 B victim clears the physical 350 B shortfall, but the
        # 64 B hold still leaves only 320 B of fit headroom. LRU must continue
        # to the second cold victim before admitting the request.
        manager = _manager(extra_capacity=640 + 64)
        _seed_local(manager, "old", 10)
        _seed_local(manager, "new", 20)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 64)
        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 64, 1: 0})

        fit = manager.ensure_physical_fit(0, (350,), 30, "next")
        self.assertTrue(fit.admitted)
        self.assertFalse(fit.blocked_by_workspace)
        self.assertEqual(
            [transfer.session_id for transfer in fit.evictions], ["old", "new"]
        )
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 640, 1: 0})

    def test_workspace_only_margin_block_preserves_cold_candidate_for_retry(self) -> None:
        # Per-rank KV capacity is 1600 B. The active session holds 960 B,
        # the cold candidate 320 B, and 320 B remain. Reclaiming the cold
        # session would give 640 B, enough for 480 B without a workspace hold;
        # a 240 B hold reduces that post-reclaim margin to 400 B.
        manager = _manager(extra_capacity=1600)
        active = manager.prepare_history(
            "active", 0, 0, 10, "active0", required_context_tokens=30
        )
        self.assertFalse(active.admission_blocked)
        self.assertTrue(manager.grow_prefill("active", 30, 11, "active0").admitted)
        _seed_local(manager, "cold", 20, tokens=10)
        self.assertEqual(manager.hbm_snapshots(0)[0].resident_kv_bytes, 1280)
        self.assertEqual(manager.hbm_snapshots(0)[0].remaining_bytes, 320)
        manager.set_external_workspace_view({0: 1, 1: 1}, {0: 240, 1: 0})

        blocked = manager.ensure_physical_fit(0, (480,), 30, "retry")
        self.assertFalse(blocked.admitted)
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.deep_gap_events, 0)
        self.assertEqual(manager.session_snapshot("cold").location, LOCAL_HBM)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 0, 1: 0})

        epoch = manager.set_external_workspace_view(
            {0: 2, 1: 2}, {0: 0, 1: 0}
        )
        self.assertEqual(epoch, (0, 1))
        retry = manager.ensure_physical_fit(0, (480,), 31, "retry")
        self.assertTrue(retry.admitted)
        self.assertEqual([transfer.session_id for transfer in retry.evictions], ["cold"])
        self.assertEqual(manager.deep_gap_events, 0)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 320, 1: 0})

    def test_workspace_blocked_remote_restore_keeps_pool_until_release(self) -> None:
        manager = _manager()
        _seed_local(manager, "remote", 10)
        eviction = manager.ensure_physical_fit(0, (1,), 11, "evict")
        self.assertTrue(eviction.admitted)
        self.assertEqual([transfer.kind for transfer in eviction.evictions], ["remote_store"])
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 320, 1: 0})

        # Rank 1 has enough physical headroom for the full restore but the
        # temporary one-byte hold blocks it. This must not debit remote pool
        # bytes or create a transfer object.
        epoch = manager.set_external_workspace_view(
            {0: 1, 1: 1}, {0: 0, 1: 1}
        )
        self.assertEqual(epoch, (0, 1))
        blocked = manager.prepare_history(
            "remote", 1, 10, 12, "remote1", required_context_tokens=10
        )
        self.assertTrue(blocked.admission_blocked)
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertEqual(blocked.action, REMOTE_RESTORE)
        self.assertEqual(blocked.transfers, ())
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.session_snapshot("remote").location, REMOTE_MEMORY)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 320, 1: 0})
        self.assertEqual(manager.deep_gap_events, 0)

        changed = manager.set_external_workspace_view(
            {0: 2, 1: 2}, {0: 0, 1: 0}
        )
        self.assertEqual(changed, (0, 1))
        restored = manager.prepare_history(
            "remote", 1, 10, 13, "remote1", required_context_tokens=10
        )
        self.assertFalse(restored.admission_blocked)
        self.assertEqual(restored.action, REMOTE_RESTORE)
        self.assertEqual([transfer.kind for transfer in restored.transfers], ["remote_load"])
        self.assertEqual(manager.session_snapshot("remote").location, LOCAL_HBM)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 0, 1: 0})

        manager.mark_complete("remote", 14, "remote1")
        repeated = manager.prepare_history(
            "remote", 1, 10, 15, "remote2", required_context_tokens=10
        )
        self.assertEqual(repeated.action, "LOCAL_HIT")
        self.assertEqual(repeated.transfers, ())
        self.assertEqual(
            sum(event.event_type == "remote_restore" for event in manager.events), 1
        )

    def test_permanent_floor_remote_restore_failure_keeps_remote_pool_unchanged(self) -> None:
        manager = _manager(
            prospective_workspace_bytes_by_rank={0: 0, 1: 64}
        )
        _seed_local(manager, "remote", 10)
        eviction = manager.ensure_physical_fit(0, (1,), 11, "evict")
        self.assertTrue(eviction.admitted)
        self.assertEqual(manager.session_snapshot("remote").location, REMOTE_MEMORY)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 320, 1: 0})

        # The P rank can hold the session only without its immutable floor.
        # Fail explicitly before remote pool debit or restore mutation; no
        # workspace revision can make this request fit.
        with self.assertRaisesRegex(
            ValueError, "cannot fit an empty instance after prospective workspace"
        ):
            manager.prepare_history(
                "remote", 1, 10, 12, "remote1", required_context_tokens=10
            )
        self.assertEqual(manager.session_snapshot("remote").location, REMOTE_MEMORY)
        self.assertEqual(manager.remote_bytes_by_rank(), {0: 320, 1: 0})
        self.assertEqual(manager.deep_gap_events, 0)

    def test_prospective_floor_survives_reservation_rollback_and_net_credit_swap(self) -> None:
        manager = _manager(
            extra_capacity=1000,
            prospective_workspace_bytes_by_rank={0: 64, 1: 0},
        )
        _seed_local(manager, "session", 10)
        final_shards = kv_cache_shard_bytes_for_tokens(
            manager.model, 20, manager.tp_degree
        )
        self.assertEqual(final_shards, (640,))
        credit = manager.session_snapshot("session").local_shard_bytes
        self.assertEqual(credit, (320,))

        # The gross reservation is workspace-only blocked because the old
        # 320 B resident prefix is protected. The net reservation fits with E.
        full = manager.reserve_request_capacity(
            "session-request-1", "session", 0, final_shards, 20
        )
        self.assertFalse(full.admitted)
        self.assertTrue(full.blocked_by_workspace)
        self.assertEqual(manager.hbm_snapshots(0)[0].reserved_request_bytes, 0)
        net_shards = tuple(
            max(0, final - have)
            for final, have in zip(final_shards, credit)
        )
        net = manager.reserve_request_capacity(
            "session-request-1", "session", 0, net_shards, 20
        )
        self.assertTrue(net.admitted)
        self.assertEqual(manager.hbm_snapshots(0)[0].reserved_request_bytes, 320)

        moved = manager.prepare_history(
            "session", 1, 10, 21, "session-request-1",
            required_context_tokens=20,
        )
        self.assertEqual(moved.source_instance_index, 0)
        self.assertEqual(moved.action, "NOC_MIGRATE")
        extended = manager.extend_request_capacity(
            "session-request-1", credit, 21,
            reason="history_vacated_decode_target",
        )
        self.assertEqual(extended.shard_bytes, final_shards)
        self.assertEqual(manager.hbm_snapshots(0)[0].resident_kv_bytes, 0)
        self.assertEqual(manager.hbm_snapshots(0)[0].reserved_request_bytes, 640)
        manager.release_request_capacity("session-request-1", 22)
        self.assertEqual(manager.hbm_snapshots(0)[0].reserved_request_bytes, 0)
        self.assertEqual(
            dict(manager.prospective_workspace_bytes_by_rank), {0: 64, 1: 0}
        )
        manager._check_invariants()

    def test_workspace_view_survives_kv_journal_rollback(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = MemoryActionRecorder(
                MemoryMetricsObserver(2),
                journal_path=Path(temp_dir) / "kv_delta_journal.jsonl",
            )
            self.addCleanup(recorder.close_journal)
            set_metrics_observer(recorder)
            manager = _manager(
                prospective_workspace_bytes_by_rank={0: 64, 1: 32}
            )
            manager.set_external_workspace_view(
                {0: 9, 1: 9}, {0: 8, 1: 0}
            )
            before = manager.workspace_capacity_snapshot()
            reconcile = manager._journal_reconcile_transaction

            def fail_reconcile(*_args):
                raise RuntimeError("forced reconcile failure")

            manager._journal_reconcile_transaction = fail_reconcile
            with self.assertRaisesRegex(RuntimeError, "forced reconcile failure"):
                manager.reserve_request_capacity("r0", "s0", 0, (1,), 1)
            manager._journal_reconcile_transaction = reconcile

            self.assertEqual(manager.workspace_capacity_snapshot(), before)
            self.assertEqual(manager.hbm_snapshots(0)[0].reserved_request_bytes, 0)
            self.assertEqual(manager._external_workspace_bytes_by_rank, {0: 8, 1: 0})
            self.assertEqual(
                dict(manager.prospective_workspace_bytes_by_rank),
                {0: 64, 1: 32},
            )

    def test_legacy_no_view_and_empty_instance_overflow_contract(self) -> None:
        manager = _manager()
        # With no external caller, the absolute view is zero and physical-fit
        # checks retain the previous behavior.
        self.assertEqual(manager._fit_remaining(0), manager._remaining(0))
        fit = manager.ensure_physical_fit(0, (10,), 1, "r0")
        self.assertTrue(fit.admitted)
        with self.assertRaisesRegex(ValueError, "request cannot fit an empty instance"):
            manager.ensure_physical_fit(0, (321,), 2, "too-large")


if __name__ == "__main__":
    unittest.main()
