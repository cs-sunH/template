#!/usr/bin/env python3
"""Workspace v2 bridge and external KV fit integration tests."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.decision_bridge import BridgeError, BridgeServer  # noqa: E402
from online.online_service import _workspace_admission_mode  # noqa: E402
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402
from online.wsc_llm_online_scheduler import WscLlmOnlineScheduler  # noqa: E402
from online.test_admission_net_credit import _real_manager  # noqa: E402


def _workspace_state(manager, revisions=None, active_bytes=None):
    revisions = revisions or {}
    active_bytes = active_bytes or {}
    return {
        "protocol_version": 1,
        "ranks": [
            {
                "rank": rank,
                "workspace_revision": revisions.get(rank, 0),
                "active_workspace_bytes": active_bytes.get(rank, 0),
            }
            for rank in manager.rank_ids
        ],
    }


def _delta(manager, seq, tick, revisions=None, active_bytes=None):
    return {
        "schema_version": 2,
        "delivery_sequence": seq,
        "delivery_epoch": seq,
        "tick": tick,
        "deferred_from_tick": tick,
        "reasons": ["TICK"],
        "arrivals": [],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": seq, "tick": tick},
        "workspace_state": _workspace_state(
            manager, revisions=revisions, active_bytes=active_bytes),
    }


class _WorkspaceGateScheduler(OnlineSchedulerBase):
    """Base pipeline with the production WSC workspace extension hooks."""

    _apply_delivery_extension = WscLlmOnlineScheduler._apply_delivery_extension
    _attach_response_extension = WscLlmOnlineScheduler._attach_response_extension
    _note_capacity_change = WscLlmOnlineScheduler._note_capacity_change

    def run_variant_policy(self, delta):
        pass


def _scheduler(manager):
    scheduler = _WorkspaceGateScheduler(
        manifest={"requests": []},
        config=SimpleNamespace(),
        workspace_admission="required",
    )
    scheduler.kv_manager = manager
    # 2026-10-20 方案二:_apply_delivery_extension 逐 rank 峰值簿记需要
    # 该属性(生产路径由 WscLlmOnlineScheduler.__init__ 初始化)。
    scheduler._workspace_peak_bytes_by_rank = {}
    scheduler._workspace_rank_to_instance = {
        rank: snapshot.instance_index
        for snapshot in manager.hbm_snapshots()
        for rank in (snapshot.rank,)
    }
    scheduler.capacity_epoch = [0 for _ in manager.topology.instances]
    return scheduler


def _tiny_capacity_manager():
    """Real one-rank-TP manager with exact 100-byte HBM capacity."""
    from online.test_admission_net_credit import (
        WscLlmHardware,
        WscLlmInstanceSpec,
        WscLlmModel,
        build_instances,
    )
    from session_kv_manager import SessionKVCacheManager

    hardware = WscLlmHardware(
        mesh_rows=1, mesh_cols=2, local_hbm_capacity_bytes=100,
        local_hbm_bandwidth_gbps=1.0, d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)
    topology = build_instances(
        hardware,
        (
            WscLlmInstanceSpec("prefill", "1", (0,), "prefill"),
            WscLlmInstanceSpec("decode", "2", (1,), "decode"),
        ))
    model = WscLlmModel(1, 2, 2, 1, 2, 1, "gelu")
    manager = SessionKVCacheManager(topology, model, strict_invariants=True)
    assert manager.model_weight_bytes_by_tp_rank == (40,)
    return manager


def _set_workspace(manager, rank, revision, active_bytes):
    revisions = {item: 0 for item in manager.rank_ids}
    active = {item: 0 for item in manager.rank_ids}
    revisions[rank] = revision
    active[rank] = active_bytes
    return manager.set_external_workspace_view(revisions, active)


class ExternalWorkspaceManagerTest(unittest.TestCase):
    def test_hold_blocks_temporarily_then_release_restores_fit(self):
        manager = _real_manager()
        rank = manager.topology.instance(0).ranks[0]
        before = manager.hbm_snapshots()
        rank_before = next(row for row in before if row.rank == rank)
        self.assertGreater(rank_before.remaining_bytes, 1)
        held_bytes = rank_before.remaining_bytes
        revisions = {item: 0 for item in manager.rank_ids}
        active = {item: 0 for item in manager.rank_ids}
        revisions[rank] = 1
        active[rank] = held_bytes

        changed = manager.set_external_workspace_view(revisions, active)
        self.assertEqual(changed, (rank,))
        fit = manager.ensure_physical_fit(
            0, (1, 0, 0, 0), 1, "workspace-blocked")
        self.assertFalse(fit.admitted)
        self.assertTrue(fit.blocked_by_workspace)
        self.assertFalse(fit.permanently_infeasible)
        self.assertEqual(fit.evictions, ())
        self.assertEqual(manager.hbm_snapshots(), before)
        self.assertEqual(manager.events, ())

        # The transient workspace hold is not part of the old KV snapshot or
        # KV journal.  The v2 response exposes the Python-owned base only.
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(snapshot["protocol_version"], 1)
        self.assertEqual(
            set(snapshot["ranks"][0]),
            {"rank", "workspace_revision", "capacity_bytes",
             "model_weight_bytes", "resident_kv_bytes",
             "reserved_request_bytes"})
        self.assertEqual(
            {row.rank: row.used_bytes for row in before},
            {row["rank"]: row["model_weight_bytes"] +
             row["resident_kv_bytes"] + row["reserved_request_bytes"]
             for row in snapshot["ranks"]})

        revisions[rank] = 2
        active[rank] = 0
        self.assertEqual(
            manager.set_external_workspace_view(revisions, active), (rank,))
        fit_after_release = manager.ensure_physical_fit(
            0, (1, 0, 0, 0), 2, "workspace-released")
        self.assertTrue(fit_after_release.admitted)
        self.assertFalse(fit_after_release.blocked_by_workspace)

    def test_public_reservation_reports_workspace_wait_without_eviction(self):
        manager = _real_manager()
        rank = manager.topology.instance(0).ranks[0]
        row = next(item for item in manager.hbm_snapshots() if item.rank == rank)
        revisions = {item: 0 for item in manager.rank_ids}
        active = {item: 0 for item in manager.rank_ids}
        revisions[rank] = 1
        active[rank] = row.remaining_bytes
        manager.set_external_workspace_view(revisions, active)

        result = manager.reserve_request_capacity(
            "workspace-wait", "workspace-session", 0, (1, 0, 0, 0), 1)
        self.assertFalse(result.admitted)
        self.assertTrue(result.blocked_by_workspace)
        self.assertFalse(result.permanently_infeasible)
        self.assertEqual(result.evictions, ())
        self.assertNotIn("workspace-wait", manager._reservations)
        self.assertEqual(manager.events, ())

    def test_workspace_impossible_even_after_evictions_waits_without_deleting_kv(self):
        manager = _real_manager()
        manager.prepare_history("candidate", 0, 0, 1, "seed")
        self.assertTrue(manager.grow_prefill("candidate", 70, 2, "seed").admitted)
        manager.mark_complete("candidate", 3, "seed")

        rank = manager.topology.instance(0).ranks[0]
        before = next(row for row in manager.hbm_snapshots() if row.rank == rank)
        revisions = {item: 0 for item in manager.rank_ids}
        active = {item: 0 for item in manager.rank_ids}
        revisions[rank] = 1
        active[rank] = before.remaining_bytes
        manager.set_external_workspace_view(revisions, active)
        events_before = manager.events

        # This request fits on an empty rank without the hold (600 <= 896),
        # but the current hold leaves only 560 bytes even after all KV is
        # removed. The eligible cold session must remain resident until a
        # later capacity revision makes the request possible.
        blocked = manager.ensure_physical_fit(
            0, (600, 0, 0, 0), 4, "workspace-unfit-empty")
        self.assertFalse(blocked.admitted)
        self.assertTrue(blocked.blocked_by_workspace)
        self.assertFalse(blocked.permanently_infeasible)
        self.assertEqual(blocked.evictions, ())
        self.assertEqual(manager.session_snapshot("candidate").state, "RESIDENT")
        self.assertEqual(manager.events, events_before)

        revisions[rank] = 2
        active[rank] = 0
        manager.set_external_workspace_view(revisions, active)
        retried = manager.ensure_physical_fit(
            0, (600, 0, 0, 0), 5, "workspace-released")
        self.assertTrue(retried.admitted)
        self.assertEqual(len(retried.evictions), 1)

    def test_capacity_100_base_60_workspace_30_blocks_reserve_25_across_epochs(self):
        manager = _tiny_capacity_manager()
        manager.prepare_history("base", 0, 0, 1, "seed")
        self.assertTrue(manager.grow_prefill("base", 5, 2, "seed").admitted)
        rank = 0
        snapshot = next(row for row in manager.hbm_snapshots() if row.rank == rank)
        self.assertEqual(snapshot.used_bytes, 60)
        self.assertEqual(snapshot.capacity_bytes, 100)

        _set_workspace(manager, rank, 1, 30)
        first = manager.reserve_request_capacity(
            "reserve-0", "new-0", 0, (25,), 3)
        self.assertFalse(first.admitted)
        self.assertTrue(first.blocked_by_workspace)
        self.assertFalse(first.permanently_infeasible)
        self.assertEqual(first.evictions, ())
        self.assertEqual(manager.hbm_snapshots()[0].used_bytes, 60)
        self.assertEqual(60 + 30 + 25, 115)  # would overcommit this rank

        # The next delivery advances the absolute workspace revision without
        # changing its 30-byte lease; retry still sees the Python base of 60.
        _set_workspace(manager, rank, 2, 30)
        second = manager.reserve_request_capacity(
            "reserve-1", "new-1", 0, (25,), 4)
        self.assertFalse(second.admitted)
        self.assertTrue(second.blocked_by_workspace)
        self.assertEqual(manager.hbm_snapshots()[0].used_bytes, 60)

        _set_workspace(manager, rank, 3, 0)
        released = manager.reserve_request_capacity(
            "reserve-2", "new-2", 0, (25,), 5)
        self.assertTrue(released.admitted)
        rank_snapshot = manager.hbm_snapshots()[0]
        self.assertEqual(rank_snapshot.used_bytes, 85)
        self.assertLessEqual(rank_snapshot.used_bytes, 100)

    def test_workspace_only_deficit_evicts_minimal_existing_victim_prefix(self):
        manager = _tiny_capacity_manager()
        manager.prepare_history("cold-old", 0, 0, 1, "seed-old")
        self.assertTrue(manager.grow_prefill(
            "cold-old", 3, 1, "seed-old").admitted)
        manager.mark_complete("cold-old", 2, "seed-old")
        manager.prepare_history("cold-new", 0, 0, 3, "seed-new")
        self.assertTrue(manager.grow_prefill(
            "cold-new", 1, 4, "seed-new").admitted)
        manager.mark_complete("cold-new", 5, "seed-new")

        # The KV base has 44 bytes free for a 30-byte request. A 20-byte
        # external hold leaves only 24 bytes, so the deficit comes only from
        # the workspace. Reclaiming the oldest 12-byte cold session is enough;
        # the newer 4-byte session must stay resident.
        rank = 0
        before = next(row for row in manager.hbm_snapshots() if row.rank == rank)
        self.assertEqual(before.remaining_bytes, 44)
        self.assertEqual(manager._insufficient(0, (30,)), ())
        _set_workspace(manager, rank, 1, 20)
        self.assertEqual(manager._fit_insufficient(0, (30,)), (rank,))

        result = manager.ensure_physical_fit(
            0, (30,), 6, "workspace-only-cold-reclaim")
        self.assertTrue(result.admitted)
        self.assertEqual(
            tuple(item.victim_session_id for item in result.evictions),
            ("cold-old",))
        self.assertEqual(manager.session_snapshot("cold-old").state, "EVICTED")
        self.assertEqual(manager.session_snapshot("cold-new").state, "RESIDENT")
        after = next(row for row in manager.hbm_snapshots() if row.rank == rank)
        self.assertEqual(after.remaining_bytes, 56)
        self.assertLessEqual(after.used_bytes + 20 + 30, after.capacity_bytes)

    def test_history_grow_prefill_grow_decode_and_move_use_workspace_fit(self):
        history_manager = _tiny_capacity_manager()
        _set_workspace(history_manager, 0, 1, 60)
        history = history_manager.prepare_history(
            "history", 0, 0, 1, "history-request",
            required_context_tokens=1)
        self.assertTrue(history.admission_blocked)
        self.assertTrue(history.blocked_by_workspace)
        self.assertFalse(history.permanently_infeasible)
        self.assertIsNone(history_manager.session_snapshot("history"))

        prefill_manager = _tiny_capacity_manager()
        prefill_manager.prepare_history("prefill", 0, 0, 1, "seed")
        _set_workspace(prefill_manager, 0, 1, 60)
        prefill_before = prefill_manager.hbm_snapshots()
        prefill = prefill_manager.grow_prefill("prefill", 1, 2, "grow")
        self.assertFalse(prefill.admitted)
        self.assertTrue(prefill.blocked_by_workspace)
        self.assertEqual(prefill_manager.hbm_snapshots(), prefill_before)

        decode_manager = _tiny_capacity_manager()
        decode_manager.prepare_history("decode", 1, 0, 1, "seed")
        _set_workspace(decode_manager, 1, 1, 60)
        decode_before = decode_manager.hbm_snapshots()
        decode = decode_manager.grow_decode("decode", 1, 2, "grow")
        self.assertFalse(decode.admitted)
        self.assertTrue(decode.blocked_by_workspace)
        self.assertEqual(decode_manager.hbm_snapshots(), decode_before)

        move_manager = _tiny_capacity_manager()
        move_manager.prepare_history("move", 0, 0, 1, "seed")
        self.assertTrue(move_manager.grow_prefill("move", 5, 2, "seed").admitted)
        _set_workspace(move_manager, 1, 1, 60)
        move_before = move_manager.hbm_snapshots()
        move = move_manager.move_prefill_to_decode(
            "move", 1, 3, "move-request", final_context_tokens=5)
        self.assertTrue(move.admission_blocked)
        self.assertTrue(move.blocked_by_workspace)
        self.assertFalse(move.permanently_infeasible)
        self.assertIsNone(move.transfer)
        self.assertEqual(move_manager.hbm_snapshots(), move_before)

    def test_actual_prefill_epoch_retries_after_workspace_release(self):
        from online.wsc_llm_online_scheduler import _OnlineRequestRuntime

        manager = _tiny_capacity_manager()
        # Non-workspace base on Decode: weight=40 plus another request's
        # reservation=20, leaving a 24-byte final KV request blocked by a
        # 30-byte workspace hold (base headroom 40, fit headroom 10).
        self.assertTrue(manager.reserve_request_capacity(
            "other-request", "other-session", 1, (20,), 1).admitted)
        scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
        scheduler.workspace_admission = "required"
        scheduler.kv_manager = manager
        scheduler.topology = manager.topology
        scheduler.config = SimpleNamespace(model=manager.model)
        scheduler.capacity_epoch = [0, 0]
        # 2026-10-20 方案二:峰值簿记属性(__init__ 未跑的 __new__ 实例
        # 手动补齐,与 _scheduler() 同款)。
        scheduler._workspace_peak_bytes_by_rank = {}
        scheduler._workspace_rank_to_instance = {
            rank: instance.index
            for instance in manager.topology.instances
            for rank in instance.ranks
        }
        runtime = _OnlineRequestRuntime({
            "request_id": "retry-request", "session_id": "retry-session",
            "turn_index": 0, "queue_index": 0, "prefill_length": 1,
            "decode_length": 1, "history_tokens_before": 0,
            "prefill_context_tokens": 6, "final_context_tokens": 6,
        })
        runtime.prefill_instance_index = 0
        runtime.static_route = SimpleNamespace(decode_instance_index=1)

        rank = 1
        scheduler._apply_delivery_extension(_delta(
            manager, 0, 100, revisions={rank: 1}, active_bytes={rank: 30}))
        self.assertFalse(scheduler._try_admit_prefill(runtime, 2))
        first_attempt_epoch = runtime.prefill_attempt_epoch
        self.assertEqual(first_attempt_epoch, (0, 1))
        self.assertFalse(runtime.decode_capacity_reserved)

        scheduler._apply_delivery_extension(_delta(
            manager, 1, 101, revisions={rank: 2}, active_bytes={rank: 0}))
        self.assertEqual(scheduler.capacity_epoch[1], 2)
        self.assertNotEqual(
            (scheduler.capacity_epoch[0], scheduler.capacity_epoch[1]),
            first_attempt_epoch)
        self.assertTrue(scheduler._try_admit_prefill(runtime, 3))
        self.assertEqual(runtime.prefill_attempt_epoch, (0, 2))
        self.assertTrue(runtime.admitted_prefill)
        self.assertTrue(runtime.decode_capacity_reserved)
        rank_after = next(
            row for row in manager.hbm_snapshots() if row.rank == rank)
        self.assertEqual(
            manager._reservations["retry-request"].shard_bytes, (24,))
        self.assertEqual(rank_after.reserved_request_bytes, 44)
        self.assertLessEqual(rank_after.used_bytes, rank_after.capacity_bytes)

    def test_revision_and_rank_validation_are_atomic_and_infeasible_is_distinct(self):
        manager = _real_manager()
        ranks = manager.rank_ids
        revisions = {rank: 1 for rank in ranks}
        active = {rank: 0 for rank in ranks}
        manager.set_external_workspace_view(revisions, active)
        before = manager.workspace_capacity_snapshot()

        stale = dict(revisions)
        stale[ranks[0]] = 0
        with self.assertRaisesRegex(ValueError, "stale workspace revision"):
            manager.set_external_workspace_view(stale, active)
        same_revision_different_bytes = dict(active)
        same_revision_different_bytes[ranks[0]] = 1
        with self.assertRaisesRegex(ValueError, "without revision advance"):
            manager.set_external_workspace_view(
                revisions, same_revision_different_bytes)
        with self.assertRaisesRegex(ValueError, "exactly match topology"):
            manager.set_external_workspace_view(
                {rank: 1 for rank in ranks[:-1]}, active)
        with self.assertRaisesRegex(ValueError, "uint64"):
            bad = dict(active)
            bad[ranks[0]] = True
            manager.set_external_workspace_view(revisions, bad)
        rank_state = next(
            row for row in manager.hbm_snapshots() if row.rank == ranks[0])
        over_capacity_revisions = {rank: 2 for rank in ranks}
        over_capacity_bytes = {rank: 0 for rank in ranks}
        over_capacity_bytes[ranks[0]] = rank_state.remaining_bytes + 1
        with self.assertRaisesRegex(
                ValueError, "workspace hold exceeds Python HBM capacity"):
            manager.set_external_workspace_view(
                over_capacity_revisions, over_capacity_bytes)
        self.assertEqual(manager.workspace_capacity_snapshot(), before)

        too_large = list((0, 0, 0, 0))
        too_large[0] = rank_state.capacity_bytes - rank_state.model_weight_bytes + 1
        result = manager.ensure_physical_fit(
            0, too_large, 3, "permanently-infeasible")
        self.assertFalse(result.admitted)
        self.assertFalse(result.blocked_by_workspace)
        self.assertTrue(result.permanently_infeasible)


class WorkspaceV2BridgeTest(unittest.TestCase):
    def test_required_bridge_runs_real_manager_snapshot_and_v2_ack(self):
        manager = _real_manager()
        scheduler = _scheduler(manager)
        rank = manager.topology.instance(0).ranks[0]
        available = next(
            row.remaining_bytes for row in manager.hbm_snapshots()
            if row.rank == rank)
        delta = _delta(
            manager, 0, 100,
            revisions={rank: 1}, active_bytes={rank: available})

        with tempfile.TemporaryDirectory(prefix="workspace_bridge_v2.") as bridge:
            server = BridgeServer(bridge, workspace_admission="required")
            request_path = os.path.join(bridge, "request_0.json")
            with open(request_path, "w", encoding="utf-8") as target:
                json.dump(delta, target, separators=(",", ":"))
            server._handle_request(0, scheduler.on_decision_batch)

            response_path = os.path.join(bridge, "response_0.json")
            with open(response_path, "r", encoding="utf-8") as source:
                response = json.load(source)
            self.assertEqual(response["schema_version"], 2)
            self.assertEqual(response["source_delivery_sequence"], 0)
            self.assertEqual(response["batch_id"], 0)
            rank_view = next(
                row for row in response["workspace_capacity_snapshot"]["ranks"]
                if row["rank"] == rank)
            self.assertEqual(rank_view["workspace_revision"], 1)
            self.assertEqual(scheduler.capacity_epoch[0], 1)

            # The policy and response are cached together.  Replaying the same
            # delivery does not apply the workspace revision twice.
            reply = scheduler.on_decision_batch(delta)
            self.assertEqual(reply, response)
            self.assertEqual(scheduler.capacity_epoch[0], 1)

            ack_path = os.path.join(bridge, "commit_ack_0.json")
            bad_acks = [
                {"schema_version": 2, "delivery_sequence": 0,
                 "batch_id": 0},
                {"schema_version": 2, "delivery_sequence": 0,
                 "batch_id": 0, "success": 1.0},
                {"schema_version": 2, "delivery_sequence": 0,
                 "batch_id": 0, "success": 0},
                {"schema_version": 2, "delivery_sequence": 0,
                 "batch_id": 0, "success": False},
            ]
            for bad_ack in bad_acks:
                with open(ack_path, "w", encoding="utf-8") as target:
                    json.dump(bad_ack, target)
                with self.assertRaisesRegex(BridgeError, "v2 ack success"):
                    server._handle_ack(0, scheduler.on_commit_ack)
                self.assertEqual(scheduler.ack_count, 0)
                self.assertEqual(server._ack_sequences.next_seq, 0)
                self.assertIsNotNone(scheduler._delivery_reply_cache)

            with open(ack_path, "w", encoding="utf-8") as target:
                json.dump({"schema_version": 2, "delivery_sequence": 0,
                           "batch_id": 0, "success": True}, target)
            cached_reply = scheduler._delivery_reply_cache["batch"]
            cached_workspace_snapshot = cached_reply[
                "workspace_capacity_snapshot"]
            cached_snapshot_before_ack = json.loads(json.dumps(
                cached_workspace_snapshot))
            manager_snapshot_before_ack = manager.workspace_capacity_snapshot()
            server._handle_ack(0, scheduler.on_commit_ack)
            server._handle_ack(0, scheduler.on_commit_ack)
            self.assertEqual(scheduler.ack_count, 1)
            self.assertIsNone(scheduler._delivery_reply_cache)
            self.assertEqual(manager.workspace_capacity_snapshot(),
                             manager_snapshot_before_ack)
            self.assertEqual(
                cached_workspace_snapshot, cached_snapshot_before_ack)
            self.assertEqual(
                cached_snapshot_before_ack,
                response["workspace_capacity_snapshot"])
            server._close_audit_streams()

    def test_required_parser_rejects_missing_and_stale_view_before_mutation(self):
        manager = _real_manager()
        scheduler = _scheduler(manager)
        first = _delta(manager, 0, 100, revisions={0: 1})
        scheduler.on_decision_batch(first)
        scheduler.on_commit_ack({
            "schema_version": 2, "delivery_sequence": 0, "batch_id": 0,
            "success": True})
        before = manager.workspace_capacity_snapshot()
        state_before = (scheduler.delivery_count,
                        scheduler.last_applied_sequence,
                        tuple(scheduler.capacity_epoch))

        malformed = _delta(manager, 1, 101)
        malformed.pop("workspace_state")
        with self.assertRaisesRegex(ValueError, "needs workspace_state"):
            scheduler.on_decision_batch(malformed)
        self.assertEqual(manager.workspace_capacity_snapshot(), before)
        self.assertEqual(
            (scheduler.delivery_count, scheduler.last_applied_sequence,
             tuple(scheduler.capacity_epoch)), state_before)

        stale = _delta(manager, 1, 101, revisions={0: 0})
        with self.assertRaisesRegex(ValueError, "stale workspace revision"):
            scheduler.on_decision_batch(stale)
        self.assertEqual(manager.workspace_capacity_snapshot(), before)
        self.assertEqual(
            (scheduler.delivery_count, scheduler.last_applied_sequence,
             tuple(scheduler.capacity_epoch)), state_before)

    def test_required_parser_rejects_invalid_rank_table_before_mutation(self):
        manager = _real_manager()
        scheduler = _scheduler(manager)
        scheduler.on_decision_batch(_delta(manager, 0, 100))
        scheduler.on_commit_ack({
            "schema_version": 2, "delivery_sequence": 0, "batch_id": 0,
            "success": True})

        malformed_views = []
        bool_rank = _delta(manager, 1, 101)
        bool_rank["workspace_state"]["ranks"][0]["rank"] = True
        malformed_views.append((bool_rank, "uint64"))
        float_bytes = _delta(manager, 1, 101)
        float_bytes["workspace_state"]["ranks"][0][
            "active_workspace_bytes"] = 0.0
        malformed_views.append((float_bytes, "uint64"))
        overflow = _delta(manager, 1, 101)
        overflow["workspace_state"]["ranks"][0][
            "workspace_revision"] = 1 << 64
        malformed_views.append((overflow, "uint64"))
        missing_rank = _delta(manager, 1, 101)
        missing_rank["workspace_state"]["ranks"].pop()
        malformed_views.append((missing_rank, "every topology rank"))
        duplicate_rank = _delta(manager, 1, 101)
        duplicate_rank["workspace_state"]["ranks"][1]["rank"] = 0
        malformed_views.append((duplicate_rank, "ascending and unique"))
        out_of_order = _delta(manager, 1, 101)
        ranks = out_of_order["workspace_state"]["ranks"]
        ranks[0], ranks[1] = ranks[1], ranks[0]
        malformed_views.append((out_of_order, "ascending and unique"))

        manager_before = manager.workspace_capacity_snapshot()
        scheduler_before = (scheduler.delivery_count,
                            scheduler.last_applied_sequence,
                            tuple(scheduler.capacity_epoch))
        for delta, message in malformed_views:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    scheduler.on_decision_batch(delta)
                self.assertEqual(
                    manager.workspace_capacity_snapshot(), manager_before)
                self.assertEqual(
                    (scheduler.delivery_count,
                     scheduler.last_applied_sequence,
                     tuple(scheduler.capacity_epoch)), scheduler_before)

    def test_v2_required_and_v1_off_reject_mismatched_peers_before_handler(self):
        with tempfile.TemporaryDirectory(prefix="workspace_mode_gate.") as bridge:
            called = []
            required = BridgeServer(bridge, workspace_admission="required")
            path = os.path.join(bridge, "request_0.json")
            with open(path, "w", encoding="utf-8") as target:
                json.dump({"schema_version": 1, "delivery_sequence": 0}, target)
            with self.assertRaisesRegex(BridgeError, "schema_version"):
                required._handle_request(0, lambda request: called.append(request))
            self.assertEqual(called, [])
            required._close_audit_streams()

        with tempfile.TemporaryDirectory(prefix="workspace_mode_off.") as bridge:
            called = []
            off = BridgeServer(bridge, workspace_admission="off")
            path = os.path.join(bridge, "request_0.json")
            with open(path, "w", encoding="utf-8") as target:
                json.dump({"schema_version": 2, "delivery_sequence": 0,
                           "workspace_state": {"protocol_version": 1,
                                                "ranks": []}}, target)
            with self.assertRaisesRegex(BridgeError, "schema_version"):
                off._handle_request(0, lambda request: called.append(request))
            self.assertEqual(called, [])
            off._close_audit_streams()


class WorkspaceModeResolutionTest(unittest.TestCase):
    def _config(self, profile):
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False)
        json.dump({"die-model-profile": profile}, handle)
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name)
                        and os.unlink(handle.name))
        return SimpleNamespace(system_config=handle.name)

    def test_auto_selects_required_only_for_coarse_profile(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SH_WORKSPACE_ADMISSION", None)
            self.assertEqual(
                _workspace_admission_mode(self._config("coarse_die_v2")),
                "required")
            self.assertEqual(
                _workspace_admission_mode(self._config("mesh_die_v1")),
                "off")

    def test_explicit_off_fails_for_coarse_profile(self):
        with patch.dict(os.environ, {"SH_WORKSPACE_ADMISSION": "off"}):
            with self.assertRaisesRegex(ValueError, "coarse_die_v2 requires"):
                _workspace_admission_mode(self._config("coarse_die_v2"))

    def test_explicit_required_rejects_legacy_profile(self):
        with patch.dict(os.environ, {"SH_WORKSPACE_ADMISSION": "required"}):
            with self.assertRaisesRegex(ValueError, "unsupported for legacy"):
                _workspace_admission_mode(self._config("mesh_die_v1"))


if __name__ == "__main__":
    unittest.main()
