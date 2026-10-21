#!/usr/bin/env python3
"""Focused LRU scheduler/bridge tests for the workspace v2 contract."""
import json
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
_UINT64_MAX = (1 << 64) - 1
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from online.decision_bridge import BridgeError, BridgeServer  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from online.online_service import _workspace_admission_mode  # noqa: E402
from online.wsc_llm_online_scheduler import (  # noqa: E402
    WscLlmOnlineScheduler,
    _OnlineRequestRuntime,
)
from wsc_llm_scheduler import (  # noqa: E402
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
)


def _test_config(capacity_bytes=100):
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
    model = WscLlmModel(1, 2, 2, 1, 2, 1, "gelu")
    groups = [
        WscLlmInstanceSpec("prefill", "1", (0,), "prefill"),
        WscLlmInstanceSpec("decode", "2", (1,), "decode"),
    ]
    return SimpleNamespace(
        npus_count=2,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=groups,
        prefill_chunk_size=128,
        layers=model.layers,
        hidden_size=model.hidden_size,
        ffn_size=model.ffn_size,
        vocab_size=model.vocab_size,
        bytes_per_elem=model.bytes_per_elem,
        num_heads=model.num_heads,
        num_heads_for_transfer=model.num_heads,
        mlp_variant=model.mlp_variant,
        request_queue=[],
        model=model,
        hardware=hardware,
        kv_cache_policy="session_lru_tiered",
    )


def _scheduler(*, workspace_admission="required", capacity_bytes=100):
    config = _test_config(capacity_bytes)
    graph = GraphBatchBuilder(config)
    return WscLlmOnlineScheduler(
        manifest={"requests": []},
        config=config,
        graph=graph,
        workspace_admission=workspace_admission,
    )


def _delta(scheduler, sequence, tick, active_by_rank=None, revision_by_rank=None,
           *, schema_version=2):
    active_by_rank = active_by_rank or {}
    revision_by_rank = revision_by_rank or {}
    delta = {
        "schema_version": schema_version,
        "delivery_sequence": sequence,
        "delivery_epoch": sequence,
        "tick": tick,
        "deferred_from_tick": tick,
        "reasons": ["TICK"],
        "arrivals": [],
        "completed_groups": [],
        "completed_nodes": [],
        "affected_ranks": [],
        "snapshot_handle": {"epoch": sequence, "tick": tick},
    }
    if schema_version == 2:
        delta["workspace_state"] = {
            "protocol_version": 1,
            "ranks": [
                {
                    "rank": rank,
                    "workspace_revision": revision_by_rank.get(rank, 0),
                    "active_workspace_bytes": active_by_rank.get(rank, 0),
                }
                for rank in scheduler.kv_manager.rank_ids
            ],
        }
    return delta


class WorkspaceAdmissionV2SchedulerBridgeTests(unittest.TestCase):
    def test_v2_full_rank_echo_replay_and_ack(self):
        scheduler = _scheduler()
        ranks = scheduler.kv_manager.rank_ids
        delta = _delta(
            scheduler, 0, 100,
            revision_by_rank={rank: _UINT64_MAX for rank in ranks},
        )
        with tempfile.TemporaryDirectory(prefix="lru_workspace_v2.") as bridge:
            server = BridgeServer(bridge, workspace_admission="required")
            with open(os.path.join(bridge, "request_0.json"), "w",
                      encoding="utf-8") as target:
                json.dump(delta, target, separators=(",", ":"))
            server._handle_request(0, scheduler.on_decision_batch)

            with open(os.path.join(bridge, "response_0.json"), "r",
                      encoding="utf-8") as source:
                response = json.load(source)
            self.assertEqual(response["schema_version"], 2)
            self.assertEqual(response["batch_id"], 0)
            self.assertEqual(response["source_delivery_sequence"], 0)
            self.assertEqual(response["touched_ranks"], [])
            snapshot = response["workspace_capacity_snapshot"]
            self.assertEqual(
                snapshot, scheduler.kv_manager.workspace_capacity_snapshot())
            self.assertEqual(snapshot["protocol_version"], 1)
            self.assertEqual(set(snapshot), {"protocol_version", "ranks"})
            self.assertEqual(
                [row["rank"] for row in snapshot["ranks"]], list(ranks))
            self.assertEqual(
                [row["workspace_revision"] for row in snapshot["ranks"]],
                [_UINT64_MAX] * len(ranks),
            )
            expected_rank_fields = {
                "rank", "workspace_revision", "capacity_bytes",
                "model_weight_bytes", "resident_kv_bytes",
                "reserved_request_bytes",
            }
            for row in snapshot["ranks"]:
                self.assertEqual(set(row), expected_rank_fields)
                for value in row.values():
                    self.assertIs(type(value), int)
                    self.assertGreaterEqual(value, 0)
                    self.assertLessEqual(value, _UINT64_MAX)
            self.assertEqual(scheduler.capacity_epoch, [1, 1])

            # A byte-identical duplicate is handled by the delivery replay
            # cache and does not install or account the absolute view twice.
            self.assertEqual(scheduler.on_decision_batch(delta), response)
            self.assertEqual(scheduler.capacity_epoch, [1, 1])
            self.assertEqual(scheduler.delivery_count, 1)

            ack_path = os.path.join(bridge, "commit_ack_0.json")
            with open(ack_path, "w", encoding="utf-8") as target:
                json.dump({
                    "schema_version": 2,
                    "delivery_sequence": 0,
                    "batch_id": 0,
                    "success": 1.0,
                }, target)
            with self.assertRaisesRegex(BridgeError, "v2 ack success"):
                server._handle_ack(0, scheduler.on_commit_ack)
            self.assertEqual(scheduler.ack_count, 0)
            self.assertEqual(server._ack_sequences.next_seq, 0)

            with open(ack_path, "w", encoding="utf-8") as target:
                json.dump({
                    "schema_version": 2,
                    "delivery_sequence": 0,
                    "batch_id": 0,
                    "success": True,
                }, target)
            server._handle_ack(0, scheduler.on_commit_ack)
            self.assertEqual(scheduler.ack_count, 1)
            server._close_audit_streams()

    def test_workspace_table_u64_and_atomic_rejection_before_delivery(self):
        scheduler = _scheduler()
        manager = scheduler.kv_manager
        initial_workspace = manager.workspace_capacity_snapshot()
        initial_epochs = tuple(scheduler.capacity_epoch)
        malformed = []

        missing_rank = _delta(scheduler, 0, 100)
        missing_rank["workspace_state"]["ranks"].pop()
        malformed.append((missing_rank, "every topology rank"))
        bool_rank = _delta(scheduler, 0, 100)
        bool_rank["workspace_state"]["ranks"][0]["rank"] = True
        malformed.append((bool_rank, "uint64"))
        float_bytes = _delta(scheduler, 0, 100)
        float_bytes["workspace_state"]["ranks"][0][
            "active_workspace_bytes"] = 0.0
        malformed.append((float_bytes, "uint64"))
        overflow = _delta(scheduler, 0, 100)
        overflow["workspace_state"]["ranks"][0][
            "workspace_revision"] = 1 << 64
        malformed.append((overflow, "uint64"))

        for delta, message in malformed:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    scheduler.on_decision_batch(delta)
                self.assertEqual(
                    manager.workspace_capacity_snapshot(), initial_workspace)
                self.assertEqual(tuple(scheduler.capacity_epoch), initial_epochs)
                self.assertEqual(scheduler.delivery_count, 0)
                self.assertEqual(scheduler.last_applied_sequence, -1)

        # A structurally valid full table with a hold beyond the rank's
        # remaining capacity fails manager prevalidation atomically as well.
        rank = manager.rank_ids[0]
        remaining = next(
            item.remaining_bytes for item in manager.hbm_snapshots()
            if item.rank == rank)
        over_capacity = _delta(
            scheduler, 0, 100,
            active_by_rank={rank: remaining + 1},
            revision_by_rank={rank: 1},
        )
        with self.assertRaisesRegex(ValueError, "workspace hold exceeds"):
            scheduler.on_decision_batch(over_capacity)
        self.assertEqual(manager.workspace_capacity_snapshot(), initial_workspace)
        self.assertEqual(tuple(scheduler.capacity_epoch), initial_epochs)
        self.assertEqual(scheduler.delivery_count, 0)
        self.assertEqual(scheduler.last_applied_sequence, -1)

    def test_workspace_release_advances_capacity_epoch_and_retries_real_policy(self):
        scheduler = _scheduler(capacity_bytes=100)
        manager = scheduler.kv_manager
        self.assertEqual(manager.model_weight_bytes_by_tp_rank, (40,))
        self.assertTrue(manager.reserve_request_capacity(
            "other-request", "other-session", 1, (20,), 1).admitted)

        first = _delta(
            scheduler, 0, 100,
            active_by_rank={1: 30}, revision_by_rank={1: 1},
        )
        scheduler.on_decision_batch(first)
        self.assertEqual(scheduler.capacity_epoch, [0, 1])
        runtime = _OnlineRequestRuntime({
            "request_id": "retry-request",
            "session_id": "retry-session",
            "turn_index": 0,
            "queue_index": 0,
            "prefill_length": 1,
            "decode_length": 1,
            "history_tokens_before": 0,
            "prefill_context_tokens": 6,
            "final_context_tokens": 6,
        })
        runtime.prefill_instance_index = 0
        runtime.static_route = SimpleNamespace(decode_instance_index=1)
        self.assertFalse(scheduler._try_admit_prefill(runtime, 2))
        self.assertEqual(runtime.prefill_attempt_epoch, (0, 1))
        self.assertFalse(runtime.decode_capacity_reserved)
        self.assertEqual(manager.deep_gap_events, 0)
        # The failed attempt is suppressed until a capacity epoch changes.
        self.assertFalse(scheduler._try_admit_prefill(runtime, 2.5))
        self.assertEqual(runtime.prefill_attempt_epoch, (0, 1))

        released = _delta(
            scheduler, 1, 101,
            active_by_rank={1: 0}, revision_by_rank={1: 2},
        )
        scheduler.on_decision_batch(released)
        self.assertEqual(scheduler.capacity_epoch, [0, 2])
        self.assertTrue(scheduler._try_admit_prefill(runtime, 3))
        self.assertEqual(runtime.prefill_attempt_epoch, (0, 2))
        self.assertTrue(runtime.admitted_prefill)
        self.assertTrue(runtime.decode_capacity_reserved)
        self.assertEqual(manager._reservations["retry-request"].shard_bytes,
                         (24,))

    def test_v1_mode_preserves_legacy_response_shape_and_ack(self):
        scheduler = _scheduler(workspace_admission="off")
        delta = _delta(scheduler, 0, 100, schema_version=1)
        with tempfile.TemporaryDirectory(prefix="lru_workspace_v1.") as bridge:
            server = BridgeServer(bridge, workspace_admission="off")
            with open(os.path.join(bridge, "request_0.json"), "w",
                      encoding="utf-8") as target:
                json.dump(delta, target, separators=(",", ":"))
            server._handle_request(0, scheduler.on_decision_batch)
            with open(os.path.join(bridge, "response_0.json"), "r",
                      encoding="utf-8") as source:
                response = json.load(source)
            self.assertEqual(response["schema_version"], 1)
            self.assertNotIn("workspace_capacity_snapshot", response)
            with open(os.path.join(bridge, "commit_ack_0.json"), "w",
                      encoding="utf-8") as target:
                json.dump({
                    "schema_version": 1,
                    "delivery_sequence": 0,
                    "batch_id": 0,
                    "success": True,
                }, target)
            server._handle_ack(0, scheduler.on_commit_ack)
            self.assertEqual(scheduler.ack_count, 1)
            server._close_audit_streams()


class WorkspaceAdmissionModeTests(unittest.TestCase):
    def _config(self, profile):
        handle = tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", delete=False)
        json.dump({"die-model-profile": profile}, handle)
        handle.close()
        self.addCleanup(lambda: os.path.exists(handle.name)
                        and os.unlink(handle.name))
        return SimpleNamespace(system_config=handle.name)

    def test_coarse_auto_requires_v2_and_legacy_auto_keeps_v1(self):
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SH_WORKSPACE_ADMISSION", None)
            self.assertEqual(
                _workspace_admission_mode(self._config("coarse_die_v2")),
                "required")
            self.assertEqual(
                _workspace_admission_mode(self._config("mesh_die_v1")),
                "off")

    def test_coarse_cannot_be_forced_off(self):
        with patch.dict(os.environ, {"SH_WORKSPACE_ADMISSION": "off"}):
            with self.assertRaisesRegex(ValueError, "coarse_die_v2 requires"):
                _workspace_admission_mode(self._config("coarse_die_v2"))


if __name__ == "__main__":
    unittest.main()
