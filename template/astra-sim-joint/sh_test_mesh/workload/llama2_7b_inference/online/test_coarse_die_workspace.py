#!/usr/bin/env python3
"""JOINT scalar workspace handshake tests for coarse_die_v2."""

from __future__ import annotations

import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import decision_bridge  # noqa: E402
from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
)
from online.online_scheduler_base import OnlineSchedulerBase  # noqa: E402


def _manager(capacity_bytes=1_000_000):
    hardware = FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    topology = build_instances(
        hardware,
        (FaceInstanceSpec("ins0", "1", (0, 1)),
         FaceInstanceSpec("ins1", "2", (2, 3))),
    )
    model = FaceModel(
        layers=4,
        hidden_size=16,
        ffn_size=32,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )
    return KVCacheManager(topology, model)


def _state(active=0, revision=0):
    return {
        "protocol_version": 1,
        "ranks": [
            {"rank": rank, "workspace_revision": revision,
             "active_workspace_bytes": active}
            for rank in range(4)
        ],
    }


class CoarseWorkspaceViewTests(unittest.TestCase):
    def test_snapshot_is_full_rank_scalar_base_and_workspace_is_subtracted_once(self):
        manager = _manager()
        initial = manager.workspace_capacity_snapshot()
        self.assertEqual(initial["protocol_version"], 1)
        self.assertEqual([row["rank"] for row in initial["ranks"]],
                         [0, 1, 2, 3])
        self.assertNotIn("active_workspace_bytes", initial["ranks"][0])
        self.assertEqual(initial["ranks"][0]["workspace_revision"], 0)

        before = manager.instance_effective_remaining_capacity_bytes(0)
        changed = manager.set_external_workspace_view(
            _state(active=128, revision=1))
        after = manager.instance_effective_remaining_capacity_bytes(0)
        self.assertEqual(changed, (0, 1, 2, 3))
        self.assertEqual(before - after, 256)
        snapshot = manager.workspace_capacity_snapshot()
        self.assertEqual(snapshot["ranks"][0]["workspace_revision"], 1)
        self.assertEqual(snapshot["ranks"][0]["model_weight_bytes"],
                         initial["ranks"][0]["model_weight_bytes"])
        self.assertEqual(manager.set_external_workspace_view(
            _state(active=128, revision=1)), ())

    def test_rejected_view_is_atomic_and_revisions_cannot_relabel_bytes(self):
        manager = _manager(capacity_bytes=20_000)
        manager.set_external_workspace_view(_state(active=8, revision=1))
        before = manager.workspace_capacity_snapshot()

        with self.assertRaisesRegex(ValueError, "without revision advance"):
            manager.validate_external_workspace_view(
                _state(active=9, revision=1))
        with self.assertRaisesRegex(ValueError, "stale workspace revision"):
            manager.validate_external_workspace_view(
                _state(active=8, revision=0))
        too_large = _state(active=10_000, revision=2)
        with self.assertRaisesRegex(ValueError, "exceeds Python HBM capacity"):
            manager.set_external_workspace_view(too_large)
        self.assertEqual(manager.workspace_capacity_snapshot(), before)

    def test_extension_pins_mode_and_requires_capacity_owner(self):
        scheduler = SimpleNamespace(
            _workspace_admission_required=None,
            kv_manager=_manager(),
        )
        self.assertFalse(OnlineSchedulerBase._validate_workspace_extension(
            scheduler, {"delivery_sequence": 0}))
        # The mode is pinned only once the complete delivery succeeds.
        scheduler._workspace_admission_required = False
        with self.assertRaisesRegex(ValueError, "cannot change within a run"):
            OnlineSchedulerBase._validate_workspace_extension(
                scheduler, {"workspace_state": _state()})

        coarse_scheduler = SimpleNamespace(
            _workspace_admission_required=None,
            kv_manager=_manager(),
        )
        self.assertTrue(OnlineSchedulerBase._validate_workspace_extension(
            coarse_scheduler, {"workspace_state": _state()}))
        coarse_scheduler._workspace_admission_required = True
        with self.assertRaisesRegex(ValueError, "cannot change within a run"):
            OnlineSchedulerBase._validate_workspace_extension(
                coarse_scheduler, {"delivery_sequence": 1})

        ownerless = SimpleNamespace(
            _workspace_admission_required=None, kv_manager=None)
        with self.assertRaisesRegex(ValueError, "no Python KV capacity owner"):
            OnlineSchedulerBase._validate_workspace_extension(
                ownerless, {"workspace_state": _state()})


class CoarseWorkspaceBridgeShapeTests(unittest.TestCase):
    def test_bridge_v3_coarse_rows_echo_every_rank_and_revision(self):
        state = _state(active=64, revision=7)
        rows = decision_bridge._validate_workspace_state(state, 3)
        capacity = {
            "protocol_version": 1,
            "ranks": [
                {"rank": rank, "workspace_revision": 7,
                 "capacity_bytes": 1000, "model_weight_bytes": 100,
                 "resident_kv_bytes": 200, "reserved_request_bytes": 50}
                for rank in range(4)
            ],
        }
        self.assertIs(decision_bridge._validate_workspace_capacity_snapshot(
            capacity, rows, 3), capacity)

        stale = {"protocol_version": 1, "ranks": [dict(row) for row in
                 capacity["ranks"]]}
        stale["ranks"][2]["workspace_revision"] = 6
        with self.assertRaisesRegex(decision_bridge.BridgeError, "does not echo"):
            decision_bridge._validate_workspace_capacity_snapshot(
                stale, rows, 3)
        with self.assertRaisesRegex(decision_bridge.BridgeError, "contiguous"):
            bad_state = _state()
            bad_state["ranks"][1]["rank"] = 3
            decision_bridge._validate_workspace_state(bad_state, 3)


if __name__ == "__main__":
    unittest.main()
