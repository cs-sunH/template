#!/usr/bin/env python3
"""Prospective workspace floor tests for the recompute manager.

蓝本 astra-sim-wscllm-LRU test_workspace_admission_manager.py 的
prospective 相关用例段(构造负例 + max(actual, prospective) 合并语义);
其余 WL 用例测其本仓特性(remote pool/整体逐出),不随迁。
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import session_kv_manager as manager_module  # noqa: E402
from session_kv_manager import SessionKVCacheManager  # noqa: E402
from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmModel,
    build_instances,
)


def _model() -> WscLlmModel:
    # Same pinned compact model as the LRU blueprint: 320 B for
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


class ProspectiveWorkspaceFloorTests(unittest.TestCase):
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
        # The immutable floor is charged only once in the snapshot's combined
        # headroom check: 480 weight + 60 actual (< 50 floor ignored in the
        # base) keeps the snapshot fields at their v1 meanings.
        self.assertEqual(snapshot["ranks"][0]["model_weight_bytes"], 480)

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


if __name__ == "__main__":
    unittest.main()
