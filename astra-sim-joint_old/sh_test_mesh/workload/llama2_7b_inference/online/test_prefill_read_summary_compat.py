#!/usr/bin/env python3
"""test_prefill_read_summary_compat.py -- D16 配合项（修复轮二任务 4，
2026-09-27）：P 决策行 ``prefill_remote_read_summary`` 有界聚合的兼容
读取钉测。

两种来源、同一 summary 形状：

  1. 物化列表模式（当前生产形态）：runtime.prefill_remote_read_
     transfers 在场——按 ``read_tile`` 窗分桶聚合出模板等价参数；
  2. 有界模板模式（主控 D16 轮二目标形态）：plan dict 携带
     window_count/tiles_per_window|tiles_by_window/tile_bytes/
     total_bytes/read_prefix_layers/scan_passes，无物化列表——直接
     按模板构造。

非前缀读流请求 = present False + 全零/None（键恒在）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_prefill_read_summary_compat.py
      （或 pytest 同路径）
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineRequestRuntime,
)

SESSION = "session_d16_summary_0"
RID = f"{SESSION}_request_0"


def _tile_transfer(window: int, tile: int, *, bytes_: int = 4000,
                   prefix_layers: int = 4):
    return KVTransfer(
        kind="noc_migrate",
        phase="prefill_remote_read",
        reason="remote_read_prefill_prefix",
        session_id=SESSION,
        trigger_request_id=RID,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=bytes_,
        shards=(
            KVTransferShard(
                source_rank=0, target_rank=2, edge_rank=0, bytes=bytes_,
                noc_path=(0, 2), layer_start=0, layer_end=prefix_layers),
        ),
        model_layers=4,
        layer_start=0,
        layer_end=prefix_layers,
        resident_prefix_layers_before=prefix_layers,
        resident_prefix_layers_after=prefix_layers,
        stream_only=True,
        read_tile=(window, tile),
    )


def _scheduler(p_chunk: int = 512) -> Sh30OnlineScheduler:
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.p_chunk = p_chunk
    return scheduler


def _runtime(input_tokens: int = 700) -> _OnlineRequestRuntime:
    runtime = _OnlineRequestRuntime(
        {
            "request_id": RID,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_length": input_tokens,
            "decode_length": 4,
            "history_tokens_before": 96,
            "prefill_context_tokens": 96 + input_tokens,
            "final_context_tokens": 96 + input_tokens + 4,
        },
        512,
    )
    runtime.joint_input_tokens = input_tokens
    return runtime


class PrefillReadSummaryCompatTest(unittest.TestCase):
    """_prefill_remote_read_summary：两来源同形状 + 边界姿态。"""

    def test_list_mode_aggregates_by_window(self):
        # 兼容模式（物化列表）：窗 1 = 4 tile、窗 2 = 2 tile——聚合参数
        # 与逐窗分布正确、scan_passes 走闭式回退（700 token / q_c=512 /
        # q_tile=128 ⇒ ν = ⌈512/128⌉ + ⌈188/128⌉ = 4 + 2 = 6）。
        scheduler = _scheduler()
        runtime = _runtime(input_tokens=700)
        transfers = (
            _tile_transfer(1, 1), _tile_transfer(1, 2),
            _tile_transfer(1, 3), _tile_transfer(1, 4),
            _tile_transfer(2, 1), _tile_transfer(2, 2))
        runtime.prefill_remote_read_transfers = transfers
        runtime.prefill_remote_read_bytes = 6 * 4000
        runtime.prefill_remote_read_plan = {
            "home_instance": 0,
            "exec_instance": 1,
            "read_prefix_layers": 4,
            "total_bytes": 6 * 4000,
            "tile_count": 6,
            "windows": {1: transfers[:4], 2: transfers[4:]},
            "window_cursor": 1,
        }
        summary = scheduler._prefill_remote_read_summary(runtime)
        self.assertTrue(summary["present"])
        self.assertEqual(summary["home_instance"], 0)
        self.assertEqual(summary["exec_instance"], 1)
        self.assertEqual(summary["read_prefix_layers"], 4)
        self.assertEqual(summary["total_bytes"], 24000)
        self.assertEqual(summary["tile_count"], 6)
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["tiles_by_window"], [4, 2])
        self.assertEqual(summary["tile_bytes"], 24000 // 6)
        self.assertEqual(summary["scan_passes"], 6)

    def test_template_mode_builds_same_shape(self):
        # 模板模式（主控 D16 目标形态）：无物化列表，plan dict 只带
        # 有界模板参数——同一形状逐键构造。
        scheduler = _scheduler()
        runtime = _runtime(input_tokens=700)
        runtime.prefill_remote_read_transfers = ()
        runtime.prefill_remote_read_bytes = 24000
        runtime.prefill_remote_read_plan = {
            "home_instance": 0,
            "exec_instance": 1,
            "read_prefix_layers": 4,
            "total_bytes": 24000,
            "window_count": 2,
            "tiles_per_window": 3,
            "tile_bytes": 4000,
            "scan_passes": 6,
        }
        summary = scheduler._prefill_remote_read_summary(runtime)
        self.assertTrue(summary["present"])
        self.assertEqual(summary["home_instance"], 0)
        self.assertEqual(summary["exec_instance"], 1)
        self.assertEqual(summary["read_prefix_layers"], 4)
        self.assertEqual(summary["total_bytes"], 24000)
        self.assertEqual(summary["tile_count"], 6)
        self.assertEqual(summary["window_count"], 2)
        self.assertEqual(summary["tiles_by_window"], [3, 3])
        self.assertEqual(summary["tile_bytes"], 4000)
        self.assertEqual(summary["scan_passes"], 6)

    def test_both_modes_share_shape_and_totals(self):
        # 两模式同一形状（键集相等）；同一物理读流的总量/窗数一致。
        list_runtime = _runtime(input_tokens=700)
        transfers = (
            _tile_transfer(1, 1), _tile_transfer(1, 2), _tile_transfer(1, 3),
            _tile_transfer(2, 1), _tile_transfer(2, 2), _tile_transfer(2, 3))
        list_runtime.prefill_remote_read_transfers = transfers
        list_runtime.prefill_remote_read_bytes = 6 * 4000
        list_runtime.prefill_remote_read_plan = {
            "home_instance": 0, "exec_instance": 1, "read_prefix_layers": 4,
            "total_bytes": 24000, "tile_count": 6,
            "windows": {1: transfers[:3], 2: transfers[3:]},
            "window_cursor": 1}
        template_runtime = _runtime(input_tokens=700)
        template_runtime.prefill_remote_read_transfers = ()
        template_runtime.prefill_remote_read_bytes = 24000
        template_runtime.prefill_remote_read_plan = {
            "home_instance": 0, "exec_instance": 1, "read_prefix_layers": 4,
            "total_bytes": 24000, "window_count": 2, "tiles_per_window": 3,
            "tile_bytes": 4000, "scan_passes": 6}
        list_summary = _scheduler()._prefill_remote_read_summary(list_runtime)
        template_summary = (
            _scheduler()._prefill_remote_read_summary(template_runtime))
        self.assertEqual(
            sorted(list_summary), sorted(template_summary),
            "both sources must emit the identical summary shape")
        for key in list_summary:
            self.assertEqual(
                list_summary[key], template_summary[key],
                f"summary field {key!r} must agree across list/template "
                "sources for the same physical read stream")

    def test_no_stream_defaults_zeroed_shape(self):
        # 非前缀读流请求：present False + 全零/None，键恒在。
        scheduler = _scheduler()
        runtime = _runtime()
        runtime.prefill_remote_read_transfers = ()
        runtime.prefill_remote_read_bytes = 0
        runtime.prefill_remote_read_plan = None
        summary = scheduler._prefill_remote_read_summary(runtime)
        self.assertFalse(summary["present"])
        self.assertIsNone(summary["home_instance"])
        self.assertIsNone(summary["exec_instance"])
        self.assertEqual(summary["read_prefix_layers"], 0)
        self.assertEqual(summary["total_bytes"], 0)
        self.assertEqual(summary["tile_count"], 0)
        self.assertEqual(summary["window_count"], 0)
        self.assertEqual(summary["tiles_by_window"], [])
        self.assertEqual(summary["tile_bytes"], 0)
        self.assertEqual(summary["scan_passes"], 0)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
