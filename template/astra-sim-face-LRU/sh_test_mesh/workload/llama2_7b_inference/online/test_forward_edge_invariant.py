"""死锁真修（方案①，2026-10-11，自 WL 同构移植）T4 前向断言正负例。

T4：arm_dependency / arm_timer_gate 的目标节点必须已在该 rank 发射
（发射序号 = 节点 id = builder.next_id 单调空间）。本测试钉死：
- 已发射目标：arm 成功，pending 记录该 id；
- 未发射目标（>= next_id）：RuntimeError（无白名单）；
- None 目标：no-op（既有契约）。
"""

import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402


class _Config:
    npus_count = 4
    remote_operand_loads = False
    trace_granularity = "request_aggregated"
    inference_groups = [
        type("G", (), {"name": "g0", "ranks": (0, 1), "pg_name": "pg0"}),
        type("G", (), {"name": "g1", "ranks": (2, 3), "pg_name": "pg1"}),
    ]
    prefill_chunk_size = 128
    layers = 2
    hidden_size = 32
    ffn_size = 64
    vocab_size = 64
    bytes_per_elem = 2
    num_heads = 4
    mlp_variant = "gelu"
    request_queue = []
    model = type("M", (), {"layers": 2, "hidden_size": 32,
                           "bytes_per_elem": 2})
    hardware = type("H", (), {"mesh_rows": 2, "mesh_cols": 2,
                              "npus_count": 4})


class T4ForwardInvariantTest(unittest.TestCase):
    def test_arm_dependency_accepts_emitted_target(self):
        builder = GraphBatchBuilder(_Config())
        builder.begin_batch()
        builder.builders[0].comp("probe", 1, 1)
        node_id = builder.builders[0].previous_id
        self.assertIsNotNone(node_id)
        builder.builders[0].arm_dependency(node_id)
        self.assertEqual(builder.builders[0].pending_extra_dependencies,
                         [node_id])

    def test_arm_dependency_rejects_future_target(self):
        builder = GraphBatchBuilder(_Config())
        builder.begin_batch()
        future_id = builder.builders[0].next_id  # 尚未发射
        with self.assertRaises(RuntimeError):
            builder.builders[0].arm_dependency(future_id)
        self.assertEqual(builder.builders[0].pending_extra_dependencies,
                         [])

    def test_arm_dependency_rejects_foreign_magnitude(self):
        builder = GraphBatchBuilder(_Config())
        builder.begin_batch()
        with self.assertRaises(RuntimeError):
            builder.builders[0].arm_dependency(999999)

    def test_arm_timer_gate_rejects_future_target(self):
        builder = GraphBatchBuilder(_Config())
        builder.begin_batch()
        builder.builders[1].comp("probe", 1, 1)
        node_id = builder.builders[1].previous_id
        builder.builders[1].arm_timer_gate(node_id)
        self.assertEqual(builder.builders[1].pending_extra_dependencies,
                         [node_id])
        with self.assertRaises(RuntimeError):
            builder.builders[1].arm_timer_gate(node_id + 1000)

    def test_none_target_is_noop(self):
        builder = GraphBatchBuilder(_Config())
        builder.begin_batch()
        builder.builders[0].arm_dependency(None)
        builder.builders[0].arm_timer_gate(None)
        self.assertEqual(builder.builders[0].pending_extra_dependencies,
                         [])


if __name__ == "__main__":
    unittest.main()
