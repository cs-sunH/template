#!/usr/bin/env python3
"""test_graph_batch_builder.py -- 在线 chain_checkpoint/restore_chain 完整
同构修正的行为钉子测试。

背景（《5仓库本该一致却不同排查报告.md》低危表 L10 行，
2026-08-20）：
sh_3.0 在线 OnlineTraceBuilder.chain_checkpoint/restore_chain 原只存/回
previous_id，与离线蓝本（generate_trace.py）和 sh_2.0 在线版
（:153-158）的"双捕获/双回滚"不同构。补齐后 checkpoint 同时捕获
previous_id 与 pending_extra_dependencies，restore 两者一并回滚——分支内
新 arm 的依赖不泄漏到恢复点之后。本测试钉住该语义，并回归恒空场景
（现行唯一调用点形态，与改前行为逐位一致的 no-op 路径）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_graph_batch_builder.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.graph_batch_builder import (  # noqa: E402
    OnlineTraceBuilder,
)


def _edge_sources(builder, node_id):
    """本 rank 边列表中指向 node_id 的 from 集合（离线 data_deps 口径）。"""
    return {
        edge["from"]
        for edge in builder.edges
        if edge["to"] == node_id and edge["rank"] == builder.rank
    }


class ChainCheckpointRestoreTest(unittest.TestCase):
    """方案 §4.1：双捕获/双回滚语义钉子 + 恒空场景 no-op 回归。"""

    def setUp(self) -> None:
        self.builder = OnlineTraceBuilder(0, remote_operand_loads=False)

    def test_restore_rolls_back_branch_dependencies(self):
        """arm→发射（消费）→arm D1→checkpoint→分支内 arm D2 并发射→restore：
        previous_id 回到 checkpoint 值、pending_extra_dependencies == [D1]
        （D2 不残留）；恢复点后新节点重新依赖 D1 而不依赖 D2。"""
        builder = self.builder
        # arm 依赖 → 发一节点（消费）：依赖进入边、pending 清空。
        builder.comp("n0", 1, 1)
        dep_a = builder.previous_id
        builder.arm_dependency(dep_a)
        builder.comp("n1_consumes_a", 1, 1)
        self.assertIn(dep_a, _edge_sources(builder, builder.previous_id))
        self.assertEqual(builder.pending_extra_dependencies, [])

        # 再 arm D1 → checkpoint（捕获链首 + [D1]）。
        d1 = dep_a
        builder.arm_dependency(d1)
        checkpoint_previous = builder.previous_id
        handle = builder.chain_checkpoint()

        # 分支内：arm D2 并发一节点（timer_gate 不消费 pending/不动链首，
        # 仅用于产出真实节点 id；branch_node 消费 [D1, D2] 并前移链首）。
        d2 = builder.timer_gate("branch_gate", 1000)
        self.assertNotEqual(d2, d1)
        self.assertNotEqual(d2, checkpoint_previous)
        builder.arm_dependency(d2)
        builder.comp("branch_node", 1, 1)
        branch_node_id = builder.previous_id
        self.assertNotEqual(branch_node_id, checkpoint_previous)
        self.assertIn(d2, _edge_sources(builder, branch_node_id))
        self.assertEqual(builder.pending_extra_dependencies, [])

        # restore：链首与 pending 一并回滚，D2 不残留。
        builder.restore_chain(handle)
        self.assertEqual(builder.previous_id, checkpoint_previous)
        self.assertEqual(builder.pending_extra_dependencies, [d1])

        # 恢复点后发射：依赖 = checkpoint 链首 + D1，不含 D2。
        builder.comp("after_restore", 1, 1)
        deps = _edge_sources(builder, builder.previous_id)
        self.assertIn(checkpoint_previous, deps)
        self.assertIn(d1, deps)
        self.assertNotIn(d2, deps)
        self.assertEqual(builder.pending_extra_dependencies, [])

    def test_restore_with_empty_pending_is_noop_regression(self):
        """恒空场景（现行唯一调用点形态：checkpoint 时 pending 被 readiness
        barrier 清空、分支内不再 arm）：restore 仅回滚链首、pending 仍空——
        与改前行为逐位一致的 no-op 回归。"""
        builder = self.builder
        builder.comp("n0", 1, 1)
        checkpoint_previous = builder.previous_id
        self.assertEqual(builder.pending_extra_dependencies, [])
        handle = builder.chain_checkpoint()

        builder.comp("branch_node", 1, 1)
        self.assertNotEqual(builder.previous_id, checkpoint_previous)
        self.assertEqual(builder.pending_extra_dependencies, [])

        builder.restore_chain(handle)
        self.assertEqual(builder.previous_id, checkpoint_previous)
        self.assertEqual(builder.pending_extra_dependencies, [])

        # 恢复点后发射仅依赖 checkpoint 链首。
        builder.comp("after_restore", 1, 1)
        self.assertEqual(
            _edge_sources(builder, builder.previous_id),
            {checkpoint_previous},
        )


if __name__ == "__main__":
    unittest.main()
