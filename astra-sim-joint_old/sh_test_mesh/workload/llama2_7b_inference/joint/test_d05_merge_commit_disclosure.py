#!/usr/bin/env python3
"""test_d05_merge_commit_disclosure.py -- D05（修复轮二任务 6，
2026-09-27）：merge 控制确认的零时锚披露与"无幽灵段"钉测。

核实结论（写进 joint_cost_model merge notes 与本文件）：
- 执行端 merge 尾部"commit" = 零时 watch/状态转移锚——构图器
  merge_done 标记是 1-op COMP 节点（graph_batch_builder
  ``{prefix}_merge_done_rank{r}``，R11(ii)：下一轮 interval gate 前递
  依赖 + merge-done watch 成员锚点），无任何时长段；
- 调度侧 merge 事务（merge_back/标签迁移）是决策时点状态转移，
  不占物理时间轴；
- ⇒ 不虚构计价：merge_ns = Σ腿传输 + 目标空间容量等待，无
  metadata/commit 计价段；notes 显式披露 ``merge_commit_anchor=
  zero_time_watch``。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d05_merge_commit_disclosure.py   （或 pytest）
"""
import inspect
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_INCREMENT,
)
from joint.joint_cost_model import MergePlan, project_end_state  # noqa: E402
from joint.test_d12_merge_three_position import _interval  # noqa: E402
from joint.test_d18_boundary_decision import _loads, _model  # noqa: E402


def _two_position_end_state():
    """两位置少并多：home=2000/rank < exec=3000/rank → 目标 exec，
    载荷 = home 整份（真实 merge 腿在场）。"""
    return project_end_state(
        intervals=(
            _interval("base", 0, 0, 100, (1000, 1000)),
            _interval("increment", 1, 100, 150, (1500, 1500),
                      role=INTERVAL_ROLE_INCREMENT),
        ),
        decode_instance=1, logic_home=0)


class D05MergeCommitDisclosureTest(unittest.TestCase):
    """merge 控制确认 = 零时锚：披露键 + 计价无幽灵段。"""

    def test_merge_cost_equals_legs_sum_no_ghost_segment(self):
        # 有腿 merge：merge_ns 恰为 Σ腿 + 目标空间等待——控制确认
        # （commit/状态转移）零计价段。
        notes = []
        plan = _model(_loads()).merge_from_layout(
            _two_position_end_state(), notes=notes)
        self.assertTrue(plan.legs, "fixture must carry a real merge leg")
        self.assertEqual(
            plan.merge_ns,
            sum(leg.transfer_ns for leg in plan.legs)
            + plan.target_space_wait_ns,
            "merge pricing must equal the sum of transfer legs plus the "
            "target-space capacity wait (no ghost metadata/commit "
            "segment)")
        self.assertIn("merge_commit_anchor=zero_time_watch", notes)
        # 逐腿披露在场（审计面可重建每腿时长）。
        self.assertTrue(
            any(note.startswith("merge_leg[0]=") for note in notes))

    def test_merge_ns_property_is_payload_plus_space_only(self):
        # 结构钉：MergePlan.merge_ns 的合成口径只有两个分量（防未来
        # 幽灵段混入合成点）。
        plan = MergePlan(
            target_instance=1, selection_rule="retention_max",
            complete_union=False, legs=(), payload_ns=321,
            target_space_bytes_by_rank=(1, 1),
            target_space_wait_ns=55, capacity_fallback_order=(),
            retention_disclosure=())
        self.assertEqual(plan.merge_ns, 321 + 55)

    def test_zero_byte_flip_carries_zero_time_anchor_note(self):
        # 零字节翻转（无腿 merge）：同样只有零时控制锚。
        from joint.boundary_layout import INTERVAL_ROLE_BASE
        end_state = project_end_state(
            intervals=(
                _interval("base", 1, 0, 100, (1000, 1000),
                          role=INTERVAL_ROLE_BASE),
                _interval("increment", 1, 100, 150, (1500, 1500),
                          role=INTERVAL_ROLE_INCREMENT),
            ),
            decode_instance=1, logic_home=0)
        notes = []
        plan = _model(_loads()).merge_from_layout(end_state, notes=notes)
        if plan.complete_union:
            self.assertEqual(plan.merge_ns, 0)
            self.assertIn("merge_v2_zero_byte_flip", notes)
            self.assertIn("merge_commit_anchor=zero_time_watch", notes)

    def test_builder_merge_done_marker_is_one_op_zero_time_anchor(self):
        # 执行侧证据钉（源级）：merge_done 标记节点 = comp(name, 1, 1)
        # ——1-op 零时 watch/依赖锚，无时长段（R11(ii) interval gate
        # 前递依赖 + merge-done watch 成员锚点）。
        from online import graph_batch_builder
        source = inspect.getsource(graph_batch_builder)
        self.assertIn(
            '"{prefix}_merge_done_rank{rank}", 1, 1',
            source,
            "the merge_done control-confirmation marker must remain a "
            "1-op zero-time COMP anchor (no duration segment)")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
