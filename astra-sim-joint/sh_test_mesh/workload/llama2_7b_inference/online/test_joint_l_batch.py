#!/usr/bin/env python3
"""test_joint_l_batch.py -- L 批（L1–L8，2026-09-23 K 批复核审计修复）
的零后端测试。

源：用户转交 kimi 复核报告（P2×4 + P3 择要），逐条读码亲验后裁定
（执行计划 §4.3 A16' / PROVENANCE §39）。本文件钉修复面：

  L1（P2-1）quota_deferred forced 计数键初始化（动态 format 键不再
        潜伏 KeyError）+ __init__ 文本钉；
  L2（P2-2）predictor NaN 同族三处补齐（ResourceSnapshot.now_ns /
        CommittedFlow.release_eta_ns / first_block_wait_ns 两入口）+
        跨腿 rank 数等长 fail-closed；
  L5（P3）copy 层段账本内部缺口 fail-closed（块间缺口 + copy↔restore
        间缺口两形态；covered==0 热前缀保持合法）；
  L8（P3）ReverseLegOwnPath 对偶钉：背景流抬前向腿（K5-② 只钉了
        reverse 不动）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_l_batch.py   （或 pytest 同路径）
"""

import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.event_recursion_predictor import (  # noqa: E402
    CommittedFlow,
    EventRecursionError,
    LayerRecursionPredictor,
    RestoreGroupLeg,
    ResourceSnapshot,
)
from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
from joint.test_joint_fixes import _load, _model  # noqa: E402
from joint.test_joint_review3_fixes import (  # noqa: E402
    _request_view,
    _session_view,
)
from test_joint_k_batch import (  # noqa: E402
    EXEC_RANKS,
    REQUEST_K6,
    _admission_plan_k6,
    _graph_config_k6,
    _train_plan_k6,
)
from graph_batch_builder import GraphBatchBuilder  # noqa: E402

_SH30_SOURCE_PATH = os.path.join(_ONLINE_DIR, "sh30_online_scheduler.py")


# ==================================================== 1. L1（P2-1）==

class ForcedQuotaDeferredCountKeyTest(unittest.TestCase):
    """L1：quota 压塌 {recompute} 的选中计数走
    recompute_forced_quota_deferred 键——不再 KeyError（修前 __init__
    计数表只含两旧键，_note_action_selection 的动态 format 键必炸）。"""

    @staticmethod
    def _candidate(action, applicable, reason=None):
        return SimpleNamespace(
            action=action, applicable=applicable,
            inapplicable_reason=reason)

    def test_init_table_declares_quota_deferred_key(self):
        # 文本钉（K 批 wiring 先例）：__init__ 计数表必须显式声明三成因
        # 全部 forced 键——动态 format 键与初始化表解耦即回归本缺陷。
        with open(_SH30_SOURCE_PATH, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn('"recompute_forced_quota_deferred": 0,', source)

    def test_quota_deferred_selection_counts_without_keyerror(self):
        from sh30_online_scheduler import Sh30OnlineScheduler
        counts = {
            "stay": 0, "copy": 0, "remote-read": 0,
            "recompute_elected": 0,
            "recompute_forced_no_history": 0,
            "recompute_forced_quota_deferred": 0,
            "recompute_forced_evicted_permanent": 0,
        }
        fake_self = SimpleNamespace(
            _joint_action_selection_counts=counts,
            # staticmethod 可直接挂替身（_note_action_selection 只用这
            # 两个属性）。
            _recompute_selection_tier=(
                Sh30OnlineScheduler._recompute_selection_tier))
        candidates = [
            self._candidate("stay", False,
                            "history not resident at target"),
            self._candidate("copy", False, "no history to copy"),
            self._candidate("remote-read", False,
                            "quota_link: link=(0,1) remaining=0"),
            self._candidate("recompute", True)]
        session_view = SimpleNamespace(history_tokens=50)
        # 修前：counts["recompute_forced_quota_deferred"] KeyError。
        Sh30OnlineScheduler._note_action_selection(
            fake_self, "recompute", candidates, session_view)
        self.assertEqual(counts["recompute_forced_quota_deferred"], 1)
        self.assertEqual(counts["recompute_forced_no_history"], 0)


# ==================================================== 2. L2（P2-2）==

def _snapshot(**overrides):
    kwargs = dict(
        peak_bytes_per_ns={"link:0->1": 10.0, "port:0": 5.0},
        committed=(),
    )
    kwargs.update(overrides)
    return ResourceSnapshot(**kwargs)


def _leg(layer_start, layer_end, rank_bytes=(100,), ranks=1):
    return RestoreGroupLeg(
        layer_start=layer_start, layer_end=layer_end,
        bytes_by_rank=tuple(rank_bytes),
        path_by_rank=tuple(
            ("link:0->1", "port:0") for _ in range(ranks)))


class SnapshotNowNsFiniteTest(unittest.TestCase):
    """L2：now_ns 非有限拒收——NaN 进 step() 流永不激活（`start <=
    NaN` 恒 False）、`max(NaN, min(starts))` 保持 NaN ⇒ run() 死循环。"""

    def test_nan_now_ns_rejected(self):
        with self.assertRaisesRegex(EventRecursionError, "now_ns"):
            _snapshot(now_ns=float("nan"))

    def test_inf_now_ns_rejected(self):
        with self.assertRaisesRegex(EventRecursionError, "now_ns"):
            _snapshot(now_ns=float("inf"))

    def test_finite_now_ns_accepted(self):
        self.assertEqual(_snapshot(now_ns=0).now_ns, 0)


class CommittedFlowEtaFiniteTest(unittest.TestCase):
    """L2：release_eta_ns 非有限拒收（NaN `eta > now` 恒 False ⇒ 在册
    流被当已过期、竞争份额被静默忽略——乐观方向）；负值合法（已过期）。"""

    def test_nan_eta_rejected(self):
        flow = CommittedFlow(
            flow_id="f", resources=("link:0->1",),
            release_eta_ns=float("nan"))
        with self.assertRaisesRegex(EventRecursionError, "release ETA"):
            _snapshot(committed=(flow,))

    def test_inf_eta_rejected(self):
        flow = CommittedFlow(
            flow_id="f", resources=("link:0->1",),
            release_eta_ns=float("inf"))
        with self.assertRaisesRegex(EventRecursionError, "release ETA"):
            _snapshot(committed=(flow,))

    def test_negative_eta_is_past_not_rejected(self):
        flow = CommittedFlow(
            flow_id="f", resources=("link:0->1",), release_eta_ns=-100)
        self.assertEqual(_snapshot(committed=(flow,)).committed[0]
                         .release_eta_ns, -100)


class FirstBlockWaitFiniteTest(unittest.TestCase):
    """L2：first_block_wait_ns 先校验后 int()——int(NaN) 裸 ValueError
    逃出调用方只 catch EventRecursionError 的降级通道（face_scheduler
    递推入口）。构造器与合流入口同一纪律。"""

    def test_constructor_nan_rejected_as_recursion_error(self):
        with self.assertRaises(EventRecursionError):
            LayerRecursionPredictor(
                _snapshot(), [_leg(0, 1)], (),
                first_block_wait_ns=float("nan"))

    def test_constructor_inf_rejected_as_recursion_error(self):
        with self.assertRaises(EventRecursionError):
            LayerRecursionPredictor(
                _snapshot(), [_leg(0, 1)], (),
                first_block_wait_ns=float("inf"))

    def test_constructor_negative_rejected(self):
        with self.assertRaisesRegex(EventRecursionError, "non-negative"):
            LayerRecursionPredictor(
                _snapshot(), [_leg(0, 1)], (),
                first_block_wait_ns=-1)

    def test_merge_entrypoint_nan_rejected_as_recursion_error(self):
        from joint.event_recursion_predictor import (
            predict_release_and_recall)
        with self.assertRaises(EventRecursionError):
            predict_release_and_recall(
                _snapshot(), (), [_leg(0, 1)], (),
                first_block_wait_ns=float("nan"))


class CrossLegRankCountAlignmentTest(unittest.TestCase):
    """L2（P3 小项）：跨腿 rank 数不一致 fail-closed——修前
    bytes_by_rank[rank_index] 以 leg0 定 rank_count、短腿裸 IndexError
    （生产腿恒同 TP 组不可达，防御纵深与 legs-vs-segments 对齐检查同面）。"""

    def test_mismatched_rank_counts_rejected(self):
        with self.assertRaisesRegex(
                EventRecursionError, "common rank count"):
            LayerRecursionPredictor(
                _snapshot(),
                [_leg(0, 1, rank_bytes=(100, 100), ranks=2),
                 _leg(1, 2, rank_bytes=(100,), ranks=1)],
                (), first_block_wait_ns=0)

    def test_uniform_rank_counts_accepted(self):
        predictor = LayerRecursionPredictor(
            _snapshot(),
            [_leg(0, 1, rank_bytes=(100, 100), ranks=2),
             _leg(1, 2, rank_bytes=(50, 50), ranks=2)],
            (), first_block_wait_ns=0)
        self.assertEqual(len(predictor.legs), 2)


# ==================================================== 3. L5（P3）==

class CopyLedgerGapFailClosedTest(unittest.TestCase):
    """L5：copy 层段账本内部缺口 fail-closed——修前 `layer_start >
    covered` 静默插无门段（缺口层不等到达即计算，时序乐观），restore
    侧同情形是 raise。covered==0 的热前缀无门段保持合法（K6 既有测试
    已钉热前缀形态）。"""

    def setUp(self):
        self.builder = GraphBatchBuilder(_graph_config_k6())
        self.builder.begin_batch()
        # fork frontier + 准入批发射（尾块支链 + 层区间账本登记）。
        self.builder.emit_iteration_train(
            _train_plan_k6("batch_train_i1_1", [(1, 101)]))
        self.builder.emit_admission_batch(_admission_plan_k6())
        # 账本原状（K6 钉）：{1: (8,16), 2: (16,24)} 连续。

    def _emit_body(self):
        self.builder.emit_iteration_train(
            _train_plan_k6(
                "batch_train_i1_2", [(1, 201), (1, 202), (1, 203)]))

    def test_internal_gap_between_tail_chunks_raises(self):
        # 篡改账本：chunk 2 起点上移留 [16,20) 内部缺口。
        self.builder._copy_handoff_layers[REQUEST_K6] = {
            1: (8, 16), 2: (20, 24)}
        with self.assertRaisesRegex(RuntimeError, "internal gap"):
            self._emit_body()

    def test_gap_between_copy_and_restore_raises(self):
        # copy 尾段止于 16，恢复组自 20 起——两账本并集缺口 [16,20)。
        self.builder._copy_handoff_layers[REQUEST_K6] = {1: (8, 16)}
        gates = {rank: "gate-restore-g0" for rank in EXEC_RANKS}
        self.builder._suffix_restore_arms[REQUEST_K6] = [
            ("g0", 20, 24, gates)]
        with self.assertRaisesRegex(RuntimeError, "internal gap"):
            self._emit_body()

    def test_hot_prefix_gap_at_zero_stays_legal(self):
        # 回归钉：covered==0 的缺口是热前缀合法无门段（chunk 1 起点上
        # 移到 10——[0,10) 热前缀加宽不触发 raise）。
        self.builder._copy_handoff_layers[REQUEST_K6] = {
            1: (10, 17), 2: (17, 24)}
        self._emit_body()   # 不 raise 即通过
        self.assertNotIn(REQUEST_K6, self.builder._copy_handoff_layers)


# ============================================ 4. L8（P3 补钉对偶）==

class ForwardLegContendedByBackgroundTest(unittest.TestCase):
    """L8：K5-② ReverseLegOwnPathTest 只钉了"背景流不抬反向腿"；本类
    钉对偶面——同一背景流（驻 exec→home 有向边）确实抬前向腿（方向
    特异性的正向证据）。"""

    def _forward_ns(self, with_background):
        flows = LinkFlowRegistry()
        if with_background:
            flows.register_path((1, 0), owner="f-bg")
        model = _model({0: _load(), 1: _load()}, flows=flows)
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(input_tokens=5, decode=0,
                                  input_bytes=(5, 5)),
            instance_index=1, action="remote-read", remote_enabled=True)
        notes = candidate.breakdown.notes
        forward_note = next(
            note for note in notes
            if note.startswith("merge_v2_forward_ns="))
        return int(forward_note.split("=")[1])

    def test_background_on_forward_edge_raises_forward_leg(self):
        clean = self._forward_ns(with_background=False)
        contended = self._forward_ns(with_background=True)
        # 除数 (f+n)/(n·(f+1)) 抬升 ⇒ 前向腿变慢（方向特异性正向钉）。
        self.assertGreater(contended, clean)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
