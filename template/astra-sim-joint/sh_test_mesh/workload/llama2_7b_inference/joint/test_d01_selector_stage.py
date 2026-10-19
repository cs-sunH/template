#!/usr/bin/env python3
"""test_d01_selector_stage.py -- D01 选择器保留与阶段扩展钉测。

对应 joint修改文档.md D01 / v2.0 §21.1 D01、§22 D01、C07。本条目对
选择器与估价器是**保留＋显式化**要求（不重写成新优化器），钉测五面：

  1. 保留全实例×二动作 argmin 与初始平局序 —— joint 主臂 argmin 键
     ``(cost, (instance_index, ACTION_ORDER))`` 不变（C07 现状）；D
     边界式(1c) 平局键不回流初始时刻（P 候选恒非 continuation）。
  2. 阶段参数 —— 两阶段入口（``estimate_P_with_continuation`` /
     ``estimate_D_remaining``）的候选 notes 以同键
     ``decision_stage=P``/``decision_stage=D`` 自描述决策阶段（D18
     的 ``boundary_stage=D`` 披露位保留不动）。
  3. 当前布局参数 —— P 入口布局 = 决策时点 SessionKVView；D 入口
     布局 = PDBoundaryView（D19 派生 BoundaryKVLayout；候选面按边界
     布局划面，不消费到达时视图——字节口径由 test_d19 钉测）。
  4. 继续标记归并到 (i_P, action) 记录 —— 每阶段恰 2N 条记录
     （2026-10-15 二动作；原 4N/3N），``is_continuation`` 恰在原地
     继续记录上置位。
  5. 每阶段最多 2N 记录、两次线性遍历 —— 选择器恰调 2N 次估价器、
     初始时刻零 D 估价（初始候选不做全 D 搜索）、无组合枚举；
     两个时刻的全部实例都有审计项（不保留最近/最空闲实例偏侧）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d01_selector_stage.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_ORDER,
    ACTION_STAY,
    ActionCandidate,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import (  # noqa: E402
    select_decode_target_and_action,
    select_instance_and_action,
)

KV_TOKEN_BYTES = 128


def _route(source, target):
    return ((source, target), 1)


def _cost_model(loads):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route)


def _load(index, queued_ns=0, running_ns=0):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=queued_ns,
        running_task_load_ns=running_ns,
        active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=(10 ** 9, 10 ** 9),
        reclaimable_bytes_by_tp_rank=(10 ** 9, 10 ** 9))


def _fresh_session():
    """turn-0 无历史新会话（stay 全实例适用——本地建立，copy 拒）。"""
    return SessionKVView(
        session_id="s", home_instance=None, resident_instance=None,
        location="none", history_tokens=0, resident_prefix_layers=0,
        history_bytes_by_tp_rank=(0, 0), missing_bytes_by_tp_rank=(0, 0))


def _request(input_tokens=50, decode_estimate=10):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=0,
        estimated_decode_tokens=decode_estimate,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=(25, 25))


def _complete_boundary(**overrides):
    """完整型边界（布局恒完整型单族；P=stay，H_B=100 token 全量在
    i_P=0，home=0）。"""
    values = dict(
        session_id="s", home_instance=0, p_instance_index=0,
        p_action="stay", history_tokens=100,
        working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2,
        decode_steps=8, kv_token_bytes=KV_TOKEN_BYTES)
    values.update(overrides)
    return PDBoundaryView(**values)


class _CountingModel:
    """真实 JCM 的调用计数包装（选择器只经由本面触达估价器）。"""

    def __init__(self, base):
        self._base = base
        self.p_calls = []
        self.d_calls = []

    @property
    def loads(self):
        return self._base.loads

    def estimate_P_with_continuation(self, **kwargs):
        self.p_calls.append(
            (kwargs["instance_index"], kwargs["action"]))
        return self._base.estimate_P_with_continuation(**kwargs)

    def estimate_D_remaining(self, **kwargs):
        self.d_calls.append((kwargs["instance_index"], kwargs["action"]))
        return self._base.estimate_D_remaining(**kwargs)


# ================================================== 1. 初始 argmin ==


class D01InitialArgminTieOrderTest(unittest.TestCase):
    """保留全实例×二动作 argmin 与初始平局序（v2.0 §22 D01"初始 tie
    不变"；等价瘦身时初始选择保持）。"""

    def test_initial_tie_resolves_by_instance_then_action_order(self):
        # turn-0 双实例等负载：stay × 两实例两候选真平局（本地建立
        # 同工作量）。初始平局序
        # = (instance_index, ACTION_ORDER)：最低实例的 stay 胜出。
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            )
        applicable = [c for c in record.candidates if c.applicable]
        self.assertEqual(len(applicable), 2)
        self.assertEqual(
            len({c.cost_ns for c in applicable}), 1)
        self.assertEqual(
            (record.chosen.instance_index, record.chosen.action),
            (0, ACTION_STAY))
        self.assertEqual(
            record.chosen.order_key(),
            (0, ACTION_ORDER.index(ACTION_STAY)))

    def test_initial_tie_note_discloses_initial_order(self):
        # 平局披露键 = 初始序，不是 D 边界的继续优先键。
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            )
        self.assertIn("tie among 2", record.tie_break_note)
        self.assertIn("(instance_index, action order)",
                      record.tie_break_note)
        self.assertNotIn("continuation", record.tie_break_note)

    def test_initial_candidates_never_continuation(self):
        # 式(1c) 非继续标记优先不回流初始时刻：P 候选恒
        # is_continuation=False（初始无继续语义）。
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            )
        self.assertFalse(any(
            c.is_continuation for c in record.candidates))


# ====================================== 2. 候选面与两次线性遍历 ==


class D01CandidateShapeAndTraversalTest(unittest.TestCase):
    """每阶段最多 2N 记录、两次线性遍历、不枚举组合；两个时刻
    的全部实例都有审计项（v2.0 §22 D01 验收）。"""

    def test_p_stage_records_2n_all_instances_audited(self):
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            )
        # 恰 2N 条 (instance, action) 记录，无重复、无缺位。
        self.assertEqual(len(record.candidates), 4)
        self.assertEqual(
            [(c.instance_index, c.action) for c in record.candidates],
            [(i, a) for i in (0, 1) for a in ACTION_ORDER])
        # 全实例审计项：不可用候选也带理由在案。
        for candidate in record.candidates:
            if not candidate.applicable:
                self.assertTrue(candidate.inapplicable_reason)

    def test_d_stage_records_2n_all_instances_audited(self):
        record = select_decode_target_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            boundary=_complete_boundary(), )
        self.assertEqual(len(record.candidates), 4)
        self.assertEqual(
            [(c.instance_index, c.action) for c in record.candidates],
            [(i, a) for i in (0, 1) for a in ACTION_ORDER])
        self.assertEqual(
            {c.instance_index for c in record.candidates}, {0, 1})

    def test_two_linear_traversals_no_combinatorial_search(self):
        # 同一请求两决策时刻：初始恰 2N 次 P 估价且零 D 估价（初始候选
        # 不做全 D 搜索——组合枚举不存在）；边界恰 2N 次 D 估价；
        # 合计两次线性遍历（2×2N）。
        model = _CountingModel(_cost_model({0: _load(0), 1: _load(1)}))
        record_p = select_instance_and_action(
            mode="joint", cost_model=model, session=_fresh_session(),
            request=_request(), )
        self.assertEqual(len(model.p_calls), 4)
        self.assertEqual(
            sorted(model.p_calls),
            sorted((i, a) for i in (0, 1) for a in ACTION_ORDER))
        self.assertEqual(model.d_calls, [])
        self.assertEqual(len(record_p.candidates), 4)
        record_d = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_complete_boundary(), )
        self.assertEqual(len(model.d_calls), 4)
        self.assertEqual(
            sorted(model.d_calls),
            sorted((i, a) for i in (0, 1) for a in ACTION_ORDER))
        self.assertEqual(len(model.p_calls), 4)  # 无第三次位置决策入口
        self.assertEqual(len(record_d.candidates), 4)


# ==================================== 3. 继续归并与边界平局键 ==


class D01ContinuationMergeTest(unittest.TestCase):
    """继续标记归并到对应 (i_P, action) 记录、不作第五动作；D 边界
    平局键 = (cost, 非继续标记, instance_index, action_order)。"""

    def test_complete_family_continuation_on_p_stay_record(self):
        model = _cost_model({0: _load(0), 1: _load(1)})
        boundary = _complete_boundary()
        candidates = [
            model.estimate_D_remaining(
                boundary=boundary, instance_index=i, action=a,
                )
            for i in (0, 1) for a in ACTION_ORDER]
        # 2N 条记录（2026-10-15 二动作；原 4N/3N）：x0 归并在
        # (i_P, stay) 上。
        self.assertEqual(len(candidates), 4)
        continuations = [c for c in candidates if c.is_continuation]
        self.assertEqual(len(continuations), 1)
        self.assertEqual(
            (continuations[0].instance_index, continuations[0].action),
            (0, ACTION_STAY))
        self.assertTrue(continuations[0].applicable)

    def test_boundary_tie_key_exact_form(self):
        continuation = ActionCandidate(
            instance_index=0, action=ACTION_STAY, applicable=True,
            inapplicable_reason=None, cost_ns=100, is_continuation=True)
        switch = ActionCandidate(
            instance_index=1, action="copy", applicable=True,
            inapplicable_reason=None, cost_ns=100)
        # 键形 = (非继续标记, instance_index, action_order)；成本并列时
        # 继续记录恒优先（式 1c）。
        self.assertEqual(
            continuation.boundary_order_key(),
            (0,) + continuation.order_key())
        self.assertEqual(
            switch.boundary_order_key(), (1,) + switch.order_key())
        self.assertLess(
            continuation.boundary_order_key(),
            switch.boundary_order_key())
        # 同为非继续：仍按 (instance_index, action_order) 定序。
        other = ActionCandidate(
            instance_index=0, action="copy", applicable=True,
            inapplicable_reason=None, cost_ns=100)
        self.assertLess(other.boundary_order_key(),
                        switch.boundary_order_key())


# ============================================== 4. 阶段披露 ==


class D01StageDisclosureTest(unittest.TestCase):
    """阶段参数显式化：两阶段候选与选择记录自描述决策阶段
    （decision_stage=P/D 同键对称；boundary_stage=D 保留）。"""

    def test_p_candidates_and_record_disclose_stage_p(self):
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            )
        self.assertEqual(record.diagnostics["decision_stage"], "P")
        for candidate in record.candidates:
            if candidate.applicable:
                self.assertIn("decision_stage=P",
                              candidate.breakdown.notes)

    def test_sequential_mode_record_discloses_stage_p(self):
        record = select_instance_and_action(
            mode="load-first", cost_model=_cost_model(
                {0: _load(0, queued_ns=100), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            )
        self.assertEqual(record.diagnostics["decision_stage"], "P")
        self.assertEqual(
            record.diagnostics["sequential_instance"], 1)

    def test_d_candidates_and_record_disclose_stage_d(self):
        model = _cost_model({0: _load(0), 1: _load(1)})
        boundary = _complete_boundary()
        candidate = model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY,
            )
        self.assertTrue(candidate.applicable)
        self.assertIn("decision_stage=D", candidate.breakdown.notes)
        # D18 原披露位保留不动（向后兼容位）。
        self.assertIn("boundary_stage=D", candidate.breakdown.notes)
        record = select_decode_target_and_action(
            mode="joint", cost_model=model, boundary=boundary,
            )
        self.assertEqual(record.diagnostics["decision_stage"], "D")


if __name__ == "__main__":
    unittest.main()
