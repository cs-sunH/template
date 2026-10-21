#!/usr/bin/env python3
"""test_joint_tie_rotation.py -- v9（2026-10-20）joint 主臂跨实例平局
轮转裁决钉测。

对象：``joint_scheduler.joint_tie_rotation_pick`` 纯函数（最优价并列
实例识别 + 编号环序轮转）与 ``select_instance_and_action`` 的
``tie_cursor`` 注入面。``joint_config.MODEL_SEMANTICS_VERSION`` 已升
"v9-tie-roundrobin-20261020"（单一裁决点）。语义要点：

* 跨实例平局（|tie_instances| >= 2）：自游标起沿实例全集环序（升序、
  越最大值回绕）取首个在集实例，实例内 stay/copy 同价并列仍由
  ACTION_ORDER 终裁（与旧序数学等价）；
* 单实例动作平局 / 唯一最优实例：rotated=False，行为与旧固定
  ``(instance_index, ACTION_ORDER)`` 确定序恒等（不轮转）；
* 纯函数不推进游标——推进义务在在线调度器"最终有效选择"处
  （sh30_online_scheduler 的 ``_consume_joint_tie_rotation``）；
* 顺序参照（load-first / affinity-first）与 face_static 对照臂不读
  ``tie_cursor``；D 边界式(1c) 平局键（继续优先）与本入口正交。

fixture 构造参照 test_d01_selector_stage（本地 _cost_model/_load/
_fresh_session/_request）。单实例动作平局在真实 JCM 的适用性规则下
不可达（同一实例上 stay 与 copy 的适用条件互斥——见
joint_cost_model.JointCostModel.applicable_actions），故该组经
``_StubModel`` 注入精确候选面：选择入口对 cost_model 只鸭子类型消费
``loads`` 与 ``estimate_P_with_continuation``（与 test_d01 的
``_CountingModel`` 包装同一构造风格）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_joint_tie_rotation.py   （或 pytest 同路径）
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
    ACTION_COPY,
    ACTION_ORDER,
    ACTION_STAY,
    ActionCandidate,
    InstanceLoadView,
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
)
from joint.joint_scheduler import (  # noqa: E402
    JointSchedulerError,
    joint_tie_rotation_pick,
    select_instance_and_action,
)


def _route(source, target):
    return ((source, target), 1)


def _cost_model(loads, *, allowed_actions=None):
    rates = JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=10, pool_latency_ns=100)
    return JointCostModel(
        rates=rates, loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        allowed_actions=allowed_actions)


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


def _cand(instance_index, action, cost_ns, *, applicable=True):
    """P 入口形状的候选（P 阶段恒非 continuation；不可适用候选
    cost_ns=None——与真实 JCM 候选面同形状）。"""
    return ActionCandidate(
        instance_index=instance_index, action=action,
        applicable=applicable,
        inapplicable_reason=None if applicable else "inapplicable",
        cost_ns=cost_ns if applicable else None)


def _applicable_only(candidates):
    """调用方契约：joint_tie_rotation_pick 收到的是 select_instance_
    and_action 过滤后的适用候选子集（不可适用候选不计价、不进
    tie 集）。"""
    return [c for c in candidates if c.applicable]


class _StubModel:
    """注入精确候选面的鸭子类型 cost model（joint 分支只消费 loads
    与 estimate_P_with_continuation——见 select_instance_and_action
    的 joint 主臂路径）。"""

    def __init__(self, loads, candidates):
        self._loads = loads
        self._by_pair = {
            (candidate.instance_index, candidate.action): candidate
            for candidate in candidates}

    @property
    def loads(self):
        return self._loads

    def estimate_P_with_continuation(self, **kwargs):
        return self._by_pair[(kwargs["instance_index"], kwargs["action"])]


# ================================================ a) 全平局轮转 ==


class FullTieRotationTest(unittest.TestCase):
    """a) N 实例同价：tie_cursor=0..N-1 依次选出该编号实例；游标越
    全集最大值回绕到首个在集成员；纯函数不推进游标（推进义务在
    调用方）。"""

    N = 4
    COST_NS = 100

    def _candidates(self):
        candidates = []
        for index in range(self.N):
            candidates.append(_cand(index, ACTION_STAY, self.COST_NS))
            candidates.append(_cand(
                index, ACTION_COPY, self.COST_NS, applicable=False))
        return candidates

    def test_cursor_picks_that_instance_for_each_position(self):
        candidates = self._candidates()
        universe = tuple(range(self.N))
        for cursor in range(self.N):
            chosen, tie_instances = joint_tie_rotation_pick(
                _applicable_only(candidates), cursor, universe)
            self.assertEqual(chosen.instance_index, cursor)
            self.assertEqual(chosen.action, ACTION_STAY)
            self.assertEqual(tie_instances, tuple(range(self.N)))

    def test_cursor_at_last_instance_stays_no_advance(self):
        # 游标 = N-1 选 N-1；纯函数不推进游标——同游标重复调用结果
        # 恒同（游标推进义务在调用方，见模块 docstring）。
        candidates = self._candidates()
        universe = tuple(range(self.N))
        first, _ = joint_tie_rotation_pick(
            _applicable_only(candidates), self.N - 1, universe)
        second, _ = joint_tie_rotation_pick(
            _applicable_only(candidates), self.N - 1, universe)
        self.assertEqual(first.instance_index, self.N - 1)
        self.assertEqual(
            (first.instance_index, first.action),
            (second.instance_index, second.action))

    def test_cursor_beyond_universe_max_wraps_to_first(self):
        candidates = self._candidates()
        universe = tuple(range(self.N))
        for cursor in (self.N, self.N + 3, 99):
            chosen, _ = joint_tie_rotation_pick(
                _applicable_only(candidates), cursor, universe)
            self.assertEqual(chosen.instance_index, 0, cursor)

    def test_integration_real_model_rotates_by_cursor(self):
        # 真实 JCM：N 实例等负载 turn-0 会话 → N 个 stay 候选同价。
        loads = {index: _load(index) for index in range(3)}
        for cursor in range(3):
            record = select_instance_and_action(
                mode="joint", cost_model=_cost_model(loads),
                session=_fresh_session(), request=_request(),
                tie_cursor=cursor)
            self.assertEqual(
                (record.chosen.instance_index, record.chosen.action),
                (cursor, ACTION_STAY))
            self.assertEqual(
                record.diagnostics["joint_tie_instances"], (0, 1, 2))
            self.assertIs(record.diagnostics["joint_tie_rotated"], True)


# ================================================ b) 缩集平局 ==


class ShrunkTieSetTest(unittest.TestCase):
    """b) 某实例成本更高（被负载项排除在最优价外）：tie 集不含它；
    游标指向被排除实例时取环序下一个在集成员；游标越全集最大值时
    回绕取首个在集成员。"""

    BEST_NS = 100
    WORSE_NS = 500

    def _candidates(self):
        candidates = []
        for index, cost in (
                (0, self.BEST_NS), (1, self.BEST_NS), (2, self.WORSE_NS)):
            candidates.append(_cand(index, ACTION_STAY, cost))
            candidates.append(_cand(
                index, ACTION_COPY, cost, applicable=False))
        return candidates

    def test_tie_set_excludes_costlier_instance(self):
        _chosen, tie_instances = joint_tie_rotation_pick(
            _applicable_only(self._candidates()), 0, (0, 1, 2))
        self.assertEqual(tie_instances, (0, 1))

    def test_cursor_on_excluded_instance_takes_next_in_set(self):
        # 游标 2 指向被排除实例：环序 2 → 越过 → 回绕取首个在集成员 0。
        chosen, tie_instances = joint_tie_rotation_pick(
            _applicable_only(self._candidates()), 2, (0, 1, 2))
        self.assertEqual(tie_instances, (0, 1))
        self.assertEqual(chosen.instance_index, 0)
        # 对照：游标 1 直接落在在集成员上，取 1（不跳过）。
        chosen_one, _ = joint_tie_rotation_pick(
            _applicable_only(self._candidates()), 1, (0, 1, 2))
        self.assertEqual(chosen_one.instance_index, 1)

    def test_cursor_beyond_max_wraps_to_first_in_set(self):
        chosen, _ = joint_tie_rotation_pick(
            _applicable_only(self._candidates()), 99, (0, 1, 2))
        self.assertEqual(chosen.instance_index, 0)

    def test_integration_load_term_excludes_instance_from_tie(self):
        # 真实 JCM：实例 2 带重负载（负载项抬高其 stay 价），tie 集
        # 收缩为 (0, 1)；游标 2 → 回绕取 0；游标 1 → 取 1；游标 99 →
        # 回绕取 0。
        loads = {0: _load(0), 1: _load(1),
                 2: _load(2, queued_ns=10 ** 9)}
        for cursor, expected in ((2, 0), (1, 1), (99, 0)):
            record = select_instance_and_action(
                mode="joint", cost_model=_cost_model(loads),
                session=_fresh_session(), request=_request(),
                tie_cursor=cursor)
            self.assertEqual(
                record.diagnostics["joint_tie_instances"], (0, 1))
            self.assertEqual(
                record.chosen.instance_index, expected, cursor)


# ======================================= c) 单实例动作平局 ==


class SingleInstanceActionTieTest(unittest.TestCase):
    """c) 同实例 stay/copy 同价：rotated=False、选择与旧确定序恒等
    （ACTION_ORDER 小者胜）、joint_tie_instances 为单元素元组。

    真实 JCM 适用性规则下同一实例 stay/copy 互斥（历史驻留判定互
    补），本组经 _StubModel 注入候选面（见模块 docstring）。"""

    def _record(self, cursor):
        candidates = [
            _cand(0, ACTION_STAY, 100),
            _cand(0, ACTION_COPY, 100),
            _cand(1, ACTION_STAY, 500, applicable=False),
            _cand(1, ACTION_COPY, 500, applicable=False),
        ]
        return select_instance_and_action(
            mode="joint", cost_model=_StubModel(
                {0: _load(0), 1: _load(1)}, candidates),
            session=_fresh_session(), request=_request(),
            tie_cursor=cursor)

    def test_not_rotated_and_old_order_identical_any_cursor(self):
        # 单实例平局不是放置争用，不轮转：任意游标下选择都与旧固定
        # 序 (cost, order_key) 恒等——ACTION_ORDER 小者 stay 胜出。
        for cursor in (0, 1, 7):
            record = self._record(cursor)
            self.assertIs(
                record.diagnostics["joint_tie_rotated"], False, cursor)
            self.assertEqual(
                record.diagnostics["joint_tie_instances"], (0,))
            self.assertEqual(
                (record.chosen.instance_index, record.chosen.action),
                (0, ACTION_STAY))

    def test_order_key_min_within_tied_instance(self):
        # 实例内终裁 = order_key 最小（ACTION_ORDER 序：stay 先于 copy）。
        record = self._record(0)
        self.assertEqual(
            record.chosen.order_key(),
            (0, ACTION_ORDER.index(ACTION_STAY)))


# ========================================= d) 唯一最优实例 ==


class UniqueBestInstanceTest(unittest.TestCase):
    """d) 唯一最优实例：rotated=False，任意游标不变选择。"""

    def _record(self, cursor):
        loads = {0: _load(0), 1: _load(1, queued_ns=10 ** 9)}
        return select_instance_and_action(
            mode="joint", cost_model=_cost_model(loads),
            session=_fresh_session(), request=_request(),
            tie_cursor=cursor)

    def test_any_cursor_keeps_unique_best(self):
        for cursor in (0, 1, 5):
            record = self._record(cursor)
            self.assertIs(
                record.diagnostics["joint_tie_rotated"], False, cursor)
            self.assertEqual(
                record.diagnostics["joint_tie_instances"], (0,))
            self.assertEqual(
                (record.chosen.instance_index, record.chosen.action),
                (0, ACTION_STAY))

    def test_no_tie_note_when_unique_best(self):
        self.assertEqual(self._record(1).tie_break_note, "")


# ============ e) 顺序参照 / face_static 不读 tie_cursor ==


class CursorOpaqueModesTest(unittest.TestCase):
    """e) 顺序模式与 face_static：同输入不同 tie_cursor 值 → record
    逐字段一致（参数不被读）；joint 模式同输入随游标变。"""

    LOADS = {0: _load(0), 1: _load(1)}  # 等负载：joint 跨实例平局场景

    @staticmethod
    def _record_fields(record):
        return (
            (record.chosen.instance_index, record.chosen.action,
             record.chosen.cost_ns),
            record.instance_rule_note,
            record.tie_break_note,
            dict(record.diagnostics),
            tuple(
                (candidate.instance_index, candidate.action,
                 candidate.applicable, candidate.cost_ns)
                for candidate in record.candidates),
        )

    def _select(self, mode, cursor):
        return select_instance_and_action(
            mode=mode, cost_model=_cost_model(self.LOADS),
            session=_fresh_session(), request=_request(),
            tie_cursor=cursor)

    def test_sequential_and_face_static_ignore_cursor(self):
        for mode in ("load-first", "affinity-first", "face_static"):
            base = self._record_fields(self._select(mode, 0))
            for cursor in (1, 3):
                varied = self._record_fields(self._select(mode, cursor))
                self.assertEqual(varied, base, (mode, cursor))

    def test_joint_mode_follows_cursor_on_same_input(self):
        # 对照：joint 主臂同输入下选择随游标变（轮转生效的判别面）。
        at_zero = self._select("joint", 0)
        at_one = self._select("joint", 1)
        self.assertEqual(at_zero.chosen.instance_index, 0)
        self.assertEqual(at_one.chosen.instance_index, 1)
        self.assertEqual(at_zero.diagnostics["joint_tie_cursor"], 0)
        self.assertEqual(at_one.diagnostics["joint_tie_cursor"], 1)


# ==================================== f) 零适用 fail-closed ==


class ZeroApplicableCursorInvarianceTest(unittest.TestCase):
    """f) 零适用 raise：任意 tie_cursor 下 JointSchedulerError 不变
    （copy-only 掩码 × turn-0 会话——M2 形态，沿用
    test_joint_mechanisms 的构造；游标不改变 fail-closed 语义）。"""

    def test_zero_applicable_raises_for_any_cursor(self):
        model = _cost_model(
            {0: _load(0), 1: _load(1)},
            allowed_actions=frozenset({ACTION_COPY}))
        for cursor in (0, 1, 7):
            with self.assertRaises(JointSchedulerError, msg=repr(cursor)):
                select_instance_and_action(
                    mode="joint", cost_model=model,
                    session=_fresh_session(), request=_request(),
                    tie_cursor=cursor)


# ================================== g) diagnostics 三键形状 ==


class DiagnosticsShapeTest(unittest.TestCase):
    """g) diagnostics 三键形状：joint 模式在场且类型正确
    （int/tuple/bool）；顺序参照与 face_static 分支无这三键。"""

    def test_joint_keys_present_and_typed(self):
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            tie_cursor=1)
        cursor = record.diagnostics["joint_tie_cursor"]
        instances = record.diagnostics["joint_tie_instances"]
        rotated = record.diagnostics["joint_tie_rotated"]
        self.assertIs(type(cursor), int)
        self.assertEqual(cursor, 1)  # 本次消费的游标值如实回显
        self.assertIs(type(instances), tuple)
        self.assertEqual(instances, (0, 1))
        self.assertTrue(all(type(item) is int for item in instances))
        self.assertIs(type(rotated), bool)
        self.assertIs(rotated, True)

    def test_joint_unique_best_rotated_false_typed(self):
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1, queued_ns=10 ** 9)}),
            session=_fresh_session(), request=_request(),
            tie_cursor=1)
        self.assertIs(type(record.diagnostics["joint_tie_rotated"]), bool)
        self.assertIs(record.diagnostics["joint_tie_rotated"], False)
        self.assertIs(type(record.diagnostics["joint_tie_instances"]), tuple)
        self.assertEqual(
            record.diagnostics["joint_tie_instances"], (0,))

    def test_other_branches_have_no_tie_keys(self):
        for mode in ("load-first", "affinity-first", "face_static"):
            record = select_instance_and_action(
                mode=mode, cost_model=_cost_model(
                    {0: _load(0, queued_ns=100), 1: _load(1)}),
                session=_fresh_session(), request=_request(),
                tie_cursor=1)
            for key in ("joint_tie_cursor", "joint_tie_instances",
                        "joint_tie_rotated"):
                self.assertNotIn(key, record.diagnostics, (mode, key))


# ==================================== h) tie_break_note 披露 ==


class TieBreakNoteDisclosureTest(unittest.TestCase):
    """h) tie_break_note：跨实例平局文本含实例列表与游标值；单实例
    平局保留 "(instance_index, action order)" 旧文本。"""

    def test_cross_instance_note_lists_instances_and_cursor(self):
        record = select_instance_and_action(
            mode="joint", cost_model=_cost_model(
                {0: _load(0), 1: _load(1)}),
            session=_fresh_session(), request=_request(),
            tie_cursor=1)
        note = record.tie_break_note
        self.assertIn("tie among 2", note)
        self.assertIn("[0, 1]", note)            # 并列实例清单
        self.assertIn("round-robin from cursor 1", note)  # 游标值
        self.assertIn(f"at cost {record.chosen.cost_ns}ns", note)

    def test_single_instance_note_keeps_instance_action_order_text(self):
        # 单实例动作平局（_StubModel 注入，见 c 组）：旧文案保留，
        # 无轮转披露。
        candidates = [
            _cand(0, ACTION_STAY, 100),
            _cand(0, ACTION_COPY, 100),
            _cand(1, ACTION_STAY, 500, applicable=False),
            _cand(1, ACTION_COPY, 500, applicable=False),
        ]
        record = select_instance_and_action(
            mode="joint", cost_model=_StubModel(
                {0: _load(0), 1: _load(1)}, candidates),
            session=_fresh_session(), request=_request(),
            tie_cursor=0)
        self.assertIn("tie among 2", record.tie_break_note)
        self.assertIn("(instance_index, action order)",
                      record.tie_break_note)
        self.assertNotIn("round-robin", record.tie_break_note)


if __name__ == "__main__":
    unittest.main()
