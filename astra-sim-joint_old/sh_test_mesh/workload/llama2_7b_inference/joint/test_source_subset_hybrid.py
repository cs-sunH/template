#!/usr/bin/env python3
"""test_source_subset_hybrid.py -- W10 按来源 hybrid 扩展（V10）零后端
单测（方案 §7.5/§12.1 W10/§11.1 V10；主文 §6.7/§10）。

覆盖（V10 判据逐项）：

* **端点等价**：L=S hybrid 图 ≡ 纯 copy 候选、L=∅ ≡ 纯 remote-read
  （同成本同物理执行——腿集/资源/门/成本审计逐字段，仅身份前缀不同）；
  计价面对端点显式不适用（去重不重复计价）；
* **枚举与去重**：2 源 4 子集、3 源 8 子集完备（端点标 pure 映射回纯
  动作、真子集字典序确定）；|S|<2 空枚举；本地已有区间不进 S；
* **子集预算**：|S| > hybrid_max_remote_sources ⇒ 显式"不支持此扩展"
  （2^|S| 枚举数披露；不静默换别种混合最优）；
* **真实计价**：部分复制占用持久容量（prepare 含 ΣL copy 字节 + R 暂
  存，不免费）；多来源共享瓶颈（目标 HBM/链路竞争进资源集与除数）；
  remote/copy 有效字节核账（n̂ × ΣR / ΣL）；
* **归因分离**：hybrid 候选 notes 标 ``hybrid=1`` + 子集/预算披露；
  pure_four 缺省（action_space 未开）不产生任何 hybrid 候选/入口；
* **distributed 组合**（fixture 级）：hybrid + settlement=distributed
  图合法（零载荷 settle 腿）；
* **选择器面**：纯四动作完整参选且成本零漂移；hybrid 平局键子集字典
  序确定；接续语义不受影响。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest joint/test_source_subset_hybrid.py -q
"""
import os
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_PARENT = os.path.dirname(_HERE)
for _p in (_HERE, _PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.action_work_graph import (  # noqa: E402
    HYBRID_NOTE_TAG,
    SUBSET_ROLE_PROPER,
    SUBSET_ROLE_PURE,
    build_d_candidate,
    effective_remote_sources,
    enumerate_source_subset_plans,
    hybrid_support_status,
)
from joint.boundary_layout import (  # noqa: E402
    INTERVAL_ROLE_BASE,
    INTERVAL_ROLE_INCREMENT,
    BoundaryKVLayout,
    BoundaryKvInterval,
)
from joint.joint_cost_model import (  # noqa: E402
    ACTION_COPY,
    ACTION_REMOTE,
    ACTION_STAY,
    JointCostError,
)
from joint.joint_scheduler import (  # noqa: E402
    select_decode_target_and_action,
)
from joint.prediction_config import (  # noqa: E402
    ACTION_SPACE_SOURCE_SUBSET_HYBRID,
    JointPredictionConfig,
    PredictionParams,
    SETTLEMENT_DISTRIBUTED,
)
from joint.test_action_work_graph import (  # noqa: E402
    _boundary,
    _model,
)

KV_TOKEN_BYTES = 128
BASE_TOKENS = 100
INPUT_TOKENS = 40
P_INSTANCE = 0
TARGET = 1
HOME = 2


def _hybrid_config(**kw):
    return JointPredictionConfig(
        action_space=ACTION_SPACE_SOURCE_SUBSET_HYBRID, **kw)


def _layout(*intervals, home=HOME, p_instance=P_INSTANCE,
            history=BASE_TOKENS + INPUT_TOKENS):
    return BoundaryKVLayout(
        session_id="s", home_instance=home, p_instance_index=p_instance,
        history_tokens=history, kv_token_bytes=KV_TOKEN_BYTES,
        intervals=tuple(intervals))


def _interval(holder, start, end, role=INTERVAL_ROLE_BASE,
              replica=None, tp=2):
    return BoundaryKvInterval(
        replica_id=replica or f"iv_{holder}_{start}_{end}",
        role=role, holder_instance=int(holder), authority=True,
        token_start=int(start), token_end=int(end),
        layer_start=0, layer_end=4,
        bytes_by_tp_rank=(KV_TOKEN_BYTES * (end - start),) * tp)


def _two_source_layout():
    """增量族 @第三目标：base H@home（100 tok）＋ increment U@i_P（40 tok）。"""
    return _layout(
        _interval(P_INSTANCE, BASE_TOKENS, BASE_TOKENS + INPUT_TOKENS,
                  role=INTERVAL_ROLE_INCREMENT),
        _interval(HOME, 0, BASE_TOKENS),
    )


def _three_source_layout():
    """distributed 跨轮留存形态：3 持有者、3 区间（token 不相交）。"""
    return _layout(
        _interval(3, 0, 32, replica="retained_a"),
        _interval(4, 32, 64, replica="retained_b"),
        _interval(P_INSTANCE, 64, 96,
                  role=INTERVAL_ROLE_INCREMENT, replica="working"),
    )


def _cost_audit_keys():
    return (
        "target_wait_ns", "history_prep_ns", "compute_ns", "remote_read_ns",
        "merge_ns", "eviction_wait_ns", "contention_divisor",
        "copy_effective_bytes", "remote_effective_bytes",
        "phase_space_bytes", "staging_buffer_peak_bytes",
        "is_continuation", "move_done_ns",
    )


def _leg_signature(graph, work_id):
    """腿物理签名（剥身份前缀——端点等价比较口径）。"""
    prefix = work_id + "#"
    return tuple(
        (leg.leg_id.replace(prefix, ""),
         leg.kind,
         getattr(leg, "bytes_amount", None),
         tuple(leg.route),
         tuple(d.replace(prefix, "") for d in leg.deps),
         getattr(leg, "latency_ns", None),
         getattr(leg, "duration_ns", None))
        for leg in graph.work.legs)


def _gate_signature(graph, work_id):
    prefix = work_id + "#"
    return tuple(
        (gate.gate_id.replace(prefix, ""), gate.kind, gate.capacity_units)
        for gate in graph.gates)


# ======================================================== 端点等价 ==


class EndpointEquivalenceTest(unittest.TestCase):
    """V10：全 copy/全 remote 端点与纯动作同成本同物理执行。"""

    def test_full_copy_endpoint_equals_pure_copy_graph(self):
        model = _model()
        boundary = _boundary("increment")
        config = JointPredictionConfig(params=PredictionParams(
        window_tile_limit=1, window_local_error_ppm=10000))  # exact 口径（发布默认为 K2 验收档后，直接投影须 exact）
        hybrid = build_d_candidate(
            model, boundary=boundary, instance_index=TARGET, action="hybrid",
            copy_subset=frozenset({P_INSTANCE, HOME}),
            prediction_config=config)
        pure = build_d_candidate(
            model, boundary=boundary, instance_index=TARGET,
            action=ACTION_COPY, prediction_config=config)
        # 同物理执行：腿/资源/门逐字段一致（仅身份前缀差异——去重后端
        # 点复用纯动作候选，不重复计价）。
        self.assertEqual(
            _leg_signature(hybrid, hybrid.work.work_id),
            _leg_signature(pure, pure.work.work_id))
        self.assertEqual(
            sorted(r.key for r in hybrid.resources),
            sorted(r.key for r in pure.resources))
        self.assertEqual(
            _gate_signature(hybrid, hybrid.work.work_id),
            _gate_signature(pure, pure.work.work_id))
        for key in _cost_audit_keys():
            self.assertEqual(
                hybrid.audits[key], pure.audits[key], key)
        # 同成本：前推 settle_done 逐位一致。
        world = model._world_context()
        self.assertEqual(
            hybrid.project_against(world, config.params)
            .predicted_settle_done_ns,
            pure.project_against(world, config.params)
            .predicted_settle_done_ns)

    def test_full_remote_endpoint_equals_pure_remote_graph(self):
        model = _model()
        boundary = _boundary("increment")
        config = JointPredictionConfig(params=PredictionParams(
        window_tile_limit=1, window_local_error_ppm=10000))  # exact 口径（发布默认为 K2 验收档后，直接投影须 exact）
        hybrid = build_d_candidate(
            model, boundary=boundary, instance_index=TARGET, action="hybrid",
            copy_subset=frozenset(), prediction_config=config)
        pure = build_d_candidate(
            model, boundary=boundary, instance_index=TARGET,
            action=ACTION_REMOTE, prediction_config=config)
        self.assertEqual(
            _leg_signature(hybrid, hybrid.work.work_id),
            _leg_signature(pure, pure.work.work_id))
        self.assertEqual(
            sorted(r.key for r in hybrid.resources),
            sorted(r.key for r in pure.resources))
        self.assertEqual(
            _gate_signature(hybrid, hybrid.work.work_id),
            _gate_signature(pure, pure.work.work_id))
        for key in _cost_audit_keys():
            self.assertEqual(
                hybrid.audits[key], pure.audits[key], key)
        world = model._world_context()
        self.assertEqual(
            hybrid.project_against(world, config.params)
            .predicted_settle_done_ns,
            pure.project_against(world, config.params)
            .predicted_settle_done_ns)

    def test_pricing_face_refuses_endpoints_with_dedup_reason(self):
        model = _model()
        model.prediction_config = _hybrid_config()
        boundary = _boundary("increment")
        for subset, name in (
                (frozenset(), "empty"), ({P_INSTANCE, HOME}, "full")):
            candidate = model.estimate_D_hybrid_remaining(
                boundary=boundary, instance_index=TARGET,
                copy_subset=subset, remote_enabled=True)
            self.assertFalse(candidate.applicable, name)
            self.assertIsNone(candidate.cost_ns, name)
            self.assertIn(
                "deduplicates to the pure action candidate",
                candidate.inapplicable_reason, name)


    def test_non_executable_shape_marks_inapplicable(self):
        """预测/执行一致（真实冒烟暴露的失配形态收口）：L 只含 home
        （基础复制而增量留源）在计价面显式判不适用——不产生"已计价不可
        执行"候选（执行面 fail-closed 的对偶；单 span 工作副本域模型外，
        登记 W11）。"""
        model = _model()
        model.prediction_config = _hybrid_config()
        candidate = model.estimate_D_hybrid_remaining(
            boundary=_boundary("increment"), instance_index=TARGET,
            copy_subset=frozenset({HOME}), remote_enabled=True)
        self.assertFalse(candidate.applicable)
        self.assertIsNone(candidate.cost_ns)
        self.assertIn(
            "hybrid shape not executable", candidate.inapplicable_reason)
        self.assertIn("registered W11", candidate.inapplicable_reason)


# ==================================================== 枚举与去重 ==


class SubsetEnumerationTest(unittest.TestCase):
    """V10：子集去重不遗漏合法方案（2 源 4 子集、3 源 8 子集）。"""

    def test_two_sources_enumerate_all_four_subsets(self):
        layout = _two_source_layout()
        self.assertEqual(effective_remote_sources(layout, TARGET),
                         (P_INSTANCE, HOME))
        plans = enumerate_source_subset_plans(layout, TARGET)
        self.assertEqual(len(plans), 4)
        by_subset = {tuple(sorted(p.copy_sources)): p for p in plans}
        self.assertEqual(
            set(by_subset),
            {(), (P_INSTANCE,), (HOME,), (P_INSTANCE, HOME)})
        # 端点标记去重：∅ 映射 remote-read、S 映射 copy。
        self.assertEqual(by_subset[()].role, SUBSET_ROLE_PURE)
        self.assertEqual(by_subset[()].pure_action, ACTION_REMOTE)
        self.assertEqual(
            by_subset[(P_INSTANCE, HOME)].role, SUBSET_ROLE_PURE)
        self.assertEqual(
            by_subset[(P_INSTANCE, HOME)].pure_action, ACTION_COPY)
        # 真子集（2 个）＋ R = S∖L 补全。
        for subset in ((P_INSTANCE,), (HOME,)):
            plan = by_subset[subset]
            self.assertEqual(plan.role, SUBSET_ROLE_PROPER, subset)
            self.assertEqual(
                plan.remote_sources,
                frozenset({P_INSTANCE, HOME}) - frozenset(subset))

    def test_three_sources_enumerate_all_eight_subsets(self):
        layout = _three_source_layout()
        self.assertEqual(
            effective_remote_sources(layout, TARGET), (0, 3, 4))
        plans = enumerate_source_subset_plans(layout, TARGET)
        self.assertEqual(len(plans), 8)
        subsets = {tuple(sorted(p.copy_sources)) for p in plans}
        self.assertEqual(len(subsets), 8)     # 无重复遗漏
        expected = {
            (), (0,), (3,), (4,), (0, 3), (0, 4), (3, 4), (0, 3, 4)}
        self.assertEqual(subsets, expected)
        proper = [p for p in plans if p.role == SUBSET_ROLE_PROPER]
        self.assertEqual(len(proper), 6)
        # 真子集字典序确定（平局键确定性）。
        self.assertEqual(
            [p.order_key for p in proper],
            sorted(p.order_key for p in proper))

    def test_single_source_layout_has_no_subsets(self):
        # 完整族单源（S={i_P}）——无真子集，枚举空（非错误）。
        layout = _layout(_interval(P_INSTANCE, 0, 100))
        self.assertEqual(enumerate_source_subset_plans(layout, TARGET), ())

    def test_locally_held_intervals_excluded_from_sources(self):
        # 目标已持有的区间不进子集搜索（missing 排除 target-held）。
        layout = _layout(
            _interval(TARGET, 0, 50, replica="local_copy"),
            _interval(HOME, 50, 100),
        )
        self.assertEqual(effective_remote_sources(layout, TARGET), (HOME,))
        self.assertEqual(enumerate_source_subset_plans(layout, TARGET), ())


# ======================================================== 子集预算 ==


class SubsetBudgetTest(unittest.TestCase):
    """V10：超限显式不支持（不静默换别种"混合最优"）。"""

    def _many_source_layout(self, count):
        span = 32
        return _layout(*(
            _interval(10 + index, index * span, (index + 1) * span,
                      replica=f"src_{index}")
            for index in range(count)))

    def test_over_budget_reports_explicit_unsupported(self):
        layout = self._many_source_layout(5)   # |S|=5 > 预算 4
        status = hybrid_support_status(
            layout, TARGET, JointPredictionConfig())
        self.assertFalse(status.supported)
        self.assertIn("hybrid_max_remote_sources=4", status.reason)
        self.assertEqual(status.source_count, 5)
        self.assertEqual(status.enumeration_count, 2 ** 5)
        self.assertEqual(status.proper_subset_count, 2 ** 5 - 2)

    def test_at_budget_supported_with_enumeration_disclosure(self):
        layout = self._many_source_layout(4)
        status = hybrid_support_status(
            layout, TARGET, JointPredictionConfig())
        self.assertTrue(status.supported)
        self.assertEqual(status.enumeration_count, 16)
        custom = JointPredictionConfig(
            params=PredictionParams(hybrid_max_remote_sources=3))
        self.assertFalse(hybrid_support_status(
            layout, TARGET, custom).supported)

    def test_selector_discloses_unsupported_and_keeps_pure_four(self):
        from joint.action_work_graph import HybridSupportStatus
        model = _model()
        model.prediction_config = _hybrid_config()
        original = model.hybrid_support_status

        def capped(*, boundary, instance_index, _orig=original):
            """把支持状态强制压到超限（选择器 wiring 的测试缝——不改
            生产面；|S|>预算的真实态由上方 status 级测试承载）。"""
            status = _orig(boundary=boundary, instance_index=instance_index)
            if status.supported:
                return HybridSupportStatus(
                    supported=False,
                    reason="source subset hybrid not supported: "
                           "effective remote source count 2 exceeds "
                           "hybrid_max_remote_sources=1",
                    source_count=2, enumeration_count=4,
                    proper_subset_count=2)
            return status

        model.hybrid_support_status = capped
        record = select_decode_target_and_action(
            mode="joint", cost_model=model,
            boundary=_boundary("increment"), remote_enabled=True)
        # 超限：hybrid 面显式不支持——零 hybrid 候选 + 诊断披露（预算/
        # 枚举数），不静默换别种混合最优。
        self.assertEqual(
            [c for c in record.candidates if c.action == "hybrid"], [])
        unsupported = record.diagnostics.get("hybrid_unsupported")
        self.assertTrue(unsupported)
        self.assertEqual(unsupported[0]["source_count"], 2)
        self.assertEqual(unsupported[0]["enumeration_count"], 4)
        self.assertIn("hybrid_max_remote_sources=1",
                      unsupported[0]["reason"])
        # 纯四动作完整参选不受影响。
        self.assertEqual(
            len([c for c in record.candidates
                 if c.action != "hybrid"]), 12)


# ==================================================== 真实计价 ==


class HybridPricingTest(unittest.TestCase):
    """V10：部分复制真实占用持久容量；共享瓶颈进成本；字节核账。"""

    def _hybrid(self, copy_subset, config=None):
        model = _model()
        model.prediction_config = (
            config if config is not None else _hybrid_config())
        return model, model.estimate_D_hybrid_remaining(
            boundary=_boundary("increment"), instance_index=TARGET,
            copy_subset=frozenset(copy_subset), remote_enabled=True)

    def test_partial_copy_occupies_persistent_capacity(self):
        # 可执行形态 L={i_P}（U 迁 j）＋ R={home} 远读（§6.7 动机形态）。
        model, candidate = self._hybrid({P_INSTANCE})
        self.assertTrue(candidate.applicable)
        notes = list(candidate.breakdown.notes)
        # 部分复制不免费：持久 copy 腿（mvS）真实存在且带 ΣL 字节
        # （纯 remote 无任何 copy 腿）——L 字节进目标 prepare 峰值。
        graph = build_d_candidate(
            model, boundary=_boundary("increment"), instance_index=TARGET,
            action="hybrid", copy_subset=frozenset({P_INSTANCE}),
            prediction_config=JointPredictionConfig())
        copy_legs = [
            leg for leg in graph.work.legs
            if "#mvS" in leg.leg_id
            and getattr(leg, "bytes_amount", 0)]
        u_bytes = KV_TOKEN_BYTES * INPUT_TOKENS * 2
        self.assertEqual(
            sum(leg.bytes_amount for leg in copy_legs), u_bytes)
        pure_remote_graph = build_d_candidate(
            model, boundary=_boundary("increment"), instance_index=TARGET,
            action=ACTION_REMOTE, prediction_config=JointPredictionConfig())
        self.assertFalse([
            leg for leg in pure_remote_graph.work.legs
            if "#mvS" in leg.leg_id])
        prepare = candidate.breakdown.phase_space_bytes[0]
        self.assertGreaterEqual(
            max(prepare), u_bytes,
            "persistent copy bytes must enter the prepare footprint")

    def test_effective_bytes_accounting(self):
        # L={i_P}（U 持久 copy）＋R={home}（H 每遍远读）：
        # copy_effective = ΣL（40 tok × 128 × 2 rank）、
        # remote_effective = n̂ × ΣR（8 × 100 × 128 × 2 rank）。
        model = _model()
        graph = build_d_candidate(
            model, boundary=_boundary("increment"), instance_index=TARGET,
            action="hybrid", copy_subset=frozenset({P_INSTANCE}),
            prediction_config=JointPredictionConfig())
        self.assertEqual(
            graph.audits["copy_effective_bytes"],
            KV_TOKEN_BYTES * INPUT_TOKENS * 2)
        self.assertEqual(
            graph.audits["remote_effective_bytes"],
            8 * KV_TOKEN_BYTES * BASE_TOKENS * 2)

    def test_shared_bottleneck_resources_compete(self):
        # 两链共享目标 HBM/链路：copy 腿与 credit 读流的资源集都进图。
        graph = build_d_candidate(
            _model(), boundary=_boundary("increment"), instance_index=TARGET,
            action="hybrid", copy_subset=frozenset({P_INSTANCE}),
            prediction_config=JointPredictionConfig())
        resource_keys = {r.key for r in graph.resources}
        self.assertIn(f"hbm:{TARGET}", resource_keys)      # 目标 HBM 共享
        copy_links = [k for k in resource_keys
                      if k.startswith(f"link:{2 * P_INSTANCE}->")]
        remote_links = [k for k in resource_keys
                        if k.startswith(f"link:{2 * HOME}->")]
        self.assertTrue(copy_links, "copy legs' NoC links must be present")
        self.assertTrue(remote_links,
                        "remote credit legs' NoC links must be present")
        # 争用除数 ≥ 1（两条链各自登记 path family 竞争）。
        self.assertGreaterEqual(graph.audits["contention_divisor"], 1)

    def test_hybrid_notes_and_attribution_separation(self):
        _model_inst, candidate = self._hybrid({P_INSTANCE})
        notes = list(candidate.breakdown.notes)
        self.assertIn(HYBRID_NOTE_TAG, notes)
        self.assertIn("hybrid_copy_sources=0", notes)
        self.assertIn("hybrid_remote_sources=2", notes)
        self.assertIn("hybrid_effective_source_count=2", notes)
        self.assertIn("hybrid_subset_enumeration=4", notes)
        self.assertEqual(candidate.breakdown.decode_action, "hybrid")
        # 纯四动作候选不带 hybrid 标记（归因分离）。
        model = _model()
        model.prediction_config = _hybrid_config()
        for action in ("stay", "copy", "remote-read", "recompute"):
            pure = model.estimate_D_remaining(
                boundary=_boundary("increment"), instance_index=TARGET,
                action=action, remote_enabled=True)
            if pure.breakdown is None:
                continue
            self.assertNotIn(
                HYBRID_NOTE_TAG, pure.breakdown.notes, action)

    def test_pure_four_default_never_prices_hybrid(self):
        model = _model()      # 缺省 pure_four
        self.assertFalse(model.hybrid_action_space_enabled())
        with self.assertRaises(JointCostError):
            model.estimate_D_hybrid_remaining(
                boundary=_boundary("increment"), instance_index=TARGET,
                copy_subset=frozenset({HOME}), remote_enabled=True)

    def test_distributed_combination_is_legal(self):
        model, candidate = self._hybrid(
            {P_INSTANCE}, config=_hybrid_config(
                settlement_policy=SETTLEMENT_DISTRIBUTED))
        self.assertTrue(candidate.applicable)
        self.assertIsNotNone(candidate.cost_ns)
        # 零载荷 settle 腿（W9 §7.4 note 在场）。
        self.assertTrue(any(
            str(note).startswith("settlement=distributed")
            for note in candidate.breakdown.notes))
        # 与 consolidate 同图的 merge 载荷差 = distributed 组合真实生效。
        _m2, consolidate = self._hybrid({P_INSTANCE})
        self.assertGreater(
            consolidate.breakdown.merge_ns, candidate.breakdown.merge_ns)


# ==================================================== 选择器面 ==


class SelectorFaceTest(unittest.TestCase):
    """V10：纯四动作完整参选不受影响；平局键确定；接续语义不变。"""

    def test_hybrid_candidates_join_pure_four(self):
        boundary = _boundary("increment")
        model = _model()
        model.prediction_config = _hybrid_config()
        record = select_decode_target_and_action(
            mode="joint", cost_model=model, boundary=boundary,
            remote_enabled=True)
        hybrid = [c for c in record.candidates if c.action == "hybrid"]
        pure = [c for c in record.candidates if c.action != "hybrid"]
        # 纯四动作 × 3 实例完整在场 + 2 真子集 hybrid（仅 @第三目标
        # 的 S=2 来源面）；可执行形态（i_P ∈ L）适用、域模型外形态
        # （L 只含 home）显式不适用（预测/执行一致）。
        self.assertEqual(len(pure), 12)
        self.assertEqual(len(hybrid), 2)
        self.assertEqual(
            [tuple(sorted(c.copy_subset)) for c in hybrid],
            [(P_INSTANCE,), (HOME,)])
        for candidate in hybrid:
            self.assertFalse(candidate.is_continuation)
        applicable_hybrid = [c for c in hybrid if c.applicable]
        self.assertEqual(
            [tuple(sorted(c.copy_subset)) for c in applicable_hybrid],
            [(P_INSTANCE,)])
        blocked = [c for c in hybrid
                   if tuple(sorted(c.copy_subset)) == (HOME,)]
        self.assertEqual(len(blocked), 1)
        self.assertFalse(blocked[0].applicable)
        self.assertIn(
            "not executable", blocked[0].inapplicable_reason)

    def test_pure_candidate_costs_zero_drift_under_extension(self):
        boundary = _boundary("increment")
        pure_model = _model()
        pure_record = select_decode_target_and_action(
            mode="joint", cost_model=pure_model, boundary=boundary,
            remote_enabled=True)
        hybrid_model = _model()
        hybrid_model.prediction_config = _hybrid_config()
        hybrid_record = select_decode_target_and_action(
            mode="joint", cost_model=hybrid_model, boundary=boundary,
            remote_enabled=True)
        pure_costs = {
            (c.instance_index, c.action): c.cost_ns
            for c in pure_record.candidates if c.cost_ns is not None}
        for candidate in hybrid_record.candidates:
            if candidate.action == "hybrid" or candidate.cost_ns is None:
                continue
            self.assertEqual(
                pure_costs[(candidate.instance_index, candidate.action)],
                candidate.cost_ns,
                "pure-four costs must not drift under the hybrid face")

    def test_tie_break_key_is_deterministic_subset_order(self):
        boundary = _boundary("increment")
        model = _model()
        model.prediction_config = _hybrid_config()
        record = select_decode_target_and_action(
            mode="joint", cost_model=model, boundary=boundary,
            remote_enabled=True)
        hybrid = [c for c in record.candidates if c.action == "hybrid"]
        keys = [c.order_key() for c in hybrid]
        self.assertEqual(keys, sorted(keys))
        self.assertEqual(
            keys[0], (TARGET, 4, (P_INSTANCE,)))
        self.assertEqual(keys[1], (TARGET, 4, (HOME,)))

    def test_pure_four_default_face_unchanged(self):
        boundary = _boundary("increment")
        model = _model()
        record = select_decode_target_and_action(
            mode="joint", cost_model=model, boundary=boundary,
            remote_enabled=True)
        self.assertEqual(
            [c for c in record.candidates if c.action == "hybrid"], [])
        self.assertNotIn("hybrid_unsupported", record.diagnostics)
        # 接续候选（remote-read@i_P 增量族）照常在场且平局优先。
        continuations = [c for c in record.candidates
                         if c.is_continuation and c.applicable]
        self.assertTrue(continuations)

    def test_remote_disabled_marks_hybrid_inapplicable(self):
        # remote 关闭时 R 读流无能力——hybrid 判不适用（不静默换别种
        # 混合形态）；增量族接续候选同样不适用属既有语义，不经选择器
        # 面构造矛盾状态，直接驱动计价入口。
        model = _model()
        model.prediction_config = _hybrid_config()
        for subset in ({P_INSTANCE}, {HOME}):
            candidate = model.estimate_D_hybrid_remaining(
                boundary=_boundary("increment"), instance_index=TARGET,
                copy_subset=frozenset(subset), remote_enabled=False)
            self.assertFalse(candidate.applicable)
            self.assertIsNone(candidate.cost_ns)
            self.assertEqual(
                candidate.inapplicable_reason, "remote actions disabled")


if __name__ == "__main__":
    unittest.main()
