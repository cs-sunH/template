#!/usr/bin/env python3
"""test_event_recursion_predictor.py -- C15 E 运行期事件递推预测器的零后端
验证（2026-09-22；仓库设计方案 §5.2/§5.4/§5.5/§5.7；F12）。

覆盖（§5.7 清单的模块级项）：
  1. 解析例回归：L=32、c=1ms、q=0、r=0.5/2/4ms → k=1/17/25；k=0/L
     边界、首层赶上后层断供、无缺失层；
  2. 剪枝口径：解析下界 = 逐 rank 独享速率串行累计（非 Σ_j max_r 逐层
     近似——后者会误剪，反例断言）；剪枝枚举 vs 全枚举一致性抽检；
  3. 期限钉死：D̂ = 无候选恢复流量的消费时刻；候选自致计算减速单列
     （不回移期限）；
  4. 写回方向：对象顺序逐 victim 决策 + 已定写回流登记进共享快照
     （同批 victim 不得各自按独享带宽估算）；写回×恢复合流（q̂ 关键
     路径 max）；
  5. 状态标注：未知释放 ETA / 解析不可用（自 0 全枚举 + fallback 来源）
     / 遥测覆盖不足——不当零代价、不删候选；
  6. 重试复核断言口径（§5.5 末段）：同一拆分重交 + 保护/合法性复核；
  7. η/γ 因果更新规则（θ/α/τ/首样本初始化/无效样本拒绝/缺可分离样本
     保持原估计并标记不可观测）；
  8. R/D 方向用例：长历史短输入、带宽高低、竞争升降各 ≥1 例。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_event_recursion_predictor.py   （或 pytest 同路径）
"""
import math
import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _p in (_JOINT_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.event_recursion_predictor import (  # noqa: E402
    CommittedFlow,
    ComputeLayerSegment,
    ConservationViolationError,
    EfficiencyFactors,
    EventRecursionError,
    INVALIDATE_INTERVAL_RELEASED,
    INVALIDATE_LAYER_IN_USE,
    KHidePrediction,
    LayerRecursionPredictor,
    PREDICTOR_SOURCE_ANALYTIC_FALLBACK,
    PREDICTOR_SOURCE_RECURSION,
    RESUBMIT_SAME,
    ResourceSnapshot,
    RestoreGroupLeg,
    STATUS_ANALYTIC_UNAVAILABLE,
    STATUS_UNKNOWN_RELEASE_ETA,
    ServiceFactorGroup,
    UnifiedTimeline,
    WritebackVictim,
    CommittedPlanView,
    predict_release_and_recall,
    predict_space_available,
    revalidate_committed_plan,
)
from joint.layer_eviction_policy import k_hide_deadline  # noqa: E402


def _uniform_case(layers, r_ns, c_ns=1000.0, *, q_ns=0, ranks=1,
                  pool_peak=1000.0, port_peak=1000.0, committed=(),
                  memory_bytes=0, group="prefill", gamma=1.0):
    """单/对称多 rank 均匀层夹具（逐层腿；路径 = 池→端口）。"""
    legs = tuple(
        RestoreGroupLeg(
            layer_start=j, layer_end=j + 1,
            bytes_by_rank=(int(r_ns * pool_peak),) * ranks,
            path_by_rank=(("pool:0", "port:0"),) * ranks,
        )
        for j in range(layers)
    )
    segments = tuple(
        ComputeLayerSegment(
            layer=ell,
            base_ns_by_rank=(c_ns,) * ranks,
            memory_bytes_by_rank=(memory_bytes,) * ranks,
            port_by_rank=("port:0",) * ranks,
            group=group,
        )
        for ell in range(1, layers + 1)
    )
    snapshot = ResourceSnapshot(
        peak_bytes_per_ns={"pool:0": pool_peak, "port:0": port_peak},
        committed=committed,
    )
    efficiency = EfficiencyFactors(
        eta={"pool": 1.0}, gamma={group: gamma})
    return LayerRecursionPredictor(snapshot, legs, segments, q_ns, efficiency)


class AnalyticFixtureTests(unittest.TestCase):
    """§5.7-1：解析例回归 + 边界。"""

    def test_L32_fixture_1_17_25(self):
        for r_ns, expected in ((500, 1), (2000, 17), (4000, 25)):
            predictor = _uniform_case(32, r_ns)
            result = predictor.predict_k_hide()
            self.assertEqual(result.k_hide, expected, f"r={r_ns}")
            # 剪枝起点的解析最小可行 k 与闭式公式一致（对称单 rank）。
            closed = k_hide_deadline([1000] * 32, [r_ns] * 32, 0)
            self.assertEqual(result.analytic_min_k, closed.k_hide)
            self.assertEqual(result.source, PREDICTOR_SOURCE_RECURSION)
            self.assertEqual(result.selected_trace.exposed_stall_ns, 0)

    def test_r_equals_2ms_k16_exposes_1ms(self):
        # §5.7 注：r=2ms 保留 16 层在末层暴露 1ms（2*16=32 > 31）——
        # 不能判为完全隐藏；递推口径下 k=16 trace 的 Ŝ = 1000。
        predictor = _uniform_case(32, 2000)
        deadlines, _reference = predictor.reference_deadlines()
        trace16 = predictor._trace_for_k(16, deadlines, _reference)
        self.assertEqual(trace16.exposed_stall_ns, 1000)
        self.assertFalse(trace16.feasible)
        self.assertEqual(trace16.binding_layer, 32)

    def test_k0_blocked_by_first_layer_and_kL_boundary(self):
        # q=0、r>0、D̂_1=0 ⇒ k=0 永不满足首层。
        result = _uniform_case(4, 50, c_ns=100).predict_k_hide()
        self.assertEqual(result.k_hide, 1)
        # r=0（无缺失字节代价）⇒ k=0。
        result0 = _uniform_case(4, 0, c_ns=100).predict_k_hide()
        self.assertEqual(result0.k_hide, 0)
        # 无缺失层：k=L 是公式边界（条件为空），binding=None。
        slow = _uniform_case(4, 10**9, c_ns=10).predict_k_hide()
        self.assertEqual(slow.k_hide, 4)
        self.assertIsNone(slow.selected_trace.binding_layer)
        # 首层赶上但后层断供：late 层恢复特慢 ⇒ binding 在 late 层。
        legs = (
            RestoreGroupLeg(0, 1, (50,), (("pool:0", "port:0"),)),
            RestoreGroupLeg(1, 2, (50,), (("pool:0", "port:0"),)),
            RestoreGroupLeg(2, 3, (2_000_000,), (("pool:0", "port:0"),)),
            RestoreGroupLeg(3, 4, (50,), (("pool:0", "port:0"),)),
        )
        segments = tuple(
            ComputeLayerSegment(ell, (100.0,), (0,), ("port:0",))
            for ell in range(1, 5))
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"pool:0": 1000.0, "port:0": 1000.0})
        predictor = LayerRecursionPredictor(snapshot, legs, segments, 0)
        result_late = predictor.predict_k_hide()
        # 保留 2 层时层 3 恢复 2000ns > D̂_3=200ns ⇒ k=3（层 3 驻留）。
        self.assertEqual(result_late.k_hide, 3)
        # k=2 trace 的绑定层 = 层 3（2000 > 200，后层断供如实入账）。
        deadlines, reference = predictor.reference_deadlines()
        trace_k2 = predictor._trace_for_k(2, deadlines, reference)
        self.assertFalse(trace_k2.feasible)
        self.assertEqual(trace_k2.binding_layer, 3)
        self.assertEqual(trace_k2.exposed_stall_ns, 2000 - 200)


class PruningRuleTests(unittest.TestCase):
    """§5.2 剪枝口径（钉死两条）。"""

    def test_bound_is_per_rank_serial_not_per_layer_max(self):
        # 反例构造：2 rank 交替持有大字节层——Σ_j max_r r̂_j（逐层近似）
        # 远大于递推 R̂（每层只有持有大字节的 rank 慢；跨层两 rank 并行
        # 串行链各自累计）。按 Σ_j max_r 剪枝会误剪可行 k；按钉死口径
        # （逐 rank 串行累计取 max）不会。
        legs = []
        for j in range(8):
            if j % 2 == 0:
                bytes_by_rank = (4000, 40)
            else:
                bytes_by_rank = (40, 4000)
            legs.append(RestoreGroupLeg(
                layer_start=j, layer_end=j + 1,
                bytes_by_rank=bytes_by_rank,
                path_by_rank=(("pool:0", "port:0"),
                              ("pool:1", "port:1")),
            ))
        segments = tuple(
            ComputeLayerSegment(ell, (500.0, 500.0), (0, 0),
                                ("port:0", "port:1"))
            for ell in range(1, 9))
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={
                "pool:0": 1000.0, "pool:1": 1000.0,
                "port:0": 1000.0, "port:1": 1000.0})
        predictor = LayerRecursionPredictor(snapshot, tuple(legs), segments, 0)
        bound = predictor.exclusive_rank_serial_bound_ns(0, 8)
        # 逐 rank 串行：每 rank 大层 4 × 4000/1000 + 小层 4 × 40/1000。
        self.assertAlmostEqual(bound, 4 * 4.0 + 4 * 0.04)
        # 逐层近似 Σ_j max_r = 8 × 4.0 = 32 —— 显著大于钉死下界。
        per_layer_max_sum = 8 * 4.0
        self.assertLess(bound, per_layer_max_sum)
        # 递推可行性与全枚举一致（无漏剪）。
        pruned = predictor.predict_k_hide(prune_from_analytic=True)
        full = predictor.predict_k_hide(prune_from_analytic=False)
        self.assertEqual(pruned.k_hide, full.k_hide)

    def test_prune_vs_full_enumeration_consistency(self):
        # §5.7 抽检：抽样决策两口径分别运行，选层结果必须一致。
        import random
        rng = random.Random(20260922)
        for trial in range(60):
            n = rng.randint(2, 12)
            legs = tuple(
                RestoreGroupLeg(
                    j, j + 1,
                    (rng.randint(1, 400), rng.randint(1, 400)),
                    ((("pool:0", "port:0") if j % 2 == 0
                      else ("pool:1", "port:1")),) * 2)
                for j in range(n))
            segments = tuple(
                ComputeLayerSegment(
                    ell,
                    (float(rng.randint(50, 200)),
                     float(rng.randint(50, 200))),
                    (0, 0), ("port:0", "port:1"))
                for ell in range(1, n + 1))
            snapshot = ResourceSnapshot(
                peak_bytes_per_ns={
                    "pool:0": rng.choice([1.0, 2.0, 4.0]),
                    "pool:1": rng.choice([1.0, 2.0, 4.0]),
                    "port:0": 2.0, "port:1": 2.0})
            predictor = LayerRecursionPredictor(
                snapshot, legs, segments, rng.choice([0, 50]))
            a = predictor.predict_k_hide(prune_from_analytic=True)
            b = predictor.predict_k_hide(prune_from_analytic=False)
            self.assertEqual(
                a.k_hide, b.k_hide, f"trial {trial}: prune/full mismatch")


class DeadlinePinningTests(unittest.TestCase):
    """§5.2 剪枝段其二：期限钉死 + 候选自致减速单列。"""

    def test_deadlines_do_not_move_with_candidate_contention(self):
        # 恢复写腿与计算 memory 腿同端口竞争：候选变大 ⇒ 消费推迟（减速
        # 单列 > 0），但 D̂（reference 递推，无候选恢复流量）不变。
        def run(candidate_bytes):
            legs = (
                RestoreGroupLeg(0, 1, (candidate_bytes,),
                                (("pool:0", "port:0"),)),
                RestoreGroupLeg(1, 2, (candidate_bytes,),
                                (("pool:0", "port:0"),)),
            )
            segments = tuple(
                ComputeLayerSegment(ell, (100.0,), (500,), ("port:0",))
                for ell in range(1, 3))
            snapshot = ResourceSnapshot(
                peak_bytes_per_ns={"pool:0": 2.0, "port:0": 2.0})
            predictor = LayerRecursionPredictor(snapshot, legs, segments, 0)
            deadlines, reference = predictor.reference_deadlines()
            trace0 = predictor._trace_for_k(0, deadlines, reference)
            return deadlines, trace0

        deadlines_small, trace_small = run(400)
        deadlines_big, trace_big = run(4000)
        self.assertEqual(deadlines_small, deadlines_big)  # 期限钉死
        self.assertGreater(
            trace_big.compute_slowdown_ns, trace_small.compute_slowdown_ns)
        self.assertGreater(trace_big.compute_slowdown_ns, 0)
        # k=0 下减速单列且可行性判定用 R̂ vs 钉死 D̂（首层 R̂>0=D̂_1）。
        self.assertFalse(trace_small.feasible)

    def test_compute_slowdown_zero_without_candidates(self):
        predictor = _uniform_case(4, 100, c_ns=100, memory_bytes=500)
        deadlines, reference = predictor.reference_deadlines()
        trace_full = predictor._trace_for_k(4, deadlines, reference)
        self.assertEqual(trace_full.compute_slowdown_ns, 0)


class WritebackDirectionTests(unittest.TestCase):
    """§5.2 递推段：写回方向对象顺序 + 同批共享快照。"""

    def _snapshot(self):
        return ResourceSnapshot(
            peak_bytes_per_ns={
                "pool:0": 1.0, "port:0": 1.0, "port:1": 1.0})

    def test_sequential_victims_share_bandwidth(self):
        # 三个 victim 共享池端口（F2 修正后按分段恒速积分结算，ETA 截断
        # 区间内已定 victim 的预留份额照样服务候选字节）：
        #   v1 独享 [0,1000) → 1000；
        #   v2 与 v1 均分池 [0,1000)（服务 500）后独享 → 1000+500=1500；
        #   v3 三分 [0,1000)（1000/3）、二分 [1000,1500)（500）、独享余量
        #     → 1916.67 → ceil 1917。
        # 同批不得各自按独享带宽估算（各 1000 错；修复前 [1000,2000,3000]
        # 为截断区间字节凭空消失的缺陷值）。
        victims = (
            WritebackVictim("v1", (1000,), (("port:0", "pool:0"),)),
            WritebackVictim("v2", (1000,), (("port:1", "pool:0"),)),
            WritebackVictim("v3", (1000,), (("port:0", "pool:0"),)),
        )
        result = predict_space_available(self._snapshot(), victims)
        self.assertEqual(
            [v.space_available_ns for v in result.victims],
            [1000, 1500, 1917])

    def test_unknown_eta_annotated_not_zero_cost(self):
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"pool:0": 1.0, "port:0": 1.0},
            committed=(CommittedFlow("bg", ("pool:0",), None),))
        victim = WritebackVictim("v1", (1000,), (("port:0", "pool:0"),))
        result = predict_space_available(snapshot, (victim,))
        self.assertEqual(result.victims[0].space_available_ns, 2000)
        self.assertIn(STATUS_UNKNOWN_RELEASE_ETA, result.statuses)
        # 候选不被删、不当零代价：ETA 未知流持续占用份额（分母 +1）。

    def test_release_and_recall_merged_qhat(self):
        # 写回流空间可用抬升 q̂（接收空间依赖）且恢复在含写回流的共享
        # 快照上递推（k 变大）。
        snapshot = self._snapshot()
        victims = (WritebackVictim("v1", (1000,), (("port:0", "pool:0"),)),)
        legs = tuple(
            RestoreGroupLeg(j, j + 1, (10,), (("pool:0", "port:0"),))
            for j in range(4))
        segments = tuple(
            ComputeLayerSegment(ell, (5.0,), (0,), ("port:0",))
            for ell in range(1, 5))
        merged = predict_release_and_recall(
            snapshot, victims, legs, segments, first_block_wait_ns=0)
        self.assertEqual(merged.first_block_wait_ns, 1000)
        solo = LayerRecursionPredictor(
            snapshot, legs, segments, 0).predict_k_hide()
        self.assertGreaterEqual(merged.k_hide.k_hide, solo.k_hide)


class StatusAnnotationTests(unittest.TestCase):
    """§5.2：状态标注（不当零代价、不删实例候选）。"""

    def test_analytic_unavailable_enumerates_from_zero(self):
        # 调用方显式声明解析校验不可用（analytic_min_k=None）⇒ 自 0 全
        # 枚举 + fallback 来源 + 状态标注；递推仍给结果（候选不删）。
        predictor = _uniform_case(4, 100)
        result = predictor.predict_k_hide(analytic_min_k=None)
        self.assertEqual(result.source, PREDICTOR_SOURCE_ANALYTIC_FALLBACK)
        self.assertIsNone(result.analytic_min_k)
        self.assertIn(STATUS_ANALYTIC_UNAVAILABLE, result.statuses)
        baseline = predictor.predict_k_hide()
        self.assertEqual(result.k_hide, baseline.k_hide)

    def test_empty_path_leg_fails_closed_not_zero_cost(self):
        # 有字节而无路径 = 缺模型事实：fail-closed（不得当零代价——
        # 无路径意味着该 rank 无法恢复，静默按零会误导剪枝）。
        segments = (ComputeLayerSegment(1, (100.0,), (0,), ("port:0",)),)
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"pool:0": 1.0, "port:0": 1.0})
        with self.assertRaises(EventRecursionError):
            legs = (RestoreGroupLeg(0, 1, (100,), ((),)),)
            LayerRecursionPredictor(snapshot, legs, segments, 0)

    def test_coverage_statuses_disclosed(self):
        predictor = _uniform_case(4, 100)
        result = predictor.predict_k_hide()
        self.assertIn("coverage_no_collective_telemetry", result.statuses)
        self.assertIn("coverage_no_link_model", result.statuses)
        self.assertIn("coverage_port_registry_executor", result.statuses)


class RevalidationRuleTests(unittest.TestCase):
    """§5.5 末段/§5.6：已提交计划的重试复核断言口径。"""

    def test_resubmit_same_when_only_background_flow_changed(self):
        steps = (CommittedPlanView("victim", 24, 32),)
        # 仅流量观测变化（无在用/在途、区间有效）⇒ 重交同一拆分。
        self.assertEqual(
            revalidate_committed_plan(
                steps, layers_now_in_use_or_inflight=(),
                still_valid_interval=True),
            RESUBMIT_SAME)

    def test_invalidate_when_layer_in_use_or_inflight(self):
        steps = (CommittedPlanView("victim", 24, 32),)
        self.assertEqual(
            revalidate_committed_plan(
                steps, layers_now_in_use_or_inflight=(28,),
                still_valid_interval=True),
            INVALIDATE_LAYER_IN_USE)

    def test_invalidate_when_interval_released(self):
        steps = (CommittedPlanView("victim", 24, 32),)
        self.assertEqual(
            revalidate_committed_plan(
                steps, layers_now_in_use_or_inflight=(),
                still_valid_interval=False),
            INVALIDATE_INTERVAL_RELEASED)

    def test_empty_plan_resubmits(self):
        self.assertEqual(
            revalidate_committed_plan(
                (), layers_now_in_use_or_inflight=(1,),
                still_valid_interval=False),
            RESUBMIT_SAME)


class ServiceFactorTests(unittest.TestCase):
    """§5.4：η/γ 因果更新规则（θ/α/τ/首样本/无效样本/不可观测）。"""

    def test_first_sample_initializes_then_ewma(self):
        group = ServiceFactorGroup(initial=1.0)
        value = group.observe_valid_service(
            "prefill", observed_ratio=2.0, completion_ns=1000,
            service_duration_ns=800)
        self.assertEqual(value, 2.0)
        # α = 1 − exp(−Δt/τ)，τ = 上一有效样本正服务时长 800。
        value2 = group.observe_valid_service(
            "prefill", observed_ratio=1.0, completion_ns=1800,
            service_duration_ns=1000)
        alpha = 1.0 - math.exp(-800 / 800)
        self.assertAlmostEqual(value2, (1 - alpha) * 2.0 + alpha * 1.0)
        self.assertFalse(group.cold_start)

    def test_invalid_sample_rejected_and_recorded(self):
        group = ServiceFactorGroup()
        group.observe_valid_service(
            "pool", observed_ratio=1.5, completion_ns=100,
            service_duration_ns=100)
        value = group.observe_valid_service(
            "pool", observed_ratio=-1.0, completion_ns=300,
            service_duration_ns=100)
        self.assertEqual(value, 1.5)  # 拒绝后保持原估计
        self.assertIn("invalid_ratio", group.rejected_updates("pool"))

    def test_mixed_wait_marked_unobservable_keeps_estimate(self):
        group = ServiceFactorGroup()
        group.observe_valid_service(
            "pool", observed_ratio=1.2, completion_ns=100,
            service_duration_ns=100)
        group.mark_unobservable("pool", reason="mixed_wait")
        self.assertEqual(group.value("pool"), 1.2)
        self.assertEqual(group.unobservable_samples["pool"], 1)
        snapshot = group.snapshot()
        self.assertEqual(
            snapshot["groups"]["pool"]["unobservable_samples"], 1)

    def test_missing_tau_rejected(self):
        group = ServiceFactorGroup()
        group.observe_valid_service(
            "prefill", observed_ratio=1.0, completion_ns=100,
            service_duration_ns=None)
        # 首样本缺 τ 也可初始化（LEP 口径）；第二样本必须携带 τ。
        value = group.observe_valid_service(
            "prefill", observed_ratio=2.0, completion_ns=200,
            service_duration_ns=None)
        self.assertEqual(value, 1.0)
        self.assertIn("missing_tau", group.rejected_updates("prefill"))


class TimelineIntegrationTests(unittest.TestCase):
    """分段恒速积分（N-way 均分）与多 rank 共享仲裁的方向断言。"""

    def test_timeline_piecewise_integration(self):
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"pool:0": 2.0, "port:0": 2.0}))
        timeline.add_flow("a", ("pool:0", "port:0"), 400, start_ns=0)
        timeline.add_flow("b", ("port:0",), 500, start_ns=200)
        timeline.add_flow("c", ("pool:0", "port:0"), 400, start_ns=200)
        timeline.run()
        # a: 0-200 独享（rate 2）；b/c: 200 起共享 port（各 1）→ c 于
        # 600 完成、b 剩 100 后独享 rate 2 → 650。
        self.assertEqual(timeline.completions, {
            "a": 200, "c": 600, "b": 650})

    def test_rd_direction_cases(self):
        """§5.7-3 方向用例（各 ≥1 例）：长历史短输入 / 带宽高低 / 竞争升降。"""
        # 长历史短输入：恢复字节大（r 大）而计算短（c 小）⇒ k 大。
        long_history = _uniform_case(16, 3000, c_ns=100).predict_k_hide()
        short_history = _uniform_case(16, 100, c_ns=100).predict_k_hide()
        self.assertGreater(long_history.k_hide, short_history.k_hide)
        # 带宽高 vs 低：同字节、同池/端口峰值同比缩放（瓶颈速率真实
        # 变化，不随字节归一）——峰值减半 ⇒ 恢复加倍 ⇒ k 变大。
        def bandwidth_case(pool_peak, port_peak):
            legs = tuple(
                RestoreGroupLeg(
                    j, j + 1, (2_000_000,), (("pool:0", "port:0"),))
                for j in range(16))
            segments = tuple(
                ComputeLayerSegment(ell, (1000.0,), (0,), ("port:0",))
                for ell in range(1, 17))
            snapshot = ResourceSnapshot(
                peak_bytes_per_ns={
                    "pool:0": pool_peak, "port:0": port_peak})
            return LayerRecursionPredictor(
                snapshot, legs, segments, 0).predict_k_hide()

        high_bw = bandwidth_case(2000.0, 2000.0)
        low_bw = bandwidth_case(1000.0, 1000.0)
        self.assertGreater(low_bw.k_hide, high_bw.k_hide)
        # 竞争上升：在册他流（未知 ETA）占池端口份额 ⇒ k 变大；竞争
        # 下降（无他流）⇒ k 回落。
        contended = _uniform_case(
            16, 1000, c_ns=1000,
            committed=(CommittedFlow("bg", ("pool:0",), None),),
        ).predict_k_hide()
        clean = _uniform_case(16, 1000, c_ns=1000).predict_k_hide()
        self.assertGreater(contended.k_hide, clean.k_hide)
        self.assertIn(STATUS_UNKNOWN_RELEASE_ETA, contended.statuses)


class EtaTruncationTests(unittest.TestCase):
    """F2：ETA 截断分支的分段恒速积分回归。

    已知 ETA 的在册流与活跃候选流共享资源、且 ETA 早于最早完成时刻
    时，step() 截断推进到 expiry 的同时必须按 elapsed×rate 扣减活跃流
    remaining——[now, expiry) 区间的服务量不得凭空消失（否则完成时刻
    系统性偏晚；步内守恒断言 _CONSERVATION_CHECK 亦在此拦截）。
    """

    def test_audit_scenario_completes_at_2_5ns(self):
        # 审计复现场景：1000 B/ns 资源、2000B 候选流、共享流 ETA=1ns。
        # [0,1) 两流均分（候选 500 B/ns，服务 500B）；[1,∞) 独享
        # 1000 B/ns（余 1500B）→ 物理完成 2.5ns。修复前截断分支只推时
        # 刻不扣字节 → 完成时刻 3ns（偏晚 20%）。
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0},
            committed=(CommittedFlow("bg", ("port:0",), 1),))
        timeline = UnifiedTimeline(snapshot)
        timeline.add_flow("cand", ("port:0",), 2000, start_ns=0)
        timeline.run()
        self.assertAlmostEqual(timeline.now_ns, 2.5, places=9)
        # 完成账本为 int ns（ceil 量化）：ceil(2.5) = 3。
        self.assertEqual(timeline.completions, {"cand": 3})

    def test_eta_equal_to_finish_takes_completion_branch(self):
        # 变体 1：ETA 恰等于（共享份额下的）最早完成时刻 → 不进截断分
        # 支（expiry < next_time 严格小于），正常完成分支按半速积分精
        # 确结算：500B @ 500 B/ns → 1.0ns 完成。
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0},
            committed=(CommittedFlow("bg", ("port:0",), 1),))
        timeline = UnifiedTimeline(snapshot)
        timeline.add_flow("cand", ("port:0",), 500, start_ns=0)
        timeline.run()
        self.assertAlmostEqual(timeline.now_ns, 1.0, places=9)
        self.assertEqual(timeline.completions, {"cand": 1})

    def test_tiny_eta_relative_to_volume(self):
        # 变体 2：ETA 极小（1ns）相对海量字节（2e9B @ 1e6 B/ns 独享）：
        # [0,1) 半速服务 5e5B，余 1_999_500_000B 独享 → 2000.5ns 完成
        # （修复前 2001ns）。
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1_000_000.0},
            committed=(CommittedFlow("bg", ("port:0",), 1),))
        timeline = UnifiedTimeline(snapshot)
        timeline.add_flow("cand", ("port:0",), 2_000_000_000, start_ns=0)
        timeline.run()
        self.assertAlmostEqual(timeline.now_ns, 2000.5, places=6)
        self.assertEqual(timeline.completions, {"cand": 2001})

    def test_mixed_multi_eta_multi_flow_piecewise(self):
        # 变体 3：多流混合 ETA——两条候选流（a 仅 port:0；b 跨 pool:0+
        # port:0）× 两个不同到期时刻的在册流（c1@port:0 ETA=1、
        # c2@pool:0 ETA=3）：
        #   [0,1)：port 三分（c1+a+b）→ 各 1000/3 B/ns；
        #   [1,3)：port 二分（a+b）各 500、pool 仍被 c2 占（b 瓶颈 500）；
        #   [3,∞)：各 500（b 的 pool 独享 1000 非瓶颈）；
        # 每流累计 1000/3 + 1000 + 2000/3 = 2000B → 同刻 13/3 ≈ 4.333ns
        # 完成（守恒：两流服务量之和恰为 4000B）。
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0, "pool:0": 1000.0},
            committed=(
                CommittedFlow("c1", ("port:0",), 1),
                CommittedFlow("c2", ("pool:0",), 3),
            ))
        timeline = UnifiedTimeline(snapshot)
        timeline.add_flow("a", ("port:0",), 2000, start_ns=0)
        timeline.add_flow("b", ("pool:0", "port:0"), 2000, start_ns=0)
        timeline.run()
        self.assertAlmostEqual(timeline.now_ns, 13.0 / 3.0, places=9)
        self.assertEqual(timeline.completions, {"a": 5, "b": 5})

    def test_conservation_toggle_does_not_change_numerics(self):
        # 开关仅控制守恒复核（A11' 后为 raise 级 ConservationViolation
        # Error，热路径可关）：关闭后同一场景数值行为不变。
        from joint import event_recursion_predictor as erp
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0},
            committed=(CommittedFlow("bg", ("port:0",), 1),))
        timeline = UnifiedTimeline(snapshot)
        timeline.add_flow("cand", ("port:0",), 2000, start_ns=0)
        original = erp._CONSERVATION_CHECK
        erp._CONSERVATION_CHECK = False
        try:
            timeline.run()
        finally:
            erp._CONSERVATION_CHECK = original
        self.assertAlmostEqual(timeline.now_ns, 2.5, places=9)
        self.assertEqual(timeline.completions, {"cand": 3})


class ConservationChannelAndActivationTests(unittest.TestCase):
    """A11'：守恒校验通道统一（raise 级 ConservationViolationError）+
    大时间基座量化容差 + pending 激活事件切割积分区间。"""

    def test_byte_theft_raises_conservation_violation(self):
        # 通道①（字节账目）隔离钉（A13'/H2 重写：原注释"pre=100、
        # post=100 流仍在册"与实际不符——step() 已把流结算移册，
        # post=0，真实形态是"减少量 100 ≠ served 50"）：注入
        # pre−post ≠ served 而 ②式联动量恰一致——删①不红即本钉失
        # 效。ConservationViolationError 为 EventRecursionError 子类
        # （face 调用方降级通道接得住——通道分裂消除；raise 级对
        # python -O 免疫）。
        self.assertTrue(issubclass(
            ConservationViolationError, EventRecursionError))
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0}))
        timeline.add_flow("f", ("port:0",), 1000, start_ns=0)
        timeline.step()  # 1000B@1000B/ns 于 t=1.0 完成移册：post=0
        # served=500 = pre−post ⇒ ②式 |500−1000×0.5|=0 恒过；①式
        # |1000−0−500|=500 ≫ 容差 ⇒ 只炸①（消息钉通道名）。
        timeline.now_ns = 0.5
        with self.assertRaisesRegex(
                ConservationViolationError, "byte conservation"):
            timeline._assert_conservation(
                1000.0, 500.0, {"f": 1000.0}, 0.0)

    def test_time_linkage_violation_raises(self):
        # 通道②（时刻-服务联动）隔离钉（A13'/H2 重写：原版 step() 已
        # 把 10000B 流结算移册、post=0，①式 |pre−0−0| 先炸——②从未
        # 被评估，删②测试仍绿的零钉住缺陷）：注入 served 与速率×时
        # 长失配而 ①式字节账恰平——删②不红即本钉失效。
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0}))
        timeline.add_flow("f", ("port:0",), 1000, start_ns=0)
        timeline.step()  # 完成移册：post=0、now=1.0
        # served=1000 = pre−post ⇒ ①式恒过；rewind now=0.5 ⇒ ②式
        # expected=1000×0.5=500 ≠ served=1000 ⇒ 只炸②（消息钉通道名）。
        timeline.now_ns = 0.5
        with self.assertRaisesRegex(
                ConservationViolationError, "time-service linkage"):
            timeline._assert_conservation(
                1000.0, 1000.0, {"f": 1000.0}, 0.0)

    def test_large_time_base_no_false_positive(self):
        # 大时间基座（now=1e12 ns、小 remaining、高速率）：finishes =
        # now + rem/rate 的 catastrophic cancellation 残差由 Σrate×
        # ulp(now) 量化松弛吸收（A11' 前此处实证误杀；完成时刻误差
        # ≤ rate×ulp(1e12) ≈ 0.12B 的时刻当量）。
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1000.0}))
        timeline.add_flow("cand", ("port:0",), 100, start_ns=0)
        timeline.now_ns = 1e12  # 活跃态大基座（pending start=0 ≤ now）
        timeline.run()
        self.assertLess(abs(timeline.now_ns - (1e12 + 0.1)), 1e-3)
        self.assertEqual(timeline.completions, {"cand": 1_000_000_000_001})

    def test_pending_activation_splits_interval(self):
        # A11' 激活切割：A(1000B@0) 与 B(1000B@100) 共享 1B/ns——
        # [0,100) A 独享服务 100B；[100,1900) 均分各 0.5（A 余 900、
        # B 余 900，同刻 1900 完成 A）；[1900,2000) B 独享余 100 →
        # 2000。修复前激活不切割：A=1000、B=2000（在册流"提前完成"
        # 的可行性乐观偏差，docstring"任何事件重算速率"承诺失实）。
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1.0}))
        timeline.add_flow("a", ("port:0",), 1000, start_ns=0)
        timeline.add_flow("b", ("port:0",), 1000, start_ns=100)
        timeline.run()
        self.assertEqual(timeline.completions, {"a": 1900, "b": 2000})


class InputHardeningTests(unittest.TestCase):
    """A14'（H4，2026-09-22 第三轮复审）：predictor 直连输入面加固
    ——η/γ 有限正验证（与 peak 同教义）、重复 id 检查补在册 active、
    start_ns 有限性；大基座激活切割回归钉（幸存流负尘埃零化后时刻恒
    不回退、无误炸 drained-below-zero）。"""

    def test_eta_gamma_must_be_finite_positive(self):
        for bad in (0.0, -1.0, float("inf"), float("nan")):
            with self.assertRaisesRegex(
                    EventRecursionError, "eta factor"):
                EfficiencyFactors(eta={"port:0": bad})
            with self.assertRaisesRegex(
                    EventRecursionError, "gamma factor"):
                EfficiencyFactors(gamma={"prefill": bad})
        # 合法值照常（缺省 1.0 冷启动不受影响）。
        self.assertEqual(
            EfficiencyFactors(eta={"port:0": 0.5}).eta_of("port:0"), 0.5)

    def test_duplicate_flow_id_in_active_rejected(self):
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1.0}))
        timeline.add_flow("a", ("port:0",), 100, start_ns=0)
        timeline.add_flow("b", ("port:0",), 500, start_ns=0)
        timeline.step()  # a 于 200ns 完成移册；b 幸存（在册 active）
        self.assertIn("a", timeline.completions)
        with self.assertRaisesRegex(
                EventRecursionError, "duplicate flow id 'b'"):
            timeline.add_flow("b", ("port:0",), 50, start_ns=300)

    def test_non_finite_start_ns_rejected(self):
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 1.0}))
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.assertRaisesRegex(
                    EventRecursionError, "start_ns must be finite"):
                timeline.add_flow("f", ("port:0",), 10, start_ns=bad)

    def test_large_base_activation_split_no_time_reversal(self):
        # 大基座（1e12）+ 区间中途激活回归钉：分段恒速金值（[t,t+5)
        # a 独享 7B/ns 服务 35B 余 298；此后均分 3.5 → a 于 t+90.142857…
        # 结清、b 余 479；[t+90.14…, t+158.571428…) b 独享 7B/ns），
        # ceil 入整型 ns 账本 ⇒ {t+91, t+159}；全程时刻单调（H4 尘埃
        # 零化后幸存流无 finishes<now 回退、无误炸 drained-below-zero）。
        base = 1_000_000_000_000
        timeline = UnifiedTimeline(ResourceSnapshot(
            peak_bytes_per_ns={"port:0": 7.0}, now_ns=base))
        timeline.add_flow("a", ("port:0",), 333, start_ns=base)
        timeline.add_flow("b", ("port:0",), 777, start_ns=base + 5)
        previous = float(timeline.now_ns)
        while timeline.has_work():
            self.assertTrue(timeline.step())
            self.assertGreaterEqual(timeline.now_ns, previous)
            previous = float(timeline.now_ns)
        self.assertEqual(
            timeline.completions, {"a": base + 91, "b": base + 159})


class FallbackFailClosedTests(unittest.TestCase):
    def test_misaligned_legs_fail_closed(self):
        legs = (
            RestoreGroupLeg(1, 2, (10,), (("pool:0", "port:0"),)),
            RestoreGroupLeg(0, 1, (10,), (("pool:0", "port:0"),)),
        )
        segments = (ComputeLayerSegment(1, (1.0,), (0,), ("port:0",)),)
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"pool:0": 1.0, "port:0": 1.0})
        with self.assertRaises(EventRecursionError):
            LayerRecursionPredictor(snapshot, legs, segments, 0)

    def test_unknown_resource_fails_closed(self):
        legs = (RestoreGroupLeg(0, 1, (10,), (("nope:0",)),),)
        segments = (ComputeLayerSegment(1, (1.0,), (0,), ("port:0",)),)
        snapshot = ResourceSnapshot(
            peak_bytes_per_ns={"pool:0": 1.0, "port:0": 1.0})
        with self.assertRaises(EventRecursionError):
            LayerRecursionPredictor(snapshot, legs, segments, 0)


if __name__ == "__main__":
    unittest.main()
