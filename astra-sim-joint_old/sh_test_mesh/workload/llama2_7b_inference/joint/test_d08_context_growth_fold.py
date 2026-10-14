#!/usr/bin/env python3
"""test_d08_context_growth_fold.py -- D08 语义的 W3 重锚（v4-projection
世代，2026-09-29）：decode 服务按上下文增长 = **逐步精确 Στ(z_k)**——
K=min(8,n̂) 区段中点折叠自正式路径退役（方案 §5.3 不再固定 8 段）；
``fold_repeat_service_ns`` 保留为登记的局部近似算子（sh30 N4 prefill
求值器消费，FoldRepeatServiceTest 继续钉其算术）。

条目（joint修改文档 D08）：
  - ``decode_ns_per_token × 步数`` 接口扩为按当前上下文求值的固定成本
    模板（``decode_task_load_ns_fn``）；剩余 m 步用 K=min(8,m) 区段中
    点折叠（式 16），不在估计器内循环 n̂ 次；
  - prefill 长 chunk 序列同规则（SH 注入半钉于 online/
    test_joint_decode_context.py）；
  - 字节总量与空间峰值仍按整数闭式（式 18b：nH_z + n(n+1)/2）。

验收钉（W3 重锚后）：
  1. 长/短上下文 decode 服务有差异（P 默认延续 + D 边界两入口）；
  2. 逐步精确：任意 n̂ 的服务总量 == Σ_k τ(z_k)（对任意单调 τ 精确，
     取代折叠的 K=8 求值预算——τ 求值次数 = n̂，线性披露，事件/预算
     上界由 PredictionParams.projection_event_budget 承担）；
  3. KV 读量守恒：折叠只改时间服务——远读字节（式 18a 基数×遍数）、
     增长/空间字节与 merge 计价逐位不变；累计逻辑上下文按式 18b 整数
     闭式披露（decode_ctx_weighted_tokens notes 位）。
  4. 标量回退零漂移：模板缺省（直接构造的测试夹具）恒 = n̂ × 率。
"""
import os
import sys
import unittest

_JOINT_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_JOINT_DIR)
for _p in (_JOINT_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    ACTION_REMOTE,
    ACTION_STAY,
    InstanceLoadView,
    JointHardwareRates,
    JointCostModel,
    LinkFlowRegistry,
    PDBoundaryView,
    RequestView,
    REPEAT_FOLD_MAX_SEGMENTS,
    ServiceFactors,
    SessionKVView,
    decode_context_weighted_tokens,
    fold_repeat_service_ns,
)

KV_TOKEN_BYTES = 128


def _rates():
    return JointHardwareRates.from_gbps(
        noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
        d2d_latency_ns=0, pool_latency_ns=0)


def _load(index, total=0, remaining=(10 ** 9, 10 ** 9)):
    return InstanceLoadView(
        instance_index=index, queued_task_load_ns=total,
        running_task_load_ns=0, active_decode_task_load_ns=0,
        hbm_remaining_bytes_by_tp_rank=remaining,
        reclaimable_bytes_by_tp_rank=remaining)


def _session(history_tokens=100, history_bytes=(1000, 1000),
             missing=(0, 0), location="local_hbm", prefix=4):
    return SessionKVView(
        session_id="s", home_instance=1, resident_instance=1,
        location=location, history_tokens=history_tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=history_bytes,
        missing_bytes_by_tp_rank=missing)


def _request(input_tokens=50, decode=10, input_bytes=(25, 25)):
    return RequestView(
        request_id="r", session_id="s", input_tokens=input_tokens,
        history_tokens_before=100, estimated_decode_tokens=decode,
        horizon_source="session_online_mean",
        input_kv_bytes_by_tp_rank=input_bytes)


def _route(source, target):
    return ((source, target), 1)


def _model(loads, *, tau=None, decode_rate=2.0):
    return JointCostModel(
        rates=_rates(), loads=loads, flow_registry=LinkFlowRegistry(),
        service_factors=ServiceFactors(),
        prefill_ns_per_token=1.0, decode_ns_per_token=decode_rate,
        model_layers=4, instance_tp_size=2, route_fn=_route,
        decode_task_load_ns_fn=tau)


def _tau_linear(scale=1.0):
    """单调线性模板 τ(z) = scale·z（上下文成比例——长短上下文差异的
    最小可析形态；中点折回对该形态精确）。"""
    def tau(context_tokens, *, hbm_share_divisor: int = 1):
        return scale * int(context_tokens)
    return tau


def _note_value(breakdown, key):
    prefix = key + "="
    for note in breakdown.notes:
        if note.startswith(prefix):
            return note[len(prefix):]
    raise AssertionError(f"note {key!r} missing in {breakdown.notes}")


def _complete_boundary(**overrides):
    values = dict(
        session_id="s", home_instance=0, p_instance_index=0,
        p_action="stay", history_tokens=100,
        working_bytes_by_tp_rank=(KV_TOKEN_BYTES * 100,) * 2,
        base_bytes_by_tp_rank=(0, 0), decode_steps=8,
        base_tokens=100, kv_token_bytes=KV_TOKEN_BYTES)
    values.update(overrides)
    return PDBoundaryView(**values)


# ==================================================== 式(16) 折叠原语 ==

class FoldRepeatServiceTest(unittest.TestCase):
    """fold_repeat_service_ns：区段端点、精确性、调用界。"""

    def test_zero_steps_returns_zero(self):
        self.assertEqual(fold_repeat_service_ns(
            steps=0, first_context_tokens=100, context_stride_tokens=1,
            service_fn=lambda z: z), 0.0)

    def test_negative_steps_fail_closed(self):
        with self.assertRaises(Exception):
            fold_repeat_service_ns(
                steps=-1, first_context_tokens=100,
                context_stride_tokens=1, service_fn=lambda z: z)
        with self.assertRaises(Exception):
            fold_repeat_service_ns(
                steps=10, first_context_tokens=100,
                context_stride_tokens=0, service_fn=lambda z: z)

    def test_small_m_evaluates_each_step_exactly(self):
        """m ≤ 8：区段宽 1、中点整——逐次求值 == 直逐步和（对该服务
        模型精确）。"""
        for steps in (1, 2, 7, 8):
            folded = fold_repeat_service_ns(
                steps=steps, first_context_tokens=100,
                context_stride_tokens=1, service_fn=float)
            expected = sum(100 + k - 1 for k in range(1, steps + 1))
            self.assertEqual(folded, float(expected), f"steps={steps}")

    def test_segment_partition_conservation(self):
        """区段端点公式（l_j/r_j/w_j）：Σw_j = m 恒等守恒、相邻区段
        连续（r_j + 1 = l_{j+1}）、K = min(8, m)。"""
        for steps in range(1, 45):
            count = min(REPEAT_FOLD_MAX_SEGMENTS, steps)
            covered = 0
            for j in range(1, count + 1):
                left = ((j - 1) * steps) // count + 1
                right = (j * steps) // count
                self.assertGreaterEqual(right, left)
                self.assertEqual(left, covered + 1)
                covered = right
            self.assertEqual(covered, steps)

    def test_linear_service_fold_exact_for_any_m(self):
        """仿射 τ：中点规则精确——任意 m 折叠 == 逐步和（中点插值对
        线性形态无近似损失）。"""
        for steps in (9, 100, 1000):
            folded = fold_repeat_service_ns(
                steps=steps, first_context_tokens=150,
                context_stride_tokens=1, service_fn=_tau_linear(1.0))
            expected = sum(150 + k - 1 for k in range(1, steps + 1))
            self.assertAlmostEqual(folded, float(expected), places=6)

    def test_quadratic_fold_within_eq17_bounds(self):
        """单调 τ（二次）：折叠落在式(17) 单调诊断区间
        Σw_j τ(l_j) ≤ 真值 ≤ Σw_j τ(r_j) 内（区间不剪枝、仅诊断）。"""
        steps = 100
        count = min(REPEAT_FOLD_MAX_SEGMENTS, steps)
        lower = upper = 0
        for j in range(1, count + 1):
            left = ((j - 1) * steps) // count + 1
            right = (j * steps) // count
            weight = right - left + 1
            lower += weight * (100 + left - 1) ** 2
            upper += weight * (100 + right - 1) ** 2
        exact = sum((100 + k - 1) ** 2 for k in range(1, steps + 1))
        folded = fold_repeat_service_ns(
            steps=steps, first_context_tokens=100,
            context_stride_tokens=1, service_fn=lambda z: float(z) ** 2)
        self.assertLessEqual(lower, exact)
        self.assertLessEqual(exact, upper)
        self.assertLessEqual(lower, folded)
        self.assertLessEqual(folded, upper)

    def test_call_count_bounded_for_thousands_of_steps(self):
        """验收红线：n̂ 增大数千倍，τ 求值次数仍受 K=8 界（≤ 2K）——
        不在估计器内循环 n̂ 次。"""
        def counted(steps):
            calls = []
            folded = fold_repeat_service_ns(
                steps=steps, first_context_tokens=100,
                context_stride_tokens=1,
                service_fn=lambda z: calls.append(z) or float(z))
            return folded, len(calls)
        _, small = counted(8)
        _, mid = counted(8_000)
        _, huge = counted(8_000_000)
        self.assertLessEqual(small, REPEAT_FOLD_MAX_SEGMENTS)
        self.assertLessEqual(mid, 2 * REPEAT_FOLD_MAX_SEGMENTS)
        self.assertLessEqual(huge, 2 * REPEAT_FOLD_MAX_SEGMENTS)


class ContextWeightedTokensTest(unittest.TestCase):
    """式(18b) 整数闭式：Σ_{k=1}^n (H_z+k) = nH_z + n(n+1)/2。"""

    def test_identity_integer_closed_form(self):
        for base, steps in ((0, 5), (1, 1), (7, 8), (150, 10),
                            (100_000, 8_000)):
            expected = steps * base + steps * (steps + 1) // 2
            self.assertEqual(
                decode_context_weighted_tokens(base, steps), expected)
            self.assertIsInstance(
                decode_context_weighted_tokens(base, steps), int)


# ==================================================== P 默认延续入口 ==

class PEntryContextGrowthTest(unittest.TestCase):
    """P 默认延续（H_z = H+U × n̂_D）的 D08 语义。"""

    def _stay_candidate(self, model, history_tokens=100):
        return model.estimate_P_with_continuation(
            session=_session(history_tokens=history_tokens),
            request=_request(input_tokens=50, decode=10),
            instance_index=1, action=ACTION_STAY, remote_enabled=True)

    def test_scalar_fallback_zero_drift(self):
        """标量回退（模板缺省）：decode 服务恒 = n̂ × 率（零漂移锚）；
        notes 披露 fold 形态（share=1 独占）。"""
        model = _model({0: _load(0), 1: _load(1)})
        candidate = self._stay_candidate(model)
        self.assertTrue(candidate.applicable)
        # prefill = 50×1.0；decode = 10×2.0 ⇒ compute = 70。
        self.assertEqual(candidate.breakdown.compute_ns, 50 + 20)
        exact_note = _note_value(candidate.breakdown, "decode_ctx_exact")
        self.assertEqual(exact_note, "base=150,steps=10,ns=20")
        self.assertNotIn(
            "decode_ctx_weighted_tokens",
            candidate.breakdown.notes)  # 披露位仅模板在场

    def test_long_short_context_decode_service_differs(self):
        """验收 1：长/短上下文 decode 服务有差异——同候选只改历史
        （100 → 100_000），decode_ctx_fold ns 单调增长。"""
        short = self._stay_candidate(
            _model({0: _load(0), 1: _load(1)}, tau=_tau_linear(1.0)),
            history_tokens=100)
        long_ctx = self._stay_candidate(
            _model({0: _load(0), 1: _load(1)}, tau=_tau_linear(1.0)),
            history_tokens=100_000)
        short_ns = int(_note_value(short.breakdown, "decode_ctx_exact"
                                   ).rsplit(",ns=")[-1])
        long_ns = int(_note_value(long_ctx.breakdown, "decode_ctx_exact"
                                  ).rsplit(",ns=")[-1])
        self.assertGreater(long_ns, short_ns)
        self.assertGreater(long_ctx.breakdown.compute_ns,
                           short.breakdown.compute_ns)

    def test_call_count_bounded_when_steps_grow_thousands_fold(self):
        """验收 2（W3 重锚）：逐步精确——任意 n̂ 的服务总量 ==
        Σ_k τ(z_k)（对任意单调 τ 精确，取代折叠近似）；τ 求值次数
        = n̂（线性，披露于 decode_ctx_exact steps 位；复杂度上界由
        ``PredictionParams.projection_event_budget`` 承担，超限
        fail-closed 不返回截断时刻）。"""
        def counted(steps):
            calls = []

            def tau(context_tokens, *, hbm_share_divisor: int = 1):
                calls.append((context_tokens, hbm_share_divisor))
                return 1.0 * int(context_tokens)

            model = _model({0: _load(0), 1: _load(1)}, tau=tau)
            candidate = model.estimate_P_with_continuation(
                session=_session(history_tokens=100),
                request=_request(input_tokens=50, decode=steps),
                instance_index=1, action=ACTION_STAY,
                remote_enabled=True)
            return candidate, len(calls)

        candidate_small, calls_small = counted(8)
        candidate_mid, calls_mid = counted(8_000)
        # 逐步求值：恰 n̂ 次、上下文 z_k = base + k − 1（式 18b 口径）。
        self.assertEqual(calls_small, 8)
        self.assertEqual(calls_mid, 8_000)
        # 精确性（折叠近似退役）：总量 == 逐步和（任意 τ 形态）。
        base = 100 + 50
        exact_mid = sum(1.0 * (base + k - 1) for k in range(1, 8_001))
        self.assertEqual(
            int(_note_value(candidate_mid.breakdown,
                            "decode_ctx_exact").rsplit(",ns=")[-1]),
            int(exact_mid))
        exact_small = sum(1.0 * (base + k - 1) for k in range(1, 9))
        self.assertEqual(
            int(_note_value(candidate_small.breakdown,
                            "decode_ctx_exact").rsplit(",ns=")[-1]),
            int(exact_small))

    def test_ctx_weighted_tokens_disclosed_integer_closed_form(self):
        """式(18b) 披露位：base=150、n̂=10 ⇒ 10·150 + 55 = 1555（整数
        闭式，字节/累计上下文核账锚）。"""
        model = _model({0: _load(0), 1: _load(1)}, tau=_tau_linear(1.0))
        candidate = self._stay_candidate(model)
        self.assertEqual(
            int(_note_value(candidate.breakdown,
                            "decode_ctx_weighted_tokens")),
            10 * 150 + 10 * 11 // 2)

    def test_remote_bytes_conserved_across_fold(self):
        """验收 3（KV 读量守恒）：模板在场与否只改 decode 时间服务——
        remote 候选的远读字节账（式 18a 基数×遍数 ⇒ remote_read_ns）、
        空间/驱逐账与 merge 计价逐位不变。（remote 适用面：基础历史
        驻留他实例——resident=1、目标 0。）"""
        session = _session(
            history_tokens=100, history_bytes=(1000, 1000),
            missing=(1000, 1000), location="partial_hbm_remote", prefix=2)
        loads = {0: _load(0), 1: _load(1)}
        scalar = _model(loads).estimate_P_with_continuation(
            session=session, request=_request(input_tokens=50, decode=10),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        folded = _model(loads, tau=_tau_linear(1.0)
                        ).estimate_P_with_continuation(
            session=session, request=_request(input_tokens=50, decode=10),
            instance_index=0, action=ACTION_REMOTE, remote_enabled=True)
        self.assertEqual(
            scalar.breakdown.remote_read_ns,
            folded.breakdown.remote_read_ns)
        self.assertEqual(
            scalar.breakdown.eviction_wait_ns,
            folded.breakdown.eviction_wait_ns)
        self.assertEqual(scalar.breakdown.merge_ns, folded.breakdown.merge_ns)
        self.assertEqual(
            scalar.breakdown.target_wait_ns,
            folded.breakdown.target_wait_ns)
        # 时间服务面确实改变（否则本对照无信息量）。
        self.assertNotEqual(
            scalar.breakdown.compute_ns, folded.breakdown.compute_ns)


# ====================================================== D 边界入口 ==

class DBoundaryContextGrowthTest(unittest.TestCase):
    """D 边界（H_B × n̂_B）的 D08 语义。"""

    def _boundary_candidate(self, model, boundary):
        return model.estimate_D_remaining(
            boundary=boundary, instance_index=0, action=ACTION_STAY,
            remote_enabled=True)

    def test_scalar_fallback_zero_drift(self):
        """标量回退：decode 服务 = n̂_B × 率（8 × 2 = 16——与
        test_d03 的 DECODE_COMPUTE_NS 钉同值）。"""
        model = _model({0: _load(0), 1: _load(1)})
        candidate = self._boundary_candidate(model, _complete_boundary())
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.compute_ns, 16)
        self.assertEqual(
            _note_value(candidate.breakdown, "decode_ctx_exact"),
            "base=100,steps=8,ns=16")

    def test_long_short_boundary_context_differs(self):
        """验收 1（D 半）：同边界只改 H_B（100 → 100_000），decode
        服务随上下文增长。"""
        short = self._boundary_candidate(
            _model({0: _load(0), 1: _load(1)}, tau=_tau_linear(1.0)),
            _complete_boundary(history_tokens=100))
        long_ctx = self._boundary_candidate(
            _model({0: _load(0), 1: _load(1)}, tau=_tau_linear(1.0)),
            _complete_boundary(history_tokens=100_000))
        self.assertGreater(
            int(_note_value(long_ctx.breakdown, "decode_ctx_exact"
                            ).rsplit(",ns=")[-1]),
            int(_note_value(short.breakdown, "decode_ctx_exact"
                            ).rsplit(",ns=")[-1]))

    def test_call_count_bounded_when_steps_grow_thousands_fold(self):
        """验收 2（D 半，W3 重锚）：逐步精确——n̂_B 8 → 8000 服务总量
        == Σ_k τ(z_k)（任意单调 τ 精确）；τ 求值次数 = n̂（+1 为
        predicted_next_token_done 的单步服务，常数开销）；n̂_B ≥ 1
        下界（式 4a）保持。"""
        def counted(steps):
            calls = []

            def tau(context_tokens, *, hbm_share_divisor: int = 1):
                calls.append(context_tokens)
                return 1.0 * int(context_tokens)

            model = _model({0: _load(0), 1: _load(1)}, tau=tau)
            candidate = self._boundary_candidate(
                model, _complete_boundary(decode_steps=steps))
            return candidate, len(calls)
        candidate_small, calls_small = counted(8)
        candidate_mid, calls_mid = counted(8_000)
        self.assertEqual(calls_small, 8 + 1)
        self.assertEqual(calls_mid, 8_000 + 1)
        self.assertEqual(
            int(_note_value(candidate_mid.breakdown,
                            "decode_ctx_exact").rsplit(",ns=")[-1]),
            sum(1.0 * (100 + k - 1) for k in range(1, 8_001)))
        self.assertEqual(
            int(_note_value(candidate_small.breakdown,
                            "decode_ctx_exact").rsplit(",ns=")[-1]),
            sum(1.0 * (100 + k - 1) for k in range(1, 9)))

    def test_boundary_bytes_conserved_across_fold(self):
        """验收 3（D 半）：折叠不改字节账——增长字节（kv_token_bytes ×
        n̂_B）与驱逐/空间账逐位不变（模板 vs 标量对照）。"""
        loads = {0: _load(0), 1: _load(1)}
        boundary = _complete_boundary(decode_steps=8)
        scalar = self._boundary_candidate(_model(loads), boundary)
        folded = self._boundary_candidate(
            _model(loads, tau=_tau_linear(1.0)), boundary)
        self.assertEqual(
            scalar.breakdown.eviction_wait_ns,
            folded.breakdown.eviction_wait_ns)
        self.assertEqual(scalar.breakdown.merge_ns, folded.breakdown.merge_ns)
        self.assertNotEqual(
            scalar.breakdown.compute_ns, folded.breakdown.compute_ns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
