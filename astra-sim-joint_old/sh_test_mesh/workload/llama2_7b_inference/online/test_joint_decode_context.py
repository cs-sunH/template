#!/usr/bin/env python3
"""test_joint_decode_context.py -- D08（v2.0 §13.2 式(16)/(18b)）decode
计价上下文增长 + 固定 8 区段折叠钉子（SH 注入半）。

背景（2026-09-22，joint 改造 C3 → 2026-09-27 D08 改宗）：SH
`_joint_cost_model` 的 decode 计价旧口径（C3 之前 context=1/长度 1）经
C3 换为"真实上下文 × horizon 逐步求和折回 per-token 均值"——该机制
把 O(n̂) 循环挂在构造处、且 per-token 均值无法携带候选阶段 HBM 份额。
D08（v2.0 §13.2 式 16）按设计裁定**直接替换**：

  - SH 构造注入**一次前向服务模板** ``decode_task_load_ns_fn(z,
    hbm_share_divisor)``（z = 该前向 KV 可见上下文；式(6) 阶段份额分
    母由 JCM 按候选阶段传入）；
  - P 默认延续（H_z = H+U × n̂_D）与 D 边界（H_B × n̂_B）的上下文/
    时域由 JCM 两入口现场取（式 18b 的 H_z 口径），构造期不再传参；
  - 剩余 n̂ 步总量 = JCM 端 K=min(8,n̂) 区段中点折叠
    （``fold_repeat_service_ns``，式 16）——不逐 n̂ 循环，τ 求值次数
    ≤ 2K 与 n̂ 无关；
  - ``decode_ns_per_token`` 标量位 = context=1 单步名义率（模板缺省
    的夹具/离线回退锚，模板在场时不消费——N4 prefill 名义率同族）；
  - KV 读量守恒不经折叠：远读 = 式(18a) 基数×遍数、累计逻辑上下文 =
    式(18b) ``decode_context_weighted_tokens`` 整数闭式。

旧 C3 钉子的语义映射（同名意图、新机制载体）：
  1. 长短上下文计价差异 → τ(z) 单调（本文件 + joint/
     test_d08_context_growth_fold.py 的候选级钉）；
  2. "均值 × horizon == 逐步总量" → m ≤ 8 折叠 == 逐步总量（精确钉）
     ；m > 8 折叠落在式(17) 单调诊断区间内；
  3. 冷启动 32 前向步（D14 式(4) 冻结先验）→ n̂ = 32 折叠的界与区间；
  4. "构造处一次求值 O(1)" → "每候选 τ 求值受 K=8 界 + memo 条目有
     界"（不随 n̂ 增长）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_decode_context.py   （或 pytest 同路径）
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
    estimate_decode_remaining_task_load_ns,
    estimate_iteration_time_ns,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    CausalHorizonEstimator,
    JointHardwareRates,
    LinkFlowRegistry,
    REPEAT_FOLD_MAX_SEGMENTS,
    ServiceFactors,
    decode_context_weighted_tokens,
    fold_repeat_service_ns,
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineInstanceState,
    _TASK_LOAD_CACHE_CAPACITY,  # F6 销账：替身对齐 __init__ 初值
)


def _make_scheduler() -> Sh30OnlineScheduler:
    """绕过 __init__（需 manifest/graph/bridge），只装配 _joint_cost_model
    用到的属性（小参数 Roofline 配置与 test_sh30_task_load_snapshot.py /
    test_admit_gate.py 同款）。"""
    hardware = FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=1_000_000_000,
        local_hbm_bandwidth_gbps=100.0,
        d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    model = FaceModel(
        layers=2,
        hidden_size=16,
        ffn_size=32,
        num_heads=4,
        vocab_size=32,
        bytes_per_elem=2,
        mlp_variant="swiglu",
    )
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    scheduler.instances = [
        _OnlineInstanceState(index=i)
        for i in range(len(scheduler.topology.instances))
    ]
    scheduler.hardware = hardware
    scheduler.model = model
    scheduler.kv_manager = KVCacheManager(scheduler.topology, model)
    # 生产同款冷启动缺省（SH __init__：cold_start_default_forward_steps
    # =32，D14 v2.0 式(4) 冻结先验）。
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_forward_steps=32)
    scheduler._joint_flows = LinkFlowRegistry()
    scheduler._joint_factors = ServiceFactors()
    scheduler._joint_rates = JointHardwareRates.from_gbps(
        noc_link_gbps=hardware.d2d_bandwidth_gbps,
        pool_port_gbps=50.0,
        local_hbm_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_latency_ns=int(hardware.d2d_latency_ns),
        pool_latency_ns=0,
    )
    # _pool_port_divisor 只在 _instance_edge_ports 命中时读 _pool_ports；
    # 空映射 → 恒 1（单流全带宽，单测口径）。D04：JCM 构造注入
    # pool_edge_flows_fn=_pool_ports.count——真实空注册表（冷启动恒
    # 0 在途，与 __init__ 同款）。
    from online.sh30_online_scheduler import _PoolPortRegistry
    scheduler._pool_ports = _PoolPortRegistry()
    scheduler._instance_edge_ports = {}
    scheduler.joint_config = parse_joint_config(env={})
    scheduler._prefill_task_cache = {}
    scheduler._decode_task_load_cache = {}
    # 对齐 __init__ 初值（F6 销账：软门已删，替身漏设 = AttributeError）。
    scheduler._task_load_cache_capacity = _TASK_LOAD_CACHE_CAPACITY
    scheduler._quota_tracker = None  # off 档（parse_joint_config 缺省）
    scheduler._link_telemetry_rates = {}
    # M1：C8 遥测状态 __init__ 同款（时间加权活跃流数）。
    scheduler._link_telemetry_flow_counts = {}  # C8 遥测状态 __init__ 同款（F6）
    # ACC（验收纪律第 3 类，v2.0 §19.3）：预测器开销账本 __init__ 同款
    # 初值（F6 销账口径：决策/摘要记账直访——替身漏设 = AttributeError）。
    from joint.predictor_cost_accounting import PredictorCostAccounting
    scheduler.predictor_cost = PredictorCostAccounting()
    scheduler.p_chunk = 512  # N4：_joint_prefill_total_load_ns 同形切分（F6 替身补设）
    # 对齐 __init__ 初值（F6 销账：JCM 构造直达传 _hbm_ports——
    # 替身漏设 = AttributeError）。
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._snapshot_verify = False
    return scheduler


def _stepped_total(scheduler, context: int, steps: int) -> int:
    """逐步剩余总量真值（estimate_decode_remaining_task_load_ns，
    average = steps / generated = 0——与旧 C3 语义钉同一直调口径）。"""
    return estimate_decode_remaining_task_load_ns(
        scheduler.hardware, scheduler.model,
        instance_size=scheduler.topology.instances[0].size,
        current_context_tokens=context, generated_tokens=0,
        average_decode_length=float(steps),
        running_step_fraction_remaining=1.0)


def _segments(steps: int):
    """式(16) 区段端点（l_j, r_j）——与 fold_repeat_service_ns 同式。"""
    count = min(REPEAT_FOLD_MAX_SEGMENTS, steps)
    return [
        (((j - 1) * steps) // count + 1, (j * steps) // count)
        for j in range(1, count + 1)
    ]


class JointDecodeContextTest(unittest.TestCase):
    """D08：decode 一次前向服务模板（上下文 z）+ K=min(8,n̂) 折叠。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()
        self.instance_size = self.scheduler.topology.instances[0].size

    def _cost_model(self):
        return self.scheduler._joint_cost_model(0)

    def test_template_injected_and_one_forward_context_sensitive(self):
        """模板注入钉 + 单前向上下文敏感（长短上下文差异的最小单元）：
        τ(99000) > τ(1) > 0，且 τ 与单步 roofline 直调逐位同值
        （d_batch=1、d_token=z；式(6) 份额缺省 1 = 独占）。"""
        model = self._cost_model()
        self.assertTrue(callable(model.decode_task_load_ns_fn))
        short = model.decode_task_load_ns_fn(1)
        long_at = model.decode_task_load_ns_fn(99_000)
        self.assertGreater(short, 0)
        self.assertGreater(long_at, short)
        for context in (1, 2048, 99_000):
            self.assertEqual(
                model.decode_task_load_ns_fn(context),
                self.scheduler._decode_step_load_ns(
                    self.instance_size, context))
            self.assertEqual(
                model.decode_task_load_ns_fn(context),
                estimate_iteration_time_ns(
                    self.scheduler.hardware, self.scheduler.model,
                    instance_size=self.instance_size,
                    p_chunk=0, d_batch=1, d_token=context))

    def test_fold_matches_stepped_total_for_small_horizon(self):
        """语义钉（旧 C3 "均值 × horizon == 逐步总量"的新载体）：m ≤ 8
        时区段宽 1、中点整——折叠 == 直调逐步剩余总量（对该服务模型
        精确，v2.0 §13.2）。"""
        model = self._cost_model()
        context = 2048
        for steps in (1, 5, 8):
            folded = fold_repeat_service_ns(
                steps=steps, first_context_tokens=context,
                context_stride_tokens=1,
                service_fn=model.decode_task_load_ns_fn)
            self.assertEqual(int(folded), _stepped_total(
                self.scheduler, context, steps))

    def test_fold_within_monotone_diagnostic_bounds_cold_start(self):
        """冷启动 32 前向步（D14 冻结先验，不隐藏不放大）进折叠：
        n̂ = 32 > K ⇒ 折叠为近似，须落在式(17) 单调诊断区间
        Σw_j τ(l_j) ≤ 逐步总量 ≤ Σw_j τ(r_j) 内（端点同式求值）。"""
        cold_tokens, source = self.scheduler._joint_horizon.estimate(
            "cold_session")
        self.assertEqual((cold_tokens, source), (32, "cold_start_default"))
        model = self._cost_model()
        context = 4096
        tau = model.decode_task_load_ns_fn
        stepped = _stepped_total(self.scheduler, context, cold_tokens)
        lower = upper = 0
        for left, right in _segments(cold_tokens):
            weight = right - left + 1
            lower += weight * tau(context + left - 1)
            upper += weight * tau(context + right - 1)
        folded = fold_repeat_service_ns(
            steps=cold_tokens, first_context_tokens=context,
            context_stride_tokens=1, service_fn=tau)
        self.assertLessEqual(lower, stepped)
        self.assertLessEqual(stepped, upper)
        self.assertLessEqual(lower, folded)
        self.assertLessEqual(folded, upper)

    def test_call_count_bounded_under_k8_for_thousands_of_steps(self):
        """验收红线：n̂ 增大数千倍，τ 求值次数仍受 K=8 界（≤ 2K，
        与 n̂ 无关——不在估计器内循环 n̂ 次）。"""
        model = self._cost_model()

        def _counted(steps: int):
            calls = []

            def tau(context_tokens):
                calls.append(context_tokens)
                return model.decode_task_load_ns_fn(context_tokens)

            folded = fold_repeat_service_ns(
                steps=steps, first_context_tokens=100_000,
                context_stride_tokens=1, service_fn=tau)
            return folded, len(calls)

        base_folded, base_calls = _counted(8)
        self.assertLessEqual(base_calls, REPEAT_FOLD_MAX_SEGMENTS)
        _, mid_calls = _counted(8_000)
        _, huge_calls = _counted(8_000_000)
        self.assertEqual(mid_calls, huge_calls)
        self.assertLessEqual(mid_calls, 2 * REPEAT_FOLD_MAX_SEGMENTS)
        # K=min(8, n̂)：n̂ ≤ 8 时逐次求值（对该服务模型精确）。
        self.assertEqual(base_calls, 8)

    def test_memo_entries_bounded_under_long_horizon(self):
        """旧 "构造处一次求值" O(1) 钉的新载体：memo 条目有界——8000
        步折叠经生产（memo 化）模板只新增 ≤ 2K 条单步记录，与 n̂ 无
        关；重复折叠零新增（全命中）。"""
        model = self._cost_model()
        # 名义标量位求值（context=1）占用的一条 memo 记录清出后计折叠。
        self.scheduler._decode_task_load_cache.clear()
        fold_repeat_service_ns(
            steps=8_000, first_context_tokens=100_000,
            context_stride_tokens=1,
            service_fn=model.decode_task_load_ns_fn)
        self.assertLessEqual(
            len(self.scheduler._decode_task_load_cache),
            2 * REPEAT_FOLD_MAX_SEGMENTS)
        before = len(self.scheduler._decode_task_load_cache)
        fold_repeat_service_ns(
            steps=8_000, first_context_tokens=100_000,
            context_stride_tokens=1,
            service_fn=model.decode_task_load_ns_fn)
        self.assertEqual(
            len(self.scheduler._decode_task_load_cache), before)

    def test_nominal_scalar_rate_positive_and_single_step_anchored(self):
        """标量位 = context=1 单步名义率（模板缺省回退锚，模板在场时
        不消费）：恒正、与单步口径同值。"""
        model = self._cost_model()
        self.assertGreater(model.decode_ns_per_token, 0.0)
        self.assertEqual(
            model.decode_ns_per_token,
            float(self.scheduler._decode_step_load_ns(
                self.instance_size, 1)))


class PrefillChunkFoldTest(unittest.TestCase):
    """D08"prefill 长 chunk 序列同规则"（SH 注入函数半）：首 chunk 逐值
    + 整 chunk K=min(8,m) 折叠 + 尾 chunk 单独计。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()
        self.instance_size = self.scheduler.topology.instances[0].size
        self.scheduler._prefill_task_cache.clear()

    def _exact_chunk_sum(self, input_tokens: int, history: int) -> int:
        """逐 chunk 精确和（旧实现字面口径，对照锚；执行侧列车台账
        仍按此精确核账——两账分离）。"""
        total = 0
        completed = 0
        while completed < input_tokens:
            chunk_tokens = min(self.scheduler.p_chunk,
                               input_tokens - completed)
            total += self.scheduler._prefill_chunk_task_load_ns(
                instance_size=self.instance_size,
                chunk_tokens=chunk_tokens,
                context_tokens=history + completed + chunk_tokens)
            completed += chunk_tokens
        return total

    def test_single_partial_chunk_exact(self):
        """U < q_c：单一尾 chunk 逐值（首 chunk 即尾——恢复启动分开）。"""
        self.assertEqual(
            self.scheduler._joint_prefill_total_load_ns(
                100, 7000),
            self.scheduler._prefill_chunk_task_load_ns(
                instance_size=self.instance_size, chunk_tokens=100,
                context_tokens=7100))

    def test_small_chunk_count_matches_exact_sum(self):
        """m ≤ 8：逐 chunk 求值 == 逐 chunk 精确和（9 个整 chunk =
        首 1 + 折叠 8）。"""
        self.assertEqual(
            self.scheduler._joint_prefill_total_load_ns(
                9 * self.scheduler.p_chunk, 512),
            self._exact_chunk_sum(9 * self.scheduler.p_chunk, 512))

    def test_call_count_bounded_under_k8_for_long_chunk_sequences(self):
        """验收红线（prefill 半）：整 chunk 数增大数千倍，τ（逐 chunk
        roofline）求值次数仍受 K=8 界 + 首/尾两笔。"""
        chunk = self.scheduler.p_chunk
        calls = []

        original = self.scheduler._prefill_chunk_task_load_ns

        def counting(**kwargs):
            calls.append(kwargs["context_tokens"])
            return original(**kwargs)

        self.scheduler._prefill_chunk_task_load_ns = counting
        try:
            self.scheduler._joint_prefill_total_load_ns(
                8_001 * chunk, 512)
        finally:
            self.scheduler._prefill_chunk_task_load_ns = original
        self.assertLessEqual(
            len(calls), 2 * REPEAT_FOLD_MAX_SEGMENTS + 2)

    def test_tail_chunk_priced_at_terminal_context(self):
        """尾 chunk 单独计（不等长不折）：U = 2·q_c + 300 的结果 =
        首 chunk + 折叠(1) + 尾 chunk(H+U) 三笔之和（m = 1 精确）。"""
        chunk = self.scheduler.p_chunk
        input_tokens = 2 * chunk + 300
        history = 512
        expected = (
            self.scheduler._prefill_chunk_task_load_ns(
                instance_size=self.instance_size, chunk_tokens=chunk,
                context_tokens=history + chunk)
            + self.scheduler._prefill_chunk_task_load_ns(
                instance_size=self.instance_size, chunk_tokens=chunk,
                context_tokens=history + 2 * chunk)
            + self.scheduler._prefill_chunk_task_load_ns(
                instance_size=self.instance_size, chunk_tokens=300,
                context_tokens=history + input_tokens))
        self.assertEqual(
            self.scheduler._joint_prefill_total_load_ns(
                input_tokens, history),
            expected)


class ContextWeightedTokensTest(unittest.TestCase):
    """式(18b) 整数闭式（KV 读量/累计上下文核账锚——不经式(16) 近似）。"""

    def test_identity_matches_closed_form(self):
        for base, steps in ((0, 5), (1, 1), (7, 8), (150, 10), (99_000, 32)):
            self.assertEqual(
                decode_context_weighted_tokens(base, steps),
                steps * base + steps * (steps + 1) // 2)
            self.assertEqual(
                decode_context_weighted_tokens(base, steps),
                sum(base + k for k in range(1, steps + 1)))

    def test_zero_steps_is_zero(self):
        self.assertEqual(decode_context_weighted_tokens(1234, 0), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
