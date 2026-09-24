#!/usr/bin/env python3
"""test_joint_decode_context.py -- C3（WP1c）decode 计价上下文依赖钉子。

背景（2026-09-22，joint 改造 C3）：SH `_joint_cost_model` 的 decode
per-token 率旧口径在 context=1 / average_decode_length=1.0 求值（单步
零上下文），长历史的 attention 二次项不进计价。C3 换口径：

  - ``decode_context_tokens = history + input``（准入时已知、对全部候选
    同值，非 oracle——设计文档 §3.2"decode 计算：使用已知上下文和合法
    预测时域"）；
  - ``average_decode_length = CausalHorizonEstimator`` 现值（冷启动 1
    如实保留，不隐藏）；
  - attention 二次项由 estimate_decode_remaining_task_load_ns 的
    d_token 逐步（face_scheduler.py）自然进入；
  - 剩余总量按 horizon 折回 per-token 均值——JCM 消费端
    ``estimated_decode_tokens × decode_ns_per_token`` 形态不动（同值
    horizon 乘回 ≡ 逐步剩余总量）。

钉子：
  1. 方向断言：100k 长历史夹具的 decode 计价 > context=1 旧口径；
  2. 语义钉：per-token 均值 × horizon == 直调逐步剩余总量（幂次 2 的
     horizon 下浮点除法精确，可做恒等断言）；
  3. 冷启动路径回归：无观测样本时 estimator 返回 (1, cold_start_default)
     并原样进求值（horizon=1 单步、不隐藏不放大）；
  4. O(1) 钉：上下文对全部候选同值 → 构造处一次求值（memo 恰一条）。

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
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    CausalHorizonEstimator,
    JointHardwareRates,
    LinkFlowRegistry,
    ServiceFactors,
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
    # 生产同款冷启动缺省（SH __init__：cold_start_default_tokens=1）。
    scheduler._joint_horizon = CausalHorizonEstimator(
        cold_start_default_tokens=1)
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
    # 空映射 → 恒 1（单流全带宽，单测口径）。
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
    scheduler.p_chunk = 512  # N4：_joint_prefill_total_load_ns 同形切分（F6 替身补设）
    # 对齐 __init__ 初值（F6 销账：JCM 构造直达传 _hbm_ports——
    # 替身漏设 = AttributeError）。
    from joint.hbm_port_flow_registry import HbmPortFlowRegistry
    scheduler._hbm_ports = HbmPortFlowRegistry()
    scheduler._snapshot_verify = False
    return scheduler


def _legacy_decode_rate(scheduler) -> float:
    """C3 之前 context=1 / 长度 1 旧口径的字面复刻（对照锚）。"""
    return float(
        estimate_decode_remaining_task_load_ns(
            scheduler.hardware, scheduler.model,
            instance_size=scheduler.topology.instances[0].size,
            current_context_tokens=1, generated_tokens=0,
            average_decode_length=1.0))


class JointDecodeContextTest(unittest.TestCase):
    """C3（WP1c）：decode 计价真实上下文 + estimator 现值 horizon。"""

    def setUp(self) -> None:
        self.scheduler = _make_scheduler()

    def _cost_model(self, *, context_tokens, average_length):
        return self.scheduler._joint_cost_model(
            0,
            decode_context_tokens=context_tokens,
            decode_average_length=average_length)

    def test_long_history_pricing_exceeds_legacy_context_one(self):
        """方向断言：100k 长历史（99k history + 1k input）的 decode 计价
        严格大于 context=1 旧口径（冷启动 horizon=1 下亦成立）。"""
        cold_tokens, source = self.scheduler._joint_horizon.estimate(
            "session_long")
        self.assertEqual((cold_tokens, source), (1, "cold_start_default"))
        long_context = self.scheduler._joint_cost_model(
            0,
            decode_context_tokens=99_000 + 1_000,
            decode_average_length=cold_tokens)
        self.assertGreater(
            long_context.decode_ns_per_token,
            _legacy_decode_rate(self.scheduler))

    def test_per_token_rate_times_horizon_is_stepped_total(self):
        """语义钉：均值折回口径——decode_ns_per_token × horizon == 直调
        estimate_decode_remaining_task_load_ns 的逐步剩余总量（horizon=8
        为 2 的幂，int/8.0 浮点除法精确，可做恒等断言）。"""
        context = 2048
        horizon = 8
        model = self._cost_model(
            context_tokens=context, average_length=horizon)
        stepped_total = estimate_decode_remaining_task_load_ns(
            self.scheduler.hardware, self.scheduler.model,
            instance_size=self.scheduler.topology.instances[0].size,
            current_context_tokens=context, generated_tokens=0,
            average_decode_length=float(horizon),
            running_step_fraction_remaining=1.0)
        self.assertEqual(
            int(model.decode_ns_per_token * horizon), stepped_total)

    def test_cold_start_horizon_flows_through_unchanged(self):
        """冷启动路径回归：无观测样本时 horizon=1 原样进求值——率 ==
        真实上下文处的单步负载（旧口径的"长度 1"形态保留、上下文换真），
        且 JointCostModel 校验通过（率恒正）。"""
        context = 4096
        horizon, source = self.scheduler._joint_horizon.estimate(
            "cold_session")
        self.assertEqual((horizon, source), (1, "cold_start_default"))
        model = self._cost_model(
            context_tokens=context, average_length=horizon)
        single_step = estimate_decode_remaining_task_load_ns(
            self.scheduler.hardware, self.scheduler.model,
            instance_size=self.scheduler.topology.instances[0].size,
            current_context_tokens=context, generated_tokens=0,
            average_decode_length=1.0,
            running_step_fraction_remaining=1.0)
        self.assertEqual(model.decode_ns_per_token, float(single_step))
        self.assertGreater(model.decode_ns_per_token, 0.0)

    def test_estimator_current_value_drives_horizon(self):
        """estimator 现值（非冷启动）进 horizon：session 在线均值 8 与
        显式传 8 的计价率逐位一致。"""
        self.scheduler._joint_horizon.observe_completed("warm_session", 8)
        estimated, source = self.scheduler._joint_horizon.estimate(
            "warm_session")
        self.assertEqual((estimated, source), (8, "session_online_mean"))
        via_estimator = self._cost_model(
            context_tokens=1024, average_length=estimated)
        explicit = self._cost_model(
            context_tokens=1024, average_length=8)
        self.assertEqual(
            via_estimator.decode_ns_per_token,
            explicit.decode_ns_per_token)

    def test_context_evaluated_once_per_construction(self):
        """O(1) 钉：上下文对全部候选同值 → 构造处一次求值（memo 恰新增
        一条 decode 记录；实例快照不触发 decode memo）。"""
        self.scheduler._decode_task_load_cache.clear()
        self._cost_model(context_tokens=100_000, average_length=1)
        self.assertEqual(len(self.scheduler._decode_task_load_cache), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
