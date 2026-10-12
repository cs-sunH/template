#!/usr/bin/env python3
"""test_d14_horizon_service_factors.py -- D14（v2.0 C01/式(4)/(4a)/§9.2）。

第一批"等价瘦身"+ 语义冻结值变更的钉子（joint修改文档 D14 条目）：

  1. **sum/count 等价**：``CausalHorizonEstimator`` 每会话 ``(count,
     sum)`` 常数空间与旧"完整列表 + sum//len"实现在同一已完成长度序列
     下逐次一致（session/run/cold 三级来源 + version 全等）；百万轮不
     存百万样本（常数空间验收）。
  2. **冷启动先验 32 前向步**（v2.0 式(4) 冻结值）：生产构造点
     （sh30_online_scheduler.__init__）显式 32，估计来源披露
     ``cold_start_default`` 不变。
  3. **式(4a) 边界估计**：``estimate_boundary = max(1, n̂)``——未终止
     至少还要一次前向；不引入样本、不改状态（D18 边界决策的契约入口）。
  4. **ServiceFactors 固定 1/16 EWMA**（v2.0 §9.2，更新规则改变非旧规
     则错误）：平滑对象是效率 η（η_sample = base/actual、汇总样本
     Σbase/Σactual），η_k = 15/16·η_{k−1} + 1/16·η_sample，因子 =
     1/η_k（取倒数与线性平均不可交换，不得对 actual/base 比值直接平
     滑）；首样本同规则、自 η=1 出发；roofline 样本零漂移；flush 幂等。
  5. **transfer_factor 正式链路效率**：作为 ``eta=1/transfer_factor``
     注入 WorldContext 的 NoC/link stream service rate；HBM/pool 无独立
     样本时保持物理峰值，prefill/decode 因子仍只进 pure compute floor。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d14_horizon_service_factors.py  （或 pytest）
"""
import inspect
import os
import random
import sys
import unittest

_HERE = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_HERE)
for _p in (_HERE, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.joint_cost_model import (  # noqa: E402
    JointCostModel,
    JointHardwareRates,
    LinkFlowRegistry,
    RequestView,
    ServiceFactors,
    SessionKVView,
    InstanceLoadView,
    CausalHorizonEstimator,
)


# ====================================================== 旧实现参照 =====


class _LegacyListEstimator:
    """旧"每会话完整列表 + sum//len"实现的字面复刻（等价性参照）。"""

    def __init__(self, cold_start: int) -> None:
        self._cold = cold_start
        self._sessions: dict[str, list[int]] = {}
        self._run_count = 0
        self._run_sum = 0
        self.version = 0

    def observe_completed(self, session_id: str, tokens: int) -> None:
        self._sessions.setdefault(session_id, []).append(tokens)
        self._run_count += 1
        self._run_sum += tokens
        self.version += 1

    def estimate(self, session_id: str) -> tuple[int, str]:
        samples = self._sessions.get(session_id)
        if samples:
            return sum(samples) // len(samples), "session_online_mean"
        if self._run_count:
            return self._run_sum // self._run_count, "run_online_mean"
        return self._cold, "cold_start_default"


class SumCountEquivalenceTest(unittest.TestCase):
    """同一已完成长度序列下与旧均值逐次一致（D14 等价优化验收）。"""

    def test_per_step_identical_to_legacy_list_mean(self):
        rng = random.Random(20260926)
        session_ids = ["s_a", "s_b", "s_c", "s_fresh"]
        new_est = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        old_est = _LegacyListEstimator(cold_start=32)
        for step in range(4000):
            sid = session_ids[rng.randrange(len(session_ids))]
            # 含 0 长度样本：下取整均值与 cold-start 级别路径都受考验。
            tokens = rng.choice([0, 1, 2, 3, 7, 16, 32, 64, 255])
            new_est.observe_completed(sid, tokens)
            old_est.observe_completed(sid, tokens)
            self.assertEqual(new_est.version, old_est.version)
            for probe in session_ids:
                self.assertEqual(
                    new_est.estimate(probe), old_est.estimate(probe),
                    msg=f"step={step} session={probe}")
            # 边界式(4a)同样逐次一致（max(1, ·) 双方同取）。
            self.assertEqual(
                new_est.estimate_boundary(probe),
                (max(1, old_est.estimate(probe)[0]),
                 old_est.estimate(probe)[1]),
                msg=f"step={step} boundary session={probe}")

    def test_constant_space_million_rounds(self):
        # 百万轮不存百万样本：单会话 (count, sum) 常数空间。
        est = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        total = 0
        for step in range(1_000_000):
            tokens = step % 17
            est.observe_completed("hot_session", tokens)
            total += tokens
        self.assertEqual(len(est._session_stats), 1)
        count, total_sum = est._session_stats["hot_session"]
        self.assertEqual(count, 1_000_000)
        self.assertEqual(total_sum, total)
        self.assertEqual(tuple(len(x) for x in est._session_stats.values()),
                         (2,))  # 存储即 (count, sum) 二元组，常数大小
        self.assertEqual(
            est.estimate("hot_session"),
            (total // 1_000_000, "session_online_mean"))
        self.assertEqual(est.version, 1_000_000)


# ====================================================== 式(4)/(4a) =====


class ColdStartAndBoundaryTest(unittest.TestCase):
    """冷启动 32 前向步（式 4 冻结值）+ 边界 max(1, n̂)（式 4a）。"""

    def test_cold_start_default_32_forward_steps(self):
        est = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        self.assertEqual(est.estimate("no_samples"),
                         (32, "cold_start_default"))
        self.assertEqual(est.estimate_boundary("no_samples"),
                         (32, "cold_start_default"))

    def test_production_site_pins_32(self):
        # 生产构造点（sh30 __init__）显式冻结 32 前向步。
        from online.sh30_online_scheduler import Sh30OnlineScheduler
        source = inspect.getsource(Sh30OnlineScheduler.__init__)
        self.assertIn("cold_start_default_forward_steps=32", source)

    def test_boundary_clamps_sub_one_estimate(self):
        est = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        # 0 长度样本把 session 均值拉到 0：边界未终止仍至少 1 步。
        est.observe_completed("s", 0)
        self.assertEqual(est.estimate("s"), (0, "session_online_mean"))
        self.assertEqual(est.estimate_boundary("s"),
                         (1, "session_online_mean"))
        # run 级同理。
        self.assertEqual(est.estimate("other"), (0, "run_online_mean"))
        self.assertEqual(est.estimate_boundary("other"),
                         (1, "run_online_mean"))

    def test_boundary_is_readonly_view(self):
        # 边界读取不引入样本、不改状态（式 4a 是同一估计器的消费口径）。
        est = CausalHorizonEstimator(cold_start_default_forward_steps=32)
        est.observe_completed("s", 9)
        version_before = est.version
        self.assertEqual(
            est.estimate_boundary("s"), (9, "session_online_mean"))
        self.assertEqual(est.version, version_before)
        self.assertEqual(est._session_stats["s"], (1, 9))


# ================================================== ServiceFactors ====


class FixedCoefEwmaTest(unittest.TestCase):
    """D14：固定 1/16 EWMA（v2.0 §9.2；更新规则改变非旧规则错误）。"""

    def test_closed_form_trajectory(self):
        # η 轨迹闭式：η_k = 15/16·η_{k−1} + 1/16·(1/ratio_k)，因子 =
        # 1/η_k（首样本同规则、自 η=1 出发；tick 间隔不影响权值）。
        factors = ServiceFactors()
        ratios = [1.0, 2.0, 3.0, 100.0, 1.5]
        eta = 1.0
        for i, ratio in enumerate(ratios):
            factors.observe_decode(
                actual_ns=int(ratio * 100), base_ns=100, tick_ns=10 * i)
            factors.flush()
            eta = (15.0 / 16.0) * eta + (1.0 / 16.0) * (1.0 / ratio)
            self.assertAlmostEqual(factors.decode_factor, 1.0 / eta,
                                   places=12)
        self.assertEqual(factors.updates["decode_factor"], len(ratios))

    def test_roofline_ratio_is_zero_drift(self):
        # 恰在 roofline 上的样本（ratio=1）不改变因子（不动点）。
        factors = ServiceFactors()
        for i in range(5):
            factors.observe_prefill(
                actual_ns=100, base_ns=100, tick_ns=i)
            factors.flush()
            self.assertEqual(factors.prefill_factor, 1.0)
        self.assertEqual(factors.updates["prefill_factor"], 5)

    def test_flush_idempotent(self):
        factors = ServiceFactors()
        factors.observe_decode(actual_ns=300, base_ns=100, tick_ns=7)
        factors.flush()
        value = factors.decode_factor
        updates = factors.updates["decode_factor"]
        factors.flush()
        self.assertEqual(factors.decode_factor, value)
        self.assertEqual(factors.updates["decode_factor"], updates)


class TransferFactorProjectionTest(unittest.TestCase):
    """transfer_factor enters formal link service without changing bytes."""

    @staticmethod
    def _model(factors: ServiceFactors) -> JointCostModel:
        rates = JointHardwareRates.from_gbps(
            noc_link_gbps=10.0, pool_port_gbps=5.0, local_hbm_gbps=100.0,
            d2d_latency_ns=10, pool_latency_ns=100)
        loads = {
            0: InstanceLoadView(
                instance_index=0, queued_task_load_ns=0,
                running_task_load_ns=0, active_decode_task_load_ns=0,
                hbm_remaining_bytes_by_tp_rank=(10**9, 10**9),
                reclaimable_bytes_by_tp_rank=(10**9, 10**9)),
            1: InstanceLoadView(
                instance_index=1, queued_task_load_ns=0,
                running_task_load_ns=0, active_decode_task_load_ns=0,
                hbm_remaining_bytes_by_tp_rank=(10**9, 10**9),
                reclaimable_bytes_by_tp_rank=(10**9, 10**9)),
        }
        return JointCostModel(
            rates=rates, loads=loads,
            flow_registry=LinkFlowRegistry(),
            service_factors=factors,
            prefill_ns_per_token=1.0, decode_ns_per_token=2.0,
            model_layers=4, instance_tp_size=2,
            route_fn=lambda source, target: ((source, target), 1))

    @staticmethod
    def _session() -> SessionKVView:
        return SessionKVView(
            session_id="s", home_instance=0, resident_instance=0,
            location="local_hbm", history_tokens=64,
            resident_prefix_layers=2,
            history_bytes_by_tp_rank=(128, 128),
            missing_bytes_by_tp_rank=(0, 0))

    @staticmethod
    def _request() -> RequestView:
        return RequestView(
            request_id="r", session_id="s", input_tokens=16,
            history_tokens_before=64, estimated_decode_tokens=8,
            horizon_source="session_online_mean",
            input_kv_bytes_by_tp_rank=(32, 32))

    def test_transfer_factor_slows_link_stream(self):
        # 远端历史使 copy 产生 NoC stream；factor=2 => link eta=1/2。
        baseline = self._model(ServiceFactors())
        inflated = self._model(ServiceFactors(transfer_factor=2.0))
        session = SessionKVView(
            session_id="s", home_instance=0, resident_instance=1,
            location="local_hbm", history_tokens=64,
            resident_prefix_layers=2, history_bytes_by_tp_rank=(128, 128),
            missing_bytes_by_tp_rank=(0, 0))
        request = self._request()
        base = baseline.estimate_P_with_continuation(
            session=session, request=request, instance_index=0,
            action="copy", remote_enabled=True)
        priced = inflated.estimate_P_with_continuation(
            session=session, request=request, instance_index=0,
            action="copy", remote_enabled=True)
        self.assertGreater(priced.cost_ns, base.cost_ns)
        self.assertAlmostEqual(
            inflated._world_context().efficiency.eta_of("link:0->1"),
            0.5)
        self.assertEqual(
            inflated._world_context().efficiency.eta_of("hbm:0"), 1.0)

    def test_prefill_decode_factors_still_price_action(self):
        # 对照组：prefill/decode 因子是计价输入（动代价）。
        baseline = self._model(ServiceFactors())
        slower = self._model(ServiceFactors(prefill_factor=2.0,
                                            decode_factor=3.0))
        base = baseline.estimate_P_with_continuation(
            session=self._session(), request=self._request(),
            instance_index=0, action="stay", remote_enabled=True)
        priced = slower.estimate_P_with_continuation(
            session=self._session(), request=self._request(),
            instance_index=0, action="stay", remote_enabled=True)
        self.assertIsNotNone(base.cost_ns)
        self.assertIsNotNone(priced.cost_ns)
        self.assertNotEqual(base.cost_ns, priced.cost_ns)


if __name__ == "__main__":
    unittest.main(verbosity=2)
