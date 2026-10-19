#!/usr/bin/env python3
"""test_joint_m_batch.py -- M 批（M1–M8，2026-09-23 chatgpt 验收级审查
修复）的零后端测试。

源：用户转交 chatgpt 审查报告（C1–C19 验收视角 11 条），逐条读码亲验
后裁定（执行计划 §4.3 A17' / PROVENANCE §40）。本文件钉修复面：

  M1（①+⑥）遥测物理口径统一——流数口径除数（合计吞吐无法反推流数）
        + AIMD 一致分母（collective 不再归因给 KV 流）+ ingest 解析；
  M2（②）配额拒候选保留配额前成本（feasible 判定不变）；
  M4（⑤）γ_prefill 生产桥接（2026-09-30 §8.1 单账本修复后改钉：
        PublishedServiceFactorGroup 只读发布视图取代 face 本地更新通
        道——M4 时间衰减桥接退役）；
  M5（⑦）零字节 shard 不计 wall-time 逐跳时延。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_m_batch.py   （或 pytest 同路径）
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

from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
from test_joint_k_batch import _model  # noqa: E402


# ==================================================== 1. M1（①）==

class FlowCountDivisorCaliberTest(unittest.TestCase):
    """M1：流数在场 ⇒ 遥测除数取流数口径（物理除数），弃 capacity/合计
    速率旧口径（满载退 1 / 下游瓶颈误判争用双症状）。"""

    def _view(self, *, rates, flows):
        registry = LinkFlowRegistry()
        return registry.with_effective_rates(
            {(0, 1): rates},
            link_capacity_bytes_per_ns=10.0,
            link_flow_counts={(0, 1): flows} if flows is not None else None)

    def test_saturated_multi_flow_not_collapsed_to_one(self):
        # 症状①（对拍）：3 流均分满载——合计吞吐=B=10（旧口径除数
        # 10/10=1），流数口径除数 = 3（修前注册表漏计无法补足）。
        view = self._view(rates=10.0, flows=3.0)
        self.assertEqual(view.divisor_effective((0, 1)), 3.0)

    def test_downstream_capped_single_flow_not_misread(self):
        # 症状②（对拍）：单流受下游瓶颈 0.1B——合计=1（旧口径除数
        # 10/1=10 误判严重争用），流数口径除数 = 1（流慢≠流多）。
        view = self._view(rates=1.0, flows=1.0)
        self.assertEqual(view.divisor_effective((0, 1)), 1.0)

    def test_flow_count_absent_reports_unknown_uses_registered(self):
        # D13：流数缺席（旧二进制速率-only 形态）= 显式未知——除数退
        # 在册流近似（0 在册 → 下界 1），capacity/速率旧口径冒充废除。
        view = self._view(rates=10.0 / 3.0, flows=None)
        self.assertIsNone(view.effective_divisor((0, 1)))
        self.assertEqual(view.divisor_effective((0, 1)), 1.0)
        self.assertEqual(view.disclosure()["unknown_flow_count_links"], 1)

    def test_flow_count_below_one_rejected(self):
        registry = LinkFlowRegistry()
        with self.assertRaises(Exception):
            registry.with_effective_rates(
                {(0, 1): 10.0},
                link_capacity_bytes_per_ns=10.0,
                link_flow_counts={(0, 1): 0.5})


# ==================================================== 2. M1（⑥）==

class AimdCollectiveConsistentDenominatorTest(unittest.TestCase):
    """M1（⑥）：AIMD per_flow 分母 = max(在册流数, 遥测流数)——KV 流+
    collective 均分 B 时真实份额 B/2 触发收缩（修前分母只含在册 KV 流
    ⇒ 输入 B ≥ 1.2·r̂ 恒 comfort 的反向信号）。"""

    def test_collective_share_triggers_shrink(self):
        from joint.link_quota import LinkQuotaTracker
        tracker = LinkQuotaTracker(
            mode="aimd", noc_link_bytes_per_ns=100.0,
            local_hbm_bytes_per_ns=100.0, delta_adm_ns=0)
        # 一条 KV 流在册；链上物理 2 条流（KV + collective）均分
        # B=100 ⇒ KV 份额 50 < r̂=60 ⇒ shrink。
        disclosure = tracker.observe_telemetry(
            1000, {(0, 1): 50.0}, r_hat_kv_bytes_per_ns=60.0)
        # 收缩触发集成员（50 < r̂=60 ⇒ shrink；Q_init 已触底报
        # shrink_floor——同为收缩方向。修前误注入 100 ⇒ comfort
        # 反向信号、quota 扩张）。
        self.assertIn(
            disclosure["links"]["(0, 1)"]["action"],
            ("shrink", "shrink_floor"))
        state = tracker.aimd_link_state((0, 1))
        self.assertEqual(state["last_signal"], "shrink")
        self.assertEqual(tracker.link_quota((0, 1)), 1)

    def test_sh30_denominator_takes_max_with_telemetry_flows(self):
        # SH 调用点口径：_quota_observe_link_telemetry 的 per_flow 分母
        # 取 max(occupancy, flow_counts)——文本钉 + 数值钉。
        from test_joint_quota_integration import _scheduler
        scheduler = _scheduler()
        scheduler._quota_tracker = SimpleNamespace(
            mode="aimd", link_occupancy=lambda link: 1)
        observed = {}

        def fake_observe(now_ns, per_flow, r_hat_kv_bytes_per_ns,
                         allow_expansion=True):
            # O6②：签名跟随 observe_telemetry 的 keyword-only
            # allow_expansion（r̂ 回退窗冻结扩张）；本替身只钉分母口径，
            # 真实行为断言不动。
            observed.update(per_flow)
            return {"links": {}}

        scheduler._quota_tracker.observe_telemetry = fake_observe
        scheduler._link_telemetry_rates = {(0, 1): 100.0}
        scheduler._link_telemetry_flow_counts = {(0, 1): 2.0}
        scheduler._quota_ingest_telemetry(1000)
        # 分母 2（含 collective）：100/2 = 50（修前 100/1 = 100）。
        self.assertEqual(observed, {(0, 1): 50.0})


# ============================================ 3. M1 ingest 解析 ==

class IngestFlowCountParsingTest(unittest.TestCase):
    """M1：link_telemetry 样本 active_flows 字段落账 + 非法值
    fail-closed。"""

    def setUp(self):
        from test_joint_quota_integration import _scheduler
        self.scheduler = _scheduler()
        # F6 教义：__new__ 替身不走 __init__——本测试面需的 M1 新状态
        # 自行补设（生产 __init__ 区恒设）。
        self.scheduler._link_telemetry_flow_counts = {}

    def _ingest(self, sample_extra):
        delta = {
            "tick": 1000,
            "link_telemetry": [{
                "link_id": 0, "served_bytes": 100, "active_ns": 10,
                "window_start_ns": 0, "window_end_ns": 1000,
                **sample_extra}],
        }
        self.scheduler._ingest_link_telemetry(delta)

    def test_active_flows_lands_in_flow_counts(self):
        self._ingest({"active_flows": 2.5})
        self.assertEqual(
            self.scheduler._link_telemetry_flow_counts,
            {(0, 1): 2.5})

    def test_missing_field_leaves_counts_empty(self):
        self._ingest({})
        self.assertEqual(self.scheduler._link_telemetry_flow_counts, {})

    def test_negative_flow_count_fail_closed(self):
        with self.assertRaises(ValueError):
            self._ingest({"active_flows": -1.0})


# ==================================================== 4. M2（②）==

class QuotaRejectedKeepsCostTest(unittest.TestCase):
    """M2：配额拒候选保留 cost_ns（applicable=False 已挡 feasible/
    argmin——配额前经济域可从决策日志重建）。"""

    def test_rejected_candidate_keeps_pre_quota_cost(self):
        from dataclasses import dataclass

        @dataclass
        class _Cand:
            action: str
            applicable: bool
            cost_ns: object = None
            breakdown: object = None
            inapplicable_reason: object = None
            instance_index: int = 1

        from sh30_online_scheduler import Sh30OnlineScheduler
        candidate = _Cand(
            action="copy", applicable=True, cost_ns=1234,
            breakdown="b", instance_index=1)
        verdict = SimpleNamespace(
            admitted=False, inapplicable_reason="quota_link: remaining=0")
        replaced = type(candidate)(
            **{**candidate.__dict__,
               "applicable": False,
               "inapplicable_reason": verdict.inapplicable_reason})
        # 生产替换式（_quota_filter_candidates M1/M2 形态）：只动
        # applicable/inapplicable_reason，cost_ns/breakdown 原样保留。
        self.assertFalse(replaced.applicable)
        self.assertEqual(replaced.cost_ns, 1234)
        # feasible 判据（:2690 同式）仍排除它。
        feasible = [c for c in (replaced,)
                    if c.applicable and c.cost_ns is not None]
        self.assertEqual(feasible, [])

    def test_sh30_wiring_keeps_cost(self):
        # 文本钉（K/L 批 wiring 先例）：_quota_filter_candidates 的
        # 替换不再清 cost_ns/breakdown。
        with open(os.path.join(
                _ONLINE_DIR, "sh30_online_scheduler.py"),
                encoding="utf-8") as handle:
            source = handle.read()
        self.assertNotIn(
            "candidate, applicable=False, cost_ns=None,\n"
            "                    breakdown=None,", source)


# ==================================================== 5. M4（⑤）==

class GammaPrefillPublishedViewTest(unittest.TestCase):
    """§8.1 单账本（2026-09-30 修复，取代 M4 时间衰减桥接）：纯 prefill
    列车样本只更新统一估计器（ServiceSampleRecorder）一次；face γ/
    k_hide 经 PublishedServiceFactorGroup 读只读发布值（与 J 计价
    ServiceFactors 同一发布规则、同一冻结 1/16 η 平滑），本地 observe
    通道 fail-loud 拒绝。"""

    def test_published_view_reads_unified_estimator(self):
        from joint.history_estimator import (
            ServiceSampleRecorder, ServiceWorkSample)
        from joint.prediction_config import PredictionParams
        from joint.event_recursion_predictor import (
            PublishedServiceFactorGroup)
        recorder = ServiceSampleRecorder(PredictionParams())
        view = PublishedServiceFactorGroup(recorder)
        self.assertTrue(view.cold_start)
        self.assertEqual(view.value("prefill"), 1.0)
        recorder.observe(ServiceWorkSample(
            work_id="prefill#1000", category="prefill",
            observation_tick_ns=1000,
            baseline_work_ns=400, busy_service_ns=500))
        recorder.flush()
        # η_sample = base/actual = 400/500 = 0.8 ⇒
        # η_new = 15/16 + (1/16)·0.8，发布因子 = 1/η_new > 1.0（与 J
        # 侧 ServiceFactors 同规则同值）；decode 未收样恒冷启动 1.0。
        self.assertFalse(view.cold_start)
        eta_new = (15.0 / 16.0) * 1.0 + (1.0 / 16.0) * 0.8
        self.assertAlmostEqual(view.value("prefill"), 1.0 / eta_new)
        self.assertAlmostEqual(view.value("decode"), 1.0)
        self.assertEqual(view.snapshot()["groups"]["prefill"]["samples"], 1)

    def test_published_view_rejects_local_sample_channel(self):
        from joint.event_recursion_predictor import (
            PublishedServiceFactorGroup)
        view = PublishedServiceFactorGroup(object())
        with self.assertRaises(RuntimeError):
            view.observe_valid_service(
                "prefill", observed_ratio=1.25,
                completion_ns=1000, service_duration_ns=500)

    def test_sh30_wiring_single_ledger_text_pin(self):
        # 文本钉：sh30 装配注入统一估计器发布源（kv_manager 的
        # service_factor_source + _joint_factors 的 unified_source），
        # 旧 M4 时间衰减桥接调用不在案（同一服务样本二次更新已退役，
        # §8.1"只更新一次统一估计器，再发布只读版本"）。
        with open(os.path.join(
                _ONLINE_DIR, "sh30_online_scheduler.py"),
                encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(
            "service_factor_source=self._joint_service_samples", source)
        self.assertIn("unified_source=self._joint_service_samples", source)
        self.assertNotIn(
            "self.kv_manager.service_factors.observe_valid_service(",
            source)
        with open(os.path.join(
                _WORKLOAD_DIR, "face_scheduler.py"),
                encoding="utf-8") as handle:
            face_source = handle.read()
        self.assertIn("PublishedServiceFactorGroup(", face_source)


# ==================================================== 6. M5（⑦）==

class ZeroByteShardNoHopLatencyTest(unittest.TestCase):
    """M5：零字节 shard 不计 wall-time 逐跳时延（与除数并集过滤、空
    shard 契约同语义族）。chatgpt 同形构造：有数据一跳 + 零字节十跳。"""

    def test_zero_byte_long_path_does_not_inflate_wall(self):
        from joint.joint_cost_model import _transfer_ns_shards
        from joint.test_joint_shard_pricing import _rates
        rates = _rates(noc=10.0, hbm=1e9, lat=10)   # 每跳 10ns
        # N10 加固（chatgpt 指认 6000B/610ns 主导使新旧实现同返 610 钉
        # 不住旧 bug）：数据 shard 改 1B——rank0 真实发流一跳 = 0 startup
        # + 10 hop + ⌈1/10⌉=1 stream（D17 腿服务向上取整）⇒ 11；
        # rank1 零字节十跳修前 10×10=100 抬高 wall、修后免 hop 且零字
        # 节腿严格为 0 ⇒ wall=11（修前本测试得 100）。
        # （2026-10-14 远读退役：kind 取值域缩 ("copy","merge")——原
        # "read" 腿计价分支已删，本构造换 "copy" 同形承载零字节 shard
        # 免 hop 语义。）
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 6),),
                           ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),)),
            bytes_by_rank=(1, 0),
            include_self=True, kind="copy")
        self.assertEqual(wall, 11)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
