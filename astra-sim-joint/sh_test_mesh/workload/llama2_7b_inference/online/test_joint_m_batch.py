#!/usr/bin/env python3
"""test_joint_m_batch.py -- M 批（M1–M8，2026-09-23 chatgpt 验收级审查
修复）的零后端测试。

源：用户转交 chatgpt 审查报告（C1–C19 验收视角 11 条），逐条读码亲验
后裁定（执行计划 §4.3 A17' / PROVENANCE §40）。本文件钉修复面：

  M1（①+⑥）遥测物理口径统一——流数口径除数（合计吞吐无法反推流数）
        + AIMD 一致分母（collective 不再归因给 KV 流）+ ingest 解析；
  M2（②）配额拒候选保留配额前成本（feasible 判定不变）；
  M4（⑤）γ_prefill 生产桥接（face ServiceFactorGroup 离开冷启动）；
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

from joint.joint_cost_model import (  # noqa: E402
    LinkFlowRegistry,
    TelemetryLinkFlowView,
)
from joint.test_joint_fixes import _load  # noqa: E402
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

    def test_flow_count_absent_falls_back_to_rate_caliber(self):
        # 兼容：流数缺席（旧二进制）退 capacity/速率旧口径——单流速率
        # 语义注入（既有 DivisorEffectiveTest 同款形态）。
        view = self._view(rates=10.0 / 3.0, flows=None)
        self.assertEqual(view.divisor_effective((0, 1)), 3.0)

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
        # 自行补设（生产 __init__ :553 区恒设）。
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
            action="remote-read", applicable=True, cost_ns=1234,
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

class GammaPrefillProductionBridgeTest(unittest.TestCase):
    """M4：纯 prefill 列车样本桥接 face ServiceFactorGroup 的 γ——
    value 离开冷启动 1.0（修前零生产喂入点、恒 1.0）。"""

    def test_pure_prefill_sample_updates_gamma(self):
        from joint.event_recursion_predictor import ServiceFactorGroup
        group = ServiceFactorGroup()
        group.observe_valid_service(
            "prefill", observed_ratio=1.25,
            completion_ns=1000, service_duration_ns=500)
        self.assertFalse(group.cold_start)
        self.assertEqual(group.value("prefill"), 1.25)

    def test_sh30_wiring_feeds_kv_manager_factors(self):
        # 文本钉：_observe_service_factors 纯 prefill 分支桥接调用在案
        #（observe_valid_service + kv_manager.service_factors）。
        with open(os.path.join(
                _ONLINE_DIR, "sh30_online_scheduler.py"),
                encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(
            "self.kv_manager.service_factors.observe_valid_service(",
            source)


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
        # + 10 hop + 1/10≈0 stream ⇒ 10；rank1 零字节十跳修前 10×10=100
        # 抬高 wall、修后免 hop ⇒ wall=10（修前本测试得 100）。
        wall = _transfer_ns_shards(
            rates, LinkFlowRegistry(), None,
            paths_by_rank=(((0, 6),),
                           ((0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10),)),
            bytes_by_rank=(1, 0),
            include_self=True, kind="read")
        self.assertEqual(wall, 10)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
