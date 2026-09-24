#!/usr/bin/env python3
"""test_joint_n_batch.py -- N 批（N1–N13，2026-09-23 chatgpt 对 M 批交付
的复核）的零后端测试。

源：用户转交复核报告（6 生产问题 + 2 证据认领 + 4 回归覆盖），逐条
读码亲验 + 证据实测后裁定（执行计划 §4.3 A18' / PROVENANCE §41）。
本文件钉修复面：

  N1（生产1）配额同事务槽位复用——读流与 merge 预留 per-link max
        （判据端）+ tracker 借槽/转移（空链 Q=2 TP=2 准入恢复、
        ρ<1（Q=1）结构性排除恢复、settle 转移守恒、预留先撤消解）；
  N2（生产2）流数除数叠加候选——旧流并发两口径取大 + 候选恒叠加；
  N3（生产3）仅流数遥测保留——零速率窗不丢流数、A9' 剪除并集；
  N4（生产4）prefill 基数生产同形——chunk 切分 + 累计 context roofline
        （SH 同形对拍、JCM fn 消费；recompute 的 history 由 JCM 内部按
        驻留二分——O1 后调用方传语义值，"recompute 0 基"表述已失实）；
  N5（生产5）+ N12 上一轮 merge 尾门纯度排除 + 生产列车核销路径
        （γ 经 _observe_service_factors 离开冷启动）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_n_batch.py   （或 pytest 同路径）
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

from joint.link_quota import (  # noqa: E402
    FLOW_ONESHOT,
    FLOW_REALTIME,
    LinkQuotaTracker,
)
from joint.joint_cost_model import LinkFlowRegistry  # noqa: E402
from joint.test_joint_shard_pricing import (  # noqa: E402
    ACTION_COPY,
    _model,
    _rates,
    _request,
    _session,
)

from face_scheduler import estimate_prefill_task_load_ns  # noqa: E402
from joint.joint_cost_model import ServiceFactors  # noqa: E402
from sh30_online_scheduler import Sh30OnlineScheduler  # noqa: E402
from test_joint_quota_integration import (  # noqa: E402
    _hardware,
    _model as _face_model,
    _scheduler,
    _topology,
)


# ==================================================== 1. N1（生产1）==

class QuotaSameTransactionSlotReuseTest(unittest.TestCase):
    """N1：读流与 merge 预留是同事务时序先后阶段——同事务同链槽位取
    max（借槽），跨事务叠加不变。修前 sum 叠加使空链路 Q=2 时 TP=2
    共链 need=3 恒拒（M 证据 128/189 全拒、stress 1928/2430）。"""

    def test_empty_link_tp2_admitted_with_borrow(self):
        tracker = LinkQuotaTracker(
            mode="static", noc_link_bytes_per_ns=200.0,
            local_hbm_bytes_per_ns=100.0, delta_adm_ns=0)  # ρ=2 ⇒ Q=2
        link = (0, 1)
        reverse = (1, 0)
        verdict = tracker.admit_flow(
            owner="r1#readplan", flow_class=FLOW_REALTIME,
            links=(link, link), port_id=0,   # TP=2 共链读流 2 槽
            r_hat_kv_bytes_per_ns=0.5)
        self.assertTrue(verdict.admitted)
        reserve = tracker.reserve_merge(
            "r1", links_forward={link}, links_reverse={reverse},
            port_forward=1, port_reverse=0)
        # 修前：链 (0,1) 需再 1 槽、remaining = 2-2-0 = 0 ⇒ 拒（need=3
        # 空链判据同构）。修后：同 rid 读流已占 2 槽，预留需求 2 全部
        # 借槽 ⇒ 准入，reserved 不增。
        self.assertTrue(reserve.admitted)
        self.assertEqual(tracker.link_occupancy(link), 2)
        self.assertEqual(tracker.link_reserved(link), 0)
        self.assertEqual(tracker.link_reserved(reverse), 1)
        # 跨事务叠加不变：他事务新流在满载链仍被拒。
        other = tracker.admit_flow(
            owner="r2#x", flow_class=FLOW_ONESHOT, links=(link,))
        self.assertFalse(other.admitted)

    def test_borrow_transfers_to_reserve_on_settle(self):
        tracker = LinkQuotaTracker(
            mode="static", noc_link_bytes_per_ns=200.0,
            local_hbm_bytes_per_ns=100.0, delta_adm_ns=0)
        link = (0, 1)
        tracker.admit_flow(
            owner="r1#readplan", flow_class=FLOW_REALTIME,
            links=(link, link), port_id=0, r_hat_kv_bytes_per_ns=0.5)
        # fwd = {link, (1,0)}：链 link 的预留需求 1（per-direction 各 1
        # 槽），借 1（occ 2 ≥ 1）⇒ reserved 增量 0。
        tracker.reserve_merge("r1", links_forward={link, (1, 0)},
                              port_forward=1)
        self.assertEqual(
            tracker.link_occupancy(link) + tracker.link_reserved(link), 2)
        tracker.release_flow("r1#readplan")
        # settle 转移：读流 2 槽释放、借记 1 槽转移为 reserved ⇒ occ 0
        # + res 1（总计数 2 不变 ⇒ 无新准入窗口；另 1 槽为预留撤销后
        # 的真实余量——demand 1 只借 1）。
        self.assertEqual(tracker.link_occupancy(link), 0)
        self.assertEqual(tracker.link_reserved(link), 1)
        self.assertEqual(
            tracker.link_occupancy(link) + tracker.link_reserved(link), 1)
        tracker.release_merge("r1")
        self.assertEqual(tracker.link_reserved(link), 0)

    def test_reserve_canceled_before_settle_dissolves_borrow(self):
        tracker = LinkQuotaTracker(
            mode="static", noc_link_bytes_per_ns=200.0,
            local_hbm_bytes_per_ns=100.0, delta_adm_ns=0)
        link = (0, 1)
        tracker.admit_flow(
            owner="r1#readplan", flow_class=FLOW_REALTIME,
            links=(link,), port_id=0, r_hat_kv_bytes_per_ns=0.5)
        tracker.reserve_merge("r1", links_forward={link}, port_forward=1)
        # 读流未 settle、预留先撤（未裁决整体撤销）：借记就地消解，
        # 读流 occupancy 回归自身语义，账目全清。
        tracker.release_merge("r1")
        self.assertEqual(tracker.link_occupancy(link), 1)
        self.assertEqual(tracker.link_reserved(link), 0)
        tracker.release_flow("r1#readplan")
        self.assertEqual(tracker.link_occupancy(link), 0)
        self.assertEqual(tracker._merge_borrow, {})

    def test_rho_below_one_single_tp_no_longer_structurally_excluded(self):
        # ρ=1（noc=hbm=100）⇒ Q_init = max(1, floor(1)) = 1；TP=1 读流
        # 1 槽 + merge 预留双向各 1（不同链）。修前 sum 判据：读链需
        # 2 > Q=1 恒拒（计划 C9 "ρ<1 域空化由定价涌现"被违反）；修后
        # max 判据链 (0,1) 需 1、借槽 ⇒ 准入。
        tracker = LinkQuotaTracker(
            mode="static", noc_link_bytes_per_ns=100.0,
            local_hbm_bytes_per_ns=100.0, delta_adm_ns=0)
        self.assertEqual(tracker.link_quota((0, 1)), 1)
        verdict = tracker.admit_flow(
            owner="r#readplan", flow_class=FLOW_REALTIME,
            links=((0, 1),), port_id=0, r_hat_kv_bytes_per_ns=0.5)
        self.assertTrue(verdict.admitted)
        reserve = tracker.reserve_merge(
            "r", links_forward={(0, 1)}, links_reverse={(1, 0)},
            port_forward=1, port_reverse=0)
        self.assertTrue(reserve.admitted)
        self.assertEqual(tracker.link_reserved((0, 1)), 0)  # 借槽

    def test_sh30_verdict_uses_per_link_max(self):
        # 文本钉：判据端 demand = per-link max(read, merge)（A18'(a)）
        # ——与 tracker 借槽（link_quota.py）同构（杜绝"判据过而入册
        # 拒"防御分支成为常规路径）。
        with open(os.path.join(
                _ONLINE_DIR, "sh30_online_scheduler.py"),
                encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("merge_demand", source)
        self.assertIn(
            "demand[edge] = max(demand.get(edge, 0), merge_need)",
            source)
        with open(os.path.join(
                _WORKLOAD_DIR, "joint", "link_quota.py"),
                encoding="utf-8") as handle:
            quota_source = handle.read()
        self.assertIn("_owner_rid_slots_on_link", quota_source)
        self.assertIn("_merge_borrow", quota_source)


# ==================================================== 2. N2（生产2）==

class DivisorCandidateStackingTest(unittest.TestCase):
    """N2：旧流并发两口径（注册 flows / 遥测流数）取大 + 候选恒叠加。
    修前 max(注册含候选, 遥测旧流) 漏"候选 + 未登记旧流"——两未登记
    旧流 + 候选实际 3 模型算 2。"""

    def test_unregistered_flows_stack_with_candidate(self):
        registry = LinkFlowRegistry()   # 注册侧空：两条 collective 未登记
        view = registry.with_effective_rates(
            {(0, 1): 10.0 / 3.0},
            link_capacity_bytes_per_ns=10.0,
            link_flow_counts={(0, 1): 2.0})
        # 无候选：旧流并发 = max(0, 2) = 2。
        self.assertEqual(view.divisor_effective((0, 1)), 2.0)
        # 候选（include_self + 份额 1）：2 + 1 = 3（修前 max(1, 2) = 2）。
        self.assertEqual(
            view.divisor_effective(
                (0, 1), include_self=True, self_overlap=1), 3.0)
        self.assertEqual(
            view.divisor_multi(((0, 1),), include_self=True), 3.0)
        self.assertEqual(
            view.divisor((0, 1, 2), include_self=True), 3.0)

    def test_registered_side_still_dominates_when_higher(self):
        # 注册 5 流 + 遥测 2（注册侧占优——C7 max 防漏计保守性保留）
        # + 候选 ⇒ 6（与既有 test_registered_bottleneck_dominates 同值）。
        registry = LinkFlowRegistry()
        for _ in range(5):
            registry.register(0, 1)
        view = registry.with_effective_rates(
            {(0, 1): 5.0}, link_capacity_bytes_per_ns=10.0,
            link_flow_counts={(0, 1): 2.0})
        self.assertEqual(
            view.divisor_effective(
                (0, 1), include_self=True, self_overlap=1), 6.0)


# ==================================================== 3. N3（生产3）==

class FlowOnlyTelemetryRetentionTest(unittest.TestCase):
    """N3：served=0 ∧ active>0（C++ 整字节进位合法短窗）流数条目保留
    ——该窗口 collective 等未登记流在场的唯一证据；A9' 剪除遍历两字
    典并集（纯流数条目不漏剪陈旧驻留）。"""

    @staticmethod
    def _ingest(scheduler, tick, link_id, served, active, flows):
        # F6：替身补设（quota_integration 的 _scheduler 未覆盖 ingest
        # 零速率窗计数器——漏设 = AttributeError）。
        if not hasattr(scheduler, "_telemetry_zero_rate_dropped"):
            scheduler._telemetry_zero_rate_dropped = 0
        scheduler._ingest_link_telemetry({
            "tick": tick,
            "link_telemetry": [{
                "link_id": link_id, "served_bytes": served,
                "active_ns": active, "active_flows": flows,
                "window_start_ns": tick - 1000, "window_end_ns": tick}]})

    def test_zero_rate_window_keeps_flow_count(self):
        scheduler = _scheduler(quota_mode="static")
        # link_id 5 ⇒ 端点 (3, 2)（A5'/B3 换算表实测方向）。
        self._ingest(scheduler, 1000, 5, served=0, active=500, flows=2.0)
        self.assertNotIn((3, 2), scheduler._link_telemetry_rates)
        # 修前：流数同窗被 pop——丢 collective 在场证据。
        self.assertEqual(
            scheduler._link_telemetry_flow_counts.get((3, 2)), 2.0)
        self.assertEqual(scheduler._telemetry_zero_rate_dropped, 1)

    def test_flow_only_entry_pruned_by_presence_check(self):
        scheduler = _scheduler(quota_mode="static")
        self._ingest(scheduler, 1000, 5, served=0, active=500, flows=2.0)
        # 下一包该链缺席（C++ 按契约省略 = 空闲信号）⇒ 纯流数条目同受
        # A9' 在场核对剪除（修前只遍历 rates 键 ⇒ 漏剪陈旧驻留）。
        scheduler._ingest_link_telemetry({
            "tick": 2000, "link_telemetry": []})
        self.assertNotIn((3, 2), scheduler._link_telemetry_flow_counts)

    def test_jcm_disclosure_reports_flow_count_coverage(self):
        registry = LinkFlowRegistry()
        view_both = registry.with_effective_rates(
            {(0, 1): 5.0}, link_capacity_bytes_per_ns=10.0,
            link_flow_counts={(0, 1): 2.0})
        report = view_both.disclosure()
        self.assertEqual(report["flow_count_links"], 1)
        self.assertEqual(report["legacy_rate_divisor_links"], 0)
        view_legacy = registry.with_effective_rates(
            {(1, 2): 5.0}, link_capacity_bytes_per_ns=10.0)
        report_legacy = view_legacy.disclosure()
        self.assertEqual(report_legacy["flow_count_links"], 0)
        # 旧二进制缺 active_flows ⇒ capacity/速率旧口径降级可辨认。
        self.assertEqual(report_legacy["legacy_rate_divisor_links"], 1)


# ==================================================== 4. N4（生产4）==

class PrefillShapedBaseTest(unittest.TestCase):
    """N4：prefill 整段负载生产同形——p_chunk 切分 + 累计 context 逐
    chunk roofline 求和（线性外推丢二次形状：10 输入 + 90 历史预测
    1800 vs 单 chunk roofline 4464，chatgpt 复算例）。"""

    @staticmethod
    def _sh():
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.hardware = _hardware()
        scheduler.model = _face_model()
        scheduler.topology = _topology()
        scheduler._prefill_task_cache = {}
        scheduler._task_load_cache_capacity = 64
        scheduler.p_chunk = 512
        return scheduler

    def test_shaped_total_matches_manual_chunk_sum(self):
        scheduler = self._sh()
        instance_size = scheduler.topology.instances[0].size
        manual = 0
        completed = 0
        while completed < 600:                     # 512 + 88 两 chunk
            chunk = min(512, 600 - completed)
            manual += estimate_prefill_task_load_ns(
                scheduler.hardware, scheduler.model,
                instance_size=instance_size, chunk_tokens=chunk,
                context_tokens=100 + completed + chunk)
            completed += chunk
        self.assertEqual(
            scheduler._joint_prefill_total_load_ns(600, 100), manual)

    def test_shaped_diverges_from_linear_extrapolation(self):
        # 形状差异实证：chunk 化整段负载 ≠ (1,1) 单 token 线性外推
        #（chatgpt 指控实质；其复算例数字 1800/4464 未复现——口径未
        # 披露，A18'(d) 如实登记）。本测试硬件（低带宽小 perf）下线性
        # 外推**高估** 3.5–7×（每 chunk 摊薄固定项主导）；生产参数
        #（4050/1640/261TF）下同向（330 vs 3100）。绝对值保真是修复
        # 实质，偏差方向逐参数区。
        scheduler = self._sh()
        shaped = scheduler._joint_prefill_total_load_ns(10, 90)
        per_token = float(estimate_prefill_task_load_ns(
            scheduler.hardware, scheduler.model,
            instance_size=scheduler.topology.instances[0].size,
            chunk_tokens=1, context_tokens=1))
        self.assertNotEqual(shaped, 10 * per_token)
        self.assertLess(shaped, 10 * per_token)

    def test_jcm_consumes_fn_and_recompute_passes_zero_history(self):
        import dataclasses
        linear_model = _model(rates=_rates(noc=10.0, hbm=1e9))
        shaped_model = dataclasses.replace(
            linear_model,
            prefill_task_load_ns_fn=lambda tokens, history: (
                1000 + tokens + history))
        kwargs = dict(session=_session(), request=_request(),
                      instance_index=1, action=ACTION_COPY,
                      remote_enabled=True)
        linear = linear_model.estimate_action(**kwargs).cost_ns
        shaped = shaped_model.estimate_action(**kwargs).cost_ns
        # copy：prefill_tokens = 50（input）、history = 100（session 视
        # 图）；因子冷启动 1.0 ⇒ 差 = fn(50,100) − 50×1.0 = 1100。
        self.assertEqual(shaped - linear, 1100)

        seen = {}

        def recording_fn(tokens, history):
            seen["tokens"] = tokens
            seen["history"] = history
            return 5000

        recompute_model = dataclasses.replace(
            linear_model, prefill_task_load_ns_fn=recording_fn)
        candidate = recompute_model.estimate_action(
            session=_session(missing=(1000, 1000)), request=_request(),
            instance_index=1, action="recompute", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        # R13：recompute 跨实例物化副本 0 基——history 恒传 0；tokens
        # = input + missing 折算（> 50）。
        self.assertEqual(seen["history"], 0)
        self.assertGreater(seen["tokens"], 50)


# ============================================= 5. N5 + N12（生产5/回归）==

class MergeTailGatePurityTest(unittest.TestCase):
    """N5：上一轮 merge 尾门（R11(ii)）hold 下一轮列车物理计算 ⇒ span
    混入门等待；纯度排除。N12：走 _observe_service_factors 生产核销
    路径（修前 M4 测试只手工喂因子 + 文本钉）。"""

    @staticmethod
    def _run(merge_tail_gated: bool):
        from joint.event_recursion_predictor import ServiceFactorGroup
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler.hardware = _hardware()
        scheduler.model = _face_model()
        scheduler.topology = _topology()
        scheduler._prefill_task_cache = {}
        scheduler._task_load_cache_capacity = 64
        scheduler._joint_factors = ServiceFactors()
        scheduler.kv_manager = SimpleNamespace(
            service_factors=ServiceFactorGroup())
        scheduler.runtime_by_request_id = {
            "r1": SimpleNamespace(
                joint_span_base_context=0, prefill_tokens_completed=0)}
        state = SimpleNamespace(index=0, last_train_finalize_tick=100)
        train = {
            "emit_tick": 100, "members": [],
            "prefill_chunk_tokens": [("r1", 8)],
            "had_joiners": False, "suffix_gated": False,
            "history_transfer_gated": False,
            "merge_tail_gated": merge_tail_gated}
        scheduler._observe_service_factors(state, train, 200)
        return scheduler

    def test_ungated_train_feeds_both_factor_channels(self):
        scheduler = self._run(merge_tail_gated=False)
        # span = 200−100 = 100、base = roofline(8, context 8) > 0 ⇒
        # JCM 因子（_pending 缓冲在案）与 face γ 双通道入样（N12 生产
        # 路径——修前 M4 测试只手工喂因子 + 文本钉）。
        self.assertIn(
            "prefill_factor", scheduler._joint_factors._pending)
        self.assertFalse(
            scheduler.kv_manager.service_factors.cold_start)
        self.assertNotEqual(
            scheduler.kv_manager.service_factors.value("prefill"), 1.0)

    def test_merge_tail_gated_train_excluded(self):
        scheduler = self._run(merge_tail_gated=True)
        self.assertNotIn(
            "prefill_factor", scheduler._joint_factors._pending)
        self.assertTrue(scheduler.kv_manager.service_factors.cold_start)
        self.assertEqual(
            scheduler.kv_manager.service_factors.value("prefill"), 1.0)

    def test_sh30_purity_gate_text_pin(self):
        path = os.path.join(_ONLINE_DIR, "sh30_online_scheduler.py")
        with open(path, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn("merge_tail_gated", source)
        self.assertIn("_pending_merge_alarms", source)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
