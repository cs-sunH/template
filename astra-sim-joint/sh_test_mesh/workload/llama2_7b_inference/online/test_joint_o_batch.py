#!/usr/bin/env python3
"""test_joint_o_batch.py -- O 批（O1–O14，2026-09-23 13 路终轮审计）SH 侧
修复的零后端测试。

源：终轮审计问题单 SH 侧 10 项（O2/O3/O6/O7/O10/O12/O14），逐项读码
亲验 + 证据实测。本文件钉修复面（按任务单序）：

  任务1（O2，P1）quota 判据不可感知本请求逐出足迹 ⇒ 入册序主先支后 +
        支链失败披露降级（不再 RuntimeError 整 run abort）；
  任务2（O3，P1）merge_tail_gated 纯度门扩域——本实例任意 session 未
        交付 merge watch ∪ 本实例 frontier 未交付完成批尾段；decode
        分支补同款排除（全走 _emit_train 生产路径）；
  任务3（O6①）遥测断供期推进 _telemetry_seq（缺席早退空调用 +
        _quota_ingest_telemetry 空窗不早退）；
  任务4（O6②）r̂ 代表值回退窗 allow_expansion=False（冻结扩张）；
  任务5（O7①）active_flows 同 epoch 混合缺席 fail-closed；
  任务6（O7②）双零样本（带 active_flows 字段）速率/流数同步 pop；
  任务7（O7③）遥测降级三键（telemetry_degradation 块）接入覆盖度裁决；
  任务8（O7④）ingest docstring 窗口均值 ≠ 决策瞬时口径警告；
  任务9（O10①③）run 尾零账断言——tracker snapshot 空账 + R15 三
        注册表 leaked_owners（_PoolPortRegistry 补视图）；
  任务10（O10④/O12/O14）GB _credit_arms 完成残留 fail-closed（登记/
        完成/尾标记三侧）+ remote-read PARTIAL 基可行性分类口径同步
        N1(a) 解除 + quota_oneshot_overflow 计入 run 级配额指标 +
        base_location 空串 fail-closed + 失实 docstring/头注释订正钉。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_o_batch.py   （或 pytest 同路径）
"""

import math
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
)
from online.sh30_online_scheduler import (  # noqa: E402
    Sh30OnlineScheduler,
    _OnlineRequestRuntime,
    _PoolPortRegistry,
)
from test_joint_quota_integration import (  # noqa: E402
    _candidate,
    _decision_rows,
    _make_runtime,
    _scheduler,
    _session_view,
)


# ==================================================== 任务 1（O2，P1）==

class O2QuotaEvictionFootprintEnrollmentTest(unittest.TestCase):
    """O2：判据时刻本请求的逐出足迹**不可知**（history_evictions 由准入
    事务产出——受害选择依赖执行期 KV 账本，不做试探性事务即不可前瞻；
    事务成功才在 :4449 落账，而判据在事务前 :2705 执行）⇒ 判据 demand
    只含主流程足迹（读流/merge 预留的 per-link max 语义由 N1 批钉死，
    本任务不动判据）。事故链（探针实证）：判据 PASS → 支链 #evict 先入
    册占掉末槽 → 主流程 #readplan 入册 FAIL → LinkQuotaError → 调用方
    对已物化预约 fail-closed RuntimeError 整 run abort。修复 = 入册侧
    分派：主流程恒先于逐出支链（结构性杜绝该链），支链失败按
    quota_oneshot_overflow 披露降级不 abort；主流程失败保持 raise
    （真异常——单线程内判据与入册间 tracker 零变更）。"""

    @staticmethod
    def _tight_scheduler():
        """Q=2（static）；预占链路 (1,0) 1 槽 ⇒ 余 1。判据 demand
        (read, merge 同链取 max) = 1 恰过；逐出支链再需 1 槽即撞墙。"""
        scheduler = _scheduler(quota_mode="static")
        # F6：替身补设（quota_integration 夹具未覆盖 O12 计数器）。
        scheduler._quota_oneshot_overflow_events = 0
        verdict = scheduler._quota_tracker.admit_flow(
            owner="pre#copy", flow_class=FLOW_ONESHOT, links=((1, 0),))
        assert verdict.admitted
        return scheduler

    @staticmethod
    def _eviction_transfer():
        """假逐出支链 transfer（remote_store 池写）：noc_path 途经
        (1,0)，与主流程读流共链。"""
        return SimpleNamespace(
            kind="remote_store",
            shards=[SimpleNamespace(noc_path=(0, 1, 0), edge_rank=None)])

    def test_verdict_blind_then_main_first_eviction_degraded(self):
        scheduler = self._tight_scheduler()
        tracker = scheduler._quota_tracker
        session_view = _session_view(resident=1)
        r_hat = scheduler._quota_r_hat_kv_bytes_per_ns(
            session_view.history_tokens)
        # 判据不感知本请求逐出足迹（verdict 签名根本无 runtime）⇒ 同链
        # 支链再需 1 槽的场景下判据仍 PASS——事故链的第一环如实复现。
        verdict = scheduler._quota_candidate_verdict(
            _candidate(0, "remote-read"), session_view, r_hat)
        self.assertTrue(verdict.admitted)
        runtime = _make_runtime("r9")
        runtime.history_evictions = (self._eviction_transfer(),)
        # 修复后：入册主先支后——主流程读流拿到末槽，支链撞墙按披露
        # 降级：返回 True（不 abort）、支链未入册、披露行在案。
        self.assertTrue(scheduler._quota_enroll_admission(
            runtime, session_view, "remote-read", 0, now_ns=0))
        self.assertIn("r9#readplan", scheduler._quota_enrolled)
        self.assertIn("r9#readplan#src", scheduler._quota_enrolled)
        self.assertNotIn("r9#evict", scheduler._quota_enrolled)
        overflow_rows = _decision_rows(scheduler, "quota_oneshot_overflow")
        self.assertEqual(len(overflow_rows), 1)
        self.assertEqual(overflow_rows[0]["request_id"], "r9")
        self.assertIn("O2", overflow_rows[0]["decision"]["note"])
        # 链路 (1,0)：pre 1 + 读流 1 = 2（支链的 +1 未进账——披露降级
        # 是 occupancy-only，释放路径幂等空放）。
        self.assertEqual(tracker.link_occupancy((1, 0)), 2)
        # 守恒：主流程整体释放后账目回落（merge 预留随 release_merge）。
        scheduler._quota_release_readplan_stream("r9", tick=100)
        scheduler._quota_release_admission_phase("r9", tick=100)
        scheduler._quota_tracker.release_merge("r9")
        self.assertEqual(tracker.link_occupancy((1, 0)), 1)
        self.assertEqual(scheduler._quota_enrolled, {})

    def test_copy_main_flow_precedes_eviction_branch(self):
        # copy 同款序：主流程 rid/rid#src 先于支链；支链降级不阻断。
        scheduler = self._tight_scheduler()
        session_view = _session_view(resident=1)
        verdict = scheduler._quota_candidate_verdict(
            _candidate(0, "copy"), session_view, 0.5)
        self.assertTrue(verdict.admitted)
        runtime = _make_runtime("r9")
        runtime.history_evictions = (self._eviction_transfer(),)
        self.assertTrue(scheduler._quota_enroll_admission(
            runtime, session_view, "copy", 0, now_ns=0))
        self.assertIn("r9", scheduler._quota_enrolled)
        self.assertIn("r9#src", scheduler._quota_enrolled)
        self.assertNotIn("r9#evict", scheduler._quota_enrolled)
        self.assertEqual(
            len(_decision_rows(scheduler, "quota_oneshot_overflow")), 1)

    def test_main_flow_failure_stays_fail_closed(self):
        # 主流程入册失败（真异常形态：判据后、入册前 tracker 被外部
        # 扰动偷走末槽——单线程内正常不可达，防御分支的触发形态）保持
        # raise → return False 语义，且已入册流全回滚、零半册。
        scheduler = self._tight_scheduler()
        session_view = _session_view(resident=1)
        verdict = scheduler._quota_candidate_verdict(
            _candidate(0, "remote-read"), session_view, 0.5)
        self.assertTrue(verdict.admitted)
        scheduler._quota_tracker.admit_flow(
            owner="thief", flow_class=FLOW_ONESHOT, links=((1, 0),))
        runtime = _make_runtime("r9")
        runtime.history_evictions = ()
        self.assertFalse(scheduler._quota_enroll_admission(
            runtime, session_view, "remote-read", 0, now_ns=0))
        self.assertEqual(scheduler._quota_enrolled, {})
        # 主流程失败不走披露降级（quota_oneshot_overflow 零行）。
        self.assertEqual(
            len(_decision_rows(scheduler, "quota_oneshot_overflow")), 0)

    def test_stay_local_action_eviction_branch_degraded(self):
        # stay（无跨实例主流程）：唯一入册就是支链——撞墙同样降级不
        # abort（返回 True、无半册）。支链足迹 ×2（(1,0) 双槽）使
        # need=2 > remaining=1 真实撞墙（单槽支链在 stay 场景下入册
        # 成功属合法路径，不构成降级）。
        scheduler = self._tight_scheduler()
        session_view = _session_view(resident=1)
        runtime = _make_runtime("r9")
        runtime.history_evictions = (SimpleNamespace(
            kind="remote_store",
            shards=[SimpleNamespace(noc_path=(0, 1, 0, 1, 0),
                                    edge_rank=None)]),)
        self.assertTrue(scheduler._quota_enroll_admission(
            runtime, session_view, "stay", 1, now_ns=0))
        self.assertNotIn("r9#evict", scheduler._quota_enrolled)
        self.assertEqual(
            len(_decision_rows(scheduler, "quota_oneshot_overflow")), 1)


# ==================================================== 任务 2（O3，P1）==

class O3MergeTailGateExpansionTest(unittest.TestCase):
    """O3：merge_tail_gated 纯度门扩域——① 未交付 merge watch 从"同
    session"扩为"本实例任意 session"（R11(ii) interval gate 按实例
    frontier 锚定，非按 session）；② 新增"本实例 frontier 未交付完成
    批尾段"判据（graph.pending_store_tails 在案、edge_rank ∈ 本实例
    ranks）；③ decode 纯度分支（member_parts 分支）补同款排除（原只
    有 prefill 分支排除）。测试全走 _emit_train 生产路径（_plan_train
    → _emit_train 真 graph 发射），不手工喂因子、不搜源码字符串。"""

    @staticmethod
    def _stay_decode_runtime(request_id="rd", decode=8):
        from online.test_remote_credit_stream import (
            EXEC_INSTANCE, HOME_INSTANCE)
        rt = _OnlineRequestRuntime({
            "request_id": request_id,
            "session_id": "s_" + request_id,
            "turn_index": 0, "queue_index": 0,
            "prefill_length": 16, "decode_length": decode,
            "history_tokens_before": 0,
            "prefill_context_tokens": 16,
            "final_context_tokens": 16 + decode,
        }, 512)
        rt.joint_action = "stay"
        rt.prefill_instance_index = EXEC_INSTANCE
        rt.decode_instance_index = EXEC_INSTANCE
        rt.origin_home_instance = HOME_INSTANCE
        return rt

    @classmethod
    def _emit_pure_decode_train(cls, mutate):
        """生产路径发射一列纯 decode 列车（joiner 空 ⇒ had_joiners
        False），mutate(scheduler) 在发射前注入污染场景。"""
        from online.test_remote_credit_stream import (
            EXEC_INSTANCE, _scheduler)
        from joint.event_recursion_predictor import ServiceFactorGroup
        from joint.joint_cost_model import ServiceFactors
        s = _scheduler()
        rt = cls._stay_decode_runtime()
        s.runtime_by_request_id[rt.request_id] = rt
        s.kv_manager = SimpleNamespace(
            service_factors=ServiceFactorGroup(), _sessions={},
            tp_degree=2)
        s._joint_factors = ServiceFactors()
        state = s.instances[EXEC_INSTANCE]
        state.active_decode.append(rt)
        state.active_decode_lookup.add(rt)
        plan = s._plan_train(state)
        mutate(s)
        s._emit_train(state, plan, [], 1)
        return s, state, plan

    def test_cross_session_alarm_on_this_instance_gates(self):
        # 跨 session（"s_other" ≠ 列车成员 session "s_rd"）、实例归属 =
        # 本实例 ⇒ 门 True（修前同 session 匹配 ⇒ False，漏检）。
        def mutate(s):
            s._pending_merge_alarms["batch_train_merge_r_x"] = {
                "request_id": "r_x", "instance_index": 0,
                "session_id": "s_other"}
        _s, _state, plan = self._emit_pure_decode_train(mutate)
        self.assertTrue(plan["merge_tail_gated"])

    def test_alarm_on_other_instance_does_not_gate(self):
        # 他实例（HOME_INSTANCE=1）的未交付 merge watch 不 hold 本实例
        # frontier ⇒ 门 False（不误排）。
        def mutate(s):
            s._pending_merge_alarms["batch_train_merge_r_x"] = {
                "request_id": "r_x", "instance_index": 1,
                "session_id": "s_other"}
        _s, _state, plan = self._emit_pure_decode_train(mutate)
        self.assertFalse(plan["merge_tail_gated"])

    def test_pending_store_tails_on_this_instance_gates(self):
        # 本实例边缘 rank（0 ∈ ranks(0,1)）有在案完成批尾段 ⇒ 门 True；
        # 他实例边缘 rank（2）在案 ⇒ 门 False。
        def mutate_home_edge(s):
            s.graph.pending_store_tails["s_x"] = [
                (0, 901, 902, 1, 2)]
        _s, _state, plan = self._emit_pure_decode_train(mutate_home_edge)
        self.assertTrue(plan["merge_tail_gated"])

        def mutate_other_edge(s):
            s.graph.pending_store_tails["s_x"] = [
                (2, 901, 902, 1, 2)]
        _s, _state, plan = self._emit_pure_decode_train(mutate_other_edge)
        self.assertFalse(plan["merge_tail_gated"])

    def test_gated_decode_train_excluded_from_factor_channels(self):
        # 生产核销路径：门控列车的 emit→核销 span 不进 JCM 因子与 face
        # γ（修前 decode 分支无 merge_tail_gated 排除 ⇒ 污染样本入样）。
        def mutate(s):
            s._pending_merge_alarms["batch_train_merge_r_x"] = {
                "request_id": "r_x", "instance_index": 0,
                "session_id": "s_other"}
        s, state, plan = self._emit_pure_decode_train(mutate)
        self.assertTrue(plan["merge_tail_gated"])
        s._observe_service_factors(state, plan, 200)
        self.assertNotIn("decode_factor", s._joint_factors._pending)
        self.assertTrue(s.kv_manager.service_factors.cold_start)

    def test_ungated_decode_train_feeds_factor_channels(self):
        # 对照组：无门控 ⇒ 纯 decode 样本正常入样（排除不是无条件丢弃）。
        s, state, plan = self._emit_pure_decode_train(lambda s: None)
        self.assertFalse(plan["merge_tail_gated"])
        s._observe_service_factors(state, plan, 200)
        self.assertIn("decode_factor", s._joint_factors._pending)


# ===================================================== 任务 3（O6①）==

class O6AbsentEpochAdvancesTelemetrySeqTest(unittest.TestCase):
    """O6①：遥测断供期（缺席早退）也推进 tracker._telemetry_seq——修前
    缺席 epoch 不消耗序号，缺测后首个样本 dt 含整段断供时长、K7 连续
    采样纪律被架空（断供期 dt 误计 quiet/进度，单样本触发扩张的缺测
    变体）。修法：缺席早退空调用 _quota_ingest_telemetry（rates 已清 ⇒
    空字典采样，仅序号/遥测钟簿记）+ _quota_ingest_telemetry 空窗不再
    早退（恒推进序号）。"""

    @staticmethod
    def _aimd_scheduler_with_flow():
        scheduler = _scheduler(quota_mode="aimd")
        tracker = scheduler._quota_tracker
        link = scheduler._telemetry_endpoint_link_key(0)
        # 在册流（per_flow 非空的前提）+ 一笔流寿命样本（t_expand 非
        # 冷启动——否则 quiet 永不累计，断供误计无从体现）。
        assert tracker.admit_flow(
            owner="warm", flow_class=FLOW_ONESHOT, links=(link,),
            now_ns=0).admitted
        tracker.release_flow("warm", now_ns=100)
        assert tracker.t_expand_ns is not None
        assert tracker.admit_flow(
            owner="f1", flow_class=FLOW_ONESHOT, links=(link,),
            now_ns=0).admitted
        return scheduler, tracker, link

    @staticmethod
    def _ingest(scheduler, tick, link_id):
        scheduler._ingest_link_telemetry({
            "tick": tick,
            "link_telemetry": [{
                "link_id": link_id, "served_bytes": 1000, "active_ns": 1000,
                "window_start_ns": tick - 1000, "window_end_ns": tick}]})

    def test_absent_epoch_advances_seq_and_gap_dt_not_counted(self):
        scheduler, tracker, link = self._aimd_scheduler_with_flow()
        link_id = 0
        # epoch 1（在场）。
        self._ingest(scheduler, 1000, link_id)
        self.assertEqual(tracker._telemetry_seq, 1)
        # epoch 2（断供：无 link_telemetry 键）——修前 seq 停摆。
        scheduler._ingest_link_telemetry({"tick": 2000})
        self.assertEqual(tracker._telemetry_seq, 2)
        self.assertEqual(scheduler._link_telemetry_rates, {})
        # epoch 3（缺测后首个样本）：contiguous 断开 ⇒ 断供期 dt 不计入
        # quiet_ns、不触发扩张（T_expand = 10×100 = 1000 << 断供 1000ns
        # +窗 1000ns 的合供 2000——修前 contiguous=True ⇒ quiet=2000 ≥
        # T_expand ⇒ 单样本扩张）。
        disclosures = []
        real_observe = tracker.observe_telemetry

        def capture(now_ns, per_flow, **kwargs):
            disclosure = real_observe(now_ns, per_flow, **kwargs)
            disclosures.append(disclosure)
            return disclosure

        tracker.observe_telemetry = capture
        quota_before = tracker.link_quota(link)
        self._ingest(scheduler, 3000, link_id)
        self.assertEqual(tracker._telemetry_seq, 3)
        entry = disclosures[-1]["links"][repr(link)]
        self.assertFalse(entry["contiguous"])
        self.assertEqual(entry["quiet_ns"], 0)
        self.assertEqual(entry["quota_after"], quota_before)

    def test_idle_epoch_with_no_registered_flow_also_advances_seq(self):
        # _quota_ingest_telemetry 空窗不早退的旁证：链路无在册流时
        # per_flow 恒空，修前 observe 永不调用、序号停摆；修后每 epoch
        # 序号推进（遥测钟按 epoch 走，恢复在册后首样本 contiguous 断开）。
        scheduler = _scheduler(quota_mode="aimd")
        tracker = scheduler._quota_tracker
        self.assertEqual(tracker._telemetry_seq, 0)
        self._ingest(scheduler, 1000, 0)
        self.assertEqual(tracker._telemetry_seq, 1)


# ===================================================== 任务 4（O6②）==

class O6RepresentativeRHatFreezesExpansionTest(unittest.TestCase):
    """O6②：r̂ 代表值回退窗（active_decode 全实例瞬空）向 observe_telemetry
    传 allow_expansion=False——代表信号不是真实舒适证据，不得驱动
    additive-increase 扩张（舒适判定照常披露，streak 不计入、解冻后
    重新累计满 T_expand——防解冻瀑布）。"""

    @staticmethod
    def _capture(scheduler):
        tracker = scheduler._quota_tracker
        disclosures = []
        real_observe = tracker.observe_telemetry

        def capture(now_ns, per_flow, **kwargs):
            disclosure = real_observe(now_ns, per_flow, **kwargs)
            disclosures.append(disclosure)
            return disclosure

        tracker.observe_telemetry = capture
        return disclosures

    @staticmethod
    def _aimd_scheduler():
        scheduler = _scheduler(quota_mode="aimd")
        tracker = scheduler._quota_tracker
        link = scheduler._telemetry_endpoint_link_key(0)
        assert tracker.admit_flow(
            owner="warm", flow_class=FLOW_ONESHOT, links=(link,),
            now_ns=0).admitted
        tracker.release_flow("warm", now_ns=100)
        assert tracker.t_expand_ns == 1000  # k=10 × EWMA(100)
        assert tracker.admit_flow(
            owner="f1", flow_class=FLOW_ONESHOT, links=(link,),
            now_ns=0).admitted
        return scheduler, tracker, link

    def _comfort_ingest(self, scheduler, tick, link_id, r_hat):
        # 速率取 1.5×r̂（恒 comfort 带）× 1000ns 窗。
        served = int(math.ceil(1.5 * r_hat * 1000.0))
        scheduler._ingest_link_telemetry({
            "tick": tick,
            "link_telemetry": [{
                "link_id": link_id, "served_bytes": served,
                "active_ns": 1000,
                "window_start_ns": tick - 1000, "window_end_ns": tick}]})

    def test_representative_window_disallows_expansion(self):
        scheduler, tracker, link = self._aimd_scheduler()
        r_hat = scheduler._quota_r_hat_kv_bytes_per_ns()
        disclosures = self._capture(scheduler)
        quota_before = tracker.link_quota(link)
        # 连续两个 comfort 窗（各 1000ns ⇒ 合计 2000 ≥ T_expand=1000）：
        # 修前第二窗即扩张；修后全程冻结（comfort_frozen、quiet 零结转）。
        self._comfort_ingest(scheduler, 1000, 0, r_hat)
        self._comfort_ingest(scheduler, 2000, 0, r_hat)
        self.assertEqual(
            [d["allow_expansion"] for d in disclosures], [False, False])
        entry = disclosures[-1]["links"][repr(link)]
        self.assertEqual(entry["action"], "comfort_frozen")
        self.assertEqual(entry["quiet_ns"], 0)
        self.assertEqual(entry["quota_after"], quota_before)
        self.assertEqual(tracker.link_quota(link), quota_before)

    def test_real_decode_window_allows_expansion(self):
        scheduler, tracker, link = self._aimd_scheduler()
        # active_decode 非空 ⇒ r̂ 为真实负载视图信号 ⇒ 扩张不冻结。
        scheduler.instances[0].active_decode.append(SimpleNamespace(
            prefill_context_tokens=64, decode_tokens_consumed=0))
        r_hat = scheduler._quota_r_hat_kv_bytes_per_ns()
        disclosures = self._capture(scheduler)
        self._comfort_ingest(scheduler, 1000, 0, r_hat)
        self._comfort_ingest(scheduler, 2000, 0, r_hat)
        self.assertEqual(
            [d["allow_expansion"] for d in disclosures], [True, True])
        entry = disclosures[-1]["links"][repr(link)]
        self.assertNotEqual(entry["action"], "comfort_frozen")


# ============================================ 任务 5/6（O7①/O7②）==

class O7MixedFlowsPresenceFailClosedTest(unittest.TestCase):
    """O7①：同一 epoch 内 active_flows 部分在场部分缺席 = 不可能出自
    同一版本二进制（字段随版本整体缺席/在场）⇒ producer 异常，
    fail-closed raise；全缺席按旧二进制容忍、全在场正常处理。"""

    def setUp(self):
        self.scheduler = _scheduler(quota_mode="static")
        self.endpoint = self.scheduler._telemetry_endpoint_link_key(0)

    def _ingest(self, samples):
        self.scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": samples})

    @staticmethod
    def _sample(link_id, served=100, active=100, flows=None):
        sample = {
            "link_id": link_id, "served_bytes": served, "active_ns": active,
            "window_start_ns": 0, "window_end_ns": 1000}
        if flows is not None:
            sample["active_flows"] = flows
        return sample

    def test_mixed_presence_raises(self):
        with self.assertRaises(ValueError):
            self._ingest([
                self._sample(0, flows=2.0),
                self._sample(1),
            ])

    def test_all_absent_legacy_tolerated(self):
        # 全缺席 = 旧二进制 ⇒ 容忍、流数不落账。
        self._ingest([self._sample(0), self._sample(1)])
        self.assertEqual(self.scheduler._link_telemetry_flow_counts, {})
        self.assertEqual(
            self.scheduler._link_telemetry_rates, {
                self.scheduler._telemetry_endpoint_link_key(0): 1.0,
                self.scheduler._telemetry_endpoint_link_key(1): 1.0,
            })

    def test_all_present_processed(self):
        self._ingest([
            self._sample(0, flows=2.0),
            self._sample(1, flows=3.0),
        ])
        self.assertEqual(
            self.scheduler._link_telemetry_flow_counts, {
                self.scheduler._telemetry_endpoint_link_key(0): 2.0,
                self.scheduler._telemetry_endpoint_link_key(1): 3.0,
            })

    def test_active_window_rejects_less_than_one_flow(self):
        for flows in (0, 0.5):
            with self.subTest(flows=flows), self.assertRaisesRegex(
                    ValueError, "active_flows"):
                self._ingest([self._sample(0, flows=flows)])
            self.assertNotIn(
                self.endpoint, self.scheduler._link_telemetry_flow_counts)

    def test_idle_window_rejects_positive_flow_count(self):
        with self.assertRaisesRegex(ValueError, "active_flows"):
            self._ingest([self._sample(0, served=0, active=0, flows=1)])


class O7DoubleZeroSyncClearTest(unittest.TestCase):
    """O7②：双零样本（served=0 ∧ active=0）的缓存速率与流数同步 pop——
    修前速率不 pop（在场键豁免 A9' 剪除 ⇒ 陈旧速率驻留）、流数可能被
    本窗字段重落地（max(1.0, 0) = 1 的幽灵流）⇒ 只清一边。"""

    def setUp(self):
        self.scheduler = _scheduler(quota_mode="static")
        self.endpoint = self.scheduler._telemetry_endpoint_link_key(0)
        # F6：替身补设（quota_integration 夹具未覆盖零速率窗计数器）。
        if not hasattr(self.scheduler, "_telemetry_zero_rate_dropped"):
            self.scheduler._telemetry_zero_rate_dropped = 0

    def _ingest(self, samples):
        self.scheduler._ingest_link_telemetry({
            "tick": 1000, "link_telemetry": samples})

    def test_double_zero_sync_pops_rate_and_flow_count(self):
        # 前窗忙碌：速率 + 流数均在案。双零窗带 active_flows 字段
        # （C++ 显式全闲置窗）⇒ 同步 pop。
        self.scheduler._link_telemetry_rates[self.endpoint] = 5.0
        self.scheduler._link_telemetry_flow_counts[self.endpoint] = 3.0
        self._ingest([{
            "link_id": 0, "served_bytes": 0, "active_ns": 0,
            "active_flows": 0,
            "window_start_ns": 0, "window_end_ns": 1000}])
        self.assertNotIn(self.endpoint, self.scheduler._link_telemetry_rates)
        self.assertNotIn(
            self.endpoint, self.scheduler._link_telemetry_flow_counts)

    def test_double_zero_with_zero_flows_field_no_ghost_flow(self):
        # 双零窗带 active_flows=0：修前 max(1.0, 0) = 1.0 幽灵流落地；
        # 修后同步 pop ⇒ 流数不落账。
        self.scheduler._link_telemetry_rates[self.endpoint] = 5.0
        self._ingest([{
            "link_id": 0, "served_bytes": 0, "active_ns": 0,
            "active_flows": 0,
            "window_start_ns": 0, "window_end_ns": 1000}])
        self.assertNotIn(self.endpoint, self.scheduler._link_telemetry_rates)
        self.assertNotIn(
            self.endpoint, self.scheduler._link_telemetry_flow_counts)

    def test_double_zero_without_flows_field_keeps_presence_vouch(self):
        # 字段缺席的双零样本（旧二进制）无流数通道：维持 A9' "包内样本
        # 为链路在场作保"语义——旧速率保留（A9TelemetryRateExpiryTest
        # 既有钉，O7② 不越权改钉）。
        self.scheduler._link_telemetry_rates[self.endpoint] = 5.0
        self._ingest([{
            "link_id": 0, "served_bytes": 0, "active_ns": 0,
            "window_start_ns": 0, "window_end_ns": 1000}])
        self.assertEqual(
            self.scheduler._link_telemetry_rates[self.endpoint], 5.0)

    def test_zero_rate_window_keeps_flow_count_unchanged(self):
        # N3 对照：served=0 ∧ active>0 的零速率窗——速率 pop、流数保留
        # （collective 在场证据），O7② 不触碰该分支。
        self.scheduler._link_telemetry_rates[self.endpoint] = 5.0
        self._ingest([{
            "link_id": 0, "served_bytes": 0, "active_ns": 500,
            "active_flows": 2.0,
            "window_start_ns": 0, "window_end_ns": 1000}])
        self.assertNotIn(self.endpoint, self.scheduler._link_telemetry_rates)
        self.assertEqual(
            self.scheduler._link_telemetry_flow_counts[self.endpoint], 2.0)


# =========================================== 任务 7/8（O7③/O7④）==

class O7CoverageDecisionDegradationKeysTest(unittest.TestCase):
    """O7③：遥测降级三键（telemetry_degradation 块）接入 run 级覆盖度
    裁决——修前 JCM disclosure 的降级可辨认键只存在于逐决策重建的
    JCM 实例内，run 级零生产落点。三键 = 旧二进制回退（速率在场 ∧
    流数缺席）/ 仅流数条目（N3 零速率窗 carry）/ 零速率窗丢弃计数。"""

    def setUp(self):
        self.scheduler = _scheduler(quota_mode="static")
        # F6：替身补设（quota_integration 夹具未覆盖零速率窗计数器）。
        if not hasattr(self.scheduler, "_telemetry_zero_rate_dropped"):
            self.scheduler._telemetry_zero_rate_dropped = 0

    def _ingest(self, samples):
        self.scheduler._ingest_link_telemetry(
            {"tick": 1000, "link_telemetry": samples})

    @staticmethod
    def _sample(link_id, served=100, active=100, flows=None):
        sample = {
            "link_id": link_id, "served_bytes": served, "active_ns": active,
            "window_start_ns": 0, "window_end_ns": 1000}
        if flows is not None:
            sample["active_flows"] = flows
        return sample

    def _degradation(self):
        return self.scheduler._telemetry_coverage_decision()[
            "telemetry_degradation"]

    def test_legacy_binary_fallback_visible(self):
        # 旧二进制：全样本无 active_flows ⇒ 速率条目无流数配套。
        self._ingest([
            self._sample(0),
            self._sample(1),
        ])
        block = self._degradation()
        self.assertEqual(block["legacy_rate_divisor_entries"], 2)
        self.assertEqual(block["flow_count_only_entries"], 0)
        self.assertEqual(block["zero_rate_samples_dropped"], 0)

    def test_flow_count_only_entry_visible(self):
        # N3 零速率窗：速率丢弃、流数保留 ⇒ 仅流数条目可见。
        self._ingest([self._sample(0, served=0, active=500, flows=2.0)])
        block = self._degradation()
        self.assertEqual(block["flow_count_only_entries"], 1)
        self.assertEqual(block["legacy_rate_divisor_entries"], 0)
        self.assertEqual(block["zero_rate_samples_dropped"], 1)

    def test_clean_full_coverage_no_degradation(self):
        # 全样本带流数 ⇒ 无降级面。
        self._ingest([
            self._sample(0, flows=1.0),
            self._sample(1, flows=2.0),
        ])
        block = self._degradation()
        self.assertEqual(block["legacy_rate_divisor_entries"], 0)
        self.assertEqual(block["flow_count_only_entries"], 0)
        self.assertEqual(block["zero_rate_samples_dropped"], 0)


class O7IngestDocstringCaliberWarningTest(unittest.TestCase):
    """O7④：ingest docstring 的口径警告落字（窗口时间加权均值 ≠ 决策
    瞬时流数——A18'(b) 登记补履行，措辞与 JCM TelemetryLinkFlowView
    docstring 同源）。"""

    def test_ingest_docstring_carries_window_mean_caveat(self):
        doc = Sh30OnlineScheduler._ingest_link_telemetry.__doc__
        self.assertIn("窗口时间加权均值 ≠ 决策瞬时流数", doc)
        self.assertIn("O7④", doc)
        self.assertIn("低估旧流并发", doc)


# ============================================= 任务 9（O10①/O10③）==

class O10PoolPortLeakedOwnersTest(unittest.TestCase):
    """O10③：_PoolPortRegistry.leaked_owners 视图——{owner: tuple(在途
    edge_rank)}、只计在途；注册→漏释放 = 非空，release_owner → 空。"""

    def test_register_then_release_clean(self):
        registry = _PoolPortRegistry()
        registry.register(3, owner="r1#merge")
        registry.register(3, owner="r1#merge")
        registry.register(5, owner="r2")
        self.assertEqual(
            registry.leaked_owners(),
            {"r1#merge": (3, 3), "r2": (5,)})
        registry.release_owner("r1#merge")
        self.assertEqual(registry.leaked_owners(), {"r2": (5,)})
        registry.release_owner("r2")
        self.assertEqual(registry.leaked_owners(), {})

    def test_release_unknown_owner_empty(self):
        registry = _PoolPortRegistry()
        self.assertEqual(registry.release_owner("nobody"), 0)
        self.assertEqual(registry.leaked_owners(), {})


class _RecordingOwnerRegistry:
    def __init__(self):
        self.released = []

    def release_owner(self, owner):
        self.released.append(owner)


class EvictionWatchSettlementTest(unittest.TestCase):
    """逐出 watch 按支链独立 settle，drain 不代替物理尾标记。"""

    @staticmethod
    def _scheduler():
        scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
        scheduler._batch = {"watches": []}
        scheduler._pending_eviction_watches = {}
        scheduler._quota_tracker = None
        scheduler._joint_flows = _RecordingOwnerRegistry()
        scheduler._pool_ports = _RecordingOwnerRegistry()
        scheduler._hbm_ports = _RecordingOwnerRegistry()
        return scheduler

    def test_each_eviction_watch_releases_only_its_own_handle(self):
        scheduler = self._scheduler()
        watches = [
            {
                "request_id": "batch_train_evict_r9_side_0000",
                "owner_request_id": "r9",
                "members": {2: 7},
            },
            {
                "request_id": "batch_train_evict_r9_side_0001",
                "owner_request_id": "r9",
                "members": {3: 9},
            },
        ]
        for index, watch in enumerate(watches):
            scheduler._register_eviction_watch(
                watch, flow_owners=(f"r9#evict{index}",))
        self.assertEqual(len(scheduler._batch["watches"]), 2)

        scheduler._on_eviction_watch(
            watches[0]["request_id"], "prefill", 10)
        self.assertEqual(scheduler._joint_flows.released, ["r9#evict0"])
        self.assertEqual(scheduler._pool_ports.released, ["r9#evict0"])
        self.assertEqual(scheduler._hbm_ports.released, ["r9#evict0"])
        self.assertEqual(
            set(scheduler._pending_eviction_watches),
            {watches[1]["request_id"]})

        scheduler._on_eviction_watch(
            watches[1]["request_id"], "prefill", 11)
        self.assertEqual(
            scheduler._joint_flows.released, ["r9#evict0", "r9#evict1"])
        self.assertEqual(scheduler._pending_eviction_watches, {})
        with self.assertRaisesRegex(RuntimeError, "unknown or duplicate"):
            scheduler._on_eviction_watch(
                watches[1]["request_id"], "prefill", 12)

    def test_prefill_drain_does_not_release_eviction_quota_owner(self):
        scheduler = self._scheduler()
        released = []
        scheduler._quota_release_owner = lambda owner, now_ns: released.append(
            (owner, now_ns))
        scheduler._quota_release_admission_phase("r9", 100)
        self.assertEqual(released, [("r9", 100), ("r9#src", 100)])


class O10RunEndZeroLedgerAssertionTest(unittest.TestCase):
    """O10①/O10③：verify_run_end 的 run 尾零账断言——tracker snapshot
    occ/res/enroll/bulk/merge_borrowed 全零（quota=off 条件化跳过）+
    R15 三注册表 leaked_owners 全空。注入残留 ⇒ fail-closed RuntimeError；
    正常簿记 ⇒ 不误报。"""

    @staticmethod
    def _verify_ready_scheduler(quota_mode="static"):
        scheduler = _scheduler(quota_mode=quota_mode)
        # verify_run_end 全链路的 __new__ 替身补设（F6 教义：本测试面
        # 所需的基类协议账本自行装配）。
        scheduler._runtime_index = {}
        scheduler.runtimes = []
        scheduler._pending_first_steps = set()
        scheduler.arrival_heap = []
        scheduler._ready_frontier = set()
        scheduler._pending_merge_alarms = {}
        scheduler._pending_eviction_watches = {}
        scheduler._stalled_by_instance = {}
        scheduler.completed_requests = 0
        scheduler.expected_request_count = 0
        scheduler.unseen_request_ids = set()
        scheduler.arrived_request_count = 0
        scheduler.completed_request_count = 0
        scheduler.ack_count = 0
        scheduler.delivery_count = 0
        scheduler.last_applied_sequence = -1
        scheduler._delivery_reply_cache = None
        scheduler.in_flight = set()
        scheduler._emitted_by_delivery = set()
        scheduler._acked_through = -1
        scheduler._acked_out_of_order = []
        scheduler._provisional_kv_actions = {}
        scheduler.sensing_enabled = False
        # F6：telemetry 块替身补设（quota_integration 夹具未覆盖）。
        scheduler._telemetry_zero_rate_dropped = 0
        scheduler._quota_oneshot_overflow_events = 0
        # F6：_joint_flows 换真注册表（quota_integration 的 SimpleNamespace
        # 替身无 leaked_owners——O10③ run 尾审计的真实载体）。
        from joint.joint_cost_model import LinkFlowRegistry
        scheduler._joint_flows = LinkFlowRegistry()
        scheduler.kv_manager = SimpleNamespace(
            tp_degree=2, _sessions={},
            assert_final_state=lambda: None)
        return scheduler

    def test_clean_run_passes(self):
        scheduler = self._verify_ready_scheduler()
        scheduler.verify_run_end()  # 不 raise = 零账断言不误报

    def test_tracker_occupancy_residue_fails_closed(self):
        scheduler = self._verify_ready_scheduler()
        tracker = scheduler._quota_tracker
        link = scheduler._telemetry_endpoint_link_key(0)
        assert tracker.admit_flow(
            owner="stuck", flow_class=FLOW_ONESHOT, links=(link,),
            now_ns=0).admitted
        with self.assertRaisesRegex(
                RuntimeError, "quota tracker ledgers non-empty"):
            scheduler.verify_run_end()

    def test_tracker_merge_borrow_residue_fails_closed(self):
        # N1 借槽账残留（readplan settle 未达）⇒ merge_borrowed 非零。
        scheduler = self._verify_ready_scheduler()
        tracker = scheduler._quota_tracker
        link = scheduler._telemetry_endpoint_link_key(0)
        assert tracker.admit_flow(
            owner="r9#readplan", flow_class=FLOW_REALTIME,
            links=(link,), port_id=0, r_hat_kv_bytes_per_ns=0.5,
            now_ns=0).admitted
        assert tracker.reserve_merge(
            "r9", links_forward={link}, links_reverse={(1, 0)},
            port_forward=1, port_reverse=0).admitted
        with self.assertRaisesRegex(
                RuntimeError, "quota tracker ledgers non-empty"):
            scheduler.verify_run_end()

    def test_registry_leak_fails_closed(self):
        scheduler = self._verify_ready_scheduler()
        # 经生产登记口登记一笔跨实例流（链路 + 池端口 + HBM 端点），
        # 完成事件未达 ⇒ 三注册表同在途。
        transfer = SimpleNamespace(
            kind="noc_migrate",
            shards=[SimpleNamespace(noc_path=(0, 2), edge_rank=2)])
        scheduler._register_transfer_flows((transfer,), owner="r9")
        with self.assertRaisesRegex(
                RuntimeError, "leaked owners at run end"):
            scheduler.verify_run_end()
        # 注销后不误报。
        scheduler._release_transfer_flows("r9")
        scheduler.verify_run_end()

    def test_pending_or_scheduled_eviction_handle_fails_closed(self):
        for scheduled in (False, True):
            with self.subTest(scheduled=scheduled):
                scheduler = self._verify_ready_scheduler()
                watch_id = "batch_train_evict_r9_side_0000"
                scheduler._pending_eviction_watches[watch_id] = {
                    "request_id": "r9",
                    "flow_owners": (),
                    "quota_owners": (),
                    "scheduled": scheduled,
                }
                with self.assertRaisesRegex(
                        RuntimeError,
                        "pending/scheduled eviction handles"):
                    scheduler.verify_run_end()

    def test_quota_off_run_skips_tracker_assertion(self):
        scheduler = self._verify_ready_scheduler(quota_mode="off")
        self.assertIsNone(scheduler._quota_tracker)
        # 跟踪器断言被条件化跳过：换成必 raise 的替身也不被调用；注册表
        # 断言（与 quota 无关）仍执行。
        def _boom():
            raise AssertionError("tracker assertion must be skipped")
        scheduler._assert_quota_tracker_ledgers_clean = _boom
        scheduler.verify_run_end()


# ============================================ 任务 10（O10④/O12/O14）==

class O10CreditArmsResidueFailClosedTest(unittest.TestCase):
    """O10④：GB _credit_arms 完成残留 fail-closed（copy/restore 同款
    纪律镜像）——登记侧前序残留 raise（旧账被 setdefault 静默吞并）、
    完成侧未消费残留 raise、单块切片合法缺席不误报。"""

    @staticmethod
    def _builder():
        from online.graph_batch_builder import GraphBatchBuilder
        from online.test_graph_batch_builder import _make_config
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        return builder

    @staticmethod
    def _tail_blocks():
        from face_scheduler import KVTransfer, KVTransferShard
        transfer = KVTransfer(
            kind="noc_migrate", phase="decode", reason="test",
            session_id="s1", trigger_request_id="r9",
            source_instance_index=0, target_instance_index=1,
            total_bytes=8,
            shards=(KVTransferShard(
                source_rank=0, target_rank=2, edge_rank=None, bytes=8,
                noc_path=(0, 1, 2), layer_start=0, layer_end=2),),
            model_layers=2, layer_start=0, layer_end=2,
            resident_prefix_layers_before=2,
            resident_prefix_layers_after=2)
        return ((2, transfer),)

    def test_registration_rejects_stale_ledger(self):
        builder = self._builder()
        member_plan = {
            "request_id": "r9", "session_id": "s1",
            "turn_index": 0, "queue_index": 0}
        blocks = self._tail_blocks()
        builder._emit_credit_stream_tail(
            member_plan, blocks, stage="remote_credit")
        self.assertIn("r9", builder._credit_arms)
        # 前序列车尾标记未结清（残留）⇒ 再登记 fail-closed（修前
        # setdefault 静默吞并旧账）。
        with self.assertRaisesRegex(RuntimeError, "still holds blocks"):
            builder._emit_credit_stream_tail(
                member_plan, blocks, stage="remote_credit")

    def test_completion_rejects_unconsumed_credit_arms(self):
        builder = self._builder()
        builder.set_next_plan({"rX": None})
        builder._block_ends["rX"] = {"seg2": {2: 0, 3: 0}}
        builder._credit_arms["rX"] = {2: {2: 999}}
        with self.assertRaisesRegex(RuntimeError, "unconsumed credit arms"):
            builder.emit_completion_batch({
                "request_id": "rX", "session_id": "sX", "turn_index": 0,
                "queue_index": 0, "decode_instance_index": 1,
                "completion_evictions": [],
                "kv_location_after_completion": "local_hbm"})
        # 账本随 raise 前的 pop 清账（不留半账）。
        self.assertNotIn("rX", builder._credit_arms)

    def test_completion_without_arms_passes(self):
        # 单块切片无尾块：账本合法缺席，完成批不误报。
        builder = self._builder()
        builder.set_next_plan({"rX": None})
        builder._block_ends["rX"] = {"seg2": {2: 0, 3: 0}}
        builder.emit_completion_batch({
            "request_id": "rX", "session_id": "sX", "turn_index": 0,
            "queue_index": 0, "decode_instance_index": 1,
            "completion_evictions": [],
            "kv_location_after_completion": "local_hbm"})


class O12PhysicalFeasibilityRemoteReadPartialTest(unittest.TestCase):
    """O12：_classify_physical_feasibility 的 remote-read 适用面与
    N1(a) 解除同步——PARTIAL 基参与判定（修前仅 LOCAL 基，PARTIAL 基
    上 remote-read 唯一可行时被误报 structural_infeasible）。"""

    def _scheduler_with_action_recorder(self, **config_over):
        from joint.joint_config import JointMechanismConfig
        scheduler = _scheduler()
        config_kwargs = dict(
            category_mode="typed", scheduler_mode="joint",
            layer_policy="adaptive", remote_actions="on",
            quota_mode="static")
        config_kwargs.update(config_over)
        scheduler.joint_config = JointMechanismConfig(**config_kwargs)
        calls = []

        def _feasible(**kwargs):
            calls.append(kwargs["action"])
            return [True, True, True, True]

        scheduler.kv_manager = SimpleNamespace(
            tp_degree=2, _sessions={},
            request_hbm_eventually_feasible_instances=_feasible)
        return scheduler, calls

    @staticmethod
    def _partial_session_view():
        from joint.joint_cost_model import SessionKVView
        return SessionKVView(
            session_id="s1", home_instance=0, resident_instance=0,
            location="partial_hbm_remote", history_tokens=100,
            resident_prefix_layers=1,
            history_bytes_by_tp_rank=(400, 400),
            missing_bytes_by_tp_rank=(300, 300))

    def test_partial_base_remote_read_participates(self):
        scheduler, calls = self._scheduler_with_action_recorder()
        runtime = _make_runtime("r9")
        structural, _detail = scheduler._classify_physical_feasibility(
            runtime, self._partial_session_view())
        self.assertFalse(structural)
        # remote-read 进入判定集合（O12 修前缺席）。
        self.assertIn("remote-read", calls)

    def test_partial_ablation_excludes_remote_read(self):
        # JOINT_REMOTE_READ_PARTIAL=off：PARTIAL 基照旧不进判定（与 JCM
        # action_applicability 消融条件同构）。
        scheduler, calls = self._scheduler_with_action_recorder(
            remote_read_partial="off")
        runtime = _make_runtime("r9")
        scheduler._classify_physical_feasibility(
            runtime, self._partial_session_view())
        self.assertNotIn("remote-read", calls)

    def test_remote_base_still_excluded(self):
        # REMOTE 基（无主 session）仍拒——与 N1(a) 裁定③同口径。
        from joint.joint_cost_model import SessionKVView
        scheduler, calls = self._scheduler_with_action_recorder()
        runtime = _make_runtime("r9")
        view = SessionKVView(
            session_id="s1", home_instance=0, resident_instance=0,
            location="remote_memory", history_tokens=100,
            resident_prefix_layers=0,
            history_bytes_by_tp_rank=(400, 400),
            missing_bytes_by_tp_rank=(400, 400))
        scheduler._classify_physical_feasibility(runtime, view)
        self.assertNotIn("remote-read", calls)


class O12OneshotOverflowRunMetricsTest(unittest.TestCase):
    """O12：quota_oneshot_overflow 事件计入 run 级配额指标
    （joint_decision_metrics 行 quota_events.oneshot_overflows）。"""

    def test_overflow_counted_in_run_metrics(self):
        scheduler = O10RunEndZeroLedgerAssertionTest.\
            _verify_ready_scheduler()
        # 占满链路 (1,0)（Q=2 余 0），decode 相 oneshot 入册必撞墙。
        tracker = scheduler._quota_tracker
        for index in range(2):
            assert tracker.admit_flow(
                owner="pre#{}".format(index), flow_class=FLOW_ONESHOT,
                links=((1, 0),)).admitted
        transfer = SimpleNamespace(
            kind="remote_store",
            shards=[SimpleNamespace(noc_path=(0, 1, 0), edge_rank=None)])
        scheduler._quota_enroll_decode_phase("r9", (transfer,), 100)
        self.assertEqual(scheduler._quota_oneshot_overflow_events, 1)
        # O2 准入逐出支链降级同计数。
        runtime = _make_runtime("rA")
        runtime.history_evictions = (transfer,)
        self.assertTrue(scheduler._quota_enroll_admission(
            runtime, _session_view(resident=0), "stay", 0, now_ns=0))
        self.assertEqual(scheduler._quota_oneshot_overflow_events, 2)
        # 预占流释放后跑 run 尾断言（本测试面为指标计数，非零账）。
        for index in range(2):
            tracker.release_flow("pre#{}".format(index), now_ns=200)
        scheduler.verify_run_end()
        metrics = [row for row in scheduler.online_log_rows
                   if row["kind"] == "joint_decision_metrics"]
        self.assertEqual(len(metrics), 1)
        self.assertEqual(
            metrics[0]["decision"]["quota_events"]["oneshot_overflows"], 2)


class O14JointSessionViewBaseLocationTest(unittest.TestCase):
    """O14：_joint_session_view 的 base_location 显式化——空串
    fail-closed（镜像 face_scheduler.py merge_back 同款），None 落
    REMOTE_MEMORY，实值原样保留。"""

    def _scheduler_with_session(self, base_location):
        scheduler = _scheduler()
        session = SimpleNamespace(
            home_instance=1, working_kind="remote-read",
            base_location=base_location,
            base_history_tokens=100, base_resident_prefix_layers=2)
        scheduler.kv_manager = SimpleNamespace(
            tp_degree=2,
            REMOTE_MEMORY="remote_memory",
            has_session=lambda sid: True,
            session_snapshot=lambda sid: SimpleNamespace(
                location="local_hbm", context_tokens=100,
                resident_prefix_layers=4, instance_index=1),
            _sessions={"s1": session})
        return scheduler

    def test_empty_string_base_location_fails_closed(self):
        scheduler = self._scheduler_with_session("")
        with self.assertRaisesRegex(
                RuntimeError, "empty-string base_location"):
            scheduler._joint_session_view("s1")

    def test_none_base_location_falls_to_remote_memory(self):
        scheduler = self._scheduler_with_session(None)
        view = scheduler._joint_session_view("s1")
        self.assertEqual(view.location, "remote_memory")

    def test_real_base_location_preserved(self):
        scheduler = self._scheduler_with_session("local_hbm")
        view = scheduler._joint_session_view("s1")
        self.assertEqual(view.location, "local_hbm")


class O12DocstringCorrectionTextPinTest(unittest.TestCase):
    """O10-e/O10-f：失实 docstring/头注释的订正钉——JCM 已内部按
    _resident_here 二分 recompute 的 history，"调用方传 history=0"
    旧表述不得残留。"""

    def test_prefill_total_load_docstring_current_semantics(self):
        doc = Sh30OnlineScheduler._joint_prefill_total_load_ns.__doc__
        self.assertIn("_resident_here", doc)
        self.assertNotIn("调用方传 history=0", doc)

    def test_n_batch_header_comment_current_semantics(self):
        path = os.path.join(_ONLINE_DIR, "test_joint_n_batch.py")
        with open(path, encoding="utf-8") as handle:
            header = handle.read(3000)
        # 旧失实搭配（"JCM fn 消费、recompute 0 基"）不得残留；订正注记
        # 在案（"驻留二分"）。
        self.assertNotIn("JCM fn 消费、recompute 0 基", header)
        self.assertIn("驻留二分", header)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
