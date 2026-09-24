#!/usr/bin/env python3
"""test_joint_k_batch.py -- K 批（K1–K8，2026-09-23 外部六路审计修复）
的零后端测试。

源：用户转交 kimi 审计报告（P1×6 + P2×11 + P3 择要），逐条读码亲验
后裁定（执行计划 §4.3 A15' / PROVENANCE §38）。本文件钉修复面：

  K1（P1-①）r̂_KV 的 active_decode 成员平均真正生效；
  K2（P1-②）终轮 merge watch 注册（预留释放 + 披露行 None 字段）；
  K3（P1-⑤）reserve_merge 端口 bulk 需求聚合（同端口双方向）；
  K4（P1-⑥）零字节恢复腿的 rank 内串行链；
  K5（P1-④/P2-1/2/3/5）merge 方向判据镜像物化 / 反向腿自身路径 /
        空 reclaimable 保守化 / 深缺口 covered 拆叠 / copy@home×REMOTE 门；
  K6（P1-③）copy 交接首体块层段门控（M≥3 尾块层区间段）；
  K7（P2-4/7/8/10/11）quota_deferred 分位 / AIMD 缺测不进 streak /
        同值 set 零 bump / 离线平局序镜像 / quota_admissible 派生；
  K8（P3）predictor NaN 拒收 / EvictionPlan 零缺口等长零向量 /
        cold_start 接受后翻。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_k_batch.py   （或 pytest 同路径）
"""

import math
import os
import sys
import unittest
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
_SLO_DIR = os.path.join(_WORKLOAD_DIR, "..", "..", "slo_tools")
for _p in (_ONLINE_DIR, _WORKLOAD_DIR, os.path.abspath(_SLO_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    COPY_HANDOFF_CHUNK_LAYERS,
    KVTransfer,
    KVTransferShard,
    plan_copy_handoff_layer_chunks,
)
from graph_batch_builder import GraphBatchBuilder  # noqa: E402
from joint.event_recursion_predictor import (  # noqa: E402
    EfficiencyFactors,
    EventRecursionError,
    LayerRecursionPredictor,
    RestoreGroupLeg,
    ResourceSnapshot,
    ServiceFactorGroup,
)
from joint.joint_cost_model import _pool_transfer_ns  # noqa: E402
from joint.layer_eviction_policy import (  # noqa: E402
    LayerEvictionPolicy,
    VictimView,
)
from joint.link_quota import (  # noqa: E402
    AIMD_ACTION_EXPAND,
    LinkQuotaTracker,
)
from test_joint_copy_handoff import (  # noqa: E402
    EXEC_RANKS,
    HOME_RANKS,
    _graph_config,
)
from joint.test_joint_fixes import _load, _model  # noqa: E402
from test_joint_quota_integration import _scheduler  # noqa: E402
from joint.test_joint_review3_fixes import (  # noqa: E402
    _request_view,
    _session_view,
)

_SH30_SOURCE_PATH = os.path.join(_ONLINE_DIR, "sh30_online_scheduler.py")


# ==================================================== 1. K1（P1-①）==

class QuotaRhatActiveDecodeAverageTest(unittest.TestCase):
    """r̂_KV 用 active_decode 成员的真实平均上下文（非代表值兜底）。

    审计实证：修前 active_decode 放 context=150 的成员仍返回代表值
    777 的结果（差 2.2 倍）——成员是 runtime 对象、原实现当字符串键查
    runtime_by_request_id.get() 恒 None。"""

    def setUp(self):
        self.scheduler = _scheduler()

    def _member(self, context, consumed):
        return SimpleNamespace(
            prefill_context_tokens=context,
            decode_tokens_consumed=consumed)

    def test_member_average_used_not_representative(self):
        self.scheduler.instances[0].active_decode.extend([
            self._member(100, 0), self._member(180, 20)])
        with_members = self.scheduler._quota_r_hat_kv_bytes_per_ns(
            representative_context_tokens=777)
        self.scheduler.instances[0].active_decode.clear()
        equivalent_representative = (
            self.scheduler._quota_r_hat_kv_bytes_per_ns(
                representative_context_tokens=150))
        representative_777 = self.scheduler._quota_r_hat_kv_bytes_per_ns(
            representative_context_tokens=777)
        self.assertGreater(with_members, 0.0)
        self.assertEqual(with_members, equivalent_representative)
        self.assertNotEqual(with_members, representative_777)


# ==================================================== 2. K2（P1-②）==

class FinalRoundMergeWatchTest(unittest.TestCase):
    """终轮（无 following）merge watch：预留释放 + 披露行 None 字段。

    审计缺陷：watch 仅在 following is not None 时注册 ⇒ 终轮胜者侧
    merge 预留滞留 ⇒ verify_run_end fail-closed abort；修 = 注册条件
    收窄为 runtime.merge_transfers 单条件。"""

    def test_on_merge_done_with_final_round_pending_record(self):
        scheduler = _scheduler(quota_mode="static")
        scheduler.kv_manager = SimpleNamespace(
            kv_delta_find=lambda rid: {"session_id": "s", "seq": 1})
        scheduler._joint_flows = SimpleNamespace(
            release_owner=lambda owner: None)
        scheduler._pending_merge_alarms = {
            "batch_train_merge_r9": {
                "request_id": "r9",
                "session_id": "s",
                "following_request_id": None,
                "next_arrival_world_ns": None,
            },
        }
        scheduler._quota_merge_reserves["r9"] = {
            "state": "adjudicated",
            "winner": "forward", "source": 0, "target": 1}
        # tracker 侧配对登记 + service_done 同款裁决（胜者侧到 merge_done）。
        self.assertTrue(scheduler._quota_tracker.reserve_merge(
            "r9", links_forward=[(0, 1)], links_reverse=[(1, 0)]).admitted)
        scheduler._quota_tracker.adjudicate_merge_direction("r9", "forward")
        scheduler._on_merge_done("batch_train_merge_r9", 5000)
        # 胜者侧预留经 merge_done 释放（终轮同样闭合）。
        self.assertNotIn("r9", scheduler._quota_merge_reserves)
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "merge_done"]
        self.assertEqual(len(rows), 1)
        decision = rows[0]["decision"]
        self.assertIsNone(decision["next_turn_request_id"])
        self.assertIsNone(decision["next_arrival_world_ns"])
        self.assertIsNone(decision["next_turn_arrived_before_merge_done"])

    def test_wiring_final_round_registration_condition(self):
        """接线钉：注册条件已去掉 following 门槛 + 披露行 None 条件化
        （文本级——完整 run 目录夹具对单测不成比例，冒烟矩阵 quota 臂
        为执行级实证载体）。"""
        with open(_SH30_SOURCE_PATH, encoding="utf-8") as handle:
            source = handle.read()
        self.assertIn(
            "            if runtime.merge_transfers:",
            source)
        self.assertNotIn(
            "if (following is not None and runtime.merge_transfers):",
            source)
        self.assertIn(
            '"following_request_id": (\n'
            "                        following.request_id "
            "if following is not None",
            source)
        self.assertIn(
            "(tick > pending[\"next_arrival_world_ns\"])\n"
            "                    if pending[\"next_arrival_world_ns\"] "
            "is not None",
            source)


# ==================================================== 3. K3（P1-⑤）==

class PortBulkAggregationTest(unittest.TestCase):
    """同端口双方向：端口 bulk 名额按聚合需求裁决（N_bulk=2）。"""

    def setUp(self):
        self.t = LinkQuotaTracker(
            mode="aimd", noc_link_bytes_per_ns=4.0,
            local_hbm_bytes_per_ns=2.0, delta_adm_ns=0)
        self.assertEqual(self.t.n_bulk, 2)

    def test_same_port_double_direction_needs_two_slots(self):
        # 先占 port 7 的 1 个 bulk 名额（异请求单方向）。
        self.assertTrue(self.t.reserve_merge(
            "rid-0", links_forward=[(0, 1)], links_reverse=[(1, 0)],
            port_forward=7, port_reverse=8).admitted)
        verdict = self.t.reserve_merge(
            "rid-1", links_forward=[(2, 3)], links_reverse=[(3, 2)],
            port_forward=7, port_reverse=7)
        self.assertFalse(verdict.admitted)
        self.assertIn("aggregated need=2", verdict.inapplicable_reason)
        # 修前：逐方向各验 1 槽 → admitted=True 且 bulk_used(7)=3 破 N_bulk。

    def test_same_port_fits_when_both_slots_free(self):
        self.assertTrue(self.t.reserve_merge(
            "rid-1", links_forward=[(2, 3)], links_reverse=[(3, 2)],
            port_forward=7, port_reverse=7).admitted)
        self.assertEqual(self.t.bulk_used(7), 2)


# ==================================================== 4. K4（P1-⑥）==

class ZeroByteLegSerialChainTest(unittest.TestCase):
    """零字节腿遵守 rank 内串行链（等前驱完成）。

    审计实证：g0=3e6B、g1=0B、g2=1e6B @ 1000B/ns ⇒ 真值 g2 完成于
    4000，修前 R̂=2000（低估 50%%）。计算段（每层 c=100、无内存流）
    驱动时间线步进。"""

    @staticmethod
    def _segments(layers):
        from joint.event_recursion_predictor import ComputeLayerSegment
        return tuple(
            ComputeLayerSegment(
                layer=ell, base_ns_by_rank=(100.0,),
                memory_bytes_by_rank=(0,),
                port_by_rank=("port:0",), group="prefill")
            for ell in range(1, layers + 1))

    def _case(self):
        from joint.event_recursion_predictor import ComputeLayerSegment  # noqa: F401
        legs = (
            RestoreGroupLeg(
                layer_start=0, layer_end=1,
                bytes_by_rank=(3_000_000,),
                path_by_rank=(("pool:0", "port:0"),)),
            RestoreGroupLeg(
                layer_start=1, layer_end=2,
                bytes_by_rank=(0,),
                path_by_rank=(("pool:0", "port:0"),)),
            RestoreGroupLeg(
                layer_start=2, layer_end=3,
                bytes_by_rank=(1_000_000,),
                path_by_rank=(("pool:0", "port:0"),)),
        )
        predictor = LayerRecursionPredictor(
            ResourceSnapshot(peak_bytes_per_ns={"pool:0": 1000.0,
                                                "port:0": 1000.0}),
            legs, self._segments(3), 0,
            EfficiencyFactors(eta={"pool": 1.0}, gamma={"prefill": 1.0}))
        return predictor, legs

    def test_zero_byte_leg_waits_for_predecessor(self):
        predictor, legs = self._case()
        consumption, by_layer_rank, _by_layer = (
            predictor._run_dependency_driven(legs))
        # 层 ℓ（1-based）↔ 腿 ℓ-1：g0 完成于 3000（3e6/1000）。
        self.assertEqual(by_layer_rank[1][0], 3000)
        # 零字节腿：done = max(前驱, now+q̂) = 3000（修前 = 0）。
        self.assertEqual(by_layer_rank[2][0], 3000)
        # 后继腿串行接力：3000 + 1e6/1000（修前 = 1000）。
        self.assertEqual(by_layer_rank[3][0], 4000)
        # 层 3 消费 ≥ 其恢复门 4000（串行链语义传导到消费侧；
        # 修前 g2 提前解锁 ⇒ 消费 ~1100）。
        self.assertGreaterEqual(consumption[3], 4000)

    def test_zero_byte_first_leg_still_immediate(self):
        legs = (
            RestoreGroupLeg(
                layer_start=0, layer_end=1, bytes_by_rank=(0,),
                path_by_rank=(("pool:0", "port:0"),)),
            RestoreGroupLeg(
                layer_start=1, layer_end=2, bytes_by_rank=(5_000,),
                path_by_rank=(("pool:0", "port:0"),)),
        )
        predictor = LayerRecursionPredictor(
            ResourceSnapshot(peak_bytes_per_ns={"pool:0": 1000.0,
                                                "port:0": 1000.0}),
            legs, self._segments(2), 7,
            EfficiencyFactors(eta={"pool": 1.0}, gamma={"prefill": 1.0}))
        _c, by_layer_rank, _b = predictor._run_dependency_driven(legs)
        # 首腿（层 1）无前驱：q̂ 时刻就绪（7）。
        self.assertEqual(by_layer_rank[1][0], 7)


# ============================================ 5. K5（JCM 计价四件）==

class MergeDirectionMirrorTest(unittest.TestCase):
    """K5（P1-④）：方向判据 = 字节比大小镜像物化侧（F15 平局含等号取
    forward），不再取 min（test_home_wait_in_forward_formula 已重钉非平局
    形态——本类补平局钉）。"""

    def test_exact_tie_selects_forward(self):
        model = _model({0: _load(), 1: _load()})
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(500, 500), location="partial_hbm_remote",
                prefix=3),
            # input_bytes=(2500,2500) ⇒ exec_retained = missing(500)+input
            # = (3000,3000) 与 history 相等 ⇒ 平局 → forward（物化同判据）。
            request=_request_view(
                input_tokens=5, decode=0, input_bytes=(2500, 2500)),
            instance_index=1, action="remote-read", remote_enabled=True)
        notes = candidate.breakdown.notes
        self.assertIn("merge_v2_direction=forward", notes)
        self.assertIn("merge_v2_direction_rule=byte_le_tie_forward", notes)
        # 平局前向被选中：merge_ns = 前向腿值（notes 里两腿都披露）。
        forward_note = next(
            note for note in notes
            if note.startswith("merge_v2_forward_ns="))
        self.assertEqual(
            candidate.breakdown.merge_ns, int(forward_note.split("=")[1]))


class ReverseLegOwnPathTest(unittest.TestCase):
    """K5（P2-1）：反向腿沿自身方向（home→exec）计价——前向边上的他流
    抬前向腿、不抬反向腿。"""

    def _reverse_ns(self, with_background):
        from joint.joint_cost_model import LinkFlowRegistry
        flows = LinkFlowRegistry()
        if with_background:
            # 背景 he 流：驻在 exec→home 有向边（前向腿路径，rank 链
            # (1,0) 单跳）。register_path 逐 shard 完整链路序列登记。
            flows.register_path((1, 0), owner="f-bg")
        model = _model({0: _load(), 1: _load()}, flows=flows)
        candidate = model.estimate_action(
            session=_session_view(
                home=0, resident=0, history_bytes=(3000, 3000),
                missing=(1000, 1000), location="partial_hbm_remote",
                prefix=3),
            request=_request_view(input_tokens=5, decode=0,
                                  input_bytes=(5, 5)),
            instance_index=1, action="remote-read", remote_enabled=True)
        notes = candidate.breakdown.notes
        reverse_note = next(
            note for note in notes
            if note.startswith("merge_v2_reverse_ns="))
        return int(reverse_note.split("=")[1])

    def test_background_on_forward_edge_does_not_raise_reverse_leg(self):
        clean = self._reverse_ns(with_background=False)
        contended = self._reverse_ns(with_background=True)
        self.assertEqual(clean, contended)


class EmptyReclaimableDeepGapTest(unittest.TestCase):
    """K5（P2-2）：缺省空 reclaimable = "零可回收"（保守深缺口升级）。"""

    def test_empty_reclaimable_treated_as_nothing_reclaimable(self):
        model = _model({0: _load(), 1: _load()})
        load = SimpleNamespace(
            queued_task_load_ns=0,
            running_task_load_ns=10**7,
            active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(100, 100),
            reclaimable_bytes_by_tp_rank=(),   # 缺省视图：无信息
        )
        notes = []
        wait = model._eviction_wait_estimate(
            load, (5000, 5000), notes, instance_index=1)
        writeback = _pool_transfer_ns(
            total_bytes=4900, divisor=model._pool_divisor(1),
            rates=model.rates)
        # 深缺口升级：max(写回, 活跃剩余) —— 修前 = 写回（静默关闭）。
        self.assertEqual(wait, max(writeback, 10**7))
        self.assertIn("execution_growth_deep_gap_unresolved", notes)


class DeepGapCoveredDecompositionTest(unittest.TestCase):
    """K5（P2-3）：主调用侧深缺口分量被 target_wait 覆盖 ⇒ 不双计。"""

    def test_covered_wait_collapses_deep_gap_component(self):
        model = _model({0: _load(), 1: _load()})
        load = SimpleNamespace(
            queued_task_load_ns=2 * 10**6,
            running_task_load_ns=10**7,
            active_decode_task_load_ns=0,
            hbm_remaining_bytes_by_tp_rank=(100, 100),
            reclaimable_bytes_by_tp_rank=(0, 0),
        )
        notes = []
        wait = model._eviction_wait_estimate(
            load, (5000, 5000), notes, instance_index=1,
            covered_wait_ns=12 * 10**6)   # = target_wait（含全部活跃）
        writeback = _pool_transfer_ns(
            total_bytes=4900, divisor=model._pool_divisor(1),
            rates=model.rates)
        # 深缺口分量 = max(0, 1e7 − 1.2e7) = 0 ⇒ 等待 = 写回。
        self.assertEqual(wait, writeback)
        # 注记语义 = 深缺口条件成立（"存在"与"额外等待"分列）。
        self.assertIn("execution_growth_deep_gap_unresolved", notes)
        # 零 covered（home/exec-merge 侧）：分量完整保留。
        notes2 = []
        wait2 = model._eviction_wait_estimate(
            load, (5000, 5000), notes2, instance_index=1,
            covered_wait_ns=0)
        self.assertEqual(wait2, 10**7)


class CopyAtHomeRemoteBaseSettleTest(unittest.TestCase):
    """K5（P2-5）：REMOTE 基 ×copy@home = 裁定③就地转正——物化侧
    merge_back 不 raise、方向 in_place、零传输（修前 N9 对 home==exec
    一刀切 ⇒ 确定性 abort；JCM 计价面由
    RemoteBaseAtHomeMergePricingTest 原样钉住）。"""

    def test_remote_base_copy_at_home_settles_in_place(self):
        from joint.test_joint_fixes import _manager
        from joint.test_joint_mechanisms import _seed
        kv = _manager()
        _seed(kv, "s", 0, 10, 10, "human")
        session = kv._sessions["s"]
        # 整份逐出 ⇒ REMOTE 基（home 保留 0，仅池 backing）。
        kv._evict_session(
            session, phase="history", reason="test_full_evict",
            trigger_request_id="seed")
        self.assertEqual(session.location, kv.REMOTE_MEMORY)
        self.assertEqual(session.home_instance, 0)
        before, _transfers, _evictions = kv.prepare_prefill(
            session_id="s", target_instance_index=0,
            history_tokens=10, trigger_request_id="t1",
            action="copy")
        self.assertEqual(session.working_kind, "copy")
        kv.expand_prefill(
            session_id="s", instance_index=0, context_tokens=13,
            trigger_request_id="t1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="t1", new_tokens=3)
        # 裁定③：就地保留——零传输、终态 LOCAL@home。
        self.assertEqual(transfers, ())
        merged = kv.session_snapshot("s")
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(merged.home_instance, 0)
        self.assertEqual(merged.location, kv.LOCAL_HBM)
        self.assertEqual(kv.last_merge_outcome["direction"], "in_place")


# ==================================================== 6. K7 六件 ==

class QuotaDeferredTierTest(unittest.TestCase):
    """K7（P2-4）：配额压塌成 {recompute} ⇒ forced_reason=quota_deferred
    （C5 枚举扩一值，A15' 规格变更登记）。"""

    @staticmethod
    def _candidate(action, applicable, reason=None):
        return SimpleNamespace(
            action=action, applicable=applicable,
            inapplicable_reason=reason)

    def _tier(self, candidates, history_tokens):
        from sh30_online_scheduler import Sh30OnlineScheduler
        session_view = SimpleNamespace(history_tokens=history_tokens)
        return Sh30OnlineScheduler._recompute_selection_tier(
            candidates, session_view)

    def test_quota_collapse_labeled_quota_deferred(self):
        tier = self._tier([
            self._candidate("stay", False, "history not resident at target"),
            self._candidate("copy", False, "no history to copy"),
            self._candidate("remote-read", False,
                            "quota_link: link=(0,1) remaining=0"),
            self._candidate("recompute", True)], history_tokens=50)
        self.assertEqual(tier["tier"], "forced")
        self.assertEqual(tier["forced_reason"], "quota_deferred")

    def test_structural_collapse_still_evicted_permanent(self):
        tier = self._tier([
            self._candidate("stay", False, "history not resident at target"),
            self._candidate("copy", False, "no history to copy"),
            self._candidate("remote-read", False, "remote disabled"),
            self._candidate("recompute", True)], history_tokens=50)
        self.assertEqual(tier["forced_reason"], "evicted_permanent")

    def test_first_turn_still_no_history(self):
        tier = self._tier([
            self._candidate("recompute", True),
            self._candidate("remote-read", False,
                            "quota_port: slots exhausted")],
            history_tokens=0)
        self.assertEqual(tier["forced_reason"], "no_history")

    def test_multi_applicable_still_elected(self):
        tier = self._tier([
            self._candidate("stay", True),
            self._candidate("recompute", True)], history_tokens=50)
        self.assertEqual(tier["tier"], "elected")


class AimdGapNotCreditedTest(unittest.TestCase):
    """K7（P2-7）：缺测段"streak 冻结不进不出"——重观测首个样本 dt 不
    计入 quiet_ns（调用序号衔接才计），streak 不重置。"""

    def setUp(self):
        # noc=240/hbm=100（稳定夹具同参数）：r̂=10 端口平价可入
        #（(u+1)×10 ≤ 100）。寿命样本 EWMA=1000 ⇒ T_expand=10000。
        self.t = LinkQuotaTracker(
            mode="aimd", noc_link_bytes_per_ns=240.0,
            local_hbm_bytes_per_ns=100.0, delta_adm_ns=0)
        self.t.admit_flow(
            owner="svc#decode#0", flow_class="realtime",
            links=[(9, 10)], port_id=6,
            r_hat_kv_bytes_per_ns=10.0, now_ns=0)
        self.t.release_flow("svc#decode#0", now_ns=1000)
        self.comfort = 12.0   # >= 1.2 × r_KV=10
        self.r_hat = 10.0

    def test_gap_between_observations_not_credited(self):
        self.t.observe_telemetry(1000, {(0, 1): self.comfort}, self.r_hat)
        # 缺测：中间一次调用本链路缺席（无在册流 ⇒ 字典缺席——另一
        # 链路在册），随后 1000→11000 重观测：
        self.t.observe_telemetry(2000, {(9, 10): self.comfort}, self.r_hat)
        obs = self.t.observe_telemetry(11000, {(0, 1): self.comfort},
                                       self.r_hat)
        record = obs["links"]["(0, 1)"]
        # dt=10000 含整段缺测——不进 streak（修前 quiet=10000 ≥ T_expand
        # 即触发扩张）。
        self.assertFalse(record["contiguous"])
        self.assertEqual(record["action"], "comfort")
        self.assertEqual(self.t.aimd_link_state((0, 1))["quiet_ns"], 0)
        self.assertEqual(self.t.link_quota((0, 1)), 2)
        # 连续样本恢复计 dt。
        obs2 = self.t.observe_telemetry(12000, {(0, 1): self.comfort},
                                        self.r_hat)
        self.assertTrue(obs2["links"]["(0, 1)"]["contiguous"])
        self.assertEqual(self.t.aimd_link_state((0, 1))["quiet_ns"], 1000)

    def test_contiguous_comfort_still_expands(self):
        self.t.observe_telemetry(1000, {(0, 1): self.comfort}, self.r_hat)
        for now in range(2000, 11000, 1000):
            self.t.observe_telemetry(now, {(0, 1): self.comfort}, self.r_hat)
        # quiet = 9000 < T_expand=10000，未扩张。
        self.assertEqual(self.t.link_quota((0, 1)), 2)
        obs = self.t.observe_telemetry(12000, {(0, 1): self.comfort},
                                       self.r_hat)
        self.assertEqual(obs["links"]["(0, 1)"]["action"],
                         AIMD_ACTION_EXPAND)
        self.assertEqual(self.t.link_quota((0, 1)), 3)


class SameValueSetNoBumpTest(unittest.TestCase):
    """K7（P2-8）：set_link_quota 同值调用零 bump（无虚假唤醒）。"""

    def test_same_value_keeps_epoch(self):
        t = LinkQuotaTracker(
            mode="aimd", noc_link_bytes_per_ns=4.0,
            local_hbm_bytes_per_ns=2.0, delta_adm_ns=0)
        t.set_link_quota((0, 1), 5)
        before = t.quota_retry_key()
        t.set_link_quota((0, 1), 5)
        t.set_link_quota((0, 1), 7)
        t.set_link_quota((0, 1), 7)
        after = t.quota_retry_key()
        # 两次真实变化 = 2 次 bump（5→7 一次；两次同值 5/7 不 bump）。
        self.assertEqual(after[0], before[0] + 1)


class OfflineTieMirrorTest(unittest.TestCase):
    """K7（P2-10）：离线 _min_cost 平局键序镜像在线 argmin
    (cost, instance, action)。"""

    def test_instance_first_tie_break_mirrors_online(self):
        from domain_metrics import _min_cost
        candidates = {
            0: {"copy": {"applicable": True, "cost_ns": 100}},
            1: {"stay": {"applicable": True, "cost_ns": 100}},
        }
        # 旧键 (cost, action_rank, instance)：stay(rank0) 胜 ⇒ (100,1,stay)
        # 新键 (cost, instance, action)：instance 0 胜 ⇒ (100,0,copy)
        self.assertEqual(_min_cost(candidates, ("stay", "copy")),
                         (100, 0, "copy"))


class QuotaAdmissibleFlagTest(unittest.TestCase):
    """K7（P2-11）：quota_admissible 列派生（配额侧结构可行域）。"""

    def test_quota_blocked_counts_as_structurally_feasible(self):
        from domain_metrics import _quota_admissible_flag
        per_instance = {
            "remote-read": {
                "applicable": False,
                "reason": "quota_link: link=(0,1) remaining=0"},
            "stay": {
                "applicable": False,
                "reason": "history not resident at target"},
        }
        self.assertEqual(_quota_admissible_flag(per_instance), 1)

    def test_structural_only_block_is_zero(self):
        from domain_metrics import _quota_admissible_flag
        per_instance = {
            "stay": {
                "applicable": False,
                "reason": "history not resident at target"},
        }
        self.assertEqual(_quota_admissible_flag(per_instance), 0)

    def test_plain_applicable_is_one(self):
        from domain_metrics import _quota_admissible_flag
        self.assertEqual(
            _quota_admissible_flag({"stay": {"applicable": True}}), 1)


# ============================================== 7. K6（P1-③ 图侧）==

_LAYERS_K6 = 24   # 3 chunk × COPY_HANDOFF_CHUNK_LAYERS(8)
SESSION_K6 = "sess-k6"
REQUEST_K6 = "req-k6"


def _graph_config_k6():
    config = _graph_config()
    config.layers = _LAYERS_K6
    return config


def _handoff_transfer_k6(chunk_index, layer_start, layer_end, *, bytes_):
    return KVTransfer(
        kind="noc_migrate",
        phase="history",
        reason=("history_prefix_working_copy" if chunk_index == 0
                else "history_prefix_handoff_tail"),
        session_id=SESSION_K6,
        trigger_request_id=REQUEST_K6,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=2 * bytes_,
        shards=tuple(
            KVTransferShard(
                source_rank=source, target_rank=source + 2, edge_rank=None,
                bytes=bytes_, noc_path=(source, source + 2),
                layer_start=layer_start, layer_end=layer_end)
            for source in HOME_RANKS),
        model_layers=_LAYERS_K6,
        layer_start=layer_start,
        layer_end=layer_end,
        resident_prefix_layers_before=layer_start,
        resident_prefix_layers_after=layer_end,
        handoff_chunk=chunk_index,
    )


def _admission_plan_k6():
    chunks = plan_copy_handoff_layer_chunks(_LAYERS_K6)
    assert len(chunks) == 3, "fixture must exercise M>=3 (K6 缺陷形态)"
    return {
        "request_id": REQUEST_K6,
        "session_id": SESSION_K6,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 1,
        "decode_instance_index": 1,
        "admission_time_ns": 1000,
        "history_location_before": None,
        "history_transfer": None,
        "history_transfers": [
            _handoff_transfer_k6(index, start, end, bytes_=400)
            for index, (start, end) in enumerate(chunks)],
        "history_evictions": [],
        "prefill_evictions": [],
        "history_tokens_before": 10,
        "prefill_context_tokens": 300,
        "joint_action": "copy",
        "completion_evictions": [],
        "merge_transfers": [],
    }


def _train_plan_k6(train_id, spans):
    member = _admission_plan_k6()
    return {
        "train_id": train_id,
        "instance_index": 1,
        "stage": "prefill",
        "joiners": [],
        "members": [],
        "pass_spans": list(spans),
        "iterations": len(spans),
        "prefill_chunk_tokens": [(REQUEST_K6, span[0]) for span in spans],
        "prefill_start_member": member,
        "first_chunk_member": member,
        "drain_members": [],
        "exit_members": [],
        "head_request_id": REQUEST_K6,
    }


class CopyLayerSegmentGatingTest(unittest.TestCase):
    """M=3（24 层生产形态）：首体块计算按尾块层区间分段，段 i 只等
    尾块 i 的 recv——尾块 2/3 的层计算不再早于其到达（修前体块 1 的
    全层聚合 pass 只等尾块 1）。"""

    def setUp(self):
        self.builder = GraphBatchBuilder(_graph_config_k6())
        self.builder.begin_batch()
        # fork frontier + 准入批发射（尾块支链）。
        self.builder.emit_iteration_train(
            _train_plan_k6("batch_train_i1_1", [(1, 101)]))
        self.builder.emit_admission_batch(_admission_plan_k6())
        self.arm_layers = self.builder._copy_handoff_layers[REQUEST_K6]

    def _snapshot_recv_nodes(self):
        """体发射消费 arms 账本——发射前快照 recv 门节点。"""
        arms = self.builder._copy_handoff_arms[REQUEST_K6]
        return {
            chunk_index: dict(per_rank)
            for chunk_index, per_rank in arms.items()}

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _edges(self):
        return self.builder.batch["parent_edges"]

    def test_layers_ledger_and_segment_consumption(self):
        self.assertEqual(
            self.arm_layers,
            {1: (8, 16), 2: (16, 24)})
        self.builder.emit_iteration_train(
            _train_plan_k6(
                "batch_train_i1_2", [(1, 201), (1, 202), (1, 203)]))
        # 消费后账本弹出（层区间账本随 arms 同步）。
        self.assertNotIn(REQUEST_K6, self.builder._copy_handoff_arms)
        self.assertNotIn(REQUEST_K6, self.builder._copy_handoff_layers)

    def test_first_body_block_segments_gate_on_all_tails(self):
        recv_nodes = self._snapshot_recv_nodes()
        self.builder.emit_iteration_train(
            _train_plan_k6(
                "batch_train_i1_2", [(1, 201), (1, 202), (1, 203)]))
        edges = self._edges()
        # 尾块 2 的 recv 门必须 arm 进本列车体节点（修前体块 1 只等
        # 尾块 1 ⇒ 尾块 2 门不进任何体节点）。
        for rank in EXEC_RANKS:
            recv = recv_nodes[2][rank]
            targets = {edge["to"] for edge in edges
                       if edge["from"] == recv and edge["rank"] == rank}
            body = {node["id"] for node in self._nodes()
                    if node["request_id"] == "batch_train_i1_2"}
            self.assertTrue(
                targets & body,
                f"tail-2 recv gate not armed into the train body on "
                f"rank {rank} (K6 layer-segment gating missing)")

    def test_first_body_block_segments_gate_on_tail_one(self):
        """尾块 1 门也进体节点（段 [8,16) 的计算等尾块 1）——与尾块 2
        分段独立（层段化后两门分别 gate 各自层段，而非同挂一个聚合
        pass）。"""
        recv_nodes = self._snapshot_recv_nodes()
        self.builder.emit_iteration_train(
            _train_plan_k6(
                "batch_train_i1_2", [(1, 201), (1, 202), (1, 203)]))
        edges = self._edges()
        body = {node["id"] for node in self._nodes()
                if node["request_id"] == "batch_train_i1_2"}
        for chunk_index in (1, 2):
            for rank in EXEC_RANKS:
                recv = recv_nodes[chunk_index][rank]
                targets = {edge["to"] for edge in edges
                           if edge["from"] == recv and edge["rank"] == rank}
                self.assertTrue(
                    targets & body,
                    f"tail-{chunk_index} recv gate not armed into the "
                    f"train body on rank {rank}")


# ================================================ 8. K8 输入加固 ==

class PredictorNaNRejectTest(unittest.TestCase):
    """K8：NaN/inf 数值字段以 EventRecursionError 拒收（不逃 int() 的
    裸 ValueError）。"""

    def test_restore_leg_nan_bytes(self):
        with self.assertRaisesRegex(EventRecursionError, "finite"):
            RestoreGroupLeg(
                layer_start=0, layer_end=1,
                bytes_by_rank=(float("nan"),),
                path_by_rank=(("pool:0", "port:0"),))

    def test_restore_leg_inf_startup(self):
        with self.assertRaisesRegex(EventRecursionError, "startup"):
            RestoreGroupLeg(
                layer_start=0, layer_end=1, bytes_by_rank=(100,),
                path_by_rank=(("pool:0", "port:0"),),
                startup_ns=float("inf"))

    def test_writeback_victim_nan_bytes(self):
        from joint.event_recursion_predictor import WritebackVictim
        with self.assertRaisesRegex(EventRecursionError, "finite"):
            WritebackVictim(
                victim_id="v", bytes_by_rank=(float("nan"),),
                path_by_rank=(("pool:0",),))


class SatisfiedZeroVectorTest(unittest.TestCase):
    """K8：零缺口早退回等长零向量（zip 截断隐患封死）。"""

    def test_zero_gap_returns_equal_length_zero_vector(self):
        policy = LayerEvictionPolicy(mode="adaptive", model_layers=4)
        plan = policy.plan_release(
            gap_bytes_by_tp_rank=(0, 0, 0),
            victims=())
        self.assertTrue(plan.satisfied)
        self.assertEqual(plan.released_bytes_by_tp_rank, (0, 0, 0))
        self.assertEqual(
            len(plan.released_bytes_by_tp_rank),
            len(plan.gap_bytes_by_tp_rank))


class ColdStartAfterAcceptTest(unittest.TestCase):
    """K8：cold_start 只在接受样本后翻转（被拒样本不提前结束冷启动）。"""

    def test_rejected_sample_keeps_cold_start(self):
        group = ServiceFactorGroup()
        group.observe_valid_service(
            "prefill", observed_ratio=float("nan"),
            completion_ns=1000, service_duration_ns=100)
        self.assertTrue(group.cold_start)
        group.observe_valid_service(
            "prefill", observed_ratio=1.5,
            completion_ns=1000, service_duration_ns=100)
        self.assertFalse(group.cold_start)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
