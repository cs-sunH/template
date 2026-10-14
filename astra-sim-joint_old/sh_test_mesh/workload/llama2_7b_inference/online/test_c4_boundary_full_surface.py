#!/usr/bin/env python3
"""test_c4_boundary_full_surface.py -- C4 钉测：D 边界完整动作面的执行
语义（v2.0 §19.1 P→D 专项 / D10 静默重建 / D19 多源与按缺失区间 copy）。

覆盖（执行侧 + 计价口径一致性；视图层基础钉测在 joint/test_d19）：
  1. **回 home copy 仅补 U**：增量型边界 copy@home 只搬 U（传输字节 =
     增量区间），home 持完整 H_B 就地折叠本地（merge = stay 零流量）；
  2. **第三目标 copy 补 H+U**：两腿（U 迁移 + H 复制），目标持并集、
     home 保留重复副本（轮末 merge 零字节翻转释放）；
  3. **增量@i_P copy（基材料化）**：H home→i_P 复制、工作副本转完整
     上下文（copy 形态）；
  4. **remote 反复读两个真实来源**：多源远读不迁移 U——读计划来源集
     = {home(H), i_P(U)}，逐列车切片按来源边界排空逐源出 tile（块数
     = steps × Σ源 tile 数；块 shard 源端分属两来源 rank 组）；
  5. **三位置确定性合并**（式 21 运行时）：多源远读末态 {H@home,
     U@i_P, growth@j} → 保留量最大者为目标、其余整份搬入；recompute
     重建副本持完整并集 → 零载荷（重复副本释放）；
  6. **D recompute 静默重建且首 token 只消费一次**：重建 span 前置首
     个 decode 列车（(H_B, H_B) 首 span）、首 token 状态恰一次消费
     （二次消费 fail-closed 由 BoundaryFirstTokenState 构造层钉）；
  7. **JOINT_ALLOWED_ACTIONS=recompute + load-first 全链路**：P
     recompute → 边界 recompute@j → 静默重建 → decode → 完成结算
     （C1 遗留的全链路验收）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_c4_boundary_full_surface.py -q
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

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.test_d18_boundary_decision import (  # noqa: E402
    _complete_boundary,
    _increment_boundary,
)
from joint.test_d17_schema_version_rounding import (  # noqa: E402
    _loads,
    _loads3,
)
from joint.test_joint_shard_pricing import (  # noqa: E402
    _load,
    _model,
    _request,
    _session,
)
from joint.joint_scheduler import select_instance_and_action  # noqa: E402
from joint.test_joint_mechanisms import _cost_model  # noqa: E402
from online.test_decision_route_serialization import (  # noqa: E402
    _GraphStub,
    _decision_shell,
    _make_runtime,
)
from online.test_joint_arrival_contract import (  # noqa: E402
    R0,
    SERVICE_DONE_NS,
)

HISTORY_TOKENS = 96        # 增量型基础区间 H（token）
INPUT_TOKENS = 16          # 增量型 U（token）
HOME_INSTANCE = 2          # 增量型权威基础驻留实例（三实例几何）
P_INSTANCE = 0             # i_P（P = remote-read 执行实例）
THIRD_INSTANCE = 1         # 第三目标


def _three_instance_topology(hardware):
    """三实例几何（home=2 / i_P=0 / 第三目标=1；2×3 mesh 6 rank，
    逐实例一列 × 两 rank = 轴对齐矩形）。"""
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _three_hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _instance_bytes(kv, instance_index):
    return tuple(
        kv._rank_states[rank].kv_cache_bytes
        for rank in kv.topology.instance(instance_index).ranks)


def _increment_kv():
    """三实例真 KVCacheManager + 增量型会话（P = remote-read@i_P：
    U 在 i_P、H 在 home）——C4 边界事务的执行基底。"""
    hardware = _three_hardware()
    model = _tiny_model()
    topology = _three_instance_topology(hardware)
    kv = KVCacheManager(topology, model)
    # 基础历史先建立在 home（turn-0 语义），完成后作为下一轮基础。
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    kv.mark_complete("s", 100)
    # P = remote-read@i_P：工作副本零化（仅增量物化）。
    kv.prepare_prefill(
        session_id="s", target_instance_index=P_INSTANCE,
        history_tokens=HISTORY_TOKENS, trigger_request_id="r0",
        action="remote-read")
    kv.expand_prefill(
        session_id="s", instance_index=P_INSTANCE,
        context_tokens=INPUT_TOKENS, trigger_request_id="r0")
    return kv, model, topology


def _full_hb_bytes(model, kv):
    return kv_cache_shard_bytes_for_tokens(
        model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)


def _u_bytes(model, kv):
    return kv_cache_shard_bytes_for_tokens(
        model, INPUT_TOKENS, kv.tp_degree)


def _h_bytes(model, kv):
    return kv_cache_shard_bytes_for_tokens(
        model, HISTORY_TOKENS, kv.tp_degree)


# ------------------------------------------------------ 边界事务（KV 账本） --


class C4HomeCopyFillsOnlyUTest(unittest.TestCase):
    """§19.1：回 home copy 仅补 U（执行侧）。"""

    def test_home_copy_moves_u_and_folds_local(self):
        kv, model, _topology = _increment_kv()
        transfer, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "increment_copy_home_u_only")
        # 只搬 U：传输字节 = 增量区间（非 H_B 全量）。
        self.assertEqual(transfer.kind, "noc_migrate")
        self.assertEqual(transfer.total_bytes, sum(_u_bytes(model, kv)))
        self.assertLess(
            transfer.total_bytes, sum(_full_hb_bytes(model, kv)))
        # home 持完整 H_B、就地折叠本地（working_kind 清空，基础记录
        # 并入主记录——下一轮 stay@home 直达）。
        session = kv._sessions["s"]
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertIsNone(session.working_kind)
        self.assertEqual(
            session.context_tokens, HISTORY_TOKENS + INPUT_TOKENS)
        self.assertEqual(
            session.shard_bytes, _full_hb_bytes(model, kv))
        self.assertIsNone(session.base_instance_index)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(v) for v in _full_hb_bytes(model, kv)))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        # 轮末 merge = stay（本地提交，零流量）。
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0", new_tokens=8)
        self.assertEqual(transfers, ())
        self.assertEqual(kv.last_merge_outcome["direction"], "stay")


class C4ThirdTargetCopyCoversHandUTest(unittest.TestCase):
    """§19.1：第三目标 copy 补 H+U（两腿逐区间搬运）。"""

    def test_third_copy_two_legs_and_union_at_target(self):
        kv, model, _topology = _increment_kv()
        primary, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(
            outcome["kind"], "increment_copy_third_h_plus_u")
        self.assertEqual(outcome["sources"], (HOME_INSTANCE, P_INSTANCE))
        # 两腿：U 迁移（i_P→j）+ H 复制（home→j）；总字节 = H+U。
        self.assertIsInstance(primary, tuple)
        self.assertEqual(len(primary), 2)
        leg_bytes = sum(leg.total_bytes for leg in primary)
        self.assertEqual(leg_bytes, sum(_full_hb_bytes(model, kv)))
        by_source = {
            leg.source_instance_index: leg.total_bytes for leg in primary}
        self.assertEqual(
            by_source[P_INSTANCE], sum(_u_bytes(model, kv)))
        self.assertEqual(
            by_source[HOME_INSTANCE], sum(_h_bytes(model, kv)))
        # 目标持完整 H_B 工作副本（copy 形态）；home 保留基础副本。
        session = kv._sessions["s"]
        self.assertEqual(session.instance_index, THIRD_INSTANCE)
        self.assertEqual(session.working_kind, "copy")
        self.assertEqual(
            session.shard_bytes, _full_hb_bytes(model, kv))
        self.assertEqual(
            _instance_bytes(kv, THIRD_INSTANCE),
            tuple(int(v) for v in _full_hb_bytes(model, kv)))
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(v) for v in _h_bytes(model, kv)))
        # 轮末 merge：执行端持并集 ⊇ home 侧 → 零字节翻转（home 释放）；
        # decode 增长先物化（列车核销钩子），new_tokens = 输入 + decode。
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS + 8,
            trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + 8)
        self.assertEqual(transfers, ())
        outcome_row = kv.last_merge_outcome
        self.assertEqual(outcome_row["direction"], "reverse")
        self.assertTrue(outcome_row["zero_byte_flip"])
        self.assertEqual(outcome_row["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE), (0, 0))
        self.assertEqual(
            _instance_bytes(kv, THIRD_INSTANCE),
            tuple(int(v) for v in kv_cache_shard_bytes_for_tokens(
                model, HISTORY_TOKENS + INPUT_TOKENS + 8, kv.tp_degree)))


class C4BaseMaterializationAtExecTest(unittest.TestCase):
    """增量型@i_P copy：基材料化（H home→i_P，home 保留）。"""

    def test_base_materialize_extends_working_copy(self):
        kv, model, _topology = _increment_kv()
        primary, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=P_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(
            outcome["kind"], "increment_copy_base_materialize")
        # 原地腿（local_hit 零节点）+ 基材料化腿（home→i_P 的 H 复制）。
        self.assertIsInstance(primary, tuple)
        self.assertEqual(len(primary), 2)
        self.assertEqual(primary[0].kind, "local_hit")
        self.assertEqual(
            primary[1].source_instance_index, HOME_INSTANCE)
        self.assertEqual(
            primary[1].total_bytes, sum(_h_bytes(model, kv)))
        session = kv._sessions["s"]
        self.assertEqual(session.instance_index, P_INSTANCE)
        self.assertEqual(session.working_kind, "copy")
        self.assertEqual(
            session.shard_bytes, _full_hb_bytes(model, kv))
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE),
            tuple(int(v) for v in _full_hb_bytes(model, kv)))
        # home 保留基础（重复副本，merge 释放）。
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(v) for v in _h_bytes(model, kv)))


class C4MultiSourceRemoteTest(unittest.TestCase):
    """§19.1：remote 反复读两个真实来源（执行侧：事务 + 读计划 + 切片
    + 三位置合并）。"""

    def _transition(self):
        kv, model, topology = _increment_kv()
        hardware = _three_hardware()
        transfer, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        return kv, model, topology, hardware, transfer, outcome

    def test_no_migration_two_sources_records(self):
        kv, model, _topology, _hw, transfer, outcome = self._transition()
        self.assertEqual(outcome["kind"], "remote_multi_source")
        self.assertEqual(outcome["sources"], (HOME_INSTANCE, P_INSTANCE))
        self.assertEqual(outcome["moved_bytes"], 0)
        self.assertEqual(transfer.kind, "local_hit")
        session = kv._sessions["s"]
        # j 侧工作副本零化（仅 decode 增长物化）；两来源在册。
        self.assertEqual(session.instance_index, THIRD_INSTANCE)
        self.assertEqual(session.working_kind, "remote-read")
        self.assertEqual(session.shard_bytes, (0, 0))
        self.assertEqual(session.context_tokens, 0)
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.boundary_source_instance_index, P_INSTANCE)
        self.assertEqual(
            session.boundary_source_history_tokens, INPUT_TOKENS)
        # 账本守恒：U 留 i_P、H 留 home、j 零字节。
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE),
            tuple(int(v) for v in _u_bytes(model, kv)))
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(v) for v in _h_bytes(model, kv)))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))

    def test_credit_plan_reads_both_sources_per_pass(self):
        kv, model, topology, hardware, _transfer, _outcome = (
            self._transition())
        scheduler = _decision_shell(
            kv, hardware=hardware, model=model, topology=topology)
        runtime = _make_runtime(
            "r0", prefill_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            decode_length=4, history_tokens_before=HISTORY_TOKENS)
        runtime.session_id = "s"
        scheduler.runtime_by_request_id["r0"] = runtime
        plan = scheduler._joint_remote_read_credit_plan(
            runtime, THIRD_INSTANCE)
        self.assertIsNotNone(plan)
        # 两个真实来源：home(H) + i_P(U)，token 并集进 tile 裁决。
        self.assertEqual(
            [source["instance"] for source in plan["sources"]],
            [HOME_INSTANCE, P_INSTANCE])
        self.assertEqual(
            plan["base_tokens"], HISTORY_TOKENS + INPUT_TOKENS)
        self.assertEqual(plan["steps"], 4)
        # 逐遍字节 = 两来源并集（式 18a）。
        per_pass = sum(
            spec[2] for source in plan["sources"]
            for spec in source["specs"])
        self.assertEqual(
            per_pass, sum(_full_hb_bytes(model, kv)))
        self.assertEqual(plan["total_bytes"], per_pass * 4)
        # 切片：每 pass 按来源边界排空逐源出 tile；块 shard 的源端分属
        # 两来源的 rank 组（home ranks {2,5}、i_P ranks {0,3}）。
        runtime.remote_read_credit_plan = plan
        blocks = scheduler._joint_remote_read_slice(runtime, 1)
        source_ranks = {
            tuple(sorted({
                shard.source_rank
                for shard in block.shards if shard.bytes > 0}))
            for block in blocks}
        self.assertIn((0, 3), source_ranks)
        self.assertIn((2, 5), source_ranks)
        # 单 pass 总字节 = 并集（逐块求和守恒）。
        self.assertEqual(
            sum(block.total_bytes for block in blocks),
            sum(_full_hb_bytes(model, kv)))

    def test_three_position_merge_moves_losers_into_retention_max(self):
        kv, model, _topology, _hw, _transfer, _outcome = self._transition()
        # decode 增长物化在 j（4 token）。
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=4, trigger_request_id="r0")
        growth_bytes = kv_cache_shard_bytes_for_tokens(
            model, 4, kv.tp_degree)
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + 4)
        # 保留量：H@home 最大 → 目标 home；U 与 growth 两腿整份搬入。
        outcome_row = kv.last_merge_outcome
        self.assertTrue(outcome_row["three_position"])
        self.assertEqual(outcome_row["winner_instance"], HOME_INSTANCE)
        self.assertEqual(
            outcome_row["loser_instances"], (P_INSTANCE, THIRD_INSTANCE))
        self.assertEqual(
            sum(leg.total_bytes for leg in transfers),
            sum(_u_bytes(model, kv)) + sum(growth_bytes))
        leg_sources = sorted(
            leg.source_instance_index for leg in transfers)
        self.assertEqual(leg_sources, [P_INSTANCE, THIRD_INSTANCE])
        # 终态：全层 LOCAL@home，其余两位置清零；守恒 = H_B+增长全量。
        final = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS + 4, kv.tp_degree)
        session = kv._sessions["s"]
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertEqual(session.home_instance, HOME_INSTANCE)
        self.assertIsNone(session.boundary_source_instance_index)
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE),
                         tuple(int(v) for v in final))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))


class C5D12ThreePositionMergeSurfaceTest(unittest.TestCase):
    """C5/D12 钉测：三位置实体合并的回退与重合持有者语义（式 21 运行时）。"""

    def test_fallback_target_receives_all_holder_positions(self):
        """容量回退：回退目标接收其余全部持有位置（含偏好序更高的最大
        保留量位置）——不得搁浅任何持有者（孤儿字节 = 账本破损）。"""
        kv, model, topology = _increment_kv()
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=4, trigger_request_id="r0")
        # 受保护的活跃 decoy 占满 home 余量（active 不可逐出）→ 保留量
        # 最大者 home 容量失败，按式(21) 偏好序回退到次持有位置。
        per_rank_need = (
            sum(_u_bytes(model, kv))
            + sum(kv_cache_shard_bytes_for_tokens(model, 4, kv.tp_degree))
        ) // kv.tp_degree
        home_rank = topology.instance(HOME_INSTANCE).ranks[0]
        free = kv._rank_states[home_rank].remaining_bytes
        bpt = kv_cache_shard_bytes_for_tokens(model, 1, kv.tp_degree)[0]
        decoy_tokens = max(1, (free - per_rank_need // 2) // bpt)
        kv.prepare_prefill(
            session_id="decoy", target_instance_index=HOME_INSTANCE,
            history_tokens=0, trigger_request_id="d0")
        kv.expand_prefill(
            session_id="decoy", instance_index=HOME_INSTANCE,
            context_tokens=decoy_tokens, trigger_request_id="d0")
        self.assertLess(
            kv._rank_states[home_rank].remaining_bytes, per_rank_need)
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + 4)
        # 回退目标 = 次偏好 i_P（U 侧）；home（最大保留量）与 third 的
        # 字节都整份并入——回退目标同样持完整并集。
        row = kv.last_merge_outcome
        self.assertTrue(row["three_position"])
        self.assertEqual(row["winner_instance"], P_INSTANCE)
        self.assertEqual(row["loser_instances"], (THIRD_INSTANCE,
                                                  HOME_INSTANCE))
        legs = {(leg.source_instance_index, leg.target_instance_index):
                leg.total_bytes for leg in transfers}
        self.assertEqual(
            set(legs), {(HOME_INSTANCE, P_INSTANCE),
                        (THIRD_INSTANCE, P_INSTANCE)})
        self.assertEqual(
            legs[(HOME_INSTANCE, P_INSTANCE)], sum(_h_bytes(model, kv)))
        final = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS + 4, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE), tuple(int(v) for v in final))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))
        # home 侧会话内容清零（只剩 decoy 自身字节）。
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(v) for v in kv_cache_shard_bytes_for_tokens(
                model, decoy_tokens, kv.tp_degree)))

    def test_recompute_at_base_holder_releases_duplicate_zero_payload(self):
        """D 边界 recompute@基础驻留端（增量型 j==home）：重建副本持并
        集，同实例 base 记录是重复副本——零载荷就地释放（式 21"已有完
        整并集者优先"运行时；与计价侧 merge_ns=0 一致，N9 不拦）。"""
        kv, model, _topology = _increment_kv()
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="recompute",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "recompute_rebuild_elsewhere")
        # 重建物化（H_B → H_B＋4 增长）；home 同时挂 base H＋重建副本。
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS + 4,
            trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + 4)
        self.assertEqual(transfers, ())
        row = kv.last_merge_outcome
        self.assertTrue(row["three_position"])
        self.assertTrue(row["zero_byte_flip"])
        self.assertEqual(row["winner_instance"], HOME_INSTANCE)
        # 终态：home = 完整并集（重复 base 副本释放）；i_P 的 U 清零。
        final = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS + 4, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE), tuple(int(v) for v in final))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        session = kv._sessions["s"]
        self.assertIsNone(session.working_kind)
        self.assertIsNone(session.base_instance_index)
        self.assertIsNone(session.boundary_source_instance_index)
        self.assertEqual(session.home_instance, HOME_INSTANCE)
        # 计价一致性：同形态候选（j==base 驻留端）merge 计价为 0。
        candidate = _model(_loads3()).estimate_D_remaining(
            boundary=_increment_boundary(), instance_index=HOME_INSTANCE,
            action="recompute", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.merge_ns, 0)


class C4RecomputeRebuildElsewhereTest(unittest.TestCase):
    """recompute@j≠i_P：工作副本不迁移 + 重建副本持并集（三位置零载荷
    合并）。"""

    def test_rebuild_elsewhere_zero_payload_merge(self):
        kv, model, _topology = _increment_kv()
        transfer, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="recompute",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "recompute_rebuild_elsewhere")
        self.assertEqual(outcome["moved_bytes"], 0)
        session = kv._sessions["s"]
        self.assertEqual(session.instance_index, THIRD_INSTANCE)
        self.assertEqual(session.working_kind, "recompute")
        self.assertEqual(session.shard_bytes, (0, 0))
        # 来源侧：H@home（base）+ U@i_P（boundary source）。
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.boundary_source_instance_index, P_INSTANCE)
        # 静默重建物化 H_B（decode 增长钩子：expand_decode 至 H_B+n）。
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS + 4,
            trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + 4)
        # 重建副本持完整并集 ⊇ 其他位置 → 零载荷（重复副本直接释放）。
        self.assertEqual(transfers, ())
        outcome_row = kv.last_merge_outcome
        self.assertTrue(outcome_row["three_position"])
        self.assertTrue(outcome_row["zero_byte_flip"])
        self.assertEqual(outcome_row["winner_instance"], THIRD_INSTANCE)
        final = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS + 4, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, THIRD_INSTANCE),
            tuple(int(v) for v in final))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))


class C4CompleteFamilyTransitionsTest(unittest.TestCase):
    """完整型（P ∈ {stay,copy,recompute}）边界：recompute/remote 的
    单源形态（base_* 转记录，无第二来源）。"""

    def _complete_kv(self):
        hardware = _three_hardware()
        model = _tiny_model()
        topology = _three_instance_topology(hardware)
        kv = KVCacheManager(topology, model)
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=0, trigger_request_id="r0")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        return kv, model

    def test_complete_recompute_elsewhere_single_source(self):
        kv, model = self._complete_kv()
        hb = HISTORY_TOKENS + INPUT_TOKENS
        transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="recompute",
            history_tokens=hb, trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "recompute_rebuild_elsewhere")
        self.assertEqual(outcome["sources"], (P_INSTANCE,))
        session = kv._sessions["s"]
        # 旧 H_B 副本转 base_*（无第二来源）；j 侧零化。
        self.assertEqual(session.base_instance_index, P_INSTANCE)
        self.assertIsNone(session.boundary_source_instance_index)
        self.assertEqual(session.shard_bytes, (0, 0))
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE),
            tuple(int(v) for v in kv_cache_shard_bytes_for_tokens(
                model, hb, kv.tp_degree)))
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=hb + 4, trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0", new_tokens=4)
        self.assertEqual(transfers, ())
        row = kv.last_merge_outcome
        self.assertTrue(row["zero_byte_flip"])
        self.assertEqual(row["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))

    def test_complete_remote_elsewhere_reads_single_source(self):
        kv, model = self._complete_kv()
        hb = HISTORY_TOKENS + INPUT_TOKENS
        hardware = _three_hardware()
        topology = kv.topology
        transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="remote-read",
            history_tokens=hb, trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "remote_multi_source")
        self.assertEqual(outcome["sources"], (P_INSTANCE,))
        scheduler = _decision_shell(
            kv, hardware=hardware, model=model, topology=topology)
        runtime = _make_runtime(
            "r0", prefill_context_tokens=hb, decode_length=2,
            history_tokens_before=HISTORY_TOKENS)
        runtime.session_id = "s"
        scheduler.runtime_by_request_id["r0"] = runtime
        plan = scheduler._joint_remote_read_credit_plan(
            runtime, THIRD_INSTANCE)
        self.assertIsNotNone(plan)
        self.assertEqual(
            [source["instance"] for source in plan["sources"]],
            [P_INSTANCE])
        self.assertEqual(plan["base_tokens"], hb)
        # 增长物化后两位置 merge（remote 家族：少并多真实传输）。
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=2, trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0", new_tokens=2)
        row = kv.last_merge_outcome
        self.assertIn(row["direction"], ("forward", "reverse"))
        # 终态守恒：胜者持全量、败者清零。
        winner = row["winner_instance"]
        final = kv_cache_shard_bytes_for_tokens(model, hb + 2, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, winner), tuple(int(v) for v in final))
        loser = P_INSTANCE if winner == THIRD_INSTANCE else THIRD_INSTANCE
        self.assertEqual(_instance_bytes(kv, loser), (0, 0))


# ------------------------------------------------------------ 计价口径 --

class C4RecomputePricingTest(unittest.TestCase):
    """recompute@j 计价：H_B 完整静默重建（历史基数 0）+ 正常剩余 D
    前向；工作副本不迁移（零准备腿）。"""

    def test_rebuild_charged_full_history_zero_base(self):
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=1,
            action="recompute", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertFalse(candidate.is_continuation)
        # 零准备腿（工作副本不迁移）；计算段 = 重建（H_B × prefill）+
        # decode（8 步 × 2 ns）；merge 0（重建副本持并集）。
        self.assertEqual(candidate.breakdown.history_prep_ns, 0)
        self.assertEqual(candidate.breakdown.remote_read_ns, 0)
        self.assertEqual(
            candidate.breakdown.compute_ns, 100 * 1 + 16)
        self.assertEqual(
            candidate.cost_ns,
            candidate.breakdown.target_wait_ns
            + candidate.breakdown.compute_ns)
        notes = candidate.breakdown.notes
        self.assertIn("boundary_rebuild_tokens=100", notes)
        self.assertIn("boundary_rebuild_silent=1", notes)
        self.assertIn("boundary_rebuild_history_base=0", notes)
        self.assertIn("boundary_rebuild_ns=100", notes)

    def test_in_place_recompute_reuses_space(self):
        """recompute@i_P（完整型）：驻留旧分配空间复用（D10）——足迹
        新增 = 0（不触发驱逐等待），重建计算照常全额。"""
        model = _model(_loads())
        candidate = model.estimate_D_remaining(
            boundary=_complete_boundary(), instance_index=0,
            action="recompute", remote_enabled=True)
        self.assertTrue(candidate.applicable)
        self.assertEqual(candidate.breakdown.eviction_wait_ns, 0)
        self.assertEqual(candidate.breakdown.history_prep_ns, 0)
        self.assertEqual(
            candidate.breakdown.compute_ns, 100 * 1 + 16)
        self.assertFalse(candidate.is_continuation)


# ---------------------------------------------------- 全链路（recompute 臂） --


class _TrainGraphStub(_GraphStub):
    """发射面 stub：列车 + 完成批（merge 零传输 → 无 merge watch）。"""

    last_train_plan = None
    pending_store_tails = {}

    def emit_iteration_train(self, train_plan):
        self.last_train_plan = train_plan
        return {
            "drain_members": {},
            "exit_members": {
                member["request_id"]: {
                    0: "exit_{}".format(member["request_id"]),
                    1: "exit_{}".format(member["request_id"])}
                for member in train_plan["exit_members"]
            },
            "sentinel_members": {},
            "block_ends": {0: "end_0", 1: "end_1"},
            "eviction_watches": [],
        }

    def emit_completion_batch(self, plan, *, block_end_stage="seg2"):
        return {"merge_done_members": []}


def _recompute_arm_shell(decode_length=8):
    """{recompute}+load-first 臂全链路基底：真 KVCacheManager + 真
    drain/train/finalize/完成链（d18 drain 壳 + 列车发射面）。"""
    hardware = _tiny_hardware()
    model = _tiny_model()
    topology = _two_instance_topology(hardware)
    kv = KVCacheManager(topology, model)
    scheduler = _decision_shell(kv, hardware=hardware, model=model,
                                topology=topology)
    scheduler.graph = _TrainGraphStub()
    scheduler.next_request = {}
    scheduler._runtime_index = {}
    scheduler.runtimes = [None]
    scheduler.completed_requests = 0
    scheduler.config = SimpleNamespace(request_queue=[
        SimpleNamespace(next_trigger_type="human",
                        inter_request_interval_ns=1000)])
    scheduler.train_ledger_rows = []
    scheduler.train_ledger_sink = None
    scheduler._batch["watches"] = []
    scheduler._batch["tick"] = 2000
    scheduler._train_max_iter = 0        # 不截断（列车终点 = 自然边界）
    scheduler._train_instance_index = {}
    scheduler._pending_eviction_watches = {}
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler._stalled_by_instance = {}
    scheduler._pending_merge_alarms = {}
    runtime = _make_runtime("r0", decode_length=decode_length)
    scheduler.runtime_by_request_id["r0"] = runtime
    scheduler.next_request["r0"] = None
    scheduler._runtime_index["r0"] = 0
    scheduler.runtimes.append(runtime)
    kv.reserve_request_capacity(
        request_id="r0", session_id=runtime.session_id, instance_index=0,
        final_context_tokens=runtime.prefill_context_tokens)
    kv.prepare_prefill(
        session_id=runtime.session_id, target_instance_index=0,
        history_tokens=0, trigger_request_id="r0")
    kv.expand_prefill(
        session_id=runtime.session_id, instance_index=0,
        context_tokens=runtime.prefill_context_tokens,
        trigger_request_id="r0")
    runtime.prefill_instance_index = 0
    runtime.joint_stage = "P_RUNNING"
    scheduler.instances[0].qp.append(runtime)
    runtime.queued_chunk_load_ns = 0
    config = parse_joint_config({
        "JOINT_ALLOWED_ACTIONS": "recompute",
        "JOINT_SCHEDULER_MODE": "load-first",
    })
    scheduler.joint_config = config
    scheduler._joint_mode = config.scheduler_mode
    # P 阶段动作按臂语义标注（P 工作副本已在实例 0 物化）。
    runtime.joint_action = "recompute"
    # 实例 0 压在服负载 → 边界 argmin 翻转到实例 1（recompute@j≠i_P）。
    for seq in range(3):
        hog = _make_runtime(f"active0_{seq}")
        scheduler.runtime_by_request_id[f"active0_{seq}"] = hog
        scheduler.instances[0].active_decode.append(hog)
        scheduler.instances[0].active_decode_lookup.add(hog)
    return scheduler, kv, runtime


class C4CompleteRemoteArmTrainTest(unittest.TestCase):
    """完整型 remote@j（多源机制的单源形态）端到端：边界选中后
    decode credit 切片照常逐列车发射（判据 = 边界动作族，非 P 动作
    名——P=stay 的完整型远读同样有 credit 体块门）。"""

    def test_complete_remote_joiner_carries_credit_slices(self):
        scheduler, kv, runtime = _recompute_arm_shell()
        # 允许面掩成 {remote-read, stay}：stay 放行（式 1c 回退腿），
        # 实例 0 在服负载重 → remote@1（读 i_P 的 H_B，无迁移）胜出。
        config = parse_joint_config({
            "JOINT_ALLOWED_ACTIONS": "remote-read,stay",
            "JOINT_SCHEDULER_MODE": "load-first",
        })
        scheduler.joint_config = config
        scheduler._joint_mode = config.scheduler_mode
        runtime.joint_action = "stay"
        scheduler._on_prefill_drain(runtime.request_id, 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertEqual(runtime.joint_decode_action, "remote-read")
        self.assertEqual(
            runtime.boundary_decision["boundary_transition"]["kind"],
            "remote_multi_source")
        # 会话：旧 H_B 副本留实例 0（base 记录）、j 工作副本零化。
        session = kv._sessions[runtime.session_id]
        self.assertEqual(session.base_instance_index, 0)
        self.assertEqual(session.shard_bytes, (0, 0))
        self.assertTrue(runtime.joint_remote_growth_only)
        scheduler._plan_and_emit_trains(2000)
        self.assertEqual(runtime.joint_stage, "D_RUNNING")
        # credit 切片在册：首列车块 1 进 joiner 主链槽位、逐块门控。
        self.assertEqual(runtime.remote_read_credit_trains, 1)
        summary = runtime.remote_read_slice_summaries[0]
        self.assertGreater(summary["block_count"], 0)
        train = scheduler.instances[1].in_flight_train
        self.assertNotIn("r0", train["rebuild_members"])
        # 首 token 恰一次消费（首个 D 前向排定）。
        self.assertTrue(runtime.joint_boundary_first_token["consumed"])


class C4RecomputeArmFullChainTest(unittest.TestCase):
    """JOINT_ALLOWED_ACTIONS=recompute + load-first 全链路：P recompute
    → 边界 recompute@j → 静默重建 → decode → 完成结算。"""

    def test_full_chain_recompute_arm(self):
        scheduler, kv, runtime = _recompute_arm_shell()
        hb = runtime.history_tokens_before + runtime.joint_input_tokens
        # ---- P 腿：真实选择器（load-first + {recompute} 掩码）。----
        config = scheduler.joint_config
        p_model = _cost_model(
            {0: _load(0, 100), 1: _load(1, 5000)},
            allowed_actions=config.allowed_action_set)
        p = select_instance_and_action(
            mode=config.scheduler_mode, cost_model=p_model,
            session=_session(home=0, resident=1), request=_request(),
            remote_enabled=True)
        self.assertEqual(
            (p.chosen.instance_index, p.chosen.action), (0, "recompute"))
        # ---- 边界腿：真 drain → recompute@1（实例 0 在服负载重）。----
        scheduler._on_prefill_drain(runtime.request_id, 1000)
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(runtime.decode_instance_index, 1)
        self.assertEqual(runtime.joint_decode_action, "recompute")
        self.assertFalse(runtime.boundary_decision["fallback"])
        self.assertEqual(
            runtime.boundary_decision["boundary_transition"]["kind"],
            "recompute_rebuild_elsewhere")
        self.assertEqual(runtime.joint_decode_rebuild_tokens, hb)
        self.assertTrue(runtime.joint_decode_rebuild_pending)
        self.assertTrue(runtime.joint_suppress_sampling)
        self.assertTrue(runtime.joint_suppress_output)
        first_token = runtime.joint_boundary_first_token
        self.assertTrue(first_token["issued"])
        self.assertFalse(first_token["consumed"])
        self.assertEqual(first_token["rng_advances"], 1)
        # 会话：旧 H_B 副本留实例 0（base 记录）、实例 1 工作副本零化。
        session = kv._sessions[runtime.session_id]
        self.assertEqual(session.instance_index, 1)
        self.assertEqual(session.base_instance_index, 0)
        self.assertEqual(session.shard_bytes, (0, 0))
        # ---- 列车腿：静默重建 span 前置 + 首 token 恰一次消费。----
        scheduler._plan_and_emit_trains(2000)
        self.assertEqual(runtime.joint_stage, "D_RUNNING")
        train = scheduler.instances[1].in_flight_train
        self.assertIsNotNone(train)
        # 重建 span = 成员首 span（(H_B, H_B)——工作量与上下文终点）。
        self.assertEqual(train["pass_spans"][0], (hb, hb))
        decode_spans = train["pass_spans"][1:]
        self.assertEqual(
            decode_spans,
            [(1, runtime.prefill_context_tokens + step)
             for step in range(1, runtime.decode_length + 1)])
        self.assertFalse(runtime.joint_decode_rebuild_pending)
        # 首 token 恰一次消费（首个 D 前向排定；重复消费被状态守卫拦）。
        self.assertTrue(runtime.joint_boundary_first_token["consumed"])
        self.assertEqual(runtime.joint_boundary_first_token["rng_advances"], 1)
        scheduler._consume_boundary_first_token(runtime)
        self.assertTrue(runtime.joint_boundary_first_token["consumed"])
        # ---- 完成腿：核销 → 重建物化 + 预约注销 → merge → DONE。----
        scheduler._finalize_completed_trains(
            drained=(), completed_now=("r0",), sentinel_trains=(),
            tick=2100)
        # 核销点：重建副本已物化在实例 1（H_B + decode 增长全量）、旧
        # 副本仍在实例 0（零载荷释放发生在 merge）；目标侧预约在重建
        # 物化点注销（"final KV allocation"守卫的因果点）。
        final = kv_cache_shard_bytes_for_tokens(
            scheduler.model,
            runtime.final_context_tokens, kv.tp_degree)
        old_copy = kv_cache_shard_bytes_for_tokens(
            scheduler.model,
            runtime.prefill_context_tokens, kv.tp_degree)
        self.assertEqual(_instance_bytes(kv, 1),
                         tuple(int(v) for v in final))
        self.assertEqual(_instance_bytes(kv, 0),
                         tuple(int(v) for v in old_copy))
        self.assertFalse(runtime.joint_rebuild_reservation_hold)
        self.assertNotIn("r0", kv._reservations)
        # 完成结算（生产由交付处理器在核销后调用——exit 标记核销 +
        # 完成归属实例一致）。
        scheduler._complete_requests(("r0",), 2100)
        self.assertTrue(runtime.completed)
        self.assertEqual(runtime.joint_stage, "SETTLE_DONE")
        self.assertEqual(runtime.decode_tokens_consumed,
                         runtime.decode_length)
        self.assertEqual(session.instance_index, 1)
        self.assertIsNone(session.working_kind)
        # merge 披露：零字节翻转（重建副本持并集；旧副本释放）；终轮
        # 无后续请求 → 会话 KV 随 retire 释放（两实例清零）。
        self.assertTrue(runtime.merge_outcome["zero_byte_flip"])
        self.assertEqual(runtime.merge_outcome["winner_instance"], 1)
        self.assertEqual(_instance_bytes(kv, 0), (0, 0))
        self.assertEqual(_instance_bytes(kv, 1), (0, 0))
        # 首 token 元数据核验：恰一次采样（rng=1）、已消费、重建期抑制。
        self.assertEqual(
            runtime.plan_dict()["joint_boundary_first_token"],
            {"issued": True, "consumed": True, "rng_advances": 1,
             "rebuild_sampling_suppressed": True})



# --------------------------------------------- G3 冒烟矩阵回归钉（2026-09-27）--

DECODE_TOKENS = 8             # D 增长（token）


def _turn0_local_prefill(kv, tokens, request_id):
    """turn-0 完整型 P（本地 stay 起步，无工作副本）。"""
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id=request_id)
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=tokens, trigger_request_id=request_id)


def _growth_only_remote_after_local_p():
    """G3 真实形态一：完整型 P（本地）→ D 边界多源远读——base 记录 =
    i_P 侧 H_B（**已含本轮 input**），j 侧工作副本仅 decode 增长。"""
    hardware = _three_hardware()
    model = _tiny_model()
    kv = KVCacheManager(_three_instance_topology(hardware), model)
    _turn0_local_prefill(kv, INPUT_TOKENS, "r0")
    _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
        session_id="s", p_instance_index=HOME_INSTANCE,
        target_instance_index=THIRD_INSTANCE, action="remote-read",
        history_tokens=INPUT_TOKENS, trigger_request_id="r0")
    assert outcome["kind"] == "remote_multi_source", outcome
    return kv, model


class G3GrowthOnlyRemoteMergeIncrementDomainTest(unittest.TestCase):
    """迁移族 merge 的 new_tokens 增量域钉测（守恒断言不放松）：

    growth-only 布局下 base 记录已含本轮 input，new_tokens 只能计
    decode 增长——按 input+decode 传值（修复前完成路径的取值）必须在
    守恒断言处 fail-closed；按增量传值守恒闭合、终态全层 LOCAL@胜者。
    """

    def test_growth_only_layout_records_input_in_base(self):
        kv, _model = _growth_only_remote_after_local_p()
        session = kv._sessions["s"]
        self.assertEqual(session.working_kind, "remote-read")
        self.assertEqual(session.instance_index, THIRD_INSTANCE)
        # base 记录 = i_P 侧 H_B：已含本轮 input（U 留来源侧）。
        self.assertEqual(session.base_history_tokens, INPUT_TOKENS)
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        self.assertEqual(session.context_tokens, 0)
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=DECODE_TOKENS, trigger_request_id="r0")
        self.assertEqual(session.context_tokens, DECODE_TOKENS)

    def test_double_counted_new_tokens_fails_closed(self):
        kv, model = _growth_only_remote_after_local_p()
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=DECODE_TOKENS, trigger_request_id="r0")
        with self.assertRaisesRegex(
                RuntimeError, "merge settlement conservation failed"):
            kv.merge_back(
                session_id="s", trigger_request_id="r0",
                new_tokens=INPUT_TOKENS + DECODE_TOKENS)
        del model

    def test_increment_domain_new_tokens_closes_conservation(self):
        kv, model = _growth_only_remote_after_local_p()
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=DECODE_TOKENS, trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=DECODE_TOKENS)
        session = kv._sessions["s"]
        final = kv_cache_shard_bytes_for_tokens(
            model, INPUT_TOKENS + DECODE_TOKENS, kv.tp_degree)
        self.assertEqual(
            session.context_tokens, INPUT_TOKENS + DECODE_TOKENS)
        self.assertEqual(session.shard_bytes, final)
        self.assertIsNone(session.working_kind)
        # 前向腿（base 侧占优）：工作副本整份搬 base 驻留侧。
        self.assertEqual(kv.last_merge_outcome["direction"], "forward")
        self.assertEqual(kv.last_merge_outcome["winner_instance"],
                         HOME_INSTANCE)
        self.assertTrue(transfers)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(value) for value in final))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))
        kv.mark_complete("s", INPUT_TOKENS + DECODE_TOKENS)


class G3CompleteCopyOntoBaseFoldsTest(unittest.TestCase):
    """G3 真实形态二：完整型 copy 的 D 边界迁移落到 base 驻留实例——
    就地并基折叠（R13 消除的组合不得到达轮末 merge）。

    P 阶段 copy 建 C13 逐块交接 journal（home 侧基础轮内释放），D 边
    界再选 copy 回 base 驻留实例：折叠清 working_kind/基础记录、闭合
    journal，轮末 merge = stay 本地提交（不再撞 N9 防御）。
    """

    def _arrange(self):
        hardware = _three_hardware()
        model = _tiny_model()
        kv = KVCacheManager(_three_instance_topology(hardware), model)
        _turn0_local_prefill(kv, HISTORY_TOKENS, "seed")
        kv.mark_complete("s", HISTORY_TOKENS)
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        kv._settle_copy_handoffs(
            "s", boundary="prefill_drain", trigger_request_id="r1")
        return kv, model

    def test_copy_onto_base_folds_and_settles_stay(self):
        kv, model = self._arrange()
        session = kv._sessions["s"]
        self.assertEqual(session.working_kind, "copy")
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        self.assertEqual(outcome["kind"], "complete_copy_base_fold")
        # 折叠后：本地全上下文，无工作副本、无基础记录、journal 闭合。
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.context_tokens, HISTORY_TOKENS + INPUT_TOKENS)
        self.assertIsNone(session.base_instance_index)
        self.assertEqual(session.base_history_tokens, 0)
        self.assertIsNone(session.copy_handoff)
        full = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)
        self.assertEqual(session.shard_bytes, full)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(value) for value in full))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        # 轮末 merge = stay 本地提交（零流量），守恒不经过 N9 防御。
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=(HISTORY_TOKENS + INPUT_TOKENS
                                  + DECODE_TOKENS),
            trigger_request_id="r1")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r1",
            new_tokens=INPUT_TOKENS + DECODE_TOKENS)
        self.assertEqual(transfers, ())
        self.assertEqual(kv.last_merge_outcome["direction"], "stay")
        kv.mark_complete(
            "s", HISTORY_TOKENS + INPUT_TOKENS + DECODE_TOKENS)


class G3RemoteReadOntoBaseFoldsTest(unittest.TestCase):
    """远读（增量型）落到 base 驻留实例的同族退化折叠：U 并基为本地
    全上下文（working_kind 清空），轮末 merge = stay。"""

    def test_remote_onto_base_folds_local(self):
        kv, model, _topology = _increment_kv()
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "remote_read_base_fold")
        session = kv._sessions["s"]
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.context_tokens, HISTORY_TOKENS + INPUT_TOKENS)
        self.assertIsNone(session.base_instance_index)
        full = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)
        self.assertEqual(session.shard_bytes, full)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(value) for value in full))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        kv.expand_decode(
            session_id="s", instance_index=HOME_INSTANCE,
            final_context_tokens=(HISTORY_TOKENS + INPUT_TOKENS
                                  + DECODE_TOKENS),
            trigger_request_id="r0")
        transfers = kv.merge_back(
            session_id="s", trigger_request_id="r0",
            new_tokens=INPUT_TOKENS + DECODE_TOKENS)
        self.assertEqual(transfers, ())
        self.assertEqual(kv.last_merge_outcome["direction"], "stay")


class G3RemoteReadOntoBaseReservationSizingTest(unittest.TestCase):
    """J 仲裁形态钉测：远读落 base 驻留实例的 D 边界折叠，阶段预约必须
    按折叠物理结局 sizing（stay 族整份基数、无跨实例后缀、无远读暂存
    ——折叠后 decode 切回完整上下文族、无读流）。resize 若沿用多源
    远读的零基数口径，折叠驻留立即超出预约，move_prefill_to_decode
    的物化守卫 fail-closed（"reservation is smaller than its resident
    KV"）。"""

    def test_increment_remote_stage_resize_sizes_fold_full_context(self):
        kv, model, _topology = _increment_kv()
        kv.reserve_request_capacity(
            request_id="r0", session_id="s", instance_index=P_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            action="remote-read")
        full = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)
        zero = (0,) * kv.tp_degree
        # 折叠组合（远读落点 = base 驻留实例）：足迹按 stay 族整份基数。
        fold = kv.stage_reservation_footprint(
            session_id="s", target_instance_index=HOME_INSTANCE,
            action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, input_tokens=0)
        self.assertEqual(fold.basis_shard_bytes, full)
        self.assertEqual(fold.suffix_shard_bytes, zero)
        self.assertEqual(fold.staging_shard_bytes, zero)
        self.assertEqual(fold.staging_tile_tokens, 0)
        # 对照：非折叠的多源远读（落第三目标）维持零基数既有口径。
        multi = kv.stage_reservation_footprint(
            session_id="s", target_instance_index=THIRD_INSTANCE,
            action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, input_tokens=0)
        self.assertEqual(multi.basis_shard_bytes, zero)
        # D20 resize → 折叠事务 → 预约注销（无读流、无 lease 转移）。
        kv.commit_stage_reservation_transaction(
            request_id="r0", target_instance_index=HOME_INSTANCE,
            stage_action="remote-read", stage_epoch=1,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, input_tokens=0)
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r0")
        self.assertEqual(outcome["kind"], "remote_read_base_fold")
        reservation = kv._reservations["r0"]
        self.assertEqual(kv._reservation_extra_shards(reservation), zero)
        kv.release_request_capacity_reservation("r0")

    def test_copy_admitted_partial_base_fold_reservation_closes(self):
        # P = joint copy 落 PARTIAL 基（生产失败形态：预约按 copy 族整份
        # 基数准入，C13 逐块交接在 drain 结算），D 边界选远读回 base
        # 驻留实例——resize 与折叠驻留口径配套，物化守卫不触发。
        hardware = _three_hardware()
        model = _tiny_model()
        kv = KVCacheManager(_three_instance_topology(hardware), model)
        kv.prepare_prefill(
            session_id="s", target_instance_index=HOME_INSTANCE,
            history_tokens=0, trigger_request_id="seed")
        kv.expand_prefill(
            session_id="s", instance_index=HOME_INSTANCE,
            context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
        kv.mark_complete("s", 100)
        kv._evict_suffix(
            kv._sessions["s"], phase="completion", reason="fixture_partial",
            trigger_request_id="fixture", layer_start=2)
        kv.reserve_request_capacity(
            request_id="r1", session_id="s", instance_index=P_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            action="copy")
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        kv._settle_copy_handoffs(
            "s", boundary="prefill_drain", trigger_request_id="r1")
        kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=HOME_INSTANCE,
            stage_action="remote-read", stage_epoch=1,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, input_tokens=0)
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        self.assertEqual(outcome["kind"], "remote_read_base_fold")
        reservation = kv._reservations["r1"]
        self.assertEqual(
            kv._reservation_extra_shards(reservation), (0,) * kv.tp_degree)
        kv.release_request_capacity_reservation("r1")
        session = kv._sessions["s"]
        self.assertIsNone(session.working_kind)
        self.assertEqual(session.instance_index, HOME_INSTANCE)
        self.assertEqual(
            session.context_tokens, HISTORY_TOKENS + INPUT_TOKENS)

    def test_copy_round_fold_journal_closes_before_next_round(self):
        # P=copy 轮的交接 journal 随远读折叠闭合——悬挂的陈旧 journal
        # （home 残量恒零）在下一轮跨实例 prepare 时遮蔽 base 驻留贡献，
        # 折叠驻留变孤儿字节（增量期望表 "KV accounting mismatch"）。
        kv, model, _topology = self._folded_after_copy_round()
        session = kv._sessions["s"]
        self.assertIsNone(session.copy_handoff)
        # 完成本轮：stay 早退 merge + 完成记账。
        kv.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=INPUT_TOKENS)
        kv.mark_complete("s", HISTORY_TOKENS + INPUT_TOKENS)
        # 下一轮跨实例事务（remote-read@第三实例）：base 驻留贡献在账。
        kv.reserve_request_capacity(
            request_id="r2", session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            action="remote-read")
        kv.prepare_prefill(
            session_id="s", target_instance_index=THIRD_INSTANCE,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r2", action="remote-read")
        base = kv_cache_shard_bytes_for_tokens(
            model, HISTORY_TOKENS + INPUT_TOKENS, kv.tp_degree)
        self.assertEqual(
            _instance_bytes(kv, HOME_INSTANCE),
            tuple(int(value) for value in base))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))

    @staticmethod
    def _folded_after_copy_round():
        hardware = _three_hardware()
        model = _tiny_model()
        kv = KVCacheManager(_three_instance_topology(hardware), model)
        kv.prepare_prefill(
            session_id="s", target_instance_index=HOME_INSTANCE,
            history_tokens=0, trigger_request_id="seed")
        kv.expand_prefill(
            session_id="s", instance_index=HOME_INSTANCE,
            context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
        kv.mark_complete("s", 100)
        kv._evict_suffix(
            kv._sessions["s"], phase="completion", reason="fixture_partial",
            trigger_request_id="fixture", layer_start=2)
        kv.reserve_request_capacity(
            request_id="r1", session_id="s", instance_index=P_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            action="copy")
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="copy")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        kv._settle_copy_handoffs(
            "s", boundary="prefill_drain", trigger_request_id="r1")
        kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=HOME_INSTANCE,
            stage_action="remote-read", stage_epoch=1,
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, input_tokens=0)
        _transfer, _evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=HOME_INSTANCE, action="remote-read",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        assert outcome["kind"] == "remote_read_base_fold"
        return kv, model, None


class G3CompletionNewTokensDomainTest(unittest.TestCase):
    """完成路径 new_tokens 口径钉（调度器侧）：迁移族按工作副本增量域
    取值（growth-only → 仅 decode；P 相远读 → input+decode），零字节
    翻转族维持 input+decode 披露口径。"""

    @staticmethod
    def _completed_scheduler():
        from online.test_joint_arrival_contract import (
            _bare_scheduler,
        )
        scheduler = _bare_scheduler()
        runtime = scheduler.runtime_by_request_id[R0]
        runtime.joint_input_tokens = INPUT_TOKENS
        runtime.decode_length = DECODE_TOKENS
        runtime.decode_tokens_consumed = DECODE_TOKENS
        return scheduler, runtime

    def _merge_new_tokens(self, scheduler):
        for name, kwargs in scheduler.kv_manager.calls:
            if name == "merge_back":
                return kwargs["new_tokens"]
        raise AssertionError("merge_back was not called")

    def test_growth_only_remote_passes_decode_increment(self):
        scheduler, runtime = self._completed_scheduler()
        runtime.joint_action = "remote-read"
        runtime.joint_decode_action = "remote-read"
        runtime.joint_remote_growth_only = True
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        # 增量域 = decode 增长（base 记录已含 input，不得双计）。
        self.assertEqual(self._merge_new_tokens(scheduler), DECODE_TOKENS)

    def test_plain_remote_passes_input_plus_decode(self):
        scheduler, runtime = self._completed_scheduler()
        runtime.joint_action = "remote-read"
        runtime.joint_decode_action = "remote-read"
        runtime.joint_remote_growth_only = False
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        self.assertEqual(
            self._merge_new_tokens(scheduler),
            INPUT_TOKENS + DECODE_TOKENS)

    def test_flip_family_keeps_disclosure_formula(self):
        scheduler, runtime = self._completed_scheduler()
        runtime.joint_action = "copy"
        runtime.joint_decode_action = "copy"
        scheduler._complete_requests([R0], SERVICE_DONE_NS)
        self.assertEqual(
            self._merge_new_tokens(scheduler),
            INPUT_TOKENS + DECODE_TOKENS)


if __name__ == "__main__":
    unittest.main()
