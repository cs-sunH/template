#!/usr/bin/env python3
"""test_r5_three_position_source_union_pin.py -- 修复轮三 R5 钉测：
P=recompute 全量重建副本转记 boundary_source 后的三位置结算守恒
（2026-09-27）。

背景（probe_20260927T213113Z / session_3_request_1__P__recompute 臂红，
delivery_sequence=173）：P=recompute 在 i_P 全量重建 H_B（base 驻留
第三方实例不动）→ D 边界 remote-read@j≠i_P 把 i_P 旧副本（物理持有
base＋U 完整 token 域）转记 boundary_source → 轮末
``_merge_back_three_position`` 直和结算把 base 位置与"已覆盖 base 域
的来源"字节相加（probe 实测：11837＋18271＋1 = 30109 token ≠ 全史
18272 token，多出恰一个 base）→ 守恒断言 fail-closed：

``RuntimeError: three-position merge settlement conservation failed ...
(positions [(4, (1163624448, ...)), (6, (1796112384, ...)),
(1, (98304, ...))], base_history_tokens=11837,
source_history_tokens=6434, new_tokens=1)``

R4 钉测（test_r4_boundary_fold_kv_pin）与 sweep（stageR4/
repro_kv_drift.py）都只走到边界事务 + 审计，**从未跑轮末 merge 结算**
——本钉测补全链：base@第三方实例 + P=recompute@i_P + D
recompute/remote-read@j（异地）+ decode 增长 + 三位置 merge 结算。

被钉语义（修复后）：
  1. 来源侧完整并集（i_P 旧副本 = P=recompute 重建，token 域 ⊇ base）
     结算：base 重复副本零载荷释放、不并入 winner 侧、不计入容量需
     求；胜者 = 保留量最大者（重建来源，或 decode 增长超过来源后的
     exec）恰持全史字节，其余位置归零；
  2. ``账本字节 ≡ 物理字节``全程成立：merge 后全量 strict 审计
     （_check_invariants）与增量通道
     （_check_invariants_after_mutation）双绿，会话 base_*/boundary_
     source_*/working_kind 全清、boundary_source_complete_union 复位；
  3. D=recompute@j（complete_union，重建副本在 exec 侧）回归锚：base
     与来源两处重复副本都零载荷释放，胜者恰持全史。

修复面：face_scheduler.py ``commit_boundary_decode_transition``（来源
转记分支登记 ``boundary_source_complete_union``）＋
``_merge_back_three_position``（重复副本零载荷释放、capacity/winner
口径剔除）。守恒断言本身未动。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_r5_three_position_source_union_pin.py -q
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
    kv_cache_shard_bytes_for_tokens,
)

# probe 等比缩放（probe: base 11837 / U 6434 / tp6；fixture: 96/16/tp2，
# 三实例分离——base@2 第三方、i_P=0、j=1，与 probe 的 base@4/i_P@6/
# j@1 同构）。
HISTORY_TOKENS = 96   # turn-0 终态上下文 = base 历史
INPUT_TOKENS = 16     # turn-1 prefill 输入（P=recompute 重建 H_B=112）
HOME_INSTANCE = 2     # base 驻留（第三方实例）
P_INSTANCE = 0        # turn-1 P=recompute 执行实例（全量重建侧）
THIRD_INSTANCE = 1    # D 边界异地目标 j


def _model():
    return FaceModel(
        layers=2, hidden_size=64, ffn_size=128, num_heads=8,
        vocab_size=256, bytes_per_elem=2, mlp_variant="gelu")


def _hardware():
    return FaceHardware(
        mesh_rows=2, mesh_cols=3,
        local_hbm_capacity_bytes=10 ** 9,
        local_hbm_bandwidth_gbps=100.0, d2d_bandwidth_gbps=200.0,
        peak_perf_tflops=1.0, d2d_latency_ns=0, local_hbm_latency_ns=0)


def _topology(hardware):
    return build_instances(hardware, (
        FaceInstanceSpec("ins0", "1", (0, 3)),
        FaceInstanceSpec("ins1", "2", (1, 4)),
        FaceInstanceSpec("ins2", "3", (2, 5)),
    ))


def _instance_bytes(kv, instance_index):
    return tuple(
        kv._rank_states[rank].kv_cache_bytes
        for rank in kv.topology.instance(instance_index).ranks)


def _p_recompute_kv():
    """turn-0 基础建立于第三方实例 → turn-1 P=recompute（完整 H+U 重
    放）后的工作副本状态：base@2（96 token）、H_B@0（112 token）。"""
    kv = KVCacheManager(_topology(_hardware()), _model())
    kv.prepare_prefill(
        session_id="s", target_instance_index=HOME_INSTANCE,
        history_tokens=0, trigger_request_id="seed")
    kv.expand_prefill(
        session_id="s", instance_index=HOME_INSTANCE,
        context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
    kv.mark_complete("s", 100)
    kv.prepare_prefill(
        session_id="s", target_instance_index=P_INSTANCE,
        history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
        action="recompute")
    kv.expand_prefill(
        session_id="s", instance_index=P_INSTANCE,
        context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
        trigger_request_id="r1")
    return kv


def _drive_to_merge(kv, *, action, growth_tokens):
    """D 边界异地切换 + decode 增长 + 轮末三位置 merge（probe 全链）。"""
    final_tokens = HISTORY_TOKENS + INPUT_TOKENS
    kv.commit_boundary_decode_transition(
        session_id="s", p_instance_index=P_INSTANCE,
        target_instance_index=THIRD_INSTANCE, action=action,
        history_tokens=final_tokens, trigger_request_id="r1")
    session = kv._sessions["s"]
    # 修复面 1：P=recompute 旧副本转记来源时必须登记完整并集标志
    # （getattr 兼容：修复前面无此字段，钉测须走到 merge 撞守恒 raise
    # ——红在真根因，不红在探针自身）。
    if action in ("recompute", "remote-read"):
        self_union = getattr(
            session, "boundary_source_complete_union", False)
    if action == "recompute":
        # D 重建副本按绝对口径物化至 H_B＋增长（sh30
        # _joint_working_context 的 joint_decode_rebuild_tokens 分支）。
        materialized = final_tokens + growth_tokens
    elif action == "remote-read":
        # 多源远读工作副本零化起步，仅承载 decode 增量。
        materialized = growth_tokens
    else:
        materialized = None
    if materialized is not None and (
            action != "remote-read" or growth_tokens):
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=materialized,
            trigger_request_id="r1")
    kv.merge_back(
        session_id="s", trigger_request_id="r1", new_tokens=growth_tokens)
    return session


class SourceUnionThreePositionMergePin(unittest.TestCase):
    """probe 红同型钉测：P=recompute → D remote-read@异地 → merge 三位
    置结算——base 重复副本零载荷释放，胜者恰持全史（修复前守恒炸）。"""

    def test_merge_settles_conservation_with_duplicate_base_release(self):
        kv = _p_recompute_kv()
        session = _drive_to_merge(
            kv, action="remote-read", growth_tokens=1)
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + 1
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        outcome = kv.last_merge_outcome
        self.assertTrue(outcome["three_position"])
        # 保留量最大者（重建来源 = i_P）为胜者；exec 增量整份搬入。
        self.assertEqual(outcome["winner_instance"], P_INSTANCE)
        self.assertEqual(outcome["loser_instances"], (THIRD_INSTANCE,))
        # 修复面 2：胜者侧恰持全史字节（修复前直和 = base 计两遍 →
        # conservation raise）。
        self.assertEqual(
            _instance_bytes(kv, P_INSTANCE),
            tuple(int(v) for v in full_shards))
        # base 重复副本与 exec 增量侧都归零（无 stranded 字节）。
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, THIRD_INSTANCE), (0, 0))
        # 账本终态：会话字段全清、上下文 = 全史、字节 = 账本真值。
        self.assertEqual(session.context_tokens, total_tokens)
        self.assertEqual(
            tuple(int(v) for v in session.shard_bytes),
            tuple(int(v) for v in full_shards))
        self.assertIsNone(session.working_kind)
        self.assertIsNone(session.base_instance_index)
        self.assertIsNone(session.boundary_source_instance_index)
        self.assertFalse(session.boundary_source_complete_union)
        # 全程"账本字节 ≡ 物理字节"：全量 strict + 增量通道双绿。
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))

    def test_merge_capacity_demand_excludes_duplicate_base(self):
        """重复副本不计入容量需求：胜者只需接纳 exec 增量（probe 形态
        的隐性收益——修复前 required 含整份 base，可能诱发无谓逐出）。"""
        kv = _p_recompute_kv()
        _drive_to_merge(kv, action="remote-read", growth_tokens=1)
        fallback = kv.last_merge_outcome["fallback_sequence"]
        committed = [step for step in fallback
                     if step["outcome"] == "committed"]
        self.assertEqual(len(committed), 1)
        increment = kv_cache_shard_bytes_for_tokens(
            kv.model, 1, kv.tp_degree)
        self.assertEqual(
            tuple(int(v) for v in committed[0]["required"]),
            tuple(int(v) for v in increment))

    def test_merge_with_exec_growth_beyond_source(self):
        """decode 增长超过来源保留量后胜者翻到 exec 侧——同一守恒式在
        两种胜者选择下都成立（式(21) 保留量序回归锚）。"""
        kv = _p_recompute_kv()
        growth = HISTORY_TOKENS + INPUT_TOKENS + 8
        session = _drive_to_merge(
            kv, action="remote-read", growth_tokens=growth)
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + growth
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        outcome = kv.last_merge_outcome
        self.assertEqual(outcome["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(
            _instance_bytes(kv, THIRD_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        self.assertEqual(session.context_tokens, total_tokens)
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


class CompleteUnionExecRebuildRegressionPin(unittest.TestCase):
    """回归锚：D=recompute@异地（complete_union，重建副本在 exec 侧）
    依旧绿——base 与来源两处重复副本零载荷释放，胜者恰持全史。"""

    def test_recompute_elsewhere_merge_releases_both_duplicates(self):
        kv = _p_recompute_kv()
        session = _drive_to_merge(
            kv, action="recompute", growth_tokens=1)
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + 1
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        outcome = kv.last_merge_outcome
        self.assertTrue(outcome["zero_byte_flip"])
        self.assertEqual(outcome["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(
            sorted(outcome["loser_instances"]),
            sorted([HOME_INSTANCE, P_INSTANCE]))
        self.assertEqual(
            _instance_bytes(kv, THIRD_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


class IncrementalSourceRegressionPin(unittest.TestCase):
    """回归锚：增量型来源（P=remote-read 的 U 残留 i_P，不覆盖 base 域）
    三位置直和结算语义不变——base＋source＋exec 直和恰为全史。"""

    def test_incremental_remote_read_source_still_additive(self):
        kv = KVCacheManager(_topology(_hardware()), _model())
        kv.prepare_prefill(
            session_id="s", target_instance_index=HOME_INSTANCE,
            history_tokens=0, trigger_request_id="seed")
        kv.expand_prefill(
            session_id="s", instance_index=HOME_INSTANCE,
            context_tokens=HISTORY_TOKENS, trigger_request_id="seed")
        kv.mark_complete("s", 100)
        # P=remote-read（LOCAL 基）：基础留 home，工作副本仅增量。
        kv.prepare_prefill(
            session_id="s", target_instance_index=P_INSTANCE,
            history_tokens=HISTORY_TOKENS, trigger_request_id="r1",
            action="remote-read")
        kv.expand_prefill(
            session_id="s", instance_index=P_INSTANCE,
            context_tokens=INPUT_TOKENS, trigger_request_id="r1")
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="recompute",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1")
        session = kv._sessions["s"]
        # 增量型旧副本（U 段）不得登记完整并集标志。
        self.assertFalse(getattr(
            session, "boundary_source_complete_union", False))
        kv.expand_decode(
            session_id="s", instance_index=THIRD_INSTANCE,
            final_context_tokens=(HISTORY_TOKENS + INPUT_TOKENS + 1),
            trigger_request_id="r1")
        kv.merge_back(session_id="s", trigger_request_id="r1", new_tokens=1)
        total_tokens = HISTORY_TOKENS + INPUT_TOKENS + 1
        full_shards = kv_cache_shard_bytes_for_tokens(
            kv.model, total_tokens, kv.tp_degree)
        outcome = kv.last_merge_outcome
        self.assertEqual(outcome["winner_instance"], THIRD_INSTANCE)
        self.assertEqual(
            _instance_bytes(kv, THIRD_INSTANCE),
            tuple(int(v) for v in full_shards))
        self.assertEqual(_instance_bytes(kv, HOME_INSTANCE), (0, 0))
        self.assertEqual(_instance_bytes(kv, P_INSTANCE), (0, 0))
        kv._check_invariants()
        kv._check_invariants_after_mutation(session_ids=("s",))


if __name__ == "__main__":
    unittest.main()
