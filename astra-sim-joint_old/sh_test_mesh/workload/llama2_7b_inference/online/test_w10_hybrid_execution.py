#!/usr/bin/env python3
"""test_w10_hybrid_execution.py -- W10 hybrid 执行面钉测（V10 执行侧）。

覆盖（方案 §7.5/§12.1 W10；主文 §6.7"组合既有两链"）：

* **hybrid 边界事务**（``commit_boundary_decode_transition(action=
  "hybrid", copy_sources=L)``）：i_P ∈ L 的真子集 = U 迁 j（copy 链
  move_prefill_to_decode——容量真实 T+E）＋ home ∈ R 基础留源端
  （remote 链 credit 读流）；终态 = 远读原地继续同族状态（工作副本在
  j、基础在 home）；守恒审计绿；
* **部分复制真实占用持久容量**（不免费）：目标 rank 字节实增 ΣL；
* **端点/形态 fail-closed**：L=∅/L=S（端点须走纯动作）与 L={home}
  （单 span 工作副本域模型外）显式拒绝，不静默换别种混合执行；
* **远读子集读流收敛**：hybrid 后目录来源面（credit 计划的枚举入口）
  = 仅 home 单源（j 缺失区间收敛到单来源——W9 遗留 2 的同族收敛面）；
* **stage 预约足迹**：hybrid 走远读族基数（input/U），暂存分量在场；
* **有效远读源门**（W9 遗留 2 解除路径，§7.5）：重复/被含副本不构成
  第二有效源（subset copy 收敛后单源协议恢复适用），真多源维持门。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_w10_hybrid_execution.py -q
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

HISTORY_TOKENS = 96   # turn-0 基础历史（home 驻留）
INPUT_TOKENS = 16     # turn-1 增量 U（P=remote-read 工作副本）
HOME_INSTANCE = 2
P_INSTANCE = 0
THIRD_INSTANCE = 1


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


def _increment_kv():
    """turn-0 基础 @home → turn-1 P=remote-read @i_P 的增量族状态：
    base H@home（权威）＋工作副本 U@i_P（S = {home, i_P}）。D09 tile
    协议参数注入（生产装配口径——暂存预约真实分量）。"""
    kv = KVCacheManager(
        _topology(_hardware()), _model(),
        remote_read_tile_tokens=64,
        remote_read_kv_token_bytes=kv_cache_shard_bytes_for_tokens(
            _model(), 1, 2)[0])
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
        action="remote-read")
    kv.expand_prefill(
        session_id="s", instance_index=P_INSTANCE,
        context_tokens=INPUT_TOKENS,  # 生产口径：remote-read 工作副本仅 U
        trigger_request_id="r1")
    return kv


def _u_shards(kv):
    return kv_cache_shard_bytes_for_tokens(
        kv.model, INPUT_TOKENS, kv.tp_degree)


# ================================================= hybrid 边界事务 ==


class HybridBoundaryTransitionTest(unittest.TestCase):
    """选中 hybrid 后的执行 = copy 链（U 迁 j）+ remote 链（基础留源）。"""

    def test_hybrid_moves_u_and_keeps_base_remote(self):
        kv = _increment_kv()
        before_third = sum(_instance_bytes(kv, THIRD_INSTANCE))
        before_p = sum(_instance_bytes(kv, P_INSTANCE))
        transfer, evictions, outcome = kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="hybrid",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1",
            copy_sources=frozenset({P_INSTANCE}))
        self.assertEqual(outcome["kind"], "hybrid_move_u_remote_base")
        self.assertEqual(outcome["sources"], (HOME_INSTANCE,))
        self.assertEqual(outcome["hybrid_copy_sources"], (P_INSTANCE,))
        self.assertEqual(outcome["hybrid_remote_sources"], (HOME_INSTANCE,))
        session = kv._sessions["s"]
        # copy 链：U 迁 j（move_prefill_to_decode——目标侧实占 ΣL 字节，
        # 部分复制不免费）。
        u_bytes = sum(_u_shards(kv))
        self.assertEqual(outcome["moved_bytes"], u_bytes)
        self.assertEqual(
            sum(_instance_bytes(kv, THIRD_INSTANCE)),
            before_third + u_bytes)
        self.assertEqual(
            sum(_instance_bytes(kv, P_INSTANCE)), before_p - u_bytes)
        self.assertEqual(session.instance_index, THIRD_INSTANCE)
        # remote 链：基础留 home 权威（base_* 不动）、工作副本口径 =
        # 远读族（decode 相 credit 读流）。
        self.assertEqual(session.base_instance_index, HOME_INSTANCE)
        self.assertEqual(session.working_kind, "remote-read")
        self.assertEqual(
            session.resident_token_offset, HISTORY_TOKENS)  # U 域 [base,H)
        # 守恒（事务内 _check_invariants_after_mutation 已跑；全量复核）。
        kv._check_invariants()
        kv.kv_directory.assert_conservation()

    def test_credit_plan_source_face_converges_to_home_only(self):
        """远读子集读流：hybrid 后目标 j 的片上来源面（credit 计划的
        目录枚举入口）= 仅 home——j 缺失区间收敛到单来源。"""
        kv = _increment_kv()
        kv.commit_boundary_decode_transition(
            session_id="s", p_instance_index=P_INSTANCE,
            target_instance_index=THIRD_INSTANCE, action="hybrid",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r1",
            copy_sources=frozenset({P_INSTANCE}))
        sources = {
            int(entry.holder)
            for entry in kv.kv_directory.query_sources(
                "s", THIRD_INSTANCE)}
        self.assertEqual(sources, {HOME_INSTANCE})

    def test_endpoint_subsets_fail_closed(self):
        kv = _increment_kv()
        for subset, name in ((frozenset(), "empty"),
                             (frozenset({HOME_INSTANCE, P_INSTANCE}),
                              "full")):
            with self.assertRaisesRegex(
                    RuntimeError, "endpoint subset"):
                kv.commit_boundary_decode_transition(
                    session_id="s", p_instance_index=P_INSTANCE,
                    target_instance_index=THIRD_INSTANCE, action="hybrid",
                    history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
                    trigger_request_id="r1", copy_sources=subset)
            # fail-closed 零残留（事务先于任何字节移动拒绝）。
            kv._check_invariants()

    def test_base_copy_shape_fails_closed(self):
        """L={home}（基础复制、增量留源）= 单 span 工作副本域模型外的
        形态——显式不支持（登记 W11），不静默换别种混合执行。"""
        kv = _increment_kv()
        with self.assertRaisesRegex(RuntimeError, "single-span"):
            kv.commit_boundary_decode_transition(
                session_id="s", p_instance_index=P_INSTANCE,
                target_instance_index=THIRD_INSTANCE, action="hybrid",
                history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
                trigger_request_id="r1",
                copy_sources=frozenset({HOME_INSTANCE}))
        kv._check_invariants()

    def test_unknown_action_still_rejected(self):
        kv = _increment_kv()
        with self.assertRaises(ValueError):
            kv.commit_boundary_decode_transition(
                session_id="s", p_instance_index=P_INSTANCE,
                target_instance_index=THIRD_INSTANCE, action="nonsense",
                history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
                trigger_request_id="r1")

    def test_hybrid_two_phase_commit_sequence(self):
        """D20 阶段预约事务（stage_action="hybrid"）+ 边界事务的提交序
        （容量承诺先行——预约先于字节搬运；事务回执动作身份 = hybrid）。"""
        kv = _increment_kv()
        kv.reserve_request_capacity(
            request_id="r1", session_id="s", instance_index=P_INSTANCE,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            action="remote-read")
        transaction = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=THIRD_INSTANCE,
            stage_action="hybrid", stage_epoch=1,
            history_tokens=HISTORY_TOKENS,
            input_tokens=INPUT_TOKENS)
        self.assertEqual(transaction.stage_action, "hybrid")
        self.assertEqual(transaction.target_instance_index, THIRD_INSTANCE)
        # 同 epoch 幂等重放（§16.4）。
        replay = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=THIRD_INSTANCE,
            stage_action="hybrid", stage_epoch=1,
            history_tokens=HISTORY_TOKENS,
            input_tokens=INPUT_TOKENS)
        self.assertEqual(replay, transaction)
        # 边界事务照常（容量承诺先行的提交序）。
        _transfer, _evictions, outcome = (
            kv.commit_boundary_decode_transition(
                session_id="s", p_instance_index=P_INSTANCE,
                target_instance_index=THIRD_INSTANCE, action="hybrid",
                history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
                trigger_request_id="r1",
                copy_sources=frozenset({P_INSTANCE})))
        self.assertEqual(outcome["kind"], "hybrid_move_u_remote_base")
        kv._check_invariants()
        kv.kv_directory.assert_conservation()


# ==================================================== stage 预约足迹 ==


class HybridStageFootprintTest(unittest.TestCase):
    """hybrid 的 D 阶段承诺足迹 = 远读族基数（input/U）+ 暂存分量。"""

    def test_hybrid_footprint_remote_family_basis_with_staging(self):
        kv = _increment_kv()
        footprint = kv.stage_reservation_footprint(
            session_id="s", target_instance_index=THIRD_INSTANCE,
            action="hybrid", history_tokens=HISTORY_TOKENS,
            input_tokens=INPUT_TOKENS)
        self.assertEqual(footprint.action, "hybrid")
        # 远读族 token 基数 = input（U 迁 j 后的因果增长口径）。
        self.assertEqual(footprint.basis_context_tokens, INPUT_TOKENS)
        self.assertEqual(
            footprint.basis_shard_bytes,
            kv_cache_shard_bytes_for_tokens(
                kv.model, INPUT_TOKENS, kv.tp_degree))
        # 暂存分量在场（R 读流双缓冲 tile——remote 链真实记账）。
        self.assertTrue(any(footprint.staging_shard_bytes))
        self.assertGreater(footprint.staging_tile_tokens, 0)

    def test_hybrid_reservation_context_tokens_remote_family(self):
        kv = _increment_kv()
        self.assertEqual(
            kv.joint_reservation_context_tokens(
                action="hybrid", history_tokens=HISTORY_TOKENS,
                input_tokens=INPUT_TOKENS),
            INPUT_TOKENS)


# ================================================== 有效远读源门 ==


class EffectiveRemoteSourceGateTest(unittest.TestCase):
    """W9 遗留 2 解除路径：subset copy 收敛后的单源协议适用性恢复。"""

    def _scheduler(self):
        from online.test_settlement_state_machine import (
            _settle_ready_scheduler,
        )
        scheduler, kv, runtime = _settle_ready_scheduler(decode_length=0)
        return scheduler, kv, runtime

    def _register(self, kv, session_id, holder, t0, t1, layers=2):
        kv.kv_directory.register_entry(
            session_id=session_id, holder=holder, role="base",
            layer_start=0, layer_end=layers, token_start=t0, token_end=t1,
            bytes_by_tp_rank=(128 * (t1 - t0), 128 * (t1 - t0)),
            state="READY", release_condition="test")

    def test_duplicate_holders_do_not_count_as_second_source(self):
        """subset copy 收敛形态：持有者 2 的内容 ⊆ 持有者 1（重复/被含
        副本）——有效远读源 = 1，单源前缀读流协议适用性恢复。"""
        scheduler, kv, _runtime = self._scheduler()
        self._register(kv, "gate_s", 0, 0, 64)
        self._register(kv, "gate_s", 1, 0, 32)    # 被含副本
        self.assertEqual(
            scheduler._joint_effective_remote_source_count(
                "gate_s", {0, 1}), 1)
        # 相同内容完全重复同理。
        self._register(kv, "gate_s2", 0, 0, 64)
        self._register(kv, "gate_s2", 1, 0, 64)
        self.assertEqual(
            scheduler._joint_effective_remote_source_count(
                "gate_s2", {0, 1}), 1)

    def test_disjoint_holders_remain_multi_source(self):
        """真多源（无单一覆盖者）维持 W9 门——有效源 = 原始持有者数。"""
        scheduler, kv, _runtime = self._scheduler()
        self._register(kv, "gate_s", 0, 0, 32)
        self._register(kv, "gate_s", 1, 32, 64)
        self.assertEqual(
            scheduler._joint_effective_remote_source_count(
                "gate_s", {0, 1}), 2)

    def test_layer_split_union_needs_both(self):
        """层维拆分（无单持有者覆盖全层并集）仍是多源。"""
        scheduler, kv, _runtime = self._scheduler()
        kv.kv_directory.register_entry(
            session_id="gate_s", holder=0, role="base",
            layer_start=0, layer_end=1, token_start=0, token_end=64,
            bytes_by_tp_rank=(128 * 64, 128 * 64),
            state="READY", release_condition="test")
        kv.kv_directory.register_entry(
            session_id="gate_s", holder=1, role="base",
            layer_start=1, layer_end=2, token_start=0, token_end=64,
            bytes_by_tp_rank=(128 * 64, 128 * 64),
            state="READY", release_condition="test")
        self.assertEqual(
            scheduler._joint_effective_remote_source_count(
                "gate_s", {0, 1}), 2)

    def test_single_holder_is_effective_one(self):
        scheduler, kv, _runtime = self._scheduler()
        self._register(kv, "gate_s", 0, 0, 64)
        self.assertEqual(
            scheduler._joint_effective_remote_source_count("gate_s", {0}),
            1)


if __name__ == "__main__":
    unittest.main()
