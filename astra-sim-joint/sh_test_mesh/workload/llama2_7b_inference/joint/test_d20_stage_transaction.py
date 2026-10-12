#!/usr/bin/env python3
"""test_d20_stage_transaction.py -- D20 跨阶段预约事务、引用交接与边界
构图单元测试。

验收红线逐条落测（joint修改文档 D20 / v2.0 §14.5/§16.4）：
  1. 阶段引用保护：P 末尾到 D 提交期间本请求基础 KV 不进冷 victim
     集合（其他请求的容量事务与可回收视图都看不到它）；引用释放幂
     等；引用与预约不得双活到结算之后（release 守卫）；
  2. 跨阶段预约事务：同一事务内换址 + resize（式 20a 的
     ``Z^{D,add} = max(0, P_peak − O^{self,reuse} − Z^{transferable})``
     逐 rank 缺口）——增长 T+E 真实准备、缩小释放未被物理工作占用
     的旧承诺（D09 暂存分量）；
  3. 幂等 epoch：同 (request, epoch) 重复提交原子重放回执（重复 P
     完成通知/在途 ack 不产生第二份换址或逐出）；epoch 回退/冲突
     fail-closed；提交失败（容量类）epoch 不在案——换状态后同一
     epoch 重评合法（§16.4）；
  4. 新目标容量失败保持旧方案可继续（abort 回执记录 epoch 结局，
     继续态足迹恢复提交前快照）；
  5. 同 tick 多请求边界竞争：逐请求事务互不串账（无引用/预约双份
     或丢失）；
  6. 构图器边界守卫：未提交（未定位置/动作）的 D 永不入列车（拒绝
     不做任何账面变更）；完成归属在边界提交点锁定、完成信号按归属
     核销（缺归属 fail-closed）；
  7. 无全系统栅栏：边界引用只收缩本会话的 victim 资格，其他请求
     的容量事务照常完成（不停顿）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_d20_stage_transaction.py   （或 pytest 同路径）
"""
import os
import sys
import unittest
_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import (  # noqa: E402
    KVCapacityError,
    KVCacheManager,
    kv_cache_shard_bytes_for_tokens,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _seed,
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)


def _manager(capacity_bytes=1_000_000_000, **kwargs):
    hardware = _tiny_hardware(capacity_bytes=capacity_bytes)
    model = _tiny_model()
    topology = _two_instance_topology(hardware)
    return KVCacheManager(topology, model, **kwargs)


def _token_shards(kv, tokens):
    return kv_cache_shard_bytes_for_tokens(kv.model, tokens, kv.tp_degree)


def _seed_local_base(kv, session_id, tokens):
    """实例 0 驻留 ``tokens`` token 的活跃 LOCAL 会话（D09 夹具同款）。"""
    kv.prepare_prefill(
        session_id=session_id, target_instance_index=0,
        history_tokens=0, trigger_request_id=f"{session_id}_seed")
    kv.expand_prefill(
        session_id=session_id, instance_index=0,
        context_tokens=tokens, trigger_request_id=f"{session_id}_seed")


class StageReferenceProtectionTest(unittest.TestCase):
    """D20 阶段引用（§14.5）：P 末尾到 D 提交的基础 KV 冷 victim 保护。"""

    KV_BYTES_PER_TOKEN_PER_RANK = 128

    def test_referenced_session_is_not_a_cold_victim(self):
        """持阶段引用的完成会话不被其他请求的容量事务选中逐出；释放
        后同一事务即可经真实 T+E 逐出它（保护面 = 引用本身，"暂时没
        有活跃 D"不解除引用）。"""
        kv = _manager(capacity_bytes=100_000)
        _seed(kv, "cold", 0, 500, 20, "human")      # 500 tok 驻留实例 0
        kv.reserve_request_capacity(
            request_id="r2", session_id="s2", instance_index=1,
            final_context_tokens=500)
        ref = kv.acquire_stage_reference(
            "cold", request_id="r0", stage_epoch=1,
            reason="pd_boundary_base_kv")
        self.assertEqual(
            kv.session_stage_references("cold"),
            ("stage_ref:r0#1#pd_boundary_base_kv",))
        # 引用在册：r2 换址到实例 0 的缺口只能逐 cold 满足——被保护
        # ⇒ 深缺口失败。
        with self.assertRaises(KVCapacityError):
            kv.commit_stage_reservation_transaction(
                request_id="r2", target_instance_index=0,
                stage_action="stay", stage_epoch=1,
                history_tokens=0, input_tokens=500)
        # 释放引用：同一事务立即成功——被保护字节重新可经真实 T+E
        # 逐出回收（本请求自己的基础 KV 由真实安全事件释放）。
        self.assertTrue(kv.release_stage_reference("cold", reference_id=ref))
        self.assertEqual(kv.session_stage_references("cold"), ())
        kv.commit_stage_reservation_transaction(
            request_id="r2", target_instance_index=0,
            stage_action="stay", stage_epoch=1,
            history_tokens=0, input_tokens=500)
        self.assertEqual(kv._sessions["cold"].location, kv.REMOTE_MEMORY)

    def test_reference_release_is_idempotent(self):
        """重复释放（双交付/取消与提交竞合）安全空放返回 False。"""
        kv = _manager()
        _seed(kv, "cold", 0, 10, 20, "human")
        ref = kv.acquire_stage_reference(
            "cold", request_id="r0", stage_epoch=3)
        self.assertTrue(kv.release_stage_reference("cold", reference_id=ref))
        self.assertFalse(kv.release_stage_reference("cold", reference_id=ref))
        self.assertFalse(kv.release_stage_reference(
            "cold", reference_id="stage_ref:r0#3#pd_boundary_base_kv"))

    def test_release_guard_pins_reference_and_reservation_lifecycle(self):
        """预约注销 = 边界窗口闭合点：本请求阶段引用未释放时注销预约
        fail-closed（引用与预约不得双活到结算之后）；注销后事务账
        （epoch/回执/回退快照）一并清账。"""
        kv = _manager()
        _seed_local_base(kv, "s1", 10)
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=10)
        kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=0, stage_action="stay",
            stage_epoch=1, history_tokens=0, input_tokens=10)
        ref = kv.acquire_stage_reference("s1", request_id="r1", stage_epoch=1)
        with self.assertRaisesRegex(RuntimeError, "live stage references"):
            kv.release_request_capacity_reservation("r1")
        self.assertTrue(kv.release_stage_reference("s1", reference_id=ref))
        kv.release_request_capacity_reservation("r1")
        self.assertIsNone(kv.last_stage_transaction("r1"))
        self.assertNotIn("r1", kv._stage_epochs)
        self.assertNotIn("r1", kv._stage_previous_footprints)

    def test_reference_does_not_pause_other_requests(self):
        """无全系统栅栏（§16.4）：边界引用只收缩被引用会话的 victim
        资格——其他请求在同一实例上的容量事务照常完成，不停顿。"""
        kv = _manager(capacity_bytes=100_000)
        _seed(kv, "victim_ok", 0, 500, 20, "tool")   # 可合法逐出
        kv.reserve_request_capacity(
            request_id="r2", session_id="s2", instance_index=1,
            final_context_tokens=500)
        # 边界中的 r0：先落自身基础会话，再持阶段引用（引用的是它自己
        # 的基础 KV；victim_ok 不是它的会话）。
        _seed_local_base(kv, "r0_session", 10)
        kv.acquire_stage_reference(
            "r0_session", request_id="r0", stage_epoch=1)
        kv.commit_stage_reservation_transaction(
            request_id="r2", target_instance_index=0,
            stage_action="stay", stage_epoch=1,
            history_tokens=0, input_tokens=500)
        self.assertEqual(
            kv._sessions["victim_ok"].location, kv.REMOTE_MEMORY)


class StageTransactionCommitTest(unittest.TestCase):
    """D20 跨阶段预约事务（§14.5）：transfer/resize/幂等 epoch。

    夹具基元：实例 0 驻留 base_tokens token LOCAL 基（远读语义下的
    基础历史），r1 预约落在实例 1（执行端）——每 rank 每 token KV =
    128 B。事务足迹按显式 history/input 推导（事务面是候选无关的
    协议基础设施，动作适用性 policing 在 JCM）。
    """

    def _prepared(self, *, capacity_bytes=1_000_000_000, action=None,
                  base_tokens=40, final_tokens=40, reserve_instance=1,
                  **kwargs):
        kv = _manager(capacity_bytes=capacity_bytes, **kwargs)
        _seed_local_base(kv, "s1", base_tokens)
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1",
            instance_index=reserve_instance,
            final_context_tokens=final_tokens, action=action)
        return kv

    def test_cross_instance_transfer_rebases_to_stage_footprint(self):
        """跨实例切换 + resize（40 tok 完整承诺 → 20 tok 远读承诺）：
        目标只按 D 阶段新承诺 T+E（式 20a 缺口），占用视图随换址原子
        迁移（无双份、无丢失），承诺记录重定基到 D 阶段足迹。"""
        kv = self._prepared(reserve_instance=0)
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1,
            stage_action="remote-read", stage_epoch=1,
            history_tokens=20, input_tokens=20)
        self.assertEqual(txn.source_instance_index, 0)
        self.assertEqual(txn.target_instance_index, 1)
        self.assertEqual(txn.released_shard_bytes, (0, 0))
        self.assertEqual(txn.reserved_shard_bytes, _token_shards(kv, 20))
        self.assertFalse(txn.irreversible)   # 目标余量充足：零逐出
        self.assertFalse(txn.aborted)
        self.assertEqual(kv._reservations["r1"].instance_index, 1)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(0)), (0, 0))
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)),
            _token_shards(kv, 20))
        self.assertEqual(kv._reservations["r1"].final_context_tokens, 20)
        self.assertEqual(kv._reservations["r1"].action, "remote-read")

    def test_in_place_growth_prepares_real_gap(self):
        """远读承诺（20 tok input 基数 @实例 1）原地转完整接续（40 tok
        = H_B）：增长半边对缺口 T+E 真实准备（式 20a），released 为
        零、逐出为零（余量充足）。"""
        kv = self._prepared(action="remote-read", base_tokens=20,
                            final_tokens=40)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)),
            _token_shards(kv, 20))
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=1, history_tokens=20, input_tokens=20)
        self.assertEqual(txn.released_shard_bytes, (0, 0))
        self.assertEqual(txn.reserved_shard_bytes, _token_shards(kv, 40))
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)),
            _token_shards(kv, 40))
        self.assertFalse(txn.irreversible)
        self.assertFalse(txn.aborted)

    def test_shrink_releases_unused_staging_commitment(self):
        """缩小组件释放未被物理工作占用的旧承诺（D09 暂存）：远读承诺
        （20 tok 基数 + 8192 B/rank 暂存）切到完整 stay 足迹（60 tok）
        ——暂存分量随事务释放（released = 旧申索 − 新申索 = 3072）。"""
        kv = self._prepared(
            action="remote-read", base_tokens=40, final_tokens=60,
            remote_read_tile_tokens=32, remote_read_kv_token_bytes=128)
        reservation = kv._reservations["r1"]
        self.assertEqual(
            reservation.staging_shard_bytes, (2 * 32 * 128,) * 2)
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=1, history_tokens=40, input_tokens=20)
        old_claim = _token_shards(kv, 20)[0] + 2 * 32 * 128
        new_claim = _token_shards(kv, 60)[0]
        self.assertEqual(
            txn.released_shard_bytes,
            (old_claim - new_claim,) * kv.tp_degree)
        self.assertEqual(
            txn.reserved_shard_bytes, _token_shards(kv, 60))
        # 暂存字段随 resize 清账（bytes 与 tile tokens 同生同灭）。
        self.assertEqual(kv._reservations["r1"].staging_shard_bytes, ())
        self.assertEqual(kv._reservations["r1"].staging_tile_tokens, 0)

    def test_resize_below_own_resident_fails_closed(self):
        """缩到自有驻留之下不是合法承诺（驻留字节仍在账）——fail-
        closed，事务不落账、预约原样。"""
        kv = self._prepared()
        with self.assertRaisesRegex(ValueError, "below its own resident"):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=0,
                stage_action="remote-read", stage_epoch=1,
                history_tokens=20, input_tokens=20)
        self.assertIsNone(kv.last_stage_transaction("r1"))
        self.assertEqual(kv._reservations["r1"].final_context_tokens, 40)
        self.assertEqual(kv._reservations["r1"].action, None)

    def test_idempotent_replay_and_stale_epoch(self):
        """同 (request, epoch) 目标/动作一致的重复提交原子重放同一回执
        （恰一次提交）；同 epoch 冲突目标与 epoch 回退 fail-closed；
        重放不产生第二份占用（占用视图恒一份）。"""
        kv = self._prepared(reserve_instance=0)
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=2, history_tokens=20, input_tokens=20)
        replay = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=2, history_tokens=20, input_tokens=20)
        self.assertIs(replay, txn)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)),
            _token_shards(kv, 40))   # 重放不叠加
        with self.assertRaises(ValueError):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=0,
                stage_action="stay", stage_epoch=2,
                history_tokens=20, input_tokens=20)
        with self.assertRaises(ValueError):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=1,
                stage_action="stay", stage_epoch=1,
                history_tokens=20, input_tokens=20)

    def test_capacity_failure_epoch_stays_clean_then_same_epoch_retry(self):
        """新目标容量失败 → KVCapacityError、预约留在源实例、epoch 不
        在案；状态变化（占位会话完成成合法 victim）后同一 epoch 重评
        成功（§16.4"回到同一未提交边界重评估"）；重评的已提交逐出
        在回执上（irreversible）。"""
        kv = self._prepared(capacity_bytes=100_000, reserve_instance=0)
        # 实例 1 活跃占位（未 mark_complete，非合法 victim）吃空容量。
        kv.prepare_prefill(
            session_id="hog", target_instance_index=1,
            history_tokens=0, trigger_request_id="hog")
        kv.expand_prefill(
            session_id="hog", instance_index=1, context_tokens=690,
            trigger_request_id="hog")
        with self.assertRaises(KVCapacityError):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=1,
                stage_action="stay", stage_epoch=1,
                history_tokens=20, input_tokens=20)
        self.assertEqual(kv._reservations["r1"].instance_index, 0)
        self.assertIsNone(kv.last_stage_transaction("r1"))
        self.assertNotIn("r1", kv._stage_epochs)
        # 占位会话完成 → 合法 victim → 同 epoch 重评成功。
        kv.mark_complete("hog", 30, "human")
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1,
            stage_action="stay", stage_epoch=1,
            history_tokens=20, input_tokens=20)
        self.assertEqual(txn.target_instance_index, 1)
        self.assertTrue(txn.irreversible)   # 真实 T+E 逐出已提交
        self.assertEqual(len(txn.evictions), 1)
        self.assertEqual(kv._reservations["r1"].instance_index, 1)

    def test_abort_reverses_flip_and_records_epoch_outcome(self):
        """abort（§16.4"保留继续方案的合法状态"）：换址半边回退到源
        实例 + 足迹恢复提交前快照；epoch 结局记 aborted（重复 abort
        重放同一回执；同 epoch 的成功提交重放 fail-closed）。"""
        kv = self._prepared(reserve_instance=0)
        kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=1, history_tokens=20, input_tokens=20)
        self.assertEqual(kv._reservations["r1"].instance_index, 1)
        aborted = kv.abort_stage_reservation_transaction(
            request_id="r1", source_instance_index=0, stage_epoch=1)
        self.assertTrue(aborted.aborted)
        self.assertEqual(aborted.target_instance_index, 0)
        # 足迹恢复 = 提交前快照（完整 40 tok 承诺在源实例；源侧驻留
        # 40 tok 全额覆盖 ⇒ 净申索为 0——占用视图与提交前逐位一致）。
        self.assertEqual(kv._reservations["r1"].instance_index, 0)
        self.assertEqual(kv._reservations["r1"].final_context_tokens, 40)
        self.assertEqual(kv._reservations["r1"].action, None)
        self.assertEqual(tuple(kv._reserved_bytes_by_tp_rank(0)), (0, 0))
        self.assertEqual(tuple(kv._reserved_bytes_by_tp_rank(1)), (0, 0))
        # 幂等：重复 abort 原子重放；同 epoch 成功提交重放 fail-closed。
        replay = kv.abort_stage_reservation_transaction(
            request_id="r1", source_instance_index=0, stage_epoch=1)
        self.assertIs(replay, aborted)
        with self.assertRaises(ValueError):
            kv.commit_stage_reservation_transaction(
                request_id="r1", target_instance_index=1,
                stage_action="stay", stage_epoch=1,
                history_tokens=20, input_tokens=20)

    def test_abort_without_committed_flip_is_idempotent_noop(self):
        """提交在容量检查即失败（无在案成功回执）时 abort 幂等空放返
        回 None；预约不在源实例又无在案回执 = 账目脱钩，fail-closed。"""
        kv = self._prepared(reserve_instance=0)
        self.assertIsNone(kv.abort_stage_reservation_transaction(
            request_id="r1", source_instance_index=0, stage_epoch=1))
        object.__setattr__(kv._reservations["r1"], "instance_index", 1)
        with self.assertRaisesRegex(RuntimeError, "sits on"):
            kv.abort_stage_reservation_transaction(
                request_id="r1", source_instance_index=0, stage_epoch=1)

    def test_same_tick_competition_between_two_requests(self):
        """同 tick 两请求边界事务竞争：逐请求 epoch 独立、互不串账，
        占用视图 = 两份承诺之和（无双份、无丢失）。"""
        kv = _manager()
        for rid, sid in (("r1", "s1"), ("r2", "s2")):
            _seed_local_base(kv, sid, 10)
            kv.reserve_request_capacity(
                request_id=rid, session_id=sid, instance_index=0,
                final_context_tokens=10)
        txn_r1 = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1, stage_action="stay",
            stage_epoch=1, history_tokens=0, input_tokens=10, at_ns=1000)
        txn_r2 = kv.commit_stage_reservation_transaction(
            request_id="r2", target_instance_index=1, stage_action="stay",
            stage_epoch=1, history_tokens=0, input_tokens=10, at_ns=1000)
        self.assertEqual(txn_r1.stage_epoch, txn_r2.stage_epoch)
        self.assertIsNot(txn_r1, txn_r2)
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(1)),
            _token_shards(kv, 20))   # 10 + 10，无双份
        self.assertEqual(
            tuple(kv._reserved_bytes_by_tp_rank(0)), (0, 0))
        self.assertEqual(kv.last_stage_transaction("r1"), txn_r1)
        self.assertEqual(kv.last_stage_transaction("r2"), txn_r2)


# ------------------------------------------------------------ 状态机 --

from joint.test_d18_boundary_decision import (  # noqa: E402
    _drain_ready_scheduler,
)
from online.sh30_online_scheduler import (  # noqa: E402
    _OnlineRequestRuntime,
)


class BoundaryStateMachineTest(unittest.TestCase):
    """D20 在线边界（§16.4）：重复通知幂等 / 披露与完成归属 / 构图
    守卫 / 同 tick 竞争。"""

    def test_duplicate_prefill_done_notification_is_idempotent(self):
        """重复 P_COMPUTE_DONE 通知：幂等忽略（只落披露行），状态机、
        epoch、就绪队列、事务账零副作用，无第二次 D 决策。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        stage_after_first = runtime.joint_stage
        epoch_after_first = runtime.joint_boundary_epoch
        rows_after_first = list(scheduler.online_log_rows)
        pending_after_first = list(
            scheduler.instances[0].pending_decode_ready)
        scheduler._on_prefill_drain("r0", 1000)   # 重复通知
        self.assertEqual(runtime.joint_stage, stage_after_first)
        self.assertEqual(runtime.joint_boundary_epoch, epoch_after_first)
        self.assertEqual(
            scheduler.instances[0].pending_decode_ready,
            pending_after_first)
        new_rows = scheduler.online_log_rows[len(rows_after_first):]
        self.assertEqual(
            [row["kind"] for row in new_rows],
            ["boundary_duplicate_ignored"])

    def test_boundary_records_epoch_transaction_and_attribution(self):
        """边界提交点：阶段 epoch / 事务回执 / 完成归属三件套入
        boundary_decision 与 plan_dict；引用窗口随边界闭合（会话零
        在持引用）；预约按 drain 纪律注销且事务账随清（生命周期一致
        ——无引用/预约双活）。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        decision = runtime.boundary_decision
        self.assertEqual(decision["stage_epoch"], 1)
        txn_summary = decision["stage_transaction"]
        self.assertEqual(txn_summary["stage_epoch"], 1)
        self.assertEqual(txn_summary["stage_action"], "stay")
        self.assertEqual(txn_summary["source_instance_index"], 0)
        self.assertEqual(txn_summary["target_instance_index"], 0)
        self.assertFalse(txn_summary["aborted"])
        self.assertFalse(txn_summary["irreversible"])
        self.assertEqual(runtime.completion_instance_index, 0)
        self.assertEqual(
            runtime.plan_dict()["completion_instance_index"], 0)
        self.assertEqual(runtime.plan_dict()["joint_boundary_epoch"], 1)
        # 边界窗口闭合：P 阶段引用已释放（无引用/预约双活）。
        self.assertIsNone(runtime.joint_stage_reference_id)
        self.assertEqual(kv.session_stage_references(runtime.session_id), ())
        # 预约已按 drain 纪律注销，事务账随清。
        self.assertNotIn("r0", kv._reservations)
        self.assertIsNone(kv.last_stage_transaction("r0"))

    def test_plan_dict_decode_instance_zero_is_not_prefill_fallback(self):
        """回归钉（smoke-matrix combo_J/TJ/JE/TJE seq377 fail-closed）：
        plan_dict 的 decode 实例回退判据必须显式 ``is not None``——实例
        0 是合法 decode 实例（J 边界可跨实例选中），按真值回退会把
        decode@0 请求的 plan 误标为 prefill 实例；_emit_completion 以该
        字段取 decode_group 查 _block_ends 块末账本，实例不符即误报
        "segment block ends ('seg2') missing on decode ranks"（现场：
        session_1_request_3 prefill@5 → decode@0，退出列车在实例 0 落
        seg2，完成批却在实例 5 的 ranks 上查账）。字段为 None（P 阶段
        未定）才回退 prefill 实例。"""
        runtime = _OnlineRequestRuntime({
            "request_id": "rz", "session_id": "rz_session",
            "turn_index": 3, "queue_index": 4,
            "prefill_length": 594, "decode_length": 325,
            "history_tokens_before": 12794,
            "prefill_context_tokens": 13388,
            "final_context_tokens": 13713,
        })
        runtime.prefill_instance_index = 5
        runtime.decode_instance_index = 0
        self.assertEqual(runtime.plan_dict()["decode_instance_index"], 0)
        runtime.decode_instance_index = None
        self.assertEqual(runtime.plan_dict()["decode_instance_index"], 5)

    def test_terminal_settle_records_attribution_and_closes_reference(self):
        """已知终止（decode_length=0）：完成归属锁定 P 实例、引用窗口
        闭合、无事务回执（无 D 提交）。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=0)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(runtime.completion_instance_index, 0)
        self.assertEqual(runtime.boundary_decision["stage_epoch"], 1)
        self.assertIsNone(runtime.boundary_decision["stage_transaction"])
        self.assertIsNone(runtime.joint_stage_reference_id)
        self.assertEqual(kv.session_stage_references(runtime.session_id), ())

    def test_composer_rejects_precommit_stage_joiner(self):
        """构图器边界守卫：未提交阶段（D_DECISION_PENDING）的请求混入
        就绪队列后，列车构图 fail-closed 且不做任何账面变更（就绪队列
        保持原样、已提交成员不被半途迁移）；守卫拒绝之外的其他发射面
        错误不等于守卫放行（正控制）。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler._on_prefill_drain("r0", 1000)
        self.assertEqual(
            scheduler.instances[0].pending_decode_ready, [runtime])
        intruder = _OnlineRequestRuntime({
            "request_id": "rx", "session_id": "rx_session",
            "turn_index": 0, "queue_index": 1,
            "prefill_length": 50, "decode_length": 4,
            "history_tokens_before": 0,
            "prefill_context_tokens": 50,
            "final_context_tokens": 54,
        })
        intruder.joint_stage = "D_DECISION_PENDING"
        intruder.decode_instance_index = 0
        scheduler.runtime_by_request_id["rx"] = intruder
        scheduler.instances[0].pending_decode_ready.append(intruder)
        with self.assertRaisesRegex(RuntimeError, "train composer"):
            scheduler._plan_and_emit_trains(1001)
        # 拒绝原子性：账面零变更（先验全体成员，拒绝时不迁移任何成员）。
        self.assertEqual(
            scheduler.instances[0].pending_decode_ready, [runtime, intruder])
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertEqual(intruder.joint_stage, "D_DECISION_PENDING")
        self.assertNotIn(intruder, scheduler.instances[0].active_decode)

    def test_completion_signal_without_attribution_fails_closed(self):
        """完成信号按边界锁定的完成归属核销；缺归属（未经边界）= 状
        态机破损，fail-closed。"""
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        runtime.completion_instance_index = None
        with self.assertRaisesRegex(
                RuntimeError, "boundary completion attribution"):
            scheduler._finalize_completed_trains([], ["r0"], [], 1000)

    def test_same_tick_two_request_boundaries_no_cross_talk(self):
        """同 tick 两个请求先后过边界：各自 epoch=1、事务与归属逐请
        求正确、就绪队列两成员（同 tick 竞争无串账）。"""
        scheduler, kv, _ = _drain_ready_scheduler(decode_length=8)
        runtime_r0 = scheduler.runtime_by_request_id["r0"]
        runtime_r1 = _OnlineRequestRuntime({
            "request_id": "r1", "session_id": "r1_session",
            "turn_index": 0, "queue_index": 1,
            "prefill_length": 1024, "decode_length": 8,
            "history_tokens_before": 0,
            "prefill_context_tokens": 1024,
            "final_context_tokens": 1032,
        })
        kv.reserve_request_capacity(
            request_id="r1", session_id="r1_session", instance_index=0,
            final_context_tokens=1024)
        kv.prepare_prefill(
            session_id="r1_session", target_instance_index=0,
            history_tokens=0, trigger_request_id="r1")
        kv.expand_prefill(
            session_id="r1_session", instance_index=0, context_tokens=1024,
            trigger_request_id="r1")
        runtime_r1.prefill_instance_index = 0
        runtime_r1.joint_stage = "P_RUNNING"
        runtime_r1.queued_chunk_load_ns = 0
        scheduler.runtime_by_request_id["r1"] = runtime_r1
        scheduler.next_request["r1"] = None
        scheduler.instances[0].qp.append(runtime_r1)
        scheduler._on_prefill_drain("r0", 1000)
        scheduler._on_prefill_drain("r1", 1000)   # 同 tick 竞争
        for runtime in (runtime_r0, runtime_r1):
            self.assertEqual(runtime.joint_boundary_epoch, 1)
            self.assertEqual(runtime.completion_instance_index, 0)
            self.assertEqual(runtime.joint_stage, "D_COMMITTED")
            self.assertEqual(
                runtime.boundary_decision["stage_transaction"]
                ["target_instance_index"], 0)
        self.assertEqual(
            len(scheduler.instances[0].pending_decode_ready), 2)
        self.assertIsNone(kv.last_stage_transaction("r0"))
        self.assertIsNone(kv.last_stage_transaction("r1"))




# ============================================================ W5 事务扩展 ==


class W5TransactionExtensionTest(unittest.TestCase):
    """W5（§7.2）：提交点版本校验 / 所有来源引用 / A22 分记 / attempt
    维度 / 义务解析器（G-B 门 V5 组：失败/重复回调/同 tick 竞争无预约
    或引用泄漏）。"""

    def test_commit_point_version_conflict_detected(self):
        """expected_versions 与提交点漂移（目录版本/阶段纪元表）→
        KVStageVersionConflict（不沿用决策时点容量承诺）；匹配 → 正常
        提交。"""
        from face_scheduler import KVStageVersionConflict
        kv = self._prepared_w5()
        versions = {
            "directory_version": kv.kv_directory.version,
            "stage_epochs": tuple(sorted(kv._stage_epochs.items())),
        }
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1,
            stage_action="stay", stage_epoch=1,
            history_tokens=20, input_tokens=20,
            expected_versions=versions)
        self.assertEqual(txn.target_instance_index, 1)
        # 目录版本漂移（决策后被同 tick 竞争推进）→ fail-closed（r9 =
        # 另一请求的提交尝试；预约先行建立）。
        _seed_local_base(kv, "s9", 10)
        kv.reserve_request_capacity(
            request_id="r9", session_id="s9", instance_index=0,
            final_context_tokens=20)
        kv._kv_directory._bump()
        with self.assertRaises(KVStageVersionConflict):
            kv.commit_stage_reservation_transaction(
                request_id="r9", target_instance_index=1,
                stage_action="stay", stage_epoch=1,
                history_tokens=20, input_tokens=20,
                expected_versions=versions)

    @staticmethod
    def _prepared_w5(capacity_bytes=1_000_000_000):
        kv = _manager(capacity_bytes=capacity_bytes)
        _seed_local_base(kv, "s1", 10)
        kv.reserve_request_capacity(
            request_id="r1", session_id="s1", instance_index=0,
            final_context_tokens=20)
        kv.prepare_prefill(
            session_id="s1", target_instance_index=0,
            history_tokens=10, trigger_request_id="r1")
        kv.expand_prefill(
            session_id="s1", instance_index=0, context_tokens=20,
            trigger_request_id="r1")
        return kv

    def test_source_references_acquired_and_released_conserved(self):
        """所有来源引用经目录 acquire_entry_reference（执行 pin）——
        引用守恒：acquire 后在册、release 后清零；settle 前（release
        前的）条目不可被 mirror retire（源保护）。来源条目 = 目标缺失
        内容（[20,30)@1——目标 0 已持 [0,20)，并集口径下目标已持内容
        不是"来源"，§6.1/§6.3）。"""
        kv = self._prepared_w5()
        directory = kv.kv_directory
        entry = directory.register_entry(
            session_id="s1", holder=1, role="boundary-source",
            layer_start=0, layer_end=kv.model.layers,
            token_start=20, token_end=30,
            bytes_by_tp_rank=_token_shards(kv, 10),
            state="READY")
        scheduler = self._scheduler_shell(kv)
        runtime = scheduler.runtime_by_request_id["r0"]
        handles = scheduler._joint_acquire_source_references(runtime, 0)
        runtime.joint_source_reference_ids = handles  # 生产边界提交点同款
        self.assertIsNotNone(handles)
        self.assertEqual(handles, ((entry.entry_id, True),))
        self.assertGreater(
            directory._read_refs.get(entry.entry_id, 0), 0)
        # 源保护：持引用条目不可被 retire（mirror 同步路径的守恒门）。
        from joint.kv_location_directory import KvDirectoryError
        with self.assertRaises(KvDirectoryError):
            directory.retire_entry(entry.entry_id, reason="test_guard")
        # 释放：配对守恒（幂等二放 fail-closed 由目录保证）。
        scheduler._joint_release_source_references("r0")
        self.assertEqual(
            directory._read_refs.get(entry.entry_id, 0), 0)
        self.assertIsNone(runtime.joint_source_reference_ids)
        # 幂等空放。
        scheduler._joint_release_source_references("r0")

    @staticmethod
    def _scheduler_shell(kv):
        from joint.test_d18_boundary_decision import _drain_ready_scheduler
        scheduler, _kv, runtime = _drain_ready_scheduler(decode_length=4)
        # 换 kv 与 runtime 会话对齐（来源条目挂在 s1/r0_session 上）。
        scheduler.kv_manager = kv
        runtime.session_id = "s1"
        return scheduler

    def test_a22_retained_obligations_and_attempt_recorded(self):
        """A22：阶段提交成功状态与不可撤销物理子义务分记——真实 T+E
        逐出（irreversible）保留在成功/aborted 回执的 retained_
        obligations（身份/逐 rank 字节）；attempt 维度在案（首试 0）。"""
        kv = self._prepared_w5(capacity_bytes=100_000)
        kv.prepare_prefill(
            session_id="hog", target_instance_index=1,
            history_tokens=0, trigger_request_id="hog")
        kv.expand_prefill(
            session_id="hog", instance_index=1, context_tokens=690,
            trigger_request_id="hog")
        kv.mark_complete("hog", 30, "human")
        txn = kv.commit_stage_reservation_transaction(
            request_id="r1", target_instance_index=1,
            stage_action="stay", stage_epoch=1,
            history_tokens=20, input_tokens=20)
        self.assertTrue(txn.irreversible)
        self.assertEqual(txn.attempt, 0)
        self.assertTrue(txn.retained_obligations)
        obligation = txn.retained_obligations[0]
        self.assertEqual(obligation["kind"], "eviction_transfer")
        self.assertIn("bytes_by_rank", obligation)
        # abort 分记：不可撤销物理工作保留身份（不取消、不漏计）。
        aborted = kv.abort_stage_reservation_transaction(
            request_id="r1", source_instance_index=0, stage_epoch=1)
        self.assertTrue(aborted.aborted)
        self.assertTrue(aborted.irreversible)
        self.assertEqual(
            aborted.retained_obligations, txn.retained_obligations)
        self.assertEqual(aborted.attempt, txn.attempt)

    def test_obligation_resolver_gates_settle_in_boundary_model(self):
        """C6：t_B 义务表 → JCM obligation_resolver（未完义务参竞争 +
        门控 settle；无义务 = None 降级披露）。"""
        from joint.boundary_layout import InheritedObligation
        scheduler, kv, runtime = self._drain_shell()
        obligations = (
            InheritedObligation(
                obligation_id="r0#readplan", kind="remote_read_plan",
                receiver_instance=0),
        )
        resolver = scheduler._joint_obligation_resolver(obligations)
        self.assertIsNotNone(resolver)
        view = resolver("r0#readplan")
        self.assertEqual(view.kind, "remote_read_plan")
        self.assertTrue(view.gate_settle)
        self.assertFalse(view.done)
        self.assertIsNone(scheduler._joint_obligation_resolver(()))
        # 边界代价模型注入（JCM 构造面）。
        boundary = scheduler._joint_boundary_view(
            runtime, scheduler.instances[0])
        model = scheduler._joint_boundary_cost_model(
            runtime, boundary, 1000, obligations=obligations)
        self.assertIsNotNone(getattr(model, "obligation_resolver", None))

    @staticmethod
    def _drain_shell():
        from joint.test_d18_boundary_decision import _drain_ready_scheduler
        return _drain_ready_scheduler(decode_length=4)

    def test_boundary_decision_discloses_control_latency_and_recheck(self):
        """C5：边界决策行披露 control_latency_ns 与提交点复查面（0 =
        assumed-zero 口径不变；非零 = 预测侧 #ctrl 腿 + 执行侧复查）。"""
        scheduler, kv, runtime = self._drain_shell()
        scheduler._on_prefill_drain("r0", 1000)
        decision = runtime.boundary_decision
        self.assertIn("control_latency_ns", decision)
        self.assertEqual(decision["control_latency_ns"], 0)
        self.assertEqual(
            decision["control_commit_recheck"],
            "versions+capacity+source_references")


if __name__ == "__main__":
    unittest.main()
