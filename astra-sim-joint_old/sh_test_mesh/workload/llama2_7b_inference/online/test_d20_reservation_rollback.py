#!/usr/bin/env python3
"""test_d20_reservation_rollback.py -- D20 修复轮二：P→D 边界通用异常
路径的预约事务回滚补齐钉测。

审查结论（修复轮二）："通用事务回滚证据不足"——drain 边界 try 块内若
阶段预约事务（``commit_stage_reservation_transaction``：resize/transfer
半边）已提交后再遇**非容量**异常，原 except 块只幂等注销 staging lease
与阶段引用，预约不显式 abort——泄漏。本轮在通用异常路径补幂等回滚：

回滚矩阵（异常注入点 × 释放断言）：

  A. **换址半边已提交、物理迁移未落地**（注入 ``commit_boundary_
     decode_transition`` 抛 RuntimeError，选中 = 实例 1 的 copy 切换）
     → 显式 ``abort_stage_reservation_transaction``：epoch 结局记
     aborted、预约换址回源实例、足迹恢复提交前快照；lease/引用/就绪
     队列成员全释放，无悬挂。
  A'. **同实例 resize 半边已提交**（同注入点，选中 = stay）→ 同路
     abort（纯 resize 的足迹恢复），幂等重放不双重恢复。
  B. **D_COMMITTED 之后**（注入 ``_emit_join_decision`` 抛错——物理
     工作已提交、abort 合同窗口已闭）→ **不伪造换址回退**（事务保持
     已提交结局），但 lease/引用/成员仍全清理。
  C. **事务提交前**（注入 ``_joint_boundary_cost_model`` 抛错）→ 无
     在案事务可回滚（空放），引用/lease 照常释放——不误触发 abort。

幂等性：A 注入后再次调用 abort（同 epoch/同源）原子重放 aborted 回执，
不二次恢复/不抛错；重复 drain 通知走幂等门（无第二份回滚）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_d20_reservation_rollback.py -q
"""
import os
import sys
import unittest
from unittest import mock

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from joint.test_d18_boundary_decision import (  # noqa: E402
    _drain_ready_scheduler,
)


def _assert_trio_released(test, scheduler, kv, runtime):
    """三件套释放断言：阶段引用、暂存 lease、就绪队列成员（无悬挂）。"""
    # 阶段引用：runtime 句柄置空 + 会话引用集为空（幂等双保险）。
    test.assertIsNone(runtime.joint_stage_reference_id)
    test.assertEqual(
        kv.session_stage_references(runtime.session_id), ())
    # 暂存 lease：无在册（替身断言手法 = 真实 KVCacheManager 查询面）。
    test.assertIsNone(kv.staging_lease_for_request(runtime.request_id))
    # 就绪队列成员资格不悬挂（异常点晚于 append 时被回滚移除）。
    for state in scheduler.instances:
        test.assertNotIn(runtime, state.pending_decode_ready)


class SwitchTransitionFailureRollbackTest(unittest.TestCase):
    """矩阵 A：换址半边已提交 + 迁移未落地 → 显式 abort。"""

    def _switch_ready(self):
        """实例 0 在服负载压重（3 活跃 decode 成员）+ 掩码 {copy,stay}
        ——argmin 翻到实例 1 的 copy 切换（容量宽裕，换址半边可提交）。"""
        from joint.joint_config import parse_joint_config
        from online.test_decision_route_serialization import _make_runtime
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler.joint_config = parse_joint_config({
            "JOINT_ALLOWED_ACTIONS": "copy,stay"})
        scheduler._joint_mode = scheduler.joint_config.scheduler_mode
        for seq in range(3):
            hog = _make_runtime(f"active0_{seq}")
            scheduler.runtime_by_request_id[f"active0_{seq}"] = hog
            scheduler.instances[0].active_decode.append(hog)
        return scheduler, kv, runtime

    def test_committed_switch_transaction_aborts_on_generic_failure(self):
        scheduler, kv, runtime = self._switch_ready()
        with mock.patch.object(
                kv, "commit_boundary_decode_transition",
                side_effect=RuntimeError("injected transition failure")):
            with self.assertRaises(RuntimeError):
                scheduler._on_prefill_drain("r0", 1000)
        # 选中确为切换（非 stay）——事务的换址半边真的提交过。
        session = kv._sessions[runtime.session_id]
        self.assertEqual(session.instance_index, 0)  # 迁移未落地（注入点）
        # 预约事务显式回滚：epoch 结局 aborted、预约回源实例。
        txn = kv.last_stage_transaction("r0")
        self.assertIsNotNone(txn)
        self.assertTrue(txn.aborted)
        self.assertEqual(txn.stage_epoch, runtime.joint_boundary_epoch)
        self.assertEqual(txn.target_instance_index, 0)  # aborted 回执 = 源
        reservation = kv._reservations["r0"]
        self.assertEqual(reservation.instance_index, 0)
        # 三件套全释放（lease/引用/成员）。
        _assert_trio_released(self, scheduler, kv, runtime)

    def test_abort_is_idempotent_under_repeated_rollback(self):
        """重复异常/嵌套清理不双重释放：同 epoch 再次 abort 原子重放
        aborted 回执（不抛错、不二次恢复）。"""
        scheduler, kv, runtime = self._switch_ready()
        with mock.patch.object(
                kv, "commit_boundary_decode_transition",
                side_effect=RuntimeError("injected transition failure")):
            with self.assertRaises(RuntimeError):
                scheduler._on_prefill_drain("r0", 1000)
        replay = kv.abort_stage_reservation_transaction(
            request_id="r0", source_instance_index=0,
            stage_epoch=runtime.joint_boundary_epoch)
        self.assertIsNotNone(replay)
        self.assertTrue(replay.aborted)
        self.assertEqual(kv._reservations["r0"].instance_index, 0)
        # 重复 drain 通知走 幂等门——不重做决策/回滚。
        scheduler._on_prefill_drain("r0", 1000)
        rows = [row for row in scheduler.online_log_rows
                if row["kind"] == "boundary_duplicate_ignored"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(
            kv._reservations["r0"].instance_index, 0)


class StayResizeFailureRollbackTest(unittest.TestCase):
    """矩阵 A'：同实例 resize 半边已提交（stay）→ 同路 abort。"""

    def test_committed_stay_resize_aborts_on_generic_failure(self):
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        with mock.patch.object(
                kv, "commit_boundary_decode_transition",
                side_effect=RuntimeError("injected transition failure")):
            with self.assertRaises(RuntimeError):
                scheduler._on_prefill_drain("r0", 1000)
        # stay 选择（均衡负载平局优先原地继续）——事务是纯 resize 半边。
        txn = kv.last_stage_transaction("r0")
        self.assertIsNotNone(txn)
        self.assertFalse(txn.irreversible)  # 宽裕容量：零逐出可回退
        self.assertTrue(txn.aborted)
        self.assertEqual(
            kv._reservations["r0"].instance_index, 0)
        _assert_trio_released(self, scheduler, kv, runtime)


class PostCommitFailureTest(unittest.TestCase):
    """矩阵 B：D_COMMITTED 之后——abort 合同窗口已闭，不伪造回退。"""

    def test_post_commit_failure_keeps_transaction_but_cleans_trio(self):
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        with mock.patch.object(
                scheduler, "_emit_join_decision",
                side_effect=RuntimeError("injected emit failure")):
            with self.assertRaises(RuntimeError):
                scheduler._on_prefill_drain("r0", 1000)
        # 物理工作已提交（不伪造 aborted）：正常路径已走到预约注销点
        # （stay 无读流 → 预约随边界注销、事务账随预约清账）。若通用
        # 路径误 abort，预约会被恢复回源实例在账（release 不发生）——
        # 此处 NotIn 即"未伪造回退"的可观测证明。
        self.assertNotIn("r0", kv._reservations)
        self.assertIsNone(kv.last_stage_transaction("r0"))
        # lease/引用/成员照常清理（引用在 正常闭合，幂等空放）。
        _assert_trio_released(self, scheduler, kv, runtime)


class PreCommitFailureTest(unittest.TestCase):
    """矩阵 C：事务提交前异常——无在案事务，回滚空放不误触发。"""

    def test_pre_decision_failure_releases_reference_without_abort(self):
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        with mock.patch.object(
                scheduler, "_joint_boundary_cost_model",
                side_effect=RuntimeError("injected cost model failure")):
            with self.assertRaises(RuntimeError):
                scheduler._on_prefill_drain("r0", 1000)
        # 无在案事务（首边界从未提交）——abort 路空放。
        self.assertIsNone(kv.last_stage_transaction("r0"))
        # 预约仍在准入态（源实例、未动）——不是泄漏，是尚未开始。
        self.assertEqual(kv._reservations["r0"].instance_index, 0)
        _assert_trio_released(self, scheduler, kv, runtime)


class CapacityFallbackPathUntouchedTest(unittest.TestCase):
    """已知容量失败有专门回退路径（保持不动）：KVCapacityError 仍走
    D 边界 abort + stay 重提交（通用路径不抢跑/不双跑）。"""

    def test_capacity_failure_uses_dedicated_fallback_not_generic_rollback(self):
        from joint.joint_config import parse_joint_config
        from online.test_decision_route_serialization import _make_runtime
        scheduler, kv, runtime = _drain_ready_scheduler(decode_length=8)
        scheduler.joint_config = parse_joint_config({
            "JOINT_ALLOWED_ACTIONS": "copy,stay"})
        scheduler._joint_mode = scheduler.joint_config.scheduler_mode
        for seq in range(3):
            hog = _make_runtime(f"active0_{seq}")
            scheduler.runtime_by_request_id[f"active0_{seq}"] = hog
            scheduler.instances[0].active_decode.append(hog)
        # 实例 1 驻留 ACTIVE 会话吃空容量：切换提交容量失败 → 专门回退。
        kv.prepare_prefill(
            session_id="hog", target_instance_index=1,
            history_tokens=0, trigger_request_id="hog")
        kv.expand_prefill(
            session_id="hog", instance_index=1, context_tokens=7_812_000,
            trigger_request_id="hog")
        scheduler._on_prefill_drain("r0", 1000)
        # 回退成功收尾（非异常路径）：继续方案合法、回退披露行在；预约
        # 与事务账随 drain 终点的常规注销清账（专门路径自闭环——通用
        # 回滚不介入）。
        self.assertEqual(runtime.joint_stage, "D_COMMITTED")
        self.assertTrue(runtime.boundary_decision["fallback"])
        self.assertNotIn("r0", kv._reservations)
        self.assertEqual(len([row for row in scheduler.online_log_rows
                               if row["kind"]
                               == "boundary_commit_fallback"]), 1)


if __name__ == "__main__":
    unittest.main()
