#!/usr/bin/env python3
"""test_remote_credit_staging_lease.py -- D09/D20 双缓冲暂存 lease 生命
周期钉测（审查 P1-2：decode 双缓冲 HBM 暂存预约在读流前被注销）。

被测语义（PARTIAL 基 remote-read，真实 KVCacheManager + 真实列车发射
+ 真实完成/合并链；夹具 = test_prefill_remote_read_lifecycle 同底盘）：

  1. drain 转换：预约注销时暂存分量原样转入
     ``(request_id, boundary_epoch)`` 持有的执行期 lease——reserved
     视图无空洞（审查复现的 drain 后 +32768/rank 提前回流在此 fail；
     预约语义=准入容量、lease=执行期持有，分开记账）；
  2. 最后 credit tile 消费（Tj 核销边界）注销：部分消费在册、全消费
     归零，reserved 视图同步回零；完成结算收敛路幂等空放；
  3. 异常回滚：P→D 边界窗口内异常把 lease 与阶段引用一并注销，不
     泄漏也不落决策行；
  4. lease API 守卫：同键同字节幂等、同请求跨 epoch 双持 fail-closed、
     epoch 失配释放 fail-closed、字节/tile 元数据校验。
  5. 装备未注入替身面：预约无暂存分量时按计划声明口径持有（生产恒
     注入并 fail-closed 校验；此路保证读流在册期间暂存恒有持有者）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 -m pytest online/test_remote_credit_staging_lease.py -q
"""
import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import kv_cache_shard_bytes_for_tokens  # noqa: E402
from online.test_prefill_remote_read_lifecycle import (  # noqa: E402
    EXEC_INSTANCE,
    HOME_INSTANCE,
    RID,
    _LifecycleRun,
    _make_manager,
    _model,
)


class _ConfiguredLifecycleRun(_LifecycleRun):
    """生产装配口径：注入 D09 tile 协议参数（sh30 __init__ 同款解析值
    ——预约携带真实暂存分量 32768/rank，lease 原样承接）。基类夹具
    未注入 = 替身面，走计划声明口径（FallbackLeaseTest 覆盖）。"""

    def __init__(self):
        super().__init__()
        self.manager._remote_read_tile_tokens = 256
        self.manager._remote_read_kv_token_bytes = (
            kv_cache_shard_bytes_for_tokens(
                self.model, 1, self.manager.tp_degree)[0])


class DrainConversionTest(unittest.TestCase):
    """drain 边界：预约暂存分量 → 执行期 lease，reserved 视图无空洞。"""

    def setUp(self):
        self.run = _ConfiguredLifecycleRun()
        self.run.admit()
        self.manager = self.run.manager
        self.runtime = self.run.runtime

    def test_lease_holds_reservation_staging_after_drain(self):
        reservation = self.manager._reservations[RID]
        staging = tuple(reservation.staging_shard_bytes)
        # D09 prefill 侧 lease 覆盖（2026-09-27）：P 阶段预约不再携带
        # 暂存分量——准入起的持有者 = prefill 读流 lease（窗基数裁决，
        # 双缓冲字节经 lease 进真实占用视图，预约/lease 不双计）。
        self.assertEqual(staging, ())
        prefill_leases = [
            lease for lease in self.manager.staging_leases()
            if lease.boundary_epoch == self.runtime.joint_boundary_epoch - 1
            or lease.boundary_epoch == self.runtime.joint_boundary_epoch]
        reserved_before = tuple(
            self.manager._reserved_bytes_by_tp_rank(EXEC_INSTANCE))
        next_allocation_before = (
            self.manager.peek_next_staging_allocation_id())
        remaining_before = tuple(
            self.manager._effective_remaining_by_tp_rank(EXEC_INSTANCE))
        self.run.drain()
        # 预约已注销（准入容量语义终结）……
        self.assertNotIn(RID, self.manager._reservations)
        # ……读计划在册（执行期读流存续）……
        self.assertIsNotNone(self.runtime.remote_read_credit_plan)
        # ……prefill lease 已在 drain 注销（读流正常关闭点 = 末 tile
        # 消费），decode lease 按计划声明口径承接 (request, epoch)。
        leases = self.manager.staging_leases()
        self.assertEqual(len(leases), 1)
        lease = leases[0]
        self.assertEqual(lease.request_id, RID)
        self.assertEqual(lease.allocation_id, next_allocation_before)
        self.assertEqual(
            self.manager.peek_next_staging_allocation_id(),
            next_allocation_before + 1)
        self.assertEqual(lease.lease_id,
                         "staging_lease:{}#{}".format(
                             RID, self.runtime.joint_boundary_epoch))
        self.assertEqual(lease.boundary_epoch,
                         self.runtime.joint_boundary_epoch)
        self.assertEqual(lease.staging_shard_bytes, (32768, 32768))
        self.assertEqual(lease.staging_tile_tokens, 256)
        self.assertEqual(lease.instance_index, EXEC_INSTANCE)
        self.assertEqual(lease.session_id, self.runtime.session_id)
        # 字节守恒（分量口径）：P 阶段 lease 与 drain 后 decode lease 的
        # 暂存字节同额承接（本夹具 P/decode 基数同源）——reserved 视图
        # 恰含暂存字节（审查复现的 reserved=0 在此 fail）。
        staging = lease.staging_shard_bytes
        self.assertTrue(prefill_leases or reserved_before)
        reserved_after = tuple(
            self.manager._reserved_bytes_by_tp_rank(EXEC_INSTANCE))
        self.assertEqual(reserved_after, staging)
        # 暂存字节不再显示为可用：每 rank 可用 = 裸剩余 − 暂存（审查
        # 复现的可用 +32768/rank 暂存整份回流在此 fail）。
        remaining_after = tuple(
            self.manager._effective_remaining_by_tp_rank(EXEC_INSTANCE))
        ranks = self.manager.topology.instance(EXEC_INSTANCE).ranks
        raw_after = tuple(
            self.manager._rank_states[rank].remaining_bytes
            for rank in ranks)
        self.assertEqual(
            remaining_after,
            tuple(raw - staging_value
                  for raw, staging_value in zip(raw_after, staging)))
        self.assertNotEqual(reserved_after, (0, 0))
        # 两阶段 acquire：准入（prefill 读流 lease）+ drain（decode
        # credit lease，epoch = 边界值）。
        rows = self.run._log_rows("staging_lease_acquire")
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[-1]["decision"]["staging_shard_bytes"], list(staging))

    def test_degenerate_plan_releases_staging_with_reservation(self):
        # 计划缺位（无读流：home == exec 退化）= 无暂存使用——暂存随
        # 预约合法消失，不建立 lease（无持有需求即无持有）。退化在
        # drain 建计划点判生（计划字段在转换点前才冻结），打桩捕获。
        scheduler = self.run.scheduler
        original_plan = scheduler._joint_remote_read_credit_plan
        scheduler._joint_remote_read_credit_plan = (
            lambda runtime, exec_instance: None)
        try:
            self.run.drain()
        finally:
            scheduler._joint_remote_read_credit_plan = original_plan
        self.assertIsNone(self.runtime.remote_read_credit_plan)
        self.assertNotIn(RID, self.manager._reservations)
        self.assertEqual(self.manager.staging_leases(), ())


class LastTileConsumptionReleaseTest(unittest.TestCase):
    """最后一个 credit tile 消费（Tj 核销边界）= 读流正常关闭点。"""

    def setUp(self):
        self.run = _ConfiguredLifecycleRun()
        self.run.admit()
        self.run.drain()
        self.manager = self.run.manager
        self.runtime = self.run.runtime

    def test_lease_survives_partial_consumption(self):
        # T1：5/8 步（T_max 截断）——读流未完，暂存必须仍在册。
        self.run._emit_and_settle_train(2000, 3000, ())
        self.assertEqual(self.runtime.decode_tokens_consumed, 5)
        self.assertEqual(len(self.manager.staging_leases()), 1)

    def test_lease_released_when_last_credit_tile_consumed(self):
        self.run._emit_and_settle_train(2000, 3000, ())
        self.run._emit_and_settle_train(3500, 4000, (RID,))
        self.assertEqual(self.runtime.decode_tokens_consumed, 8)
        self.assertEqual(self.manager.staging_leases(), ())
        # 暂存持有者注销后 reserved 视图同步回零（执行端无其他预约）。
        self.assertEqual(
            tuple(self.manager._reserved_bytes_by_tp_rank(EXEC_INSTANCE)),
            (0, 0))
        rows = self.run._log_rows("staging_lease_release")
        # 两阶段释放：drain（prefill 读流关闭）+ 末 credit tile 消费。
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[0]["decision"]["reason"],
            "prefill_read_stream_consumed")
        self.assertEqual(
            rows[1]["decision"]["reason"], "credit_stream_fully_consumed")

    def test_completion_settlement_after_tile_release_is_idempotent(self):
        # 正常路径：最后 tile 消费点已注销，完成结算收敛路幂等空放
        # （不产生第三条释放行、不复活 lease、无残留）。
        self.run.run_decode_trains()
        self.run.complete_and_merge()
        self.assertEqual(self.manager.staging_leases(), ())
        self.assertEqual(len(self.run._log_rows("staging_lease_release")), 2)


class SettlementConvergenceTest(unittest.TestCase):
    """完成结算收敛注销路（tile 消费点之外的残漏闭合；单一出口）。"""

    def setUp(self):
        self.run = _ConfiguredLifecycleRun()
        self.run.admit()
        self.run.drain()

    def test_settlement_releases_once_then_idempotent_noop(self):
        released = self.run.scheduler._release_staging_lease(
            self.run.runtime, tick=4500,
            reason="request_completion_settlement")
        self.assertTrue(released)
        self.assertEqual(self.run.manager.staging_leases(), ())
        again = self.run.scheduler._release_staging_lease(
            self.run.runtime, tick=4501,
            reason="request_completion_settlement")
        self.assertFalse(again)
        self.assertEqual(self.run.manager.staging_leases(), ())
        rows = self.run._log_rows("staging_lease_release")
        # drain 的 prefill 释放 + 本结算释放；幂等二次空放不落行。
        self.assertEqual(len(rows), 2)
        self.assertEqual(
            rows[-1]["decision"]["reason"], "request_completion_settlement")


class RollbackPathTest(unittest.TestCase):
    """异常/回滚：边界窗口内异常把 lease 与阶段引用一并注销。"""

    def test_boundary_exception_rolls_back_lease_without_log_row(self):
        run = _ConfiguredLifecycleRun()
        run.admit()
        scheduler = run.scheduler
        manager = run.manager
        runtime = run.runtime

        def _boom(*args, **kwargs):
            raise RuntimeError("join decision exploded")

        # 转换点在边界 try 块末段——join 决策发射前，异常路径必经
        # except 清理（引用 + lease 幂等注销后重抛）。
        scheduler._emit_join_decision = _boom
        runtime.queued_chunk_load_ns = 0
        with self.assertRaises(RuntimeError):
            scheduler._on_prefill_drain(RID, 1000)
        self.assertEqual(manager.staging_leases(), ())
        self.assertIsNone(runtime.joint_stage_reference_id)
        self.assertNotIn(RID, manager._reservations)
        # 回滚语境不落决策行（日志面故障不得掩盖原始异常）——drain
        # 前段的 prefill 读流 lease 正常关闭行除外（非回滚产物）。
        self.assertEqual(
            [row for row in scheduler.online_log_rows
             if row["kind"] == "staging_lease_release"
             and row["decision"]["reason"]
             != "prefill_read_stream_consumed"], [])


class FallbackLeaseTest(unittest.TestCase):
    """装备未注入替身面：预约无暂存分量时按计划声明口径持有。"""

    def test_uninjected_manager_leases_plan_declared_bytes(self):
        run = _LifecycleRun()
        run.admit()
        run.drain()
        manager = run.manager
        plan = run.runtime.remote_read_credit_plan
        self.assertIsNotNone(plan)
        leases = manager.staging_leases()
        self.assertEqual(len(leases), 1)
        lease = leases[0]
        self.assertEqual(
            lease.staging_shard_bytes,
            tuple(plan["staging_bytes_per_rank"]
                  for _ in range(manager.tp_degree)))
        self.assertEqual(lease.staging_tile_tokens, plan["tile_tokens"])
        # 可用容量视图扣减 lease 字节（不回流为可挪用空间）。
        reserved = tuple(
            manager._reserved_bytes_by_tp_rank(EXEC_INSTANCE))
        self.assertEqual(
            reserved, lease.staging_shard_bytes)
        run.run_decode_trains()
        run.complete_and_merge()
        self.assertEqual(manager.staging_leases(), ())


class StagingLeaseApiGuardsTest(unittest.TestCase):
    """KVCacheManager lease API 守卫（幂等/冲突/元数据 fail-closed）。"""

    def setUp(self):
        self.model = _model()
        _hw, _topo, self.manager = _make_manager(self.model)
        self.terms = dict(
            session_id="s9",
            instance_index=HOME_INSTANCE,
            boundary_epoch=3,
            staging_shard_bytes=(32768, 32768),
            staging_tile_tokens=256,
        )

    def _acquire(self, request_id="r9", **overrides):
        return self.manager.acquire_staging_lease(
            request_id, **{**self.terms, **overrides})

    def test_duplicate_acquire_with_same_terms_is_idempotent(self):
        first = self._acquire()
        first_allocation = self.manager.staging_lease_for_request(
            "r9").allocation_id
        second = self._acquire()
        self.assertEqual(first, second)
        self.assertEqual(
            self.manager.staging_lease_for_request("r9").allocation_id,
            first_allocation)
        self.assertEqual(self.manager.peek_next_staging_allocation_id(), 2)
        self.assertEqual(len(self.manager.staging_leases()), 1)

    def test_staging_identity_state_preserves_live_lease_and_counter(self):
        self._acquire()
        state = self.manager.staging_allocation_state()
        self.assertEqual(state["next_staging_allocation_id"], 2)
        self.assertEqual(state["live_leases"], [{
            "lease_id": "staging_lease:r9#3", "allocation_id": 1}])
        self.manager._next_staging_allocation_id = 1
        self.manager.restore_staging_allocation_state(state)
        self.assertEqual(self.manager.peek_next_staging_allocation_id(), 2)
        self.manager.release_staging_lease("r9")
        self._acquire(boundary_epoch=4)
        self.assertEqual(
            self.manager.staging_lease_for_request("r9").allocation_id, 2)

    def test_staging_identity_restore_rejects_live_lease_rename(self):
        self._acquire()
        state = self.manager.staging_allocation_state()
        state["live_leases"][0]["allocation_id"] = 2
        with self.assertRaises(RuntimeError):
            self.manager.restore_staging_allocation_state(state)

    def test_second_epoch_for_same_request_fails_closed(self):
        self._acquire()
        with self.assertRaises(RuntimeError):
            self._acquire(boundary_epoch=4)

    def test_conflicting_terms_for_same_key_fail_closed(self):
        self._acquire()
        with self.assertRaises(RuntimeError):
            self._acquire(staging_shard_bytes=(16384, 16384))

    def test_release_epoch_mismatch_fails_closed(self):
        self._acquire()
        with self.assertRaises(RuntimeError):
            self.manager.release_staging_lease("r9", boundary_epoch=7)
        # 失配释放不产生副作用。
        self.assertEqual(len(self.manager.staging_leases()), 1)

    def test_release_then_reacquire_new_epoch(self):
        lease_id = self._acquire()
        first_allocation = self.manager.staging_lease_for_request(
            "r9").allocation_id
        self.assertTrue(self.manager.release_staging_lease(
            "r9", boundary_epoch=self.terms["boundary_epoch"]))
        self.assertEqual(self.manager.staging_leases(), ())
        # 已注销后重复释放幂等空放。
        self.assertFalse(self.manager.release_staging_lease(
            "r9", boundary_epoch=self.terms["boundary_epoch"]))
        # 新 epoch 重新获取合法（同请求串行：旧 lease 已注销）。
        new_id = self._acquire(boundary_epoch=5)
        self.assertEqual(new_id, "staging_lease:r9#5")
        self.assertNotEqual(new_id, lease_id)
        self.assertEqual(
            self.manager.staging_lease_for_request("r9").allocation_id,
            first_allocation + 1)

    def test_lease_bytes_enter_reserved_view_and_exit_on_release(self):
        self._acquire()
        self.assertEqual(
            tuple(self.manager._reserved_bytes_by_tp_rank(HOME_INSTANCE)),
            (32768, 32768))
        self.assertEqual(
            tuple(self.manager._reserved_bytes_by_tp_rank(EXEC_INSTANCE)),
            (0, 0))
        self.manager.release_staging_lease("r9")
        self.assertEqual(
            tuple(self.manager._reserved_bytes_by_tp_rank(HOME_INSTANCE)),
            (0, 0))

    def test_invalid_metadata_fails_closed(self):
        with self.assertRaises(ValueError):
            self._acquire(staging_shard_bytes=())
        with self.assertRaises(ValueError):
            self._acquire(staging_shard_bytes=(32768,))
        with self.assertRaises(ValueError):
            self._acquire(staging_shard_bytes=(0, 32768))
        with self.assertRaises(ValueError):
            self._acquire(staging_tile_tokens=0)
        with self.assertRaises(ValueError):
            self._acquire(instance_index=99)


if __name__ == "__main__":
    unittest.main()
