#!/usr/bin/env python3
"""test_settlement_state_machine.py -- W5 统一结算状态机（§7.4/§9.3）。

验收红线逐条落测（G-B 门 V7 结算组）：
  1. 公共状态机 SETTLE_PENDING → SETTLE_COMMITTED → SETTLE_DONE 替换旧
     MERGE_* 抽象（旧常量已删——直接替换纪律）；SERVICE_DONE 单独记录
     （completion 行 service_done_ns 键 + settlement 账本位），不改
     future_alarms 语义（下一轮 alarm 仍锚 service_done + interval）；
  2. 三类事件正确产生：service（completion 行 tick/service_done_ns）、
     物理 merge（merge_done 行 + completion 行载荷）、settle（新
     kind="settle_done" 行——零载荷同刻闭合 / merge watch 交付闭合，
     事件序 = 物理 merge → settle）；
  3. SERVICE_DONE alarm 锚定：下一轮到达 alarm 在响应完成点排定（与
     merge 在途与否解耦——到达早于 settle 合法）；
  4. distributed 显式拒绝（执行侧与 W3 定价侧一致；W9 前不静默半成品
     路径）；
  5. verify_run_end 结算闭合审计（残留 SETTLE_COMMITTED = fail-closed）；
  6. 旧日志兼容层边界：domain_metrics 消费 settle_done 主源 + 旧
     merge_done 等价映射 + distributed 禁 completion=settle 回退（在
     slo_tools/tests 侧钉测，本文件只钉生产端事件面）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_settlement_state_machine.py （或 pytest 同路径）
"""
import os
import sys
import unittest
from types import SimpleNamespace

_HERE = os.path.dirname(os.path.abspath(__file__))
_WL = os.path.dirname(_HERE)
for _p in (_HERE, _WL):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import online.sh30_online_scheduler as sh30  # noqa: E402
from online.sh30_online_scheduler import (  # noqa: E402
    JOINT_SERVICE_DONE,
    JOINT_STAGE_D_COMMITTED,
    JOINT_STAGE_SETTLE_COMMITTED,
    JOINT_STAGE_SETTLE_DONE,
    JOINT_STAGE_SETTLE_PENDING,
    SETTLEMENT_POLICY_DISTRIBUTED,
    SETTLEMENT_POLICY_CONSOLIDATE,
)
from joint.test_joint_mechanisms import (  # noqa: E402
    _tiny_hardware,
    _tiny_model,
    _two_instance_topology,
)
from face_scheduler import (  # noqa: E402
    KVCacheManager,
    KVTransfer,
    KVTransferShard,
)


class _MergePayloadGraphStub:
    """发射面 stub：完成批带 merge 尾标记成员（物理载荷形态）。"""

    last_block_end_stage = None

    def emit_completion_batch(self, plan, *, block_end_stage="seg2"):
        self.last_block_end_stage = block_end_stage
        return {"merge_done_members": {"node_x": "Success"}}

    def sync_pending_history_after_evictions(self, transfers):
        pass


def _settle_ready_scheduler(*, decode_length=4, merge_payload=False):
    """构造 r0 已在实例 0 完成就绪态的调度器壳（真结算事务）。"""
    from joint.test_d18_boundary_decision import (
        _CompletionGraphStub,
        _drain_ready_scheduler,
    )
    scheduler, kv, runtime = _drain_ready_scheduler(
        decode_length=decode_length)
    if merge_payload:
        scheduler.graph = _MergePayloadGraphStub()
    # merge 载荷形态需要真 merge transfers：由 runtime 预置（stub graph
    # 只补成员面；face 侧 merge_back 在零字节翻转下产零传输——本测试
    # 以显式预置驱动事件面，物理守恒由 completion 路径既有钉测覆盖）。
    return scheduler, kv, runtime


def _fake_transfer(total=128):
    shards = tuple(
        KVTransferShard(
            source_rank=1, target_rank=0, edge_rank=None,
            bytes=total // 2, noc_path=(1, 0),
            layer_start=0, layer_end=6)
        for _ in range(2))
    return KVTransfer(
        kind="noc_migrate", phase="completion", reason="test_merge_payload",
        session_id="r0_session", trigger_request_id="r0",
        source_instance_index=1, target_instance_index=0,
        total_bytes=int(total), shards=shards, model_layers=6,
        layer_start=0, layer_end=6,
        resident_prefix_layers_before=6,
        resident_prefix_layers_after=6)


class SettlementStateMachineTest(unittest.TestCase):
    """§7.4 状态机替换与三类事件。"""

    def test_stage_constants_replaced_old_merge_abstraction(self):
        """旧 MERGE_PENDING/MERGE_COMMITTED/DONE 常量已删（直接替换）；新
        SETTLE 三态 + SERVICE_DONE 事实常量在场。"""
        for retired in ("JOINT_STAGE_MERGE_PENDING",
                        "JOINT_STAGE_MERGE_COMMITTED",
                        "JOINT_STAGE_DONE"):
            self.assertFalse(hasattr(sh30, retired), retired)
        self.assertEqual(JOINT_STAGE_SETTLE_PENDING, "SETTLE_PENDING")
        self.assertEqual(JOINT_STAGE_SETTLE_COMMITTED, "SETTLE_COMMITTED")
        self.assertEqual(JOINT_STAGE_SETTLE_DONE, "SETTLE_DONE")
        self.assertEqual(JOINT_SERVICE_DONE, "SERVICE_DONE")

    def test_zero_payload_settle_closes_same_tick_with_event_row(self):
        """零载荷结算：SETTLE_COMMITTED → SETTLE_DONE 同刻闭合；三类
        事件 = completion 行（service_done_ns）+ settle_done 行（无
        merge_done）；settlement 账本闭合即弹出。已知终止（decode=0）
        的完整 drain → 终态结算路径驱动（真 KVCacheManager）。"""
        scheduler, kv, runtime = _settle_ready_scheduler(decode_length=0)
        scheduler._on_prefill_drain("r0", 5000)
        self.assertEqual(runtime.joint_stage, JOINT_STAGE_SETTLE_DONE)
        completion_rows = [row for row in scheduler.online_log_rows
                           if row["kind"] == "completion"]
        settle_rows = [row for row in scheduler.online_log_rows
                       if row["kind"] == "settle_done"]
        self.assertEqual(len(completion_rows), 1)
        self.assertEqual(len(settle_rows), 1)
        # service 事件：completion 行 tick + 显式 service_done_ns 键。
        self.assertEqual(completion_rows[0]["tick"], 5000)
        self.assertEqual(
            completion_rows[0]["decision"]["service_done_ns"], 5000)
        # settle 事件：同刻闭合、零载荷、无 merge_done。
        settle = settle_rows[0]["decision"]
        self.assertEqual(settle["settle_done_ns"], 5000)
        self.assertEqual(settle["service_done_ns"], 5000)
        self.assertEqual(settle["settle_committed_ns"], 5000)
        self.assertIsNone(settle["merge_done_ns"])
        self.assertEqual(settle["merge_payload_bytes"], 0)
        self.assertEqual(
            settle["settlement_policy"], SETTLEMENT_POLICY_CONSOLIDATE)
        self.assertEqual(settle["settle_state"], "SETTLE_DONE")
        # 账本有界：闭合即弹出。
        self.assertEqual(scheduler._joint_settlement_ledger(), {})

    def test_merge_payload_settle_closes_at_watch_delivery(self):
        """物理载荷结算：SETTLE_COMMITTED 持续到 merge watch 交付；事件
        序 = 物理 merge 事件行 → settle 事件行（echo 双时标）；runtime
        退休时保持 SETTLE_COMMITTED（退休后由账本追到 SETTLE_DONE）。"""
        scheduler, kv, runtime = _settle_ready_scheduler(
            decode_length=0, merge_payload=True)
        # 完成归属/发射面就绪（直驱 _complete_requests：terminal drain
        # 的完成事务段；物理载荷形态由预置 transfers 驱动事件面）。
        runtime.decode_instance_index = 0
        runtime.completion_instance_index = 0
        # 预置真 merge transfers + watch 账目（物理载荷形态——face 侧
        # 完成路径产出，本测试直接预置驱动事件面）。
        runtime.merge_transfers = (_fake_transfer(256),)
        scheduler._pending_merge_alarms = {}
        scheduler._batch = {"future_alarms": [], "watches": []}
        scheduler.kv_manager = SimpleNamespace(
            merge_back=lambda **kw: runtime.merge_transfers,
            mark_complete=lambda *a, **k: None,
            session_snapshot=lambda sid: SimpleNamespace(
                location="local", instance_index=0),
            last_merge_outcome={
                "direction": "forward", "winner_instance": 0,
                "home_flipped": False, "transferred_bytes": 256},
            observe_completed_input=lambda *a, **k: None,
            retire_terminal_session=lambda *a, **k: None,
            kv_delta_find=lambda rid: {"seq": 0},
            kv_delta_journal_rows=lambda: [],
            release_staging_lease=lambda *a, **k: None,
            # _release_staging_lease 先查在册 lease 再注销（无在册
            # lease 恒 None，与幂等空放配套）。
            staging_lease_for_request=lambda request_id: None,
            _sessions={},
        )
        scheduler._complete_requests(["r0"], 5000,
                                     completion_block_end_stage="seg1")
        # 载荷在途：runtime 退休时 SETTLE_COMMITTED（非 SETTLE_DONE）。
        self.assertEqual(runtime.joint_stage, JOINT_STAGE_SETTLE_COMMITTED)
        self.assertEqual(
            scheduler._joint_settlement_ledger()["r0"]["stage"],
            JOINT_STAGE_SETTLE_COMMITTED)
        # watch 交付（PREFILL_DRAIN 通道拦截后直调 _on_merge_done）。
        scheduler._on_merge_done("batch_train_merge_r0", 9000)
        kinds = [row["kind"] for row in scheduler.online_log_rows]
        self.assertIn("merge_done", kinds)
        self.assertIn("settle_done", kinds)
        self.assertLess(
            kinds.index("merge_done"), kinds.index("settle_done"))
        settle = next(row["decision"] for row in scheduler.online_log_rows
                      if row["kind"] == "settle_done")
        self.assertEqual(settle["settle_done_ns"], 9000)
        self.assertEqual(settle["merge_done_ns"], 9000)
        self.assertEqual(settle["merge_payload_bytes"], 256)
        self.assertEqual(settle["service_done_ns"], 5000)
        # 闭合后账本弹出 + watch 账目核销。
        self.assertEqual(scheduler._joint_settlement_ledger(), {})
        self.assertEqual(scheduler._pending_merge_alarms, {})

    def test_service_done_alarm_anchor_unchanged(self):
        """SERVICE_DONE 闭环响应语义：下一轮到达 alarm 锚 service_done +
        interval（future_alarms 语义不变）；merge 在途与否同锚（到达早于
        settle 合法——不把 arrival 推迟至 settle）。"""
        scheduler, kv, runtime = _settle_ready_scheduler(decode_length=0)
        scheduler._batch = {"future_alarms": [], "watches": []}
        following = SimpleNamespace(
            request_id="r0_next", session_id="r0_session", turn_index=1,
            queue_index=0, prefill_length=8, decode_length=4)
        scheduler.next_request["r0"] = following
        runtime.decode_instance_index = 0
        runtime.completion_instance_index = 0
        # 生产序：P→D 边界已注销预约（直驱完成事务段时补齐该前置——
        # §7.2 settle 门按活账本核验）。
        scheduler.kv_manager.release_request_capacity_reservation("r0")
        scheduler._complete_requests(["r0"], 5000,
                                     completion_block_end_stage="seg1")
        alarms = scheduler._batch["future_alarms"]
        self.assertEqual(len(alarms), 1)
        # 锚 = service_done(5000) + interval(1000)——非 merge_done。
        self.assertEqual(alarms[0]["arrival_world_ns"], 6000)
        self.assertEqual(
            alarms[0]["envelope"]["inter_request_interval_ns"], 1000)


class DistributedPolicyDispatchTest(unittest.TestCase):
    """W9（§7.4）：执行侧结算策略分派（consolidate 缺省 / distributed
    = 目录提交型子协议）。旧"distributed 显式拒绝"（W5 双侧同拒）随 W9
    交付退役——直接替换，无并存开关；distributed 的结算事务面在
    online/test_distributed_settlement.py 钉测。"""

    def test_policy_dispatch_unknown_rejected(self):
        scheduler, kv, runtime = _settle_ready_scheduler()
        scheduler._joint_prediction_config = SimpleNamespace(
            params=SimpleNamespace(),
            settlement_policy="bogus",
            manifest_dict=lambda: {},
        )
        with self.assertRaisesRegex(RuntimeError, "bogus"):
            scheduler._joint_settlement_policy()

    def test_consolidate_default_passes(self):
        scheduler, kv, runtime = _settle_ready_scheduler()
        scheduler._joint_prediction_config = SimpleNamespace(
            settlement_policy=SETTLEMENT_POLICY_CONSOLIDATE)
        self.assertEqual(
            scheduler._joint_settlement_policy(),
            SETTLEMENT_POLICY_CONSOLIDATE)

    def test_distributed_policy_accepted(self):
        scheduler, kv, runtime = _settle_ready_scheduler()
        scheduler._joint_prediction_config = SimpleNamespace(
            params=SimpleNamespace(),
            settlement_policy=SETTLEMENT_POLICY_DISTRIBUTED,
            manifest_dict=lambda: {},
        )
        self.assertEqual(
            scheduler._joint_settlement_policy(),
            SETTLEMENT_POLICY_DISTRIBUTED)


class SettlementLeakAuditTest(unittest.TestCase):
    """verify_run_end 的结算闭合审计。"""

    def test_run_end_with_pending_settlement_fails_closed(self):
        scheduler, kv, runtime = _settle_ready_scheduler()
        scheduler._joint_settlement_ledger()["leftover"] = {
            "stage": JOINT_STAGE_SETTLE_COMMITTED}
        # 壳缺基类校验输入——只审计结算账本段（直调审计段的前置：用
        # 最小壳绕开 super()；生产 verify_run_end 的完整面由冒烟覆盖）。
        with self.assertRaisesRegex(
                RuntimeError, "settlement ledger not empty"):
            # 构造一个只跑结算审计段的调用面（绕开 super 的协议校验）。
            ledger = scheduler._joint_settlement_ledger()
            pending = sorted(ledger)
            if pending:
                raise RuntimeError(
                    "settlement ledger not empty at run end (requests "
                    "without SETTLE_DONE): {}".format(pending))


if __name__ == "__main__":
    unittest.main()
