#!/usr/bin/env python3
"""test_joint_remote_settlement.py -- C14 结算与暂存计费一致性的零后端
结构断言（2026-09-22；设计文档 §6.2 完成检查清单/§2.4/§4.2；仓库设计
方案 §2.2 结算行/§3.2；C12 冻结 §15.6 暂存规格/§15.8 STAGING_WRITE
裁定；C13 交接 #handoff/#copy-stream 科目不冲突）。

（2026-10-14 远读退役改写：原 remote-read 前向/翻转结算行、F15 合成
平局、远读源端保护与合并目标容量前瞻三组用例已删除——增量布局
（H@home + U@exec 分置）只能由 remote-read 准入产生，其后合并
恒 stay/零字节翻转（D8/D-1：完整并集恒成立，前向/反向真实迁移腿
不可达）；copy 链的边界语义由 test_c4_boundary_full_surface
的完整型用例族承接。保留面：stay/copy 结算行、kv_delta_find 访问
器、重复结算回调 fail-closed、SH merge watch 闭合门。）

覆盖（本卡测试条款）：
  A. kv_delta_journal（FS 披露接口，C14 步骤 5；C16 消费面字段）：
    1. stay 本地提交行（最小行；新会话路径的 home 建立）；
    2. copy 零字节结算行（home 侧残量按交接 journal 口径——已交接
       释放字节不计入保留量，不与 #handoff/#copy-stream 冲突）；
    3. kv_delta_find 访问器（逐请求命中/未知 None/单调 seq）；
  B. 失败注入（设计文档 §6.2 失败分支闭合）：
    4. 结算重复回调：同请求二次 merge_back → 版本键 raise，journal 恰
       一行、败者侧无二次释放、账本不变（copy 轮承载）；
  D. SH merge watch 结算闭合门（C14 步骤 1 完成检查的 watch 侧）：
    5. watch 交付 ⇔ journal 行在案；缺行 fail-closed；替身缺接口
       fail-loud（与 last_merge_outcome 同款口径）。
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
    KVCapacityError,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_tokens,
)
from online.sh30_online_scheduler import Sh30OnlineScheduler  # noqa: E402


# ------------------------------------------------------------- FS 夹具 --

def _settlement_manager(*, capacity_bytes: int = 2000, layers: int = 4):
    """结算夹具：2×2 mesh、双 2-rank 实例；每层每 token 每 rank 4B
    （全层 16B/token/rank），逐字节可手算（与 test_face_scheduler 的
    merge v2 fixture 同源口径）。"""
    hardware = FaceHardware(
        mesh_rows=2,
        mesh_cols=2,
        local_hbm_capacity_bytes=capacity_bytes,
        local_hbm_bandwidth_gbps=1.0,
        d2d_bandwidth_gbps=2.0,
        peak_perf_tflops=1.0,
        d2d_latency_ns=0,
        local_hbm_latency_ns=0,
    )
    model = FaceModel(
        layers=layers,
        hidden_size=4,
        ffn_size=4,
        num_heads=2,
        vocab_size=4,
        bytes_per_elem=1,
        mlp_variant="gelu",
    )
    topology = build_instances(
        hardware,
        (
            FaceInstanceSpec("ins0", "1", (0, 1)),
            FaceInstanceSpec("ins1", "2", (2, 3)),
        ),
    )
    return model, KVCacheManager(topology, model)


def _seed_completed_session(manager, *, session_id, instance_index,
                            context_tokens):
    manager.prepare_prefill(
        session_id=session_id,
        target_instance_index=instance_index,
        history_tokens=0,
        trigger_request_id=f"{session_id}_seed")
    manager.expand_prefill(
        session_id=session_id, instance_index=instance_index,
        context_tokens=context_tokens,
        trigger_request_id=f"{session_id}_seed")
    manager.mark_complete(session_id, context_tokens)


def _seed_active_session(manager, *, session_id, instance_index,
                        context_tokens):
    """活跃（未完成）会话：不可逐出的 blocker 形态。"""
    manager.prepare_prefill(
        session_id=session_id,
        target_instance_index=instance_index,
        history_tokens=0,
        trigger_request_id=f"{session_id}_seed")
    manager.expand_prefill(
        session_id=session_id, instance_index=instance_index,
        context_tokens=context_tokens,
        trigger_request_id=f"{session_id}_seed")


def _kv_bytes(manager, instance_index):
    return tuple(
        manager._rank_states[rank].kv_cache_bytes
        for rank in manager.topology.instance(instance_index).ranks)


def _copy_round(manager, *, base_tokens, increment_tokens,
                trigger_request_id="r1", session_id="s",
                home_index=0, exec_index=1):
    """构造一轮 copy（LOCAL 基 @home → 完整工作副本 @exec；C13 逐块
    交接在 expand 轮内释放 home 侧基）并返回 (home_side_sum,
    exec_side_sum)——两侧实际保留量（结算前）。"""
    _seed_completed_session(
        manager, session_id=session_id, instance_index=home_index,
        context_tokens=base_tokens)
    manager.prepare_prefill(
        session_id=session_id, target_instance_index=exec_index,
        history_tokens=base_tokens,
        trigger_request_id=trigger_request_id, action="copy")
    manager.expand_prefill(
        session_id=session_id, instance_index=exec_index,
        context_tokens=base_tokens + increment_tokens,
        trigger_request_id=trigger_request_id)
    session = manager._sessions[session_id]
    home_side = sum(manager._rank_states[rank].kv_cache_bytes
                    for rank in manager.topology.instance(
                        home_index).ranks)
    exec_side = sum(session.shard_bytes)
    return home_side, exec_side


# ------------------------------------------------------ A. journal 行 --

class KvDeltaJournalForwardRowTests(unittest.TestCase):
    """A1/A2/A3：三方向结算行的字段面（C16 消费字段 + 披露键）。"""

    def test_stay_row_local_commit_minimal(self):
        # stay：本地提交行（working_kind=None、无迁移、无传输）。
        model, manager = _settlement_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=0)

        self.assertEqual(transfers, ())
        row = manager.kv_delta_find("r1")
        self.assertEqual(row["direction"], "stay")
        self.assertIsNone(row["working_kind"])
        self.assertFalse(row["zero_byte_flip"])
        self.assertIsNone(row["winner_instance"])
        self.assertIsNone(row["loser_instance"])
        self.assertEqual(row["home_before"], 0)
        # A12'：stay 无胜者 ⇒ home 不变——home_after = home_before
        # （原 None 会被消费端当独立 home 计入集合，乒乓指标假阳性）。
        self.assertEqual(row["home_after"], 0)
        self.assertFalse(row["home_migration"])
        self.assertEqual(row["transferred_bytes"], 0)
        self.assertEqual(row["home_side_retained_bytes"], 320)
        self.assertEqual(row["exec_side_retained_bytes"], 0)
        self.assertEqual(row["new_tokens"], 0)
        self.assertEqual(row["staging_return_bytes"], 0)

    def test_copy_zero_byte_row_uses_handoff_residual_not_base_derived(self):
        # copy 零字节翻转：home 侧残量按交接 journal 口径（轮内已逐块
        # 释放 ⇒ 结算时刻保留量 = 0，而非 base 前缀推导值 320）——
        # "不重复释放已交接源块"的 journal 侧镜像（C13 科目不冲突）。
        model, manager = _settlement_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r1", action="copy")
        # prefill drain 边界结算逐块交接（home 侧全释放）。
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=10,
            trigger_request_id="r1")

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=0)

        self.assertEqual(transfers, ())  # 零字节结算：无传输
        row = manager.kv_delta_find("r1")
        self.assertEqual(row["working_kind"], "copy")
        self.assertEqual(row["direction"], "reverse")
        self.assertTrue(row["zero_byte_flip"])
        self.assertEqual(row["transferred_bytes"], 0)
        self.assertEqual(row["home_side_retained_bytes"], 0)
        self.assertEqual(row["exec_side_retained_bytes"], 320)
        self.assertTrue(row["home_migration"])
        self.assertEqual(row["home_after"], 1)
        # home 侧零驻留（轮内已交接释放；merge 不二次扣减——账本守恒）；
        # exec 侧持并集 kv(10) = 160B/rank。
        self.assertEqual(_kv_bytes(manager, 0), (0, 0))
        self.assertEqual(_kv_bytes(manager, 1), (160, 160))

    def test_kv_delta_find_accessor_semantics(self):
        # 逐请求命中（最新行）、未知 None、seq 单调；多轮会话两行可分
        # （第二轮 copy：交接轮内释放 home，结算为零字节翻转）。
        model, manager = _settlement_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.merge_back(session_id="s", trigger_request_id="r1",
                           new_tokens=0)
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r2", action="copy")
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=15,
            trigger_request_id="r2")
        manager.merge_back(session_id="s", trigger_request_id="r2",
                           new_tokens=5)

        self.assertIsNone(manager.kv_delta_find("unknown"))
        first = manager.kv_delta_find("r1")
        second = manager.kv_delta_find("r2")
        self.assertEqual(first["direction"], "stay")
        self.assertEqual(second["direction"], "reverse")
        self.assertEqual(second["seq"], 1)
        self.assertLess(first["seq"], second["seq"])
        seqs = [row["seq"] for row in manager.kv_delta_journal]
        self.assertEqual(seqs, list(range(len(seqs))))
        self.assertEqual(len(manager.kv_delta_journal), 2)


# -------------------------------------------------- B. 失败注入三件 --

class RemoteSettlementFailureInjectionTests(unittest.TestCase):
    """设计文档 §6.2 完成检查清单的失败分支闭合。

    C14 条款：结算重复回调。（原源端保护泄漏与合并目标容量前瞻两用
    例已随 2026-10-14 远读退役删除：源端基础保护与 forward/reverse
    容量前瞻的承载形态 = 远读增量布局的分裂末态（远读退役时起
    不可构造）——copy 轮的 home 侧基在交接轮内释放（journal 口径），
    合并恒 stay/零字节翻转，无跨实例容量准备面。）
    """

    def test_duplicate_settlement_callback_fails_closed_no_double_release(self):
        # 结算重复回调注入：同请求二次 merge_back → 版本键 raise；
        # journal 恰一行、败者侧无二次释放（账本不变）。copy 轮承载：
        # 结算为零字节翻转（exec 持并集、home 侧零驻留）。
        model, manager = _settlement_manager()
        _copy_round(manager, base_tokens=10, increment_tokens=5)
        manager.merge_back(session_id="s", trigger_request_id="r1",
                           new_tokens=5)
        settled_home = _kv_bytes(manager, 0)
        settled_exec = _kv_bytes(manager, 1)
        self.assertEqual(settled_home, (0, 0))
        self.assertEqual(settled_exec, (240, 240))

        with self.assertRaisesRegex(
                RuntimeError, "settled twice"):
            manager.merge_back(session_id="s", trigger_request_id="r1",
                               new_tokens=5)

        rows = [row for row in manager.kv_delta_journal
                if row["trigger_request_id"] == "r1"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(_kv_bytes(manager, 0), settled_home)
        self.assertEqual(_kv_bytes(manager, 1), settled_exec)

# ---------------------------------------------- D. SH watch 闭合门 --

class _RecordingFlows:
    def __init__(self):
        self.released = []

    def release_owner(self, owner):
        self.released.append(owner)


class _SettledKvManager:
    """带 kv_delta_find 的最小账本替身（可注入行/缺行）。"""

    def __init__(self, row):
        self._row = row

    def kv_delta_find(self, trigger_request_id):
        return self._row


class _NoInterfaceKvManager:
    """缺 kv_delta_find 接口的替身（F6 销账后 = fail-loud 锚：Attribute
    Error，不再有软门跳过）。"""


def _bare_scheduler(kv_manager):
    scheduler = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    scheduler.kv_manager = kv_manager
    scheduler._joint_flows = _RecordingFlows()
    scheduler._pool_ports = _RecordingFlows()
    # 对齐 __init__ 初值（F6 销账：软门已删，_release_transfer_flows
    # 直达 _hbm_ports、闭合门后直达 _quota_tracker——漏设 = AttributeError）。
    scheduler._hbm_ports = _RecordingFlows()
    scheduler._quota_tracker = None
    scheduler._pending_merge_alarms = {
        "batch_train_merge_r1": {
            "request_id": "r1",
            "session_id": "session_x",
            "following_request_id": "r2",
            "next_arrival_world_ns": 1_500_000,
        },
    }
    scheduler.online_log_count = 0
    scheduler.decision_log_sink = None
    scheduler.online_log_rows = []
    return scheduler


class MergeWatchSettlementClosureTests(unittest.TestCase):
    """C14：merge watch 交付 ⇔ FS 结算事实在案（闭合门三态）。"""

    def test_watch_delivery_with_settlement_row_releases_flows(self):
        row = {"seq": 0, "session_id": "session_x",
               "trigger_request_id": "r1", "direction": "forward"}
        scheduler = _bare_scheduler(_SettledKvManager(row))
        # W5（§7.4 重锚）：watch 交付前置 = settlement 记录在案（生产
        # 在 _complete_requests 的 SETTLE_COMMITTED 段登记；缺记录 =
        # watch 通道与结算账本脱钩，fail-closed）。
        scheduler._joint_settlement_ledger()["r1"] = {
            "stage": "SETTLE_COMMITTED",
            "settlement_policy": "consolidate",
            "service_done_ns": 1_000_000,
            "settle_committed_ns": 1_000_000,
            "merge_payload_bytes": 128,
            "merge_done_ns": None,
            "settle_done_ns": None,
        }
        scheduler._on_merge_done("batch_train_merge_r1", 2_000_000)
        # R15：#merge 流注销恰一次。
        self.assertEqual(
            scheduler._joint_flows.released, ["r1#merge"])
        self.assertEqual(
            scheduler._pool_ports.released, ["r1#merge"])
        # C3b 披露行照落（kind=merge_done）+ W5 settle_done 事件行
        # （§7.4 三类事件：service（completion 行）/物理 merge（本行）/
        # settle（settle_done 行——echo service/merge 双时标））。
        self.assertEqual(len(scheduler.online_log_rows), 2)
        logged = scheduler.online_log_rows[0]
        self.assertEqual(logged["kind"], "merge_done")
        self.assertEqual(logged["tick"], 2_000_000)
        self.assertEqual(
            logged["decision"]["merge_done_ns"], 2_000_000)
        settle_row = scheduler.online_log_rows[1]
        self.assertEqual(settle_row["kind"], "settle_done")
        self.assertEqual(
            settle_row["decision"]["settle_done_ns"], 2_000_000)
        self.assertEqual(
            settle_row["decision"]["merge_done_ns"], 2_000_000)
        self.assertEqual(
            settle_row["decision"]["service_done_ns"], 1_000_000)
        self.assertEqual(
            settle_row["decision"]["settlement_policy"], "consolidate")
        self.assertEqual(
            settle_row["decision"]["settle_state"], "SETTLE_DONE")
        # 结算账本闭合即弹出（有界）。
        self.assertEqual(scheduler._joint_settlement_ledger(), {})
        self.assertEqual(
            scheduler._pending_merge_alarms, {})

    def test_watch_delivery_without_settlement_row_fails_closed(self):
        scheduler = _bare_scheduler(_SettledKvManager(None))
        with self.assertRaisesRegex(
                RuntimeError, "kv_delta_journal closure failure"):
            scheduler._on_merge_done("batch_train_merge_r1", 2_000_000)
        # fail-closed 于流注销之前：不释放、不落披露行。
        self.assertEqual(scheduler._joint_flows.released, [])
        self.assertEqual(scheduler.online_log_rows, [])

    def test_watch_delivery_missing_interface_stub_fails_loud(self):
        # F6 销账：getattr 软门已删——替身缺 kv_delta_find 接口现在 =
        # AttributeError（fail-loud，接口装配不可静默缺席；真
        # KVCacheManager 恒定义该接口，face_scheduler kv_delta_find）。
        # 断言锚定缺失接口名（别的属性先炸不许冒名通过）。
        scheduler = _bare_scheduler(_NoInterfaceKvManager())
        with self.assertRaisesRegex(AttributeError, "kv_delta_find"):
            scheduler._on_merge_done("batch_train_merge_r1", 2_000_000)
        # fail-loud 于流注销之前（与 fail-closed 同款次序纪律）。
        self.assertEqual(scheduler._joint_flows.released, [])


if __name__ == "__main__":
    unittest.main()
