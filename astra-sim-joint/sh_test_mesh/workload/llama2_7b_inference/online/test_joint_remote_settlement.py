#!/usr/bin/env python3
"""test_joint_remote_settlement.py -- C14 remote 完整结算与暂存计费一致性
的零后端结构断言（2026-09-22；设计文档 §6.2 remote 完成检查清单/§2.4/
§4.2；仓库设计方案 §2.2 remote 行/§3.2；C12 冻结 §15.6 暂存规格/§15.8
STAGING_WRITE 裁定 + GAPS G1 处置（PROVENANCE §20.1 = (a) 补价移交 C2
rider，本卡零 JCM 改动）；C13 交接 #handoff/#copy-stream 科目不冲突）。

覆盖（本卡测试条款）：
  A. kv_delta_journal（FS 披露接口，C14 步骤 5；C16 消费面字段）：
    1. remote-read 前向结算行全字段（方向/零字节/两侧实际保留量/
       home 迁移/传输字节/staging_return_bytes=0 = F14 无操作披露位）；
    2. remote-read 翻转结算行 home 迁移轨迹（home_before/home_after）；
    3. stay 本地提交行（最小行；新会话路径的 home 建立）；
    4. copy 零字节结算行（home 侧残量按交接 journal 口径——已交接
       释放字节不计入保留量，不与 #handoff/#copy-stream 冲突）；
    5. kv_delta_find 访问器（逐请求命中/未知 None/单调 seq）；
  B. 失败注入三件（设计文档 §6.2 失败分支闭合）：
    6. 源端保护泄漏：远读在飞期对 home 侧基础占容/直接逐出 → 双通道
       fail-closed 且源端字节原样在账（结构保护 = primary 实例迁移 +
       active 位 + 守恒审计含 base 贡献）；
    7. 结算重复回调：同请求二次 merge_back → 版本键 raise，journal 恰
       一行、败者侧无二次释放、账本不变；
    8. 合并目标容量未准备：发射时空闲/结算时被占——胜者侧取完成时刻
       真实剩余做空间准备（可逐 victim 则前向成立、活跃 blocker 占满
       则方向回退翻转），不沿用发射时预测空闲；
  C. F15 合成平局（P0 冻结规则，C12 §15.4）：
    9. 两侧保留量严格相等（等值先断言 = 构造验证，非肉眼比对）→
       前向合并、home 标签不变、传输字节 ≡ 任一侧保留量（整数字节
       严格相等）；
  D. SH merge watch 结算闭合门（C14 步骤 1 完成检查的 watch 侧）：
    10. watch 交付 ⇔ journal 行在案；缺行 fail-closed；替身缺接口
        软跳过（并行兼容锚，与 last_merge_outcome 同款口径）；
  E. 覆盖率与失败分类互斥闭合（C14 步骤 4 报告口径的单测钉）：
    11. 成功结算序列各行恰一方向互斥、深缺口失败零行——完成覆盖
        （journal 行集）与失败分类（deep_gap 台账）互斥且并集闭合。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_remote_settlement.py   （或 pytest 同路径）
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


def _remote_round(manager, *, base_tokens, increment_tokens,
                  trigger_request_id="r1", seed=True, session_id="s",
                  home_index=0, exec_index=1):
    """构造一轮 remote-read（LOCAL 基 @home → exec @异地）并返回
    (home_side_sum, exec_side_sum)——两侧实际保留量（结算前）。

    seed=True 首轮播种（新会话）；seed=False 续轮（既有会话，要求
    context_tokens == base_tokens 的链式口径——多轮连续结算场景）。
    """
    if seed:
        _seed_completed_session(
            manager, session_id=session_id, instance_index=home_index,
            context_tokens=base_tokens)
    manager.prepare_prefill(
        session_id=session_id, target_instance_index=exec_index,
        history_tokens=base_tokens,
        trigger_request_id=trigger_request_id, action="remote-read")
    manager.expand_prefill(
        session_id=session_id, instance_index=exec_index,
        context_tokens=increment_tokens,
        trigger_request_id=trigger_request_id)
    session = manager._sessions[session_id]
    home_side = sum(kv_cache_shard_bytes_for_tokens(
        manager.model, base_tokens, manager.tp_degree))
    exec_side = sum(session.shard_bytes)
    return home_side, exec_side


# ------------------------------------------------------ A. journal 行 --

class KvDeltaJournalForwardRowTests(unittest.TestCase):
    """A1/A2/A3：三方向结算行的字段面（C16 消费字段 + 披露键）。"""

    def test_remote_read_forward_row_settlement_facts(self):
        # B(kv(10)=320) ≥ I(kv(5)=160) → 前向。行字段逐项断言。
        model, manager = _settlement_manager()
        home_side, exec_side = _remote_round(
            manager, base_tokens=10, increment_tokens=5)
        self.assertEqual((home_side, exec_side), (320, 160))

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(len(transfers), 1)  # 一笔 noc_migrate exec→home
        row = manager.kv_delta_find("r1")
        self.assertIsNotNone(row)
        self.assertEqual(row["seq"], 0)
        self.assertEqual(row["session_id"], "s")
        self.assertEqual(row["trigger_request_id"], "r1")
        self.assertEqual(row["working_kind"], "remote-read")
        self.assertEqual(row["direction"], "forward")
        self.assertFalse(row["zero_byte_flip"])
        self.assertEqual(row["winner_instance"], 0)
        self.assertEqual(row["loser_instance"], 1)
        self.assertEqual(row["home_before"], 0)
        self.assertEqual(row["home_after"], 0)  # home_after := winner
        self.assertFalse(row["home_migration"])
        # 少并多裁决输入 = 两侧实际保留量（账本真值）；传输字节 ≡ 败者侧。
        self.assertEqual(row["home_side_retained_bytes"], home_side)
        self.assertEqual(row["exec_side_retained_bytes"], exec_side)
        self.assertEqual(row["transferred_bytes"], exec_side)
        self.assertEqual(row["new_tokens"], 5)
        # F14（§15.6）：无留存型暂存 ⇒ 暂存归还 = 无操作（披露位恒 0）；
        # 前向后执行端零驻留（staging 无残留）。
        self.assertEqual(row["staging_return_bytes"], 0)
        self.assertEqual(_kv_bytes(manager, 1), (0, 0))
        self.assertEqual(len(manager.kv_delta_journal), 1)
        # 与 last_merge_outcome 披露快照一致（两通路同源事实）。
        outcome = manager.last_merge_outcome
        self.assertEqual(outcome["direction"], row["direction"])
        self.assertEqual(
            outcome["transferred_bytes"], row["transferred_bytes"])
        self.assertEqual(
            outcome["home_flipped"], row["home_migration"])

    def test_remote_read_reverse_row_home_migration_trajectory(self):
        # B(kv(5)=160) < I(kv(10)=320) → 翻转：home 迁移轨迹三键。
        model, manager = _settlement_manager()
        home_side, exec_side = _remote_round(
            manager, base_tokens=5, increment_tokens=10)
        self.assertEqual((home_side, exec_side), (160, 320))

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=10)

        self.assertEqual(len(transfers), 1)
        row = manager.kv_delta_find("r1")
        self.assertEqual(row["direction"], "reverse")
        self.assertFalse(row["zero_byte_flip"])
        self.assertEqual(row["winner_instance"], 1)
        self.assertEqual(row["loser_instance"], 0)
        self.assertEqual(row["home_before"], 0)
        self.assertEqual(row["home_after"], 1)
        self.assertTrue(row["home_migration"])
        self.assertEqual(row["home_side_retained_bytes"], 160)
        self.assertEqual(row["exec_side_retained_bytes"], 320)
        self.assertEqual(row["transferred_bytes"], 160)  # ≡ 败者侧保留量
        # 翻转后 home 侧零驻留（源端保护随结算解除、败者侧释放）。
        self.assertEqual(_kv_bytes(manager, 0), (0, 0))

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
        # （续轮 seed=False：既有会话 context 链式对齐）。
        model, manager = _settlement_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.merge_back(session_id="s", trigger_request_id="r1",
                           new_tokens=0)
        home_side, exec_side = _remote_round(
            manager, base_tokens=10, increment_tokens=5,
            trigger_request_id="r2", seed=False)
        manager.merge_back(session_id="s", trigger_request_id="r2",
                           new_tokens=5)

        self.assertIsNone(manager.kv_delta_find("unknown"))
        first = manager.kv_delta_find("r1")
        second = manager.kv_delta_find("r2")
        self.assertEqual(first["direction"], "stay")
        self.assertEqual(second["direction"], "forward")
        self.assertEqual(second["seq"], 1)
        self.assertLess(first["seq"], second["seq"])
        seqs = [row["seq"] for row in manager.kv_delta_journal]
        self.assertEqual(seqs, list(range(len(seqs))))
        self.assertEqual(len(manager.kv_delta_journal), 2)


# -------------------------------------------------- B. 失败注入三件 --

class RemoteSettlementFailureInjectionTests(unittest.TestCase):
    """设计文档 §6.2 完成检查清单的失败分支闭合（各一）。

    C14 条款：源端保护泄漏 / 结算重复回调 / 合并目标容量未准备。
    """

    def test_source_protection_leak_fails_closed_both_channels(self):
        # 源端保护泄漏注入：远读在飞期（prepare 后、merge 前）对 home
        # 侧基础历史施加逐出压力/直接逐出——两通道均 fail-closed 且
        # 源端字节原样在账（权威副本不丢）。
        model, manager = _settlement_manager()
        _remote_round(manager, base_tokens=10, increment_tokens=5)
        session = manager._sessions["s"]
        self.assertEqual(session.working_kind, "remote-read")
        base_kv = kv_cache_shard_bytes_for_tokens(model, 10, 2)

        # 通道 1：home 实例（ins0）容量压力——活跃远读会话的源端基础
        # 不是合法 victim（primary instance 已指向 exec ⇒ 候选集结构性
        # 不含本会话；无其他可逐对象）→ KVCapacityError fail-closed。
        # F4（复审修复）：原 assertRaises(Exception) + 仅反向排除
        # RuntimeError 过宽（几乎任何异常都绿）——实测该路径抛
        # KVCapacityError（face_scheduler 的容量 fail-closed 类型，
        # 2026-09-22 探针验证），改精确类型（与通道 2 的
        # assertRaisesRegex 风格对齐）。
        with self.assertRaisesRegex(
                KVCapacityError, "insufficient target HBM"):
            manager._ensure_capacity(
                0, tuple(b << 40 for b in base_kv),
                phase="decode", reason="probe_source_protection",
                trigger_request_id="probe")
        # 源端字节原样在账（未被静默释放/外迁）。
        self.assertEqual(_kv_bytes(manager, 0), base_kv)
        # 守恒审计通道含 home 侧基础贡献（_incremental_base_
        # contribution——审计不把保护中的源端误判为泄漏）。
        contribution = manager._incremental_base_contribution(session)
        self.assertEqual(contribution, (0, base_kv))

        # 通道 2：直接逐出入口的 active 位守卫（绕过候选集的注入面）。
        with self.assertRaisesRegex(
                RuntimeError,
                "only completed inactive sessions may be evicted"):
            manager._evict_suffix(
                session, phase="decode", reason="probe_direct_evict",
                trigger_request_id="probe", layer_start=2)
        self.assertEqual(_kv_bytes(manager, 0), base_kv)

        # 结算后保护解除（forward：home 持并集成合法终态；本会话此刻
        # 仍 active，mark_complete 后方可成 victim——解除时机正确）。
        manager.merge_back(session_id="s", trigger_request_id="r1",
                           new_tokens=5)
        self.assertEqual(
            _kv_bytes(manager, 0),
            kv_cache_shard_bytes_for_tokens(model, 15, 2))
        manager.mark_complete("s", 15)
        self.assertFalse(manager._sessions["s"].active)

    def test_duplicate_settlement_callback_fails_closed_no_double_release(self):
        # 结算重复回调注入：同请求二次 merge_back → 版本键 raise；
        # journal 恰一行、败者侧无二次释放（账本不变）。
        model, manager = _settlement_manager()
        _remote_round(manager, base_tokens=10, increment_tokens=5)
        manager.merge_back(session_id="s", trigger_request_id="r1",
                           new_tokens=5)
        settled_home = _kv_bytes(manager, 0)
        settled_exec = _kv_bytes(manager, 1)
        self.assertEqual(settled_exec, (0, 0))

        with self.assertRaisesRegex(
                RuntimeError, "settled twice"):
            manager.merge_back(session_id="s", trigger_request_id="r1",
                               new_tokens=5)

        rows = [row for row in manager.kv_delta_journal
                if row["trigger_request_id"] == "r1"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(_kv_bytes(manager, 0), settled_home)
        self.assertEqual(_kv_bytes(manager, 1), settled_exec)

    def test_winner_capacity_prepared_at_merge_time_not_launch_time(self):
        # 合并目标容量未准备注入：发射（prepare）时 ins0 大片空闲；
        # 结算前被 filler 占据——merge 必须按**完成时刻**真实剩余做
        # 胜者侧空间准备（可逐 victim ⇒ 前向成立 + 统一逐出发射），
        # 不得沿用发射时预测空闲直接入账。
        model, manager = _settlement_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r1", action="remote-read")
        # 发射后、结算前：filler（已完成可逐）吃掉 ins0 的剩余空间，
        # 使前向所需 W(kv(5)=80/rank) 超出当前空闲（free=32/rank < 80）。
        _seed_completed_session(
            manager, session_id="filler", instance_index=0,
            context_tokens=98)
        remaining = manager._effective_remaining_by_tp_rank(0)
        self.assertLess(min(remaining), 80)  # 前向空间缺口已实在
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        # 真实空间准备：统一 T+E 逐出发射（filler 的 remote_store——
        # 按缺口逐层/整会话释放）＋前向 noc 腿；方向维持前向、终态守恒
        # （ins0 = kv(15) + filler 残量，精确逐 rank 相等）。
        victim_stores = [t for t in transfers if t.kind == "remote_store"]
        self.assertTrue(victim_stores)
        self.assertEqual(
            {t.session_id for t in victim_stores}, {"filler"})
        nocs = [t for t in transfers if t.kind == "noc_migrate"]
        self.assertEqual(len(nocs), 1)
        self.assertEqual(nocs[0].reason, "merge_working_copy_to_home")
        row = manager.kv_delta_find("r1")
        self.assertEqual(row["direction"], "forward")
        filler = manager.session_snapshot("filler")
        self.assertLess(
            filler.resident_prefix_layers, 4)  # filler 被真实逐出
        expected_ins0 = tuple(
            kv + res for kv, res in zip(
                kv_cache_shard_bytes_for_tokens(model, 15, 2),
                filler.local_shard_bytes))
        self.assertEqual(_kv_bytes(manager, 0), expected_ins0)
        self.assertEqual(_kv_bytes(manager, 1), (0, 0))

    def test_winner_capacity_blocker_after_launch_falls_back_reverse(self):
        # 变体：发射后占位者是活跃 blocker（不可逐）——前向真实准备
        # 失败 → 双向二选一兜底改试反向（exec 容纳 B）；若按发射时
        # 预测空闲则会误判前向可行（发射时 ins0 尚空）。
        model, manager = _settlement_manager()
        _seed_completed_session(
            manager, session_id="s", instance_index=0, context_tokens=10)
        manager.prepare_prefill(
            session_id="s", target_instance_index=1, history_tokens=10,
            trigger_request_id="r1", action="remote-read")
        _seed_active_session(
            manager, session_id="blocker", instance_index=0,
            context_tokens=100)
        manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=5,
            trigger_request_id="r1")

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        nocs = [t for t in transfers if t.kind == "noc_migrate"]
        self.assertEqual(len(nocs), 1)
        self.assertEqual(nocs[0].reason, "merge_base_to_exec")
        row = manager.kv_delta_find("r1")
        self.assertEqual(row["direction"], "reverse")
        self.assertTrue(row["home_migration"])
        self.assertEqual(row["home_after"], 1)
        self.assertEqual(
            _kv_bytes(manager, 0),
            kv_cache_shard_bytes_for_tokens(model, 100, 2))
        self.assertEqual(
            _kv_bytes(manager, 1),
            kv_cache_shard_bytes_for_tokens(model, 15, 2))


# ------------------------------------------------------ C. F15 平局 --

class F15SyntheticTieTests(unittest.TestCase):
    """F15（P0 冻结，C12 §15.4）：两侧保留量严格相等的确定性规则。"""

    def test_strict_tie_merges_forward_home_unchanged_bytes_equal_either_side(self):
        # 构造：base 5 tokens @home、increment 5 tokens @exec——两侧保留
        # 量在整数字节账本下严格相等（先断言等值 = 构造验证，非肉眼
        # 比对比较运算符）。F15 规则 = 前向合并（<= 判据落 forward）、
        # home 标签不变、传输字节 ≡ 任一侧保留量。
        model, manager = _settlement_manager()
        home_side, exec_side = _remote_round(
            manager, base_tokens=5, increment_tokens=5)
        self.assertEqual(home_side, exec_side)  # 严格平局构造钉
        self.assertEqual(home_side, 160)

        transfers = manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)

        self.assertEqual(len(transfers), 1)
        noc = transfers[0]
        self.assertEqual(noc.kind, "noc_migrate")
        self.assertEqual(noc.reason, "merge_working_copy_to_home")
        self.assertEqual(noc.source_instance_index, 1)
        self.assertEqual(noc.target_instance_index, 0)
        row = manager.kv_delta_find("r1")
        self.assertEqual(row["direction"], "forward")
        self.assertFalse(row["home_migration"])
        self.assertEqual(row["home_before"], 0)
        self.assertEqual(row["home_after"], 0)
        # 传输字节 ≡ 任一侧保留量（整数字节严格相等）。
        self.assertEqual(noc.total_bytes, home_side)
        self.assertEqual(noc.total_bytes, exec_side)
        self.assertEqual(row["transferred_bytes"], home_side)
        self.assertEqual(row["home_side_retained_bytes"], home_side)
        self.assertEqual(row["exec_side_retained_bytes"], exec_side)
        # home 标签不变（快照终态钉）+ 胜者持并集。
        merged = manager.session_snapshot("s")
        self.assertEqual(merged.home_instance, 0)
        self.assertEqual(merged.instance_index, 0)
        self.assertEqual(
            merged.shard_bytes,
            kv_cache_shard_bytes_for_tokens(model, 10, 2))
        self.assertEqual(_kv_bytes(manager, 1), (0, 0))


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
        scheduler._on_merge_done("batch_train_merge_r1", 2_000_000)
        # R15：#merge 流注销恰一次。
        self.assertEqual(
            scheduler._joint_flows.released, ["r1#merge"])
        self.assertEqual(
            scheduler._pool_ports.released, ["r1#merge"])
        # C3b 披露行照落（kind=merge_done）。
        self.assertEqual(len(scheduler.online_log_rows), 1)
        logged = scheduler.online_log_rows[0]
        self.assertEqual(logged["kind"], "merge_done")
        self.assertEqual(logged["tick"], 2_000_000)
        self.assertEqual(
            logged["decision"]["merge_done_ns"], 2_000_000)
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


# ------------------------------------------- E. 覆盖率互斥闭合 --

class SettlementCoverageClosureTests(unittest.TestCase):
    """C14 步骤 4 报告口径单测钉：完成覆盖（journal 行集）与失败分类
    （deep_gap 台账 + 异常）互斥、并对全部结算尝试并集闭合。"""

    def test_success_and_failure_classification_mutually_exclusive(self):
        # 字节预算（kv(n)=16n B/rank、容量 1920 B/rank-权重外）：逐轮
        # 手算钉到"前向缺口 48 > 可逐 32、反向缺口 64 > 可逐 0"的双侧
        # 深缺口形态（活跃 blocker 两翼占满）。
        model, manager = _settlement_manager()
        # 1) stay 成功（s_stay 2 tokens @ins0 = 32B/rank，恰为前向侧
        #    唯一可逐对象且不足以覆盖缺口）。
        _seed_completed_session(
            manager, session_id="s_stay", instance_index=0,
            context_tokens=2)
        manager.merge_back(
            session_id="s_stay", trigger_request_id="r_stay", new_tokens=0)
        # 2) remote 前向成功（base 6 → inc 3；H=192 ≥ W=96）。
        _remote_round(
            manager, base_tokens=6, increment_tokens=3,
            trigger_request_id="r_fwd")
        manager.merge_back(
            session_id="s", trigger_request_id="r_fwd", new_tokens=3)
        # 3) remote 翻转成功（续轮 base 9 → inc 12；H=288 < W=384，
        #    home 迁移到 exec=ins1，context 21）。
        _remote_round(
            manager, base_tokens=9, increment_tokens=12,
            trigger_request_id="r_rev", seed=False)
        manager.merge_back(
            session_id="s", trigger_request_id="r_rev", new_tokens=12)
        # 4) F15 平局成功（续轮 home=1 → exec=0；base 21 → inc 21，
        #    两侧 672 = 672 严格相等）。
        _remote_round(
            manager, base_tokens=21, increment_tokens=21,
            trigger_request_id="r_tie", seed=False,
            home_index=1, exec_index=0)
        self.assertEqual(
            sum(manager._sessions["s"].shard_bytes), 672)
        manager.merge_back(
            session_id="s", trigger_request_id="r_tie", new_tokens=21)
        settled = 4

        # 5) 失败分类：双侧活跃 blocker 占满 → 双向深缺口 fail-closed。
        #    ins0 = 32(s_stay) + 64(s_gap 基) + 1664(blk_home) 满容；
        #    ins1 = 672(s) + 1040(blk_exec) + 48(s_gap 增量) 满容。
        _seed_completed_session(
            manager, session_id="s_gap", instance_index=0,
            context_tokens=4)
        manager.prepare_prefill(
            session_id="s_gap", target_instance_index=1,
            history_tokens=4, trigger_request_id="r_gap",
            action="remote-read")
        _seed_active_session(
            manager, session_id="blk_home", instance_index=0,
            context_tokens=104)
        _seed_active_session(
            manager, session_id="blk_exec", instance_index=1,
            context_tokens=65)
        manager.expand_prefill(
            session_id="s_gap", instance_index=1, context_tokens=3,
            trigger_request_id="r_gap")
        with self.assertRaisesRegex(RuntimeError, "dual-sided deep gap"):
            manager.merge_back(
                session_id="s_gap", trigger_request_id="r_gap",
                new_tokens=3)

        # 闭合断言：成功 = journal 恰 settled 行（互斥方向枚举、逐请求
        # 恰一行）；失败 = 零 journal 行 + deep_gap 台账在案。
        journal = manager.kv_delta_journal
        self.assertEqual(len(journal), settled)
        directions = {row["direction"] for row in journal}
        self.assertTrue(directions <= {
            "stay", "forward", "reverse", "in_place"})
        self.assertEqual(directions, {"stay", "forward", "reverse"})
        request_ids = [row["trigger_request_id"] for row in journal]
        self.assertEqual(len(request_ids), len(set(request_ids)))
        self.assertNotIn("r_gap", request_ids)
        self.assertIsNone(manager.kv_delta_find("r_gap"))
        self.assertTrue(manager.deep_gap_events)
        self.assertTrue(all(
            record["reason"] == "merge_winner_capacity"
            for record in manager.deep_gap_events))


if __name__ == "__main__":
    unittest.main()
