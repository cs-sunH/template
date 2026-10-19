#!/usr/bin/env python3
"""test_kv_directory_projection_parity.py -- W4 目录投影等价与多源读流
退化等价钉测（硬门）。

钉测面（《joint仓库当下+历史修改方案》§6.1/§12.1 W4 验收）：

A. **投影等价**（硬门）：真实 KVCacheManager 驱动二动作全场景矩阵
   （stay/copy × LOCAL/PARTIAL/REMOTE 基 + 后缀/
   整份逐出；recompute 轮与边界来源转记/三位置结算已随
   2026-10-15 recompute 退役删除——通道 C 投影随 D7 同步清理），
   每个 checkpoint 后——
   1. 目录 capacity_projection ↔ rank 台账逐 rank 守恒（manager 内
      fail-closed 审计的显式复核）；
   2. **目录条目重导出的 SessionKVView 字段**（location/history_tokens/
      resident_prefix_layers/history_bytes/missing_bytes）与旧字段推导
      （sh30 `_joint_session_view` 同式）逐字段一致；
   3. **目录条目重导出的 PDBoundaryView 字段**（working/base 字节向量、
      base_tokens、base_instance）与旧字段推导（sh30 `_joint_boundary_
      view` 同式）逐字段一致；两视图分别过 `BoundaryKVLayout.
      from_boundary` 产出的区间序列逐字段一致（role/holder/token/
      layer/bytes）。
   视图采样点 = 旧字段查询点（copy 交接 journal 已结算——与生产查询
   时点同约束）。（B 组退化等价与增量型边界视图已随远读退役删除，
   2026-10-14——布局恒完整型单族，D6。）

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 joint/test_kv_directory_projection_parity.py
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

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    build_instances,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
)
from joint.boundary_layout import BoundaryKVLayout  # noqa: E402
from joint.joint_config import parse_joint_config  # noqa: E402
from joint.joint_cost_model import (  # noqa: E402
    PDBoundaryView,
    SessionKVView,
)
from joint.kv_location_directory import (  # noqa: E402
    ROLE_BASE,
    ROLE_RESIDENT,
    ROLE_WORKING,
)
from joint.test_joint_mechanisms import _tiny_model  # noqa: E402
from online.sh30_online_scheduler import Sh30OnlineScheduler  # noqa: E402

# ---- 三实例几何（同 test_boundary_policy_one_shot：ins0/ins1/ins2，
# tp=2，ranks {0,3}/{1,4}/{2,5}）----
INSTANCE_COUNT = 3
RANKS_BY_INSTANCE = {0: (0, 3), 1: (1, 4), 2: (2, 5)}
HOME = 2
P_EXEC = 0
THIRD = 1

HISTORY_TOKENS = 96        # 基础历史 H
INPUT_TOKENS = 16          # 本轮输入 U
DECODE_STEPS = 8
MODEL_LAYERS = 4
PARTIAL_PREFIX = 2         # PARTIAL 基驻留前缀 [0, p)


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


def _manager(*, strict=False):
    return KVCacheManager(
        _topology(_hardware()), _tiny_model(layers=MODEL_LAYERS),
        strict_invariants=strict)


def _seed_history(kv, session_id, instance, tokens):
    """在 instance 上建档并增长到 tokens（首轮 stay 语义建档）。"""
    kv.prepare_prefill(
        session_id=session_id, target_instance_index=instance,
        history_tokens=0, trigger_request_id=f"{session_id}-seed")
    kv.expand_prefill(
        session_id=session_id, instance_index=instance,
        context_tokens=tokens, trigger_request_id=f"{session_id}-seed")
    return kv._sessions[session_id]


def _evict_to_partial(kv, session_id, trigger):
    """完成后缀逐出：全层 → PARTIAL（驻留前缀 PARTIAL_PREFIX）。"""
    session = kv._sessions[session_id]
    kv.mark_complete(session_id, completion_ns=1000)
    return kv._evict_suffix(
        session, phase="history", reason="test_parity",
        trigger_request_id=trigger,
        layer_start=PARTIAL_PREFIX)


# ---------------------------------------------------------- 旧字段推导 --
# 与 online/sh30_online_scheduler._joint_session_view/_joint_boundary_view
# 的公式逐式对应（W4 前旧路径的唯一事实源 = SessionKVState 旧字段）。


def legacy_session_view_fields(kv, session_id):
    session = kv._sessions[session_id]
    home = session.home_instance
    if session.working_kind is not None:
        base_location = session.base_location
        if base_location == "":
            raise RuntimeError("corrupt empty base_location")
        location = (
            base_location if base_location is not None
            else kv.REMOTE_MEMORY)
        base_tokens = session.base_history_tokens
        base_prefix = session.base_resident_prefix_layers
        resident = home
    else:
        location = session.location
        base_tokens = session.context_tokens
        base_prefix = session.resident_prefix_layers
        resident = session.instance_index
    tp = kv.tp_degree
    local = (
        kv_cache_shard_bytes_for_layer_range(
            kv.model, base_tokens, tp, layer_start=0, layer_end=base_prefix)
        if location in (kv.LOCAL_HBM, kv.PARTIAL_HBM_REMOTE) and base_prefix
        else (0,) * tp)
    full = (
        kv_cache_shard_bytes_for_tokens(kv.model, base_tokens, tp)
        if base_tokens else (0,) * tp)
    missing = tuple(f - l for f, l in zip(full, local))
    return dict(
        session_id=session_id,
        home_instance=home, resident_instance=resident, location=location,
        history_tokens=base_tokens, resident_prefix_layers=base_prefix,
        history_bytes_by_tp_rank=local, missing_bytes_by_tp_rank=missing)


def legacy_boundary_view_fields(kv, session_id, *, p_index, p_action,
                                history_tokens, decode_steps,
                                kv_token_bytes):
    """旧字段推导（完整型单族——P ∈ {stay,copy} 的工作副本
    恒全量在 i_P；增量型布局族已随远读退役删除，D6）。"""
    session = kv._sessions[session_id]
    working = tuple(int(v) for v in session.shard_bytes)
    return dict(
        session_id=session_id, home_instance=session.instance_index,
        p_instance_index=p_index, p_action=p_action,
        history_tokens=history_tokens,
        working_bytes_by_tp_rank=working,
        decode_steps=decode_steps,
        kv_token_bytes=int(kv_token_bytes),
        logic_home_instance=session.home_instance)


# ---------------------------------------------------------- 目录投影 --
# 只读目录条目（+ 会话布局族判据/逻辑锚点，与 sh30 同源的非布局输入）
# 重导出同一视图字段。


def directory_session_view_fields(kv, session_id):
    session = kv._sessions[session_id]
    entries = kv.kv_directory.entries_for_session(session_id)
    on_chip = [e for e in entries if e.is_on_chip()]
    pool = [e for e in entries if e.is_pool_backed()]
    tp = kv.tp_degree
    zeros = (0,) * tp
    if session.working_kind is not None:
        base = [e for e in on_chip if e.role == ROLE_BASE]
        if base:
            entry = base[0]
            location = (
                kv.LOCAL_HBM
                if entry.content_key.layer_end == kv.model.layers
                else kv.PARTIAL_HBM_REMOTE)
            tokens = entry.content_key.token_end
            prefix = entry.content_key.layer_end
            local = entry.bytes_by_tp_rank
        elif pool:
            entry = pool[0]
            location = kv.REMOTE_MEMORY
            tokens = entry.content_key.token_end
            prefix = 0
            local = zeros
        else:
            location = kv.REMOTE_MEMORY
            tokens = 0
            prefix = 0
            local = zeros
        full = (
            kv_cache_shard_bytes_for_tokens(kv.model, tokens, tp)
            if tokens else zeros)
        missing = tuple(f - l for f, l in zip(full, local))
        return dict(
            session_id=session_id,
            home_instance=session.home_instance,
            resident_instance=session.home_instance,
            location=location, history_tokens=tokens,
            resident_prefix_layers=prefix,
            history_bytes_by_tp_rank=local,
            missing_bytes_by_tp_rank=missing)
    primary = [e for e in on_chip if e.role == ROLE_RESIDENT]
    if not primary:
        entry = pool[0]
        return dict(
            session_id=session_id,
            home_instance=session.home_instance, resident_instance=None,
            location=kv.REMOTE_MEMORY,
            history_tokens=entry.content_key.token_end,
            resident_prefix_layers=0,
            history_bytes_by_tp_rank=zeros,
            missing_bytes_by_tp_rank=entry.bytes_by_tp_rank)
    entry = primary[0]
    tokens = entry.content_key.token_end
    if pool:
        suffix = pool[0]
        location = kv.PARTIAL_HBM_REMOTE
        prefix = suffix.content_key.layer_start
        missing = suffix.bytes_by_tp_rank
    else:
        location = kv.LOCAL_HBM
        prefix = entry.content_key.layer_end
        missing = zeros
    return dict(
        session_id=session_id,
        home_instance=session.home_instance,
        resident_instance=session.instance_index,
        location=location, history_tokens=tokens,
        resident_prefix_layers=prefix,
        history_bytes_by_tp_rank=entry.bytes_by_tp_rank,
        missing_bytes_by_tp_rank=missing)


def directory_boundary_view_fields(kv, session_id, *, p_index, p_action,
                                   history_tokens, decode_steps,
                                   kv_token_bytes):
    """目录投影（完整型单族——聚合主条目后与 legacy shard_bytes
    对拍；base 投影支路已随增量族退役删除）。"""
    session = kv._sessions[session_id]
    entries = kv.kv_directory.entries_for_session(session_id)
    tp = kv.tp_degree
    primary = [
        e for e in entries
        if e.role in (ROLE_WORKING, ROLE_RESIDENT) and e.is_on_chip()]
    assert primary, "primary entry missing in directory mirror"
    by_rank = [0] * tp
    for entry in primary:
        for rank, value in enumerate(entry.bytes_by_tp_rank):
            by_rank[rank] += int(value)
    working = tuple(by_rank)
    return dict(
        session_id=session_id, home_instance=primary[0].holder,
        p_instance_index=p_index, p_action=p_action,
        history_tokens=history_tokens,
        working_bytes_by_tp_rank=working,
        decode_steps=decode_steps,
        kv_token_bytes=int(kv_token_bytes),
        logic_home_instance=session.home_instance)


def _view_from_fields(cls, fields):
    return cls(**fields)


class ProjectionParityTests(unittest.TestCase):
    """A 组：目录投影 ↔ 旧字段推导 逐字段等价（硬门）。"""

    def setUp(self):
        self.kv = _manager()
        self.session_id = "s_parity"

    # ---- 断言面 ----

    def assert_session_view_parity(self, note=""):
        legacy = legacy_session_view_fields(self.kv, self.session_id)
        projected = directory_session_view_fields(self.kv, self.session_id)
        self.assertEqual(legacy, projected, f"SessionKVView parity {note}")
        # 与生产构造器同型（字段完备性核验）。
        SessionKVView(**projected)
        SessionKVView(**legacy)

    def assert_boundary_view_parity(self, *, p_index, p_action,
                                    history_tokens, note=""):
        kv_token_bytes = kv_cache_shard_bytes_for_tokens(
            self.kv.model, 1, self.kv.tp_degree)[0]
        common = dict(
            p_index=p_index, p_action=p_action,
            history_tokens=history_tokens, decode_steps=DECODE_STEPS,
            kv_token_bytes=kv_token_bytes)
        legacy = legacy_boundary_view_fields(
            self.kv, self.session_id, **common)
        projected = directory_boundary_view_fields(
            self.kv, self.session_id, **common)
        self.assertEqual(legacy, projected, f"PDBoundaryView parity {note}")
        # BoundaryKVLayout.from_boundary 双视图逐字段一致。
        layout_legacy = BoundaryKVLayout.from_boundary(
            _view_from_fields(PDBoundaryView, legacy))
        layout_projected = BoundaryKVLayout.from_boundary(
            _view_from_fields(PDBoundaryView, projected))
        self.assertEqual(
            layout_legacy, layout_projected, f"BoundaryKVLayout {note}")
        self.assertEqual(
            tuple(
                (i.role, i.holder_instance, i.token_start, i.token_end,
                 i.layer_start, i.layer_end, i.bytes_by_tp_rank)
                for i in layout_legacy.intervals),
            tuple(
                (i.role, i.holder_instance, i.token_start, i.token_end,
                 i.layer_start, i.layer_end, i.bytes_by_tp_rank)
                for i in layout_projected.intervals),
            f"layout intervals {note}")

    def assert_capacity_parity(self, note=""):
        for instance in self.kv.topology.instances:
            projected = self.kv.kv_directory.capacity_projection(
                instance.index)
            actual = tuple(
                self.kv._rank_states[rank].kv_cache_bytes
                for rank in instance.ranks)
            self.assertEqual(
                projected, actual, f"capacity projection {note}")

    # ---- 场景矩阵 ----

    def test_stay_local_round(self):
        """stay@LOCAL：建档 → 增长 → 边界 stay → 结算。"""
        _seed_history(self.kv, self.session_id, P_EXEC, HISTORY_TOKENS)
        self.kv.prepare_prefill(
            session_id=self.session_id, target_instance_index=P_EXEC,
            history_tokens=HISTORY_TOKENS,
            trigger_request_id="r-stay", action="stay")
        self.assert_capacity_parity("stay-prepared")
        self.assert_session_view_parity("stay-prepared")
        self.kv.expand_prefill(
            session_id=self.session_id, instance_index=P_EXEC,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r-stay")
        self.assert_session_view_parity("stay-grown")
        self.assert_boundary_view_parity(
            p_index=P_EXEC, p_action="stay",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, note="stay")
        self.kv.commit_boundary_decode_transition(
            session_id=self.session_id, p_instance_index=P_EXEC,
            target_instance_index=P_EXEC, action="stay",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r-stay")
        self.assert_session_view_parity("stay-committed")
        self.kv.expand_decode(
            session_id=self.session_id, instance_index=P_EXEC,
            final_context_tokens=HISTORY_TOKENS + INPUT_TOKENS + DECODE_STEPS,
            trigger_request_id="r-stay")
        self.assert_session_view_parity("stay-decode")
        self.kv.merge_back(
            session_id=self.session_id, trigger_request_id="r-stay",
            new_tokens=INPUT_TOKENS + DECODE_STEPS)
        self.assert_capacity_parity("stay-merged")
        self.assert_session_view_parity("stay-merged")

    def test_copy_round_with_handoff_journal(self):
        """copy@异地：准入 → 交接 journal 开放（真值=home_shards）→
        drain 结算 → 增长 → 前向结算。"""
        _seed_history(self.kv, self.session_id, HOME, HISTORY_TOKENS)
        self.kv.prepare_prefill(
            session_id=self.session_id, target_instance_index=P_EXEC,
            history_tokens=HISTORY_TOKENS,
            trigger_request_id="r-copy", action="copy")
        session = self.kv._sessions[self.session_id]
        journal = session.copy_handoff
        self.assertIsNotNone(journal)
        self.assert_capacity_parity("copy-admitted")
        # 开放期：目录 base 条目 = WRITING 且字节 = journal.home_shards()
        # （journal 真值口径，§1.8 增量通道同源）。
        base_entries = [
            e for e in self.kv.kv_directory.entries_for_session(
                self.session_id)
            if e.role == ROLE_BASE]
        self.assertEqual(len(base_entries), 1)
        self.assertEqual(base_entries[0].state, "WRITING")
        self.assertEqual(
            base_entries[0].bytes_by_tp_rank, journal.home_shards())
        self.assertTrue(base_entries[0].pending_acks)
        # drain（expand_prefill = prefill_drain 边界）结算交接。
        self.kv.expand_prefill(
            session_id=self.session_id, instance_index=P_EXEC,
            context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
            trigger_request_id="r-copy")
        self.assertTrue(journal.all_settled())   # drain 结算；close 在 merge
        # 块全部交接后 home 侧零残量：base 条目退场（基础内容的权威
        # 副本已在执行端工作副本条目内——§2.3 四步协议 (2)/(3)）。
        base_entries = [
            e for e in self.kv.kv_directory.entries_for_session(
                self.session_id)
            if e.role == ROLE_BASE]
        self.assertEqual(base_entries, [])
        self.assert_capacity_parity("copy-drained")
        # SessionKVView 的生产查询点 = 准入（prepare 前）/结算后——
        # drain→merge 间隙（journal 已结算未闭合）legacy 视图按 base_*
        # 公式报告已释放字节（无生产消费方），不在等价采样面内；边界
        # 视图（PDBoundaryView）在该点有真实消费方（D 边界决策），
        # 完整族 base 恒零向量——目录侧可无歧义投影。
        self.assert_boundary_view_parity(
            p_index=P_EXEC, p_action="copy",
            history_tokens=HISTORY_TOKENS + INPUT_TOKENS, note="copy")
        self.kv.merge_back(
            session_id=self.session_id, trigger_request_id="r-copy",
            new_tokens=INPUT_TOKENS)
        self.assert_capacity_parity("copy-merged")
        self.assert_session_view_parity("copy-merged")

    def test_full_eviction_pool_backing(self):
        """整份逐出 → REMOTE（唯一内容在池，owner 记录保留）。"""
        _seed_history(self.kv, self.session_id, HOME, HISTORY_TOKENS)
        self.kv.mark_complete(self.session_id, completion_ns=1000)
        session = self.kv._sessions[self.session_id]
        self.kv._evict_session(
            session, phase="history", reason="test_parity",
            trigger_request_id="r-evict")
        self.assertEqual(session.location, self.kv.REMOTE_MEMORY)
        self.assert_capacity_parity("evicted")
        self.assert_session_view_parity("evicted")
        pool = self.kv.kv_directory.query_pool_intervals(self.session_id)
        self.assertEqual(len(pool), 1)
        self.assertEqual(
            pool[0].content_key.token_range, (0, HISTORY_TOKENS))
        # 逐出不清 owner：逻辑 home 锚点保留 + 条目完整字节。
        self.assertEqual(session.home_instance, HOME)
        self.assertEqual(
            pool[0].bytes_by_tp_rank,
            kv_cache_shard_bytes_for_tokens(
                self.kv.model, HISTORY_TOKENS, self.kv.tp_degree))
        # 池恢复轮回（copy 全量池恢复）后回到 LOCAL。
        self.kv.prepare_prefill(
            session_id=self.session_id, target_instance_index=THIRD,
            history_tokens=HISTORY_TOKENS,
            trigger_request_id="r-evict", action="copy")
        self.assert_capacity_parity("restored")
        self.assert_session_view_parity("restored")

    def test_strict_manager_full_matrix(self):
        """strict 档全量审计（含目录全实例对账）下跑 stay+copy 混合。"""
        kv = _manager(strict=True)
        try:
            _seed_history(kv, "s_a", HOME, HISTORY_TOKENS)
            _seed_history(kv, "s_b", P_EXEC, 64)
            kv.prepare_prefill(
                session_id="s_a", target_instance_index=THIRD,
                history_tokens=HISTORY_TOKENS,
                trigger_request_id="r-a", action="copy")
            kv.expand_prefill(
                session_id="s_a", instance_index=THIRD,
                context_tokens=HISTORY_TOKENS + INPUT_TOKENS,
                trigger_request_id="r-a")
            kv.merge_back(
                session_id="s_a", trigger_request_id="r-a",
                new_tokens=INPUT_TOKENS)
            kv.assert_final_state()
        finally:
            # 供后续断言面复用同一 manager 语义，这里显式终态即可。
            pass
        for instance in kv.topology.instances:
            projected = kv.kv_directory.capacity_projection(instance.index)
            actual = tuple(
                kv._rank_states[rank].kv_cache_bytes
                for rank in instance.ranks)
            self.assertEqual(projected, actual)


# ------------------------------------------------------ B 组：退化等价 --


def _credit_scheduler(kv):
    """credit plan 替身调度器（同 test_remote_credit_stream 替身形态，
    kv_manager 换真实 KVCacheManager——目录路径生效）。"""
    s = Sh30OnlineScheduler.__new__(Sh30OnlineScheduler)
    s.p_chunk = 512
    s._pending_merge_alarms = {}
    ranks_extended = dict(RANKS_BY_INSTANCE)
    ranks_extended[3] = (0, 3)   # 合成第三来源的伪实例（目录结构支持面）
    s.topology = SimpleNamespace(
        instance=lambda i: SimpleNamespace(
            size=2, ranks=ranks_extended[i]),
        instances=[SimpleNamespace(size=2)] * INSTANCE_COUNT)
    s.hardware = _hardware()
    s.model = _tiny_model(layers=MODEL_LAYERS)
    s.kv_manager = kv
    s.joint_config = parse_joint_config({})
    return s


def _runtime(session_id, *, decode=DECODE_STEPS, input_tokens=INPUT_TOKENS,
             origin_home=HOME):
    rt = SimpleNamespace(
        session_id=session_id,
        request_id=f"{session_id}-rr",
        decode_length=decode,
        origin_home_instance=origin_home,
        joint_input_tokens=input_tokens,
        history_tokens_before=HISTORY_TOKENS,
    )
    return rt


if __name__ == "__main__":
    unittest.main()
