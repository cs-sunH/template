#!/usr/bin/env python3
"""test_joint_copy_handoff.py -- C13 copy 块级交接与源端立即释放的零后端
结构断言（2026-09-22；设计文档 §2.3 四步交接协议 + 守恒式 + §6.1；
仓库设计方案 §2.2；C12 冻结规格 §5）。

覆盖（本卡测试条款）：
  A. 账本侧（FS）：
    1. 块切分确定性（含 base_prefix ≤ 8 单块回归锚）；
    2. prepare 逐块化：M 块 NoC 交接流 + journal 登记 + 守恒式成立
       （准入物化口径：H_exec=H、D_handoff=H）；
    3. prefill drain 边界结算：home 侧逐块立即释放（不等轮末）、
       D_handoff 归零、逐块 #handoff/#copy-stream 事件；
    4. merge 零字节结算不重复释放已交接源块（home 侧不再二次扣减）；
       直接 merge（未走 drain）防御兜底结算；
    5. 逐 session 守恒审计通道（_incremental_base_contribution）随释放
       收缩；
  B. 失败注入（FS）：
    6. 双释放 / 交接前释放 / 重复交接 / 乱序交接 / 非法结算边界 /
       源块在途（working 副本未闭合即再发起 copy）各一，均 fail-closed；
  C. 图侧（GB）：
    7. 准入发射：交接块 0 上主链、尾块旁挂支链 + 逐 rank recv 完成门
       （_copy_handoff_arms）+ ack_recv 释放挂点（release anchors）；
    8. fork 顺序：尾块支链首节点 parent = 头块发射后的主链 frontier；
    9. readiness barrier 只等主链（不等尾块——不设"先整份搬运后计算"
       串行段）；
    10. 列车体逐块就绪门控：尾块 recv 完成门 arm 进覆盖其迭代区间的
        体块首节点；消费后账本弹出；
    11. completion 批弹出交接账本（残留 fail-closed）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_joint_copy_handoff.py   （或 pytest 同路径）
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
    COPY_HANDOFF_CHUNK_LAYERS,
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVCacheManager,
    KVTransfer,
    KVTransferShard,
    build_instances,
    kv_cache_shard_bytes_for_layer_range,
    kv_cache_shard_bytes_for_tokens,
    plan_copy_handoff_layer_chunks,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402


# ------------------------------------------------------------- FS 夹具 --

def _handoff_manager(*, capacity_bytes: int = 4000, layers: int = 16):
    """16 层双实例 fixture：每 token 每层每 rank 4B（2 头 × K+V × 1B）。
    layers=16 ⇒ COPY_HANDOFF_CHUNK_LAYERS(8) 切出 2 块。"""
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


def _seed_home_session(manager, *, tokens=10):
    manager.prepare_prefill(
        session_id="s",
        target_instance_index=0,
        history_tokens=0,
        trigger_request_id="s_seed")
    manager.expand_prefill(
        session_id="s", instance_index=0, context_tokens=tokens,
        trigger_request_id="s_seed")
    manager.mark_complete("s", tokens)


def _handoff_kv_bytes(manager):
    return tuple(
        manager._rank_states[rank].kv_cache_bytes for rank in (0, 1, 2, 3))


class CopyHandoffChunkPlanningTests(unittest.TestCase):
    def test_deterministic_planning_with_single_chunk_anchor(self):
        # base_prefix ≤ COPY_HANDOFF_CHUNK_LAYERS 时单块 = 旧单笔口径
        # 回归锚（test_face_scheduler 的两腿/单腿结构断言依赖）。
        self.assertEqual(plan_copy_handoff_layer_chunks(0), ())
        for prefix in (1, 2, 4, COPY_HANDOFF_CHUNK_LAYERS):
            self.assertEqual(
                plan_copy_handoff_layer_chunks(prefix), ((0, prefix),))
        self.assertEqual(
            plan_copy_handoff_layer_chunks(16), ((0, 8), (8, 16)))
        # 17 层 → 3 块均衡（n = ceil(17/8) = 3、span = ceil(17/3) = 6）。
        self.assertEqual(
            plan_copy_handoff_layer_chunks(17),
            ((0, 6), (6, 12), (12, 17)))
        self.assertEqual(
            plan_copy_handoff_layer_chunks(32),
            ((0, 8), (8, 16), (16, 24), (24, 32)))
        for prefix in range(1, 65):
            ranges = plan_copy_handoff_layer_chunks(prefix)
            # 覆盖完备 + 顺序邻接 + 层跨度不超目标 + 确定性。
            self.assertEqual(ranges[0][0], 0)
            self.assertEqual(ranges[-1][1], prefix)
            for (_, end), (start, _) in zip(ranges, ranges[1:]):
                self.assertEqual(end, start)
            for start, end in ranges:
                self.assertLessEqual(
                    end - start, COPY_HANDOFF_CHUNK_LAYERS)
            self.assertEqual(ranges, plan_copy_handoff_layer_chunks(prefix))


class CopyHandoffLedgerTests(unittest.TestCase):
    def setUp(self):
        self.model, self.manager = _handoff_manager()
        _seed_home_session(self.manager, tokens=10)

    def _prepare_copy(self):
        return self.manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="copy",
        )

    def test_prepare_multi_chunk_transfers_and_journal(self):
        before, transfers, evictions = self._prepare_copy()
        self.assertEqual(
            [(t.kind, t.layer_start, t.layer_end, t.handoff_chunk)
             for t in transfers],
            [("noc_migrate", 0, 8, 0), ("noc_migrate", 8, 16, 1)],
        )
        self.assertEqual(
            [t.reason for t in transfers],
            ["history_prefix_working_copy", "history_prefix_handoff_tail"],
        )
        session = self.manager._sessions["s"]
        journal = session.copy_handoff
        self.assertIsNotNone(journal)
        self.assertEqual(len(journal.chunks), 2)
        # 准入相口径：exec 物化（容量保守）+ home 未释放 ⇒ 双持有 +
        # D_handoff = H；守恒式 H_home + H_exec = H + D 成立。
        total = journal.total_shards
        self.assertEqual(journal.h_exec_shards, total)
        self.assertEqual(journal.h_home_shards, total)
        journal.assert_conservation()
        per_rank = kv_cache_shard_bytes_for_tokens(self.model, 10, 2)
        # 双持有（准入物化口径）：home 与 exec 各持整份基础历史。
        self.assertEqual(
            _handoff_kv_bytes(self.manager), tuple(per_rank) * 2)
        # 逐块字节与层区间派生一致（守恒基数）。
        for chunk in journal.chunks:
            self.assertEqual(
                chunk.shard_bytes,
                kv_cache_shard_bytes_for_layer_range(
                    self.model, 10, 2,
                    layer_start=chunk.layer_start,
                    layer_end=chunk.layer_end))
        # launch + 流计划科目已入台账（分计项另列）。
        subjects = [event["subject"] for event
                    in self.manager.copy_handoff_events]
        self.assertIn("r1#handoff", subjects)
        self.assertIn("r1#copy-stream", subjects)
        plan_event = next(
            event for event in self.manager.copy_handoff_events
            if event["subject"] == "r1#copy-stream"
            and event["event"] == "plan")
        self.assertEqual(plan_event["payload_bytes"], sum(total))
        self.assertTrue(plan_event["separate_ledgers"])

    def test_settle_at_prefill_drain_releases_home_immediately(self):
        self._prepare_copy()
        session = self.manager._sessions["s"]
        journal = session.copy_handoff
        per_rank = kv_cache_shard_bytes_for_tokens(self.model, 10, 2)
        # prefill drain 边界（expand_prefill）触发逐块结算。
        self.manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=15,
            trigger_request_id="r1")
        self.assertTrue(journal.all_settled())
        self.assertEqual(journal.released_indices, {0, 1})
        journal.assert_conservation()
        # home 侧（rank 0/1）已释放；exec 侧（rank 2/3）= 基础 + 增长。
        self.assertEqual(_handoff_kv_bytes(self.manager)[:2], (0, 0))
        self.assertEqual(
            _handoff_kv_bytes(self.manager)[2:],
            tuple(kv_cache_shard_bytes_for_tokens(self.model, 15, 2)))
        # 逐块事件（各恰一次）+ D_handoff 归零。
        handoffs = [
            event for event in self.manager.copy_handoff_events
            if event["subject"] == "r1#handoff"
            and event["event"] == "handoff"]
        streams = [
            event for event in self.manager.copy_handoff_events
            if event["subject"] == "r1#copy-stream"
            and event["event"] == "stream-once"]
        self.assertEqual([event["chunk"] for event in handoffs], [0, 1])
        self.assertEqual([event["chunk"] for event in streams], [0, 1])
        self.assertEqual(
            sum(event["bytes"] for event in streams), sum(per_rank))
        self.assertEqual(handoffs[-1]["d_handoff_bytes"], 0)
        self.assertEqual(handoffs[-1]["boundary"], "prefill_drain")

    def test_merge_after_settle_zero_byte_no_double_release(self):
        self._prepare_copy()
        self.manager.expand_prefill(
            session_id="s", instance_index=1, context_tokens=15,
            trigger_request_id="r1")
        merge_transfers = self.manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)
        self.assertEqual(merge_transfers, ())
        # home 侧已逐块释放：merge 零字节结算不再扣减（不重复释放），
        # 终态与旧口径一致。
        self.assertEqual(_handoff_kv_bytes(self.manager), (0, 0) + tuple(
            kv_cache_shard_bytes_for_tokens(self.model, 15, 2)))
        merged = self.manager.session_snapshot("s")
        self.assertEqual(merged.home_instance, 1)
        self.assertIsNone(merged.working_kind)
        self.assertIsNone(self.manager._sessions["s"].copy_handoff)
        # 轮末闭合事件入账。
        close = [
            event for event in self.manager.copy_handoff_events
            if event["subject"] == "r1#handoff"
            and event["event"] == "close"]
        self.assertEqual(len(close), 1)
        self.assertEqual(close[0]["d_handoff_bytes"], 0)
        self.assertEqual(close[0]["h_home_bytes"], 0)

    def test_merge_direct_fallback_settles_pending_chunks(self):
        # 直接 merge（未走 drain——直接 API 调用面）：merge 头部防御
        # 兜底结算后零字节闭合。decode 相中段的 expand_decode 不结算
        # （非安全边界）——home 侧保持驻留直到 merge。
        self._prepare_copy()
        self.manager.expand_decode(
            session_id="s", instance_index=1, final_context_tokens=15,
            trigger_request_id="r1")
        journal = self.manager._sessions["s"].copy_handoff
        self.assertFalse(journal.all_settled())
        per_rank = kv_cache_shard_bytes_for_tokens(self.model, 10, 2)
        self.assertEqual(
            _handoff_kv_bytes(self.manager)[:2], per_rank)
        merge_transfers = self.manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=5)
        self.assertEqual(merge_transfers, ())
        self.assertEqual(_handoff_kv_bytes(self.manager), (0, 0) + tuple(
            kv_cache_shard_bytes_for_tokens(self.model, 15, 2)))
        handoffs = [
            event for event in self.manager.copy_handoff_events
            if event["event"] == "handoff"]
        self.assertEqual(
            [event["boundary"] for event in handoffs], ["merge", "merge"])

    def test_incremental_base_contribution_shrinks_with_release(self):
        self._prepare_copy()
        session = self.manager._sessions["s"]
        contribution = self.manager._incremental_base_contribution(session)
        self.assertEqual(contribution[0], 0)
        self.assertEqual(
            contribution[1],
            kv_cache_shard_bytes_for_tokens(self.model, 10, 2))
        journal = session.copy_handoff
        self.manager._apply_copy_handoff_event(
            session, 0, boundary="prefill_drain", trigger_request_id="r1")
        contribution = self.manager._incremental_base_contribution(session)
        self.assertEqual(
            contribution[1],
            kv_cache_shard_bytes_for_layer_range(
                self.model, 10, 2, layer_start=8, layer_end=16))
        self.manager._apply_copy_handoff_event(
            session, 1, boundary="prefill_drain", trigger_request_id="r1")
        self.assertIsNone(
            self.manager._incremental_base_contribution(session))
        self.assertTrue(journal.all_settled())

    def test_unknown_settle_boundary_rejected(self):
        # decode 相中段不是安全边界（多列车 prefill 下尾块可在途）——
        # 拒绝结算 = 交接前释放的边界形式防护。
        self._prepare_copy()
        with self.assertRaisesRegex(RuntimeError, "unknown copy handoff"):
            self.manager._settle_copy_handoffs(
                "s", boundary="decode_mid", trigger_request_id="r1")

    def test_source_safety_rejects_inflight_round(self):
        # 在途读取防护（C12 §5）：上一轮工作副本/账本未闭合（working
        # 副本在途）时源块不可安全交接——动作不适用，fail-closed。直接
        # 驱动守卫（prepare_prefill 顶层对 working 会话另有 copy@home
        # 退化分支先截获，不构成本守卫的注入面）。
        self._prepare_copy()
        session = self.manager._sessions["s"]
        self.assertEqual(session.working_kind, "copy")
        with self.assertRaisesRegex(RuntimeError, "not safely handable"):
            self.manager._assert_copy_handoff_source_safety(session, 0)
        # 闭合后（merge 完成本轮）守卫放行。
        self.manager.merge_back(
            session_id="s", trigger_request_id="r1", new_tokens=0)
        self.assertIsNone(session.working_kind)
        self.manager._assert_copy_handoff_source_safety(session, 0)


class CopyHandoffFailureInjectionTests(unittest.TestCase):
    """失败注入三件（双释放 / 交接前释放 / 重复交接）+ 乱序守卫。"""

    def setUp(self):
        self.model, self.manager = _handoff_manager()
        _seed_home_session(self.manager, tokens=10)
        self.manager.prepare_prefill(
            session_id="s",
            target_instance_index=1,
            history_tokens=10,
            trigger_request_id="r1",
            action="copy",
        )
        self.session = self.manager._sessions["s"]

    def test_injection_duplicate_handoff(self):
        # 重复交接：同一块二次 apply → 合同类违规。
        self.manager._apply_copy_handoff_event(
            self.session, 0, boundary="prefill_drain",
            trigger_request_id="r1")
        with self.assertRaisesRegex(RuntimeError, "duplicate copy handoff"):
            self.manager._apply_copy_handoff_event(
                self.session, 0, boundary="prefill_drain",
                trigger_request_id="r1")

    def test_injection_release_before_handoff(self):
        # 交接前释放：块仍 PENDING（权威性未转移）即请求释放 home 侧。
        chunk = self.session.copy_handoff.chunks[1]
        self.assertEqual(chunk.state, "PENDING")
        with self.assertRaisesRegex(
                RuntimeError, "before its handoff completed"):
            self.manager._release_copy_handoff_chunk(
                self.session, 1,
                boundary="prefill_drain", trigger_request_id="r1")

    def test_injection_double_release(self):
        # 双释放：已释放的块再次进入释放路径。
        self.manager._apply_copy_handoff_event(
            self.session, 0, boundary="prefill_drain",
            trigger_request_id="r1")
        self.assertIn(0, self.session.copy_handoff.released_indices)
        with self.assertRaisesRegex(RuntimeError, "double release"):
            self.manager._release_copy_handoff_chunk(
                self.session, 0,
                boundary="prefill_drain", trigger_request_id="r1")
        # 守恒未被破坏（home 计数器仍与块状态一致）。
        self.session.copy_handoff.assert_conservation()

    def test_injection_out_of_order_handoff(self):
        # 乱序交接：块 1 先于块 0（消费顺序协议）。
        with self.assertRaisesRegex(RuntimeError, "out-of-order"):
            self.manager._apply_copy_handoff_event(
                self.session, 1, boundary="prefill_drain",
                trigger_request_id="r1")


# ------------------------------------------------------------- GB 夹具 --

HOME_RANKS = (0, 1)
EXEC_RANKS = (2, 3)
SESSION_G = "session_copy_g"
REQUEST_G = "session_copy_g_request_0"


def _graph_config(layers: int = 16):
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=HOME_RANKS, pg_name="tp_prefill"),
            SimpleNamespace(ranks=EXEC_RANKS, pg_name="tp_decode"),
        ],
        layers=layers,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _handoff_transfer(chunk_index, layer_start, layer_end, *, bytes_,
                      model_layers: int = 16):
    return KVTransfer(
        kind="noc_migrate",
        phase="history",
        reason=("history_prefix_working_copy" if chunk_index == 0
                else "history_prefix_handoff_tail"),
        session_id=SESSION_G,
        trigger_request_id=REQUEST_G,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=2 * bytes_,
        shards=tuple(
            KVTransferShard(
                source_rank=source, target_rank=source + 2, edge_rank=None,
                bytes=bytes_, noc_path=(source, source + 2),
                layer_start=layer_start, layer_end=layer_end)
            for source in HOME_RANKS),
        model_layers=model_layers,
        layer_start=layer_start,
        layer_end=layer_end,
        resident_prefix_layers_before=layer_start,
        resident_prefix_layers_after=layer_end,
        handoff_chunk=chunk_index,
    )


def _copy_admission_plan():
    return {
        "request_id": REQUEST_G,
        "session_id": SESSION_G,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 1,
        "decode_instance_index": 1,
        "admission_time_ns": 1000,
        "history_location_before": None,
        "history_transfer": None,
        "history_transfers": [
            _handoff_transfer(0, 0, 8, bytes_=400),
            _handoff_transfer(1, 8, 16, bytes_=400),
        ],
        "history_evictions": [],
        "prefill_evictions": [],
        "history_tokens_before": 10,
        "prefill_context_tokens": 300,
        "joint_action": "copy",
        "completion_evictions": [],
        "merge_transfers": [],
    }


def _copy_admission_plan_24():
    """24 层 3 块变体（T2：handoff 块 0/1/2——两个尾块）。"""
    plan = dict(_copy_admission_plan())
    plan["history_transfers"] = [
        _handoff_transfer(0, 0, 8, bytes_=400, model_layers=24),
        _handoff_transfer(1, 8, 16, bytes_=400, model_layers=24),
        _handoff_transfer(2, 16, 24, bytes_=400, model_layers=24),
    ]
    return plan


def _copy_train_plan(train_id, spans, *, exits=False):
    plan = {
        "train_id": train_id,
        "instance_index": 1,
        "stage": "prefill",
        "joiners": [],
        "members": [],
        "pass_spans": list(spans),
        "iterations": len(spans),
        "prefill_chunk_tokens": [(REQUEST_G, span[0]) for span in spans],
        "prefill_start_member": _copy_admission_plan(),
        "first_chunk_member": _copy_admission_plan(),
        "drain_members": [],
        "exit_members": [],
        "head_request_id": REQUEST_G,
    }
    if exits:
        plan["exit_members"] = [_copy_admission_plan()]
    return plan


def _copy_train_plan_24(train_id, spans):
    plan = dict(_copy_train_plan(train_id, spans))
    plan["prefill_start_member"] = _copy_admission_plan_24()
    plan["first_chunk_member"] = _copy_admission_plan_24()
    return plan


class CopyHandoffGraphStructureTests(unittest.TestCase):
    def setUp(self):
        self.builder = GraphBatchBuilder(_graph_config())
        self.builder.begin_batch()

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _edges(self):
        return self.builder.batch["parent_edges"]

    def _parents_by_rank(self):
        parents = {}
        for edge in self._edges():
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        return parents

    def _emit_admission(self):
        # fork frontier：先发射一趟先行列车，尾块支链 fork 点在其后。
        self.builder.emit_iteration_train(
            _copy_train_plan("batch_train_i1_1", [(1, 101)]))
        fork = {
            rank: self.builder.builders[rank].previous_id
            for rank in EXEC_RANKS + HOME_RANKS}
        self.builder.emit_admission_batch(_copy_admission_plan())
        return fork

    def test_tail_branch_arms_and_release_anchors(self):
        self._emit_admission()
        arms = self.builder._copy_handoff_arms.get(REQUEST_G)
        anchors = self.builder._copy_handoff_release_anchors.get(REQUEST_G)
        self.assertIsNotNone(arms)
        self.assertIsNotNone(anchors)
        self.assertEqual(sorted(arms), [1])
        self.assertEqual(sorted(anchors), [1])
        self.assertEqual(sorted(arms[1]), list(EXEC_RANKS))
        self.assertEqual(sorted(anchors[1]), list(HOME_RANKS))
        names = {node["name"] for node in self._nodes()}
        self.assertTrue(
            any("_handoff1_recv" in name for name in names))
        self.assertTrue(
            any("_handoff1_ack_from_rank" in name for name in names))
        # 头块（块 0）走主链 history_transfer 命名（非 handoff 支链名）。
        self.assertTrue(
            any("history_transfer" in name and "action" in name
                and "noc_migrate" in name for name in names))

    def test_fork_order_tail_after_head(self):
        self._emit_admission()
        parents = self._parents_by_rank()
        for rank in HOME_RANKS + EXEC_RANKS:
            tail_nodes = sorted(
                node["id"] for node in self._nodes()
                if "_handoff1_" in node["name"] and node["rank"] == rank)
            self.assertTrue(tail_nodes)
            # fork frontier = 头块发射后该 rank 主链的末节点（home = 头块
            # ack_recv；exec = 头块 ack_send）。
            head_frontier = max(
                node["id"] for node in self._nodes()
                if "history_transfer" in node["name"]
                and node["rank"] == rank)
            self.assertIn(
                head_frontier, parents.get((rank, min(tail_nodes)), ()),
                "copy handoff tail branch does not fork from the "
                "post-head main-chain frontier")

    def test_readiness_barrier_waits_head_only(self):
        self._emit_admission()
        parents = self._parents_by_rank()
        barrier_nodes = {
            node["rank"]: node["id"] for node in self._nodes()
            if node["name"].endswith("_prefill_kv_ready_barrier")}
        self.assertEqual(sorted(barrier_nodes), list(EXEC_RANKS))
        tail_ids = {
            node["id"] for node in self._nodes()
            if "_handoff1_" in node["name"]}
        for rank, barrier in barrier_nodes.items():
            # barrier 直接父节点不含尾块（尾块不 join 主链）。
            self.assertFalse(
                set(parents.get((rank, barrier), ())) & tail_ids,
                "readiness barrier waits for a copy tail chunk "
                "(whole-history serialization)")

    def test_train_body_gates_on_tail_ready_event(self):
        self._emit_admission()
        recv_nodes = {
            rank: node_id for rank, node_id
            in self.builder._copy_handoff_arms[REQUEST_G][1].items()}
        self.builder.emit_iteration_train(
            _copy_train_plan(
                "batch_train_i1_2", [(1, 201), (1, 202), (1, 203)]))
        # 尾块 recv 完成门被覆盖其迭代区间的体块节点 arm 消费（逐块
        # 就绪门控；体块节点归属列车命名空间）。
        body_nodes = {
            node["id"] for node in self._nodes()
            if node["request_id"] == "batch_train_i1_2"}
        armed_targets = {
            edge["to"] for edge in self._edges()
            if edge["from"] == recv_nodes.get(edge["rank"])}
        self.assertTrue(
            armed_targets & body_nodes,
            "train body did not arm on the copy tail recv gate")
        # 消费后账本弹出（joiner 恰一次）。
        self.assertNotIn(REQUEST_G, self.builder._copy_handoff_arms)

    def test_completion_pops_release_anchors(self):
        self._emit_admission()
        self.builder.emit_iteration_train(
            _copy_train_plan("batch_train_i1_2", [(1, 201)], exits=True))
        self.assertIn(REQUEST_G, self.builder._copy_handoff_release_anchors)
        self.builder.next_plan[REQUEST_G] = None
        self.builder.emit_completion_batch(_copy_admission_plan())
        self.assertNotIn(
            REQUEST_G, self.builder._copy_handoff_release_anchors)
        self.assertNotIn(REQUEST_G, self.builder._copy_handoff_arms)

    def test_completion_with_leftover_arms_fails_closed(self):
        self._emit_admission()
        # 夹具：手工落 seg2 块末（完成路径前置账），保持交接门账本
        # 残留（尾块从未被体块消费）——完成边界必须 fail-closed。
        barrier = max(node["id"] for node in self._nodes())
        self.builder._block_ends[REQUEST_G] = {
            "seg1": {rank: barrier for rank in EXEC_RANKS},
            "seg2": {rank: barrier for rank in EXEC_RANKS},
        }
        self.builder.next_plan[REQUEST_G] = None
        with self.assertRaisesRegex(RuntimeError, "unconsumed copy"):
            self.builder.emit_completion_batch(_copy_admission_plan())


class CopyHandoffParallelTailTests(unittest.TestCase):
    """T2（P1 锚，PARTIAL 跨实例 copy 流水化 2026-09-25）：尾块逐笔并行
    支链（layers=24 → handoff 块 0/1/2，两个尾块）：

      (a) 块间无边：同 home rank 的 handoff1/handoff2 send 父集合相等且
          = 头块发射后的主链 frontier（exec 侧 recv 同理）——首块之后的
          D2D 事务多笔同时在飞的图侧前提；
      (b) 块内新形态边：ack_send_c 父 = {recv_c}、ack_recv_c 父 =
          {send_c}；跨 rank send/recv 无图边（因果由 tag 配对承载）；
      (c) 三账本对块 1/2 登记完整且落 EXEC_RANKS/HOME_RANKS；
      (d) readiness barrier 直连父与祖先闭包都不含任何尾块（首块仍是
          唯一启动栅栏）；
      (e) 列车体逐块消费全部尾块门（账本弹空）。
    """

    def setUp(self):
        self.builder = GraphBatchBuilder(_graph_config(layers=24))
        self.builder.begin_batch()

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _parents_by_rank(self):
        parents = {}
        for edge in self.builder.batch["parent_edges"]:
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        return parents

    def _find_node(self, rank, name_fragment):
        matches = [
            node for node in self._nodes()
            if node["rank"] == rank and name_fragment in node["name"]]
        self.assertEqual(
            len(matches), 1,
            f"expected exactly one node on rank {rank} matching "
            f"{name_fragment!r}, got {len(matches)}")
        return matches[0]

    def _emit_admission(self):
        self.builder.emit_iteration_train(
            _copy_train_plan_24("batch_train_i1_1", [(1, 101)]))
        self.builder.emit_admission_batch(_copy_admission_plan_24())

    def _head_frontier(self, rank):
        # fork frontier = 头块发射后该 rank 主链的末节点（同
        # test_fork_order_tail_after_head 口径）。
        return max(
            node["id"] for node in self._nodes()
            if "history_transfer" in node["name"] and node["rank"] == rank)

    def test_tail_chunks_fork_frontier_with_no_inter_chunk_edges(self):
        self._emit_admission()
        parents = self._parents_by_rank()
        for rank in HOME_RANKS:
            frontier = self._head_frontier(rank)
            send_parents = {
                chunk: set(parents.get(
                    (rank,
                     self._find_node(rank, f"_handoff{chunk}_send")["id"]),
                    ()))
                for chunk in (1, 2)}
            self.assertEqual(
                send_parents[1], {frontier},
                "tail chunk send must fork from the post-head frontier")
            self.assertEqual(
                send_parents[1], send_parents[2],
                "tail chunks must not chain to each other on the home "
                "rank (parallel side branches)")
        for rank in EXEC_RANKS:
            frontier = self._head_frontier(rank)
            recv_parents = {
                chunk: set(parents.get(
                    (rank,
                     self._find_node(rank, f"_handoff{chunk}_recv")["id"]),
                    ()))
                for chunk in (1, 2)}
            self.assertEqual(
                recv_parents[1], {frontier},
                "tail chunk recv must fork from the post-head frontier")
            self.assertEqual(
                recv_parents[1], recv_parents[2],
                "tail chunks must not chain to each other on the exec "
                "rank (parallel side branches)")

    def test_chunk_internal_ack_edges_new_shape(self):
        self._emit_admission()
        parents = self._parents_by_rank()
        for chunk in (1, 2):
            for source in HOME_RANKS:
                target = source + 2
                send_id = self._find_node(
                    source, f"_handoff{chunk}_send")["id"]
                recv_id = self._find_node(
                    target, f"_handoff{chunk}_recv")["id"]
                ack_to = self._find_node(
                    target, f"_handoff{chunk}_ack_to_rank{source}")
                ack_from = self._find_node(
                    source, f"_handoff{chunk}_ack_from_rank{target}")
                self.assertEqual(
                    set(parents.get((target, ack_to["id"]), ())), {recv_id},
                    "ack_send must hang on its own chunk's recv (new "
                    "block-internal shape), not on a tail ack chain")
                self.assertEqual(
                    set(parents.get((source, ack_from["id"]), ())), {send_id},
                    "ack_recv must hang on its own chunk's send (new "
                    "block-internal shape), not on a tail ack chain")

    def test_ledgers_register_every_tail_chunk_on_exec_ranks(self):
        self._emit_admission()
        arms = self.builder._copy_handoff_arms.get(REQUEST_G)
        layers = self.builder._copy_handoff_layers.get(REQUEST_G)
        anchors = self.builder._copy_handoff_release_anchors.get(REQUEST_G)
        self.assertEqual(sorted(arms), [1, 2])
        self.assertEqual(sorted(layers), [1, 2])
        self.assertEqual(sorted(anchors), [1, 2])
        self.assertEqual(layers[1], (8, 16))
        self.assertEqual(layers[2], (16, 24))
        for chunk in (1, 2):
            self.assertEqual(sorted(arms[chunk]), list(EXEC_RANKS),
                             "recv gates must land on the exec instance")
            self.assertEqual(sorted(anchors[chunk]), list(HOME_RANKS),
                             "release anchors must land on the home ranks")

    def test_readiness_barrier_ancestors_exclude_all_tail_chunks(self):
        self._emit_admission()
        parents = self._parents_by_rank()
        barrier_nodes = {
            node["rank"]: node["id"] for node in self._nodes()
            if node["name"].endswith("_prefill_kv_ready_barrier")}
        self.assertEqual(sorted(barrier_nodes), list(EXEC_RANKS))
        # 节点 id 是 per-rank 空间——尾块 id 集合按 rank 收集。
        tail_ids_by_rank = {}
        for node in self._nodes():
            if "_handoff1_" in node["name"] or "_handoff2_" in node["name"]:
                tail_ids_by_rank.setdefault(node["rank"], set()).add(
                    node["id"])
        for rank, barrier in barrier_nodes.items():
            tail_ids = tail_ids_by_rank.get(rank, set())
            # 直连父不含尾块。
            self.assertFalse(
                set(parents.get((rank, barrier), ())) & tail_ids,
                "readiness barrier waits for a copy tail chunk "
                "(whole-history serialization)")
            # 祖先闭包同样不含尾块（首块是唯一启动栅栏的完整口径）。
            seen = {barrier}
            frontier = [barrier]
            while frontier:
                current = frontier.pop()
                for parent in parents.get((rank, current), ()):
                    if parent not in seen:
                        seen.add(parent)
                        frontier.append(parent)
            self.assertFalse(
                seen & tail_ids,
                "readiness barrier ancestor closure reaches a copy tail "
                "chunk")

    def test_train_body_gates_on_every_tail_chunk(self):
        self._emit_admission()
        recv_gates = {
            chunk: dict(self.builder._copy_handoff_arms[REQUEST_G][chunk])
            for chunk in (1, 2)}
        self.builder.emit_iteration_train(
            _copy_train_plan_24(
                "batch_train_i1_2", [(1, 201), (1, 202), (1, 203)]))
        body_nodes = {
            node["id"] for node in self._nodes()
            if node["request_id"] == "batch_train_i1_2"}
        parents = self._parents_by_rank()
        for chunk in (1, 2):
            for rank, gate in recv_gates[chunk].items():
                armed = {
                    target for (node_rank, target) in parents
                    if node_rank == rank
                    and gate in parents[(node_rank, target)]}
                self.assertTrue(
                    armed & body_nodes,
                    f"chunk {chunk} recv gate on rank {rank} did not arm a "
                    "train-body layer segment")
        # 全部尾块消费后账本弹空（joiner 恰一次）。
        self.assertNotIn(REQUEST_G, self.builder._copy_handoff_arms)
        self.assertNotIn(REQUEST_G, self.builder._copy_handoff_layers)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
