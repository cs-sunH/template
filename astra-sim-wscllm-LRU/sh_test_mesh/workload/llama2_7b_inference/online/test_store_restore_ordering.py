#!/usr/bin/env python3
"""test_store_restore_ordering.py -- store→restore 前递依赖排序钉子测试
(2026-09-13,KV 逐出与 request 推理并行化,主方案 §3.3)。

逐出支链悬空后,"同会话逐出池写先于其下一轮池读"的传递性保障失效,
回迁发射统一入口(REMOTE_RESTORE 全量 remote_load,唯一远端恢复路径)
发射前查 pending_store_tails 补边。断言:

  1. 全量回迁(同缘):restore 边缘 rank 的回迁链首节点直接依赖该
     会话全部在飞 store 的 mem_store 节点——整体逐出每会话恰一条
     store(restore 读全区间);
  2. 全量回迁(跨缘):store 边缘发 1B 中继(data_dep 挂 mem_store)、
     回迁边缘收 1B,其后的池读(mem_load)链在中继之后(桥拒绝跨
     rank 直边,1B p2p 是协议内唯一载体);
  3. REMOTE_RESTORE 全量回迁:补边落在全量恢复发射上(单一全层就绪
     屏障保障恢复完成前不消费;prefix/suffix 双栅栏已随 PARTIAL 流水
     删除);
  4. retire_terminal_session 边界(retire_completion_gate)清登记;
  5. 回迁一次性消费(2026-09-26 M2-01):补边完成后登记表弹出该会话
     ——多轮「逐出→回迁」不再对早已满足的旧 store 重复补边/重发 1B
     中继(补边目标早已完成,弹出保持发射时序不变)。
  6. C-2/M-25 修复回归(2026-09-30):#2 预占逐出决策时点就近发射
     (emit_decode_eviction_actions)——倒序发射(受害者回迁先于逐出者
     store 发射)形态下回迁仍被门住;terminal retire 后无泄漏;空表 +
     已决策未发射 store 探针命中 fail-closed。

Run: python3 online/test_store_restore_ordering.py
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

from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from session_kv_manager import KVTransfer, KVTransferShard  # noqa: E402

# 4x4 mesh:rank 5=(1,1) 近缘 edge 1;rank 6=(1,2) 近缘 edge 2;
# rank 0/1 为边界 rank(直连池端点,近缘 = 自身)。
PREFILL_RANKS = (0, 1)
DECODE_RANKS = (5, 6)
VICTIM = "victim_store_restore"
LAYERS = 4


def _make_config():
    return SimpleNamespace(
        npus_count=16,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(name="g_prefill", ranks=PREFILL_RANKS,
                            pg_name="tp_prefill"),
            SimpleNamespace(name="g_decode", ranks=DECODE_RANKS,
                            pg_name="tp_decode"),
        ],
        prefill_chunk_size=128,
        layers=LAYERS,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(session_arrival_time_ns=5000,
                            inter_request_interval_ns=None),
            SimpleNamespace(session_arrival_time_ns=None,
                            inter_request_interval_ns=1000),
        ],
        model=SimpleNamespace(layers=LAYERS, hidden_size=64,
                              bytes_per_elem=2),
        hardware=SimpleNamespace(mesh_rows=4, mesh_cols=4, npus_count=16),
    )


def _shard(**kwargs):
    return KVTransferShard(
        source_rank=kwargs.get("source_rank"),
        target_rank=kwargs.get("target_rank"),
        edge_rank=kwargs.get("edge_rank"),
        bytes=kwargs["bytes"],
        noc_path=tuple(kwargs.get("noc_path", ())),
        layer_start=kwargs.get("layer_start", 0),
        layer_end=kwargs.get("layer_end", LAYERS),
    )


def _whole_store():
    """整体逐出(session 级唯一逐出形态):decode rank 5→edge 1、
    rank 6→edge 2,全层域 [0,4),本地驻留清零。两 shard 行 = 单逻辑
    victim(同一条 KVTransfer、同一会话)。"""
    shards = (
        _shard(source_rank=5, target_rank=1, edge_rank=1, bytes=1500,
               noc_path=(5, 1)),
        _shard(source_rank=6, target_rank=2, edge_rank=2, bytes=1500,
               noc_path=(6, 2)),
    )
    return KVTransfer(
        kind="remote_store", phase="history",
        reason="watermark_admission_session",
        session_id=VICTIM, trigger_request_id="req_evict",
        source_instance_index=1, target_instance_index=None,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=0,
    )


def _load_transfer(layer_start, layer_end, targets, target_instance=0):
    """remote_load:shards 按 (target_rank, edge_rank) 对给定,层域
    [layer_start, layer_end)。REMOTE_RESTORE 全量回迁恒 [0, L)。"""
    shards = tuple(
        _shard(source_rank=edge, target_rank=target, edge_rank=edge,
               bytes=900, noc_path=(edge, target),
               layer_start=layer_start, layer_end=layer_end)
        for target, edge in targets)
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_remote_restore", session_id=VICTIM,
        trigger_request_id="req_restore",
        source_instance_index=None, target_instance_index=target_instance,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=layer_start,
        resident_prefix_layers_after=layer_end,
    )


def _restore_plan(*, prefill_instance=0, history_source=1,
                  history_action=None,
                  transfers=(), resident_layers=None, location=None):
    plan = {
        "request_id": "req_restore_turn1",
        "session_id": VICTIM,
        "turn_index": 1,
        "queue_index": 1,
        "prefill_instance_index": prefill_instance,
        "decode_instance_index": 1,
        "history_action": history_action,
        "history_source_instance_index": history_source,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 64,
        "prefill_context_tokens": 64 + 256,
        "prefill_length": 256,
        "decode_length": 4,
        "history_transfers": tuple(transfers),
        "history_evictions": (),
        "prefill_evictions": (),
    }
    if resident_layers is not None:
        plan["history_resident_prefix_layers"] = resident_layers
    if location is not None:
        plan["history_location_before"] = location
    return plan


def _evict_session_via_train(builder, evictions):
    """批次 1:joiner decode 逐出支链发射 + store 尾部登记。"""
    prefill_plan = {
        "request_id": "req_evict_turn0", "session_id": "session_other",
        "turn_index": 0, "queue_index": 0,
        "prefill_instance_index": 0, "decode_instance_index": 1,
        "history_action": None, "history_source_instance_index": None,
        "history_transfer_bytes": 0, "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_context_tokens": 320, "prefill_length": 256,
        "decode_length": 4, "history_transfers": (),
        "history_evictions": (), "prefill_evictions": (),
    }
    builder.emit_prefill_batch(prefill_plan)
    builder.begin_batch()
    joiner = dict(prefill_plan, decode_evictions=tuple(evictions))
    builder.emit_iteration_train({
        "train_id": "batch_train_i1_evict",
        "instance_index": 1,
        "joiners": [joiner],
        "pass_spans": [(1, 321)],
        "iterations": 1,
        "exit_members": [],
    })


def _emit_reserve_eviction_at_decision_time(builder, evictions, *, gated=True):
    """C-2/M-25 修复后形态(2026-09-30):逐出者 prefill 批 + #2 预占逐出
    决策时点就近发射(emit_decode_eviction_actions)——store 尾部随发射
    登记,先于受害会话任何后续回迁/退休。gated=True = 段块末触发门
    (准入成功随 prefill 批);False = 无门(阻塞 attempt 立即发射)。"""
    evictor_plan = {
        "request_id": "req_evict_turn0", "session_id": "session_other",
        "turn_index": 0, "queue_index": 0,
        "prefill_instance_index": 0, "decode_instance_index": 1,
        "history_action": None, "history_source_instance_index": None,
        "history_transfer_bytes": 0, "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_context_tokens": 320, "prefill_length": 256,
        "decode_length": 4, "history_transfers": (),
        "history_evictions": (), "prefill_evictions": (),
    }
    builder.emit_prefill_batch(evictor_plan)
    builder.emit_decode_eviction_actions(
        evictor_plan, tuple(evictions),
        gated_on_prefill_segment_end=gated)


def _nodes_of(nodes, rank=None, fragment=None, suffix=None):
    pool = nodes if rank is None else [n for n in nodes if n["rank"] == rank]
    if fragment is not None:
        pool = [n for n in pool if fragment in n["name"]]
    if suffix is not None:
        pool = [n for n in pool if n["name"].endswith(suffix)]
    return pool


def _parents_of(edges, rank, node_id):
    return {
        edge["from"] for edge in edges
        if edge["rank"] == rank and edge["to"] == node_id}


def _ancestors_of(edges, rank, node_id):
    parents = {}
    for edge in edges:
        if edge["rank"] == rank:
            parents.setdefault(edge["to"], set()).add(edge["from"])
    seen = {node_id}
    frontier = [node_id]
    while frontier:
        for parent in parents.get(frontier.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                frontier.append(parent)
    return seen


def _store_tail_ids(builder, session=VICTIM):
    """登记表里该会话的 (edge_rank, mem_store_id) 对。"""
    return [(edge, store) for edge, store
            in builder.pending_store_tails.get(session, ())]


class FullRestoreSameEdgeOrderingTest(unittest.TestCase):
    """1. 逐出后下一轮全量回迁(同缘):整体逐出的单条 store 被直接依赖。"""

    def test_whole_session_store_gates_the_restore_chain(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_whole_store(),))
        tails = dict()
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        # 每个边缘恰 1 笔 store(整体逐出单条;两 shard 行 = 单 victim)。
        self.assertEqual({edge: len(ids) for edge, ids in tails.items()},
                         {1: 1, 2: 1})
        # 回迁批次:V 的 turn-1 全量回迁到 decode 实例(同缘 edge 1/2)。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=1, history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((5, 1), (6, 2)),
                                      target_instance=1),),
            location="remote_memory"))
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        # edge 1 的回迁链首节点(1B request recv):直接依赖该 store 的
        # mem_store 节点(arm 在发射前)。
        first_on_edge1 = min(
            (n for n in _nodes_of(nodes, 1)
             if "_remote_load" in n["name"] or "_request_from_rank" in n["name"]),
            key=lambda n: n["id"])
        parents = _parents_of(edges, 1, first_on_edge1["id"])
        self.assertEqual(parents & tails[1], tails[1],
                         "the whole-session store must gate the restore chain")
        # mem_load 传递性晚于 mem_store。
        load = _nodes_of(nodes, 1, suffix="_remote_load")
        self.assertEqual(len(load), 1)
        self.assertTrue(
            tails[1] <= _ancestors_of(edges, 1, load[0]["id"]))
        # edge 2 同构。
        load2 = _nodes_of(nodes, 2, suffix="_remote_load")
        self.assertEqual(len(load2), 1)
        self.assertTrue(
            tails[2] <= _ancestors_of(edges, 2, load2[0]["id"]))
        # 登记表一次性消费(M2-01,2026-09-26):本轮回迁已把全部在飞
        # store 尾部补边消费完毕,登记表不再保留该会话(后续回迁若需
        # 补边,只会针对再逐出的新 store)。
        self.assertNotIn(VICTIM, builder.pending_store_tails)


class FullRestoreCrossEdgeRelayTest(unittest.TestCase):
    """2. 逐出后下一轮全量回迁(跨缘):1B 中继承载跨缘时序。"""

    def test_relay_carries_store_ordering_across_edges(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_whole_store(),))
        tails = dict()
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        self.assertEqual(set(tails), {1, 2})
        # 回迁到 prefill 实例(0,1):restore 经 edge 0/1 读池。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=0, history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((0, 0), (1, 1))),),
            location="remote_memory"))
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        # 同缘 tail(edge 1):rank 1 回迁链首节点直接依赖 mem_store。
        first_on_edge1 = min(
            _nodes_of(nodes, 1, "_remote_load"),
            key=lambda n: n["id"])
        self.assertTrue(
            tails[1] <= _parents_of(edges, 1, first_on_edge1["id"]))
        # 跨缘 tail(edge 2):store 边缘的 1B 中继 send 直接依赖
        # mem_store;回迁边缘(edge 0 与 1)各有中继 recv,其后的池读
        # 链在中继之后。
        for target_edge in (0, 1):
            relay_recv = _nodes_of(
                nodes, target_edge, "_store_tail_relay_from_e2")
            self.assertEqual(len(relay_recv), 1,
                             "missing relay recv on edge {}".format(
                                 target_edge))
            load = _nodes_of(nodes, target_edge, suffix="_remote_load")
            self.assertEqual(len(load), 1)
            self.assertIn(relay_recv[0]["id"],
                          _ancestors_of(edges, target_edge, load[0]["id"]),
                          "pool read must chain after the relay")
        relay_send = _nodes_of(nodes, 2, "_store_tail_relay_e2")
        self.assertEqual(len(relay_send), 2)  # 2→0 与 2→1
        for send in relay_send:
            self.assertTrue(
                tails[2] <= _parents_of(edges, 2, send["id"]),
                "relay send must depend on the store mem_store")
        # 中继 tag 走 TransferTagAllocator(H8/2026-09-24 后基址 10^8;旧
        # 10^7 基址已与 queue_index*10000 的 _stage_tag 段重叠而弃用)。
        for send in relay_send:
            self.assertGreaterEqual(send["comm"]["tag"], 100_000_000)


class RemoteRestoreFullLayerOrderingTest(unittest.TestCase):
    """3. REMOTE_RESTORE 全量回迁:补边 + 单一全层就绪屏障(无层段拆分)。"""

    def test_full_restore_gated_by_store_tails_with_single_barrier(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_whole_store(),))
        tails = dict()
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        self.assertEqual(set(tails), {1, 2})
        # REMOTE_RESTORE 全量回迁:门源 = 上一 turn decode 实例 1,
        # 全层 [0,4) 恢复到 prefill 实例 0(经 edge 0/1)。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=0, history_source=1,
            history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((0, 0), (1, 1))),),
            resident_layers=0, location="remote_memory"))
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        # 全量恢复链的 mem_load 传递性晚于全部在飞 store(同缘直接依赖,
        # 跨缘经中继)。
        for rank in PREFILL_RANKS:
            load = _nodes_of(nodes, rank, suffix="_remote_load")
            self.assertEqual(len(load), 1)
            for edge, store_ids in tails.items():
                for store_id in store_ids:
                    ancestors = _ancestors_of(edges, rank, load[0]["id"])
                    if edge == rank:
                        self.assertIn(store_id, ancestors)
            # 单一全层就绪屏障;prefix/suffix 双栅栏与层段拆分不存在。
            barriers = _nodes_of(nodes, rank, "history_tp_ready_barrier")
            self.assertEqual(len(barriers), 1)
            for banned in ("prefix_ready_barrier", "suffix_ready_barrier",
                           "first_chunk_prefix", "first_chunk_suffix"):
                self.assertFalse(_nodes_of(nodes, rank, banned), banned)


class RetireClearsStoreTailsTest(unittest.TestCase):
    """4. retire_terminal_session 边界清登记。"""

    def test_retire_completion_gate_clears_tails(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_whole_store(),))
        self.assertIn(VICTIM, builder.pending_store_tails)
        builder.retire_completion_gate(VICTIM)
        self.assertNotIn(VICTIM, builder.pending_store_tails)


class RestoreConsumesStoreTailsTest(unittest.TestCase):
    """5. 回迁一次性消费(M2-01,2026-09-26):弹出 + 不重复补边。"""

    @staticmethod
    def _tails_by_edge(builder, session=VICTIM):
        """登记表里该会话的 {edge_rank: {mem_store_id}} 视图。"""
        tails = {}
        for edge, store in _store_tail_ids(builder, session):
            tails.setdefault(edge, set()).add(store)
        return tails

    def test_second_restore_does_not_rearm_consumed_store(self):
        """多轮「逐出→回迁」:第二轮回迁只为第二轮 store 补边,首轮已
        消费的 store 不再产生冗余依赖/重复 1B 中继(发射时序不变)。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        # 第一轮逐出(store S1)→ 登记表持有 S1。
        _evict_session_via_train(builder, (_whole_store(),))
        first_tails = self._tails_by_edge(builder)
        self.assertEqual({edge: len(ids) for edge, ids in first_tails.items()},
                         {1: 1, 2: 1})
        # 第一轮回迁:补边消费 S1,登记表弹出该会话。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=0, history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((0, 0), (1, 1))),),
            location="remote_memory"))
        self.assertNotIn(VICTIM, builder.pending_store_tails)
        self.assertEqual(len(_nodes_of(
            list(builder.batch["nodes"]), 2, "_store_tail_relay_e2")), 2)

        # 第二轮:再逐出(store S2)→ 只登记 S2(全新 mem_store 节点)。
        builder.begin_batch()
        _evict_session_via_train(builder, (_whole_store(),))
        second_tails = self._tails_by_edge(builder)
        self.assertEqual({edge: len(ids) for edge, ids in second_tails.items()},
                         {1: 1, 2: 1})
        for edge in (1, 2):
            self.assertFalse(
                second_tails[edge] & first_tails[edge],
                "re-eviction must register a fresh store tail")
        # 第二轮回迁:恰好为 S2 补边(同缘 arm + 跨缘 2→0/2→1 中继各一),
        # 不再为首轮旧 store 发射任何额外中继/冗余依赖。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn2", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(dict(
            _restore_plan(
                prefill_instance=0, history_action="REMOTE_RESTORE",
                transfers=(_load_transfer(0, LAYERS, ((0, 0), (1, 1))),),
                location="remote_memory"),
            request_id="req_restore_turn2"))
        edges = list(builder.batch["parent_edges"])
        self.assertNotIn(VICTIM, builder.pending_store_tails)
        relay_sends = _nodes_of(
            list(builder.batch["nodes"]), 2, "_store_tail_relay_e2")
        self.assertEqual(
            len(relay_sends), 2,
            "second restore must relay only the fresh store tail once "
            "per target edge")
        for send in relay_sends:
            parents = _parents_of(edges, 2, send["id"])
            self.assertTrue(
                second_tails[2] <= parents,
                "relay send must depend on the fresh store mem_store")
            self.assertFalse(
                first_tails[2] & parents,
                "consumed first-round store must not gate again")

    def test_same_request_prefill_eviction_still_registers_after_restore(self):
        """同 request 内的登记次序不受弹出影响:history 回迁消费在飞尾部
        后,其后的 prefill 增长逐出照常重新登记(下一轮回迁仍被门住)。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_whole_store(),))
        initial_tails = self._tails_by_edge(builder)
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        # turn-1 plan:history 回迁消费旧尾部 + prefill 增长逐出(S2)——
        # 发射后登记表恰持 S2(弹出只清已消费部分,不影响其后登记)。
        plan = _restore_plan(
            prefill_instance=0, history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((0, 0), (1, 1))),),
            location="remote_memory")
        plan["prefill_evictions"] = (_whole_store(),)
        builder.emit_prefill_batch(plan)
        remaining = self._tails_by_edge(builder)
        self.assertEqual({edge: len(ids) for edge, ids in remaining.items()},
                         {1: 1, 2: 1})
        for edge in (1, 2):
            self.assertFalse(
                remaining[edge] & initial_tails[edge],
                "prefill eviction must register a fresh store tail after "
                "the history restore consumed the previous one")


class DecisionTimeEmissionOrderingTest(unittest.TestCase):
    """6. C-2/M-25 修复回归(2026-09-30):#2 预占逐出决策时点就近发射。

    旧行为把预占逐出随 joiner 首列车发射,决策(prefill 准入)到发射
    隔着逐出者整个 prefill 时长——窗口内受害会话回迁查不到登记(无边
    无异常)、已 terminal 会话 store 退休后仍物理发射且登记条目泄漏。
    修复后 store 支链在决策时点进入图,倒序发射形态不可达。"""

    def test_restore_emitted_after_decision_time_store_is_gated(self):
        """倒序发射场景回归:决策时点 store(批次 1)→ 受害者回迁
        (批次 2)——回迁链必须被 store 尾部门住(旧列车头发射路径下
        此形态无边、无异常、审计干净)。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _emit_reserve_eviction_at_decision_time(
            builder, (_whole_store(),), gated=True)
        tails = {}
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        self.assertEqual({edge: len(ids) for edge, ids in tails.items()},
                         {1: 1, 2: 1})
        # 批次 1 同时持有 prefill 主体与 store 支链(同一 GraphBatch)。
        self.assertTrue(_nodes_of(
            list(builder.batch["nodes"]), 1, "_remote_store"))
        # 批次 2:受害者回迁——store→restore 前递边必须存在。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=1, history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((5, 1), (6, 2)),
                                      target_instance=1),),
            location="remote_memory"))
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        for edge in (1, 2):
            load = _nodes_of(nodes, edge, suffix="_remote_load")
            self.assertEqual(len(load), 1)
            self.assertTrue(
                tails[edge] <= _ancestors_of(edges, edge, load[0]["id"]),
                "decision-time store must gate the later restore")
        self.assertNotIn(VICTIM, builder.pending_store_tails)

    def test_gated_emission_requires_prefill_segment_end(self):
        """段块末触发门:准入成功形态的发射门 = prefill 段块末节点
        (与旧行为同门);段块末缺席 fail-closed。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        evictor_plan = {
            "request_id": "req_evict_turn0", "session_id": "session_other",
            "turn_index": 0, "queue_index": 0,
            "prefill_instance_index": 0, "decode_instance_index": 1,
            "history_action": None, "history_source_instance_index": None,
            "history_transfer_bytes": 0, "history_recompute_tokens": 0,
            "history_tokens_before": 0,
            "prefill_context_tokens": 320, "prefill_length": 256,
            "decode_length": 4, "history_transfers": (),
            "history_evictions": (), "prefill_evictions": (),
        }
        with self.assertRaisesRegex(RuntimeError, "segment block end"):
            builder.emit_decode_eviction_actions(
                evictor_plan, (_whole_store(),),
                gated_on_prefill_segment_end=True)
        # 无门形态(阻塞 attempt)不依赖段块末,照常发射并登记。
        builder.emit_decode_eviction_actions(
            evictor_plan, (_whole_store(),),
            gated_on_prefill_segment_end=False)
        tails = {}
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        self.assertEqual({edge: len(ids) for edge, ids in tails.items()},
                         {1: 1, 2: 1})

    def test_terminal_retire_after_decision_time_emission_leaves_no_leak(self):
        """terminal-retire 场景回归(M-25):store 支链在决策时点(会话
        在世时)已物理发射并登记,其后退休边界清登记——不再存在"退休后
        物理发射 + 条目永久泄漏 → run 末 fail-closed 作废整跑"形态。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _emit_reserve_eviction_at_decision_time(
            builder, (_whole_store(),), gated=True)
        # store 支链已随 prefill 批物理发射(真实 NoC/远端池流量,与
        # 决策时点的账本核销同界,不再是退休后的死亡流量)。
        nodes = list(builder.batch["nodes"])
        self.assertTrue(_nodes_of(nodes, 1, "_remote_store"))
        self.assertTrue(_nodes_of(nodes, 5, "_send_to_edge"))
        self.assertIn(VICTIM, builder.pending_store_tails)
        # 退休边界(REQUEST_COMPLETE terminal):登记清除,run 末审计
        # (pending_store_tails 必空)可通过。
        builder.retire_completion_gate(VICTIM)
        self.assertNotIn(VICTIM, builder.pending_store_tails)

    def test_arm_fail_closed_when_decided_store_not_emitted(self):
        """fail-closed 探针:空表但调度器报告该会话仍有"已决策未发射"
        的 decode 侧 remote_store → 回迁发射拒绝(补边无载体,静默放行
        会复现池读先于池写);探针否定时维持懒处理 early-return。"""
        def _restore_builder(probe_hit):
            builder = GraphBatchBuilder(_make_config())
            builder.begin_batch()
            builder.set_pending_remote_store_probe(
                lambda session_id: probe_hit and session_id == VICTIM)
            builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
            builder.register_pending_history(
                request_id="req_restore_turn1", session_id=VICTIM,
                source_instance_index=1, location="remote_memory")
            return builder

        # 探针命中:回迁发射 fail-closed(独立 builder,中断发射的
        # 半程节点不跨 builder 混入)。
        with self.assertRaisesRegex(RuntimeError, "decided-but-unemitted"):
            _restore_builder(True).emit_prefill_batch(_restore_plan(
                prefill_instance=1, history_action="REMOTE_RESTORE",
                transfers=(_load_transfer(0, LAYERS, ((5, 1), (6, 2)),
                                          target_instance=1),),
                location="remote_memory"))
        # 探针否定(无待发射 store):空表 early-return,回迁照常。
        builder = _restore_builder(False)
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=1, history_action="REMOTE_RESTORE",
            transfers=(_load_transfer(0, LAYERS, ((5, 1), (6, 2)),
                                      target_instance=1),),
            location="remote_memory"))
        self.assertTrue(_nodes_of(
            list(builder.batch["nodes"]), 1, suffix="_remote_load"))


if __name__ == "__main__":
    unittest.main()
