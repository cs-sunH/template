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
  4. retire_terminal_session 边界(retire_completion_gate)清登记。

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
        # 登记表保持(retire 前不清;懒处理)。
        self.assertIn(VICTIM, builder.pending_store_tails)


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
        # 中继 tag 走 TransferTagAllocator(10^7 基址段)。
        for send in relay_send:
            self.assertGreaterEqual(send["comm"]["tag"], 10_000_000)


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


if __name__ == "__main__":
    unittest.main()
