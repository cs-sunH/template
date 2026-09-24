#!/usr/bin/env python3
"""test_store_restore_ordering.py -- store→restore 前递依赖排序钉子测试
(2026-09-13,KV 逐出与 request 推理并行化,主方案 §3.3)。

逐出支链悬空后,"同会话逐出池写先于其下一轮池读"的传递性保障失效,
回迁发射统一入口(非 partial 全量 remote_load + partial 流水 suffix
恢复)发射前查 pending_store_tails 补边。断言:

  1. 全量回迁(同缘):restore 边缘 rank 的回迁链首节点直接依赖该
     会话全部在飞 store 的 mem_store 节点——两段式逐出的 suffix/full
     两笔 store 均被依赖(restore 读全区间);
  2. 全量回迁(跨缘):store 边缘发 1B 中继(data_dep 挂 mem_store)、
     回迁边缘收 1B,其后的池读(mem_load)链在中继之后(桥拒绝跨
     rank 直边,1B p2p 是协议内唯一载体);
  3. PARTIAL 回迁:补边落在 partial fork 快照之后(挂在 suffix 恢复
     支链首节点,不泄回主链);
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


def _store_transfer(layer_start, layer_end, shards):
    return KVTransfer(
        kind="remote_store", phase="history",
        reason="watermark_admission_suffix_half",
        session_id=VICTIM, trigger_request_id="req_evict",
        source_instance_index=1, target_instance_index=None,
        total_bytes=sum(s.bytes for s in shards), shards=tuple(shards),
        model_layers=LAYERS, layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=LAYERS if layer_start == 0 else LAYERS,
        resident_prefix_layers_after=layer_start,
    )


def _suffix_store():
    """两段式 stage-1:decode rank 5→edge 1、rank 6→edge 2,层域 [2,4)。"""
    return _store_transfer(2, LAYERS, (
        _shard(source_rank=5, target_rank=1, edge_rank=1, bytes=800,
               noc_path=(5, 1), layer_start=2),
        _shard(source_rank=6, target_rank=2, edge_rank=2, bytes=800,
               noc_path=(6, 2), layer_start=2),
    ))


def _full_store():
    """两段式 stage-2:剩余前缀 [0,2) 经同两端口外迁。"""
    return _store_transfer(0, 2, (
        _shard(source_rank=5, target_rank=1, edge_rank=1, bytes=700,
               noc_path=(5, 1), layer_start=0, layer_end=2),
        _shard(source_rank=6, target_rank=2, edge_rank=2, bytes=700,
               noc_path=(6, 2), layer_start=0, layer_end=2),
    ))


def _load_transfer(layer_start, layer_end, targets, target_instance=0):
    """remote_load:shards 按 (target_rank, edge_rank) 对给定,层域
    [layer_start, layer_end)。"""
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
    """1. 逐出后下一轮全量回迁(同缘):两段式两笔 store 均被直接依赖。"""

    def test_both_stage_stores_gate_the_restore_chain(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_suffix_store(), _full_store()))
        tails = dict()
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        # 每个边缘 2 笔 store(suffix + full)。
        self.assertEqual({edge: len(ids) for edge, ids in tails.items()},
                         {1: 2, 2: 2})
        # 回迁批次:V 的 turn-1 全量回迁到 decode 实例(同缘 edge 1/2)。
        builder.completion_gates[VICTIM] = (1, {5: None, 6: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=1, location="remote_memory")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=1, history_action=None,
            transfers=(_load_transfer(0, LAYERS, ((5, 1), (6, 2)),
                                      target_instance=1),),
            location="remote_memory"))
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        # edge 1 的回迁链首节点(1B request recv):直接依赖两笔 store
        # 的 mem_store 节点(arm 在发射前,两 id 同被首节点消费)。
        first_on_edge1 = min(
            (n for n in _nodes_of(nodes, 1)
             if "_remote_load" in n["name"] or "_request_from_rank" in n["name"]),
            key=lambda n: n["id"])
        parents = _parents_of(edges, 1, first_on_edge1["id"])
        self.assertEqual(parents & tails[1], tails[1],
                         "both stage stores must gate the restore chain")
        # mem_load 传递性晚于两笔 mem_store。
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
        _evict_session_via_train(builder, (_suffix_store(),))
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
            prefill_instance=0, history_action=None,
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


class PartialRestoreOrderingTest(unittest.TestCase):
    """3. 逐出后下一轮 PARTIAL 回迁:补边落在 partial fork 之内。"""

    def test_suffix_restore_branch_gated_by_store_tails(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_suffix_store(),))
        tails = dict()
        for edge, store in _store_tail_ids(builder):
            tails.setdefault(edge, set()).add(store)
        self.assertEqual(set(tails), {1, 2})
        # PARTIAL 回迁(REMOTE_LOAD 同实例口径):门源 = prefill 实例 0,
        # suffix load 层域 [2,4) 经 edge 0/1。
        builder.completion_gates[VICTIM] = (0, {0: None, 1: None})
        builder.register_pending_history(
            request_id="req_restore_turn1", session_id=VICTIM,
            source_instance_index=0, location="partial_hbm_remote")
        builder.begin_batch()
        builder.emit_prefill_batch(_restore_plan(
            prefill_instance=0, history_source=0,
            history_action="REMOTE_LOAD",
            transfers=(_load_transfer(2, LAYERS, ((0, 0), (1, 1))),),
            resident_layers=2, location="partial_hbm_remote"))
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        # edge 1(与 store 同缘):suffix 恢复支链上的 mem_load(或链首)
        # 直接依赖 mem_store。
        load1 = _nodes_of(nodes, 1, suffix="_remote_load")
        self.assertEqual(len(load1), 1)
        chain_first_on_1 = min(
            _nodes_of(nodes, 1, "_remote_load"),
            key=lambda n: n["id"])
        self.assertTrue(
            tails[1] <= _ancestors_of(edges, 1, load1[0]["id"]))
        self.assertTrue(
            tails[1]
            <= (_parents_of(edges, 1, chain_first_on_1["id"])
                | _ancestors_of(edges, 1, chain_first_on_1["id"])))
        # edge 0(跨缘):中继 recv 在 mem_load 之前;中继 send 挂
        # mem_store。且补边在 partial fork 内——主链首 chunk prefix 段
        # 节点不依赖中继/pool 读(prefix 不等 suffix 恢复,既有流水
        # 语义保持)。
        relay_recv = _nodes_of(nodes, 0, "_store_tail_relay_from_e2")
        self.assertEqual(len(relay_recv), 1)
        load0 = _nodes_of(nodes, 0, suffix="_remote_load")
        self.assertEqual(len(load0), 1)
        self.assertIn(relay_recv[0]["id"],
                      _ancestors_of(edges, 0, load0[0]["id"]))
        relay_send = _nodes_of(nodes, 2, "_store_tail_relay_e2")
        self.assertTrue(relay_send)
        for send in relay_send:
            self.assertTrue(
                tails[2] <= _parents_of(edges, 2, send["id"]))
        prefix_nodes = _nodes_of(nodes, 0, "first_chunk_prefix")
        self.assertTrue(prefix_nodes)
        relay_ids = {n["id"] for n in relay_recv}
        for node in prefix_nodes:
            self.assertEqual(
                _ancestors_of(edges, 0, node["id"]) & relay_ids, set(),
                "prefix segment must not wait for the relayed suffix tail")


class RetireClearsStoreTailsTest(unittest.TestCase):
    """4. retire_terminal_session 边界清登记。"""

    def test_retire_completion_gate_clears_tails(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        _evict_session_via_train(builder, (_suffix_store(),))
        self.assertIn(VICTIM, builder.pending_store_tails)
        builder.retire_completion_gate(VICTIM)
        self.assertNotIn(VICTIM, builder.pending_store_tails)


if __name__ == "__main__":
    unittest.main()
