#!/usr/bin/env python3
"""test_store_restore_ordering.py -- store→restore 前递依赖排序测试
（2026-09-13，KV 逐出与 request 推理并行化，主方案 §3.3）。

逐出支链悬空后，"同会话 store 池写先于其 restore 池读"的主链传递性
保障失效；回迁发射前 _arm_pending_store_edges 查 pending_store_tails
补边。本测试合成"逐出后下一轮回迁"序列钉住排序：

  1. REMOTE 全量回迁 · 同缘：restore 边缘 mem_load 的（跨 rank 扩展）
     依赖闭包含指向 store 边缘 mem_store 的直接 data_dep；
  2. REMOTE 全量回迁 · 跨缘：依赖经 1B p2p 中继（store_sidelink
     send/recv 对，tag 配对）承载；
  3. 同会话两段式（suffix + full fallback）：两笔 store 均被依赖；
  4. PARTIAL 后缀恢复支链（同实例钉扎）：分支内回迁链同样带排序；
  5. 消费即清：回迁后 pending_store_tails 条目弹出；
  6. 中继节点命名避开 batch_train_ / first_token 唤醒路由锚点。

依赖闭包口径：同 rank parent_edges + comm_send→comm_recv 的
(src,dst,tag) 配对（跨 rank 依赖在桥协议内只经 p2p 承载）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_store_restore_ordering.py
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

from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402
from generate_trace import COMM_RECV_NODE, COMM_SEND_NODE  # noqa: E402
from generate_face_trace import PendingHistoryGate  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

SESSION_X = "session_x"
LAYERS = 2


def _make_config(npus_count, groups, edge_npus):
    return SimpleNamespace(
        npus_count=npus_count,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=ranks, pg_name=pg_name)
            for pg_name, ranks in groups],
        layers=LAYERS,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        remote_memory=SimpleNamespace(edge_npus=edge_npus),
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _remote_store(session_id, shard_specs, layer_start=0, layer_end=LAYERS,
                  resident_after=0):
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=edge, edge_rank=edge,
            bytes=bytes, noc_path=(), layer_start=layer_start,
            layer_end=layer_end)
        for source, edge, bytes in shard_specs)
    return KVTransfer(
        kind="remote_store",
        phase="admission",
        reason="admission_capacity_full_fallback",
        session_id=session_id,
        trigger_request_id="session_new_request_0",
        source_instance_index=0,
        target_instance_index=None,
        total_bytes=sum(spec[2] for spec in shard_specs),
        shards=shards,
        model_layers=LAYERS,
        layer_start=layer_start,
        layer_end=layer_end,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=resident_after,
    )


def _remote_load(session_id, shard_specs, layer_start=0, layer_end=LAYERS):
    shards = tuple(
        KVTransferShard(
            source_rank=edge, target_rank=target, edge_rank=edge,
            bytes=bytes, noc_path=(), layer_start=layer_start,
            layer_end=layer_end)
        for target, edge, bytes in shard_specs)
    return KVTransfer(
        kind="remote_load",
        phase="admission",
        reason="history_remote",
        session_id=session_id,
        trigger_request_id="session_x_request_1",
        source_instance_index=None,
        target_instance_index=0,
        total_bytes=sum(spec[2] for spec in shard_specs),
        shards=shards,
        model_layers=LAYERS,
        layer_start=layer_start,
        layer_end=layer_end,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=LAYERS,
    )


class _OrderingHarness:
    """共享发射/闭包工具。"""

    def __init__(self, config):
        self.builder = GraphBatchBuilder(config)
        self.builder.begin_batch()

    def seed_gate_nodes(self, ranks):
        """为 pending gate 提供真实节点 id（每 rank 一个 comp 节点）。"""
        gate_ids = {}
        for rank in ranks:
            self.builder.builders[rank].comp(
                f"gate_seed_rank{rank}", 1, 1)
            gate_ids[rank] = self.builder.builders[rank].previous_id
        return gate_ids

    def emit_eviction(self, store_transfer):
        """turn-0 准入携带 history 逐出（旁路支链 + 尾部登记）。"""
        plan = {
            "request_id": "session_new_request_0",
            "session_id": "session_new",
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 0,
            "admission_time_ns": 1000,
            "history_location_before": None,
            "history_transfer": None,
            "history_evictions": [store_transfer],
            "prefill_evictions": [],
            "history_tokens_before": 0,
            "prefill_context_tokens": 300,
        }
        self.builder.emit_admission_batch(plan)

    def emit_restore(self, load_transfer, *, location, resident_layers=0,
                     request_id="session_x_request_1"):
        """turn>0 准入携带 history_transfer 回迁（补边点）。"""
        gate_ids = self.seed_gate_nodes(
            self.builder.group_by_index[0].ranks)
        self.builder.pending_history[request_id] = PendingHistoryGate(
            source_instance_index=0,
            timer_gates=tuple(
                gate_ids[rank]
                for rank in self.builder.group_by_index[0].ranks),
            location=location,
        )
        plan = {
            "request_id": request_id,
            "session_id": load_transfer.session_id,
            "turn_index": 1,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 0,
            "admission_time_ns": None,
            "history_location_before": SimpleNamespace(
                location=location,
                instance_index=0 if location == "partial_hbm_remote"
                else None,
                resident_prefix_layers=resident_layers,
            ),
            "history_transfer": load_transfer,
            "history_evictions": [],
            "prefill_evictions": [],
            "history_tokens_before": 100,
            "prefill_context_tokens": 300,
        }
        self.builder.emit_admission_batch(plan)

    # ------------------------------------------------------------- 闭包 --

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _edges(self):
        return self.builder.batch["parent_edges"]

    def _parents_map(self):
        parents = {}
        for edge in self._edges():
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                (edge["rank"], edge["from"]))
        # comm 配对：recv 依赖 send（同 src/dst/tag）。
        sends = {}
        for node in self._nodes():
            if node["type"] == COMM_SEND_NODE:
                comm = node["comm"]
                sends[(comm["src"], comm["dst"], comm["tag"])] = (
                    node["rank"], node["id"])
        for node in self._nodes():
            if node["type"] == COMM_RECV_NODE:
                comm = node["comm"]
                send = sends.get((comm["src"], comm["dst"], comm["tag"]))
                if send is not None:
                    parents.setdefault(
                        (node["rank"], node["id"]), []).append(send)
        return parents

    def ancestors(self, rank, node_id):
        parents = self._parents_map()
        seen = {(rank, node_id)}
        frontier = [(rank, node_id)]
        while frontier:
            current = frontier.pop()
            for parent in parents.get(current, ()):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return seen

    def mem_load_nodes(self):
        return [
            node for node in self._nodes()
            if node["name"].endswith("_remote_load")]

    def store_tail_ids(self, session_id):
        return {
            (edge_rank, store_id)
            for edge_rank, store_id, _ack
            in self.builder.pending_store_tails.get(session_id, ())}


class SameEdgeOrderingTest(unittest.TestCase):
    """REMOTE 全量回迁 · 同缘：store mem_store 直接 arm 进回迁链。"""

    def test_restore_waits_for_same_edge_store(self):
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(2,)))
        harness.emit_eviction(_remote_store(SESSION_X, [(0, 2, 1000)]))
        tails = harness.builder.pending_store_tails[SESSION_X]
        self.assertEqual(len(tails), 1)
        self.assertEqual(tails[0][0], 2)      # edge rank
        self.assertIsNotNone(tails[0][1])     # edge mem_store node id
        self.assertIsNotNone(tails[0][2])     # source ack recv node id
        store_key = (tails[0][0], tails[0][1])
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 1000)]),
            location="remote_memory")
        loads = harness.mem_load_nodes()
        self.assertEqual(len(loads), 1)
        self.assertIn(
            store_key,
            harness.ancestors(loads[0]["rank"], loads[0]["id"]),
            "same-edge restore mem_load does not depend on the "
            "store mem_store")


class CrossEdgeRelayTest(unittest.TestCase):
    """REMOTE 全量回迁 · 跨缘：依赖经 1B p2p 中继承载。"""

    def _run(self):
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(1, 2)))
        harness.emit_eviction(_remote_store(SESSION_X, [(0, 1, 1000)]))
        tails = harness.builder.pending_store_tails[SESSION_X]
        self.assertEqual(len(tails), 1)
        store_key = (tails[0][0], tails[0][1])
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 1000)]),
            location="remote_memory")
        return harness, store_key

    def test_relay_carries_store_dependency(self):
        harness, store_key = self._run()
        loads = harness.mem_load_nodes()
        self.assertEqual(len(loads), 1)
        self.assertEqual(loads[0]["rank"], 2)
        self.assertIn(
            store_key,
            harness.ancestors(loads[0]["rank"], loads[0]["id"]),
            "cross-edge restore mem_load does not reach the store "
            "mem_store through the 1B relay")
        relay = [
            node for node in harness._nodes()
            if "store_sidelink" in node["name"]]
        self.assertEqual(len(relay), 2)
        for node in relay:
            self.assertNotIn("batch_train_", node["name"])
            self.assertNotIn("first_token", node["name"])

    def test_tails_consumed_after_restore(self):
        harness, _store_key = self._run()
        self.assertNotIn(
            SESSION_X, harness.builder.pending_store_tails,
            "pending_store_tails entry must be consumed by the restore")


class TwoStageStoreTest(unittest.TestCase):
    """同会话两段式（suffix + full fallback）：两笔 store 均被依赖。"""

    def test_restore_waits_for_both_stores(self):
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(1, 2)))
        # 两段式：suffix store（半层）+ full fallback store（前缀层）。
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 1, 500)], layer_start=1, layer_end=2,
            resident_after=1))
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 2, 500)], layer_start=0, layer_end=1))
        self.assertEqual(
            len(harness.builder.pending_store_tails[SESSION_X]), 2)
        store_keys = harness.store_tail_ids(SESSION_X)
        self.assertEqual(len(store_keys), 2)
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 1000)]),
            location="remote_memory")
        loads = harness.mem_load_nodes()
        self.assertEqual(len(loads), 1)
        closure = harness.ancestors(loads[0]["rank"], loads[0]["id"])
        for store_key in store_keys:
            self.assertIn(
                store_key, closure,
                f"two-stage store {store_key} missing from the restore "
                "dependency closure")


class PartialSuffixOrderingTest(unittest.TestCase):
    """PARTIAL 后缀恢复支链（同实例钉扎）：分支内回迁链同样带排序。"""

    def test_partial_restore_waits_for_store(self):
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(2,)))
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 2, 500)], layer_start=1, layer_end=2,
            resident_after=1))
        store_key = next(iter(harness.store_tail_ids(SESSION_X)))
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 500)], layer_start=1,
                         layer_end=2),
            location="partial_hbm_remote", resident_layers=1)
        loads = harness.mem_load_nodes()
        self.assertEqual(len(loads), 1)
        self.assertIn(
            store_key,
            harness.ancestors(loads[0]["rank"], loads[0]["id"]),
            "partial suffix restore does not depend on the store "
            "mem_store")
        self.assertNotIn(SESSION_X, harness.builder.pending_store_tails)


class NoHazardNoEdgeTest(unittest.TestCase):
    """无登记（无逐出在飞）时回迁链不引入任何补边节点。"""

    def test_no_store_no_sidelink(self):
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(1, 2)))
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 1000)]),
            location="remote_memory")
        self.assertFalse([
            node for node in harness._nodes()
            if "store_sidelink" in node["name"]])
        self.assertEqual(len(harness.mem_load_nodes()), 1)


if __name__ == "__main__":
    unittest.main()
