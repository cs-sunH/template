#!/usr/bin/env python3
"""test_store_restore_ordering.py -- store→restore 前递依赖排序测试
（2026-09-13，KV 逐出与 request 推理并行化，主方案 §3.3；2026-09-17
§4.2.4 前递补边硬化）。

逐出支链悬空后，"同会话 store 池写先于其 restore 池读"的主链传递性
保障失效；回迁发射前 _arm_pending_store_edges 查 pending_store_tails
补边。本测试合成"逐出后下一轮回迁"序列钉住排序：

  1. REMOTE 全量回迁 · 同缘：restore 边缘 mem_load 的（跨 rank 扩展）
     依赖闭包含指向 store 边缘 mem_store 的直接 data_dep；
  2. REMOTE 全量回迁 · 跨缘：依赖经 1B p2p 中继（store_sidelink
     send/recv 对，tag 配对）承载；
  3. 同会话两段式（suffix + full fallback）：两笔 store 均被依赖；
  4. PARTIAL 后缀恢复支链（同实例钉扎）：分支内回迁链同样带排序；
  5. 消费即清：回迁后 pending_store_tails 匹配条目弹出；
  6. 中继节点命名避开 batch_train_ / first_token 唤醒路由锚点。

§4.2.4 硬化（2026-09-17，条目扩层区间 + 交集选择性消费 + fail-closed）：

  7. 交集选择性：条目 = (edge_rank, store_node, ack_node, ls, le)；
     restore 区间只消费有交集条目，未匹配保留（后续消费者可再消费）；
  8. fail-closed：restore 区间与全部条目无交集 → raise（含无条目态）；
  9. 交错竞态（kimi）：整体外迁 [0,L) 写尾在途登记 + 立即回迁 [p,L)
     → 交集消费成功（写尾未落盘即回迁）。

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


def _make_config(npus_count, groups, edge_npus, layers=LAYERS):
    return SimpleNamespace(
        npus_count=npus_count,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=ranks, pg_name=pg_name)
            for pg_name, ranks in groups],
        layers=layers,
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
                  resident_after=0, model_layers=LAYERS):
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
        model_layers=model_layers,
        layer_start=layer_start,
        layer_end=layer_end,
        resident_prefix_layers_before=model_layers,
        resident_prefix_layers_after=resident_after,
    )


def _remote_load(session_id, shard_specs, layer_start=0, layer_end=LAYERS,
                 model_layers=LAYERS):
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
        model_layers=model_layers,
        layer_start=layer_start,
        layer_end=layer_end,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=model_layers,
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
            for edge_rank, store_id, _ack, _ls, _le
            in self.builder.pending_store_tails.get(session_id, ())}

    def store_tail_ranges(self, session_id):
        return sorted(
            (layer_start, layer_end)
            for _edge_rank, _store_id, _ack, layer_start, layer_end
            in self.builder.pending_store_tails.get(session_id, ()))


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
        # §4.2.4：条目携带层区间（transfer.layer_start/layer_end）。
        self.assertEqual(tails[0][3], 0)
        self.assertEqual(tails[0][4], LAYERS)
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


class NoRegistrationFailClosedTest(unittest.TestCase):
    """§4.2.4 fail-closed：无任何条目（本会话此前无池写登记却要回迁
    读池，账目不一致）→ raise（原 fail-open 静默返回已退役）。"""

    def test_no_store_restore_raises(self):
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(1, 2)))
        with self.assertRaises(RuntimeError) as ctx:
            harness.emit_restore(
                _remote_load(SESSION_X, [(0, 2, 1000)]),
                location="remote_memory")
        message = str(ctx.exception)
        self.assertIn(SESSION_X, message)
        self.assertIn("[0, 2)", message)      # restore 区间入消息
        self.assertIn("none", message)        # 现有条目区间 = 无


class DisjointRangeFailClosedTest(unittest.TestCase):
    """§4.2.4 fail-closed：有条目但与 restore 区间全不交集 → raise
    （消息含 session_id / restore 区间 / 现有条目区间）。"""

    def test_disjoint_restore_raises(self):
        # L=32：登记 [16, 32)（深层后缀逐出），restore [4, 8)（合成
        # 中段区间——钉选择公式 max(ls)<min(le)，非现行后缀形消费者）。
        layers = 32
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(2,), layers=layers))
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 2, 1000)], layer_start=16, layer_end=layers,
            resident_after=16, model_layers=layers))
        self.assertEqual(
            harness.store_tail_ranges(SESSION_X), [(16, layers)])
        with self.assertRaises(RuntimeError) as ctx:
            harness.emit_restore(
                _remote_load(SESSION_X, [(0, 2, 400)], layer_start=4,
                             layer_end=8, model_layers=layers),
                location="remote_memory")
        message = str(ctx.exception)
        self.assertIn(SESSION_X, message)
        self.assertIn("[4, 8)", message)      # restore 区间
        self.assertIn("[16, 32)", message)    # 现有条目区间


class IntersectiveSelectiveConsumptionTest(unittest.TestCase):
    """§4.2.4 交集选择性消费：restore 只消费有交集条目，未匹配条目
    保留在 pending_store_tails 供后续消费者（PARTIAL remote-read 后缀
    恢复 [p, L)）。"""

    def test_disjoint_tail_retained_then_consumed(self):
        # L=16：登记 [8, 16) 与 [4, 16) 两条目（两段后缀逐出）；restore
        # [4, 8) 只交集 [4, 16)（[8,16) ∩ [4,8) = ∅ → 保留）；再一次
        # restore [8, 16) 消费保留条目（"再一次 restore 可消费"）。
        layers = 16
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(2,), layers=layers))
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 2, 500)], layer_start=8, layer_end=layers,
            resident_after=8, model_layers=layers))
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 2, 500)], layer_start=4, layer_end=layers,
            resident_after=4, model_layers=layers))
        self.assertEqual(
            harness.store_tail_ranges(SESSION_X), [(4, layers), (8, layers)])
        # 第一次回迁 [4, 8)：交集选择性消费。
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 400)], layer_start=4,
                         layer_end=8, model_layers=layers),
            location="remote_memory")
        self.assertEqual(len(harness.mem_load_nodes()), 1)
        self.assertEqual(
            harness.store_tail_ranges(SESSION_X), [(8, layers)],
            "disjoint [8, L) tail must survive the [4, 8) restore")
        # 第二次回迁 [8, 16)：保留条目可被后续消费者消费。
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 500)], layer_start=8,
                         layer_end=layers, model_layers=layers),
            location="remote_memory", request_id="session_x_request_2")
        self.assertEqual(len(harness.mem_load_nodes()), 2)
        self.assertNotIn(
            SESSION_X, harness.builder.pending_store_tails,
            "retained tail must be consumed by the matching restore")


class InterleavedFullEvictionImmediateRestoreTest(unittest.TestCase):
    """§4.2.4 交错竞态（kimi）：整体外迁 [0, L) 写尾在途登记 + 立即
    回迁 [p, L) → 交集消费成功（覆盖"写尾未落盘即回迁"竞态）。"""

    def test_immediate_partial_restore_after_full_eviction(self):
        layers = 16
        prefix = 4
        harness = _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(1, 2)))
        # 整体外迁 [0, L)：store 尾登记在途（store 物理未完成）。
        harness.emit_eviction(_remote_store(
            SESSION_X, [(0, 1, 1000)], layer_start=0, layer_end=layers,
            resident_after=0, model_layers=layers))
        store_key = next(iter(harness.store_tail_ids(SESSION_X)))
        # 立即回迁 [p, L)（跨缘：store 边缘 1、restore 边缘 2）：交集
        # [p, L) 非空 → 消费成功，回迁链经 1B 中继排序于在途 store 后。
        harness.emit_restore(
            _remote_load(SESSION_X, [(0, 2, 750)], layer_start=prefix,
                         layer_end=layers, model_layers=layers),
            location="remote_memory")
        loads = harness.mem_load_nodes()
        self.assertEqual(len(loads), 1)
        self.assertIn(
            store_key,
            harness.ancestors(loads[0]["rank"], loads[0]["id"]),
            "immediate suffix restore does not wait for the in-flight "
            "full-eviction store tail")
        self.assertNotIn(
            SESSION_X, harness.builder.pending_store_tails)


class StoreForwardingHelperTests(unittest.TestCase):
    """T8（P2-R2 助手级单测，PARTIAL 跨实例 copy 流水化 2026-09-25）：
    _arm_pending_store_edges 的三段分解——_match_store_entries（纯查询
    + 历史文案 fail-closed）/ _emit_store_relays_once（每 (store_edge,
    restore_edge) 对恰一次发射 + relay_cache 去重）/ _consume_store_
    entries（恰移除 matched、其余保留）。公开 _arm_pending_store_edges
    的消费/arm/中级行为不变由上面各直调用例钉住。"""

    def _harness(self):
        return _OrderingHarness(_make_config(
            npus_count=4,
            groups=[("tp_prefill", (0,)), ("tp_decode", (3,))],
            edge_npus=(1, 2)))

    def test_match_is_pure_query_with_intersection_semantics(self):
        builder = self._harness().builder
        builder.pending_store_tails[SESSION_X] = [
            (1, 11, 11, 0, 8),
            (2, 22, 22, 8, 16),
            (2, 33, 33, 20, 32),
        ]
        matched, retained = builder._match_store_entries(SESSION_X, 4, 12)
        self.assertEqual(
            [(entry[0], entry[1]) for entry in matched],
            [(1, 11), (2, 22)],
            "entries intersecting [4, 12) match; [20, 32) does not")
        self.assertEqual(
            [(entry[0], entry[1]) for entry in retained], [(2, 33)])
        # 纯查询：账本不被改动（消费由 _consume_store_entries 单独承担）。
        self.assertEqual(len(builder.pending_store_tails[SESSION_X]), 3)

    def test_match_no_overlap_raises_with_historical_message(self):
        builder = self._harness().builder
        builder.pending_store_tails[SESSION_X] = [(2, 33, 33, 20, 32)]
        with self.assertRaises(RuntimeError) as ctx:
            builder._match_store_entries(SESSION_X, 8, 16)
        message = str(ctx.exception)
        self.assertIn(SESSION_X, message)
        self.assertIn("[8, 16)", message)      # restore 区间
        self.assertIn("[20, 32)", message)     # 现有条目区间

    def test_consume_removes_exactly_matched_entries(self):
        builder = self._harness().builder
        entries = [(1, 11, 11, 0, 8), (2, 22, 22, 8, 16), (2, 33, 33, 20, 32)]
        builder.pending_store_tails[SESSION_X] = list(entries)
        builder._consume_store_entries(SESSION_X, [entries[0], entries[2]])
        self.assertEqual(
            builder.pending_store_tails[SESSION_X], [entries[1]],
            "only the matched entries are removed; retained survive")
        builder._consume_store_entries(SESSION_X, [entries[1]])
        self.assertNotIn(
            SESSION_X, builder.pending_store_tails,
            "emptying the matched set pops the session key")

    def test_relays_emit_once_per_pair_and_cache_dedups(self):
        builder = self._harness().builder
        # 同一跨缘对 (1,2) 的两条 matched 条目共享一次中继发射。
        matched = [(1, 11, 11, 0, 8), (1, 12, 12, 8, 16)]
        relay_cache = {}
        first = builder._emit_store_relays_once(
            matched, [2], "relay_probe", relay_cache=relay_cache)

        def relay_nodes(b):
            return [
                node for rank_builder in b.builders.values()
                for node in rank_builder.nodes
                if "store_sidelink" in node["name"]]

        sends = [
            node for node in relay_nodes(builder)
            if node["type"] == COMM_SEND_NODE]
        recvs = [
            node for node in relay_nodes(builder)
            if node["type"] == COMM_RECV_NODE]
        self.assertEqual(len(sends), 1)
        self.assertEqual(len(recvs), 1)
        self.assertEqual(
            sends[0]["comm"]["tag"], recvs[0]["comm"]["tag"],
            "relay send/recv are tag-paired")
        self.assertEqual(sends[0]["comm"]["bytes"], 1)
        self.assertIn((1, 2), relay_cache)
        self.assertEqual(first, {2: relay_cache[(1, 2)]})
        # 同批同对第二次调用命中缓存：零新节点（评审 #2 中继膨胀回归
        # 钉——恢复组支链的中继提升到组循环之前，组数无关）。
        nodes_after_first = len(relay_nodes(builder))
        second = builder._emit_store_relays_once(
            matched, [2], "relay_probe", relay_cache=relay_cache)
        self.assertEqual(second, first)
        self.assertEqual(len(relay_nodes(builder)), nodes_after_first)
        # 无缓存调用面（公开 _arm_pending_store_edges 路径）不去重——
        # 每调用恰一次发射（对现有三消费者语义 = 历史行为）。
        other = self._harness().builder
        uncached = other._emit_store_relays_once(matched, [2], "relay_probe")
        self.assertEqual(len(uncached), 1)
        self.assertEqual(len(relay_nodes(other)), 2)


if __name__ == "__main__":
    unittest.main()
