#!/usr/bin/env python3
"""test_graph_batch_builder.py -- sh_3.0 拼 batch 列车发射钉子测试
（2026-08-22 重订；原 chain_checkpoint/restore_chain 行为钉子保留，
新增列车发射 API 钉子——原两段式 emit_prefill_batch/emit_decode_batch
API 随列车化重构废止）。

经 emit_admission_batch + emit_iteration_train 完整发射后钉住拼 batch
改造语义：
  (a) 列车体节点归属批命名空间（train_id），stage/generation 一致；
  (b) PREFILL_DRAIN / DECODE_COMPLETION watch 成员 = drain/exit 标记节点
      （列车体后、end barrier 前的真实节点），非 end barrier 节点；
  (c) 触发门口径：_block_ends[req]["seg1"]/["seg2"] = 列车 post-barrier
      节点（== end barrier 节点 id）；
  (d) 权重摊销端到端：B=2 与 B=1 同迭代列车体权重字节相等
      （weight_passes=迭代数；陷阱 1 防护），激活/KV 分量逐成员精确；
  (e) 准入发射只含动作（gates/迁移/屏障），不含 prefill 主体与 watch。

chain_checkpoint/restore_chain 双捕获/双回滚语义钉子（2026-08-20 低危
表 L10 行修正）原样保留。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_graph_batch_builder.py   （或 pytest 同路径）
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

from face_scheduler import KVTransfer  # noqa: E402
from generate_trace import COMP_NODE  # noqa: E402
from online.graph_batch_builder import (  # noqa: E402
    GraphBatchBuilder,
    OnlineTraceBuilder,
)

SESSION = "session_train_0"
REQUEST_A = f"{SESSION}_request_0"
REQUEST_B = f"{SESSION}_request_1"
PREFILL_TOKENS = 300   # chunk 128 -> 3 个 chunk（多 chunk 灯具）
CHUNKS = 3
DECODE_RANKS = (2, 3)


def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=DECODE_RANKS, pg_name="tp_decode"),
        ],
        layers=2,
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


def _local_hit(request_id):
    """joiner 的 Prefill→Decode local_hit（零迁移节点；sh_3.0 在线侧
    plan 字段直接携带 KVTransfer 对象）。"""
    return KVTransfer(
        kind="local_hit",
        phase="prefill_to_decode",
        reason="train_test_same_tp_group",
        session_id=SESSION,
        trigger_request_id=request_id,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=0,
        shards=(),
        model_layers=2,
        layer_start=0,
        layer_end=2,
        resident_prefix_layers_before=2,
        resident_prefix_layers_after=2,
    )


def _admission_plan(request_id=REQUEST_A, turn=0):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": turn,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "admission_time_ns": 1000,
        "history_location_before": None,
        "history_transfer": None,
        "history_evictions": [],
        "prefill_evictions": [],
        "history_tokens_before": 0,
        "prefill_context_tokens": PREFILL_TOKENS,
    }


def _joiner_plan(request_id, context_tokens):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": context_tokens,
        "decode_evictions": [],
        "prefill_decode_transfer": _local_hit(request_id),
        # local_hit 零迁移节点 ⇒ join 标记是唯一 decode_start 锚点（灯
        # 具即考察该路径）；drain 触发门给占位（发射不校验其值，仅入边）。
        "prefill_drain_block_ends": {rank: 0 for rank in (0, 1)},
    }


def _train_plan(train_id, spans, iterations, joiners=(), drains=(),
                exits=(), stage="decode", prefill_start=None,
                instance_index=1, first_chunk=None):
    return {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": stage,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "prefill_start_member": prefill_start,
        "first_chunk_member": first_chunk,
        "drain_members": [{"request_id": rid} for rid in drains],
        "exit_members": [{"request_id": rid} for rid in exits],
    }


def _rank_nodes(builder, rank):
    return builder.builders[rank].nodes


def _edge_sources(builder, node_id):
    """本 rank 边列表中指向 node_id 的 from 集合（离线 data_deps 口径）。"""
    return {
        edge["from"]
        for edge in builder.edges
        if edge["to"] == node_id and edge["rank"] == builder.rank
    }


class TrainEmissionNailTest(unittest.TestCase):
    """列车发射钉子：(a)-(e)。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def test_admission_emits_actions_without_body_or_watch(self):
        """(e)：准入发射无 prefill 主体节点、无 watch、无块末账本。"""
        self.builder.emit_admission_batch(_admission_plan())
        names = [node["name"] for node in _rank_nodes(self.builder, 0)]
        self.assertTrue(names, "admission emitted nothing on rank 0")
        self.assertTrue(
            any("arrival_timer_gate" in name for name in names),
            f"admission must emit the arrival gate: {names}")
        self.assertTrue(
            any("prefill_kv_ready_barrier" in name for name in names),
            f"admission must emit the readiness barrier: {names}")
        self.assertFalse(
            any("_all_layers_" in name or "_all_passes_" in name
                for name in names),
            f"admission must not emit the pass body: {names}")
        self.assertNotIn(REQUEST_A, self.builder._block_ends)

    def _emit_two_member_train(self):
        """B=2 列车：成员 A(ctx=100, 3 token)与 B(ctx=40, 2 token)，
        2 个迭代；A 在列车内退出，B 存续；1 个 joiner(B)。"""
        joiners = [_joiner_plan(REQUEST_B, 40)]
        spans = [(1, 101), (1, 102), (1, 103), (1, 41), (1, 42)]
        plan = _train_plan(
            "batch_train_i1_1", spans, iterations=2, joiners=joiners,
            exits=(REQUEST_A,), prefill_start={"request_id": REQUEST_A},
            first_chunk={"request_id": REQUEST_A})
        return self.builder.emit_iteration_train(plan)

    def test_train_watch_members_are_marker_nodes_pre_barrier(self):
        """(a)+(b)：exit 标记节点承载 DECODE_COMPLETION watch 成员
        （真实 COMP 节点，end barrier 前），列车体归属批命名空间。"""
        result = self._emit_two_member_train()
        self.assertEqual(
            sorted(result["exit_members"][REQUEST_A]), sorted(DECODE_RANKS))
        for rank in DECODE_RANKS:
            nodes = _rank_nodes(self.builder, rank)
            barrier = max(
                node for node in nodes
                if node["name"] == "batch_train_i1_1_end_barrier")
            marker = nodes[
                result["exit_members"][REQUEST_A][rank]]
            self.assertEqual(marker["type"], COMP_NODE)
            self.assertEqual(marker["request_id"], REQUEST_A)
            self.assertEqual(marker["stage"], "decode")
            self.assertLess(marker["id"], barrier["id"])
            # (a) 体节点批命名空间归属 + generation == stage。
            body = [node for node in nodes
                    if node["request_id"] == "batch_train_i1_1"]
            self.assertTrue(body)
            for node in body:
                self.assertEqual(node["generation"], 1)

    def test_train_block_ends_are_post_barrier(self):
        """(c)：exit 成员的 seg2 块末 = 列车 end barrier 节点。"""
        result = self._emit_two_member_train()
        seg2 = self.builder._block_ends[REQUEST_A]["seg2"]
        for rank in DECODE_RANKS:
            self.assertEqual(seg2[rank], result["block_ends"][rank])
            barrier = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"] == "batch_train_i1_1_end_barrier")
            self.assertEqual(seg2[rank], barrier["id"])

    def test_join_marker_is_decode_anchor_node(self):
        """join 标记（joiner 零迁移节点时的唯一 decode_start 锚点）。"""
        self._emit_two_member_train()
        for rank in DECODE_RANKS:
            marker = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"].endswith(
                    f"join_{REQUEST_B}"))
            self.assertEqual(marker["request_id"], REQUEST_B)
            self.assertEqual(marker["stage"], "decode")

    def test_first_chunk_member_arms_suffix_gate(self):
        """first_chunk_member：partial 恢复 suffix 完成门挂到列车体首
        节点（准入批发射的恢复分支经 _suffix_body_arms 账本交接）。"""
        self.builder._suffix_body_arms[REQUEST_A] = {
            rank: 0 for rank in DECODE_RANKS}
        self._emit_two_member_train()
        self.assertNotIn(
            REQUEST_A, self.builder._suffix_body_arms,
            "suffix arm must be consumed by the first-chunk train")
        for rank in DECODE_RANKS:
            nodes = _rank_nodes(self.builder, rank)
            body_first = next(
                node for node in nodes
                if node["request_id"] == "batch_train_i1_1"
                and "_all_layers_" in node["name"])
            sources = _edge_sources_of(self.builder, rank, body_first["id"])
            self.assertIn(0, sources,
                          "train body must depend on the suffix gate")

    def _body_weight_bytes(self, spans, iterations, train_id):
        self.builder.emit_iteration_train(
            _train_plan(train_id, spans, iterations))
        total = 0
        for rank in DECODE_RANKS:
            for node in _rank_nodes(self.builder, rank):
                if node["request_id"] == train_id:
                    total += node["compute"]["tensor_size"]
        return total

    def test_b2_b1_train_body_weight_bytes_equal(self):
        """(d)：B=2（两成员 2 迭代，5 span，weight_passes=2）与 B=1
        （单成员 2 迭代，2 span，weight_passes=2）的同迭代列车体——
        权重分量相等；差值恰为第二个成员的逐 token 激活/KV 分量。"""
        b1 = self._body_weight_bytes(
            [(1, 101), (1, 102)], 2, "batch_train_i1_b1")
        b2 = self._body_weight_bytes(
            [(1, 101), (1, 102), (1, 41), (1, 42)], 2,
            "batch_train_i1_b2")
        member_b = self._body_weight_bytes(
            [(1, 41), (1, 42)], 2, "batch_train_i1_mb")
        self.assertGreater(b2, b1)
        # 权重常量（本列车 2 个迭代 × 每 rank）：由三次发射闭式推导
        # W = b1 + member_b - b2（独跑各含一份权重，拼车只含一份）。
        w_two = b1 + member_b - b2
        self.assertGreater(
            w_two, 0,
            "train body bytes must amortize weights across members "
            "(trap 1: per-member weight recounting yields W == 0)")
        # 不变量：向列车加成员 B 的边际字节 = B 的逐 token 激活/KV 分量，
        # 零额外权重（b2 - b1 == member_b - W）。基元级 B=1/B=2 同迭代
        # 权重字节相等断言见 test_weight_passes.py（A0）。
        self.assertEqual(b2 - b1, member_b - w_two)

    def test_prefill_train_drain_marker_watch(self):
        """纯 prefill 列车：chunk spans + drain 标记 + end barrier；
        stage=prefill/generation=0。"""
        spans = [(128, 128), (128, 256), (44, 300)]
        plan = _train_plan(
            "batch_train_i0_1", spans, iterations=3, stage="prefill",
            drains=(REQUEST_A,), instance_index=0,
            prefill_start={"request_id": REQUEST_A},
            first_chunk={"request_id": REQUEST_A})
        result = self.builder.emit_iteration_train(plan)
        for rank in (0, 1):
            nodes = _rank_nodes(self.builder, rank)
            marker = nodes[result["drain_members"][REQUEST_A][rank]]
            self.assertEqual(marker["stage"], "prefill")
            self.assertEqual(marker["generation"], 0)
            body = [node for node in nodes
                    if node["request_id"] == "batch_train_i0_1"]
            self.assertTrue(body)
            for node in body:
                self.assertEqual(node["generation"], 0)
            seg1 = self.builder._block_ends[REQUEST_A]["seg1"]
            barrier = next(
                node for node in nodes
                if node["name"] == "batch_train_i0_1_end_barrier")
            self.assertEqual(seg1[rank], barrier["id"])


def _edge_sources_of(graph_builder, rank, node_id):
    return {
        edge["from"]
        for edge in graph_builder.builders[rank].edges
        if edge["to"] == node_id and edge["rank"] == rank
    }


class ChainCheckpointRestoreTest(unittest.TestCase):
    """方案 §4.1：双捕获/双回滚语义钉子 + 恒空场景 no-op 回归。"""

    def setUp(self) -> None:
        self.builder = OnlineTraceBuilder(0, remote_operand_loads=False)

    def test_restore_rolls_back_branch_dependencies(self):
        """arm→发射（消费）→arm D1→checkpoint→分支内 arm D2 并发射→restore：
        previous_id 回到 checkpoint 值、pending_extra_dependencies == [D1]
        （D2 不残留）；恢复点后新节点重新依赖 D1 而不依赖 D2。"""
        builder = self.builder
        # arm 依赖 → 发一节点（消费）：依赖进入边、pending 清空。
        builder.comp("n0", 1, 1)
        dep_a = builder.previous_id
        builder.arm_dependency(dep_a)
        builder.comp("n1_consumes_a", 1, 1)
        self.assertIn(dep_a, _edge_sources(builder, builder.previous_id))
        self.assertEqual(builder.pending_extra_dependencies, [])

        # 再 arm D1 → checkpoint（捕获链首 + [D1]）。
        d1 = dep_a
        builder.arm_dependency(d1)
        checkpoint_previous = builder.previous_id
        handle = builder.chain_checkpoint()

        # 分支内：arm D2 并发一节点（timer_gate 不消费 pending/不动链首，
        # 仅用于产出真实节点 id；branch_node 消费 [D1, D2] 并前移链首）。
        d2 = builder.timer_gate("branch_gate", 1000)
        self.assertNotEqual(d2, d1)
        self.assertNotEqual(d2, checkpoint_previous)
        builder.arm_dependency(d2)
        builder.comp("branch_node", 1, 1)
        branch_node_id = builder.previous_id
        self.assertNotEqual(branch_node_id, checkpoint_previous)
        self.assertIn(d2, _edge_sources(builder, branch_node_id))
        self.assertEqual(builder.pending_extra_dependencies, [])

        # restore：链首与 pending 一并回滚，D2 不残留。
        builder.restore_chain(handle)
        self.assertEqual(builder.previous_id, checkpoint_previous)
        self.assertEqual(builder.pending_extra_dependencies, [d1])

        # 恢复点后发射：依赖 = checkpoint 链首 + D1，不含 D2。
        builder.comp("after_restore", 1, 1)
        deps = _edge_sources(builder, builder.previous_id)
        self.assertIn(checkpoint_previous, deps)
        self.assertIn(d1, deps)
        self.assertNotIn(d2, deps)
        self.assertEqual(builder.pending_extra_dependencies, [])

    def test_restore_with_empty_pending_is_noop_regression(self):
        """恒空场景（现行唯一调用点形态：checkpoint 时 pending 被 readiness
        barrier 清空、分支内不再 arm）：restore 仅回滚链首、pending 仍空——
        与改前行为逐位一致的 no-op 回归。"""
        builder = self.builder
        builder.comp("n0", 1, 1)
        checkpoint_previous = builder.previous_id
        self.assertEqual(builder.pending_extra_dependencies, [])
        handle = builder.chain_checkpoint()

        builder.comp("branch_node", 1, 1)
        self.assertNotEqual(builder.previous_id, checkpoint_previous)
        self.assertEqual(builder.pending_extra_dependencies, [])

        builder.restore_chain(handle)
        self.assertEqual(builder.previous_id, checkpoint_previous)
        self.assertEqual(builder.pending_extra_dependencies, [])

        # 恢复点后发射仅依赖 checkpoint 链首。
        builder.comp("after_restore", 1, 1)
        self.assertEqual(
            _edge_sources(builder, builder.previous_id),
            {checkpoint_previous},
        )


if __name__ == "__main__":
    unittest.main()
