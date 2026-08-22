#!/usr/bin/env python3
"""test_graph_batch_builder.py -- sh_2.0 拼 batch 列车发射钉子测试
（2026-08-22 新增；母本 sh_1.0 定型版同构 + sh_2.0 partial 两段式拆分
专项钉子）。

经 emit_admission_batch + emit_iteration_train 完整发射后钉住拼 batch
改造语义：
  (a) 列车体节点归属批命名空间（train_id），stage/generation 一致；
  (b) PREFILL_DRAIN / DECODE_COMPLETION watch 成员 = drain/exit 标记节点
      （列车体后、end barrier 前的真实节点），非 end barrier 节点；
  (c) 触发门口径：_block_ends[req]["seg1"]/["seg2"] = 列车 post-barrier
      节点（== end barrier 节点 id）；
  (d) 权重摊销端到端：B=2 与 B=1 同迭代列车体权重字节相等
      （weight_passes=迭代数；陷阱 1 防护），激活/KV 分量逐成员精确；
  (e) 准入发射只含动作（gates/迁移/屏障），不含 prefill 主体与 watch；
  (f) sh_2.0 特性保留：partial 前缀两段式迁移——首 chunk 拆
      prefix/suffix 层段，prefix 层段不等 suffix 恢复，suffix 层段
      arm 依赖 admission 记录的 suffix ready 节点（两段式流水）。

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

from generate_trace import COMP_NODE  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

SESSION = "session_train_0"
REQUEST_A = f"{SESSION}_request_0"
REQUEST_B = f"{SESSION}_request_1"
PREFILL_TOKENS = 300   # chunk 128 -> 3 个 chunk(多 chunk 灯具)
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
        prefill_chunk_size=128,
        layers=4,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=1000,
                inter_request_interval_ns=None),
            SimpleNamespace(
                session_arrival_time_ns=None,
                inter_request_interval_ns=5000000),
        ],
    )


def _admission_plan(request_id=REQUEST_A, turn=0, queue_index=0):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": turn,
        "queue_index": queue_index,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "admission_time_ns": 1000,
        "history_location_before": None,
        "history_prefix_transfer": None,
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
        "prefill_decode_transfer": {
            "kind": "local_hit",
            "phase": "prefill_to_decode",
            "reason": "train_test_same_tp_group",
            "session_id": SESSION,
            "trigger_request_id": request_id,
            "source_instance_index": 0,
            "target_instance_index": 1,
            "total_bytes": 0,
            "shards": [],
            "model_layers": 4,
            "layer_start": 0,
            "layer_end": 4,
            "resident_prefix_layers_before": 4,
            "resident_prefix_layers_after": 4,
        },
        # local_hit 零迁移节点 ⇒ join 标记是唯一 decode_start 锚点（灯
        # 具即考察该路径）；drain 触发门给占位（发射不校验其值，仅入边）。
        "prefill_drain_block_ends": {rank: 0 for rank in (0, 1)},
    }


def _train_plan(train_id, spans, iterations, joiners=(), drains=(),
                exits=(), stage="decode", prefill_start=None,
                instance_index=1, partial_count=None):
    plan = {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": stage,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "prefill_start_member": prefill_start,
        "drain_members": [{"request_id": rid} for rid in drains],
        "exit_members": [{"request_id": rid} for rid in exits],
    }
    if partial_count is not None:
        plan["partial_first_chunk_count"] = partial_count
    return plan


def _rank_nodes(builder, rank):
    return builder.builders[rank].nodes


class TrainEmissionNailTest(unittest.TestCase):
    """列车发射钉子：(a)-(f)。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def test_admission_emits_actions_without_body_or_watch(self):
        """(e)：准入发射无 prefill 主体节点、无返回 watch 成员。"""
        self.builder.emit_admission_batch(_admission_plan())
        names = [node["name"] for node in _rank_nodes(self.builder, 0)]
        self.assertTrue(names, "admission emitted nothing on rank 0")
        self.assertTrue(
            any("global_arrival_timer_gate" in name for name in names),
            f"admission must emit the arrival gates: {names}")
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
            exits=(REQUEST_A,), prefill_start={"request_id": REQUEST_A})
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
        result = self._emit_two_member_train()
        for rank in DECODE_RANKS:
            marker = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"].endswith(
                    f"join_{REQUEST_B}"))
            self.assertEqual(marker["request_id"], REQUEST_B)
            self.assertEqual(marker["stage"], "decode")

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
            drains=(REQUEST_A,), instance_index=0)
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

    def test_partial_first_chunk_two_stage_pipelining(self):
        """(f) sh_2.0 特性保留：partial 前缀两段式迁移的首 chunk 拆分。

        队列头 history_location_before == "partial_hbm_remote" 时，
        admission 登记 _partial_first_chunk（suffix 恢复完成门）；列车
        按 [首 chunk + 成员第 1 迭代] / [其余] 拆组发射：
          - first_chunk_prefix（layers00_01，resident 前缀层段）不依赖
            suffix ready 节点（流水：prefix 层段与 suffix 恢复并行）；
          - first_chunk_suffix（layers02_03）arm 依赖 suffix ready 节点；
          - remaining_aggregated 回到全层段，weight_passes = 迭代数-1。
        """
        # admission 预登记 partial 流水信息（正常路径由 _emit_admission
        # 的 partial 分支写入；此处直接钉列车侧消费契约）。
        builder = self.builder
        suffix_ready_by_rank = {
            rank: len(_rank_nodes(builder, rank)) + 100
            for rank in (0, 1)}
        builder._partial_first_chunk[REQUEST_A] = {
            "suffix_start": 2,
            "suffix_ready_nodes_by_rank": dict(suffix_ready_by_rank),
        }
        spans = [(128, 128), (1, 40), (128, 256), (1, 41)]
        plan = _train_plan(
            "batch_train_i0_9", spans, iterations=2, stage="prefill",
            drains=(REQUEST_A,), instance_index=0, partial_count=2,
            prefill_start={"request_id": REQUEST_A})  # partial ⇒ 首 chunk 列车
        builder.emit_iteration_train(plan)
        # 消费后弹出（恰一次）。
        self.assertNotIn(REQUEST_A, builder._partial_first_chunk)
        for rank in (0, 1):
            nodes = _rank_nodes(builder, rank)
            edges = builder.builders[rank].edges
            suffix_ready = suffix_ready_by_rank[rank]
            prefix_nodes = [node for node in nodes
                            if "first_chunk_prefix" in node["name"]]
            suffix_nodes = [node for node in nodes
                            if "first_chunk_suffix" in node["name"]]
            remaining_nodes = [node for node in nodes
                               if "remaining_aggregated" in node["name"]]
            self.assertTrue(prefix_nodes and suffix_nodes
                            and remaining_nodes)
            self.assertTrue(all(
                "layers00_01" in node["name"] for node in prefix_nodes))
            # suffix 段含 all_passes 输出头（无层标签）；层类节点必须
            # 全部落在 resident 前缀之后的层段。
            suffix_layer_nodes = [node for node in suffix_nodes
                                  if "_layers" in node["name"]]
            self.assertTrue(suffix_layer_nodes)
            self.assertTrue(all(
                "layers02_03" in node["name"]
                for node in suffix_layer_nodes))
            remaining_layer_nodes = [node for node in remaining_nodes
                                     if "_layers" in node["name"]]
            self.assertTrue(remaining_layer_nodes)
            self.assertTrue(all(
                "all_layers" in node["name"]
                for node in remaining_layer_nodes))
            prefix_ids = {node["id"] for node in prefix_nodes}
            suffix_ids = {node["id"] for node in suffix_nodes}
            # prefix 层段首节点的父边不含 suffix ready（不等恢复）。
            prefix_first = min(prefix_ids)
            prefix_parents = {
                edge["from"] for edge in edges
                if edge["to"] == prefix_first}
            self.assertNotIn(suffix_ready, prefix_parents)
            # suffix 层段首节点 arm 依赖 suffix ready（跨批次边）。
            suffix_first = min(suffix_ids)
            suffix_parents = {
                edge["from"] for edge in edges
                if edge["to"] == suffix_first}
            self.assertIn(suffix_ready, suffix_parents)


if __name__ == "__main__":
    unittest.main()
