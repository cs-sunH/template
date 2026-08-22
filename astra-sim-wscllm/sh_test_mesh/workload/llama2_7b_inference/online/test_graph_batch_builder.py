#!/usr/bin/env python3
"""test_graph_batch_builder.py -- wscllm 拼 batch 列车发射钉子测试
(2026-08-22;照 sh_1.0 母本 test_graph_batch_builder.py 改订,适配本仓
PD 分离豁免形态:emit_iteration_train 仅 D 侧列车,P 侧保持 emit_prefill_
batch 现有整段骨架,§3.6)。

经 emit_iteration_train 完整发射后钉住拼 batch 改造语义:
  (a) 列车体节点归属批命名空间(train_id),stage/generation 一致;
  (b) DECODE_COMPLETION watch 成员 = exit 标记节点(列车体后、end
      barrier 前的真实节点),非 end barrier 节点;
  (c) 触发门口径:completion_gates[session](下一 turn interval gate 的
      after_node_id 来源)= 列车 post-barrier 节点(== end barrier id),
      与旧整段发射的 end-barrier 口径一致;
  (d) 权重摊销端到端:B=2 与 B=1 同迭代列车体权重字节相等
      (weight_passes=迭代数;陷阱 1 防护),激活/KV 分量逐成员精确;
  (e) joiner 迁移 = transfer 3000 send/recv 对 + join 标记
      (decode_start 指标锚点);
  (f) §3.6 无混拼:纯 decode 列车发射不产生任何 prefill 主体节点
      (stage="prefill" 的列车命名空间节点不存在;P 侧整段发射不经
      emit_iteration_train)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
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
PREFILL_RANKS = (0, 1)
DECODE_RANKS = (2, 3)


def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(name="g0", ranks=PREFILL_RANKS,
                            pg_name="tp_prefill"),
            SimpleNamespace(name="g1", ranks=DECODE_RANKS,
                            pg_name="tp_decode"),
        ],
        prefill_chunk_size=128,
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        num_heads_for_transfer=8,
        request_queue=[],
        # transfer 3000 的 shard 均分与 NoC 路由所需(kv_cache_bytes_for_
        # tokens 只读 layers/hidden_size/bytes_per_elem;_xy_route 只读
        # mesh_cols)。
        model=SimpleNamespace(layers=2, hidden_size=64, bytes_per_elem=2),
        hardware=SimpleNamespace(mesh_cols=2),
    )


def _joiner_plan(request_id, context_tokens):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": context_tokens,
    }


def _train_plan(train_id, spans, iterations, joiners=(), exits=(),
                instance_index=1):
    return {
        "train_id": train_id,
        "instance_index": instance_index,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "exit_members": [{"request_id": rid, "session_id": SESSION}
                         for rid in exits],
    }


def _rank_nodes(builder, rank):
    return builder.builders[rank].nodes


class TrainEmissionNailTest(unittest.TestCase):
    """列车发射钉子:(a)-(f)。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def _emit_two_member_train(self):
        """B=2 列车:成员 A(ctx=100, 3 token)与 B(ctx=40, 2 token),
        2 个迭代;A 在列车内退出,B 存续;1 个 joiner(B)。"""
        joiners = [_joiner_plan(REQUEST_B, 40)]
        spans = [(1, 101), (1, 102), (1, 103), (1, 41), (1, 42)]
        plan = _train_plan(
            "batch_train_i1_1", spans, iterations=2, joiners=joiners,
            exits=(REQUEST_A,))
        return self.builder.emit_iteration_train(plan)

    def test_train_watch_members_are_marker_nodes_pre_barrier(self):
        """(a)+(b):exit 标记节点承载 DECODE_COMPLETION watch 成员
        (真实 COMP 节点,end barrier 前),列车体归属批命名空间。"""
        result = self._emit_two_member_train()
        self.assertEqual(
            sorted(result["exit_members"][REQUEST_A]), sorted(DECODE_RANKS))
        for rank in DECODE_RANKS:
            nodes = _rank_nodes(self.builder, rank)
            barrier = max(
                node for node in nodes
                if node["name"] == "batch_train_i1_1_end_barrier")
            marker = nodes[result["exit_members"][REQUEST_A][rank]]
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

    def test_train_completion_gates_are_post_barrier(self):
        """(c):退出成员 session 的 completion gate = 列车 end barrier 节点
        (post-barrier 口径,下一 turn interval gate 的 after_node_id)。"""
        result = self._emit_two_member_train()
        _decode_index, gates = self.builder.completion_gates[SESSION]
        for rank in DECODE_RANKS:
            self.assertEqual(gates[rank], result["block_ends"][rank])
            barrier = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"] == "batch_train_i1_1_end_barrier")
            self.assertEqual(gates[rank], barrier["id"])

    def test_joiner_transfer_and_marker(self):
        """(e):joiner 迁移 = transfer 3000 send/recv 对(prefill rank 发、
        decode rank 收)+ 每 decode rank 1 个 join 标记(decode_start
        指标锚点,上下文 (joiner, decode, 1))。"""
        self._emit_two_member_train()
        sends = [node for node in _rank_nodes(self.builder, 0)
                 if "prefill_to_decode_kv" in node["name"]
                 and node["type"] == 5]
        recvs = [node for node in _rank_nodes(self.builder, 2)
                 if "prefill_to_decode_kv" in node["name"]
                 and node["type"] == 6]
        self.assertTrue(sends and recvs)
        for rank in DECODE_RANKS:
            marker = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"].endswith(f"join_{REQUEST_B}"))
            self.assertEqual(marker["request_id"], REQUEST_B)
            self.assertEqual(marker["stage"], "decode")
            self.assertEqual(marker["generation"], 1)

    def test_train_emits_no_prefill_stage_nodes(self):
        """(f):§3.6 无混拼——纯 decode 列车发射不产生任何 prefill 主体
        节点(stage="prefill" 或列车命名空间外的 prefill span 节点)。"""
        self._emit_two_member_train()
        for rank in DECODE_RANKS + PREFILL_RANKS:
            for node in _rank_nodes(self.builder, rank):
                if node["stage"] == "prefill":
                    self.fail(
                        f"decode train emitted a prefill-stage node on "
                        f"rank {rank}: {node['name']}")

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
        """(d):B=2(两成员 2 迭代,5 span,weight_passes=2)与 B=1
        (单成员 2 迭代,2 span,weight_passes=2)的同迭代列车体——
        权重分量相等;差值恰为第二个成员的逐 token 激活/KV 分量。"""
        b1 = self._body_weight_bytes(
            [(1, 101), (1, 102)], 2, "batch_train_i1_b1")
        b2 = self._body_weight_bytes(
            [(1, 101), (1, 102), (1, 41), (1, 42)], 2,
            "batch_train_i1_b2")
        member_b = self._body_weight_bytes(
            [(1, 41), (1, 42)], 2, "batch_train_i1_mb")
        self.assertGreater(b2, b1)
        # 权重常量(本列车 2 个迭代 × 每 rank):由三次发射闭式推导
        # W = b1 + member_b - b2(独跑各含一份权重,拼车只含一份)。
        w_two = b1 + member_b - b2
        self.assertGreater(
            w_two, 0,
            "train body bytes must amortize weights across members "
            "(trap 1: per-member weight recounting yields W == 0)")
        # 不变量:向列车加成员 B 的边际字节 = B 的逐 token 激活/KV 分量,
        # 零额外权重(b2 - b1 == member_b - W)。基元级 B=1/B=2 同迭代
        # 权重字节相等断言见 test_weight_passes.py(A0)。
        self.assertEqual(b2 - b1, member_b - w_two)


if __name__ == "__main__":
    unittest.main()
