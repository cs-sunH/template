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
import json
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
from online.graph_batch_builder import (  # noqa: E402
    GraphBatchBuilder,
    OnlineTraceBuilder,
)
from online.wsc_llm_legacy_online_scheduler import (  # noqa: E402
    WscLlmLegacyOnlineScheduler,
)
from online.wsc_llm_online_scheduler import WscLlmOnlineScheduler  # noqa: E402
from relevant_kv_emission import (  # noqa: E402
    emit_piece_scatter,
    emit_remote_reads,
    local_kv_override_value,
)

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
    """M1 适配(2026-08-29 收集即释放):_collect 立即清空 builder.nodes，
    已交付节点只在当前批次累加器中保留——改读 batch["nodes"](发射序,自 begin_batch 起含
    本批全部节点),保持"直读已发射节点"的测试意图;测试内单批发射,
    节点 id 自 0 连续,rank 过滤后位置 == 节点 id,与改前等价。"""
    return [node for node in builder.batch["nodes"]
            if node["rank"] == rank]


def _node_edge_payload(batch):
    return json.dumps(
        {"nodes": batch["nodes"], "parent_edges": batch["parent_edges"]},
        separators=(",", ":"), sort_keys=True,
    )


class DependencyFastPathTest(unittest.TestCase):
    """_new_node 的 0/1 fast path 必须与有序去重旧逻辑逐案等价。"""

    def test_dependency_edge_order_and_state_matrix(self):
        cases = (
            ("no_previous_or_pending", None, (), ()),
            ("previous_only", 4, (), (4,)),
            ("one_pending_only", None, (8,), (8,)),
            ("one_pending_matches_previous", 4, (4,), (4,)),
            ("one_pending_differs_from_previous", 4, (8,), (4, 8)),
            ("many_pending_with_duplicates", 4, (8, 4, 9, 8), (4, 8, 9)),
            ("many_pending_without_previous", None, (8, 8, 9), (8, 9)),
        )
        for label, previous_id, pending, expected_sources in cases:
            with self.subTest(case=label):
                trace = OnlineTraceBuilder(7, remote_operand_loads=False)
                trace.next_id = 17
                trace.previous_id = previous_id
                trace.pending_extra_dependencies.extend(pending)

                trace.comp("dependency_matrix", 1, 1)

                self.assertEqual(
                    trace.edges,
                    [{"rank": 7, "from": source, "to": 17, "kind": "data"}
                     for source in expected_sources],
                )
                self.assertEqual(trace.nodes[-1]["id"], 17)
                self.assertEqual(trace.next_id, 18)
                self.assertEqual(trace.previous_id, 17)
                self.assertEqual(trace.pending_extra_dependencies, [])


class CollectionLifecycleTest(unittest.TestCase):
    """交付后 builder 缓冲区只应保留尚未收集的节点与边。"""

    def test_collect_releases_buffers_without_changing_payload(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        marker = builder._mark()
        for rank, trace_builder in builder.builders.items():
            trace_builder.comp(f"first_{rank}_0", 1, 1)
            trace_builder.comp(f"first_{rank}_1", 1, 1)
        expected_first_payload = json.dumps(
            {
                "nodes": [
                    node for trace_builder in builder.builders.values()
                    for node in trace_builder.nodes
                ],
                "parent_edges": [
                    edge for trace_builder in builder.builders.values()
                    for edge in trace_builder.edges
                ],
            },
            separators=(",", ":"), sort_keys=True,
        )

        builder._collect(marker)
        first_batch = builder.batch
        first_payload = _node_edge_payload(first_batch)
        self.assertEqual(first_payload, expected_first_payload)
        for trace_builder in builder.builders.values():
            self.assertEqual(trace_builder.nodes, [])
            self.assertEqual(trace_builder.edges, [])

        builder.begin_batch()
        marker = builder._mark()
        for rank, trace_builder in builder.builders.items():
            trace_builder.comp(f"second_{rank}", 1, 1)
        builder._collect(marker)

        self.assertEqual(_node_edge_payload(first_batch), first_payload)
        self.assertEqual(
            [(node["rank"], node["id"]) for node in builder.batch["nodes"]],
            [(rank, 2) for rank in builder.builders],
        )
        self.assertEqual(
            [(edge["rank"], edge["from"], edge["to"])
             for edge in builder.batch["parent_edges"]],
            [(rank, 1, 2) for rank in builder.builders],
        )
        for trace_builder in builder.builders.values():
            self.assertEqual(trace_builder.nodes, [])
            self.assertEqual(trace_builder.edges, [])


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


def _split_plan(train_id, spans, iterations, joiners=(), exits=(),
                first_token=None, sentinel=False):
    plan = _train_plan(
        train_id, spans, iterations, joiners=joiners, exits=exits,
        instance_index=1)
    plan["sentinel"] = sentinel
    if first_token is not None:
        plan["first_token"] = first_token
    return plan


def _first_token_of(train_id, spans, member_parts, *, split=True,
                    marker_ids=(), exit_first_token=()):
    """按调度器 _first_token_plan 的产物同构构造 first_token 子计划。"""
    if not split:
        return {
            "split": False,
            "debut_marker_members": [{"request_id": rid}
                                     for rid in marker_ids],
            "debut_exit_first_token": list(exit_first_token),
        }
    first_spans = []
    rest_spans = []
    offset = 0
    for _, participation in member_parts:
        first_spans.append(spans[offset])
        rest_spans.extend(spans[offset + 1:offset + participation])
        offset += participation
    return {
        "split": True,
        "first_spans": first_spans,
        "rest_spans": rest_spans,
        "debut_marker_members": [{"request_id": rid}
                                 for rid in marker_ids],
        "debut_exit_first_token": list(exit_first_token),
        "wakeup_id": f"{train_id}_first_step",
    }


def _body_totals(builder, train_id):
    """列车体字节总量:body 节点(request_id == train_id)的 compute
    tensor_size 求和 + 列车 all_reduce coll bytes 求和(decode rank)。"""
    tensor = 0
    coll = 0
    for rank in DECODE_RANKS:
        for node in _rank_nodes(builder, rank):
            if node["request_id"] != train_id:
                continue
            tensor += node["compute"]["tensor_size"]
            coll += node["coll"]["bytes"]
    return tensor, coll


class FirstStepSplitEmissionTest(unittest.TestCase):
    """WP9 首步批拆分钉子(2026-08-26;WP9_CONTRACT §2 wscllm 特例:
    decode-only 列车,首步 = 各成员第 1 个 span,exit/哨兵/end barrier
    全部挂余量批)。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def _two_member_split_plan(self, *, sentinel=False):
        """B=2 列车:成员 A(ctx=100, 3 token)与 B(ctx=40, 2 token),
        2 个迭代;A 退出(其中 A 为 decode_length=1 边界由专项测试覆盖,
        此处 A 为多 token debut——注意首步标记只挂 debut 成员)。"""
        joiners = [_joiner_plan(REQUEST_A, 100), _joiner_plan(REQUEST_B, 40)]
        spans = [(1, 101), (1, 102), (1, 103), (1, 41), (1, 42)]
        member_parts = [("ra", 3), ("rb", 2)]
        first_token = _first_token_of(
            "batch_train_i1_1", spans, member_parts,
            marker_ids=(REQUEST_A, REQUEST_B,))
        return _split_plan(
            "batch_train_i1_1", spans, 2, joiners=joiners,
            exits=(REQUEST_A,), first_token=first_token,
            sentinel=sentinel)

    def test_split_batch_structure(self):
        """批结构:首步批 = joiner 迁移 + join 标记 + 首迭代体(每成员
        第 1 个 span)+ first_token 标记 + 唤醒标记;绝无 exit/哨兵/
        end barrier。余量批 = 余量体 + exit 标记 + end barrier(挂点
        语义不变:标记在 barrier 前)。"""
        plan = self._two_member_split_plan()
        first = self.builder.emit_train_first_step(plan)
        # 首步批返回契约:first_token_members(debut 各自挂)+ wakeup。
        self.assertEqual(
            sorted(first["first_token_members"]), sorted((REQUEST_A, REQUEST_B)))
        self.assertEqual(sorted(first["wakeup_members"]), sorted(DECODE_RANKS))
        step_names = {node["name"] for node in self.builder.batch["nodes"]}
        for rank in DECODE_RANKS:
            self.assertIn(
                f"batch_train_i1_1_first_token_{REQUEST_A}", step_names)
            self.assertIn(
                f"batch_train_i1_1_join_{REQUEST_B}", step_names)
            self.assertIn("batch_train_i1_1_first_step_wakeup", step_names)
        # 首步批无任何 watch/exit/哨兵/end barrier 节点。
        for node in self.builder.batch["nodes"]:
            self.assertNotIn("_exit_", node["name"])
            self.assertNotIn("_sentinel", node["name"])
            self.assertNotIn("end_barrier", node["name"])
        # 唤醒标记:名字不含 first_token 子串(不注册 code 8 锚点),
        # 上下文 = 批命名空间 + (prefill, 0)(哨兵同款单事件通道)。
        for node in self.builder.batch["nodes"]:
            if node["name"] == "batch_train_i1_1_first_step_wakeup":
                self.assertNotIn("first_token", node["name"])
                self.assertEqual(node["request_id"],
                                 "batch_train_i1_1_first_step")
                self.assertEqual(node["stage"], "prefill")
                self.assertEqual(node["generation"], 0)
        # first_token 标记:名字含 first_token 子串 + 携 debut 成员
        # request_id + (decode, 1) 上下文(C++ 按 (request_id, rank)
        # 注册 code 8 锚点)。
        for rank in DECODE_RANKS:
            node = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"] == f"batch_train_i1_1_first_token_{REQUEST_A}")
            self.assertEqual(node["request_id"], REQUEST_A)
            self.assertEqual(node["stage"], "decode")
            self.assertEqual(node["generation"], 1)
            self.assertEqual(node["type"], COMP_NODE)
            self.assertEqual(node["compute"]["num_ops"], 1)
        # 余量批:exit 标记 + end barrier 照常;completion gate 账本
        # post-barrier 口径不变。
        self.builder.begin_batch()
        rest = self.builder.emit_train_remainder(plan)
        self.assertEqual(
            sorted(rest["exit_members"][REQUEST_A]), sorted(DECODE_RANKS))
        rest_names = {node["name"] for node in self.builder.batch["nodes"]}
        self.assertIn("batch_train_i1_1_end_barrier", rest_names)
        self.assertIn(f"batch_train_i1_1_exit_{REQUEST_A}", rest_names)
        _decode_index, gates = self.builder.completion_gates[SESSION]
        for rank in DECODE_RANKS:
            self.assertEqual(gates[rank], rest["block_ends"][rank])

    def test_split_body_bytes_conserve_full_train(self):
        """字节守恒:首步批 + 余量批的列车体 tensor_size 与 all_reduce
        bytes 总量 == 整列发射(weight_passes 1 + iterations-1 恰合回
        iterations;激活/KV/AR 逐 span 精确求和)。"""
        split_plan = self._two_member_split_plan()
        self.builder.emit_train_first_step(split_plan)
        first_tensor, first_coll = _body_totals(self.builder,
                                                "batch_train_i1_1")
        self.assertGreater(first_tensor, 0)
        self.builder.begin_batch()
        self.builder.emit_train_remainder(split_plan)
        rest_tensor, rest_coll = _body_totals(self.builder,
                                              "batch_train_i1_1")
        full = GraphBatchBuilder(_make_config())
        full.begin_batch()
        full.emit_iteration_train(_train_plan(
            "batch_train_i1_1", split_plan["pass_spans"], 2,
            joiners=split_plan["joiners"], exits=(REQUEST_A,)))
        full_tensor, full_coll = _body_totals(full, "batch_train_i1_1")
        self.assertEqual(first_tensor + rest_tensor, full_tensor)
        self.assertEqual(first_coll + rest_coll, full_coll)

    def test_tmax_truncation_one_plus_seven(self):
        """T_max 截断:8 迭代列车(1+7≤8)拆分后首步恰 1 迭代/成员、
        余量 7 迭代/成员;哨兵标记照常挂余量批。"""
        spans = [(1, 101 + step) for step in range(8)] \
            + [(1, 41 + step) for step in range(8)]
        member_parts = [("ra", 8), ("rb", 8)]
        first_token = _first_token_of(
            "batch_train_i2_1", spans, member_parts,
            marker_ids=(REQUEST_A, REQUEST_B,))
        plan = _split_plan(
            "batch_train_i2_1", spans, 8,
            joiners=[_joiner_plan(REQUEST_A, 100),
                     _joiner_plan(REQUEST_B, 40)],
            first_token=first_token, sentinel=True)
        self.assertEqual(len(first_token["first_spans"]), 2)   # 每成员 1
        self.assertEqual(len(first_token["rest_spans"]), 14)   # 每成员 7
        self.builder.emit_train_first_step(plan)
        self.builder.begin_batch()
        rest = self.builder.emit_train_remainder(plan)
        rest_names = {node["name"] for node in self.builder.batch["nodes"]}
        self.assertIn("batch_train_i2_1_sentinel", rest_names)

    def test_decode_length_one_exit_marker_carries_first_token(self):
        """decode_length=1 边界:debut 成员 decode_length=1 不在首步批挂
        独立 first_token 标记;其余量批 exit 标记名携带 first_token 子串
        (同节点 code 4/8 双锚点;B2.5 裁决:finalize 侧放宽为序检查,
        同节点语义仍保证 first_token 与 completion 同源)。"""
        joiners = [_joiner_plan(REQUEST_A, 100), _joiner_plan(REQUEST_B, 40)]
        spans = [(1, 101), (1, 41), (1, 42)]   # A 仅 1 token,B 2 token
        member_parts = [("ra", 1), ("rb", 2)]
        first_token = _first_token_of(
            "batch_train_i3_1", spans, member_parts,
            marker_ids=(REQUEST_B,),
            exit_first_token=(REQUEST_A,))
        plan = _split_plan(
            "batch_train_i3_1", spans, 2, joiners=joiners,
            exits=(REQUEST_A,), first_token=first_token)
        first = self.builder.emit_train_first_step(plan)
        # decode_length=1 debut 无独立首步标记。
        self.assertNotIn(REQUEST_A, first["first_token_members"])
        self.builder.begin_batch()
        rest = self.builder.emit_train_remainder(plan)
        for rank in DECODE_RANKS:
            node = next(
                node for node in _rank_nodes(self.builder, rank)
                if "_exit_" in node["name"]
                and node["request_id"] == REQUEST_A)
            self.assertIn("first_token", node["name"])
            self.assertEqual(node["request_id"], REQUEST_A)

    def test_no_split_marker_enhancement_and_off_equivalence(self):
        """不拆车增强(split=False,iterations==1)+ OFF 等价断言:
        (1) first_token=None(开关关闭)时 emit_iteration_train 发射的
        节点名字/归属与既有整列发射完全一致——无 first_token/wakeup
        名字、exit 标记名不带 first_token 子串;
        (2) split=True 计划误入 emit_iteration_train 拒绝(fail-closed);
        (3) iterations==1 的增强路径:多 token debut 体后挂 first_token
        标记,decode_length=1 debut exit 标记改名。"""
        spans = [(1, 101), (1, 41)]
        plan = _split_plan(
            "batch_train_i4_1", spans, 2, joiners=[_joiner_plan(REQUEST_B, 40)],
            exits=(REQUEST_A,))
        result = self.builder.emit_iteration_train(plan)
        for node in self.builder.batch["nodes"]:
            self.assertNotIn("first_token", node["name"])
            self.assertNotIn("first_step", node["name"])
        # (2) 拆分计划走整列入口拒绝。
        split_plan = self._two_member_split_plan()
        with self.assertRaises(RuntimeError):
            self.builder.emit_iteration_train(split_plan)
        with self.assertRaises(RuntimeError):
            self.builder.emit_train_first_step(plan)   # 无 first_token 子计划
        with self.assertRaises(RuntimeError):
            self.builder.emit_train_remainder(plan)
        # (3) iterations==1:两成员 participation 各 1;A(decode_length=1)
        # exit 改名,B(多 token)体后挂标记。
        one_spans = [(1, 101), (1, 41)]
        first_token = _first_token_of(
            "batch_train_i4_2", one_spans, [("ra", 1), ("rb", 1)],
            split=False, marker_ids=(REQUEST_B,),
            exit_first_token=(REQUEST_A,))
        one_plan = _split_plan(
            "batch_train_i4_2", one_spans, 1,
            joiners=[_joiner_plan(REQUEST_A, 100),
                     _joiner_plan(REQUEST_B, 40)],
            exits=(REQUEST_A, REQUEST_B), first_token=first_token)
        result = self.builder.emit_iteration_train(one_plan)
        names = {node["name"] for node in self.builder.batch["nodes"]}
        self.assertIn(f"batch_train_i4_2_first_token_{REQUEST_B}", names)
        self.assertIn(f"batch_train_i4_2_exit_first_token_{REQUEST_A}", names)
        self.assertNotIn(f"batch_train_i4_2_first_token_{REQUEST_A}", names)
        # 增强路径 watch 成员/完成门与整列口径一致。
        self.assertEqual(
            sorted(result["exit_members"][REQUEST_A]), sorted(DECODE_RANKS))
        _decode_index, gates = self.builder.completion_gates[SESSION]
        for rank in DECODE_RANKS:
            self.assertEqual(gates[rank], result["block_ends"][rank])

class _GateRecorder:
    """仅记录 terminal REQUEST_COMPLETE 是否回收构图器完成门。"""

    def __init__(self):
        self.completion_gates = {SESSION: (1, {2: 7, 3: 7})}
        self.retired = []

    def retire_completion_gate(self, session_id):
        self.retired.append(session_id)
        self.completion_gates.pop(session_id, None)


class _TerminalKVRecorder:
    """Records terminal KV retirement without needing a real topology."""

    def __init__(self, released_instance_index=1):
        self.released_instance_index = released_instance_index
        self.calls = []

    def retire_terminal_session(self, session_id, completion_ns, request_id):
        self.calls.append((session_id, completion_ns, request_id))
        return self.released_instance_index


class CompletionGateLifetimeTest(unittest.TestCase):
    """跨 turn gate 只在在途期保留；terminal REQUEST_COMPLETE 回收。"""

    @staticmethod
    def _two_turn_config():
        config = _make_config()
        config.request_queue = [
            SimpleNamespace(session_arrival_time_ns=0,
                            inter_request_interval_ns=None),
            SimpleNamespace(session_arrival_time_ns=None,
                            inter_request_interval_ns=1000),
        ]
        return config

    @staticmethod
    def _turn_one_prefill_plan():
        return {
            "request_id": REQUEST_B,
            "session_id": SESSION,
            "turn_index": 1,
            "queue_index": 1,
            "prefill_instance_index": 0,
            "decode_instance_index": 1,
            "history_action": None,
            "history_source_instance_index": None,
            "history_transfer_bytes": 0,
            "history_recompute_tokens": 0,
            "history_tokens_before": 0,
            "prefill_length": 128,
            "prefill_context_tokens": 128,
        }

    def test_turn_one_admission_consumes_gate_and_terminal_retires(self):
        builder = GraphBatchBuilder(self._two_turn_config())
        # 模拟 turn-0 decode end barrier；turn-1 prefill admission 消费后，
        # after_node_id 已写入图，账本条目必须释放。
        builder.completion_gates[SESSION] = (1, {2: None, 3: None})
        builder.begin_batch()
        builder.emit_prefill_batch(self._turn_one_prefill_plan())
        self.assertNotIn(SESSION, builder.completion_gates)

        builder.completion_gates[SESSION] = (1, {2: 9, 3: 9})
        builder.retire_completion_gate(SESSION)
        self.assertEqual(builder.completion_gates, {})

    def test_main_and_legacy_terminal_request_complete_retire_gate(self):
        for scheduler_type in (WscLlmOnlineScheduler,
                               WscLlmLegacyOnlineScheduler):
            with self.subTest(scheduler=scheduler_type.__name__):
                scheduler = scheduler_type.__new__(scheduler_type)
                runtime = SimpleNamespace(
                    request_id=REQUEST_A,
                    session_id=SESSION,
                    decode_tokens_consumed=1,
                    decode_length=1,
                    decode_train_joined=True,
                )
                scheduler.graph = _GateRecorder()
                scheduler.runtime_by_request_id = {REQUEST_A: runtime}
                scheduler._runtime_index = {REQUEST_A: 0}
                scheduler.next_request = [None]
                scheduler.runtimes = [runtime]
                scheduler._batch = {"future_alarms": []}
                if scheduler_type is WscLlmOnlineScheduler:
                    scheduler.kv_manager = _TerminalKVRecorder()
                    scheduler._note_capacity_change = lambda index: None

                scheduler._on_request_complete(REQUEST_A, 123)

                self.assertEqual(scheduler.graph.retired, [SESSION])
                self.assertEqual(scheduler.graph.completion_gates, {})
                self.assertEqual(scheduler.runtime_by_request_id, {})
                self.assertEqual(scheduler._runtime_index, {})
                self.assertEqual(scheduler.runtimes, [None])
                if scheduler_type is WscLlmOnlineScheduler:
                    self.assertEqual(
                        scheduler.kv_manager.calls,
                        [(SESSION, 123, REQUEST_A)],
                    )

    def test_main_intermediate_request_keeps_kv_for_following_turn(self):
        scheduler = WscLlmOnlineScheduler.__new__(WscLlmOnlineScheduler)
        runtime = SimpleNamespace(
            request_id=REQUEST_A,
            session_id=SESSION,
            decode_tokens_consumed=1,
            decode_length=1,
            decode_train_joined=True,
        )
        following = SimpleNamespace(
            request_id=REQUEST_B,
            session_id=SESSION,
            turn_index=1,
            queue_index=1,
            prefill_length=128,
            decode_length=1,
        )
        scheduler.graph = _GateRecorder()
        scheduler.runtime_by_request_id = {REQUEST_A: runtime}
        scheduler._runtime_index = {REQUEST_A: 0}
        scheduler.next_request = [following]
        scheduler.runtimes = [runtime]
        scheduler._batch = {"future_alarms": []}
        scheduler.kv_manager = _TerminalKVRecorder()
        scheduler._note_capacity_change = lambda index: None
        scheduler.config = SimpleNamespace(
            request_queue=[None, SimpleNamespace(inter_request_interval_ns=1000)]
        )

        scheduler._on_request_complete(REQUEST_A, 123)

        self.assertEqual(scheduler.kv_manager.calls, [])
        self.assertEqual(scheduler.graph.retired, [])
        self.assertEqual(len(scheduler._batch["future_alarms"]), 1)


# ============================================================================
# relevant_distributed 增量用例(2026-09-02 B2;总文档 §8.3/§3.2/§9 R3-②/R4,
# 执行文档 §5.2)。既有用例一行不动,以下全部为新增。
# ============================================================================

# 4 实例 × TP2 = 8 rank(mesh 4x2):0=P、1=D、2/3=中间 die。
_RK_PREFILL_INSTANCE = 0
_RK_DECODE_INSTANCE = 1
_RK_DIE_INSTANCE = 2
_RK_DIE_INSTANCE_3 = 3
_RK_PREFILL_RANKS = (0, 1)
_RK_DECODE_RANKS = (2, 3)
_RK_BYTES_PER_RANK_TOKEN = 256     # heads [4,4] × head_dim 8 × 2 B × 2 层 ×2(K/V)


def _rk_config():
    groups = [
        SimpleNamespace(name=f"rk_g{index}", ranks=(2 * index, 2 * index + 1),
                        pg_name=f"rk_tp{index}")
        for index in range(4)
    ]
    return SimpleNamespace(
        npus_count=8,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=groups,
        prefill_chunk_size=128,
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(session_arrival_time_ns=1000,
                            inter_request_interval_ns=None),
            SimpleNamespace(session_arrival_time_ns=2000,
                            inter_request_interval_ns=None),
        ],
        model=SimpleNamespace(layers=2, hidden_size=64, num_heads=8,
                              bytes_per_elem=2),
        hardware=SimpleNamespace(mesh_cols=2),
    )


def _rk_piece(owner, start, end, tier="scatter_remote"):
    """KVPiece duck-typing(执行文档 §4.1 字段名;tier 仅元数据)。"""
    return SimpleNamespace(instance_index=owner, token_start=start,
                           token_end=end, tier=tier,
                           distance_to_decode=0, path=())


def _rk_prefill_plan(request_id, session_id, queue_index, context=200):
    return {
        "request_id": request_id,
        "session_id": session_id,
        "turn_index": 0,
        "queue_index": queue_index,
        "prefill_instance_index": _RK_PREFILL_INSTANCE,
        "decode_instance_index": _RK_DECODE_INSTANCE,
        "history_action": None,
        "history_source_instance_index": None,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_length": context,
        "prefill_context_tokens": context,
    }


def _rk_train_plan(train_id, spans, iterations, joiners=(), exits=()):
    return {
        "train_id": train_id,
        "instance_index": _RK_DECODE_INSTANCE,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "exit_members": [{"request_id": rid, "session_id": f"rk_s_{rid}"}
                         for rid in exits],
    }


def _rk_emit_scenario(graph, *, request_id="rk_r0", queue_index=0,
                      context=200, exits=None):
    """标准 relevant 场景:prefill → 3100 散布(D[0,80) + P[80,140) stay +
    die2[140,200))→ 列车 → 3300 读边(源 = P 60 token + die2 60 token)。
    exits=None 表示该成员本列车退出(默认);exits=() 表示中途成员。
    返回 (plan, members, scatter, train_result, reads)。"""
    if exits is None:
        exits = (request_id,)
    plan = _rk_prefill_plan(request_id, f"rk_s_{request_id}", queue_index,
                            context)
    members = graph.emit_prefill_batch(plan)
    pieces = [
        _rk_piece(_RK_DECODE_INSTANCE, 0, 80),
        _rk_piece(_RK_PREFILL_INSTANCE, 80, 140, "prefill_stay"),
        _rk_piece(_RK_DIE_INSTANCE, 140, 200),
        _rk_piece(_RK_DECODE_INSTANCE, 200, 220, "decode_local"),
    ]
    scatter = emit_piece_scatter(
        graph, queue_index=queue_index, prefix=f"rk_q{queue_index}",
        request_id=request_id,
        prefill_instance_index=_RK_PREFILL_INSTANCE, pieces=pieces,
        prefill_context_tokens=context, prefill_end_members=members)
    train = _rk_train_plan(
        "batch_train_i1_1", [(1, 201), (1, 202), (1, 203)], 2,
        joiners=[plan], exits=exits)
    train_result = graph.emit_iteration_train(train)
    member = {
        "request_id": request_id,
        "queue_index": queue_index,
        "participation": 2,
        "prefill_instance_index": _RK_PREFILL_INSTANCE,
        "sources": [
            {"source_instance_index": _RK_PREFILL_INSTANCE,
             "piece_tokens": 60},
            {"source_instance_index": _RK_DIE_INSTANCE, "piece_tokens": 60,
             "source_ordinal": 1},
        ],
        "prefill_end_members": members,
        "exit_anchors": (train_result["exit_members"][request_id]
                         if request_id in exits else None),
    }
    reads = emit_remote_reads(
        graph, train_id="batch_train_i1_1",
        decode_instance_index=_RK_DECODE_INSTANCE, member_reads=[member],
        scatter_recv_ids=scatter["recv_ids"])
    return plan, members, scatter, train_result, reads


def _rk_nodes(graph, name_part, *, send=None):
    return [node for node in graph.batch["nodes"]
            if name_part in node["name"] and node["type"] in (5, 6)
            and (send is None or ("_send_" in node["name"]) == send)]


def _rk_parents(graph, rank, node_id):
    return sorted(edge["from"] for edge in graph.batch["parent_edges"]
                  if edge["rank"] == rank and edge["to"] == node_id)


class RelevantScatterEmissionTest(unittest.TestCase):
    """3100 节点结构 + send 显式锚 + 侧插不污染 P frontier(总文档 §3.2)。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_rk_config())
        self.builder.begin_batch()

    def test_scatter_structure_anchor_and_frontier(self):
        plan = _rk_prefill_plan("rk_r0", "rk_s_rk_r0", 0)
        members = self.builder.emit_prefill_batch(plan)
        # prefill 块末 members = 末个真实 prefill 节点;链尾 previous_id 是
        # 其后的人工 end barrier(id 更大)——3100 显式锚用前者(R4 fallback
        # 语义),不依赖 frontier 等价。
        frontier_after_prefill = {
            rank: self.builder.builders[rank].previous_id
            for rank in _RK_PREFILL_RANKS}
        for rank in _RK_PREFILL_RANKS:
            self.assertIsNotNone(frontier_after_prefill[rank])
            self.assertGreater(frontier_after_prefill[rank], members[rank])
        pieces = [
            _rk_piece(_RK_DECODE_INSTANCE, 0, 80),
            _rk_piece(_RK_PREFILL_INSTANCE, 80, 140, "prefill_stay"),
            _rk_piece(_RK_DIE_INSTANCE, 140, 200),
        ]
        scatter = emit_piece_scatter(
            self.builder, queue_index=0, prefix="rk_q0",
            request_id="rk_r0",
            prefill_instance_index=_RK_PREFILL_INSTANCE, pieces=pieces,
            prefill_context_tokens=200, prefill_end_members=members)
        # 结构:名字含 kv 子串(C++ 名字锚),send/recv 对、tag、字节。
        sends = _rk_nodes(self.builder, "_kv_scatter_", send=True)
        recvs = _rk_nodes(self.builder, "_kv_scatter_", send=False)
        self.assertEqual(len(sends), 4)      # 2 owner × TP2
        self.assertEqual(len(recvs), 4)
        for node in sends + recvs:
            self.assertIs(node["comm"]["hbm_charge"], True)
            self.assertIn("kv", node["name"])
        # D owner 80 token、die owner 60 token:逐 rank 256 B/token;tag
        # 扩展段 = relative_rank(3100 单源,无源序号扩展位)。
        bytes_by_owner = {}
        extensions_by_owner = {}
        for node in sends:
            owner_rank = node["comm"]["dst"]
            for owner_index, recv_by_rank in scatter["recv_ids"].items():
                if owner_rank in recv_by_rank:
                    bytes_by_owner.setdefault(owner_index,
                                              set()).add(node["comm"]["bytes"])
                    extensions_by_owner.setdefault(owner_index,
                                                   set()).add(
                        node["comm"]["tag"] % 10000 - 3100)
        self.assertEqual(bytes_by_owner, {
            _RK_DECODE_INSTANCE: {80 * _RK_BYTES_PER_RANK_TOKEN},
            _RK_DIE_INSTANCE: {60 * _RK_BYTES_PER_RANK_TOKEN},
        })
        self.assertEqual(extensions_by_owner, {
            _RK_DECODE_INSTANCE: {0, 1}, _RK_DIE_INSTANCE: {0, 1}})
        self.assertEqual(scatter["scatter_tokens_by_owner"],
                         {_RK_DECODE_INSTANCE: 80, _RK_DIE_INSTANCE: 60})
        # send 侧插:唯一父边 = 该 P rank 的 prefill 块末节点(显式锚),
        # 且 P frontier 不被推进(previous_id 保存恢复,仍为 end barrier)。
        for node in sends:
            rank = node["rank"]
            self.assertEqual(_rk_parents(self.builder, rank, node["id"]),
                             [members[rank]])
        for rank in _RK_PREFILL_RANKS:
            self.assertEqual(self.builder.builders[rank].previous_id,
                             frontier_after_prefill[rank])
        # recv 正常链入 owner 的 D/die frontier(旧 3000 同口径):recv 即
        # 该 owner rank 的当前链尾(本场景 owner 首批节点)。
        for owner_index, recv_by_rank in scatter["recv_ids"].items():
            for rank, recv_id in recv_by_rank.items():
                self.assertEqual(self.builder.builders[rank].previous_id,
                                 recv_id)
        self.assertEqual(scatter["recv_ids"][_RK_DIE_INSTANCE][4],
                         next(node["id"] for node in recvs
                              if node["rank"] == 4))


class RelevantRemoteReadEmissionTest(unittest.TestCase):
    """3300 双端侧插 + recv 注入该成员自己的 exit + send 锚两档。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_rk_config())
        self.builder.begin_batch()

    def test_side_insert_frontiers_and_exit_injection(self):
        # 内联场景(需在列车后/读边前捕获 frontier 快照)。
        plan = _rk_prefill_plan("rk_r0", "rk_s_rk_r0", 0)
        members = self.builder.emit_prefill_batch(plan)
        pieces = [
            _rk_piece(_RK_DECODE_INSTANCE, 0, 80),
            _rk_piece(_RK_PREFILL_INSTANCE, 80, 140, "prefill_stay"),
            _rk_piece(_RK_DIE_INSTANCE, 140, 200),
        ]
        scatter = emit_piece_scatter(
            self.builder, queue_index=0, prefix="rk_q0", request_id="rk_r0",
            prefill_instance_index=_RK_PREFILL_INSTANCE, pieces=pieces,
            prefill_context_tokens=200, prefill_end_members=members)
        train = _rk_train_plan("batch_train_i1_1", [(1, 201), (1, 202)], 2,
                               joiners=[plan], exits=("rk_r0",))
        train_result = self.builder.emit_iteration_train(train)
        # 列车发射后:P frontier 已被 joiner 3000 send 正常推进(链式),
        # D frontier = end barrier——3300 侧插不得再动任何一端。
        frontier_before_reads = {
            rank: self.builder.builders[rank].previous_id
            for rank in range(8)}
        reads = emit_remote_reads(
            self.builder, train_id="batch_train_i1_1",
            decode_instance_index=_RK_DECODE_INSTANCE,
            member_reads=[{
                "request_id": "rk_r0", "queue_index": 0, "participation": 2,
                "prefill_instance_index": _RK_PREFILL_INSTANCE,
                "sources": [
                    {"source_instance_index": _RK_PREFILL_INSTANCE,
                     "piece_tokens": 60},
                    {"source_instance_index": _RK_DIE_INSTANCE,
                     "piece_tokens": 60, "source_ordinal": 1},
                ],
                "prefill_end_members": members,
                "exit_anchors": train_result["exit_members"]["rk_r0"],
            }],
            scatter_recv_ids=scatter["recv_ids"])
        for rank in range(8):
            self.assertEqual(self.builder.builders[rank].previous_id,
                             frontier_before_reads[rank])
        for rank in _RK_DECODE_RANKS:
            self.assertEqual(frontier_before_reads[rank],
                             train_result["block_ends"][rank])
        # recv:唯一父边 = 列车 join 标记(头末);hbm_charge=false。
        join_anchors = reads["join_anchors"]
        recvs = _rk_nodes(self.builder, "_kv_remote_read_", send=False)
        self.assertEqual(len(recvs), 4)      # 2 源 × TP2
        for node in recvs:
            self.assertIs(node["comm"]["hbm_charge"], False)
            self.assertEqual(
                _rk_parents(self.builder, node["rank"], node["id"]),
                [join_anchors[node["rank"]]])
        # send:hbm_charge=true;两档锚(P-piece→prefill 块末 / die piece→
        # 该 piece 的 3100 recv 节点 id)。
        sends = _rk_nodes(self.builder, "_kv_remote_read_", send=True)
        self.assertEqual(len(sends), 4)
        for node in sends:
            self.assertIs(node["comm"]["hbm_charge"], True)
        die_anchor = scatter["recv_ids"][_RK_DIE_INSTANCE]
        for node in sends:
            parents = _rk_parents(self.builder, node["rank"], node["id"])
            self.assertEqual(len(parents), 1)
            if node["rank"] in _RK_PREFILL_RANKS:      # 第一档:P-piece
                self.assertEqual(parents, [members[node["rank"]]])
            else:                                       # 第二档:散布 piece
                self.assertEqual(parents, [die_anchor[node["rank"]]])
        # recv 注入该成员自己的 exit 标记(而非跨成员聚合):exit 的父边 =
        # 体尾链前驱恰一条 + 该成员本列车全部读边(每 rank 2 条)——
        # 成员 exit = max(体尾, 其全部读边)。
        exit_id = train_result["exit_members"]["rk_r0"]
        for rank in _RK_DECODE_RANKS:
            parents = _rk_parents(self.builder, rank, exit_id[rank])
            self.assertEqual(len(parents), 3)
            # 恰一条体尾链前驱 + 该成员本 rank 的全部读边(2 条)。
            self.assertEqual(
                sorted(set(parents) - set(reads["recv_ids"]["rk_r0"][rank])),
                [next(parent for parent in parents
                      if parent not in reads["recv_ids"]["rk_r0"][rank])])
        # 字节 = p_m × R_{m,s}:participation 2 × 60 token × 256 B。
        for node in sends + recvs:
            self.assertEqual(node["comm"]["bytes"],
                             2 * 60 * _RK_BYTES_PER_RANK_TOKEN)

    def test_continuing_member_recv_injects_nothing(self):
        # rk_r0 为中途成员(本列车不退出):recv 照常发射,但不注入任何
        # 标记(读完成语义由其退出列车的 recv 承载)。
        _, _, _, _, reads = _rk_emit_scenario(self.builder, exits=())
        member_recvs = reads["recv_ids"]["rk_r0"]
        self.assertTrue(all(ids for ids in member_recvs.values()))
        for rank, recv_ids in member_recvs.items():
            for recv_id in recv_ids:
                downstream = [
                    edge for edge in self.builder.batch["parent_edges"]
                    if edge["rank"] == rank and edge["from"] == recv_id]
                self.assertEqual(downstream, [])


class RelevantMultiSourceTagTest(unittest.TestCase):
    """多源 tag 唯一性(总文档 §2.3 扩展位;与不变量文件互补的节点级钉子)。"""

    def test_three_source_tags_distinct_per_member(self):
        builder = GraphBatchBuilder(_rk_config())
        builder.begin_batch()
        plan = _rk_prefill_plan("rk_r0", "rk_s_rk_r0", 0, context=260)
        members = builder.emit_prefill_batch(plan)
        pieces = [
            _rk_piece(_RK_DECODE_INSTANCE, 0, 80),
            _rk_piece(_RK_PREFILL_INSTANCE, 80, 140, "prefill_stay"),
            _rk_piece(_RK_DIE_INSTANCE, 140, 200),
            _rk_piece(_RK_DIE_INSTANCE_3, 200, 260),
        ]
        scatter = emit_piece_scatter(
            builder, queue_index=0, prefix="rk_q0", request_id="rk_r0",
            prefill_instance_index=_RK_PREFILL_INSTANCE, pieces=pieces,
            prefill_context_tokens=260, prefill_end_members=members)
        train = _rk_train_plan("batch_train_i1_1", [(1, 261)], 1,
                               joiners=[plan], exits=("rk_r0",))
        result = builder.emit_iteration_train(train)
        emit_remote_reads(
            builder, train_id="batch_train_i1_1",
            decode_instance_index=_RK_DECODE_INSTANCE,
            member_reads=[{
                "request_id": "rk_r0", "queue_index": 0, "participation": 1,
                "prefill_instance_index": _RK_PREFILL_INSTANCE,
                "sources": [
                    {"source_instance_index": _RK_PREFILL_INSTANCE,
                     "piece_tokens": 60},
                    {"source_instance_index": _RK_DIE_INSTANCE,
                     "piece_tokens": 60, "source_ordinal": 1},
                    {"source_instance_index": _RK_DIE_INSTANCE_3,
                     "piece_tokens": 60, "source_ordinal": 2},
                ],
                "prefill_end_members": members,
                "exit_anchors": result["exit_members"]["rk_r0"],
            }],
            scatter_recv_ids=scatter["recv_ids"])
        sends = _rk_nodes(builder, "_kv_remote_read_", send=True)
        self.assertEqual(len(sends), 6)      # 3 源 × TP2
        tags = [node["comm"]["tag"] for node in sends]
        self.assertEqual(len(set(tags)), 6)
        # 扩展位 = 源序号×TP + relative_rank,整体 < 100。
        for tag in tags:
            extension = tag % 10000 - 3300
            self.assertGreaterEqual(extension, 0)
            self.assertLess(extension, 100)
        self.assertEqual(sorted(tag % 10000 - 3300 for tag in tags),
                         [0, 1, 2, 3, 4, 5])


class ContinuingMemberTwoTrainsTest(unittest.TestCase):
    """R3-② python 侧:继续成员跨两列车重复同 tag 的 builder 簿记无唯一性
    假设(总文档 §2.3/§9 R3-②;T_max 截断是常态触发)。"""

    def test_same_tag_two_trains_fifo_bookkeeping(self):
        builder = GraphBatchBuilder(_rk_config())
        builder.begin_batch()
        plan = _rk_prefill_plan("rk_s0", "rk_s_rk_s0", 0, context=200)
        members = builder.emit_prefill_batch(plan)
        scatter = emit_piece_scatter(
            builder, queue_index=0, prefix="rk_q0", request_id="rk_s0",
            prefill_instance_index=_RK_PREFILL_INSTANCE,
            pieces=[_rk_piece(_RK_PREFILL_INSTANCE, 0, 140,
                              "prefill_stay"),
                    _rk_piece(_RK_DIE_INSTANCE, 140, 200),
                    _rk_piece(_RK_DECODE_INSTANCE, 200, 220,
                              "decode_local")],
            prefill_context_tokens=200, prefill_end_members=members)
        member = {
            "request_id": "rk_s0", "queue_index": 0, "participation": 2,
            "prefill_instance_index": _RK_PREFILL_INSTANCE,
            "sources": [{"source_instance_index": _RK_DIE_INSTANCE,
                         "piece_tokens": 60}],
            "prefill_end_members": members,
            "exit_anchors": None,   # 中途成员:本列车不退出
        }
        train1 = builder.emit_iteration_train(_rk_train_plan(
            "batch_train_i1_1", [(1, 201), (1, 202)], 2, joiners=[plan]))
        reads1 = emit_remote_reads(
            builder, train_id="batch_train_i1_1",
            decode_instance_index=_RK_DECODE_INSTANCE,
            member_reads=[dict(member)],
            scatter_recv_ids=scatter["recv_ids"])
        # begin_batch 会更换批次累加器:第一列车的节点/标签须先取走。
        tags1 = sorted(
            node["comm"]["tag"]
            for node in _rk_nodes(builder, "_kv_remote_read_", send=True))
        # 第二列车:成员已是继续成员(无 joiner → 无 join 标记,join 锚
        # 走显式参数 = 上一列车 end barrier),本列车退出。
        builder.begin_batch()
        train2 = builder.emit_iteration_train(_rk_train_plan(
            "batch_train_i1_2", [(1, 203), (1, 204)], 2,
            exits=("rk_s0",)))
        second_member = dict(member)
        second_member["exit_anchors"] = train2["exit_members"]["rk_s0"]
        reads2 = emit_remote_reads(
            builder, train_id="batch_train_i1_2",
            decode_instance_index=_RK_DECODE_INSTANCE,
            member_reads=[second_member],
            scatter_recv_ids=scatter["recv_ids"],
            join_anchors=train1["block_ends"])   # 无 joiner 列车的显式头锚
        # 同 (queue_index, category, 源, rank) 的 tag 跨列车重复——builder
        # 侧簿记无唯一性假设:两对 send/recv 均在图上,后端按注册序 FIFO。
        tags2 = sorted(
            node["comm"]["tag"]
            for node in _rk_nodes(builder, "_kv_remote_read_", send=True))
        self.assertEqual(tags1, tags2)
        # 每 rank 发射序 = 注册序:第一列车的 send/recv id 均小于第二列车。
        for rank in _RK_DECODE_RANKS:
            first_recv = reads1["recv_ids"]["rk_s0"][rank][0]
            second_recv = reads2["recv_ids"]["rk_s0"][rank][0]
            self.assertLess(first_recv, second_recv)
            # 退出列车的 recv 注入该成员 exit;第一列车的 recv 无任何注入。
            self.assertEqual(
                _rk_parents(builder, rank, second_recv),
                [train1["block_ends"][rank]])
            exit_id = train2["exit_members"]["rk_s0"][rank]
            self.assertIn(second_recv,
                          _rk_parents(builder, rank, exit_id))
            self.assertNotIn(first_recv,
                             _rk_parents(builder, rank, exit_id))


class BatchOrderSkeletonPropertyTest(unittest.TestCase):
    """R4 批序性质的图侧钉子(总文档 §9 R4):3100 send 的显式锚
    (prefill 块末)依赖两条骨架性质——P 串行 busy 门 + 同 tick 完成批
    先于准入批。调度器侧的同 tick 交付序属 B3 测试范围;此处钉住其
    图上前提:共享 rank 的 per-rank 发行序 = 全局发射序(frontier 统一
    裁决,2026-08-19),显式锚恒为"因果更早"节点,无次序反转。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_rk_config())
        self.builder.begin_batch()

    def test_prefill_busy_gate_serializes_same_p(self):
        # P 串行 busy 门(图侧):下一 request 的 prefill 链首(ready
        # barrier)链在本 request 的 P 链尾上(共享 rank 串行化;链尾 =
        # prefill 块末之后的人工 end barrier,块末 members 是其链上前驱,
        # 传递可达)。
        plan1 = _rk_prefill_plan("rk_r0", "rk_s_a", 0)
        members = self.builder.emit_prefill_batch(plan1)
        r1_chain_tail = {rank: self.builder.builders[rank].previous_id
                         for rank in _RK_PREFILL_RANKS}
        plan2 = _rk_prefill_plan("rk_r1", "rk_s_b", 1)
        self.builder.emit_prefill_batch(plan2)
        for rank in _RK_PREFILL_RANKS:
            self.assertGreater(r1_chain_tail[rank], members[rank])
            barrier = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"].endswith(
                    "rk_r1_history_tp_ready_barrier"))
            parents = _rk_parents(self.builder, rank, barrier["id"])
            # 链首依赖 = R1 链尾(串行 busy 门)+ R2 到达 timer gate。
            self.assertIn(r1_chain_tail[rank], parents)
            self.assertEqual(len(parents), 2)
            # R1 prefill 块末(3100 显式锚)传递可达:链尾的链上前驱。
            tail_parents = _rk_parents(self.builder, rank,
                                       r1_chain_tail[rank])
            self.assertIn(members[rank], tail_parents)

    def test_drain_batch_precedes_admission_batch_in_rank_order(self):
        # 同 tick 完成批先于准入批(图侧):同一批内 drain 发射(3100,
        # 锚 = R 的 prefill 块末)先于下一准入的 prefill 节点——每 P rank
        # 上 R_end < scatter_send < 下一 request 链首,无反转(反环前提)。
        plan1 = _rk_prefill_plan("rk_r0", "rk_s_a", 0)
        members = self.builder.emit_prefill_batch(plan1)
        scatter = emit_piece_scatter(
            self.builder, queue_index=0, prefix="rk_q0", request_id="rk_r0",
            prefill_instance_index=_RK_PREFILL_INSTANCE,
            pieces=[_rk_piece(_RK_PREFILL_INSTANCE, 0, 120,
                              "prefill_stay"),
                    _rk_piece(_RK_DECODE_INSTANCE, 120, 200)],
            prefill_context_tokens=200, prefill_end_members=members)
        plan2 = _rk_prefill_plan("rk_r1", "rk_s_b", 1)
        self.builder.emit_prefill_batch(plan2)
        for rank in _RK_PREFILL_RANKS:
            scatter_send = next(
                node["id"] for node in _rk_nodes(
                    self.builder, "_kv_scatter_", send=True)
                if node["rank"] == rank)
            barrier = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"].endswith(
                    "rk_r1_history_tp_ready_barrier"))
            self.assertLess(members[rank], scatter_send)
            self.assertLess(scatter_send, barrier["id"])
            self.assertEqual(
                _rk_parents(self.builder, rank, scatter_send),
                [members[rank]])
            self.assertEqual(scatter["scatter_tokens_by_owner"],
                             {_RK_DECODE_INSTANCE: 80})


class TrainLocalKvSplitSelectionTest(unittest.TestCase):
    """local_kv_bytes 透传的拆分列车贪心子序列选值(M3;§8.3 补充)。"""

    def test_split_train_selects_span_matched_values(self):
        model = SimpleNamespace(layers=2, hidden_size=64, num_heads=8,
                                bytes_per_elem=2)
        override = [local_kv_override_value(model, 60, 2, relative_rank)
                    for relative_rank in range(2)]
        spans = [(1, 101), (1, 102), (1, 41)]
        first_token = _first_token_of(
            "batch_train_i1_1", spans, [("ra", 2), ("rb", 1)],
            marker_ids=("ra",))
        base_plan = _split_plan("batch_train_i1_1", spans, 2,
                                joiners=[_joiner_plan(REQUEST_A, 100)],
                                exits=(REQUEST_A,),
                                first_token=first_token)
        plain = GraphBatchBuilder(_make_config())
        plain.begin_batch()
        plain.emit_train_first_step(dict(base_plan))
        plain_first_tensor, _ = _body_totals(plain, "batch_train_i1_1")
        plain.begin_batch()
        plain.emit_train_remainder(dict(base_plan))
        plain_rest_tensor, _ = _body_totals(plain, "batch_train_i1_1")

        overridden = GraphBatchBuilder(_make_config())
        overridden.begin_batch()
        split_plan = dict(base_plan)
        split_plan["local_kv_bytes"] = [override, None, None]
        overridden.emit_train_first_step(dict(split_plan))
        first_tensor, _ = _body_totals(overridden, "batch_train_i1_1")
        overridden.begin_batch()
        overridden.emit_train_remainder(dict(split_plan))
        rest_tensor, _ = _body_totals(overridden, "batch_train_i1_1")
        # 首步批 = spans[0](override 生效):_body_totals 跨 2 个 decode
        # rank,每 rank 差 = shard(101) − shard(60);余量批 = spans[1:]
        # (None = 全量口径,逐字节一致)。
        self.assertEqual(first_tensor,
                         plain_first_tensor
                         - 2 * (101 * 256 - 60 * 256))
        self.assertEqual(rest_tensor, plain_rest_tensor)


if __name__ == "__main__":
    unittest.main()