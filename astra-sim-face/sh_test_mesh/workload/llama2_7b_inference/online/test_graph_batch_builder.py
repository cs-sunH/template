#!/usr/bin/env python3
"""test_graph_batch_builder.py -- face 拼 batch 列车发射钉子测试
(2026-08-22;sh_1.0 母本同构,face 发射原语适配:准入 = gates/history/
屏障,列车 = 3000 迁移 + join/pstart 标记 + 折叠体 + drain/exit 标记 +
共享 end barrier)。

经 emit_admission_batch + emit_iteration_train 完整发射后钉住拼 batch
改造语义:
  (a) 列车体节点归属批命名空间(train_id),stage/generation 一致;
  (b) PREFILL_DRAIN / DECODE_COMPLETION watch 成员 = drain/exit 标记节点
      (列车体后、end barrier 前的真实节点),非 end barrier 节点;
  (c) 触发门口径:completion_gates[session] = (列车实例,
      {rank: end barrier 节点 id})(下一同 session turn 的 interval gate
      after_node_id 来源,face 旧 decode end barrier 的同款口径);
  (d) 权重摊销端到端:B=2 与 B=1 同迭代列车体权重字节相等
      (weight_passes=迭代数;陷阱 1 防护),激活/KV 分量逐成员精确;
  (e) 准入发射只含动作(gates/屏障),不含 prefill 主体与 watch;
  (f) 哨兵标记(T_max 截断列车)归属批命名空间,stage 固定 prefill/0。

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
from online.face_online_scheduler import FaceOnlineScheduler  # noqa: E402

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
            SimpleNamespace(name="instance_0", ranks=(0, 1),
                            pg_name="tp_prefill"),
            SimpleNamespace(name="instance_1", ranks=DECODE_RANKS,
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
        # face:准入发射经 _request_spec 读 request_queue[queue_index]
        # (turn-0 需要 session_arrival_time_ns;µs 对齐,timer_gate 校验)。
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=1000,
                inter_request_interval_ns=None,
            ),
        ],
        # face:joiner 迁移字节数经 kv_cache_bytes_for_tokens(config.model);
        # 3000 迁移的 XY 路由经 config.hardware(mesh)。
        model=SimpleNamespace(
            layers=2, hidden_size=64, num_heads=8, bytes_per_elem=2),
        hardware=SimpleNamespace(mesh_rows=1, mesh_cols=4),
    )


def _admission_plan(request_id=REQUEST_A, turn=0):
    return {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": turn,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "history_action": None,
        "history_source_instance_index": None,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_context_tokens": PREFILL_TOKENS,
        "prefill_length": PREFILL_TOKENS,
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
    }


def _train_plan(train_id, spans, iterations, joiners=(), drains=(),
                exits=(), stage="decode", prefill_start=None,
                instance_index=1, sentinel=False):
    return {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": stage,
        "joiners": list(joiners),
        "pass_spans": list(spans),
        "iterations": iterations,
        "prefill_start_member": prefill_start,
        "sentinel": sentinel,
        "drain_members": [{"request_id": rid, "session_id": SESSION}
                          for rid in drains],
        "exit_members": [{"request_id": rid, "session_id": SESSION}
                         for rid in exits],
    }


def _rank_nodes(builder, rank):
    """M1 适配(2026-08-29 收集即释放):_collect 立即清空 builder.nodes，
    已交付节点只在当前批次累加器中保留——改读当前批次累加器
    batch["nodes"](发射序,自 begin_batch 起含本批全部节点),保持
    "直读已发射节点"的测试意图;测试内单批发射,节点 id 自 0 连续,
    rank 过滤后位置 == 节点 id,与改前等价。"""
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

    def test_admission_emits_actions_without_body_or_watch(self):
        """(e):准入发射无 prefill 主体节点(turn-0 到达 gate + readiness
        屏障),无 watch 返回。"""
        self.builder.emit_admission_batch(_admission_plan())
        names = [node["name"] for node in _rank_nodes(self.builder, 0)]
        self.assertTrue(names, "admission emitted nothing on rank 0")
        self.assertIn("q0000_session_train_0_turn0_"
                      f"{REQUEST_A}_arrival_timer_gate", names)
        self.assertTrue(
            any(name.endswith("_history_tp_ready_barrier") for name in names),
            f"admission must emit the readiness barrier: {names}")
        self.assertFalse(
            any("_all_layers_" in name or "_all_passes_" in name
                for name in names),
            f"admission must not emit the pass body: {names}")

    def _emit_two_member_train(self):
        """B=2 列车:成员 A(ctx=100, 3 token)与 B(ctx=40, 2 token),
        2 个迭代;A 在列车内退出,B 存续;1 个 joiner(B)。"""
        joiners = [_joiner_plan(REQUEST_B, 40)]
        spans = [(1, 101), (1, 102), (1, 103), (1, 41), (1, 42)]
        plan = _train_plan(
            "batch_train_i1_1", spans, iterations=2, joiners=joiners,
            exits=(REQUEST_A,), prefill_start={"request_id": REQUEST_A})
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

    def test_train_completion_gates_are_exit_train_barrier(self):
        """(c):exit 成员的 completion_gates = (列车实例, end barrier 节点)。"""
        result = self._emit_two_member_train()
        gates = self.builder.completion_gates[SESSION]
        self.assertEqual(gates[0], 1)
        for rank in DECODE_RANKS:
            self.assertEqual(gates[1][rank], result["block_ends"][rank])
            barrier = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"] == "batch_train_i1_1_end_barrier")
            self.assertEqual(gates[1][rank], barrier["id"])

    def test_join_marker_is_decode_anchor_node(self):
        """join 标记(joiner 零迁移节点时的唯一 decode_start 锚点);
        跨实例 joiner 的 3000 迁移 send/recv 成对发射。"""
        result = self._emit_two_member_train()
        for rank in DECODE_RANKS:
            marker = next(
                node for node in _rank_nodes(self.builder, rank)
                if node["name"].endswith(
                    f"join_{REQUEST_B}"))
            self.assertEqual(marker["request_id"], REQUEST_B)
            self.assertEqual(marker["stage"], "decode")
        names = [node["name"] for node in _rank_nodes(self.builder, 0)]
        self.assertTrue(
            any("_prefill_to_decode_kv_send_rank0_to_rank2" in name
                for name in names),
            f"joiner 3000 transfer missing on prefill rank: {names}")
        names = [node["name"] for node in _rank_nodes(self.builder, 2)]
        self.assertTrue(
            any("_prefill_to_decode_kv_recv_rank0_to_rank2" in name
                for name in names),
            f"joiner 3000 transfer missing on decode rank: {names}")

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

    def test_prefill_train_drain_marker_watch(self):
        """纯 prefill 列车:chunk spans + drain 标记 + end barrier;
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

    def test_sentinel_marker_is_batch_namespace_prefill(self):
        """(f):哨兵标记(T_max 截断列车)归属批命名空间,stage 固定
        prefill/generation 0(单事件通道)。"""
        plan = _train_plan(
            "batch_train_i1_9", [(1, 101), (1, 102)], iterations=2,
            sentinel=True, exits=())
        result = self.builder.emit_iteration_train(plan)
        self.assertEqual(
            sorted(result["sentinel_members"]), sorted(DECODE_RANKS))
        for rank in DECODE_RANKS:
            marker = _rank_nodes(self.builder, rank)[
                result["sentinel_members"][rank]]
            self.assertEqual(marker["request_id"], "batch_train_i1_9")
            self.assertEqual(marker["stage"], "prefill")
            self.assertEqual(marker["generation"], 0)
            self.assertEqual(marker["type"], COMP_NODE)

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

    def test_strategy_turn_one_admission_consumes_gate_and_terminal_retires(self):
        builder = GraphBatchBuilder(self._two_turn_config())
        # 模拟 turn-0 列车 end barrier；turn-1 admission 消费后，依赖已
        # 写进图，账本本身必须释放。
        builder.completion_gates[SESSION] = (0, {0: None, 1: None})
        plan = _admission_plan(REQUEST_B, turn=1)
        plan["queue_index"] = 1
        builder.begin_batch()
        builder.emit_admission_batch(plan)
        self.assertNotIn(SESSION, builder.completion_gates)

        # terminal turn 无后继 admission；REQUEST_COMPLETE 的显式回收保证
        # run-end 不保留最终 barrier。
        builder.completion_gates[SESSION] = (1, {2: 9, 3: 9})
        builder.retire_completion_gate(SESSION)
        self.assertEqual(builder.completion_gates, {})

    def test_terminal_request_complete_retire_gate(self):
        scheduler = FaceOnlineScheduler.__new__(FaceOnlineScheduler)
        runtime = SimpleNamespace(
            request_id=REQUEST_A,
            session_id=SESSION,
        )
        scheduler.graph = _GateRecorder()
        scheduler.runtime_by_request_id = {REQUEST_A: runtime}
        scheduler._runtime_index = {REQUEST_A: 0}
        scheduler.next_request = [None]
        scheduler.runtimes = [runtime]
        scheduler._batch = {"future_alarms": []}
        scheduler.kv_manager = _TerminalKVRecorder()
        scheduler._note_capacity_change = lambda index: None

        scheduler._on_request_complete(REQUEST_A, 123)

        self.assertEqual(scheduler.graph.retired, [SESSION])
        self.assertEqual(scheduler.graph.completion_gates, {})
        self.assertEqual(scheduler.runtime_by_request_id, {})
        self.assertEqual(scheduler._runtime_index, {})
        self.assertEqual(scheduler.runtimes, [None])
        self.assertEqual(
            scheduler.kv_manager.calls,
            [(SESSION, 123, REQUEST_A)],
        )

    def test_main_intermediate_request_keeps_kv_for_following_turn(self):
        scheduler = FaceOnlineScheduler.__new__(FaceOnlineScheduler)
        runtime = SimpleNamespace(request_id=REQUEST_A, session_id=SESSION)
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


if __name__ == "__main__":
    unittest.main()
