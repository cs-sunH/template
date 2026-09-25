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
from generate_face_trace import (  # noqa: E402
    TransferTagAllocator,
    _emit_kv_transfer,
)
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
    plan = {
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
    return plan


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
        stage=prefill/generation=0。B3:drain 成员的 seg1 块末 = 列车
        post-barrier 节点(joiner decode_evictions 触发门口径)。"""
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

    def test_remote_restore_session_train_is_single_aggregate(self):
        """Session 级 Tiered-LRU(2026-09-25):远端恢复会话的列车统一
        聚合体发射——无 first_chunk_prefix/suffix 层段拆分节点、无
        remaining_aggregated 节点,_partial_first_chunk 账本不存在
        (PARTIAL 真流水已随两阶段流程删除;权重经 weight_passes 摊销
        不变)。"""
        builder = self.builder
        self.assertFalse(hasattr(builder, "_partial_first_chunk"))
        spans = [(128, 128), (128, 256), (44, 300)]
        plan = _train_plan(
            "batch_train_i0_9", spans, iterations=3, stage="prefill",
            drains=(REQUEST_A,), instance_index=0,
            prefill_start={"request_id": REQUEST_A})
        builder.emit_iteration_train(plan)
        for rank in (0, 1):
            names = [node["name"] for node in _rank_nodes(builder, rank)]
            body = [name for name in names
                    if name.startswith("batch_train_i0_9")]
            # 统一聚合体:恰一个列车体节点序列(train_id 即 phase),
            # 无任何层段拆分/余量节点。
            self.assertTrue(body)
            self.assertFalse(
                any("first_chunk_prefix" in name for name in names))
            self.assertFalse(
                any("first_chunk_suffix" in name for name in names))
            self.assertFalse(
                any("remaining_aggregated" in name for name in names))
            layer_nodes = [node for node in _rank_nodes(builder, rank)
                           if node["request_id"] == "batch_train_i0_9"
                           and "_layers" in node["name"]]
            self.assertTrue(layer_nodes)
            # 全部层类节点都在全层域(all_layers,无 0_0/1_1 层段标签)。
            self.assertTrue(
                all("all_layers" in node["name"] for node in layer_nodes))

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

class KVTransferBillingNailTest(unittest.TestCase):
    """B3(2026-09-06):remote_store/remote_load 物理链 HBM 计费键钉子
    (契约 §12.5 计费三险 + §6 tag 基址 + §7 命名禁区)。

    3x3 mesh(rank 4 为内部,边缘集合 = 8 个边界 rank;最近边缘平局取
    编号最小):source 4 → edge 1(链 A)、source 0 → edge 0(链 B 直连)、
    remote_load edge 1 → target 4(全链)。"""

    def _make_builder(self):
        config = SimpleNamespace(
            npus_count=9,
            remote_operand_loads=False,
            trace_granularity="request_aggregated",
            inference_groups=[
                SimpleNamespace(name="instance_0", ranks=(0, 4),
                                pg_name="tp0"),
                SimpleNamespace(name="instance_1", ranks=(8,),
                                pg_name="tp1"),
            ],
            layers=4,
            hidden_size=64,
            ffn_size=128,
            vocab_size=256,
            bytes_per_elem=2,
            num_heads=8,
            mlp_variant="gelu",
            hardware=SimpleNamespace(
                mesh_rows=3, mesh_cols=3, npus_count=9),
        )
        builder = GraphBatchBuilder(config)
        builder.begin_batch()
        return builder

    @staticmethod
    def _shard(source_rank, target_rank, edge_rank, bytes_, path):
        from session_kv_manager import KVTransferShard
        return KVTransferShard(
            source_rank=source_rank, target_rank=target_rank,
            edge_rank=edge_rank, bytes=bytes_, noc_path=path,
            layer_start=0, layer_end=4)

    def _store(self, shards):
        from session_kv_manager import KVTransfer
        return KVTransfer(
            kind="remote_store", phase="history",
            reason="evict_history_and_prefill_admission_full:"
                   "layers0-4",
            session_id=SESSION, trigger_request_id=REQUEST_A,
            source_instance_index=0, target_instance_index=None,
            total_bytes=sum(shard.bytes for shard in shards),
            shards=tuple(shards), model_layers=4,
            layer_start=0, layer_end=4,
            resident_prefix_layers_before=4,
            resident_prefix_layers_after=0)

    def _load(self, shards):
        from session_kv_manager import KVTransfer
        return KVTransfer(
            kind="remote_load", phase="history",
            reason="history_remote_restore",
            session_id=SESSION, trigger_request_id=REQUEST_A,
            source_instance_index=None, target_instance_index=0,
            total_bytes=sum(shard.bytes for shard in shards),
            shards=tuple(shards), model_layers=4,
            layer_start=0, layer_end=4,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=4)

    def _emit(self, builder, transfer, action, gate=None):
        marker = builder._mark()
        record = _emit_kv_transfer(
            config=builder.config,
            builders=builder.builders,
            group_by_index=builder.group_by_index,
            tag_allocator=builder._tag_allocator,
            transfer=transfer,
            action_name=action,
            pending_gate=gate)
        # 直驱 _emit_kv_transfer 不经 emit_*_batch,手动收集到当前批次。
        builder._collect(marker)
        return record

    def test_remote_store_chain_a_billing_keys(self):
        """链 A(source≠edge):源 comm_send 唯一 COMM_READ 计费(默认
        charged,无 hbm_charge 键);edge comm_recv 过路 hbm_charge=False;
        edge mem_store 池写零本地计费(无 hbm_access_mode 键);1B ack
        双端。"""
        builder = self._make_builder()
        record = self._emit(
            builder,
            self._store([self._shard(4, 1, 1, 512, (4, 1))]),
            "kvtest_remote_store_chain_a")
        self.assertFalse(record["shards"][0]["direct_edge_access"])
        send = next(node for node in builder.batch["nodes"]
                    if node["name"].endswith("shard0_send_to_edge1"))
        self.assertNotIn("hbm_charge", send["comm"])
        passthrough = next(node for node in builder.batch["nodes"]
                           if node["name"].endswith("shard0_recv_from_rank4"))
        self.assertIs(passthrough["comm"].get("hbm_charge"), False)
        pool_write = next(node for node in builder.batch["nodes"]
                          if node["name"].endswith("shard0_remote_store"))
        self.assertNotIn("hbm_access_mode", pool_write["compute"])
        ack_send = next(node for node in builder.batch["nodes"]
                        if node["name"].endswith("shard0_ack_to_rank4"))
        ack_recv = next(node for node in builder.batch["nodes"]
                        if node["name"].endswith("shard0_ack_from_edge1"))
        self.assertEqual(ack_send["comm"]["bytes"], 1)
        self.assertEqual(ack_recv["comm"]["bytes"], 1)

    def test_remote_store_chain_b_billing_keys(self):
        """链 B(source==edge 直连):mem_store(hbm_access_mode=1) =
        POOL_READ 唯一计费。"""
        builder = self._make_builder()
        record = self._emit(
            builder,
            self._store([self._shard(0, 0, 0, 512, (0,))]),
            "kvtest_remote_store_chain_b")
        self.assertTrue(record["shards"][0]["direct_edge_access"])
        edge_store = next(node for node in builder.batch["nodes"]
                          if node["name"].endswith("shard0_edge_store"))
        self.assertEqual(edge_store["compute"].get("hbm_access_mode"), 1)
        self.assertEqual(edge_store["compute"]["tensor_size"], 512)

    def test_remote_load_billing_keys(self):
        """remote_load:edge mem_load 仅池 FIFO;edge comm_send / target
        comm_recv 均 hbm_charge=False;target local_hbm_kv_restore =
        RESTORE 唯一数据计费(不得退化普通 mem_load)。"""
        from generate_face_trace import PendingHistoryGate
        builder = self._make_builder()
        gate = PendingHistoryGate(
            source_instance_index=0, timer_gates=(None, 12),
            location="remote_memory")
        record = self._emit(
            builder,
            self._load([self._shard(1, 4, 1, 512, (1, 4))]),
            "kvtest_remote_load", gate=gate)
        self.assertFalse(record["shards"][0]["direct_edge_access"])
        # control rank 4(target 组相对位 1)arm gate 12 后 1B request 到
        # edge 1——门/arm 边均在同 rank。
        request_send = next(node for node in builder.batch["nodes"]
                            if node["name"].endswith(
                                "shard0_request_to_edge1"))
        self.assertEqual(request_send["comm"]["bytes"], 1)
        edges_from_gate = [
            edge for edge in builder.batch["parent_edges"]
            if edge["from"] == 12]
        self.assertTrue(edges_from_gate)
        pool_load = next(node for node in builder.batch["nodes"]
                         if node["name"].endswith("shard0_remote_load"))
        self.assertNotIn("hbm_access_mode", pool_load["compute"])
        edge_send = next(node for node in builder.batch["nodes"]
                         if node["name"].endswith("shard0_send_to_rank4"))
        self.assertIs(edge_send["comm"].get("hbm_charge"), False)
        target_recv = next(node for node in builder.batch["nodes"]
                           if node["name"].endswith("shard0_recv_from_edge1"))
        self.assertIs(target_recv["comm"].get("hbm_charge"), False)
        restore = next(node for node in builder.batch["nodes"]
                       if node["name"].endswith("shard0_target_hbm_write"))
        self.assertIs(restore.get("is_local_hbm_kv_restore"), True)
        self.assertEqual(record["shards"][0]["target_hbm_completion_node_id"],
                         restore["id"])

    def test_tag_base_and_name_anchors(self):
        """契约 §6:tag 分配器基址 10_000_000(错开 face
        queue*10000+{1000,1900,3000} 段);契约 §7:发射节点名不得含
        first_token/batch_train_ 子串。"""
        from generate_face_trace import PendingHistoryGate
        self.assertEqual(TransferTagAllocator().take(), 10_000_000)
        builder = self._make_builder()
        gate = PendingHistoryGate(
            source_instance_index=0, timer_gates=(None, 12),
            location="remote_memory")
        self._emit(builder,
                   self._store([self._shard(0, 0, 0, 512, (0,))]),
                   "kvtest_anchor_store")
        self._emit(builder,
                   self._load([self._shard(1, 4, 1, 512, (1, 4))]),
                   "kvtest_anchor_load", gate=gate)
        tags = [node["comm"]["tag"] for node in builder.batch["nodes"]
                if node["comm"]["bytes"] > 1]
        self.assertTrue(tags)
        self.assertTrue(all(tag >= 10_000_000 for tag in tags))
        for node in builder.batch["nodes"]:
            self.assertNotIn("first_token", node["name"])
            self.assertNotIn("batch_train_", node["name"])


class _GateRecorder:
    """仅记录 terminal REQUEST_COMPLETE 是否回收构图器完成门。"""

    def __init__(self):
        self.completion_gates = {SESSION: (1, {2: 7, 3: 7})}
        self.retired = []
        self.completions = []
        self.retired_requests = []

    def retire_completion_gate(self, session_id):
        self.retired.append(session_id)
        self.completion_gates.pop(session_id, None)

    def note_request_complete(self, session_id, following_request_id,
                              completion_location):
        # B3:调度器 REQUEST_COMPLETE 边界的 pending location 链登记桩。
        self.completions.append(
            (session_id, following_request_id, completion_location))

    def retire_request_state(self, request_id):
        # B3:逐出块末/action 计数账本核销桩。
        self.retired_requests.append(request_id)


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
        # B3:turn>0 准入携带上一 turn 完成时预置的 pending location
        # (生产路径 runtime.kv_pending_location,REQUEST_COMPLETE 边界写入)
        # 与真实 history_action(此处 LOCAL_HIT:同实例零开销复用)。
        plan["kv_location_after_completion"] = "local_hbm"
        plan["history_action"] = "LOCAL_HIT"
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
            kv_state_after_completion="local_hbm",
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
        # B3:terminal 完成的 pending location 链登记(无 following)与
        # 请求级账本核销。
        self.assertEqual(
            scheduler.graph.completions, [(SESSION, None, "local_hbm")])
        self.assertEqual(scheduler.graph.retired_requests, [REQUEST_A])
        self.assertEqual(scheduler.runtime_by_request_id, {})
        self.assertEqual(scheduler._runtime_index, {})
        self.assertEqual(scheduler.runtimes, [None])
        self.assertEqual(
            scheduler.kv_manager.calls,
            [(SESSION, 123, REQUEST_A)],
        )

    def test_main_intermediate_request_keeps_kv_for_following_turn(self):
        scheduler = FaceOnlineScheduler.__new__(FaceOnlineScheduler)
        runtime = SimpleNamespace(
            request_id=REQUEST_A, session_id=SESSION,
            kv_state_after_completion="local_hbm")
        following = SimpleNamespace(
            request_id=REQUEST_B,
            session_id=SESSION,
            turn_index=1,
            queue_index=1,
            prefill_length=128,
            decode_length=1,
            kv_pending_location=None,
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
        # B3:intermediate 完成登记下一 turn 的 pending location 链 +
        # following runtime 预置 location(下一 turn 准入时经 plan 消费)。
        self.assertEqual(
            scheduler.graph.completions,
            [(SESSION, REQUEST_B, "local_hbm")])
        self.assertEqual(following.kv_pending_location, "local_hbm")


if __name__ == "__main__":
    unittest.main()
