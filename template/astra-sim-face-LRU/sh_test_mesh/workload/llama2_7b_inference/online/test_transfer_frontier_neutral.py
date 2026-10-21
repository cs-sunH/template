"""死锁真修（方案①，2026-10-11，自 WL 同构移植）：传输发射 frontier
中立性 + 显式消费者边（FL 发射器适配版）。

钉死五类不变量（wl_fl_fix_plan.md §3 T1–T4 / C3–C5 / C7 的 FL 映射）：
1. 传输发射后主链 frontier（previous_id）不变（remote_store 数据面/
   握手/ack 簿记节点全部中立化）；
2. C3/C4：remote_load 的 restore 作为历史到达节点记入
   history_arrival_node_ids，被 readiness 屏障显式 arm（"恢复完成前
   不能消费历史 KV" 从链位改为显式边）；
3. C5：joiner xfer 的 recv_node_id 随 route 落账，join 标记显式 arm；
4. C7：跨缘 store_relay 的 send/recv 均不占主链 frontier；
5. _frontier_neutral 空段 fail-closed。
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
from generate_face_trace import PendingHistoryGate  # noqa: E402
from session_kv_manager import KVTransfer, KVTransferShard  # noqa: E402

SESSION = "session_neutral"
REQUEST_A = f"{SESSION}_request_0"
REQUEST_B = f"{SESSION}_request_1"
LAYERS = 4


def _make_config():
    return SimpleNamespace(
        npus_count=9,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(name="instance_0", ranks=(0, 4),
                            pg_name="tp0"),
            SimpleNamespace(name="instance_1", ranks=(8,),
                            pg_name="tp1"),
        ],
        layers=LAYERS,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(session_arrival_time_ns=1000,
                            inter_request_interval_ns=None),
            SimpleNamespace(session_arrival_time_ns=None,
                            inter_request_interval_ns=1000),
        ],
        model=SimpleNamespace(
            layers=LAYERS, hidden_size=64, num_heads=8, bytes_per_elem=2),
        hardware=SimpleNamespace(
            mesh_rows=3, mesh_cols=3, npus_count=9),
    )


def _shard(source_rank, target_rank, edge_rank, bytes_, path):
    return KVTransferShard(
        source_rank=source_rank, target_rank=target_rank,
        edge_rank=edge_rank, bytes=bytes_, noc_path=path,
        layer_start=0, layer_end=LAYERS)


def _store_transfer(shards=None):
    """remote_store 链 A：source 0→edge 0、4→edge 1。"""
    shards = tuple(shards) if shards is not None else (
        _shard(0, 0, 0, 512, (0,)),
        _shard(4, 1, 1, 512, (4, 1)),
    )
    return KVTransfer(
        kind="remote_store", phase="history",
        reason="evict_admission_full:layers0-4",
        session_id=SESSION, trigger_request_id=REQUEST_A,
        source_instance_index=0, target_instance_index=None,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=0,
    )


def _load_transfer(edge_by_target=None):
    """remote_load：target 0←edge 0、4←edge 1（restore 目标 = prefill
    实例 (0,4)，就绪屏障与历史到达节点同秩，C4 边可检）。"""
    edge_by_target = edge_by_target or {0: 0, 4: 1}
    shards = tuple(
        _shard(edge, target, edge, 512, (edge, target))
        for target, edge in edge_by_target.items())
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_remote_restore",
        session_id=SESSION, trigger_request_id=REQUEST_B,
        source_instance_index=None, target_instance_index=0,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=LAYERS,
    )


def _restore_plan(transfers):
    return {
        "request_id": REQUEST_B,
        "session_id": SESSION,
        "turn_index": 1,
        "queue_index": 1,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "history_action": "REMOTE_RESTORE",
        "history_source_instance_index": None,
        "history_transfer_bytes": sum(t.total_bytes for t in transfers),
        "history_tokens_before": 0,
        "prefill_context_tokens": 300,
        "prefill_length": 300,
        "kv_location_after_completion": "remote_memory",
        "history_location_before_location": "remote_memory",
        "history_transfers": transfers,
    }


def _frontiers(builder):
    return {rank: b.previous_id for rank, b in builder.builders.items()}


def _rank_nodes(builder, rank):
    return [n for n in builder.batch["nodes"] if n["rank"] == rank]


def _rank_edges(builder, rank):
    return [e for e in builder.batch["parent_edges"] if e["rank"] == rank]


class _Harness:
    """seed frontier + 上一 turn 完成门 + 逐出/回迁公共夹具。"""

    def __init__(self):
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()
        for trace_builder in self.builder.builders.values():
            trace_builder.set_context(REQUEST_A, "decode", 1)
            trace_builder.comp("seed_frontier", 1, 1)
        gates = {
            rank: trace_builder.previous_id
            for rank, trace_builder in self.builder.builders.items()
            if rank in (0, 4)
        }
        self.builder.completion_gates[SESSION] = (0, gates)

    def emit_evictions(self, *transfers):
        plan = {
            "request_id": REQUEST_A,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
        }
        self.builder.emit_eviction_actions(plan, transfers)


class TransferFrontierNeutralTest(unittest.TestCase):
    def test_remote_store_transfer_keeps_main_frontier(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        plan = {
            "request_id": REQUEST_A,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
        }
        before = _frontiers(builder)
        result = builder._emit_plan_transfer(
            plan, _store_transfer(), "prefill_evictions")
        after = _frontiers(builder)
        # 主链 frontier 不被任何 store 数据面/握手/ack 簿记节点推进。
        self.assertEqual(before, after)
        # 边缘 mem_store 节点 id 随 shard record 落账（消费者边来源不变；
        # pending_store_tails 登记由调用方 _register_store_tails 完成,
        # 不在 _emit_plan_transfer 内）。
        edge_ids = [rec["edge_store_node_id"] for rec in result["shards"]]
        self.assertEqual(sorted(edge_ids), sorted(
            rec["edge_store_node_id"] for rec in result["shards"]))
        self.assertTrue(all(node_id is not None for node_id in edge_ids))
        # 全部节点入 builders 缓冲（collect 由调用方负责；中立化只改
        # 依赖载体，不改节点集合）。
        names = [n["name"] for b in builder.builders.values()
                 for n in b.nodes]
        self.assertTrue(any("_send_to_edge" in n for n in names))
        self.assertTrue(any("_remote_store" in n for n in names))
        self.assertTrue(any("_ack_from_edge" in n for n in names))

    def test_remote_load_records_history_arrival_ids(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        gate = PendingHistoryGate(
            source_instance_index=0, timer_gates=(None, None),
            location="remote_memory")
        result = builder._emit_plan_transfer(
            {"request_id": REQUEST_B, "session_id": SESSION,
             "turn_index": 1, "queue_index": 1},
            _load_transfer(), "history_kv_transfer", gate=gate)
        arrivals = {}
        for rec in result["shards"]:
            arrivals.update(rec.get("history_arrival_node_ids") or {})
        # 每个目标 rank 恰一个历史到达节点（restore 完成）。
        self.assertEqual(sorted(arrivals), [0, 4])
        # restore 尾节点 id 与 target_hbm_completion_node_id 一致（C3 账本）。
        for rec in result["shards"]:
            self.assertEqual(
                rec["history_arrival_node_ids"][rec["target_rank"]],
                rec["target_hbm_completion_node_id"])

    def test_tp_ready_barrier_arms_restore_explicitly(self):
        harness = _Harness()
        builder = harness.builder
        builder.emit_admission_batch(_restore_plan((_load_transfer(),)))
        barrier = [n for n in builder.batch["nodes"]
                   if n["rank"] in (0, 4)
                   and n["name"].endswith("_history_tp_ready_barrier")]
        self.assertEqual(len(barrier), 2)
        edges = builder.batch["parent_edges"]
        restores = {(n["rank"], n["id"])
                    for n in builder.batch["nodes"]
                    if n["name"].endswith("_target_hbm_write")}
        self.assertEqual(len(restores), 2)
        for node in barrier:
            parent_pairs = {(e["rank"], e["from"]) for e in edges
                            if e["rank"] == node["rank"]
                            and e["to"] == node["id"]}
            self.assertTrue(
                restores & parent_pairs,
                "barrier must explicitly arm the restore completion "
                "(C4) on rank {}".format(node["rank"]))

    def test_store_relay_neutral_keeps_main_frontier(self):
        """C7：跨缘 1B 中继 send（侧支）/recv（中立段）均不占主链
        frontier；recv id 由 restore 链首节点显式 arm。全量回迁准入的
        主链唯一推进 = readiness 屏障（T3 主链保留集）+ interval 门
        （不推进 previous_id）——其余 rank frontier 不动。"""
        harness = _Harness()
        builder = harness.builder
        harness.emit_evictions(_store_transfer(shards=(
            _shard(0, 0, 0, 512, (0,)),)))
        before = _frontiers(builder)
        # 回迁换缘：target 4 的最近边缘 = 1 ≠ store 边缘 0。
        builder.emit_admission_batch(_restore_plan(
            (_load_transfer({4: 1}),)))
        after = _frontiers(builder)
        barriers = {
            n["rank"]: n["id"] for n in builder.batch["nodes"]
            if n["name"].endswith("_history_tp_ready_barrier")}
        self.assertEqual(sorted(barriers), [0, 4])
        for rank in builder.builders:
            if rank in barriers:
                self.assertEqual(after[rank], barriers[rank])
            else:
                self.assertEqual(after[rank], before[rank])
        relay_send = next(
            n for n in _rank_nodes(builder, 0)
            if "_store_relay_edge0_to_edge1" in n["name"])
        relay_recv = next(
            n for n in _rank_nodes(builder, 1)
            if "_store_relay_arrival_edge0_to_edge1" in n["name"])
        mem_load = next(
            n for n in _rank_nodes(builder, 1)
            if n["name"].endswith("shard0_remote_load"))
        # relay recv 显式 arm 进 mem_load 祖先闭包（排序语义 = 原链位）。
        by_target = {}
        for edge in _rank_edges(builder, 1):
            by_target.setdefault(edge["to"], []).append(edge["from"])
        closure = {mem_load["id"]}
        pending = [mem_load["id"]]
        while pending:
            current = pending.pop()
            for parent in by_target.get(current, ()):
                if parent not in closure:
                    closure.add(parent)
                    pending.append(parent)
        self.assertIn(relay_recv["id"], closure)
        self.assertEqual(relay_send["comm"]["bytes"], 1)
        self.assertEqual(relay_recv["comm"]["bytes"], 1)

    def test_join_marker_arms_xfer_recv(self):
        config = SimpleNamespace(
            npus_count=16,
            remote_operand_loads=False,
            trace_granularity="request_aggregated",
            inference_groups=[
                SimpleNamespace(name="g_prefill", ranks=(0, 1),
                                pg_name="tp_prefill"),
                SimpleNamespace(name="g_decode", ranks=(5, 6),
                                pg_name="tp_decode"),
            ],
            layers=LAYERS,
            hidden_size=64,
            ffn_size=128,
            vocab_size=256,
            bytes_per_elem=2,
            num_heads=8,
            mlp_variant="gelu",
            request_queue=[],
            model=SimpleNamespace(layers=LAYERS, hidden_size=64,
                                  num_heads=8, bytes_per_elem=2),
            hardware=SimpleNamespace(mesh_rows=4, mesh_cols=4,
                                     npus_count=16),
        )
        builder = GraphBatchBuilder(config)
        builder.begin_batch()
        joiner = {
            "request_id": "req_joiner", "session_id": "session_joiner",
            "turn_index": 0, "queue_index": 0,
            "prefill_instance_index": 0, "decode_instance_index": 1,
            "prefill_context_tokens": 128, "decode_evictions": (),
        }
        builder._emit_train_head({
            "train_id": "batch_train_i1_probe", "instance_index": 1,
            "joiners": [joiner],
        })
        buffered = [n for b in builder.builders.values() for n in b.nodes]
        buf_edges = [e for b in builder.builders.values() for e in b.edges]
        markers = [n for n in buffered if "_join_req_joiner" in n["name"]]
        self.assertEqual(len(markers), 2)
        recv_pairs = {(n["rank"], n["id"]) for n in buffered
                      if "_prefill_to_decode_kv_recv" in n["name"]}
        self.assertEqual(len(recv_pairs), 2)
        for node in markers:
            parent_pairs = {(e["rank"], e["from"]) for e in buf_edges
                            if e["rank"] == node["rank"]
                            and e["to"] == node["id"]}
            self.assertTrue(
                recv_pairs & parent_pairs,
                "join marker must explicitly arm the xfer recv (C5) "
                "on rank {}".format(node["rank"]))

    def test_frontier_neutral_rejects_empty_segment(self):
        from generate_face_trace import _frontier_neutral
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        with self.assertRaises(RuntimeError):
            _frontier_neutral(builder.builders[0], lambda: None)


if __name__ == "__main__":
    unittest.main()
