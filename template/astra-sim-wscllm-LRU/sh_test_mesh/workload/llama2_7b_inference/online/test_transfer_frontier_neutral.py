"""死锁真修（方案①，2026-10-11）：传输发射 frontier 中立性 + 显式消费者边。

钉死四类不变量（wl_fl_fix_plan.md §3 T1–T4 / C3–C5 / C7）：
1. 传输发射后主链 frontier（previous_id）不变（数据面/握手/簿记节点
   全部中立化）；
2. C3/C4：remote_load 的 restore 与 noc_migrate 的 recv_kv 作为历史
   到达节点记入 history_arrival_node_ids，被 tp_ready_barrier 显式
   arm（"恢复完成前不能消费历史 KV" 从链位改为显式边）；
3. C5：joiner xfer 的 recv_node_id 随 route 落账，join 标记显式 arm；
4. C7：跨缘 store_tail relay 的 send/recv 均不占主链 frontier。
"""

import os
import sys
import unittest

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from generate_wsc_llm_trace import PendingHistoryGate  # noqa: E402
from session_kv_manager import KVTransfer, KVTransferShard  # noqa: E402

PREFILL_RANKS = (0, 1)
DECODE_RANKS = (5, 6)
LAYERS = 4
VICTIM = "victim_neutral"


def _make_config():
    from types import SimpleNamespace
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


def _store_transfer():
    """remote_store 链 A：decode 5→edge 1、6→edge 2。"""
    shards = (
        _shard(source_rank=5, target_rank=1, edge_rank=1, bytes=1500,
               noc_path=(5, 1)),
        _shard(source_rank=6, target_rank=2, edge_rank=2, bytes=1500,
               noc_path=(6, 2)),
    )
    return KVTransfer(
        kind="remote_store", phase="history",
        reason="watermark_admission_session",
        session_id=VICTIM, trigger_request_id="req_evict",
        source_instance_index=1, target_instance_index=None,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=0,
    )


def _load_transfer():
    """remote_load：target 0←edge 5、1←edge 6（restore 目标 = 本计划
    的 prefill 实例 (0,1)，就绪屏障与历史到达节点同秩，C4 边可检）。"""
    shards = (
        _shard(source_rank=1, target_rank=0, edge_rank=1, bytes=900,
               noc_path=(1, 0)),
        _shard(source_rank=2, target_rank=1, edge_rank=2, bytes=900,
               noc_path=(2, 1)),
    )
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_remote_restore", session_id=VICTIM,
        trigger_request_id="req_restore",
        source_instance_index=None, target_instance_index=0,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=LAYERS,
    )


def _migrate_transfer():
    """noc_migrate：decode 5→prefill 0、6→1。"""
    shards = (
        _shard(source_rank=5, target_rank=0, edge_rank=None, bytes=700,
               noc_path=(5, 0)),
        _shard(source_rank=6, target_rank=1, edge_rank=None, bytes=700,
               noc_path=(6, 1)),
    )
    return KVTransfer(
        kind="noc_migrate", phase="history",
        reason="history_noc_migration", session_id=VICTIM,
        trigger_request_id="req_migrate",
        source_instance_index=1, target_instance_index=0,
        total_bytes=sum(s.bytes for s in shards), shards=shards,
        model_layers=LAYERS, layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=LAYERS,
    )


def _plan(**overrides):
    plan = {
        "request_id": "req_neutral", "session_id": "session_neutral",
        "turn_index": 1, "queue_index": 1,
        "prefill_instance_index": 0, "decode_instance_index": 1,
        "history_action": "REMOTE_RESTORE",
        "history_source_instance_index": None,
        "history_transfer_bytes": 0, "history_recompute_tokens": 0,
        "history_tokens_before": 64,
        "prefill_context_tokens": 64 + 256,
        "prefill_length": 256, "decode_length": 4,
        "history_transfers": (), "history_evictions": (),
        "prefill_evictions": (),
    }
    plan.update(overrides)
    return plan


def _frontiers(builder):
    return {rank: b.previous_id for rank, b in builder.builders.items()}


def _history_gate():
    return PendingHistoryGate(
        source_instance_index=1, timer_gates=(None, None),
        location="remote_memory")


class TransferFrontierNeutralTest(unittest.TestCase):
    def test_remote_store_transfer_keeps_main_frontier(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        builder.emit_prefill_batch(_plan(turn_index=0, queue_index=0))
        before = _frontiers(builder)
        builder.begin_batch()
        result = builder._emit_plan_transfer(
            _plan(session_id=VICTIM), _store_transfer(), "prefill_evictions")
        after = _frontiers(builder)
        # 主链 frontier 不被任何 store 数据面/簿记节点推进。
        self.assertEqual(before, after)
        # 边缘 mem_store 节点 id 随 shard record 落账（消费者边来源不变；
        # pending_store_tails 登记由调用方 _register_store_tails 完成,
        # 不在 _emit_plan_transfer 内）。
        edge_ids = [rec["edge_store_node_id"] for rec in result["shards"]]
        self.assertEqual(len(edge_ids), 2)
        self.assertTrue(all(node_id is not None for node_id in edge_ids))
        # 全部节点入 builders 缓冲（collect 由调用方负责；中立化只改
        # 依赖载体，不改节点集合）。
        names = [n["name"] for b in builder.builders.values()
                 for n in b.nodes]
        self.assertTrue(any("_send_to_edge" in n for n in names))
        self.assertTrue(any("_remote_store" in n for n in names))

    def test_remote_load_records_history_arrival_ids(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        result = builder._emit_plan_transfer(
            _plan(session_id=VICTIM),
            _load_transfer(), "history_transfer", gate=_history_gate())
        arrivals = {}
        for rec in result["shards"]:
            arrivals.update(rec.get("history_arrival_node_ids") or {})
        # 每个目标 rank 恰一个历史到达节点（restore 完成）。
        self.assertEqual(sorted(arrivals), [0, 1])
        # restore 尾节点 id 与 target_hbm_completion_node_id 一致（C3 账本）。
        for rec in result["shards"]:
            self.assertEqual(
                rec["history_arrival_node_ids"][rec["target_rank"]],
                rec["target_hbm_completion_node_id"])

    def test_noc_migrate_records_recv_kv_as_arrival(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        result = builder._emit_plan_transfer(
            _plan(session_id=VICTIM),
            _migrate_transfer(), "history_transfer", gate=_history_gate())
        arrivals = {}
        for rec in result["shards"]:
            arrivals.update(rec.get("history_arrival_node_ids") or {})
        self.assertEqual(sorted(arrivals), [0, 1])

    def test_tp_ready_barrier_arms_restore_explicitly(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        builder.completion_gates[VICTIM] = (0, {0: None, 1: None})
        builder.register_pending_history(
            request_id="req_neutral", session_id=VICTIM,
            source_instance_index=0, location="remote_memory")
        builder.emit_prefill_batch(_plan(
            session_id=VICTIM,
            history_transfers=(_load_transfer(),),
            history_action="REMOTE_RESTORE", location="remote_memory"))
        barrier = [n for n in builder.batch["nodes"]
                   if n["rank"] in (0, 1)
                   and "_history_tp_ready_barrier" in n["name"]]
        self.assertEqual(len(barrier), 2)
        edges = builder.batch["parent_edges"]
        restores = {(n["rank"], n["id"])
                    for n in builder.batch["nodes"]
                    if "_target_hbm_write_kv" in n["name"]}
        for node in barrier:
            parent_pairs = {(e["rank"], e["from"]) for e in edges
                            if e["rank"] == node["rank"]
                            and e["to"] == node["id"]}
            self.assertTrue(
                restores & parent_pairs,
                "barrier must explicitly arm the restore completion "
                "(C4) on rank {}".format(node["rank"]))

    def test_join_marker_arms_xfer_recv(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        prefill_plan = _plan(turn_index=0, queue_index=0,
                             request_id="req_joiner",
                             session_id="session_joiner")
        builder.emit_prefill_batch(prefill_plan)
        builder.begin_batch()
        joiner = dict(prefill_plan, decode_evictions=())
        builder._prefill_segment_ends["req_joiner"] = {
            rank: builder.builders[rank].previous_id
            for rank in PREFILL_RANKS
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
                      if "_prefill_to_decode_xfer_recv" in n["name"]}
        self.assertEqual(len(recv_pairs), 2)
        edges = buf_edges
        for node in markers:
            parent_pairs = {(e["rank"], e["from"]) for e in edges
                            if e["rank"] == node["rank"]
                            and e["to"] == node["id"]}
            self.assertTrue(
                recv_pairs & parent_pairs,
                "join marker must explicitly arm the xfer recv (C5) "
                "on rank {}".format(node["rank"]))

    def test_frontier_neutral_rejects_empty_segment(self):
        from generate_wsc_llm_trace import _frontier_neutral
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        with self.assertRaises(RuntimeError):
            _frontier_neutral(builder.builders[0], lambda: None)


if __name__ == "__main__":
    unittest.main()
