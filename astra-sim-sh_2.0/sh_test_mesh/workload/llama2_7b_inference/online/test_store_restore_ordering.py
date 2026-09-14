#!/usr/bin/env python3
"""test_store_restore_ordering.py -- store→restore 前递依赖补边钉子测试
（逐出并行，2026-09-13；主方案 §3.3）。

逐出（remote_store）支链化悬空后，"同会话逐出池写先于其下一轮池读"
的传递性排序失效；本测试压"逐出后立刻再到达"序列，断言回迁
（remote_load）的边缘 mem_load 带指向逐出边缘 mem_store 的依赖：
  1. REMOTE 全量回迁 + 同缘：直接 data 边（store mem_store →
     mem_load，同 rank、跨批次持久 (rank,id) 解析）；
  2. REMOTE 全量回迁 + 跨缘（跨实例回迁可换实例）：1B p2p 中继
     （send 悬挂于 store 边缘、arm 依赖 store mem_store；recv 链入
     回迁链首，mem_load 落其下游）；
  3. 两段式逐出（suffix 半层 + full 整体）两笔 store 均被依赖；
  4. PARTIAL 后缀恢复支链（chain checkpoint 分支内）同样补边，且
     _partial_first_chunk 两段式流水登记不受影响。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_store_restore_ordering.py（或 pytest）
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

from generate_face_trace import PendingHistoryGate  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

NEW_SESSION = "session_new_0"
NEW_REQUEST = f"{NEW_SESSION}_request_0"
VICTIM_SESSION = "session_old"
VICTIM_NEXT = f"{VICTIM_SESSION}_request_1"
LAYERS = 4
# instance 0 = ranks (0,1)（边缘 NPU 1）；instance 1 = ranks (2,3)
# （边缘 NPU 3）。


def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        remote_memory=SimpleNamespace(edge_npus=(1, 3)),
        inference_groups=[
            SimpleNamespace(ranks=(0, 1), pg_name="tp_prefill"),
            SimpleNamespace(ranks=(2, 3), pg_name="tp_decode"),
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
            SimpleNamespace(session_arrival_time_ns=1000,
                            inter_request_interval_ns=None),
            SimpleNamespace(session_arrival_time_ns=None,
                            inter_request_interval_ns=1000),
        ],
    )


def _remote_store_log(session_id, trigger_request_id, layer_start,
                      layer_end, resident_after):
    shards = []
    for source_rank in (0, 1):
        shards.append({
            "source_rank": source_rank, "target_rank": 1, "edge_rank": 1,
            "bytes": 1024, "noc_path": (source_rank, 1),
            "layer_start": layer_start, "layer_end": layer_end,
        })
    return {
        "kind": "remote_store",
        "phase": "admission",
        "reason": "hbm_pressure",
        "session_id": session_id,
        "trigger_request_id": trigger_request_id,
        "source_instance_index": 0,
        "target_instance_index": None,
        "total_bytes": 2048,
        "shards": shards,
        "model_layers": LAYERS,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "resident_prefix_layers_before": LAYERS,
        "resident_prefix_layers_after": resident_after,
    }


def _remote_load_log(session_id, target_instance, edge_rank, targets,
                     layer_start=0, layer_end=LAYERS):
    """REMOTE 全量/后缀回迁：source_rank == edge_rank（校验要求）。"""
    shards = [{
        "source_rank": edge_rank, "target_rank": target_rank,
        "edge_rank": edge_rank, "bytes": 1024,
        "noc_path": (edge_rank, target_rank),
        "layer_start": layer_start, "layer_end": layer_end,
    } for target_rank in targets]
    return {
        "kind": "remote_load",
        "phase": "history",
        "reason": "hbm_restore",
        "session_id": session_id,
        "trigger_request_id": f"{session_id}_request_1",
        "source_instance_index": 0,
        "target_instance_index": target_instance,
        "total_bytes": 1024 * len(targets),
        "shards": shards,
        "model_layers": LAYERS,
        "layer_start": layer_start,
        "layer_end": layer_end,
        "resident_prefix_layers_before": 0,
        "resident_prefix_layers_after": LAYERS,
    }


class _Graph:
    def __init__(self, builder):
        self.nodes = {
            (node["rank"], node["id"]): node
            for node in builder.batch["nodes"]}
        # M1 收集即释放：发射窗口外产生的节点（如测试直发的 interval
        # timer gate）滞留在 builder 缓冲、不在批次累加器——合并进视图。
        for rank, trace_builder in builder.builders.items():
            for node in trace_builder.nodes:
                self.nodes.setdefault((rank, node["id"]), node)
        self.parents = {}
        for edge in builder.batch["parent_edges"]:
            self.parents.setdefault(
                (edge["rank"], edge["to"]), set()).add(
                (edge["rank"], edge["from"]))
        for rank, trace_builder in builder.builders.items():
            for edge in trace_builder.edges:
                self.parents.setdefault(
                    (edge["rank"], edge["to"]), set()).add(
                    (edge["rank"], edge["from"]))

    def ancestors(self, key):
        seen = set()
        frontier = list(self.parents.get(key, ()))
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(self.parents.get(current, ()))
        return seen

    def find(self, rank, name_contains):
        return [node for node in self.nodes.values()
                if node["rank"] == rank and name_contains in node["name"]]


class StoreRestoreOrderingTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()
        marker = self.builder._mark()
        for rank, builder in self.builder.builders.items():
            builder.set_context("warmup", "prefill", 0)
            builder.comp(f"warmup_rank{rank}", 1, 1)
        self.builder._collect(marker)

    def _evict_victim(self, stores):
        """turn-0 准入触发逐出（旁路支链），登记 store 尾部。"""
        plan = {
            "request_id": NEW_REQUEST,
            "session_id": NEW_SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 0,
            "admission_time_ns": 1000,
            "history_location_before": None,
            "history_prefix_transfer": None,
            "history_transfer": None,
            "history_evictions": list(stores),
            "prefill_evictions": [],
            "history_tokens_before": 0,
            "prefill_context_tokens": 300,
        }
        self.builder.emit_admission_batch(plan)
        tails = self.builder.pending_store_tails.get(VICTIM_SESSION)
        self.assertTrue(tails, "store tails were not registered")
        return tails

    def _admit_victim_turn1(self, *, target_instance, history_transfer,
                            location, resident_prefix=0):
        """victim 会话下一 turn 立刻再到达（逐出后立即回迁）：手工预置
        pending_history gate（正常路径由前一 request 的 completion 段
        写入），驱动 admission 的回迁发射统一入口。"""
        gate_ranks = self.builder.config.inference_groups[
            target_instance].ranks
        # interval timer gates 属于前一 completion 批的发射窗口——按
        # marker→发射→collect 进批次累加器（窗口外直发会被下一批的
        # _collect marker 丢弃，M1 收集即释放语义）。
        gate_marker = self.builder._mark()
        timers = tuple(
            self.builder.builders[rank].timer_gate(
                f"victim_turn1_rank{rank}_interval_gate", 1000)
            for rank in gate_ranks)
        self.builder._collect(gate_marker)
        self.builder.pending_history[VICTIM_NEXT] = PendingHistoryGate(
            source_instance_index=target_instance,
            timer_gates=timers,
            location=location,
        )
        plan = {
            "request_id": VICTIM_NEXT,
            "session_id": VICTIM_SESSION,
            "turn_index": 1,
            "queue_index": 1,
            "prefill_instance_index": target_instance,
            "decode_instance_index": target_instance,
            "admission_time_ns": 1000,
            "history_location_before_location": location,
            "history_location_before_instance_index": 0,
            "history_resident_prefix_layers": resident_prefix,
            "history_prefix_transfer": None,
            "history_transfer": history_transfer,
            "history_evictions": [],
            "prefill_evictions": [],
            "history_tokens_before": 128,
            "prefill_context_tokens": 428,
        }
        self.builder.emit_admission_batch(plan)

    def _mem_load_nodes(self, graph, edge_rank):
        nodes = graph.find(edge_rank, "_shard")
        loads = [n for n in nodes if n["name"].endswith("_remote_load")]
        self.assertTrue(loads, "no restore mem_load emitted on the edge")
        return loads

    def test_remote_full_restore_same_edge_direct_edge(self):
        """1：REMOTE 全量回迁 + 同缘——mem_load 直接依赖 store 的
        mem_store（同 rank data 边）。"""
        tails = self._evict_victim([
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST, 0, LAYERS, 0)])
        self._admit_victim_turn1(
            target_instance=0,
            history_transfer=_remote_load_log(
                VICTIM_SESSION, 0, 1, (0, 1)),
            location="remote_memory")
        graph = _Graph(self.builder)
        store_ids = {store_id for _edge, store_id, _ack in tails}
        self.assertEqual(len(store_ids), 2)  # 每 shard 一条
        for load in self._mem_load_nodes(graph, 1):
            parents = graph.parents[(1, load["id"])]
            self.assertTrue(
                parents & {(1, s) for s in store_ids},
                f"mem_load {load['id']} misses a direct store dependency")

    def test_two_stage_stores_both_dependent(self):
        """3：两段式逐出（suffix 半层 + full 整体）两笔 store 的全部
        mem_store 尾部都被回迁 mem_load 依赖（restore 读全区间）。"""
        tails = self._evict_victim([
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST, 2, LAYERS, 2),
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST, 0, 2, 0),
        ])
        self._admit_victim_turn1(
            target_instance=0,
            history_transfer=_remote_load_log(
                VICTIM_SESSION, 0, 1, (0, 1)),
            location="remote_memory")
        graph = _Graph(self.builder)
        store_ids = {store_id for _edge, store_id, _ack in tails}
        self.assertEqual(len(store_ids), 4)  # 两笔 store × 两 shard
        for load in self._mem_load_nodes(graph, 1):
            parents = graph.parents[(1, load["id"])]
            covered = {s for (rank, s) in parents if rank == 1}
            self.assertTrue(
                store_ids <= covered,
                f"mem_load {load['id']} misses store tails: "
                f"{store_ids - covered}")

    def test_remote_full_restore_cross_edge_relay(self):
        """2：REMOTE 全量回迁 + 跨缘（跨实例回迁）——1B p2p 中继：send
        悬挂于 store 边缘（arm 依赖 store mem_store），recv 链入回迁
        边缘，mem_load 落其下游。"""
        tails = self._evict_victim([
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST, 0, LAYERS, 0)])
        self._admit_victim_turn1(
            target_instance=1,
            history_transfer=_remote_load_log(
                VICTIM_SESSION, 1, 3, (2, 3)),
            location="remote_memory")
        graph = _Graph(self.builder)
        store_edge, store_id, _ack = tails[0]
        self.assertEqual(store_edge, 1)
        # 中继 send：store 边缘 rank 1 上的悬空节点，父含 store mem_store。
        relay_sends = graph.find(1, "_storetail0_relay_to_rank3")
        self.assertEqual(len(relay_sends), 1)
        send_parents = graph.parents[(1, relay_sends[0]["id"])]
        self.assertIn(
            (1, store_id), send_parents,
            "relay send must depend on the store mem_store node")
        # 中继 recv：回迁边缘 rank 3，mem_load 落其下游。
        relay_recvs = graph.find(3, "_storetail0_relay_from_rank1")
        self.assertEqual(len(relay_recvs), 1)
        recv_key = (3, relay_recvs[0]["id"])
        loads = [n for n in graph.find(3, "_remote_load")
                 if n["name"].endswith("_remote_load")]
        self.assertTrue(loads)
        for load in loads:
            self.assertIn(
                recv_key, graph.ancestors((3, load["id"])),
                "mem_load must be downstream of the cross-edge relay recv")
        # 跨缘不得出现跨 rank 直边（桥协议）：全部 parent 边两端同 rank
        # （节点 id 是 per-rank 计数器，from 端必须在该 rank 的节点表内）。
        for edge in self.builder.batch["parent_edges"]:
            from_node = graph.nodes.get((edge["rank"], edge["from"]))
            self.assertIsNotNone(
                from_node,
                f"edge source ({edge['rank']},{edge['from']}) is not a "
                "known node on that rank (cross-rank direct edge)")
        # 补边只能经中继覆盖跨缘组合：store 节点在 rank 1，其依赖承载于
        # rank 1 的 relay send；rank 3 侧由 relay recv 传递（上面已断言
        # mem_load 落 recv 下游）。

    def test_partial_suffix_restore_branch_covered(self):
        """4：PARTIAL 后缀恢复支链（chain checkpoint 分支）同样补边，
        且 _partial_first_chunk 两段式流水登记不受影响。"""
        self._evict_victim([
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST, 2, LAYERS, 2)])
        tails = self.builder.pending_store_tails[VICTIM_SESSION]
        store_ids = {store_id for _e, store_id, _a in tails}
        self._admit_victim_turn1(
            target_instance=0,
            history_transfer=_remote_load_log(
                VICTIM_SESSION, 0, 1, (0, 1), layer_start=2, layer_end=4),
            location="partial_hbm_remote",
            resident_prefix=2)
        graph = _Graph(self.builder)
        for load in self._mem_load_nodes(graph, 1):
            parents = graph.parents[(1, load["id"])]
            self.assertTrue(
                parents & {(1, s) for s in store_ids},
                "partial suffix mem_load misses the store dependency")
        self.assertIn(
            VICTIM_NEXT, self.builder._partial_first_chunk,
            "partial first-chunk pipelining ledger must stay registered")
        self.assertEqual(
            self.builder._partial_first_chunk[VICTIM_NEXT]["suffix_start"],
            2)

    def test_terminal_session_clears_registry(self):
        """终态会话清登记：completion 段 following_plan is None 分支
        （= 调度器 retire_terminal_session 决策边界）。"""
        self._evict_victim([
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST, 0, LAYERS, 0)])
        self.assertIn(VICTIM_SESSION, self.builder.pending_store_tails)
        plan = {
            "request_id": VICTIM_NEXT,
            "session_id": VICTIM_SESSION,
            "turn_index": 1,
            "queue_index": 1,
            "decode_instance_index": 0,
            "history_evictions": [],
            "completion_evictions": [],
        }
        # 预置 seg2 块末（completion 触发门来源）。
        self.builder._block_ends[VICTIM_NEXT] = {
            "seg2": {0: 5, 1: 5}}
        self.builder.begin_batch()
        self.builder.emit_completion_batch(plan, None)
        self.assertNotIn(
            VICTIM_SESSION, self.builder.pending_store_tails,
            "terminal session must clear its store-tail registry")


if __name__ == "__main__":
    unittest.main()
