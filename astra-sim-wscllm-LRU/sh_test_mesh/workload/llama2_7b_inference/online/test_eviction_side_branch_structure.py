#!/usr/bin/env python3
"""test_eviction_side_branch_structure.py -- 逐出旁路支链结构钉子测试
(2026-09-13,KV 逐出与 request 推理并行化)。

逐出发射点(history / prefill 增长 / joiner decode 三段)经
_emit_side_branch fork 到旁路分支后,断言主方案 §4.6 结构四条:

  (a) 逐出尾节点(数据 send/边缘 mem_store/ack 对/触发对)不是列车体/
      prefill 主体任何节点的祖先;
  (b) fork 节点(fork 时刻的 rank frontier)是逐出分支首节点与主链
      后续节点的公共祖先;
  (c) 主链节点(就绪屏障/迁移/join 标记/列车体)的直接 parent 不含
      逐出节点 id;
  (d) 触发门仍挂在分支首节点(启动时机不变):history 逐出的到达门
      1B 触发、decode 逐出的 prefill 段末门 1B 触发;无门逐出(prefill
      增长逐出)支链根 = 纯 fork frontier,fork 点的主链到达门 arm 由
      helper 暂存清空、恢复后归还原主链消费者(2026-09-13 统一契约)。

另钉 decode 段 _prefill_segment_ends 门语义保留(块末来源/弹出/
无逐出免登记/fail-closed)与 helper 的 fail-closed / 主链恢复行为。

Run: python3 online/test_eviction_side_branch_structure.py
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
from session_kv_manager import KVTransfer, KVTransferShard  # noqa: E402

# 4x4 mesh(16 rank):边界 = 行0/行3/列0/列3;内部 = {5,6,9,10}。
PREFILL_RANKS = (0, 1)
DECODE_RANKS = (5, 6)
SESSION = "session_struct_0"
VICTIM = "victim_session_struct"
LAYERS = 4

# 逐出支链节点的 action 段名判别(joiner 迁移 transfer-3000 的 send/
# ack 不带逐出段名,不会误配)。
_EVICTION_STAGE_FRAGMENTS = (
    "_history_evictions_", "_prefill_evictions_", "_decode_evictions_")


def _make_config():
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
            SimpleNamespace(session_arrival_time_ns=9000,
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


def _store(source_rank, edge_rank, *, source_instance=1, layer_start=2,
           after=2, session=VICTIM):
    """受害者(VICTIM)的 remote_store 逐出 transfer(链 A/B 由
    source==edge 与否自然选择)。"""
    return KVTransfer(
        kind="remote_store", phase="history",
        reason="watermark_admission_suffix_half",
        session_id=session, trigger_request_id="req",
        source_instance_index=source_instance, target_instance_index=None,
        total_bytes=900,
        shards=(_shard(source_rank=source_rank, target_rank=edge_rank,
                       edge_rank=edge_rank, bytes=900,
                       noc_path=(source_rank, edge_rank),
                       layer_start=layer_start, layer_end=LAYERS),),
        model_layers=LAYERS, layer_start=layer_start, layer_end=LAYERS,
        resident_prefix_layers_before=LAYERS,
        resident_prefix_layers_after=after,
    )


def _plan(*, turn_index=0, queue_index=0, session=SESSION,
          request_id=None, history_evictions=(), prefill_evictions=()):
    return {
        "request_id": request_id or "req_struct_turn{}".format(turn_index),
        "session_id": session,
        "turn_index": turn_index,
        "queue_index": queue_index,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "history_action": None,
        "history_source_instance_index": None,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 64,
        "prefill_context_tokens": 64 + 256,
        "prefill_length": 256,
        "decode_length": 4,
        "history_transfers": (),
        "history_evictions": tuple(history_evictions),
        "prefill_evictions": tuple(prefill_evictions),
    }


# ------------------------------------------------------------- 图查询 --
# 节点 id 仅 per-rank 唯一,跨 rank 比较一律用 (rank, id) 键。


def _nodes_of(nodes, rank=None, fragment=None):
    pool = nodes if rank is None else [n for n in nodes if n["rank"] == rank]
    if fragment is not None:
        pool = [n for n in pool if fragment in n["name"]]
    return pool


def _parents_of(edges, rank, node_id):
    return {
        edge["from"] for edge in edges
        if edge["rank"] == rank and edge["to"] == node_id}


def _ancestors_of(edges, rank, node_id):
    parents = {}
    for edge in edges:
        if edge["rank"] == rank:
            parents.setdefault(edge["to"], set()).add(edge["from"])
    seen = {node_id}
    frontier = [node_id]
    while frontier:
        for parent in parents.get(frontier.pop(), ()):
            if parent not in seen:
                seen.add(parent)
                frontier.append(parent)
    return seen


def _eviction_keys(nodes):
    """全部逐出支链节点 (rank, id)。"""
    return {
        (node["rank"], node["id"]) for node in nodes
        if any(f in node["name"] for f in _EVICTION_STAGE_FRAGMENTS)}


class HistoryEvictionSideBranchTest(unittest.TestCase):
    """history 逐出支链:fork 公共祖先 / 主链不含逐出祖先 / 到达门
    挂分支首节点(1B 触发)。"""

    def setUp(self):
        self.config = _make_config()
        self.builder = GraphBatchBuilder(self.config)
        self.builder.begin_batch()
        # 批 1:普通 prefill(在 prefill 实例 rank 上建立 fork frontier)。
        self.builder.emit_prefill_batch(_plan())
        # decode rank 先发一趟列车建立 frontier。
        warm_train = {
            "train_id": "batch_train_i1_warm",
            "instance_index": 1,
            "joiners": [_plan(request_id="req_struct_warm")],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        self.builder.begin_batch()
        self.builder.emit_iteration_train(warm_train)
        self.forks = {
            rank: self.builder.builders[rank].previous_id
            for rank in range(self.config.npus_count)}
        self.builder.begin_batch()
        plan = _plan(turn_index=0, queue_index=1, session="session_struct_1")
        plan["history_evictions"] = (_store(5, 1),)
        self.builder.emit_prefill_batch(plan)
        self.nodes = list(self.builder.batch["nodes"])
        self.edges = list(self.batch_edges())
        self.tails = _eviction_keys(self.nodes)

    def batch_edges(self):
        return self.builder.batch["parent_edges"]

    def test_eviction_tail_not_ancestor_of_prefill_body(self):
        """(a):逐出支链节点不是就绪屏障/prefill 主体节点的祖先。"""
        self.assertTrue(self.tails)
        body = [node for node in self.nodes
                if "current_prefill" in node["name"]
                or "history_tp_ready_barrier" in node["name"]]
        self.assertTrue(body)
        for node in body:
            self.assertNotIn((node["rank"], node["id"]), self.tails)
            ancestors = _ancestors_of(self.edges, node["rank"], node["id"])
            offenders = {(node["rank"], a) for a in ancestors} & self.tails
            self.assertEqual(offenders, set(),
                             "eviction reachable from {}".format(
                                 node["name"]))

    def test_fork_node_is_common_ancestor(self):
        """(b):分支首节点与主链后续节点同挂 fork 节点(fork 时 frontier)。"""
        # 分支首节点:rank 0 的 1B 触发 send(control = 门源实例 0;
        # 受害源 rank 5 在 decode 实例 → 跨实例 1B 触发)。
        trigger = _nodes_of(self.nodes, 0, "_trigger_to_rank5")
        self.assertEqual(len(trigger), 1)
        trigger_parents = _parents_of(self.edges, 0, trigger[0]["id"])
        self.assertIn(self.forks[0], trigger_parents)
        # 受害源 rank 5 的分支首节点(触发 recv):跨 rank 无直边(桥
        # 拒绝,p2p tag 配对承载触发时序),父边 = {fork_5}(warm 列车使
        # rank 5 已有 frontier)。
        trigger_recv = _nodes_of(self.nodes, 5, "_trigger_from_rank0")
        self.assertEqual(len(trigger_recv), 1)
        recv_parents = _parents_of(self.edges, 5, trigger_recv[0]["id"])
        self.assertEqual(recv_parents, {self.forks[5]})
        # 主链后续节点(rank 0 就绪屏障)同挂 fork_0。
        barrier = _nodes_of(self.nodes, 0, "history_tp_ready_barrier")
        self.assertTrue(barrier)
        barrier_parents = _parents_of(self.edges, 0, barrier[0]["id"])
        self.assertIn(self.forks[0], barrier_parents)

    def test_main_chain_parents_exclude_eviction_ids(self):
        """(c):主链节点直接 parent 不含逐出节点 id。"""
        for node in self.nodes:
            if "current_prefill" not in node["name"] and (
                    "history_tp_ready_barrier" not in node["name"]):
                continue
            self.assertNotIn((node["rank"], node["id"]), self.tails)
            for parent in _parents_of(self.edges, node["rank"], node["id"]):
                self.assertNotIn((node["rank"], parent), self.tails,
                                 "main chain depends on eviction node")

    def test_arrival_gate_on_branch_first_node(self):
        """(d):到达门经 1B 触发挂分支首节点;主链屏障同样等到达门
        (无 history 传输段的主链 arm 语义保持)。"""
        arrival = _nodes_of(self.nodes, 0, "arrival_timer_gate")
        self.assertEqual(len(arrival), 1)
        trigger = _nodes_of(self.nodes, 0, "_trigger_to_rank5")[0]
        self.assertIn(arrival[0]["id"],
                      _parents_of(self.edges, 0, trigger["id"]))
        barrier = _nodes_of(self.nodes, 0, "history_tp_ready_barrier")[0]
        self.assertIn(arrival[0]["id"],
                      _parents_of(self.edges, 0, barrier["id"]))


class PrefillEvictionGateAlignmentTest(unittest.TestCase):
    """无门逐出(prefill 增长逐出)支链:支链根 = 纯 fork frontier(不
    带主链 armed 门);fork 点的主链到达门 arm 由 helper 暂存归还、由
    主链屏障消费;逐出与 prefill 主体并行。"""

    def _builder_with_frontiers(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        builder.emit_prefill_batch(_plan())
        warm_train = {
            "train_id": "batch_train_i1_warm",
            "instance_index": 1,
            "joiners": [_plan(request_id="req_struct_warm")],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        builder.begin_batch()
        builder.emit_iteration_train(warm_train)
        return builder

    def test_same_instance_victim_branch_root_is_pure_frontier(self):
        """统一契约(2026-09-13):无门逐出支链根 = 纯 fork frontier(不带
        armed 门);fork 点的主链 armed 依赖(到达门 arm)由 helper 暂存
        清空、不进分支,恢复后归还主链消费者(就绪屏障)。[原断言钉旧
        变通"支链首节点消费到达门",按统一契约改写。]"""
        builder = self._builder_with_frontiers()
        fork0 = builder.builders[0].previous_id
        builder.begin_batch()
        plan = _plan(turn_index=0, queue_index=1, session="session_struct_1")
        # 受害者在 prefill 实例 rank 0 上(边界 rank,近缘 edge=0 直连)。
        plan["prefill_evictions"] = (_store(0, 0, source_instance=0),)
        builder.emit_prefill_batch(plan)
        nodes = list(builder.batch["nodes"])
        edges = list(builder.batch["parent_edges"])
        arrival = _nodes_of(nodes, 0, "arrival_timer_gate")
        self.assertEqual(len(arrival), 1)
        # 分支首节点(rank 0 直连 mem_store)parent 恰为 fork frontier,
        # 不携带到达门(暂存契约:主链 pending 不进分支)。
        store = _nodes_of(nodes, 0, "_edge_store")
        self.assertEqual(len(store), 1)
        store_parents = _parents_of(edges, 0, store[0]["id"])
        self.assertEqual(store_parents, {fork0})
        # 主链就绪屏障 parent 含到达门(暂存 pending 归还主链消费者)。
        barrier = _nodes_of(nodes, 0, "history_tp_ready_barrier")[0]
        self.assertIn(arrival[0]["id"],
                      _parents_of(edges, 0, barrier["id"]))
        # (a) 逐出节点不是 prefill 主体的祖先(并行化成立)。
        tails = _eviction_keys(nodes)
        self.assertIn((0, store[0]["id"]), tails)
        for node in _nodes_of(nodes, fragment="current_prefill"):
            tail_ids = {a for (r, a) in tails if r == node["rank"]}
            self.assertEqual(
                _ancestors_of(edges, node["rank"], node["id"]) & tail_ids,
                set())

    def test_cross_instance_victim_keeps_frontier_root(self):
        """受害者在他实例:支链根 = 该 rank frontier(纯 frontier,无
        任何 armed 门)。"""
        builder = self._builder_with_frontiers()
        fork5 = builder.builders[5].previous_id
        self.assertIsNotNone(fork5)
        builder.begin_batch()
        plan = _plan(turn_index=0, queue_index=1, session="session_struct_1")
        plan["prefill_evictions"] = (_store(5, 1),)
        builder.emit_prefill_batch(plan)
        edges = list(builder.batch["parent_edges"])
        nodes = list(builder.batch["nodes"])
        send = _nodes_of(nodes, 5, "_send_to_edge1")[0]
        self.assertIn(fork5, _parents_of(edges, 5, send["id"]))


class DecodeEvictionSideBranchTest(unittest.TestCase):
    """joiner decode 逐出支链:prefill 段末门保留、列车体不含逐出祖先、
    joiner 迁移保持主链、_prefill_segment_ends 语义不变。"""

    def setUp(self):
        self.config = _make_config()
        self.builder = GraphBatchBuilder(self.config)
        self.builder.begin_batch()
        prefill_plan = _plan()
        self.builder.emit_prefill_batch(prefill_plan)
        self.segment_ends = dict(
            self.builder._prefill_segment_ends[prefill_plan["request_id"]])
        warm_train = {
            "train_id": "batch_train_i1_warm",
            "instance_index": 1,
            "joiners": [_plan(request_id="req_struct_warm")],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        self.builder.begin_batch()
        self.builder.emit_iteration_train(warm_train)
        self.forks = {
            rank: self.builder.builders[rank].previous_id
            for rank in range(self.config.npus_count)}
        joiner = _plan()
        joiner["decode_evictions"] = (_store(5, 1),)
        self.train_plan = {
            "train_id": "batch_train_i1_struct",
            "instance_index": 1,
            "joiners": [joiner],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        self.builder.begin_batch()
        self.builder.emit_iteration_train(self.train_plan)
        self.nodes = list(self.builder.batch["nodes"])
        self.edges = list(self.builder.batch["parent_edges"])
        self.tails = _eviction_keys(self.nodes)

    def test_segment_end_gate_preserved_on_branch(self):
        """(d):prefill 段末门 1B 触发挂分支首节点;块末登记弹出。"""
        trigger = _nodes_of(self.nodes, 0, "_trigger_to_rank5")
        self.assertEqual(len(trigger), 1)
        trigger_parents = _parents_of(self.edges, 0, trigger[0]["id"])
        self.assertIn(self.segment_ends[0], trigger_parents)
        # 块末登记弹出恰一次(门语义保留)。
        self.assertNotIn(self.train_plan["joiners"][0]["request_id"],
                         self.builder._prefill_segment_ends)

    def test_train_body_has_no_eviction_ancestor(self):
        """(a)+(c):列车体(迁移/join 标记/聚合体/end barrier)不含逐出
        节点祖先,直接 parent 不含逐出 id;迁移保持主链。"""
        self.assertTrue(self.tails)
        body = [node for node in self.nodes
                if any(fragment in node["name"] for fragment in (
                    "prefill_to_decode_kv", "_join_", "batch_train_i1",
                    "end_barrier"))]
        self.assertTrue(body)
        tail_ids_by_rank = {}
        for rank, node_id in self.tails:
            tail_ids_by_rank.setdefault(rank, set()).add(node_id)
        for node in body:
            self.assertNotIn((node["rank"], node["id"]), self.tails)
            self.assertEqual(
                _ancestors_of(self.edges, node["rank"], node["id"])
                & tail_ids_by_rank.get(node["rank"], set()), set(),
                "eviction reachable from {}".format(node["name"]))
        # joiner 迁移首节点(rank 5 迁移 recv)直接挂 fork_5(主链)。
        migration = _nodes_of(self.nodes, 5, "prefill_to_decode_kv")
        self.assertTrue(migration)
        migration_parents = _parents_of(self.edges, 5, migration[0]["id"])
        self.assertIn(self.forks[5], migration_parents)
        self.assertEqual(migration_parents & tail_ids_by_rank.get(5, set()),
                         set())

    def test_joiner_without_evictions_needs_no_segment_end(self):
        builder2 = GraphBatchBuilder(_make_config())
        builder2.begin_batch()
        prefill_plan = _plan()
        builder2.emit_prefill_batch(prefill_plan)
        joiner = _plan()
        joiner["decode_evictions"] = ()
        train_plan = dict(self.train_plan, joiners=[joiner],
                          train_id="batch_train_i1_struct2")
        builder2.begin_batch()
        builder2.emit_iteration_train(train_plan)
        self.assertNotIn(joiner["request_id"],
                         builder2._prefill_segment_ends)

    def test_evictions_without_segment_end_fail_closed(self):
        builder3 = GraphBatchBuilder(_make_config())
        builder3.begin_batch()
        joiner = _plan()
        joiner["decode_evictions"] = (_store(5, 1),)
        train_plan = dict(self.train_plan, joiners=[joiner],
                          train_id="batch_train_i1_struct3")
        builder3.begin_batch()
        with self.assertRaises(RuntimeError):
            builder3.emit_iteration_train(train_plan)


class SideBranchHelperTest(unittest.TestCase):
    """helper 统一契约(2026-09-13):主链 pending 暂存归还 + 发射后泄漏
    fail-closed + 主链恢复行为。[原 test_fork_with_pending_dependency_
    raises 钉旧"fork 点 pending 非空即拒绝"前设,按统一契约改写。]"""

    def test_branch_trigger_leak_fails_closed(self):
        """分支内部 arming 而未被任何节点消费(触发门泄漏)→ fail-closed。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        with self.assertRaises(RuntimeError):
            builder._emit_side_branch(
                lambda: builder.builders[3].arm_dependency(0))

    def test_stashed_main_chain_pending_is_returned(self):
        """fork 点的主链 armed 依赖不进分支,恢复后归还原主链消费者。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        builder.emit_prefill_batch(_plan())
        gate = 12345  # 独立 id 充当主链 armed 门桩(id 仅 per-rank 唯一)
        builder.begin_batch()
        # 主链 armed pending(fork 点合法携带)。
        builder.builders[0].arm_dependency(gate)

        def _branch():
            builder.builders[1].comp("branch_only_rank1", 1, 1)

        marker = builder._mark()
        builder._emit_side_branch(_branch)
        builder._collect(marker)
        # 暂存归还:rank 0 的 pending 仍在(未被分支消费/丢弃)。
        self.assertEqual(builder.builders[0].pending_extra_dependencies,
                         [gate])
        # 主链下一节点消费归还的门依赖。
        marker2 = builder._mark()
        builder.builders[0].comp("main_after_stash", 1, 1)
        main_id = builder.builders[0].previous_id
        builder._collect(marker2)
        self.assertIn(gate,
                      _parents_of(builder.batch["parent_edges"], 0,
                                  main_id))
        self.assertEqual(builder.builders[0].pending_extra_dependencies, [])
        # 分支节点(rank 1)不携带该门(暂存契约:不进分支)。
        branch_node = _nodes_of(builder.batch["nodes"], 1,
                                "branch_only_rank1")
        self.assertEqual(len(branch_node), 1)
        self.assertNotIn(gate, _parents_of(builder.batch["parent_edges"], 1,
                                           branch_node[0]["id"]))

    def test_side_branch_restores_chain(self):
        """正常路径:分支发射后主链 frontier 回滚到 fork 点;主链后续
        节点 parent = fork(非分支节点);分支节点随既有批发射。"""
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        builder.emit_prefill_batch(_plan())
        forks = {rank: builder.builders[rank].previous_id
                 for rank in builder.builders}
        builder.begin_batch()

        def _branch():
            for rank in (0, 1):
                builder.builders[rank].comp("branch_node_rank{}".format(
                    rank), 1, 1)
            branch_ids.update({
                rank: builder.builders[rank].previous_id
                for rank in (0, 1)})

        branch_ids = {}
        marker = builder._mark()
        builder._emit_side_branch(_branch)
        builder._collect(marker)
        self.assertEqual(
            {rank: builder.builders[rank].previous_id
             for rank in builder.builders}, forks)
        names = [node["name"] for node in builder.batch["nodes"]]
        self.assertIn("branch_node_rank0", names)
        # 主链恢复后继续发射:新节点 parent = fork(非分支节点)。
        marker2 = builder._mark()
        builder.builders[0].comp("main_after_branch", 1, 1)
        main_id = builder.builders[0].previous_id
        builder._collect(marker2)
        main_parents = _parents_of(builder.batch["parent_edges"], 0,
                                   main_id)
        self.assertIn(forks[0], main_parents)
        self.assertNotIn(branch_ids[0], main_parents)


if __name__ == "__main__":
    unittest.main()
