#!/usr/bin/env python3
"""test_eviction_side_branch_structure.py -- KV 逐出旁路支链结构钉子测试
（逐出并行，2026-09-13；主方案 §3.1/§3.2）。

给定一批逐出 + 其后的实例迭代列车，断言四条支链化结构不变量：
  (a) 逐出尾节点不是列车体任何节点的祖先（旁路分支悬空，不 join）；
  (b) fork 节点（发射前 frontier）是逐出分支与列车体的公共祖先；
  (c) 主链节点（readiness barrier / 列车首节点）的 parent 不含逐出
      节点 id；
  (d) 触发门（到达 timer gate / drain barrier）仍挂在分支首节点上
      （启动时机不变，去掉的只是"完成阻塞"）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_eviction_side_branch_structure.py（或 pytest）
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

NEW_SESSION = "session_new_0"
NEW_REQUEST = f"{NEW_SESSION}_request_0"
VICTIM_SESSION = "session_old"
VICTIM_REQUEST = f"{VICTIM_SESSION}_request_0"
LAYERS = 4
# instance 0 = ranks (0,1)；instance 1 = ranks (2,3)；边缘 NPU = 1, 3。


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
                            inter_request_interval_ns=5000000),
        ],
    )


def _remote_store_log(session_id, trigger_request_id, *, resident_after=0):
    """decision-log 序列化形态的 remote_store 逐出（S2 特有：plan 里的
    逐出是 dict，经 kv_transfer_from_log 还原后才进入发射）。"""
    return {
        "kind": "remote_store",
        "phase": "admission",
        "reason": "hbm_pressure",
        "session_id": session_id,
        "trigger_request_id": trigger_request_id,
        "source_instance_index": 0,
        "target_instance_index": None,
        "total_bytes": 2048,
        "shards": [
            {
                "source_rank": 0, "target_rank": 1, "edge_rank": 1,
                "bytes": 1024, "noc_path": (0, 1),
                "layer_start": 0, "layer_end": LAYERS,
            },
            {
                "source_rank": 1, "target_rank": 1, "edge_rank": 1,
                "bytes": 1024, "noc_path": (1,),
                "layer_start": 0, "layer_end": LAYERS,
            },
        ],
        "model_layers": LAYERS,
        "layer_start": 0,
        "layer_end": LAYERS,
        "resident_prefix_layers_before": LAYERS,
        "resident_prefix_layers_after": resident_after,
    }


def _admission_plan(**overrides):
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
        "history_evictions": [],
        "prefill_evictions": [],
        "history_tokens_before": 0,
        "prefill_context_tokens": 300,
    }
    plan.update(overrides)
    return plan


def _train_plan(train_id, spans, iterations, drains=(), stage="prefill",
                instance_index=0):
    return {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": stage,
        "joiners": [],
        "pass_spans": list(spans),
        "iterations": iterations,
        "prefill_start_member": None,
        "drain_members": [{"request_id": rid} for rid in drains],
        "exit_members": [],
    }


class _Graph:
    """批内 DAG 视图（parent_edges → 同 rank 祖先闭包）。"""

    def __init__(self, builder):
        self.nodes = {
            (node["rank"], node["id"]): node for node in builder.batch["nodes"]}
        self.parents = {}
        for edge in builder.batch["parent_edges"]:
            self.parents.setdefault((edge["rank"], edge["to"]), set()).add(
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

    def rank_ids(self, rank, name_contains=None):
        return {
            node["id"] for node in self.nodes.values()
            if node["rank"] == rank and (
                name_contains is None or name_contains in node["name"])}


class EvictionSideBranchStructureTest(unittest.TestCase):
    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()
        # 预热主链：各 rank 发射并交付一个前置节点，使逐出支链 fork 自
        # 真实 frontier（fresh builder 的 previous_id=None 无 fork 边）。
        marker = self.builder._mark()
        for rank, builder in self.builder.builders.items():
            builder.set_context("warmup", "prefill", 0)
            builder.comp(f"warmup_rank{rank}", 1, 1)
        self.builder._collect(marker)

    def _emit_admission_with_eviction(self):
        """turn-0 准入 + history 逐出（victim = session_old，源 instance 0
        两 shard 经边缘 rank 1 外迁）。fork 前各 rank frontier 已有节点
        （timer gates）——fork 节点 = timer gate 尾部。"""
        plan = _admission_plan(history_evictions=[
            _remote_store_log(VICTIM_SESSION, NEW_REQUEST)])
        forks = {
            rank: builder.previous_id
            for rank, builder in self.builder.builders.items()}
        self.builder.emit_admission_batch(plan)
        return plan, forks

    def _emit_prefill_train(self):
        return self.builder.emit_iteration_train(_train_plan(
            "batch_train_i0_1", [(128, 128), (128, 256), (44, 300)], 3,
            drains=(NEW_REQUEST,)))

    def test_eviction_tail_not_ancestor_of_train(self):
        """(a)：逐出尾（边缘 mem_store / 源端 ack recv / send）不是列车体
        任何节点的祖先——分支悬空，物理链不阻塞主链。"""
        self._emit_admission_with_eviction()
        self._emit_prefill_train()
        graph = _Graph(self.builder)
        for rank in (0, 1):
            eviction_ids = graph.rank_ids(
                rank, name_contains=VICTIM_SESSION)
            self.assertTrue(eviction_ids, "no eviction nodes emitted")
            train_ids = graph.rank_ids(rank, name_contains="batch_train_i0_1")
            self.assertTrue(train_ids, "no train body nodes emitted")
            for node_id in train_ids:
                key = (rank, node_id)
                overlap = graph.ancestors(key) & {
                    (rank, ev) for ev in eviction_ids}
                self.assertFalse(
                    overlap,
                    f"train node {node_id} on rank {rank} has eviction "
                    f"ancestors: {overlap}")

    def test_fork_is_common_ancestor(self):
        """(b)：fork 节点（发射前该 rank 的 frontier）是逐出分支与列车体
        的公共祖先——两条子链自同一分叉点赛跑（时间重叠的图上表达）。"""
        _plan, forks = self._emit_admission_with_eviction()
        self._emit_prefill_train()
        graph = _Graph(self.builder)
        for rank in (0, 1):
            fork = forks[rank]
            self.assertIsNotNone(fork, "fork frontier must exist")
            eviction_ids = graph.rank_ids(rank, VICTIM_SESSION)
            train_ids = graph.rank_ids(rank, "batch_train_i0_1")
            eviction_first = min(eviction_ids)
            train_first = min(
                node_id for node_id in train_ids
                # 列车体在该 rank 的首节点 = 最小 id 且 fork 在其祖先中
                if (rank, fork) in graph.ancestors((rank, node_id)))
            self.assertIn(
                (rank, fork), graph.ancestors((rank, eviction_first)),
                "fork is not an ancestor of the eviction branch")
            self.assertIn(
                (rank, fork), graph.ancestors((rank, train_first)),
                "fork is not an ancestor of the train body")

    def test_main_chain_parents_exclude_eviction_ids(self):
        """(c)：主链节点（准入 readiness barrier、列车首节点）的 parent
        不含逐出节点 id——restore 之后主链从 fork 继续。"""
        self._emit_admission_with_eviction()
        admission_nodes = list(self.builder.batch["nodes"])
        self._emit_prefill_train()
        graph = _Graph(self.builder)
        for rank in (0, 1):
            eviction_ids = graph.rank_ids(rank, VICTIM_SESSION)
            barrier = next(
                node for node in admission_nodes
                if node["rank"] == rank
                and "prefill_kv_ready_barrier" in node["name"])
            barrier_parents = graph.parents[(rank, barrier["id"])]
            self.assertFalse(
                barrier_parents & {(rank, ev) for ev in eviction_ids},
                "readiness barrier depends on eviction nodes")
            train_first = min(graph.rank_ids(rank, "batch_train_i0_1"))
            train_parents = graph.parents[(rank, train_first)]
            self.assertFalse(
                train_parents & {(rank, ev) for ev in eviction_ids},
                "train first node depends on eviction nodes")

    def test_trigger_gate_still_on_branch_first_node(self):
        """(d)：到达 timer gate 仍挂在逐出分支各 shard 链头节点（启动
        时机不变——rank 0 的 shard0 send 与 rank 1 的直连 mem_store 都
        携带门依赖）；且 turn-0 到达门 armed dep 由主链 readiness
        barrier 携带（支链化次序规则：先 checkpoint 后 arm）。"""
        self._emit_admission_with_eviction()
        graph = _Graph(self.builder)
        for rank in (0, 1):
            timer_gate = next(
                node for node in self.builder.batch["nodes"]
                if node["rank"] == rank
                and node["name"].endswith("_global_arrival_timer_gate"))
            # rank 0：分支首节点 = shard0 的 NoC send（fork + 门双依赖）。
            if rank == 0:
                head = min(graph.rank_ids(rank, VICTIM_SESSION))
                parents = graph.parents[(rank, head)]
                self.assertIn(
                    (rank, timer_gate["id"]), parents,
                    "arrival gate must gate the eviction branch head")
                self.assertGreaterEqual(
                    len(parents), 2,
                    "branch head must carry fork chain + trigger gate")
            # rank 1：shard1（source==edge 直连路径）链头 = edge_store
            # mem_store，门依赖由其携带（arm 被下一个 _new_node 消费）。
            else:
                store_head = next(
                    node_id for node_id in graph.rank_ids(
                        rank, VICTIM_SESSION)
                    if "shard1_edge_store" in graph.nodes[
                        (rank, node_id)]["name"])
                self.assertIn(
                    (rank, timer_gate["id"]),
                    graph.parents[(rank, store_head)],
                    "arrival gate must gate the direct-edge store head")
            # 主链 barrier 的祖先不含逐出节点、但含 timer gate。
            barrier = next(
                node for node in self.builder.batch["nodes"]
                if node["rank"] == rank
                and "prefill_kv_ready_barrier" in node["name"])
            self.assertIn(
                (rank, timer_gate["id"]),
                graph.ancestors((rank, barrier["id"])),
                "arrival gate must stay on the main chain via the barrier")
            eviction_nodes = graph.rank_ids(rank, VICTIM_SESSION)
            self.assertFalse(
                graph.ancestors((rank, barrier["id"])) & {
                    (rank, ev) for ev in eviction_nodes},
                "barrier must not be downstream of the eviction branch")

    def test_decode_eviction_branch_and_joiner_migration_on_main_chain(self):
        """joiner decode 逐出支链化 + prefill_decode_transfer 保持主链：
        逐出尾不是列车体祖先；joiner 的 noc_migrate 迁移节点仍在主链
        （列车首节点的祖先含迁移节点）。"""
        builder = self.builder
        # 先在 instance 0 发一列 prefill 列车建立 drain 块末（seg1 门）。
        self._emit_admission_with_eviction()
        admission_result = self._emit_prefill_train()
        seg1 = builder._block_ends[NEW_REQUEST]["seg1"]
        joiner = {
            "request_id": NEW_REQUEST,
            "session_id": NEW_SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 1,
            "prefill_context_tokens": 300,
            "decode_evictions": [
                _remote_store_log(VICTIM_SESSION, NEW_REQUEST)],
            "prefill_decode_transfer": {
                "kind": "noc_migrate",
                "phase": "prefill_to_decode",
                "reason": "train_test_cross_tp_group",
                "session_id": NEW_SESSION,
                "trigger_request_id": NEW_REQUEST,
                "source_instance_index": 0,
                "target_instance_index": 1,
                "total_bytes": 2048,
                "shards": [
                    {
                        "source_rank": 0, "target_rank": 2,
                        "edge_rank": None, "bytes": 1024,
                        "noc_path": (0, 2),
                        "layer_start": 0, "layer_end": LAYERS,
                    },
                    {
                        "source_rank": 1, "target_rank": 3,
                        "edge_rank": None, "bytes": 1024,
                        "noc_path": (1, 3),
                        "layer_start": 0, "layer_end": LAYERS,
                    },
                ],
                "model_layers": LAYERS,
                "layer_start": 0,
                "layer_end": LAYERS,
                "resident_prefix_layers_before": LAYERS,
                "resident_prefix_layers_after": LAYERS,
            },
            "prefill_drain_block_ends": dict(seg1),
        }
        train = _train_plan(
            "batch_train_i1_1", [(1, 301)], 1, stage="decode",
            instance_index=1)
        train["joiners"] = [joiner]
        train["exit_members"] = [{"request_id": NEW_REQUEST}]
        pre_train_keys = {
            (node["rank"], node["id"])
            for node in builder.batch["nodes"]}
        builder.emit_iteration_train(train)
        graph = _Graph(builder)

        def train_batch_ids(rank, name_contains):
            return {
                node["id"] for node in builder.batch["nodes"]
                if node["rank"] == rank and name_contains in node["name"]
                and (rank, node["id"]) not in pre_train_keys}

        # decode 逐出发生在源 instance 0 的 rank 0/1（边缘 1）；decode
        # 实例 rank 2/3 上是列车体/屏障节点。
        for rank in (0, 1):
            # 分类按 action stage token（逐出 action 名前缀含触发 request
            # 命名空间，按 session id 过滤会同时命中两类节点）。
            eviction_ids = train_batch_ids(rank, "_decode_evictions_action")
            self.assertTrue(eviction_ids, "no decode eviction nodes emitted")
            # rank 0/1 的主链（本批）= joiner 的 prefill→decode noc_migrate
            # 迁移链（恢复类，保持主链）——(a)：不落逐出下游。
            main_chain_ids = train_batch_ids(
                rank, "_prefill_decode_transfer_action")
            self.assertTrue(main_chain_ids,
                            "joiner migration nodes must stay on ranks 0/1")
            for node_id in main_chain_ids:
                overlap = graph.ancestors((rank, node_id)) & {
                    (rank, ev) for ev in eviction_ids}
                self.assertFalse(
                    overlap,
                    f"joiner migration node {node_id} has eviction "
                    f"ancestors: {overlap}")
            # (d) drain barrier 触发门挂在各 shard 链头（rank 0 = shard0
            # 的 send（fork + drain gate 双依赖）；rank 1 = shard1 直连
            # 路径的 edge_store）。
            drain_gate = seg1[rank]
            if rank == 0:
                branch_head = min(eviction_ids)
                self.assertIn(
                    (rank, drain_gate),
                    graph.parents[(rank, branch_head)],
                    "drain barrier gate must stay on the branch head")
            else:
                store_head = next(
                    node_id for node_id in eviction_ids
                    if "shard1_edge_store" in graph.nodes[
                        (rank, node_id)]["name"])
                self.assertIn(
                    (rank, drain_gate),
                    graph.parents[(rank, store_head)],
                    "drain barrier gate must gate the direct-edge store")
        for rank in (2, 3):
            body_ids = train_batch_ids(rank, "batch_train_i1_1")
            self.assertTrue(body_ids)
            eviction_ids = train_batch_ids(rank, VICTIM_SESSION)
            for node_id in body_ids:
                overlap = graph.ancestors((rank, node_id)) & {
                    (rank, ev) for ev in eviction_ids}
                self.assertFalse(
                    overlap, f"train node {node_id} has eviction ancestors")
        # joiner 迁移（noc_migrate，NEW_SESSION 名下）保持主链：decode
        # 实例 rank 2 上列车首节点的祖先含迁移 recv 节点。
        migration_ids = graph.rank_ids(2, NEW_SESSION)
        self.assertTrue(migration_ids)
        train_first_decode = min(graph.rank_ids(2, "batch_train_i1_1"))
        self.assertTrue(
            graph.ancestors((2, train_first_decode))
            & {(2, m) for m in migration_ids},
            "joiner migration must stay on the main chain")

    def test_decode_eviction_dangling_tail_edges(self):
        """(a) 的跨 rank 补充：全图不存在任何以 decode 逐出节点为父的
        主链节点（逐出 id 只允许出现在分支内部与自身 recv 边）。"""
        builder = self.builder
        self.test_decode_eviction_branch_and_joiner_migration_on_main_chain()
        graph = _Graph(builder)
        eviction_keys = {
            key for key in graph.nodes
            if VICTIM_SESSION in graph.nodes[key]["name"]}
        # 准入批逐出 + 列车批逐出都在内：任何非逐出节点的父边都不指向
        # 逐出节点，除非它自身也在同一逐出链上（名字同含 victim id）。
        for (rank, node_id), parents in graph.parents.items():
            if (rank, node_id) in eviction_keys:
                continue
            overlap = parents & eviction_keys
            # 同名 action 前缀的 trigger recv（控制 rank 侧 1B 触发）也
            # 属于逐出链自身；逐出链节点全部携带 victim id，故非逐出
            # 节点不得有任何逐出父节点。
            self.assertFalse(
                overlap,
                f"non-eviction node ({rank},{node_id}) depends on "
                f"eviction nodes {overlap}")


if __name__ == "__main__":
    unittest.main()
