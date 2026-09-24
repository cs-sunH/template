#!/usr/bin/env python3
"""test_eviction_side_branch_structure.py -- KV 逐出旁路支链结构断言
（2026-09-13，KV 逐出与 request 推理并行化）。

给定"先行列车 → 带逐出的准入动作 → 后续列车"的发射序列，钉住旁路
支链化的结构语义（主方案 §3.1/§3.2）：

  (a) 逐出节点不是列车体任何节点的祖先（逐出完成不阻塞推理计算）；
  (b) fork 节点（先行列车 end barrier，逐出发射前各 rank 的 frontier）
      是逐出分支与列车体的公共祖先；
  (c) 主链节点（readiness barrier / 列车体首节点）的 parent 不含任何
      逐出节点 id；
  (d) 触发门（turn-0 arrival timer gate）仍挂在分支首节点（逐出开始
      时刻不变）；
  (e) fork 时既有主链 armed 依赖（turn-0 arrival arm）不被分支消费、
      原样留给主链 readiness barrier（stash-and-clear 语义）；
  (f) remote_store 逐出支链尾部登记进 pending_store_tails（逐 shard）。

对照：支链化前 (a)/(c) 不成立（逐出链横在主链上，是其后一切节点的
祖先），本测试即钉住改造语义。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_eviction_side_branch_structure.py
"""
import os
import sys
import unittest
from dataclasses import replace
from types import SimpleNamespace

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _p in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from face_scheduler import KVTransfer, KVTransferShard  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402

PREFILL_RANKS = (0, 1)
DECODE_RANKS = (2, 3)
EDGE_RANKS = (2, 3)
SESSION_VICTIM = "session_victim"
REQUEST_TRIG = "session_new_request_0"


def _make_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=PREFILL_RANKS, pg_name="tp_prefill"),
            SimpleNamespace(ranks=DECODE_RANKS, pg_name="tp_decode"),
        ],
        layers=2,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        remote_memory=SimpleNamespace(edge_npus=EDGE_RANKS),
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=None,
            ),
        ],
    )


def _remote_store_victim():
    """受害者会话的整会话外迁（admission 逐出，2 shard 到边缘 2/3）。"""
    return KVTransfer(
        kind="remote_store",
        phase="admission",
        reason="admission_capacity_full_fallback",
        session_id=SESSION_VICTIM,
        trigger_request_id=REQUEST_TRIG,
        source_instance_index=0,
        target_instance_index=None,
        total_bytes=2000,
        shards=(
            KVTransferShard(
                source_rank=0, target_rank=2, edge_rank=2, bytes=1000,
                noc_path=(), layer_start=0, layer_end=2),
            KVTransferShard(
                source_rank=1, target_rank=3, edge_rank=3, bytes=1000,
                noc_path=(), layer_start=0, layer_end=2),
        ),
        model_layers=2,
        layer_start=0,
        layer_end=2,
        resident_prefix_layers_before=2,
        resident_prefix_layers_after=0,
    )


def _admission_plan():
    return {
        "request_id": REQUEST_TRIG,
        "session_id": "session_new",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "admission_time_ns": 1000,
        "history_location_before": None,
        "history_transfer": None,
        "history_evictions": [_remote_store_victim()],
        "prefill_evictions": [_remote_store_victim()],
        "history_tokens_before": 0,
        "prefill_context_tokens": 300,
    }


def _train_plan(train_id, spans, instance_index=0):
    return {
        "train_id": train_id,
        "instance_index": instance_index,
        "stage": "prefill",
        "joiners": [],
        "pass_spans": list(spans),
        "iterations": 1,
        "prefill_start_member": None,
        "first_chunk_member": None,
        "drain_members": [],
        "exit_members": [],
    }


def _decode_joiner_plan():
    """带 decode_evictions 的 joiner（受害者会话 shard 落在 decode 实例
    rank 2/3，边缘同 rank；pd_transfer 用 local_hit 零节点）。"""
    return {
        "request_id": "session_new_request_0",
        "session_id": "session_new",
        "turn_index": 0,
        "queue_index": 0,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "prefill_context_tokens": 300,
        "decode_evictions": [
            KVTransfer(
                kind="remote_store",
                phase="decode",
                reason="decode_capacity_full_fallback",
                session_id=SESSION_VICTIM,
                trigger_request_id="session_new_request_0",
                source_instance_index=1,
                target_instance_index=None,
                total_bytes=2000,
                shards=(
                    KVTransferShard(
                        source_rank=2, target_rank=2, edge_rank=2,
                        bytes=1000, noc_path=(), layer_start=0,
                        layer_end=2),
                    KVTransferShard(
                        source_rank=3, target_rank=3, edge_rank=3,
                        bytes=1000, noc_path=(), layer_start=0,
                        layer_end=2),
                ),
                model_layers=2,
                layer_start=0,
                layer_end=2,
                resident_prefix_layers_before=2,
                resident_prefix_layers_after=0,
            ),
        ],
        "prefill_decode_transfer": _local_hit("session_new_request_0"),
        "prefill_drain_block_ends": {rank: 0 for rank in PREFILL_RANKS},
    }


def _local_hit(request_id):
    return KVTransfer(
        kind="local_hit",
        phase="prefill_to_decode",
        reason="train_test_same_tp_group",
        session_id="session_new",
        trigger_request_id=request_id,
        source_instance_index=0,
        target_instance_index=1,
        total_bytes=0,
        shards=(),
        model_layers=2,
        layer_start=0,
        layer_end=2,
        resident_prefix_layers_before=2,
        resident_prefix_layers_after=2,
    )


class EvictionSideBranchStructureTest(unittest.TestCase):
    def setUp(self):
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def _nodes(self):
        return self.builder.batch["nodes"]

    def _edges(self):
        return self.builder.batch["parent_edges"]

    def _parents_by_rank(self):
        parents = {}
        for edge in self._edges():
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        return parents

    def _ancestors(self, rank, node_id):
        """同 rank 依赖闭包（跨 rank 依赖在 S3 只经 p2p 对，不进边表）；
        返回 (rank, id) 二元组集合，与 _eviction_node_ids 同口径。"""
        parents = self._parents_by_rank()
        seen = set()
        frontier = [node_id]
        while frontier:
            current = frontier.pop()
            for parent in parents.get((rank, current), ()):
                if parent not in seen:
                    seen.add(parent)
                    frontier.append(parent)
        return {(rank, node) for node in seen}

    def _eviction_node_ids(self, stage):
        return {
            (node["rank"], node["id"]) for node in self._nodes()
            if stage in node["name"]}

    def _emit_scenario(self):
        """先行列车（frontier）→ 准入（history+prefill 逐出）→ 后续列车。"""
        self.builder.emit_iteration_train(
            _train_plan("batch_train_i0_1", [(1, 101), (1, 102)]))
        fork = {
            rank: self.builder.builders[rank].previous_id
            for rank in PREFILL_RANKS}
        self.builder.emit_admission_batch(_admission_plan())
        self.builder.emit_iteration_train(
            _train_plan("batch_train_i0_2", [(1, 201), (1, 202)]))
        return fork

    # ------------------------------------------------------------- 断言 --

    def test_eviction_not_ancestor_of_train_body(self):
        """(a)+(c)：逐出节点不是列车体/readiness barrier 的祖先。"""
        self._emit_scenario()
        train_body = {
            node["id"] for node in self._nodes()
            if node["request_id"] == "batch_train_i0_2"}
        self.assertTrue(train_body)
        barrier = {
            (node["rank"], node["id"]) for node in self._nodes()
            if node["name"].endswith("_prefill_kv_ready_barrier")}
        self.assertEqual(len(barrier), len(PREFILL_RANKS))
        for stage in ("history_evictions", "prefill_evictions"):
            evictions = self._eviction_node_ids(stage)
            self.assertTrue(evictions)
            for rank in PREFILL_RANKS:
                main_chain_rank = {
                    node["id"] for node in self._nodes()
                    if node["request_id"] == "batch_train_i0_2"
                    and node["rank"] == rank} | {
                    node["id"] for node in self._nodes()
                    if (node["rank"], node["id"]) in barrier
                    and node["rank"] == rank}
                for node_id in main_chain_rank:
                    self.assertFalse(
                        self._ancestors(rank, node_id) & evictions,
                        f"{stage} node leaked into ancestors of main "
                        f"chain node {rank}/{node_id}")

    def test_fork_node_is_common_ancestor(self):
        """(b)：fork 节点（先行列车 end barrier）是分支与主链公共祖先。"""
        fork = self._emit_scenario()
        for stage in ("history_evictions", "prefill_evictions"):
            for rank in PREFILL_RANKS:
                branch_head = min(
                    node["id"] for node in self._nodes()
                    if stage in node["name"] and node["rank"] == rank)
                self.assertIn(
                    (rank, fork[rank]),
                    self._ancestors(rank, branch_head),
                    "fork node is not an ancestor of the branch head")
        for rank in PREFILL_RANKS:
            body_head = min(
                node["id"] for node in self._nodes()
                if node["request_id"] == "batch_train_i0_2"
                and node["rank"] == rank)
            self.assertIn(
                (rank, fork[rank]), self._ancestors(rank, body_head),
                "train body does not descend from the fork node")
            barrier = next(
                node["id"] for node in self._nodes()
                if node["name"].endswith("_prefill_kv_ready_barrier")
                and node["rank"] == rank)
            self.assertIn(
                fork[rank],
                self._parents_by_rank().get((rank, barrier), ()),
                "readiness barrier does not chain from the fork node")

    def test_trigger_gate_on_branch_head_and_stash_semantics(self):
        """(d)+(e)：触发门挂分支首节点；turn-0 arrival arm 不被分支消费、
        原样留给 readiness barrier。"""
        self._emit_scenario()
        gates = {
            node["rank"]: node["id"] for node in self._nodes()
            if node["name"].endswith("_global_arrival_timer_gate")}
        self.assertEqual(sorted(gates), sorted(PREFILL_RANKS))
        parents = self._parents_by_rank()
        for stage in ("history_evictions", "prefill_evictions"):
            for rank in PREFILL_RANKS:
                branch_nodes = sorted(
                    node["id"] for node in self._nodes()
                    if stage in node["name"] and node["rank"] == rank)
                self.assertTrue(branch_nodes)
                head_parents = set(parents.get((rank, branch_nodes[0]), ()))
                if stage == "history_evictions":
                    self.assertIn(
                        gates[rank], head_parents,
                        "trigger gate missing on history-eviction branch "
                        "head")
                else:
                    self.assertNotIn(
                        gates[rank], head_parents,
                        "main-chain armed gate leaked into the prefill-"
                        "eviction branch (stash-and-clean violated)")
        for rank in PREFILL_RANKS:
            barrier = next(
                node["id"] for node in self._nodes()
                if node["name"].endswith("_prefill_kv_ready_barrier")
                and node["rank"] == rank)
            self.assertIn(
                gates[rank],
                parents.get((rank, barrier), ()),
                "stashed arrival arm was not returned to the readiness "
                "barrier on the main chain")

    def test_store_tails_registered_per_shard(self):
        """(f)：remote_store 逐出支链尾部逐 shard 登记。"""
        self._emit_scenario()
        tails = self.builder.pending_store_tails.get(SESSION_VICTIM)
        self.assertIsNotNone(tails)
        self.assertEqual(len(tails), 4)  # history + prefill 各 2 shard
        store_nodes = {
            (node["rank"], node["id"]) for node in self._nodes()
            if node["name"].endswith("_remote_store")
            or node["name"].endswith("_edge_store")}
        self.assertEqual(len(store_nodes), 4)
        # §4.2.4（2026-09-17）：条目扩层区间 (edge_rank, store_id,
        # ack_id, layer_start, layer_end)——逐出池写恒为后缀形 [k, L)。
        for edge_rank, store_id, _ack_id, _ls, _le in tails:
            self.assertIn((edge_rank, store_id), store_nodes)
        self.assertEqual(
            sorted((t[3], t[4]) for t in tails),
            [(0, 2)] * 4)
        self.assertEqual(
            sorted(t[0] for t in tails), [2, 2, 3, 3])

    def test_admission_watch_members_follow_emitted_nodes(self):
        """有物理节点的逐出支链返回独立尾标记成员。"""
        result = self.builder.emit_admission_batch(_admission_plan())
        watches = result["eviction_watches"]
        self.assertEqual(
            {watch["request_id"] for watch in watches},
            {
                f"batch_train_evict_{REQUEST_TRIG}_admission_history",
                f"batch_train_evict_{REQUEST_TRIG}_admission_prefill",
            },
        )
        for watch in watches:
            self.assertEqual(watch["owner_request_id"], REQUEST_TRIG)
            self.assertTrue(watch["members"])
            self.assertEqual(set(watch["members"]), set(self.builder.builders))
            for rank, node_id in watch["members"].items():
                marker = next(
                    node for node in self._nodes()
                    if node["rank"] == rank and node["id"] == node_id)
                self.assertIn("eviction_done_rank", marker["name"])

    def test_zero_byte_remote_store_is_rejected_without_watch(self):
        """非法零字节 shard 在 emitter 校验处 fail-closed，不能留下 watch。"""
        positive = _remote_store_victim()
        transfer = replace(
            positive,
            total_bytes=0,
            shards=(replace(positive.shards[0], bytes=0),),
        )
        plan = _admission_plan()
        plan["history_evictions"] = [transfer]
        plan["prefill_evictions"] = []
        with self.assertRaisesRegex(ValueError, "positive bytes"):
            self.builder.emit_admission_batch(plan)
        self.assertFalse(any(
            "eviction_done_rank" in node["name"]
            for builder in self.builder.builders.values()
            for node in builder.nodes))
        self.assertFalse(self.builder.pending_store_tails)

    def test_remote_store_with_no_shards_creates_no_watch_or_tail(self):
        """空 shard 列表无图节点，因此不得返回/登记空成员 watch。"""
        transfer = replace(
            _remote_store_victim(), total_bytes=0, shards=())
        plan = _admission_plan()
        plan["history_evictions"] = [transfer]
        plan["prefill_evictions"] = []
        result = self.builder.emit_admission_batch(plan)
        self.assertEqual(result["eviction_watches"], [])
        self.assertFalse(any(
            "history_evictions" in node["name"]
            or "eviction_done_rank" in node["name"]
            for builder in self.builder.builders.values()
            for node in builder.nodes))
        self.assertFalse(self.builder.pending_store_tails)

    def test_no_eviction_batch_invented(self):
        """逐出节点仍随准入批发射（无独立逐出批）：同一 batch 累加器内。"""
        self._emit_scenario()
        stages = {
            node["stage"] for node in self._nodes()
            if "history_evictions" in node["name"]
            or "prefill_evictions" in node["name"]}
        self.assertEqual(stages, {"prefill"})

    def test_naming_avoids_router_anchors(self):
        """新增节点命名避开 batch_train_ / first_token 唤醒路由锚点。"""
        self._emit_scenario()
        for node in self._nodes():
            if "evictions" in node["name"]:
                self.assertFalse(node["name"].startswith("batch_train_"))
                self.assertNotIn("first_token", node["name"])

    def test_decode_eviction_branch_and_pd_transfer_on_main_chain(self):
        """发射点 3：joiner decode 逐出走旁路支链（drain 门挂分支首节点），
        readiness barrier 与紧随的 prefill_decode_transfer 保持主链。"""
        joiner = _decode_joiner_plan()
        drain_gates = dict(joiner["prefill_drain_block_ends"])
        for rank in DECODE_RANKS:
            self.builder.builders[rank].comp(
                f"decode_frontier_rank{rank}", 1, 1)
        decode_fork = {
            rank: self.builder.builders[rank].previous_id
            for rank in DECODE_RANKS}
        train_plan = _train_plan("batch_train_i1_1", [(1, 101), (1, 102)])
        train_plan["instance_index"] = 1
        train_plan["stage"] = "decode"
        train_plan["joiners"] = [joiner]
        result = self.builder.emit_iteration_train(train_plan)
        eviction_watches = result["eviction_watches"]
        self.assertEqual(len(eviction_watches), 1)
        self.assertEqual(
            eviction_watches[0]["owner_request_id"], joiner["request_id"])
        self.assertTrue(eviction_watches[0]["members"])
        # decode 逐出支链：分支首节点祖先含 fork 节点（decode rank 的
        # 既有 frontier），不阻塞其后主链（readiness barrier/列车体）。
        for rank in DECODE_RANKS:
            branch = sorted(
                node["id"] for node in self._nodes()
                if "decode_evictions" in node["name"]
                and node["rank"] == rank)
            self.assertTrue(branch)
            self.assertIn(
                (rank, decode_fork[rank]),
                self._ancestors(rank, branch[0]))
        # 主链核验：readiness barrier 与列车体不含逐出祖先。
        evictions = self._eviction_node_ids("decode_evictions")
        self.assertTrue(evictions)
        for node in self._nodes():
            if (node["request_id"] == "batch_train_i1_1"
                    or node["name"].endswith("_decode_kv_ready_barrier")):
                self.assertFalse(
                    self._ancestors(node["rank"], node["id"]) & evictions,
                    f"decode-eviction node leaked into ancestors of main "
                    f"chain node {node['rank']}/{node['id']}")
        # drain 触发门（plan 的 block_ends 占位 id）挂在 control rank
        # （prefill 实例）的 trigger 1B 发送上——分支首节点的触发门锚点。
        trigger_sends = [
            node for node in self._nodes()
            if "decode_evictions" in node["name"]
            and "trigger_to_rank" in node["name"]]
        self.assertTrue(trigger_sends)
        for node in trigger_sends:
            self.assertIn(
                (node["rank"], drain_gates[node["rank"]]),
                self._ancestors(node["rank"], node["id"]))
        # store 尾部已登记（REMOTE 全量逐出）。
        self.assertEqual(
            len(self.builder.pending_store_tails.get(SESSION_VICTIM, ())), 2)


if __name__ == "__main__":
    unittest.main()
