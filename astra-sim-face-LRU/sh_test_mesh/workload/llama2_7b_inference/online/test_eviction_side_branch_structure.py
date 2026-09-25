#!/usr/bin/env python3
"""test_eviction_side_branch_structure.py -- B4(2026-09-13)逐出旁路支链
结构钉子测试(主方案《逐出与request执行的并行修改方案》§4.6)。

四处逐出发射点(history_evictions / prefill_evictions / decode_evictions /
blocked_admission_evictions)经 _emit_side_branch fork 到旁路分支后钉住:

  (a) 逐出尾节点不是列车体/主链任何节点的祖先(逐出不再 head-of-line
      阻塞其后计算);
  (b) fork 节点(发射前 frontier)是逐出分支与主链 continuation 的公共
      祖先(两链自同一点赛跑 = 时间重叠);
  (c) 主链节点的直接 parent 不含逐出节点 id(含 blocked 路径);
  (d) 触发门仍挂在分支首节点(启动时机不变——门只对齐逐出的开始时刻);
  (e) blocked 路径 context stamping 仍为 (request, prefill, 0) 且节点
      随本批收集交付(_mark/_collect 留在包裹外,"无 watch 纯 mem 批
      不驱动决策交付"语义不变);
  (f) turn-0 prefill 增长逐出走 pending 暂存路径时,主链 readiness 屏障
      仍消费到达门(暂存-回填不吞门)。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_eviction_side_branch_structure.py(或 pytest)
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

SESSION = "session_branch_0"
REQUEST_A = f"{SESSION}_request_0"
REQUEST_B = f"{SESSION}_request_1"

# 3x3 mesh:rank 4 为内部,边缘集合 = 8 个边界 rank(与计费钉子测试同构);
# instance_0 = (0, 4) 承载 prefill,instance_1 = (2, 3) 承载 decode(等 TP
# 度——joiner 3000 迁移的 FACE 直连配对校验要求)。
EVICT_STAGE_MARKS = (
    "_history_evictions_",
    "_prefill_evictions_",
    "_decode_evictions_",
    "_blocked_admission_evictions_",
)


def _make_config():
    return SimpleNamespace(
        npus_count=9,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(name="instance_0", ranks=(0, 4),
                            pg_name="tp0"),
            SimpleNamespace(name="instance_1", ranks=(2, 3),
                            pg_name="tp1"),
        ],
        prefill_chunk_size=128,
        layers=4,
        hidden_size=64,
        ffn_size=128,
        vocab_size=256,
        bytes_per_elem=2,
        num_heads=8,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=1000,
                inter_request_interval_ns=None,
            ),
            SimpleNamespace(
                session_arrival_time_ns=None,
                inter_request_interval_ns=1000,
            ),
        ],
        model=SimpleNamespace(
            layers=4, hidden_size=64, num_heads=8, bytes_per_elem=2),
        hardware=SimpleNamespace(
            mesh_rows=3, mesh_cols=3, npus_count=9),
    )


def _store_transfer(session_id=SESSION, trigger=REQUEST_A):
    """Session 级 Tiered-LRU:整体 store fixture(全量层域 [0,4)、
    before=4/after=0)。"""
    from session_kv_manager import KVTransfer, KVTransferShard
    shards = (
        KVTransferShard(source_rank=0, target_rank=0, edge_rank=0,
                        bytes=512, noc_path=(0,), layer_start=0, layer_end=4),
        KVTransferShard(source_rank=4, target_rank=1, edge_rank=1,
                        bytes=512, noc_path=(4, 1), layer_start=0, layer_end=4),
    )
    return KVTransfer(
        kind="remote_store", phase="history",
        reason="evict_history_admission_full:layers0-4",
        session_id=session_id, trigger_request_id=trigger,
        source_instance_index=0, target_instance_index=None,
        total_bytes=1024, shards=shards, model_layers=4,
        layer_start=0, layer_end=4,
        resident_prefix_layers_before=4,
        resident_prefix_layers_after=0,
    )


def _admission_plan(request_id, turn, *, extra=None):
    plan = {
        "request_id": request_id,
        "session_id": SESSION,
        "turn_index": turn,
        "queue_index": turn,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "history_action": None,
        "history_source_instance_index": None,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_context_tokens": 300,
        "prefill_length": 300,
    }
    plan.update(extra or {})
    return plan


def _seed_frontier(builder, tag):
    """每 rank 预置一个主链节点,返回 {rank: fork 节点 id}。"""
    for trace_builder in builder.builders.values():
        trace_builder.set_context("seed_{}".format(tag), "prefill", 0)
        trace_builder.comp("seed_frontier_{}".format(tag), 1, 1)
    return {rank: trace_builder.previous_id
            for rank, trace_builder in builder.builders.items()}


def _rank_nodes(builder, rank):
    return [node for node in builder.batch["nodes"]
            if node["rank"] == rank]


def _rank_edges(builder, rank):
    return [edge for edge in builder.batch["parent_edges"]
            if edge["rank"] == rank]


def _ancestors(builder, rank, node_id):
    """同 rank parent 边祖先闭包(含自身)。"""
    closure = {node_id}
    pending = [node_id]
    by_target = {}
    for edge in _rank_edges(builder, rank):
        by_target.setdefault(edge["to"], []).append(edge["from"])
    while pending:
        current = pending.pop()
        for parent in by_target.get(current, ()):
            if parent not in closure:
                closure.add(parent)
                pending.append(parent)
    return closure


def _eviction_node_ids(builder, rank):
    return {
        node["id"] for node in _rank_nodes(builder, rank)
        if any(mark in node["name"] for mark in EVICT_STAGE_MARKS)}


class EvictionSideBranchStructureTest(unittest.TestCase):
    """(a)-(f)结构断言。"""

    def setUp(self) -> None:
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def _assert_branch_off_main_chain(self, fork_ids, main_probe):
        """(a)(b)(c) 通用核验:main_probe = {rank: 主链 continuation 节点 id}。"""
        builder = self.builder
        for rank, probe_id in main_probe.items():
            eviction_ids = _eviction_node_ids(builder, rank)
            self.assertTrue(
                eviction_ids, f"rank {rank} emitted no eviction nodes")
            closure = _ancestors(builder, rank, probe_id)
            # (a) 逐出节点不是主链 continuation 的祖先。
            self.assertEqual(
                closure & eviction_ids, set(),
                f"rank {rank}: eviction nodes are ancestors of the "
                f"main-chain continuation {probe_id}")
            # (b) fork 节点是两链公共祖先。
            fork_id = fork_ids.get(rank)
            if fork_id is not None:
                self.assertIn(fork_id, closure)
                first_eviction = min(eviction_ids)
                self.assertIn(
                    fork_id, _ancestors(builder, rank, first_eviction),
                    f"rank {rank}: fork node is not an ancestor of the "
                    f"eviction branch head")
            # (c) 直接 parent 不含逐出节点 id。
            direct_parents = {
                edge["from"] for edge in _rank_edges(builder, rank)
                if edge["to"] == probe_id}
            self.assertEqual(direct_parents & eviction_ids, set())

    def test_turn0_prefill_evictions_branch_and_gate_rearm(self):
        """(f)+ (a)-(c):turn-0 NO_HISTORY + prefill 增长逐出——分支无门,
        主链 readiness 屏障仍消费到达门(暂存-回填)。"""
        builder = self.builder
        fork_ids = _seed_frontier(builder, "t0")
        plan = _admission_plan(
            REQUEST_A, 0,
            extra={"prefill_evictions": (_store_transfer(),)})
        builder.emit_admission_batch(plan)
        for rank in (0, 4):
            barrier = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"].endswith("_history_tp_ready_barrier"))
            gate = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"].endswith("_arrival_timer_gate"))
            # (f) 屏障父链含到达门(暂存-回填没有吞掉主链的门)。
            self.assertIn(gate["id"], _ancestors(builder, rank, barrier["id"]))
            self._assert_branch_off_main_chain(fork_ids, {rank: barrier["id"]})

    def test_turn0_history_evictions_branch_and_gate_rearm(self):
        """(f)+ (a)-(c)+(d):turn-0 也会携带 history 逐出(为给新请求腾
        容量逐出其他会话)——到达门 pending 属于主链,统一 helper 暂存
        归还;分支首节点经 trigger_gate 数据面仍挂到达门。"""
        builder = self.builder
        fork_ids = _seed_frontier(builder, "t0h")
        plan = _admission_plan(
            REQUEST_A, 0,
            extra={"history_evictions": (_store_transfer(),)})
        builder.emit_admission_batch(plan)
        for rank in (0, 4):
            barrier = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"].endswith("_history_tp_ready_barrier"))
            gate = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"].endswith("_arrival_timer_gate"))
            self.assertIn(gate["id"], _ancestors(builder, rank, barrier["id"]))
            # (d) 分支首节点挂到达门(trigger_gate = pending_gate 同源)。
            eviction_ids = sorted(_eviction_node_ids(builder, rank))
            first_parents = {
                edge["from"] for edge in _rank_edges(builder, rank)
                if edge["to"] == eviction_ids[0]}
            self.assertIn(gate["id"], first_parents)
            self._assert_branch_off_main_chain(fork_ids, {rank: barrier["id"]})

    def test_turn1_history_evictions_keep_trigger_gate(self):
        """(d)+ (a)-(c):turn>0 history 逐出——分支首节点仍挂 interval 触发门。"""
        builder = self.builder
        _seed_frontier(builder, "t1")
        # 上一 turn 完成门(turn-1 end barrier 桩节点);fork 锚点 = 发射前
        # 一刻的 frontier(桩节点)。
        gate_ids = {}
        for rank in (0, 4):
            trace_builder = builder.builders[rank]
            trace_builder.set_context(REQUEST_A, "decode", 1)
            trace_builder.comp("prior_turn_end_barrier", 1, 1)
            gate_ids[rank] = trace_builder.previous_id
        builder.completion_gates[SESSION] = (0, dict(gate_ids))
        fork_ids = dict(gate_ids)
        plan = _admission_plan(
            REQUEST_B, 1,
            extra={
                "kv_location_after_completion": "local_hbm",
                "history_action": "LOCAL_HIT",
                "history_evictions": (_store_transfer(trigger=REQUEST_B),),
            })
        builder.emit_admission_batch(plan)
        for rank in (0, 4):
            interval_gate = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"].endswith("_interval_timer_gate"))
            eviction_ids = sorted(_eviction_node_ids(builder, rank))
            first_eviction_parents = {
                edge["from"] for edge in _rank_edges(builder, rank)
                if edge["to"] == eviction_ids[0]}
            # (d) 触发门挂在分支首节点。
            self.assertIn(
                interval_gate["id"], first_eviction_parents,
                f"rank {rank}: interval gate is not armed on the eviction "
                f"branch head")
            barrier = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"].endswith("_history_tp_ready_barrier"))
            self._assert_branch_off_main_chain(fork_ids, {rank: barrier["id"]})

    def test_decode_evictions_branch_keeps_drain_gate(self):
        """(d)+ (a)-(c):joiner decode 逐出——drain 块末门挂分支首节点,
        其后的 joiner 3000 迁移与列车体保持主链。"""
        builder = self.builder
        _seed_frontier(builder, "dec")
        drain_gates = {}
        for rank in (0, 4):
            trace_builder = builder.builders[rank]
            trace_builder.set_context(REQUEST_A, "prefill", 0)
            trace_builder.comp("drain_train_end_barrier", 1, 1)
            drain_gates[rank] = trace_builder.previous_id
        fork_ids = dict(drain_gates)
        joiner = {
            "request_id": REQUEST_B,
            "session_id": SESSION,
            "turn_index": 0,
            "queue_index": 0,
            "prefill_instance_index": 0,
            "decode_instance_index": 1,
            "prefill_context_tokens": 40,
            "decode_evictions": (_store_transfer(trigger=REQUEST_B),),
            "prefill_drain_block_ends": dict(drain_gates),
        }
        train_plan = {
            "train_id": "batch_train_i1_1",
            "instance_index": 1,
            "stage": "decode",
            "joiners": [joiner],
            "pass_spans": [(1, 41)],
            "iterations": 1,
            "prefill_start_member": None,
            "sentinel": False,
            "drain_members": [],
            "exit_members": [],
        }
        builder.emit_iteration_train(train_plan)
        for rank in (0, 4):
            eviction_ids = sorted(_eviction_node_ids(builder, rank))
            first_parents = {
                edge["from"] for edge in _rank_edges(builder, rank)
                if edge["to"] == eviction_ids[0]}
            self.assertIn(
                drain_gates[rank], first_parents,
                f"rank {rank}: drain block-end gate is not armed on the "
                f"decode eviction branch head")
        # (a)(b)(c):decode 实例无逐出节点(逐出只在 prefill 组 rank 上),
        # 端屏障自然无逐出祖先;prefill 侧 3000 迁移 send 在主链——其
        # 闭包不含同 rank 的 decode 逐出支链。
        self.assertFalse(_eviction_node_ids(builder, 2))
        barrier = next(
            node for node in _rank_nodes(builder, 2)
            if node["name"] == "batch_train_i1_1_end_barrier")
        self.assertTrue(barrier["id"])
        migration = next(
            node for node in _rank_nodes(builder, 0)
            if "_prefill_to_decode_kv_send_" in node["name"])
        self._assert_branch_off_main_chain(fork_ids, {0: migration["id"]})

    def test_blocked_admission_evictions_context_and_collection(self):
        """(e)+ (a)-(c):blocked 逐出——context stamping (prefill, 0)、节点
        随本批收集;主链 continuation(parent = fork)不含逐出祖先。"""
        builder = self.builder
        fork_ids = _seed_frontier(builder, "blk")
        plan = _admission_plan(REQUEST_B, 0)
        builder.emit_eviction_actions(plan, (_store_transfer(trigger=REQUEST_B),))
        # 包裹外继续主链(下一批发射前同批补一个主链节点并收集)。
        marker = builder._mark()
        for rank in (0, 4):
            trace_builder = builder.builders[rank]
            trace_builder.set_context(REQUEST_B, "prefill", 0)
            trace_builder.comp("main_chain_continuation", 1, 1)
        builder._collect(marker)
        # (e) 上下文 stamping + 收集交付。
        for rank in (0, 4):
            eviction_nodes = [
                node for node in _rank_nodes(builder, rank)
                if "_blocked_admission_evictions_" in node["name"]]
            self.assertTrue(eviction_nodes)
            for node in eviction_nodes:
                self.assertEqual(node["request_id"], REQUEST_B)
                self.assertEqual(node["stage"], "prefill")
                self.assertEqual(node["generation"], 0)
        self.assertTrue(builder.batch["nodes"])
        for rank in (0, 4):
            continuation = next(
                node for node in _rank_nodes(builder, rank)
                if node["name"] == "main_chain_continuation")
            self._assert_branch_off_main_chain(
                fork_ids, {rank: continuation["id"]})
        # fork 恢复后主链 continuation 的直接 parent 就是 fork 节点。
        for rank in (0, 4):
            continuation = next(
                node["id"] for node in _rank_nodes(builder, rank)
                if node["name"] == "main_chain_continuation")
            direct_parents = {
                edge["from"] for edge in _rank_edges(builder, rank)
                if edge["to"] == continuation}
            self.assertEqual(direct_parents, {fork_ids[rank]})

    def test_side_branch_leak_fails_closed_and_stash_returns(self):
        """统一契约(主方案 §3.2 勘误):分支内部 arming 未被任何节点消费
        (触发门泄漏)→ 发射后 fail-closed;fork 点的主链 armed pending
        由 helper 暂存归还,不进分支、原消费者照常消费。"""
        builder = self.builder
        # (泄漏守卫)emit_fn 内 arm 而不发射节点 → 发射后即拒。
        def _leaky():
            builder.builders[0].arm_dependency(12345)
        with self.assertRaises(RuntimeError):
            builder._emit_side_branch(_leaky)
        # (暂存归还)fork 点携带主链 pending → 发射后原样归还;
        # 分支节点不阻塞主链(previous_id 回滚到 fork 点)。泄漏场景
        # 在生产即 run 中止,干净路径段换全新 builder。
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()

        def _clean():
            for trace_builder in builder.builders.values():
                trace_builder.comp("branch_node", 1, 1)
        builder.builders[4].arm_dependency(777)
        previous_on_0 = builder.builders[0].previous_id
        builder._emit_side_branch(_clean)
        self.assertEqual(builder.builders[4].pending_extra_dependencies, [777])
        self.assertEqual(builder.builders[0].previous_id, previous_on_0)
        self.assertEqual(builder.builders[0].pending_extra_dependencies, [])

    def test_registration_and_retire_cleanup(self):
        """登记表生命周期:发射即登记、terminal 回收清除。"""
        builder = self.builder
        plan = _admission_plan(REQUEST_B, 0)
        builder.emit_eviction_actions(plan, (_store_transfer(),))
        self.assertIn(SESSION, builder.pending_store_tails)
        self.assertEqual(len(builder.pending_store_tails[SESSION]), 2)
        builder.retire_completion_gate(SESSION)
        self.assertNotIn(SESSION, builder.pending_store_tails)


if __name__ == "__main__":
    unittest.main()
