#!/usr/bin/env python3
"""test_store_restore_ordering.py -- B4(2026-09-13)store→restore 前递依赖
排序钉子测试(主方案《逐出与request执行的并行修改方案》§3.3/§4.6)。

逐出支链悬空后,"同会话逐出池写先于其下一轮池读"的链序传递性失效——
回迁发射统一入口(全量 remote_load / PARTIAL 后缀恢复分支)查
pending_store_tails 补边。合成"逐出后立刻再到达"序列钉住:

  (a) 同缘:restore 边缘 mem_load(或其前 1B request 节点,经串行链)
      的祖先闭包包含 store 的边缘 mem_store 节点(直接 data_dep);
  (b) 跨缘:1B p2p 中继对(store_relay send 的直接 parent = mem_store;
      restore 边缘 relay arrival recv 在 mem_load 祖先闭包内);
  (c) PARTIAL 后缀恢复分支同款补边(分支内发射,主链不受污染);
  (d) 两段式逐出的 suffix store 与 full store 两笔均被依赖(restore 读
      全区间须等齐);
  (e) 登记表生命周期:回迁消费即清除;中继 tag 走 TransferTagAllocator
      (≥10^7),命名避开 first_token/batch_train_ 锚点。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_store_restore_ordering.py(或 pytest)
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

SESSION = "session_order_0"
REQUEST_A = f"{SESSION}_request_0"
REQUEST_B = f"{SESSION}_request_1"


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
        layers=4,
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
            layers=4, hidden_size=64, num_heads=8, bytes_per_elem=2),
        hardware=SimpleNamespace(
            mesh_rows=3, mesh_cols=3, npus_count=9),
    )


def _store_transfer(layer_start, layer_end, *, shards, trigger=REQUEST_A):
    from session_kv_manager import KVTransfer
    return KVTransfer(
        kind="remote_store", phase="history",
        reason=f"evict_admission:layers{layer_start}-{layer_end}",
        session_id=SESSION, trigger_request_id=trigger,
        source_instance_index=0, target_instance_index=None,
        total_bytes=sum(shard.bytes for shard in shards),
        shards=tuple(shards), model_layers=4,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=4 if layer_start == 2 else 2,
        resident_prefix_layers_after=(
            2 if layer_start == 2 else 0),
    )


def _suffix_store_shards():
    from session_kv_manager import KVTransferShard
    return (
        KVTransferShard(source_rank=0, target_rank=0, edge_rank=0,
                        bytes=256, noc_path=(0,), layer_start=2, layer_end=4),
        KVTransferShard(source_rank=4, target_rank=1, edge_rank=1,
                        bytes=256, noc_path=(4, 1), layer_start=2, layer_end=4),
    )


def _full_store_shards():
    from session_kv_manager import KVTransferShard
    return (
        KVTransferShard(source_rank=0, target_rank=0, edge_rank=0,
                        bytes=256, noc_path=(0,), layer_start=0, layer_end=2),
        KVTransferShard(source_rank=4, target_rank=1, edge_rank=1,
                        bytes=256, noc_path=(4, 1), layer_start=0, layer_end=2),
    )


def _load_transfer(layer_start, layer_end, *, shards):
    from session_kv_manager import KVTransfer
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_remote_restore",
        session_id=SESSION, trigger_request_id=REQUEST_B,
        source_instance_index=None, target_instance_index=0,
        total_bytes=sum(shard.bytes for shard in shards),
        shards=tuple(shards), model_layers=4,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=(
            4 if layer_start == 0 else 2),
        resident_prefix_layers_after=4,
    )


def _load_shards(edge_by_target):
    from session_kv_manager import KVTransferShard
    return tuple(
        KVTransferShard(
            source_rank=edge, target_rank=target, edge_rank=edge,
            bytes=256, noc_path=(edge, target),
            layer_start=0, layer_end=4)
        for target, edge in edge_by_target.items())


def _suffix_load_shards():
    from session_kv_manager import KVTransferShard
    return (
        KVTransferShard(source_rank=0, target_rank=0, edge_rank=0,
                        bytes=256, noc_path=(0,), layer_start=2, layer_end=4),
        KVTransferShard(source_rank=1, target_rank=4, edge_rank=1,
                        bytes=256, noc_path=(1, 4), layer_start=2, layer_end=4),
    )


def _restore_plan(location, action, transfers, *, history_instance=None,
                  resident_prefix=0):
    return {
        "request_id": REQUEST_B,
        "session_id": SESSION,
        "turn_index": 1,
        "queue_index": 1,
        "prefill_instance_index": 0,
        "decode_instance_index": 0,
        "history_action": action,
        "history_source_instance_index": None,
        "history_transfer_bytes": sum(
            transfer.total_bytes for transfer in transfers),
        "history_recompute_tokens": 0,
        "history_tokens_before": 0,
        "prefill_context_tokens": 300,
        "prefill_length": 300,
        "kv_location_after_completion": location,
        "history_location_before_location": location,
        "history_location_before_instance_index": history_instance,
        "history_resident_prefix_layers": resident_prefix,
        "history_transfers": transfers,
    }


class _Harness:
    """共用夹具:seed frontier + 上一 turn 完成门 + 逐出发射。"""

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
        return {
            edge_rank: node_id
            for edge_rank, node_id, _ack
            in self.builder.pending_store_tails.get(SESSION, ())
        }


def _rank_nodes(builder, rank):
    return [node for node in builder.batch["nodes"]
            if node["rank"] == rank]


def _rank_edges(builder, rank):
    return [edge for edge in builder.batch["parent_edges"]
            if edge["rank"] == rank]


def _ancestors(builder, rank, node_id):
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


def _mem_load_node(builder, rank, shard_index=0):
    return next(
        node for node in _rank_nodes(builder, rank)
        if node["name"].endswith(
            f"shard{shard_index}_remote_load"))


class StoreRestoreOrderingTest(unittest.TestCase):
    """(a)-(e)排序断言。"""

    def test_same_edge_full_restore_direct_dependency(self):
        """(a):同缘直挂——restore mem_load 祖先闭包含 store 的 mem_store。"""
        harness = _Harness()
        builder = harness.builder
        store_ids = harness.emit_evictions(
            _store_transfer(2, 4, shards=_suffix_store_shards()))
        self.assertEqual(store_ids, {0: store_ids[0], 1: store_ids[1]})
        plan = _restore_plan(
            "remote_memory", "REMOTE_RESTORE",
            (_load_transfer(0, 4, shards=_load_shards({0: 0, 4: 1})),))
        builder.emit_admission_batch(plan)
        # (a) 每个 restore 边缘 rank 的 mem_load 都等到同缘 mem_store。
        for rank, edge in ((0, 0), (4, 1)):
            mem_load = _mem_load_node(
                builder, edge, shard_index=0 if edge == 0 else 1)
            closure = _ancestors(builder, edge, mem_load["id"])
            self.assertIn(
                store_ids[edge], closure,
                f"edge {edge}: mem_load does not wait for the store "
                f"mem_store {store_ids[edge]}")
        # (e) 登记表消费即清除。
        self.assertNotIn(SESSION, builder.pending_store_tails)

    def test_cross_edge_full_restore_uses_relay(self):
        """(b):跨缘 1B 中继——relay send 直接依赖 mem_store,relay arrival
        recv 在 restore mem_load 祖先闭包内。"""
        from session_kv_manager import KVTransferShard
        harness = _Harness()
        builder = harness.builder
        direct_shard = KVTransferShard(
            source_rank=0, target_rank=0, edge_rank=0,
            bytes=256, noc_path=(0,), layer_start=2, layer_end=4)
        store_ids = harness.emit_evictions(
            _store_transfer(2, 4, shards=(direct_shard,)))
        self.assertEqual(sorted(store_ids), [0])
        # 回迁换缘:target 4 的最近边缘 = 1 ≠ store 边缘 0。
        from session_kv_manager import KVTransferShard
        cross_shard = KVTransferShard(
            source_rank=1, target_rank=4, edge_rank=1,
            bytes=256, noc_path=(1, 4), layer_start=0, layer_end=4)
        plan = _restore_plan(
            "remote_memory", "REMOTE_RESTORE",
            (_load_transfer(0, 4, shards=(cross_shard,)),))
        builder.emit_admission_batch(plan)
        relay_send = next(
            node for node in _rank_nodes(builder, 0)
            if "_store_relay_edge0_to_edge1" in node["name"])
        relay_recv = next(
            node for node in _rank_nodes(builder, 1)
            if "_store_relay_arrival_edge0_to_edge1" in node["name"])
        # (b) send 直接 parent = mem_store(同 rank data_dep)。
        send_parents = {
            edge["from"] for edge in _rank_edges(builder, 0)
            if edge["to"] == relay_send["id"]}
        self.assertIn(store_ids[0], send_parents)
        self.assertEqual(relay_send["comm"]["bytes"], 1)
        self.assertEqual(relay_recv["comm"]["bytes"], 1)
        self.assertGreaterEqual(relay_send["comm"]["tag"], 10_000_000)
        # (b) mem_load 经串行链排在中继 recv 之后。
        mem_load = _mem_load_node(builder, 1, shard_index=0)
        closure = _ancestors(builder, 1, mem_load["id"])
        self.assertIn(relay_recv["id"], closure)
        self.assertNotIn(SESSION, builder.pending_store_tails)
        # (e) 命名纪律。
        for node in builder.batch["nodes"]:
            self.assertNotIn("first_token", node["name"])
            self.assertNotIn("batch_train_", node["name"])

    def test_partial_suffix_restore_patches_inside_branch(self):
        """(c):PARTIAL 后缀恢复分支内补边——mem_load 等齐同缘 mem_store,
        且主链(resident 前缀屏障)不含逐出支链污染。"""
        harness = _Harness()
        builder = harness.builder
        store_ids = harness.emit_evictions(
            _store_transfer(2, 4, shards=_suffix_store_shards()))
        plan = _restore_plan(
            "partial_hbm_remote", "REMOTE_SUFFIX_RESTORE",
            (_load_transfer(2, 4, shards=_suffix_load_shards()),),
            history_instance=0, resident_prefix=2)
        builder.emit_admission_batch(plan)
        for shard_index, edge in ((0, 0), (1, 1)):
            mem_load = _mem_load_node(builder, edge, shard_index=shard_index)
            closure = _ancestors(builder, edge, mem_load["id"])
            self.assertIn(store_ids[edge], closure)
        # 流水登记仍在(首 chunk 列车消费)。
        self.assertIn(REQUEST_B, builder._partial_first_chunk)
        # 主链 resident 前缀屏障不以逐出支链为祖先。
        barrier = next(
            node for node in _rank_nodes(builder, 0)
            if node["name"].endswith(
                "_prefill_resident_prefix_ready_barrier"))
        eviction_ids = {
            node["id"] for node in _rank_nodes(builder, 0)
            if "_blocked_admission_evictions_" in node["name"]}
        self.assertTrue(eviction_ids)
        self.assertEqual(
            _ancestors(builder, 0, barrier["id"]) & eviction_ids, set())
        self.assertNotIn(SESSION, builder.pending_store_tails)

    def test_two_stage_stores_both_required(self):
        """(d):两段式两笔 store(suffix + full)均在 restore 依赖闭包内。"""
        harness = _Harness()
        builder = harness.builder
        harness.emit_evictions(
            _store_transfer(2, 4, shards=_suffix_store_shards()))
        full_ids = harness.emit_evictions(
            _store_transfer(0, 2, shards=_full_store_shards()))
        suffix_ids = dict(full_ids)
        tails = builder.pending_store_tails[SESSION]
        suffix_ids = {
            edge: node_id for edge, node_id, _ in tails[:2]}
        full_ids = {
            edge: node_id for edge, node_id, _ in tails[2:]}
        self.assertNotEqual(suffix_ids[0], full_ids[0])
        plan = _restore_plan(
            "remote_memory", "REMOTE_RESTORE",
            (_load_transfer(0, 4, shards=_load_shards({0: 0, 4: 1})),))
        builder.emit_admission_batch(plan)
        for rank, edge in ((0, 0), (4, 1)):
            mem_load = _mem_load_node(
                builder, edge, shard_index=0 if edge == 0 else 1)
            closure = _ancestors(builder, edge, mem_load["id"])
            # (d) 两笔 store 的同缘 mem_store 都被依赖。
            self.assertIn(suffix_ids[edge], closure)
            self.assertIn(full_ids[edge], closure)
        self.assertNotIn(SESSION, builder.pending_store_tails)

    def test_restore_without_tails_is_noop(self):
        """无登记时回迁零增补(懒处理:无 store 支链即无额外节点)。"""
        harness = _Harness()
        builder = harness.builder
        before_nodes = len(builder.batch["nodes"])
        plan = _restore_plan(
            "remote_memory", "REMOTE_RESTORE",
            (_load_transfer(0, 4, shards=_load_shards({0: 0, 4: 1})),))
        builder.emit_admission_batch(plan)
        relay_nodes = [
            node for node in builder.batch["nodes"]
            if "_store_relay" in node["name"]]
        self.assertEqual(relay_nodes, [])
        self.assertGreater(len(builder.batch["nodes"]), before_nodes)


if __name__ == "__main__":
    unittest.main()
