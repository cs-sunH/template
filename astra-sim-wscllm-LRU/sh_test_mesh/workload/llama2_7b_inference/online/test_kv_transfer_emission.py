#!/usr/bin/env python3
"""test_kv_transfer_emission.py -- B3(2026-09-06)KV 转移发射钉子测试。

照抄 sh_2.0 test_graph_batch_builder.py:388-460(流水锁定)并适配本仓
P/D 分离发射骨架,另钉 remote_store/remote_load/noc_migrate 三链的
HBM 计费键(策略文档 §9 地图,计费三险逐节点核对):

  (a) remote_store 链 A(source≠edge):源 comm_send 默认 charged(无
      hbm_charge 键)→ edge comm_recv 带 hbm_charge=False → edge
      mem_store 无 hbm_access_mode 键(池写零计费)→ 1B ack 对;
  (b) remote_store 链 B(source==edge 直连):mem_store 带
      compute.hbm_access_mode == 1(POOL_READ 唯一计费);
  (c) remote_load 链:edge mem_load 无计费键 → edge comm_send 与
      target comm_recv 均 hbm_charge=False → target
      local_hbm_kv_restore 节点(is_local_hbm_kv_restore=True,
      RESTORE 唯一数据计费);
  (d) noc_migrate:send/recv 默认 charged + 1B ack 对;全部转移 tag
      ≥ 100_000_000(契约 §6 基址,H8 修正后与 _stage_tag 段错开);
  (e) REMOTE_RESTORE 全量恢复(session 级 Tiered-LRU 唯一远端路径):
      全层 [0, L) 单段 remote_load 发射 + 单一全层就绪屏障
      (history_tp_ready_barrier;恢复完成前不能消费历史 KV);
      prefix/suffix 双栅栏与首 chunk 层段拆分真流水已随 PARTIAL
      恢复路径删除;
  (f) pending history 门账本:sync 镜像逐出(整体逐出后位置恒
      remote_memory;非零残留 fail-closed)、turn-0 deferred 通道、
      retire 清理;
  (g) 列车头 joiner decode 逐出:触发门 = prefill 段块末(post-
      barrier),control rank 与源 rank 跨实例时 1B p2p 触发。

运行:cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_kv_transfer_emission.py   （或 pytest 同路径）
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

from generate_wsc_llm_trace import (  # noqa: E402
    TransferTagAllocator,
    _emit_kv_transfer,
)
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
from session_kv_manager import KVTransfer, KVTransferShard  # noqa: E402

# 4x4 mesh(16 rank):边界 = 行0/行3/列0/列3;内部 = {5,6,9,10}。
PREFILL_RANKS = (0, 1)    # 边界行实例(直连池端点:edge == source)
DECODE_RANKS = (5, 6)     # 内部实例(近缘 edge = 4 / 1)
SESSION = "session_emit_0"
LAYERS = 4


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
            SimpleNamespace(session_arrival_time_ns=0,
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


def _transfer(kind, shards, **kwargs):
    return KVTransfer(
        kind=kind,
        phase=kwargs.get("phase", "history"),
        reason=kwargs.get("reason", "history_remote_restore"),
        session_id=kwargs.get("session_id", SESSION),
        trigger_request_id=kwargs.get("trigger_request_id", "req"),
        source_instance_index=kwargs.get("source_instance_index"),
        target_instance_index=kwargs.get("target_instance_index"),
        total_bytes=kwargs.get("total_bytes",
                               sum(shard.bytes for shard in shards)),
        shards=tuple(shards),
        model_layers=LAYERS,
        layer_start=kwargs.get("layer_start", 0),
        layer_end=kwargs.get("layer_end", LAYERS),
        resident_prefix_layers_before=kwargs.get(
            "resident_prefix_layers_before", LAYERS),
        resident_prefix_layers_after=kwargs.get(
            "resident_prefix_layers_after", LAYERS),
    )


def _emit(builder, transfer, *, pending_gate=None, trigger_gate=None):
    marker = builder._mark()
    record = _emit_kv_transfer(
        config=_make_config(),
        builders=builder.builders,
        group_by_index=builder.group_by_index,
        tag_allocator=builder._tag_allocator,
        transfer=transfer,
        action_name="test_action",
        pending_gate=pending_gate,
        trigger_gate=trigger_gate,
    )
    builder._collect(marker)
    return record


def _nodes(builder, rank):
    return [node for node in builder.batch["nodes"]
            if node["rank"] == rank]


def _by_name(nodes, fragment):
    return [node for node in nodes if fragment in node["name"]]


class RemoteStoreChainTest(unittest.TestCase):
    """(a)+(b):remote_store 两链的 HBM 计费键逐节点钉死。"""

    def setUp(self):
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def test_remote_store_via_edge_four_step_chain(self):
        """链 A(source=5 内部,edge=1):源端唯一数据计费 + 过路零计费。"""
        transfer = _transfer(
            "remote_store", [_shard(source_rank=5, target_rank=1,
                                    edge_rank=1, bytes=4000,
                                    noc_path=(5, 1))],
            source_instance_index=1,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=0,
        )
        # (source 5 ∈ decode 实例;近缘 edge = 1,source≠edge → 链 A)
        record = _emit(self.builder, transfer)[ "shards" ][0]
        self.assertFalse(record["direct_edge_access"])
        source_nodes = _nodes(self.builder, 5)
        edge_nodes = _nodes(self.builder, 1)
        # 源 rank:数据 send(默认 charged——无 hbm_charge 键)+ ack recv。
        send = _by_name(source_nodes, "_send_to_edge1")
        self.assertEqual(len(send), 1)
        self.assertNotIn("hbm_charge", send[0]["comm"])
        self.assertEqual(send[0]["comm"]["bytes"], 4000)
        ack_recv = _by_name(source_nodes, "_ack_from_edge1")
        self.assertEqual(len(ack_recv), 1)
        self.assertEqual(ack_recv[0]["comm"]["bytes"], 1)
        # 边缘 rank:recv 过路零计费 + mem_store 池写无计费键 + ack send。
        recv = _by_name(edge_nodes, "_recv_from_rank5")
        self.assertEqual(len(recv), 1)
        self.assertIs(recv[0]["comm"]["hbm_charge"], False)
        store = _by_name(edge_nodes, "_remote_store")
        self.assertEqual(len(store), 1)
        self.assertNotIn("hbm_access_mode", store[0]["compute"])
        self.assertEqual(store[0]["compute"]["tensor_size"], 4000)
        self.assertEqual(len(_by_name(edge_nodes, "_ack_to_rank5")), 1)
        self.assertEqual(record["source_release_dependency"],
                         "remote_store_ack_recv")
        # tag 基址:全部转移 tag ≥ 100_000_000(契约 §6;H8 修正后基址)。
        for node in source_nodes + edge_nodes:
            if node["comm"]["bytes"] > 0 or "_ack_" in node["name"]:
                self.assertGreaterEqual(node["comm"]["tag"], 100_000_000)

    def test_remote_store_direct_edge_pool_read(self):
        """链 B(source==edge=1):POOL_READ 唯一计费(hbm_access_mode=1)。"""
        transfer = _transfer(
            "remote_store", [_shard(source_rank=0, target_rank=0,
                                    edge_rank=0, bytes=2000,
                                    noc_path=(0,))],
            source_instance_index=0,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=0,
        )
        record = _emit(self.builder, transfer)["shards"][0]
        self.assertTrue(record["direct_edge_access"])
        edge_nodes = _nodes(self.builder, 0)
        store = _by_name(edge_nodes, "_edge_store")
        self.assertEqual(len(store), 1)
        self.assertEqual(store[0]["compute"]["hbm_access_mode"], 1)
        self.assertEqual(store[0]["compute"]["tensor_size"], 2000)
        # 直连路径无 comm 节点(无双计费面)。
        self.assertFalse([
            node for node in edge_nodes if node["comm"]["bytes"] > 1])
        self.assertEqual(record["source_release_dependency"],
                         "mem_store_completion")


class RemoteLoadChainTest(unittest.TestCase):
    """(c):remote_load 链的 RESTORE 唯一计费。"""

    def setUp(self):
        self.builder = GraphBatchBuilder(_make_config())
        self.builder.begin_batch()

    def test_remote_load_restore_only_charging(self):
        """edge=1 mem_load → 1→5 send/recv 均零计费 → rank5 restore。"""
        from generate_wsc_llm_trace import PendingHistoryGate
        pending_gate = PendingHistoryGate(
            source_instance_index=1, timer_gates=(None, None),
            location="remote_memory")
        transfer = _transfer(
            "remote_load", [_shard(source_rank=1, target_rank=5,
                                   edge_rank=1, bytes=3000,
                                   noc_path=(1, 5))],
            target_instance_index=1,
            layer_start=0, layer_end=LAYERS,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=LAYERS,
        )
        record = _emit(self.builder, transfer,
                       pending_gate=pending_gate)["shards"][0]
        edge_nodes = _nodes(self.builder, 1)
        target_nodes = _nodes(self.builder, 5)
        load = _by_name(edge_nodes, "_remote_load")
        self.assertEqual(len(load), 1)
        self.assertNotIn("hbm_access_mode", load[0]["compute"])
        send = _by_name(edge_nodes, "_send_to_rank5")
        self.assertEqual(len(send), 1)
        self.assertIs(send[0]["comm"]["hbm_charge"], False)
        recv = _by_name(target_nodes, "_recv_from_edge1")
        self.assertEqual(len(recv), 1)
        self.assertIs(recv[0]["comm"]["hbm_charge"], False)
        restore = _by_name(target_nodes, "_target_hbm_write")
        self.assertEqual(len(restore), 1)
        self.assertTrue(restore[0]["is_local_hbm_kv_restore"])
        self.assertEqual(restore[0]["compute"]["tensor_size"], 3000)
        self.assertIsInstance(record["target_hbm_completion_node_id"], int)


class NocMigrateChainTest(unittest.TestCase):
    """(d):noc_migrate send/recv + ack;tag 基址。"""

    def test_noc_migrate_send_recv_ack(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        transfer = _transfer(
            "noc_migrate", [_shard(source_rank=5, target_rank=0,
                                   bytes=8000, noc_path=(5, 4, 0))],
            source_instance_index=1, target_instance_index=0,
        )
        record = _emit(builder, transfer)["shards"][0]
        source_nodes = _nodes(builder, 5)
        target_nodes = _nodes(builder, 0)
        self.assertEqual(len(_by_name(source_nodes, "_send")), 1)
        self.assertEqual(len(_by_name(target_nodes, "_recv")), 1)
        self.assertEqual(len(_by_name(target_nodes, "_ack_to_rank5")), 1)
        self.assertEqual(len(_by_name(source_nodes, "_ack_from_rank0")), 1)
        for node in source_nodes + target_nodes:
            self.assertNotIn("hbm_charge", node["comm"])
        self.assertGreaterEqual(record["data_tag"], 100_000_000)
        self.assertGreaterEqual(record["ack_tag"], 100_000_000)
        self.assertEqual(TransferTagAllocator().take(), 100_000_000)


def _prefill_plan(*, turn_index=0, history_action=None,
                  history_source=None, history_transfers=(),
                  resident_layers=None, location=None):
    plan = {
        "request_id": "req_turn{}".format(turn_index),
        "session_id": SESSION,
        "turn_index": turn_index,
        "queue_index": turn_index,
        "prefill_instance_index": 0,
        "decode_instance_index": 1,
        "history_action": history_action,
        "history_source_instance_index": history_source,
        "history_transfer_bytes": 0,
        "history_recompute_tokens": 0,
        "history_tokens_before": 64,
        "prefill_context_tokens": 64 + 256,
        "prefill_length": 256,
        "decode_length": 4,
        "history_transfers": tuple(history_transfers),
        "history_evictions": (),
        "prefill_evictions": (),
    }
    if resident_layers is not None:
        plan["history_resident_prefix_layers"] = resident_layers
    if location is not None:
        plan["history_location_before"] = location
    return plan


def _full_load_transfer():
    # 分片覆盖 prefill 实例全部 rank(0/1 均为边界 rank → 直连池端点);
    # REMOTE_RESTORE 全层 [0, L) 单段恢复(多 shard 行 = 单逻辑 victim)。
    return _transfer(
        "remote_load",
        [_shard(source_rank=0, target_rank=0, edge_rank=0, bytes=1500,
                noc_path=(0,)),
         _shard(source_rank=1, target_rank=1, edge_rank=1, bytes=1500,
                noc_path=(1,))],
        target_instance_index=0,
        reason="history_remote_restore",
        layer_start=0, layer_end=LAYERS,
        resident_prefix_layers_before=0,
        resident_prefix_layers_after=LAYERS,
    )


class RemoteRestoreEmissionTest(unittest.TestCase):
    """(e):REMOTE_RESTORE 全量恢复发射(单一全层就绪屏障,无层段拆分)。"""

    def setUp(self):
        self.config = _make_config()
        self.builder = GraphBatchBuilder(self.config)
        self.builder.begin_batch()

    def _arm_pending_gate(self, request_id, location="remote_memory",
                          source_instance=1):
        self.builder.completion_gates[SESSION] = (
            source_instance, {rank: None for rank in
                              self.builder.group_by_index[
                                  source_instance].ranks})
        self.builder.register_pending_history(
            request_id=request_id, session_id=SESSION,
            source_instance_index=source_instance, location=location)

    def test_remote_restore_full_layer_single_segment(self):
        """REMOTE_RESTORE:全层 [0, L) 单段 remote_load + 单一
        history_tp_ready_barrier;恢复完成前不能消费历史 KV。"""
        plan = _prefill_plan(
            turn_index=1,
            history_action="REMOTE_RESTORE", history_source=1,
            history_transfers=(_full_load_transfer(),),
            resident_layers=0, location="remote_memory")
        self._arm_pending_gate(plan["request_id"])
        self.builder.emit_prefill_batch(plan)
        # 恢复链:每 prefill rank 一个 target HBM 写回节点(两 shard 行
        # = 单逻辑 victim 的物理表达)。
        for rank in PREFILL_RANKS:
            nodes = _nodes(self.builder, rank)
            restores = _by_name(nodes, "_target_hbm_write")
            self.assertEqual(len(restores), 1)
            self.assertTrue(restores[0]["is_local_hbm_kv_restore"])
            # 单一全层就绪屏障(既有命名与位置)。
            barriers = _by_name(nodes, "history_tp_ready_barrier")
            self.assertEqual(len(barriers), 1)
            self.assertTrue(all(
                node["coll"]["bytes"] >= 1 for node in barriers))
            # 无层段拆分/无 prefix-suffix 双栅栏(PARTIAL 真流水已删除)。
            for banned in ("first_chunk_prefix", "first_chunk_suffix",
                           "remaining_aggregated", "prefix_ready_barrier",
                           "suffix_ready_barrier",
                           "resident_prefix_ready_barrier"):
                self.assertFalse(_by_name(nodes, banned), banned)
        # 恢复转移对象为单段全层 remote_load。
        records = [
            node for node in self.builder.batch["nodes"]
            if "_remote_load" in node["name"]]
        self.assertTrue(records)

    def test_multi_shard_store_registers_single_victim_tails(self):
        """多 shard = 单逻辑 victim:一条两 shard 的 remote_store 整体
        逐出,store 尾部登记在该会话名下(单 victim,非多 victim)。"""
        from generate_wsc_llm_trace import PendingHistoryGate
        store = _transfer(
            "remote_store",
            [_shard(source_rank=5, target_rank=1, edge_rank=1, bytes=900,
                    noc_path=(5, 1)),
             _shard(source_rank=6, target_rank=0, edge_rank=0, bytes=900,
                    noc_path=(6, 5, 0))],
            source_instance_index=1,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=0)
        record = _emit(self.builder, store)
        self.assertEqual(record["session_id"], SESSION)
        self.assertEqual(len(record["shards"]), 2)
        self.builder._register_store_tails([record])
        self.assertEqual(
            len(self.builder.pending_store_tails[SESSION]), 2)
        # 单会话键(非按 shard 拆分多会话/多 victim)。
        self.assertEqual(list(self.builder.pending_store_tails), [SESSION])

    def test_plain_history_has_no_layer_split(self):
        """无历史/LOCAL_HIT:无层段拆分节点(单次全层 request_aggregated)。"""
        self.builder.emit_prefill_batch(
            _prefill_plan(turn_index=0, history_action=None))
        names = {node["name"] for node in self.builder.batch["nodes"]}
        self.assertFalse([name for name in names
                          if "first_chunk_prefix" in name
                          or "first_chunk_suffix" in name])
        self.assertTrue([name for name in names
                         if "request_aggregated" in name])

    def test_missing_pending_gate_fails_closed(self):
        plan = _prefill_plan(turn_index=1, history_action=None)
        self.builder.completion_gates[SESSION] = (
            1, {rank: None for rank in DECODE_RANKS})
        with self.assertRaises(RuntimeError):
            self.builder.emit_prefill_batch(plan)

    def test_location_mismatch_fails_closed(self):
        plan = _prefill_plan(
            turn_index=1, history_action=None, location="remote_memory")
        self._arm_pending_gate(plan["request_id"],
                               location="local_hbm")
        with self.assertRaises(RuntimeError):
            self.builder.emit_prefill_batch(plan)

    def test_partial_history_action_fails_closed(self):
        """PARTIAL 消费侧不可达:携带 legacy PARTIAL 迁移对象/动作的
        合成 plan 直接被拒(多于一段的恢复)。"""
        plan = _prefill_plan(
            turn_index=1,
            history_action="PARTIAL_MIGRATE", history_source=1,
            history_transfers=(_full_load_transfer(),
                               _full_load_transfer()),
            resident_layers=2, location="remote_memory")
        self._arm_pending_gate(plan["request_id"])
        with self.assertRaises(RuntimeError):
            self.builder.emit_prefill_batch(plan)


class HistoryEvictionTriggerTest(unittest.TestCase):
    """history 逐出的触发门:turn-0 = 到达门(同实例直接 arm);
    turn>0 = interval 门在 decode 实例上,逐出源在 prefill 实例 →
    1B p2p 触发(跨实例,照抄 sh :1084-1092)。"""

    def _store_transfer(self):
        return _transfer(
            "remote_store",
            [_shard(source_rank=5, target_rank=1, edge_rank=1,
                    bytes=900, noc_path=(5, 1))],
            source_instance_index=1,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=0)

    def test_turn0_history_eviction_arms_arrival_gate(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        plan = _prefill_plan(turn_index=0, history_action=None)
        plan["history_evictions"] = (self._store_transfer(),)
        builder.emit_prefill_batch(plan)
        # 逐出源 rank 5(decode 实例)与门源(turn-0 = prefill 实例 0/1)
        # 不同实例 → 触发 1B p2p(prefill rank → decode rank)。
        trigger = _by_name(_nodes(builder, 0), "_trigger_to_rank5")
        self.assertEqual(len(trigger), 1)
        self.assertEqual(trigger[0]["comm"]["bytes"], 1)
        self.assertEqual(
            len(_by_name(_nodes(builder, 5), "_trigger_from_rank0")), 1)
        # 逐出链本体:rank 5 数据 send(charged 默认)+ edge 1 过路。
        self.assertEqual(
            len(_by_name(_nodes(builder, 5), "_send_to_edge1")), 1)
        recv = _by_name(_nodes(builder, 1), "_recv_from_rank5")
        self.assertIs(recv[0]["comm"]["hbm_charge"], False)

    def test_turn1_history_eviction_uses_pending_gate(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        plan = _prefill_plan(turn_index=1, history_action=None)
        plan["history_evictions"] = (self._store_transfer(),)
        builder.completion_gates[SESSION] = (
            1, {rank: None for rank in DECODE_RANKS})
        builder.register_pending_history(
            request_id=plan["request_id"], session_id=SESSION,
            source_instance_index=1, location="local_hbm")
        builder.emit_prefill_batch(plan)
        # 触发门 = decode 实例 interval 门(control rank 5 == 源 rank 5,
        # 同 rank → 直接 arm,无 1B 触发)。
        self.assertFalse([
            node for node in builder.batch["nodes"]
            if "_trigger_to_rank" in node["name"]])
        self.assertEqual(
            len(_by_name(_nodes(builder, 5), "_send_to_edge1")), 1)
        # 逐出链首节点(target rank 5)的父边含 interval gate 节点。
        gates = _by_name(_nodes(builder, 5), "interval_timer_gate")
        self.assertEqual(len(gates), 1)
        send = _by_name(_nodes(builder, 5), "_send_to_edge1")[0]
        edges = [edge for edge in builder.batch["parent_edges"]
                 if edge["rank"] == 5 and edge["to"] == send["id"]]
        self.assertIn(
            gates[0]["id"], {edge["from"] for edge in edges})


class PendingHistoryLedgerTest(unittest.TestCase):
    """(f):pending 门 sync 镜像 / turn-0 deferred 通道 / retire 清理。"""

    def setUp(self):
        self.builder = GraphBatchBuilder(_make_config())

    def _store(self, after):
        return _transfer(
            "remote_store", [_shard(source_rank=5, target_rank=1,
                                    edge_rank=1, bytes=10,
                                    noc_path=(5, 1))],
            source_instance_index=1, session_id=SESSION,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=after,
        )

    def test_sync_updates_gate_location(self):
        # 整体逐出(resident→0)把门位置镜像为 remote_memory;
        # 半驻留残留(after 非零)是不可达的旧机制,fail-closed。
        self.builder.register_pending_history(
            request_id="r1", session_id=SESSION,
            source_instance_index=1, location="local_hbm")
        self.builder.sync_pending_history_after_evictions(
            (self._store(0),))
        self.assertEqual(
            self.builder.pending_history["r1"].location,
            "remote_memory")
        with self.assertRaises(RuntimeError):
            self.builder.sync_pending_history_after_evictions(
                (self._store(2),))

    def test_turn0_deferred_channel(self):
        """有 session 键无门(turn-0 在途)→ 延迟;下一登记点消费。"""
        self.builder.pending_request_by_session[SESSION] = "r0"
        self.builder.sync_pending_history_after_evictions(
            (self._store(0),))
        self.assertEqual(
            self.builder.deferred_session_locations[SESSION],
            "remote_memory")
        self.assertEqual(
            self.builder.pop_deferred_session_location(SESSION),
            "remote_memory")
        self.assertNotIn(SESSION, self.builder.deferred_session_locations)

    def test_retire_clears_session_ledgers(self):
        self.builder.register_pending_history(
            request_id="r1", session_id=SESSION,
            source_instance_index=1, location="local_hbm")
        self.builder.retire_completion_gate(SESSION)
        self.assertNotIn(SESSION, self.builder.pending_request_by_session)
        # 门本体按 request_id 归属,由下一 turn 发射或 run-end 审计兜底。
        self.assertNotIn(SESSION, self.builder.completion_gates)


class TrainHeadDecodeEvictionTest(unittest.TestCase):
    """(g):joiner decode 逐出发射(触发门 = prefill 段块末)。"""

    def test_decode_eviction_gated_on_prefill_segment_end(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        prefill_plan = _prefill_plan(turn_index=0, history_action=None)
        builder.emit_prefill_batch(prefill_plan)
        segment_ends = dict(builder._prefill_segment_ends[
            prefill_plan["request_id"]])
        # 块末 = prefill 段 end barrier 节点(post-barrier 口径)。
        for rank in PREFILL_RANKS:
            barrier = _by_name(_nodes(builder, rank),
                               "chunks_aggregated_end_barrier")
            self.assertEqual(segment_ends[rank], barrier[-1]["id"])

        eviction = _transfer(
            "remote_store",
            [_shard(source_rank=5, target_rank=1, edge_rank=1,
                    bytes=700, noc_path=(5, 1))],
            source_instance_index=1,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=0)
        joiner = _prefill_plan(turn_index=0, history_action=None)
        joiner["decode_evictions"] = (eviction,)
        train_plan = {
            "train_id": "batch_train_i1_1",
            "instance_index": 1,
            "joiners": [joiner],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        builder.begin_batch()
        builder.emit_iteration_train(train_plan)
        # 块末已消费。
        self.assertNotIn(joiner["request_id"],
                         builder._prefill_segment_ends)
        # 触发门:prefill rank(0) → decode rank(5)的 1B p2p。
        prefill_nodes = _nodes(builder, 0)
        trigger = _by_name(prefill_nodes, "_trigger_to_rank5")
        self.assertEqual(len(trigger), 1)
        self.assertEqual(trigger[0]["comm"]["bytes"], 1)
        trigger_recv = _by_name(_nodes(builder, 5), "_trigger_from_rank0")
        self.assertEqual(len(trigger_recv), 1)
        # 触发门的父边 = prefill 段块末节点(同 rank)。
        edges = [edge for edge in builder.batch["parent_edges"]
                 if edge["rank"] == 0]
        trigger_parents = {edge["from"] for edge in edges
                           if edge["to"] == trigger[0]["id"]}
        self.assertIn(segment_ends[0], trigger_parents)
        # 逐出链本体:decode rank 5 的数据 send + edge 1 过路。
        self.assertEqual(
            len(_by_name(_nodes(builder, 5), "_send_to_edge1")), 1)
        self.assertEqual(
            len(_by_name(_nodes(builder, 1), "_recv_from_rank5")), 1)

    def test_joiner_without_evictions_needs_no_segment_end(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        joiner = _prefill_plan(turn_index=0, history_action=None)
        joiner["decode_evictions"] = ()
        train_plan = {
            "train_id": "batch_train_i1_2",
            "instance_index": 1,
            "joiners": [joiner],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        builder.emit_iteration_train(train_plan)
        self.assertNotIn(joiner["request_id"],
                         builder._prefill_segment_ends)

    def test_evictions_without_segment_end_fail_closed(self):
        builder = GraphBatchBuilder(_make_config())
        builder.begin_batch()
        eviction = _transfer(
            "remote_store",
            [_shard(source_rank=5, target_rank=1, edge_rank=1,
                    bytes=700, noc_path=(5, 1))],
            source_instance_index=1,
            resident_prefix_layers_before=LAYERS,
            resident_prefix_layers_after=0)
        joiner = _prefill_plan(turn_index=0, history_action=None)
        joiner["decode_evictions"] = (eviction,)
        train_plan = {
            "train_id": "batch_train_i1_3",
            "instance_index": 1,
            "joiners": [joiner],
            "pass_spans": [(1, 321)],
            "iterations": 1,
            "exit_members": [],
        }
        with self.assertRaises(RuntimeError):
            builder.emit_iteration_train(train_plan)


if __name__ == "__main__":
    unittest.main()
