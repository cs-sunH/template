#!/usr/bin/env python3
"""test_prefill_remote_read_graph.py -- prefill remote-read 前缀读流
（规格书§三，2026-09-25；D09 逐 tile 化 2026-09-27）GraphBatchBuilder 侧
回归。

目标语义（PARTIAL 基，exec != home）：准入批发射读流**触发链**（恰一次
消费 interval/arrival gate），数据 tile 不在准入批——逐窗×tile 的
send/recv 随消费窗（逐 chunk 迭代体块）在各 prefill 列车头/体发射：
tile 1 上主链、tile 2 即时旁挂、tile 3+ 待发（缓冲复用释放门 = 消费体
块发射点——tile i+2 生产等 tile i 消费完成，任意时刻远读暂存 ≤2
tile/rank）；后缀 [p,L) 池恢复腿仍在准入批发射（两腿共享准入 frontier
并行分叉）。

覆盖：
  1. §三.1/§三.3 准入发射：跨实例 interval gate 重建到本轮 prefill
     （exec）实例后经 TransferTriggerGate 1B relay 触发——exec timer →
     exec trigger send → home trigger recv；准入批零数据 tile 节点
     （消费驱动逐 tile 发射，不整轮预建）；
  2. LOCAL 基：无 history transfers 的 remote-read 准入只发触发链；
  3. fail-closed 面：remote-read 无读计划 / 读计划挂错动作 / tile 身份
     缺失或重复 / 窗口门欠覆盖或过覆盖 / 完成残留窗口；
  4. 列车头窗口发射：tile 1 主链（_prefill_read 命名）+ tile 2 即时 +
     tile 3+ 待发账本；体块门 (rid, tile) 并集恰覆盖窗口登记面；
  5. 释放链结构：tile i+2 的生产（home data send）经 release ack 锚定
     消费 tile i 的体块（消费驱动，慢计算不停泵不越 2 tile）；
  6. §三.1/§三.7 并行分叉结构：触发链支、后缀恢复支与 readiness
     barrier 同 frontier 分叉；barrier 祖先集不含分支尾部；
  7. 真实调度器集成（真 KVCacheManager + 真规划器 + 真 _plan_train +
     _emit_train 首列车）与首步拆分拒绝（整列发射才是逐 tile 消费驱动
     发射的载体）。

运行：cd sh_test_mesh/workload/llama2_7b_inference &&
      python3 online/test_prefill_remote_read_graph.py
      （或 pytest 同路径）
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

from face_scheduler import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
    KVTransferSequence,
    plan_suffix_restore_groups,
)
from generate_face_trace import PendingHistoryGate  # noqa: E402
from online.graph_batch_builder import GraphBatchBuilder  # noqa: E402
import online.sh30_online_scheduler as sh30  # noqa: E402
from online.test_prefill_remote_read_scheduler import (  # noqa: E402
    EXEC_INSTANCE as INTEG_EXEC_INSTANCE,
    _force_remote_read,
    _make_manager,
    _make_runtime,
    _make_scheduler,
    _model,
    _seed_partial,
    _seed_resident,
)

HOME_INSTANCE = 0   # ranks (0, 1)——历史前缀驻留（逻辑 home）
EXEC_INSTANCE = 1   # ranks (2, 3)——本轮 prefill 执行实例
HOME_RANKS = (0, 1)
EXEC_RANKS = (2, 3)
MODEL_LAYERS = 16
PREFIX_LAYERS = 8   # PARTIAL 基驻留前缀 p
HISTORY_TOKENS = 100
REQUEST_R = "req_prr"
SESSION = "s"
#: 每 token 每层每 rank 字节（与 layer_restore 夹具同参：4B）。
_BYTES_PER_TOKEN_LAYER_RANK = 4


def _graph_config():
    return SimpleNamespace(
        npus_count=4,
        remote_operand_loads=False,
        trace_granularity="request_aggregated",
        inference_groups=[
            SimpleNamespace(ranks=HOME_RANKS, pg_name="tp_ins0"),
            SimpleNamespace(ranks=EXEC_RANKS, pg_name="tp_ins1"),
        ],
        layers=MODEL_LAYERS,
        hidden_size=4,
        ffn_size=4,
        vocab_size=4,
        bytes_per_elem=1,
        num_heads=2,
        mlp_variant="gelu",
        request_queue=[
            SimpleNamespace(
                session_arrival_time_ns=0,
                inter_request_interval_ns=1000,
            ),
        ],
        remote_memory=SimpleNamespace(edge_npus=EXEC_RANKS),
    )


def _shard_bytes(layer_start, layer_end):
    return (
        _BYTES_PER_TOKEN_LAYER_RANK * (layer_end - layer_start)
        * HISTORY_TOKENS)


def _read_tile_transfer(window_index, tile_index, *, bytes_per_tile):
    """home→exec 前缀读流 tile（KV 管理器逐窗×tile 规划器同构夹具：
    noc_migrate / stream_only / read_tile 身份 / 前缀层区间 [0, p)）。"""
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=target, edge_rank=None,
            bytes=bytes_per_tile,
            noc_path=(source, target),
            layer_start=0, layer_end=PREFIX_LAYERS,
        )
        for source, target in zip(HOME_RANKS, EXEC_RANKS))
    return KVTransfer(
        kind="noc_migrate", phase="prefill",
        reason="remote_read_prefill_prefix",
        session_id=SESSION, trigger_request_id=REQUEST_R,
        source_instance_index=HOME_INSTANCE,
        target_instance_index=EXEC_INSTANCE,
        total_bytes=2 * bytes_per_tile,
        shards=shards, model_layers=MODEL_LAYERS,
        layer_start=0, layer_end=PREFIX_LAYERS,
        resident_prefix_layers_before=PREFIX_LAYERS,
        resident_prefix_layers_after=PREFIX_LAYERS,
        stream_only=True,
        read_tile=(window_index, tile_index),
    )


def _read_tiles(window_count, *, bytes_per_tile=800):
    """逐窗×tile 序列（每窗 2 tile —— 双缓冲释放链的非退化形态）。"""
    return tuple(
        _read_tile_transfer(window, tile, bytes_per_tile=bytes_per_tile)
        for window in range(1, window_count + 1)
        for tile in (1, 2))


def _suffix_restore_transfer(layer_start, layer_end, group_index, *,
                             bytes_scale=1):
    """remote pool→exec 后缀恢复组（prepare_prefill remote-read 分支
    同构夹具：remote_load / restore_group 标记 / 同缘直读）。"""
    shards = tuple(
        KVTransferShard(
            source_rank=edge, target_rank=target, edge_rank=edge,
            bytes=bytes_scale * _shard_bytes(layer_start, layer_end),
            noc_path=(edge, target),
            layer_start=layer_start, layer_end=layer_end,
        )
        for target, edge in zip(EXEC_RANKS, EXEC_RANKS))
    return KVTransfer(
        kind="remote_load", phase="history",
        reason="history_suffix_pool_restore_working_copy",
        session_id=SESSION, trigger_request_id=REQUEST_R,
        source_instance_index=None,
        target_instance_index=EXEC_INSTANCE,
        total_bytes=2 * bytes_scale * _shard_bytes(
            layer_start, layer_end),
        shards=shards, model_layers=MODEL_LAYERS,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=layer_start,
        resident_prefix_layers_after=layer_end,
        restore_group=group_index,
    )


def _read_plan_template(read_tiles, *, prefix_layers=PREFIX_LAYERS,
                        location="partial_hbm_remote"):
    """准入计划携带的 prefill 远读计划快照（D16 有界模板形态——调度器
    plan_dict()["prefill_remote_read_plan"] 同构：O(窗数) 模板参数 +
    builder 模板级复检所需 session/rank/scan 向量；不物化 tile 列表）。

    模板参数由夹具 tile 序列派生并保持内部自洽（单扫描窗）：
    window_scans 恒 1 ⇒ base_scans_shards = 每窗每 rank tile 字节和、
    tiles_by_window[w] = ⌈base_scans_shards/tile_bytes⌉（与真实规划器
    的窗内切法同构）。"""
    tiles_by_window = {}
    for transfer in read_tiles:
        window_index, _tile = transfer.read_tile
        tiles_by_window[window_index] = (
            tiles_by_window.get(window_index, 0) + 1)
    window_count = len(tiles_by_window)
    tile_bytes = read_tiles[0].shards[0].bytes
    # 每窗每 rank 单扫描字节（夹具每窗 tile 数一致——取窗 1 派生）。
    base_scans_shards = tuple(
        tiles_by_window[1] * tile_bytes for _ in HOME_RANKS)
    # token 向下对齐的夹具口径（kv_token_bytes 取整除 tile_bytes 的
    # 协议参数；仅模板形状披露，不被 builder 消费）。
    kv_token_bytes = _BYTES_PER_TOKEN_LAYER_RANK * prefix_layers
    assert tile_bytes % kv_token_bytes == 0
    return {
        "home_instance": HOME_INSTANCE,
        "exec_instance": EXEC_INSTANCE,
        "read_prefix_layers": prefix_layers,
        "total_bytes": sum(t.total_bytes for t in read_tiles),
        "tile_count": len(read_tiles),
        "window_count": window_count,
        "tiles_by_window": [
            tiles_by_window[window] for window in range(1, window_count + 1)],
        "tile_bytes": tile_bytes,
        "tile_tokens": tile_bytes // kv_token_bytes,
        "kv_token_bytes": kv_token_bytes,
        "scan_passes": window_count,
        "session_id": SESSION,
        "source_ranks": tuple(HOME_RANKS),
        "target_ranks": tuple(EXEC_RANKS),
        "base_scans_shards": base_scans_shards,
    }


def _admission_plan(read_tiles, suffix_groups, *, turn=1,
                    joint_action="remote-read",
                    location="partial_hbm_remote"):
    return {
        "request_id": REQUEST_R,
        "session_id": SESSION,
        "turn_index": turn,
        "queue_index": 0,
        "prefill_instance_index": EXEC_INSTANCE,
        "decode_instance_index": EXEC_INSTANCE,
        "admission_time_ns": None,
        "hbm_wait_ns": 0,
        "joint_action": joint_action,
        "history_location_before": type("Before", (), {
            "location": location,
            "instance_index": HOME_INSTANCE,
            "resident_prefix_layers": PREFIX_LAYERS,
        })(),
        "history_transfers": list(suffix_groups),
        "history_evictions": [],
        "prefill_evictions": [],
        # D16：读计划快照 = 有界模板（模板级合同复检；数据 tile 由
        # 列车规划期经 kv_manager.prefill_read_window_tiles 现算）。
        "prefill_remote_read_plan": _read_plan_template(
            read_tiles,
            prefix_layers=(
                MODEL_LAYERS if location == "local_hbm" else PREFIX_LAYERS)),
        "prefill_remote_read_bytes": sum(
            transfer.total_bytes for transfer in read_tiles),
        "history_tokens_before": HISTORY_TOKENS,
        "prefill_context_tokens": 10,
    }


def _seed_gate(builder, *, source_instance, ranks):
    """前置 1：上一轮 completion 批建立的 interval gate（在线模式等价
    直连构造）；source_instance = 上一轮执行实例。返回 (gate, ids)。"""
    gate_ids = {}
    for rank in ranks:
        builder.builders[rank].comp(f"gate_seed_rank{rank}", 1, 1)
        gate_ids[rank] = builder.builders[rank].previous_id
    gate = PendingHistoryGate(
        source_instance_index=source_instance,
        timer_gates=tuple(gate_ids[rank] for rank in ranks),
        location="partial_hbm_remote")
    builder.pending_history[REQUEST_R] = gate
    return gate, gate_ids


def _store_transfer(layer_start, layer_end):
    """历史后缀池写（remote_store；上一轮 completion 逐出的登记夹具）。"""
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=edge, edge_rank=edge,
            bytes=_shard_bytes(layer_start, layer_end),
            noc_path=(source, edge),
            layer_start=layer_start, layer_end=layer_end,
        )
        for source, edge in zip(HOME_RANKS, EXEC_RANKS))
    return KVTransfer(
        kind="remote_store", phase="completion",
        reason="fixture_store",
        session_id=SESSION, trigger_request_id="s_seed",
        source_instance_index=HOME_INSTANCE,
        target_instance_index=None,
        total_bytes=2 * _shard_bytes(layer_start, layer_end),
        shards=shards, model_layers=MODEL_LAYERS,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=layer_end,
        resident_prefix_layers_after=layer_start,
    )


def _seed_store_tails(builder, groups):
    """前置 2：同会话在飞池写登记（store→restore 前递 fail-closed 前
    提；池写尾部锚 = exec 缘种子节点，同缘直接 arm，id 序合法）。"""
    for rank in EXEC_RANKS:
        builder.builders[rank].comp(f"store_seed_rank{rank}", 1, 1)
        anchor = builder.builders[rank].previous_id
        builder._register_store_tails(
            _store_transfer(groups[0].layer_start, groups[-1].layer_end),
            {"shards": [
                {"edge_rank": rank,
                 "edge_store_node_id": anchor,
                 "source_ack_recv_node_id": anchor},
            ]})


class _GraphHarness:
    """准入 + 列车发射夹具（新 builder + 批累加器）。"""

    def __init__(self):
        self.builder = GraphBatchBuilder(_graph_config())
        self.builder.begin_batch()

    def nodes(self):
        return self.builder.batch["nodes"]

    def edges(self):
        return self.builder.batch["parent_edges"]

    def parents(self):
        parents = {}
        for edge in self.edges():
            parents.setdefault((edge["rank"], edge["to"]), []).append(
                edge["from"])
        return parents

    def by_name(self, needle, rank=None):
        return [
            node for node in self.nodes()
            if needle in node["name"]
            and (rank is None or node["rank"] == rank)]

    def trigger_sends(self):
        """前缀读流触发 1B send（exec rank，依赖 interval gate）。"""
        return {
            node["rank"]: node for node in self.by_name("_prefill_read")
            if "_trigger_to_rank" in node["name"]}

    def trigger_recvs(self):
        """前缀读流触发 1B recv（home rank，读流启动因果锚）。"""
        return {
            node["rank"]: node for node in self.by_name("_prefill_read")
            if "_trigger_from_rank" in node["name"]}


def _partial_groups(*, suffix_bytes_scale=1):
    """缺省 PARTIAL 组：前缀读流 tile 序列 + 后缀 [8,16) 单组。"""
    prefix_tiles = _read_tiles(1)
    suffix_groups = tuple(
        _suffix_restore_transfer(start, end, index,
                                 bytes_scale=suffix_bytes_scale)
        for index, (start, end) in enumerate(
            plan_suffix_restore_groups(PREFIX_LAYERS, MODEL_LAYERS)))
    return prefix_tiles, suffix_groups


def _local_prefix_tiles(window_count=1):
    """LOCAL 基全层读流 tile 序列（层区间恒 [0, L=16)）。"""
    return tuple(
        _read_tile_transfer(window, tile, bytes_per_tile=800)
        .__class__(
            kind="noc_migrate", phase="prefill",
            reason="remote_read_prefill_prefix",
            session_id=SESSION, trigger_request_id=REQUEST_R,
            source_instance_index=HOME_INSTANCE,
            target_instance_index=EXEC_INSTANCE,
            total_bytes=2 * 800 * MODEL_LAYERS // PREFIX_LAYERS,
            shards=tuple(
                KVTransferShard(
                    source_rank=source, target_rank=target, edge_rank=None,
                    bytes=800 * MODEL_LAYERS // PREFIX_LAYERS,
                    noc_path=(source, target),
                    layer_start=0, layer_end=MODEL_LAYERS,
                )
                for source, target in zip(HOME_RANKS, EXEC_RANKS)),
            model_layers=MODEL_LAYERS,
            layer_start=0, layer_end=MODEL_LAYERS,
            resident_prefix_layers_before=MODEL_LAYERS,
            resident_prefix_layers_after=MODEL_LAYERS,
            stream_only=True,
            read_tile=(window, tile),
        )
        for window in range(1, window_count + 1)
        for tile in (1, 2))


def _emit_partial_admission(harness, *, gate_source=None):
    """PARTIAL 基准入发射（触发链 + 后缀恢复分支）。gate_source 缺
    省 = exec（同实例 gate；非 None = 上一轮在他实例 → R5 重建路径）。"""
    prefix_tiles, suffix_groups = _partial_groups()
    _seed_gate(
        harness.builder,
        source_instance=(
            EXEC_INSTANCE if gate_source is None else gate_source),
        ranks=(HOME_RANKS if gate_source is not None else EXEC_RANKS))
    _seed_store_tails(harness.builder, suffix_groups)
    harness.builder.emit_admission_batch(_admission_plan(
        prefix_tiles, suffix_groups))
    return prefix_tiles, suffix_groups


def _prefill_read_entry(tiles, *, request_id=REQUEST_R, spans_per_window=1):
    """列车计划的 prefill_read 规格（SH _plan_prefill_read_window 同构：
    逐迭代体块 + (rid, tile) 门并集）。tile 序号 = 传入序（1 基）。"""
    window_count = len({transfer.read_tile[0] for transfer in tiles})
    blocks = []
    pipeline_tile_runs = []
    tile_cursor = 0
    for window in range(1, window_count + 1):
        window_tiles = [
            transfer for transfer in tiles
            if transfer.read_tile[0] == window]
        gates = []
        for _ in window_tiles:
            tile_cursor += 1
            gates.append((request_id, tile_cursor))
        if window_tiles:
            shards = tuple({
                "global_shard_ordinal": shard_ordinal,
                "source_rank": int(first.source_rank),
                "target_rank": int(first.target_rank),
                "total_data_bytes": sum(
                    int(transfer.shards[shard_ordinal].bytes)
                    for transfer in window_tiles),
                "bytes_per_unit": 1,
                "noc_path": tuple(int(rank) for rank in first.noc_path),
            } for shard_ordinal, first in enumerate(window_tiles[0].shards)
                if sum(int(transfer.shards[shard_ordinal].bytes)
                       for transfer in window_tiles) > 0)
            units_per_tile = int(window_tiles[0].shards[0].bytes)
            total_units = max(
                int(shard["total_data_bytes"]) for shard in shards)
            pipeline_tile_runs.append({
                "first_tile": tile_cursor - len(window_tiles) + 1,
                "tile_count": len(window_tiles),
                "pass_or_window_ordinal": window,
                "source_ordinal": 0,
                "source_key": "window:{}".format(window),
                "layer_start": int(window_tiles[0].layer_start),
                "layer_end": int(window_tiles[0].layer_end),
                "axis_unit": "stream_bytes",
                "total_units": total_units,
                "units_per_tile": units_per_tile,
                "tail_units": total_units - (len(window_tiles) - 1)
                    * units_per_tile,
                "shards": shards,
            })
        blocks.append({
            "start_iter": window,
            "end_iter": window,
            "weight_passes": 1,
            "spans": [(10, 100 + window) for _ in range(spans_per_window)],
            "gates": gates,
        })
    tile_entries = tuple(
        (index, transfer) for index, transfer in enumerate(tiles, start=1))
    tile_sequence = KVTransferSequence(
        len(tile_entries), lambda index: tile_entries[index],
        pipeline_tile_runs=pipeline_tile_runs)
    return {
        "request_id": request_id,
        "plan": {
            "request_id": request_id, "session_id": SESSION,
            "turn_index": 1, "queue_index": 0,
            # Explicit actual-emission seed matching the scheduler's selected
            # P request plan; fixture lease ID is fixed and ordinal preview is
            # committed only at GraphBatch finalization.
            "pipeline_identity_seeds": tuple({
                "request_id": request_id, "stage": "prefill",
                "generation": 0, "stage_epoch": 0, "rank": rank,
                "staging_allocation_id": 1,
            } for rank in EXEC_RANKS),
        },
        "tiles": tile_sequence,
        "body_blocks": blocks,
        "window_span": (1, window_count),
    }


def _train_plan(train_id, spans, *, stage="prefill", first_chunk=None,
                iterations=None, prefill_read=None):
    spans = list(spans)
    return {
        "train_id": train_id,
        "instance_index": EXEC_INSTANCE,
        "stage": stage,
        "joiners": [],
        "members": [],
        "pass_spans": spans,
        "iterations": len(spans) if iterations is None else iterations,
        "prefill_start_member": (
            {"request_id": REQUEST_R} if first_chunk else None),
        "first_chunk_member": (
            {"request_id": REQUEST_R} if first_chunk else None),
        "drain_members": [],
        "exit_members": [],
        "head_request_id": REQUEST_R if first_chunk else None,
        "prefill_read": prefill_read,
    }


def _split_first_token(train_id, first_spans, rest_spans):
    return {
        "split": True,
        "first_spans": list(first_spans),
        "rest_spans": list(rest_spans),
        "debut_marker_members": [],
        "debut_exit_first_token": [],
        "wakeup_id": f"{train_id}_first_step",
    }


def _completion_plan():
    """completion 批 plan 夹具（残留检查路径共用）。"""
    return {
        "request_id": REQUEST_R,
        "session_id": SESSION,
        "turn_index": 1,
        "queue_index": 0,
        "decode_instance_index": EXEC_INSTANCE,
        "completion_evictions": [],
        "merge_transfers": [],
        "kv_location_after_completion": "local_hbm",
    }


def _parents_by_rank(harness):
    """(rank, to) → [from, ...] 批内父边索引（parent_edges 恒 rank 内
    边；跨 rank 同步由 comm tag 配对承载、不入图边）。"""
    parents = {}
    for edge in harness.edges():
        parents.setdefault((edge["rank"], edge["to"]), []).append(
            edge["from"])
    return parents


def _ancestors(parents, rank, node_id):
    """rank 内父向 BFS 可达集（同 rank 偏序足以裁决门控因果）。"""
    seen = set()
    stack = [node_id]
    while stack:
        for parent in parents.get((rank, stack.pop()), ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def _branch_ids_by_rank(harness, needle):
    """分支节点 (rank → id 集）。"""
    ids = {}
    for node in harness.nodes():
        if needle in node["name"]:
            ids.setdefault(node["rank"], set()).add(node["id"])
    return ids


def _barrier_by_rank(harness):
    return {
        node["rank"]: node
        for node in harness.by_name("_prefill_kv_ready_barrier")}


class PrefixReadStreamAdmissionTests(unittest.TestCase):
    """§三.1-§三.3：准入批发射的读流触发链（数据 tile 不在准入批）。"""

    def setUp(self):
        self.h = _GraphHarness()

    def test_cross_instance_gate_rebuild_triggers_prefix_stream(self):
        """上一轮 gate 在 home（跨实例）→ R5 重建到 exec → 1B relay 触发
        home；准入批只有触发链节点（数据 tile 随消费窗发射）。"""
        _emit_partial_admission(self.h, gate_source=HOME_INSTANCE)
        sends = self.h.trigger_sends()
        recvs = self.h.trigger_recvs()
        self.assertEqual(sorted(sends), sorted(EXEC_RANKS))
        self.assertEqual(sorted(recvs), sorted(HOME_RANKS))
        for rank in EXEC_RANKS:
            self.assertIn("_trigger_to_rank", sends[rank]["name"])
        for rank in HOME_RANKS:
            self.assertIn("_trigger_from_rank", recvs[rank]["name"])
        # 准入批零数据 tile 节点（消费驱动逐 tile 发射——不整轮预建）。
        data_nodes = [
            node for node in self.h.nodes()
            if "_prefill_read_action" in node["name"]
            and (node["name"].endswith("_send")
                 or node["name"].endswith("_recv"))]
        self.assertEqual(data_nodes, [])
        # interval gate 恰一次消费（准入发射弹出 pending_history——
        # 触发链是其唯一消费者）。
        self.assertNotIn(REQUEST_R, self.h.builder.pending_history)

    def test_local_base_remote_read_emits_trigger_without_history(self):
        """LOCAL 基：无 history transfers 的 remote-read 准入只发触发链
        （后缀腿缺席——读计划 tiles 供体在）。"""
        tiles = _local_prefix_tiles()
        _seed_gate(self.h.builder, source_instance=HOME_INSTANCE,
                   ranks=HOME_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(
            tiles, (), location="local_hbm"))
        self.assertEqual(sorted(self.h.trigger_recvs()), sorted(HOME_RANKS))
        self.assertEqual(
            [node for node in self.h.nodes()
             if "history_transfer" in node["name"]], [])

    def test_remote_read_without_read_plan_fails_closed(self):
        plan = _admission_plan(_local_prefix_tiles(), ())
        plan["prefill_remote_read_plan"] = None
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)

    def test_read_plan_with_wrong_action_fails_closed(self):
        plan = _admission_plan(
            _local_prefix_tiles(), (), joint_action="copy")
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)

    # ---- D16 模板级合同复检（builder _emit_prefill_remote_read_branch
    # 的新 fail-closed 路径）：逐 tile 结构不变量（read_tile 身份连续
    # 唯一、noc_migrate/stream_only）由物化器构造同源恒成立，钉测承
    # 担移至 test_prefill_remote_read_manager；此处钉模板级失败形态。----

    def test_read_plan_degenerate_window_template_fails_closed(self):
        # 窗界退化（窗数 0 / 逐窗 tile 数 0）：非退化窗模板合同。
        for mutate in (
                lambda template: template.update(window_count=0),
                lambda template: template.update(
                    window_count=1, tiles_by_window=[0]),
        ):
            plan = _admission_plan(_local_prefix_tiles(), ())
            mutate(plan["prefill_remote_read_plan"])
            with self.assertRaises(RuntimeError):
                self.h.builder.emit_admission_batch(plan)

    def test_read_plan_window_count_mismatch_fails_closed(self):
        # tiles_by_window 与 window_count 失配（模板参数自洽性）。
        plan = _admission_plan(_local_prefix_tiles(), ())
        plan["prefill_remote_read_plan"]["tiles_by_window"] = [2]
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)

    def test_read_plan_session_mismatch_fails_closed(self):
        # 会话归属错配：读计划模板戳属于另一会话。
        plan = _admission_plan(_local_prefix_tiles(), ())
        plan["prefill_remote_read_plan"]["session_id"] = "other_session"
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)

    def test_read_plan_home_instance_mismatch_fails_closed(self):
        # 端点错配：模板 home 与 history location 驻留实例不一致。
        plan = _admission_plan(_local_prefix_tiles(), ())
        plan["prefill_remote_read_plan"]["home_instance"] = (
            HOME_INSTANCE + 5)
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)

    def test_read_plan_prefix_layers_mismatch_fails_closed(self):
        # 前缀层界错配：LOCAL 基模板必须读 [0, L) 整段。
        plan = _admission_plan(
            _local_prefix_tiles(), (), location="local_hbm")
        template = plan["prefill_remote_read_plan"]
        template["read_prefix_layers"] = template["read_prefix_layers"] - 1
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)

    def test_read_plan_tp_relative_position_broken_fails_closed(self):
        # TP 相对位破坏：source_ranks 换序后与 target_ranks 的相对位
        # 映射不再保持（rank i 的 KV 不得漂移到另一相对位）。
        plan = _admission_plan(_local_prefix_tiles(), ())
        plan["prefill_remote_read_plan"]["source_ranks"] = tuple(
            reversed(HOME_RANKS))
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(plan)


def _seed_gate_ids(harness):
    return harness.builder.pending_history.get(REQUEST_R), None


class PrefillReadWindowTrainTests(unittest.TestCase):
    """D09：列车头窗口发射 + 体块门覆盖 + 释放链结构。"""

    def setUp(self):
        self.h = _GraphHarness()

    def _admit_and_train(self, window_count, *, spans_per_window=1):
        tiles = _read_tiles(window_count)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(tiles, ()))
        train = _train_plan(
            "batch_train_i1_1",
            [(10, 100 + i) for i in range(window_count)],
            first_chunk=True,
            prefill_read=_prefill_read_entry(
                tiles, spans_per_window=spans_per_window))
        self.h.builder.emit_iteration_train(train)
        return tiles, train

    def test_window_tiles_emit_per_train_head_and_gates_cover(self):
        """tile 1 主链 + tile 2 即时 + tile 3+ 待发；体块门并集恰覆盖
        窗口登记面（欠/过覆盖 fail-closed 由下述注入用例钉）。compact
        发射：per-tile send/recv 在 prefill_read_window descriptor 草稿
        的 endpoint_runs 内（两 owner rank 各一视图；物理节点面 =
        pipeline 锚 + 触发链）。"""
        tiles, _train = self._admit_and_train(4)
        drafts = sorted(
            (draft for draft in self.h.builder._pipeline_drafts.values()
             if draft.stream_kind == "prefill_read_window"),
            key=lambda draft: draft.identity.rank)
        self.assertEqual(
            [draft.identity.rank for draft in drafts], list(EXEC_RANKS))
        legs_by_tile = {}
        for draft in drafts:
            tile_by_group = {
                group["group_ordinal"]:
                    group["tile_axis"]["global_tile_first"]
                for group in draft.groups}
            for run in draft.endpoint_runs:
                first_global = (tile_by_group[run["group_ordinal"]]
                                + int(run["first_tile"]) - 1)
                for offset in range(int(run["tile_count"])):
                    tile = first_global + offset
                    for leg in run["legs"]:
                        legs_by_tile.setdefault(tile, set()).add(
                            (leg["endpoint"], int(run["rank"])))
        # 8 tile（4 窗 × 2）全部有 data_send（home 侧）与 data_recv
        #（exec 侧）；fill_ack 回执反向同构；release 腿恰自 tile ≥3
        # 起（双缓冲复用：tile 1..2 初始占用，无释放依赖）。
        self.assertEqual(sorted(legs_by_tile), list(range(1, 9)))
        for tile, legs in legs_by_tile.items():
            expected = {
                ("data_send", rank) for rank in HOME_RANKS} | {
                ("fill_ack_recv", rank) for rank in HOME_RANKS} | {
                ("data_recv", rank) for rank in EXEC_RANKS} | {
                ("fill_ack_send", rank) for rank in EXEC_RANKS}
            if tile >= 3:
                expected |= {
                    ("release_ack_send", rank) for rank in EXEC_RANKS} | {
                    ("release_ack_recv", rank) for rank in HOME_RANKS}
            self.assertEqual(legs, expected, tile)
        # 窗口登记账本在列车尾结清（下一列车可重登记）。
        self.assertNotIn(REQUEST_R, self.h.builder._prefill_read_window_tiles)
        self.assertNotIn(REQUEST_R, self.h.builder._credit_arms)
        self.assertNotIn(REQUEST_R, self.h.builder._credit_pending_tiles)

    def test_release_chain_lag_two_consumption_driven(self):
        """D09 缓冲复用释放链（tile ≥ 3）：tile t 的 exec 侧 recv 以释
        放 ack 为前置，且其支链发射于消费 tile t-2 的体块之后（生产等
        消费，慢计算停泵、任意时刻 ≤2 tile/rank 的结构表达；decode
        credit 同款钉法）。compact 发射：fork 因果由 descriptor 的
        initial_parent_json_id 承载；物理面上 tile t 的 data_recv 锚
        （pipeline frontier）发射序夹在消费 tile t-2 的体块末与 tile t
        的消费体块之间（per-rank 节点 id = 发射序）。"""
        tiles, _train = self._admit_and_train(4)
        exec_rank = EXEC_RANKS[0]
        train_id = "batch_train_i1_1"
        draft = next(
            draft for draft in self.h.builder._pipeline_drafts.values()
            if draft.stream_kind == "prefill_read_window"
            and draft.identity.rank == exec_rank)
        tile_by_group = {
            group["group_ordinal"]: (
                group["tile_axis"]["global_tile_first"],
                group["tile_axis"]["tile_count"])
            for group in draft.groups}
        # (tile, data_recv) → 支链 fork 锚的物理节点 id。
        anchor_by_tile = {}
        for anchor in draft.anchors:
            selector = anchor.get("selector", {})
            if (selector.get("kind") != "tile_endpoint_leg"
                    or selector.get("endpoint") != "data_recv"):
                continue
            first, _count = tile_by_group[selector["group_ordinal"]]
            anchor_by_tile[first + selector["tile_ordinal"] - 1] = (
                anchor["node_id"])
        body_nodes = {
            node["id"] for node in self.h.nodes()
            if f"{train_id}_rcb" in node["name"]
            and node["rank"] == exec_rank}
        self.assertTrue(body_nodes)

        for tile in range(3, 9):
            # tile t 的消费体块 = 窗 ceil(t/2) 的迭代体块。
            consumer_window = -(-tile // 2)
            producer_window = -(-(tile - 2) // 2)
            consumer_block = {
                node["id"] for node in self.h.nodes()
                if f"{train_id}_rcb{consumer_window}" in node["name"]
                and node["rank"] == exec_rank}
            producer_block = {
                node["id"] for node in self.h.nodes()
                if f"{train_id}_rcb{producer_window}" in node["name"]
                and node["rank"] == exec_rank}
            self.assertTrue(consumer_block)
            # tile t 的支链锚定在消费 tile t-2 的体块之后、tile t 的
            # 消费体块之前（释放边真实存在——发射序上的 lag-2 锚定）。
            self.assertIn(tile, anchor_by_tile)
            self.assertGreater(
                anchor_by_tile[tile], max(producer_block),
                f"tile {tile} production is not anchored after tile "
                f"{tile - 2} consumption")
            self.assertLess(
                anchor_by_tile[tile], min(consumer_block),
                f"tile {tile} production must precede its own consumption")

    def test_body_gate_undercoverage_fails_closed(self):
        tiles = _read_tiles(2)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(tiles, ()))
        entry = _prefill_read_entry(tiles)
        # 注入欠覆盖：末体块门删除 tile 4。
        entry["body_blocks"][-1]["gates"] = \
            entry["body_blocks"][-1]["gates"][:1]
        train = _train_plan(
            "batch_train_i1_1", [(10, 100), (10, 101)],
            first_chunk=True, prefill_read=entry)
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_iteration_train(train)

    def test_body_gate_overcoverage_fails_closed(self):
        tiles = _read_tiles(2)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(tiles, ()))
        entry = _prefill_read_entry(tiles)
        entry["body_blocks"][0]["gates"].append((REQUEST_R, 99))
        train = _train_plan(
            "batch_train_i1_1", [(10, 100), (10, 101)],
            first_chunk=True, prefill_read=entry)
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_iteration_train(train)

    def test_completion_leftover_window_fails_closed(self):
        """请求完成而窗口登记从未被列车体门覆盖 → raise（C13/C15 同款
        完成残留纪律）。"""
        tiles = _read_tiles(2)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(tiles, ()))
        self.h.builder._prefill_read_window_tiles[REQUEST_R] = {1, 2, 3, 4}
        self.h.builder._block_ends[REQUEST_R] = {
            "seg2": {rank: 0 for rank in EXEC_RANKS}}
        self.h.builder.next_plan[REQUEST_R] = None
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_completion_batch(_completion_plan())

    def test_split_train_with_window_fails_closed(self):
        """首步拆分列车不支持窗口体门控（整列发射才是逐 tile 消费驱动
        发射的载体；SH_FIRST_TOKEN_SPLIT 缺省关，fail-closed 纪律）。"""
        tiles = _read_tiles(2)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(tiles, ()))
        train = _train_plan(
            "batch_train_i1_1", [(10, 100), (10, 101)],
            first_chunk=True,
            prefill_read=_prefill_read_entry(tiles))
        train["first_token"] = _split_first_token(
            "batch_train_i1_1", [(10, 100)], [(10, 101)])
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_train_first_step(train)

    def test_tile_numbering_gap_fails_closed(self):
        tiles = _read_tiles(1)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        self.h.builder.emit_admission_batch(_admission_plan(tiles, ()))
        entry = _prefill_read_entry(tiles)
        entry["tiles"] = ((2, tiles[0]), (3, tiles[1]))
        train = _train_plan(
            "batch_train_i1_1", [(10, 100)], first_chunk=True,
            prefill_read=entry)
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_iteration_train(train)


def _block_consuming(nodes_by_id, tile_index):
    """消费给定 tile 序号的体块节点集（按 rcb phase 序——体块发射序 =
    迭代序，tile (2w-1, 2w) 归迭代 w）。"""
    window = tile_index // 2 + 1
    return {
        node["id"] for node in nodes_by_id.values()
        if f"_rcb{window}" in node["name"]
        and node["request_id"] == "batch_train_i1_1"}


class BranchForkParallelismTests(unittest.TestCase):
    """§三.1/§三.7：触发链支 / 后缀恢复支与 barrier 同 frontier 分叉。"""

    def test_branches_fork_from_same_admission_frontier(self):
        """触发链支与后缀恢复支首节点共享同一 parent = 准入 frontier；
        barrier 祖先集不含两支尾部（结构性主链节点）。"""
        harness = _GraphHarness()
        _seed_gate(harness.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        _prefix_tiles, suffix_groups = _partial_groups()
        _seed_store_tails(harness.builder, suffix_groups)
        frontier = {
            rank: harness.builder.builders[rank].previous_id
            for rank in EXEC_RANKS}
        harness.builder.emit_admission_batch(_admission_plan(
            _prefix_tiles, suffix_groups))
        parents = _parents_by_rank(harness)
        trigger_ids = _branch_ids_by_rank(harness, "_prefill_read")
        restore_ids = _branch_ids_by_rank(harness, "_history_transfer")
        barrier = _barrier_by_rank(harness)
        for rank in EXEC_RANKS:
            first_trigger = min(trigger_ids.get(rank, ()))
            self.assertIn(
                frontier[rank],
                parents.get((rank, first_trigger), ()),
                "trigger branch must fork from the admission frontier")
            first_restore = min(restore_ids.get(rank, ()))
            self.assertIn(
                frontier[rank],
                parents.get((rank, first_restore), ()),
                "restore branch must fork from the same frontier")
            ancestors = _ancestors(parents, rank, barrier[rank]["id"])
            self.assertFalse(
                ancestors & trigger_ids.get(rank, set()),
                "readiness barrier must not wait for the trigger branch")
            self.assertFalse(
                ancestors & restore_ids.get(rank, set()),
                "readiness barrier must not wait for the restore branch")

    def test_fork_survives_cross_instance_gate_rebuild(self):
        """跨实例 gate 重建形态下分叉 frontier 不变（R5 路径同 frontier）。"""
        harness = _GraphHarness()
        _emit_partial_admission(harness, gate_source=HOME_INSTANCE)
        barrier = _barrier_by_rank(harness)
        self.assertEqual(sorted(barrier), sorted(EXEC_RANKS))


class RealSchedulerGraphIntegrationTests(unittest.TestCase):
    """真实调度器准入事务（真 KVCacheManager + 逐窗×tile 规划器 + 真
    plan_dict）→ 真构图器准入发射 → 真实 _plan_train / _emit_train 首列
    车——生产形态（非手工夹具）的端到端衔接。"""

    INTEG_PREFIX_LAYERS = 12   # 调度器夹具的 PARTIAL 基驻留前缀 p
    INTEG_MODEL_LAYERS = 16

    def _make_integration(self, *, partial=True):
        model = _model()
        hardware, topology, manager = _make_manager(model)
        if partial:
            _seed_partial(manager)  # p=12 PARTIAL 基（home=0）
        else:
            _seed_resident(manager)  # LOCAL 基（home=0，p=16）
        scheduler = _make_scheduler(model, hardware, topology, manager)
        runtime = _make_runtime()
        scheduler.graph = GraphBatchBuilder(_graph_config())
        scheduler.graph.begin_batch()
        scheduler._pending_merge_alarms = {}
        scheduler._train_instance_index = {}
        scheduler.train_ledger_rows = []
        scheduler.train_ledger_sink = None
        gate_ids = {}
        for rank in HOME_RANKS:
            scheduler.graph.builders[rank].comp(
                f"integ_gate_seed_rank{rank}", 1, 1)
            gate_ids[rank] = scheduler.graph.builders[rank].previous_id
        scheduler.graph.pending_history[runtime.request_id] = (
            PendingHistoryGate(
                source_instance_index=HOME_INSTANCE,
                timer_gates=tuple(gate_ids[rank] for rank in HOME_RANKS),
                location=(
                    "partial_hbm_remote" if partial else "local_hbm")))
        if partial:
            from face_scheduler import KVTransferShard as _Shard
            for rank in EXEC_RANKS:
                scheduler.graph.builders[rank].comp(
                    f"integ_store_seed_rank{rank}", 1, 1)
                anchor = scheduler.graph.builders[rank].previous_id
                store = KVTransfer(
                    kind="remote_store", phase="completion",
                    reason="integration_seed_store",
                    session_id=SESSION, trigger_request_id="s_seed",
                    source_instance_index=HOME_INSTANCE,
                    target_instance_index=None,
                    total_bytes=2 * _shard_bytes(
                        self.INTEG_PREFIX_LAYERS,
                        self.INTEG_MODEL_LAYERS),
                    shards=tuple(
                        _Shard(
                            source_rank=source, target_rank=edge,
                            edge_rank=edge,
                            bytes=_shard_bytes(
                                self.INTEG_PREFIX_LAYERS,
                                self.INTEG_MODEL_LAYERS),
                            noc_path=(source, edge),
                            layer_start=self.INTEG_PREFIX_LAYERS,
                            layer_end=self.INTEG_MODEL_LAYERS)
                        for source, edge in zip(HOME_RANKS, EXEC_RANKS)),
                    model_layers=self.INTEG_MODEL_LAYERS,
                    layer_start=self.INTEG_PREFIX_LAYERS,
                    layer_end=self.INTEG_MODEL_LAYERS,
                    resident_prefix_layers_before=self.INTEG_MODEL_LAYERS,
                    resident_prefix_layers_after=self.INTEG_PREFIX_LAYERS,
                )
                scheduler.graph._register_store_tails(store, {"shards": [
                    {"edge_rank": rank,
                     "edge_store_node_id": anchor,
                     "source_ack_recv_node_id": anchor},
                ]})
        original_select = _force_remote_read(scheduler)
        try:
            self.assertTrue(scheduler._try_admit_request(runtime, 500))
        finally:
            sh30.select_instance_and_action = original_select
        scheduler.runtime_by_request_id[runtime.request_id] = runtime
        state = scheduler.instances[INTEG_EXEC_INSTANCE]
        plan = scheduler._plan_train(state)
        self.assertIsNotNone(plan)
        scheduler._emit_train(state, plan, [], 600)
        return scheduler, runtime, plan

    def test_partial_flow_real_shapes_tile_and_consume(self):
        """PARTIAL 基真形态：真规划逐窗×tile 前缀读流 + 真后缀组
        [12,16)；准入触发链 + 首列车窗口发射后账本恰结清。"""
        scheduler, runtime, plan = self._make_integration(partial=True)
        graph = scheduler.graph
        # 窗口账本结清（首列车已消费全部窗口 tile 门）。
        self.assertNotIn(
            runtime.request_id, graph._prefill_read_window_tiles)
        self.assertNotIn(runtime.request_id, graph._credit_arms)
        self.assertNotIn(runtime.request_id, graph._credit_pending_tiles)
        # 真实窗口 tile 流在场（compact 发射：prefill_read_window
        # descriptor 草稿携带逐 tile 的 data 腿；物理节点面 = pipeline
        # 锚 + 触发链）。
        window_drafts = [
            draft for draft in graph._pipeline_drafts.values()
            if draft.stream_kind == "prefill_read_window"]
        self.assertTrue(window_drafts)
        for draft in window_drafts:
            self.assertTrue(draft.endpoint_runs)
        # 计划窗数 == 列车 chunk 数（窗口随列车切片；D16 有界模板键）。
        read_plan = runtime.prefill_remote_read_plan
        window_count = int(read_plan["window_count"])
        self.assertEqual(
            window_count, len(plan["prefill_chunk_tokens"]))
        self.assertEqual(
            read_plan["window_cursor"], window_count + 1)
        # 后缀恢复组段在场（PARTIAL 基真后缀 [12,16)）。
        names = [node["name"] for node in graph.batch["nodes"]]
        self.assertTrue(
            any("layers12_15" in name for name in names),
            "real-shape suffix restore segment must be emitted")
        drain_watches = [
            watch for watch in scheduler._batch["watches"]
            if watch["request_id"] == runtime.request_id
            and watch["stage"] == "prefill"]
        self.assertEqual(len(drain_watches), 1)
        admission_rows = [
            row for row in scheduler.online_log_rows
            if row["kind"] == "joint_admission"
            and row["request_id"] == runtime.request_id]
        self.assertEqual(len(admission_rows), 1)
        self.assertEqual(
            admission_rows[0]["decision"]["prefill_remote_read_layers"],
            self.INTEG_PREFIX_LAYERS)

    def test_local_flow_real_shapes_full_layer_stream(self):
        """LOCAL 基真形态：全层 [0,16) 读流（层区间恒 [0, L)）。D16：
        runtime.prefill_remote_read_transfers 恒 ()（物化移到消费点），
        层界断言经 kv_manager.prefill_read_window_tiles 逐窗物化。"""
        scheduler, runtime, _plan = self._make_integration(partial=False)
        self.assertIsNotNone(runtime.prefill_remote_read_plan)
        self.assertEqual(runtime.prefill_remote_read_transfers, ())
        read_plan = runtime.prefill_remote_read_plan
        template = read_plan["template"]
        self.assertEqual(template["read_prefix_layers"],
                         self.INTEG_MODEL_LAYERS)
        transfers = []
        for window in range(1, read_plan["window_count"] + 1):
            transfers.extend(scheduler.kv_manager.prefill_read_window_tiles(
                template, window))
        self.assertTrue(transfers)
        for transfer in transfers:
            self.assertEqual(
                (transfer.layer_start, transfer.layer_end),
                (0, self.INTEG_MODEL_LAYERS))
        graph = scheduler.graph
        self.assertNotIn(
            runtime.request_id, graph._prefill_read_window_tiles)


class SplitRejectionTests(unittest.TestCase):
    """调度器侧拆分防御：prefill_read 列车不产拆分计划（builder 侧
    fail-closed 由 PrefillReadWindowTrainTests 钉）。"""

    def test_first_token_plan_degrades_for_read_window(self):
        scheduler = Sh30Stub()
        original = sh30.first_token_split_enabled
        sh30.first_token_split_enabled = lambda: True
        try:
            plan = {"prefill_read": {"request_id": REQUEST_R},
                    "iterations": 4}
            result = scheduler._first_token_plan(plan, [])
        finally:
            sh30.first_token_split_enabled = original
        self.assertIsNotNone(result)
        self.assertFalse(result["split"])


class Sh30Stub(sh30.Sh30OnlineScheduler):
    """_first_token_plan 直连替身（仅该方法的输入面）。"""

    def __init__(self):
        pass


if __name__ == "__main__":
    unittest.main()
