#!/usr/bin/env python3
"""test_prefill_remote_read_graph.py -- 规格书§三（2026-09-25 prefill
remote-read 分阶段）GraphBatchBuilder 侧回归。

目标语义（PARTIAL 基，exec != home）：prefill 两腿从同一准入 frontier
并行分叉——home 前缀 [0,p) 经 NoC 前缀读流（stream_only 瞬时流）、
remote pool 后缀 [p,L) 经 remote_load 池恢复；前缀层计算等 home→exec
逐组 recv 完成门、后缀层计算等池恢复逐组 HBM 写门，[0,L) 铺满
fail-closed；readiness barrier 只作结构性主链节点；LOCAL 基 = 全层
[0,L) 前缀读流（无 history transfers——旧 pass 分支废除）。

覆盖：
  1. §三.1/§三.2/§三.3 准入发射：跨实例 interval gate 重建到本轮
     prefill（exec）实例（R5 路径）后经 TransferTriggerGate 1B relay
     触发——exec timer → exec trigger send → home trigger recv →
     home KV send → exec KV recv；组间 home send 链 / exec recv 链
     顺序连接（组 0 后的 send/recv 不再依赖 interval gate）；逐组
     recv 完成门 + 层区间入 _prefill_remote_read_arms/_layers 双账本；
  2. LOCAL 基：无 history transfers 的 remote-read 准入发射全层前缀
     读流（旧 pass 分支的 gate 悬空缺陷废除）；无后缀腿；
  3. fail-closed 面：remote-read 无前缀读流 / 前缀读流挂错动作 /
     组区间缺口（gap）/ 重复准入登记在发射即 raise；
  4. §三.6 列车体层段铺满：[0,p) 前缀组段逐 rank 等前缀 recv 门、
     [p,L) 后缀组段等恢复 HBM 写门、层段发射与整段单次发射字节守恒
     （计费总量不因门控漂移）；
  5. §三.8 first-token split：首步批恰消费全部前缀组门、余量批不
     重复消费、列车尾残留检查 fail-closed、completion 残留
     fail-closed（C13/C15 同款纪律）；
  6. §三.1/§三.7 并行分叉结构：前缀读流支首节点、后缀恢复支首节点、
     readiness barrier 三者每个 exec rank 共享同一 parent = 准入
     frontier（含跨实例 gate 重建形态）；barrier 祖先集不含任何分支
     节点（结构性主链节点，不等两支尾部）；可区分传输字节（图侧时延
     载体）+ 支间互不可达偏序证明两腿重叠而非串行；
  7. §三.8 T_max 截断：8 迭代截断同构拆 1+7——首步批消费全部前缀组
     门、余量批哨兵核销后零残留 arm/层区间账本、拆分两批与整列发射
     体字节守恒；漏发首步批 / 首 chunk 请求错配 fail-closed。

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
    RESTORE_GROUP_LAYERS,
    plan_layer_groups,
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


def _prefix_read_transfer(layer_start, layer_end, group_index):
    """home→exec 前缀读流组（KV 管理器 plan_prefill_remote_read_
    transfers 同构夹具：noc_migrate / stream_only / 前缀层区间）。"""
    shards = tuple(
        KVTransferShard(
            source_rank=source, target_rank=target, edge_rank=None,
            bytes=_shard_bytes(layer_start, layer_end),
            noc_path=(source, target),
            layer_start=layer_start, layer_end=layer_end,
        )
        for source, target in zip(HOME_RANKS, EXEC_RANKS))
    return KVTransfer(
        kind="noc_migrate", phase="prefill",
        reason="remote_read_prefill_prefix",
        session_id=SESSION, trigger_request_id=REQUEST_R,
        source_instance_index=HOME_INSTANCE,
        target_instance_index=EXEC_INSTANCE,
        total_bytes=2 * _shard_bytes(layer_start, layer_end),
        shards=shards, model_layers=MODEL_LAYERS,
        layer_start=layer_start, layer_end=layer_end,
        resident_prefix_layers_before=PREFIX_LAYERS,
        resident_prefix_layers_after=PREFIX_LAYERS,
        stream_only=True,
    )


def _suffix_restore_transfer(layer_start, layer_end, group_index, *,
                             bytes_scale=1):
    """remote pool→exec 后缀恢复组（prepare_prefill remote-read 分支
    同构夹具：remote_load / restore_group 标记 / 同缘直读）。bytes_scale
    供并行偏序测试施加可区分传输规模（图侧时延载体 = 字节×拓扑，经
    C++ 代价模型派生时延，无独立时延参数）。"""
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


def _admission_plan(prefix_groups, suffix_groups, *,
                    turn=1, joint_action="remote-read",
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
        "prefill_remote_read_transfers": list(prefix_groups),
        "prefill_remote_read_bytes": sum(
            transfer.total_bytes for transfer in prefix_groups),
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
        """前缀读流触发 1B recv（home rank，触发 home send 链）。"""
        return {
            node["rank"]: node for node in self.by_name("_prefill_read")
            if "_trigger_from_rank" in node["name"]}


def _partial_groups(*, suffix_bytes_scale=1):
    """缺省 PARTIAL 组：前缀 [0,8) 单组 + 后缀 [8,16) 单组。"""
    prefix_groups = tuple(
        _prefix_read_transfer(start, end, index)
        for index, (start, end) in enumerate(
            plan_layer_groups(0, PREFIX_LAYERS)))
    suffix_groups = tuple(
        _suffix_restore_transfer(start, end, index,
                                 bytes_scale=suffix_bytes_scale)
        for index, (start, end) in enumerate(
            plan_layer_groups(PREFIX_LAYERS, MODEL_LAYERS)))
    return prefix_groups, suffix_groups


def _local_prefix_groups():
    """LOCAL 基全层前缀读流组（[0, 16) 两组）。"""
    return tuple(
        _prefix_read_transfer(start, end, index)
        for index, (start, end) in enumerate(
            plan_layer_groups(0, MODEL_LAYERS)))


def _emit_partial_admission(harness, *, gate_source=None):
    """PARTIAL 基准入发射（前缀 [0,8) + 后缀 [8,16)）。gate_source 缺
    省 = exec（同实例 gate；非 None = 上一轮在他实例 → R5 重建路径）。"""
    prefix_groups, suffix_groups = _partial_groups()
    _seed_gate(
        harness.builder,
        source_instance=(
            EXEC_INSTANCE if gate_source is None else gate_source),
        ranks=(HOME_RANKS if gate_source is not None else EXEC_RANKS))
    _seed_store_tails(harness.builder, suffix_groups)
    harness.builder.emit_admission_batch(_admission_plan(
        prefix_groups, suffix_groups))
    return prefix_groups, suffix_groups


def _train_plan(train_id, spans, *, stage="prefill", first_chunk=None,
                iterations=None):
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


def _emit_partial_admission_capture_frontier(*, suffix_bytes_scale=1):
    """PARTIAL 基准入发射 + 分叉 frontier 捕获：frontier = 支链发射前
    各 exec rank 的主链尾节点（gate/store 种子链尾）。_emit_side_branch
    的 fork 语义（graph_batch_builder :1925 起）："分支首节点 parent 恰
    为 fork frontier"——两支与 readiness barrier 是否同 frontier 分叉
    以该捕获值为锚断言。"""
    harness = _GraphHarness()
    _seed_gate(harness.builder, source_instance=EXEC_INSTANCE,
               ranks=EXEC_RANKS)
    prefix_groups, suffix_groups = _partial_groups(
        suffix_bytes_scale=suffix_bytes_scale)
    _seed_store_tails(harness.builder, suffix_groups)
    frontier = {
        rank: harness.builder.builders[rank].previous_id
        for rank in EXEC_RANKS}
    harness.builder.emit_admission_batch(_admission_plan(
        prefix_groups, suffix_groups))
    return harness, frontier


def _parents_by_rank(harness):
    """(rank, to) → [from, ...] 批内父边索引（parent_edges 恒 rank 内
    边；跨 rank 同步由 comm tag 配对承载、不入图边）。"""
    parents = {}
    for edge in harness.edges():
        parents.setdefault((edge["rank"], edge["to"]), []).append(
            edge["from"])
    return parents


def _ancestors(parents, rank, node_id):
    """rank 内父向 BFS 可达集（同 rank 偏序足以裁决两支串行/并行）。"""
    seen = set()
    stack = [node_id]
    while stack:
        for parent in parents.get((rank, stack.pop()), ()):
            if parent not in seen:
                seen.add(parent)
                stack.append(parent)
    return seen


def _branch_ids_by_rank(harness, needle):
    """分支节点 (rank → id 集)。命名空间识别：前缀读流支节点名含
    "_prefill_read"（触发链 + 组 send/recv/ack）；后缀恢复支节点名含
    "_history_transfer"（本发射形态主链无 history_transfer 节点——
    restore 组全部摘入旁挂分支，主链循环遍历空集）。"""
    ids = {}
    for node in harness.nodes():
        if needle in node["name"]:
            ids.setdefault(node["rank"], set()).add(node["id"])
    return ids


def _first_branch_node_by_rank(harness, needle):
    """分支首节点（每 rank 最小节点 id = 该 rank 支内最先发射节点，
    id 随发射单调递增）。"""
    first = {}
    for node in harness.nodes():
        if needle in node["name"]:
            rank = node["rank"]
            if rank not in first or node["id"] < first[rank]["id"]:
                first[rank] = node
    return first


def _barrier_by_rank(harness):
    return {
        node["rank"]: node
        for node in harness.by_name("_prefill_kv_ready_barrier")}


def _node_by_id(harness, rank, node_id):
    return next(
        node for node in harness.nodes()
        if node["rank"] == rank and node["id"] == node_id)


class PrefixReadStreamAdmissionTests(unittest.TestCase):
    """§三.1-§三.3/§三.5：准入批发射的前缀读流旁挂分支。"""

    def setUp(self):
        self.h = _GraphHarness()

    def test_cross_instance_gate_rebuild_triggers_prefix_stream(self):
        """上一轮 gate 在 home（跨实例）→ R5 重建到 exec → 1B relay 触发
        home send；组 0 send/recv 链序后随触发链（§三.2/§三.3）。"""
        _emit_partial_admission(self.h, gate_source=HOME_INSTANCE)
        # 重建后的 exec 实例 interval gate 节点在场。
        rebuilt_by_rank = {
            node["rank"]: node["id"]
            for node in self.h.by_name("interval_gate_rebuilt")}
        self.assertEqual(sorted(rebuilt_by_rank), sorted(EXEC_RANKS))
        # 触发链：exec trigger send（依赖重建 gate）→ home trigger recv。
        trigger_sends = self.h.trigger_sends()
        trigger_recvs = self.h.trigger_recvs()
        self.assertEqual(sorted(trigger_sends), sorted(EXEC_RANKS))
        self.assertEqual(sorted(trigger_recvs), sorted(HOME_RANKS))
        parents = self.h.parents()
        for rank in EXEC_RANKS:
            self.assertIn(
                rebuilt_by_rank[rank],
                parents.get((rank, trigger_sends[rank]["id"]), []),
                f"exec trigger send on rank {rank} must consume the "
                "rebuilt interval gate")
        # 前缀组 0 send/recv 的父依赖 = 触发链尾（链序触发）。
        first_sends = {
            node["rank"]: node for node in self.h.by_name(
                "_prefill_read_action000_")
            if node["name"].endswith("_send")}
        first_recvs = {
            node["rank"]: node for node in self.h.by_name(
                "_prefill_read_action000_")
            if node["name"].endswith("_recv")}
        for source_rank in HOME_RANKS:
            self.assertIn(
                trigger_recvs[source_rank]["id"],
                parents.get(
                    (source_rank, first_sends[source_rank]["id"]), []),
                f"home group-0 send on rank {source_rank} must follow "
                "the trigger recv")
        for target_rank in EXEC_RANKS:
            self.assertIn(
                trigger_sends[target_rank]["id"],
                parents.get(
                    (target_rank, first_recvs[target_rank]["id"]), []),
                f"exec group-0 recv on rank {target_rank} must follow "
                "the exec trigger send")
        # 双账本登记：逐组 recv 完成门 + 层区间。
        arms = self.h.builder._prefill_remote_read_arms.get(REQUEST_R)
        layers = self.h.builder._prefill_remote_read_layers.get(REQUEST_R)
        self.assertEqual(sorted(arms), [0])
        self.assertEqual(layers, {0: (0, PREFIX_LAYERS)})
        self.assertEqual(sorted(arms[0]), sorted(EXEC_RANKS))
        for target_rank in EXEC_RANKS:
            self.assertEqual(
                arms[0][target_rank], first_recvs[target_rank]["id"])

    def test_later_groups_chain_without_reconsuming_interval_gate(self):
        """组 1 的 home send / exec recv 只链接组 0 链尾，不再消费
        interval gate / 触发链（§三.3：不逐组重复消费）。LOCAL 基全层
        读流给出多组形态（同一分支发射，结构断言与 PARTIAL 基同路径）。"""
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        gate_seed_ids = set(
            self.h.builder.pending_history[REQUEST_R].timer_gates)
        self.h.builder.emit_admission_batch(_admission_plan(
            _local_prefix_groups(), (), location="local_hbm"))
        parents = self.h.parents()
        forbidden = gate_seed_ids | set(
            node["id"] for node in self.h.trigger_sends().values())
        sends_by_group = {0: {}, 1: {}}
        recvs_by_group = {0: {}, 1: {}}
        for group in (0, 1):
            for source_rank in HOME_RANKS:
                sends_by_group[group][source_rank] = next(
                    node for node in self.h.by_name(
                        f"_prefill_read_action{group:03d}_")
                    if node["name"].endswith("_send")
                    and node["rank"] == source_rank)
            for target_rank in EXEC_RANKS:
                recvs_by_group[group][target_rank] = next(
                    node for node in self.h.by_name(
                        f"_prefill_read_action{group:03d}_")
                    if node["name"].endswith("_recv")
                    and node["rank"] == target_rank)
        for source_rank in HOME_RANKS:
            gate_parents = set(
                parents.get(
                    (source_rank,
                     sends_by_group[1][source_rank]["id"]), ()))
            self.assertFalse(
                gate_parents & forbidden,
                "group-1 home send must not re-consume the interval "
                "gate or the trigger relay")
        for target_rank in EXEC_RANKS:
            gate_parents = set(
                parents.get(
                    (target_rank,
                     recvs_by_group[1][target_rank]["id"]), ()))
            self.assertFalse(
                gate_parents & forbidden,
                "group-1 exec recv must not re-consume the interval "
                "gate or the trigger relay")
        # 层区间账本 = [0, 8) + [8, 16) 连续铺满（LOCAL 基全层读流）。
        self.assertEqual(
            self.h.builder._prefill_remote_read_layers[REQUEST_R],
            {0: (0, 8), 1: (8, 16)})

    def test_local_base_remote_read_emits_stream_without_history(self):
        """LOCAL 基（无 history transfers）remote-read 准入：旧 pass 分支
        废除——gate 被前缀读流触发消费，全层读流登记，无后缀腿。"""
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        gate_seed_ids = {
            rank: gate_id for rank, gate_id in zip(
                EXEC_RANKS,
                self.h.builder.pending_history[REQUEST_R].timer_gates)}
        self.h.builder.emit_admission_batch(_admission_plan(
            _local_prefix_groups(), (), location="local_hbm"))
        parents = self.h.parents()
        trigger_sends = self.h.trigger_sends()
        self.assertEqual(sorted(trigger_sends), sorted(EXEC_RANKS))
        for rank in EXEC_RANKS:
            self.assertIn(
                gate_seed_ids[rank],
                parents.get((rank, trigger_sends[rank]["id"]), []),
                f"exec trigger send on rank {rank} must consume the "
                "arrival/interval gate (old pass left it dangling)")
        # arms/层区间 = 全层两组；后缀恢复账本缺席。
        arms = self.h.builder._prefill_remote_read_arms[REQUEST_R]
        self.assertEqual(
            self.h.builder._prefill_remote_read_layers[REQUEST_R],
            {0: (0, 8), 1: (8, 16)})
        self.assertEqual(sorted(arms), [0, 1])
        self.assertNotIn(REQUEST_R, self.h.builder._suffix_restore_arms)
        # 结构性 readiness barrier 保留（无后缀腿的 LOCAL 基同样发射）。
        self.assertTrue(self.h.by_name("_prefill_kv_ready_barrier"))

    def test_remote_read_without_prefix_stream_fails_closed(self):
        """remote-read 准入无前缀读流（plan 形态破损）→ raise。"""
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(_admission_plan((), ()))

    def test_prefix_stream_with_wrong_action_fails_closed(self):
        """前缀读流挂在非 remote-read 动作上 → raise（计划形态破损）。"""
        prefix_groups = tuple(
            _prefix_read_transfer(start, end, index)
            for index, (start, end) in enumerate(
                plan_layer_groups(0, PREFIX_LAYERS)))
        # turn-0：arrival gate 内联构造，绕开 turn>0 无 history 的旧
        # raise，使断言命中本分支的 fail-closed 检查。
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(_admission_plan(
                prefix_groups, (), turn=0, joint_action="stay"))

    def test_prefix_group_gap_fails_closed(self):
        """前缀组区间缺口（gap）→ 准入发射即 raise（§三.6 发射侧复检）。"""
        prefix_groups = (
            _prefix_read_transfer(0, 4, 0),
            _prefix_read_transfer(6, PREFIX_LAYERS, 1))  # [4,6) 缺口
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(_admission_plan(
                prefix_groups, ()))

    def test_double_admission_registration_fails_closed(self):
        """同一请求重复准入发射 → 组门账本重复登记 raise（重复消费
        fail-closed 的登记半边）。"""
        _emit_partial_admission(self.h)
        _seed_gate(self.h.builder, source_instance=EXEC_INSTANCE,
                   ranks=EXEC_RANKS)
        _seed_store_tails(self.h.builder, _partial_groups()[1])
        with self.assertRaises(RuntimeError):
            self.h.builder.emit_admission_batch(
                _admission_plan(*_partial_groups()))


class PrefixLayerSegmentBodyTests(unittest.TestCase):
    """§三.6/§三.7：首 chunk 列车体的层段铺满与 per-rank 门。"""

    def setUp(self):
        self.h = _GraphHarness()
        self.prefix_groups, self.suffix_groups = _emit_partial_admission(
            self.h)
        self.prefix_arms = dict(
            self.h.builder._prefill_remote_read_arms[REQUEST_R])
        self.restore_arms = list(
            self.h.builder._suffix_restore_arms[REQUEST_R])
        self.h.builder.emit_iteration_train(_train_plan(
            "batch_train_i1_1",
            [(10, 100), (10, 101), (10, 102)],
            first_chunk=True))

    def test_segments_tile_prefix_and_suffix(self):
        """[0,p) 前缀组段 + [p,L) 后缀组段铺满 [0, L)：段节点在场（组 0
        自层 0 起，无热前缀无门段）、账本随体消费弹出（恰一次）。"""
        names = [node["name"] for node in self.h.nodes()]
        self.assertTrue(
            any("layers00_07" in name for name in names),
            "prefix group-0 segment [0, 8) must be emitted")
        self.assertTrue(
            any("layers08_15" in name for name in names),
            "suffix restore segment [8, 16) must be emitted")
        self.assertNotIn(
            REQUEST_R, self.h.builder._prefill_remote_read_arms)
        self.assertNotIn(
            REQUEST_R, self.h.builder._prefill_remote_read_layers)
        self.assertNotIn(REQUEST_R, self.h.builder._suffix_restore_arms)

    def test_prefix_segment_gated_on_recv_per_rank(self):
        """前缀层段逐 rank 等该组 home→exec recv 完成门（§三.6）。"""
        parents = self.h.parents()
        segment_targets = {
            node["id"] for node in self.h.nodes()
            if node["request_id"] == "batch_train_i1_1"
            and "layers00_07" in node["name"]}
        self.assertTrue(segment_targets)
        for target_rank in EXEC_RANKS:
            gate_node = self.prefix_arms[0][target_rank]
            gated = {
                target for (node_rank, target), froms in parents.items()
                if node_rank == target_rank and gate_node in froms}
            self.assertTrue(
                gated & segment_targets,
                f"prefix recv gate must parent the prefix layer segment "
                f"on rank {target_rank}")

    def test_suffix_segment_gated_on_restore_write_per_rank(self):
        """后缀层段逐 rank 等该组池恢复 HBM 写完成门（既有 C15 口径在
        remote-read 混合形态下的衔接）。"""
        parents = self.h.parents()
        segment_targets = {
            node["id"] for node in self.h.nodes()
            if node["request_id"] == "batch_train_i1_1"
            and "layers08_15" in node["name"]}
        self.assertTrue(segment_targets)
        gates = self.restore_arms[0][3]
        for target_rank in EXEC_RANKS:
            gate_node = gates[target_rank]
            gated = {
                target for (node_rank, target), froms in parents.items()
                if node_rank == target_rank and gate_node in froms}
            self.assertTrue(
                gated & segment_targets,
                f"restore write gate must parent the suffix layer "
                f"segment on rank {target_rank}")

    def test_layer_segment_bytes_conserve(self):
        """层段发射 vs 整段单次发射：num_ops/tensor/comm 总量逐项相等
        （重叠门控不改变计费总量——不得在成本模型外偷漏计费）。"""
        def totals(nodes):
            return (
                sum(node["compute"]["num_ops"] for node in nodes),
                sum(node["compute"]["tensor_size"] for node in nodes),
                sum(node["mem"]["tensor_size"] for node in nodes),
                sum(node["comm"]["bytes"] for node in nodes),
                sum(node["coll"]["bytes"] for node in nodes),
            )
        segmented = [
            node for node in self.h.nodes()
            if node["request_id"] == "batch_train_i1_1"]
        other = GraphBatchBuilder(_graph_config())
        other.begin_batch()
        other.emit_iteration_train(_train_plan(
            "batch_train_i1_1", [(10, 100), (10, 101), (10, 102)],
            first_chunk=True))
        whole = [
            node for node in other.batch["nodes"]
            if node["request_id"] == "batch_train_i1_1"]
        self.assertEqual(totals(segmented), totals(whole))


class BranchForkParallelismTests(unittest.TestCase):
    """§三.1/§三.7：两腿并行分叉的结构证明——同准入 frontier fork、
    readiness barrier 结构性（不等两支尾部）、支间互不可达偏序
    （重叠而非串行；时延载体 = 可区分传输字节）。"""

    def test_branches_fork_from_same_admission_frontier(self):
        """前缀读流支首节点（exec trigger send）、后缀恢复支首节点、
        readiness barrier 三者在每个 exec rank 上共享同一 parent =
        分叉前主链尾（同一准入 frontier 分叉，§三.1）；frontier 节点
        本身是主链节点、不属于任何一支。"""
        harness, frontier = _emit_partial_admission_capture_frontier()
        parents = _parents_by_rank(harness)
        prefix_first = harness.trigger_sends()
        restore_first = _first_branch_node_by_rank(
            harness, "_history_transfer")
        barrier = _barrier_by_rank(harness)
        self.assertEqual(sorted(prefix_first), sorted(EXEC_RANKS))
        self.assertEqual(sorted(restore_first), sorted(EXEC_RANKS))
        self.assertEqual(sorted(barrier), sorted(EXEC_RANKS))
        for rank in EXEC_RANKS:
            for label, node in (
                    ("prefix trigger send", prefix_first[rank]),
                    ("suffix restore first node", restore_first[rank]),
                    ("readiness barrier", barrier[rank])):
                self.assertIn(
                    frontier[rank],
                    parents.get((rank, node["id"]), []),
                    f"{label} on rank {rank} must fork at the admission "
                    "frontier")
            frontier_node = _node_by_id(harness, rank, frontier[rank])
            self.assertNotIn(
                "_prefill_read", frontier_node["name"],
                "fork frontier must be a main-chain node, not a prefix "
                "branch node")
            self.assertNotIn(
                "_history_transfer", frontier_node["name"],
                "fork frontier must be a main-chain node, not a restore "
                "branch node")

    def test_fork_survives_cross_instance_gate_rebuild(self):
        """上一轮 gate 在 home（R5 批内重建）形态：重建 gate 节点成为
        exec 主链尾——两支首节点与 readiness barrier 仍同 parent 于重建
        节点（门重建先于两支 fork 归一，§三.2/§三.3）。"""
        harness = _GraphHarness()
        _emit_partial_admission(harness, gate_source=HOME_INSTANCE)
        rebuilt = {
            node["rank"]: node["id"]
            for node in harness.by_name("interval_gate_rebuilt")}
        self.assertEqual(sorted(rebuilt), sorted(EXEC_RANKS))
        parents = _parents_by_rank(harness)
        prefix_first = harness.trigger_sends()
        restore_first = _first_branch_node_by_rank(
            harness, "_history_transfer")
        barrier = _barrier_by_rank(harness)
        for rank in EXEC_RANKS:
            for label, node in (
                    ("prefix trigger send", prefix_first[rank]),
                    ("suffix restore first node", restore_first[rank]),
                    ("readiness barrier", barrier[rank])):
                self.assertIn(
                    rebuilt[rank],
                    parents.get((rank, node["id"]), []),
                    f"{label} on rank {rank} must fork at the rebuilt "
                    "interval gate")

    def test_branches_overlap_not_serial_under_asymmetric_bytes(self):
        """可区分传输规模（后缀字节 ×3）下两支互不可达：任一支的任何
        节点都不是另一支任何节点的祖先（同 rank 偏序无支→支边 = 并行
        而非串行——任意时延指派下两腿重叠）。字节载体断言：前缀支数据
        send comm = 前缀组字节、后缀支池读 mem = 3 倍后缀组字节（图侧
        时延由字节×拓扑经 C++ 代价模型派生，无独立时延参数可施）。"""
        harness, _frontier = _emit_partial_admission_capture_frontier(
            suffix_bytes_scale=3)
        prefix_ids = _branch_ids_by_rank(harness, "_prefill_read")
        restore_ids = _branch_ids_by_rank(harness, "_history_transfer")
        prefix_bytes = _shard_bytes(0, PREFIX_LAYERS)
        suffix_bytes = 3 * _shard_bytes(PREFIX_LAYERS, MODEL_LAYERS)
        self.assertNotEqual(prefix_bytes, suffix_bytes)
        data_sends = [
            node for node in harness.nodes()
            if "_prefill_read_action000_" in node["name"]
            and node["name"].endswith("_send")]
        self.assertTrue(data_sends, "prefix group-0 data sends must exist")
        for node in data_sends:
            self.assertEqual(
                node["comm"]["bytes"], prefix_bytes,
                "prefix branch data send must carry the prefix group "
                "bytes")
        restore_reads = [
            node for node in harness.nodes()
            if "_history_transfer" in node["name"]
            and node["name"].endswith("_remote_load")]
        self.assertTrue(restore_reads, "suffix pool reads must exist")
        for node in restore_reads:
            self.assertEqual(
                node["mem"]["tensor_size"], suffix_bytes,
                "suffix restore pool read must carry the scaled suffix "
                "group bytes")
        parents = _parents_by_rank(harness)
        for rank in EXEC_RANKS:
            for node_id in prefix_ids.get(rank, ()):
                self.assertFalse(
                    _ancestors(parents, rank, node_id)
                    & restore_ids.get(rank, set()),
                    f"prefix branch node {node_id} on rank {rank} must "
                    "not depend on the suffix restore branch")
            for node_id in restore_ids.get(rank, ()):
                self.assertFalse(
                    _ancestors(parents, rank, node_id)
                    & prefix_ids.get(rank, set()),
                    f"suffix restore node {node_id} on rank {rank} must "
                    "not depend on the prefix read branch")

    def test_readiness_barrier_does_not_wait_for_branch_tails(self):
        """readiness barrier 祖先集不含任何分支节点（结构性主链节点，
        不等两支尾部——§三.7；数据就绪由列车体层段 per-rank 门另行
        保证）；barrier 直接挂在分叉 frontier 上（支链发射未推进主链）。"""
        harness, frontier = _emit_partial_admission_capture_frontier()
        parents = _parents_by_rank(harness)
        prefix_ids = _branch_ids_by_rank(harness, "_prefill_read")
        restore_ids = _branch_ids_by_rank(harness, "_history_transfer")
        barrier = _barrier_by_rank(harness)
        self.assertEqual(sorted(barrier), sorted(EXEC_RANKS))
        for rank in EXEC_RANKS:
            ancestors = _ancestors(parents, rank, barrier[rank]["id"])
            self.assertFalse(
                ancestors & prefix_ids.get(rank, set()),
                f"readiness barrier on rank {rank} must not wait on the "
                "prefix read branch")
            self.assertFalse(
                ancestors & restore_ids.get(rank, set()),
                f"readiness barrier on rank {rank} must not wait on the "
                "suffix restore branch")
            self.assertIn(
                frontier[rank],
                parents.get((rank, barrier[rank]["id"]), []),
                f"readiness barrier on rank {rank} must chain on the "
                "admission frontier (structural main-chain node)")


class RealSchedulerGraphIntegrationTests(unittest.TestCase):
    """真实调度器准入事务（真 KVCacheManager + plan_prefill_remote_read_
    transfers + 真 plan_dict）→ 真构图器准入发射 → 真实 _plan_train /
    _emit_train 首列车——生产形态（非手工夹具）的端到端衔接。

    前提注记：本 fixture 的会话历史无上一轮图侧池写登记（turn-0 未过
    图），PARTIAL 基的后缀池恢复 fail-closed 前提（store→restore 前递）
    须按在线语义直连补登记——与 _seed_gate 同款的等价直连构造。"""
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
        # 真构图器替换 _GraphStub（生产准入发射 + 列车发射的真实衔接面；
        # 拓扑同参：HOME=0 ranks(0,1)，EXEC=1 ranks(2,3)，16 层）。
        scheduler.graph = GraphBatchBuilder(_graph_config())
        scheduler.graph.begin_batch()
        # _emit_train 的机制账本（F6 替身补设纪律：漏设 = AttributeError）。
        scheduler._pending_merge_alarms = {}
        scheduler._train_instance_index = {}
        scheduler.train_ledger_rows = []
        scheduler.train_ledger_sink = None
        # turn-1 runtime 的 interval gate（上一轮 completion 批建立的
        # 在线等价直连构造；source = home = 上一轮执行实例 → 生产 R5
        # 重建路径在真图上被真实触发）。
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
            # 后缀池恢复的 store→restore 前递前提（直连等价构造；池写尾
            # 锚 = exec 缘种子节点，同缘直接 arm）。层区间 = 真后缀
            # [p, L) = [12, 16)。
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
            # 真实准入事务全链（reserve/prepare/规划/登记/披露）——生产
            # _emit_admission 就在事务尾段以真 plan_dict 驱动真构图器。
            self.assertTrue(scheduler._try_admit_request(runtime, 500))
        finally:
            sh30.select_instance_and_action = original_select
        # 真实列车规划 + 发射（准入 runtime 已入 qp；首 chunk 在首列车）。
        # runtime_by_request_id 在生产由 _push_arrival 登记——本夹具直连
        # _try_admit_request，补登记（_register_train_watches 的读者）。
        scheduler.runtime_by_request_id[runtime.request_id] = runtime
        state = scheduler.instances[INTEG_EXEC_INSTANCE]
        plan = scheduler._plan_train(state)
        self.assertIsNotNone(plan)
        scheduler._emit_train(state, plan, [], 600)
        return scheduler, runtime, plan

    def test_partial_flow_real_shapes_tile_and_consume(self):
        """PARTIAL 基真形态：真规划前缀组 [0,6)+[6,12)（RESTORE_GROUP_
        LAYERS 切分）+ 真后缀组 [12,16)；准入 + 首列车后账本恰消费。"""
        scheduler, runtime, plan = self._make_integration(partial=True)
        graph = scheduler.graph
        arms = graph._prefill_remote_read_arms.get(runtime.request_id)
        self.assertIsNone(
            arms, "train body must consume the prefix arm ledger")
        layers = graph._prefill_remote_read_layers.get(runtime.request_id)
        self.assertIsNone(layers)
        # 准入批 + 列车批节点并存：真形态层段铺满（p=12 → 前缀两组
        # [0,6)/[6,12) + 后缀一组 [12,16)）。
        names = [node["name"] for node in graph.batch["nodes"]]
        for expected in ("layers00_05", "layers06_11", "layers12_15"):
            self.assertTrue(
                any(expected in name for name in names),
                f"real-shape segment {expected} must be emitted")
        # drain watch 已注册（生产列车发射路径的核销通道）。
        drain_watches = [
            watch for watch in scheduler._batch["watches"]
            if watch["request_id"] == runtime.request_id
            and watch["stage"] == "prefill"]
        self.assertEqual(len(drain_watches), 1)
        # 决策行披露字段在场（§二.3 真实生产披露面）。
        admission_rows = [
            row for row in scheduler.online_log_rows
            if row["kind"] == "joint_admission"
            and row["request_id"] == runtime.request_id]
        self.assertEqual(len(admission_rows), 1)
        self.assertGreater(
            admission_rows[0]["decision"]["prefill_remote_read_bytes"], 0)
        self.assertEqual(
            admission_rows[0]["decision"]["prefill_remote_read_layers"],
            self.INTEG_PREFIX_LAYERS)

    def test_local_flow_real_shapes_full_layer_stream(self):
        """LOCAL 基真形态：全层前缀读流 [0,16) 两组、无后缀腿；准入 +
        首列车端到端（真 remote-read 准入的无 history-transfers 路径）。"""
        scheduler, runtime, plan = self._make_integration(partial=False)
        graph = scheduler.graph
        self.assertIsNone(
            graph._prefill_remote_read_arms.get(runtime.request_id))
        names = [node["name"] for node in graph.batch["nodes"]]
        self.assertTrue(any("layers00_07" in name for name in names))
        self.assertTrue(any("layers08_15" in name for name in names))
        admission_rows = [
            row for row in scheduler.online_log_rows
            if row["kind"] == "joint_admission"
            and row["request_id"] == runtime.request_id]
        self.assertEqual(len(admission_rows), 1)
        self.assertEqual(
            admission_rows[0]["decision"]["prefill_remote_read_layers"],
            self.INTEG_MODEL_LAYERS)


class SplitAndLifecycleTests(unittest.TestCase):
    """§三.8：first-token split 消费纪律 + 完成残留 fail-closed。"""

    def _emit_admission_only(self):
        harness = _GraphHarness()
        _emit_partial_admission(harness)
        return harness

    def test_first_step_consumes_and_remainder_does_not(self):
        """首步批恰消费全部前缀组门；余量批不重复消费、尾部检查通过。"""
        harness = self._emit_admission_only()
        train = _train_plan(
            "batch_train_i1_1", [(10, 100), (10, 101), (10, 102)],
            first_chunk=True)
        train["first_token"] = _split_first_token(
            "batch_train_i1_1", [(10, 100)], [(10, 101), (10, 102)])
        first_result = harness.builder.emit_train_first_step(train)
        self.assertNotIn(
            REQUEST_R, harness.builder._prefill_remote_read_arms)
        self.assertNotIn(
            REQUEST_R, harness.builder._prefill_remote_read_layers)
        self.assertIn("first_token_members", first_result)
        self.assertIn("wakeup_members", first_result)
        # 首步批层段在场（首 chunk 恒在首步批）。
        names = [node["name"] for node in harness.nodes()]
        self.assertTrue(
            any("layers00_07" in name for name in names))
        # 余量批：不重复消费（体不传 first_chunk_member），尾部检查在
        # 账本已空时恒通过。
        harness.builder.emit_train_remainder(train)
        self.assertNotIn(
            REQUEST_R, harness.builder._prefill_remote_read_arms)

    def test_train_tail_leftover_arms_fail_closed(self):
        """列车尾残留检查：体发射绕过首 chunk 消费路径（发射链破损注
        入）→ 尾检查 fail-closed。"""
        harness = self._emit_admission_only()
        train = _train_plan(
            "batch_train_i1_1", [(10, 100)], first_chunk=True)
        # 体发射不传 first_chunk_member（模拟消费点缺席——账本仍在场）。
        harness.builder._emit_train_body(train, [(10, 100)], 1)
        self.assertIn(
            REQUEST_R, harness.builder._prefill_remote_read_arms)
        with self.assertRaises(RuntimeError):
            harness.builder._assert_prefill_read_arms_consumed(train)

    def test_completion_leftover_prefix_arms_fail_closed(self):
        """请求完成而前缀读流从未被首 chunk 体消费 → raise（C13/C15
        同款完成残留纪律）。"""
        harness = self._emit_admission_only()
        harness.builder._block_ends[REQUEST_R] = {
            "seg2": {rank: 0 for rank in EXEC_RANKS}}
        harness.builder.next_plan[REQUEST_R] = None
        with self.assertRaises(RuntimeError):
            harness.builder.emit_completion_batch(_completion_plan())

    def _tmax_split_train(self, train_id, *, first_chunk_request=REQUEST_R):
        """T_max=8 截断同构列车计划（sh30 _train_max_iter 有界 → 单列
        8 迭代、无自然 drain/exit 标记，哨兵以 train_id 批命名空间核销，
        sh30_online_scheduler "T_max 截断时头部未必 drain" 路径）：首
        token 拆 1+7，首 chunk span 恒在首步批。"""
        spans = [(10, 100 + offset) for offset in range(8)]
        train = _train_plan(train_id, spans, first_chunk=True, iterations=8)
        train["first_chunk_member"] = {"request_id": first_chunk_request}
        train["sentinel"] = True
        train["first_token"] = _split_first_token(
            train_id, spans[:1], spans[1:])
        return train

    def test_tmax_truncated_split_leaves_no_residual_arms(self):
        """T_max 截断同构拆 1+7：首步批消费全部前缀组门（双账本清空）、
        余量批不重复消费——哨兵核销后两账本全局零残留。"""
        harness = self._emit_admission_only()
        train = self._tmax_split_train("batch_train_i1_1")
        harness.builder.emit_train_first_step(train)
        self.assertNotIn(
            REQUEST_R, harness.builder._prefill_remote_read_arms)
        self.assertNotIn(
            REQUEST_R, harness.builder._prefill_remote_read_layers)
        # 首步批层段在场（前缀组 + 后缀组铺满——消费点真实门控）。
        names = [node["name"] for node in harness.nodes()]
        self.assertTrue(any("layers00_07" in name for name in names))
        self.assertTrue(any("layers08_15" in name for name in names))
        result = harness.builder.emit_train_remainder(train)
        # T_max 截断列车无自然 drain/exit 标记——哨兵是唯一核销通道。
        self.assertEqual(result["drain_members"], {})
        self.assertEqual(result["exit_members"], {})
        self.assertEqual(sorted(result["sentinel_members"]),
                         sorted(EXEC_RANKS))
        self.assertEqual(
            [node["name"] for node in harness.nodes()
             if node["name"].endswith("_sentinel")],
            ["batch_train_i1_1_sentinel"] * len(EXEC_RANKS))
        # 零残留终态：余量批重复消费 = 空弹无害，残留只可能来自首步批
        # 漏消费（上方断言已钉）——此处钉全局账本终态。
        self.assertEqual(harness.builder._prefill_remote_read_arms, {})
        self.assertEqual(harness.builder._prefill_remote_read_layers, {})

    def test_tmax_split_matches_whole_train_totals(self):
        """T_max 拆分（1+7）与整列发射（8 迭代）体字节守恒：首步层段化
        （weight_passes=1）+ 余量整段（=7）== 整列车层段化（=8）——拆分
        不重复计费、不丢字节（与门控消费正交的计费不变量）。"""
        def totals(harness):
            nodes = [
                node for node in harness.nodes()
                if node["request_id"] == "batch_train_i1_1"
                and node["rank"] in EXEC_RANKS]
            return tuple(
                sum(node[section][key] for node in nodes)
                for section, key in (
                    ("compute", "num_ops"),
                    ("compute", "tensor_size"),
                    ("mem", "tensor_size"),
                    ("comm", "bytes"),
                    ("coll", "bytes"),
                ))
        whole = _GraphHarness()
        _emit_partial_admission(whole)
        whole_train = _train_plan(
            "batch_train_i1_1",
            [(10, 100 + offset) for offset in range(8)],
            first_chunk=True, iterations=8)
        whole_train["sentinel"] = True
        whole.builder.emit_iteration_train(whole_train)
        split = self._emit_admission_only()
        train = self._tmax_split_train("batch_train_i1_1")
        split.builder.emit_train_first_step(train)
        split.builder.emit_train_remainder(train)
        self.assertEqual(totals(split), totals(whole))

    def test_tmax_remainder_without_first_step_fails_closed(self):
        """T_max 拆分漏发首步批（余量批直发）→ 尾检查发现残留 arm →
        raise（未消费 fail-closed；失败后账本原样保留，不静默吞）。"""
        harness = self._emit_admission_only()
        train = self._tmax_split_train("batch_train_i1_1")
        with self.assertRaises(RuntimeError):
            harness.builder.emit_train_remainder(train)
        self.assertIn(REQUEST_R, harness.builder._prefill_remote_read_arms)

    def test_first_step_request_mismatch_fail_closed(self):
        """首 chunk 成员与账本请求错配：错配 rid 空弹（双账本不消费）、
        真实请求账本残留 → completion 残留检查 raise（错配不被静默
        吞，§三.8 "request 错配 fail-closed"）。"""
        harness = self._emit_admission_only()
        train = self._tmax_split_train(
            "batch_train_i1_1", first_chunk_request="req_other")
        harness.builder.emit_train_first_step(train)
        self.assertIn(REQUEST_R, harness.builder._prefill_remote_read_arms)
        self.assertIn(REQUEST_R, harness.builder._prefill_remote_read_layers)
        harness.builder._block_ends[REQUEST_R] = {
            "seg2": {rank: 0 for rank in EXEC_RANKS}}
        harness.builder.next_plan[REQUEST_R] = None
        with self.assertRaises(RuntimeError):
            harness.builder.emit_completion_batch(_completion_plan())

    def test_group_size_matches_shared_planner(self):
        """组切分与 KV 管理器公共规划器同参（RESTORE_GROUP_LAYERS，
        §三.4）：夹具组区间 == plan_layer_groups 输出。"""
        ranges = plan_layer_groups(0, PREFIX_LAYERS)
        self.assertEqual(ranges, ((0, 8),))
        for start, end in ranges:
            self.assertLessEqual(end - start, RESTORE_GROUP_LAYERS)


if __name__ == "__main__":
    unittest.main()
