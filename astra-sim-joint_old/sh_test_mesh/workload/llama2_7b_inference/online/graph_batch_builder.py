#!/usr/bin/env python3
"""graph_batch_builder.py -- sh_3.0 在线 GraphBatch 构图器（方案 §4 步骤 1-8 操作 4）。

阶段 1 最关键的对齐点：复用共享发射原语的节点结构。per-request 发射
（generate_face_trace.py 模块级发射助手函数）组成
（_emit_kv_transfer / _emit_tp_readiness_barrier /
_emit_tp_point_to_point_readiness_barrier / transformer_pass_aggregated）——
本模块直接 import 它们，用 OnlineTraceBuilder（与 TraceBuilder 同构的在线侧
builder）驱动，保证节点属性、插入顺序、rank ownership 跨 request 链一致
（节点级审计口径）。

sh_3.0 发射结构（拼 batch 改造，2026-08-22；原两段式（准入+prefill 整段 /
decode 整段）的列车化重构，设计文档《层次 B Continuous Batching 改造》§3.2）：
  - 准入动作（ARRIVAL 边界，emit_admission_batch）= 到达/interval gate →
    history_evictions → history_transfer（含 partial 流水恢复全部节点）→
    prefill remote-read 前缀读流旁挂分支（2026-09-25 规格书§三：与后缀
    池恢复并行、同一准入 frontier 分叉，interval gate 触发链恰一次消费 +
    数据 tile 随消费窗在列车头发射、双 tile 缓冲 lag-2 复用与
    _prefill_read_window_tiles 体块覆盖账本）→
    prefill_evictions → prefill readiness barrier（结构性主链节点）；prefill 主体不再在此发射；
  - KV 逐出并行化（2026-09-13）：history/prefill/decode 三类逐出发射循环
    经 _emit_side_branch 包成旁路支链（fork 自各 rank 当前 frontier，触发门
    挂分支首节点，不 join 回主链）——逐出物理传输与主链上的推理计算并行，
    HBM 带宽由 C++ LocalHbmBandwidthModel 的 N-way 均分在线裁决；"同会话
    store 池写先于其 restore 池读"的前递保障改由 pending_store_tails 登记
    + 回迁发射前 _arm_pending_store_edges 补边承担（同缘 arm / 跨缘 1B
    p2p 中继）；prefill_decode_transfer 等恢复类迁移保持主链不动；
  - 迭代列车（各决策边界，emit_iteration_train）= joiner 的 decode_evictions
    （触发门 = drain 列车 barrier）→ prefill→decode 迁移 →（有 joiner 时）
    共享 readiness barrier → join/pstart 标记 → 折叠列车体（成员×迭代 span，
    weight_passes=迭代数：权重每迭代只读一次）→ drain/exit 标记节点（挂
    PREFILL_DRAIN / DECODE_COMPLETION watch）→ 每列车一个共享 end barrier；
  - 完成批（REQUEST_COMPLETE 边界，emit_completion_batch）=
    completion_evictions + 下一 turn interval gate 的依赖登记。

在线语义差异（刻意，注释标注）：
  - timer gate 始终发射节点（结构保留）但 runtime_ns=0：到达时间由 C++ 的
    arrival alarm（future_alarms）替代；
  - timer gate 的 duration 语义与共享参考同参（duration=admission_time_ns /
    interval + hbm_wait_ns；0 时不发射 gate 节点，在线同款跳过）；
  - pending_history / deferred_session_locations 账本与共享调度语义同构
    （跨请求的 arrival/interval gate 关联）；
  - partial 流水恢复的 chain_checkpoint/restore_chain 段内分支并行机制
    原样支持（拼 batch 改造起 body 移入列车，suffix 恢复完成门经
    _suffix_body_arms 账本挂到含其首 chunk 的列车体，见
    emit_iteration_train 的 first_chunk_member 处理）；
  - strategy 保持物理链与真实 MEM/HBM 物理时长；
  - watch 锚点与共享指标口径一致（拼 batch 改造起由列车标记承载）：
    PREFILL_DRAIN = drain 标记节点（列车体后、end barrier 前）；DECODE_
    COMPLETION/REQUEST_COMPLETE = exit 标记节点（同位置）。触发门角色
    （joiner decode_evictions 的 node_gates、completion_evictions 触发门、
    下一 turn interval gate 的 after_node_id、块末账本）仍用 post-barrier
    节点（列车 end barrier；五仓一致口径，不随 watch 锚点变化）。
"""

import os
import sys
from collections.abc import Sequence

# --------------------------------------------------------------------------
# import 路径：本文件位于 workload/llama2_7b_inference/online/，共享发射与
# 配置模块在上一级。路径只做 import 用途（红线：generate_face_trace.py /
# face_scheduler.py 只读 import 与注释）。
# --------------------------------------------------------------------------
# v4 G-line: cache-miss sentinel for the per-batch pipeline-group identity
# cache (a cached None result is a legitimate lookup outcome).
_GROUP_MISS = object()

_ONLINE_DIR = os.path.dirname(os.path.abspath(__file__))
_WORKLOAD_DIR = os.path.dirname(_ONLINE_DIR)
for _path in (_ONLINE_DIR, _WORKLOAD_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from generate_trace import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    transformer_pass_aggregated,
)
from generate_face_trace import (  # noqa: E402
    PendingHistoryGate,
    TransferTagAllocator,
    TransferTriggerGate,
    _emit_kv_transfer,
    _emit_tp_point_to_point_readiness_barrier,
    _emit_tp_readiness_barrier,
    _emit_transfer_trigger,
    _history_control,
    reconcile_pending_history_location,
    sanitize_node_prefix,
)
from joint.pipeline_identity import (  # noqa: E402
    PipelineIdentityAllocator,
    PipelineIdentitySeed,
)
from joint.compact_pipeline import PhysicalPipelineDraft  # noqa: E402


class CreditGateRange:
    """Lazy contiguous tile-gate run for one request.

    The scheduler stores one range per request/body intersection instead of
    one ``(request_id, tile)`` tuple per tile. Consumers that still need the
    historical pairs can iterate this object; compact emitters can inspect
    ``first_tile`` and ``count`` directly.
    """

    __slots__ = ("request_id", "first_tile", "count")

    def __init__(self, request_id, first_tile, count):
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("credit gate request_id must be nonempty")
        if type(first_tile) is not int or first_tile <= 0:
            raise ValueError("credit gate first_tile must be positive")
        if type(count) is not int or count <= 0:
            raise ValueError("credit gate count must be positive")
        self.request_id = request_id
        self.first_tile = first_tile
        self.count = count

    @property
    def last_tile(self):
        return self.first_tile + self.count - 1

    def __len__(self):
        return self.count

    def __iter__(self):
        for offset in range(self.count):
            yield self.request_id, self.first_tile + offset

    def __contains__(self, item):
        if not isinstance(item, tuple) or len(item) != 2:
            return False
        request_id, tile = item
        return (request_id == self.request_id
                and type(tile) is int
                and self.first_tile <= tile <= self.last_tile)

    def __eq__(self, other):
        if isinstance(other, CreditGateRange):
            return (self.request_id, self.first_tile, self.count) == (
                other.request_id, other.first_tile, other.count)
        if isinstance(other, Sequence):
            if len(other) != self.count:
                return False
            return all(pair == expected for pair, expected in zip(self, other))
        return NotImplemented

    def __repr__(self):
        return (f"CreditGateRange({self.request_id!r}, "
                f"{self.first_tile}, {self.count})")


class CreditGateSet(Sequence):
    """Immutable structural runs with a lazy flat-pair sequence view."""

    __slots__ = ("ranges", "_count")

    def __init__(self, ranges=()):
        copied = tuple(ranges)
        if any(not isinstance(run, CreditGateRange) for run in copied):
            raise TypeError("CreditGateSet only stores CreditGateRange values")
        self.ranges = copied
        self._count = sum(len(run) for run in copied)

    def __len__(self):
        return self._count

    def __iter__(self):
        for run in self.ranges:
            yield from run

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return tuple(self[i] for i in range(start, stop, step))
        if type(index) is not int:
            raise TypeError("credit gate index must be an integer")
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        for run in self.ranges:
            if index < len(run):
                return run.request_id, run.first_tile + index
            index -= len(run)
        raise IndexError(index)

    def request_ids(self):
        return frozenset(run.request_id for run in self.ranges)

    def contains(self, request_id, tile):
        return any(run.request_id == request_id
                   and run.first_tile <= tile <= run.last_tile
                   for run in self.ranges)

    def __eq__(self, other):
        if isinstance(other, CreditGateSet):
            return self.ranges == other.ranges
        if isinstance(other, Sequence):
            if len(other) != self._count:
                return False
            return all(pair == expected for pair, expected in zip(self, other))
        return NotImplemented

    def __repr__(self):
        return f"CreditGateSet({self.ranges!r})"


def _gate_runs(gates):
    """Return compact runs for new plans and normalize legacy pair lists."""
    if isinstance(gates, CreditGateSet):
        return gates.ranges
    result = []
    for request_id, tile in gates or ():
        if (result and result[-1].request_id == request_id
                and result[-1].last_tile + 1 == tile):
            prior = result.pop()
            result.append(CreditGateRange(
                request_id, prior.first_tile, prior.count + 1))
        else:
            result.append(CreditGateRange(request_id, tile, 1))
    return tuple(result)


def _iter_gate_pairs(gates):
    if isinstance(gates, CreditGateSet):
        yield from gates
        return
    yield from (gates or ())


def _gate_request_ids(gates):
    if isinstance(gates, CreditGateSet):
        return gates.request_ids()
    return frozenset(request_id for request_id, _tile in gates or ())


def _gate_contains(gates, request_id, tile):
    if isinstance(gates, CreditGateSet):
        return gates.contains(request_id, tile)
    return (request_id, tile) in (gates or ())


def _clip_pipeline_tile_run(run, first_tile, tile_count):
    """Make one structural source/pass extent for an emitted subrange."""
    first_tile = int(first_tile)
    tile_count = int(tile_count)
    run_first = int(run["first_tile"])
    run_stop = run_first + int(run["tile_count"])
    if tile_count <= 0 or first_tile < run_first \
            or first_tile + tile_count > run_stop:
        raise ValueError("pipeline tile subrange lies outside its source run")
    offset_units = (first_tile - run_first) * int(run["units_per_tile"])
    width_units = tile_count * int(run["units_per_tile"])
    clipped = dict(run)
    clipped["first_tile"] = first_tile
    clipped["tile_count"] = tile_count
    if run["axis_unit"] == "source_tokens":
        total_units = min(width_units,
                          int(run["total_units"]) - offset_units)
        clipped_shards = []
        for shard in run["shards"]:
            row = dict(shard)
            row["total_data_bytes"] = (
                total_units * int(row["bytes_per_unit"]))
            clipped_shards.append(row)
    else:
        clipped_shards = []
        shard_totals = []
        for shard in run["shards"]:
            row = dict(shard)
            remaining = max(0, int(row["total_data_bytes"]) - offset_units)
            total_bytes = min(width_units, remaining)
            row["total_data_bytes"] = total_bytes
            clipped_shards.append(row)
            shard_totals.append(total_bytes)
        total_units = min(width_units, max(shard_totals, default=0))
    if total_units <= 0:
        raise ValueError("pipeline emitted subrange carries no axis units")
    tail_units = total_units - (tile_count - 1) * int(run["units_per_tile"])
    if not 0 < tail_units <= int(run["units_per_tile"]):
        raise ValueError("clipped pipeline tail units are inconsistent")
    clipped["total_units"] = total_units
    clipped["tail_units"] = tail_units
    clipped["shards"] = tuple(clipped_shards)
    return clipped


def _pipeline_tile_run_segments(runs, first_tile, tile_count):
    """Yield source/pass extents split only where the active shard set changes.

    The group keeps the complete source/pass axis and all original shards.
    These shorter intervals are endpoint/tag run ranges inside that shared
    group; they do not create smaller groups or zero-byte endpoint slots.
    """
    first_tile = int(first_tile)
    stop_tile = first_tile + int(tile_count)
    cursor = first_tile
    for run in runs:
        run_first = int(run["first_tile"])
        run_stop = run_first + int(run["tile_count"])
        left = max(cursor, run_first)
        right = min(stop_tile, run_stop)
        if right <= left:
            continue
        if left != cursor:
            raise RuntimeError(
                "planner source/pass ranges do not cover emitted tiles")
        units_per_tile = int(run["units_per_tile"])
        shard_ends = []
        for shard in run["shards"]:
            if run["axis_unit"] == "source_tokens":
                active_tiles = (
                    -(-int(run["total_units"]) // units_per_tile)
                    if int(shard["bytes_per_unit"]) > 0 else 0)
            else:
                total_bytes = int(shard["total_data_bytes"])
                active_tiles = -(-total_bytes // units_per_tile) if total_bytes else 0
            end_tile = run_first + active_tiles
            if left < end_tile < right:
                shard_ends.append(end_tile)
        boundaries = [left, *sorted(set(shard_ends)), right]
        for begin, end in zip(boundaries, boundaries[1:]):
            active = []
            for shard in run["shards"]:
                if run["axis_unit"] == "source_tokens":
                    has_bytes = int(shard["bytes_per_unit"]) > 0
                    shard_end = run_first + (
                        -(-int(run["total_units"]) // units_per_tile)
                        if has_bytes else 0)
                else:
                    total_bytes = int(shard["total_data_bytes"])
                    shard_end = run_first + (
                        -(-total_bytes // units_per_tile) if total_bytes else 0)
                if begin >= shard_end:
                    continue
                active.append(shard)
            if not active:
                raise RuntimeError("emitted planner extent has no active shards")
            yield run, begin, end - begin, tuple(active)
        cursor = right
        if cursor == stop_tile:
            break
    if cursor != stop_tile:
        raise RuntimeError("planner source/pass ranges ended before emission")


def _active_shards_at(run, tile):
    """Return the original-order shards with nonzero bytes at one global tile."""
    tile = int(tile)
    first = int(run["first_tile"])
    units_per_tile = int(run["units_per_tile"])
    offset = (tile - first) * units_per_tile
    active = []
    for shard in run["shards"]:
        if run["axis_unit"] == "source_tokens":
            has_data = int(shard["bytes_per_unit"]) > 0
        else:
            has_data = offset < int(shard["total_data_bytes"])
        if has_data:
            active.append(shard)
    return tuple(active)


def _paired_pipeline_tile_segments(runs, first_consumer_tile, tile_count,
                                   *, lag=2):
    """Split consumer/successor pairs at both structural and tail boundaries.

    Each yielded range has one consumer source/pass group, one successor
    source/pass group, and fixed active shard sets on both sides. The groups
    themselves remain the full original planner extents.
    """
    first = int(first_consumer_tile)
    stop = first + int(tile_count)
    cursor = first
    ordered = tuple(runs)
    while cursor < stop:
        consumer = next((run for run in ordered
                         if int(run["first_tile"]) <= cursor <
                         int(run["first_tile"]) + int(run["tile_count"])),
                        None)
        successor_tile = cursor + lag
        successor = next((run for run in ordered
                          if int(run["first_tile"]) <= successor_tile <
                          int(run["first_tile"]) + int(run["tile_count"])),
                         None)
        if consumer is None or successor is None:
            raise RuntimeError(
                "planner ranges do not cover a consumer/successor pair")
        consumer_stop = (int(consumer["first_tile"])
                         + int(consumer["tile_count"]))
        successor_stop = (int(successor["first_tile"])
                          + int(successor["tile_count"]) - lag)
        end = min(stop, consumer_stop, successor_stop)
        consumer_units = int(consumer["units_per_tile"])
        successor_units = int(successor["units_per_tile"])
        for shard in consumer["shards"]:
            if consumer["axis_unit"] == "source_tokens":
                active_count = (-(-int(consumer["total_units"])
                                 // consumer_units)
                                if int(shard["bytes_per_unit"]) > 0 else 0)
            else:
                amount = int(shard["total_data_bytes"])
                active_count = -(-amount // consumer_units) if amount else 0
            shard_end = int(consumer["first_tile"]) + active_count
            if cursor < shard_end < end:
                end = shard_end
        for shard in successor["shards"]:
            if successor["axis_unit"] == "source_tokens":
                active_count = (-(-int(successor["total_units"])
                                 // successor_units)
                                if int(shard["bytes_per_unit"]) > 0 else 0)
            else:
                amount = int(shard["total_data_bytes"])
                active_count = -(-amount // successor_units) if amount else 0
            shard_end = int(successor["first_tile"]) + active_count - lag
            if cursor < shard_end < end:
                end = shard_end
        if end <= cursor:
            raise RuntimeError("consumer/successor segmentation made no progress")
        consumer_active = _active_shards_at(consumer, cursor)
        successor_active = _active_shards_at(successor, cursor + lag)
        if not consumer_active or not successor_active:
            raise RuntimeError(
                "consumer/successor extent contains a zero-byte tile")
        yield (consumer, successor, cursor, end - cursor,
               consumer_active, successor_active)
        cursor = end

def first_token_split_enabled() -> bool:
    """WP9 首步批拆分总开关（SH_FIRST_TOKEN_SPLIT，B4 起缺省 "0" 关）。

    缺省翻转（2026-08-26，主规格 §1.6 A 类处置）：B3_S3 60s 决策等价
    门-2 失败——拆分的物理扰动（首列拆批 → tick 漂移 → 闭环逐轮放大 →
    决策边界穿越 → 实例选择翻转，tput +13.2%）使 ON/OFF 字节等价在
    60s 窗不可达（2s 门保持）。证据：/tmp/slo_wps/b3/S3/t3_full_off_split
    与 /tmp/slo_wps/gates/B3_S3.FAILED（wp9_gate_60s）。默认口径退回
    proxy（first_token_source=train_interpolated，见 metrics_postprocess），
    本开关显式置 "1" 仍可启用拆分取 exact 首 token（研究/对拍用）。

    "1" = 拆分开启：debut 列车两段式发射 + first_token 标记；
    其他值（含缺省）= 拆分完全关闭：调度器不拆列车、不发射 first_token
    标记，构图/决策/账本产物与拆分上线前逐字节一致（A/B 对拍的 OFF
    侧）。每次 _emit_train 现场读取（而非构造期缓存），测试与复跑可在
    进程内切换。
    """
    return os.environ.get("SH_FIRST_TOKEN_SPLIT", "0") == "1"


class _CreditPendingRange:
    """One lazy lag-2 tail range; no per-tile transfer/name dictionary."""

    __slots__ = (
        "blocks", "first_block", "count", "first_pending_offset",
        "next_block", "action_base", "prefix", "stage", "session_id",
        "kind")

    def __init__(self, blocks, *, first_pending_offset, action_base,
                 prefix, stage, session_id, kind):
        self.blocks = blocks
        self.first_block = int(blocks[0][0])
        self.count = len(blocks)
        self.first_pending_offset = int(first_pending_offset)
        self.next_block = self.first_block + self.first_pending_offset
        self.action_base = int(action_base)
        self.prefix = prefix
        self.stage = stage
        self.session_id = session_id
        self.kind = kind

    @property
    def last_block(self):
        return self.first_block + self.count - 1

    @property
    def pending_count(self):
        return max(0, self.last_block - self.next_block + 1)

    def has(self, block_index):
        return self.next_block <= int(block_index) <= self.last_block

    def take(self, block_index):
        block_index = int(block_index)
        if not self.has(block_index):
            return None
        if block_index != self.next_block:
            raise RuntimeError(
                "credit pending range released out of order: expected tile "
                "{} but got {}".format(self.next_block, block_index))
        offset = block_index - self.first_block
        recorded_index, transfer = self.blocks[offset]
        if recorded_index != block_index:
            raise RuntimeError("credit pending range has a non-contiguous tile index")
        name = (
            f"{self.prefix}_{self.stage}_action"
            f"{self.action_base + offset:03d}_"
            f"{sanitize_node_prefix(self.session_id)}_{self.kind}")
        self.next_block += 1
        return transfer, name

    def transfer_at(self, block_index):
        block_index = int(block_index)
        if not self.first_block <= block_index <= self.last_block:
            raise IndexError(block_index)
        recorded_index, transfer = self.blocks[block_index - self.first_block]
        if recorded_index != block_index:
            raise RuntimeError(
                "credit pending range has a non-contiguous tile index")
        return transfer

    def segments(self, first_block, count):
        """Return source/pass intersections without making tile objects."""
        first_block = int(first_block)
        count = int(count)
        if count <= 0 or first_block != self.next_block:
            raise RuntimeError(
                "credit pending range segment must begin at its next tile")
        if first_block + count - 1 > self.last_block:
            raise RuntimeError("credit pending range segment exceeds its tail")
        runs = getattr(self.blocks, "pipeline_tile_runs", ())
        if not runs:
            raise RuntimeError(
                "compact credit release requires planner source/pass ranges")
        return tuple((run, left, width, active)
                     for run, left, width, active
                     in _pipeline_tile_run_segments(runs, first_block, count))

    def advance(self, count):
        count = int(count)
        if count <= 0 or self.next_block + count - 1 > self.last_block:
            raise RuntimeError("invalid credit pending range advance")
        self.next_block += count

    def __bool__(self):
        return self.pending_count > 0

    def __iter__(self):
        return iter(range(self.next_block, self.last_block + 1))

    def __len__(self):
        return self.pending_count

    def describe_pending(self):
        if not self.pending_count:
            return "empty"
        return "{}..{} ({} tiles)".format(
            self.next_block, self.last_block, self.pending_count)


class _CreditArmRange:
    """Bounded data-recv anchors for a structural tile range."""

    __slots__ = ("entries", "coverage", "mainline")

    def __init__(self):
        self.entries = []
        self.coverage = []
        self.mainline = []

    def add_extent(self, first_tile, tile_count, target_ranks):
        first_tile = int(first_tile)
        tile_count = int(tile_count)
        ranks = tuple(sorted(set(int(rank) for rank in target_ranks)))
        if first_tile <= 0 or tile_count <= 0 or not ranks:
            raise ValueError("credit arm extent is invalid")
        self.coverage.append((first_tile, first_tile + tile_count - 1, ranks))

    def covers(self, first_tile, last_tile, rank):
        first_tile = int(first_tile)
        last_tile = int(last_tile)
        rank = int(rank)
        cursor = first_tile
        for left, right, ranks in sorted(self.coverage):
            if right < cursor or rank not in ranks:
                continue
            if left > cursor:
                return False
            cursor = max(cursor, right + 1)
            if cursor > last_tile:
                return True
        return cursor > last_tile

    def add_mainline_extent(self, first_tile, tile_count, target_ranks):
        first_tile = int(first_tile)
        tile_count = int(tile_count)
        ranks = tuple(sorted(set(int(rank) for rank in target_ranks)))
        if first_tile <= 0 or tile_count <= 0 or not ranks:
            raise ValueError("credit mainline extent is invalid")
        self.mainline.append((first_tile, first_tile + tile_count - 1, ranks))

    def mainline_covers(self, tile, rank):
        tile = int(tile)
        rank = int(rank)
        return any(left <= tile <= right and rank in ranks
                   for left, right, ranks in self.mainline)

    def add(self, tile, per_rank):
        tile = int(tile)
        if tile <= 0 or not per_rank:
            return
        normalized = {
            int(rank): ((int(node_id),) if type(node_id) is int else
                        tuple(int(item) for item in node_id))
            for rank, node_id in per_rank.items()}
        for index, (prior_tile, prior) in enumerate(self.entries):
            if prior_tile == tile:
                merged = dict(prior)
                for rank, node_ids in normalized.items():
                    merged[rank] = tuple(dict.fromkeys(
                        (*merged.get(rank, ()), *node_ids)))
                self.entries[index] = (tile, merged)
                return
        self.entries.append((tile, normalized))

    def get(self, tile):
        tile = int(tile)
        for entry_tile, per_rank in reversed(self.entries):
            if entry_tile == tile:
                return {rank: (nodes[0] if len(nodes) == 1 else nodes)
                        for rank, nodes in per_rank.items()}
        return None

    def dependencies(self, first_tile, last_tile):
        first_tile = int(first_tile)
        last_tile = int(last_tile)
        result = {}
        for tile, per_rank in self.entries:
            if first_tile <= tile <= last_tile:
                for rank, node_ids in per_rank.items():
                    result.setdefault(int(rank), []).extend(node_ids)
        return {rank: tuple(dict.fromkeys(node_ids))
                for rank, node_ids in result.items()}

    def __bool__(self):
        return bool(self.entries)

    def __iter__(self):
        return (tile for tile, _per_rank in self.entries)

    def __len__(self):
        return len(self.entries)

    def describe(self):
        if not self.entries:
            return "empty"
        return "{}..{} ({} bounded anchors)".format(
            min(tile for tile, _map in self.entries),
            max(tile for tile, _map in self.entries), len(self.entries))


class _IndexedTransferBlocks:
    """Lazy `(block_number, transfer)` range over a transfer sequence."""

    __slots__ = ("transfers", "first_offset", "first_block", "count",
                 "pipeline_tile_runs")

    def __init__(self, transfers, *, first_offset, first_block):
        self.transfers = transfers
        self.first_offset = int(first_offset)
        self.first_block = int(first_block)
        self.count = max(0, len(transfers) - self.first_offset)
        self.pipeline_tile_runs = tuple(
            dict(run) for run in getattr(transfers,
                                         "pipeline_tile_runs", ()))

    def __len__(self):
        return self.count

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self.count)
            return tuple(self[i] for i in range(start, stop, step))
        index = int(index)
        if index < 0:
            index += self.count
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.first_block + index, self.transfers[self.first_offset + index]

    def __iter__(self):
        for index in range(self.count):
            yield self[index]


class _TransferPayloadSequence:
    """O(1) adapter for transfer streams with or without numbered entries."""

    __slots__ = ("source", "pipeline_tile_runs")

    def __init__(self, source):
        self.source = source
        self.pipeline_tile_runs = tuple(
            dict(run) for run in getattr(source, "pipeline_tile_runs", ()))

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        value = self.source[index]
        if (isinstance(value, tuple) and len(value) == 2
                and type(value[0]) is int
                and hasattr(value[1], "shards")):
            return value[1]
        return value

    def __iter__(self):
        for index in range(len(self)):
            yield self[index]


class OnlineTraceBuilder:
    """与共享 TraceBuilder 接口同构的在线侧每-rank builder。

    同一接口面：timer_gate / arm_timer_gate / arm_dependency /
    chain_checkpoint / restore_chain / comp / all_reduce / comm_send /
    comm_recv / mem_store / mem_load / local_hbm_kv_restore / next_id /
    previous_id / node_count——共享助手函数可直接驱动。

    与 TraceBuilder 的差异：节点发射为 GraphBatch nodes[] dict（而非
    ChakraNode 对象），依赖记录为 parent_edges[]（共享 builder 的 data_deps
    内联在节点里，在线按边列表携带）；timer_gate 忽略 duration
    （runtime_ns=0，alarm 替代等待）。

    per-rank id 自 0 起全局递增，跨批次保持连续。
    """

    def __init__(self, rank: int, *, remote_operand_loads: bool):
        self.rank = rank
        self.remote_operand_loads = remote_operand_loads
        self.next_id = 0
        self.previous_id = None
        self.pending_extra_dependencies = []
        self._checkpoints = {}
        # M1 收集即释放（2026-08-29）：这两个 list 只保存自上次 _collect
        # 以来尚未交付的记录；交付后立即 clear，跨批状态只由 next_id、
        # previous_id 与账本保存。禁止按 list 位置回读节点（id 来自
        # next_id 计数器，与位置无关）。
        self.nodes = []   # 本 rank 尚未交付的节点 dict（发射序）
        self.edges = []   # 本 rank 尚未交付的 parent edges
        self.node_count = 0
        # Logical IDs occupied by compact remote-read endpoints.  These IDs
        # are represented by descriptor runs, not ordinary node payloads;
        # any later ordinary parent must therefore land on a structural
        # physical anchor instead of one of these holes.
        # Sparse affine holes, not a Python set of one ID per tile.
        self.virtual_endpoint_ranges = []
        # 当前 request-stage 反向索引上下文（每次 per-request 发射前设置）。
        self.request_id = ""
        self.stage = ""
        self.generation = 0

    def set_context(self, request_id: str, stage: str, generation: int) -> None:
        self.request_id = request_id
        self.stage = stage
        self.generation = generation

    # ------------------------------------------------------------- 节点发射 --

    def _new_node(self, name: str, node_type: int, *, is_cpu_op: bool = False,
                  is_timer_op: bool = False) -> dict:
        node = {
            "rank": self.rank,
            "id": self.next_id,
            "name": name,
            "type": node_type,
            "is_cpu_op": is_cpu_op,
            "is_timer_op": is_timer_op,
            "inputs_values": "",
            "request_id": self.request_id,
            "stage": self.stage,
            "generation": self.generation,
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "mem": {"tensor_size": 0, "is_local_hbm_kv_restore": False,
                    "hbm_access_mode": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0,
                     "hbm_charge": True},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        previous_id = self.previous_id
        pending_dependencies = self.pending_extra_dependencies
        if (self.is_virtual_endpoint(previous_id)
                or any(self.is_virtual_endpoint(dep)
                       for dep in pending_dependencies)):
            raise RuntimeError(
                "ordinary node depends on a compact virtual endpoint; the "
                "producer must preserve/use its bounded physical frontier")
        # The common serial chain has no extra dependency: avoid allocating a
        # temporary list and deduplication dict.  The multi-dependency fallback
        # deliberately keeps dict.fromkeys() for its stable first-seen order.
        if not pending_dependencies:
            if previous_id is not None:
                self.edges.append({
                    "rank": self.rank,
                    "from": previous_id,
                    "to": self.next_id,
                    "kind": "data",
                })
        elif len(pending_dependencies) == 1:
            if previous_id is not None:
                self.edges.append({
                    "rank": self.rank,
                    "from": previous_id,
                    "to": self.next_id,
                    "kind": "data",
                })
            dependency_id = pending_dependencies[0]
            if previous_id is None or dependency_id != previous_id:
                self.edges.append({
                    "rank": self.rank,
                    "from": dependency_id,
                    "to": self.next_id,
                    "kind": "data",
                })
        else:
            dependency_ids = []
            if previous_id is not None:
                dependency_ids.append(previous_id)
            dependency_ids.extend(pending_dependencies)
            for dependency_id in dict.fromkeys(dependency_ids):
                self.edges.append({
                    "rank": self.rank,
                    "from": dependency_id,
                    "to": self.next_id,
                    "kind": "data",
                })
        pending_dependencies.clear()
        self.previous_id = self.next_id
        self.next_id += 1
        self.nodes.append(node)
        self.node_count += 1
        return node

    def is_virtual_endpoint(self, node_id) -> bool:
        if node_id is None:
            return False
        node_id = int(node_id)
        for first, count, stride in self.virtual_endpoint_ranges:
            delta = node_id - first
            if delta >= 0 and delta % stride == 0 and delta // stride < count:
                return True
        return False

    def mark_virtual_endpoint_progression(self, first_id: int, count: int,
                                          stride: int) -> None:
        """Register a descriptor hole as one arithmetic progression."""
        if (type(first_id) is not int or first_id < 0
                or type(count) is not int or count <= 0
                or type(stride) is not int or stride <= 0):
            raise ValueError("virtual endpoint progression is invalid")
        end_id = first_id + (count - 1) * stride
        if end_id >= self.next_id:
            raise ValueError("virtual endpoint progression exceeds reserved IDs")
        if self.virtual_endpoint_ranges:
            prior_first, prior_count, prior_stride = self.virtual_endpoint_ranges[-1]
            expected = prior_first + prior_count * prior_stride
            if count == 1 and prior_count == 1 and first_id > prior_first:
                self.virtual_endpoint_ranges[-1] = (
                    prior_first, 2, first_id - prior_first)
                return
            if (prior_count == 1 and first_id > prior_first
                    and first_id == expected):
                self.virtual_endpoint_ranges[-1] = (
                    prior_first, 2, first_id - prior_first)
                return
            if (first_id == expected and stride == prior_stride):
                self.virtual_endpoint_ranges[-1] = (
                    prior_first, prior_count + count, prior_stride)
                return
        self.virtual_endpoint_ranges.append((first_id, count, stride))

    def reserve_endpoint_extent(self, count: int) -> int:
        """Reserve a contiguous logical ID extent without per-ID objects.

        The caller then registers virtual APs and materializes only bounded
        physical anchors from within that extent.
        """
        if type(count) is not int or count <= 0:
            raise ValueError("endpoint extent count must be positive")
        first = self.next_id
        self.next_id += count
        self.node_count += count
        self.previous_id = self.next_id - 1
        self.pending_extra_dependencies.clear()
        return first

    def reserve_virtual_endpoint(self) -> int:
        """Compatibility helper for a singleton endpoint extent."""
        node_id = self.reserve_endpoint_extent(1)
        self.mark_virtual_endpoint_progression(node_id, 1, 1)
        return node_id

    def emit_reserved_pipeline_anchor(self, name: str, *, segment_id: str,
                                      role: str, node_id: int,
                                      legacy_context: dict) -> int:
        """Materialize an anchor at an already reserved ID without advancing
        the rank allocator a second time.
        """
        if role not in ("start", "frontier"):
            raise ValueError("pipeline physical anchor role must be start/frontier")
        if (type(node_id) is not int or node_id < 0
                or node_id >= self.next_id):
            raise ValueError("pipeline anchor ID must be inside a reserved extent")
        context = dict(legacy_context)
        if set(context) != {"request_id", "stage", "generation"}:
            raise ValueError("pipeline anchor legacy context has invalid keys")
        node = {
            "rank": self.rank,
            "id": node_id,
            "name": name,
            "type": COMP_NODE,
            "is_cpu_op": False,
            "is_timer_op": False,
            "inputs_values": "",
            "request_id": context["request_id"],
            "stage": context["stage"],
            "generation": context["generation"],
            "compute": {
                "num_ops": 0, "tensor_size": 0, "runtime_ns": 0,
                "exec_role": "remote_read_pipeline_anchor",
                "pipeline_anchor": {"segment_id": segment_id,
                                    "role": role},
            },
            "mem": {"tensor_size": 0, "is_local_hbm_kv_restore": False,
                    "hbm_access_mode": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0,
                     "hbm_charge": True},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        self.nodes.append(node)
        return node_id

    def emit_pipeline_anchor(self, name: str, *, segment_id: str,
                             role: str) -> int:
        """Materialize a zero-cost COMP at one endpoint slot selected as an
        execution frontier.  The descriptor dispatches the original endpoint
        (including body-marker cost) through this structural node.
        """
        if role not in ("start", "frontier"):
            raise ValueError("pipeline physical anchor role must be start/frontier")
        node = {
            "rank": self.rank,
            "id": self.next_id,
            "name": name,
            "type": COMP_NODE,
            "is_cpu_op": False,
            "is_timer_op": False,
            "inputs_values": "",
            "request_id": self.request_id,
            "stage": self.stage,
            "generation": self.generation,
            "compute": {
                "num_ops": 0, "tensor_size": 0, "runtime_ns": 0,
                "exec_role": "remote_read_pipeline_anchor",
                "pipeline_anchor": {"segment_id": segment_id,
                                    "role": role},
            },
            "mem": {"tensor_size": 0, "is_local_hbm_kv_restore": False,
                    "hbm_access_mode": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0,
                     "hbm_charge": True},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        node_id = self.next_id
        # The endpoint's causal parents live in the descriptor and are gated
        # by RemoteReadExecutor.  A normal parent edge here could illegally
        # target a sparse virtual endpoint.
        self.pending_extra_dependencies.clear()
        self.next_id += 1
        self.previous_id = node_id
        self.nodes.append(node)
        self.node_count += 1
        return node_id

    @staticmethod
    def _uint64(value: int) -> int:
        return max(1, int(value))

    def arm_timer_gate(self, timer_node_id) -> None:
        if timer_node_id is not None:
            self.pending_extra_dependencies.append(int(timer_node_id))

    # 与共享 TraceBuilder.arm_dependency 同构。
    def arm_dependency(self, node_id) -> None:
        if node_id is not None:
            self.pending_extra_dependencies.append(int(node_id))

    # 与共享 TraceBuilder.chain_checkpoint/restore_chain
    # 同构（sh_2.0 在线版同款完整恢复）：
    # partial 流水恢复的段内分支并行机制（prefix 计算与 suffix 恢复并行）。
    # checkpoint 同时捕获 previous_id 与 pending_extra_dependencies，
    # restore 两者一并回滚——分支内新 arm 的依赖不泄漏到恢复点之后。
    # 返回 checkpoint 句柄。（2026-08-20 前在线版只存/回 previous_id，
    # 因唯一调用点 checkpoint 时 pending_extra_dependencies 恒被
    # readiness barrier 清空而无实际差异；补齐为完整同构以消除潜伏
    # 分叉，对现行行为零变化。）
    def chain_checkpoint(self):
        handle = len(self._checkpoints)
        self._checkpoints[handle] = (
            self.previous_id, tuple(self.pending_extra_dependencies))
        return handle

    def restore_chain(self, handle) -> None:
        self.previous_id, dependencies = self._checkpoints.pop(handle)
        self.pending_extra_dependencies = list(dependencies)

    def timer_gate(self, name: str, duration_ns: int, *,
                   after_node_id=None):
        """在线 timer gate：发射节点（runtime_ns=0，is_timer_op）。

        与共享 TraceBuilder.timer_gate 同构：
        - duration_ns == 0 → 直接返回 after_node_id，不发射节点
            （共享接口同款；interval==0 的 request 不产生 interval gate
            节点）；
          - 否则直接创建节点（不经 _new_node）——不链 previous_id、不消费
            pending_extra_dependencies、不更新 previous_id；仅
            after_node_id 依赖（interval gate 依赖上一 request 完成
            barrier 节点）。
        调度参考语义（duration = admission/interval 等待）由 C++ arrival alarm
        （future_alarms）替代——gate 只保留结构与依赖，保持时长会双重等待。
        L15（2026-09-29）：保留整微秒校验不变（K5/§9-低6 已让全部调用点
        显式 µs 下取整，本断言是离线同构守卫；duration 不进节点，C++ 消费
        端仅经 0 跳过判定读它），仅把报错改为可定位口径（含字段名与值）。
        """
        if duration_ns < 0 or duration_ns % 1000 != 0:
            raise ValueError(
                "timer duration must be a non-negative whole number of "
                "microseconds (duration_ns must be µs-aligned; callers round "
                "via duration -= duration % 1000): got {} for node "
                "'{}'".format(duration_ns, name)
            )
        if duration_ns == 0:
            return after_node_id
        node = {
            "rank": self.rank,
            "id": self.next_id,
            "name": name,
            "type": COMP_NODE,
            "is_cpu_op": True,
            "is_timer_op": True,
            "inputs_values": "",
            "request_id": self.request_id,
            "stage": self.stage,
            "generation": self.generation,
            "compute": {"num_ops": 0, "tensor_size": 0, "runtime_ns": 0},
            "mem": {"tensor_size": 0, "is_local_hbm_kv_restore": False,
                    "hbm_access_mode": 0},
            "comm": {"bytes": 0, "src": 0, "dst": 0, "tag": 0,
                     "hbm_charge": True},
            "coll": {"comm_type": 0, "bytes": 0, "priority": 0,
                     "pg_name": "", "involved_dim": []},
        }
        if after_node_id is not None:
            self.edges.append({
                "rank": self.rank,
                "from": int(after_node_id),
                "to": self.next_id,
                "kind": "data",
            })
        self.next_id += 1
        self.nodes.append(node)
        self.node_count += 1
        return node["id"]

    def comp(self, name: str, num_ops: int, tensor_size: int,
             remote_read_size: int = 0, *,
             die_spec: dict = None, op_role: str = None) -> None:
        """COMP 节点发射（die 后端第一期扩展，契约 §2 compute 键集）。

        die_spec 非 None 时附 compute.die_op_spec（dict；die 后端执行该
        节点，runtime_ns 保持缺省 0——die_op_spec 存在 ⇒ runtime_ns 必为
        0）；op_role 缺省 "compute"（不写键，schema 缺省即 compute），仅
        marker/哨兵显式写 "control_marker"（契约 §7 裁定 8）。"""
        node = self._new_node(name, COMP_NODE)
        node["compute"]["num_ops"] = self._uint64(num_ops)
        node["compute"]["tensor_size"] = self._uint64(tensor_size)
        if op_role is not None:
            if op_role not in ("compute", "control_marker"):
                raise ValueError(
                    "op_role must be 'compute' or 'control_marker', got "
                    "{!r}".format(op_role))
            node["compute"]["op_role"] = op_role
        if die_spec is not None:
            if node["compute"]["runtime_ns"] != 0:
                raise ValueError(
                    "die_op_spec requires runtime_ns == 0 (contract "
                    "section 2)")
            node["compute"]["die_op_spec"] = die_spec
        if self.remote_operand_loads and remote_read_size:
            node["compute"]["remote_weight_bytes"] = self._uint64(remote_read_size)

    def all_reduce(self, name: str, comm_size: int, pg_name: str) -> None:
        node = self._new_node(name, COMM_COLL_NODE)
        node["coll"]["comm_type"] = ALL_REDUCE
        node["coll"]["bytes"] = self._uint64(comm_size)
        node["coll"]["priority"] = 0
        node["coll"]["pg_name"] = pg_name
        node["coll"]["involved_dim"] = [True, True]

    def comm_send(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_SEND_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        node["comm"]["hbm_charge"] = bool(hbm_charge)

    def comm_recv(self, name: str, *, src: int, dst: int, comm_size: int,
                  comm_tag: int, hbm_charge: bool = True) -> None:
        node = self._new_node(name, COMM_RECV_NODE)
        node["comm"]["src"] = int(src)
        node["comm"]["dst"] = int(dst)
        node["comm"]["bytes"] = self._uint64(comm_size)
        node["comm"]["tag"] = int(comm_tag)
        node["comm"]["hbm_charge"] = bool(hbm_charge)

    def mem_store(self, name: str, tensor_size: int, *,
                  hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_STORE_NODE)
        node["mem"]["tensor_size"] = self._uint64(tensor_size)
        node["mem"]["hbm_access_mode"] = int(hbm_access_mode)
        node["compute"]["tensor_size"] = node["mem"]["tensor_size"]

    def mem_load(self, name: str, tensor_size: int, *,
                 hbm_access_mode: int = 0) -> None:
        node = self._new_node(name, MEM_LOAD_NODE)
        node["mem"]["tensor_size"] = self._uint64(tensor_size)
        node["mem"]["hbm_access_mode"] = int(hbm_access_mode)
        node["compute"]["tensor_size"] = node["mem"]["tensor_size"]

    def local_hbm_kv_restore(self, name: str, tensor_size: int) -> None:
        """发射目标-HBM DMA 写节点（is_local_hbm_kv_restore 路由）。"""
        node = self._new_node(name, MEM_LOAD_NODE)
        node["mem"]["tensor_size"] = self._uint64(tensor_size)
        node["mem"]["is_local_hbm_kv_restore"] = True
        node["compute"]["tensor_size"] = node["mem"]["tensor_size"]

    # ------------------------------------------------------------- 只读属性 --


class GraphBatchBuilder:
    """sh_3.0 在线构图器：持有 per-rank OnlineTraceBuilder（状态跨批次），
    按决策边界发射准入动作批 / 实例迭代列车 / completion 批，并维护
    pending_history / completion gates 账本（与离线 writer 同构）。"""

    def __init__(self, config, *, digest_sink=None):
        self.config = config
        # Run-scoped candidate/emission descriptor identity allocator. Candidate
        # graphs only preview ordinals; an accepted physical GraphBatch commits
        # them at its finalized publication point.
        self.pipeline_identity_allocator = PipelineIdentityAllocator()
        # Compact physical descriptor state is bounded to the current
        # publication batch. OnlineTraceBuilder retains virtual membership as
        # per-rank affine progressions rather than one ID per tile.
        self._pipeline_drafts = {}
        self._pipeline_credit_slots = {}
        # v4 G-line: per-batch identity cache for _physical_pipeline_group
        # (repeated same-(plan, stage, owner, run) credit-tile lookups skip
        # rebuilding the content key; objects live through the batch).
        self._pipeline_group_ids = {}
        self._pipeline_plan_by_request = {}
        self._pipeline_start_reserved = set()
        # die 后端发射语境（第一期；None = legacy）。online_service.py 构造
        # 本构图器时不感知 backend——语境由 OnlineSchedulerBase 从 manifest
        # 的 compute-cost-model/die-model-config 键构造后在每批发射前经
        # set_die_context 注入（online_scheduler_base._begin_graph_batch；
        # manifest 是 backend 的唯一声明源，契约 §7 裁定 2/7）。
        self.die_context = None
        # strategy 模式保持物理跨 request 链。
        self.builders = {
            rank: OnlineTraceBuilder(
                rank, remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
        self.group_by_index = dict(enumerate(config.inference_groups))
        self.tag_allocator = TransferTagAllocator()
        # 跨请求 history gate 账本（request_id → PendingHistoryGate）。
        self.pending_history = {}
        self.pending_request_by_session = {}
        self.deferred_session_locations = {}
        # request_id -> {"seg1": {rank: id}, "seg2": {rank: id}}
        # 列车块末账本（post-barrier 口径；emitted-ranks-only 语义）：seg1 =
        # 覆盖其最后 prefill chunk 的列车 end barrier（joiner decode_evictions
        # 触发门），seg2 = 覆盖其退出迭代的列车 end barrier（completion_
        # evictions 触发门 + 下一 turn interval gate after_node_id）。不再
        # 恢复进 builders——frontier 接续裁决 2026-08-19，见 emit_iteration_train。
        self._block_ends = {}
        # request_id -> {rank: suffix 恢复完成门节点 id}（partial 流水恢复
        # 的 body 依赖账本，拼 batch 改造起首 chunk 主体移入列车：恢复分支
        # 仍在准入批发射（与 suffix 恢复并行的语义改为经此账本挂到含其首
        # chunk 的列车体），见 emit_iteration_train 的 first_chunk_member。
        self._suffix_body_arms = {}
        # C15 逐组恢复门（2026-09-22，设计方案 §5.1）：request_id ->
        # ((group_index, layer_start, layer_end, {rank: 该 rank 该组目标
        # HBM 写完成门节点 id}), ...)——后缀逐组恢复支链（rank 内消费顺
        # 序串行链）的就绪门登记；列车体按层段门控消费（首 chunk 的逐层
        # 段各自等其组门，替换"整列车等整段后缀"的列车级保守门——
        # C12-G7/E17 登记的缺口闭合）。旧单笔口径（restore_group 无标记
        # 的 remote_load）仍走 _suffix_body_arms 整段门（回归锚）。
        self._suffix_restore_arms = {}
        # KV 逐出并行化（2026-09-13）：store→restore 前递依赖登记表——
        # session_id → 该会话全部在飞 remote_store 逐出支链的尾部
        # (edge_rank, edge mem_store 节点 id, 源端 ack_recv 节点 id) 列表。
        # 同会话两段式逐出（suffix + full fallback）逐 shard 分别登记
        # （restore 读全区间，须等齐全部在飞 store）；回迁发射时统一补边
        # 消费（_arm_pending_store_edges），会话终结时清除（_emit_completion
        # 的 terminal 分支）。
        self.pending_store_tails = {}
        # remote-read credit（唯一执行口径）：request_id ->
        # {切片块号 b: {rank: 该 rank 上块 b 的 recv 完成门节点 id}}——
        # 旁挂支链发射的尾块（块 2..M）流式到达语义的 body 依赖账本
        # （_suffix_body_arms 同款手法：支链登记、体块消费点 arm）。列车
        # 尾标记发射时清账（每列车恰一次——整列/余量批同路径）。
        self._credit_arms = {}
        # D09 有限字节双缓冲（2026-09-26，v2.0 §12）：request_id ->
        # {tile 块号 b: (KVTransfer, action_name)}——切片块 3..M 的待发
        # 账本。块 t 的生产受缓冲复用释放门约束（消费后归还 credit）：
        # 块 t 的旁挂支链在**消费块 t-2 的体块发射点**才发射（fork =
        # 该体块末节点 = 消费锚；exec 侧 tile recv 挂直接父边、home 侧
        # 经释放 ack tag 配对承载跨 rank 因果）——慢计算不导致源端无限
        # 泵出，任意时刻远读暂存 ≤2 块。列车头只发射块 2（第二缓冲并
        # 行预取）；体发射逐块释放；尾标记结清（残留 fail-closed）。
        # request_id -> one lazy lag-2 range (not a per-tile dict). The range
        # points at a bounded source plan and computes transfer/action names
        # only when a consumed tile releases its successor.
        self._credit_pending_tiles = {}
        # C13 copy 逐 chunk 交接（2026-09-22，设计文档 §2.3 四步协议）：
        # request_id -> {交接块号 c: {rank: 该 rank 上块 c 的 recv 完成
        # 门节点 id}}——准入批发射的尾块（块 1..M-1）旁挂支链，流式到达
        # 语义与 _credit_arms 同款；含首 chunk 的列车体按消费顺序逐块
        # arm（I2 门并集同源），块 0 在准入主链（readiness barrier 只等
        # 它——不设"先整份搬运后计算"串行段）。列车体发射时消费（joiner
        # 恰一次），残留即 fail-closed。
        self._copy_handoff_arms = {}
        # K6（P1-③，2026-09-23 外部审计）：request_id -> {交接块号 c:
        # (layer_start, layer_end)}——尾块层区间账本（首体块层段门控的
        # 段边界来源；生命周期与 _copy_handoff_arms 同步：发射登记 /
        # 体块消费弹出 / 完成残留 fail-closed）。
        self._copy_handoff_layers = {}
        # C13：request_id -> {交接块号 c: {home_rank: 该 rank 上块 c 的
        # ack_recv 节点 id}}——交接完成事件在源端的物理挂点（source
        # release dependency = noc_migration_ack_recv；FS 账本侧的逐块
        # home 释放即以此为到达事实锚）。completion 批弹出（审计面）。
        self._copy_handoff_release_anchors = {}
        # D09 prefill 远读逐 tile 读流（2026-09-27）：request_id -> 本列车
        # 读流窗口已登记的 tile 序号集（列车头窗口发射处登记、列车体门
        # 并集覆盖检查 + 列车尾结清弹出；残留 = 窗口发射/消费链破损，
        # fail-closed）。tile 物理节点与完成门复用 decode credit 的
        # _credit_arms/_credit_pending_tiles 账本（同一双缓冲释放协议）。
        self._prefill_read_window_tiles = {}
        # D10 静默重建落执行图（v2.0 §6.2，2026-09-27）：本列车在飞发射
        # 的静默重建 span 欠账（span 值 → 待匹配数）——joiner plan 携带
        # joint_suppress_output=True + joint_prefill_stage="context_rebuild"
        # 时，其 (H_B, H_B) 重建 span 的输出侧副作用（final norm/logits
        # 输出位、debut first_token 观测标记）被抑制，注意力/MLP/KV 算子
        # 时间照常聚合（重建计算量真实存在）。发射期公共方法开账
        # （_open/_consume/_close），体块发射点按 span 值贪心匹配消费，
        # 尾检查全量核销（fail-closed）。
        self._rebuild_output_debt = {}
        self.batch = None  # 当前批次累加器（由 begin_batch 建立）
        # WP9 首步批标记（2026-08-26）：本批是否含首步批发射（digests 行
        # first_step=True 标记的来源；OFF 侧恒 False，不进任何产物）。
        self.batch_first_step = False
        # request_id -> 跨阶段连续的 action 序号（离线 writer 的 per-
        # request action_sequence 闭包在列车发射下的等价物：准入/列车/
        # completion 批共享同一计数器，保证 actionNNN 命名与逐字节稳定
        # ——canonical 命名对照前提）。
        self._action_seq = {}
        # request_id -> 下一 turn 的 request_id / plan dict / None。每个
        # 当前 request 只会在 completion 发射时读取一次，随后立即释放。
        self.next_plan = {}
        self._plan_resolver = lambda request_id: None

    # ------------------------------------------------------------- 批次 --

    def set_die_context(self, die_context) -> None:
        """注入 die 发射语境（幂等；None 恢复 legacy 发射）。"""
        self.die_context = die_context

    def begin_batch(self) -> None:
        self.batch = {
            "nodes": [],
            "parent_edges": [],
            "pipeline_descriptors": [],
            # Private exact ledger: _collect already knows the source rank of
            # every appended node, so downstream GraphBatch metadata need not
            # rescan the complete node payload.
            "_touched_ranks": set(),
            "watches": [],
            "assignments": [],
            "kv_actions": [],
            "future_alarms": [],
        }
        self.batch_first_step = False
        self._pipeline_identity_seeds = []
        self._pipeline_identity_seed_by_key = {}
        self._pipeline_identity_preview = {}
        self._pipeline_identity_finalized = False
        self._pipeline_drafts = {}
        self._pipeline_credit_slots = {}
        self._pipeline_group_ids = {}
        self._pipeline_plan_by_request = {}
        self._pipeline_start_reserved = set()

    def pipeline_identity_for(self, request_plan: dict, stage: str,
                              rank: int):
        """Preview the explicit actual-emission seed for one owner rank.

        The request plan carries stage epoch and allocation IDs from the
        scheduler's live lease. Previewing is side-effect free; the ordinal
        advances only from finalize_pipeline_identities after the enclosing
        GraphBatch is completely built.
        """
        if stage not in ("prefill", "decode"):
            raise ValueError("pipeline stage must be prefill or decode")
        matching = [
            row for row in request_plan.get("pipeline_identity_seeds", ())
            if row.get("stage") == stage and int(row.get("rank", -1)) == int(rank)
        ]
        if len(matching) != 1:
            raise RuntimeError(
                "actual remote-read emission requires exactly one explicit "
                f"{stage} identity seed for rank {rank}")
        row = matching[0]
        seed = PipelineIdentitySeed(
            request_id=str(row["request_id"]),
            stage=str(row["stage"]),
            generation=row["generation"],
            stage_epoch=row["stage_epoch"],
            rank=row["rank"],
            staging_allocation_id=row["staging_allocation_id"],
        )
        key = (seed.request_id, seed.stage, seed.generation,
               seed.stage_epoch, seed.rank)
        prior = self._pipeline_identity_seed_by_key.get(key)
        if prior is not None and prior != seed:
            raise RuntimeError(
                "one pipeline identity key carries multiple staging "
                "allocation IDs in a GraphBatch")
        if prior is None:
            self._pipeline_identity_seed_by_key[key] = seed
            self._pipeline_identity_seeds.append(seed)
            identities = self.pipeline_identity_allocator.preview_batch(
                self._pipeline_identity_seeds)
            self._pipeline_identity_preview = {
                (item.request_id, item.stage, item.generation,
                 item.stage_epoch, item.rank): item
                for item in identities}
        return self._pipeline_identity_preview[key]

    def finalize_pipeline_identities(self) -> None:
        """Commit actual descriptor ordinals once the GraphBatch is finalized."""
        if self._pipeline_identity_finalized:
            return
        committed = self.pipeline_identity_allocator.commit_batch(
            self._pipeline_identity_seeds)
        expected = tuple(
            self._pipeline_identity_preview[
                (seed.request_id, seed.stage, seed.generation,
                 seed.stage_epoch, seed.rank)]
            for seed in self._pipeline_identity_seeds)
        if committed != expected:
            raise RuntimeError(
                "pipeline identity preview changed before GraphBatch "
                "finalization")
        for draft in self._pipeline_drafts.values():
            self.batch["pipeline_descriptors"].append(
                draft.finish().to_wire())
        self._pipeline_identity_finalized = True

    def _physical_pipeline_draft(self, request_plan, stream_stage, owner_rank):
        stream_kind = ("prefill_read_window" if stream_stage == "prefill_read"
                       else "decode_remote_credit")
        identity_stage = "prefill" if stream_stage == "prefill_read" else "decode"
        identity = self.pipeline_identity_for(
            request_plan, identity_stage, owner_rank)
        key = identity.segment_id
        draft = self._pipeline_drafts.get(key)
        if draft is None:
            draft = PhysicalPipelineDraft(
                identity=identity, stream_kind=stream_kind)
            self._pipeline_drafts[key] = draft
        elif draft.stream_kind != stream_kind:
            raise RuntimeError("one pipeline identity changed physical stream kind")
        return key, draft

    def _physical_pipeline_group(self, request_plan, stream_stage,
                                 owner_rank, tile_run):
        """Get/create the complete rank-local source/pass group.

        v4 G-line: an identity-keyed per-batch cache fronts the content-key
        rebuild (the shard filter + key tuple dominate repeated credit-tile
        release lookups).  request_plan/tile_run live for the whole batch, so
        id() keys are stable; the cache resets in begin_batch.  Construction
        lives in _physical_pipeline_group_build.
        """
        identity_key = (
            id(request_plan), stream_stage, int(owner_rank), id(tile_run))
        cached = self._pipeline_group_ids.get(identity_key, _GROUP_MISS)
        if cached is not _GROUP_MISS:
            return cached
        slot = self._physical_pipeline_group_build(
            request_plan, stream_stage, owner_rank, tile_run)
        self._pipeline_group_ids[identity_key] = slot
        return slot

    def _physical_pipeline_group_build(self, request_plan, stream_stage,
                                       owner_rank, tile_run):
        """Original rank-local source/pass group construction.

        Short-shard tails split endpoint/tag runs only. The descriptor group
        retains the planner's original shared tile axis and every shard routed
        to this owner, including shards that become inactive before the axis
        ends.
        """
        segment_id, draft = self._physical_pipeline_draft(
            request_plan, stream_stage, owner_rank)
        shards = tuple(
            dict(shard) for shard in tile_run["shards"]
            if int(shard["target_rank"]) == int(owner_rank))
        if not shards:
            return None
        # Group identity follows the source/pass boundary, not the release
        # call that happened to expose a portion of this range.
        group_key = (
            segment_id, int(tile_run["pass_or_window_ordinal"]),
            int(tile_run["source_ordinal"]), str(tile_run["source_key"]),
            int(tile_run["first_tile"]), int(tile_run["tile_count"]),
            int(tile_run["layer_start"]), int(tile_run["layer_end"]),
            str(tile_run["axis_unit"]), int(tile_run["total_units"]),
            int(tile_run["units_per_tile"]), int(tile_run["tail_units"]),
            tuple((int(shard["global_shard_ordinal"]),
                   int(shard["source_rank"]), int(shard["target_rank"]),
                   int(shard.get("total_data_bytes", -1)),
                   int(shard.get("bytes_per_unit", -1)),
                   tuple(int(rank) for rank in shard["noc_path"]))
                  for shard in shards),
        )
        slot = self._pipeline_credit_slots.get(group_key)
        if slot is not None:
            return slot
        total_bytes = tuple(
            (int(tile_run["total_units"]) * int(shard["bytes_per_unit"])
             if tile_run["axis_unit"] == "source_tokens"
             else int(shard["total_data_bytes"]))
            for shard in shards)
        bytes_per_unit = tuple(
            int(shard.get("bytes_per_unit", 1)) for shard in shards)
        group_ordinal = draft.add_group_range(
            pass_or_window_ordinal=int(tile_run["pass_or_window_ordinal"]),
            source_ordinal=int(tile_run["source_ordinal"]),
            source_key=str(tile_run["source_key"]),
            layer_start=int(tile_run["layer_start"]),
            layer_end=int(tile_run["layer_end"]),
            global_tile_first=int(tile_run["first_tile"]),
            tile_count=int(tile_run["tile_count"]),
            axis_unit=str(tile_run["axis_unit"]),
            total_units=int(tile_run["total_units"]),
            units_per_tile=int(tile_run["units_per_tile"]),
            tail_units=int(tile_run["tail_units"]),
            owner_shards=shards,
            shard_total_bytes=total_bytes,
            shard_bytes_per_unit=bytes_per_unit)
        slot = {
            "segment_id": segment_id,
            "draft": draft,
            "group_ordinal": group_ordinal,
            "identity_rank": int(owner_rank),
            "tile_run": dict(tile_run),
            "owner_shards": shards,
            "global_to_local_shard": {
                int(shard["global_shard_ordinal"]): local
                for local, shard in enumerate(shards)},
        }
        self._pipeline_credit_slots[group_key] = slot
        return slot

    def _emit_compact_credit_extent(
            self, member_plan, clipped_run, transfer, *, stage,
            release_anchor, first_tile, tile_count, parent_by_rank=None):
        """Bulk-reserve one fixed-shard source/pass endpoint extent.

        This is called only at an old emission point (immediate tail or one
        body release point). The range has already been split at planner and
        per-shard short-tail boundaries, so each represented tile has the
        same endpoint/tag shape. No tile payloads or endpoint dictionaries
        are retained.
        """
        request_id = member_plan["request_id"]
        first_tile = int(first_tile)
        tile_count = int(tile_count)
        if tile_count <= 0:
            raise ValueError("compact credit extent must be nonempty")
        shards = tuple(clipped_run["shards"])
        if not shards:
            raise RuntimeError("compact credit extent has no active shards")
        expected_shards = tuple(transfer.shards)
        if len(expected_shards) != len(shards):
            raise RuntimeError(
                "planner active-shard extent disagrees with its first tile")
        for expected, planned in zip(expected_shards, shards):
            units = min(int(clipped_run["units_per_tile"]),
                        int(clipped_run["total_units"]))
            expected_bytes = (
                units * int(planned["bytes_per_unit"])
                if clipped_run["axis_unit"] == "source_tokens"
                else min(int(clipped_run["units_per_tile"]),
                         int(planned["total_data_bytes"])))
            if (int(expected.source_rank) != int(planned["source_rank"])
                    or int(expected.target_rank) != int(planned["target_rank"])
                    or tuple(int(rank) for rank in expected.noc_path)
                    != tuple(int(rank) for rank in planned["noc_path"])
                    or int(expected.bytes) != expected_bytes):
                raise RuntimeError(
                    "planner structural extent does not match the legacy "
                    "first transfer payload")

        stream_stage = "prefill_read" if stage == "prefill_read" else "remote_credit"
        owner_ranks = tuple(dict.fromkeys(
            int(shard["target_rank"]) for shard in shards))
        slots = {
            owner: self._physical_pipeline_group(
                member_plan, stream_stage, owner, clipped_run)
            for owner in owner_ranks}
        if any(slot is None for slot in slots.values()):
            raise RuntimeError("compact credit owner group was not created")

        tag_per_shard = 3 if release_anchor else 2
        tag_count = tile_count * len(shards) * tag_per_shard
        first_tag, emission_ordinal = self.tag_allocator.reserve_with_ordinal(
            tag_count)
        rank_events = {}
        tile_events = []
        if release_anchor:
            endpoint_pattern = (
                ("release_ack_send", 0), ("release_ack_recv", 0),
                ("data_send", 1), ("data_recv", 1),
                ("fill_ack_send", 2), ("fill_ack_recv", 2),
            )
        else:
            endpoint_pattern = (
                ("data_send", 0), ("data_recv", 0),
                ("fill_ack_send", 1), ("fill_ack_recv", 1),
            )
        for global_shard_index, shard in enumerate(shards):
            for endpoint, tag_in_shard in endpoint_pattern:
                source_side = endpoint in (
                    "data_send", "fill_ack_recv", "release_ack_recv")
                rank = int(shard["source_rank"] if source_side
                           else shard["target_rank"])
                event = {
                    "global_shard_index": global_shard_index,
                    "endpoint": endpoint,
                    "rank": rank,
                    "tag_in_shard": tag_in_shard,
                }
                tile_events.append(event)
                rank_events.setdefault(rank, []).append(event)

        captured_context = {
            rank: {
                "request_id": builder.request_id,
                "stage": builder.stage,
                "generation": builder.generation,
            }
            for rank, builder in self.builders.items()}
        if parent_by_rank is None:
            parent_by_rank = {
                rank: self.builders[rank].previous_id
                for rank in rank_events}
        else:
            parent_by_rank = {
                rank: parent_by_rank.get(rank, self.builders[rank].previous_id)
                for rank in rank_events}

        rank_bases = {}
        for rank, events in rank_events.items():
            rank_bases[rank] = self.builders[rank].reserve_endpoint_extent(
                tile_count * len(events))
            self.batch["_touched_ranks"].add(int(rank))

        # One original TransferTagAllocator take() span is shared by all
        # owner descriptors. The local group/shard mapping differs, while the
        # allocation identity and global tag formula stay common.
        for owner, slot in slots.items():
            slot["draft"].add_tag_allocation_reference(
                allocator_scope_id="graph-batch-transfer-tags-v1",
                emission_ordinal=emission_ordinal,
                first_tag=first_tag,
                count=tag_count,
                group_ordinal=slot["group_ordinal"],
                first_tile=1,
                tile_count=tile_count,
                first_shard=0,
                shard_count=len(slot["owner_shards"]))

        recv_anchors = {}
        owner_local_by_global = {
            owner: slot["global_to_local_shard"]
            for owner, slot in slots.items()}
        event_position = {
            (event["rank"], event["global_shard_index"], event["endpoint"]): pos
            for rank, events in rank_events.items()
            for pos, event in enumerate(events)}
        descriptor_hole_starts = {
            owner: len(slot["draft"]._holes) for owner, slot in slots.items()}
        descriptor_anchor_starts = {
            owner: len(slot["draft"].anchors) for owner, slot in slots.items()}

        # Plan the bounded anchor exceptions before registering dependency
        # kinds, because a parent that lands on one of these slots is an
        # ordinary physical terminal while other endpoint parents are holes.
        anchor_specs_by_event = {}
        for owner, slot in slots.items():
            owner_events = [event for event in tile_events
                            if int(shards[event["global_shard_index"]]["target_rank"])
                            == owner]
            start_event = owner_events[0]
            start_global_shard = start_event["global_shard_index"]
            start_local_shard = owner_local_by_global[owner][
                int(shards[start_global_shard]["global_shard_ordinal"])]
            anchor_specs_by_event.setdefault(
                (owner, start_global_shard, start_event["endpoint"]), []).append(
                    (1, start_local_shard, "start"))
            for global_shard_index, shard in enumerate(shards):
                if int(shard["target_rank"]) != owner:
                    continue
                local_shard = owner_local_by_global[owner][
                    int(shard["global_shard_ordinal"])]
                recv_tiles = (range(1, tile_count + 1) if tile_count <= 2
                              else range(tile_count - 1, tile_count + 1))
                for local_tile in recv_tiles:
                    anchor_specs_by_event.setdefault(
                        (owner, global_shard_index, "data_recv"), []).append(
                            (local_tile, local_shard, "frontier"))
                    event_pos = event_position[(
                        int(shard["target_rank"]), global_shard_index,
                        "data_recv")]
                    node_id = (rank_bases[int(shard["target_rank"])]
                               + (local_tile - 1)
                               * len(rank_events[int(shard["target_rank"])])
                               + event_pos)
                    recv_anchors.setdefault(local_tile, {}).setdefault(
                        owner, []).append(node_id)

        planned_anchor_ids = set()
        for (owner, global_shard_index, endpoint), specifications in \
                anchor_specs_by_event.items():
            rank = next(event["rank"] for event in tile_events
                        if event["global_shard_index"] == global_shard_index
                        and event["endpoint"] == endpoint)
            local_shard = owner_local_by_global[owner][
                int(shards[global_shard_index]["global_shard_ordinal"])]
            pos = event_position[(rank, global_shard_index, endpoint)]
            for local_tile, _shard_ordinal, _role in specifications:
                planned_anchor_ids.add((
                    rank, rank_bases[rank]
                    + (local_tile - 1) * len(rank_events[rank]) + pos))

        # The start selector is the first endpoint slot emitted for this
        # owner-domain group. The final two data_recv slots per source shard
        # remain bounded body frontiers; intermediate endpoints are holes.
        for owner, slot in slots.items():
            draft = slot["draft"]
            for event in tile_events:
                global_shard_index = event["global_shard_index"]
                shard = shards[global_shard_index]
                if int(shard["target_rank"]) != owner:
                    continue
                local_shard = owner_local_by_global[owner][
                    int(shard["global_shard_ordinal"])]
                rank = event["rank"]
                pos = event_position[(rank, global_shard_index,
                                      event["endpoint"])]
                anchor_exceptions = []
                specifications = anchor_specs_by_event.get(
                    (owner, global_shard_index, event["endpoint"]), ())
                for anchor_tile, anchor_shard, role in specifications:
                    if role == "start" and slot["segment_id"] in \
                            self._pipeline_start_reserved:
                        continue
                    anchor_exceptions.append((
                        anchor_tile, anchor_shard, event["endpoint"], role))
                run = {
                    "rank": rank,
                    "group_ordinal": slot["group_ordinal"],
                    "first_tile": 1,
                    "tile_count": tile_count,
                    "shard_ordinals": [local_shard],
                    "first_json_id": rank_bases[rank] + pos,
                    "tile_stride": len(rank_events[rank]),
                    "shard_stride": 1,
                    "initial_parent_json_id": None,
                    "legacy_context": dict(captured_context[rank]),
                    "legs": [{
                        "endpoint": event["endpoint"],
                        "json_offset": 0,
                        "tag_emission_ordinal": emission_ordinal,
                        "tag_offset": (
                            global_shard_index * tag_per_shard
                            + event["tag_in_shard"]),
                        "tag_tile_stride": len(shards) * tag_per_shard,
                        "tag_shard_stride": tag_per_shard,
                    }],
                }
                parent = (parent_by_rank[rank] if pos == 0 else
                          rank_bases[rank] + pos - 1)
                run["initial_parent_json_id"] = parent
                run_index = draft.add_endpoint_run(
                    run, anchor_exceptions=anchor_exceptions)
                if parent is not None:
                    if pos == 0:
                        dep_first = int(parent)
                        dep_stride = 0
                    else:
                        dep_first = rank_bases[rank] + pos - 1
                        dep_stride = len(rank_events[rank])
                    parent_is_anchor = (rank, dep_first) in planned_anchor_ids
                    parent_virtual = (
                        False if parent_is_anchor else
                        (self.builders[rank].is_virtual_endpoint(dep_first)
                         if pos == 0 else True))
                    draft.add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": slot["group_ordinal"],
                        "first_tile": 1,
                        "tile_count": tile_count,
                        "shard_ordinals": [local_shard],
                        "endpoint": event["endpoint"],
                        "dependency_kind": (
                            "virtual_endpoint_terminal"
                            if parent_virtual else "ordinary_node_terminal"),
                        "first_dependency_json_id": dep_first,
                        "dependency_tile_stride": dep_stride,
                        "dependency_shard_stride": 0,
                    })
            if slot["segment_id"] not in self._pipeline_start_reserved:
                self._pipeline_start_reserved.add(slot["segment_id"])

        # Add sparse endpoint membership only after all run anchor exceptions
        # are known. This makes ordinary builders reject any accidental edge
        # to an unmaterialized endpoint before the next ordinary node emits.
        for owner, slot in slots.items():
            draft = slot["draft"]
            for hole in draft._holes[descriptor_hole_starts[owner]:]:
                self.builders[int(hole["rank"])].mark_virtual_endpoint_progression(
                    int(hole["first_json_id"]), int(hole["count"]),
                    int(hole["stride"]))
        seen_anchor_keys = set()
        for owner, slot in slots.items():
            for anchor in slot["draft"].anchors[
                    descriptor_anchor_starts[owner]:]:
                key = (owner, anchor["rank"], anchor["node_id"])
                if key in seen_anchor_keys:
                    continue
                seen_anchor_keys.add(key)
                builder = self.builders[int(anchor["rank"])]
                draft = slot["draft"]
                selector = anchor["selector"]
                context = None
                for run in draft.endpoint_runs:
                    if (run["group_ordinal"] == selector.get("group_ordinal")
                            and run["rank"] == anchor["rank"]
                            and run["first_tile"] <= selector.get("tile_ordinal", 0)
                            < run["first_tile"] + run["tile_count"]
                            and selector.get("shard_ordinal")
                            in run["shard_ordinals"]
                            and any(leg["endpoint"] == selector.get("endpoint")
                                    for leg in run["legs"])):
                        context = run["legacy_context"]
                        break
                if context is None:
                    raise RuntimeError(
                        "pipeline anchor has no matching endpoint context")
                builder.emit_reserved_pipeline_anchor(
                    "{}_pipeline_{}_{}_{}".format(
                        sanitize_node_prefix(request_id), stage,
                        anchor["role"], anchor["node_id"]),
                    segment_id=str(slot["segment_id"]),
                    role=anchor["role"], node_id=int(anchor["node_id"]),
                    legacy_context=context)

        arm_range = self._credit_arms.setdefault(
            request_id, _CreditArmRange())
        for local_tile, per_rank in recv_anchors.items():
            arm_range.add(first_tile + local_tile - 1, per_rank)
        return {
            "first_tag": first_tag,
            "emission_ordinal": emission_ordinal,
            "tile_count": tile_count,
            "rank_bases": rank_bases,
            "slots": slots,
        }

    def _emit_compact_initial_range(
            self, member_plan, tile_run, transfer, active_shards, *, stage,
            first_tile, tile_count, parent_by_rank=None):
        """Bulk emit an un-released source/pass range at its real emit point."""
        request_id = member_plan["request_id"]
        first_tile = int(first_tile)
        tile_count = int(tile_count)
        active_shards = tuple(active_shards)
        if tile_count <= 0 or not active_shards:
            raise ValueError("initial compact extent must contain active tiles")
        first_offset = ((first_tile - int(tile_run["first_tile"]))
                        * int(tile_run["units_per_tile"]))
        units = min(int(tile_run["units_per_tile"]),
                    int(tile_run["total_units"]) - first_offset)
        expected = []
        for shard in active_shards:
            if tile_run["axis_unit"] == "source_tokens":
                amount = units * int(shard["bytes_per_unit"])
            else:
                amount = max(0, min(
                    int(shard["total_data_bytes"]), first_offset + units)
                    - first_offset)
            if amount:
                expected.append((int(shard["source_rank"]),
                                 int(shard["target_rank"]),
                                 tuple(int(rank) for rank in shard["noc_path"]),
                                 amount))
        actual = [(int(shard.source_rank), int(shard.target_rank),
                   tuple(int(rank) for rank in shard.noc_path),
                   int(shard.bytes)) for shard in transfer.shards]
        if expected != actual:
            raise RuntimeError(
                "compact initial range disagrees with its frozen transfer "
                "factory at the first tile")

        stream_stage = "prefill_read" if stage == "prefill_read" else "remote_credit"
        owner_ranks = tuple(dict.fromkeys(
            int(shard["target_rank"]) for shard in active_shards))
        slots = {
            owner: self._physical_pipeline_group(
                member_plan, stream_stage, owner, tile_run)
            for owner in owner_ranks}
        local_first = first_tile - int(tile_run["first_tile"]) + 1
        tag_per_shard = 2
        tag_count = tile_count * len(active_shards) * tag_per_shard
        first_tag, emission_ordinal = self.tag_allocator.reserve_with_ordinal(
            tag_count)
        endpoint_pattern = (
            ("data_send", 0), ("data_recv", 0),
            ("fill_ack_send", 1), ("fill_ack_recv", 1),
        )
        events = []
        active_index = {}
        for index, shard in enumerate(active_shards):
            global_ordinal = int(shard["global_shard_ordinal"])
            active_index[global_ordinal] = index
            owner = int(shard["target_rank"])
            slot = slots[owner]
            local_shard = slot["global_to_local_shard"][global_ordinal]
            for endpoint, tag_stage in endpoint_pattern:
                source_side = endpoint in ("data_send", "fill_ack_recv")
                rank = int(shard["source_rank"] if source_side
                           else shard["target_rank"])
                events.append({
                    "rank": rank, "owner": owner, "endpoint": endpoint,
                    "tag_stage": tag_stage, "shard": shard,
                    "local_shard": local_shard, "slot": slot,
                })
        events_by_rank = {}
        positions = {}
        for event in events:
            rank = int(event["rank"])
            positions[id(event)] = len(events_by_rank.setdefault(rank, []))
            events_by_rank[rank].append(event)
        checkpoint = {rank: builder.previous_id
                      for rank, builder in self.builders.items()}
        if parent_by_rank is None:
            parent_by_rank = checkpoint
        else:
            parent_by_rank = {
                rank: parent_by_rank.get(rank, checkpoint[rank])
                for rank in events_by_rank}
        for rank in events_by_rank:
            if self.builders[rank].pending_extra_dependencies:
                raise RuntimeError(
                    "compact initial range encountered unconsumed ordinary "
                    "dependencies")

        rank_bases = {}
        cycles = {}
        for rank, rank_events in events_by_rank.items():
            cycles[rank] = len(rank_events)
            rank_bases[rank] = self.builders[rank].reserve_endpoint_extent(
                tile_count * len(rank_events))
            self.batch["_touched_ranks"].add(rank)
        for owner, slot in slots.items():
            slot["draft"].add_tag_allocation_reference(
                allocator_scope_id="graph-batch-transfer-tags-v1",
                emission_ordinal=emission_ordinal,
                first_tag=first_tag, count=tag_count,
                group_ordinal=slot["group_ordinal"],
                first_tile=local_first, tile_count=tile_count,
                first_shard=0, shard_count=len(slot["owner_shards"]))

        anchor_specs = {}
        planned_anchor_ids = set()
        for owner, slot in slots.items():
            owner_events = [event for event in events
                            if int(event["owner"]) == owner]
            if not owner_events:
                raise RuntimeError("owner group has no emitted endpoint")
            if slot["segment_id"] not in self._pipeline_start_reserved:
                event = owner_events[0]
                pos = positions[id(event)]
                node_id = rank_bases[int(event["rank"])] + pos
                anchor_specs.setdefault(id(event), []).append(
                    (local_first, int(event["local_shard"]),
                     event["endpoint"], "start"))
                planned_anchor_ids.add((int(event["rank"]), node_id))
            for event in owner_events:
                if event["endpoint"] != "data_recv":
                    continue
                rank = int(event["rank"])
                pos = positions[id(event)]
                for local_tile in range(max(0, tile_count - 2), tile_count):
                    node_id = rank_bases[rank] + local_tile * cycles[rank] + pos
                    anchor_specs.setdefault(id(event), []).append(
                        (local_first + local_tile,
                         int(event["local_shard"]), event["endpoint"],
                         "frontier"))
                    planned_anchor_ids.add((rank, node_id))

        # Preserve the exact old rank-chain tail at this emission point for
        # ordinary nodes that follow the transfer (for example P's pstart
        # marker). The tail can be an ACK endpoint, so retain that one bounded
        # terminal per physical rank even when it is not a data_recv frontier.
        terminal_by_rank = {}
        for event in events:
            terminal_by_rank[int(event["rank"])] = event
        for rank, event in terminal_by_rank.items():
            pos = positions[id(event)]
            node_id = (rank_bases[rank]
                       + (tile_count - 1) * cycles[rank] + pos)
            anchor_specs.setdefault(id(event), []).append(
                (local_first + tile_count - 1,
                 int(event["local_shard"]), event["endpoint"], "frontier"))
            planned_anchor_ids.add((rank, node_id))

        hole_starts = {id(slot["draft"]): len(slot["draft"]._holes)
                       for slot in slots.values()}
        anchor_starts = {id(slot["draft"]): len(slot["draft"].anchors)
                         for slot in slots.values()}
        for event in events:
            slot = event["slot"]
            rank = int(event["rank"])
            pos = positions[id(event)]
            cycle = cycles[rank]
            parent = (int(parent_by_rank[rank]) if pos == 0
                      and parent_by_rank[rank] is not None
                      else (None if pos == 0 else rank_bases[rank] + pos - 1))
            active_pos = active_index[
                int(event["shard"]["global_shard_ordinal"])]
            tag_offset = active_pos * tag_per_shard + int(event["tag_stage"])
            run = {
                "rank": rank,
                "group_ordinal": int(slot["group_ordinal"]),
                "first_tile": local_first,
                "tile_count": tile_count,
                "shard_ordinals": [int(event["local_shard"])],
                "first_json_id": rank_bases[rank] + pos,
                "tile_stride": cycle,
                "shard_stride": 1,
                "initial_parent_json_id": parent,
                "legacy_context": {
                    "request_id": self.builders[rank].request_id,
                    "stage": self.builders[rank].stage,
                    "generation": self.builders[rank].generation,
                },
                "legs": [{
                    "endpoint": event["endpoint"], "json_offset": 0,
                    "tag_emission_ordinal": emission_ordinal,
                    "tag_offset": tag_offset,
                    "tag_tile_stride": len(active_shards) * tag_per_shard,
                    "tag_shard_stride": 1,
                }],
            }
            slot["draft"].add_endpoint_run(
                run, anchor_exceptions=anchor_specs.get(id(event), ()))
            if pos == 0 and tile_count > 1:
                # The first emitted tile inherits the old chain checkpoint,
                # which may legitimately be empty. Every later tile starts
                # from the previous rank-local cycle tail in either case.
                if parent is not None:
                    checkpoint_kind = (
                        "virtual_endpoint_terminal"
                        if self.builders[rank].is_virtual_endpoint(parent)
                        else "ordinary_node_terminal")
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": local_first,
                        "tile_count": 1,
                        "shard_ordinals": [int(event["local_shard"])],
                        "endpoint": event["endpoint"],
                        "dependency_kind": checkpoint_kind,
                        "first_dependency_json_id": int(parent),
                        "dependency_tile_stride": 0,
                        "dependency_shard_stride": 0,
                    })
                slot["draft"].add_endpoint_dependency_run({
                    "rank": rank,
                    "group_ordinal": int(slot["group_ordinal"]),
                    "first_tile": local_first + 1,
                    "tile_count": tile_count - 1,
                    "shard_ordinals": [int(event["local_shard"])],
                    "endpoint": event["endpoint"],
                    "dependency_kind": "virtual_endpoint_terminal",
                    "first_dependency_json_id": rank_bases[rank] + cycle - 1,
                    "dependency_tile_stride": cycle,
                    "dependency_shard_stride": 0,
                })
                continue
            if parent is not None:
                if pos == 0:
                    dep_first = parent
                    dep_stride = 0
                else:
                    dep_first = rank_bases[rank] + pos - 1
                    dep_stride = cycle
                exceptional = set()
                if dep_stride:
                    for anchor_rank, anchor_id in planned_anchor_ids:
                        if anchor_rank == rank and anchor_id >= dep_first:
                            delta = anchor_id - dep_first
                            if delta % dep_stride == 0:
                                offset = delta // dep_stride
                                if offset < tile_count:
                                    exceptional.add(offset)
                default_kind = (
                    "virtual_endpoint_terminal" if pos
                    else ("virtual_endpoint_terminal"
                          if self.builders[rank].is_virtual_endpoint(parent)
                          else "ordinary_node_terminal"))
                if pos == 0 and tile_count > 1:
                    checkpoint_kind = default_kind
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": local_first,
                        "tile_count": 1,
                        "shard_ordinals": [int(event["local_shard"])],
                        "endpoint": event["endpoint"],
                        "dependency_kind": checkpoint_kind,
                        "first_dependency_json_id": dep_first,
                        "dependency_tile_stride": 0,
                        "dependency_shard_stride": 0,
                    })
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": local_first + 1,
                        "tile_count": tile_count - 1,
                        "shard_ordinals": [int(event["local_shard"])],
                        "endpoint": event["endpoint"],
                        "dependency_kind": "virtual_endpoint_terminal",
                        "first_dependency_json_id": (
                            rank_bases[rank] + cycle - 1),
                        "dependency_tile_stride": cycle,
                        "dependency_shard_stride": 0,
                    })
                    continue
                cursor = 0
                for offset in sorted(exceptional):
                    if offset > cursor:
                        slot["draft"].add_endpoint_dependency_run({
                            "rank": rank,
                            "group_ordinal": int(slot["group_ordinal"]),
                            "first_tile": local_first + cursor,
                            "tile_count": offset - cursor,
                            "shard_ordinals": [int(event["local_shard"])],
                            "endpoint": event["endpoint"],
                            "dependency_kind": default_kind,
                            "first_dependency_json_id": dep_first
                                + cursor * dep_stride,
                            "dependency_tile_stride": dep_stride,
                            "dependency_shard_stride": 0,
                        })
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": local_first + offset,
                        "tile_count": 1,
                        "shard_ordinals": [int(event["local_shard"])],
                        "endpoint": event["endpoint"],
                        "dependency_kind": "ordinary_node_terminal",
                        "first_dependency_json_id": dep_first
                            + offset * dep_stride,
                        "dependency_tile_stride": 0,
                        "dependency_shard_stride": 0,
                    })
                    cursor = offset + 1
                if cursor < tile_count:
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": local_first + cursor,
                        "tile_count": tile_count - cursor,
                        "shard_ordinals": [int(event["local_shard"])],
                        "endpoint": event["endpoint"],
                        "dependency_kind": default_kind,
                        "first_dependency_json_id": dep_first
                            + cursor * dep_stride,
                        "dependency_tile_stride": dep_stride,
                        "dependency_shard_stride": 0,
                    })

        for slot in slots.values():
            draft = slot["draft"]
            for hole in draft._holes[hole_starts[id(draft)]:]:
                self.builders[int(hole["rank"])].mark_virtual_endpoint_progression(
                    int(hole["first_json_id"]), int(hole["count"]),
                    int(hole["stride"]))
        anchors_to_emit = []
        seen = set()
        for slot in slots.values():
            draft = slot["draft"]
            for anchor in draft.anchors[anchor_starts[id(draft)]:]:
                key = (anchor["rank"], anchor["node_id"])
                if key in seen:
                    continue
                seen.add(key)
                selector = anchor["selector"]
                context = next((run["legacy_context"]
                                for run in draft.endpoint_runs
                                if run["group_ordinal"]
                                    == selector.get("group_ordinal")
                                and run["rank"] == anchor["rank"]
                                and run["first_tile"]
                                    <= selector.get("tile_ordinal", 0)
                                    < run["first_tile"] + run["tile_count"]
                                and selector.get("shard_ordinal")
                                    in run["shard_ordinals"]
                                and any(leg["endpoint"]
                                        == selector.get("endpoint")
                                        for leg in run["legs"])), None)
                if context is None:
                    raise RuntimeError("compact anchor has no endpoint context")
                anchors_to_emit.append((anchor, context, slot["segment_id"]))
        anchors_to_emit.sort(key=lambda item: (item[0]["rank"], item[0]["node_id"]))
        for anchor, context, segment_id in anchors_to_emit:
            self.builders[int(anchor["rank"])].emit_reserved_pipeline_anchor(
                "{}_pipeline_{}_{}_{}".format(
                    sanitize_node_prefix(request_id), stage,
                    anchor["role"], anchor["node_id"]),
                segment_id=str(segment_id), role=anchor["role"],
                node_id=int(anchor["node_id"]), legacy_context=context)
        for rank in rank_bases:
            self.builders[rank].previous_id = (
                rank_bases[rank] + tile_count * cycles[rank] - 1)

        arm_range = self._credit_arms.setdefault(request_id, _CreditArmRange())
        arm_range.add_extent(first_tile, tile_count, owner_ranks)
        recv_anchors = {}
        for event in events:
            if event["endpoint"] != "data_recv":
                continue
            rank = int(event["rank"])
            owner = int(event["owner"])
            shard_ordinal = int(event["shard"]["global_shard_ordinal"])
            pos = positions[id(event)]
            for offset in range(max(0, tile_count - 2), tile_count):
                node_id = rank_bases[rank] + offset * cycles[rank] + pos
                per_owner = recv_anchors.setdefault(first_tile + offset, {})
                prior = per_owner.get(owner)
                if prior is None or shard_ordinal > prior[0]:
                    per_owner[owner] = (shard_ordinal, node_id)
        for tile, endpoints in recv_anchors.items():
            per_owner = {owner: value[1]
                         for owner, value in endpoints.items()}
            arm_range.add(tile, per_owner)
        for owner, slot in slots.items():
            self._pipeline_start_reserved.add(slot["segment_id"])
        return {
            "first_tag": first_tag,
            "emission_ordinal": emission_ordinal,
            "tile_count": tile_count,
            "rank_bases": rank_bases,
            "cycle_by_rank": cycles,
            "slots": slots,
        }

    def _emit_compact_cross_body_range(
            self, request_id, pending, tile_run, first_tile, tile_count,
            active_shards, stage):
        """Bulk emit successor tiles forked from one completed body block.

        Each tile's first endpoint keeps the same body-tail parent (stride 0);
        rank-local endpoint chains remain inside that tile. The range is only
        widened across gates that share this original body emission point.
        """
        request_plan = self._pipeline_plan_by_request.get(request_id)
        if request_plan is None:
            raise RuntimeError(
                "compact cross-body release has no explicit pipeline identity seed")
        first_tile = int(first_tile)
        tile_count = int(tile_count)
        active_shards = tuple(active_shards)
        if tile_count <= 0 or not active_shards:
            raise ValueError("cross-body compact extent must be nonempty")
        stream_stage = pending.stage
        if stream_stage not in ("prefill_read", "remote_credit"):
            raise RuntimeError("unknown compact credit stream stage")

        first_transfer = pending.transfer_at(first_tile)
        first_offset = ((first_tile - int(tile_run["first_tile"]))
                        * int(tile_run["units_per_tile"]))
        units = min(int(tile_run["units_per_tile"]),
                    int(tile_run["total_units"]) - first_offset)
        expected = []
        for shard in active_shards:
            if tile_run["axis_unit"] == "source_tokens":
                amount = units * int(shard["bytes_per_unit"])
            else:
                amount = max(0, min(
                    int(shard["total_data_bytes"]), first_offset + units)
                    - first_offset)
            if amount:
                expected.append((int(shard["source_rank"]),
                                 int(shard["target_rank"]),
                                 tuple(int(rank) for rank in shard["noc_path"]),
                                 amount))
        actual = [(int(shard.source_rank), int(shard.target_rank),
                   tuple(int(rank) for rank in shard.noc_path),
                   int(shard.bytes)) for shard in first_transfer.shards]
        if actual != expected:
            raise RuntimeError(
                "compact cross-body range disagrees with its frozen transfer "
                "factory at the first tile")

        owner_ranks = tuple(dict.fromkeys(
            int(shard["target_rank"]) for shard in active_shards))
        slots = {
            owner: self._physical_pipeline_group(
                request_plan, stream_stage, owner, tile_run)
            for owner in owner_ranks}
        if any(slot is None for slot in slots.values()):
            raise RuntimeError("cross-body range has an empty owner group")
        for slot in slots.values():
            slot["identity_rank"] = int(slot["draft"].identity.rank)

        release_pattern = (
            ("release_ack_send", 0), ("release_ack_recv", 0),
            ("data_send", 1), ("data_recv", 1),
            ("fill_ack_send", 2), ("fill_ack_recv", 2),
        )
        active_position = {
            int(shard["global_shard_ordinal"]): index
            for index, shard in enumerate(active_shards)}
        events = []
        for shard in active_shards:
            owner = int(shard["target_rank"])
            slot = slots[owner]
            ordinal = int(shard["global_shard_ordinal"])
            local_shard = slot["global_to_local_shard"][ordinal]
            for endpoint, tag_stage in release_pattern:
                source_side = endpoint in (
                    "release_ack_recv", "data_send", "fill_ack_recv")
                rank = int(shard["source_rank"] if source_side else
                           shard["target_rank"])
                events.append({
                    "rank": rank, "owner": owner,
                    "endpoint": endpoint, "tag_stage": tag_stage,
                    "shard": shard, "local_shard": local_shard,
                    "slot": slot,
                })
        events_by_rank = {}
        positions = {}
        for event in events:
            rank = int(event["rank"])
            positions[id(event)] = len(events_by_rank.setdefault(rank, []))
            events_by_rank[rank].append(event)
        if not events_by_rank:
            raise RuntimeError("cross-body compact event cycle is empty")

        checkpoints = {rank: builder.previous_id
                       for rank, builder in self.builders.items()}
        if any(self.builders[rank].pending_extra_dependencies
               for rank in events_by_rank):
            raise RuntimeError(
                "cross-body compact range encountered unconsumed ordinary "
                "dependencies before a fork")
        rank_bases = {}
        cycles = {}
        for rank, rank_events in events_by_rank.items():
            cycles[rank] = len(rank_events)
            rank_bases[rank] = self.builders[rank].reserve_endpoint_extent(
                tile_count * cycles[rank])
            self.batch["_touched_ranks"].add(rank)

        tag_count = tile_count * 3 * len(active_shards)
        first_tag, emission_ordinal = self.tag_allocator.reserve_with_ordinal(
            tag_count)
        local_first = first_tile - int(tile_run["first_tile"]) + 1
        for owner, slot in slots.items():
            slot["draft"].add_tag_allocation_reference(
                allocator_scope_id="graph-batch-transfer-tags-v1",
                emission_ordinal=emission_ordinal,
                first_tag=first_tag, count=tag_count,
                group_ordinal=slot["group_ordinal"],
                first_tile=local_first, tile_count=tile_count,
                first_shard=0, shard_count=len(slot["owner_shards"]))

        planned_anchor_ids = set()
        anchor_specs = {}
        for event in events:
            if event["endpoint"] != "data_recv":
                continue
            rank = int(event["rank"])
            pos = positions[id(event)]
            for offset in range(max(0, tile_count - 2), tile_count):
                node_id = rank_bases[rank] + offset * cycles[rank] + pos
                anchor_specs.setdefault(id(event), []).append(
                    (local_first + offset, int(event["local_shard"]),
                     event["endpoint"], "frontier"))
                planned_anchor_ids.add((rank, node_id))

        for owner, slot in slots.items():
            if slot["segment_id"] in self._pipeline_start_reserved:
                continue
            event = next((row for row in events
                          if int(row["owner"]) == owner), None)
            if event is None:
                raise RuntimeError("cross-body owner has no endpoint")
            rank = int(event["rank"])
            pos = positions[id(event)]
            node_id = rank_bases[rank] + pos
            anchor_specs.setdefault(id(event), []).append(
                (local_first, int(event["local_shard"]),
                 event["endpoint"], "start"))
            planned_anchor_ids.add((rank, node_id))

        hole_starts = {id(slot["draft"]): len(slot["draft"]._holes)
                       for slot in slots.values()}
        anchor_starts = {id(slot["draft"]): len(slot["draft"].anchors)
                         for slot in slots.values()}
        for event in events:
            slot = event["slot"]
            rank = int(event["rank"])
            pos = positions[id(event)]
            cycle = cycles[rank]
            parent = (rank_bases[rank] + pos - 1 if pos
                      else checkpoints[rank])
            active_index = active_position[
                int(event["shard"]["global_shard_ordinal"])]
            run = {
                "rank": rank,
                "group_ordinal": int(slot["group_ordinal"]),
                "first_tile": local_first,
                "tile_count": tile_count,
                "shard_ordinals": [int(event["local_shard"])],
                "first_json_id": rank_bases[rank] + pos,
                "tile_stride": cycle,
                "shard_stride": 1,
                "initial_parent_json_id": parent,
                "legacy_context": {
                    "request_id": self.builders[rank].request_id,
                    "stage": self.builders[rank].stage,
                    "generation": self.builders[rank].generation,
                },
                "legs": [{
                    "endpoint": event["endpoint"], "json_offset": 0,
                    "tag_emission_ordinal": emission_ordinal,
                    "tag_offset": active_index * 3
                    + int(event["tag_stage"]),
                    "tag_tile_stride": 3 * len(active_shards),
                    "tag_shard_stride": 1,
                }],
            }
            slot["draft"].add_endpoint_run(
                run, anchor_exceptions=anchor_specs.get(id(event), ()))

            if pos:
                dep_first = rank_bases[rank] + pos - 1
                dep_stride = cycle
                default_kind = "virtual_endpoint_terminal"
            else:
                dep_first = checkpoints[rank]
                dep_stride = 0
                if dep_first is None:
                    continue
                default_kind = (
                    "virtual_endpoint_terminal"
                    if self.builders[rank].is_virtual_endpoint(dep_first)
                    else "ordinary_node_terminal")
            exceptional = set()
            if dep_stride:
                for anchor_rank, anchor_id in planned_anchor_ids:
                    if anchor_rank != rank or anchor_id < dep_first:
                        continue
                    delta = anchor_id - dep_first
                    if delta % dep_stride == 0:
                        offset = delta // dep_stride
                        if offset < tile_count:
                            exceptional.add(offset)
            cursor = 0
            for offset in sorted(exceptional):
                if offset > cursor:
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": local_first + cursor,
                        "tile_count": offset - cursor,
                        "shard_ordinals": [int(event["local_shard"])],
                        "endpoint": event["endpoint"],
                        "dependency_kind": default_kind,
                        "first_dependency_json_id": dep_first
                            + cursor * dep_stride,
                        "dependency_tile_stride": dep_stride,
                        "dependency_shard_stride": 0,
                    })
                slot["draft"].add_endpoint_dependency_run({
                    "rank": rank,
                    "group_ordinal": int(slot["group_ordinal"]),
                    "first_tile": local_first + offset,
                    "tile_count": 1,
                    "shard_ordinals": [int(event["local_shard"])],
                    "endpoint": event["endpoint"],
                    "dependency_kind": "ordinary_node_terminal",
                    "first_dependency_json_id": dep_first
                        + offset * dep_stride,
                    "dependency_tile_stride": 0,
                    "dependency_shard_stride": 0,
                })
                cursor = offset + 1
            if cursor < tile_count:
                slot["draft"].add_endpoint_dependency_run({
                    "rank": rank,
                    "group_ordinal": int(slot["group_ordinal"]),
                    "first_tile": local_first + cursor,
                    "tile_count": tile_count - cursor,
                    "shard_ordinals": [int(event["local_shard"])],
                    "endpoint": event["endpoint"],
                    "dependency_kind": default_kind,
                    "first_dependency_json_id": dep_first + cursor * dep_stride,
                    "dependency_tile_stride": dep_stride,
                    "dependency_shard_stride": 0,
                })

        for slot in slots.values():
            draft = slot["draft"]
            for hole in draft._holes[hole_starts[id(draft)]:]:
                self.builders[int(hole["rank"])].mark_virtual_endpoint_progression(
                    int(hole["first_json_id"]), int(hole["count"]),
                    int(hole["stride"]))
        anchors_to_emit = []
        seen = set()
        for slot in slots.values():
            draft = slot["draft"]
            for anchor in draft.anchors[anchor_starts[id(draft)]:]:
                key = (anchor["rank"], anchor["node_id"])
                if key in seen:
                    continue
                seen.add(key)
                selector = anchor["selector"]
                context = next((run["legacy_context"]
                                for run in draft.endpoint_runs
                                if run["group_ordinal"]
                                    == selector.get("group_ordinal")
                                and run["rank"] == anchor["rank"]
                                and run["first_tile"]
                                    <= selector.get("tile_ordinal", 0)
                                    < run["first_tile"] + run["tile_count"]
                                and selector.get("shard_ordinal")
                                    in run["shard_ordinals"]
                                and any(leg["endpoint"]
                                        == selector.get("endpoint")
                                        for leg in run["legs"])), None)
                if context is None:
                    raise RuntimeError(
                        "cross-body anchor has no matching endpoint context")
                anchors_to_emit.append((anchor, context, slot["segment_id"]))
        anchors_to_emit.sort(key=lambda item: (item[0]["rank"],
                                               item[0]["node_id"]))
        for anchor, context, segment_id in anchors_to_emit:
            self.builders[int(anchor["rank"])].emit_reserved_pipeline_anchor(
                "{}_pipeline_frontier_{}".format(
                    sanitize_node_prefix(request_id), anchor["node_id"]),
                segment_id=str(segment_id), role=anchor["role"],
                node_id=int(anchor["node_id"]), legacy_context=context)

        arm_range = self._credit_arms.setdefault(request_id, _CreditArmRange())
        arm_range.add_extent(first_tile, tile_count, owner_ranks)
        recv_anchors = {}
        for event in events:
            if event["endpoint"] != "data_recv":
                continue
            rank = int(event["rank"])
            owner = int(event["owner"])
            shard_ordinal = int(event["shard"]["global_shard_ordinal"])
            pos = positions[id(event)]
            for offset in range(max(0, tile_count - 2), tile_count):
                global_tile = first_tile + offset
                node_id = rank_bases[rank] + offset * cycles[rank] + pos
                per_owner = recv_anchors.setdefault(global_tile, {})
                prior = per_owner.get(owner)
                if prior is None or shard_ordinal > prior[0]:
                    per_owner[owner] = (shard_ordinal, node_id)
        for global_tile, endpoints in recv_anchors.items():
            arm_range.add(global_tile, {
                owner: value[1] for owner, value in endpoints.items()})
        self._pipeline_start_reserved.update(
            slot["segment_id"] for slot in slots.values())
        return {"first_tag": first_tag, "emission_ordinal": emission_ordinal,
                "rank_bases": rank_bases, "cycle_by_rank": cycles}

    @staticmethod
    def _endpoint_run_reference(slot, global_tile, global_shard_ordinal,
                                endpoint):
        """Resolve one prior endpoint through the bounded run table."""
        local_tile = (int(global_tile)
                      - int(slot["tile_run"]["first_tile"]) + 1)
        local_shard = slot["global_to_local_shard"].get(
            int(global_shard_ordinal))
        if local_shard is None:
            raise RuntimeError("endpoint reference shard is outside owner group")
        draft = slot["draft"]
        group_ordinal = int(slot["group_ordinal"])
        for run_index in range(len(draft.endpoint_runs) - 1, -1, -1):
            run = draft.endpoint_runs[run_index]
            if (run["group_ordinal"] != group_ordinal
                    or run["rank"] != int(slot["identity_rank"])
                    or local_tile < run["first_tile"]
                    or local_tile >= run["first_tile"] + run["tile_count"]
                    or local_shard not in run["shard_ordinals"]):
                continue
            leg = next((leg for leg in run["legs"]
                        if leg["endpoint"] == endpoint), None)
            if leg is None:
                continue
            local_index = run["shard_ordinals"].index(local_shard)
            node_id = (run["first_json_id"]
                       + (local_tile - run["first_tile"])
                       * run["tile_stride"]
                       + local_index * run["shard_stride"]
                       + leg["json_offset"])
            anchor = any(item["rank"] == run["rank"]
                         and item["node_id"] == node_id
                         for item in draft.anchors)
            return {
                "node_id": node_id,
                "tile_stride": run["tile_stride"],
                "run_first_tile": run["first_tile"],
                "run_tile_count": run["tile_count"],
                "anchor": anchor,
            }
        raise RuntimeError(
            "same-body marker has no prior compact data_recv run for "
            f"tile {global_tile}, shard {global_shard_ordinal}")

    def _emit_compact_same_body_range(
            self, request_id, pending, consumer_run, successor_run,
            first_consumer_tile, tile_count, consumer_shards,
            successor_shards, stage):
        """Emit one bounded same-body marker/release range.

        The event cycle is the legacy order for one consumer/successor pair:
        one charged body marker per target rank, then the successor's six
        communication endpoints per active shard. IDs follow each rank's
        exact cycle; transfer tags follow the active global shard order.
        """
        request_plan = self._pipeline_plan_by_request.get(request_id)
        if request_plan is None:
            raise RuntimeError(
                "compact same-body release has no explicit pipeline identity seed")
        first_consumer_tile = int(first_consumer_tile)
        tile_count = int(tile_count)
        if tile_count <= 0:
            raise ValueError("same-body compact extent must be nonempty")
        lag = 2
        first_successor_tile = first_consumer_tile + lag
        stream_stage = pending.stage
        if stream_stage not in ("prefill_read", "remote_credit"):
            raise RuntimeError("unknown compact credit stream stage")

        consumer_active = tuple(consumer_shards)
        successor_active = tuple(successor_shards)
        consumer_targets = tuple(sorted({int(row["target_rank"])
                                         for row in consumer_active}))
        successor_owners = tuple(sorted({int(row["target_rank"])
                                         for row in successor_active}))
        if not consumer_targets or not successor_owners:
            raise RuntimeError("same-body pair has an empty active owner set")

        consumer_slots = {
            owner: self._physical_pipeline_group(
                request_plan, stream_stage, owner, consumer_run)
            for owner in consumer_targets}
        successor_slots = {
            owner: self._physical_pipeline_group(
                request_plan, stream_stage, owner, successor_run)
            for owner in successor_owners}
        slots = {}
        for slot in (*consumer_slots.values(), *successor_slots.values()):
            slots[slot["segment_id"]] = slot
        if any(slot is None for slot in slots.values()):
            raise RuntimeError("same-body compact range has no owner group")
        for slot in (*consumer_slots.values(), *successor_slots.values()):
            slot["identity_rank"] = int(slot["draft"].identity.rank)

        # Confirm the first successor still matches the authoritative legacy
        # transfer factory. The remaining tiles are represented by the shared
        # planner axis and per-shard byte caps, not materialized objects.
        first_transfer = pending.transfer_at(first_successor_tile)
        successor_offset = ((first_successor_tile
                             - int(successor_run["first_tile"]))
                            * int(successor_run["units_per_tile"]))
        successor_units = min(
            int(successor_run["units_per_tile"]),
            int(successor_run["total_units"]) - successor_offset)
        expected = []
        for shard in successor_active:
            if successor_run["axis_unit"] == "source_tokens":
                amount = successor_units * int(shard["bytes_per_unit"])
            else:
                amount = max(0, min(
                    int(shard["total_data_bytes"]),
                    successor_offset + successor_units) - successor_offset)
            if amount:
                expected.append((int(shard["source_rank"]),
                                 int(shard["target_rank"]),
                                 tuple(int(rank) for rank in shard["noc_path"]),
                                 amount))
        actual = [(int(shard.source_rank), int(shard.target_rank),
                   tuple(int(rank) for rank in shard.noc_path),
                   int(shard.bytes)) for shard in first_transfer.shards]
        if actual != expected:
            raise RuntimeError(
                "compact same-body range disagrees with its frozen transfer "
                "factory at the first successor tile")

        # Rank-local cycle: all main-chain consumer markers first, followed by
        # each active successor shard in its original global order.
        events = []
        marker_event_by_owner = {}
        for owner in consumer_targets:
            slot = consumer_slots[owner]
            local_candidates = [
                row for row in consumer_active
                if int(row["target_rank"]) == owner]
            canonical = min(
                local_candidates,
                key=lambda row: int(row["global_shard_ordinal"]))
            local_shard = slot["global_to_local_shard"][
                int(canonical["global_shard_ordinal"])]
            event = {
                "kind": "marker", "owner": owner, "rank": owner,
                "endpoint": "body_marker", "shard": canonical,
                "local_shard": local_shard, "slot": slot,
            }
            events.append(event)
            marker_event_by_owner[owner] = event

        release_pattern = (
            ("release_ack_send", 0), ("release_ack_recv", 0),
            ("data_send", 1), ("data_recv", 1),
            ("fill_ack_send", 2), ("fill_ack_recv", 2),
        )
        active_position = {}
        for active_index, shard in enumerate(successor_active):
            ordinal = int(shard["global_shard_ordinal"])
            active_position[ordinal] = active_index
            owner = int(shard["target_rank"])
            slot = successor_slots[owner]
            local_shard = slot["global_to_local_shard"][ordinal]
            for endpoint, tag_stage in release_pattern:
                source_side = endpoint in (
                    "release_ack_recv", "data_send", "fill_ack_recv")
                rank = int(shard["source_rank"] if source_side else
                           shard["target_rank"])
                events.append({
                    "kind": "release", "owner": owner, "rank": rank,
                    "endpoint": endpoint, "shard": shard,
                    "local_shard": local_shard, "slot": slot,
                    "tag_stage": tag_stage,
                })

        events_by_rank = {}
        event_positions = {}
        for event in events:
            rank = int(event["rank"])
            event_positions[id(event)] = len(events_by_rank.setdefault(rank, []))
            events_by_rank[rank].append(event)
        if not events_by_rank:
            raise RuntimeError("same-body compact event cycle is empty")

        checkpoint_by_rank = {
            rank: builder.previous_id for rank, builder in self.builders.items()
        }
        pending_by_rank = {
            rank: tuple(builder.pending_extra_dependencies)
            for rank, builder in self.builders.items() if events_by_rank.get(rank)
        }
        if any(pending_by_rank.values()):
            raise RuntimeError(
                "same-body compact range encountered unconsumed ordinary "
                "dependencies before a marker cycle")

        rank_bases = {}
        cycle_by_rank = {}
        for rank, rank_events in events_by_rank.items():
            cycle = len(rank_events)
            cycle_by_rank[rank] = cycle
            rank_bases[rank] = self.builders[rank].reserve_endpoint_extent(
                tile_count * cycle)
            self.batch["_touched_ranks"].add(rank)

        tag_count = tile_count * 3 * len(successor_active)
        first_tag, emission_ordinal = self.tag_allocator.reserve_with_ordinal(
            tag_count)
        first_successor_local = (
            first_successor_tile - int(successor_run["first_tile"]) + 1)
        for owner, slot in successor_slots.items():
            slot["draft"].add_tag_allocation_reference(
                allocator_scope_id="graph-batch-transfer-tags-v1",
                emission_ordinal=emission_ordinal,
                first_tag=first_tag, count=tag_count,
                group_ordinal=slot["group_ordinal"],
                first_tile=first_successor_local,
                tile_count=tile_count, first_shard=0,
                shard_count=len(slot["owner_shards"]))

        planned_anchor_ids = set()
        anchor_specs = {}
        # A marker is a physical frontier only at this body range's final
        # tile. It remains one charged synthetic microjob in execution.
        for owner, event in marker_event_by_owner.items():
            rank = int(event["rank"])
            pos = event_positions[id(event)]
            tile_local = (first_consumer_tile
                          - int(consumer_run["first_tile"]) + tile_count)
            node_id = rank_bases[rank] + (tile_count - 1) * cycle_by_rank[rank] + pos
            anchor_specs.setdefault(id(event), []).append(
                (tile_local, int(event["local_shard"]),
                 event["endpoint"], "frontier"))
            planned_anchor_ids.add((rank, node_id))

        # Data-recv tails are the bounded ordinary frontiers used by later
        # body dependencies. Each active target shard keeps at most two.
        for event in events:
            if event["kind"] != "release" or event["endpoint"] != "data_recv":
                continue
            rank = int(event["rank"])
            pos = event_positions[id(event)]
            for tile_offset in range(max(0, tile_count - 2), tile_count):
                tile_local = first_successor_local + tile_offset
                node_id = rank_bases[rank] + tile_offset * cycle_by_rank[rank] + pos
                anchor_specs.setdefault(id(event), []).append(
                    (tile_local, int(event["local_shard"]),
                     event["endpoint"], "frontier"))
                planned_anchor_ids.add((rank, node_id))

        # The first charged successor endpoint starts a descriptor domain if
        # an earlier extent has not already installed its start anchor.
        first_release_event_by_owner = {}
        for event in events:
            if event["kind"] == "release":
                first_release_event_by_owner.setdefault(int(event["owner"]), event)
        for owner, slot in successor_slots.items():
            if slot["segment_id"] in self._pipeline_start_reserved:
                continue
            event = first_release_event_by_owner.get(owner)
            if event is None:
                raise RuntimeError("successor owner has no physical endpoint")
            pos = event_positions[id(event)]
            rank = int(event["rank"])
            node_id = rank_bases[rank] + pos
            tile_local = first_successor_local
            anchor_specs.setdefault(id(event), []).append(
                (tile_local, int(event["local_shard"]),
                 event["endpoint"], "start"))
            planned_anchor_ids.add((rank, node_id))

        # Preserve the main-chain checkpoint before endpoint runs are emitted.
        marker_first_id = {
            owner: rank_bases[owner]
            + event_positions[id(event)]
            for owner, event in marker_event_by_owner.items()}

        hole_starts = {
            id(slot["draft"]): len(slot["draft"]._holes)
            for slot in slots.values()}
        anchor_starts = {
            id(slot["draft"]): len(slot["draft"].anchors)
            for slot in slots.values()}
        for event in events:
            slot = event["slot"]
            rank = int(event["rank"])
            pos = event_positions[id(event)]
            cycle = cycle_by_rank[rank]
            if event["kind"] == "marker":
                local_first = (first_consumer_tile
                               - int(consumer_run["first_tile"]) + 1)
                group_local_shard = int(event["local_shard"])
                initial_parent = checkpoint_by_rank[rank]
                legs = [{
                    "endpoint": "body_marker", "json_offset": 0,
                    "tag_emission_ordinal": None, "tag_offset": None,
                    "tag_tile_stride": None, "tag_shard_stride": None,
                }]
            else:
                local_first = first_successor_local
                group_local_shard = int(event["local_shard"])
                if pos:
                    initial_parent = rank_bases[rank] + pos - 1
                elif rank in marker_first_id:
                    initial_parent = marker_first_id[rank]
                else:
                    initial_parent = checkpoint_by_rank[rank]
                active_index = active_position[
                    int(event["shard"]["global_shard_ordinal"])]
                tag_offset = active_index * 3 + int(event["tag_stage"])
                legs = [{
                    "endpoint": event["endpoint"], "json_offset": 0,
                    "tag_emission_ordinal": emission_ordinal,
                    "tag_offset": tag_offset,
                    "tag_tile_stride": 3 * len(successor_active),
                    "tag_shard_stride": 1,
                }]
            run = {
                "rank": rank,
                "group_ordinal": int(slot["group_ordinal"]),
                "first_tile": local_first,
                "tile_count": tile_count,
                "shard_ordinals": [group_local_shard],
                "first_json_id": rank_bases[rank] + pos,
                "tile_stride": cycle,
                "shard_stride": 1,
                "initial_parent_json_id": initial_parent,
                "legacy_context": {
                    "request_id": self.builders[rank].request_id,
                    "stage": self.builders[rank].stage,
                    "generation": self.builders[rank].generation,
                },
                "legs": legs,
            }
            run_index = slot["draft"].add_endpoint_run(
                run, anchor_exceptions=anchor_specs.get(id(event), ()))
            event["run_index"] = run_index

            def add_dependency_pieces(*, first_tile, count, first_id, stride,
                                      default_kind, shard=group_local_shard,
                                      endpoint=event["endpoint"],
                                      slot=slot, rank=rank):
                if count <= 0:
                    return
                exceptional = set()
                if stride:
                    lo = int(first_id)
                    for anchor_rank, anchor_id in planned_anchor_ids:
                        if anchor_rank != rank or anchor_id < lo:
                            continue
                        delta = anchor_id - lo
                        if delta % int(stride) == 0:
                            index = delta // int(stride)
                            if index < count:
                                exceptional.add(index)
                cursor = 0
                for index in sorted(exceptional):
                    if index > cursor:
                        slot["draft"].add_endpoint_dependency_run({
                            "rank": rank,
                            "group_ordinal": int(slot["group_ordinal"]),
                            "first_tile": int(first_tile) + cursor,
                            "tile_count": index - cursor,
                            "shard_ordinals": [int(shard)],
                            "endpoint": endpoint,
                            "dependency_kind": default_kind,
                            "first_dependency_json_id": int(first_id)
                                + cursor * int(stride),
                            "dependency_tile_stride": int(stride),
                            "dependency_shard_stride": 0,
                        })
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": int(first_tile) + index,
                        "tile_count": 1,
                        "shard_ordinals": [int(shard)],
                        "endpoint": endpoint,
                        "dependency_kind": "ordinary_node_terminal",
                        "first_dependency_json_id": int(first_id)
                            + index * int(stride),
                        "dependency_tile_stride": 0,
                        "dependency_shard_stride": 0,
                    })
                    cursor = index + 1
                if cursor < count:
                    slot["draft"].add_endpoint_dependency_run({
                        "rank": rank,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": int(first_tile) + cursor,
                        "tile_count": count - cursor,
                        "shard_ordinals": [int(shard)],
                        "endpoint": endpoint,
                        "dependency_kind": default_kind,
                        "first_dependency_json_id": int(first_id)
                            + cursor * int(stride),
                        "dependency_tile_stride": int(stride),
                        "dependency_shard_stride": 0,
                    })

            if event["kind"] == "marker":
                checkpoint = checkpoint_by_rank[rank]
                if checkpoint is not None:
                    checkpoint_kind = (
                        "virtual_endpoint_terminal"
                        if self.builders[rank].is_virtual_endpoint(checkpoint)
                        else "ordinary_node_terminal")
                    add_dependency_pieces(
                        first_tile=local_first, count=1,
                        first_id=checkpoint, stride=0,
                        default_kind=checkpoint_kind)
                if tile_count > 1:
                    add_dependency_pieces(
                        first_tile=local_first + 1, count=tile_count - 1,
                        first_id=rank_bases[rank] + pos,
                        stride=cycle,
                        default_kind="virtual_endpoint_terminal")

                # The first main-chain tile has no receive gate in the old
                # builder. Later markers carry one recv parent per owner rank;
                # keep that bounded terminal relation without inventing it
                # for tile 1.
                recv_offset = max(0, 2 - first_consumer_tile)
                recv_count = tile_count - recv_offset
                if recv_count > 0:
                    owner_shards = [
                        row for row in consumer_active
                        if int(row["target_rank"]) == int(event["owner"])]
                    recv_shard = max(
                        owner_shards,
                        key=lambda row: int(row["global_shard_ordinal"]))
                    recv_slot = consumer_slots[int(event["owner"])]
                    consumer_local_first = (
                        first_consumer_tile + recv_offset
                        - int(consumer_run["first_tile"]) + 1)
                    recv_local = recv_slot["global_to_local_shard"][
                        int(recv_shard["global_shard_ordinal"])]
                    matching_recv_run = next(
                        run for run in reversed(
                            recv_slot["draft"].endpoint_runs)
                        if run["group_ordinal"] == recv_slot["group_ordinal"]
                        and run["rank"] == rank
                        and run["first_tile"] <= consumer_local_first
                        and consumer_local_first + recv_count - 1
                            < run["first_tile"] + run["tile_count"]
                        and recv_local in run["shard_ordinals"]
                        and any(leg["endpoint"] == "data_recv"
                                for leg in run["legs"]))
                    recv_leg = next(
                        leg for leg in matching_recv_run["legs"]
                        if leg["endpoint"] == "data_recv")
                    recv_run_shard_index = (
                        matching_recv_run["shard_ordinals"].index(recv_local))
                    recv_first_id = (
                        matching_recv_run["first_json_id"]
                        + (consumer_local_first
                           - matching_recv_run["first_tile"])
                        * matching_recv_run["tile_stride"]
                        + recv_run_shard_index
                        * matching_recv_run["shard_stride"]
                        + recv_leg["json_offset"])
                    anchor_offsets = set()
                    for anchor in recv_slot["draft"].anchors:
                        selector = anchor["selector"]
                        if (anchor["rank"] == rank
                                and selector.get("group_ordinal")
                                    == recv_slot["group_ordinal"]
                                and selector.get("endpoint") == "data_recv"
                                and selector.get("shard_ordinal")
                                    == recv_local):
                            offset = selector.get("tile_ordinal", 0) \
                                - consumer_local_first
                            if 0 <= offset < recv_count:
                                anchor_offsets.add(offset)
                    cursor = 0
                    for offset in sorted(anchor_offsets):
                        if offset > cursor:
                            slot["draft"].add_endpoint_dependency_run({
                                "rank": rank,
                                "group_ordinal": int(slot["group_ordinal"]),
                                "first_tile": local_first + recv_offset + cursor,
                                "tile_count": offset - cursor,
                                "shard_ordinals": [group_local_shard],
                                "endpoint": "body_marker",
                                "dependency_kind": "virtual_endpoint_terminal",
                                "first_dependency_json_id": recv_first_id
                                    + cursor * matching_recv_run["tile_stride"],
                                "dependency_tile_stride": (
                                    matching_recv_run["tile_stride"]),
                                "dependency_shard_stride": 0,
                            })
                        slot["draft"].add_endpoint_dependency_run({
                            "rank": rank,
                            "group_ordinal": int(slot["group_ordinal"]),
                            "first_tile": local_first + recv_offset + offset,
                            "tile_count": 1,
                            "shard_ordinals": [group_local_shard],
                            "endpoint": "body_marker",
                            "dependency_kind": "ordinary_node_terminal",
                            "first_dependency_json_id": recv_first_id
                                + offset * matching_recv_run["tile_stride"],
                            "dependency_tile_stride": 0,
                            "dependency_shard_stride": 0,
                        })
                        cursor = offset + 1
                    if cursor < recv_count:
                        slot["draft"].add_endpoint_dependency_run({
                            "rank": rank,
                            "group_ordinal": int(slot["group_ordinal"]),
                            "first_tile": local_first + recv_offset + cursor,
                            "tile_count": recv_count - cursor,
                            "shard_ordinals": [group_local_shard],
                            "endpoint": "body_marker",
                            "dependency_kind": "virtual_endpoint_terminal",
                            "first_dependency_json_id": recv_first_id
                                + cursor * matching_recv_run["tile_stride"],
                            "dependency_tile_stride": (
                                matching_recv_run["tile_stride"]),
                            "dependency_shard_stride": 0,
                        })

                # All local target shards share the one owner-rank marker as
                # their consume terminal. The final marker is physical; the
                # prefix markers remain virtual endpoint terminals.
                local_consumer_shards = sorted(
                    consumer_slots[int(event["owner"])]
                    ["global_to_local_shard"][
                        int(row["global_shard_ordinal"])]
                    for row in consumer_active
                    if int(row["target_rank"]) == int(event["owner"]))
                completion_slot = consumer_slots[int(event["owner"])]
                if tile_count > 1:
                    completion_slot["draft"].add_consume_completion_run({
                        "rank": int(event["owner"]),
                        "group_ordinal": int(completion_slot["group_ordinal"]),
                        "first_tile": local_first,
                        "tile_count": tile_count - 1,
                        "shard_ordinals": local_consumer_shards,
                        "dependency_kind": "virtual_endpoint_terminal",
                        "first_dependency_json_id": marker_first_id[
                            int(event["owner"])],
                        "dependency_tile_stride": cycle_by_rank[
                            int(event["owner"])],
                        "dependency_shard_stride": 0,
                    })
                completion_slot["draft"].add_consume_completion_run({
                    "rank": int(event["owner"]),
                    "group_ordinal": int(completion_slot["group_ordinal"]),
                    "first_tile": local_first + tile_count - 1,
                    "tile_count": 1,
                    "shard_ordinals": local_consumer_shards,
                    "dependency_kind": "ordinary_node_terminal",
                    "first_dependency_json_id": marker_first_id[
                        int(event["owner"])]
                        + (tile_count - 1) * cycle_by_rank[
                            int(event["owner"])],
                    "dependency_tile_stride": 0,
                    "dependency_shard_stride": 0,
                })
            elif pos:
                add_dependency_pieces(
                    first_tile=local_first, count=tile_count,
                    first_id=rank_bases[rank] + pos - 1,
                    stride=cycle,
                    default_kind="virtual_endpoint_terminal")
            elif rank in marker_first_id:
                add_dependency_pieces(
                    first_tile=local_first, count=tile_count,
                    first_id=marker_first_id[rank], stride=cycle,
                    default_kind="virtual_endpoint_terminal")
            else:
                checkpoint = checkpoint_by_rank[rank]
                if checkpoint is not None:
                    kind = ("virtual_endpoint_terminal"
                            if self.builders[rank].is_virtual_endpoint(checkpoint)
                            else "ordinary_node_terminal")
                    add_dependency_pieces(
                        first_tile=local_first, count=tile_count,
                        first_id=checkpoint, stride=0,
                        default_kind=kind)

        # Register sparse holes only after the bounded physical exceptions
        # have been determined for every owner descriptor.
        for slot in slots.values():
            draft = slot["draft"]
            start = hole_starts[id(draft)]
            for hole in draft._holes[start:]:
                self.builders[int(hole["rank"])].mark_virtual_endpoint_progression(
                    int(hole["first_json_id"]), int(hole["count"]),
                    int(hole["stride"]))

        # Materialize only the named bounded anchors, in each rank's original
        # ID order. Their payload is zero-cost; native endpoint execution
        # retains the old COMM/marker charge through the selector.
        anchors_to_emit = []
        seen_anchors = set()
        for slot in slots.values():
            draft = slot["draft"]
            for anchor in draft.anchors[anchor_starts[id(draft)]:]:
                key = (anchor["rank"], anchor["node_id"])
                if key in seen_anchors:
                    continue
                seen_anchors.add(key)
                context = None
                selector = anchor["selector"]
                for run in draft.endpoint_runs:
                    if (run["group_ordinal"] == selector.get("group_ordinal")
                            and run["rank"] == anchor["rank"]
                            and run["first_tile"] <= selector.get("tile_ordinal", 0)
                            < run["first_tile"] + run["tile_count"]
                            and selector.get("shard_ordinal")
                            in run["shard_ordinals"]
                            and any(leg["endpoint"] == selector.get("endpoint")
                                    for leg in run["legs"])):
                        context = run["legacy_context"]
                        break
                if context is None:
                    raise RuntimeError("pipeline anchor has no matching emitted run")
                anchors_to_emit.append((anchor, context, slot["segment_id"]))
        anchors_to_emit.sort(key=lambda item: (item[0]["rank"], item[0]["node_id"]))
        for anchor, context, segment_id in anchors_to_emit:
            self.builders[int(anchor["rank"])].emit_reserved_pipeline_anchor(
                "{}_pipeline_frontier_{}".format(
                    sanitize_node_prefix(request_id), anchor["node_id"]),
                segment_id=str(segment_id), role=anchor["role"],
                node_id=int(anchor["node_id"]), legacy_context=context)

        # Keep the main chain at the last marker per owner; release endpoints
        # are side branches and must never become the body tail.
        for rank, checkpoint in checkpoint_by_rank.items():
            if rank in marker_first_id:
                marker_event = marker_event_by_owner[rank]
                pos = event_positions[id(marker_event)]
                self.builders[rank].previous_id = (
                    rank_bases[rank]
                    + (tile_count - 1) * cycle_by_rank[rank] + pos)
            elif rank in events_by_rank:
                self.builders[rank].previous_id = checkpoint

        arm_range = self._credit_arms.setdefault(request_id, _CreditArmRange())
        arm_range.add_extent(first_successor_tile, tile_count, successor_owners)
        arm_range.add_mainline_extent(
            first_consumer_tile, tile_count, consumer_targets)
        recv_anchors = {}
        for event in events:
            if event["kind"] != "release" or event["endpoint"] != "data_recv":
                continue
            owner = int(event["owner"])
            rank = int(event["rank"])
            shard_ordinal = int(event["shard"]["global_shard_ordinal"])
            pos = event_positions[id(event)]
            for tile_offset in range(max(0, tile_count - 2), tile_count):
                global_tile = first_successor_tile + tile_offset
                node_id = rank_bases[rank] + tile_offset * cycle_by_rank[rank] + pos
                per_owner = recv_anchors.setdefault(global_tile, {})
                prior = per_owner.get(owner)
                if prior is None or shard_ordinal > prior[0]:
                    per_owner[owner] = (shard_ordinal, node_id)
        for global_tile, endpoints in recv_anchors.items():
            per_owner = {owner: value[1]
                         for owner, value in endpoints.items()}
            arm_range.add(global_tile, per_owner)

        for owner, slot in successor_slots.items():
            if slot["segment_id"] not in self._pipeline_start_reserved:
                self._pipeline_start_reserved.add(slot["segment_id"])
        return {
            "tile_count": tile_count,
            "rank_bases": rank_bases,
            "cycle_by_rank": cycle_by_rank,
            "first_tag": first_tag,
            "emission_ordinal": emission_ordinal,
        }

    def _collect(self, marker: dict) -> None:
        # R6-2（2026-09-14）：逐秩边审计**代码内默认常开**（SH_EDGE_AUDIT=0
        # 显式关闭）。O(E)/批只读校验：每条 parent 边的 from/to 必须是本
        # rank 自身 id 空间内 from ≤ to 的既有节点（悬挂/前向引用立即带
        # 节点名 fail-closed）——把 D3 类静默错连提前到 Python 侧。跨
        # rank 的 id 串用（id 数值碰撞于两个计数器空间）由 R5 的结构性
        # 重建 + source_ranks 断言承担，本审计不重复覆盖。
        audit_enabled = os.environ.get("SH_EDGE_AUDIT", "1") != "0"
        for rank, builder in self.builders.items():
            node_mark, edge_mark = marker[rank]
            edges = builder.edges
            # 正常路径的 marker 为 0，直接 extend 避免临时 slice；非零
            # marker 仅保留本次新增尾部。所有 _mark() 都在同一发射调用内
            # 被单次 _collect() 消费，故旧前缀已在先前批次交付，可立即
            # clear 释放对节点/边 dict 的最后一层 builder 引用。
            if len(nodes := builder.nodes) > node_mark:
                self.batch["_touched_ranks"].add(int(rank))
            self.batch["nodes"].extend(
                nodes if node_mark == 0 else nodes[node_mark:])
            self.batch["parent_edges"].extend(
                edges if edge_mark == 0 else edges[edge_mark:])
            if audit_enabled and edge_mark < len(edges):
                next_id = builder.next_id
                for edge in edges[edge_mark:]:
                    source, target = edge["from"], edge["to"]
                    if not (
                        isinstance(source, int) and 0 <= source < next_id
                        and isinstance(target, int) and 0 <= target < next_id
                        and source <= target
                    ):
                        raise RuntimeError(
                            "edge audit failure on rank {}: {} (ids must "
                            "reference existing earlier nodes, from <= to; "
                            "next_id={})".format(rank, edge, next_id))
            nodes.clear()
            edges.clear()

    def _mark(self) -> dict:
        return {
            rank: (len(builder.nodes), len(builder.edges))
            for rank, builder in self.builders.items()
        }

    # ------------------------------------------------------------- 发射 --

    def emit_admission_batch(self, request_plan: dict) -> dict:
        """发射 request 的准入动作（ARRIVAL 决策的图；拼 batch 改造
        2026-08-22）：到达/interval gate + history_evictions +
        history_transfer（含 partial 流水恢复全部节点）+ prefill_evictions +
        prefill readiness barrier。

        prefill 主体（chunk 序列）与 PREFILL_DRAIN watch 不再在此发射——
        移入实例迭代列车（emit_iteration_train 的折叠体与 drain 标记）。
        独立逐出支链返回自己的尾节点 watch，不能借 prefill drain 提前核销。

        strategy 恒走 roofline 物理时钟；[frontier 接续裁决，strategy
        死锁修复统一(2026-08-19)] 的无条件接续语义不变：准入动作链到该
        rank 当前 frontier（实例列车在飞时物理排在列车后）。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        self._set_context(request_plan, "prefill", 0)
        marker = self._mark()
        eviction_watches = self._emit_admission_actions(request_plan)
        self._collect(marker)
        return {"eviction_watches": eviction_watches}

    def emit_iteration_train(self, train_plan: dict) -> dict:
        """发射一趟实例迭代列车（拼 batch 改造核心，2026-08-22；设计
        文档 §3.2"迭代列车聚合发射"；WP9 拆分重构 2026-08-26 起
        头/体/尾分解为 _emit_train_head/_emit_train_body/
        _emit_train_tail_markers，整列路径行为逐字节不变）。

        train_plan（调度器冻结的成员快照）字段：
          train_id            批命名空间 id（"batch_train_i<实例>_<序号>"；
                              共享体节点归属，物理完成与逻辑请求完成分离）
          instance_index      列车所在实例
          stage               "decode"（有 decode 成员，含混合迭代）或
                              "prefill"（纯 prefill 列车）
          joiners             新成员 request_plan 列表（含 decode_evictions/
                              prefill_decode_transfer/prefill_drain_block_ends
                              {rank: drain 列车 barrier 节点 id}）——迁移
                              节点先于列车体，经共享 readiness barrier 栅栏
          pass_spans          成员×迭代展开的 (tokens, kv) 平铺列表
          iterations          迭代数（= weight_passes：权重字节 ×迭代数，
                              与批成员数无关；激活/KV/AR 逐 span 精确）
          first_chunk_member  队列头首 chunk 在本列车时的成员（pstart 标记
                              + partial 恢复 suffix 门挂接；None = 无）
          drain_members       本列车内完成最后 prefill chunk 的请求 plan 列表
          exit_members        本列车内退出 decode 的成员 plan 列表
          first_token         WP9 首 token 观测（2026-08-26；缺省 None =
                              拆分开关关闭或无 debut，行为与上线前逐字节
                              一致）：{"split": True} 拒绝本方法（两段式
                              经 emit_train_first_step/emit_train_remainder
                              发射）；{"split": False} 为不拆车标记增强
                              （首 token 标记/exit 改名在尾标记处挂）。

        每实例 rank 上的结构（链序）：
          [joiner 迁移 ...] → [共享 readiness barrier] → 17 类聚合体节点
          (transformer_pass_aggregated, weight_passes=iterations) →
          [first_token 标记（不拆车增强时）] → [drain 标记 ...] →
          [exit 标记 ...] → 共享 end barrier。

        D10 静默重建（v2.0 §6.2，2026-09-27）：joiner plan 携带
        joint_suppress_output/joint_suppress_sampling=True（D 边界
        recompute 的 context_rebuild 段）时，其 (H_B, H_B) 重建 span
        在构图器同步抑制输出侧副作用——不计入 final norm/logits 聚合
        的输出位（transformer_pass_aggregated 的逐 span 输出分量）、
        不产生/不推进 debut first_token 观测标记、exit 标记不加
        first_token 双锚点；注意力/MLP/KV 算子时间照常聚合（静默重建
        的计算量真实存在）；RNG/采样状态不因重建 span 前进（事件侧
        无采样算子，RNG 口径由 joint_boundary_first_token 元数据承载，
        本侧不新增观测）。首 token 观测仍只出现一次（来自 P 末，
        §5.2 首 token 归属）——重建 span 不新增观测点。

        返回 {"drain_members": {request_id: {rank: 标记节点 id}},
              "exit_members": {request_id: {rank: 标记节点 id}},
              "sentinel_members": {rank: 标记节点 id},
              "block_ends": {rank: end barrier 节点 id}}——标记即
        PREFILL_DRAIN / DECODE_COMPLETION watch 成员（barrier 前末节点
        口径：标记是列车体后、end barrier 前的真实节点）；块末账本
        _block_ends[req]["seg1"]/["seg2"] = 本列车 post-barrier 节点
        （decode_evictions/completion_evictions 触发门与下一 turn
        interval gate after_node_id 的来源，五仓一致口径不随 watch 锚点
        变化）。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        first_token = train_plan.get("first_token")
        if first_token is not None and first_token.get("split"):
            raise RuntimeError(
                "split train plans must be emitted through "
                "emit_train_first_step/emit_train_remainder")
        marker = self._mark()
        # D10 静默重建：输出抑制账本开账（体发射点逐块匹配消费、尾
        # 标记前全量核销，见 _open/_consume/_close_rebuild_output_
        # suppression）。
        self._open_rebuild_output_suppression(train_plan)
        credit_blocks = (train_plan.get("remote_credit") or {}).get(
            "body_blocks")
        prefill_read_blocks = (
            train_plan.get("prefill_read") or {}).get("body_blocks")
        if credit_blocks is not None and prefill_read_blocks is not None:
            # 修复轮三 R4（2026-09-28）：跨请求并存（decode 成员 credit
            # 切片 + 队头 prefill 前缀读窗）合法且可达（argmin 正常路径
            # 即可命中）——两体块源同为逐迭代块、同一 pass_spans 划分、
            # 同一 (rid, 块号) 门账本（_credit_arms/_credit_pending_tiles
            # 逐请求键），按迭代合并为单体块序列（门 = 并集，I2 同款纪
            # 律）。"一请求同一阶段至多一条远读流" 的互斥语义收窄到请
            # 求粒度：同一请求同时贡献两源 = 计划形态破损，fail-closed
            # （先验于列车头发射——破损计划不得产出任何图节点）。
            overlap = (
                 set().union(*(
                     _gate_request_ids(block["gates"])
                     for block in credit_blocks))
                & set().union(*(
                    _gate_request_ids(block["gates"])
                    for block in prefill_read_blocks)))
            if overlap:
                raise RuntimeError(
                    "request(s) {} carry both decode remote-credit slices "
                    "and a prefill remote-read window in one train (a "
                    "request streams at most one remote read per "
                    "stage)".format(sorted(overlap)))
            credit_blocks = self._merge_remote_read_body_blocks(
                credit_blocks, prefill_read_blocks)
        eviction_watches = self._emit_train_head(train_plan)
        copy_blocks, copy_armed = self._copy_handoff_body_blocks(
            train_plan, list(train_plan["pass_spans"]))
        if (credit_blocks is not None or prefill_read_blocks is not None
                ) and copy_blocks is not None:
            raise RuntimeError(
                "train carries both remote-read body blocks and copy "
                "handoff arms -- a request is either remote-read or copy; "
                "mixed block splitting is unsupported (fail-closed)")
        self._emit_train_body(
            train_plan, list(train_plan["pass_spans"]),
            int(train_plan["iterations"]),
            first_chunk_member=train_plan.get("first_chunk_member"),
            credit_blocks=(credit_blocks if credit_blocks is not None
                           else (prefill_read_blocks
                                 if prefill_read_blocks is not None
                                 else copy_blocks)))
        if copy_armed:
            # C13：交接尾块门账本随体发射消费（joiner 恰一次）；残留尾块
            # 未被任何体块 arm = 切分/发射 bug，fail-closed。K6：层区间
            # 账本同步弹出（段边界已进入首体块层段发射）。
            consumed = self._copy_handoff_arms.pop(copy_armed[0], None)
            self._copy_handoff_layers.pop(copy_armed[0], None)
            if consumed and any(
                    block_index not in {
                        gate[1] for block in (copy_blocks or ())
                        for gate in block["gates"]}
                    for block_index in consumed):
                raise RuntimeError(
                    "copy handoff arms were not fully consumed by the "
                    "train body (un-gated tail chunks would let compute "
                    "outrun the migration stream)")
        # 规格书§三.8：前缀读流组门账本的列车尾检查（首 chunk 成员的
        # 账本必须已被体消费；未消费 fail-closed）。
        self._assert_prefill_read_arms_consumed(train_plan)
        # D10：重建 span 全量核销（声明了抑制而未进体 = 计划/发射布局
        # 失配，fail-closed）。
        self._close_rebuild_output_suppression()
        result = self._emit_train_tail_markers(train_plan, first_token)
        result["eviction_watches"] = eviction_watches
        self._collect(marker)
        return result

    @staticmethod
    def _merge_remote_read_body_blocks(credit_blocks, read_blocks):
        """R4（2026-09-28）：decode credit 体块与 prefill 远读窗体块的
        逐迭代合并（跨请求并存列车，见 emit_iteration_train）。

        两源块序列同为 1..iterations 逐迭代一块（weight_passes=1、span
        同源于 pass_spans 平铺）：span 取 credit 块（`_train_span_iterations`
        的成员步进映射 = 权威口径——成员第 s 步 span 恰落迭代 s，与该成
        员 tile pass 归属对齐），gates 取并集（(rid, 块号) 逐请求键，两
        源请求集已验证不相交，无重复门）。迭代界失配 = 计划形态漂移，
        fail-closed。"""
        if len(credit_blocks) != len(read_blocks):
            raise RuntimeError(
                "remote-read body block streams cover different iteration "
                "counts ({} credit blocks vs {} prefill-read window "
                "blocks) -- train composition drifted".format(
                    len(credit_blocks), len(read_blocks)))
        merged = []
        for credit_block, read_block in zip(credit_blocks, read_blocks):
            span = (credit_block["start_iter"], credit_block["end_iter"])
            if span != (read_block["start_iter"], read_block["end_iter"]):
                raise RuntimeError(
                    "remote-read body block streams disagree on iteration "
                    "coverage (credit {} vs prefill-read {}) -- train "
                    "composition drifted".format(span, (
                        read_block["start_iter"], read_block["end_iter"])))
            merged.append({
                "start_iter": credit_block["start_iter"],
                "end_iter": credit_block["end_iter"],
                "weight_passes": 1,
                "spans": credit_block["spans"],
                "gates": CreditGateSet(
                    _gate_runs(credit_block["gates"])
                    + _gate_runs(read_block["gates"])),
            })
        return merged

    def emit_train_first_step(self, train_plan: dict) -> dict:
        """WP9 首步批发射（2026-08-26；拆分阶段 1）。

        首步批 = 所有列车成员的第 1 个 span（decode 成员首迭代）+ prefill
        队头的第 1 个 chunk，加上按迭代位置锚定在首步内的 joiner 迁移/
        readiness barrier/起始标记/partial 恢复 suffix 门（挂点语义与整列
        发射完全一致，发射序逐节点不变）。列车体以 weight_passes=1 折叠
        （权重恰读一次），体后挂各 debut 成员的 first_token 标记节点
        （1-op COMP，名字含 "first_token" 子串——C++ 锚点按名字子串注册
        code 8，取每 rank min tick）。

        本批不含任何请求级 watch/drain/exit/哨兵标记；唯一附加物是尾部的
        批命名空间唤醒标记（每 rank 1 个小节点，request_id =
        "<train_id>_first_step"，前缀 batch_train_，复用 C++ 哨兵 watch
        通道）：其 fire 经 PREFILL_DRAIN 通道送回调度器，作为余量批的
        交付边界——没有它，无 watch 的首步批完成后不存在任何决策工作，
        C++ tick-end 门不会再交付，运行尾部（全部其余工作已排空）将永久
        等待（死锁）。这是对"首步批无任何 watch 标记"的必要工程化偏移
        （与 sh_1.0/sh_2.0 同构）：唤醒 watch 不挂任何请求、不触发核销/
        记账/决策（调度器按 first_step id 识别后无操作）。

        返回 {"first_token_members": {request_id: {rank: 标记节点 id}},
              "wakeup_members": {rank: 唤醒标记节点 id}}。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        first_token = train_plan.get("first_token")
        if first_token is None or not first_token.get("split"):
            raise RuntimeError(
                "emit_train_first_step requires a split first_token plan")
        if self._train_has_pending_copy_arms(train_plan):
            # C13：首步拆分列车不支持 copy 交接体门控（SH_FIRST_TOKEN_SPLIT
            # 缺省关——B3 裁定后 OFF 为唯一生产姿态；整列发射路径才是
            # 四步协议的体块化载体）。
            raise RuntimeError(
                "WP9 first-step split trains with pending copy handoff "
                "arms are unsupported (whole-train body gating required)")
        if train_plan.get("prefill_read"):
            # D09：首步拆分同样不支持 prefill 远读窗口体门控（整列发射
            # 才是逐 tile 消费驱动发射的载体；SH_FIRST_TOKEN_SPLIT 缺省
            # 关，与 copy 同款 fail-closed 纪律）。
            raise RuntimeError(
                "WP9 first-step split trains with a prefill remote-read "
                "window are unsupported (whole-train body gating "
                "required)")
        # D10：首步拆分不支持静默重建 span（重建段属整列体的前置工作
        # 量，拆两批会把 H_B 重放误当首步迭代——调度侧已降级不拆车，
        # 此处防御 fail-closed）。开账后即查（首步批无抑制面）。
        self._open_rebuild_output_suppression(train_plan)
        if self._rebuild_output_debt:
            raise RuntimeError(
                "WP9 first-step split trains with a silent-rebuild span "
                "are unsupported (whole-train body emission required)")
        marker = self._mark()
        eviction_watches = self._emit_train_head(train_plan)
        self._emit_train_body(
            train_plan, list(first_token["first_spans"]), 1,
            first_chunk_member=train_plan.get("first_chunk_member"),
            credit_blocks=first_token.get("first_body_blocks"))
        # 规格书§三.8：首步批消费前缀读流组门（首 chunk 恒在首步批）——
        # 尾检查钉"首步已消费、余量批不重复消费"。
        self._assert_prefill_read_arms_consumed(train_plan)
        # D10：首步批账本核销（拆分列车禁重建 span，此处恒空——防御性
        # 成对开/关，破损即暴露）。
        self._close_rebuild_output_suppression()
        first_token_members = self._emit_first_token_markers(
            train_plan, first_token["debut_marker_members"])
        group = self.group_by_index[train_plan["instance_index"]]
        wakeup_id = first_token["wakeup_id"]
        for builder in self.builders.values():
            builder.set_context(wakeup_id, "prefill", 0)
        wakeup_members = {
            rank: self._emit_train_marker(rank, f"{wakeup_id}_wakeup")
            for rank in group.ranks
        }
        self.batch_first_step = True
        self._collect(marker)
        return {
            "first_token_members": first_token_members,
            "wakeup_members": wakeup_members,
            "eviction_watches": eviction_watches,
        }

    def emit_train_remainder(self, train_plan: dict) -> dict:
        """WP9 余量批发射（2026-08-26；拆分阶段 2，唤醒交付处调用）。

        余量批 = 剩余迭代（成员第 2 个 span 起）+ prefill 队头剩余 chunk
        （weight_passes = iterations-1，与首步批的 1 次恰合回整列的迭代
        数；激活/KV/AR 逐 span 精确，总量与整列发射一致）。drain/exit/
        哨兵标记与共享 end barrier 全部照常挂本批（挂点语义不变，仍
        "barrier 前末节点"）；decode_length=1 的 debut 成员 exit 标记
        改名携带 first_token 子串（同节点 code 4/8 双锚点，保证
        first_token_ns == completion_ns）。返回结构与
        emit_iteration_train 相同。"""
        if self.config.trace_granularity != "request_aggregated":
            raise RuntimeError(
                "online emission supports request_aggregated granularity "
                "only (production config)")
        first_token = train_plan.get("first_token")
        if first_token is None or not first_token.get("split"):
            raise RuntimeError(
                "emit_train_remainder requires a split first_token plan")
        # D10：余量批开空账本（重建 span 属首步批成员首 span，拆分列车
        # 禁重建——首步批入口已 fail-closed；成对开/关防账本跨批渗漏）。
        self._open_rebuild_output_suppression(train_plan)
        if self._rebuild_output_debt:
            raise RuntimeError(
                "WP9 remainder batches must not carry silent-rebuild "
                "spans (rebuild spans belong to a whole-train emission)")
        marker = self._mark()
        self._emit_train_body(
            train_plan, list(first_token["rest_spans"]),
            int(train_plan["iterations"]) - 1,
            credit_blocks=first_token.get("rest_body_blocks"))
        # 规格书§三.8：余量批不重复消费前缀读流组门（体不传
        # first_chunk_member）——尾检查在首步批已消费时恒通过；残留 =
        # 首步批未消费，fail-closed。
        self._assert_prefill_read_arms_consumed(train_plan)
        self._close_rebuild_output_suppression()
        result = self._emit_train_tail_markers(train_plan, first_token)
        self._collect(marker)
        return result

    def _emit_train_head(self, train_plan: dict) -> list[dict]:
        """列车头：joiner 迁移 + 共享 readiness barrier + 起始标记节点。

        拆分时整段归首步批（迁移/栅栏/起始标记锚定的迭代位置在第 1
        迭代内，语义不变）；发射序与整列发射逐节点一致。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]

        # Validate the previous train's bounded arm/pending ledgers before
        # this train emits any side tail. Within one train the tail is allowed
        # to precede its head, so this check belongs at the transaction entry,
        # not in the per-head emitter.
        credit_owners = {
            row["request_id"] for row in train_plan.get("joiners", ())
            if (row.get("remote_credit_tail")
                or row.get("remote_credit_head"))}
        credit_owners.update(
            member["plan"]["request_id"]
            for member in (train_plan.get("remote_credit") or {}).get(
                "continuations", ()))
        prefill_window = train_plan.get("prefill_read")
        if prefill_window is not None:
            credit_owners.add(prefill_window["request_id"])
        for request_id in credit_owners:
            if (request_id in self._credit_arms
                    or request_id in self._credit_pending_tiles):
                raise RuntimeError(
                    "remote-read head for {} encountered an unsettled prior "
                    "train ledger".format(request_id))

        # ---- joiner 迁移（触发门 = 该成员 drain 列车的 post-barrier
        #      块末；上下文 (joiner, decode, 1) = decode_start 指标锚点） ----
        joiners = list(train_plan.get("joiners", ()))
        eviction_watches = []
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            self._action_seq.setdefault(joiner["request_id"], [0])
            prefix = _prefix_of(joiner)
            drain_gates = joiner.get("prefill_drain_block_ends") or {}
            prefill_group = self.group_by_index[
                joiner["prefill_instance_index"]]
            decode_eviction_trigger = TransferTriggerGate(
                control_instance_index=joiner["prefill_instance_index"],
                node_gates=tuple(
                    drain_gates[rank] for rank in prefill_group.ranks),
            )

            def emit_transfer(transfer, stage_name: str, *,
                              gate=None, trigger_gate=None) -> dict:
                action_state = self._action_seq[joiner["request_id"]]
                action_sequence = action_state[0]
                action_name = (
                    f"{prefix}_{stage_name}_"
                    f"action{action_sequence:03d}_"
                    f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
                )
                record = _emit_kv_transfer(
                    config=self.config,
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    transfer=transfer,
                    action_name=action_name,
                    pending_gate=gate,
                    trigger_gate=trigger_gate,
                )
                record["sequence_stage"] = stage_name
                record["action_sequence"] = action_sequence
                action_state[0] = action_sequence + 1
                # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
                # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
                return record

            # KV 逐出并行化（2026-09-13）：joiner decode 逐出搬进旁路分支
            # ——drain 列车 barrier 触发门保留在分支首节点（逐出开始时刻
            # 不变），分支不 join（逐出物理完成不再阻塞 readiness barrier
            # 与列车体）；紧随其后的 prefill_decode_transfer（恢复类迁移，
            # decode_start 指标锚点）保持主链不动。
            decode_transfers = joiner.get("decode_evictions") or ()
            if decode_transfers:
                watch_id = joiner.get("decode_eviction_watch_id")
                if watch_id is None:
                    watch_id = (
                        "batch_train_evict_{}_decode_{}".format(
                            joiner["request_id"], train_id))
                def emit_decode_evictions() -> None:
                    for transfer in decode_transfers:
                        record = emit_transfer(
                            transfer, "decode_evictions",
                            trigger_gate=decode_eviction_trigger)
                        self._register_store_tails(transfer, record)
                members = self._emit_side_branch(
                    emit_decode_evictions,
                    watch_context=(watch_id, "prefill", 0))
                if members:
                    eviction_watches.append({
                        "request_id": watch_id,
                        "owner_request_id": joiner["request_id"],
                        "branch": "decode_joiner",
                        "members": members})
            # remote-read credit 首列车（唯一执行口径）：切片
            # 块 2..M 旁挂支链（D2 发射序 = checkpoint → 支链尾块 →
            # restore → 块 1 上主链——restore_chain 会把主链回滚到
            # checkpoint（:246-248 回滚语义），块 1 先发射会被回滚抹掉，
            # 次序不可反）。尾块 recv 完成门入 _credit_arms 账本，由
            # 对应体块首节点 arm 消费（块 b ← 门 b，I2）。
            credit_tail = joiner.get("remote_credit_tail") or ()
            if credit_tail:
                def emit_joiner_credit_tail(
                        joiner=joiner, credit_tail=credit_tail) -> None:
                    self._emit_credit_stream_tail(
                        joiner, credit_tail, stage="remote_credit")
                self._emit_side_branch(emit_joiner_credit_tail)
            pd_transfer = joiner.get("prefill_decode_transfer")
            if pd_transfer is None:
                raise RuntimeError(
                    "joiner is missing its Prefill-to-Decode KV action")
            # C4（D19 按缺失区间 copy-in）：joiner 迁移槽可承载多腿
            # （元组形态——增量型第三目标 copy 的 U 迁移 + H 复制两腿）；
            # 腿序即发射序，全部先于 readiness barrier（与单腿同语义）。
            pd_transfers = (
                pd_transfer if isinstance(pd_transfer, tuple)
                else (pd_transfer,))
            for transfer in pd_transfers:
                if (joiner.get("remote_credit_head") is not None
                        and transfer == joiner.get("remote_credit_head")):
                    self._emit_credit_head_transfer(
                        joiner, transfer, stage="remote_credit",
                        pipeline_tile_runs=joiner.get(
                            "remote_credit_pipeline_tile_runs"))
                else:
                    emit_transfer(transfer, "prefill_decode_transfer")

        # ---- remote-read credit 续列车（T2+ 续坐成员，D7 新发射槽位）：
        #      本列车切片块 1 上主链（发射序先于列车体；无 barrier——
        #      per-rank 链序保证先行，§4.1 v3：与 T1 的形态区别仅缺
        #      barrier），块 2..M 旁挂 + arm 门，与 T1 同构。上下文
        #      (member, decode, 1) 与 joiner 槽位同构。 ----
        for member in ((train_plan.get("remote_credit") or {}).get(
                "continuations") or ()):
            member_plan = member["plan"]
            self._set_context(member_plan, "decode", 1)
            tail = member.get("tail") or ()
            if tail:
                def emit_cont_credit_tail(
                        member_plan=member_plan, tail=tail) -> None:
                    self._emit_credit_stream_tail(
                        member_plan, tail, stage="remote_credit")
                self._emit_side_branch(emit_cont_credit_tail)
            self._emit_credit_head_transfer(
                member_plan, member["block1"],
                pipeline_tile_runs=member.get("pipeline_tile_runs"))

        # ---- D09 prefill 远读逐 tile 窗口（2026-09-27）：本列车消费窗
        #      （逐 chunk 迭代）的读流 tile 在列车头发射——tile 1 上主链
        #      （发射序先于列车体；链序保证先行，§4.1 v3 与 decode 续列
        #      车同构），tile 2 即时旁挂、tile 3+ 记待发账本（缓冲复用
        #      释放门 = 消费体块发射点，_emit_credit_stream_tail 同款）。
        #      窗口 tile 登记账本供列车体门并集覆盖检查（fail-closed）。
        #      跨列车 lag-2 由链序承载（本列车头节点在前列车体之后）。
        prefill_read = train_plan.get("prefill_read")
        if prefill_read:
            self._emit_prefill_read_window(prefill_read)

        # ---- 共享 readiness barrier（仅在有 joiner 时发射；无 joiner 的
        #      列车成员 KV 已就绪，无需再栅栏） ----
        if joiners:
            _emit_tp_readiness_barrier(
                builders=self.builders,
                group=group,
                name=f"{train_id}_decode_kv_ready_barrier",
            )

        # ---- 起始标记节点（指标锚点；§3.5：decode_start = 请求加入后
        #      第一个迭代所在列车节点；prefill_start = 请求首个 chunk
        #      所在列车的首节点。joiner 迁移零节点（如 local_hit）时这是
        #      唯一锚点；迁移有节点时 min-tick 语义取更早者，不冲突） ----
        for joiner in joiners:
            self._set_context(joiner, "decode", 1)
            for rank in group.ranks:
                self._emit_train_marker(
                    rank, f"{train_id}_join_"
                    f"{sanitize_node_prefix(joiner['request_id'])}")
        prefill_start_member = train_plan.get("prefill_start_member")
        if prefill_start_member is not None:
            self._set_context(prefill_start_member, "prefill", 0)
            for rank in group.ranks:
                self._emit_train_marker(
                    rank, f"{train_id}_pstart_"
                    f"{sanitize_node_prefix(prefill_start_member['request_id'])}")
        return eviction_watches

    def _emit_train_body(self, train_plan: dict, pass_spans,
                         weight_passes: int, *,
                         first_chunk_member=None,
                         credit_blocks=None) -> None:
        """折叠列车体（17 类聚合节点；weight_passes = 权重读取次数）。

        拆分时首步批传首步 span 组 + weight_passes=1（含队列头首 chunk
        与 partial 恢复 suffix 门挂接），余量批传余量组 + iterations-1；
        两批激活/KV/AR 字节按 span 求和与整列一致。

        remote-read credit 体块化（唯一执行口径，
        credit_blocks 非 None 时）：体按块切分发射——每体块一次
        transformer_pass_aggregated（块内 spans 子集 + 块迭代数的
        weight_passes；跨块求和与整列一致），首节点额外依赖 = 覆盖
        该块迭代区间的全部 remote-read 成员切片块完成门之并集（I2，
        经 _credit_arms 账本逐 rank arm；块 b 不得依赖后续块）。suffix
        恢复门挂首块（首 chunk 恒在首个体块）。credit_blocks=None =
        本列车无 remote-credit 参与，单段发射路径与 v1 逐字节一致。

        多块集体节点命名（A6' 修复，2026-09-22）：C++ commit 预检按
        (pg_name, 节点名) 计集体参与者——同名集体在单 rank 出现 N 次
        即 N participants（期望恰 1），auto-K 多块（M>1）下 8 个体块
        同用 train_id 作 phase 会使 ``{train_id}_all_layers_*_all_
        reduce`` 同名 8 份、预检拒批（C4b 首次后端受控执行暴露）。
        规则与 C13 copy 体块 ``_cb{k}``` 后缀同款：无显式 phase 的
        credit 体块（remote-read 路径，SH 规划侧不带 phase 键）在
        M>1 时逐块挂 ``{train_id}_rcb{b}`` 唯一后缀；M=1 不加后缀
        ——K≥S 单块 = v1 等价锚逐字节不变（I3a）。copy 体块自带
        phase（_copy_handoff_body_blocks）不受影响。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        # partial 流水恢复的 suffix 完成门：首 chunk 主体自拼 batch 改造起
        # 移入列车体，恢复分支仍在准入批发射——其完成门经 _suffix_body_arms
        # 账本挂到含其首 chunk 的列车体首节点（拆分时首 chunk 恒在首步批，
        # 账本在首步批发射处弹出；原"首 chunk 前缀层并行计算"降级为整列车
        # 等恢复，物理保守方向，见类 docstring）。
        suffix_arms = None
        restore_arms = None
        if first_chunk_member is not None:
            suffix_arms = self._suffix_body_arms.pop(
                first_chunk_member["request_id"], None)
            restore_arms = self._suffix_restore_arms.pop(
                first_chunk_member["request_id"], None)
            # D09 prefill 远读逐 tile（2026-09-27）：前缀读流门不再经组
            # 账本——窗口 tile 复用 _credit_arms（列车头登记、体块按
            # gates 消费、列车尾结清）；本处只保留旧组账本的成对登记
            # 检查废除（账本已删）。
            if suffix_arms and restore_arms:
                raise RuntimeError(
                    "train head carries both legacy whole-suffix arms and "
                    "per-group restore arms (plan shape bug)")
            if suffix_arms:
                missing = [
                    rank for rank in group.ranks
                    if suffix_arms.get(rank) is None]
                if missing:
                    raise RuntimeError(
                        "suffix restore gate missing on ranks {}".format(
                            missing))
            if restore_arms:
                # 分组覆盖校验：层区间自驻留前缀顶向上连续铺到 L（不连
                # 续 = 发射/切分 bug，fail-closed）；组序 = 消费顺序。
                expected_start = restore_arms[0][1]
                for _g, layer_start, layer_end, _gates in restore_arms:
                    if layer_start != expected_start or layer_end <= layer_start:
                        raise RuntimeError(
                            "restore group arms are not contiguous in "
                            f"consumption order (got [{layer_start}, "
                            f"{layer_end}), expected start {expected_start})")
                    expected_start = layer_end
                if expected_start != self.config.layers:
                    raise RuntimeError(
                        "restore group arms do not tile the suffix up to the "
                        f"model layer count ({expected_start} != "
                        f"{self.config.layers})")
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        tensor_parallel = len(group.ranks)

        def emit_pass(spans, passes, rank, relative_rank, phase=None,
                      layer_start=0, layer_end=None, include_output=True,
                      extra_gates_by_rank=None,
                      suppress_output_span_indices=None,
                      die_kv_identity="local") -> None:
            """单次聚合体发射漏斗（每 rank 一调）。

            入口 block/credit/readiness gate 的依赖语义不变（arm_dependency
            照旧）；die 后端（第一期）在此按本次发射的 spans/TP/rank 调
            begin_emission 写入发射语境（span 记录带绝对位置——
            derive_span_records 按本仓 span 恒等式还原），并把层区间/
            输出抑制/kv_identity 如实传给 transformer_pass_aggregated 的
            die 分支；legacy（die_context 为 None）发射逐字节不变。"""
            if extra_gates_by_rank is not None:
                gate = extra_gates_by_rank.get(rank)
                if gate is not None:
                    self.builders[rank].arm_dependency(gate)
            die_ctx = None
            if self.die_context is not None:
                self.die_context.begin_emission(
                    phase=stage,
                    tensor_parallel=tensor_parallel,
                    tensor_parallel_rank=relative_rank,
                    span_namespace=train_id,
                    pass_spans=spans,
                )
                die_ctx = self.die_context
            transformer_pass_aggregated(
                self.builders[rank],
                phase=phase if phase is not None else train_id,
                pass_spans=spans,
                layers=self.config.layers,
                hidden_size=self.config.hidden_size,
                ffn_size=self.config.ffn_size,
                tensor_parallel=tensor_parallel,
                pg_name=group.pg_name,
                vocab_size=self.config.vocab_size,
                bytes_per_elem=self.config.bytes_per_elem,
                num_heads=self.config.num_heads,
                tensor_parallel_rank=relative_rank,
                mlp_variant=self.config.mlp_variant,
                weight_passes=passes,
                layer_start=layer_start,
                layer_end=layer_end,
                include_output=include_output,
                suppress_output_span_indices=(
                    suppress_output_span_indices or ()),
                die_ctx=die_ctx,
                die_kv_identity=die_kv_identity,
            )

        def emit_layer_segmented(spans, passes, phase=None,
                                 suppress_indices=None):
            """C15+K6：含首 chunk 计算体的逐层段发射（设计方案 §5.1 逐层
            恢复与 prefill 重叠；2026-09-23 扩展至 copy 交接尾块；2026-09-27
            起 prefill remote-read 前缀门改逐 tile 体块门——组层段废除，
            前缀数据就绪由逐 chunk 体块的窗口 tile 门承担（D09））。

            层段 = [0, 首段起点) 的热前缀段（无门——copy 块 0 在准入主链
            /restore 前的零 KV 层区间）＋各 copy 交接尾块层段（逐 rank 等
            该尾块 recv 完成门——K6/P1-③：C13(4)"未到达块按就绪事件等
            待"的层粒度兑现；原体块粒度门只挂尾块 c，全层聚合 pass 计算
            尾块 c+1..M-1 层不等其到达 = 时序乐观，生产 32 层 4-chunk 必
            现）＋各恢复组层段（逐 rank 等该组目标 HBM 写完成门）。段集
            连续铺满 [0, L]（不连续 = 发射/切分 bug，fail-closed）；每段
            一次 transformer_pass_aggregated（同一 spans 集、该段层区间、
            weight_passes=passes）——跨段求和与整段单次发射逐字节一致
            （激活/KV/AR 按 Σ 段层跨度 × spans、权重按 passes × Σ 层跨
            度、final norm/logits 仅末段一次），物理上首 chunk 的计算
            按层序分段推进：段 i 只等其数据源——**不设**"全部冷层就绪
            才开始 prefill"串行段。
            """
            segments = []
            covered = 0
            copy_gate_request_id = (
                first_chunk_member or {}).get("request_id")
            copy_layers = (
                self._copy_handoff_layers.get(copy_gate_request_id)
                if copy_gate_request_id is not None else None)
            if copy_layers:
                copy_arms = self._copy_handoff_arms.get(
                    copy_gate_request_id, {})
                for chunk_index in sorted(copy_layers):
                    layer_start, layer_end = copy_layers[chunk_index]
                    if layer_start != covered:
                        if layer_start > covered:
                            if covered > 0:
                                # L5（2026-09-23 复核审计）：账本内部缺
                                # 口 = 不变式破坏（_emit_copy_handoff_
                                # tail 登记面恒连续）——静默插无门段会
                                # 让缺口层不等到达即计算（时序乐观），与
                                # 下方 restore 侧同情形 raise 对称。
                                raise RuntimeError(
                                    "copy handoff tail chunk {} layers "
                                    "[{}, {}) leave an internal gap after "
                                    "covered {} (ledger must be "
                                    "contiguous)".format(
                                        chunk_index, layer_start,
                                        layer_end, covered))
                            # 热前缀段（covered==0：copy 块 0 走准入主
                            # 链，无旁挂门——合法无门段）。
                            segments.append((covered, layer_start, None))
                            covered = layer_start
                        else:
                            raise RuntimeError(
                                "copy handoff tail chunk {} layers "
                                "[{}, {}) overlap the covered prefix "
                                "{}".format(
                                    chunk_index, layer_start, layer_end,
                                    covered))
                    arm_map = copy_arms.get(chunk_index)
                    if arm_map is None:
                        raise RuntimeError(
                            "copy handoff arm ledger is missing chunk {} "
                            "of request {} (tail emission must precede "
                            "the train body)".format(
                                chunk_index, copy_gate_request_id))
                    segments.append((layer_start, layer_end, arm_map))
                    covered = layer_end
            if restore_arms:
                first_group_start = restore_arms[0][1]
                if first_group_start > covered:
                    if covered > 0:
                        # L5：copy 尾段与恢复组之间缺口 = 账本并集不变
                        # 式破坏（同上 fail-closed；covered==0 的纯
                        # restore 热前缀仍是合法无门段）。
                        raise RuntimeError(
                            "restore group layers [{}, {}) leave an "
                            "internal gap after covered {} (copy/restore "
                            "ledgers must tile contiguously)".format(
                                first_group_start,
                                restore_arms[0][2], covered))
                    segments.append((covered, first_group_start, None))
                    covered = first_group_start
                for _group, layer_start, layer_end, gates in restore_arms:
                    if layer_start != covered or layer_end <= layer_start:
                        raise RuntimeError(
                            "layer segments are not contiguous in "
                            "consumption order (got [{}, {}) covered "
                            "{})".format(
                                layer_start, layer_end, covered))
                    segments.append((layer_start, layer_end, gates))
                    covered = layer_end
            if not segments:
                raise RuntimeError(
                    "layer-segmented emission invoked with no segments")
            if covered < self.config.layers:
                # 尾部零 KV 层区间（PARTIAL 基零后缀字节形态——无恢复组
                # 覆盖）：无门段补齐到 L（跨段字节守恒需 [0, L) 全覆盖）。
                segments.append((covered, self.config.layers, None))
                covered = self.config.layers
            if covered != self.config.layers:
                raise RuntimeError(
                    "layer segments do not tile the model layer count "
                    "({} != {})".format(covered, self.config.layers))
            last_index = len(segments) - 1
            # D10 静默重建：层段共用同一 spans 集（段=层区间切分），抑制
            # 下标恰消费一次——调用方已按块消费时透传（suppress_indices
            # 非 None），本入口自耗时才在此消费（段内重复消费会撕裂账本）。
            if suppress_indices is None:
                suppress_indices = self._consume_rebuild_output_suppression(
                    spans)
            for position, (layer_start, layer_end, gates) in enumerate(
                    segments):
                for relative_rank, rank in enumerate(group.ranks):
                    emit_pass(
                        spans, passes, rank, relative_rank, phase=phase,
                        layer_start=layer_start, layer_end=layer_end,
                        include_output=(position == last_index),
                        extra_gates_by_rank=gates,
                        suppress_output_span_indices=suppress_indices)

        if credit_blocks is None:
            if restore_arms is None:
                suppress_indices = self._consume_rebuild_output_suppression(
                    pass_spans)
                for relative_rank, rank in enumerate(group.ranks):
                    if suffix_arms:
                        self.builders[rank].arm_dependency(suffix_arms[rank])
                    emit_pass(pass_spans, weight_passes, rank, relative_rank,
                              suppress_output_span_indices=suppress_indices)
                return
            emit_layer_segmented(pass_spans, weight_passes)
            return
        # K6（P1-③）：copy 交接尾块层区间在场 ⇒ 首体块走层段化发射
        # （段边界与逐 rank 门见 emit_layer_segmented）；remote-credit
        # 体块（无 layers 账本）路径不变。
        copy_gate_request_id = (first_chunk_member or {}).get("request_id")
        copy_tail_layers = (
            self._copy_handoff_layers.get(copy_gate_request_id)
            if copy_gate_request_id is not None else None)
        multi_credit_blocks = len(credit_blocks) > 1
        for position, block in enumerate(credit_blocks):
            if not block["spans"]:
                raise RuntimeError(
                    "credit body block [{},{}] has no spans".format(
                        block["start_iter"], block["end_iter"]))
            # A6'：无显式 phase 的 credit 体块在多块形态下逐块挂唯一
            # phase 后缀（同名集体节点单 rank 8 participants 预检拒批
            # 的修复面；单块不加——v1 等价锚逐字节不变）。copy 体块
            # 自带 phase（_cb{k}），原样透传。
            block_phase = block.get("phase")
            if block_phase is None and multi_credit_blocks:
                block_phase = "{}_rcb{}".format(train_id, position + 1)
            # D09 修复轮二（逐 tile 流水消费）：同体块 lag-2 后继（b+2
            # 同属本块门集——tiles_per_pass > 2 的窗内/块内 tile）在本块
            # 计算节点**之前**的逐 tile 消费 marker 点释放——tile t 消费
            # → 槽位释放 → tile t+2 可入，任意时刻每 rank 暂存占用 ≤2
            # tile 的不变量在 N>2 时仍成立（marker 链 = 本块计算对逐
            # tile 数据的流水接收序；跨块后继仍锚定体块末节点，慢计算
            # 不导致源端越过下一消费窗无限泵出）。N ≤ 2 时无同块后继，
            # 发射逐字节与旧协议一致（I3a/I2/D2 锚不变）。
            self._release_intra_block_credit_tiles(block)
            # D10 静默重建：重建 span 随其成员首迭代入首体块——逐块消费
            # 抑制账本（块间 spans 不相交，总和恰一次）。
            block_suppress_indices = (
                self._consume_rebuild_output_suppression(block["spans"]))
            segment_first_block = (
                position == 0
                and (restore_arms is not None or copy_tail_layers))
            for rank in group.ranks:
                if position == 0 and suffix_arms:
                    self.builders[rank].arm_dependency(suffix_arms[rank])
                for gate_request_id, block_index in _iter_gate_pairs(
                        block["gates"]):
                    if segment_first_block and copy_tail_layers:
                        # K6：首体块的 copy 尾块门由层段发射逐段精确
                        # arm（块粒度 arm 与段门并存 = 同门双重依赖，
                        # 冗余且掩盖段边界语义）。
                        continue
                    if block_index < 2 and not (
                            self._copy_handoff_arms.get(
                                gate_request_id, {}).get(block_index)):
                        # 切片块 1 无旁挂门：主链先行性承担（T1 经
                        # barrier、T2+ 经 per-rank 链序，§4.1）。C13 copy
                        # 交接尾块从 1 起编号（块 0 在准入主链）——copy 门
                        # 的块 1 有旁挂支链，须正常 arm。
                        continue
                    arm_map = self._credit_arms.get(gate_request_id)
                    if arm_map is None:
                        arm_map = self._copy_handoff_arms.get(
                            gate_request_id, {}).get(block_index)
                    if arm_map is None:
                        # 尾块发射恒先于体（_emit_train_head 先于
                        # _emit_train_body；copy 尾块在准入批发射）——门
                        # 账本整块缺失只可能是 bug（尾块未发射/未登记/
                        # 块号错位），fail-closed（2026-09-17 交付后复核
                        # 硬化；C13 copy 同款）。
                        raise RuntimeError(
                            "credit arm ledger is missing block {} of "
                            "request {} (tail emission must precede the "
                            "train body)".format(
                                block_index, gate_request_id))
                    if isinstance(arm_map, _CreditArmRange):
                        if arm_map.mainline_covers(block_index, rank):
                            # Same-body virtual consumers are chained through
                            # their bounded marker frontier; only the final
                            # two receive slots need explicit body edges.
                            continue
                        per_rank_gates = arm_map.get(block_index)
                    else:
                        per_rank_gates = arm_map
                    if per_rank_gates is None:
                        raise RuntimeError(
                            "credit arm ledger has no bounded receive anchor "
                            f"for block {block_index} rank {rank}")
                    gate_node = per_rank_gates.get(rank)
                    if gate_node is not None:
                        # 该 rank 有 shard 才有 recv 门；零字节 rank 与
                        # v1 同口径跳过（无数据即无到达可等）。
                        self.builders[rank].arm_dependency(gate_node)
            if segment_first_block:
                # C15+K6+§三.6 组合形态：首块（覆盖迭代 1 = 首 chunk）按
                # 层段切分发射——恢复组门/copy 尾块门/前缀读流组门逐段挂
                # （见 emit_layer_segmented）；块 ≥ 1 原样（其迭代的层消
                # 费已被首块各段门传递覆盖：块 1+ 的 spans 计算晚于首块
                # 全部层段完成 ⇒ 晚于全部尾块/恢复组/前缀组——既有
                # per-block 门成为冗余无害保险）。
                emit_layer_segmented(
                    block["spans"], block["weight_passes"],
                    phase=block_phase,
                    suppress_indices=block_suppress_indices)
                self._record_credit_body_consumptions(block)
                self._release_credit_pending_tiles(block)
                continue
            for relative_rank, rank in enumerate(group.ranks):
                emit_pass(
                    block["spans"], block["weight_passes"],
                    rank, relative_rank,
                    phase=block_phase,
                    suppress_output_span_indices=block_suppress_indices,
                    # die spec 注记（第一期）：remote-read credit 体块的
                    # KV 前缀经 D09 双缓冲暂存到达——kv_access.identity
                    # 如实标 staged（库内无成本语义，只进缓存键/报表）。
                    die_kv_identity="staged")
            self._record_credit_body_consumptions(block)
            # D09 缓冲复用释放门：本体块消费其 gates 列出的全部切片块
            # （fork = 本体块末节点 = 消费锚）——块 t-2 的消费在此归还
            # 暂存缓冲，块 t 的旁挂支链此刻才发射（生产等消费）。
            self._release_credit_pending_tiles(block)

    def _record_credit_body_consumptions(self, block):
        """Bind non-marker tile consumption to the ordinary body tail.

        Same-body consumers already have explicit marker terminal runs. Other
        active target shards are consumed by this block's last task on their
        owner rank. The range is split only at shard tails and marker bounds.
        """
        for gate in _gate_runs(block.get("gates") or ()):
            request_id = gate.request_id
            arm_range = self._credit_arms.get(request_id)
            if not isinstance(arm_range, _CreditArmRange):
                continue
            slots = [slot for slot in self._pipeline_credit_slots.values()
                     if slot["draft"].identity.request_id == request_id]
            if not slots:
                continue
            for slot in slots:
                owner = int(slot["draft"].identity.rank)
                tile_run = slot["tile_run"]
                global_first = int(tile_run["first_tile"])
                global_last = global_first + int(tile_run["tile_count"]) - 1
                first = max(gate.first_tile, global_first)
                last = min(gate.last_tile, global_last)
                if first > last:
                    continue
                body_terminal = self.builders[owner].previous_id
                if body_terminal is None:
                    raise RuntimeError(
                        "remote-read consumer body has no owner-rank tail")

                active_ends = []
                for local_shard, shard in enumerate(slot["owner_shards"]):
                    if tile_run["axis_unit"] == "source_tokens":
                        active_count = (int(tile_run["tile_count"])
                                        if int(shard["bytes_per_unit"]) > 0
                                        else 0)
                    else:
                        amount = int(shard["total_data_bytes"])
                        units = int(tile_run["units_per_tile"])
                        active_count = ((amount + units - 1) // units
                                        if amount else 0)
                    if active_count:
                        active_ends.append((
                            local_shard,
                            min(global_last, global_first + active_count - 1)))
                boundaries = {first, last + 1}
                for _local, active_last in active_ends:
                    if first <= active_last < last:
                        boundaries.add(active_last + 1)
                marker_ranges = []
                for left, right, ranks in arm_range.mainline:
                    if owner not in ranks or right < first or left > last:
                        continue
                    marker_ranges.append((max(first, left), min(last, right)))
                    boundaries.add(max(first, left))
                    boundaries.add(min(last, right) + 1)
                sorted_boundaries = sorted(boundaries)
                for begin, stop in zip(sorted_boundaries,
                                       sorted_boundaries[1:]):
                    if begin >= stop:
                        continue
                    end = stop - 1
                    if any(left <= begin <= right
                           for left, right in marker_ranges):
                        continue
                    local_shards = [
                        local for local, active_last in active_ends
                        if active_last >= end]
                    if not local_shards:
                        continue
                    slot["draft"].add_consume_completion_run({
                        "rank": owner,
                        "group_ordinal": int(slot["group_ordinal"]),
                        "first_tile": begin - global_first + 1,
                        "tile_count": stop - begin,
                        "shard_ordinals": local_shards,
                        "dependency_kind": "ordinary_node_terminal",
                        "first_dependency_json_id": int(body_terminal),
                        "dependency_tile_stride": 0,
                        "dependency_shard_stride": 0,
                    })

    def _release_intra_block_credit_tiles(self, block) -> None:
        """D09 修复轮二：同体块内 lag-2 后继 tile 的逐 tile 消费释放。

        tiles_per_pass（或 prefill 消费窗 tile 数）> 2 时，tile b 与
        b+2 同属一个消费体块——"b+2 生产等 b 消费"的锚不能再是体块末
        节点（计算节点等全部 tile 到达才发射，锚回自身即死锁）。本方法
        在体块计算节点之前逐 tile 发射**消费 marker**（每 tile 每
        target rank 一个 1-op COMP 节点，armed 该 tile 的 per-rank recv
        完成门——tile 1 主链链序天然先行，无旁挂门可 arm），marker 即
        tile 的流水消费锚：tile 数据已被本块计算队列接收、缓冲槽位归
        还。b+2 的旁挂支链从 marker fork（经 _emit_released_credit_tile
        的释放 ack 对），后继 tile 的 recv/门登记照常。

        消费 marker 无物理服务（1 op）：体块计算节点仍 armed 本块全部
        tile 的 recv 门（数据就绪不变量不动）；下一消费窗的首两 tile
        仍由体块末节点释放（跨块后继 = 慢计算停泵面，保守耦合保持）。
        """
        gates = block.get("gates") or ()
        for gate in _gate_runs(gates):
            gate_request_id = gate.request_id
            pending = self._credit_pending_tiles.get(gate_request_id)
            if pending is None or not isinstance(pending, _CreditPendingRange):
                continue
            successor = pending.next_block
            consumer = successor - 2
            if not (gate.first_tile <= consumer <= gate.last_tile
                    and gate.first_tile <= successor <= gate.last_tile
                    and pending.has(successor)):
                continue
            maximum = min(gate.last_tile - successor + 1,
                          pending.last_block - successor + 1)
            runs = getattr(pending.blocks, "pipeline_tile_runs", ())
            if not runs:
                raise RuntimeError(
                    "same-body compact release requires planner source/pass ranges")
            for (consumer_run, successor_run, first_consumer, width,
                 consumer_shards, successor_shards) in \
                    _paired_pipeline_tile_segments(
                        runs, consumer, maximum, lag=2):
                offset = 0
                while offset < width:
                    pair_first = first_consumer + offset
                    if pair_first != pending.next_block - 2:
                        raise RuntimeError(
                            "same-body pair cursor changed during range emission")
                    available = self._consumer_recv_run_remaining(
                        gate_request_id, pending.stage, consumer_run,
                        pair_first, consumer_shards)
                    piece = min(width - offset, available)
                    if piece <= 0:
                        raise RuntimeError(
                            "same-body range has no prior affine receive run")
                    self._emit_compact_same_body_range(
                        gate_request_id, pending, consumer_run, successor_run,
                        pair_first, piece, consumer_shards,
                        successor_shards, pending.stage)
                    pending.advance(piece)
                    offset += piece
            if not pending:
                self._credit_pending_tiles.pop(gate_request_id, None)

    def _consumer_recv_run_remaining(
            self, request_id, stage, tile_run, global_tile, active_shards):
        """Bound a marker range by prior data-recv runs at the real emit points."""
        request_plan = self._pipeline_plan_by_request.get(request_id)
        if request_plan is None:
            raise RuntimeError(
                "same-body receive lookup has no pipeline identity seed")
        remaining = None
        by_owner = {}
        for shard in active_shards:
            owner = int(shard["target_rank"])
            prior = by_owner.get(owner)
            if (prior is None or int(shard["global_shard_ordinal"])
                    > int(prior["global_shard_ordinal"])):
                by_owner[owner] = shard
        for owner, shard in by_owner.items():
            slot = self._physical_pipeline_group(
                request_plan, stage, owner, tile_run)
            local_tile = (int(global_tile)
                          - int(tile_run["first_tile"]) + 1)
            local_shard = slot["global_to_local_shard"].get(
                int(shard["global_shard_ordinal"]))
            match = None
            for run in reversed(slot["draft"].endpoint_runs):
                if (run["group_ordinal"] == slot["group_ordinal"]
                        and run["rank"] == owner
                        and run["first_tile"] <= local_tile
                        < run["first_tile"] + run["tile_count"]
                        and local_shard in run["shard_ordinals"]
                        and any(leg["endpoint"] == "data_recv"
                                for leg in run["legs"])):
                    match = run
                    break
            if match is None:
                raise RuntimeError(
                    "same-body consumer has no prior compact data_recv run "
                    f"for rank {owner}, tile {global_tile}")
            width = (int(match["first_tile"]) + int(match["tile_count"])
                     - local_tile)
            remaining = width if remaining is None else min(remaining, width)
        if remaining is None:
            raise RuntimeError("same-body consumer has no active target rank")
        return remaining

    def _emit_credit_consume_marker_and_release(
            self, request_id: str, block_index: int, transfer,
            action_name: str) -> None:
        """单 tile 消费 marker + lag-2 后继支链发射（体块计算节点前）。

        marker 名携带请求与 tile 序（批内唯一）；exec 侧 marker armed
        tile ``block_index`` 的 per-rank recv 门（账本缺失 = 发射/登记
        链破损，fail-closed——tile 1 主链形态例外，合法无旁挂门）。"""
        recv_map = self._credit_arms.get(request_id, {}).get(block_index)
        marker_ranks = sorted({
            shard.target_rank for shard in transfer.shards})
        if recv_map is None and block_index >= 2:
            raise RuntimeError(
                "credit consume marker for tile {} of request {} has no "
                "registered recv gates (tail emission must precede the "
                "train body)".format(block_index, request_id))
        for rank in marker_ranks:
            gate = (recv_map or {}).get(rank)
            if gate is not None:
                self.builders[rank].arm_dependency(gate)
            # op_role=control_marker（契约 §7 裁定 8）：tile 消费 marker 是
            # comp(1,1) 微节点——die 后端下保持旧微作业语义，不走 die plan。
            self.builders[rank].comp(
                "{}_credit{}_consumed_rank{}".format(
                    sanitize_node_prefix(request_id), block_index, rank),
                1, 1, op_role="control_marker")
        if not marker_ranks:
            raise RuntimeError(
                "credit consume marker for tile {} of request {} has no "
                "shard target ranks (empty tile transfer)".format(
                    block_index, request_id))

        def emit_released(
                request_id=request_id, block_index=block_index,
                transfer=transfer, action_name=action_name):
            member_plan = {"request_id": request_id}
            self._emit_released_credit_tile(
                member_plan, block_index + 2, transfer, action_name)

        self._emit_side_branch(emit_released)

    def _release_credit_pending_tiles(self, block) -> None:
        """D09：体块消费锚处的待发 tile 释放发射（生产/消费门真实缓冲
        复用依赖）。本体块 gates 列出的每个 (rid, 块号 b) 消费完成后，
        块 b+2（若在待发账本）经 _emit_side_branch 发射——fork = 本体块
        末节点；exec 侧 data_recv 挂消费锚父边、home 侧 data_send 被
        释放 ack 栅栏（跨 rank 因果由 tag 配对承载）。同体块 lag-2 后继
        已在 _release_intra_block_credit_tiles（marker 点）提前释放，
        此处只剩跨体块后继（下一消费窗首两 tile——慢计算停泵面）。"""
        for gate in _gate_runs(block["gates"]):
            gate_request_id = gate.request_id
            pending = self._credit_pending_tiles.get(gate_request_id)
            if pending is None or not isinstance(pending, _CreditPendingRange):
                continue
            successor = pending.next_block
            consumer = successor - 2
            if not (gate.first_tile <= consumer <= gate.last_tile
                    and pending.has(successor)):
                continue
            # Same-body markers have already advanced every successor whose
            # consumer and successor both belong to this body block. The
            # remaining prefix is released from the shared body-tail fork.
            count = min(gate.last_tile - consumer + 1,
                        pending.last_block - successor + 1)
            segments = pending.segments(successor, count)
            for tile_run, first_tile, width, active_shards in segments:
                def emit_released_range(
                        gate_request_id=gate_request_id,
                        pending=pending, tile_run=tile_run,
                        first_tile=first_tile, width=width,
                        active_shards=active_shards, stage=pending.stage):
                    self._emit_compact_cross_body_range(
                        gate_request_id, pending, tile_run, first_tile, width,
                        active_shards, stage)
                self._emit_side_branch(emit_released_range)
                pending.advance(width)
            if not pending:
                self._credit_pending_tiles.pop(gate_request_id, None)

    def _rebuild_suppressed_request_ids(self, train_plan: dict) -> set:
        """D10 静默重建（v2.0 §6.2）：本列车声明输出抑制的 joiner 请求
        集（plan_dict 的 joint_suppress_output=True 且阶段 =
        context_rebuild 的 D 边界 recompute 成员；joint_suppress_sampling
        同源成对置位——事件侧无采样算子，采样/RNG 抑制由
        joint_boundary_first_token 元数据承载，本集只承载输出侧副作用的
        抑制面：first_token 观测标记与 final norm/logits 输出位）。"""
        return {
            joiner["request_id"]
            for joiner in (train_plan.get("joiners") or ())
            if joiner.get("joint_suppress_output")
            and joiner.get("joint_prefill_stage") == "context_rebuild"}

    def _open_rebuild_output_suppression(self, train_plan: dict) -> None:
        """D10：开本列车的静默重建 span 欠账（span 值 → 数量）。

        重建 span 形态 = (H_B, H_B)（H_B = joint_decode_rebuild_tokens，
        调度侧 _plan_train 以该形态前置为成员首 span）；本方法只登记，
        匹配消费在体发射点（_consume_rebuild_output_suppression）、全量
        核销在列车尾检查（_close_rebuild_output_suppression）。"""
        debt = {}
        for joiner in (train_plan.get("joiners") or ()):
            if not joiner.get("joint_suppress_output"):
                continue
            if joiner.get("joint_prefill_stage") != "context_rebuild":
                continue
            rebuild_tokens = int(joiner.get("joint_decode_rebuild_tokens")
                                 or 0)
            if rebuild_tokens <= 0:
                continue
            span = (rebuild_tokens, rebuild_tokens)
            debt[span] = debt.get(span, 0) + 1
        self._rebuild_output_debt = debt

    def _consume_rebuild_output_suppression(self, spans) -> tuple:
        """D10：在一段体 spans 内贪心匹配静默重建 span（倒序优先——
        成员 span 位于 chunk span 之后，重建 span 是成员首 span；倒序
        匹配先达成员区，避免与队头首 chunk 的 (c, c) 形态歧义），返回
        本段内被抑制 span 的下标组（供 transformer_pass_aggregated 的
        suppress_output_span_indices）。账本跨体块/层段累计，恰一次。"""
        debt = self._rebuild_output_debt
        if not debt:
            return ()
        indices = []
        for index in range(len(spans) - 1, -1, -1):
            span = tuple(spans[index])
            count = debt.get(span, 0)
            if count <= 0:
                continue
            if count == 1:
                del debt[span]
            else:
                debt[span] = count - 1
            indices.append(index)
        return tuple(sorted(indices))

    def _close_rebuild_output_suppression(self) -> None:
        """D10：列车发射收尾核销——声明的重建 span 必须已全部进入体
        （残留 = 计划/发射布局失配，fail-closed），随后清账供下列车。"""
        debt, self._rebuild_output_debt = self._rebuild_output_debt, {}
        if debt:
            raise RuntimeError(
                "train plan declares silent-rebuild spans {} that never "
                "entered the emitted body (plan/emission layout "
                "mismatch)".format(sorted(debt)))

    def _emit_first_token_markers(self, train_plan: dict,
                                  debut_members) -> dict:
        """WP9 first_token 标记节点（debut 成员各自挂；每 rank 1 个
        1-op COMP 小节点，名字含 "first_token" 子串，C++ 按
        (request_id, rank) 取 min tick 收 code 8 事件）。锚定位置 =
        该成员链条在所在批内列车体后的末节点（共享折叠体 ⇒ 各 debut
        标记顺序挂体后）。

        D10 静默重建（2026-09-27）：输出抑制成员（joint_suppress_output
        =True 的 context_rebuild joiner）不产生 debut first_token 观测
        标记——其首 token 已由 P 末前向产出并恰一次消费（§5.2 归属），
        重建 span 不新增观测点。"""
        train_id = train_plan["train_id"]
        group = self.group_by_index[train_plan["instance_index"]]
        suppressed = self._rebuild_suppressed_request_ids(train_plan)
        members = {}
        for member in debut_members:
            request_id = member["request_id"]
            if request_id in suppressed:
                continue
            self._set_context(member, "decode", 1)
            members[request_id] = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_first_token_"
                    f"{sanitize_node_prefix(request_id)}")
                for rank in group.ranks
            }
        return members

    def _emit_train_tail_markers(self, train_plan: dict,
                                 first_token) -> dict:
        """列车尾：first_token 标记（不拆车增强时）+ drain/exit 标记 +
        哨兵标记 + 共享 end barrier + 块末账本（挂点语义与整列发射
        一致）。"""
        instance_index = train_plan["instance_index"]
        group = self.group_by_index[instance_index]
        train_id = train_plan["train_id"]
        stage = train_plan["stage"]
        generation = 1 if stage == "decode" else 0
        iterations = int(train_plan["iterations"])

        # ---- WP9：不拆车时的 first_token 标记（iterations==1 时整列即
        #      首步，无需两段式；拆车时标记已在首步批挂过，余量批不重挂） ----
        if first_token is not None and not first_token.get("split"):
            self._emit_first_token_markers(
                train_plan, first_token["debut_marker_members"])

        # ---- drain / exit 标记（列车体后、end barrier 前；每成员每 rank
        #      1 个小节点，承载该请求的 PREFILL_DRAIN / DECODE_COMPLETION
        #      watch 与指标 end 锚点；物理完成时刻 = 标记完成时刻。
        #      WP9：decode_length=1 的 debut 成员 exit 标记名附加 first_
        #      token 子串——同节点双锚点（code 4 watch + code 8 名字），
        #      first_token_ns == completion_ns 不变量由同一节点保证。
        #      D10：输出抑制成员（静默重建 joiner）不加 first_token 双
        #      锚点——重建 span 不新增首 token 观测（首 token 归 P 末，
        #      恰一次）；code 4 完成锚点照常（完成非输出侧副作用） ----
        exit_first_token = set(
            (first_token or {}).get("debut_exit_first_token") or ()) - (
            self._rebuild_suppressed_request_ids(train_plan))
        drain_members = {}
        for member in train_plan.get("drain_members", ()):
            request_id = member["request_id"]
            self._set_context(member, "prefill", 0)
            drain_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_drain_"
                    f"{sanitize_node_prefix(request_id)}")
                for rank in group.ranks
            }
        exit_members = {}
        for member in train_plan.get("exit_members", ()):
            request_id = member["request_id"]
            self._set_context(member, "decode", 1)
            exit_members[request_id] = {
                rank: self._emit_train_marker(
                    rank, (
                        f"{train_id}_exit_first_token_"
                        f"{sanitize_node_prefix(request_id)}"
                        if request_id in exit_first_token else
                        f"{train_id}_exit_"
                        f"{sanitize_node_prefix(request_id)}"))
                for rank in group.ranks
            }

        # ---- 共享 end barrier（每列车一个，替代每请求一个） ----
        for builder in self.builders.values():
            builder.set_context(train_id, stage, generation)
        # 哨兵标记(T_max 截断且无自然 drain/exit 标记的列车):
        #      request_id = train_id(批命名空间),固定 (prefill, 0) 单事件
        #      通道;fire 后经 PREFILL_DRAIN 通道送回,调度器按 train_id 核销。
        sentinel_members = {}
        if train_plan.get("sentinel"):
            for builder in self.builders.values():
                builder.set_context(train_id, "prefill", 0)
            sentinel_members = {
                rank: self._emit_train_marker(
                    rank, f"{train_id}_sentinel")
                for rank in group.ranks
            }

        for rank in group.ranks:
            self.builders[rank].all_reduce(
                f"{train_id}_end_barrier",
                iterations,
                group.pg_name,
            )
        block_ends = {
            rank: self.builders[rank].previous_id for rank in group.ranks}
        if any(node_id is None for node_id in block_ends.values()):
            raise RuntimeError("train end barrier node IDs were not generated")

        # ---- 块末账本（post-barrier 口径，五仓一致）：drain 成员写 seg1
        #      （joiner decode_evictions 触发门），exit 成员写 seg2
        #      （completion_evictions 触发门 + 下一 turn interval gate） ----
        for member in train_plan.get("drain_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg1"] = dict(block_ends)
        for member in train_plan.get("exit_members", ()):
            self._block_ends.setdefault(member["request_id"], {})[
                "seg2"] = dict(block_ends)

        # remote-read credit（D09 字节 tile）：清本列车的 arm 账本与待发
        # tile 账本——尾标记每列车恰发射一次（整列发射/余量批同路径），
        # 此刻全部体块消费点均已发射、全部待发 tile 已在消费锚体块点
        # 释放（末体块消费末块 tile ⇒ 无残留）；下一列车该请求的切片在
        # 列车头重新登记（块号从 2 起重用，不残留陈旧门）。O10④：本
        # 列车发射过尾块（remote_credit_tail 非空）的请求在 pop 点必须
        # 有账本——发射登记与尾标记结清同列车成对出现，缺失 = 发射/
        # 记账链破损，fail-closed；单块切片无尾块（账本合法缺席，pop
        # 空放）。待发 tile 残留 = 释放链破损，同 fail-closed。
        credit_plan = train_plan.get("remote_credit") or {}
        expected_arm_owners = set()
        for joiner in train_plan.get("joiners") or ():
            if joiner.get("remote_credit_tail"):
                expected_arm_owners.add(joiner["request_id"])
        for member in credit_plan.get("continuations") or ():
            if member.get("tail"):
                expected_arm_owners.add(member["plan"]["request_id"])
        for request_id in credit_plan.get("request_ids") or ():
            leftover_arms = self._credit_arms.pop(request_id, None)
            if request_id in expected_arm_owners and leftover_arms is None:
                raise RuntimeError(
                    "credit arm ledger for {} is missing at train "
                    "tail-marker settlement although this train emitted "
                    "its credit tail blocks (emission/bookkeeping chain "
                    "broken)".format(request_id))
            leftover_pending = self._credit_pending_tiles.pop(
                request_id, None)
            if leftover_pending:
                raise RuntimeError(
                    "credit pending tiles {} for request {} were never "
                    "released by a consuming body block (buffer-reuse "
                    "release chain broken)".format(
                        leftover_pending.describe_pending(), request_id))

        # D09 prefill 远读逐 tile（2026-09-27）：本列车远读窗口的待发/
        # 门账本同款结清——窗口 tile 全部经体块门覆盖（_assert_prefill_
        # read_arms_consumed 已查）且无残留待发 tile（释放链闭合）；登
        #记账本弹空供下一列车窗口重登记。零 tile 窗口（防御）无账本。
        prefill_read_request = (
            train_plan.get("prefill_read") or {}).get("request_id")
        if prefill_read_request is not None:
            leftover_window = self._prefill_read_window_tiles.pop(
                prefill_read_request, None)
            if leftover_window:
                if self._credit_pending_tiles.pop(
                        prefill_read_request, None):
                    raise RuntimeError(
                        "prefill remote-read pending tiles for request "
                        "{} were never released by a consuming body "
                        "block (buffer-reuse release chain "
                        "broken)".format(prefill_read_request))
                if self._credit_arms.pop(prefill_read_request, None) is None \
                        and len(leftover_window) > 1:
                    raise RuntimeError(
                        "prefill remote-read window ledger for request {} "
                        "was settled without any registered tile arms "
                        "(emission/bookkeeping chain broken)".format(
                            prefill_read_request))

        return {
            "drain_members": drain_members,
            "exit_members": exit_members,
            "sentinel_members": sentinel_members,
            "block_ends": block_ends,
        }

    def _emit_train_marker(self, rank: int, name: str) -> int:
        """列车标记节点（每 rank 1 个小 COMP 节点；上下文由调用方设置）。

        契约 §7 裁定 8：一切 comp(name,1,1) 微节点（first_token/drain/
        exit/sentinel/join/pstart/wakeup 等 marker）一律显式
        op_role="control_marker"——不靠"数值很小"识别（设计文档 §7.3）；
        legacy 后端下该属性惰性（C++ 分发保持旧微作业语义，裁定 3）。"""
        self.builders[rank].comp(name, 1, 1, op_role="control_marker")
        return self.builders[rank].previous_id

    # ---------------------------------- remote-read credit 发射（v2）--

    def _emit_credit_stream_tail(self, member_plan, blocks, *, stage) -> None:
        """remote-read credit 尾块（切片块 2..M）的发射与登记（D09 有限
        字节双缓冲，经 _emit_side_branch 包裹调用——分支首节点 parent
        = fork frontier，分支不 join）：

        - 块 2（第二缓冲）即时旁挂发射：首节点 parent = fork frontier
          （与块 1 主链并行预取——两缓冲初始皆空，块 2 无释放依赖）；
        - 块 3..M 记入 ``_credit_pending_tiles`` 待发账本（名字在头部
          预分配，保证 action 序号稳定）：块 t 的生产受缓冲复用释放门
          约束（消费后归还 credit）——由**消费块 t-2 的体块发射点**
          经 ``_emit_released_credit_tile`` 延迟发射（fork = 该体块末
          节点 = 消费锚；慢计算不导致源端无限泵出，任意时刻远读暂存
          ≤2 块）；
        - 即时块 2 的逐 rank recv 完成门记入 _credit_arms[rid][2]，由
          覆盖其迭代区间的体块 arm 消费（I2）；延迟块的 recv 门在其
          发射点登记，消费点必然更晚（块 t 在消费块 t-2 的逐 tile
          marker/体块末才发射，同块后继经 marker 链保序——修复轮二起
          tiles_per_pass 可 >2，lag-2 恒成立）。

        blocks = ((块号 b, KVTransfer), ...)——块号为切片内绝对序
        （体块门规格按同一块号引用）。M=1 时无尾块（调度器不置
        remote_credit_tail），切片块 1 走主链路径。"""
        request_id = member_plan["request_id"]
        self._pipeline_plan_by_request[request_id] = member_plan
        # A joiner tail side branch may be emitted before its main-chain head
        # transfer. That is one structural train, so the first range can be
        # the tail itself. Stale-arm detection belongs to the head emitter;
        # the pending-range check below still rejects an unclosed prior train.
        if request_id in self._credit_pending_tiles:
            pending = self._credit_pending_tiles[request_id]
            raise RuntimeError(
                "credit pending-tile ledger for {} still holds tiles {} "
                "from a previous train (tail-marker settlement missed "
                "this request -- bookkeeping bug)".format(
                    request_id, pending.describe_pending()))
        block_count = len(blocks)
        if block_count == 0:
            return
        first_block, first_transfer = blocks[0]
        prefix = _prefix_of(member_plan)
        action_state = self._action_seq.setdefault(request_id, [0])
        action_base = action_state[0]
        # Reserve the exact old action sequence interval at the original tail
        # emission point. Future names are derived from ordinals on demand;
        # no named_blocks/action_name array is retained.
        action_state[0] = action_base + block_count
        first_pending_offset = 0
        if first_block < 3:
            immediate_count = min(block_count, 3 - first_block)
            runs = getattr(blocks, "pipeline_tile_runs", ())
            if not runs:
                raise RuntimeError(
                    "compact credit head requires frozen planner tile ranges")
            immediate_segments = _pipeline_tile_run_segments(
                runs, first_block, immediate_count)
            emitted_tiles = 0
            for tile_run, segment_first, width, active_shards in immediate_segments:
                if segment_first >= 3:
                    break
                width = min(width, 3 - segment_first)
                if width <= 0:
                    continue
                tile_offset = segment_first - first_block
                block_index, transfer = blocks[tile_offset]
                if block_index != segment_first:
                    raise RuntimeError(
                        "credit tail block range is not contiguous")
                if width != 1:
                    raise RuntimeError(
                        "the legacy immediate credit window is at most one "
                        "tile; planner segmentation widened it")
                self._emit_compact_initial_range(
                    member_plan, tile_run, transfer, active_shards,
                    stage=stage, first_tile=segment_first,
                    tile_count=width)
                emitted_tiles += width
            if emitted_tiles != immediate_count:
                raise RuntimeError(
                    "compact immediate credit range did not cover its "
                    "legacy emission extent")
            first_pending_offset = immediate_count
        if first_pending_offset < block_count:
            pending = _CreditPendingRange(
                blocks, first_pending_offset=first_pending_offset,
                action_base=action_base, prefix=prefix, stage=stage,
                session_id=first_transfer.session_id,
                kind=first_transfer.kind)
            self._credit_pending_tiles[request_id] = pending

    def _emit_immediate_credit_tile(
            self, member_plan, block_index, transfer, action_name) -> None:
        """无释放依赖的 credit tile（块 2）旁挂发射：结构 = 旧并行支链
        形态（块间无边、fork 主链 frontier；块内 ack_recv ← 本块 send、
        ack_send ← 本块 recv，跨 rank 因果由 tag 配对承载）。"""
        request_id = member_plan["request_id"]
        per_rank = self._emit_credit_tile_shards(
            block_index, transfer, action_name, release_anchor=False)
        self._credit_arms.setdefault(request_id, {})[block_index] = per_rank

    def _emit_released_credit_tile(
            self, member_plan, block_index, transfer, action_name) -> None:
        """缓冲复用释放门下的 credit tile（块 ≥ 3）发射（D09）：调用点
        = 消费块 (block_index - 2) 的体块发射完成处（本方法整体经
        _emit_side_branch 包裹——fork = 该体块末节点，主链不被扰动）。

        逐 shard 新形态：
        - exec 侧：释放 ack_send（1B，tag τ，parent = fork = 消费锚）
          → 本块 data_recv（链序随 ack_send 后；真正的到达因果由 tag
          配对承载）——data_recv 的直接父边锚定"缓冲已归还"；
        - home 侧：释放 ack_recv（tag τ）→ 本块 data_send（链序）——
          源端泵出被释放 ack 栅栏（慢计算 ⇒ 源端停泵）；
        - 到达 ack 沿旧形态（ack_send ← data_recv / ack_recv ← 链序）。
        块的逐 rank data_recv 完成门入 _credit_arms[rid][b]。"""
        request_id = member_plan["request_id"]
        per_rank = self._emit_credit_tile_shards(
            block_index, transfer, action_name, release_anchor=True)
        self._credit_arms.setdefault(request_id, {})[block_index] = per_rank

    def _emit_credit_tile_shards(
            self, block_index, transfer, action_name, *,
            release_anchor: bool) -> dict:
        """单个 credit tile 的逐 shard 物理发射（release_anchor=True 时
        前置缓冲释放 ack 对——D09 生产/消费门真实缓冲复用依赖）。返回
        逐 target rank 的 data_recv 完成门表。"""
        per_rank = {}
        for shard_index, shard in enumerate(transfer.shards):
            if release_anchor:
                release_tag = self.tag_allocator.take()
                # exec 侧：释放 ack_send 挂 fork（消费锚）——信号 =
                # "块 b-2 已消费、缓冲归还"，先于本块 data_recv（链序）。
                self.builders[shard.target_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_credit{block_index}_release_to_rank{shard.source_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=release_tag)
                # home 侧：释放 ack_recv（tag 配对）→ data_send 栅栏。
                self.builders[shard.source_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_credit{block_index}_release_from_rank{shard.target_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=release_tag)
            data_tag = self.tag_allocator.take()
            self.builders[shard.source_rank].comm_send(
                f"{action_name}_shard{shard_index}"
                f"_credit{block_index}_send",
                src=shard.source_rank, dst=shard.target_rank,
                comm_size=shard.bytes, comm_tag=data_tag)
            self.builders[shard.target_rank].comm_recv(
                f"{action_name}_shard{shard_index}"
                f"_credit{block_index}_recv",
                src=shard.source_rank, dst=shard.target_rank,
                comm_size=shard.bytes, comm_tag=data_tag)
            recv_node = self.builders[shard.target_rank].previous_id
            if recv_node is None:
                raise RuntimeError(
                    "credit stream recv did not generate a node")
            per_rank[shard.target_rank] = recv_node
            ack_tag = self.tag_allocator.take()
            self.builders[shard.target_rank].comm_send(
                f"{action_name}_shard{shard_index}"
                f"_credit{block_index}_ack_to_rank{shard.source_rank}",
                src=shard.target_rank, dst=shard.source_rank,
                comm_size=1, comm_tag=ack_tag)
            self.builders[shard.source_rank].comm_recv(
                f"{action_name}_shard{shard_index}"
                f"_credit{block_index}_ack_from_rank{shard.target_rank}",
                src=shard.target_rank, dst=shard.source_rank,
                comm_size=1, comm_tag=ack_tag)
        return per_rank

    def _emit_prefill_read_window(self, prefill_read: dict) -> None:
        """D09 prefill 远读逐 tile 窗口发射（2026-09-27）：本列车消费窗
        （逐 chunk 迭代）的读流 tile——tile 1 上主链（发射序先于列车体，
        链序保证先行）、tile 2 即时旁挂、tile 3+ 记 ``_credit_pending_
        tiles`` 待发账本（块 t 的生产受缓冲复用释放门约束：由消费块
        t-2 的体块发射点经 ``_emit_released_credit_tile`` 延迟发射——
        tile i+2 生产等 tile i 消费完成，慢计算不导致源端无限泵出，
        任意时刻远读暂存 ≤2 tile）。完成门与待发账本与 decode credit
        共用（_credit_arms/_credit_pending_tiles 同一协议）；窗口 tile
        序号集登记 ``_prefill_read_window_tiles``（列车体门并集覆盖检
        查 + 列车尾结清）。

        prefill_read 字段（调度器 _plan_train 冻结）：
          request_id / plan（_prefix_of 所需的请求 plan dict 子集）/
          tiles = ((tile 序号 1..M, KVTransfer), ...)——列车内窗口
          展平序；body_blocks 由 _emit_train_body 经 credit_blocks 槽位
          消费（gates = (rid, tile) 对）。"""
        request_id = prefill_read["request_id"]
        member_plan = prefill_read["plan"]
        self._pipeline_plan_by_request[request_id] = member_plan
        tiles = prefill_read.get("tiles")
        tile_count = 0 if tiles is None else len(tiles)
        if tile_count == 0:
            raise RuntimeError(
                "prefill remote-read window for request {} carries no "
                "tiles (a consuming train must be preceded by its window "
                "plan)".format(request_id))
        if request_id in self._prefill_read_window_tiles:
            raise RuntimeError(
                "prefill remote-read window ledger for {} still holds "
                "tiles {} from a previous train (tail-marker settlement "
                "missed this request)".format(
                    request_id,
                    "{}..{} ({} tiles)".format(
                        self._prefill_read_window_tiles[request_id].start,
                        self._prefill_read_window_tiles[request_id].stop - 1,
                        len(self._prefill_read_window_tiles[request_id]))))
        self._prefill_read_window_tiles[request_id] = range(1, tile_count + 1)
        self._set_context(member_plan, "prefill", 0)
        # tile 1 上主链（首块数据先于列车体到达的链序保证；与 decode
        # 续列车块 1 同构——无 barrier，per-rank 链序承担先行性）。
        transfer_tiles = _TransferPayloadSequence(tiles)
        first_tile = tiles[0]
        if (isinstance(first_tile, tuple) and len(first_tile) == 2
                and type(first_tile[0]) is int
                and first_tile[0] != 1):
            raise RuntimeError(
                "prefill remote-read window must begin at tile 1")
        self._emit_credit_head_transfer(
            member_plan, transfer_tiles[0], stage="prefill_read",
            pipeline_tile_runs=getattr(
                tiles, "pipeline_tile_runs", ()))
        if tile_count > 1:
            tail_blocks = _IndexedTransferBlocks(
                transfer_tiles, first_offset=1, first_block=2)
            def emit_prefill_read_tail(
                    member_plan=member_plan, tail_blocks=tail_blocks):
                self._emit_credit_stream_tail(
                    member_plan, tail_blocks, stage="prefill_read")
            self._emit_side_branch(emit_prefill_read_tail)

    def _emit_copy_handoff_tail(self, member_plan, chunks, *,
                                pending_gate=None) -> None:
        """C13 copy 交接尾块（块 1..M-1）旁挂支链的物理发射（四步协议
        的图侧 (2)：源 HBM 读取 + 传输；目标 HBM 写完成 = exec 侧 recv
        节点——逐块就绪门控/交接完成事件挂点；ack_recv = 源端释放依赖
        挂点）。经 _emit_side_branch 包裹调用——分支首节点 parent =
        fork frontier，分支不 join。

        P1（PARTIAL 跨实例 copy 流水化 2026-09-25）：尾块逐笔并行支链
        ——每块对涉及 rank 做 chain_checkpoint/restore_chain 段内分支：

        - 块间无边：全部块的首节点 parent = fork frontier（头块后的主
          链），home send / exec recv 逐笔互并发（首块仍是唯一启动栅
          栅栏——块 0 在准入主链，本方法只接块 ≥ 1）；
        - 块内父边为新形态：ack_recv_c ← 本块 send_c（source builder
          链序）、ack_send_c ← 本块 recv_c（target builder 链序）；跨
          rank send/recv 无图边，因果由 tag 配对在运行时承载（
          CommonNetworkApi：流完成后迟到的 sim_recv 立即补回调——任意
          发射序安全）；旧「home send 直连链 / exec recv 顺序链 / ack
          链尾随整条数据链」的整段串行依赖移除，不保留；
        - 块 c 的逐 rank recv 完成门记入 _copy_handoff_arms[rid][c]，
          由含首 chunk 的列车体按消费顺序逐块 arm 消费（I2 门并集同源；
          块 c 的消费区间覆盖由 _copy_handoff_body_blocks 比例映射）；
        - 逐块 ack_recv 节点记入 _copy_handoff_release_anchors[rid][c]
          （ack_recv 物理到达事实锚；账本侧在 prefill drain 边界逐块
          结算释放，见 FS _settle_copy_handoffs）。

        触发门：gate 实例 == 源实例时每块逐 source rank 重 arm timer
        gate（与头块 noc_migrate 主链同策略；每块 send 都带门依赖，等
        价旧链式「全部尾块等 gate」）；跨实例 gate（上轮执行实例）时
        无门控发射（NEW-1 同款）。
        """
        request_id = member_plan["request_id"]
        prefix = _prefix_of(member_plan)
        action_state = self._action_seq.setdefault(request_id, [0])
        stage = "history_handoff"
        named_chunks = []
        for transfer in chunks:
            chunk_index = transfer.handoff_chunk
            if transfer.kind != "noc_migrate":
                raise RuntimeError(
                    "copy handoff tail chunk must be a NoC migration "
                    f"(got {transfer.kind!r})")
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            action_state[0] = action_sequence + 1
            named_chunks.append((chunk_index, transfer, action_name))
        named_chunks.sort(key=lambda item: item[0])
        for chunk_index, transfer, action_name in named_chunks:
            involved_ranks = []
            for shard in transfer.shards:
                for rank in (shard.source_rank, shard.target_rank):
                    if rank not in involved_ranks:
                        involved_ranks.append(rank)
            checkpoints = {
                rank: self.builders[rank].chain_checkpoint()
                for rank in involved_ranks}
            # 触发门 arming（分支内完成——_emit_side_branch 的
            # stash-and-clear 契约）：仅 gate 实例 == 源实例时（头块同
            # 策略）；_history_control 的 control_rank==source_rank 校验
            # 保留。
            if (
                pending_gate is not None
                and transfer.source_instance_index is not None
                and pending_gate.source_instance_index
                == transfer.source_instance_index
            ):
                source_group = self.group_by_index[
                    transfer.source_instance_index]
                for relative_index, source_rank in enumerate(
                        source_group.ranks):
                    control_rank, timer_gate = _history_control(
                        group_by_index=self.group_by_index,
                        pending_gate=pending_gate,
                        relative_index=relative_index,
                    )
                    if control_rank != source_rank:
                        raise RuntimeError(
                            "copy handoff tail control rank is not its "
                            "source rank")
                    self.builders[source_rank].arm_timer_gate(timer_gate)
            per_rank = {}
            release_per_rank = {}
            for shard_index, shard in enumerate(transfer.shards):
                data_tag = self.tag_allocator.take()
                self.builders[shard.source_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_send",
                    src=shard.source_rank, dst=shard.target_rank,
                    comm_size=shard.bytes, comm_tag=data_tag)
                self.builders[shard.target_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_recv",
                    src=shard.source_rank, dst=shard.target_rank,
                    comm_size=shard.bytes, comm_tag=data_tag)
                recv_node = self.builders[shard.target_rank].previous_id
                if recv_node is None:
                    raise RuntimeError(
                        "copy handoff recv did not generate a node")
                per_rank[shard.target_rank] = recv_node
                # 块内 ack 新形态：ack_send 尾随本块 recv（target 链序）、
                # ack_recv 尾随本块 send（source 链序）。
                ack_tag = self.tag_allocator.take()
                self.builders[shard.target_rank].comm_send(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_ack_to_rank{shard.source_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=ack_tag)
                self.builders[shard.source_rank].comm_recv(
                    f"{action_name}_shard{shard_index}"
                    f"_handoff{chunk_index}_ack_from_rank{shard.target_rank}",
                    src=shard.target_rank, dst=shard.source_rank,
                    comm_size=1, comm_tag=ack_tag)
                ack_node = self.builders[shard.source_rank].previous_id
                if ack_node is not None:
                    release_per_rank[shard.source_rank] = ack_node
            self._copy_handoff_arms.setdefault(
                request_id, {})[chunk_index] = per_rank
            # K6（P1-③）：尾块层区间随发射登记（首体块层段门控的段
            # 边界——块 c 的 recv 门只应 gate 其层区间 [start, end) 的
            # 计算段，全层聚合 pass 在 c 之前的层段不等 c 即可计算）。
            self._copy_handoff_layers.setdefault(
                request_id, {})[chunk_index] = (
                    transfer.layer_start, transfer.layer_end)
            self._copy_handoff_release_anchors.setdefault(
                request_id, {})[chunk_index] = release_per_rank
            for rank in involved_ranks:
                self.builders[rank].restore_chain(checkpoints[rank])

    def _train_has_pending_copy_arms(self, train_plan) -> bool:
        """本列车成员/队列头是否有待消费的 copy 交接尾块门（C13）。"""
        if not self._copy_handoff_arms:
            return False
        request_ids = {
            request_id for request_id, _ in train_plan.get("members") or ()}
        request_ids.add(train_plan.get("head_request_id"))
        request_ids.discard(None)
        return any(request_id in self._copy_handoff_arms
                   for request_id in request_ids)

    def _assert_prefill_read_arms_consumed(self, train_plan) -> None:
        """D09 prefill 远读逐 tile（2026-09-27）：本列车载有远读窗口时，
        窗口登记的 tile 序号集必须恰被体块门并集覆盖——欠覆盖 = 读流
        tile 无消费门（数据到达不被计算等待，时序乐观），过覆盖 = 门引
        用未登记 tile（账本错位）；两端 fail-closed。非远读列车无账本，
        恒通过。"""
        prefill_read = train_plan.get("prefill_read") or {}
        request_id = prefill_read.get("request_id")
        if request_id is None:
            return
        registered = self._prefill_read_window_tiles.get(request_id)
        if not registered:
            # 无登记 = 本列车无远读窗口（窗口在后续列车）或窗口零 tile
            # （零字节流）：恒通过；残留由列车尾结清与完成批检查兜底。
            return
        expected_tile = 1
        for block in prefill_read.get("body_blocks") or ():
            for gate_request_id, tile_index in _iter_gate_pairs(
                    block["gates"]):
                if gate_request_id != request_id:
                    raise RuntimeError(
                        "prefill remote-read body block gates reference "
                        f"another request {gate_request_id!r} under "
                        f"window owner {request_id!r}")
                if tile_index != expected_tile:
                    raise RuntimeError(
                        "prefill remote-read body tiles for request {} are "
                        "not a contiguous single-use prefix (expected {}, "
                        "got {})".format(
                            request_id, expected_tile, tile_index))
                expected_tile += 1
        if expected_tile - 1 != len(registered):
            raise RuntimeError(
                "prefill remote-read window tiles for request {} were not "
                "exactly covered by the train body gates (registered {} vs "
                "covered {}) -- the per-tile read stream must gate exactly "
                "the blocks that consume it".format(
                    request_id, len(registered), expected_tile - 1))

    def _copy_handoff_body_blocks(self, train_plan, pass_spans):
        """C13：含首 chunk 列车体的逐块就绪门控切分（消费顺序映射）。

        体块数 B = min(尾块数 + 1, 迭代数)（首体块无门——readiness
        barrier 已等交接块 0；迭代数不足时尾块并入最后体块，仍保证
        "列车体完成 ⇒ 全部尾块到达"⇒ prefill drain 结算因果成立）。尾块
        c → 体块 min(c, B)：1:1 流水（体块 c 等尾块 c，读流与计算重
        叠，不设"先整份搬运后计算"串行段）。

        span 划分（布局契约：SH _plan_train 平铺序 = [队列头 chunk
        spans（每迭代恰一条，共 iterations 条）][成员连续段]——GB 侧
        train_plan 不携带 members/prefill_chunk_tokens，头部 span 恒为
        前 iterations 条）：头部 span 按迭代区间精确入块（迭代 i ↔ 第
        i 条）；成员 span 按块迭代数比例确定性地分配到尾随块（每 span
        恰入一块、块序保持；权重字节按块迭代数——跨块求和与整列一
        致）。返回 (blocks, (request_id,)) 或 (None, ())。
        """
        first_chunk_member = train_plan.get("first_chunk_member")
        if first_chunk_member is None:
            return None, ()
        request_id = first_chunk_member["request_id"]
        arms = self._copy_handoff_arms.get(request_id)
        if not arms:
            return None, ()
        tail_indices = sorted(arms)
        iterations = int(train_plan["iterations"])
        if iterations <= 0:
            raise RuntimeError(
                "copy handoff train has no iterations to gate")
        if len(pass_spans) < iterations:
            raise RuntimeError(
                "copy handoff train span layout does not match the "
                "frozen plan (fewer spans than head iterations)")
        head_spans = pass_spans[:iterations]
        member_spans = pass_spans[iterations:]
        block_count = min(len(tail_indices) + 1, iterations)
        per_block = -(-iterations // block_count)
        # C15 修复（C13 先在缺陷，跨车道披露——非 C15 引入）：per_block
        # 上取整可使尾块数超出覆盖迭代所需块数，空尾块（end_iter <
        # start_iter）在 stress 10s 窗实测 raise（crash 形 (it, tails) ∈
        # {(4,2),(5,3),(6,3),(9,3),(6,4),(7,4),…}）。收紧 block_count 至
        # 全覆盖所需最小块数；多出尾块经 min(tail_index, block_count)
        # 并入末块（与"迭代数不足时尾块并入最后体块"的既有披露语义一
        # 致，"体块完成 ⇒ 全部尾块到达"不变量保持）。
        block_count = min(block_count, -(-iterations // per_block))
        # 成员 span 的比例分配（确定性，块序保持；最后块吃余量）。
        member_allocation: list[int] = []
        assigned_members = 0
        remaining_members = len(member_spans)
        remaining_iters = iterations
        for block_position in range(1, block_count + 1):
            count = min(
                per_block, iterations - (block_position - 1) * per_block)
            if block_position == block_count:
                take = remaining_members
            else:
                take = remaining_members * count // max(1, remaining_iters)
            member_allocation.append(take)
            assigned_members += take
            remaining_members -= take
            remaining_iters -= count
        if assigned_members != len(member_spans):
            raise RuntimeError(
                "copy handoff member-span allocation lost spans "
                f"({assigned_members} != {len(member_spans)})")
        member_cursor = 0
        blocks = []
        for block_position in range(1, block_count + 1):
            start_iter = (block_position - 1) * per_block + 1
            end_iter = min(block_position * per_block, iterations)
            if end_iter < start_iter:
                raise RuntimeError(
                    "copy handoff body block covers no iterations")
            take = member_allocation[block_position - 1]
            spans = (
                list(head_spans[start_iter - 1:end_iter])
                + member_spans[member_cursor:member_cursor + take])
            member_cursor += take
            if not spans:
                raise RuntimeError(
                    "copy handoff body block covers no spans")
            gates = [
                (request_id, tail_index)
                for tail_index in tail_indices
                if min(tail_index, block_count) == block_position]
            blocks.append({
                "start_iter": start_iter,
                "end_iter": end_iter,
                "weight_passes": end_iter - start_iter + 1,
                "spans": spans,
                "gates": gates,
                # 块唯一 phase 标签：同列车多体块的聚合节点/集体节点不
                # 可重名（C++ commit 预检按名字对集体做跨 rank 签名一
                # 致性校验——同块内逐 rank 字节一致，跨块字节不同，重
                # 名即误配；remote-credit 多块在现行 auto-K 下恒单块，
                # 本后缀是首个真实多块路径的必要区分）。
                "phase": f"{train_plan['train_id']}_cb{block_position}",
            })
        return blocks, (request_id,)

    def _emit_credit_head_transfer(
            self, member_plan, transfer, *, stage: str = "remote_credit",
            pipeline_tile_runs=None,
    ) -> dict:
        """remote-read 切片块 1 的主链发射（exec 实例 per-rank 链序保
        证其先于列车体；无 barrier；decode 续列车与 prefill 远读窗口
        tile 1 共用——``stage`` 区分命名/登记身份）。发射走 v1 同一
        _emit_kv_transfer noc_migrate 路径，命名经 _action_seq 闭包
        （与 joiner pd_transfer 同款手法）。"""
        request_id = member_plan["request_id"]
        self._pipeline_plan_by_request[request_id] = member_plan
        runs = tuple(pipeline_tile_runs or ())
        if not runs:
            raise RuntimeError(
                "remote-read head requires its frozen planner tile ranges")
        run = next((row for row in runs
                    if int(row["first_tile"]) <= 1
                    < int(row["first_tile"]) + int(row["tile_count"])),
                   None)
        if run is None:
            raise RuntimeError(
                "remote-read planner ranges do not contain head tile 1")
        active = _active_shards_at(run, 1)
        action_state = self._action_seq.setdefault(request_id, [0])
        action_sequence = action_state[0]
        action_state[0] = action_sequence + 1
        emitted = self._emit_compact_initial_range(
            member_plan, run, transfer, active, stage=stage,
            first_tile=1, tile_count=1)
        return {
            "sequence_stage": stage,
            "action_sequence": action_sequence,
            "pipeline_extent": emitted,
        }

    def emit_completion_batch(self, request_plan: dict, *,
                              block_end_stage: str = "seg2") -> dict:
        """completion 批（DECODE_COMPLETION/REQUEST_COMPLETE 边界）：
        completion_evictions + 下一 turn interval gate 的依赖登记。
        R11（2026-09-14）：merge 流存在时返回 merge 尾标记节点表
        （{"merge_done_members": {rank: id}, "has_merge": bool}）——调度器
        据此注册 merge-done watch（到达重锚 merge_done）并把下一轮
        interval gate 前递依赖挂到标记上；无 merge 流返回 has_merge=False
        （行为与改造前一致）。

        D18（2026-09-26）：``block_end_stage`` = completion 触发门锚的
        块末账本段（缺省 "seg2" = 退出迭代列车块末，全部常规请求不变；
        已知终止请求无退出列车，调度器终端结算路径传 "seg1"——drain
        块末即 P 完成，语义精确）。"""
        self._set_context(request_plan, "completion", 1)
        marker = self._mark()
        merge_info = self._emit_completion(
            request_plan, block_end_stage=block_end_stage)
        self._collect(marker)
        # M4 核销即删（2026-08-23）：completion 批是本请求图发射的终点
        # （seg2 块末在 _emit_completion 内消费、action 序号此后无读者，
        # 全仓 grep 证实无更晚读者）——逐出块末/action 计数账本条目在请求
        # 完成后即死重，当场弹出（下一 turn 是不同 request_id）。
        self._block_ends.pop(request_plan["request_id"], None)
        self._action_seq.pop(request_plan["request_id"], None)
        # C13：交接门账本残留 = 请求完成而尾块从未被体块消费（发射/切分
        # bug）——fail-closed；释放挂点账本为审计面，完成后弹出。
        leftover_arms = self._copy_handoff_arms.pop(
            request_plan["request_id"], None)
        if leftover_arms:
            raise RuntimeError(
                "request {} completed with unconsumed copy handoff arms "
                "{} -- the migration stream was never gated into a train "
                "body".format(
                    request_plan["request_id"], sorted(leftover_arms)))
        self._copy_handoff_layers.pop(request_plan["request_id"], None)
        self._copy_handoff_release_anchors.pop(
            request_plan["request_id"], None)
        # C15：恢复组门账本残留 = 请求完成而恢复组从未被体块消费（发射/
        # 切分 bug）——fail-closed（C13 copy 同款纪律）。
        leftover_restore_arms = self._suffix_restore_arms.pop(
            request_plan["request_id"], None)
        if leftover_restore_arms:
            raise RuntimeError(
                "request {} completed with unconsumed restore group arms "
                "{} -- the suffix restore stream was never gated into a "
                "train body".format(
                    request_plan["request_id"],
                    [arm[0] for arm in leftover_restore_arms]))
        # O10④：credit arm 账本残留 = 请求完成而其切片尾块门账本未被
        # 列车尾标记结清（发射/切分 bug）——fail-closed（C13 copy /
        # C15 restore 同款纪律；单块切片无尾块，账本合法缺席）。D09：
        # 待发 tile 账本残留 = 缓冲复用释放链破损，同 fail-closed。
        leftover_credit_arms = self._credit_arms.pop(
            request_plan["request_id"], None)
        if leftover_credit_arms:
            raise RuntimeError(
                "request {} completed with unconsumed credit arms "
                "{} -- the credit tail stream was never gated into a "
                "train body".format(
                    request_plan["request_id"],
                    sorted(leftover_credit_arms)))
        leftover_credit_pending = self._credit_pending_tiles.pop(
            request_plan["request_id"], None)
        if leftover_credit_pending:
            raise RuntimeError(
                "request {} completed with unreleased credit tiles {} -- "
                "the buffer-reuse release chain was never anchored into a "
                "train body".format(
                    request_plan["request_id"],
                    leftover_credit_pending.describe_pending()))
        # D09 prefill 远读逐 tile（2026-09-27）：窗口 tile 登记账本残留 =
        # 请求完成而其读流窗口从未被列车体门覆盖（发射/切分 bug）——
        # fail-closed（C13 copy / C15 restore 同款纪律；非远读请求无账本，
        # pop 空放）。
        leftover_window_tiles = self._prefill_read_window_tiles.pop(
            request_plan["request_id"], None)
        if leftover_window_tiles:
            if isinstance(leftover_window_tiles, range):
                window_extent = (leftover_window_tiles.start,
                                 leftover_window_tiles.stop - 1,
                                 len(leftover_window_tiles))
            else:
                tile_ids = tuple(sorted(leftover_window_tiles))
                window_extent = (
                    tile_ids[0] if tile_ids else 0,
                    tile_ids[-1] if tile_ids else 0,
                    len(tile_ids))
            raise RuntimeError(
                "request {} completed with unconsumed prefill remote-read "
                "window tiles {} -- the prefix read stream was never gated "
                "into a train body".format(
                    request_plan["request_id"],
                    "{}..{} ({} tiles)".format(
                        *window_extent)))
        return merge_info

    def _set_context(self, request_plan: dict, stage: str,
                     generation: int) -> None:
        request_id = request_plan["request_id"]
        for builder in self.builders.values():
            builder.set_context(request_id, stage, generation)

    # ------------------------------------------------- per-request 发射主体 --

    def sync_pending_history_after_evictions(self, transfers) -> None:
        """决策时点补偿：KV 变更点返回的逐出转移，立即镜像到 pending 门。
        唯一标记路径（2026-08-23 seq4689 修订：发射侧重复标记在多级逐出
        发射乱序时会把门回退到过期位置，已移除）。partial_hbm_remote /
        remote_memory 的半驻留语义由 _mark_pending_history_store 自身推导。"""
        for transfer in transfers or ():
            if transfer.kind == "remote_store":
                self._mark_pending_history_store(transfer)

    def _mark_pending_history_store(self, transfer) -> None:
        """remote_store 发射后的会话位置登记
        （resident_prefix_layers_after 折算 location）。"""
        if transfer.resident_prefix_layers_after == 0:
            location = "remote_memory"
        elif transfer.resident_prefix_layers_after < transfer.model_layers:
            location = "partial_hbm_remote"
        else:
            raise RuntimeError("remote store did not reduce resident KV layers")
        session_id = transfer.session_id
        pending_request_id = self.pending_request_by_session.get(session_id)
        if pending_request_id is None:
            self.deferred_session_locations[session_id] = location
            return
        gate = self.pending_history.get(pending_request_id)
        if gate is None:
            # Backport 2026-08-16 (对比报告 §5.2, verified in the 3-min
            # test copies): an in-flight turn-0 request has its session key
            # registered in pending_request_by_session at prefill emission
            # WITHOUT a pending_history gate (turn-0's gate is built inline
            # as "new_session" and never stored; turn>0 gates are popped at
            # arrival). When another request's completion eviction
            # (remote_store) targets this session, the old direct subscript
            # raised KeyError (20/50 档 delivery 3575/1553 两起历史事故,
            # sh_2.0 改造实录登记). Defer to
            # session completion instead -- mirroring the turn>0 no-entry
            # semantics above (offline planner equivalence: the eviction
            # applies to the session state; the next turn's
            # history_location_before reflects the post-eviction location).
            self.deferred_session_locations[session_id] = location
            return
        gate.location = location

    # ------------------------------------ KV 逐出旁路支链（2026-09-13）--

    def _emit_side_branch(
            self, emit_fn, *, watch_context=None) -> dict[int, int]:
        """把一段逐出发射包成旁路分支（主方案 §3.2）：全 rank 暂存并清空
        既有 pending 依赖 → fork 快照 → 发射（分支内自行接续成链；触发门
        的 arming 必须在 emit_fn 内部完成）→ 恢复主链 → 归还暂存依赖。
        分支不 join——被包裹的物理逐出链不再阻塞主链任何节点，HBM 争用由
        C++ LocalHbmBandwidthModel 的 N-way 均分模型在线裁决。

        fork 前非空 pending 的处理（对任务书原案的偏差，REPORT
        DEVIATIONS 登记）：S3 源码现实中 prefill_evictions 发射点之前存在
        合法的主链 armed 依赖（turn-0 arrival gate 的 arm_timer_gate 与
        local_hit history 的 arm），按原案字面 fail-closed 会在正常流量
        崩溃。改为 stash-and-clear 的语义等价实现：分支首节点 parent 恰为
        fork frontier（不带任何 armed 门），主链恢复后原依赖照常由其本来
        的消费者（readiness barrier 等）消费——与逐出链在主链上时完全一致
        的图依赖。分支内部 arming 而未被任何节点消费（触发门泄漏）仍
        fail-closed。
        """
        stashed = {}
        node_counts = {}
        for rank, builder in self.builders.items():
            stashed[rank] = builder.pending_extra_dependencies
            node_counts[rank] = len(builder.nodes)
            if builder.pending_extra_dependencies:
                builder.pending_extra_dependencies = []
        checkpoints = {
            rank: builder.chain_checkpoint()
            for rank, builder in self.builders.items()
        }
        try:
            emit_fn()
            for builder in self.builders.values():
                if builder.pending_extra_dependencies:
                    raise RuntimeError(
                        "side-branch left unconsumed pending deps on "
                        f"rank {builder.rank}")
            members = {}
            touched_ranks = [
                rank for rank, builder in self.builders.items()
                if len(builder.nodes) > node_counts[rank]
            ]
            if watch_context is not None and touched_ranks:
                request_id, stage, generation = watch_context
                for rank in touched_ranks:
                    builder = self.builders[rank]
                    builder.set_context(request_id, stage, generation)
                    # op_role=control_marker（契约 §7 裁定 8）：eviction
                    # done 标记是 comp(1,1) 微节点。
                    builder.comp(
                        "eviction_done_rank{}".format(rank), 1, 1,
                        op_role="control_marker")
                    members[rank] = builder.previous_id
            return members
        finally:
            for rank, builder in self.builders.items():
                builder.restore_chain(checkpoints[rank])
                if stashed[rank]:
                    builder.pending_extra_dependencies.extend(stashed[rank])

    def _register_store_tails(self, transfer, record) -> None:
        """逐出支链的边缘 mem_store 尾部登记（主方案 §3.3 登记侧；§4.2.4
        层区间化 2026-09-17）。

        仅 remote_store 逐出登记；record 为 _emit_kv_transfer 的返回值
        （2026-09-13 起携带 edge_store_node_id / source_ack_recv_node_id）。
        条目 = (edge_rank, edge_store_node_id, source_ack_recv_node_id,
        layer_start, layer_end)——层区间取 transfer.layer_start/layer_end
        （逐出池写恒为后缀形 [k, L)）；消费侧按区间交集选择性消费
        （_arm_pending_store_edges）。
        """
        if transfer.kind != "remote_store":
            return
        for shard_record in record["shards"]:
            tails = self.pending_store_tails.setdefault(
                transfer.session_id, [])
            tails.append((
                shard_record["edge_rank"],
                shard_record["edge_store_node_id"],
                shard_record["source_ack_recv_node_id"],
                transfer.layer_start,
                transfer.layer_end,
            ))

    def _match_store_entries(self, session_id, layer_start, layer_end):
        """P2-R2 助手 1/3（PARTIAL 跨实例 copy 流水化 2026-09-25）：纯
        查询——按 §4.2.4 层区间交集判据（max(ls1,ls2) < min(le1,le2)）
        返回 (matched, retained)；不发射节点、不改账本。无匹配 raise
        （fail-closed 文案与 _arm_pending_store_edges 历史一致：本会话
        此前无池写登记却要回迁读池，账目不一致）。"""
        entries = self.pending_store_tails.get(session_id) or []
        matched = []
        retained = []
        for entry in entries:
            entry_layer_start, entry_layer_end = entry[3], entry[4]
            if (max(entry_layer_start, layer_start)
                    < min(entry_layer_end, layer_end)):
                matched.append(entry)
            else:
                retained.append(entry)
        if not matched:
            registered = (
                ", ".join(
                    "[{}, {})".format(entry[3], entry[4])
                    for entry in entries)
                or "none")
            raise RuntimeError(
                "store->restore 前递保障失效: session {!r} restore 区间 "
                "[{}, {}) 无交集池写登记条目（现有条目区间: {}）".format(
                    session_id,
                    layer_start, layer_end,
                    registered))
        return matched, retained

    def _emit_store_relays_once(self, matched, restore_edges, name_prefix):
        """P2-R2 助手 2/3：跨缘 1B p2p 中继（桥拒绝跨 rank 直边，1B
        p2p 是协议内唯一合法载体），每个 (store_edge, restore_edge) 对
        **恰发射一次**——send 侧 arm 该对全部 store node ids、restore
        侧收 1B（recv 节点 = 恢复链的链序前递锚）。

        返回 {restore_edge: relay_recv_node_id}（本调用实际发射的对）。
        发射次序契约：先中继、后同缘 arm（此时 restore 边缘 rank 尚无
        armed 依赖，中继 recv 不吞并同缘 arm）。"""
        relay_arms = {}
        for edge_rank, store_node_id, _ack_node_id, _ls, _le in matched:
            for restore_edge in restore_edges:
                if edge_rank == restore_edge:
                    continue
                relay_arms.setdefault(
                    (edge_rank, restore_edge),
                    []).append(store_node_id)
        relay_recv_by_edge = {}
        for (store_edge, restore_edge), store_ids in relay_arms.items():
            tag = self.tag_allocator.take()
            store_builder = self.builders[store_edge]
            for node_id in store_ids:
                store_builder.arm_dependency(node_id)
            store_builder.comm_send(
                f"{name_prefix}_store_sidelink_s{store_edge}"
                f"_r{restore_edge}",
                src=store_edge,
                dst=restore_edge,
                comm_size=1,
                comm_tag=tag,
            )
            self.builders[restore_edge].comm_recv(
                f"{name_prefix}_store_sidelink_s{store_edge}"
                f"_r{restore_edge}",
                src=store_edge,
                dst=restore_edge,
                comm_size=1,
                comm_tag=tag,
            )
            relay_recv_node = self.builders[restore_edge].previous_id
            relay_recv_by_edge[restore_edge] = relay_recv_node
        return relay_recv_by_edge

    def _consume_store_entries(self, session_id, matched) -> None:
        """P2-R2 助手 3/3：恰移除 matched 条目（对象身份相等），其余
        保留在 pending_store_tails 供后续消费者；清空即弹会话键。消费
        窗口契约：恢复组支链在**同批发射内**调用（store 发射→restore
        发射一轮内），pending_store_tails 的在案窗口不放大——sh30
        merge_tail_gated 判据（本实例边缘 rank 有在案尾部 ⇒ 列车采样
        排除）的输入面与旧 consume 口径同阶。"""
        if not matched:
            return
        matched_ids = {id(entry) for entry in matched}
        remaining = [
            entry
            for entry in (self.pending_store_tails.get(session_id) or [])
            if id(entry) not in matched_ids]
        if remaining:
            self.pending_store_tails[session_id] = remaining
        else:
            self.pending_store_tails.pop(session_id, None)

    def _restore_group_involved_ranks(self, group_transfer, prefill_group):
        """P2-R2：单组恢复腿涉及的 rank 集（组区域 checkpoint/restore
        的范围）= Prefill ranks ∪ shard edge ranks ∪ shard target
        ranks——组间并行要求每组对全部涉及 rank 独立 checkpoint。"""
        ranks = set(prefill_group.ranks)
        for shard in group_transfer.shards:
            if shard.edge_rank is not None:
                ranks.add(shard.edge_rank)
            if shard.target_rank is not None:
                ranks.add(shard.target_rank)
        return sorted(ranks)

    def _arm_pending_store_edges(self, transfer, name_prefix: str) -> None:
        """回迁发射前的 store→restore 前递依赖补偿（主方案 §3.3；§4.2.4
        硬化 2026-09-17；2026-09-25 起机制实现于三个助手
        _match_store_entries/_emit_store_relays_once/_consume_store_
        entries——公开签名与行为对本方法消费者逐字节不变）。

        逐出支链化后"同会话 store 池写先于其 restore 池读"的主链传递性
        保障失效；本方法在回迁（remote_load）发射前查 pending_store_tails
        补边，粒度取 store 的边缘 mem_store 完成（池写落盘）：

        - 同边缘 rank：直接 arm（同 rank 显式 data_dep 跨批次合法，桥经
          持久 (rank,id) 解析）；
        - 跨边缘 rank：1B p2p 中继（助手 2；restore 边缘收 1B，回迁链
          链其后）。R1' 去钉扎（2026-09-14）后 REMOTE 全量回迁与
          **PARTIAL 跨实例 copy 的后缀池恢复**均可达跨缘（后缀 store 在
          home 边缘、restore 在执行实例边缘；1B 中继路径两用）。

        §4.2.4 层区间交集选择性消费：restore 传输区间
        [transfer.layer_start, transfer.layer_end)；匹配条目 = 区间有
        交集（max(ls1,ls2) < min(le1,le2)）的全部条目；只移除匹配条目，
        未匹配保留在 pending_store_tails 供后续消费者（新增消费者 =
        PARTIAL remote-read 后缀恢复 [p, L)）。行为保持注记：现有全部
        逐出池写层区间恒为后缀形 [k,L)，现有消费者区间（REMOTE 全量
        [0,L) / copy 后缀 [p,L)）与之恒有交集 ⇒ 交集选择性消费与旧
        pop-all 在现行模式下行为等价，硬化属防御性改造，零时间线扰动。

        fail-closed（原 fail-open 静默返回已退役）：会话在
        pending_store_tails 无任何条目与 restore 区间交集（"有条目但
        都不交集"与"无条目"两态同罪——本会话此前无池写登记却要回迁
        读池，账目不一致）→ raise（store→restore 前递保障失效）。

        store 早已物理完成时补边即刻满足（懒处理，无需判在飞）。消费即
        清（匹配条目）：该回迁的池读已排序于全部匹配 store 之后；未匹配
        条目的层区间池数据不受本次读影响。发射次序：先跨缘中继（此时
        restore 边缘 rank 尚无 armed 依赖，中继 recv 不吞并同缘 arm），
        后同缘 arm（由回迁链在该 rank 的首节点消费）。"""
        matched, _retained = self._match_store_entries(
            transfer.session_id, transfer.layer_start, transfer.layer_end)
        self._consume_store_entries(transfer.session_id, matched)
        restore_edges = []
        for shard in transfer.shards:
            if (shard.edge_rank is not None
                    and shard.edge_rank not in restore_edges):
                restore_edges.append(shard.edge_rank)
        self._emit_store_relays_once(matched, restore_edges, name_prefix)
        for edge_rank, store_node_id, _ack_node_id, _ls, _le in matched:
            if edge_rank in restore_edges:
                self.builders[edge_rank].arm_dependency(store_node_id)

    def _rebuild_interval_gate_on_target(
        self, pending_gate, request_plan: dict,
    ):
        """R5：把跨实例的 interval gate 结构性重建到本轮 prefill 实例。

        每相对位 rank 对：源 rank arm 原 gate 后发 1B p2p（源 gate 触发
        后启动）→ 目标 rank 收 1B → 目标 rank 发射重建 timer 节点
        （runtime_ns=0，仅依赖该 recv）。返回源实例 = prefill 实例的新
        PendingHistoryGate。R16-5（2026-09-15）按代码重枚举四个真实消费
        点（交叉断言过：history_evictions 触发门（发射器
        remote_store 分支消费 trigger）/ no-transfer arm /
        partial 恢复 arm / 通用 remote_load arm（→发射器 remote_load
        分支 arm+1B 中继））——旧 docstring
        的"noc 源端触发"是死参数分支（发射器 noc_migrate 分支从不消费
        trigger_gate，实参已删）、"local_hit arm"是幻影（通用循环对
        local_hit 先 continue，发射器 local_hit+gate 路径自本构建器
        不可达；pd_transfer 不传 gate）。孤儿 timer 节点
        O(跨实例轮次 × TP)、runtime_ns=0，可忽略。"""
        source_group = self.group_by_index[
            pending_gate.source_instance_index]
        prefill_group_ranks = self.group_by_index[
            request_plan["prefill_instance_index"]].ranks
        if len(source_group.ranks) != len(prefill_group_ranks):
            raise RuntimeError(
                "cross-instance gate rebuild requires equal TP sizes")
        prefix = _prefix_of(request_plan)
        following_request = self.config.request_queue[
            request_plan["queue_index"]]
        interval = following_request.inter_request_interval_ns
        if interval is None:
            raise RuntimeError("later request lost its inter-request interval")
        # K5（kimi 复审，2026-09-14）：interval + hbm_wait_ns 为任意 ns
        # 粒度，而 timer_gate 的离线同构校验要求整 µs（% 1000 != 0 即
        # raise）——跨实例轮换 + 非零准入等待（恰为 R5/R2 目标工况）会
        # 确定性崩溃。对齐 turn-0 先例（本文件 arrival gate）做 µs 下
        # 取整：duration 不进节点（runtime_ns=0、不存储），仅驱动校验
        # 与 0 跳过，下取整对既有通过路径零影响。
        duration = interval + request_plan.get("hbm_wait_ns", 0)
        duration -= duration % 1000
        rebuilt_gates = []
        for relative_index, (source_rank, target_rank) in enumerate(
                zip(source_group.ranks, prefill_group_ranks)):
            source_gate = pending_gate.timer_gates[relative_index]
            if source_gate is None:
                raise RuntimeError(
                    "cross-instance gate rebuild found a missing source "
                    "gate node (rank {})".format(source_rank))
            relay_tag = self.tag_allocator.take()
            self.builders[source_rank].set_context(
                request_plan["request_id"], "prefill", 0)
            self.builders[source_rank].arm_dependency(source_gate)
            self.builders[source_rank].comm_send(
                f"{prefix}_gate_relay_s{source_rank}_r{target_rank}",
                src=source_rank,
                dst=target_rank,
                comm_size=1,
                comm_tag=relay_tag,
            )
            self.builders[target_rank].set_context(
                request_plan["request_id"], "prefill", 0)
            self.builders[target_rank].comm_recv(
                f"{prefix}_gate_relay_s{source_rank}_r{target_rank}",
                src=source_rank,
                dst=target_rank,
                comm_size=1,
                comm_tag=relay_tag,
            )
            recv_node_id = self.builders[target_rank].previous_id
            rebuilt_gates.append(self.builders[target_rank].timer_gate(
                f"{prefix}_interval_gate_rebuilt_rank{target_rank}",
                duration,
                after_node_id=recv_node_id,
            ))
        return PendingHistoryGate(
            source_instance_index=request_plan["prefill_instance_index"],
            timer_gates=tuple(rebuilt_gates),
            location=pending_gate.location,
        )

    def emit_eviction_side_branch(
            self, transfers, tick: int, *, watch_id: str | None = None):
        """R2/D7（2026-09-14）：准入事务失败路径上**已提交**逐出的图侧
        发射（旁路支链、无触发门——与 prefill_evictions 的发射形态同构；
        失败请求不入队，逐出是合法容量释放，其池写必须进图，否则 C++
        水位盲区 + pending store 不登记）。"""
        transfers = tuple(transfers or ())
        if not transfers:
            return
        marker = self._mark()
        first = transfers[0]
        context_plan = {
            "request_id": first.trigger_request_id,
            "session_id": first.session_id,
        }
        self._set_context(context_plan, "prefill", 0)
        action_state = self._action_seq.setdefault(
            first.trigger_request_id, [0])
        if watch_id is None:
            watch_id = "batch_train_evict_{}_failed_q{:03d}".format(
                first.trigger_request_id, action_state[0])

        def emit_failed_admission_evictions() -> None:
            for transfer in transfers:
                action_name = (
                    f"evict_{sanitize_node_prefix(transfer.trigger_request_id)}"
                    f"_action{action_state[0]:03d}_"
                    f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
                )
                record = _emit_kv_transfer(
                    config=self.config,
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    transfer=transfer,
                    action_name=action_name,
                )
                record["sequence_stage"] = "failed_admission_evictions"
                record["action_sequence"] = action_state[0]
                action_state[0] += 1
                self._register_store_tails(transfer, record)

        members = self._emit_side_branch(
            emit_failed_admission_evictions,
            watch_context=(watch_id, "prefill", 0))
        self._collect(marker)
        if not members:
            return None
        return {
            "request_id": watch_id,
            "owner_request_id": first.trigger_request_id,
            "branch": "failed_admission",
            "members": members,
        }

    def _emit_admission_actions(self, request_plan: dict) -> list[dict]:
        """turn-gates / history / prefill 准入动作的在线发射
        （拼 batch 改造，2026-08-22）。
        request_plan 为 dict（strategy 自在线账本；
        字段与 FaceRequestPlan 同名）。

        [frontier 接续裁决,strategy 死锁修复统一(2026-08-19,对齐 sh_1.0/
        sh_2.0)] strategy **不做任何块末恢复/段内清链**:per-rank
        previous_id 无条件接续当前 frontier(= 离线 writer 跨 request
        物理链同构),per-rank 发行序 = 全局发射序——跨请求 P2P 与
        collective 参与序不可能反转成环。

        prefill 主体（chunk 序列 + end barrier）与 PREFILL_DRAIN watch 自
        拼 batch 改造起移入实例迭代列车（emit_iteration_train 的折叠体与
        drain 标记）；此处止于准入动作（到达 gates/历史迁移/逐出/屏障）。"""
        builders = self.builders
        config = self.config
        group_by_index = self.group_by_index
        prefill_group = group_by_index[request_plan["prefill_instance_index"]]
        request = config.request_queue[request_plan["queue_index"]]
        prefix = _prefix_of(request_plan)
        action_state = self._action_seq.setdefault(
            request_plan["request_id"], [0])
        eviction_watches = []

        def emit_transfer(transfer, stage: str, *, gate=None,
                          trigger_gate=None) -> dict:
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config,
                builders=builders,
                group_by_index=group_by_index,
                tag_allocator=self.tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
            )
            record["sequence_stage"] = stage
            record["action_sequence"] = action_sequence
            action_state[0] = action_sequence + 1
            # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
            # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
            return record

        # ---- arrival / interval gate（turn-0 到达 timer / turn>0 pending gate）----
        if request_plan["turn_index"] == 0:
            # 在线：turn-0 arrival timer gate 在本批次发射（runtime=0，
            # 到达时间由 C++ arrival alarm 替代；duration 语义与离线同参：
            # admission_time_ns，0 时离线也不发射 gate 节点）。
            arrival = request_plan.get("admission_time_ns")
            if arrival is None:
                arrival = request.session_arrival_time_ns
            if arrival is None:
                raise RuntimeError("first request lost its session arrival")
            # turn-0 gate duration 先做 µs 下取整：admission_time_ns 是准入
            # 时刻的虚拟 tick（Roofline 任意 ns 粒度，如 153516157242），
            # timer_gate 的离线同构校验要求整 µs（duration_ns % 1000 != 0
            # 即 raise）。该 duration 不进节点（runtime_ns=0、不存储），仅
            # 驱动校验与 0 跳过，下取整对既有通过路径零影响；消除
            # 2026-08-22 30s 窗 delivery seq=1145 确定性崩溃（阻塞 turn-0
            # 在非 µs 对齐 tick 准入时触发）。
            duration = arrival
            duration -= duration % 1000
            # turn-0 gate 命名用短前缀（q{queue:04d}_{request_id}，
            # 无 session/turn 段）——canonical 命名 name 键逐字节一致前提。
            short_prefix = (
                f"q{request_plan['queue_index']:04d}_"
                f"{sanitize_node_prefix(request_plan['request_id'])}")
            timers = tuple(
                builders[rank].timer_gate(
                    f"{short_prefix}_global_arrival_timer_gate", duration)
                for rank in prefill_group.ranks
            )
            pending_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=timers,
                location="new_session",
            )
        else:
            pending_gate = self.pending_history.pop(
                request_plan["request_id"], None)
            if pending_gate is None:
                raise RuntimeError(
                    f"request {request_plan['request_id']} has no "
                    "arrival/history gate")
            self.pending_request_by_session.pop(request_plan["session_id"],
                                                None)
            # R5（D3，2026-09-14）：跨实例轮换的 interval gate 重建。上轮
            # gate 节点在源执行实例的 rank id 空间；本轮 prefill 实例不同
            # 时直接 arm 会形成跨实例悬空依赖（id 数值在目标空间碰撞 =
            # 静默错连；不碰撞 = C++ 拒绝跨 rank 边，冒烟尺度起即不稳）。
            # 重建 = 目标实例 ranks 上的结构性 timer 节点（duration 沿用
            # 原值、runtime_ns=0——时序因果由 C++ arrival calendar 承载，
            # 重建只迁移结构依赖），逐相对位经 1B p2p 中继依赖源 gate。
            # 同实例路径逐字节不变；保留 source_ranks == prefill_ranks
            # 断言作 backstop。
            if (pending_gate.source_instance_index
                    != request_plan["prefill_instance_index"]):
                pending_gate = self._rebuild_interval_gate_on_target(
                    pending_gate, request_plan)
            else:
                source_group = self.group_by_index[
                    pending_gate.source_instance_index]
                if source_group.ranks != prefill_group.ranks:
                    raise RuntimeError(
                        "interval gate ranks do not match the Prefill "
                        "instance ranks (backstop assertion)")
        # sh_3.0 裁决（登记合同⑦/§13，两模式统一）：pending-gate location
        # 是 writer 侧的派生缓存（历史静态实现曾用全局 KV 因果预排序
        # 保持与 planner 一致）；当前在线路径以 kv_manager 权威账本/图依赖为准，
        # 触发 request 的准入/完成逐出（reserve/prepare/expand/enforce）与
        # gate 注册/消费的相对顺序不同，缓存会滞后于权威账本（strategy 权威
        # = kv_manager 快照,即 history_location_before 本身）。故弹出 gate 时统一归一化到
        # history_location_before；物理时序由图依赖保证（store 节点挂触发
        # request 链），gate location 仅是发射期元数据。结构性错误（缺
        # gate/缺 location）仍 fail-closed。
        if request_plan["history_location_before"] is not None:
            pending_gate.location = (
                request_plan["history_location_before"].location)
        else:
            reconcile_pending_history_location(
                pending_gate, _as_plan(request_plan))

        # ---- history_evictions（KV 逐出并行化 2026-09-13：旁路支链化；
        #      到达/间隔 timer gate 触发门在分支内构建并挂分支首节点——
        #      逐出开始时刻不变，物理完成不再阻塞其后主链的一切计算）----
        if request_plan["history_evictions"]:
            watch_id = request_plan.get(
                "history_eviction_watch_id",
                "batch_train_evict_{}_admission_history".format(
                    request_plan["request_id"]))
            def emit_history_evictions() -> None:
                history_eviction_trigger = TransferTriggerGate(
                    control_instance_index=pending_gate.source_instance_index,
                    node_gates=pending_gate.timer_gates,
                )
                for transfer in request_plan["history_evictions"]:
                    record = emit_transfer(
                        transfer, "history_evictions",
                        trigger_gate=history_eviction_trigger)
                    self._register_store_tails(transfer, record)
            members = self._emit_side_branch(
                emit_history_evictions,
                watch_context=(watch_id, "prefill", 0))
            if members:
                eviction_watches.append({
                    "request_id": watch_id,
                    "owner_request_id": request_plan["request_id"],
                    "branch": "admission_history",
                    "members": members})

        partial_history_restore = (
            request_plan["history_location_before"] is not None
            and request_plan["history_location_before"].location
            == "partial_hbm_remote"
        )
        suffix_ready_nodes_by_rank = {}

        # ---- history_transfers（joint：stay 单笔 / copy 前缀 NoC+后缀池
        #      恢复两笔 / recompute 与 remote-read 零搬运）----
        joint_action = request_plan.get("joint_action", "stay")
        if "history_transfers" in request_plan:
            history_transfers = tuple(
                transfer
                for transfer in (request_plan.get("history_transfers") or ())
                if transfer is not None
            )
        else:
            # 旧 plan 形态（单数字段；fixtures/verify 路径）回退。
            legacy_transfer = _as_plan(request_plan).history_transfer
            history_transfers = (
                (legacy_transfer,) if legacy_transfer is not None else ())
        single_history_transfer = (
            history_transfers[0]
            if len(history_transfers) == 1 else None)
        # C13 copy 逐 chunk 交接：交接块 1..M-1（handoff_chunk ≥ 1）旁挂
        # 支链流水发射（fork 点 = 头块发射后的主链 frontier），块 0 与
        # 其余腿（池恢复后缀等）走下方既有主链循环——readiness barrier
        # 只等主链节点，即只等交接块 0（计算不等整份搬运完成，不设
        # "先整份搬运后计算"串行段；尾块由列车体逐块就绪门控消费）。
        copy_handoff_tail = tuple(
            transfer for transfer in history_transfers
            if (getattr(transfer, "handoff_chunk", None) or 0) >= 1)
        # C15 后缀逐组恢复腿（restore_group 标记；消费顺序 = 层自低向
        # 高）：从主链循环摘出——readiness barrier 不再等整段后缀；逐组
        # 旁挂/分支发射（rank 内串行链）+ 逐组就绪门登记（列车体按层段
        # 门控消费，_suffix_restore_arms）。旧单笔口径（无标记）不变。
        restore_group_transfers = tuple(sorted(
            (
                transfer for transfer in history_transfers
                if transfer.kind == "remote_load"
                and getattr(transfer, "restore_group", None) is not None
            ),
            key=lambda transfer: transfer.restore_group))

        def _is_restore_group_tail(transfer) -> bool:
            return (
                transfer.kind == "remote_load"
                and getattr(transfer, "restore_group", None) is not None)

        # 规格书§三.1（2026-09-25 prefill remote-read 前缀读流；D09 逐
        # tile 化 2026-09-27）：独立于 history_transfers 的前缀读流计划
        # （调度器 plan_dict 新字段）——home 驻留前缀 [0, p) 的逐窗×tile
        # noc_migrate 瞬时读流（stream_only，不物化入容量账本）。准入批
        # 只消费触发链；数据 tile 随消费窗口在列车头/体发射。PARTIAL 基
        # 与后缀池恢复两腿并行、共享同一准入 frontier；LOCAL 基 = 全层
        # [0, L) 读流（history_transfers 恒空）。旧 plan 形态无该键 → None。
        prefill_read_plan = request_plan.get("prefill_remote_read_plan")

        if copy_handoff_tail or restore_group_transfers:
            main_history_transfers = tuple(
                transfer for transfer in history_transfers
                if (getattr(transfer, "handoff_chunk", None) or 0) < 1
                and not _is_restore_group_tail(transfer))
        else:
            main_history_transfers = history_transfers
        # partial 流水恢复分支仅适用于 stay（同实例单笔后缀恢复）；
        # 跨实例 copy 的前缀+后缀走通用发射：C13 交接尾块（handoff_
        # chunk>=1）与 C15 后缀逐组恢复腿被上方主链选择摘出主链、旁挂
        # 流水发射，规格书§三 前缀读流组则独立旁挂分支——readiness
        # barrier 只等主链（即交接块 0/首块），计算不等整份搬运或整段
        # 后缀恢复。C15：逐组恢复腿不走旧单笔 partial 分支（分组发射
        # 替换整段门；单组亦然——restore_group 有标记即走分组路径）。
        partial_history_restore = (
            partial_history_restore
            and single_history_transfer is not None
            and getattr(single_history_transfer, "restore_group", None)
            is None
            and request_plan["history_location_before"].instance_index
            == request_plan["prefill_instance_index"])

        if not history_transfers:
            if joint_action == "remote-read":
                # remote-read（LOCAL 基——history_transfers 恒空）：本批
                # 无历史搬运主链节点（旧口径此处 pass = interval/arrival
                # gate 从未被消费 + 前缀层计算无数据门，2026-09-25 规格
                # 书§三.2 废除）。前缀读流由下方独立旁挂分支发射：gate
                # 的触发消费、home→exec 层组读流均在分支内完成（前缀
                # 计算的层段门由列车体消费 recv 完成门承担）；gate 结构
                # 性重建（跨实例轮换 → 本轮 prefill 实例）已在 gate 段
                # 经 R5 _rebuild_interval_gate_on_target 完成。decode 相
                # 的 home 前缀 credit 读流仍经 joiner 动作发射（drain
                # 边界），与本批无交集。
                if prefill_read_plan is None:
                    raise RuntimeError(
                        "remote-read admission without its prefill prefix "
                        "read plan (plan_prefill_remote_read_transfers "
                        f"must supply it; request "
                        f"{request_plan['request_id']!r})")
            elif joint_action == "recompute":
                # 重算历史在执行期物化——本批无历史搬运节点。D10
                # （v2.0 §6.2，2026-09-26）：物化 = 完整 H+U 重放（全形
                # 态 span 基 0、重建基数 = 整份已保存上下文，旧缺失层折
                # 算取消）；逐 chunk 计算体由列车 pass_spans 冻结（调度
                # 器侧 span 基改写），plan_dict 阶段字段
                # joint_rebuild_context_tokens / joint_prefill_stage
                # （= prefill_replay）/ joint_suppress_sampling /
                # joint_suppress_output 随 plan 透传（工作描述三方同
                # 源：runtime、决策日志、本构图器消费的 plan_dict）。
                pass
            elif request_plan["turn_index"] != 0:
                raise RuntimeError(
                    "later request is missing its history transfer action")
            else:
                source_group = group_by_index[
                    pending_gate.source_instance_index]
                if source_group.ranks != prefill_group.ranks:
                    raise RuntimeError(
                        "first-request arrival gate is not on its Prefill ranks")
                for relative_index, rank in enumerate(prefill_group.ranks):
                    builders[rank].arm_timer_gate(
                        pending_gate.timer_gates[relative_index])
        elif not partial_history_restore:
            # gate location 已在弹出时归一化到权威快照（见上方裁决注释）。
            for history_transfer in main_history_transfers:
                if history_transfer.kind == "local_hit":
                    # 零节点本地命中（stay 驻留复用）——无发射节点。
                    continue
                if history_transfer.kind == "remote_load":
                    # KV 逐出并行化（2026-09-13）：REMOTE 全量回迁入口的
                    # store→restore 前递补边（跨实例回迁可达跨缘，1B 中继
                    # 仅此路径需要）。
                    self._arm_pending_store_edges(
                        history_transfer, f"{prefix}_history_transfer")
                if (
                    history_transfer.kind == "noc_migrate"
                    and history_transfer.source_instance_index is not None
                    and pending_gate.source_instance_index
                    != history_transfer.source_instance_index
                ):
                    # joint（§2.2）：上一轮在异地执行时，到达/间隔 gate 在
                    # 上一执行实例的 rank 上，而 copy 的源端在 home。在线
                    # 模式下因果由"图批于准入时刻喂入"承载（NEW-1：发射器
                    # noc_migrate 分支从不消费 trigger_gate，R16-5 已删原
                    # 死参数实参——本分支为无门控发射，链式顺序兜底；若
                    # 未来接通离线/回放模式，接通 noc_migrate 的 trigger
                    # 消费是前置条件，见 PROVENANCE 登记）。
                    emit_transfer(history_transfer, "history_transfer")
                else:
                    emit_transfer(history_transfer,
                                  "history_transfer", gate=pending_gate)
            if copy_handoff_tail:
                # C13：交接尾块支链（fork 顺序 = 头块主链发射之后；分支
                # 不 join——readiness barrier 不等尾块，列车体逐块 arm 消
                # 费）。home 侧 send 链全速泵出，exec 侧 recv 链流式到达
                # （与 _emit_credit_stream_tail 同款拓扑）；ack 链尾随整
                # 条数据链（source release dependency 挂点）。
                def emit_copy_handoff_branch(
                        tail=copy_handoff_tail, gate=pending_gate):
                    self._emit_copy_handoff_tail(
                        request_plan, tail, pending_gate=gate)
                self._emit_side_branch(emit_copy_handoff_branch)

        # ---- prefill remote-read 前缀读流旁挂分支（规格书§三.1-§三.3，
        #      2026-09-25；D09 逐 tile 化 2026-09-27）：准入批只发射读流
        #      触发链（恰一次消费 interval/arrival gate，1B relay）；数据
        #      tile 随消费窗口（逐 chunk 迭代体块）在各 prefill 列车头/
        #      体发射——消费驱动逐 tile 发射 + 双缓冲释放链（tile i+2 生
        #      产等 tile i 消费完成），不整轮预建事件（v2.0 §17.1）。
        if prefill_read_plan is not None:
            if joint_action != "remote-read":
                raise RuntimeError(
                    "prefill remote-read prefix stream requires "
                    f"joint_action 'remote-read' (got {joint_action!r}; "
                    f"request {request_plan['request_id']!r})")
            self._emit_prefill_remote_read_branch(
                request_plan, prefill_read_plan, pending_gate)
        elif joint_action == "remote-read":
            # PLAN 形态破损 fail-closed：remote-read 必有前缀读流（LOCAL
            # 基全层 / PARTIAL 基前缀 [0, p)；plan_prefill_remote_read_
            # transfers 的适用性合同保证非空）。
            raise RuntimeError(
                "remote-read admission without its prefill prefix read "
                "plan (plan_prefill_remote_read_transfers must supply "
                f"it; request {request_plan['request_id']!r})")

        # ---- prefill_evictions（KV 逐出并行化 2026-09-13：旁路支链化，
        #      无触发门；fork 时可能存在主链 armed 依赖（turn-0 arrival
        #      gate 的 arm_timer_gate——R16-5 订正：local_hit 在通用循环
        #      先 continue、不 arm，原"local_hit 的 arm"表述失实），由
        #      _emit_side_branch 的 stash-and-clear 保真留给主链屏障消费）
        #      ----
        # R17-1a'（2026-09-17 死通道钉死）：joint 下本发射块条件恒假
        # ——prefill_evictions 结构性恒空（准入 R1' 预约覆盖全动作足迹、
        # drain expand gap≡0；2026-09-16 三方裁决），读但永不触发（净
        # 效果=死块）。保留不删（R16 §8 红线：删死只删实参/保守），本
        # 注释消除下一个"伪消费者"式误读；非 joint 调用面若未来启用该
        # 字段，此处语义照旧。
        if request_plan["prefill_evictions"]:
            watch_id = request_plan.get(
                "prefill_eviction_watch_id",
                "batch_train_evict_{}_admission_prefill".format(
                    request_plan["request_id"]))
            def emit_prefill_evictions() -> None:
                for transfer in request_plan["prefill_evictions"]:
                    record = emit_transfer(transfer, "prefill_evictions")
                    self._register_store_tails(transfer, record)
            members = self._emit_side_branch(
                emit_prefill_evictions,
                watch_context=(watch_id, "prefill", 0))
            if members:
                eviction_watches.append({
                    "request_id": watch_id,
                    "owner_request_id": request_plan["request_id"],
                    "branch": "admission_prefill",
                    "members": members})

        # ---- readiness barrier / partial 流水恢复 / C15 逐组恢复 ----
        restore_groups_partial = bool(
            restore_group_transfers
            and request_plan["history_location_before"] is not None
            and request_plan["history_location_before"].location
            == "partial_hbm_remote"
            and request_plan["history_location_before"].instance_index
            == request_plan["prefill_instance_index"])
        if restore_groups_partial:
            # C15（设计方案 §5.1）：后缀逐组恢复——同实例 partial 快速
            # 路径的同款拓扑（驻留前缀屏障 + checkpoint 分支）。P2-R2
            # （PARTIAL 跨实例 copy 流水化 2026-09-25）：恢复组并行化
            # ——组间无边、互并发（旧的"同 rank 下一组首节点链在上一组
            # 之后 = 物理串行"整段串行依赖移除），逐组逐 rank 目标 HBM
            # 写完成门登记 _suffix_restore_arms，由含首 chunk 的列车体
            # 按层段门控消费——**不设**"整段后缀恢复完才开始 prefill"
            # 的串行门（无 suffix p2p readiness barrier——逐 rank 门控
            # 替换整段栅栏，C12-G7/E17 登记的列车级保守门闭合）。
            history_before = request_plan["history_location_before"]
            if pending_gate.location != history_before.location:
                raise RuntimeError(
                    f"history gate location {pending_gate.location!r} does "
                    f"not match planned {history_before.location!r}")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index])
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_resident_prefix_ready_barrier")
            checkpoints = {
                rank: builders[rank].chain_checkpoint()
                for rank in prefill_group.ranks
            }
            prefix_barrier_nodes = tuple(
                builders[rank].previous_id for rank in prefill_group.ranks)
            branch_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=prefix_barrier_nodes,
                location=history_before.location,
            )
            # store→restore 前递（P2-R2 形态，①-④）：
            #   ① fail-closed 面与现状逐字节一致：先对组 0 区间做无匹配
            #      raise（_match_store_entries 历史文案；后缀形条目下组 0
            #      有匹配 ⇒ 各组至少与该条目相交）；
            #   ② 组循环前对并集区间 [min_start, max_end) 发射跨缘中继，
            #      每跨缘对恰一次（中继 recv 落在各 restore 缘 rank 的分
            #      支链上——组区域 checkpoint 捕获其后 previous_id ⇒ 每组
            #      首节点经链序排在全部并集匹配 store 完成之后）；
            #   ③ 每组：checkpoint → 同缘 arm（**只 arm 与本组层区间交
            #      集**的条目——交集判据 max(ls1,ls2)<min(le1,le2)，后缀
            #      条目与 end≤k 的组无交集）→ emit_transfer → 组门登记
            #      → restore_chain；组间无边，组 i 首节点依赖 = 同缘交集
            #      store（直接 arm）∪ 本缘共享中继（链序），不弱于旧
            #      "组 i 链在组 i-1 后"的传递性保障；
            #   ④ 组循环后同批消费并集匹配条目（span 外条目保留）——消
            #      费窗口回到 store 发射→restore 发射一轮内，
            #      pending_store_tails 在案窗口不放大（sh30
            #      merge_tail_gated 判据输入不变）。
            _session_id = restore_group_transfers[0].session_id
            self._match_store_entries(
                _session_id,
                restore_group_transfers[0].layer_start,
                restore_group_transfers[0].layer_end)
            _union_start = min(
                group.layer_start for group in restore_group_transfers)
            _union_end = max(
                group.layer_end for group in restore_group_transfers)
            group_matched, _group_retained = self._match_store_entries(
                _session_id, _union_start, _union_end)
            _group_restore_edges = []
            for group in restore_group_transfers:
                for shard in group.shards:
                    if (shard.edge_rank is not None
                            and shard.edge_rank not in _group_restore_edges):
                        _group_restore_edges.append(shard.edge_rank)
            self._emit_store_relays_once(
                group_matched, _group_restore_edges,
                f"{prefix}_history_transfer")
            group_arms = []
            for group_transfer in restore_group_transfers:
                involved_ranks = self._restore_group_involved_ranks(
                    group_transfer, prefill_group)
                group_checkpoints = {
                    rank: builders[rank].chain_checkpoint()
                    for rank in involved_ranks
                }
                for entry_edge, entry_store, _ack, entry_ls, entry_le in (
                        group_matched):
                    if (entry_edge in _group_restore_edges
                            and max(entry_ls, group_transfer.layer_start)
                            < min(entry_le, group_transfer.layer_end)):
                        builders[entry_edge].arm_dependency(entry_store)
                group_record = emit_transfer(
                    group_transfer, "history_transfer", gate=branch_gate)
                group_arms.append(self._restore_group_arm(
                    group_transfer, group_record, prefill_group))
                for rank in involved_ranks:
                    builders[rank].restore_chain(group_checkpoints[rank])
            self._consume_store_entries(_session_id, group_matched)
            for rank in prefill_group.ranks:
                builders[rank].restore_chain(checkpoints[rank])
            # 链回滚保留（恢复分支与本 rank 后续发射并行，与旧口径一致）。
            self._suffix_restore_arms[request_plan["request_id"]] = (
                tuple(group_arms))
        elif partial_history_restore:
            history_transfer = single_history_transfer
            history_before = request_plan["history_location_before"]
            if (history_transfer.kind != "remote_load"
                    or history_transfer.layer_start
                    != history_before.resident_prefix_layers
                    or history_transfer.layer_end != config.layers):
                raise RuntimeError(
                    "partial history load does not match its suffix")
            if (request_plan["prefill_instance_index"]
                    != history_before.instance_index):
                raise RuntimeError(
                    "partial history lost Prefill instance affinity")
            if pending_gate.location != history_before.location:
                raise RuntimeError(
                    f"history gate location {pending_gate.location!r} does "
                    f"not match planned {history_before.location!r}")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index])
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_resident_prefix_ready_barrier")
            checkpoints = {
                rank: builders[rank].chain_checkpoint()
                for rank in prefill_group.ranks
            }
            prefix_barrier_nodes = tuple(
                builders[rank].previous_id for rank in prefill_group.ranks)
            branch_gate = PendingHistoryGate(
                source_instance_index=request_plan["prefill_instance_index"],
                timer_gates=prefix_barrier_nodes,
                location=history_before.location,
            )
            # KV 逐出并行化（2026-09-13）：PARTIAL 后缀恢复支链入口的
            # store→restore 前递补边——arm 必须在 checkpoint 之后、回迁
            # 发射之前（分支区域内部），由回迁链在该 rank 的首节点消费
            # （PARTIAL 钉扎同实例，实测恒同缘直接 arm）。
            self._arm_pending_store_edges(
                history_transfer, f"{prefix}_history_transfer")
            history_record = emit_transfer(history_transfer,
                                           "history_transfer", gate=branch_gate)
            suffix_restore_nodes_by_rank = {}
            for shard_record in history_record["shards"]:
                target_rank = shard_record.get("target_rank")
                completion_node = shard_record.get(
                    "target_hbm_completion_node_id")
                if not isinstance(target_rank, int) or not isinstance(
                        completion_node, int):
                    raise RuntimeError(
                        "suffix restore is missing a target HBM gate")
                suffix_restore_nodes_by_rank[target_rank] = completion_node
            if set(suffix_restore_nodes_by_rank) != set(prefill_group.ranks):
                raise RuntimeError(
                    "suffix restore did not cover every Prefill rank")
            suffix_readiness = _emit_tp_point_to_point_readiness_barrier(
                builders=builders, group=prefill_group,
                tag_allocator=self.tag_allocator,
                name=f"{prefix}_prefill_suffix_ready_barrier")
            suffix_ready_nodes_by_rank = {
                int(rank): int(node_id)
                for rank, node_id in suffix_readiness["node_ids_by_rank"]
            }
            for rank in prefill_group.ranks:
                builders[rank].restore_chain(checkpoints[rank])
            # 拼 batch 改造（2026-08-22）：首 chunk 主体不再在本批发射，
            # suffix 恢复完成门从"本批首 chunk 前缀层并行计算"改为经
            # _suffix_body_arms 账本挂到含其首 chunk 的列车体首节点
            # （emit_iteration_train 的 first_chunk_member）。链回滚保留
            # ——恢复分支与本 rank 后续发射（其他请求的准入动作）保持
            # 并行，与改造前的跨请求并行口径一致；列车体本身等恢复
            # （保守方向，类 docstring 登记）。
            self._suffix_body_arms[request_plan["request_id"]] = (
                suffix_ready_nodes_by_rank)
        else:
            if restore_group_transfers:
                # C15：跨实例后缀逐组恢复（copy/remote-read 工作副本的
                # 池恢复后缀）——主链迁移腿发射后旁挂支链逐组流水发射
                # （fork 点 = 主链迁移后的 frontier；分支不 join——
                # readiness barrier 只等主链，计算不等整段后缀恢复）。
                # P2-R2（2026-09-25）：组间并行化（组间无边、互并发）+
                # 每组交集直接 arm + 每跨缘对恰一次共享中继 + 并集条目
                # 同批消费（①-④ 同 stay 实例分支注释；消费窗口有界，
                # merge_tail_gated 判据输入不放大）。
                def emit_restore_group_branch(
                        groups=restore_group_transfers, gate=pending_gate):
                    session_id = groups[0].session_id
                    self._match_store_entries(
                        session_id, groups[0].layer_start,
                        groups[0].layer_end)
                    union_start = min(g.layer_start for g in groups)
                    union_end = max(g.layer_end for g in groups)
                    matched, _retained = self._match_store_entries(
                        session_id, union_start, union_end)
                    restore_edges = []
                    for group in groups:
                        for shard in group.shards:
                            if (shard.edge_rank is not None
                                    and shard.edge_rank
                                    not in restore_edges):
                                restore_edges.append(shard.edge_rank)
                    self._emit_store_relays_once(
                        matched, restore_edges,
                        f"{prefix}_history_transfer")
                    group_arms = []
                    for group_transfer in groups:
                        involved_ranks = self._restore_group_involved_ranks(
                            group_transfer, prefill_group)
                        group_checkpoints = {
                            rank: self.builders[rank].chain_checkpoint()
                            for rank in involved_ranks
                        }
                        for (entry_edge, entry_store, _ack, entry_ls,
                             entry_le) in matched:
                            if (entry_edge in restore_edges
                                    and max(entry_ls,
                                            group_transfer.layer_start)
                                    < min(entry_le,
                                          group_transfer.layer_end)):
                                self.builders[entry_edge].arm_dependency(
                                    entry_store)
                        group_record = emit_transfer(
                            group_transfer, "history_transfer", gate=gate)
                        group_arms.append(self._restore_group_arm(
                            group_transfer, group_record, prefill_group))
                        for rank in involved_ranks:
                            self.builders[rank].restore_chain(
                                group_checkpoints[rank])
                    self._consume_store_entries(session_id, matched)
                    self._suffix_restore_arms[
                        request_plan["request_id"]] = tuple(group_arms)
                self._emit_side_branch(emit_restore_group_branch)
            _emit_tp_readiness_barrier(
                builders=builders, group=prefill_group,
                name=f"{prefix}_prefill_kv_ready_barrier")
        # prefill 主体（chunk spans + end barrier + seg1 块末）自拼 batch
        # 改造（2026-08-22）起移入 emit_iteration_train 的折叠体与 drain
        # 标记；此处止于准入动作（到达 gates/历史迁移/逐出/屏障）。
        return eviction_watches

    def _restore_group_arm(self, group_transfer, group_record, prefill_group):
        """C15：单组恢复腿的逐 rank 就绪门（目标 HBM 写完成节点）。

        逐 rank 门控（非 TP 栅栏）：rank r 的列车体层段只等 rank r 的
        该组写完成——与预测器的逐 rank 恢复链口径一致；零字节 rank 无
        shard 即无门（无数据即无到达可等）。组序/层区间取 transfer 值。
        """
        gates_by_rank = {}
        for shard_record in group_record["shards"]:
            target_rank = shard_record.get("target_rank")
            completion_node = shard_record.get("target_hbm_completion_node_id")
            if not isinstance(target_rank, int) or not isinstance(
                    completion_node, int):
                raise RuntimeError(
                    "restore group is missing a target HBM gate")
            gates_by_rank[target_rank] = completion_node
        if not set(gates_by_rank) <= set(prefill_group.ranks):
            raise RuntimeError(
                "restore group gates leaked outside the Prefill instance")
        return (
            int(getattr(group_transfer, "restore_group")),
            int(group_transfer.layer_start),
            int(group_transfer.layer_end),
            gates_by_rank,
        )

    def _emit_prefill_remote_read_branch(
            self, request_plan: dict, read_plan, pending_gate) -> None:
        """prefill remote-read 前缀读流旁挂分支（规格书§三.1-§三.3，
        2026-09-25；D09 逐 tile 化 2026-09-27）：准入批发射**读流触发链**
        （恰一次消费 interval/arrival gate），数据 tile 不在本分支——
        逐 tile send/recv 与双缓冲释放链随**消费窗口**（逐 chunk 迭代）
        在各 prefill 列车头/体发射（消费驱动，不整轮预建；窗口 tile 复
        用 decode credit 的 _credit_arms/_credit_pending_tiles 账本）。

        合同复检（fail-closed；D16 有界模板修复轮二）：读计划快照携带
        O(窗数) 模板参数（不物化整轮 tile 列表）——本分支核验模板级合
        同：窗/tile 计数非退化、home/exec 端点与 history location 一致、
        会话归属、前缀层界 = 驻留前缀、TP 相对位保持；逐 tile 结构不变
        量（noc_migrate/stream_only/read_tile 身份连续唯一）由物化器
        ``prefill_read_window_tiles`` 钉测承担（构造同源，见
        face_scheduler）。触发门实例 = 本轮 prefill（exec）实例。"""
        request_id = request_plan["request_id"]
        exec_instance = request_plan["prefill_instance_index"]
        window_count = int(read_plan.get("window_count") or 0)
        tiles_by_window = tuple(read_plan.get("tiles_by_window") or ())
        if (window_count <= 0
                or len(tiles_by_window) != window_count
                or any(int(count) <= 0 for count in tiles_by_window)):
            raise RuntimeError(
                "prefill remote-read admission requires a non-degenerate "
                "window template (request {!r}; windows={} "
                "tiles_by_window={!r})".format(
                    request_id, window_count, tiles_by_window))
        home_instance = int(read_plan["home_instance"])
        plan_session_id = read_plan.get("session_id")
        if plan_session_id != request_plan["session_id"]:
            raise RuntimeError(
                "prefill remote-read plan belongs to another session "
                f"({plan_session_id!r}; request {request_id!r})")
        history_before = request_plan.get("history_location_before")
        if history_before is None:
            raise RuntimeError(
                "prefill remote-read requires a history location "
                f"snapshot (request {request_id!r})")
        if history_before.instance_index != home_instance:
            raise RuntimeError(
                "prefill remote-read prefix stream home instance "
                f"{home_instance} does not match the history location "
                f"instance {history_before.instance_index} (request "
                f"{request_id!r})")
        if history_before.location == "local_hbm":
            expected_prefix_end = self.config.layers
        elif history_before.location == "partial_hbm_remote":
            expected_prefix_end = history_before.resident_prefix_layers
        else:
            raise RuntimeError(
                "prefill remote-read requires a LOCAL/PARTIAL base "
                "history location (got "
                f"{history_before.location!r}; request {request_id!r})")
        if int(read_plan["read_prefix_layers"]) != expected_prefix_end:
            raise RuntimeError(
                "prefill remote-read template prefix "
                f"[0, {read_plan['read_prefix_layers']}) does not match "
                f"the resident prefix [0, {expected_prefix_end})")
        exec_group = self.group_by_index[exec_instance]
        home_group = self.group_by_index[home_instance]
        if len(exec_group.ranks) != len(home_group.ranks):
            raise RuntimeError(
                "prefill remote-read requires equal TP sizes")
        source_ranks = tuple(read_plan.get("source_ranks") or ())
        target_ranks = tuple(read_plan.get("target_ranks") or ())
        if (len(source_ranks) != len(home_group.ranks)
                or len(target_ranks) != len(exec_group.ranks)
                or any(
                    home_group.ranks.index(source_rank)
                    != exec_group.ranks.index(target_rank)
                    for source_rank, target_rank in zip(
                        source_ranks, target_ranks))):
            raise RuntimeError(
                "prefill remote-read template does not preserve relative "
                "TP rank (home ranks {} -> exec ranks {})".format(
                    source_ranks, target_ranks))
        if pending_gate.source_instance_index != exec_instance:
            raise RuntimeError(
                "prefill remote-read trigger gate is not on the prefill "
                f"instance (gate source {pending_gate.source_instance_index},"
                f" prefill {exec_instance}; the interval/arrival gate "
                "rebuild must precede the prefix read stream)")
        prefix = _prefix_of(request_plan)
        # §三.3：跨实例触发门构造——control = 本轮 prefill（exec）实例，
        # node_gates = 重建后的 exec 实例 timer gates（同实例退化时即原
        # arrival gate；exec != home 恒成立，1B relay 恒走跨实例路径）。
        # 首窗 tile 1 的 home send 链序后随 home trigger recv（触发链 =
        # 读流的启动因果，不设"全流到达才计算"栅栏）。
        trigger_gate = TransferTriggerGate(
            control_instance_index=exec_instance,
            node_gates=pending_gate.timer_gates,
        )

        def emit_prefill_read_trigger() -> None:
            # 触发链（恰一次消费 interval/arrival gate）：每个出现数据
            # shard 的相对位 rank 对，exec timer gate arm 后经
            # _emit_transfer_trigger 同款 1B relay 触发 home。零字节相对
            # 位守卫为 fail-closed raise（模板缺任一 rank 的扫描字节即
            # raise，非静默跳过）。数据位 = 模板逐 rank 单扫描字节
            # （D16：不物化 tile 列表即可判定位）。
            data_relative_indexes = {
                relative_index
                for relative_index, scan_shard_bytes in enumerate(
                    read_plan.get("base_scans_shards") or ())
                if scan_shard_bytes > 0}
            if len(data_relative_indexes) != len(source_ranks):
                raise RuntimeError(
                    "prefill remote-read template lacks its per-rank scan "
                    "byte vector (request {!r})".format(request_id))
            for relative_index in sorted(data_relative_indexes):
                _emit_transfer_trigger(
                    builders=self.builders,
                    group_by_index=self.group_by_index,
                    tag_allocator=self.tag_allocator,
                    trigger_gate=trigger_gate,
                    relative_index=relative_index,
                    source_rank=home_group.ranks[relative_index],
                    action_name=f"{prefix}_prefill_read",
                    shard_index=relative_index,
                )

        self._emit_side_branch(emit_prefill_read_trigger)

    def _emit_completion(self, request_plan: dict, *,
                         block_end_stage: str = "seg2") -> None:
        """completion_evictions + 下一 turn interval gate 段的在线发射
        （拼 batch 改造，2026-08-22：触发门改经 _block_ends["seg2"] =
        覆盖其退出迭代的列车 post-barrier 块末，与旧 decode 段末口径一致；
        D18：``block_end_stage`` 可锚 seg1——已知终止请求无退出列车，
        drain 块末即 P 完成）。"""
        builders = self.builders
        config = self.config
        prefix = _prefix_of(request_plan)
        decode_instance_index = request_plan["decode_instance_index"]
        decode_group = self.group_by_index[decode_instance_index]
        decode_completion_nodes = tuple(
            self._block_ends.get(request_plan["request_id"], {})
            .get(block_end_stage, {}).get(rank)
            for rank in decode_group.ranks)
        if any(node_id is None for node_id in decode_completion_nodes):
            # 列车发射后退出成员必有 seg2 块末；缺即账本损坏（fail-closed）。
            # D18（2026-09-26）：已知终止请求无退出列车，其 completion 门
            # 锚 = drain 列车 seg1 块末（P 完成即本轮全部计算完成——
            # ``block_end_stage="seg1"``，调度器终端结算路径传入）。
            raise RuntimeError(
                f"segment block ends ({block_end_stage!r}) missing on "
                "decode ranks")
        action_state = self._action_seq.setdefault(
            request_plan["request_id"], [0])

        def emit_transfer(transfer, stage: str) -> dict:
            action_sequence = action_state[0]
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config, builders=builders,
                group_by_index=self.group_by_index,
                tag_allocator=self.tag_allocator, transfer=transfer,
                action_name=action_name,
                trigger_gate=TransferTriggerGate(
                    control_instance_index=decode_instance_index,
                    node_gates=decode_completion_nodes,
                ))
            record["sequence_stage"] = stage
            record["action_sequence"] = action_sequence
            action_state[0] = action_sequence + 1
            # 2026-08-23 修订：发射侧标记已移除（决策时点同步是唯一路径，
            # 迟到的旧转移标记会把门回退到过期位置，见 sh_2.0 seq4689 事故）。
            return record

        # joint merge 回传流（§2.2/§3.2）：compute_done 后的新增量归并
        # origin_home 的真实传输（noc 回传 + 池写回），触发门 = 退出迭代
        # 的 seg2 块末；store 尾部登记入 pending_store_tails——下一轮对本
        # 会话的池读/回迁经前递补边等待写入完成（合并成本进入下一轮
        # 数据等待；所有动作/八组合一致处理）。
        merge_transfers = list(request_plan.get("merge_transfers") or ())
        for transfer in merge_transfers:
            record = emit_transfer(transfer, "merge_transfers")
            self._register_store_tails(transfer, record)
        # R11(ii)：merge 尾标记（每 decode rank 1 个小 COMP 节点，链在
        # 本 rank 的 merge 传输链之后）——下一轮 interval gate 的前递
        # 依赖（消除"下一轮物理消费尚未落地的 KV"）与 merge-done watch
        # 的成员锚点。
        merge_done_members = {}
        if merge_transfers:
            # 标记节点用 watch 命名空间上下文（batch_train_merge_<rid> /
            # prefill / 0——C++ 提交预检要求 watch 成员节点的
            # request/stage/generation 与注册一致，哨兵标记同款）。
            watch_context_id = (
                "batch_train_merge_" + request_plan["request_id"])
            for rank in decode_group.ranks:
                self.builders[rank].set_context(
                    watch_context_id, "prefill", 0)
                # op_role=control_marker（契约 §7 裁定 8）：merge done 尾
                # 标记是 comp(1,1) 微节点（每 decode rank 1 op/1 B COMP，
                # 零载荷字节、无 NoC 流——D05 式 22 控制确认的图侧载体）。
                self.builders[rank].comp(
                    f"{prefix}_merge_done_rank{rank}", 1, 1,
                    op_role="control_marker")
                merge_done_members[rank] = self.builders[rank].previous_id

        for transfer in request_plan["completion_evictions"]:
            emit_transfer(transfer, "completion_evictions")

        request_id = request_plan["request_id"]
        try:
            # completion 是 next_plan 唯一消费者；即使值为 terminal None
            # 也必须移除，避免预建的全量 request 链永久驻留。
            following = self.next_plan.pop(request_id)
        except KeyError as exc:
            raise RuntimeError(
                "completion request has no next-plan entry: {!r}".format(
                    request_id)) from exc
        if isinstance(following, str):
            # strategy 模式：值是下一 turn 的 request_id，经 resolver 取
            # 实时 plan dict（hbm_wait_ns 在准入时才落账本）。
            following = self._plan_resolver(following)
        if following is not None:
            following_request = config.request_queue[following["queue_index"]]
            interval = following_request.inter_request_interval_ns
            if interval is None:
                raise RuntimeError(
                    "later request lost its inter-request interval")
            # R11(ii)：下一轮 interval gate 前递依赖 merge 尾标记（本会话
            # 增量落地之前下一轮不得物理消费）；无 merge 流时保持 seg2
            # 块末（与改造前一致）。
            gate_anchor_by_rank = (
                merge_done_members if merge_done_members else
                {rank: decode_completion_nodes[relative_index]
                 for relative_index, rank in enumerate(decode_group.ranks)}
            )
            timers = tuple(
                builders[rank].timer_gate(
                    f"q{following['queue_index']:04d}_"
                    f"{sanitize_node_prefix(following['request_id'])}_"
                    f"history_rank{rank}_interval_gate",
                    # §9-低6：同 K5 族——interval + hbm_wait 任意 ns 粒度
                    # 须 µs 下取整（timer_gate 离线同构校验；现行输入恰在
                    # µs 网格上惰性通过，非 µs 对齐输入会确定性崩溃）。
                    (interval + following.get("hbm_wait_ns", 0)) // 1000 * 1000,
                    after_node_id=gate_anchor_by_rank[rank],
                )
                for rank in decode_group.ranks
            )
            completion_location = request_plan.get(
                "kv_location_after_completion")
            if request_plan["session_id"] in self.deferred_session_locations:
                completion_location = (
                    self.deferred_session_locations.pop(
                        request_plan["session_id"]))
            if completion_location not in {
                "local_hbm", "partial_hbm_remote", "remote_memory",
            }:
                raise RuntimeError(
                    "completed request has no valid KV location")
            self.pending_history[following["request_id"]] = PendingHistoryGate(
                source_instance_index=decode_instance_index,
                timer_gates=timers,
                location=completion_location,
            )
            self.pending_request_by_session[request_plan["session_id"]] = (
                following["request_id"])
        else:
            self.deferred_session_locations.pop(request_plan["session_id"],
                                                None)
            # KV 逐出并行化（2026-09-13）：终态会话清 store 尾部登记
            # （不再有回迁读者；等价于 kv_manager.retire_terminal_session
            # 的登记侧镜像，改动面约束下落点在构图器）。完成路径零逐出
            # 维持，不发任何传输。残余：终结完成之后才被逐出的"永不再
            # 来"会话条目驻留至 run 结束（O(会话数) 上界，无正确性影响）。
            self.pending_store_tails.pop(request_plan["session_id"], None)
        return {
            "merge_done_members": merge_done_members,
            "has_merge": bool(merge_done_members),
        }

    # ------------------------------------------------------------- 属性 --

    def set_next_plan(self, next_plan: dict) -> None:
        """request_id -> 下一 turn 的 request_id / plan dict / None。

        completion 发射是每项唯一消费点：对应 key 将在其 interval gate
        发射后 pop；terminal None 同样消费，以保持账本随在途窗口有界。
        """
        self.next_plan = next_plan

    def set_plan_resolver(self, resolver) -> None:
        """strategy 模式：request_id -> 实时 plan dict 的解析器
        （下一 turn 的 hbm_wait_ns 等在准入时才落账本）。"""
        self._plan_resolver = resolver


def _prefix_of(request_plan: dict) -> str:
    return (
        f"q{request_plan['queue_index']:04d}_"
        f"{sanitize_node_prefix(request_plan['session_id'])}_"
        f"turn{request_plan['turn_index']}_"
        f"{sanitize_node_prefix(request_plan['request_id'])}"
    )


class _PlanShim:
    """把 request_plan dict 包装为离线 helper 期望的 FaceRequestPlan 属性面
    （history_transfer 等 KVTransfer 对象在 dict 中直接携带）。"""

    __slots__ = ("_plan",)

    def __init__(self, plan: dict):
        self._plan = plan

    def __getattr__(self, name):
        try:
            return self._plan[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_plan(request_plan: dict):
    # _PlanShim 为纯代理（仅 _plan 字段 + __getattr__ 转发，无状态），每
    # 次直接构造（每请求约 5 次调用的对象构造开销可忽略）——不用 id() 做
    # 模块级缓存：id 复用隐患 + 强引用把每请求 plan 永久钉在内存（O(N)
    # 驻留，R4-11）。
    return _PlanShim(request_plan)
