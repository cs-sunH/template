#!/usr/bin/env python3
"""Pure FACE request-mapping semantic reference used by online scheduling.

The FACE paper describes a live host scheduler.  The online strategy reuses
this module's deterministic queue ordering, schedulable-instance range, direct
Roofline matching, per-die incremental decode cost, and local-first KV allocation at
dynamic decision boundaries.  It does not construct or serialize static ET
files and has no protobuf or ASTRA-sim runtime dependency.
"""

from __future__ import annotations

import heapq
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional, Sequence

from joint.eviction_priority import (
    EvictionClassPass,
    classify_session_class,
    eviction_class_order,
)
from joint.layer_eviction_policy import (
    LayerEvictionPolicy,
    LayerEvictionError,
    VictimView,
)
from joint.event_recursion_predictor import (
    ComputeLayerSegment,
    EfficiencyFactors as _EfficiencyFactors,
    EventRecursionError as _EventRecursionError,
    LayerRecursionPredictor,
    ResourceSnapshot as _ResourceSnapshot,
    CommittedFlow as _CommittedFlow,
    RestoreGroupLeg,
    ServiceFactorGroup,
    PREDICTOR_SOURCE_RECURSION,
)

PREFILL_CHUNK_SIZE = 512


class KVCapacityError(ValueError):
    """容量类失败（N3' 按调用点类型化）：deep gap / 逐出耗尽仍不足。

    与合同类异常（重复预约、context 收缩、账本破损）严格区分——后者
    继续 ValueError/RuntimeError 原样上抛。``evictions`` 携带 raise 前
    已提交的逐出转移（D7 清账语义：调用方捕获后必须 bump 纪元并同步
    图侧 pending store，不得丢弃）。``deep_gap_records`` 携带逐 rank
    缺口记录（K6：容量类失败在 joint 下有多条可恢复捕获路径——准入
    延迟/merge 方向回退（2026-09-17 起，R4 自降级已删除）/decode 停滞
    ——**raise 时不落 deep_gap_events 台账**，仅在确认不可恢复的提交
    点（死锁守卫/merge v2 双侧深缺口）经 ``commit_deep_gap_records``
    落账，恢复"落账 = run 终止"语义）。
    """

    def __init__(self, message: str, *, evictions: tuple = (),
                 deep_gap_records: tuple = ()) -> None:
        super().__init__(message)
        self.evictions = tuple(evictions)
        self.deep_gap_records = tuple(deep_gap_records)


class KVPhysicalInfeasibleError(ValueError):
    """结构性不可行：请求的终态 KV 对全部 (instance, action) 组合都
    永远放不下（设计方案 §3.3-7 显式 fail-closed 报告，带逐 rank 缺口）。
    """


# Read-only metrics observation (implementation doc sec.7).  When a recorder
# is installed through set_metrics_observer(), every KVCacheManager state
# mutation below is mirrored to it *after* the mutation completes; the
# recorder never feeds anything back into admission, eviction, or placement
# decisions.  When no recorder is installed (the default) none of the
# observation bookkeeping runs at all, so behavior and performance are
# unchanged.
_METRICS_RECORDER: Any = None


def _strict_kv_invariants_from_environment() -> bool:
    """Return whether every KV mutation must also run the complete audit."""

    return os.environ.get("SH_STRICT_KV_INVARIANTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def set_metrics_observer(recorder: Any) -> None:
    """Install the metrics recorder picked up by subsequently constructed
    managers (``None`` disables observation)."""

    global _METRICS_RECORDER
    _METRICS_RECORDER = recorder


def _metrics_anchor_for_phase(phase: str) -> str:
    """Map a planner phase to the doc sec.7.8 anchor for its memory actions."""

    return {
        "history": "prefill_start",
        "prefill": "prefill_start",
        "admission": "prefill_start",
        "decode": "decode_start",
        "prefill_decode": "decode_start",
        "completion": "completion",
    }.get(phase, "stage_boundary")


@dataclass(frozen=True)
class FaceHardware:
    mesh_rows: int
    mesh_cols: int
    local_hbm_capacity_bytes: int
    local_hbm_bandwidth_gbps: float
    d2d_bandwidth_gbps: float
    peak_perf_tflops: float
    d2d_latency_ns: int
    local_hbm_latency_ns: int
    label: str = "FACE"

    def __post_init__(self) -> None:
        integer_fields = {
            "mesh_rows": self.mesh_rows,
            "mesh_cols": self.mesh_cols,
            "local_hbm_capacity_bytes": self.local_hbm_capacity_bytes,
        }
        for name, value in integer_fields.items():
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name, value in {
            "d2d_latency_ns": self.d2d_latency_ns,
            "local_hbm_latency_ns": self.local_hbm_latency_ns,
        }.items():
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        for name, value in {
            "local_hbm_bandwidth_gbps": self.local_hbm_bandwidth_gbps,
            "d2d_bandwidth_gbps": self.d2d_bandwidth_gbps,
            "peak_perf_tflops": self.peak_perf_tflops,
        }.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
                raise ValueError(f"{name} must be positive")

    @property
    def npus_count(self) -> int:
        return self.mesh_rows * self.mesh_cols

    @property
    def schedulable_distance_limit(self) -> float:
        return self.d2d_bandwidth_gbps / self.local_hbm_bandwidth_gbps


@dataclass(frozen=True)
class FaceModel:
    layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    vocab_size: int
    bytes_per_elem: int
    mlp_variant: str = "gelu"

    def __post_init__(self) -> None:
        for name in (
            "layers",
            "hidden_size",
            "ffn_size",
            "num_heads",
            "vocab_size",
            "bytes_per_elem",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        if self.mlp_variant not in {"gelu", "swiglu"}:
            raise ValueError("mlp_variant must be gelu or swiglu")


@dataclass(frozen=True)
class FaceInstanceSpec:
    name: str
    pg_name: str
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class FaceInstance:
    index: int
    name: str
    pg_name: str
    ranks: tuple[int, ...]
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    center_row: float
    center_col: float

    @property
    def size(self) -> int:
        return len(self.ranks)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_max - self.row_min + 1, self.col_max - self.col_min + 1)


@dataclass(frozen=True)
class FaceTopology:
    hardware: FaceHardware
    instances: tuple[FaceInstance, ...]
    adjacency: tuple[tuple[int, ...], ...]

    def instance(self, index: int) -> FaceInstance:
        return self.instances[index]


def rank_coordinates(hardware: FaceHardware, rank: int) -> tuple[int, int]:
    if rank < 0 or rank >= hardware.npus_count:
        raise ValueError(f"rank {rank} is outside 0-{hardware.npus_count - 1}")
    return divmod(rank, hardware.mesh_cols)


def build_instances(
    hardware: FaceHardware,
    specs: Sequence[FaceInstanceSpec],
    *,
    require_equal_size: bool = True,
) -> FaceTopology:
    if not specs:
        raise ValueError("at least one FACE instance is required")

    all_ranks: list[int] = []
    instances: list[FaceInstance] = []
    rank_owner: dict[int, int] = {}
    pg_names: set[str] = set()

    for index, spec in enumerate(specs):
        if not spec.name:
            raise ValueError("instance name must not be empty")
        if not spec.pg_name or spec.pg_name == "0":
            raise ValueError("instance pg_name must be non-zero")
        if spec.pg_name in pg_names:
            raise ValueError(f"duplicate pg_name: {spec.pg_name}")
        pg_names.add(spec.pg_name)
        if not spec.ranks:
            raise ValueError(f"instance {spec.name} has no ranks")
        if len(set(spec.ranks)) != len(spec.ranks):
            raise ValueError(f"instance {spec.name} contains duplicate ranks")

        coordinates: set[tuple[int, int]] = set()
        for rank in spec.ranks:
            if rank in rank_owner:
                raise ValueError(
                    f"rank {rank} belongs to both instance {rank_owner[rank]} and {index}"
                )
            row, col = rank_coordinates(hardware, rank)
            coordinates.add((row, col))
            rank_owner[rank] = index
            all_ranks.append(rank)

        rows = [row for row, _ in coordinates]
        cols = [col for _, col in coordinates]
        row_min, row_max = min(rows), max(rows)
        col_min, col_max = min(cols), max(cols)
        expected = {
            (row, col)
            for row in range(row_min, row_max + 1)
            for col in range(col_min, col_max + 1)
        }
        if coordinates != expected:
            raise ValueError(
                f"instance {spec.name} must be a filled axis-aligned rectangle"
            )
        instances.append(
            FaceInstance(
                index=index,
                name=spec.name,
                pg_name=spec.pg_name,
                ranks=tuple(spec.ranks),
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                center_row=(row_min + row_max) / 2.0,
                center_col=(col_min + col_max) / 2.0,
            )
        )

    if sorted(all_ranks) != list(range(hardware.npus_count)):
        raise ValueError("FACE instances must cover every configured NPU exactly once")
    if require_equal_size and len({instance.size for instance in instances}) != 1:
        raise ValueError(
            "the online GraphBatch scheduler requires equal-size instances for "
            "KV shard pairing"
        )

    adjacency_sets = [set() for _ in instances]
    for rank in range(hardware.npus_count):
        row, col = rank_coordinates(hardware, rank)
        owner = rank_owner[rank]
        for drow, dcol in ((0, 1), (1, 0)):
            other_row, other_col = row + drow, col + dcol
            if other_row >= hardware.mesh_rows or other_col >= hardware.mesh_cols:
                continue
            other_rank = other_row * hardware.mesh_cols + other_col
            other_owner = rank_owner[other_rank]
            if owner != other_owner:
                adjacency_sets[owner].add(other_owner)
                adjacency_sets[other_owner].add(owner)

    if len(instances) > 1 and any(not neighbors for neighbors in adjacency_sets):
        raise ValueError("instance tiling must form a connected adjacency graph")

    return FaceTopology(
        hardware=hardware,
        instances=tuple(instances),
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency_sets),
    )


def estimate_model_weight_bytes(model: FaceModel) -> int:
    # Q/K/V + output projection, norms, and untied input/output embeddings.
    # LLaMA uses a gated SwiGLU MLP (gate, up, and down matrices) and RMSNorm;
    # the legacy GELU form has two MLP matrices and LayerNorm scale+bias.
    mlp_matrices = 3 if model.mlp_variant == "swiglu" else 2
    norm_elements = 2 * model.hidden_size if model.mlp_variant == "swiglu" else 4 * model.hidden_size
    per_layer_elements = (
        4 * model.hidden_size * model.hidden_size
        + mlp_matrices * model.hidden_size * model.ffn_size
        + norm_elements
    )
    embedding_elements = 2 * model.vocab_size * model.hidden_size
    final_norm_elements = model.hidden_size if model.mlp_variant == "swiglu" else 0
    return (
        model.layers * per_layer_elements + embedding_elements + final_norm_elements
    ) * model.bytes_per_elem


def kv_cache_bytes_for_tokens(model: FaceModel, tokens: int) -> int:
    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    return 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem


def partition_bytes_exact(total_bytes: int, partitions: int) -> tuple[int, ...]:
    """Split bytes deterministically while preserving the exact total."""

    if total_bytes < 0:
        raise ValueError("byte count must be non-negative")
    if partitions <= 0:
        raise ValueError("partition count must be positive")
    quotient, remainder = divmod(total_bytes, partitions)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(partitions)
    )


def attention_heads_by_tp_rank(num_heads: int, tp_degree: int) -> tuple[int, ...]:
    """Assign whole attention heads in relative TP-rank order."""

    if num_heads <= 0 or tp_degree <= 0:
        raise ValueError("num_heads and tp_degree must be positive")
    quotient, remainder = divmod(num_heads, tp_degree)
    return tuple(
        quotient + (1 if index < remainder else 0)
        for index in range(tp_degree)
    )


def model_weight_shard_bytes_by_tp_rank(
    model: FaceModel,
    tp_degree: int,
) -> tuple[int, ...]:
    """Match model-weight HBM bytes to the exact deployed TP partition.

    Attention matrices follow whole-head ownership, MLP matrices follow the
    exact FFN extents, and both untied embeddings follow the exact vocabulary
    extents.  Norm parameters form the small residual between those matrices
    and the existing global model-size estimate; that residual is distributed
    deterministically so the per-rank bytes preserve the global total exactly.
    """

    if tp_degree <= 0:
        raise ValueError("tp_degree must be positive")
    mlp_matrices = 3 if model.mlp_variant == "swiglu" else 2
    head_dim = model.hidden_size // model.num_heads
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    ffn_extents = partition_bytes_exact(model.ffn_size, tp_degree)
    vocab_extents = partition_bytes_exact(model.vocab_size, tp_degree)

    matrix_shards = tuple(
        (
            model.layers
            * (
                4 * model.hidden_size * head_count * head_dim
                + mlp_matrices * model.hidden_size * ffn_extent
            )
            + 2 * model.hidden_size * vocab_extent
        )
        * model.bytes_per_elem
        for head_count, ffn_extent, vocab_extent in zip(
            heads,
            ffn_extents,
            vocab_extents,
        )
    )
    total_weight_bytes = estimate_model_weight_bytes(model)
    residual_bytes = total_weight_bytes - sum(matrix_shards)
    if residual_bytes < 0:
        raise RuntimeError("model weight matrix shards exceed the global estimate")
    residual_shards = partition_bytes_exact(residual_bytes, tp_degree)
    shards = tuple(
        matrix_bytes + residual_bytes_for_rank
        for matrix_bytes, residual_bytes_for_rank in zip(
            matrix_shards,
            residual_shards,
        )
    )
    if sum(shards) != total_weight_bytes:
        raise RuntimeError("model weight TP shards do not preserve total bytes")
    return shards


def kv_cache_shard_bytes_for_tokens(
    model: FaceModel,
    tokens: int,
    tp_degree: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV bytes for each relative TP rank."""

    return kv_cache_shard_bytes_for_layer_range(
        model,
        tokens,
        tp_degree,
        layer_start=0,
        layer_end=model.layers,
    )


def kv_cache_shard_bytes_for_layer_range(
    model: FaceModel,
    tokens: int,
    tp_degree: int,
    *,
    layer_start: int,
    layer_end: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV bytes for ``[layer_start, layer_end)``.

    Layer ranges are derived from ``model.layers``.  This keeps suffix
    offloading model-independent and avoids any LLaMA-2-7B-specific layer
    constants in the KV manager.
    """

    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    if (
        isinstance(layer_start, bool)
        or isinstance(layer_end, bool)
        or not isinstance(layer_start, int)
        or not isinstance(layer_end, int)
        or not 0 <= layer_start <= layer_end <= model.layers
    ):
        raise ValueError("KV layer range must be within the configured model")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    bytes_per_head = (
        2
        * (layer_end - layer_start)
        * tokens
        * (model.hidden_size // model.num_heads)
        * model.bytes_per_elem
    )
    shards = tuple(head_count * bytes_per_head for head_count in heads)
    expected_total = (
        kv_cache_bytes_for_tokens(model, tokens)
        * (layer_end - layer_start)
        // model.layers
    )
    if sum(shards) != expected_total:
        raise RuntimeError(
            "whole-head layer-range KV partition does not preserve total bytes"
        )
    return shards


def physical_edge_ranks(hardware: FaceHardware) -> tuple[int, ...]:
    """Return all ranks on the deterministic physical mesh boundary."""

    edges = []
    for rank in range(hardware.npus_count):
        row, col = rank_coordinates(hardware, rank)
        if (
            row == 0
            or row == hardware.mesh_rows - 1
            or col == 0
            or col == hardware.mesh_cols - 1
        ):
            edges.append(rank)
    return tuple(edges)


def edge_free_instance_mask(
    topology: FaceTopology,
    edge_ranks: Sequence[int],
) -> tuple[bool, ...]:
    """逐实例返回 True 表示该实例不包含任何边缘 rank。

    edge_ranks 来源：KVCacheManager.edge_ranks（:1387，已归一化）。
    返回长度 == len(topology.instances)，按下标对齐，与
    select_prefill_instance 的 hbm_feasible_instances 掩码同形，
    可直接逐元素 AND 复合。
    """

    edge_rank_set = frozenset(edge_ranks)
    return tuple(
        not any(rank in edge_rank_set for rank in instance.ranks)
        for instance in topology.instances
    )


def edge_instance_mask(
    topology: FaceTopology,
    edge_ranks: Sequence[int],
) -> tuple[bool, ...]:
    """逐实例返回 True 表示该实例包含至少一个边缘 rank。

    edge_free_instance_mask 的逐元素取反；edge_ranks 来源、返回长度与
    下标对齐约定同 edge_free_instance_mask，可直接与
    select_prefill_instance 的 hbm_feasible_instances 掩码逐元素 AND
    复合。
    """

    return tuple(
        not edge_free
        for edge_free in edge_free_instance_mask(topology, edge_ranks)
    )


def manhattan_hops(hardware: FaceHardware, source: int, target: int) -> int:
    source_row, source_col = rank_coordinates(hardware, source)
    target_row, target_col = rank_coordinates(hardware, target)
    return abs(source_row - target_row) + abs(source_col - target_col)


def deterministic_xy_route(
    hardware: FaceHardware,
    source: int,
    target: int,
) -> tuple[int, ...]:
    """Route by columns first and rows second along a minimum-hop path."""

    row, col = rank_coordinates(hardware, source)
    target_row, target_col = rank_coordinates(hardware, target)
    route = [source]
    while col != target_col:
        col += 1 if target_col > col else -1
        route.append(row * hardware.mesh_cols + col)
    while row != target_row:
        row += 1 if target_row > row else -1
        route.append(row * hardware.mesh_cols + col)
    return tuple(route)


def nearest_edge_rank(
    hardware: FaceHardware,
    rank: int,
    edge_ranks: Sequence[int],
) -> int:
    if not edge_ranks:
        raise ValueError("at least one remote-memory edge rank is required")
    rank_coordinates(hardware, rank)
    return min(
        edge_ranks,
        key=lambda edge: (manhattan_hops(hardware, rank, edge), edge),
    )


@dataclass(frozen=True)
class FaceRooflineEstimate:
    instance_size: int
    p_chunk: int
    d_batch: int
    d_token: int
    iteration_time_ns: int
    source: str = "analytical_roofline"


def estimate_iteration_time_ns(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
    p_context_tokens: Optional[int] = None,
) -> int:
    if instance_size <= 0 or p_chunk < 0 or d_batch < 0 or d_token < 0:
        raise ValueError("invalid Roofline workload parameters")
    if p_context_tokens is None:
        p_context_tokens = p_chunk
    if p_context_tokens < 0:
        raise ValueError("p_context_tokens must be non-negative")
    if p_chunk == 0 and p_context_tokens != 0:
        raise ValueError("p_context_tokens must be zero without Prefill work")
    if p_chunk > 0 and p_context_tokens < p_chunk:
        raise ValueError("p_context_tokens must include the current Prefill chunk")
    if p_chunk == 0 and d_batch == 0:
        return 0

    h = model.hidden_size
    ffn = model.ffn_size
    layers = model.layers
    bytes_per_elem = model.bytes_per_elem
    aggregate_perf = instance_size * hardware.peak_perf_tflops * 1e12
    aggregate_bw = instance_size * hardware.local_hbm_bandwidth_gbps * 1e9
    local_hbm_latency_seconds = hardware.local_hbm_latency_ns / 1e9

    linear_tokens = p_chunk + d_batch
    mlp_ops_per_token = 6 * h * ffn if model.mlp_variant == "swiglu" else 4 * h * ffn
    linear_ops_per_token = layers * (8 * h * h + mlp_ops_per_token)
    linear_ops = linear_tokens * linear_ops_per_token
    weight_bytes = estimate_model_weight_bytes(model)
    activation_bytes = layers * linear_tokens * h * bytes_per_elem * 8
    linear_seconds = max(
        linear_ops / aggregate_perf,
        local_hbm_latency_seconds +
        (weight_bytes + activation_bytes) / aggregate_bw,
    )

    prefill_attn_ops = layers * 4 * p_chunk * max(1, p_context_tokens) * h
    prefill_attn_bytes = (
        layers * p_chunk * max(1, p_context_tokens) * bytes_per_elem * 4
    )
    prefill_seconds = 0.0
    if p_chunk:
        prefill_seconds = max(
            prefill_attn_ops / aggregate_perf,
            local_hbm_latency_seconds + prefill_attn_bytes / aggregate_bw,
        )

    decode_attn_ops = layers * 4 * d_batch * d_token * h
    decode_attn_bytes = layers * 2 * d_batch * d_token * h * bytes_per_elem
    decode_seconds = 0.0
    if d_batch:
        decode_seconds = max(
            decode_attn_ops / aggregate_perf,
            local_hbm_latency_seconds + decode_attn_bytes / aggregate_bw,
        )

    # FACE overlaps the two attention phases; linear operators remain a shared
    # batched portion.  A small fixed controller term keeps non-idle entries
    # positive and deterministic.
    total_seconds = linear_seconds + max(prefill_seconds, decode_seconds)
    return max(1, math.ceil(total_seconds * 1e9) + hardware.d2d_latency_ns)


def estimate_prefill_task_load_ns(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    chunk_tokens: int,
    context_tokens: int,
) -> int:
    """Return Roofline service-time load for one Prefill chunk.

    ``context_tokens`` is the complete KV-visible context at the end of the
    chunk, so attention work includes both prior session history and earlier
    chunks from the current request.
    """

    if chunk_tokens <= 0:
        raise ValueError("chunk_tokens must be positive")
    return estimate_iteration_time_ns(
        hardware,
        model,
        instance_size=instance_size,
        p_chunk=chunk_tokens,
        d_batch=0,
        d_token=0,
        p_context_tokens=context_tokens,
    )


def estimate_decode_remaining_task_load_ns(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    current_context_tokens: int,
    generated_tokens: int,
    average_decode_length: float,
    running_step_fraction_remaining: float = 1.0,
) -> int:
    """Estimate one active Decode request's remaining Roofline load.

    Future output length is the arithmetic mean supplied by the workload, not
    the request's known trace completion length.  Once ``generated_tokens`` is
    at or beyond that mean, the estimated remaining load is zero.  A fractional
    mean contributes the corresponding fraction of its final expected token.
    """

    if instance_size <= 0:
        raise ValueError("instance_size must be positive")
    if current_context_tokens <= 0 or generated_tokens < 0:
        raise ValueError("Decode context must be positive and generated_tokens non-negative")
    if (
        isinstance(average_decode_length, bool)
        or not isinstance(average_decode_length, (int, float))
        or not math.isfinite(average_decode_length)
        or average_decode_length <= 0
    ):
        raise ValueError("average_decode_length must be a positive finite number")
    if not 0.0 <= running_step_fraction_remaining <= 1.0:
        raise ValueError("running_step_fraction_remaining must be in [0, 1]")

    expected_tokens = max(float(average_decode_length) - generated_tokens, 0.0)
    if expected_tokens == 0.0:
        return 0

    load_ns = 0.0
    expected_steps = math.ceil(expected_tokens)
    for offset in range(expected_steps):
        token_fraction = min(1.0, expected_tokens - offset)
        if offset == 0:
            token_fraction *= running_step_fraction_remaining
        if token_fraction <= 0.0:
            continue
        token_time_ns = estimate_iteration_time_ns(
            hardware,
            model,
            instance_size=instance_size,
            p_chunk=0,
            d_batch=1,
            d_token=current_context_tokens + offset,
        )
        load_ns += token_fraction * token_time_ns
    return math.ceil(load_ns)


@dataclass(frozen=True)
class InstanceTaskLoadSnapshot:
    instance_index: int
    running_prefill_task_load_ns: int
    queued_prefill_task_load_ns: int
    active_decode_task_load_ns: int
    last_arrival_ns: Optional[int]

    def __post_init__(self) -> None:
        if (
            isinstance(self.instance_index, bool)
            or not isinstance(self.instance_index, int)
            or self.instance_index < 0
        ):
            raise ValueError("instance_index must be a non-negative integer")
        for name in (
            "running_prefill_task_load_ns",
            "queued_prefill_task_load_ns",
            "active_decode_task_load_ns",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        if self.last_arrival_ns is not None and (
            isinstance(self.last_arrival_ns, bool)
            or not isinstance(self.last_arrival_ns, int)
            or self.last_arrival_ns < 0
        ):
            raise ValueError("last_arrival_ns must be None or a non-negative integer")

    @property
    def total_task_load_ns(self) -> int:
        return (
            self.running_prefill_task_load_ns
            + self.queued_prefill_task_load_ns
            + self.active_decode_task_load_ns
        )

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (
            self.total_task_load_ns,
            -1 if self.last_arrival_ns is None else self.last_arrival_ns,
            self.instance_index,
        )


def select_prefill_instance(
    loads: Sequence[InstanceTaskLoadSnapshot],
    hbm_feasible_instances: Sequence[bool],
) -> int:
    if not loads:
        raise ValueError("at least one instance task-load snapshot is required")
    if len(hbm_feasible_instances) <= max(load.instance_index for load in loads):
        raise ValueError(
            "hbm_feasible_instances must cover every Prefill instance index"
        )
    if any(not isinstance(feasible, bool) for feasible in hbm_feasible_instances):
        raise ValueError("hbm_feasible_instances must contain booleans")
    feasible_loads = [
        load
        for load in loads
        if hbm_feasible_instances[load.instance_index]
    ]
    if not feasible_loads:
        raise ValueError(
            "no Prefill instance has enough reclaimable per-rank HBM for the request"
        )
    return min(feasible_loads, key=lambda load: load.ordering_key).instance_index


class WeightedInstanceGraph:
    def __init__(self, topology: FaceTopology) -> None:
        self.topology = topology
        self.weights: dict[tuple[int, int], int] = {}
        for source, neighbors in enumerate(topology.adjacency):
            for target in neighbors:
                self.weights[self._edge(source, target)] = 1

    @staticmethod
    def _edge(source: int, target: int) -> tuple[int, int]:
        return (source, target) if source < target else (target, source)

    def edge_weight(self, source: int, target: int) -> int:
        edge = self._edge(source, target)
        if edge not in self.weights:
            raise ValueError(f"instances {source} and {target} are not adjacent")
        return self.weights[edge]

    def shortest_path(self, source: int, target: int) -> tuple[float, tuple[int, ...]]:
        count = len(self.topology.instances)
        if source < 0 or source >= count or target < 0 or target >= count:
            raise ValueError("instance index is out of range")
        if source == target:
            return 0.0, (source,)

        best: dict[int, tuple[float, tuple[int, ...]]] = {source: (0.0, (source,))}
        heap: list[tuple[float, tuple[int, ...], int]] = [(0.0, (source,), source)]
        while heap:
            distance, path, node = heapq.heappop(heap)
            if best.get(node) != (distance, path):
                continue
            if node == target:
                return distance, path
            for neighbor in self.topology.adjacency[node]:
                next_distance = distance + self.edge_weight(node, neighbor)
                next_path = path + (neighbor,)
                previous = best.get(neighbor)
                if previous is None or (next_distance, next_path) < previous:
                    best[neighbor] = (next_distance, next_path)
                    heapq.heappush(heap, (next_distance, next_path, neighbor))
        raise ValueError(f"no instance path from {source} to {target}")

    def schedulable_instances(self, source: int, limit: float) -> tuple[tuple[int, float], ...]:
        candidates = []
        for instance in self.topology.instances:
            distance, _ = self.shortest_path(source, instance.index)
            if distance <= limit + 1e-12:
                candidates.append((instance.index, distance))
        return tuple(candidates)

    def increase_path(self, path: Sequence[int], amount: int = 1) -> None:
        if amount <= 0:
            raise ValueError("path weight increment must be positive")
        for source, target in zip(path, path[1:]):
            edge = self._edge(source, target)
            if edge not in self.weights:
                raise ValueError(f"invalid path edge: {source}->{target}")
            self.weights[edge] += amount

    def decrease_path(self, path: Sequence[int], amount: int = 1) -> None:
        if amount <= 0:
            raise ValueError("path weight decrement must be positive")
        for source, target in zip(path, path[1:]):
            edge = self._edge(source, target)
            if self.weights.get(edge, 0) - amount < 1:
                raise ValueError("instance-edge weight cannot fall below one")
            self.weights[edge] -= amount


@dataclass(frozen=True)
class DecodeCandidateCost:
    instance_index: int
    weighted_distance: float
    current_roofline: FaceRooflineEstimate
    updated_roofline: FaceRooflineEstimate
    delta_time_ns: int
    per_die_delta_ns: float
    remaining_hbm_capacity_bytes: int
    hbm_feasible: bool


def select_decode_instance(
    *,
    hardware: FaceHardware,
    model: FaceModel,
    topology: FaceTopology,
    graph: WeightedInstanceGraph,
    fixed_p_chunk: int,
    prefill_instance_index: int,
    has_prefill_work: Sequence[bool],
    decode_token_lengths: Sequence[Sequence[int]],
    new_request_token_length: int,
    remaining_hbm_capacity_bytes: Sequence[int],
    hbm_feasible_instances: Sequence[bool],
) -> tuple[int, tuple[DecodeCandidateCost, ...]]:
    if len(has_prefill_work) != len(topology.instances):
        raise ValueError("has_prefill_work length must match instances")
    if len(decode_token_lengths) != len(topology.instances):
        raise ValueError("decode_token_lengths length must match instances")
    if len(remaining_hbm_capacity_bytes) != len(topology.instances):
        raise ValueError("remaining_hbm_capacity_bytes length must match instances")
    if len(hbm_feasible_instances) != len(topology.instances):
        raise ValueError("hbm_feasible_instances length must match instances")
    for capacity_bytes in remaining_hbm_capacity_bytes:
        if (
            isinstance(capacity_bytes, bool)
            or not isinstance(capacity_bytes, int)
            or capacity_bytes < 0
        ):
            raise ValueError(
                "remaining_hbm_capacity_bytes must contain non-negative integers"
            )
    if any(not isinstance(feasible, bool) for feasible in hbm_feasible_instances):
        raise ValueError("hbm_feasible_instances must contain booleans")
    candidates = graph.schedulable_instances(
        prefill_instance_index,
        topology.hardware.schedulable_distance_limit,
    )
    if not candidates:
        raise ValueError("decode schedulable instance map is empty")

    costs: list[DecodeCandidateCost] = []
    for instance_index, distance in candidates:
        instance = topology.instance(instance_index)
        active_tokens = tuple(decode_token_lengths[instance_index])
        p_chunk = fixed_p_chunk if has_prefill_work[instance_index] else 0
        d_batch = len(active_tokens)
        d_token = max(active_tokens, default=0)
        current = FaceRooflineEstimate(
            instance_size=instance.size,
            p_chunk=p_chunk,
            d_batch=d_batch,
            d_token=d_token,
            iteration_time_ns=estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=instance.size,
                p_chunk=p_chunk,
                d_batch=d_batch,
                d_token=d_token,
            ),
        )
        updated_d_token = max(d_token, new_request_token_length)
        updated = FaceRooflineEstimate(
            instance_size=instance.size,
            p_chunk=p_chunk,
            d_batch=d_batch + 1,
            d_token=updated_d_token,
            iteration_time_ns=estimate_iteration_time_ns(
                hardware,
                model,
                instance_size=instance.size,
                p_chunk=p_chunk,
                d_batch=d_batch + 1,
                d_token=updated_d_token,
            ),
        )
        delta = updated.iteration_time_ns - current.iteration_time_ns
        costs.append(
            DecodeCandidateCost(
                instance_index=instance_index,
                weighted_distance=distance,
                current_roofline=current,
                updated_roofline=updated,
                delta_time_ns=delta,
                per_die_delta_ns=delta / instance.size,
                remaining_hbm_capacity_bytes=(
                    remaining_hbm_capacity_bytes[instance_index]
                ),
                hbm_feasible=hbm_feasible_instances[instance_index],
            )
        )
    feasible_costs = [cost for cost in costs if cost.hbm_feasible]
    if not feasible_costs:
        raise ValueError(
            "no schedulable Decode instance has enough reclaimable per-rank HBM"
        )
    # Preserve FACE cost priority among capacity-feasible candidates; consult
    # live HBM only for exact cost ties.
    selected = min(
        feasible_costs,
        key=lambda cost: (
            cost.per_die_delta_ns,
            -cost.remaining_hbm_capacity_bytes,
            cost.instance_index,
        ),
    )
    return selected.instance_index, tuple(costs)


@dataclass(frozen=True)
class KVAllocationPiece:
    instance_index: int
    bytes: int
    weighted_distance: float
    path: tuple[int, ...]
    rank_bytes: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class KVAllocation:
    request_id: str
    decode_instance_index: int
    total_bytes: int
    pieces: tuple[KVAllocationPiece, ...]


class KVAllocator:
    def __init__(
        self,
        topology: FaceTopology,
        graph: WeightedInstanceGraph,
        *,
        model_weight_bytes: int,
    ) -> None:
        self.topology = topology
        self.graph = graph
        self.model_weight_bytes = model_weight_bytes
        self.remaining_capacity = []
        for instance in topology.instances:
            capacity = (
                instance.size * topology.hardware.local_hbm_capacity_bytes
                - model_weight_bytes
            )
            if capacity < 0:
                raise ValueError(
                    f"model does not fit in instance {instance.name}: "
                    f"needs {model_weight_bytes}, has "
                    f"{instance.size * topology.hardware.local_hbm_capacity_bytes}"
                )
            self.remaining_capacity.append(capacity)

    def allocate(
        self,
        *,
        request_id: str,
        decode_instance_index: int,
        candidate_indices: Sequence[int],
        total_bytes: int,
    ) -> KVAllocation:
        if total_bytes < 0:
            raise ValueError("KV allocation size must be non-negative")
        unique_candidates = tuple(dict.fromkeys(candidate_indices))
        if decode_instance_index not in unique_candidates:
            raise ValueError("decode instance must be in the KV candidate set")

        ordered: list[tuple[int, float, tuple[int, ...]]] = []
        for index in unique_candidates:
            distance, path = self.graph.shortest_path(decode_instance_index, index)
            ordered.append((index, distance, path))
        ordered.sort(
            key=lambda item: (
                item[0] != decode_instance_index,
                item[1],
                -self.remaining_capacity[item[0]],
                item[0],
            )
        )

        if sum(self.remaining_capacity[index] for index, _, _ in ordered) < total_bytes:
            raise ValueError(
                f"insufficient candidate HBM capacity for request {request_id}: "
                f"need {total_bytes} bytes"
            )

        remaining = total_bytes
        pieces: list[KVAllocationPiece] = []
        for index, distance, path in ordered:
            if remaining == 0:
                break
            take = min(remaining, self.remaining_capacity[index])
            if take == 0:
                continue
            self.remaining_capacity[index] -= take
            remaining -= take
            piece = KVAllocationPiece(
                instance_index=index,
                bytes=take,
                weighted_distance=distance,
                path=path,
            )
            pieces.append(piece)
            if index != decode_instance_index:
                self.graph.increase_path(path)

        return KVAllocation(
            request_id=request_id,
            decode_instance_index=decode_instance_index,
            total_bytes=total_bytes,
            pieces=tuple(pieces),
        )

    def release(self, allocation: KVAllocation) -> None:
        for piece in allocation.pieces:
            self.remaining_capacity[piece.instance_index] += piece.bytes
            if piece.instance_index != allocation.decode_instance_index:
                self.graph.decrease_path(piece.path)


@dataclass(frozen=True)
class NodeHBMSnapshot:
    rank: int
    instance_index: int
    capacity_bytes: int
    model_weight_bytes: int
    kv_cache_bytes: int
    used_bytes: int
    remaining_bytes: int


@dataclass
class NodeHBMState:
    rank: int
    instance_index: int
    capacity_bytes: int
    model_weight_bytes: int
    kv_cache_bytes: int = 0

    @property
    def used_bytes(self) -> int:
        return self.model_weight_bytes + self.kv_cache_bytes

    @property
    def remaining_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def snapshot(self) -> NodeHBMSnapshot:
        return NodeHBMSnapshot(
            rank=self.rank,
            instance_index=self.instance_index,
            capacity_bytes=self.capacity_bytes,
            model_weight_bytes=self.model_weight_bytes,
            kv_cache_bytes=self.kv_cache_bytes,
            used_bytes=self.used_bytes,
            remaining_bytes=self.remaining_bytes,
        )


@dataclass(frozen=True)
class SessionKVSnapshot:
    session_id: str
    location: str
    instance_index: Optional[int]
    context_tokens: int
    total_bytes: int
    shard_bytes: tuple[int, ...]
    resident_prefix_layers: int
    local_bytes: int
    remote_bytes: int
    local_shard_bytes: tuple[int, ...]
    remote_shard_bytes: tuple[int, ...]
    rank_bytes: tuple[tuple[int, int], ...]
    last_completion_ns: Optional[int]
    active: bool
    # joint 语义（§2.1/§2.2）：逻辑 home 与跨实例执行的工作副本状态。
    home_instance: Optional[int] = None
    working_kind: Optional[str] = None
    working_instance_index: Optional[int] = None


@dataclass(frozen=True)
class _BasePrefixView:
    """home 侧基础前缀只读视图（merge v2 反向腿，2026-09-17；primary
    session 的 instance/context 字段此刻描述执行端工作副本，不可直接喂
    ``_noc_transfer``）。字段恰好是 ``_noc_transfer`` 鸭子类型所需的
    全部：session_id/instance_index/context_tokens/resident_prefix_layers。"""

    session_id: str
    instance_index: int
    context_tokens: int
    resident_prefix_layers: int


@dataclass
class SessionKVState:
    session_id: str
    location: str
    instance_index: Optional[int]
    context_tokens: int
    total_bytes: int
    shard_bytes: tuple[int, ...]
    resident_prefix_layers: int
    last_completion_ns: Optional[int] = None
    active: bool = False
    # Trigger type of the session's next request ("human"/"tool"), recorded
    # by mark_complete from the completed request's next_trigger_type. It
    # drives the typed eviction order; None (unspecified) is treated as the
    # human class (user ruling: sessions with no successor are human class).
    next_request_type: Optional[str] = None
    # ---- joint 三机制语义（设计方案 §2） ----
    # 逻辑 home：首轮发射建立；异地执行（copy/recompute/remote-read）与
    # 逐出到片外都不改变 home（§2.1）。字段缺省 None 仅为 dataclass 默认
    # 值兼容；在线运行中首个 prepare_prefill 一定会建立。
    home_instance: Optional[int] = None
    # 跨实例执行的工作副本记账（§2.2 表）：执行 instance 持有工作副本时，
    # location/instance_index/shard_bytes 描述**工作副本**（本地全量、随
    # 执行增长），base_* 描述 home 侧权威基础历史的驻留状态；merge_back
    # 后清空并恢复 base。working_kind：
    #   "copy"       基础历史整份复制到执行端 + 增量
    #   "remote-read" 基础历史留在 home，执行端仅持增量
    #   "recompute"  执行端重算历史 + 增量（base 若在即权威副本仍有效）
    working_kind: Optional[str] = None
    base_history_tokens: int = 0
    base_shard_bytes: tuple[int, ...] = ()
    base_resident_prefix_layers: int = 0
    base_location: str = ""
    # merge 事务版本键（R4/N9，2026-09-14）：最近一次已结算的
    # merge_back 触发请求；同请求重复 merge 即合同类违规 fail-closed。
    last_merged_request_id: Optional[str] = None
    # C13 copy 逐 chunk 四步交接账本（设计文档 §2.3；None = 本轮非
    # copy 跨实例或已闭合）。轮内权威：home 侧残量的唯一事实源。
    copy_handoff: Optional["CopyHandoffJournal"] = None
    # C15 后缀逐组恢复区间账本（守恒科目 rid#restore；None = 本轮无
    # 后缀池恢复）。issue 于准入相，complete 于 prefill drain/merge。
    restore_journal: Optional["RestoreGroupJournal"] = None

    @property
    def working_instance_index(self) -> Optional[int]:
        if self.working_kind is None:
            return None
        return self.instance_index


#: C13 copy 逐 chunk 交接的块层跨度目标（确定性切分：块跨度 = 8 层，
#: n = ceil(base_prefix / 8)；base_prefix ≤ 8 时单块 = 旧单笔口径
#: 回归锚——test_face_scheduler 的两腿/单腿结构断言依赖单块形态）。
COPY_HANDOFF_CHUNK_LAYERS = 8

#: C15 后缀逐组恢复的组层跨度目标（确定性切分，与 C13 同款规划器形
#: 状；消费顺序 = 层自低向高）。后缀 ≤ 8 层时单组 = 旧单笔口径回归锚
#: （transfers 数量/区间不变，仅多 restore_group 标记）。
RESTORE_GROUP_LAYERS = 8


def plan_suffix_restore_groups(
    suffix_start: int, model_layers: int,
) -> tuple[tuple[int, int], ...]:
    """C15：后缀 [suffix_start, model_layers) 的消费顺序层组切分。

    与 plan_copy_handoff_layer_chunks 同款确定性规划：组数 n =
    ceil(后缀层数 / RESTORE_GROUP_LAYERS)，组内跨度均衡；返回
    [(layer_start, layer_end), ...]（层自低向高 = 消费顺序）。
    """
    span_total = model_layers - suffix_start
    if span_total <= 0:
        return ()
    group_count = -(-span_total // RESTORE_GROUP_LAYERS)
    span = -(-span_total // group_count)
    ranges: list[tuple[int, int]] = []
    start = suffix_start
    while start < model_layers:
        end = min(start + span, model_layers)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


def plan_copy_handoff_layer_chunks(
    base_prefix_layers: int,
) -> tuple[tuple[int, int], ...]:
    """C13：copy 驻留前缀 [0, base_prefix) 的消费顺序层块切分。

    消费顺序 = 层自 0 向上（transformer 深度序）；块数 n =
    ceil(base_prefix / COPY_HANDOFF_CHUNK_LAYERS)，块内跨度均衡
    （span = ceil(base_prefix / n)）。确定性、无运行期状态输入。
    """
    if base_prefix_layers <= 0:
        return ()
    chunk_count = -(-base_prefix_layers // COPY_HANDOFF_CHUNK_LAYERS)
    span = -(-base_prefix_layers // chunk_count)
    ranges: list[tuple[int, int]] = []
    start = 0
    while start < base_prefix_layers:
        end = min(start + span, base_prefix_layers)
        ranges.append((start, end))
        start = end
    return tuple(ranges)


@dataclass
class CopyHandoffChunk:
    """单个 KV 层块的四步交接状态（PENDING → HANDED_OFF）。"""

    index: int
    layer_start: int
    layer_end: int
    shard_bytes: tuple[int, ...]
    state: str = "PENDING"

    @property
    def total_bytes(self) -> int:
        return sum(self.shard_bytes)


@dataclass
class CopyHandoffJournal:
    """C13 copy 块级交接账本（设计文档 §2.3 守恒式的被检对象）。

    逐 chunk 四步协议的账本侧：准入相（prepare_prefill）落实安全前提
    与目标空间并登记块序列；每个交接完成事件在同一事件内完成"目标侧
    权威化 + home 侧立即释放"（不等轮末）；轮末（merge_back）断言闭
    合。守恒式（逐 rank 线性算术精确、fail-closed）：

        H_home(t) + H_exec(t) = H + D_handoff(t)

    其中 H = 本次迁移的基础历史（驻留前缀）；H_home = 尚未释放的块字
    节；H_exec = 执行端已物化的迁移历史（本仓口径：工作副本在准入相
    即物化入执行端容量账——容量保守上界，物理到达时刻由图侧逐块门
    控）；D_handoff = 已形成但尚未释放的重复有效载荷（该口径下 =
    H_home：准入物化 ∧ 未释放）。事件级计量入 ``events``（科目
    ``rid#handoff`` / ``rid#copy-stream``，与 R15 流登记 owner 串同
    命名纪律）。
    """

    session_id: str
    trigger_request_id: str
    home_instance: int
    exec_instance: int
    base_history_tokens: int
    chunks: tuple[CopyHandoffChunk, ...]
    events: list[dict] = field(default_factory=list)
    # 独立维护的守恒计数器（与 chunks 状态交叉核验，防漂移）：
    # h_home 随释放递减；h_exec 在 launch 时置 H（准入物化口径）。
    h_home_shards: tuple[int, ...] = ()
    h_exec_shards: tuple[int, ...] = ()
    released_indices: set = field(default_factory=set)

    @property
    def total_shards(self) -> tuple[int, ...]:
        if not self.chunks:
            return ()
        return tuple(
            sum(chunk.shard_bytes[rank] for chunk in self.chunks)
            for rank in range(len(self.chunks[0].shard_bytes)))

    def home_shards(self) -> tuple[int, ...]:
        """H_home(t)：home 侧尚未释放的残量（= Σ 未释放块）。"""
        if not self.chunks:
            return ()
        return tuple(
            sum(chunk.shard_bytes[rank] for chunk in self.chunks
                if chunk.index not in self.released_indices)
            for rank in range(len(self.chunks[0].shard_bytes)))

    def released_shards(self) -> tuple[int, ...]:
        if not self.chunks:
            return ()
        return tuple(
            sum(chunk.shard_bytes[rank] for chunk in self.chunks
                if chunk.index in self.released_indices)
            for rank in range(len(self.chunks[0].shard_bytes)))

    def d_handoff_shards(self) -> tuple[int, ...]:
        """D_handoff(t)：已物化（准入）∧ 尚未释放的重复有效载荷。"""
        total = self.total_shards
        released = self.released_shards()
        return tuple(t - r for t, r in zip(total, released))

    def pending_chunks(self) -> tuple[CopyHandoffChunk, ...]:
        return tuple(chunk for chunk in self.chunks
                     if chunk.state == "PENDING")

    def next_pending_index(self) -> Optional[int]:
        pending = self.pending_chunks()
        return pending[0].index if pending else None

    def all_settled(self) -> bool:
        return all(chunk.state == "HANDED_OFF" for chunk in self.chunks)

    def assert_conservation(self) -> None:
        """守恒式逐 rank 精确断言 + 计数器/块状态交叉核验（fail-closed）。"""
        total = self.total_shards
        h_home = self.home_shards()
        d_handoff = self.d_handoff_shards()
        for rank, (home_bytes, exec_bytes, d_bytes, total_bytes) in enumerate(
                zip(h_home, self.h_exec_shards, d_handoff, total)):
            if home_bytes + exec_bytes != total_bytes + d_bytes:
                raise RuntimeError(
                    f"copy handoff conservation failed for session "
                    f"{self.session_id} rank {rank}: H_home({home_bytes}) + "
                    f"H_exec({exec_bytes}) != H({total_bytes}) + "
                    f"D_handoff({d_bytes}) -- ledger integrity failure")
        # 交叉核验：独立计数器与块状态推导一致（计数器漂移 = 账本破损）。
        if h_home != self.h_home_shards:
            raise RuntimeError(
                f"copy handoff home-side counter drifted from chunk states "
                f"for session {self.session_id}: "
                f"{self.h_home_shards} != {h_home}")
        if any(value < 0 for value in d_handoff) or any(
                value < 0 for value in self.h_home_shards):
            raise RuntimeError(
                f"copy handoff released more than the migrated history for "
                f"session {self.session_id} (double/early release)")


@dataclass
class RestoreGroupEntry:
    """C15：单个后缀恢复组的区间账本条目（ISSUED → CONSUMED）。"""

    index: int
    layer_start: int
    layer_end: int
    shard_bytes: tuple[int, ...]
    state: str = "ISSUED"

    @property
    def total_bytes(self) -> int:
        return sum(self.shard_bytes)


@dataclass
class RestoreGroupJournal:
    """C15 逐组恢复区间账本（守恒科目 ``rid#restore`` issue/complete）。

    准入相（prepare_prefill）按消费顺序登记后缀恢复组（issue）；各组
    物理到达由图侧逐组就绪门控（GB ``_suffix_restore_arms``——列车体
    按层段消费）；结算边界（prefill drain；merge 兜底）把全部已登记组
    置 CONSUMED（complete）。守恒式（逐 rank 精确、fail-closed）：

        Σ issued_bytes = Σ in_flight_bytes + Σ consumed_bytes

    迟到组的消费等待如实由图依赖承载（该层段消费等待入账，运行期不补
    救：不切读路径、不在线调参——§5.6）。
    """

    session_id: str
    trigger_request_id: str
    target_instance_index: int
    suffix_start: int
    entries: tuple[RestoreGroupEntry, ...]
    events: list[dict] = field(default_factory=list)

    def in_flight_bytes_by_rank(self) -> tuple[int, ...]:
        pending = [e for e in self.entries if e.state == "ISSUED"]
        if not pending:
            return ()
        return tuple(
            sum(e.shard_bytes[rank] for e in pending)
            for rank in range(len(pending[0].shard_bytes)))

    def consumed_bytes_by_rank(self) -> tuple[int, ...]:
        done = [e for e in self.entries if e.state == "CONSUMED"]
        if not done:
            return ()
        return tuple(
            sum(e.shard_bytes[rank] for e in done)
            for rank in range(len(done[0].shard_bytes)))

    def issued_bytes_by_rank(self) -> tuple[int, ...]:
        if not self.entries:
            return ()
        return tuple(
            sum(e.shard_bytes[rank] for e in self.entries)
            for rank in range(len(self.entries[0].shard_bytes)))

    def all_consumed(self) -> bool:
        return all(e.state == "CONSUMED" for e in self.entries)

    def settle(self, boundary: str) -> None:
        """结算边界：全部已登记组置 CONSUMED（重复结算 fail-closed）。"""
        if self.all_consumed() and self.entries:
            raise RuntimeError(
                f"restore group journal for {self.session_id} settled "
                f"twice (boundary {boundary!r})")
        for entry in self.entries:
            if entry.state != "ISSUED":
                raise RuntimeError(
                    f"restore group {entry.index} of {self.session_id} has "
                    f"illegal state {entry.state!r}")
            entry.state = "CONSUMED"
        self.events.append({
            "subject": f"{self.trigger_request_id}#restore",
            "event": "complete",
            "boundary": boundary,
            "groups": len(self.entries),
            "issued_bytes": sum(self.issued_bytes_by_rank()),
        })

    def assert_conservation(self) -> None:
        issued = self.issued_bytes_by_rank()
        in_flight = self.in_flight_bytes_by_rank()
        consumed = self.consumed_bytes_by_rank()
        for rank, (i, f, c) in enumerate(zip(issued, in_flight, consumed)):
            if i != f + c:
                raise RuntimeError(
                    f"restore group conservation failed for session "
                    f"{self.session_id} rank {rank}: issued({i}) != "
                    f"in_flight({f}) + consumed({c})")


@dataclass(frozen=True)
class KVCapacityReservation:
    request_id: str
    session_id: str
    instance_index: int
    final_context_tokens: int
    final_shard_bytes: tuple[int, ...]
    action: Optional[str] = None
    suffix_history_tokens: Optional[int] = None
    suffix_start_layer: Optional[int] = None


@dataclass(frozen=True)
class KVTransferShard:
    source_rank: Optional[int]
    target_rank: Optional[int]
    edge_rank: Optional[int]
    bytes: int
    noc_path: tuple[int, ...]
    layer_start: int
    layer_end: int

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("KV transfer shard bytes must be non-negative")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("KV transfer shard requires a non-empty layer range")


@dataclass(frozen=True)
class KVTransfer:
    kind: str
    phase: str
    reason: str
    session_id: str
    trigger_request_id: str
    source_instance_index: Optional[int]
    target_instance_index: Optional[int]
    total_bytes: int
    shards: tuple[KVTransferShard, ...]
    model_layers: int
    layer_start: int
    layer_end: int
    resident_prefix_layers_before: int
    resident_prefix_layers_after: int
    # C13 copy 逐 chunk 交接：本传输承载的交接块序号（0 基，消费顺序）。
    # None = 非交接块（stay/remote-read/recompute 腿、copy 的池恢复后缀
    # 腿等）。块 0 = 准入主链头块（readiness barrier 只等它——不设
    # "先整份搬运后计算"串行段）；块 ≥ 1 由构图器旁挂支链流水发射。
    handoff_chunk: Optional[int] = None
    # C15 逐组恢复：本传输承载的后缀恢复组序号（0 基，消费顺序 = 层自
    # 低向高）。None = 非逐组恢复腿（旧单笔口径/逐出写回/交接腿）。
    # 构图器按组旁挂恢复支链（rank 内串行链），列车体按层段就绪门控
    # 消费——替换"整列车等整段后缀"的保守门（GB _suffix_restore_arms）。
    restore_group: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind not in {
            "local_hit",
            "noc_migrate",
            "remote_load",
            "remote_store",
        }:
            raise ValueError(f"unsupported KV transfer kind: {self.kind}")
        if self.total_bytes < 0:
            raise ValueError("KV transfer bytes must be non-negative")
        if (
            self.model_layers <= 0
            or not 0 <= self.layer_start < self.layer_end <= self.model_layers
        ):
            raise ValueError("KV transfer layer range is outside the model")
        if not (
            0 <= self.resident_prefix_layers_before <= self.model_layers
            and 0 <= self.resident_prefix_layers_after <= self.model_layers
        ):
            raise ValueError("KV resident-prefix metadata is outside the model")
        shard_total = sum(shard.bytes for shard in self.shards)
        if self.kind == "local_hit" and self.shards:
            raise ValueError("local-hit KV transfer must not contain network shards")
        if self.kind != "local_hit" and shard_total != self.total_bytes:
            raise ValueError("KV transfer shard bytes do not match total bytes")
        if any(
            shard.layer_start != self.layer_start
            or shard.layer_end != self.layer_end
            for shard in self.shards
        ):
            raise ValueError("KV transfer shards must use the transfer layer range")


#: C14 kv_delta_journal 行的冻结字段面（F3 序列化导出口径；构造序 =
#: _append_kv_delta_row 的 append 序，§20.3 落字字段名不发明新口径：
#: trigger_request_id=请求标识、direction=方向（stay|forward|reverse|
#: in_place）、transferred_bytes=delta 字节、winner/loser/home_before/
#: home_after/home_migration=home 迁移轨迹、两侧 retained=new_tokens
#: 后两侧实际保留量、staging_return_bytes=F14 恒 0 披露位）。
#: journal 为**逐请求结算行**——rank/层区间不在 C14 记账结构（copy 块层
#: 区间在 copy_handoff_events、后缀恢复组层区间在 restore_events，各有
#: 其科目），序列化导出不代拟逐 rank/逐层拆分。
KV_DELTA_JOURNAL_FIELDS = (
    "seq", "session_id", "trigger_request_id", "working_kind",
    "direction", "zero_byte_flip", "winner_instance", "loser_instance",
    "home_before", "home_after", "home_migration", "transferred_bytes",
    "home_side_retained_bytes", "exec_side_retained_bytes", "new_tokens",
    "staging_return_bytes",
)


class KVCacheManager:
    """Per-rank HBM manager with model-derived layer-suffix offloading."""

    LOCAL_HBM = "local_hbm"
    PARTIAL_HBM_REMOTE = "partial_hbm_remote"
    REMOTE_MEMORY = "remote_memory"

    def __init__(
        self,
        topology: FaceTopology,
        model: FaceModel,
        *,
        edge_ranks: Optional[Sequence[int]] = None,
        strict_invariants: Optional[bool] = None,
        category_mode: str = "typed",
        layer_policy: str = "legacy_half",
        pool_bandwidth_gbps: Optional[float] = None,
        pool_latency_ns: Optional[int] = None,
        prefill_ns_per_token: Optional[float] = None,
        pool_divisor_fn=None,
    ) -> None:
        if strict_invariants is not None and not isinstance(strict_invariants, bool):
            raise ValueError("strict_invariants must be a bool or None")
        instance_sizes = {instance.size for instance in topology.instances}
        if len(instance_sizes) != 1:
            raise ValueError("KVCacheManager requires equal-size TP instances")

        self.topology = topology
        self.model = model
        self.tp_degree = topology.instances[0].size
        # ---- joint 三机制注入（设计方案 §4/§5/§7.1）----
        # T：victim 类别顺序（typed=人类优先严格序 / lru=类型无关）。
        self.category_mode = category_mode
        self._eviction_passes: tuple[EvictionClassPass, ...] = (
            eviction_class_order(category_mode))
        # E：层数策略（adaptive/minimal_layer_groups/legacy_half）。
        self.layer_policy_mode = layer_policy
        self.layer_policy = LayerEvictionPolicy(layer_policy, model.layers)
        # adaptive 的消费期限估计输入（§5.3/§5.4）：池速率来自硬件配置
        # （remote-memory.bandwidth-gbps / latency-ns）；缺省 None 时
        # adaptive 目标退化为保守保留（cold start，等价 E-off 行为并标记）。
        if pool_bandwidth_gbps is not None and pool_bandwidth_gbps <= 0:
            raise ValueError("pool_bandwidth_gbps must be positive when given")
        self.pool_bandwidth_gbps = pool_bandwidth_gbps
        self.pool_latency_ns = int(pool_latency_ns or 0)
        # P1（R15-3）：池端口仲裁份额注入（instance_index -> 除数，含
        # 候选/恢复流自身 +1）；None = 单流全带宽（单测/离线口径）。
        self._pool_divisor_fn = pool_divisor_fn
        # prefill 逐 token 服务时长（roofline 派生，事前定义）：调度器从
        # 硬件/模型派生注入；缺省用 roofline 估计函数的均值近似。
        self._prefill_ns_per_token = prefill_ns_per_token
        # E 的在线输入长度估计（§5.5：session 均值 → run 均值 → cold）。
        self._input_length_stats: dict[str, list[int]] = {}
        self._run_input_count = 0
        self._run_input_sum = 0
        # Keep the larger half resident for odd-layer models, so exactly
        # floor(L/2) trailing layers are selected for the first-stage offload.
        self.partial_resident_prefix_layers = (
            model.layers - model.layers // 2
        )
        self.model_weight_bytes_by_tp_rank = model_weight_shard_bytes_by_tp_rank(
            model,
            self.tp_degree,
        )

        if edge_ranks is None:
            normalized_edges = physical_edge_ranks(topology.hardware)
        else:
            provided_edges = tuple(edge_ranks)
            if any(
                isinstance(rank, bool) or not isinstance(rank, int)
                for rank in provided_edges
            ):
                raise ValueError("edge_ranks must contain integer ranks")
            if len(set(provided_edges)) != len(provided_edges):
                raise ValueError("edge_ranks must not contain duplicates")
            normalized_edges = tuple(sorted(provided_edges))
        if not normalized_edges:
            raise ValueError("at least one remote-memory edge rank is required")
        for rank in normalized_edges:
            rank_coordinates(topology.hardware, rank)
        self.edge_ranks = normalized_edges

        self._rank_states: dict[int, NodeHBMState] = {}
        self._rank_relative_index: dict[int, int] = {}
        for instance in topology.instances:
            for relative_index, rank in enumerate(instance.ranks):
                state = NodeHBMState(
                    rank=rank,
                    instance_index=instance.index,
                    capacity_bytes=topology.hardware.local_hbm_capacity_bytes,
                    model_weight_bytes=(
                        self.model_weight_bytes_by_tp_rank[relative_index]
                    ),
                )
                if state.remaining_bytes < 0:
                    raise ValueError(
                        f"model weight shard does not fit rank {rank}: "
                        f"needs {state.model_weight_bytes}, "
                        f"has {state.capacity_bytes}"
                    )
                self._rank_states[rank] = state
                self._rank_relative_index[rank] = relative_index
        self._sessions: dict[str, SessionKVState] = {}
        self._reservations: dict[str, KVCapacityReservation] = {}
        self._strict_kv_invariants = (
            _strict_kv_invariants_from_environment()
            if strict_invariants is None
            else strict_invariants
        )
        self._check_invariants()
        self._initialize_incremental_invariants()
        # Metrics observation state (doc sec.7.4): resident KV is tracked as
        # per-session parts that carry their own token counts and layer
        # ranges, so suffix-half evictions, restores, and truncations always
        # remove the exact recorded distribution of an earlier add under its
        # own allocation key (never a merged guess).
        self._metrics_recorder = _METRICS_RECORDER
        self._metrics_parts: dict[str, list[dict[str, Any]]] = {}
        self._metrics_segment_counters: dict[str, int] = {}
        # D4 (2026-09-05) + K6 (2026-09-14)：深缺口台账——落账 = 确认
        # 不可恢复（死锁守卫/降级耗尽提交；GREEN run 恒空，可恢复失败
        # 经 KVCapacityError.deep_gap_records 携带不入账）。
        self.deep_gap_events: list[dict[str, object]] = []
        # R4 (2026-09-14)：merge 自降级事件台账——**已冻结**（2026-09-17
        # 裁定④：merge v2 上线、R4 自降级整段删除，属性仅为历史 run 侧车
        # 兼容读取保留，恒空、不再 append）。
        self.merge_degrade_events: list[dict[str, object]] = []
        # 合并方向 v2（2026-09-17，§4.3.2）：每次 merge_back（含 stay 早退）
        # 结束前设置的披露快照——键 session_id/direction("stay"|"forward"|
        # "reverse"|"in_place")/zero_byte_flip/winner_instance/loser_instance/
        # transferred_bytes/home_flipped；供调度器完成行日志与水印重放消费。
        self.last_merge_outcome: Optional[dict] = None
        # C13（2026-09-22）：copy 块级交接守恒科目台账——全部会话的
        # #handoff / #copy-stream 事件流（launch/handoff/close；与
        # deep_gap_events 同披露纪律：GREEN run 亦保留——守恒审计输入，
        # 非失败台账）。科目 owner 串与 R15 流登记（rid#readplan /
        # rid#decode#j / rid#merge）同命名纪律。
        self.copy_handoff_events: list[dict[str, object]] = []
        # C15：后缀逐组恢复的 issue/complete 审计台账（科目 rid#restore）。
        self.restore_events: list[dict[str, object]] = []
        # C15：adaptive 决策披露侧车（不改 SH 的通道）——预测器来源
        # （递推/解析）与覆盖状态逐决策落账（PROVENANCE §23）。
        self.adaptive_decisions: list[dict[str, object]] = []
        # C15：η/γ 在线效率估计组（§5.4 因果更新；逐扫描点重置 = 新
        # manager 实例，同 run 持续更新）。缺可分离样本保持原估计并
        # 标记不可观测（mark_unobservable）。
        self.service_factors = ServiceFactorGroup()
        # C14（2026-09-22）：kv_delta_journal——结算时刻逐请求事实侧车
        # （披露接口声明，C14 步骤 5）：每次 merge_back 成功结算（含 stay
        # 早退的本地提交）恰追加一行；失败结算（双侧深缺口 fail-closed）
        # 不追加——行存在 ⇔ 结算完成，失败分类另走 deep_gap_events，两
        # 科目互斥闭合。字段面向 C16 消费面（home 迁移轨迹 = session_id +
        # home_before/home_after/home_migration + trigger_request_id）与
        # 逐请求合并披露（merge_direction/zero_byte_flip/传输字节/两侧
        # 实际保留量）。**不进 C5 冻结的 decision-log schema**（决策时刻
        # 记录）；run 级 sidecar 序列化按 C13 (a) 同款登记归
        # online_service 属主（C16/C19 顺带；F3 已履行——行源 =
        # kv_delta_journal_rows，第四键落 dump_joint_kv_ledgers）。
        # staging_return_bytes 恒 0
        # = F14 无留存型暂存的"暂存归还 = 无操作"披露位（§15.6）。
        self.kv_delta_journal: list[dict[str, object]] = []
        # O5: trigger_request_id → 最新结算行索引（热路径 kv_delta_find
        # O(1)——原 reversed() 线性扫全 run O(n²)；仅 _append_kv_delta_row
        # 唯一写点维护，append-only 账本索引只增不改，语义=最新命中）。
        self._kv_delta_index: dict[str, dict[str, object]] = {}
        if self._metrics_recorder is not None:
            for rank in sorted(self._rank_states):
                self._metrics_recorder.initialize_rank(
                    rank, self._rank_states[rank].capacity_bytes
                )
            for rank in sorted(self._rank_states):
                state = self._rank_states[rank]
                self._metrics_recorder.record(
                    planner_time_ns=0,
                    anchor_kind="tick_zero",
                    request_id=None,
                    session_id=None,
                    rank=rank,
                    allocation_key=f"weight:{rank}",
                    weight_delta_bytes=state.model_weight_bytes,
                    cause="model_weight_preload",
                )

    @property
    def node_states(self) -> tuple[NodeHBMState, ...]:
        return tuple(self._rank_states[rank] for rank in sorted(self._rank_states))

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    def has_session(self, session_id: str) -> bool:
        """Return session membership without materializing the sorted view."""
        return session_id in self._sessions

    def hbm_snapshots(
        self,
        instance_index: Optional[int] = None,
    ) -> tuple[NodeHBMSnapshot, ...]:
        if instance_index is None:
            ranks = tuple(sorted(self._rank_states))
        else:
            ranks = self.topology.instance(instance_index).ranks
        return tuple(self._rank_states[rank].snapshot() for rank in ranks)

    def instance_remaining_capacity_bytes(self, instance_index: int) -> int:
        return sum(
            self._rank_states[rank].remaining_bytes
            for rank in self.topology.instance(instance_index).ranks
        )

    def instance_remaining_capacity_totals(self) -> tuple[int, ...]:
        return tuple(
            self.instance_remaining_capacity_bytes(instance.index)
            for instance in self.topology.instances
        )

    def _local_session_shards(
        self,
        session_id: str,
        instance_index: int,
    ) -> tuple[int, ...]:
        session = self._sessions.get(session_id)
        if (
            session is None
            or session.instance_index != instance_index
            or session.location == self.REMOTE_MEMORY
        ):
            return tuple(0 for _ in range(self.tp_degree))
        return kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=session.resident_prefix_layers,
        )

    def _reservation_extra_shards(
        self,
        reservation: KVCapacityReservation,
    ) -> tuple[int, ...]:
        suffix_shards = self._reservation_suffix_shards(reservation)
        local_shards = self._reservation_local_shards(
            reservation.session_id,
            reservation.instance_index,
        )
        reserved_final = tuple(
            final_bytes + suffix_bytes
            for final_bytes, suffix_bytes in zip(
                reservation.final_shard_bytes,
                suffix_shards,
            )
        )
        extra = tuple(
            final_bytes - local_bytes
            for final_bytes, local_bytes in zip(
                reserved_final,
                local_shards,
            )
        )
        if any(value < 0 for value in extra):
            raise RuntimeError(
                f"request {reservation.request_id} reservation is smaller than "
                "its resident KV"
            )
        return extra

    def _reservation_local_shards(
        self,
        session_id: str,
        instance_index: int,
    ) -> tuple[int, ...]:
        session = self._sessions.get(session_id)
        if (
            session is not None
            and session.instance_index == instance_index
            and session.location != self.REMOTE_MEMORY
            and self._working_copy_uses_shard_truth(session)
        ):
            # A PARTIAL remote-read working copy carries the restored history
            # suffix in shard_bytes while context_tokens tracks input only.
            # Count physical bytes so materialization consumes its reservation.
            return tuple(int(value) for value in session.shard_bytes)
        return self._local_session_shards(session_id, instance_index)

    def _reservation_suffix_shards(
        self,
        reservation: KVCapacityReservation,
    ) -> tuple[int, ...]:
        if (
            reservation.suffix_history_tokens is None
            and reservation.suffix_start_layer is None
        ):
            return tuple(0 for _ in range(self.tp_degree))
        if (
            reservation.action != "remote-read"
            or reservation.suffix_history_tokens is None
            or reservation.suffix_start_layer is None
            or reservation.suffix_history_tokens <= 0
            or not 0 < reservation.suffix_start_layer < self.model.layers
        ):
            raise RuntimeError("HBM reservation has invalid suffix metadata")
        return kv_cache_shard_bytes_for_layer_range(
            self.model,
            reservation.suffix_history_tokens,
            self.tp_degree,
            layer_start=reservation.suffix_start_layer,
            layer_end=self.model.layers,
        )

    def _partial_remote_read_suffix(
        self,
        *,
        session: Optional[SessionKVState],
        history_tokens: int,
        action: Optional[str],
        target_instance_index: int,
    ) -> tuple[tuple[int, ...], Optional[int], Optional[int]]:
        """Return the separately reserved hot suffix for PARTIAL remote-read.

        LOCAL-base remote-read keeps its input-only footprint. A PARTIAL base
        must restore its missing suffix on a remote execution instance, so
        capacity decisions include those exact per-rank bytes alongside the
        input-token reservation.
        """
        zero = tuple(0 for _ in range(self.tp_degree))
        if action != "remote-read" or session is None:
            return zero, None, None
        if session.working_kind is None:
            base_location = session.location
            base_instance = session.instance_index
            suffix_start = session.resident_prefix_layers
        else:
            base_location = session.base_location
            base_instance = session.home_instance
            suffix_start = session.base_resident_prefix_layers
        if base_location != self.PARTIAL_HBM_REMOTE:
            return zero, None, None
        if base_instance is None or suffix_start is None:
            raise RuntimeError("PARTIAL remote-read base lost its home metadata")
        if target_instance_index == base_instance:
            return zero, None, None
        suffix = kv_cache_shard_bytes_for_layer_range(
            self.model,
            history_tokens,
            self.tp_degree,
            layer_start=suffix_start,
            layer_end=self.model.layers,
        )
        return suffix, history_tokens, suffix_start

    def _reservation_capacity_bytes(
        self,
        *,
        basis_shards: Sequence[int],
        suffix_shards: Sequence[int],
    ) -> tuple[int, ...]:
        if len(basis_shards) != self.tp_degree or len(suffix_shards) != self.tp_degree:
            raise RuntimeError("HBM reservation shard count does not match TP")
        return tuple(
            int(basis) + int(suffix)
            for basis, suffix in zip(basis_shards, suffix_shards)
        )

    def _reserved_bytes_by_tp_rank(
        self,
        instance_index: int,
        *,
        exclude_request_id: Optional[str] = None,
    ) -> tuple[int, ...]:
        reserved = [0 for _ in range(self.tp_degree)]
        for reservation in self._reservations.values():
            if (
                reservation.instance_index != instance_index
                or reservation.request_id == exclude_request_id
            ):
                continue
            reserved = [
                total + extra
                for total, extra in zip(
                    reserved,
                    self._reservation_extra_shards(reservation),
                )
            ]
        return tuple(reserved)

    def _effective_remaining_by_tp_rank(
        self,
        instance_index: int,
        *,
        exclude_request_id: Optional[str] = None,
    ) -> tuple[int, ...]:
        instance = self.topology.instance(instance_index)
        reserved = self._reserved_bytes_by_tp_rank(
            instance_index,
            exclude_request_id=exclude_request_id,
        )
        return tuple(
            self._rank_states[rank].remaining_bytes - reserved_bytes
            for rank, reserved_bytes in zip(instance.ranks, reserved)
        )

    def instance_effective_remaining_capacity_bytes(
        self,
        instance_index: int,
        *,
        exclude_request_id: Optional[str] = None,
    ) -> int:
        return sum(
            self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=exclude_request_id,
            )
        )

    def _instance_reclaimable_capacity_by_tp_rank(
        self,
        instance_index: int,
        *,
        exclude_session_id: Optional[str] = None,
        exclude_request_id: Optional[str] = None,
    ) -> tuple[int, ...]:
        """Return maximum free bytes after evicting every inactive session."""

        reclaimable = list(
            self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=exclude_request_id,
            )
        )
        for session in self._completed_resident_candidates(instance_index):
            if session.session_id == exclude_session_id:
                continue
            local_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.context_tokens,
                self.tp_degree,
                layer_start=0,
                layer_end=session.resident_prefix_layers,
            )
            reclaimable = [
                available + session_bytes
                for available, session_bytes in zip(reclaimable, local_shards)
            ]
        return tuple(reclaimable)

    def joint_reservation_context_tokens(
        self,
        *,
        action: Optional[str],
        history_tokens: int,
        input_tokens: int,
    ) -> int:
        """R1' 动作感知预约足迹的单源口径（feasible / reserve /
        eventually_feasible 三处共用，防口径分叉）。

        * remote-read：该 token 基数只承载新增量（input；decode 按实际
          进展因果增长、不预约，§3.1）。PARTIAL 基跨实例时，缺失历史后缀
          另按逐 rank 字节预约，不折进 token 基数；
        * 这样避免 F1 幻影预约（整份 history+input 预约虚增占用、污染
          其他请求的可行性判定与 hbm_remaining 视图）；
        * 其余动作（stay/copy/recompute）：终态工作副本 = history+input
          整份（驻留前缀复用部分经 ``final − local`` 自然扣除）。
        ``action=None`` 保留基底调用面（final = history+input 全量）。
        """
        if action is None or action in ("stay", "copy", "recompute"):
            return history_tokens + input_tokens
        if action == "remote-read":
            return input_tokens
        raise ValueError(f"unknown joint action for footprint: {action!r}")

    def request_hbm_feasible_instances(
        self,
        *,
        session_id: str,
        final_context_tokens: int,
        reservation_request_id: Optional[str] = None,
        action: Optional[str] = None,
    ) -> tuple[bool, ...]:
        """Report which instances can hold a request's final local KV state.

        Existing local or partially resident KV already consumes HBM on its
        resident instance, so only the missing bytes are required there.  A
        different instance must admit the complete final KV.  ``action``
        narrows the required basis per R1'（remote-read 的 token 基数为
        input 增量；PARTIAL 跨实例另计后缀物化字节）。

        joint R1'（去钉扎，总纲 §13.1/设计方案 §1.1）：PARTIAL 驻留不再
        构成实例亲和掩码——容量只影响所需字节（final − resident），不做
        候选集过滤；选中后的预约失败由准入事务（R2）按容量类延迟处理。
        """

        session = self._sessions.get(session_id)
        history_tokens = 0
        if session is not None:
            history_tokens = (
                session.base_history_tokens
                if session.working_kind is not None
                else session.context_tokens)
            if history_tokens > final_context_tokens:
                raise ValueError("request final context cannot shrink the KV cache")
        input_tokens = max(0, final_context_tokens - history_tokens)
        basis_context = self.joint_reservation_context_tokens(
            action=action,
            history_tokens=history_tokens,
            input_tokens=input_tokens,
        )
        basis_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            basis_context,
            self.tp_degree,
        )
        resident_instance_index: Optional[int] = None
        resident_shards = tuple(0 for _ in basis_shards)
        if session is not None:
            if session.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}:
                if session.instance_index is None:
                    raise RuntimeError("resident KV session has no instance")
                resident_instance_index = session.instance_index
                resident_shards = self._reservation_local_shards(
                    session_id,
                    resident_instance_index,
                )
            elif session.location != self.REMOTE_MEMORY:
                raise RuntimeError(f"unknown KV location: {session.location}")

        feasible: list[bool] = []
        for instance in self.topology.instances:
            suffix_shards, _suffix_history_tokens, _suffix_start_layer = (
                self._partial_remote_read_suffix(
                    session=session,
                    history_tokens=history_tokens,
                    action=action,
                    target_instance_index=instance.index,
                )
            )
            required_basis = self._reservation_capacity_bytes(
                basis_shards=basis_shards,
                suffix_shards=suffix_shards,
            )
            if instance.index == resident_instance_index:
                required = tuple(
                    basis_bytes - local_bytes
                    for basis_bytes, local_bytes in zip(
                        required_basis,
                        resident_shards,
                    )
                )
                if any(value < 0 for value in required):
                    if action == "remote-read":
                        # remote-read @ 驻留实例 = 不适用组合（适用性已
                        # 排除；本函数对全实例求值时允许到达）。驻留前缀
                        # 不参与复用，需求退化为整份 input 基数。
                        required = required_basis
                    else:
                        raise ValueError(
                            "request final KV is smaller than resident KV")
            else:
                required = required_basis
            reclaimable = self._instance_reclaimable_capacity_by_tp_rank(
                instance.index,
                exclude_session_id=session_id,
                exclude_request_id=reservation_request_id,
            )
            feasible.append(
                all(
                    available_bytes >= required_bytes
                    for available_bytes, required_bytes in zip(
                        reclaimable,
                        required,
                    )
                )
            )
        return tuple(feasible)

    def request_hbm_eventually_feasible_instances(
        self,
        *,
        session_id: str,
        final_context_tokens: int,
        action: Optional[str] = None,
    ) -> tuple[bool, ...]:
        """Report whether the request fits after all other sessions finish.

        This distinguishes a temporary active-session capacity conflict from a
        request whose final KV can never fit on an otherwise empty instance.
        R1'：动作感知口径（remote-read 的 input 基数 + PARTIAL 跨实例
        后缀）+ 去 PARTIAL 钉扎。可行性按每个目标实例分别计入其物理足迹。
        """

        session = self._sessions.get(session_id)
        history_tokens = 0
        if session is not None:
            history_tokens = (
                session.base_history_tokens
                if session.working_kind is not None
                else session.context_tokens)
            if history_tokens > final_context_tokens:
                raise ValueError("request final context cannot shrink the KV cache")
        input_tokens = max(0, final_context_tokens - history_tokens)
        basis_context = self.joint_reservation_context_tokens(
            action=action,
            history_tokens=history_tokens,
            input_tokens=input_tokens,
        )
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            basis_context,
            self.tp_degree,
        )

        feasible: list[bool] = []
        for instance in self.topology.instances:
            suffix_shards, _suffix_history_tokens, _suffix_start_layer = (
                self._partial_remote_read_suffix(
                    session=session,
                    history_tokens=history_tokens,
                    action=action,
                    target_instance_index=instance.index,
                )
            )
            required_basis = self._reservation_capacity_bytes(
                basis_shards=final_shards,
                suffix_shards=suffix_shards,
            )
            feasible.append(
                all(
                    final_bytes
                    <= self._rank_states[rank].capacity_bytes
                    - self._rank_states[rank].model_weight_bytes
                    for rank, final_bytes in zip(instance.ranks, required_basis)
                )
            )
        return tuple(feasible)

    def reserve_request_capacity(
        self,
        *,
        request_id: str,
        session_id: str,
        instance_index: int,
        final_context_tokens: int,
        action: Optional[str] = None,
    ) -> tuple[KVTransfer, ...]:
        """Commit final local-KV capacity before a request enters Prefill.

        R1'：预约量按动作足迹（``joint_reservation_context_tokens`` 单源）
        ——remote-read 以 input 为 token 基数，PARTIAL 跨实例另预约后缀；
        其余动作预约整份。容量不足抛
        ``KVCapacityError``（携带已提交逐出，准入事务 R2 清账）。选中
        不可行实例同样按容量类处理（重选时机由纪元门驱动，非合同违规）。
        """

        if request_id in self._reservations:
            raise ValueError(f"duplicate HBM reservation for request {request_id}")
        session = self._sessions.get(session_id)
        history_tokens = 0
        if session is not None:
            history_tokens = (
                session.base_history_tokens
                if session.working_kind is not None
                else session.context_tokens)
        basis_context = self.joint_reservation_context_tokens(
            action=action,
            history_tokens=history_tokens,
            input_tokens=max(0, final_context_tokens - history_tokens),
        )
        basis_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            basis_context,
            self.tp_degree,
        )
        suffix_shards, suffix_history_tokens, suffix_start_layer = (
            self._partial_remote_read_suffix(
                session=session,
                history_tokens=history_tokens,
                action=action,
                target_instance_index=instance_index,
            )
        )
        reserved_shards = self._reservation_capacity_bytes(
            basis_shards=basis_shards,
            suffix_shards=suffix_shards,
        )
        feasible = self.request_hbm_feasible_instances(
            session_id=session_id,
            final_context_tokens=final_context_tokens,
            reservation_request_id=request_id,
            action=action,
        )
        if not feasible[instance_index]:
            raise KVCapacityError(
                f"request {request_id} was reserved on an infeasible "
                f"instance {instance_index} (action={action!r})")
        local_shards = self._reservation_local_shards(
            session_id, instance_index)
        required = tuple(
            final_bytes - local_bytes
            for final_bytes, local_bytes in zip(reserved_shards, local_shards)
        )
        if any(value < 0 for value in required):
            raise ValueError("request final KV is smaller than resident KV")
        evictions = self._ensure_capacity(
            instance_index,
            required,
            phase="history",
            reason="request_admission_capacity",
            trigger_request_id=request_id,
            protected_session_id=session_id,
        )
        self._reservations[request_id] = KVCapacityReservation(
            request_id=request_id,
            session_id=session_id,
            instance_index=instance_index,
            final_context_tokens=basis_context,
            final_shard_bytes=basis_shards,
            action=action,
            suffix_history_tokens=suffix_history_tokens,
            suffix_start_layer=suffix_start_layer,
        )
        self._check_invariants_after_mutation(reservation_ids=(request_id,))
        if self._metrics_recorder is not None:
            # Admission reservation (doc sec.7.3): the committed-but-not-yet-
            # resident final-KV delta, anchored to the request Prefill start.
            for rank, value in zip(
                self.topology.instance(instance_index).ranks, required
            ):
                if not value:
                    continue
                self._metrics_recorder.record(
                    anchor_kind="prefill_start",
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=int(value),
                    cause="request_admission_reservation",
                )
        return evictions

    def move_request_capacity_reservation(
        self,
        *,
        request_id: str,
        target_instance_index: int,
    ) -> tuple[KVTransfer, ...]:
        """Move an active request's committed final-KV capacity to Decode.

        M3 钉死（2026-09-14）：joint 的 decode 固定 prefill 同实例
        （红线 #4），本方法在运行期恒为同实例 no-op；其下方沿袭基底的
        钉扎可行性检查 + raise 路径是未测试的遗留面——未来 P/D 拆分若
        沉默激活该路径会绕过 R1'/R2 的事务语义。此处显式 fail-closed：
        跨实例预约移动必须先经准入事务重设计，不得沉默复用。
        """

        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        reservation = self._reservations[request_id]
        if reservation.instance_index == target_instance_index:
            return ()
        raise RuntimeError(
            f"request {request_id} attempted a cross-instance decode "
            f"reservation move ({reservation.instance_index} -> "
            f"{target_instance_index}); joint pins Decode to the Prefill "
            "instance (red line #4) -- reroute through the admission "
            "transaction instead of the legacy move path")

    def release_request_capacity_reservation(self, request_id: str) -> None:
        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        reservation = self._reservations[request_id]
        if any(self._reservation_extra_shards(reservation)):
            raise RuntimeError(
                f"request {request_id} released HBM reservation before final KV allocation"
            )
        del self._reservations[request_id]
        self._check_invariants_after_mutation(reservation_ids=(request_id,))

    def decode_hbm_feasible_instances(
        self,
        *,
        session_id: str,
        final_context_tokens: int,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[bool, ...]:
        """Report whether each instance can admit the request's final KV.

        The Prefill instance already owns the current KV shards, so keeping
        Decode local requires only the Decode-growth delta.  Moving Decode to
        another instance requires the complete final KV shards.  Capacity
        includes bytes reclaimable from completed inactive sessions because
        ``_ensure_capacity`` can evict those sessions before admission.
        """

        if session_id not in self._sessions:
            raise KeyError(f"unknown KV session: {session_id}")
        session = self._sessions[session_id]
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("Decode feasibility requires local Prefill KV")
        return self.request_hbm_feasible_instances(
            session_id=session_id,
            final_context_tokens=final_context_tokens,
            reservation_request_id=reservation_request_id,
        )

    def session_snapshot(self, session_id: str) -> SessionKVSnapshot:
        if session_id not in self._sessions:
            raise KeyError(f"unknown KV session: {session_id}")
        state = self._sessions[session_id]
        if self._working_copy_uses_shard_truth(state):
            # D1 两口径分离（2026-09-17）：混合形态 primary 驻留取
            # shard_bytes 物理真值（context_tokens 为增量口径）；池侧
            # 旧 backing 为陈旧副本，账面忽略（§4.3.1）。
            local_shard_bytes = tuple(state.shard_bytes)
            remote_shard_bytes = tuple(0 for _ in state.shard_bytes)
        else:
            local_shard_bytes = kv_cache_shard_bytes_for_layer_range(
                self.model,
                state.context_tokens,
                self.tp_degree,
                layer_start=0,
                layer_end=state.resident_prefix_layers,
            )
            remote_shard_bytes = kv_cache_shard_bytes_for_layer_range(
                self.model,
                state.context_tokens,
                self.tp_degree,
                layer_start=state.resident_prefix_layers,
                layer_end=self.model.layers,
            )
        rank_bytes: tuple[tuple[int, int], ...] = ()
        if state.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}:
            if state.instance_index is None:
                raise RuntimeError("local KV session has no instance")
            instance = self.topology.instance(state.instance_index)
            rank_bytes = tuple(zip(instance.ranks, local_shard_bytes))
        return SessionKVSnapshot(
            session_id=state.session_id,
            location=state.location,
            instance_index=state.instance_index,
            context_tokens=state.context_tokens,
            total_bytes=state.total_bytes,
            shard_bytes=state.shard_bytes,
            resident_prefix_layers=state.resident_prefix_layers,
            local_bytes=sum(local_shard_bytes),
            remote_bytes=sum(remote_shard_bytes),
            local_shard_bytes=local_shard_bytes,
            remote_shard_bytes=remote_shard_bytes,
            rank_bytes=rank_bytes,
            last_completion_ns=state.last_completion_ns,
            active=state.active,
            home_instance=state.home_instance,
            working_kind=state.working_kind,
            working_instance_index=state.working_instance_index,
        )

    def session_snapshots(self) -> tuple[SessionKVSnapshot, ...]:
        return tuple(self.session_snapshot(session_id) for session_id in self.session_ids)

    def nearest_edge(self, rank: int) -> int:
        return nearest_edge_rank(
            self.topology.hardware,
            rank,
            self.edge_ranks,
        )

    # ------------------------------------------------------------------
    # Incremental invariant ledger.  The complete checker below remains the
    # source of truth for construction, strict mode, and terminal audits.
    # Normal mutation paths refresh only the changed session(s), reservation(s)
    # and their TP ranks instead of rebuilding every session's layer-range
    # shard vector after every generated token.
    # ------------------------------------------------------------------

    def _initialize_incremental_invariants(self) -> None:
        self._invariant_expected_kv_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_expected_reserved_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_session_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._invariant_base_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._invariant_reservation_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._invariant_reservation_sessions: dict[str, str] = {}
        self._invariant_reservations_by_session: dict[str, set[str]] = {}
        self._refresh_incremental_sessions(self._sessions)
        self._refresh_incremental_reservations(self._reservations)

    def _working_copy_uses_shard_truth(self, session: SessionKVState) -> bool:
        """混合形态（remote-read×PARTIAL 基，2026-09-17 用户裁定）判定。

        D1 两口径分离：该形态下 primary ``shard_bytes`` 是物理真值（准入相
        池恢复的后缀 S ＋ 增量），而 ``context_tokens`` 是增量 token 口径
        （0 起步）——审计/快照派生期望驻留字节时必须取 shard_bytes，否则
        exec 侧驻留被低估（不变量误报泄漏）。LOCAL 基 remote-read 两口径
        恒等（shard_bytes == kv(context_tokens)@全层），无需特判；copy/
        recompute 的 context 覆盖完整上下文，亦恒等。"""
        return (
            session.working_kind == "remote-read"
            and session.base_location == self.PARTIAL_HBM_REMOTE
            and 0 < session.base_resident_prefix_layers < self.model.layers
        )

    def _incremental_session_contribution(
        self, session: SessionKVState
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        if (
            session.location not in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            or session.instance_index is None
            or len(session.shard_bytes) != self.tp_degree
        ):
            return None
        if session.location == self.LOCAL_HBM:
            if session.resident_prefix_layers != self.model.layers:
                return None
        elif not 0 < session.resident_prefix_layers < self.model.layers:
            return None
        if self._working_copy_uses_shard_truth(session):
            local_shards = tuple(int(value) for value in session.shard_bytes)
        else:
            local_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.context_tokens,
                self.tp_degree,
                layer_start=0,
                layer_end=session.resident_prefix_layers,
            )
        return session.instance_index, tuple(int(value) for value in local_shards)

    def _incremental_base_contribution(
        self, session: SessionKVState
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        """跨实例执行期间 home 侧基础历史的守恒贡献（joint §2.2）。

        工作副本活跃时 primary 描述执行端；home 侧的基础驻留前缀字节
        仍在该实例的 rank 台账上，必须计入期望值，否则逐 session 守恒
        审计把基础副本误判为泄漏。基础 REMOTE（池 backing）无本地字节。
        """
        if session.working_kind is None or session.home_instance is None:
            return None
        if session.base_location not in {
            self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE,
        }:
            return None
        journal = session.copy_handoff
        if journal is not None:
            # C13：copy 的 home 侧残量以交接账本为准（逐块释放不等轮末；
            # 全部释放后贡献为零——不重复释放已交接源块）。
            shards = journal.home_shards()
            if not any(shards):
                return None
            return session.home_instance, shards
        prefix = session.base_resident_prefix_layers
        if prefix <= 0:
            return None
        local_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.base_history_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=prefix,
        )
        return session.home_instance, tuple(int(value) for value in local_shards)

    def _incremental_reservation_contribution(
        self, reservation: KVCapacityReservation
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        if len(reservation.final_shard_bytes) != self.tp_degree:
            return None
        extra = self._reservation_extra_shards(reservation)
        if len(extra) != self.tp_degree:
            return None
        return reservation.instance_index, tuple(int(value) for value in extra)

    def _apply_incremental_contribution(
        self,
        expected_by_rank: dict[int, int],
        contribution: Optional[tuple[int, tuple[int, ...]]],
        multiplier: int,
    ) -> set[int]:
        if contribution is None:
            return set()
        instance_index, shards = contribution
        instance = self.topology.instance(instance_index)
        affected_ranks = set(instance.ranks)
        for rank, value in zip(instance.ranks, shards):
            expected_by_rank[rank] += multiplier * value
        return affected_ranks

    def _refresh_incremental_sessions(self, session_ids: Iterable[str]) -> set[int]:
        affected_ranks: set[int] = set()
        for session_id in dict.fromkeys(session_ids):
            old = self._invariant_session_contributions.pop(session_id, None)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_kv_by_rank, old, -1
                )
            )
            old_base = self._invariant_base_contributions.pop(
                session_id, None)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_kv_by_rank, old_base, -1
                )
            )
            session = self._sessions.get(session_id)
            if session is None:
                continue
            new = self._incremental_session_contribution(session)
            self._invariant_session_contributions[session_id] = new
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_kv_by_rank, new, 1
                )
            )
            new_base = self._incremental_base_contribution(session)
            self._invariant_base_contributions[session_id] = new_base
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_kv_by_rank, new_base, 1
                )
            )
        return affected_ranks

    def _refresh_incremental_reservations(
        self, reservation_ids: Iterable[str]
    ) -> set[int]:
        affected_ranks: set[int] = set()
        for request_id in dict.fromkeys(reservation_ids):
            old = self._invariant_reservation_contributions.pop(request_id, None)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_reserved_by_rank, old, -1
                )
            )
            old_session_id = self._invariant_reservation_sessions.pop(
                request_id, None
            )
            if old_session_id is not None:
                indexed = self._invariant_reservations_by_session.get(old_session_id)
                if indexed is not None:
                    indexed.discard(request_id)
                    if not indexed:
                        del self._invariant_reservations_by_session[old_session_id]

            reservation = self._reservations.get(request_id)
            if reservation is None:
                continue
            new = self._incremental_reservation_contribution(reservation)
            self._invariant_reservation_contributions[request_id] = new
            self._invariant_reservation_sessions[request_id] = reservation.session_id
            self._invariant_reservations_by_session.setdefault(
                reservation.session_id, set()
            ).add(request_id)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_reserved_by_rank, new, 1
                )
            )
        return affected_ranks

    def _check_incremental_session_invariants(self, session: SessionKVState) -> None:
        if session.location not in {
            self.LOCAL_HBM,
            self.PARTIAL_HBM_REMOTE,
            self.REMOTE_MEMORY,
        }:
            raise RuntimeError(f"invalid KV location for {session.session_id}")
        if sum(session.shard_bytes) != session.total_bytes:
            raise RuntimeError(
                f"KV shards do not preserve total for {session.session_id}"
            )
        if session.location == self.REMOTE_MEMORY:
            if session.instance_index is not None:
                raise RuntimeError("remote KV session retained a local instance")
            if session.resident_prefix_layers != 0:
                raise RuntimeError("remote KV session retained resident layers")
            return
        if session.location == self.LOCAL_HBM:
            if session.resident_prefix_layers != self.model.layers:
                raise RuntimeError("fully local KV session is missing layers")
        elif not 0 < session.resident_prefix_layers < self.model.layers:
            raise RuntimeError("partial KV session has an invalid prefix length")
        if session.instance_index is None:
            raise RuntimeError("local KV session has no instance")
        instance = self.topology.instance(session.instance_index)
        if len(session.shard_bytes) != instance.size:
            raise RuntimeError("KV shard count does not match TP instance")

    def _check_incremental_reservation_invariants(
        self, request_id: str, reservation: KVCapacityReservation
    ) -> None:
        if reservation.request_id != request_id:
            raise RuntimeError("HBM reservation key does not match request ID")
        if len(reservation.final_shard_bytes) != self.tp_degree:
            raise RuntimeError("HBM reservation shard count does not match TP")
        if reservation.action not in {
            None, "stay", "copy", "recompute", "remote-read",
        }:
            raise RuntimeError("HBM reservation has an unknown action")
        expected_final = kv_cache_shard_bytes_for_tokens(
            self.model,
            reservation.final_context_tokens,
            self.tp_degree,
        )
        if reservation.final_shard_bytes != expected_final:
            raise RuntimeError("HBM reservation final KV metadata is inconsistent")
        self._reservation_suffix_shards(reservation)

    def _check_invariants_after_mutation(
        self,
        *,
        session_ids: Iterable[str] = (),
        reservation_ids: Iterable[str] = (),
    ) -> None:
        """Validate the ledger entries touched by a completed KV mutation."""

        changed_sessions = tuple(dict.fromkeys(session_ids))
        changed_reservation_ids = set(reservation_ids)
        for session_id in changed_sessions:
            changed_reservation_ids.update(
                self._invariant_reservations_by_session.get(session_id, ())
            )
        changed_reservations = tuple(sorted(changed_reservation_ids))
        affected_ranks = self._refresh_incremental_sessions(changed_sessions)
        affected_ranks.update(
            self._refresh_incremental_reservations(changed_reservations)
        )

        for rank in sorted(affected_ranks):
            state = self._rank_states[rank]
            if state.used_bytes != state.model_weight_bytes + state.kv_cache_bytes:
                raise RuntimeError(f"rank {rank} HBM used-byte invariant failed")
            if state.remaining_bytes != state.capacity_bytes - state.used_bytes:
                raise RuntimeError(f"rank {rank} HBM remaining-byte invariant failed")
            if state.kv_cache_bytes < 0 or state.remaining_bytes < 0:
                raise RuntimeError(f"rank {rank} HBM capacity was exceeded")
        for session_id in changed_sessions:
            session = self._sessions.get(session_id)
            if session is not None:
                self._check_incremental_session_invariants(session)
        for request_id in changed_reservations:
            reservation = self._reservations.get(request_id)
            if reservation is not None:
                self._check_incremental_reservation_invariants(request_id, reservation)

        affected_instances = {
            self._rank_states[rank].instance_index for rank in affected_ranks
        }
        for instance_index in sorted(affected_instances):
            instance = self.topology.instance(instance_index)
            for rank in instance.ranks:
                state = self._rank_states[rank]
                expected_kv = self._invariant_expected_kv_by_rank[rank]
                if state.kv_cache_bytes != expected_kv:
                    raise RuntimeError(
                        f"rank {rank} KV accounting mismatch: "
                        f"state={state.kv_cache_bytes}, expected={expected_kv}"
                    )
            effective = tuple(
                self._rank_states[rank].remaining_bytes
                - self._invariant_expected_reserved_by_rank[rank]
                for rank in instance.ranks
            )
            if any(value < 0 for value in effective):
                raise RuntimeError(
                    f"instance {instance.index} HBM reservations exceed free capacity"
                )

        if self._strict_kv_invariants:
            self._check_invariants()

    def _check_invariants(self) -> None:
        expected_kv = dict.fromkeys(self._rank_states, 0)
        for session in self._sessions.values():
            if session.location not in {
                self.LOCAL_HBM,
                self.PARTIAL_HBM_REMOTE,
                self.REMOTE_MEMORY,
            }:
                raise RuntimeError(f"invalid KV location for {session.session_id}")
            if sum(session.shard_bytes) != session.total_bytes:
                raise RuntimeError(
                    f"KV shards do not preserve total for {session.session_id}"
                )
            if session.location == self.REMOTE_MEMORY:
                if session.instance_index is not None:
                    raise RuntimeError("remote KV session retained a local instance")
                if session.resident_prefix_layers != 0:
                    raise RuntimeError("remote KV session retained resident layers")
                continue
            if session.location == self.LOCAL_HBM:
                if session.resident_prefix_layers != self.model.layers:
                    raise RuntimeError("fully local KV session is missing layers")
            elif not (
                0 < session.resident_prefix_layers < self.model.layers
            ):
                raise RuntimeError("partial KV session has an invalid prefix length")
            if session.instance_index is None:
                raise RuntimeError("local KV session has no instance")
            instance = self.topology.instance(session.instance_index)
            if len(session.shard_bytes) != instance.size:
                raise RuntimeError("KV shard count does not match TP instance")
            if self._working_copy_uses_shard_truth(session):
                # D1 两口径分离（2026-09-17）：混合形态 primary 驻留取
                # shard_bytes 物理真值（context_tokens 为增量口径）。
                local_shards = tuple(session.shard_bytes)
            else:
                local_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    session.context_tokens,
                    self.tp_degree,
                    layer_start=0,
                    layer_end=session.resident_prefix_layers,
                )
            for rank, shard_bytes in zip(instance.ranks, local_shards):
                expected_kv[rank] += shard_bytes
            # 跨实例执行期间 home 侧基础驻留（与增量通道
            # _incremental_base_contribution 同式）：全量审计补齐——旧代码
            # 无 strict+working 可达组合掩盖了缺口，merge v2/混合形态改造
            # 后 strict 档必须与增量通道一致（2026-09-17）。
            base_contribution = self._incremental_base_contribution(session)
            if base_contribution is not None:
                base_instance = self.topology.instance(base_contribution[0])
                for rank, shard_bytes in zip(
                    base_instance.ranks, base_contribution[1]
                ):
                    expected_kv[rank] += shard_bytes

        for rank, state in self._rank_states.items():
            if state.kv_cache_bytes != expected_kv[rank]:
                raise RuntimeError(
                    f"rank {rank} KV accounting mismatch: "
                    f"state={state.kv_cache_bytes}, expected={expected_kv[rank]}"
                )
            if state.used_bytes != state.model_weight_bytes + state.kv_cache_bytes:
                raise RuntimeError(f"rank {rank} HBM used-byte invariant failed")
            if state.remaining_bytes != state.capacity_bytes - state.used_bytes:
                raise RuntimeError(f"rank {rank} HBM remaining-byte invariant failed")
            if state.kv_cache_bytes < 0 or state.remaining_bytes < 0:
                raise RuntimeError(f"rank {rank} HBM capacity was exceeded")

        for request_id, reservation in self._reservations.items():
            if reservation.request_id != request_id:
                raise RuntimeError("HBM reservation key does not match request ID")
            if len(reservation.final_shard_bytes) != self.tp_degree:
                raise RuntimeError("HBM reservation shard count does not match TP")
            if reservation.action not in {
                None, "stay", "copy", "recompute", "remote-read",
            }:
                raise RuntimeError("HBM reservation has an unknown action")
            expected_final = kv_cache_shard_bytes_for_tokens(
                self.model,
                reservation.final_context_tokens,
                self.tp_degree,
            )
            if reservation.final_shard_bytes != expected_final:
                raise RuntimeError("HBM reservation final KV metadata is inconsistent")
            self._reservation_suffix_shards(reservation)
        for instance in self.topology.instances:
            effective = self._effective_remaining_by_tp_rank(instance.index)
            if any(value < 0 for value in effective):
                raise RuntimeError(
                    f"instance {instance.index} HBM reservations exceed free capacity"
                )

    def assert_final_state(self) -> None:
        """Run the complete invariant audit at the planner's terminal boundary."""

        self._check_invariants()

    def _add_local_shards(
        self,
        instance_index: int,
        shard_bytes: Sequence[int],
    ) -> None:
        instance = self.topology.instance(instance_index)
        if len(shard_bytes) != instance.size:
            raise ValueError("KV shard count must match target instance size")
        for rank, added_bytes in zip(instance.ranks, shard_bytes):
            if added_bytes < 0:
                raise ValueError("KV shard increment must be non-negative")
            if self._rank_states[rank].remaining_bytes < added_bytes:
                raise KVCapacityError(
                    f"rank {rank} has insufficient local HBM: "
                    f"needs {added_bytes}, "
                    f"remaining {self._rank_states[rank].remaining_bytes}")
        for rank, added_bytes in zip(instance.ranks, shard_bytes):
            self._rank_states[rank].kv_cache_bytes += added_bytes

    def _remove_local_shards(
        self,
        instance_index: int,
        shard_bytes: Sequence[int],
    ) -> None:
        instance = self.topology.instance(instance_index)
        if len(shard_bytes) != instance.size:
            raise ValueError("KV shard count must match source instance size")
        for rank, removed_bytes in zip(instance.ranks, shard_bytes):
            if removed_bytes < 0:
                raise ValueError("KV shard decrement must be non-negative")
            if self._rank_states[rank].kv_cache_bytes < removed_bytes:
                raise RuntimeError(f"rank {rank} KV usage would become negative")
        for rank, removed_bytes in zip(instance.ranks, shard_bytes):
            self._rank_states[rank].kv_cache_bytes -= removed_bytes

    # ------------------------------------------------------------------
    # Read-only metrics observation helpers (doc sec.7).  Every method here
    # is a no-op unless a recorder was installed at construction time, and
    # none of them feeds back into manager decisions.  Resident KV is tracked
    # as per-session parts that carry their own token count and layer range;
    # because kv_cache_shard_bytes_for_layer_range is exactly linear in
    # tokens, suffix-half evictions and restores remove precisely the
    # distribution an earlier add recorded under the same allocation key
    # (doc sec.7.4/7.6).
    # ------------------------------------------------------------------

    def _metrics_find_reservation(
        self,
        session_id: str,
        instance_index: int,
    ) -> Optional[KVCapacityReservation]:
        for reservation in self._reservations.values():
            if (
                reservation.session_id == session_id
                and reservation.instance_index == instance_index
            ):
                return reservation
        return None

    def _metrics_add_segment(
        self,
        session_id: str,
        instance_index: int,
        tokens: int,
        layer_start: int,
        layer_end: int,
        shards: Sequence[int],
        *,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        counter = self._metrics_segment_counters.get(session_id, 0)
        self._metrics_segment_counters[session_id] = counter + 1
        segment_id = f"seg{counter}"
        self._metrics_parts.setdefault(session_id, []).append(
            {
                "segment_id": segment_id,
                "instance_index": instance_index,
                "tokens": int(tokens),
                "layer_start": int(layer_start),
                "layer_end": int(layer_end),
                "shards": tuple(int(value) for value in shards),
            }
        )
        instance = self.topology.instance(instance_index)
        # While the request's final-KV reservation is active, every resident
        # byte added for its session commits reservation capacity, so the
        # reserved ledger shrinks first (keeping physical <= committed per
        # rank after every single delta) and the resident add follows.
        reservation = self._metrics_find_reservation(session_id, instance_index)
        if reservation is not None:
            for rank, value in zip(instance.ranks, shards):
                if not value:
                    continue
                recorder.record(
                    anchor_kind=anchor_kind,
                    request_id=reservation.request_id,
                    session_id=session_id,
                    rank=rank,
                    allocation_key=f"reservation:{reservation.request_id}",
                    reserved_kv_delta_bytes=-int(value),
                    cause=f"{cause}_reservation_commit",
                )
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            recorder.record(
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                allocation_key=f"resident:{session_id}:{segment_id}",
                resident_kv_delta_bytes=int(value),
                cause=cause,
            )

    def _metrics_remove_part(
        self,
        session_id: str,
        part: dict[str, Any],
        shards: Sequence[int],
        *,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        instance = self.topology.instance(part["instance_index"])
        key = f"resident:{session_id}:{part['segment_id']}"
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            recorder.record(
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                allocation_key=key,
                resident_kv_delta_bytes=-int(value),
                cause=cause,
            )

    def _metrics_remove_session_parts(
        self,
        session_id: str,
        *,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Full-session eviction: remove every resident part under its key."""

        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.pop(session_id, [])
        for part in parts:
            self._metrics_remove_part(
                session_id,
                part,
                part["shards"],
                anchor_kind=anchor_kind,
                request_id=request_id,
                cause=cause,
            )

    def _metrics_suffix_evict_parts(
        self,
        session_id: str,
        suffix_start: int,
        *,
        anchor_kind: str,
        request_id: str,
        cause: str,
        instance_index: Optional[int] = None,
    ) -> None:
        """Suffix-half eviction: clamp every part to its intersection with
        the retained prefix layers, removing the exact per-part suffix
        distribution (never a merged guess, doc sec.7.4/9.4).

        ``instance_index``（自查 B，2026-09-15）：只截该实例上的 parts
        ——跨实例工作副本会话（base@home + working@exec 同 session_id）
        降级 base 时必须只动 home 端 parts；缺省 None = 全部（基底
        _evict_suffix 语义：victim 会话 parts 全在同一实例，行为不变）。"""

        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.get(session_id, [])
        kept_parts: list[dict[str, Any]] = []
        for part in parts:
            if (instance_index is not None
                    and part["instance_index"] != instance_index):
                kept_parts.append(part)
                continue
            new_layer_end = min(part["layer_end"], suffix_start)
            if new_layer_end <= part["layer_start"]:
                removed = part["shards"]
                new_shards = None
            else:
                new_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    part["tokens"],
                    self.tp_degree,
                    layer_start=part["layer_start"],
                    layer_end=new_layer_end,
                )
                removed = tuple(
                    old - new for old, new in zip(part["shards"], new_shards)
                )
            if any(removed):
                self._metrics_remove_part(
                    session_id,
                    part,
                    removed,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    cause=cause,
                )
            if new_shards is not None:
                part["layer_end"] = new_layer_end
                part["shards"] = new_shards
                kept_parts.append(part)
        if kept_parts:
            self._metrics_parts[session_id] = kept_parts
        else:
            self._metrics_parts.pop(session_id, None)

    def _metrics_prefix_release_parts(
        self,
        session_id: str,
        layer_end: int,
        *,
        instance_index: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """C13（2026-09-22）：home 侧**前缀**段释放——_metrics_suffix_
        evict_parts 的前缀镜像。

        copy 交接按消费顺序自层 0 向上释放（transformer 深度序），home
        侧保留的是高层段 [layer_end, L)：本方法把该实例上每个 part 截
        到其与 [layer_end, L) 的交（移除 [part_start, layer_end) 精确
        分布，永不合并猜测——与 suffix 版同纪律）。
        """
        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.get(session_id, [])
        kept_parts: list[dict[str, Any]] = []
        for part in parts:
            if part["instance_index"] != instance_index:
                kept_parts.append(part)
                continue
            new_layer_start = max(part["layer_start"], layer_end)
            if new_layer_start >= part["layer_end"]:
                removed = part["shards"]
                new_shards = None
            else:
                new_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    part["tokens"],
                    self.tp_degree,
                    layer_start=new_layer_start,
                    layer_end=part["layer_end"],
                )
                removed = tuple(
                    old - new for old, new in zip(part["shards"], new_shards)
                )
            if any(removed):
                self._metrics_remove_part(
                    session_id,
                    part,
                    removed,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    cause=cause,
                )
            if new_shards is not None:
                part["layer_start"] = new_layer_start
                part["shards"] = new_shards
                kept_parts.append(part)
        if kept_parts:
            self._metrics_parts[session_id] = kept_parts
        else:
            self._metrics_parts.pop(session_id, None)

    def _metrics_move_session_parts(
        self,
        session_id: str,
        target_instance_index: int,
        tokens: int,
        resident_layers: int,
        total_shards: Sequence[int],
        *,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Mirror a NoC migration: the target add is recorded as one
        consolidated segment (a fresh allocation key carries the full
        magnitude, so the chiplet projection stays exactly removable), then
        each original part is removed from the source ranks under its own
        key (doc sec.7.4/7.6).  Target add precedes source release, matching
        the runtime KV-transition ordering."""

        if self._metrics_recorder is None:
            return
        self._metrics_add_segment(
            session_id,
            target_instance_index,
            tokens,
            0,
            resident_layers,
            total_shards,
            anchor_kind=anchor_kind,
            request_id=request_id,
            cause=f"{cause}_target_add",
        )
        parts = self._metrics_parts.get(session_id, [])
        consolidated = parts[-1]
        for part in parts[:-1]:
            self._metrics_remove_part(
                session_id,
                part,
                part["shards"],
                anchor_kind=anchor_kind,
                request_id=request_id,
                cause=f"{cause}_source_remove",
            )
        self._metrics_parts[session_id] = [consolidated]

    def _metrics_reconsolidate_parts(
        self,
        session_id: str,
        instance_index: int,
        tokens: int,
        resident_layers: int,
        new_shards: Sequence[int],
        *,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """History truncation mirrors the manager's own uniform per-token
        accounting exactly: every old part is removed under its key, then the
        remaining resident prefix is re-added as one fresh segment carrying
        the manager's own post-truncation shard values."""

        if self._metrics_recorder is None:
            return
        self._metrics_remove_session_parts(
            session_id,
            anchor_kind=anchor_kind,
            request_id=request_id,
            cause=f"{cause}_remove",
        )
        self._metrics_add_segment(
            session_id,
            instance_index,
            tokens,
            0,
            resident_layers,
            new_shards,
            anchor_kind=anchor_kind,
            request_id=request_id,
            cause=f"{cause}_add",
        )

    def _local_hit_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
    ) -> KVTransfer:
        return KVTransfer(
            kind="local_hit",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=session.instance_index,
            target_instance_index=session.instance_index,
            total_bytes=session.total_bytes,
            shards=(),
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=self.model.layers,
            resident_prefix_layers_before=self.model.layers,
            resident_prefix_layers_after=self.model.layers,
        )

    def _noc_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        source_instance_index: int,
        target_instance_index: int,
        layer_start: int,
        layer_end: int,
        handoff_chunk: Optional[int] = None,
    ) -> KVTransfer:
        """R16-1（2026-09-15）：层区间 NoC 迁移原语。

        守卫改区间包含判定：``0 <= layer_start < layer_end <=
        resident_prefix_layers``——驻留恒为前缀形态（不变量 ：2166-2172），
        区间 ⊆ [0, resident_prefix) ⟺ 所传字节物理在场（旧全驻留守卫是
        ``layer_end == L`` 特例，语义严格变宽而非放松）。字节源按层区间
        从 context_tokens 派生（隐式依赖准入不变量的
        ``context_tokens == history_tokens``，登记 PROVENANCE P3-b），
        **不读** ``session.shard_bytes``（全量声明口径，PARTIAL 下是幻影
        字节源——D2-1 根因）。``resident_prefix_layers_before/after`` 沿
        本仓区间传输惯例 = layer_start/layer_end（:2800/:3871/:3924 同例）。
        """
        if not (
            0 <= layer_start < layer_end <= session.resident_prefix_layers
        ):
            raise RuntimeError(
                f"NoC migration layer range [{layer_start}, {layer_end}) is "
                f"not contained in the resident prefix "
                f"[0, {session.resident_prefix_layers}) "
                f"(session {session.session_id})")
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        transfer_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.context_tokens,
            self.tp_degree,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=shard_bytes,
                noc_path=deterministic_xy_route(
                    self.topology.hardware, source_rank, target_rank
                ),
                layer_start=layer_start,
                layer_end=layer_end,
            )
            for source_rank, target_rank, shard_bytes in zip(
                source.ranks, target.ranks, transfer_shards
            )
            if shard_bytes > 0
        )
        return KVTransfer(
            kind="noc_migrate",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=target_instance_index,
            total_bytes=sum(transfer_shards),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=layer_start,
            resident_prefix_layers_after=layer_end,
            handoff_chunk=handoff_chunk,
        )

    def _remote_load_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        target_instance_index: int,
        layer_start: int,
        layer_end: int,
        restore_group: Optional[int] = None,
    ) -> KVTransfer:
        target = self.topology.instance(target_instance_index)
        transfer_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.context_tokens,
            self.tp_degree,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        shards = []
        for target_rank, shard_bytes in zip(target.ranks, transfer_shards):
            if shard_bytes == 0:
                continue
            edge = self.nearest_edge(target_rank)
            shards.append(
                KVTransferShard(
                    source_rank=edge,
                    target_rank=target_rank,
                    edge_rank=edge,
                    bytes=shard_bytes,
                    noc_path=deterministic_xy_route(
                        self.topology.hardware, edge, target_rank
                    ),
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
            )
        return KVTransfer(
            kind="remote_load",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=None,
            target_instance_index=target_instance_index,
            total_bytes=sum(transfer_shards),
            shards=tuple(shards),
            model_layers=self.model.layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=layer_start,
            resident_prefix_layers_after=layer_end,
            restore_group=restore_group,
        )

    def _plan_suffix_restore_transfers(
        self,
        *,
        session: SessionKVState,
        trigger_request_id: str,
        target_instance_index: int,
        suffix_start: int,
        history_tokens: int,
        reason: str,
    ) -> tuple[KVTransfer, ...]:
        """C15：后缀 [suffix_start, L) 的逐组恢复发射计划 + 区间账本。

        按 plan_suffix_restore_groups 切分（消费顺序 = 层自低向高；
        后缀 ≤ RESTORE_GROUP_LAYERS 时单组 = 旧单笔口径回归锚——
        transfers 数量/区间/字节不变，仅多 restore_group 标记）。账本
        科目 ``rid#restore`` 的 issue 事件在此落账（complete 于 prefill
        drain/merge 兜底；守恒断言见 RestoreGroupJournal）。逐组物理
        到达由构图器旁挂恢复支链承载（GB _suffix_restore_arms），列车
        体按层段就绪门控消费——替换"整列车等整段后缀"的保守门。
        """
        group_ranges = plan_suffix_restore_groups(
            suffix_start, self.model.layers)
        if not group_ranges:
            return ()
        transfers: list[KVTransfer] = []
        entries: list[RestoreGroupEntry] = []
        for group_index, (layer_start, layer_end) in enumerate(group_ranges):
            shard_bytes = kv_cache_shard_bytes_for_layer_range(
                self.model, history_tokens, self.tp_degree,
                layer_start=layer_start, layer_end=layer_end)
            transfers.append(self._remote_load_transfer(
                phase="history",
                reason=reason,
                session=session,
                trigger_request_id=trigger_request_id,
                target_instance_index=target_instance_index,
                layer_start=layer_start,
                layer_end=layer_end,
                restore_group=group_index,
            ))
            entries.append(RestoreGroupEntry(
                index=group_index,
                layer_start=layer_start,
                layer_end=layer_end,
                shard_bytes=shard_bytes,
            ))
        journal = RestoreGroupJournal(
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            target_instance_index=target_instance_index,
            suffix_start=suffix_start,
            entries=tuple(entries),
        )
        journal.assert_conservation()
        journal.events.append({
            "subject": f"{trigger_request_id}#restore",
            "event": "issue",
            "session_id": session.session_id,
            "target_instance": target_instance_index,
            "suffix_start": suffix_start,
            "groups": [
                {"index": e.index, "layer_start": e.layer_start,
                 "layer_end": e.layer_end, "bytes": e.total_bytes}
                for e in entries],
            "issued_bytes": sum(journal.issued_bytes_by_rank()),
        })
        if session.restore_journal is not None and not (
                session.restore_journal.all_consumed()):
            raise RuntimeError(
                f"session {session.session_id} re-issued a suffix restore "
                "while the previous journal is unsettled (double issue)")
        session.restore_journal = journal
        self.restore_events.extend(journal.events)
        return tuple(transfers)

    def _remote_store_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        layer_start: int,
        layer_end: int,
        resident_prefix_layers_after: int,
    ) -> KVTransfer:
        if session.instance_index is None:
            raise RuntimeError("cannot store a non-local KV session")
        source_instance_index = session.instance_index
        source = self.topology.instance(source_instance_index)
        transfer_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.context_tokens,
            self.tp_degree,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        shards = []
        for source_rank, shard_bytes in zip(source.ranks, transfer_shards):
            if shard_bytes == 0:
                continue
            edge = self.nearest_edge(source_rank)
            shards.append(
                KVTransferShard(
                    source_rank=source_rank,
                    target_rank=edge,
                    edge_rank=edge,
                    bytes=shard_bytes,
                    noc_path=deterministic_xy_route(
                        self.topology.hardware, source_rank, edge
                    ),
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
            )
        return KVTransfer(
            kind="remote_store",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=None,
            total_bytes=sum(transfer_shards),
            shards=tuple(shards),
            model_layers=self.model.layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=session.resident_prefix_layers,
            resident_prefix_layers_after=resident_prefix_layers_after,
        )

    @staticmethod
    def _fifo_sort(candidates: list[SessionKVState]) -> list[SessionKVState]:
        candidates.sort(
            key=lambda session: (
                int(session.last_completion_ns),
                session.session_id,
            )
        )
        return candidates

    @staticmethod
    def _eviction_class(session: SessionKVState) -> str:
        # 类别判定唯一来源 = joint.eviction_priority（SH fallback 裁定：
        # 仅显式 "tool" 归工具类，human/None 及未知值按 human 优先级；
        # 未知值的原始值与 fallback 标记披露走 classify_with_fallback）。
        return classify_session_class(session.next_request_type)

    def _completed_full_candidates(
        self,
        instance_index: int,
        trigger_type: Optional[str] = None,
    ) -> list[SessionKVState]:
        if self.partial_resident_prefix_layers == self.model.layers:
            return []
        return self._fifo_sort([
            session
            for session in self._sessions.values()
            if session.location == self.LOCAL_HBM
            and session.instance_index == instance_index
            and not session.active
            and session.last_completion_ns is not None
            and (
                trigger_type is None
                or self._eviction_class(session) == trigger_type
            )
        ])

    def _completed_resident_candidates(
        self,
        instance_index: int,
        trigger_type: Optional[str] = None,
    ) -> list[SessionKVState]:
        return self._fifo_sort([
            session
            for session in self._sessions.values()
            if session.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            and session.instance_index == instance_index
            and not session.active
            and session.last_completion_ns is not None
            and (
                trigger_type is None
                or self._eviction_class(session) == trigger_type
            )
        ])

    def _evict_suffix(
        self,
        session: SessionKVState,
        *,
        phase: str,
        reason: str,
        trigger_request_id: str,
        layer_start: Optional[int] = None,
    ) -> KVTransfer:
        """Release the session's resident suffix ``[layer_start, h)`` to the pool.

        joint 改造（设计方案 §5.6）：``layer_start`` 显式化——legacy_half
        缺省仍为半层常量；minimal_layer_groups/adaptive 由释放计划给出
        任意合法前缀边界。支持 LOCAL（释放 [start, L)）与已 PARTIAL
        （释放 [start, h)，h = 当前驻留前缀）两种 victim；逐出后变
        PARTIAL，**home 不变**。
        """
        if session.active or session.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if (
            session.location not in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            or session.instance_index is None
        ):
            raise RuntimeError("suffix eviction requires a resident session")
        resident_layers = session.resident_prefix_layers
        suffix_start = (
            self.partial_resident_prefix_layers
            if layer_start is None else int(layer_start)
        )
        if not 0 < suffix_start < resident_layers:
            raise RuntimeError(
                "suffix eviction layer boundary must split the resident "
                f"prefix (got layer_start={suffix_start}, resident="
                f"{resident_layers})")
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_suffix_half",
            session=session,
            trigger_request_id=trigger_request_id,
            layer_start=suffix_start,
            layer_end=resident_layers,
            resident_prefix_layers_after=suffix_start,
        )
        self._remove_local_shards(
            session.instance_index,
            kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.context_tokens,
                self.tp_degree,
                layer_start=suffix_start,
                layer_end=resident_layers,
            ),
        )
        session.location = self.PARTIAL_HBM_REMOTE
        session.resident_prefix_layers = suffix_start
        self._check_invariants_after_mutation(session_ids=(session.session_id,))
        self._metrics_suffix_evict_parts(
            session.session_id,
            suffix_start,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=(
                f"evict_{reason}_suffix_half:"
                f"layers{suffix_start}-{resident_layers}"
            ),
        )
        return transfer

    def _evict_session(
        self,
        session: SessionKVState,
        *,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> KVTransfer:
        if session.active or session.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if (
            session.location not in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            or session.instance_index is None
        ):
            raise RuntimeError("only locally resident sessions may be evicted")
        resident_layers = session.resident_prefix_layers
        if resident_layers <= 0:
            raise RuntimeError("session has no local layers left to evict")
        local_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=resident_layers,
        )
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_full_fallback",
            session=session,
            trigger_request_id=trigger_request_id,
            layer_start=0,
            layer_end=resident_layers,
            resident_prefix_layers_after=0,
        )
        self._remove_local_shards(session.instance_index, local_shards)
        session.location = self.REMOTE_MEMORY
        session.instance_index = None
        session.resident_prefix_layers = 0
        self._check_invariants_after_mutation(session_ids=(session.session_id,))
        self._metrics_remove_session_parts(
            session.session_id,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=f"evict_{reason}_full_fallback:layers0-{resident_layers}",
        )
        return transfer

    def _insufficient_ranks(
        self,
        instance_index: int,
        required_bytes_by_tp_rank: Sequence[int],
        *,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[int, ...]:
        instance = self.topology.instance(instance_index)
        if len(required_bytes_by_tp_rank) != instance.size:
            raise ValueError("required KV shard count must match target instance")
        effective_remaining = self._effective_remaining_by_tp_rank(
            instance_index,
            exclude_request_id=reservation_request_id,
        )
        return tuple(
            rank
            for rank, available_bytes, required_bytes in zip(
                instance.ranks,
                effective_remaining,
                required_bytes_by_tp_rank,
            )
            if available_bytes < required_bytes
        )

    def commit_deep_gap_records(self, records) -> None:
        """K6：把确认不可恢复的容量缺口记录落 deep_gap_events 台账。

        记录由 KVCapacityError.deep_gap_records 携带（raise 时不落账）；
        仅终态确认点调用——R14 死锁守卫（全停滞无事件源）、merge v2
        双侧深缺口（2026-09-17 起，R4 降级耗尽路径已删除）。可恢复捕获
        路径（准入延迟/停滞/merge 方向回退成功）不调用，台账保持
        "落账 = run 终止"语义。"""
        self.deep_gap_events.extend(records)

    def _ensure_capacity(
        self,
        instance_index: int,
        required_bytes_by_tp_rank: Sequence[int],
        *,
        phase: str,
        reason: str,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
        protected_session_id: Optional[str] = None,
    ) -> tuple[KVTransfer, ...]:
        evictions: list[KVTransfer] = []
        # joint 改造（设计方案 §4/§5.6/§7.1）：T 与 E 正交注入——
        #   * 类别遍历序来自 eviction_priority（typed: human→tool 严格序；
        #     lru: 单遍类型无关，同一合法 victim 集合）；
        #   * 每类的释放层数由 LayerEvictionPolicy 计划（legacy_half 原
        #     两段式 / minimal_layer_groups 最少完整层组 / adaptive 目标
        #     外→目标内两次扫描）；
        #   * 缺口满足即停；human 类内可释放层未耗尽且缺口未满足时不得
        #     转向 tool（§4）；保护/active/未完成 session 不是合法 victim。
        instance = self.topology.instance(instance_index)
        if len(required_bytes_by_tp_rank) != instance.size:
            raise ValueError(
                "required KV shard count must match target instance")

        def _gap_by_tp_rank() -> tuple[int, ...]:
            effective_remaining = self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=reservation_request_id,
            )
            return tuple(
                max(0, required - available)
                for required, available in zip(
                    required_bytes_by_tp_rank, effective_remaining)
            )

        gap = _gap_by_tp_rank()
        if not any(gap):
            return ()

        for eviction_pass in self._eviction_passes:
            if not any(gap):
                break
            trigger_type = eviction_pass.trigger_type
            candidates = self._completed_resident_candidates(
                instance_index, trigger_type)
            if protected_session_id is not None:
                candidates = [
                    session
                    for session in candidates
                    if session.session_id != protected_session_id
                ]
            if not candidates:
                continue

            def layer_group_bytes(
                session_tokens: int,
                layer_start: int,
                layer_end: int,
            ) -> tuple[int, ...]:
                return kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    session_tokens,
                    self.tp_degree,
                    layer_start=layer_start,
                    layer_end=layer_end,
                )

            victims = [
                VictimView(
                    session_id=session.session_id,
                    resident_prefix_layers=session.resident_prefix_layers,
                    layer_group_bytes_fn=(
                        lambda start, end, s=session: layer_group_bytes(
                            s.context_tokens, start, end)),
                    retention_target_layers=0,
                    retention_target_layers_fn=(
                        lambda sid=session.session_id,
                        ctx=session.context_tokens,
                        inst=instance_index:
                        self._adaptive_retention_target_tokens(
                            sid, ctx, inst)),
                    next_request_type=session.next_request_type,
                    last_completion_ns=session.last_completion_ns,
                )
                for session in candidates
            ]
            try:
                plan = self.layer_policy.plan_release(
                    gap_bytes_by_tp_rank=gap,
                    victims=victims,
                )
            except LayerEvictionError as exc:
                raise RuntimeError(
                    f"layer eviction plan failed in instance "
                    f"{instance_index}: {exc}") from exc
            for step in plan.steps:
                victim = self._sessions[step.session_id]
                if step.layer_start <= 0:
                    evictions.append(
                        self._evict_session(
                            victim,
                            phase=phase,
                            reason=reason,
                            trigger_request_id=trigger_request_id,
                        )
                    )
                else:
                    evictions.append(
                        self._evict_suffix(
                            victim,
                            phase=phase,
                            reason=reason,
                            trigger_request_id=trigger_request_id,
                            layer_start=step.layer_start,
                        )
                    )
                gap = _gap_by_tp_rank()
                if not any(gap):
                    break

        # Every class is exhausted while the requirement is still
        # unmet: report the failing ranks.
        insufficient = self._insufficient_ranks(
            instance_index,
            required_bytes_by_tp_rank,
            reservation_request_id=reservation_request_id,
        )
        if insufficient:
            effective_remaining = self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=reservation_request_id,
            )
            # D4 (2026-09-05) + K6 (2026-09-14)：deep-gap 缺口记录随异常
            # 携带、raise 时不落 deep_gap_events 台账——joint 下容量类
            # 失败有多条可恢复捕获路径（准入延迟/merge 方向回退（v2，
            # 2026-09-17 起）/R14 停滞），只有确认不可恢复的提交点（死锁
            # 守卫/双侧深缺口）才落账，台账保持"落账 = run 终止"语义
            #（可恢复事件不得污染）。
            gap_records = tuple({
                "instance_index": instance_index,
                "rank": rank,
                "phase": phase,
                "reason": reason,
                "trigger_request_id": trigger_request_id,
                "remaining_bytes": (
                    effective_remaining[self._rank_relative_index[rank]]),
                "required_bytes": (
                    required_bytes_by_tp_rank[
                        self._rank_relative_index[rank]]),
                "gap_bytes": (
                    required_bytes_by_tp_rank[
                        self._rank_relative_index[rank]]
                    - effective_remaining[
                        self._rank_relative_index[rank]]),
            } for rank in insufficient)
            details = ", ".join(
                f"rank {rank}: remaining="
                f"{effective_remaining[self._rank_relative_index[rank]]}, "
                f"required={required_bytes_by_tp_rank[self._rank_relative_index[rank]]}"
                for rank in insufficient
            )
            # N3'：容量类异常（准入语境可转 False 延迟、merge 语境可回退
            # 另一方向（2026-09-17 起 v2；R4 自降级已删除）、decode 语境
            # 停滞 R14）；evictions 携带 raise 前已提交的逐出（D7 清账：
            # 调用方须 bump 纪元 + 同步图侧）。
            raise KVCapacityError(
                f"insufficient target HBM in instance {instance_index}; "
                f"{details}",
                evictions=tuple(evictions),
                deep_gap_records=gap_records,
            )
        # D4-I1 (2026-09-05): 返回前复核每个被逐会话均已完成且 inactive
        # （_evict_suffix/_evict_session 入口已有同款守卫，此处捕获两阶段
        # 循环间的状态漂移）。
        for transfer in evictions:
            evicted = self._sessions[transfer.session_id]
            if evicted.active or evicted.last_completion_ns is None:
                raise RuntimeError(
                    "passive eviction victim is active or uncompleted: "
                    f"{transfer.session_id}")
        # D4-I3 (2026-09-05): 撤销最后一笔逐出必须至少让一个受影响 rank
        # 回到不满足（否则该笔逐出并非恰好所需，属过度逐出）。
        if evictions:
            freed_by_rank: dict[int, int] = {}
            for transfer in evictions:
                for shard in transfer.shards:
                    if shard.source_rank is None:
                        continue
                    freed_by_rank[shard.source_rank] = (
                        freed_by_rank.get(shard.source_rank, 0) + shard.bytes
                    )
            last_affected_ranks = {
                shard.source_rank
                for shard in evictions[-1].shards
                if shard.source_rank is not None
            }
            effective_now = self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=reservation_request_id,
            )
            for rank in last_affected_ranks:
                relative = self._rank_relative_index[rank]
                freed = freed_by_rank.get(rank, 0)
                if effective_now[relative] - freed < required_bytes_by_tp_rank[relative]:
                    break
            else:
                raise RuntimeError(
                    "passive eviction over-evicted: undoing the last "
                    f"eviction ({evictions[-1].session_id}) still leaves "
                    f"every affected rank satisfied in instance "
                    f"{instance_index}")
        return tuple(evictions)

    def prepare_prefill(
        self,
        *,
        session_id: str,
        target_instance_index: int,
        history_tokens: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
        action: str = "stay",
    ) -> tuple[
        Optional[SessionKVSnapshot],
        tuple[KVTransfer, ...],
        tuple[KVTransfer, ...],
    ]:
        """为本轮执行准备历史（joint 四动作语义，设计方案 §2.2 表）。

        动作与账本效果（primary 字段在跨实例执行期间描述**工作副本**，
        base_* 快照 home 侧权威基础历史；merge_back 在完成后消费）：

        * ``stay``：历史驻留目标实例——LOCAL 命中 local_hit；PARTIAL
          池恢复缺失后缀（原语义，home == target）。
        * ``copy``：基础历史在别处——LOCAL/PARTIAL 基础的**驻留前缀**
          经 NoC 复制为工作副本 + PARTIAL 缺失后缀从池恢复；REMOTE
          基础从池恢复整份到目标。基础副本保留在 home（不删除）。
        * ``recompute``：不搬运历史；工作副本从 0 起步，由重算 prefill
          逐 chunk 物化（expand_prefill 增长）。调度器须同时把历史
          token 折入 prefill 工作量。
        * ``remote-read``：基础前缀留在 home 不动（执行期 decode 相逐迭代
          远读 [0, p) 层，由构图器发射流）。LOCAL 基（p == L，回归锚）：
          工作副本零化起算，仅增量驻留执行端；PARTIAL 基（p < L，混合
          形态，N1(a) 解除 2026-09-17）：缺失后缀 [p, L) 准入相从池恢复
          物化为热 KV（工作副本初始 = S）＋增量——两口径分离（D1）：
          ``shard_bytes`` 为物理真值 S＋增量，``context_tokens`` 为增量
          token 数（0 起步）。

        返回 ``(before, transfers, evictions)``：transfers 为需要物理
        发射的 KV 传输序列（stay-local 为空）；home 语义见 §2.1——
        异地执行不改变逻辑 home，逐出也不清除。
        """
        if action not in ("stay", "copy", "recompute", "remote-read"):
            raise ValueError(f"unknown joint action: {action!r}")
        expected_shards = kv_cache_shard_bytes_for_tokens(
            self.model, history_tokens, self.tp_degree
        )
        expected_total = sum(expected_shards)
        if session_id not in self._sessions:
            if history_tokens != 0:
                raise ValueError(
                    f"new session {session_id} cannot have historical KV tokens"
                )
            # 首轮发射建立 home（§2.1）；此后异地执行/逐出均不改 home。
            self._sessions[session_id] = SessionKVState(
                session_id=session_id,
                location=self.LOCAL_HBM,
                instance_index=target_instance_index,
                home_instance=target_instance_index,
                context_tokens=0,
                total_bytes=0,
                shard_bytes=expected_shards,
                resident_prefix_layers=self.model.layers,
                active=True,
            )
            self._check_invariants_after_mutation(session_ids=(session_id,))
            return None, (), ()

        session = self._sessions[session_id]
        before = self.session_snapshot(session_id)
        if (
            session.context_tokens != history_tokens
            or session.total_bytes != expected_total
            or session.shard_bytes != expected_shards
        ):
            raise ValueError(
                f"session {session_id} historical KV metadata does not match request"
            )
        if session.home_instance is None:
            # D10：home 只在建立点写入——既有会话缺 home 属账本完整性
            # 破损，fail-closed 上报而非防御性补建（偏差登记见
            # PROVENANCE.md）。
            raise RuntimeError(
                f"session {session_id} lost its logical home before "
                "prepare_prefill; ledger integrity failure")

        transfers: list[KVTransfer] = []
        evictions: list[KVTransfer] = []
        session.active = True

        base_location = session.location
        base_instance = session.instance_index
        base_prefix = session.resident_prefix_layers
        exec_instance = target_instance_index

        def _snapshot_base() -> None:
            session.base_location = base_location
            session.base_history_tokens = history_tokens
            session.base_resident_prefix_layers = base_prefix

        if action == "stay":
            if session.location == self.LOCAL_HBM:
                if session.instance_index != exec_instance:
                    raise RuntimeError(
                        "stay action requires resident history at the "
                        "execution instance")
                transfers.append(self._local_hit_transfer(
                    phase="history",
                    reason="history_local_reuse",
                    session=session,
                    trigger_request_id=trigger_request_id,
                ))
            elif session.location == self.PARTIAL_HBM_REMOTE:
                if session.instance_index != exec_instance:
                    raise RuntimeError(
                        "stay action requires resident history at the "
                        "execution instance")
                suffix_start = session.resident_prefix_layers
                suffix_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    history_tokens,
                    self.tp_degree,
                    layer_start=suffix_start,
                    layer_end=self.model.layers,
                )
                evictions.extend(self._ensure_capacity(
                    exec_instance,
                    suffix_shards,
                    phase="history",
                    reason="history_suffix_target_capacity",
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                ))
                transfers.extend(self._plan_suffix_restore_transfers(
                    session=session,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=exec_instance,
                    suffix_start=suffix_start,
                    history_tokens=history_tokens,
                    reason="history_remote_suffix_restore",
                ))
                self._add_local_shards(exec_instance, suffix_shards)
                session.location = self.LOCAL_HBM
                session.resident_prefix_layers = self.model.layers
                self._metrics_add_segment(
                    session_id,
                    exec_instance,
                    history_tokens,
                    suffix_start,
                    self.model.layers,
                    suffix_shards,
                    anchor_kind="transfer_complete",
                    request_id=trigger_request_id,
                    cause="history_remote_suffix_restore",
                )
            else:
                raise RuntimeError(
                    f"stay action requires resident history (got "
                    f"{session.location})")
            self._check_invariants_after_mutation(session_ids=(session_id,))
            return before, tuple(transfers), tuple(evictions)

        # ---- 跨实例动作：建立工作副本（基础历史保留在原位置）。 ----
        if (
            action == "copy"
            and base_location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            and base_instance == exec_instance
        ):
            if base_location == self.LOCAL_HBM:
                # 退化防御：copy 选在基础历史驻留实例上 == stay（成本模型
                # 已排除该候选；此处不建立第二份工作副本）。
                transfers.append(self._local_hit_transfer(
                    phase="history",
                    reason="history_local_reuse",
                    session=session,
                    trigger_request_id=trigger_request_id,
                ))
                self._check_invariants_after_mutation(session_ids=(session_id,))
                return before, tuple(transfers), tuple(evictions)
            # R16-6（GLM 四审 P1，2026-09-15）：copy@home×PARTIAL —— 此前
            # 守卫只拦 LOCAL，PARTIAL 落穿 Case B 后会在 :3556 把整份工作
            # 副本叠在驻留前缀上双计、完成结算撞 N9 防御确定性 raise（适用
            # 性现排除该组合，但"不可达"不是"无害"）。镜像 stay-partial
            # 模板退化：只恢复缺失后缀、不建工作副本、working_kind 保持
            # None（与适用性注释"退化为 stay、不构成第二份工作副本"的
            # 声明语义对齐——实现追上文档）。
            suffix_start = base_prefix
            suffix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                history_tokens,
                self.tp_degree,
                layer_start=suffix_start,
                layer_end=self.model.layers,
            )
            evictions.extend(self._ensure_capacity(
                exec_instance,
                suffix_shards,
                phase="history",
                reason="history_suffix_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
            ))
            transfers.extend(self._plan_suffix_restore_transfers(
                session=session,
                trigger_request_id=trigger_request_id,
                target_instance_index=exec_instance,
                suffix_start=suffix_start,
                history_tokens=history_tokens,
                reason="history_remote_suffix_restore",
            ))
            self._add_local_shards(exec_instance, suffix_shards)
            session.location = self.LOCAL_HBM
            session.resident_prefix_layers = self.model.layers
            self._metrics_add_segment(
                session_id,
                exec_instance,
                history_tokens,
                suffix_start,
                self.model.layers,
                suffix_shards,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_remote_suffix_restore",
            )
            self._check_invariants_after_mutation(session_ids=(session_id,))
            return before, tuple(transfers), tuple(evictions)
        if (
            action == "recompute"
            and base_instance == exec_instance
            and base_location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            and base_prefix > 0
        ):
            # R13（N7(b)，2026-09-14 裁定）：recompute@home——驻留前缀保持
            # 权威，不清零 shard_bytes、不孤儿化、不建工作副本（与 stay
            # 同构的本地事务）。缺失后缀层 [base_prefix, L) 由重算物化：
            # prepare 点记账对齐 stay-partial 的"恢复入账在 prepare、物理
            # 完成由图门控"抽象——重算的物理计算量由调度器折入列车 span
            # （joint_prefill_work = 缺失后缀折算 token + input）。
            # working_kind 保持 None：home==exec 的 recompute 是 stay 等价
            # 的本地提交（merge 零流量，§2.2），base 快照不建立（消除
            # M2 幻影驻留与执行期瞬时双足迹）。
            suffix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                history_tokens,
                self.tp_degree,
                layer_start=base_prefix,
                layer_end=self.model.layers,
            )
            if any(suffix_shards):
                evictions.extend(self._ensure_capacity(
                    exec_instance,
                    suffix_shards,
                    phase="history",
                    reason="history_recompute_suffix_materialize",
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                ))
                self._add_local_shards(exec_instance, suffix_shards)
                self._metrics_add_segment(
                    session_id,
                    exec_instance,
                    history_tokens,
                    base_prefix,
                    self.model.layers,
                    suffix_shards,
                    anchor_kind="prefill_start",
                    request_id=trigger_request_id,
                    cause="history_recompute_suffix_materialize",
                )
            session.location = self.LOCAL_HBM
            # F4-a 口径注记（M4，kimi 复审登记）：resident_prefix_layers
            # 在 prepare 事务内推进到 L——后缀层经 _add_local_shards 已
            # **同步物化入账本**（非幻影路径：物化与推进同事务完成，与
            # stay-partial 抽象同型）；规格原文"物化完成推进"按事务提交
            # 点解读，偏离已登记 PROVENANCE §7-10。
            session.resident_prefix_layers = self.model.layers
            self._check_invariants_after_mutation(session_ids=(session_id,))
            return before, tuple(transfers), tuple(evictions)
        if base_location == self.REMOTE_MEMORY or base_prefix == 0:
            # 基础历史仅存池 backing（曾被整份逐出）。
            _snapshot_base()
            full_shards = expected_shards
            if action == "copy":
                evictions.extend(self._ensure_capacity(
                    exec_instance,
                    full_shards,
                    phase="history",
                    reason="history_target_capacity",
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                ))
                transfers.extend(self._plan_suffix_restore_transfers(
                    session=session,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=exec_instance,
                    suffix_start=0,
                    history_tokens=history_tokens,
                    reason="history_pool_restore_working_copy",
                ))
                self._add_local_shards(exec_instance, full_shards)
                session.shard_bytes = full_shards
                session.total_bytes = sum(full_shards)
            elif action in ("recompute", "remote-read"):
                # 重算：工作副本 0 起步（重算 prefill 物化）；远读：仅
                # 增量驻留执行端（新增 token 的 KV，从 0 增长）。
                zero_shards = tuple(0 for _ in full_shards)
                session.shard_bytes = zero_shards
                session.total_bytes = 0
                session.context_tokens = 0
            session.location = self.LOCAL_HBM
            session.instance_index = exec_instance
            session.resident_prefix_layers = self.model.layers
            session.working_kind = action
            self._check_invariants_after_mutation(session_ids=(session_id,))
            return before, tuple(transfers), tuple(evictions)

        # 基础驻留在 home 侧实例（LOCAL 全量或 PARTIAL 前缀）。
        home_side_instance = base_instance
        _snapshot_base()
        prefix_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            history_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=base_prefix,
        )
        suffix_start = base_prefix
        suffix_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            history_tokens,
            self.tp_degree,
            layer_start=suffix_start,
            layer_end=self.model.layers,
        )
        if action == "copy":
            working_shards = tuple(
                a + b for a, b in zip(prefix_shards, suffix_shards))
            evictions.extend(self._ensure_capacity(
                exec_instance,
                working_shards,
                phase="history",
                reason="history_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
            ))
            if home_side_instance != exec_instance:
                # R16-2：前缀腿只传驻留前缀 [0, base_prefix)——R16-1
                # 层区间化后 PARTIAL 基不再撞全驻留守卫；LOCAL 基
                # （base_prefix == L）与旧全层口径逐位一致。
                # C13（2026-09-22）：前缀腿改为**逐 chunk 交接流**（设计
                # 文档 §2.3 四步协议）——驻留前缀按消费顺序切成层块：
                # 块 0 走准入主链（readiness barrier 只等它，计算不等
                # 整份搬运完成）；块 ≥ 1 由构图器旁挂支链流水发射、逐
                # 块就绪门控列车体（GB _copy_handoff_arms）。base_prefix
                # ≤ COPY_HANDOFF_CHUNK_LAYERS 时单块 = 旧单笔口径回归
                # 锚。账本侧：源块安全前提断言 + 交接 journal 登记；home
                # 侧字节在逐块交接完成事件中释放（不等轮末）。
                self._assert_copy_handoff_source_safety(
                    session, home_side_instance)
                chunk_ranges = plan_copy_handoff_layer_chunks(base_prefix)
                chunks = tuple(
                    CopyHandoffChunk(
                        index=chunk_index,
                        layer_start=layer_start,
                        layer_end=layer_end,
                        shard_bytes=kv_cache_shard_bytes_for_layer_range(
                            self.model,
                            history_tokens,
                            self.tp_degree,
                            layer_start=layer_start,
                            layer_end=layer_end,
                        ),
                    )
                    for chunk_index, (layer_start, layer_end)
                    in enumerate(chunk_ranges)
                )
                for chunk in chunks:
                    transfers.append(self._noc_transfer(
                        phase="history",
                        reason=("history_prefix_working_copy"
                                if chunk.index == 0
                                else "history_prefix_handoff_tail"),
                        session=session,
                        trigger_request_id=trigger_request_id,
                        source_instance_index=home_side_instance,
                        target_instance_index=exec_instance,
                        layer_start=chunk.layer_start,
                        layer_end=chunk.layer_end,
                        handoff_chunk=chunk.index,
                    ))
                journal = CopyHandoffJournal(
                    session_id=session_id,
                    trigger_request_id=trigger_request_id,
                    home_instance=home_side_instance,
                    exec_instance=exec_instance,
                    base_history_tokens=history_tokens,
                    chunks=chunks,
                )
                total = journal.total_shards
                journal.h_exec_shards = total
                journal.h_home_shards = total
                session.copy_handoff = journal
                journal.assert_conservation()
                # 守恒科目：launch（#handoff）+ 流计划（#copy-stream——
                # 被迁移原有历史有效载荷恰好一遍；池恢复后缀/增量/协议
                # 工作另列分计，不入 payload）。
                launch_event = {
                    "subject": f"{trigger_request_id}#handoff",
                    "event": "launch",
                    "session_id": session_id,
                    "home_instance": home_side_instance,
                    "exec_instance": exec_instance,
                    "chunks": len(chunks),
                    "h_bytes": sum(total),
                    "h_home_bytes": sum(total),
                    "h_exec_bytes": sum(total),
                    "d_handoff_bytes": sum(total),
                }
                stream_plan_event = {
                    "subject": f"{trigger_request_id}#copy-stream",
                    "event": "plan",
                    "session_id": session_id,
                    "chunks": [
                        {"index": chunk.index,
                         "layer_start": chunk.layer_start,
                         "layer_end": chunk.layer_end,
                         "bytes": chunk.total_bytes}
                        for chunk in chunks],
                    "payload_bytes": sum(total),
                    # 另列分计（不入 #copy-stream payload）：
                    "separate_ledgers": [
                        "pool_suffix_restore", "incremental_growth",
                        "hop_transport", "victim_writeback",
                        "protocol_ack_overhead"],
                }
                journal.events.append(launch_event)
                journal.events.append(stream_plan_event)
                self.copy_handoff_events.append(launch_event)
                self.copy_handoff_events.append(stream_plan_event)
            if any(suffix_shards):
                transfers.extend(self._plan_suffix_restore_transfers(
                    session=session,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=exec_instance,
                    suffix_start=suffix_start,
                    history_tokens=history_tokens,
                    reason="history_suffix_pool_restore_working_copy",
                ))
            # C13 口径注记：工作副本（前缀+后缀）在准入相物化入执行端容量
            # 账（容量保守上界，预约→物化的分配关系经 _reservation_extra_
            # shards 核账）；**物理到达**由图侧逐块门控（readiness barrier
            # 只等块 0），home 侧字节则在逐块交接完成事件中释放（prefill
            # drain 边界结算，不等轮末——_settle_copy_handoffs）。
            self._add_local_shards(exec_instance, working_shards)
            session.shard_bytes = working_shards
            session.total_bytes = sum(working_shards)
            self._metrics_add_segment(
                session_id,
                exec_instance,
                history_tokens,
                0,
                self.model.layers,
                working_shards,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_working_copy",
            )
        elif action == "recompute":
            zero_shards = tuple(0 for _ in prefix_shards)
            session.shard_bytes = zero_shards
            session.total_bytes = 0
            session.context_tokens = 0
        else:  # remote-read：前缀基础留在 home；后缀按基形态处置。
            if base_instance == exec_instance:
                # 防御 fail-closed：remote-read 的执行实例不得等于基础
                # 驻留实例（适用性已排除该组合；到达即选点/适用性合同
                # 破损，零化工作副本会摧毁权威基础）。
                raise RuntimeError(
                    "remote-read action requires the base history to stay "
                    "at another instance (exec == resident instance)")
            if base_location == self.PARTIAL_HBM_REMOTE:
                # 混合形态（N1(a) 解除，2026-09-17 用户裁定）：PARTIAL 基
                # 的缺失后缀 [p, L) 在准入相从池恢复物化为热 KV（读流只
                # 覆盖 home 前缀 [0, p)）。两口径分离（D1 裁决）：
                # shard_bytes = 物理真值 S（后缀物化入账），context_tokens
                # = 增量 token 数（0 起步照旧）——merge v2 前向腿按
                # shard_bytes 真值搬运。
                evictions.extend(self._ensure_capacity(
                    exec_instance,
                    suffix_shards,
                    phase="history",
                    reason="history_suffix_target_capacity",
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                ))
                transfers.extend(self._plan_suffix_restore_transfers(
                    session=session,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=exec_instance,
                    suffix_start=base_prefix,
                    history_tokens=history_tokens,
                    reason="history_suffix_pool_restore_working_copy",
                ))
                self._add_local_shards(exec_instance, suffix_shards)
                session.shard_bytes = suffix_shards
                session.total_bytes = sum(suffix_shards)
                session.context_tokens = 0
                self._metrics_add_segment(
                    session_id,
                    exec_instance,
                    history_tokens,
                    base_prefix,
                    self.model.layers,
                    suffix_shards,
                    anchor_kind="transfer_complete",
                    request_id=trigger_request_id,
                    cause="history_suffix_pool_restore_working_copy",
                )
            else:
                # LOCAL 基（回归锚）：工作副本零化起算，逐字节与旧口径
                # 一致——仅增量驻留执行端（新增 token 的 KV，从 0 增长）。
                zero_shards = tuple(0 for _ in prefix_shards)
                session.shard_bytes = zero_shards
                session.total_bytes = 0
                session.context_tokens = 0

        session.location = self.LOCAL_HBM
        session.instance_index = exec_instance
        session.resident_prefix_layers = self.model.layers
        session.working_kind = action
        self._check_invariants_after_mutation(session_ids=(session_id,))
        return before, tuple(transfers), tuple(evictions)
    def _expand_local_session(
        self,
        *,
        session_id: str,
        instance_index: int,
        context_tokens: int,
        phase: str,
        reason: str,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[KVTransfer, ...]:
        session = self._sessions[session_id]
        if (
            session.location != self.LOCAL_HBM
            or session.instance_index != instance_index
        ):
            raise RuntimeError(
                f"session {session_id} is not local to instance {instance_index}"
            )
        if phase == "prefill":
            # C13：prefill drain 边界（expand_prefill 的唯一在线调用面 =
            # _on_prefill_drain，全部 prefill 列车已核销）结算 copy 逐块
            # 交接——home 侧在此逐块释放（不等轮末；decode 相中段的
            # expand_decode 不是安全边界，见 _settle_copy_handoffs）。
            self._settle_copy_handoffs(
                session_id,
                boundary="prefill_drain",
                trigger_request_id=trigger_request_id,
            )
            # C15：同边界结算后缀逐组恢复（rid#restore complete——逐组
            # 就绪门控的消费保证已在图侧兑现）。
            self._settle_restore_groups(
                session_id, boundary="prefill_drain")
        if self._working_copy_uses_shard_truth(session):
            # 混合形态（D1 两口径分离，2026-09-17）：工作副本物理量 =
            # 池恢复后缀（冻结于 base_history_tokens 的 [p, L) 层——历史
            # 后缀不随增量增长）＋ 增量全层 kv(context_tokens)（此处
            # context_tokens 为增量 token 口径）。delta 仍恰为增量增长。
            restored_suffix = kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.base_history_tokens,
                self.tp_degree,
                layer_start=session.base_resident_prefix_layers,
                layer_end=self.model.layers,
            )
            incremental = kv_cache_shard_bytes_for_tokens(
                self.model, context_tokens, self.tp_degree
            )
            new_shards = tuple(
                suffix_bytes + incremental_bytes
                for suffix_bytes, incremental_bytes in zip(
                    restored_suffix, incremental)
            )
        else:
            new_shards = kv_cache_shard_bytes_for_tokens(
                self.model, context_tokens, self.tp_degree
            )
        delta = tuple(
            new_bytes - old_bytes
            for new_bytes, old_bytes in zip(new_shards, session.shard_bytes)
        )
        if any(value < 0 for value in delta):
            raise ValueError("KV cache context cannot shrink during a request")
        evictions = self._ensure_capacity(
            instance_index,
            delta,
            phase=phase,
            reason=reason,
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
        )
        self._add_local_shards(instance_index, delta)
        previous_tokens = session.context_tokens
        session.context_tokens = context_tokens
        session.total_bytes = sum(new_shards)
        session.shard_bytes = new_shards
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_add_segment(
            session_id,
            instance_index,
            context_tokens - previous_tokens,
            0,
            self.model.layers,
            delta,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=reason,
        )
        return evictions

    def expand_prefill(
        self,
        *,
        session_id: str,
        instance_index: int,
        context_tokens: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[KVTransfer, ...]:
        return self._expand_local_session(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=context_tokens,
            phase="prefill",
            reason="prefill_growth_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
        )

    def move_prefill_to_decode(
        self,
        *,
        session_id: str,
        target_instance_index: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[KVTransfer, tuple[KVTransfer, ...]]:
        session = self._sessions[session_id]
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("Prefill KV must be local before Decode mapping")
        source_instance_index = session.instance_index
        if source_instance_index == target_instance_index:
            return (
                self._local_hit_transfer(
                    phase="prefill_decode",
                    reason="prefill_decode_local_reuse",
                    session=session,
                    trigger_request_id=trigger_request_id,
                ),
                (),
            )
        evictions = self._ensure_capacity(
            target_instance_index,
            session.shard_bytes,
            phase="decode",
            reason="decode_target_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
        )
        transfer = self._noc_transfer(
            phase="prefill_decode",
            reason="prefill_decode_instance_migrate",
            session=session,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=target_instance_index,
            layer_start=0,
            layer_end=self.model.layers,
        )
        self._remove_local_shards(source_instance_index, session.shard_bytes)
        self._add_local_shards(target_instance_index, session.shard_bytes)
        session.instance_index = target_instance_index
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_move_session_parts(
            session_id,
            target_instance_index,
            session.context_tokens,
            session.resident_prefix_layers,
            session.shard_bytes,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="prefill_decode_instance_migrate",
        )
        return transfer, evictions

    def expand_decode(
        self,
        *,
        session_id: str,
        instance_index: int,
        final_context_tokens: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[KVTransfer, ...]:
        return self._expand_local_session(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=final_context_tokens,
            phase="decode",
            reason="decode_growth_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
        )

    # ------------------------------------------------ joint 合并事务（§2.2/§3.2） --

    def _working_copy_noc_transfer(
        self,
        *,
        session: SessionKVState,
        trigger_request_id: str,
        source_instance_index: int,
        target_instance_index: int,
    ) -> KVTransfer:
        """merge v2 前向腿：整份工作副本 NoC 迁移（exec → home 胜者）。

        逐 rank 字节**直接取 ``session.shard_bytes``**（账本真值；D1 裁决，
        2026-09-17）——混合形态（remote-read×PARTIAL 基）下 context_tokens
        是增量口径（0 起步），从 context 派生会丢掉准入相池恢复的后缀字节
        （LOCAL 基两口径恒等，逐字节与旧增量腿一致）。层区间 [0, L)；
        ``resident_prefix_layers_before/after`` 沿本仓区间传输惯例 =
        layer_start/layer_end（:2800/:3924 同例）。
        """
        layer_start = 0
        layer_end = self.model.layers
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=shard_bytes,
                noc_path=deterministic_xy_route(
                    self.topology.hardware, source_rank, target_rank),
                layer_start=layer_start,
                layer_end=layer_end,
            )
            for source_rank, target_rank, shard_bytes in zip(
                source.ranks, target.ranks, session.shard_bytes)
            if shard_bytes > 0
        )
        return KVTransfer(
            kind="noc_migrate",
            phase="completion",
            reason="merge_working_copy_to_home",
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=target_instance_index,
            total_bytes=sum(session.shard_bytes),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=layer_start,
            resident_prefix_layers_after=layer_end,
        )

    # ------------------------------------- C13 copy 块级交接（§2.3）--

    def _assert_copy_handoff_source_safety(
        self,
        session: SessionKVState,
        home_instance: int,
    ) -> None:
        """四步协议第 (1) 步：源块安全交接前提（C12 冻结规格 §5）。

        * 在途读取：本会话上一轮的工作副本/交接账本必须已闭合
          （working_kind is None 且 copy_handoff is None）——否则源块仍
          有上一轮读流在途，动作不适用（fail-closed，不静默排队）。
        * 共享引用：本账本的 KV 物理块为**会话私有**（无跨会话引用计数
          /共享前缀机制——C12 §5"不引入新的共享前缀复制机制"，仓库现
          状即无该机制），源块唯一合法使用者即本会话本轮迁移流；若未
          来引入共享机制，必须先在此处理引用关系再放行。
        """
        if session.working_kind is not None or session.copy_handoff is not None:
            raise RuntimeError(
                f"copy handoff for session {session.session_id} launched "
                "while a previous working copy or handoff journal is still "
                "open -- source blocks have in-flight readers from an "
                "earlier round; the copy action is not safely handable in "
                "this state (C12 frozen spec sec.5: handle the reference "
                "relation first or mark the action inapplicable)")

    def _settle_copy_handoffs(
        self,
        session_id: str,
        *,
        boundary: str,
        trigger_request_id: str,
    ) -> int:
        """在因果安全边界结算全部待交接块（四步协议 (2)+(3) 的账本侧）。

        边界白名单（因果序：块到达已被图侧钉在 prefill 列车体完成之前，
        两边界均 ≥ 该时刻——**不等轮末**）：
          * ``"prefill_drain"``：expand_prefill（唯一调用面 = 调度器
            _on_prefill_drain——全部 prefill 列车核销之后，逐块就绪门
            控 guarantee 已兑现）；
          * ``"merge"``：merge_back 头部（防御兜底——正常路径已在
            drain 边界结算；merge 时仍未结算只可能来自直接 API 调用面）。
        其余边界（如 decode 相中段 expand_decode）**不是**安全边界：
        多列车 prefill 下尾块可能仍在途，early release 会被本方法拒绝。
        """
        if boundary not in ("prefill_drain", "merge"):
            raise RuntimeError(
                f"unknown copy handoff settle boundary {boundary!r}")
        session = self._sessions[session_id]
        journal = session.copy_handoff
        if journal is None:
            return 0
        settled = 0
        while journal.next_pending_index() is not None:
            self._apply_copy_handoff_event(
                session,
                journal.next_pending_index(),
                boundary=boundary,
                trigger_request_id=trigger_request_id,
            )
            settled += 1
        if settled:
            self._check_invariants_after_mutation(session_ids=(session_id,))
        return settled

    def _settle_restore_groups(
        self,
        session_id: str,
        *,
        boundary: str,
    ) -> int:
        """C15：后缀逐组恢复的结算边界（科目 rid#restore 的 complete）。

        与 copy 交接同一边界白名单（``prefill_drain`` / ``merge``）：
        prefill drain = 全部 prefill 列车核销后，逐组就绪门控的消费保
        证已兑现（迟到组的消费等待已按层段如实入图依赖）；merge = 兜
        底。结算 = 全部组置 CONSUMED + 守恒断言；完成时未结算 = 账本
        破损 fail-closed（mark_complete 处检查）。
        """
        if boundary not in ("prefill_drain", "merge", "mark_complete"):
            raise RuntimeError(
                f"unknown restore group settle boundary {boundary!r}")
        session = self._sessions[session_id]
        journal = session.restore_journal
        if journal is None:
            return 0
        journal.settle(boundary)
        journal.assert_conservation()
        self.restore_events.extend(journal.events[-1:])
        session.restore_journal = None
        return len(journal.entries)

    def _apply_copy_handoff_event(
        self,
        session: SessionKVState,
        chunk_index: int,
        *,
        boundary: str,
        trigger_request_id: str,
    ) -> None:
        """单个交接完成事件：目标侧权威化 + home 侧立即释放（同一事件）。

        顺序协议：必须按消费顺序（块 i-1 已交接是块 i 的前提）；重复
        交接/乱序交接 = 合同类违规 fail-closed。
        """
        journal = session.copy_handoff
        if journal is None:
            raise RuntimeError(
                f"copy handoff event for session {session.session_id} "
                "without an open journal")
        chunk = journal.chunks[chunk_index]
        if chunk.state == "HANDED_OFF":
            raise RuntimeError(
                f"duplicate copy handoff for chunk {chunk_index} of session "
                f"{session.session_id} (request {trigger_request_id}) -- "
                "the chunk was already handed off (contract violation)")
        if chunk.index != journal.next_pending_index():
            raise RuntimeError(
                f"out-of-order copy handoff for chunk {chunk_index} of "
                f"session {session.session_id}: next pending chunk is "
                f"{journal.next_pending_index()} (consumption-order "
                "protocol)")
        # (2) 目标侧权威化：块位置/所有权转移到执行端，后续读取路由到
        #     目标有效副本（执行端物理字节已在准入相物化——保守容量
        #     口径；权威副本自此在执行端）。
        chunk.state = "HANDED_OFF"
        # (3) 同一交接完成事件中立即释放 home 侧对应 HBM、撤销驻留占用
        #     记账（不等轮末）。
        self._release_copy_handoff_chunk(
            session, chunk_index,
            boundary=boundary, trigger_request_id=trigger_request_id)
        journal.assert_conservation()
        handoff_event = {
            "subject": f"{journal.trigger_request_id}#handoff",
            "event": "handoff",
            "session_id": session.session_id,
            "chunk": chunk.index,
            "layer_start": chunk.layer_start,
            "layer_end": chunk.layer_end,
            "bytes": chunk.total_bytes,
            "boundary": boundary,
            "h_home_bytes": sum(journal.h_home_shards),
            "h_exec_bytes": sum(journal.h_exec_shards),
            "d_handoff_bytes": sum(journal.d_handoff_shards()),
        }
        stream_event = {
            "subject": f"{journal.trigger_request_id}#copy-stream",
            "event": "stream-once",
            "session_id": session.session_id,
            "chunk": chunk.index,
            "bytes": chunk.total_bytes,
            "boundary": boundary,
        }
        journal.events.append(handoff_event)
        journal.events.append(stream_event)
        self.copy_handoff_events.append(handoff_event)
        self.copy_handoff_events.append(stream_event)

    def _release_copy_handoff_chunk(
        self,
        session: SessionKVState,
        chunk_index: int,
        *,
        boundary: str,
        trigger_request_id: str,
    ) -> None:
        """home 侧对应块的立即释放（守卫：交接前释放/双释放均 fail-closed）。"""
        journal = session.copy_handoff
        if journal is None:
            raise RuntimeError(
                f"copy handoff release for session {session.session_id} "
                "without an open journal")
        chunk = journal.chunks[chunk_index]
        if chunk.state != "HANDED_OFF":
            raise RuntimeError(
                f"copy handoff release for chunk {chunk_index} of session "
                f"{session.session_id} before its handoff completed -- "
                "releasing the home-side copy before the target-side "
                "authority transfer is an early release (four-step "
                "protocol violation)")
        if chunk_index in journal.released_indices:
            raise RuntimeError(
                f"double release of copy handoff chunk {chunk_index} for "
                f"session {session.session_id} (request {trigger_request_id})"
                " -- the home-side block was already released")
        self._remove_local_shards(journal.home_instance, chunk.shard_bytes)
        journal.released_indices.add(chunk_index)
        journal.h_home_shards = tuple(
            home - chunk_bytes for home, chunk_bytes in zip(
                journal.h_home_shards, chunk.shard_bytes))
        # 分配关系核账：home 侧不得残留锁定已释放字节的预约（预约挂执
        # 行实例；结构性断言，防"源块释放后仍被整轮预留锁住"回归）。
        for reservation in self._reservations.values():
            if (reservation.session_id == session.session_id
                    and reservation.instance_index == journal.home_instance):
                raise RuntimeError(
                    f"copy handoff released home-side bytes of session "
                    f"{session.session_id} while a reservation still pins "
                    "them at the home instance (allocation-relation "
                    "violation)")
        self._metrics_prefix_release_parts(
            session.session_id,
            chunk.layer_end,
            instance_index=journal.home_instance,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="copy_handoff_source_release",
        )

    def _close_copy_handoff_journal(
        self,
        session: SessionKVState,
        *,
        boundary: str,
    ) -> None:
        """轮末闭合（merge_back 尾部）：断言轮末闭合 + 清账本。"""
        journal = session.copy_handoff
        if journal is None:
            return
        if not journal.all_settled():
            raise RuntimeError(
                f"copy handoff journal for session {session.session_id} "
                "did not close at round end -- base history never fully "
                "landed on the execution instance (copy completion "
                "requires the full base history resident at the exec side; "
                "stage names do not substitute for the arrival events)")
        if any(journal.home_shards()) or any(journal.d_handoff_shards()):
            raise RuntimeError(
                f"copy handoff journal for session {session.session_id} "
                "closed with residual home-side or duplicated bytes "
                f"(H_home={journal.home_shards()}, "
                f"D={journal.d_handoff_shards()})")
        journal.assert_conservation()
        close_event = {
            "subject": f"{journal.trigger_request_id}#handoff",
            "event": "close",
            "session_id": session.session_id,
            "boundary": boundary,
            "chunks": len(journal.chunks),
            "h_home_bytes": 0,
            "h_exec_bytes": sum(journal.h_exec_shards),
            "d_handoff_bytes": 0,
            "streamed_once_bytes": sum(journal.released_shards()),
        }
        journal.events.append(close_event)
        self.copy_handoff_events.append(close_event)
        session.copy_handoff = None

    def merge_back(
        self,
        *,
        session_id: str,
        trigger_request_id: str,
        new_tokens: int,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[KVTransfer, ...]:
        """compute_done 后的合并事务 v2：少并多（2026-09-17 用户裁定）。

        ``new_tokens`` = 本轮新增 token 数（输入 + 实际完成 decode——已
        观测长度，允许用于实际结算）。方向裁决按完成时刻两侧保留量比
        大小、小的整份搬给大的（§4.3.1 真值表），结果恒**全层 LOCAL@
        胜者**、败者侧释放、``home_instance`` 迁移到胜者（I7；全仓第二
        个赋值点）：

        * remote-read×LOCAL 基：B（home 全层基础）对 I（exec 增量）——
          B ≥ I 前向搬 I（与旧 v1 前向腿逐字节一致，回归锚）；B < I
          翻转搬 B（home→exec 含传输）。
        * remote-read×PARTIAL 基（混合形态）：H（home 前缀基础）对 S＋I
          （exec 池恢复后缀＋增量，账本真值 ``shard_bytes``）——
          H ≥ S＋I 前向；否则翻转搬 H。
        * copy/recompute（LOCAL/PARTIAL 基）：exec 恒持并集 ⊇ home 侧——
          **零字节翻转**（无传输，home 侧基础释放，胜者空间准备量 = 0）。
        * REMOTE 基（无主，裁定③）：**就地保留**（in_place）——零传输
          零池写，工作副本直接转正，home := exec。

        热 KV 裁定（2026-09-17）封死了"执行完再踢出去"：merge 事务**零
        池写（I6，无 remote_store）**；空间准备只走统一 T+E 逐出（I8，
        无 merge 专用驱逐）——R4 自降级与 k=0 池归并兜底已按裁定④直接
        删除（单实现，不设开关）。首选方向容量不可得时改试另一方向
        （仅 remote-read 两方向均有真实意义；零字节翻转/in_place 的准
        备量 = 0 不可能失败）；双侧统一逐出耗尽仍不可得 = 双侧深缺口，
        逐 rank 缺口落账 deep-gap 后 fail-closed——合同从"merge 永不
        失败"改为"双侧深缺口才失败"。stay 路径（无工作副本）为本地
        提交，不产生流量；恰好归并一次由 ``last_merged_request_id``
        版本键断言保证（重复写 KV/重复释放源副本/重复计 token 均为
        合同类违规）。每次调用（含 stay 早退）结束前设置
        ``self.last_merge_outcome`` 披露快照（改造二契约，供调度器
        日志/水印重放消费）。

        N9（2026-09-14，语义收窄 2026-09-17）：home==exec 的 working
        组合仍是账本破损组合（R13 已消除合法来源：recompute@home 本地
        提交、copy@home 退化 local_hit、remote-read@home 被适用性排除）
        ——结算前 fail-closed；翻转/in_place 产生的 home==exec 是合法
        终态。K5（P2-5，2026-09-23）：pool-only 基（REMOTE 基/前缀 0）
        豁免 raise——REMOTE 基 ×copy@home 是裁定③就地转正形态（基础
        不在 home，非 R13 消除的"基础在 home"组合），方向裁决走
        in_place 免单。
        """
        session = self._sessions[session_id]
        if session.last_merged_request_id == trigger_request_id:
            # 版本键（R4/N9）：同一请求重复 merge = 重复写 KV/重复释放源
            # 副本/重复计 token 的合同类违规（stay 的空 merge 也计一次）。
            raise RuntimeError(
                f"merge_back settled twice for request {trigger_request_id} "
                f"(session {session_id}) -- duplicate merge transaction")
        if session.working_kind is None:
            # stay：本地提交，无跨 instance 回传（§2.2：执行位置等于
            # home 时数据提交不生成虚构 D2D 流量）。
            session.last_merged_request_id = trigger_request_id
            self.last_merge_outcome = {
                "session_id": session_id,
                "direction": "stay",
                "zero_byte_flip": False,
                "winner_instance": None,
                "loser_instance": None,
                "transferred_bytes": 0,
                "home_flipped": False,
            }
            self._append_kv_delta_row(
                session_id=session_id,
                trigger_request_id=trigger_request_id,
                working_kind=None,
                direction="stay",
                zero_byte_flip=False,
                winner_instance=None,
                loser_instance=None,
                home_before=session.home_instance,
                transferred_bytes=0,
                home_side_retained_bytes=int(session.total_bytes),
                exec_side_retained_bytes=0,
                new_tokens=new_tokens,
            )
            return ()
        if new_tokens < 0:
            raise ValueError("new_tokens must be non-negative")
        exec_instance = session.instance_index
        if exec_instance is None:
            raise RuntimeError("working copy lost its execution instance")
        home = session.home_instance
        if home is None:
            raise RuntimeError("working copy has no origin home")
        if home == exec_instance:
            # K5（P2-5，2026-09-23 外部审计）：pool-only 基（REMOTE 基/
            # 前缀 0）豁免——REMOTE 基 ×copy@home 是裁定③的就地转正
            # 形态（home 侧无基础保留量 ⇒ 方向裁决走 in_place 免单），
            # 原 N9 对 home==exec 一刀切把该合法来源拦死（确定性
            # raise）。R13/N9 消除的是"基础在 home 的 copy@home 退化
            # stay"；无主基不在其列。remote-read 不达此处（适用性排除
            # REMOTE 基，下方 fail-closed 保持）。
            # O14：空串 base_location 属会话元数据破损（None=无主基是
            # 唯一合法未置形态），不得经 `or REMOTE` 静默借道豁免。
            base_location_n9 = session.base_location
            if base_location_n9 == "":
                raise RuntimeError(
                    f"session {session_id} has empty-string base_location "
                    "at merge_back -- session metadata corrupt (None is "
                    "the only legal unset form for a pool-only base)")
            if base_location_n9 is None:
                base_location_n9 = self.REMOTE_MEMORY
            if (base_location_n9 == self.REMOTE_MEMORY
                    or not session.base_resident_prefix_layers):
                pass
            else:
                # N9 防御：home==exec 的 working 组合不应存在（见
                # docstring），到达即选点/结算合同破损——显式 fail-closed
                # 而非按旧口径抹账。
                raise RuntimeError(
                    f"session {session_id} has a working copy "
                    f"({session.working_kind}) at its own home instance; "
                    "R13 removed this combination (recompute@home settles "
                    "locally, copy@home degenerates to stay)")
        # C13：copy 逐块交接的轮末兜底结算（正常路径已在 prefill drain
        # 边界结算；因果序：块到达被图侧钉在 prefill 列车体完成之前 ≤
        # drain < merge——此处只可能是直接 API 调用面未走 drain）。
        self._settle_copy_handoffs(
            session_id, boundary="merge", trigger_request_id=trigger_request_id)
        # C15：后缀逐组恢复的轮末兜底（同 C13 边界语义）。
        self._settle_restore_groups(session_id, boundary="merge")
        # C14：home 侧物理残量快照（结算时刻事实，_close 前捕获）——copy
        # 逐块交接已在轮内释放的字节不得计入"两侧实际保留量"（C12 §2
        # 规则 4 的 journal 口径镜像）；无 journal（remote-read/recompute）
        # 取 None ⇒ 纪录 base 前缀推导值（物理驻留真值）。
        _copy_journal_at_merge = session.copy_handoff
        _home_residual_bytes = (
            sum(_copy_journal_at_merge.home_shards())
            if _copy_journal_at_merge is not None else None)
        transfers: list[KVTransfer] = []

        # ---- 方向裁决（§4.3.1 真值表，少并多）。 ----
        # O14：空串 base_location 显式 fail-closed（同上方 N9 注释）。
        base_location = session.base_location
        if base_location == "":
            raise RuntimeError(
                f"session {session_id} has empty-string base_location "
                "at merge_back -- session metadata corrupt (None is "
                "the only legal unset form for a pool-only base)")
        if base_location is None:
            base_location = self.REMOTE_MEMORY
        base_prefix = session.base_resident_prefix_layers
        base_history_tokens = session.base_history_tokens
        if base_location == self.REMOTE_MEMORY or base_prefix == 0:
            # home 侧基础保留量 H（逐 rank 账本真值；REMOTE 基/前缀 0 → 空）。
            home_shards: tuple[int, ...] = ()
        else:
            home_shards = kv_cache_shard_bytes_for_layer_range(
                self.model, base_history_tokens, self.tp_degree,
                layer_start=0, layer_end=base_prefix)
        # exec 侧工作副本保留量 W（账本真值：混合形态含池恢复后缀 S）。
        working_shards = tuple(session.shard_bytes)
        # C14：结算时刻的工作副本类别快照（终态清零 working_kind 之前
        # 捕获——kv_delta_journal 行的字段来源）。
        working_kind_at_settlement = session.working_kind
        zero_byte_flip = session.working_kind in ("copy", "recompute")
        if not home_shards:
            if session.working_kind == "remote-read":
                # 适用性排除 remote-read×REMOTE 基（正常不可达）：远读
                # 工作副本从不物化基础历史，"就地转正"无从谈起——到达即
                # 选点/适用性合同破损，fail-closed。
                raise RuntimeError(
                    f"session {session_id} merged a remote-read working copy "
                    "over a pool-only base; remote-read applicability "
                    "excludes REMOTE bases (no base was ever materialized "
                    "at the execution instance)")
            direction = "in_place"
        elif zero_byte_flip:
            # copy/recompute：exec 恒持并集 ⊇ home 侧——零字节翻转
            #（无传输；胜者空间准备量 = 0，不可能失败）。
            direction = "reverse"
        else:
            # remote-read：两侧保留量比大小，小的整份搬给大的（与最小
            # 传输量选择恒一致——合并零池写前提下无口径分歧，§4.3.1）。
            direction = (
                "forward"
                if sum(working_shards) <= sum(home_shards)
                else "reverse")

        # ---- 空间准备（I8：统一 T+E 逐出，无 merge 专用驱逐）。 ----
        def _prepare_winner_capacity(
            instance_index: int,
            required_bytes_by_tp_rank: tuple[int, ...],
        ) -> None:
            transfers.extend(self._ensure_capacity(
                instance_index,
                required_bytes_by_tp_rank,
                phase="completion",
                reason="merge_winner_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
            ))

        def _requirement_for(candidate: str) -> tuple[int, tuple[int, ...]]:
            return (
                (home, working_shards)
                if candidate == "forward"
                else (exec_instance, home_shards))

        if direction in ("forward", "reverse") and not zero_byte_flip:
            # 仅 remote-read 的两方向有真实容量需求；零字节翻转/in_place
            # 的准备量 = 0（并集已在胜者侧）。
            fallback = "reverse" if direction == "forward" else "forward"
            try:
                primary_target, primary_required = _requirement_for(direction)
                _prepare_winner_capacity(primary_target, primary_required)
            except KVCapacityError as primary_exc:
                # D7 清账：raise 前已提交的逐出并入返回值（同步图侧），
                # 然后改试另一方向（双向二选一兜底）。
                transfers.extend(primary_exc.evictions)
                try:
                    fallback_target, fallback_required = _requirement_for(
                        fallback)
                    _prepare_winner_capacity(
                        fallback_target, fallback_required)
                except KVCapacityError as fallback_exc:
                    # 双侧深缺口：合同变更终态（2026-09-17）——两个 exc
                    # 的逐 rank 缺口都要落账，fail-closed（热 KV 裁定封死
                    # "执行完再踢出去"的自外迁兜底）。
                    transfers.extend(fallback_exc.evictions)
                    self.commit_deep_gap_records(primary_exc.deep_gap_records)
                    self.commit_deep_gap_records(
                        fallback_exc.deep_gap_records)
                    raise RuntimeError(
                        "merge_back v2 exhausted unified eviction on both "
                        f"directions (session {session_id}, request "
                        f"{trigger_request_id}) -- dual-sided deep gap; "
                        f"primary {direction}: {primary_exc}; "
                        f"fallback {fallback}: {fallback_exc}"
                    ) from fallback_exc
                direction = fallback

        # ---- 传输发射（I6：merge 事务零池写——不产生任何 remote_store）。 ----
        transferred_bytes = 0
        if direction == "forward":
            # 一笔 noc_migrate exec→home，逐 rank 字节 = 账本真值
            # session.shard_bytes（不从 context_tokens 派生——D1）。
            transfer = self._working_copy_noc_transfer(
                session=session,
                trigger_request_id=trigger_request_id,
                source_instance_index=exec_instance,
                target_instance_index=home,
            )
            transfers.append(transfer)
            transferred_bytes = transfer.total_bytes
        elif direction == "reverse" and not zero_byte_flip:
            # 翻转腿：home 侧基础前缀 [0, base_prefix) 经 NoC 搬到 exec。
            # 字节从 _BasePrefixView.context_tokens（= base_history_tokens）
            # 派生——home 侧基础账本即真值。
            transfer = self._noc_transfer(
                phase="completion",
                reason="merge_base_to_exec",
                session=_BasePrefixView(
                    session_id=session_id,
                    instance_index=home,
                    context_tokens=base_history_tokens,
                    resident_prefix_layers=base_prefix,
                ),
                trigger_request_id=trigger_request_id,
                source_instance_index=home,
                target_instance_index=exec_instance,
                layer_start=0,
                layer_end=base_prefix,
            )
            transfers.append(transfer)
            transferred_bytes = transfer.total_bytes

        # ---- 账本结算：统一终态 = 全层 LOCAL@胜者，败者侧释放。 ----
        if direction == "forward":
            winner, loser = home, exec_instance
            # 败者 = exec：释放整份工作副本（含混合形态的池恢复后缀）。
            self._remove_local_shards(exec_instance, working_shards)
            self._metrics_suffix_evict_parts(
                session_id,
                0,
                anchor_kind=_metrics_anchor_for_phase("completion"),
                request_id=trigger_request_id,
                cause="merge_working_copy_release",
                instance_index=exec_instance,
            )
            # 胜者 = home：入账 W。metrics 镜像按 parts 模型拆段——单段
            # tokens×层区间无法精确表达混合形态的 W（后缀复份段冻结于
            # base_history_tokens 的 [p, L)，增量段 [0, L)@增量 token；
            # LOCAL 基 p == L 退化为单增量段，与旧口径同形）。
            self._add_local_shards(home, working_shards)
            if 0 < base_prefix < self.model.layers:
                suffix_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model, base_history_tokens, self.tp_degree,
                    layer_start=base_prefix, layer_end=self.model.layers)
                self._metrics_add_segment(
                    session_id, home, base_history_tokens,
                    base_prefix, self.model.layers, suffix_shards,
                    anchor_kind="completion",
                    request_id=trigger_request_id,
                    cause="merge_working_copy_to_home",
                )
                increment_shards = tuple(
                    working - suffix for working, suffix in zip(
                        working_shards, suffix_shards))
            else:
                increment_shards = working_shards
            if any(increment_shards):
                self._metrics_add_segment(
                    session_id, home, session.context_tokens,
                    0, self.model.layers, increment_shards,
                    anchor_kind="completion",
                    request_id=trigger_request_id,
                    cause="merge_working_copy_to_home",
                )
        else:
            if direction == "reverse":
                # 翻转（含零字节）：胜者 = exec，败者 = home。
                winner, loser = exec_instance, home
                if session.copy_handoff is not None:
                    # C13：copy 的 home 侧基础已在轮内逐块交接释放（不
                    # 等轮末）——轮末零字节结算**不重复释放已交接源块**
                    #（C12 冻结 §2 规则 4）；断言闭合（基础历史最终完整
                    # 落到执行端是 copy 完成条件）。
                    journal = session.copy_handoff
                    if not journal.all_settled():
                        raise RuntimeError(
                            f"copy merge for session {session_id} reached "
                            "settlement with unsettled handoff chunks -- "
                            "the base history never fully landed on the "
                            "execution instance (copy completion requires "
                            "the actual arrival events, not stage names)")
                    if any(journal.home_shards()):
                        raise RuntimeError(
                            f"copy merge for session {session_id} found "
                            "residual home-side bytes after full handoff "
                            f"({journal.home_shards()}) -- ledger drift")
                else:
                    # recompute（或无 journal 的 copy 形态）：败者 = home
                    # 释放基础驻留前缀。
                    self._remove_local_shards(home, home_shards)
                    self._metrics_suffix_evict_parts(
                        session_id,
                        0,
                        anchor_kind=_metrics_anchor_for_phase("completion"),
                        request_id=trigger_request_id,
                        cause="merge_base_release",
                        instance_index=home,
                    )
                if not zero_byte_flip:
                    # 含传输翻转：胜者 exec 入账 H。
                    self._add_local_shards(exec_instance, home_shards)
                    self._metrics_add_segment(
                        session_id, exec_instance, base_history_tokens,
                        0, base_prefix, home_shards,
                        anchor_kind="completion",
                        request_id=trigger_request_id,
                        cause="merge_base_to_exec",
                    )
            else:
                # in_place：无释放无入账无传输（无主，裁定③——工作副本
                # 已在 exec 账上直接转正）。
                winner, loser = exec_instance, home

        # 守恒断言：胜者侧现有字节合计 == 合并后全量（线性算术应精确
        # 相等；不等 = 账本破损/调用方口径漂移，fail-closed）。remote-read
        # 两方向胜者 = 基础＋工作副本之和；零字节翻转（copy/recompute）
        # 与 in_place 胜者持并集 ⊇ 基础——只核对工作副本本身。
        final_tokens = base_history_tokens + new_tokens
        full_shards = kv_cache_shard_bytes_for_tokens(
            self.model, final_tokens, self.tp_degree)
        winner_existing = (
            tuple(home + working for home, working in zip(
                home_shards, working_shards))
            if direction in ("forward", "reverse") and not zero_byte_flip
            else working_shards)
        if winner_existing != full_shards:
            raise RuntimeError(
                f"merge settlement conservation failed for session "
                f"{session_id}: winner-side per-rank bytes {winner_existing} "
                f"!= full-history shards {full_shards} "
                f"(home base {home_shards}, working copy {working_shards}, "
                f"base_history_tokens={base_history_tokens}, "
                f"new_tokens={new_tokens}, "
                f"working context_tokens={session.context_tokens}) -- "
                "ledger integrity failure")

        session.working_kind = None
        session.location = self.LOCAL_HBM
        session.instance_index = winner
        session.resident_prefix_layers = self.model.layers
        session.context_tokens = final_tokens
        session.total_bytes = sum(full_shards)
        session.shard_bytes = full_shards
        # home 迁移（I7）：事务内一次写——merge v2 的第二个赋值点
        #（首轮建立 :3323 之外）。
        session.home_instance = winner
        session.base_location = ""
        session.base_history_tokens = 0
        session.base_resident_prefix_layers = 0
        session.base_shard_bytes = ()
        session.last_merged_request_id = trigger_request_id
        # C13：copy 交接账本轮末闭合（断言闭合 + 科目 close 事件 + 清账）。
        self._close_copy_handoff_journal(session, boundary="merge")
        self.last_merge_outcome = {
            "session_id": session_id,
            "direction": direction,
            "zero_byte_flip": zero_byte_flip and direction == "reverse",
            "winner_instance": winner,
            "loser_instance": loser,
            "transferred_bytes": transferred_bytes,
            "home_flipped": winner != home,
        }
        # C14：kv_delta_journal 结算行（结算时刻事实；home_before 用局部
        # 变量 home——session.home_instance 已在上方迁移赋值为胜者）。
        self._append_kv_delta_row(
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            working_kind=working_kind_at_settlement,
            direction=direction,
            zero_byte_flip=zero_byte_flip and direction == "reverse",
            winner_instance=winner,
            loser_instance=loser,
            home_before=home,
            transferred_bytes=transferred_bytes,
            home_side_retained_bytes=(
                _home_residual_bytes
                if _home_residual_bytes is not None else sum(home_shards)),
            exec_side_retained_bytes=sum(working_shards),
            new_tokens=new_tokens,
        )
        self._check_invariants_after_mutation(session_ids=(session_id,))
        return tuple(transfers)

    def _append_kv_delta_row(
        self,
        *,
        session_id: str,
        trigger_request_id: str,
        working_kind: Optional[str],
        direction: str,
        zero_byte_flip: bool,
        winner_instance: Optional[int],
        loser_instance: Optional[int],
        home_before: Optional[int],
        transferred_bytes: int,
        home_side_retained_bytes: int,
        exec_side_retained_bytes: int,
        new_tokens: int,
    ) -> None:
        """C14 kv_delta_journal 行构造（merge_back 两出口共用；字段面
        = C16 消费面 + 逐请求合并披露，见 __init__ 处接口声明）。

        home_after/home_migration 由 home_before 与 winner 派生（A12'：
        winner 缺席 = stay 等无胜者出口，home 不变——home_after 取
        home_before，不得落 None：消费端把 None 当独立 home 计入集合，
        sessions_with_multiple_homes 乒乓指标假阳性）；
        staging_return_bytes 恒 0——F14 无留存型暂存口径下"暂存归还"
        为无操作（§15.6 披露位，非可变状态）。行一经追加即为冻结事实
        （append-only，不回写）。
        """
        row = {
            "seq": len(self.kv_delta_journal),
            "session_id": session_id,
            "trigger_request_id": trigger_request_id,
            "working_kind": working_kind,
            "direction": direction,
            "zero_byte_flip": zero_byte_flip,
            "winner_instance": winner_instance,
            "loser_instance": loser_instance,
            "home_before": home_before,
            "home_after": (
                winner_instance if winner_instance is not None
                else home_before),
            "home_migration": (
                winner_instance is not None
                and home_before is not None
                and winner_instance != home_before),
            "transferred_bytes": transferred_bytes,
            "home_side_retained_bytes": home_side_retained_bytes,
            "exec_side_retained_bytes": exec_side_retained_bytes,
            "new_tokens": new_tokens,
            "staging_return_bytes": 0,
        }
        self.kv_delta_journal.append(row)
        self._kv_delta_index[trigger_request_id] = row

    def kv_delta_find(self, trigger_request_id: str) -> Optional[dict]:
        """按触发请求 id 取 kv_delta_journal 结算行（最新命中；无则
        None）。C14 结算闭合审计（SH merge watch 闭合门）与 C16 消费
        共用入口。O5：O(1) 索引查询（_kv_delta_index 与账本同源）。"""
        return self._kv_delta_index.get(trigger_request_id)

    def kv_delta_journal_rows(self) -> tuple[dict[str, object], ...]:
        """C14 kv_delta_journal 序列化导出（F3：PROVENANCE §20.3/§22.7-1
        移交义务履行——run 级 sidecar 的行源，与 copy_handoff_events /
        restore_events 同披露纪律：GREEN run 亦保留，非失败台账）。

        行 = 结算时刻冻结事实（append-only，不回写）；导出为逐行 dict
        浅拷贝（消费方改写不回写账本）。字段面 = KV_DELTA_JOURNAL_FIELDS
        （C14 §20.3 冻结口径，见常量处注记）。完整性 fail-closed：字段
        集漂移/seq 断链 = 账本破损（构造面唯一入口 _append_kv_delta_row
        保证 seq 恒为追加序号，到达即破损，不得静默降级）。
        """
        rows: list[dict[str, object]] = []
        for index, row in enumerate(self.kv_delta_journal):
            if tuple(row) != KV_DELTA_JOURNAL_FIELDS:
                raise RuntimeError(
                    f"kv_delta_journal row {index} field set drifted from "
                    f"the C14 frozen schema: {tuple(row)} != "
                    f"{KV_DELTA_JOURNAL_FIELDS} -- ledger integrity failure")
            if row["seq"] != index:
                raise RuntimeError(
                    f"kv_delta_journal seq chain broken at index {index}: "
                    f"seq={row['seq']!r} != {index} -- ledger integrity "
                    f"failure")
            rows.append(dict(row))
        return tuple(rows)

    def observe_completed_input(
        self,
        session_id: str,
        input_tokens: int,
    ) -> None:
        """E 的在线输入长度统计（§5.5：仅已完成轮次入样）。"""
        if input_tokens < 0:
            raise ValueError("completed input tokens must be >= 0")
        self._input_length_stats.setdefault(session_id, []).append(input_tokens)
        self._run_input_count += 1
        self._run_input_sum += input_tokens

    def observe_valid_service_sample(
        self,
        group: str,
        *,
        observed_ratio: float,
        completion_ns: int,
        service_duration_ns: int,
    ) -> float:
        """C15（§5.4）：喂入一个**可分离**的有效服务样本（η/γ 更新）。

        ``observed_ratio`` = 实际量 / 同区间基础模型预测量（分母不含
        修正因子——避免循环校正）；``service_duration_ns`` = 正服务时
        长（τ 来源）。仅已完成或已实际发生的服务可入样；受排队/冷 KV
        等待污染的墙钟**不得**经此入样（调用方保证可分离性——FS 侧
        的完成时刻默认按不可观测处理，见 mark_complete）。
        """
        return self.service_factors.observe_valid_service(
            group,
            observed_ratio=observed_ratio,
            completion_ns=completion_ns,
            service_duration_ns=service_duration_ns)

    def adaptive_decision_find(
        self, session_id: str,
    ) -> Optional[dict]:
        """按会话取最近一条 adaptive 决策披露行（无则 None）。"""
        for row in reversed(self.adaptive_decisions):
            if row["session_id"] == session_id:
                return row
        return None

    def _estimate_next_input(
        self, session_id: str,
    ) -> Optional[int]:
        """session 均值 → run 均值 → None（cold_start_unknown_input）。"""
        samples = self._input_length_stats.get(session_id)
        if samples:
            return sum(samples) // len(samples)
        if self._run_input_count:
            return self._run_input_sum // self._run_input_count
        return None

    def _prefill_ns_per_token_effective(self) -> float:
        if self._prefill_ns_per_token is None:
            self._prefill_ns_per_token = float(
                estimate_prefill_task_load_ns(
                    self.topology.hardware,
                    self.model,
                    instance_size=self.tp_degree,
                    chunk_tokens=1,
                    context_tokens=1,
                )
            )
        return self._prefill_ns_per_token

    def _prefill_chunk_load_ns(self, *, chunk_tokens: int,
                               context_tokens: int) -> int:
        """N4：chunk 级 roofline memo（E 递推整段负载的构造件；与 SH
        _prefill_chunk_task_load_ns 同式——face 侧无 SH 的共享缓存，
        自带 per-instance 小表，键含 (chunk, context) 全形）。"""
        cache = getattr(self, "_prefill_chunk_cache", None)
        if cache is None:
            cache = {}
            self._prefill_chunk_cache = cache
        key = (self.tp_degree, chunk_tokens, context_tokens)
        if key not in cache:
            cache[key] = estimate_prefill_task_load_ns(
                self.topology.hardware,
                self.model,
                instance_size=self.tp_degree,
                chunk_tokens=chunk_tokens,
                context_tokens=context_tokens,
            )
        return cache[key]

    def _adaptive_retention_target(self, session: SessionKVState) -> int:
        return self._adaptive_retention_target_tokens(
            session.session_id, session.context_tokens, session.instance_index)

    def _pool_effective_bytes_per_ns(
        self, instance_index: Optional[int],
    ) -> Optional[float]:
        """P1（R15-3）：池端口仲裁后的有效速率（bytes/ns）。

        注入的 ``pool_divisor_fn(instance_index)`` 给出该实例边缘端口的
        当前仲裁份额（在途池流 + 候选自身 +1）；未配置池速率返回 None
        （保守保留全部 L 层），未注入 divisor_fn 时单流全带宽。
        """
        if self.pool_bandwidth_gbps is None:
            return None
        divisor = 1
        if self._pool_divisor_fn is not None and instance_index is not None:
            divisor = max(1, int(self._pool_divisor_fn(instance_index)))
        return self.pool_bandwidth_gbps / divisor  # 1 GB/s == 1 B/ns

    def _adaptive_reference_ranks(
        self, instance_index: Optional[int],
    ) -> tuple[int, ...]:
        """递推预测的参考 rank 集（instance_index None → 首实例参考，
        与旧解析口径的"无 divisor 注入 = 单流全带宽"语义对齐）。"""
        if instance_index is None:
            return tuple(self.topology.instances[0].ranks)
        return tuple(self.topology.instance(instance_index).ranks)

    def _record_adaptive_decision(
        self,
        session_id: str,
        *,
        k_target: int,
        source: str,
        statuses,
        **extra,
    ) -> None:
        """决策披露侧车（C15：不改 SH 的通道）——预测器来源（递推/解
        析）与覆盖状态逐决策落账；无时间戳（确定性测试友好）。"""
        self.adaptive_decisions.append({
            "session_id": session_id,
            "k_target": int(k_target),
            "source": source,
            "statuses": list(statuses),
            **extra,
        })

    def _adaptive_retention_target_tokens(
        self,
        session_id: str,
        context_tokens: int,
        instance_index: Optional[int] = None,
    ) -> int:
        """adaptive 软保留目标（§5.5 调用 2：等待 session 的预测目标）。

        C15（F12）起由**事件递推预测器**计算——adaptive 的正式在线实
        现替换原解析在线模型（同公式同因果边界：解析例 L=32/c=1ms/
        q=0/r=0.5|2|4ms → k=1/17/25 两径一致）：同一只读资源快照的逐
        层 R/D 递推（写腿×计算 memory 腿同端口仲裁）+ 钉死剪枝（逐
        rank 独享速率串行累计下界、期限钉在无候选恢复流量的消费时刻）。
        腿粒度 = 逐层（与 §5.2 公式同粒度；执行侧按 ≤8 层组门控，组内
        保守方向披露）。输入全为因果可见量：池速率/时延（硬件配置）、
        逐层字节（区间账本口径 kv_cache_shard_bytes_for_layer_range）、
        逐层计算（roofline prefill 均摊）、η/γ 在线因子（§5.4 因果更
        新）。预测器来源与覆盖状态落 adaptive_decisions 侧车。无样本
        /缺池速率仍保守保留全部 L（cold_start_unknown_input）。tokens/
        instance 参数化口径沿基础前缀受限 victim 视图的旧复用（R4 已
        删除，2026-09-17；签名保留供 _ensure_capacity 的 victim 视图）。
        """
        layers = self.model.layers
        if self.layer_policy_mode != "adaptive":
            return layers
        if self.pool_bandwidth_gbps is None:
            return layers
        estimated_input = self._estimate_next_input(session_id)
        if estimated_input is None or estimated_input <= 0:
            self._record_adaptive_decision(
                session_id, k_target=layers, source="cold_start",
                statuses=("cold_start_unknown_input",))
            return layers
        total_context = context_tokens + estimated_input
        ranks = self._adaptive_reference_ranks(instance_index)
        if not ranks:
            return layers
        # ---- 只读资源快照（峰值表 + 在册他流）----
        peaks: dict[str, float] = {}
        for rank in ranks:
            edge = self.nearest_edge(rank)
            peaks[f"pool:{edge}"] = float(self.pool_bandwidth_gbps)
            peaks[f"port:{rank}"] = float(
                self.topology.hardware.local_hbm_bandwidth_gbps)
        pool_divisor = 1
        if self._pool_divisor_fn is not None and instance_index is not None:
            pool_divisor = max(
                1, int(self._pool_divisor_fn(instance_index)))
        committed: list[_CommittedFlow] = []
        if pool_divisor > 1:
            # P1 口径：divisor 含候选自身 +1 → 在册他流 = divisor−1，以
            # 未知 ETA 提交流入快照（状态标注 unknown_release_eta，不当
            # 零代价）；该实例涉及的每个池边缘端口各计入（保守披露）。
            edges = sorted({self.nearest_edge(rank) for rank in ranks})
            for edge_index, edge in enumerate(edges):
                for flow_index in range(pool_divisor - 1):
                    committed.append(_CommittedFlow(
                        flow_id=(
                            f"pool-external-i{instance_index}"
                            f"-e{edge_index}-{flow_index}"),
                        resources=(f"pool:{edge}",),
                        release_eta_ns=None))
        # ---- 恢复腿（逐层；路径 = 池边缘 → 目标端口）与计算段 ----
        legs = []
        for layer_index in range(layers):
            legs.append(RestoreGroupLeg(
                layer_start=layer_index,
                layer_end=layer_index + 1,
                bytes_by_rank=kv_cache_shard_bytes_for_layer_range(
                    self.model, total_context, self.tp_degree,
                    layer_start=layer_index, layer_end=layer_index + 1),
                path_by_rank=tuple(
                    (f"pool:{self.nearest_edge(rank)}", f"port:{rank}")
                    for rank in ranks),
            ))
        per_layer_kv = kv_cache_shard_bytes_for_layer_range(
            self.model, total_context, self.tp_degree,
            layer_start=0, layer_end=1)
        # N4（2026-09-23 复核审计4）：prefill 整段负载改生产同形——
        # PREFILL_CHUNK_SIZE 切分 + 累计 context 逐 chunk roofline 求和
        # （_prefill_ns_per_token_effective 的 context=1/chunk=1 线性
        # 外推丢二次形状：10 输入+90 历史线性 1800 vs 单 chunk roofline
        # 4464）。history 基 = session 现有 context（total_context 的
        # 减侧）；缓存走 _prefill_chunk_load_memo（与 SH 同款）。
        prefill_total_ns = 0
        completed = 0
        while completed < estimated_input:
            chunk_tokens = min(
                PREFILL_CHUNK_SIZE, estimated_input - completed)
            prefill_total_ns += self._prefill_chunk_load_ns(
                chunk_tokens=chunk_tokens,
                context_tokens=context_tokens + completed + chunk_tokens)
            completed += chunk_tokens
        prefill_total_ns = max(1, prefill_total_ns)
        c_j = max(1.0, prefill_total_ns / layers)
        segments = [
            ComputeLayerSegment(
                layer=layer,
                base_ns_by_rank=tuple(c_j for _ in ranks),
                memory_bytes_by_rank=per_layer_kv,
                port_by_rank=tuple(f"port:{rank}" for rank in ranks),
                group="prefill",
            )
            for layer in range(1, layers + 1)
        ]
        snapshot = _ResourceSnapshot(
            peak_bytes_per_ns=peaks,
            committed=tuple(committed),
            port_telemetry=False,
            link_model=False,
        )
        efficiency = _EfficiencyFactors(
            eta={"pool": self.service_factors.value("pool")},
            gamma={"prefill": self.service_factors.value("prefill")},
        )
        try:
            predictor = LayerRecursionPredictor(
                snapshot, legs, segments,
                self.pool_latency_ns, efficiency)
            result = predictor.predict_k_hide()
        except _EventRecursionError:
            # fail-closed：保守保留全部层并披露（不当零代价、不删候选）。
            self._record_adaptive_decision(
                session_id, k_target=layers, source="recursion_error",
                statuses=("recursion_fail_closed",),
                pool_divisor=pool_divisor)
            return layers
        self._record_adaptive_decision(
            session_id,
            k_target=result.k_hide,
            source=result.source,
            statuses=result.statuses,
            analytic_min_k=result.analytic_min_k,
            pool_divisor=pool_divisor,
            exposed_stall_ns=result.selected_trace.exposed_stall_ns,
            compute_slowdown_ns=result.selected_trace.compute_slowdown_ns,
            gamma_prefill=self.service_factors.value("prefill"),
            eta_pool=self.service_factors.value("pool"),
        )
        return result.k_hide


    def allocation_for_session(
        self,
        *,
        session_id: str,
        request_id: str,
    ) -> KVAllocation:
        session = self._sessions[session_id]
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("Decode allocation must be local")
        instance = self.topology.instance(session.instance_index)
        return KVAllocation(
            request_id=request_id,
            decode_instance_index=session.instance_index,
            total_bytes=session.total_bytes,
            pieces=(
                KVAllocationPiece(
                    instance_index=session.instance_index,
                    bytes=session.total_bytes,
                    weighted_distance=0.0,
                    path=(session.instance_index,),
                    rank_bytes=tuple(zip(instance.ranks, session.shard_bytes)),
                ),
            ),
        )

    def mark_complete(
        self,
        session_id: str,
        completion_ns: int,
        next_request_type: Optional[str] = None,
    ) -> None:
        if next_request_type not in (None, "human", "tool"):
            raise ValueError(
                "next_request_type must be None, 'human' or 'tool', got "
                f"{next_request_type!r}"
            )
        session = self._sessions[session_id]
        if session.working_kind is not None:
            # merge 事务必须先于完成标记（结算边界不变量）。09-21 新
            # §2.3 合同序：service_done（响应完成）在前、merge_done
            # （结算完成）在后；本标记是 merge_done 之后的收尾点
            # （session 转为可逐出的 inactive），不是响应完成锚——
            # 到达时工作副本必须已结算（无未合并副本标记完成）。
            raise RuntimeError(
                "session completed with an unmerged working copy")
        if session.location not in {
            self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE, self.REMOTE_MEMORY,
        }:
            raise RuntimeError("completed request KV has an invalid location")
        if (
            session.location != self.REMOTE_MEMORY
            and session.instance_index is None
        ):
            raise RuntimeError("resident completed session has no instance")
        if session.restore_journal is not None and not (
                session.restore_journal.all_consumed()):
            # C15：直连 API 调用面可能不经 prefill drain（在线路径的
            # 正常结算边界）——完成标记兜底结算（因果上 completion 晚
            # 于 drain/merge，消费保证已兑现；事件按 boundary=
            # mark_complete 落账披露，与 C13 merge 兜底同款语义）。
            self._settle_restore_groups(
                session_id, boundary="mark_complete")
        if session.restore_journal is not None:
            raise RuntimeError(
                f"session {session_id} completed with unsettled restore "
                "groups (settlement failed at every boundary)")
        session.active = False
        session.last_completion_ns = completion_ns
        # C15（§5.4 在线效率更新）：完成时刻的墙钟含排队与冷 KV 等待的
        # 混合等待——无可分离服务区间时不入样：保持原估计并标记不可观
        # 测（不把受混合等待污染的墙钟当纯服务样本）。可分离样本经
        # observe_valid_service_sample 因果喂入。
        self.service_factors.mark_unobservable("prefill")
        self.service_factors.mark_unobservable("pool")
        # Record the idle session's trigger class so the typed eviction
        # order sees it (the event loop batches mark_complete before later
        # admissions).
        session.next_request_type = next_request_type
        self._check_invariants_after_mutation(session_ids=(session_id,))

    def retire_terminal_session(
        self,
        session_id: str,
        completion_ns: int,
        request_id: Optional[str] = None,
    ) -> Optional[int]:
        """Forget terminal KV while preserving the partial/remote state model.

        No eviction transfer is emitted here.  Only the prefix that is still
        physically resident in local HBM is decremented; remote-only bytes are
        discarded with their terminal session metadata.
        """

        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown KV session: {session_id}")
        if session.active or session.last_completion_ns is None:
            raise RuntimeError(
                "only inactive completed sessions may be terminally retired"
            )
        if request_id is not None:
            requested_reservation = self._reservations.get(request_id)
            if (
                requested_reservation is not None
                and requested_reservation.session_id != session_id
            ):
                raise RuntimeError(
                    "terminal retirement request owns another session reservation"
                )
        terminal_request_id = request_id or session_id

        reservation_ids = sorted(
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if reservation.session_id == session_id
        )
        for reservation_id in reservation_ids:
            reservation = self._reservations[reservation_id]
            reservation_extra = self._reservation_extra_shards(reservation)
            if self._metrics_recorder is not None:
                for rank, value in zip(
                    self.topology.instance(reservation.instance_index).ranks,
                    reservation_extra,
                ):
                    if not value:
                        continue
                    self._metrics_recorder.record(
                        anchor_kind="completion",
                        request_id=reservation_id,
                        session_id=session_id,
                        rank=rank,
                        allocation_key=f"reservation:{reservation_id}",
                        reserved_kv_delta_bytes=-int(value),
                        cause="terminal_session_retire_reservation",
                    )
            del self._reservations[reservation_id]

        released_instance_index: Optional[int] = None
        if session.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}:
            if session.instance_index is None:
                raise RuntimeError("local completed session has no instance")
            released_instance_index = session.instance_index
            local_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.context_tokens,
                self.tp_degree,
                layer_start=0,
                layer_end=session.resident_prefix_layers,
            )
            self._remove_local_shards(released_instance_index, local_shards)
        elif session.location != self.REMOTE_MEMORY:
            raise RuntimeError(f"invalid KV location for {session_id}")

        self._metrics_remove_session_parts(
            session_id,
            anchor_kind="completion",
            request_id=terminal_request_id,
            cause="terminal_session_retire",
        )
        # Helpers intentionally no-op without a recorder; remove the outer
        # ownership maps regardless so one-shot sessions cannot accumulate.
        self._metrics_parts.pop(session_id, None)
        self._metrics_segment_counters.pop(session_id, None)
        del self._sessions[session_id]
        self._check_invariants_after_mutation(
            session_ids=(session_id,), reservation_ids=reservation_ids
        )
        return released_instance_index
