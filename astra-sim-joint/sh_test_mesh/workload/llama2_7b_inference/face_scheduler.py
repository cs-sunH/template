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
import time
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence

from joint.capacity_summary import (
    CapacityBucket,
    CapacityBucketKey,
    CapacitySummary,
    InstanceCapacitySummary,
    SUFFIX_CLASS_ORDER,
    SUFFIX_EASY,
    SUFFIX_REST,
    CATEGORY_ALL,
    CAPACITY_SUMMARY_ERROR_SOURCES,
    shard_ratio_signature,
)
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
    PublishedServiceFactorGroup,
    ResourceSnapshot as _ResourceSnapshot,
    CommittedFlow as _CommittedFlow,
    RestoreGroupLeg,
    ServiceFactorGroup,
)
from joint.kv_location_directory import (
    ENTRY_STATE_EVICTED_TO_POOL,
    ENTRY_STATE_READY,
    ENTRY_STATE_RESTORING,
    ENTRY_STATE_WRITING,
    POOL_HOLDER,
    ROLE_BASE,
    ROLE_BOUNDARY_SOURCE,
    ROLE_POOL_BACKED,
    ROLE_RESIDENT,
    ROLE_WORKING,
    KvContentKey,
    KvDirectoryEntry,
    KvLocationDirectory,
)
from joint.distributed_settlement import (
    DirectoryOverheadMeter,
    DistributedSettleOutcome,
    RetainedSegment,
    assert_full_content_locatable,
    assert_no_dangling_session_state,
    enumerate_source_surface,
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

    @property
    def size(self) -> int:
        return len(self.ranks)


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
        raise RuntimeError(
            f"model weight matrix shards exceed the global estimate "
            f"(layers={model.layers}, tp_degree={tp_degree})")
    residual_shards = partition_bytes_exact(residual_bytes, tp_degree)
    shards = tuple(
        matrix_bytes + residual_bytes_for_rank
        for matrix_bytes, residual_bytes_for_rank in zip(
            matrix_shards,
            residual_shards,
        )
    )
    if sum(shards) != total_weight_bytes:
        raise RuntimeError(
            f"model weight TP shards do not preserve total bytes "
            f"(layers={model.layers}, tp_degree={tp_degree})")
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


@dataclass(frozen=True)
class FaceWorkProfile:
    """One forward's pure FLOP floor and per-rank HBM byte work.

    ``pure_compute_ns`` deliberately excludes the roofline bandwidth term and
    protocol latency.  ``hbm_bytes_per_rank`` is the complete model-weight,
    activation, and attention-memory traffic assigned to one equal-size TP
    rank.  Consumers that need the isolated roofline time must combine these
    two fields through the resource projection, rather than treating the
    roofline ``max`` as pure compute.
    """

    pure_compute_ns: int
    hbm_bytes_per_rank: int
    # The FACE roofline has one local-HBM startup for the linear phase and
    # one for the attention phase.  Keep those fixed legs separate from both
    # the FLOP floor and byte work; callers attach them to the phase graph.
    hbm_latency_ns: int = 0
    controller_latency_ns: int = 0


def _iteration_work_terms(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
    p_context_tokens: Optional[int] = None,
) -> tuple[int, int, int, int, int, int, float]:
    """Return shared FLOP/byte terms for the FACE iteration model.

    The tuple is ``(linear_ops, prefill_attn_ops, decode_attn_ops,
    linear_bytes, prefill_attn_bytes, decode_attn_bytes, aggregate_perf)``.
    Keeping this decomposition beside ``estimate_iteration_time_ns`` makes
    the online pure-compute/HBM profile use exactly the same weight, KV, and
    activation model as FACE's roofline estimator.
    """
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

    h = model.hidden_size
    ffn = model.ffn_size
    layers = model.layers
    bytes_per_elem = model.bytes_per_elem
    aggregate_perf = instance_size * hardware.peak_perf_tflops * 1e12
    linear_tokens = p_chunk + d_batch
    mlp_ops_per_token = (
        6 * h * ffn if model.mlp_variant == "swiglu" else 4 * h * ffn)
    linear_ops_per_token = layers * (8 * h * h + mlp_ops_per_token)
    linear_ops = linear_tokens * linear_ops_per_token
    weight_bytes = estimate_model_weight_bytes(model)
    activation_bytes = layers * linear_tokens * h * bytes_per_elem * 8
    linear_bytes = weight_bytes + activation_bytes

    prefill_attn_ops = layers * 4 * p_chunk * max(1, p_context_tokens) * h
    prefill_attn_bytes = (
        layers * p_chunk * max(1, p_context_tokens)
        * bytes_per_elem * 4)
    decode_attn_ops = layers * 4 * d_batch * d_token * h
    decode_attn_bytes = (
        layers * 2 * d_batch * d_token * h * bytes_per_elem)
    return (
        linear_ops, prefill_attn_ops, decode_attn_ops, linear_bytes,
        prefill_attn_bytes, decode_attn_bytes, aggregate_perf)


def estimate_iteration_workload(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
    p_context_tokens: Optional[int] = None,
) -> FaceWorkProfile:
    """Return FACE's pure-compute/HBM decomposition for one iteration.

    This is the profile consumed by online JCM callbacks.  It returns the
    FLOP-only duration and complete HBM bytes per TP rank; bandwidth sharing
    and the resulting roofline completion remain the projection engine's job.
    """
    terms = _iteration_work_terms(
        hardware, model, instance_size=instance_size, p_chunk=p_chunk,
        d_batch=d_batch, d_token=d_token,
        p_context_tokens=p_context_tokens)
    (linear_ops, prefill_attn_ops, decode_attn_ops, linear_bytes,
     prefill_attn_bytes, decode_attn_bytes, aggregate_perf) = terms
    if p_chunk == 0 and d_batch == 0:
        return FaceWorkProfile(0, 0)
    attention_ops = max(
        prefill_attn_ops if p_chunk else 0.0,
        decode_attn_ops if d_batch else 0.0)
    attention_bytes = max(
        prefill_attn_bytes if p_chunk else 0.0,
        decode_attn_bytes if d_batch else 0.0)
    pure_compute_ns = max(
        1, math.ceil((linear_ops + attention_ops) / aggregate_perf * 1e9))
    total_hbm_bytes = linear_bytes + attention_bytes
    hbm_bytes = max(
        0, (total_hbm_bytes + instance_size - 1) // instance_size)
    return FaceWorkProfile(
        pure_compute_ns,
        hbm_bytes,
        hbm_latency_ns=2 * int(hardware.local_hbm_latency_ns),
        controller_latency_ns=int(hardware.d2d_latency_ns),
    )


def estimate_prefill_workload(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    input_tokens: int,
    history_tokens: int,
    p_chunk: int,
) -> FaceWorkProfile:
    """Return a bounded, exact-work profile for a chunked prefill span.

    The service-time estimator uses the same chunking shape (first chunk,
    bounded middle fold, tail).  Work bytes and FLOPs are additive physical
    work, so their arithmetic progression is summed in closed form here;
    profile construction does not loop over a potentially long input.
    """
    if (instance_size <= 0 or input_tokens < 0 or history_tokens < 0
            or p_chunk <= 0):
        raise ValueError("invalid Prefill workload parameters")
    if input_tokens == 0:
        return FaceWorkProfile(0, 0)
    chunk_tokens = min(p_chunk, input_tokens)
    full_chunks, tail_tokens = divmod(input_tokens, chunk_tokens)

    first = _iteration_work_terms(
        hardware, model, instance_size=instance_size,
        p_chunk=chunk_tokens, d_batch=0, d_token=0,
        p_context_tokens=history_tokens + chunk_tokens)
    (first_linear_ops, first_prefill_ops, _first_decode_ops,
     first_linear_bytes, first_prefill_bytes, _first_decode_bytes,
     aggregate_perf) = first
    total_ops = full_chunks * (first_linear_ops + first_prefill_ops)
    total_bytes = full_chunks * (
        first_linear_bytes + first_prefill_bytes)
    if full_chunks > 1:
        second = _iteration_work_terms(
            hardware, model, instance_size=instance_size,
            p_chunk=chunk_tokens, d_batch=0, d_token=0,
            p_context_tokens=history_tokens + 2 * chunk_tokens)
        delta_ops = (
            second[0] + second[1] - first_linear_ops - first_prefill_ops)
        delta_bytes = (
            second[3] + second[4] - first_linear_bytes - first_prefill_bytes)
        progression = full_chunks * (full_chunks - 1) // 2
        total_ops += progression * delta_ops
        total_bytes += progression * delta_bytes
    if tail_tokens > 0:
        tail = _iteration_work_terms(
            hardware, model, instance_size=instance_size,
            p_chunk=tail_tokens, d_batch=0, d_token=0,
            p_context_tokens=history_tokens + input_tokens)
        total_ops += tail[0] + tail[1]
        total_bytes += tail[3] + tail[4]
    chunk_count = full_chunks + int(tail_tokens > 0)
    return FaceWorkProfile(
        max(1, math.ceil(total_ops / aggregate_perf * 1e9)),
        max(0, (total_bytes + instance_size - 1) // instance_size),
        hbm_latency_ns=(2 * int(hardware.local_hbm_latency_ns)
                        * int(chunk_count)),
        controller_latency_ns=(int(hardware.d2d_latency_ns)
                               * int(chunk_count)),
    )


def estimate_iteration_time_ns(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
    p_context_tokens: Optional[int] = None,
    hbm_share_divisor: int = 1,
) -> int:
    """Roofline 服务时间（本地算子求值器，v2.0 式(8)）。

    D06（2026-09-27，v2.0 §9.1 式(6)）：``hbm_share_divisor`` = 该算子
    所在候选阶段的 HBM 端口份额分母 m_e + k_e（在册后台流 + 候选阶段
    并发逻辑流，计算读恒为其中一条流）——HBM 带宽项按共享份额
    B/(m_e + k_e) 计价，计算读与同端口的 copy 写/pool 恢复写/remote
    暂存读写同一分母，不各自独占整端口。缺省 1 = 独占带宽（零漂移）；
    值必须为正整数（fail-closed）。峰值 FLOP 项与时延项不受份额影响。
    """
    if instance_size <= 0 or p_chunk < 0 or d_batch < 0 or d_token < 0:
        raise ValueError("invalid Roofline workload parameters")
    if (
        isinstance(hbm_share_divisor, bool)
        or not isinstance(hbm_share_divisor, int)
        or hbm_share_divisor < 1
    ):
        raise ValueError("hbm_share_divisor must be a positive integer")
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

    terms = _iteration_work_terms(
        hardware, model, instance_size=instance_size, p_chunk=p_chunk,
        d_batch=d_batch, d_token=d_token,
        p_context_tokens=p_context_tokens)
    (linear_ops, prefill_attn_ops, decode_attn_ops, linear_bytes,
     prefill_attn_bytes, decode_attn_bytes, aggregate_perf) = terms
    aggregate_bw = instance_size * hardware.local_hbm_bandwidth_gbps * 1e9
    local_hbm_latency_seconds = hardware.local_hbm_latency_ns / 1e9
    # D06 式(6)：阶段份额带宽（TP 聚合口径下每 rank 端口同分母）。
    shared_bw = aggregate_bw / hbm_share_divisor

    linear_seconds = max(
        linear_ops / aggregate_perf,
        local_hbm_latency_seconds + linear_bytes / shared_bw,
    )

    prefill_seconds = 0.0
    if p_chunk:
        prefill_seconds = max(
            prefill_attn_ops / aggregate_perf,
            local_hbm_latency_seconds + prefill_attn_bytes / shared_bw,
        )

    decode_seconds = 0.0
    if d_batch:
        decode_seconds = max(
            decode_attn_ops / aggregate_perf,
            local_hbm_latency_seconds + decode_attn_bytes / shared_bw,
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
    hbm_share_divisor: int = 1,
) -> int:
    """Return Roofline service-time load for one Prefill chunk.

    ``context_tokens`` is the complete KV-visible context at the end of the
    chunk, so attention work includes both prior session history and earlier
    chunks from the current request.  ``hbm_share_divisor`` (D06, v2.0
    式(6)) is the candidate-stage HBM port share denominator m_e + k_e;
    default 1 keeps the exclusive-bandwidth pricing.
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
        hbm_share_divisor=hbm_share_divisor,
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
    hbm_share_divisor: int = 1,
) -> int:
    """Estimate one active Decode request's remaining Roofline load.

    Future output length is the arithmetic mean supplied by the workload, not
    the request's known trace completion length.  Once ``generated_tokens`` is
    at or beyond that mean, the estimated remaining load is zero.  A fractional
    mean contributes the corresponding fraction of its final expected token.
    ``hbm_share_divisor`` (D06, v2.0 式(6)) is the candidate-stage HBM port
    share denominator m_e + k_e; default 1 keeps exclusive-bandwidth pricing.
    """

    if instance_size <= 0:
        raise ValueError("instance_size must be positive")
    if (
        isinstance(hbm_share_divisor, bool)
        or not isinstance(hbm_share_divisor, int)
        or hbm_share_divisor < 1
    ):
        raise ValueError("hbm_share_divisor must be a positive integer")
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
            hbm_share_divisor=hbm_share_divisor,
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
    # 逻辑 home：首轮发射建立；异地执行（copy/recompute）与
    # 逐出到片外都不改变 home（§2.1）。字段缺省 None 仅为 dataclass 默认
    # 值兼容；在线运行中首个 prepare_prefill 一定会建立。
    home_instance: Optional[int] = None
    # 跨实例执行的工作副本记账（§2.2 表）：执行 instance 持有工作副本时，
    # location/instance_index/shard_bytes 描述**工作副本**（本地全量、随
    # 执行增长），base_* 描述 home 侧权威基础历史的驻留状态；merge_back
    # 后清空并恢复 base。working_kind：
    #   "copy"       基础历史整份复制到执行端 + 增量
    #   "recompute"  执行端重算历史 + 增量（D10，v2.0 §6.2：完整 H+U
    #                重放、不依赖旧 KV——base 驻留字节仅作重建空间复用，
    #                非权威上下文来源）
    working_kind: Optional[str] = None
    base_history_tokens: int = 0
    base_resident_prefix_layers: int = 0
    base_location: str = ""
    # 物理基础驻留实例（D-BI，2026-09-26）：工作副本事务准备相
    # （prepare_prefill._snapshot_base）快照的 base 驻留所在 instance——
    # base_* 字段描述的权威基础历史物理落在该实例的 rank 台账上。
    # 与逻辑 home_instance 解耦：P→D 迁移（move_prefill_to_decode）与
    # merge stay 提交会让会话终驻偏离逻辑 home（§2.1 home 粘滞；设计
    # 文档 ω_B 仅平局用），下一轮跨实例事务的基础可不在 home。池基
    # （REMOTE 基/前缀 0）无片上驻留 → None 合法（merge 走 in_place）。
    base_instance_index: Optional[int] = None
    # merge 事务版本键（R4/N9，2026-09-14）：最近一次已结算的
    # merge_back 触发请求；同请求重复 merge 即合同类违规 fail-closed。
    last_merged_request_id: Optional[str] = None
    # C13 copy 逐 chunk 四步交接账本（设计文档 §2.3；None = 本轮非
    # copy 跨实例或已闭合）。轮内权威：home 侧残量的唯一事实源。
    copy_handoff: Optional["CopyHandoffJournal"] = None
    # C15 后缀逐组恢复区间账本（守恒科目 rid#restore；None = 本轮无
    # 后缀池恢复）。issue 于准入相，complete 于 prefill drain/merge。
    restore_journal: Optional["RestoreGroupJournal"] = None
    # ---- D19/C4（多源边界来源区间）----
    # D 边界动作（recompute@j≠i_P）**不迁移** i_P 侧旧工作副本：旧副本转
    # 为与 base_* 同语义的第二来源记录（片上位置 ≤3 界的第三位——home、
    # i_P、D 新目标 j）。设定点 =
    # ``commit_boundary_decode_transition``；清账点 = 完成结算
    # （``merge_back`` 的三位置确定性合并）。空串/None = 无第二来源。
    boundary_source_history_tokens: int = 0
    boundary_source_shard_bytes: tuple[int, ...] = ()
    boundary_source_instance_index: Optional[int] = None
    boundary_source_resident_prefix_layers: int = 0
    boundary_source_location: str = ""
    # R5（修复轮三，2026-09-27）：来源侧完整并集标志——i_P 侧旧副本本身
    # 是 P=copy/recompute 的完整副本（物理持有 base＋U 完整 token 域，⊇
    # base_* 记录的基础历史）。三位置结算据此把 base 位置按重复副本零
    # 载荷释放（对称 complete_union 的重复副本处置，§15.1"重复副本不
    # 增加 V"），不得把 base 字节再并入 winner 侧（否则守恒式把基础
    # 计两遍）。设定点 =
    # ``commit_boundary_decode_transition`` 的来源转记分支；清账点 =
    # 三位置结算（与 boundary_source_* 同生命周期）。
    boundary_source_complete_union: bool = False
    # ---- W9（§7.4 distributed 跨轮留存）----
    # 目录提交型结算标记：首次 settle_distributed 置位，run 内不复位
    # （distributed 一次 run 固定）；retire_terminal_session 清理。置位后
    # prepare_prefill 的历史元数据校验切换为目录权威覆盖断言（有效内容
    # 按区间并集 == 期望并集字节），跨实例动作的 base 捕获改自 base_*。
    distributed_layout: bool = False
    # distributed settle 版本键（与 last_merged_request_id 同语义：一请求
    # 恰一次结算；两字段分立——policy 一次 run 固定，不混用防口径漂移）。
    last_settled_request_id: Optional[str] = None

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


def plan_layer_groups(
    start: int, end: int, group_layers: int = RESTORE_GROUP_LAYERS,
) -> tuple[tuple[int, int], ...]:
    """层区间 [start, end) 的消费顺序层组切分（公共规划器）。

    C15 后缀恢复组 / C13 copy 交接块 同款确定性规划：组数 n =
    ceil(跨度 / group_layers)，组内跨度均衡（span = ceil(跨度 / n)）；
    返回 [(group_start, group_end), ...]
    （层自低向高 = 消费顺序）。确定性、无运行期状态输入——前缀组与
    后缀组切分方式一致，方便构图器铺层段。
    """
    span_total = end - start
    if span_total <= 0:
        return ()
    group_count = -(-span_total // group_layers)
    span = -(-span_total // group_count)
    ranges: list[tuple[int, int]] = []
    cursor = start
    while cursor < end:
        group_end = min(cursor + span, end)
        ranges.append((cursor, group_end))
        cursor = group_end
    return tuple(ranges)


def plan_suffix_restore_groups(
    suffix_start: int, model_layers: int,
) -> tuple[tuple[int, int], ...]:
    """C15：后缀 [suffix_start, model_layers) 的消费顺序层组切分。

    plan_layer_groups 的后缀恢复特化（group_layers =
    RESTORE_GROUP_LAYERS）：后缀 ≤ RESTORE_GROUP_LAYERS 时单组 = 旧
    单笔口径回归锚。
    """
    return plan_layer_groups(suffix_start, model_layers)


def plan_copy_handoff_layer_chunks(
    base_prefix_layers: int,
) -> tuple[tuple[int, int], ...]:
    """C13：copy 驻留前缀 [0, base_prefix) 的消费顺序层块切分。

    plan_layer_groups 的 copy 交接特化（group_layers =
    COPY_HANDOFF_CHUNK_LAYERS，与 RESTORE_GROUP_LAYERS 同值）：消费
    顺序 = 层自 0 向上（transformer 深度序）。
    """
    return plan_layer_groups(0, base_prefix_layers, COPY_HANDOFF_CHUNK_LAYERS)


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

    def settle(self, boundary: str, *, at_ns: Optional[int] = None) -> None:
        """结算边界：全部已登记组置 CONSUMED（重复结算 fail-closed）。

        M4（2026-09-26）：complete 事件携带可选 ``at_ns``（结算时刻；
        缺省 None 保确定性测试）——与 issue 事件的 ``at_ns`` 配对得
        恢复窗口下界（issue→complete 时刻差）。
        """
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
            "at_ns": at_ns,
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


@dataclass(frozen=True)
class StageReservationFootprint:
    """D20（v2.0 §14.5 式 20a）：阶段承诺足迹（reservation resize 单源）。

    P→D 边界阶段事务按 D 阶段动作与目标实例重新推导的逐 rank 承诺：
    ``basis``（token 基数 = history+input 全量，``reserve_request_capacity``
    同口径）。与准入 ``reserve_request_capacity`` 同一套单源助手派生，
    防口径分叉。
    """

    action: str
    target_instance_index: int
    basis_context_tokens: int
    basis_shard_bytes: tuple[int, ...]

    def committed_shard_bytes(self) -> tuple[int, ...]:
        """承诺总量（basis，逐 rank 直和）。"""
        return tuple(int(value) for value in self.basis_shard_bytes)


def _stage_retained_obligations(
        evictions: tuple["KVTransfer", ...]) -> tuple[dict, ...]:
    """A22（§7.2）：不可撤销物理子义务描述子（kind/身份/逐 rank 字节）
    ——从已提交 T+E 逐出转移派生（真实容量事件；成功/失败回执分记）。"""
    descriptors = []
    for transfer in evictions:
        descriptors.append({
            "kind": "eviction_transfer",
            "source_instance_index": transfer.source_instance_index,
            "target_instance_index": transfer.target_instance_index,
            "bytes_by_rank": tuple(
                int(shard.bytes) for shard in transfer.shards),
            "reason": transfer.reason,
        })
    return tuple(descriptors)


class KVStageVersionConflict(ValueError):
    """W5（§7.2 事务顺序第 1 步）：提交点版本校验失败——决策时点与提交
    点之间 layout（目录）/reservation（阶段纪元）版本漂移（同 tick 竞争
    的其他容量事务先提交）。调用方按 §7.2 刷新快照后重评（或回退继续
    方案），不得沿用决策时点的容量承诺。"""


@dataclass(frozen=True)
class KVStageTransaction:
    """D20（v2.0 §14.5/§16.4）+ W5（§7.2 A22/attempt）扩展：跨阶段预约
    事务回执（幂等 epoch 键定）。

    一次 P→D 边界阶段提交的完整记录：预约换址（transfer）+ 承诺
    resize（式 20a：``Z^{D,add} = max(0, P^{D,peak} − O^{self,reuse} −
    Z^{self,transferable})`` 的逐 rank 缺口，经 ``_ensure_capacity``
    真实 T+E）+ 已提交逐出与不可逆位。

    幂等语义：同 ``(request_id, stage_epoch)`` 的重复提交原子重放本
    回执（不重复换址、不重复 resize、不重复逐出）；epoch 回退或同
    epoch 目标/动作冲突的重放 fail-closed。``aborted=True`` 的回执
    表示换址半边已按 §16.4"提交失败且无不可逆工作时保留继续方案"
    回退到源实例（回退后同 epoch 重放同样命中本回执，不再重做）。

    irreversible：本事务是否已发生不可逆工作（真实 T+E 逐出 = 物理
    容量事件，照常入账不回滚）。
    """

    request_id: str
    session_id: str
    stage_epoch: int
    stage_action: str
    source_instance_index: int
    target_instance_index: int
    reserved_shard_bytes: tuple[int, ...]
    # claim 对 claim 差值披露（本事务 reserved − 前账承诺的逐 rank 差），
    # 非物理释放字节；物理释放以逐出 transfer 与换址账本为准。
    released_shard_bytes: tuple[int, ...]
    evictions: tuple[KVTransfer, ...] = ()
    irreversible: bool = False
    aborted: bool = False
    # ---- W5（§7.2）扩展 ----
    # attempt 维度：同 (request_id, stage_epoch) 内的尝试序号（幂等
    # 重放命中同回执不递增；首次提交 = 0）。
    attempt: int = 0
    # A22：与"阶段成功提交"分记的不可撤销物理子义务（保留身份/剩余量
    # /容量与源保护/释放依赖，供后续候选继承——aborted 回执的 evictions
    # 即该集合；描述子 {kind, instance_index, bytes_by_rank}）。
    retained_obligations: tuple[dict, ...] = ()

    def __post_init__(self) -> None:
        if self.stage_epoch < 0:
            raise ValueError("stage transaction epoch must be non-negative")
        if self.action_stage() not in {
            "stay", "copy", "recompute",
        }:
            raise ValueError(
                f"unknown stage transaction action: {self.stage_action!r}")
        if self.aborted and self.target_instance_index != (
                self.source_instance_index):
            raise ValueError(
                "aborted stage transaction must target the source instance")
        if len(self.reserved_shard_bytes) != len(self.released_shard_bytes):
            raise ValueError(
                "stage transaction reserved/released vectors must match")

    def action_stage(self) -> str:
        return self.stage_action


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
    # None = 非交接块（stay/recompute 腿、copy 的池恢复后缀
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


class KVTransferSequence:
    """Indexable transfer range that constructs one KVTransfer on demand.

    Planner state keeps only frozen algebraic inputs and a bounded range
    length. Slices are an explicit materialization request for legacy
    inspection callers; production code can use integer indexing/iteration.
    """

    __slots__ = ("_count", "_factory", "_flow_representatives", "total_bytes")

    def __init__(self, count, factory, *, flow_representatives=(),
                 total_bytes=None):
        count = int(count)
        if count < 0:
            raise ValueError("KV transfer sequence count must be nonnegative")
        self._count = count
        self._factory = factory
        self._flow_representatives = tuple(flow_representatives)
        self.total_bytes = None if total_bytes is None else int(total_bytes)

    def __len__(self):
        return self._count

    def __bool__(self):
        return self._count > 0

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(self._count)
            return tuple(self._factory(i) for i in range(start, stop, step))
        index = int(index)
        if index < 0:
            index += self._count
        if not 0 <= index < self._count:
            raise IndexError(index)
        return self._factory(index)

    def __iter__(self):
        for index in range(self._count):
            yield self._factory(index)

    def flow_representatives(self):
        """Return bounded per-source examples for serial-flow registration."""
        return self._flow_representatives

    def __eq__(self, other):
        if isinstance(other, (tuple, list, KVTransferSequence)):
            return len(self) == len(other) and all(
                left == right for left, right in zip(self, other))
        return NotImplemented


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


class _CapacityBucketAggregate:
    """D11：单桶可变聚合（增量维护；max_atom 经逐值多重集精确回收）。

    字段序 = (R, clean, dirty, 原子个数, 最大原子)，逐 rank 向量；
    ``_atom_multisets[rank][atom_bytes] = 计数`` 支撑减法侧精确维护
    最大原子（减到 0 计数才回收该值——最大值重扫仅在顶值耗尽时发生）。
    """

    __slots__ = ("reclaimable", "clean", "dirty", "atom_count", "max_atom",
                 "_atom_multisets")

    def __init__(self, size: int) -> None:
        self.reclaimable = [0 for _ in range(size)]
        self.clean = [0 for _ in range(size)]
        self.dirty = [0 for _ in range(size)]
        self.atom_count = [0 for _ in range(size)]
        self.max_atom = [0 for _ in range(size)]
        self._atom_multisets: list[dict[int, int]] = [
            {} for _ in range(size)]

    def add(self, bundle: tuple[tuple[int, ...], ...]) -> None:
        reclaimable, clean, dirty, atoms, max_atoms = bundle
        for rank, value in enumerate(reclaimable):
            self.reclaimable[rank] += value
        for rank, value in enumerate(clean):
            self.clean[rank] += value
        for rank, value in enumerate(dirty):
            self.dirty[rank] += value
        for rank, value in enumerate(atoms):
            self.atom_count[rank] += value
        for rank, value in enumerate(max_atoms):
            if value <= 0:
                continue
            self.max_atom[rank] = max(self.max_atom[rank], value)
            multiset = self._atom_multisets[rank]
            multiset[value] = multiset.get(value, 0) + 1

    def remove(self, bundle: tuple[tuple[int, ...], ...]) -> None:
        reclaimable, clean, dirty, atoms, max_atoms = bundle
        for rank, value in enumerate(reclaimable):
            self.reclaimable[rank] -= value
            if self.reclaimable[rank] < 0:
                raise RuntimeError(
                    "capacity summary reclaimable went negative on removal")
        for rank, value in enumerate(clean):
            self.clean[rank] -= value
        for rank, value in enumerate(dirty):
            self.dirty[rank] -= value
        for rank, value in enumerate(atoms):
            self.atom_count[rank] -= value
        for rank, value in enumerate(max_atoms):
            if value <= 0:
                continue
            multiset = self._atom_multisets[rank]
            remaining = multiset.get(value, 0) - 1
            if remaining > 0:
                multiset[value] = remaining
                continue
            if remaining < 0:
                raise RuntimeError(
                    "capacity summary atom multiset went negative on removal")
            del multiset[value]
            if self.max_atom[rank] == value:
                self.max_atom[rank] = max(multiset, default=0)

    def is_empty(self) -> bool:
        return not any(self.reclaimable)

    def snapshot(self) -> tuple[tuple[int, ...], ...]:
        return (
            tuple(self.reclaimable),
            tuple(self.clean),
            tuple(self.dirty),
            tuple(self.atom_count),
            tuple(self.max_atom),
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
        service_factor_source=None,
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
        # coarse_die_v2 transient COMP workspace, supplied by the C++ rank
        # ledgers at each decision boundary. This is a scalar external hold;
        # it does not enter the JOINT KV directory or duplicate its owners.
        self._external_workspace_revision_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._external_workspace_bytes_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        # ---- W4（§6.1）：唯一权威区间目录（镜像层） ----
        # 目录与旧字段同事务更新（写点收拢于 _check_invariants_after_
        # mutation 咽喉——本类 §1.9 全部写点均以该检查收尾）；逐 rank
        # 守恒对账见 _kv_directory_audit。model_version 维挂
        # MODEL_SEMANTICS_VERSION 纪律（单模型仓常量，字段保留——多模
        # 型仓升级只换常量来源）。目录不自带第二套容量真值：实例容量
        # 仍由 _rank_states 裁决，目录提供 capacity_projection 供对账。
        from joint.joint_config import MODEL_SEMANTICS_VERSION
        self._kv_directory = KvLocationDirectory(
            tp_degree=self.tp_degree,
            model_version=MODEL_SEMANTICS_VERSION,
        )
        # ---- W9（§7.4）：distributed 跨轮留存段（目录通道 E）----
        # 仅 settlement_policy=distributed 的结算/驱逐/恢复写（settle_
        # distributed / _evict_retained_segment / restore_retained_segment）；
        # consolidate 缺省恒空 → 三消费点（目录镜像/守恒期望表/容量摘要）
        # 零贡献，缺省路径零漂移。键 = (instance, layer_start, layer_end,
        # token_start, token_end)。
        self._distributed_retained: dict[
            str, dict[tuple[int, int, int, int, int], RetainedSegment]] = {}
        #: 最近一次 distributed 结算披露快照（None = 尚无；调度器完成行
        #: 消费——与 last_merge_outcome 同装配契约）。
        self.last_distributed_settle_outcome: Optional[
            DistributedSettleOutcome] = None
        #: W9 开销计量（§10）：目录操作计数/累计墙时/逐会话逐轮来源面
        #: （计量不改行为；run 末 snapshot 并入 manifest）。
        self.distributed_directory_meter = DirectoryOverheadMeter()
        # D20（v2.0 §14.5/§16.4）：跨阶段预约事务账——每请求最近一次
        # 阶段提交的 epoch 与回执（幂等重放/回退的单一事实源）、abort
        # 时恢复继续方案足迹的快照，以及会话级阶段引用（P 末尾到 D
        # 提交期间基础 KV 的冷 victim 保护，§14.5）。
        self._stage_epochs: dict[str, int] = {}
        self._stage_transactions: dict[str, KVStageTransaction] = {}
        # W5（attempt 维度）：(request_id, stage_epoch) -> 已发生尝试数。
        self._stage_transaction_attempts: dict[tuple[str, int], int] = {}
        self._stage_previous_footprints: dict[str, KVCapacityReservation] = {}
        self._stage_references: dict[str, set[str]] = {}
        self._strict_kv_invariants = (
            _strict_kv_invariants_from_environment()
            if strict_invariants is None
            else strict_invariants
        )
        # D11：摘要增量账容器先于首次 ``_check_invariants`` 就位（strict
        # 构造期审计含摘要对账；贡献重索引仍在
        # ``_initialize_incremental_invariants`` 一并执行）。
        self._capacity_bucket_aggregates: dict[
            int, dict[CapacityBucketKey, _CapacityBucketAggregate]
        ] = {
            instance.index: {} for instance in self.topology.instances
        }
        self._capacity_session_contributions: dict[
            str,
            Optional[tuple[int, dict[CapacityBucketKey, tuple[tuple[int, ...], ...]]]],
        ] = {}
        # W9 通道 E：留存段的摘要贡献（逐会话多元组——段可跨实例）。
        self._capacity_retained_contributions: dict[
            str,
            tuple[tuple[int, dict[CapacityBucketKey, tuple[tuple[int, ...], ...]]], ...],
        ] = {}
        # ACC（验收纪律第 3 类，v2.0 §19.3）：容量摘要事件侧增量维护
        # （_refresh_capacity_summary）的墙时账——维护次数/累计耗时在
        # manager 分列（查询侧样本由调度器在 capacity_summary_snapshot
        # 计时入 predictor_cost 同一通道）；notifier（可选回调）把逐次
        # 维护墙时送入调度器的 summary_maintenance 统计通道（缺省 None
        # = 零行为差——本类独立可用，接线归调度器装配点）。
        self._capacity_summary_maintenance_count = 0
        self._capacity_summary_maintenance_ns = 0
        # §5.5（2026-09-30 修复）：摘要版本 = 累计维护事件数（每次
        # _refresh_capacity_summary +1；快照/决策日志按版本对账"摘要
        # 预测 vs 实际 E"时判定两侧是否同一维护代）。
        self._capacity_summary_version = 0
        self.capacity_summary_maintenance_notifier = None
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
        # 非失败台账）。科目 owner 串与 R15 流登记（rid#decode#j /
        # rid#merge）同命名纪律。
        self.copy_handoff_events: list[dict[str, object]] = []
        # C15：后缀逐组恢复的 issue/complete 审计台账（科目 rid#restore）。
        self.restore_events: list[dict[str, object]] = []
        # C15：adaptive 决策披露侧车（不改 SH 的通道）——预测器来源
        # （递推/解析）与覆盖状态逐决策落账（PROVENANCE §23）。
        self.adaptive_decisions: list[dict[str, object]] = []
        # C15：η/γ 在线效率估计组（§5.4 因果更新；逐扫描点重置 = 新
        # manager 实例，同 run 持续更新）。缺可分离样本保持原估计并
        # 标记不可观测（mark_unobservable）。
        # §8.1 单账本发布模式（2026-09-30 修复）：注入
        # ``service_factor_source``（W6 统一服务样本估计器）时 γ/η 消费
        # 方改读其只读发布值（PublishedServiceFactorGroup——与 J 计价
        # 同一发布规则；observe_valid_service fail-loud，同一服务样本
        # 不得二次更新）；None = standalone 装配，保持 ServiceFactorGroup
        # （生产无喂入点，恒冷启动 1.0）。
        if service_factor_source is not None:
            self.service_factors = PublishedServiceFactorGroup(
                service_factor_source)
        else:
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
    def session_ids(self) -> tuple[str, ...]:
        # R4-P1（2026-09-29 face 移植）：manifest 信任边界外可出现混合
        # 类型 session_id（int 与 str 混排）——裸 sorted() 对混排键集
        # TypeError；key=str 后键集排序恒有全序（全 str 键序与裸排序
        # 逐字节一致，官方路径零扰动）。
        return tuple(sorted(self._sessions, key=str))

    @property
    def kv_directory(self) -> KvLocationDirectory:
        """W4 权威区间目录（只读消费面：query_sources/query_missing_
        intervals/capacity_projection 等）。写路径 = 镜像同步单一裁决
        点（_kv_directory_sync_sessions，挂 _check_invariants_after_
        mutation 咽喉），外部不得直接改目录条目。"""
        return self._kv_directory

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
        local_shards = self._reservation_local_shards(
            reservation.session_id,
            reservation.instance_index,
        )
        extra = tuple(
            final_bytes - local_bytes
            for final_bytes, local_bytes in zip(
                reservation.final_shard_bytes,
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
        return self._local_session_shards(session_id, instance_index)

    def _reservation_committed_shards(
        self,
        reservation: KVCapacityReservation,
    ) -> tuple[int, ...]:
        """预约的真实额外占用（extra KV）——占用视图/不变量
        期望/终端注销指标回冲同源。与 ``_reservation_extra_shards`` 同一
        口径。"""
        return self._reservation_extra_shards(reservation)

    def _reservation_capacity_bytes(
        self,
        *,
        basis_shards: Sequence[int],
    ) -> tuple[int, ...]:
        if len(basis_shards) != self.tp_degree:
            raise RuntimeError("HBM reservation shard count does not match TP")
        return tuple(int(basis) for basis in basis_shards)

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
                    self._reservation_committed_shards(reservation),
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
            - self._external_workspace_bytes_by_rank[rank]
            for rank, reserved_bytes in zip(instance.ranks, reserved)
        )

    @staticmethod
    def _require_workspace_uint64(value: int, name: str) -> None:
        if (type(value) is not int or value < 0
                or value > (1 << 64) - 1):
            raise ValueError("{} must be an exact uint64 integer".format(name))

    def validate_external_workspace_view(self, workspace_state: dict):
        """Validate a full C++ workspace view without changing KV state."""
        if not isinstance(workspace_state, dict) or set(workspace_state) != {
                "protocol_version", "ranks"}:
            raise ValueError("workspace_state must have protocol_version/ranks")
        if (type(workspace_state["protocol_version"]) is not int
                or workspace_state["protocol_version"] != 1):
            raise ValueError("workspace_state protocol_version must be 1")
        rows = workspace_state["ranks"]
        if not isinstance(rows, list) or len(rows) != len(self._rank_states):
            raise ValueError(
                "workspace_state must cover every configured rank exactly")
        revisions = {}
        active_bytes = {}
        for expected_rank, row in enumerate(rows):
            where = "workspace_state.ranks[{}]".format(expected_rank)
            if not isinstance(row, dict) or set(row) != {
                    "rank", "workspace_revision", "active_workspace_bytes"}:
                raise ValueError("{} has an invalid key set".format(where))
            rank = row["rank"]
            revision = row["workspace_revision"]
            active = row["active_workspace_bytes"]
            self._require_workspace_uint64(rank, where + ".rank")
            self._require_workspace_uint64(
                revision, where + ".workspace_revision")
            self._require_workspace_uint64(
                active, where + ".active_workspace_bytes")
            if rank != expected_rank or rank not in self._rank_states:
                raise ValueError(
                    "{} ranks must cover [0, N) in ascending order".format(
                        where))
            previous_revision = self._external_workspace_revision_by_rank[rank]
            previous_bytes = self._external_workspace_bytes_by_rank[rank]
            if revision < previous_revision:
                raise ValueError(
                    "stale workspace revision for rank {}: {} < {}".format(
                        rank, revision, previous_revision))
            if revision == previous_revision and active != previous_bytes:
                raise ValueError(
                    "workspace bytes changed without revision advance on "
                    "rank {}".format(rank))
            revisions[rank] = revision
            active_bytes[rank] = active

        reserved_by_rank = {rank: 0 for rank in self._rank_states}
        for instance in self.topology.instances:
            reserved = self._reserved_bytes_by_tp_rank(instance.index)
            for rank, amount in zip(instance.ranks, reserved):
                total = reserved_by_rank[rank] + amount
                self._require_workspace_uint64(
                    total, "reserved request bytes for rank {}".format(rank))
                reserved_by_rank[rank] = total
        for rank, active in active_bytes.items():
            state = self._rank_states[rank]
            base_used = (state.model_weight_bytes + state.kv_cache_bytes
                         + reserved_by_rank[rank])
            if base_used > state.capacity_bytes or \
                    active > state.capacity_bytes - base_used:
                raise ValueError(
                    "workspace hold exceeds Python HBM capacity on rank "
                    "{}: base={}, workspace={}, capacity={}".format(
                        rank, base_used, active, state.capacity_bytes))
        return revisions, active_bytes

    def set_external_workspace_view(self, workspace_state: dict):
        """Atomically install C++ workspace counters and report changed ranks."""
        revisions, active_bytes = self.validate_external_workspace_view(
            workspace_state)
        changed = tuple(
            rank for rank in sorted(revisions)
            if (revisions[rank] != self._external_workspace_revision_by_rank[rank]
                or active_bytes[rank] != self._external_workspace_bytes_by_rank[rank])
        )
        self._external_workspace_revision_by_rank = revisions
        self._external_workspace_bytes_by_rank = active_bytes
        return changed

    def workspace_capacity_snapshot(self) -> dict[str, Any]:
        """Return the compact Python KV base view for coarse_die_v2."""
        reserved_by_rank = {rank: 0 for rank in self._rank_states}
        for instance in self.topology.instances:
            reserved = self._reserved_bytes_by_tp_rank(instance.index)
            for rank, amount in zip(instance.ranks, reserved):
                reserved_by_rank[rank] += amount
        rows = []
        for rank in sorted(self._rank_states):
            state = self._rank_states[rank]
            revision = self._external_workspace_revision_by_rank[rank]
            active = self._external_workspace_bytes_by_rank[rank]
            for name, value in (
                ("rank", rank),
                ("workspace_revision", revision),
                ("capacity_bytes", state.capacity_bytes),
                ("model_weight_bytes", state.model_weight_bytes),
                ("resident_kv_bytes", state.kv_cache_bytes),
                ("reserved_request_bytes", reserved_by_rank[rank]),
                ("active_workspace_bytes", active),
            ):
                self._require_workspace_uint64(
                    value, "workspace capacity snapshot {} for rank {}".format(
                        name, rank))
            base = (state.model_weight_bytes + state.kv_cache_bytes
                    + reserved_by_rank[rank])
            if base > state.capacity_bytes or \
                    active > state.capacity_bytes - base:
                raise RuntimeError(
                    "workspace capacity snapshot exceeds HBM capacity on "
                    "rank {}".format(rank))
            rows.append({
                "rank": rank,
                "workspace_revision": revision,
                "capacity_bytes": state.capacity_bytes,
                "model_weight_bytes": state.model_weight_bytes,
                "resident_kv_bytes": state.kv_cache_bytes,
                "reserved_request_bytes": reserved_by_rank[rank],
            })
        return {"protocol_version": 1, "ranks": rows}

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
        """Return maximum free bytes after evicting every inactive session.

        D11（2026-09-27，v2.0 §14.2）：旧实现逐次扫描全部合法 inactive
        会话（查询成本随已完成 session 累计数增长）——改为从增量维护
        的桶聚合取 ΣR（O(桶数)，与 ``capacity_summary_snapshot`` 同一
        账本）；逐出候选的逐字节判定（同一 layer-range 公式、同一合法
        victim 谓词）不变，数值与旧逐会话扫描逐位一致（钉测
        ``test_d11_capacity_summary.py::ReclaimableEquivalenceTest``）。
        """

        reclaimable = list(
            self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=exclude_request_id,
            )
        )
        # D20：候选集已在源头排除持阶段引用的会话（冷 victim 保护同
        # 一面）——被保护字节不进可回收视图（引用变化即时重索引）。
        buckets = self._capacity_bucket_aggregates.get(
            instance_index, {})
        for aggregate in buckets.values():
            reclaimable = [
                available + reclaimable_bytes
                for available, reclaimable_bytes in zip(
                    reclaimable, aggregate.reclaimable)
            ]
        if exclude_session_id is not None:
            excluded = self._capacity_session_contributions.get(
                exclude_session_id)
            if excluded is not None:
                excluded_instance, contributions = excluded
                if excluded_instance == instance_index:
                    for bundle in contributions.values():
                        reclaimable = [
                            available - reclaimable_bytes
                            for available, reclaimable_bytes in zip(
                                reclaimable, bundle[0])
                        ]
        return tuple(reclaimable)

    # ------------------------------------------------------------------
    # D11（2026-09-27，v2.0 §14.2）：容量驱逐摘要——增量维护与只读快照。
    #
    # 摘要按 T 类别 × E 易释放后缀/其余层 × 写回端口/路径归属分桶；同一
    # 桶的释放原子共享同一种逐 rank 分片比例（KV 字节对 token 精确线性
    # ⇒ 实例内比例恒同，gcd 归一签名分桶）。维护事件（§14.2）：完成、
    # 激活、引用变化、真实释放——全部经咽喉点/引用入口增量重索引；候选
    # 只查询冻结快照。干净/脏按池 backing 交叠真实推导（§4.3.1：账面
    # 忽略池侧陈旧副本 ⇒ 驻留字节无有效 backing、本协议下 clean 恒空，
    # 结构保留、计价分支钉测）。
    # ------------------------------------------------------------------

    def _summary_category_of(self, session: SessionKVState) -> str:
        """T 桶轴：typed 模式按会话类别，lru 模式单遍 all（与
        ``self._eviction_passes`` 的真实遍历结构一一对应）。"""
        if (
            len(self._eviction_passes) == 1
            and self._eviction_passes[0].trigger_type is None
        ):
            return CATEGORY_ALL
        return self._eviction_class(session)

    def _summary_suffix_boundary(self, session: SessionKVState) -> int:
        """E 桶轴：易释放后缀的层边界（[boundary, prefix) = easy）。

        * legacy_half：固定策略常量 ``legacy_resident_prefix``（E 第一
          段释放形状）；
        * minimal_layer_groups：0（全部驻留层都在可释放后缀内——策略
          ``retention_target`` 恒 0）；
        * adaptive：E 的软目标使用**最后一个合法在线规划结果**（§14.2）
          ——该会话最近一次真实释放把驻留前缀恰好停在规划边界上，故
          取 ``resident_prefix_layers``（未逐出过的会话 = 保全部，
          easy 为空——不逐候选调用 ``k_hide``，也不在摘要维护期凭空
          计算 k_hide）。
        """
        if self.layer_policy.mode == "legacy_half":
            return min(
                int(self.layer_policy.legacy_resident_prefix),
                session.resident_prefix_layers,
            )
        if self.layer_policy.mode == "minimal_layer_groups":
            return 0
        return session.resident_prefix_layers

    def _summary_pool_backed_bytes(
        self,
        session: SessionKVState,
        layer_start: int,
        layer_end: int,
    ) -> int:
        """逐出区间与该会话有效池 backing 的层交叠字节（干净部分）。

        账面语义（§4.3.1）：池侧旧 backing 为陈旧副本时账面忽略——
        有效池区间 = 驻留前缀的补集 [prefix, L)。待逐出区间 ⊆ [0,
        prefix)，交叠恒空 ⇒ 本协议下 clean 恒 0（真实推导而非硬编码；
        若执行协议未来引入去重释放，此处随账面语义更新）。"""
        if session.location not in {
            self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE,
        }:
            return 0
        pool_start = session.resident_prefix_layers
        overlap = min(layer_end, self.model.layers) - max(
            layer_start, pool_start)
        if overlap <= 0:
            return 0
        return sum(kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.context_tokens,
            self.tp_degree,
            layer_start=max(layer_start, pool_start),
            layer_end=min(layer_end, self.model.layers),
        ))

    def _capacity_session_contribution(
        self, session: SessionKVState,
    ) -> Optional[tuple[int, dict[CapacityBucketKey, tuple[tuple[int, ...], ...]]]]:
        """单会话对摘要的当前贡献（None = 非合法 victim，零贡献）。

        victim 谓词与 ``_completed_resident_candidates`` 逐条一致（除
        trigger_type 过滤——类别进桶键）；字节公式与旧可回收扫描同源
        （layer-range 驻留字节）。每个合法会话贡献两个层类桶：
        easy_suffix = [boundary, prefix)、rest_layers = [0, boundary)，
        二者不相交、并集 = 全部可释放驻留字节。释放原子 = 该会话该层
        类一次 ``_remote_store_transfer`` 事务（TP 耦合，原子个数逐
        rank 相等；最大原子 = 该类区间字节向量）。"""
        if session.session_id in self._stage_references:
            return None
        if (
            session.active
            or session.last_completion_ns is None
            or session.location not in {
                self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE,
            }
            or session.instance_index is None
        ):
            return None
        prefix = session.resident_prefix_layers
        if prefix <= 0:
            return None
        boundary = self._summary_suffix_boundary(session)
        category = self._summary_category_of(session)
        ports = tuple(
            self.nearest_edge(rank)
            for rank in self.topology.instance(session.instance_index).ranks
        )
        contributions: dict[
            CapacityBucketKey, tuple[tuple[int, ...], ...]] = {}
        for suffix_class, layer_start, layer_end in (
            (SUFFIX_EASY, boundary, prefix),
            (SUFFIX_REST, 0, boundary),
        ):
            if layer_end <= layer_start:
                continue
            reclaimable = tuple(int(value) for value in
                                self._distributed_resident_layer_bytes(
                                    session, layer_start, layer_end))
            if not any(reclaimable):
                continue
            clean_bytes = self._summary_pool_backed_bytes(
                session, layer_start, layer_end)
            # 交叠字节按比例分摊到 rank（线性 ⇒ 逐 rank 按层交叠精确
            # 分摊；本协议交叠恒空，分摊仅为推导完整性保留）。
            clean_by_rank = tuple(
                int(round(reclaim * (clean_bytes / total_reclaim)))
                if total_reclaim else 0
                for reclaim in reclaimable
                for total_reclaim in (sum(reclaimable),)
            )
            clean_by_rank = tuple(
                min(reclaim, clean) for reclaim, clean in zip(
                    reclaimable, clean_by_rank))
            dirty_by_rank = tuple(
                reclaim - clean for reclaim, clean in zip(
                    reclaimable, clean_by_rank))
            atoms = tuple(1 if value > 0 else 0 for value in reclaimable)
            key = CapacityBucketKey(
                category=category,
                suffix_class=suffix_class,
                shard_ratio=shard_ratio_signature(reclaimable),
                writeback_ports=ports,
            )
            contributions[key] = (
                reclaimable, clean_by_rank, dirty_by_rank,
                atoms, reclaimable)
        if not contributions:
            return None
        return session.instance_index, contributions

    def _capacity_retained_contribution_bundle(
        self, session_id: str,
    ) -> tuple[tuple[int, dict[CapacityBucketKey, tuple[tuple[int, ...], ...]]], ...]:
        """W9 通道 E 摘要贡献：跨轮留存段按**所在实例/实际区间**分桶求和。

        victim 谓词与 ``_completed_retained_segments`` 逐条一致（会话已
        完成/非 active/无阶段引用/段片上 READY）；桶键 = 会话 T 类别 ×
        层类（段层区间对会话保留边界的相对位置）× 分片比例 × 持有实例
        写回端口——θ 公式与增量维护框架复用（capacity_summary 原实现，
        不新建第二套）。释放原子 = 单段一次 store 事务（TP 耦合，原子数
        逐 rank 相等；最大原子 = 段字节向量）。distributed 缺省恒空。"""
        session = self._sessions.get(session_id)
        if session is None or self._is_retained_victim_blocked(session):
            return ()
        boundary = self._summary_suffix_boundary(session)
        category = self._summary_category_of(session)
        per_instance: dict[
            int, dict[CapacityBucketKey, tuple[tuple[int, ...], ...]]] = {}
        for segment in self._distributed_retained.get(
                session_id, {}).values():
            if not segment.is_on_chip() or not any(segment.bytes_by_tp_rank):
                continue
            suffix_class = (
                SUFFIX_EASY if segment.layer_start >= boundary
                else SUFFIX_REST)
            ports = tuple(
                self.nearest_edge(rank)
                for rank in self.topology.instance(
                    segment.instance_index).ranks)
            key = CapacityBucketKey(
                category=category,
                suffix_class=suffix_class,
                shard_ratio=shard_ratio_signature(segment.bytes_by_tp_rank),
                writeback_ports=ports,
            )
            reclaimable = tuple(
                int(value) for value in segment.bytes_by_tp_rank)
            atoms = tuple(1 if value > 0 else 0 for value in reclaimable)
            bundle = (
                reclaimable,
                tuple(0 for _ in reclaimable),   # clean：无有效池 backing
                reclaimable,                     # dirty：释放前写回
                atoms, reclaimable)
            by_key = per_instance.setdefault(segment.instance_index, {})
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = bundle
            else:
                # 同一会话的多个留存区间经常落入同一桶。旧实现直接
                # 赋值，后登记的段覆盖先登记的段，摘要因此少算可驱逐
                # 字节/原子；真实 E 仍能释放全部段，预测与执行便分叉。
                by_key[key] = (
                    tuple(a + b for a, b in zip(previous[0], bundle[0])),
                    tuple(a + b for a, b in zip(previous[1], bundle[1])),
                    tuple(a + b for a, b in zip(previous[2], bundle[2])),
                    tuple(a + b for a, b in zip(previous[3], bundle[3])),
                    tuple(max(a, b) for a, b in zip(previous[4], bundle[4])),
                )
        return tuple(
            (instance_index, vectors)
            for instance_index, vectors in sorted(per_instance.items()))

    def _is_retained_victim_blocked(self, session: SessionKVState) -> bool:
        """留存段 victim 的会话级保护谓词（与主位置 victim 同款保护面：
        active / 未完成 / 阶段引用在场——D20 冷 victim 保护同源）。"""
        return (
            session.active
            or session.last_completion_ns is None
            or session.session_id in self._stage_references
        )

    def _capacity_bucket_order_key(
        self, key: CapacityBucketKey,
    ) -> tuple[int, int, tuple[int, ...], tuple[int, ...]]:
        """快照固定桶序：T 类别遍历序 × 层类序 × 键余（确定性）。"""
        category_order = {
            (passthrough.trigger_type if passthrough.trigger_type
             is not None else CATEGORY_ALL): order
            for order, passthrough in enumerate(self._eviction_passes)
        }
        category_rank = category_order.get(key.category, len(category_order))
        suffix_rank = (
            SUFFIX_CLASS_ORDER.index(key.suffix_class)
            if key.suffix_class in SUFFIX_CLASS_ORDER
            else len(SUFFIX_CLASS_ORDER))
        return (
            category_rank, suffix_rank,
            key.shard_ratio, key.writeback_ports,
        )

    def _apply_capacity_contribution(
        self,
        instance_index: int,
        key: CapacityBucketKey,
        bundle: tuple[tuple[int, ...], ...],
        multiplier: int,
    ) -> None:
        buckets = self._capacity_bucket_aggregates.setdefault(
            instance_index, {})
        aggregate = buckets.get(key)
        if aggregate is None:
            aggregate = _CapacityBucketAggregate(len(bundle[0]))
            buckets[key] = aggregate
        if multiplier > 0:
            aggregate.add(bundle)
        else:
            aggregate.remove(bundle)

    def _refresh_capacity_summary(self, session_ids: Iterable[str]) -> None:
        """增量重索引给定会话的摘要贡献（先减旧、再加新；O(桶数)/会话）。

        ACC：逐次调用的墙时入维护账（计数/累计 + 可选 notifier 送调度
        器统计通道）——真实事件增量维护耗时与查询侧同通道分列入账。"""
        maintenance_start = time.perf_counter_ns()
        try:
            for session_id in dict.fromkeys(session_ids):
                old = self._capacity_session_contributions.pop(session_id, None)
                if old is not None:
                    old_instance, old_vectors = old
                    for key, vector_bundle in old_vectors.items():
                        self._apply_capacity_contribution(
                            old_instance, key, vector_bundle, -1)
                    self._prune_capacity_buckets(old_instance)
                # W9 通道 E：留存段逐段贡献（先减旧；distributed 缺省恒空
                # → 零贡献零漂移）。
                old_retained = self._capacity_retained_contributions.pop(
                    session_id, ())
                for old_instance_r, old_vectors_r in old_retained:
                    for key, vector_bundle in old_vectors_r.items():
                        self._apply_capacity_contribution(
                            old_instance_r, key, vector_bundle, -1)
                    self._prune_capacity_buckets(old_instance_r)
                session = self._sessions.get(session_id)
                if session is None:
                    self._capacity_session_contributions.pop(session_id, None)
                    continue
                new = self._capacity_session_contribution(session)
                if new is not None:
                    new_instance, new_vectors = new
                    for key, vector_bundle in new_vectors.items():
                        self._apply_capacity_contribution(
                            new_instance, key, vector_bundle, +1)
                self._capacity_session_contributions[session_id] = new
                new_retained = self._capacity_retained_contribution_bundle(session_id)
                if new_retained:
                    self._capacity_retained_contributions[session_id] = (
                        new_retained)
                    for new_instance_r, new_vectors_r in new_retained:
                        for key, vector_bundle in new_vectors_r.items():
                            self._apply_capacity_contribution(
                                new_instance_r, key, vector_bundle, +1)
        finally:
            elapsed_ns = time.perf_counter_ns() - maintenance_start
            self._capacity_summary_maintenance_count += 1
            self._capacity_summary_maintenance_ns += elapsed_ns
            # §5.5（2026-09-30 修复）：每次维护事件推进摘要版本——快照
            # （capacity_summary_snapshot）与决策日志按版本对账。
            self._capacity_summary_version += 1
            notifier = self.capacity_summary_maintenance_notifier
            if notifier is not None:
                notifier(elapsed_ns)

    def capacity_summary_maintenance_stats(self) -> dict[str, int]:
        """事件侧摘要维护账（ACC）：维护次数 + 累计墙时（整数纳秒）。

        供调度器 run 末并入 predictor 开销快照的 summary_maintenance
        块（查询侧样本与维护侧计数同块分列）。"""
        return {
            "count": self._capacity_summary_maintenance_count,
            "total_ns": self._capacity_summary_maintenance_ns,
        }

    def _prune_capacity_buckets(self, instance_index: int) -> None:
        buckets = self._capacity_bucket_aggregates.get(instance_index, {})
        for key in [key for key, aggregate in buckets.items()
                    if aggregate.is_empty()]:
            del buckets[key]

    def capacity_summary_snapshot(self) -> CapacitySummary:
        """冻结摘要快照（候选查询入口；O(桶数×rank 数)，不扫描会话）。

        §14.2：快照是近似视图——实际提交仍由 ``_ensure_capacity`` 按
        真实规则重新确认，软目标/份额不视为永不变化的物理事实。

        §5.5（2026-09-30 修复）：快照携带版本与误差来源——版本 = 累计
        维护事件数（``capacity_summary_version``），误差来源 = 登记标
        签（θ 份额连续近似 + 增量维护滞后）；随摘要进决策日志与快照
        容量面，"摘要预测 vs 实际 E"对账按版本判定同代性。"""
        version = self._capacity_summary_version
        instances: dict[int, InstanceCapacitySummary] = {}
        for instance_index in sorted(self._capacity_bucket_aggregates):
            buckets = self._capacity_bucket_aggregates[instance_index]
            frozen = tuple(
                CapacityBucket(
                    key=key,
                    reclaimable_by_rank=snapshot[0],
                    clean_by_rank=snapshot[1],
                    dirty_by_rank=snapshot[2],
                    atom_count_by_rank=snapshot[3],
                    max_atom_by_rank=snapshot[4],
                )
                for key, snapshot in sorted(
                    ((key, aggregate.snapshot())
                     for key, aggregate in buckets.items()),
                    key=lambda item: self._capacity_bucket_order_key(
                        item[0]))
            )
            instances[instance_index] = InstanceCapacitySummary(
                instance_index=instance_index, buckets=frozen,
                snapshot_version=version,
                maintenance_event_count=(
                    self._capacity_summary_maintenance_count),
                error_sources=CAPACITY_SUMMARY_ERROR_SOURCES)
        return CapacitySummary(
            instances=instances, snapshot_version=version,
            error_sources=CAPACITY_SUMMARY_ERROR_SOURCES)

    @property
    def capacity_summary_version(self) -> int:
        """§5.5：摘要版本（累计维护事件数；快照/日志披露面读口）。"""
        return self._capacity_summary_version

    @property
    def capacity_summary_error_sources(self) -> tuple[str, ...]:
        """§5.5：摘要登记的结构性误差来源标签。"""
        return CAPACITY_SUMMARY_ERROR_SOURCES

    def capacity_ledger_by_tp_rank(
        self,
        instance_index: int,
    ) -> tuple[tuple[int, int, int, int], ...]:
        """逐 rank 容量账 (C, W, O, Z)——分量分明（式 19）。

        F = C − W − O − Z 由消费方（``_space_ready_ns_estimate``）对
        ``hbm_remaining_bytes_by_tp_rank`` 交叉核账（Z→O 转账不变
        C−W 总量的守恒锚随单测钉）。"""
        instance = self.topology.instance(instance_index)
        reserved = self._reserved_bytes_by_tp_rank(instance_index)
        return tuple(
            (
                int(self._rank_states[rank].capacity_bytes),
                int(self._rank_states[rank].model_weight_bytes),
                int(self._rank_states[rank].kv_cache_bytes),
                int(reserved[relative]),
            )
            for relative, rank in enumerate(instance.ranks)
        )

    def _check_capacity_summary_invariants(self) -> None:
        """全量审计（strict/终态）：桶聚合 == 从零逐会话重建。

        (2026-10-01 审查修复) 重建侧必须同时重放两条贡献通道：会话
        通道（_capacity_session_contribution）**与 W9 通道 E 的留存段
        bundle（_capacity_retained_contribution_bundle）**——增量侧
        _refresh_capacity_summary 把两路都叠进同一组
        _capacity_bucket_aggregates，重建漏通道 E 时任何持有片上 READY
        留存段的 distributed run 终态审计必误报 "bucket keys drift"。"""
        rebuilt: dict[int, dict[CapacityBucketKey, _CapacityBucketAggregate]] = {
            instance.index: {} for instance in self.topology.instances
        }

        def _fold(entries) -> None:
            for instance_index, vectors in entries:
                for key, bundle in vectors.items():
                    aggregate = rebuilt[instance_index].get(key)
                    if aggregate is None:
                        aggregate = _CapacityBucketAggregate(len(bundle[0]))
                        rebuilt[instance_index][key] = aggregate
                    aggregate.add(bundle)

        for session in self._sessions.values():
            contribution = self._capacity_session_contribution(session)
            if contribution is not None:
                _fold((contribution,))
            # W9 通道 E：跨轮留存段（distributed；consolidate 恒空）。
            _fold(self._capacity_retained_contribution_bundle(
                session.session_id))
        fields = ("reclaimable", "clean", "dirty", "atom_count", "max_atom")
        for instance_index, buckets in self._capacity_bucket_aggregates.items():
            expected = rebuilt.get(instance_index, {})
            if set(buckets) != set(expected):
                raise RuntimeError(
                    f"capacity summary bucket keys drift in instance "
                    f"{instance_index}: {sorted(map(str, buckets))} != "
                    f"{sorted(map(str, expected))}")
            for key, aggregate in buckets.items():
                for field_name in fields:
                    actual = getattr(aggregate, field_name)
                    expected_values = getattr(expected[key], field_name)
                    if actual != expected_values:
                        raise RuntimeError(
                            f"capacity summary bucket {key} field "
                            f"{field_name} drift in instance "
                            f"{instance_index}: {actual} != {expected_values}")

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
        different instance must admit the complete final KV.  三动作
        （stay/copy/recompute）的终态工作副本恒为 history+input 整份
        （驻留前缀复用部分经 ``final − local`` 自然扣除）。

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
        basis_context = history_tokens + input_tokens
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
            required_basis = self._reservation_capacity_bytes(
                basis_shards=basis_shards,
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
        R1'：去 PARTIAL 钉扎。可行性按每个目标实例分别计入其物理足迹。
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
        basis_context = history_tokens + input_tokens
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            basis_context,
            self.tp_degree,
        )

        feasible: list[bool] = []
        for instance in self.topology.instances:
            required_basis = self._reservation_capacity_bytes(
                basis_shards=final_shards,
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
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        """Commit final local-KV capacity before a request enters Prefill.

        R1'：预约量 = history+input 整份足迹；容量不足抛
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
        basis_context = history_tokens + max(
            0, final_context_tokens - history_tokens)
        basis_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            basis_context,
            self.tp_degree,
        )
        reserved_shards = self._reservation_capacity_bytes(
            basis_shards=basis_shards,
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
            at_ns=at_ns,
        )
        self._reservations[request_id] = KVCapacityReservation(
            request_id=request_id,
            session_id=session_id,
            instance_index=instance_index,
            final_context_tokens=basis_context,
            final_shard_bytes=basis_shards,
            action=action,
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

    # ------------------------------------------------------------------
    # D20（v2.0 §14.5/§16.4）：跨阶段预约事务（reservation transfer /
    # resize + 幂等 epoch）。P→D 边界阶段提交的单一事务面——D18 的
    # move_request_capacity_reservation / revert_request_capacity_
    # reservation_move 两段式调用口已被本事务面整体替换（直接替换，无
    # 开关并存；调用点 = 在线调度器 drain 边界 + 本文件钉测）。
    # ------------------------------------------------------------------

    def stage_reservation_footprint(
        self,
        *,
        session_id: str,
        target_instance_index: int,
        action: str,
        history_tokens: int,
        input_tokens: int,
    ) -> StageReservationFootprint:
        """D 阶段承诺足迹单源（式 20a 的 ``P^{D,peak}`` 逐 rank 向量）。

        与准入 ``reserve_request_capacity`` 同一套口径（token 基数 =
        history+input 整份），防估计/事务口径分叉。
        ``history/input`` 由调用方按边界布局给出（完整型 = H_B/0）。
        """
        if action not in {"stay", "copy", "recompute"}:
            raise ValueError(f"unknown stage action: {action!r}")
        basis_context = (
            max(0, int(history_tokens)) + max(0, int(input_tokens)))
        basis_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            basis_context,
            self.tp_degree,
        )
        return StageReservationFootprint(
            action=action,
            target_instance_index=int(target_instance_index),
            basis_context_tokens=int(basis_context),
            basis_shard_bytes=tuple(int(value) for value in basis_shards),
        )

    def commit_stage_reservation_transaction(
        self,
        *,
        request_id: str,
        target_instance_index: int,
        stage_action: str,
        stage_epoch: int,
        history_tokens: int,
        input_tokens: int,
        at_ns: Optional[int] = None,
        expected_versions: Optional[Mapping[str, object]] = None,
    ) -> KVStageTransaction:
        """跨阶段预约事务提交（v2.0 §14.5：同一事务内创建新目标承诺、
        转用/换址承诺、释放未被物理工作占用的旧承诺；§16.4：阶段
        epoch/idempotency key 保证一次提交）。

        * 幂等门：同 ``(request_id, stage_epoch)`` 且目标/动作一致的
          重复提交原子重放已记录回执（重复 P 完成通知/重复在途 ack
          不产生第二份换址或第二份逐出）；epoch 回退或同 epoch 冲突
          重放 fail-closed。
        * resize 半边（式 20a）：按 D 阶段动作 + 目标实例重推承诺足迹
          （``stage_reservation_footprint`` 单源），``required =
          max(0, committed − own local at target)``——own 驻留计
          ``O^{self,reuse}``，其他请求的 O/Z 不是可扣项（目标缺口只
          对本请求自己的新承诺负责）；footprint 缩小释放未被物理工作
          占用的旧承诺分量（``released_shard_bytes`` 逐 rank 披露），
          不允许缩到自有驻留之下（fail-closed——驻留字节仍在账）。
        * transfer 半边：跨实例 = 目标先 ``_ensure_capacity`` 真实
          T+E（容量不足抛 ``KVCapacityError``，**原预约原样保留、
          epoch 不记录**——调用方按 §16.4 保留继续方案或换状态后重
          评），成功后原子换址 + 足迹落账（frozen dataclass 受控唯一
          变更点）；同实例 = 纯 resize（原地继续是合法终态，式 1c）。

        事务语义：承诺先行于物理搬运（字节搬运由
        ``move_prefill_to_decode`` 承担；两者都成功才是 D_COMMITTED
        的预约半边闭环）；解码增长不进本承诺（§3.1 因果增长纪律不变）。
        """
        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        if not isinstance(stage_epoch, int) or isinstance(stage_epoch, bool):
            raise ValueError(
                f"stage epoch must be an int, got {stage_epoch!r}")
        if stage_epoch < 0:
            raise ValueError(
                f"stage epoch must be non-negative, got {stage_epoch}")
        recorded = self._stage_transactions.get(request_id)
        last_epoch = self._stage_epochs.get(request_id)
        if recorded is not None and last_epoch is not None:
            if stage_epoch == last_epoch:
                if (not recorded.aborted
                        and recorded.target_instance_index
                        == int(target_instance_index)
                        and recorded.stage_action == stage_action):
                    # 幂等重放：恰一次提交语义（§16.4 阶段 epoch）。
                    return recorded
                raise ValueError(
                    f"conflicting stage transaction replay for request "
                    f"{request_id}: recorded epoch {last_epoch} "
                    f"(target={recorded.target_instance_index}, "
                    f"action={recorded.stage_action!r}, "
                    f"aborted={recorded.aborted}) conflicts with "
                    f"(target={target_instance_index}, "
                    f"action={stage_action!r})")
            if stage_epoch < last_epoch:
                raise ValueError(
                    f"stale stage epoch {stage_epoch} for request "
                    f"{request_id} (last committed {last_epoch})")
        # W5（§7.2 事务顺序第 1 步）：提交点版本校验——决策时点快照的
        # layout（目录事务版本）/reservation（阶段纪元表）与提交点一致
        # 才继续；漂移 = 同 tick 竞争先提交，抛 KVStageVersionConflict
        # （不沿用旧容量承诺；幂等重放命中上方回执分支，不受影响）。
        if expected_versions is not None:
            if not isinstance(expected_versions, Mapping):
                raise ValueError(
                    "expected_versions must be a mapping or None")
            directory = getattr(self, "_kv_directory", None)
            expected_directory = expected_versions.get("directory_version")
            if (expected_directory is not None
                    and directory is not None
                    and int(directory.version)
                    != int(expected_directory)):
                raise KVStageVersionConflict(
                    f"stage transaction version conflict for request "
                    f"{request_id}: kv_directory version "
                    f"{directory.version} != decision-time "
                    f"{expected_directory}")
            expected_stage_epochs = expected_versions.get("stage_epochs")
            if expected_stage_epochs is not None:
                current_stage_epochs = tuple(sorted(
                    self._stage_epochs.items()))
                if current_stage_epochs != tuple(expected_stage_epochs):
                    raise KVStageVersionConflict(
                        f"stage transaction version conflict for request "
                        f"{request_id}: stage epochs {current_stage_epochs} "
                        f"!= decision-time "
                        f"{tuple(expected_stage_epochs)}")
        reservation = self._reservations[request_id]
        source_instance_index = reservation.instance_index
        footprint = self.stage_reservation_footprint(
            session_id=reservation.session_id,
            target_instance_index=target_instance_index,
            action=stage_action,
            history_tokens=history_tokens,
            input_tokens=input_tokens,
        )
        # O^{self,reuse}：目标实例上本请求自己的驻留字节（仍有效、未
        # 被其他操作占用）——式 20a 的自有可扣项。
        local_at_target = self._reservation_local_shards(
            reservation.session_id, target_instance_index)
        basis = footprint.basis_shard_bytes
        # 物化守卫口径：驻留字节必须被 basis 覆盖——驻留字节仍在账，
        # 承诺不得缩到自有驻留之下（fail-closed）。
        if any(c < l for c, l in zip(basis, local_at_target)):
            raise ValueError(
                f"stage resize for request {request_id} would commit "
                f"{sum(basis)} bytes below its own resident "
                f"working copy {sum(local_at_target)} at instance "
                f"{target_instance_index} (resident bytes stay claimed)")
        committed_at_target = footprint.committed_shard_bytes()
        # 旧承诺申索 = old extra（own local 之上）；新申索 =
        # 式 20a 的 required。released 披露 = 旧申索中未被新阶段工作
        # 占用、随本事务释放的分量（claim 对 claim 同维比较）。
        old_committed = self._reservation_committed_shards(reservation)
        # 式 20a：Z^{D,add}_{j,r} = max(0, P^{D,peak} − O^{self,reuse} −
        # Z^{self,transferable})——目标侧真实新预约缺口（own 换址随
        # 成功半边原子迁移，不计双份）。
        required = tuple(
            max(0, committed - local)
            for committed, local in zip(committed_at_target, local_at_target)
        )
        released_shard_bytes = tuple(
            max(0, old - new)
            for old, new in zip(old_committed, required))
        irreversible = False
        evictions: tuple[KVTransfer, ...] = ()
        if any(required):
            evictions = self._ensure_capacity(
                target_instance_index,
                required,
                phase="decode",
                reason=(
                    "stage_reservation_transfer"
                    if target_instance_index != source_instance_index
                    else "stage_reservation_resize"),
                trigger_request_id=request_id,
                reservation_request_id=request_id,
                protected_session_id=reservation.session_id,
                at_ns=at_ns,
            )
            # 不可逆位 = 真实 T+E 逐出已提交（物理容量事件）；纯余量
            # 核验（零逐出）不改变可回退性。
            irreversible = bool(evictions)
        # 成功半边：原子换址 + resize 落账（此后占用视图/指标锚随新
        # 实例与新足迹；源实例侧 resident 字节由 move_prefill_to_decode
        # 的物理搬运负责）。先快照旧足迹供 abort 回退继续方案（§16.4）。
        self._stage_previous_footprints[request_id] = KVCapacityReservation(
            request_id=reservation.request_id,
            session_id=reservation.session_id,
            instance_index=reservation.instance_index,
            final_context_tokens=reservation.final_context_tokens,
            final_shard_bytes=reservation.final_shard_bytes,
            action=reservation.action,
        )
        for name, value in (
                ("instance_index", int(target_instance_index)),
                ("action", stage_action),
                ("final_context_tokens", footprint.basis_context_tokens),
                ("final_shard_bytes", basis),
        ):
            object.__setattr__(reservation, name, value)
        self._check_invariants_after_mutation(reservation_ids=(request_id,))
        # W5（attempt 维度）：同 epoch 尝试计数（幂等重放在上方早退，
        # 到此即新尝试）。
        attempt_key = (request_id, stage_epoch)
        self._stage_transaction_attempts[attempt_key] = (
            self._stage_transaction_attempts.get(attempt_key, 0))
        transaction = KVStageTransaction(
            request_id=request_id,
            session_id=reservation.session_id,
            stage_epoch=stage_epoch,
            stage_action=stage_action,
            source_instance_index=source_instance_index,
            target_instance_index=int(target_instance_index),
            reserved_shard_bytes=committed_at_target,
            released_shard_bytes=released_shard_bytes,
            evictions=evictions,
            irreversible=irreversible,
            attempt=self._stage_transaction_attempts[attempt_key],
            # A22：成功提交同样分记已产生的不可撤销物理子义务（真实
            # T+E 逐出——真实容量事件，保留在账供审计/继承）。
            retained_obligations=_stage_retained_obligations(evictions),
        )
        self._stage_transaction_attempts[attempt_key] += 1
        self._stage_epochs[request_id] = stage_epoch
        self._stage_transactions[request_id] = transaction
        return transaction

    def abort_stage_reservation_transaction(
        self,
        *,
        request_id: str,
        source_instance_index: int,
        stage_epoch: int,
    ) -> Optional[KVStageTransaction]:
        """D20 阶段事务回退半边（§16.4：改变方案提交失败且无不可逆
        工作时，保留继续方案的合法状态）。

        仅在字节搬运尚未发射的提交窗口内合法（
        ``move_prefill_to_decode`` 的 KVCapacityError 发生在任何字节
        移动之前）。换址半边已提交（本 epoch 有在案成功回执）时：换址
        回退到源实例 + 足迹恢复提交前快照，并把本 epoch 结局改记为
        aborted（已提交的 T+E 逐出照常携带入账——真实容量事件不回
        滚）；同 epoch 的重复回退/重复通知原子重放 aborted 回执。换址
        半边从未提交（提交在容量检查即失败、无在案回执）时：预约应已
        在源实例，幂等空放返回 None。"""
        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        reservation = self._reservations[request_id]
        recorded = self._stage_transactions.get(request_id)
        last_epoch = self._stage_epochs.get(request_id)
        if (recorded is not None and last_epoch is not None
                and last_epoch == stage_epoch):
            if recorded.aborted:
                if recorded.target_instance_index != int(source_instance_index):
                    raise ValueError(
                        f"conflicting stage abort replay for request "
                        f"{request_id}: aborted at "
                        f"{recorded.target_instance_index}, source now "
                        f"{source_instance_index}")
                return recorded
            previous = self._stage_previous_footprints.pop(request_id, None)
            object.__setattr__(
                reservation, "instance_index", int(source_instance_index))
            if previous is not None:
                for name, value in (
                        ("action", previous.action),
                        ("final_context_tokens",
                         previous.final_context_tokens),
                        ("final_shard_bytes", previous.final_shard_bytes),
                ):
                    object.__setattr__(reservation, name, value)
            self._check_invariants_after_mutation(
                reservation_ids=(request_id,))
            aborted = KVStageTransaction(
                request_id=request_id,
                session_id=reservation.session_id,
                stage_epoch=stage_epoch,
                stage_action=recorded.stage_action,
                source_instance_index=recorded.source_instance_index,
                target_instance_index=int(source_instance_index),
                reserved_shard_bytes=recorded.reserved_shard_bytes,
                released_shard_bytes=recorded.released_shard_bytes,
                evictions=recorded.evictions,
                irreversible=recorded.irreversible,
                aborted=True,
                attempt=recorded.attempt,
                # A22：阶段提交失败与不可撤销物理子义务分记——已提交
                # T+E 逐出保留身份/剩余量（供后续候选继承），不取消、
                # 不漏计（§7.2 方案锁定原文）。
                retained_obligations=_stage_retained_obligations(
                    recorded.evictions),
            )
            self._stage_transactions[request_id] = aborted
            return aborted
        # 无在案成功回执（提交在容量检查即失败）：换址从未生效，预约
        # 必须仍在源实例——否则提交/回退账目脱钩，fail-closed。
        if reservation.instance_index != int(source_instance_index):
            raise RuntimeError(
                f"stage abort for request {request_id}: no committed "
                f"transaction for epoch {stage_epoch} but reservation "
                f"sits on instance {reservation.instance_index} "
                f"(expected source {source_instance_index})")
        return None

    def last_stage_transaction(
        self, request_id: str,
    ) -> Optional[KVStageTransaction]:
        """D20：请求最近一次阶段事务回执（无 = 尚未提交过）。"""
        return self._stage_transactions.get(request_id)

    # ------------------------------------------------------------------
    # D20（§14.5）：会话阶段引用——P 末尾到 D 提交期间本请求基础 KV
    # 的冷 victim 集合保护。"暂时没有活跃 D"不解除引用；D 各来源引用
    # 与目标预留建立后才释放 P 阶段引用。
    # ------------------------------------------------------------------

    def acquire_stage_reference(
        self,
        session_id: str,
        *,
        request_id: str,
        stage_epoch: int,
        reason: str = "pd_boundary_base_kv",
    ) -> str:
        """登记会话阶段引用（返回引用句柄；同会话多引用并存计数）。"""
        session = self._sessions.get(session_id)
        if session is None:
            raise KeyError(f"unknown KV session: {session_id}")
        reference_id = "stage_ref:{}#{}#{}".format(
            request_id, int(stage_epoch), reason)
        self._stage_references.setdefault(session_id, set()).add(
            reference_id)
        # D11：引用变化 = 合法 victim 集合变化——摘要增量重索引（被
        # 引用会话立即退出可回收视图）。
        self._refresh_capacity_summary((session_id,))
        return reference_id

    def release_stage_reference(self, session_id: str,
                                *, reference_id: str) -> bool:
        """释放阶段引用。幂等：重复释放（双交付/取消与提交竞合）安全
        空放返回 False；未知会话同理。"""
        references = self._stage_references.get(session_id)
        if references is None or reference_id not in references:
            return False
        references.discard(reference_id)
        if not references:
            del self._stage_references[session_id]
        # D11：引用变化 = 合法 victim 集合变化——摘要增量重索引。
        self._refresh_capacity_summary((session_id,))
        return True

    def session_stage_references(self, session_id: str) -> tuple[str, ...]:
        """会话当前在持阶段引用（空元组 = 无保护）。"""
        return tuple(sorted(self._stage_references.get(session_id, ())))

    def release_request_capacity_reservation(self, request_id: str) -> None:
        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        reservation = self._reservations[request_id]
        if any(self._reservation_extra_shards(reservation)):
            raise RuntimeError(
                f"request {request_id} released HBM reservation before final KV allocation"
            )
        # D20（§14.5/§16.4）：预约注销 = 本请求边界窗口闭合点——该请求
        # 的阶段引用必须已全部释放（引用与预约不得双活到结算之后），
        # 事务账随预约一并清账（请求生命周期终止，防长 run 常驻）。
        dangling = tuple(
            reference_id
            for reference_id in self._stage_references.get(
                reservation.session_id, ())
            if reference_id.startswith(f"stage_ref:{request_id}#"))
        if dangling:
            raise RuntimeError(
                f"request {request_id} released its reservation with live "
                f"stage references {dangling} (boundary window never "
                f"closed)")
        self._stage_epochs.pop(request_id, None)
        self._stage_transactions.pop(request_id, None)
        self._stage_previous_footprints.pop(request_id, None)
        # (2026-10-01 审查修复/R2 补正) 逐请求事务尝试账随其余 stage
        # 账一并清理：键为 (request_id, stage_epoch) 元组——R1 修复按裸
        # 字符串 pop 永不命中（no-op），此处按前缀逐键删除。
        for attempt_key in [
                key for key in list(self._stage_transaction_attempts)
                if key[0] == request_id]:
            del self._stage_transaction_attempts[attempt_key]
        del self._reservations[request_id]
        self._check_invariants_after_mutation(reservation_ids=(request_id,))

    # ------------------------------------------------------------------
    # D18（2026-09-26）：``decode_hbm_feasible_instances`` 随同址钉扎
    # 一并删除（死代码——全仓零调用点；其"完整本地 P KV"前提与
    # "Decode 固定 prefill 实例"同属被 D18 废除的协议，候选面过滤由
    # JCM 适用性表与预约事务容量裁定承担，不做可行性掩码）。

    def session_snapshot(self, session_id: str) -> SessionKVSnapshot:
        if session_id not in self._sessions:
            raise KeyError(f"unknown KV session: {session_id}")
        state = self._sessions[session_id]
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
        # D19/C4：边界第二来源（i_P 侧旧副本）贡献的期望表镜像。
        self._invariant_boundary_source_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        # W9 通道 E：跨轮留存段贡献的期望表镜像（逐段多元组——一个会话
        # 可在多实例持有多段）。
        self._invariant_retained_contributions: dict[
            str, tuple[tuple[int, tuple[int, ...]], ...]
        ] = {}
        self._invariant_reservation_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._invariant_reservation_sessions: dict[str, str] = {}
        self._invariant_reservations_by_session: dict[str, set[str]] = {}
        self._refresh_incremental_sessions(self._sessions)
        self._refresh_incremental_reservations(self._reservations)
        # D11（2026-09-27，v2.0 §14.2）：容量驱逐摘要的增量账——桶聚合
        # 已在构造期先行就位（见 ``__init__``）；此处对既有会话做首次
        # 贡献重索引。完成/激活/引用变化/真实释放事件经
        # ``_refresh_capacity_summary`` 增量维护（挂在
        # ``_check_invariants_after_mutation`` 咽喉点 + 阶段引用两入口）；
        # 候选只查询冻结快照（``capacity_summary_snapshot``，O(桶数×
        # rank 数)，不随已完成 session 累计数增长）。真实 T/E/victim
        # 决策保留在 ``_ensure_capacity`` 原位、不挪进候选循环。
        self._refresh_capacity_summary(self._sessions)

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

        工作副本活跃时 primary 描述执行端；基础驻留前缀字节仍在**物理
        基础驻留实例**（``base_instance_index``，D-BI）的 rank 台账上，
        必须计入期望值，否则逐 session 守恒审计把基础副本误判为泄漏。
        逻辑 home（§2.1 粘滞）≠ 基础所在：P→D 迁移 + merge stay 提交后
        下一轮事务的基础可不在 home（冒烟 v20-b2 rank20 幻影贡献根因）。
        基础 REMOTE（池 backing）无本地字节。"""
        if session.working_kind is None:
            return None
        if session.base_location not in {
            self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE,
        }:
            return None
        if session.base_instance_index is None:
            raise RuntimeError(
                f"session {session.session_id} has a resident base "
                f"({session.base_location}) without a recorded base "
                "instance; ledger integrity failure")
        journal = session.copy_handoff
        if journal is not None:
            # C13：copy 的 home 侧残量以交接账本为准（逐块释放不等轮末；
            # 全部释放后贡献为零——不重复释放已交接源块）。
            shards = journal.home_shards()
            if not any(shards):
                return None
            return session.base_instance_index, shards
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
        return session.base_instance_index, tuple(
            int(value) for value in local_shards)

    def _incremental_boundary_source_contribution(
        self, session: SessionKVState
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        """D19/C4：边界第二来源区间（i_P 侧旧工作副本，recompute@j≠i_P
        不迁移）的守恒贡献——与 ``_incremental_base_contribution``
        同语义的第三位置字节（片上位置 ≤3 界）。无第二来源（None/空串
        location）返回 None。"""
        if session.boundary_source_instance_index is None:
            return None
        if session.boundary_source_location not in {
            self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE,
        }:
            return None
        shards = tuple(
            int(value) for value in session.boundary_source_shard_bytes)
        if not any(shards):
            return None
        return session.boundary_source_instance_index, shards

    def _incremental_reservation_contribution(
        self, reservation: KVCapacityReservation
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        if len(reservation.final_shard_bytes) != self.tp_degree:
            return None
        committed = self._reservation_committed_shards(reservation)
        if len(committed) != self.tp_degree:
            return None
        return reservation.instance_index, tuple(
            int(value) for value in committed)

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
            old_source = self._invariant_boundary_source_contributions.pop(
                session_id, None)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_kv_by_rank, old_source, -1
                )
            )
            old_retained = self._invariant_retained_contributions.pop(
                session_id, ())
            for contribution in old_retained:
                affected_ranks.update(
                    self._apply_incremental_contribution(
                        self._invariant_expected_kv_by_rank, contribution, -1
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
            new_source = self._incremental_boundary_source_contribution(
                session)
            self._invariant_boundary_source_contributions[session_id] = (
                new_source)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_kv_by_rank, new_source, 1
                )
            )
            # W9 通道 E：跨轮留存段的守恒贡献（片上 READY 段占持有实例
            # rank 台账；池/RESTORING 段不占实例容量——恢复物理腿由
            # restore 事务显式入账）。
            new_retained = self._incremental_retained_contributions(session_id)
            self._invariant_retained_contributions[session_id] = new_retained
            for contribution in new_retained:
                affected_ranks.update(
                    self._apply_incremental_contribution(
                        self._invariant_expected_kv_by_rank, contribution, 1
                    )
                )
        return affected_ranks

    def _incremental_retained_contributions(
        self, session_id: str,
    ) -> tuple[tuple[int, tuple[int, ...]], ...]:
        """通道 E 守恒贡献：片上 READY 留存段逐段计入持有实例期望。"""
        contributions = []
        for segment in self._distributed_retained.get(
                session_id, {}).values():
            if not segment.is_on_chip():
                continue
            if not any(segment.bytes_by_tp_rank):
                continue
            contributions.append(
                (segment.instance_index,
                 tuple(int(value) for value in segment.bytes_by_tp_rank)))
        return tuple(contributions)

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
                raise RuntimeError(
                    f"remote KV session retained a local instance "
                    f"for {session.session_id}"
                )
            if session.resident_prefix_layers != 0:
                raise RuntimeError(
                    f"remote KV session retained resident layers "
                    f"for {session.session_id}"
                )
            return
        if session.location == self.LOCAL_HBM:
            if session.resident_prefix_layers != self.model.layers:
                raise RuntimeError(
                    f"fully local KV session is missing layers "
                    f"for {session.session_id}"
                )
        elif not 0 < session.resident_prefix_layers < self.model.layers:
            raise RuntimeError(
                f"partial KV session has an invalid prefix length "
                f"for {session.session_id}"
            )
        if session.instance_index is None:
            raise RuntimeError(
                f"local KV session has no instance "
                f"for {session.session_id}"
            )
        instance = self.topology.instance(session.instance_index)
        if len(session.shard_bytes) != instance.size:
            raise RuntimeError(
                f"KV shard count does not match TP instance "
                f"for {session.session_id}"
            )

    def _check_incremental_reservation_invariants(
        self, request_id: str, reservation: KVCapacityReservation
    ) -> None:
        if reservation.request_id != request_id:
            raise RuntimeError(
                f"HBM reservation key does not match request ID "
                f"({request_id!r} != {reservation.request_id!r})")
        if len(reservation.final_shard_bytes) != self.tp_degree:
            raise RuntimeError(
                f"HBM reservation shard count does not match TP "
                f"for {request_id}")
        if reservation.action not in {
            None, "stay", "copy", "recompute",
        }:
            raise RuntimeError(
                f"HBM reservation has an unknown action "
                f"for {request_id}")
        expected_final = kv_cache_shard_bytes_for_tokens(
            self.model,
            reservation.final_context_tokens,
            self.tp_degree,
        )
        if reservation.final_shard_bytes != expected_final:
            raise RuntimeError(
                f"HBM reservation final KV metadata is inconsistent "
                f"for {request_id}")

    # ------------------------------------------------ W4 权威区间目录镜像 --
    #
    # 镜像层纪律（§6.1"渐进迁移"）：目录条目 = 旧字段的同事务重投影
    # （按 §1.9 写入口清单的每一点在本咽喉点收拢），旧字段语义零变化；
    # 目录侧守恒（同内容同持有者唯一 / 引用守恒 / 状态机合法）由
    # KvLocationDirectory.assert_conservation fail-closed 裁决，实例容
    # 量守恒由 _kv_directory_audit 对账 rank 台账。条目通道与增量期望
    # 表三贡献通道（_incremental_*_contribution）逐通道对齐 + 池条目
    # （§7.4 逐出不清 owner：唯一内容在池保留 owner/角色/生成版本）。

    def _kv_directory_sync_sessions(self, session_ids: Iterable[str]) -> None:
        """把变更会话的目录条目集合收敛到旧字段重投影（同事务更新）。"""
        for session_id in dict.fromkeys(session_ids):
            session = self._sessions.get(session_id)
            if session is None:
                # retire_terminal_session：会话终局清理（引用>0 由目录
                # fail-closed——W4 镜像不持引用，恒可清）。
                self._kv_directory.forget_session(
                    session_id, reason="terminal_retire")
                continue
            target_entries = self._kv_directory_target_entries(session)
            target_by_key = {
                (entry.holder, entry.role,
                 entry.content_key.layer_range,
                 entry.content_key.token_range): entry
                for entry in target_entries
            }
            copy_journal = session.copy_handoff

            def _copy_ack_is_closed(ack_id: str) -> bool:
                """Only journal-completed chunks may leave the directory."""
                prefix = "copy-chunk-"
                if not ack_id.startswith(prefix):
                    return False
                try:
                    chunk_index = int(ack_id[len(prefix):])
                except ValueError:
                    return False
                if copy_journal is None:
                    # The journal is cleared only by _close_copy_handoff_journal
                    # after every chunk has been handed off and accounted for.
                    return True
                if not 0 <= chunk_index < len(copy_journal.chunks):
                    return False
                return copy_journal.chunks[chunk_index].state == "HANDED_OFF"

            acknowledged_acks: set[str] = set()
            for current in self._kv_directory.entries_for_session(
                    session_id, valid_only=False):
                match_key = (
                    current.holder, current.role,
                    current.content_key.layer_range,
                    current.content_key.token_range)
                target = target_by_key.get(match_key)
                target_pending = set(target.pending_acks) if target else set()
                acknowledged_acks.update(
                    ack_id for ack_id in current.pending_acks
                    if ack_id not in target_pending
                    and _copy_ack_is_closed(ack_id))
            self._kv_directory.sync_entries(
                session_id, target_entries,
                acknowledged_acks=acknowledged_acks,
                allow_restore_issue=(session.restore_journal is not None))

    def _kv_directory_target_entries(
        self, session: SessionKVState,
    ) -> list[KvDirectoryEntry]:
        """旧字段 → 目标条目集合（纯投影；通道 = 增量期望表三通道+池）。"""
        entries: list[KvDirectoryEntry] = []
        # 通道 A：主工作副本/驻留（_incremental_session_contribution 同式）。
        primary = self._incremental_session_contribution(session)
        copy_journal = session.copy_handoff
        if primary is not None:
            instance_index, shards = primary
            role = (
                ROLE_WORKING if session.working_kind is not None
                else ROLE_RESIDENT)
            # 层覆盖 = 物理驻留前缀（工作副本恒全层——prepare 置
            # resident=L；非工作 PARTIAL 会话 = 驻留前缀，池后缀条目
            # 另列通道 D）。字节真值在 bytes_by_tp_rank（贡献通道同式）。
            # token 域 = 完整上下文 [0, context)（copy/recompute 工作副本
            # 覆盖完整并集）。
            if any(shards) and session.context_tokens > 0:
                entries.append(self._kv_directory_make_entry(
                    session=session, holder=instance_index, role=role,
                    layer_start=0,
                    layer_end=session.resident_prefix_layers,
                    token_start=0,
                    token_end=session.context_tokens,
                    bytes_by_tp_rank=shards,
                    state=self._kv_directory_copy_state(copy_journal),
                    pending_acks=self._kv_directory_copy_acks(copy_journal),
                    release_condition=(
                        "working-copy;mirror-of:shard_bytes")))
        # 通道 B：权威基础历史（_incremental_base_contribution 同式——
        # copy 交接轮字节真值 = journal.home_shards()，层覆盖 = 未释放
        # 块的连续高层层段：块按消费序释放，未释放块恒为尾部连续段）。
        base = self._incremental_base_contribution(session)
        if base is not None:
            instance_index, shards = base
            if copy_journal is not None and copy_journal.chunks:
                released = len(copy_journal.released_indices)
                base_layer_start = (
                    copy_journal.chunks[released].layer_start
                    if released < len(copy_journal.chunks)
                    else copy_journal.chunks[-1].layer_end)
            else:
                base_layer_start = 0
            entries.append(self._kv_directory_make_entry(
                session=session, holder=instance_index, role=ROLE_BASE,
                layer_start=base_layer_start,
                layer_end=session.base_resident_prefix_layers,
                token_start=0, token_end=session.base_history_tokens,
                bytes_by_tp_rank=shards,
                state=self._kv_directory_copy_state(copy_journal),
                pending_acks=self._kv_directory_copy_acks(copy_journal),
                release_condition=(
                    f"base;mirror-of:base_*@{instance_index}")))
        # 通道 C：边界第二来源（_incremental_boundary_source_contribution
        # 同式）。来源侧（i_P 旧副本 = P=copy/recompute 的完整副本）的
        # token 域已含完整并集，投影为 [0, H+U)；否则下一轮目标会重复
        # 物化基础区间。
        source = self._incremental_boundary_source_contribution(session)
        if source is not None:
            instance_index, shards = source
            if session.boundary_source_complete_union:
                token_start = 0
                token_end = (
                    int(session.base_history_tokens)
                    + int(session.boundary_source_history_tokens))
            else:
                token_start = session.base_history_tokens
                token_end = (
                    token_start + session.boundary_source_history_tokens)
            entries.append(self._kv_directory_make_entry(
                session=session, holder=instance_index,
                role=ROLE_BOUNDARY_SOURCE,
                layer_start=0,
                layer_end=(
                    session.boundary_source_resident_prefix_layers
                    or self.model.layers),
                token_start=token_start, token_end=token_end,
                bytes_by_tp_rank=shards,
                state=ENTRY_STATE_READY,
                release_condition="boundary-source;settled-by:merge"))
        # 通道 D：池条目（§7.4 逐出不清 owner——唯一内容在池；恢复在途
        # = restore journal 开放期（RESTORING），池条目不占实例容量）。
        entries.extend(
            self._kv_directory_pool_entries(session))
        # 通道 E（W9 §7.4）：distributed 跨轮留存段——旧基础/边界来源段
        # 在目录提交型结算后仍驻留片上（或经目录 evict_to_pool 转池），
        # 无旧字段描述、无三位置上限。consolidate 缺省恒空（零贡献）。
        entries.extend(self._kv_directory_retained_entries(session.session_id))
        return entries

    def _kv_directory_retained_entries(
        self, session_id: str,
    ) -> list[KvDirectoryEntry]:
        """通道 E 投影：留存段 → 目录条目（owner 记录在 role/history 链）。

        READY 段照实投；EVICTED_TO_POOL 段 holder=池（不占实例容量）；
        RESTORING 段落回目标实例（holder=target，state=RESTORING）。生成
        版本随段携带（settle 提交/驱逐/恢复时 +1——区间条目权威化）。
        """
        entries: list[KvDirectoryEntry] = []
        for segment in self._distributed_retained.get(session_id, {}).values():
            if segment.state == ENTRY_STATE_RESTORING:
                holder = int(segment.restoring_to or 0)
            elif segment.state == ENTRY_STATE_EVICTED_TO_POOL:
                holder = POOL_HOLDER
            else:
                holder = segment.instance_index
            entries.append(KvDirectoryEntry(
                entry_id=(
                    f"{session_id}#retained#{segment.role}#L"
                    f"{segment.layer_start}-{segment.layer_end}#T"
                    f"{segment.token_start}-{segment.token_end}"
                    f"#g{segment.generation}"),
                content_key=KvContentKey(
                    session_id=session_id,
                    model_version=self._kv_directory.model_version,
                    layer_start=segment.layer_start,
                    layer_end=segment.layer_end,
                    token_start=segment.token_start,
                    token_end=segment.token_end,
                ),
                holder=holder,
                role=(
                    ROLE_POOL_BACKED
                    if segment.state == ENTRY_STATE_EVICTED_TO_POOL
                    else segment.role),
                bytes_by_tp_rank=tuple(segment.bytes_by_tp_rank),
                generation=segment.generation,
                state=segment.state,
                release_condition=";".join((
                    "distributed-retained",
                    *segment.history)) or "distributed-retained",
            ))
        return entries

    def _kv_directory_pool_entries(
        self, session: SessionKVState,
    ) -> list[KvDirectoryEntry]:
        """池通道：整份逐出（REMOTE）/ 后缀逐出（PARTIAL 尾层段）/
        池恢复在途（restore journal 开放——issue 字节在途）。"""
        entries: list[KvDirectoryEntry] = []
        restore_journal = session.restore_journal
        if restore_journal is not None:
            issued = restore_journal.issued_bytes_by_rank()
            if any(issued):
                entries.append(self._kv_directory_make_entry(
                    session=session, holder=POOL_HOLDER,
                    role=ROLE_POOL_BACKED,
                    layer_start=int(restore_journal.suffix_start),
                    layer_end=self.model.layers,
                    token_start=0, token_end=session.context_tokens,
                    bytes_by_tp_rank=issued,
                    state=ENTRY_STATE_RESTORING,
                    release_condition=(
                        "pool-restore-in-flight;settled-by:"
                        "prefill_drain/merge/mark_complete")))
                return entries
        if session.distributed_layout and session.working_kind is None:
            # Mixed primary evictions are materialized as channel-E retained
            # rectangles by _evict_suffix/_evict_session.  Reconstructing a
            # legacy single pool rectangle here would lose the distributed
            # token domains and make the directory claim bytes it cannot
            # restore.
            return entries
        if (session.working_kind is not None
                and session.base_history_tokens > 0
                and not (
                    session.base_location in {
                        self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
                    and session.base_resident_prefix_layers > 0)):
            # 跨实例执行期的基础历史仅存池 backing（REMOTE 基/前缀 0
            # 基）：唯一内容在池（§7.4 逐出不清 owner——池条目保留
            # base token 域与字节，SessionKVView 的 missing 口径由此
            # 投影）。
            base_pool_shards = kv_cache_shard_bytes_for_layer_range(
                self.model, session.base_history_tokens, self.tp_degree,
                layer_start=0, layer_end=self.model.layers)
            if any(base_pool_shards):
                entries.append(self._kv_directory_make_entry(
                    session=session, holder=POOL_HOLDER,
                    role=ROLE_POOL_BACKED,
                    layer_start=0, layer_end=self.model.layers,
                    token_start=0, token_end=session.base_history_tokens,
                    bytes_by_tp_rank=base_pool_shards,
                    state=ENTRY_STATE_EVICTED_TO_POOL,
                    release_condition=(
                        "pool-base-backing;cleared-by:merge")))
                return entries
        if session.location == self.REMOTE_MEMORY:
            # 整份逐出：唯一内容全层在池（owner 记录保留——role/generation
            # 由目录登记保持，不清 session 侧 owner 锚点 home_instance）。
            pool_shards = kv_cache_shard_bytes_for_layer_range(
                self.model, session.context_tokens, self.tp_degree,
                layer_start=0, layer_end=self.model.layers)
            if any(pool_shards):
                entries.append(self._kv_directory_make_entry(
                    session=session, holder=POOL_HOLDER,
                    role=ROLE_POOL_BACKED,
                    layer_start=0, layer_end=self.model.layers,
                    token_start=0, token_end=session.context_tokens,
                    bytes_by_tp_rank=pool_shards,
                    state=ENTRY_STATE_EVICTED_TO_POOL,
                    release_condition="pool-backed;restore-on:next-prepare"))
            return entries
        if (
            session.location == self.PARTIAL_HBM_REMOTE
            and session.context_tokens > 0
            and 0 < session.resident_prefix_layers < self.model.layers
        ):
            suffix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model, session.context_tokens, self.tp_degree,
                layer_start=session.resident_prefix_layers,
                layer_end=self.model.layers)
            if any(suffix_shards):
                entries.append(self._kv_directory_make_entry(
                    session=session, holder=POOL_HOLDER,
                    role=ROLE_POOL_BACKED,
                    layer_start=session.resident_prefix_layers,
                    layer_end=self.model.layers,
                    token_start=0, token_end=session.context_tokens,
                    bytes_by_tp_rank=suffix_shards,
                    state=ENTRY_STATE_EVICTED_TO_POOL,
                    release_condition="pool-suffix;restore-on:next-prepare"))
        return entries

    @staticmethod
    def _kv_directory_copy_state(
        copy_journal: Optional["CopyHandoffJournal"],
    ) -> str:
        """copy 交接轮的条目状态（journal ready/ack 协议位挂接目录状态）。"""
        if copy_journal is None or copy_journal.all_settled():
            return ENTRY_STATE_READY
        return ENTRY_STATE_WRITING

    @staticmethod
    def _kv_directory_copy_acks(
        copy_journal: Optional["CopyHandoffJournal"],
    ) -> tuple[str, ...]:
        """未交接块的 ack 义务位（chunk tag——journal 协议位镜像）。"""
        if copy_journal is None:
            return ()
        return tuple(
            f"copy-chunk-{chunk.index}"
            for chunk in copy_journal.pending_chunks())

    def _kv_directory_make_entry(
        self,
        *,
        session: SessionKVState,
        holder: int,
        role: str,
        layer_start: int,
        layer_end: int,
        token_start: int,
        token_end: int,
        bytes_by_tp_rank: tuple[int, ...],
        state: str,
        pending_acks: tuple[str, ...] = (),
        release_condition: str = "",
    ) -> KvDirectoryEntry:
        """目标条目构造（entry_id 由 sync_entries 落地时编定）。"""
        return KvDirectoryEntry(
            entry_id=(
                f"{session.session_id}#{role}#L{layer_start}-{layer_end}"
                f"#T{token_start}-{token_end}"),
            content_key=KvContentKey(
                session_id=session.session_id,
                model_version=self._kv_directory.model_version,
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                token_start=int(token_start),
                token_end=int(token_end),
            ),
            holder=int(holder),
            role=role,
            bytes_by_tp_rank=tuple(int(value) for value in bytes_by_tp_rank),
            state=state,
            pending_acks=tuple(pending_acks),
            release_condition=release_condition,
        )

    def _kv_directory_audit(self, instance_indexes: Iterable[int]) -> None:
        """目录 capacity_projection ↔ rank 台账逐 rank 守恒对账（O(条目数)）。"""
        for instance_index in sorted(set(instance_indexes)):
            instance = self.topology.instance(instance_index)
            projected = self._kv_directory.capacity_projection(instance_index)
            for rank, relative_index in zip(instance.ranks, projected):
                actual = self._rank_states[rank].kv_cache_bytes
                if relative_index != actual:
                    raise RuntimeError(
                        f"kv directory capacity projection mismatch on "
                        f"instance {instance_index} rank {rank}: "
                        f"directory={relative_index}, ledger={actual} "
                        "(directory mirror drifted from the rank ledger)")

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
        # D11：完成/激活/真实释放等账本变化同步增量重索引容量驱逐摘要
        # （同一咽喉点；摘要聚合自身有 strict 全量对账兜底）。
        self._refresh_capacity_summary(changed_sessions)
        # W4：权威区间目录镜像同步（与旧字段同事务——§1.9 全部写点收拢
        # 于本咽喉点）+ 目录自身守恒断言（fail-closed）。
        self._kv_directory_sync_sessions(changed_sessions)
        self._kv_directory.assert_conservation()

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
        # W4：目录投影 ↔ rank 台账逐 rank 守恒对账（受影响实例全查——
        # 含本事务中失去该会话条目的 loser 侧实例）。置于主账本检查之
        # 后：主账本（expected_kv）先行裁决，目录对账兜底镜像漂移。
        self._kv_directory_audit(affected_instances)

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
                    raise RuntimeError(
                        f"remote KV session retained a local instance "
                        f"for {session.session_id}")
                if session.resident_prefix_layers != 0:
                    raise RuntimeError(
                        f"remote KV session retained resident layers "
                        f"for {session.session_id}")
                continue
            if session.location == self.LOCAL_HBM:
                if session.resident_prefix_layers != self.model.layers:
                    raise RuntimeError(
                        f"fully local KV session is missing layers "
                        f"for {session.session_id}")
            elif not (
                0 < session.resident_prefix_layers < self.model.layers
            ):
                raise RuntimeError(
                    f"partial KV session has an invalid prefix length "
                    f"for {session.session_id}")
            if session.instance_index is None:
                raise RuntimeError(
                    f"local KV session has no instance "
                    f"for {session.session_id}")
            instance = self.topology.instance(session.instance_index)
            if len(session.shard_bytes) != instance.size:
                raise RuntimeError(
                    f"KV shard count does not match TP instance "
                    f"for {session.session_id}")
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
            # 无 strict+working 可达组合掩盖了缺口，merge v2 改造
            # 后 strict 档必须与增量通道一致（2026-09-17）。
            base_contribution = self._incremental_base_contribution(session)
            if base_contribution is not None:
                base_instance = self.topology.instance(base_contribution[0])
                for rank, shard_bytes in zip(
                    base_instance.ranks, base_contribution[1]
                ):
                    expected_kv[rank] += shard_bytes
            # 修复轮三 R4（2026-09-28）：边界第二来源（i_P 侧旧工作副本，
            # D19/C4 重建他处）与增量通道同式入全量审计——此前
            # 漏加该项，合法多源状态被 strict 档/终态审计误判泄漏（sweep
            # 实证：rank state=28672 对 expected=0）。
            source_contribution = (
                self._incremental_boundary_source_contribution(session))
            if source_contribution is not None:
                source_instance = self.topology.instance(
                    source_contribution[0])
                for rank, shard_bytes in zip(
                    source_instance.ranks, source_contribution[1]
                ):
                    expected_kv[rank] += shard_bytes
        # W9 通道 E：跨轮留存段与增量通道同式入全量审计（片上 READY 段）。
        for retained_contributions in (
                self._incremental_retained_contributions(session_id)
                for session_id in self._distributed_retained):
            for contribution in retained_contributions:
                retained_instance = self.topology.instance(contribution[0])
                for rank, shard_bytes in zip(
                    retained_instance.ranks, contribution[1]
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
                raise RuntimeError(
                    f"HBM reservation key does not match request ID "
                    f"({request_id!r} != {reservation.request_id!r})")
            if len(reservation.final_shard_bytes) != self.tp_degree:
                raise RuntimeError(
                    f"HBM reservation shard count does not match TP "
                    f"for {request_id}")
            if reservation.action not in {
                None, "stay", "copy", "recompute",
            }:
                raise RuntimeError(
                    f"HBM reservation has an unknown action "
                    f"for {request_id}")
            expected_final = kv_cache_shard_bytes_for_tokens(
                self.model,
                reservation.final_context_tokens,
                self.tp_degree,
            )
            if reservation.final_shard_bytes != expected_final:
                raise RuntimeError(
                    f"HBM reservation final KV metadata is inconsistent "
                    f"for {request_id}")
        for instance in self.topology.instances:
            effective = self._effective_remaining_by_tp_rank(instance.index)
            if any(value < 0 for value in effective):
                raise RuntimeError(
                    f"instance {instance.index} HBM reservations exceed free capacity"
                )
        # D11：容量驱逐摘要全量对账（增量账 vs 从零逐会话重建）——
        # 仅 strict/终态审计路径运行，热路径不扫描会话。
        self._check_capacity_summary_invariants()
        # W4：目录镜像全量审计——目录守恒断言 + 全实例 capacity_projection
        # 对账 rank 台账（镜像写点在 _check_invariants_after_mutation 咽喉
        # 同步；此处兜底全量核对）。
        self._kv_directory.assert_conservation()
        self._kv_directory_audit(
            instance.index for instance in self.topology.instances)

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
        token_start: int = 0,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        counter = self._metrics_segment_counters.get(session_id, 0)
        self._metrics_segment_counters[session_id] = counter + 1
        segment_id = f"seg{counter}"
        # (2026-10-01 R4 审查修复) part 记录**绝对** token 起点（缺省 0 =
        # 自 [0, tokens) 起的全段；增量增长段传 previous_tokens）——
        # _metrics_remove_segment_parts 的盒交集此前把每个 part 当
        # [0, tokens) 解释，token_start>0 的驱逐盒对增量 parts 交集恒
        # 空（镜像零移除、物理侧已扣全量 = 观测账本永久幻影驻留）。
        self._metrics_parts.setdefault(session_id, []).append(
            {
                "segment_id": segment_id,
                "instance_index": instance_index,
                "tokens": int(tokens),
                "token_start": int(token_start),
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

    def _metrics_remove_segment_parts(
        self,
        session_id: str,
        *,
        instance_index: int,
        layer_start: int,
        layer_end: int,
        token_start: int,
        token_end: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Segment-scoped eviction mirror (2026-10-01 R2 重写).

        Retained segments are added as their own parts (matching instance/
        layer extents), so the dominant case is the box FULLY covering a
        part -> remove the whole part (R1 版在全包含形态下 removed=0、
        既不记录也不删 part——净残差永久留在观测账本)。Partial overlap
        clips the part on the layer axis (top or bottom run; a strictly
        interior box splits conceptually -- recorded removal + shrunk
        shards with a conservative extent note, since add_segment never
        creates such shapes)."""

        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.get(session_id, [])
        kept_parts: list[dict[str, Any]] = []
        for part in parts:
            if (part["instance_index"] != instance_index
                    or part["layer_end"] <= layer_start
                    or part["layer_start"] >= layer_end):
                kept_parts.append(part)
                continue
            # (2026-10-01 R4 审查修复) token 轴按**绝对区间**求交：part
            # 覆盖 [token_start_part, token_start_part + tokens)（增量段
            # 自 previous_tokens 起，非自 0），驱逐盒 [token_start,
            # token_end)。旧实现把 part 当 [0, tokens) 解释，增量 parts
            # 与偏移盒交集恒空（镜像零移除、物理侧已扣全量）。
            part_token_start = int(part.get("token_start", 0))
            part_token_end = part_token_start + int(part["tokens"])
            clip_tokens = max(
                0,
                min(part_token_end, token_end)
                - max(part_token_start, token_start))
            if clip_tokens <= 0:
                kept_parts.append(part)
                continue
            clip_layer_start = max(part["layer_start"], layer_start)
            clip_layer_end = min(part["layer_end"], layer_end)
            removed = kv_cache_shard_bytes_for_layer_range(
                self.model, clip_tokens, self.tp_degree,
                layer_start=clip_layer_start, layer_end=clip_layer_end)
            if clip_layer_start == part["layer_start"]                     and clip_layer_end == part["layer_end"]                     and clip_tokens >= part["tokens"]:
                # full containment: whole-part removal
                if any(removed):
                    self._metrics_remove_part(
                        session_id, part, removed,
                        anchor_kind=anchor_kind, request_id=request_id,
                        cause=cause)
                continue
            if any(removed):
                self._metrics_remove_part(
                    session_id, part, removed,
                    anchor_kind=anchor_kind, request_id=request_id,
                    cause=cause)
            residual = tuple(
                int(old) - int(gone)
                for old, gone in zip(part["shards"], removed))
            if any(residual):
                # (2026-10-01 R3 审查修复) 层轴截除方向与姊妹实现
                # （_metrics_suffix_evict_parts/_metrics_prefix_release_parts）
                # 对齐：residual 是**保留**字节，extents 必须收缩到保留侧——
                # 剪掉高层段 [clip_layer_start, layer_end) ⇒ 保留低段，
                # layer_end = clip_layer_start；剪掉低层段 ⇒ layer_start =
                # clip_layer_end。R2 版两分支赋值颠倒（保留字节配被删
                # 区间 extents，层宽与 shards 失配）。
                if clip_layer_end == part["layer_end"]:
                    part["layer_end"] = clip_layer_start
                elif clip_layer_start == part["layer_start"]:
                    part["layer_start"] = clip_layer_end
                else:
                    # strictly interior: extents stay conservative
                    pass
                part["tokens"] = max(0, part["tokens"] - clip_tokens)
                # 盒覆盖 part 的 token 低侧时，保留段绝对起点随盒终点
                # 前移；高侧/内部截除起点不动（保守 extents 同层轴纪律）。
                if token_start <= part_token_start < token_end:
                    part["token_start"] = max(part_token_start, token_end)
                part["shards"] = residual
                kept_parts.append(part)
        if kept_parts:
            self._metrics_parts[session_id] = kept_parts
        else:
            self._metrics_parts.pop(session_id, None)

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
        resident_prefix_layers``——驻留恒为前缀形态（不变量锚：
        ``_check_incremental_session_invariants``，KV 容量账增量 strict
        对账的前缀形态判定），
        区间 ⊆ [0, resident_prefix) ⟺ 所传字节物理在场（旧全驻留守卫是
        ``layer_end == L`` 特例，语义严格变宽而非放松）。字节源按层区间
        从 context_tokens 派生（隐式依赖准入不变量的
        ``context_tokens == history_tokens``，登记 PROVENANCE P3-b），
        **不读** ``session.shard_bytes``（全量声明口径，PARTIAL 下是幻影
        字节源——D2-1 根因）。``resident_prefix_layers_before/after`` 沿
        本仓区间传输惯例 = layer_start/layer_end（同例：
        ``_remote_load_transfer``/``_remote_store_transfer``/
        ``_working_copy_noc_transfer``）。
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
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        """C15：后缀 [suffix_start, L) 的逐组恢复发射计划 + 区间账本。

        按 plan_suffix_restore_groups 切分（消费顺序 = 层自低向高；
        后缀 ≤ RESTORE_GROUP_LAYERS 时单组 = 旧单笔口径回归锚——
        transfers 数量/区间/字节不变，仅多 restore_group 标记）。账本
        科目 ``rid#restore`` 的 issue 事件在此落账（complete 于 prefill
        drain/merge 兜底；守恒断言见 RestoreGroupJournal）。逐组物理
        到达由构图器旁挂恢复支链承载（GB _suffix_restore_arms），列车
        体按层段就绪门控消费——替换"整列车等整段后缀"的保守门。

        M4（2026-09-26）：issue 事件携带可选 ``at_ns``（发射时刻；缺省
        None 保确定性测试）——与 complete 事件配对得恢复窗口下界。
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
            "at_ns": at_ns,
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
        # 层段字节统一经 helper 公式（单一事实源）。
        transfer_shards = self._distributed_resident_layer_bytes(
            session, layer_start, layer_end)
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
                # R4-P1（2026-09-29 face 移植）：session_id str 化——混排
                # （int/str 混合）键集在 last_completion_ns 平局时对原始
                # 值元组比较即 TypeError（首次驱逐排序崩溃）；str 键恒有
                # 全序，官方 CSV 路径（session_id 本就是 str）序不变。
                str(session.session_id),
            )
        )
        return candidates

    @staticmethod
    def _eviction_class(session: SessionKVState) -> str:
        # 类别判定唯一来源 = joint.eviction_priority（SH fallback 裁定：
        # 仅显式 "tool" 归工具类，human/None 及未知值按 human 优先级；
        # 未知值的原始值与 fallback 标记披露走 classify_with_fallback）。
        return classify_session_class(session.next_request_type)

    def _completed_resident_candidates(
        self,
        instance_index: int,
        trigger_type: Optional[str] = None,
    ) -> list[SessionKVState]:
        # D20（v2.0 §14.5）：持阶段引用的会话不进冷 victim 集合——
        # P 末尾到 D 提交期间本请求基础 KV 保持阶段引用保护，不因
        # "暂时没有活跃 D"被其他请求的容量事务选中逐出。
        return self._fifo_sort([
            session
            for session in self._sessions.values()
            if session.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
            and session.instance_index == instance_index
            and not session.active
            and session.last_completion_ns is not None
            and session.session_id not in self._stage_references
            and (
                trigger_type is None
                or self._eviction_class(session) == trigger_type
            )
        ])

    def _retain_distributed_primary_parts(
        self,
        session: SessionKVState,
        *,
        layer_start: int,
        layer_end: int,
        trigger_request_id: str,
    ) -> None:
        """Move removed mixed primary rectangles into channel E.

        A distributed primary after settlement covers the complete union
        ([0, context) over the resident prefix).  Keeping those rectangles
        as retained segments makes a later restore addressable by the same
        directory protocol as any other source.
        """
        if not (session.distributed_layout and session.working_kind is None):
            return
        if session.instance_index is None:
            raise RuntimeError(
                f"distributed primary {session.session_id} has no source "
                "instance while being evicted")
        instance_index = int(session.instance_index)
        registry = self._distributed_retained.setdefault(
            session.session_id, {})

        parts = []
        if session.context_tokens > 0 and int(layer_start) < int(layer_end):
            parts.append((
                int(layer_start), int(layer_end),
                0, int(session.context_tokens),
                kv_cache_shard_bytes_for_layer_range(
                    self.model, session.context_tokens, self.tp_degree,
                    layer_start=int(layer_start), layer_end=int(layer_end)),
                "distributed-primary-resident"))
        for part_start, part_end, token_start, token_end, shards, role in parts:
            if not any(shards):
                continue
            segment = RetainedSegment(
                session_id=session.session_id,
                instance_index=instance_index,
                layer_start=part_start,
                layer_end=part_end,
                token_start=token_start,
                token_end=token_end,
                bytes_by_tp_rank=tuple(int(value) for value in shards),
                role=ROLE_RESIDENT,
                state=ENTRY_STATE_EVICTED_TO_POOL,
                history=(f"{role}:{trigger_request_id}",),
            )
            if segment.key in registry:
                raise RuntimeError(
                    f"distributed primary eviction collides with retained "
                    f"segment {segment.key} for session {session.session_id}")
            registry[segment.key] = segment

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
        removed_shards = self._distributed_resident_layer_bytes(
            session, suffix_start, resident_layers)
        source_instance = int(session.instance_index)
        self._retain_distributed_primary_parts(
            session,
            layer_start=suffix_start,
            layer_end=resident_layers,
            trigger_request_id=trigger_request_id,
        )
        self._remove_local_shards(source_instance, removed_shards)
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
        local_shards = self._distributed_resident_layer_bytes(
            session, 0, resident_layers)
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_full_fallback",
            session=session,
            trigger_request_id=trigger_request_id,
            layer_start=0,
            layer_end=resident_layers,
            resident_prefix_layers_after=0,
        )
        source_instance = int(session.instance_index)
        self._retain_distributed_primary_parts(
            session,
            layer_start=0,
            layer_end=resident_layers,
            trigger_request_id=trigger_request_id,
        )
        self._remove_local_shards(source_instance, local_shards)
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
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        """M4（2026-09-26）：``at_ns`` 透传 adaptive 决策披露行（缺省
        None 保确定性测试）；SH 在线调用面传 tick，决策行带时刻供两阶
        段决策验证。驱逐计划与等待时间零变化。"""
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

            def layer_group_bytes(
                session: SessionKVState,
                layer_start: int,
                layer_end: int,
            ) -> tuple[int, ...]:
                return self._distributed_resident_layer_bytes(
                    session, layer_start, layer_end)

            victims = [
                VictimView(
                    session_id=session.session_id,
                    resident_prefix_layers=session.resident_prefix_layers,
                    layer_group_bytes_fn=(
                        lambda start, end, s=session: layer_group_bytes(
                            s, start, end)),
                    retention_target_layers=0,
                    retention_target_layers_fn=(
                        lambda sid=session.session_id,
                        ctx=session.context_tokens,
                        inst=instance_index,
                        at=at_ns:
                        self._adaptive_retention_target_tokens(
                            sid, ctx, inst, at)),
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
            # W9（§7.4 区间级 E）：主位置 victim 耗尽仍不足时逐留存段驱逐
            # （目录 owner 过滤——段所在实例即持有者；FIFO 序继承会话完
            # 成时刻；驱逐经目录 evict_to_pool：owner 记录保留、唯一内容
            # 在池）。distributed 缺省无留存段 → 零行为差。
            if any(gap):
                for victim, segment in self._completed_retained_segments(
                        instance_index, trigger_type):
                    if protected_session_id is not None and (
                            victim.session_id == protected_session_id):
                        continue
                    evictions.append(
                        self._evict_retained_segment(
                            victim, segment,
                            phase=phase,
                            reason=reason,
                            trigger_request_id=trigger_request_id,
                        )
                    )
                    gap = _gap_by_tp_rank()
                    if not any(gap):
                        break
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
        # (2026-10-01 审查修复) "撤销最后一笔"减的应是**末笔在该 rank 的
        # 字节**——旧实现减全部逐出的累计（≥ 末笔），不等式更易成立，
        # 真实过度逐出被系统性漏检（审计-only 防线弱于声明）。
        if evictions:
            last_freed_by_rank: dict[int, int] = {}
            for shard in evictions[-1].shards:
                if shard.source_rank is None:
                    continue
                last_freed_by_rank[shard.source_rank] = (
                    last_freed_by_rank.get(shard.source_rank, 0)
                    + shard.bytes
                )
            effective_now = self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=reservation_request_id,
            )
            for rank, freed in last_freed_by_rank.items():
                relative = self._rank_relative_index[rank]
                if (effective_now[relative] - freed
                        < required_bytes_by_tp_rank[relative]):
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
        at_ns: Optional[int] = None,
    ) -> tuple[
        Optional[SessionKVSnapshot],
        tuple[KVTransfer, ...],
        tuple[KVTransfer, ...],
    ]:
        """为本轮执行准备历史（joint 三动作语义，设计方案 §2.2 表）。

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

        返回 ``(before, transfers, evictions)``：transfers 为需要物理
        发射的 KV 传输序列（stay-local 为空）；home 语义见 §2.1——
        异地执行不改变逻辑 home，逐出也不清除。
        """
        if action not in ("stay", "copy", "recompute"):
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
        if session.distributed_layout:
            # W9（§7.4 distributed 跨轮留存）：distributed settle 后会话
            # 内容跨多位置分布（目录权威）——prepare 走目录驱动分支
            # （覆盖断言取代单位置等式；来源/池恢复按目录条目迭代）。
            # consolidate 缺省不进此分支（distributed_layout 恒 False）。
            return self._prepare_prefill_distributed(
                session=session,
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                expected_shards=expected_shards,
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
                action=action,
                at_ns=at_ns,
            )
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
        # active 早置刻意（防自选 victim：本请求的容量腾退不得逐出自己
        # 的基础驻留）；_ensure_capacity 异常路径依赖重试幂等重入与请求
        # 终局复位自愈，不补异常复位（会在回退窗口引入基础驻留被逐出
        # 风险）。
        session.active = True

        base_location = session.location
        base_instance = session.instance_index
        base_prefix = session.resident_prefix_layers
        exec_instance = target_instance_index

        def _snapshot_base() -> None:
            session.base_location = base_location
            session.base_history_tokens = history_tokens
            session.base_resident_prefix_layers = base_prefix
            session.base_instance_index = base_instance

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
                    at_ns=at_ns,
                ))
                transfers.extend(self._plan_suffix_restore_transfers(
                    session=session,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=exec_instance,
                    suffix_start=suffix_start,
                    history_tokens=history_tokens,
                    reason="history_remote_suffix_restore",
                    at_ns=at_ns,
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
            # 守卫只拦 LOCAL，PARTIAL 落穿 Case B 后会在建立工作副本时
            # 把整份工作副本叠在驻留前缀上双计、完成结算撞 N9 防御确定
            # 性 raise（适用性现排除该组合，但"不可达"不是"无害"）。镜像
            # stay-partial 模板退化：只恢复缺失后缀、不建工作副本、
            # working_kind 保持 None（与适用性注释"退化为 stay、不构成第
            # 二份工作副本"的声明语义对齐——实现追上文档）。
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
                at_ns=at_ns,
            ))
            transfers.extend(self._plan_suffix_restore_transfers(
                session=session,
                trigger_request_id=trigger_request_id,
                target_instance_index=exec_instance,
                suffix_start=suffix_start,
                history_tokens=history_tokens,
                reason="history_remote_suffix_restore",
                at_ns=at_ns,
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
            # R13/N9 本地事务骨架 + D10 语义重定义（v2.0 §6.2，
            # 2026-09-26）：recompute@home——驻留字节不清零、不孤儿化、
            # 不建工作副本（与 stay 同构的本地事务）；**账本字节是空间
            # 复用口径，不是计算口径**——完整 H+U 重放不依赖旧 KV，驻留
            # 旧分配仅转作重建空间（缺失后缀层 [base_prefix, L) 的
            # prepare 入账 = 重建上下文的空间占位，重放物理写满全层；
            # "只重算缺失后缀层"的旧折算语义已取消）。重算的物理计算
            # 量由调度器折入列车 span（joint_prefill_work = 整份 H +
            # input，span 基 = 0，见 sh30
            # _apply_joint_recompute_work_description）。
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
                    at_ns=at_ns,
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
                    at_ns=at_ns,
                ))
                transfers.extend(self._plan_suffix_restore_transfers(
                    session=session,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=exec_instance,
                    suffix_start=0,
                    history_tokens=history_tokens,
                    reason="history_pool_restore_working_copy",
                    at_ns=at_ns,
                ))
                self._add_local_shards(exec_instance, full_shards)
                session.shard_bytes = full_shards
                session.total_bytes = sum(full_shards)
            elif action == "recompute":
                # 重算：工作副本 0 起步（重算 prefill 物化）。
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
                at_ns=at_ns,
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
                    at_ns=at_ns,
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

        session.location = self.LOCAL_HBM
        session.instance_index = exec_instance
        session.resident_prefix_layers = self.model.layers
        session.working_kind = action
        self._check_invariants_after_mutation(session_ids=(session_id,))
        return before, tuple(transfers), tuple(evictions)

    # ---------------------------------- W9（§7.4）目录驱动 prepare 分支 --

    def _distributed_directory_entries(self, session_id: str):
        """会话的目录条目快照（片上/池/恢复在途三组；确定性序）。"""
        on_chip = []
        pooled = []
        for entry in self._kv_directory.entries_for_session(session_id):
            if not entry.is_valid():
                continue
            if entry.is_pool_backed():
                pooled.append(entry)
            else:
                on_chip.append(entry)
        return on_chip, pooled

    def _assert_distributed_union_coverage(
        self, session: SessionKVState, history_tokens: int,
    ) -> None:
        """W9 覆盖断言：目录有效内容并集 == 期望全量（fail-closed）。

        取代 legacy 单位置 ``context_tokens == history_tokens`` 等式——
        distributed 会话内容跨多位置，权威事实在目录。覆盖按**单元格级**
        并集审计（``assert_full_content_locatable``——distributed 跨轮
        留存段会产生跨持有者的部分重叠区间，区间身份去重会把重叠字节
        计两遍；单元格面积完备 + 本仓 KV 字节线性 ⇔ 字节完备）。数据
        未就绪/目录漂移在此拦截（早到下一轮的消费受数据与目录依赖限
        制，§7.4）。"""
        assert_full_content_locatable(
            self._kv_directory, session.session_id,
            history_tokens=history_tokens,
            model_layers=self.model.layers)

    def _retained_segment_of_entry(
        self, session_id: str, entry,
    ) -> Optional[tuple[int, int, int, int, int]]:
        """目录条目 → 留存段键（通道 E 条目；非留存条目返回 None）。"""
        key = entry.content_key
        candidate = (
            int(entry.holder), key.layer_start, key.layer_end,
            key.token_start, key.token_end)
        if candidate in self._distributed_retained.get(session_id, {}):
            return candidate
        # 池/恢复在途条目 holder 已换：按内容区间反查。
        for segment in self._distributed_retained.get(session_id, {}).values():
            if (segment.layer_start, segment.layer_end) == key.layer_range \
                    and (segment.token_start, segment.token_end) == key.token_range:
                return segment.key
        return None

    def _checkout_retained_base_anchor(
        self, session: SessionKVState,
    ) -> None:
        """把 base 锚留存段（token_start 最小的片上段）检入 legacy base_*。

        下游（credit plan 的 prefix 口径/边界视图/逐出保护）读 base_*——
        检出段同时从通道 E 移除（守恒单计：base_* 通道接管）。settle 时
        base_* 重新移入通道 E。"""
        session_id = session.session_id
        segments = self._distributed_retained.get(session_id, {})
        on_chip_segments = [
            segment for segment in segments.values()
            if segment.is_on_chip() and any(segment.bytes_by_tp_rank)]
        if not on_chip_segments:
            return
        # 锚必须是从 layer 0 开始的 token-root 段；后缀 S 也从 token 0
        # 开始，但不能冒充完整前缀，否则下一轮 credit 会把 suffix 当
        # 作 [0, p) 的权威基础。
        root_segments = [
            segment for segment in on_chip_segments
            if segment.token_start == 0 and segment.layer_start == 0]
        if not root_segments:
            return
        pool_to_pick = root_segments
        anchor = max(
            pool_to_pick,
            key=lambda segment: (
                segment.token_end, segment.layer_end, segment.key))
        session.base_location = (
            self.LOCAL_HBM
            if anchor.layer_end >= self.model.layers
            else self.PARTIAL_HBM_REMOTE)
        session.base_history_tokens = anchor.token_end
        session.base_resident_prefix_layers = anchor.layer_end
        session.base_instance_index = anchor.instance_index
        segments.pop(anchor.key)

    def _prepare_prefill_distributed(
        self,
        *,
        session: SessionKVState,
        target_instance_index: int,
        history_tokens: int,
        expected_shards: tuple[int, ...],
        trigger_request_id: str,
        reservation_request_id: Optional[str],
        action: str,
        at_ns: Optional[int] = None,
    ) -> tuple[
        Optional[SessionKVSnapshot],
        tuple[KVTransfer, ...],
        tuple[KVTransfer, ...],
    ]:
        """W9 distributed 会话的目录驱动 prepare（§7.4 跨轮留存面）。

        与 legacy 分支的差别：内容跨多位置（目录权威迭代来源/池区间）；
        元数据校验 = 目录覆盖断言；copy 不走 chunk
        handoff（来源段全部保留——distributed 不释放来源，chunk 释放是
        consolidate 的优化）；base 锚留存段检出入 base_*（下游口径兼容，
        守恒单计）。
        """
        session_id = session.session_id
        exec_instance = int(target_instance_index)
        before = self.session_snapshot(session_id)
        self._assert_distributed_union_coverage(
            session, history_tokens)
        # active 早置（同 legacy：防自选 victim）。
        session.active = True

        transfers: list[KVTransfer] = []
        evictions: list[KVTransfer] = []

        # 目标缺失区间（目录权威；单元格并集去重——目标已持/在途不重
        # 搬，同内容多副本冻结序只选读一副本，部分重叠只补差集片段；
        # 条目直和 = 缺失并集字节，§6.1/§6.3）。逐片段按来源发射。
        missing = self._kv_directory.query_missing_intervals(
            session_id, exec_instance)
        missing_entries = [
            entry for entry in missing if any(entry.bytes_by_tp_rank)]

        # 池区间恢复（目录驱动；只恢复目标缺失并集中的池片段）。目录
        # 可能同时保留多个重叠副本；无条件扫描/恢复全部池条目会把目标
        # 已持有的内容再次物化，既抬高 HBM 又破坏“缺失区间只搬一次”
        # 守恒。query_missing_intervals 已按目标驻留/在途覆盖做过单元格
        # 并集去重，后续所有动作只消费其中的池条目。
        pool_entries = tuple(
            entry for entry in missing_entries if entry.is_pool_backed())

        def _restore_pool_segments_for(needed_entries) -> tuple[int, ...]:
            restored = [0] * self.tp_degree
            for entry in needed_entries:
                key = self._retained_segment_of_entry(session_id, entry)
                if key is None:
                    raise RuntimeError(
                        f"distributed pool entry {entry.entry_id} for "
                        f"session {session_id} has no retained owner")
                transfers.append(self.restore_retained_segment(
                    session_id, key, target_instance=exec_instance,
                    trigger_request_id=trigger_request_id, at_ns=at_ns))
                # 物化入账（与 legacy stay-partial 准入相恢复同款同步
                # 入账模式：转移腿发射 + 台账即时物化）。
                self.complete_retained_restore(session_id, key)
                for rank, value in enumerate(entry.bytes_by_tp_rank):
                    restored[rank] += int(value)
            return tuple(restored)

        if action == "stay":
            # §7.4 跨轮留存的 stay 语义：**目录裁决**目标是否持完整并集
            #（primary 只描述当前轮工作副本——完整并集可在任一留存位置；
            # stay 到该位置 = 并集段转正为新 primary，旧 primary 移入通
            # 道 E，零字节移动）。
            if not self._distributed_exec_holds_union(
                    session, exec_instance, history_tokens):
                if pool_entries:
                    # (2026-10-01 审查修复) 池段恢复前先做 T+E 容量预备
                    #（copy/recompute 分支同款单失败点）：complete_
                    # retained_restore 的 _add_local_shards 容量不足时
                    # 抛 KVCapacityError，会把留存段永久卡在 RESTORING
                    #（重试改撞"not pool-backed (state=RESTORING)"合同
                    # 错——可恢复失败被降级为不可恢复）。预备先行使失败
                    # 发生在任何状态变更之前（§16.4 回退半边零移动守卫）。
                    pool_total = tuple(
                        sum(entry.bytes_by_tp_rank[rank]
                            for entry in pool_entries)
                        for rank in range(self.tp_degree))
                    if any(pool_total):
                        evictions.extend(self._ensure_capacity(
                            exec_instance,
                            pool_total,
                            phase="history",
                            reason="history_stay_pool_restore_capacity",
                            trigger_request_id=trigger_request_id,
                            reservation_request_id=reservation_request_id,
                            protected_session_id=session_id,
                            at_ns=at_ns,
                        ))
                    _restore_pool_segments_for(pool_entries)
                if not self._distributed_exec_holds_union(
                        session, exec_instance, history_tokens):
                    raise RuntimeError(
                        f"stay action under distributed layout requires "
                        f"the full history resident at the execution "
                        f"instance (session {session_id}, target "
                        f"{exec_instance}) -- union spans other positions; "
                        "pick copy/recompute instead")
            if session.instance_index != exec_instance:
                self._migrate_primary_to_retained(session, trigger_request_id)
                absorbed = self._absorb_retained_at_instance(
                    session, exec_instance, trigger_request_id)
                session.shard_bytes = absorbed
                session.total_bytes = sum(absorbed)
                session.context_tokens = history_tokens
                # primary 迁至并集持有位置（守恒期望表随实例键更新）。
                session.instance_index = exec_instance
            transfers.append(self._local_hit_transfer(
                phase="history",
                reason="history_local_reuse",
                session=session,
                trigger_request_id=trigger_request_id,
            ))
            self._check_invariants_after_mutation(session_ids=(session_id,))
            return before, tuple(transfers), tuple(evictions)

        # ---- 跨实例/重建动作：先做描述迁移（无字节移动）。 ----
        # 换目标时旧 primary 位置移入通道 E；copy 族在目标把既有留存段
        # 并入工作副本描述（absorb——守恒单计：并入段退出通道 E）。
        moved_primary = False
        if session.instance_index != exec_instance:
            self._migrate_primary_to_retained(session, trigger_request_id)
            moved_primary = True

        if action == "recompute":
            # 重算：目标工作副本 0 起步（不依赖旧 KV；来源段全部保留）；
            # 工作副本 = 完整上下文 [0, context)。
            evictions.extend(self._ensure_capacity(
                exec_instance,
                expected_shards,
                phase="history",
                reason="history_recompute_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
                protected_session_id=session_id,
                at_ns=at_ns,
            ))
            session.shard_bytes = tuple(0 for _ in expected_shards)
            session.total_bytes = 0
            session.context_tokens = 0
        elif action == "copy":
            # 目录驱动多源 copy：逐缺失区间自来源 holder 发射 NoC 腿；
            # 池区间恢复物化；目标既有留存段并入工作副本描述。来源段全
            # 部保留（不释放——distributed 留存语义；chunk handoff 的逐
            # 块源释放是 consolidate 的优化路径）。
            absorbed = self._absorb_retained_at_instance(
                session, exec_instance, trigger_request_id)
            if moved_primary:
                session.shard_bytes = absorbed
                session.total_bytes = sum(absorbed)
                session.context_tokens = 0
            else:
                session.shard_bytes = tuple(
                    a + b for a, b in zip(session.shard_bytes, absorbed))
                session.total_bytes = sum(session.shard_bytes)
            source_groups: dict[int, list] = {}
            for entry in missing_entries:
                if entry.is_pool_backed():
                    continue
                source_groups.setdefault(int(entry.holder), []).append(entry)
            total_missing = tuple(
                sum(entry.bytes_by_tp_rank[rank] for entry in missing_entries)
                for rank in range(self.tp_degree))
            evictions.extend(self._ensure_capacity(
                exec_instance,
                total_missing,
                phase="history",
                reason="history_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
                protected_session_id=session_id,
                at_ns=at_ns,
            ))
            if pool_entries:
                _restore_pool_segments_for(pool_entries)
            working_bytes = [0] * self.tp_degree
            for source_instance in sorted(source_groups):
                if source_instance == exec_instance:
                    continue
                for entry in source_groups[source_instance]:
                    key = entry.content_key
                    transfers.append(self._shard_vector_noc_transfer(
                        session_id=session_id,
                        trigger_request_id=trigger_request_id,
                        source_instance_index=source_instance,
                        target_instance_index=exec_instance,
                        shard_bytes=tuple(entry.bytes_by_tp_rank),
                        reason="history_distributed_copy_leg",
                    ))
                    for rank, value in enumerate(entry.bytes_by_tp_rank):
                        working_bytes[rank] += int(value)
                    self._metrics_add_segment(
                        session_id, exec_instance, history_tokens,
                        key.layer_start, key.layer_end,
                        tuple(entry.bytes_by_tp_rank),
                        anchor_kind="prefill_start",
                        request_id=trigger_request_id,
                        cause="history_distributed_copy_leg",
                    )
            # 物化入账（与 legacy REMOTE 基 copy 准入相同步入账模式）。
            self._add_local_shards(exec_instance, tuple(working_bytes))
            session.shard_bytes = tuple(
                a + b for a, b in zip(session.shard_bytes, working_bytes))
            session.total_bytes = sum(session.shard_bytes)
            # copy 的 context 覆盖完整并集（工作副本持有并集——与 legacy
            # copy 口径一致；settle 的零字节翻转族守恒断言据此成立）。
            session.context_tokens = history_tokens
            self._checkout_retained_base_anchor(session)

        session.location = self.LOCAL_HBM
        session.instance_index = exec_instance
        session.resident_prefix_layers = self.model.layers
        session.working_kind = action
        self._check_invariants_after_mutation(session_ids=(session_id,))
        return before, tuple(transfers), tuple(evictions)

    def _absorb_retained_at_instance(
        self, session: SessionKVState, instance_index: int,
        trigger_request_id: str,
    ) -> tuple[int, ...]:
        """copy 族目标实例既有留存段并入工作副本描述（absorb）。

        物理字节不动（已在目标台账）；描述从通道 E 移交 channel A——
        工作副本 = 目标并集（与 legacy copy 的"exec 持完整并集"口径一
        致，settle 守恒断言据此闭合）。返回并入字节向量。"""
        registry = self._distributed_retained.setdefault(
            session.session_id, {})
        absorbed = [0] * self.tp_degree
        for key in [
                key for key, segment in registry.items()
                if segment.is_on_chip()
                and segment.instance_index == instance_index]:
            segment = registry.pop(key)
            for rank, value in enumerate(segment.bytes_by_tp_rank):
                absorbed[rank] += int(value)
            self.distributed_directory_meter.record_op("retained_absorb")
        return tuple(absorbed)

    def _distributed_exec_holds_union(
        self, session: SessionKVState, instance_index: int,
        history_tokens: int,
    ) -> bool:
        """目录判定：instance 是否独立持有完整并集（单元格级覆盖——
        同实例重叠区间按格并集，与 _assert_distributed_union_coverage
        同审计器）。"""
        if instance_index is None:
            return False
        from joint.distributed_settlement import cell_coverage_complete
        complete, covered = cell_coverage_complete(
            self._kv_directory, session.session_id,
            history_tokens=history_tokens,
            model_layers=self.model.layers,
            holder=instance_index)
        return complete and covered == history_tokens * self.model.layers

    def _migrate_primary_to_retained(
        self, session: SessionKVState, trigger_request_id: str,
    ) -> None:
        """当前位置（primary 描述的字节）移入通道 E（描述职责移交）。

        零字节移动——登记留存段（token 域 = 完整并集形态 [0, context)）
        并清零 primary 描述（守恒单计：channel A → channel E）。"""
        if session.instance_index is None or not any(session.shard_bytes):
            session.shard_bytes = tuple(
                0 for _ in session.shard_bytes) if session.shard_bytes else ()
            return
        resident_layers = int(session.resident_prefix_layers)
        layer_bytes = self._distributed_resident_layer_bytes(
            session, 0, resident_layers)
        if not any(layer_bytes):
            return
        registry = self._distributed_retained.setdefault(
            session.session_id, {})
        segments = []
        segments.append((
            0, resident_layers, 0,
            int(session.context_tokens), layer_bytes))
        for layer_start, layer_end, token_start, token_end, shards in segments:
            if not any(shards):
                continue
            segment = RetainedSegment(
                session_id=session.session_id,
                instance_index=int(session.instance_index),
                layer_start=int(layer_start),
                layer_end=int(layer_end),
                token_start=int(token_start),
                token_end=int(token_end),
                bytes_by_tp_rank=tuple(int(value) for value in shards),
                role=ROLE_RESIDENT,
                history=(f"primary-migrated:{trigger_request_id}",),
            )
            if segment.key in registry:
                raise RuntimeError(
                    f"primary migration collides with retained segment "
                    f"{segment.key} for session {session.session_id}")
            registry[segment.key] = segment
        session.shard_bytes = tuple(0 for _ in layer_bytes)
        session.total_bytes = 0
        session.context_tokens = 0

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
        at_ns: Optional[int] = None,
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
            # 就绪门控的消费保证已在图侧兑现）。M4：at_ns 随边界透传。
            self._settle_restore_groups(
                session_id, boundary="prefill_drain", at_ns=at_ns)
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
            at_ns=at_ns,
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
            token_start=previous_tokens,
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
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        return self._expand_local_session(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=context_tokens,
            phase="prefill",
            reason="prefill_growth_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
            at_ns=at_ns,
        )

    def move_prefill_to_decode(
        self,
        *,
        session_id: str,
        target_instance_index: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
        at_ns: Optional[int] = None,
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
            at_ns=at_ns,
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
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        return self._expand_local_session(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=final_context_tokens,
            phase="decode",
            reason="decode_growth_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
            at_ns=at_ns,
        )

    # ------------------------------------- C4 D 边界工作副本事务（D19/D10）--

    def commit_boundary_decode_transition(
        self,
        *,
        session_id: str,
        p_instance_index: int,
        target_instance_index: int,
        action: str,
        history_tokens: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, tuple[KVTransfer, ...], dict]:
        """D 边界工作副本事务（C4，v2.0 §19.1/D10/D19 执行面）：按选中
        D 动作处置 i_P 侧工作副本与目标 j 侧新工作副本——与
        ``move_prefill_to_decode`` 同源物理事务（stay/copy 直接
        复用其 local_hit/整份迁移路径）。

        * ``stay``（j==i_P）：本地命中（零字节）。
        * ``copy``（按缺失区间搬运）：
          - j≠i_P：H_B 整份迁移（旧路径复用）；
          - 回 home：就地并基折叠（working 折叠为本地全上下文）。
        * ``recompute``（工作副本不迁移，D10）：j≠i_P 时旧副本转来源侧
          记录（base_* 空闲则记 base，否则记 boundary_source——片上位
          置 ≤3）；j 侧工作副本零化起步（重建由 decode 增长钩子物化，
          驻留旧分配仅空间复用不免 FLOP）。j==i_P 原地重建。

        返回 ``(primary_transfer, evictions, outcome)``：primary =
        joiner 列车 ``prefill_decode_transfer`` 槽位的迁移腿（copy 多
        腿时为按腿序的元组）；outcome = 披露摘要（kind/sources/
        moved_bytes）。调用时点 = D20 阶段预约事务提交之后（容量承诺
        先行），字节搬运随加入列车发射（此刻零字节移动）。
        """
        if action not in ("stay", "copy", "recompute"):
            raise ValueError(f"unknown boundary decode action: {action!r}")
        session = self._sessions[session_id]
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError(
                "boundary decode transition requires a local working copy")
        if session.instance_index != int(p_instance_index):
            raise RuntimeError(
                f"session {session_id} working copy is not at the boundary "
                f"P instance {p_instance_index} (at "
                f"{session.instance_index})")
        target = int(target_instance_index)
        outcome: dict = {"action": action, "sources": (), "moved_bytes": 0}
        if action in ("stay", "copy"):
            # stay / copy：既有物理事务（local_hit / 整份迁移）。
            if (action == "copy"
                    and session.base_instance_index is not None
                    and target == int(session.base_instance_index)
                    and session.base_location in {
                        self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
                    and session.base_resident_prefix_layers > 0):
                # copy 落到自身 base 驻留实例 = R13 消除的组合（选点面
                # 不可见 base 驻留位，计价无法排除）——退化为"就地并基"：
                # 迁移腿照发（数据仍在 i_P），落点即 base 驻留侧。C13
                # 逐块交接形态下 home 侧基础字节已在轮内逐块释放，本处
                # 只清账闭合 journal；无 journal 的重复基础副本（增量型
                # 基材料化来源）就地释放，不折叠会在轮末 merge 撞 N9
                # 防御确定性 raise。
                fold_journal = session.copy_handoff
                fold_base_prefix = session.base_resident_prefix_layers
                fold_base_history = session.base_history_tokens
                transfer, evictions = self.move_prefill_to_decode(
                    session_id=session_id,
                    target_instance_index=target,
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                    at_ns=at_ns)
                if fold_journal is not None:
                    self._close_copy_handoff_journal(
                        session, boundary="prefill_decode_fold")
                else:
                    fold_prefix_shards = (
                        kv_cache_shard_bytes_for_layer_range(
                            self.model, fold_base_history, self.tp_degree,
                            layer_start=0, layer_end=fold_base_prefix))
                    if any(fold_prefix_shards):
                        self._remove_local_shards(
                            target, fold_prefix_shards)
                        self._metrics_suffix_evict_parts(
                            session_id,
                            0,
                            anchor_kind=_metrics_anchor_for_phase(
                                "prefill_decode"),
                            request_id=trigger_request_id,
                            cause="boundary_copy_base_fold_release",
                            instance_index=target,
                        )
                session.working_kind = None
                session.base_history_tokens = 0
                session.base_instance_index = None
                session.base_resident_prefix_layers = 0
                session.base_location = ""
                self._check_invariants_after_mutation(
                    session_ids=(session_id,))
                outcome.update(
                    kind="complete_copy_base_fold",
                    moved_bytes=transfer.total_bytes)
                return transfer, evictions, outcome
            transfer, evictions = self.move_prefill_to_decode(
                session_id=session_id,
                target_instance_index=target,
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
                at_ns=at_ns)
            outcome["kind"] = (
                "stay_local_hit" if transfer.kind == "local_hit"
                else "complete_copy_migration")
            outcome["moved_bytes"] = (
                0 if transfer.kind == "local_hit" else transfer.total_bytes)
            return transfer, evictions, outcome
        if session.boundary_source_instance_index is not None:
            raise RuntimeError(
                f"session {session_id} already carries a boundary source "
                "record; a second boundary transition without an "
                "intervening settlement is a contract violation")
        # 跨实例 P 动作遗留的 base_*：描述 home 侧权威基础（copy 基础
        # 驻留 / recompute 来源侧转记录时消费）。
        base_prefix = session.base_resident_prefix_layers
        base_instance = session.base_instance_index
        base_history_tokens = session.base_history_tokens
        base_location = session.base_location or self.REMOTE_MEMORY
        full_shards = kv_cache_shard_bytes_for_tokens(
            self.model, max(0, int(history_tokens)), self.tp_degree)
        old_working = tuple(int(v) for v in session.shard_bytes)
        if action == "copy":
            if target == base_instance:
                # 回 home 仅补 U：工作副本（S+U）i_P→home，与 home 侧
                # 基础前缀合成完整 H_B——就地折叠为本地全上下文（与
                # P 侧 copy@home 退化同构，working_kind 清空）。
                fold_journal = session.copy_handoff
                fold_base_prefix = session.base_resident_prefix_layers
                fold_base_history = session.base_history_tokens
                fold_working_ctx = session.context_tokens
                fold_working_layers = session.resident_prefix_layers
                transfer, evictions = self.move_prefill_to_decode(
                    session_id=session_id,
                    target_instance_index=target,
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                    at_ns=at_ns)
                # R6（修复轮三，2026-09-27）：折叠清账的对称物理释放，
                # 与 complete_copy_base_fold 同族同科目——无 journal
                # 且工作副本 token 域覆盖基础域
                # （[0,ctx) ⊇ [0,base_history)、驻留层 ⊇ 基础驻留层）
                # 时，base 前缀是重复副本，就地零载荷释放；不释放则
                # base_* 复位后账本=full_shards 对物理=base＋full 断裂
                # （实测 state=53248 对 expected=28672）。增量形态
                # （工作副本仅持 U，ctx<base_history）base 字节是全史
                # 的组成部分，不得释放（base＋U 直和恰为 full_shards）。
                # §15.1 重复副本不增加 V：不传输、不并入、不计胜者容量。
                if fold_journal is not None:
                    self._close_copy_handoff_journal(
                        session, boundary="prefill_decode_fold")
                elif (int(fold_working_ctx) >= int(fold_base_history)
                        and int(fold_working_layers or 0) >= int(
                            fold_base_prefix)):
                    fold_prefix_shards = (
                        kv_cache_shard_bytes_for_layer_range(
                            self.model, fold_base_history, self.tp_degree,
                            layer_start=0, layer_end=fold_base_prefix))
                    if any(fold_prefix_shards):
                        self._remove_local_shards(
                            target, fold_prefix_shards)
                        self._metrics_suffix_evict_parts(
                            session_id,
                            0,
                            anchor_kind=_metrics_anchor_for_phase(
                                "prefill_decode"),
                            request_id=trigger_request_id,
                            cause="boundary_copy_home_u_fold_release",
                            instance_index=target,
                        )
                session.context_tokens = max(0, int(history_tokens))
                session.shard_bytes = full_shards
                session.total_bytes = sum(full_shards)
                session.resident_prefix_layers = self.model.layers
                session.working_kind = None
                session.base_history_tokens = 0
                session.base_instance_index = None
                session.base_resident_prefix_layers = 0
                session.base_location = ""
                self._check_invariants_after_mutation(
                    session_ids=(session_id,))
                outcome.update(kind="increment_copy_home_u_only",
                               moved_bytes=transfer.total_bytes)
                return transfer, evictions, outcome
            # @i_P（基材料化）或 @第三目标（补 H+U）：容量先行（工作副
            # 本 + 基础前缀一次性 T+E——单失败点先于任何账本变更，§16.4
            # 回退半边零字节移动的守卫口径），再迁移工作副本（S+U）到
            # j、复制基础前缀 home→j（home 保留重复副本，轮末 merge 零
            # 字节翻转释放）。
            # (2026-10-01 审查修复) 逐出元组先绑定 ()：下方 any(prefix_
            # shards) 为假（base_history==0 退化）时 evictions 未绑定，
            # 尾部 evictions + move_evictions = NameError。
            evictions: tuple[KVTransfer, ...] = ()
            prefix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model, base_history_tokens, self.tp_degree,
                layer_start=0, layer_end=base_prefix)
            old_working_now = tuple(int(v) for v in session.shard_bytes)
            if any(prefix_shards):
                evictions = self._ensure_capacity(
                    target,
                    tuple(working + prefix for working, prefix in zip(
                        old_working_now, prefix_shards)),
                    phase="decode",
                    reason="boundary_base_working_copy",
                    trigger_request_id=trigger_request_id,
                    reservation_request_id=reservation_request_id,
                    at_ns=at_ns)
            transfer, move_evictions = self.move_prefill_to_decode(
                session_id=session_id,
                target_instance_index=target,
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
                at_ns=at_ns)
            evictions = evictions + move_evictions
            base_leg: Optional[KVTransfer] = None
            if any(prefix_shards):
                base_leg = self._noc_transfer(
                    phase="prefill_decode",
                    reason="boundary_base_working_copy",
                    session=_BasePrefixView(
                        session_id=session_id,
                        instance_index=int(base_instance),
                        context_tokens=base_history_tokens,
                        resident_prefix_layers=base_prefix,
                    ),
                    trigger_request_id=trigger_request_id,
                    source_instance_index=int(base_instance),
                    target_instance_index=target,
                    layer_start=0,
                    layer_end=base_prefix,
                )
                self._add_local_shards(target, prefix_shards)
                session.shard_bytes = tuple(
                    working + prefix for working, prefix in zip(
                        session.shard_bytes, prefix_shards))
                session.total_bytes = sum(session.shard_bytes)
            session.context_tokens = max(0, int(history_tokens))
            session.resident_prefix_layers = self.model.layers
            session.working_kind = "copy"
            self._check_invariants_after_mutation(
                session_ids=(session_id,))
            primary: KVTransfer | tuple[KVTransfer, ...] = (
                (transfer, base_leg) if base_leg is not None else transfer)
            at_exec = target == int(p_instance_index)
            outcome.update(
                kind=("increment_copy_base_materialize" if at_exec
                      else "increment_copy_third_h_plus_u"),
                sources=((int(base_instance),)
                         if at_exec
                         else (int(base_instance), int(p_instance_index))),
                moved_bytes=(
                    transfer.total_bytes
                    + (base_leg.total_bytes if base_leg else 0)))
            return primary, evictions, outcome
        # ---- recompute：工作副本不迁移（来源侧转记录）。----
        if target == int(p_instance_index):
            # 原地：recompute 重建覆盖本实例（驻留即 H_B，无新字节、
            # 无事务——本地命中）。
            transfer = self._local_hit_transfer(
                phase="prefill_decode",
                reason="boundary_recompute_in_place",
                session=session,
                trigger_request_id=trigger_request_id)
            outcome.update(
                kind="recompute_in_place",
                moved_bytes=0)
            return transfer, (), outcome
        # j≠i_P：旧工作副本转来源侧记录（base_* 无片上基础保留量时记
        # base（完整型 P 本地 / 池基）；否则记 boundary_source——增量型
        # 的 U、完整型 P 跨实例残留 base 时的 i_P 旧副本。占用判据 =
        # 物理保留（base_instance + 驻留前缀 + LOCAL/PARTIAL 位）。
        # R9（修复轮三，2026-09-27）：占用判据的元数据代理对 C13 P=copy
        # 交接轮不成立——交接 journal 已在 prefill drain 边界（joiner 列
        # 车结算，四步协议 (3)）把 base 侧物理留存逐块释放（此时账本必
        # 已全结算：D 边界严序后于 drain；未结算 = 因果序破损，close 处
        # fail-closed）。base 域权威自此 = i_P 侧交接产物（旧工作副本持
        # 完整 H_B）：本处先闭合账本，再按 pool-only 基同科目（上方
        # record_to_base 分支）把 base_* 重录到 i_P 侧——轮末 merge 退化
        # 为常规两位置（base@i_P 对 exec 增量），不再虚构"base@已释放
        # 实例 + source@i_P"的三位置直和形态。三位置守卫语义不变：
        # boundary copy 腿随加入列车结算，journal 不得残留到轮末 merge。
        if session.copy_handoff is not None:
            self._close_copy_handoff_journal(
                session, boundary="boundary_source_record")
            record_to_base = True
        else:
            record_to_base = not (
                base_instance is not None
                and base_prefix > 0
                and session.base_location in {
                    self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE})
        old_location = session.location
        if record_to_base:
            session.base_history_tokens = max(0, int(history_tokens))
            session.base_instance_index = int(p_instance_index)
            session.base_resident_prefix_layers = self.model.layers
            session.base_location = old_location
            sources = (int(p_instance_index),)
        else:
            session.boundary_source_history_tokens = (
                max(0, int(history_tokens) - base_history_tokens)
                if int(history_tokens) >= base_history_tokens else 0)
            session.boundary_source_shard_bytes = old_working
            session.boundary_source_instance_index = int(p_instance_index)
            session.boundary_source_resident_prefix_layers = (
                self.model.layers)
            session.boundary_source_location = old_location
            # P=copy/recompute 的旧副本已经物化完整 H+U。完整并集标志
            # 同时驱动目录 token 矩形与三位置结算，防止把完整字节向量
            # 错误声明成部分区间。
            session.boundary_source_complete_union = True
            sources = (int(base_instance), int(p_instance_index))
        # j 侧新工作副本零化起步（rebuild/decode 增长物化）。recompute
        # 重建覆盖全域。
        zero_shards = tuple(0 for _ in old_working)
        session.shard_bytes = zero_shards
        session.total_bytes = 0
        session.context_tokens = 0
        session.instance_index = target
        session.location = self.LOCAL_HBM
        session.resident_prefix_layers = self.model.layers
        session.working_kind = action
        self._check_invariants_after_mutation(session_ids=(session_id,))
        outcome.update(
            kind="recompute_rebuild_elsewhere",
            sources=sources,
            moved_bytes=0)
        transfer = self._local_hit_transfer(
            phase="prefill_decode",
            reason="boundary_rebuild_elsewhere",
            session=session,
            trigger_request_id=trigger_request_id)
        return transfer, (), outcome

    # ------------------------------------------------ joint 合并事务（§2.2/§3.2） --

    def _working_copy_noc_transfer(
        self,
        *,
        session: SessionKVState,
        trigger_request_id: str,
        source_instance_index: int,
        target_instance_index: int,
    ) -> KVTransfer:
        """merge 前向腿：整份工作副本 NoC 迁移（exec → home 胜者）。

        逐 rank 字节**直接取 ``session.shard_bytes``**（账本真值；D1 裁决，
        2026-09-17——工作副本字节可能含池恢复后缀，从 context 派生会
        丢字节）。层区间 [0, L)；
        ``resident_prefix_layers_before/after`` 沿本仓区间传输惯例 =
        layer_start/layer_end（同例：``_noc_transfer``/
        ``_remote_load_transfer``）。
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
        at_ns: Optional[int] = None,
    ) -> int:
        """C15：后缀逐组恢复的结算边界（科目 rid#restore 的 complete）。

        与 copy 交接同一边界白名单（``prefill_drain`` / ``merge``）：
        prefill drain = 全部 prefill 列车核销后，逐组就绪门控的消费保
        证已兑现（迟到组的消费等待已按层段如实入图依赖）；merge = 兜
        底。结算 = 全部组置 CONSUMED + 守恒断言；完成时未结算 = 账本
        破损 fail-closed（mark_complete 处检查）。

        M4（2026-09-26）：``at_ns`` = 结算时刻（可选；缺省 None 保确
        定性测试）透传 complete 事件——与 issue 事件配对得恢复窗口下
        界。SH 在线调用面经 expand/merge/mark_complete 传 tick。
        """
        if boundary not in ("prefill_drain", "merge", "mark_complete"):
            raise RuntimeError(
                f"unknown restore group settle boundary {boundary!r}")
        session = self._sessions[session_id]
        journal = session.restore_journal
        if journal is None:
            return 0
        journal.settle(boundary, at_ns=at_ns)
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
        # 先完成预约守卫，再将块状态切换为 HANDED_OFF。这样容量冲突
        # 不会留下“状态已交接但源字节仍在账本中”的半事务。
        for reservation in self._reservations.values():
            if (reservation.session_id == session.session_id
                    and reservation.instance_index == journal.home_instance):
                raise RuntimeError(
                    f"copy handoff released home-side bytes of session "
                    f"{session.session_id} while a reservation still pins "
                    "them at the home instance (allocation-relation "
                    "violation)")
        chunk.state = "HANDED_OFF"
        try:
            # (2)+(3) 目标侧权威化与 home 侧释放在同一事件中落账。
            self._release_copy_handoff_chunk(
                session, chunk_index,
                boundary=boundary, trigger_request_id=trigger_request_id)
        except Exception:
            # _remove_local_shards validates every rank before mutating, so
            # restoring the state is sufficient to keep a failed event
            # retryable.
            chunk.state = "PENDING"
            raise
        # 目标侧权威化：块位置/所有权转移到执行端，后续读取路由到目标
        # 有效副本（执行端物理字节已在准入相物化——保守容量口径）。
        chunk.state = "HANDED_OFF"
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
        if chunk_index in journal.released_indices:
            raise RuntimeError(
                f"double release of copy handoff chunk {chunk_index} for "
                f"session {session.session_id} (request {trigger_request_id})"
                " -- the home-side block was already released")
        if chunk.state != "HANDED_OFF":
            raise RuntimeError(
                f"copy handoff release for chunk {chunk_index} of session "
                f"{session.session_id} before its handoff completed -- "
                "releasing the home-side copy before the target-side "
                "authority transfer is an early release (four-step "
                "protocol violation)")
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
        self._remove_local_shards(journal.home_instance, chunk.shard_bytes)
        journal.released_indices.add(chunk_index)
        journal.h_home_shards = tuple(
            home - chunk_bytes for home, chunk_bytes in zip(
                journal.h_home_shards, chunk.shard_bytes))
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
        at_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        """compute_done 后的合并事务（少并多真值表的零翻转/in_place 两态）。

        ``new_tokens`` = 本轮新增 token 数（输入 + 实际完成 decode——已
        观测长度，允许用于实际结算）。三动作（stay/copy/recompute）下
        working_kind ∈ {copy, recompute}，结果恒**全层 LOCAL@胜者**、
        败者侧释放、``home_instance`` 迁移到胜者（I7；全仓第二个赋值点）：

        * copy/recompute（LOCAL/PARTIAL 基）：exec 恒持并集 ⊇ home 侧——
          **零字节翻转**（无传输，home 侧基础释放，胜者空间准备量 = 0）。
        * REMOTE 基（无主，裁定③）：**就地保留**（in_place）——零传输
          零池写，工作副本直接转正，home := exec。

        热 KV 裁定（2026-09-17）封死了"执行完再踢出去"：merge 事务**零
        池写（I6，无 remote_store）**；空间准备只走统一 T+E 逐出（I8，
        无 merge 专用驱逐）——零字节翻转/in_place 的准备量 = 0 不可能
        失败。stay 路径（无工作副本）为本地提交，不产生流量；恰好归并
        一次由 ``last_merged_request_id`` 版本键断言保证（重复写 KV/
        重复释放源副本/重复计 token 均为合同类违规）。每次调用（含
        stay 早退）结束前设置 ``self.last_merge_outcome`` 披露快照
        （改造二契约，供调度器日志/水印重放消费）。

        N9（2026-09-14，语义收窄 2026-09-17）：home==exec 的 working
        组合仍是账本破损组合（R13 已消除合法来源：recompute@home 本地
        提交、copy@home 退化 local_hit）——结算前 fail-closed；翻转/
        in_place 产生的 home==exec 是合法终态。K5（P2-5，2026-09-23）：
        pool-only 基（REMOTE 基/前缀 0）豁免 raise——REMOTE 基 ×copy@home
        是裁定③就地转正形态（基础不在 home，非 R13 消除的"基础在 home"
        组合），方向裁决走 in_place 免单。
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
        # D-BI：物理基础驻留实例——base_* 快照记录的片上所在（与逻辑
        # home 解耦；P→D 迁移 + merge stay 提交后基础可不在 home）。池基
        # （REMOTE 基/前缀 0）无片上驻留 → None 合法，方向裁决走
        # in_place；有片上基础保留量而缺实例记录 = 账本完整性破损，
        # 于 home_shards 计算后 fail-closed。
        base_side = session.base_instance_index
        # D-BI 注记：exec==逻辑 home 而基础在别处（P→D 迁移后回 home
        # copy 等合法形态）不是 R13 消除的组合——N9 守卫只钉物理基础
        # 驻留于执行端的组合，按基础所在继续结算。
        if base_side == exec_instance:
            # K5（P2-5，2026-09-23 外部审计）：pool-only 基（REMOTE 基/
            # 前缀 0）豁免——REMOTE 基 ×copy@执行端 是裁定③的就地转正
            # 形态（执行端侧无基础保留量 ⇒ 方向裁决走 in_place 免单），
            # 原 N9 对 base==exec 一刀切把该合法来源拦死（确定性
            # raise）。R13/N9 消除的是"基础在执行端 的 copy@执行端 退化
            # stay"；无主基不在其列。
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
            elif (session.boundary_source_instance_index is not None
                    and session.working_kind == "recompute"):
                # D12：D 边界 recompute@基础驻留端（j==base）——重建副本
                # 持完整并集、同实例 base 记录是重复副本，结算走三位置
                # 完整并集零载荷（base 重复副本就地释放），不是 R13 消除
                # 的"基础在执行端的退化组合"。
                pass
            else:
                # N9 防御：base==exec 的 working 组合不应存在（见
                # docstring），到达即选点/结算合同破损——显式 fail-closed
                # 而非按旧口径抹账。
                raise RuntimeError(
                    f"session {session_id} has a working copy "
                    f"({session.working_kind}) on its own base-resident "
                    "instance; R13 removed this combination "
                    "(recompute@base settles locally, copy@base "
                    "degenerates to stay)")
        # C13：copy 逐块交接的轮末兜底结算（正常路径已在 prefill drain
        # 边界结算；因果序：块到达被图侧钉在 prefill 列车体完成之前 ≤
        # drain < merge——此处只可能是直接 API 调用面未走 drain）。
        self._settle_copy_handoffs(
            session_id, boundary="merge", trigger_request_id=trigger_request_id)
        # C15：后缀逐组恢复的轮末兜底（同 C13 边界语义）。M4：at_ns 随边界透传。
        self._settle_restore_groups(
            session_id, boundary="merge", at_ns=at_ns)
        # C14：home 侧物理残量快照（结算时刻事实，_close 前捕获）——copy
        # 逐块交接已在轮内释放的字节不得计入"两侧实际保留量"（C12 §2
        # 规则 4 的 journal 口径镜像）；无 journal（recompute）取 None
        # ⇒ 纪录 base 前缀推导值（物理驻留真值）。
        _copy_journal_at_merge = session.copy_handoff
        _home_residual_bytes = (
            sum(_copy_journal_at_merge.home_shards())
            if _copy_journal_at_merge is not None else None)
        transfers: list[KVTransfer] = []

        # ---- D12/C4：三位置确定性合并（式(21) 摘要的运行时形态）。----
        # 边界第二来源（i_P 侧旧副本，recompute 不迁移）在
        # 场时，结算面 = {base_*（home 侧基础）、boundary_source（i_P 侧
        # 旧副本）、exec 工作副本}：完整并集持有者优先（recompute 的
        # 重建副本 = H_B+增长 ⊇ 其他位置——零载荷，重复副本直接释放）；
        # 否则保留量最大者为目标（平局 → 逻辑 home → D 实例 → 编号），
        # 其余位置整份搬入。结果不依赖两两合并顺序（一次性按全局序）。
        if session.boundary_source_instance_index is not None:
            return self._merge_back_three_position(
                session=session,
                trigger_request_id=trigger_request_id,
                new_tokens=new_tokens,
                reservation_request_id=reservation_request_id,
                at_ns=at_ns,
                home_residual_bytes=_home_residual_bytes)

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
            if base_side is None:
                # D-BI：有片上基础保留量而缺物理驻留实例记录 = 账本完整
                # 性破损（合法事务面均经 _snapshot_base 记录），fail-closed。
                raise RuntimeError(
                    f"session {session_id} has a resident base "
                    f"({base_location}) without a recorded base instance "
                    "at merge_back; ledger integrity failure")
        # exec 侧工作副本保留量 W（账本真值：copy 的池恢复后缀也在账内）。
        working_shards = tuple(session.shard_bytes)
        # C14：结算时刻的工作副本类别快照（终态清零 working_kind 之前
        # 捕获——kv_delta_journal 行的字段来源）。
        working_kind_at_settlement = session.working_kind
        if working_kind_at_settlement not in ("copy", "recompute"):
            raise RuntimeError(
                f"session {session_id} carries an unknown working copy kind "
                f"at merge_back ({working_kind_at_settlement!r}); ledger "
                "integrity failure")
        # copy/recompute：exec 恒持并集 ⊇ home 侧——零字节翻转（无传输；
        # 胜者空间准备量 = 0，不可能失败）；home 侧无片上基础保留量
        # （REMOTE 基/前缀 0）时 in_place（零传输零池写，工作副本直接
        # 转正，裁定③）。
        if not home_shards:
            direction = "in_place"
        else:
            direction = "reverse"
        zero_byte_flip = direction == "reverse"

        # ---- 传输发射（I6：merge 事务零池写——不产生任何 remote_store；
        # 零字节翻转/in_place 均零传输——merge 的真实搬移腿只在三位置
        # 合并分支发射）。 ----
        transferred_bytes = 0

        # ---- 账本结算：统一终态 = 全层 LOCAL@胜者，败者侧释放。 ----
        if direction == "reverse":
            # 零字节翻转：胜者 = exec，败者 = 基础驻留侧（D-BI：物理基础
            # 所在，≠ 逻辑 home 时以基础所在为准）。
            winner, loser = exec_instance, base_side
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
                # recompute（或无 journal 的 copy 形态）：败者 = 基础
                # 驻留侧，释放基础驻留前缀（D-BI：从物理所在释放）。
                self._remove_local_shards(base_side, home_shards)
                self._metrics_suffix_evict_parts(
                    session_id,
                    0,
                    anchor_kind=_metrics_anchor_for_phase("completion"),
                    request_id=trigger_request_id,
                    cause="merge_base_release",
                    instance_index=base_side,
                )
        else:
            # in_place：无释放无入账无传输（无主，裁定③——工作副本
            # 已在 exec 账上直接转正）。
            winner, loser = exec_instance, home

        # 守恒断言：胜者侧现有字节合计 == 合并后全量（线性算术应精确
        # 相等；不等 = 账本破损/调用方口径漂移，fail-closed）。零字节
        # 翻转（copy/recompute）与 in_place 胜者持并集 ⊇ 基础——核对
        # 工作副本本身。
        # C4：终态 token 基数按工作副本类别——copy/recompute（零字节
        # 翻转族）的工作副本 context 已覆盖完整并集（D 边界 recompute
        # 的 base 记录 = 旧 H_B 副本、已含本轮输入，base+new 会重复计
        # input；P 相两口径恒等：base=H、working context=H+U+增长）。
        final_tokens = int(session.context_tokens)
        full_shards = kv_cache_shard_bytes_for_tokens(
            self.model, final_tokens, self.tp_degree)
        winner_existing = working_shards
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
        #（首轮建立在 ``prepare_prefill`` 的会话建档处，§2.1）。
        session.home_instance = winner
        session.base_location = ""
        session.base_history_tokens = 0
        session.base_resident_prefix_layers = 0
        session.base_instance_index = None
        session.last_merged_request_id = trigger_request_id
        # C13：copy 交接账本轮末闭合（断言闭合 + 科目 close 事件 + 清账）。
        self._close_copy_handoff_journal(session, boundary="merge")
        self.last_merge_outcome = {
            "session_id": session_id,
            "direction": direction,
            "zero_byte_flip": zero_byte_flip,
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
            zero_byte_flip=zero_byte_flip,
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

    # ---------------------------------------- W9（§7.4）目录提交型结算 --

    def settle_distributed(
        self,
        *,
        session_id: str,
        trigger_request_id: str,
        new_tokens: int,
        at_ns: Optional[int] = None,
    ) -> DistributedSettleOutcome:
        """目录提交型结算（settlement_policy=distributed；§7.4）。

        与 ``merge_back``（consolidate 的物理 merge 子协议）对称的
        distributed 子协议——**不执行物理归并**（零 merge 载荷、零池写、
        零 rank 台账搬动）：

        1. 版本键防重（``last_settled_request_id``——一请求恰一次结算）；
        2. 闭合必要旧义务：copy 交接 journal / restore group 于
           boundary="merge" 兜底结算（未完 ack/在途访问）；
        3. 旧基础段/边界来源段移入跨轮留存通道 E（目录条目权威化——
           owner/角色保留、生成版本 +1；无三位置上限）；
        4. working_kind 清空（round 结束——工作副本转正为该区间的主驻
           留；mark_complete 不再视为未合并副本）；primary/home 不迁移
           （home 粘滞 §2.1，A13 不暗中重置 home）；
        5. 目录事务提交（``_check_invariants_after_mutation``：镜像同步
           + 目录守恒 + rank 台账对账 = 提交完整有效目录）。

        家族判定与 ``merge_back`` 同口径：copy/recompute = 零字节翻转族
        （exec 持完整并集，primary 即并集）。
        """
        session = self._sessions[session_id]
        if session.last_settled_request_id == trigger_request_id:
            raise RuntimeError(
                f"settle_distributed settled twice for request "
                f"{trigger_request_id} (session {session_id}) -- duplicate "
                "settlement transaction")
        if new_tokens < 0:
            raise ValueError("new_tokens must be non-negative")

        # ---- 2. 必要旧义务闭合（C13/C15 轮末兜底边界）。 ----
        self._settle_copy_handoffs(
            session_id, boundary="merge", trigger_request_id=trigger_request_id)
        self._settle_restore_groups(session_id, boundary="merge", at_ns=at_ns)

        working_kind_at_settlement = session.working_kind
        exec_instance = session.instance_index
        base_history_tokens = int(session.base_history_tokens)
        retained_registry = self._distributed_retained.setdefault(
            session_id, {})

        def _register_retained(
                segment: RetainedSegment) -> None:
            retained_registry[segment.key] = segment

        # ---- 3. 旧位置移入通道 E（owner 记录保留；无字节移动）。 ----
        # 基础段：copy 族的 journal 残量登记分支已删（2026-10-01 死代码
        # 审计）——头部 _settle_copy_handoffs(boundary="merge") 已把全部
        # PENDING 块置 HANDED_OFF 并释放 home 字节，此后
        # journal.home_shards() 恒零，该分支不可达（残量语义由下方
        # base/boundary_source 登记与 _close_copy_handoff_journal 承担）。
        # recompute 族基础完整保留在源端。
        journal = session.copy_handoff
        if (
                session.working_kind is not None
                and session.base_location in {
                    self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
                and session.base_resident_prefix_layers > 0
                and session.base_instance_index is not None):
            base_prefix = int(session.base_resident_prefix_layers)
            _register_retained(RetainedSegment(
                session_id=session_id,
                instance_index=int(session.base_instance_index),
                layer_start=0,
                layer_end=base_prefix,
                token_start=0,
                token_end=base_history_tokens,
                bytes_by_tp_rank=(
                    kv_cache_shard_bytes_for_layer_range(
                        self.model, base_history_tokens, self.tp_degree,
                        layer_start=0, layer_end=base_prefix)),
                role=ROLE_BASE,
                history=(f"base-retained:{trigger_request_id}",),
            ))
        # 边界来源段（i_P 侧旧副本——D19/C4 第三位置；consolidate 在三
        # 位置合并释放，distributed 保留为跨轮来源）。
        if (
                session.boundary_source_instance_index is not None
                and session.boundary_source_location in {
                    self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}
                and any(session.boundary_source_shard_bytes)):
            boundary_tokens = int(session.boundary_source_history_tokens)
            if session.boundary_source_complete_union:
                boundary_token_start = 0
                boundary_token_end = base_history_tokens + boundary_tokens
            else:
                boundary_token_start = base_history_tokens
                boundary_token_end = base_history_tokens + boundary_tokens
            _register_retained(RetainedSegment(
                session_id=session_id,
                instance_index=int(session.boundary_source_instance_index),
                layer_start=0,
                layer_end=int(
                    session.boundary_source_resident_prefix_layers
                    or self.model.layers),
                token_start=boundary_token_start,
                token_end=boundary_token_end,
                bytes_by_tp_rank=tuple(
                    int(v) for v in session.boundary_source_shard_bytes),
                role=ROLE_BOUNDARY_SOURCE,
                history=(f"boundary-source-retained:{trigger_request_id}",),
            ))

        # ---- 4. 会话状态收敛（无字节搬动；home 不迁移）。 ----
        if working_kind_at_settlement is None:
            # stay 族本地提交：primary 已是本地全量（无工作副本事务）。
            final_tokens = int(session.context_tokens)
            family = "stay_local"
        elif working_kind_at_settlement in ("copy", "recompute"):
            # 零字节翻转族：exec 持完整并集（口径 = merge_back final_tokens
            # 同式：copy/recompute 的 working context 已覆盖完整并集）。
            final_tokens = int(session.context_tokens)
            full_shards = kv_cache_shard_bytes_for_tokens(
                self.model, final_tokens, self.tp_degree)
            if tuple(session.shard_bytes) != tuple(full_shards):
                raise RuntimeError(
                    f"distributed settlement conservation failed for "
                    f"session {session_id}: zero-byte-flip family exec-side "
                    f"bytes {session.shard_bytes} != full union shards "
                    f"{full_shards} (ledger integrity failure)")
            family = "zero_byte_flip"
        else:
            raise RuntimeError(
                f"distributed settlement for {session_id} saw an unknown "
                f"working copy kind {working_kind_at_settlement!r}; ledger "
                "integrity failure")

        if journal is not None:
            if any(journal.home_shards()):
                # (2026-10-01 R3 审查修复) 本分支不可达：头部
                # _settle_copy_handoffs(boundary="merge") 已把全部 PENDING
                # 块置 HANDED_OFF 并释放 home 字节。旧行为静默丢弃带残量
                # 的 journal（字节不释放不留段——fail-open）；改为
                # fail-closed 点名（与 _close_copy_handoff_journal 的零
                # 残量断言同姿态）。
                raise RuntimeError(
                    f"copy handoff journal for session {session_id} still "
                    f"carries home-side residual shards at distributed "
                    f"settle (head-of-settle handoff closure invariant "
                    f"broken)")
            else:
                self._close_copy_handoff_journal(session, boundary="merge")
        session.working_kind = None
        # base_*/boundary_source_* 清空：描述职责移交通道 E（目录条目）。
        session.base_location = ""
        session.base_history_tokens = 0
        session.base_resident_prefix_layers = 0
        session.base_instance_index = None
        session.boundary_source_history_tokens = 0
        session.boundary_source_shard_bytes = ()
        session.boundary_source_instance_index = None
        session.boundary_source_resident_prefix_layers = 0
        session.boundary_source_location = ""
        session.boundary_source_complete_union = False
        session.distributed_layout = True
        session.last_settled_request_id = trigger_request_id

        # ---- 5. 目录事务提交 + 披露（§7.4：settle 事件仍产生——零载荷
        # 形态由 W5 状态机闭合；来源面/区间数/位置数分别披露）。 ----
        self._check_invariants_after_mutation(session_ids=(session_id,))
        surface = enumerate_source_surface(self._kv_directory, session_id)
        outcome = DistributedSettleOutcome(
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            family=family,
            final_tokens=final_tokens,
            merge_payload_bytes=0,
            same_round_merge_saved_bytes=sum(
                segment.total_bytes()
                for segment in retained_registry.values()
                if segment.is_on_chip()),
            source_count=surface.source_count,
            interval_count=surface.interval_count,
            retained_keys=tuple(sorted(retained_registry)),
            directory_version=self._kv_directory.version,
            commit_ns=int(at_ns) if at_ns is not None else 0,
        )
        self.last_distributed_settle_outcome = outcome
        self.distributed_directory_meter.record_directory_commit(
            self._kv_directory, session_id, outcome)
        self._append_kv_delta_row(
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            working_kind=working_kind_at_settlement,
            direction="distributed",
            zero_byte_flip=False,
            winner_instance=None,
            loser_instance=None,
            home_before=session.home_instance,
            transferred_bytes=0,
            home_side_retained_bytes=outcome.same_round_merge_saved_bytes,
            exec_side_retained_bytes=sum(
                int(v) for v in session.shard_bytes),
            new_tokens=new_tokens,
        )
        return outcome

    def _distributed_resident_layer_bytes(
        self, session: SessionKVState, layer_start: int, layer_end: int,
    ) -> tuple[int, ...]:
        """主位置逐层段字节（统一公式入口；调用面 = _evict_suffix/
        _evict_session/_summary 字节/retire 等）。"""
        return kv_cache_shard_bytes_for_layer_range(
            self.model, session.context_tokens, self.tp_degree,
            layer_start=layer_start, layer_end=layer_end)

    def _completed_retained_segments(
        self,
        instance_index: int,
        trigger_type: Optional[str] = None,
    ) -> list[tuple[SessionKVState, RetainedSegment]]:
        """W9 区间级 E 的留存段 victim 集（目录 owner 过滤）。

        谓词：段片上 READY、holder==instance（**目录 owner 过滤**——
        取代"会话单实例"假设）、会话已完成/非 active/无阶段引用保护；
        T 类别过滤与主位置 victim 同源。FIFO 序继承会话完成时刻。"""
        candidates: list[tuple[SessionKVState, RetainedSegment]] = []
        for session_id, segments in self._distributed_retained.items():
            session = self._sessions.get(session_id)
            if session is None or self._is_retained_victim_blocked(session):
                continue
            if trigger_type is not None and (
                    self._eviction_class(session) != trigger_type):
                continue
            for segment in segments.values():
                if (segment.is_on_chip()
                        and segment.instance_index == instance_index
                        and any(segment.bytes_by_tp_rank)):
                    candidates.append((session, segment))
        candidates.sort(key=lambda item: (
            item[0].last_completion_ns or 0,
            item[1].key))
        return candidates

    def _evict_retained_segment(
        self,
        session: SessionKVState,
        segment: RetainedSegment,
        *,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> KVTransfer:
        """留存段驱逐（§7.4：经目录 ``evict_to_pool``——owner 记录保留、
        唯一内容在池；不清 owner）。物理腿 = remote_store 池写（与主位置
        逐出同协议）；rank 台账同步释放。"""
        if not segment.is_on_chip():
            raise RuntimeError(
                f"retained segment {segment.key} for session "
                f"{session.session_id} is not on-chip (state={segment.state})")
        registry = self._distributed_retained[session.session_id]
        directory_entry = None
        for entry in self._kv_directory.entries_for_session(
                session.session_id):
            key = entry.content_key
            if (entry.holder == segment.instance_index
                    and key.layer_range == (
                        segment.layer_start, segment.layer_end)
                    and key.token_range == (
                        segment.token_start, segment.token_end)):
                directory_entry = entry
                break
        if directory_entry is None:
            raise RuntimeError(
                f"retained segment {segment.key} for session "
                f"{session.session_id} has no matching directory entry")
        if directory_entry.state != ENTRY_STATE_READY:
            raise RuntimeError(
                f"retained segment {segment.key} for session "
                f"{session.session_id} is not READY in the directory "
                f"(state={directory_entry.state})")
        if directory_entry.pending_acks:
            raise RuntimeError(
                f"retained segment {segment.key} for session "
                f"{session.session_id} has pending directory obligations "
                f"{directory_entry.pending_acks}")
        if self._kv_directory.read_refs(directory_entry.entry_id) > 0:
            raise RuntimeError(
                f"retained segment {segment.key} for session "
                f"{session.session_id} is protected by a live directory "
                "read reference")
        if self._kv_directory.is_exec_pinned(directory_entry.entry_id):
            raise RuntimeError(
                f"retained segment {segment.key} for session "
                f"{session.session_id} is protected by an execution pin")
        # 物理腿：逐层段池写（与 _evict_suffix 同源构造——留存段字节向
        # 量直取真值，remote_store 以层区间为原子）。
        transfer = self._retained_segment_store_transfer(
            session, segment, phase=phase, reason=reason,
            trigger_request_id=trigger_request_id)
        self._remove_local_shards(
            segment.instance_index, segment.bytes_by_tp_rank)
        # 目录侧：真实 evict_to_pool（owner 记录保留——role→pool-backed、
        # 原角色/holder 入审计链；引用>0/pin fail-closed 在目录侧裁决）。
        self._kv_directory.evict_to_pool(directory_entry.entry_id)
        registry[segment.key] = RetainedSegment(
            session_id=segment.session_id,
            instance_index=segment.instance_index,
            layer_start=segment.layer_start,
            layer_end=segment.layer_end,
            token_start=segment.token_start,
            token_end=segment.token_end,
            bytes_by_tp_rank=segment.bytes_by_tp_rank,
            role=segment.role,
            state=ENTRY_STATE_EVICTED_TO_POOL,
            generation=segment.generation + 1,
            history=segment.history + (
                f"evicted-to-pool@{segment.instance_index}:{reason}",),
        )
        self.distributed_directory_meter.record_op("evict_to_pool")
        # (2026-10-01 审查修复) metrics 镜像按**段区间**截除（同
        # _metrics_suffix_evict_parts 的实例/区间过滤形态），不再整会话
        # 删光——distributed 会话主副本/其他段仍片上，整删使观测侧驻留
        # 字节提前消失、后续段释放无记录。
        self._metrics_remove_segment_parts(
            session.session_id,
            instance_index=int(segment.instance_index),
            layer_start=int(segment.layer_start),
            layer_end=int(segment.layer_end),
            token_start=int(segment.token_start),
            token_end=int(segment.token_end),
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=(f"evict_{reason}_retained_segment:layers"
                   f"{segment.layer_start}-{segment.layer_end}"),
        )
        self._check_invariants_after_mutation(
            session_ids=(session.session_id,))
        return transfer

    def _retained_segment_store_transfer(
        self,
        session: SessionKVState,
        segment: RetainedSegment,
        *,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> KVTransfer:
        """留存段的 remote_store 物理腿（逐 rank 池写；层区间原子）。"""
        instance = self.topology.instance(segment.instance_index)
        shards = tuple(
            KVTransferShard(
                source_rank=rank,
                target_rank=None,
                edge_rank=self.nearest_edge(rank),
                bytes=bytes_value,
                noc_path=deterministic_xy_route(
                    self.topology.hardware, rank, self.nearest_edge(rank)),
                layer_start=segment.layer_start,
                layer_end=segment.layer_end,
            )
            for rank, bytes_value in zip(
                instance.ranks, segment.bytes_by_tp_rank)
            if bytes_value > 0
        )
        return KVTransfer(
            kind="remote_store",
            phase=phase,
            reason=f"{reason}_retained_segment",
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=segment.instance_index,
            target_instance_index=None,
            total_bytes=sum(segment.bytes_by_tp_rank),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=segment.layer_start,
            layer_end=segment.layer_end,
            resident_prefix_layers_before=session.resident_prefix_layers,
            resident_prefix_layers_after=session.resident_prefix_layers,
        )

    def restore_retained_segment(
        self,
        session_id: str,
        segment_key: tuple[int, int, int, int, int],
        *,
        target_instance: int,
        trigger_request_id: str,
        at_ns: Optional[int] = None,
    ) -> KVTransfer:
        """留存段池恢复（§7.4：目录 ``restore_from_pool`` → RESTORING →
        物理池读腿 → 就绪回 READY；RestoreGroupJournal 语义由目录状态
        承载——跨轮留存段不挂 session.restore_journal）。

        恢复完成（调用 ``complete_retained_restore``）后段回片上 READY
        （字节入目标实例台账）。"""
        registry = self._distributed_retained.get(session_id, {})
        segment = registry.get(segment_key)
        if segment is None or segment.state != ENTRY_STATE_EVICTED_TO_POOL:
            raise RuntimeError(
                f"retained segment {segment_key} for session {session_id} "
                f"is not pool-backed (state={getattr(segment, 'state', None)})")
        directory_entry = None
        for entry in self._kv_directory.query_pool_intervals(session_id):
            key = entry.content_key
            if (key.layer_range == (segment.layer_start, segment.layer_end)
                    and key.token_range == (
                        segment.token_start, segment.token_end)):
                directory_entry = entry
                break
        if directory_entry is not None:
            self._kv_directory.restore_from_pool(
                directory_entry.entry_id, target_instance=target_instance)
        registry[segment_key] = RetainedSegment(
            session_id=segment.session_id,
            instance_index=segment.instance_index,
            layer_start=segment.layer_start,
            layer_end=segment.layer_end,
            token_start=segment.token_start,
            token_end=segment.token_end,
            bytes_by_tp_rank=segment.bytes_by_tp_rank,
            role=segment.role,
            state=ENTRY_STATE_RESTORING,
            generation=segment.generation + 1,
            restoring_to=int(target_instance),
            history=segment.history + (
                f"restoring->{target_instance}:{trigger_request_id}",),
        )
        self.distributed_directory_meter.record_op("restore_from_pool")
        # 物理池读腿（与 _remote_load_transfer 同形；字节在完成事务入账）。
        transfer = self._retained_segment_load_transfer(
            session_id, registry[segment_key], trigger_request_id)
        # RESTORING is a directory in-flight state.  Its target holder is
        # visible to directory readers before the physical HBM add, so the
        # rank mirror is checked at complete_retained_restore, after the
        # matching arrival event, rather than at issue.
        return transfer

    def _retained_segment_load_transfer(
        self, session_id: str, segment: RetainedSegment,
        trigger_request_id: str,
    ) -> KVTransfer:
        target = self.topology.instance(int(segment.restoring_to or 0))
        shards = tuple(
            KVTransferShard(
                source_rank=None,
                target_rank=rank,
                edge_rank=self.nearest_edge(rank),
                bytes=bytes_value,
                noc_path=deterministic_xy_route(
                    self.topology.hardware, self.nearest_edge(rank), rank),
                layer_start=segment.layer_start,
                layer_end=segment.layer_end,
            )
            for rank, bytes_value in zip(
                target.ranks, segment.bytes_by_tp_rank)
            if bytes_value > 0
        )
        return KVTransfer(
            kind="remote_load",
            phase="history",
            reason="retained_segment_pool_restore",
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=None,
            target_instance_index=int(segment.restoring_to or 0),
            total_bytes=sum(segment.bytes_by_tp_rank),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=segment.layer_start,
            layer_end=segment.layer_end,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=segment.layer_end,
        )

    def complete_retained_restore(
        self,
        session_id: str,
        segment_key: tuple[int, int, int, int, int],
    ) -> None:
        """留存段恢复完成事务（RESTORING → READY@target；字节入账）。"""
        registry = self._distributed_retained.get(session_id, {})
        segment = registry.get(segment_key)
        if segment is None or segment.state != ENTRY_STATE_RESTORING:
            raise RuntimeError(
                f"retained segment {segment_key} for session {session_id} "
                "is not RESTORING (nothing to complete)")
        target_instance = int(segment.restoring_to or 0)
        target_key = (
            target_instance, segment.layer_start, segment.layer_end,
            segment.token_start, segment.token_end)
        existing = registry.get(target_key)
        if existing is not None:
            raise RuntimeError(
                f"retained segment restore for session {session_id} "
                f"collides with an in-registry entry at the target key "
                f"{target_key} (existing state {existing.state})")
        directory_matches = tuple(
            entry for entry in self._kv_directory.entries_for_session(
                session_id, valid_only=False)
            if (entry.holder == target_instance
                and entry.content_key.layer_range == (
                    segment.layer_start, segment.layer_end)
                and entry.content_key.token_range == (
                    segment.token_start, segment.token_end))
        )
        if len(directory_matches) != 1:
            raise RuntimeError(
                f"retained segment restore for session {session_id} "
                f"has {len(directory_matches)} matching directory entries "
                f"at target {target_key}")
        directory_entry = directory_matches[0]
        if directory_entry.state != ENTRY_STATE_RESTORING:
            raise RuntimeError(
                f"retained segment restore for session {session_id} "
                f"is not RESTORING in the directory "
                f"(state={directory_entry.state})")
        self._add_local_shards(
            target_instance, segment.bytes_by_tp_rank)
        # (2026-10-01 审查修复) 恢复换实例时先 pop 旧键、再按新实例键
        # 写入：RetainedSegment.key 含 instance_index——旧实现把换实例
        # 的段写回**旧键**，registry 里 {旧键: 值(.key=新键)}，后续逐出
        # （registry[segment.key] = EVICTED 在新键下插入新条目、旧条目
        # 残留同字节）与锚检出（segments.pop(anchor.key) KeyError）全
        # 部错乱，_incremental_retained_contributions 双计撞守恒断言。
        restored = RetainedSegment(
            session_id=segment.session_id,
            instance_index=target_instance,
            layer_start=segment.layer_start,
            layer_end=segment.layer_end,
            token_start=segment.token_start,
            token_end=segment.token_end,
            bytes_by_tp_rank=segment.bytes_by_tp_rank,
            role=segment.role,
            state=ENTRY_STATE_READY,
            generation=segment.generation + 1,
            history=segment.history + (
                f"restored@{target_instance}",),
        )
        registry.pop(segment_key, None)
        registry[restored.key] = restored
        self.distributed_directory_meter.record_op("restore_complete")
        self._check_invariants_after_mutation(session_ids=(session_id,))

    def _shard_vector_noc_transfer(
        self,
        *,
        session_id: str,
        trigger_request_id: str,
        source_instance_index: int,
        target_instance_index: int,
        shard_bytes: tuple[int, ...],
        reason: str,
    ) -> KVTransfer:
        """D12/C4：任意逐 rank 字节向量的 NoC 迁移腿（三位置合并用——
        与 ``_working_copy_noc_transfer`` 同形，字节直取账本真值向量）。"""
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=bytes_value,
                noc_path=deterministic_xy_route(
                    self.topology.hardware, source_rank, target_rank),
                layer_start=0,
                layer_end=self.model.layers,
            )
            for source_rank, target_rank, bytes_value in zip(
                source.ranks, target.ranks, shard_bytes)
            if bytes_value > 0
        )
        return KVTransfer(
            kind="noc_migrate",
            phase="completion",
            reason=reason,
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=target_instance_index,
            total_bytes=sum(int(value) for value in shard_bytes),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=self.model.layers,
            resident_prefix_layers_before=0,
            resident_prefix_layers_after=self.model.layers,
        )

    def _merge_back_three_position(
        self,
        *,
        session: SessionKVState,
        trigger_request_id: str,
        new_tokens: int,
        reservation_request_id: Optional[str],
        at_ns: Optional[int],
        home_residual_bytes: Optional[int],
    ) -> tuple[KVTransfer, ...]:
        """D12/C4 三位置确定性合并（``merge_back`` 的多源/重建终态分
        支；式(21)：完整并集优先 → 保留量最大 → 逻辑 home → D 实例 →
        编号；容量回退按该确定性序最多逐位重试——回退候选同样是完整
        目标（其余全部持有位置并入），双侧耗尽 = 深缺口 fail-closed）。
        结果恒全层 LOCAL@胜者、零池写（I6 不变）、home 迁移到胜者（I7）。
        recompute@基础驻留端的 base==exec 重合持有者是合法形态（重建副
        本持并集、同实例 base 记录是重复副本，零载荷就地释放）。
        R5（修复轮三）：来源侧完整并集（boundary_source_complete_union
        ——i_P 旧副本 = P=copy/recompute 完整副本，token 域 ⊇ base）同式处
        置 base 重复副本：零载荷释放、不入 winner 侧（保留量最大者 =
        重建来源直接为胜者，仅缺 exec 增长域）。"""
        session_id = session.session_id
        if session.copy_handoff is not None:
            # 三位置形态不经 C13 逐块交接（边界 copy 腿随加入列车整段
            # 发射）；残留 journal = 形态混用，fail-closed。
            raise RuntimeError(
                f"three-position merge for session {session_id} cannot "
                "carry a copy handoff journal (boundary copy legs settle "
                "with the joiner train)")
        self._settle_restore_groups(
            session_id, boundary="merge", at_ns=at_ns)
        exec_instance = session.instance_index
        if exec_instance is None:
            raise RuntimeError("working copy lost its execution instance")
        home = session.home_instance
        base_prefix = session.base_resident_prefix_layers
        base_history_tokens = session.base_history_tokens
        base_location = session.base_location
        if base_location == "":
            raise RuntimeError(
                f"session {session_id} has empty-string base_location at "
                "merge_back -- session metadata corrupt")
        if base_location is None:
            base_location = self.REMOTE_MEMORY
        if base_location == self.REMOTE_MEMORY or base_prefix <= 0:
            base_shards: tuple[int, ...] = ()
            base_side = session.base_instance_index
        else:
            base_side = session.base_instance_index
            if base_side is None:
                raise RuntimeError(
                    f"session {session_id} has a resident base without a "
                    "recorded base instance at merge_back; ledger "
                    "integrity failure")
            base_shards = kv_cache_shard_bytes_for_layer_range(
                self.model, base_history_tokens, self.tp_degree,
                layer_start=0, layer_end=base_prefix)
        source_side = session.boundary_source_instance_index
        source_shards = tuple(
            int(value) for value in session.boundary_source_shard_bytes)
        working_shards = tuple(int(v) for v in session.shard_bytes)
        complete_union = session.working_kind == "recompute"
        # R5（修复轮三，2026-09-27）：来源侧完整并集——i_P 侧旧副本是
        # P=copy/recompute 的完整副本（boundary_source_complete_union 登记，
        # 设定点见 commit_boundary_decode_transition），物理 token 域
        # ⊇ base_* 基础：base 位置是重复副本，按零载荷释放（对称
        # complete_union 的重复副本处置，§15.1 去重），不并入 winner
        # 侧、不计入容量需求（胜者已持并集，仅缺 exec 增长域）。probe
        # 实测（session_3 seq 173）：base@4(11837 token) 被 source@6
        # (18271 token = 11837＋6434) 覆盖，直和结算把基础计两遍
        # （30109 ≠ 18272 token）撞守恒 fail-closed。
        source_union = (
            not complete_union
            and session.boundary_source_complete_union
            and base_side is not None
            and any(base_shards)
            and source_side is not None
            and any(source_shards))
        positions: list[tuple[int, tuple[int, ...]]] = []
        duplicate_positions: list[tuple[int, tuple[int, ...]]] = []
        if base_side is not None and any(base_shards):
            if source_union:
                duplicate_positions.append((int(base_side), base_shards))
            else:
                positions.append((int(base_side), base_shards))
        if source_side is not None and any(source_shards):
            positions.append((int(source_side), source_shards))
        positions.append((int(exec_instance), working_shards))
        # 重合持有者只允许"完整并集 × base 与 exec 同实例"（D 边界
        # recompute@基础驻留端：重建副本是并集权威，同侧 base 记录是
        # 重复副本，零载荷就地释放）；其余任何重合仍是边界事务合同
        # 违规，fail-closed。
        if (len({side for side, _ in positions}) != len(positions)
                and not (complete_union and base_side == exec_instance)):
            raise RuntimeError(
                f"three-position merge for session {session_id} collapsed "
                f"to coincident holders {positions} -- boundary transition "
                "contract violation")

        def _tie_key(side: int, shards: tuple[int, ...]) -> tuple:
            # 式(21) 平局序：逻辑 home → D 实例（exec）→ 编号；主键 =
            # 保留量降序（完整并集优先在下方单独裁决）。
            return (-sum(shards),
                    0 if side == home else 1,
                    0 if side == exec_instance else 1,
                    side)

        transfers: list[KVTransfer] = []
        # D12 修复轮二（2026-09-27）：容量回退序列披露——逐候选尝试
        # {step, target, required(逐 rank), outcome} 完整记录"主目标 →
        # 回退1 → (回退2) → 锁定"（此前只在 KVCapacityError 文本里），
        # 随 last_merge_outcome 落账（加字段不删旧字段——sh30 的
        # 既有消费方按键读取，前向兼容）。
        fallback_sequence: list[dict] = []
        if complete_union:
            # 重建副本持完整并集（H_B＋增长 ⊇ 其他位置）——零载荷：重
            # 复副本直接释放（不增加 V、不重传，§15.1 去重）。含与胜者
            # 同实例的 base 记录（重建副本已覆盖其 token 域）。
            target = int(exec_instance)
            losers = [(side, shards) for side, shards in positions[:-1]]
            incoming: list[tuple[int, tuple[int, ...]]] = []
            fallback_sequence.append({
                "step": 0, "target": target,
                "required": [0] * self.tp_degree,
                "outcome": "complete_union_zero_payload"})
        else:
            ordered = sorted(positions, key=lambda item: _tie_key(*item))
            target = None
            incoming = []
            losers = []
            capacity_errors: list[KVCapacityError] = []
            for index, (side, shards) in enumerate(ordered):
                # 回退序的每个候选都是完整目标：其余全部持有位置把目标
                # 缺失区间搬入（与 merge_from_layout 的腿构造一致）——
                # 不得丢下偏好序更高的持有位置（否则其字节成孤儿）。
                incoming = [
                    (other_side, other_shards)
                    for other_index, (other_side, other_shards)
                    in enumerate(ordered)
                    if other_index != index and any(other_shards)]
                required = tuple(
                    sum(shards_list[rank] for _side, shards_list in incoming)
                    for rank in range(self.tp_degree))
                if not any(required):
                    fallback_sequence.append({
                        "step": len(fallback_sequence), "target": int(side),
                        "required": [int(value) for value in required],
                        "outcome": "zero_required_committed"})
                    target = side
                    losers = [
                        (other_side, other_shards)
                        for other_index, (other_side, other_shards)
                        in enumerate(ordered) if other_index != index]
                    break
                try:
                    transfers.extend(self._ensure_capacity(
                        side, required,
                        phase="completion",
                        reason="merge_winner_capacity",
                        trigger_request_id=trigger_request_id,
                        reservation_request_id=reservation_request_id,
                        at_ns=at_ns))
                    fallback_sequence.append({
                        "step": len(fallback_sequence), "target": int(side),
                        "required": [int(value) for value in required],
                        "outcome": "committed"})
                    target = side
                    losers = [
                        (other_side, other_shards)
                        for other_index, (other_side, other_shards)
                        in enumerate(ordered) if other_index != index]
                    break
                except KVCapacityError as exc:
                    transfers.extend(exc.evictions)
                    fallback_sequence.append({
                        "step": len(fallback_sequence), "target": int(side),
                        "required": [int(value) for value in required],
                        "outcome": "kv_capacity_error",
                        "error": str(exc)})
                    capacity_errors.append(exc)
            if target is None:
                for exc in capacity_errors:
                    self.commit_deep_gap_records(exc.deep_gap_records)
                raise RuntimeError(
                    "three-position merge exhausted unified eviction on "
                    f"every candidate target (session {session_id}, "
                    f"request {trigger_request_id}) -- multi-sided deep "
                    "gap")
        # 锁定披露：首笔载荷提交后目标锁定（merge_capacity_lock 同语义
        # ——后续不再尝试任何回退候选）。
        fallback_sequence.append({
            "step": len(fallback_sequence), "target": int(target),
            "required": [0] * self.tp_degree,
            "outcome": "locked"})
        transferred_bytes = 0
        for side, shards in duplicate_positions:
            if not any(shards):
                continue
            # R5：重复副本零载荷释放（重建来源已持其 token 域的权威内
            # 容；§15.1 去重——不传输、不并入、不增加 V）。与
            # complete_union 的同实例 base 重复副本释放同科目。
            self._remove_local_shards(side, shards)
        for side, shards in losers:
            if not any(shards):
                continue
            if complete_union:
                # 重复副本直接释放（不传输、不并入——重建副本已持这些
                # token 域的权威内容；§15.1 去重："重复副本不增加 V"）。
                self._remove_local_shards(side, shards)
                continue
            transfer = self._shard_vector_noc_transfer(
                session_id=session_id,
                trigger_request_id=trigger_request_id,
                source_instance_index=side,
                target_instance_index=target,
                shard_bytes=shards,
                reason="merge_three_position_into_winner")
            transfers.append(transfer)
            transferred_bytes += transfer.total_bytes
            self._remove_local_shards(side, shards)
            self._add_local_shards(target, shards)
        # 守恒断言：胜者侧现有字节 == 合并后全量（全部位置字节都终到
        # 胜者——线性算术精确相等）。token 域按动作族：
        # * 重建（recompute）：exec 工作副本 token 域已覆盖 base＋
        #   source（重建复现 H_B）＋增长——context_tokens 即并集基数；
        # * 非重建族（通用守恒公式）：base 域 ＋ source 域 ＋ exec
        #   增长域（零化起步）。
        final_tokens = (
            int(session.context_tokens) if complete_union
            else base_history_tokens
            + session.boundary_source_history_tokens
            + int(session.context_tokens))
        full_shards = kv_cache_shard_bytes_for_tokens(
            self.model, final_tokens, self.tp_degree)
        if complete_union:
            # 重建副本持并集：胜者终态 = 工作副本字节（重复副本——含与
            # 胜者同实例的 base 记录——释放、不并入，§15.1 去重"重复副
            # 本不增加 V"）。
            winner_existing = tuple(int(v) for v in working_shards)
        else:
            winner_existing = tuple(
                sum(shards_list[rank]
                    for _side, shards_list in positions)
                for rank in range(self.tp_degree))
        if winner_existing != full_shards:
            raise RuntimeError(
                f"three-position merge settlement conservation failed for "
                f"session {session_id}: winner-side per-rank bytes "
                f"{winner_existing} != full-history shards {full_shards} "
                f"(positions {positions}, base_history_tokens="
                f"{base_history_tokens}, source_history_tokens="
                f"{session.boundary_source_history_tokens}, new_tokens="
                f"{new_tokens}) -- ledger integrity failure")
        working_kind_at_settlement = session.working_kind
        zero_byte_flip = bool(complete_union)
        session.working_kind = None
        session.location = self.LOCAL_HBM
        session.instance_index = target
        session.resident_prefix_layers = self.model.layers
        session.context_tokens = final_tokens
        session.total_bytes = sum(full_shards)
        session.shard_bytes = full_shards
        session.home_instance = target
        session.base_location = ""
        session.base_history_tokens = 0
        session.base_resident_prefix_layers = 0
        session.base_instance_index = None
        session.boundary_source_history_tokens = 0
        session.boundary_source_shard_bytes = ()
        session.boundary_source_instance_index = None
        session.boundary_source_resident_prefix_layers = 0
        session.boundary_source_location = ""
        session.boundary_source_complete_union = False
        session.last_merged_request_id = trigger_request_id
        loser_sides = tuple(sorted({
            side for side, _shards in losers if side != target}))
        self.last_merge_outcome = {
            "session_id": session_id,
            "direction": ("reverse" if target == exec_instance
                          else "forward"),
            "zero_byte_flip": zero_byte_flip,
            "winner_instance": target,
            "loser_instance": (
                loser_sides[0] if loser_sides else None),
            "transferred_bytes": transferred_bytes,
            "home_flipped": target != home,
            "three_position": True,
            "loser_instances": loser_sides,
            # D12 修复轮二：容量回退序列（主目标失败→回退1→(回退2)→
            # 锁定的完整审计面；加字段不删旧字段，既有消费方兼容）。
            "fallback_sequence": tuple(fallback_sequence),
        }
        self._append_kv_delta_row(
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            working_kind=working_kind_at_settlement,
            direction=self.last_merge_outcome["direction"],
            zero_byte_flip=zero_byte_flip,
            winner_instance=target,
            loser_instance=self.last_merge_outcome["loser_instance"],
            home_before=home,
            transferred_bytes=transferred_bytes,
            home_side_retained_bytes=(
                home_residual_bytes
                if home_residual_bytes is not None else sum(base_shards)),
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
        at_ns: Optional[int] = None,
        **extra,
    ) -> None:
        """决策披露侧车（C15：不改 SH 的通道）——预测器来源（递推/解
        析）与覆盖状态逐决策落账。

        M4（2026-09-26）：行携带可选 ``at_ns``（决策时刻；缺省 None
        保确定性测试——直构 KVCacheManager 的测试夹具零漂移）；SH 在
        线调用面经 _ensure_capacity 链路透传 tick，供两阶段决策验证。
        """
        self.adaptive_decisions.append({
            "session_id": session_id,
            "k_target": int(k_target),
            "source": source,
            "statuses": list(statuses),
            "at_ns": at_ns,
            **extra,
        })

    def _adaptive_retention_target_tokens(
        self,
        session_id: str,
        context_tokens: int,
        instance_index: Optional[int] = None,
        at_ns: Optional[int] = None,
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
                statuses=("cold_start_unknown_input",), at_ns=at_ns)
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
        # (2026-10-01 审查修复) eta 槽位是**效率**（≤1 的速率乘子，
        # effective = peak × η，见 event_recursion_predictor 的 _rate_of
        # 与解析下界），而 service_factors.value() 的因子族口径 = **时间
        # 倍数 ≥1**（值 = 1/η，见 history_estimator 的 efficiency_factor
        # 与 JCM ServiceFactors docstring）——旧接线直接把时间倍数塞进
        # eta 槽（池劣化 2× 时预测器把池带宽算成 2×peak，k_hide 乐观偏
        # 小）。当前生产 η_pool 被 A17'(d) 裁定恒 1.0 掩盖；一旦裁定
        # 解除或测试喂样即方向反转。此处按 1/factor 换算；gamma 槽是
        # 时间倍数（乘 base_ns），原方向正确不动。
        pool_factor = float(self.service_factors.value("pool"))
        if pool_factor <= 0.0:
            raise RuntimeError(
                "service factor for 'pool' must be a positive time "
                f"multiplier (got {pool_factor}) -- cannot invert into "
                "the predictor's eta efficiency slot")
        efficiency = _EfficiencyFactors(
            eta={"pool": 1.0 / pool_factor},
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
                statuses=("recursion_fail_closed",), at_ns=at_ns,
                pool_divisor=pool_divisor)
            return layers
        self._record_adaptive_decision(
            session_id,
            k_target=result.k_hide,
            source=result.source,
            at_ns=at_ns,
            statuses=result.statuses,
            analytic_min_k=result.analytic_min_k,
            pool_divisor=pool_divisor,
            exposed_stall_ns=result.selected_trace.exposed_stall_ns,
            cumulative_layer_offset_ns=(
                result.selected_trace.cumulative_layer_offset_ns),
            end_to_end_slowdown_ns=(
                result.selected_trace.end_to_end_slowdown_ns),
            gamma_prefill=self.service_factors.value("prefill"),
            eta_pool=1.0 / pool_factor,
        )
        return result.k_hide


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
            # M4：at_ns = completion_ns（完成时刻，SH 已传 tick）。
            self._settle_restore_groups(
                session_id, boundary="mark_complete", at_ns=completion_ns)
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
            # D09：终端注销回冲 = 真实占用（extra KV + 暂存），与准入
            # 指标（required 含暂存）对称。
            reservation_extra = self._reservation_committed_shards(
                reservation)
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

        # W9（§7.4 会话真正结束）：留存段清理与未完访问/ack 协调——悬挂
        # 引用/未完 ack/在途 copy fail-closed（**先审计后释放**——审计
        # 前置到任何字节释放之前，失败路径零残留）。
        if self._distributed_retained.get(session_id):
            assert_no_dangling_session_state(self._kv_directory, session_id)

        released_instance_index: Optional[int] = None
        if session.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}:
            if session.instance_index is None:
                raise RuntimeError("local completed session has no instance")
            released_instance_index = session.instance_index
            local_shards = self._distributed_resident_layer_bytes(
                session, 0, session.resident_prefix_layers)
            self._remove_local_shards(released_instance_index, local_shards)
        elif session.location != self.REMOTE_MEMORY:
            raise RuntimeError(f"invalid KV location for {session_id}")
        # 片上留存段字节随会话终局释放（池段无实例字节）。
        if self._distributed_retained.get(session_id):
            for segment in self._distributed_retained[session_id].values():
                if segment.is_on_chip() and any(segment.bytes_by_tp_rank):
                    self._remove_local_shards(
                        segment.instance_index, segment.bytes_by_tp_rank)
            self._distributed_retained.pop(session_id, None)

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
        # (2026-10-01 审查修复) 终局会话的输入长度样本一并回收（此后
        # _estimate_next_input 不可能再查该会话——只写不删在长 run 单调
        # 增长，与上方两个观测 map 的清理承诺一致）。
        self._input_length_stats.pop(session_id, None)
        del self._sessions[session_id]
        self._check_invariants_after_mutation(
            session_ids=(session_id,), reservation_ids=reservation_ids
        )
        return released_instance_index
