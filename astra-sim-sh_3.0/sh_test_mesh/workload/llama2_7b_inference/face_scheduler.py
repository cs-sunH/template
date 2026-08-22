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
from dataclasses import dataclass
from typing import Any, Optional, Sequence


PREFILL_CHUNK_SIZE = 512


# Read-only metrics observation (implementation doc sec.7).  When a recorder
# is installed through set_metrics_observer(), every KVCacheManager state
# mutation below is mirrored to it *after* the mutation completes; the
# recorder never feeds anything back into admission, eviction, or placement
# decisions.  When no recorder is installed (the default) none of the
# observation bookkeeping runs at all, so behavior and performance are
# unchanged.
_METRICS_RECORDER: Any = None


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
        "watermark": "prefill_start",
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


@dataclass(frozen=True)
class KVCapacityReservation:
    request_id: str
    session_id: str
    instance_index: int
    final_context_tokens: int
    final_shard_bytes: tuple[int, ...]


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
        reserve_context_tokens: int = 1_000_000,
    ) -> None:
        if (
            isinstance(reserve_context_tokens, bool)
            or not isinstance(reserve_context_tokens, int)
            or reserve_context_tokens < 0
        ):
            raise ValueError("reserve_context_tokens must be a non-negative integer")
        instance_sizes = {instance.size for instance in topology.instances}
        if len(instance_sizes) != 1:
            raise ValueError("KVCacheManager requires equal-size TP instances")

        self.topology = topology
        self.model = model
        self.tp_degree = topology.instances[0].size
        self.reserve_context_tokens = reserve_context_tokens
        # Keep the larger half resident for odd-layer models, so exactly
        # floor(L/2) trailing layers are selected for the first-stage offload.
        self.partial_resident_prefix_layers = (
            model.layers - model.layers // 2
        )
        self.model_weight_bytes_by_tp_rank = model_weight_shard_bytes_by_tp_rank(
            model,
            self.tp_degree,
        )
        self.reserve_bytes_by_tp_rank = kv_cache_shard_bytes_for_tokens(
            model, reserve_context_tokens, self.tp_degree
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
        self._check_invariants()
        # Metrics observation state (doc sec.7.4): resident KV is tracked as
        # per-session parts that carry their own token counts and layer
        # ranges, so suffix-half evictions, restores, and truncations always
        # remove the exact recorded distribution of an earlier add under its
        # own allocation key (never a merged guess).
        self._metrics_recorder = _METRICS_RECORDER
        self._metrics_parts: dict[str, list[dict[str, Any]]] = {}
        self._metrics_segment_counters: dict[str, int] = {}
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
        local_shards = self._local_session_shards(
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

    def request_hbm_feasible_instances(
        self,
        *,
        session_id: str,
        final_context_tokens: int,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[bool, ...]:
        """Report which instances can hold a request's final local KV state.

        Existing local or partially resident KV already consumes HBM on its
        resident instance, so only the missing bytes are required there.  A
        different instance must admit the complete final KV.  Partially
        resident sessions retain their resident-prefix affinity.
        """

        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            final_context_tokens,
            self.tp_degree,
        )
        session = self._sessions.get(session_id)
        resident_instance_index: Optional[int] = None
        resident_shards = tuple(0 for _ in final_shards)
        partial_affinity = False
        if session is not None:
            if session.context_tokens > final_context_tokens:
                raise ValueError("request final context cannot shrink the KV cache")
            if session.location in {self.LOCAL_HBM, self.PARTIAL_HBM_REMOTE}:
                if session.instance_index is None:
                    raise RuntimeError("resident KV session has no instance")
                resident_instance_index = session.instance_index
                resident_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    session.context_tokens,
                    self.tp_degree,
                    layer_start=0,
                    layer_end=session.resident_prefix_layers,
                )
                partial_affinity = session.location == self.PARTIAL_HBM_REMOTE
            elif session.location != self.REMOTE_MEMORY:
                raise RuntimeError(f"unknown KV location: {session.location}")

        feasible: list[bool] = []
        for instance in self.topology.instances:
            if partial_affinity and instance.index != resident_instance_index:
                feasible.append(False)
                continue
            if instance.index == resident_instance_index:
                required = tuple(
                    final_bytes - local_bytes
                    for final_bytes, local_bytes in zip(
                        final_shards,
                        resident_shards,
                    )
                )
            else:
                required = final_shards
            if any(value < 0 for value in required):
                raise ValueError("request final KV is smaller than resident KV")
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
    ) -> tuple[bool, ...]:
        """Report whether the request fits after all other sessions finish.

        This distinguishes a temporary active-session capacity conflict from a
        request whose final KV can never fit on an otherwise empty instance.
        """

        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            final_context_tokens,
            self.tp_degree,
        )
        session = self._sessions.get(session_id)
        affinity_instance: Optional[int] = None
        if session is not None and session.location == self.PARTIAL_HBM_REMOTE:
            if session.instance_index is None:
                raise RuntimeError("partial KV session has no resident instance")
            affinity_instance = session.instance_index

        feasible: list[bool] = []
        for instance in self.topology.instances:
            if affinity_instance is not None and instance.index != affinity_instance:
                feasible.append(False)
                continue
            feasible.append(
                all(
                    final_bytes
                    <= self._rank_states[rank].capacity_bytes
                    - self._rank_states[rank].model_weight_bytes
                    for rank, final_bytes in zip(instance.ranks, final_shards)
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
    ) -> tuple[KVTransfer, ...]:
        """Commit final local-KV capacity before a request enters Prefill."""

        if request_id in self._reservations:
            raise ValueError(f"duplicate HBM reservation for request {request_id}")
        feasible = self.request_hbm_feasible_instances(
            session_id=session_id,
            final_context_tokens=final_context_tokens,
        )
        if not feasible[instance_index]:
            raise RuntimeError(
                f"request {request_id} was reserved on an infeasible instance"
            )
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            final_context_tokens,
            self.tp_degree,
        )
        local_shards = self._local_session_shards(session_id, instance_index)
        required = tuple(
            final_bytes - local_bytes
            for final_bytes, local_bytes in zip(final_shards, local_shards)
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
            final_context_tokens=final_context_tokens,
            final_shard_bytes=final_shards,
        )
        self._check_invariants()
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
        """Move an active request's committed final-KV capacity to Decode."""

        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        reservation = self._reservations[request_id]
        if reservation.instance_index == target_instance_index:
            return ()
        feasible = self.request_hbm_feasible_instances(
            session_id=reservation.session_id,
            final_context_tokens=reservation.final_context_tokens,
            reservation_request_id=request_id,
        )
        if not feasible[target_instance_index]:
            raise RuntimeError(
                f"request {request_id} Decode reservation target became infeasible"
            )
        local_shards = self._local_session_shards(
            reservation.session_id,
            target_instance_index,
        )
        required = tuple(
            final_bytes - local_bytes
            for final_bytes, local_bytes in zip(
                reservation.final_shard_bytes,
                local_shards,
            )
        )
        evictions = self._ensure_capacity(
            target_instance_index,
            required,
            phase="decode",
            reason="decode_reservation_capacity",
            trigger_request_id=request_id,
            reservation_request_id=request_id,
        )
        old_reservation = reservation
        self._reservations[request_id] = KVCapacityReservation(
            request_id=reservation.request_id,
            session_id=reservation.session_id,
            instance_index=target_instance_index,
            final_context_tokens=reservation.final_context_tokens,
            final_shard_bytes=reservation.final_shard_bytes,
        )
        self._check_invariants()
        if self._metrics_recorder is not None:
            # Reservation move (doc sec.7.3): release the committed delta on
            # the source instance ranks, then commit it on the target ranks.
            old_extra = tuple(
                final_bytes - local_bytes
                for final_bytes, local_bytes in zip(
                    old_reservation.final_shard_bytes,
                    self._local_session_shards(
                        old_reservation.session_id,
                        old_reservation.instance_index,
                    ),
                )
            )
            for rank, value in zip(
                self.topology.instance(old_reservation.instance_index).ranks,
                old_extra,
            ):
                if not value:
                    continue
                self._metrics_recorder.record(
                    anchor_kind="decode_start",
                    request_id=request_id,
                    session_id=old_reservation.session_id,
                    rank=rank,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=-int(value),
                    cause="decode_reservation_move_source_remove",
                )
            for rank, value in zip(
                self.topology.instance(target_instance_index).ranks,
                required,
            ):
                if not value:
                    continue
                self._metrics_recorder.record(
                    anchor_kind="decode_start",
                    request_id=request_id,
                    session_id=old_reservation.session_id,
                    rank=rank,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=int(value),
                    cause="decode_reservation_move_target_add",
                )
        return evictions

    def release_request_capacity_reservation(self, request_id: str) -> None:
        if request_id not in self._reservations:
            raise KeyError(f"unknown HBM reservation for request {request_id}")
        reservation = self._reservations[request_id]
        if any(self._reservation_extra_shards(reservation)):
            raise RuntimeError(
                f"request {request_id} released HBM reservation before final KV allocation"
            )
        del self._reservations[request_id]
        self._check_invariants()

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
        )

    def session_snapshots(self) -> tuple[SessionKVSnapshot, ...]:
        return tuple(self.session_snapshot(session_id) for session_id in self.session_ids)

    def nearest_edge(self, rank: int) -> int:
        return nearest_edge_rank(
            self.topology.hardware,
            rank,
            self.edge_ranks,
        )

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
            local_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.context_tokens,
                self.tp_degree,
                layer_start=0,
                layer_end=session.resident_prefix_layers,
            )
            for rank, shard_bytes in zip(instance.ranks, local_shards):
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
            expected_final = kv_cache_shard_bytes_for_tokens(
                self.model,
                reservation.final_context_tokens,
                self.tp_degree,
            )
            if reservation.final_shard_bytes != expected_final:
                raise RuntimeError("HBM reservation final KV metadata is inconsistent")
        for instance in self.topology.instances:
            effective = self._effective_remaining_by_tp_rank(instance.index)
            if any(value < 0 for value in effective):
                raise RuntimeError(
                    f"instance {instance.index} HBM reservations exceed free capacity"
                )

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
                raise ValueError(
                    f"rank {rank} has insufficient local HBM: "
                    f"needs {added_bytes}, "
                    f"remaining {self._rank_states[rank].remaining_bytes}"
                )
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
    ) -> None:
        """Suffix-half eviction: clamp every part to its intersection with
        the retained prefix layers, removing the exact per-part suffix
        distribution (never a merged guess, doc sec.7.4/9.4)."""

        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.get(session_id, [])
        kept_parts: list[dict[str, Any]] = []
        for part in parts:
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
    ) -> KVTransfer:
        if session.resident_prefix_layers != self.model.layers:
            raise RuntimeError("NoC migration requires a fully resident session")
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=shard_bytes,
                noc_path=deterministic_xy_route(
                    self.topology.hardware, source_rank, target_rank
                ),
                layer_start=0,
                layer_end=self.model.layers,
            )
            for source_rank, target_rank, shard_bytes in zip(
                source.ranks, target.ranks, session.shard_bytes
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
            total_bytes=session.total_bytes,
            shards=shards,
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=self.model.layers,
            resident_prefix_layers_before=self.model.layers,
            resident_prefix_layers_after=self.model.layers,
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
        )

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
        # Typed eviction (2026-08-18): an idle session's class is the trigger
        # type recorded at its latest completion. Sessions whose next request
        # type is "tool" form the tool class; everything else -- "human" and
        # None (no successor / unspecified) -- forms the human class. The
        # None-is-human mapping is a user ruling: a session with no recorded
        # successor is evicted with human-return sessions.
        return "tool" if session.next_request_type == "tool" else "human"

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
    ) -> KVTransfer:
        if session.active or session.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("suffix eviction requires a fully local session")
        suffix_start = self.partial_resident_prefix_layers
        if suffix_start >= self.model.layers:
            raise RuntimeError("configured model has no non-empty half-layer suffix")
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_suffix_half",
            session=session,
            trigger_request_id=trigger_request_id,
            layer_start=suffix_start,
            layer_end=self.model.layers,
            resident_prefix_layers_after=suffix_start,
        )
        self._remove_local_shards(
            session.instance_index,
            kv_cache_shard_bytes_for_layer_range(
                self.model,
                session.context_tokens,
                self.tp_degree,
                layer_start=suffix_start,
                layer_end=self.model.layers,
            ),
        )
        session.location = self.PARTIAL_HBM_REMOTE
        session.resident_prefix_layers = suffix_start
        self._check_invariants()
        self._metrics_suffix_evict_parts(
            session.session_id,
            suffix_start,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=(
                f"evict_{reason}_suffix_half:"
                f"layers{suffix_start}-{self.model.layers}"
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
        self._check_invariants()
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
        # Typed two-stage reclamation (2026-08-18): classes are visited in
        # the fixed order ("human", "tool"); within each class the original
        # two stages run -- stage 1 halves each inactive full-local session
        # (suffix offload), stage 2 evicts complete resident sessions. The
        # watermark is rechecked after every single eviction, so the loops
        # stop as soon as the requirement is satisfied.
        for trigger_type in ("human", "tool"):
            # Stage 1: oldest-first within the class, move only each
            # eligible session's latter half.
            while self._insufficient_ranks(
                instance_index,
                required_bytes_by_tp_rank,
                reservation_request_id=reservation_request_id,
            ):
                candidates = self._completed_full_candidates(
                    instance_index, trigger_type
                )
                if protected_session_id is not None:
                    candidates = [
                        session
                        for session in candidates
                        if session.session_id != protected_session_id
                    ]
                if not candidates:
                    break
                evictions.append(
                    self._evict_suffix(
                        candidates[0],
                        phase=phase,
                        reason=reason,
                        trigger_request_id=trigger_request_id,
                    )
                )

            # Stage 2: only after every inactive full session of this class
            # has been halved, evict complete sessions (the remaining prefix
            # for partial sessions) in the same deterministic FIFO order.
            while self._insufficient_ranks(
                instance_index,
                required_bytes_by_tp_rank,
                reservation_request_id=reservation_request_id,
            ):
                candidates = self._completed_resident_candidates(
                    instance_index, trigger_type
                )
                if protected_session_id is not None:
                    candidates = [
                        session
                        for session in candidates
                        if session.session_id != protected_session_id
                    ]
                if not candidates:
                    break
                evictions.append(
                    self._evict_session(
                        candidates[0],
                        phase=phase,
                        reason=reason,
                        trigger_request_id=trigger_request_id,
                    )
                )

        # Every class and stage is exhausted while the watermark is still
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
            details = ", ".join(
                f"rank {rank}: remaining="
                f"{effective_remaining[self._rank_relative_index[rank]]}, "
                f"required={required_bytes_by_tp_rank[self._rank_relative_index[rank]]}"
                for rank in insufficient
            )
            raise ValueError(
                f"insufficient target HBM in instance {instance_index}; {details}"
            )
        return tuple(evictions)

    def prepare_prefill(
        self,
        *,
        session_id: str,
        target_instance_index: int,
        history_tokens: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[
        Optional[SessionKVSnapshot],
        Optional[KVTransfer],
        tuple[KVTransfer, ...],
    ]:
        expected_shards = kv_cache_shard_bytes_for_tokens(
            self.model, history_tokens, self.tp_degree
        )
        expected_total = sum(expected_shards)
        if session_id not in self._sessions:
            if history_tokens != 0:
                raise ValueError(
                    f"new session {session_id} cannot have historical KV tokens"
                )
            self._sessions[session_id] = SessionKVState(
                session_id=session_id,
                location=self.LOCAL_HBM,
                instance_index=target_instance_index,
                context_tokens=0,
                total_bytes=0,
                shard_bytes=expected_shards,
                resident_prefix_layers=self.model.layers,
                active=True,
            )
            self._check_invariants()
            return None, None, ()

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

        if session.location == self.LOCAL_HBM:
            if session.instance_index is None:
                raise RuntimeError("local history has no source instance")
            source_instance_index = session.instance_index
            if source_instance_index == target_instance_index:
                session.active = True
                transfer = self._local_hit_transfer(
                    phase="history",
                    reason="history_local_reuse",
                    session=session,
                    trigger_request_id=trigger_request_id,
                )
                self._check_invariants()
                return before, transfer, ()

            # The arriving request owns this session before any capacity
            # reclamation starts; it must never become its own eviction victim.
            session.active = True
            evictions = self._ensure_capacity(
                target_instance_index,
                session.shard_bytes,
                phase="history",
                reason="history_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
            )
            transfer = self._noc_transfer(
                phase="history",
                reason="history_other_instance",
                session=session,
                trigger_request_id=trigger_request_id,
                source_instance_index=source_instance_index,
                target_instance_index=target_instance_index,
            )
            self._remove_local_shards(source_instance_index, session.shard_bytes)
            self._add_local_shards(target_instance_index, session.shard_bytes)
            session.instance_index = target_instance_index
            self._check_invariants()
            self._metrics_move_session_parts(
                session_id,
                target_instance_index,
                session.context_tokens,
                session.resident_prefix_layers,
                session.shard_bytes,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_other_instance",
            )
            return before, transfer, evictions

        if session.location == self.PARTIAL_HBM_REMOTE:
            if session.instance_index != target_instance_index:
                raise ValueError(
                    "partially resident history must retain instance affinity"
                )
            suffix_start = session.resident_prefix_layers
            suffix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                history_tokens,
                self.tp_degree,
                layer_start=suffix_start,
                layer_end=self.model.layers,
            )
            session.active = True
            evictions = self._ensure_capacity(
                target_instance_index,
                suffix_shards,
                phase="history",
                reason="history_suffix_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
            )
            transfer = self._remote_load_transfer(
                phase="history",
                reason="history_remote_suffix_restore",
                session=session,
                trigger_request_id=trigger_request_id,
                target_instance_index=target_instance_index,
                layer_start=suffix_start,
                layer_end=self.model.layers,
            )
            self._add_local_shards(target_instance_index, suffix_shards)
            session.location = self.LOCAL_HBM
            session.resident_prefix_layers = self.model.layers
            self._check_invariants()
            # Suffix restore gets its own segment id / allocation key (doc
            # sec.7.4/9.4): a later suffix eviction removes exactly this
            # distribution again.
            self._metrics_add_segment(
                session_id,
                target_instance_index,
                history_tokens,
                suffix_start,
                self.model.layers,
                suffix_shards,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_remote_suffix_restore",
            )
            return before, transfer, evictions

        if session.location != self.REMOTE_MEMORY:
            raise RuntimeError(f"unknown history location: {session.location}")
        session.active = True
        evictions = self._ensure_capacity(
            target_instance_index,
            session.shard_bytes,
            phase="history",
            reason="history_target_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
        )
        transfer = self._remote_load_transfer(
            phase="history",
            reason="history_remote_restore",
            session=session,
            trigger_request_id=trigger_request_id,
            target_instance_index=target_instance_index,
            layer_start=0,
            layer_end=self.model.layers,
        )
        self._add_local_shards(target_instance_index, session.shard_bytes)
        session.location = self.LOCAL_HBM
        session.instance_index = target_instance_index
        session.resident_prefix_layers = self.model.layers
        self._check_invariants()
        self._metrics_add_segment(
            session_id,
            target_instance_index,
            session.context_tokens,
            0,
            self.model.layers,
            session.shard_bytes,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="history_remote_restore",
        )
        return before, transfer, evictions

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
        self._check_invariants()
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
        )
        self._remove_local_shards(source_instance_index, session.shard_bytes)
        self._add_local_shards(target_instance_index, session.shard_bytes)
        session.instance_index = target_instance_index
        self._check_invariants()
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
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("completed request KV must be local")
        session.active = False
        session.last_completion_ns = completion_ns
        # Record the idle session's trigger class before any follow-up
        # enforce_reserve pass so the typed eviction order sees it (the
        # event loop batches mark_complete before enforce_reserve).
        session.next_request_type = next_request_type
        self._check_invariants()

    def enforce_reserve(
        self,
        *,
        instance_index: int,
        trigger_request_id: str,
    ) -> tuple[tuple[KVTransfer, ...], tuple[int, ...]]:
        instance = self.topology.instance(instance_index)

        def unmet() -> tuple[int, ...]:
            return tuple(
                rank
                for relative_index, rank in enumerate(instance.ranks)
                if self._rank_states[rank].remaining_bytes
                < self.reserve_bytes_by_tp_rank[relative_index]
            )

        evictions: list[KVTransfer] = []
        # Same typed two-stage order as _ensure_capacity: ("human", "tool")
        # classes, half-suffix stage before full-session stage inside each
        # class, watermark rechecked after every eviction.
        for trigger_type in ("human", "tool"):
            while unmet():
                candidates = self._completed_full_candidates(
                    instance_index, trigger_type
                )
                if not candidates:
                    break
                evictions.append(
                    self._evict_suffix(
                        candidates[0],
                        phase="completion",
                        reason="reserve_threshold",
                        trigger_request_id=trigger_request_id,
                    )
                )

            while unmet():
                candidates = self._completed_resident_candidates(
                    instance_index, trigger_type
                )
                if not candidates:
                    break
                evictions.append(
                    self._evict_session(
                        candidates[0],
                        phase="completion",
                        reason="reserve_threshold",
                        trigger_request_id=trigger_request_id,
                    )
                )
        reserve_unmet_ranks = unmet()
        self._check_invariants()
        return tuple(evictions), reserve_unmet_ranks

    def complete_request(
        self,
        *,
        session_id: str,
        completion_ns: int,
        trigger_request_id: str,
        next_request_type: Optional[str] = None,
    ) -> tuple[tuple[KVTransfer, ...], tuple[int, ...]]:
        self.mark_complete(session_id, completion_ns, next_request_type)
        instance_index = self._sessions[session_id].instance_index
        if instance_index is None:
            raise RuntimeError("completed session lost its local instance")
        return self.enforce_reserve(
            instance_index=instance_index,
            trigger_request_id=trigger_request_id,
        )


