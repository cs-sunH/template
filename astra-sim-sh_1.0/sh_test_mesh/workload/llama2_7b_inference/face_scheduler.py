#!/usr/bin/env python3
"""Pure FACE request-mapping planner used by the Chakra trace generator.

The FACE paper describes a live host scheduler.  ASTRA-sim consumes static ET
graphs, so this module performs a deterministic discrete-event planning pass at
trace-generation time.  It implements the paper's queue ordering,
schedulable-instance range, LUT matching, per-die incremental decode cost, and
local-first KV allocation without depending on protobuf or ASTRA-sim internals.
"""

from __future__ import annotations

import csv
import heapq
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence


# Read-only metrics observation (implementation doc sec.7).  When a recorder
# is installed through set_metrics_observer(), every KVCacheManager state
# mutation below is mirrored to it *after* the mutation completes; the
# recorder never feeds anything back into admission, eviction, reservation,
# or placement decisions.  When no recorder is installed (the default) none
# of the observation bookkeeping runs at all, so behavior and performance
# are unchanged.
_METRICS_RECORDER: Any = None


def set_metrics_observer(recorder: Any) -> None:
    """Install the metrics recorder picked up by subsequently constructed
    managers (``None`` disables observation)."""

    global _METRICS_RECORDER
    _METRICS_RECORDER = recorder


# Optional read-only streaming planner-LUT statistics hook (doc sec.8.8).
# The trace generator installs it around plan_face_requests(); it observes
# every planning iteration's LUT lookup and never feeds back into scheduling.
_ITERATION_STATS_HOOK = None


def set_iteration_stats_hook(hook) -> None:
    """Install (or clear, with ``None``) the planner iteration stats hook."""

    global _ITERATION_STATS_HOOK
    _ITERATION_STATS_HOOK = hook


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
            "the static ET adapter requires equal-size instances for KV shard pairing"
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

    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    bytes_per_head = (
        2
        * model.layers
        * tokens
        * (model.hidden_size // model.num_heads)
        * model.bytes_per_elem
    )
    shards = tuple(head_count * bytes_per_head for head_count in heads)
    if sum(shards) != kv_cache_bytes_for_tokens(model, tokens):
        raise RuntimeError("whole-head KV partition does not preserve total bytes")
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
class FaceLutEntry:
    instance_size: int
    p_chunk: int
    d_batch: int
    d_token: int
    iteration_time_ns: int
    source: str = "analytical_roofline"


def _power_of_two_token_bins(max_token: int) -> tuple[int, ...]:
    if max_token < 0:
        raise ValueError("max_token must be non-negative")
    bins = [0]
    value = 1
    while value < max(1, max_token):
        value *= 2
        if value >= 128:
            bins.append(value)
    if len(bins) == 1:
        bins.append(128)
    if bins[-1] < max_token:
        bins.append(bins[-1] * 2)
    return tuple(dict.fromkeys(bins))


def estimate_iteration_time_ns(
    hardware: FaceHardware,
    model: FaceModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
) -> int:
    if instance_size <= 0 or p_chunk < 0 or d_batch < 0 or d_token < 0:
        raise ValueError("invalid LUT workload parameters")
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

    prefill_attn_ops = layers * 4 * p_chunk * max(1, p_chunk) * h
    prefill_attn_bytes = layers * p_chunk * max(1, p_chunk) * bytes_per_elem * 4
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


class FaceLut:
    def __init__(self, entries: Iterable[FaceLutEntry]) -> None:
        self.entries = tuple(entries)
        if not self.entries:
            raise ValueError("FACE LUT must contain at least one entry")
        index: dict[tuple[int, int, int], list[FaceLutEntry]] = {}
        seen: set[tuple[int, int, int, int]] = set()
        for entry in self.entries:
            key4 = (
                entry.instance_size,
                entry.p_chunk,
                entry.d_batch,
                entry.d_token,
            )
            if key4 in seen:
                raise ValueError(f"duplicate FACE LUT entry: {key4}")
            seen.add(key4)
            index.setdefault(key4[:3], []).append(entry)
        self._index = {
            key: tuple(sorted(rows, key=lambda row: row.d_token))
            for key, rows in index.items()
        }

    @classmethod
    def build(
        cls,
        hardware: FaceHardware,
        model: FaceModel,
        *,
        instance_sizes: Iterable[int],
        p_chunk: int,
        request_count: int,
        max_d_token: int,
    ) -> "FaceLut":
        if p_chunk <= 0:
            raise ValueError("p_chunk must be positive")
        if request_count <= 0:
            raise ValueError("request_count must be positive")
        token_bins = _power_of_two_token_bins(max_d_token)
        rows: list[FaceLutEntry] = []
        for instance_size in sorted(set(instance_sizes)):
            for chunk in (0, p_chunk):
                for d_batch in range(request_count + 1):
                    for d_token in token_bins:
                        if d_batch == 0 and d_token != 0:
                            continue
                        if d_batch > 0 and d_token == 0:
                            continue
                        rows.append(
                            FaceLutEntry(
                                instance_size=instance_size,
                                p_chunk=chunk,
                                d_batch=d_batch,
                                d_token=d_token,
                                iteration_time_ns=estimate_iteration_time_ns(
                                    hardware,
                                    model,
                                    instance_size=instance_size,
                                    p_chunk=chunk,
                                    d_batch=d_batch,
                                    d_token=d_token,
                                ),
                            )
                        )
        return cls(rows)

    def lookup(
        self,
        *,
        instance_size: int,
        p_chunk: int,
        d_batch: int,
        d_token: int,
    ) -> FaceLutEntry:
        rows = self._index.get((instance_size, p_chunk, d_batch))
        if not rows:
            raise KeyError(
                "FACE LUT has no exact instance_size/p_chunk/d_batch match for "
                f"({instance_size}, {p_chunk}, {d_batch})"
            )
        return min(rows, key=lambda row: (abs(row.d_token - d_token), row.d_token))

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "instance_size",
                    "p_chunk",
                    "d_batch",
                    "d_token",
                    "iteration_time_ns",
                    "source",
                ),
            )
            writer.writeheader()
            for entry in sorted(
                self.entries,
                key=lambda row: (
                    row.instance_size,
                    row.p_chunk,
                    row.d_batch,
                    row.d_token,
                ),
            ):
                writer.writerow(entry.__dict__)


@dataclass(frozen=True)
class PrefillQueueSnapshot:
    instance_index: int
    remaining_chunks: int
    last_arrival_ns: Optional[int]

    @property
    def ordering_key(self) -> tuple[int, int, int]:
        return (
            self.remaining_chunks,
            -1 if self.last_arrival_ns is None else self.last_arrival_ns,
            self.instance_index,
        )


def select_prefill_instance(
    queues: Sequence[PrefillQueueSnapshot],
    hbm_feasible_instances: Optional[Sequence[bool]] = None,
) -> int:
    if not queues:
        raise ValueError("at least one prefill queue is required")
    if hbm_feasible_instances is None:
        hbm_feasible_instances = tuple(
            True for _ in range(max(queue.instance_index for queue in queues) + 1)
        )
    if len(hbm_feasible_instances) <= max(queue.instance_index for queue in queues):
        raise ValueError("hbm_feasible_instances must cover every Prefill instance")
    if any(not isinstance(feasible, bool) for feasible in hbm_feasible_instances):
        raise ValueError("hbm_feasible_instances must contain booleans")
    feasible_queues = [
        queue
        for queue in queues
        if hbm_feasible_instances[queue.instance_index]
    ]
    if not feasible_queues:
        raise ValueError("no Prefill instance has enough reclaimable per-rank HBM")
    return min(feasible_queues, key=lambda queue: queue.ordering_key).instance_index


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
    current_lut: FaceLutEntry
    updated_lut: FaceLutEntry
    delta_time_ns: int
    per_die_delta_ns: float


def select_decode_instance(
    *,
    topology: FaceTopology,
    graph: WeightedInstanceGraph,
    lut: FaceLut,
    fixed_p_chunk: int,
    prefill_instance_index: int,
    has_prefill_work: Sequence[bool],
    decode_token_lengths: Sequence[Sequence[int]],
    new_request_token_length: int,
    hbm_feasible_instances: Optional[Sequence[bool]] = None,
) -> tuple[int, tuple[DecodeCandidateCost, ...]]:
    if len(has_prefill_work) != len(topology.instances):
        raise ValueError("has_prefill_work length must match instances")
    if len(decode_token_lengths) != len(topology.instances):
        raise ValueError("decode_token_lengths length must match instances")
    if hbm_feasible_instances is None:
        hbm_feasible_instances = tuple(True for _ in topology.instances)
    if len(hbm_feasible_instances) != len(topology.instances):
        raise ValueError("hbm_feasible_instances length must match instances")
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
        current = lut.lookup(
            instance_size=instance.size,
            p_chunk=p_chunk,
            d_batch=d_batch,
            d_token=d_token,
        )
        updated = lut.lookup(
            instance_size=instance.size,
            p_chunk=p_chunk,
            d_batch=d_batch + 1,
            d_token=max(d_token, new_request_token_length),
        )
        delta = updated.iteration_time_ns - current.iteration_time_ns
        costs.append(
            DecodeCandidateCost(
                instance_index=instance_index,
                weighted_distance=distance,
                current_lut=current,
                updated_lut=updated,
                delta_time_ns=delta,
                per_die_delta_ns=delta / instance.size,
            )
        )
    feasible_costs = [
        cost for cost in costs if hbm_feasible_instances[cost.instance_index]
    ]
    if not feasible_costs:
        raise ValueError("no schedulable Decode instance has reclaimable per-rank HBM")
    selected = min(
        feasible_costs,
        key=lambda cost: (cost.per_die_delta_ns, cost.instance_index),
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
    last_completion_ns: Optional[int] = None
    active: bool = False


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

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("KV transfer shard bytes must be non-negative")


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
        shard_total = sum(shard.bytes for shard in self.shards)
        if self.kind == "local_hit" and self.shards:
            raise ValueError("local-hit KV transfer must not contain network shards")
        if self.kind != "local_hit" and shard_total != self.total_bytes:
            raise ValueError("KV transfer shard bytes do not match total bytes")


class KVCacheManager:
    """Deterministic per-rank HBM and complete-session KV state manager."""

    LOCAL_HBM = "local_hbm"
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
        # Read-only metrics observation state (doc sec.7).  The recorder is
        # captured at construction time; when none is installed every helper
        # below is a no-op.  ``_metrics_segments`` tracks one live allocation
        # segment list per session for chiplet-projection add/remove symmetry
        # (doc sec.7.4/7.6); ``_metrics_reservation_extras`` mirrors the
        # derived per-rank reservation extras so the observer ledger always
        # equals _reserved_bytes_by_tp_rank().
        self._metrics_recorder = _METRICS_RECORDER
        self._metrics_segments: dict[str, dict[str, list[object]]] = {}
        self._metrics_segment_counters: dict[str, int] = {}
        self._metrics_reservation_extras: dict[
            str, tuple[int, tuple[int, ...]]
        ] = {}
        if self._metrics_recorder is not None:
            for rank in sorted(self._rank_states):
                self._metrics_recorder.initialize_rank(
                    rank, self._rank_states[rank].capacity_bytes
                )
            for rank in sorted(self._rank_states):
                state = self._rank_states[rank]
                if not state.model_weight_bytes:
                    continue
                self._metrics_recorder.record(
                    planner_time_ns=0,
                    anchor_kind="tick_zero",
                    request_id=None,
                    session_id=None,
                    rank=rank,
                    allocation_key=f"weight:{rank}",
                    weight_delta_bytes=int(state.model_weight_bytes),
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
            or session.location != self.LOCAL_HBM
            or session.instance_index != instance_index
        ):
            return tuple(0 for _ in range(self.tp_degree))
        return session.shard_bytes

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

    def _instance_reclaimable_capacity_by_tp_rank(
        self,
        instance_index: int,
        *,
        exclude_session_id: Optional[str] = None,
        exclude_request_id: Optional[str] = None,
    ) -> tuple[int, ...]:
        reclaimable = list(
            self._effective_remaining_by_tp_rank(
                instance_index,
                exclude_request_id=exclude_request_id,
            )
        )
        for session in self._completed_local_candidates(instance_index):
            if session.session_id == exclude_session_id:
                continue
            reclaimable = [
                available + session_bytes
                for available, session_bytes in zip(
                    reclaimable,
                    session.shard_bytes,
                )
            ]
        return tuple(reclaimable)

    def request_hbm_feasible_instances(
        self,
        *,
        session_id: str,
        final_context_tokens: int,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[bool, ...]:
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            final_context_tokens,
            self.tp_degree,
        )
        session = self._sessions.get(session_id)
        if session is not None and session.context_tokens > final_context_tokens:
            raise ValueError("request final context cannot shrink the KV cache")

        feasible: list[bool] = []
        for instance in self.topology.instances:
            local_shards = self._local_session_shards(session_id, instance.index)
            required = tuple(
                final_bytes - local_bytes
                for final_bytes, local_bytes in zip(final_shards, local_shards)
            )
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
        del session_id
        final_shards = kv_cache_shard_bytes_for_tokens(
            self.model,
            final_context_tokens,
            self.tp_degree,
        )
        return tuple(
            all(
                final_bytes
                <= self._rank_states[rank].capacity_bytes
                - self._rank_states[rank].model_weight_bytes
                for rank, final_bytes in zip(instance.ranks, final_shards)
            )
            for instance in self.topology.instances
        )

    def reserve_request_capacity(
        self,
        *,
        request_id: str,
        session_id: str,
        instance_index: int,
        final_context_tokens: int,
        now_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
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
        evictions = self._ensure_capacity(
            instance_index,
            required,
            phase="history",
            reason="request_admission_capacity",
            trigger_request_id=request_id,
            protected_session_id=session_id,
            now_ns=now_ns,
        )
        self._reservations[request_id] = KVCapacityReservation(
            request_id=request_id,
            session_id=session_id,
            instance_index=instance_index,
            final_context_tokens=final_context_tokens,
            final_shard_bytes=final_shards,
        )
        self._check_invariants()
        self._metrics_sync_reservation(
            request_id,
            now_ns=now_ns,
            anchor_kind="prefill_start",
            cause="reserve_request_capacity",
        )
        return evictions

    def move_request_capacity_reservation(
        self,
        *,
        request_id: str,
        target_instance_index: int,
        now_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
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
            now_ns=now_ns,
        )
        self._reservations[request_id] = KVCapacityReservation(
            request_id=reservation.request_id,
            session_id=reservation.session_id,
            instance_index=target_instance_index,
            final_context_tokens=reservation.final_context_tokens,
            final_shard_bytes=reservation.final_shard_bytes,
        )
        self._check_invariants()
        self._metrics_sync_reservation(
            request_id,
            now_ns=now_ns,
            anchor_kind="decode_start",
            cause="move_request_capacity_reservation",
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
        self._metrics_release_reservation(request_id)

    def decode_hbm_feasible_instances(
        self,
        *,
        session_id: str,
        final_context_tokens: int,
        reservation_request_id: Optional[str] = None,
    ) -> tuple[bool, ...]:
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
        rank_bytes: tuple[tuple[int, int], ...] = ()
        if state.location == self.LOCAL_HBM:
            if state.instance_index is None:
                raise RuntimeError("local KV session has no instance")
            instance = self.topology.instance(state.instance_index)
            rank_bytes = tuple(zip(instance.ranks, state.shard_bytes))
        return SessionKVSnapshot(
            session_id=state.session_id,
            location=state.location,
            instance_index=state.instance_index,
            context_tokens=state.context_tokens,
            total_bytes=state.total_bytes,
            shard_bytes=state.shard_bytes,
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
        expected_kv = {rank: 0 for rank in self._rank_states}
        for session in self._sessions.values():
            if session.location not in {self.LOCAL_HBM, self.REMOTE_MEMORY}:
                raise RuntimeError(f"invalid KV location for {session.session_id}")
            if sum(session.shard_bytes) != session.total_bytes:
                raise RuntimeError(
                    f"KV shards do not preserve total for {session.session_id}"
                )
            if session.location == self.REMOTE_MEMORY:
                if session.instance_index is not None:
                    raise RuntimeError("remote KV session retained a local instance")
                continue
            if session.instance_index is None:
                raise RuntimeError("local KV session has no instance")
            instance = self.topology.instance(session.instance_index)
            if len(session.shard_bytes) != instance.size:
                raise RuntimeError("KV shard count does not match TP instance")
            for rank, shard_bytes in zip(instance.ranks, session.shard_bytes):
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
    # none of them feeds back into manager decisions.  Each is invoked only
    # *after* the identical manager state change has completed (doc sec.7.2).
    # ------------------------------------------------------------------

    def _metrics_now(self, now_ns: Optional[int]) -> int:
        if now_ns is None:
            raise RuntimeError(
                "metrics observation requires the planner to pass now_ns"
            )
        return int(now_ns)

    def _metrics_add_segment(
        self,
        session_id: str,
        instance_index: int,
        shards: Sequence[int],
        *,
        now_ns: Optional[int],
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
        self._metrics_segments.setdefault(session_id, {})[segment_id] = [
            instance_index,
            tuple(int(value) for value in shards),
        ]
        instance = self.topology.instance(instance_index)
        planner_time_ns = self._metrics_now(now_ns)
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            recorder.record(
                planner_time_ns=planner_time_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                allocation_key=f"resident:{session_id}:{segment_id}",
                resident_kv_delta_bytes=int(value),
                cause=cause,
            )

    def _metrics_remove_session_segments(
        self,
        session_id: str,
        *,
        now_ns: Optional[int],
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        segments = self._metrics_segments.get(session_id, {})
        planner_time_ns = self._metrics_now(now_ns)
        for segment_id in sorted(segments):
            instance_index, shards = segments[segment_id]
            instance = self.topology.instance(instance_index)
            for rank, value in zip(instance.ranks, shards):
                if not value:
                    continue
                recorder.record(
                    planner_time_ns=planner_time_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    allocation_key=f"resident:{session_id}:{segment_id}",
                    resident_kv_delta_bytes=-int(value),
                    cause=cause,
                )
        segments.clear()

    def _metrics_move_session_segments(
        self,
        session_id: str,
        target_instance_index: int,
        total_shards: Sequence[int],
        *,
        now_ns: Optional[int],
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Mirror a migration: the target add is recorded as one consolidated
        segment (a fresh allocation key carries the full magnitude, so the
        chiplet projection stays exactly removable), then each original
        segment is removed from the source ranks under its own key (doc
        sec.7.4/7.6).  Target add precedes source release, matching the
        static ET ordering."""

        recorder = self._metrics_recorder
        if recorder is None:
            return
        segments = self._metrics_segments.get(session_id, {})
        target_ranks = self.topology.instance(target_instance_index).ranks
        counter = self._metrics_segment_counters.get(session_id, 0)
        self._metrics_segment_counters[session_id] = counter + 1
        segment_id = f"seg{counter}"
        consolidated_key = f"resident:{session_id}:{segment_id}"
        planner_time_ns = self._metrics_now(now_ns)
        for rank, value in zip(target_ranks, total_shards):
            if not value:
                continue
            recorder.record(
                planner_time_ns=planner_time_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                allocation_key=consolidated_key,
                resident_kv_delta_bytes=int(value),
                cause=f"{cause}_target_add",
            )
        for old_segment_id in sorted(segments):
            source_instance_index, shards = segments[old_segment_id]
            source_ranks = self.topology.instance(source_instance_index).ranks
            old_key = f"resident:{session_id}:{old_segment_id}"
            for rank, value in zip(source_ranks, shards):
                if not value:
                    continue
                recorder.record(
                    planner_time_ns=planner_time_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    allocation_key=old_key,
                    resident_kv_delta_bytes=-int(value),
                    cause=f"{cause}_source_remove",
                )
        self._metrics_segments[session_id] = {
            segment_id: [target_instance_index, tuple(int(v) for v in total_shards)]
        }

    def _metrics_sync_reservation(
        self,
        request_id: Optional[str],
        *,
        now_ns: Optional[int],
        anchor_kind: str,
        cause: str,
    ) -> None:
        """Mirror the *derived* reservation extra (final shards minus the
        session's local shards on the reservation instance) for one request.

        SH1 does not store reservation bytes; the effective reservation
        changes whenever the reservation is created/moved or the session's
        local KV grows or lands on the reservation instance.  Recomputing
        the extra after each such mutation and recording only the difference
        keeps the observer ledger exactly equal to
        ``_reserved_bytes_by_tp_rank()`` without touching any decision.
        """

        recorder = self._metrics_recorder
        if recorder is None or request_id is None:
            return
        reservation = self._reservations.get(request_id)
        mirrored = self._metrics_reservation_extras.get(request_id)
        if reservation is None and mirrored is None:
            return
        if reservation is None:
            new_instance_index: Optional[int] = None
            new_extra: Optional[tuple[int, ...]] = None
        else:
            new_instance_index = reservation.instance_index
            new_extra = self._reservation_extra_shards(reservation)
        if mirrored is None:
            old_instance_index: Optional[int] = None
            old_extra: Optional[tuple[int, ...]] = None
        else:
            old_instance_index, old_extra = mirrored
        if old_instance_index == new_instance_index and old_extra == new_extra:
            return
        planner_time_ns = self._metrics_now(now_ns)
        allocation_key = f"reservation:{request_id}"
        session_id = reservation.session_id if reservation is not None else None
        if (
            old_instance_index is not None
            and old_extra is not None
            and old_instance_index != new_instance_index
        ):
            old_ranks = self.topology.instance(old_instance_index).ranks
            for rank, value in zip(old_ranks, old_extra):
                if not value:
                    continue
                recorder.record(
                    planner_time_ns=planner_time_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    allocation_key=allocation_key,
                    reserved_kv_delta_bytes=-int(value),
                    cause=f"{cause}_source_remove",
                )
        if new_instance_index is not None and new_extra is not None:
            new_ranks = self.topology.instance(new_instance_index).ranks
            if old_instance_index == new_instance_index and old_extra is not None:
                for rank, old_value, new_value in zip(
                    new_ranks, old_extra, new_extra
                ):
                    delta = int(new_value) - int(old_value)
                    if not delta:
                        continue
                    recorder.record(
                        planner_time_ns=planner_time_ns,
                        anchor_kind=anchor_kind,
                        request_id=request_id,
                        session_id=session_id,
                        rank=rank,
                        allocation_key=allocation_key,
                        reserved_kv_delta_bytes=delta,
                        cause=cause,
                    )
            else:
                for rank, value in zip(new_ranks, new_extra):
                    if not value:
                        continue
                    recorder.record(
                        planner_time_ns=planner_time_ns,
                        anchor_kind=anchor_kind,
                        request_id=request_id,
                        session_id=session_id,
                        rank=rank,
                        allocation_key=allocation_key,
                        reserved_kv_delta_bytes=int(value),
                        cause=f"{cause}_target_add",
                    )
        if new_instance_index is None:
            self._metrics_reservation_extras.pop(request_id, None)
        else:
            assert new_extra is not None
            self._metrics_reservation_extras[request_id] = (
                new_instance_index,
                new_extra,
            )

    def _metrics_release_reservation(self, request_id: str) -> None:
        """Drop the reservation mirror at release; the manager guarantees the
        derived extra is zero by then, so any nonzero mirror is a bug."""

        if self._metrics_recorder is None:
            return
        mirrored = self._metrics_reservation_extras.pop(request_id, None)
        if mirrored is not None and any(mirrored[1]):
            raise RuntimeError(
                f"metrics reservation mirror for request {request_id} diverged: "
                f"unreleased extra shards {mirrored[1]}"
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
        )

    def _remote_load_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        target_instance_index: int,
    ) -> KVTransfer:
        target = self.topology.instance(target_instance_index)
        shards = []
        for target_rank, shard_bytes in zip(target.ranks, session.shard_bytes):
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
            total_bytes=session.total_bytes,
            shards=tuple(shards),
        )

    def _remote_store_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
    ) -> KVTransfer:
        if session.instance_index is None:
            raise RuntimeError("cannot store a non-local KV session")
        source_instance_index = session.instance_index
        source = self.topology.instance(source_instance_index)
        shards = []
        for source_rank, shard_bytes in zip(source.ranks, session.shard_bytes):
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
            total_bytes=session.total_bytes,
            shards=tuple(shards),
        )

    def _completed_local_candidates(
        self,
        instance_index: int,
    ) -> list[SessionKVState]:
        candidates = [
            session
            for session in self._sessions.values()
            if session.location == self.LOCAL_HBM
            and session.instance_index == instance_index
            and not session.active
            and session.last_completion_ns is not None
        ]
        candidates.sort(
            key=lambda session: (
                int(session.last_completion_ns),
                session.session_id,
            )
        )
        return candidates

    def _evict_session(
        self,
        session: SessionKVState,
        *,
        phase: str,
        reason: str,
        trigger_request_id: str,
        now_ns: Optional[int] = None,
    ) -> KVTransfer:
        if session.active or session.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("only local sessions may be evicted")
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=reason,
            session=session,
            trigger_request_id=trigger_request_id,
        )
        self._remove_local_shards(session.instance_index, session.shard_bytes)
        session.location = self.REMOTE_MEMORY
        session.instance_index = None
        self._check_invariants()
        self._metrics_remove_session_segments(
            session.session_id,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=f"evict_remote_store:{reason}",
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
        now_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        evictions: list[KVTransfer] = []
        while self._insufficient_ranks(
            instance_index,
            required_bytes_by_tp_rank,
            reservation_request_id=reservation_request_id,
        ):
            candidates = self._completed_local_candidates(instance_index)
            if protected_session_id is not None:
                candidates = [
                    session
                    for session in candidates
                    if session.session_id != protected_session_id
                ]
            if not candidates:
                insufficient = self._insufficient_ranks(
                    instance_index,
                    required_bytes_by_tp_rank,
                    reservation_request_id=reservation_request_id,
                )
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
            evictions.append(
                self._evict_session(
                    candidates[0],
                    phase=phase,
                    reason=reason,
                    trigger_request_id=trigger_request_id,
                    now_ns=now_ns,
                )
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
        now_ns: Optional[int] = None,
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

            evictions = self._ensure_capacity(
                target_instance_index,
                session.shard_bytes,
                phase="history",
                reason="history_target_capacity",
                trigger_request_id=trigger_request_id,
                reservation_request_id=reservation_request_id,
                now_ns=now_ns,
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
            session.active = True
            self._check_invariants()
            # Observation order matters for the ledger invariant: the
            # reservation consumption is mirrored before the resident add so
            # committed never transiently counts the same bytes twice.
            self._metrics_sync_reservation(
                (
                    reservation_request_id
                    if reservation_request_id is not None
                    else trigger_request_id
                ),
                now_ns=now_ns,
                anchor_kind="transfer_complete",
                cause="history_transfer_reservation_consumption",
            )
            self._metrics_move_session_segments(
                session_id,
                target_instance_index,
                session.shard_bytes,
                now_ns=now_ns,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_noc_migrate",
            )
            return before, transfer, evictions

        if session.location != self.REMOTE_MEMORY:
            raise RuntimeError(f"unknown history location: {session.location}")
        evictions = self._ensure_capacity(
            target_instance_index,
            session.shard_bytes,
            phase="history",
            reason="history_target_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
            now_ns=now_ns,
        )
        transfer = self._remote_load_transfer(
            phase="history",
            reason="history_remote_restore",
            session=session,
            trigger_request_id=trigger_request_id,
            target_instance_index=target_instance_index,
        )
        self._add_local_shards(target_instance_index, session.shard_bytes)
        session.location = self.LOCAL_HBM
        session.instance_index = target_instance_index
        session.active = True
        self._check_invariants()
        self._metrics_sync_reservation(
            (
                reservation_request_id
                if reservation_request_id is not None
                else trigger_request_id
            ),
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            cause="history_transfer_reservation_consumption",
        )
        self._metrics_add_segment(
            session_id,
            target_instance_index,
            session.shard_bytes,
            now_ns=now_ns,
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
        now_ns: Optional[int] = None,
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
            now_ns=now_ns,
        )
        self._add_local_shards(instance_index, delta)
        session.context_tokens = context_tokens
        session.total_bytes = sum(new_shards)
        session.shard_bytes = new_shards
        self._check_invariants()
        growth_anchor = "completion" if phase == "decode" else "prefill_start"
        self._metrics_sync_reservation(
            (
                reservation_request_id
                if reservation_request_id is not None
                else trigger_request_id
            ),
            now_ns=now_ns,
            anchor_kind=growth_anchor,
            cause=f"{phase}_growth_reservation_consumption",
        )
        self._metrics_add_segment(
            session_id,
            instance_index,
            delta,
            now_ns=now_ns,
            anchor_kind=growth_anchor,
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
        now_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        return self._expand_local_session(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=context_tokens,
            phase="prefill",
            reason="prefill_growth_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
            now_ns=now_ns,
        )

    def move_prefill_to_decode(
        self,
        *,
        session_id: str,
        target_instance_index: int,
        trigger_request_id: str,
        reservation_request_id: Optional[str] = None,
        now_ns: Optional[int] = None,
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
            now_ns=now_ns,
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
        self._metrics_sync_reservation(
            (
                reservation_request_id
                if reservation_request_id is not None
                else trigger_request_id
            ),
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            cause="prefill_decode_transfer_reservation_consumption",
        )
        self._metrics_move_session_segments(
            session_id,
            target_instance_index,
            session.shard_bytes,
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="prefill_decode_migrate",
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
        now_ns: Optional[int] = None,
    ) -> tuple[KVTransfer, ...]:
        return self._expand_local_session(
            session_id=session_id,
            instance_index=instance_index,
            context_tokens=final_context_tokens,
            phase="decode",
            reason="decode_growth_capacity",
            trigger_request_id=trigger_request_id,
            reservation_request_id=reservation_request_id,
            now_ns=now_ns,
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

    def mark_complete(self, session_id: str, completion_ns: int) -> None:
        session = self._sessions[session_id]
        if session.location != self.LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("completed request KV must be local")
        session.active = False
        session.last_completion_ns = completion_ns
        self._check_invariants()

    def enforce_reserve(
        self,
        *,
        instance_index: int,
        trigger_request_id: str,
        now_ns: Optional[int] = None,
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
        while unmet():
            candidates = self._completed_local_candidates(instance_index)
            if not candidates:
                break
            evictions.append(
                self._evict_session(
                    candidates[0],
                    phase="completion",
                    reason="reserve_threshold",
                    trigger_request_id=trigger_request_id,
                    now_ns=now_ns,
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
        now_ns: Optional[int] = None,
    ) -> tuple[tuple[KVTransfer, ...], tuple[int, ...]]:
        self.mark_complete(session_id, completion_ns)
        instance_index = self._sessions[session_id].instance_index
        if instance_index is None:
            raise RuntimeError("completed session lost its local instance")
        return self.enforce_reserve(
            instance_index=instance_index,
            trigger_request_id=trigger_request_id,
            now_ns=now_ns,
        )


@dataclass(frozen=True)
class FaceRequest:
    queue_index: int
    session_id: str
    turn_index: int
    request_id: str
    prefill_length: int
    decode_length: int
    session_arrival_time_ns: Optional[int]
    inter_request_interval_ns: Optional[int]

    def __post_init__(self) -> None:
        if self.queue_index < 0 or self.turn_index < 0:
            raise ValueError("queue_index and turn_index must be non-negative")
        if not self.session_id or not self.request_id:
            raise ValueError("session_id and request_id must not be empty")
        if self.prefill_length <= 0 or self.decode_length <= 0:
            raise ValueError("prefill_length and decode_length must be positive")
        if self.turn_index == 0:
            if self.session_arrival_time_ns is None or self.session_arrival_time_ns < 0:
                raise ValueError("first turn requires a non-negative session arrival")
            if self.inter_request_interval_ns is not None:
                raise ValueError("first turn must not define an inter-request interval")
        else:
            if self.session_arrival_time_ns is not None:
                raise ValueError("later turns must not define a session arrival")
            if (
                self.inter_request_interval_ns is None
                or self.inter_request_interval_ns < 0
            ):
                raise ValueError("later turns require a non-negative interval")


@dataclass(frozen=True)
class FaceIterationRecord:
    instance_index: int
    iteration_index: int
    start_ns: int
    end_ns: int
    prefill_request_id: Optional[str]
    prefill_chunk_tokens: int
    decode_request_ids: tuple[str, ...]
    lut_entry: FaceLutEntry


@dataclass(frozen=True)
class FaceRequestPlan:
    queue_index: int
    session_id: str
    turn_index: int
    request_id: str
    history_tokens_before: int
    prefill_context_tokens: int
    final_context_tokens: int
    estimated_arrival_ns: int
    admission_time_ns: int
    hbm_wait_ns: int
    prefill_instance_index: int
    prefill_assignment_key: tuple[int, int, int]
    prefill_start_ns: int
    prefill_complete_ns: int
    decode_instance_index: int
    decode_candidates: tuple[DecodeCandidateCost, ...]
    decode_start_ns: int
    completion_ns: int
    kv_allocation: KVAllocation
    history_source_instance_index: Optional[int]
    history_transfer_bytes: int
    history_location_before: Optional[SessionKVSnapshot] = None
    history_transfer: Optional[KVTransfer] = None
    history_evictions: tuple[KVTransfer, ...] = ()
    prefill_evictions: tuple[KVTransfer, ...] = ()
    prefill_decode_transfer: Optional[KVTransfer] = None
    decode_evictions: tuple[KVTransfer, ...] = ()
    completion_evictions: tuple[KVTransfer, ...] = ()
    kv_location_after_completion: Optional[str] = None
    kv_instance_after_completion: Optional[int] = None
    hbm_after_completion: tuple[NodeHBMSnapshot, ...] = ()
    reserve_unmet_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class FacePlan:
    p_chunk: int
    topology: FaceTopology
    lut: FaceLut
    requests: tuple[FaceRequestPlan, ...]
    iterations: tuple[FaceIterationRecord, ...]
    iterations_recorded: bool
    final_edge_weights: tuple[tuple[int, int, int], ...]
    final_remaining_capacity_bytes: tuple[int, ...]
    edge_ranks: tuple[int, ...] = ()
    reserve_context_tokens: int = 1_000_000
    final_hbm_states: tuple[NodeHBMSnapshot, ...] = ()
    final_session_states: tuple[SessionKVSnapshot, ...] = ()


@dataclass
class _RequestRuntime:
    request: FaceRequest
    history_tokens_before: int
    prefill_context_tokens: int
    final_context_tokens: int
    remaining_chunks: int
    prompt_tokens_processed: int = 0
    decode_steps_remaining: int = 0
    current_decode_token: int = 0
    estimated_arrival_ns: Optional[int] = None
    admission_time_ns: Optional[int] = None
    prefill_instance_index: Optional[int] = None
    prefill_assignment_key: Optional[tuple[int, int, int]] = None
    prefill_start_ns: Optional[int] = None
    prefill_complete_ns: Optional[int] = None
    decode_instance_index: Optional[int] = None
    decode_candidates: tuple[DecodeCandidateCost, ...] = ()
    decode_start_ns: Optional[int] = None
    completion_ns: Optional[int] = None
    kv_allocation: Optional[KVAllocation] = None
    history_source_instance_index: Optional[int] = None
    history_transfer_bytes: int = 0
    history_location_before: Optional[SessionKVSnapshot] = None
    history_transfer: Optional[KVTransfer] = None
    history_evictions: tuple[KVTransfer, ...] = ()
    prefill_evictions: tuple[KVTransfer, ...] = ()
    prefill_decode_transfer: Optional[KVTransfer] = None
    decode_evictions: tuple[KVTransfer, ...] = ()
    completion_evictions: tuple[KVTransfer, ...] = ()
    kv_location_after_completion: Optional[str] = None
    kv_instance_after_completion: Optional[int] = None
    hbm_after_completion: tuple[NodeHBMSnapshot, ...] = ()
    reserve_unmet_ranks: tuple[int, ...] = ()


@dataclass
class _InstanceRuntime:
    index: int
    qp: deque[int] = field(default_factory=deque)
    active_decode: list[int] = field(default_factory=list)
    last_arrival_ns: Optional[int] = None
    busy: bool = False
    iteration_count: int = 0


def _validate_and_expand_requests(
    requests: Sequence[FaceRequest],
    p_chunk: int,
) -> tuple[list[_RequestRuntime], dict[int, Optional[int]]]:
    if not requests:
        raise ValueError("request list must not be empty")
    if len({request.request_id for request in requests}) != len(requests):
        raise ValueError("request IDs must be unique")
    if len({request.queue_index for request in requests}) != len(requests):
        raise ValueError("queue indexes must be unique")

    by_session: dict[str, list[int]] = {}
    for index, request in enumerate(requests):
        by_session.setdefault(request.session_id, []).append(index)

    runtimes: list[Optional[_RequestRuntime]] = [None] * len(requests)
    next_request: dict[int, Optional[int]] = {}
    for session_id, indexes in by_session.items():
        ordered = sorted(indexes, key=lambda idx: requests[idx].queue_index)
        turns = [requests[idx].turn_index for idx in ordered]
        if turns != list(range(len(ordered))):
            raise ValueError(
                f"session {session_id} turn indexes must be contiguous from zero"
            )
        history = 0
        for position, index in enumerate(ordered):
            request = requests[index]
            prefill_context = history + request.prefill_length
            final_context = prefill_context + request.decode_length
            runtimes[index] = _RequestRuntime(
                request=request,
                history_tokens_before=history,
                prefill_context_tokens=prefill_context,
                final_context_tokens=final_context,
                remaining_chunks=math.ceil(request.prefill_length / p_chunk),
                decode_steps_remaining=request.decode_length,
                current_decode_token=prefill_context,
            )
            next_request[index] = ordered[position + 1] if position + 1 < len(ordered) else None
            history = final_context
    return [runtime for runtime in runtimes if runtime is not None], next_request


def plan_face_requests(
    *,
    hardware: FaceHardware,
    model: FaceModel,
    instance_specs: Sequence[FaceInstanceSpec],
    requests: Sequence[FaceRequest],
    edge_ranks: Optional[Sequence[int]] = None,
    reserve_context_tokens: int = 1_000_000,
    record_iterations: bool = True,
    prefill_chunk_size: Optional[int] = None,
) -> FacePlan:
    topology = build_instances(hardware, instance_specs, require_equal_size=True)
    tp_degree = topology.instances[0].size
    # LLaMA 2 7B dimensions are partitioned exactly across the configured TP.
    # ET generation assigns whole heads and exact MLP/vocabulary slices
    # unevenly by rank whenever a model dimension is not divisible by TP.

    if prefill_chunk_size is not None:
        if (
            isinstance(prefill_chunk_size, bool)
            or not isinstance(prefill_chunk_size, int)
            or prefill_chunk_size <= 0
        ):
            raise ValueError("prefill_chunk_size must be a positive integer")
        p_chunk = prefill_chunk_size
    else:
        p_chunk = math.ceil(
            sum(request.prefill_length for request in requests) / len(requests)
        )
    runtimes, next_request = _validate_and_expand_requests(requests, p_chunk)
    max_d_token = max(runtime.final_context_tokens for runtime in runtimes)
    lut = FaceLut.build(
        hardware,
        model,
        instance_sizes=(instance.size for instance in topology.instances),
        p_chunk=p_chunk,
        request_count=len(requests),
        max_d_token=max_d_token,
    )
    graph = WeightedInstanceGraph(topology)
    kv_manager = KVCacheManager(
        topology,
        model,
        edge_ranks=edge_ranks,
        reserve_context_tokens=reserve_context_tokens,
    )
    instances = [_InstanceRuntime(index=i) for i in range(len(topology.instances))]

    # Event tuple: (time_ns, priority, sequence, kind, payload).  All events at
    # one timestamp are drained before starting new iterations.  Completion has
    # priority over arrival, as required by the adapter contract.
    event_heap: list[tuple[int, int, int, str, object]] = []
    sequence = 0

    def push_event(time_ns: int, priority: int, kind: str, payload: object) -> None:
        nonlocal sequence
        heapq.heappush(event_heap, (time_ns, priority, sequence, kind, payload))
        sequence += 1

    for index, runtime in enumerate(runtimes):
        if runtime.request.turn_index == 0:
            arrival = runtime.request.session_arrival_time_ns
            if arrival is None:
                raise RuntimeError("validated first request lost its arrival time")
            push_event(arrival, 1, "arrival", index)

    iterations: list[FaceIterationRecord] = []
    pending_admissions: deque[int] = deque()

    def queue_snapshot(state: _InstanceRuntime) -> PrefillQueueSnapshot:
        return PrefillQueueSnapshot(
            instance_index=state.index,
            remaining_chunks=sum(runtimes[index].remaining_chunks for index in state.qp),
            last_arrival_ns=state.last_arrival_ns,
        )

    def try_admit_request(request_index: int, now_ns: int) -> bool:
        runtime = runtimes[request_index]
        if runtime.estimated_arrival_ns is None:
            raise RuntimeError("request cannot be admitted before arrival")
        hbm_feasible_instances = kv_manager.request_hbm_feasible_instances(
            session_id=runtime.request.session_id,
            final_context_tokens=runtime.final_context_tokens,
        )
        if not any(hbm_feasible_instances):
            eventually_feasible = (
                kv_manager.request_hbm_eventually_feasible_instances(
                    session_id=runtime.request.session_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
            )
            if not any(eventually_feasible):
                raise ValueError(
                    f"request {runtime.request.request_id} final KV cannot fit on "
                    "any empty instance; "
                    f"final_context_tokens={runtime.final_context_tokens}"
                )
            return False

        snapshots = tuple(queue_snapshot(state) for state in instances)
        selected = select_prefill_instance(
            snapshots,
            hbm_feasible_instances,
        )
        selected_snapshot = snapshots[selected]
        admission_evictions = kv_manager.reserve_request_capacity(
            request_id=runtime.request.request_id,
            session_id=runtime.request.session_id,
            instance_index=selected,
            final_context_tokens=runtime.final_context_tokens,
            now_ns=now_ns,
        )
        runtime.prefill_instance_index = selected
        runtime.prefill_assignment_key = selected_snapshot.ordering_key
        runtime.admission_time_ns = now_ns
        (
            runtime.history_location_before,
            runtime.history_transfer,
            prepare_evictions,
        ) = kv_manager.prepare_prefill(
            session_id=runtime.request.session_id,
            target_instance_index=selected,
            history_tokens=runtime.history_tokens_before,
            trigger_request_id=runtime.request.request_id,
            reservation_request_id=runtime.request.request_id,
            now_ns=now_ns,
        )
        runtime.history_evictions = admission_evictions + prepare_evictions
        if runtime.request.turn_index > 0:
            if runtime.history_location_before is None:
                raise RuntimeError(
                    f"session {runtime.request.session_id} has no prior KV state"
                )
            runtime.history_source_instance_index = (
                runtime.history_location_before.instance_index
            )
            runtime.history_transfer_bytes = kv_cache_bytes_for_tokens(
                model,
                runtime.history_tokens_before,
            )
        instances[selected].qp.append(request_index)
        instances[selected].last_arrival_ns = now_ns
        return True

    def admit_waiting_requests(now_ns: int) -> None:
        blocked: deque[int] = deque()
        while pending_admissions:
            request_index = pending_admissions.popleft()
            if not try_admit_request(request_index, now_ns):
                blocked.append(request_index)
        pending_admissions.extend(blocked)

    def start_ready_iterations(now_ns: int) -> None:
        for state in instances:
            if state.busy or (not state.qp and not state.active_decode):
                continue
            prefill_index = state.qp[0] if state.qp else None
            decode_indexes = tuple(state.active_decode)
            chunk_tokens = 0
            if prefill_index is not None:
                runtime = runtimes[prefill_index]
                chunk_tokens = min(
                    p_chunk,
                    runtime.request.prefill_length - runtime.prompt_tokens_processed,
                )
                if runtime.prefill_start_ns is None:
                    runtime.prefill_start_ns = now_ns
            for request_index in decode_indexes:
                if runtimes[request_index].decode_start_ns is None:
                    runtimes[request_index].decode_start_ns = now_ns
            d_tokens = [runtimes[index].current_decode_token for index in decode_indexes]
            entry = lut.lookup(
                instance_size=topology.instance(state.index).size,
                p_chunk=p_chunk if prefill_index is not None else 0,
                d_batch=len(decode_indexes),
                d_token=max(d_tokens, default=0),
            )
            end_ns = now_ns + entry.iteration_time_ns
            if _ITERATION_STATS_HOOK is not None:
                _ITERATION_STATS_HOOK(entry, now_ns, end_ns)
            record = FaceIterationRecord(
                instance_index=state.index,
                iteration_index=state.iteration_count,
                start_ns=now_ns,
                end_ns=end_ns,
                prefill_request_id=(
                    None if prefill_index is None else runtimes[prefill_index].request.request_id
                ),
                prefill_chunk_tokens=chunk_tokens,
                decode_request_ids=tuple(
                    runtimes[index].request.request_id for index in decode_indexes
                ),
                lut_entry=entry,
            )
            if record_iterations:
                iterations.append(record)
            state.iteration_count += 1
            state.busy = True
            push_event(
                end_ns,
                0,
                "iteration_complete",
                (state.index, prefill_index, chunk_tokens, decode_indexes),
            )

    completed_requests = 0
    while event_heap:
        now_ns = event_heap[0][0]
        batch = []
        while event_heap and event_heap[0][0] == now_ns:
            batch.append(heapq.heappop(event_heap))
        batch.sort(key=lambda item: (item[1], item[2]))
        completed_now: list[int] = []
        retry_admissions = False

        for _, _, _, kind, payload in batch:
            if kind != "iteration_complete":
                continue
            state_index, prefill_index, chunk_tokens, decode_indexes = payload
            state = instances[state_index]
            state.busy = False

            if prefill_index is not None:
                runtime = runtimes[prefill_index]
                runtime.prompt_tokens_processed += chunk_tokens
                runtime.remaining_chunks -= 1
                if runtime.remaining_chunks < 0:
                    raise RuntimeError("prefill remaining chunk count became negative")
                if runtime.remaining_chunks == 0:
                    retry_admissions = True
                    if not state.qp or state.qp[0] != prefill_index:
                        raise RuntimeError("prefill FCFS queue order was corrupted")
                    state.qp.popleft()
                    runtime.prefill_complete_ns = now_ns
                    runtime.prefill_evictions = kv_manager.expand_prefill(
                        session_id=runtime.request.session_id,
                        instance_index=state_index,
                        context_tokens=runtime.prefill_context_tokens,
                        trigger_request_id=runtime.request.request_id,
                        reservation_request_id=runtime.request.request_id,
                        now_ns=now_ns,
                    )
                    has_prefill = [bool(instance.qp) for instance in instances]
                    active_tokens = [
                        [runtimes[index].current_decode_token for index in instance.active_decode]
                        for instance in instances
                    ]
                    selected, costs = select_decode_instance(
                        topology=topology,
                        graph=graph,
                        lut=lut,
                        fixed_p_chunk=p_chunk,
                        prefill_instance_index=state_index,
                        has_prefill_work=has_prefill,
                        decode_token_lengths=active_tokens,
                        new_request_token_length=runtime.prefill_context_tokens,
                        hbm_feasible_instances=(
                            kv_manager.decode_hbm_feasible_instances(
                                session_id=runtime.request.session_id,
                                final_context_tokens=runtime.final_context_tokens,
                                reservation_request_id=runtime.request.request_id,
                            )
                        ),
                    )
                    runtime.decode_instance_index = selected
                    runtime.decode_candidates = costs
                    reservation_move_evictions = (
                        kv_manager.move_request_capacity_reservation(
                            request_id=runtime.request.request_id,
                            target_instance_index=selected,
                            now_ns=now_ns,
                        )
                    )
                    (
                        runtime.prefill_decode_transfer,
                        decode_move_evictions,
                    ) = kv_manager.move_prefill_to_decode(
                        session_id=runtime.request.session_id,
                        target_instance_index=selected,
                        trigger_request_id=runtime.request.request_id,
                        reservation_request_id=runtime.request.request_id,
                        now_ns=now_ns,
                    )
                    decode_growth_evictions = kv_manager.expand_decode(
                        session_id=runtime.request.session_id,
                        instance_index=selected,
                        final_context_tokens=runtime.final_context_tokens,
                        trigger_request_id=runtime.request.request_id,
                        reservation_request_id=runtime.request.request_id,
                        now_ns=now_ns,
                    )
                    runtime.decode_evictions = (
                        reservation_move_evictions
                        + decode_move_evictions
                        + decode_growth_evictions
                    )
                    runtime.kv_allocation = kv_manager.allocation_for_session(
                        session_id=runtime.request.session_id,
                        request_id=runtime.request.request_id,
                    )
                    kv_manager.release_request_capacity_reservation(
                        runtime.request.request_id
                    )
                    instances[selected].active_decode.append(prefill_index)

            for request_index in decode_indexes:
                runtime = runtimes[request_index]
                runtime.decode_steps_remaining -= 1
                runtime.current_decode_token += 1
                if runtime.decode_steps_remaining < 0:
                    raise RuntimeError("decode remaining step count became negative")
                if runtime.decode_steps_remaining == 0:
                    retry_admissions = True
                    if request_index not in state.active_decode:
                        raise RuntimeError("decode queue membership was corrupted")
                    state.active_decode.remove(request_index)
                    runtime.completion_ns = now_ns
                    completed_requests += 1
                    completed_now.append(request_index)
                    following = next_request[request_index]
                    if following is not None:
                        next_runtime = runtimes[following]
                        interval = next_runtime.request.inter_request_interval_ns
                        if interval is None:
                            raise RuntimeError("validated later request lost its interval")
                        push_event(now_ns + interval, 1, "arrival", following)

        completion_order = sorted(
            completed_now,
            key=lambda index: (
                runtimes[index].request.session_id,
                runtimes[index].request.request_id,
                runtimes[index].request.queue_index,
            ),
        )
        for request_index in completion_order:
            kv_manager.mark_complete(
                runtimes[request_index].request.session_id,
                now_ns,
            )
        for request_index in completion_order:
            runtime = runtimes[request_index]
            if runtime.decode_instance_index is None:
                raise RuntimeError("completed request has no Decode instance")
            (
                runtime.completion_evictions,
                runtime.reserve_unmet_ranks,
            ) = kv_manager.enforce_reserve(
                instance_index=runtime.decode_instance_index,
                trigger_request_id=runtime.request.request_id,
                now_ns=now_ns,
            )
        completion_hbm_snapshot = kv_manager.hbm_snapshots()
        for request_index in completion_order:
            runtime = runtimes[request_index]
            completion_snapshot = kv_manager.session_snapshot(
                runtime.request.session_id
            )
            runtime.kv_location_after_completion = completion_snapshot.location
            runtime.kv_instance_after_completion = (
                completion_snapshot.instance_index
            )
            runtime.hbm_after_completion = completion_hbm_snapshot

        for _, _, _, kind, payload in batch:
            if kind != "arrival":
                continue
            request_index = int(payload)
            runtime = runtimes[request_index]
            if runtime.estimated_arrival_ns is not None:
                raise RuntimeError("request arrival was delivered more than once")
            runtime.estimated_arrival_ns = now_ns
            pending_admissions.append(request_index)
            retry_admissions = True

        if retry_admissions:
            admit_waiting_requests(now_ns)
        start_ready_iterations(now_ns)

    if pending_admissions:
        pending_ids = [
            runtimes[index].request.request_id for index in pending_admissions
        ]
        raise RuntimeError(
            f"FACE planning ended with blocked HBM admissions: {pending_ids[:5]}"
        )
    if completed_requests != len(runtimes):
        raise RuntimeError(
            f"FACE planning stopped with {completed_requests}/{len(runtimes)} requests complete"
        )
    if any(state.busy or state.qp or state.active_decode for state in instances):
        raise RuntimeError("FACE planning ended with non-idle instance state")

    plans: list[FaceRequestPlan] = []
    for runtime in runtimes:
        required = {
            "estimated_arrival_ns": runtime.estimated_arrival_ns,
            "admission_time_ns": runtime.admission_time_ns,
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": runtime.prefill_assignment_key,
            "prefill_start_ns": runtime.prefill_start_ns,
            "prefill_complete_ns": runtime.prefill_complete_ns,
            "decode_instance_index": runtime.decode_instance_index,
            "decode_start_ns": runtime.decode_start_ns,
            "completion_ns": runtime.completion_ns,
            "kv_allocation": runtime.kv_allocation,
            "prefill_decode_transfer": runtime.prefill_decode_transfer,
            "kv_location_after_completion": (
                runtime.kv_location_after_completion
            ),
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                f"request {runtime.request.request_id} is missing plan fields: {missing}"
            )
        plans.append(
            FaceRequestPlan(
                queue_index=runtime.request.queue_index,
                session_id=runtime.request.session_id,
                turn_index=runtime.request.turn_index,
                request_id=runtime.request.request_id,
                history_tokens_before=runtime.history_tokens_before,
                prefill_context_tokens=runtime.prefill_context_tokens,
                final_context_tokens=runtime.final_context_tokens,
                estimated_arrival_ns=int(runtime.estimated_arrival_ns),
                admission_time_ns=int(runtime.admission_time_ns),
                hbm_wait_ns=(
                    int(runtime.admission_time_ns)
                    - int(runtime.estimated_arrival_ns)
                ),
                prefill_instance_index=int(runtime.prefill_instance_index),
                prefill_assignment_key=tuple(runtime.prefill_assignment_key),
                prefill_start_ns=int(runtime.prefill_start_ns),
                prefill_complete_ns=int(runtime.prefill_complete_ns),
                decode_instance_index=int(runtime.decode_instance_index),
                decode_candidates=runtime.decode_candidates,
                decode_start_ns=int(runtime.decode_start_ns),
                completion_ns=int(runtime.completion_ns),
                kv_allocation=runtime.kv_allocation,
                history_source_instance_index=runtime.history_source_instance_index,
                history_transfer_bytes=runtime.history_transfer_bytes,
                history_location_before=runtime.history_location_before,
                history_transfer=runtime.history_transfer,
                history_evictions=runtime.history_evictions,
                prefill_evictions=runtime.prefill_evictions,
                prefill_decode_transfer=runtime.prefill_decode_transfer,
                decode_evictions=runtime.decode_evictions,
                completion_evictions=runtime.completion_evictions,
                kv_location_after_completion=runtime.kv_location_after_completion,
                kv_instance_after_completion=runtime.kv_instance_after_completion,
                hbm_after_completion=runtime.hbm_after_completion,
                reserve_unmet_ranks=runtime.reserve_unmet_ranks,
            )
        )

    edge_weights = tuple(
        (source, target, weight)
        for (source, target), weight in sorted(graph.weights.items())
    )
    return FacePlan(
        p_chunk=p_chunk,
        topology=topology,
        lut=lut,
        requests=tuple(sorted(plans, key=lambda plan: plan.queue_index)),
        iterations=tuple(
            sorted(
                iterations,
                key=lambda item: (item.start_ns, item.instance_index, item.iteration_index),
            )
        ),
        iterations_recorded=record_iterations,
        final_edge_weights=edge_weights,
        final_remaining_capacity_bytes=(
            kv_manager.instance_remaining_capacity_totals()
        ),
        edge_ranks=kv_manager.edge_ranks,
        reserve_context_tokens=reserve_context_tokens,
        final_hbm_states=kv_manager.hbm_snapshots(),
        final_session_states=kv_manager.session_snapshots(),
    )
