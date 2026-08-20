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
from typing import Iterable, Optional, Sequence

from session_kv_manager import (
    EVICTED,
    LOCAL_HIT,
    NOC_MIGRATE,
    NO_HISTORY,
    RECOMPUTE,
    EvictionRecord,
    KVCacheEvent,
    KVTransfer,
    NodeHBMSnapshot,
    SessionKVCacheManager,
    SessionKVSnapshot,
    attention_heads_by_tp_rank,
    kv_cache_shard_bytes_for_tokens,
    model_weight_shard_bytes_by_tp_rank,
)

# Optional read-only streaming planner-LUT statistics hook (doc sec.8.8).
# The trace generator installs it around plan_face_requests(); it observes
# every planning iteration's LUT lookup and never feeds back into scheduling.
_ITERATION_STATS_HOOK = None


def set_iteration_stats_hook(hook) -> None:
    """Install (or clear, with ``None``) the planner iteration stats hook."""

    global _ITERATION_STATS_HOOK
    _ITERATION_STATS_HOOK = hook


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
    # The full workload can observe many transient batch sizes.  Timing is
    # analytical, so retaining every observed tuple offers little value and
    # violates the bounded-memory requirement for plan generation.
    _LAZY_CACHE_MAX_ENTRIES = 8_192

    def __init__(self, entries: Iterable[FaceLutEntry]) -> None:
        self._entries = tuple(entries)
        if not self._entries:
            raise ValueError("FACE LUT must contain at least one entry")
        index: dict[tuple[int, int, int], list[FaceLutEntry]] = {}
        seen: set[tuple[int, int, int, int]] = set()
        for entry in self._entries:
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
        # Small tests and callers that construct an explicit LUT retain the
        # original eager behaviour.  ``build`` below opts into a compact,
        # analytical cache so the full three-minute workload does not create
        # O(request_count * token_bins) Python objects before planning starts.
        self._lazy_hardware: Optional[FaceHardware] = None
        self._lazy_model: Optional[FaceModel] = None
        self._lazy_instance_sizes: frozenset[int] = frozenset()
        self._lazy_p_chunk: Optional[int] = None
        self._lazy_request_count: Optional[int] = None
        self._lazy_token_bins: tuple[int, ...] = ()
        self._lazy_entries: dict[tuple[int, int, int, int], FaceLutEntry] = {}

    @property
    def entries(self) -> tuple[FaceLutEntry, ...]:
        """Materialized timing rows, including entries demanded by planning.

        This intentionally exposes the same public shape as the previous
        eager LUT.  In full-workload mode it contains just the seed rows and
        lookup combinations actually used by the event loop.
        """

        if not self._lazy_entries:
            return self._entries
        return tuple(
            (*self._entries, *(
                self._lazy_entries[key] for key in sorted(self._lazy_entries)
            ))
        )

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
        sizes = tuple(sorted(set(instance_sizes)))
        if not sizes:
            raise ValueError("instance_sizes must not be empty")
        # Preserve the two legal idle rows in the exported LUT.  Every
        # non-idle tuple is generated lazily on its first lookup.
        rows = [
            FaceLutEntry(
                instance_size=instance_size,
                p_chunk=chunk,
                d_batch=0,
                d_token=0,
                iteration_time_ns=estimate_iteration_time_ns(
                    hardware,
                    model,
                    instance_size=instance_size,
                    p_chunk=chunk,
                    d_batch=0,
                    d_token=0,
                ),
            )
            for instance_size in sizes
            for chunk in (0, p_chunk)
        ]
        lut = cls(rows)
        lut._lazy_hardware = hardware
        lut._lazy_model = model
        lut._lazy_instance_sizes = frozenset(sizes)
        lut._lazy_p_chunk = p_chunk
        lut._lazy_request_count = request_count
        lut._lazy_token_bins = _power_of_two_token_bins(max_d_token)
        return lut

    def lookup(
        self,
        *,
        instance_size: int,
        p_chunk: int,
        d_batch: int,
        d_token: int,
    ) -> FaceLutEntry:
        rows = self._index.get((instance_size, p_chunk, d_batch))
        if rows:
            return min(rows, key=lambda row: (abs(row.d_token - d_token), row.d_token))
        if (
            self._lazy_hardware is not None
            and self._lazy_model is not None
            and instance_size in self._lazy_instance_sizes
            and p_chunk in {0, self._lazy_p_chunk}
            and self._lazy_request_count is not None
            and 0 <= d_batch <= self._lazy_request_count
            and ((d_batch == 0 and d_token == 0) or (d_batch > 0 and d_token > 0))
        ):
            token_bin = min(
                self._lazy_token_bins,
                key=lambda token: (abs(token - d_token), token),
            )
            key = (instance_size, p_chunk, d_batch, token_bin)
            entry = self._lazy_entries.get(key)
            if entry is None:
                if len(self._lazy_entries) >= self._LAZY_CACHE_MAX_ENTRIES:
                    self._lazy_entries.clear()
                entry = FaceLutEntry(
                    instance_size=instance_size,
                    p_chunk=p_chunk,
                    d_batch=d_batch,
                    d_token=token_bin,
                    iteration_time_ns=estimate_iteration_time_ns(
                        self._lazy_hardware,
                        self._lazy_model,
                        instance_size=instance_size,
                        p_chunk=p_chunk,
                        d_batch=d_batch,
                        d_token=token_bin,
                    ),
                )
                self._lazy_entries[key] = entry
            return entry
        raise KeyError(
            "FACE LUT has no exact instance_size/p_chunk/d_batch match for "
            f"({instance_size}, {p_chunk}, {d_batch})"
        )

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


def select_prefill_instance(queues: Sequence[PrefillQueueSnapshot]) -> int:
    if not queues:
        raise ValueError("at least one prefill queue is required")
    return min(queues, key=lambda queue: queue.ordering_key).instance_index


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


class DecodeTieCounter:
    """decode 平局轮流裁决计数器（中-1 裁决 2026-08-20）：仅当候选集中
    ≥2 个 per_die_delta_ns 精确相等（真实平局）时前进；tied 集按
    instance_index 升序，取 counter % len(tied)。同一决策序列下确定可
    重放；不传入计数器时保持旧"最小 instance_index"行为。"""

    def __init__(self) -> None:
        self.value = 0


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
    tie_counter: Optional[DecodeTieCounter] = None,
) -> tuple[int, tuple[DecodeCandidateCost, ...]]:
    """平局规则（中-1 裁决 2026-08-20）：per_die_delta_ns 精确并列时按
    instance_index 升序轮流（传入 tie_counter 时）；否则选最小
    instance_index（旧行为）。"""
    if len(has_prefill_work) != len(topology.instances):
        raise ValueError("has_prefill_work length must match instances")
    if len(decode_token_lengths) != len(topology.instances):
        raise ValueError("decode_token_lengths length must match instances")
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
    min_delta = min(cost.per_die_delta_ns for cost in costs)
    tied = sorted(
        (cost for cost in costs if cost.per_die_delta_ns == min_delta),
        key=lambda cost: cost.instance_index,
    )
    if len(tied) > 1 and tie_counter is not None:
        selected = tied[tie_counter.value % len(tied)]
        tie_counter.value += 1
    else:
        selected = tied[0]
    return selected.instance_index, tuple(costs)


@dataclass(frozen=True)
class KVAllocationPiece:
    instance_index: int
    bytes: int
    weighted_distance: float
    path: tuple[int, ...]


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
    history_cache_state_before: str = "UNTRACKED"
    history_action: str = NO_HISTORY
    history_recompute_tokens: int = 0
    effective_prefill_tokens: int = 0
    history_transfer: Optional[KVTransfer] = None
    prefill_decode_transfer: Optional[KVTransfer] = None
    admission_evictions: tuple[EvictionRecord, ...] = ()
    decode_target_evictions: tuple[EvictionRecord, ...] = ()
    completion_evictions: tuple[EvictionRecord, ...] = ()
    kv_state_after_completion: str = "UNTRACKED"
    kv_instance_after_completion: Optional[int] = None
    hbm_before_request: tuple[NodeHBMSnapshot, ...] = ()
    hbm_after_completion: tuple[NodeHBMSnapshot, ...] = ()


@dataclass(frozen=True)
class FacePlan:
    p_chunk: int
    topology: FaceTopology
    lut: FaceLut
    requests: tuple[FaceRequestPlan, ...]
    iterations: tuple[FaceIterationRecord, ...]
    final_edge_weights: tuple[tuple[int, int, int], ...]
    final_remaining_capacity_bytes: tuple[int, ...]
    kv_cache_policy: str = "legacy"
    reserve_context_tokens: int = 0
    kv_events: tuple[KVCacheEvent, ...] = ()
    final_hbm_snapshots: tuple[NodeHBMSnapshot, ...] = ()
    final_session_snapshots: tuple[SessionKVSnapshot, ...] = ()
    planning_iteration_count: int = 0


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
    history_cache_state_before: str = "UNTRACKED"
    history_action: str = NO_HISTORY
    history_recompute_tokens: int = 0
    history_transfer: Optional[KVTransfer] = None
    prefill_decode_transfer: Optional[KVTransfer] = None
    admission_evictions: tuple[EvictionRecord, ...] = ()
    decode_target_evictions: tuple[EvictionRecord, ...] = ()
    completion_evictions: tuple[EvictionRecord, ...] = ()
    hbm_before_request: tuple[NodeHBMSnapshot, ...] = ()
    hbm_after_completion: tuple[NodeHBMSnapshot, ...] = ()
    kv_state_after_completion: str = "UNTRACKED"
    kv_instance_after_completion: Optional[int] = None
    history_recompute_processed: int = 0
    admitted_prefill: bool = False
    waiting_decode_admission: bool = False


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


def _plan_face_requests_legacy(
    *,
    hardware: FaceHardware,
    model: FaceModel,
    instance_specs: Sequence[FaceInstanceSpec],
    requests: Sequence[FaceRequest],
) -> FacePlan:
    topology = build_instances(hardware, instance_specs, require_equal_size=True)
    tp_degree = topology.instances[0].size
    # LLaMA 2 7B dimensions are partitioned exactly across the configured TP.
    # ET generation assigns whole heads and exact MLP/vocabulary slices
    # unevenly by rank whenever a model dimension is not divisible by TP.

    p_chunk = math.ceil(sum(request.prefill_length for request in requests) / len(requests))
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
    allocator = KVAllocator(
        topology,
        graph,
        model_weight_bytes=estimate_model_weight_bytes(model),
    )
    # 中-1 裁决（2026-08-20）：decode 平局按 instance_index 升序轮流；
    # 本函数对应一次 plan 调用，全程共享同一个计数器（仅真实平局前进）。
    decode_tie_counter = DecodeTieCounter()
    instances = [_InstanceRuntime(index=i) for i in range(len(topology.instances))]
    session_allocations: dict[str, KVAllocation] = {}

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

    def queue_snapshot(state: _InstanceRuntime) -> PrefillQueueSnapshot:
        return PrefillQueueSnapshot(
            instance_index=state.index,
            remaining_chunks=sum(runtimes[index].remaining_chunks for index in state.qp),
            last_arrival_ns=state.last_arrival_ns,
        )

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
                    if not state.qp or state.qp[0] != prefill_index:
                        raise RuntimeError("prefill FCFS queue order was corrupted")
                    state.qp.popleft()
                    runtime.prefill_complete_ns = now_ns
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
                        tie_counter=decode_tie_counter,
                    )
                    runtime.decode_instance_index = selected
                    runtime.decode_candidates = costs
                    candidate_indices = tuple(cost.instance_index for cost in costs)
                    runtime.kv_allocation = allocator.allocate(
                        request_id=runtime.request.request_id,
                        decode_instance_index=selected,
                        candidate_indices=candidate_indices,
                        total_bytes=kv_cache_bytes_for_tokens(
                            model, runtime.final_context_tokens
                        ),
                    )
                    session_allocations[runtime.request.session_id] = runtime.kv_allocation
                    instances[selected].active_decode.append(prefill_index)

            for request_index in decode_indexes:
                runtime = runtimes[request_index]
                runtime.decode_steps_remaining -= 1
                runtime.current_decode_token += 1
                if runtime.decode_steps_remaining < 0:
                    raise RuntimeError("decode remaining step count became negative")
                if runtime.decode_steps_remaining == 0:
                    if request_index not in state.active_decode:
                        raise RuntimeError("decode queue membership was corrupted")
                    state.active_decode.remove(request_index)
                    runtime.completion_ns = now_ns
                    completed_requests += 1
                    following = next_request[request_index]
                    if following is not None:
                        next_runtime = runtimes[following]
                        interval = next_runtime.request.inter_request_interval_ns
                        if interval is None:
                            raise RuntimeError("validated later request lost its interval")
                        push_event(now_ns + interval, 1, "arrival", following)

        for _, _, _, kind, payload in batch:
            if kind != "arrival":
                continue
            request_index = int(payload)
            runtime = runtimes[request_index]
            runtime.estimated_arrival_ns = now_ns
            previous_allocation = session_allocations.pop(runtime.request.session_id, None)
            if runtime.request.turn_index > 0:
                if previous_allocation is None:
                    raise RuntimeError(
                        f"session {runtime.request.session_id} has no prior KV allocation"
                    )
                runtime.history_source_instance_index = (
                    previous_allocation.decode_instance_index
                )
                runtime.history_transfer_bytes = kv_cache_bytes_for_tokens(
                    model, runtime.history_tokens_before
                )
                allocator.release(previous_allocation)

            snapshots = tuple(queue_snapshot(state) for state in instances)
            selected = select_prefill_instance(snapshots)
            selected_snapshot = snapshots[selected]
            runtime.prefill_instance_index = selected
            runtime.prefill_assignment_key = selected_snapshot.ordering_key
            instances[selected].qp.append(request_index)
            instances[selected].last_arrival_ns = now_ns

        start_ready_iterations(now_ns)

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
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": runtime.prefill_assignment_key,
            "prefill_start_ns": runtime.prefill_start_ns,
            "prefill_complete_ns": runtime.prefill_complete_ns,
            "decode_instance_index": runtime.decode_instance_index,
            "decode_start_ns": runtime.decode_start_ns,
            "completion_ns": runtime.completion_ns,
            "kv_allocation": runtime.kv_allocation,
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
        final_edge_weights=edge_weights,
        final_remaining_capacity_bytes=tuple(allocator.remaining_capacity),
    )


def _plan_face_session_lru_recompute(
    *,
    hardware: FaceHardware,
    model: FaceModel,
    instance_specs: Sequence[FaceInstanceSpec],
    requests: Sequence[FaceRequest],
    p_chunk: int,
    reserve_context_tokens: int,
    record_planning_iterations: bool,
) -> FacePlan:
    """Plan FACE without allowing KV state to influence its mappings.

    The original planner remains available above as a legacy reference.  This
    branch keeps its queue/LUT/Instance_map decisions but replaces the
    multi-instance allocator with the session state machine required by the
    three-minute workload.
    """

    if isinstance(p_chunk, bool) or not isinstance(p_chunk, int) or p_chunk <= 0:
        raise ValueError("p_chunk must be a positive integer")
    topology = build_instances(hardware, instance_specs, require_equal_size=True)
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
    kv_manager = SessionKVCacheManager(
        topology,
        model,
        reserve_context_tokens=reserve_context_tokens,
    )
    # 中-1 裁决（2026-08-20）：decode 平局按 instance_index 升序轮流；
    # 本函数对应一次 plan 调用，全程共享同一个计数器（仅真实平局前进）。
    decode_tie_counter = DecodeTieCounter()
    instances = [_InstanceRuntime(index=index) for index in range(len(topology.instances))]

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
    iteration_count = 0
    # A decode handoff is rare relative to decode iterations.  Keeping an
    # explicit pending queue avoids rescanning every one of the 9,179 runtime
    # records after each iteration of the full workload.
    waiting_decode_admissions: dict[int, deque[int]] = {
        instance.index: deque() for instance in topology.instances
    }
    # Failed admissions are retried only after the HBM state could have
    # changed.  Without this epoch a head-of-line request blocked by active
    # KV would emit the same blocked/deferred event on every Decode token.
    capacity_epoch = [0 for _ in topology.instances]
    prefill_attempt_epoch: dict[int, int] = {}
    decode_admission_epoch = [-1 for _ in topology.instances]
    decode_admission_dirty: set[int] = set()

    def note_capacity_change(*instance_indexes: Optional[int]) -> None:
        """Mark only the TP instances whose admission state may have changed."""

        for instance_index in set(instance_indexes):
            if instance_index is not None:
                capacity_epoch[instance_index] += 1

    def queue_snapshot(state: _InstanceRuntime) -> PrefillQueueSnapshot:
        return PrefillQueueSnapshot(
            instance_index=state.index,
            remaining_chunks=sum(runtimes[index].remaining_chunks for index in state.qp),
            last_arrival_ns=state.last_arrival_ns,
        )

    def history_transfer_for(runtime: _RequestRuntime) -> Optional[KVTransfer]:
        decision = runtime.history_transfer
        return decision

    def try_admit_prefill(request_index: int, now_ns: int) -> bool:
        runtime = runtimes[request_index]
        if runtime.admitted_prefill:
            return True
        if runtime.prefill_instance_index is None:
            raise RuntimeError("prefill admission lost its fixed mapping")
        target_instance = runtime.prefill_instance_index
        if prefill_attempt_epoch.get(request_index) == capacity_epoch[target_instance]:
            return False
        prefill_attempt_epoch[request_index] = capacity_epoch[target_instance]
        before_snapshot = kv_manager.session_snapshot(runtime.request.session_id)
        runtime.history_cache_state_before = (
            "ABSENT" if before_snapshot is None else before_snapshot.state
        )
        runtime.hbm_before_request = kv_manager.hbm_snapshots(runtime.prefill_instance_index)
        decision = kv_manager.prepare_history(
            runtime.request.session_id,
            runtime.prefill_instance_index,
            runtime.history_tokens_before,
            now_ns,
            runtime.request.request_id,
            required_context_tokens=runtime.prefill_context_tokens,
        )
        runtime.admission_evictions = decision.evictions
        if decision.admission_blocked:
            if decision.evictions:
                note_capacity_change(
                    target_instance,
                    *(record.victim_instance_index for record in decision.evictions),
                )
            return False
        runtime.history_action = decision.action
        runtime.history_source_instance_index = decision.source_instance_index
        runtime.history_transfer_bytes = decision.transfer_bytes
        runtime.history_recompute_tokens = decision.recompute_tokens
        if decision.action == NOC_MIGRATE:
            runtime.history_transfer = KVTransfer(
                action=NOC_MIGRATE,
                phase="history",
                reason="history_other_instance",
                session_id=runtime.request.session_id,
                trigger_request_id=runtime.request.request_id,
                source_instance_index=decision.source_instance_index,
                target_instance_index=runtime.prefill_instance_index,
                history_tokens=runtime.history_tokens_before,
                total_bytes=decision.transfer_bytes,
                shards=decision.transfer_shards,
            )
        growth = kv_manager.grow_prefill(
            runtime.request.session_id,
            runtime.prefill_context_tokens,
            now_ns,
            runtime.request.request_id,
        )
        if not growth.admitted:
            # prepare_history preflights the complete prefill state, so a
            # failure here would leave an active cache with no runnable work.
            raise RuntimeError("prefill growth lost its successful admission reservation")
        runtime.admission_evictions = tuple((*runtime.admission_evictions, *growth.evictions))
        runtime.history_recompute_processed = 0
        runtime.remaining_chunks = (
            math.ceil(runtime.history_recompute_tokens / p_chunk)
            + math.ceil(runtime.request.prefill_length / p_chunk)
        )
        runtime.admitted_prefill = True
        note_capacity_change(
            target_instance,
            decision.source_instance_index,
            *(record.victim_instance_index for record in decision.evictions),
            *(record.victim_instance_index for record in growth.evictions),
        )
        return True

    def try_admit_waiting_decodes(now_ns: int) -> None:
        ready_targets = tuple(
            instance_index
            for instance_index, queue in sorted(waiting_decode_admissions.items())
            if queue
            and (
                instance_index in decode_admission_dirty
                or decode_admission_epoch[instance_index]
                != capacity_epoch[instance_index]
            )
        )
        for target_instance in ready_targets:
            decode_admission_epoch[target_instance] = capacity_epoch[target_instance]
            decode_admission_dirty.discard(target_instance)
            queue = waiting_decode_admissions[target_instance]
            pending_count = len(queue)
            for _ in range(pending_count):
                request_index = queue.popleft()
                runtime = runtimes[request_index]
                if not runtime.waiting_decode_admission:
                    continue
                if runtime.decode_instance_index != target_instance:
                    raise RuntimeError("decode admission target queue was corrupted")
                move = kv_manager.move_prefill_to_decode(
                    runtime.request.session_id,
                    target_instance,
                    now_ns,
                    runtime.request.request_id,
                    final_context_tokens=runtime.final_context_tokens,
                )
                runtime.decode_target_evictions = tuple(
                    (*runtime.decode_target_evictions, *move.evictions)
                )
                if move.admission_blocked:
                    if move.evictions:
                        note_capacity_change(
                            target_instance,
                            *(record.victim_instance_index for record in move.evictions),
                        )
                    queue.append(request_index)
                    continue
                runtime.prefill_decode_transfer = move.transfer
                growth = kv_manager.grow_decode(
                    runtime.request.session_id,
                    runtime.final_context_tokens,
                    now_ns,
                    runtime.request.request_id,
                )
                if not growth.admitted:
                    raise RuntimeError("decode growth lost its successful target admission")
                runtime.decode_target_evictions = tuple(
                    (*runtime.decode_target_evictions, *growth.evictions)
                )
                instances[target_instance].active_decode.append(request_index)
                runtime.waiting_decode_admission = False
                note_capacity_change(
                    target_instance,
                    move.source_instance_index,
                    *(record.victim_instance_index for record in move.evictions),
                    *(record.victim_instance_index for record in growth.evictions),
                )

    def start_ready_iterations(now_ns: int) -> None:
        nonlocal iteration_count
        try_admit_waiting_decodes(now_ns)
        for state in instances:
            if state.busy or (not state.qp and not state.active_decode):
                continue
            prefill_index: Optional[int] = state.qp[0] if state.qp else None
            if prefill_index is not None and not try_admit_prefill(prefill_index, now_ns):
                prefill_index = None
            decode_indexes = tuple(state.active_decode)
            if prefill_index is None and not decode_indexes:
                continue
            chunk_tokens = 0
            history_chunk = False
            if prefill_index is not None:
                runtime = runtimes[prefill_index]
                history_remaining = (
                    runtime.history_recompute_tokens - runtime.history_recompute_processed
                )
                if history_remaining > 0:
                    chunk_tokens = min(p_chunk, history_remaining)
                    history_chunk = True
                else:
                    chunk_tokens = min(
                        p_chunk,
                        runtime.request.prefill_length - runtime.prompt_tokens_processed,
                    )
                if chunk_tokens <= 0:
                    raise RuntimeError("Prefill has a non-positive chunk")
                if runtime.prefill_start_ns is None:
                    runtime.prefill_start_ns = now_ns
            for request_index in decode_indexes:
                if runtimes[request_index].decode_start_ns is None:
                    runtimes[request_index].decode_start_ns = now_ns
            entry = lut.lookup(
                instance_size=topology.instance(state.index).size,
                p_chunk=p_chunk if prefill_index is not None else 0,
                d_batch=len(decode_indexes),
                d_token=max(
                    (runtimes[index].current_decode_token for index in decode_indexes),
                    default=0,
                ),
            )
            end_ns = now_ns + entry.iteration_time_ns
            if _ITERATION_STATS_HOOK is not None:
                _ITERATION_STATS_HOOK(entry, now_ns, end_ns)
            if record_planning_iterations:
                iterations.append(
                    FaceIterationRecord(
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
                )
            state.iteration_count += 1
            iteration_count += 1
            state.busy = True
            push_event(
                end_ns,
                0,
                "iteration_complete",
                (state.index, prefill_index, chunk_tokens, history_chunk, decode_indexes),
            )

    completed_requests = 0
    while event_heap:
        now_ns = event_heap[0][0]
        batch: list[tuple[int, int, int, str, object]] = []
        while event_heap and event_heap[0][0] == now_ns:
            batch.append(heapq.heappop(event_heap))
        batch.sort(key=lambda item: (item[1], item[2]))

        for _, _, _, kind, payload in batch:
            if kind != "iteration_complete":
                continue
            state_index, prefill_index, chunk_tokens, history_chunk, decode_indexes = payload
            state = instances[int(state_index)]
            state.busy = False
            if prefill_index is not None:
                runtime = runtimes[int(prefill_index)]
                if bool(history_chunk):
                    runtime.history_recompute_processed += int(chunk_tokens)
                else:
                    runtime.prompt_tokens_processed += int(chunk_tokens)
                runtime.remaining_chunks -= 1
                if runtime.remaining_chunks < 0:
                    raise RuntimeError("prefill remaining chunk count became negative")
                if runtime.remaining_chunks == 0:
                    if not state.qp or state.qp[0] != prefill_index:
                        raise RuntimeError("prefill FCFS queue order was corrupted")
                    if runtime.history_recompute_processed != runtime.history_recompute_tokens:
                        raise RuntimeError("history recompute chunks did not cover the planned history")
                    if runtime.prompt_tokens_processed != runtime.request.prefill_length:
                        raise RuntimeError("Prefill chunks did not cover the current prompt")
                    state.qp.popleft()
                    runtime.prefill_complete_ns = now_ns
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
                        prefill_instance_index=state.index,
                        has_prefill_work=has_prefill,
                        decode_token_lengths=active_tokens,
                        new_request_token_length=runtime.prefill_context_tokens,
                        tie_counter=decode_tie_counter,
                    )
                    runtime.decode_instance_index = selected
                    runtime.decode_candidates = costs
                    runtime.waiting_decode_admission = True
                    waiting_decode_admissions[selected].append(int(prefill_index))
                    decode_admission_dirty.add(selected)
                    # Preserve the legacy planner's event ordering: a
                    # schedulable P→D handoff becomes visible to subsequent
                    # same-timestamp Prefill completions before they evaluate
                    # their unchanged FACE Decode mapping cost.  Capacity can
                    # still defer this handoff, but it never remaps it.
                    try_admit_waiting_decodes(now_ns)

            for request_index in decode_indexes:
                runtime = runtimes[int(request_index)]
                runtime.decode_steps_remaining -= 1
                runtime.current_decode_token += 1
                if runtime.decode_steps_remaining < 0:
                    raise RuntimeError("decode remaining step count became negative")
                if runtime.decode_steps_remaining == 0:
                    if request_index not in state.active_decode:
                        raise RuntimeError("decode queue membership was corrupted")
                    state.active_decode.remove(request_index)
                    runtime.completion_ns = now_ns
                    completed_requests += 1
                    runtime.completion_evictions = kv_manager.mark_complete(
                        runtime.request.session_id,
                        now_ns,
                        runtime.request.request_id,
                    )
                    note_capacity_change(
                        state.index,
                        *(record.victim_instance_index for record in runtime.completion_evictions),
                    )
                    completed_snapshot = kv_manager.session_snapshot(runtime.request.session_id)
                    if completed_snapshot is None:
                        raise RuntimeError("completed session disappeared from KV manager")
                    runtime.kv_state_after_completion = completed_snapshot.state
                    runtime.kv_instance_after_completion = completed_snapshot.instance_index
                    runtime.hbm_after_completion = kv_manager.hbm_snapshots(state.index)
                    following = next_request[int(request_index)]
                    if following is not None:
                        interval = runtimes[following].request.inter_request_interval_ns
                        if interval is None:
                            raise RuntimeError("validated later request lost its interval")
                        push_event(now_ns + interval, 1, "arrival", following)

        for _, _, _, kind, payload in batch:
            if kind != "arrival":
                continue
            request_index = int(payload)
            runtime = runtimes[request_index]
            runtime.estimated_arrival_ns = now_ns
            snapshots = tuple(queue_snapshot(state) for state in instances)
            selected = select_prefill_instance(snapshots)
            selected_snapshot = snapshots[selected]
            runtime.prefill_instance_index = selected
            runtime.prefill_assignment_key = selected_snapshot.ordering_key
            instances[selected].qp.append(request_index)
            instances[selected].last_arrival_ns = now_ns

        start_ready_iterations(now_ns)

    if completed_requests != len(runtimes):
        raise RuntimeError(
            f"FACE planning stopped with {completed_requests}/{len(runtimes)} requests complete"
        )
    if any(state.busy or state.qp or state.active_decode for state in instances):
        raise RuntimeError("FACE planning ended with non-idle instance state")
    kv_manager.assert_final_state()

    plans: list[FaceRequestPlan] = []
    for runtime in runtimes:
        required = {
            "estimated_arrival_ns": runtime.estimated_arrival_ns,
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": runtime.prefill_assignment_key,
            "prefill_start_ns": runtime.prefill_start_ns,
            "prefill_complete_ns": runtime.prefill_complete_ns,
            "decode_instance_index": runtime.decode_instance_index,
            "decode_start_ns": runtime.decode_start_ns,
            "completion_ns": runtime.completion_ns,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise RuntimeError(
                f"request {runtime.request.request_id} is missing plan fields: {missing}"
            )
        final_shards = kv_cache_shard_bytes_for_tokens(
            model, runtime.final_context_tokens, topology.instances[0].size
        )
        allocation = KVAllocation(
            request_id=runtime.request.request_id,
            decode_instance_index=int(runtime.decode_instance_index),
            total_bytes=sum(final_shards),
            pieces=(
                KVAllocationPiece(
                    instance_index=int(runtime.decode_instance_index),
                    bytes=sum(final_shards),
                    weighted_distance=0.0,
                    path=(),
                ),
            ),
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
                prefill_instance_index=int(runtime.prefill_instance_index),
                prefill_assignment_key=tuple(runtime.prefill_assignment_key),
                prefill_start_ns=int(runtime.prefill_start_ns),
                prefill_complete_ns=int(runtime.prefill_complete_ns),
                decode_instance_index=int(runtime.decode_instance_index),
                decode_candidates=runtime.decode_candidates,
                decode_start_ns=int(runtime.decode_start_ns),
                completion_ns=int(runtime.completion_ns),
                kv_allocation=allocation,
                history_source_instance_index=runtime.history_source_instance_index,
                history_transfer_bytes=runtime.history_transfer_bytes,
                history_cache_state_before=runtime.history_cache_state_before,
                history_action=runtime.history_action,
                history_recompute_tokens=runtime.history_recompute_tokens,
                effective_prefill_tokens=(
                    runtime.request.prefill_length + runtime.history_recompute_tokens
                ),
                history_transfer=history_transfer_for(runtime),
                prefill_decode_transfer=runtime.prefill_decode_transfer,
                admission_evictions=runtime.admission_evictions,
                decode_target_evictions=runtime.decode_target_evictions,
                completion_evictions=runtime.completion_evictions,
                kv_state_after_completion=runtime.kv_state_after_completion,
                kv_instance_after_completion=runtime.kv_instance_after_completion,
                hbm_before_request=runtime.hbm_before_request,
                hbm_after_completion=runtime.hbm_after_completion,
            )
        )
    edge_weights = tuple(
        (source, target, weight)
        for (source, target), weight in sorted(graph.weights.items())
    )
    final_sessions = tuple(
        snapshot
        for session_id in kv_manager.session_ids
        if (snapshot := kv_manager.session_snapshot(session_id)) is not None
    )
    return FacePlan(
        p_chunk=p_chunk,
        topology=topology,
        lut=lut,
        requests=tuple(sorted(plans, key=lambda item: item.queue_index)),
        iterations=tuple(iterations),
        final_edge_weights=edge_weights,
        final_remaining_capacity_bytes=tuple(
            sum(snapshot.remaining_bytes for snapshot in kv_manager.hbm_snapshots(instance.index))
            for instance in topology.instances
        ),
        kv_cache_policy="session_lru_recompute",
        reserve_context_tokens=reserve_context_tokens,
        kv_events=kv_manager.events,
        final_hbm_snapshots=kv_manager.hbm_snapshots(),
        final_session_snapshots=final_sessions,
        planning_iteration_count=iteration_count,
    )


def plan_face_requests(
    *,
    hardware: FaceHardware,
    model: FaceModel,
    instance_specs: Sequence[FaceInstanceSpec],
    requests: Sequence[FaceRequest],
    p_chunk: Optional[int] = None,
    kv_cache_policy: str = "legacy",
    reserve_context_tokens: int = 1_000_000,
    record_planning_iterations: bool = True,
) -> FacePlan:
    """Plan FACE requests under the requested cache-management policy."""

    if kv_cache_policy == "legacy":
        if p_chunk is not None:
            # The old planner derives its chunk size from the workload.  Keep
            # this branch stable for its existing regression fixtures.
            raise ValueError("legacy FACE planning does not accept an explicit p_chunk")
        return _plan_face_requests_legacy(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=requests,
        )
    if kv_cache_policy != "session_lru_recompute":
        raise ValueError(f"unsupported kv_cache_policy: {kv_cache_policy!r}")
    if p_chunk is None:
        raise ValueError("session_lru_recompute requires an explicit p_chunk")
    return _plan_face_session_lru_recompute(
        hardware=hardware,
        model=model,
        instance_specs=instance_specs,
        requests=requests,
        p_chunk=p_chunk,
        reserve_context_tokens=reserve_context_tokens,
        record_planning_iterations=record_planning_iterations,
    )
