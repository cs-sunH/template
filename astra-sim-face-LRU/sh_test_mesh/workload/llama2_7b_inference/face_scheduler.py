#!/usr/bin/env python3
"""Pure FACE request-mapping model shared by the online scheduling path.

The FACE paper describes a live host scheduler.  This module captures its
deterministic queue ordering, schedulable-instance range, direct Roofline
estimates, per-die incremental decode cost, and local-first KV allocation
without depending on protobuf or ASTRA-sim internals.  The online scheduler
reuses these semantics while timing and progress come from the running
simulator.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Optional, Sequence


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
            "direct KV shard pairing requires equal-size instances"
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
    # batched portion.  A small fixed controller term keeps non-idle estimates
    # positive and deterministic.
    total_seconds = linear_seconds + max(prefill_seconds, decode_seconds)
    return max(1, math.ceil(total_seconds * 1e9) + hardware.d2d_latency_ns)


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
    current_roofline: FaceRooflineEstimate
    updated_roofline: FaceRooflineEstimate
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
    hardware: FaceHardware,
    model: FaceModel,
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


