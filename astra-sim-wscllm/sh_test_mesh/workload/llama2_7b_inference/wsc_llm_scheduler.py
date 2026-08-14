#!/usr/bin/env python3
"""Deterministic WSC-LLM PD-disaggregated request-mapping planner.

WSC-LLM describes a live central scheduler, whereas ASTRA-sim consumes static
Chakra ET DAGs.  This module therefore performs a trace-generation-time event
planning pass.  Whole configured instances are the placement unit: every
instance keeps its existing six-rank TP communicator and is dedicated to
either Prefill or Decode for the complete run.
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


PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
PHASE_ROLES = frozenset((PREFILL_ROLE, DECODE_ROLE))


# Optional read-only streaming planner-LUT statistics hook (doc sec.8.8).
# The trace generator installs it around plan_wsc_llm_requests(); it observes
# every planning iteration's LUT lookup and never feeds back into scheduling.
_ITERATION_STATS_HOOK = None


def set_iteration_stats_hook(hook) -> None:
    """Install (or clear, with ``None``) the planner iteration stats hook."""

    global _ITERATION_STATS_HOOK
    _ITERATION_STATS_HOOK = hook


@dataclass(frozen=True)
class WscLlmHardware:
    mesh_rows: int
    mesh_cols: int
    local_hbm_capacity_bytes: int
    local_hbm_bandwidth_gbps: float
    d2d_bandwidth_gbps: float
    peak_perf_tflops: float
    d2d_latency_ns: int
    local_hbm_latency_ns: int
    label: str = "WSC-LLM"

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
    def d2d_to_hbm_bandwidth_ratio(self) -> float:
        """Expose the inherited hardware ratio as metadata, not a route bound."""

        return self.d2d_bandwidth_gbps / self.local_hbm_bandwidth_gbps


@dataclass(frozen=True)
class WscLlmModel:
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
class WscLlmInstanceSpec:
    name: str
    pg_name: str
    ranks: tuple[int, ...]
    phase_role: str


@dataclass(frozen=True)
class WscLlmInstance:
    index: int
    name: str
    pg_name: str
    ranks: tuple[int, ...]
    phase_role: str
    row_min: int
    row_max: int
    col_min: int
    col_max: int
    center_row: float
    center_col: float
    wafer_center_manhattan_distance: float

    @property
    def size(self) -> int:
        return len(self.ranks)

    @property
    def shape(self) -> tuple[int, int]:
        return (self.row_max - self.row_min + 1, self.col_max - self.col_min + 1)


@dataclass(frozen=True)
class WscLlmTopology:
    hardware: WscLlmHardware
    instances: tuple[WscLlmInstance, ...]
    adjacency: tuple[tuple[int, ...], ...]

    def instance(self, index: int) -> WscLlmInstance:
        return self.instances[index]

    def indices_for_role(self, phase_role: str) -> tuple[int, ...]:
        if phase_role not in PHASE_ROLES:
            raise ValueError(f"unsupported phase role: {phase_role!r}")
        return tuple(
            instance.index
            for instance in self.instances
            if instance.phase_role == phase_role
        )


def rank_coordinates(hardware: WscLlmHardware, rank: int) -> tuple[int, int]:
    if rank < 0 or rank >= hardware.npus_count:
        raise ValueError(f"rank {rank} is outside 0-{hardware.npus_count - 1}")
    return divmod(rank, hardware.mesh_cols)


def build_instances(
    hardware: WscLlmHardware,
    specs: Sequence[WscLlmInstanceSpec],
    *,
    require_equal_size: bool = True,
) -> WscLlmTopology:
    if not specs:
        raise ValueError("at least one WSC-LLM instance is required")

    all_ranks: list[int] = []
    instances: list[WscLlmInstance] = []
    rank_owner: dict[int, int] = {}
    pg_names: set[str] = set()
    wafer_center_row = (hardware.mesh_rows - 1) / 2.0
    wafer_center_col = (hardware.mesh_cols - 1) / 2.0

    for index, spec in enumerate(specs):
        if not spec.name:
            raise ValueError("instance name must not be empty")
        if not spec.pg_name or spec.pg_name == "0":
            raise ValueError("instance pg_name must be non-zero")
        if spec.pg_name in pg_names:
            raise ValueError(f"duplicate pg_name: {spec.pg_name}")
        pg_names.add(spec.pg_name)
        if spec.phase_role not in PHASE_ROLES:
            raise ValueError(
                f"instance {spec.name} phase_role must be prefill or decode"
            )
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
        center_row = (row_min + row_max) / 2.0
        center_col = (col_min + col_max) / 2.0
        instances.append(
            WscLlmInstance(
                index=index,
                name=spec.name,
                pg_name=spec.pg_name,
                ranks=tuple(spec.ranks),
                phase_role=spec.phase_role,
                row_min=row_min,
                row_max=row_max,
                col_min=col_min,
                col_max=col_max,
                center_row=center_row,
                center_col=center_col,
                wafer_center_manhattan_distance=(
                    abs(center_row - wafer_center_row)
                    + abs(center_col - wafer_center_col)
                ),
            )
        )

    if sorted(all_ranks) != list(range(hardware.npus_count)):
        raise ValueError("WSC-LLM instances must cover every configured NPU exactly once")
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

    prefill_instances = [
        instance for instance in instances if instance.phase_role == PREFILL_ROLE
    ]
    decode_instances = [
        instance for instance in instances if instance.phase_role == DECODE_ROLE
    ]
    if not prefill_instances or not decode_instances:
        raise ValueError("WSC-LLM requires at least one Prefill and one Decode instance")
    max_decode_distance = max(
        instance.wafer_center_manhattan_distance for instance in decode_instances
    )
    min_prefill_distance = min(
        instance.wafer_center_manhattan_distance for instance in prefill_instances
    )
    if max_decode_distance > min_prefill_distance + 1e-12:
        raise ValueError(
            "Decode instances must be center-prioritized: maximum Decode center "
            "distance exceeds minimum Prefill center distance"
        )

    return WscLlmTopology(
        hardware=hardware,
        instances=tuple(instances),
        adjacency=tuple(tuple(sorted(neighbors)) for neighbors in adjacency_sets),
    )


def estimate_model_weight_bytes(model: WscLlmModel) -> int:
    mlp_matrices = 3 if model.mlp_variant == "swiglu" else 2
    norm_elements = (
        2 * model.hidden_size
        if model.mlp_variant == "swiglu"
        else 4 * model.hidden_size
    )
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


def kv_cache_bytes_for_tokens(model: WscLlmModel, tokens: int) -> int:
    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    return 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem


class InstanceGraph:
    """Unweighted whole-instance adjacency used by WSC-LLM offline mapping."""

    def __init__(self, topology: WscLlmTopology) -> None:
        self.topology = topology

    @staticmethod
    def edge(source: int, target: int) -> tuple[int, int]:
        return (source, target) if source < target else (target, source)

    def _validate_index(self, index: int) -> None:
        if index < 0 or index >= len(self.topology.instances):
            raise ValueError(f"instance index is out of range: {index}")

    def all_shortest_paths(self, source: int, target: int) -> tuple[tuple[int, ...], ...]:
        self._validate_index(source)
        self._validate_index(target)
        if source == target:
            return ((source,),)

        distances = {source: 0}
        parents: dict[int, list[int]] = {source: []}
        queue = deque((source,))
        while queue:
            node = queue.popleft()
            next_distance = distances[node] + 1
            for neighbor in self.topology.adjacency[node]:
                previous_distance = distances.get(neighbor)
                if previous_distance is None:
                    distances[neighbor] = next_distance
                    parents[neighbor] = [node]
                    queue.append(neighbor)
                elif previous_distance == next_distance:
                    parents[neighbor].append(node)
        if target not in distances:
            raise ValueError(f"no instance path from {source} to {target}")

        cache: dict[int, tuple[tuple[int, ...], ...]] = {}

        def rebuild(node: int) -> tuple[tuple[int, ...], ...]:
            if node == source:
                return ((source,),)
            if node in cache:
                return cache[node]
            paths = tuple(
                sorted(
                    prefix + (node,)
                    for parent in sorted(parents[node])
                    for prefix in rebuild(parent)
                )
            )
            cache[node] = paths
            return paths

        return rebuild(target)

    def shortest_distance(self, source: int, target: int) -> int:
        return len(self.all_shortest_paths(source, target)[0]) - 1


@dataclass(frozen=True)
class StaticPdRoute:
    prefill_instance_index: int
    decode_instance_index: int
    path: tuple[int, ...]
    hop_count: int
    shared_edges: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StaticPdMapping:
    routes: tuple[StaticPdRoute, ...]
    total_hops: int
    shared_edge_occurrences: int
    edge_use_counts: tuple[tuple[int, int, int], ...]
    alpha: float
    adjusted_transfer_cost: float

    def route_for_prefill(self, prefill_instance_index: int) -> StaticPdRoute:
        for route in self.routes:
            if route.prefill_instance_index == prefill_instance_index:
                return route
        raise KeyError(f"no static Decode route for Prefill instance {prefill_instance_index}")

    def routes_for_decode(self, decode_instance_index: int) -> tuple[StaticPdRoute, ...]:
        routes = tuple(
            route
            for route in self.routes
            if route.decode_instance_index == decode_instance_index
        )
        if not routes:
            raise KeyError(f"no static Prefill routes target Decode instance {decode_instance_index}")
        return routes


def build_static_pd_mapping(
    topology: WscLlmTopology,
    *,
    alpha: float = 1.0,
) -> StaticPdMapping:
    """Choose deterministic nearest-Decode shortest routes for all Prefill instances."""

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or alpha < 1:
        raise ValueError("alpha must be a number greater than or equal to one")
    graph = InstanceGraph(topology)
    prefill_indices = topology.indices_for_role(PREFILL_ROLE)
    decode_indices = topology.indices_for_role(DECODE_ROLE)
    route_candidates: list[tuple[int, tuple[tuple[int, tuple[int, ...]], ...]]] = []

    for prefill_index in prefill_indices:
        decode_distances = {
            decode_index: graph.shortest_distance(prefill_index, decode_index)
            for decode_index in decode_indices
        }
        nearest_distance = min(decode_distances.values())
        candidates = tuple(
            sorted(
                (decode_index, path)
                for decode_index in decode_indices
                if decode_distances[decode_index] == nearest_distance
                for path in graph.all_shortest_paths(prefill_index, decode_index)
            )
        )
        route_candidates.append((prefill_index, candidates))

    best_objective: Optional[tuple[object, ...]] = None
    best_selection: Optional[tuple[tuple[int, int, tuple[int, ...]], ...]] = None

    def choose(
        position: int,
        selection: list[tuple[int, int, tuple[int, ...]]],
    ) -> None:
        nonlocal best_objective, best_selection
        if position == len(route_candidates):
            edge_counts: dict[tuple[int, int], int] = {}
            total_hops = 0
            for _, _, path in selection:
                total_hops += len(path) - 1
                for source, target in zip(path, path[1:]):
                    edge = graph.edge(source, target)
                    edge_counts[edge] = edge_counts.get(edge, 0) + 1
            shared_occurrences = sum(max(0, count - 1) for count in edge_counts.values())
            signature = tuple(selection)
            objective: tuple[object, ...] = (
                total_hops,
                shared_occurrences,
                signature,
            )
            if best_objective is None or objective < best_objective:
                best_objective = objective
                best_selection = signature
            return

        prefill_index, candidates = route_candidates[position]
        for decode_index, path in candidates:
            selection.append((prefill_index, decode_index, path))
            choose(position + 1, selection)
            selection.pop()

    choose(0, [])
    if best_selection is None:
        raise RuntimeError("failed to construct WSC-LLM static P-to-D mapping")

    edge_counts: dict[tuple[int, int], int] = {}
    for _, _, path in best_selection:
        for source, target in zip(path, path[1:]):
            edge = graph.edge(source, target)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    routes = tuple(
        StaticPdRoute(
            prefill_instance_index=prefill_index,
            decode_instance_index=decode_index,
            path=path,
            hop_count=len(path) - 1,
            shared_edges=tuple(
                graph.edge(source, target)
                for source, target in zip(path, path[1:])
                if edge_counts[graph.edge(source, target)] > 1
            ),
        )
        for prefill_index, decode_index, path in best_selection
    )
    total_hops = sum(route.hop_count for route in routes)
    shared_occurrences = sum(max(0, count - 1) for count in edge_counts.values())
    return StaticPdMapping(
        routes=routes,
        total_hops=total_hops,
        shared_edge_occurrences=shared_occurrences,
        edge_use_counts=tuple(
            (source, target, count)
            for (source, target), count in sorted(edge_counts.items())
        ),
        alpha=float(alpha),
        adjusted_transfer_cost=(
            total_hops + (float(alpha) - 1.0) * shared_occurrences
        ),
    )


@dataclass(frozen=True)
class WscLlmTimingEntry:
    instance_size: int
    phase_role: str
    p_chunk: int
    d_batch: int
    d_token: int
    iteration_time_ns: int
    source: str = "analytical_roofline_phase_timing"

    def __post_init__(self) -> None:
        if self.phase_role == PREFILL_ROLE:
            valid = self.p_chunk > 0 and self.d_batch == 0 and self.d_token == 0
        elif self.phase_role == DECODE_ROLE:
            valid = self.p_chunk == 0 and self.d_batch > 0 and self.d_token > 0
        else:
            valid = False
        if not valid:
            raise ValueError(
                "timing entries must contain a phase-exclusive Prefill or Decode workload"
            )


def _power_of_two_token_bins(max_token: int) -> tuple[int, ...]:
    if max_token <= 0:
        raise ValueError("max_token must be positive")
    bins: list[int] = []
    value = 128
    while value < max_token:
        bins.append(value)
        value *= 2
    bins.append(value)
    return tuple(bins)


def estimate_iteration_time_ns(
    hardware: WscLlmHardware,
    model: WscLlmModel,
    *,
    instance_size: int,
    p_chunk: int,
    d_batch: int,
    d_token: int,
) -> int:
    if instance_size <= 0 or p_chunk < 0 or d_batch < 0 or d_token < 0:
        raise ValueError("invalid timing workload parameters")
    if (p_chunk > 0) == (d_batch > 0):
        raise ValueError("timing workload must contain exactly one of Prefill or Decode")
    if d_batch > 0 and d_token <= 0:
        raise ValueError("Decode timing requires a positive token/KV length")

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

    phase_seconds = 0.0
    if p_chunk:
        attention_ops = layers * 4 * p_chunk * p_chunk * h
        attention_bytes = layers * p_chunk * p_chunk * bytes_per_elem * 4
        phase_seconds = max(
            attention_ops / aggregate_perf,
            local_hbm_latency_seconds + attention_bytes / aggregate_bw,
        )
    else:
        attention_ops = layers * 4 * d_batch * d_token * h
        attention_bytes = layers * 2 * d_batch * d_token * h * bytes_per_elem
        phase_seconds = max(
            attention_ops / aggregate_perf,
            local_hbm_latency_seconds + attention_bytes / aggregate_bw,
        )

    return max(
        1,
        math.ceil((linear_seconds + phase_seconds) * 1e9)
        + hardware.d2d_latency_ns,
    )


class WscLlmTimingLut:
    """Analytical phase timing only; it never participates in allocation."""

    _LAZY_CACHE_MAX_ENTRIES = 8_192

    def __init__(self, entries: Iterable[WscLlmTimingEntry]) -> None:
        self._entries = tuple(entries)
        if not self._entries:
            raise ValueError("WSC-LLM timing LUT must contain at least one entry")
        index: dict[tuple[str, int, int, int], list[WscLlmTimingEntry]] = {}
        seen: set[tuple[str, int, int, int, int]] = set()
        for entry in self._entries:
            key5 = (
                entry.phase_role,
                entry.instance_size,
                entry.p_chunk,
                entry.d_batch,
                entry.d_token,
            )
            if key5 in seen:
                raise ValueError(f"duplicate WSC-LLM timing entry: {key5}")
            seen.add(key5)
            index.setdefault(key5[:4], []).append(entry)
        self._index = {
            key: tuple(sorted(rows, key=lambda row: row.d_token))
            for key, rows in index.items()
        }
        # ``build`` enables a compact, analytical decode cache.  The prior
        # implementation eagerly emitted one Python object for every possible
        # batch-size/token-bin combination, which is prohibitive for 9,179
        # requests even though the event loop touches only a small subset.
        self._lazy_hardware: Optional[WscLlmHardware] = None
        self._lazy_model: Optional[WscLlmModel] = None
        self._lazy_instance_sizes: frozenset[int] = frozenset()
        self._lazy_request_count: Optional[int] = None
        self._lazy_token_bins: tuple[int, ...] = ()
        self._lazy_entries: dict[tuple[int, int, int], WscLlmTimingEntry] = {}

    @property
    def entries(self) -> tuple[WscLlmTimingEntry, ...]:
        """Materialized rows, including lazy entries used during planning."""

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
        hardware: WscLlmHardware,
        model: WscLlmModel,
        *,
        instance_sizes: Iterable[int],
        p_chunk: int,
        request_count: int,
        max_d_token: int,
    ) -> "WscLlmTimingLut":
        if p_chunk <= 0:
            raise ValueError("p_chunk must be positive")
        if request_count <= 0:
            raise ValueError("request_count must be positive")
        sizes = tuple(sorted(set(instance_sizes)))
        if not sizes:
            raise ValueError("instance_sizes must not be empty")
        rows = [
            WscLlmTimingEntry(
                instance_size=instance_size,
                phase_role=PREFILL_ROLE,
                p_chunk=p_chunk,
                d_batch=0,
                d_token=0,
                iteration_time_ns=estimate_iteration_time_ns(
                    hardware,
                    model,
                    instance_size=instance_size,
                    p_chunk=p_chunk,
                    d_batch=0,
                    d_token=0,
                ),
            )
            for instance_size in sizes
        ]
        lut = cls(rows)
        lut._lazy_hardware = hardware
        lut._lazy_model = model
        lut._lazy_instance_sizes = frozenset(sizes)
        lut._lazy_request_count = request_count
        lut._lazy_token_bins = _power_of_two_token_bins(max_d_token)
        return lut

    def lookup(
        self,
        *,
        phase_role: str,
        instance_size: int,
        p_chunk: int,
        d_batch: int,
        d_token: int,
    ) -> WscLlmTimingEntry:
        rows = self._index.get((phase_role, instance_size, p_chunk, d_batch))
        if rows:
            return min(rows, key=lambda row: (abs(row.d_token - d_token), row.d_token))
        if (
            self._lazy_hardware is not None
            and self._lazy_model is not None
            and phase_role == DECODE_ROLE
            and instance_size in self._lazy_instance_sizes
            and p_chunk == 0
            and self._lazy_request_count is not None
            and 1 <= d_batch <= self._lazy_request_count
            and d_token > 0
        ):
            token_bin = min(
                self._lazy_token_bins,
                key=lambda token: (abs(token - d_token), token),
            )
            key = (instance_size, d_batch, token_bin)
            entry = self._lazy_entries.get(key)
            if entry is None:
                if len(self._lazy_entries) >= self._LAZY_CACHE_MAX_ENTRIES:
                    self._lazy_entries.clear()
                entry = WscLlmTimingEntry(
                    instance_size=instance_size,
                    phase_role=DECODE_ROLE,
                    p_chunk=0,
                    d_batch=d_batch,
                    d_token=token_bin,
                    iteration_time_ns=estimate_iteration_time_ns(
                        self._lazy_hardware,
                        self._lazy_model,
                        instance_size=instance_size,
                        p_chunk=0,
                        d_batch=d_batch,
                        d_token=token_bin,
                    ),
                )
                self._lazy_entries[key] = entry
            return entry
        raise KeyError(
            "WSC-LLM timing LUT has no exact phase/instance_size/p_chunk/"
            f"d_batch match for ({phase_role}, {instance_size}, {p_chunk}, {d_batch})"
        )

    def export_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(
                output,
                fieldnames=(
                    "instance_size",
                    "phase_role",
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
                    row.phase_role,
                    row.p_chunk,
                    row.d_batch,
                    row.d_token,
                ),
            ):
                writer.writerow(entry.__dict__)


@dataclass(frozen=True)
class PrefillQueueSnapshot:
    instance_index: int
    request_count: int

    def __post_init__(self) -> None:
        if self.instance_index < 0 or self.request_count < 0:
            raise ValueError("Prefill queue snapshot values must be non-negative")

    @property
    def ordering_key(self) -> tuple[int, int]:
        return (self.request_count, self.instance_index)


def select_prefill_instance(queues: Sequence[PrefillQueueSnapshot]) -> int:
    if not queues:
        raise ValueError("at least one Prefill queue is required")
    return min(queues, key=lambda queue: queue.ordering_key).instance_index


@dataclass(frozen=True)
class KVAllocationPiece:
    instance_index: int
    bytes: int
    distance_to_decode: int
    location_priority: str
    path: tuple[int, ...]


@dataclass(frozen=True)
class KVAllocation:
    request_id: str
    prefill_instance_index: int
    decode_instance_index: int
    relevant_instance_indices: tuple[int, ...]
    static_route: tuple[int, ...]
    total_bytes: int
    pieces: tuple[KVAllocationPiece, ...]


class WscRelevantKvAllocator:
    """WSC Relevant(P,D) allocator at the user-required instance abstraction."""

    def __init__(
        self,
        topology: WscLlmTopology,
        static_mapping: StaticPdMapping,
        *,
        model_weight_bytes: int,
    ) -> None:
        self.topology = topology
        self.static_mapping = static_mapping
        self.model_weight_bytes = model_weight_bytes
        self.initial_capacity: list[int] = []
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
            self.initial_capacity.append(capacity)
        self.remaining_capacity = self.initial_capacity.copy()

    def _validate_route(self, route: StaticPdRoute) -> None:
        if not route.path or len(set(route.path)) != len(route.path):
            raise ValueError("static P-to-D route must be a non-empty simple path")
        if route.path[0] != route.prefill_instance_index:
            raise ValueError("static route must start at its Prefill instance")
        if route.path[-1] != route.decode_instance_index:
            raise ValueError("static route must end at its Decode instance")
        if self.topology.instance(route.prefill_instance_index).phase_role != PREFILL_ROLE:
            raise ValueError("static route source is not a Prefill instance")
        if self.topology.instance(route.decode_instance_index).phase_role != DECODE_ROLE:
            raise ValueError("static route destination is not a Decode instance")
        for source, target in zip(route.path, route.path[1:]):
            if target not in self.topology.adjacency[source]:
                raise ValueError(f"invalid static route edge: {source}->{target}")

    def _ordered_locations(
        self,
        route: StaticPdRoute,
    ) -> tuple[tuple[int, int, str, tuple[int, ...]], ...]:
        self._validate_route(route)
        last = len(route.path) - 1
        ordered: list[tuple[int, int, str, tuple[int, ...]]] = [
            (route.decode_instance_index, 0, "decode", (route.decode_instance_index,))
        ]
        seen = {route.decode_instance_index}

        # The selected request's own static path has strict priority, walking
        # from the Decode endpoint back to its source Prefill instance.
        for position in range(last - 1, -1, -1):
            instance_index = route.path[position]
            distance = last - position
            transfer_path = tuple(reversed(route.path[position:]))
            label = (
                "selected_prefill"
                if position == 0
                else "selected_path_intermediate"
            )
            ordered.append((instance_index, distance, label, transfer_path))
            seen.add(instance_index)

        # Relevant(P,D) then includes the rest of the same static Decode domain:
        # sibling Prefill instances and their selected paths.  This is fixed by
        # offline mapping and never widens to another Decode domain.
        sibling_candidates: dict[
            int, tuple[int, str, tuple[int, ...]]
        ] = {}
        for sibling_route in self.static_mapping.routes_for_decode(
            route.decode_instance_index
        ):
            self._validate_route(sibling_route)
            sibling_last = len(sibling_route.path) - 1
            for position in range(sibling_last - 1, -1, -1):
                instance_index = sibling_route.path[position]
                if instance_index in seen:
                    continue
                distance = sibling_last - position
                transfer_path = tuple(reversed(sibling_route.path[position:]))
                label = (
                    "sibling_prefill"
                    if position == 0
                    else "sibling_path_intermediate"
                )
                candidate = (distance, label, transfer_path)
                previous = sibling_candidates.get(instance_index)
                if previous is None or (distance, transfer_path) < (
                    previous[0],
                    previous[2],
                ):
                    sibling_candidates[instance_index] = candidate
        ordered.extend(
            (
                instance_index,
                distance,
                label,
                transfer_path,
            )
            for instance_index, (distance, label, transfer_path) in sorted(
                sibling_candidates.items(),
                key=lambda item: (
                    item[1][0],
                    -self.remaining_capacity[item[0]],
                    item[0],
                    item[1][2],
                ),
            )
        )
        return tuple(ordered)

    def allocate(
        self,
        *,
        request_id: str,
        route: StaticPdRoute,
        total_bytes: int,
    ) -> KVAllocation:
        if total_bytes < 0:
            raise ValueError("KV allocation size must be non-negative")
        ordered = self._ordered_locations(route)
        available = sum(
            self.remaining_capacity[index] for index, _, _, _ in ordered
        )
        if available < total_bytes:
            raise ValueError(
                "insufficient WSC static Decode-domain Relevant(P,D) HBM capacity "
                f"for request {request_id} on static route {route.path}: need {total_bytes} "
                f"bytes, available {available}; Decode remapping and unrelated-"
                "instance widening are disabled"
            )

        remaining = total_bytes
        pieces: list[KVAllocationPiece] = []
        for instance_index, distance, priority, path in ordered:
            if remaining == 0:
                break
            take = min(remaining, self.remaining_capacity[instance_index])
            if take == 0:
                continue
            self.remaining_capacity[instance_index] -= take
            remaining -= take
            pieces.append(
                KVAllocationPiece(
                    instance_index=instance_index,
                    bytes=take,
                    distance_to_decode=distance,
                    location_priority=priority,
                    path=path,
                )
            )

        return KVAllocation(
            request_id=request_id,
            prefill_instance_index=route.prefill_instance_index,
            decode_instance_index=route.decode_instance_index,
            relevant_instance_indices=tuple(index for index, _, _, _ in ordered),
            static_route=route.path,
            total_bytes=total_bytes,
            pieces=tuple(pieces),
        )

    def available_bytes(self, route: StaticPdRoute) -> int:
        """Return current free bytes in the request's fixed Relevant(P,D) set."""

        return sum(
            self.remaining_capacity[index]
            for index, _, _, _ in self._ordered_locations(route)
        )

    def empty_domain_capacity_bytes(self, route: StaticPdRoute) -> int:
        """Return usable bytes when the fixed Relevant(P,D) set is empty."""

        return sum(
            self.initial_capacity[index]
            for index, _, _, _ in self._ordered_locations(route)
        )

    def try_allocate(
        self,
        *,
        request_id: str,
        route: StaticPdRoute,
        total_bytes: int,
    ) -> Optional[KVAllocation]:
        """Apply Algorithm 2's capacity check without changing the static route.

        A request that can never fit in its empty Relevant(P,D) domain is an
        invalid configuration.  Temporary pressure returns ``None`` so the
        central scheduler can stop this request and the following work in its
        Prefill FCFS queue until a later scheduling opportunity.
        """

        if total_bytes < 0:
            raise ValueError("KV allocation size must be non-negative")
        maximum = self.empty_domain_capacity_bytes(route)
        if maximum < total_bytes:
            raise ValueError(
                "WSC request exceeds empty static Decode-domain Relevant(P,D) "
                f"HBM capacity for request {request_id} on static route "
                f"{route.path}: need {total_bytes} bytes, empty-domain capacity "
                f"{maximum}; the configured request cannot fit without changing "
                "the WSC-LLM mapping or hardware configuration"
            )
        if self.available_bytes(route) < total_bytes:
            return None
        return self.allocate(
            request_id=request_id,
            route=route,
            total_bytes=total_bytes,
        )

    def release(self, allocation: KVAllocation) -> None:
        for piece in allocation.pieces:
            index = piece.instance_index
            self.remaining_capacity[index] += piece.bytes
            if self.remaining_capacity[index] > self.initial_capacity[index]:
                raise RuntimeError(
                    f"KV release exceeds initial capacity for instance {index}"
                )


@dataclass(frozen=True)
class WscLlmRequest:
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
class WscLlmIterationRecord:
    instance_index: int
    phase_role: str
    iteration_index: int
    start_ns: int
    end_ns: int
    prefill_request_id: Optional[str]
    prefill_chunk_tokens: int
    decode_request_ids: tuple[str, ...]
    timing_entry: WscLlmTimingEntry

    def __post_init__(self) -> None:
        if self.phase_role == PREFILL_ROLE:
            valid = (
                self.prefill_request_id is not None
                and self.prefill_chunk_tokens > 0
                and not self.decode_request_ids
            )
        elif self.phase_role == DECODE_ROLE:
            valid = (
                self.prefill_request_id is None
                and self.prefill_chunk_tokens == 0
                and bool(self.decode_request_ids)
            )
        else:
            valid = False
        if not valid or self.timing_entry.phase_role != self.phase_role:
            raise ValueError("WSC-LLM iteration must contain work from exactly one phase")


@dataclass(frozen=True)
class WscLlmRequestPlan:
    queue_index: int
    session_id: str
    turn_index: int
    request_id: str
    history_tokens_before: int
    prefill_context_tokens: int
    final_context_tokens: int
    estimated_arrival_ns: int
    prefill_instance_index: int
    prefill_assignment_key: tuple[int, int]
    prefill_start_ns: int
    prefill_complete_ns: int
    decode_instance_index: int
    static_route: StaticPdRoute
    decode_queue_depth_before_enqueue: int
    decode_start_ns: int
    completion_ns: int
    kv_allocation: KVAllocation
    terminal_kv_release_at_completion: bool
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
class WscLlmPlan:
    p_chunk: int
    topology: WscLlmTopology
    timing_lut: WscLlmTimingLut
    static_mapping: StaticPdMapping
    requests: tuple[WscLlmRequestPlan, ...]
    iterations: tuple[WscLlmIterationRecord, ...]
    final_remaining_capacity_bytes: tuple[int, ...]
    kv_cache_policy: str = "legacy"
    reserve_context_tokens: int = 0
    kv_events: tuple[KVCacheEvent, ...] = ()
    final_hbm_snapshots: tuple[NodeHBMSnapshot, ...] = ()
    final_session_snapshots: tuple[SessionKVSnapshot, ...] = ()
    planning_iteration_count: int = 0


@dataclass
class _RequestRuntime:
    request: WscLlmRequest
    history_tokens_before: int
    prefill_context_tokens: int
    final_context_tokens: int
    remaining_chunks: int
    prompt_tokens_processed: int = 0
    decode_steps_remaining: int = 0
    current_decode_token: int = 0
    estimated_arrival_ns: Optional[int] = None
    prefill_instance_index: Optional[int] = None
    prefill_assignment_key: Optional[tuple[int, int]] = None
    prefill_start_ns: Optional[int] = None
    prefill_complete_ns: Optional[int] = None
    decode_instance_index: Optional[int] = None
    static_route: Optional[StaticPdRoute] = None
    decode_queue_depth_before_enqueue: Optional[int] = None
    decode_start_ns: Optional[int] = None
    completion_ns: Optional[int] = None
    kv_allocation: Optional[KVAllocation] = None
    terminal_kv_release_at_completion: bool = False
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
    decode_capacity_reserved: bool = False
    waiting_decode_admission: bool = False


@dataclass
class _InstanceRuntime:
    index: int
    phase_role: str
    qp: deque[int] = field(default_factory=deque)
    active_decode: list[int] = field(default_factory=list)
    busy: bool = False
    iteration_count: int = 0


def _validate_and_expand_requests(
    requests: Sequence[WscLlmRequest],
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
            next_request[index] = (
                ordered[position + 1] if position + 1 < len(ordered) else None
            )
            history = final_context
    return [runtime for runtime in runtimes if runtime is not None], next_request


def _plan_wsc_llm_requests_legacy(
    *,
    hardware: WscLlmHardware,
    model: WscLlmModel,
    instance_specs: Sequence[WscLlmInstanceSpec],
    requests: Sequence[WscLlmRequest],
    alpha: float = 1.0,
) -> WscLlmPlan:
    topology = build_instances(hardware, instance_specs, require_equal_size=True)
    static_mapping = build_static_pd_mapping(topology, alpha=alpha)

    p_chunk = math.ceil(sum(request.prefill_length for request in requests) / len(requests))
    runtimes, next_request = _validate_and_expand_requests(requests, p_chunk)
    max_d_token = max(runtime.final_context_tokens for runtime in runtimes)
    timing_lut = WscLlmTimingLut.build(
        hardware,
        model,
        instance_sizes=(instance.size for instance in topology.instances),
        p_chunk=p_chunk,
        request_count=len(requests),
        max_d_token=max_d_token,
    )
    allocator = WscRelevantKvAllocator(
        topology,
        static_mapping,
        model_weight_bytes=estimate_model_weight_bytes(model),
    )
    instances = [
        _InstanceRuntime(index=instance.index, phase_role=instance.phase_role)
        for instance in topology.instances
    ]
    session_allocations: dict[str, KVAllocation] = {}

    # (time_ns, priority, sequence, kind, payload).  Completion events precede
    # arrivals at the same timestamp, then all ready dedicated instances start.
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

    iterations: list[WscLlmIterationRecord] = []

    def prefill_snapshots() -> tuple[PrefillQueueSnapshot, ...]:
        return tuple(
            PrefillQueueSnapshot(
                instance_index=state.index,
                request_count=len(state.qp),
            )
            for state in instances
            if state.phase_role == PREFILL_ROLE
        )

    def start_ready_iterations(now_ns: int) -> None:
        for state in instances:
            if state.busy:
                continue
            instance = topology.instance(state.index)
            if state.phase_role == PREFILL_ROLE:
                if state.active_decode:
                    raise RuntimeError("Prefill-only instance contains Decode work")
                if not state.qp:
                    continue
                request_index = state.qp[0]
                runtime = runtimes[request_index]
                if runtime.kv_allocation is None:
                    route = static_mapping.route_for_prefill(state.index)
                    decode_state = instances[route.decode_instance_index]
                    if decode_state.phase_role != DECODE_ROLE:
                        raise RuntimeError(
                            "static mapping selected a non-Decode instance"
                        )
                    if runtime.request.session_id in session_allocations:
                        raise RuntimeError(
                            f"session {runtime.request.session_id} already has a "
                            "retained KV allocation before Prefill admission"
                        )
                    allocation = allocator.try_allocate(
                        request_id=runtime.request.request_id,
                        route=route,
                        total_bytes=kv_cache_bytes_for_tokens(
                            model, runtime.final_context_tokens
                        ),
                    )
                    if allocation is None:
                        # WSC-LLM Algorithm 2 line 6: stop this request and
                        # every subsequent request in the same Prefill FCFS
                        # queue.  A later scheduler event rechecks capacity.
                        continue
                    runtime.decode_instance_index = route.decode_instance_index
                    runtime.static_route = route
                    runtime.kv_allocation = allocation
                    session_allocations[runtime.request.session_id] = allocation
                chunk_tokens = min(
                    p_chunk,
                    runtime.request.prefill_length - runtime.prompt_tokens_processed,
                )
                if runtime.prefill_start_ns is None:
                    runtime.prefill_start_ns = now_ns
                entry = timing_lut.lookup(
                    phase_role=PREFILL_ROLE,
                    instance_size=instance.size,
                    p_chunk=p_chunk,
                    d_batch=0,
                    d_token=0,
                )
                decode_indexes: tuple[int, ...] = ()
                prefill_request_id: Optional[str] = runtime.request.request_id
                payload: object = (
                    PREFILL_ROLE,
                    state.index,
                    request_index,
                    chunk_tokens,
                    decode_indexes,
                )
            else:
                if state.qp:
                    raise RuntimeError("Decode-only instance contains Prefill work")
                if not state.active_decode:
                    continue
                decode_indexes = tuple(state.active_decode)
                for request_index in decode_indexes:
                    if runtimes[request_index].decode_start_ns is None:
                        runtimes[request_index].decode_start_ns = now_ns
                d_token = max(
                    runtimes[index].current_decode_token for index in decode_indexes
                )
                entry = timing_lut.lookup(
                    phase_role=DECODE_ROLE,
                    instance_size=instance.size,
                    p_chunk=0,
                    d_batch=len(decode_indexes),
                    d_token=d_token,
                )
                request_index = -1
                chunk_tokens = 0
                prefill_request_id = None
                payload = (
                    DECODE_ROLE,
                    state.index,
                    request_index,
                    chunk_tokens,
                    decode_indexes,
                )

            end_ns = now_ns + entry.iteration_time_ns
            if _ITERATION_STATS_HOOK is not None:
                _ITERATION_STATS_HOOK(entry, now_ns, end_ns)
            iterations.append(
                WscLlmIterationRecord(
                    instance_index=state.index,
                    phase_role=state.phase_role,
                    iteration_index=state.iteration_count,
                    start_ns=now_ns,
                    end_ns=end_ns,
                    prefill_request_id=prefill_request_id,
                    prefill_chunk_tokens=chunk_tokens,
                    decode_request_ids=tuple(
                        runtimes[index].request.request_id for index in decode_indexes
                    ),
                    timing_entry=entry,
                )
            )
            state.iteration_count += 1
            state.busy = True
            push_event(end_ns, 0, "iteration_complete", payload)

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
            phase_role, state_index, prefill_index, chunk_tokens, decode_indexes = payload
            state = instances[state_index]
            state.busy = False
            if phase_role != state.phase_role:
                raise RuntimeError("iteration phase does not match dedicated instance role")

            if phase_role == PREFILL_ROLE:
                runtime = runtimes[prefill_index]
                runtime.prompt_tokens_processed += chunk_tokens
                runtime.remaining_chunks -= 1
                if runtime.remaining_chunks < 0:
                    raise RuntimeError("Prefill remaining chunk count became negative")
                if runtime.remaining_chunks == 0:
                    if not state.qp or state.qp[0] != prefill_index:
                        raise RuntimeError("Prefill FCFS queue order was corrupted")
                    state.qp.popleft()
                    runtime.prefill_complete_ns = now_ns
                    route = runtime.static_route
                    if route is None or runtime.kv_allocation is None:
                        raise RuntimeError(
                            "Prefill completed without a reserved WSC KV allocation"
                        )
                    decode_state = instances[route.decode_instance_index]
                    if decode_state.phase_role != DECODE_ROLE:
                        raise RuntimeError("static mapping selected a non-Decode instance")
                    runtime.decode_queue_depth_before_enqueue = len(
                        decode_state.active_decode
                    )
                    decode_state.active_decode.append(prefill_index)
            else:
                for request_index in decode_indexes:
                    runtime = runtimes[request_index]
                    runtime.decode_steps_remaining -= 1
                    runtime.current_decode_token += 1
                    if runtime.decode_steps_remaining < 0:
                        raise RuntimeError("Decode remaining step count became negative")
                    if runtime.decode_steps_remaining == 0:
                        if request_index not in state.active_decode:
                            raise RuntimeError("Decode queue membership was corrupted")
                        state.active_decode.remove(request_index)
                        runtime.completion_ns = now_ns
                        completed_requests += 1
                        following = next_request[request_index]
                        if following is not None:
                            next_runtime = runtimes[following]
                            interval = next_runtime.request.inter_request_interval_ns
                            if interval is None:
                                raise RuntimeError(
                                    "validated later request lost its interval"
                                )
                            push_event(now_ns + interval, 1, "arrival", following)
                        else:
                            terminal_allocation = session_allocations.pop(
                                runtime.request.session_id,
                                None,
                            )
                            if terminal_allocation is None:
                                raise RuntimeError(
                                    "terminal request has no retained KV allocation"
                                )
                            if terminal_allocation.request_id != runtime.request.request_id:
                                raise RuntimeError(
                                    "terminal session KV allocation does not match request"
                                )
                            allocator.release(terminal_allocation)
                            runtime.terminal_kv_release_at_completion = True

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

            snapshots = prefill_snapshots()
            selected = select_prefill_instance(snapshots)
            selected_snapshot = next(
                snapshot
                for snapshot in snapshots
                if snapshot.instance_index == selected
            )
            runtime.prefill_instance_index = selected
            runtime.prefill_assignment_key = selected_snapshot.ordering_key
            instances[selected].qp.append(request_index)

        start_ready_iterations(now_ns)

    if completed_requests != len(runtimes):
        raise RuntimeError(
            "WSC-LLM planning stopped with "
            f"{completed_requests}/{len(runtimes)} requests complete"
        )
    if any(state.busy or state.qp or state.active_decode for state in instances):
        raise RuntimeError("WSC-LLM planning ended with non-idle instance state")
    if session_allocations:
        raise RuntimeError(
            "WSC-LLM planning ended with terminal session KV still retained: "
            f"{sorted(session_allocations)}"
        )

    plans: list[WscLlmRequestPlan] = []
    for runtime in runtimes:
        required = {
            "estimated_arrival_ns": runtime.estimated_arrival_ns,
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": runtime.prefill_assignment_key,
            "prefill_start_ns": runtime.prefill_start_ns,
            "prefill_complete_ns": runtime.prefill_complete_ns,
            "decode_instance_index": runtime.decode_instance_index,
            "static_route": runtime.static_route,
            "decode_queue_depth_before_enqueue": (
                runtime.decode_queue_depth_before_enqueue
            ),
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
            WscLlmRequestPlan(
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
                static_route=runtime.static_route,
                decode_queue_depth_before_enqueue=int(
                    runtime.decode_queue_depth_before_enqueue
                ),
                decode_start_ns=int(runtime.decode_start_ns),
                completion_ns=int(runtime.completion_ns),
                kv_allocation=runtime.kv_allocation,
                terminal_kv_release_at_completion=(
                    runtime.terminal_kv_release_at_completion
                ),
                history_source_instance_index=runtime.history_source_instance_index,
                history_transfer_bytes=runtime.history_transfer_bytes,
            )
        )

    return WscLlmPlan(
        p_chunk=p_chunk,
        topology=topology,
        timing_lut=timing_lut,
        static_mapping=static_mapping,
        requests=tuple(sorted(plans, key=lambda plan: plan.queue_index)),
        iterations=tuple(
            sorted(
                iterations,
                key=lambda item: (
                    item.start_ns,
                    item.instance_index,
                    item.iteration_index,
                ),
            )
        ),
        final_remaining_capacity_bytes=tuple(allocator.remaining_capacity),
    )


def _plan_wsc_llm_session_lru_recompute(
    *,
    hardware: WscLlmHardware,
    model: WscLlmModel,
    instance_specs: Sequence[WscLlmInstanceSpec],
    requests: Sequence[WscLlmRequest],
    alpha: float,
    p_chunk: int,
    reserve_context_tokens: int,
    record_planning_iterations: bool,
) -> WscLlmPlan:
    """Plan WSC-LLM with static P→D routes and resident/evicted session KV."""

    if isinstance(p_chunk, bool) or not isinstance(p_chunk, int) or p_chunk <= 0:
        raise ValueError("p_chunk must be a positive integer")
    topology = build_instances(hardware, instance_specs, require_equal_size=True)
    static_mapping = build_static_pd_mapping(topology, alpha=alpha)
    runtimes, next_request = _validate_and_expand_requests(requests, p_chunk)
    max_d_token = max(runtime.final_context_tokens for runtime in runtimes)
    timing_lut = WscLlmTimingLut.build(
        hardware,
        model,
        instance_sizes=(instance.size for instance in topology.instances),
        p_chunk=p_chunk,
        request_count=len(requests),
        max_d_token=max_d_token,
    )
    kv_manager = SessionKVCacheManager(
        topology,
        model,
        reserve_context_tokens=reserve_context_tokens,
    )
    instances = [
        _InstanceRuntime(index=instance.index, phase_role=instance.phase_role)
        for instance in topology.instances
    ]
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

    iterations: list[WscLlmIterationRecord] = []
    iteration_count = 0
    # Decode handoffs occur once per request, whereas Decode iterations occur
    # once per generated token.  Track pending handoffs explicitly instead of
    # scanning every runtime record at every scheduling timestamp.
    waiting_decode_admissions: dict[int, deque[int]] = {
        instance.index: deque() for instance in topology.instances
    }
    # Do not repeatedly invoke the HBM manager for an admission that remains
    # blocked solely by active KV.  A new attempt is meaningful only after a
    # planner-time HBM state transition or after a new Decode handoff arrives.
    capacity_epoch = [0 for _ in topology.instances]
    prefill_attempt_epoch: dict[int, tuple[int, int]] = {}
    decode_admission_epoch = [-1 for _ in topology.instances]
    decode_admission_dirty: set[int] = set()

    def note_capacity_change(*instance_indexes: Optional[int]) -> None:
        for instance_index in set(instance_indexes):
            if instance_index is not None:
                capacity_epoch[instance_index] += 1

    def prefill_snapshots() -> tuple[PrefillQueueSnapshot, ...]:
        return tuple(
            PrefillQueueSnapshot(instance_index=state.index, request_count=len(state.qp))
            for state in instances
            if state.phase_role == PREFILL_ROLE
        )

    def try_admit_prefill(request_index: int, now_ns: int) -> bool:
        runtime = runtimes[request_index]
        if runtime.admitted_prefill:
            return True
        if runtime.prefill_instance_index is None or runtime.static_route is None:
            raise RuntimeError("WSC Prefill admission lost its static mapping")
        prefill_instance = runtime.prefill_instance_index
        decode_instance = runtime.static_route.decode_instance_index
        admission_epoch = (
            capacity_epoch[prefill_instance],
            capacity_epoch[decode_instance],
        )
        if prefill_attempt_epoch.get(request_index) == admission_epoch:
            return False
        prefill_attempt_epoch[request_index] = admission_epoch
        final_shards = kv_cache_shard_bytes_for_tokens(
            model, runtime.final_context_tokens, topology.instances[0].size
        )
        reservation = kv_manager.reserve_request_capacity(
            runtime.request.request_id,
            runtime.request.session_id,
            decode_instance,
            final_shards,
            now_ns,
            phase="prefill_admission",
            reason="static_decode_final_kv_reservation",
        )
        runtime.decode_target_evictions = tuple(
            (*runtime.decode_target_evictions, *reservation.evictions)
        )
        if not reservation.admitted:
            if reservation.evictions:
                note_capacity_change(
                    decode_instance,
                    *(record.victim_instance_index for record in reservation.evictions),
                )
            return False
        runtime.decode_capacity_reserved = True
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
            kv_manager.release_request_capacity(runtime.request.request_id)
            runtime.decode_capacity_reserved = False
            note_capacity_change(
                prefill_instance,
                decode_instance,
                decision.source_instance_index,
                *(record.victim_instance_index for record in reservation.evictions),
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
            raise RuntimeError("Prefill growth lost its successful capacity preflight")
        runtime.admission_evictions = tuple((*runtime.admission_evictions, *growth.evictions))
        runtime.history_recompute_processed = 0
        runtime.remaining_chunks = (
            math.ceil(runtime.history_recompute_tokens / p_chunk)
            + math.ceil(runtime.request.prefill_length / p_chunk)
        )
        runtime.admitted_prefill = True
        note_capacity_change(
            prefill_instance,
            decode_instance,
            decision.source_instance_index,
            *(record.victim_instance_index for record in reservation.evictions),
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
                    raise RuntimeError("WSC Decode target queue was corrupted")
                if not runtime.decode_capacity_reserved:
                    raise RuntimeError("WSC Decode target was not reserved before Prefill")
                kv_manager.release_request_capacity(runtime.request.request_id)
                runtime.decode_capacity_reserved = False
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
                    raise RuntimeError("released static Decode reservation was not physically usable")
                runtime.prefill_decode_transfer = move.transfer
                growth = kv_manager.grow_decode(
                    runtime.request.session_id,
                    runtime.final_context_tokens,
                    now_ns,
                    runtime.request.request_id,
                )
                if not growth.admitted:
                    raise RuntimeError("Decode growth lost its successful target admission")
                runtime.decode_target_evictions = tuple(
                    (*runtime.decode_target_evictions, *growth.evictions)
                )
                decode_state = instances[target_instance]
                if decode_state.phase_role != DECODE_ROLE:
                    raise RuntimeError("static route selected a non-Decode instance")
                runtime.decode_queue_depth_before_enqueue = len(decode_state.active_decode)
                decode_state.active_decode.append(request_index)
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
            if state.busy:
                continue
            instance = topology.instance(state.index)
            if state.phase_role == PREFILL_ROLE:
                if state.active_decode:
                    raise RuntimeError("Prefill-only instance contains Decode work")
                if not state.qp:
                    continue
                request_index = state.qp[0]
                if not try_admit_prefill(request_index, now_ns):
                    continue
                runtime = runtimes[request_index]
                history_remaining = (
                    runtime.history_recompute_tokens - runtime.history_recompute_processed
                )
                history_chunk = history_remaining > 0
                chunk_tokens = min(
                    p_chunk,
                    history_remaining if history_chunk else runtime.request.prefill_length - runtime.prompt_tokens_processed,
                )
                if chunk_tokens <= 0:
                    raise RuntimeError("Prefill has a non-positive chunk")
                if runtime.prefill_start_ns is None:
                    runtime.prefill_start_ns = now_ns
                entry = timing_lut.lookup(
                    phase_role=PREFILL_ROLE,
                    instance_size=instance.size,
                    p_chunk=p_chunk,
                    d_batch=0,
                    d_token=0,
                )
                payload: object = (PREFILL_ROLE, state.index, request_index, chunk_tokens, history_chunk, ())
                prefill_request_id: Optional[str] = runtime.request.request_id
                decode_indexes: tuple[int, ...] = ()
            else:
                if state.qp:
                    raise RuntimeError("Decode-only instance contains Prefill work")
                if not state.active_decode:
                    continue
                decode_indexes = tuple(state.active_decode)
                for request_index in decode_indexes:
                    if runtimes[request_index].decode_start_ns is None:
                        runtimes[request_index].decode_start_ns = now_ns
                entry = timing_lut.lookup(
                    phase_role=DECODE_ROLE,
                    instance_size=instance.size,
                    p_chunk=0,
                    d_batch=len(decode_indexes),
                    d_token=max(runtimes[index].current_decode_token for index in decode_indexes),
                )
                request_index = -1
                chunk_tokens = 0
                history_chunk = False
                payload = (DECODE_ROLE, state.index, request_index, chunk_tokens, history_chunk, decode_indexes)
                prefill_request_id = None
            end_ns = now_ns + entry.iteration_time_ns
            if _ITERATION_STATS_HOOK is not None:
                _ITERATION_STATS_HOOK(entry, now_ns, end_ns)
            if record_planning_iterations:
                iterations.append(
                    WscLlmIterationRecord(
                        instance_index=state.index,
                        phase_role=state.phase_role,
                        iteration_index=state.iteration_count,
                        start_ns=now_ns,
                        end_ns=end_ns,
                        prefill_request_id=prefill_request_id,
                        prefill_chunk_tokens=chunk_tokens,
                        decode_request_ids=tuple(
                            runtimes[index].request.request_id for index in decode_indexes
                        ),
                        timing_entry=entry,
                    )
                )
            state.iteration_count += 1
            iteration_count += 1
            state.busy = True
            push_event(end_ns, 0, "iteration_complete", payload)

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
            phase_role, state_index, prefill_index, chunk_tokens, history_chunk, decode_indexes = payload
            state = instances[int(state_index)]
            state.busy = False
            if phase_role != state.phase_role:
                raise RuntimeError("iteration phase does not match dedicated instance role")
            if phase_role == PREFILL_ROLE:
                runtime = runtimes[int(prefill_index)]
                if bool(history_chunk):
                    runtime.history_recompute_processed += int(chunk_tokens)
                else:
                    runtime.prompt_tokens_processed += int(chunk_tokens)
                runtime.remaining_chunks -= 1
                if runtime.remaining_chunks < 0:
                    raise RuntimeError("Prefill remaining chunk count became negative")
                if runtime.remaining_chunks == 0:
                    if not state.qp or state.qp[0] != prefill_index:
                        raise RuntimeError("Prefill FCFS queue order was corrupted")
                    if runtime.history_recompute_processed != runtime.history_recompute_tokens:
                        raise RuntimeError("history recompute chunks did not cover planned history")
                    if runtime.prompt_tokens_processed != runtime.request.prefill_length:
                        raise RuntimeError("Prefill chunks did not cover current prompt")
                    state.qp.popleft()
                    runtime.prefill_complete_ns = now_ns
                    runtime.waiting_decode_admission = True
                    if runtime.decode_instance_index is None:
                        raise RuntimeError("WSC Prefill completion lost Decode target")
                    waiting_decode_admissions[runtime.decode_instance_index].append(
                        int(prefill_index)
                    )
                    decode_admission_dirty.add(runtime.decode_instance_index)
            else:
                for request_index in decode_indexes:
                    runtime = runtimes[int(request_index)]
                    runtime.decode_steps_remaining -= 1
                    runtime.current_decode_token += 1
                    if runtime.decode_steps_remaining < 0:
                        raise RuntimeError("Decode remaining step count became negative")
                    if runtime.decode_steps_remaining == 0:
                        if request_index not in state.active_decode:
                            raise RuntimeError("Decode queue membership was corrupted")
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
                        snapshot = kv_manager.session_snapshot(runtime.request.session_id)
                        if snapshot is None:
                            raise RuntimeError("completed session disappeared from KV manager")
                        runtime.kv_state_after_completion = snapshot.state
                        runtime.kv_instance_after_completion = snapshot.instance_index
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
            snapshots = prefill_snapshots()
            selected = select_prefill_instance(snapshots)
            selected_snapshot = next(
                snapshot for snapshot in snapshots if snapshot.instance_index == selected
            )
            route = static_mapping.route_for_prefill(selected)
            runtime.prefill_instance_index = selected
            runtime.prefill_assignment_key = selected_snapshot.ordering_key
            runtime.static_route = route
            runtime.decode_instance_index = route.decode_instance_index
            instances[selected].qp.append(request_index)

        start_ready_iterations(now_ns)

    if completed_requests != len(runtimes):
        raise RuntimeError(
            "WSC-LLM planning stopped with "
            f"{completed_requests}/{len(runtimes)} requests complete"
        )
    if any(state.busy or state.qp or state.active_decode for state in instances):
        raise RuntimeError("WSC-LLM planning ended with non-idle instance state")
    kv_manager.assert_final_state()

    plans: list[WscLlmRequestPlan] = []
    for runtime in runtimes:
        required = {
            "estimated_arrival_ns": runtime.estimated_arrival_ns,
            "prefill_instance_index": runtime.prefill_instance_index,
            "prefill_assignment_key": runtime.prefill_assignment_key,
            "prefill_start_ns": runtime.prefill_start_ns,
            "prefill_complete_ns": runtime.prefill_complete_ns,
            "decode_instance_index": runtime.decode_instance_index,
            "static_route": runtime.static_route,
            "decode_queue_depth_before_enqueue": runtime.decode_queue_depth_before_enqueue,
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
        route = runtime.static_route
        if route is None:
            raise RuntimeError("request lost its static route")
        allocation = KVAllocation(
            request_id=runtime.request.request_id,
            prefill_instance_index=int(runtime.prefill_instance_index),
            decode_instance_index=int(runtime.decode_instance_index),
            relevant_instance_indices=(int(runtime.decode_instance_index),),
            static_route=route.path,
            total_bytes=sum(final_shards),
            pieces=(
                KVAllocationPiece(
                    instance_index=int(runtime.decode_instance_index),
                    bytes=sum(final_shards),
                    distance_to_decode=0,
                    location_priority="decode",
                    path=(int(runtime.decode_instance_index),),
                ),
            ),
        )
        plans.append(
            WscLlmRequestPlan(
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
                static_route=route,
                decode_queue_depth_before_enqueue=int(runtime.decode_queue_depth_before_enqueue),
                decode_start_ns=int(runtime.decode_start_ns),
                completion_ns=int(runtime.completion_ns),
                kv_allocation=allocation,
                terminal_kv_release_at_completion=False,
                history_source_instance_index=runtime.history_source_instance_index,
                history_transfer_bytes=runtime.history_transfer_bytes,
                history_cache_state_before=runtime.history_cache_state_before,
                history_action=runtime.history_action,
                history_recompute_tokens=runtime.history_recompute_tokens,
                effective_prefill_tokens=(
                    runtime.request.prefill_length + runtime.history_recompute_tokens
                ),
                history_transfer=runtime.history_transfer,
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
    final_sessions = tuple(
        snapshot
        for session_id in kv_manager.session_ids
        if (snapshot := kv_manager.session_snapshot(session_id)) is not None
    )
    return WscLlmPlan(
        p_chunk=p_chunk,
        topology=topology,
        timing_lut=timing_lut,
        static_mapping=static_mapping,
        requests=tuple(sorted(plans, key=lambda item: item.queue_index)),
        iterations=tuple(iterations),
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


def plan_wsc_llm_requests(
    *,
    hardware: WscLlmHardware,
    model: WscLlmModel,
    instance_specs: Sequence[WscLlmInstanceSpec],
    requests: Sequence[WscLlmRequest],
    alpha: float = 1.0,
    p_chunk: Optional[int] = None,
    kv_cache_policy: str = "legacy",
    reserve_context_tokens: int = 1_000_000,
    record_planning_iterations: bool = True,
) -> WscLlmPlan:
    """Plan WSC-LLM requests under the selected KV policy."""

    if kv_cache_policy == "legacy":
        if p_chunk is not None:
            raise ValueError("legacy WSC-LLM planning does not accept an explicit p_chunk")
        return _plan_wsc_llm_requests_legacy(
            hardware=hardware,
            model=model,
            instance_specs=instance_specs,
            requests=requests,
            alpha=alpha,
        )
    if kv_cache_policy != "session_lru_recompute":
        raise ValueError(f"unsupported kv_cache_policy: {kv_cache_policy!r}")
    if p_chunk is None:
        raise ValueError("session_lru_recompute requires an explicit p_chunk")
    return _plan_wsc_llm_session_lru_recompute(
        hardware=hardware,
        model=model,
        instance_specs=instance_specs,
        requests=requests,
        alpha=alpha,
        p_chunk=p_chunk,
        reserve_context_tokens=reserve_context_tokens,
        record_planning_iterations=record_planning_iterations,
    )
