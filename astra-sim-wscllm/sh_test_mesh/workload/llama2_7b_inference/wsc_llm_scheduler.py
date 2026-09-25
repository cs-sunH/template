#!/usr/bin/env python3
"""Deterministic WSC-LLM PD-disaggregated request-mapping model.

WSC-LLM describes a live central scheduler.  This module provides the
placement, queueing, and KV-allocation semantics reused by the online scheduler
and its focused tests.  Whole configured instances are the placement unit:
every instance keeps its existing six-rank TP communicator and is dedicated to
either Prefill or Decode for the complete run.
"""

from __future__ import annotations

import csv
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence


PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
PHASE_ROLES = frozenset((PREFILL_ROLE, DECODE_ROLE))


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


