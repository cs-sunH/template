#!/usr/bin/env python3
"""Deterministic WSC-LLM PD-disaggregated request-mapping model.

WSC-LLM describes a live central scheduler.  This module provides the
placement and queueing semantics reused by the online scheduler and its
focused tests.  Whole configured instances are the placement unit: every
instance keeps its existing six-rank TP communicator and is dedicated to
either Prefill or Decode for the complete run.  (The historical offline
``WscRelevantKvAllocator``/timing-LUT estimation family was dead code in the
online-only pipeline and was removed 2026-09-27; per-session KV placement now
lives exclusively in ``session_kv_manager``.)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence


PREFILL_ROLE = "prefill"
DECODE_ROLE = "decode"
PHASE_ROLES = frozenset((PREFILL_ROLE, DECODE_ROLE))

# Route-enumeration bounds (fail-closed): the shortest-path materialization
# and the per-Prefill candidate Cartesian product are exponential in the
# number of equivalent shortest paths, so grid/adversarial topologies must be
# rejected instead of exploding CPU/memory at startup.  Both bounds are
# orders of magnitude above every checked-in layout (the real 9-instance
# layout has exactly one nearest-Decode candidate per Prefill instance), so
# results are unchanged within the bounds.
MAX_EQUIVALENT_SHORTEST_PATHS = 64
MAX_ROUTE_CANDIDATE_COMBINATIONS = 4096


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
    ranks: tuple[int, ...]
    phase_role: str
    wafer_center_manhattan_distance: float

    @property
    def size(self) -> int:
        return len(self.ranks)


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
                ranks=tuple(spec.ranks),
                phase_role=spec.phase_role,
                wafer_center_manhattan_distance=(
                    abs(center_row - wafer_center_row)
                    + abs(center_col - wafer_center_col)
                ),
            )
        )

    if sorted(all_ranks) != list(range(hardware.npus_count)):
        raise ValueError("WSC-LLM instances must cover every configured NPU exactly once")
    if len({instance.size for instance in instances}) != 1:
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
            if len(paths) > MAX_EQUIVALENT_SHORTEST_PATHS:
                raise ValueError(
                    f"topology has more than {MAX_EQUIVALENT_SHORTEST_PATHS} "
                    f"equivalent shortest paths from {source} to {node} "
                    "(route enumeration bound exceeded; refusing the "
                    "combinatorial blow-up)"
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
    # 以下三个汇总字段为测试专用(仅 test_wsc_llm_scheduler.py 断言消费,
    # 生产决策只读 routes);保留作映射构建不变量的独立校验口径。
    total_hops: int
    shared_edge_occurrences: int
    edge_use_counts: tuple[tuple[int, int, int], ...]

    def route_for_prefill(self, prefill_instance_index: int) -> StaticPdRoute:
        for route in self.routes:
            if route.prefill_instance_index == prefill_instance_index:
                return route
        raise KeyError(f"no static Decode route for Prefill instance {prefill_instance_index}")


def build_static_pd_mapping(topology: WscLlmTopology) -> StaticPdMapping:
    """Choose deterministic nearest-Decode shortest routes for all Prefill instances."""

    graph = InstanceGraph(topology)
    prefill_indices = topology.indices_for_role(PREFILL_ROLE)
    decode_indices = topology.indices_for_role(DECODE_ROLE)
    # SCHED-8(2026-09-28)边界加固:空 Prefill 拓扑此前静默返回空映射
    # (routes=()),直到消费方 route_for_prefill 才 KeyError;现构造期
    # fail-closed。生产拓扑唯一构造器 build_instances 已强制双角色齐备,
    # 该守卫覆盖绕过它手工构造 WscLlmTopology 直接调本函数的调用方。
    # 空 Decode 拓扑由下方 min(decode_distances.values()) 对空序列抛
    # ValueError,同样不静默。
    if not prefill_indices:
        raise ValueError(
            "static P-D mapping requires at least one Prefill instance")
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

    combination_count = 1
    for _, candidates in route_candidates:
        combination_count *= len(candidates)
    if combination_count > MAX_ROUTE_CANDIDATE_COMBINATIONS:
        raise ValueError(
            f"static P-D route enumeration would explore {combination_count} "
            f"candidate combinations (bound "
            f"{MAX_ROUTE_CANDIDATE_COMBINATIONS}); refusing the combinatorial "
            "blow-up"
        )

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
    # SCHED-8:best_selection 的 None 检查已删(不可达防御)。choose 的
    # 基例(position == len(route_candidates))对每个 prefill 至少评估一次
    # ——decode_indices 非空(空则上方 min 抛 ValueError)、
    # all_shortest_paths 无路即抛,故 route_candidates 各条目候选非空,
    # best_selection 恒被赋值。

    edge_counts: dict[tuple[int, int], int] = {}
    for _, _, path in best_selection:
        for source, target in zip(path, path[1:]):
            edge = graph.edge(source, target)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    # 3.17.3 fix (2026-09-29, doc §5.1 contract): 策略文档 §5.1 声明向心
    # P/D 布局保证"每个 P 的最近 D 都在邻接位置，P→D 交接只需一跳链路；
    # 离线规划同时最小化总跳数与共享链路数"。此前的选择器只做最小化
    # (总跳数、共边次数进 objective),不校验保证本身——一份自绘布局若
    # 实际只能给出多跳/共边路由,会静默带病运行,而文档口径与运行事实
    # 漂移。这里把两条文档性质升格为构造期 fail-closed 不变量:被选中
    # 的每条 P→D 路由必须一跳,且任一物理边不得被两条路由共享。在役
    # 6P:3D 布局(六条互不相交一跳边)不受影响。
    multi_hop = [
        (prefill_index, decode_index, len(path) - 1)
        for prefill_index, decode_index, path in best_selection
        if len(path) - 1 != 1
    ]
    if multi_hop:
        raise ValueError(
            "centripetal P/D layout contract violated: selected P->D "
            f"routes must be one hop, got multi-hop {multi_hop[:5]} -- the "
            "layout does not place every Prefill next to its nearest "
            "Decode instance (doc §5.1)"
        )
    shared_edges = sorted(
        edge for edge, count in edge_counts.items() if count > 1
    )
    if shared_edges:
        raise ValueError(
            "centripetal P/D layout contract violated: P->D handoff edges "
            f"must be pairwise disjoint, shared edges {shared_edges[:5]} -- "
            "the layout forces routes to share physical links (doc §5.1)"
        )
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
    )


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
