#!/usr/bin/env python3
"""Generate WSC-LLM PD-disaggregated Chakra ET traces for the wafer scenario."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shlex
import shutil
import sys
import tempfile
import uuid
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SH_TEST_DIR = MODULE_DIR.parents[1]
if str(SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SH_TEST_DIR))

from wsc_llm_scheduler import (  # noqa: E402
    DECODE_ROLE,
    PREFILL_ROLE,
    WscLlmHardware,
    WscLlmInstanceSpec,
    WscLlmTimingEntry,
    WscLlmModel,
    WscLlmPlan,
    WscLlmRequest,
    WscLlmRequestPlan,
    KVAllocation,
    kv_cache_bytes_for_tokens,
    plan_wsc_llm_requests,
)
import wsc_llm_scheduler  # noqa: E402
from session_kv_manager import NOC_MIGRATE, RECOMPUTE  # noqa: E402
import session_kv_manager  # noqa: E402
from metrics_integration import (  # noqa: E402
    EVENT_DECODE_END,
    EVENT_DECODE_START,
    EVENT_MEMORY_ANCHOR_COMPLETE,
    EVENT_PREFILL_END,
    EVENT_PREFILL_START,
    PlannerLutStatsAccumulator,
    ServiceMetrics,
    kv_event_payload_legacy,
    kv_event_payload_session_lru,
    resolve_metrics_detail,
    write_planner_lut_stats,
)
from generate_trace import (  # noqa: E402
    ChakraAttr,
    ChakraNode,
    GlobalMetadata,
    PROJECT_ROOT,
    REQUEST_QUEUE_COLUMNS,
    RequestSpec,
    TraceBuilder,
    clean_csv_row,
    encode_message,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    parse_rank_spec,
    range_label,
    request_queue_digest,
    sanitize_node_prefix,
    shard_extent,
    transformer_pass,
    transformer_pass_aggregated,
)
from config_resolver import (  # noqa: E402
    ResolvedHardware,
    load_hardware_config,
    materialize_runtime_configs,
)


CONFIG_CSV_PATH = MODULE_DIR / "trace_config.csv"
CONFIG_COLUMNS = (
    "kind",
    "key",
    "value",
    "group_name",
    "pg_name",
    "ranks",
    "phase_role",
)
REQUIRED_CONFIG_KEYS = (
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
    "model_name",
    "mlp_variant",
    "output_prefix",
    "output_dir",
    "request_queue_csv",
    "hardware_config",
    "local_hbm_capacity_profile",
    "system_template",
    "remote_operand_loads",
)
OPTIONAL_CONFIG_DEFAULTS = {
    "request_queue_session_limit": "0",
    "trace_granularity": "token_expanded",
    "prefill_chunk_size": "512",
    "kv_cache_policy": "legacy",
    "kv_reserve_context_tokens": "0",
    "record_planning_iterations": "true",
}
SUPPORTED_CONFIG_KEYS = set(REQUIRED_CONFIG_KEYS) | set(OPTIONAL_CONFIG_DEFAULTS)
INT_CONFIG_KEYS = {
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
}
PATH_CONFIG_KEYS = {
    "output_dir",
    "request_queue_csv",
    "hardware_config",
    "system_template",
}
OPERATOR_GRANULARITY = (
    "rmsnorm,qkv,qk,scale_mask,softmax,av,out_proj,residual,"
    "mlp_gate_up,swiglu,mlp_down,logits"
)
IDLE_SENTINEL_DURATION_NS = 1000


@dataclass(frozen=True)
class WscLlmInferenceGroup:
    name: str
    pg_name: str
    ranks: tuple[int, ...]
    phase_role: str


@dataclass(frozen=True)
class WscLlmTraceConfig:
    config_csv: Path
    npus_count: int
    layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    vocab_size: int
    bytes_per_elem: int
    model_name: str
    mlp_variant: str
    output_prefix: str
    output_dir: Optional[Path]
    request_queue_csv: Path
    request_queue: tuple[RequestSpec, ...]
    hardware_config: Path
    hardware_capacity_profile: str
    hardware: WscLlmHardware
    hardware_metadata: dict[str, object]
    system_template: Path
    system_config: Path
    network_config: Path
    comm_group_config: Path
    remote_memory_config: Path
    remote_operand_loads: bool
    inference_groups: tuple[WscLlmInferenceGroup, ...]
    request_queue_session_limit: int
    selected_session_ids: tuple[str, ...]
    source_request_count: int
    source_session_count: int
    trace_granularity: str
    prefill_chunk_size: int
    kv_cache_policy: str
    kv_reserve_context_tokens: int
    record_planning_iterations: bool
    configuration_digest: str

    @property
    def model(self) -> WscLlmModel:
        return WscLlmModel(
            layers=self.layers,
            hidden_size=self.hidden_size,
            ffn_size=self.ffn_size,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            bytes_per_elem=self.bytes_per_elem,
            mlp_variant=self.mlp_variant,
        )


@dataclass(frozen=True)
class HistoryPieceGate:
    source_instance_index: int
    total_bytes: int
    timer_gates: dict[int, Optional[int]]


def _add_idle_rank_sentinels(
    builders: dict[int, TraceBuilder],
) -> tuple[int, ...]:
    """Give every otherwise-empty rank one root CPU timer for ASTRA compatibility."""

    idle_ranks: list[int] = []
    for rank in sorted(builders):
        builder = builders[rank]
        if builder.nodes:
            continue
        node_id = builder.timer_gate(
            f"wsc_llm_idle_rank_{rank:04d}_sentinel",
            IDLE_SENTINEL_DURATION_NS,
        )
        if node_id is None:
            raise RuntimeError(f"rank {rank}: idle sentinel timer was not created")
        idle_ranks.append(rank)
    return tuple(idle_ranks)


def _validate_rank_dag(rank: int, builder: TraceBuilder) -> None:
    """Validate one rank-local Chakra DAG in O(V+E) time."""

    if not builder.nodes:
        raise RuntimeError(f"rank {rank}: Chakra ET DAG is empty")

    nodes_by_id: dict[int, ChakraNode] = {}
    for node in builder.nodes:
        node_id = int(node.id)
        if node_id in nodes_by_id:
            raise RuntimeError(f"rank {rank}: duplicate node ID {node_id}")
        nodes_by_id[node_id] = node

    indegree = {node_id: 0 for node_id in nodes_by_id}
    dependents: dict[int, list[int]] = {node_id: [] for node_id in nodes_by_id}
    for node_id, node in nodes_by_id.items():
        for raw_dependency_id in node.data_deps:
            dependency_id = int(raw_dependency_id)
            if dependency_id == node_id:
                raise RuntimeError(
                    f"rank {rank}: node {node_id} has a self-dependency"
                )
            if dependency_id not in nodes_by_id:
                raise RuntimeError(
                    f"rank {rank}: node {node_id} references missing dependency "
                    f"{dependency_id}"
                )
            indegree[node_id] += 1
            dependents[dependency_id].append(node_id)

    ready = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        node_id = ready.popleft()
        visited += 1
        for dependent_id in dependents[node_id]:
            indegree[dependent_id] -= 1
            if indegree[dependent_id] == 0:
                ready.append(dependent_id)
    if visited != len(nodes_by_id):
        cyclic_ids = [node_id for node_id, degree in indegree.items() if degree > 0]
        raise RuntimeError(
            f"rank {rank}: Chakra ET DAG contains a cycle involving node IDs "
            f"{cyclic_ids}"
        )


def _validate_all_rank_dags(builders: dict[int, TraceBuilder]) -> None:
    for rank in sorted(builders):
        _validate_rank_dag(rank, builders[rank])


def _resolve_from_sh_test(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SH_TEST_DIR / path).resolve()


def _resolve_request_queue(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SH_TEST_DIR / "workload" / path).resolve()


def _parse_config_value(key: str, value: str) -> object:
    if key in INT_CONFIG_KEYS:
        return parse_int(value, key)
    if key == "remote_operand_loads":
        return parse_bool(value, key)
    if key == "request_queue_session_limit":
        return parse_nonnegative_int(value, key)
    if key == "prefill_chunk_size":
        return parse_int(value, key)
    if key == "kv_reserve_context_tokens":
        return parse_nonnegative_int(value, key)
    if key == "record_planning_iterations":
        return parse_bool(value, key)
    if key == "kv_cache_policy":
        if value not in {"legacy", "session_lru_recompute"}:
            raise ValueError(
                "config key kv_cache_policy must be legacy or "
                "session_lru_recompute"
            )
        return value
    if key == "trace_granularity":
        if value not in {"token_expanded", "request_aggregated"}:
            raise ValueError(
                "config key trace_granularity must be token_expanded or "
                f"request_aggregated, got {value!r}"
            )
        return value
    if key == "mlp_variant":
        if value not in {"gelu", "swiglu"}:
            raise ValueError("config key mlp_variant must be gelu or swiglu")
        return value
    if key in PATH_CONFIG_KEYS:
        if key == "output_dir" and not value:
            return None
        if not value:
            raise ValueError(f"config key {key} must not be empty")
        return Path(value)
    if not value:
        raise ValueError(f"config key {key} must not be empty")
    return value


def _configuration_digest(paths: Sequence[Path]) -> str:
    digest = hashlib.sha1()
    for path in paths:
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()[:8]


def _to_wsc_llm_hardware(hardware: ResolvedHardware) -> WscLlmHardware:
    return WscLlmHardware(
        mesh_rows=hardware.mesh_rows,
        mesh_cols=hardware.mesh_cols,
        local_hbm_capacity_bytes=hardware.local_hbm_capacity_bytes,
        local_hbm_bandwidth_gbps=hardware.local_hbm_bandwidth_gbps,
        d2d_bandwidth_gbps=hardware.d2d_bandwidth_gbps,
        peak_perf_tflops=hardware.peak_perf_tflops,
        d2d_latency_ns=hardware.d2d_latency_ns,
        local_hbm_latency_ns=hardware.local_hbm_latency_ns,
        label=hardware.label,
    )


def _validate_no_memory_expansion(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"remote-memory config not found: {path}")
    with path.open(encoding="utf-8") as source:
        raw = json.load(source)
    if raw.get("memory-type") != "NO_MEMORY_EXPANSION":
        raise ValueError("WSC-LLM default must use NO_MEMORY_EXPANSION")


def select_first_session_requests(
    requests: Sequence[RequestSpec],
    session_limit: int,
) -> tuple[tuple[RequestSpec, ...], tuple[str, ...]]:
    """Select all rows for the first N distinct sessions in source row order."""

    if (
        isinstance(session_limit, bool)
        or not isinstance(session_limit, int)
        or session_limit < 0
    ):
        raise ValueError("request_queue_session_limit must be a non-negative integer")
    session_ids = tuple(
        dict.fromkeys(request.session_id for request in requests)
    )
    if session_limit > len(session_ids):
        raise ValueError(
            "request_queue_session_limit exceeds available sessions: "
            f"requested {session_limit}, available {len(session_ids)}"
        )
    selected_session_ids = (
        session_ids if session_limit == 0 else session_ids[:session_limit]
    )
    selected_set = set(selected_session_ids)
    selected_requests = tuple(
        request for request in requests if request.session_id in selected_set
    )
    return selected_requests, selected_session_ids


def load_wsc_llm_trace_config(config_csv: Path = CONFIG_CSV_PATH) -> WscLlmTraceConfig:
    if not config_csv.exists():
        raise FileNotFoundError(f"trace config CSV not found: {config_csv}")
    values: dict[str, str] = {}
    groups: list[WscLlmInferenceGroup] = []
    with config_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"trace config CSV is empty: {config_csv}")
        columns = {column.strip() for column in reader.fieldnames if column is not None}
        missing_columns = [column for column in CONFIG_COLUMNS if column not in columns]
        if missing_columns:
            raise ValueError(
                f"trace config CSV is missing columns: {', '.join(missing_columns)}"
            )
        for line_number, raw_row in enumerate(reader, start=2):
            row = clean_csv_row(raw_row)
            kind = row.get("kind", "").lower()
            if not kind or kind.startswith("#") or kind == "comment":
                continue
            if kind == "config":
                key = row.get("key", "")
                if key not in SUPPORTED_CONFIG_KEYS:
                    raise ValueError(f"line {line_number}: unsupported config key {key!r}")
                if key in values:
                    raise ValueError(f"line {line_number}: duplicate config key {key!r}")
                values[key] = row.get("value", "")
                continue
            if kind == "inference_group":
                name = row.get("group_name", "")
                pg_name = row.get("pg_name", "")
                phase_role = row.get("phase_role", "").lower()
                if not name or not pg_name or not phase_role:
                    raise ValueError(
                        f"line {line_number}: inference_group requires name, pg_name, "
                        "and phase_role"
                    )
                if phase_role not in {PREFILL_ROLE, DECODE_ROLE}:
                    raise ValueError(
                        f"line {line_number}: phase_role must be prefill or decode"
                    )
                groups.append(
                    WscLlmInferenceGroup(
                        name=name,
                        pg_name=pg_name,
                        ranks=parse_rank_spec(row.get("ranks", "")),
                        phase_role=phase_role,
                    )
                )
                continue
            raise ValueError(f"line {line_number}: unsupported row kind {kind!r}")

    missing = [key for key in REQUIRED_CONFIG_KEYS if key not in values]
    if missing:
        raise ValueError(f"trace config CSV is missing keys: {', '.join(missing)}")
    if not groups:
        raise ValueError("trace config requires at least one inference_group")
    parsed = {
        key: _parse_config_value(key, values.get(key, default_value))
        for key, default_value in {
            **{key: values[key] for key in REQUIRED_CONFIG_KEYS},
            **OPTIONAL_CONFIG_DEFAULTS,
        }.items()
    }

    request_queue_csv = _resolve_request_queue(parsed["request_queue_csv"])
    source_request_queue = load_request_queue(request_queue_csv)
    request_queue, selected_session_ids = select_first_session_requests(
        source_request_queue,
        int(parsed["request_queue_session_limit"]),
    )
    source_session_count = len(
        {request.session_id for request in source_request_queue}
    )
    hardware_path = _resolve_from_sh_test(parsed["hardware_config"])
    hardware_capacity_profile = str(parsed["local_hbm_capacity_profile"])
    resolved_hardware = load_hardware_config(
        hardware_path,
        hardware_capacity_profile,
    )
    hardware = _to_wsc_llm_hardware(resolved_hardware)
    hardware_metadata = resolved_hardware.metadata
    npus_count = resolved_hardware.npus_count
    system_template = _resolve_from_sh_test(parsed["system_template"])
    runtime_config_dir = (
        SH_TEST_DIR
        / "generated"
        / "runtime_config"
        / (
            f"{resolved_hardware.slug}__{hardware_capacity_profile}__"
            f"{resolved_hardware.remote_memory_runtime_label}"
        )
    )
    runtime_configs = materialize_runtime_configs(
        hardware=resolved_hardware,
        system_template_path=system_template,
        inference_groups=tuple((group.pg_name, group.ranks) for group in groups),
        output_dir=runtime_config_dir,
    )
    _validate_no_memory_expansion(runtime_configs.remote_memory)
    configuration_digest = _configuration_digest(
        (
            config_csv.resolve(),
            hardware_path,
            system_template,
        )
    )

    return WscLlmTraceConfig(
        config_csv=config_csv.resolve(),
        npus_count=npus_count,
        layers=int(parsed["layers"]),
        hidden_size=int(parsed["hidden_size"]),
        ffn_size=int(parsed["ffn_size"]),
        num_heads=int(parsed["num_heads"]),
        vocab_size=int(parsed["vocab_size"]),
        bytes_per_elem=int(parsed["bytes_per_elem"]),
        model_name=str(parsed["model_name"]),
        mlp_variant=str(parsed["mlp_variant"]),
        output_prefix=str(parsed["output_prefix"]),
        output_dir=parsed["output_dir"],
        request_queue_csv=request_queue_csv,
        request_queue=request_queue,
        hardware_config=hardware_path,
        hardware_capacity_profile=hardware_capacity_profile,
        hardware=hardware,
        hardware_metadata=hardware_metadata,
        system_template=system_template,
        system_config=runtime_configs.system,
        network_config=runtime_configs.network,
        comm_group_config=runtime_configs.comm_group,
        remote_memory_config=runtime_configs.remote_memory,
        remote_operand_loads=bool(parsed["remote_operand_loads"]),
        inference_groups=tuple(groups),
        request_queue_session_limit=int(parsed["request_queue_session_limit"]),
        selected_session_ids=selected_session_ids,
        source_request_count=len(source_request_queue),
        source_session_count=source_session_count,
        trace_granularity=str(parsed["trace_granularity"]),
        prefill_chunk_size=int(parsed["prefill_chunk_size"]),
        kv_cache_policy=str(parsed["kv_cache_policy"]),
        kv_reserve_context_tokens=int(parsed["kv_reserve_context_tokens"]),
        record_planning_iterations=bool(parsed["record_planning_iterations"]),
        configuration_digest=configuration_digest,
    )


def _to_scheduler_requests(requests: Sequence[RequestSpec]) -> tuple[WscLlmRequest, ...]:
    return tuple(
        WscLlmRequest(
            queue_index=index,
            session_id=request.session_id,
            turn_index=request.turn_index,
            request_id=request.request_id,
            prefill_length=request.prefill_length,
            decode_length=request.decode_length,
            session_arrival_time_ns=request.session_arrival_time_ns,
            inter_request_interval_ns=request.inter_request_interval_ns,
        )
        for index, request in enumerate(requests)
    )


def build_wsc_llm_plan(config: WscLlmTraceConfig) -> WscLlmPlan:
    specs = tuple(
        WscLlmInstanceSpec(
            name=group.name,
            pg_name=group.pg_name,
            ranks=group.ranks,
            phase_role=group.phase_role,
        )
        for group in config.inference_groups
    )
    return plan_wsc_llm_requests(
        hardware=config.hardware,
        model=config.model,
        instance_specs=specs,
        requests=_to_scheduler_requests(config.request_queue),
        p_chunk=(
            config.prefill_chunk_size
            if config.kv_cache_policy == "session_lru_recompute"
            else None
        ),
        kv_cache_policy=config.kv_cache_policy,
        reserve_context_tokens=config.kv_reserve_context_tokens,
        record_planning_iterations=config.record_planning_iterations,
    )


def build_trace_label(config: WscLlmTraceConfig, plan: WscLlmPlan) -> str:
    session_count = len({request.session_id for request in config.request_queue})
    prefill_label = range_label(
        tuple(request.prefill_length for request in config.request_queue)
    )
    decode_label = range_label(tuple(request.decode_length for request in config.request_queue))
    hardware_tag = str(config.hardware_metadata["slug"]).replace("-", "_")
    return (
        f"{config.npus_count}npus_{hardware_tag}_wsc_llm_pd_6p3d_"
        f"{len(config.inference_groups)}inst_tp{len(config.inference_groups[0].ranks)}_"
        f"{session_count}sess_{len(config.request_queue)}req_"
        f"pc{plan.p_chunk}_p{prefill_label}_d{decode_label}_"
        f"g{config.trace_granularity}_q{request_queue_digest(config.request_queue)}_"
        f"c{config.configuration_digest}"
    )


def resolve_output_dir(config: WscLlmTraceConfig, plan: WscLlmPlan) -> Path:
    if config.output_dir is None:
        return SH_TEST_DIR / "generated" / f"{config.output_prefix}_{build_trace_label(config, plan)}"
    if config.output_dir.is_absolute():
        return config.output_dir
    return (PROJECT_ROOT / config.output_dir).resolve()


def _stage_tag(queue_index: int, category: int, relative_rank: int) -> int:
    return queue_index * 10000 + category + relative_rank


def _xy_route(hardware: WscLlmHardware, source: int, target: int) -> list[int]:
    row, col = divmod(source, hardware.mesh_cols)
    target_row, target_col = divmod(target, hardware.mesh_cols)
    route = [source]
    while col != target_col:
        col += 1 if target_col > col else -1
        route.append(row * hardware.mesh_cols + col)
    while row != target_row:
        row += 1 if target_row > row else -1
        route.append(row * hardware.mesh_cols + col)
    return route


def _paired_transfer(
    *,
    config: WscLlmTraceConfig,
    builders: dict[int, TraceBuilder],
    queue_index: int,
    category: int,
    name: str,
    source_group: WscLlmInferenceGroup,
    target_group: WscLlmInferenceGroup,
    total_bytes: int,
    timer_gates: Optional[dict[int, Optional[int]]] = None,
) -> list[dict[str, object]]:
    if len(source_group.ranks) != len(target_group.ranks):
        raise ValueError("WSC-LLM ET adapter requires equal TP for direct KV shard pairing")
    if source_group.name == target_group.name:
        if timer_gates is not None:
            for rank in target_group.ranks:
                builders[rank].arm_timer_gate(timer_gates.get(rank))
        return []

    # Full KV transfers retain the configured whole-head ownership.  An
    # arbitrary capacity-split remainder falls back to exact byte partitioning
    # so the total transfer volume is never rounded up.
    if total_bytes % config.num_heads == 0:
        bytes_per_head = total_bytes // config.num_heads
        bytes_by_relative_rank = [
            bytes_per_head
            * shard_extent(config.num_heads, len(source_group.ranks), index)
            for index in range(len(source_group.ranks))
        ]
    else:
        bytes_by_relative_rank = [
            shard_extent(total_bytes, len(source_group.ranks), index)
            for index in range(len(source_group.ranks))
        ]
    routes: list[dict[str, object]] = []
    for relative_index, (source, target) in enumerate(
        zip(source_group.ranks, target_group.ranks)
    ):
        if timer_gates is not None:
            builders[source].arm_timer_gate(timer_gates.get(source))
        tag = _stage_tag(queue_index, category, relative_index)
        builders[source].comm_send(
            f"{name}_send_rank{source}_to_rank{target}",
            src=source,
            dst=target,
            comm_size=bytes_by_relative_rank[relative_index],
            comm_tag=tag,
        )
        builders[target].comm_recv(
            f"{name}_recv_rank{source}_to_rank{target}",
            src=source,
            dst=target,
            comm_size=bytes_by_relative_rank[relative_index],
            comm_tag=tag,
        )
        path = _xy_route(config.hardware, source, target)
        routes.append(
            {
                "relative_shard": relative_index,
                "source_rank": source,
                "target_rank": target,
                "bytes": bytes_by_relative_rank[relative_index],
                "noc_path": path,
                "noc_hops": len(path) - 1,
            }
        )
    return routes


def _build_metadata(
    config: WscLlmTraceConfig,
    plan: WscLlmPlan,
    *,
    rank: int,
) -> GlobalMetadata:
    instance = next(
        instance for instance in plan.topology.instances if rank in instance.ranks
    )
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend(
        [
            ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
            ChakraAttr(name="model", string_val=config.model_name),
            ChakraAttr(name="mlp_variant", string_val=config.mlp_variant),
            ChakraAttr(
                name="execution_mode",
                string_val="wsc_llm_pd_disaggregated_static_et",
            ),
            ChakraAttr(name="scheduler", string_val="WSC_LLM_STATIC_PD"),
            ChakraAttr(name="hardware_case", string_val="INHERITED_54_NPU_WAFER"),
            ChakraAttr(name="npus_count", uint64_val=config.npus_count),
            ChakraAttr(name="rank", uint64_val=rank),
            ChakraAttr(name="instance_name", string_val=instance.name),
            ChakraAttr(name="instance_index", uint64_val=instance.index),
            ChakraAttr(name="pg_name", string_val=instance.pg_name),
            ChakraAttr(name="phase_role", string_val=instance.phase_role),
            ChakraAttr(name="tensor_parallel", uint64_val=instance.size),
            ChakraAttr(name="prefill_chunk_size", uint64_val=plan.p_chunk),
            ChakraAttr(
                name="trace_granularity", string_val=config.trace_granularity
            ),
            ChakraAttr(
                name="selected_request_count",
                uint64_val=len(config.request_queue),
            ),
            ChakraAttr(
                name="selected_session_count",
                uint64_val=len(config.selected_session_ids),
            ),
            ChakraAttr(name="request_queue_csv", string_val=str(config.request_queue_csv)),
            ChakraAttr(name="hardware_config", string_val=str(config.hardware_config)),
            ChakraAttr(name="static_trace_adaptation", bool_val=True),
            ChakraAttr(name="remote_memory_expansion", bool_val=False),
        ]
    )
    return metadata


def _timing_entry_dict(entry: WscLlmTimingEntry) -> dict[str, object]:
    return {
        "instance_size": entry.instance_size,
        "phase_role": entry.phase_role,
        "p_chunk": entry.p_chunk,
        "d_batch": entry.d_batch,
        "d_token": entry.d_token,
        "iteration_time_ns": entry.iteration_time_ns,
        "source": entry.source,
    }


def _allocation_dict(allocation: KVAllocation, plan: WscLlmPlan) -> dict[str, object]:
    return {
        "request_id": allocation.request_id,
        "prefill_instance_index": allocation.prefill_instance_index,
        "decode_instance_index": allocation.decode_instance_index,
        "static_route": list(allocation.static_route),
        "relevant_instance_indices_in_priority_order": list(
            allocation.relevant_instance_indices
        ),
        "total_bytes": allocation.total_bytes,
        "pieces": [
            {
                "instance_index": piece.instance_index,
                "instance_name": plan.topology.instance(piece.instance_index).name,
                "bytes": piece.bytes,
                "distance_to_decode": piece.distance_to_decode,
                "location_priority": piece.location_priority,
                "decode_to_storage_path": list(piece.path),
            }
            for piece in allocation.pieces
        ],
    }


def _request_plan_dict(
    request: RequestSpec,
    request_plan: WscLlmRequestPlan,
    plan: WscLlmPlan,
    *,
    trace_granularity: str,
    history_routes: list[dict[str, object]],
    prefill_decode_routes: list[dict[str, object]],
    kv_offload_routes: list[dict[str, object]],
) -> dict[str, object]:
    prefill_instance = plan.topology.instance(request_plan.prefill_instance_index)
    decode_instance = plan.topology.instance(request_plan.decode_instance_index)
    return {
        "queue_index": request_plan.queue_index,
        "session_id": request_plan.session_id,
        "turn_index": request_plan.turn_index,
        "request_id": request_plan.request_id,
        "prefill_length": request.prefill_length,
        "decode_length": request.decode_length,
        "trace_representation": {
            "granularity": trace_granularity,
            "prefill_chunk_count_represented": (
                request.prefill_length + plan.p_chunk - 1
            )
            // plan.p_chunk,
            "decode_step_count_represented": request.decode_length,
        },
        "history_tokens_before": request_plan.history_tokens_before,
        "prefill_context_tokens": request_plan.prefill_context_tokens,
        "final_context_tokens": request_plan.final_context_tokens,
        "planned_timing_ns": {
            "arrival": request_plan.estimated_arrival_ns,
            "prefill_start": request_plan.prefill_start_ns,
            "prefill_complete": request_plan.prefill_complete_ns,
            "decode_start": request_plan.decode_start_ns,
            "completion": request_plan.completion_ns,
        },
        "prefill_assignment": {
            "instance_index": prefill_instance.index,
            "instance_name": prefill_instance.name,
            "ranks": list(prefill_instance.ranks),
            "queue_key_before_assignment": list(request_plan.prefill_assignment_key),
            "policy": "least_request_count_including_active_head,config_order",
        },
        "decode_assignment": {
            "instance_index": decode_instance.index,
            "instance_name": decode_instance.name,
            "ranks": list(decode_instance.ranks),
            "policy": "offline_static_nearest_decode_no_runtime_reselection",
            "queue_depth_before_enqueue": (
                request_plan.decode_queue_depth_before_enqueue
            ),
            "route": list(request_plan.static_route.path),
            "hop_count": request_plan.static_route.hop_count,
            "shared_edges": [
                list(edge) for edge in request_plan.static_route.shared_edges
            ],
        },
        "kv_allocation": _allocation_dict(request_plan.kv_allocation, plan),
        "terminal_kv_release_at_completion": (
            request_plan.terminal_kv_release_at_completion
        ),
        "history_source_instance_index": request_plan.history_source_instance_index,
        "history_transfer_bytes": request_plan.history_transfer_bytes,
        "history_routes": history_routes,
        "prefill_decode_routes": prefill_decode_routes,
        "kv_offload_routes": kv_offload_routes,
        "stages": [
            "arrival_or_previous_completion_interval_gate",
            "history_kv_transfer_if_needed",
            "wsc_relevant_pd_capacity_check_and_kv_reservation",
            "chunked_prefill",
            "prefill_to_decode_kv_transfer_if_needed",
            "decode",
            "kv_storage_transfer_for_reserved_allocation",
        ],
    }


KV_CACHE_EVENT_COLUMNS = (
    "event_index", "planner_time_ns", "phase", "event_type", "reason",
    "trigger_request_id", "session_id", "source_instance_index",
    "target_instance_index", "context_tokens", "total_bytes", "shard_bytes",
    "last_completion_ns", "instance_remaining_before_bytes",
    "instance_remaining_after_bytes", "insufficient_ranks",
)


def _write_kv_events_csv(path: Path, events: Sequence[object]) -> None:
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=KV_CACHE_EVENT_COLUMNS)
        writer.writeheader()
        for event in events:
            writer.writerow({
                "event_index": event.event_index,
                "planner_time_ns": event.planner_time_ns,
                "phase": event.phase,
                "event_type": event.event_type,
                "reason": event.reason,
                "trigger_request_id": event.trigger_request_id,
                "session_id": event.session_id or "",
                "source_instance_index": "" if event.source_instance_index is None else event.source_instance_index,
                "target_instance_index": "" if event.target_instance_index is None else event.target_instance_index,
                "context_tokens": event.context_tokens,
                "total_bytes": event.total_bytes,
                "shard_bytes": json.dumps(list(event.shard_bytes)),
                "last_completion_ns": "" if event.last_completion_ns is None else event.last_completion_ns,
                "instance_remaining_before_bytes": json.dumps(list(event.instance_remaining_before_bytes)),
                "instance_remaining_after_bytes": json.dumps(list(event.instance_remaining_after_bytes)),
                "insufficient_ranks": json.dumps(list(event.insufficient_ranks)),
            })


def _atomic_publish_directory(staging: Path, output_dir: Path) -> None:
    """Publish a fully written trace directory with a single directory rename."""

    backup: Optional[Path] = None
    if output_dir.exists():
        backup = output_dir.parent / f".{output_dir.name}.previous-{uuid.uuid4().hex}"
        os.replace(output_dir, backup)
    try:
        os.replace(staging, output_dir)
    except BaseException:
        if backup is not None and backup.exists():
            os.replace(backup, output_dir)
        raise
    if backup is not None:
        shutil.rmtree(backup, ignore_errors=True)


def _emit_control_trigger(
    *,
    builders: dict[int, TraceBuilder],
    queue_index: int,
    name: str,
    source_group: WscLlmInferenceGroup,
    target_group: WscLlmInferenceGroup,
    timer_gates: dict[int, Optional[int]],
) -> None:
    """Represent an inter-instance timing dependency as one control byte."""

    if source_group.name == target_group.name:
        for rank in target_group.ranks:
            builders[rank].arm_timer_gate(timer_gates.get(rank))
        return
    source, target = source_group.ranks[0], target_group.ranks[0]
    builders[source].arm_timer_gate(timer_gates.get(source))
    tag = _stage_tag(queue_index, 1900, 0)
    builders[source].comm_send(
        f"{name}_control_send_rank{source}_to_rank{target}",
        src=source, dst=target, comm_size=1, comm_tag=tag,
    )
    builders[target].comm_recv(
        f"{name}_control_recv_rank{source}_to_rank{target}",
        src=source, dst=target, comm_size=1, comm_tag=tag,
    )


def _emit_prefill_stage(
    *,
    config: WscLlmTraceConfig,
    builders: dict[int, TraceBuilder],
    group: WscLlmInferenceGroup,
    prefix: str,
    stage: str,
    tokens: int,
    initial_context_tokens: int,
) -> dict[int, tuple[int, int]]:
    """Emit one chunked prefill stage; returns, per rank, the ids of the first
    and last *real* operator/collective nodes (the artificial one-byte
    ``*_end_barrier`` collectives are excluded), or an empty mapping when the
    stage has no tokens.  Used only for metrics boundary recording (doc
    sec.6.2/6.3); the emitted nodes are unchanged."""

    if tokens == 0:
        return {}
    bounds: dict[int, list[int]] = {}
    spans: list[tuple[int, int]] = []
    processed = 0
    while processed < tokens:
        chunk = min(config.prefill_chunk_size, tokens - processed)
        spans.append((chunk, initial_context_tokens + processed + chunk))
        processed += chunk
    tp = len(group.ranks)
    if config.trace_granularity == "token_expanded":
        for chunk_index, (chunk, kv_length) in enumerate(spans):
            for relative_rank, rank in enumerate(group.ranks):
                first_node_id = builders[rank].next_id
                transformer_pass(
                    builders[rank], phase=f"{prefix}_{stage}_chunk{chunk_index:04d}",
                    tokens=chunk, kv_length=kv_length, layers=config.layers,
                    hidden_size=config.hidden_size, ffn_size=config.ffn_size,
                    tensor_parallel=tp, pg_name=group.pg_name,
                    vocab_size=config.vocab_size, bytes_per_elem=config.bytes_per_elem,
                    num_heads=config.num_heads, tensor_parallel_rank=relative_rank,
                    mlp_variant=config.mlp_variant,
                )
                last_node_id = builders[rank].previous_id
                if rank in bounds:
                    bounds[rank][1] = last_node_id
                else:
                    bounds[rank] = [first_node_id, last_node_id]
                builders[rank].all_reduce(
                    f"{prefix}_{stage}_chunk{chunk_index:04d}_end_barrier", 1, group.pg_name
                )
    else:
        for relative_rank, rank in enumerate(group.ranks):
            first_node_id = builders[rank].next_id
            pass_count = transformer_pass_aggregated(
                builders[rank], phase=f"{prefix}_{stage}_request_aggregated",
                pass_spans=spans, layers=config.layers, hidden_size=config.hidden_size,
                ffn_size=config.ffn_size, tensor_parallel=tp, pg_name=group.pg_name,
                vocab_size=config.vocab_size, bytes_per_elem=config.bytes_per_elem,
                num_heads=config.num_heads, tensor_parallel_rank=relative_rank,
                mlp_variant=config.mlp_variant,
            )
            bounds[rank] = [first_node_id, builders[rank].previous_id]
            builders[rank].all_reduce(
                f"{prefix}_{stage}_chunks_aggregated_end_barrier", pass_count, group.pg_name
            )
    return {rank: (pair[0], pair[1]) for rank, pair in bounds.items()}


def _hbm_snapshot_dict(snapshot: object) -> dict[str, object]:
    return {
        "rank": snapshot.rank, "instance_index": snapshot.instance_index,
        "capacity_bytes": snapshot.capacity_bytes,
        "model_weight_bytes": snapshot.model_weight_bytes,
        "resident_kv_bytes": snapshot.resident_kv_bytes,
        "reserved_request_bytes": snapshot.reserved_request_bytes,
        "used_bytes": snapshot.used_bytes, "remaining_bytes": snapshot.remaining_bytes,
    }


def _eviction_dict(record: object) -> dict[str, object]:
    return {
        "time_ns": record.time_ns, "phase": record.phase, "reason": record.reason,
        "trigger_request_id": record.trigger_request_id,
        "victim_session_id": record.victim_session_id,
        "victim_instance_index": record.victim_instance_index,
        "victim_last_completion_ns": record.victim_last_completion_ns,
        "context_tokens": record.context_tokens, "shard_bytes": list(record.shard_bytes),
    }


def _session_lru_record(
    request: RequestSpec,
    request_plan: WscLlmRequestPlan,
    history_routes: list[dict[str, object]],
    prefill_decode_routes: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "queue_index": request_plan.queue_index,
        "session_id": request_plan.session_id,
        "turn_index": request_plan.turn_index,
        "request_id": request_plan.request_id,
        "prefill_length": request.prefill_length,
        "decode_length": request.decode_length,
        "history_tokens_before": request_plan.history_tokens_before,
        "history_cache_state_before": request_plan.history_cache_state_before,
        "history_action": request_plan.history_action,
        "history_source_instance_index": request_plan.history_source_instance_index,
        "history_target_instance_index": request_plan.prefill_instance_index,
        "history_transfer_bytes": request_plan.history_transfer_bytes,
        "history_recompute_tokens": request_plan.history_recompute_tokens,
        "effective_prefill_tokens": request_plan.effective_prefill_tokens,
        "prefill_context_tokens": request_plan.prefill_context_tokens,
        "final_context_tokens": request_plan.final_context_tokens,
        "prefill_instance_index": request_plan.prefill_instance_index,
        "decode_instance_index": request_plan.decode_instance_index,
        "static_route": list(request_plan.static_route.path),
        "history_routes": history_routes,
        "prefill_decode_routes": prefill_decode_routes,
        "terminal_kv_release_at_completion": False,
        "evicted_sessions_before_prefill": [_eviction_dict(record) for record in request_plan.admission_evictions],
        "evicted_sessions_before_decode": [_eviction_dict(record) for record in request_plan.decode_target_evictions],
        "evicted_sessions_after_completion": [_eviction_dict(record) for record in request_plan.completion_evictions],
        "kv_state_after_completion": request_plan.kv_state_after_completion,
        "kv_instance_after_completion": request_plan.kv_instance_after_completion,
        "hbm_before_request": [_hbm_snapshot_dict(item) for item in request_plan.hbm_before_request],
        "hbm_after_completion": [_hbm_snapshot_dict(item) for item in request_plan.hbm_after_completion],
        "planned_timing_ns": {
            "arrival": request_plan.estimated_arrival_ns,
            "prefill_start": request_plan.prefill_start_ns,
            "prefill_complete": request_plan.prefill_complete_ns,
            "decode_start": request_plan.decode_start_ns,
            "completion": request_plan.completion_ns,
        },
    }


def _write_wsc_session_lru_trace(
    config: WscLlmTraceConfig,
    plan: WscLlmPlan,
    *,
    metrics: Optional[ServiceMetrics] = None,
) -> None:
    output_dir = resolve_output_dir(config, plan)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{config.output_prefix}.staging-", dir=output_dir.parent))
    outputs: dict[int, object] = {}
    try:
        group_by_index = dict(enumerate(config.inference_groups))
        builders: dict[int, TraceBuilder] = {}
        for rank in range(config.npus_count):
            output = (staging / f"{config.output_prefix}.{rank}.et").open("wb")
            outputs[rank] = output
            encode_message(output, _build_metadata(config, plan, rank=rank))
            builders[rank] = TraceBuilder(
                remote_operand_loads=config.remote_operand_loads,
                node_sink=lambda node, output=output: encode_message(output, node),
                retain_nodes=False,
            )
        request_by_queue = dict(enumerate(config.request_queue))
        completion_gates: dict[str, tuple[int, dict[int, Optional[int]]]] = {}
        records: list[dict[str, object]] = []
        for request_plan in sorted(plan.requests, key=lambda item: (item.prefill_start_ns, item.queue_index)):
            request = request_by_queue[request_plan.queue_index]
            prefill_group = group_by_index[request_plan.prefill_instance_index]
            decode_group = group_by_index[request_plan.decode_instance_index]
            prefix = (
                f"q{request_plan.queue_index:04d}_{sanitize_node_prefix(request_plan.session_id)}_"
                f"turn{request_plan.turn_index}_{sanitize_node_prefix(request_plan.request_id)}"
            )
            history_routes: list[dict[str, object]] = []
            if request_plan.turn_index == 0:
                if request.session_arrival_time_ns is None:
                    raise RuntimeError("first request lost its session arrival")
                timers = {
                    rank: builders[rank].timer_gate(f"{prefix}_arrival_timer_gate", request.session_arrival_time_ns)
                    for rank in prefill_group.ranks
                }
                for rank in prefill_group.ranks:
                    builders[rank].arm_timer_gate(timers[rank])
            else:
                previous = completion_gates.get(request_plan.session_id)
                if previous is None or request.inter_request_interval_ns is None:
                    raise RuntimeError("later request has no completion interval gate")
                previous_index, previous_nodes = previous
                previous_group = group_by_index[previous_index]
                timers = {
                    rank: builders[rank].timer_gate(
                        f"{prefix}_interval_timer_gate", request.inter_request_interval_ns,
                        after_node_id=previous_nodes.get(rank),
                    ) for rank in previous_group.ranks
                }
                if request_plan.history_action == NOC_MIGRATE:
                    source_index = request_plan.history_source_instance_index
                    if source_index is None:
                        raise RuntimeError("NoC history action has no source")
                    source_group = group_by_index[source_index]
                    if source_group.name != previous_group.name:
                        _emit_control_trigger(
                            builders=builders, queue_index=request_plan.queue_index,
                            name=f"{prefix}_history_source_control", source_group=previous_group,
                            target_group=source_group, timer_gates=timers,
                        )
                        timers = {rank: None for rank in source_group.ranks}
                    history_piece_routes = _paired_transfer(
                        config=config, builders=builders, queue_index=request_plan.queue_index,
                        category=1000, name=f"{prefix}_history_kv", source_group=source_group,
                        target_group=prefill_group, total_bytes=request_plan.history_transfer_bytes,
                        timer_gates=timers,
                    )
                    history_routes.extend(history_piece_routes)
                    if metrics is not None:
                        # Doc sec.4.3 code 7: transfer-arrival memory anchor =
                        # the last transfer node on each target rank.
                        for route in history_piece_routes:
                            target_rank = int(route["target_rank"])
                            metrics.add_event(
                                target_rank,
                                builders[target_rank].previous_id,
                                EVENT_MEMORY_ANCHOR_COMPLETE,
                                request_plan.queue_index,
                            )
                else:
                    _emit_control_trigger(
                        builders=builders, queue_index=request_plan.queue_index,
                        name=f"{prefix}_interval_control", source_group=previous_group,
                        target_group=prefill_group, timer_gates=timers,
                    )
            # All six TP ranks wait for history transfer/control/recompute
            # readiness before entering current-prompt Prefill.
            recompute_bounds: dict[int, tuple[int, int]] = {}
            if request_plan.history_action == RECOMPUTE:
                recompute_bounds = _emit_prefill_stage(
                    config=config, builders=builders, group=prefill_group, prefix=prefix,
                    stage="history_recompute", tokens=request_plan.history_recompute_tokens,
                    initial_context_tokens=0,
                )
            for rank in prefill_group.ranks:
                builders[rank].all_reduce(f"{prefix}_history_tp_ready_barrier", 1, prefill_group.pg_name)
            prefill_bounds = _emit_prefill_stage(
                config=config, builders=builders, group=prefill_group, prefix=prefix,
                stage="current_prefill", tokens=request.prefill_length,
                initial_context_tokens=request_plan.history_tokens_before,
            )
            if metrics is not None:
                for rank in prefill_group.ranks:
                    # Doc sec.6.2: with history recompute the Prefill start is
                    # the first real recompute node; sec.6.3: the Prefill end
                    # is the last real prefill node, not the end barrier.
                    start_pair = recompute_bounds.get(rank, prefill_bounds[rank])
                    metrics.add_event(
                        rank,
                        start_pair[0],
                        EVENT_PREFILL_START,
                        request_plan.queue_index,
                    )
                    metrics.add_event(
                        rank,
                        prefill_bounds[rank][1],
                        EVENT_PREFILL_END,
                        request_plan.queue_index,
                    )

            prefill_decode_routes = _paired_transfer(
                config=config, builders=builders, queue_index=request_plan.queue_index,
                category=3000, name=f"{prefix}_prefill_to_decode_kv",
                source_group=prefill_group, target_group=decode_group,
                total_bytes=kv_cache_bytes_for_tokens(config.model, request_plan.prefill_context_tokens),
            )
            if metrics is not None:
                for route in prefill_decode_routes:
                    target_rank = int(route["target_rank"])
                    metrics.add_event(
                        target_rank,
                        builders[target_rank].previous_id,
                        EVENT_MEMORY_ANCHOR_COMPLETE,
                        request_plan.queue_index,
                    )
            tp = len(decode_group.ranks)
            if config.trace_granularity == "token_expanded":
                for relative_rank, rank in enumerate(decode_group.ranks):
                    decode_first_node = builders[rank].next_id
                    for step in range(request.decode_length):
                        transformer_pass(
                            builders[rank], phase=f"{prefix}_decode{step:04d}", tokens=1,
                            kv_length=request_plan.prefill_context_tokens + step + 1,
                            layers=config.layers, hidden_size=config.hidden_size,
                            ffn_size=config.ffn_size, tensor_parallel=tp,
                            pg_name=decode_group.pg_name, vocab_size=config.vocab_size,
                            bytes_per_elem=config.bytes_per_elem, num_heads=config.num_heads,
                            tensor_parallel_rank=relative_rank, mlp_variant=config.mlp_variant,
                        )
                    # Doc sec.6.4: completion candidate = per-rank previous_id
                    # saved before the decode_request_end_barrier.
                    decode_last_node = builders[rank].previous_id
                    builders[rank].all_reduce(f"{prefix}_decode_request_end_barrier", 1, decode_group.pg_name)
                    if metrics is not None:
                        metrics.add_event(
                            rank,
                            decode_first_node,
                            EVENT_DECODE_START,
                            request_plan.queue_index,
                        )
                        metrics.add_event(
                            rank,
                            decode_last_node,
                            EVENT_DECODE_END,
                            request_plan.queue_index,
                        )
            else:
                spans = tuple((1, request_plan.prefill_context_tokens + step + 1) for step in range(request.decode_length))
                for relative_rank, rank in enumerate(decode_group.ranks):
                    decode_first_node = builders[rank].next_id
                    transformer_pass_aggregated(
                        builders[rank], phase=f"{prefix}_decode_request_aggregated", pass_spans=spans,
                        layers=config.layers, hidden_size=config.hidden_size,
                        ffn_size=config.ffn_size, tensor_parallel=tp, pg_name=decode_group.pg_name,
                        vocab_size=config.vocab_size, bytes_per_elem=config.bytes_per_elem,
                        num_heads=config.num_heads, tensor_parallel_rank=relative_rank,
                        mlp_variant=config.mlp_variant,
                    )
                    decode_last_node = builders[rank].previous_id
                    builders[rank].all_reduce(f"{prefix}_decode_request_end_barrier", 1, decode_group.pg_name)
                    if metrics is not None:
                        metrics.add_event(
                            rank,
                            decode_first_node,
                            EVENT_DECODE_START,
                            request_plan.queue_index,
                        )
                        metrics.add_event(
                            rank,
                            decode_last_node,
                            EVENT_DECODE_END,
                            request_plan.queue_index,
                        )
            completion_gates[request_plan.session_id] = (
                request_plan.decode_instance_index,
                {rank: builders[rank].previous_id for rank in decode_group.ranks},
            )
            records.append(_session_lru_record(request, request_plan, history_routes, prefill_decode_routes))

        for output in outputs.values():
            output.close()
        outputs.clear()
        plan.timing_lut.export_csv(staging / "wsc_llm_timing_lut.csv")
        _write_kv_events_csv(staging / "kv_cache_events.csv", plan.kv_events)
        records.sort(key=lambda record: int(record["queue_index"]))
        final_resident = sum(item.state == "RESIDENT" for item in plan.final_session_snapshots)
        event_counts = {
            name: sum(event.event_type == name for event in plan.kv_events)
            for name in ("local_hit", "noc_migrate", "recompute", "evict_delete", "admission_blocked")
        }
        if metrics is not None:
            metrics.write_manifest(
                output_dir=staging,
                config=config,
                plan=plan,
                node_count_by_rank={
                    rank: builder.node_count for rank, builder in builders.items()
                },
                et_paths_by_rank={
                    rank: staging / f"{config.output_prefix}.{rank}.et"
                    for rank in builders
                },
                request_records=records,
                kv_digest_payload=kv_event_payload_session_lru(plan.kv_events),
            )
        manifest = {
            "output_prefix": str(output_dir / config.output_prefix),
            "execution_mode": "wsc_llm_session_lru_recompute_static_et",
            "trace_granularity": config.trace_granularity,
            "request_queue_csv": str(config.request_queue_csv),
            "selected_request_count": len(config.request_queue),
            "selected_session_count": len(config.selected_session_ids),
            "prefill_chunk_size": plan.p_chunk,
            "planning_iteration_count": plan.planning_iteration_count,
            "planning_iterations": [],
            "requests": records,
            "kv_cache_events_csv": "kv_cache_events.csv",
            "static_pd_mapping": {
                "runtime_decode_reselection": False,
                "routes": [
                    {"prefill_instance_index": route.prefill_instance_index,
                     "decode_instance_index": route.decode_instance_index,
                     "instance_path": list(route.path)}
                    for route in plan.static_mapping.routes
                ],
            },
            "kv_management": {
                "policy": "session_lru_recompute",
                "reserve_context_tokens": plan.reserve_context_tokens,
                "watermark_scope": "per_physical_npu_exact_tp_shard",
                "eviction_order": "last_completion_ns_then_session_id",
                "delete_cost": "zero", "remote_memory_used": False,
                "terminal_kv_release_at_completion": False,
                "local_hit_count": event_counts["local_hit"],
                "noc_migrate_count": event_counts["noc_migrate"],
                "recompute_count": event_counts["recompute"],
                "eviction_count": event_counts["evict_delete"],
                "admission_blocked_count": event_counts["admission_blocked"],
                "final_resident_session_count": final_resident,
                "final_evicted_session_count": len(plan.final_session_snapshots) - final_resident,
                "final_hbm_snapshots": [_hbm_snapshot_dict(item) for item in plan.final_hbm_snapshots],
                "final_session_snapshots": [
                    {
                        "session_id": item.session_id,
                        "logical_context_tokens": item.logical_context_tokens,
                        "state": item.state,
                        "instance_index": item.instance_index,
                        "shard_bytes": list(item.shard_bytes),
                        "last_completion_ns": item.last_completion_ns,
                        "active": item.active,
                        "last_request_id": item.last_request_id,
                        "evicted_at_ns": item.evicted_at_ns,
                        "evicted_by_request_id": item.evicted_by_request_id,
                    }
                    for item in plan.final_session_snapshots
                ],
            },
            "hardware": {"capacity_profile": config.hardware_capacity_profile,
                         "local_hbm_capacity_bytes_per_npu": config.hardware.local_hbm_capacity_bytes},
            "nodes_per_rank": {str(rank): builder.node_count for rank, builder in builders.items()},
            "total_nodes": sum(builder.node_count for builder in builders.values()),
            "remote_memory": {"memory_type": "NO_MEMORY_EXPANSION", "used": False},
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        _atomic_publish_directory(staging, output_dir)
        print(json.dumps({"output_dir": str(output_dir), "requests": len(records), "events": len(plan.kv_events), "nodes": manifest["total_nodes"]}))
    finally:
        for output in outputs.values():
            try:
                output.close()
            except OSError:
                pass
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


def write_wsc_llm_trace(
    config: WscLlmTraceConfig,
    plan: Optional[WscLlmPlan] = None,
    *,
    metrics: Optional[ServiceMetrics] = None,
) -> None:
    """Write ET from a supplied plan, or build one for programmatic callers."""

    if plan is None:
        if metrics is not None:
            session_kv_manager.set_metrics_observer(metrics.memory)
        try:
            plan = build_wsc_llm_plan(config)
        finally:
            session_kv_manager.set_metrics_observer(None)
    if config.kv_cache_policy == "session_lru_recompute":
        _write_wsc_session_lru_trace(config, plan, metrics=metrics)
        return
    if metrics is not None:
        # The legacy planner does not run the session KV manager; mirror the
        # preloaded model weights directly (doc sec.7.3, anchor tick 0).
        metrics.record_weight_preload(config, plan)
    output_dir = resolve_output_dir(config, plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    timing_lut_path = output_dir / "wsc_llm_timing_lut.csv"
    plan.timing_lut.export_csv(timing_lut_path)

    group_by_index = {
        index: group for index, group in enumerate(config.inference_groups)
    }
    builders = {
        rank: TraceBuilder(remote_operand_loads=config.remote_operand_loads)
        for rank in range(config.npus_count)
    }
    request_by_queue = {
        index: request for index, request in enumerate(config.request_queue)
    }
    plans_by_session: dict[str, list[WscLlmRequestPlan]] = {}
    for request_plan in plan.requests:
        plans_by_session.setdefault(request_plan.session_id, []).append(request_plan)
    for session_plans in plans_by_session.values():
        session_plans.sort(key=lambda item: item.turn_index)
    next_plan: dict[str, Optional[WscLlmRequestPlan]] = {}
    for session_plans in plans_by_session.values():
        for current, following in zip(session_plans, session_plans[1:]):
            next_plan[current.request_id] = following
        next_plan[session_plans[-1].request_id] = None

    pending_history: dict[str, tuple[HistoryPieceGate, ...]] = {}
    for request_plan in plan.requests:
        if request_plan.turn_index != 0:
            continue
        group = group_by_index[request_plan.prefill_instance_index]
        request = request_by_queue[request_plan.queue_index]
        arrival = request.session_arrival_time_ns
        if arrival is None:
            raise RuntimeError("first request lost its session arrival time")
        prefix = (
            f"q{request_plan.queue_index:04d}_"
            f"{sanitize_node_prefix(request_plan.request_id)}"
        )
        timers = {
            rank: builders[rank].timer_gate(
                f"{prefix}_global_arrival_timer_gate",
                arrival,
            )
            for rank in group.ranks
        }
        pending_history[request_plan.request_id] = (
            HistoryPieceGate(request_plan.prefill_instance_index, 0, timers),
        )

    request_records: list[dict[str, object]] = []
    ordered_plans = sorted(
        plan.requests,
        key=lambda item: (item.prefill_start_ns, item.queue_index),
    )
    for request_plan in ordered_plans:
        request = request_by_queue[request_plan.queue_index]
        prefill_group = group_by_index[request_plan.prefill_instance_index]
        decode_group = group_by_index[request_plan.decode_instance_index]
        prefix = (
            f"q{request_plan.queue_index:04d}_"
            f"{sanitize_node_prefix(request_plan.session_id)}_"
            f"turn{request_plan.turn_index}_"
            f"{sanitize_node_prefix(request_plan.request_id)}"
        )

        gates = pending_history.pop(request_plan.request_id, None)
        if gates is None:
            raise RuntimeError(f"request {request_plan.request_id} has no arrival/history gate")
        history_routes: list[dict[str, object]] = []
        for piece_number, gate in enumerate(gates):
            source_group = group_by_index[gate.source_instance_index]
            if gate.total_bytes == 0 and source_group.name == prefill_group.name:
                for rank in prefill_group.ranks:
                    builders[rank].arm_timer_gate(gate.timer_gates.get(rank))
                continue
            history_piece_routes = _paired_transfer(
                config=config,
                builders=builders,
                queue_index=request_plan.queue_index,
                category=1000 + piece_number * 100,
                name=f"{prefix}_history_piece{piece_number}",
                source_group=source_group,
                target_group=prefill_group,
                total_bytes=gate.total_bytes,
                timer_gates=gate.timer_gates,
            )
            history_routes.extend(history_piece_routes)
            if metrics is not None:
                # Doc sec.4.3 code 7: transfer-arrival memory anchor = the
                # last transfer node on each target rank.
                for route in history_piece_routes:
                    target_rank = int(route["target_rank"])
                    metrics.add_event(
                        target_rank,
                        builders[target_rank].previous_id,
                        EVENT_MEMORY_ANCHOR_COMPLETE,
                        request_plan.queue_index,
                    )

        prefill_first_nodes: dict[int, int] = {}
        prefill_last_nodes: dict[int, int] = {}
        tensor_parallel = len(prefill_group.ranks)
        if config.trace_granularity == "token_expanded":
            processed = 0
            chunk_index = 0
            while processed < request.prefill_length:
                chunk_tokens = min(plan.p_chunk, request.prefill_length - processed)
                kv_length = (
                    request_plan.history_tokens_before + processed + chunk_tokens
                )
                for relative_rank, rank in enumerate(prefill_group.ranks):
                    if rank not in prefill_first_nodes:
                        prefill_first_nodes[rank] = builders[rank].next_id
                    transformer_pass(
                        builders[rank],
                        phase=f"{prefix}_prefill_chunk{chunk_index:04d}",
                        tokens=chunk_tokens,
                        kv_length=kv_length,
                        layers=config.layers,
                        hidden_size=config.hidden_size,
                        ffn_size=config.ffn_size,
                        tensor_parallel=tensor_parallel,
                        pg_name=prefill_group.pg_name,
                        vocab_size=config.vocab_size,
                        bytes_per_elem=config.bytes_per_elem,
                        num_heads=config.num_heads,
                        tensor_parallel_rank=relative_rank,
                        mlp_variant=config.mlp_variant,
                    )
                    # Doc sec.6.3: last real prefill node before the one-byte
                    # prefill end barrier.
                    prefill_last_nodes[rank] = builders[rank].previous_id
                    builders[rank].all_reduce(
                        f"{prefix}_prefill_chunk{chunk_index:04d}_end_barrier",
                        1,
                        prefill_group.pg_name,
                    )
                processed += chunk_tokens
                chunk_index += 1
        else:
            prefill_spans: list[tuple[int, int]] = []
            processed = 0
            while processed < request.prefill_length:
                chunk_tokens = min(plan.p_chunk, request.prefill_length - processed)
                prefill_spans.append(
                    (
                        chunk_tokens,
                        request_plan.history_tokens_before
                        + processed
                        + chunk_tokens,
                    )
                )
                processed += chunk_tokens
            for relative_rank, rank in enumerate(prefill_group.ranks):
                prefill_first_nodes[rank] = builders[rank].next_id
                pass_count = transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_prefill_request_aggregated",
                    pass_spans=prefill_spans,
                    layers=config.layers,
                    hidden_size=config.hidden_size,
                    ffn_size=config.ffn_size,
                    tensor_parallel=tensor_parallel,
                    pg_name=prefill_group.pg_name,
                    vocab_size=config.vocab_size,
                    bytes_per_elem=config.bytes_per_elem,
                    num_heads=config.num_heads,
                    tensor_parallel_rank=relative_rank,
                    mlp_variant=config.mlp_variant,
                )
                prefill_last_nodes[rank] = builders[rank].previous_id
                builders[rank].all_reduce(
                    f"{prefix}_prefill_chunks_aggregated_end_barrier",
                    pass_count,
                    prefill_group.pg_name,
                )
        if metrics is not None:
            for rank in prefill_group.ranks:
                metrics.add_event(
                    rank,
                    prefill_first_nodes[rank],
                    EVENT_PREFILL_START,
                    request_plan.queue_index,
                )
                metrics.add_event(
                    rank,
                    prefill_last_nodes[rank],
                    EVENT_PREFILL_END,
                    request_plan.queue_index,
                )

        prefill_decode_routes = _paired_transfer(
            config=config,
            builders=builders,
            queue_index=request_plan.queue_index,
            category=3000,
            name=f"{prefix}_prefill_to_decode_kv",
            source_group=prefill_group,
            target_group=decode_group,
            total_bytes=kv_cache_bytes_for_tokens(
                config.model, request_plan.prefill_context_tokens
            ),
        )
        if metrics is not None:
            for route in prefill_decode_routes:
                target_rank = int(route["target_rank"])
                metrics.add_event(
                    target_rank,
                    builders[target_rank].previous_id,
                    EVENT_MEMORY_ANCHOR_COMPLETE,
                    request_plan.queue_index,
                )

        tensor_parallel = len(decode_group.ranks)
        if config.trace_granularity == "token_expanded":
            for relative_rank, rank in enumerate(decode_group.ranks):
                decode_first_node = builders[rank].next_id
                for step in range(request.decode_length):
                    transformer_pass(
                        builders[rank],
                        phase=f"{prefix}_decode{step:04d}",
                        tokens=1,
                        kv_length=request_plan.prefill_context_tokens + step + 1,
                        layers=config.layers,
                        hidden_size=config.hidden_size,
                        ffn_size=config.ffn_size,
                        tensor_parallel=tensor_parallel,
                        pg_name=decode_group.pg_name,
                        vocab_size=config.vocab_size,
                        bytes_per_elem=config.bytes_per_elem,
                        num_heads=config.num_heads,
                        tensor_parallel_rank=relative_rank,
                        mlp_variant=config.mlp_variant,
                    )
                # Doc sec.6.4: completion candidate = per-rank previous_id
                # saved before the decode_request_end_barrier.
                decode_last_node = builders[rank].previous_id
                builders[rank].all_reduce(
                    f"{prefix}_decode_request_end_barrier",
                    1,
                    decode_group.pg_name,
                )
                if metrics is not None:
                    metrics.add_event(
                        rank,
                        decode_first_node,
                        EVENT_DECODE_START,
                        request_plan.queue_index,
                    )
                    metrics.add_event(
                        rank,
                        decode_last_node,
                        EVENT_DECODE_END,
                        request_plan.queue_index,
                    )
        else:
            decode_spans = tuple(
                (1, request_plan.prefill_context_tokens + step + 1)
                for step in range(request.decode_length)
            )
            for relative_rank, rank in enumerate(decode_group.ranks):
                decode_first_node = builders[rank].next_id
                transformer_pass_aggregated(
                    builders[rank],
                    phase=f"{prefix}_decode_request_aggregated",
                    pass_spans=decode_spans,
                    layers=config.layers,
                    hidden_size=config.hidden_size,
                    ffn_size=config.ffn_size,
                    tensor_parallel=tensor_parallel,
                    pg_name=decode_group.pg_name,
                    vocab_size=config.vocab_size,
                    bytes_per_elem=config.bytes_per_elem,
                    num_heads=config.num_heads,
                    tensor_parallel_rank=relative_rank,
                    mlp_variant=config.mlp_variant,
                )
                decode_last_node = builders[rank].previous_id
                builders[rank].all_reduce(
                    f"{prefix}_decode_request_end_barrier",
                    1,
                    decode_group.pg_name,
                )
                if metrics is not None:
                    metrics.add_event(
                        rank,
                        decode_first_node,
                        EVENT_DECODE_START,
                        request_plan.queue_index,
                    )
                    metrics.add_event(
                        rank,
                        decode_last_node,
                        EVENT_DECODE_END,
                        request_plan.queue_index,
                    )

        kv_offload_routes: list[dict[str, object]] = []
        for piece_number, piece in enumerate(request_plan.kv_allocation.pieces):
            storage_group = group_by_index[piece.instance_index]
            if storage_group.name == decode_group.name:
                continue
            kv_offload_routes.extend(
                _paired_transfer(
                    config=config,
                    builders=builders,
                    queue_index=request_plan.queue_index,
                    category=5000 + piece_number * 100,
                    name=f"{prefix}_kv_allocate_piece{piece_number}",
                    source_group=decode_group,
                    target_group=storage_group,
                    total_bytes=piece.bytes,
                )
            )

        following = next_plan[request_plan.request_id]
        if following is not None:
            following_request = request_by_queue[following.queue_index]
            interval = following_request.inter_request_interval_ns
            if interval is None:
                raise RuntimeError("later request lost its inter-request interval")
            history_pieces: list[HistoryPieceGate] = []
            for piece_number, piece in enumerate(request_plan.kv_allocation.pieces):
                storage_group = group_by_index[piece.instance_index]
                timers = {
                    rank: builders[rank].timer_gate(
                        f"q{following.queue_index:04d}_"
                        f"{sanitize_node_prefix(following.request_id)}_"
                        f"history_piece{piece_number}_interval_gate",
                        interval,
                        after_node_id=builders[rank].previous_id,
                    )
                    for rank in storage_group.ranks
                }
                history_pieces.append(
                    HistoryPieceGate(
                        source_instance_index=piece.instance_index,
                        total_bytes=piece.bytes,
                        timer_gates=timers,
                    )
                )
            pending_history[following.request_id] = tuple(history_pieces)

        request_records.append(
            _request_plan_dict(
                request,
                request_plan,
                plan,
                trace_granularity=config.trace_granularity,
                history_routes=history_routes,
                prefill_decode_routes=prefill_decode_routes,
                kv_offload_routes=kv_offload_routes,
            )
        )

    if pending_history:
        raise RuntimeError(f"unconsumed request gates remain: {sorted(pending_history)}")

    idle_ranks = _add_idle_rank_sentinels(builders)
    _validate_all_rank_dags(builders)

    for rank in range(config.npus_count):
        et_path = output_dir / f"{config.output_prefix}.{rank}.et"
        with et_path.open("wb") as output:
            encode_message(output, _build_metadata(config, plan, rank=rank))
            for node in builders[rank].nodes:
                encode_message(output, node)

    nodes_per_rank = {
        str(rank): len(builders[rank].nodes) for rank in range(config.npus_count)
    }
    request_records.sort(key=lambda record: int(record["queue_index"]))
    if metrics is not None:
        metrics.write_manifest(
            output_dir=output_dir,
            config=config,
            plan=plan,
            node_count_by_rank={
                rank: len(builders[rank].nodes) for rank in range(config.npus_count)
            },
            et_paths_by_rank={
                rank: output_dir / f"{config.output_prefix}.{rank}.et"
                for rank in range(config.npus_count)
            },
            request_records=request_records,
            kv_digest_payload=kv_event_payload_legacy(plan),
        )
    manifest = {
        "output_prefix": str(output_dir / config.output_prefix),
        "trace_label": build_trace_label(config, plan),
        "source_config_digest": config.configuration_digest,
        "execution_mode": "wsc_llm_pd_disaggregated_static_et",
        "mapping_strategy": (
            "WSC-LLM dedicated Prefill/Decode instances; least-request-count "
            "Prefill queue; offline static nearest-Decode route; WSC Relevant(P,D) KV"
        ),
        "static_trace_adaptation": (
            "The planner models Decode FCFS continuous batching for timing, but "
            "ASTRA-sim consumes a static Chakra ET DAG. In request_aggregated mode "
            "each request emits its Decode phase as a separate aggregated DAG, not "
            "one fused cross-request batch node; changing that ET boundary is outside "
            "this request-mapping change."
        ),
        "trace_granularity": config.trace_granularity,
        "trace_representation": {
            "mode": config.trace_granularity,
            "request_aggregated_preserves": [
                "request_identity",
                "request_timing",
                "WSC_LLM_static_prefill_decode_mapping",
                "KV_transfer_routes_and_bytes",
                "aggregate_FLOPs",
                "aggregate_tensor_bytes",
                "aggregate_All-Reduce_payload_bytes",
            ],
            "request_aggregated_compresses": [
                "identical_layer_invocation_count",
                "prefill_chunk_collective_invocation_and_startup_count",
                "decode_token_collective_invocation_and_startup_count",
                "cross_request_decode_batch_fusion",
            ],
            "decode_batching_boundary": (
                "planner_timing_uses_continuous_batching_but_each_request_decode_"
                "phase_is_an_independent_aggregated_ET_DAG"
            ),
            "token_expanded_available": True,
        },
        "hardware": {
            "case": config.hardware_metadata.get("paper-case"),
            "configuration": str(config.hardware_config),
            "capacity_profile": config.hardware_capacity_profile,
            "label": config.hardware.label,
            "mesh_rows": config.hardware.mesh_rows,
            "mesh_cols": config.hardware.mesh_cols,
            "npus_count": config.hardware.npus_count,
            "local_hbm_capacity_bytes_per_npu": config.hardware.local_hbm_capacity_bytes,
            "local_hbm_bandwidth_gbps_per_npu": config.hardware.local_hbm_bandwidth_gbps,
            "local_hbm_latency_ns": config.hardware.local_hbm_latency_ns,
            "adjacent_d2d_bandwidth_gbps": config.hardware.d2d_bandwidth_gbps,
            "d2d_to_hbm_ratio_metadata_only": (
                config.hardware.d2d_to_hbm_bandwidth_ratio
            ),
            "peak_perf_tflops_per_npu": config.hardware.peak_perf_tflops,
            "d2d_latency_ns": config.hardware.d2d_latency_ns,
            "paper_and_simulation_notes": [
                *config.hardware_metadata.get("notes", []),
                str(config.hardware_metadata["selected-capacity-note"]),
            ],
        },
        "model": {
            "name": config.model_name,
            "layers": config.layers,
            "hidden_size": config.hidden_size,
            "ffn_size": config.ffn_size,
            "num_heads": config.num_heads,
            "vocab_size": config.vocab_size,
            "bytes_per_elem": config.bytes_per_elem,
            "mlp_variant": config.mlp_variant,
            "tp_partition": (
                "whole_attention_heads; exact_uneven_mlp_and_vocabulary_slices"
            ),
            "weights": "preloaded_and_distributed_per_instance",
        },
        "instances": [
            {
                "index": instance.index,
                "name": instance.name,
                "pg_name": instance.pg_name,
                "ranks": list(instance.ranks),
                "shape_rows_columns": list(instance.shape),
                "center_row_column": [instance.center_row, instance.center_col],
                "wafer_center_manhattan_distance": (
                    instance.wafer_center_manhattan_distance
                ),
                "phase_role": instance.phase_role,
            }
            for instance in plan.topology.instances
        ],
        "resource_partition": {
            "prefill_instance_count": len(
                plan.topology.indices_for_role(PREFILL_ROLE)
            ),
            "decode_instance_count": len(
                plan.topology.indices_for_role(DECODE_ROLE)
            ),
            "ratio": "6P:3D=2:1",
            "instance_shape": "3x2_NPUs",
            "tensor_parallel_per_instance": len(
                config.inference_groups[0].ranks
            ),
            "assumption": (
                "WSC-LLM does not publish the final LLaMA2-7B/workload allocation. "
                "The deterministic 6P/3D split is not claimed optimal; it preserves "
                "the paper examples' 2:1 resource ratio (Fig.7 NP=8,ND=4 and Fig.8 "
                "24 Prefill dies versus 12 Decode dies) while keeping the user-fixed "
                "six-NPU instance as the minimum placement unit."
            ),
            "center_priority_validation": (
                "max_decode_manhattan_distance_to_wafer_center <= "
                "min_prefill_manhattan_distance_to_wafer_center"
            ),
        },
        "static_pd_mapping": {
            "policy": (
                "nearest Decode instance; globally deterministic equal-shortest-path "
                "combination minimizing total hops, shared-edge occurrences, then "
                "config-order/path signature"
            ),
            "runtime_decode_reselection": False,
            "alpha": plan.static_mapping.alpha,
            "alpha_note": (
                "The paper does not publish a general alpha formula. The checked-in "
                "routes are edge-disjoint, so alpha does not change their cost."
            ),
            "total_hops": plan.static_mapping.total_hops,
            "shared_edge_occurrences": (
                plan.static_mapping.shared_edge_occurrences
            ),
            "adjusted_transfer_cost": (
                plan.static_mapping.adjusted_transfer_cost
            ),
            "edge_use_counts": [
                list(edge) for edge in plan.static_mapping.edge_use_counts
            ],
            "routes": [
                {
                    "prefill_instance_index": route.prefill_instance_index,
                    "decode_instance_index": route.decode_instance_index,
                    "instance_path": list(route.path),
                    "hop_count": route.hop_count,
                    "shared_edges": [list(edge) for edge in route.shared_edges],
                }
                for route in plan.static_mapping.routes
            ],
        },
        "prefill_chunk_size": plan.p_chunk,
        "prefill_queue_policy": {
            "key": [
                "queued_request_count_including_active_head",
                "trace_config_order_tie_break",
            ],
            "occupancy_interpretation": (
                "request count; WSC-LLM says least occupied but does not publish a "
                "formal occupancy equation or final tie-break"
            ),
            "eligible_roles": [PREFILL_ROLE],
        },
        "decode_policy": {
            "assignment": "static_pd_mapping[prefill_instance]",
            "runtime_load_balancing": False,
            "queue": "FCFS_insertion_order_with_continuous_batching_timing",
            "tokens_per_iteration": "one_per_active_request",
        },
        "timing_lut": {
            "path": str(timing_lut_path),
            "source": "analytical_roofline_from_shared_hardware_and_model",
            "role": "phase_timing_only_not_allocation",
            "phase_exclusive_entries": True,
            "entry_count": len(plan.timing_lut.entries),
            "match": (
                "exact phase_role,instance_size,p_chunk,d_batch; nearest d_token "
                "for Decode"
            ),
            "operator_tile_sizes": "not available in the paper and not fabricated",
        },
        "request_queue_csv": str(config.request_queue_csv),
        "request_queue_columns": list(REQUEST_QUEUE_COLUMNS),
        "source_request_count": config.source_request_count,
        "source_session_count": config.source_session_count,
        "selected_request_count": len(config.request_queue),
        "selected_session_count": len(config.selected_session_ids),
        "selected_session_ids": list(config.selected_session_ids),
        "request_queue_session_limit": config.request_queue_session_limit,
        "session_limit_semantics": (
            "0 selects all source sessions; positive N selects the first N "
            "distinct session_id values in source CSV row order and includes "
            "every row belonging to those sessions without rebasing timestamps"
        ),
        "requests": request_records,
        "planning_iterations": [
            {
                "instance_index": iteration.instance_index,
                "phase_role": iteration.phase_role,
                "iteration_index": iteration.iteration_index,
                "start_ns": iteration.start_ns,
                "end_ns": iteration.end_ns,
                "prefill_request_id": iteration.prefill_request_id,
                "prefill_chunk_tokens": iteration.prefill_chunk_tokens,
                "decode_request_ids": list(iteration.decode_request_ids),
                "timing_entry": _timing_entry_dict(iteration.timing_entry),
            }
            for iteration in plan.iterations
        ],
        "kv_management": {
            "policy": (
                "WSC Relevant(P,D) instance-level static Decode domain: Decode D, "
                "the selected request's P-to-D route, and all sibling Prefill "
                "instances/routes statically paired to the same D"
            ),
            "location_priority": [
                "decode_first",
                "selected_request_path_from_decode_to_source_prefill",
                "same_decode_sibling_paths_by_distance",
                "remaining_capacity_secondary_within_sibling_distance",
                "trace_config_order_final",
            ],
            "paper_ambiguity": (
                "Relevant(P,D) construction and exact location priority are not fully "
                "published. Fig.7(b)'s only concrete example includes D1, P1, and P2 "
                "DRAM, which directly motivates the fixed same-Decode domain; the "
                "ordering remains an explicit instance-level simulation interpretation"
            ),
            "capacity_failure": (
                "reserve full-request KV before Prefill; when current capacity is "
                "insufficient, Algorithm 2 line 6 stops the FCFS head and all "
                "subsequent work on that Prefill instance until a later scheduler "
                "event finds enough capacity in the same fixed Decode domain; raise "
                "only when the request cannot fit even in the empty domain; never "
                "widen across Decode domains or remap Decode"
            ),
            "model_weight_reservation_per_instance": True,
            "session_history": (
                "non_terminal_KV_retained_until_next_turn_arrival; terminal_KV_"
                "released_at_decode_completion_for_the_known_finite_trace"
            ),
            "terminal_kv_release_at_completion": True,
            "release_note": (
                "WSC-LLM does not publish finite-trace terminal release details; "
                "the planner follows the known session lifecycle instead of retaining "
                "the final turn forever"
            ),
            "final_remaining_capacity_bytes": list(
                plan.final_remaining_capacity_bytes
            ),
        },
        "remote_memory": {
            "configuration": str(config.remote_memory_config),
            "source_configuration": str(config.hardware_config),
            "memory_type": "NO_MEMORY_EXPANSION",
            "off_chip_gateway_path": False,
        },
        "system_config": str(config.system_config),
        "system_template": str(config.system_template),
        "network_config": str(config.network_config),
        "comm_group_config": str(config.comm_group_config),
        "nodes_per_rank": nodes_per_rank,
        "idle_rank_sentinel": {
            "duration_ns": IDLE_SENTINEL_DURATION_NS,
            "node_type": "root_CPU_timer_COMP",
            "semantics": (
                "ASTRA compatibility sentinel for an otherwise-empty rank; the "
                "CPU timer does not consume modeled NPU hardware resources"
            ),
            "idle_ranks": list(idle_ranks),
            "idle_rank_count": len(idle_ranks),
        },
        "nodes_per_instance": {
            instance.name: sum(nodes_per_rank[str(rank)] for rank in instance.ranks)
            for instance in plan.topology.instances
        },
        "total_nodes": sum(nodes_per_rank.values()),
        "operator_granularity": OPERATOR_GRANULARITY,
        "all_reduce_layout": "2d_mesh_dimension_ordered",
        "remote_operand_loads": config.remote_operand_loads,
    }
    manifest_text = json.dumps(manifest, indent=2)
    (output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    print(manifest_text)


def print_shell_config(config: WscLlmTraceConfig, plan: WscLlmPlan) -> None:
    output_dir = resolve_output_dir(config, plan)
    session_count = len({request.session_id for request in config.request_queue})
    assignments = {
        "TRACE_PREFIX": config.output_prefix,
        "TRACE_DIR": str(output_dir),
        "TRACE_LABEL": build_trace_label(config, plan),
        "MODEL_NAME": config.model_name,
        "MLP_VARIANT": config.mlp_variant,
        "NPUS_COUNT": config.npus_count,
        "MESH_SHAPE": (
            f"{config.hardware.mesh_rows}x{config.hardware.mesh_cols}"
        ),
        "HARDWARE_LABEL": config.hardware.label,
        "CONFIG_DIGEST": config.configuration_digest,
        "REQUEST_COUNT": len(config.request_queue),
        "SESSION_COUNT": session_count,
        "INSTANCE_COUNT": len(config.inference_groups),
        "PREFILL_INSTANCE_COUNT": len(
            plan.topology.indices_for_role(PREFILL_ROLE)
        ),
        "DECODE_INSTANCE_COUNT": len(
            plan.topology.indices_for_role(DECODE_ROLE)
        ),
        "TP_DEGREE": len(config.inference_groups[0].ranks),
        "STATIC_PD_ROUTES": ",".join(
            f"P{route.prefill_instance_index}->D{route.decode_instance_index}"
            for route in plan.static_mapping.routes
        ),
        "PREFILL_CHUNK_SIZE": plan.p_chunk,
        "TRACE_GRANULARITY": config.trace_granularity,
        "PREFILL_RANGE": range_label(
            tuple(request.prefill_length for request in config.request_queue)
        ),
        "DECODE_RANGE": range_label(
            tuple(request.decode_length for request in config.request_queue)
        ),
        "HARDWARE_CONFIG": str(config.hardware_config),
        "HARDWARE_CAPACITY_PROFILE": config.hardware_capacity_profile,
        "SYSTEM_CONFIG": str(config.system_config),
        "NETWORK_CONFIG": str(config.network_config),
        "COMM_GROUP_CONFIG": str(config.comm_group_config),
        "REMOTE_MEMORY": str(config.remote_memory_config),
    }
    for key, value in assignments.items():
        print(f"{key}={shlex.quote(str(value))}")


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    print_shell = False
    plan_only = False
    metrics_detail_arg: Optional[str] = None
    while args and args[0].startswith("--"):
        option = args.pop(0)
        if option == "--print-shell-config":
            print_shell = True
        elif option == "--plan-only":
            plan_only = True
        elif option.startswith("--metrics-detail="):
            metrics_detail_arg = option.split("=", 1)[1]
        else:
            raise SystemExit(
                f"Unknown option {option!r}. "
                "Usage: generate_wsc_llm_trace.py [--print-shell-config|--plan-only] "
                "[--metrics-detail=off|summary|full] [trace_config.csv]. "
                f"Default config: {CONFIG_CSV_PATH}"
            )
    if print_shell and plan_only:
        raise SystemExit("--print-shell-config and --plan-only cannot be combined")
    if len(args) > 1:
        raise SystemExit(
            "Usage: generate_wsc_llm_trace.py [--print-shell-config|--plan-only] "
            "[--metrics-detail=off|summary|full] [trace_config.csv]. "
            f"Default config: {CONFIG_CSV_PATH}"
        )
    config_csv = CONFIG_CSV_PATH if not args else Path(args[0])
    config = load_wsc_llm_trace_config(config_csv)
    # Metrics sidecar switch (doc sec.10): CLI > METRICS_DETAIL >
    # ENABLE_METRICS > default on.  When off, no metrics_manifest.json is
    # written and the .et output is byte-identical to the metrics-on output.
    metrics: Optional[ServiceMetrics] = None
    if not print_shell and not plan_only:
        metrics_detail = resolve_metrics_detail(metrics_detail_arg, os.environ)
        if metrics_detail != "off":
            metrics = ServiceMetrics(metrics_detail)
    # Streaming planner-LUT iteration aggregates (doc sec.8.8): observed
    # during planning, emitted after a successful trace write; never stored
    # per-iteration and never fed back into the planner.
    lut_stats: Optional[PlannerLutStatsAccumulator] = None
    if metrics is not None:
        session_kv_manager.set_metrics_observer(metrics.memory)
        lut_stats = PlannerLutStatsAccumulator()
        wsc_llm_scheduler.set_iteration_stats_hook(lut_stats.record_lut_iteration)
    try:
        plan = build_wsc_llm_plan(config)
    finally:
        session_kv_manager.set_metrics_observer(None)
        wsc_llm_scheduler.set_iteration_stats_hook(None)
    if print_shell:
        print_shell_config(config, plan)
    elif plan_only:
        print(json.dumps({
            "requests": len(plan.requests),
            "kv_events": len(plan.kv_events),
            "planning_iteration_count": plan.planning_iteration_count,
            "stored_planning_iterations": len(plan.iterations),
        }))
    else:
        write_wsc_llm_trace(config, plan, metrics=metrics)
        if metrics is not None and lut_stats is not None:
            write_planner_lut_stats(lut_stats, output_dir=resolve_output_dir(config, plan))


if __name__ == "__main__":
    main()
