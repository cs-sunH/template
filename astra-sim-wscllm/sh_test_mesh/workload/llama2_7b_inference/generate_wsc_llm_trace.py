#!/usr/bin/env python3
"""Shared WSC-LLM configuration and GraphBatch-emission primitives for online runs."""

from __future__ import annotations

import csv
import hashlib
import sys
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
    WscLlmModel,
    kv_cache_bytes_for_tokens,
)
from session_kv_manager import NOC_MIGRATE, RECOMPUTE  # noqa: E402
from generate_trace import (  # noqa: E402
    ChakraNode,
    RequestSpec,
    TraceBuilder,
    clean_csv_row,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    parse_rank_spec,
    sanitize_node_prefix,
    shard_extent,
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
    "kv_cache_policy": "session_lru_recompute",
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
        # A.5(2026-09-05):legacy 与 relevant 两个历史变体已清除,唯一
        # 合法取值 session_lru_recompute;其余取值 fail-closed。
        if value != "session_lru_recompute":
            raise ValueError(
                "config key kv_cache_policy must be "
                f"session_lru_recompute, got {value!r}"
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
    if not request_queue_csv.is_file():
        print(
            f"missing request queue: {request_queue_csv};"
            "请按 traces/derive_20_first_30_seconds.py 物化输入"
            "(运行 stdout 即权威 provenance 记录)",
            file=sys.stderr,
        )
        sys.exit(1)
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
    digest_paths = [
            config_csv.resolve(),
            request_queue_csv,
            hardware_path,
            system_template,
    ]
    configuration_digest = _configuration_digest(tuple(digest_paths))

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
        raise ValueError("WSC-LLM direct KV shard pairing requires equal TP degree")
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
    and last *real* operator/collective nodes (the artificial
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









def _eviction_record_dict(eviction) -> dict[str, object]:
    """Serialize one session-KV EvictionRecord for the replay decision log."""

    return {
        "time_ns": eviction.time_ns,
        "phase": eviction.phase,
        "reason": eviction.reason,
        "trigger_request_id": eviction.trigger_request_id,
        "victim_session_id": eviction.victim_session_id,
        "victim_instance_index": eviction.victim_instance_index,
        "victim_last_completion_ns": eviction.victim_last_completion_ns,
        "context_tokens": eviction.context_tokens,
        "shard_bytes": list(eviction.shard_bytes),
    }


def _kv_transfer_dict(
    transfer, hardware: Optional[WscLlmHardware] = None
) -> Optional[dict[str, object]]:
    """Serialize one session-KV KVTransfer (or None) for the decision log.

    问题 4b（2026-09-05）：hardware 非 None 时每个 shard 附加
    noc_path/noc_hops（rank 级 XY 路由，与 _paired_transfer 的 routes
    行同构同源；生成侧只读派生，供 hopbytes shard 级 mesh 跳口径消费，
    A/B 对拍剥离清单条目）。缺省 hardware=None 行为与旧版逐字节一致
    （兼容既有调用与旧产物回退路径）。
    """

    if transfer is None:
        return None
    shard_rows: list[dict[str, object]] = []
    for shard in transfer.shards:
        row: dict[str, object] = {
            "relative_tp_rank": shard.relative_tp_rank,
            "source_rank": shard.source_rank,
            "target_rank": shard.target_rank,
            "bytes": shard.bytes,
        }
        if hardware is not None:
            path = _xy_route(hardware, shard.source_rank, shard.target_rank)
            row["noc_path"] = list(path)
            row["noc_hops"] = max(0, len(path) - 1)
        shard_rows.append(row)
    return {
        "action": transfer.action,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "session_id": transfer.session_id,
        "trigger_request_id": transfer.trigger_request_id,
        "source_instance_index": transfer.source_instance_index,
        "target_instance_index": transfer.target_instance_index,
        "history_tokens": transfer.history_tokens,
        "total_bytes": transfer.total_bytes,
        "shards": shard_rows,
    }


def main(argv=None) -> None:  # noqa: ARG001
    """fail-closed 拒绝桩(2026-08-18 起生效):离线静态全管线入口已删除。

    本模块保留的仅是③④在线路径只读 import 的符号(config 装载/发射辅助/
    估算函数)。③④ 的输入物化入口是
    plan_materializer.py(manifest/metrics/runtime_config);
    静态 ET 生成入口不再存在。
    """
    raise SystemExit(
        "path-1 (offline static full pipeline) was removed on 2026-08-18; "
        "use plan_materializer.py for the online routes' plan-dir inputs")


if __name__ == "__main__":
    main()
