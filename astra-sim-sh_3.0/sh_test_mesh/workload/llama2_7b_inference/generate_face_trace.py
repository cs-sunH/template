#!/usr/bin/env python3
"""Generate FACE-mapped Chakra ET traces for the configured wafer scenario."""

from __future__ import annotations

import csv
import dataclasses
import heapq
import hashlib
import io
import json
import multiprocessing as mp
import os
import pickle
import queue as queue_module
import shlex
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass, replace
from pathlib import Path
from typing import BinaryIO, Callable, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SH_TEST_DIR = MODULE_DIR.parents[1]
if str(SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SH_TEST_DIR))

from face_scheduler import (  # noqa: E402
    DecodeCandidateCost,
    FaceHardware,
    FaceInstanceSpec,
    FaceLutEntry,
    FaceModel,
    FacePlan,
    FaceRequest,
    FaceRequestPlan,
    InstanceTaskLoadSnapshot,
    KVAllocation,
    KVTransfer,
    KVTransferShard,
    NodeHBMSnapshot,
    PREFILL_CHUNK_SIZE,
    SessionKVSnapshot,
    kv_cache_bytes_for_tokens,
    plan_face_requests,
    set_iteration_stats_hook,
    set_metrics_observer,
)
from metrics_integration import (  # noqa: E402
    EVENT_DECODE_END,
    EVENT_DECODE_START,
    EVENT_MEMORY_ANCHOR_COMPLETE,
    EVENT_PREFILL_END,
    EVENT_PREFILL_START,
    PlannerLutStatsAccumulator,
    ServiceMetrics,
    kv_event_payload_sh2,
    resolve_metrics_detail,
    write_planner_lut_stats,
)
from generate_trace import (  # noqa: E402
    ChakraAttr,
    GlobalMetadata,
    InferenceGroup,
    PROJECT_ROOT,
    REQUEST_QUEUE_COLUMNS,
    RemoteMemoryConfig,
    RequestSpec,
    TraceCommand,
    TraceCommandRecorder,
    TraceBuilder,
    encode_message,
    load_config_rows,
    load_remote_memory_config,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    range_label,
    request_queue_digest,
    sanitize_node_prefix,
    transformer_pass,
    transformer_pass_aggregated,
)
from config_resolver import (  # noqa: E402
    ResolvedHardware,
    load_hardware_config,
    materialize_runtime_configs,
)


CONFIG_CSV_PATH = MODULE_DIR / "trace_config.csv"
PLANNER_CACHE_VERSION = "face-prefix-hbm-v5-admission-wait"
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
    "kv_reserve_context_tokens",
)
OPTIONAL_CONFIG_DEFAULTS = {
    "request_queue_session_limit": "0",
    "trace_granularity": "token_expanded",
    "request_queue_context_csv": "",
}
SUPPORTED_CONFIG_KEYS = set(REQUIRED_CONFIG_KEYS) | set(OPTIONAL_CONFIG_DEFAULTS)
INT_CONFIG_KEYS = {
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
    "kv_reserve_context_tokens",
}
PATH_CONFIG_KEYS = {
    "output_dir",
    "request_queue_csv",
    "request_queue_context_csv",
    "hardware_config",
    "system_template",
}
OPERATOR_GRANULARITY = (
    "rmsnorm,qkv,qk,scale_mask,softmax,av,out_proj,residual,"
    "mlp_gate_up,swiglu,mlp_down,logits"
)


@dataclass(frozen=True)
class FaceTraceConfig:
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
    request_queue_context_csv: Optional[Path]
    request_queue: tuple[RequestSpec, ...]
    hardware_config: Path
    hardware_capacity_profile: str
    hardware: FaceHardware
    hardware_metadata: dict[str, object]
    system_template: Path
    system_config: Path
    network_config: Path
    comm_group_config: Path
    remote_memory: RemoteMemoryConfig
    remote_operand_loads: bool
    kv_reserve_context_tokens: int
    inference_groups: tuple[InferenceGroup, ...]
    request_queue_session_limit: int
    selected_session_ids: tuple[str, ...]
    source_request_count: int
    source_session_count: int
    source_average_decode_length: float
    trace_granularity: str
    configuration_digest: str

    @property
    def model(self) -> FaceModel:
        return FaceModel(
            layers=self.layers,
            hidden_size=self.hidden_size,
            ffn_size=self.ffn_size,
            num_heads=self.num_heads,
            vocab_size=self.vocab_size,
            bytes_per_elem=self.bytes_per_elem,
            mlp_variant=self.mlp_variant,
        )


@dataclass
class PendingHistoryGate:
    source_instance_index: int
    timer_gates: tuple[Optional[int], ...]
    location: str


@dataclass(frozen=True)
class TransferTriggerGate:
    control_instance_index: int
    node_gates: tuple[Optional[int], ...]


class TransferTagAllocator:
    def __init__(self) -> None:
        self._next_tag = 1

    def take(self) -> int:
        if self._next_tag > 0xFFFFFFFF:
            raise OverflowError("KV transfer communication tag space was exhausted")
        tag = self._next_tag
        self._next_tag += 1
        return tag






def reconcile_pending_history_location(
    pending_gate: PendingHistoryGate,
    request_plan: FaceRequestPlan,
) -> None:
    """Apply planner-side zero-context normalization to the ET history gate."""

    history_before = request_plan.history_location_before
    if history_before is None or pending_gate.location == history_before.location:
        return
    if (
        request_plan.history_tokens_discarded > 0
        and request_plan.history_tokens_before == 0
        and pending_gate.location == "partial_hbm_remote"
        and history_before.location == "local_hbm"
        and history_before.total_bytes == 0
    ):
        pending_gate.location = history_before.location
        return
    raise RuntimeError(
        f"history gate location {pending_gate.location!r} does not match "
        f"planned location {history_before.location!r}; "
        f"request={request_plan.request_id}, "
        f"discarded_tokens={request_plan.history_tokens_discarded}"
    )


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
        if key in {"output_dir", "request_queue_context_csv"} and not value:
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


def _to_face_hardware(hardware: ResolvedHardware) -> FaceHardware:
    return FaceHardware(
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


def load_request_prefix_tokens(
    context_csv: Path,
    requests: Sequence[RequestSpec],
) -> tuple[RequestSpec, ...]:
    """Attach exact per-request historical-prefix lengths from a sidecar CSV."""

    if not context_csv.is_file():
        raise FileNotFoundError(
            f"request queue context CSV not found: {context_csv}"
        )
    required_columns = {
        "session_id",
        "turn_index",
        "request_id",
        "prefix_tokens",
        "input_tokens_total",
    }
    metadata: dict[tuple[str, int, str], tuple[int, int]] = {}
    with context_csv.open(newline="", encoding="utf-8-sig") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None:
            raise ValueError(f"request context CSV is empty: {context_csv}")
        columns = {str(column).strip() for column in reader.fieldnames}
        missing = sorted(required_columns - columns)
        if missing:
            raise ValueError(
                "request context CSV is missing columns: " + ", ".join(missing)
            )
        for line_number, row in enumerate(reader, start=2):
            session_id = str(row.get("session_id", "")).strip()
            request_id = str(row.get("request_id", "")).strip()
            turn_text = str(row.get("turn_index", "")).strip()
            prefix_text = str(row.get("prefix_tokens", "")).strip()
            input_text = str(row.get("input_tokens_total", "")).strip()
            if not all((session_id, request_id, turn_text, prefix_text, input_text)):
                raise ValueError(
                    f"request context CSV line {line_number} has empty identity/context fields"
                )
            turn_index = parse_nonnegative_int(turn_text, "turn_index")
            prefix_tokens = parse_nonnegative_int(prefix_text, "prefix_tokens")
            input_tokens_total = parse_int(input_text, "input_tokens_total")
            key = (session_id, turn_index, request_id)
            if key in metadata:
                raise ValueError(
                    f"duplicate request context metadata for {request_id}"
                )
            metadata[key] = (prefix_tokens, input_tokens_total)

    enriched: list[RequestSpec] = []
    used_keys: set[tuple[str, int, str]] = set()
    for request in requests:
        key = (request.session_id, request.turn_index, request.request_id)
        if key not in metadata:
            raise ValueError(
                f"request context CSV has no metadata for {request.request_id}"
            )
        prefix_tokens, input_tokens_total = metadata[key]
        if input_tokens_total != prefix_tokens + request.prefill_length:
            raise ValueError(
                f"request {request.request_id} input_tokens_total does not equal "
                "prefix_tokens + prefill_length"
            )
        enriched.append(
            replace(
                request,
                prefix_tokens=prefix_tokens,
                input_tokens_total=input_tokens_total,
            )
        )
        used_keys.add(key)
    extra_keys = set(metadata) - used_keys
    if extra_keys:
        raise ValueError(
            "request context CSV contains requests absent from the ASTRA queue"
        )
    return tuple(enriched)


def load_face_trace_config(config_csv: Path = CONFIG_CSV_PATH) -> FaceTraceConfig:
    values, groups = load_config_rows(config_csv, SUPPORTED_CONFIG_KEYS)

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
    # ED 改造阶段 0（步骤 0-1）：正式入口 fail-closed。必须在调用
    # load_request_queue 之前拦截——该函数在文件缺失时会静默调用
    # create_default_request_queue 生成随机 4-request 队列（request-neutral
    # 红线）。随机 stub 保留仅供显式 fixture 使用。
    if not request_queue_csv.exists():
        sys.exit(
            f"missing request queue: {request_queue_csv}；"
            "request-neutral 仓库不绑定默认队列——请按方案文档 "
            "(sh_3.0仓库改造详细执行方案.md §3 步骤 0-1) 物化 "
            "sidecar_restore 三件套后在 trace_config.csv 指定"
        )
    source_request_queue = load_request_queue(request_queue_csv)
    request_queue_context_csv = parsed["request_queue_context_csv"]
    if request_queue_context_csv is not None:
        request_queue_context_csv = _resolve_request_queue(
            request_queue_context_csv
        )
        source_request_queue = load_request_prefix_tokens(
            request_queue_context_csv,
            source_request_queue,
        )
    source_average_decode_length = sum(
        request.decode_length for request in source_request_queue
    ) / len(source_request_queue)
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
    hardware = _to_face_hardware(resolved_hardware)
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
    remote_memory = load_remote_memory_config(
        runtime_configs.remote_memory,
        npus_count,
        mesh_shape=(hardware.mesh_rows, hardware.mesh_cols),
    )
    digest_paths = [
            config_csv.resolve(),
            request_queue_csv,
            hardware_path,
            system_template,
    ]
    if request_queue_context_csv is not None:
        digest_paths.append(request_queue_context_csv)
    configuration_digest = _configuration_digest(tuple(digest_paths))

    return FaceTraceConfig(
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
        request_queue_context_csv=request_queue_context_csv,
        request_queue=request_queue,
        hardware_config=hardware_path,
        hardware_capacity_profile=hardware_capacity_profile,
        hardware=hardware,
        hardware_metadata=hardware_metadata,
        system_template=system_template,
        system_config=runtime_configs.system,
        network_config=runtime_configs.network,
        comm_group_config=runtime_configs.comm_group,
        remote_memory=remote_memory,
        remote_operand_loads=bool(parsed["remote_operand_loads"]),
        kv_reserve_context_tokens=int(parsed["kv_reserve_context_tokens"]),
        inference_groups=groups,
        request_queue_session_limit=int(parsed["request_queue_session_limit"]),
        selected_session_ids=selected_session_ids,
        source_request_count=len(source_request_queue),
        source_session_count=source_session_count,
        source_average_decode_length=source_average_decode_length,
        trace_granularity=str(parsed["trace_granularity"]),
        configuration_digest=configuration_digest,
    )






def _transfer_trigger_time_ns(
    request_plan: FaceRequestPlan,
    transfer: KVTransfer,
) -> int:
    if transfer.phase == "history":
        return request_plan.admission_time_ns
    if transfer.phase in {"prefill", "prefill_decode", "decode"}:
        return request_plan.prefill_complete_ns
    if transfer.phase == "completion":
        return request_plan.completion_ns
    raise ValueError(f"unsupported KV transfer phase: {transfer.phase}")


def _remote_store_transfers(
    request_plan: FaceRequestPlan,
) -> tuple[KVTransfer, ...]:
    return tuple(
        transfer
        for transfer in (
            *request_plan.history_evictions,
            *request_plan.prefill_evictions,
            *request_plan.decode_evictions,
            *request_plan.completion_evictions,
        )
        if transfer.kind == "remote_store"
    )


def order_plans_for_static_emission(
    plan: FacePlan,
) -> tuple[FaceRequestPlan, ...]:
    """Topologically order whole-request ET emission by KV causality.

    Start-time order alone is insufficient when a long-running request later
    evicts KV produced by a shorter request that started after it.  The ET
    producer must be emitted before that store trigger, and every store must be
    emitted before the affected session's following request consumes history.
    Kahn ordering preserves those edges while retaining Prefill-start order as
    the deterministic priority among otherwise independent requests.
    """

    plans = tuple(plan.requests)
    by_id = {request.request_id: request for request in plans}
    if len(by_id) != len(plans):
        raise ValueError("FACE request IDs must be unique for ET emission")
    successors: dict[str, set[str]] = {request.request_id: set() for request in plans}
    indegree = dict.fromkeys(successors, 0)

    def add_edge(source: FaceRequestPlan, target: FaceRequestPlan) -> None:
        if source.request_id == target.request_id:
            return
        if target.request_id not in successors[source.request_id]:
            successors[source.request_id].add(target.request_id)
            indegree[target.request_id] += 1

    session_plans: dict[str, list[FaceRequestPlan]] = {}
    for request in plans:
        session_plans.setdefault(request.session_id, []).append(request)
    following_by_request_id: dict[str, FaceRequestPlan] = {}
    for requests in session_plans.values():
        requests.sort(key=lambda request: request.turn_index)
        for current, following in zip(requests, requests[1:]):
            add_edge(current, following)
            following_by_request_id[current.request_id] = following

    store_events_by_session: dict[
        str,
        list[tuple[int, int, FaceRequestPlan]],
    ] = {}
    for trigger in plans:
        for sequence, transfer in enumerate(_remote_store_transfers(trigger)):
            trigger_time_ns = _transfer_trigger_time_ns(trigger, transfer)
            producers = [
                candidate
                for candidate in session_plans[transfer.session_id]
                if candidate.completion_ns <= trigger_time_ns
            ]
            if not producers:
                raise RuntimeError(
                    f"remote store for session {transfer.session_id} has no "
                    "completed KV producer"
                )
            producer = max(
                producers,
                key=lambda request: (
                    request.completion_ns,
                    request.turn_index,
                    request.queue_index,
                ),
            )
            add_edge(producer, trigger)
            following = following_by_request_id.get(producer.request_id)
            if following is not None:
                following_admission_ns = getattr(
                    following,
                    "admission_time_ns",
                    following.estimated_arrival_ns,
                )
                if trigger_time_ns > following_admission_ns:
                    raise RuntimeError(
                        "KV store trigger occurs after the affected session's "
                        f"following request admission: {transfer.session_id}"
                    )
                add_edge(trigger, following)
            store_events_by_session.setdefault(transfer.session_id, []).append(
                (trigger_time_ns, sequence, trigger)
            )

    for events in store_events_by_session.values():
        events.sort(
            key=lambda item: (
                item[0],
                item[2].queue_index,
                item[1],
            )
        )
        for previous, following in zip(events, events[1:]):
            add_edge(previous[2], following[2])

    ready: list[tuple[int, int, str]] = []
    for request in plans:
        if indegree[request.request_id] == 0:
            heapq.heappush(
                ready,
                (request.prefill_start_ns, request.queue_index, request.request_id),
            )
    ordered: list[FaceRequestPlan] = []
    while ready:
        _, _, request_id = heapq.heappop(ready)
        request = by_id[request_id]
        ordered.append(request)
        for successor_id in sorted(successors[request_id]):
            indegree[successor_id] -= 1
            if indegree[successor_id] == 0:
                successor = by_id[successor_id]
                heapq.heappush(
                    ready,
                    (
                        successor.prefill_start_ns,
                        successor.queue_index,
                        successor.request_id,
                    ),
                )
    if len(ordered) != len(plans):
        blocked = sorted(
            request_id for request_id, degree in indegree.items() if degree > 0
        )
        raise RuntimeError(
            "KV-causal static ET request ordering contains a cycle: "
            + ", ".join(blocked[:8])
        )
    return tuple(ordered)


def derive_prefill_work_tokens(
    requests: Sequence[RequestSpec],
) -> tuple[int, ...]:
    """Derive per-request Prefill work without running the FACE scheduler."""

    by_session: dict[str, list[int]] = {}
    for index, request in enumerate(requests):
        by_session.setdefault(request.session_id, []).append(index)

    work_tokens = [0] * len(requests)
    for session_id, indexes in by_session.items():
        ordered = sorted(indexes, key=lambda index: requests[index].turn_index)
        turns = [requests[index].turn_index for index in ordered]
        if turns != list(range(len(ordered))):
            raise ValueError(
                f"session {session_id} turn indexes must be contiguous from zero"
            )
        previous_final_context = 0
        for index in ordered:
            request = requests[index]
            if request.prefix_tokens is None:
                history_tokens = previous_final_context
                prefill_context_tokens = history_tokens + request.prefill_length
                request_work_tokens = request.prefill_length
            else:
                history_tokens = min(
                    request.prefix_tokens,
                    previous_final_context,
                )
                if request.input_tokens_total is None:
                    raise ValueError(
                        f"request {request.request_id} is missing input_tokens_total"
                    )
                prefill_context_tokens = request.input_tokens_total
                request_work_tokens = prefill_context_tokens - history_tokens
                if request_work_tokens <= 0:
                    raise ValueError(
                        f"session {session_id} request {request.request_id} has no "
                        "Prefill work after prefix reuse"
                    )
            work_tokens[index] = request_work_tokens
            previous_final_context = (
                prefill_context_tokens + request.decode_length
            )
    return tuple(work_tokens)







def _validate_transfer_shard(
    config: FaceTraceConfig,
    transfer: KVTransfer,
    shard: KVTransferShard,
) -> None:
    if shard.bytes <= 0:
        raise ValueError("emitted KV transfer shards must contain positive bytes")
    if (
        shard.layer_start != transfer.layer_start
        or shard.layer_end != transfer.layer_end
    ):
        raise ValueError("KV transfer shard layer range does not match its action")
    if transfer.kind == "noc_migrate":
        if shard.source_rank is None or shard.target_rank is None:
            raise ValueError("NoC migration shard requires source and target ranks")
        if shard.edge_rank is not None:
            raise ValueError("NoC migration shard must not define an edge rank")
    elif transfer.kind == "remote_store":
        if (
            shard.source_rank is None
            or shard.target_rank is None
            or shard.edge_rank is None
            or shard.target_rank != shard.edge_rank
        ):
            raise ValueError("remote store shard must route source to its edge rank")
    elif transfer.kind == "remote_load":
        if (
            shard.source_rank is None
            or shard.target_rank is None
            or shard.edge_rank is None
            or shard.source_rank != shard.edge_rank
        ):
            raise ValueError("remote load shard must route its edge rank to target")

    if shard.edge_rank is not None and shard.edge_rank not in config.remote_memory.edge_npus:
        raise ValueError(
            f"rank {shard.edge_rank} is not a configured remote-memory edge"
        )
    if shard.noc_path:
        if shard.source_rank is None or shard.target_rank is None:
            raise ValueError("NoC path requires source and target ranks")
        if (
            shard.noc_path[0] != shard.source_rank
            or shard.noc_path[-1] != shard.target_rank
        ):
            raise ValueError("KV transfer NoC path endpoints do not match its shard")


def _kv_transfer_dict(
    transfer: KVTransfer,
    shard_records: Sequence[dict[str, object]],
) -> dict[str, object]:
    return {
        "kind": transfer.kind,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "session_id": transfer.session_id,
        "trigger_request_id": transfer.trigger_request_id,
        "source_instance_index": transfer.source_instance_index,
        "target_instance_index": transfer.target_instance_index,
        "total_bytes": transfer.total_bytes,
        "model_layers": transfer.model_layers,
        "layer_start": transfer.layer_start,
        "layer_end": transfer.layer_end,
        "resident_prefix_layers_before": (
            transfer.resident_prefix_layers_before
        ),
        "resident_prefix_layers_after": transfer.resident_prefix_layers_after,
        "shards": list(shard_records),
    }


def _history_control(
    *,
    group_by_index: dict[int, InferenceGroup],
    pending_gate: PendingHistoryGate,
    relative_index: int,
) -> tuple[int, Optional[int]]:
    source_group = group_by_index[pending_gate.source_instance_index]
    if relative_index >= len(source_group.ranks):
        raise ValueError("history control rank index exceeds source TP degree")
    return source_group.ranks[relative_index], pending_gate.timer_gates[relative_index]


def _emit_transfer_trigger(
    *,
    builders: dict[int, TraceBuilder],
    group_by_index: dict[int, InferenceGroup],
    tag_allocator: TransferTagAllocator,
    trigger_gate: Optional[TransferTriggerGate],
    relative_index: int,
    source_rank: int,
    action_name: str,
    shard_index: int,
) -> dict[str, object]:
    if trigger_gate is None:
        return {}
    control_group = group_by_index[trigger_gate.control_instance_index]
    if len(control_group.ranks) != len(trigger_gate.node_gates):
        raise ValueError("KV transfer trigger gate does not match its TP degree")
    if relative_index >= len(control_group.ranks):
        raise ValueError("KV transfer trigger rank index exceeds its TP degree")
    control_rank = control_group.ranks[relative_index]
    node_gate = trigger_gate.node_gates[relative_index]
    builders[control_rank].arm_timer_gate(node_gate)
    if control_rank == source_rank:
        return {
            "trigger_control_rank": control_rank,
            "trigger_dependency_node_id": node_gate,
            "trigger_tag": None,
        }

    trigger_tag = tag_allocator.take()
    builders[control_rank].comm_send(
        f"{action_name}_shard{shard_index}_trigger_to_rank{source_rank}",
        src=control_rank,
        dst=source_rank,
        comm_size=1,
        comm_tag=trigger_tag,
    )
    builders[source_rank].comm_recv(
        f"{action_name}_shard{shard_index}_trigger_from_rank{control_rank}",
        src=control_rank,
        dst=source_rank,
        comm_size=1,
        comm_tag=trigger_tag,
    )
    return {
        "trigger_control_rank": control_rank,
        "trigger_dependency_node_id": node_gate,
        "trigger_tag": trigger_tag,
    }


def _emit_kv_transfer(
    *,
    config: FaceTraceConfig,
    builders: dict[int, TraceBuilder],
    group_by_index: dict[int, InferenceGroup],
    tag_allocator: TransferTagAllocator,
    transfer: KVTransfer,
    action_name: str,
    pending_gate: Optional[PendingHistoryGate] = None,
    trigger_gate: Optional[TransferTriggerGate] = None,
    transfer_anchor_sink: Optional[Callable[[int, int], None]] = None,
) -> dict[str, object]:
    if transfer.kind == "local_hit":
        if transfer.shards:
            raise ValueError("local-hit transfer must not contain data shards")
        if pending_gate is not None:
            if transfer.target_instance_index is None:
                raise ValueError("history local hit requires a target instance")
            target_group = group_by_index[transfer.target_instance_index]
            if len(target_group.ranks) != len(pending_gate.timer_gates):
                raise ValueError("history local-hit TP degree does not match its gate")
            for relative_index, target_rank in enumerate(target_group.ranks):
                control_rank, timer_gate = _history_control(
                    group_by_index=group_by_index,
                    pending_gate=pending_gate,
                    relative_index=relative_index,
                )
                if control_rank != target_rank:
                    raise ValueError(
                        "local history hit control rank does not match target rank"
                    )
                builders[target_rank].arm_timer_gate(timer_gate)
        return _kv_transfer_dict(transfer, ())

    if sum(shard.bytes for shard in transfer.shards) != transfer.total_bytes:
        raise ValueError("KV transfer shards do not preserve the action byte total")

    source_group = (
        None
        if transfer.source_instance_index is None
        else group_by_index[transfer.source_instance_index]
    )
    target_group = (
        None
        if transfer.target_instance_index is None
        else group_by_index[transfer.target_instance_index]
    )
    shard_records: list[dict[str, object]] = []
    for shard_index, shard in enumerate(transfer.shards):
        _validate_transfer_shard(config, transfer, shard)
        record: dict[str, object] = {
            "shard_index": shard_index,
            "source_rank": shard.source_rank,
            "target_rank": shard.target_rank,
            "edge_rank": shard.edge_rank,
            "bytes": shard.bytes,
            "layer_start": shard.layer_start,
            "layer_end": shard.layer_end,
            "noc_path": list(shard.noc_path),
            "noc_hops": max(0, len(shard.noc_path) - 1),
        }

        if transfer.kind == "noc_migrate":
            assert shard.source_rank is not None
            assert shard.target_rank is not None
            if source_group is None or target_group is None:
                raise ValueError("NoC migration requires source and target instances")
            relative_index = source_group.ranks.index(shard.source_rank)
            if target_group.ranks[relative_index] != shard.target_rank:
                raise ValueError("NoC migration does not preserve relative TP rank")
            if pending_gate is not None:
                control_rank, timer_gate = _history_control(
                    group_by_index=group_by_index,
                    pending_gate=pending_gate,
                    relative_index=relative_index,
                )
                if control_rank != shard.source_rank:
                    raise ValueError(
                        "history NoC migration control rank is not its source rank"
                    )
                builders[control_rank].arm_timer_gate(timer_gate)
            data_tag = tag_allocator.take()
            builders[shard.source_rank].comm_send(
                f"{action_name}_shard{shard_index}_send",
                src=shard.source_rank,
                dst=shard.target_rank,
                comm_size=shard.bytes,
                comm_tag=data_tag,
            )
            builders[shard.target_rank].comm_recv(
                f"{action_name}_shard{shard_index}_recv",
                src=shard.source_rank,
                dst=shard.target_rank,
                comm_size=shard.bytes,
                comm_tag=data_tag,
            )
            if transfer_anchor_sink is not None:
                # Metrics code-7 anchor (doc sec.4.3): the target-side data
                # arrival node of this NoC migration shard.  Observation
                # only; the emitted nodes are unchanged.
                target_recv_node_id = builders[shard.target_rank].previous_id
                if target_recv_node_id is not None:
                    transfer_anchor_sink(shard.target_rank, target_recv_node_id)
            ack_tag = tag_allocator.take()
            builders[shard.target_rank].comm_send(
                f"{action_name}_shard{shard_index}_ack_to_rank{shard.source_rank}",
                src=shard.target_rank,
                dst=shard.source_rank,
                comm_size=1,
                comm_tag=ack_tag,
            )
            builders[shard.source_rank].comm_recv(
                f"{action_name}_shard{shard_index}_ack_from_rank{shard.target_rank}",
                src=shard.target_rank,
                dst=shard.source_rank,
                comm_size=1,
                comm_tag=ack_tag,
            )
            record.update(
                {
                    "data_tag": data_tag,
                    "ack_tag": ack_tag,
                    "source_release_dependency": "noc_migration_ack_recv",
                }
            )

        elif transfer.kind == "remote_store":
            assert shard.source_rank is not None
            assert shard.edge_rank is not None
            source_rank = shard.source_rank
            edge_rank = shard.edge_rank
            if source_group is None:
                raise ValueError("remote store requires a source instance")
            relative_index = source_group.ranks.index(source_rank)
            record.update(
                _emit_transfer_trigger(
                    builders=builders,
                    group_by_index=group_by_index,
                    tag_allocator=tag_allocator,
                    trigger_gate=trigger_gate,
                    relative_index=relative_index,
                    source_rank=source_rank,
                    action_name=action_name,
                    shard_index=shard_index,
                )
            )
            if source_rank == edge_rank:
                builders[edge_rank].mem_store(
                    f"{action_name}_shard{shard_index}_edge_store",
                    shard.bytes,
                )
                record.update(
                    {
                        "direct_edge_access": True,
                        "source_release_dependency": "mem_store_completion",
                    }
                )
            else:
                data_tag = tag_allocator.take()
                ack_tag = tag_allocator.take()
                builders[source_rank].comm_send(
                    f"{action_name}_shard{shard_index}_send_to_edge{edge_rank}",
                    src=source_rank,
                    dst=edge_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                )
                builders[edge_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_recv_from_rank{source_rank}",
                    src=source_rank,
                    dst=edge_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                )
                builders[edge_rank].mem_store(
                    f"{action_name}_shard{shard_index}_remote_store",
                    shard.bytes,
                )
                builders[edge_rank].comm_send(
                    f"{action_name}_shard{shard_index}_ack_to_rank{source_rank}",
                    src=edge_rank,
                    dst=source_rank,
                    comm_size=1,
                    comm_tag=ack_tag,
                )
                builders[source_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_ack_from_edge{edge_rank}",
                    src=edge_rank,
                    dst=source_rank,
                    comm_size=1,
                    comm_tag=ack_tag,
                )
                record.update(
                    {
                        "direct_edge_access": False,
                        "data_tag": data_tag,
                        "ack_tag": ack_tag,
                        "source_release_dependency": "remote_store_ack_recv",
                    }
                )

        elif transfer.kind == "remote_load":
            assert shard.target_rank is not None
            assert shard.edge_rank is not None
            if target_group is None or pending_gate is None:
                raise ValueError("history remote load requires target and timer gate")
            target_rank = shard.target_rank
            edge_rank = shard.edge_rank
            relative_index = target_group.ranks.index(target_rank)
            control_rank, timer_gate = _history_control(
                group_by_index=group_by_index,
                pending_gate=pending_gate,
                relative_index=relative_index,
            )
            request_tag: Optional[int] = None
            if control_rank == edge_rank:
                builders[edge_rank].arm_timer_gate(timer_gate)
            else:
                request_tag = tag_allocator.take()
                builders[control_rank].arm_timer_gate(timer_gate)
                builders[control_rank].comm_send(
                    f"{action_name}_shard{shard_index}_request_to_edge{edge_rank}",
                    src=control_rank,
                    dst=edge_rank,
                    comm_size=1,
                    comm_tag=request_tag,
                )
                builders[edge_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_request_from_rank{control_rank}",
                    src=control_rank,
                    dst=edge_rank,
                    comm_size=1,
                    comm_tag=request_tag,
                )
            builders[edge_rank].mem_load(
                f"{action_name}_shard{shard_index}_remote_load",
                shard.bytes,
            )
            data_tag: Optional[int] = None
            if edge_rank != target_rank:
                data_tag = tag_allocator.take()
                builders[edge_rank].comm_send(
                    f"{action_name}_shard{shard_index}_send_to_rank{target_rank}",
                    src=edge_rank,
                    dst=target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                )
                builders[target_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_recv_from_edge{edge_rank}",
                    src=edge_rank,
                    dst=target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                )
            builders[target_rank].local_hbm_kv_restore(
                f"{action_name}_shard{shard_index}_target_hbm_write",
                shard.bytes,
            )
            target_hbm_completion_node_id = builders[target_rank].previous_id
            if target_hbm_completion_node_id is None:
                raise RuntimeError("target HBM restore did not generate a node")
            if transfer_anchor_sink is not None:
                # Metrics code-7 anchor (doc sec.4.3): the target HBM write
                # of this remote-load (restore) shard.  Observation only.
                transfer_anchor_sink(target_rank, target_hbm_completion_node_id)
            record.update(
                {
                    "control_rank": control_rank,
                    "request_tag": request_tag,
                    "data_tag": data_tag,
                    "direct_edge_access": edge_rank == target_rank,
                    "target_hbm_completion_node_id": (
                        target_hbm_completion_node_id
                    ),
                    "target_hbm_bandwidth_policy": (
                        "half_with_inference_then_full_after_peer_completion"
                    ),
                    "control_dependencies": [
                        "previous_decode_interval_gate",
                        "control_rank_prior_operations_including_remote_store_ack",
                    ],
                }
            )

        shard_records.append(record)

    return _kv_transfer_dict(transfer, shard_records)


def _emit_tp_readiness_barrier(
    *,
    builders: dict[int, TraceBuilder],
    group: InferenceGroup,
    name: str,
) -> dict[str, object]:
    """Synchronize a TP group after every rank-local KV preparation step."""

    node_ids_by_rank: list[list[int]] = []
    for rank in group.ranks:
        builders[rank].all_reduce(name, 1, group.pg_name)
        node_id = builders[rank].previous_id
        if node_id is None:
            raise RuntimeError("TP readiness barrier did not generate a node")
        node_ids_by_rank.append([rank, node_id])
    return {
        "name": name,
        "collective": "all_reduce",
        "comm_size_bytes": 1,
        "pg_name": group.pg_name,
        "ranks": list(group.ranks),
        "node_ids_by_rank": node_ids_by_rank,
    }


def _emit_tp_point_to_point_readiness_barrier(
    *,
    builders: dict[int, TraceBuilder],
    group: InferenceGroup,
    tag_allocator: TransferTagAllocator,
    name: str,
) -> dict[str, object]:
    """Synchronize a parallel branch without consuming collective order."""

    control_rank = group.ranks[0]
    if len(group.ranks) == 1:
        node_id = builders[control_rank].previous_id
        if node_id is None:
            raise RuntimeError("single-rank readiness barrier has no predecessor")
        return {
            "name": name,
            "collective": None,
            "protocol": "single_rank_dependency",
            "comm_size_bytes": 0,
            "control_rank": control_rank,
            "ranks": list(group.ranks),
            "node_ids_by_rank": [[control_rank, node_id]],
            "arrival_actions": [],
            "release_actions": [],
        }

    arrival_actions: list[dict[str, int]] = []
    for rank in group.ranks[1:]:
        tag = tag_allocator.take()
        builders[rank].comm_send(
            f"{name}_rank{rank}_arrive_send",
            src=rank,
            dst=control_rank,
            comm_size=1,
            comm_tag=tag,
        )
        send_node_id = builders[rank].previous_id
        builders[control_rank].comm_recv(
            f"{name}_rank{rank}_arrive_recv",
            src=rank,
            dst=control_rank,
            comm_size=1,
            comm_tag=tag,
        )
        recv_node_id = builders[control_rank].previous_id
        if send_node_id is None or recv_node_id is None:
            raise RuntimeError("TP readiness arrival did not generate nodes")
        arrival_actions.append(
            {
                "rank": rank,
                "tag": tag,
                "send_node_id": send_node_id,
                "control_recv_node_id": recv_node_id,
            }
        )

    release_actions: list[dict[str, int]] = []
    completion_nodes: dict[int, int] = {}
    for rank in group.ranks[1:]:
        tag = tag_allocator.take()
        builders[control_rank].comm_send(
            f"{name}_rank{rank}_release_send",
            src=control_rank,
            dst=rank,
            comm_size=1,
            comm_tag=tag,
        )
        send_node_id = builders[control_rank].previous_id
        builders[rank].comm_recv(
            f"{name}_rank{rank}_release_recv",
            src=control_rank,
            dst=rank,
            comm_size=1,
            comm_tag=tag,
        )
        recv_node_id = builders[rank].previous_id
        if send_node_id is None or recv_node_id is None:
            raise RuntimeError("TP readiness release did not generate nodes")
        completion_nodes[rank] = recv_node_id
        release_actions.append(
            {
                "rank": rank,
                "tag": tag,
                "control_send_node_id": send_node_id,
                "recv_node_id": recv_node_id,
            }
        )
    control_completion = builders[control_rank].previous_id
    if control_completion is None:
        raise RuntimeError("TP readiness control rank has no release completion")
    completion_nodes[control_rank] = control_completion

    return {
        "name": name,
        "collective": None,
        "protocol": "point_to_point_arrival_then_release",
        "comm_size_bytes": 1,
        "control_rank": control_rank,
        "ranks": list(group.ranks),
        "node_ids_by_rank": [
            [rank, completion_nodes[rank]] for rank in group.ranks
        ],
        "arrival_actions": arrival_actions,
        "release_actions": release_actions,
    }











def main(argv=None) -> None:  # noqa: ARG001
    """fail-closed 拒绝桩(2026-08-18 起生效):离线静态全管线入口已删除。

    本模块保留的仅是③④在线路径只读 import 的符号(config 装载/发射辅助/
    估算函数,见《路径功能代码对应说明.md》§4-a)。③④ 的输入物化入口是
    plan_materializer.py(manifest/metrics/runtime_config/face_lut);
    静态 ET 生成入口不再存在。
    """
    raise SystemExit(
        "path-1 (offline static full pipeline) was removed on 2026-08-18; "
        "use plan_materializer.py for the online routes' plan-dir inputs")


if __name__ == "__main__":
    main()
