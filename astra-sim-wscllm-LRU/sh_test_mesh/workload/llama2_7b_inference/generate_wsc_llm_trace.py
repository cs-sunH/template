#!/usr/bin/env python3
"""Shared WSC-LLM configuration and GraphBatch-emission primitives for online runs."""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Sequence


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
from session_kv_manager import (  # noqa: E402
    KVTransfer,
    KVTransferShard,
    physical_edge_ranks,
)
from generate_trace import (  # noqa: E402
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
    "kv_cache_policy": "session_lru_tiered",
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


# ---------------------------------------------------------------------------
# B3(2026-09-06) KV 转移发射三件套(照抄 sh_2.0 generate_face_trace.py
# :145-167 的类形与语义;契约 §6:tag 基址与 _stage_tag 的
# queue_index*10000+{1000,1900,3000} 段错开)。
# H8(2026-09-24)修正:基址自 10_000_000 上移至 100_000_000——30s 窗口
# 源 trace 实测 1177 行,queue_index≥1000 的 _stage_tag 已进入旧基址的
# 分配器独占段;新基址下队列 ≤9999 行时两段保持不相交,且 _stage_tag
# 侧补了越界 fail-closed 守卫(见 _stage_tag)。
# ---------------------------------------------------------------------------


@dataclass
class PendingHistoryGate:
    """跨请求 history 门:上一 turn 完成 → 下一 turn 准入之间的会话
    KV 位置账本(timer_gates 在下一 turn 发射时由 interval gate 节点 id
    填充;location 由 sync_pending_history_after_evictions 镜像逐出)。"""

    source_instance_index: int
    timer_gates: tuple[Optional[int], ...]
    location: str


@dataclass(frozen=True)
class TransferTriggerGate:
    """逐出发射的触发门(control 实例各 rank 的节点门;control rank 与
    数据源 rank 不同实例时以 1B p2p 触发,同 rank 直接 arm)。"""

    control_instance_index: int
    node_gates: tuple[Optional[int], ...]


class TransferTagAllocator:
    """KV 转移通信 tag 单调分配器(基址 100_000_000;上限 0xFFFFFFFF)。

    契约 §6 硬约束:sh_2.0 原版从 1 起会撞进本仓 ``queue_index*10000+
    {1000,1900,3000}`` 的 _stage_tag 段,统一错开到安全高位段。H8
    (2026-09-24):基址 10_000_000 只能容纳 <1000 行队列(30s 窗口实测
    1177 行已重叠),上移至 100_000_000(队列 ≤9999 行不相交;tag 仍处
    C++ NATIVE 段 [0, 5e8) 内,见 Sys.hh FrontEndSendRecvType)。"""

    _TAG_BASE = 100_000_000

    def __init__(self) -> None:
        self._next_tag = self._TAG_BASE

    def take(self) -> int:
        if self._next_tag > 0xFFFFFFFF:
            raise OverflowError("KV transfer communication tag space was exhausted")
        tag = self._next_tag
        self._next_tag += 1
        return tag


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
        # 唯一取值 session_lru_tiered(session 级二态冷热管理:完整本地/
        # 完整远端、整体 LRU 逐出 + 远端池全量恢复;2026-09-25 session 级
        # Tiered-LRU 批起原 PARTIAL 半层化三态与两段式逐出已物理移除);
        # 旧值别名与其同调度器(无档位分支),
        # 已随 2026-09-25 命名卫生直接清除。其余取值 fail-closed。
        if value != "session_lru_tiered":
            raise ValueError(
                "config key kv_cache_policy must be session_lru_tiered, "
                f"got {value!r}"
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


def _stage_tag(queue_index: int, category: int, relative_rank: int) -> int:
    tag = queue_index * 10000 + category + relative_rank
    if tag >= TransferTagAllocator._TAG_BASE:
        raise OverflowError(
            f"stage tag {tag} (queue_index={queue_index}) would enter the "
            "KV transfer allocator's exclusive tag segment "
            f"[{TransferTagAllocator._TAG_BASE}, 0xFFFFFFFF); shrink the "
            "request queue or raise the allocator base")
    return tag


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
    sec.6.2/6.3); the emitted nodes are unchanged.

    分 chunk 聚合发射(trace chunking)是合法物理机制,完整保留;首 chunk
    层段拆分真流水(B3,只为 PARTIAL 部分恢复服务)已随该恢复路径删除。
    """

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


def _validate_transfer_shard(
    config: WscLlmTraceConfig,
    transfer: KVTransfer,
    shard: KVTransferShard,
) -> None:
    """KV 转移 shard 发射前校验(照抄 sh_2.0 :465-510;边缘成员资格按
    本仓 hardware 的网格边界 rank 集合判定——B2 构造期 nearest_edge
    与此同源)。"""
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

    if shard.edge_rank is not None and shard.edge_rank not in (
            physical_edge_ranks(config.hardware)):
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
    """发射侧 KVTransfer 记录(照抄 sh_2.0 :513-534;含发射产生的
    tag/依赖元数据 shard_records)。"""
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
    group_by_index: dict[int, WscLlmInferenceGroup],
    pending_gate: PendingHistoryGate,
    relative_index: int,
) -> tuple[int, Optional[int]]:
    """history 门控制通道:pending 门源实例的 relative_index rank 与其
    timer gate(照抄 sh_2.0 :537-546)。"""
    source_group = group_by_index[pending_gate.source_instance_index]
    if relative_index >= len(source_group.ranks):
        raise ValueError("history control rank index exceeds source TP degree")
    return source_group.ranks[relative_index], pending_gate.timer_gates[relative_index]


def _emit_transfer_trigger(
    *,
    builders: dict[int, TraceBuilder],
    group_by_index: dict[int, WscLlmInferenceGroup],
    tag_allocator: TransferTagAllocator,
    trigger_gate: Optional[TransferTriggerGate],
    relative_index: int,
    source_rank: int,
    action_name: str,
    shard_index: int,
) -> dict[str, object]:
    """逐出动作的触发门发射(照抄 sh_2.0 :549-596):control rank 与
    数据源 rank 同 rank → 直接 arm 节点门;跨 rank → 1B p2p 触发
    (trigger 按字节正常计费,量级可忽略)。"""
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
    config: WscLlmTraceConfig,
    builders: dict[int, TraceBuilder],
    group_by_index: dict[int, WscLlmInferenceGroup],
    tag_allocator: TransferTagAllocator,
    transfer: KVTransfer,
    action_name: str,
    pending_gate: Optional[PendingHistoryGate] = None,
    trigger_gate: Optional[TransferTriggerGate] = None,
    transfer_anchor_sink: Optional[Callable[[int, int], None]] = None,
) -> dict[str, object]:
    """发射一个 KV 转移动作(照抄 sh_2.0 :599-902,适配本仓 config/组
    类型;HBM 计费三险逐节点核对见策略文档 §9 地图):

    - local_hit:无数据 shard,仅按 pending 门 arm 到达/interval gates;
    - noc_migrate:shard send/recv(两端默认 charged)+ 1B ack 对(源端
      收 ack 后才物理释放);
    - remote_store 链 A(source≠edge):源 comm_send(源端 COMM_READ
      唯一数据计费)→ edge comm_recv(hbm_charge=False 过路零计费)→
      edge mem_store(池写,本地零计费)→ 1B ack 双端各 1B;
    - remote_store 链 B(source==edge 直连):mem_store(hbm_access_mode=1)
      = POOL_READ 唯一计费 + 池端口 FIFO 双异步 join;
    - remote_load:1B arm/trigger → edge mem_load(仅池 FIFO)→ edge
      comm_send(hbm_charge=False)→ target comm_recv(hbm_charge=False,
      目标写由 restore 承担)→ target local_hbm_kv_restore(RESTORE
      唯一数据计费)+ code-7 anchor 观测不计费。

    计费单次性:每字节每链恰好一次本地 HBM 计费 + 一次池端口 FIFO;
    禁止漏 hbm_charge=False / 直连漏 hbm_access_mode=1 / restore 退化
    普通 mem_load(契约 §12.5 红线)。"""
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
                # 池存储端点即边缘rank(单 mem_store 路径):边缘rank从本地
                # HBM 读出该分片再经 SerDes 端口写入池——HBM 读计费
                # (hbm_access_mode=1,bytes=tensor_size),MEM 完成=join(端口
                # 事务, HBM 作业)。
                builders[edge_rank].mem_store(
                    f"{action_name}_shard{shard_index}_edge_store",
                    shard.bytes,
                    hbm_access_mode=1,
                )
                # 逐出旁路支链尾部观测(2026-09-13):直连路径的池写完成
                # 节点即源端释放节点。发射内部零改动,仅暴露节点 id 供
                # builder 登记 pending_store_tails(store→restore 前递依赖)。
                edge_store_node_id = builders[edge_rank].previous_id
                record.update(
                    {
                        "direct_edge_access": True,
                        "source_release_dependency": "mem_store_completion",
                        "edge_store_node_id": edge_store_node_id,
                        "source_ack_recv_node_id": edge_store_node_id,
                    }
                )
            else:
                data_tag = tag_allocator.take()
                ack_tag = tag_allocator.take()
                # 源rank的 comm_send 自动计费(发送端 HBM 读);数据经 NoC
                # 到达边缘rank后由 SerDes 直通写池——边缘rank的 comm_recv
                # 与 mem_store 均为直通,不占其本地 HBM(hbm_charge=False /
                # 不标注)。
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
                    hbm_charge=False,
                )
                builders[edge_rank].mem_store(
                    f"{action_name}_shard{shard_index}_remote_store",
                    shard.bytes,
                )
                # 逐出旁路支链尾部观测(2026-09-13):池写完成节点 + 源端
                # ack recv 节点。发射内部零改动,仅暴露节点 id 供 builder
                # 登记 pending_store_tails(store→restore 前递依赖)。
                edge_store_node_id = builders[edge_rank].previous_id
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
                source_ack_recv_node_id = builders[source_rank].previous_id
                record.update(
                    {
                        "direct_edge_access": False,
                        "data_tag": data_tag,
                        "ack_tag": ack_tag,
                        "source_release_dependency": "remote_store_ack_recv",
                        "edge_store_node_id": edge_store_node_id,
                        "source_ack_recv_node_id": source_ack_recv_node_id,
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
                # 边缘rank的 mem_load(SerDes->NoC 直通)与 comm_send
                # (NoC 发射端直通)均不占其本地 HBM;目标rank的 comm_recv
                # 也不计费——其 HBM 写由紧随的 restore 节点(串行语义)承担,
                # 每字节流恰好计费一次。
                builders[edge_rank].comm_send(
                    f"{action_name}_shard{shard_index}_send_to_rank{target_rank}",
                    src=edge_rank,
                    dst=target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                    hbm_charge=False,
                )
                builders[target_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_recv_from_edge{edge_rank}",
                    src=edge_rank,
                    dst=target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                    hbm_charge=False,
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
                        "n_way_equal_split_with_all_active_hbm_users"
                    ),
                    "control_dependencies": [
                        "previous_decode_interval_gate",
                        "control_rank_prior_operations_including_remote_store_ack",
                    ],
                }
            )

        shard_records.append(record)

    return _kv_transfer_dict(transfer, shard_records)


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
