#!/usr/bin/env python3
"""Shared WSC-LLM configuration and GraphBatch-emission primitives for online runs."""

from __future__ import annotations

import csv
import math
import sys
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
    # kv_reserve_context_tokens 为遗留观测键:loader/生成管线零消费,仅在管
    # trace_config.csv 携带、由 slo_tools/hbm_watermark.py 原始读 CSV 作 SLO
    # 观测口径(R_kv 保留水位机制未落地,策略说明 §4.4/§7 已标注)。
    # record_planning_iterations/output_prefix/output_dir 三零消费键已随
    # CSV 行退役(2026-09-27)。
    "kv_reserve_context_tokens": "0",
    # die 后端(wscllm 推广):模型位置编码声明(rope|none)。rope 时新 Q/K
    # 的旋转归入 qkv_rope 壳的末段(设计文档 §7.1);缺省 none 向后兼容。
    "position_encoding": "none",
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
    "request_queue_csv",
    "hardware_config",
    "system_template",
}
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
    request_queue_csv: Path
    request_queue: tuple[RequestSpec, ...]
    # P2-21（2026-09-28）：queue_index 两端同源 C++。WindowedTraceReader
    # 以未过滤全量 CSV 数据行序（0 起）编号 queue_index
    # （process_indexed_row: queue_index = data_rows_-1），而
    # request_queue 是 request_queue_session_limit 过滤后的保序子序列。
    # request_queue_indices 与 request_queue 逐位对齐，给出每行的源 CSV
    # 行序（= C++ queue_index，limit==0 时恒等）；queue_position_by_index
    # 为其逆映射（源行序 -> 过滤位次），供 config.request_queue[pos]
    # 定位（graph_batch_builder/scheduler）。
    request_queue_indices: tuple[int, ...]
    queue_position_by_index: dict[int, int]
    hardware_config: Path
    hardware_capacity_profile: str
    hardware: WscLlmHardware
    system_template: Path
    system_config: Path
    network_config: Path
    comm_group_config: Path
    remote_operand_loads: bool
    inference_groups: tuple[WscLlmInferenceGroup, ...]
    request_queue_session_limit: int
    trace_granularity: str
    prefill_chunk_size: int
    kv_cache_policy: str
    # die 后端(wscllm 推广):模型位置编码(rope|none;缺省 none)。
    position_encoding: str = "none"

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
        parsed_flag = parse_bool(value, key)
        # 3.2 fix (2026-09-29): this repo has no remote weight pool --
        # a node carrying remote_weight_bytes now fails at the C++ parse
        # boundary, and the flag that would emit such nodes fails HERE,
        # at the config boundary, instead of surviving load and aborting
        # the simulation at first GPU-comp issue.
        if parsed_flag:
            raise ValueError(
                f"config key {key}=true is not supported by this repo "
                "(no remote weight pool; weights are always resident in "
                "NPU local memory). Keep the key at false or remove it"
            )
        return parsed_flag
    if key == "request_queue_session_limit":
        return parse_nonnegative_int(value, key)
    if key == "prefill_chunk_size":
        return parse_int(value, key)
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
        if not value:
            raise ValueError(f"config key {key} must not be empty")
        return Path(value)
    if not value:
        raise ValueError(f"config key {key} must not be empty")
    return value


def _to_wsc_llm_hardware(hardware: ResolvedHardware) -> WscLlmHardware:
    die_peak_int4_ops_s = None
    die_peak_vec_eqops_s = None
    if hardware.die_profile == "coarse_die_v2":
        derived = hardware.die_derived
        summary = derived.get("hardware_summary") if isinstance(derived, dict) else None
        if not isinstance(summary, dict):
            raise ValueError(
                "coarse_die_v2 hardware is missing typed hardware_summary")

        def typed_positive_rate(key: str) -> float:
            value = summary.get(key)
            try:
                finite = math.isfinite(value)
            except (OverflowError, TypeError):
                finite = False
            if (isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not finite or value <= 0):
                raise ValueError(
                    "coarse_die_v2 hardware_summary.{} must be a finite "
                    "positive rate in its declared units".format(key))
            return value

        die_peak_int4_ops_s = typed_positive_rate("peak_int4_ops_per_s")
        die_peak_vec_eqops_s = typed_positive_rate(
            "peak_vec_fp32_eq_ops_per_s")
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
        die_profile=hardware.die_profile,
        die_peak_int4_ops_s=die_peak_int4_ops_s,
        die_peak_vec_eqops_s=die_peak_vec_eqops_s,
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


def queue_index_mappings(
    source_request_queue: Sequence[RequestSpec],
    selected_requests: Sequence[RequestSpec],
) -> tuple[tuple[int, ...], dict[int, int]]:
    """P2-21（2026-09-28）：queue_index 两端同源 C++ 的映射对。

    C++ WindowedTraceReader 对它读到的队列 CSV 按"数据行序（0 起）"编号
    queue_index（process_indexed_row: queue_index = data_rows_ - 1）；
    session_limit 过滤只发生在 Python 侧（selected_requests 为源序列的
    保序子序列），按过滤后序 enumerate 会与 C++ 错位（潜伏缺陷：
    生产 trace_config 恒 limit=0 不触发）。返回：

      * ``request_queue_indices`` —— 与 selected_requests 逐位对齐的
        源 CSV 行序元组（manifest/metrics_manifest 的 requests[].
        queue_index 取此值，与 C++ 注册/连接序一致）；
      * ``queue_position_by_index`` —— 源行序 -> 过滤位次的逆映射
        （config.request_queue[pos] 定位用；limit==0 时恒等）。
    """
    selected_ids = {request.request_id for request in selected_requests}
    indices: list[int] = []
    position_by_index: dict[int, int] = {}
    for ordinal, request in enumerate(source_request_queue):
        if request.request_id in selected_ids:
            position_by_index[ordinal] = len(indices)
            indices.append(ordinal)
    return tuple(indices), position_by_index


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
    request_queue, _ = select_first_session_requests(
        source_request_queue,
        int(parsed["request_queue_session_limit"]),
    )
    # P2-21（2026-09-28）：queue_index 两端同源 C++（未过滤全量 CSV 数据
    # 行序）——过滤态下按过滤后序枚举与 WindowedTraceReader 错位。
    request_queue_indices, queue_position_by_index = queue_index_mappings(
        source_request_queue, request_queue)
    hardware_path = _resolve_from_sh_test(parsed["hardware_config"])
    hardware_capacity_profile = str(parsed["local_hbm_capacity_profile"])
    resolved_hardware = load_hardware_config(
        hardware_path,
        hardware_capacity_profile,
    )
    hardware = _to_wsc_llm_hardware(resolved_hardware)
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

    # die 后端:位置编码键的值域校验(rope|none;其他值 fail-closed,与
    # face 侧 generate_face_trace.py 同款最小增量)。
    position_encoding = str(parsed["position_encoding"])
    if position_encoding not in ("rope", "none"):
        raise ValueError(
            "config key 'position_encoding' must be 'rope' or 'none', got "
            f"{position_encoding!r}")

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
        request_queue_csv=request_queue_csv,
        request_queue=request_queue,
        request_queue_indices=request_queue_indices,
        queue_position_by_index=queue_position_by_index,
        hardware_config=hardware_path,
        hardware_capacity_profile=hardware_capacity_profile,
        hardware=hardware,
        system_template=system_template,
        system_config=runtime_configs.system,
        network_config=runtime_configs.network,
        comm_group_config=runtime_configs.comm_group,
        remote_operand_loads=bool(parsed["remote_operand_loads"]),
        inference_groups=tuple(groups),
        request_queue_session_limit=int(parsed["request_queue_session_limit"]),
        trace_granularity=str(parsed["trace_granularity"]),
        prefill_chunk_size=int(parsed["prefill_chunk_size"]),
        kv_cache_policy=str(parsed["kv_cache_policy"]),
        position_encoding=position_encoding,
    )






def _stage_tag(queue_index: int, category: int, relative_rank: int) -> int:
    # 4.1 tag-domain guards (2026-09-29): the 10000-wide per-request block
    # assumes category+relative_rank stays inside it -- a TP/rank setup
    # pushing past the block width would alias the NEXT request's tags
    # (cross-request tag crosstalk, silent). And the total must stay inside
    # the C++ NATIVE front-end window [0, 5e8): queue_index >= 50000 made
    # the C++ sys_panic mid-run instead of failing here at the producer.
    # Both bounds fail closed at the single tag factory.
    if category + relative_rank >= 10000:
        raise ValueError(
            f"stage tag block overflow: category({category}) + "
            f"relative_rank({relative_rank}) >= 10000 -- the per-request "
            "tag block is 10000 wide and would alias the next request's "
            "tags (rank/TP configuration too large for the tag domain)"
        )
    tag = queue_index * 10000 + category + relative_rank
    if tag >= 500000000:
        raise ValueError(
            f"stage tag {tag} out of the C++ NATIVE window [0, 500000000): "
            f"queue_index {queue_index} too large (max 49999)"
        )
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
    builders: dict[int, "OnlineTraceBuilder"],
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
    builders: dict[int, "OnlineTraceBuilder"],
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
    builders: dict[int, "OnlineTraceBuilder"],
    group: WscLlmInferenceGroup,
    prefix: str,
    stage: str,
    tokens: int,
    initial_context_tokens: int,
    die_ctx=None,
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
        rank_die_ctx = None
        if die_ctx is not None:
            # die 后端(wscllm 推广):prefill/recompute 阶段逐 rank 写入
            # 发射语境(phase=prefill;span 记录带绝对位置,恒等式见
            # die_model_adapter 模块 docstring)。
            die_ctx.begin_emission(
                phase="prefill",
                tensor_parallel=tp,
                tensor_parallel_rank=relative_rank,
                span_namespace=f"{prefix}_{stage}",
                pass_spans=spans,
            )
            rank_die_ctx = die_ctx
        pass_count = transformer_pass_aggregated(
            builders[rank], phase=f"{prefix}_{stage}_request_aggregated",
            pass_spans=spans, layers=config.layers, hidden_size=config.hidden_size,
            ffn_size=config.ffn_size, tensor_parallel=tp, pg_name=group.pg_name,
            vocab_size=config.vocab_size, bytes_per_elem=config.bytes_per_elem,
            num_heads=config.num_heads, tensor_parallel_rank=relative_rank,
            mlp_variant=config.mlp_variant,
            die_ctx=rank_die_ctx,
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
