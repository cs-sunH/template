#!/usr/bin/env python3
"""Shared FACE configuration and GraphBatch-emission primitives for online runs."""

from __future__ import annotations

import csv
import json
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

from face_scheduler import (  # noqa: E402
    DecodeCandidateCost,
    FaceHardware,
    FaceRooflineEstimate,
    FaceModel,
    kv_cache_bytes_for_tokens,
)
from session_kv_manager import (  # noqa: E402
    NOC_MIGRATE,
    KVTransfer,
    KVTransferShard,
    physical_edge_ranks,
)
from generate_trace import (  # noqa: E402
    InferenceGroup,
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
CONFIG_COLUMNS = ("kind", "key", "value", "group_name", "pg_name", "ranks")
REQUIRED_CONFIG_KEYS = (
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
    "mlp_variant",
    "request_queue_csv",
    "hardware_config",
    "local_hbm_capacity_profile",
    "system_template",
    "remote_operand_loads",
)
OPTIONAL_CONFIG_DEFAULTS = {
    "request_queue_session_limit": "0",
    # 缺省即在线主链唯一支持的档位(graph_batch_builder 全部发射入口对
    # 非 request_aggregated fail-closed);token_expanded 属已删除的离线
    # 静态管线(main 自 2026-08-18 起为 fail-closed 拒绝桩),不再作默认。
    "trace_granularity": "request_aggregated",
    "prefill_chunk_size": "512",
    "kv_cache_policy": "session_lru_tiered",
    "kv_reserve_context_tokens": "0",
    # 阶段 C(契约 §2/设计文档 §7.1;face-LRU 随迁 2026-10-03):模型位置
    # 编码声明;无该键默认 none(向后兼容——旧 trace_config 行为不变)。
    # rope 时新 Q/K 旋转归入 qkv_rope 壳的末段
    # (die_op_spec.model_dims.position_encoding)。
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
class FaceTraceConfig:
    config_csv: Path
    npus_count: int
    layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    vocab_size: int
    bytes_per_elem: int
    mlp_variant: str
    request_queue_csv: Path
    request_queue: tuple[RequestSpec, ...]
    hardware_config: Path
    hardware_capacity_profile: str
    hardware: FaceHardware
    # P3(face-LRU 审计,2026-09-29):原 system_template 字段删除(全仓
    # grep 零读者;模板路径仍经 materialize_runtime_configs 的
    # system_template_path 参数消费,parse/REQUIRED_CONFIG_KEYS 不变)。
    system_config: Path
    network_config: Path
    comm_group_config: Path
    remote_operand_loads: bool
    inference_groups: tuple[InferenceGroup, ...]
    request_queue_session_limit: int
    selected_session_ids: tuple[str, ...]
    source_request_count: int
    source_session_count: int
    trace_granularity: str
    prefill_chunk_size: int
    kv_cache_policy: str
    kv_reserve_context_tokens: int
    position_encoding: str

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



def _resolve_from_sh_test(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SH_TEST_DIR / path).resolve()


def _resolve_request_queue(path: Path) -> Path:
    if path.is_absolute():
        return path
    return (SH_TEST_DIR / "workload" / path).resolve()


def _frontier_neutral(builder, emit_fn, *arm_ids):
    """死锁真修（方案①，2026-10-11，自 WL 同构移植）：把一段发射包成
    **不推进主链 frontier** 的中立段——checkpoint → 暂存/清空 pending
    依赖 → 显式 arm 前驱节点（排序语义由显式边承担，不再靠链位）→
    发射（段内节点自行接续成链）→ restore 主链。与
    graph_batch_builder._emit_side_branch 的差别：单 builder 粒度、
    arm 显式化。

    - arm_ids：进入本段前显式 arm 的同 rank 前驱节点 id（None 跳过）；
      由本段首节点消费。
    - 返回值：段尾节点 id（emit_fn 末个发射节点；调用方显式边落账用，
      与取 previous_id 同款口径）。emit_fn 必须至少发射一个节点（arm
      未消费即触发上面的泄漏 raise）。
    - 泄漏 fail-closed：emit_fn 若未消费全部显式 arm、或留下新的未消费
      pending 依赖（触发门泄漏），raise——与逐出旁路支链同款纪律。
    """
    checkpoint = builder.chain_checkpoint()
    if builder.pending_extra_dependencies:
        builder.pending_extra_dependencies = []
    for arm_id in arm_ids:
        if arm_id is not None:
            builder.arm_dependency(arm_id)
    try:
        emit_fn()
        if builder.pending_extra_dependencies:
            raise RuntimeError(
                "frontier-neutral emission left unconsumed pending "
                "dependencies on rank {}".format(builder.rank))
        tail = builder.previous_id
        if tail is None or tail == checkpoint[0]:
            raise RuntimeError(
                "frontier-neutral emission produced no node on rank "
                "{}".format(builder.rank))
    finally:
        builder.restore_chain(checkpoint)
    return tail


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
    if key == "kv_cache_policy":
        # 唯一取值 session_lru_tiered(session 级二态冷热管理:完整本地/
        # 完整远端、单阶段完整 session LRU 逐出 + 远端池全量恢复;
        # 2026-09-25 session 级 Tiered-LRU 批起原 PARTIAL 半层化三态与
        # 两段式逐出已物理移除);旧值别名与其同调度器(无档位分支),
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
    if key == "position_encoding":
        if value not in {"rope", "none"}:
            raise ValueError(
                "config key position_encoding must be rope or none, got "
                f"{value!r}"
            )
        return value
    if key in PATH_CONFIG_KEYS:
        if not value:
            raise ValueError(f"config key {key} must not be empty")
        return Path(value)
    if not value:
        raise ValueError(f"config key {key} must not be empty")
    return value


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


def load_face_trace_config(config_csv: Path = CONFIG_CSV_PATH) -> FaceTraceConfig:
    if not config_csv.exists():
        raise FileNotFoundError(f"trace config CSV not found: {config_csv}")
    values: dict[str, str] = {}
    groups: list[InferenceGroup] = []
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
        # py-sched-E5 (2026-09-30): extra-column branch. The old check only
        # detected MISSING columns -- a typo'd or surplus header column (and
        # the restkey debris its overlong data rows produce) passed freely
        # and was silently ignored downstream. Extra columns are rejected
        # at this earliest boundary unless they are the frozen
        # documentation-only column ("description" -- this repo's
        # trace_config.csv carries it as inline documentation; the loader
        # never reads it), aligning the header gate with the C++ windowed
        # reader's strict column-count stance.
        extra_columns = sorted(
            columns - set(CONFIG_COLUMNS) - {"description"})
        if extra_columns:
            raise ValueError(
                "trace config CSV has unknown column(s) "
                f"{', '.join(extra_columns)}; allowed columns are exactly: "
                + ",".join(CONFIG_COLUMNS)
                + " (plus the documentation-only 'description' column)"
            )
        for line_number, raw_row in enumerate(reader, start=2):
            row = clean_csv_row(raw_row, context=f"line {line_number}: ")
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
                if not name or not pg_name:
                    raise ValueError(
                        f"line {line_number}: inference_group requires name and pg_name"
                    )
                # P2(face-LRU 审计,2026-09-29):重名组 fail-closed 拒绝。
                # 实例同一性在两处以不同口径判定:调度器按 instance 索引、
                # 发射器(_paired_transfer/_emit_control_trigger)按组名字符
                # 串相等;重名组会让"跨实例迁移"在发射层被误判为同实例
                # 零节点,迁移链零字节零时延,物理-逻辑静默分叉(fail-open)。
                # 组名是实例身份,必须唯一。
                if any(existing.name == name for existing in groups):
                    raise ValueError(
                        f"line {line_number}: duplicate inference_group "
                        f"name {name!r} (instance identity is name-based; "
                        f"group names must be unique)"
                    )
                groups.append(
                    InferenceGroup(
                        name=name,
                        pg_name=pg_name,
                        ranks=parse_rank_spec(row.get("ranks", "")),
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
        # fail-closed (request-neutral): missing input must abort at the
        # official entry point with a materialization hint
        # (load_request_queue itself is also fail-closed and never
        # synthesizes a queue).
        sys.exit(
            f"missing request queue: {request_queue_csv}; "
            "materialize the input via traces/derive_20_first_30_seconds.py "
            "(its stdout is the authoritative provenance record)"
        )
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
    hardware = _to_face_hardware(resolved_hardware)
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

    return FaceTraceConfig(
        config_csv=config_csv.resolve(),
        npus_count=npus_count,
        layers=int(parsed["layers"]),
        hidden_size=int(parsed["hidden_size"]),
        ffn_size=int(parsed["ffn_size"]),
        num_heads=int(parsed["num_heads"]),
        vocab_size=int(parsed["vocab_size"]),
        bytes_per_elem=int(parsed["bytes_per_elem"]),
        mlp_variant=str(parsed["mlp_variant"]),
        request_queue_csv=request_queue_csv,
        request_queue=request_queue,
        hardware_config=hardware_path,
        hardware_capacity_profile=hardware_capacity_profile,
        hardware=hardware,
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
        position_encoding=str(parsed["position_encoding"]),
    )






def _stage_tag(queue_index: int, category: int, relative_rank: int) -> int:
    """Stage tag = ``queue_index*10000 + category + relative_rank``.

    值域边界(契约 §6):queue_index < 1000 时恒在 KV-transfer 分配器基址
    10_000_000 之下;queue_index >= 1000 的请求(30s 验收队列 1177 请求
    即存在)其 stage tag 数值上进入分配器值域——当前无守卫拦截,属已登记
    的潜伏重叠:执行驱动 C++ 路径不以 comm_tag 配对(Workload.cc 仅有
    TODO 记载),重叠仅是审计层面的值域混用,不改变任何发射时序,故此处
    不 fail-closed(会在验收规模直接拒绝合法运行)。
    """
    return queue_index * 10000 + category + relative_rank


def _xy_route(hardware: FaceHardware, source: int, target: int) -> list[int]:
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
    config: FaceTraceConfig,
    builders: dict[int, "OnlineTraceBuilder"],  # (2026-10-01 R2) 注解修正:函数体使用 mem_store(hbm_access_mode)/comm_recv(hbm_charge)/local_hbm_kv_restore——均为 OnlineTraceBuilder 专有 API,基类 TraceBuilder 无之(按旧注解驱动会 TypeError)。
    queue_index: int,
    category: int,
    name: str,
    source_group: InferenceGroup,
    target_group: InferenceGroup,
    total_bytes: int,
    timer_gates: Optional[dict[int, Optional[int]]] = None,
    extra_send_arm_ids: Optional[dict[int, Sequence[int]]] = None,
) -> list[dict[str, object]]:
    """死锁真修（方案①，2026-10-11，自 WL 同构移植）：send/recv 中立化——
    不占主链 frontier；门与前驱改显式 arm；recv 节点 id 随 route 落账
    （消费方显式边用：join 标记 C5=arm(xfer_recv)、准入屏障 C4=
    arm(data recv)）。extra_send_arm_ids：send 中立段进入前显式 arm 的
    额外同 rank 前驱表 {source_rank: [节点 id]}（跨实例时序控制 recv 等，
    承载原链位序）。"""
    if len(source_group.ranks) != len(target_group.ranks):
        raise ValueError("FACE direct KV shard pairing requires equal TP degree")
    if source_group.name == target_group.name:
        if timer_gates is not None:
            for rank in target_group.ranks:
                builders[rank].arm_timer_gate(timer_gates.get(rank))
        return []

    # Full KV transfers retain the configured whole-head ownership.  An
    # arbitrary capacity-split remainder falls back to exact byte partitioning
    # so the total transfer volume is never rounded up.
    # fail-closed(与 _validate_transfer_shard 的 shard.bytes<=0 门同款):
    # 0 值经 comm_size 的 max(1,·) 会被静默钳成 1 字节,绝不发射。
    if total_bytes <= 0:
        raise ValueError("FACE paired KV transfers must carry positive bytes")
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
        if bytes_by_relative_rank[relative_index] <= 0:
            raise ValueError("emitted KV transfer shards must contain positive bytes")
        send_arm_ids = [timer_gates.get(source)] if timer_gates is not None else []
        send_arm_ids.extend((extra_send_arm_ids or {}).get(source) or ())
        tag = _stage_tag(queue_index, category, relative_index)
        _frontier_neutral(
            builders[source],
            lambda source=source, target=target, tag=tag:
                builders[source].comm_send(
                    f"{name}_send_rank{source}_to_rank{target}",
                    src=source,
                    dst=target,
                    comm_size=bytes_by_relative_rank[relative_index],
                    comm_tag=tag),
            *send_arm_ids)
        recv_node_id = _frontier_neutral(
            builders[target],
            lambda source=source, target=target, tag=tag:
                builders[target].comm_recv(
                    f"{name}_recv_rank{source}_to_rank{target}",
                    src=source,
                    dst=target,
                    comm_size=bytes_by_relative_rank[relative_index],
                    comm_tag=tag))
        path = _xy_route(config.hardware, source, target)
        routes.append(
            {
                "relative_shard": relative_index,
                "source_rank": source,
                "target_rank": target,
                "bytes": bytes_by_relative_rank[relative_index],
                "noc_path": path,
                "noc_hops": len(path) - 1,
                "recv_node_id": recv_node_id,
            }
        )
    return routes



def _roofline_estimate_dict(entry: FaceRooflineEstimate) -> dict[str, object]:
    return {
        "instance_size": entry.instance_size,
        "p_chunk": entry.p_chunk,
        "d_batch": entry.d_batch,
        "d_token": entry.d_token,
        "iteration_time_ns": entry.iteration_time_ns,
        "source": entry.source,
    }


def _candidate_dict(candidate: DecodeCandidateCost) -> dict[str, object]:
    return {
        "instance_index": candidate.instance_index,
        "weighted_distance": candidate.weighted_distance,
        "current_roofline": _roofline_estimate_dict(candidate.current_roofline),
        "updated_roofline": _roofline_estimate_dict(candidate.updated_roofline),
        "delta_time_ns": candidate.delta_time_ns,
        "per_die_delta_ns": candidate.per_die_delta_ns,
    }




def _eviction_dict(record: object) -> dict[str, object]:
    return {
        "time_ns": record.time_ns,
        "phase": record.phase,
        "reason": record.reason,
        "trigger_request_id": record.trigger_request_id,
        "victim_session_id": record.victim_session_id,
        "victim_instance_index": record.victim_instance_index,
        "victim_last_completion_ns": record.victim_last_completion_ns,
        "context_tokens": record.context_tokens,
        "shard_bytes": list(record.shard_bytes),
    }


def _emit_control_trigger(
    *,
    builders: dict[int, TraceBuilder],
    queue_index: int,
    name: str,
    source_group: InferenceGroup,
    target_group: InferenceGroup,
    timer_gates: dict[int, Optional[int]],
) -> dict[int, int]:
    """Carry a timing dependency across instances without inventing KV bytes.

    死锁真修（方案①，2026-10-11，自 WL 同构移植）：1B control send/recv
    为簿记握手节点，改经 _frontier_neutral 中立化——不占主链 frontier、
    排序改显式边（send 段显式 arm 控制门；recv 段叶）。返回值
    {target_rank: control_recv_node_id}——调用方把 recv id 交给目标 rank
    的后续主链消费者显式 arm（准入 readiness 屏障 C4 式），跨实例时序
    承载从链位改为显式边，语义不变。"""

    if source_group.name == target_group.name:
        for rank in target_group.ranks:
            builders[rank].arm_timer_gate(timer_gates.get(rank))
        return {}
    source = source_group.ranks[0]
    target = target_group.ranks[0]
    tag = _stage_tag(queue_index, 1900, 0)
    _frontier_neutral(
        builders[source],
        lambda: builders[source].comm_send(
            f"{name}_control_send_rank{source}_to_rank{target}",
            src=source,
            dst=target,
            comm_size=1,
            comm_tag=tag),
        timer_gates.get(source))
    control_recv_id = _frontier_neutral(
        builders[target],
        lambda: builders[target].comm_recv(
            f"{name}_control_recv_rank{source}_to_rank{target}",
            src=source,
            dst=target,
            comm_size=1,
            comm_tag=tag))
    return {target: control_recv_id}


# ---------------------------------------------------------------------------
# B3 (2026-09-06, tiered-KV physical emission; sh_2.0 generate_face_trace.py
# 的 KV-transfer 发射族抽取适配): the KV-transfer emission family for the
# remote_store / remote_load physical chains, the history/trigger gates, and
# the p2p readiness barrier.  Node naming copies the sh_2.0 call sites
# (contract §7); no emitted name may contain the "first_token" or
# "batch_train_" substrings (C++ name anchors owned by other mechanisms).
# ---------------------------------------------------------------------------


@dataclass
class PendingHistoryGate:
    """sh_2.0 generate_face_trace.py 的 PendingHistoryGate: cross-request
    history gate bookkeeping.

    ``timer_gates`` carries the arrival (turn-0) or interval (turn>0) gate
    node ids (``None`` entries arm nothing); ``location`` tracks the session
    KV location as of the last completion / eviction mirroring point.
    """

    source_instance_index: int
    timer_gates: tuple[Optional[int], ...]
    location: str


@dataclass(frozen=True)
class TransferTriggerGate:
    """sh_2.0 generate_face_trace.py 的 TransferTriggerGate: control-side
    node gates for eviction transfers."""

    control_instance_index: int
    node_gates: tuple[Optional[int], ...]


class TransferTagAllocator:
    """sh_2.0 generate_face_trace.py 的 TransferTagAllocator,with the -LRU
    tag-space offset (contract §6).

    face's stage tag formula ``queue_index*10000+{1000,1900,3000}`` occupies
    the low segments only while ``queue_index < 1000`` (see _stage_tag); the
    monotonic allocator starts at 10_000_000 so the two spaces stay disjoint
    under that queue-scale assumption.  comm_tag is not used for p2p matching
    by the execution-driven C++ path, so the numeric overlap beyond that
    scale is audit-level only.
    """

    def __init__(self) -> None:
        self._next_tag = 10_000_000

    def take(self) -> int:
        if self._next_tag > 0xFFFFFFFF:
            raise OverflowError("KV transfer communication tag space was exhausted")
        tag = self._next_tag
        self._next_tag += 1
        return tag


def _config_edge_ranks(config: FaceTraceConfig) -> tuple[int, ...]:
    """Edge ranks for shard validation: the deterministic mesh-boundary
    derivation (npu-selection is "mesh-boundary").  FaceTraceConfig carries
    no resolver remote-memory override, so this is the only edge-rank
    source."""
    return physical_edge_ranks(config.hardware)


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

    if shard.edge_rank is not None and shard.edge_rank not in _config_edge_ranks(config):
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
    """sh_2.0 generate_face_trace.py 的触发门发射(按函数名对应):control rank
    与数据源 rank 同 rank → 直接 arm 节点门;跨 rank → 1B p2p 触发
    (trigger 按字节正常计费,量级可忽略)。

    死锁真修（方案①，2026-10-11，自 WL 同构移植）：trigger send/recv 为
    簿记握手节点，改经 _frontier_neutral 中立化——不占主链 frontier、
    排序改显式边（send 段显式 arm trigger 门；跨 rank 时数据源侧的数据
    send 由调用方显式 arm trigger_recv_node_id）。门在 control==source 时
    不在此 arm，由返回的 trigger_dependency_node_id 交给调用方传入数据
    send 的显式 arm（pending 门不可留在主链等下一节点——中立化会暂存
    恢复）。"""
    if trigger_gate is None:
        return {}
    control_group = group_by_index[trigger_gate.control_instance_index]
    if len(control_group.ranks) != len(trigger_gate.node_gates):
        raise ValueError("KV transfer trigger gate does not match its TP degree")
    if relative_index >= len(control_group.ranks):
        raise ValueError("KV transfer trigger rank index exceeds its TP degree")
    control_rank = control_group.ranks[relative_index]
    node_gate = trigger_gate.node_gates[relative_index]
    if control_rank == source_rank:
        return {
            "trigger_control_rank": control_rank,
            "trigger_dependency_node_id": node_gate,
            "trigger_tag": None,
        }

    trigger_tag = tag_allocator.take()
    _frontier_neutral(
        builders[control_rank],
        lambda: builders[control_rank].comm_send(
            f"{action_name}_shard{shard_index}_trigger_to_rank{source_rank}",
            src=control_rank,
            dst=source_rank,
            comm_size=1,
            comm_tag=trigger_tag),
        node_gate)
    trigger_recv_id = _frontier_neutral(
        builders[source_rank],
        lambda: builders[source_rank].comm_recv(
            f"{action_name}_shard{shard_index}_trigger_from_rank{control_rank}",
            src=control_rank,
            dst=source_rank,
            comm_size=1,
            comm_tag=trigger_tag))
    return {
        "trigger_control_rank": control_rank,
        "trigger_dependency_node_id": node_gate,
        "trigger_tag": trigger_tag,
        "trigger_recv_node_id": trigger_recv_id,
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
    transfer_anchor_sink=None,
    extra_arm_ids: Optional[dict[int, Sequence[int]]] = None,
) -> dict[str, object]:
    """sh_2.0 generate_face_trace.py 的 KV-transfer 发射函数抽取适配
    (KVTransfer/kind 载体来自本仓 B2 内核)。

    extra_arm_ids（死锁真修 方案①,2026-10-11，自 WL 同构移植）:
    store→restore 补偿边的显式传参形态 {rank: [节点 id]};remote_load 链
    在对应 rank 的首节点显式 arm（数据面中立化后主链 pending 不再被
    传输链消费）。

    HBM 计费地图(契约 §12.5/§9,照抄 sh,禁止"优化";端口计时按 README
    §D 2026-09-24 并发化口径:池端口为并发在途+流体均分,端口内无 FIFO
    等待队列):
      - remote_store 链 A(source≠edge):源 comm_send = 源端 COMM_READ 唯一
        数据计费;edge comm_recv 过路 hbm_charge=False;edge mem_store 池写
        本地零计费(仅池端口流体计时);1B ack 双端各 1B(源端收 ack 后才
        物理释放);
      - remote_store 链 B(source==edge 直连):mem_store(hbm_access_mode=1)
        = POOL_READ 唯一计费 + 池端口事务双异步 join(端口腿为并发流体
        计时);
      - remote_load:edge mem_load 仅池端口流体计时;edge comm_send / target
        comm_recv 均 hbm_charge=False(目标写由 restore 承担);target
        local_hbm_kv_restore = RESTORE 唯一数据计费;
      - noc_migrate:本仓 fail-closed 拒绝(_emit_kv_transfer 不发射
        noc_migrate;1000 类迁移经 _paired_transfer 承载)。
    """
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
    # DEAD-cleanup (face-LRU audit, 2026-09-29): the former noc_migrate
    # branch (data send/recv + ack pair, 63 lines) was unreachable --
    # every production NoC migration (history 1000 / joiner 3000) is
    # emitted through _paired_transfer, and _emit_kv_transfer is only
    # ever fed remote_store / remote_load (tests included).  Reaching
    # this point with a noc_migrate transfer is now a fail-closed
    # contract violation instead of silently dead code.
    # (2026-10-01 R2 门位前移:原位置在 shard 循环体内,零 shard 的
    # noc_migrate 对象会绕过该门静默返回空记录——门对全部形态闭合。)
    if transfer.kind == "noc_migrate":
        raise ValueError(
            "_emit_kv_transfer does not emit noc_migrate transfers "
            "(they are _paired_transfer's contract); kind="
            f"{transfer.kind!r}")

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

        if transfer.kind == "remote_store":
            assert shard.source_rank is not None
            assert shard.edge_rank is not None
            source_rank = shard.source_rank
            edge_rank = shard.edge_rank
            if source_group is None:
                raise ValueError("remote store requires a source instance")
            relative_index = source_group.ranks.index(source_rank)
            trigger_info = _emit_transfer_trigger(
                builders=builders,
                group_by_index=group_by_index,
                tag_allocator=tag_allocator,
                trigger_gate=trigger_gate,
                relative_index=relative_index,
                source_rank=source_rank,
                action_name=action_name,
                shard_index=shard_index,
            )
            record.update(trigger_info)
            # 死锁真修（方案①，2026-10-11，自 WL 同构移植）：跨 rank
            # trigger 时数据 send 显式 arm trigger recv；同 rank 时显式
            # arm trigger 门。
            data_gate = trigger_info.get("trigger_recv_node_id")
            if data_gate is None:
                data_gate = trigger_info.get("trigger_dependency_node_id")
            if source_rank == edge_rank:
                # 池存储端点即边缘rank(单 mem_store 路径):边缘rank从本地
                # HBM 读出该分片再经 SerDes 端口写入池——HBM 读计费
                # (hbm-access-mode:1,bytes=tensor_size),MEM 完成=join(端口
                # 事务, HBM 作业)。
                edge_store_node_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].mem_store(
                        f"{action_name}_shard{shard_index}_edge_store",
                        shard.bytes,
                        hbm_access_mode=1),
                    data_gate)
                # 逐出旁路支链尾部观测(2026-09-13):直连路径的池写完成
                # 节点即源端释放节点。中立化后 previous_id 不再前进,
                # id 改取中立段尾节点(同一节点,取法变更)。
                record.update(
                    {
                        "direct_edge_access": True,
                        "source_release_dependency": "mem_store_completion",
                        "edge_store_node_id": edge_store_node_id,
                        "source_ack_node_id": None,
                    }
                )
            else:
                data_tag = tag_allocator.take()
                ack_tag = tag_allocator.take()
                # 源rank的 comm_send 自动计费(发送端 HBM 读);数据经 NoC
                # 到达边缘rank后由 SerDes 直通写池——边缘rank的 comm_recv
                # 与 mem_store 均为直通,不占其本地 HBM(hbm-charge:false /
                # 不标注)。
                _frontier_neutral(
                    builders[source_rank],
                    lambda: builders[source_rank].comm_send(
                        f"{action_name}_shard{shard_index}_send_to_edge{edge_rank}",
                        src=source_rank,
                        dst=edge_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag),
                    data_gate)
                data_recv_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_recv_from_rank{source_rank}",
                        src=source_rank,
                        dst=edge_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag,
                        hbm_charge=False))
                edge_store_node_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].mem_store(
                        f"{action_name}_shard{shard_index}_remote_store",
                        shard.bytes),
                    data_recv_id)
                # 2026-10-11 死锁真修(方案①,自 WL 同构移植):ack 对簿记
                # 节点改中立段(显式 arm mem_store,排序语义=链位边不变),
                # 双方主链 frontier 不再被 ack 推进。
                _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_send(
                        f"{action_name}_shard{shard_index}_ack_to_rank{source_rank}",
                        src=edge_rank,
                        dst=source_rank,
                        comm_size=1,
                        comm_tag=ack_tag),
                    edge_store_node_id)
                source_ack_node_id = _frontier_neutral(
                    builders[source_rank],
                    lambda: builders[source_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_ack_from_edge{edge_rank}",
                        src=edge_rank,
                        dst=source_rank,
                        comm_size=1,
                        comm_tag=ack_tag))
                # 逐出旁路支链尾部观测(2026-09-13):池写完成节点(消费方
                # 唯一:builder 登记 pending_store_tails 的 store→restore
                # 前递依赖);取法改中立段尾节点。
                record.update(
                    {
                        "direct_edge_access": False,
                        "data_tag": data_tag,
                        "ack_tag": ack_tag,
                        "source_release_dependency": "remote_store_ack_recv",
                        "edge_store_node_id": edge_store_node_id,
                        "source_ack_node_id": source_ack_node_id,
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
            # 死锁真修（方案①，2026-10-11，自 WL 同构移植）：remote_load
            # 数据面全链中立化（request 握手/mem_load/data/restore），门
            # 与前驱改显式 arm；restore 节点 id 仍经
            # target_hbm_completion_node_id 落账（C3）。
            request_tag: Optional[int] = None
            pool_read_gate = None
            # store→restore 补偿边的显式形态（方案①）：本 rank 的 arm
            # 表由 remote_load 链首节点（request_recv 或 mem_load）消费。
            restore_arms = tuple((extra_arm_ids or {}).get(edge_rank) or ())
            if control_rank == edge_rank:
                pool_read_gate = timer_gate
            else:
                request_tag = tag_allocator.take()
                _frontier_neutral(
                    builders[control_rank],
                    lambda: builders[control_rank].comm_send(
                        f"{action_name}_shard{shard_index}_request_to_edge{edge_rank}",
                        src=control_rank,
                        dst=edge_rank,
                        comm_size=1,
                        comm_tag=request_tag),
                    timer_gate)
                request_recv_id = _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_request_from_rank{control_rank}",
                        src=control_rank,
                        dst=edge_rank,
                        comm_size=1,
                        comm_tag=request_tag),
                    *restore_arms)
                pool_read_gate = request_recv_id
                restore_arms = ()
            mem_load_id = _frontier_neutral(
                builders[edge_rank],
                lambda: builders[edge_rank].mem_load(
                    f"{action_name}_shard{shard_index}_remote_load",
                    shard.bytes),
                *((pool_read_gate,)
                  if pool_read_gate is not None else ()),
                *restore_arms)
            data_tag: Optional[int] = None
            restore_gate = mem_load_id
            if edge_rank != target_rank:
                data_tag = tag_allocator.take()
                # 边缘rank的 mem_load(SerDes->NoC 直通)与 comm_send
                # (NoC 发射端直通)均不占其本地 HBM;目标rank的 comm_recv
                # 也不计费——其 HBM 写由紧随的 restore 节点(串行语义)承担,
                # 每字节流恰好计费一次。
                _frontier_neutral(
                    builders[edge_rank],
                    lambda: builders[edge_rank].comm_send(
                        f"{action_name}_shard{shard_index}_send_to_rank{target_rank}",
                        src=edge_rank,
                        dst=target_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag,
                        hbm_charge=False),
                    mem_load_id)
                data_recv_id = _frontier_neutral(
                    builders[target_rank],
                    lambda: builders[target_rank].comm_recv(
                        f"{action_name}_shard{shard_index}_recv_from_edge{edge_rank}",
                        src=edge_rank,
                        dst=target_rank,
                        comm_size=shard.bytes,
                        comm_tag=data_tag,
                        hbm_charge=False))
                restore_gate = data_recv_id
            target_hbm_completion_node_id = _frontier_neutral(
                builders[target_rank],
                lambda: builders[target_rank].local_hbm_kv_restore(
                    f"{action_name}_shard{shard_index}_target_hbm_write",
                    shard.bytes),
                restore_gate)
            if target_hbm_completion_node_id is None:
                raise RuntimeError("target HBM restore did not generate a node")
            if transfer_anchor_sink is not None:
                # Metrics code-7 anchor (doc sec.4.3): the target HBM write
                # of this remote-load (restore) shard.  Observation only.
                transfer_anchor_sink(target_rank, target_hbm_completion_node_id)
            # C4 历史到达落账（方案①）：restore 完成节点 = 就绪屏障的显式
            # arm 目标（"恢复完成前不能消费历史 KV" 从链位改为显式边）。
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
                    "history_arrival_node_ids": {
                        target_rank: target_hbm_completion_node_id},
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
