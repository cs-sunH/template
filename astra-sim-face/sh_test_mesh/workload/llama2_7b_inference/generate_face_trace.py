#!/usr/bin/env python3
"""Shared FACE configuration and GraphBatch-emission primitives for online runs."""

from __future__ import annotations

import csv
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
    coarse_die_face_peak_scalars,
)
from session_kv_manager import NOC_MIGRATE  # noqa: E402  (re-export: graph_batch_builder import)
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
    "kv_reserve_context_tokens": "0",
    # 阶段 C(契约 §2/设计文档 §7.1):模型位置编码声明;无该键默认
    # none(向后兼容——旧 trace_config 行为不变)。rope 时新 Q/K 旋转
    # 归入 qkv_rope 壳的末段(die_op_spec.model_dims.position_encoding)。
    "position_encoding": "none",
}
SUPPORTED_CONFIG_KEYS = (
    set(REQUIRED_CONFIG_KEYS)
    | set(OPTIONAL_CONFIG_DEFAULTS)
)
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
    model_name: str
    mlp_variant: str
    request_queue_csv: Path
    request_queue: tuple[RequestSpec, ...]
    hardware_config: Path
    hardware_capacity_profile: str
    hardware: FaceHardware
    system_template: Path
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
    if key == "position_encoding":
        if value not in {"rope", "none"}:
            raise ValueError(
                "config key position_encoding must be rope or none, got "
                f"{value!r}")
        return value
    if key in PATH_CONFIG_KEYS:
        if not value:
            raise ValueError(f"config key {key} must not be empty")
        return Path(value)
    if not value:
        raise ValueError(f"config key {key} must not be empty")
    return value


def _to_face_hardware(hardware: ResolvedHardware) -> FaceHardware:
    peak_perf = hardware.peak_perf_tflops
    hbm_bandwidth = hardware.local_hbm_bandwidth_gbps
    if hardware.coarse_die_config is not None:
        # The resolver intentionally has no legacy peak scalar for typed hardware.
        # Reuse its actual library summary solely for the FACE selector units.
        peak_perf, hbm_bandwidth = coarse_die_face_peak_scalars(
            hardware.die_summary or {})
    return FaceHardware(
        mesh_rows=hardware.mesh_rows,
        mesh_cols=hardware.mesh_cols,
        local_hbm_capacity_bytes=hardware.local_hbm_capacity_bytes,
        local_hbm_bandwidth_gbps=hbm_bandwidth,
        d2d_bandwidth_gbps=hardware.d2d_bandwidth_gbps,
        peak_perf_tflops=peak_perf,
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
    # (2026-09-30, R2 audit) instance-name uniqueness bookkeeping -- see the
    # duplicate-name fail-closed check in the inference_group branch below.
    seen_group_names: set[str] = set()
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
                if not name or not pg_name:
                    raise ValueError(
                        f"line {line_number}: inference_group requires name and pg_name"
                    )
                # (2026-09-30, R2 audit) group_name must be UNIQUE: the
                # migration emitters decide "same instance, zero nodes" by
                # source_group.name == target_group.name, while every ledger
                # and decision path keys instances by index/rank set -- two
                # same-named groups would move ledger bytes with NO physical
                # migration nodes and no error anywhere.
                if name in seen_group_names:
                    raise ValueError(
                        f"line {line_number}: duplicate inference_group "
                        f"name {name!r} (instance identity for the "
                        "migration emitters is name-based; duplicate "
                        "names would silently suppress cross-instance "
                        "migration nodes)"
                    )
                seen_group_names.add(name)
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
        model_name=str(parsed["model_name"]),
        mlp_variant=str(parsed["mlp_variant"]),
        request_queue_csv=request_queue_csv,
        request_queue=request_queue,
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
    # comm_tag 经 ChakraAttr uint32_val 序列化，C++ 侧按有符号 int 消费：
    # queue_index >= 214,748 时 queue_index*10000 越过 2^31，tag 会以负数
    # 参与网络层配对（错配/死锁）；越过 2^32 则 protobuf 赋值直接抛错。
    # 全量 astra_compute_20 级(千万行)trace 物化可达该量级，故在生成侧
    # fail-closed 拒绝，而不是静默回绕。
    tag = queue_index * 10000 + category + relative_rank
    if tag >= 2 ** 31:
        raise ValueError(
            "comm_tag {} (queue_index={}, category={}, relative_rank={}) "
            "exceeds the 31-bit safe range of the uint32/int32 ET tag "
            "field; the materialized queue is too large for the "
            "tag-per-queue*10000 encoding".format(
                tag, queue_index, category, relative_rank))
    return tag


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
    builders: dict[int, TraceBuilder],
    queue_index: int,
    category: int,
    name: str,
    source_group: InferenceGroup,
    target_group: InferenceGroup,
    total_bytes: int,
    timer_gates: Optional[dict[int, Optional[int]]] = None,
    source_release_ack: bool = False,
    source_release_ack_tag_factory=None,
    source_release_arm_ids: Optional[dict[int, Sequence[int]]] = None,
) -> list[dict[str, object]]:
    if source_release_ack and not callable(source_release_ack_tag_factory):
        raise ValueError("source release ACK requires its disjoint tag allocator")
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
        send_node_id = builders[source].previous_id
        # A previous release on the destination rank must precede this new
        # HBM-resident copy.  The optional arm is absent in legacy mode, so
        # the historical graph is byte-for-byte unchanged.
        for release_node_id in (source_release_arm_ids or {}).get(target, ()):
            builders[target].pending_extra_dependencies.append(
                int(release_node_id))
        builders[target].comm_recv(
            f"{name}_recv_rank{source}_to_rank{target}",
            src=source,
            dst=target,
            comm_size=bytes_by_relative_rank[relative_index],
            comm_tag=tag,
        )
        recv_node_id = builders[target].previous_id
        path = _xy_route(config.hardware, source, target)
        route = {
            "relative_shard": relative_index,
            "source_rank": source,
            "target_rank": target,
            "bytes": bytes_by_relative_rank[relative_index],
            "noc_path": path,
            "noc_hops": len(path) - 1,
        }
        if source_release_ack:
            # The destination can acknowledge only after its matching data
            # receive completed.  The source-side receive is therefore a
            # physical source-release tail; its explicit local send edge also
            # keeps the ACK wait behind the outgoing data operation.
            ack_tag = source_release_ack_tag_factory()
            builders[target].comm_send(
                f"{name}_shard{relative_index}_ack_to_rank{source}",
                src=target,
                dst=source,
                comm_size=1,
                comm_tag=ack_tag,
            )
            builders[source].pending_extra_dependencies.append(int(send_node_id))
            builders[source].comm_recv(
                f"{name}_shard{relative_index}_ack_from_rank{target}",
                src=target,
                dst=source,
                comm_size=1,
                comm_tag=ack_tag,
            )
            route.update({
                "ack_tag": ack_tag,
                "source_release_dependency": "noc_migration_ack_recv",
                "source_release_node_id": builders[source].previous_id,
            })
        routes.append(route)
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


def _transfer_dict(transfer: object | None) -> object:
    if transfer is None:
        return None
    return {
        "action": transfer.action,
        "phase": transfer.phase,
        "reason": transfer.reason,
        "source_instance_index": transfer.source_instance_index,
        "target_instance_index": transfer.target_instance_index,
        "history_tokens": transfer.history_tokens,
        "total_bytes": transfer.total_bytes,
        "shards": [
            {
                "relative_tp_rank": shard.relative_tp_rank,
                "source_rank": shard.source_rank,
                "target_rank": shard.target_rank,
                "bytes": shard.bytes,
            }
            for shard in transfer.shards
        ],
    }




def _emit_control_trigger(
    *,
    builders: dict[int, TraceBuilder],
    queue_index: int,
    name: str,
    source_group: InferenceGroup,
    target_group: InferenceGroup,
    timer_gates: dict[int, Optional[int]],
) -> None:
    """Carry a timing dependency across instances without inventing KV bytes."""

    if source_group.name == target_group.name:
        for rank in target_group.ranks:
            builders[rank].arm_timer_gate(timer_gates.get(rank))
        return
    source = source_group.ranks[0]
    target = target_group.ranks[0]
    builders[source].arm_timer_gate(timer_gates.get(source))
    tag = _stage_tag(queue_index, 1900, 0)
    builders[source].comm_send(
        f"{name}_control_send_rank{source}_to_rank{target}",
        src=source,
        dst=target,
        comm_size=1,
        comm_tag=tag,
    )
    builders[target].comm_recv(
        f"{name}_control_recv_rank{source}_to_rank{target}",
        src=source,
        dst=target,
        comm_size=1,
        comm_tag=tag,
    )


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
