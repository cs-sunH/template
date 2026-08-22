#!/usr/bin/env python3
"""FACE online configuration and shared GraphBatch-emission primitives."""

from __future__ import annotations

import csv
import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SH_TEST_DIR = MODULE_DIR.parents[1]
if str(SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SH_TEST_DIR))

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceInstanceSpec,
    FaceModel,
    KVTransfer,
    KVTransferShard,
)
from generate_trace import (  # noqa: E402
    InferenceGroup,
    RemoteMemoryConfig,
    RequestSpec,
    clean_csv_row,
    load_remote_memory_config,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    parse_rank_spec,
    sanitize_node_prefix,
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
    "output_prefix",
    "output_dir",
    "request_queue_csv",
    "prefill_chunk_size",
    "hardware_config",
    "local_hbm_capacity_profile",
    "system_template",
    "remote_operand_loads",
    "kv_reserve_context_tokens",
)
OPTIONAL_CONFIG_DEFAULTS = {
    "request_queue_session_limit": "0",
    "trace_granularity": "token_expanded",
}
SUPPORTED_CONFIG_KEYS = set(REQUIRED_CONFIG_KEYS) | set(OPTIONAL_CONFIG_DEFAULTS)
INT_CONFIG_KEYS = {
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
    "prefill_chunk_size",
    "kv_reserve_context_tokens",
}
PATH_CONFIG_KEYS = {
    "output_dir",
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
    output_prefix: str
    output_dir: Optional[Path]
    request_queue_csv: Path
    request_queue: tuple[RequestSpec, ...]
    prefill_chunk_size: int
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
    # fail-closed (stage 0, step 0-1): a missing request queue must abort the
    # official entry point with a materialization hint
    # (load_request_queue itself is also fail-closed and never synthesizes
    # a queue).
    if not request_queue_csv.is_file():
        sys.exit(
            f"missing request queue: {request_queue_csv}；"
            "请按 traces/derive_20_first_30_seconds.py 物化输入(运行 stdout 即权威 provenance 记录)"
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
        request_queue=request_queue,
        prefill_chunk_size=int(parsed["prefill_chunk_size"]),
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
        inference_groups=tuple(groups),
        request_queue_session_limit=int(parsed["request_queue_session_limit"]),
        selected_session_ids=selected_session_ids,
        source_request_count=len(source_request_queue),
        source_session_count=source_session_count,
        trace_granularity=str(parsed["trace_granularity"]),
        configuration_digest=configuration_digest,
    )


def _validate_transfer_shard(
    config: FaceTraceConfig,
    transfer: KVTransfer,
    shard: KVTransferShard,
) -> None:
    if shard.bytes <= 0:
        raise ValueError("emitted KV transfer shards must contain positive bytes")
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
    builders: dict[int, Any],
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
    builders: dict[int, Any],
    group_by_index: dict[int, InferenceGroup],
    tag_allocator: TransferTagAllocator,
    transfer: KVTransfer,
    action_name: str,
    pending_gate: Optional[PendingHistoryGate] = None,
    trigger_gate: Optional[TransferTriggerGate] = None,
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
                # HBM endpoint accounting (rule a): the edge rank IS the
                # data endpoint of this pool store -- its local HBM read
                # (bytes leaving the die) is charged once here
                # (hbm-access-mode 1); no NoC segment exists.
                builders[edge_rank].mem_store(
                    f"{action_name}_shard{shard_index}_edge_store",
                    shard.bytes,
                    hbm_access_mode=1,
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
                # S is a real data endpoint: its comm_send is auto-charged
                # (COMM_READ) by the C++ HBM contention model.
                builders[source_rank].comm_send(
                    f"{action_name}_shard{shard_index}_send_to_edge{edge_rank}",
                    src=source_rank,
                    dst=edge_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                )
                # HBM endpoint accounting (rule b): the edge rank only
                # passes NoC->SerDes through here (hbm-charge false -- its
                # HBM is not touched); the following mem_store is likewise
                # unmarked (the pool-side byte stream is charged at S).
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
            # HBM endpoint accounting (rules c/d): when the edge rank IS the
            # target, it is the data endpoint of this pool load -- its local
            # HBM write is charged once here (hbm-access-mode 2). Otherwise
            # the edge rank only passes SerDes->NoC through (mem_load
            # unmarked; the data comm_send below is hbm-charge false), and
            # the target's comm_recv is auto-charged (COMM_WRITE) as the
            # real data endpoint.
            builders[edge_rank].mem_load(
                f"{action_name}_shard{shard_index}_remote_load",
                shard.bytes,
                hbm_access_mode=2 if edge_rank == target_rank else 0,
            )
            data_tag: Optional[int] = None
            if edge_rank != target_rank:
                data_tag = tag_allocator.take()
                # Rule d: NoC->on-die pass-through for the edge rank -- its
                # HBM is not touched for this byte stream.
                builders[edge_rank].comm_send(
                    f"{action_name}_shard{shard_index}_send_to_rank{target_rank}",
                    src=edge_rank,
                    dst=target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                    hbm_charge=False,
                )
                # T is the real data endpoint: auto-charged COMM_WRITE.
                builders[target_rank].comm_recv(
                    f"{action_name}_shard{shard_index}_recv_from_edge{edge_rank}",
                    src=edge_rank,
                    dst=target_rank,
                    comm_size=shard.bytes,
                    comm_tag=data_tag,
                )
            record.update(
                {
                    "control_rank": control_rank,
                    "request_tag": request_tag,
                    "data_tag": data_tag,
                    "direct_edge_access": edge_rank == target_rank,
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
    builders: dict[int, Any],
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
