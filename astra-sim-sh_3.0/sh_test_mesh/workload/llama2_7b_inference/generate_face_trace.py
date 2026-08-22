#!/usr/bin/env python3
"""Shared FACE configuration and GraphBatch emission helpers for online simulation."""

from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence


MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

SH_TEST_DIR = MODULE_DIR.parents[1]
if str(SH_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(SH_TEST_DIR))

from face_scheduler import (  # noqa: E402
    FaceHardware,
    FaceModel,
    KVTransfer,
    KVTransferShard,
)
from generate_trace import (  # noqa: E402
    InferenceGroup,
    RemoteMemoryConfig,
    RequestSpec,
    TraceBuilder,
    load_config_rows,
    load_remote_memory_config,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    sanitize_node_prefix,
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
    "prefill_chunk_size": "512",
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
    prefill_chunk_size: int
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
    request_plan: Any,
) -> None:
    """Apply the zero-context history-gate normalization.

    ``request_plan`` is the online request-plan attribute surface (dict shim
    exposing ``history_location_before`` / ``history_tokens_*`` / ``request_id``).
    """

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
    if key == "prefill_chunk_size":
        return parse_int(value, key)
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
    # The online entry is fail-closed: the shared queue loader does not
    # synthesize a request queue when materialized input is missing.
    if not request_queue_csv.is_file():
        sys.exit(
            f"missing request queue: {request_queue_csv}；"
            "request-neutral 仓库不绑定默认队列——请按 "
            "(traces/materialize_20_30s.py) 物化 "
            "折入 recompute 队列后在 trace_config.csv 指定"
        )
    source_request_queue = load_request_queue(request_queue_csv)
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
        prefill_chunk_size=int(parsed["prefill_chunk_size"]),
        selected_session_ids=selected_session_ids,
        source_request_count=len(source_request_queue),
        source_session_count=source_session_count,
        source_average_decode_length=source_average_decode_length,
        trace_granularity=str(parsed["trace_granularity"]),
        configuration_digest=configuration_digest,
    )






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
            history_tokens = previous_final_context
            prefill_context_tokens = history_tokens + request.prefill_length
            request_work_tokens = request.prefill_length
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
                # N-way HBM contention: the edge rank itself is the pool
                # data endpoint -- its HBM read is charged through the
                # mem_store node (hbm-access-mode 1). The pool-side
                # transaction is the remote-memory port; node completion
                # joins both sides in the engine.
                builders[edge_rank].mem_store(
                    f"{action_name}_shard{shard_index}_edge_store",
                    shard.bytes,
                    hbm_access_mode=1,
                )
                record.update(
                    {
                        "direct_edge_access": True,
                        "source_release_dependency": "mem_store_completion",
                        "hbm_charge": "edge_mem_store_read",
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
                # N-way HBM contention: the edge rank only passes the bytes
                # from the NoC router to its SerDes pool port -- a
                # NoC<->SerDes direct-through that must not touch the edge
                # rank's local HBM. The pool write itself goes through the
                # remote-memory port below without a local-HBM charge (the
                # bytes leave the chip; the pool-side write is not a local
                # HBM access), and the source rank's HBM read is charged by
                # its own comm_send above.
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
                        "hbm_charge": (
                            "source_send_read;edge_pass_through"
                        ),
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
            # N-way HBM contention: the edge rank reads the bytes out of the
            # pool through its SerDes port and hands them to the NoC router
            # -- a SerDes<->NoC direct-through that must not touch the edge
            # rank's local HBM (the mem_load below stays uncharged).
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
                    hbm_charge=False,
                )
                # The target's HBM write is carried by the serial restore
                # node emitted below -- the NoC-stage recv at the target is
                # a router<->HBM-DMA handoff, not the endpoint itself, so it
                # must not charge (avoids double counting with the restore).
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
                    "hbm_charge": (
                        "target_restore_write"
                        + (";edge_pass_through" if data_tag is not None
                           else ";edge_mem_load_uncharged_restore_covers")
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
    估算函数)。③④ 的输入物化入口是
    plan_materializer.py(manifest/metrics/runtime_config);
    静态 ET 生成入口不再存在。
    """
    raise SystemExit(
        "path-1 (offline static full pipeline) was removed on 2026-08-18; "
        "use plan_materializer.py for the online routes' plan-dir inputs")


if __name__ == "__main__":
    main()
