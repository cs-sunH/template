#!/usr/bin/env python3
"""Generate FACE-mapped Chakra ET traces for the configured wafer scenario."""

from __future__ import annotations

import csv
import heapq
import hashlib
import io
import json
import math
import multiprocessing as mp
import os
import queue as queue_module
import shlex
import shutil
import sys
import tempfile
import traceback
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Optional, Sequence


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
    KVAllocation,
    KVTransfer,
    KVTransferShard,
    NodeHBMSnapshot,
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
    kv_event_payload_sh1,
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
    clean_csv_row,
    encode_message,
    load_remote_memory_config,
    load_request_queue,
    parse_bool,
    parse_int,
    parse_nonnegative_int,
    parse_rank_spec,
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


def _encode_message_bytes(message: object) -> bytes:
    destination = io.BytesIO()
    encode_message(destination, message)
    return destination.getvalue()


def _replay_trace_worker(
    worker_index: int,
    command_queue: object,
    result_queue: object,
    rank_outputs: tuple[tuple[int, str, bytes], ...],
    remote_operand_loads: bool,
) -> None:
    """Build and serialize a fixed set of rank-local ET streams in one process."""

    outputs: dict[int, BinaryIO] = {}
    builders: dict[int, TraceBuilder] = {}
    try:
        for rank, temporary_path, metadata in rank_outputs:
            output = Path(temporary_path).open("wb")
            outputs[rank] = output
            output.write(metadata)
            builders[rank] = TraceBuilder(
                remote_operand_loads=remote_operand_loads,
                node_sink=lambda node, output=output: encode_message(output, node),
                retain_nodes=False,
            )

        while True:
            batch = command_queue.get()
            if batch is None:
                break
            for rank, command in batch:
                method, args, kwargs = command
                getattr(builders[rank], method)(*args, **kwargs)

        for output in outputs.values():
            output.close()
        outputs.clear()
        result_queue.put((
            "ok",
            worker_index,
            {rank: builder.node_count for rank, builder in builders.items()},
        ))
    except BaseException:
        for output in outputs.values():
            try:
                output.close()
            except OSError:
                pass
        result_queue.put(("error", worker_index, traceback.format_exc()))
        # Do not let an undrained parent result pipe keep a failed child alive.
        result_queue.cancel_join_thread()
        raise


class ParallelTraceOutputs:
    """Replay rank-local ET commands in bounded CPU worker processes.

    The coordinator keeps the original FACE planning, instance mapping, KV state,
    timer order, and communication-tag allocation serial.  Workers only replay
    immutable commands for disjoint rank files, so output construction can use
    multiple CPU cores without changing any scheduler decision.
    """

    _BATCH_SIZE = 512
    _QUEUE_BATCH_CAPACITY = 2

    def __init__(
        self,
        *,
        config: FaceTraceConfig,
        plan: FacePlan,
        output_dir: Path,
        jobs: int,
    ) -> None:
        if jobs < 2:
            raise ValueError("parallel ET output requires at least two jobs")
        self._final_paths = {
            rank: output_dir / f"{config.output_prefix}.{rank}.et"
            for rank in range(config.npus_count)
        }
        self._finalized = False
        self._aborted = False
        self._workers: list[mp.Process] = []
        self._command_queues: list[object] = []
        self._result_queue: Optional[object] = None
        self._staging_dir: Optional[Path] = None
        try:
            staging_dir = Path(tempfile.mkdtemp(
                prefix=".trace-generation-",
                dir=output_dir,
            ))
            self._staging_dir = staging_dir
            self._temporary_paths = {
                rank: staging_dir / path.name
                for rank, path in self._final_paths.items()
            }
            self._worker_count = min(jobs, config.npus_count)
            self._worker_by_rank = {
                rank: rank % self._worker_count
                for rank in range(config.npus_count)
            }
            self._context = mp.get_context("spawn")
            self._command_queues = [
                self._context.Queue(maxsize=self._QUEUE_BATCH_CAPACITY)
                for _ in range(self._worker_count)
            ]
            self._result_queue = self._context.Queue()
            self._pending_batches = [
                [] for _ in range(self._worker_count)
            ]
            worker_outputs: list[list[tuple[int, str, bytes]]] = [
                [] for _ in range(self._worker_count)
            ]
            for rank in range(config.npus_count):
                worker_outputs[self._worker_by_rank[rank]].append((
                    rank,
                    str(self._temporary_paths[rank]),
                    _encode_message_bytes(_build_metadata(config, plan, rank=rank)),
                ))
            for worker_index, rank_outputs in enumerate(worker_outputs):
                worker = self._context.Process(
                    target=_replay_trace_worker,
                    args=(
                        worker_index,
                        self._command_queues[worker_index],
                        self._result_queue,
                        tuple(rank_outputs),
                        config.remote_operand_loads,
                    ),
                )
                worker.start()
                self._workers.append(worker)
        except BaseException:
            self.abort()
            raise

    def _put(self, worker_index: int, batch: object) -> None:
        command_queue = self._command_queues[worker_index]
        worker = self._workers[worker_index]
        while True:
            if not worker.is_alive():
                raise RuntimeError(
                    f"ET worker {worker_index} exited before completing its work"
                )
            try:
                command_queue.put(batch, timeout=0.5)
                return
            except queue_module.Full:
                continue

    def _flush(self, worker_index: int) -> None:
        batch = self._pending_batches[worker_index]
        if not batch:
            return
        self._put(worker_index, batch)
        self._pending_batches[worker_index] = []

    def write_command(self, rank: int, command: TraceCommand) -> None:
        worker_index = self._worker_by_rank[rank]
        batch = self._pending_batches[worker_index]
        batch.append((rank, command))
        if len(batch) >= self._BATCH_SIZE:
            self._flush(worker_index)

    def _close_queues(self, *, discard_pending: bool = False) -> None:
        for command_queue in getattr(self, "_command_queues", []):
            try:
                if discard_pending:
                    command_queue.cancel_join_thread()
                command_queue.close()
                if not discard_pending:
                    command_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass
        result_queue = getattr(self, "_result_queue", None)
        if result_queue is not None:
            try:
                if discard_pending:
                    result_queue.cancel_join_thread()
                result_queue.close()
                if not discard_pending:
                    result_queue.join_thread()
            except (AttributeError, OSError, ValueError):
                pass

    def finalize(self, expected_nodes_per_rank: dict[str, int]) -> None:
        try:
            for worker_index in range(self._worker_count):
                self._flush(worker_index)
            for worker_index in range(self._worker_count):
                self._put(worker_index, None)

            results: list[tuple[object, ...]] = []
            reported_workers: set[int] = set()
            while len(results) < self._worker_count:
                try:
                    result = self._result_queue.get(timeout=0.5)
                except queue_module.Empty:
                    missing_failed_workers = [
                        index
                        for index, worker in enumerate(self._workers)
                        if worker.exitcode is not None and index not in reported_workers
                    ]
                    if missing_failed_workers:
                        raise RuntimeError(
                            "parallel ET worker exited without a result: "
                            f"{missing_failed_workers}"
                        )
                    continue
                worker_index = int(result[1])
                if worker_index in reported_workers:
                    raise RuntimeError(
                        f"parallel ET worker {worker_index} reported more than once"
                    )
                reported_workers.add(worker_index)
                results.append(result)
            for worker in self._workers:
                worker.join()
            errors = [
                str(result[2])
                for result in results
                if result and result[0] == "error"
            ]
            nonzero_workers = [
                index
                for index, worker in enumerate(self._workers)
                if worker.exitcode != 0
            ]
            if errors or nonzero_workers or len(results) != self._worker_count:
                details = "\n".join(errors)
                if nonzero_workers:
                    details = (
                        f"{details}\nworkers with non-zero exit status: "
                        f"{nonzero_workers}"
                    ).strip()
                if len(results) != self._worker_count:
                    details = (
                        f"{details}\nreceived {len(results)} of "
                        f"{self._worker_count} worker results"
                    ).strip()
                raise RuntimeError(f"parallel ET generation failed: {details}")

            actual_nodes_per_rank: dict[int, int] = {}
            for result in results:
                if result[0] == "ok":
                    actual_nodes_per_rank.update(result[2])
            expected = {
                int(rank): int(count)
                for rank, count in expected_nodes_per_rank.items()
            }
            if actual_nodes_per_rank != expected:
                raise RuntimeError(
                    "parallel ET replay node counts do not match the coordinator"
                )

            for rank, temporary_path in self._temporary_paths.items():
                temporary_path.replace(self._final_paths[rank])
            self._staging_dir.rmdir()
            self._finalized = True
            self._close_queues()
        except BaseException:
            self.abort()
            raise

    def abort(self) -> None:
        if getattr(self, "_finalized", False) or getattr(self, "_aborted", False):
            return
        self._aborted = True
        for worker in getattr(self, "_workers", []):
            if worker.is_alive():
                worker.terminate()
        for worker in getattr(self, "_workers", []):
            worker.join(timeout=5)
            if worker.is_alive():
                worker.kill()
                worker.join()
        self._close_queues(discard_pending=True)
        staging_dir = getattr(self, "_staging_dir", None)
        if staging_dir is not None:
            shutil.rmtree(staging_dir, ignore_errors=True)

    def __del__(self) -> None:
        self.abort()


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
    configuration_digest = _configuration_digest(
        (
            config_csv.resolve(),
            hardware_path,
            system_template,
        )
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


def _to_scheduler_requests(requests: Sequence[RequestSpec]) -> tuple[FaceRequest, ...]:
    return tuple(
        FaceRequest(
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


def build_face_plan(config: FaceTraceConfig) -> FacePlan:
    specs = tuple(
        FaceInstanceSpec(group.name, group.pg_name, group.ranks)
        for group in config.inference_groups
    )
    return plan_face_requests(
        hardware=config.hardware,
        model=config.model,
        instance_specs=specs,
        requests=_to_scheduler_requests(config.request_queue),
        edge_ranks=config.remote_memory.edge_npus,
        reserve_context_tokens=config.kv_reserve_context_tokens,
        record_iterations=config.trace_granularity == "token_expanded",
        prefill_chunk_size=config.prefill_chunk_size,
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
    """Order whole-request ET emission by session and KV-store causality."""

    plans = tuple(plan.requests)
    by_id = {request.request_id: request for request in plans}
    if len(by_id) != len(plans):
        raise ValueError("FACE request IDs must be unique for ET emission")
    successors: dict[str, set[str]] = {
        request.request_id: set() for request in plans
    }
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
        raise RuntimeError(f"KV ET emission dependencies contain a cycle: {blocked[:8]}")
    return tuple(ordered)


def build_trace_label(
    config: FaceTraceConfig,
    plan: Optional[FacePlan] = None,
) -> str:
    session_count = len({request.session_id for request in config.request_queue})
    p_chunk = config.prefill_chunk_size if plan is None else plan.p_chunk
    prefill_label = range_label(
        tuple(request.prefill_length for request in config.request_queue)
    )
    decode_label = range_label(tuple(request.decode_length for request in config.request_queue))
    hbm_gib = config.hardware.local_hbm_capacity_bytes // (1024**3)
    hardware_tag = str(config.hardware_metadata["slug"]).replace("-", "_")
    return (
        f"{config.npus_count}npus_{hardware_tag}_"
        f"hbm{hbm_gib}_kvfifo_r{config.kv_reserve_context_tokens}_"
        f"{len(config.inference_groups)}inst_tp{len(config.inference_groups[0].ranks)}_"
        f"{session_count}sess_{len(config.request_queue)}req_"
        f"pc{p_chunk}_p{prefill_label}_d{decode_label}_"
        f"g{config.trace_granularity}_q{request_queue_digest(config.request_queue)}_"
        f"c{config.configuration_digest}"
    )


def resolve_output_dir(
    config: FaceTraceConfig,
    plan: Optional[FacePlan] = None,
) -> Path:
    if config.output_dir is None:
        return SH_TEST_DIR / "generated" / f"{config.output_prefix}_{build_trace_label(config, plan)}"
    if config.output_dir.is_absolute():
        return config.output_dir
    return (PROJECT_ROOT / config.output_dir).resolve()


def _hbm_snapshot_dict(snapshot: NodeHBMSnapshot) -> dict[str, object]:
    return {
        "rank": snapshot.rank,
        "instance_index": snapshot.instance_index,
        "capacity_bytes": snapshot.capacity_bytes,
        "model_weight_bytes": snapshot.model_weight_bytes,
        "kv_cache_bytes": snapshot.kv_cache_bytes,
        "used_bytes": snapshot.used_bytes,
        "remaining_bytes": snapshot.remaining_bytes,
    }


def _session_snapshot_dict(snapshot: SessionKVSnapshot) -> dict[str, object]:
    return {
        "session_id": snapshot.session_id,
        "location": snapshot.location,
        "instance_index": snapshot.instance_index,
        "context_tokens": snapshot.context_tokens,
        "total_bytes": snapshot.total_bytes,
        "shard_bytes": list(snapshot.shard_bytes),
        "rank_bytes": [list(item) for item in snapshot.rank_bytes],
        "last_completion_ns": snapshot.last_completion_ns,
        "active": snapshot.active,
    }


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


def _record_transfer_anchor_events(
    *,
    metrics: Optional[ServiceMetrics],
    builders: dict[int, TraceBuilder],
    record: dict[str, object],
    queue_index: int,
) -> None:
    """Doc sec.4.3 code 7: the memory-transfer arrival anchor is the last
    transfer node on each arrival rank (target rank for NoC migrations and
    remote loads, edge rank for remote stores).

    The node id is read from the coordinator-side builder/command recorder,
    which assigns the same authoritative ids the rank-local workers replay,
    so the manifest always matches the parallel-built ET (doc sec.9.3).
    Pure observation: no command stream or dependency is touched."""

    if metrics is None:
        return
    kind = record["kind"]
    if kind == "local_hit":
        return
    arrival_ranks: set[int] = set()
    for shard in record["shards"]:  # type: ignore[union-attr]
        if kind == "remote_store":
            rank = shard["edge_rank"]
        else:
            rank = shard["target_rank"]
        if rank is not None:
            arrival_ranks.add(int(rank))
    for rank in sorted(arrival_ranks):
        node_id = builders[rank].previous_id
        if node_id is None:
            raise RuntimeError(
                f"transfer anchor rank {rank} has no emitted node to anchor on"
            )
        metrics.add_event(
            rank,
            node_id,
            EVENT_MEMORY_ANCHOR_COMPLETE,
            queue_index,
        )


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


def _build_metadata(
    config: FaceTraceConfig,
    plan: FacePlan,
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
            ChakraAttr(name="execution_mode", string_val="face_das_omm_static_et"),
            ChakraAttr(name="scheduler", string_val="FACE_DAS_OMM"),
            ChakraAttr(name="hardware_case", string_val="A_FACE_CASE3"),
            ChakraAttr(name="npus_count", uint64_val=config.npus_count),
            ChakraAttr(name="rank", uint64_val=rank),
            ChakraAttr(name="instance_name", string_val=instance.name),
            ChakraAttr(name="instance_index", uint64_val=instance.index),
            ChakraAttr(name="pg_name", string_val=instance.pg_name),
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
            ChakraAttr(name="remote_memory_expansion", bool_val=True),
            ChakraAttr(
                name="remote_memory_type",
                string_val="PER_NPU_MEMORY_EXPANSION",
            ),
            ChakraAttr(
                name="remote_memory_edge_npus",
                string_val=",".join(str(edge) for edge in config.remote_memory.edge_npus),
            ),
            ChakraAttr(
                name="remote_memory_logical_pool",
                string_val=config.remote_memory.logical_pool,
            ),
            ChakraAttr(
                name="remote_memory_bandwidth_gbps_per_edge",
                string_val=str(config.remote_memory.remote_mem_bw_gbps),
            ),
            ChakraAttr(
                name="remote_memory_latency_ns",
                uint64_val=config.remote_memory.remote_mem_latency_ns,
            ),
            ChakraAttr(
                name="kv_reserve_context_tokens",
                uint64_val=config.kv_reserve_context_tokens,
            ),
            ChakraAttr(
                name="kv_reserve_total_bytes_all_tp_ranks",
                uint64_val=kv_cache_bytes_for_tokens(
                    config.model, config.kv_reserve_context_tokens
                ),
            ),
        ]
    )
    return metadata


def _lut_entry_dict(entry: FaceLutEntry) -> dict[str, object]:
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
        "current_lut": _lut_entry_dict(candidate.current_lut),
        "updated_lut": _lut_entry_dict(candidate.updated_lut),
        "delta_time_ns": candidate.delta_time_ns,
        "per_die_delta_ns": candidate.per_die_delta_ns,
    }


def _allocation_dict(allocation: KVAllocation, plan: FacePlan) -> dict[str, object]:
    return {
        "request_id": allocation.request_id,
        "decode_instance_index": allocation.decode_instance_index,
        "total_bytes": allocation.total_bytes,
        "pieces": [
            {
                "instance_index": piece.instance_index,
                "instance_name": plan.topology.instance(piece.instance_index).name,
                "bytes": piece.bytes,
                "weighted_distance": piece.weighted_distance,
                "instance_path": list(piece.path),
                "rank_bytes": [list(item) for item in piece.rank_bytes],
            }
            for piece in allocation.pieces
        ],
    }


def _request_plan_dict(
    request: RequestSpec,
    request_plan: FaceRequestPlan,
    plan: FacePlan,
    *,
    trace_granularity: str,
    transfers_by_stage: dict[str, list[dict[str, object]]],
    readiness_barriers: dict[str, dict[str, object]],
) -> dict[str, object]:
    prefill_instance = plan.topology.instance(request_plan.prefill_instance_index)
    decode_instance = plan.topology.instance(request_plan.decode_instance_index)
    transfer_actions = [
        action
        for stage in (
            "history_evictions",
            "history_transfer",
            "prefill_evictions",
            "decode_evictions",
            "prefill_decode_transfer",
            "completion_evictions",
        )
        for action in transfers_by_stage.get(stage, ())
    ]
    remote_store_actions = [
        action for action in transfer_actions if action["kind"] == "remote_store"
    ]
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
            "hbm_admission": request_plan.admission_time_ns,
            "hbm_wait": request_plan.hbm_wait_ns,
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
            "policy": (
                "per_rank_hbm_feasible_then_remaining_chunks,"
                "last_arrival_time,config_order"
            ),
        },
        "decode_assignment": {
            "instance_index": decode_instance.index,
            "instance_name": decode_instance.name,
            "ranks": list(decode_instance.ranks),
            "distance_limit": plan.topology.hardware.schedulable_distance_limit,
            "candidates": [
                _candidate_dict(candidate) for candidate in request_plan.decode_candidates
            ],
            "policy": (
                "per_rank_hbm_feasible_then_min((T_prime-T)/instance_size),"
                "config_order"
            ),
        },
        "kv_allocation": _allocation_dict(request_plan.kv_allocation, plan),
        "history_source_instance_index": request_plan.history_source_instance_index,
        "history_transfer_bytes": request_plan.history_transfer_bytes,
        "history_location_before": (
            None
            if request_plan.history_location_before is None
            else _session_snapshot_dict(request_plan.history_location_before)
        ),
        "transfers_by_stage": transfers_by_stage,
        "tp_readiness_barriers": readiness_barriers,
        "kv_transfers": transfer_actions,
        "history_routes": transfers_by_stage.get("history_transfer", []),
        "prefill_decode_routes": transfers_by_stage.get(
            "prefill_decode_transfer", []
        ),
        "kv_offload_routes": remote_store_actions,
        "completion_kv_location": {
            "location": request_plan.kv_location_after_completion,
            "instance_index": request_plan.kv_instance_after_completion,
        },
        "hbm_after_completion": [
            _hbm_snapshot_dict(snapshot)
            for snapshot in request_plan.hbm_after_completion
        ],
        "reserve_unmet": bool(request_plan.reserve_unmet_ranks),
        "reserve_unmet_ranks": list(request_plan.reserve_unmet_ranks),
        "stages": [
            "arrival_or_previous_completion_interval_gate",
            "history_evictions",
            "history_kv_local_hit_noc_migrate_or_remote_load",
            "prefill_evictions",
            "prefill_tp_kv_ready_barrier",
            "chunked_prefill",
            "decode_evictions",
            "prefill_to_decode_local_hit_or_noc_migrate",
            "decode_tp_kv_ready_barrier",
            "decode",
            "completion_fifo_remote_stores",
        ],
    }


def write_face_trace(
    config: FaceTraceConfig,
    *,
    jobs: int = 1,
    metrics: Optional[ServiceMetrics] = None,
) -> None:
    if jobs < 1:
        raise ValueError("ET generation jobs must be at least one")
    # Read-only metrics observation (doc sec.7): the embedded KVCacheManager
    # mirrors every state mutation to the recorder *after* applying it.  When
    # metrics is None no recorder is installed and planning is untouched.
    # The planner-LUT accumulator (doc sec.8.8) observes one LUT lookup per
    # planning iteration; it is streaming-only and never fed back.
    lut_stats: Optional[PlannerLutStatsAccumulator] = None
    if metrics is not None:
        set_metrics_observer(metrics.memory)
        lut_stats = PlannerLutStatsAccumulator()
        set_iteration_stats_hook(lut_stats.record_lut_iteration)
    try:
        plan = build_face_plan(config)
    finally:
        set_metrics_observer(None)
        set_iteration_stats_hook(None)
    if plan.edge_ranks != tuple(sorted(config.remote_memory.edge_npus)):
        raise RuntimeError("FACE plan remote-memory edge ranks do not match config")
    if plan.reserve_context_tokens != config.kv_reserve_context_tokens:
        raise RuntimeError("FACE plan KV reserve threshold does not match config")
    output_dir = resolve_output_dir(config, plan)
    output_dir.mkdir(parents=True, exist_ok=True)
    lut_path = output_dir / "face_lut.csv"
    plan.lut.export_csv(lut_path)

    group_by_index = {
        index: group for index, group in enumerate(config.inference_groups)
    }
    parallel_output = jobs > 1 and config.npus_count > 1
    trace_outputs: Optional[ParallelTraceOutputs] = None
    if parallel_output:
        worker_count = min(jobs, config.npus_count)
        print(
            f"[sh_test] Replaying ET nodes with {worker_count} CPU worker "
            f"process{'es' if worker_count != 1 else ''}.",
            file=sys.stderr,
        )
        trace_outputs = ParallelTraceOutputs(
            config=config,
            plan=plan,
            output_dir=output_dir,
            jobs=worker_count,
        )
        builders = {
            rank: TraceCommandRecorder(
                remote_operand_loads=config.remote_operand_loads,
                command_sink=(
                    lambda command, rank=rank: trace_outputs.write_command(
                        rank, command
                    )
                ),
            )
            for rank in range(config.npus_count)
        }
    else:
        builders = {
            rank: TraceBuilder(remote_operand_loads=config.remote_operand_loads)
            for rank in range(config.npus_count)
        }
    tag_allocator = TransferTagAllocator()
    request_by_queue = {
        index: request for index, request in enumerate(config.request_queue)
    }
    plans_by_session: dict[str, list[FaceRequestPlan]] = {}
    for request_plan in plan.requests:
        plans_by_session.setdefault(request_plan.session_id, []).append(request_plan)
    for session_plans in plans_by_session.values():
        session_plans.sort(key=lambda item: item.turn_index)
    next_plan: dict[str, Optional[FaceRequestPlan]] = {}
    for session_plans in plans_by_session.values():
        for current, following in zip(session_plans, session_plans[1:]):
            next_plan[current.request_id] = following
        next_plan[session_plans[-1].request_id] = None

    pending_history: dict[str, PendingHistoryGate] = {}
    pending_request_by_session: dict[str, str] = {}
    deferred_remote_sessions: set[str] = set()

    def mark_pending_history_remote(session_id: str) -> None:
        pending_request_id = pending_request_by_session.get(session_id)
        if pending_request_id is None:
            deferred_remote_sessions.add(session_id)
            return
        pending_history[pending_request_id].location = "remote_memory"

    for request_plan in plan.requests:
        if request_plan.turn_index != 0:
            continue
        group = group_by_index[request_plan.prefill_instance_index]
        request = request_by_queue[request_plan.queue_index]
        arrival = request.session_arrival_time_ns
        if arrival is None:
            raise RuntimeError("first request lost its session arrival time")
        if request_plan.admission_time_ns < arrival:
            raise RuntimeError("request HBM admission precedes source arrival")
        prefix = (
            f"q{request_plan.queue_index:04d}_"
            f"{sanitize_node_prefix(request_plan.request_id)}"
        )
        timers = tuple(
            builders[rank].timer_gate(
                f"{prefix}_global_arrival_timer_gate",
                request_plan.admission_time_ns,
            )
            for rank in group.ranks
        )
        pending_history[request_plan.request_id] = PendingHistoryGate(
            source_instance_index=request_plan.prefill_instance_index,
            timer_gates=timers,
            location="new_session",
        )
        pending_request_by_session[request_plan.session_id] = request_plan.request_id

    request_records: list[dict[str, object]] = []
    ordered_plans = order_plans_for_static_emission(plan)
    emission_index = {
        request_plan.request_id: index
        for index, request_plan in enumerate(ordered_plans)
    }

    for trigger_plan in ordered_plans:
        transfers = (
            *trigger_plan.history_evictions,
            *trigger_plan.prefill_evictions,
            *trigger_plan.decode_evictions,
            *trigger_plan.completion_evictions,
        )
        for transfer in transfers:
            if transfer.kind != "remote_store":
                continue
            trigger_time_ns = _transfer_trigger_time_ns(trigger_plan, transfer)
            producers = [
                candidate
                for candidate in plans_by_session[transfer.session_id]
                if candidate.completion_ns <= trigger_time_ns
            ]
            if not producers:
                raise RuntimeError(
                    f"remote store for session {transfer.session_id} has no "
                    "completed KV producer"
                )
            producer = max(
                producers,
                key=lambda item: (
                    item.completion_ns,
                    item.turn_index,
                    item.queue_index,
                ),
            )
            if emission_index[producer.request_id] > emission_index[trigger_plan.request_id]:
                raise RuntimeError(
                    "static ET request ordering would store session "
                    f"{transfer.session_id} before producer {producer.request_id} "
                    f"is emitted; trigger={trigger_plan.request_id}. "
                    "Use an event/stage-ordered adapter for this workload."
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

        pending_gate = pending_history.pop(request_plan.request_id, None)
        if pending_gate is None:
            raise RuntimeError(f"request {request_plan.request_id} has no arrival/history gate")
        pending_request_by_session.pop(request_plan.session_id, None)
        transfers_by_stage: dict[str, list[dict[str, object]]] = {
            "history_evictions": [],
            "history_transfer": [],
            "prefill_evictions": [],
            "decode_evictions": [],
            "prefill_decode_transfer": [],
            "completion_evictions": [],
        }
        readiness_barriers: dict[str, dict[str, object]] = {}
        action_sequence = 0

        def emit_transfer(
            transfer: KVTransfer,
            stage: str,
            *,
            gate: Optional[PendingHistoryGate] = None,
            trigger_gate: Optional[TransferTriggerGate] = None,
        ) -> dict[str, object]:
            nonlocal action_sequence
            action_name = (
                f"{prefix}_{stage}_action{action_sequence:03d}_"
                f"{sanitize_node_prefix(transfer.session_id)}_{transfer.kind}"
            )
            record = _emit_kv_transfer(
                config=config,
                builders=builders,
                group_by_index=group_by_index,
                tag_allocator=tag_allocator,
                transfer=transfer,
                action_name=action_name,
                pending_gate=gate,
                trigger_gate=trigger_gate,
            )
            record["sequence_stage"] = stage
            record["action_sequence"] = action_sequence
            record["trace_node_prefix"] = action_name
            transfers_by_stage[stage].append(record)
            action_sequence += 1
            if transfer.kind == "remote_store":
                mark_pending_history_remote(transfer.session_id)
            return record

        history_eviction_trigger = TransferTriggerGate(
            control_instance_index=pending_gate.source_instance_index,
            node_gates=pending_gate.timer_gates,
        )
        for transfer in request_plan.history_evictions:
            emit_transfer(
                transfer,
                "history_evictions",
                trigger_gate=history_eviction_trigger,
            )

        if request_plan.history_transfer is None:
            if request_plan.turn_index != 0:
                raise RuntimeError("later request is missing its history transfer action")
            source_group = group_by_index[pending_gate.source_instance_index]
            if source_group.ranks != prefill_group.ranks:
                raise RuntimeError("first-request arrival gate is not on its Prefill ranks")
            for relative_index, rank in enumerate(prefill_group.ranks):
                builders[rank].arm_timer_gate(
                    pending_gate.timer_gates[relative_index]
                )
        else:
            if request_plan.history_location_before is None:
                raise RuntimeError("history transfer is missing its location snapshot")
            if pending_gate.location != request_plan.history_location_before.location:
                raise RuntimeError(
                    f"history gate location {pending_gate.location!r} does not match "
                    f"planned location {request_plan.history_location_before.location!r}"
                )
            emit_record = emit_transfer(
                request_plan.history_transfer,
                "history_transfer",
                gate=pending_gate,
            )
            _record_transfer_anchor_events(
                metrics=metrics,
                builders=builders,
                record=emit_record,
                queue_index=request_plan.queue_index,
            )

        for transfer in request_plan.prefill_evictions:
            emit_transfer(transfer, "prefill_evictions")

        readiness_barriers["prefill"] = _emit_tp_readiness_barrier(
            builders=builders,
            group=prefill_group,
            name=f"{prefix}_prefill_kv_ready_barrier",
        )

        tensor_parallel = len(prefill_group.ranks)
        # Metrics boundary capture (doc sec.6.1/6.3): per-rank ids of the
        # first and last *real* prefill operator/collective nodes; the
        # artificial one-byte end barriers are excluded.  The ids are read
        # from the coordinator-side builders/command recorders, which hold
        # the authoritative parallel-replay ids (doc sec.9.3).  Pure reads;
        # the emitted nodes are unchanged.
        prefill_first_nodes: dict[int, int] = {}
        prefill_last_nodes: dict[int, int] = {}
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

        prefill_completion_nodes = tuple(
            builders[rank].previous_id for rank in prefill_group.ranks
        )
        if any(node_id is None for node_id in prefill_completion_nodes):
            raise RuntimeError("Prefill completion node IDs were not generated")
        decode_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan.prefill_instance_index,
            node_gates=prefill_completion_nodes,
        )
        for transfer in request_plan.decode_evictions:
            emit_transfer(
                transfer,
                "decode_evictions",
                trigger_gate=decode_eviction_trigger,
            )
        if request_plan.prefill_decode_transfer is None:
            raise RuntimeError("request is missing its Prefill-to-Decode KV action")
        prefill_decode_record = emit_transfer(
            request_plan.prefill_decode_transfer,
            "prefill_decode_transfer",
        )
        _record_transfer_anchor_events(
            metrics=metrics,
            builders=builders,
            record=prefill_decode_record,
            queue_index=request_plan.queue_index,
        )

        readiness_barriers["decode"] = _emit_tp_readiness_barrier(
            builders=builders,
            group=decode_group,
            name=f"{prefix}_decode_kv_ready_barrier",
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

        decode_completion_nodes = tuple(
            builders[rank].previous_id for rank in decode_group.ranks
        )
        if any(node_id is None for node_id in decode_completion_nodes):
            raise RuntimeError("Decode completion node IDs were not generated")

        completion_eviction_trigger = TransferTriggerGate(
            control_instance_index=request_plan.decode_instance_index,
            node_gates=decode_completion_nodes,
        )
        for transfer in request_plan.completion_evictions:
            emit_transfer(
                transfer,
                "completion_evictions",
                trigger_gate=completion_eviction_trigger,
            )

        following = next_plan[request_plan.request_id]
        if following is not None:
            following_request = request_by_queue[following.queue_index]
            interval = following_request.inter_request_interval_ns
            if interval is None:
                raise RuntimeError("later request lost its inter-request interval")
            timers = tuple(
                builders[rank].timer_gate(
                        f"q{following.queue_index:04d}_"
                        f"{sanitize_node_prefix(following.request_id)}_"
                        f"history_rank{rank}_interval_gate",
                        interval + following.hbm_wait_ns,
                        after_node_id=decode_completion_nodes[relative_index],
                    )
                for relative_index, rank in enumerate(decode_group.ranks)
            )
            completion_location = request_plan.kv_location_after_completion
            if request_plan.session_id in deferred_remote_sessions:
                completion_location = "remote_memory"
                deferred_remote_sessions.remove(request_plan.session_id)
            if completion_location not in {"local_hbm", "remote_memory"}:
                raise RuntimeError("completed request has no valid KV location")
            pending_history[following.request_id] = PendingHistoryGate(
                source_instance_index=request_plan.decode_instance_index,
                timer_gates=timers,
                location=completion_location,
            )
            pending_request_by_session[request_plan.session_id] = following.request_id
        else:
            deferred_remote_sessions.discard(request_plan.session_id)

        request_records.append(
            _request_plan_dict(
                request,
                request_plan,
                plan,
                trace_granularity=config.trace_granularity,
                transfers_by_stage=transfers_by_stage,
                readiness_barriers=readiness_barriers,
            )
        )

    if pending_history:
        raise RuntimeError(f"unconsumed request gates remain: {sorted(pending_history)}")
    if pending_request_by_session:
        raise RuntimeError(
            "unconsumed per-session history gates remain: "
            f"{sorted(pending_request_by_session)}"
        )

    if parallel_output:
        assert trace_outputs is not None
        trace_outputs.finalize({
            str(rank): builders[rank].node_count
            for rank in range(config.npus_count)
        })
        nodes_per_rank = {
            str(rank): builders[rank].node_count
            for rank in range(config.npus_count)
        }
    else:
        for rank in range(config.npus_count):
            et_path = output_dir / f"{config.output_prefix}.{rank}.et"
            with et_path.open("wb") as output:
                encode_message(output, _build_metadata(config, plan, rank=rank))
                for node in builders[rank].nodes:
                    encode_message(output, node)
        nodes_per_rank = {
            str(rank): len(builders[rank].nodes)
            for rank in range(config.npus_count)
        }
    request_records.sort(key=lambda record: int(record["queue_index"]))
    if metrics is not None:
        # Sidecar manifest (doc sec.4): written only after every .et file is
        # final, so trace_digest covers the exact bytes the simulator reads.
        metrics.write_manifest(
            output_dir=output_dir,
            config=config,
            plan=plan,
            node_count_by_rank={
                int(rank): int(count) for rank, count in nodes_per_rank.items()
            },
            et_paths_by_rank={
                rank: output_dir / f"{config.output_prefix}.{rank}.et"
                for rank in range(config.npus_count)
            },
            request_records=request_records,
            kv_digest_payload=kv_event_payload_sh1(plan),
        )
    manifest = {
        "output_prefix": str(output_dir / config.output_prefix),
        "trace_label": build_trace_label(config, plan),
        "source_config_digest": config.configuration_digest,
        "execution_mode": "face_das_omm_static_et",
        "mapping_strategy": (
            "FACE QP remaining-chunk ordering; weighted Instance_map bounded by "
            "D2D/DRAM; LUT minimum per-die incremental decode latency; node-private "
            "HBM with complete-session FIFO remote stores and location-aware restores"
        ),
        "static_trace_adaptation": (
            "FACE is planned deterministically at trace-generation time because "
            "ASTRA-sim consumes a static Chakra ET DAG; operations within one "
            "instance may serialize even when the planning LUT models PD overlap."
        ),
        "trace_granularity": config.trace_granularity,
        "trace_representation": {
            "mode": config.trace_granularity,
            "request_aggregated_preserves": [
                "request_identity",
                "request_timing",
                "FACE_prefill_and_decode_mapping",
                "KV_transfer_routes_and_bytes",
                "aggregate_FLOPs",
                "aggregate_tensor_bytes",
                "aggregate_All-Reduce_payload_bytes",
            ],
            "request_aggregated_compresses": [
                "identical_layer_invocation_count",
                "prefill_chunk_collective_invocation_and_startup_count",
                "decode_token_collective_invocation_and_startup_count",
            ],
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
            "d2d_to_dram_ratio": config.hardware.schedulable_distance_limit,
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
            "weights": "preloaded_and_sharded_per_npu_in_every_instance",
        },
        "instances": [
            {
                "index": instance.index,
                "name": instance.name,
                "pg_name": instance.pg_name,
                "ranks": list(instance.ranks),
                "shape_rows_columns": list(instance.shape),
                "center_row_column": [instance.center_row, instance.center_col],
                "role": "unified_prefill_decode",
            }
            for instance in plan.topology.instances
        ],
        "prefill_chunk_size": plan.p_chunk,
        "prefill_queue_policy": [
            "per_rank_hbm_feasible_with_final_kv_reservation",
            "remaining_chunk_count",
            "last_enqueued_arrival_time",
            "trace_config_order_tie_break",
        ],
        "decode_policy": {
            "candidate_range": "weighted_distance <= D2D_BW/DRAM_BW",
            "capacity_filter": "per_rank_hbm_feasible_with_reservations",
            "lut_match": "exact instance_size,p_chunk,d_batch; nearest d_token",
            "cost": "(T_prime-T)/instance_size",
            "tie_break": "trace_config_order",
        },
        "lut": {
            "path": str(lut_path),
            "source": "analytical_roofline_from_shared_hardware_and_model",
            "entry_count": len(plan.lut.entries),
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
        "final_hbm_states": [
            _hbm_snapshot_dict(snapshot) for snapshot in plan.final_hbm_states
        ],
        "final_session_states": [
            _session_snapshot_dict(snapshot)
            for snapshot in plan.final_session_states
        ],
        "planning_iterations": [
            {
                "instance_index": iteration.instance_index,
                "iteration_index": iteration.iteration_index,
                "start_ns": iteration.start_ns,
                "end_ns": iteration.end_ns,
                "prefill_request_id": iteration.prefill_request_id,
                "prefill_chunk_tokens": iteration.prefill_chunk_tokens,
                "decode_request_ids": list(iteration.decode_request_ids),
                "lut": _lut_entry_dict(iteration.lut_entry),
            }
            for iteration in plan.iterations
        ],
        "planning_iterations_recorded": plan.iterations_recorded,
        "kv_management": {
            "policy": "node_private_hbm_complete_session_fifo_remote_pool",
            "admission_control": (
                "final_kv_per_rank_capacity_reservation_with_event_driven_wait_retry"
            ),
            "session_history": (
                "cumulative_KV_location_tracked_as_local_hbm_or_remote_memory"
            ),
            "reserve_context_tokens": plan.reserve_context_tokens,
            "reserve_total_bytes_all_tp_ranks": kv_cache_bytes_for_tokens(
                config.model, plan.reserve_context_tokens
            ),
            "reserve_trigger": "remaining_bytes_below_per_tp_rank_reserve",
            "fifo_key": "last_request_completion_ns_then_session_id",
            "transfer_order_per_request": [
                "history_evictions",
                "history_transfer",
                "prefill_evictions",
                "prefill_tp_kv_ready_barrier",
                "prefill",
                "decode_evictions",
                "prefill_decode_transfer",
                "decode_tp_kv_ready_barrier",
                "decode",
                "completion_evictions",
            ],
            "edge_ranks": list(plan.edge_ranks),
            "final_edge_weights": [list(edge) for edge in plan.final_edge_weights],
            "final_remaining_capacity_bytes": list(
                plan.final_remaining_capacity_bytes
            ),
            "final_hbm_states": [
                _hbm_snapshot_dict(snapshot) for snapshot in plan.final_hbm_states
            ],
            "final_session_states": [
                _session_snapshot_dict(snapshot)
                for snapshot in plan.final_session_states
            ],
        },
        "remote_memory": {
            "configuration": str(config.remote_memory.path),
            "source_configuration": str(config.hardware_config),
            "memory_type": "PER_NPU_MEMORY_EXPANSION",
            "edge_npus": list(config.remote_memory.edge_npus),
            "mesh_shape_rows_columns": [
                config.remote_memory.mesh_rows,
                config.remote_memory.mesh_cols,
            ],
            "selection_policy": "nearest_manhattan_then_min_rank",
            "logical_pool": config.remote_memory.logical_pool,
            "logical_pool_shared_across_edges": True,
            "remote_mem_latency_ns": config.remote_memory.remote_mem_latency_ns,
            "remote_mem_bw_gbps_per_edge": (
                config.remote_memory.remote_mem_bw_gbps
            ),
            "non_edge_transport": "native_point_to_point_over_noc",
        },
        "system_config": str(config.system_config),
        "system_template": str(config.system_template),
        "network_config": str(config.network_config),
        "comm_group_config": str(config.comm_group_config),
        "nodes_per_rank": nodes_per_rank,
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
    if metrics is not None and lut_stats is not None:
        # Doc sec.8.8: emitted only after the trace (and its sidecars) were
        # written successfully; generation stdout carries the [METRIC] echo.
        write_planner_lut_stats(lut_stats, output_dir=output_dir)


def print_shell_config(
    config: FaceTraceConfig,
    plan: Optional[FacePlan] = None,
) -> None:
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
        "TP_DEGREE": len(config.inference_groups[0].ranks),
        "PREFILL_CHUNK_SIZE": (
            config.prefill_chunk_size if plan is None else plan.p_chunk
        ),
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
        "REMOTE_MEMORY": str(config.remote_memory.path),
    }
    for key, value in assignments.items():
        print(f"{key}={shlex.quote(str(value))}")


def _parse_jobs(value: str, source: str) -> int:
    try:
        jobs = int(value)
    except ValueError as error:
        raise SystemExit(
            f"{source} must be a positive integer, got {value!r}"
        ) from error
    if jobs < 1:
        raise SystemExit(f"{source} must be a positive integer, got {value!r}")
    return jobs


def _default_jobs() -> int:
    configured = os.environ.get("TRACE_GEN_JOBS")
    if configured is not None:
        return _parse_jobs(configured, "TRACE_GEN_JOBS")
    return os.cpu_count() or 1


def main(argv: Optional[Sequence[str]] = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    print_shell = False
    jobs: Optional[int] = None
    metrics_detail_arg: Optional[str] = None
    config_paths: list[str] = []
    while args:
        argument = args.pop(0)
        if argument == "--print-shell-config":
            print_shell = True
        elif argument in {"-j", "--jobs"}:
            if not args:
                raise SystemExit(f"{argument} requires a positive integer")
            jobs = _parse_jobs(args.pop(0), argument)
        elif argument.startswith("--jobs="):
            jobs = _parse_jobs(argument.split("=", 1)[1], "--jobs")
        elif argument.startswith("--metrics-detail="):
            metrics_detail_arg = argument.split("=", 1)[1]
        elif argument in {"-h", "--help"}:
            raise SystemExit(
                "Usage: generate_trace.py [--print-shell-config] "
                "[-j JOBS|--jobs JOBS] [--metrics-detail=off|summary|full] "
                "[trace_config.csv]. "
                "Set TRACE_GEN_JOBS to choose the default worker count. "
                f"Default config: {CONFIG_CSV_PATH}"
            )
        elif argument.startswith("-"):
            raise SystemExit(f"unknown option: {argument}")
        else:
            config_paths.append(argument)
    if len(config_paths) > 1:
        raise SystemExit(
            "Usage: generate_trace.py [--print-shell-config] "
            "[-j JOBS|--jobs JOBS] [--metrics-detail=off|summary|full] "
            "[trace_config.csv]. "
            f"Default config: {CONFIG_CSV_PATH}"
        )
    config_csv = CONFIG_CSV_PATH if not config_paths else Path(config_paths[0])
    config = load_face_trace_config(config_csv)
    if print_shell:
        print_shell_config(config)
        return
    # Metrics sidecar switch (doc sec.10): CLI > METRICS_DETAIL >
    # ENABLE_METRICS > default on.  When off, no metrics_manifest.json is
    # written and the .et output is byte-identical to the metrics-on output.
    metrics: Optional[ServiceMetrics] = None
    metrics_detail = resolve_metrics_detail(metrics_detail_arg, os.environ)
    if metrics_detail != "off":
        metrics = ServiceMetrics(metrics_detail)
    write_face_trace(
        config,
        jobs=_default_jobs() if jobs is None else jobs,
        metrics=metrics,
    )


if __name__ == "__main__":
    main()
