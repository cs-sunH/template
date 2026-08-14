#!/usr/bin/env python3
"""Generate Chakra ET traces for LLaMA-family inference workloads."""

import csv
import hashlib
import json
import random
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for parent in (current.parent, *current.parents):
        if (parent / "extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py").exists():
            return parent
    raise RuntimeError("Cannot find ASTRA-sim project root from generator path.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from extern.graph_frontend.chakra.schema.protobuf.et_def_pb2 import (  # noqa: E402
    ALL_REDUCE,
    COMM_COLL_NODE,
    COMM_RECV_NODE,
    COMM_SEND_NODE,
    COMP_NODE,
    MEM_LOAD_NODE,
    MEM_STORE_NODE,
    AttributeProto as ChakraAttr,
    BoolList,
    GlobalMetadata,
    Node as ChakraNode,
)
from extern.graph_frontend.chakra.src.third_party.utils.protolib import (  # noqa: E402
    encodeMessage as encode_message,
)


CONFIG_CSV_PATH = Path(__file__).with_name("trace_config.csv")
SH_TEST_DIR = CONFIG_CSV_PATH.resolve().parents[2]
DEFAULT_REQUEST_QUEUE_CSV_PATH = SH_TEST_DIR / "workload" / "workload_request_queue.csv"
REMOTE_WEIGHT_ATTR = "remote_weight_bytes"
OPERATOR_GRANULARITY = (
    "rmsnorm,qkv,qk,scale_mask,softmax,av,out_proj,residual,"
    "mlp_gate_up,swiglu,mlp_down,logits"
)
DEFAULT_PREFILL_RANGE = (128, 512)
DEFAULT_DECODE_RANGE = (32, 64)
EXPECTED_SESSION_COUNT = 2
EXPECTED_REQUESTS_PER_SESSION = 2

CONFIG_COLUMNS = ("kind", "key", "value", "group_name", "pg_name", "ranks")
REQUEST_QUEUE_COLUMNS = (
    "session_id",
    "turn_index",
    "request_id",
    "prefill_length",
    "decode_length",
    "session_arrival_time_ns",
    "inter_request_interval_ns",
    "description",
)
REQUIRED_CONFIG_KEYS = (
    "npus_count",
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
    "output_prefix",
    "output_dir",
    "remote_memory_config",
    "remote_operand_loads",
)
OPTIONAL_CONFIG_DEFAULTS = {
    "request_queue_csv": "",
}
SUPPORTED_CONFIG_KEYS = set(REQUIRED_CONFIG_KEYS) | set(OPTIONAL_CONFIG_DEFAULTS)
INT_CONFIG_KEYS = {
    "npus_count",
    "layers",
    "hidden_size",
    "ffn_size",
    "num_heads",
    "vocab_size",
    "bytes_per_elem",
}
BOOL_CONFIG_KEYS = {"remote_operand_loads"}


@dataclass(frozen=True)
class InferenceGroup:
    name: str
    pg_name: str
    ranks: tuple[int, ...]


@dataclass(frozen=True)
class RequestSpec:
    session_id: str
    turn_index: int
    request_id: str
    prefill_length: int
    decode_length: int
    session_arrival_time_ns: Optional[int]
    inter_request_interval_ns: Optional[int]


@dataclass(frozen=True)
class RemoteMemoryConfig:
    path: Path
    edge_npus: tuple[int, ...]
    mesh_rows: int
    mesh_cols: int
    remote_mem_latency_ns: int
    remote_mem_bw_gbps: float
    logical_pool: str


@dataclass(frozen=True)
class TraceConfig:
    npus_count: int
    inference_groups: tuple[InferenceGroup, ...]
    request_queue: tuple[RequestSpec, ...]
    request_queue_csv: Path
    layers: int
    hidden_size: int
    ffn_size: int
    num_heads: int
    vocab_size: int
    bytes_per_elem: int
    output_prefix: str
    output_dir: Optional[Path]
    remote_memory: RemoteMemoryConfig
    remote_operand_loads: bool


@dataclass(frozen=True)
class ScheduledRequest:
    queue_index: int
    request: RequestSpec
    history_tokens_before: int
    prefill_context_tokens: int
    history_tokens_after: int


def parse_int(value: str, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"config key {key} must be an integer, got {value!r}") from exc
    if parsed <= 0:
        raise ValueError(f"config key {key} must be positive, got {parsed}")
    return parsed


def parse_nonnegative_int(value: str, key: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{key} must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"{key} must be non-negative, got {parsed}")
    return parsed


def quantize_timer_duration_ns(duration_ns: int) -> int:
    """Round a planned timer up to Chakra's whole-microsecond resolution.

    Scheduler and roofline calculations operate in nanoseconds, while Chakra's
    ``duration_micros`` field cannot encode a fractional microsecond.  Ceiling
    rounding preserves the planned causal lower bound: a dependent request can
    be delayed by at most 999 ns, but can never be released early.
    """

    if duration_ns < 0:
        raise ValueError("timer duration must be non-negative")
    return ((duration_ns + 999) // 1000) * 1000


def parse_bool(value: str, key: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "no", "n", "off"}:
        return False
    raise ValueError(f"config key {key} must be a boolean, got {value!r}")


def parse_rank_id(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"rank id must be an integer, got {value!r}") from exc
    if parsed < 0:
        raise ValueError(f"rank id must be non-negative, got {parsed}")
    return parsed


def parse_rank_spec(rank_spec: str) -> tuple[int, ...]:
    ranks: list[int] = []
    for part in rank_spec.replace(";", ",").split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_text, end_text = token.split("-", 1)
            start = parse_rank_id(start_text.strip())
            end = parse_rank_id(end_text.strip())
            if end < start:
                raise ValueError(f"rank range must be ascending, got {token!r}")
            ranks.extend(range(start, end + 1))
        else:
            ranks.append(parse_rank_id(token))

    if not ranks:
        raise ValueError("inference group ranks must not be empty")
    if len(set(ranks)) != len(ranks):
        raise ValueError(f"inference group ranks contain duplicates: {rank_spec!r}")
    return tuple(ranks)


def clean_csv_row(row: dict[str, Optional[str]]) -> dict[str, str]:
    return {(key or "").strip(): (value or "").strip() for key, value in row.items()}


def parse_config_value(key: str, value: str) -> object:
    if key in INT_CONFIG_KEYS:
        return parse_int(value, key)
    if key in BOOL_CONFIG_KEYS:
        return parse_bool(value, key)
    if key == "output_dir":
        return Path(value) if value else None
    if key == "request_queue_csv":
        return Path(value) if value else None
    if key == "remote_memory_config":
        if not value:
            raise ValueError("config key remote_memory_config must not be empty")
        return Path(value)
    if not value:
        raise ValueError(f"config key {key} must not be empty")
    return value


def resolve_request_queue_path(queue_csv: Optional[Path]) -> Path:
    if queue_csv is None:
        return DEFAULT_REQUEST_QUEUE_CSV_PATH
    if queue_csv.is_absolute():
        return queue_csv
    return (SH_TEST_DIR / "workload" / queue_csv).resolve()


def resolve_remote_memory_path(remote_memory_config: Path) -> Path:
    if remote_memory_config.is_absolute():
        return remote_memory_config
    return (SH_TEST_DIR / remote_memory_config).resolve()


def load_remote_memory_config(
    path: Path,
    npus_count: int,
    mesh_shape: Optional[tuple[int, int]] = None,
) -> RemoteMemoryConfig:
    if not path.exists():
        raise FileNotFoundError(f"remote-memory config not found: {path}")
    with path.open(encoding="utf-8") as config_file:
        raw = json.load(config_file)

    memory_type = raw.get("memory-type")
    if memory_type != "PER_NPU_MEMORY_EXPANSION":
        raise ValueError(
            "remote-memory config must use PER_NPU_MEMORY_EXPANSION, got "
            f"{memory_type!r}"
        )

    raw_edge_npus = raw.get("npu-ids")
    if not isinstance(raw_edge_npus, list) or not raw_edge_npus:
        raise ValueError("remote-memory config npu-ids must be a non-empty list")
    if any(isinstance(rank, bool) or not isinstance(rank, int) for rank in raw_edge_npus):
        raise ValueError("remote-memory config npu-ids must contain integer ranks")
    edge_npus = tuple(raw_edge_npus)
    if len(set(edge_npus)) != len(edge_npus):
        raise ValueError("remote-memory config npu-ids must not contain duplicates")
    invalid_ranks = [rank for rank in edge_npus if rank < 0 or rank >= npus_count]
    if invalid_ranks:
        raise ValueError(
            f"remote-memory config contains ranks outside 0-{npus_count - 1}: "
            f"{invalid_ranks}"
        )

    configured_mesh_shape = raw.get("mesh-shape")
    if mesh_shape is None:
        if (
            not isinstance(configured_mesh_shape, list)
            or len(configured_mesh_shape) != 2
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in configured_mesh_shape
            )
            or any(value <= 0 for value in configured_mesh_shape)
        ):
            raise ValueError(
                "remote-memory config requires mesh_shape from the authoritative "
                "hardware config"
            )
        mesh_rows, mesh_cols = configured_mesh_shape
    else:
        mesh_rows, mesh_cols = mesh_shape
        if (
            any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (mesh_rows, mesh_cols)
            )
            or mesh_rows <= 0
            or mesh_cols <= 0
        ):
            raise ValueError("mesh_shape must contain positive integer rows and columns")
        if configured_mesh_shape is not None:
            if (
                not isinstance(configured_mesh_shape, list)
                or len(configured_mesh_shape) != 2
                or any(
                    isinstance(value, bool) or not isinstance(value, int)
                    for value in configured_mesh_shape
                )
                or any(value <= 0 for value in configured_mesh_shape)
            ):
                raise ValueError(
                    "remote-memory mesh-shape must contain positive integer rows "
                    "and columns"
                )
            if tuple(configured_mesh_shape) != mesh_shape:
                raise ValueError(
                    "remote-memory mesh-shape does not match the authoritative "
                    "hardware mesh"
                )
    if mesh_rows * mesh_cols != npus_count:
        raise ValueError(
            f"mesh-shape {[mesh_rows, mesh_cols]} does not contain "
            f"npus_count={npus_count} ranks"
        )

    remote_mem_latency_ns = raw.get("remote-mem-latency", 0)
    remote_mem_bw_gbps = raw.get("remote-mem-bw", 0)
    if (
        isinstance(remote_mem_latency_ns, bool)
        or not isinstance(remote_mem_latency_ns, int)
        or remote_mem_latency_ns < 0
    ):
        raise ValueError("remote-mem-latency must be a non-negative integer")
    if (
        isinstance(remote_mem_bw_gbps, bool)
        or not isinstance(remote_mem_bw_gbps, (int, float))
        or remote_mem_bw_gbps <= 0
    ):
        raise ValueError("remote-mem-bw must be positive")

    return RemoteMemoryConfig(
        path=path,
        edge_npus=edge_npus,
        mesh_rows=mesh_rows,
        mesh_cols=mesh_cols,
        remote_mem_latency_ns=remote_mem_latency_ns,
        remote_mem_bw_gbps=float(remote_mem_bw_gbps),
        logical_pool=str(raw.get("logical-pool", "unified-kv-cache-pool")),
    )


def create_default_request_queue(queue_csv: Path) -> None:
    queue_csv.parent.mkdir(parents=True, exist_ok=True)
    rng = random.Random()
    with queue_csv.open("w", newline="", encoding="utf-8") as queue_file:
        writer = csv.DictWriter(
            queue_file,
            fieldnames=REQUEST_QUEUE_COLUMNS,
        )
        writer.writeheader()
        for session_index in range(EXPECTED_SESSION_COUNT):
            for turn_index in range(EXPECTED_REQUESTS_PER_SESSION):
                writer.writerow({
                    "session_id": f"session_{session_index}",
                    "turn_index": turn_index,
                    "request_id": f"session_{session_index}_request_{turn_index}",
                    "prefill_length": rng.randint(*DEFAULT_PREFILL_RANGE),
                    "decode_length": rng.randint(*DEFAULT_DECODE_RANGE),
                    "session_arrival_time_ns": 0 if turn_index == 0 else "",
                    "inter_request_interval_ns": (
                        "" if turn_index == 0 else 1_000_000_000
                    ),
                    "description": "auto-generated queued session request",
                })


def parse_request_length(row: dict[str, str], key: str, aliases: tuple[str, ...]) -> int:
    for column in (key, *aliases):
        value = row.get(column, "")
        if value:
            return parse_int(value, column)
    allowed = ", ".join((key, *aliases))
    raise ValueError(f"request queue row is missing one of: {allowed}")


def load_request_queue(queue_csv: Path) -> tuple[RequestSpec, ...]:
    if not queue_csv.exists():
        create_default_request_queue(queue_csv)

    requests: list[RequestSpec] = []
    with queue_csv.open(newline="", encoding="utf-8-sig") as queue_file:
        reader = csv.DictReader(queue_file)
        if reader.fieldnames is None:
            raise ValueError(f"request queue CSV is empty: {queue_csv}")
        columns = tuple((column or "").strip() for column in reader.fieldnames)
        if columns != REQUEST_QUEUE_COLUMNS:
            raise ValueError(
                "request queue CSV columns must be exactly: "
                + ",".join(REQUEST_QUEUE_COLUMNS)
            )

        for line_number, raw_row in enumerate(reader, start=2):
            row = clean_csv_row(raw_row)
            session_id = row.get("session_id", "")
            request_id = row.get("request_id", "") or f"request_{len(requests)}"
            if request_id.startswith("#"):
                continue
            if not session_id:
                raise ValueError(f"request queue line {line_number} is missing session_id")
            turn_index_text = row.get("turn_index", "")
            if not turn_index_text:
                raise ValueError(f"request queue line {line_number} is missing turn_index")
            turn_index = parse_nonnegative_int(turn_index_text, "turn_index")
            session_arrival_text = row.get("session_arrival_time_ns", "")
            inter_request_interval_text = row.get("inter_request_interval_ns", "")
            if turn_index == 0:
                if not session_arrival_text:
                    raise ValueError(
                        f"request queue line {line_number} first request requires "
                        "session_arrival_time_ns"
                    )
                if inter_request_interval_text:
                    raise ValueError(
                        f"request queue line {line_number} first request must leave "
                        "inter_request_interval_ns empty"
                    )
                session_arrival_time_ns = parse_nonnegative_int(
                    session_arrival_text, "session_arrival_time_ns"
                )
                inter_request_interval_ns = None
            else:
                if session_arrival_text:
                    raise ValueError(
                        f"request queue line {line_number} non-first request must leave "
                        "session_arrival_time_ns empty"
                    )
                if not inter_request_interval_text:
                    raise ValueError(
                        f"request queue line {line_number} non-first request requires "
                        "inter_request_interval_ns"
                    )
                session_arrival_time_ns = None
                inter_request_interval_ns = parse_nonnegative_int(
                    inter_request_interval_text, "inter_request_interval_ns"
                )
            timing_ns = (
                session_arrival_time_ns
                if session_arrival_time_ns is not None
                else inter_request_interval_ns
            )
            if timing_ns is None or timing_ns % 1000 != 0:
                raise ValueError(
                    f"request queue line {line_number} timing must be divisible by 1000 ns"
                )
            prefill_length = parse_request_length(
                row,
                "prefill_length",
                ("prompt_length", "prefill"),
            )
            decode_length = parse_request_length(row, "decode_length", ("decode",))
            requests.append(
                RequestSpec(
                    session_id=session_id,
                    turn_index=turn_index,
                    request_id=request_id,
                    prefill_length=prefill_length,
                    decode_length=decode_length,
                    session_arrival_time_ns=session_arrival_time_ns,
                    inter_request_interval_ns=inter_request_interval_ns,
                )
            )

    if not requests:
        raise ValueError(f"request queue CSV must contain at least one request: {queue_csv}")
    return tuple(requests)


def load_trace_config(config_csv: Path = CONFIG_CSV_PATH) -> TraceConfig:
    if not config_csv.exists():
        raise FileNotFoundError(f"trace config CSV not found: {config_csv}")

    values: dict[str, str] = {}
    groups: list[InferenceGroup] = []
    with config_csv.open(newline="", encoding="utf-8-sig") as config_file:
        reader = csv.DictReader(config_file)
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
                value = row.get("value", "")
                if not key:
                    raise ValueError(f"line {line_number}: config row is missing key")
                if key not in SUPPORTED_CONFIG_KEYS:
                    raise ValueError(f"line {line_number}: unknown config key {key!r}")
                if key in values:
                    raise ValueError(f"line {line_number}: duplicate config key {key!r}")
                values[key] = value
                continue

            if kind == "inference_group":
                name = row.get("group_name", "")
                pg_name = row.get("pg_name", "")
                rank_spec = row.get("ranks", "")
                if not name:
                    raise ValueError(f"line {line_number}: inference_group row is missing group_name")
                if not pg_name:
                    raise ValueError(f"line {line_number}: inference_group row is missing pg_name")
                groups.append(
                    InferenceGroup(
                        name=name,
                        pg_name=pg_name,
                        ranks=parse_rank_spec(rank_spec),
                    )
                )
                continue

            raise ValueError(f"line {line_number}: unsupported config row kind {kind!r}")

    missing_keys = [key for key in REQUIRED_CONFIG_KEYS if key not in values]
    if missing_keys:
        raise ValueError(f"trace config CSV is missing keys: {', '.join(missing_keys)}")
    if not groups:
        raise ValueError("trace config CSV must contain at least one inference_group row")

    parsed_values = {
        key: parse_config_value(key, values.get(key, default_value))
        for key, default_value in {
            **{key: values[key] for key in REQUIRED_CONFIG_KEYS},
            **OPTIONAL_CONFIG_DEFAULTS,
        }.items()
    }

    request_queue_csv = resolve_request_queue_path(
        parsed_values.pop("request_queue_csv")
    )
    request_queue = load_request_queue(request_queue_csv)
    npus_count = int(parsed_values["npus_count"])
    remote_memory_path = resolve_remote_memory_path(
        parsed_values.pop("remote_memory_config")
    )
    remote_memory = load_remote_memory_config(remote_memory_path, npus_count)
    return TraceConfig(
        inference_groups=tuple(groups),
        request_queue=request_queue,
        request_queue_csv=request_queue_csv,
        remote_memory=remote_memory,
        **parsed_values,
    )


def request_queue_digest(request_queue: tuple[RequestSpec, ...]) -> str:
    digest_input = "\n".join(
        f"{request.session_id},{request.turn_index},{request.request_id},"
        f"{request.prefill_length},{request.decode_length},"
        f"{'' if request.session_arrival_time_ns is None else request.session_arrival_time_ns},"
        f"{'' if request.inter_request_interval_ns is None else request.inter_request_interval_ns}"
        for request in request_queue
    )
    return hashlib.sha1(digest_input.encode("utf-8")).hexdigest()[:8]


def range_label(values: tuple[int, ...]) -> str:
    low = min(values)
    high = max(values)
    return str(low) if low == high else f"{low}-{high}"


def build_trace_label(config: TraceConfig) -> str:
    group_sizes = "_".join(str(len(group.ranks)) for group in config.inference_groups)
    session_count = len({request.session_id for request in config.request_queue})
    prefill_label = range_label(
        tuple(request.prefill_length for request in config.request_queue)
    )
    decode_label = range_label(
        tuple(request.decode_length for request in config.request_queue)
    )
    mapping_label = (
        f"{len(config.inference_groups)}inst_fullflow_{session_count}sess_"
        f"{EXPECTED_REQUESTS_PER_SESSION}reqps_{len(config.request_queue)}req_"
        f"tp{group_sizes}_p{prefill_label}_d{decode_label}_"
        f"q{request_queue_digest(config.request_queue)}"
    )
    return f"{config.npus_count}npus_{mapping_label}"


def default_output_dir(config: TraceConfig) -> Path:
    return (
        SH_TEST_DIR
        / "generated"
        / f"{config.output_prefix}_{build_trace_label(config)}"
    )


def shard_size(value: int, shards: int) -> int:
    return (value + shards - 1) // shards


def shard_extent(value: int, shards: int, shard_index: int) -> int:
    """Return an exact, deterministic uneven TP slice size.

    The first ``value % shards`` ranks receive one additional element.  This
    permits LLaMA 2 7B's 4096 hidden channels and 32 attention heads to run on
    the configured FACE instance TP degree without padding the model.
    """

    if value <= 0 or shards <= 0 or not 0 <= shard_index < shards:
        raise ValueError("invalid uneven shard specification")
    return value // shards + (1 if shard_index < value % shards else 0)


class TraceBuilder:
    def __init__(
        self,
        *,
        remote_operand_loads: bool,
        node_sink: Optional[Callable[[ChakraNode], None]] = None,
        retain_nodes: bool = True,
    ) -> None:
        self.remote_operand_loads = remote_operand_loads
        self.node_sink = node_sink
        self.retain_nodes = retain_nodes
        self.next_id = 0
        self.previous_id: Optional[int] = None
        self.pending_extra_dependencies: list[int] = []
        self.nodes: list[ChakraNode] = []
        self.node_count = 0

    @staticmethod
    def _uint64_attr(name: str, value: int) -> ChakraAttr:
        return ChakraAttr(name=name, uint64_val=max(1, int(value)))

    def _new_node(
        self,
        name: str,
        node_type: int,
        *,
        is_cpu_op: bool = False,
    ) -> ChakraNode:
        node = ChakraNode()
        node.id = self.next_id
        self.next_id += 1
        node.name = name
        node.type = node_type
        node.attr.append(ChakraAttr(name="is_cpu_op", bool_val=is_cpu_op))
        dependency_ids: list[int] = []
        if self.previous_id is not None:
            dependency_ids.append(self.previous_id)
        dependency_ids.extend(self.pending_extra_dependencies)
        for dependency_id in dict.fromkeys(dependency_ids):
            node.data_deps.append(dependency_id)
        self.pending_extra_dependencies.clear()
        self.previous_id = node.id
        return node

    def _commit_node(self, node: ChakraNode) -> None:
        if self.node_sink is not None:
            self.node_sink(node)
        if self.retain_nodes:
            self.nodes.append(node)
        self.node_count += 1

    def arm_timer_gate(self, timer_node_id: Optional[int]) -> None:
        if timer_node_id is not None:
            self.pending_extra_dependencies.append(timer_node_id)

    def timer_gate(
        self,
        name: str,
        duration_ns: int,
        *,
        after_node_id: Optional[int] = None,
    ) -> Optional[int]:
        duration_ns = quantize_timer_duration_ns(duration_ns)
        if duration_ns == 0:
            return after_node_id

        node = ChakraNode()
        node.id = self.next_id
        self.next_id += 1
        node.name = name
        node.type = COMP_NODE
        node.attr.extend([
            ChakraAttr(name="is_cpu_op", bool_val=True),
            ChakraAttr(name="is_timer_op", bool_val=True),
        ])
        if after_node_id is not None:
            node.data_deps.append(after_node_id)
        node.duration_micros = duration_ns // 1000
        self._commit_node(node)
        return node.id

    def mem_store(self, name: str, tensor_size: int) -> None:
        node = self._new_node(name, MEM_STORE_NODE)
        node.attr.append(self._uint64_attr("tensor_size", tensor_size))
        self._commit_node(node)

    def mem_load(self, name: str, tensor_size: int) -> None:
        node = self._new_node(name, MEM_LOAD_NODE)
        node.attr.append(self._uint64_attr("tensor_size", tensor_size))
        self._commit_node(node)

    def comp(
        self,
        name: str,
        num_ops: int,
        tensor_size: int,
        remote_read_size: int = 0,
    ) -> None:
        node = self._new_node(name, COMP_NODE)
        node.attr.append(self._uint64_attr("num_ops", num_ops))
        node.attr.append(self._uint64_attr("tensor_size", tensor_size))
        if self.remote_operand_loads and remote_read_size:
            node.attr.append(self._uint64_attr(REMOTE_WEIGHT_ATTR, remote_read_size))
        self._commit_node(node)

    def all_reduce(self, name: str, comm_size: int, pg_name: str) -> None:
        node = self._new_node(name, COMM_COLL_NODE)
        node.attr.append(ChakraAttr(name="comm_type", uint64_val=ALL_REDUCE))
        node.attr.append(self._uint64_attr("comm_size", comm_size))
        node.attr.append(ChakraAttr(name="comm_priority", uint32_val=0))
        node.attr.append(ChakraAttr(name="pg_name", string_val=pg_name))
        node.attr.append(
            ChakraAttr(
                name="involved_dim", bool_list=BoolList(values=[True, True])
            )
        )
        self._commit_node(node)

    def comm_send(
        self,
        name: str,
        *,
        src: int,
        dst: int,
        comm_size: int,
        comm_tag: int,
    ) -> None:
        node = self._new_node(name, COMM_SEND_NODE)
        node.attr.extend([
            ChakraAttr(name="comm_src", uint32_val=src),
            ChakraAttr(name="comm_dst", uint32_val=dst),
            self._uint64_attr("comm_size", comm_size),
            ChakraAttr(name="comm_tag", uint32_val=comm_tag),
        ])
        self._commit_node(node)

    def comm_recv(
        self,
        name: str,
        *,
        src: int,
        dst: int,
        comm_size: int,
        comm_tag: int,
    ) -> None:
        node = self._new_node(name, COMM_RECV_NODE)
        node.attr.extend([
            ChakraAttr(name="comm_src", uint32_val=src),
            ChakraAttr(name="comm_dst", uint32_val=dst),
            self._uint64_attr("comm_size", comm_size),
            ChakraAttr(name="comm_tag", uint32_val=comm_tag),
        ])
        self._commit_node(node)


TraceCommand = tuple[str, tuple[object, ...], dict[str, object]]


class TraceCommandRecorder:
    """Record replayable ``TraceBuilder`` calls without making protobuf nodes.

    The FACE planner, mapping, KV state, timer order, and communication tags stay
    in the coordinator.  Rank-local CPU workers replay this exact command stream
    into their own builders, so parallel construction cannot change scheduling
    decisions or inter-rank dependencies.
    """

    def __init__(
        self,
        *,
        remote_operand_loads: bool,
        command_sink: Callable[[TraceCommand], None],
    ) -> None:
        self.remote_operand_loads = remote_operand_loads
        self.command_sink = command_sink
        self.next_id = 0
        self.previous_id: Optional[int] = None
        self.pending_extra_dependencies: list[int] = []
        self.node_count = 0

    def _record(
        self,
        method: str,
        *args: object,
        **kwargs: object,
    ) -> None:
        self.command_sink((method, args, kwargs))

    def _new_node(self) -> int:
        node_id = self.next_id
        self.next_id += 1
        self.pending_extra_dependencies.clear()
        self.previous_id = node_id
        self.node_count += 1
        return node_id

    def arm_timer_gate(self, timer_node_id: Optional[int]) -> None:
        if timer_node_id is None:
            return
        self.pending_extra_dependencies.append(timer_node_id)
        self._record("arm_timer_gate", timer_node_id)

    def timer_gate(
        self,
        name: str,
        duration_ns: int,
        *,
        after_node_id: Optional[int] = None,
    ) -> Optional[int]:
        duration_ns = quantize_timer_duration_ns(duration_ns)
        if duration_ns == 0:
            return after_node_id
        node_id = self.next_id
        self.next_id += 1
        self.node_count += 1
        self._record(
            "timer_gate",
            name,
            duration_ns,
            after_node_id=after_node_id,
        )
        return node_id

    def mem_store(self, name: str, tensor_size: int) -> None:
        self._new_node()
        self._record("mem_store", name, tensor_size)

    def mem_load(self, name: str, tensor_size: int) -> None:
        self._new_node()
        self._record("mem_load", name, tensor_size)

    def comp(
        self,
        name: str,
        num_ops: int,
        tensor_size: int,
        remote_read_size: int = 0,
    ) -> None:
        self._new_node()
        self._record("comp", name, num_ops, tensor_size, remote_read_size)

    def all_reduce(self, name: str, comm_size: int, pg_name: str) -> None:
        self._new_node()
        self._record("all_reduce", name, comm_size, pg_name)

    def comm_send(
        self,
        name: str,
        *,
        src: int,
        dst: int,
        comm_size: int,
        comm_tag: int,
    ) -> None:
        self._new_node()
        self._record(
            "comm_send",
            name,
            src=src,
            dst=dst,
            comm_size=comm_size,
            comm_tag=comm_tag,
        )

    def comm_recv(
        self,
        name: str,
        *,
        src: int,
        dst: int,
        comm_size: int,
        comm_tag: int,
    ) -> None:
        self._new_node()
        self._record(
            "comm_recv",
            name,
            src=src,
            dst=dst,
            comm_size=comm_size,
            comm_tag=comm_tag,
        )


def tensor_bytes(elements: int, bytes_per_elem: int) -> int:
    return max(1, int(elements) * bytes_per_elem)


def matmul_ops(m: int, n: int, k: int) -> int:
    return 2 * int(m) * int(n) * int(k)


def transformer_pass(
    builder: TraceBuilder,
    *,
    phase: str,
    tokens: int,
    kv_length: int,
    layers: int,
    hidden_size: int,
    ffn_size: int,
    tensor_parallel: int,
    pg_name: str,
    vocab_size: int,
    bytes_per_elem: int,
    num_heads: Optional[int] = None,
    tensor_parallel_rank: int = 0,
    mlp_variant: str = "gelu",
) -> None:
    if mlp_variant not in {"gelu", "swiglu"}:
        raise ValueError("mlp_variant must be gelu or swiglu")
    if num_heads is None:
        attention_hidden_per_rank = shard_extent(
            hidden_size, tensor_parallel, tensor_parallel_rank
        )
        heads_per_rank = 1
    else:
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        heads_per_rank = shard_extent(
            num_heads, tensor_parallel, tensor_parallel_rank
        )
        attention_hidden_per_rank = heads_per_rank * (hidden_size // num_heads)
    ffn_per_rank = shard_extent(ffn_size, tensor_parallel, tensor_parallel_rank)
    vocab_per_rank = shard_extent(vocab_size, tensor_parallel, tensor_parallel_rank)

    activation_elems = tokens * hidden_size
    activation_shard_elems = tokens * attention_hidden_per_rank
    ffn_shard_elems = tokens * ffn_per_rank
    score_elems = tokens * kv_length * heads_per_rank
    kv_cache_elems = kv_length * attention_hidden_per_rank

    activation_bytes = tensor_bytes(activation_elems, bytes_per_elem)
    activation_shard_bytes = tensor_bytes(activation_shard_elems, bytes_per_elem)
    ffn_shard_bytes = tensor_bytes(ffn_shard_elems, bytes_per_elem)
    score_bytes = tensor_bytes(score_elems, bytes_per_elem)
    k_cache_bytes = tensor_bytes(kv_cache_elems, bytes_per_elem)
    v_cache_bytes = tensor_bytes(kv_cache_elems, bytes_per_elem)
    norm_param_bytes = tensor_bytes(
        hidden_size if mlp_variant == "swiglu" else 2 * hidden_size,
        bytes_per_elem,
    )
    norm_ops = 5 * activation_elems if mlp_variant == "swiglu" else 8 * activation_elems
    qkv_weight_bytes = hidden_size * (3 * attention_hidden_per_rank) * bytes_per_elem
    out_weight_bytes = attention_hidden_per_rank * hidden_size * bytes_per_elem
    mlp_projection_factor = 2 if mlp_variant == "swiglu" else 1
    mlp_up_weight_bytes = (
        mlp_projection_factor * hidden_size * ffn_per_rank * bytes_per_elem
    )
    mlp_down_weight_bytes = ffn_per_rank * hidden_size * bytes_per_elem
    mlp_projection_name = (
        "mlp_gate_up_projection" if mlp_variant == "swiglu" else "mlp_up_projection"
    )
    mlp_activation_name = "mlp_swiglu" if mlp_variant == "swiglu" else "mlp_gelu"
    mlp_activation_ops = 6 * ffn_shard_elems if mlp_variant == "swiglu" else 8 * ffn_shard_elems
    mlp_activation_tensor_bytes = (
        3 * ffn_shard_bytes if mlp_variant == "swiglu" else 2 * ffn_shard_bytes
    )
    mlp_activation_remote_bytes = (
        2 * ffn_shard_bytes if mlp_variant == "swiglu" else ffn_shard_bytes
    )

    def comp(
        prefix: str,
        suffix: str,
        num_ops: int,
        tensor_size: int,
        remote_read_size: int,
    ) -> None:
        builder.comp(f"{prefix}_{suffix}", num_ops, tensor_size, remote_read_size)

    for layer in range(layers):
        layer_name = f"{phase}_layer{layer:02d}"

        comp(
            layer_name,
            "attention_input_rmsnorm" if mlp_variant == "swiglu" else "attention_input_layernorm",
            norm_ops,
            3 * activation_bytes + norm_param_bytes,
            activation_bytes + norm_param_bytes,
        )

        qkv_ops = matmul_ops(tokens, 3 * attention_hidden_per_rank, hidden_size)
        qkv_output_bytes = 3 * activation_shard_bytes
        comp(
            layer_name,
            "attention_qkv_projection",
            qkv_ops,
            activation_bytes + qkv_weight_bytes + qkv_output_bytes,
            activation_bytes + qkv_weight_bytes,
        )

        qk_ops = matmul_ops(tokens, kv_length, attention_hidden_per_rank)
        comp(
            layer_name,
            "attention_qk_matmul",
            qk_ops,
            activation_shard_bytes + k_cache_bytes + score_bytes,
            activation_shard_bytes + k_cache_bytes,
        )

        comp(
            layer_name,
            "attention_scale_mask",
            2 * score_elems,
            2 * score_bytes,
            score_bytes,
        )

        comp(
            layer_name,
            "attention_softmax",
            5 * score_elems,
            3 * score_bytes,
            score_bytes,
        )

        av_ops = matmul_ops(tokens, attention_hidden_per_rank, kv_length)
        comp(
            layer_name,
            "attention_av_matmul",
            av_ops,
            score_bytes + v_cache_bytes + activation_shard_bytes,
            score_bytes + v_cache_bytes,
        )

        out_proj_ops = matmul_ops(tokens, hidden_size, attention_hidden_per_rank)
        comp(
            layer_name,
            "attention_output_projection",
            out_proj_ops,
            activation_shard_bytes + out_weight_bytes + activation_bytes,
            activation_shard_bytes + out_weight_bytes,
        )
        builder.all_reduce(
            f"{layer_name}_attention_all_reduce",
            activation_bytes,
            pg_name,
        )

        comp(
            layer_name,
            "attention_residual_add",
            activation_elems,
            3 * activation_bytes,
            2 * activation_bytes,
        )

        comp(
            layer_name,
            "mlp_input_rmsnorm" if mlp_variant == "swiglu" else "mlp_input_layernorm",
            norm_ops,
            3 * activation_bytes + norm_param_bytes,
            activation_bytes + norm_param_bytes,
        )

        mlp_up_ops = matmul_ops(
            tokens, mlp_projection_factor * ffn_per_rank, hidden_size
        )
        comp(
            layer_name,
            mlp_projection_name,
            mlp_up_ops,
            activation_bytes
            + mlp_up_weight_bytes
            + mlp_projection_factor * ffn_shard_bytes,
            activation_bytes + mlp_up_weight_bytes,
        )

        comp(
            layer_name,
            mlp_activation_name,
            mlp_activation_ops,
            mlp_activation_tensor_bytes,
            mlp_activation_remote_bytes,
        )

        mlp_down_ops = matmul_ops(tokens, hidden_size, ffn_per_rank)
        comp(
            layer_name,
            "mlp_down_projection",
            mlp_down_ops,
            ffn_shard_bytes + mlp_down_weight_bytes + activation_bytes,
            ffn_shard_bytes + mlp_down_weight_bytes,
        )
        builder.all_reduce(
            f"{layer_name}_mlp_all_reduce",
            activation_bytes,
            pg_name,
        )

        comp(
            layer_name,
            "mlp_residual_add",
            activation_elems,
            3 * activation_bytes,
            2 * activation_bytes,
        )

    logits_weight_bytes = hidden_size * vocab_per_rank * bytes_per_elem
    logits_output_bytes = tensor_bytes(tokens * vocab_per_rank, bytes_per_elem)
    comp(
        phase,
        "final_rmsnorm" if mlp_variant == "swiglu" else "final_layernorm",
        norm_ops,
        3 * activation_bytes + norm_param_bytes,
        activation_bytes + norm_param_bytes,
    )
    logits_ops = matmul_ops(tokens, vocab_per_rank, hidden_size)
    comp(
        phase,
        "logits_projection",
        logits_ops,
        activation_bytes + logits_weight_bytes + logits_output_bytes,
        activation_bytes + logits_weight_bytes,
    )


def transformer_pass_aggregated(
    builder: TraceBuilder,
    *,
    phase: str,
    pass_spans: Iterable[tuple[int, int]],
    layers: int,
    hidden_size: int,
    ffn_size: int,
    tensor_parallel: int,
    pg_name: str,
    vocab_size: int,
    bytes_per_elem: int,
    num_heads: Optional[int] = None,
    tensor_parallel_rank: int = 0,
    mlp_variant: str = "gelu",
) -> int:
    """Fold repeated Transformer passes into 17 aggregate Chakra nodes.

    Each ``(tokens, kv_length)`` span represents one call to
    :func:`transformer_pass`.  The aggregate nodes preserve the exact sums of
    ``num_ops``, ``tensor_size``, optional remote-read bytes, and All-Reduce
    payload bytes from those expanded calls.  What is intentionally compressed
    is the number of layer, chunk/token, and collective invocations.
    """

    spans = tuple(pass_spans)
    if not spans:
        raise ValueError("pass_spans must contain at least one Transformer pass")
    if layers <= 0:
        raise ValueError("layers must be positive")
    if hidden_size <= 0 or ffn_size <= 0 or tensor_parallel <= 0:
        raise ValueError("model dimensions and tensor_parallel must be positive")
    if vocab_size <= 0 or bytes_per_elem <= 0:
        raise ValueError("vocab_size and bytes_per_elem must be positive")
    for tokens, kv_length in spans:
        if tokens <= 0 or kv_length <= 0:
            raise ValueError("each pass span must contain positive tokens and kv_length")

    if mlp_variant not in {"gelu", "swiglu"}:
        raise ValueError("mlp_variant must be gelu or swiglu")
    if num_heads is None:
        attention_hidden_per_rank = shard_extent(
            hidden_size, tensor_parallel, tensor_parallel_rank
        )
        heads_per_rank = 1
    else:
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")
        heads_per_rank = shard_extent(
            num_heads, tensor_parallel, tensor_parallel_rank
        )
        attention_hidden_per_rank = heads_per_rank * (hidden_size // num_heads)
    ffn_per_rank = shard_extent(ffn_size, tensor_parallel, tensor_parallel_rank)
    vocab_per_rank = shard_extent(vocab_size, tensor_parallel, tensor_parallel_rank)
    norm_param_bytes = tensor_bytes(
        hidden_size if mlp_variant == "swiglu" else 2 * hidden_size,
        bytes_per_elem,
    )
    norm_ops_factor = 5 if mlp_variant == "swiglu" else 8
    qkv_weight_bytes = hidden_size * (3 * attention_hidden_per_rank) * bytes_per_elem
    out_weight_bytes = attention_hidden_per_rank * hidden_size * bytes_per_elem
    mlp_projection_factor = 2 if mlp_variant == "swiglu" else 1
    mlp_up_weight_bytes = (
        mlp_projection_factor * hidden_size * ffn_per_rank * bytes_per_elem
    )
    mlp_down_weight_bytes = ffn_per_rank * hidden_size * bytes_per_elem
    logits_weight_bytes = hidden_size * vocab_per_rank * bytes_per_elem
    attention_norm_name = (
        "attention_input_rmsnorm" if mlp_variant == "swiglu" else "attention_input_layernorm"
    )
    mlp_norm_name = "mlp_input_rmsnorm" if mlp_variant == "swiglu" else "mlp_input_layernorm"
    mlp_projection_name = (
        "mlp_gate_up_projection" if mlp_variant == "swiglu" else "mlp_up_projection"
    )
    mlp_activation_name = "mlp_swiglu" if mlp_variant == "swiglu" else "mlp_gelu"
    final_norm_name = "final_rmsnorm" if mlp_variant == "swiglu" else "final_layernorm"

    category_names = (
        attention_norm_name,
        "attention_qkv_projection",
        "attention_qk_matmul",
        "attention_scale_mask",
        "attention_softmax",
        "attention_av_matmul",
        "attention_output_projection",
        "attention_residual_add",
        mlp_norm_name,
        mlp_projection_name,
        mlp_activation_name,
        "mlp_down_projection",
        "mlp_residual_add",
    )
    layer_totals = {name: [0, 0, 0] for name in category_names}
    attention_collective_bytes = 0
    mlp_collective_bytes = 0
    final_layernorm = [0, 0, 0]
    logits = [0, 0, 0]

    def add(category: list[int], ops: int, tensor_size: int, remote_read: int) -> None:
        category[0] += ops
        category[1] += tensor_size
        category[2] += remote_read

    for tokens, kv_length in spans:
        activation_elems = tokens * hidden_size
        activation_shard_elems = tokens * attention_hidden_per_rank
        ffn_shard_elems = tokens * ffn_per_rank
        score_elems = tokens * kv_length * heads_per_rank
        kv_cache_elems = kv_length * attention_hidden_per_rank

        activation_bytes = tensor_bytes(activation_elems, bytes_per_elem)
        activation_shard_bytes = tensor_bytes(
            activation_shard_elems, bytes_per_elem
        )
        ffn_shard_bytes = tensor_bytes(ffn_shard_elems, bytes_per_elem)
        score_bytes = tensor_bytes(score_elems, bytes_per_elem)
        k_cache_bytes = tensor_bytes(kv_cache_elems, bytes_per_elem)
        v_cache_bytes = tensor_bytes(kv_cache_elems, bytes_per_elem)

        add(
            layer_totals[attention_norm_name],
            norm_ops_factor * activation_elems,
            3 * activation_bytes + norm_param_bytes,
            activation_bytes + norm_param_bytes,
        )
        add(
            layer_totals["attention_qkv_projection"],
            matmul_ops(tokens, 3 * attention_hidden_per_rank, hidden_size),
            activation_bytes + qkv_weight_bytes + 3 * activation_shard_bytes,
            activation_bytes + qkv_weight_bytes,
        )
        add(
            layer_totals["attention_qk_matmul"],
            matmul_ops(tokens, kv_length, attention_hidden_per_rank),
            activation_shard_bytes + k_cache_bytes + score_bytes,
            activation_shard_bytes + k_cache_bytes,
        )
        add(
            layer_totals["attention_scale_mask"],
            2 * score_elems,
            2 * score_bytes,
            score_bytes,
        )
        add(
            layer_totals["attention_softmax"],
            5 * score_elems,
            3 * score_bytes,
            score_bytes,
        )
        add(
            layer_totals["attention_av_matmul"],
            matmul_ops(tokens, attention_hidden_per_rank, kv_length),
            score_bytes + v_cache_bytes + activation_shard_bytes,
            score_bytes + v_cache_bytes,
        )
        add(
            layer_totals["attention_output_projection"],
            matmul_ops(tokens, hidden_size, attention_hidden_per_rank),
            activation_shard_bytes + out_weight_bytes + activation_bytes,
            activation_shard_bytes + out_weight_bytes,
        )
        attention_collective_bytes += activation_bytes
        add(
            layer_totals["attention_residual_add"],
            activation_elems,
            3 * activation_bytes,
            2 * activation_bytes,
        )
        add(
            layer_totals[mlp_norm_name],
            norm_ops_factor * activation_elems,
            3 * activation_bytes + norm_param_bytes,
            activation_bytes + norm_param_bytes,
        )
        add(
            layer_totals[mlp_projection_name],
            matmul_ops(tokens, mlp_projection_factor * ffn_per_rank, hidden_size),
            activation_bytes
            + mlp_up_weight_bytes
            + mlp_projection_factor * ffn_shard_bytes,
            activation_bytes + mlp_up_weight_bytes,
        )
        add(
            layer_totals[mlp_activation_name],
            6 * ffn_shard_elems if mlp_variant == "swiglu" else 8 * ffn_shard_elems,
            3 * ffn_shard_bytes if mlp_variant == "swiglu" else 2 * ffn_shard_bytes,
            2 * ffn_shard_bytes if mlp_variant == "swiglu" else ffn_shard_bytes,
        )
        add(
            layer_totals["mlp_down_projection"],
            matmul_ops(tokens, hidden_size, ffn_per_rank),
            ffn_shard_bytes + mlp_down_weight_bytes + activation_bytes,
            ffn_shard_bytes + mlp_down_weight_bytes,
        )
        mlp_collective_bytes += activation_bytes
        add(
            layer_totals["mlp_residual_add"],
            activation_elems,
            3 * activation_bytes,
            2 * activation_bytes,
        )
        add(
            final_layernorm,
            norm_ops_factor * activation_elems,
            3 * activation_bytes + norm_param_bytes,
            activation_bytes + norm_param_bytes,
        )
        logits_output_bytes = tensor_bytes(
            tokens * vocab_per_rank, bytes_per_elem
        )
        add(
            logits,
            matmul_ops(tokens, vocab_per_rank, hidden_size),
            activation_bytes + logits_weight_bytes + logits_output_bytes,
            activation_bytes + logits_weight_bytes,
        )

    def aggregate_comp(category_name: str) -> None:
        ops, tensor_size, remote_read = layer_totals[category_name]
        builder.comp(
            f"{phase}_all_layers_{category_name}",
            ops * layers,
            tensor_size * layers,
            remote_read * layers,
        )

    aggregate_comp(attention_norm_name)
    aggregate_comp("attention_qkv_projection")
    aggregate_comp("attention_qk_matmul")
    aggregate_comp("attention_scale_mask")
    aggregate_comp("attention_softmax")
    aggregate_comp("attention_av_matmul")
    aggregate_comp("attention_output_projection")
    builder.all_reduce(
        f"{phase}_all_layers_attention_all_reduce",
        attention_collective_bytes * layers,
        pg_name,
    )
    aggregate_comp("attention_residual_add")
    aggregate_comp(mlp_norm_name)
    aggregate_comp(mlp_projection_name)
    aggregate_comp(mlp_activation_name)
    aggregate_comp("mlp_down_projection")
    builder.all_reduce(
        f"{phase}_all_layers_mlp_all_reduce",
        mlp_collective_bytes * layers,
        pg_name,
    )
    aggregate_comp("mlp_residual_add")
    builder.comp(
        f"{phase}_all_passes_{final_norm_name}",
        final_layernorm[0],
        final_layernorm[1],
        final_layernorm[2],
    )
    builder.comp(
        f"{phase}_all_passes_logits_projection",
        logits[0],
        logits[1],
        logits[2],
    )
    return len(spans)


def validate_inference_groups(config: TraceConfig) -> None:
    if not config.inference_groups:
        raise ValueError("inference_groups must not be empty")
    if len(config.inference_groups) != 2:
        raise ValueError("session-to-instance simulation requires exactly two instances")

    all_ranks = []
    pg_names = set()
    for group in config.inference_groups:
        if not group.ranks:
            raise ValueError(f"inference group {group.name} must contain at least one rank")
        invalid_ranks = [
            rank for rank in group.ranks if rank < 0 or rank >= config.npus_count
        ]
        if invalid_ranks:
            raise ValueError(
                f"inference group {group.name} contains rank(s) outside "
                f"0-{config.npus_count - 1}: {invalid_ranks}"
            )
        if group.pg_name in pg_names:
            raise ValueError(f"duplicate pg_name in inference groups: {group.pg_name}")
        if group.pg_name in ("", "0"):
            raise ValueError("pg_name must be a non-zero communicator group id")
        pg_names.add(group.pg_name)
        all_ranks.extend(group.ranks)

    if len(set(all_ranks)) != len(all_ranks):
        raise ValueError("inference groups must not overlap")
    if sorted(all_ranks) != list(range(config.npus_count)):
        raise ValueError("inference groups must cover every configured NPU rank")
    first_group, second_group = config.inference_groups
    if len(first_group.ranks) != len(second_group.ranks):
        raise ValueError("both full-flow instances must use the same TP degree")

    tensor_parallel = len(first_group.ranks)
    if config.hidden_size % config.num_heads != 0:
        raise ValueError("hidden_size must be divisible by num_heads")
    if config.hidden_size % tensor_parallel != 0:
        raise ValueError("hidden_size must be divisible by the TP degree")
    if config.num_heads % tensor_parallel != 0:
        raise ValueError("num_heads must be divisible by the TP degree")


def sanitize_node_prefix(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )


def resolve_output_dir(config: TraceConfig) -> Path:
    if config.output_dir is None:
        return default_output_dir(config)
    if config.output_dir.is_absolute():
        return config.output_dir
    return (PROJECT_ROOT / config.output_dir).resolve()


def validate_session_scenario(config: TraceConfig) -> None:
    validate_inference_groups(config)

    requests_by_session: dict[str, list[tuple[int, RequestSpec]]] = {}
    for queue_index, request in enumerate(config.request_queue):
        requests_by_session.setdefault(request.session_id, []).append(
            (queue_index, request)
        )

    if len(requests_by_session) != EXPECTED_SESSION_COUNT:
        raise ValueError(
            f"scenario requires exactly {EXPECTED_SESSION_COUNT} sessions, got "
            f"{len(requests_by_session)}"
        )
    if len({request.request_id for request in config.request_queue}) != len(
        config.request_queue
    ):
        raise ValueError("request IDs must be unique")

    for session_id, queued_requests in requests_by_session.items():
        if len(queued_requests) != EXPECTED_REQUESTS_PER_SESSION:
            raise ValueError(
                f"session {session_id} requires exactly "
                f"{EXPECTED_REQUESTS_PER_SESSION} requests"
            )
        turn_indexes = [request.turn_index for _, request in queued_requests]
        if turn_indexes != list(range(EXPECTED_REQUESTS_PER_SESSION)):
            raise ValueError(
                f"session {session_id} turn_index values must be "
                f"0..{EXPECTED_REQUESTS_PER_SESSION - 1} in queue order"
            )
        for _, request in queued_requests:
            if request.turn_index == 0:
                if request.session_arrival_time_ns is None:
                    raise ValueError(
                        f"session {session_id} first request requires "
                        "session_arrival_time_ns"
                    )
                if request.inter_request_interval_ns is not None:
                    raise ValueError(
                        f"session {session_id} first request must not define "
                        "inter_request_interval_ns"
                    )
                timing_ns = request.session_arrival_time_ns
            else:
                if request.session_arrival_time_ns is not None:
                    raise ValueError(
                        f"session {session_id} non-first request must not define "
                        "session_arrival_time_ns"
                    )
                if request.inter_request_interval_ns is None:
                    raise ValueError(
                        f"session {session_id} non-first request requires "
                        "inter_request_interval_ns"
                    )
                timing_ns = request.inter_request_interval_ns
            if timing_ns < 0 or timing_ns % 1000 != 0:
                raise ValueError(
                    f"session {session_id} request timing must be a non-negative "
                    "multiple of 1000 ns"
                )


def session_instance_assignments(
    config: TraceConfig,
) -> tuple[tuple[str, InferenceGroup], ...]:
    session_ids = tuple(dict.fromkeys(
        request.session_id for request in config.request_queue
    ))
    return tuple(zip(session_ids, config.inference_groups))


def build_session_schedule(config: TraceConfig) -> tuple[ScheduledRequest, ...]:
    history_tokens: dict[str, int] = {}
    schedule: list[ScheduledRequest] = []
    for queue_index, request in enumerate(config.request_queue):
        history_before = history_tokens.get(request.session_id, 0)
        prefill_context = history_before + request.prefill_length
        history_after = prefill_context + request.decode_length
        schedule.append(
            ScheduledRequest(
                queue_index=queue_index,
                request=request,
                history_tokens_before=history_before,
                prefill_context_tokens=prefill_context,
                history_tokens_after=history_after,
            )
        )
        history_tokens[request.session_id] = history_after
    return tuple(schedule)


def kv_cache_bytes_for_tokens(config: TraceConfig, context_tokens: int) -> int:
    tensor_parallel = len(config.inference_groups[0].ranks)
    hidden_per_rank = config.hidden_size // tensor_parallel
    return (
        2
        * config.layers
        * context_tokens
        * hidden_per_rank
        * config.bytes_per_elem
    )


def rank_coordinates(config: TraceConfig, rank: int) -> tuple[int, int]:
    return divmod(rank, config.remote_memory.mesh_cols)


def manhattan_distance(config: TraceConfig, src: int, dst: int) -> int:
    src_row, src_col = rank_coordinates(config, src)
    dst_row, dst_col = rank_coordinates(config, dst)
    return abs(src_row - dst_row) + abs(src_col - dst_col)


def nearest_edge_npu(config: TraceConfig, rank: int) -> int:
    return min(
        config.remote_memory.edge_npus,
        key=lambda edge_rank: (manhattan_distance(config, rank, edge_rank), edge_rank),
    )


def xy_route(config: TraceConfig, src: int, dst: int) -> list[int]:
    row, col = rank_coordinates(config, src)
    dst_row, dst_col = rank_coordinates(config, dst)
    route = [src]
    while col != dst_col:
        col += 1 if dst_col > col else -1
        route.append(row * config.remote_memory.mesh_cols + col)
    while row != dst_row:
        row += 1 if dst_row > row else -1
        route.append(row * config.remote_memory.mesh_cols + col)
    return route


def stage_tag(queue_index: int, offset: int, rank_or_index: int) -> int:
    return queue_index * 1000 + offset + rank_or_index


def append_remote_restore_stage(
    config: TraceConfig,
    builders: dict[int, TraceBuilder],
    scheduled: ScheduledRequest,
    instance_group: InferenceGroup,
    request_prefix: str,
) -> list[dict[str, object]]:
    tensor_size = kv_cache_bytes_for_tokens(
        config, scheduled.history_tokens_before
    )
    routes: list[dict[str, object]] = []
    for owner_rank in instance_group.ranks:
        edge_rank = nearest_edge_npu(config, owner_rank)
        request_tag = stage_tag(scheduled.queue_index, 400, owner_rank)
        data_tag = stage_tag(scheduled.queue_index, 500, owner_rank)
        if edge_rank == owner_rank:
            builders[owner_rank].mem_load(
                f"{request_prefix}_history_kv_remote_load_local_edge{edge_rank}",
                tensor_size,
            )
        else:
            builders[owner_rank].comm_send(
                f"{request_prefix}_history_kv_restore_request_to_edge{edge_rank}",
                src=owner_rank,
                dst=edge_rank,
                comm_size=1,
                comm_tag=request_tag,
            )
            builders[edge_rank].comm_recv(
                f"{request_prefix}_history_kv_restore_request_from_rank{owner_rank}",
                src=owner_rank,
                dst=edge_rank,
                comm_size=1,
                comm_tag=request_tag,
            )
            builders[edge_rank].mem_load(
                f"{request_prefix}_history_kv_remote_load_for_rank{owner_rank}",
                tensor_size,
            )
            builders[edge_rank].comm_send(
                f"{request_prefix}_history_kv_restore_send_to_rank{owner_rank}",
                src=edge_rank,
                dst=owner_rank,
                comm_size=tensor_size,
                comm_tag=data_tag,
            )
            builders[owner_rank].comm_recv(
                f"{request_prefix}_history_kv_restore_recv_from_edge{edge_rank}",
                src=edge_rank,
                dst=owner_rank,
                comm_size=tensor_size,
                comm_tag=data_tag,
            )
        path = xy_route(config, edge_rank, owner_rank)
        routes.append({
            "owner_rank": owner_rank,
            "edge_rank": edge_rank,
            "bytes": tensor_size,
            "noc_path": path,
            "noc_hops": len(path) - 1,
        })

    for rank in instance_group.ranks:
        builders[rank].all_reduce(
            f"{request_prefix}_history_kv_restore_all_ranks_barrier",
            1,
            instance_group.pg_name,
        )
    return routes


def append_remote_store_stage(
    config: TraceConfig,
    builders: dict[int, TraceBuilder],
    scheduled: ScheduledRequest,
    instance_group: InferenceGroup,
    request_prefix: str,
) -> list[dict[str, object]]:
    tensor_size = kv_cache_bytes_for_tokens(config, scheduled.history_tokens_after)
    routes: list[dict[str, object]] = []
    for owner_rank in instance_group.ranks:
        edge_rank = nearest_edge_npu(config, owner_rank)
        data_tag = stage_tag(scheduled.queue_index, 100, owner_rank)
        ack_tag = stage_tag(scheduled.queue_index, 200, owner_rank)
        if edge_rank == owner_rank:
            builders[owner_rank].mem_store(
                f"{request_prefix}_history_kv_remote_store_local_edge{edge_rank}",
                tensor_size,
            )
        else:
            builders[owner_rank].comm_send(
                f"{request_prefix}_history_kv_store_send_to_edge{edge_rank}",
                src=owner_rank,
                dst=edge_rank,
                comm_size=tensor_size,
                comm_tag=data_tag,
            )
            builders[edge_rank].comm_recv(
                f"{request_prefix}_history_kv_store_recv_from_rank{owner_rank}",
                src=owner_rank,
                dst=edge_rank,
                comm_size=tensor_size,
                comm_tag=data_tag,
            )
            builders[edge_rank].mem_store(
                f"{request_prefix}_history_kv_remote_store_for_rank{owner_rank}",
                tensor_size,
            )
            builders[edge_rank].comm_send(
                f"{request_prefix}_history_kv_store_ack_to_rank{owner_rank}",
                src=edge_rank,
                dst=owner_rank,
                comm_size=1,
                comm_tag=ack_tag,
            )
            builders[owner_rank].comm_recv(
                f"{request_prefix}_history_kv_store_ack_from_edge{edge_rank}",
                src=edge_rank,
                dst=owner_rank,
                comm_size=1,
                comm_tag=ack_tag,
            )
        path = xy_route(config, owner_rank, edge_rank)
        routes.append({
            "owner_rank": owner_rank,
            "edge_rank": edge_rank,
            "bytes": tensor_size,
            "noc_path": path,
            "noc_hops": len(path) - 1,
        })

    for rank in instance_group.ranks:
        builders[rank].all_reduce(
            f"{request_prefix}_history_kv_remote_store_all_ranks_barrier",
            1,
            instance_group.pg_name,
        )
    return routes


def build_session_metadata(
    config: TraceConfig,
    assigned_schedule: tuple[ScheduledRequest, ...],
    *,
    rank: int,
    group: InferenceGroup,
    session_id: str,
) -> GlobalMetadata:
    metadata = GlobalMetadata(version="0.0.4")
    metadata.attr.extend([
        ChakraAttr(name="schema", string_val="1.0.2-chakra.0.0.4"),
        ChakraAttr(name="model", string_val="llama2_7b"),
        ChakraAttr(
            name="execution_mode",
            string_val="instance_local_prefill_decode_session_queue_remote_kv",
        ),
        ChakraAttr(name="npus_count", uint64_val=config.npus_count),
        ChakraAttr(name="rank", uint64_val=rank),
        ChakraAttr(name="request_queue_csv", string_val=str(config.request_queue_csv)),
        ChakraAttr(name="request_count", uint64_val=len(config.request_queue)),
        ChakraAttr(
            name="assigned_request_count",
            uint64_val=len(assigned_schedule),
        ),
        ChakraAttr(name="session_count", uint64_val=EXPECTED_SESSION_COUNT),
        ChakraAttr(
            name="request_timing_source",
            string_val="request_queue_csv",
        ),
        ChakraAttr(
            name="assigned_session_arrival_times_ns",
            string_val=",".join(
                ""
                if item.request.session_arrival_time_ns is None
                else str(item.request.session_arrival_time_ns)
                for item in assigned_schedule
            ),
        ),
        ChakraAttr(
            name="assigned_inter_request_intervals_ns",
            string_val=",".join(
                ""
                if item.request.inter_request_interval_ns is None
                else str(item.request.inter_request_interval_ns)
                for item in assigned_schedule
            ),
        ),
        ChakraAttr(
            name="assigned_requests",
            string_val=",".join(
                item.request.request_id for item in assigned_schedule
            ),
        ),
        ChakraAttr(name="assigned_session", string_val=session_id),
        ChakraAttr(
            name="history_tokens_before",
            string_val=",".join(
                str(item.history_tokens_before) for item in assigned_schedule
            ),
        ),
        ChakraAttr(name="inference_group", string_val=group.name),
        ChakraAttr(name="instance_role", string_val="prefill_decode"),
        ChakraAttr(name="pg_name", string_val=group.pg_name),
        ChakraAttr(name="tensor_parallel", uint64_val=len(group.ranks)),
        ChakraAttr(
            name="rank_group",
            string_val=",".join(str(group_rank) for group_rank in group.ranks),
        ),
        ChakraAttr(
            name="remote_memory_edge_npus",
            string_val=",".join(str(edge) for edge in config.remote_memory.edge_npus),
        ),
        ChakraAttr(name="nearest_edge_policy", string_val="manhattan_then_min_rank"),
    ])
    return metadata


def build_session_mapping_strategy(config: TraceConfig) -> str:
    edges = ",".join(str(rank) for rank in config.remote_memory.edge_npus)
    assignments = "; ".join(
        f"{session_id}->{group.name} ranks "
        f"{','.join(str(rank) for rank in group.ranks)}"
        for session_id, group in session_instance_assignments(config)
    )
    return (
        f"two independent sessions with two requests each in the global CSV "
        f"execution queue; "
        f"fixed session-to-instance mapping [{assignments}]; each instance performs "
        f"both prefill and decode locally with TP degree "
        f"{len(config.inference_groups[0].ranks)}, without inter-instance KV transfer; "
        f"request timing comes from the request queue CSV: the first request uses a "
        f"non-resource global-arrival timer gate, and each later request uses a "
        f"non-resource interval gate armed after the previous request's complete "
        f"remote store; final KV shards use the nearest Manhattan edge NPU from "
        f"[{edges}] for remote store and restore"
    )


def write_session_trace(config: TraceConfig) -> None:
    validate_session_scenario(config)
    schedule = build_session_schedule(config)
    assignments = session_instance_assignments(config)
    session_to_group = dict(assignments)
    output_dir = resolve_output_dir(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    tensor_parallel = len(config.inference_groups[0].ranks)
    builders = {
        rank: TraceBuilder(remote_operand_loads=config.remote_operand_loads)
        for rank in range(config.npus_count)
    }
    request_records: list[dict[str, object]] = []
    schedules_by_session: dict[str, list[ScheduledRequest]] = {}
    for scheduled in schedule:
        schedules_by_session.setdefault(scheduled.request.session_id, []).append(
            scheduled
        )
    next_request_by_queue_index: dict[int, ScheduledRequest] = {}
    for session_schedule in schedules_by_session.values():
        for current, following in zip(session_schedule, session_schedule[1:]):
            next_request_by_queue_index[current.queue_index] = following

    request_timer_gates: dict[int, dict[int, Optional[int]]] = {}
    for session_id, instance_group in assignments:
        first_scheduled = schedules_by_session[session_id][0]
        first_request = first_scheduled.request
        arrival_time_ns = first_request.session_arrival_time_ns
        if arrival_time_ns is None:
            raise ValueError(
                f"session {session_id} first request is missing session arrival time"
            )
        first_request_prefix = (
            f"q{first_scheduled.queue_index:02d}_{sanitize_node_prefix(session_id)}_"
            f"turn{first_request.turn_index}_"
            f"{sanitize_node_prefix(first_request.request_id)}"
        )
        request_timer_gates[first_scheduled.queue_index] = {
            rank: builders[rank].timer_gate(
                f"{first_request_prefix}_global_arrival_timer_gate",
                arrival_time_ns,
            )
            for rank in instance_group.ranks
        }

    for scheduled in schedule:
        request = scheduled.request
        instance_group = session_to_group[request.session_id]
        request_prefix = (
            f"q{scheduled.queue_index:02d}_{sanitize_node_prefix(request.session_id)}_"
            f"turn{request.turn_index}_{sanitize_node_prefix(request.request_id)}"
        )
        timer_gates = request_timer_gates.pop(scheduled.queue_index, None)
        if timer_gates is None or set(timer_gates) != set(instance_group.ranks):
            raise RuntimeError(
                f"request {request.request_id} is missing timer gates for its instance"
            )
        for rank in instance_group.ranks:
            builders[rank].arm_timer_gate(timer_gates[rank])

        restore_routes: list[dict[str, object]] = []
        if request.turn_index > 0:
            restore_routes = append_remote_restore_stage(
                config,
                builders,
                scheduled,
                instance_group,
                request_prefix,
            )

        for rank in instance_group.ranks:
            transformer_pass(
                builders[rank],
                phase=f"{request_prefix}_prefill",
                tokens=request.prefill_length,
                kv_length=scheduled.prefill_context_tokens,
                layers=config.layers,
                hidden_size=config.hidden_size,
                ffn_size=config.ffn_size,
                tensor_parallel=tensor_parallel,
                pg_name=instance_group.pg_name,
                vocab_size=config.vocab_size,
                bytes_per_elem=config.bytes_per_elem,
            )
            builders[rank].all_reduce(
                f"{request_prefix}_prefill_request_end_all_ranks_barrier",
                1,
                instance_group.pg_name,
            )

        for rank in instance_group.ranks:
            for step in range(request.decode_length):
                transformer_pass(
                    builders[rank],
                    phase=f"{request_prefix}_decode{step:04d}",
                    tokens=1,
                    kv_length=scheduled.prefill_context_tokens + step + 1,
                    layers=config.layers,
                    hidden_size=config.hidden_size,
                    ffn_size=config.ffn_size,
                    tensor_parallel=tensor_parallel,
                    pg_name=instance_group.pg_name,
                    vocab_size=config.vocab_size,
                    bytes_per_elem=config.bytes_per_elem,
                )
            builders[rank].all_reduce(
                f"{request_prefix}_decode_request_end_all_ranks_barrier",
                1,
                instance_group.pg_name,
            )

        store_routes = append_remote_store_stage(
            config,
            builders,
            scheduled,
            instance_group,
            request_prefix,
        )

        next_scheduled = next_request_by_queue_index.get(scheduled.queue_index)
        if next_scheduled is not None:
            next_request = next_scheduled.request
            interval_ns = next_request.inter_request_interval_ns
            if interval_ns is None:
                raise ValueError(
                    f"request {next_request.request_id} is missing its inter-request interval"
                )
            next_request_prefix = (
                f"q{next_scheduled.queue_index:02d}_"
                f"{sanitize_node_prefix(next_request.session_id)}_"
                f"turn{next_request.turn_index}_"
                f"{sanitize_node_prefix(next_request.request_id)}"
            )
            request_timer_gates[next_scheduled.queue_index] = {
                rank: builders[rank].timer_gate(
                    f"{next_request_prefix}_previous_completion_interval_timer_gate",
                    interval_ns,
                    after_node_id=builders[rank].previous_id,
                )
                for rank in instance_group.ranks
            }

        wait_before_request_ns = (
            request.session_arrival_time_ns
            if request.turn_index == 0
            else request.inter_request_interval_ns
        )
        if wait_before_request_ns is None:
            raise RuntimeError(
                f"request {request.request_id} has no applicable request timing"
            )
        if request.turn_index == 0:
            timing_stage = (
                "global_arrival_timer_gate"
                if wait_before_request_ns > 0
                else "global_arrival_zero_delay"
            )
        else:
            timing_stage = (
                "previous_completion_interval_timer_gate"
                if wait_before_request_ns > 0
                else "previous_completion_zero_interval"
            )
        request_stages = [timing_stage]
        if request.turn_index > 0:
            request_stages.append("remote_restore")
        request_stages.extend(["prefill", "decode", "remote_store"])

        request_records.append({
            "queue_index": scheduled.queue_index,
            "session_id": request.session_id,
            "turn_index": request.turn_index,
            "request_id": request.request_id,
            "instance_name": instance_group.name,
            "instance_pg_name": instance_group.pg_name,
            "instance_ranks": list(instance_group.ranks),
            "prefill_length": request.prefill_length,
            "decode_length": request.decode_length,
            "session_arrival_time_ns": request.session_arrival_time_ns,
            "inter_request_interval_ns": request.inter_request_interval_ns,
            "history_tokens_before": scheduled.history_tokens_before,
            "prefill_context_tokens": scheduled.prefill_context_tokens,
            "history_tokens_after": scheduled.history_tokens_after,
            "wait_before_request_ns": wait_before_request_ns,
            "remote_restore_bytes_per_rank": (
                kv_cache_bytes_for_tokens(config, scheduled.history_tokens_before)
                if request.turn_index > 0
                else 0
            ),
            "inter_instance_kv_transfer_bytes_per_rank": 0,
            "remote_store_bytes_per_rank": kv_cache_bytes_for_tokens(
                config, scheduled.history_tokens_after
            ),
            "restore_routes": restore_routes,
            "store_routes": store_routes,
            "stages": request_stages,
        })

    if request_timer_gates:
        raise RuntimeError("unconsumed request timer gates remain after trace generation")

    for session_id, group in assignments:
        assigned_schedule = tuple(
            item for item in schedule if item.request.session_id == session_id
        )
        for rank in group.ranks:
            et_path = output_dir / f"{config.output_prefix}.{rank}.et"
            metadata = build_session_metadata(
                config,
                assigned_schedule,
                rank=rank,
                group=group,
                session_id=session_id,
            )
            with et_path.open("wb") as et:
                encode_message(et, metadata)
                for node in builders[rank].nodes:
                    encode_message(et, node)

    nodes_per_rank = {
        str(rank): len(builders[rank].nodes) for rank in range(config.npus_count)
    }
    manifest = {
        "output_prefix": str(output_dir / config.output_prefix),
        "npus_count": config.npus_count,
        "trace_label": build_trace_label(config),
        "execution_mode": "instance_local_prefill_decode_session_queue_remote_kv",
        "mapping_strategy": build_session_mapping_strategy(config),
        "request_queue_csv": str(config.request_queue_csv),
        "request_timing_source": "request_queue_csv",
        "request_queue": request_records,
        "session_count": EXPECTED_SESSION_COUNT,
        "requests_per_session": EXPECTED_REQUESTS_PER_SESSION,
        "session_instance_mapping": [
            {
                "session_id": session_id,
                "instance_name": group.name,
                "role": "prefill_decode",
                "pg_name": group.pg_name,
                "ranks": list(group.ranks),
                "tensor_parallel": tensor_parallel,
            }
            for session_id, group in assignments
        ],
        "inter_instance_kv_transfer": False,
        "remote_memory": {
            "configuration": str(config.remote_memory.path),
            "memory_type": "PER_NPU_MEMORY_EXPANSION",
            "edge_npus": list(config.remote_memory.edge_npus),
            "mesh_shape_rows_columns": [
                config.remote_memory.mesh_rows,
                config.remote_memory.mesh_cols,
            ],
            "selection_policy": "nearest_manhattan_then_min_rank",
            "logical_pool": config.remote_memory.logical_pool,
            "logical_pool_shared_across_edges": True,
            "independent_transaction_fifo_per_edge": True,
            "remote_mem_latency_ns": config.remote_memory.remote_mem_latency_ns,
            "remote_mem_bw_gbps_per_edge": config.remote_memory.remote_mem_bw_gbps,
            "non_edge_transport": "native_point_to_point_over_noc",
        },
        "weight_placement": "preloaded_npu_local_memory",
        "simulate_weight_startup_load": False,
        "nodes_per_rank": nodes_per_rank,
        "nodes_per_instance": {
            group.name: sum(nodes_per_rank[str(rank)] for rank in group.ranks)
            for group in config.inference_groups
        },
        "total_nodes": sum(nodes_per_rank.values()),
        "operator_granularity": OPERATOR_GRANULARITY,
        "all_reduce_layout": "2d_mesh_dimension_ordered",
        "all_reduce_dimensions": ["x", "y"],
        "all_reduce_phases": [
            "reduce_scatter_x",
            "all_reduce_y",
            "all_gather_x",
        ],
        "remote_operand_loads": config.remote_operand_loads,
        "request_end_barrier_all_reduce_bytes": 1,
    }
    manifest_text = json.dumps(manifest, indent=2)
    (output_dir / "manifest.json").write_text(manifest_text, encoding="utf-8")
    print(manifest_text)


def print_shell_config(config: TraceConfig) -> None:
    output_dir = resolve_output_dir(config)
    prefill_lengths = tuple(
        request.prefill_length for request in config.request_queue
    )
    decode_lengths = tuple(request.decode_length for request in config.request_queue)
    session_arrivals = ",".join(
        f"{request.session_id}:{request.session_arrival_time_ns}"
        for request in config.request_queue
        if request.turn_index == 0
    )
    inter_request_intervals = ",".join(
        f"{request.session_id}:turn{request.turn_index}:"
        f"{request.inter_request_interval_ns}"
        for request in config.request_queue
        if request.turn_index > 0
    )
    assignments = {
        "TRACE_PREFIX": config.output_prefix,
        "TRACE_DIR": str(output_dir),
        "TRACE_LABEL": build_trace_label(config),
        "NPUS_COUNT": config.npus_count,
        "REQUEST_COUNT": len(config.request_queue),
        "SESSION_COUNT": len({request.session_id for request in config.request_queue}),
        "REQUESTS_PER_SESSION": EXPECTED_REQUESTS_PER_SESSION,
        "REQUEST_TIMING_SOURCE": "request_queue_csv",
        "SESSION_ARRIVAL_TIMES_NS": session_arrivals,
        "INTER_REQUEST_INTERVALS_NS": inter_request_intervals,
        "PREFILL_RANGE": range_label(prefill_lengths),
        "DECODE_RANGE": range_label(decode_lengths),
        "EDGE_NPUS": ",".join(str(rank) for rank in config.remote_memory.edge_npus),
        "REMOTE_MEMORY": str(config.remote_memory.path),
    }
    for key, value in assignments.items():
        print(f"{key}={shlex.quote(str(value))}")


# Keep the module-level writer entry point aligned with the session-aware flow.
write_trace = write_session_trace


def main() -> None:
    """Dispatch the checked-in scenario to the FACE trace generator.

    The legacy helper classes and operator builder above remain importable so
    the FACE generator can reuse the existing Chakra/Transformer implementation.
    """
    from generate_face_trace import main as face_main

    face_main(sys.argv[1:])


if __name__ == "__main__":
    main()
