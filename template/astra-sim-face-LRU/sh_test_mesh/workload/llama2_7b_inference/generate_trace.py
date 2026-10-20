#!/usr/bin/env python3
"""Shared node-construction and configuration primitives for online LLaMA runs.

The online GraphBatch path imports this module's protobuf constants, in-memory
builder contract, request parsing, and transformer emission helpers.  It does
not serialize business Chakra ET files.
"""

import csv
import json
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
    Node as ChakraNode,
)


REMOTE_WEIGHT_ATTR = "remote_weight_bytes"

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


def clean_csv_row(
    row: dict[str, Optional[str]], context: str = ""
) -> dict[str, str]:
    """Strip row keys/values; fail closed on restkey (overlong) rows.

    (py-sched-E5, 2026-09-30) A data row carrying more fields than the
    header lands its surplus in csv.DictReader's restkey (``row[None]``);
    both that residue and any non-str non-None cell value used to vanish
    silently under the old ``(key or "")`` / ``(value or "")`` strip --
    the surplus fields were dropped without a diagnostic while the C++
    windowed reader's strict column-count gate fail-closes on the same
    overlong row.  Raise ValueError with the caller's line context
    instead of silently mis-shaping the row.
    """
    if None in row:
        extras = row[None]
        extra_count = len(extras) if isinstance(extras, list) else 1
        raise ValueError(
            f"{context}row carries {extra_count} more field(s) than the "
            f"CSV header (unparsed restkey residue {extras!r}); refusing "
            "to silently drop the surplus (the C++ windowed reader's "
            "column-count gate rejects overlong rows too)"
        )
    cleaned: dict[str, str] = {}
    for key, value in row.items():
        if value is not None and not isinstance(value, str):
            raise ValueError(
                f"{context}field {key!r} is not a CSV string cell "
                f"(got {type(value).__name__} {value!r})"
            )
        cleaned[(key or "").strip()] = (value or "").strip()
    return cleaned


def parse_request_length(row: dict[str, str], key: str, aliases: tuple[str, ...]) -> int:
    for column in (key, *aliases):
        value = row.get(column, "")
        if value:
            return parse_int(value, column)
    allowed = ", ".join((key, *aliases))
    raise ValueError(f"request queue row is missing one of: {allowed}")


def load_request_queue(queue_csv: Path) -> tuple[RequestSpec, ...]:
    if not queue_csv.exists():
        raise FileNotFoundError(f"request queue CSV not found: {queue_csv}")

    requests: list[RequestSpec] = []
    # py-sched-E4 (2026-09-30): request identity fail-closed at the loader
    # -- a duplicate request_id silently collides in every request_id-keyed
    # consumer (canonical sidecar join, metrics registration), and a
    # duplicate (session_id, turn_index) silently shadows the previous
    # turn in the session-turn keyed derivations (plan_materializer's
    # queue_by_session_turn map).  '#' comment rows never register.
    seen_request_ids: set[str] = set()
    seen_session_turns: set[tuple[str, int]] = set()
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
            row = clean_csv_row(
                raw_row, context=f"request queue line {line_number}: ")
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
            if request_id in seen_request_ids:
                raise ValueError(
                    f"request queue line {line_number}: duplicate "
                    f"request_id {request_id!r} (already used by an "
                    "earlier row)"
                )
            if (session_id, turn_index) in seen_session_turns:
                raise ValueError(
                    f"request queue line {line_number}: duplicate "
                    f"(session_id, turn_index) ({session_id!r}, "
                    f"{turn_index}); each session may carry each turn "
                    "index at most once"
                )
            seen_request_ids.add(request_id)
            seen_session_turns.add((session_id, turn_index))
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
    """P3(face-LRU 审计,2026-09-29):保留——生产构造点已随离线静态管线
    删除(2026-08-18),现存构造仅 test_face_scheduler 的测试对拍基准;
    generate_face_trace 的 builders 形参注解仍引用本类。在线构图用
    graph_batch_builder.OnlineTraceBuilder(独立实现,勿混淆)。"""

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
        if duration_ns < 0 or duration_ns % 1000 != 0:
            raise ValueError(
                "timer duration must be a non-negative whole number of microseconds"
            )
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
        *,
        die_spec: Optional[dict] = None,
        op_role: Optional[str] = None,
    ) -> None:
        # die 后端(阶段 C;face-LRU 随迁 2026-10-03):die_spec 存在 =>
        # runtime_ns 语义由在线侧 OnlineTraceBuilder 保证为 0(ET 侧无
        # runtime 字段);op_role 缺省 "compute"(契约 §2),仅 marker 显式
        # 写 "control_marker"。
        node = self._new_node(name, COMP_NODE)
        node.attr.append(self._uint64_attr("num_ops", num_ops))
        node.attr.append(self._uint64_attr("tensor_size", tensor_size))
        if op_role is not None:
            node.attr.append(ChakraAttr(name="op_role", string_val=op_role))
        if die_spec is not None:
            node.attr.append(
                ChakraAttr(
                    name="die_op_spec",
                    string_val=json.dumps(die_spec, sort_keys=True),
                )
            )
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
    # P3(face-LRU 审计,2026-09-29):逐层展开版保留——全仓唯一调用 =
    # test_face_scheduler 的测试对拍基准(与 transformer_pass_aggregated
    # 的逐字节总额对拍;在线路径只用聚合版)。
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
    layer_start: int = 0,
    layer_end: Optional[int] = None,
    include_output: bool = True,
    weight_passes: Optional[int] = None,
    die_ctx=None,
) -> int:
    """Fold repeated Transformer passes into 17 aggregate Chakra nodes.

    Each ``(tokens, kv_length)`` span represents one call to
    :func:`transformer_pass`.  The aggregate nodes preserve the exact sums of
    ``num_ops``, ``tensor_size``, optional remote-read bytes, and All-Reduce
    payload bytes from those expanded calls.  What is intentionally compressed
    is the number of layer, chunk/token, and collective invocations.

    ``weight_passes`` (拼 batch 口径, 2026-08-22): the number of physical
    forward passes that read the model weights.  The default ``None`` (=
    ``len(pass_spans)``) keeps the historical byte-exact behavior -- every
    span is one weight-reading pass (batch=1 serial semantics; the offline
    fixtures and the five-repo consistency audit depend on it).  An
    iteration-train caller passes the ITERATION COUNT instead: a train of k
    iterations with B decode members carries k+B spans but reads the weights
    only k times (once per iteration, shared by all batch members; 权重只读
    一次 per iteration).  Activation / KV / AllReduce bytes stay per-span
    exact regardless of ``weight_passes``.  Layer-range calls (partial
    prefix-restore pipelines) keep the default: one span over a layer slice
    is exactly one weight-reading pass of that slice.

    ``die_ctx``(阶段 C;face-LRU 随迁 2026-10-03):die 后端发射语境
    (online/die_model_adapter.DieModelContext;调用方 graph_batch_builder
    已按列车/TP/rank 调过 begin_emission)。None = legacy 路径,发射逐字节
    与历史一致;非 None 时每类 COMP 附 die_op_spec(spans 带绝对位置),
    可融合纯本地链经 compile_regions 编译成 attention_region /
    ffn_region(num_ops/tensor_size 兼容统计 = 各成分之和,collective
    位置/数量/bytes 不变)。
    """

    spans = tuple(pass_spans)
    if not spans:
        raise ValueError("pass_spans must contain at least one Transformer pass")
    if layers <= 0:
        raise ValueError("layers must be positive")
    if layer_end is None:
        layer_end = layers
    if not 0 <= layer_start < layer_end <= layers:
        raise ValueError("Transformer layer range is outside the model")
    active_layers = layer_end - layer_start
    layer_label = (
        "all_layers"
        if layer_start == 0 and layer_end == layers
        else f"layers{layer_start:02d}_{layer_end - 1:02d}"
    )
    if hidden_size <= 0 or ffn_size <= 0 or tensor_parallel <= 0:
        raise ValueError("model dimensions and tensor_parallel must be positive")
    if vocab_size <= 0 or bytes_per_elem <= 0:
        raise ValueError("vocab_size and bytes_per_elem must be positive")
    for tokens, kv_length in spans:
        if tokens <= 0 or kv_length <= 0:
            raise ValueError("each pass span must contain positive tokens and kv_length")

    if weight_passes is None:
        weight_passes = len(spans)
    if weight_passes <= 0:
        raise ValueError("weight_passes must be positive")
    if weight_passes > len(spans):
        raise ValueError(
            "weight_passes must not exceed the span count (one weight-reading "
            "pass contributes at least one span)")

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

        # 拼批量权重基线拆分(2026-08-22):per-span 累加只含激活/KV/score
        # 分量;权重分量(norm 参数/投影矩阵)每物理前向只读一次,统一在
        # span 循环后按 weight_passes 计入(默认 = len(spans) 与历史
        # batch=1 串行口径逐字节一致;迭代列车传迭代数,见 docstring)。
        add(
            layer_totals[attention_norm_name],
            norm_ops_factor * activation_elems,
            3 * activation_bytes,
            activation_bytes,
        )
        add(
            layer_totals["attention_qkv_projection"],
            matmul_ops(tokens, 3 * attention_hidden_per_rank, hidden_size),
            activation_bytes + 3 * activation_shard_bytes,
            activation_bytes,
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
            activation_shard_bytes + activation_bytes,
            activation_shard_bytes,
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
            3 * activation_bytes,
            activation_bytes,
        )
        add(
            layer_totals[mlp_projection_name],
            matmul_ops(tokens, mlp_projection_factor * ffn_per_rank, hidden_size),
            activation_bytes
            + mlp_projection_factor * ffn_shard_bytes,
            activation_bytes,
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
            ffn_shard_bytes + activation_bytes,
            ffn_shard_bytes,
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
            3 * activation_bytes,
            activation_bytes,
        )
        logits_output_bytes = tensor_bytes(
            tokens * vocab_per_rank, bytes_per_elem
        )
        add(
            logits,
            matmul_ops(tokens, vocab_per_rank, hidden_size),
            activation_bytes + logits_output_bytes,
            activation_bytes,
        )

    # 权重分量按 weight_passes 计入(每个物理前向读一遍;与上方 span 循环
    # 的激活/KV 分量求和后即为最终 category 总量)。weight_passes 缺省 =
    # len(spans) 时与拆分前的历史总量逐字节相等。
    weight_tensor_bytes = {
        attention_norm_name: norm_param_bytes,
        "attention_qkv_projection": qkv_weight_bytes,
        "attention_output_projection": out_weight_bytes,
        mlp_norm_name: norm_param_bytes,
        mlp_projection_name: mlp_up_weight_bytes,
        "mlp_down_projection": mlp_down_weight_bytes,
    }
    for category_name, weight_bytes in weight_tensor_bytes.items():
        layer_totals[category_name][1] += weight_bytes * weight_passes
        layer_totals[category_name][2] += weight_bytes * weight_passes
    final_layernorm[1] += norm_param_bytes * weight_passes
    final_layernorm[2] += norm_param_bytes * weight_passes
    logits[1] += logits_weight_bytes * weight_passes
    logits[2] += logits_weight_bytes * weight_passes

    if die_ctx is not None:
        # die 后端(阶段 C,契约 §2/§7.2):同一份 layer_totals / tail /
        # collective 账本,经 die 语境做生成期 region 编译后发射(每类或
        # 融合组附 die_op_spec,spans 带绝对位置)。legacy 分支
        # (die_ctx is None)保持逐字节不变。
        _emit_die_aggregated(
            builder,
            die_ctx,
            phase=phase,
            spans=spans,
            active_layers=active_layers,
            layer_label=layer_label,
            weight_passes=weight_passes,
            layer_totals=layer_totals,
            attention_collective_bytes=attention_collective_bytes,
            mlp_collective_bytes=mlp_collective_bytes,
            final_layernorm=final_layernorm,
            logits=logits,
            attention_norm_name=attention_norm_name,
            mlp_norm_name=mlp_norm_name,
            mlp_projection_name=mlp_projection_name,
            mlp_activation_name=mlp_activation_name,
            final_norm_name=final_norm_name,
            include_output=include_output,
            pg_name=pg_name,
            tensor_parallel=tensor_parallel,
            tensor_parallel_rank=tensor_parallel_rank,
        )
        return len(spans)

    def aggregate_comp(category_name: str) -> None:
        ops, tensor_size, remote_read = layer_totals[category_name]
        builder.comp(
            f"{phase}_{layer_label}_{category_name}",
            ops * active_layers,
            tensor_size * active_layers,
            remote_read * active_layers,
        )

    aggregate_comp(attention_norm_name)
    aggregate_comp("attention_qkv_projection")
    aggregate_comp("attention_qk_matmul")
    aggregate_comp("attention_scale_mask")
    aggregate_comp("attention_softmax")
    aggregate_comp("attention_av_matmul")
    aggregate_comp("attention_output_projection")
    builder.all_reduce(
        f"{phase}_{layer_label}_attention_all_reduce",
        attention_collective_bytes * active_layers,
        pg_name,
    )
    aggregate_comp("attention_residual_add")
    aggregate_comp(mlp_norm_name)
    aggregate_comp(mlp_projection_name)
    aggregate_comp(mlp_activation_name)
    aggregate_comp("mlp_down_projection")
    builder.all_reduce(
        f"{phase}_{layer_label}_mlp_all_reduce",
        mlp_collective_bytes * active_layers,
        pg_name,
    )
    aggregate_comp("mlp_residual_add")
    if not include_output:
        return len(spans)
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


def _emit_die_aggregated(
    builder: TraceBuilder,
    die_ctx,
    *,
    phase: str,
    spans: tuple[tuple[int, int], ...],
    active_layers: int,
    layer_label: str,
    weight_passes: int,
    layer_totals: dict[str, list[int]],
    attention_collective_bytes: int,
    mlp_collective_bytes: int,
    final_layernorm: list[int],
    logits: list[int],
    attention_norm_name: str,
    mlp_norm_name: str,
    mlp_projection_name: str,
    mlp_activation_name: str,
    final_norm_name: str,
    include_output: bool,
    pg_name: str,
    tensor_parallel: int,
    tensor_parallel_rank: int,
) -> None:
    """die 后端的聚合发射(阶段 C;transformer_pass_aggregated 的 die 分支;
    face-LRU 随迁 2026-10-03,命名随本仓 layer_label/变体类别名适配)。

    与 legacy 发射共用同一份 layer_totals / tail / collective 账本(字节
    守恒);差异仅在:

      - 每个发射组附 die_op_spec(契约 §2 schema;spans 带绝对位置,
        layer_count/weight_passes 如实传递,num_ops/tensor_size 照旧发射
        作兼容统计,die 后端不把它们当第二份收费来源);
      - 13 类层类别先经 die_ctx.compile_regions 做 §7.2 纯本地链融合
        (qk+scale_mask+softmax+av -> attention_region;mlp_up+
        activation+down -> ffn_region);融合组的 num_ops/tensor_size/
        remote_read = 各成分之和(×active_layers 口径与 legacy 一致);
      - 两个 All-Reduce 的位置/bytes 与 legacy 完全一致(边界类别
        attention_output_projection / mlp_down_projection 之后,融合
        不移动 collective)。

    阻断/同质事实(§7.2 融合条件,逐项来自本发射骨架的真实结构):
    聚合列车体是严格线性链(previous_id 单前驱,无 pending 额外依赖),
    链内无外部分支消费者(external_consumers=False);通信边界仅存在于
    attention_output_projection 与 mlp_down_projection 之后(boundary_
    after=True,二者分别是各自融合链之外/链尾,链内无边界);无 KV gate
    与提交边界(gate 在 admission/列车边界,由调度器侧发射)。identity
    携带 TP/rank/层段/span 集合供 compile_regions 做同质复核。
    """
    if len(die_ctx.span_records) != len(spans):
        raise ValueError(
            "die context span records ({} spans) disagree with the "
            "aggregated pass spans ({}); call begin_emission with the "
            "same pass_spans".format(len(die_ctx.span_records), len(spans)))
    identity = {
        "tp": tensor_parallel,
        "rank": tensor_parallel_rank,
        "layer": active_layers,
        "span_set": spans,
    }
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
    records = []
    for category in category_names:
        ops, tensor_size, remote_read = layer_totals[category]
        records.append({
            "name": f"{phase}_{layer_label}_{category}",
            "category": category,
            "op_kind": die_ctx.op_kind_of(category),
            "num_ops": ops * active_layers,
            "tensor_size": tensor_size * active_layers,
            "remote_read": remote_read * active_layers,
            "layer_count": active_layers,
            "weight_passes": weight_passes,
            "external_consumers": False,
            "boundary_after": category in (
                "attention_output_projection", "mlp_down_projection"),
            "identity": dict(identity),
        })
    for group in die_ctx.compile_regions(records):
        builder.comp(
            group["name"],
            group["num_ops"],
            group["tensor_size"],
            group["remote_read"],
            die_spec=die_ctx.build_spec(
                group["op_kind"],
                layer_count=group["layer_count"],
                weight_passes=group["weight_passes"],
                fusion_policy="auto" if group["fused"] else "none",
            ),
        )
        if "attention_output_projection" in group["member_categories"]:
            builder.all_reduce(
                f"{phase}_{layer_label}_attention_all_reduce",
                attention_collective_bytes * active_layers,
                pg_name,
            )
        elif "mlp_down_projection" in group["member_categories"]:
            builder.all_reduce(
                f"{phase}_{layer_label}_mlp_all_reduce",
                mlp_collective_bytes * active_layers,
                pg_name,
            )
    if not include_output:
        return
    for category, totals, layer_count in (
        (final_norm_name, final_layernorm, 1),
        ("logits_projection", logits, 1),
    ):
        builder.comp(
            f"{phase}_all_passes_{category}",
            totals[0],
            totals[1],
            totals[2],
            die_spec=die_ctx.build_spec(
                die_ctx.op_kind_of(category),
                layer_count=layer_count,
                weight_passes=weight_passes,
                fusion_policy="none",
            ),
        )


def sanitize_node_prefix(value: str) -> str:
    return "".join(
        character if character.isalnum() or character in {"_", "-"} else "_"
        for character in value
    )


def main() -> None:
    """fail-closed 拒绝桩(2026-08-18 起生效):离线静态 ET 生成入口已删除。

    本模块保留的是③④在线路径只读 import 的符号库(Chakra 常量/
    TraceBuilder/transformer_pass(_aggregated)/RequestSpec 等)。
    ③④ 的输入物化入口是 plan_materializer.py;静态 ET 生成入口不再存在。
    """
    raise SystemExit(
        "path-1 (offline static full pipeline) was removed on 2026-08-18; "
        "use plan_materializer.py for the online routes' plan-dir inputs")


if __name__ == "__main__":
    main()
