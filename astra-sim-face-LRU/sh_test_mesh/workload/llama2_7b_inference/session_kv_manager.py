"""Session-scoped tiered KV cache management (LOCAL / PARTIAL / REMOTE).

B2 (2026-09-06, face-LRU 三态化): the two-state RESIDENT/EVICTED model with
zero-cost delete + full recompute is replaced by the sh_2.0 three-state
tiered model, adapted onto face's existing external integration surface
(KVCacheEvent audit stream, deep-gap int counter, admission_blocked retry
semantics, reservation API):

- ``LOCAL_HBM``           all ``model.layers`` layers resident on one instance;
- ``PARTIAL_HBM_REMOTE``  leading ``partial_resident_prefix_layers`` layers
  resident, trailing half in the remote pool;
- ``REMOTE_MEMORY``       every layer in the remote pool, no local instance.

Reclamation is the de-typed two-stage LRU order (no ``_eviction_class``, no
``trigger_type``, no ``next_request_type``): stage 1 suffix-halves completed
inactive LOCAL sessions oldest-first; only after every candidate is halved
does stage 2 fully offload resident sessions (the sole fallback).  The
requirement is rechecked after every single eviction.  Exhaustion keeps the
face failure semantics (``admission_blocked`` graceful deferral with the
deep-gap ledger; only a structurally impossible request raises).

The blueprint is sh_2.0 ``face_scheduler.py``'s ``KVCacheManager``; face's
mapping module (``face_scheduler.py`` in this repository) is untouched.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Legacy two-state action vocabulary (kept importable: face_scheduler.py /
# generate_face_trace.py / graph_batch_builder.py re-export or compare these
# names; after B2 the manager itself never produces RECOMPUTE decisions and
# the RECOMPUTE constant itself is gone — B2 deleted the recompute path).
# ---------------------------------------------------------------------------
RESIDENT = "RESIDENT"
EVICTED = "EVICTED"
NO_HISTORY = "NO_HISTORY"
LOCAL_HIT = "LOCAL_HIT"
NOC_MIGRATE = "NOC_MIGRATE"

# B2 three-state session locations and the restore-path history actions.
LOCAL_HBM = "local_hbm"
PARTIAL_HBM_REMOTE = "partial_hbm_remote"
REMOTE_MEMORY = "remote_memory"
_REMOTE_SUFFIX_RESTORE = "REMOTE_SUFFIX_RESTORE"
_PARTIAL_REMOTE_MIGRATE = "PARTIAL_REMOTE_MIGRATE"
_REMOTE_RESTORE = "REMOTE_RESTORE"

# B1 已消费前缀的摊销压缩水位(2026-08-28,风格对齐 graph_batch_builder
# 的 M1 压缩):_events 已消费水位达到该值且不小于现存总量一半时才整段
# 删除前缀,均摊 O(1)/事件。
_EVENTS_COMPACT_THRESHOLD = 8192

# Read-only metrics observation (implementation doc sec.7).  When a recorder
# is installed through set_metrics_observer(), every state mutation below is
# mirrored to it *after* the mutation completes; the recorder never feeds
# anything back into admission, eviction, or placement decisions.  When no
# recorder is installed (the default) none of the observation bookkeeping
# runs at all, so behavior and performance are unchanged.
_METRICS_RECORDER: Any = None


def _strict_kv_invariants_from_environment() -> bool:
    """Return whether every mutation must also run the complete audit."""

    return os.environ.get("SH_STRICT_KV_INVARIANTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def set_metrics_observer(recorder: Any) -> None:
    """Install the metrics recorder picked up by subsequently constructed
    managers (``None`` disables observation)."""

    global _METRICS_RECORDER
    _METRICS_RECORDER = recorder


def _metrics_anchor_for_phase(phase: str) -> str:
    """Map a planner phase to the doc sec.7.8 anchor for its memory actions."""

    return {
        "history": "prefill_start",
        "prefill": "prefill_start",
        "admission": "prefill_start",
        "decode": "decode_start",
        "prefill_decode": "decode_start",
        "completion": "completion",
    }.get(phase, "stage_boundary")


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def partition_values_exact(total: int, partitions: int) -> tuple[int, ...]:
    """Split an integer without padding, putting remainders on low ranks."""

    _require_nonnegative_int(total, "total")
    if isinstance(partitions, bool) or not isinstance(partitions, int) or partitions <= 0:
        raise ValueError("partitions must be a positive integer")
    quotient, remainder = divmod(total, partitions)
    return tuple(quotient + (1 if index < remainder else 0) for index in range(partitions))


def attention_heads_by_tp_rank(num_heads: int, tp_degree: int) -> tuple[int, ...]:
    """Return the whole-head ownership of each relative TP rank."""

    if (
        isinstance(num_heads, bool)
        or isinstance(tp_degree, bool)
        or not isinstance(num_heads, int)
        or not isinstance(tp_degree, int)
        or num_heads <= 0
        or tp_degree <= 0
    ):
        raise ValueError("num_heads and tp_degree must be positive integers")
    return partition_values_exact(num_heads, tp_degree)


def estimate_model_weight_bytes(model: Any) -> int:
    """Match the existing LLaMA-family model-size estimate exactly."""

    mlp_matrices = 3 if getattr(model, "mlp_variant", "gelu") == "swiglu" else 2
    norm_elements = (
        2 * model.hidden_size
        if getattr(model, "mlp_variant", "gelu") == "swiglu"
        else 4 * model.hidden_size
    )
    per_layer_elements = (
        4 * model.hidden_size * model.hidden_size
        + mlp_matrices * model.hidden_size * model.ffn_size
        + norm_elements
    )
    embedding_elements = 2 * model.vocab_size * model.hidden_size
    final_norm_elements = (
        model.hidden_size if getattr(model, "mlp_variant", "gelu") == "swiglu" else 0
    )
    return (
        model.layers * per_layer_elements + embedding_elements + final_norm_elements
    ) * model.bytes_per_elem


def model_weight_shard_bytes_by_tp_rank(model: Any, tp_degree: int) -> tuple[int, ...]:
    """Return exact TP weight shards aligned with whole-head KV ownership."""

    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    ffn_extents = partition_values_exact(model.ffn_size, tp_degree)
    vocab_extents = partition_values_exact(model.vocab_size, tp_degree)
    head_dim = model.hidden_size // model.num_heads
    mlp_matrices = 3 if getattr(model, "mlp_variant", "gelu") == "swiglu" else 2
    matrix_shards = tuple(
        (
            model.layers
            * (
                4 * model.hidden_size * head_count * head_dim
                + mlp_matrices * model.hidden_size * ffn_extent
            )
            + 2 * model.hidden_size * vocab_extent
        )
        * model.bytes_per_elem
        for head_count, ffn_extent, vocab_extent in zip(heads, ffn_extents, vocab_extents)
    )
    total = estimate_model_weight_bytes(model)
    residual = total - sum(matrix_shards)
    if residual < 0:
        raise RuntimeError("TP weight matrix shards exceed the model total")
    shards = tuple(
        shard + remainder
        for shard, remainder in zip(matrix_shards, partition_values_exact(residual, tp_degree))
    )
    if sum(shards) != total:
        raise RuntimeError("TP weight shards do not preserve the model total")
    return shards


def kv_cache_shard_bytes_for_tokens(
    model: Any,
    tokens: int,
    tp_degree: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV shard bytes for a complete model cache."""

    return kv_cache_shard_bytes_for_layer_range(
        model,
        tokens,
        tp_degree,
        layer_start=0,
        layer_end=model.layers,
    )


def kv_cache_shard_bytes_for_layer_range(
    model: Any,
    tokens: int,
    tp_degree: int,
    *,
    layer_start: int,
    layer_end: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV bytes for ``[layer_start, layer_end)``.

    B2 (2026-09-06, sh_2.0 ``face_scheduler.py:400-443``): layer ranges are
    derived from ``model.layers`` so suffix offloading stays model-independent;
    the per-rank whole-head split and the total-byte conservation assertion
    match the blueprint exactly.
    """

    _require_nonnegative_int(tokens, "tokens")
    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    if (
        isinstance(layer_start, bool)
        or isinstance(layer_end, bool)
        or not isinstance(layer_start, int)
        or not isinstance(layer_end, int)
        or not 0 <= layer_start <= layer_end <= model.layers
    ):
        raise ValueError("KV layer range must be within the configured model")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    head_dim = model.hidden_size // model.num_heads
    bytes_per_head = (
        2
        * (layer_end - layer_start)
        * tokens
        * head_dim
        * model.bytes_per_elem
    )
    shards = tuple(head_count * bytes_per_head for head_count in heads)
    expected_total = (
        2 * tokens * model.hidden_size * model.bytes_per_elem * (layer_end - layer_start)
    )
    if sum(shards) != expected_total:
        raise RuntimeError(
            "whole-head layer-range KV partition does not preserve total bytes"
        )
    return shards


def _rank_coordinates(hardware: Any, rank: int) -> tuple[int, int]:
    """Lazy re-export of the mapping module's coordinate helper.

    face_scheduler.py imports this module at load time (NOC_MIGRATE
    re-export), so a module-level back-import would be circular; the lazy
    call site keeps a single authoritative implementation.
    """

    from face_scheduler import rank_coordinates

    return rank_coordinates(hardware, rank)


def physical_edge_ranks(hardware: Any) -> tuple[int, ...]:
    """Return all ranks on the deterministic physical mesh boundary.

    B2 (2026-09-06, sh_2.0 ``face_scheduler.py:446-459``): the remote-memory
    port set is the complete mesh boundary.
    """

    edges = []
    for rank in range(hardware.npus_count):
        row, col = _rank_coordinates(hardware, rank)
        if (
            row == 0
            or row == hardware.mesh_rows - 1
            or col == 0
            or col == hardware.mesh_cols - 1
        ):
            edges.append(rank)
    return tuple(edges)


def manhattan_hops(hardware: Any, source: int, target: int) -> int:
    """B2 (sh ``:462-465``): Manhattan distance in mesh coordinates."""

    source_row, source_col = _rank_coordinates(hardware, source)
    target_row, target_col = _rank_coordinates(hardware, target)
    return abs(source_row - target_row) + abs(source_col - target_col)


def nearest_edge_rank(
    hardware: Any,
    rank: int,
    edge_ranks: Sequence[int],
) -> int:
    """B2 (sh ``:487-498``): nearest boundary rank, ties by lowest index."""

    if not edge_ranks:
        raise ValueError("at least one remote-memory edge rank is required")
    _rank_coordinates(hardware, rank)
    return min(
        edge_ranks,
        key=lambda edge: (manhattan_hops(hardware, rank, edge), edge),
    )


@dataclass(frozen=True)
class NodeHBMSnapshot:
    rank: int
    instance_index: int
    capacity_bytes: int
    model_weight_bytes: int
    resident_kv_bytes: int
    reserved_request_bytes: int
    used_bytes: int
    remaining_bytes: int


@dataclass
class NodeHBMState:
    rank: int
    instance_index: int
    capacity_bytes: int
    model_weight_bytes: int
    resident_kv_bytes: int = 0
    reserved_request_bytes: int = 0

    @property
    def used_bytes(self) -> int:
        return self.model_weight_bytes + self.resident_kv_bytes + self.reserved_request_bytes

    @property
    def remaining_bytes(self) -> int:
        return self.capacity_bytes - self.used_bytes

    def snapshot(self) -> NodeHBMSnapshot:
        return NodeHBMSnapshot(
            rank=self.rank,
            instance_index=self.instance_index,
            capacity_bytes=self.capacity_bytes,
            model_weight_bytes=self.model_weight_bytes,
            resident_kv_bytes=self.resident_kv_bytes,
            reserved_request_bytes=self.reserved_request_bytes,
            used_bytes=self.used_bytes,
            remaining_bytes=self.remaining_bytes,
        )


@dataclass(frozen=True)
class SessionKVSnapshot:
    """B2 three-state snapshot with local/remote byte derivations.

    ``shard_bytes`` remains the full-model logical shard vector (all layers
    at ``logical_context_tokens``); the physically resident distribution is
    ``local_shard_bytes`` (layer range ``[0, resident_prefix_layers)``) and
    the remote remainder is ``remote_shard_bytes`` (sh ``:1748-1787``).
    """

    session_id: str
    location: str
    instance_index: Optional[int]
    logical_context_tokens: int
    total_bytes: int
    shard_bytes: tuple[int, ...]
    resident_prefix_layers: int
    local_bytes: int
    remote_bytes: int
    local_shard_bytes: tuple[int, ...]
    remote_shard_bytes: tuple[int, ...]
    last_completion_ns: Optional[int]
    active: bool
    last_request_id: Optional[str] = None
    evicted_at_ns: Optional[int] = None
    evicted_by_request_id: Optional[str] = None


@dataclass
class SessionKVState:
    session_id: str
    location: str
    instance_index: Optional[int]
    logical_context_tokens: int
    total_bytes: int
    shard_bytes: tuple[int, ...]
    resident_prefix_layers: int
    last_completion_ns: Optional[int] = None
    active: bool = False
    last_request_id: Optional[str] = None
    evicted_at_ns: Optional[int] = None
    evicted_by_request_id: Optional[str] = None


@dataclass(frozen=True)
class KVTransferShard:
    """B2 KVTransfer shard (contract §4): physical edge port + XY path +
    layer range per shard (sh ``:1119-1133``)."""

    source_rank: Optional[int]
    target_rank: Optional[int]
    edge_rank: Optional[int]
    bytes: int
    noc_path: tuple[int, ...]
    layer_start: int
    layer_end: int

    def __post_init__(self) -> None:
        if self.bytes < 0:
            raise ValueError("KV transfer shard bytes must be non-negative")
        if self.layer_start < 0 or self.layer_end <= self.layer_start:
            raise ValueError("KV transfer shard requires a non-empty layer range")


@dataclass(frozen=True)
class KVTransfer:
    """B2 KVTransfer (contract §4, sh ``:1136-1183``): kind carrier with the
    ``local_hit / noc_migrate / remote_load / remote_store`` whitelist and
    layer-domain conservation validation."""

    kind: str
    phase: str
    reason: str
    session_id: str
    trigger_request_id: str
    source_instance_index: Optional[int]
    target_instance_index: Optional[int]
    total_bytes: int
    shards: tuple[KVTransferShard, ...]
    model_layers: int
    layer_start: int
    layer_end: int
    resident_prefix_layers_before: int
    resident_prefix_layers_after: int

    def __post_init__(self) -> None:
        if self.kind not in {
            "local_hit",
            "noc_migrate",
            "remote_load",
            "remote_store",
        }:
            raise ValueError(f"unsupported KV transfer kind: {self.kind}")
        if self.total_bytes < 0:
            raise ValueError("KV transfer bytes must be non-negative")
        if (
            self.model_layers <= 0
            or not 0 <= self.layer_start < self.layer_end <= self.model_layers
        ):
            raise ValueError("KV transfer layer range is outside the model")
        if not (
            0 <= self.resident_prefix_layers_before <= self.model_layers
            and 0 <= self.resident_prefix_layers_after <= self.model_layers
        ):
            raise ValueError("KV resident-prefix metadata is outside the model")
        shard_total = sum(shard.bytes for shard in self.shards)
        if self.kind == "local_hit" and self.shards:
            raise ValueError("local-hit KV transfer must not contain network shards")
        if self.kind != "local_hit" and shard_total != self.total_bytes:
            raise ValueError("KV transfer shard bytes do not match total bytes")
        if any(
            shard.layer_start != self.layer_start
            or shard.layer_end != self.layer_end
            for shard in self.shards
        ):
            raise ValueError("KV transfer shards must use the transfer layer range")


@dataclass(frozen=True)
class EvictionRecord:
    time_ns: int
    phase: str
    reason: str
    trigger_request_id: str
    victim_session_id: str
    victim_instance_index: int
    victim_last_completion_ns: int
    context_tokens: int
    shard_bytes: tuple[int, ...]
    # B2: the tiered-eviction KVTransfer (remote_store, layer-ranged) emitted
    # by the same mutation.  The legacy eight-field row above keeps the
    # existing decision-log serialization runnable; consumers of the tiered
    # fields (kind / layers / edge ports) read them off ``transfer``.
    transfer: Optional[KVTransfer] = None


@dataclass(frozen=True)
class CapacityResult:
    evictions: tuple[EvictionRecord, ...]
    admitted: bool
    insufficient_ranks: tuple[int, ...]


@dataclass(frozen=True)
class RequestCapacityReservation:
    request_id: str
    session_id: str
    instance_index: int
    shard_bytes: tuple[int, ...]


@dataclass(frozen=True)
class HistoryDecision:
    action: str
    source_instance_index: Optional[int]
    target_instance_index: int
    history_tokens: int
    transfer_shards: tuple[KVTransferShard, ...]
    recompute_tokens: int
    evictions: tuple[EvictionRecord, ...]
    admission_blocked: bool = False
    insufficient_ranks: tuple[int, ...] = ()
    # B2: contract §4 KVTransfer objects for the restore/migrate path.  A
    # PARTIAL cross-instance restore carries both segments (prefix NoC
    # migrate + suffix remote load) in ONE decision so the scheduler logs a
    # single prefill record (hbm_watermark "same request + same kind twice
    # fails" guard).
    transfers: tuple[KVTransfer, ...] = ()
    location_before: Optional[str] = None
    resident_prefix_layers_before: Optional[int] = None

    @property
    def transfer_bytes(self) -> int:
        return sum(transfer.total_bytes for transfer in self.transfers)


@dataclass(frozen=True)
class MoveDecision:
    action: str
    source_instance_index: int
    target_instance_index: int
    transfer: KVTransfer
    evictions: tuple[EvictionRecord, ...]
    admission_blocked: bool = False
    insufficient_ranks: tuple[int, ...] = ()


@dataclass(frozen=True)
class KVCacheEvent:
    event_index: int
    planner_time_ns: int
    phase: str
    event_type: str
    reason: str
    trigger_request_id: str
    session_id: Optional[str]
    source_instance_index: Optional[int]
    target_instance_index: Optional[int]
    context_tokens: int
    total_bytes: int
    shard_bytes: tuple[int, ...]
    last_completion_ns: Optional[int]
    instance_remaining_before_bytes: tuple[int, ...]
    instance_remaining_after_bytes: tuple[int, ...]
    insufficient_ranks: tuple[int, ...]


class SessionKVCacheManager:
    """Exact TP-sharded session KV state machine with tiered reclamation.

    All mutating operations are planner-time operations.  The caller emits
    the corresponding NoC / remote-pool / restore ET nodes from the returned
    decisions; the remote pool itself has no capacity limit and never
    displaces on its own (sh semantics).
    """

    def __init__(
        self,
        topology: Any,
        model: Any,
        *,
        edge_ranks: Optional[Sequence[int]] = None,
        strict_invariants: Optional[bool] = None,
    ) -> None:
        if strict_invariants is not None and not isinstance(strict_invariants, bool):
            raise ValueError("strict_invariants must be a bool or None")
        instances = tuple(topology.instances)
        if not instances:
            raise ValueError("at least one TP instance is required")
        instance_sizes = {len(instance.ranks) for instance in instances}
        if len(instance_sizes) != 1:
            raise ValueError("SessionKVCacheManager requires equal-size TP instances")
        self.topology = topology
        self.model = model
        self.tp_degree = next(iter(instance_sizes))
        # B2 (sh ``:1210-1214``): keep the larger half resident for odd-layer
        # models; exactly floor(L/2) trailing layers form the stage-1 suffix.
        # Not configurable (contract §9).
        self.partial_resident_prefix_layers = model.layers - model.layers // 2
        self.model_weight_bytes_by_tp_rank = model_weight_shard_bytes_by_tp_rank(
            model, self.tp_degree
        )
        # B2 (sh ``:1220-1236``): remote-memory edge ports, defaulting to the
        # complete mesh boundary.
        if edge_ranks is None:
            normalized_edges = physical_edge_ranks(topology.hardware)
        else:
            provided_edges = tuple(edge_ranks)
            if any(
                isinstance(rank, bool) or not isinstance(rank, int)
                for rank in provided_edges
            ):
                raise ValueError("edge_ranks must contain integer ranks")
            if len(set(provided_edges)) != len(provided_edges):
                raise ValueError("edge_ranks must not contain duplicates")
            normalized_edges = tuple(sorted(provided_edges))
        if not normalized_edges:
            raise ValueError("at least one remote-memory edge rank is required")
        for rank in normalized_edges:
            _rank_coordinates(topology.hardware, rank)
        self.edge_ranks = normalized_edges

        self._rank_states: dict[int, NodeHBMState] = {}
        self._rank_relative_indexes: dict[int, int] = {}
        for instance in instances:
            if len(instance.ranks) != self.tp_degree:
                raise ValueError("inconsistent TP degree in topology")
            for relative_rank, rank in enumerate(instance.ranks):
                state = NodeHBMState(
                    rank=rank,
                    instance_index=instance.index,
                    capacity_bytes=topology.hardware.local_hbm_capacity_bytes,
                    model_weight_bytes=self.model_weight_bytes_by_tp_rank[relative_rank],
                )
                if state.remaining_bytes < 0:
                    raise ValueError(
                        f"model weight shard does not fit rank {rank}: "
                        f"needs {state.model_weight_bytes}, "
                        f"has {state.capacity_bytes}"
                    )
                self._rank_states[rank] = state
                self._rank_relative_indexes[rank] = relative_rank
        self._sessions: dict[str, SessionKVState] = {}
        self._reservations: dict[str, RequestCapacityReservation] = {}
        self._events: list[KVCacheEvent] = []
        # B1(2026-08-28):kv 事件流水号独立于列表位置——前缀压缩后
        # len(self._events) 不再可用作 event_index(会与存活事件撞号),
        # 改由单调计数器分配,流内 event_index 语义/取值与改前一致。
        self._event_index_next = 0
        # A blocked request can be reconsidered many times while unrelated
        # active Decode work advances.  Keep the audit trail bounded: retain
        # the first pressure observation and one explicit retry per request /
        # phase / target rather than appending an identical event per token.
        self._pressure_event_keys: set[tuple[str, str, str, int]] = set()
        # Completion must retire the dedup keys of every turn, not merely the
        # terminal turn.  Keep an ownership index so retirement is O(keys for
        # this request) instead of rebuilding the process-wide set each time.
        self._pressure_event_keys_by_request: dict[
            str, set[tuple[str, str, str, int]]
        ] = {}
        # D4 (2026-09-05): 深缺口(逐无可逐/结构不可行)累计计数——
        # ensure_physical_fit 两个失败出口前 +1 并落 deep_gap 事件;
        # 实证负载预期恒 0,>0 即批次安全边界被触及的观测信号。
        # B2: 维持 face 现有形态(int 计数 + KVCacheEvent),不搬 sh 的
        # 结构化 dict 台账(契约 §12-6)。
        self._deep_gap_events: int = 0
        self._strict_kv_invariants = (
            _strict_kv_invariants_from_environment()
            if strict_invariants is None
            else strict_invariants
        )
        self._check_invariants()
        self._initialize_incremental_invariants()
        # Metrics observation state (doc sec.7.4): resident KV is tracked as
        # per-session parts that carry their own token counts and layer
        # ranges (B2, sh ``:1267-1274``), so suffix-half evictions and
        # restores always remove the exact recorded distribution of an
        # earlier add under its own allocation key.
        self._metrics_recorder = _METRICS_RECORDER
        self._metrics_parts: dict[str, list[dict[str, Any]]] = {}
        self._metrics_segment_counters: dict[str, int] = {}
        if self._metrics_recorder is not None:
            for rank in sorted(self._rank_states):
                self._metrics_recorder.initialize_rank(
                    rank, self._rank_states[rank].capacity_bytes
                )
            for rank in sorted(self._rank_states):
                state = self._rank_states[rank]
                self._metrics_recorder.record(
                    planner_time_ns=0,
                    anchor_kind="tick_zero",
                    request_id=None,
                    session_id=None,
                    rank=rank,
                    allocation_key=f"weight:{rank}",
                    weight_delta_bytes=state.model_weight_bytes,
                    cause="model_weight_preload",
                )

    @property
    def events(self) -> tuple[KVCacheEvent, ...]:
        return tuple(self._events)

    # B1 增量读取 + 水位压缩(2026-08-28):生产消费方(在线调度器的
    # kv_actions 流)本就持有 _kv_events_emitted 游标,改走 events_since
    # 增量切片,消除每决策批 events property 的全量 tuple() 拷贝(O(n²)
    # CPU 残留);已消费前缀按水位整段删除(风格对齐 graph_batch_builder
    # 的 M1 压缩:≥8192 且不小于一半才删,均摊 O(1)/事件)。event_index
    # 由单调计数器分配,压缩不改变任何已发射事件的载荷。
    def events_since(self, index: int) -> list[KVCacheEvent]:
        """B1:列表位置 index 起的增量事件(消费方维护游标)。"""
        return self._events[index:]

    def compact_events(self, consumed: int) -> int:
        """B1:已消费水位 ≥8192 且不小于现存一半时整段删除前缀,返回实际
        删除条数(调用方游标同步减去该值;0 = 未压缩)。"""
        if (consumed >= _EVENTS_COMPACT_THRESHOLD
                and consumed * 2 >= len(self._events)):
            del self._events[:consumed]
            return consumed
        return 0

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    @property
    def deep_gap_events(self) -> int:
        """D4 (2026-09-05): 深缺口累计计数(passive 观测口径,预期恒 0)。"""
        return self._deep_gap_events

    def session_snapshot(self, session_id: str) -> Optional[SessionKVSnapshot]:
        state = self._sessions.get(session_id)
        return None if state is None else self._snapshot_of(state)

    def session_snapshots(self) -> tuple[SessionKVSnapshot, ...]:
        """B2 (sh ``session_snapshots :1789-1790``): all snapshots, sorted."""

        return tuple(
            self._snapshot_of(self._sessions[session_id])
            for session_id in self.session_ids
        )

    def _snapshot_of(self, state: SessionKVState) -> SessionKVSnapshot:
        """B2 (sh ``session_snapshot :1748-1787``): local/remote derivation."""
        local_shard_bytes = self._local_shards_for(state)
        remote_shard_bytes = kv_cache_shard_bytes_for_layer_range(
            self.model,
            state.logical_context_tokens,
            self.tp_degree,
            layer_start=state.resident_prefix_layers,
            layer_end=self.model.layers,
        )
        return SessionKVSnapshot(
            session_id=state.session_id,
            location=state.location,
            instance_index=state.instance_index,
            logical_context_tokens=state.logical_context_tokens,
            total_bytes=state.total_bytes,
            shard_bytes=state.shard_bytes,
            resident_prefix_layers=state.resident_prefix_layers,
            local_bytes=sum(local_shard_bytes),
            remote_bytes=sum(remote_shard_bytes),
            local_shard_bytes=local_shard_bytes,
            remote_shard_bytes=remote_shard_bytes,
            last_completion_ns=state.last_completion_ns,
            active=state.active,
            last_request_id=state.last_request_id,
            evicted_at_ns=state.evicted_at_ns,
            evicted_by_request_id=state.evicted_by_request_id,
        )

    def hbm_snapshots(self, instance_index: Optional[int] = None) -> tuple[NodeHBMSnapshot, ...]:
        if instance_index is None:
            ranks = tuple(sorted(self._rank_states))
        else:
            ranks = tuple(self.topology.instance(instance_index).ranks)
        return tuple(self._rank_states[rank].snapshot() for rank in ranks)

    def nearest_edge(self, rank: int) -> int:
        """B2: nearest remote-memory edge port for a rank."""

        return nearest_edge_rank(self.topology.hardware, rank, self.edge_ranks)

    def _remaining(self, instance_index: int) -> tuple[int, ...]:
        return tuple(
            self._rank_states[rank].remaining_bytes
            for rank in self.topology.instance(instance_index).ranks
        )

    def _insufficient(
        self,
        instance_index: int,
        required_shards: Sequence[int],
    ) -> tuple[int, ...]:
        instance = self.topology.instance(instance_index)
        if len(required_shards) != len(instance.ranks):
            raise ValueError("required KV shards must match the target TP degree")
        return tuple(
            rank
            for rank, remaining, required in zip(
                instance.ranks, self._remaining(instance_index), required_shards
            )
            if remaining < required
        )

    def _local_shards_for(self, state: SessionKVState) -> tuple[int, ...]:
        """Resident (local) shards = layer range [0, resident_prefix_layers)."""

        return kv_cache_shard_bytes_for_layer_range(
            self.model,
            state.logical_context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=state.resident_prefix_layers,
        )

    @staticmethod
    def _fifo_sort(candidates: list[SessionKVState]) -> list[SessionKVState]:
        """B2 (sh ``:2667-2675``): (last_completion_ns, session_id) 升序 = LRU."""

        candidates.sort(
            key=lambda session: (
                int(session.last_completion_ns),
                session.session_id,
            )
        )
        return candidates

    def _completed_full_candidates(
        self,
        instance_index: int,
    ) -> list[SessionKVState]:
        """Stage-1 pool: completed inactive LOCAL sessions on the instance.

        B2 去类型化(契约 §9):无 trigger_type 参数、无 _eviction_class。
        """

        if self.partial_resident_prefix_layers == self.model.layers:
            return []
        return self._fifo_sort([
            session
            for session in self._sessions.values()
            if session.location == LOCAL_HBM
            and session.instance_index == instance_index
            and not session.active
            and session.last_completion_ns is not None
        ])

    def _completed_resident_candidates(
        self,
        instance_index: int,
    ) -> list[SessionKVState]:
        """Stage-2 pool: any completed inactive locally resident session."""

        return self._fifo_sort([
            session
            for session in self._sessions.values()
            if session.location in {LOCAL_HBM, PARTIAL_HBM_REMOTE}
            and session.instance_index == instance_index
            and not session.active
            and session.last_completion_ns is not None
        ])

    def _event(
        self,
        *,
        now_ns: int,
        phase: str,
        event_type: str,
        reason: str,
        trigger_request_id: str,
        session_id: Optional[str] = None,
        source_instance_index: Optional[int] = None,
        target_instance_index: Optional[int] = None,
        context_tokens: int = 0,
        total_bytes: int = 0,
        shard_bytes: Sequence[int] = (),
        last_completion_ns: Optional[int] = None,
        before: Sequence[int] = (),
        after: Sequence[int] = (),
        insufficient_ranks: Sequence[int] = (),
    ) -> None:
        self._events.append(
            KVCacheEvent(
                event_index=self._event_index_next,
                planner_time_ns=now_ns,
                phase=phase,
                event_type=event_type,
                reason=reason,
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                source_instance_index=source_instance_index,
                target_instance_index=target_instance_index,
                context_tokens=context_tokens,
                total_bytes=total_bytes,
                shard_bytes=tuple(int(value) for value in shard_bytes),
                last_completion_ns=last_completion_ns,
                instance_remaining_before_bytes=tuple(int(value) for value in before),
                instance_remaining_after_bytes=tuple(int(value) for value in after),
                insufficient_ranks=tuple(int(rank) for rank in insufficient_ranks),
            )
        )
        self._event_index_next += 1

    def _pressure_event(
        self,
        *,
        now_ns: int,
        phase: str,
        event_type: str,
        reason: str,
        trigger_request_id: str,
        target_instance_index: int,
        before: Sequence[int],
        after: Sequence[int],
        insufficient_ranks: Sequence[int],
    ) -> None:
        """Write bounded audit events for a repeated admission pressure state."""

        key = (event_type, phase, trigger_request_id, target_instance_index)
        if key in self._pressure_event_keys:
            retry_key = ("admission_retry", phase, trigger_request_id, target_instance_index)
            if retry_key not in self._pressure_event_keys:
                self._remember_pressure_event_key(retry_key)
                self._event(
                    now_ns=now_ns,
                    phase=phase,
                    event_type="admission_retry",
                    reason="retry_after_previous_capacity_block",
                    trigger_request_id=trigger_request_id,
                    target_instance_index=target_instance_index,
                    before=before,
                    after=after,
                    insufficient_ranks=insufficient_ranks,
                )
            return
        self._remember_pressure_event_key(key)
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type=event_type,
            reason=reason,
            trigger_request_id=trigger_request_id,
            target_instance_index=target_instance_index,
            before=before,
            after=after,
            insufficient_ranks=insufficient_ranks,
        )

    def _remember_pressure_event_key(
        self, key: tuple[str, str, str, int]
    ) -> None:
        self._pressure_event_keys.add(key)
        self._pressure_event_keys_by_request.setdefault(key[2], set()).add(key)

    def _forget_pressure_event_keys(self, request_id: str) -> None:
        for key in self._pressure_event_keys_by_request.pop(request_id, ()):
            self._pressure_event_keys.discard(key)

    def _record_deep_gap(
        self,
        *,
        instance_index: int,
        now_ns: int,
        phase: str,
        reason: str,
        trigger_request_id: str,
        required: Sequence[int],
        insufficient_ranks: Sequence[int],
    ) -> None:
        """D4 (2026-09-05): 深缺口台账——计数 +1 并发 deep_gap KVCacheEvent。

        语义(§7.3):被动逐出的边界观测。候选耗尽分支 = admission_blocked
        优雅推迟(reason=exhausted_completed_candidates,去重压力事件旁
        无条件记录);结构不可行分支 = fail-closed raise 前
        (reason=request_exceeds_empty_instance)。事件携带触发时点的
        before/after remaining(两出口均无 KV 变更,故相等)与
        insufficient_ranks,落 kv_actions 流供 campaign 复核。
        """
        self._deep_gap_events += 1
        before = self._remaining(instance_index)
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type="deep_gap",
            reason=reason,
            trigger_request_id=trigger_request_id,
            target_instance_index=instance_index,
            total_bytes=sum(int(value) for value in required),
            shard_bytes=tuple(int(value) for value in required),
            before=before,
            after=before,
            insufficient_ranks=insufficient_ranks,
        )

    def _add_shards(self, instance_index: int, shards: Sequence[int]) -> None:
        instance = self.topology.instance(instance_index)
        if len(shards) != self.tp_degree or any(value < 0 for value in shards):
            raise ValueError("invalid KV shard vector")
        for rank, value in zip(instance.ranks, shards):
            self._rank_states[rank].resident_kv_bytes += int(value)

    def _remove_shards(self, instance_index: int, shards: Sequence[int]) -> None:
        instance = self.topology.instance(instance_index)
        if len(shards) != self.tp_degree or any(value < 0 for value in shards):
            raise ValueError("invalid KV shard vector")
        for rank, value in zip(instance.ranks, shards):
            state = self._rank_states[rank]
            if state.resident_kv_bytes < value:
                raise RuntimeError("attempted to release more KV than the rank owns")
            state.resident_kv_bytes -= int(value)

    # ------------------------------------------------------------------
    # Incremental invariant ledger.  The complete checker below remains the
    # source of truth for construction, explicit strict mode, and the terminal
    # audit.  Normal mutation paths update only the contribution(s) they
    # changed, which avoids rebuilding process-wide session/reservation totals
    # after every generated token.
    # ------------------------------------------------------------------

    def _initialize_incremental_invariants(self) -> None:
        self._invariant_expected_resident_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_expected_reserved_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_session_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._invariant_reservation_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._refresh_incremental_sessions(self._sessions)
        self._refresh_incremental_reservations(self._reservations)

    def _incremental_session_contribution(
        self, session: SessionKVState
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        # B2 (sh ``:1825-1846``): the rank ledger only ever sees the resident
        # prefix layer range of a LOCAL/PARTIAL session.
        if (
            session.location not in {LOCAL_HBM, PARTIAL_HBM_REMOTE}
            or session.instance_index is None
            or len(session.shard_bytes) != self.tp_degree
        ):
            return None
        if session.location == LOCAL_HBM:
            if session.resident_prefix_layers != self.model.layers:
                return None
        elif not 0 < session.resident_prefix_layers < self.model.layers:
            return None
        return session.instance_index, tuple(
            int(value) for value in self._local_shards_for(session)
        )

    def _incremental_reservation_contribution(
        self, reservation: RequestCapacityReservation
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        if len(reservation.shard_bytes) != self.tp_degree:
            return None
        return reservation.instance_index, tuple(
            int(value) for value in reservation.shard_bytes
        )

    def _apply_incremental_contribution(
        self,
        expected_by_rank: dict[int, int],
        contribution: Optional[tuple[int, tuple[int, ...]]],
        multiplier: int,
    ) -> set[int]:
        if contribution is None:
            return set()
        instance_index, shards = contribution
        instance = self.topology.instance(instance_index)
        affected_ranks = set(instance.ranks)
        for rank, value in zip(instance.ranks, shards):
            expected_by_rank[rank] += multiplier * value
        return affected_ranks

    def _refresh_incremental_sessions(self, session_ids: Iterable[str]) -> set[int]:
        affected_ranks: set[int] = set()
        for session_id in dict.fromkeys(session_ids):
            old = self._invariant_session_contributions.pop(session_id, None)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_resident_by_rank, old, -1
                )
            )
            session = self._sessions.get(session_id)
            if session is None:
                continue
            new = self._incremental_session_contribution(session)
            self._invariant_session_contributions[session_id] = new
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_resident_by_rank, new, 1
                )
            )
        return affected_ranks

    def _refresh_incremental_reservations(
        self, reservation_ids: Iterable[str]
    ) -> set[int]:
        affected_ranks: set[int] = set()
        for request_id in dict.fromkeys(reservation_ids):
            old = self._invariant_reservation_contributions.pop(request_id, None)
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_reserved_by_rank, old, -1
                )
            )
            reservation = self._reservations.get(request_id)
            if reservation is None:
                continue
            new = self._incremental_reservation_contribution(reservation)
            self._invariant_reservation_contributions[request_id] = new
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_reserved_by_rank, new, 1
                )
            )
        return affected_ranks

    def _check_incremental_session_invariants(self, session: SessionKVState) -> None:
        # B2 (sh ``:1931-1957``): three-state location whitelist, layer-domain
        # assertions, REMOTE without instance or resident layers.
        if session.location not in {
            LOCAL_HBM,
            PARTIAL_HBM_REMOTE,
            REMOTE_MEMORY,
        }:
            raise RuntimeError(f"unknown KV location: {session.location}")
        if sum(session.shard_bytes) != session.total_bytes:
            raise RuntimeError(
                f"KV shards do not preserve total for {session.session_id}"
            )
        if session.location == REMOTE_MEMORY:
            if session.instance_index is not None:
                raise RuntimeError("remote KV session retained a local instance")
            if session.resident_prefix_layers != 0:
                raise RuntimeError("remote KV session retained resident layers")
            return
        if session.location == LOCAL_HBM:
            if session.resident_prefix_layers != self.model.layers:
                raise RuntimeError("fully local KV session is missing layers")
        elif not 0 < session.resident_prefix_layers < self.model.layers:
            raise RuntimeError("partial KV session has an invalid prefix length")
        if session.instance_index is None:
            raise RuntimeError("local KV session has no instance")
        if len(session.shard_bytes) != self.tp_degree:
            raise RuntimeError("KV shard count does not match TP instance")

    def _check_incremental_reservation_invariants(
        self, reservation: RequestCapacityReservation
    ) -> None:
        if len(reservation.shard_bytes) != self.tp_degree:
            raise RuntimeError("reservation has invalid TP shard vector")

    def _check_invariants_after_mutation(
        self,
        *,
        session_ids: Iterable[str] = (),
        reservation_ids: Iterable[str] = (),
    ) -> None:
        """Validate only the state touched by a completed manager mutation."""

        changed_sessions = tuple(dict.fromkeys(session_ids))
        changed_reservations = tuple(dict.fromkeys(reservation_ids))
        affected_ranks = self._refresh_incremental_sessions(changed_sessions)
        affected_ranks.update(
            self._refresh_incremental_reservations(changed_reservations)
        )

        # Keep the first part of the complete checker local and fail closed
        # before any caller can observe the mutated state.
        for rank in sorted(affected_ranks):
            state = self._rank_states[rank]
            if state.resident_kv_bytes < 0 or state.reserved_request_bytes < 0:
                raise RuntimeError("negative HBM accounting")
            if state.used_bytes > state.capacity_bytes:
                raise RuntimeError(
                    f"HBM capacity exceeded on rank {state.rank}: "
                    f"used={state.used_bytes}, capacity={state.capacity_bytes}"
                )
        for session_id in changed_sessions:
            session = self._sessions.get(session_id)
            if session is not None:
                self._check_incremental_session_invariants(session)
        for request_id in changed_reservations:
            reservation = self._reservations.get(request_id)
            if reservation is not None:
                self._check_incremental_reservation_invariants(reservation)

        affected_instances = {
            self._rank_states[rank].instance_index for rank in affected_ranks
        }
        for instance_index in sorted(affected_instances):
            instance = self.topology.instance(instance_index)
            actual = tuple(
                self._rank_states[rank].resident_kv_bytes for rank in instance.ranks
            )
            expected = tuple(
                self._invariant_expected_resident_by_rank[rank]
                for rank in instance.ranks
            )
            if actual != expected:
                raise RuntimeError(
                    f"session/rank KV accounting mismatch for instance {instance.index}: "
                    f"states={actual}, sessions={expected}"
                )
            actual_reserved = tuple(
                self._rank_states[rank].reserved_request_bytes for rank in instance.ranks
            )
            expected_reserved = tuple(
                self._invariant_expected_reserved_by_rank[rank]
                for rank in instance.ranks
            )
            if actual_reserved != expected_reserved:
                raise RuntimeError(
                    f"reservation/rank accounting mismatch for instance {instance.index}: "
                    f"states={actual_reserved}, reservations={expected_reserved}"
                )

        if self._strict_kv_invariants:
            self._check_invariants()

    # ------------------------------------------------------------------
    # Read-only metrics observation helpers (doc sec.7).  Every method here
    # is a no-op unless a recorder was installed at construction time, and
    # none of them feeds back into manager decisions.  B2: resident KV is
    # tracked as per-session parts that carry their own token count and
    # layer range (sh ``:2151-2418``); because
    # kv_cache_shard_bytes_for_layer_range is exactly linear in tokens,
    # suffix-half evictions and restores remove precisely the distribution
    # an earlier add recorded under the same allocation key.
    # ------------------------------------------------------------------

    def _metrics_add_segment(
        self,
        session_id: str,
        instance_index: int,
        tokens: int,
        layer_start: int,
        layer_end: int,
        shards: Sequence[int],
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        counter = self._metrics_segment_counters.get(session_id, 0)
        self._metrics_segment_counters[session_id] = counter + 1
        segment_id = f"seg{counter}"
        self._metrics_parts.setdefault(session_id, []).append(
            {
                "segment_id": segment_id,
                "instance_index": instance_index,
                "tokens": int(tokens),
                "layer_start": int(layer_start),
                "layer_end": int(layer_end),
                "shards": tuple(int(value) for value in shards),
            }
        )
        instance = self.topology.instance(instance_index)
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            recorder.record(
                planner_time_ns=now_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                allocation_key=f"resident:{session_id}:{segment_id}",
                resident_kv_delta_bytes=int(value),
                cause=cause,
            )

    def _metrics_remove_part(
        self,
        session_id: str,
        part: dict[str, Any],
        shards: Sequence[int],
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        instance = self.topology.instance(part["instance_index"])
        key = f"resident:{session_id}:{part['segment_id']}"
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            recorder.record(
                planner_time_ns=now_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                allocation_key=key,
                resident_kv_delta_bytes=-int(value),
                cause=cause,
            )

    def _metrics_remove_session_parts(
        self,
        session_id: str,
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Full-session eviction: remove every resident part under its key."""

        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.pop(session_id, [])
        for part in parts:
            self._metrics_remove_part(
                session_id,
                part,
                part["shards"],
                now_ns=now_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                cause=cause,
            )

    def _metrics_suffix_evict_parts(
        self,
        session_id: str,
        suffix_start: int,
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Suffix-half eviction: clamp every part to its intersection with
        the retained prefix layers, removing the exact per-part suffix
        distribution (never a merged guess, doc sec.7.4/9.4)."""

        if self._metrics_recorder is None:
            return
        parts = self._metrics_parts.get(session_id, [])
        kept_parts: list[dict[str, Any]] = []
        for part in parts:
            new_layer_end = min(part["layer_end"], suffix_start)
            if new_layer_end <= part["layer_start"]:
                removed = part["shards"]
                new_shards = None
            else:
                new_shards = kv_cache_shard_bytes_for_layer_range(
                    self.model,
                    part["tokens"],
                    self.tp_degree,
                    layer_start=part["layer_start"],
                    layer_end=new_layer_end,
                )
                removed = tuple(
                    old - new for old, new in zip(part["shards"], new_shards)
                )
            if any(removed):
                self._metrics_remove_part(
                    session_id,
                    part,
                    removed,
                    now_ns=now_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    cause=cause,
                )
            if new_shards is not None:
                part["layer_end"] = new_layer_end
                part["shards"] = new_shards
                kept_parts.append(part)
        if kept_parts:
            self._metrics_parts[session_id] = kept_parts
        else:
            self._metrics_parts.pop(session_id, None)

    def _metrics_move_session_parts(
        self,
        session_id: str,
        target_instance_index: int,
        tokens: int,
        resident_layers: int,
        total_shards: Sequence[int],
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Mirror a NoC migration: the target add is recorded as one
        consolidated segment (a fresh allocation key carries the full
        magnitude, so the chiplet projection stays exactly removable), then
        each original part is removed from the source ranks under its own
        key (doc sec.7.4/7.6).  Target add precedes source release, matching
        the dynamic GraphBatch transfer dependency ordering."""

        if self._metrics_recorder is None:
            return
        self._metrics_add_segment(
            session_id,
            target_instance_index,
            tokens,
            0,
            resident_layers,
            total_shards,
            now_ns=now_ns,
            anchor_kind=anchor_kind,
            request_id=request_id,
            cause=f"{cause}_target_add",
        )
        parts = self._metrics_parts.get(session_id, [])
        consolidated = parts[-1]
        for part in parts[:-1]:
            self._metrics_remove_part(
                session_id,
                part,
                part["shards"],
                now_ns=now_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                cause=f"{cause}_source_remove",
            )
        self._metrics_parts[session_id] = [consolidated]

    # ------------------------------------------------------------------
    # B2 transfer constructors (contract §4/§5; sh ``:2420-2665``).
    # ------------------------------------------------------------------

    def _xy_path(self, source: int, target: int) -> tuple[int, ...]:
        """XY route via face's existing ``_xy_route`` (no dual implementation;
        contract §9).  Lazy import: generate_face_trace imports this module."""

        from generate_face_trace import _xy_route

        return tuple(_xy_route(self.topology.hardware, source, target))

    def _local_hit_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
    ) -> KVTransfer:
        return KVTransfer(
            kind="local_hit",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=session.instance_index,
            target_instance_index=session.instance_index,
            total_bytes=session.total_bytes,
            shards=(),
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=self.model.layers,
            resident_prefix_layers_before=self.model.layers,
            resident_prefix_layers_after=self.model.layers,
        )

    def _noc_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        source_instance_index: int,
        target_instance_index: int,
    ) -> KVTransfer:
        if session.resident_prefix_layers != self.model.layers:
            raise RuntimeError("NoC migration requires a fully resident session")
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=shard_bytes,
                noc_path=self._xy_path(source_rank, target_rank),
                layer_start=0,
                layer_end=self.model.layers,
            )
            for source_rank, target_rank, shard_bytes in zip(
                source.ranks, target.ranks, session.shard_bytes
            )
            if shard_bytes > 0
        )
        return KVTransfer(
            kind="noc_migrate",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=target_instance_index,
            total_bytes=session.total_bytes,
            shards=shards,
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=self.model.layers,
            resident_prefix_layers_before=self.model.layers,
            resident_prefix_layers_after=self.model.layers,
        )

    def _partial_prefix_noc_transfer(
        self,
        *,
        session: SessionKVState,
        trigger_request_id: str,
        source_instance_index: int,
        target_instance_index: int,
    ) -> KVTransfer:
        """B2 (sh ``:2493-2554``): history action for a partial resident
        prefix — transfers only the locally resident prefix and leaves the
        suffix to the canonical remote-load segment."""

        resident_layers = session.resident_prefix_layers
        if not 0 < resident_layers < self.model.layers:
            raise RuntimeError("partial-prefix migration requires partial KV")
        if source_instance_index == target_instance_index:
            raise ValueError("partial-prefix migration requires distinct instances")
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        prefix_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.logical_context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=resident_layers,
        )
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=shard_bytes,
                noc_path=self._xy_path(source_rank, target_rank),
                layer_start=0,
                layer_end=resident_layers,
            )
            for source_rank, target_rank, shard_bytes in zip(
                source.ranks, target.ranks, prefix_shards
            )
            if shard_bytes > 0
        )
        return KVTransfer(
            kind="noc_migrate",
            phase="history",
            reason="history_partial_prefix_migrate",
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=target_instance_index,
            total_bytes=sum(prefix_shards),
            shards=shards,
            model_layers=self.model.layers,
            layer_start=0,
            layer_end=resident_layers,
            resident_prefix_layers_before=resident_layers,
            resident_prefix_layers_after=resident_layers,
        )

    def _remote_load_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        target_instance_index: int,
        layer_start: int,
        layer_end: int,
    ) -> KVTransfer:
        target = self.topology.instance(target_instance_index)
        transfer_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.logical_context_tokens,
            self.tp_degree,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        shards = []
        for target_rank, shard_bytes in zip(target.ranks, transfer_shards):
            if shard_bytes == 0:
                continue
            edge = self.nearest_edge(target_rank)
            shards.append(
                KVTransferShard(
                    source_rank=edge,
                    target_rank=target_rank,
                    edge_rank=edge,
                    bytes=shard_bytes,
                    noc_path=self._xy_path(edge, target_rank),
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
            )
        return KVTransfer(
            kind="remote_load",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=None,
            target_instance_index=target_instance_index,
            total_bytes=sum(transfer_shards),
            shards=tuple(shards),
            model_layers=self.model.layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=layer_start,
            resident_prefix_layers_after=layer_end,
        )

    def _remote_store_transfer(
        self,
        *,
        phase: str,
        reason: str,
        session: SessionKVState,
        trigger_request_id: str,
        layer_start: int,
        layer_end: int,
        resident_prefix_layers_after: int,
    ) -> KVTransfer:
        if session.instance_index is None:
            raise RuntimeError("cannot store a non-local KV session")
        source_instance_index = session.instance_index
        source = self.topology.instance(source_instance_index)
        transfer_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.logical_context_tokens,
            self.tp_degree,
            layer_start=layer_start,
            layer_end=layer_end,
        )
        shards = []
        for source_rank, shard_bytes in zip(source.ranks, transfer_shards):
            if shard_bytes == 0:
                continue
            edge = self.nearest_edge(source_rank)
            shards.append(
                KVTransferShard(
                    source_rank=source_rank,
                    target_rank=edge,
                    edge_rank=edge,
                    bytes=shard_bytes,
                    noc_path=self._xy_path(source_rank, edge),
                    layer_start=layer_start,
                    layer_end=layer_end,
                )
            )
        return KVTransfer(
            kind="remote_store",
            phase=phase,
            reason=reason,
            session_id=session.session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance_index,
            target_instance_index=None,
            total_bytes=sum(transfer_shards),
            shards=tuple(shards),
            model_layers=self.model.layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=session.resident_prefix_layers,
            resident_prefix_layers_after=resident_prefix_layers_after,
        )

    # ------------------------------------------------------------------
    # B2 tiered eviction (contract §9; sh ``_evict_suffix :2725-2772`` /
    # ``_evict_session :2774-2819``, adapted with face's KVCacheEvent audit
    # stream and EvictionRecord currency).
    # ------------------------------------------------------------------

    def _evict_suffix(
        self,
        victim: SessionKVState,
        *,
        now_ns: int,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> EvictionRecord:
        if victim.active or victim.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if victim.location != LOCAL_HBM or victim.instance_index is None:
            raise RuntimeError("suffix eviction requires a fully local session")
        suffix_start = self.partial_resident_prefix_layers
        if suffix_start >= self.model.layers:
            raise RuntimeError("configured model has no non-empty half-layer suffix")
        instance_index = victim.instance_index
        suffix_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            victim.logical_context_tokens,
            self.tp_degree,
            layer_start=suffix_start,
            layer_end=self.model.layers,
        )
        before = self._remaining(instance_index)
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_suffix_half",
            session=victim,
            trigger_request_id=trigger_request_id,
            layer_start=suffix_start,
            layer_end=self.model.layers,
            resident_prefix_layers_after=suffix_start,
        )
        self._remove_shards(instance_index, suffix_shards)
        victim.location = PARTIAL_HBM_REMOTE
        victim.resident_prefix_layers = suffix_start
        self._check_invariants_after_mutation(session_ids=(victim.session_id,))
        self._metrics_suffix_evict_parts(
            victim.session_id,
            suffix_start,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=(
                f"evict_{reason}_suffix_half:"
                f"layers{suffix_start}-{self.model.layers}"
            ),
        )
        after = self._remaining(instance_index)
        record = EvictionRecord(
            time_ns=now_ns,
            phase=phase,
            reason=reason,
            trigger_request_id=trigger_request_id,
            victim_session_id=victim.session_id,
            victim_instance_index=instance_index,
            victim_last_completion_ns=victim.last_completion_ns,
            context_tokens=victim.logical_context_tokens,
            shard_bytes=suffix_shards,
            transfer=transfer,
        )
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type="evict_suffix",
            reason=(
                f"evict_{reason}_suffix_half:"
                f"layers{suffix_start}-{self.model.layers}"
            ),
            trigger_request_id=trigger_request_id,
            session_id=victim.session_id,
            source_instance_index=instance_index,
            target_instance_index=None,
            context_tokens=victim.logical_context_tokens,
            total_bytes=sum(suffix_shards),
            shard_bytes=suffix_shards,
            last_completion_ns=victim.last_completion_ns,
            before=before,
            after=after,
        )
        return record

    def _evict_session(
        self,
        victim: SessionKVState,
        *,
        now_ns: int,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> EvictionRecord:
        if victim.active or victim.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if (
            victim.location not in {LOCAL_HBM, PARTIAL_HBM_REMOTE}
            or victim.instance_index is None
        ):
            raise RuntimeError("only locally resident sessions may be evicted")
        instance_index = victim.instance_index
        resident_layers = victim.resident_prefix_layers
        if resident_layers <= 0:
            raise RuntimeError("session has no local layers left to evict")
        local_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            victim.logical_context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=resident_layers,
        )
        before = self._remaining(instance_index)
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_full_fallback",
            session=victim,
            trigger_request_id=trigger_request_id,
            layer_start=0,
            layer_end=resident_layers,
            resident_prefix_layers_after=0,
        )
        self._remove_shards(instance_index, local_shards)
        victim.location = REMOTE_MEMORY
        victim.instance_index = None
        victim.resident_prefix_layers = 0
        victim.evicted_at_ns = now_ns
        victim.evicted_by_request_id = trigger_request_id
        self._check_invariants_after_mutation(session_ids=(victim.session_id,))
        self._metrics_remove_session_parts(
            victim.session_id,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=f"evict_{reason}_full_fallback:layers0-{resident_layers}",
        )
        after = self._remaining(instance_index)
        record = EvictionRecord(
            time_ns=now_ns,
            phase=phase,
            reason=reason,
            trigger_request_id=trigger_request_id,
            victim_session_id=victim.session_id,
            victim_instance_index=instance_index,
            victim_last_completion_ns=victim.last_completion_ns,
            context_tokens=victim.logical_context_tokens,
            shard_bytes=local_shards,
            transfer=transfer,
        )
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type="evict_full",
            reason=f"evict_{reason}_full_fallback:layers0-{resident_layers}",
            trigger_request_id=trigger_request_id,
            session_id=victim.session_id,
            source_instance_index=instance_index,
            target_instance_index=None,
            context_tokens=victim.logical_context_tokens,
            total_bytes=sum(local_shards),
            shard_bytes=local_shards,
            last_completion_ns=victim.last_completion_ns,
            before=before,
            after=after,
        )
        return record

    def ensure_physical_fit(
        self,
        instance_index: int,
        required_shards: Sequence[int],
        now_ns: int,
        trigger_request_id: str,
        protected_sessions: Iterable[str] = (),
        *,
        phase: str = "admission",
        reason: str = "request_physical_fit",
    ) -> CapacityResult:
        """B2 de-typed two-stage tiered reclamation (contract §9).

        Stage 1 suffix-halves completed inactive LOCAL sessions oldest-first
        (LRU); only when every candidate is halved does stage 2 fully
        offload resident sessions (the sole fallback).  The shortfall is
        rechecked after every single eviction.  Exhaustion keeps the face
        failure semantics: deep-gap ledger + ``admission_blocked`` graceful
        deferral (sh's raise-on-exhaustion is deliberately NOT imported);
        only a request that cannot fit an empty instance raises.
        """

        required = tuple(int(value) for value in required_shards)
        instance = self.topology.instance(instance_index)
        if len(required) != self.tp_degree or any(value < 0 for value in required):
            raise ValueError("required KV shard vector is invalid")
        maximum = tuple(
            self._rank_states[rank].capacity_bytes - self._rank_states[rank].model_weight_bytes
            for rank in instance.ranks
        )
        impossible = tuple(
            rank for rank, needed, available in zip(instance.ranks, required, maximum)
            if needed > available
        )
        if impossible:
            # D4 (2026-09-05): 结构不可行出口——fail-closed raise 前记
            # 深缺口(空实例都装不下,被动逐出无解)。
            self._record_deep_gap(
                instance_index=instance_index,
                now_ns=now_ns,
                phase=phase,
                reason="request_exceeds_empty_instance",
                trigger_request_id=trigger_request_id,
                required=required,
                insufficient_ranks=impossible,
            )
            details = ", ".join(
                f"rank={rank}, required={needed}, empty_available={available}"
                for rank, needed, available in zip(instance.ranks, required, maximum)
                if rank in impossible
            )
            raise ValueError(f"request cannot fit an empty instance: {details}")
        evictions: list[EvictionRecord] = []
        protected = set(protected_sessions)
        insufficient = self._insufficient(instance_index, required)

        # Stage 1: oldest-first, move only each eligible session's latter half.
        # A suffix eviction affects only its selected victim, so the remaining
        # stage candidates retain both eligibility and deterministic LRU
        # order; the snapshot is taken once (sh ``:2873-2885``).
        if insufficient:
            suffix_candidates = [
                session
                for session in self._completed_full_candidates(instance_index)
                if session.session_id not in protected
            ]
            suffix_candidate_index = 0
            while insufficient:
                if suffix_candidate_index >= len(suffix_candidates):
                    break
                victim = suffix_candidates[suffix_candidate_index]
                suffix_candidate_index += 1
                # I1 (2026-09-05): 被动逐出受害者必须是已完成 inactive 的
                # 本实例驻留会话(镜像逐出守卫与候选过滤,捕获候选快照与
                # 执行间的状态漂移)。
                if (
                    victim.location != LOCAL_HBM
                    or victim.instance_index != instance_index
                    or victim.active
                    or victim.last_completion_ns is None
                ):
                    raise RuntimeError(
                        "passive suffix-eviction victim must be a completed "
                        f"inactive local resident of instance {instance_index}: "
                        f"{victim.session_id}"
                    )
                evictions.append(
                    self._evict_suffix(
                        victim,
                        now_ns=now_ns,
                        phase=phase,
                        reason=reason,
                        trigger_request_id=trigger_request_id,
                    )
                )
                insufficient = self._insufficient(instance_index, required)

        # Stage 2: only after every inactive full session has been halved,
        # evict complete resident sessions (the remaining prefix for partial
        # sessions) in the same deterministic LRU order; rebuilt because the
        # newly partial sessions become eligible here (sh ``:2902-2933``).
        if insufficient:
            resident_candidates = [
                session
                for session in self._completed_resident_candidates(instance_index)
                if session.session_id not in protected
            ]
            resident_candidate_index = 0
            while insufficient:
                if resident_candidate_index >= len(resident_candidates):
                    break
                victim = resident_candidates[resident_candidate_index]
                resident_candidate_index += 1
                if (
                    victim.location not in {LOCAL_HBM, PARTIAL_HBM_REMOTE}
                    or victim.instance_index != instance_index
                    or victim.active
                    or victim.last_completion_ns is None
                ):
                    raise RuntimeError(
                        "passive full-eviction victim must be a completed "
                        f"inactive resident of instance {instance_index}: "
                        f"{victim.session_id}"
                    )
                evictions.append(
                    self._evict_session(
                        victim,
                        now_ns=now_ns,
                        phase=phase,
                        reason=reason,
                        trigger_request_id=trigger_request_id,
                    )
                )
                insufficient = self._insufficient(instance_index, required)

        if insufficient:
            before = self._remaining(instance_index)
            # D4 (2026-09-05): 候选耗尽出口——优雅推迟(admission_
            # blocked)前记深缺口(两阶段逐光全部已完成会话仍不够)。
            self._record_deep_gap(
                instance_index=instance_index,
                now_ns=now_ns,
                phase=phase,
                reason="exhausted_completed_candidates",
                trigger_request_id=trigger_request_id,
                required=required,
                insufficient_ranks=insufficient,
            )
            self._pressure_event(
                now_ns=now_ns,
                phase=phase,
                event_type="admission_blocked",
                reason=reason,
                trigger_request_id=trigger_request_id,
                target_instance_index=instance_index,
                before=before,
                after=before,
                insufficient_ranks=insufficient,
            )
            return CapacityResult(tuple(evictions), False, insufficient)

        # D4-I1 复核 (sh ``:2973-2981``): 返回前逐笔复查被逐会话均已完成
        # 且 inactive(入口守卫之外的循环间状态漂移捕获)。
        for record in evictions:
            evicted = self._sessions.get(record.victim_session_id)
            if evicted is None or evicted.active or evicted.last_completion_ns is None:
                raise RuntimeError(
                    "passive eviction victim is active or uncompleted: "
                    f"{record.victim_session_id}")
        # D4-I3 过度逐出守卫 (sh ``:2982-3009``,face 现版无、B2 新增):
        # 撤销最后一笔逐出后至少一个受影响 rank 必须回到不满足,否则该笔
        # 逐出超出"恰好够"边界。
        if evictions:
            last_eviction = evictions[-1]
            last_transfer = last_eviction.transfer
            if last_transfer is not None:
                freed_by_rank: dict[int, int] = {}
                for shard in last_transfer.shards:
                    if shard.source_rank is None:
                        continue
                    freed_by_rank[shard.source_rank] = (
                        freed_by_rank.get(shard.source_rank, 0) + shard.bytes
                    )
                if freed_by_rank:
                    remaining_now = dict(
                        zip(
                            self.topology.instance(instance_index).ranks,
                            self._remaining(instance_index),
                        )
                    )
                    if all(
                        remaining_now[rank] - freed_bytes
                        >= required[self._rank_relative_indexes[rank]]
                        for rank, freed_bytes in freed_by_rank.items()
                    ):
                        raise RuntimeError(
                            "passive eviction over-evicted: last eviction of "
                            f"{last_eviction.victim_session_id} freed {freed_by_rank} "
                            "bytes beyond the admission shortfall (instance "
                            f"{instance_index}, phase {phase}, reason {reason}, "
                            f"trigger {trigger_request_id})")
        return CapacityResult(tuple(evictions), True, ())

    def reserve_request_capacity(
        self,
        request_id: str,
        session_id: str,
        instance_index: int,
        required_shards: Sequence[int],
        now_ns: int,
        *,
        phase: str = "admission",
        reason: str = "request_reservation",
    ) -> CapacityResult:
        """Reserve a target's final KV footprint without placing KV there."""

        if request_id in self._reservations:
            raise RuntimeError(f"duplicate KV reservation for request {request_id}")
        required = tuple(int(value) for value in required_shards)
        fit = self.ensure_physical_fit(
            instance_index,
            required,
            now_ns,
            request_id,
            (session_id,),
            phase=phase,
            reason=reason,
        )
        evictions = fit.evictions
        if not fit.admitted:
            return CapacityResult(evictions, False, fit.insufficient_ranks)
        for rank, value in zip(self.topology.instance(instance_index).ranks, required):
            self._rank_states[rank].reserved_request_bytes += value
        self._reservations[request_id] = RequestCapacityReservation(
            request_id=request_id,
            session_id=session_id,
            instance_index=instance_index,
            shard_bytes=required,
        )
        self._check_invariants_after_mutation(reservation_ids=(request_id,))
        if self._metrics_recorder is not None:
            for rank, value in zip(
                self.topology.instance(instance_index).ranks, required
            ):
                if not value:
                    continue
                self._metrics_recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind=_metrics_anchor_for_phase(phase),
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=int(value),
                    cause=reason,
                )
        return CapacityResult(evictions, True, ())

    def release_request_capacity(self, request_id: str) -> RequestCapacityReservation:
        reservation = self._reservations.pop(request_id)
        for rank, value in zip(
            self.topology.instance(reservation.instance_index).ranks,
            reservation.shard_bytes,
        ):
            state = self._rank_states[rank]
            if state.reserved_request_bytes < value:
                raise RuntimeError("request reservation accounting underflow")
            state.reserved_request_bytes -= value
        self._check_invariants_after_mutation(reservation_ids=(request_id,))
        if self._metrics_recorder is not None:
            # This API carries no planner timestamp; reuse the most recent
            # event time so the replay stream stays non-decreasing.
            now_ns = self._events[-1].planner_time_ns if self._events else 0
            for rank, value in zip(
                self.topology.instance(reservation.instance_index).ranks,
                reservation.shard_bytes,
            ):
                if not value:
                    continue
                self._metrics_recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind="prefill_start",
                    request_id=request_id,
                    session_id=reservation.session_id,
                    rank=rank,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=-int(value),
                    cause="reservation_release",
                )
        return reservation

    def _history_shards(self, tokens: int) -> tuple[int, ...]:
        return kv_cache_shard_bytes_for_tokens(self.model, tokens, self.tp_degree)

    def prepare_history(
        self,
        session_id: str,
        target_instance_index: int,
        history_tokens: int,
        now_ns: int,
        trigger_request_id: str,
        *,
        required_context_tokens: Optional[int] = None,
        phase: str = "history",
    ) -> HistoryDecision:
        """B2 six-branch history preparation (contract §1.4; RECOMPUTE gone).

        无历史 -> NO_HISTORY;LOCAL 同实例 -> LOCAL_HIT;LOCAL 跨实例 ->
        NOC_MIGRATE;PARTIAL 同实例 -> REMOTE_SUFFIX_RESTORE(后缀回迁);
        PARTIAL 跨实例 -> PARTIAL_REMOTE_MIGRATE(前缀 NoC 迁移 + 后缀远端
        恢复,两段合一条决策);REMOTE -> REMOTE_RESTORE(全量回迁)。
        容量收敛先于恢复(先 ensure_physical_fit 再放置),恢复容量检查
        本身可触发对其他会话的两段式逐出。
        """

        _require_nonnegative_int(history_tokens, "history_tokens")
        if required_context_tokens is None:
            required_context_tokens = history_tokens
        _require_nonnegative_int(required_context_tokens, "required_context_tokens")
        if required_context_tokens < history_tokens:
            raise ValueError("required context cannot be smaller than history")
        state = self._sessions.get(session_id)
        if state is None and history_tokens != 0:
            raise ValueError(f"new session {session_id} cannot have historical KV")
        if state is not None and state.logical_context_tokens != history_tokens:
            raise ValueError(
                f"session {session_id} has logical context {state.logical_context_tokens}, "
                f"not requested history {history_tokens}"
            )
        if state is not None and state.active:
            raise RuntimeError(f"session {session_id} received an overlapping request")

        desired = self._history_shards(required_context_tokens)
        location_before = None if state is None else state.location
        prefix_layers_before = (
            None if state is None else state.resident_prefix_layers
        )

        def blocked_decision(
            action: str,
            source_instance_index: Optional[int],
            fit: CapacityResult,
        ) -> HistoryDecision:
            return HistoryDecision(
                action=action,
                source_instance_index=source_instance_index,
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfer_shards=(),
                recompute_tokens=0,
                evictions=fit.evictions,
                admission_blocked=True,
                insufficient_ranks=fit.insufficient_ranks,
                location_before=location_before,
                resident_prefix_layers_before=prefix_layers_before,
            )

        if state is None:
            # 无历史:建片上记录(容量门 = 全量 desired)。
            fit = self.ensure_physical_fit(
                target_instance_index,
                desired,
                now_ns,
                trigger_request_id,
                (),
                phase=phase,
                reason="history_and_prefill_admission",
            )
            if not fit.admitted:
                return blocked_decision(NO_HISTORY, None, fit)
            state = SessionKVState(
                session_id=session_id,
                location=LOCAL_HBM,
                instance_index=target_instance_index,
                logical_context_tokens=0,
                total_bytes=0,
                shard_bytes=tuple(0 for _ in desired),
                resident_prefix_layers=self.model.layers,
                active=True,
                last_request_id=trigger_request_id,
            )
            self._sessions[session_id] = state
            self._check_invariants_after_mutation(session_ids=(session_id,))
            before = self._remaining(target_instance_index)
            self._event(
                now_ns=now_ns,
                phase=phase,
                event_type="no_history",
                reason="window_first_request",
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                target_instance_index=target_instance_index,
                before=before,
                after=before,
            )
            return HistoryDecision(
                action=NO_HISTORY,
                source_instance_index=None,
                target_instance_index=target_instance_index,
                history_tokens=0,
                transfer_shards=(),
                recompute_tokens=0,
                evictions=fit.evictions,
                location_before=None,
                resident_prefix_layers_before=None,
            )

        if state.location == LOCAL_HBM:
            if state.instance_index is None:
                raise RuntimeError("resident session is missing its instance")
            source_instance = state.instance_index
            history_shards = self._history_shards(history_tokens)
            existing_target = (
                state.shard_bytes
                if source_instance == target_instance_index
                else tuple(0 for _ in desired)
            )
            needed = tuple(
                want - current for want, current in zip(desired, existing_target)
            )
            if any(value < 0 for value in needed):
                raise RuntimeError("session KV would shrink during history preparation")
            fit = self.ensure_physical_fit(
                target_instance_index,
                needed,
                now_ns,
                trigger_request_id,
                (session_id,),
                phase=phase,
                reason="history_and_prefill_admission",
            )
            if not fit.admitted:
                if source_instance == target_instance_index:
                    return blocked_decision(LOCAL_HIT, source_instance, fit)
                return blocked_decision(NOC_MIGRATE, source_instance, fit)

            state.active = True
            state.last_request_id = trigger_request_id
            if source_instance == target_instance_index:
                # LOCAL 同实例:零开销复用。
                self._check_invariants_after_mutation(session_ids=(session_id,))
                transfer = self._local_hit_transfer(
                    phase=phase,
                    reason="history_local_reuse",
                    session=state,
                    trigger_request_id=trigger_request_id,
                )
                before = self._remaining(target_instance_index)
                self._event(
                    now_ns=now_ns,
                    phase=phase,
                    event_type="local_hit",
                    reason="history_local_reuse",
                    trigger_request_id=trigger_request_id,
                    session_id=session_id,
                    source_instance_index=source_instance,
                    target_instance_index=target_instance_index,
                    context_tokens=history_tokens,
                    total_bytes=sum(history_shards),
                    shard_bytes=history_shards,
                    last_completion_ns=state.last_completion_ns,
                    before=before,
                    after=before,
                )
                return HistoryDecision(
                    action=LOCAL_HIT,
                    source_instance_index=source_instance,
                    target_instance_index=target_instance_index,
                    history_tokens=history_tokens,
                    transfer_shards=(),
                    recompute_tokens=0,
                    evictions=fit.evictions,
                    transfers=(transfer,),
                    location_before=location_before,
                    resident_prefix_layers_before=prefix_layers_before,
                )

            # LOCAL 跨实例:整份 NoC 迁移(face 既有 1000 类路径,语义保留)。
            transfer = self._noc_transfer(
                phase=phase,
                reason="history_other_instance",
                session=state,
                trigger_request_id=trigger_request_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
            )
            before = self._remaining(target_instance_index)
            # The dynamic transfer contract admits target shards before releasing
            # the source; planner state advances atomically after that admission.
            self._add_shards(target_instance_index, history_shards)
            self._remove_shards(source_instance, history_shards)
            state.instance_index = target_instance_index
            self._check_invariants_after_mutation(session_ids=(session_id,))
            self._metrics_move_session_parts(
                session_id,
                target_instance_index,
                history_tokens,
                state.resident_prefix_layers,
                history_shards,
                now_ns=now_ns,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_other_instance",
            )
            after = self._remaining(target_instance_index)
            self._event(
                now_ns=now_ns,
                phase=phase,
                event_type="noc_migrate",
                reason="history_other_instance",
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                context_tokens=history_tokens,
                total_bytes=sum(history_shards),
                shard_bytes=history_shards,
                last_completion_ns=state.last_completion_ns,
                before=before,
                after=after,
            )
            return HistoryDecision(
                action=NOC_MIGRATE,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfer_shards=transfer.shards,
                recompute_tokens=0,
                evictions=fit.evictions,
                transfers=(transfer,),
                location_before=location_before,
                resident_prefix_layers_before=prefix_layers_before,
            )

        if state.location == PARTIAL_HBM_REMOTE:
            if state.instance_index is None:
                raise RuntimeError("partial history has no source instance")
            source_instance = state.instance_index
            suffix_start = state.resident_prefix_layers
            prefix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                history_tokens,
                self.tp_degree,
                layer_start=0,
                layer_end=suffix_start,
            )
            suffix_shards = kv_cache_shard_bytes_for_layer_range(
                self.model,
                history_tokens,
                self.tp_degree,
                layer_start=suffix_start,
                layer_end=self.model.layers,
            )
            if source_instance == target_instance_index:
                # PARTIAL 同实例:只回迁后缀层段(容量门 = 后缀 + 增长)。
                existing_target = prefix_shards
                needed = tuple(
                    want - current for want, current in zip(desired, existing_target)
                )
                if any(value < 0 for value in needed):
                    raise RuntimeError(
                        "session KV would shrink during history preparation"
                    )
                fit = self.ensure_physical_fit(
                    target_instance_index,
                    needed,
                    now_ns,
                    trigger_request_id,
                    (session_id,),
                    phase=phase,
                    reason="history_and_prefill_admission",
                )
                if not fit.admitted:
                    return blocked_decision(
                        _REMOTE_SUFFIX_RESTORE, source_instance, fit
                    )
                transfer = self._remote_load_transfer(
                    phase=phase,
                    reason="history_remote_suffix_restore",
                    session=state,
                    trigger_request_id=trigger_request_id,
                    target_instance_index=target_instance_index,
                    layer_start=suffix_start,
                    layer_end=self.model.layers,
                )
                before = self._remaining(target_instance_index)
                self._add_shards(target_instance_index, suffix_shards)
                state.location = LOCAL_HBM
                state.resident_prefix_layers = self.model.layers
                state.active = True
                state.last_request_id = trigger_request_id
                self._check_invariants_after_mutation(session_ids=(session_id,))
                # Suffix restore gets its own segment id / allocation key (doc
                # sec.7.4/9.4): a later suffix eviction removes exactly this
                # distribution again.
                self._metrics_add_segment(
                    session_id,
                    target_instance_index,
                    history_tokens,
                    suffix_start,
                    self.model.layers,
                    suffix_shards,
                    now_ns=now_ns,
                    anchor_kind="transfer_complete",
                    request_id=trigger_request_id,
                    cause="history_remote_suffix_restore",
                )
                after = self._remaining(target_instance_index)
                self._event(
                    now_ns=now_ns,
                    phase=phase,
                    event_type="remote_load",
                    reason="history_remote_suffix_restore",
                    trigger_request_id=trigger_request_id,
                    session_id=session_id,
                    source_instance_index=source_instance,
                    target_instance_index=target_instance_index,
                    context_tokens=history_tokens,
                    total_bytes=sum(suffix_shards),
                    shard_bytes=suffix_shards,
                    last_completion_ns=state.last_completion_ns,
                    before=before,
                    after=after,
                )
                return HistoryDecision(
                    action=_REMOTE_SUFFIX_RESTORE,
                    source_instance_index=source_instance,
                    target_instance_index=target_instance_index,
                    history_tokens=history_tokens,
                    transfer_shards=(),
                    recompute_tokens=0,
                    evictions=fit.evictions,
                    transfers=(transfer,),
                    location_before=location_before,
                    resident_prefix_layers_before=prefix_layers_before,
                )

            # PARTIAL 跨实例:两段链——前缀 NoC 迁移 + 后缀远端恢复。
            # 两段承载于同一条决策(B2 契约:hbm_watermark 同请求同 kind
            # 两次即 fail,拆两条 prefill 记录被禁止)。
            fit = self.ensure_physical_fit(
                target_instance_index,
                desired,
                now_ns,
                trigger_request_id,
                (session_id,),
                phase=phase,
                reason="history_and_prefill_admission",
            )
            if not fit.admitted:
                return blocked_decision(_PARTIAL_REMOTE_MIGRATE, source_instance, fit)
            prefix_transfer = self._partial_prefix_noc_transfer(
                session=state,
                trigger_request_id=trigger_request_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
            )
            before = self._remaining(target_instance_index)
            self._remove_shards(source_instance, prefix_shards)
            self._add_shards(target_instance_index, prefix_shards)
            # The session remains partial while its remote suffix is restored,
            # but physical ownership now follows the target.
            state.instance_index = target_instance_index
            self._check_invariants_after_mutation(session_ids=(session_id,))
            self._metrics_move_session_parts(
                session_id,
                target_instance_index,
                history_tokens,
                suffix_start,
                prefix_shards,
                now_ns=now_ns,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_partial_prefix_migrate",
            )
            after_prefix = self._remaining(target_instance_index)
            self._event(
                now_ns=now_ns,
                phase=phase,
                event_type="noc_migrate",
                reason="history_partial_prefix_migrate",
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                context_tokens=history_tokens,
                total_bytes=sum(prefix_shards),
                shard_bytes=prefix_shards,
                last_completion_ns=state.last_completion_ns,
                before=before,
                after=after_prefix,
            )
            suffix_transfer = self._remote_load_transfer(
                phase=phase,
                reason="history_remote_suffix_restore",
                session=state,
                trigger_request_id=trigger_request_id,
                target_instance_index=target_instance_index,
                layer_start=suffix_start,
                layer_end=self.model.layers,
            )
            self._add_shards(target_instance_index, suffix_shards)
            state.location = LOCAL_HBM
            state.resident_prefix_layers = self.model.layers
            state.active = True
            state.last_request_id = trigger_request_id
            self._check_invariants_after_mutation(session_ids=(session_id,))
            self._metrics_add_segment(
                session_id,
                target_instance_index,
                history_tokens,
                suffix_start,
                self.model.layers,
                suffix_shards,
                now_ns=now_ns,
                anchor_kind="transfer_complete",
                request_id=trigger_request_id,
                cause="history_remote_suffix_restore",
            )
            after = self._remaining(target_instance_index)
            self._event(
                now_ns=now_ns,
                phase=phase,
                event_type="remote_load",
                reason="history_remote_suffix_restore",
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                context_tokens=history_tokens,
                total_bytes=sum(suffix_shards),
                shard_bytes=suffix_shards,
                last_completion_ns=state.last_completion_ns,
                before=after_prefix,
                after=after,
            )
            return HistoryDecision(
                action=_PARTIAL_REMOTE_MIGRATE,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfer_shards=prefix_transfer.shards,
                recompute_tokens=0,
                evictions=fit.evictions,
                transfers=(prefix_transfer, suffix_transfer),
                location_before=location_before,
                resident_prefix_layers_before=prefix_layers_before,
            )

        if state.location != REMOTE_MEMORY:
            raise RuntimeError(f"unknown history location: {state.location}")
        # REMOTE:全量回迁到选定实例。
        history_shards = self._history_shards(history_tokens)
        fit = self.ensure_physical_fit(
            target_instance_index,
            desired,
            now_ns,
            trigger_request_id,
            (session_id,),
            phase=phase,
            reason="history_and_prefill_admission",
        )
        if not fit.admitted:
            return blocked_decision(_REMOTE_RESTORE, None, fit)
        transfer = self._remote_load_transfer(
            phase=phase,
            reason="history_remote_restore",
            session=state,
            trigger_request_id=trigger_request_id,
            target_instance_index=target_instance_index,
            layer_start=0,
            layer_end=self.model.layers,
        )
        before = self._remaining(target_instance_index)
        self._add_shards(target_instance_index, history_shards)
        state.location = LOCAL_HBM
        state.instance_index = target_instance_index
        state.resident_prefix_layers = self.model.layers
        state.active = True
        state.last_request_id = trigger_request_id
        state.evicted_at_ns = None
        state.evicted_by_request_id = None
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_add_segment(
            session_id,
            target_instance_index,
            history_tokens,
            0,
            self.model.layers,
            history_shards,
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="history_remote_restore",
        )
        after = self._remaining(target_instance_index)
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type="remote_load",
            reason="history_remote_restore",
            trigger_request_id=trigger_request_id,
            session_id=session_id,
            source_instance_index=None,
            target_instance_index=target_instance_index,
            context_tokens=history_tokens,
            total_bytes=sum(history_shards),
            shard_bytes=history_shards,
            last_completion_ns=state.last_completion_ns,
            before=before,
            after=after,
        )
        return HistoryDecision(
            action=_REMOTE_RESTORE,
            source_instance_index=None,
            target_instance_index=target_instance_index,
            history_tokens=history_tokens,
            transfer_shards=(),
            recompute_tokens=0,
            evictions=fit.evictions,
            transfers=(transfer,),
            location_before=location_before,
            resident_prefix_layers_before=prefix_layers_before,
        )

    def _grow(
        self,
        *,
        session_id: str,
        instance_index: int,
        context_tokens: int,
        now_ns: int,
        trigger_request_id: str,
        phase: str,
        reason: str,
    ) -> CapacityResult:
        state = self._sessions[session_id]
        if (
            state.location != LOCAL_HBM
            or state.instance_index != instance_index
            or not state.active
        ):
            raise RuntimeError(f"session {session_id} is not active on target instance")
        desired = self._history_shards(context_tokens)
        delta = tuple(want - current for want, current in zip(desired, state.shard_bytes))
        if any(value < 0 for value in delta):
            raise ValueError("session KV context cannot shrink")
        fit = self.ensure_physical_fit(
            instance_index,
            delta,
            now_ns,
            trigger_request_id,
            (session_id,),
            phase=phase,
            reason=reason,
        )
        evictions = fit.evictions
        if not fit.admitted:
            return CapacityResult(evictions, False, fit.insufficient_ranks)
        self._add_shards(instance_index, delta)
        previous_tokens = state.logical_context_tokens
        state.shard_bytes = desired
        state.total_bytes = sum(desired)
        state.logical_context_tokens = context_tokens
        state.last_request_id = trigger_request_id
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_add_segment(
            session_id,
            instance_index,
            context_tokens - previous_tokens,
            0,
            self.model.layers,
            delta,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=reason,
        )
        return CapacityResult(evictions, True, ())

    def grow_prefill(
        self,
        session_id: str,
        prefill_context_tokens: int,
        now_ns: int,
        trigger_request_id: str,
    ) -> CapacityResult:
        state = self._sessions[session_id]
        if state.instance_index is None:
            raise RuntimeError("active prefill session has no instance")
        return self._grow(
            session_id=session_id,
            instance_index=state.instance_index,
            context_tokens=prefill_context_tokens,
            now_ns=now_ns,
            trigger_request_id=trigger_request_id,
            phase="prefill",
            reason="prefill_growth_capacity",
        )

    def move_prefill_to_decode(
        self,
        session_id: str,
        target_instance_index: int,
        now_ns: int,
        trigger_request_id: str,
        *,
        final_context_tokens: Optional[int] = None,
    ) -> MoveDecision:
        state = self._sessions[session_id]
        if state.location != LOCAL_HBM or state.instance_index is None or not state.active:
            raise RuntimeError("Prefill KV must be active and local before Decode")
        source_instance = state.instance_index
        desired_final = self._history_shards(
            state.logical_context_tokens if final_context_tokens is None else final_context_tokens
        )
        existing_target = (
            state.shard_bytes if source_instance == target_instance_index else tuple(0 for _ in desired_final)
        )
        required = tuple(want - current for want, current in zip(desired_final, existing_target))
        fit = self.ensure_physical_fit(
            target_instance_index,
            required,
            now_ns,
            trigger_request_id,
            (session_id,),
            phase="decode",
            reason="decode_target_capacity",
        )
        evictions = fit.evictions
        if not fit.admitted:
            placeholder = self._local_hit_transfer(
                phase="prefill_decode",
                reason="prefill_decode_local_reuse",
                session=state,
                trigger_request_id=trigger_request_id,
            )
            return MoveDecision(
                action=LOCAL_HIT if source_instance == target_instance_index else NOC_MIGRATE,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                transfer=placeholder,
                evictions=evictions,
                admission_blocked=True,
                insufficient_ranks=fit.insufficient_ranks,
            )
        if source_instance == target_instance_index:
            transfer = self._local_hit_transfer(
                phase="prefill_decode",
                reason="prefill_decode_local_reuse",
                session=state,
                trigger_request_id=trigger_request_id,
            )
            before = self._remaining(target_instance_index)
            self._event(
                now_ns=now_ns,
                phase="prefill_decode",
                event_type="local_hit",
                reason="prefill_decode_local_reuse",
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                context_tokens=state.logical_context_tokens,
                total_bytes=sum(state.shard_bytes),
                shard_bytes=state.shard_bytes,
                last_completion_ns=state.last_completion_ns,
                before=before,
                after=before,
            )
            return MoveDecision(LOCAL_HIT, source_instance, target_instance_index, transfer, evictions)

        transfer = self._noc_transfer(
            phase="prefill_decode",
            reason="prefill_decode_instance_migrate",
            session=state,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance,
            target_instance_index=target_instance_index,
        )
        before = self._remaining(target_instance_index)
        self._add_shards(target_instance_index, state.shard_bytes)
        self._remove_shards(source_instance, state.shard_bytes)
        state.instance_index = target_instance_index
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_move_session_parts(
            session_id,
            target_instance_index,
            state.logical_context_tokens,
            state.resident_prefix_layers,
            state.shard_bytes,
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="prefill_decode_instance_migrate",
        )
        after = self._remaining(target_instance_index)
        self._event(
            now_ns=now_ns,
            phase="prefill_decode",
            event_type="noc_migrate",
            reason="prefill_decode_instance_migrate",
            trigger_request_id=trigger_request_id,
            session_id=session_id,
            source_instance_index=source_instance,
            target_instance_index=target_instance_index,
            context_tokens=state.logical_context_tokens,
            total_bytes=transfer.total_bytes,
            shard_bytes=state.shard_bytes,
            last_completion_ns=state.last_completion_ns,
            before=before,
            after=after,
        )
        return MoveDecision(NOC_MIGRATE, source_instance, target_instance_index, transfer, evictions)

    def grow_decode(
        self,
        session_id: str,
        final_context_tokens: int,
        now_ns: int,
        trigger_request_id: str,
    ) -> CapacityResult:
        state = self._sessions[session_id]
        if state.instance_index is None:
            raise RuntimeError("active decode session has no instance")
        return self._grow(
            session_id=session_id,
            instance_index=state.instance_index,
            context_tokens=final_context_tokens,
            now_ns=now_ns,
            trigger_request_id=trigger_request_id,
            phase="decode",
            reason="decode_growth_capacity",
        )

    def mark_complete(
        self,
        session_id: str,
        completion_ns: int,
        request_id: str,
    ) -> tuple[EvictionRecord, ...]:
        """Completion boundary: retain KV (完成路径零逐出)。

        B2 去类型化(契约 §9):不引入 next_request_type——候选池排序键
        即 (last_completion_ns, session_id),无类型分层。
        """

        state = self._sessions[session_id]
        if state.location != LOCAL_HBM or state.instance_index is None or not state.active:
            raise RuntimeError("only active local sessions may complete")
        state.active = False
        state.last_completion_ns = completion_ns
        state.last_request_id = request_id
        self._check_invariants_after_mutation(session_ids=(session_id,))
        before = self._remaining(state.instance_index)
        self._event(
            now_ns=completion_ns,
            phase="completion",
            event_type="retain_complete",
            reason="request_completed_keep_kv",
            trigger_request_id=request_id,
            session_id=session_id,
            target_instance_index=state.instance_index,
            context_tokens=state.logical_context_tokens,
            total_bytes=sum(state.shard_bytes),
            shard_bytes=state.shard_bytes,
            last_completion_ns=completion_ns,
            before=before,
            after=before,
        )
        # The request can never retry admission after completion.  Retire its
        # bounded-dedup state even when the session has a later turn.
        self._forget_pressure_event_keys(request_id)
        return ()

    def retire_terminal_session(
        self,
        session_id: str,
        completion_ns: int,
        request_id: Optional[str] = None,
    ) -> Optional[int]:
        """Forget terminal KV while preserving the partial/remote state model.

        B2 (sh ``:3434-3510``): no eviction transfer is emitted and this is
        not counted as an eviction.  Only the prefix still physically
        resident in local HBM is decremented; remote-only bytes are silently
        written off with their terminal session metadata (远端账面静默核销,
        不发传输、不算逐出)。face 语义保留:不发 KVCacheEvent。
        """

        state = self._sessions.get(session_id)
        if state is None:
            raise KeyError(f"unknown KV session: {session_id}")
        if state.active or state.last_completion_ns is None:
            raise RuntimeError(
                "only inactive completed sessions may be terminally retired"
            )
        if (
            request_id is not None
            and state.last_request_id is not None
            and request_id != state.last_request_id
        ):
            raise RuntimeError(
                "terminal retirement request does not own the completed session"
            )
        terminal_request_id = request_id or state.last_request_id
        if terminal_request_id is None:
            raise RuntimeError("completed session has no terminal request ID")
        requested_reservation = self._reservations.get(terminal_request_id)
        if (
            requested_reservation is not None
            and requested_reservation.session_id != session_id
        ):
            raise RuntimeError(
                "terminal retirement request owns another session reservation"
            )

        # A session can own at most its own request reservations.  Release all
        # matching entries rather than retaining a terminal-accounting
        # tombstone; never touch another session's reservation.
        reservation_ids = sorted(
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if reservation.session_id == session_id
        )
        for reservation_id in reservation_ids:
            self.release_request_capacity(reservation_id)

        released_instance_index: Optional[int] = None
        if state.location in {LOCAL_HBM, PARTIAL_HBM_REMOTE}:
            if state.instance_index is None:
                raise RuntimeError("local completed session has no instance")
            released_instance_index = state.instance_index
            local_shards = self._local_shards_for(state)
            self._remove_shards(released_instance_index, local_shards)
        elif state.location != REMOTE_MEMORY:
            raise RuntimeError(f"unknown KV location: {state.location}")

        self._metrics_remove_session_parts(
            session_id,
            now_ns=completion_ns,
            anchor_kind="completion",
            request_id=terminal_request_id,
            cause="terminal_session_retire",
        )
        # Metrics helpers intentionally no-op without an observer; retirement
        # must nevertheless remove the per-session containers in both modes.
        self._metrics_parts.pop(session_id, None)
        self._metrics_segment_counters.pop(session_id, None)
        self._forget_pressure_event_keys(terminal_request_id)
        del self._sessions[session_id]
        self._check_invariants_after_mutation(
            session_ids=(session_id,), reservation_ids=reservation_ids
        )
        return released_instance_index

    def assert_final_state(self) -> None:
        self._check_invariants()
        if self._reservations:
            raise RuntimeError(
                f"planning ended with outstanding KV reservations: {sorted(self._reservations)}"
            )
        if self._pressure_event_keys or self._pressure_event_keys_by_request:
            raise RuntimeError("planning ended with stale KV pressure dedup state")
        active = [state.session_id for state in self._sessions.values() if state.active]
        if active:
            raise RuntimeError(f"planning ended with active KV sessions: {sorted(active)}")

    def _check_invariants(self) -> None:
        """B2 full audit (sh ``:2037-2108``): three-state location whitelist,
        layer-domain assertions, REMOTE without instance or resident layers,
        and per-rank expected-KV recomputation from the resident prefix."""

        for state in self._rank_states.values():
            if state.resident_kv_bytes < 0 or state.reserved_request_bytes < 0:
                raise RuntimeError("negative HBM accounting")
            if state.used_bytes > state.capacity_bytes:
                raise RuntimeError(
                    f"HBM capacity exceeded on rank {state.rank}: "
                    f"used={state.used_bytes}, capacity={state.capacity_bytes}"
                )
        accumulated: dict[int, list[int]] = {
            instance.index: [0] * self.tp_degree for instance in self.topology.instances
        }
        reserved: dict[int, list[int]] = {
            instance.index: [0] * self.tp_degree for instance in self.topology.instances
        }
        for session in self._sessions.values():
            if session.location not in {
                LOCAL_HBM,
                PARTIAL_HBM_REMOTE,
                REMOTE_MEMORY,
            }:
                raise RuntimeError(f"unknown KV location: {session.location}")
            if sum(session.shard_bytes) != session.total_bytes:
                raise RuntimeError(
                    f"KV shards do not preserve total for {session.session_id}"
                )
            if session.location == REMOTE_MEMORY:
                if session.instance_index is not None:
                    raise RuntimeError("remote KV session retained a local instance")
                if session.resident_prefix_layers != 0:
                    raise RuntimeError("remote KV session retained resident layers")
                continue
            if session.location == LOCAL_HBM:
                if session.resident_prefix_layers != self.model.layers:
                    raise RuntimeError("fully local KV session is missing layers")
            elif not 0 < session.resident_prefix_layers < self.model.layers:
                raise RuntimeError("partial KV session has an invalid prefix length")
            if session.instance_index is None:
                raise RuntimeError("local KV session has no instance")
            if len(session.shard_bytes) != self.tp_degree:
                raise RuntimeError("KV shard count does not match TP instance")
            local_shards = self._local_shards_for(session)
            accumulated[session.instance_index] = [
                total + value
                for total, value in zip(accumulated[session.instance_index], local_shards)
            ]
        for reservation in self._reservations.values():
            reserved[reservation.instance_index] = [
                total + value
                for total, value in zip(
                    reserved[reservation.instance_index], reservation.shard_bytes
                )
            ]
        for instance in self.topology.instances:
            actual = tuple(
                self._rank_states[rank].resident_kv_bytes for rank in instance.ranks
            )
            if actual != tuple(accumulated[instance.index]):
                raise RuntimeError(
                    f"session/rank KV accounting mismatch for instance {instance.index}: "
                    f"states={actual}, sessions={tuple(accumulated[instance.index])}"
                )
            actual_reserved = tuple(
                self._rank_states[rank].reserved_request_bytes for rank in instance.ranks
            )
            if actual_reserved != tuple(reserved[instance.index]):
                raise RuntimeError(
                    f"reservation/rank accounting mismatch for instance {instance.index}: "
                    f"states={actual_reserved}, reservations={tuple(reserved[instance.index])}"
                )
