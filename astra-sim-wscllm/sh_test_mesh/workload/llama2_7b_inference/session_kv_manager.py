"""Session-scoped local-HBM KV cache management.

This module deliberately models only the policy used by the three-minute
TraceLab workload: a complete session KV is either resident in one TP
instance or it has been deleted.  Deleted KV retains its logical context so a
later turn can recompute it.  There is no remote-memory tier, partial-layer
state, or capacity-driven remapping in this implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import hashlib
import json
import os
from typing import Any, Iterable, Optional, Sequence


RESIDENT = "RESIDENT"
EVICTED = "EVICTED"
NO_HISTORY = "NO_HISTORY"
LOCAL_HIT = "LOCAL_HIT"
NOC_MIGRATE = "NOC_MIGRATE"
RECOMPUTE = "RECOMPUTE"

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
#
# P1 权威 HBM delta journal(2026-08-30):recorder 处于 journal 模式
# (MemoryActionRecorder(journal_path=...))时,manager 每次公开 mutation 构成
# 一个 journal 事务(③):入口 begin,内部嵌套的公开复用(如准入前置的
# ensure_physical_fit)只加深深度并入同一事务;最外层正常返回 = commit 点,
# 在 _check_invariants_after_mutation 已逐内层通过之后执行 ④ 逐 rank 对账
# (manager 实际状态 vs journal before+Σdelta 推出的 after)。journal 是纯
# 观测旁路:不改任何准入/逐出/放置决策,不改变任何返回值。
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
        # WSC-LLM reserves the static final Decode KV at Prefill admission
        # (wsc_llm_scheduler try_admit_prefill); doc sec.7.8 anchors
        # admission/reservation actions at the request Prefill start.
        "prefill_admission": "prefill_start",
        "decode": "decode_start",
        "prefill_decode": "decode_start",
        "completion": "completion",
    }.get(phase, "stage_boundary")


def _journal_transaction(method: Any) -> Any:
    """③ 公开 mutation = journal 事务(原子校验边界)装饰器。

    语义(doc §6-P1 ③/④):
    - recorder 未装或非 journal 模式:透传,零开销零行为差(journal 是
      纯观测旁路,关闭时与改前逐字节一致)。
    - 最外层公开调用:入口向 recorder 申请单调 transaction_id(journal
      各行携带该 id);正常返回 = commit 点,此时方法体内的全部
      _check_invariants_after_mutation 已通过,随即执行 ④ 逐 rank 对账
      _journal_reconcile_transaction(manager 快照 vs journal 记账)。
    - 嵌套公开复用(retire_terminal_session 内部的 release_request_
      capacity、准入前置路径内部的 ensure_physical_fit 等):
      深度 +1 并入外层同一事务,不另起 id——事务粒度 = 调度器视角的一次
      manager 公开 mutation。
    - 异常路径:放弃事务登记(abort),不执行对账(避免二次异常掩盖
      原始错误);已写出的部分行保留在 journal 中,本轮 run 本就
      fail-closed 终止。
    """

    @functools.wraps(method)
    def wrapper(self: "SessionKVCacheManager", *args: Any, **kwargs: Any) -> Any:
        recorder = self._metrics_recorder
        if recorder is None or not recorder.journal_enabled:
            return method(self, *args, **kwargs)
        if self._journal_transaction_depth:
            self._journal_transaction_depth += 1
            try:
                return method(self, *args, **kwargs)
            finally:
                self._journal_transaction_depth -= 1
        recorder.begin_transaction()
        self._journal_transaction_depth = 1
        try:
            result = method(self, *args, **kwargs)
        except BaseException:
            self._journal_transaction_depth = 0
            recorder.abort_transaction()
            raise
        self._journal_transaction_depth = 0
        transaction_id, ranks = recorder.finish_transaction()
        self._journal_reconcile_transaction(transaction_id, ranks)
        return result

    return wrapper


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

    _require_nonnegative_int(tokens, "tokens")
    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    head_dim = model.hidden_size // model.num_heads
    bytes_per_head = 2 * model.layers * tokens * head_dim * model.bytes_per_elem
    shards = tuple(
        bytes_per_head * head_count
        for head_count in attention_heads_by_tp_rank(model.num_heads, tp_degree)
    )
    expected = 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem
    if sum(shards) != expected:
        raise RuntimeError("whole-head KV shards do not preserve total KV bytes")
    return shards


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
    session_id: str
    logical_context_tokens: int
    state: str
    instance_index: Optional[int]
    shard_bytes: tuple[int, ...]
    last_completion_ns: Optional[int]
    active: bool
    last_request_id: Optional[str]
    evicted_at_ns: Optional[int]
    evicted_by_request_id: Optional[str]

    @property
    def context_tokens(self) -> int:
        """Compatibility alias used by manifest and small fixture code."""

        return self.logical_context_tokens

    @property
    def total_bytes(self) -> int:
        return sum(self.shard_bytes)


@dataclass
class SessionKVState:
    session_id: str
    logical_context_tokens: int
    state: str
    instance_index: Optional[int]
    shard_bytes: tuple[int, ...]
    last_completion_ns: Optional[int] = None
    active: bool = False
    last_request_id: Optional[str] = None
    evicted_at_ns: Optional[int] = None
    evicted_by_request_id: Optional[str] = None

    def snapshot(self) -> SessionKVSnapshot:
        return SessionKVSnapshot(
            session_id=self.session_id,
            logical_context_tokens=self.logical_context_tokens,
            state=self.state,
            instance_index=self.instance_index,
            shard_bytes=self.shard_bytes,
            last_completion_ns=self.last_completion_ns,
            active=self.active,
            last_request_id=self.last_request_id,
            evicted_at_ns=self.evicted_at_ns,
            evicted_by_request_id=self.evicted_by_request_id,
        )


@dataclass(frozen=True)
class KVTransferShard:
    relative_tp_rank: int
    source_rank: Optional[int]
    target_rank: Optional[int]
    bytes: int


@dataclass(frozen=True)
class KVTransfer:
    action: str
    phase: str
    reason: str
    session_id: str
    trigger_request_id: str
    source_instance_index: Optional[int]
    target_instance_index: Optional[int]
    history_tokens: int
    total_bytes: int
    shards: tuple[KVTransferShard, ...]


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

    @property
    def transfer_bytes(self) -> int:
        return sum(shard.bytes for shard in self.transfer_shards)


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
    """Exact TP-sharded local-HBM session KV state machine.

    All mutating operations are planner-time operations.  The caller emits the
    corresponding NoC or recompute ET nodes from the returned decisions; an
    ``evict_delete`` intentionally has no ET representation.
    """

    def __init__(
        self,
        topology: Any,
        model: Any,
        *,
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
        self.model_weight_bytes_by_tp_rank = model_weight_shard_bytes_by_tp_rank(
            model, self.tp_degree
        )
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
        # D4 (2026-09-05): 被动逐出深缺口计数——ensure_physical_fit 候选
        # 耗尽(活跃会话占满、无可逐冷会话)优雅推迟 admitted=False 时递增,
        # 由 online_service summary 行(kv_deep_gap_events)落盘观测;
        # 实证负载预期恒 0(纯被动反事实:峰值占用 90%~97%、深缺口=0)。
        self.deep_gap_events: int = 0
        self._strict_kv_invariants = (
            _strict_kv_invariants_from_environment()
            if strict_invariants is None
            else strict_invariants
        )
        self._check_invariants()
        self._initialize_incremental_invariants()
        # Metrics observation state (doc sec.7.4): resident KV is tracked as
        # per-session segments so that chiplet-projection removes always walk
        # back the exact recorded distribution of an earlier add.
        self._metrics_recorder = _METRICS_RECORDER
        self._metrics_segments: dict[str, dict[str, list[Any]]] = {}
        self._metrics_segment_counters: dict[str, int] = {}
        # ③ journal 事务深度(0 = 无事务;公开 mutation 装饰器维护)。
        self._journal_transaction_depth = 0
        if self._metrics_recorder is not None:
            for rank in sorted(self._rank_states):
                self._metrics_recorder.initialize_rank(
                    rank, self._rank_states[rank].capacity_bytes
                )
            for rank in sorted(self._rank_states):
                state = self._rank_states[rank]
                # 权重预载是 journal 首 record:无请求上下文,时刻取 0
                # (run 首记录),transaction_id=0(构造期,先于任何事务)。
                self._metrics_recorder.record(
                    planner_time_ns=0,
                    anchor_kind="tick_zero",
                    request_id=None,
                    session_id=None,
                    rank=rank,
                    instance_index=state.instance_index,
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
    def node_states(self) -> tuple[NodeHBMState, ...]:
        return tuple(self._rank_states[rank] for rank in sorted(self._rank_states))

    def session_snapshot(self, session_id: str) -> Optional[SessionKVSnapshot]:
        state = self._sessions.get(session_id)
        return None if state is None else state.snapshot()

    def hbm_snapshots(self, instance_index: Optional[int] = None) -> tuple[NodeHBMSnapshot, ...]:
        if instance_index is None:
            ranks = tuple(sorted(self._rank_states))
        else:
            ranks = tuple(self.topology.instance(instance_index).ranks)
        return tuple(self._rank_states[rank].snapshot() for rank in ranks)

    def final_session_counts(self) -> tuple[int, int]:
        resident = sum(state.state == RESIDENT for state in self._sessions.values())
        return resident, len(self._sessions) - resident

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

    def _candidate_sessions(
        self,
        instance_index: int,
        protected_sessions: Iterable[str],
    ) -> list[SessionKVState]:
        protected = set(protected_sessions)
        candidates = [
            state
            for state in self._sessions.values()
            if state.state == RESIDENT
            and state.instance_index == instance_index
            and not state.active
            and state.last_completion_ns is not None
            and state.session_id not in protected
        ]
        candidates.sort(key=lambda state: (int(state.last_completion_ns), state.session_id))
        return candidates

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
        if (
            session.state != RESIDENT
            or session.instance_index is None
            or len(session.shard_bytes) != self.tp_degree
        ):
            return None
        return session.instance_index, tuple(int(value) for value in session.shard_bytes)

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
        if session.state not in {RESIDENT, EVICTED}:
            raise RuntimeError(f"unknown KV state: {session.state}")
        if session.state == RESIDENT:
            if session.instance_index is None:
                raise RuntimeError("resident session has no instance")
            if len(session.shard_bytes) != self.tp_degree:
                raise RuntimeError("resident session has invalid TP shard vector")
        elif session.instance_index is not None:
            raise RuntimeError("evicted session retained an instance")

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
    # none of them feeds back into manager decisions.
    # ------------------------------------------------------------------

    def _metrics_add_segment(
        self,
        session_id: str,
        instance_index: int,
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
        self._metrics_segments.setdefault(session_id, {})[segment_id] = [
            instance_index,
            tuple(int(value) for value in shards),
        ]
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
                instance_index=instance_index,
                allocation_key=f"resident:{session_id}:{segment_id}",
                resident_kv_delta_bytes=int(value),
                cause=cause,
            )

    def _metrics_remove_session_segments(
        self,
        session_id: str,
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        recorder = self._metrics_recorder
        if recorder is None:
            return
        segments = self._metrics_segments.get(session_id, {})
        for segment_id in sorted(segments):
            instance_index, shards = segments[segment_id]
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
                    instance_index=instance_index,
                    allocation_key=f"resident:{session_id}:{segment_id}",
                    resident_kv_delta_bytes=-int(value),
                    cause=cause,
                )
        segments.clear()

    def _metrics_move_session_segments(
        self,
        session_id: str,
        target_instance_index: int,
        total_shards: Sequence[int],
        *,
        now_ns: int,
        anchor_kind: str,
        request_id: str,
        cause: str,
    ) -> None:
        """Mirror a migration: the target add is recorded as one consolidated
        segment (a fresh allocation key carries the full magnitude, so the
        chiplet projection stays exactly removable), then each original
        segment is removed from the source ranks under its own key (doc
        sec.7.4/7.6).  Target add precedes source release, matching the
        dynamic GraphBatch transfer dependency ordering."""

        recorder = self._metrics_recorder
        if recorder is None:
            return
        segments = self._metrics_segments.get(session_id, {})
        target_ranks = self.topology.instance(target_instance_index).ranks
        counter = self._metrics_segment_counters.get(session_id, 0)
        self._metrics_segment_counters[session_id] = counter + 1
        segment_id = f"seg{counter}"
        consolidated_key = f"resident:{session_id}:{segment_id}"
        for rank, value in zip(target_ranks, total_shards):
            if not value:
                continue
            recorder.record(
                planner_time_ns=now_ns,
                anchor_kind=anchor_kind,
                request_id=request_id,
                session_id=session_id,
                rank=rank,
                instance_index=target_instance_index,
                allocation_key=consolidated_key,
                resident_kv_delta_bytes=int(value),
                cause=f"{cause}_target_add",
            )
        for old_segment_id in sorted(segments):
            source_instance_index, shards = segments[old_segment_id]
            source_ranks = self.topology.instance(source_instance_index).ranks
            old_key = f"resident:{session_id}:{old_segment_id}"
            for rank, value in zip(source_ranks, shards):
                if not value:
                    continue
                recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    instance_index=source_instance_index,
                    allocation_key=old_key,
                    resident_kv_delta_bytes=-int(value),
                    cause=f"{cause}_source_remove",
                )
        self._metrics_segments[session_id] = {
            segment_id: [target_instance_index, tuple(int(v) for v in total_shards)]
        }

    # ------------------------------------------------------------------
    # P1 journal 事务对账与 run 末 checksum 门。两者都只在 recorder 处于
    # journal 模式时生效,且都是纯观测校验:任何不一致 raise(fail-closed),
    # 不做任何静默修复/钳位——发现账本 bug 正是 journal 的价值。
    # ------------------------------------------------------------------

    def _journal_reconcile_transaction(
        self, transaction_id: int, ranks: Iterable[int]
    ) -> None:
        """④ commit 点对账:manager 实际状态 vs journal 记账逐 rank 比对。

        journal 每行的 before/after 由 recorder 的逐 rank 运行计数推出,
        该计数 = 全部已写行 Σdelta;故此处比对等价于"该事务 before+Σdelta
        推出的 after vs manager 快照"。只查本事务写过的 rank(未写行的
        rank 无字节变化,不可能漂移);weight 由增量不变量不覆盖,此处
        一并核对(journal 对 weight 同样记账)。"""

        recorder = self._metrics_recorder
        mismatches = []
        for rank in sorted(set(ranks)):
            state = self._rank_states[rank]
            journal_totals = recorder.rank_totals(rank)
            manager_totals = (
                state.model_weight_bytes,
                state.resident_kv_bytes,
                state.reserved_request_bytes,
            )
            if journal_totals != manager_totals:
                mismatches.append(
                    f"rank {rank}: manager=(weight={manager_totals[0]}, "
                    f"resident={manager_totals[1]}, reserved={manager_totals[2]}), "
                    f"journal=(weight={journal_totals[0]}, "
                    f"resident={journal_totals[1]}, "
                    f"reserved={journal_totals[2]})"
                )
        if mismatches:
            raise RuntimeError(
                "KV delta journal reconciliation failed for transaction "
                f"{transaction_id}: " + "; ".join(mismatches)
            )

    def verify_journal_checksum(self) -> Optional[dict[str, Any]]:
        """Run 末 checksum 门(fail-closed):流式重放权威 journal 并对账。

        三重断言,任一不成立 raise(错误信息含逐 rank 差额表):
        1. journal 重放逐 rank 终态(weight/resident/reserved)与 manager
           快照完全一致(重放本身先校验 sequence 连续/前后快照自洽/
           planner_time_ns 与 transaction_id 单调/断行报行号);
        2. 写出条数一致(recorder 内存计数 vs 重放行数,流式丢失即暴露);
        3. 守恒:journal 重放终态 resident==0、reserved==0、
           physical==weight_bytes(权重驻留、无会话/预约残留)。

        产物:kv_delta_journal_checksum.json(journal 同目录,由 runner
        归档进 run_dir/results/)。recorder 未装或非 journal 模式时返回
        None(门只对 journal-on 的 run 生效,旁路态零行为差)。"""

        recorder = self._metrics_recorder
        if recorder is None or not recorder.journal_enabled:
            return None
        replay = recorder.replay_journal()
        if replay["line_count"] != recorder.record_count:
            raise RuntimeError(
                "KV delta journal run-end checksum mismatch: replayed "
                f"{replay['line_count']} lines but the recorder wrote "
                f"{recorder.record_count}"
            )
        problems: list[str] = []
        for rank in sorted(self._rank_states):
            state = self._rank_states[rank]
            replayed = replay["ranks"].get(rank)
            if replayed is None:
                problems.append(
                    f"rank {rank}: journal replay has no state "
                    f"(manager=(weight={state.model_weight_bytes}, "
                    f"resident={state.resident_kv_bytes}, "
                    f"reserved={state.reserved_request_bytes}))"
                )
                continue
            manager_totals = (
                state.model_weight_bytes,
                state.resident_kv_bytes,
                state.reserved_request_bytes,
            )
            journal_totals = (
                replayed["weight"],
                replayed["resident"],
                replayed["reserved"],
            )
            if journal_totals != manager_totals:
                problems.append(
                    f"rank {rank}: manager=(weight={manager_totals[0]}, "
                    f"resident={manager_totals[1]}, "
                    f"reserved={manager_totals[2]}), "
                    f"journal_replay=(weight={journal_totals[0]}, "
                    f"resident={journal_totals[1]}, "
                    f"reserved={journal_totals[2]}), "
                    f"delta=(weight={manager_totals[0] - journal_totals[0]}, "
                    f"resident={manager_totals[1] - journal_totals[1]}, "
                    f"reserved={manager_totals[2] - journal_totals[2]})"
                )
        for rank in sorted(replay["ranks"]):
            if rank not in self._rank_states:
                problems.append(
                    f"rank {rank}: journal replay knows a rank the manager "
                    "does not"
                )
        if problems:
            raise RuntimeError(
                "KV delta journal run-end checksum mismatch: "
                + "; ".join(problems)
            )
        # 守恒断言:合法 run 终态必须是"只剩权重"。若真实 workload 允许
        # 会话 KV 驻留终态,这里会 raise——按 P1 裁决保留 fail-closed,
        # 不放宽门,上报裁决而不是改门。
        conservation: list[str] = []
        for rank in sorted(replay["ranks"]):
            replayed = replay["ranks"][rank]
            if (
                replayed["resident"] != 0
                or replayed["reserved"] != 0
                or replayed["physical"] != replayed["weight"]
            ):
                conservation.append(
                    f"rank {rank}: resident={replayed['resident']}, "
                    f"reserved={replayed['reserved']}, "
                    f"physical={replayed['physical']}, "
                    f"weight={replayed['weight']}"
                )
        if conservation:
            raise RuntimeError(
                "KV delta journal conservation violated at run end "
                "(resident must be 0, reserved must be 0, physical must "
                "equal weight): " + "; ".join(conservation)
            )
        summary = {
            "schema_version": 1,
            "journal": recorder.journal_path.name,
            "line_count": replay["line_count"],
            "sha256": replay["sha256"],
            "transaction_count": replay["transaction_count"],
            "max_transaction_id": replay["max_transaction_id"],
            "checks": {
                "manager_state_match": True,
                "residual_resident_zero": True,
                "residual_reserved_zero": True,
                "physical_equals_weight": True,
            },
            "ranks": replay["ranks"],
        }
        checksum_path = (
            recorder.journal_path.parent / type(recorder).JOURNAL_CHECKSUM_NAME
        )
        checksum_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return summary


    def _delete(
        self,
        victim: SessionKVState,
        *,
        now_ns: int,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> EvictionRecord:
        if (
            victim.state != RESIDENT
            or victim.instance_index is None
            or victim.active
            or victim.last_completion_ns is None
        ):
            raise RuntimeError("only completed inactive resident sessions may be deleted")
        source_instance = victim.instance_index
        before = self._remaining(source_instance)
        self._remove_shards(source_instance, victim.shard_bytes)
        victim.state = EVICTED
        victim.instance_index = None
        victim.evicted_at_ns = now_ns
        victim.evicted_by_request_id = trigger_request_id
        self._check_invariants_after_mutation(session_ids=(victim.session_id,))
        self._metrics_remove_session_segments(
            victim.session_id,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=f"evict_delete:{reason}",
        )
        after = self._remaining(source_instance)
        record = EvictionRecord(
            time_ns=now_ns,
            phase=phase,
            reason=reason,
            trigger_request_id=trigger_request_id,
            victim_session_id=victim.session_id,
            victim_instance_index=source_instance,
            victim_last_completion_ns=victim.last_completion_ns,
            context_tokens=victim.logical_context_tokens,
            shard_bytes=tuple(victim.shard_bytes),
        )
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type="evict_delete",
            reason=reason,
            trigger_request_id=trigger_request_id,
            session_id=victim.session_id,
            source_instance_index=source_instance,
            context_tokens=victim.logical_context_tokens,
            total_bytes=sum(victim.shard_bytes),
            shard_bytes=victim.shard_bytes,
            last_completion_ns=victim.last_completion_ns,
            before=before,
            after=after,
        )
        return record

    @_journal_transaction
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
            details = ", ".join(
                f"rank={rank}, required={needed}, empty_available={available}"
                for rank, needed, available in zip(instance.ranks, required, maximum)
                if rank in impossible
            )
            raise ValueError(f"request cannot fit an empty instance: {details}")
        evictions: list[EvictionRecord] = []
        insufficient = self._insufficient(instance_index, required)
        # See ensure_physical_fit(): eligibility and LRU order of unselected
        # candidates are static inside this single reclamation stage.
        candidates = (
            self._candidate_sessions(instance_index, protected_sessions)
            if insufficient
            else []
        )
        candidate_index = 0
        while insufficient:
            if candidate_index >= len(candidates):
                before = self._remaining(instance_index)
                # D4 (2026-09-05): 被动逐出深缺口——候选耗尽(活跃会话占满、
                # 无可逐冷会话)优雅推迟 admitted=False;capacity_epoch 重试
                # 机制保留(无死锁风险),计数入 summary 行观测。
                self.deep_gap_events += 1
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
            victim = candidates[candidate_index]
            candidate_index += 1
            evictions.append(
                self._delete(
                    victim,
                    now_ns=now_ns,
                    phase=phase,
                    reason=reason,
                    trigger_request_id=trigger_request_id,
                )
            )
            insufficient = self._insufficient(instance_index, required)
        if self._insufficient(instance_index, required):
            # I3 (2026-09-05): 逐出循环退出即必须已满足本请求 required
            # (被动逐出"逐到刚好够即停"的显式断言;多逐或少逐都是缺陷)。
            raise RuntimeError(
                "physical-fit stage returned success while still insufficient")
        return CapacityResult(tuple(evictions), True, ())

    @_journal_transaction
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
                    instance_index=instance_index,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=int(value),
                    cause=reason,
                )
        return CapacityResult(evictions, True, ())

    @_journal_transaction
    def release_request_capacity(
        self,
        request_id: str,
        now_ns: int,
    ) -> RequestCapacityReservation:
        """Release one request's decode-capacity reservation.

        ``now_ns`` 是本 mutation 的真实 planner 时刻(P1 语义修改①:改前
        本 API 无时间戳参数,metrics 复用最近 KV 事件时间——journal 化后
        该近似会污染权威账本时序,故改为必传参数,全部调用方显式给值;
        传入时刻若早于 journal 已有最后时刻,recorder 直接 raise)。"""

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
                    instance_index=reservation.instance_index,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=-int(value),
                    cause="reservation_release",
                )
        return reservation

    @_journal_transaction
    def extend_request_capacity(
        self,
        request_id: str,
        extra_shards: Sequence[int],
        now_ns: int,
        *,
        reason: str = "reservation_extend",
    ) -> RequestCapacityReservation:
        """按 extra_shards 增量扩补一笔已登记的 decode 容量预约。

        准入预占净额修正(2026-09-06)的回补侧:``_try_admit_prefill`` 在全量
        预约撞 deep-gap 时改按净额(终态 − 本会话旧驻留)重试,而
        ``prepare_history`` 的 NOC 迁移把旧 KV 从 decode 目标实例删掉后,
        必须把净额预约回补到全量——复刻原设计"从准入占位到 P→D move 完成"
        的防抢占语义(move 时刻 KV 已物理搬到 prefill 实例,目标侧
        existing=0,需要全量空间)。resident→reserved 1:1 换位:迁移删除
        的旧驻留字节与回补的预约字节相等,任何瞬间 weight+resident+
        reserved ≤ capacity 不超订。
        """

        reservation = self._reservations.get(request_id)
        if reservation is None:
            raise RuntimeError(f"unknown KV reservation for request {request_id}")
        extra = tuple(int(value) for value in extra_shards)
        if len(extra) != self.tp_degree or any(value < 0 for value in extra):
            raise ValueError("reservation extension shard vector is invalid")
        for rank, value in zip(
            self.topology.instance(reservation.instance_index).ranks, extra
        ):
            self._rank_states[rank].reserved_request_bytes += value
        # frozen dataclass:replace 重建并重赋值(增量账本随后从新值刷新)。
        extended = replace(
            reservation,
            shard_bytes=tuple(
                current + addition
                for current, addition in zip(reservation.shard_bytes, extra)
            ),
        )
        self._reservations[request_id] = extended
        self._check_invariants_after_mutation(reservation_ids=(request_id,))
        if self._metrics_recorder is not None:
            for rank, value in zip(
                self.topology.instance(extended.instance_index).ranks, extra
            ):
                if not value:
                    continue
                self._metrics_recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind="prefill_start",
                    request_id=request_id,
                    session_id=extended.session_id,
                    rank=rank,
                    instance_index=extended.instance_index,
                    allocation_key=f"reservation:{request_id}",
                    reserved_kv_delta_bytes=int(value),
                    cause=f"reservation_extend:{reason}",
                )
        return extended

    def _history_shards(self, tokens: int) -> tuple[int, ...]:
        return kv_cache_shard_bytes_for_tokens(self.model, tokens, self.tp_degree)

    @_journal_transaction
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
        existing_target = (
            state.shard_bytes
            if state is not None
            and state.state == RESIDENT
            and state.instance_index == target_instance_index
            else tuple(0 for _ in desired)
        )
        needed = tuple(want - current for want, current in zip(desired, existing_target))
        if any(value < 0 for value in needed):
            raise RuntimeError("session KV would shrink during history preparation")
        protected = (session_id,) if state is not None else ()
        fit = self.ensure_physical_fit(
            target_instance_index,
            needed,
            now_ns,
            trigger_request_id,
            protected,
            phase=phase,
            reason="history_and_prefill_admission",
        )
        evictions = fit.evictions
        if not fit.admitted:
            action = NO_HISTORY if state is None else (RECOMPUTE if state.state == EVICTED else LOCAL_HIT)
            return HistoryDecision(
                action=action,
                source_instance_index=(None if state is None else state.instance_index),
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfer_shards=(),
                recompute_tokens=history_tokens if state is not None and state.state == EVICTED else 0,
                evictions=evictions,
                admission_blocked=True,
                insufficient_ranks=fit.insufficient_ranks,
            )

        history_shards = self._history_shards(history_tokens)
        if state is None:
            state = SessionKVState(
                session_id=session_id,
                logical_context_tokens=0,
                state=RESIDENT,
                instance_index=target_instance_index,
                shard_bytes=tuple(0 for _ in history_shards),
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
                evictions=evictions,
            )

        if state.state == EVICTED:
            before = self._remaining(target_instance_index)
            self._add_shards(target_instance_index, history_shards)
            state.state = RESIDENT
            state.instance_index = target_instance_index
            state.shard_bytes = history_shards
            state.active = True
            state.last_request_id = trigger_request_id
            state.evicted_at_ns = None
            state.evicted_by_request_id = None
            self._check_invariants_after_mutation(session_ids=(session_id,))
            self._metrics_add_segment(
                session_id,
                target_instance_index,
                history_shards,
                now_ns=now_ns,
                anchor_kind="prefill_start",
                request_id=trigger_request_id,
                cause="recompute_history_restore",
            )
            after = self._remaining(target_instance_index)
            self._event(
                now_ns=now_ns,
                phase=phase,
                event_type="recompute",
                reason="deleted_history_recompute",
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                target_instance_index=target_instance_index,
                context_tokens=history_tokens,
                total_bytes=sum(history_shards),
                shard_bytes=history_shards,
                last_completion_ns=state.last_completion_ns,
                before=before,
                after=after,
            )
            return HistoryDecision(
                action=RECOMPUTE,
                source_instance_index=None,
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfer_shards=(),
                recompute_tokens=history_tokens,
                evictions=evictions,
            )

        if state.instance_index is None:
            raise RuntimeError("resident session is missing its instance")
        source_instance = state.instance_index
        state.active = True
        state.last_request_id = trigger_request_id
        if source_instance == target_instance_index:
            self._check_invariants_after_mutation(session_ids=(session_id,))
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
                evictions=evictions,
            )

        source_ranks = self.topology.instance(source_instance).ranks
        target_ranks = self.topology.instance(target_instance_index).ranks
        transfer_shards = tuple(
            KVTransferShard(relative_rank, source_rank, target_rank, byte_count)
            for relative_rank, (source_rank, target_rank, byte_count) in enumerate(
                zip(source_ranks, target_ranks, history_shards)
            )
        )
        before = self._remaining(target_instance_index)
        # The dynamic transfer contract admits target shards before releasing
        # the source; planner state advances atomically after that admission.
        self._add_shards(target_instance_index, history_shards)
        self._remove_shards(source_instance, history_shards)
        state.instance_index = target_instance_index
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_move_session_segments(
            session_id,
            target_instance_index,
            history_shards,
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="history_noc_migrate",
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
            transfer_shards=transfer_shards,
            recompute_tokens=0,
            evictions=evictions,
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
        if state.state != RESIDENT or state.instance_index != instance_index or not state.active:
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
        state.shard_bytes = desired
        state.logical_context_tokens = context_tokens
        state.last_request_id = trigger_request_id
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_add_segment(
            session_id,
            instance_index,
            delta,
            now_ns=now_ns,
            anchor_kind=(
                "completion" if phase == "decode" else "prefill_start"
            ),
            request_id=trigger_request_id,
            cause=reason,
        )
        return CapacityResult(evictions, True, ())

    @_journal_transaction
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

    @_journal_transaction
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
        if state.state != RESIDENT or state.instance_index is None or not state.active:
            raise RuntimeError("Prefill KV must be active and resident before Decode")
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
        empty_transfer = KVTransfer(
            action=LOCAL_HIT,
            phase="prefill_decode",
            reason="prefill_decode_local_reuse",
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance,
            target_instance_index=target_instance_index,
            history_tokens=state.logical_context_tokens,
            total_bytes=0,
            shards=(),
        )
        if not fit.admitted:
            return MoveDecision(
                action=LOCAL_HIT if source_instance == target_instance_index else NOC_MIGRATE,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                transfer=empty_transfer,
                evictions=evictions,
                admission_blocked=True,
                insufficient_ranks=fit.insufficient_ranks,
            )
        if source_instance == target_instance_index:
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
            return MoveDecision(LOCAL_HIT, source_instance, target_instance_index, empty_transfer, evictions)

        source_ranks = self.topology.instance(source_instance).ranks
        target_ranks = self.topology.instance(target_instance_index).ranks
        shards = tuple(
            KVTransferShard(relative_rank, source_rank, target_rank, byte_count)
            for relative_rank, (source_rank, target_rank, byte_count) in enumerate(
                zip(source_ranks, target_ranks, state.shard_bytes)
            )
        )
        transfer = KVTransfer(
            action=NOC_MIGRATE,
            phase="prefill_decode",
            reason="prefill_decode_instance_migrate",
            session_id=session_id,
            trigger_request_id=trigger_request_id,
            source_instance_index=source_instance,
            target_instance_index=target_instance_index,
            history_tokens=state.logical_context_tokens,
            total_bytes=sum(state.shard_bytes),
            shards=shards,
        )
        before = self._remaining(target_instance_index)
        self._add_shards(target_instance_index, state.shard_bytes)
        self._remove_shards(source_instance, state.shard_bytes)
        state.instance_index = target_instance_index
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_move_session_segments(
            session_id,
            target_instance_index,
            state.shard_bytes,
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="prefill_decode_migrate",
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

    @_journal_transaction
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

    @_journal_transaction
    def mark_complete(
        self,
        session_id: str,
        completion_ns: int,
        request_id: str,
    ) -> tuple[EvictionRecord, ...]:
        state = self._sessions[session_id]
        if state.state != RESIDENT or state.instance_index is None or not state.active:
            raise RuntimeError("only active resident sessions may complete")
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

    @_journal_transaction
    def retire_terminal_session(
        self,
        session_id: str,
        completion_ns: int,
        request_id: Optional[str] = None,
    ) -> Optional[int]:
        """Drop a completed terminal session without synthesizing a KV action.

        ``mark_complete`` deliberately retains KV for a possible following
        turn.  The online scheduler calls this only after it has proved that
        the completed request has no successor and has emitted every output
        that reads the completion snapshot.
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
            # ① 嵌套公开复用:retire 事务内以本 mutation 的 completion_ns
            # 作为 release 的真实 planner 时刻(并入同一 journal 事务)。
            self.release_request_capacity(reservation_id, completion_ns)

        released_instance_index: Optional[int] = None
        if state.state == RESIDENT:
            if state.instance_index is None:
                raise RuntimeError("resident completed session has no instance")
            released_instance_index = state.instance_index
            self._remove_shards(released_instance_index, state.shard_bytes)
            self._metrics_remove_session_segments(
                session_id,
                now_ns=completion_ns,
                anchor_kind="completion",
                request_id=terminal_request_id,
                cause="terminal_session_retire",
            )
        elif state.state != EVICTED:
            raise RuntimeError(f"unknown KV state: {state.state}")

        # Metrics helpers intentionally no-op without an observer; retirement
        # must nevertheless remove the per-session containers in both modes.
        self._metrics_segments.pop(session_id, None)
        self._metrics_segment_counters.pop(session_id, None)
        self._forget_pressure_event_keys(terminal_request_id)
        del self._sessions[session_id]
        self._check_invariants_after_mutation(session_ids=(session_id,))
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
            if session.state not in {RESIDENT, EVICTED}:
                raise RuntimeError(f"unknown KV state: {session.state}")
            if session.state == RESIDENT:
                if session.instance_index is None:
                    raise RuntimeError("resident session has no instance")
                if len(session.shard_bytes) != self.tp_degree:
                    raise RuntimeError("resident session has invalid TP shard vector")
                accumulated[session.instance_index] = [
                    total + value for total, value in zip(
                        accumulated[session.instance_index], session.shard_bytes
                    )
                ]
            elif session.instance_index is not None:
                raise RuntimeError("evicted session retained an instance")
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
