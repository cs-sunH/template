"""Session-scoped tiered local-HBM/remote-pool KV cache management.

Two-state session model (2026-09 session-level Tiered-LRU, blueprint sh_2.0
``face_scheduler.py`` ``KVCacheManager`` adapted into the wscllm shell):

- ``LOCAL_HBM``     all ``L`` layers resident on one TP instance;
- ``REMOTE_MEMORY`` every layer in the remote pool, no local instance.

The only runtime states are "fully local" and "fully remote" (plus the
stateless no-history / retired bookkeeping).  A session is never partially
resident: the legacy ``PARTIAL_HBM_REMOTE`` runtime state and the suffix-half
eviction/restore paths were removed.  The ``PARTIAL_HBM_REMOTE`` constant is
kept only as a rejected value of the two-state whitelist — nothing parses
old logs with it, it is unreachable at runtime, and ``_check_invariants``
plus the consumer whitelists both raise on it (its sole remaining consumer
is the fail-closed invariant test).

Reclamation is a single-stage whole-session LRU (``ensure_physical_fit``):
candidates are completed inactive fully-local sessions of the instance in
``(last_completion_ns, session_id)`` order; each selected victim is offloaded
whole (remote_store of layers ``[0, L)`` -> REMOTE, then the full local
release), the shortfall is rechecked after every single eviction, and
over-release is allowed.  Exhaustion keeps the wscllm failure semantics
(graceful ``admission_blocked`` head-of-line blocking + ``deep_gap_events``
counter), never the sh blueprint's raise.  Recovery from the remote pool is
likewise whole-session (``REMOTE_RESTORE`` restores layers ``[0, L)`` before
the prefill may consume any history KV).

The WSC-LLM net-credit reservation machinery (``reserve_request_capacity`` /
``extend_request_capacity`` / ``release_request_capacity``) is preserved with
its resident->reserved 1:1 swap semantics; its internal capacity convergence
now runs through the same single-stage whole-session reclamation.

Every mutation point (whole-session eviction, remote restore) is covered by
the ``_journal_transaction`` decorator through its enclosing public method and
is mirrored into the per-rank HBM delta journal including the ``remote``
pool column (see ``verify_journal_checksum``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import functools
import json
import os
from typing import Any, Iterable, Optional, Sequence


# ---------------------------------------------------------------------------
# Session location states (two-state runtime model) and history actions.
# ---------------------------------------------------------------------------
LOCAL_HBM = "local_hbm"
PARTIAL_HBM_REMOTE = "partial_hbm_remote"
REMOTE_MEMORY = "remote_memory"

NO_HISTORY = "NO_HISTORY"
LOCAL_HIT = "LOCAL_HIT"
NOC_MIGRATE = "NOC_MIGRATE"
# The legacy partial-era action constants (REMOTE_LOAD / PARTIAL_MIGRATE)
# are gone from this module: nothing imported them (old-log fixtures feed
# the raw string literals), the runtime cannot emit them (the partial
# restore paths were removed) and _check_invariants plus the consumer
# whitelists both raise on non-two-state values.
REMOTE_RESTORE = "REMOTE_RESTORE"    # REMOTE: full restore (the only remote path)

# B1 已消费前缀的摊销压缩水位(2026-08-28,风格对齐 graph_batch_builder
# 的 M1 压缩):_events 已消费水位达到该值即整段删除前缀(单一水位条件,
# 无二次门槛——见 compact_events),均摊 O(1)/事件。
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
#
# B2(2026-09):journal 行 schema 扩 remote 列(远端池账面,按"该 rank
# 存入远端池的字节"逐 rank 记账);终态守恒口径改为"本地 resident==0
# 且 remote 字节与会话账本守恒"(retire 静默核销后 remote 亦归零)。
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
        # "prefill_decode"/"completion" 键已删:两者从不流入本映射的唯一
        # 三个调用点(预约/增长/整会话逐出的 anchor 标注)——prefill_decode
        # 只作为 KVTransfer 的 phase 标签存在,completion 边界的预约释放与
        # 核销在 retire_terminal_session 内直接硬编码 "completion"。
    }.get(phase, "stage_boundary")


def _journal_transaction(method: Any) -> Any:
    """③ 公开 mutation = journal 事务(原子校验边界)装饰器。

    语义(doc §6-P1 ③/④):
    - recorder 未装或非 journal 模式:透传,零开销零行为差(journal 是
      纯观测旁路,关闭时与改前逐字节一致)。
    - 最外层公开调用:入口向 recorder 申请单调 transaction_id(journal
      各行携带该 id),并对 manager 全量可变状态拍回滚快照;正常返回 =
      commit 点,此时方法体内的全部 _check_invariants_after_mutation 已
      通过,随即执行 ④ 逐 rank 对账 _journal_reconcile_transaction
      (manager 快照 vs journal 记账)。
    - 嵌套公开复用(retire_terminal_session 内部的预约释放、准入前置
      路径内部的 ensure_physical_fit 等):
      深度 +1 并入外层同一事务,不另起 id——事务粒度 = 调度器视角的一次
      manager 公开 mutation。
    - 失败路径(方法体异常 / journal 行写入失败 / commit 对账不一致):
      manager 整体回滚到入口快照并放弃事务登记(abort),原异常继续上抛
      ——事务对 manager 真原子(该原子性断言仅适用于 journal 模式;旁路
      态——recorder 未装或非 journal 模式——直接透传,无快照亦无回滚,
      全部内部 raise 点均被前置守卫结构性排除,故旁路态下不存在可达的
      半更新路径):要么整体生效,要么整体回到入口状态,
      失败重试不会叠加在半更新状态上。journal 是行级流式落盘,已写出
      的行不可回收(见 MemoryActionRecorder.abort_transaction);observer
      侧账本的补偿属 recorder/observer 域,两者都由本轮 fail-closed 终止
      兜底。
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
        snapshot = self._journal_rollback_snapshot()
        try:
            result = method(self, *args, **kwargs)
        except BaseException:
            self._journal_transaction_depth = 0
            self._restore_journal_rollback_snapshot(snapshot)
            recorder.abort_transaction()
            raise
        self._journal_transaction_depth = 0
        try:
            transaction_id, ranks = recorder.finish_transaction()
            self._journal_reconcile_transaction(transaction_id, ranks)
        except BaseException:
            self._restore_journal_rollback_snapshot(snapshot)
            recorder.abort_transaction()
            raise
        return result

    return wrapper


def _require_nonnegative_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


def _require_int_vector(values: Sequence[Any], name: str) -> tuple[int, ...]:
    """Reject non-int shard inputs before any coercion: ``int()`` would
    silently truncate 1.9 to 1 and accept True as 1 (under-reservation)."""

    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in values
    ):
        raise ValueError(f"{name} must contain integers")
    return tuple(values)


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
    """LLaMA-family model-size estimate (manager twin).

    Known in-repo divergence (2026-09-30 audit, disclosed -- the old
    "Match the existing LLaMA-family model-size estimate exactly" claim
    was false): for the gelu variant this twin counts the final norm as 0
    elements while wsc_llm_scheduler.estimate_model_weight_bytes counts a
    final LayerNorm (weight + bias, 2*hidden_size).  The two agree exactly
    on the swiglu variant -- the production trace_config.csv model, and
    the shape every pinned test value assumes.  Converging either side is
    a frozen-formula family ruling (it would silently move HBM capacity
    accounting here / the W_bytes metric proxy in metrics_postprocess,
    which treats the scheduler twin as authoritative, and desync the
    sh/face lineage); left explicit for a user-level ruling instead of a
    silent "fix"."""

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


def kv_cache_bytes_for_tokens(model: Any, tokens: int) -> int:
    """Total (all layers, all heads) KV bytes for ``tokens`` tokens."""

    _require_nonnegative_int(tokens, "tokens")
    return 2 * model.layers * tokens * model.hidden_size * model.bytes_per_elem


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


# 纯函数按 (model, tokens, tp, 层域) 复用(G5-11):同一会话状态会在增量
# 不变量、快照与全量审计中重复求值;model 恒为可哈希 frozen dataclass,
# maxsize 有界防长跑驻留增长。异常不入缓存,非法输入仍逐次 fail-closed。
@functools.lru_cache(maxsize=4096)
def kv_cache_shard_bytes_for_layer_range(
    model: Any,
    tokens: int,
    tp_degree: int,
    *,
    layer_start: int,
    layer_end: int,
) -> tuple[int, ...]:
    """Return exact whole-head KV bytes for ``[layer_start, layer_end)``.

    Layer ranges are derived from ``model.layers``; bytes scale linearly with
    the number of layers so any layer-range split is byte-conservative by
    construction (blueprint sh_2.0 ``face_scheduler.py``)."""

    if tokens < 0:
        raise ValueError("KV token count must be non-negative")
    if (
        isinstance(layer_start, bool)
        or isinstance(layer_end, bool)
        or not isinstance(layer_start, int)
        or not isinstance(layer_end, int)
        or not 0 <= layer_start <= layer_end <= model.layers
    ):
        raise ValueError("KV layer range must be within the configured model")
    if model.hidden_size % model.num_heads:
        raise ValueError("hidden_size must be divisible by num_heads")
    heads = attention_heads_by_tp_rank(model.num_heads, tp_degree)
    bytes_per_head = (
        2
        * (layer_end - layer_start)
        * tokens
        * (model.hidden_size // model.num_heads)
        * model.bytes_per_elem
    )
    shards = tuple(head_count * bytes_per_head for head_count in heads)
    expected_total = (
        kv_cache_bytes_for_tokens(model, tokens)
        * (layer_end - layer_start)
        // model.layers
    )
    if sum(shards) != expected_total:
        raise RuntimeError(
            "whole-head layer-range KV partition does not preserve total bytes"
        )
    return shards


def physical_edge_ranks(hardware: Any) -> tuple[int, ...]:
    """Return all ranks on the deterministic physical mesh boundary."""

    edges = []
    for rank in range(hardware.npus_count):
        row, col = divmod(rank, hardware.mesh_cols)
        if (
            row == 0
            or row == hardware.mesh_rows - 1
            or col == 0
            or col == hardware.mesh_cols - 1
        ):
            edges.append(rank)
    return tuple(edges)


def _require_valid_rank(hardware: Any, rank: int) -> None:
    if rank < 0 or rank >= hardware.npus_count:
        raise ValueError(f"rank {rank} is outside 0-{hardware.npus_count - 1}")


def manhattan_hops(hardware: Any, source: int, target: int) -> int:
    source_row, source_col = divmod(source, hardware.mesh_cols)
    target_row, target_col = divmod(target, hardware.mesh_cols)
    return abs(source_row - target_row) + abs(source_col - target_col)


def nearest_edge_rank(
    hardware: Any,
    rank: int,
    edge_ranks: Sequence[int],
) -> int:
    """Closest mesh-boundary rank (min Manhattan hops, lowest id on ties)."""

    if not edge_ranks:
        raise ValueError("at least one remote-memory edge rank is required")
    _require_valid_rank(hardware, rank)
    return min(
        edge_ranks,
        key=lambda edge: (manhattan_hops(hardware, rank, edge), edge),
    )


def _deterministic_xy_route(hardware: Any, source: int, target: int) -> tuple[int, ...]:
    """XY dimension-ordered deterministic route (columns first, rows second).

    Single implementation rule: this reuses the loader's ``_xy_route``
    (generate_wsc_llm_trace.py, the structural twin of the sh blueprint's
    ``deterministic_xy_route``).  The import is deferred to call time because
    the loader itself imports constants from this module at load time.
    """

    from generate_wsc_llm_trace import _xy_route

    return tuple(_xy_route(hardware, source, target))


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
    """Two-state read-only session view.

    ``local_shard_bytes`` is the physically resident distribution (all ``L``
    layers when ``LOCAL_HBM``, the zero vector when ``REMOTE_MEMORY``).
    ``remote_bytes`` and ``logical_context_tokens`` currently have test-only
    readers (the tiered-eviction sequence and scheduler unit tests assert on
    them; the rest of the legacy full-vector/eviction-metadata fields were
    write-only and removed).
    """

    session_id: str
    location: str
    instance_index: Optional[int]
    logical_context_tokens: int
    resident_prefix_layers: int
    local_shard_bytes: tuple[int, ...]
    remote_bytes: int
    last_completion_ns: Optional[int]
    active: bool


@dataclass
class SessionKVState:
    session_id: str
    location: str
    instance_index: Optional[int]
    logical_context_tokens: int
    # Full L-layer per-rank KV vector for the session's logical context.  The
    # two-state model keeps ``resident_prefix_layers`` at either
    # ``model_layers`` (LOCAL_HBM: the local share is the whole vector) or 0
    # (REMOTE_MEMORY: the local share is the zero vector).
    shard_bytes: tuple[int, ...]
    resident_prefix_layers: int
    # Remote-pool account: per-rank bytes stored from remote_source_instance_
    # index's ranks (the instance the session was evicted from).  Zero vector
    # whenever the session has no remote share.
    remote_source_instance_index: Optional[int] = None
    remote_shard_bytes: tuple[int, ...] = ()
    last_completion_ns: Optional[int] = None
    active: bool = False
    last_request_id: Optional[str] = None


@dataclass(frozen=True)
class KVTransferShard:
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
    """Tiered KV move (the single cross-batch public transfer face).

    ``kind`` whitelist: ``local_hit / noc_migrate / remote_load / remote_store``
    (contract B0 §4); shard layer ranges must match the transfer's layer
    domain and non-local kinds must be byte-conservative with ``total_bytes``.
    """

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
class CapacityResult:
    """Reclamation outcome; ``evictions`` are the remote_store KVTransfers."""

    evictions: tuple[KVTransfer, ...]
    admitted: bool
    # 逐 rank 深度诊断(deep-gap 定位用),决策边界后仅测试断言消费,
    # 生产链不读(2026-09-30 审查登记:写后不读,保留作诊断信息;跨仓
    # 序列化裁决不在本仓做)。
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
    # 2026-10-01 审查登记:写后不读(完备性字段)——决策边界后生产链与
    # 测试均零读者(消费方只读 action/evictions/transfers/
    # source_instance_index/admission_blocked),保留仅令结果记录类型
    # 完备;删除会改变结果记录形状,故登记而不删。
    target_instance_index: int
    # 2026-10-01 审查登记:写后不读(完备性字段)——运行时历史 token 数
    # 由调用方自带(record["history_tokens_before"]),本字段零读者,
    # 处置口径同 target_instance_index。
    history_tokens: int
    transfers: tuple[KVTransfer, ...]
    evictions: tuple[KVTransfer, ...]
    admission_blocked: bool = False
    # 逐 rank 深度诊断,决策边界后仅测试断言消费(同 CapacityResult 注)。
    insufficient_ranks: tuple[int, ...] = ()

    @property
    def transfer_bytes(self) -> int:
        return sum(transfer.total_bytes for transfer in self.transfers)


@dataclass(frozen=True)
class MoveDecision:
    # 2026-10-01 审查登记:action / target_instance_index 均为写后不读
    # (完备性字段)——决策边界后生产链与测试均零读者(消费方只读
    # transfer/evictions/admission_blocked/source_instance_index),保留
    # 仅令结果记录类型完备;blocked 分支里 action 只是 empty_transfer
    # 同级的惰性占位,真实语义在 admission_blocked。
    action: str
    source_instance_index: int
    target_instance_index: int
    transfer: KVTransfer
    evictions: tuple[KVTransfer, ...]
    admission_blocked: bool = False
    # 逐 rank 深度诊断,决策边界后仅测试断言消费(同 CapacityResult 注)。
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
    """Exact TP-sharded tiered session KV state machine.

    All mutating operations are planner-time operations.  The caller emits the
    corresponding NoC / remote-pool transfer ET nodes from the returned
    decisions; capacity reclamation reports its ``remote_store`` transfers so
    nothing is silently dropped from the ledger.
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
        self.model_layers = model.layers
        self.model_weight_bytes_by_tp_rank = model_weight_shard_bytes_by_tp_rank(
            model, self.tp_degree
        )

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
            _require_valid_rank(topology.hardware, rank)
        self.edge_ranks = normalized_edges

        self._rank_states: dict[int, NodeHBMState] = {}
        for instance in instances:
            if len(instance.ranks) != self.tp_degree:
                raise ValueError("inconsistent TP degree in topology")
            for relative_rank, rank in enumerate(instance.ranks):
                # "every NPU exactly once"纪律(对齐 loader 侧拓扑校验):
                # 跨实例重复 rank 会让后建的实例状态静默覆盖先建状态
                # (last-writer-wins),污染逐 rank 账本与守恒断言基准,
                # 这里 fail-closed 而非静默吞掉。
                if rank in self._rank_states:
                    raise ValueError(
                        f"rank {rank} appears in more than one TP instance; "
                        "each NPU must belong to exactly one instance"
                    )
                state = NodeHBMState(
                    rank=rank,
                    instance_index=instance.index,
                    capacity_bytes=topology.hardware.local_hbm_capacity_bytes,
                    model_weight_bytes=self.model_weight_bytes_by_tp_rank[relative_rank],
                )
                self._rank_states[rank] = state
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
        # D4 (2026-09-05): 被动逐出深缺口计数——ensure_physical_fit 整会话
        # LRU 候选耗尽(活跃会话占满、无可逐冷会话)优雅推迟 admitted=False
        # 时递增,由 online_service summary 行(kv_deep_gap_events)落盘观测。
        # wscllm 形态维持 int 计数 + KVCacheEvent(不搬 sh 的结构化台账)。
        self.deep_gap_events: int = 0
        self._strict_kv_invariants = (
            _strict_kv_invariants_from_environment()
            if strict_invariants is None
            else strict_invariants
        )
        # Remote-pool per-rank account (bytes stored from that rank into the
        # shared pool); mirrored into the journal as the remote column.
        # Initialized before the first invariant audit below.
        self._remote_bytes_by_rank: dict[int, int] = {
            rank: 0 for rank in self._rank_states
        }
        self._check_invariants()
        self._initialize_incremental_invariants()
        # Metrics observation state: resident KV is tracked as per-session
        # parts that carry their own token counts and layer ranges, so
        # whole-session evictions and restores always remove the exact
        # recorded distribution of an earlier add under its own allocation
        # key (blueprint sh_2.0 仓 ``face_scheduler.py``（按函数名对应）).
        self._metrics_recorder = _METRICS_RECORDER
        self._metrics_parts: dict[str, list[dict[str, Any]]] = {}
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
        # test-only 读面(2026-09-30 审查登记):生产事件流走 events_since
        # 增量切片;本 property 仅测试探针消费。
        return tuple(self._events)

    # B1 增量读取 + 水位压缩(2026-08-28):生产消费方(在线调度器的
    # kv_actions 流)本就持有 _kv_events_emitted 游标,改走 events_since
    # 增量切片,消除每决策批 events property 的全量 tuple() 拷贝(O(n²)
    # CPU 残留);已消费前缀按水位整段删除(本 manager 的 compact_events
    # 唯一调用方在游标追平末端后调用,故 ≥8192 即整表清空,均摊 O(1)/
    # 事件;风格源头是 graph_batch_builder 的 M1 压缩)。event_index
    # 由单调计数器分配,压缩不改变任何已发射事件的载荷。
    def events_since(self, index: int) -> list[KVCacheEvent]:
        """B1:列表位置 index 起的增量事件(消费方维护游标)。"""
        return self._events[index:]

    def compact_events(self, consumed: int) -> int:
        """B1:已消费水位 ≥8192 时整段删除前缀,返回实际删除条数(调用方
        游标同步减去该值;0 = 未压缩)。唯一调用方在发射游标追平列表末端
        之后才调用,consumed 恒等于现存总量,故 ≥8192 即整表清空(均摊
        O(1)/事件);无"不小于现存一半"的二次门槛(改前该条件在唯一调用
        方下恒真,属死条件,已删)。"""
        if consumed >= _EVENTS_COMPACT_THRESHOLD:
            del self._events[:consumed]
            return consumed
        return 0

    @property
    def session_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._sessions))

    @property
    def node_states(self) -> tuple[NodeHBMState, ...]:
        # test-only 读面(2026-09-30 审查登记):仅单测消费,生产链不读;
        # 保留作测试探针,不对外承诺。
        return tuple(self._rank_states[rank] for rank in sorted(self._rank_states))

    def session_snapshot(self, session_id: str) -> Optional[SessionKVSnapshot]:
        state = self._sessions.get(session_id)
        if state is None:
            return None
        local_shard_bytes = self._local_prefix_shards(state)
        return SessionKVSnapshot(
            session_id=state.session_id,
            location=state.location,
            instance_index=state.instance_index,
            logical_context_tokens=state.logical_context_tokens,
            resident_prefix_layers=state.resident_prefix_layers,
            local_shard_bytes=local_shard_bytes,
            remote_bytes=sum(self._remote_shards(state)),
            last_completion_ns=state.last_completion_ns,
            active=state.active,
        )

    def hbm_snapshots(self, instance_index: Optional[int] = None) -> tuple[NodeHBMSnapshot, ...]:
        # test-only 读面(2026-09-30 审查登记):仅单测消费,生产链不读;
        # 保留作测试探针,不对外承诺。
        if instance_index is None:
            ranks = tuple(sorted(self._rank_states))
        else:
            ranks = tuple(self.topology.instance(instance_index).ranks)
        return tuple(self._rank_states[rank].snapshot() for rank in ranks)

    def remote_bytes_by_rank(self) -> dict[int, int]:
        """Read-only view of the per-rank remote-pool account.

        test-only 读面(2026-09-30 审查登记):生产审计走 journal 重放
        与 _check_invariants 的逐 rank 重算;本视图仅测试探针消费。"""
        return dict(self._remote_bytes_by_rank)

    def nearest_edge(self, rank: int) -> int:
        return nearest_edge_rank(self.topology.hardware, rank, self.edge_ranks)

    # ------------------------------------------------------------- helpers --

    def _local_prefix_shards(self, state: SessionKVState) -> tuple[int, ...]:
        return kv_cache_shard_bytes_for_layer_range(
            self.model,
            state.logical_context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=state.resident_prefix_layers,
        )

    def _remote_shards(self, state: SessionKVState) -> tuple[int, ...]:
        if state.remote_source_instance_index is None:
            return tuple(0 for _ in range(self.tp_degree))
        return tuple(state.remote_shard_bytes)

    def _layer_shards(self, tokens: int, layer_start: int, layer_end: int) -> tuple[int, ...]:
        return kv_cache_shard_bytes_for_layer_range(
            self.model, tokens, self.tp_degree,
            layer_start=layer_start, layer_end=layer_end,
        )

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

    @staticmethod
    def _fifo_sort(candidates: list[SessionKVState]) -> list[SessionKVState]:
        """(last_completion_ns, session_id) 升序 = 纯 LRU(去类型化)。

        last_completion_ns 只经 mark_complete 写入(非负 int、会话内非
        递减),此处不做 int() 静默收编——坏类型直接在比较处暴露。"""
        candidates.sort(
            key=lambda session: (
                session.last_completion_ns,
                session.session_id,
            )
        )
        return candidates

    def _completed_local_candidates(
        self,
        instance_index: int,
    ) -> list[SessionKVState]:
        """Whole-session LRU pool: completed inactive fully-local sessions,
        ordered ``(last_completion_ns, session_id)`` ascending = pure LRU
        with a deterministic tie-break)."""
        return self._fifo_sort([
            session
            for session in self._sessions.values()
            if session.location == LOCAL_HBM
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

    # ------------------------------------------------- remote-pool ledger --

    def _remote_add(
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
        """Credit the remote pool with bytes stored from ``instance_index``."""
        instance = self.topology.instance(instance_index)
        # A10(2026-09-30):与 _add_shards/_remove_shards 同款长度/符号门
        # ——旧实现无校验,zip 不齐会静默截断(M1-07 模式在 remote 账本
        # 的残余面);当前全部调用方都传 tp 长度向量,属纵深防御。
        if len(shards) != self.tp_degree or any(value < 0 for value in shards):
            raise ValueError("invalid remote-pool KV shard vector")
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            self._remote_bytes_by_rank[rank] += int(value)
            if self._metrics_recorder is not None:
                self._metrics_recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    instance_index=instance_index,
                    allocation_key=f"remote:{session_id}",
                    remote_delta_bytes=int(value),
                    cause=cause,
                )

    def _remote_remove(
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
        """Debit the remote pool for bytes restored / written off."""
        instance = self.topology.instance(instance_index)
        # A10(2026-09-30):同 _remote_add 的长度/符号门。
        if len(shards) != self.tp_degree or any(value < 0 for value in shards):
            raise ValueError("invalid remote-pool KV shard vector")
        for rank, value in zip(instance.ranks, shards):
            if not value:
                continue
            if self._remote_bytes_by_rank[rank] < value:
                raise RuntimeError(
                    f"remote-pool accounting underflow on rank {rank}"
                )
            self._remote_bytes_by_rank[rank] -= int(value)
            if self._metrics_recorder is not None:
                self._metrics_recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind=anchor_kind,
                    request_id=request_id,
                    session_id=session_id,
                    rank=rank,
                    instance_index=instance_index,
                    allocation_key=f"remote:{session_id}",
                    remote_delta_bytes=-int(value),
                    cause=cause,
                )

    # ------------------------------------------------------------------
    # Incremental invariant ledger.  The complete checker below remains the
    # source of truth for construction, explicit strict mode, and the terminal
    # audit.  Normal mutation paths update only the contribution(s) they
    # changed, which avoids rebuilding process-wide session/reservation totals
    # after every generated token.  B2: extended to the two-state local
    # session ledger plus the per-rank remote-pool ledger.
    # ------------------------------------------------------------------

    def _initialize_incremental_invariants(self) -> None:
        self._invariant_expected_resident_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_expected_reserved_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_expected_remote_by_rank = {
            rank: 0 for rank in self._rank_states
        }
        self._invariant_session_contributions: dict[
            str, Optional[tuple[int, tuple[int, ...]]]
        ] = {}
        self._invariant_session_remote_contributions: dict[
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
            session.location != LOCAL_HBM
            or session.instance_index is None
            or session.resident_prefix_layers != self.model_layers
            or len(session.shard_bytes) != self.tp_degree
        ):
            return None
        local_shards = kv_cache_shard_bytes_for_layer_range(
            self.model,
            session.logical_context_tokens,
            self.tp_degree,
            layer_start=0,
            layer_end=self.model_layers,
        )
        return session.instance_index, tuple(int(value) for value in local_shards)

    def _incremental_session_remote_contribution(
        self, session: SessionKVState
    ) -> Optional[tuple[int, tuple[int, ...]]]:
        if session.remote_source_instance_index is None:
            return None
        if not any(session.remote_shard_bytes):
            return None
        if len(session.remote_shard_bytes) != self.tp_degree:
            return None
        return (
            session.remote_source_instance_index,
            tuple(int(value) for value in session.remote_shard_bytes),
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
            old_remote = self._invariant_session_remote_contributions.pop(
                session_id, None
            )
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_remote_by_rank, old_remote, -1
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
            new_remote = self._incremental_session_remote_contribution(session)
            self._invariant_session_remote_contributions[session_id] = new_remote
            affected_ranks.update(
                self._apply_incremental_contribution(
                    self._invariant_expected_remote_by_rank, new_remote, 1
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
        if session.location not in {LOCAL_HBM, REMOTE_MEMORY}:
            # PARTIAL_HBM_REMOTE is a legacy parsing-only value; a runtime
            # session can never carry it (whole-session reclamation only).
            raise RuntimeError(f"invalid KV location for {session.session_id}")
        if session.location == REMOTE_MEMORY:
            if session.instance_index is not None:
                raise RuntimeError("remote KV session retained a local instance")
            if session.resident_prefix_layers != 0:
                raise RuntimeError("remote KV session retained resident layers")
        else:
            if session.resident_prefix_layers != self.model_layers:
                raise RuntimeError("fully local KV session is missing layers")
            if session.instance_index is None:
                raise RuntimeError("local KV session has no instance")
        if len(session.shard_bytes) != self.tp_degree:
            raise RuntimeError("KV shard count does not match TP instance")
        local_shards = self._local_prefix_shards(session)
        remote_shards = self._remote_shards(session)
        if (
            len(local_shards) != self.tp_degree
            or len(remote_shards) != self.tp_degree
        ):
            # zip 对长度不齐会静默截断守恒断言(M1-07):长度必须在此
            # 显式断言失败(shard_bytes 长度已在上方检查)。
            raise RuntimeError("KV shard count does not match TP instance")
        for local_bytes, remote_bytes, total_bytes in zip(
            local_shards, remote_shards, session.shard_bytes
        ):
            if local_bytes + remote_bytes != total_bytes:
                raise RuntimeError(
                    f"local+remote split does not preserve total KV bytes for "
                    f"{session.session_id}"
                )

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
            if self._remote_bytes_by_rank[rank] < 0:
                raise RuntimeError("negative remote-pool accounting")
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
            for ledger, actual_of in (
                (
                    self._invariant_expected_resident_by_rank,
                    lambda rank: self._rank_states[rank].resident_kv_bytes,
                ),
                (
                    self._invariant_expected_reserved_by_rank,
                    lambda rank: self._rank_states[rank].reserved_request_bytes,
                ),
                (
                    self._invariant_expected_remote_by_rank,
                    lambda rank: self._remote_bytes_by_rank[rank],
                ),
            ):
                actual = tuple(actual_of(rank) for rank in instance.ranks)
                expected = tuple(ledger[rank] for rank in instance.ranks)
                if actual != expected:
                    raise RuntimeError(
                        f"session/rank KV accounting mismatch for instance "
                        f"{instance_index}: states={actual}, sessions={expected}"
                    )

        if self._strict_kv_invariants:
            self._check_invariants()

    # ------------------------------------------------------------------
    # Read-only metrics observation helpers (doc sec.7).  Every method here
    # is a no-op unless a recorder was installed at construction time, and
    # none of them feeds back into manager decisions.  Resident KV is tracked
    # as per-session parts that carry their own token count and layer range;
    # because kv_cache_shard_bytes_for_layer_range is exactly linear in
    # tokens, whole-session evictions and restores remove precisely the
    # distribution an earlier add recorded under the same allocation key
    # (blueprint sh_2.0 仓 ``face_scheduler.py``（按函数名对应）).
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
                instance_index=instance_index,
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
                instance_index=part["instance_index"],
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
        magnitude, so the projection stays exactly removable), then each
        original part is removed from the source ranks under its own key.
        Target add precedes source release, matching the dynamic transfer
        dependency ordering."""

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

    # ------------------------------------------------ transfer constructors --
    # (blueprint sh_2.0 仓 ``face_scheduler.py``（按函数名对应）, adapted to the
    # wscllm topology/hardware types and the shared _xy_route)

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
            total_bytes=sum(session.shard_bytes),
            shards=(),
            model_layers=self.model_layers,
            layer_start=0,
            layer_end=self.model_layers,
            resident_prefix_layers_before=self.model_layers,
            resident_prefix_layers_after=self.model_layers,
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
        if session.resident_prefix_layers != self.model_layers:
            raise RuntimeError("NoC migration requires a fully resident session")
        source = self.topology.instance(source_instance_index)
        target = self.topology.instance(target_instance_index)
        shards = tuple(
            KVTransferShard(
                source_rank=source_rank,
                target_rank=target_rank,
                edge_rank=None,
                bytes=shard_bytes,
                noc_path=_deterministic_xy_route(
                    self.topology.hardware, source_rank, target_rank
                ),
                layer_start=0,
                layer_end=self.model_layers,
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
            total_bytes=sum(session.shard_bytes),
            shards=shards,
            model_layers=self.model_layers,
            layer_start=0,
            layer_end=self.model_layers,
            resident_prefix_layers_before=self.model_layers,
            resident_prefix_layers_after=self.model_layers,
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
                    noc_path=_deterministic_xy_route(
                        self.topology.hardware, edge, target_rank
                    ),
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
            model_layers=self.model_layers,
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
                    noc_path=_deterministic_xy_route(
                        self.topology.hardware, source_rank, edge
                    ),
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
            model_layers=self.model_layers,
            layer_start=layer_start,
            layer_end=layer_end,
            resident_prefix_layers_before=session.resident_prefix_layers,
            resident_prefix_layers_after=resident_prefix_layers_after,
        )

    # ------------------------------------------------------- eviction family --

    def _evict_session(
        self,
        session: SessionKVState,
        *,
        now_ns: int,
        phase: str,
        reason: str,
        trigger_request_id: str,
    ) -> KVTransfer:
        """Whole-session reclamation (the single eviction path): offload all
        ``L`` layers of a completed inactive LOCAL session to the remote pool
        (-> REMOTE), then release its entire local residency."""

        if session.active or session.last_completion_ns is None:
            raise RuntimeError("only completed inactive sessions may be evicted")
        if session.location != LOCAL_HBM or session.instance_index is None:
            raise RuntimeError("only fully local sessions may be evicted")
        resident_layers = self.model_layers
        if session.resident_prefix_layers != resident_layers:
            raise RuntimeError("local session is missing layers")
        source_instance = session.instance_index
        cause = f"evict_{reason}_session:layers0-{self.model_layers}"
        session_shards = self._layer_shards(
            session.logical_context_tokens, 0, resident_layers
        )
        transfer = self._remote_store_transfer(
            phase=phase,
            reason=f"{reason}_session",
            session=session,
            trigger_request_id=trigger_request_id,
            layer_start=0,
            layer_end=resident_layers,
            resident_prefix_layers_after=0,
        )
        before = self._remaining(source_instance)
        self._remove_shards(source_instance, session_shards)
        prior_remote = self._remote_shards(session)
        session.location = REMOTE_MEMORY
        session.instance_index = None
        session.resident_prefix_layers = 0
        session.remote_source_instance_index = source_instance
        session.remote_shard_bytes = tuple(
            old + new for old, new in zip(prior_remote, session_shards)
        )
        self._remote_add(
            session.session_id,
            source_instance,
            session_shards,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=cause,
        )
        self._check_invariants_after_mutation(session_ids=(session.session_id,))
        self._metrics_remove_session_parts(
            session.session_id,
            now_ns=now_ns,
            anchor_kind=_metrics_anchor_for_phase(phase),
            request_id=trigger_request_id,
            cause=cause,
        )
        after = self._remaining(source_instance)
        self._event(
            now_ns=now_ns,
            phase=phase,
            event_type="evict_session",
            reason=cause,
            trigger_request_id=trigger_request_id,
            session_id=session.session_id,
            source_instance_index=source_instance,
            context_tokens=session.logical_context_tokens,
            total_bytes=sum(session_shards),
            shard_bytes=session_shards,
            last_completion_ns=session.last_completion_ns,
            before=before,
            after=after,
        )
        return transfer

    # ------------------------------------------------------------------
    # P1 journal 事务对账与 run 末 checksum 门。两者都只在 recorder 处于
    # journal 模式时生效,且都是纯观测校验:任何不一致 raise(fail-closed),
    # 不做任何静默修复/钳位——发现账本 bug 正是 journal 的价值。
    # B2:对账元组扩为 (weight, resident, reserved, remote)。
    # ------------------------------------------------------------------

    def _journal_reconcile_transaction(
        self, transaction_id: int, ranks: Iterable[int]
    ) -> None:
        """④ commit 点对账:manager 实际状态 vs journal 记账逐 rank 比对。

        journal 每行的 before/after 由 recorder 的逐 rank 运行计数推出,
        该计数 = 全部已写行 Σdelta;故此处比对等价于"该事务 before+Σdelta
        推出的 after vs manager 快照"。只查本事务写过的 rank(未写行的
        rank 无字节变化,不可能漂移);weight 由增量不变量不覆盖,此处
        一并核对(journal 对 weight 同样记账)。remote 列为 B2 新增:远端
        池账面按"从该 rank 存入的字节"逐 rank 记账,与会话 remote 账本
        同源守恒。"""

        recorder = self._metrics_recorder
        mismatches = []
        for rank in sorted(set(ranks)):
            state = self._rank_states[rank]
            journal_totals = recorder.rank_totals(rank)
            manager_totals = (
                state.model_weight_bytes,
                state.resident_kv_bytes,
                state.reserved_request_bytes,
                self._remote_bytes_by_rank[rank],
            )
            if journal_totals != manager_totals:
                mismatches.append(
                    f"rank {rank}: manager=(weight={manager_totals[0]}, "
                    f"resident={manager_totals[1]}, reserved={manager_totals[2]}, "
                    f"remote={manager_totals[3]}), "
                    f"journal=(weight={journal_totals[0]}, "
                    f"resident={journal_totals[1]}, "
                    f"reserved={journal_totals[2]}, "
                    f"remote={journal_totals[3]})"
                )
        if mismatches:
            raise RuntimeError(
                "KV delta journal reconciliation failed for transaction "
                f"{transaction_id}: " + "; ".join(mismatches)
            )

    def _journal_rollback_snapshot(self) -> dict[str, Any]:
        """事务入口的 manager 全量可变状态快照(仅 journal 模式最外层)。

        只拷贝容器与可变 dataclass 实例(shard 向量等字段均为不可变
        tuple,浅拷字段即可);_events 事务内只追加,记长度即可。"""

        return {
            "rank_states": {
                rank: replace(state) for rank, state in self._rank_states.items()
            },
            "sessions": {
                session_id: replace(state)
                for session_id, state in self._sessions.items()
            },
            "reservations": dict(self._reservations),
            "event_count": len(self._events),
            "event_index_next": self._event_index_next,
            "deep_gap_events": self.deep_gap_events,
            "pressure_event_keys": set(self._pressure_event_keys),
            "pressure_event_keys_by_request": {
                request_id: set(keys)
                for request_id, keys in self._pressure_event_keys_by_request.items()
            },
            "remote_bytes_by_rank": dict(self._remote_bytes_by_rank),
            "expected_resident_by_rank": dict(
                self._invariant_expected_resident_by_rank
            ),
            "expected_reserved_by_rank": dict(
                self._invariant_expected_reserved_by_rank
            ),
            "expected_remote_by_rank": dict(
                self._invariant_expected_remote_by_rank
            ),
            "session_contributions": dict(self._invariant_session_contributions),
            "session_remote_contributions": dict(
                self._invariant_session_remote_contributions
            ),
            "reservation_contributions": dict(
                self._invariant_reservation_contributions
            ),
            "metrics_parts": {
                session_id: [dict(part) for part in parts]
                for session_id, parts in self._metrics_parts.items()
            },
            "metrics_segment_counters": dict(self._metrics_segment_counters),
        }

    def _restore_journal_rollback_snapshot(self, snapshot: dict[str, Any]) -> None:
        """失败路径:manager 整体回到事务入口快照(每事务至多一次)。"""

        if len(self._events) < snapshot["event_count"]:
            # 事务内 _events 只追加(消费方压缩发生在事务之外);跌破
            # 快照长度说明追加式不变量被破坏,fail-closed。
            raise RuntimeError(
                "journal rollback: event log shrank mid-transaction"
            )
        self._rank_states = {
            rank: replace(state) for rank, state in snapshot["rank_states"].items()
        }
        self._sessions = {
            session_id: replace(state)
            for session_id, state in snapshot["sessions"].items()
        }
        self._reservations = dict(snapshot["reservations"])
        del self._events[snapshot["event_count"]:]
        self._event_index_next = snapshot["event_index_next"]
        self.deep_gap_events = snapshot["deep_gap_events"]
        self._pressure_event_keys = set(snapshot["pressure_event_keys"])
        self._pressure_event_keys_by_request = {
            request_id: set(keys)
            for request_id, keys in snapshot["pressure_event_keys_by_request"].items()
        }
        self._remote_bytes_by_rank = dict(snapshot["remote_bytes_by_rank"])
        self._invariant_expected_resident_by_rank = dict(
            snapshot["expected_resident_by_rank"]
        )
        self._invariant_expected_reserved_by_rank = dict(
            snapshot["expected_reserved_by_rank"]
        )
        self._invariant_expected_remote_by_rank = dict(
            snapshot["expected_remote_by_rank"]
        )
        self._invariant_session_contributions = dict(
            snapshot["session_contributions"]
        )
        self._invariant_session_remote_contributions = dict(
            snapshot["session_remote_contributions"]
        )
        self._invariant_reservation_contributions = dict(
            snapshot["reservation_contributions"]
        )
        self._metrics_parts = {
            session_id: [dict(part) for part in parts]
            for session_id, parts in snapshot["metrics_parts"].items()
        }
        self._metrics_segment_counters = dict(
            snapshot["metrics_segment_counters"]
        )

    def verify_journal_checksum(self) -> Optional[dict[str, Any]]:
        """Run 末 checksum 门(fail-closed):流式重放权威 journal 并对账。

        三重断言,任一不成立 raise(错误信息含逐 rank 差额表):
        1. journal 重放逐 rank 终态(weight/resident/reserved/remote)与
           manager 快照完全一致(重放本身先校验 sequence 连续/前后快照自洽/
           planner_time_ns 与 transaction_id 单调/断行报行号);
        2. 写出条数一致(recorder 内存计数 vs 重放行数,流式丢失即暴露);
        3. 守恒(B2 三列账本口径,resident/reserved/remote——位置模型
           恒二态,见模块头):本地 resident==0、reserved==0、
           physical==weight(权重驻留、无会话/预约残留),且 remote==0
           ——retire_terminal_session 对远端账面静默核销后,远端池账本与
           会话账本同归零,"remote 字节与会话账本守恒"。

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
                    f"reserved={state.reserved_request_bytes}, "
                    f"remote={self._remote_bytes_by_rank[rank]}))"
                )
                continue
            manager_totals = (
                state.model_weight_bytes,
                state.resident_kv_bytes,
                state.reserved_request_bytes,
                self._remote_bytes_by_rank[rank],
            )
            journal_totals = (
                replayed["weight"],
                replayed["resident"],
                replayed["reserved"],
                replayed["remote"],
            )
            if journal_totals != manager_totals:
                problems.append(
                    f"rank {rank}: manager=(weight={manager_totals[0]}, "
                    f"resident={manager_totals[1]}, "
                    f"reserved={manager_totals[2]}, "
                    f"remote={manager_totals[3]}), "
                    f"journal_replay=(weight={journal_totals[0]}, "
                    f"resident={journal_totals[1]}, "
                    f"reserved={journal_totals[2]}, "
                    f"remote={journal_totals[3]}), "
                    f"delta=(weight={manager_totals[0] - journal_totals[0]}, "
                    f"resident={manager_totals[1] - journal_totals[1]}, "
                    f"reserved={manager_totals[2] - journal_totals[2]}, "
                    f"remote={manager_totals[3] - journal_totals[3]})"
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
        # 守恒断言(B2 重定义):本地三列归零只剩权重;remote 列与会话
        # 账本守恒(retire 静默核销后同为 0)。若真实 workload 允许会话
        # KV/远端驻留终态,这里会 raise——保留 fail-closed,不放宽门。
        conservation: list[str] = []
        for rank in sorted(replay["ranks"]):
            replayed = replay["ranks"][rank]
            if (
                replayed["resident"] != 0
                or replayed["reserved"] != 0
                or replayed["physical"] != replayed["weight"]
                or replayed["remote"] != 0
            ):
                conservation.append(
                    f"rank {rank}: resident={replayed['resident']}, "
                    f"reserved={replayed['reserved']}, "
                    f"physical={replayed['physical']}, "
                    f"weight={replayed['weight']}, "
                    f"remote={replayed['remote']}"
                )
        if conservation:
            raise RuntimeError(
                "KV delta journal conservation violated at run end "
                "(local resident must be 0, reserved must be 0, physical "
                "must equal weight, and the remote pool account must be "
                "zero after terminal write-off): " + "; ".join(conservation)
            )
        summary = {
            "schema_version": recorder.JOURNAL_SCHEMA_VERSION,
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
                "remote_account_zero": True,
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

    # ------------------------------------------------------------ capacity --

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
        """Single-stage whole-session LRU reclamation (wscllm semantics).

        Candidates are the completed inactive fully-local sessions of the
        instance in ``(last_completion_ns, session_id)`` order; each selected
        victim is evicted whole (all layers, all TP shards) and the shortfall
        is rechecked after every single eviction.  Over-release is allowed:
        the loop stops at the first eviction after which the requirement
        fits, never at an exact byte boundary.  The eviction loop is entered
        only when the requirement could still fit once every reclaimable
        session is evicted.  Outstanding request reservations
        (``reserved_request_bytes``) are not reclaimable by eviction and
        count into that judgment alongside the non-candidate residency.
        Callers pass the gross requirement on the
        first attempt and a vector net of the protected sessions' own
        residency on the failure retry; because neither that residency nor
        the co-resident active sessions' residency is ever a candidate, and
        both are counted into the margin judgment below, a gross first
        attempt that could not fit alongside them runs the graceful
        exhaustion path without evicting anything.  Exhaustion keeps the
        graceful wscllm failure semantics:
        ``deep_gap_events`` counter + bounded pressure event +
        ``admitted=False`` — never the sh blueprint's raise.  Two contract
        violations still raise, and neither is capacity exhaustion: a shard
        vector that cannot fit an empty instance raises ``ValueError`` (no
        reclamation could ever admit it), and a loop that returns success
        while the requirement still does not fit raises ``RuntimeError``
        (internal accounting bug).
        """

        required = _require_int_vector(required_shards, "required KV shard vector")
        instance = self.topology.instance(instance_index)
        if len(required) != self.tp_degree or any(value < 0 for value in required):
            raise ValueError("required KV shard vector is invalid")
        # Every protected session is excluded from the candidate pool; the
        # LRU order is unchanged for the single-session callers
        # (prepare_history / reserve / grow / move).
        protected = set(protected_sessions)
        # Residency that can never be freed by this reclamation is counted
        # into the margin judgment: the protected sessions' own residency
        # plus every active session on this instance (active sessions are
        # never candidates, so their residency survives the loop).  Callers
        # pass the gross requirement on the first attempt and net the
        # protected residency out of ``required`` only on the failure retry
        # — a requirement that cannot fit alongside the non-candidate
        # residency goes straight to the graceful exhaustion path instead of
        # evicting candidates it could never admit with
        # (M1-10: 无效逐出循环).  Active-and-protected sessions are counted
        # exactly once (the two sets are united, not intersected twice).
        noncandidate_resident = [0] * self.tp_degree
        for session in self._sessions.values():
            if (
                session.location == LOCAL_HBM
                and session.instance_index == instance_index
                and (session.active or session.session_id in protected)
            ):
                for index, value in enumerate(self._local_prefix_shards(session)):
                    noncandidate_resident[index] += value
        maximum = tuple(
            self._rank_states[rank].capacity_bytes - self._rank_states[rank].model_weight_bytes
            for rank in instance.ranks
        )
        # Outstanding decode-capacity reservations are never reclaimable by
        # session eviction, so they count into the margin judgment exactly
        # like active/protected residency (per-rank state of THIS instance's
        # ranks, zipped in the same order as maximum/required).
        reserved = tuple(
            self._rank_states[rank].reserved_request_bytes
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
        evictions: list[KVTransfer] = []
        insufficient = self._insufficient(instance_index, required)
        # Single stage: evict complete sessions in deterministic LRU order
        # until the requirement fits (or the pool is exhausted).
        if insufficient and all(
            needed <= available - own - held
            for needed, available, own, held in zip(
                required, maximum, noncandidate_resident, reserved
            )
        ):
            candidates = [
                session
                for session in self._completed_local_candidates(instance_index)
                if session.session_id not in protected
            ]
            candidate_index = 0
            while insufficient:
                if candidate_index >= len(candidates):
                    break
                victim = candidates[candidate_index]
                candidate_index += 1
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
        # D4-I1 (2026-09-05; 2026-09-30 补全规格复核项):在 deep-gap 与
        # 成功两条返回路径之前统一复核每个受害者——非活跃、非保护会话、
        # 且已落 REMOTE(规格 §4.2"复查全部受害者确已非活跃、非保护会
        # 话";旧实现漏保护项/位置项,且 deep-gap 返回路径完全跳过复核;
        # 因边际门保证 blocked ⟹ evictions==(),deep-gap 路径复核当前
        # 恒空转;若该不变量未来被打破,此处复核恰是第一道兜底)。
        for transfer in evictions:
            evicted = self._sessions[transfer.session_id]
            if evicted.active or evicted.last_completion_ns is None:
                raise RuntimeError(
                    "passive eviction victim is active or uncompleted: "
                    f"{transfer.session_id}")
            if evicted.session_id in protected:
                raise RuntimeError(
                    "passive eviction victim is a protected session: "
                    f"{transfer.session_id}")
            if evicted.location != REMOTE_MEMORY:
                raise RuntimeError(
                    "passive eviction victim is not fully remote after "
                    f"eviction: {transfer.session_id}")
        if insufficient:
            before = self._remaining(instance_index)
            # D4 (2026-09-05): 被动逐出深缺口——整会话候选全耗尽(活跃会话
            # 占满、无可逐冷会话)优雅推迟 admitted=False;capacity_epoch 重试
            # 机制保留(无死锁风险),计数入 summary 行观测。口径注记(2026-
            # 09-29):计数按"每次 fit 尝试失败"递增、不去重——同一请求经
            # capacity_epoch/准入重试再次撞深缺口会重复计数,故该值语义是
            # 深缺口阻塞事件数而非被阻塞请求数(纯观测列,不进任何策略判据
            # 与守恒校验;改去重会改变 summary 行数值,非本列本意)。
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
        # 逐出收敛后必须满足本请求 required 的显式断言(整体逐出允许过量
        # 释放,不检查"恰好够")。
        if self._insufficient(instance_index, required):
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
        required = _require_int_vector(required_shards, "required KV shard vector")
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
        *,
        anchor_kind: str = "prefill_start",
    ) -> RequestCapacityReservation:
        """Release one request's decode-capacity reservation.

        ``now_ns`` 是本 mutation 的真实 planner 时刻(P1 语义修改①:改前
        本 API 无时间戳参数,metrics 复用最近 KV 事件时间——journal 化后
        该近似会污染权威账本时序,故改为必传参数,全部调用方显式给值;
        传入时刻若早于 journal 已有最后时刻,recorder 直接 raise)。
        ``anchor_kind`` 只标注 journal 行的审计边界列(默认 prefill_start
        = Prefill 准入边界;decode 准入边界的释放由调用方显式传
        decode_start),不影响任何逐出/配额行为。"""

        # 先查后改:全部 rank 校验通过才减账并注销预约——中途失败时预约
        # 原样保留,失败重试仍是同一笔已知预约(幂等),不会变成 unknown。
        # 未知预约与 extend_request_capacity 同族:带上下文 RuntimeError
        # (改前此处是裸 KeyError,与同族错误风格不一致)。
        reservation = self._reservations.get(request_id)
        if reservation is None:
            raise RuntimeError(f"unknown KV reservation for request {request_id}")
        ranks = self.topology.instance(reservation.instance_index).ranks
        for rank, value in zip(ranks, reservation.shard_bytes):
            if self._rank_states[rank].reserved_request_bytes < value:
                raise RuntimeError("request reservation accounting underflow")
        for rank, value in zip(ranks, reservation.shard_bytes):
            self._rank_states[rank].reserved_request_bytes -= value
        del self._reservations[request_id]
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
                    anchor_kind=anchor_kind,
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
        ``prepare_history`` 的迁移把旧 KV 从 decode 目标实例删掉后,
        必须把净额预约回补到全量——复刻原设计"从准入占位到 P→D move 完成"
        的防抢占语义。resident→reserved 1:1 换位:迁移删除
        的旧驻留字节与回补的预约字节相等,任何瞬间 weight+resident+
        reserved ≤ capacity 不超订。
        """

        reservation = self._reservations.get(request_id)
        if reservation is None:
            raise RuntimeError(f"unknown KV reservation for request {request_id}")
        extra = _require_int_vector(
            extra_shards, "reservation extension shard vector"
        )
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
        """Four-branch session recovery (session-level Tiered-LRU):

        no-history / LOCAL same (local_hit) / LOCAL cross (noc_migrate) /
        REMOTE (full remote_load = REMOTE_RESTORE, the only remote path).
        Capacity convergence for every branch runs through the single-stage
        whole-session ``ensure_physical_fit`` first; a blocked fit returns
        the graceful ``admission_blocked`` decision with whatever reclamation
        it already performed.
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

        def blocked(action: str, fit: CapacityResult) -> HistoryDecision:
            return HistoryDecision(
                action=action,
                source_instance_index=(None if state is None else state.instance_index),
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfers=(),
                evictions=fit.evictions,
                admission_blocked=True,
                insufficient_ranks=fit.insufficient_ranks,
            )

        # ---- branch 1: no history ----
        if state is None:
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
                return blocked(NO_HISTORY, fit)
            history_shards = self._history_shards(history_tokens)
            state = SessionKVState(
                session_id=session_id,
                location=LOCAL_HBM,
                instance_index=target_instance_index,
                logical_context_tokens=0,
                shard_bytes=tuple(0 for _ in history_shards),
                resident_prefix_layers=self.model_layers,
                active=True,
                last_request_id=trigger_request_id,
            )
            self._sessions[session_id] = state
            self._check_invariants_after_mutation(session_ids=(session_id,))
            before = self._remaining(target_instance_index)
            # no_history 本身不搬移任何 KV(新会话零字节落位):before/
            # after 只框定本事件,两者都取自上方 fit 逐出之后的余量——
            # 逐出动作已自带逐笔 evict_session 事件,不在此重复。
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
                transfers=(),
                evictions=fit.evictions,
            )

        if state.instance_index is None and state.location != REMOTE_MEMORY:
            raise RuntimeError("resident session is missing its instance")

        # Existing local bytes on the target (LOCAL same-instance: the full
        # vector; REMOTE/cross-instance: zero).
        if (
            state.location == LOCAL_HBM
            and state.instance_index == target_instance_index
        ):
            existing_target = self._local_prefix_shards(state)
        else:
            existing_target = tuple(0 for _ in desired)
        needed = tuple(want - current for want, current in zip(desired, existing_target))
        if any(value < 0 for value in needed):
            raise RuntimeError("session KV would shrink during history preparation")
        protected = (session_id,)

        # ---- branch 2: LOCAL, same instance ----
        # 防御分支(PD 分离拓扑生产不可达:准入目标恒为 prefill 实例而
        # 历史 LOCAL 驻留只出现在 decode 实例;2026-09-30 审查登记保留
        # 作单测替身路径),勿当死代码清除。
        if state.location == LOCAL_HBM and state.instance_index == target_instance_index:
            fit = self.ensure_physical_fit(
                target_instance_index,
                needed,
                now_ns,
                trigger_request_id,
                protected,
                phase=phase,
                reason="history_and_prefill_admission",
            )
            if not fit.admitted:
                return blocked(LOCAL_HIT, fit)
            source_instance = state.instance_index
            state.active = True
            state.last_request_id = trigger_request_id
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
                total_bytes=sum(state.shard_bytes),
                shard_bytes=state.shard_bytes,
                last_completion_ns=state.last_completion_ns,
                before=before,
                after=before,
            )
            return HistoryDecision(
                action=LOCAL_HIT,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                history_tokens=history_tokens,
                transfers=(),
                evictions=fit.evictions,
            )

        # ---- branch 3: LOCAL, cross instance ----
        if state.location == LOCAL_HBM:
            if state.instance_index is None:
                raise RuntimeError("local history has no source instance")
            source_instance = state.instance_index
            fit = self.ensure_physical_fit(
                target_instance_index,
                needed,
                now_ns,
                trigger_request_id,
                protected,
                phase=phase,
                reason="history_target_capacity",
            )
            if not fit.admitted:
                return blocked(NOC_MIGRATE, fit)
            history_shards = self._history_shards(history_tokens)
            state.active = True
            state.last_request_id = trigger_request_id
            transfer = self._noc_transfer(
                phase="history",
                reason="history_other_instance",
                session=state,
                trigger_request_id=trigger_request_id,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
            )
            before = self._remaining(target_instance_index)
            # The dynamic transfer contract admits target shards before
            # releasing the source; planner state advances atomically after
            # that admission.
            self._add_shards(target_instance_index, history_shards)
            self._remove_shards(source_instance, history_shards)
            state.instance_index = target_instance_index
            self._check_invariants_after_mutation(session_ids=(session_id,))
            self._metrics_move_session_parts(
                session_id,
                target_instance_index,
                history_tokens,
                self.model_layers,
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
                transfers=(transfer,),
                evictions=fit.evictions,
            )

        # ---- branch 4: REMOTE_MEMORY ----
        if state.location != REMOTE_MEMORY:
            # PARTIAL_HBM_REMOTE is a legacy parsing-only location; the
            # runtime can no longer hold a session there (whole-session
            # reclamation only), so an unknown value here is fail-closed.
            raise RuntimeError(f"unknown history location: {state.location}")
        remote_instance = state.remote_source_instance_index
        if remote_instance is None:
            raise RuntimeError("remote history has no remote account")
        remote_shards = self._remote_shards(state)
        fit = self.ensure_physical_fit(
            target_instance_index,
            needed,
            now_ns,
            trigger_request_id,
            protected,
            phase=phase,
            reason="history_target_capacity",
        )
        if not fit.admitted:
            return blocked(REMOTE_RESTORE, fit)
        history_shards = self._history_shards(history_tokens)
        state.active = True
        state.last_request_id = trigger_request_id
        transfer = self._remote_load_transfer(
            phase="history",
            reason="history_remote_restore",
            session=state,
            trigger_request_id=trigger_request_id,
            target_instance_index=target_instance_index,
            layer_start=0,
            layer_end=self.model_layers,
        )
        before = self._remaining(target_instance_index)
        self._add_shards(target_instance_index, history_shards)
        self._remote_remove(
            session_id,
            remote_instance,
            remote_shards,
            now_ns=now_ns,
            anchor_kind="transfer_complete",
            request_id=trigger_request_id,
            cause="history_remote_restore",
        )
        state.location = LOCAL_HBM
        state.instance_index = target_instance_index
        state.resident_prefix_layers = self.model_layers
        state.remote_source_instance_index = None
        state.remote_shard_bytes = tuple(0 for _ in history_shards)
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_add_segment(
            session_id,
            target_instance_index,
            history_tokens,
            0,
            self.model_layers,
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
            event_type="remote_restore",
            reason="history_remote_restore",
            trigger_request_id=trigger_request_id,
            session_id=session_id,
            source_instance_index=remote_instance,
            target_instance_index=target_instance_index,
            context_tokens=history_tokens,
            total_bytes=sum(history_shards),
            shard_bytes=history_shards,
            last_completion_ns=state.last_completion_ns,
            before=before,
            after=after,
        )
        return HistoryDecision(
            action=REMOTE_RESTORE,
            source_instance_index=None,
            target_instance_index=target_instance_index,
            history_tokens=history_tokens,
            transfers=(transfer,),
            evictions=fit.evictions,
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
        state.logical_context_tokens = context_tokens
        state.last_request_id = trigger_request_id
        self._check_invariants_after_mutation(session_ids=(session_id,))
        self._metrics_add_segment(
            session_id,
            instance_index,
            context_tokens - previous_tokens,
            0,
            self.model_layers,
            delta,
            now_ns=now_ns,
            # anchor 与逐出/预约同走中心映射(M1-02:decode 增长曾硬编码
            # completion,与 _metrics_anchor_for_phase 的 decode_start 冲突)。
            anchor_kind=_metrics_anchor_for_phase(phase),
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
        """Migrate a local prefill session onto the Decode instance.

        Contract: the session KV context cannot shrink through a move —
        ``final_context_tokens`` (defaulting to the current context) must be
        ``>=`` the current ``logical_context_tokens``.  This mirrors the
        ``_grow`` guard so a shrink request fails fast with a clean
        ``ValueError`` instead of slipping past the target-side fit check
        (which only sees the smaller final footprint on the cross-instance
        path) and blowing up later at placement time.
        """

        state = self._sessions[session_id]
        if state.location != LOCAL_HBM or state.instance_index is None or not state.active:
            raise RuntimeError("Prefill KV must be active and local before Decode")
        source_instance = state.instance_index
        final_tokens = (
            state.logical_context_tokens if final_context_tokens is None else final_context_tokens
        )
        # 与 _grow 同款"上下文不可收缩"守卫(2026-10-01 审查修正):跨实例
        # 路径的 fit 只按 shards(final) 判决,缺此守卫时收缩入参会在落位
        # 阶段才以超容 RuntimeError 晚爆。生产调用方恒传 final >= 当前
        # (final = prefill + decode,decode 段非负),本守卫纯防御。
        if final_tokens < state.logical_context_tokens:
            raise ValueError("session KV context cannot shrink")
        desired_final = self._history_shards(final_tokens)
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
            # Blocked move:``empty_transfer`` 只是让 MoveDecision 类型完备
            # 的惰性占位(local_hit 形态、零网络 shard、target 钉在被阻塞
            # 前的本地位)——它与 decision.action 的 NOC_MIGRATE 语义无关,
            # 不代表任何字节搬移。消费方必须把 admission_blocked=True 视为
            # "无迁移":在线调度器在读 transfer 之前即对该旗标 raise。
            empty_transfer = self._local_hit_transfer(
                phase="prefill_decode",
                reason="prefill_decode_local_reuse",
                session=state,
                trigger_request_id=trigger_request_id,
            )
            return MoveDecision(
                action=LOCAL_HIT if source_instance == target_instance_index else NOC_MIGRATE,
                source_instance_index=source_instance,
                target_instance_index=target_instance_index,
                transfer=empty_transfer,
                evictions=evictions,
                admission_blocked=True,
                insufficient_ranks=fit.insufficient_ranks,
            )
        # 防御分支(source==target,PD 分离拓扑生产不可达:move 恒从
        # prefill 实例迁往 decode 实例,两集合不相交;2026-09-30 审查登记
        # 保留作单测替身路径),勿当死代码清除。
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
            self.model_layers,
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
    ) -> tuple[()]:  # 完成期零逐出为既定架构:恒空元组,注解如实校准。
        # 完成时刻是 LRU 的唯一时间戳写入点:非负 int + 会话内非递减
        # (float/bool/回退时刻会改变完成判断与逐出顺序,fail-closed)。
        _require_nonnegative_int(completion_ns, "completion_ns")
        state = self._sessions[session_id]
        if state.location != LOCAL_HBM or state.instance_index is None or not state.active:
            raise RuntimeError("only active local sessions may complete")
        if (
            state.last_completion_ns is not None
            and completion_ns < state.last_completion_ns
        ):
            raise ValueError(
                f"completion time {completion_ns} moves backwards for session "
                f"{session_id} (last completion {state.last_completion_ns})"
            )
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
        that reads the completion snapshot.  Two-state extension: only the
        fully-local residency decrements local HBM; a remote share is
        silently written off (layer-domain decrement + remote write-off, not
        an eviction, no transfer emitted).
        """

        _require_nonnegative_int(completion_ns, "completion_ns")
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
        # matching entries in-transaction under the terminal-retire cause
        # (contract §5: terminal_session_retire_reservation) rather than
        # retaining a tombstone; never touch another session's reservation.
        reservation_ids = sorted(
            reservation_id
            for reservation_id, reservation in self._reservations.items()
            if reservation.session_id == session_id
        )
        # 先查后改(对齐 release_request_capacity 的两段式纪律):全部预约
        # 的逐 rank 校验都通过后才统一减账并注销预约——中途失败时预约
        # 原样保留、任何 rank 都未减账,失败重试仍是同一笔已知预约,
        # 不会留下"预约已注销但部分 rank 已减账"的半更新残留。
        for reservation_id in reservation_ids:
            reservation = self._reservations[reservation_id]
            for rank, value in zip(
                self.topology.instance(reservation.instance_index).ranks,
                reservation.shard_bytes,
            ):
                if self._rank_states[rank].reserved_request_bytes < value:
                    raise RuntimeError("request reservation accounting underflow")
        for reservation_id in reservation_ids:
            reservation = self._reservations.pop(reservation_id)
            for rank, value in zip(
                self.topology.instance(reservation.instance_index).ranks,
                reservation.shard_bytes,
            ):
                self._rank_states[rank].reserved_request_bytes -= value
            if self._metrics_recorder is not None:
                for rank, value in zip(
                    self.topology.instance(reservation.instance_index).ranks,
                    reservation.shard_bytes,
                ):
                    if not value:
                        continue
                    self._metrics_recorder.record(
                        planner_time_ns=completion_ns,
                        anchor_kind="completion",
                        # 归属与本地/远端核销行一致(M1-06):统一记
                        # terminal_request_id;allocation_key 仍精确到
                        # 具体预约,账本粒度不变。
                        request_id=terminal_request_id,
                        session_id=session_id,
                        rank=rank,
                        instance_index=reservation.instance_index,
                        allocation_key=f"reservation:{reservation_id}",
                        reserved_kv_delta_bytes=-int(value),
                        cause="terminal_session_retire_reservation",
                    )

        released_instance_index: Optional[int] = None
        if state.location == LOCAL_HBM:
            if state.instance_index is None:
                raise RuntimeError("resident completed session has no instance")
            released_instance_index = state.instance_index
            local_shards = self._local_prefix_shards(state)
            self._remove_shards(released_instance_index, local_shards)
            self._metrics_remove_session_parts(
                session_id,
                now_ns=completion_ns,
                anchor_kind="completion",
                request_id=terminal_request_id,
                cause="terminal_session_retire",
            )
        elif state.location != REMOTE_MEMORY:
            raise RuntimeError(f"unknown KV location: {state.location}")
        # Remote silent write-off: no transfer is emitted; the per-rank pool
        # account of the storing instance is debited under the same terminal
        # cause so the journal conservation closes at zero.
        remote_instance = state.remote_source_instance_index
        remote_shards = self._remote_shards(state)
        if remote_instance is not None and any(remote_shards):
            self._remote_remove(
                session_id,
                remote_instance,
                remote_shards,
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
        """Complete audit (strict mode / terminal boundary): two-state
        session assertions (LOCAL_HBM fully resident / REMOTE_MEMORY zero
        local; ``PARTIAL_HBM_REMOTE`` is unreachable at runtime), local
        session + remote pool ledgers per rank, capacity bounds, and
        reservation accounting."""

        expected_resident: dict[int, int] = dict.fromkeys(self._rank_states, 0)
        expected_remote: dict[int, int] = dict.fromkeys(self._rank_states, 0)
        for session in self._sessions.values():
            if session.location not in {LOCAL_HBM, REMOTE_MEMORY}:
                # PARTIAL_HBM_REMOTE is a legacy parsing-only value; a
                # runtime session carrying it is a hard bug (the partial
                # eviction/restore paths no longer exist).
                raise RuntimeError(f"invalid KV location for {session.session_id}")
            if session.location == REMOTE_MEMORY:
                if session.instance_index is not None:
                    raise RuntimeError("remote KV session retained a local instance")
                if session.resident_prefix_layers != 0:
                    raise RuntimeError("remote KV session retained resident layers")
            else:
                if session.resident_prefix_layers != self.model_layers:
                    raise RuntimeError("fully local KV session is missing layers")
                if session.instance_index is None:
                    raise RuntimeError("local KV session has no instance")
            if len(session.shard_bytes) != self.tp_degree:
                raise RuntimeError("KV shard count does not match TP instance")
            local_shards = self._local_prefix_shards(session)
            remote_shards = self._remote_shards(session)
            if (
                len(local_shards) != self.tp_degree
                or len(remote_shards) != self.tp_degree
            ):
                # 同 M1-07:zip 截断会让守恒断言被静默跳过,长度先行断言。
                raise RuntimeError("KV shard count does not match TP instance")
            for local_bytes, remote_bytes, total_bytes in zip(
                local_shards, remote_shards, session.shard_bytes
            ):
                if local_bytes + remote_bytes != total_bytes:
                    raise RuntimeError(
                        "local+remote split does not preserve total KV bytes "
                        f"for {session.session_id}"
                    )
            if session.location == LOCAL_HBM:
                instance = self.topology.instance(session.instance_index)
                for rank, shard_bytes in zip(instance.ranks, local_shards):
                    expected_resident[rank] += shard_bytes
            if session.remote_source_instance_index is not None and any(remote_shards):
                remote_instance = self.topology.instance(
                    session.remote_source_instance_index
                )
                for rank, remote_bytes in zip(remote_instance.ranks, remote_shards):
                    expected_remote[rank] += remote_bytes

        for rank, state in self._rank_states.items():
            if state.resident_kv_bytes < 0 or state.reserved_request_bytes < 0:
                raise RuntimeError("negative HBM accounting")
            if self._remote_bytes_by_rank[rank] < 0:
                raise RuntimeError("negative remote-pool accounting")
            if state.used_bytes > state.capacity_bytes:
                raise RuntimeError(
                    f"HBM capacity exceeded on rank {state.rank}: "
                    f"used={state.used_bytes}, capacity={state.capacity_bytes}"
                )
            if state.resident_kv_bytes != expected_resident[rank]:
                raise RuntimeError(
                    f"session/rank KV accounting mismatch: rank {rank} "
                    f"state={state.resident_kv_bytes}, "
                    f"sessions={expected_resident[rank]}"
                )
            if self._remote_bytes_by_rank[rank] != expected_remote[rank]:
                raise RuntimeError(
                    f"session/rank remote-pool accounting mismatch: "
                    f"rank {rank} account={self._remote_bytes_by_rank[rank]}, "
                    f"sessions={expected_remote[rank]}"
                )

        reserved: dict[int, list[int]] = {
            instance.index: [0] * self.tp_degree for instance in self.topology.instances
        }
        for reservation in self._reservations.values():
            if len(reservation.shard_bytes) != self.tp_degree:
                raise RuntimeError("reservation has invalid TP shard vector")
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
            if actual != tuple(
                expected_resident[rank] for rank in instance.ranks
            ):
                raise RuntimeError(
                    f"session/rank KV accounting mismatch for instance {instance.index}: "
                    f"states={actual}, sessions={tuple(expected_resident[rank] for rank in instance.ranks)}"
                )
            actual_reserved = tuple(
                self._rank_states[rank].reserved_request_bytes for rank in instance.ranks
            )
            if actual_reserved != tuple(reserved[instance.index]):
                raise RuntimeError(
                    f"reservation/rank accounting mismatch for instance {instance.index}: "
                    f"states={actual_reserved}, reservations={tuple(reserved[instance.index])}"
                )
            actual_remote = tuple(
                self._remote_bytes_by_rank[rank] for rank in instance.ranks
            )
            if actual_remote != tuple(
                expected_remote[rank] for rank in instance.ranks
            ):
                raise RuntimeError(
                    f"remote-pool/rank accounting mismatch for instance {instance.index}: "
                    f"states={actual_remote}, sessions={tuple(expected_remote[rank] for rank in instance.ranks)}"
                )
