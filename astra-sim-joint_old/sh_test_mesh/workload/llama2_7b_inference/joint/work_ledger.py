"""work_ledger.py -- W1 因果运行账本（schema ``workledger-v1-20260929``）。

《joint仓库当下+历史修改方案》§4（当前账本与一致性快照）与 W1 段
（618-624 行）的 Python 侧落地；版本常量从 ``joint.schema_registry``
引用（W0 冻结，不得散落硬编码）。

核心对象（盘点 /tmp/joint_w0/inv_w1.md §4.1 字段级清单）：

* :class:`WorkIdentity` —— run/request/session/turn、stage/generation（与
  NodeStore 分组键一致）、operation、node_uid=(rank, node_id)、
  resource_legs、shard/chunk/tile、owner_key、reservation/obligation 链接。
  ``work_id`` 规范化为 ``"{request_id}#{stage}:{generation}:{node_uid}"``；
  资源腿级工作（HBM job / pool job / NoC flow）用
  :func:`leg_work_id` 加 ``@`` 后缀派生，同一物理腿在节点视图与资源视图
  两张摘要中出现时按 (work_id, leg) 去重，不重复计费（§4.1:146）。
* :class:`CommittedWork` —— 状态机
  committed→ready→issued→in_service→completed→acknowledged→released；
  total/served/remaining 三量（ops/bytes/latency_ns）；
  ``progress_source`` ∈ authoritative|inferred|stale（§4.3:174：尚无进度
  遥测时不得把原始总量继续当"精确剩余"，也不得从 ETA 倒推隐藏未来）。
* :class:`LedgerEvent` —— 六类生命周期事件（commit/issue/progress/
  completion/ack/release）+ 幂等应用器：``(work_id, kind, local_seq)``
  重放容忍；tick 倒退 / local_seq 跳号 fail-closed（与
  ``online_scheduler_base._gate_delivery_sequence`` 同型的门）。
* :class:`WorkLedger` —— 账本本体：全量（full）/增量（delta）两种装载；
  版本水位（``version``/``event_watermark``）；retired 对象仍被依赖的
  release_condition 保留（§4.3:178：对象退休与日志压缩不得丢失仍被依赖
  的义务）；守恒校验接口（Σserved 增量与 Σremaining 减量一致）。

状态派生式桥接（盘点 §4.3 第 4 点推荐形态）：C++ 侧在 tick-end 将
连续资源积分到当前静态点后读取快照（NodeStore/FluidScheduler/
AnalyticalRemoteMemory/LocalHbmBandwidthModel/HardwareResource 五类），首交付全量、后续交付由
前后帧 diff 派生的增量事件（producer=``work_ledger_diff``，local_seq 单
调）；"事件"在 Python 端按本模块的状态机语义应用，progress 一律标
``inferred``（真值来自快照差分时标），权威进度只来自后端积分本身。
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import copy
from collections.abc import Mapping as _RuntimeMapping
from types import MappingProxyType
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Tuple

from joint.schema_registry import BRIDGE_PROTOCOL_VERSION, WORK_LEDGER_SCHEMA_VERSION

#: request JSON additive 字段 ``work_ledger`` 的整型子版本（与桥
#: ``schema_version`` 的 v1 语义同面：additive 字段缺席 = 旧对端兼容）。
WORK_LEDGER_FIELD_VERSION = 1

#: 六类生命周期事件（方案 W1 任务原文）。
EVENT_KINDS: Tuple[str, ...] = (
    "commit", "issue", "progress", "completion", "ack", "release")

#: 快照装载模式。
SNAPSHOT_KINDS: Tuple[str, ...] = ("full", "delta")

#: 进度来源标记（§4.3:174）。
PROGRESS_SOURCES: Tuple[str, ...] = ("authoritative", "inferred", "stale")

#: 工作状态机（§4.1:146）。
STATES: Tuple[str, ...] = (
    "committed", "ready", "issued", "in_service", "completed",
    "acknowledged", "released")

#: 合法的状态转移（release 后终态）。
_TRANSITIONS: Dict[Tuple[str, str], bool] = {
    ("committed", "ready"): True,
    ("ready", "issued"): True,
    ("committed", "issued"): True,
    ("issued", "in_service"): True,
    ("committed", "in_service"): True,
    ("ready", "in_service"): True,
    ("in_service", "completed"): True,
    ("issued", "completed"): True,
    ("committed", "completed"): True,
    ("ready", "completed"): True,
    ("completed", "acknowledged"): True,
    ("acknowledged", "released"): True,
    ("completed", "released"): True,
}

#: 事件可作用于的状态域（应用前的存在性校验）。
_KIND_SOURCE_STATES: Dict[str, Tuple[str, ...]] = {
    "issue": ("committed", "ready"),
    "progress": ("issued", "in_service", "committed", "ready"),
    # completion 对 completed 幂等重申：紧凑 vanished 清单一律展开为
    # completion+release 两个事件（与 C++ 计数器每 vanished 消耗 2 个
    # local_seq 对齐）。
    "completion": ("committed", "ready", "issued", "in_service",
                   "completed"),
    "ack": ("completed",),
    "release": ("completed", "acknowledged", "released"),
}


#: 事件对象的白名单字段面（判别字段 + commit payload 稳定字段；与 C++
#: DecisionBridge.cc 的 make_event/commit_payload 字段面一一对应——
#: test_work_ledger.TestBridgeSchemaSync 钉死两端同步）。
EVENT_PAYLOAD_WHITELIST = frozenset({
    # 判别字段
    "kind", "work_id", "tick", "event_id", "delta_served",
    # 生命周期状态（commit 的初始状态）
    "state",
    # commit_payload 携带的稳定身份/总量字段（C++ kCarry 面）
    "request_id", "stage", "generation", "rank", "node_id", "operation",
    "leg", "num_ops", "bytes", "runtime_ns", "tensor_size", "total_bytes",
    "dependencies", "parents", "service_unit", "route", "flow_id",
    "port_index", "issue_seq", "total", "total_latency_ns",
    "release_conditions",
    # 资源腿 commit 的权威剩余量（§4.3:174：增量交付中首次见到的腿必须
    # 携带后端已扣进度的剩余量，progress_source=authoritative；缺席 =
    # 上界占位降级，见 _work_from_event 与消费侧披露）。
    "remaining_ops", "remaining_bytes", "remaining_latency_ns",
})


class WorkLedgerError(ValueError):
    """fail-closed 账本协议错误（schema 不符/跳号/时间倒退/非法转移）。"""


@dataclass(frozen=True)
class ApplyReceipt:
    """Constant-size successful delivery result.

    ``applied_events + replayed_events`` is the fully accepted event prefix
    (and the full event count on success); ``virtual_count`` is the number of
    compact-v1 events synthesized within that prefix, including replayed
    virtual events. A failing delivery returns no receipt: already-applied
    prefix effects remain as before, while the envelope watermark, delivery
    sequence, version and event sink retain their existing commit boundary.
    """

    snapshot_kind: str
    applied_events: int
    replayed_events: int
    virtual_count: int
    event_watermark: int
    ledger_version: int


@dataclass(frozen=True)
class LedgerProjectionRow:
    """Immutable, whitelist-only row shared by snapshots and world sources."""

    identity: "WorkIdentity"
    state: str
    total: "WorkAmount"
    served: "WorkAmount"
    remaining: "WorkAmount"
    progress_source: str
    service_unit: str
    dependencies: Tuple[str, ...]
    release_conditions: Tuple[str, ...]
    route: Optional[Tuple[int, ...]]
    weight: int
    leg: Optional[str]
    committed_tick: Optional[int]
    issued_tick: Optional[int]
    completed_tick: Optional[int]
    acked_tick: Optional[int]
    released_tick: Optional[int]
    as_of_tick: Optional[int]

    @property
    def work_id(self) -> str:
        if self.leg is None:
            return self.identity.work_id
        return leg_work_id(self.identity.work_id, self.leg)


@dataclass(frozen=True)
class LedgerProjection:
    """Point-in-time immutable rows and row indexes owned by a WorkLedger."""

    rows: Tuple[LedgerProjectionRow, ...]
    pricing_rows: Tuple[LedgerProjectionRow, ...]
    committed_not_issued: int
    resource_index: Mapping[str, Tuple[LedgerProjectionRow, ...]]
    request_stage_index: Mapping[Tuple[str, str], Tuple[LedgerProjectionRow, ...]]
    rank_index: Mapping[int, Tuple[LedgerProjectionRow, ...]]
    dependency_index: Mapping[str, Tuple[LedgerProjectionRow, ...]]
    release_condition_index: Mapping[str, Tuple[LedgerProjectionRow, ...]]
    unfinished_state_index: Mapping[str, Tuple[LedgerProjectionRow, ...]]
    resource_version_vector: Tuple[Tuple[str, int], ...]


@dataclass(frozen=True)
class LedgerPricingSnapshot:
    """Small immutable frontier needed by the hot pricing path only."""

    pricing_rows: Tuple[LedgerProjectionRow, ...]
    committed_not_issued: int


def _amount_number(value: Any, field_name: str) -> float:
    """Decode one wire amount without accepting coercions or non-finite data.

    ``json.loads`` accepts ``NaN``/``Infinity`` by default and Python's
    ``float``/comparison operators otherwise let those values flow through
    conservation arithmetic.  A string such as ``"10"`` is also not a wire
    number: accepting it would make malformed C++/JSON producers silently
    change the protocol type.  Keep this gate shared by direct WorkAmount
    construction and all snapshot/event decoders.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise WorkLedgerError(
            "work amount {!r} must be a finite number, got {!r}"
            .format(field_name, value))
    number = float(value)
    if not math.isfinite(number):
        raise WorkLedgerError(
            "work amount {!r} must be finite, got {!r}"
            .format(field_name, value))
    return number


def _wire_integer(value: Any, field_name: str) -> int:
    """Decode an integer wire field without bool/float/string truncation."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise WorkLedgerError(
            "work ledger field {!r} must be an integer, got {!r}"
            .format(field_name, value))
    return value


def _freeze_state_tree(value: Any) -> Any:
    """Freeze a JSON-like ledger export for sharing across read-only builders."""
    if issubclass(type(value), _RuntimeMapping):
        return MappingProxyType({
            key: _freeze_state_tree(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_state_tree(item) for item in value)
    return value


# ---------------------------------------------------------------- 身份 --


def canonical_work_id(
    request_id: str,
    stage: str,
    generation: int,
    rank: int,
    node_id: int,
) -> str:
    """规范化全局 work_id：``{request_id}#{stage}:{generation}:{node_uid}``。

    node_uid 编码为 ``r{rank}n{node_id}``（NodeStore 分组键 + 仓内
    (rank, store_id) 身份；json id 须经 GraphBatchCommitter 的 affine
    映射才是 store id，本账本只消费 store id）。
    """
    return "{}#{}:{}:r{}n{}".format(
        request_id, stage, int(generation), int(rank), int(node_id))


def leg_work_id(work_id: str, leg: str) -> str:
    """资源腿级工作的派生 id：``{work_id}@{leg}``。

    leg 形如 ``hbm:COMM_READ``、``pool:0:17``、``noc:42``。同一物理腿在
    节点摘要与资源摘要两张快照里出现时身份一致，按 (work_id, leg)
    去重——"同一物理腿不能因来自两张摘要被计两次"（§4.1:146）。
    """
    return "{}@{}".format(work_id, leg)


@dataclass(frozen=True)
class WorkIdentity:
    """§4.1 WorkIdentity：身份字段全集（字段级清单见盘点 §4.1）。"""

    request_id: str
    stage: str
    generation: int
    rank: int
    node_id: int
    operation: str = ""            # compute/comm_send/.../pool_read/...
    session_id: str = ""
    turn_index: Optional[int] = None
    run_id: str = ""
    resource_legs: Tuple[str, ...] = ()
    shard: Optional[int] = None
    chunk: Optional[int] = None
    tile: Optional[int] = None
    owner_key: str = ""            # 与 LinkFlowRegistry owner 同源
    reservation_id: Optional[str] = None
    obligation_link: Optional[str] = None

    @property
    def work_id(self) -> str:
        return canonical_work_id(
            self.request_id, self.stage, self.generation, self.rank,
            self.node_id)

    @property
    def node_uid(self) -> Tuple[int, int]:
        return (self.rank, self.node_id)


# ---------------------------------------------------------------- 量 --


#: 浮点容差（C++ 资源剩余量是 double：LocalHbmBandwidthModel Job /
#: FluidScheduler Flow / AnalyticalRemoteMemory PortJob 全为浮点积分值；
#: 守恒校验按容差判定，不用整数截断伪装精确）。
AMOUNT_EPSILON = 1e-6


@dataclass(frozen=True)
class WorkAmount:
    """工作量三元组（ops/bytes/latency_ns；None = 该维度不适用）。

    值为 number（int 或 float）：节点静态属性是整数，资源腿剩余量是
    后端浮点积分值。负值超出容差 = 非法（fail-closed）。
    """

    ops: Optional[float] = None
    bytes: Optional[float] = None
    latency_ns: Optional[float] = None

    def validate(self) -> "WorkAmount":
        for name in ("ops", "bytes", "latency_ns"):
            value = getattr(self, name)
            if value is None:
                continue
            value = _amount_number(value, name)
            if value < -AMOUNT_EPSILON:
                raise WorkLedgerError(
                    "work amount {!r} must be non-negative, got {!r}"
                    .format(name, value))
        return self

    def plus(self, other: "WorkAmount") -> "WorkAmount":
        return WorkAmount(
            ops=_opt_add(self.ops, other.ops),
            bytes=_opt_add(self.bytes, other.bytes),
            latency_ns=_opt_add(self.latency_ns, other.latency_ns),
        ).validate()

    def minus(self, other: "WorkAmount") -> "WorkAmount":
        return WorkAmount(
            ops=_opt_sub(self.ops, other.ops),
            bytes=_opt_sub(self.bytes, other.bytes),
            latency_ns=_opt_sub(self.latency_ns, other.latency_ns),
        ).validate()

    def as_dict(self) -> Dict[str, float]:
        return {
            "ops": self.ops, "bytes": self.bytes,
            "latency_ns": self.latency_ns}


def _opt_add(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None:
        return b
    if b is None:
        return a
    return a + b


def _opt_sub(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return a
    return a - b


def amount_from_mapping(payload: Optional[Mapping[str, Any]]) -> WorkAmount:
    if payload is None:
        return WorkAmount()
    unknown = set(payload) - {"ops", "bytes", "latency_ns"}
    if unknown:
        raise WorkLedgerError(
            "unknown work-amount keys {!r}".format(sorted(unknown)))
    return WorkAmount(
        ops=(_amount_number(payload["ops"], "ops")
             if payload.get("ops") is not None else None),
        bytes=(_amount_number(payload["bytes"], "bytes")
               if payload.get("bytes") is not None else None),
        latency_ns=(_amount_number(payload["latency_ns"], "latency_ns")
                    if payload.get("latency_ns") is not None else None),
    ).validate()


# ---------------------------------------------------------------- 工作 --


@dataclass
class CommittedWork:
    """§4.1 CommittedWork：状态机 + 三量 + 依赖与释放条件。"""

    identity: WorkIdentity
    state: str = "committed"
    total: WorkAmount = field(default_factory=WorkAmount)
    served: WorkAmount = field(default_factory=WorkAmount)
    remaining: WorkAmount = field(default_factory=WorkAmount)
    progress_source: str = "stale"
    service_unit: str = "job"
    dependencies: Tuple[str, ...] = ()
    release_conditions: Tuple[str, ...] = ()
    route: Optional[Tuple[int, ...]] = None
    weight: int = 0
    leg: Optional[str] = None          # 资源腿名（None = 节点级工作）
    committed_tick: Optional[int] = None
    issued_tick: Optional[int] = None
    completed_tick: Optional[int] = None
    acked_tick: Optional[int] = None
    released_tick: Optional[int] = None
    #: 快照积分时点（资源腿工作专有：remaining 积分到的 tick；None = 未披露）。
    as_of_tick: Optional[int] = None

    def __post_init__(self) -> None:
        if self.state not in STATES:
            raise WorkLedgerError(
                "unknown work state {!r}".format(self.state))
        if self.progress_source not in PROGRESS_SOURCES:
            raise WorkLedgerError(
                "unknown progress_source {!r}".format(self.progress_source))
        self.total.validate()
        self.served.validate()
        self.remaining.validate()

    @property
    def work_id(self) -> str:
        if self.leg is None:
            return self.identity.work_id
        return leg_work_id(self.identity.work_id, self.leg)

    def transition(self, target: str, tick: int) -> None:
        if self.state == target:
            return  # 幂等（重放容忍由应用器先行拦截，这里防御）
        if not _TRANSITIONS.get((self.state, target)):
            raise WorkLedgerError(
                "illegal transition {!r} -> {!r} for {} (states: committed->"
                "ready->issued->in_service->completed->acknowledged->"
                "released)".format(self.state, target, self.work_id))
        self.state = target
        if target == "issued" and self.issued_tick is None:
            self.issued_tick = tick
        if target == "completed":
            self.completed_tick = tick
        if target == "acknowledged":
            self.acked_tick = tick
        if target == "released":
            self.released_tick = tick

    def apply_progress(
        self, delta_served: WorkAmount, tick: int,
        progress_source: str = "inferred",
    ) -> None:
        """进度推进：served 增量与 remaining 减量守恒（§4.3:174）。

        剩余量扣减到负 = 物理账本不合法，fail-closed（方案 §4.3:176
        "物理账本不合法时拒绝提交"）。
        """
        if self.state in ("completed", "acknowledged", "released"):
            raise WorkLedgerError(
                "progress event on finished work {}".format(self.work_id))
        for dimension in ("ops", "bytes", "latency_ns"):
            delta = getattr(delta_served, dimension)
            if delta is not None and getattr(self.remaining, dimension) is None:
                raise WorkLedgerError(
                    "progress event {} serves an unavailable {} dimension"
                    .format(self.work_id, dimension))
        self.served = self.served.plus(delta_served).validate()
        self.remaining = self.remaining.minus(delta_served).validate()
        if self.progress_source != "authoritative":
            self.progress_source = progress_source
        if self.state in ("committed", "ready", "issued"):
            self.state = "in_service"
        if self.issued_tick is None:
            self.issued_tick = tick

    def finish(self, tick: int) -> None:
        """完成：剩余清零（先核对 served+remaining==total 的守恒接口位）。"""
        if self.state not in ("completed", "acknowledged", "released"):
            self.transition("completed", tick)
        self.remaining = WorkAmount()

    def summary(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "work_id": self.work_id,
            "state": self.state,
            "operation": self.identity.operation,
            "node_uid": "r{}n{}".format(
                self.identity.rank, self.identity.node_id),
            "total": self.total.as_dict(),
            "served": self.served.as_dict(),
            "remaining": self.remaining.as_dict(),
            "progress_source": self.progress_source,
            "service_unit": self.service_unit,
            "dependencies": list(self.dependencies),
            "release_conditions": list(self.release_conditions),
            "committed_tick": self.committed_tick,
            "issued_tick": self.issued_tick,
            "completed_tick": self.completed_tick,
            "acked_tick": self.acked_tick,
            "released_tick": self.released_tick,
        }
        if self.leg is not None:
            payload["leg"] = self.leg
        if self.route is not None:
            payload["route"] = list(self.route)
        return payload


# ---------------------------------------------------------------- 事件 --


@dataclass(frozen=True)
class LedgerEvent:
    """六类生命周期事件（§4.2 事件生命周期与可见性）。"""

    kind: str
    work_id: str
    tick: int
    event_id: Mapping[str, Any]        # {tick, delivery_sequence, producer, local_seq}
    delta_served: Optional[WorkAmount] = None
    payload: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in EVENT_KINDS:
            raise WorkLedgerError(
                "unknown ledger event kind {!r} (kinds: {})".format(
                    self.kind, list(EVENT_KINDS)))

    @property
    def idempotency_key(self) -> Tuple[str, str, int]:
        """幂等键 (work_id, kind, local_seq)：重复 = 重放。"""
        return (self.work_id, self.kind,
                int(self.event_id.get("local_seq", -1)))

    def as_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "kind": self.kind,
            "work_id": self.work_id,
            "tick": self.tick,
            "event_id": dict(self.event_id),
        }
        if self.delta_served is not None:
            payload["delta_served"] = self.delta_served.as_dict()
        payload.update(self.payload)
        return payload


@dataclass(frozen=True)
class _TypedLedgerEvent:
    """Internal event fields used by the shared typed application primitive."""

    kind: str
    work_id: str
    tick: int
    producer: str
    delivery_sequence: int
    local_seq: int
    delta_served: Optional[WorkAmount]
    payload: Mapping[str, Any]
    canonical_content: Any
    virtual: bool = False
    compact_entry: Optional[Mapping[str, Any]] = None


_EMPTY_EVENT_PAYLOAD: Mapping[str, Any] = MappingProxyType({})


def event_from_mapping(payload: Mapping[str, Any]) -> LedgerEvent:
    """Materialize the compatibility event object from a validated wire map."""
    return _ledger_event_for_observer(_typed_event_from_mapping(payload))


def _typed_event_from_mapping(payload: Mapping[str, Any]) -> _TypedLedgerEvent:
    """Validate ordinary events without building a LedgerEvent or payload copy."""
    if not issubclass(type(payload), _RuntimeMapping):
        raise WorkLedgerError(
            "ledger event must be a mapping, got {!r}"
            .format(type(payload).__name__))
    # 事件级白名单 = 判别字段 + C++ commit payload 可携带的稳定身份/总量
    # 字段（DecisionBridge.cc commit_payload 的 kCarry 面 + state/total）；
    # 桥接两端字段面由 test_work_ledger.py 的 schema 同步测试钉死。
    unknown = set(payload) - EVENT_PAYLOAD_WHITELIST
    if unknown:
        raise WorkLedgerError(
            "unknown ledger event keys {!r}".format(sorted(unknown)))
    for required in ("kind", "work_id", "tick", "event_id"):
        if required not in payload:
            raise WorkLedgerError(
                "ledger event missing field {!r}".format(required))
    event_id = payload["event_id"]
    if (not issubclass(type(event_id), _RuntimeMapping)
            or "local_seq" not in event_id):
        raise WorkLedgerError(
            "ledger event_id must be a mapping with local_seq (§4.2:166 "
            "monotonic local sequence; delivery_sequence alone is only a "
            "callback epoch)")
    unknown_event_id = set(event_id) - {
        "tick", "delivery_sequence", "producer", "local_seq"}
    if unknown_event_id:
        raise WorkLedgerError(
            "unknown ledger event_id keys {!r}"
            .format(sorted(unknown_event_id)))
    if not isinstance(payload["kind"], str):
        raise WorkLedgerError("ledger event kind must be a string")
    if not isinstance(payload["work_id"], str) or not payload["work_id"]:
        raise WorkLedgerError("ledger event work_id must be a non-empty string")
    if payload["kind"] not in ("commit", "progress") and any(
            key in payload for key in (
                "remaining_ops", "remaining_bytes",
                "remaining_latency_ns")):
        raise WorkLedgerError(
            "remaining_* fields are only valid on commit/progress events")
    event_tick = _wire_integer(payload["tick"], "event.tick")
    event_id_tick = _wire_integer(event_id.get("tick"), "event_id.tick")
    if event_id_tick != event_tick:
        raise WorkLedgerError(
            "ledger event_id.tick {} != event tick {}"
            .format(event_id_tick, event_tick))
    event_delivery = _wire_integer(
        event_id.get("delivery_sequence"), "event_id.delivery_sequence")
    producer = event_id.get("producer")
    if not isinstance(producer, str) or not producer:
        raise WorkLedgerError(
            "ledger event_id.producer must be a non-empty string")
    local_seq = _wire_integer(event_id["local_seq"], "event_id.local_seq")
    for key in ("remaining_ops", "remaining_bytes", "remaining_latency_ns"):
        if key in payload:
            _amount_number(payload[key], "event." + key)
    delta_served = amount_from_mapping(payload.get("delta_served"))
    kind = payload["kind"]
    work_id = payload["work_id"]
    canonical = _event_canonical_content(
        kind=kind,
        work_id=work_id,
        tick=event_tick,
        event_id_pairs=(
            ("tick", event_id_tick),
            ("delivery_sequence", event_delivery),
            ("producer", producer),
            ("local_seq", local_seq),
        ),
        delta_served=delta_served,
        payload_items=(
            (key, value) for key, value in payload.items()
            if key in EVENT_PAYLOAD_WHITELIST
            and key not in ("kind", "work_id", "tick", "event_id",
                            "delta_served")),
    )
    return _TypedLedgerEvent(
        kind=kind,
        work_id=work_id,
        tick=event_tick,
        producer=producer,
        delivery_sequence=event_delivery,
        local_seq=local_seq,
        delta_served=delta_served,
        # The input is already validated against the complete whitelist and
        # remains alive for this synchronous event application only.
        payload=payload,
        canonical_content=canonical,
    )


def _typed_v3_node_event_from_validated_scalars(
    *,
    kind: str,
    work_id: str,
    tick: int,
    delivery_sequence: int,
    producer: str,
    local_seq: int,
    payload: Mapping[str, Any],
    build_canonical_content: bool,
) -> _TypedLedgerEvent:
    """Build a v3 node DTO after PipelineLedger validated its row/header.

    This private fast path accepts only validated scalar headers and a
    whitelist-checked event body. It retains the body in a detached top-level
    dict, applies the same body-dependent checks as the generic legacy parser,
    and constructs replay canonical content only for the transaction's final
    node event.
    """
    # Keep the same top-level sender detach as the legacy parser, without
    # merging the already-validated header fields into this body mapping.
    event_payload = dict(payload)
    if kind not in ("commit", "progress") and any(
            key in event_payload for key in (
                "remaining_ops", "remaining_bytes",
                "remaining_latency_ns")):
        raise WorkLedgerError(
            "remaining_* fields are only valid on commit/progress events")
    for key in ("remaining_ops", "remaining_bytes", "remaining_latency_ns"):
        if key in event_payload:
            _amount_number(event_payload[key], "event." + key)
    delta_served = amount_from_mapping(event_payload.get("delta_served"))
    canonical = None
    if build_canonical_content:
        canonical = _event_canonical_content(
            kind=kind,
            work_id=work_id,
            tick=tick,
            event_id_pairs=(
                ("tick", tick),
                ("delivery_sequence", delivery_sequence),
                ("producer", producer),
                ("local_seq", local_seq),
            ),
            delta_served=delta_served,
            payload_items=(
                (key, value) for key, value in event_payload.items()
                if key in EVENT_PAYLOAD_WHITELIST
                and key != "delta_served"),
        )
    return _TypedLedgerEvent(
        kind=kind,
        work_id=work_id,
        tick=tick,
        producer=producer,
        delivery_sequence=delivery_sequence,
        local_seq=local_seq,
        delta_served=delta_served,
        payload=event_payload,
        canonical_content=canonical,
    )


def _ledger_event_for_observer(event: _TypedLedgerEvent) -> LedgerEvent:
    """Rebuild the old public event view only when an observer requests it."""
    if event.compact_entry is not None:
        entry = event.compact_entry
        payload: Dict[str, Any] = {}
        for key in (
                "request_id", "stage", "generation", "rank", "node_id",
                "operation", "num_ops", "bytes", "runtime_ns",
                "tensor_size", "parents"):
            if key in entry:
                payload[key] = entry[key]
        total = {}
        if entry.get("num_ops") is not None:
            total["ops"] = entry["num_ops"]
        if entry.get("bytes") is not None:
            total["bytes"] = entry["bytes"]
        if entry.get("runtime_ns") is not None:
            total["latency_ns"] = entry["runtime_ns"]
        if total:
            payload["total"] = total
        payload["state"] = (
            "completed" if bool(entry.get("finished", False))
            else "issued" if bool(entry.get("issued", False))
            else "committed")
    else:
        payload = {
            key: value for key, value in event.payload.items()
            if key in EVENT_PAYLOAD_WHITELIST
            and key not in ("kind", "work_id", "tick", "event_id",
                            "delta_served")}
    return LedgerEvent(
        kind=event.kind,
        work_id=event.work_id,
        tick=event.tick,
        event_id={
            "tick": event.tick,
            "delivery_sequence": event.delivery_sequence,
            "producer": event.producer,
            "local_seq": event.local_seq,
        },
        delta_served=event.delta_served,
        payload=payload,
    )


# ---------------------------------------------------------------- 账本 --


class WorkLedger:
    """因果运行账本（全量/增量装载 + 幂等应用 + 守恒校验）。

    应用器门（与 ``online_scheduler_base._gate_delivery_sequence``
    同型，见 online_scheduler_base.py 的投递序列门）：

    * 幂等：与本应用器已见的**最后一条**事件 (producer, local_seq,
      digest) 完全一致的重复事件 = 重放，零副作用；
    * 单调：producer 的 local_seq 跳号（!= last+1）fail-closed；
    * 时间：事件 tick < 账本水位 tick fail-closed（tick 倒退）；
    * 水位：payload.event_watermark 单调不减。
    """

    def __init__(self, *, event_sink=None):
        self._works: Dict[str, CommittedWork] = {}
        self._retained_obligations: Dict[str, Tuple[str, ...]] = {}
        # Reverse dependency lookup makes retirement proportional to the
        # number of dependents rather than the size of the active ledger.
        self._dependents_by_work: Dict[str, set] = {}
        # Rows are refreshed only for touched works.  Immutable point-in-time
        # tuples/indexes are materialized lazily and then shared by snapshots.
        self._projection_rows: Dict[str, LedgerProjectionRow] = {}
        self._projection_order: Dict[str, int] = {}
        self._next_projection_order = 0
        self._pricing_rows: Dict[str, LedgerProjectionRow] = {}
        self._committed_not_issued = 0
        self._projection_indexes: Dict[str, Dict[Any, Dict[str, LedgerProjectionRow]]] = {
            "resource_index": {},
            "request_stage_index": {},
            "rank_index": {},
            "dependency_index": {},
            "release_condition_index": {},
            "unfinished_state_index": {},
        }
        self._projection_cache: Optional[LedgerProjection] = None
        self._projection_state_cache: Optional[Mapping[str, Any]] = None
        self._pricing_snapshot_cache: Optional[LedgerPricingSnapshot] = None
        self._resource_key_versions: Dict[str, int] = {}
        self.version: int = 0
        self.event_watermark: int = 0
        self.last_tick: int = -1
        self.last_delivery_sequence: int = -1
        self.applied_event_count: int = 0
        self.replayed_event_count: int = 0
        self.full_load_count: int = 0
        self.delta_load_count = 0
        # 应用器门状态：
        #   _last_event             -- 最后一个已应用事件 (producer, local_seq,
        #                               类型敏感冻结内容)；相邻重复只在命中
        #                               同 seq 后走 legacy JSON 比较。
        #   _expected_next_local_seq-- 下一个合法 local_seq（全量装载后 =
        #                               payload.event_watermark；随后严格 +1）。
        # 有意不保留逐事件 seen 表：交付层门（delivery_sequence 严格 +1）
        # 已排除跨交付重放，事件层只需容忍同交付内的紧邻重复并对其余一切
        # fail-closed——内存 O(1) 而非 O(事件数)。
        self._last_event: Optional[Tuple[str, int, Any]] = None
        self._expected_next_local_seq: Optional[int] = None
        self._event_sink = event_sink

    # ------------------------------------------------------------ 装载 --

    def apply_payload(
        self,
        payload: Mapping[str, Any],
        *,
        delivery_sequence: int,
        tick: int,
        observer: Optional[Callable[[LedgerEvent, bool], None]] = None,
    ) -> ApplyReceipt:
        """应用 request JSON 的顶层 additive 字段 ``work_ledger``。

        full 模式重置账本为快照终态；delta 模式逐事件流式应用。正常路径
        只保留常量大小计数与当前事件，不构造本批事件对象列表。可选
        ``observer`` 仅供差分验收：每处理一个事件回调一次
        ``(LedgerEvent, applied)``，由调用方决定是否保存证据。
        """
        _validate_payload_envelope(payload, delivery_sequence, tick)
        kind = payload["snapshot_kind"]
        watermark = _wire_integer(
            payload.get("event_watermark", 0), "event_watermark")
        if watermark < 0:
            raise WorkLedgerError(
                "work_ledger event_watermark must be non-negative, got {}"
                .format(watermark))
        if watermark < self.event_watermark:
            raise WorkLedgerError(
                "work_ledger event_watermark regressed: {} < {} (delivery "
                "{})".format(watermark, self.event_watermark, delivery_sequence))
        if delivery_sequence <= self.last_delivery_sequence and (
                self.last_delivery_sequence >= 0):
            raise WorkLedgerError(
                "work_ledger delivery_sequence not monotonic: {} after {}"
                .format(delivery_sequence, self.last_delivery_sequence))
        applied_count = 0
        replayed_count = 0
        virtual_count = 0
        if kind == "full":
            self.event_watermark = watermark
            self._load_full(payload.get("state"), delivery_sequence, tick)
            self.full_load_count += 1
        else:
            events = payload.get("events")
            if not isinstance(events, list):
                raise WorkLedgerError(
                    "work_ledger delta requires an events[] list")
            compact = payload.get("compact")
            # 旧实现先完整展开 compact 再触碰任何事件。保留这个失败边界，
            # 但只预扫输入结构；事件映射按序临时生成，不留整批对象列表。
            seq_base = self._preflight_compact(compact)
            event_iter = self._iter_payload_events(
                compact, seq_base, events, delivery_sequence, tick)
            for event in event_iter:
                if event.virtual:
                    virtual_count += 1
                was_applied = self._apply_event(event)
                if was_applied:
                    applied_count += 1
                else:
                    replayed_count += 1
                if observer is not None:
                    observer(_ledger_event_for_observer(event), was_applied)
            self.delta_load_count += 1
        self.event_watermark = max(self.event_watermark, watermark)
        self.last_delivery_sequence = delivery_sequence
        self.version += 1
        if self._event_sink is not None:
            self._event_sink({
                "delivery_sequence": delivery_sequence,
                "tick": tick,
                "snapshot_kind": kind,
                "applied_events": applied_count,
                "replayed_events": replayed_count,
                "event_watermark": watermark,
                "active_works": len(self._works),
                "ledger_version": self.version,
            })
        return ApplyReceipt(
            snapshot_kind=kind,
            applied_events=applied_count,
            replayed_events=replayed_count,
            virtual_count=virtual_count,
            event_watermark=self.event_watermark,
            ledger_version=self.version,
        )

    def begin_protocol_v3_transaction(
        self, *, delivery_sequence: int, tick: int, snapshot_kind: str,
    ):
        """Create an uncommitted COW transaction for interleaved v3 events.

        The pipeline consumer calls ``stage_event`` at the event's exact
        position in the shared logical sequence, then calls ``finish`` only
        after the complete pipeline envelope has passed validation.
        """
        if snapshot_kind not in SNAPSHOT_KINDS:
            raise WorkLedgerError("v3 snapshot_kind must be full or delta")
        delivery_sequence = _wire_integer(
            delivery_sequence, "v3.delivery_sequence")
        tick = _wire_integer(tick, "v3.tick")
        if self.last_delivery_sequence >= delivery_sequence:
            raise WorkLedgerError("v3 delivery sequence is not monotonic")
        if tick < self.last_tick:
            raise WorkLedgerError("v3 envelope tick regressed")
        if snapshot_kind == "full" and (
                self._works or self.full_load_count or self.delta_load_count):
            raise WorkLedgerError("v3 full snapshot requires a pristine WorkLedger")
        transaction = _WorkLedgerV2Transaction(self)
        transaction.delivery_sequence = delivery_sequence
        transaction.tick = tick
        transaction.snapshot_kind = snapshot_kind
        transaction._previous_seq = 0
        transaction._previous_tick = self.last_tick
        return transaction


    def _preflight_compact(
        self,
        compact: Any,
    ) -> Optional[int]:
        """Validate the entire compact structure before applying any event.

        JSON array containers are repeatable, so the second pass can emit one
        event mapping at a time without retaining expanded event objects.
        """
        if compact is None:
            return None
        if not issubclass(type(compact), _RuntimeMapping):
            raise WorkLedgerError(
                "work_ledger compact must be a mapping, got {!r}".format(
                    type(compact).__name__))
        unknown = set(compact) - {
            "node_commits", "node_issues", "node_completions",
            "node_vanished", "seq_base"}
        if unknown:
            raise WorkLedgerError(
                "unknown work_ledger compact keys {!r}".format(
                    sorted(unknown)))
        seq = _wire_integer(compact.get("seq_base", 0), "compact.seq_base")
        if seq < 0:
            raise WorkLedgerError(
                "compact.seq_base must be non-negative, got {!r}"
                .format(seq))
        for field_name in (
                "node_commits", "node_issues", "node_completions",
                "node_vanished"):
            entries = compact.get(field_name, ())
            if not isinstance(entries, (list, tuple)):
                raise WorkLedgerError(
                    "compact {} must be an array".format(field_name))
        for entry in compact.get("node_commits", ()):
            if (not issubclass(type(entry), _RuntimeMapping)
                    or "work_id" not in entry):
                raise WorkLedgerError(
                    "compact node_commits entries must carry work_id")
        return seq

    def _iter_payload_events(
        self,
        compact: Any,
        seq_base: Optional[int],
        events: List[Mapping[str, Any]],
        delivery_sequence: int,
        tick: int,
    ) -> Iterator[_TypedLedgerEvent]:
        """Yield typed compact and explicit events in original wire order."""
        seq = seq_base

        def emit(
            kind: str,
            work_id: str,
            compact_entry: Optional[Mapping[str, Any]] = None,
        ) -> _TypedLedgerEvent:
            nonlocal seq
            assert seq is not None
            payload_items = []
            if compact_entry is not None:
                payload_items.extend(
                    (key, compact_entry[key]) for key in (
                        "request_id", "stage", "generation", "rank",
                        "node_id", "operation", "num_ops", "bytes",
                        "runtime_ns", "tensor_size", "parents")
                    if key in compact_entry)
                total = {}
                if compact_entry.get("num_ops") is not None:
                    total["ops"] = compact_entry["num_ops"]
                if compact_entry.get("bytes") is not None:
                    total["bytes"] = compact_entry["bytes"]
                if compact_entry.get("runtime_ns") is not None:
                    total["latency_ns"] = compact_entry["runtime_ns"]
                if total:
                    payload_items.append(("total", total))
                payload_items.append((
                    "state",
                    "completed" if bool(compact_entry.get("finished", False))
                    else "issued" if bool(compact_entry.get("issued", False))
                    else "committed"))
            canonical = _event_canonical_content(
                kind=kind,
                work_id=work_id,
                tick=tick,
                event_id_pairs=(
                    ("tick", tick),
                    ("delivery_sequence", delivery_sequence),
                    ("producer", "work_ledger_diff"),
                    ("local_seq", seq),
                ),
                delta_served=WorkAmount(),
                payload_items=payload_items,
            )
            event = _TypedLedgerEvent(
                kind=kind,
                work_id=work_id,
                tick=tick,
                producer="work_ledger_diff",
                delivery_sequence=delivery_sequence,
                local_seq=seq,
                delta_served=WorkAmount(),
                payload=_EMPTY_EVENT_PAYLOAD,
                canonical_content=canonical,
                virtual=True,
                compact_entry=compact_entry,
            )
            seq += 1
            return event

        if compact is not None:
            for entry in compact.get("node_commits", ()):
                yield emit(
                    "commit", str(entry["work_id"]), compact_entry=entry)
            for work_id in compact.get("node_issues", ()):
                yield emit("issue", str(work_id))
            for work_id in compact.get("node_completions", ()):
                yield emit("completion", str(work_id))
            for work_id in compact.get("node_vanished", ()):
                # Each vanished id consumes two adjacent sequence numbers.
                yield emit("completion", str(work_id))
                yield emit("release", str(work_id))
        for raw in events:
            yield _typed_event_from_mapping(raw)

    def _load_full(
        self,
        state: Any,
        delivery_sequence: int,
        tick: int,
    ) -> None:
        """全量装载：以快照终态重建（等价性验收的参照面）。"""
        rebuilt = parse_work_ledger_state(state)
        self._works = rebuilt
        self._rebuild_projection_indexes()
        self._last_event = None
        # 全量装载后基线 = 生产者已消费的计数器（C++ provider 的
        # event_watermark；后续 delta 的 local_seq 从该值严格 +1 续编）。
        self._expected_next_local_seq = self.event_watermark
        self.last_tick = tick

    @staticmethod
    def _projection_row(work: CommittedWork) -> LedgerProjectionRow:
        """Copy the approved immutable work fields, never a runtime object."""
        return LedgerProjectionRow(
            identity=work.identity,
            state=work.state,
            total=work.total,
            served=work.served,
            remaining=work.remaining,
            progress_source=work.progress_source,
            service_unit=work.service_unit,
            dependencies=tuple(work.dependencies),
            release_conditions=tuple(work.release_conditions),
            route=(tuple(work.route) if work.route is not None else None),
            weight=int(work.weight),
            leg=work.leg,
            committed_tick=work.committed_tick,
            issued_tick=work.issued_tick,
            completed_tick=work.completed_tick,
            acked_tick=work.acked_tick,
            released_tick=work.released_tick,
            as_of_tick=work.as_of_tick,
        )

    @staticmethod
    def _resource_index_keys(row: LedgerProjectionRow) -> Tuple[str, ...]:
        leg = row.leg or ""
        if leg.startswith("hbm:"):
            return ("hbm_rank:{}".format(row.identity.rank),)
        if leg.startswith("pool:"):
            try:
                port = int(leg.split(":")[1])
            except (IndexError, ValueError):
                return ()
            return ("pool_port:{}".format(port),)
        if leg.startswith("noc:"):
            return tuple(dict.fromkeys(
                "noc_link_id:{}".format(int(link))
                for link in (row.route or ())))
        return ()

    def _bump_resource_versions(self, keys) -> None:
        for key in keys:
            self._resource_key_versions[key] = (
                self._resource_key_versions.get(key, 0) + 1)

    def _rebuild_projection_indexes(self) -> None:
        old_keys = {
            key for row in self._projection_rows.values()
            for key in self._resource_index_keys(row)}
        rows: Dict[str, LedgerProjectionRow] = {}
        projection_order: Dict[str, int] = {}
        pricing_rows: Dict[str, LedgerProjectionRow] = {}
        committed_not_issued = 0
        dependents: Dict[str, set] = {}
        new_keys = set()
        for order, (work_id, work) in enumerate(self._works.items()):
            row = self._projection_row(work)
            rows[work_id] = row
            projection_order[work_id] = order
            if row.state in ("committed", "ready"):
                committed_not_issued += 1
            if self._is_pricing_row(row):
                pricing_rows[work_id] = row
            for dependency in work.dependencies:
                dependents.setdefault(dependency, set()).add(work_id)
            new_keys.update(self._resource_index_keys(row))
        self._projection_rows = rows
        self._projection_order = projection_order
        self._next_projection_order = len(projection_order)
        self._pricing_rows = pricing_rows
        self._committed_not_issued = committed_not_issued
        self._dependents_by_work = dependents
        self._projection_indexes = {
            "resource_index": {},
            "request_stage_index": {},
            "rank_index": {},
            "dependency_index": {},
            "release_condition_index": {},
            "unfinished_state_index": {},
        }
        for work_id, row in rows.items():
            self._update_row_indexes(work_id, None, row)
        self._bump_resource_versions(old_keys | new_keys)
        self._projection_cache = None
        self._projection_state_cache = None
        self._pricing_snapshot_cache = None

    @staticmethod
    def _is_pricing_row(row: LedgerProjectionRow) -> bool:
        return (row.state not in ("committed", "ready") and row.leg
                and row.leg.startswith(("hbm:", "pool:", "noc:")))

    @classmethod
    def _projection_index_memberships(
        cls, row: LedgerProjectionRow,
    ) -> Tuple[Tuple[Any, ...], ...]:
        # Keep the fixed six-index shape positional. This method runs for each
        # accepted event, so allocating a short-lived name→tuple dict here is
        # needless; _projection_indexes is initialized in this same order.
        return (
            cls._resource_index_keys(row),
            ((row.identity.request_id, row.identity.stage),),
            (row.identity.rank,),
            tuple(row.dependencies),
            tuple(row.release_conditions),
            (row.state,),
        )

    def _update_row_indexes(
        self,
        work_id: str,
        previous: Optional[LedgerProjectionRow],
        current: Optional[LedgerProjectionRow],
    ) -> None:
        old_memberships = (
            self._projection_index_memberships(previous)
            if previous is not None else {})
        new_memberships = (
            self._projection_index_memberships(current)
            if current is not None else {})
        old_items = (old_memberships if previous is not None else
                     ((),) * len(self._projection_indexes))
        new_items = (new_memberships if current is not None else
                     ((),) * len(self._projection_indexes))
        for buckets, old_keys, new_keys in zip(
                self._projection_indexes.values(), old_items, new_items):
            # Most events only change lifecycle ticks/amounts while all index
            # memberships remain the same. Update the immutable row reference
            # directly without constructing temporary sets for that hot case.
            if old_keys == new_keys:
                for key in new_keys:
                    buckets[key][work_id] = current
                continue
            if not old_keys:
                for key in new_keys:
                    buckets.setdefault(key, {})[work_id] = current
                continue
            if not new_keys:
                for key in old_keys:
                    bucket = buckets.get(key)
                    if bucket is not None:
                        bucket.pop(work_id, None)
                        if not bucket:
                            buckets.pop(key, None)
                continue
            # Membership lists are normally tiny (resource routes and a few
            # dependencies). Tuple membership avoids six per-event set
            # allocations while preserving duplicate-tolerant bucket writes.
            for key in old_keys:
                if key in new_keys:
                    continue
                bucket = buckets.get(key)
                if bucket is not None:
                    bucket.pop(work_id, None)
                    if not bucket:
                        buckets.pop(key, None)
            for key in new_keys:
                if key in old_keys:
                    # Dict replacement keeps the original insertion position.
                    buckets[key][work_id] = current
                else:
                    buckets.setdefault(key, {})[work_id] = current

    def _register_work(self, work: CommittedWork) -> None:
        """Insert one new work and update all lifecycle indexes atomically."""
        self._works[work.work_id] = work
        for dependency in work.dependencies:
            self._dependents_by_work.setdefault(dependency, set()).add(
                work.work_id)
        self._refresh_projection_work(work.work_id)

    def _refresh_projection_work(self, work_id: str) -> None:
        previous = self._projection_rows.get(work_id)
        work = self._works.get(work_id)
        if work is None:
            if previous is None:
                return
            self._update_row_indexes(work_id, previous, None)
            del self._projection_rows[work_id]
            self._projection_order.pop(work_id, None)
            self._pricing_rows.pop(work_id, None)
            if previous.state in ("committed", "ready"):
                self._committed_not_issued -= 1
            self._bump_resource_versions(
                self._resource_index_keys(previous))
        else:
            row = self._projection_row(work)
            self._update_row_indexes(work_id, previous, row)
            self._projection_rows[work_id] = row
            if previous is None:
                self._projection_order[work_id] = self._next_projection_order
                self._next_projection_order += 1
            if previous is not None and previous.state in ("committed", "ready"):
                self._committed_not_issued -= 1
            if row.state in ("committed", "ready"):
                self._committed_not_issued += 1
            if self._is_pricing_row(row):
                self._pricing_rows[work_id] = row
            else:
                self._pricing_rows.pop(work_id, None)
            current_keys = self._resource_index_keys(row)
            previous_keys = (
                self._resource_index_keys(previous)
                if previous is not None else ())
            if current_keys == previous_keys:
                self._bump_resource_versions(current_keys)
            else:
                self._bump_resource_versions(
                    set(current_keys).union(previous_keys))
        self._projection_cache = None
        self._projection_state_cache = None
        self._pricing_snapshot_cache = None

    def pricing_snapshot(self) -> LedgerPricingSnapshot:
        """Return only the current resource frontier used by online pricing.

        This hot view avoids rebuilding the complete work/index projection and
        its historical node rows on every delivery. Row membership and the
        committed-not-issued count are maintained incrementally as events land.
        """
        if self._pricing_snapshot_cache is None:
            ordered_rows = sorted(
                ((self._projection_order[work_id], row)
                 for work_id, row in self._pricing_rows.items()),
                key=lambda item: item[0])
            self._pricing_snapshot_cache = LedgerPricingSnapshot(
                pricing_rows=tuple(row for _order, row in ordered_rows),
                committed_not_issued=self._committed_not_issued)
        return self._pricing_snapshot_cache

    def projection_snapshot(self) -> LedgerProjection:
        """Return a stable point-in-time row/index view, cached until mutation."""
        if self._projection_cache is None:
            rows = tuple(self._projection_rows.values())
            pricing_rows = tuple(
                row for row in rows if self._is_pricing_row(row))
            committed_not_issued = sum(
                row.state in ("committed", "ready") for row in rows)
            frozen_indexes = {
                name: MappingProxyType({
                    key: tuple(bucket.values()) for key, bucket in buckets.items()
                })
                for name, buckets in self._projection_indexes.items()
            }
            self._projection_cache = LedgerProjection(
                rows=rows,
                pricing_rows=pricing_rows,
                committed_not_issued=committed_not_issued,
                resource_index=frozen_indexes["resource_index"],
                request_stage_index=frozen_indexes["request_stage_index"],
                rank_index=frozen_indexes["rank_index"],
                dependency_index=frozen_indexes["dependency_index"],
                release_condition_index=frozen_indexes[
                    "release_condition_index"],
                unfinished_state_index=frozen_indexes[
                    "unfinished_state_index"],
                resource_version_vector=tuple(sorted(
                    self._resource_key_versions.items())),
            )
        return self._projection_cache

    def snapshot_state(self) -> Mapping[str, Any]:
        """Return cached immutable ledger state for read-only snapshot builders."""
        if self._projection_state_cache is None:
            self._projection_state_cache = _freeze_state_tree(self.export_state())
        return self._projection_state_cache

    # ------------------------------------------------------------ 应用 --

    def _apply_event(self, event: _TypedLedgerEvent) -> bool:
        """幂等应用一条事件；返回 False = 重放（零副作用）。

        门（与 ``online_scheduler_base._gate_delivery_sequence`` 同型）：

        * local_seq 必须 == 期望值（全量装载后 = event_watermark 起步，
          严格 +1）；紧邻重复（== 期望值-1 且与最后一条已应用事件内容
          一致）= 生产者重试追加，重放容忍零副作用；同 seq 不同内容
          fail-closed；
        * 事件 tick < 账本水位 tick fail-closed（时间倒退）。
        """
        producer = event.producer
        local_seq = event.local_seq
        if local_seq < 0:
            raise WorkLedgerError(
                "ledger event local_seq must be non-negative, got {!r}"
                .format(local_seq))
        if self._last_event is not None:
            last_producer, last_seq, last_content = self._last_event
            if local_seq == last_seq and producer == last_producer:
                if (_legacy_canonical_json(event.canonical_content)
                        == _legacy_canonical_json(last_content)):
                    self.replayed_event_count += 1
                    return False  # 紧邻重复：重放容忍
                raise WorkLedgerError(
                    "duplicate ledger event (work_id={}, kind={}, local_seq="
                    "{}) carries different content -- idempotent replay must "
                    "match the applied event".format(
                        event.work_id, event.kind, local_seq))
        if self._expected_next_local_seq is not None:
            expected = self._expected_next_local_seq
            if local_seq != expected:
                raise WorkLedgerError(
                    "ledger event local_seq gap: got {} expected {} "
                    "(producer {!r}; monotonic +1, adjacent duplicates "
                    "replay only)".format(local_seq, expected, producer))
        else:
            # 首个 delta 且无全量基线：以本事件 seq 为基线（官方对端总是
            # 首交付全量，此分支只服务嵌入/测试路径）。
            self._expected_next_local_seq = local_seq
        if event.tick < self.last_tick:
            raise WorkLedgerError(
                "ledger event tick regressed: {} after {} (work_id={})"
                .format(event.tick, self.last_tick, event.work_id))
        try:
            self._apply_event_semantics(event)
        except Exception:
            # A failed event can still have made a partial in-place mutation
            # (for example progress served arithmetic before a later residual
            # validation). Keep cached read views aligned with that prefix.
            self._refresh_projection_work(event.work_id)
            raise
        self._refresh_projection_work(event.work_id)
        self._last_event = (producer, local_seq, event.canonical_content)
        self._expected_next_local_seq = local_seq + 1
        self.last_tick = max(self.last_tick, event.tick)
        self.applied_event_count += 1
        return True

    def _apply_event_semantics(self, event: _TypedLedgerEvent) -> None:
        work = self._works.get(event.work_id)
        if event.kind == "commit":
            if work is not None:
                raise WorkLedgerError(
                    "commit event for already-known work {}".format(
                        event.work_id))
            work = (
                _work_from_compact_event(event)
                if event.compact_entry is not None
                else _work_from_event(event))
            self._works[event.work_id] = work
            for dependency in work.dependencies:
                self._dependents_by_work.setdefault(dependency, set()).add(
                    event.work_id)
            return
        if work is None:
            raise WorkLedgerError(
                "{} event for unknown work {} (full resync missing?)"
                .format(event.kind, event.work_id))
        source_states = _KIND_SOURCE_STATES.get(event.kind)
        if source_states is not None and work.state not in source_states:
            raise WorkLedgerError(
                "{} event on work {} in state {!r} (expected one of {!r})"
                .format(event.kind, event.work_id, work.state, source_states))
        if event.kind == "issue":
            work.transition("issued", event.tick)
        elif event.kind == "progress":
            delta = event.delta_served or WorkAmount()
            remaining_updates = {}
            for field_name, dimension in (
                    ("remaining_ops", "ops"),
                    ("remaining_bytes", "bytes"),
                    ("remaining_latency_ns", "latency_ns")):
                if field_name not in event.payload:
                    continue
                wire_remaining = _amount_number(
                    event.payload[field_name], "event." + field_name)
                delta_value = getattr(delta, dimension)
                current_value = getattr(work.remaining, dimension)
                if delta_value is None or current_value is None:
                    raise WorkLedgerError(
                        "progress event {} carries {} without a matching "
                        "remaining/delta dimension".format(
                            event.work_id, field_name))
                expected_remaining = current_value - delta_value
                tolerance = AMOUNT_EPSILON + 1e-9 * abs(wire_remaining)
                if abs(expected_remaining - wire_remaining) > tolerance:
                    raise WorkLedgerError(
                        "progress event {} {} disagrees with delta: "
                        "expected {!r}, got {!r}".format(
                            event.work_id, field_name, expected_remaining,
                            wire_remaining))
                remaining_updates[dimension] = wire_remaining
            work.apply_progress(delta, event.tick, progress_source="inferred")
            if remaining_updates:
                # Preserve the backend's exact tick-end residual after the
                # arithmetic check above.  Recompute served for those
                # dimensions so the conservation line remains exact after
                # replacing a rounded subtraction with the wire value.
                values = {
                    dimension: getattr(work.remaining, dimension)
                    for dimension in ("ops", "bytes", "latency_ns")}
                values.update(remaining_updates)
                work.remaining = WorkAmount(**values).validate()
                served_values = {
                    dimension: getattr(work.served, dimension)
                    for dimension in ("ops", "bytes", "latency_ns")}
                for dimension in remaining_updates:
                    total_value = getattr(work.total, dimension)
                    if total_value is not None:
                        served_values[dimension] = (
                            total_value - remaining_updates[dimension])
                work.served = WorkAmount(**served_values).validate()
        elif event.kind == "completion":
            work.finish(event.tick)
        elif event.kind == "ack":
            work.transition("acknowledged", event.tick)
        elif event.kind == "release":
            self._release_work(event.work_id, event.tick)

    def _release_work(self, work_id: str, tick: int) -> None:
        work = self._works.get(work_id)
        if work is None:
            raise WorkLedgerError(
                "release event for unknown work {}".format(work_id))
        if work.state == "released":
            return
        work.transition("released", tick)
        # §4.3:178 -- retired 对象保留仍被依赖的 release_condition：若任一
        # 活跃工作的依赖边指向本工作，其 release_condition 留档。
        if work.release_conditions:
            still_depended = any(
                dependent_id != work_id
                and dependent_id in self._works
                and self._works[dependent_id].state != "released"
                for dependent_id in self._dependents_by_work.get(work_id, ()))
            if still_depended:
                self._retained_obligations[work_id] = (
                    work.release_conditions)
        # 退休对象本体从活跃集合移除（有界内存）；仍被依赖的义务可经
        # retained_release_conditions() 审计（§4.3:178 与 NodeStore GC 协调：
        # GC 只回收 finished 且子节点全完的对象，义务天然结清）。
        for dependency in work.dependencies:
            dependents = self._dependents_by_work.get(dependency)
            if dependents is not None:
                dependents.discard(work_id)
                if not dependents:
                    self._dependents_by_work.pop(dependency, None)
                    # A released dependency is retained only while an active
                    # work still references it. Drop the compact release
                    # conditions as soon as the last reference disappears;
                    # leaving them behind pins one historical entry per
                    # virtual endpoint even though the public query filters
                    # those entries out.
                    self._retained_obligations.pop(dependency, None)
        del self._works[work_id]

    # ------------------------------------------------------------ 查询 --

    def iter_active_works(self):
        """活跃工作对象的只读迭代（W5：LedgerWorldSource 背景投影的
        低开销通道——不做 works_summary 的逐工作 dict 物化）。"""
        return iter(self._works.values())

    def resource_version_vector(self) -> Tuple[Tuple[str, int], ...]:
        """Stable per-resource version vector for decision cache keys."""
        return tuple(sorted(self._resource_key_versions.items()))

    def work(self, work_id: str) -> CommittedWork:
        try:
            return self._works[work_id]
        except KeyError:
            raise WorkLedgerError("unknown work {!r}".format(work_id))

    def works_summary(self) -> Dict[str, Dict[str, Any]]:
        return {wid: work.summary() for wid, work in self._works.items()}

    def export_state(self) -> Dict[str, Any]:
        """把活跃工作集合导出为 C++ ``work_ledger.state`` 同构快照形态。

        §4.1 DecisionSnapshot 的只读资源索引输入（``build_decision_
        snapshot`` 的 ``ledger_state``）：桥接协议首交付 full、其后恒
        delta——delta 不携带 state，快照消费面按本导出重建累积事实，
        不再静默空缺（§4.1:144 "仲裁及只读资源索引"不得名存实亡）。
        导出面：

        * ``nodes[]``：节点级工作（leg=None）按 rank 分组——身份 +
          issued/finished 状态布尔 + 静态总量 + canonical 依赖；
        * ``hbm[]``：``hbm:{KIND}`` 腿按 rank 分组——权威剩余三维 +
          as_of_tick（后端积分时点）；
        * ``pool[]``：``pool:{port}:{seq}`` 腿按 port 分组——bytes/
          remaining + C++ 生命周期字符串（in_service→Streaming /
          其余→LatencyWaiting）+ as_of_ns；
        * ``noc``：``noc:{flow}`` 腿——flow_id/total/remaining/route；
          ``links[]`` 由活跃流 route 聚合 active_flow_count（Python 账
          本无逐链路聚合账本，导出面确定性派生）；
        * ``slots``：Python 账本不持有槽位门状态——恒空列表（slot_gate
          列空是事实而非虚构，不补零冒装）。

        导出**只读**（不 mutate 账本）；与 :func:`parse_work_ledger_state`
        的读面同构（state 字符串/字段名对齐 C++ 序列化形态），但含
        additive 键（``dependencies``/``progress_source``/``work_id``）
        ——C++ 侧与全量装载路径均容忍未知键。
        """
        nodes_by_rank: Dict[int, List[Dict[str, Any]]] = {}
        hbm_by_rank: Dict[int, Dict[str, Any]] = {}
        pool_by_port: Dict[int, Dict[str, Any]] = {}
        noc_flows: List[Dict[str, Any]] = []
        link_flow_counts: Dict[int, int] = {}
        noc_as_of: Optional[int] = None

        def _total_field(amount: WorkAmount, name: str):
            value = getattr(amount, name)
            return value if value is not None else 0

        for work in self._works.values():
            identity = work.identity
            issued = work.state in ("issued", "in_service")
            finished = work.state in (
                "completed", "acknowledged", "released")
            leg = work.leg
            if leg is None:
                nodes_by_rank.setdefault(identity.rank, []).append({
                    "work_id": work.work_id,
                    "request_id": identity.request_id,
                    "stage": identity.stage,
                    "generation": int(identity.generation),
                    "rank": int(identity.rank),
                    "node_id": int(identity.node_id),
                    "kind": identity.operation or "invalid",
                    "issued": issued,
                    "finished": finished,
                    "num_ops": _total_field(work.total, "ops"),
                    "bytes": _total_field(work.total, "bytes"),
                    "runtime_ns": _total_field(work.total, "latency_ns"),
                    "dependencies": list(work.dependencies),
                    "progress_source": work.progress_source,
                })
                continue
            if leg.startswith("hbm:"):
                block = hbm_by_rank.setdefault(identity.rank, {
                    "rank": int(identity.rank), "as_of_tick": None,
                    "jobs": []})
                block["jobs"].append({
                    "work_id": work.work_id,
                    "node_id": int(identity.node_id),
                    "kind": leg.split(":", 1)[1],
                    "request_id": identity.request_id,
                    "stage": identity.stage,
                    "generation": int(identity.generation),
                    "num_ops": _total_field(work.total, "ops"),
                    "bytes": _total_field(work.total, "bytes"),
                    "total_latency_ns": _total_field(
                        work.total, "latency_ns"),
                    "remaining_ops": work.remaining.ops,
                    "remaining_bytes": work.remaining.bytes,
                    "remaining_latency_ns": work.remaining.latency_ns,
                    "state": work.state,
                    "progress_source": work.progress_source,
                })
                if work.as_of_tick is not None:
                    if (block["as_of_tick"] is None
                            or int(work.as_of_tick) > int(
                                block["as_of_tick"])):
                        block["as_of_tick"] = int(work.as_of_tick)
                continue
            if leg.startswith("pool:"):
                parts = leg.split(":")
                try:
                    port = int(parts[1])
                    issue_seq = int(parts[2]) if len(parts) > 2 else 0
                except (IndexError, ValueError):
                    raise WorkLedgerError(
                        "pool leg id {!r} must encode port:issue_seq"
                        .format(leg))
                block = pool_by_port.setdefault(port, {
                    "port_index": port, "as_of_ns": None, "jobs": []})
                block["jobs"].append({
                    "work_id": work.work_id,
                    "issue_seq": issue_seq,
                    "sys_id": int(identity.rank),
                    "node_id": int(identity.node_id),
                    "request_id": identity.request_id,
                    "stage": identity.stage,
                    "generation": int(identity.generation),
                    "bytes": _total_field(work.total, "bytes"),
                    "remaining_bytes": work.remaining.bytes,
                    "state": (
                        "Streaming" if work.state == "in_service"
                        else "LatencyWaiting"),
                    "progress_source": work.progress_source,
                })
                as_of_ns = getattr(work, "payload_as_of_tick", None)
                if as_of_ns is not None:
                    if (block["as_of_ns"] is None
                            or int(as_of_ns) > int(block["as_of_ns"])):
                        block["as_of_ns"] = int(as_of_ns)
                continue
            if leg.startswith("noc:"):
                flow_id = int(identity.node_id)
                route = tuple(int(link) for link in (work.route or ()))
                noc_flows.append({
                    "work_id": work.work_id,
                    "flow_id": flow_id,
                    "request_id": identity.request_id,
                    "stage": identity.stage,
                    "generation": int(identity.generation),
                    "total_bytes": _total_field(work.total, "bytes"),
                    "remaining_bytes": work.remaining.bytes,
                    "route": list(route),
                    "state": work.state,
                    "progress_source": work.progress_source,
                })
                for link in set(route):
                    link_flow_counts[link] = (
                        link_flow_counts.get(link, 0) + 1)
                as_of = getattr(work, "payload_as_of_tick", None)
                if as_of is not None:
                    if noc_as_of is None or int(as_of) > noc_as_of:
                        noc_as_of = int(as_of)
                continue
            # 未知 leg 前缀：导出面跳过（构建面只产 hbm:/pool:/noc: 三族；
            # 出现即对端协议漂移，由消费面按缺键披露，不虚构第四族）。
        return {
            "nodes": [
                {"rank": rank, "nodes": nodes_by_rank[rank]}
                for rank in sorted(nodes_by_rank)],
            "hbm": [
                hbm_by_rank[rank] for rank in sorted(hbm_by_rank)],
            "pool": [
                pool_by_port[port] for port in sorted(pool_by_port)],
            "noc": {
                "as_of_tick": noc_as_of,
                "flows": noc_flows,
                "links": [
                    {"link_id": link, "active_flow_count": count}
                    for link, count in sorted(link_flow_counts.items())],
            },
            "slots": [],
        }

    def retained_release_conditions(self) -> Dict[str, Tuple[str, ...]]:
        """仍被依赖的 retired 义务（§4.3:178 与 NodeStore GC 协调）。"""
        live_deps = set(self._dependents_by_work)
        return {
            wid: conditions
            for wid, conditions in self._retained_obligations.items()
            if wid in live_deps}

    def conservation_report(self) -> Dict[str, Any]:
        """守恒校验接口位：Σserved 增量 == Σremaining 减量的总账形式。

        对每个工作：served + remaining == total（权威维度；inferred 标记
        在场——真值来自快照 diff 时标，缺进度遥测区间不算误差）。任何
        维度不守恒 = 物理账本不合法，报告违反项（fail-closed 由调用方
        决定；本方法只报告，W3 消费时按 §4.3:176 拒绝提交）。

        ``stale_amount_works`` = 剩余量为**总量上界占位**（delta commit
        未携带权威 remaining_*，progress_source=stale）的活跃工作数——
        这些工作的"剩余量"过估首见前已服务量（§4.3:174 禁止把最初总
        字节当精确剩余），计数入审计披露面供对端修复核账。
        """
        violations = []
        inferred_count = 0
        stale_count = 0
        for work in self._works.values():
            if work.progress_source == "inferred":
                inferred_count += 1
            elif work.progress_source == "stale":
                stale_count += 1
            for dim in ("ops", "bytes", "latency_ns"):
                total = getattr(work.total, dim)
                served = getattr(work.served, dim)
                remaining = getattr(work.remaining, dim)
                if total is None:
                    continue
                served_v = served or 0
                remaining_v = (
                    remaining if remaining is not None
                    else total - served_v)
                if abs(served_v + remaining_v - total) > (
                        AMOUNT_EPSILON + 1e-9 * abs(total)):
                    violations.append({
                        "work_id": work.work_id,
                        "dimension": dim,
                        "total": total,
                        "served": served_v,
                        "remaining": remaining_v,
                    })
        return {
            "violations": violations,
            "conserved": not violations,
            "inferred_progress_works": inferred_count,
            "stale_amount_works": stale_count,
            "active_works": len(self._works),
        }


_OVERLAY_MISSING = object()


class _CowOverlay:
    """Small mutable mapping overlay that clones only accessed records."""

    def __init__(self, source, copier):
        self.source = source
        self.copier = copier
        self.writes = {}
        self.deleted = set()

    def __contains__(self, key):
        return key in self.writes or (key not in self.deleted and key in self.source)

    def __len__(self):
        return len(self.source) + len(self.writes.keys() - self.source.keys()) - len(
            self.deleted - self.writes.keys())

    def __getitem__(self, key):
        if key in self.writes:
            return self.writes[key]
        if key in self.deleted:
            raise KeyError(key)
        value = self.source[key]
        value = self.copier(value)
        self.writes[key] = value
        return value

    def get(self, key, default=None):
        if key in self.writes:
            return self.writes[key]
        if key in self.deleted or key not in self.source:
            return default
        value = self.copier(self.source[key])
        self.writes[key] = value
        return value

    def __setitem__(self, key, value):
        self.deleted.discard(key)
        self.writes[key] = value

    def __delitem__(self, key):
        if key not in self:
            raise KeyError(key)
        self.writes.pop(key, None)
        self.deleted.add(key)

    def setdefault(self, key, default=None):
        existing = self.get(key, _OVERLAY_MISSING)
        if existing is _OVERLAY_MISSING:
            self[key] = default
            return default
        return existing

    def pop(self, key, default=_OVERLAY_MISSING):
        if key not in self:
            if default is _OVERLAY_MISSING:
                raise KeyError(key)
            return default
        value = self[key]
        del self[key]
        return value

    def apply_to(self, target):
        for key in self.deleted:
            target.pop(key, None)
        target.update(self.writes)

class _WorkLedgerV2Transaction:
    """Prepared COW transaction; commit is bounded by touched works/indexes."""

    def __init__(self, ledger: WorkLedger):
        self.ledger = ledger
        self.shadow = copy.copy(ledger)
        self.shadow._works = _CowOverlay(ledger._works, copy.copy)
        self.shadow._dependents_by_work = _CowOverlay(
            ledger._dependents_by_work, set)
        self.shadow._retained_obligations = _CowOverlay(
            ledger._retained_obligations, tuple)
        self.touched_work_ids = set()
        # Compact staging bookkeeping: a staged-event count plus the real
        # typed DTO of the final node event (its canonical content is what
        # commit verifies and the ledger retains for adjacent replay).
        self._event_count = 0
        self._last_typed_event = None
        self.delivery_sequence = None
        self.tick = None
        self.logical_seq_end = None
        self.snapshot_kind = None
        self._committed = False
        self._receipt = None
        self._previous_seq = 0
        self._previous_tick = ledger.last_tick

    def stage_event(self, seq, event) -> None:
        """Validate and apply one typed node event to the detached COW view."""
        if self._committed or self.logical_seq_end is not None:
            raise WorkLedgerError("cannot stage events after transaction finish")
        if type(seq) is not int:
            seq = _wire_integer(seq, "v3 node logical_seq")
        if seq <= self._previous_seq or seq <= 0:
            raise WorkLedgerError(
                "v3 node transitions must preserve increasing logical_seq")
        if (event.local_seq != seq
                or event.delivery_sequence != self.delivery_sequence):
            raise WorkLedgerError(
                "v3 typed node event identity disagrees with envelope")
        if event.tick < self._previous_tick or event.tick > self.tick:
            raise WorkLedgerError(
                "v3 node event tick is outside monotonic envelope time")
        self.shadow._apply_event_semantics(event)
        self.touched_work_ids.add(event.work_id)
        self._event_count += 1
        self._last_typed_event = event
        self._previous_seq = seq
        self._previous_tick = event.tick

    def finish(self, logical_seq_end):
        logical_seq_end = _wire_integer(
            logical_seq_end, "v3.logical_seq_end")
        if self._previous_seq > logical_seq_end:
            raise WorkLedgerError(
                "v3 logical sequence end precedes a staged node event")
        self.logical_seq_end = logical_seq_end
        return self

    def commit(self, *, notify: bool = True) -> ApplyReceipt:
        if self._committed:
            raise WorkLedgerError("v2 WorkLedger transaction already committed")
        if self._event_count and (
                self._last_typed_event is None
                or self._last_typed_event.canonical_content is None):
            raise WorkLedgerError(
                "last v3 node event is missing adjacent-replay canonical content")
        ledger = self.ledger
        self.shadow._works.apply_to(ledger._works)
        self.shadow._dependents_by_work.apply_to(ledger._dependents_by_work)
        self.shadow._retained_obligations.apply_to(ledger._retained_obligations)
        for work_id in self.touched_work_ids:
            ledger._refresh_projection_work(work_id)
        ledger.applied_event_count += self._event_count
        if self.snapshot_kind == "full":
            ledger.full_load_count += 1
        else:
            ledger.delta_load_count += 1
        ledger.event_watermark = self.logical_seq_end
        ledger.last_delivery_sequence = self.delivery_sequence
        ledger.last_tick = max(ledger.last_tick, self.tick)
        ledger._expected_next_local_seq = self.logical_seq_end + 1
        if self._event_count:
            last = self._last_typed_event
            ledger._last_event = (last.producer, last.local_seq,
                                  last.canonical_content)
        if self._event_count:
            ledger.version += 1
        self._receipt = ApplyReceipt(
            snapshot_kind=self.snapshot_kind,
            applied_events=self._event_count, replayed_events=0,
            virtual_count=0, event_watermark=ledger.event_watermark,
            ledger_version=ledger.version)
        self._committed = True
        if notify:
            self.notify()
        return self._receipt

    def notify(self) -> None:
        if not self._committed:
            raise WorkLedgerError("cannot notify before v2 transaction commit")
        if self.ledger._event_sink is not None:
            self.ledger._event_sink({
                "delivery_sequence": self.delivery_sequence,
                "tick": self.tick,
                "snapshot_kind": self.snapshot_kind,
                "applied_events": self._event_count,
                "replayed_events": 0,
                "event_watermark": self.logical_seq_end,
                "active_works": len(self.ledger._works),
                "ledger_version": self.ledger.version,
            })


def _event_digest(event: LedgerEvent) -> str:
    import json
    return json.dumps(event.as_dict(), sort_keys=True)


def _freeze_json_value(value: Any) -> Any:
    """Type-preserving immutable form equivalent to sorted JSON structure."""
    if value is None:
        return ("null",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", str(value))
    if isinstance(value, float):
        return ("float", repr(value))
    if isinstance(value, str):
        return ("str", value)
    if issubclass(type(value), _RuntimeMapping):
        items = sorted(value.items(), key=lambda item: item[0])
        return ("object", tuple(
            (key, _freeze_json_value(item)) for key, item in items))
    if isinstance(value, (list, tuple)):
        return ("array", tuple(_freeze_json_value(item) for item in value))
    raise TypeError(
        "Object of type {} is not JSON serializable".format(
            type(value).__name__))


def _freeze_json_object_pairs(items) -> Any:
    return _frozen_json_object(tuple(
        (key, _freeze_json_value(value))
        for key, value in sorted(items, key=lambda item: item[0])))


def _frozen_json_object(items) -> Any:
    return ("object", tuple(sorted(items, key=lambda item: item[0])))


def _event_canonical_content(
    *,
    kind: str,
    work_id: str,
    tick: int,
    event_id_pairs,
    delta_served: Optional[WorkAmount],
    payload_items,
) -> Any:
    """Build the frozen sorted-JSON form without an event/payload mapping."""
    pairs = [
        ("kind", _freeze_json_value(kind)),
        ("work_id", _freeze_json_value(work_id)),
        ("tick", _freeze_json_value(tick)),
        ("event_id", _freeze_json_object_pairs(event_id_pairs)),
    ]
    if delta_served is not None:
        pairs.append(("delta_served", _freeze_json_value(delta_served.as_dict())))
    pairs.extend(
        (key, _freeze_json_value(value)) for key, value in payload_items)
    return _frozen_json_object(pairs)


def _canonical_json_value(value: Any) -> Any:
    """Unfreeze only on an adjacent replay to preserve legacy JSON comparison."""
    tag = value[0]
    if tag == "null":
        return None
    if tag == "bool":
        return value[1]
    if tag == "int":
        return int(value[1])
    if tag == "float":
        return float(value[1])
    if tag == "str":
        return value[1]
    if tag == "array":
        return [_canonical_json_value(item) for item in value[1]]
    if tag == "object":
        return {key: _canonical_json_value(item) for key, item in value[1]}
    raise AssertionError("unknown frozen JSON tag {!r}".format(tag))


def _legacy_canonical_json(value: Any) -> str:
    """Legacy ``json.dumps(event.as_dict(), sort_keys=True)`` on replay only."""
    import json
    return json.dumps(_canonical_json_value(value), sort_keys=True)


def validate_work_ledger_payload(
    payload: Any,
    delivery_sequence: int,
    tick: int,
) -> None:
    """桥侧校验入口（``online_scheduler_base._validate_schema`` 消费）。

    只校验不应用——在场则逐字段校验（additive 字段缺席 = 旧对端，由调用
    方记录缺席事实，不补零冒装）。
    """
    _validate_payload_envelope(payload, delivery_sequence, tick)


def _validate_payload_envelope(
    payload: Any,
    delivery_sequence: int,
    tick: int,
) -> None:
    if not issubclass(type(payload), _RuntimeMapping):
        raise WorkLedgerError(
            "work_ledger field must be a mapping, got {!r}".format(
                type(payload).__name__))
    unknown = set(payload) - {
        "schema_version", "bridge_protocol", "work_ledger_schema",
        "snapshot_kind", "delivery_sequence", "tick", "event_watermark",
        "state", "events", "compact"}
    if unknown:
        raise WorkLedgerError(
            "unknown work_ledger keys {!r}".format(sorted(unknown)))
    schema_version = _wire_integer(
        payload.get("schema_version"), "schema_version")
    if schema_version != WORK_LEDGER_FIELD_VERSION:
        raise WorkLedgerError(
            "work_ledger schema_version {!r} != {}".format(
                payload.get("schema_version"), WORK_LEDGER_FIELD_VERSION))
    if payload.get("bridge_protocol") != BRIDGE_PROTOCOL_VERSION:
        raise WorkLedgerError(
            "work_ledger bridge_protocol {!r} != {} (fail-closed; no "
            "auto-migration)".format(
                payload.get("bridge_protocol"), BRIDGE_PROTOCOL_VERSION))
    if payload.get("work_ledger_schema") != WORK_LEDGER_SCHEMA_VERSION:
        raise WorkLedgerError(
            "work_ledger work_ledger_schema {!r} != {}".format(
                payload.get("work_ledger_schema"), WORK_LEDGER_SCHEMA_VERSION))
    if payload.get("snapshot_kind") not in SNAPSHOT_KINDS:
        raise WorkLedgerError(
            "work_ledger snapshot_kind {!r} must be one of {}".format(
                payload.get("snapshot_kind"), list(SNAPSHOT_KINDS)))
    payload_delivery = _wire_integer(
        payload.get("delivery_sequence", -1), "delivery_sequence")
    expected_delivery = _wire_integer(
        delivery_sequence, "expected delivery_sequence")
    if payload_delivery != expected_delivery:
        raise WorkLedgerError(
            "work_ledger delivery_sequence {} != delta delivery_sequence {}"
                .format(payload.get("delivery_sequence"), delivery_sequence))
    payload_tick = _wire_integer(payload.get("tick", -1), "tick")
    expected_tick = _wire_integer(tick, "expected tick")
    if payload_tick != expected_tick:
        raise WorkLedgerError(
            "work_ledger tick {} != delta tick {}".format(
                payload.get("tick"), tick))


#: 资源腿 commit 的 C++ 侧生命周期字符串 → 账本状态机映射（pool 端口
#: job、NoC flow 与兼容旧端的生命周期名；其余未知名 fail-closed——
#: parse_work_ledger_state 的资源状态语义同源）。
_LEG_COMMIT_STATE_NAMES: Dict[str, str] = {
    "active": "in_service",
    "propagatingtail": "completed",
    "streaming": "in_service",
    "latencywaiting": "issued",
    "completed": "completed",
    "fluidcompleteawaitingcallback": "completed",
}


def _normalized_commit_state(state: Any) -> str:
    """commit 初始状态归一：状态机名直过；C++ 资源腿生命周期字符串
    映射；未知名 fail-closed（CommittedWork.__post_init__ 的域校验前置
    给出可读错误）。"""
    if not isinstance(state, str):
        raise WorkLedgerError(
            "commit state must be a string, got {!r}".format(state))
    if state in STATES:
        return state
    mapped = _LEG_COMMIT_STATE_NAMES.get(state.lower())
    if mapped is None:
        raise WorkLedgerError(
            "unknown commit state {!r} (states: {}; legacy leg names: {})"
            .format(state, list(STATES), sorted(_LEG_COMMIT_STATE_NAMES)))
    return mapped


def _work_from_event(event: _TypedLedgerEvent) -> CommittedWork:
    # (2026-10-01 审查修复) 身份字段读事件载荷**顶层**（C++ commit_payload
    # 的 kCarry 面：request_id/stage/generation/rank/node_id/operation 平
    # 铺，均在 EVENT_PAYLOAD_WHITELIST 内）。旧实现读不存在的嵌套
    # "identity" 子映射——白名单外键经 event_from_mapping fail-closed 拒
    # 收，子映射恒为空，decode 段（generation=1）的代际/操作被静默丢弃：
    # CommittedWork.work_id 属性与 _works 字典键分叉、parents 规范化用
    # 错代际生成悬空依赖 id、export_state 的 kind/generation/deps 全错。
    # 审计面扩展字段（session_id/turn_index 等）不在事件协议内，delta
    # 路径保持缺省（全量装载面由 parse_work_ledger_state 填充）。
    payload = event.payload
    # 剩余量来源（§4.3:174）：commit 携带后端已扣进度的权威剩余量
    # （remaining_ops/remaining_bytes/remaining_latency_ns 平铺面）时
    # 直接采用并标 authoritative；缺席时 remaining 退总量**上界占位**
    # ——"最初总字节"不是精确剩余（首见前已服务量不可恢复），以
    # stale/inferred 标记显式降级，消费面（LedgerWorldSource/守恒审计）
    # 据此披露过估不确定性，绝不冒充权威观测。
    remaining_input = WorkAmount(
        ops=(_amount_number(payload["remaining_ops"], "remaining_ops")
             if payload.get("remaining_ops") is not None else None),
        bytes=(_amount_number(payload["remaining_bytes"], "remaining_bytes")
               if payload.get("remaining_bytes") is not None else None),
        latency_ns=(_amount_number(payload["remaining_latency_ns"],
                                   "remaining_latency_ns")
                    if payload.get("remaining_latency_ns") is not None
                    else None),
    ).validate()
    return _work_from_commit_parts(
        work_id=event.work_id,
        tick=event.tick,
        request_id=payload.get("request_id", _request_of(event.work_id)),
        stage=payload.get("stage", _stage_of(event.work_id)),
        generation=payload.get(
            "generation", _generation_of(event.work_id)),
        rank=payload.get("rank", _rank_of(event.work_id)),
        node_id=payload.get(
            "node_id", payload.get("flow_id", _node_of(event.work_id))),
        operation=payload.get("operation", ""),
        leg=payload.get("leg"),
        total=amount_from_mapping(payload.get("total")),
        remaining_input=remaining_input,
        state=payload.get("state", "committed"),
        dependencies=payload.get("dependencies", ()),
        parents=payload.get("parents", ()),
        release_conditions=payload.get("release_conditions", ()),
        route=payload.get("route"),
        service_unit=payload.get("service_unit", "job"),
    )


def _work_from_compact_event(event: _TypedLedgerEvent) -> CommittedWork:
    """Create a commit directly from compact-v1 fields without a wire event."""
    entry = event.compact_entry
    assert entry is not None
    total = WorkAmount(
        ops=(_amount_number(entry["num_ops"], "num_ops")
             if entry.get("num_ops") is not None else None),
        bytes=(_amount_number(entry["bytes"], "bytes")
               if entry.get("bytes") is not None else None),
        latency_ns=(_amount_number(entry["runtime_ns"], "runtime_ns")
                    if entry.get("runtime_ns") is not None else None),
    ).validate()
    state = (
        "completed" if bool(entry.get("finished", False))
        else "issued" if bool(entry.get("issued", False))
        else "committed")
    return _work_from_commit_parts(
        work_id=event.work_id,
        tick=event.tick,
        request_id=entry.get("request_id", _request_of(event.work_id)),
        stage=entry.get("stage", _stage_of(event.work_id)),
        generation=entry.get("generation", _generation_of(event.work_id)),
        rank=entry.get("rank", _rank_of(event.work_id)),
        node_id=entry.get("node_id", _node_of(event.work_id)),
        operation=entry.get("operation", ""),
        leg=None,
        total=total,
        remaining_input=WorkAmount(),
        state=state,
        dependencies=(),
        parents=entry.get("parents", ()),
        release_conditions=(),
        route=None,
        service_unit="job",
    )


def _work_from_commit_parts(
    *,
    work_id: str,
    tick: int,
    request_id: Any,
    stage: Any,
    generation: Any,
    rank: Any,
    node_id: Any,
    operation: Any,
    leg: Any,
    total: WorkAmount,
    remaining_input: WorkAmount,
    state: Any,
    dependencies,
    parents,
    release_conditions,
    route,
    service_unit: Any,
) -> CommittedWork:
    identity = WorkIdentity(
        request_id=request_id,
        stage=stage,
        generation=int(generation),
        rank=int(rank),
        node_id=int(node_id),
        operation=operation,
        session_id="",
        turn_index=None,
        run_id="",
        resource_legs=(),
        shard=None,
        owner_key="",
        reservation_id=None,
    )
    state = _normalized_commit_state(state)
    # Explicit canonical dependencies take precedence; compact-v1 carries
    # source node ids and follows the legacy identity-based canonicalization.
    dependencies = tuple(dependencies)
    if not dependencies and parents:
        dependencies = tuple(
            canonical_work_id(
                identity.request_id, identity.stage, identity.generation,
                identity.rank, int(parent))
            for parent in parents)
    issued_state = state in ("issued", "in_service")
    if remaining_input != WorkAmount():
        remaining = remaining_input
        # 后端已扣进度的首见前已服务量 = total − remaining（逐维度；
        # total 缺维度时不虚构）——served+remaining==total 的守恒线自
        # 中途首见点保持（剩余为后端积分真值，其与初始总量的差 = 首见
        # 前已服务；remaining > total = 物理不合法，WorkAmount 校验
        # fail-closed）。
        served = total.minus(remaining_input)
        progress_source = "authoritative"
    else:
        remaining = total
        served = WorkAmount()
        progress_source = "inferred" if state == "in_service" else "stale"
    work = CommittedWork(
        identity=identity,
        state="committed",
        total=total,
        served=served,
        remaining=remaining,
        progress_source=progress_source,
        service_unit=service_unit,
        dependencies=dependencies,
        release_conditions=tuple(release_conditions),
        route=(tuple(route) if route else None),
        leg=leg,
        committed_tick=tick,
    )
    if issued_state:
        work.transition("issued", tick)
    if state == "in_service":
        work.transition("in_service", tick)
    if state == "completed":
        # 中途首见已完成工作（快照帧间隔内 issue+finish+GC 全部发生）：
        # 重放到终态，剩余清零、已服务=总量。
        work.transition("issued", tick)
        work.finish(tick)
        work.served = work.total
    return work


def _split_work_id(work_id: str) -> List[str]:
    node = work_id.split("@", 1)[0]
    body, _, uid = node.rpartition(":")
    request, _, rest = body.partition("#")
    stage, _, generation = rest.partition(":")
    return [request, stage, generation, uid]


def _request_of(work_id: str) -> str:
    return _split_work_id(work_id)[0]


def _stage_of(work_id: str) -> str:
    return _split_work_id(work_id)[1]


def _generation_of(work_id: str) -> int:
    """work_id 的代际段（:generation:；NoC 腿等非 canonical 形态回退 0）。"""
    try:
        return int(_split_work_id(work_id)[2])
    except (IndexError, ValueError):
        return 0


def _rank_of(work_id: str) -> int:
    # NoC flow 腿的 work_id（"noc#<flow_id>@noc:<flow_id>"）不属 canonical
    # 节点格式：身份回退取 0（腿身份已由 leg/flow_id 表达，rank 仅是
    # Python 侧 WorkIdentity 的填充位）。
    try:
        uid = _split_work_id(work_id)[3]
        return int(uid[1:].split("n", 1)[0])
    except (IndexError, ValueError):
        return 0


def _node_of(work_id: str) -> int:
    try:
        uid = _split_work_id(work_id)[3]
        return int(uid.split("n", 1)[1])
    except (IndexError, ValueError):
        return 0


# ------------------------------------------------------- 全量状态解析 --

#: C++ ``work_ledger.state`` 四类快照的 kind 名称（与 main_online.cc /
#: DecisionBridge.cc 的序列化面一一对应；桥接两端 schema 同步测试消费）。
NODE_KIND_NAMES: Dict[int, str] = {
    0: "invalid",
    1: "metadata",
    2: "mem_load",
    3: "mem_store",
    4: "compute",
    5: "comm_send",
    6: "comm_recv",
    7: "comm_collective",
}


def _node_operation(kind_value: Any) -> str:
    if isinstance(kind_value, str):
        return kind_value
    try:
        return NODE_KIND_NAMES.get(int(kind_value), "invalid")
    except (TypeError, ValueError):
        return "invalid"


def _validate_wire_work_id(
    payload: Mapping[str, Any], expected: str, context: str,
) -> None:
    """Reject a full-frame id that disagrees with parsed identity fields.

    The C++ delta indexes later events by the wire id.  Replacing a bad full
    frame id with a locally reconstructed one would make that first frame look
    valid and then address a different key on the next progress/release event.
    """
    wire_id = payload.get("work_id")
    if wire_id is not None and str(wire_id) != expected:
        raise WorkLedgerError(
            "{} work_id {!r} does not match reconstructed identity {!r}"
            .format(context, wire_id, expected))


def parse_work_ledger_state(state: Any) -> Dict[str, CommittedWork]:
    """把 C++ 全量 ``work_ledger.state`` 解析为 CommittedWork 集合。

    覆盖四类快照 + 槽位：

    * ``nodes[]``：per-rank per-node 视图（NodeStore::work_ledger_view）。
      节点级工作：total = ops/bytes/runtime_ns（静态属性）；issued 标志
      → issued 态；finished → completed；remaining 不虚构（缺进度遥测
      标 stale，§4.3:174）；parents → dependencies。
    * ``hbm[]``：per-rank job 视图（LocalHbmBandwidthModel::job_snapshot）。
      腿级工作（leg=``hbm:{kind}``）：remaining 为后端积分真值
      （progress_source=authoritative）。
    * ``pool[]``：per-port job 视图（AnalyticalRemoteMemory::port_snapshot）。
      腿级工作（leg=``pool:{port}:{issue_seq}``）。
    * ``noc``：flow/link 视图（FluidScheduler::flow_snapshot）。
      腿级工作（leg=``noc:{flow_id}``）。
    """
    if not issubclass(type(state), _RuntimeMapping):
        raise WorkLedgerError(
            "work_ledger state must be a mapping, got {!r}".format(
                type(state).__name__))
    works: Dict[str, CommittedWork] = {}

    def _int_field(payload: Mapping[str, Any], key: str) -> int:
        value = payload.get(key, 0)
        return _wire_integer(value, "state." + key)

    for rank_block in state.get("nodes", ()):  # nodes: [{rank, nodes:[...]}]
        rank = _int_field(rank_block, "rank")
        for node in rank_block.get("nodes", ()):
            node_id = _int_field(node, "node_id")
            request_id = str(node.get("request_id", ""))
            stage = str(node.get("stage", ""))
            generation = _int_field(node, "generation")
            operation = _node_operation(node.get("kind"))
            issued = bool(node.get("issued", False))
            finished = bool(node.get("finished", False))
            parents = tuple(
                canonical_work_id(request_id, stage, generation, rank, pid)
                for pid in node.get("parents", ()))
            num_ops = node.get("num_ops")
            num_bytes = node.get("bytes")
            runtime_ns = node.get("runtime_ns")
            total = WorkAmount(
                ops=(_amount_number(num_ops, "node.num_ops")
                     if num_ops is not None else None),
                bytes=(_amount_number(num_bytes, "node.bytes")
                       if num_bytes is not None else None),
                latency_ns=(_amount_number(runtime_ns, "node.runtime_ns")
                            if runtime_ns is not None else None),
            ).validate()
            legs = []
            if operation == "compute":
                legs.append("hbm:COMPUTE")
            elif operation in ("mem_load", "mem_store"):
                legs.append("pool")
            elif operation in ("comm_send", "comm_recv", "comm_collective"):
                legs.append("noc")
            work = CommittedWork(
                identity=WorkIdentity(
                    request_id=request_id, stage=stage, generation=generation,
                    rank=rank, node_id=node_id, operation=operation,
                    resource_legs=tuple(legs)),
                state="committed",
                total=total,
                served=WorkAmount(),
                remaining=total,
                progress_source="stale",
                service_unit="ops" if operation == "compute" else "bytes",
                dependencies=parents,
                committed_tick=None,
            )
            if finished:
                work.state = "completed"
                work.remaining = WorkAmount()
                work.served = total
                work.progress_source = "inferred"
            elif issued:
                work.state = "issued"
            _validate_wire_work_id(node, work.work_id, "node")
            works[work.work_id] = work

    for hbm_block in state.get("hbm", ()):  # hbm: [{rank, jobs:[...]}]
        rank = _int_field(hbm_block, "rank")
        as_of_tick = hbm_block.get("as_of_tick")
        for job in hbm_block.get("jobs", ()):
            node_id = _int_field(job, "node_id")
            kind_name = str(job.get("kind", "COMPUTE"))
            leg = "hbm:{}".format(kind_name)
            request_id = str(job.get("request_id", ""))
            stage = str(job.get("stage", ""))
            generation = _int_field(job, "generation")
            remaining_bytes = job.get("remaining_bytes")
            remaining_ops = job.get("remaining_ops")
            latency_ns = job.get("remaining_latency_ns")
            remaining = WorkAmount(
                ops=(_amount_number(remaining_ops, "hbm.remaining_ops")
                     if remaining_ops is not None else None),
                bytes=(_amount_number(remaining_bytes, "hbm.remaining_bytes")
                       if remaining_bytes is not None else None),
                latency_ns=(_amount_number(latency_ns,
                                           "hbm.remaining_latency_ns")
                            if latency_ns is not None else None),
            ).validate()
            node_work = works.get(canonical_work_id(
                request_id, stage, generation, rank, node_id))
            base_identity = (
                node_work.identity if node_work is not None
                else WorkIdentity(
                    request_id=request_id, stage=stage, generation=generation,
                    rank=rank, node_id=node_id, operation=kind_name.lower()))
            # Current C++ snapshots carry the original HBM amounts alongside
            # residuals.  Older fixtures may omit them; retaining the
            # residual as a documented fallback keeps those inputs readable
            # while current mid-flight delta commits remain conserved.
            total = WorkAmount(
                ops=(_amount_number(job["num_ops"], "hbm.num_ops")
                     if job.get("num_ops") is not None
                     else remaining.ops),
                bytes=(_amount_number(job["bytes"], "hbm.bytes")
                       if job.get("bytes") is not None
                       else remaining.bytes),
                latency_ns=(_amount_number(job["total_latency_ns"],
                                           "hbm.total_latency_ns")
                            if job.get("total_latency_ns") is not None
                            else remaining.latency_ns),
            ).validate()
            state_name = str(job.get("state", "in_service"))
            if state_name not in ("issued", "in_service"):
                raise WorkLedgerError(
                    "work_ledger.state HBM job carries unknown state name "
                    "{!r} (expected 'issued' or 'in_service')"
                    .format(state_name))
            work = CommittedWork(
                identity=base_identity,
                state=state_name,
                total=total,
                served=total.minus(remaining),
                remaining=remaining,
                progress_source="authoritative",
                service_unit="bytes",
                leg=leg,
                committed_tick=None,
            )
            if as_of_tick is not None:
                work.as_of_tick = _wire_integer(as_of_tick,
                                                "hbm.as_of_tick")
            _validate_wire_work_id(job, work.work_id, "HBM")
            works[work.work_id] = work

    for port_block in state.get("pool", ()):
        port = _int_field(port_block, "port_index")
        as_of_ns = port_block.get("as_of_ns")
        for job in port_block.get("jobs", ()):
            issue_seq = _int_field(job, "issue_seq")
            # The C++ wire record uses rank; Python's historical export used
            # sys_id.  Prefer rank and only fall back when the compatibility
            # spelling is explicitly present, avoiding a silent rank-0 alias.
            rank = _int_field(job, "rank" if "rank" in job else "sys_id")
            node_id = _int_field(job, "node_id")
            leg = "pool:{}:{}".format(port, issue_seq)
            request_id = str(job.get("request_id", ""))
            stage = str(job.get("stage", ""))
            generation = _int_field(job, "generation")
            total_bytes = job.get("bytes")
            remaining_bytes = job.get("remaining_bytes")
            state_name = str(job.get("state", "LatencyWaiting"))
            # (2026-10-01 审查修复) 未知状态名 fail-closed：旧实现把任意
            # 未知名静默映射为 "issued"（保守方向），与 _LEG_COMMIT_STATE_
            # NAMES 的注释宣称（"其余未知名 fail-closed——两侧语义同源"）
            # 相反；C++ 新增状态名时应显式登记而非静默降级。
            if state_name not in (
                    "Streaming", "LatencyWaiting",
                    "FluidCompleteAwaitingCallback"):
                raise WorkLedgerError(
                    "work_ledger.state pool job carries unknown state "
                    "name {!r} (expected a registered remote-port state); "
                    "register the new state on both sides instead of "
                    "silently downgrading it".format(state_name))
            node_work = works.get(canonical_work_id(
                request_id, stage, generation, rank, node_id))
            base_identity = (
                node_work.identity if node_work is not None
                else WorkIdentity(
                    request_id=request_id, stage=stage, generation=generation,
                    rank=rank, node_id=node_id, operation="pool_read"))
            state_value = (
                "in_service" if state_name == "Streaming"
                else "issued" if state_name == "LatencyWaiting"
                else "completed")
            total = WorkAmount(
                bytes=(_amount_number(total_bytes, "pool.bytes")
                       if total_bytes is not None else None))
            remaining = WorkAmount(
                bytes=(_amount_number(remaining_bytes,
                                      "pool.remaining_bytes")
                       if remaining_bytes is not None else None))
            if state_value == "completed" and (
                    remaining.bytes is not None
                    and remaining.bytes > AMOUNT_EPSILON):
                raise WorkLedgerError(
                    "completed pool job retains positive remaining bytes")
            served = total.minus(remaining)
            work = CommittedWork(
                identity=base_identity,
                state=state_value,
                total=total,
                served=served,
                remaining=remaining,
                progress_source="authoritative",
                service_unit="bytes",
                leg=leg,
            )
            if as_of_ns is not None:
                work.payload_as_of_tick = _wire_integer(  # type: ignore[attr-defined]
                    as_of_ns, "pool.as_of_ns")
            _validate_wire_work_id(job, work.work_id, "pool")
            works[work.work_id] = work

    noc_block = state.get("noc") or {}
    as_of_tick = noc_block.get("as_of_tick")
    for flow in noc_block.get("flows", ()):
        flow_id = _int_field(flow, "flow_id")
        leg = "noc:{}".format(flow_id)
        total_bytes = flow.get("total_bytes")
        remaining_bytes = flow.get("remaining_bytes")
        route = tuple(
            _wire_integer(link, "noc.route")
            for link in flow.get("route", ()))
        rank = _int_field(flow, "rank") if "rank" in flow else 0
        state_name = str(flow.get("state", "Active"))
        if state_name not in ("Active", "PropagatingTail", "Completed"):
            raise WorkLedgerError(
                "work_ledger.state NoC flow carries unknown state {!r}; "
                "register the new state on both sides instead of silently "
                "downgrading it".format(state_name))
        total = WorkAmount(
            bytes=(_amount_number(total_bytes, "noc.total_bytes")
                   if total_bytes is not None else None))
        remaining = WorkAmount(
            bytes=(_amount_number(remaining_bytes,
                                  "noc.remaining_bytes")
                   if remaining_bytes is not None else None))
        state_value = (
            "in_service" if state_name == "Active" else "completed")
        if state_value == "completed" and (
                remaining.bytes is not None
                and remaining.bytes > AMOUNT_EPSILON):
            raise WorkLedgerError(
                "completed NoC flow retains positive remaining bytes")
        work = CommittedWork(
            identity=WorkIdentity(
                request_id=str(flow.get("request_id", "noc")),
                stage=str(flow.get("stage", "")),
                generation=_int_field(flow, "generation"),
                rank=rank, node_id=flow_id,
                operation="comm_send"),
            state=state_value,
            total=total,
            served=total.minus(remaining),
            remaining=remaining,
            progress_source="authoritative",
            service_unit="bytes",
            leg=leg,
            route=route,
        )
        if as_of_tick is not None:
            work.payload_as_of_tick = _wire_integer(  # type: ignore[attr-defined]
                as_of_tick, "noc.as_of_tick")
        _validate_wire_work_id(flow, work.work_id, "NoC flow")
        works[work.work_id] = work

    return works
