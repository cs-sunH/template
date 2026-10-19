"""Strict protocol-v3 archive and protocol-v4 live node/work-ledger consumers.

The executor owns mutable cursors.  This module accepts one immutable v3
archive or v4 delivery at a time, validates sparse changes against the last
accepted state, and keeps only current node events and resource rows.  The
pipeline descriptor sections (``descriptor_commits`` /
``pipeline_transition_runs`` / ``cursor_patches``) retired with the
remote-read mechanism: producers no longer emit rows, a frame may carry the
keys empty (or omit them), and any non-empty pipeline section fails closed.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from collections import namedtuple
from collections.abc import Mapping as _RuntimeMapping
from types import MappingProxyType
from typing import Any, Mapping, Optional

from joint.work_ledger import (
    EVENT_PAYLOAD_WHITELIST,
    WorkLedgerError,
    _typed_v3_node_event_from_validated_scalars,
    canonical_work_id,
    leg_work_id,
)


PRODUCER = "cpp_online"
SCHEMA_VERSION = 4
PROTOCOL_VERSION = 4
BRIDGE_PROTOCOL = "bridge-v4-20261011"
WORK_LEDGER_SCHEMA = "workledger-v4-20261011"
ARCHIVE_V3_SCHEMA_VERSION = 3
ARCHIVE_V3_PROTOCOL_VERSION = 3
ARCHIVE_V3_BRIDGE_PROTOCOL = "bridge-v3-20261009"
ARCHIVE_V3_WORK_LEDGER_SCHEMA = "workledger-v3-20261009"
DESCRIPTOR_VERSION = 1
U64_MAX = (1 << 64) - 1
I32_MAX = (1 << 31) - 1

_TOP_KEYS = frozenset({
    "producer", "schema_version", "protocol_version", "bridge_protocol",
    "work_ledger_schema", "epoch", "delivery_sequence", "base_watermark",
    "logical_seq_start", "logical_seq_end", "logical_event_count",
    "replay_token", "descriptor_version", "base_state_version",
    "result_state_version", "snapshot_kind", "tick", "descriptor_commits",
    "node_transitions", "pipeline_transition_runs", "cursor_patches",
    "active_resources", "optional_sha256",
})
# Sections/scalars that retired with the remote-read pipeline mechanism: a
# frame may omit them (new producers) or carry them empty (transition-era
# producers); non-empty pipeline rows fail closed below.
_OPTIONAL_KEYS = frozenset({
    "optional_sha256", "descriptor_version", "descriptor_commits",
    "pipeline_transition_runs", "cursor_patches",
})
_EMPTY_LEGAL_KEYS = frozenset({
    "descriptor_commits", "pipeline_transition_runs", "cursor_patches",
})
# Hoisted per-row key sets shared by the offline decoder (identical
# membership; the literal was previously rebuilt once per node row).
_NODE_ROW_KEYS = frozenset(
    {"logical_seq", "tick", "kind", "work_id", "payload"})
_NODE_EVENT_KINDS = frozenset({
    "commit", "issue", "progress", "completion", "ack", "release"})
_RESOURCE_COMMON = frozenset({
    "resource_type", "work_id", "request_id", "stage", "generation", "rank",
    "node_id", "operation", "leg", "state", "total", "remaining",
    "progress_source", "service_unit", "route", "as_of_tick", "issued_tick",
    "descriptor_segment_id", "endpoint_selector",
})
_RESOURCE_EXTRAS = {
    "noc_flow": frozenset({"flow_id", "current_rate_Bpns", "last_rate_update_time"}),
    "hbm_job": frozenset({"kind", "start_tick"}),
    "pool_job": frozenset({"port_index", "issue_seq", "latency_ready_ns"}),
}


class PipelineLedgerError(ValueError):
    """Malformed v3 pipeline envelope or inconsistent state transition."""


def _mapping(value: Any, path: str, keys: frozenset[str]) -> Mapping[str, Any]:
    value_type = type(value)
    if (value_type is not dict and value_type is not MappingProxyType
            and not issubclass(value_type, _RuntimeMapping)):
        raise PipelineLedgerError(f"{path} must be an object")
    missing = keys - set(value)
    extra = set(value) - keys
    if missing or extra:
        raise PipelineLedgerError(
            f"{path} keys invalid (missing={sorted(missing)}, unknown={sorted(extra)})")
    return value


def _array(value: Any, path: str) -> tuple:
    if not isinstance(value, (list, tuple)):
        raise PipelineLedgerError(f"{path} must be an array")
    return tuple(value)


def _uint(value: Any, path: str, maximum: int = U64_MAX) -> int:
    if type(value) is not int or not 0 <= value <= maximum:
        raise PipelineLedgerError(f"{path} must be uint in [0,{maximum}] (bool rejected)")
    return value


def _rank(value: Any, path: str) -> int:
    return _uint(value, path, I32_MAX)


def _finite_nonnegative(value: Any, path: str) -> int | float:
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise PipelineLedgerError(f"{path} must be finite nonnegative int/float (bool rejected)")
    return value


def _string(value: Any, path: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value):
        raise PipelineLedgerError(f"{path} must be {'a string' if empty else 'a non-empty string'}")
    return value


def _boolean(value: Any, path: str) -> bool:
    if type(value) is not bool:
        raise PipelineLedgerError(f"{path} must be bool")
    return value


def _nullable_uint(value: Any, path: str) -> Optional[int]:
    return None if value is None else _uint(value, path)


def _amount(value: Any, path: str) -> Mapping[str, Any]:
    row = _mapping(value, path, frozenset({"ops", "bytes", "latency_ns"}))
    for dimension in ("ops", "bytes", "latency_ns"):
        if row[dimension] is not None:
            _finite_nonnegative(row[dimension], f"{path}.{dimension}")
    return row


def _canonical_rows(rows, key, path):
    values = _array(rows, path)
    previous = None
    for row in values:
        current = key(row)
        if previous is not None and current <= previous:
            raise PipelineLedgerError(
                f"{path} must be sorted and duplicate-free")
        previous = current
    return values


def _canonical_envelope_bytes(value: Any) -> bytes:
    """One exact, bounded replay representation (JSON sorts keys and keeps
    integer/float spellings distinct, unlike Python equality)."""
    try:
        return json.dumps(value, sort_keys=True, separators=(",", ":"),
                          ensure_ascii=False, allow_nan=False).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise PipelineLedgerError(
            f"envelope cannot be represented as canonical JSON: {exc}") from exc


def _typed_equal(left: Any, right: Any) -> bool:
    """Compare parsed protocol values without Python's int/float coercion."""
    left_type, right_type = type(left), type(right)
    if ((left_type is dict or left_type is MappingProxyType)
            and (right_type is dict or right_type is MappingProxyType)):
        return (left.keys() == right.keys()
                and all(_typed_equal(left[key], right[key]) for key in left))
    if (issubclass(left_type, _RuntimeMapping)
            and issubclass(right_type, _RuntimeMapping)):
        return (left.keys() == right.keys()
                and all(_typed_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return (len(left) == len(right)
                and all(_typed_equal(a, b) for a, b in zip(left, right)))
    if left_type is not right_type:
        return False
    return left == right


def _freeze_dto(value: Any) -> Any:
    value_type = type(value)
    if value_type is dict or value_type is MappingProxyType:
        return MappingProxyType({key: _freeze_dto(item) for key, item in value.items()})
    if issubclass(value_type, _RuntimeMapping):
        return MappingProxyType({key: _freeze_dto(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_dto(item) for item in value)
    return value


@dataclass(frozen=True)
class PipelineLedgerReceipt:
    snapshot_kind: str
    delivery_sequence: int
    logical_event_count: int
    base_watermark: int
    event_watermark: int
    base_state_version: int
    state_version: int
    replayed: bool = False


@dataclass(frozen=True)
class _ParsedEnvelope:
    canonical_bytes: bytes
    replay_token: str
    producer: str
    epoch: str
    delivery_sequence: int
    snapshot_kind: str
    tick: int
    base_watermark: int
    seq_start: int
    seq_end: int
    event_count: int
    base_version: int
    result_version: int
    protocol_version: int
    node_events: tuple
    resources: tuple


@dataclass(frozen=True)
class PreparedPipelineUpdate:
    parsed: _ParsedEnvelope
    resources: tuple
    changed: bool
    replayed: bool = False


class _MaterializedNodeSource:
    """Offline node row source over pre-parsed ``(seq, typed event)`` pairs."""

    __slots__ = ("events", "index")

    def __init__(self, events):
        self.events = events
        self.index = 0

    @property
    def total(self):
        return len(self.events)

    def peek_seq(self):
        if self.index < len(self.events):
            return self.events[self.index][0]
        return None

    def take(self, work_transaction):
        seq, event = self.events[self.index]
        self.index += 1
        before = work_transaction.shadow._works.get(event.work_id)
        before_dependencies = (() if before is None else
                               tuple(before.dependencies))
        work_transaction.stage_event(seq, event)
        return (seq, event.kind, event.work_id, event.payload, before,
                before_dependencies)


def _parse_resources(value: Any, path: str) -> tuple:
    rows = _array(value, path)
    parsed = []
    allowed_states = {
        "noc_flow": {"Active", "PropagatingTail", "Completed"},
        "hbm_job": {"issued", "in_service"},
        "pool_job": {"LatencyWaiting", "Streaming", "FluidCompleteAwaitingCallback"},
    }
    for i, raw in enumerate(rows):
        p = f"{path}[{i}]"
        if not issubclass(type(raw), _RuntimeMapping):
            raise PipelineLedgerError(f"{p} must be an object")
        kind = raw.get("resource_type")
        if kind not in _RESOURCE_EXTRAS:
            raise PipelineLedgerError(f"{p}.resource_type is unknown")
        row = _mapping(raw, p, _RESOURCE_COMMON | _RESOURCE_EXTRAS[kind])
        if row["state"] not in allowed_states[kind]:
            raise PipelineLedgerError(f"{p}.state is invalid for {kind}")
        for field in ("work_id", "request_id", "operation", "leg", "progress_source", "service_unit"):
            _string(row[field], p + "." + field)
        if not isinstance(row["stage"], str):
            raise PipelineLedgerError(f"{p}.stage must be a string")
        if row["service_unit"] != "bytes":
            raise PipelineLedgerError(f"{p}.service_unit must be bytes")
        _uint(row["generation"], p + ".generation")
        _rank(row["rank"], p + ".rank")
        _uint(row["node_id"], p + ".node_id")
        _uint(row["as_of_tick"], p + ".as_of_tick")
        _nullable_uint(row["issued_tick"], p + ".issued_tick")
        if row["progress_source"] != "authoritative":
            raise PipelineLedgerError(f"{p}.progress_source must be authoritative")
        _amount(row["total"], p + ".total")
        _amount(row["remaining"], p + ".remaining")
        route = _array(row["route"], p + ".route")
        for j, link in enumerate(route):
            _uint(link, f"{p}.route[{j}]")
        if row["descriptor_segment_id"] is not None \
                or row["endpoint_selector"] is not None:
            # The pipeline descriptor sections retired with remote-read;
            # a resource row still claiming a descriptor identity is an
            # unsupported old-producer frame and fails closed.
            raise PipelineLedgerError(
                f"{p} carries a retired pipeline descriptor identity")
        is_unattributed_noc = (
            kind == "noc_flow" and row["request_id"] == "noc"
            and row["stage"] == "" and row["generation"] == 0)
        # §14.3 裁决的外生背景 hbm 腿（Agent R C++ 导出侧同步）：
        # 身份解析元数据失败且 transfer-sink COMM_READ/WRITE 的后端
        # job 按 issue_seq 真实身份字面导出——恰此一种空 stage hbm 行
        # 合法，其余空 stage hbm 行保持 fail-closed。
        is_external_background_hbm = (
            kind == "hbm_job"
            and row["request_id"] == "external_background"
            and row["stage"] == ""
            and row["generation"] == 0)
        if not (is_unattributed_noc or is_external_background_hbm) \
                and row["stage"] not in (
                    "prefill", "decode", "completion"):
            raise PipelineLedgerError(f"{p}.stage is invalid")
        if kind != "noc_flow" and route:
            raise PipelineLedgerError(f"{p} non-NoC route must be empty")
        expected_work_id = leg_work_id(
            canonical_work_id(row["request_id"], row["stage"],
                             row["generation"], row["rank"], row["node_id"]),
            row["leg"])
        if row["work_id"] != expected_work_id:
            raise PipelineLedgerError(f"{p}.work_id is not canonical for its typed identity/leg")
        if kind != "hbm_job":
            for amount_name in ("total", "remaining"):
                amount = row[amount_name]
                if amount["ops"] is not None or amount["latency_ns"] is not None:
                    raise PipelineLedgerError(f"{p}.{amount_name} for {kind} may define bytes only")
        if kind == "noc_flow":
            flow_id = _uint(row["flow_id"], p + ".flow_id")
            if row["leg"] != f"noc:{flow_id}":
                raise PipelineLedgerError(f"{p}.leg must identify its original flow_id")
            if not (
                    row["request_id"] == "noc" and row["stage"] == ""
                    and row["generation"] == 0 and row["node_id"] == flow_id):
                raise PipelineLedgerError(
                    f"{p} unattributed NoC identity must preserve noc/empty-stage/flow-id form")
            _finite_nonnegative(row["current_rate_Bpns"], p + ".current_rate_Bpns")
            _finite_nonnegative(row["last_rate_update_time"], p + ".last_rate_update_time")
            if row["state"] == "Completed" and row["remaining"]["bytes"] != 0:
                raise PipelineLedgerError(f"{p} completed NoC tail must have zero remaining bytes")
        elif kind == "hbm_job":
            _string(row["kind"], p + ".kind")
            if row["request_id"] == "external_background" \
                    and row["stage"] == "" and row["generation"] == 0:
                # §14.3 外生背景腿的字面身份：rank 必须与行自身一致，
                # issue_seq 为正 uint（后端 job 序号）；kind 仍须是
                # transfer-sink 的 COMM_READ/COMM_WRITE。其余校验
                # （amounts/canonical work_id/start_tick）与普通 hbm 行
                # 同规，不放宽。
                if row["kind"] not in ("COMM_READ", "COMM_WRITE"):
                    raise PipelineLedgerError(
                        f"{p} external background HBM job kind must be a "
                        "transfer-sink COMM_READ/COMM_WRITE")
                leg_rank, _, leg_seq = row["leg"].partition("#")
                leg_rank = leg_rank.partition(":")
                if (not str(row["leg"]).startswith("hbm_external:")
                        or not leg_seq.isdigit() or int(leg_seq) < 1
                        or leg_rank[0] != "hbm_external"
                        or not leg_rank[2].isdigit()
                        or int(leg_rank[2]) != row["rank"]):
                    raise PipelineLedgerError(
                        f"{p}.leg must be the literal external background "
                        f"form hbm_external:{{rank}}#{{issue_seq}} with the "
                        "row's own rank and a positive backend issue_seq")
            elif row["leg"] != f"hbm:{row['kind']}":
                raise PipelineLedgerError(f"{p}.leg must identify its HBM job kind")
            _uint(row["start_tick"], p + ".start_tick")
        else:
            port = _uint(row["port_index"], p + ".port_index")
            issue_seq = _uint(row["issue_seq"], p + ".issue_seq")
            if row["leg"] != f"pool:{port}:{issue_seq}":
                raise PipelineLedgerError(f"{p}.leg must identify its original pool job")
            _finite_nonnegative(row["latency_ready_ns"], p + ".latency_ready_ns")
        for dimension in ("ops", "bytes", "latency_ns"):
            total = row["total"][dimension]
            remaining = row["remaining"][dimension]
            if (total is None) != (remaining is None):
                raise PipelineLedgerError(f"{p} changes {dimension} applicability")
            if total is not None and remaining > total:
                raise PipelineLedgerError(f"{p} remaining {dimension} exceeds total")
        parsed.append(dict(row))
    return _canonical_rows(parsed, lambda r: (r["rank"], r["node_id"], r["leg"], r["resource_type"]), path)


_EnvelopeHeader = namedtuple("_EnvelopeHeader", (
    "row", "canonical_bytes", "replay_token", "epoch", "delivery_sequence",
    "snapshot_kind", "tick", "base_watermark", "seq_start", "seq_end",
    "event_count", "base_version", "result_version"))


def _validate_envelope_header(envelope: Any, *, protocol_version: int
                              ) -> _EnvelopeHeader:
    """Validate every envelope-level scalar field and derive header facts.

    Shared by the exact offline decoder (``_parse_envelope``) and any fused
    production path.  ``canonical_bytes`` is computed here so both consumers
    share one bounded replay representation; the per-row sections stay owned
    by their callers.
    """
    if protocol_version not in (3, 4):
        raise PipelineLedgerError("unsupported pipeline ledger decoder version")
    if not issubclass(type(envelope), _RuntimeMapping):
        raise PipelineLedgerError("pipeline ledger envelope must be an object")
    required = _TOP_KEYS - _OPTIONAL_KEYS
    missing = required - set(envelope)
    unknown = set(envelope) - _TOP_KEYS
    if missing or unknown:
        raise PipelineLedgerError(
            "pipeline ledger envelope keys invalid "
            f"(missing={sorted(missing)}, unknown={sorted(unknown)})")
    row = envelope
    if protocol_version == 4:
        expected_identity = (4, 4, "bridge-v4-20261011",
                             "workledger-v4-20261011")
    else:
        expected_identity = (3, 3, "bridge-v3-20261009",
                             "workledger-v3-20261009")
    if (row["producer"] != PRODUCER
            or _uint(row["schema_version"], "schema_version")
                != expected_identity[0]
            or _uint(row["protocol_version"], "protocol_version")
                != expected_identity[1]):
        raise PipelineLedgerError(
            f"unsupported v{protocol_version} producer/schema/protocol")
    if (row["bridge_protocol"] != expected_identity[2]
            or row["work_ledger_schema"] != expected_identity[3]):
        raise PipelineLedgerError(
            f"unsupported v{protocol_version} bridge or work-ledger schema")
    if "descriptor_version" in row and _uint(
            row["descriptor_version"], "descriptor_version") != DESCRIPTOR_VERSION:
        raise PipelineLedgerError("unsupported descriptor_version")
    for key in _EMPTY_LEGAL_KEYS:
        if key in row and len(row[key]):
            raise PipelineLedgerError(
                f"{key} retired with the remote-read pipeline mechanism; "
                "non-empty pipeline rows are rejected")
    epoch = _string(row["epoch"], "epoch")
    delivery = _uint(row["delivery_sequence"], "delivery_sequence")
    tick = _uint(row["tick"], "tick")
    kind = row["snapshot_kind"]
    if kind not in ("full", "delta"):
        raise PipelineLedgerError("snapshot_kind must be full or delta")
    watermark = _uint(row["base_watermark"], "base_watermark")
    seq_start = _uint(row["logical_seq_start"], "logical_seq_start")
    seq_end = _uint(row["logical_seq_end"], "logical_seq_end")
    count = _uint(row["logical_event_count"], "logical_event_count")
    base_version = _uint(row["base_state_version"], "base_state_version")
    result_version = _uint(row["result_state_version"], "result_state_version")
    replay = _string(row["replay_token"], "replay_token")
    if replay != f"{PRODUCER}/{epoch}/{delivery}":
        raise PipelineLedgerError("replay_token does not match producer/epoch/delivery")
    if (count == 0 and (seq_start != watermark + 1 or seq_end != watermark)):
        raise PipelineLedgerError("empty delta sequence interval is invalid")
    if (count > 0 and (seq_start != watermark + 1 or seq_end != seq_start + count - 1)):
        raise PipelineLedgerError("logical sequence interval/count is inconsistent")
    if (kind == "full" and watermark != 0) or result_version not in (base_version, base_version + 1):
        raise PipelineLedgerError("full watermark or state-version transition is invalid")
    if "optional_sha256" in row:
        digest = row["optional_sha256"]
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            raise PipelineLedgerError("optional_sha256 must be 64 lowercase hexadecimal characters")
        without_digest = {k: v for k, v in row.items() if k != "optional_sha256"}
        actual = hashlib.sha256(_canonical_envelope_bytes(without_digest)).hexdigest()
        if actual != digest:
            raise PipelineLedgerError("optional_sha256 does not match canonical envelope JSON")
    return _EnvelopeHeader(
        row=row,
        canonical_bytes=_canonical_envelope_bytes(row),
        replay_token=replay,
        epoch=epoch,
        delivery_sequence=delivery,
        snapshot_kind=kind,
        tick=tick,
        base_watermark=watermark,
        seq_start=seq_start,
        seq_end=seq_end,
        event_count=count,
        base_version=base_version,
        result_version=result_version)


def _parse_envelope(envelope: Any, *, protocol_version: int = 3
                    ) -> _ParsedEnvelope:
    """Exact offline decoder: materialize every node event."""
    header = _validate_envelope_header(
        envelope, protocol_version=protocol_version)
    row = header.row
    delivery = header.delivery_sequence
    tick = header.tick
    node_rows = _array(row["node_transitions"], "node_transitions")
    node_events = []
    allowed_payload = EVENT_PAYLOAD_WHITELIST - {"kind", "work_id", "tick", "event_id"}
    for i, raw in enumerate(node_rows):
        p = f"node_transitions[{i}]"
        event = _mapping(raw, p, _NODE_ROW_KEYS)
        seq = _uint(event["logical_seq"], p + ".logical_seq")
        etick = _uint(event["tick"], p + ".tick")
        if seq == 0 or etick > tick:
            raise PipelineLedgerError(f"{p} logical sequence/tick is invalid")
        event_kind = _string(event["kind"], p + ".kind")
        if event_kind not in _NODE_EVENT_KINDS:
            raise PipelineLedgerError(f"{p}.kind is unknown")
        work_id = _string(event["work_id"], p + ".work_id")
        payload = event["payload"]
        if (not issubclass(type(payload), _RuntimeMapping)
                or not all(isinstance(key, str) for key in payload)):
            raise PipelineLedgerError(f"{p}.payload must be a string-keyed object")
        unknown = set(payload) - allowed_payload
        if unknown:
            raise PipelineLedgerError(f"{p}.payload has unknown keys {sorted(unknown)}")
        try:
            # The row, scalar headers and body key whitelist above establish
            # the v3 header contract before entering the private constructor.
            # WorkTxn retains only its final node event for adjacent legacy
            # replay. The full frame canonical bytes above still validate all
            # input fields before any transaction begins.
            typed = _typed_v3_node_event_from_validated_scalars(
                kind=event_kind,
                work_id=work_id,
                tick=etick,
                delivery_sequence=delivery,
                producer=PRODUCER,
                local_seq=seq,
                payload=payload,
                build_canonical_content=(i == len(node_rows) - 1))
        except (WorkLedgerError, TypeError, ValueError) as exc:
            raise PipelineLedgerError(f"{p} is not a valid typed node event: {exc}") from exc
        if node_events and seq <= node_events[-1][0]:
            raise PipelineLedgerError(
                "node_transitions must preserve unique logical sequence order")
        node_events.append((seq, typed))
    resources = _parse_resources(
        row["active_resources"], "active_resources")
    return _ParsedEnvelope(
        canonical_bytes=header.canonical_bytes,
        replay_token=header.replay_token,
        producer=PRODUCER, epoch=header.epoch,
        delivery_sequence=delivery, snapshot_kind=header.snapshot_kind,
        tick=tick,
        base_watermark=header.base_watermark, seq_start=header.seq_start,
        seq_end=header.seq_end,
        event_count=header.event_count, base_version=header.base_version,
        result_version=header.result_version, protocol_version=protocol_version,
        node_events=tuple(node_events), resources=resources)


class PipelineLedger:
    """Exact v3 archive node/resource view.

    ``apply_envelope`` first validates an envelope into detached typed values.
    It does not mutate a WorkLedger; OnlineSchedulerBase coordinates its staged
    node-event transaction and this bounded state commit atomically.

    ``validate_envelope``/``_parse_envelope`` followed by ``prepare_envelope``
    over the materialized rows is the single decoder path: production, tests
    and offline archive comparison all share it.
    """

    PROTOCOL_VERSION = 3

    def __init__(self) -> None:
        self.initialized = False
        self.epoch: Optional[str] = None
        self.delivery_sequence = -1
        self.watermark = 0
        self.state_version = 0
        self.last_tick = -1
        self.last_replay_token: Optional[str] = None
        self._last_content = None
        self._active_resources = ()
        self.replay_count = 0
        self._snapshot_state_cache = None
        self._pricing_snapshot_cache = None

    def validate_envelope(self, envelope: Any) -> _ParsedEnvelope:
        """Parse a whole input without modifying accepted state."""
        return _parse_envelope(
            envelope, protocol_version=self.PROTOCOL_VERSION)

    def prepare_envelope(self, envelope: Any, *,
                         work_transaction=None,
                         _parsed: Optional[_ParsedEnvelope] = None
                         ) -> PreparedPipelineUpdate:
        """Validate and COW-stage one globally ordered v3 delivery.

        ``validate_envelope``/``_parse_envelope`` materializes every node
        event DTO first, then this method checks the global sequence and
        applies the staged transitions through the shared ``_stage_delivery``
        core.
        """
        if _parsed is None:
            parsed = self.validate_envelope(envelope)
        elif isinstance(_parsed, _ParsedEnvelope):
            # Internal fast path for the scheduler: it parses once, keeps this
            # DTO local to the same synchronous call, then stages it here.
            # State/version/sequence checks below still run in full.
            parsed = _parsed
        else:
            raise PipelineLedgerError(
                "internal parsed envelope must be a validated pipeline DTO")
        replay = self._gate_delivery(parsed)
        if replay is not None:
            return replay

        if len(parsed.node_events) != parsed.event_count:
            raise PipelineLedgerError(
                "logical event count differs from typed transition count")
        expected_seq = parsed.seq_start
        previous_tick = self.last_tick
        for seq, event in parsed.node_events:
            if seq != expected_seq:
                raise PipelineLedgerError(
                    "node transitions do not form one global logical sequence")
            if event.tick < previous_tick or event.tick > parsed.tick:
                raise PipelineLedgerError(
                    "global node transition ticks are not monotonic")
            previous_tick = event.tick
            expected_seq += 1
        if expected_seq != parsed.seq_end + 1:
            raise PipelineLedgerError(
                "logical event count does not cover the declared sequence interval")
        if parsed.node_events and work_transaction is None:
            raise PipelineLedgerError(
                "v3 node transitions require a coordinated WorkLedger COW transaction")

        return self._stage_delivery(
            parsed,
            work_transaction=work_transaction,
            node_source=_MaterializedNodeSource(parsed.node_events))

    def _replay_update(self, parsed: _ParsedEnvelope
                       ) -> PreparedPipelineUpdate:
        """Zero-change update for an idempotent replay of the last delivery."""
        return PreparedPipelineUpdate(
            parsed=parsed,
            resources=self._active_resources,
            changed=False, replayed=True)

    def _gate_delivery(self, parsed: _ParsedEnvelope
                       ) -> Optional[PreparedPipelineUpdate]:
        """Shared delivery gates for both the offline and fused entries.

        Checks the decoder protocol match, the idempotent last-replay branch
        (returns the zero-change update), delivery continuity, full/delta
        preconditions and envelope tick monotonicity.  Returns None when the
        delivery may be applied.
        """
        if parsed.protocol_version != self.PROTOCOL_VERSION:
            raise PipelineLedgerError(
                "internal parsed envelope protocol does not match this ledger")
        if parsed.delivery_sequence == self.delivery_sequence and self.delivery_sequence >= 0:
            if parsed.epoch != self.epoch or parsed.canonical_bytes != self._last_content:
                raise PipelineLedgerError(
                    "last replay token carries typed-different canonical content")
            return self._replay_update(parsed)
        if parsed.delivery_sequence != self.delivery_sequence + 1:
            raise PipelineLedgerError("delivery_sequence gap or stale replay")
        if parsed.snapshot_kind == "full":
            if self.initialized or parsed.base_watermark != 0 or parsed.base_version != 0:
                raise PipelineLedgerError("full snapshot is only valid for initial state")
        else:
            if not self.initialized:
                raise PipelineLedgerError("delta received before initial full snapshot")
            if parsed.epoch != self.epoch:
                raise PipelineLedgerError("v3 epoch changed during run")
            if (parsed.base_watermark != self.watermark
                    or parsed.base_version != self.state_version):
                raise PipelineLedgerError("delta base watermark/state version mismatch")
        if parsed.tick < self.last_tick:
            raise PipelineLedgerError("v3 tick regressed")
        return None

    def _stage_delivery(self, parsed: _ParsedEnvelope, *,
                        work_transaction,
                        node_source) -> PreparedPipelineUpdate:
        """Shared COW application core for the delivery entries.

        ``node_source`` supplies the staged node rows in logical order
        (the pre-materialized offline event tuples); the retired pipeline
        descriptor rows no longer interleave with them.
        """
        while node_source.peek_seq() is not None:
            node_source.take(work_transaction)
        # Rows/resources are detached before they become persistent state.
        for resource in parsed.resources:
            if resource["as_of_tick"] != parsed.tick:
                raise PipelineLedgerError(
                    "active resource frame must match envelope tick")
        frozen_resources = tuple(_freeze_dto(row) for row in parsed.resources)
        visible_change = bool(
            parsed.event_count
            or not _typed_equal(self._active_resources, frozen_resources))
        expected_version = parsed.base_version + (1 if visible_change else 0)
        if parsed.result_version != expected_version:
            raise PipelineLedgerError(
                "result_state_version does not match visible state change")
        return PreparedPipelineUpdate(
            parsed=parsed,
            resources=frozen_resources,
            changed=visible_change,
        )

    def commit_prepared(self, update: PreparedPipelineUpdate) -> PipelineLedgerReceipt:
        """Publish a previously fully validated update; contains no parsing."""
        parsed = update.parsed
        if parsed.protocol_version != self.PROTOCOL_VERSION:
            raise PipelineLedgerError(
                "prepared pipeline update protocol does not match this ledger")
        if update.replayed:
            self.replay_count += 1
            return PipelineLedgerReceipt(parsed.snapshot_kind,
                parsed.delivery_sequence, parsed.event_count, parsed.base_watermark,
                self.watermark, parsed.base_version, self.state_version, True)

        self._active_resources = update.resources
        self.epoch = parsed.epoch
        self.initialized = True
        self.delivery_sequence = parsed.delivery_sequence
        self.watermark = parsed.seq_end
        self.state_version = parsed.result_version
        self.last_tick = parsed.tick
        self.last_replay_token = parsed.replay_token
        self._last_content = parsed.canonical_bytes
        self._snapshot_state_cache = None
        self._pricing_snapshot_cache = None
        return PipelineLedgerReceipt(parsed.snapshot_kind,
            parsed.delivery_sequence, parsed.event_count, parsed.base_watermark,
            self.watermark, parsed.base_version, self.state_version, False)

    def apply_envelope(self, envelope: Any) -> PipelineLedgerReceipt:
        """Standalone validate-and-commit helper for parser/unit tests."""
        return self.commit_prepared(self.prepare_envelope(envelope))

    def snapshot_state(self) -> Mapping[str, Any]:
        """Materialize a complete offline view on explicit request only."""
        if self._snapshot_state_cache is not None:
            return self._snapshot_state_cache
        self._snapshot_state_cache = MappingProxyType({
            "epoch": self.epoch,
            "delivery_sequence": self.delivery_sequence,
            "event_watermark": self.watermark,
            "state_version": self.state_version,
            "tick": self.last_tick,
            "active_resources": self._active_resources,
        })
        return self._snapshot_state_cache

    def pricing_snapshot(self) -> Mapping[str, Any]:
        """Cheap hot-path view of current prices and version metadata only."""
        if self._pricing_snapshot_cache is None:
            self._pricing_snapshot_cache = MappingProxyType({
                "epoch": self.epoch,
                "delivery_sequence": self.delivery_sequence,
                "event_watermark": self.watermark,
                "state_version": self.state_version,
                "tick": self.last_tick,
                "active_resources": self._active_resources,
            })
        return self._pricing_snapshot_cache

    def remaining_view(self) -> Mapping[str, Any]:
        """Read-only logic view of the accepted node/resource state."""
        return self.snapshot_state()


class PipelineLedgerV4(PipelineLedger):
    """Strict live v4 consumer.

    ``PipelineLedger`` remains the exact v3 full-row decoder used by offline
    archive comparison. OnlineSchedulerBase constructs only this class, so a
    v3 envelope cannot fall back through the live path.
    """

    PROTOCOL_VERSION = 4
