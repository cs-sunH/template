"""Frozen metrics schema and pure-Python contract fixtures.

This module pins down the observation-only data protocol used by the metrics
collection work: the ``metrics_manifest.json`` sidecar format, the node event
integer encoding, the planner memory ledger observed through
:class:`MemoryMetricsObserver`, and the cross-run normalization fields.

Nothing in this module feeds decisions back into the scheduler, the KV
managers, or the ET generators; it only describes and validates records that
those components may emit.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1

# Node event integer encoding (manifest protocol).  Each event is stored as
# the triple [node_id, event_code, subject_id]; the subject is a queue_index.
# M-22 (2026-09-30): the microbenchmark iteration codes 5/6 are retired from
# the legal domain -- this repo has no producer of them anymore (the C++
# collector removed their emission together with the static
# node_events_by_rank table), so a frozen schema that still accepted them
# would bless events nothing can ever emit.
EVENT_PREFILL_START = 1
EVENT_PREFILL_END = 2
EVENT_DECODE_START = 3
EVENT_DECODE_END = 4
EVENT_MEMORY_ANCHOR_COMPLETE = 7
EVENT_FIRST_TOKEN_COMPLETE = 8

EVENT_EDGE_ISSUE = "issue"
EVENT_EDGE_COMPLETE = "complete"

EVENT_EDGE_BY_CODE = {
    EVENT_PREFILL_START: EVENT_EDGE_ISSUE,
    EVENT_PREFILL_END: EVENT_EDGE_COMPLETE,
    EVENT_DECODE_START: EVENT_EDGE_ISSUE,
    EVENT_DECODE_END: EVENT_EDGE_COMPLETE,
    EVENT_MEMORY_ANCHOR_COMPLETE: EVENT_EDGE_COMPLETE,
    EVENT_FIRST_TOKEN_COMPLETE: EVENT_EDGE_COMPLETE,
}

SERVICE_EVENT_CODES = frozenset(
    {
        EVENT_PREFILL_START,
        EVENT_PREFILL_END,
        EVENT_DECODE_START,
        EVENT_DECODE_END,
        EVENT_FIRST_TOKEN_COMPLETE,
    }
)

ARRIVAL_ABSOLUTE = "absolute"
ARRIVAL_AFTER_REQUEST = "after_request"

# request_metrics.csv 闭集枚举(G7-2 修复 2026-09-27:测试契约改为断言
# 生产导出集合,写入点同步消费常量——此前 terminal_status/first_token_
# source 取值仅以字面量散在 metrics_postprocess 写入点,测试类用局部
# 常量对硬编码元组自验,生产改枚举仍 100% 通过的假绿灯)。
NA = "NA"
TERMINAL_STATUS_COMPLETED = "completed"
TERMINAL_STATUS_FAILED = "failed"
TERMINAL_STATUS_VALUES = (TERMINAL_STATUS_COMPLETED, TERMINAL_STATUS_FAILED)
# WP9 首 token 来源标记(first_token_source 列;主规格 §3.1:A 类枚举
# exact|train_interpolated|NA。exact = 事件码 8;NA = 无事件且归档
# train ledger 不可得;train_interpolated = B4 fallback proxy,
# display-only,SLO 判定路径禁入)。
FIRST_TOKEN_SOURCE_EXACT = "exact"
FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED = "train_interpolated"
FIRST_TOKEN_SOURCE_VALUES = (
    FIRST_TOKEN_SOURCE_EXACT,
    FIRST_TOKEN_SOURCE_TRAIN_INTERPOLATED,
    NA,
)

RUN_MODE_SERVICE = "service"
RUN_MODE_MICROBENCHMARK = "microbenchmark"
RUN_MODES = frozenset({RUN_MODE_SERVICE, RUN_MODE_MICROBENCHMARK})

NORMALIZATION_GROUP_MAX = "group_max"
NORMALIZATION_RATIO_TO_BASELINE = "ratio_to_baseline"
NORMALIZATION_METHODS = frozenset(
    {NORMALIZATION_GROUP_MAX, NORMALIZATION_RATIO_TO_BASELINE}
)

NATIVE_MEMORY_SCOPE = "npu_local_hbm"
CHIPLET_PROJECTION_METHOD = "measurement_only_equal_striping"
DEFAULT_CHIPLETS_PER_NPU = 4


class MetricsSchemaError(ValueError):
    """Raised whenever a metrics record violates the frozen schema."""


def _require_int(value: Any, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetricsSchemaError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise MetricsSchemaError(f"{name} must be >= {minimum}")
    return value


def _require_optional_int(value: Any, name: str, minimum: int = 0) -> Optional[int]:
    if value is None:
        return None
    return _require_int(value, name, minimum)


def _require_str(value: Any, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise MetricsSchemaError(f"{name} must be a non-empty string")
    return value


def _require_optional_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, name)


def _rank_map_key(value: Any) -> int:
    """Rank map key for the frozen-schema readers (POST-05, 2026-09-26).

    JSON object keys arrive as strings; a non-numeric key used to escape
    ``int(...)`` as a bare ValueError instead of the structured
    MetricsSchemaError every other reader violation raises.
    """

    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise MetricsSchemaError(f"rank key must be an integer, got {value!r}")
    try:
        return int(value)
    except ValueError:
        raise MetricsSchemaError(
            f"rank key must be an integer, got {value!r}"
        ) from None


def check_schema_version(version: Any) -> int:
    """Reject manifests produced by a different or malformed schema version."""

    _require_int(version, "schema_version")
    if version != SCHEMA_VERSION:
        raise MetricsSchemaError(
            f"unsupported schema_version {version}; this reader only supports {SCHEMA_VERSION}"
        )
    return version


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricsSchemaError(f"{name} must be an object")
    return value


@dataclass(frozen=True)
class Arrival:
    """Request arrival description: absolute time or parent-relative time."""

    kind: str
    value_ns: Optional[int] = None
    parent_queue_index: Optional[int] = None
    interval_ns: Optional[int] = None

    def __post_init__(self) -> None:
        if self.kind == ARRIVAL_ABSOLUTE:
            _require_int(self.value_ns, "arrival.value_ns", 0)
            if self.parent_queue_index is not None or self.interval_ns is not None:
                raise MetricsSchemaError(
                    "absolute arrival must not carry parent_queue_index/interval_ns"
                )
        elif self.kind == ARRIVAL_AFTER_REQUEST:
            _require_int(self.parent_queue_index, "arrival.parent_queue_index", 0)
            _require_int(self.interval_ns, "arrival.interval_ns", 0)
            if self.value_ns is not None:
                raise MetricsSchemaError("after_request arrival must not carry value_ns")
        else:
            raise MetricsSchemaError(f"unknown arrival kind {self.kind!r}")

    @classmethod
    def absolute(cls, value_ns: int) -> "Arrival":
        return cls(kind=ARRIVAL_ABSOLUTE, value_ns=value_ns)

    @classmethod
    def after_request(cls, parent_queue_index: int, interval_ns: int) -> "Arrival":
        return cls(
            kind=ARRIVAL_AFTER_REQUEST,
            parent_queue_index=parent_queue_index,
            interval_ns=interval_ns,
        )

    def resolve(self, parent_completion_ns: Optional[int] = None) -> int:
        """Resolve to an absolute arrival time in nanoseconds.

        A parent-relative arrival needs the parent's *actual* completion tick;
        a planner-predicted completion must never be substituted here.
        """

        if self.kind == ARRIVAL_ABSOLUTE:
            return self.value_ns
        if parent_completion_ns is None:
            raise MetricsSchemaError(
                "after_request arrival cannot resolve before the parent completion is known"
            )
        _require_int(parent_completion_ns, "parent_completion_ns", 0)
        return parent_completion_ns + self.interval_ns

    def to_dict(self) -> dict[str, Any]:
        if self.kind == ARRIVAL_ABSOLUTE:
            return {"kind": ARRIVAL_ABSOLUTE, "value_ns": self.value_ns}
        return {
            "kind": ARRIVAL_AFTER_REQUEST,
            "parent_queue_index": self.parent_queue_index,
            "interval_ns": self.interval_ns,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "Arrival":
        data = _require_mapping(data, "arrival")
        kind = data.get("kind")
        if kind == ARRIVAL_ABSOLUTE:
            unknown = set(data) - {"kind", "value_ns"}
            if unknown:
                raise MetricsSchemaError(f"absolute arrival has unknown fields {sorted(unknown)}")
            return cls.absolute(data.get("value_ns"))
        if kind == ARRIVAL_AFTER_REQUEST:
            unknown = set(data) - {"kind", "parent_queue_index", "interval_ns"}
            if unknown:
                raise MetricsSchemaError(
                    f"after_request arrival has unknown fields {sorted(unknown)}"
                )
            return cls.after_request(data.get("parent_queue_index"), data.get("interval_ns"))
        raise MetricsSchemaError(f"unknown arrival kind {kind!r}")


def _validate_rank_set(ranks: Any, name: str) -> tuple[int, ...]:
    if isinstance(ranks, (str, bytes)) or not isinstance(ranks, Sequence):
        raise MetricsSchemaError(f"{name} must be a list of rank ids")
    result = tuple(_require_int(rank, f"{name}[]", 0) for rank in ranks)
    if not result:
        raise MetricsSchemaError(f"{name} must not be empty")
    if len(set(result)) != len(result):
        raise MetricsSchemaError(f"{name} must not contain duplicates")
    return result


@dataclass(frozen=True)
class RequestMetadata:
    """One request record of the manifest ``requests`` array.

    M-22 (2026-09-30) alignment with the current protocol (plan_materializer
    writer + C++ MetricCollector reader): ``session_id`` adopts the C++
    default (empty string, also accepted explicitly) instead of rejecting it,
    and the optional ``decode_length`` passthrough (consumed by the C++
    collector for the informational decode_length==1 first-token skew) is
    part of the frozen field set.
    """

    queue_index: int
    request_id: str
    session_id: str
    turn_index: int
    arrival: Arrival
    prefill_instance: int
    prefill_ranks: tuple[int, ...]
    decode_instance: int
    decode_ranks: tuple[int, ...]
    decode_length: Optional[int] = None

    def __post_init__(self) -> None:
        _require_int(self.queue_index, "queue_index", 0)
        _require_str(self.request_id, "request_id")
        # The empty string is the C++ collector's own absent-session default;
        # rejecting it here made the frozen contract unable to parse manifests
        # the C++ reader accepts (M-22 default-value conflict).
        _require_str(self.session_id, "session_id", allow_empty=True)
        _require_int(self.turn_index, "turn_index", 0)
        if not isinstance(self.arrival, Arrival):
            raise MetricsSchemaError("arrival must be an Arrival")
        _require_int(self.prefill_instance, "prefill_instance", 0)
        object.__setattr__(
            self, "prefill_ranks", _validate_rank_set(self.prefill_ranks, "prefill_ranks")
        )
        _require_int(self.decode_instance, "decode_instance", 0)
        object.__setattr__(
            self, "decode_ranks", _validate_rank_set(self.decode_ranks, "decode_ranks")
        )
        _require_optional_int(self.decode_length, "decode_length", 0)

    def to_dict(self) -> dict[str, Any]:
        data = {
            "queue_index": self.queue_index,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "turn_index": self.turn_index,
            "arrival": self.arrival.to_dict(),
            "prefill_instance": self.prefill_instance,
            "prefill_ranks": list(self.prefill_ranks),
            "decode_instance": self.decode_instance,
            "decode_ranks": list(self.decode_ranks),
        }
        if self.decode_length is not None:
            data["decode_length"] = self.decode_length
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestMetadata":
        data = _require_mapping(data, "request")
        expected = {
            "queue_index",
            "request_id",
            "session_id",
            "turn_index",
            "arrival",
            "prefill_instance",
            "prefill_ranks",
            "decode_instance",
            "decode_ranks",
            "decode_length",
        }
        unknown = set(data) - expected
        if unknown:
            raise MetricsSchemaError(f"request has unknown fields {sorted(unknown)}")
        # session_id (C++ default "") and decode_length (C++ optional) are
        # the only optional request fields (M-22 protocol alignment).
        missing = expected - set(data) - {"decode_length", "session_id"}
        if missing:
            raise MetricsSchemaError(f"request is missing fields {sorted(missing)}")
        return cls(
            queue_index=data["queue_index"],
            request_id=data["request_id"],
            session_id=data.get("session_id", ""),
            turn_index=data["turn_index"],
            arrival=Arrival.from_dict(data["arrival"]),
            prefill_instance=data["prefill_instance"],
            prefill_ranks=tuple(data["prefill_ranks"]),
            decode_instance=data["decode_instance"],
            decode_ranks=tuple(data["decode_ranks"]),
            decode_length=data.get("decode_length"),
        )


@dataclass(frozen=True)
class NodeMetricEvent:
    """One ``[node_id, event_code, subject_id]`` manifest event triple."""

    node_id: int
    event_code: int
    subject_id: int

    def __post_init__(self) -> None:
        _require_int(self.node_id, "node_id", 0)
        _require_int(self.event_code, "event_code")
        if self.event_code not in EVENT_EDGE_BY_CODE:
            raise MetricsSchemaError(
                f"event_code {self.event_code} is not in the protocol "
                f"encoding {sorted(EVENT_EDGE_BY_CODE)} (microbenchmark "
                "iteration codes 5/6 are retired)"
            )
        _require_int(self.subject_id, "subject_id", 0)

    @property
    def edge(self) -> str:
        """The protocol-fixed trigger edge of this event code."""

        return EVENT_EDGE_BY_CODE[self.event_code]

    def to_triple(self) -> list[int]:
        return [self.node_id, self.event_code, self.subject_id]

    @classmethod
    def from_triple(cls, triple: Any) -> "NodeMetricEvent":
        if (
            isinstance(triple, (str, bytes))
            or not isinstance(triple, Sequence)
            or len(triple) != 3
        ):
            raise MetricsSchemaError("node event must be a [node_id, event_code, subject_id] triple")
        return cls(node_id=triple[0], event_code=triple[1], subject_id=triple[2])


@dataclass(frozen=True)
class MemoryProjection:
    """The measurement-only chiplet projection declaration of the manifest.

    Positioning note (dead-code audit, 2026-09-30): the manifest
    ``memory_projection`` block currently has **no producer and no
    consumer** in this repository (plan_materializer does not write it,
    neither C++ MetricCollector nor metrics_postprocess reads it; the only
    touchpoints are this schema's own round-trip tests).  Kept as part of
    the frozen cross-repo manifest schema -- deleting it is a cross-repo
    ruling, and the joint sibling carries the same dormant field."""

    native_scope: str = NATIVE_MEMORY_SCOPE
    chiplets_per_npu: int = DEFAULT_CHIPLETS_PER_NPU
    projection: str = CHIPLET_PROJECTION_METHOD

    def __post_init__(self) -> None:
        _require_str(self.native_scope, "memory_projection.native_scope")
        _require_int(self.chiplets_per_npu, "memory_projection.chiplets_per_npu", 1)
        _require_str(self.projection, "memory_projection.projection")
        # Frozen-constant comparison (POST-05, 2026-09-26): the reader used
        # to accept any non-empty scope/projection strings, letting a
        # mistyped manifest silently split the frozen vocabulary.
        if self.native_scope != NATIVE_MEMORY_SCOPE:
            raise MetricsSchemaError(
                f"memory_projection.native_scope must be "
                f"{NATIVE_MEMORY_SCOPE!r} (frozen), got {self.native_scope!r}"
            )
        if self.projection != CHIPLET_PROJECTION_METHOD:
            raise MetricsSchemaError(
                f"memory_projection.projection must be "
                f"{CHIPLET_PROJECTION_METHOD!r} (frozen), got "
                f"{self.projection!r}"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_scope": self.native_scope,
            "chiplets_per_npu": self.chiplets_per_npu,
            "projection": self.projection,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryProjection":
        data = _require_mapping(data, "memory_projection")
        # Unknown-field rejection (E9, 2026-09-30): this used to be the only
        # reader in the schema that silently ignored unknown sub-fields;
        # every other reader (snapshot / MemoryDelta / MetricsManifest)
        # fails closed on typos -- same discipline here now.
        allowed = {"native_scope", "chiplets_per_npu", "projection"}
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise MetricsSchemaError(
                f"memory_projection has unknown fields: {unknown} "
                f"(allowed: {sorted(allowed)})"
            )
        return cls(
            native_scope=data.get("native_scope", NATIVE_MEMORY_SCOPE),
            chiplets_per_npu=data.get("chiplets_per_npu", DEFAULT_CHIPLETS_PER_NPU),
            projection=data.get("projection", CHIPLET_PROJECTION_METHOD),
        )


@dataclass(frozen=True)
class MetricsManifest:
    """Validated in-memory form of ``metrics_manifest.json``.

    M-22 (2026-09-30) alignment with the current manifest protocol.  The
    producer side is ``plan_materializer.py`` (schema_version, repo_variant,
    manifest_source, requests[], slo_sampling) and the consumer side is the
    C++ MetricCollector ``load_manifest``, which treats every other top-level
    key as optional with its own default (run_mode "service", absent
    digests/npus_count/mesh, and a static ``node_events_by_rank`` table it no
    longer parses).  The frozen schema previously *required*
    trace_digest/request_mapping_digest/kv_event_digest, npus_count and the
    mesh block, so it rejected every manifest the current writer produces
    (100% false rejection once wired to a real sidecar).  Those fields are
    now optional with C++-aligned defaults; unknown top-level keys stay
    tolerated exactly like the C++ reader (additive protocol evolution never
    invalidates stored manifests), while every *known* key is type-checked.

    Positioning note (dead-code audit): this contract family still has no
    in-repo Python consumer of manifests (production writes and consumes the
    sidecar in C++/plan_materializer only); it remains the frozen, test-
    exercised specification of the protocol.
    """

    repo_variant: str
    requests: tuple[RequestMetadata, ...]
    node_events_by_rank: Mapping[int, tuple[NodeMetricEvent, ...]] = field(
        default_factory=dict
    )
    run_mode: str = RUN_MODE_SERVICE
    npus_count: Optional[int] = None
    mesh_rows: Optional[int] = None
    mesh_columns: Optional[int] = None
    trace_digest: Optional[str] = None
    request_mapping_digest: Optional[str] = None
    kv_event_digest: Optional[str] = None
    run_id: Optional[str] = None
    manifest_source: Optional[str] = None
    slo_sampling: Optional[Mapping[str, Any]] = None
    memory_projection: MemoryProjection = field(default_factory=MemoryProjection)
    memory_actions: tuple[Any, ...] = ()
    planner_memory_peaks: tuple[Any, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version)
        _require_str(self.repo_variant, "repo_variant")
        if self.run_mode not in RUN_MODES:
            raise MetricsSchemaError(f"run_mode must be one of {sorted(RUN_MODES)}")
        _require_optional_int(self.npus_count, "npus_count", 1)
        _require_optional_int(self.mesh_rows, "mesh.rows", 1)
        _require_optional_int(self.mesh_columns, "mesh.columns", 1)
        _require_optional_str(self.trace_digest, "trace_digest")
        _require_optional_str(self.request_mapping_digest, "request_mapping_digest")
        _require_optional_str(self.kv_event_digest, "kv_event_digest")
        _require_optional_str(self.run_id, "run_id")
        _require_optional_str(self.manifest_source, "manifest_source")
        if self.slo_sampling is not None and not isinstance(self.slo_sampling, Mapping):
            raise MetricsSchemaError("slo_sampling must be an object")
        if not isinstance(self.memory_projection, MemoryProjection):
            raise MetricsSchemaError("memory_projection must be a MemoryProjection")
        _validate_request_collection(self.requests)
        validated_events: dict[int, tuple[NodeMetricEvent, ...]] = {}
        for rank, events in self.node_events_by_rank.items():
            _require_int(rank, "node_events_by_rank key", 0)
            if self.npus_count is not None and rank >= self.npus_count:
                raise MetricsSchemaError(
                    f"node events listed for rank {rank} beyond npus_count {self.npus_count}"
                )
            for event in events:
                if not isinstance(event, NodeMetricEvent):
                    raise MetricsSchemaError("node events must be NodeMetricEvent instances")
            validated_events[rank] = tuple(events)
        object.__setattr__(self, "node_events_by_rank", validated_events)
        object.__setattr__(self, "requests", tuple(self.requests))

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "schema_version": self.schema_version,
            "repo_variant": self.repo_variant,
            "run_mode": self.run_mode,
            "requests": [request.to_dict() for request in self.requests],
            "node_events_by_rank": {
                str(rank): [event.to_triple() for event in events]
                for rank, events in sorted(self.node_events_by_rank.items())
            },
            "memory_projection": self.memory_projection.to_dict(),
            "memory_actions": list(self.memory_actions),
            "planner_memory_peaks": list(self.planner_memory_peaks),
        }
        if self.npus_count is not None:
            data["npus_count"] = self.npus_count
        if self.mesh_rows is not None or self.mesh_columns is not None:
            data["mesh"] = {"rows": self.mesh_rows, "columns": self.mesh_columns}
        for name in (
            "trace_digest",
            "request_mapping_digest",
            "kv_event_digest",
            "run_id",
            "manifest_source",
        ):
            value = getattr(self, name)
            if value is not None:
                data[name] = value
        if self.slo_sampling is not None:
            data["slo_sampling"] = dict(self.slo_sampling)
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricsManifest":
        data = _require_mapping(data, "manifest")
        check_schema_version(data.get("schema_version"))
        mesh: Any = data.get("mesh")
        mesh_rows: Optional[int] = None
        mesh_columns: Optional[int] = None
        if mesh is not None:
            mesh = _require_mapping(mesh, "mesh")
            mesh_rows = mesh.get("rows")
            mesh_columns = mesh.get("columns")
        raw_events = _require_mapping(data.get("node_events_by_rank", {}), "node_events_by_rank")
        events_by_rank: dict[int, tuple[NodeMetricEvent, ...]] = {}
        for rank_key, triples in raw_events.items():
            try:
                rank = int(rank_key)
            except (TypeError, ValueError):
                raise MetricsSchemaError(
                    f"node_events_by_rank key {rank_key!r} is not an integer rank"
                ) from None
            if isinstance(triples, (str, bytes)) or not isinstance(triples, Sequence):
                raise MetricsSchemaError("node_events_by_rank values must be lists of triples")
            events_by_rank[rank] = tuple(NodeMetricEvent.from_triple(t) for t in triples)
        raw_requests = data.get("requests", [])
        if isinstance(raw_requests, (str, bytes)) or not isinstance(raw_requests, Sequence):
            raise MetricsSchemaError("requests must be a list")
        raw_slo_sampling = data.get("slo_sampling")
        if raw_slo_sampling is not None and not isinstance(raw_slo_sampling, Mapping):
            raise MetricsSchemaError("slo_sampling must be an object")
        return cls(
            repo_variant=data.get("repo_variant"),
            run_mode=data.get("run_mode", RUN_MODE_SERVICE),
            npus_count=data.get("npus_count"),
            mesh_rows=mesh_rows,
            mesh_columns=mesh_columns,
            trace_digest=data.get("trace_digest"),
            request_mapping_digest=data.get("request_mapping_digest"),
            kv_event_digest=data.get("kv_event_digest"),
            run_id=data.get("run_id"),
            manifest_source=data.get("manifest_source"),
            slo_sampling=dict(raw_slo_sampling) if raw_slo_sampling is not None else None,
            requests=tuple(RequestMetadata.from_dict(r) for r in raw_requests),
            node_events_by_rank=events_by_rank,
            memory_projection=MemoryProjection.from_dict(
                data.get("memory_projection", {})
            ),
            memory_actions=tuple(data.get("memory_actions", [])),
            planner_memory_peaks=tuple(data.get("planner_memory_peaks", [])),
        )


def _validate_request_collection(requests: Sequence[RequestMetadata]) -> None:
    """Manifest completeness rules on the request array itself."""

    for request in requests:
        if not isinstance(request, RequestMetadata):
            raise MetricsSchemaError(
                "requests entries must be RequestMetadata instances, got "
                f"{type(request).__name__}"
            )
    queue_indices = [request.queue_index for request in requests]
    if len(set(queue_indices)) != len(queue_indices):
        raise MetricsSchemaError("queue_index values must be unique")
    if sorted(queue_indices) != list(range(len(queue_indices))):
        raise MetricsSchemaError(
            "queue_index values must be contiguous starting at 0"
        )
    request_ids = [request.request_id for request in requests]
    if len(set(request_ids)) != len(request_ids):
        scoped_keys = [
            (request.session_id, request.turn_index, request.request_id)
            for request in requests
        ]
        if len(set(scoped_keys)) != len(scoped_keys):
            raise MetricsSchemaError(
                "request_id values are not unique and neither is "
                "(session_id, turn_index, request_id)"
            )


@dataclass(frozen=True)
class MemoryDelta:
    """One planner memory state change handed to the observer after the fact."""

    sequence_index: int
    planner_time_ns: int
    anchor_kind: str
    trigger_queue_index: Optional[int]
    request_id: Optional[str]
    session_id: Optional[str]
    rank: int
    allocation_key: str
    weight_delta_bytes: int = 0
    resident_kv_delta_bytes: int = 0
    reserved_kv_delta_bytes: int = 0
    cause: str = ""

    def __post_init__(self) -> None:
        _require_int(self.sequence_index, "sequence_index", 0)
        _require_int(self.planner_time_ns, "planner_time_ns", 0)
        _require_str(self.anchor_kind, "anchor_kind")
        _require_optional_int(self.trigger_queue_index, "trigger_queue_index", 0)
        _require_optional_str(self.request_id, "request_id")
        _require_optional_str(self.session_id, "session_id")
        _require_int(self.rank, "rank", 0)
        _require_str(self.allocation_key, "allocation_key")
        for name in (
            "weight_delta_bytes",
            "resident_kv_delta_bytes",
            "reserved_kv_delta_bytes",
        ):
            _require_int(getattr(self, name), name)
        if not isinstance(self.cause, str):
            raise MetricsSchemaError("cause must be a string")

    def to_dict(self) -> dict[str, Any]:
        return {
            "sequence_index": self.sequence_index,
            "planner_time_ns": self.planner_time_ns,
            "anchor_kind": self.anchor_kind,
            "trigger_queue_index": self.trigger_queue_index,
            "request_id": self.request_id,
            "session_id": self.session_id,
            "rank": self.rank,
            "allocation_key": self.allocation_key,
            "weight_delta_bytes": self.weight_delta_bytes,
            "resident_kv_delta_bytes": self.resident_kv_delta_bytes,
            "reserved_kv_delta_bytes": self.reserved_kv_delta_bytes,
            "cause": self.cause,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryDelta":
        data = _require_mapping(data, "memory delta")
        expected = {
            "sequence_index",
            "planner_time_ns",
            "anchor_kind",
            "trigger_queue_index",
            "request_id",
            "session_id",
            "rank",
            "allocation_key",
            "weight_delta_bytes",
            "resident_kv_delta_bytes",
            "reserved_kv_delta_bytes",
            "cause",
        }
        unknown = set(data) - expected
        if unknown:
            raise MetricsSchemaError(f"memory delta has unknown fields {sorted(unknown)}")
        missing = expected - set(data)
        if missing:
            raise MetricsSchemaError(f"memory delta is missing fields {sorted(missing)}")
        return cls(**{name: data[name] for name in expected})


def stripe_bytes_evenly(total_bytes: int, partitions: int) -> tuple[int, ...]:
    """Deterministic equal striping of one allocation across chiplets.

    Negative values stripe their magnitude and carry the sign, so adding and
    later removing the same byte count walks the distribution back exactly.
    """

    _require_int(total_bytes, "total_bytes")
    _require_int(partitions, "partitions", 1)
    sign = 1 if total_bytes >= 0 else -1
    base, remainder = divmod(abs(total_bytes), partitions)
    return tuple(
        sign * (base + (1 if index < remainder else 0)) for index in range(partitions)
    )


@dataclass(frozen=True)
class MemoryLedgerSnapshot:
    """One authoritative ledger state (rank scope or projected chiplet scope)."""

    capacity_bytes: int
    weight_bytes: int
    resident_kv_bytes: int
    reserved_kv_bytes: int
    physical_used_bytes: int
    committed_used_bytes: int
    uncommitted_free_bytes: int

    def __post_init__(self) -> None:
        # Reader-side value validation (M27): from_dict used to check the
        # key set only.  M-21 (2026-09-30) unified the component sign
        # contract: a negative weight/resident/reserved component is ledger
        # corruption everywhere (observer, journal writer, C++ replay,
        # watermark), so the reader rejects it too.  Only capacity_bytes
        # carries a universal non-negativity bound of its own;
        # uncommitted_free_bytes stays sign-free because the measurement-only
        # equal-striping chiplet projection may legitimately overload one
        # chiplet (free < 0) while every component stays non-negative.
        for name in (
            "capacity_bytes",
            "weight_bytes",
            "resident_kv_bytes",
            "reserved_kv_bytes",
            "physical_used_bytes",
            "committed_used_bytes",
        ):
            value = _require_int(getattr(self, name), f"ledger snapshot.{name}")
            if value < 0:
                raise MetricsSchemaError(
                    f"ledger snapshot.{name} must be >= 0 (a negative "
                    "component is ledger corruption; M-21 fail-closed), got "
                    f"{value}"
                )
        _require_int(self.capacity_bytes, "ledger snapshot.capacity_bytes", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capacity_bytes": self.capacity_bytes,
            "weight_bytes": self.weight_bytes,
            "resident_kv_bytes": self.resident_kv_bytes,
            "reserved_kv_bytes": self.reserved_kv_bytes,
            "physical_used_bytes": self.physical_used_bytes,
            "committed_used_bytes": self.committed_used_bytes,
            "uncommitted_free_bytes": self.uncommitted_free_bytes,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryLedgerSnapshot":
        data = _require_mapping(data, "ledger snapshot")
        expected = {
            "capacity_bytes",
            "weight_bytes",
            "resident_kv_bytes",
            "reserved_kv_bytes",
            "physical_used_bytes",
            "committed_used_bytes",
            "uncommitted_free_bytes",
        }
        unknown = set(data) - expected
        if unknown:
            raise MetricsSchemaError(
                f"ledger snapshot has unknown fields {sorted(unknown)}"
            )
        missing = expected - set(data)
        if missing:
            raise MetricsSchemaError(
                f"ledger snapshot is missing fields {sorted(missing)}"
            )
        return cls(**{name: data[name] for name in expected})


@dataclass(frozen=True)
class MemoryPeakSnapshot:
    """Peak value plus the same-moment component snapshot at the peak."""

    scope: str
    rank: int
    chiplet_index: Optional[int]
    peak_kind: str
    peak_value_bytes: int
    capacity_bytes: int
    weight_at_peak_bytes: int
    resident_kv_at_peak_bytes: int
    reserved_kv_at_peak_bytes: int
    free_at_peak_bytes: int
    planner_time_ns: int
    sequence_index: int
    cause: str
    request_id: Optional[str]
    session_id: Optional[str]

    def __post_init__(self) -> None:
        # Reader-side value validation (M27): from_dict used to check the
        # key set only.  Sign bounds follow field semantics: rank,
        # peak_value_bytes, capacity_bytes, and planner_time_ns are
        # universally non-negative (the peak tracker starts at 0 and only
        # moves up); sequence_index is -1 in the init snapshot; the three
        # at-peak components (weight/resident/reserved) carry the same
        # explicit >= 0 reader gate as the MemoryLedgerSnapshot components
        # (M-21 unified sign contract: a negative component is ledger
        # corruption, rejected fail-closed here too), while
        # free_at_peak_bytes stays sign-free (type-checked only): the
        # measurement-only equal-striping chiplet projection may
        # legitimately overload one chiplet at the peak.
        _require_str(self.scope, "peak snapshot.scope")
        _require_str(self.peak_kind, "peak snapshot.peak_kind")
        # cause 镜像 MemoryDelta.cause 的宽松口径(缺省空串合法)。
        _require_str(self.cause, "peak snapshot.cause", allow_empty=True)
        _require_optional_str(self.request_id, "peak snapshot.request_id")
        _require_optional_str(self.session_id, "peak snapshot.session_id")
        _require_int(self.rank, "peak snapshot.rank", 0)
        _require_optional_int(
            self.chiplet_index, "peak snapshot.chiplet_index", 0)
        _require_int(
            self.peak_value_bytes, "peak snapshot.peak_value_bytes", 0)
        _require_int(self.capacity_bytes, "peak snapshot.capacity_bytes", 0)
        # M-21 读侧非负门(与姊妹类 MemoryLedgerSnapshot 同款显式门;
        # R5-C5-S1:此前仅类型检查,与上方注释声明的契约不一致)。
        for name in (
            "weight_at_peak_bytes",
            "resident_kv_at_peak_bytes",
            "reserved_kv_at_peak_bytes",
        ):
            value = _require_int(getattr(self, name), f"peak snapshot.{name}")
            if value < 0:
                raise MetricsSchemaError(
                    f"peak snapshot.{name} must be >= 0 (a negative "
                    "at-peak component is ledger corruption; M-21 "
                    "fail-closed), got "
                    f"{value}"
                )
        _require_int(self.free_at_peak_bytes, "peak snapshot.free_at_peak_bytes")
        _require_int(self.planner_time_ns, "peak snapshot.planner_time_ns", 0)
        _require_int(self.sequence_index, "peak snapshot.sequence_index")

    def to_dict(self) -> dict[str, Any]:
        return {
            "scope": self.scope,
            "rank": self.rank,
            "chiplet_index": self.chiplet_index,
            "peak_kind": self.peak_kind,
            "peak_value_bytes": self.peak_value_bytes,
            "capacity_bytes": self.capacity_bytes,
            "weight_at_peak_bytes": self.weight_at_peak_bytes,
            "resident_kv_at_peak_bytes": self.resident_kv_at_peak_bytes,
            "reserved_kv_at_peak_bytes": self.reserved_kv_at_peak_bytes,
            "free_at_peak_bytes": self.free_at_peak_bytes,
            "planner_time_ns": self.planner_time_ns,
            "sequence_index": self.sequence_index,
            "cause": self.cause,
            "request_id": self.request_id,
            "session_id": self.session_id,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryPeakSnapshot":
        data = _require_mapping(data, "peak snapshot")
        expected = {
            "scope",
            "rank",
            "chiplet_index",
            "peak_kind",
            "peak_value_bytes",
            "capacity_bytes",
            "weight_at_peak_bytes",
            "resident_kv_at_peak_bytes",
            "reserved_kv_at_peak_bytes",
            "free_at_peak_bytes",
            "planner_time_ns",
            "sequence_index",
            "cause",
            "request_id",
            "session_id",
        }
        unknown = set(data) - expected
        if unknown:
            raise MetricsSchemaError(
                f"peak snapshot has unknown fields {sorted(unknown)}"
            )
        missing = expected - set(data)
        if missing:
            raise MetricsSchemaError(
                f"peak snapshot is missing fields {sorted(missing)}"
            )
        return cls(**{name: data[name] for name in expected})


@dataclass(frozen=True)
class ChipletMemoryResult:
    chiplet_index: int
    ledger: MemoryLedgerSnapshot
    peak_physical: MemoryPeakSnapshot
    peak_committed: MemoryPeakSnapshot

    def __post_init__(self) -> None:
        # Reader-side value validation (M27); the nested snapshots are
        # validated by their own __post_init__ (also on the from_dict path).
        _require_int(self.chiplet_index, "chiplet result.chiplet_index", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "chiplet_index": self.chiplet_index,
            "ledger": self.ledger.to_dict(),
            "peak_physical": self.peak_physical.to_dict(),
            "peak_committed": self.peak_committed.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ChipletMemoryResult":
        data = _require_mapping(data, "chiplet result")
        return cls(
            chiplet_index=data.get("chiplet_index"),
            ledger=MemoryLedgerSnapshot.from_dict(data.get("ledger")),
            peak_physical=MemoryPeakSnapshot.from_dict(data.get("peak_physical")),
            peak_committed=MemoryPeakSnapshot.from_dict(data.get("peak_committed")),
        )


@dataclass(frozen=True)
class RankMemoryResult:
    rank: int
    ledger: MemoryLedgerSnapshot
    peak_physical: MemoryPeakSnapshot
    peak_committed: MemoryPeakSnapshot
    chiplets: tuple[ChipletMemoryResult, ...]

    def __post_init__(self) -> None:
        # Reader-side value validation (M27); the nested snapshots are
        # validated by their own __post_init__ (also on the from_dict path).
        _require_int(self.rank, "rank result.rank", 0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "rank": self.rank,
            "ledger": self.ledger.to_dict(),
            "peak_physical": self.peak_physical.to_dict(),
            "peak_committed": self.peak_committed.to_dict(),
            "chiplets": [chiplet.to_dict() for chiplet in self.chiplets],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RankMemoryResult":
        data = _require_mapping(data, "rank result")
        return cls(
            rank=data.get("rank"),
            ledger=MemoryLedgerSnapshot.from_dict(data.get("ledger")),
            peak_physical=MemoryPeakSnapshot.from_dict(data.get("peak_physical")),
            peak_committed=MemoryPeakSnapshot.from_dict(data.get("peak_committed")),
            chiplets=tuple(
                ChipletMemoryResult.from_dict(c) for c in data.get("chiplets", [])
            ),
        )


@dataclass(frozen=True)
class MemoryMetricsResult:
    """Final observer output: authoritative rank ledgers plus the
    measurement-only equal-striping chiplet projection."""

    chiplets_per_npu: int
    ranks: Mapping[int, RankMemoryResult]
    native_scope: str = NATIVE_MEMORY_SCOPE
    projection: str = CHIPLET_PROJECTION_METHOD

    def __post_init__(self) -> None:
        # Reader-side validation gaps closed (POST-05, 2026-09-26): a
        # missing chiplets_per_npu used to slip through as None and
        # scope/projection strings were accepted unchecked on this
        # so-called frozen schema.
        _require_int(self.chiplets_per_npu,
                     "memory_metrics.chiplets_per_npu", 1)
        if self.native_scope != NATIVE_MEMORY_SCOPE:
            raise MetricsSchemaError(
                f"memory_metrics.native_scope must be "
                f"{NATIVE_MEMORY_SCOPE!r} (frozen), got "
                f"{self.native_scope!r}"
            )
        if self.projection != CHIPLET_PROJECTION_METHOD:
            raise MetricsSchemaError(
                f"memory_metrics.projection must be "
                f"{CHIPLET_PROJECTION_METHOD!r} (frozen), got "
                f"{self.projection!r}"
            )
        # ranks value/key validation: keys must be integer ranks and values
        # RankMemoryResult instances (a foreign value used to surface later
        # as a bare AttributeError at first attribute access).
        if not isinstance(self.ranks, Mapping):
            raise MetricsSchemaError("memory_metrics.ranks must be a mapping")
        for rank, result in self.ranks.items():
            _require_int(rank, "memory_metrics.ranks key", 0)
            if not isinstance(result, RankMemoryResult):
                raise MetricsSchemaError(
                    "memory_metrics.ranks values must be RankMemoryResult "
                    f"instances, got {type(result).__name__}"
                )

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_scope": self.native_scope,
            "projection": self.projection,
            "chiplets_per_npu": self.chiplets_per_npu,
            "ranks": {
                str(rank): result.to_dict()
                for rank, result in sorted(self.ranks.items())
            },
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryMetricsResult":
        data = _require_mapping(data, "memory metrics result")
        raw_ranks = _require_mapping(data.get("ranks"), "ranks")
        return cls(
            chiplets_per_npu=_require_int(
                data.get("chiplets_per_npu"),
                "memory_metrics.chiplets_per_npu", 1),
            native_scope=data.get("native_scope", NATIVE_MEMORY_SCOPE),
            projection=data.get("projection", CHIPLET_PROJECTION_METHOD),
            ranks={
                _rank_map_key(rank): RankMemoryResult.from_dict(result)
                for rank, result in raw_ranks.items()
            },
        )


@dataclass
class _ComponentState:
    """Per-component byte counters for one scope (rank or one chiplet)."""

    capacity_bytes: int
    weight_bytes: int = 0
    resident_kv_bytes: int = 0
    reserved_kv_bytes: int = 0

    @property
    def physical(self) -> int:
        return self.weight_bytes + self.resident_kv_bytes

    @property
    def committed(self) -> int:
        return self.physical + self.reserved_kv_bytes

    def ledger(self) -> MemoryLedgerSnapshot:
        return MemoryLedgerSnapshot(
            capacity_bytes=self.capacity_bytes,
            weight_bytes=self.weight_bytes,
            resident_kv_bytes=self.resident_kv_bytes,
            reserved_kv_bytes=self.reserved_kv_bytes,
            physical_used_bytes=self.physical,
            committed_used_bytes=self.committed,
            uncommitted_free_bytes=self.capacity_bytes - self.committed,
        )


class _PeakTracker:
    """Tracks one peak kind for one scope, keeping the same-moment snapshot."""

    def __init__(
        self,
        *,
        scope: str,
        rank: int,
        chiplet_index: Optional[int],
        peak_kind: str,
        capacity_bytes: int,
    ) -> None:
        self.scope = scope
        self.rank = rank
        self.chiplet_index = chiplet_index
        self.peak_kind = peak_kind
        self.snapshot = MemoryPeakSnapshot(
            scope=scope,
            rank=rank,
            chiplet_index=chiplet_index,
            peak_kind=peak_kind,
            peak_value_bytes=0,
            capacity_bytes=capacity_bytes,
            weight_at_peak_bytes=0,
            resident_kv_at_peak_bytes=0,
            reserved_kv_at_peak_bytes=0,
            free_at_peak_bytes=capacity_bytes,
            planner_time_ns=0,
            sequence_index=-1,
            cause="init",
            request_id=None,
            session_id=None,
        )

    def consider(self, state: _ComponentState, delta: MemoryDelta) -> None:
        value = state.physical if self.peak_kind == "physical_used" else state.committed
        if value <= self.snapshot.peak_value_bytes:
            return
        self.snapshot = MemoryPeakSnapshot(
            scope=self.scope,
            rank=self.rank,
            chiplet_index=self.chiplet_index,
            peak_kind=self.peak_kind,
            peak_value_bytes=value,
            capacity_bytes=state.capacity_bytes,
            weight_at_peak_bytes=state.weight_bytes,
            resident_kv_at_peak_bytes=state.resident_kv_bytes,
            reserved_kv_at_peak_bytes=state.reserved_kv_bytes,
            free_at_peak_bytes=state.capacity_bytes - state.committed,
            planner_time_ns=delta.planner_time_ns,
            sequence_index=delta.sequence_index,
            cause=delta.cause,
            request_id=delta.request_id,
            session_id=delta.session_id,
        )


class MemoryMetricsObserver:
    """Read-only shadow ledger for planner memory state changes.

    The observer never returns decisions to the KV manager: deltas are
    recorded only after the manager already applied the identical change.
    Each rank keeps the authoritative ledger (capacity / weight / resident KV
    / reserved KV) and a measurement-only equal-striping chiplet projection;
    per-chiplet values may violate physical bounds by remainder-striping
    design (finalize reports such violations as warnings on stderr without
    altering the reported numbers).
    """

    def __init__(self, chiplets_per_npu: int = DEFAULT_CHIPLETS_PER_NPU) -> None:
        _require_int(chiplets_per_npu, "chiplets_per_npu", 1)
        self.chiplets_per_npu = chiplets_per_npu
        self._ranks: dict[int, _ComponentState] = {}
        self._chiplets: dict[int, list[_ComponentState]] = {}
        # rank -> allocation_key -> (weight dist, resident dist, reserved dist)
        self._allocations: dict[
            int, dict[str, list[tuple[int, ...]]]
        ] = {}
        self._rank_peaks: dict[int, tuple[_PeakTracker, _PeakTracker]] = {}
        self._chiplet_peaks: dict[int, list[tuple[_PeakTracker, _PeakTracker]]] = {}
        self._finalized: Optional[MemoryMetricsResult] = None

    def initialize_rank(self, rank: int, capacity_bytes: int) -> None:
        _require_int(rank, "rank", 0)
        _require_int(capacity_bytes, "capacity_bytes", 1)
        if rank in self._ranks:
            raise MetricsSchemaError(f"rank {rank} is already initialized")
        self._finalized = None
        self._ranks[rank] = _ComponentState(capacity_bytes=capacity_bytes)
        capacities = stripe_bytes_evenly(capacity_bytes, self.chiplets_per_npu)
        self._chiplets[rank] = [
            _ComponentState(capacity_bytes=capacity) for capacity in capacities
        ]
        self._allocations[rank] = {}
        self._rank_peaks[rank] = (
            _PeakTracker(
                scope="rank",
                rank=rank,
                chiplet_index=None,
                peak_kind="physical_used",
                capacity_bytes=capacity_bytes,
            ),
            _PeakTracker(
                scope="rank",
                rank=rank,
                chiplet_index=None,
                peak_kind="committed_used",
                capacity_bytes=capacity_bytes,
            ),
        )
        self._chiplet_peaks[rank] = [
            (
                _PeakTracker(
                    scope="chiplet",
                    rank=rank,
                    chiplet_index=index,
                    peak_kind="physical_used",
                    capacity_bytes=capacity,
                ),
                _PeakTracker(
                    scope="chiplet",
                    rank=rank,
                    chiplet_index=index,
                    peak_kind="committed_used",
                    capacity_bytes=capacity,
                ),
            )
            for index, capacity in enumerate(capacities)
        ]

    def record_delta(self, delta: MemoryDelta) -> None:
        if not isinstance(delta, MemoryDelta):
            raise MetricsSchemaError("record_delta expects a MemoryDelta")
        state = self._ranks.get(delta.rank)
        if state is None:
            raise MetricsSchemaError(f"rank {delta.rank} is not initialized")
        self._finalized = None

        # Validate-then-commit (MEM-05, 2026-09-26): the rank post-state and
        # the chiplet projection are both computed and checked BEFORE any
        # state changes, so a rejected delta can never leave the observer
        # with a half-applied (rank-updated / chiplets-not) ledger.
        new_weight = state.weight_bytes + delta.weight_delta_bytes
        new_resident = state.resident_kv_bytes + delta.resident_kv_delta_bytes
        new_reserved = state.reserved_kv_bytes + delta.reserved_kv_delta_bytes
        physical = new_weight + new_resident
        committed = physical + new_reserved
        # M-21 (2026-09-30): a negative post-delta COMPONENT is ledger
        # corruption -- the same fail-closed contract the journal checksum
        # replay and hbm_watermark enforce ("负分量=账本损坏").  The
        # aggregate bound below alone used to let compensating negative
        # components (weight=-100, resident=+150) reach the journal and
        # split one bad dataset into four different endings.
        for name, value in (
            ("weight", new_weight),
            ("resident_kv", new_resident),
            ("reserved_kv", new_reserved),
        ):
            if value < 0:
                raise MetricsSchemaError(
                    f"rank {delta.rank} component {name} underflowed to "
                    f"{value} after delta {delta.sequence_index} "
                    "(negative components are ledger corruption; "
                    "fail-closed)"
                )
        if not (0 <= physical <= committed <= state.capacity_bytes):
            raise MetricsSchemaError(
                f"rank {delta.rank} ledger invariant violated after delta "
                f"{delta.sequence_index}: physical={physical} "
                f"committed={committed} capacity={state.capacity_bytes}"
            )
        projection = self._plan_chiplet_projection(delta)

        state.weight_bytes = new_weight
        state.resident_kv_bytes = new_resident
        state.reserved_kv_bytes = new_reserved
        if projection is not None:
            self._commit_chiplet_projection(delta, *projection)
        self._rank_peaks[delta.rank][0].consider(state, delta)
        self._rank_peaks[delta.rank][1].consider(state, delta)
        for index, chiplet_state in enumerate(self._chiplets[delta.rank]):
            physical_peak, committed_peak = self._chiplet_peaks[delta.rank][index]
            physical_peak.consider(chiplet_state, delta)
            committed_peak.consider(chiplet_state, delta)

    def _plan_chiplet_projection(
        self, delta: MemoryDelta
    ) -> Optional[tuple[list[tuple[int, ...]], list[tuple[int, ...]]]]:
        """Compute the delta's per-chiplet allocation record without
        mutating anything; returns ``(previous_record, new_record)``, or
        ``None`` when the delta carries no byte components.  Removing from
        an unknown allocation key raises here, still before any commit."""

        allocations = self._allocations[delta.rank]
        key = delta.allocation_key
        component_deltas = (
            delta.weight_delta_bytes,
            delta.resident_kv_delta_bytes,
            delta.reserved_kv_delta_bytes,
        )
        if all(component == 0 for component in component_deltas):
            return None
        record = allocations.get(key)
        if record is None:
            if any(component < 0 for component in component_deltas):
                raise MetricsSchemaError(
                    f"cannot remove unknown allocation key {key!r} on rank {delta.rank}"
                )
            record = [
                (0,) * self.chiplets_per_npu,
                (0,) * self.chiplets_per_npu,
                (0,) * self.chiplets_per_npu,
            ]
        new_record = []
        for index, component in enumerate(component_deltas):
            stripe = stripe_bytes_evenly(component, self.chiplets_per_npu)
            new_record.append(
                tuple(previous + change for previous, change in zip(record[index], stripe))
            )
        return record, new_record

    def _commit_chiplet_projection(
        self,
        delta: MemoryDelta,
        record: list[tuple[int, ...]],
        new_record: list[tuple[int, ...]],
    ) -> None:
        """Apply a planned projection: stripe each component across the
        chiplets and attach it to the delta's allocation key; removes
        subtract the same recorded distribution for that key."""

        allocations = self._allocations[delta.rank]
        chiplet_states = self._chiplets[delta.rank]
        for chiplet_index, chiplet_state in enumerate(chiplet_states):
            chiplet_state.weight_bytes += new_record[0][chiplet_index] - record[0][chiplet_index]
            chiplet_state.resident_kv_bytes += (
                new_record[1][chiplet_index] - record[1][chiplet_index]
            )
            chiplet_state.reserved_kv_bytes += (
                new_record[2][chiplet_index] - record[2][chiplet_index]
            )
        if all(all(value == 0 for value in dist) for dist in new_record):
            allocations.pop(delta.allocation_key, None)
        else:
            allocations[delta.allocation_key] = new_record

    def _warn_projection_invariants(
        self, rank: int, state: _ComponentState, chiplet_index: Optional[int]
    ) -> None:
        """Report chiplet-scope ledger-bound violations as stderr warnings
        (M18/M19, measurement-only contract): the equal-striping chiplet
        projection may overload one chiplet (free < 0 or physical >
        chiplet capacity) by remainder-striping design even when the rank
        aggregate fits -- record_delta enforces the full rank invariant and
        the per-component non-negativity (M-21), so these overload
        observations are the only warnings left.  They never alter the
        reported numbers."""

        free = state.capacity_bytes - state.committed
        violations: list[tuple[str, int, str]] = []
        if free < 0:
            violations.append(("uncommitted_free_bytes", free, "is negative"))
        if state.physical > state.capacity_bytes:
            violations.append(
                ("physical_used_bytes",
                 state.physical,
                 "exceeds chiplet capacity (remainder-striping design)"))
        scope = f"rank {rank} chiplet {chiplet_index}"
        for name, value, reason in violations:
            print(
                f"memory-metrics warning: {scope} {name}={value} {reason} "
                "(measurement-only projection; numbers are reported as "
                "observed)",
                file=sys.stderr,
            )

    def finalize(self) -> MemoryMetricsResult:
        # Positioning note (dead-code audit, 2026-09-30): the observer's
        # runtime value is record_delta's capacity/negative-component
        # fail-closed audit (production path); this finalize() chiplet
        # projection + peak roll-up has no in-repo production consumer
        # (tests only).  Retained as part of the frozen observer surface
        # shared with the sibling repos.
        if self._finalized is not None:
            return self._finalized
        ranks: dict[int, RankMemoryResult] = {}
        for rank, state in sorted(self._ranks.items()):
            # Rank scope needs no warning pass: record_delta enforces the
            # full rank invariant (aggregate + per-component, M-21), so no
            # rank-scope bound can be violated here.
            chiplets = []
            for index, chiplet_state in enumerate(self._chiplets[rank]):
                self._warn_projection_invariants(rank, chiplet_state, index)
                physical_peak, committed_peak = self._chiplet_peaks[rank][index]
                chiplets.append(
                    ChipletMemoryResult(
                        chiplet_index=index,
                        ledger=chiplet_state.ledger(),
                        peak_physical=physical_peak.snapshot,
                        peak_committed=committed_peak.snapshot,
                    )
                )
            rank_physical_peak, rank_committed_peak = self._rank_peaks[rank]
            ranks[rank] = RankMemoryResult(
                rank=rank,
                ledger=state.ledger(),
                peak_physical=rank_physical_peak.snapshot,
                peak_committed=rank_committed_peak.snapshot,
                chiplets=tuple(chiplets),
            )
        self._finalized = MemoryMetricsResult(
            chiplets_per_npu=self.chiplets_per_npu, ranks=ranks
        )
        return self._finalized


@dataclass(frozen=True)
class NormalizationRecord:
    """One normalized metric with its full denominator provenance."""

    metric_name: str
    raw_value: float
    normalization_method: str
    normalization_group_id: str
    baseline_run_id: Optional[str]
    denominator_value: float
    normalized_value: float

    def __post_init__(self) -> None:
        _require_str(self.metric_name, "metric_name")
        if self.normalization_method not in NORMALIZATION_METHODS:
            raise MetricsSchemaError(
                f"normalization_method must be one of {sorted(NORMALIZATION_METHODS)}"
            )
        _require_str(self.normalization_group_id, "normalization_group_id")
        _require_optional_str(self.baseline_run_id, "baseline_run_id")
        if self.normalization_method == NORMALIZATION_RATIO_TO_BASELINE and (
            self.baseline_run_id is None
        ):
            raise MetricsSchemaError(
                "ratio_to_baseline normalization requires baseline_run_id"
            )
        for name in ("raw_value", "denominator_value", "normalized_value"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MetricsSchemaError(f"{name} must be numeric")
            # POST-03 (2026-09-26): NaN compares False against every bound,
            # so the numeric check alone let NaN/Inf reach the normalized
            # products; non-finite values are rejected here.
            if not math.isfinite(value):
                raise MetricsSchemaError(f"{name} must be finite")
        if self.denominator_value <= 0:
            raise MetricsSchemaError("normalization denominator must be > 0")

    def to_dict(self) -> dict[str, Any]:
        return {
            "metric_name": self.metric_name,
            "raw_value": self.raw_value,
            "normalization_method": self.normalization_method,
            "normalization_group_id": self.normalization_group_id,
            "baseline_run_id": self.baseline_run_id,
            "denominator_value": self.denominator_value,
            "normalized_value": self.normalized_value,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "NormalizationRecord":
        data = _require_mapping(data, "normalization record")
        return cls(
            metric_name=data.get("metric_name"),
            raw_value=data.get("raw_value"),
            normalization_method=data.get("normalization_method"),
            normalization_group_id=data.get("normalization_group_id"),
            baseline_run_id=data.get("baseline_run_id"),
            denominator_value=data.get("denominator_value"),
            normalized_value=data.get("normalized_value"),
        )


def select_normalization_denominator(
    method: str,
    *,
    group_values: Optional[Iterable[float]] = None,
    baseline_run_id: Optional[str] = None,
    baseline_value: Optional[float] = None,
) -> float:
    """Pick the normalization denominator for one of the two frozen methods.

    ``group_max`` divides by the largest value inside the comparison group;
    ``ratio_to_baseline`` divides by the selected baseline run's value.  A
    zero denominator is never usable.
    """

    if method == NORMALIZATION_GROUP_MAX:
        if group_values is None:
            raise MetricsSchemaError("group_max normalization requires group_values")
        values = list(group_values)
        if not values:
            raise MetricsSchemaError("group_max normalization requires a non-empty group")
        for value in values:
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise MetricsSchemaError("group values must be numeric")
            if not math.isfinite(value):
                raise MetricsSchemaError("group values must be finite")
        denominator = max(values)
    elif method == NORMALIZATION_RATIO_TO_BASELINE:
        _require_str(baseline_run_id, "baseline_run_id")
        if baseline_value is None or isinstance(baseline_value, bool) or not isinstance(
            baseline_value, (int, float)
        ):
            raise MetricsSchemaError("ratio_to_baseline requires a numeric baseline_value")
        if not math.isfinite(baseline_value):
            raise MetricsSchemaError("baseline_value must be finite")
        denominator = baseline_value
    else:
        raise MetricsSchemaError(
            f"normalization_method must be one of {sorted(NORMALIZATION_METHODS)}"
        )
    if not math.isfinite(denominator):
        raise MetricsSchemaError("normalization denominator must be finite")
    if denominator <= 0:
        raise MetricsSchemaError("normalization denominator must be > 0")
    return float(denominator)


def build_normalization_record(
    metric_name: str,
    raw_value: float,
    method: str,
    normalization_group_id: str,
    *,
    group_values: Optional[Iterable[float]] = None,
    baseline_run_id: Optional[str] = None,
    baseline_value: Optional[float] = None,
) -> NormalizationRecord:
    """Normalize one raw metric, keeping the denominator next to the result."""

    if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
        raise MetricsSchemaError("raw_value must be numeric")
    denominator = select_normalization_denominator(
        method,
        group_values=group_values,
        baseline_run_id=baseline_run_id,
        baseline_value=baseline_value,
    )
    return NormalizationRecord(
        metric_name=metric_name,
        raw_value=float(raw_value),
        normalization_method=method,
        normalization_group_id=normalization_group_id,
        baseline_run_id=baseline_run_id,
        denominator_value=denominator,
        normalized_value=float(raw_value) / denominator,
    )
