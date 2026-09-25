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

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1

# Node event integer encoding (manifest protocol).  Each event is stored as
# the triple [node_id, event_code, subject_id]; the subject is a queue_index
# in service mode and a benchmark_point_id in microbenchmark mode.
EVENT_PREFILL_START = 1
EVENT_PREFILL_END = 2
EVENT_DECODE_START = 3
EVENT_DECODE_END = 4
EVENT_MICROBENCH_ITERATION_START = 5
EVENT_MICROBENCH_ITERATION_END = 6
EVENT_MEMORY_ANCHOR_COMPLETE = 7
EVENT_FIRST_TOKEN_COMPLETE = 8

EVENT_EDGE_ISSUE = "issue"
EVENT_EDGE_COMPLETE = "complete"

EVENT_EDGE_BY_CODE = {
    EVENT_PREFILL_START: EVENT_EDGE_ISSUE,
    EVENT_PREFILL_END: EVENT_EDGE_COMPLETE,
    EVENT_DECODE_START: EVENT_EDGE_ISSUE,
    EVENT_DECODE_END: EVENT_EDGE_COMPLETE,
    EVENT_MICROBENCH_ITERATION_START: EVENT_EDGE_ISSUE,
    EVENT_MICROBENCH_ITERATION_END: EVENT_EDGE_COMPLETE,
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
    """One request record of the manifest ``requests`` array."""

    queue_index: int
    request_id: str
    session_id: str
    turn_index: int
    arrival: Arrival
    prefill_instance: int
    prefill_ranks: tuple[int, ...]
    decode_instance: int
    decode_ranks: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_int(self.queue_index, "queue_index", 0)
        _require_str(self.request_id, "request_id")
        _require_str(self.session_id, "session_id")
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

    def to_dict(self) -> dict[str, Any]:
        return {
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
        }
        unknown = set(data) - expected
        if unknown:
            raise MetricsSchemaError(f"request has unknown fields {sorted(unknown)}")
        missing = expected - set(data)
        if missing:
            raise MetricsSchemaError(f"request is missing fields {sorted(missing)}")
        return cls(
            queue_index=data["queue_index"],
            request_id=data["request_id"],
            session_id=data["session_id"],
            turn_index=data["turn_index"],
            arrival=Arrival.from_dict(data["arrival"]),
            prefill_instance=data["prefill_instance"],
            prefill_ranks=tuple(data["prefill_ranks"]),
            decode_instance=data["decode_instance"],
            decode_ranks=tuple(data["decode_ranks"]),
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
                f"event_code {self.event_code} is not in the protocol encoding 1-8"
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
    """The measurement-only chiplet projection declaration of the manifest."""

    native_scope: str = NATIVE_MEMORY_SCOPE
    chiplets_per_npu: int = DEFAULT_CHIPLETS_PER_NPU
    projection: str = CHIPLET_PROJECTION_METHOD

    def __post_init__(self) -> None:
        _require_str(self.native_scope, "memory_projection.native_scope")
        _require_int(self.chiplets_per_npu, "memory_projection.chiplets_per_npu", 1)
        _require_str(self.projection, "memory_projection.projection")

    def to_dict(self) -> dict[str, Any]:
        return {
            "native_scope": self.native_scope,
            "chiplets_per_npu": self.chiplets_per_npu,
            "projection": self.projection,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MemoryProjection":
        data = _require_mapping(data, "memory_projection")
        return cls(
            native_scope=data.get("native_scope", NATIVE_MEMORY_SCOPE),
            chiplets_per_npu=data.get("chiplets_per_npu", DEFAULT_CHIPLETS_PER_NPU),
            projection=data.get("projection", CHIPLET_PROJECTION_METHOD),
        )


@dataclass(frozen=True)
class MetricsManifest:
    """Validated in-memory form of ``metrics_manifest.json``."""

    repo_variant: str
    run_mode: str
    npus_count: int
    mesh_rows: int
    mesh_columns: int
    trace_digest: str
    request_mapping_digest: str
    kv_event_digest: str
    requests: tuple[RequestMetadata, ...]
    node_events_by_rank: Mapping[int, tuple[NodeMetricEvent, ...]]
    memory_projection: MemoryProjection = field(default_factory=MemoryProjection)
    memory_actions: tuple[Any, ...] = ()
    planner_memory_peaks: tuple[Any, ...] = ()
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        check_schema_version(self.schema_version)
        _require_str(self.repo_variant, "repo_variant")
        if self.run_mode not in RUN_MODES:
            raise MetricsSchemaError(f"run_mode must be one of {sorted(RUN_MODES)}")
        _require_int(self.npus_count, "npus_count", 1)
        _require_int(self.mesh_rows, "mesh.rows", 1)
        _require_int(self.mesh_columns, "mesh.columns", 1)
        _require_str(self.trace_digest, "trace_digest")
        _require_str(self.request_mapping_digest, "request_mapping_digest")
        _require_str(self.kv_event_digest, "kv_event_digest")
        if not isinstance(self.memory_projection, MemoryProjection):
            raise MetricsSchemaError("memory_projection must be a MemoryProjection")
        _validate_request_collection(self.requests)
        validated_events: dict[int, tuple[NodeMetricEvent, ...]] = {}
        for rank, events in self.node_events_by_rank.items():
            _require_int(rank, "node_events_by_rank key", 0)
            if rank >= self.npus_count:
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
        return {
            "schema_version": self.schema_version,
            "repo_variant": self.repo_variant,
            "run_mode": self.run_mode,
            "trace_digest": self.trace_digest,
            "request_mapping_digest": self.request_mapping_digest,
            "kv_event_digest": self.kv_event_digest,
            "npus_count": self.npus_count,
            "mesh": {"rows": self.mesh_rows, "columns": self.mesh_columns},
            "memory_projection": self.memory_projection.to_dict(),
            "requests": [request.to_dict() for request in self.requests],
            "node_events_by_rank": {
                str(rank): [event.to_triple() for event in events]
                for rank, events in sorted(self.node_events_by_rank.items())
            },
            "memory_actions": list(self.memory_actions),
            "planner_memory_peaks": list(self.planner_memory_peaks),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "MetricsManifest":
        data = _require_mapping(data, "manifest")
        check_schema_version(data.get("schema_version"))
        for digest_field in ("trace_digest", "request_mapping_digest", "kv_event_digest"):
            _require_str(data.get(digest_field), digest_field)
        mesh = _require_mapping(data.get("mesh"), "mesh")
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
        return cls(
            repo_variant=data.get("repo_variant"),
            run_mode=data.get("run_mode"),
            npus_count=data.get("npus_count"),
            mesh_rows=mesh.get("rows"),
            mesh_columns=mesh.get("columns"),
            trace_digest=data["trace_digest"],
            request_mapping_digest=data["request_mapping_digest"],
            kv_event_digest=data["kv_event_digest"],
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


class MetricManifestBuilder:
    """Sidecar manifest builder used by the ET generator while it emits traces.

    The builder performs the generation-time completeness checks: contiguous
    unique queue indices, request uniqueness, stage boundary nodes for every
    expected rank, node ids within each rank's ET node range, and digest
    presence.  Identical events recorded twice are deduplicated.
    """

    def __init__(
        self,
        *,
        repo_variant: str,
        run_mode: str,
        npus_count: int,
        mesh_rows: int,
        mesh_columns: int,
        node_count_by_rank: Mapping[int, int],
        memory_projection: Optional[MemoryProjection] = None,
    ) -> None:
        _require_str(repo_variant, "repo_variant")
        if run_mode not in RUN_MODES:
            raise MetricsSchemaError(f"run_mode must be one of {sorted(RUN_MODES)}")
        _require_int(npus_count, "npus_count", 1)
        _require_int(mesh_rows, "mesh_rows", 1)
        _require_int(mesh_columns, "mesh_columns", 1)
        self.repo_variant = repo_variant
        self.run_mode = run_mode
        self.npus_count = npus_count
        self.mesh_rows = mesh_rows
        self.mesh_columns = mesh_columns
        self.memory_projection = memory_projection or MemoryProjection()
        self._node_count_by_rank = {
            _require_int(rank, "node_count_by_rank key", 0): _require_int(
                count, "node_count_by_rank value", 0
            )
            for rank, count in node_count_by_rank.items()
        }
        self._digests: dict[str, str] = {}
        self._requests: list[RequestMetadata] = []
        self._queue_indices: set[int] = set()
        self._events_by_rank: dict[int, set[NodeMetricEvent]] = {}
        self._memory_actions: list[Any] = []
        self._planner_memory_peaks: list[Any] = []

    def set_digests(
        self,
        *,
        trace_digest: str,
        request_mapping_digest: str,
        kv_event_digest: str,
    ) -> None:
        self._digests = {
            "trace_digest": _require_str(trace_digest, "trace_digest"),
            "request_mapping_digest": _require_str(
                request_mapping_digest, "request_mapping_digest"
            ),
            "kv_event_digest": _require_str(kv_event_digest, "kv_event_digest"),
        }

    def add_request(self, metadata: RequestMetadata) -> None:
        if not isinstance(metadata, RequestMetadata):
            raise MetricsSchemaError("add_request expects a RequestMetadata")
        if metadata.queue_index in self._queue_indices:
            raise MetricsSchemaError(
                f"duplicate queue_index {metadata.queue_index}"
            )
        for rank in metadata.prefill_ranks + metadata.decode_ranks:
            if rank >= self.npus_count:
                raise MetricsSchemaError(
                    f"request {metadata.queue_index} references rank {rank} "
                    f"beyond npus_count {self.npus_count}"
                )
        self._queue_indices.add(metadata.queue_index)
        self._requests.append(metadata)

    def add_node_event(
        self, rank: int, node_id: int, event_code: int, subject_id: int
    ) -> bool:
        """Record one boundary event; returns False if it was a duplicate."""

        event = NodeMetricEvent(
            node_id=node_id, event_code=event_code, subject_id=subject_id
        )
        _require_int(rank, "rank", 0)
        if rank >= self.npus_count:
            raise MetricsSchemaError(
                f"event rank {rank} beyond npus_count {self.npus_count}"
            )
        node_count = self._node_count_by_rank.get(rank)
        if node_count is None:
            raise MetricsSchemaError(f"no ET node count known for rank {rank}")
        if event.node_id >= node_count:
            raise MetricsSchemaError(
                f"node_id {event.node_id} out of range for rank {rank} "
                f"with {node_count} ET nodes"
            )
        events = self._events_by_rank.setdefault(rank, set())
        if event in events:
            return False
        events.add(event)
        return True

    def add_memory_action(self, action: Any) -> None:
        self._memory_actions.append(action)

    def add_planner_memory_peak(self, peak: Any) -> None:
        self._planner_memory_peaks.append(peak)

    def build(self) -> MetricsManifest:
        for digest_field in ("trace_digest", "request_mapping_digest", "kv_event_digest"):
            if digest_field not in self._digests:
                raise MetricsSchemaError(f"missing digest field {digest_field}")
        requests = tuple(sorted(self._requests, key=lambda r: r.queue_index))
        events_by_rank = {
            rank: tuple(sorted(events, key=lambda e: (e.node_id, e.event_code, e.subject_id)))
            for rank, events in sorted(self._events_by_rank.items())
        }
        if self.run_mode == RUN_MODE_SERVICE:
            self._check_service_stage_boundaries(requests, events_by_rank)
        return MetricsManifest(
            repo_variant=self.repo_variant,
            run_mode=self.run_mode,
            npus_count=self.npus_count,
            mesh_rows=self.mesh_rows,
            mesh_columns=self.mesh_columns,
            memory_projection=self.memory_projection,
            requests=requests,
            node_events_by_rank=events_by_rank,
            memory_actions=tuple(self._memory_actions),
            planner_memory_peaks=tuple(self._planner_memory_peaks),
            **self._digests,
        )

    def _check_service_stage_boundaries(
        self,
        requests: tuple[RequestMetadata, ...],
        events_by_rank: Mapping[int, tuple[NodeMetricEvent, ...]],
    ) -> None:
        """Every expected rank must carry its stage boundary nodes, and every
        service event subject must reference a known queue_index."""

        known_subjects = {request.queue_index for request in requests}
        for rank, events in events_by_rank.items():
            for event in events:
                if event.event_code in SERVICE_EVENT_CODES and (
                    event.subject_id not in known_subjects
                ):
                    raise MetricsSchemaError(
                        f"rank {rank} event {event.to_triple()} references an "
                        f"unknown queue_index {event.subject_id}"
                    )
        for request in requests:
            for code, ranks in (
                (EVENT_PREFILL_START, request.prefill_ranks),
                (EVENT_PREFILL_END, request.prefill_ranks),
                (EVENT_DECODE_START, request.decode_ranks),
                (EVENT_DECODE_END, request.decode_ranks),
            ):
                for rank in ranks:
                    if not any(
                        event.event_code == code and event.subject_id == request.queue_index
                        for event in events_by_rank.get(rank, ())
                    ):
                        raise MetricsSchemaError(
                            f"request {request.queue_index} is missing the "
                            f"event_code {code} boundary node on rank {rank}"
                        )


@dataclass(frozen=True)
class RequestTiming:
    """Schema-level reconstruction of one request's stage boundaries."""

    queue_index: int
    prefill_start_ns: Optional[int]
    prefill_end_ns: Optional[int]
    decode_start_ns: Optional[int]
    completion_ns: Optional[int]
    completed: bool


class RequestTimingTracker:
    """Pure-Python mirror of the collector's per-request timing semantics.

    Across all TP ranks of a stage the request start is the *minimum* issue
    tick and the request end is the *maximum* complete tick.  A request whose
    expected rank boundaries are only partially observed stays incomplete.
    """

    _CODE_TO_FIELD = {
        EVENT_PREFILL_START: "prefill_start_ns",
        EVENT_PREFILL_END: "prefill_end_ns",
        EVENT_DECODE_START: "decode_start_ns",
        EVENT_DECODE_END: "completion_ns",
    }

    def __init__(self, manifest: MetricsManifest) -> None:
        if not isinstance(manifest, MetricsManifest):
            raise MetricsSchemaError("RequestTimingTracker expects a MetricsManifest")
        if manifest.run_mode != RUN_MODE_SERVICE:
            raise MetricsSchemaError("RequestTimingTracker only supports service manifests")
        self._requests = {r.queue_index: r for r in manifest.requests}
        # queue_index -> event_code -> rank -> tick
        self._ticks: dict[int, dict[int, dict[int, int]]] = {
            request.queue_index: {code: {} for code in self._CODE_TO_FIELD}
            for request in manifest.requests
        }

    def record_event(
        self, rank: int, event_code: int, subject_id: int, tick_ns: int
    ) -> bool:
        """Record one observed boundary tick; duplicate (rank, code, subject)
        events with the same tick are deduplicated and return False."""

        _require_int(rank, "rank", 0)
        _require_int(tick_ns, "tick_ns", 0)
        if event_code not in self._CODE_TO_FIELD:
            raise MetricsSchemaError(
                f"event_code {event_code} is not a service stage boundary"
            )
        request = self._requests.get(subject_id)
        if request is None:
            raise MetricsSchemaError(f"unknown queue_index {subject_id}")
        expected_ranks = (
            request.prefill_ranks
            if event_code in (EVENT_PREFILL_START, EVENT_PREFILL_END)
            else request.decode_ranks
        )
        if rank not in expected_ranks:
            raise MetricsSchemaError(
                f"rank {rank} is not an expected rank for event_code {event_code} "
                f"of request {subject_id}"
            )
        seen = self._ticks[subject_id][event_code]
        if rank in seen:
            if seen[rank] != tick_ns:
                raise MetricsSchemaError(
                    f"conflicting ticks for request {subject_id} event_code "
                    f"{event_code} on rank {rank}: {seen[rank]} vs {tick_ns}"
                )
            return False
        seen[rank] = tick_ns
        return True

    def timing_for(self, queue_index: int) -> RequestTiming:
        request = self._requests.get(queue_index)
        if request is None:
            raise MetricsSchemaError(f"unknown queue_index {queue_index}")
        ticks = self._ticks[queue_index]

        def stage_value(code: int, ranks: tuple[int, ...], reducer) -> Optional[int]:
            seen = ticks[code]
            if any(rank not in seen for rank in ranks):
                return None
            return reducer(seen[rank] for rank in ranks)

        prefill_start = stage_value(EVENT_PREFILL_START, request.prefill_ranks, min)
        prefill_end = stage_value(EVENT_PREFILL_END, request.prefill_ranks, max)
        decode_start = stage_value(EVENT_DECODE_START, request.decode_ranks, min)
        completion = stage_value(EVENT_DECODE_END, request.decode_ranks, max)
        completed = all(
            value is not None
            for value in (prefill_start, prefill_end, decode_start, completion)
        )
        return RequestTiming(
            queue_index=queue_index,
            prefill_start_ns=prefill_start,
            prefill_end_ns=prefill_end,
            decode_start_ns=decode_start,
            completion_ns=completion,
            completed=completed,
        )

    def resolve_arrival_ns(self, queue_index: int) -> int:
        """Resolve a request arrival; parent-relative arrivals need the
        parent's actual completion tick from this tracker."""

        request = self._requests.get(queue_index)
        if request is None:
            raise MetricsSchemaError(f"unknown queue_index {queue_index}")
        arrival = request.arrival
        if arrival.kind == ARRIVAL_ABSOLUTE:
            return arrival.resolve()
        parent = self.timing_for(arrival.parent_queue_index)
        return arrival.resolve(parent.completion_ns)


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
        return cls(
            capacity_bytes=data.get("capacity_bytes"),
            weight_bytes=data.get("weight_bytes"),
            resident_kv_bytes=data.get("resident_kv_bytes"),
            reserved_kv_bytes=data.get("reserved_kv_bytes"),
            physical_used_bytes=data.get("physical_used_bytes"),
            committed_used_bytes=data.get("committed_used_bytes"),
            uncommitted_free_bytes=data.get("uncommitted_free_bytes"),
        )


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
        return cls(
            scope=data.get("scope"),
            rank=data.get("rank"),
            chiplet_index=data.get("chiplet_index"),
            peak_kind=data.get("peak_kind"),
            peak_value_bytes=data.get("peak_value_bytes"),
            capacity_bytes=data.get("capacity_bytes"),
            weight_at_peak_bytes=data.get("weight_at_peak_bytes"),
            resident_kv_at_peak_bytes=data.get("resident_kv_at_peak_bytes"),
            reserved_kv_at_peak_bytes=data.get("reserved_kv_at_peak_bytes"),
            free_at_peak_bytes=data.get("free_at_peak_bytes"),
            planner_time_ns=data.get("planner_time_ns"),
            sequence_index=data.get("sequence_index"),
            cause=data.get("cause"),
            request_id=data.get("request_id"),
            session_id=data.get("session_id"),
        )


@dataclass(frozen=True)
class ChipletMemoryResult:
    chiplet_index: int
    ledger: MemoryLedgerSnapshot
    peak_physical: MemoryPeakSnapshot
    peak_committed: MemoryPeakSnapshot

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
            chiplets_per_npu=data.get("chiplets_per_npu"),
            native_scope=data.get("native_scope", NATIVE_MEMORY_SCOPE),
            projection=data.get("projection", CHIPLET_PROJECTION_METHOD),
            ranks={
                int(rank): RankMemoryResult.from_dict(result)
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
    / reserved KV) and a measurement-only equal-striping chiplet projection.
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

        state.weight_bytes += delta.weight_delta_bytes
        state.resident_kv_bytes += delta.resident_kv_delta_bytes
        state.reserved_kv_bytes += delta.reserved_kv_delta_bytes
        if not (0 <= state.physical <= state.committed <= state.capacity_bytes):
            raise MetricsSchemaError(
                f"rank {delta.rank} ledger invariant violated after delta "
                f"{delta.sequence_index}: physical={state.physical} "
                f"committed={state.committed} capacity={state.capacity_bytes}"
            )

        self._apply_chiplet_projection(delta)
        self._rank_peaks[delta.rank][0].consider(state, delta)
        self._rank_peaks[delta.rank][1].consider(state, delta)
        for index, chiplet_state in enumerate(self._chiplets[delta.rank]):
            physical_peak, committed_peak = self._chiplet_peaks[delta.rank][index]
            physical_peak.consider(chiplet_state, delta)
            committed_peak.consider(chiplet_state, delta)

    def _apply_chiplet_projection(self, delta: MemoryDelta) -> None:
        """Stripe each component of the delta across the chiplets and attach
        it to the delta's allocation key; removes subtract the same recorded
        distribution for that key."""

        allocations = self._allocations[delta.rank]
        key = delta.allocation_key
        component_deltas = (
            delta.weight_delta_bytes,
            delta.resident_kv_delta_bytes,
            delta.reserved_kv_delta_bytes,
        )
        if all(component == 0 for component in component_deltas):
            return
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
            allocations.pop(key, None)
        else:
            allocations[key] = new_record

    def finalize(self) -> MemoryMetricsResult:
        if self._finalized is not None:
            return self._finalized
        ranks: dict[int, RankMemoryResult] = {}
        for rank, state in sorted(self._ranks.items()):
            chiplets = []
            for index, chiplet_state in enumerate(self._chiplets[rank]):
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
        denominator = max(values)
    elif method == NORMALIZATION_RATIO_TO_BASELINE:
        _require_str(baseline_run_id, "baseline_run_id")
        if baseline_value is None or isinstance(baseline_value, bool) or not isinstance(
            baseline_value, (int, float)
        ):
            raise MetricsSchemaError("ratio_to_baseline requires a numeric baseline_value")
        denominator = baseline_value
    else:
        raise MetricsSchemaError(
            f"normalization_method must be one of {sorted(NORMALIZATION_METHODS)}"
        )
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
