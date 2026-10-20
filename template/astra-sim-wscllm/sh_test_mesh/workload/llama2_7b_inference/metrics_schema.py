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
from dataclasses import dataclass
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

# D2 fix (2026-09-28): the RUN_MODE_SERVICE / RUN_MODE_MICROBENCHMARK /
# RUN_MODES constants were deleted -- zero references anywhere (the
# postprocessor keys run modes off the raw init-record strings, and the
# C++ collector validates run_mode independently).

NORMALIZATION_GROUP_MAX = "group_max"
NORMALIZATION_RATIO_TO_BASELINE = "ratio_to_baseline"
NORMALIZATION_METHODS = frozenset(
    {NORMALIZATION_GROUP_MAX, NORMALIZATION_RATIO_TO_BASELINE}
)

# D2 fix (2026-09-28): NATIVE_MEMORY_SCOPE / CHIPLET_PROJECTION_SCOPE /
# CHIPLET_PROJECTION_METHOD and the MemoryProjection declaration class were
# deleted -- zero references outside the class itself (the chiplet
# projection of the observer is fixed by DEFAULT_CHIPLETS_PER_NPU and the
# equal-striping implementation; no manifest declaration is parsed).
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
    """One request record of the manifest ``requests`` array.

    M11-4 (2026-09-26, face 仓同源移植): ``decode_length`` is part of the
    real manifest contract -- plan_materializer writes it on every request
    entry (WP9, SLO B2-C) and the C++ collector reads it as an optional
    field.  It was previously rejected here as an unknown field, which made
    every real artifact of this repository fail this frozen schema.

    Report 3.15 fix (2026-09-29): the same drift, remaining field set -- the
    unknown-field gate now accepts every field this repository's
    plan_materializer.py actually writes into the manifests' ``requests[]``
    records (frozen-contract sync, known-field set = producer output union):

    - ``metrics_manifest.json`` requests (``_derive_metrics_requests``):
      ``human_time_ns`` / ``tool_time_ns`` (WP2 sidecar passthrough, P2-13;
      absent when the sidecar row/column is empty).
    - ``manifest.json`` requests (``_derive_requests``):
      ``prefill_length`` / ``history_tokens_before`` /
      ``prefill_context_tokens`` / ``final_context_tokens`` (queue-derived
      token bookkeeping) and ``request_type`` / ``prefix_len`` (WP2 sidecar
      passthrough, absent when the sidecar row/column is empty).

    All of them stay optional (``absent = not declared``, matching the C++
    reader) and are validated/serialized only when declared.
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
    human_time_ns: Optional[int] = None
    tool_time_ns: Optional[int] = None
    prefill_length: Optional[int] = None
    history_tokens_before: Optional[int] = None
    prefill_context_tokens: Optional[int] = None
    final_context_tokens: Optional[int] = None
    request_type: Optional[str] = None
    prefix_len: Optional[int] = None

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
        if self.decode_length is not None:
            _require_int(self.decode_length, "decode_length", 0)
        for name in (
            "human_time_ns",
            "tool_time_ns",
            "prefill_length",
            "history_tokens_before",
            "prefill_context_tokens",
            "final_context_tokens",
            "prefix_len",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_int(value, name, 0)
        if self.request_type is not None:
            _require_str(self.request_type, "request_type")

    def to_dict(self) -> dict[str, Any]:
        result = {
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
        # Optional fields: serialized only when declared, matching the C++
        # reader's "absent = not declared" semantics.
        if self.decode_length is not None:
            result["decode_length"] = self.decode_length
        for name in (
            "human_time_ns",
            "tool_time_ns",
            "prefill_length",
            "history_tokens_before",
            "prefill_context_tokens",
            "final_context_tokens",
            "prefix_len",
        ):
            value = getattr(self, name)
            if value is not None:
                result[name] = value
        if self.request_type is not None:
            result["request_type"] = self.request_type
        return result

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RequestMetadata":
        data = _require_mapping(data, "request")
        optional = {
            "decode_length",
            "human_time_ns",
            "tool_time_ns",
            "prefill_length",
            "history_tokens_before",
            "prefill_context_tokens",
            "final_context_tokens",
            "request_type",
            "prefix_len",
        }
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
        } | optional
        unknown = set(data) - expected
        if unknown:
            raise MetricsSchemaError(f"request has unknown fields {sorted(unknown)}")
        missing = expected - set(data) - optional
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
            decode_length=data.get("decode_length"),
            human_time_ns=data.get("human_time_ns"),
            tool_time_ns=data.get("tool_time_ns"),
            prefill_length=data.get("prefill_length"),
            history_tokens_before=data.get("history_tokens_before"),
            prefill_context_tokens=data.get("prefill_context_tokens"),
            final_context_tokens=data.get("final_context_tokens"),
            request_type=data.get("request_type"),
            prefix_len=data.get("prefix_len"),
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

    # (2026-10-01) to_triple/from_triple are test-support round-trip
    # helpers (consumed only by sh_test_mesh/tests/test_metrics_contract.py);
    # the production serialization path never goes through the triple form.
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

    Negative values stripe their magnitude and carry the sign.  NOTE (D2
    fix, 2026-09-28): striping is NOT additive across segments --
    stripe(a) + stripe(b) != stripe(a+b) in general (remainders stack on
    the low chiplets), so removals must NOT be computed by striping the
    aggregate magnitude; see ``_removal_distribution``.
    """

    _require_int(total_bytes, "total_bytes")
    _require_int(partitions, "partitions", 1)
    sign = 1 if total_bytes >= 0 else -1
    base, remainder = divmod(abs(total_bytes), partitions)
    return tuple(
        sign * (base + (1 if index < remainder else 0)) for index in range(partitions)
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

    def initialize_rank(self, rank: int, capacity_bytes: int) -> None:
        _require_int(rank, "rank", 0)
        _require_int(capacity_bytes, "capacity_bytes", 1)
        if rank in self._ranks:
            raise MetricsSchemaError(f"rank {rank} is already initialized")
        self._ranks[rank] = _ComponentState(capacity_bytes=capacity_bytes)
        capacities = stripe_bytes_evenly(capacity_bytes, self.chiplets_per_npu)
        self._chiplets[rank] = [
            _ComponentState(capacity_bytes=capacity) for capacity in capacities
        ]
        self._allocations[rank] = {}

    def record_delta(self, delta: MemoryDelta) -> None:
        if not isinstance(delta, MemoryDelta):
            raise MetricsSchemaError("record_delta expects a MemoryDelta")
        state = self._ranks.get(delta.rank)
        if state is None:
            raise MetricsSchemaError(f"rank {delta.rank} is not initialized")

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

    def _apply_chiplet_projection(self, delta: MemoryDelta) -> None:
        """Stripe each component of the delta across the chiplets and attach
        it to the delta's allocation key; removes subtract the same recorded
        distribution for that key.

        D2 fix (2026-09-28): removals now walk back the key's RECORDED
        per-chiplet distribution (``_removal_distribution``) instead of
        striping the negative delta's magnitude.  Striping is not additive
        across segments (stripe(a)+stripe(b) != stripe(a+b): remainders
        stack on the low chiplets), so multi-segment growth followed by a
        whole-segment release used to leave a residual distribution like
        (1, 0, 0, -1) -- silent negative per-chiplet occupancy plus a ghost
        allocation record that is never reclaimed.  Over-removal (more
        than the key holds) fails closed.
        """

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
            if component < 0:
                stripe = tuple(
                    -value
                    for value in self._removal_distribution(
                        record[index], -component)
                )
            else:
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

    def _removal_distribution(self, dist: tuple[int, ...],
                              amount: int) -> tuple[int, ...]:
        """Deterministic removal of ``amount`` bytes from a key's recorded
        per-chiplet distribution: largest recorded amount first, ties by
        chiplet index (equal-striping orders the remainder the same way, so
        a single-segment key removes exactly its own striping and walks
        back to zero).  Removing more than the key holds fails closed.
        """

        removed = [0] * len(dist)
        remaining = amount
        for index in sorted(range(len(dist)), key=lambda i: (-dist[i], i)):
            take = min(dist[index], remaining)
            removed[index] = take
            remaining -= take
            if remaining == 0:
                break
        if remaining:
            raise MetricsSchemaError(
                f"cannot remove {amount} bytes from allocation holding "
                f"{sum(dist)} across chiplets {dist}: per-chiplet "
                f"occupancy would go negative (producer ledger bug, "
                f"fail-closed)"
            )
        return tuple(removed)


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
            # P1 fix (2026-09-28): a NaN/Inf field would slip every "<=" /
            # ">" comparison (nan <= 0 is False) and silently poison the
            # normalized columns -- every numeric field must be finite.
            if not math.isfinite(float(value)):
                raise MetricsSchemaError(
                    f"{name} must be a finite number (got {value!r})"
                )
        if not (math.isfinite(float(self.denominator_value))
                and self.denominator_value > 0):
            raise MetricsSchemaError(
                "normalization denominator must be finite and > 0"
            )

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
    zero denominator is never usable; nor is a non-finite one (P1 fix,
    2026-09-28: NaN slips the former ``<= 0`` comparison -- ``nan <= 0`` is
    False -- and max() propagates a NaN that sits in the first slot, so
    EVERY group value and the baseline must be finite, and the final
    denominator must be finite and positive).
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
            if not math.isfinite(float(value)):
                raise MetricsSchemaError(
                    f"group values must be finite (got {value!r})"
                )
        denominator = max(values)
    elif method == NORMALIZATION_RATIO_TO_BASELINE:
        _require_str(baseline_run_id, "baseline_run_id")
        if baseline_value is None or isinstance(baseline_value, bool) or not isinstance(
            baseline_value, (int, float)
        ):
            raise MetricsSchemaError("ratio_to_baseline requires a numeric baseline_value")
        if not math.isfinite(float(baseline_value)):
            raise MetricsSchemaError(
                f"baseline_value must be finite (got {baseline_value!r})"
            )
        denominator = baseline_value
    else:
        raise MetricsSchemaError(
            f"normalization_method must be one of {sorted(NORMALIZATION_METHODS)}"
        )
    if not (math.isfinite(float(denominator)) and denominator > 0):
        raise MetricsSchemaError(
            "normalization denominator must be finite and > 0"
        )
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
