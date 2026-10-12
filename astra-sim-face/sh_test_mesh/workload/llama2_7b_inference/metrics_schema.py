"""Frozen metrics schema and pure-Python contract fixtures.

This module pins down the observation-only data protocol used by the metrics
collection work: the node event integer encoding and the cross-run
normalization fields.  (2026-09-29: the manifest layer -- MetricsManifest /
MetricManifestBuilder / RequestTimingTracker / MemoryMetricsObserver and
their exclusive support types -- was removed as production-dead code; the
postprocessor consumes archived jsonl plus the normalization tail only.)

Nothing in this module feeds decisions back into the scheduler, the KV
managers, or the ET generators; it only describes and validates records that
those components may emit.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Optional, Sequence


SCHEMA_VERSION = 1

# Node event integer encoding (manifest protocol).  Each event is stored as
# the triple [node_id, event_code, subject_id]; the subject is a queue_index
# in service mode. (2026-09-30 note: the microbenchmark subject flavor and
# its ingestion path were removed from this repo -- A4-5; the codes below
# stay frozen as the cross-repo manifest protocol.)
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


NORMALIZATION_GROUP_MAX = "group_max"
NORMALIZATION_RATIO_TO_BASELINE = "ratio_to_baseline"
NORMALIZATION_METHODS = frozenset(
    {NORMALIZATION_GROUP_MAX, NORMALIZATION_RATIO_TO_BASELINE}
)


class MetricsSchemaError(ValueError):
    """Raised whenever a metrics record violates the frozen schema."""


def _require_int(value: Any, name: str, minimum: Optional[int] = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise MetricsSchemaError(f"{name} must be an integer")
    if minimum is not None and value < minimum:
        raise MetricsSchemaError(f"{name} must be >= {minimum}")
    return value


def _require_str(value: Any, name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value):
        raise MetricsSchemaError(f"{name} must be a non-empty string")
    return value


def _require_optional_str(value: Any, name: str) -> Optional[str]:
    if value is None:
        return None
    return _require_str(value, name)


def _require_mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise MetricsSchemaError(f"{name} must be an object")
    return value


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


# ---------------------------------------------------------------------
# (2026-09-29) 死代码删除:manifest 层(MetricsManifest /
# MetricManifestBuilder(含零调用的 add_memory_action /
# add_planner_memory_peak)/ RequestTiming(RequestTimingTracker)/
# MemoryMetricsObserver 及其独占支持类型 Arrival/RequestMetadata/
# MemoryProjection/MemoryDelta/Memory*Snapshot/MemoryMetricsResult/
# stripe_bytes_evenly/check_schema_version 等)全仓仅测试引用、无任何
# 生产消费者(metrics_postprocess 只消费归档 jsonl 与本模块的归一化
# 尾部),整层移除;钉住它们的测试用例同步删除/改写(事件编码常量与
# 归一化契约保留,仍由存活测试钉住)。
