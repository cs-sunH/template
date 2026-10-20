#!/usr/bin/env python3
"""thermal_epoch.py -- plan D (Python side): thermal_activity -> thermal_cap.

Frozen wire (both sides; names must not drift):

Request top-level sibling (C++ -> Python, additive)::

    "thermal_activity": {
      "schema_version": 1,
      "epoch_ns": <int>, "window_start_ns": <int>, "window_end_ns": <int>,
      "coverage": "measured",
      "provenance": "coarse_runtime_counters_delta",
      "sources": [
        {"source_id": "die<N>.comp", "action_kind": "pe",        "count": <num>},
        {"source_id": "die<N>.comp", "action_kind": "vec",       "count": <num>},
        {"source_id": "die<N>.hbm",  "action_kind": "hbm_read",  "count": <num>},
        {"source_id": "die<N>.hbm",  "action_kind": "hbm_write", "count": <num>}
      ]
    }

Response top-level field (Python -> C++, present only when the request carried
``thermal_activity``)::

    "thermal_cap": {
      "schema_version": 1,
      "resource_epoch": <int>, "effective_from_ns": <int>,
      "events": [{"resource_index": <0..12>, "absolute_physical_rate": <>=0>}]
    }

Ownership: this module parses/validates the *activity* wire and translates the
joint envelope into the *cap* wire.  It copies no energy or thermal formula:
power, temperature and the throughput fraction come from the embedded
``wafercosim`` session and the joint ``thermal_policy``.  The only arithmetic
here is the already-declared ``fraction * COMPUTE_GATE_PEAK_RATE`` (the same
multiplication the live capacity install performs).

The activity source identity uses the coarse *served category* name (``comp``)
while the thermal floorplan/mapping names the compute source ``die<N>.compute``;
the parser owns exactly that taxonomy translation (``comp`` -> ``compute``).
The C++ side can only observe the *rank*, so it emits ``die<rank>.comp`` /
``die<rank>.hbm``; the parser resolves that rank to the thermal die through the
:class:`MappingSnapshot`'s ``rank_to_die`` (never inventing a mapping) before
binding the source to a physical block.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

from wafercosim.contracts import ActivityWindow, ContractError

from online.coarse_runtime_client import (
    COMPUTE_GATE_PEAK_RATE,
    RESOURCE_COMPUTE_GATE,
)

ACTIVITY_SCHEMA_VERSION = 1
CAP_SCHEMA_VERSION = 1
#: action_kind values the frozen wire models.  Anything else is *reported*
#: (collected and rejected), never silently treated as a zero count.
ALLOWED_ACTION_KINDS = ("pe", "vec", "hbm_read", "hbm_write")
_COVERAGE_VALUES = ("measured", "estimated", "assumed")
_REQUIRED_SOURCE_KEYS = ("source_id", "action_kind", "count")
#: action_kind -> the physical-source suffix the wire must name.  ``comp`` is
#: the coarse served category; ``compute`` is accepted too so a peer that
#: already emits the mapping spelling is not rejected for that alone.
_ACTION_SOURCE_SUFFIX = {
    "pe": ("comp", "compute"),
    "vec": ("comp", "compute"),
    "hbm_read": ("hbm",),
    "hbm_write": ("hbm",),
}
#: wire source suffix -> mapping source suffix (the only identity translation).
_SOURCE_SUFFIX_TO_MAPPING = {"comp": "compute", "compute": "compute", "hbm": "hbm"}


class ThermalEpochError(ContractError):
    """A ``thermal_activity`` / ``thermal_cap`` wire payload is malformed."""


def _int(value: Any, name: str, *, nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ThermalEpochError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise ThermalEpochError(f"{name} must be non-negative")
    return value


def _number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ThermalEpochError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ThermalEpochError(f"{name} must be finite")
    if number < 0.0:
        raise ThermalEpochError(f"{name} must be non-negative")
    return number


def _mapping_source(source_id: str, kind: str, label: str,
                    mapping: "Any | None" = None) -> str:
    """Translate a wire source identity to the physical mapping's source name.

    The wire names the compute source ``die<N>.comp`` (the coarse served
    category) and the HBM source ``die<N>.hbm``; the thermal mapping names the
    compute source ``die<D>.compute``.  Two identity translations happen here:

    * the served-category suffix ``comp`` -> ``compute`` (the only taxonomy
      translation), also required to match the action kind; and
    * the die index ``N``.  The C++ side can only observe the *rank* (a
      documented placeholder), so when a mapping is supplied ``N`` is first
      resolved through ``mapping.rank_to_die[N]`` to its thermal die number.
      If ``N`` is not a known rank it is treated as an already-resolved thermal
      die number and must name a source present in ``source_block_weights``.
      Only if neither holds is the source rejected, with a named diagnostic.
      ``rank_to_die`` always comes from the :class:`MappingSnapshot`; it is
      never invented here.

    Without a mapping the wire name is returned with only the suffix translated.
    """
    head, _, suffix = source_id.partition(".")
    if not head.startswith("die"):
        raise ThermalEpochError(
            label + ".source_id must name a 'die<N>.<kind>' physical source, "
            f"got {source_id!r}")
    try:
        index = int(head[3:])
    except ValueError:
        raise ThermalEpochError(
            label + f".source_id has a bad die index: {source_id!r}") from None
    if not suffix:
        raise ThermalEpochError(
            label + f".source_id is missing a source kind: {source_id!r}")
    expected = _ACTION_SOURCE_SUFFIX[kind]
    if suffix not in expected:
        raise ThermalEpochError(
            label + f": action_kind {kind!r} requires source suffix "
            f"{expected[0]!r}, got {suffix!r}")
    mapping_suffix = _SOURCE_SUFFIX_TO_MAPPING[suffix]
    if mapping is None:
        return f"die{index}.{mapping_suffix}"
    known = set(mapping.source_block_weights) | set(
        mapping.source_to_destination)
    # Rank first (the wire's only observable identity), then treat the value as
    # an already-resolved thermal die number.
    candidate_dies: list[int] = []
    if index in mapping.rank_to_die:
        candidate_dies.append(int(mapping.rank_to_die[index]))
    candidate_dies.append(index)
    for die in candidate_dies:
        candidate = f"die{die}.{mapping_suffix}"
        if candidate in known:
            return candidate
    tried = [f"die{die}.{mapping_suffix}" for die in candidate_dies]
    raise ThermalEpochError(
        label + f": source {source_id!r} is absent from the physical mapping; "
        f"rank/die index {index} is neither a mapped rank (rank_to_die has "
        f"{len(mapping.rank_to_die)} ranks) nor a present thermal die; "
        f"tried {tried}")


def parse_thermal_activity(payload: Mapping[str, Any], *, run_id: str,
                           hardware_hash: str, mapping_epoch: str,
                           resource_epoch: int, activity_seq: int,
                           mapping: "Any | None" = None) -> ActivityWindow:
    """Validate one ``thermal_activity`` wire payload into an ActivityWindow.

    The window is bound to the current bridge session (``hardware_hash`` /
    ``mapping_epoch`` / ``resource_epoch`` are taken from it).  An empty
    ``sources`` list still yields an explicit zero-activity window: idle time
    must be advanced so background leakage and cooling still occur.

    Rejected (fail-closed) cases: a non-object payload; a wrong/absent
    ``schema_version``; non-integer or negative times; ``window_end_ns !=
    epoch_ns``; ``window_start_ns >= epoch_ns``; an unknown ``coverage``; a
    non-list ``sources`` or a malformed source row; a non-finite/negative
    ``count``; an ``action_kind`` outside the four modelled kinds (reported,
    never zeroed); a source/action_kind suffix mismatch; and, when a mapping is
    supplied, a source absent from the physical mapping.
    """
    if not isinstance(payload, Mapping):
        raise ThermalEpochError("thermal_activity must be an object")
    if (type(payload.get("schema_version")) is not int
            or payload["schema_version"] != ACTIVITY_SCHEMA_VERSION):
        raise ThermalEpochError(
            "thermal_activity schema_version {!r} != {}".format(
                payload.get("schema_version"), ACTIVITY_SCHEMA_VERSION))
    for field in ("epoch_ns", "window_start_ns", "window_end_ns"):
        if field not in payload:
            raise ThermalEpochError(
                "thermal_activity missing field {!r}".format(field))
    epoch_ns = _int(payload["epoch_ns"], "thermal_activity.epoch_ns",
                    nonnegative=True)
    window_start_ns = _int(payload["window_start_ns"],
                           "thermal_activity.window_start_ns",
                           nonnegative=True)
    window_end_ns = _int(payload["window_end_ns"],
                         "thermal_activity.window_end_ns",
                         nonnegative=True)
    if window_end_ns != epoch_ns:
        raise ThermalEpochError(
            "thermal_activity window_end_ns {} != epoch_ns {} (the window "
            "must end at its epoch)".format(window_end_ns, epoch_ns))
    if window_start_ns >= epoch_ns:
        raise ThermalEpochError(
            "thermal_activity window_start_ns {} must be < epoch_ns {}".format(
                window_start_ns, epoch_ns))
    coverage = payload.get("coverage", "measured")
    if coverage not in _COVERAGE_VALUES:
        raise ThermalEpochError(
            "thermal_activity coverage {!r} must be one of {}".format(
                coverage, list(_COVERAGE_VALUES)))
    provenance = payload.get("provenance", "")
    if not isinstance(provenance, str):
        raise ThermalEpochError("thermal_activity.provenance must be a string")

    raw_sources = payload.get("sources", [])
    if not isinstance(raw_sources, list):
        raise ThermalEpochError("thermal_activity.sources must be a list")
    actions: list[tuple[str, str, float]] = []
    unmodelled: list[str] = []
    for index, row in enumerate(raw_sources):
        label = "thermal_activity.sources[{}]".format(index)
        if not isinstance(row, Mapping):
            raise ThermalEpochError(label + " must be an object")
        for key in _REQUIRED_SOURCE_KEYS:
            if key not in row:
                raise ThermalEpochError(
                    label + " missing field {!r}".format(key))
        source_id = row["source_id"]
        if not isinstance(source_id, str) or not source_id.strip():
            raise ThermalEpochError(
                label + ".source_id must be a non-empty string")
        kind = row["action_kind"]
        if not isinstance(kind, str):
            raise ThermalEpochError(label + ".action_kind must be a string")
        if kind not in ALLOWED_ACTION_KINDS:
            unmodelled.append(kind)
            continue
        count = _number(row["count"], label + ".count")
        if count > 0.0:
            actions.append((
                _mapping_source(source_id, kind, label, mapping), kind, count))
    if unmodelled:
        raise ThermalEpochError(
            "thermal_activity carries unmodelled action_kind values {!r}; "
            "unmodelled actions must be reported, not assumed zero".format(
                sorted(set(unmodelled))))
    if mapping is not None:
        known = set(mapping.source_block_weights) | set(
            mapping.source_to_destination)
        for source, kind, _ in actions:
            if source not in known:
                raise ThermalEpochError(
                    "thermal_activity source {!r} (action {!r}) is absent from "
                    "the physical mapping".format(source, kind))
    return ActivityWindow(
        run_id=run_id,
        activity_seq=activity_seq,
        start_ns=window_start_ns,
        end_ns=window_end_ns,
        hardware_hash=hardware_hash,
        mapping_epoch=mapping_epoch,
        resource_epoch=resource_epoch,
        actions=tuple(actions),
        dwell=(),
        operating_point=None,
        coverage=coverage,
        provenance=provenance,
    )


def envelope_to_thermal_cap(envelope: Any) -> dict[str, Any]:
    """Translate a joint ``ResourceEnvelope`` into the ``thermal_cap`` wire.

    One event per control domain, all on ``RESOURCE_COMPUTE_GATE`` (0) -- the
    only resource the v1 policy controls.  The frozen wire carries no rank/die
    discriminator, so a multi-domain envelope lists one event per domain and the
    C++ side applies them within its epoch scope.  Rate is the absolute
    physical rate ``fraction * COMPUTE_GATE_PEAK_RATE``.
    """
    events: list[dict[str, Any]] = []
    for row in envelope.domains:
        rate = float(row["compute_throughput_fraction"]) * COMPUTE_GATE_PEAK_RATE
        if not math.isfinite(rate) or rate < 0.0:
            raise ThermalEpochError(
                "envelope domain {!r} yields a non-finite/negative rate "
                "{!r}".format(row.get("domain"), rate))
        events.append({
            "resource_index": RESOURCE_COMPUTE_GATE,
            "absolute_physical_rate": rate,
        })
    if not events:
        raise ThermalEpochError("resource envelope carries no control domain")
    return {
        "schema_version": CAP_SCHEMA_VERSION,
        "resource_epoch": int(envelope.resource_epoch),
        "effective_from_ns": int(envelope.effective_from_ns),
        # ``applied`` distinguishes the shadow control arm from closed_loop on the
        # wire: shadow proposes the cap (applied=false) and the C++ side must NOT
        # install it, so shadow stays physically equal to off.
        "applied": bool(all(bool(row.get("applied", False))
                            for row in envelope.domains)),
        "events": events,
    }


class ThermalEpochService:
    """Drive one run-long :class:`ThermalBridge` from ``thermal_activity``.

    ``cap_for_request`` parses the wire payload into an ``ActivityWindow`` bound
    to the bridge session, ingests it (power -> temperature -> policy envelope),
    and returns the ``thermal_cap`` wire dict.  It returns ``None`` -- a cap
    absence, recorded -- when the bridge is disabled, so a request that carries
    activity but reaches an off/unwired service continues safely instead of
    crashing.  ``shadow`` marks the envelope ``applied=False``; ``closed_loop``
    marks it ``applied=True`` and advances the epoch (both handled by the
    bridge, not duplicated here).
    """

    def __init__(self, bridge: Any, *, run_id: str | None = None):
        self.bridge = bridge
        self.run_id = run_id or (
            getattr(bridge, "run_id", "") if bridge is not None else "")
        self._activity_seq = 0
        self._outcomes: list[dict[str, Any]] = []

    @property
    def enabled(self) -> bool:
        return self.bridge is not None and bool(
            getattr(self.bridge, "enabled", False))

    def cap_for_request(self, payload: Mapping[str, Any], *,
                        delivery_sequence: int | None = None
                        ) -> dict[str, Any] | None:
        if not self.enabled:
            self._record(delivery_sequence, outcome="thermal_bridge_unavailable")
            return None
        activity = parse_thermal_activity(
            payload,
            run_id=self.run_id,
            hardware_hash=self.bridge.hardware.content_hash,
            mapping_epoch=self.bridge.mapping.mapping_epoch,
            mapping=self.bridge.mapping,
            resource_epoch=self.bridge.resource_epoch,
            activity_seq=self._activity_seq,
        )
        result = self.bridge.ingest(activity)
        self._activity_seq += 1
        if result is None:
            self._record(delivery_sequence, outcome="thermal_session_absent")
            return None
        envelope = result["envelope"]
        cap = envelope_to_thermal_cap(envelope)
        self._record(
            delivery_sequence, outcome="cap_produced", envelope=envelope,
            activity_seq=activity.activity_seq, epoch_ns=activity.end_ns)
        return cap

    def _record(self, delivery_sequence: int | None, *, outcome: str,
                envelope: Any = None, activity_seq: int | None = None,
                epoch_ns: int | None = None) -> None:
        row: dict[str, Any] = {
            "delivery_sequence": delivery_sequence,
            "outcome": outcome,
        }
        if activity_seq is not None:
            row["activity_seq"] = activity_seq
        if epoch_ns is not None:
            row["epoch_ns"] = epoch_ns
        if envelope is not None:
            row["resource_epoch"] = envelope.resource_epoch
            row["effective_from_ns"] = envelope.effective_from_ns
            row["applied"] = all(
                bool(domain.get("applied")) for domain in envelope.domains)
        self._outcomes.append(row)

    def outcome_rows(self) -> list[dict[str, Any]]:
        return [dict(row) for row in self._outcomes]

    def write_audit(self, directory: str | Path) -> Path:
        """Persist the per-epoch outcome ledger (best-effort, caller-guarded)."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / "thermal_epochs.jsonl"
        with path.open("w", encoding="utf-8") as sink:
            for row in self._outcomes:
                sink.write(json.dumps(row, sort_keys=True) + "\n")
        return path


def attach_thermal_cap(delta: Mapping[str, Any], batch: dict[str, Any],
                       service: "ThermalEpochService | None") -> dict[str, Any] | None:
    """Response-assembly hook: add ``thermal_cap`` iff the request carried one.

    A request without ``thermal_activity`` never gets ``thermal_cap`` (early
    return).  A request with activity reaching a missing/disabled service
    records the absence and leaves the response untouched (safe continue).
    """
    if "thermal_activity" not in delta:
        return None
    if service is None:
        return None
    cap = service.cap_for_request(
        delta["thermal_activity"],
        delivery_sequence=delta.get("delivery_sequence"))
    if cap is not None:
        batch["thermal_cap"] = cap
    return cap


__all__ = [
    "ACTIVITY_SCHEMA_VERSION", "CAP_SCHEMA_VERSION", "ALLOWED_ACTION_KINDS",
    "ThermalEpochError", "parse_thermal_activity", "envelope_to_thermal_cap",
    "ThermalEpochService", "attach_thermal_cap",
]
