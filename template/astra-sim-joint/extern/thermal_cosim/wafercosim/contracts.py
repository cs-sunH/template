"""Stable data contracts between the joint host and the embedded thermal core.

The joint scheduler/executor owns requests, phases, KV, migration and task
progress.  It exports only *facts* to this component: a parsed hardware
snapshot, a physical mapping, incremental activity windows and an operating
point.  This module turns those facts into power windows and thermal feedback
without copying any joint scheduling responsibility back into the thermal
configuration (see ``config.py`` schema v2).

Three grids stay distinct and are never inferred from each other:

1. Astra rank / network topology (host owned),
2. per-die Core grid (host owned),
3. thermal block / floorplan grid (thermal profile owned).

Units at the boundary: simulation time ns, thermal API time s, frequency Hz,
throughput op/s or byte/s, action counts op/byte, energy J, power W,
temperature K, geometry m.  The only ns->s conversion happens in
:meth:`ThermalSession.advance_committed`.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

SCHEMA_VERSION = 1
NS_PER_S = 1_000_000_000

# Physical action semantics initialised from ``action_energy.py``.  Counts use
# the fixed unit each action registers; unknown actions must be reported, never
# silently treated as zero.
ACTION_UNITS = {
    "pe": "op",
    "vec": "op",
    "control": "action",
    "sram_read": "byte",
    "sram_write": "byte",
    "hbm": "byte",
    "hbm_read": "byte",
    "hbm_write": "byte",
    "dram": "byte",
    "dram_read": "byte",
    "dram_write": "byte",
    "noc": "byte",
    "d2d": "byte",
    "dram_io": "byte",
}

# Thermal destination per action (mirrors action_energy.THERMAL_DESTINATIONS).
ACTION_DESTINATIONS = {
    "pe": "compute",
    "vec": "compute",
    "control": "compute",
    "sram_read": "compute",
    "sram_write": "compute",
    "hbm": "hbm",
    "hbm_read": "hbm",
    "hbm_write": "hbm",
    "dram": "external",
    "dram_read": "external",
    "dram_write": "external",
    "noc": "router",
    "d2d": "router",
    "dram_io": "router",
}

_BACKGROUND_KINDS = (
    "compute_idle",
    "compute_leak",
    "sram_leak",
    "hbm_idle",
    "router_idle",
    "dram_background",
)


class ContractError(ValueError):
    """A host-supplied contract is malformed or internally inconsistent."""


def _finite(value: Any, name: str, *, positive: bool = False,
            nonnegative: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ContractError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number):
        raise ContractError(f"{name} must be finite")
    if positive and number <= 0:
        raise ContractError(f"{name} must be positive")
    if nonnegative and number < 0:
        raise ContractError(f"{name} must be non-negative")
    return number


def _integer(value: Any, name: str, *, positive: bool = False,
             nonnegative: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ContractError(f"{name} must be an integer")
    if positive and value <= 0:
        raise ContractError(f"{name} must be positive")
    if nonnegative and value < 0:
        raise ContractError(f"{name} must be non-negative")
    return value


def _nonempty(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ContractError(f"{name} must be a non-empty string")
    return value


def content_hash(payload: Mapping[str, Any]) -> str:
    """Canonical sha256 over a JSON object (sorted keys, compact)."""
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


# --------------------------------------------------------------- hardware --

@dataclass(frozen=True)
class HardwareSnapshot:
    """Normalised hardware facts exported by the joint config resolver.

    ``content_hash`` binds a run to the exact parsed hardware; the thermal core
    validates its own geometry/energy domain against these facts but never
    overwrites them with a table's reference values.
    """

    schema_version: int
    content_hash: str
    source_path: str
    physical_sources: tuple[str, ...]
    units: dict[str, str]
    mesh_rows: int
    mesh_cols: int
    dies: tuple[dict[str, Any], ...]
    resource_peaks: dict[str, float]
    sram: dict[str, Any]
    hbm: dict[str, Any]
    dram: dict[str, Any]
    noc: dict[str, Any]
    d2d_bandwidth_bytes_s: float
    clock_hz: float
    precision: str
    energy_hardware_facts: dict[str, Any]
    target_calibrated: bool = False
    notes: tuple[str, ...] = ()

    @staticmethod
    def build(*, source_path: str, units: Mapping[str, str], mesh_rows: int,
              mesh_cols: int, dies: list[dict[str, Any]],
              resource_peaks: Mapping[str, float], sram: Mapping[str, Any],
              hbm: Mapping[str, Any], dram: Mapping[str, Any],
              noc: Mapping[str, Any], d2d_bandwidth_bytes_s: float,
              clock_hz: float, precision: str,
              energy_hardware_facts: Mapping[str, Any],
              physical_sources: list[str] | None = None,
              target_calibrated: bool = False,
              notes: list[str] | None = None) -> "HardwareSnapshot":
        rows = [dict(row) for row in dies]
        if not rows:
            raise ContractError("hardware snapshot requires at least one die")
        for index, row in enumerate(rows):
            _integer(row.get("die"), f"dies[{index}].die", nonnegative=True)
            _integer(row.get("cores_x"), f"dies[{index}].cores_x", positive=True)
            _integer(row.get("cores_y"), f"dies[{index}].cores_y", positive=True)
            _finite(row.get("clock_hz"), f"dies[{index}].clock_hz", positive=True)
        peaks = {key: _finite(value, f"resource_peaks.{key}", nonnegative=True)
                 for key, value in resource_peaks.items()}
        sources = tuple(physical_sources) if physical_sources else tuple(
            f"die{row['die']}.{kind}"
            for row in rows for kind in ("compute", "hbm", "router"))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "source_path": source_path,
            "units": dict(units),
            "mesh_rows": mesh_rows,
            "mesh_cols": mesh_cols,
            "dies": rows,
            "resource_peaks": peaks,
            "sram": dict(sram),
            "hbm": dict(hbm),
            "dram": dict(dram),
            "noc": dict(noc),
            "d2d_bandwidth_bytes_s": d2d_bandwidth_bytes_s,
            "clock_hz": clock_hz,
            "precision": precision,
            "energy_hardware_facts": dict(energy_hardware_facts),
            "physical_sources": list(sources),
            "target_calibrated": bool(target_calibrated),
            "notes": list(notes or ()),
        }
        return HardwareSnapshot(
            schema_version=SCHEMA_VERSION,
            content_hash=content_hash(payload),
            source_path=source_path,
            physical_sources=sources,
            units=dict(units),
            mesh_rows=_integer(mesh_rows, "mesh_rows", positive=True),
            mesh_cols=_integer(mesh_cols, "mesh_cols", positive=True),
            dies=tuple(rows),
            resource_peaks=peaks,
            sram=dict(sram),
            hbm=dict(hbm),
            dram=dict(dram),
            noc=dict(noc),
            d2d_bandwidth_bytes_s=d2d_bandwidth_bytes_s,
            clock_hz=clock_hz,
            precision=precision,
            energy_hardware_facts=dict(energy_hardware_facts),
            target_calibrated=bool(target_calibrated),
            notes=tuple(notes or ()),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "content_hash": self.content_hash,
            "source_path": self.source_path,
            "physical_sources": list(self.physical_sources),
            "units": dict(self.units),
            "mesh_rows": self.mesh_rows,
            "mesh_cols": self.mesh_cols,
            "dies": [dict(row) for row in self.dies],
            "resource_peaks": dict(self.resource_peaks),
            "sram": dict(self.sram),
            "hbm": dict(self.hbm),
            "dram": dict(self.dram),
            "noc": dict(self.noc),
            "d2d_bandwidth_bytes_s": self.d2d_bandwidth_bytes_s,
            "clock_hz": self.clock_hz,
            "precision": self.precision,
            "energy_hardware_facts": dict(self.energy_hardware_facts),
            "target_calibrated": self.target_calibrated,
            "notes": list(self.notes),
        }

    @property
    def die_ids(self) -> tuple[int, ...]:
        return tuple(int(row["die"]) for row in self.dies)

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "HardwareSnapshot":
        """Reconstruct a snapshot from its audit JSON, re-verifying the hash."""
        snapshot = HardwareSnapshot.build(
            source_path=str(payload["source_path"]),
            units=dict(payload["units"]),
            mesh_rows=int(payload["mesh_rows"]),
            mesh_cols=int(payload["mesh_cols"]),
            dies=[dict(row) for row in payload["dies"]],
            resource_peaks=dict(payload["resource_peaks"]),
            sram=dict(payload["sram"]),
            hbm=dict(payload["hbm"]),
            dram=dict(payload["dram"]),
            noc=dict(payload["noc"]),
            d2d_bandwidth_bytes_s=float(payload["d2d_bandwidth_bytes_s"]),
            clock_hz=float(payload["clock_hz"]),
            precision=str(payload["precision"]),
            energy_hardware_facts=dict(payload["energy_hardware_facts"]),
            physical_sources=list(payload.get("physical_sources") or []),
            target_calibrated=bool(payload.get("target_calibrated", False)),
            notes=list(payload.get("notes") or []),
        )
        recorded = payload.get("content_hash")
        if recorded is not None and recorded != snapshot.content_hash:
            raise ContractError(
                "hardware snapshot content hash does not reproduce from its JSON")
        return snapshot


# ---------------------------------------------------------------- mapping --

@dataclass(frozen=True)
class MappingSnapshot:
    """Explicit physical mapping: rank/Core/source/thermals and external sinks.

    ``source_block_weights`` maps each emitting physical source to a thermal
    block distribution with non-negative weights summing to one.  Energy that
    has no on-wafer block (external DRAM) goes to ``external_sinks`` instead of
    being averaged across dies and reported as precise.
    """

    schema_version: int
    mapping_epoch: str
    hardware_hash: str
    source_block_weights: dict[str, dict[str, float]]
    rank_to_die: dict[int, int]
    source_to_destination: dict[str, str]
    external_sinks: tuple[str, ...]
    block_ids: tuple[str, ...]
    notes: tuple[str, ...] = ()

    @staticmethod
    def build(*, mapping_epoch: str, hardware_hash: str,
              source_block_weights: Mapping[str, Mapping[str, float]],
              rank_to_die: Mapping[int, int], block_ids: list[str],
              external_sinks: list[str] | None = None,
              source_to_destination: Mapping[str, str] | None = None,
              notes: list[str] | None = None) -> "MappingSnapshot":
        weights: dict[str, dict[str, float]] = {}
        known_blocks = set(block_ids)
        for source, distribution in source_block_weights.items():
            _nonempty(source, "source_block_weights key")
            row: dict[str, float] = {}
            total = 0.0
            for block, weight in distribution.items():
                if block not in known_blocks:
                    raise ContractError(
                        f"source {source!r} maps to unknown thermal block {block!r}")
                value = _finite(weight, f"weight[{source}][{block}]", nonnegative=True)
                row[block] = value
                total += value
            if not row:
                raise ContractError(f"source {source!r} has an empty block distribution")
            if abs(total - 1.0) > 1e-9:
                raise ContractError(
                    f"source {source!r} weights sum to {total}, expected 1.0")
            weights[source] = row
        sinks = tuple(external_sinks or ())
        destinations = dict(source_to_destination or {})
        for source in weights:
            destinations.setdefault(source, "on_wafer")
        for source, destination in destinations.items():
            if destination not in {"on_wafer", "external"}:
                raise ContractError(
                    f"source {source!r} destination {destination!r} is not "
                    "'on_wafer' or 'external'")
        if external_sinks and not all(isinstance(s, str) and s for s in sinks):
            raise ContractError("external_sinks must be non-empty strings")
        return MappingSnapshot(
            schema_version=SCHEMA_VERSION,
            mapping_epoch=_nonempty(mapping_epoch, "mapping_epoch"),
            hardware_hash=_nonempty(hardware_hash, "hardware_hash"),
            source_block_weights=weights,
            rank_to_die={int(k): int(v) for k, v in rank_to_die.items()},
            source_to_destination=destinations,
            external_sinks=sinks,
            block_ids=tuple(block_ids),
            notes=tuple(notes or ()),
        )

    @property
    def content_hash(self) -> str:
        return content_hash({
            "mapping_epoch": self.mapping_epoch,
            "hardware_hash": self.hardware_hash,
            "source_block_weights": self.source_block_weights,
            "rank_to_die": {str(k): v for k, v in self.rank_to_die.items()},
            "source_to_destination": self.source_to_destination,
            "external_sinks": list(self.external_sinks),
            "block_ids": list(self.block_ids),
        })

    def source_for_rank(self, rank: int, kind: str) -> str:
        if rank not in self.rank_to_die:
            raise ContractError(f"rank {rank} is absent from the physical mapping")
        return f"die{self.rank_to_die[rank]}.{kind}"

    def to_json(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "mapping_epoch": self.mapping_epoch,
            "hardware_hash": self.hardware_hash,
            "content_hash": self.content_hash,
            "source_block_weights": {s: dict(w) for s, w in self.source_block_weights.items()},
            "rank_to_die": {str(k): v for k, v in sorted(self.rank_to_die.items())},
            "source_to_destination": dict(self.source_to_destination),
            "external_sinks": list(self.external_sinks),
            "block_ids": list(self.block_ids),
            "notes": list(self.notes),
        }

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "MappingSnapshot":
        snapshot = MappingSnapshot.build(
            mapping_epoch=str(payload["mapping_epoch"]),
            hardware_hash=str(payload["hardware_hash"]),
            source_block_weights={
                str(source): {str(block): float(weight) for block, weight in weights.items()}
                for source, weights in payload["source_block_weights"].items()},
            rank_to_die={int(rank): int(die) for rank, die in payload["rank_to_die"].items()},
            block_ids=list(payload["block_ids"]),
            external_sinks=list(payload.get("external_sinks") or []),
            source_to_destination=dict(payload.get("source_to_destination") or {}),
            notes=list(payload.get("notes") or []),
        )
        recorded = payload.get("content_hash")
        if recorded is not None and recorded != snapshot.content_hash:
            raise ContractError(
                "mapping snapshot content hash does not reproduce from its JSON")
        return snapshot


# --------------------------------------------------------------- activity --

@dataclass(frozen=True)
class OperatingPoint:
    """The execution run point applied to an activity window.

    The first version supports a declared fixed voltage with frequency scaling
    only; dynamic action counts already reflect the reduced work, so no extra
    f-factor is multiplied onto the dynamic energy.
    """

    frequency_hz: float
    base_frequency_hz: float
    voltage_v: float | None = None

    def __post_init__(self) -> None:
        _finite(self.frequency_hz, "operating_point.frequency_hz", positive=True)
        _finite(self.base_frequency_hz, "operating_point.base_frequency_hz", positive=True)
        if self.voltage_v is not None:
            _finite(self.voltage_v, "operating_point.voltage_v", positive=True)

    @property
    def frequency_fraction(self) -> float:
        return self.frequency_hz / self.base_frequency_hz

    def to_json(self) -> dict[str, Any]:
        return {
            "frequency_hz": self.frequency_hz,
            "base_frequency_hz": self.base_frequency_hz,
            "voltage_v": self.voltage_v,
            "frequency_fraction": self.frequency_fraction,
        }


@dataclass(frozen=True)
class ActivityWindow:
    """Incremental, half-open ``[start_ns, end_ns)`` activity facts.

    ``actions`` carries ``(source_id, action_kind, count)`` triples accumulated
    over the interval; ``dwell`` carries ``(source_id, active_ns, idle_ns)`` so
    idle/leaked background is not mis-accounted as dynamic work.
    """

    run_id: str
    activity_seq: int
    start_ns: int
    end_ns: int
    hardware_hash: str
    mapping_epoch: str
    resource_epoch: int
    actions: tuple[tuple[str, str, float], ...]
    dwell: tuple[tuple[str, int, int], ...] = ()
    operating_point: dict[str, Any] | None = None
    coverage: str = "measured"
    provenance: str = ""

    def __post_init__(self) -> None:
        _integer(self.start_ns, "activity.start_ns", nonnegative=True)
        _integer(self.end_ns, "activity.end_ns", nonnegative=True)
        _integer(self.activity_seq, "activity.activity_seq", nonnegative=True)
        _integer(self.resource_epoch, "activity.resource_epoch", nonnegative=True)
        if self.end_ns <= self.start_ns:
            raise ContractError("activity window requires end_ns > start_ns")
        if self.coverage not in {"measured", "estimated", "assumed"}:
            raise ContractError(
                f"coverage {self.coverage!r} must be measured/estimated/assumed")
        for source, kind, count in self.actions:
            _nonempty(source, "activity.actions source_id")
            if kind not in ACTION_UNITS:
                raise ContractError(
                    f"activity action {kind!r} is not a registered action; "
                    "unmodelled actions must be reported, not assumed zero")
            _finite(count, f"activity[{source},{kind}].count", nonnegative=True)
        for source, active_ns, idle_ns in self.dwell:
            _nonempty(source, "activity.dwell source_id")
            _integer(active_ns, f"activity.dwell[{source}].active_ns", nonnegative=True)
            _integer(idle_ns, f"activity.dwell[{source}].idle_ns", nonnegative=True)

    @property
    def dt_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def dt_s(self) -> float:
        return self.dt_ns / NS_PER_S

    def operating_point_object(self, default_frequency_hz: float | None = None) -> OperatingPoint:
        point = self.operating_point or {}
        base = point.get("base_frequency_hz")
        if base is None:
            base = point.get("frequency_hz")
        if base is None:
            base = default_frequency_hz
        if base is None:
            raise ContractError(
                "activity window has no operating point and no nominal clock was "
                "supplied; the joint host must declare its run point")
        frequency = point.get("frequency_hz")
        if frequency is None:
            frequency = base
        return OperatingPoint(
            frequency_hz=frequency,
            base_frequency_hz=base,
            voltage_v=point.get("voltage_v"),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "activity_seq": self.activity_seq,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "hardware_hash": self.hardware_hash,
            "mapping_epoch": self.mapping_epoch,
            "resource_epoch": self.resource_epoch,
            "actions": [list(item) for item in self.actions],
            "dwell": [list(item) for item in self.dwell],
            "operating_point": self.operating_point,
            "coverage": self.coverage,
            "provenance": self.provenance,
        }

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "ActivityWindow":
        return ActivityWindow(
            run_id=str(payload["run_id"]),
            activity_seq=int(payload["activity_seq"]),
            start_ns=int(payload["start_ns"]),
            end_ns=int(payload["end_ns"]),
            hardware_hash=str(payload["hardware_hash"]),
            mapping_epoch=str(payload["mapping_epoch"]),
            resource_epoch=int(payload["resource_epoch"]),
            actions=tuple(
                (str(source), str(kind), float(count))
                for source, kind, count in payload.get("actions", [])),
            dwell=tuple(
                (str(source), int(active_ns), int(idle_ns))
                for source, active_ns, idle_ns in payload.get("dwell", [])),
            operating_point=payload.get("operating_point"),
            coverage=str(payload.get("coverage", "measured")),
            provenance=str(payload.get("provenance", "")),
        )


# ------------------------------------------------------------ power / heat --

@dataclass(frozen=True)
class PowerWindow:
    """Per-thermal-block power plus the dynamic/background/external ledger.

    ``block_ids`` are the stable thermal block names; ``power_w`` follows that
    same order.  ``dynamic_j``/``background_j`` are per-block ledgers so the
    per-window conservation ``sum(P)*dt + external == total_j`` can be audited.
    """

    run_id: str
    activity_seq: int
    start_ns: int
    end_ns: int
    hardware_hash: str
    mapping_epoch: str
    resource_epoch: int
    block_ids: tuple[str, ...]
    power_w: tuple[float, ...]
    dynamic_j: dict[str, float]
    background_j: dict[str, float]
    external_j: float
    total_energy_j: float
    coverage: str
    diagnostics: dict[str, Any] = field(default_factory=dict)

    @property
    def dt_ns(self) -> int:
        return self.end_ns - self.start_ns

    @property
    def dt_s(self) -> float:
        return self.dt_ns / NS_PER_S

    def power_by_block(self) -> dict[str, float]:
        return dict(zip(self.block_ids, self.power_w))

    def to_json(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "activity_seq": self.activity_seq,
            "start_ns": self.start_ns,
            "end_ns": self.end_ns,
            "hardware_hash": self.hardware_hash,
            "mapping_epoch": self.mapping_epoch,
            "resource_epoch": self.resource_epoch,
            "block_ids": list(self.block_ids),
            "power_w": list(self.power_w),
            "dynamic_j": dict(self.dynamic_j),
            "background_j": dict(self.background_j),
            "external_j": self.external_j,
            "total_energy_j": self.total_energy_j,
            "coverage": self.coverage,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class ThermalFeedback:
    """Immutable thermal result returned to the joint thermal policy."""

    thermal_epoch: int
    effective_ns: int
    committed_s: float
    block_k: dict[str, float]
    coolant_k: dict[str, float]
    die_block_max_k: dict[int, float]
    energy_residual_j: float
    input_j: float
    removed_j: float
    model_id: str
    calibration_id: str
    state_version: int
    domain_status: str
    hardware_hash: str
    mapping_epoch: str
    resource_epoch: int

    def to_json(self) -> dict[str, Any]:
        return {
            "thermal_epoch": self.thermal_epoch,
            "effective_ns": self.effective_ns,
            "committed_s": self.committed_s,
            "block_k": dict(self.block_k),
            "coolant_k": dict(self.coolant_k),
            "die_block_max_k": {str(k): v for k, v in self.die_block_max_k.items()},
            "energy_residual_j": self.energy_residual_j,
            "input_j": self.input_j,
            "removed_j": self.removed_j,
            "model_id": self.model_id,
            "calibration_id": self.calibration_id,
            "state_version": self.state_version,
            "domain_status": self.domain_status,
            "hardware_hash": self.hardware_hash,
            "mapping_epoch": self.mapping_epoch,
            "resource_epoch": self.resource_epoch,
        }


@dataclass(frozen=True)
class ResourceEnvelope:
    """Joint policy output: a capped resource envelope per control domain.

    Produced by the host thermal policy, never by the thermal core.  Recovery is
    also bounded by immutable hardware peaks and other limits.
    """

    policy_version: str
    resource_epoch: int
    effective_from_ns: int
    as_of_ns: int
    domains: tuple[dict[str, Any], ...]

    @staticmethod
    def build(*, policy_version: str, resource_epoch: int,
              effective_from_ns: int, as_of_ns: int,
              domains: list[dict[str, Any]]) -> "ResourceEnvelope":
        _nonempty(policy_version, "envelope.policy_version")
        _integer(resource_epoch, "envelope.resource_epoch", positive=True)
        _integer(effective_from_ns, "envelope.effective_from_ns", nonnegative=True)
        _integer(as_of_ns, "envelope.as_of_ns", nonnegative=True)
        rows = []
        for index, domain in enumerate(domains):
            label = f"domains[{index}]"
            _nonempty(domain.get("domain"), label + ".domain")
            fraction = _finite(domain.get("compute_throughput_fraction"),
                               label + ".compute_throughput_fraction",
                               nonnegative=True)
            if fraction > 1.0:
                raise ContractError(label + ".compute_throughput_fraction exceeds 1.0")
            rows.append({
                "domain": domain["domain"],
                "step": _integer(domain.get("step", 0), label + ".step", nonnegative=True),
                "compute_throughput_fraction": fraction,
                "resource_ceiling": _finite(domain.get("resource_ceiling", fraction),
                                            label + ".resource_ceiling", nonnegative=True),
                "temperature_margin_k": domain.get("temperature_margin_k"),
                "reason": domain.get("reason", ""),
                "applied": bool(domain.get("applied", False)),
            })
        if not rows:
            raise ContractError("resource envelope requires at least one domain")
        return ResourceEnvelope(
            policy_version=policy_version,
            resource_epoch=resource_epoch,
            effective_from_ns=effective_from_ns,
            as_of_ns=as_of_ns,
            domains=tuple(rows),
        )

    def domain_fraction(self, domain: str) -> float | None:
        for row in self.domains:
            if row["domain"] == domain:
                return row["compute_throughput_fraction"]
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "policy_version": self.policy_version,
            "resource_epoch": self.resource_epoch,
            "effective_from_ns": self.effective_from_ns,
            "as_of_ns": self.as_of_ns,
            "domains": [dict(row) for row in self.domains],
        }


def block_kinds(blocks: Any) -> dict[str, str]:
    """Thermal block name -> compute/hbm/router classification from geometry."""
    kinds: dict[str, str] = {}
    for block in blocks:
        if block.kind not in {"compute", "hbm", "router"}:
            raise ContractError(
                f"thermal block {block.name!r} has unsupported kind {block.kind!r}")
        kinds[block.name] = block.kind
    return kinds


def build_mapping(blocks: Any, *, mapping_epoch: str, hardware_hash: str,
                  rank_to_die: Mapping[int, int],
                  external_sources: list[str] | None = None,
                  notes: list[str] | None = None) -> MappingSnapshot:
    """Build an explicit physical mapping from a thermal floorplan.

    Every on-wafer source ``die<d>.<kind>`` is bound to that die's blocks of the
    same kind with equal weights summing to one.  Off-wafer sources (external
    DRAM) are declared as external sinks instead of being spread over dies.
    Rank numbers are *never* used as thermal block indices; ``rank_to_die`` must
    come from the parsed hardware.
    """
    by_die_kind: dict[tuple[int, str], list[str]] = {}
    block_ids: list[str] = []
    for block in blocks:
        if block.kind not in {"compute", "hbm", "router"}:
            raise ContractError(
                f"thermal block {block.name!r} has unsupported kind {block.kind!r}")
        block_ids.append(block.name)
        by_die_kind.setdefault((block.die, block.kind), []).append(block.name)

    sources = sorted({die for _, die in by_die_kind})
    if not sources:
        raise ContractError("thermal floorplan exposes no dies to map")
    weights: dict[str, dict[str, float]] = {}
    for die, kind in sorted(by_die_kind):
        members = by_die_kind[(die, kind)]
        share = 1.0 / len(members)
        weights[f"die{die}.{kind}"] = {name: share for name in members}

    sinks = list(external_sources or ["external.dram"])
    destinations = {source: "on_wafer" for source in weights}
    for sink in sinks:
        destinations[sink] = "external"
    return MappingSnapshot.build(
        mapping_epoch=mapping_epoch,
        hardware_hash=hardware_hash,
        source_block_weights=weights,
        rank_to_die=rank_to_die,
        block_ids=block_ids,
        external_sinks=sinks,
        source_to_destination=destinations,
        notes=notes or ["physical_mapping_from_thermal_floorplan"],
    )


__all__ = [
    "SCHEMA_VERSION", "NS_PER_S", "ACTION_UNITS", "ACTION_DESTINATIONS",
    "ContractError", "content_hash", "HardwareSnapshot", "MappingSnapshot",
    "OperatingPoint", "ActivityWindow", "PowerWindow", "ThermalFeedback",
    "ResourceEnvelope", "block_kinds", "build_mapping",
]
