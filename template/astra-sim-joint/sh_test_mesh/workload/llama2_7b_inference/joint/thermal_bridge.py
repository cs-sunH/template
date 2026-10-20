#!/usr/bin/env python3
"""thermal_bridge.py -- joint <-> embedded thermal-core adapter (plan §3.1).

Ownership split enforced here:

* joint owns requests/phases/KV/migration/progress and exports *facts*:
  parsed hardware, the physical mapping, incremental activity windows, the run
  point.
* the embedded ``wafercosim`` component owns action energy, background power,
  the spatial projection and the solid/coolant thermal state.
* this adapter only joins identities/time/activity: it does **not** copy an
  energy formula, and it does **not** create a second task queue, KV ledger or
  executor.

The bridge is created by the online service and lives for the whole run; it is
never re-created per window and never spawns a CLI subprocess.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from wafercosim.contracts import (
    ActivityWindow,
    ContractError,
    HardwareSnapshot,
    MappingSnapshot,
    ResourceEnvelope,
    build_mapping,
)
from wafercosim.session import ThermalSession
from wafercosim.thermal import LiquidRC

from .thermal_policy import ThermalPolicy, ThermalPolicyConfig
from online.coarse_runtime_client import (
    COMPUTE_GATE_PEAK_RATE,
    RESOURCE_COMPUTE_GATE,
)

THERMAL_MODES = ("off", "shadow", "closed_loop")


def _load_thermal_profile(value: str) -> dict[str, Any]:
    """Load a thermal profile by path or by packaged name (``package:...``)."""
    if value.startswith("package:"):
        from wafercosim.paths import config_path

        parts = value[len("package:"):].split("/")
        return json.loads(config_path(*parts).read_text())
    return json.loads(Path(value).read_text())


def hardware_snapshot_from_facts(facts: Mapping[str, Any]) -> HardwareSnapshot:
    """Build the thermal hardware contract from a resolver facts export.

    The facts are the joint resolver's single hardware source; nothing here
    re-derives or overrides them.  Missing host facts are carried as notes so a
    closed-loop run cannot silently claim target calibration.
    """
    if facts.get("schema_version") != "joint.thermal-hardware-facts.v1":
        raise ContractError(
            f"unexpected hardware facts schema {facts.get('schema_version')!r}")
    dies = facts.get("dies") or []
    if not dies:
        raise ContractError("hardware facts export carries no dies")
    peaks = dict(facts.get("resource_peaks") or {})
    mesh = facts.get("mesh") or {}
    energy_facts = dict(facts.get("energy_hardware_facts") or {})
    notes = list(facts.get("missing_host_facts") or []) + list(facts.get("notes") or [])
    return HardwareSnapshot.build(
        source_path=str(facts.get("source_path", "<unknown>")),
        units={
            "sim_time": "ns", "thermal_time": "s", "frequency": "Hz",
            "energy": "J", "power": "W", "temperature": "K", "geometry": "m",
        },
        mesh_rows=int(mesh.get("rows", 0)),
        mesh_cols=int(mesh.get("columns", 0)),
        dies=[dict(row) for row in dies],
        resource_peaks=peaks,
        sram=dict(facts.get("sram") or {}),
        hbm=dict(facts.get("hbm") or {}),
        dram=dict(facts.get("dram") or {}),
        noc=dict(facts.get("noc") or {}),
        d2d_bandwidth_bytes_s=float(facts.get("d2d_bytes_s", 0.0)),
        clock_hz=float(facts.get("clock_hz", 0.0)),
        precision=str(facts.get("precision", "")),
        energy_hardware_facts=energy_facts,
        target_calibrated=bool(facts.get("target_calibrated", False)),
        notes=notes,
    )


def synthetic_rank_to_die(rank_count: int, thermal_die_count: int) -> dict[int, int]:
    """Fixture rank->thermal-die map (round-robin), clearly named synthetic.

    Production must derive this from explicit architecture data; the joint
    runtime refuses closed_loop when no explicit mapping is supplied.
    """
    if thermal_die_count <= 0:
        raise ContractError("a mapping requires at least one thermal die")
    return {rank: rank % thermal_die_count for rank in range(rank_count)}


def build_fixture_mapping(hardware: HardwareSnapshot, thermal_config: Mapping[str, Any],
                          *, mapping_epoch: str, rank_to_die: Mapping[int, int],
                          notes: list[str] | None = None) -> MappingSnapshot:
    """Bind a thermal floorplan to hardware via an explicit rank->die map.

    The thermal floorplan (blocks) owns the physical geometry; the rank->die map
    is supplied explicitly and is never inferred from the thermal grid.  Fixture
    callers must label their mapping as synthetic.
    """
    thermal = LiquidRC(dict(thermal_config))
    return build_mapping(
        thermal.blocks,
        mapping_epoch=mapping_epoch,
        hardware_hash=hardware.content_hash,
        rank_to_die=rank_to_die,
        notes=notes or ["synthetic fixture mapping; not a target floorplan"],
    )


@dataclass(frozen=True)
class ThermalRunConfig:
    """Resolved thermal run configuration (joint-owned, not a thermal input)."""

    mode: str
    thermal_config_path: str
    energy_table_path: str
    energy_strict: bool
    policy_path: str | None
    mapping_epoch: str
    thermal_step_ns: int
    base_frequency_hz: float
    target_calibrated: bool = False
    # Explicit inputs produced by the trace generator / resolver.  They are the
    # only hardware and mapping sources; nothing is inferred from the thermal
    # grid when they are present.
    hardware_facts_path: str | None = None
    mapping_path: str | None = None
    # Acknowledges that the selected energy table carries assumed coefficients.
    # This is a table-status disclosure, NOT a way to hide a target-hardware
    # domain mismatch; production runs must state it explicitly.
    accept_assumed_energy: bool = False

    def __post_init__(self) -> None:
        if self.mode not in THERMAL_MODES:
            raise ContractError(
                f"thermal mode must be one of {THERMAL_MODES}, got {self.mode!r}")
        if self.thermal_step_ns <= 0:
            raise ContractError("thermal_step_ns must be positive")

    @staticmethod
    def from_json(payload: Mapping[str, Any]) -> "ThermalRunConfig":
        return ThermalRunConfig(
            mode=str(payload["mode"]),
            thermal_config_path=str(payload["thermal_config"]),
            energy_table_path=str(payload["energy_table"]),
            energy_strict=bool(payload.get("energy_strict", True)),
            policy_path=payload.get("policy"),
            mapping_epoch=str(payload.get("mapping_epoch", "joint-mapping-v1")),
            thermal_step_ns=int(payload["thermal_step_ns"]),
            base_frequency_hz=float(payload["base_frequency_hz"]),
            target_calibrated=bool(payload.get("target_calibrated", False)),
            hardware_facts_path=payload.get("hardware_facts"),
            mapping_path=payload.get("mapping"),
            accept_assumed_energy=bool(payload.get("accept_assumed_energy", False)),
        )


def load_thermal_run_config(path: Path) -> ThermalRunConfig:
    return ThermalRunConfig.from_json(json.loads(Path(path).read_text()))


def load_hardware_and_mapping(config: ThermalRunConfig) -> tuple[HardwareSnapshot, MappingSnapshot]:
    """Load the explicit hardware/mapping inputs a thermal run depends on.

    Both must be produced by the resolver/trace generator; the thermal profile
    is never used to invent either.  Absent inputs are a hard error so a run
    cannot silently fall back to a synthetic mapping.
    """
    if not config.hardware_facts_path:
        raise ContractError(
            "thermal run configuration has no hardware_facts reference; the "
            "resolver export (resolved_hardware.json) is required")
    if not config.mapping_path:
        raise ContractError(
            "thermal run configuration has no physical mapping reference; "
            "closed_loop cannot infer rank->die from the thermal grid")
    hardware = HardwareSnapshot.from_json(
        json.loads(Path(config.hardware_facts_path).read_text()))
    mapping = MappingSnapshot.from_json(
        json.loads(Path(config.mapping_path).read_text()))
    if mapping.hardware_hash != hardware.content_hash:
        raise ContractError(
            "physical mapping was built against a different hardware snapshot")
    return hardware, mapping


# ------------------------------------------------ real activity from counters --

# Cumulative served counters -> thermal actions.  The counters are lifetime
# totals (guarded by the runtime's ``accounting_history_complete`` flag), so an
# incremental window is the difference between two snapshots.  Completed work is
# never lost this way, which is exactly why a "currently active objects" diff
# would be wrong.
_COUNTER_ACTION_SOURCES = {
    "int4_ops": "pe",
    "vec_eqops": "vec",
}


@dataclass(frozen=True)
class SamplerState:
    """Cumulative counters captured at the last committed window boundary."""

    time_ns: int
    comp_int4_ops: float
    comp_vec_eqops: float
    hbm_read_bytes: float
    hbm_write_bytes: float
    compute_busy_ns: float

    @staticmethod
    def from_counters(counters: Mapping[str, Any]) -> "SamplerState":
        if not counters.get("accounting_history_complete", False):
            raise ContractError(
                "coarse runtime accounting history is incomplete; served totals "
                "are only a lower bound and must not be differenced as activity")
        categories = counters.get("served_by_category") or {}
        comp = categories.get("comp")
        if not isinstance(comp, Mapping):
            raise ContractError("coarse counters carry no 'comp' served category")
        rows = counters.get("hbm_served_bytes_by_row_direction") or []
        read_bytes = sum(float(row[0]) for row in rows)
        write_bytes = sum(float(row[1]) for row in rows)
        return SamplerState(
            time_ns=int(counters.get("now_ns", 0)),
            comp_int4_ops=float(comp.get("int4_ops", 0.0)),
            comp_vec_eqops=float(comp.get("vec_eqops", 0.0)),
            hbm_read_bytes=read_bytes,
            hbm_write_bytes=write_bytes,
            compute_busy_ns=float(counters.get("compute_busy_ns", 0.0)),
        )


class CoarseActivitySampler:
    """Turn consecutive coarse-runtime counter snapshots into ActivityWindows.

    ``rank_to_source`` maps a rank to the thermal source identity for its compute
    and HBM resources; the caller supplies it from the physical mapping so this
    class never invents identities.
    """

    def __init__(self, *, run_id: str, hardware_hash: str, mapping_epoch: str,
                 rank_to_source: Mapping[int, str]):
        self.run_id = run_id
        self.hardware_hash = hardware_hash
        self.mapping_epoch = mapping_epoch
        self.rank_to_source = dict(rank_to_source)
        self._previous: dict[int, SamplerState] = {}
        self.activity_seq = 0

    def _source(self, rank: int, kind: str) -> str:
        base = self.rank_to_source.get(rank)
        if base is None:
            raise ContractError(f"rank {rank} has no mapped thermal source")
        return f"{base}.{kind}"

    def window(self, rank: int, previous: SamplerState | None,
               current: SamplerState, *, resource_epoch: int,
               activity_seq: int | None = None) -> ActivityWindow:
        if previous is not None and current.time_ns <= previous.time_ns:
            raise ContractError(
                "counter window end must be after its start "
                f"({previous.time_ns} -> {current.time_ns} ns)")
        if previous is None:
            start_ns, base = 0, SamplerState(0, 0.0, 0.0, 0.0, 0.0, 0.0)
        else:
            start_ns, base = previous.time_ns, previous
        delta = {
            "pe": current.comp_int4_ops - base.comp_int4_ops,
            "vec": current.comp_vec_eqops - base.comp_vec_eqops,
        }
        if any(value < 0 for value in delta.values()):
            raise ContractError(
                "served counters went backwards; the runtime ledger is not "
                "monotonic and cannot be differenced")
        actions = [
            (self._source(rank, "compute"), kind, value)
            for kind, value in delta.items() if value > 0.0
        ]
        for kind, read in (("hbm_read", current.hbm_read_bytes - base.hbm_read_bytes),
                           ("hbm_write", current.hbm_write_bytes - base.hbm_write_bytes)):
            if read < 0:
                raise ContractError("served HBM counters went backwards")
            if read > 0.0:
                actions.append((self._source(rank, "hbm"), kind, read))
        window_ns = current.time_ns - start_ns
        busy_ns = min(max(current.compute_busy_ns - base.compute_busy_ns, 0.0), float(window_ns))
        dwell = ((self._source(rank, "compute"), int(busy_ns), int(window_ns - busy_ns)),)
        self._previous[rank] = current
        seq = self.activity_seq if activity_seq is None else activity_seq
        self.activity_seq = seq + 1
        return ActivityWindow(
            run_id=self.run_id, activity_seq=seq,
            start_ns=start_ns, end_ns=current.time_ns,
            hardware_hash=self.hardware_hash, mapping_epoch=self.mapping_epoch,
            resource_epoch=resource_epoch, actions=tuple(actions), dwell=dwell,
            coverage="measured",
            provenance="coarse_runtime_served_counters_delta",
        )


# ------------------------------------------------------- installing the cap --

class CoarseCapacityInstaller:
    """Install a ResourceEnvelope into the live coarse runtime.

    The envelope's per-domain ``compute_throughput_fraction`` becomes an
    *absolute* COMPUTE_GATE rate (peak 1.0 second_of_service/s) installed at the
    envelope's ``effective_from_ns``.  The exact status is reported back so the
    caller only ever treats an acknowledged install as the live resource epoch.
    """

    def __init__(self, runtime: Any, *, domain_to_rank: Mapping[str, int]):
        self.runtime = runtime
        self.domain_to_rank = dict(domain_to_rank)

    def install(self, envelope: ResourceEnvelope) -> list[dict[str, Any]]:
        applied: list[dict[str, Any]] = []
        for row in envelope.domains:
            domain = row["domain"]
            if domain not in self.domain_to_rank:
                raise ContractError(
                    f"control domain {domain!r} has no mapped runtime rank")
            rank = self.domain_to_rank[domain]
            if rank != getattr(self.runtime, "rank", rank) and hasattr(self.runtime, "for_rank"):
                target = self.runtime.for_rank(rank)
            else:
                target = self.runtime
            fraction = float(row["compute_throughput_fraction"])
            rate = COMPUTE_GATE_PEAK_RATE * fraction
            target.update_capacity(RESOURCE_COMPUTE_GATE, rate, envelope.effective_from_ns)
            applied.append({
                "domain": domain,
                "rank": rank,
                "resource": "COMPUTE_GATE",
                "resource_id": RESOURCE_COMPUTE_GATE,
                "rate": rate,
                "effective_from_ns": envelope.effective_from_ns,
                "resource_epoch": envelope.resource_epoch,
            })
        return applied


class DelegatedCapacityInstaller:
    """Closed-loop install delegated to the C++ process (no local runtime).

    The online service and the simulator run in separate processes: Python owns
    the ``thermal_activity`` -> ``thermal_cap`` translation and returns the cap on
    the response, while the C++ coarse runtime (armed by ``SH_THERMAL_EPOCH_NS``)
    consumes that cap and performs the real ``update_capacity``.  This installer
    therefore performs no capacity update; it exists so the bridge's
    ``closed_loop`` branch can mark the envelope applied and advance the resource
    epoch without a second, Python-side install (cap installed twice would be a
    bug, not a feature).
    """

    def install(self, envelope: ResourceEnvelope) -> list[dict[str, Any]]:
        return []


class ThermalBridge:
    """One long-lived thermal session + policy for a joint run."""

    def __init__(self, *, run_id: str, config: ThermalRunConfig,
                 hardware: HardwareSnapshot, mapping: MappingSnapshot,
                 policy: ThermalPolicy, audit_dir: Path | None = None,
                 installer: "CoarseCapacityInstaller | None" = None):
        self.run_id = run_id
        self.config = config
        self.hardware = hardware
        self.mapping = mapping
        self.policy = policy
        self.installer = installer
        self.audit_dir = Path(audit_dir) if audit_dir else None
        self._resource_epoch = 1
        self._latest_envelope: ResourceEnvelope | None = None
        self._rows: dict[str, list[dict[str, Any]]] = {
            "activity_windows": [], "power_windows": [],
            "thermal_feedback": [], "resource_envelopes": [],
            "thermal_decisions": [],
        }
        if config.mode == "off":
            self.session: ThermalSession | None = None
            return
        if config.mode == "closed_loop" and not mapping.rank_to_die:
            raise ContractError(
                "closed_loop requires an explicit rank->physical mapping; the "
                "thermal profile cannot infer execution topology")
        self.session = self._build_session(run_id)

    # ------------------------------------------------------------- assembly --

    @staticmethod
    def build(run_id: str, config_path: Path, hardware: HardwareSnapshot,
              mapping: MappingSnapshot, *, audit_dir: Path | None = None) -> "ThermalBridge":
        config = load_thermal_run_config(config_path)
        policy = ThermalPolicy(load_policy(config))
        return ThermalBridge(
            run_id=run_id, config=config, hardware=hardware, mapping=mapping,
            policy=policy, audit_dir=audit_dir)

    def _build_session(self, run_id: str) -> ThermalSession:
        from wafercosim.action_energy import EnergyModel

        thermal_config = _load_thermal_profile(self.config.thermal_config_path)
        energy_config = {
            "calibration": {
                "energy_table": self.config.energy_table_path,
                "strict": self.config.energy_strict,
                "accept_assumed_energy": self.config.accept_assumed_energy,
                "memory_organization_sensitivity_accept": {"hbm": True, "dram": True},
            },
            "thermal": {"initial_k": thermal_config["thermal"]["initial_k"]},
        }
        energy = EnergyModel(
            energy_config, hardware_facts=self.hardware.energy_hardware_facts)
        return ThermalSession(
            run_id=run_id, thermal=LiquidRC(thermal_config), energy=energy,
            hardware=self.hardware, mapping=self.mapping)

    # ------------------------------------------------------------- per-run --

    @property
    def enabled(self) -> bool:
        return self.session is not None

    @property
    def resource_epoch(self) -> int:
        return self._resource_epoch

    def latest_envelope(self) -> ResourceEnvelope | None:
        envelope = self._latest_envelope
        if envelope is None:
            return None
        # A later-arriving decision must read the last committed temperature and
        # the still-valid cap together, with an explicit as_of_ns.
        return envelope

    def ingest(self, activity: ActivityWindow) -> dict[str, Any] | None:
        """Consume one committed activity window and return feedback + proposed cap.

        Returns ``None`` in ``off`` mode.  In ``shadow`` mode the envelope is
        produced but never installed (the executor is untouched).  In
        ``closed_loop`` mode the envelope is handed to the attached
        :class:`CoarseCapacityInstaller`; the domain rows are only marked
        ``applied`` after the runtime C ABI acknowledges the capacity update, so
        an unacknowledged cap is never treated as the live resource epoch.
        """
        if self.session is None:
            return None
        power = self.session.estimate_power(activity)
        feedback = self.session.advance_committed(power)
        envelope = self.policy.compute_envelope(
            feedback,
            resource_epoch=self._resource_epoch,
            effective_from_ns=feedback.effective_ns,
            as_of_ns=feedback.effective_ns,
        )
        installed: list[dict[str, Any]] = []
        if self.config.mode == "closed_loop":
            if self.installer is None:
                raise ContractError(
                    "closed_loop requires a runtime capacity installer; the cap "
                    "must reach execution, not just be computed")
            installed = self.installer.install(envelope)
            envelope = ResourceEnvelope.build(
                policy_version=envelope.policy_version,
                resource_epoch=envelope.resource_epoch,
                effective_from_ns=envelope.effective_from_ns,
                as_of_ns=envelope.as_of_ns,
                domains=[dict(row, applied=True) for row in envelope.domains],
            )
            self._resource_epoch += 1
        self._latest_envelope = envelope
        self._rows["activity_windows"].append(activity.to_json())
        self._rows["power_windows"].append(power.to_json())
        self._rows["thermal_feedback"].append(feedback.to_json())
        self._rows["resource_envelopes"].append(envelope.to_json())
        if installed:
            self.record_decision({
                "as_of_ns": feedback.effective_ns,
                "resource_epoch": envelope.resource_epoch,
                "thermal_state_version": feedback.state_version,
                "installed": installed,
            })
        return {"power": power, "feedback": feedback, "envelope": envelope,
                "installed": installed}

    def record_decision(self, row: Mapping[str, Any]) -> None:
        self._rows["thermal_decisions"].append(dict(row))

    # ---------------------------------------------------------------- audit --

    def manifest(self) -> dict[str, Any]:
        payload = {
            "mode": self.config.mode,
            "run_id": self.run_id,
            "mapping_epoch": self.mapping.mapping_epoch,
            "mapping_hash": self.mapping.content_hash,
            "hardware_hash": self.hardware.content_hash,
            "hardware_target_calibrated": self.hardware.target_calibrated,
            "thermal_step_ns": self.config.thermal_step_ns,
            "thermal_profile": self.config.thermal_config_path,
            "energy_table": self.config.energy_table_path,
            "energy_strict": self.config.energy_strict,
            "accept_assumed_energy": self.config.accept_assumed_energy,
            "base_frequency_hz": self.config.base_frequency_hz,
            "policy": self.policy.manifest(),
            "resource_epoch": self._resource_epoch,
        }
        if self.session is not None:
            payload["session"] = self.session.manifest()
        return payload

    def audit_rows(self, name: str) -> list[dict[str, Any]]:
        if name in self._rows:
            rows = list(self._rows[name])
        elif self.session is not None:
            rows = self.session.audit_rows(name)
        else:
            raise ContractError(f"unknown audit stream {name!r}")
        return rows

    def write_audit(self, audit_dir: Path | None = None) -> list[Path]:
        directory = Path(audit_dir or self.audit_dir or ".")
        directory.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name in ("activity_windows", "power_windows", "thermal_feedback",
                     "resource_envelopes", "thermal_decisions"):
            rows = self.audit_rows(name)
            path = directory / f"{name}.jsonl"
            with path.open("w", encoding="utf-8") as sink:
                for row in rows:
                    sink.write(json.dumps(row, sort_keys=True) + "\n")
            written.append(path)
        (directory / "resolved_hardware.json").write_text(
            json.dumps(self.hardware.to_json(), indent=1, sort_keys=True) + "\n")
        (directory / "physical_mapping.json").write_text(
            json.dumps(self.mapping.to_json(), indent=1, sort_keys=True) + "\n")
        (directory / "thermal_manifest.json").write_text(
            json.dumps(self.manifest(), indent=1, sort_keys=True) + "\n")
        written.extend([directory / "resolved_hardware.json",
                        directory / "physical_mapping.json",
                        directory / "thermal_manifest.json"])
        return written

    def close(self) -> None:
        if self.session is not None:
            self.session.close()


def load_policy(config: ThermalRunConfig) -> ThermalPolicyConfig:
    """Load the joint-owned policy; thresholds are never implicit.

    Recovery hysteresis, the frequency ladder, minimum dwell and the sampling
    step must come from a declared policy file with a recorded source, so the
    reference 300 K initial temperature or any convenient threshold is never
    presented as a device specification.
    """
    if not config.policy_path:
        raise ContractError(
            "a thermal policy reference is required (no implicit thresholds are "
            "assumed); provide --policy with declared sources")
    return ThermalPolicyConfig.from_json(
        json.loads(Path(config.policy_path).read_text()))


__all__ = [
    "THERMAL_MODES", "ThermalRunConfig", "ThermalBridge",
    "DelegatedCapacityInstaller",
    "hardware_snapshot_from_facts", "synthetic_rank_to_die",
    "load_thermal_run_config", "load_policy",
]
