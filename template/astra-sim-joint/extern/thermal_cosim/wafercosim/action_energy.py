"""Temperature-aware runtime lookup for an Accelergy-generated action ERT.

Accelergy creates the offline primitive action table. This small online
adapter applies the table to host-supplied action counts, keeps background W
separate from dynamic J, and provides provenance and strict domain checks.
"""

from __future__ import annotations

import json
import hashlib
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent
REQUIRED_ACTIONS = {
    "pe": "op",
    "vec": "op",
    "control": "action",
    "sram_read": "byte",
    "sram_write": "byte",
    "hbm": "byte",
    "dram": "byte",
    "noc": "byte",
    "d2d": "byte",
    "dram_io": "byte",
}
DIRECTIONAL_ACTIONS = {
    "hbm_read": {"resource": "hbm", "operation": "read", "ert_component": "hbm_read_byte"},
    "hbm_write": {"resource": "hbm", "operation": "write", "ert_component": "hbm_write_byte"},
    "dram_read": {"resource": "dram", "operation": "read", "ert_component": "dram_read_byte"},
    "dram_write": {"resource": "dram", "operation": "write", "ert_component": "dram_write_byte"},
}
REQUIRED_BACKGROUND = {
    "compute_idle",
    "compute_leak",
    "sram_leak",
    "hbm_idle",
    "router_idle",
    "dram_background",
}
THERMAL_DESTINATIONS = {
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
    # The current RC model has no separate controller/PHY block. Keep this
    # on-wafer IO heat in its router/IO lump and identify that projection in
    # metadata; external DRAM cell/background energy remains outside the wafer.
    "dram_io": "router",
}


class EnergyCalibrationError(ValueError):
    """The requested action, background, temperature, or hardware is out of domain."""


def _resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    for candidate in (Path.cwd() / path, PROJECT_ROOT / path):
        if candidate.is_file():
            return candidate.resolve()
    return (PROJECT_ROOT / path).resolve()


def _finite_nonnegative(value: Any) -> bool:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and number >= 0


def _finite_positive(value: Any) -> bool:
    return _finite_nonnegative(value) and float(value) > 0


def _positive_integer(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnergyCalibrationError(f"{name} must be a positive integer")
    if isinstance(value, int):
        if value <= 0:
            raise EnergyCalibrationError(f"{name} must be a positive integer")
        return value
    if not math.isfinite(value) or value <= 0 or not value.is_integer():
        raise EnergyCalibrationError(f"{name} must be a positive integer")
    return int(value)


def _positive_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EnergyCalibrationError(f"{name} must be finite and positive")
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise EnergyCalibrationError(f"{name} must be finite and positive")
    return number


def _nonempty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise EnergyCalibrationError(f"{name} must be a non-empty string")
    return value


def _sha256(path: Path | None) -> str | None:
    if path is None or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class EnergyModel:
    """Evaluate dynamic action J and background W using explicit host facts.

    Expected config section::

        "calibration": {
          "energy_table": "configs/energy/research-v1.json",
          "strict": true,
          "accept_assumed_energy": true,
          "memory_organization_sensitivity_accept": {"hbm": true, "dram": true}
        }

    A configured table requires ``hardware_facts`` as an explicit keyword-only
    argument. It contains normalized host facts, including the NoC dimensions;
    thermal geometry is never used as a hardware source. SRAM macro count and
    capacity are supplied by the host and checked against the CACTI macro
    organization in the table. The table's macro population remains reference
    provenance only.

    `energy(kind, count, temperature_k)` returns Joules. `background(kind,
    count, temperature_k)` returns Watts. For `sram_leak`, count is a number of
    dies; macro multiplicity is applied from the validated host SRAM facts.
    """

    def __init__(
        self,
        config: dict[str, Any],
        *,
        hardware_facts: dict[str, Any] | None = None,
    ):
        self._reject_legacy_configuration_sources(config)
        self.config = config
        calibration = config.get("calibration", {})
        table_value = calibration.get("energy_table")
        self.enabled = bool(table_value)
        self.strict = bool(calibration.get("strict", True))
        self.path: Path | None = None
        self.table: dict[str, Any] = {}
        self.issues: list[str] = []
        self.accepted_memory_organization_mismatches: list[dict[str, Any]] = []
        self._last_downgrade: str | None = None
        self._noc_topology_key: str | None = None
        self.hardware_facts: dict[str, Any] | None = None
        self.hardware_facts_sha256: str | None = None
        self._sram_macro_count_per_die: int | None = None

        if not table_value:
            self.strict = False
            self.issues.append("No energy_table configured; use configured coefficient fallback.")
            return
        if hardware_facts is None:
            raise EnergyCalibrationError(
                "hardware_facts is required when calibration.energy_table is configured"
            )
        self.hardware_facts = self._normalize_hardware_facts(hardware_facts)
        facts_json = json.dumps(
            self.hardware_facts, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        self.hardware_facts_sha256 = hashlib.sha256(facts_json.encode("utf-8")).hexdigest()
        self.path = _resolve_path(table_value)
        if not self.path.is_file():
            raise FileNotFoundError(f"energy calibration table not found: {self.path}")
        self.table = json.loads(self.path.read_text())
        if self.table.get("schema_version") != "wafer-thermal-cosim.energy-action-table.v1":
            raise EnergyCalibrationError(
                f"unsupported energy table schema: {self.table.get('schema_version')!r}"
            )
        self.strict = bool(calibration.get("strict", self.table.get("strict_default", True)))
        self._validate_actions()
        self._validate_background()
        self._validate_hardware_domain()
        self._validate_assumption_acceptance(calibration)
        if self.issues and self.strict:
            raise EnergyCalibrationError(
                "energy calibration strict domain mismatch: " + "; ".join(self.issues)
            )

    @staticmethod
    def _reject_legacy_configuration_sources(config: dict[str, Any]) -> None:
        if not isinstance(config, dict):
            raise EnergyCalibrationError("energy model config must be an object")
        retired_top_level = ("hardware", "model", "workload", "simulation", "compute", "mapping", "performance")
        present = [key for key in retired_top_level if key in config]
        if present:
            raise EnergyCalibrationError(
                "legacy execution configuration is not accepted by EnergyModel: "
                + ", ".join(f"config.{key}" for key in present)
                + "; pass actual hardware through hardware_facts"
            )
        calibration = config.get("calibration", {})
        if not isinstance(calibration, dict):
            raise EnergyCalibrationError("config.calibration must be an object")
        retired_calibration = ("hardware", "service_table")
        present = [key for key in retired_calibration if key in calibration]
        if present:
            raise EnergyCalibrationError(
                "legacy calibration inputs are not accepted by EnergyModel: "
                + ", ".join(f"calibration.{key}" for key in present)
                + "; use hardware_facts and the configured energy table"
            )

    @staticmethod
    def _normalize_hardware_facts(facts: dict[str, Any]) -> dict[str, Any]:
        """Validate and canonicalize the host-fact contract before calibration checks."""
        if not isinstance(facts, dict):
            raise EnergyCalibrationError("hardware_facts must be an object")

        normalized: dict[str, Any] = {
            key: _positive_integer(facts.get(key), f"hardware_facts.{key}")
            for key in ("sram_capacity_bytes_per_die", "hbm_capacity_bytes_per_die", "dram_capacity_bytes")
        }
        normalized.update({
            key: _positive_number(facts.get(key), f"hardware_facts.{key}")
            for key in ("pe_ops_s_per_die", "vec_ops_s_per_die")
        })

        def object_at(key: str, required: tuple[str, ...]) -> dict[str, Any]:
            value = facts.get(key)
            if not isinstance(value, dict):
                raise EnergyCalibrationError(f"hardware_facts.{key} must be an object")
            missing = [field for field in required if field not in value]
            if missing:
                raise EnergyCalibrationError(
                    f"hardware_facts.{key} is missing required field(s): {', '.join(missing)}"
                )
            return value

        compute = object_at("compute", ("weight_bytes", "activation_bytes"))
        normalized["compute"] = {
            key: _positive_number(compute[key], f"hardware_facts.compute.{key}")
            for key in ("weight_bytes", "activation_bytes")
        }

        noc_fields = (
            "nx", "ny", "flit_width_bits", "num_vcs", "packet_size_flits", "router", "routing"
        )
        noc = object_at("noc", noc_fields)
        normalized["noc"] = {
            key: (
                _positive_integer(noc[key], f"hardware_facts.noc.{key}")
                if key in {"nx", "ny", "flit_width_bits", "num_vcs", "packet_size_flits"}
                else _nonempty_string(noc[key], f"hardware_facts.noc.{key}")
            )
            for key in noc_fields
        }

        for resource in ("hbm", "dram"):
            facts_for_resource = object_at(resource, ("protocol", "config"))
            normalized[resource] = {
                key: _nonempty_string(
                    facts_for_resource[key], f"hardware_facts.{resource}.{key}"
                )
                for key in ("protocol", "config")
            }

        sram = object_at(
            "sram",
            ("macro_capacity_bytes", "macro_count_per_die", "word_width_bits", "access_bytes"),
        )
        normalized["sram"] = {
            key: _positive_integer(sram[key], f"hardware_facts.sram.{key}")
            for key in ("macro_capacity_bytes", "macro_count_per_die", "word_width_bits", "access_bytes")
        }
        actual_capacity = (
            normalized["sram"]["macro_capacity_bytes"]
            * normalized["sram"]["macro_count_per_die"]
        )
        if normalized["sram_capacity_bytes_per_die"] != actual_capacity:
            raise EnergyCalibrationError(
                "hardware_facts.sram_capacity_bytes_per_die must equal "
                "hardware_facts.sram.macro_capacity_bytes * macro_count_per_die"
            )
        return normalized

    def _validate_actions(self) -> None:
        actions = self.table.get("actions", {})
        for kind, expected_unit in REQUIRED_ACTIONS.items():
            row = actions.get(kind)
            if row is None:
                self.issues.append(f"required action {kind!r} is missing")
                continue
            if row.get("unit") != expected_unit:
                self.issues.append(
                    f"action {kind!r} unit is {row.get('unit')!r}; expected {expected_unit!r}"
                )
            if not _finite_positive(row.get("joules_per_unit")):
                self.issues.append(f"action {kind!r} Joules/unit is missing or nonpositive")
            if not row.get("source"):
                self.issues.append(f"action {kind!r} has no source/provenance")
        for kind, descriptor in DIRECTIONAL_ACTIONS.items():
            base = actions.get(descriptor["resource"])
            ert = self.table.get("accelergy_ert_components", {}).get(descriptor["ert_component"])
            if base is None or base.get("unit") != "byte":
                self.issues.append(f"directional action {kind!r} has no byte-based generic source row")
            if ert is None:
                self.issues.append(
                    f"directional action {kind!r} is missing Accelergy component {descriptor['ert_component']!r}"
                )
                continue
            if ert.get("action") != "action":
                self.issues.append(f"directional action {kind!r} has an unexpected Accelergy action type")
            if not _finite_positive(ert.get("energy_pj_from_ert")):
                self.issues.append(f"directional action {kind!r} has invalid Accelergy pJ/action")
            expected_name = f".{descriptor['ert_component']}"
            if not str(ert.get("component", "")).endswith(expected_name):
                self.issues.append(
                    f"directional action {kind!r} component identity does not match {descriptor['ert_component']!r}"
                )

    def _directional_action_row(self, kind: str) -> dict[str, Any] | None:
        descriptor = DIRECTIONAL_ACTIONS.get(kind)
        if descriptor is None:
            return None
        source_row = self.table.get("actions", {}).get(descriptor["resource"])
        ert = self.table.get("accelergy_ert_components", {}).get(descriptor["ert_component"])
        if source_row is None or ert is None or not _finite_positive(ert.get("energy_pj_from_ert")):
            return None
        source = dict(source_row.get("source", {}))
        source.pop("read_write_byte_mix", None)
        source["direction"] = descriptor["operation"]
        source["ert_component"] = ert.get("component")
        source["ert_action"] = ert.get("action")
        source["ert_energy_unit"] = "pJ per Accelergy action (one byte payload)"
        background_owner = "hbm_idle" if descriptor["resource"] == "hbm" else "dram_background"
        source["energy_scope"] = (
            f"ACT + {descriptor['operation'].upper()} command energy per target byte; "
            f"REF and rank background remain in {background_owner}"
        )
        return {
            "unit": source_row.get("unit"),
            "joules_per_unit": float(ert["energy_pj_from_ert"]) * 1e-12,
            "temperature_domain_k": list(source_row.get("temperature_domain_k", [300, 400])),
            "temperature_dependence": source_row.get("temperature_dependence"),
            "ert_component": ert.get("component"),
            "ert_action": ert.get("action"),
            "ert_energy_pj": float(ert["energy_pj_from_ert"]),
            "source": source,
        }

    def _validate_background(self) -> None:
        backgrounds = self.table.get("background", {})
        for kind in sorted(REQUIRED_BACKGROUND):
            row = backgrounds.get(kind)
            if row is None:
                self.issues.append(f"required background {kind!r} is missing")
                continue
            if kind == "compute_leak":
                for key in ("watts_per_die_at_reference", "reference_temperature_k", "beta_per_k"):
                    if not _finite_nonnegative(row.get(key)):
                        self.issues.append(f"background {kind!r} has invalid {key}")
            elif kind == "sram_leak":
                if not _finite_positive(row.get("macro_count_per_die")):
                    self.issues.append("sram_leak macro_count_per_die is missing or nonpositive")
                if not _finite_positive(row.get("macro_capacity_bytes")):
                    self.issues.append("sram_leak macro_capacity_bytes is missing or nonpositive")
                rows = row.get("watts_per_macro_by_temperature_k", [])
                if len(rows) < 2 or any(
                    not _finite_nonnegative(item.get("watts_per_macro"))
                    or not _finite_positive(item.get("temperature_k"))
                    for item in rows
                ):
                    self.issues.append("sram_leak temperature table must contain finite CACTI rows")
            elif not _finite_nonnegative(row.get("watts_per_unit")):
                self.issues.append(f"background {kind!r} W/unit is missing or invalid")
            if not row.get("source"):
                self.issues.append(f"background {kind!r} has no source/provenance")

    def _assumption_ids(self) -> list[str]:
        identifiers = []
        for kind, row in self.table.get("actions", {}).items():
            status = str(row.get("status", ""))
            if status.startswith("assumed"):
                identifiers.append(f"action:{kind}")
        for kind, row in self.table.get("background", {}).items():
            source = row.get("source", {})
            status = str(source.get("status", ""))
            if status.startswith("assumed"):
                identifiers.append(f"background:{kind}")
        return sorted(identifiers)

    def _validate_assumption_acceptance(self, calibration: dict[str, Any]) -> None:
        assumptions = self._assumption_ids()
        if assumptions and not calibration.get("accept_assumed_energy", False):
            self.issues.append(
                "energy table contains assumed coefficients; set calibration.accept_assumed_energy=true "
                "to explicitly acknowledge: " + ", ".join(assumptions)
            )

    def _validate_hardware_domain(self) -> None:
        domain = self.table.get("hardware_domain", {})
        assert self.hardware_facts is not None
        facts = self.hardware_facts

        # SRAM table organization describes the CACTI macro model. Its macro
        # population/capacity are reference provenance; actual scale comes
        # only from the host's explicit, internally consistent facts.
        sram_row = self.table.get("background", {}).get("sram_leak", {})
        source = sram_row.get("source", {})
        reference_macro_capacity = sram_row.get("macro_capacity_bytes")
        if reference_macro_capacity is None:
            reference_macro_capacity = source.get("macro_bytes")
        if not _finite_positive(reference_macro_capacity):
            raise EnergyCalibrationError(
                "energy table SRAM leakage row has no valid reference macro capacity"
            )
        if int(reference_macro_capacity) != facts["sram"]["macro_capacity_bytes"]:
            raise EnergyCalibrationError(
                "hardware_facts.sram.macro_capacity_bytes does not match the "
                "energy-table CACTI macro organization"
            )
        for table_key, fact_key in (
            ("word_bits", "word_width_bits"), ("access_bytes", "access_bytes")
        ):
            reference_value = source.get(table_key)
            if reference_value is not None and int(reference_value) != facts["sram"][fact_key]:
                raise EnergyCalibrationError(
                    f"hardware_facts.sram.{fact_key} does not match the "
                    f"energy-table CACTI organization ({table_key}={reference_value!r})"
                )
        self._sram_macro_count_per_die = facts["sram"]["macro_count_per_die"]

        reference_macro_count = sram_row.get("macro_count_per_die")
        reference_capacity = domain.get("sram_capacity_bytes_per_die")
        if (
            reference_capacity is not None
            and _finite_positive(reference_macro_count)
            and not math.isclose(
                float(reference_capacity),
                float(reference_macro_capacity) * float(reference_macro_count),
                rel_tol=0.0,
                abs_tol=0.0,
            )
        ):
            self.issues.append(
                "energy table SRAM reference capacity does not match its reference macro population"
            )

        expected_compute = domain.get("compute", {})
        scaling = self.config.get("calibration", {}).get("resource_scaling", {})
        if not isinstance(scaling, dict):
            self.issues.append("calibration.resource_scaling must be an object")
            scaling = {}
        unknown_scales = set(scaling) - {
            "compute", "hbm", "dram", "hbm_capacity", "dram_capacity"
        }
        if unknown_scales:
            self.issues.append(
                "unknown calibration.resource_scaling keys: " + ", ".join(sorted(unknown_scales))
            )
        compute_scale = scaling.get("compute", 1.0)
        if isinstance(compute_scale, bool) or not _finite_positive(compute_scale):
            self.issues.append("calibration.resource_scaling.compute must be finite and positive")
            compute_scale = 1.0
        for key in ("pe_ops_s_per_die", "vec_ops_s_per_die"):
            if key in expected_compute:
                expected_rate = float(expected_compute[key]) * float(compute_scale)
                actual_rate = facts.get(key)
                try:
                    matches = math.isclose(float(actual_rate), expected_rate, rel_tol=1e-12, abs_tol=0.0)
                except (TypeError, ValueError):
                    matches = False
                if not matches:
                    self.issues.append(
                        f"hardware_facts.{key}={actual_rate!r} does not match ERT source rate "
                        f"{expected_compute[key]!r} times calibration.resource_scaling.compute={compute_scale!r}"
                    )
        for resource in ("hbm", "dram", "hbm_capacity", "dram_capacity"):
            if resource in scaling and (
                isinstance(scaling[resource], bool) or not _finite_positive(scaling[resource])
            ):
                self.issues.append(
                    f"calibration.resource_scaling.{resource} must be finite and positive"
                )

        expected_precision = {
            "weight_bytes": expected_compute.get("weight_bytes"),
            "activation_bytes": expected_compute.get("activation_bytes"),
        }
        requested_compute = facts["compute"]
        for key, expected in expected_precision.items():
            if expected is None:
                continue
            if key not in requested_compute:
                self.issues.append(f"hardware_facts.compute.{key} is required")
            elif requested_compute[key] != expected:
                self.issues.append(
                    f"hardware_facts.compute.{key}={requested_compute[key]!r} does not match ERT source {expected!r}"
                )

        expected_noc = domain.get("noc", {})
        supported_topologies = expected_noc.get("supported_topologies", [])
        requested_noc = facts["noc"]
        nx, ny = requested_noc["nx"], requested_noc["ny"]
        if supported_topologies:
            self._noc_topology_key = f"{nx}x{ny}" if nx is not None and ny is not None else None
            if not any(
                row.get("nx") == nx and row.get("ny") == ny for row in supported_topologies
            ):
                self.issues.append(
                    f"hardware mesh {nx}x{ny} has no BookSim energy row; supported: "
                    + ", ".join(f"{row.get('nx')}x{row.get('ny')}" for row in supported_topologies)
                )
            topology_values = self.table.get("actions", {}).get("noc", {}).get(
                "topology_joules_per_unit", {}
            )
            for topology in supported_topologies:
                key = f"{topology.get('nx')}x{topology.get('ny')}"
                row = topology_values.get(key)
                if row is None or not _finite_positive(row.get("joules_per_unit")):
                    self.issues.append(f"NoC energy action is missing a finite positive value for {key}")
        for field in ("flit_width_bits", "num_vcs", "packet_size_flits", "router", "routing"):
            if requested_noc[field] != expected_noc.get(field):
                self.issues.append(
                    f"hardware_facts.noc.{field}={requested_noc[field]!r} does not match {expected_noc.get(field)!r}"
                )
        calibration = self.config.get("calibration", {})
        acceptance = calibration.get("memory_organization_sensitivity_accept", {})
        if not isinstance(acceptance, dict):
            self.issues.append("calibration.memory_organization_sensitivity_accept must be an object")
            acceptance = {}
        unknown_acceptance = set(acceptance) - {"hbm", "dram"}
        if unknown_acceptance:
            self.issues.append(
                "unknown calibration.memory_organization_sensitivity_accept keys: "
                + ", ".join(sorted(unknown_acceptance))
            )

        for resource in ("hbm", "dram"):
            expected = domain.get(resource)
            if not expected:
                continue
            actual = facts[resource]
            for field in ("protocol", "config"):
                if actual[field] != expected.get(field):
                    self.issues.append(
                        f"hardware_facts.{resource}.{field}={actual[field]!r} does not match {expected.get(field)!r}"
                    )
            configured_capacity_key = {
                "hbm": "hbm_capacity_bytes_per_die",
                "dram": "dram_capacity_bytes",
            }[resource]
            table_configured_capacity = expected.get(
                "configured_capacity_bytes_per_die" if resource == "hbm" else "configured_capacity_bytes"
            )
            runtime_capacity_raw = facts.get(configured_capacity_key)
            try:
                runtime_capacity_number = float(runtime_capacity_raw)
                if not math.isfinite(runtime_capacity_number) or runtime_capacity_number <= 0:
                    runtime_capacity_number = None
            except (TypeError, ValueError):
                runtime_capacity_number = None
            scale_key = f"{resource}_capacity"
            capacity_scale = scaling.get(scale_key, 1.0)
            if isinstance(capacity_scale, bool) or not _finite_positive(capacity_scale):
                capacity_scale = 1.0
            if table_configured_capacity is None:
                if not math.isclose(float(capacity_scale), 1.0):
                    self.issues.append(
                        f"cannot apply calibration.resource_scaling.{scale_key} without an ERT source capacity"
                    )
            else:
                expected_runtime_capacity = float(table_configured_capacity) * float(capacity_scale)
                capacity_matches = runtime_capacity_number is not None and math.isclose(
                    runtime_capacity_number, expected_runtime_capacity,
                    rel_tol=1e-12, abs_tol=1e-6,
                )
                if not capacity_matches:
                    self.issues.append(
                        f"hardware_facts.{configured_capacity_key}={runtime_capacity_raw!r} does not match ERT source capacity "
                        f"{table_configured_capacity!r} times calibration.resource_scaling.{scale_key}={capacity_scale!r}"
                    )
            simulator_capacity = expected.get("modeled_capacity_bytes")
            if simulator_capacity is not None and runtime_capacity_number is not None:
                simulator_capacity = int(simulator_capacity)
                runtime_capacity_bytes = int(round(runtime_capacity_number))
                if not math.isclose(runtime_capacity_number, runtime_capacity_bytes, rel_tol=0.0, abs_tol=1e-6):
                    self.issues.append(f"hardware_facts.{configured_capacity_key} must be an integer number of bytes")
                base_organization_mismatch = (
                    table_configured_capacity is not None
                    and int(table_configured_capacity) != simulator_capacity
                )
                current_organization_mismatch = runtime_capacity_bytes != simulator_capacity
                organization_accept_required = bool(
                    expected.get("organization_sensitivity_accept_required", False)
                ) or base_organization_mismatch or current_organization_mismatch
                if organization_accept_required and acceptance.get(resource) is not True:
                    self.issues.append(
                        f"{resource} capacity/organization differs from its ERT source or DRAMSim3 organization; explicitly set "
                        f"calibration.memory_organization_sensitivity_accept.{resource}=true"
                    )
                if organization_accept_required and acceptance.get(resource) is True:
                    mismatch = {
                        "resource": resource,
                        "base_ert_capacity_bytes": int(table_configured_capacity) if table_configured_capacity is not None else None,
                        "configured_capacity_bytes": runtime_capacity_bytes,
                        "simulator_config_capacity_bytes": simulator_capacity,
                        "ratio_configured_to_simulator": runtime_capacity_bytes / simulator_capacity,
                        "resource_scaling_factor": float(capacity_scale),
                    }
                    self.accepted_memory_organization_mismatches.append(mismatch)

    def _temperature(self, temperature_k: float | None, domain: list[float] | None = None) -> float:
        if temperature_k is None:
            temperature_k = float(self.config.get("thermal", {}).get("initial_k", 300.0))
        value = float(temperature_k)
        if not math.isfinite(value) or value <= 0:
            raise EnergyCalibrationError(f"invalid temperature_k: {temperature_k!r}")
        temp_domain = domain or self.table.get("temperature_domain_k", [300.0, 400.0])
        low, high = float(temp_domain[0]), float(temp_domain[-1])
        if not low <= value <= high:
            message = f"temperature {value:g} K outside calibrated domain [{low:g}, {high:g}] K"
            if self.strict:
                raise EnergyCalibrationError(message)
            self._last_downgrade = message + "; using nearest endpoint"
            self.issues.append(self._last_downgrade)
            value = min(high, max(low, value))
        return value

    @staticmethod
    def _interpolate_rows(
        rows: list[dict[str, Any]], temperature_k: float, value_key: str
    ) -> float:
        ordered = sorted(rows, key=lambda row: float(row["temperature_k"]))
        if temperature_k <= float(ordered[0]["temperature_k"]):
            return float(ordered[0][value_key])
        if temperature_k >= float(ordered[-1]["temperature_k"]):
            return float(ordered[-1][value_key])
        for left, right in zip(ordered, ordered[1:]):
            t0, t1 = float(left["temperature_k"]), float(right["temperature_k"])
            if t0 <= temperature_k <= t1:
                v0, v1 = float(left[value_key]), float(right[value_key])
                alpha = (temperature_k - t0) / (t1 - t0)
                return v0 + (v1 - v0) * alpha
        raise EnergyCalibrationError("temperature interpolation table is not ordered")

    def _fallback_energy(self, kind: str) -> float:
        energy = self.config.get("energy", {})
        legacy = {
            "pe": energy.get("pe_j_per_op"),
            "vec": energy.get("vec_j_per_op"),
            "control": energy.get("control_j_per_action", energy.get("vec_j_per_op")),
            "sram_read": energy.get("sram_j_per_byte"),
            "sram_write": energy.get("sram_j_per_byte"),
            "hbm": energy.get("hbm_j_per_byte"),
            "hbm_read": energy.get("hbm_j_per_byte"),
            "hbm_write": energy.get("hbm_j_per_byte"),
            "dram": energy.get("dram_j_per_byte"),
            "dram_read": energy.get("dram_j_per_byte"),
            "dram_write": energy.get("dram_j_per_byte"),
            "noc": energy.get("noc_j_per_byte"),
            "d2d": 8 * float(energy.get("d2d_j_per_bit", 0)),
            "dram_io": energy.get("dram_io_j_per_byte"),
        }
        value = legacy.get(kind)
        if not _finite_nonnegative(value):
            raise EnergyCalibrationError(f"no table or legacy fallback for action {kind!r}")
        self._last_downgrade = f"action {kind} used legacy config coefficient"
        return float(value)

    def energy(self, kind: str, count: float, temperature_k: float | None) -> float:
        amount = float(count)
        if not math.isfinite(amount) or amount < 0:
            raise EnergyCalibrationError(f"invalid action count for {kind}: {count!r}")
        if not self.enabled:
            return amount * self._fallback_energy(kind)
        row = self.table.get("actions", {}).get(kind)
        if kind in DIRECTIONAL_ACTIONS:
            row = self._directional_action_row(kind)
        if row is None:
            if self.strict:
                descriptor = DIRECTIONAL_ACTIONS.get(kind)
                detail = (
                    f" (Accelergy component {descriptor['ert_component']!r})"
                    if descriptor else ""
                )
                raise EnergyCalibrationError(f"energy action {kind!r}{detail} is missing from strict table")
            fallback_kind = DIRECTIONAL_ACTIONS[kind]["resource"] if kind in DIRECTIONAL_ACTIONS else kind
            unit_j = self._fallback_energy(fallback_kind)
            self._last_downgrade = f"directional action {kind} used generic {fallback_kind} coefficient"
            return amount * unit_j
        if kind == "noc" and row.get("topology_joules_per_unit"):
            topology_key = self._noc_topology_key
            topology_row = row["topology_joules_per_unit"].get(topology_key)
            if topology_row is None:
                if self.strict:
                    raise EnergyCalibrationError(
                        f"NoC energy has no exact topology value for {topology_key!r}"
                    )
                self._last_downgrade = (
                    f"NoC energy topology {topology_key!r} unavailable; used generic table value"
                )
            else:
                row = topology_row
        rows = row.get("joules_per_unit_by_temperature_k")
        if rows:
            temp = self._temperature(
                temperature_k,
                row.get("temperature_domain_k", [rows[0]["temperature_k"], rows[-1]["temperature_k"]]),
            )
            unit_j = self._interpolate_rows(rows, temp, "joules_per_unit")
        else:
            if row.get("temperature_domain_k"):
                self._temperature(temperature_k, row["temperature_domain_k"])
            unit_j = float(row["joules_per_unit"])
        result = amount * unit_j
        if not math.isfinite(result) or result < 0:
            raise EnergyCalibrationError(f"invalid computed action energy for {kind}")
        return result

    def background(self, kind: str, count: float, temperature_k: float | None) -> float:
        amount = float(count)
        if not math.isfinite(amount) or amount < 0:
            raise EnergyCalibrationError(f"invalid background count for {kind}: {count!r}")
        if not self.enabled:
            return amount * self._fallback_background(kind, temperature_k)
        row = self.table.get("background", {}).get(kind)
        if row is None:
            if self.strict:
                raise EnergyCalibrationError(f"background {kind!r} is missing from strict table")
            return amount * self._fallback_background(kind, temperature_k)
        if kind == "compute_leak":
            temp = float(temperature_k if temperature_k is not None else 300.0)
            min_k = float(row.get("minimum_temperature_k", 300.0))
            max_k = float(row.get("maximum_temperature_k", 450.0))
            if not min_k <= temp <= max_k:
                message = f"compute leakage temperature {temp:g} K outside [{min_k:g}, {max_k:g}] K"
                if self.strict:
                    raise EnergyCalibrationError(message)
                self._last_downgrade = message + "; clamped to endpoint"
                self.issues.append(self._last_downgrade)
                temp = min(max_k, max(min_k, temp))
            watts = float(row["watts_per_die_at_reference"]) * math.exp(
                float(row["beta_per_k"])
                * (temp - float(row["reference_temperature_k"]))
            )
            return amount * watts
        if kind == "sram_leak":
            rows = row["watts_per_macro_by_temperature_k"]
            temp_domain = [float(rows[0]["temperature_k"]), float(rows[-1]["temperature_k"])]
            temp = self._temperature(temperature_k, temp_domain)
            watts_per_macro = self._interpolate_rows(rows, temp, "watts_per_macro")
            assert self._sram_macro_count_per_die is not None
            macro_count = float(self._sram_macro_count_per_die)
            return amount * macro_count * watts_per_macro
        return amount * float(row["watts_per_unit"])

    def _fallback_background(self, kind: str, temperature_k: float | None) -> float:
        energy = self.config.get("energy", {})
        temp = float(temperature_k if temperature_k is not None else 300.0)
        legacy = {
            "compute_idle": energy.get("idle_compute_w_per_die"),
            "compute_leak": float(energy.get("leak_compute_w_at_300k", 0))
            * math.exp(float(energy.get("leak_beta_per_k", 0)) * (min(temp, 450.0) - 300.0)),
            "sram_leak": 0.0,
            "hbm_idle": energy.get("idle_hbm_w_per_die"),
            "router_idle": energy.get("idle_router_w_per_die"),
            "dram_background": energy.get("dram_background_w"),
        }
        value = legacy.get(kind)
        if not _finite_nonnegative(value):
            raise EnergyCalibrationError(f"no table or legacy fallback for background {kind!r}")
        self._last_downgrade = f"background {kind} used legacy config coefficient"
        return float(value)

    def thermal_destination(self, kind: str) -> str:
        if kind not in THERMAL_DESTINATIONS:
            if self.strict:
                raise EnergyCalibrationError(f"no thermal destination for action {kind!r}")
            return "external"
        return THERMAL_DESTINATIONS[kind]

    def metadata(self) -> dict[str, Any]:
        if not self.enabled:
            return {
                "status": "not_configured",
                "issues_or_downgrades": list(self.issues),
                "legacy_fallback": "configs.<run>.energy coefficients",
            }
        base = self.table.get("metadata", {})
        resource_scaling = self.config.get("calibration", {}).get("resource_scaling", {})
        if not isinstance(resource_scaling, dict):
            resource_scaling = {}
        non_unit_scaling = False
        for value in resource_scaling.values():
            try:
                if not math.isclose(float(value), 1.0):
                    non_unit_scaling = True
            except (TypeError, ValueError):
                non_unit_scaling = True
        if self.issues:
            status = "explicit_downgrade_not_target_calibrated"
        elif self.strict:
            status = "strict_table_domain_match_mixed_model_not_target_calibrated"
        else:
            status = "table_domain_match_mixed_model_not_target_calibrated"
        return {
            **base,
            "status": status,
            "table_path": str(self.path),
            "table_sha256": _sha256(self.path),
            "schema_version": self.table.get("schema_version"),
            "strict": self.strict,
            "target_calibrated": False,
            "calibration_basis": "mixed upstream models and explicit assumptions; no target 7 nm calibration",
            "resource_scaling_sensitivity": {
                "factors": dict(resource_scaling),
                "status": (
                    "ideal resource scaling sensitivity; not a new measured table"
                    if non_unit_scaling
                    else "no non-unit resource scaling"
                ),
                "energy_assumption": "J/op remains fixed under explicit compute throughput scaling, assuming unchanged voltage and implementation",
                "capacity_assumption": "hbm_capacity and dram_capacity factors scale explicit host capacities against the ERT source; they do not characterize a new DRAM organization, so memory_organization_sensitivity_accept remains required",
            },
            "hardware_facts_source": "explicit EnergyModel hardware_facts keyword argument",
            "hardware_facts_sha256": self.hardware_facts_sha256,
            "hardware_fact_summary": {
                "sram_capacity_bytes_per_die": self.hardware_facts["sram_capacity_bytes_per_die"],
                "sram_macro_capacity_bytes": self.hardware_facts["sram"]["macro_capacity_bytes"],
                "sram_actual_macro_count_per_die": self.hardware_facts["sram"]["macro_count_per_die"],
                "sram_reference_macro_count_per_die": self.table.get("background", {}).get("sram_leak", {}).get("macro_count_per_die"),
                "noc_topology": self._noc_topology_key,
            },
            "assumption_ids": self._assumption_ids(),
            "accepted_memory_organization_mismatches": list(self.accepted_memory_organization_mismatches),
            "selected_noc_topology": self._noc_topology_key,
            "selected_noc_energy": (
                self.table.get("actions", {}).get("noc", {}).get("topology_joules_per_unit", {}).get(self._noc_topology_key)
                if self._noc_topology_key else None
            ),
            "issues_or_downgrades": list(self.issues),
            "action_units": {key: row.get("unit") for key, row in self.table.get("actions", {}).items()},
            "action_sources": {key: row.get("source") for key, row in self.table.get("actions", {}).items()},
            "directional_action_aliases": {
                kind: {
                    "unit": row.get("unit"),
                    "ert_unit": "pJ per Accelergy action (one byte payload)",
                    "joules_per_unit": row.get("joules_per_unit"),
                    "ert_component": row.get("ert_component"),
                    "ert_energy_pj": row.get("ert_energy_pj"),
                    "source": row.get("source"),
                }
                for kind in DIRECTIONAL_ACTIONS
                if (row := self._directional_action_row(kind)) is not None
            },
            "background_sources": {key: row.get("source") for key, row in self.table.get("background", {}).items()},
            "thermal_destinations": dict(THERMAL_DESTINATIONS),
        }
