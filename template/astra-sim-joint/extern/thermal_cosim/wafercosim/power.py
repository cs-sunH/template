"""Turn activity windows into per-thermal-block power (``ActivityWindow`` -> ``PowerWindow``).

This is where the thermal component answers the user's requirement that *"the
thermal repo, given the workload task, the software mapping and the hardware
configuration, provides the power and further the thermal feedback"*.

Ledger for thermal block ``b`` over an interval ``dt``::

    E_dynamic[b]    = sum(action a, source s) weight[s,b] * energy(a, count[s,a], T[s])
    E_background[b] = background_power(b, T, power_state) * dt
    P[b]            = (E_dynamic[b] + E_background[b]) / dt
    E_system        = sum_b P[b] * dt + E_external

Design rules enforced here:

* One HBM/NoC/D2D transfer is counted exactly once.  The count semantics come
  from :data:`wafercosim.contracts.ACTION_UNITS`; a caller that supplies NoC
  bytes must use the same hop convention its energy coefficient was calibrated
  with, which the table's ``topology_joules_per_unit`` selection makes explicit.
* External energy (off-wafer DRAM cells/background) goes to the external ledger
  and never to an on-wafer block.
* Background power is defined per die by the energy table, so it is evaluated at
  that die's own temperature and spread only over that die's blocks.
* Temperature dependence uses the window's starting block temperature, which is
  a declared first-order coupling; refine with a halved thermal step.
"""
from __future__ import annotations

from typing import Any, Mapping

from .action_energy import EnergyCalibrationError, EnergyModel
from .contracts import (
    ACTION_DESTINATIONS,
    ACTION_UNITS,
    ActivityWindow,
    ContractError,
    HardwareSnapshot,
    MappingSnapshot,
    PowerWindow,
)

# Background kind -> (thermal destination, per-die quantity?).
_BACKGROUND_DESTINATIONS = {
    "compute_idle": "compute",
    "compute_leak": "compute",
    "sram_leak": "compute",
    "hbm_idle": "hbm",
    "router_idle": "router",
    "dram_background": "external",
}

# Fixed voltage / frequency-only scaling declaration.  P proportional to f**3 is
# deliberately *not* applied: without a voltage table the dynamic energy per
# action stays in the calibrated domain, and a lower frequency only reduces how
# many actions complete per unit wall time.
OPERATING_POINT_MODEL = "fixed_voltage_frequency_only"


class PowerModel:
    """Compose the energy table, the thermal floorplan and the physical mapping.

    ``block_temperature_k`` supplies per-block starting temperatures for the
    temperature-dependent terms; the session refreshes it after every committed
    thermal step.
    """

    def __init__(self, energy: EnergyModel, blocks: list[Any],
                 block_kinds: Mapping[str, str]):
        self.energy = energy
        self.blocks = list(blocks)
        self.block_kinds = dict(block_kinds)
        self.block_ids = tuple(block.name for block in self.blocks)
        self._index = {name: i for i, name in enumerate(self.block_ids)}
        if len(self._index) != len(self.block_ids):
            raise ContractError("thermal block names must be unique")
        # (die, kind) -> block indices, plus die -> all indices.
        self._by_die_kind: dict[tuple[int, str], list[int]] = {}
        self._by_die: dict[int, list[int]] = {}
        for index, block in enumerate(self.blocks):
            kind = self.block_kinds.get(block.name)
            if kind not in {"compute", "hbm", "router"}:
                raise ContractError(
                    f"thermal block {block.name!r} has no compute/hbm/router "
                    "classification")
            self._by_die_kind.setdefault((block.die, kind), []).append(index)
            self._by_die.setdefault(block.die, []).append(index)
        for die in sorted(self._by_die):
            if not self._by_die_kind.get((die, "compute")):
                raise ContractError(
                    f"thermal die {die} has no compute block for power projection")
        self.dies = tuple(sorted(self._by_die))

    # ------------------------------------------------------------- helpers --

    def _block_temperature(self, index: int, block_temperature_k: Mapping[str, float],
                           initial_k: float) -> float:
        value = block_temperature_k.get(self.block_ids[index])
        return float(value) if value is not None else float(initial_k)

    def _die_temperature(self, die: int, block_temperature_k: Mapping[str, float],
                         initial_k: float) -> float:
        """Mean block temperature of one die at the window start."""
        indices = self._by_die[die]
        return sum(self._block_temperature(i, block_temperature_k, initial_k)
                   for i in indices) / len(indices)

    @staticmethod
    def _source_die(source: str) -> int | None:
        head = source.split(".", 1)[0]
        if not head.startswith("die"):
            return None
        try:
            return int(head[3:])
        except ValueError:
            return None

    @staticmethod
    def _source_kind(source: str) -> str | None:
        parts = source.split(".")
        return parts[1] if len(parts) == 2 else None

    # --------------------------------------------------------------- power --

    def estimate(self, activity: ActivityWindow, mapping: MappingSnapshot,
                 hardware: HardwareSnapshot,
                 block_temperature_k: Mapping[str, float],
                 initial_k: float) -> PowerWindow:
        if activity.hardware_hash != hardware.content_hash:
            raise ContractError(
                "activity window hardware hash does not match the snapshot")
        if activity.mapping_epoch != mapping.mapping_epoch:
            raise ContractError(
                "activity window mapping epoch does not match the mapping snapshot")
        if mapping.hardware_hash != hardware.content_hash:
            raise ContractError("mapping snapshot is bound to a different hardware hash")

        point = activity.operating_point_object(hardware.clock_hz)
        dt_s = activity.dt_s
        diagnostics: dict[str, Any] = {
            "operating_point": point.to_json(),
            "operating_point_model": OPERATING_POINT_MODEL,
            "temperature_basis": "window_start_block_k",
            "coverage": activity.coverage,
            "provenance": activity.provenance,
            "action_energy_j_by_kind": {},
            "external_sources": [],
        }

        dynamic_by_block = [0.0] * len(self.blocks)
        background_by_block = [0.0] * len(self.blocks)
        external_j = 0.0

        for source, kind, count in activity.actions:
            if count == 0.0:
                continue
            if kind not in ACTION_UNITS:
                raise ContractError(
                    f"action {kind!r} has no registered unit; unmodelled actions "
                    "must be reported instead of assumed zero")
            destination = mapping.source_to_destination.get(source)
            if destination == "external":
                external_j += self._action_energy(kind, count, initial_k)
                diagnostics["external_sources"].append(f"{source}:{kind}")
                continue
            distribution = mapping.source_block_weights.get(source)
            if distribution is None:
                raise ContractError(
                    f"source {source!r} emits on-wafer energy but has no explicit "
                    "thermal block mapping; averaging across all dies is not a "
                    "physical mapping")
            die = self._source_die(source)
            if die is None:
                raise ContractError(
                    f"source {source!r} is not a 'die<N>.<kind>' physical source")
            temperature = self._die_temperature(die, block_temperature_k, initial_k)
            joules = self._action_energy(kind, count, temperature)
            row = diagnostics["action_energy_j_by_kind"]
            row[kind] = row.get(kind, 0.0) + joules
            for block_name, weight in distribution.items():
                dynamic_by_block[self._index[block_name]] += joules * weight

        # Background: an explicit zero-activity window still advances leakage and
        # idle power.  Leakage integrates over real elapsed time, not action count.
        for die in self.dies:
            temperature = self._die_temperature(die, block_temperature_k, initial_k)
            for kind, destination in _BACKGROUND_DESTINATIONS.items():
                if destination == "external":
                    continue
                indices = self._by_die_kind.get((die, destination))
                if not indices:
                    continue
                total_j = self._background_watts(kind, temperature) * dt_s
                share = total_j / len(indices)
                for index in indices:
                    background_by_block[index] += share
        external_j += (self._background_watts("dram_background", initial_k) * dt_s
                       if self._has_background("dram_background") else 0.0)

        power_w = tuple(
            (dynamic_by_block[i] + background_by_block[i]) / dt_s
            for i in range(len(self.blocks))
        )
        total_j = (sum(dynamic_by_block) + sum(background_by_block) + external_j)
        return PowerWindow(
            run_id=activity.run_id,
            activity_seq=activity.activity_seq,
            start_ns=activity.start_ns,
            end_ns=activity.end_ns,
            hardware_hash=hardware.content_hash,
            mapping_epoch=mapping.mapping_epoch,
            resource_epoch=activity.resource_epoch,
            block_ids=self.block_ids,
            power_w=power_w,
            dynamic_j={n: dynamic_by_block[i] for i, n in enumerate(self.block_ids)},
            background_j={n: background_by_block[i] for i, n in enumerate(self.block_ids)},
            external_j=external_j,
            total_energy_j=total_j,
            coverage=activity.coverage,
            diagnostics=diagnostics,
        )

    # ---------------------------------------------------------- energy calls --

    def _has_background(self, kind: str) -> bool:
        """Whether the configured energy table/fallback can price this kind."""
        table = getattr(self.energy, "table", None) or {}
        if kind in (table.get("background") or {}):
            return True
        return not getattr(self.energy, "enabled", False)

    def _action_energy(self, kind: str, count: float, temperature_k: float) -> float:
        try:
            return float(self.energy.energy(kind, count, temperature_k))
        except EnergyCalibrationError:
            raise
        except Exception as exc:  # noqa: BLE001 -- keep the calibration domain visible
            raise ContractError(f"energy evaluation failed for {kind!r}: {exc}") from exc

    def _background_watts(self, kind: str, temperature: float) -> float:
        """Watts for one background kind at one die's representative temperature."""
        try:
            return float(self.energy.background(kind, 1.0, temperature))
        except EnergyCalibrationError:
            raise
        except Exception as exc:  # noqa: BLE001
            raise ContractError(f"background evaluation failed for {kind!r}: {exc}") from exc


__all__ = ["PowerModel", "OPERATING_POINT_MODEL"]