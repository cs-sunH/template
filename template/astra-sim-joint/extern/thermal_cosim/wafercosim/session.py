"""Long-lived thermal session: committed simulation time, power windows and feedback.

One session is created by the joint online service and lives for the whole run.
It owns the single thermal state and refuses to advance on anything but a
contiguous, non-overlapping, non-regressing sequence of committed windows:

* ``advance_committed`` requires ``power.start_ns == committed_time_ns`` and
  rejects overlap, regression and unexplained gaps (an *explicit* zero-activity
  window is how idle time is advanced, so background leakage and cooling still
  occur).
* Repeated submission of the same ``activity_seq`` returns the already committed
  result instead of integrating the heat twice.
* ``fork`` clones the physical state for candidate preview.  Discarding a fork
  must leave the committed temperature, energy ledger and time untouched, which
  is why the branch is a separate solver instance restored from one snapshot.
"""
from __future__ import annotations

from typing import Any, Mapping

from . import contracts
from .action_energy import EnergyModel
from .contracts import (
    ActivityWindow,
    ContractError,
    HardwareSnapshot,
    MappingSnapshot,
    PowerWindow,
    ThermalFeedback,
)
from .power import PowerModel, OPERATING_POINT_MODEL

# model/calibration identity recorded with every feedback row
MODEL_ID = "wafercosim.rc_liquid.v2"
CALIBRATION_ID = "research-v1-assumed"


def _build_power_model(energy: EnergyModel, thermal: Any) -> PowerModel:
    kinds = {block.name: block.kind for block in thermal.blocks}
    return PowerModel(energy, list(thermal.blocks), kinds)


class ThermalSession:
    """Own the committed thermal timeline for one joint run."""

    def __init__(self, *, run_id: str, thermal: Any, energy: EnergyModel,
                 hardware: HardwareSnapshot, mapping: MappingSnapshot,
                 initial_k: float | None = None):
        self.run_id = run_id
        self.thermal = thermal
        self.energy = energy
        self.hardware = hardware
        self.mapping = mapping
        self.power_model = _build_power_model(energy, thermal)
        self.initial_k = float(
            initial_k if initial_k is not None
            else thermal.config["thermal"]["initial_k"])
        self.committed_time_ns = 0
        self.thermal_epoch = 0
        self._committed: dict[int, PowerWindow] = {}
        self._feedback: dict[int, ThermalFeedback] = {}
        self._last_feedback: ThermalFeedback | None = None
        self._rows: dict[str, list[dict[str, Any]]] = {
            "activity_windows": [], "power_windows": [],
            "thermal_feedback": [],
        }
        self.closed = False

    # ---------------------------------------------------------- properties --

    @property
    def block_ids(self) -> tuple[str, ...]:
        return self.power_model.block_ids

    @property
    def state_version(self) -> int:
        return int(self.thermal.state_version())

    def block_temperature_k(self) -> dict[str, float]:
        return {name: float(self.thermal.T[i])
                for i, name in enumerate(self.block_ids)}

    # ------------------------------------------------------------- powering --

    def estimate_power(self, activity: ActivityWindow,
                       hardware: HardwareSnapshot | None = None,
                       mapping: MappingSnapshot | None = None) -> PowerWindow:
        """Activity -> per-block power using the window's start temperature."""
        self._require_open()
        window = self.power_model.estimate(
            activity, mapping or self.mapping, hardware or self.hardware,
            self.block_temperature_k(), self.initial_k)
        self._rows["activity_windows"].append(activity.to_json())
        self._rows["power_windows"].append(window.to_json())
        return window

    def idle_window(self, *, start_ns: int | None = None, dt_ns: int = 0,
                    resource_epoch: int = 0,
                    activity_seq: int | None = None) -> ActivityWindow:
        """Explicit zero-activity window so idle time still advances heat."""
        start = self.committed_time_ns if start_ns is None else start_ns
        if dt_ns <= 0:
            raise ContractError("an idle window requires a positive dt_ns")
        return ActivityWindow(
            run_id=self.run_id,
            activity_seq=(self.thermal_epoch if activity_seq is None else activity_seq),
            start_ns=start, end_ns=start + dt_ns,
            hardware_hash=self.hardware.content_hash,
            mapping_epoch=self.mapping.mapping_epoch,
            resource_epoch=resource_epoch,
            actions=(), dwell=(),
            coverage="measured",
            provenance="idle_zero_activity_window",
        )

    # ------------------------------------------------------------- advancing --

    def advance_committed(self, power: PowerWindow) -> ThermalFeedback:
        """Integrate one window into the committed state and return feedback.

        The window must be contiguous with ``committed_time_ns``; resubmitting a
        known ``activity_seq`` returns the previously committed feedback.
        """
        self._require_open()
        if power.run_id != self.run_id:
            raise ContractError(
                f"power window belongs to run {power.run_id!r}, session is {self.run_id!r}")
        if power.hardware_hash != self.hardware.content_hash:
            raise ContractError("power window hardware hash does not match the session")
        if power.mapping_epoch != self.mapping.mapping_epoch:
            raise ContractError("power window mapping epoch does not match the session")
        existing = self._committed.get(power.activity_seq)
        if existing is not None:
            if (existing.start_ns, existing.end_ns) != (power.start_ns, power.end_ns):
                raise ContractError(
                    "duplicate activity_seq with a different interval cannot be "
                    "re-committed")
            # Idempotent: re-submitting a committed sequence returns the already
            # committed feedback instead of integrating the same heat twice.
            return self._feedback[power.activity_seq]
        if power.start_ns != self.committed_time_ns:
            relation = ("overlaps the committed interval"
                        if power.start_ns < self.committed_time_ns
                        else "leaves an unexplained gap before it")
            raise ContractError(
                f"power window starts at {power.start_ns} ns but the committed "
                f"time is {self.committed_time_ns} ns ({relation}); idle time must "
                "be advanced by an explicit zero-activity window")
        feedback = self._integrate(power)
        self._committed[power.activity_seq] = power
        self._feedback[power.activity_seq] = feedback
        self._last_feedback = feedback
        self._rows["thermal_feedback"].append(feedback.to_json())
        return feedback

    def _integrate(self, power: PowerWindow) -> ThermalFeedback:
        # Physical step first: a rejected/invalid integration must not advance the
        # committed clock.  Only after a successful step are time and epoch moved.
        self.thermal.step(list(power.power_w), power.dt_s)
        self.committed_time_ns = power.end_ns
        self.thermal_epoch += 1
        return self._feedback_for(power.end_ns, power.resource_epoch)

    def _feedback_for(self, effective_ns: int, resource_epoch: int) -> ThermalFeedback:
        raw = self.thermal.feedback()
        domain_status = self._domain_status(raw["block_k"])
        return ThermalFeedback(
            thermal_epoch=self.thermal_epoch,
            effective_ns=effective_ns,
            committed_s=effective_ns * contracts.NS_PER_S ** -1,
            block_k=raw["block_k"],
            coolant_k=raw["coolant_k"],
            die_block_max_k=raw["die_block_max_k"],
            energy_residual_j=raw["energy_residual_j"],
            input_j=raw["input_j"],
            removed_j=raw["removed_j"],
            model_id=MODEL_ID,
            calibration_id=CALIBRATION_ID,
            state_version=int(raw["state_version"]),
            domain_status=domain_status,
            hardware_hash=self.hardware.content_hash,
            mapping_epoch=self.mapping.mapping_epoch,
            resource_epoch=resource_epoch,
        )

    def _domain_status(self, block_k: Mapping[str, float]) -> str:
        """Whether the reported temperature is inside the calibrated domain.

        The energy table's temperature domain bounds the *energy* model; a
        temperature outside it means the power ledger itself is extrapolated, so
        the status is reported instead of silently clamping.
        """
        table = getattr(self.energy, "table", None) or {}
        domain = table.get("temperature_domain_k") or [300.0, 400.0]
        low, high = float(domain[0]), float(domain[-1])
        outside = [name for name, value in block_k.items() if not low <= value <= high]
        if outside:
            return (
                f"out_of_energy_domain[{low:g},{high:g}]K:{len(outside)}_blocks")
        return f"in_energy_domain[{low:g},{high:g}]K"

    # ------------------------------------------------------------ snapshot --

    def snapshot(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "committed_time_ns": self.committed_time_ns,
            "thermal_epoch": self.thermal_epoch,
            "hardware_hash": self.hardware.content_hash,
            "mapping_epoch": self.mapping.mapping_epoch,
            "model_id": MODEL_ID,
            "calibration_id": CALIBRATION_ID,
            "thermal": self.thermal.snapshot(),
        }

    def fork(self) -> "ThermalSession":
        """A candidate branch from the committed snapshot.

        The branch has its own solver instance; discarding it cannot alter the
        committed temperature, ledger or time of this session.
        """
        self._require_open()
        if not getattr(self.thermal, "supports_snapshot", False):
            raise ContractError(
                f"thermal backend {self.thermal.name!r} cannot be forked for "
                "candidate preview (it has no snapshot capability)")
        branch = self._derived(run_id=f"{self.run_id}.fork.{self.thermal_epoch}")
        branch.thermal.restore(self.thermal.snapshot())
        branch.committed_time_ns = self.committed_time_ns
        branch.thermal_epoch = self.thermal_epoch
        return branch

    def fork_from(self, snapshot: Mapping[str, Any]) -> "ThermalSession":
        """Restore a candidate branch from a previously taken snapshot dict."""
        if snapshot.get("hardware_hash") != self.hardware.content_hash:
            raise ContractError("snapshot was taken against a different hardware hash")
        if snapshot.get("mapping_epoch") != self.mapping.mapping_epoch:
            raise ContractError("snapshot was taken against a different mapping epoch")
        branch = self._derived(run_id=snapshot.get("run_id", self.run_id))
        branch.thermal.restore(snapshot["thermal"])
        branch.committed_time_ns = int(snapshot["committed_time_ns"])
        branch.thermal_epoch = int(snapshot["thermal_epoch"])
        return branch

    def _derived(self, *, run_id: str) -> "ThermalSession":
        return type(self)(
            run_id=run_id,
            thermal=type(self.thermal)(self.thermal.config),
            energy=self.energy,
            hardware=self.hardware,
            mapping=self.mapping,
            initial_k=self.initial_k,
        )

    def reset(self) -> None:
        self.thermal.reset()
        self.committed_time_ns = 0
        self.thermal_epoch = 0
        self._committed.clear()
        self._feedback.clear()
        self._last_feedback = None
        for rows in self._rows.values():
            rows.clear()

    def close(self) -> None:
        if self.closed:
            return
        self.thermal.close()
        self.closed = True

    def _require_open(self) -> None:
        if self.closed:
            raise ContractError("thermal session is closed")

    # ----------------------------------------------------------------- audit --

    def audit_rows(self, name: str) -> list[dict[str, Any]]:
        if name not in self._rows:
            raise ContractError(
                f"unknown audit stream {name!r}; expected one of "
                f"{sorted(self._rows)}")
        return list(self._rows[name])

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": contracts.SCHEMA_VERSION,
            "run_id": self.run_id,
            "model_id": MODEL_ID,
            "calibration_id": CALIBRATION_ID,
            "operating_point_model": OPERATING_POINT_MODEL,
            "thermal_backend": self.thermal.name,
            "hardware_hash": self.hardware.content_hash,
            "hardware_target_calibrated": self.hardware.target_calibrated,
            "mapping_epoch": self.mapping.mapping_epoch,
            "mapping_hash": self.mapping.content_hash,
            "block_ids": list(self.block_ids),
            "energy": self.energy.metadata(),
            "committed_time_ns": self.committed_time_ns,
            "thermal_epoch": self.thermal_epoch,
        }


__all__ = ["ThermalSession", "MODEL_ID", "CALIBRATION_ID"]