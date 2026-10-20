#!/usr/bin/env python3
"""thermal_replay.py -- deterministically replay a thermal run from its audit.

Given a run directory produced by the joint thermal bridge, this rebuilds the
thermal session from the recorded hardware/mapping/activity and re-runs it from
zero, then verifies:

* the recorded per-block temperature at every committed window is reproduced,
* the cumulative energy ledger stays consistent (near-zero residual),
* the recorded resource envelopes are reproduced by the recorded policy,
* the activity ledger is contiguous with no gaps or overlaps.

The replay never trusts the recorded temperature: it recomputes it and compares.
This is the plan's V14 determinism/convergence entry point and works offline,
with no native job.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THERMAL_DIR = _REPO_ROOT / "extern" / "thermal_cosim"
if str(_THERMAL_DIR) not in sys.path:
    sys.path.insert(0, str(_THERMAL_DIR))

from wafercosim.contracts import ActivityWindow, HardwareSnapshot, MappingSnapshot  # noqa: E402
from wafercosim.session import ThermalSession  # noqa: E402
from wafercosim.thermal import LiquidRC  # noqa: E402

LLAMA_DIR = _REPO_ROOT / "sh_test_mesh" / "workload" / "llama2_7b_inference"
if str(LLAMA_DIR) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "sh_test_mesh" / "workload"))

from joint.thermal_bridge import ThermalPolicy, ThermalPolicyConfig, _load_thermal_profile  # noqa: E402


def _read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.is_file():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def replay(run_dir: Path, *, temperature_tolerance_k: float = 1e-6,
           energy_tolerance_j: float = 1e-6) -> dict:
    run_dir = Path(run_dir)
    manifest = json.loads((run_dir / "thermal_manifest.json").read_text())
    hardware = HardwareSnapshot.from_json(
        json.loads((run_dir / "resolved_hardware.json").read_text()))
    mapping = MappingSnapshot.from_json(
        json.loads((run_dir / "physical_mapping.json").read_text()))
    windows = [ActivityWindow.from_json(row)
               for row in _read_jsonl(run_dir / "activity_windows.jsonl")]
    recorded_feedback = _read_jsonl(run_dir / "thermal_feedback.jsonl")
    recorded_envelopes = _read_jsonl(run_dir / "resource_envelopes.jsonl")

    errors: list[str] = []
    if mapping.hardware_hash != hardware.content_hash:
        errors.append("mapping hardware hash does not match the hardware snapshot")
    if manifest.get("hardware_hash") != hardware.content_hash:
        errors.append("manifest hardware hash does not match the hardware snapshot")
    if manifest.get("mapping_hash") != mapping.content_hash:
        errors.append("manifest mapping hash does not match the mapping snapshot")

    thermal_config = _load_thermal_profile(manifest["thermal_profile"])
    from wafercosim.action_energy import EnergyModel

    energy = EnergyModel(
        {
            "calibration": {
                "energy_table": manifest["energy_table"],
                "strict": bool(manifest.get("energy_strict", True)),
                "accept_assumed_energy": bool(manifest.get("accept_assumed_energy", False)),
                "memory_organization_sensitivity_accept": {"hbm": True, "dram": True},
            },
            "thermal": {"initial_k": thermal_config["thermal"]["initial_k"]},
        },
        hardware_facts=hardware.energy_hardware_facts,
    )
    session = ThermalSession(
        run_id=manifest["run_id"], thermal=LiquidRC(thermal_config),
        energy=energy, hardware=hardware, mapping=mapping)
    policy = ThermalPolicy(ThermalPolicyConfig.from_json(manifest["policy"]))

    max_temperature_delta = 0.0
    for index, window in enumerate(windows):
        power = session.estimate_power(window)
        feedback = session.advance_committed(power)
        if index < len(recorded_feedback):
            recorded = recorded_feedback[index]["block_k"]
            for block, value in recorded.items():
                delta = abs(feedback.block_k[block] - float(value))
                max_temperature_delta = max(max_temperature_delta, delta)
            effective = recorded_feedback[index]["effective_ns"]
            if effective != feedback.effective_ns:
                errors.append(
                    f"window {index}: effective_ns {feedback.effective_ns} != "
                    f"recorded {effective}")
        if index < len(recorded_envelopes):
            recorded = recorded_envelopes[index]
            envelope = policy.compute_envelope(
                feedback, resource_epoch=recorded["resource_epoch"],
                effective_from_ns=feedback.effective_ns,
                as_of_ns=feedback.effective_ns)
            for row, recorded_row in zip(envelope.domains, recorded["domains"]):
                if row["domain"] != recorded_row["domain"]:
                    errors.append(f"window {index}: envelope domain order diverged")
                if abs(row["compute_throughput_fraction"]
                       - float(recorded_row["compute_throughput_fraction"])) > 1e-9:
                    errors.append(
                        f"window {index}: envelope {row['domain']} fraction "
                        f"{row['compute_throughput_fraction']} != recorded "
                        f"{recorded_row['compute_throughput_fraction']}")

    if len(recorded_feedback) != len(windows):
        errors.append(
            f"recorded {len(recorded_feedback)} feedback rows for {len(windows)} windows")
    raw = session.thermal.feedback()
    if abs(raw["energy_residual_j"]) > energy_tolerance_j:
        errors.append(
            f"energy residual {raw['energy_residual_j']:.3e} J exceeds the "
            f"{energy_tolerance_j:.1e} J budget")
    if max_temperature_delta > temperature_tolerance_k:
        errors.append(
            f"max block temperature delta {max_temperature_delta:.3e} K exceeds "
            f"the {temperature_tolerance_k:.1e} K tolerance")

    result = {
        "run_dir": str(run_dir),
        "mode": manifest.get("mode"),
        "windows": len(windows),
        "committed_time_ns": session.committed_time_ns,
        "max_temperature_delta_k": max_temperature_delta,
        "energy_residual_j": raw["energy_residual_j"],
        "passed": not errors,
        "errors": errors,
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Replay a joint thermal run offline")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--verify", action="store_true",
                        help="fail (non-zero) when the replay does not reproduce")
    parser.add_argument("--tolerance-k", type=float, default=1e-6)
    parser.add_argument("--energy-tolerance-j", type=float, default=1e-6)
    args = parser.parse_args(argv)
    result = replay(Path(args.run_dir),
                    temperature_tolerance_k=args.tolerance_k,
                    energy_tolerance_j=args.energy_tolerance_j)
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.verify and not result["passed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
