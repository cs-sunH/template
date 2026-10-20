#!/usr/bin/env python3
"""thermal_acceptance.py -- runnable acceptance matrix for the embedded thermal core.

Runs the checks that are possible *offline* in this environment and reports them
as pass/fail, while explicitly marking the checks that need the native online
binary / `libjoint_projection.so` as NOT RUN with the reason.  The suite never
folds a skipped native or business test into "all passed".

Usage::

    python sh_test_mesh/run_scripts/thermal_acceptance.py --mode shadow
    python sh_test_mesh/run_scripts/thermal_acceptance.py --suite all

Exit code is non-zero when a runnable check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_THERMAL_DIR = _REPO_ROOT / "extern" / "thermal_cosim"
_LLAMA_DIR = _REPO_ROOT / "sh_test_mesh" / "workload" / "llama2_7b_inference"
_RUN_SCRIPTS = _REPO_ROOT / "sh_test_mesh" / "run_scripts"
_ACCEPTANCE_SCHEMA = "joint.thermal-acceptance.v1"

for _path in (_THERMAL_DIR, _LLAMA_DIR, _RUN_SCRIPTS):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _run_unittest(target: str) -> dict:
    """Run one unittest target in a subprocess and capture pass/fail counts."""
    return _run_unittest_with_env([target], {})


def _run_unittest_with_env(targets: list, env: dict) -> dict:
    """Run unittest targets with extra env (native library paths) and capture."""
    command = [sys.executable, "-m", "unittest", *targets]
    environment = dict(os.environ)
    environment.update(env)
    completed = subprocess.run(
        command, cwd=str(_LLAMA_DIR), capture_output=True, text=True,
        env=environment)
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return {
        "target": " ".join(targets),
        "returncode": completed.returncode,
        "tail": output[-3:],
    }


def _run_thermal_core_suite() -> dict:
    command = [sys.executable, "-m", "unittest", "discover",
               "-s", "tests", "-p", "test_*.py"]
    completed = subprocess.run(
        command, cwd=str(_THERMAL_DIR), capture_output=True, text=True)
    output = (completed.stdout + completed.stderr).strip().splitlines()
    return {"target": "extern/thermal_cosim/tests", "returncode": completed.returncode,
            "tail": output[-3:]}


def _three_mode_evidence(output_dir: Path, mode: str) -> dict:
    """Drive the synthetic fixture through one thermal mode and record evidence."""
    from wafercosim.fixtures import reference_thermal_config, synthetic_thermal_facts_export

    from joint.thermal_bridge import (
        CoarseCapacityInstaller, ThermalBridge, ThermalRunConfig,
        build_fixture_mapping, hardware_snapshot_from_facts, synthetic_rank_to_die,
    )
    from joint.thermal_policy import ThermalPolicy, ThermalPolicyConfig
    from wafercosim.contracts import ActivityWindow

    class _RecorderRuntime:
        """Offline stand-in for the coarse runtime's capacity ABI."""

        def __init__(self):
            self.calls = []

        def update_capacity(self, resource, rate, now_ns):
            self.calls.append({"resource": resource, "rate": rate, "now_ns": now_ns})

    hardware = hardware_snapshot_from_facts(synthetic_thermal_facts_export())
    thermal_config = reference_thermal_config()
    mapping = build_fixture_mapping(
        hardware, thermal_config, mapping_epoch="thermal-fixture-v1",
        rank_to_die=synthetic_rank_to_die(54, 8))
    policy = ThermalPolicy(ThermalPolicyConfig.from_json(
        json.loads((_LLAMA_DIR / "thermal" / "synthetic_policy.json").read_text())))
    config = ThermalRunConfig(
        mode=mode,
        thermal_config_path="package:thermal/3dice_8die_reference.json",
        energy_table_path="configs/energy/research-v1.json",
        energy_strict=True, policy_path=str(_LLAMA_DIR / "thermal" / "synthetic_policy.json"),
        mapping_epoch="thermal-fixture-v1", thermal_step_ns=10_000_000,
        base_frequency_hz=1.0e9, accept_assumed_energy=True)
    runtime = _RecorderRuntime() if mode == "closed_loop" else None
    installer = (CoarseCapacityInstaller(
        runtime, domain_to_rank={"die0": 0, "die1": 1, "die2": 2, "die3": 3})
        if runtime is not None else None)
    bridge = ThermalBridge(run_id=f"acceptance-{mode}", config=config,
                           hardware=hardware, mapping=mapping, policy=policy,
                           installer=installer)
    temperatures: list[float] = []
    envelopes = []
    for seq in range(6):
        window = ActivityWindow(
            run_id=bridge.run_id, activity_seq=seq,
            start_ns=seq * 10_000_000, end_ns=(seq + 1) * 10_000_000,
            hardware_hash=hardware.content_hash, mapping_epoch=mapping.mapping_epoch,
            resource_epoch=bridge.resource_epoch,
            actions=(("die0.compute", "pe", 4.0e11),
                     ("die0.compute", "sram_read", 4.0e11)),
            dwell=(("die0.compute", 10_000_000, 0),),
            coverage="measured", provenance="acceptance-fixture")
        result = bridge.ingest(window)
        if result:
            temperatures.append(result["feedback"].die_block_max_k[0])
            envelopes.append(result["envelope"])
    run_dir = output_dir / mode
    bridge.write_audit(run_dir)
    gate_rates = ([c["rate"] for c in runtime.calls if c["resource"] == 0]
                  if runtime is not None else [])
    return {
        "mode": mode,
        "run_dir": str(run_dir),
        "final_die0_k": temperatures[-1] if temperatures else None,
        "de_rated": any(e.domain_fraction("die0") == 0.5 for e in envelopes),
        "applied": any(row["applied"] for e in envelopes for row in e.domains),
        "installed_compute_gate_rates": gate_rates,
    }


def build_matrix(output_dir: Path) -> dict:
    os.makedirs(output_dir, exist_ok=True)
    checks: dict[str, dict] = {}

    core = _run_thermal_core_suite()
    checks["V01_standalone_package"] = {
        "status": "pass" if core["returncode"] == 0 else "fail",
        "evidence": [str(_THERMAL_DIR / "tests")], "detail": core,
    }
    contract = _run_unittest("joint.test_thermal_bridge")
    checks["V03_V04_V05_V06_contracts"] = {
        "status": "pass" if contract["returncode"] == 0 else "fail",
        "evidence": [str(_LLAMA_DIR / "joint" / "test_thermal_bridge.py")],
        "detail": contract,
    }

    modes = {mode: _three_mode_evidence(output_dir, mode)
             for mode in ("off", "shadow", "closed_loop")}
    checks["V13_three_modes"] = {
        "status": "pass" if (
            modes["shadow"]["applied"] is False
            and modes["closed_loop"]["applied"] is True
            and 0.5 in modes["closed_loop"]["installed_compute_gate_rates"]
            and modes["off"]["final_die0_k"] is None) else "fail",
        "evidence": [str(output_dir)],
        "detail": {**modes, "closed_loop_install_note": (
            "the 0.5 COMPUTE_GATE install was recorded against an offline runtime "
            "recorder; the C++ runtime ABI call is exercised but the simulated "
            "finish-time change still needs the online build (V08)")},
    }

    from thermal_replay import replay

    replay_result = replay(output_dir / "closed_loop")
    checks["V14_replay_convergence"] = {
        "status": "pass" if replay_result["passed"] else "fail",
        "evidence": [replay_result["run_dir"]],
        "detail": {k: replay_result[k] for k in (
            "windows", "max_temperature_delta_k", "energy_residual_j")},
    }

    # Native-dependent checks: they run for real when the build tree exists, and
    # are reported as not_run (never as passed) when it does not.
    lib = _REPO_ROOT / "build" / "astra_analytical" / "build_congestion_aware" / "lib"
    native_env = {
        "NATIVE_COARSE_PROJECTION_LIBRARY": str(lib / "libjoint_projection.so"),
        "DIE_COARSE_CABI_PATH": str(lib / "libcoarse_die_cabi.so"),
        "DIE_MODEL_CABI_PATH": str(lib / "libdie_model_cabi.so"),
    }
    native_present = all(Path(value).is_file() for value in native_env.values())

    def _native_check(name, targets, note):
        if not native_present:
            checks[name] = {
                "status": "not_run",
                "reason": "build tree absent; build "
                          "build/astra_analytical/build_congestion_aware first",
            }
            return
        detail = _run_unittest_with_env(targets, native_env)
        checks[name] = {
            "status": "pass" if detail["returncode"] == 0 else "fail",
            "evidence": targets, "detail": detail, "note": note,
        }

    _native_check(
        "V02_V11_capacity_events",
        ["joint.test_thermal_hardware_export",
         "joint.test_native_coarse_projection.CoarseCapacityEventWireTest",
         "joint.test_native_coarse_projection.CoarseCapacityProjectionTest"],
        "real libcoarse_die_cabi.so export + real libjoint_projection.so capacity "
        "events (cap defers the predicted finish, recovery restores it)")
    _native_check(
        "V07_D_protocol_epoch",
        ["online.test_coarse_runtime_abi", "joint.test_thermal_epoch",
         "online.test_thermal_service_wiring"],
        "real runtime ABI cap install + THERMAL_EPOCH request/response handling")
    _native_check(
        "V16_joint_regression",
        ["joint.test_joint_telemetry_divisor.DivisorEffectiveTest",
         "joint.test_acc_predictor_cost_accounting.SnapshotContentTest"],
        "joint Python regression subset (plan appendix B)")
    checks.setdefault("V10_V13_online_closed_loop", {
        "status": "not_run",
        "reason": "the online three-arm run is driven by the docs' command chain "
                  "(off/shadow equal, closed_loop throttled); it is not part of "
                  "this offline matrix so the suite stays single-command fast",
    })

    passed = sum(1 for c in checks.values() if c["status"] == "pass")
    failed = sum(1 for c in checks.values() if c["status"] == "fail")
    return {
        "schema_version": _ACCEPTANCE_SCHEMA,
        "output_dir": str(output_dir),
        "passed": passed,
        "failed": failed,
        "not_run": sum(1 for c in checks.values() if c["status"] == "not_run"),
        "checks": checks,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Thermal co-simulation acceptance matrix")
    parser.add_argument("--mode", choices=["off", "shadow", "closed_loop", "all"],
                        default="all")
    parser.add_argument("--suite", default="all")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args(argv)
    if args.mode != "all":
        print(f"note: --mode {args.mode} selects one evidence arm; the matrix "
              "always records all three modes", file=sys.stderr)
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(tempfile.mkdtemp(prefix="thermal_acceptance_"))
    matrix = build_matrix(output_dir)
    (output_dir / "acceptance_summary.json").write_text(
        json.dumps(matrix, indent=2, sort_keys=True) + "\n")
    print(json.dumps(matrix, indent=2, sort_keys=True))
    if matrix["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
