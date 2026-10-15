#!/usr/bin/env python3
"""Run the frozen six-shape OS service-curve comparison into this directory."""
from __future__ import annotations

import hashlib
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[4]
OUTPUT = ROOT / "work/analytical-curves/axis"
OUTPUT.mkdir(parents=True, exist_ok=True)
DRIVER = ROOT / "work/build-die/coarse_compiled_gemm_oracle_driver"
ORACLE = ROOT / "die/validation/analytical_service_oracle.py"
REUSE_80_128_4096 = ROOT / "work/analytical-curves/os-80-128-4096.raw.json"
HISTORICAL_80_128_4096 = ROOT / "work/analytical-curves/os-80-128-4096.json"
CASES = (
    (16, 64, 64),
    (16, 64, 4096),
    (80, 128, 4096),
    (65, 65, 4097),
    (256, 64, 64),
    (16, 64, 6400),
)
TILES = (16, 64, 64)


def sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def current_identity() -> dict[str, Any]:
    return {
        "driver_binary_sha256": sha256(DRIVER),
        "driver_source_sha256": sha256(ROOT / "die/tests/compiled_gemm_oracle_driver.cc"),
        "oracle_source_sha256": sha256(ORACLE),
        "grouped_oracle_source_sha256": sha256(ROOT / "die/validation/grouped_gemm_oracle.py"),
        "compiled_oracle_reference_sha256": sha256(ROOT / "die/validation/compiled_gemm_oracle.py"),
    }


def case_name(shape: tuple[int, int, int]) -> str:
    return f"m{shape[0]}_n{shape[1]}_k{shape[2]}_tm{TILES[0]}_tn{TILES[1]}_tk{TILES[2]}"


def driver_args(shape: tuple[int, int, int]) -> list[str]:
    return ["--m", str(shape[0]), "--n", str(shape[1]), "--k", str(shape[2]),
            "--tm", str(TILES[0]), "--tn", str(TILES[1]), "--tk", str(TILES[2])]


def validate_raw(path: Path, shape: tuple[int, int, int]) -> tuple[bool, str]:
    try:
        raw = json.loads(path.read_text())
        plan = raw["compiled_plan"]
        summary = plan["summary"]
        got_shape = plan["task"]["tensor_shapes"]
        if raw.get("measurement_source") != "compiled_plan_and_actual_cpp_runtime":
            return False, "measurement source is not the actual C++ Runtime"
        if tuple(int(got_shape[key]) for key in ("m", "n", "k")) != shape:
            return False, "raw shape does not match requested case"
        if tuple(int(summary[key]) for key in ("tm", "tn", "tk")) != TILES:
            return False, "selected tiles do not match requested OS candidate"
        if summary.get("dataflow_id") != "output_stationary_v2":
            return False, "raw candidate is not output stationary"
        if len(raw.get("two_comp", [])) != 2 or "standalone" not in raw:
            return False, "raw lacks the driver's solo/synchronous/offset scenarios"
        offsets = sorted(float(item.get("phase_offset_s")) for item in raw["two_comp"])
        if offsets != [0.0, 0.05]:
            return False, "raw does not contain the expected 0 ms and 50 ms concurrent scenarios"
        selection = raw.get("mapping_selection", {})
        if selection.get("source") != "actual_detail::enumerate_mappings_candidate":
            return False, "raw did not select an actual Mapping enumeration candidate"
        if int(plan.get("E", 0)) <= 0 or int(plan.get("description_count", 0)) <= 0:
            return False, "raw lacks actual epoch/description counts"
        if not plan.get("service_model_provenance", {}).get("service_curves"):
            return False, "raw lacks actual service model provenance"
        return True, "validated actual compiled OS Runtime raw"
    except (OSError, ValueError, KeyError, TypeError) as error:
        return False, f"cannot validate raw: {error}"


def acquire_raw(shape: tuple[int, int, int], expected_identity: dict[str, Any]) -> dict[str, Any]:
    name = case_name(shape)
    raw_path = OUTPUT / f"{name}.raw.json"
    stderr_path = OUTPUT / f"{name}.driver.stderr.txt"
    stdout_path = OUTPUT / f"{name}.driver.stdout.txt"
    meta_path = OUTPUT / f"{name}.meta.json"
    identity = current_identity()

    if identity != expected_identity:
        return {"path": raw_path, "reuse": None, "driver_return_code": None,
                "stderr_path": None, "stdout_path": None,
                "validation": "not run: driver/source/oracle identity differs from batch baseline",
                "driver_arguments": driver_args(shape), "driver_identity": identity,
                "identity_consistent": False, "raw_available": False}

    if raw_path.is_file():
        try:
            old_meta = json.loads(meta_path.read_text())
        except (OSError, ValueError):
            old_meta = {}
        valid, why = validate_raw(raw_path, shape)
        if valid and old_meta.get("identity") == identity:
            return {"path": raw_path, "reuse": "existing_axis_raw", "driver_return_code": 0,
                    "stderr_path": str(stderr_path) if stderr_path.exists() else None,
                    "stdout_path": str(stdout_path) if stdout_path.exists() else None,
                    "validation": why, "driver_arguments": driver_args(shape),
                    "driver_identity": identity, "identity_consistent": True,
                    "raw_available": True}

    before = current_identity()
    completed = subprocess.run([str(DRIVER), *driver_args(shape)],
                               check=False, capture_output=True, text=True)
    after = current_identity()
    # Preserve the exact process streams even on a relink or driver failure.
    stdout_path.write_text(completed.stdout)
    stderr_path.write_text(completed.stderr)
    if completed.returncode != 0:
        result = {"path": raw_path, "reuse": None,
                  "driver_return_code": completed.returncode,
                  "stderr_path": str(stderr_path), "stdout_path": str(stdout_path),
                  "validation": "driver failed; original stdout/stderr preserved",
                  "driver_arguments": driver_args(shape),
                  "driver_identity": before,
                  "identity_consistent": before == after,
                  "raw_available": False}
        meta_path.write_text(json.dumps({"identity": before, "identity_after": after,
                                         "driver_return_code": completed.returncode,
                                         "stderr_path": str(stderr_path)}, indent=2) + "\n")
        return result
    try:
        json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        result = {"path": raw_path, "reuse": None,
                  "driver_return_code": completed.returncode,
                  "stderr_path": str(stderr_path), "stdout_path": str(stdout_path),
                  "validation": f"driver stdout was not JSON: {error}",
                  "driver_arguments": driver_args(shape),
                  "driver_identity": before,
                  "identity_consistent": before == after,
                  "raw_available": False}
        meta_path.write_text(json.dumps({"identity": before, "identity_after": after,
                                         "driver_return_code": completed.returncode,
                                         "stderr_path": str(stderr_path)}, indent=2) + "\n")
        return result
    raw_path.write_text(completed.stdout)
    valid, why = validate_raw(raw_path, shape)
    identity_consistent = before == after == expected_identity
    if not identity_consistent:
        why = "driver/source/oracle identity changed during batch invocation; raw preserved but comparison not run"
        valid = False
    result = {"path": raw_path, "reuse": None,
              "driver_return_code": completed.returncode,
              "stderr_path": str(stderr_path), "stdout_path": str(stdout_path),
              "validation": why if valid else f"raw rejected: {why}",
              "driver_arguments": driver_args(shape),
              "driver_identity": before,
              "identity_consistent": identity_consistent,
              "raw_available": True}
    meta_path.write_text(json.dumps({"identity": before, "identity_after": after,
                                     "driver_return_code": completed.returncode,
                                     "raw_validation": result["validation"]}, indent=2) + "\n")
    return result


def invoke_oracle(raw_path: Path, report_path: Path) -> tuple[int, str]:
    proc = subprocess.run([sys.executable, str(ORACLE), "--raw", str(raw_path.resolve()),
                           "--output", str(report_path.resolve())],
                          check=False, capture_output=True, text=True)
    stderr_path = report_path.with_suffix(".oracle.stderr.txt")
    stderr_path.write_text(proc.stderr)
    return proc.returncode, str(stderr_path)


def report_summary(report_path: Path, case: tuple[int, int, int],
                   acquisition: dict[str, Any], oracle_return: int,
                   oracle_stderr: str) -> dict[str, Any]:
    result: dict[str, Any] = {
        "shape": {"m": case[0], "n": case[1], "k": case[2]},
        "requested_tiles": {"tm": TILES[0], "tn": TILES[1], "tk": TILES[2]},
        "raw_json": str(acquisition["path"]) if acquisition.get("raw_available") else None,
        "expected_raw_path": str(acquisition["path"]),
        "reuse_source": acquisition["reuse"],
        "driver_return_code": acquisition["driver_return_code"],
        "driver_arguments": acquisition["driver_arguments"],
        "driver_identity": acquisition.get("driver_identity"),
        "driver_identity_consistent": acquisition.get("identity_consistent", False),
        "driver_stderr_path": acquisition["stderr_path"],
        "driver_stdout_path": acquisition["stdout_path"],
        "raw_validation": acquisition["validation"],
        "oracle_return_code": oracle_return,
        "oracle_stderr_path": oracle_stderr,
        "report_json": str(report_path),
    }
    if not report_path.is_file():
        result.update(status="not_run", reason="oracle report file was not produced")
        return result
    report = json.loads(report_path.read_text())
    result["status"] = report.get("status", "not_run")
    result["report_status_reason"] = report.get("status_reason", report.get("reason"))
    result["acceptance_limits"] = report.get("acceptance_limits")
    result["available_granularity_levels"] = report.get("available_granularity_levels")
    result["fixture_coverage"] = report.get("fixture_coverage")
    result["status_scope"] = report.get("status_scope")
    result["measurement_source"] = report.get("measurement_source")
    plan = report.get("compiled_plan")
    if isinstance(plan, dict):
        result["actual_plan"] = {
            "fingerprint": plan.get("fingerprint"),
            "E": plan.get("E"),
            "description_count": plan.get("description_count"),
            "frameworks": plan.get("service_model_provenance", {}).get("observed_frameworks"),
            "tiles": plan.get("tiles"),
            "service_model_provenance": plan.get("service_model_provenance"),
        }
    if result["status"] != "passed" and result["status"] != "failed":
        return result

    def compact(comparison: dict[str, Any]) -> dict[str, Any]:
        actual = comparison["actual_runtime"]
        reference = comparison["independent_discrete_oracle"]
        absolute = comparison["absolute_errors_and_shared_reference"]
        relative = comparison["relative_errors"]
        return {
            "status": comparison["status"],
            "phase_offset_s": comparison["phase_offset_s"],
            "actual_finish_s_by_request": actual["finish_s_by_request"],
            "reference_finish_s_by_request": reference["finish_s_by_request"],
            "finish_relative_error_by_request": relative["finish_by_request"],
            "finish_absolute_error_s_by_request": absolute["finish_time_s_by_request"],
            "actual_hbm_busy_s": actual["hbm_busy_s"],
            "reference_hbm_busy_s": reference["hbm_busy_s"],
            "hbm_busy_relative_error": relative["hbm_busy"],
            "hbm_busy_absolute_error_s": absolute["hbm_busy_s"],
            "actual_shared_hbm_busy_s": actual["hbm_shared_s"],
            "reference_shared_hbm_busy_s": reference["hbm_shared_s"],
            "shared_hbm_busy_relative_error": relative["shared_hbm_busy"],
            "shared_hbm_busy_absolute_error_s": absolute["shared_hbm_busy_s"],
            "reference_shared_hbm_duration_s": absolute["reference_shared_hbm_duration_s"],
            "internal_events": actual["internal_events"],
            "allocation_updates": actual["allocation_updates"],
        }

    comparisons = report["comparisons"]
    result["solo"] = compact(comparisons["standalone"])
    result["concurrent"] = [compact(item) for item in comparisons["two_comp"]]
    return result


def format_sci(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3g}"


def max_value(value: Any) -> Any:
    if isinstance(value, dict):
        return max((abs(float(x)) for x in value.values()), default=None)
    return value


def error_pair(item: dict[str, Any], relative: str, absolute: str) -> str:
    return (f"{format_sci(max_value(item.get(relative)))} / "
            f"{format_sci(max_value(item.get(absolute)))}s")


def historical_initial_metrics() -> dict[str, Any] | None:
    """Load the pre-final 80x128 report only as labeled historical context."""
    if not HISTORICAL_80_128_4096.is_file():
        return None
    try:
        report = json.loads(HISTORICAL_80_128_4096.read_text())
        comparisons = report.get("comparisons", {})
        solo = comparisons.get("standalone", {})
        concurrent = comparisons.get("two_comp", [])
        sync = next((x for x in concurrent if x.get("phase_offset_s") == 0.0), {})
        offset = next((x for x in concurrent if x.get("phase_offset_s") == 0.05), {})
        return {
            "label": "historical initial-model result; excluded from final-revision batch",
            "raw_json": str(REUSE_80_128_4096),
            "report_json": str(HISTORICAL_80_128_4096),
            "status": report.get("status"),
            "plan_E": report.get("compiled_plan", {}).get("E"),
            "description_count": report.get("compiled_plan", {}).get("description_count"),
            "description_budget": next((
                item.get("description_budget") for item in
                report.get("available_granularity_levels", {}).get("observed", [])
                if isinstance(item, dict)), None),
            "solo_finish_relative_error": max_value(solo.get("relative_errors", {}).get("finish_by_request")),
            "solo_hbm_busy_relative_error": solo.get("relative_errors", {}).get("hbm_busy"),
            "sync_hbm_busy_relative_error": sync.get("relative_errors", {}).get("hbm_busy"),
            "offset_shared_hbm_busy_relative_error": offset.get("relative_errors", {}).get("shared_hbm_busy"),
            "offset_reference_shared_hbm_duration_s": offset.get("absolute_errors_and_shared_reference", {}).get("reference_shared_hbm_duration_s"),
        }
    except (OSError, ValueError, TypeError):
        return {"label": "historical initial-model report exists but could not be parsed",
                "report_json": str(HISTORICAL_80_128_4096),
                "excluded_from_current_batch": True}


def markdown_summary(summary: dict[str, Any]) -> str:
    lines = [
        "# Analytical OS service curve: six-shape axis check",
        "",
        "Each row compares the actual selected OS Plan and Runtime with the existing grouped discrete K-wave oracle.",
        "The fixture is single-Core 1x1, FP32, with synthetic finite resource capacities. It is model-to-model evidence; it contains no silicon measurements or target calibration.",
        "Description counts and available framework rows are read from each actual serialized Plan; consistency is checked by summing each epoch's activity segment counts, with empty curves and empty epochs contributing one description.",
        "Acceptance limits are copied per case from each actual oracle report; the original standalone completion, concurrent completion, HBM busy, and shared HBM busy criteria are used unchanged.",
        "",
        "| Shape M×N×K | Plan E / descriptions | Status | Solo finish rel. / abs. | Solo HBM busy rel. / abs. | Sync finish rel. / abs. | Sync HBM busy rel. / abs. | Sync shared busy rel. / abs. / ref. duration | Offset finish rel. / abs. | Offset HBM busy rel. / abs. | Offset shared busy rel. / abs. / ref. duration | Events / allocation updates (solo; sync; offset) |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for case in summary["cases"]:
        plan = case.get("actual_plan", {})
        solo = case.get("solo", {})
        concurrent = case.get("concurrent", [])
        sync = next((x for x in concurrent if x.get("phase_offset_s") == 0.0), {})
        offset = next((x for x in concurrent if x.get("phase_offset_s") == 0.05), {})
        event_triplet = []
        for item in [solo, sync, offset]:
            if item:
                event_triplet.append(f"{item.get('internal_events', 'n/a')}/{item.get('allocation_updates', 'n/a')}")
            else:
                event_triplet.append("n/a")
        s_shape = case["shape"]
        shape_label = f"{s_shape['m']}×{s_shape['n']}×{s_shape['k']}"
        lines.append(
            f"| {shape_label} | {plan.get('E', 'n/a')} / {plan.get('description_count', 'n/a')} "
            f"| {case['status']} | {error_pair(solo, 'finish_relative_error_by_request', 'finish_absolute_error_s_by_request')} "
            f"| {format_sci(solo.get('hbm_busy_relative_error'))} / {format_sci(solo.get('hbm_busy_absolute_error_s'))}s "
            f"| {error_pair(sync, 'finish_relative_error_by_request', 'finish_absolute_error_s_by_request')} "
            f"| {format_sci(sync.get('hbm_busy_relative_error'))} / {format_sci(sync.get('hbm_busy_absolute_error_s'))}s "
            f"| {format_sci(sync.get('shared_hbm_busy_relative_error'))} / {format_sci(sync.get('shared_hbm_busy_absolute_error_s'))}s / {format_sci(sync.get('reference_shared_hbm_duration_s'))}s "
            f"| {error_pair(offset, 'finish_relative_error_by_request', 'finish_absolute_error_s_by_request')} "
            f"| {format_sci(offset.get('hbm_busy_relative_error'))} / {format_sci(offset.get('hbm_busy_absolute_error_s'))}s "
            f"| {format_sci(offset.get('shared_hbm_busy_relative_error'))} / {format_sci(offset.get('shared_hbm_busy_absolute_error_s'))}s / {format_sci(offset.get('reference_shared_hbm_duration_s'))}s "
            f"| {'; '.join(event_triplet)} |"
        )
    lines.extend([
        "",
        "## Reading the result",
        "",
        "Completion errors and HBM busy errors are reported separately. A smooth cumulative service curve can keep completion close to the discrete oracle while extending the interval classified as HBM busy; the table preserves the original busy integral and exposes the absolute error and reference duration.",
        "",
        "The tool only covers these six single-Core FP32 OS points. It does not certify other shapes, dataflows, core grids, hardware, or minimum description counts. A failed precision status remains failed.",
        "",
        f"Historical initial-model 80×128×4096 result is excluded from this batch: `{json.dumps(summary.get('historical_initial_model_reference'), sort_keys=True)}`.",
        "",
        f"Machine summary: `{summary['summary_json']}`. Per-case actual raw JSON and oracle reports are beside this file.",
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--final-driver-ready", action="store_true",
                        help="required explicit guard after the final build is confirmed ready")
    args = parser.parse_args()
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if not args.final_driver_ready:
        parser.error("wait for the build owner to confirm the final driver, then pass --final-driver-ready")
    batch_identity = current_identity()
    identity_files_available = all(value is not None for value in batch_identity.values())
    if not identity_files_available:
        print("final driver/oracle identity incomplete; no cases were run", file=sys.stderr)
        return 2
    cases = []
    for shape in CASES:
        name = case_name(shape)
        raw_path = OUTPUT / f"{name}.raw.json"
        report_path = OUTPUT / f"{name}.report.json"
        acquisition = acquire_raw(shape, batch_identity)
        oracle_return = 2
        oracle_stderr = str(report_path.with_suffix(".oracle.stderr.txt"))
        report_ready = False
        if (acquisition["driver_return_code"] == 0 and acquisition.get("identity_consistent")
                and Path(acquisition["path"]).is_file()):
            valid, why = validate_raw(Path(acquisition["path"]), shape)
            acquisition["validation"] = why
            if valid:
                oracle_return, oracle_stderr = invoke_oracle(Path(acquisition["path"]), report_path)
                if report_path.is_file():
                    report = json.loads(report_path.read_text())
                    report.setdefault("entry_parameters", {})[
                        "driver_arguments"] = acquisition["driver_arguments"]
                    report["entry_parameters"]["requested_shape"] = {
                        "m": shape[0], "n": shape[1], "k": shape[2]}
                    report["entry_parameters"]["requested_tiles"] = {
                        "tm": TILES[0], "tn": TILES[1], "tk": TILES[2]}
                    report["entry_parameters"]["actual_driver_binary"] = str(DRIVER)
                    report["entry_parameters"]["driver_identity"] = acquisition.get("driver_identity")
                    report["entry_parameters"]["raw_json"] = str(Path(acquisition["path"]).resolve())
                    report["entry_parameters"]["raw_reuse_source"] = acquisition["reuse"]
                    report["entry_parameters"]["oracle_arguments"] = [
                        sys.executable, str(ORACLE), "--raw", str(Path(acquisition["path"]).resolve()),
                        "--output", str(report_path.resolve())]
                    report_path.write_text(json.dumps(report, indent=2, sort_keys=True,
                                                       allow_nan=False) + "\n")
                    report_ready = True
        if not report_ready:
            report_path.write_text(json.dumps({
                "status": "not_run",
                "reason": acquisition["validation"],
                "driver_return_code": acquisition["driver_return_code"],
                "driver_stderr_path": acquisition["stderr_path"],
            }, indent=2, sort_keys=True) + "\n")
        cases.append(report_summary(report_path, shape, acquisition,
                                    oracle_return, oracle_stderr))
        print(f"{shape}: driver={acquisition['driver_return_code']} "
              f"oracle={cases[-1]['status']} ({acquisition['validation']})")

    summary_path = OUTPUT / "summary.json"
    executed_statuses = [item["status"] for item in cases]
    if all(status in ("passed", "failed") for status in executed_statuses):
        batch_status = "complete"
    elif any(status in ("passed", "failed") for status in executed_statuses):
        batch_status = "partial_identity_or_acquisition_failure"
    else:
        batch_status = "not_run"
    summary = {
        "schema_version": 1,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "entry_parameters": {
            "script": str(Path(__file__).resolve()),
            "driver_binary": str(DRIVER),
            "final_driver_ready_guard": args.final_driver_ready,
            "requested_shapes": [{"m": m, "n": n, "k": k} for m, n, k in CASES],
            "requested_tiles": {"tm": TILES[0], "tn": TILES[1], "tk": TILES[2]},
        },
        "batch_status": batch_status,
        "scope": {
            "fixture": "single-Core 1x1, FP32, synthetic finite capacities",
            "hardware_measurement": False,
            "target_calibration": False,
            "oracle": "analytical_service_oracle.py + existing grouped discrete K-wave oracle",
            "acceptance_limits_source": "copied from each per-case analytical_service_oracle.py report",
            "description_count_rule": "per epoch max(1, sum over activities of max(1, credit_curve_points-1) for nonempty curves, otherwise 1)",
        },
        "identity": {
            "driver_binary": str(DRIVER),
            "batch_baseline": batch_identity,
            "identity_stable_through_batch": current_identity() == batch_identity,
        },
        "requested_tiles": {"tm": TILES[0], "tn": TILES[1], "tk": TILES[2]},
        "cases": cases,
        "historical_initial_model_reference": historical_initial_metrics(),
        "summary_json": str(summary_path),
        "readable_report": str(OUTPUT / "README.md"),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True,
                                       allow_nan=False) + "\n")
    (OUTPUT / "README.md").write_text(markdown_summary(summary))
    if batch_status != "complete":
        return 2
    return 1 if any(item["status"] == "failed" for item in cases) else 0


if __name__ == "__main__":
    raise SystemExit(main())
