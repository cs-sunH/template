#!/usr/bin/env python3
"""Run compiled OS GEMM single-factor probes, the grouped oracle, and default check."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from decimal import Decimal
from pathlib import Path

ROOT = next((parent for parent in Path(__file__).resolve().parents
             if (parent / "die/src/Mapping.cc").is_file()), None)
if ROOT is None:
    raise RuntimeError("cannot locate the repository root containing die/src/Mapping.cc")
SOURCE = ROOT / "die"
sys.path.insert(0, str(SOURCE / "validation"))
import grouped_gemm_oracle as grouped  # noqa: E402

PROBES = (
    ("m80n128_k128", "k_depth", 80, 128, 128),
    ("full_tile_k4096", "k_depth", 80, 128, 4096),
    ("full_tile_k4097", "k_tail", 80, 128, 4097),
    ("mn_edge_k4097", "mn_shape", 65, 65, 4097),
    ("default_compatibility", "default_behavior", None, None, None),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plan_work_reconciliation(raw: dict) -> dict:
    """Report serialized runtime work against integer plan activities only.

    This fallback is useful when the selected template is outside the OS
    wave-oracle preconditions. Activity totals already include each epoch's
    repeated work; repeat_count is audit metadata and must not multiply them.
    """
    plan = raw["compiled_plan"]
    expected_hbm = [[0, 0], [0, 0]]
    plan_expected_vec = 0
    for epoch in plan["epochs"]:
        for activity in epoch["activities"]:
            for row in range(2):
                for direction in range(2):
                    expected_hbm[row][direction] += int(
                        activity["hbm_bytes_by_row_direction"][row][direction])
            plan_expected_vec += int(activity["vec_eqops"])

    cases = []
    modes = [("standalone", raw["standalone"], 1)]
    modes.extend(("synchronous_two_comp" if index == 0 else "offset_50ms_two_comp",
                  item, 2) for index, item in enumerate(raw["two_comp"]))
    for name, measurement, request_count in modes:
        counters = measurement["counters"]
        actual_hbm = counters.get("hbm_served_bytes_by_row_direction_exact")
        categories = counters.get("served_by_category")
        missing = []
        if actual_hbm is None:
            missing.append("hbm_served_bytes_by_row_direction_exact")
        if categories is None or not categories or any(
                "vec_eqops_exact_decimal" not in category
                for category in categories.values()):
            missing.append("served_by_category.vec_eqops_exact_decimal")
        if missing:
            cases.append({"mode": name, "status": "not_available",
                          "missing_counter_fields": missing})
            continue

        expected = [[value * request_count for value in row]
                    for row in expected_hbm]
        hbm = []
        hbm_strict = True
        hbm_within_tolerance = True
        for row in range(2):
            directions = []
            for direction in range(2):
                actual = Decimal(str(actual_hbm[row][direction]))
                wanted = Decimal(expected[row][direction])
                delta = actual - wanted
                relative = abs(delta) / max(Decimal(1), abs(wanted))
                equal = delta == 0
                hbm_strict = hbm_strict and equal
                hbm_within_tolerance = hbm_within_tolerance and relative <= Decimal("1e-9")
                directions.append({
                    "actual_serialized_decimal": str(actual),
                    "expected_integer": expected[row][direction],
                    "signed_delta": str(delta),
                    "absolute_delta": str(abs(delta)),
                    "relative_delta": str(relative),
                    "strict_integer_equality": equal,
                    "within_relative_tolerance": relative <= Decimal("1e-9"),
                })
            hbm.append(directions)
        actual_vec = sum((Decimal(category["vec_eqops_exact_decimal"])
                          for category in categories.values()), Decimal(0))
        expected_vec_total = plan_expected_vec * request_count
        vec_delta = actual_vec - Decimal(expected_vec_total)
        vec_relative = abs(vec_delta) / max(Decimal(1), Decimal(expected_vec_total))
        history_complete = counters.get("accounting_history_complete")
        within_tolerance = hbm_within_tolerance and vec_relative <= Decimal("1e-9")
        reconciliation_status = (
            "incomplete_history" if history_complete is False else
            "within_relative_tolerance" if within_tolerance else
            "outside_relative_tolerance")
        cases.append({
            "mode": name,
            "status": reconciliation_status,
            "strict_integer_equality": hbm_strict and vec_delta == 0,
            "within_relative_tolerance": within_tolerance,
            "relative_tolerance": "1E-9",
            "counter_arithmetic": "serialized_cumulative_long_double_decimal",
            "expected_arithmetic": "compiled_plan_integer_activity_totals",
            "accounting_history_complete": history_complete,
            "hbm_bytes_by_row_direction": hbm,
            "vec_equivalent_ops": {
                "actual_serialized_decimal": str(actual_vec),
                "expected_integer": expected_vec_total,
                "signed_delta": str(vec_delta),
                "absolute_delta": str(abs(vec_delta)),
                "relative_delta": str(vec_relative),
                "strict_integer_equality": vec_delta == 0,
                "within_relative_tolerance": vec_relative <= Decimal("1e-9"),
            },
        })

    plan_summary = plan["summary"]
    plan_activity_hbm_totals = [sum(row) for row in expected_hbm]
    plan_summary_checks = {
        "hbm_read_bytes_total": sum(row[0] for row in expected_hbm) ==
                                int(plan_summary["hbm_read_bytes_total"]),
        "hbm_write_bytes_total": sum(row[1] for row in expected_hbm) ==
                                 int(plan_summary["hbm_write_bytes_total"]),
        "vec_equivalent_ops": plan_expected_vec == int(plan_summary["vec_equivalent_ops"]),
    }
    return {
        "scope": "compiled_plan_activity_vs_runtime_only_no_discrete_os_oracle",
        "planned_integer_conservation": {
            "status": "passed" if all(plan_summary_checks.values()) else "failed",
            "arithmetic": "exact_integer_equality",
            "checks": plan_summary_checks,
            "compiled_activity_hbm_total_by_row": plan_activity_hbm_totals,
            "compiled_activity_vec_equivalent_ops": plan_expected_vec,
        },
        "runtime_cumulative_serialization_reconciliation": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--driver", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    driver = args.driver.resolve()
    output = args.output_dir.resolve()
    raw_dir, report_dir, log_dir = (output / name for name in ("raw", "reports", "logs"))
    for path in (raw_dir, report_dir, log_dir):
        path.mkdir(parents=True, exist_ok=True)

    files_for_identity = [
        SOURCE / "tests/compiled_gemm_oracle_driver.cc",
        SOURCE / "validation/compiled_gemm_oracle.py",
        SOURCE / "validation/grouped_gemm_oracle.py",
        SOURCE / "src/Mapping.cc",
        SOURCE / "src/Compiler.cc",
        SOURCE / "src/Execution.cc",
        SOURCE / "src/Hardware.cc",
        SOURCE / "src/Plan.cc",
    ]
    identity = {
        "git_head": subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            check=True, capture_output=True, text=True).stdout.strip(),
        "driver_path": str(driver),
        "driver_sha256": sha256(driver),
        "runner_sha256": sha256(Path(__file__).resolve()),
        "source_sha256": {path.relative_to(ROOT).as_posix(): sha256(path)
                          for path in files_for_identity},
        "build_directory": str(driver.parent),
    }
    (output / "source_identity.json").write_text(
        json.dumps(identity, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    manifest = {"schema_version": 1, "source_identity": identity, "probes": []}
    failed = False
    for name, axis, m, n, k in PROBES:
        raw_path = raw_dir / f"{name}.json"
        report_path = report_dir / f"{name}.json"
        stderr_path = log_dir / f"{name}.driver.stderr.txt"
        command = [str(driver)]
        if m is not None:
            command.extend(("--m", str(m), "--n", str(n), "--k", str(k)))
        requested_shape = (None if m is None else {"m": m, "n": n, "k": k})
        completed = subprocess.run(command, capture_output=True, text=True,
                                   check=False)
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        case = {
            "name": name,
            "axis": axis,
            "requested_shape": requested_shape,
            "command": command,
            "driver_return_code": completed.returncode,
            "raw_json": str(raw_path),
            "report_json": str(report_path),
            "driver_stderr": str(stderr_path),
        }
        if completed.returncode != 0:
            failed = True
            case["status"] = "failed_before_oracle"
            report_path.write_text(json.dumps({
                "status": "failed",
                "failure": "compiled C++ driver exited nonzero",
                "driver_return_code": completed.returncode,
                "driver_stderr": completed.stderr,
                "probe": {"name": name, "axis": axis,
                          "requested_shape": requested_shape},
            }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        else:
            raw_path.write_text(completed.stdout, encoding="utf-8")
            raw = {}
            try:
                raw = json.loads(completed.stdout)
                report = grouped.build_report(raw)
                summary = raw["compiled_plan"]["summary"]
                shape = raw["compiled_plan"]["task"]["tensor_shapes"]
                report["probe"] = {
                    "name": name,
                    "axis": axis,
                    "requested_shape": {key: int(shape[key]) for key in ("m", "n", "k")},
                    "selected_tiles": {key: int(summary[key]) for key in ("tm", "tn", "tk")},
                    "selected_tile_signature": "x".join(
                        str(int(summary[key])) for key in ("tm", "tn", "tk")),
                }
                report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n",
                                       encoding="utf-8")
                case["status"] = report["status"]
                case["selected_tiles"] = report["probe"]["selected_tiles"]
                case["selected_template"] = summary["selected_template"]
                case["dataflow_id"] = summary["dataflow_id"]
                case["comparisons"] = {
                    "standalone": report["comparisons"]["standalone"]["status"],
                    "synchronous_two_comp": report["comparisons"]["two_comp"][0]["status"],
                    "offset_50ms_two_comp": report["comparisons"]["two_comp"][1]["status"],
                }
                failed = failed or report["status"] != "passed"
            except Exception as error:  # retain machine-readable failed evidence
                failed = True
                case["status"] = "failed_during_oracle"
                plan = raw.get("compiled_plan", {})
                summary = plan.get("summary", {})
                shape = plan.get("task", {}).get("tensor_shapes", {})
                case["selected_tiles"] = {
                    key: summary.get(key) for key in ("tm", "tn", "tk")}
                case["selected_template"] = summary.get("selected_template")
                case["dataflow_id"] = summary.get("dataflow_id")
                report_path.write_text(json.dumps({
                    "status": "failed",
                    "failure": f"{type(error).__name__}: {error}",
                    "driver_stderr": completed.stderr,
                    "driver_output_prefix": completed.stdout[:2000],
                    "compiled_plan": {
                        "shape": shape,
                        "selected_tiles": case["selected_tiles"],
                        "selected_template": case["selected_template"],
                        "dataflow_id": case["dataflow_id"],
                        "summary": summary,
                    },
                    "actual_runtime": {
                        "standalone": raw.get("standalone"),
                        "two_comp": raw.get("two_comp"),
                    },
                    "compiled_plan_runtime_work": (
                        plan_work_reconciliation(raw) if raw else None),
                    "probe": {"name": name, "axis": axis,
                              "requested_shape": requested_shape},
                }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest["probes"].append(case)
        print(json.dumps(case, sort_keys=True))

    probe_map = {case["name"]: case for case in manifest["probes"]}
    tile_pairs = (
        ("k_depth_128_vs_4096", "m80n128_k128", "full_tile_k4096"),
        ("k_tail_4096_vs_4097", "full_tile_k4096", "full_tile_k4097"),
        ("mn_shape_full_vs_edge", "full_tile_k4097", "mn_edge_k4097"),
    )
    manifest["single_factor_tile_checks"] = []
    for label, left_name, right_name in tile_pairs:
        left = probe_map[left_name]
        right = probe_map[right_name]
        same_tiles = left.get("selected_tiles") == right.get("selected_tiles")
        same_template = left.get("selected_template") == right.get("selected_template")
        same_dataflow = left.get("dataflow_id") == right.get("dataflow_id")
        manifest["single_factor_tile_checks"].append({
            "comparison": label,
            "left_probe": left_name,
            "right_probe": right_name,
            "same_selected_tiles": same_tiles,
            "same_selected_template": same_template,
            "same_dataflow": same_dataflow,
            "status": ("comparable_selected_plan" if
                       same_tiles and same_template and same_dataflow else
                       "not_comparable_compiler_reselected_mapping"),
        })

    manifest["status"] = "failed" if failed else "passed"
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
