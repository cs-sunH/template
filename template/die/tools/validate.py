#!/usr/bin/env python3
"""Run public-library checks and keep unrun evidence visibly unrun.

The offline wave oracle is independent of the C++ implementation. Sensitivity
and performance measurements call the built public C ABI/library and benchmark
binary; this script never replaces missing measurements with synthetic passes.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent


def run(command: list[str], cwd: Path) -> dict[str, Any]:
    started = time.perf_counter_ns()
    try:
        result = subprocess.run(command, cwd=cwd, capture_output=True, text=True,
                                check=False)
    except OSError as exc:
        return {
            "status": "not_run",
            "command": command,
            "reason": str(exc),
            "host_elapsed_ns": time.perf_counter_ns() - started,
        }
    status = "passed" if result.returncode == 0 else "failed"
    return {
        "status": status,
        "command": command,
        "return_code": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "host_elapsed_ns": time.perf_counter_ns() - started,
    }


def read_optional_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build-dir", type=Path, default=REPO / "work/build-die")
    parser.add_argument("--output", type=Path,
                        default=ROOT / "validation/validation_report.json")
    parser.add_argument("--ctest", action=argparse.BooleanOptionalAction,
                        default=True, help="run the configured CTest public-library suite")
    parser.add_argument("--oracle", action=argparse.BooleanOptionalAction,
                        default=True, help="run the independent offline wave oracle")
    parser.add_argument("--sensitivity", action=argparse.BooleanOptionalAction,
                        default=True, help="run the full 144-case C ABI sensitivity matrix")
    parser.add_argument("--benchmark", action=argparse.BooleanOptionalAction,
                        default=True, help="run real cold/hot/estimate/runtime measurements")
    parser.add_argument("--quick-sensitivity", action="store_true",
                        help="run the 8-case smoke matrix instead of all 144 cases")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--owners", type=int, default=8)
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    output = args.output.resolve()
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "die/tools/validate.py",
        "status": "incomplete",
        "repository_root": str(REPO),
        "build_dir": str(build_dir),
        "evidence": {},
        "notes": [
            "compile estimates retain granularity_validation=not_run until a certificate exists",
            "the discrete oracle is model-to-model evidence, not physical hardware validation",
            "not_run measurements are never counted as passing",
        ],
    }

    if args.ctest:
        ctest = shutil_which("ctest")
        if not (build_dir / "CTestTestfile.cmake").is_file() or not ctest:
            report["evidence"]["ctest"] = {
                "status": "not_run",
                "reason": "configured CTest build or ctest executable is unavailable",
            }
        else:
            report["evidence"]["ctest"] = run(
                [ctest, "--test-dir", str(build_dir), "--output-on-failure"],
                REPO,
            )
    else:
        report["evidence"]["ctest"] = {"status": "not_run", "reason": "disabled by caller"}

    if args.oracle:
        oracle_path = ROOT / "validation/discrete_oracle.py"
        oracle_report_path = ROOT / "validation/oracle_granularity_report.json"
        oracle_run = run(
            [sys.executable, str(oracle_path), "--output", str(oracle_report_path),
             "--runtime-driver", str(build_dir / "coarse_oracle_granularity_test")],
            REPO,
        )
        oracle_payload = read_optional_json(oracle_report_path)
        oracle_run["report_path"] = str(oracle_report_path)
        oracle_run["granularity_validation"] = (
            oracle_payload.get("granularity_validation", "not_run")
            if oracle_payload else "not_run"
        )
        oracle_run["selected_minimum_E"] = (
            oracle_payload.get("selected_minimum_E") if oracle_payload else None
        )
        oracle_run["scenario_count"] = (
            oracle_payload.get("scenario_count") if oracle_payload else None
        )
        if oracle_payload:
            oracle_run["status"] = oracle_payload.get("status", "failed")
        elif not oracle_payload and oracle_run["status"] == "passed":
            oracle_run["status"] = "failed"
            oracle_run["reason"] = "oracle process exited successfully without a readable report"
        report["evidence"]["discrete_oracle"] = oracle_run
    else:
        report["evidence"]["discrete_oracle"] = {
            "status": "not_run", "reason": "disabled by caller",
            "granularity_validation": "not_run",
        }

    cabi = build_dir / "lib/libcoarse_die_cabi.so"
    benchmark_bin = build_dir / "coarse_die_benchmark"
    if args.sensitivity:
        command = [
            sys.executable, str(ROOT / "tools/sensitivity.py"),
            "--library", str(cabi),
            "--config", str(ROOT / "tests/fixtures/coarse_die_v2.json"),
            "--output", str(ROOT / "validation/sensitivity_report.json"),
            "--markdown", str(ROOT / "validation/sensitivity_report.md"),
        ]
        if args.quick_sensitivity:
            command.append("--quick")
        sensitivity_run = run(command, REPO)
        sensitivity_payload = read_optional_json(ROOT / "validation/sensitivity_report.json")
        if sensitivity_payload:
            sensitivity_run["case_status"] = sensitivity_payload.get("status", "not_run")
            sensitivity_run["completed_cases"] = sensitivity_payload.get("completed_cases")
            sensitivity_run["expected_cases"] = sensitivity_payload.get("expected_cases")
            sensitivity_run["rankings"] = sensitivity_payload.get("rankings")
            sensitivity_run["status"] = sensitivity_payload.get("status", "failed")
        elif sensitivity_run["status"] == "passed":
            sensitivity_run["status"] = "failed"
            sensitivity_run["reason"] = "sensitivity process exited successfully without a readable report"
        report["evidence"]["sensitivity"] = sensitivity_run
    else:
        report["evidence"]["sensitivity"] = {
            "status": "not_run", "reason": "disabled by caller",
            "library_path": str(cabi),
        }

    if args.benchmark:
        command = [
            sys.executable, str(ROOT / "tools/benchmark.py"),
            "--benchmark-bin", str(benchmark_bin),
            "--config", str(ROOT / "tests/fixtures/coarse_die_v2.json"),
            "--task", str(ROOT / "tests/fixtures/gemm_medium.json"),
            "--iterations", str(args.iterations),
            "--owners", str(args.owners),
            "--output", str(ROOT / "validation/benchmark_report.json"),
            "--markdown", str(ROOT / "validation/benchmark_report.md"),
        ]
        benchmark_run = run(command, REPO)
        benchmark_payload = read_optional_json(ROOT / "validation/benchmark_report.json")
        if benchmark_payload:
            benchmark_run["measurement_status"] = benchmark_payload.get("status", "not_run")
            benchmark_run["benchmark_result"] = benchmark_payload.get("benchmark_result")
            benchmark_run["status"] = benchmark_payload.get("status", "failed")
        elif benchmark_run["status"] == "passed":
            benchmark_run["status"] = "failed"
            benchmark_run["reason"] = "benchmark process exited successfully without a readable report"
        report["evidence"]["benchmark"] = benchmark_run
    else:
        report["evidence"]["benchmark"] = {
            "status": "not_run", "reason": "disabled by caller",
            "benchmark_binary": str(benchmark_bin),
        }

    statuses = [item.get("status", "not_run") for item in report["evidence"].values()]
    if any(status == "failed" for status in statuses):
        report["status"] = "failed"
    elif statuses and all(status == "passed" for status in statuses):
        report["status"] = "passed"
    else:
        report["status"] = "incomplete"

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                      encoding="utf-8")
    print(f"validation status={report['status']} report={output}")
    return 0 if report["status"] == "passed" else (2 if report["status"] == "incomplete" else 1)


def shutil_which(name: str) -> str | None:
    paths = os.environ.get("PATH", "").split(os.pathsep)
    for directory in paths:
        candidate = Path(directory) / name
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return str(candidate)
    return None


if __name__ == "__main__":
    raise SystemExit(main())
