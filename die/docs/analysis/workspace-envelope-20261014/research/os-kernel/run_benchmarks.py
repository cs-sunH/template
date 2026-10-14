#!/usr/bin/env python3
"""In-process timing and identity capture for the scratch Solo13 kernel."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FACTS = ROOT / "work/os-single-owner-13-20261014/current"
OUT = HERE / "performance.json"
LOG = HERE / "performance.log"
SAMPLES = 30


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def command_version(args: list[str]) -> str:
    result = subprocess.run(args, check=False, capture_output=True, text=True)
    return (result.stdout + result.stderr).strip()


def run_benchmark(binary: Path, fixture: str, samples: int) -> dict:
    facts = FACTS / f"facts_{fixture}.json"
    command = [str(binary), "--facts", str(facts), "--fixture", fixture,
               "--samples", str(samples)]
    result = subprocess.run(command, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed ({result.returncode}): {command}\n{result.stderr}")
    return {"command": command, "sha256_facts": sha(facts),
            "result": json.loads(result.stdout)}


def main() -> int:
    normal = HERE / "solo13_benchmark"
    profiled = HERE / "solo13_benchmark_profile"
    if not normal.is_file() or not profiled.is_file():
        raise RuntimeError("compile both benchmark binaries with run_benchmarks.sh first")

    fixtures = {}
    for name in ("dense", "large_k", "large_outputs"):
        fixtures[name] = run_benchmark(normal, name, SAMPLES)

    profile = run_benchmark(profiled, "large_outputs", 1)
    perf = {"available": shutil.which("perf") is not None,
            "tool_version": None, "stat": None, "record": None}
    if perf["available"]:
        perf["tool_version"] = command_version(["perf", "--version"])
        facts = FACTS / "facts_large_outputs.json"
        perf_path = HERE / "perf_stat.txt"
        perf_command = ["perf", "stat", "-e", "cycles,instructions", "-o",
                        str(perf_path), "--", str(normal), "--facts", str(facts),
                        "--fixture", "large_outputs", "--samples", "1"]
        stat_run = subprocess.run(perf_command, check=False,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                                  text=True)
        perf["stat"] = {"command": perf_command, "returncode": stat_run.returncode,
                        "stderr": stat_run.stderr,
                        "output": perf_path.read_text() if perf_path.exists() else None}
        if stat_run.returncode == 0:
            record_path = HERE / "perf.data"
            record_command = ["perf", "record", "-g", "-o", str(record_path), "--",
                              str(normal), "--facts", str(facts), "--fixture",
                              "large_outputs", "--samples", "1"]
            record_run = subprocess.run(record_command, check=False,
                                        stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE, text=True)
            report_command = ["perf", "report", "--stdio", "--sort", "symbol",
                              "-i", str(record_path)]
            report_run = (subprocess.run(report_command, check=False,
                                         capture_output=True, text=True)
                          if record_run.returncode == 0 else None)
            perf["record"] = {
                "command": record_command, "returncode": record_run.returncode,
                "stderr": record_run.stderr,
                "report_command": report_command,
                "report_returncode": None if report_run is None else report_run.returncode,
                "report": None if report_run is None else report_run.stdout,
                "report_stderr": None if report_run is None else report_run.stderr}

    def git_text(*args: str) -> str:
        result = subprocess.run(["git", *args], cwd=ROOT, check=False,
                                capture_output=True, text=True)
        return result.stdout.strip()

    identities = {
        "git_head": git_text("rev-parse", "HEAD"),
        "git_branch": git_text("branch", "--show-current"),
        "kernel_header_sha256": sha(HERE / "solo13_kernel.hh"),
        "kernel_source_sha256": sha(HERE / "solo13_kernel.cc"),
        "benchmark_source_sha256": sha(HERE / "solo13_benchmark.cc"),
        "benchmark_driver_sha256": sha(HERE / "run_benchmarks.py"),
        "rational_source_sha256": sha(ROOT / "work/os-fixedwidth-13-20261014/checked_rational256.hh"),
        "geometry_source_sha256": sha(ROOT / "die/src/GemmOSGeometry.hh"),
        "access_counter_source_sha256": sha(ROOT / "die/src/GemmOSAccessCounter.hh"),
        "normal_binary_sha256": sha(normal),
        "profile_binary_sha256": sha(profiled),
        "compiler": command_version(["g++", "--version"]).splitlines()[0],
    }
    result = {
        "schema": "solo13_inprocess_performance_v1",
        "samples_per_fixture": SAMPLES,
        "fixtures": fixtures,
        "manual_exact_operation_profile": {
            "fixture": "large_outputs", "sample_count": 1,
            "command": profile["command"],
            "facts_sha256": profile["sha256_facts"],
            "ops": profile["result"]["profiled_stop_resume_exact_ops"],
            "interpretation": "Separate compile-time instrumented run after reusable Kernel construction. Inclusive rational operation scopes overlap; these figures locate arithmetic work, not wall-time samples."},
        "perf": perf,
        "identities": identities,
        "scope": "single-owner/single-core scratch exact continuous-time kernel; no public integer-ns clock; no Runtime, production timing, or descriptor certificate",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    LOG.write_text(json.dumps({"commands": {
        name: info["command"] for name, info in fixtures.items()},
        "profile_command": profile["command"],
        "perf": perf}, indent=2) + "\n")
    print(json.dumps({"output": str(OUT), "samples_per_fixture": SAMPLES,
                      "fixtures": list(fixtures), "perf_available": perf["available"],
                      "head": identities["git_head"],
                      "normal_binary_sha256": identities["normal_binary_sha256"]}, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"benchmark orchestration failed: {exc}", file=sys.stderr)
        raise SystemExit(1)
