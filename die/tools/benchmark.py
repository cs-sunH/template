#!/usr/bin/env python3
"""Run the compiled library benchmark and preserve raw timing/count evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def find_binary(explicit: str | None) -> Path | None:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        return path if path.is_file() else None
    env = os.environ.get("COARSE_DIE_BENCHMARK")
    if env:
        path = Path(env).expanduser().resolve()
        if path.is_file():
            return path
    roots = [Path.cwd(), ROOT, ROOT.parent]
    candidates: list[Path] = []
    for root in roots:
        candidates.extend(
            [
                root / "work/build-die/coarse_die_benchmark",
                root / "build-die/coarse_die_benchmark",
                root / "build/coarse_die_benchmark",
                root / "build/die/coarse_die_benchmark",
            ]
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_markdown(report: dict[str, Any], path: Path) -> None:
    measured = report.get("benchmark_result")
    lines = [
        "# Coarse die host benchmark report",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Command: `{json.dumps(report.get('command', []), ensure_ascii=False)}`",
        "",
        f"Benchmark executable SHA-256: `{report.get('binary_sha256', 'not_run')}`",
        "",
    ]
    if measured:
        lines.extend(
            [
                "| Measurement | p50 host ns | p95 host ns | Samples |",
                "|---|---:|---:|---:|",
            ]
        )
        for key, title in (
            ("cold_create_parse_resolve_compile", "Cold file/parse/resolve/compile"),
            ("hot_compile_same_hardware_same_task", "Hot parsed/resolved compile"),
            ("estimate_alone", "estimate_alone"),
            ("shared_runtime_arbitration", "Shared runtime arbitration"),
            ("shared_runtime_arbitration_cpu", "Shared runtime CPU"),
        ):
            row = measured.get(key, {})
            lines.append(
                f"| {title} | {row.get('p50_ns')} | {row.get('p95_ns')} | {row.get('count')} |"
            )
        lines.extend(
            [
                "",
                f"Advanced runtime events: {measured.get('runtime_advance_event_count')}; "
                f"modeled finish: {measured.get('runtime_finish_ns')} ns; "
                f"peak RSS: {measured.get('max_rss_kib')} KiB.",
                "",
                "Host benchmark time and modeled device latency are separate. Raw samples and runtime counters are in the JSON report.",
            ]
        )
    else:
        lines.extend([report.get("reason", "No benchmark result was produced."), ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-bin", help="path to coarse_die_benchmark")
    parser.add_argument("--config", type=Path, default=ROOT / "tests/fixtures/coarse_die_v2.json")
    parser.add_argument("--task", type=Path, default=ROOT / "tests/fixtures/gemm_medium.json")
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--owners", type=int, default=8)
    parser.add_argument("--output", type=Path, default=ROOT / "validation/benchmark_report.json")
    parser.add_argument("--markdown", type=Path, help="optional markdown report path")
    args = parser.parse_args()

    binary = find_binary(args.benchmark_bin)
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "die/tools/benchmark.py",
        "status": "not_run",
        "benchmark_binary": str(binary) if binary else None,
        "config_path": str(args.config.resolve()),
        "config_sha256": sha256_file(args.config),
        "task_path": str(args.task.resolve()),
        "task_sha256": sha256_file(args.task),
        "iterations": args.iterations,
        "contending_owners": args.owners,
        "benchmark_result": None,
    }
    if binary is None:
        report["reason"] = "coarse_die_benchmark executable not found; no measurements were made"
    else:
        command = [
            str(binary),
            "--config", str(args.config.resolve()),
            "--task", str(args.task.resolve()),
            "--iterations", str(args.iterations),
            "--owners", str(args.owners),
        ]
        report["command"] = command
        report["binary_sha256"] = sha256_file(binary)
        started = time.perf_counter_ns()
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        report["driver_host_elapsed_ns"] = time.perf_counter_ns() - started
        report["driver_return_code"] = completed.returncode
        report["stdout"] = completed.stdout
        report["stderr"] = completed.stderr
        if completed.returncode != 0:
            report["status"] = "failed"
            report["reason"] = f"benchmark executable exited {completed.returncode}"
        else:
            try:
                result = json.loads(completed.stdout)
            except json.JSONDecodeError as exc:
                report["status"] = "failed"
                report["reason"] = f"benchmark output was not JSON: {exc}"
            else:
                report["benchmark_result"] = result
                report["status"] = "passed" if result.get("status") == "ok" else "failed"
                if report["status"] == "failed":
                    report["reason"] = "benchmark executable returned a non-ok status"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        write_markdown(report, args.markdown)
    print(f"benchmark status={report['status']} output={args.output}")
    return 0 if report["status"] == "passed" else (2 if report["status"] == "not_run" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
