#!/usr/bin/env python3
"""Jointly sweep explicit topology and hardware axes through coarse_die C ABI."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation"))
from cabi import CoarseDieLibrary, find_library  # noqa: E402


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as stream:
        return json.load(stream)


def stable_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def find_latency(value: Any) -> float | None:
    if isinstance(value, dict):
        for key in ("estimated_latency_ns", "latency_ns", "finish_ns"):
            candidate = value.get(key)
            if isinstance(candidate, (int, float)) and math.isfinite(candidate):
                return float(candidate)
        for child in value.values():
            found = find_latency(child)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = find_latency(child)
            if found is not None:
                return found
    return None


def one_result(response: Any) -> tuple[Any, Any]:
    if not isinstance(response, dict):
        return None, {"reason": "C ABI response is not an object"}
    results = response.get("results")
    errors = response.get("errors")
    result = results[0] if isinstance(results, list) and results else None
    error = errors[0] if isinstance(errors, list) and errors else None
    return result, error


def build_task(path: Path | None) -> dict[str, Any]:
    if path:
        return load_json(path)
    return {
        "op_kind": "gemm",
        "tensor_shapes": {"m": 1024, "n": 4096, "k": 4096},
        "storage_dtype": {
            "weights": "bf16",
            "activations": "bf16",
            "output": "bf16",
        },
        "execution_profile": "int4_matrix_vec_fp32_assumed_v1",
        "layer_instances": 1,
        "weight_passes": 1,
        "graph_identity": {"request": "offline_sensitivity", "region": "gemm"},
    }


def make_cases(base: dict[str, Any], quick: bool) -> list[tuple[str, dict[str, Any], dict[str, Any]]]:
    if quick:
        ns = [2]
        placements = ["same_side_two_rows", "two_sides_one_row_each"]
        widths = [256]
        grids = [(8, 11), (11, 8)]
    else:
        ns = [1, 2, 3]
        placements = ["same_side_two_rows", "two_sides_one_row_each"]
        widths = [64, 128, 256, 512]
        grids = [(8, 11), (11, 8), (10, 10)]
    topologies = ["distributed_ideal_v2", "edge_single_word_v2"]
    cases = []
    for n in ns:
        for placement in placements:
            for bits in widths:
                for nx, ny in grids:
                    for topology in topologies:
                        config = copy.deepcopy(base)
                        arch = config["architecture"]
                        arch["hbm_stacks_per_row"] = n
                        arch["hbm_placement_mode"] = placement
                        arch["noc_width_bits"] = bits
                        arch["core_columns_nx"] = nx
                        arch["core_rows_ny"] = ny
                        config["performance_profile"]["timing_topology_profile"] = topology
                        case_id = (
                            f"n{n}_{placement}_B{bits}_{nx}x{ny}_{topology}"
                        )
                        axes = {
                            "hbm_stacks_per_row": n,
                            "hbm_placement_mode": placement,
                            "noc_width_bits": bits,
                            "grid": [nx, ny],
                            "timing_topology_profile": topology,
                        }
                        cases.append((case_id, axes, config))
    return cases


def evaluate_case(lib: CoarseDieLibrary, case_id: str,
                  axes: dict[str, Any], config: dict[str, Any],
                  task: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter_ns()
    model = None
    try:
        model = lib.create_model(config)
        hw = lib.hardware_summary(model)
        estimate_response = lib.estimate_batch(model, [task])
        result, error = one_result(estimate_response)
        elapsed = time.perf_counter_ns() - started
        if error:
            return {
                "case_id": case_id,
                "axes": axes,
                "status": "failed",
                "reason": error,
                "host_elapsed_ns": elapsed,
                "hardware_fingerprint": hw.get("hardware_fingerprint"),
                "derived_hardware": hw.get("derived_hardware"),
            }
        latency = find_latency(result)
        if result is None or latency is None:
            return {
                "case_id": case_id,
                "axes": axes,
                "status": "failed",
                "reason": "estimate response omitted a finite latency field",
                "result": result,
                "host_elapsed_ns": elapsed,
                "hardware_fingerprint": hw.get("hardware_fingerprint"),
                "derived_hardware": hw.get("derived_hardware"),
            }
        return {
            "case_id": case_id,
            "axes": axes,
            "status": result.get("status", "ok_assumed") if isinstance(result, dict) else "ok_assumed",
            "estimated_latency_ns": latency,
            "plan_hash": result.get("plan_hash") if isinstance(result, dict) else None,
            "granularity_validation": result.get("granularity_validation", "not_run") if isinstance(result, dict) else "not_run",
            "host_elapsed_ns": elapsed,
            "hardware_fingerprint": hw.get("hardware_fingerprint"),
            "derived_hardware": hw.get("derived_hardware"),
            "result": result,
        }
    except Exception as exc:  # preserve the failing configuration in the report
        return {
            "case_id": case_id,
            "axes": axes,
            "status": "failed",
            "reason": str(exc),
            "host_elapsed_ns": time.perf_counter_ns() - started,
        }
    finally:
        if model is not None:
            try:
                lib.destroy_model(model)
            except Exception:
                pass


def assign_rankings(results: list[dict[str, Any]]) -> dict[str, Any]:
    by_profile: dict[str, list[dict[str, Any]]] = {}
    for row in results:
        if row.get("status") == "failed" or "estimated_latency_ns" not in row:
            continue
        profile = row["axes"]["timing_topology_profile"]
        by_profile.setdefault(profile, []).append(row)
    summaries = {}
    for profile, rows in by_profile.items():
        rows.sort(key=lambda row: (row["estimated_latency_ns"], row["case_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank_within_topology"] = rank
            if rank == 1 and len(rows) > 1:
                gap = (rows[1]["estimated_latency_ns"] - row["estimated_latency_ns"]) / max(
                    row["estimated_latency_ns"], 1.0
                )
                row["gap_to_next_fraction"] = gap
                row["best_pair_ordering"] = (
                    "undetermined_within_3_percent_granularity_budget"
                    if gap <= 0.03
                    else "provisional_assumed_model_order"
                )
        summaries[profile] = {
            "successful_cases": len(rows),
            "best_case_id": rows[0]["case_id"] if rows else None,
            "best_latency_ns": rows[0]["estimated_latency_ns"] if rows else None,
            "best_pair_ordering": rows[0].get("best_pair_ordering") if rows else None,
            "ranking_status": "provisional_assumed_model_only",
        }

    dist = by_profile.get("distributed_ideal_v2", [])
    edge = by_profile.get("edge_single_word_v2", [])
    dist_map = {row["case_id"].replace("_distributed_ideal_v2", ""): row
                for row in dist}
    edge_map = {row["case_id"].replace("_edge_single_word_v2", ""): row
                for row in edge}
    shared = sorted(set(dist_map) & set(edge_map))
    reversals = 0
    comparisons = 0
    for i, left in enumerate(shared):
        for right in shared[i + 1:]:
            dl = dist_map[left]["estimated_latency_ns"] - dist_map[right]["estimated_latency_ns"]
            el = edge_map[left]["estimated_latency_ns"] - edge_map[right]["estimated_latency_ns"]
            if abs(dl) <= 0.03 * max(dist_map[left]["estimated_latency_ns"],
                                    dist_map[right]["estimated_latency_ns"], 1.0):
                continue
            if abs(el) <= 0.03 * max(edge_map[left]["estimated_latency_ns"],
                                    edge_map[right]["estimated_latency_ns"], 1.0):
                continue
            comparisons += 1
            if dl * el < 0:
                reversals += 1
    return {
        "topology_rankings": summaries,
        "distributed_vs_edge": {
            "matched_configurations": len(shared),
            "pairwise_comparisons_outside_3_percent_band": comparisons,
            "pairwise_order_reversals": reversals,
            "interpretation": "assumed topology sensitivity; not physical topology validation",
        },
    }


def markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Coarse die joint sensitivity report",
        "",
        f"Status: **{report['status']}**",
        "",
        f"Cases: {report.get('completed_cases', 0)} / {report.get('expected_cases', 0)}; "
        f"C ABI: {report.get('abi_version', 'not_run')}; library: `{report.get('library_path', 'not found')}`.",
        "",
        "This is a ranking across explicit model assumptions. It is not hardware calibration or physical topology validation.",
        "",
        "| Timing profile | Best case | Estimated ns | Best pair status |",
        "|---|---|---:|---|",
    ]
    for profile, row in report.get("rankings", {}).get("topology_rankings", {}).items():
        lines.append(
            f"| {profile} | {row.get('best_case_id')} | {row.get('best_latency_ns')} | {row.get('best_pair_ordering')} |"
        )
    reversal = report.get("rankings", {}).get("distributed_vs_edge", {})
    if reversal:
        lines.extend([
            "",
            f"Matched pairwise comparisons outside the 3% granularity budget: {reversal.get('pairwise_comparisons_outside_3_percent_band')}; "
            f"ordering reversals: {reversal.get('pairwise_order_reversals')}.",
        ])
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", help="path to libcoarse_die_cabi.so")
    parser.add_argument("--config", type=Path, default=ROOT / "tests/fixtures/coarse_die_v2.json")
    parser.add_argument("--task", type=Path, help="optional JSON task; default is the documented sensitivity GEMM")
    parser.add_argument("--output", type=Path, default=ROOT / "validation/sensitivity_report.json")
    parser.add_argument("--markdown", type=Path, help="optional markdown report path")
    parser.add_argument("--quick", action="store_true", help="run a small 8-case API smoke sweep")
    parser.add_argument("--max-cases", type=int, help="truncate the matrix for bounded debugging; report partial status")
    args = parser.parse_args()

    base = load_json(args.config)
    task = build_task(args.task)
    cases = make_cases(base, args.quick)
    if args.max_cases is not None:
        cases = cases[:max(0, args.max_cases)]
    expected_cases = 8 if args.quick else 144
    library_path = find_library(args.library)
    report: dict[str, Any] = {
        "schema_version": 1,
        "tool": "die/tools/sensitivity.py",
        "status": "not_run",
        "library_path": str(library_path) if library_path else None,
        "config_path": str(args.config.resolve()),
        "task_path": str(args.task.resolve()) if args.task else "built-in documented sensitivity GEMM",
        "config_sha256": stable_file_hash(args.config),
        "task": task,
        "expected_cases": expected_cases,
        "selected_cases": len(cases),
        "completed_cases": 0,
        "case_results": [],
        "rankings": {},
        "granularity_budget_fraction": 0.03,
        "notes": [
            "distributed_ideal_v2 and edge_single_word_v2 are explicit structural assumptions",
            "rankings within 3% are marked undetermined; the threshold is the initial standalone oracle budget",
            "outputs are model-to-model sensitivity, not hardware validation",
        ],
    }
    if not library_path:
        report["reason"] = "C ABI library not found; no model calls were made"
    else:
        lib = CoarseDieLibrary(library_path)
        report["abi_version"] = lib.abi_version()
        if report["abi_version"] != 2:
            report["reason"] = f"expected C ABI v2, found {report['abi_version']}"
            report["status"] = "failed"
        else:
            started = time.perf_counter_ns()
            for case_id, axes, config in cases:
                report["case_results"].append(
                    evaluate_case(lib, case_id, axes, config, task)
                )
            report["host_elapsed_ns"] = time.perf_counter_ns() - started
            report["completed_cases"] = len(report["case_results"])
            successful = sum("estimated_latency_ns" in row for row in report["case_results"])
            failures = report["completed_cases"] - successful
            report["rankings"] = assign_rankings(report["case_results"])
            if failures:
                report["status"] = "failed"
                report["failed_cases"] = failures
            elif len(cases) < expected_cases:
                report["status"] = "partial"
            else:
                report["status"] = "passed"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.markdown:
        args.markdown.parent.mkdir(parents=True, exist_ok=True)
        args.markdown.write_text(markdown(report), encoding="utf-8")
    print(f"sensitivity status={report['status']} cases={report['completed_cases']}/{report['expected_cases']} output={args.output}")
    return 0 if report["status"] == "passed" else (2 if report["status"] == "not_run" else 1)


if __name__ == "__main__":
    raise SystemExit(main())
