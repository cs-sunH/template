#!/usr/bin/env python3
"""Validate cache-response retest, retain nearest-rank stats, compare archived baseline."""

import json
import math
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
BASE = ROOT.parent / "performance-current" / "raw"
SHAPES = (
    ("gemm_1x128x128", "GEMM 1×128×128"),
    ("gemm_128x128x128", "GEMM 128×128×128"),
    ("attention_small_span", "Attention q=4, kv=16, 4/2 heads, d=32"),
)
EXPECTED_DIGEST = "05c3f91c59e4ac3d22b5147a12566d93902814aa77bbf3384a9b7a5468a9e755"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def nearest_rank(values: list[float], p: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, math.ceil(p * len(ordered)) - 1))
    return ordered[index]


def summarize(values: list[float], expected_count: int = 30) -> dict:
    if len(values) != expected_count:
        raise ValueError(f"expected {expected_count} samples, got {len(values)}")
    return {
        "count": len(values),
        "p50_ns": nearest_rank(values, 0.50),
        "p95_ns": nearest_rank(values, 0.95),
        "p99_ns": nearest_rank(values, 0.99),
        "min_ns": min(values),
        "max_ns": max(values),
    }


def from_stats(stats: dict, expected_count: int = 30) -> dict:
    values = stats.get("samples_ns", [])
    measured = summarize(values, expected_count)
    # Ensure the source probes' original p50/p95 agree with the preserved runner's convention.
    for key in ("p50_ns", "p95_ns"):
        if stats.get(key) != measured[key]:
            raise ValueError(f"probe {key}={stats.get(key)} disagrees with raw {key}={measured[key]}")
    if stats.get("count") != expected_count:
        raise ValueError(f"probe reported count {stats.get('count')}, expected {expected_count}")
    return measured


def ms(ns: float) -> str:
    return f"{ns / 1_000_000:.3f}"


def us(ns: float) -> str:
    return f"{ns / 1_000:.2f}"


def ratio_pct(new: float, old: float) -> float | None:
    return None if old == 0 else 100.0 * (new - old) / old


def read_baseline(slug: str, family: str, field: str) -> dict:
    if family == "cpp":
        payload = read_json(BASE / f"cpp_compile_{slug}.json")
        if payload.get("status") != "passed":
            raise ValueError(f"baseline C++ report failed for {slug}")
        return from_stats(payload["benchmark_result"][field])
    payload = read_json(BASE / f"cabi_cache_{slug}.json")
    if payload.get("status") != "ok":
        raise ValueError(f"baseline C ABI report failed for {slug}")
    return from_stats(payload[field])


def main() -> None:
    rows = []
    rss_values = []
    for slug, label in SHAPES:
        cpp_payload = read_json(RAW / f"cpp_compile_{slug}.json")
        cabi_payload = read_json(RAW / f"cabi_cache_{slug}.json")
        if cpp_payload.get("status") != "passed":
            raise ValueError(f"C++ benchmark failed for {slug}: {cpp_payload.get('reason')}")
        if cabi_payload.get("status") != "ok":
            raise ValueError(f"C ABI probe failed for {slug}")
        cpp = cpp_payload["benchmark_result"]
        cold = from_stats(cpp["cold_create_parse_resolve_compile"])
        hot = from_stats(cpp["hot_compile_same_hardware_same_task"])
        miss = from_stats(cabi_payload["cold_cache_miss_compile_batch"])
        hit = from_stats(cabi_payload["same_task_cache_hit_compile_batch"])
        create = from_stats(cabi_payload["create_model"])

        validation = cabi_payload["cache_hit_validation"]
        required_checks = (
            "fresh_model_per_sample",
            "same_request_reused_within_model",
            "plan_hash_stable",
            "cache_entries_and_accounted_bytes_stable_on_second_call",
            "returned_plan_handles_released",
        )
        if not all(validation.get(key) is True for key in required_checks):
            raise ValueError(f"cache-hit checks failed for {slug}")
        samples = cabi_payload.get("raw_samples", [])
        if len(samples) != 30 or cabi_payload.get("samples") != 30:
            raise ValueError(f"expected 30 C ABI samples for {slug}")
        for sample in samples:
            if (sample["cache_entries_after_miss"] != 1 or
                    sample["cache_entries_after_hit"] != 1 or
                    sample["cache_bytes_after_miss"] != sample["cache_bytes_after_hit"] or
                    sample["plan_hash"] != validation["plan_hash"] or
                    sample["cold_plan_handle"] == 0 or sample["hot_plan_handle"] == 0 or
                    sample["cold_plan_handle"] == sample["hot_plan_handle"]):
                raise ValueError(f"handle/cache/hash check failed in {slug} sample {sample['sample']}")
        if cpp.get("iterations") != 30:
            raise ValueError(f"expected 30 C++ iterations for {slug}")

        current_metrics = {
            "cpp_cold_file_parse_resolve_compile": cold,
            "cpp_hot_recompile": hot,
            "cabi_cold_cache_miss": miss,
            "cabi_same_task_cache_hit": hit,
        }
        baseline_fields = {
            "cpp_cold_file_parse_resolve_compile": ("cpp", "cold_create_parse_resolve_compile"),
            "cpp_hot_recompile": ("cpp", "hot_compile_same_hardware_same_task"),
            "cabi_cold_cache_miss": ("cabi", "cold_cache_miss_compile_batch"),
            "cabi_same_task_cache_hit": ("cabi", "same_task_cache_hit_compile_batch"),
        }
        comparisons = {}
        for metric, new_stats in current_metrics.items():
            family, field = baseline_fields[metric]
            old_stats = read_baseline(slug, family, field)
            comparisons[metric] = {
                "baseline": old_stats,
                "current": new_stats,
                "delta_ns": {q: new_stats[q] - old_stats[q] for q in ("p50_ns", "p95_ns", "p99_ns")},
                "delta_percent": {q: ratio_pct(new_stats[q], old_stats[q]) for q in ("p50_ns", "p95_ns", "p99_ns")},
            }
        cpp_rss = cpp.get("max_rss_kib", -1)
        cabi_rss = cabi_payload.get("max_rss_kib", -1)
        rss_values.extend(v for v in (cpp_rss, cabi_rss) if v >= 0)
        rows.append({
            "fixture": slug,
            "shape": label,
            **current_metrics,
            "cabi_model_create": create,
            "cache_entries": validation["entries_after_miss"],
            "cache_accounted_bytes": validation["bytes_after_miss"],
            "plan_hash": validation["plan_hash"],
            "cpp_max_rss_kib": cpp_rss,
            "cabi_max_rss_kib": cabi_rss,
            "comparisons": comparisons,
        })

    environment = (ROOT / "environment.txt").read_text(encoding="utf-8")
    digest_match = re.search(r"^source_summary_digest=(\S+)$", environment, re.MULTILINE)
    if not digest_match or digest_match.group(1) != EXPECTED_DIGEST:
        raise ValueError("expected source summary digest missing or mismatched in environment record")
    report = {
        "schema": "cache_response_retest_v1",
        "source_summary_digest": EXPECTED_DIGEST,
        "baseline_source_summary_digest": "e877a4ddc2dd5e0eff332fc8c01fcc33cc3ea92cfb8aac99c856b45a24f48b7a",
        "build_type": "Release",
        "build_flags": "-O3 -DNDEBUG",
        "sample_count_per_measurement_per_shape": 30,
        "percentile_method": "nearest-rank ceil(p*n); p50=rank 15, p95=rank 29, p99=rank 30/max for n=30; copied raw arrays retained",
        "timing_scope_matches_baseline": True,
        "host_timings_are_budget_acceptance": False,
        "rss_range_kib_across_six_measured_processes": {
            "min": min(rss_values) if rss_values else None,
            "max": max(rss_values) if rss_values else None,
        },
        "thresholds": {
            "compile_cold_target_ns": 2_000_000,
            "cache_hit_target_ns": 20_000,
            "evaluation_basis": "P50/P95/P99 are each compared against the respective 2ms compile or 20us cache-hit target; host probe only, not full load acceptance",
        },
        "shapes": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    (ROOT / "baseline_comparison.json").write_text(json.dumps({
        "source_summary_digest": EXPECTED_DIGEST,
        "baseline_directory": "../performance-current",
        "baseline_digest": report["baseline_source_summary_digest"],
        "comparison": [{"fixture": row["fixture"], "shape": row["shape"], "metrics": row["comparisons"]} for row in rows],
    }, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# C ABI cache-response retest",
        "",
        f"Integrated source summary digest: `{EXPECTED_DIGEST}`. The independently captured per-file SHA-256 manifests, working-tree revision, binary hashes, environment, and load/RSS metadata are in [environment.txt](environment.txt). The archived baseline remains in [../performance-current](../performance-current); its e877 digest is retained there.",
        "",
        "Release build (`-O3 -DNDEBUG`), three fixtures, 30 measured samples per metric and shape. Probe logic and caller timing boundaries match the archived baseline: C++ cold/hot calls follow the benchmark tool; each C ABI sample creates a fresh model outside compile timing, measures an empty-cache miss followed by the same byte-identical request on that model, includes request JSON parsing and response serialization/allocation in each `compile_batch` call, and excludes request construction, result parsing/free, model creation, and plan release. Model create/release remain separate metrics. There is no in-loop warmup in either original probe; one-sample C ABI setup preflight files are kept separate and excluded from all statistics.",
        "",
        "P50/P95/P99 use nearest-rank `ceil(p×30)` to preserve the original probe's convention: rank 15, 29, and 30 respectively. P99 is the maximum for 30 samples. Raw samples, process CPU/RSS metadata from the standard-library outer runner, and probe-reported peak RSS are retained. Threshold flags below compare each percentile with the specified 2 ms cold compile or 20 µs cache-hit target; they are host-direction results and are not full budget acceptance.",
        "",
        "| Fixture | Metric | Current P50/P95/P99 | Baseline P50/P95/P99 | Δ P50/P95/P99 | Target check P50/P95/P99 |",
        "|---|---|---:|---:|---:|---|",
    ]
    metric_defs = (
        ("cpp_cold_file_parse_resolve_compile", "C++ cold parse/resolve/compile", "compile"),
        ("cpp_hot_recompile", "C++ hot recompile", "compile"),
        ("cabi_cold_cache_miss", "C ABI cache miss", "compile"),
        ("cabi_same_task_cache_hit", "C ABI same-model cache hit", "hit"),
    )
    for row in rows:
        for metric, title, target_kind in metric_defs:
            cur = row[metric]
            cmp = row["comparisons"][metric]
            old = cmp["baseline"]
            target_ns = 20_000 if target_kind == "hit" else 2_000_000
            target_name = "20 µs" if target_kind == "hit" else "2 ms"
            checks = "/".join("pass" if cur[q] <= target_ns else "fail" for q in ("p50_ns", "p95_ns", "p99_ns"))
            deltas = "/".join(f"{cmp['delta_percent'][q]:+.1f}%" for q in ("p50_ns", "p95_ns", "p99_ns"))
            lines.append(
                f"| {row['shape']} | {title} | {ms(cur['p50_ns'])}/{ms(cur['p95_ns'])}/{ms(cur['p99_ns'])} ms | "
                f"{ms(old['p50_ns'])}/{ms(old['p95_ns'])}/{ms(old['p99_ns'])} ms | {deltas} | {checks} ({target_name}) |"
            )
    lines.extend([
        "",
        "The C++ source path is unchanged for this C API response-cache adjustment, so C++ cold/hot deltas are control measurements only and are not attributed to response serialization caching. C ABI miss measurements include actual compile work and can move with host/build differences; only the same-model hit comparison isolates the cached response path most directly.",
        "Two setup attempts stopped before measured samples because `cmake` was absent from PATH and `/usr/bin/time` was unavailable; their partial logs are retained as setup-attempt-* files. The completed run used the bundled CMake for environment metadata and a Python resource wrapper, and all six measured reports passed validation.",
        "",
        "The probe verified one accounted cache entry, stable plan hash and cache bytes on the repeat, distinct nonzero plan handles for miss and hit, and successful handle release. Cache bytes are reported from the API; this run does not independently inspect the internal 4096-entry/64-MiB eviction boundary. Source manifest equality before/after and binary SHA equality are recorded in environment metadata.",
        "",
        f"Measured-process peak RSS ranged from {report['rss_range_kib_across_six_measured_processes']['min']} to {report['rss_range_kib_across_six_measured_processes']['max']} KiB. Load averages before/after, CPU model/core visibility, per-process user/system CPU and maximum RSS, and exact commands are retained. This remains a small host-direction probe, not a full contention/budget test.",
        "",
        "Artifacts: `raw/` contains six 30-sample reports plus external CPU/RSS records; `preflight/` contains the three one-sample C ABI checks; `commands.txt`, `run.log`, `environment.txt`, source manifests, and the reusable runner/probe/validator are saved beside them. No baseline files were changed.",
        "",
    ])
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"validated 6 reports, 30 samples per metric and shape; wrote {ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
