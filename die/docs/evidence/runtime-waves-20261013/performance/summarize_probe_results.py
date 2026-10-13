#!/usr/bin/env python3
"""Validate and summarize the saved current-build probe results."""

import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
SHAPES = (
    ("gemm_1x128x128", "GEMM 1×128×128"),
    ("gemm_128x128x128", "GEMM 128×128×128"),
    ("attention_small_span", "Attention q=4, kv=16, 4/2 heads, d=32"),
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def stat_pair(stats: dict) -> dict:
    if stats.get("count") != 30 or len(stats.get("samples_ns", [])) != 30:
        raise ValueError(f"expected 30 samples in {stats}")
    return {"count": stats["count"], "p50_ns": stats["p50_ns"], "p95_ns": stats["p95_ns"]}


def ms(ns: float) -> str:
    return f"{ns / 1_000_000:.3f}"


def main() -> None:
    rows = []
    rss_values = []
    for slug, label in SHAPES:
        cpp_path = RAW / f"cpp_compile_{slug}.json"
        cabi_path = RAW / f"cabi_cache_{slug}.json"
        cpp = read_json(cpp_path)
        cabi = read_json(cabi_path)
        if cpp.get("status") != "passed":
            raise ValueError(f"C++ benchmark failed for {slug}: {cpp.get('reason')}")
        cpp_data = cpp["benchmark_result"]
        cold = stat_pair(cpp_data["cold_create_parse_resolve_compile"])
        hot = stat_pair(cpp_data["hot_compile_same_hardware_same_task"])
        if cabi.get("status") != "ok":
            raise ValueError(f"C ABI probe failed for {slug}")
        miss = stat_pair(cabi["cold_cache_miss_compile_batch"])
        hit = stat_pair(cabi["same_task_cache_hit_compile_batch"])
        create = stat_pair(cabi["create_model"])
        validation = cabi["cache_hit_validation"]
        if not all((
            validation["fresh_model_per_sample"],
            validation["same_request_reused_within_model"],
            validation["plan_hash_stable"],
            validation["cache_entries_and_accounted_bytes_stable_on_second_call"],
            validation["returned_plan_handles_released"],
        )):
            raise ValueError(f"cache-hit checks failed for {slug}")
        samples = cabi["raw_samples"]
        if len(samples) != 30:
            raise ValueError(f"expected 30 C ABI samples for {slug}")
        for sample in samples:
            if (sample["cache_entries_after_miss"] != 1 or
                    sample["cache_entries_after_hit"] != 1 or
                    sample["cache_bytes_after_miss"] != sample["cache_bytes_after_hit"] or
                    sample["plan_hash"] != validation["plan_hash"]):
                raise ValueError(f"cache state/hash changed in {slug} sample {sample['sample']}")
        if cpp_data["iterations"] != 30 or cabi["samples"] != 30:
            raise ValueError(f"unexpected iteration count for {slug}")
        cpp_rss = cpp_data.get("max_rss_kib", -1)
        cabi_rss = cabi.get("max_rss_kib", -1)
        rss_values.extend(v for v in (cpp_rss, cabi_rss) if v >= 0)
        rows.append({
            "fixture": slug,
            "shape": label,
            "cpp_cold_file_parse_resolve_compile": cold,
            "cpp_hot_recompile": hot,
            "cabi_model_create": create,
            "cabi_cold_cache_miss": miss,
            "cabi_same_task_cache_hit": hit,
            "cache_entries": validation["entries_after_miss"],
            "cache_accounted_bytes": validation["bytes_after_miss"],
            "plan_hash": validation["plan_hash"],
            "cpp_max_rss_kib": cpp_rss,
            "cabi_max_rss_kib": cabi_rss,
            "raw_cpp": str(cpp_path.relative_to(ROOT)),
            "raw_cabi": str(cabi_path.relative_to(ROOT)),
        })

    environment = (ROOT / "environment.txt").read_text(encoding="utf-8")
    digest_match = re.search(r"^published_source_digest=(\S+)$", environment, re.MULTILINE)
    if not digest_match:
        raise ValueError("source digest missing from environment record")
    digest = digest_match.group(1)
    if digest != "e877a4ddc2dd5e0eff332fc8c01fcc33cc3ea92cfb8aac99c856b45a24f48b7a":
        raise ValueError(f"unexpected final source digest: {digest}")
    report = {
        "schema": "current_build_performance_probe_v1",
        "source_digest": digest,
        "build_type": "Release",
        "build_flags": "-O3 -DNDEBUG",
        "sample_count_per_measurement_per_shape": 30,
        "host_timings_are_budget_acceptance": False,
        "p95_method": "nearest-rank ceil(0.95*n), as implemented by both probe tools",
        "rss_range_kib_across_six_processes": {
            "min": min(rss_values) if rss_values else None,
            "max": max(rss_values) if rss_values else None,
        },
        "shapes": rows,
    }
    (ROOT / "summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Current public build performance probe",
        "",
        f"Source snapshot digest: `{digest}` (SHA-256, 20 synchronized vendor files).",
        "Build: `Release`, `/usr/bin/c++` GNU 14.2.0, `-O3 -DNDEBUG`; binary and source hashes, CPU metadata, and load averages are in [environment.txt](environment.txt).",
        "",
        "Each measurement below has 30 raw samples. Times are host wall-clock nanoseconds summarized in milliseconds; p95 uses nearest-rank `ceil(0.95 × 30)` (the 29th ordered sample). The raw JSON files preserve every sample.",
        "",
        "| Fixture | C++ cold file/parse/resolve/compile p50 / p95 (ms) | C++ hot compile p50 / p95 (ms) | C ABI empty-cache miss p50 / p95 (ms) | C ABI same-task cache hit p50 / p95 (ms) | Cache entries / bytes |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['shape']} | {ms(row['cpp_cold_file_parse_resolve_compile']['p50_ns'])} / {ms(row['cpp_cold_file_parse_resolve_compile']['p95_ns'])} "
            f"| {ms(row['cpp_hot_recompile']['p50_ns'])} / {ms(row['cpp_hot_recompile']['p95_ns'])} "
            f"| {ms(row['cabi_cold_cache_miss']['p50_ns'])} / {ms(row['cabi_cold_cache_miss']['p95_ns'])} "
            f"| {ms(row['cabi_same_task_cache_hit']['p50_ns'])} / {ms(row['cabi_same_task_cache_hit']['p95_ns'])} "
            f"| {row['cache_entries']} / {row['cache_accounted_bytes']} |"
        )
    lines.extend([
        "",
        "The C++ cold interval reads and parses task/config JSON, resolves hardware, then compiles. Its hot interval reuses parsed inputs and hardware but calls C++ `compile()` again; it is not a cache hit.",
        "",
        "For each C ABI sample, a fresh model begins with an empty cache. The driver times the first compile-batch call, then submits the byte-identical task to the same model and times the repeated call. It checks stable plan hash, one cache entry, and unchanged accounted bytes; both returned plan handles are released before model destruction. The cache path is independently visible in `Capi.cc::cached()`: a found key returns the retained plan before the compile fallback. The C ABI hit measurement includes JSON request parsing and response serialization/allocation, while caller-side request construction and result parsing/free are outside the interval. Model creation and handle release are reported separately in raw JSON.",
        "",
        f"Across the six benchmark processes, reported peak RSS ranged from {report['rss_range_kib_across_six_processes']['min']} to {report['rss_range_kib_across_six_processes']['max']} KiB. The system load averages immediately before and after the sequence are recorded in `environment.txt`; this remains a small host-direction probe, not full budget acceptance. No 96-active L/C/S load test or LP-pivot claim is included.",
        "",
        "Raw reports: `raw/cpp_compile_*.json` and `raw/cabi_cache_*.json`; exact commands: `commands.txt`; run transcript: `run.log`; reproducible runner and validator: `run_probes.sh`, `summarize_probe_results.py`.",
        "",
    ])
    (ROOT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"validated 6 reports, 30 samples per metric and shape; wrote {ROOT / 'summary.md'}")


if __name__ == "__main__":
    main()
