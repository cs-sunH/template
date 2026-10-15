# Frozen public-model performance baseline — 2026-10-15

Frozen source commit: `5c2de289bc5744e5c252d782901cdba8b6f62459`.
The archive used tracked SDK, tests/configs, and vendored JSON from that exact commit; concurrent main-worktree edits cannot alter this baseline. No production file was modified. Source-file SHA256, archive SHA256, build commands, actual compiler commands digest, ELF/library hashes and build ID are recorded in the JSON identities. GCC 14.2.0, CMake 3.31.10, Release (`-O3 -DNDEBUG`), Linux x86_64; 3 visible CPUs, cgroup quota 2 CPU.

Actual original command: `work/performance-baseline/build/coarse_service_curve_benchmark --samples 30 --output die/docs/evidence/continue-performance-20261015/baseline-benchmark.json` (the recorded metadata retains absolute paths). Build succeeded. Benchmark exit **1**, overall **failed**. No acceptance threshold changed.

| Metric | P50 | P95 | Existing gate | Result |
|---|---:|---:|---:|---|
| GEMM 1×128×128 uncached public compile | 367.584 µs | 569.542 µs | 2 ms | passed |
| GEMM 128×128×128 uncached public compile | 1.580086 ms | 1.957633 ms | 2 ms | passed, little margin |
| GEMM small / large / attention hot cost | 2.974 / 2.954 / 3.004 µs | 4.326 / 4.467 / 5.969 µs | 20 µs | passed |
| Compiled OS bulk J1 | 9.273 µs | 13.119 µs | 50 µs | passed |
| Compiled OS bulk J8 | 19.820 µs | 25.548 µs | 50 µs | passed |
| Compiled OS bulk J32 | 31.847 µs | **102.121 µs** | 50 µs | **failed** |

All 15 runtime-fixture coverage checks passed and the remaining 14 runtime metrics passed. OS bulk used the ratio path in all 90 measured updates; equivalent-owner reduction left 3 solver activities and 15 rows, with 0 simplex pivots. J32 had 28 samples in 26.900–35.102 µs and two tails at 102.121/130.853 µs. No compiler/test process was observed in the before/after process snapshots; the parent coordinated an idle measurement window. Cgroup throttling counters did not increase. Host scheduling interference cannot be excluded, and the raw failure is retained. This is one 30-sample measurement, not an all-load or five-repository speed certificate.

## Diagnostic stage probe

`stage_probe.cc` is an external driver linked to the same frozen Release static library, and does not replace or patch production code. For 128³ it observed 28 feasible software mappings. Stage P50/P95 in µs:

| Diagnostic call | P50 | P95 |
|---|---:|---:|
| normalize_task | 7.892 | 13.360 |
| enumerate_mappings_for_compile | 647.077 | 1150.620 |
| estimate all prebuilt candidates | 706.404 | 1100.326 |
| plan_to_json on compiled winner | 55.512 | 152.025 |
| public compile | 1750.647 | 4079.208 |

These phases reuse prebuilt objects, differ in allocation lifetime, and are **not an additive profile** of the public call. The diagnostic public-call tail remains a second real failure observation, not a substitute for the original benchmark. No instruction/call-stack profiler was available (`perf` and valgrind absent).

Cold compilation spends material time both constructing feasible candidates and executing their common Runtime estimates. Normalization alone is too small to explain the budget. Useful simple investigations are reducing repeated immutable JSON/hardware/task copies during candidate construction and repeated fixed-size allocation/validation setup during candidate estimates while retaining exact validation, ranking, software mappings and the shared Runtime model. Do not prune legal mappings merely to meet timing. Final winner serialization is smaller than either major phase.

For OS arbitration, optimize the fixed-size owner/reference collection and exact symmetry comparisons before considering a solver change: this workload already uses the three-variable ratio path, not dense simplex. Reuse scratch storage and numeric owner identity while preserving split-owner Q denominators, PRIVATE, exact curve derivative/boundary equivalence, original sorting/ties, and strict physical/credit certification. Keep a general fallback. The median is already below budget, so repeat an idle, source-bound measurement after an actual optimization; do not select the best run as acceptance evidence.

These are CPU diagnostics only. No independent target-hardware calibration, updated model accuracy oracle, public CTest suite, five-repository build, production workload or E2E CPU budget was executed here.

## Replay

Activate the existing toolchain if available (`source /workspace/cloud-setup/activate.sh`), then run from the template repository:

```bash
python3 die/docs/evidence/continue-performance-20261015/reproduce.py --root . --samples 30
```

The script archives the frozen commit into `work/performance-baseline/replay`, builds sequentially before benchmarking, preserves every command/exit/log, and never changes git state. It requires that commit to exist locally. Pause unrelated builds/tests for the benchmark. New runs are separate evidence and do not overwrite the original raw files in this directory. Stage-probe compiler and run commands, source hash, static-library hash and driver ELF hash are in `stage-probe.meta.json`.
