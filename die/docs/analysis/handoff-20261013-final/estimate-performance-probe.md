# Candidate estimate driver probe

This is a five-sample diagnostic probe, not a formal P95 result. The source tree had concurrent parent/agent edits outside this task's owned regions, including `Mapping.cc` and Runtime LP code; those files were not modified here. The exact same task/hardware fixture, compiler, sample count, and owner count are used as the saved baseline, so the result is useful as a direction check but is not an isolated microbenchmark.

## Inputs and build

- Build: Release, GCC 14.2.0, CMake 4.4.4.
- Build directory: `work/estimate-driver/cmake`.
- Config: `die/tests/fixtures/coarse_die_v2.json`.
- Task: `die/tests/fixtures/gemm_medium.json` (128×128×128 GEMM).
- Iterations: 5; arbitration owners: 8.
- Command: `work/tooling/cmake/data/bin/cmake -S die -B work/estimate-driver/cmake -DCMAKE_BUILD_TYPE=Release -DCOARSE_DIE_BUILD_TESTS=ON`, then `cmake --build work/estimate-driver/cmake --target coarse_die_benchmark -j2`.
- Probe command: `work/estimate-driver/cmake/coarse_die_benchmark --config die/tests/fixtures/coarse_die_v2.json --task die/tests/fixtures/gemm_medium.json --iterations 5 --owners 8 --output work/estimate-driver/benchmark.json`.

## Compile probe, host nanoseconds

| Measure | Saved baseline, 5 samples | New path, 5 samples |
|---|---|---|
| Cold parse/resolve/compile P95 | 18,349,689 | 6,615,726 |
| Hot compile P95 | 17,827,866 | 6,100,264 |
| Cold samples | 18,349,689; 17,804,543; 17,956,329; 17,811,542; 17,898,001 | 6,615,726; 6,246,273; 6,284,219; 6,233,885; 6,301,495 |
| Hot samples | 17,639,384; 17,745,413; 17,827,866; 17,680,636; 17,698,573 | 6,066,784; 6,056,559; 6,100,264; 6,027,736; 6,066,424 |

The saved 18 ms baseline is `work/diagnostics/baseline-host.json`. The new path reduces these diagnostic compile samples by roughly 65%; it remains above the 2 ms design target.

The standalone estimate benchmark P95 changed from 277,245 ns to 177,425 ns. Both benchmark runs use the same task hash `448f36eef3694a624f56947a0540d89399734fe90e9e4d9eaba5076b21e90a06`, hardware fingerprint `sha256:987e770486a0bb7e196980e5ee0d8b96630955c45b97a0958937b62db2c7e787`, and plan hash `1ab900c6ba24347564cacbef013b46e6136d1aa984dcc0d6b08099bace821a20`.

The complete `standalone_result` JSON is structurally identical to the saved baseline, including plan fingerprint, selected template/tuple, latency (2,195 ns), execution progress, runtime counters, and resource vectors. The raw new benchmark result is `work/estimate-driver/benchmark.json`; its stdout capture is `work/estimate-driver/benchmark.stdout.log`.

## Functional checks

- `estimate_driver_test` (compiled as a standalone executable because parent handles CMake registration): PASS. Candidate helper latency equals public `estimate_alone` and direct Runtime completion; full report fields remain present; wrong hardware fingerprint and invalid physical work are rejected; a near-`uint64_t` launch propagates Runtime overflow.
- `coarse_compile_execution_test`: PASS. Existing compiler/C ABI regression still selects and serializes the same canonical plan behavior.
- `git diff --check` over owned source/test files: PASS.

## Shared immutable hardware and typed finish follow-up

Runtime now retains an owned `shared_ptr<const Hardware>`; the original by-value constructor still creates its own immutable snapshot. `compile` copies Hardware once and all candidate Runtime instances share that snapshot, while each Runtime keeps its own capacity array and execution state. The low-cost driver reads `terminal_finish_ns(handle)` after drain and does not construct `query_progress` JSON; public `estimate_alone` still builds the complete report.

Using the same 128×128×128 GEMM fixture, hardware, Release build, and 30-sample benchmark:

| Variant | Cold compile P95 | Hot compile P95 | Cold median | Hot median |
|---|---:|---:|---:|---:|
| Shared Hardware + typed finish | 2,232,141 ns | 2,112,342 ns | 2,169,507 ns | 2,007,666 ns |
| Per-candidate Hardware copy + typed finish | 2,658,339 ns | 2,328,495 ns | 2,455,226 ns | 2,221,145 ns |
| Shared Hardware + terminal `query_progress` JSON | 4,156,641 ns | 3,673,948 ns | 3,752,906 ns | 3,586,387 ns |

This A/B indicates shared Hardware removes about 0.43 ms from cold P95 and 0.22 ms from hot P95 on this fixture. Avoiding the terminal progress JSON accounts for a larger reduction (about 1.92 ms cold P95 and 1.56 ms hot P95). The clone cost was measurable, but it was not the largest remaining cost. These are diagnostic probes, not formal P95 claims. Raw captures: `shared-hardware-benchmark.json`, `per-candidate-hardware-benchmark.json`, and `progress-json-benchmark.json` in this scratch directory.

Functional validation: `coarse_estimate_driver_test`, `coarse_compile_execution_test`, `coarse_runtime_semantics_test`, `coarse_capi_lifecycle_test`, and `coarse_runtime_credit_scale_test` all pass. Shared and by-value candidate paths select the same plan hash/tuple and estimate the same Runtime finish. Two Runtimes sharing the same immutable Hardware retain independent capacity overrides; the typed accessor returns empty while live, matches `query_progress.finish_ns` after completion, reports cancellation time, and preserves stale-handle errors. Public report output matches the progress-JSON control variant; its tiny exact-decimal counter differences from the older saved baseline predate this follow-up in the concurrent Runtime solver changes.
