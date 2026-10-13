# Coarse die host benchmark report

Status: **passed**

Command: `["/workspace/template/work/estimate-driver/cmake/coarse_die_benchmark", "--config", "/workspace/template/die/tests/fixtures/coarse_die_v2.json", "--task", "/workspace/template/die/tests/fixtures/gemm_medium.json", "--iterations", "30", "--owners", "32"]`

Benchmark executable SHA-256: `e65e1a59365b40d54040353694e801654d91fb95427397e7e4611e7960a03b17`

| Measurement | p50 host ns | p95 host ns | Samples |
|---|---:|---:|---:|
| Cold file/parse/resolve/compile | 2312462.0 | 2499403.0 | 30 |
| Hot parsed/resolved compile | 2127706.0 | 3551648.0 | 30 |
| estimate_alone | 153440.0 | 196354.0 | 30 |
| Shared runtime arbitration | 1568069.0 | 1642069.0 | 30 |
| Shared runtime CPU | 1568000.0 | 1616000.0 | 30 |

Advanced runtime events: 3; modeled finish: 47880 ns; peak RSS: 13056 KiB.

Host benchmark time and modeled device latency are separate. Raw samples and runtime counters are in the JSON report.
