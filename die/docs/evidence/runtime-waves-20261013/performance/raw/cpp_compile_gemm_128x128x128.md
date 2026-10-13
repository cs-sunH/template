# Coarse die host benchmark report

Status: **passed**

Command: `["/workspace/template/work/build-die/coarse_die_benchmark", "--config", "/workspace/template/die/tests/fixtures/coarse_die_v2.json", "--task", "/workspace/template/work/performance-current/fixtures/gemm_128x128x128.json", "--iterations", "30", "--owners", "1"]`

Benchmark executable SHA-256: `16ce959db28cccbbb6c9e44f5883e473cc3c29b45329804d21ae9defae105951`

| Measurement | p50 host ns | p95 host ns | Samples |
|---|---:|---:|---:|
| Cold file/parse/resolve/compile | 3389505.0 | 4012232.0 | 30 |
| Hot parsed/resolved compile | 3051356.0 | 3377097.0 | 30 |
| estimate_alone | 229766.0 | 268575.0 | 30 |
| Shared runtime arbitration | 9565.0 | 10336.0 | 30 |
| Shared runtime CPU | 10000.0 | 11000.0 | 30 |

Advanced runtime events: 3; modeled finish: 3293 ns; peak RSS: 13056 KiB.

Host benchmark time and modeled device latency are separate. Raw samples and runtime counters are in the JSON report.
