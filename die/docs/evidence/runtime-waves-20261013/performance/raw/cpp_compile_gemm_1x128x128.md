# Coarse die host benchmark report

Status: **passed**

Command: `["/workspace/template/work/build-die/coarse_die_benchmark", "--config", "/workspace/template/die/tests/fixtures/coarse_die_v2.json", "--task", "/workspace/template/work/performance-current/fixtures/gemm_1x128x128.json", "--iterations", "30", "--owners", "1"]`

Benchmark executable SHA-256: `16ce959db28cccbbb6c9e44f5883e473cc3c29b45329804d21ae9defae105951`

| Measurement | p50 host ns | p95 host ns | Samples |
|---|---:|---:|---:|
| Cold file/parse/resolve/compile | 867466.0 | 976060.0 | 30 |
| Hot parsed/resolved compile | 606704.0 | 639033.0 | 30 |
| estimate_alone | 232560.0 | 455146.0 | 30 |
| Shared runtime arbitration | 7802.0 | 14432.0 | 30 |
| Shared runtime CPU | 8000.0 | 15000.0 | 30 |

Advanced runtime events: 3; modeled finish: 1298 ns; peak RSS: 13056 KiB.

Host benchmark time and modeled device latency are separate. Raw samples and runtime counters are in the JSON report.
