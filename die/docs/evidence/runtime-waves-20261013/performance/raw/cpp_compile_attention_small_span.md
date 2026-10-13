# Coarse die host benchmark report

Status: **passed**

Command: `["/workspace/template/work/build-die/coarse_die_benchmark", "--config", "/workspace/template/die/tests/fixtures/coarse_die_v2.json", "--task", "/workspace/template/work/performance-current/fixtures/attention_small_span.json", "--iterations", "30", "--owners", "1"]`

Benchmark executable SHA-256: `16ce959db28cccbbb6c9e44f5883e473cc3c29b45329804d21ae9defae105951`

| Measurement | p50 host ns | p95 host ns | Samples |
|---|---:|---:|---:|
| Cold file/parse/resolve/compile | 417990.0 | 472882.0 | 30 |
| Hot parsed/resolved compile | 162274.0 | 191228.0 | 30 |
| estimate_alone | 238279.0 | 321314.0 | 30 |
| Shared runtime arbitration | 9304.0 | 10015.0 | 30 |
| Shared runtime CPU | 10000.0 | 10000.0 | 30 |

Advanced runtime events: 3; modeled finish: 337 ns; peak RSS: 13056 KiB.

Host benchmark time and modeled device latency are separate. Raw samples and runtime counters are in the JSON report.
