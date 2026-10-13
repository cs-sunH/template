# Coarse die host benchmark report

Status: **passed**

Command: `["/workspace/template/work/build-die/coarse_die_benchmark", "--config", "/workspace/template/die/tests/fixtures/coarse_die_v2.json", "--task", "/workspace/template/work/performance-cache-response/fixtures/attention_small_span.json", "--iterations", "30", "--owners", "1"]`

Benchmark executable SHA-256: `16ce959db28cccbbb6c9e44f5883e473cc3c29b45329804d21ae9defae105951`

| Measurement | p50 host ns | p95 host ns | Samples |
|---|---:|---:|---:|
| Cold file/parse/resolve/compile | 415647.0 | 1167441.0 | 30 |
| Hot parsed/resolved compile | 161003.0 | 235084.0 | 30 |
| estimate_alone | 238700.0 | 313593.0 | 30 |
| Shared runtime arbitration | 9264.0 | 32198.0 | 30 |
| Shared runtime CPU | 9000.0 | 33000.0 | 30 |

Advanced runtime events: 3; modeled finish: 337 ns; peak RSS: 13044 KiB.

Host benchmark time and modeled device latency are separate. Raw samples and runtime counters are in the JSON report.
