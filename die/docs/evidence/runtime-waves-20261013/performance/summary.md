# Current public build performance probe

Source snapshot digest: `e877a4ddc2dd5e0eff332fc8c01fcc33cc3ea92cfb8aac99c856b45a24f48b7a` (SHA-256, 20 synchronized vendor files).
Build: `Release`, `/usr/bin/c++` GNU 14.2.0, `-O3 -DNDEBUG`; binary and source hashes, CPU metadata, and load averages are in [environment.txt](environment.txt).

Each measurement below has 30 raw samples. Times are host wall-clock nanoseconds summarized in milliseconds; p95 uses nearest-rank `ceil(0.95 × 30)` (the 29th ordered sample). The raw JSON files preserve every sample.

| Fixture | C++ cold file/parse/resolve/compile p50 / p95 (ms) | C++ hot compile p50 / p95 (ms) | C ABI empty-cache miss p50 / p95 (ms) | C ABI same-task cache hit p50 / p95 (ms) | Cache entries / bytes |
|---|---:|---:|---:|---:|---:|
| GEMM 1×128×128 | 0.867 / 0.976 | 0.607 / 0.639 | 0.704 / 0.785 | 0.032 / 0.034 | 1 / 15023 |
| GEMM 128×128×128 | 3.390 / 4.012 | 3.051 / 3.377 | 3.227 / 3.538 | 0.035 / 0.036 | 1 / 15029 |
| Attention q=4, kv=16, 4/2 heads, d=32 | 0.418 / 0.473 | 0.162 / 0.191 | 0.275 / 0.398 | 0.040 / 0.042 | 1 / 16605 |

The C++ cold interval reads and parses task/config JSON, resolves hardware, then compiles. Its hot interval reuses parsed inputs and hardware but calls C++ `compile()` again; it is not a cache hit.

For each C ABI sample, a fresh model begins with an empty cache. The driver times the first compile-batch call, then submits the byte-identical task to the same model and times the repeated call. It checks stable plan hash, one cache entry, and unchanged accounted bytes; both returned plan handles are released before model destruction. The cache path is independently visible in `Capi.cc::cached()`: a found key returns the retained plan before the compile fallback. The C ABI hit measurement includes JSON request parsing and response serialization/allocation, while caller-side request construction and result parsing/free are outside the interval. Model creation and handle release are reported separately in raw JSON.

Across the six benchmark processes, reported peak RSS ranged from 3184 to 13056 KiB. The system load averages immediately before and after the sequence are recorded in `environment.txt`; this remains a small host-direction probe, not full budget acceptance. No 96-active L/C/S load test or LP-pivot claim is included.

Raw reports: `raw/cpp_compile_*.json` and `raw/cabi_cache_*.json`; exact commands: `commands.txt`; run transcript: `run-transcript.txt`; reproducible runner and validator: `run_probes.sh`, `summarize_probe_results.py`.
