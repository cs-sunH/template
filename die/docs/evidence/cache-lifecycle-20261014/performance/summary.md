# C ABI cache-response retest

Integrated source summary digest: `05c3f91c59e4ac3d22b5147a12566d93902814aa77bbf3384a9b7a5468a9e755`. The independently captured per-file SHA-256 manifests, working-tree revision, binary hashes, environment, and load/RSS metadata are in [environment.txt](environment.txt). The archived baseline remains in [../../runtime-waves-20261013/performance](../../runtime-waves-20261013/performance); its e877 digest is retained there.

Release build (`-O3 -DNDEBUG`), three fixtures, 30 measured samples per metric and shape. Probe logic and caller timing boundaries match the archived baseline: C++ cold/hot calls follow the benchmark tool; each C ABI sample creates a fresh model outside compile timing, measures an empty-cache miss followed by the same byte-identical request on that model, includes request JSON parsing and response serialization/allocation in each `compile_batch` call, and excludes request construction, result parsing/free, model creation, and plan release. Model create/release remain separate metrics. There is no in-loop warmup in either original probe; one-sample C ABI setup preflight files are kept separate and excluded from all statistics.

P50/P95/P99 use nearest-rank `ceil(p×30)` to preserve the original probe's convention: rank 15, 29, and 30 respectively. P99 is the maximum for 30 samples. Raw samples, process CPU/RSS metadata from the standard-library outer runner, and probe-reported peak RSS are retained. Threshold flags below compare each percentile with the specified 2 ms cold compile or 20 µs cache-hit target; they are host-direction results and are not full budget acceptance.

| Fixture | Metric | Current P50/P95/P99 | Baseline P50/P95/P99 | Δ P50/P95/P99 | Target check P50/P95/P99 |
|---|---|---:|---:|---:|---|
| GEMM 1×128×128 | C++ cold parse/resolve/compile | 0.875/0.957/1.231 ms | 0.867/0.976/1.205 ms | +0.9%/-1.9%/+2.2% | pass/pass/pass (2 ms) |
| GEMM 1×128×128 | C++ hot recompile | 0.604/0.634/0.786 ms | 0.607/0.639/0.654 ms | -0.5%/-0.8%/+20.1% | pass/pass/pass (2 ms) |
| GEMM 1×128×128 | C ABI cache miss | 0.674/0.754/0.865 ms | 0.704/0.785/0.899 ms | -4.3%/-4.0%/-3.8% | pass/pass/pass (2 ms) |
| GEMM 1×128×128 | C ABI same-model cache hit | 0.009/0.010/0.011 ms | 0.032/0.034/0.087 ms | -72.1%/-69.4%/-87.2% | pass/pass/pass (20 µs) |
| GEMM 128×128×128 | C++ cold parse/resolve/compile | 3.355/3.788/3.851 ms | 3.390/4.012/4.467 ms | -1.0%/-5.6%/-13.8% | fail/fail/fail (2 ms) |
| GEMM 128×128×128 | C++ hot recompile | 3.090/3.132/3.137 ms | 3.051/3.377/3.393 ms | +1.3%/-7.3%/-7.5% | fail/fail/fail (2 ms) |
| GEMM 128×128×128 | C ABI cache miss | 3.248/3.861/3.921 ms | 3.227/3.538/3.670 ms | +0.6%/+9.1%/+6.8% | fail/fail/fail (2 ms) |
| GEMM 128×128×128 | C ABI same-model cache hit | 0.011/0.014/0.017 ms | 0.035/0.036/0.049 ms | -68.2%/-60.5%/-64.7% | pass/pass/pass (20 µs) |
| Attention q=4, kv=16, 4/2 heads, d=32 | C++ cold parse/resolve/compile | 0.416/1.167/1.528 ms | 0.418/0.473/0.791 ms | -0.6%/+146.9%/+93.1% | pass/pass/pass (2 ms) |
| Attention q=4, kv=16, 4/2 heads, d=32 | C++ hot recompile | 0.161/0.235/0.239 ms | 0.162/0.191/0.210 ms | -0.8%/+22.9%/+13.9% | pass/pass/pass (2 ms) |
| Attention q=4, kv=16, 4/2 heads, d=32 | C ABI cache miss | 0.256/0.287/0.392 ms | 0.275/0.398/0.537 ms | -7.1%/-28.0%/-27.0% | pass/pass/pass (2 ms) |
| Attention q=4, kv=16, 4/2 heads, d=32 | C ABI same-model cache hit | 0.017/0.019/0.036 ms | 0.040/0.042/0.066 ms | -58.3%/-55.6%/-44.6% | pass/pass/fail (20 µs) |

The C++ source path is unchanged for this C API response-cache adjustment, so C++ cold/hot deltas are control measurements only and are not attributed to response serialization caching. C ABI miss measurements include actual compile work and can move with host/build differences; only the same-model hit comparison isolates the cached response path most directly.
Two setup attempts stopped before measured samples because `cmake` was absent from PATH and `/usr/bin/time` was unavailable; their partial logs are retained as setup-attempt-* files. The completed run used the bundled CMake for environment metadata and a Python resource wrapper, and all six measured reports passed validation.

The probe verified one accounted cache entry, stable plan hash and cache bytes on the repeat, distinct nonzero plan handles for miss and hit, and successful handle release. Cache bytes are reported from the API; this run does not independently inspect the internal 4096-entry/64-MiB eviction boundary. Source manifest equality before/after and binary SHA equality are recorded in environment metadata.

Measured-process peak RSS ranged from 9344 to 13056 KiB. Load averages before/after, CPU model/core visibility, per-process user/system CPU and maximum RSS, and exact commands are retained. This remains a small host-direction probe, not a full contention/budget test.

Artifacts: `raw/` contains six 30-sample reports plus external CPU/RSS records; `preflight/` contains the three one-sample C ABI checks; `commands.txt`, `run.txt`, `environment.txt`, source manifests, and the reusable runner/probe/validator are saved beside them. No baseline files were changed.
