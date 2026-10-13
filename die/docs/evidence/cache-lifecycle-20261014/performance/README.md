# C ABI cache-response performance retest

This directory contains the cache-response optimization retest. The source summary digest and actual source manifest are recorded in `environment.txt` and `source_manifest.*.txt`; the exact build binary hashes and environment are captured before and after measurement. The archived baseline remains unchanged in `../../runtime-waves-20261013/performance/`.

The three fixtures each have 30 C++ cold/hot recompile samples and 30 C ABI cold-miss/same-model-hit samples. The C ABI driver creates a fresh model per sample, verifies the hash and cache accounting remain stable on the repeated request, and releases both newly returned plan handles. Hit intervals include request JSON parsing and response serialization/allocation, exclude model creation and caller-side request/result processing, and match the baseline probe. The one-sample C ABI preflight is separate and excluded from statistics.

`summary.md` compares nearest-rank P50/P95/P99 against archived baseline and the 2 ms compile / 20 µs cache-hit targets. The probe is a small host-direction check, not full budget acceptance or contention testing. The C++ source path was unchanged for this C API response-cache adjustment, so C++ timing deltas are control data and are not attributed to the response cache.
