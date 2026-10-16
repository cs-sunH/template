# Final saved Core implementation: current-source validation

Recovered from remote `codex/die-single-core-20261016` at
`b8ddbcfd62e03738b0c3996444355cbbe2e99b80`, then merged the independent
`main` handoff `d3a64a6c4cb23b8ce437438ed68217d7cee4af1f` without replacing
the migrated source. Recovery checkpoint `3b653775842fff7fa43db3a0007029a8cbfe75c2`
was pushed and verified. All production source, headers, tests and configurations
still match the saved migration at this validation checkpoint.

Fresh Make/Release build and **51/51 CTest tests passed in 32.29 s**.
The DOCX audit passed against its original SHA and all 32 rows. A new current
resolver executable passed 192 explicit points (32 rows, three grids, two
topology profiles), checking DOCX installation counts, SRAM and `L0=a` geometry.
The full numeric results equal the immutable first-migration results; only
the fingerprint is excluded because the final source constant mirror changed.
This is an actual new execution, not reuse of the first 51-test result or ELF.

Both normal 30-sample performance runs **failed**. On the normal example,
128³ cold compile P95 was 2.587118 ms and OS J8 update 68.344 us. On the
authorized engineering point, cold compile was 2.568263 ms, alternating alias
56.093 us, residual cost 121.610 us, multi-domain J32 63.015 us, and OS J8/J32
104.229/277.039 us. No other build was running during either measurement;
the cgroup throttle count increased by zero in each run. This does not identify
the cause of wall-clock tails. No sample was discarded or replaced by a faster
rerun. Performance acceptance remains open.

Evidence is under `../../evidence/core-resume-20261016/`, especially
`public-checkpoint-summary.json`, `public-51-identity.json`, the two raw budget
JSON files and their command/source/library/CPU identities. Tests cover the
final input rejection, engineering-point fixture and existing capacity/work
conservation contracts; they do not certify all model accuracy or consumers.

Reproduction from a new independent build directory:

```sh
source /workspace/cloud-setup/activate.sh
cmake -S die -B work/build-core-release -G 'Unix Makefiles' -DCMAKE_BUILD_TYPE=Release
cmake --build work/build-core-release -j 2
ctest --test-dir work/build-core-release --output-on-failure
work/build-core-release/coarse_service_curve_benchmark --samples 30 --output work/budget-example.json
work/build-core-release/coarse_service_curve_benchmark --samples 30 \
  --config die/configs/tx8_compute_matched_hardware.json --output work/budget-tx8.json
```

The first attempted Ninja configuration failed because this new environment
does not contain Ninja. Its configure/build logs and identities are retained;
no test was run on that incomplete directory. Host capture initially also
failed when querying absent Ninja. A separate, genuinely new Make directory
then produced the tested binaries. The integration runner initially collided
its `build` log directory with its CMake build directory; no ABI process had
started at that point. These tool failures are not source test passes or
compiler failures.

The installed five-repository SDK remains the historical canonical24 release.
Do not publish it under a new identity or claim the new SDK, end-to-end CPU
budget, R>1 windows, Norm long rows, original busy precision or independent
target calibration has passed. Continue the final legacy W/H and isolated
current SDK checks. Engineering status is **NOT ACCEPTED**.
