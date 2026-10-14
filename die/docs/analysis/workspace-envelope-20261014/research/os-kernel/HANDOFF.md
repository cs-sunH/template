# Scratch Solo13 kernel handoff

Freeze point: user-requested handoff on 2026-10-14. No production source was
changed by this task. The current repository contains other concurrent tracked
WIP; the artifacts below are isolated under this `work/` directory.

## Rebuilt baseline: full suite passed

`bash work/os-solo-cpp-13-20261014/build_and_check.sh` completed successfully
before the later profiling/reset additions. Its exact source and run identity is
in `pre-profile/reconstructed_baseline_validation.json`; the log is
`pre-profile/reconstructed_baseline_build.log`. It reports:

- `solo13_kernel.hh`: `bc21d044ad253308e4b1e155cfcee88027bbe4ba086716244e1f0f628a2b2dd7`
- `solo13_kernel.cc`: `affee811c9e8e9a956bc3b0ade26222fb60dbfcd75a06448fcb3e4ceb260fa65`
- `check_kernel.py`: `60587309e546b9e26331958bb07c1c59b208b37d2b6bda0106c2777e9d287d65`
- validated binary: `d4cf45542e6ee13ed3dc8e9fb08a8619352fdce24a2eb3e46e7b162c07b1de15`
- HEAD recorded by the run: `fecf81c5574755209c5d58ecdb982843ff349042`

The baseline suite covers strict FP32, native and converted INT4, dense/tail,
small independent Fraction prefixes, stop/resume, capacity invalidation,
large-K and large-output exact powering, and the deliberate unpowered failure
at exactly 256 expanded events. Here 256 is this scratch discovery budget; it
is not the Runtime solver's pivot count or a production certificate.

The historical `pre-profile/kernel_validation.json` and
`pre-profile/build_and_check.log` are retained as development trail only. Their
header hash (`9a236…`) does not have a matching source in this archive and is
not the rebuilt baseline proof. The root also noted an earlier historical
identity (`be1653…` header, `3614…` source, `de47…` binary); neither historical
identity is presented as rebuildable evidence.

## Current post-baseline scratch source

The current top-level `solo13_kernel.hh` adds `Kernel::reset()` and arithmetic
profiling guarded by compile-time `SOLO13_PROFILE_EXACT`. It compiled in both
benchmark builds and the sanitizer smoke build, but the full differential suite
was not rerun after that addition. Its hash is
`2b65003a1b5b82c8af377d8eb92db4695a0f2efc33af087141a369e1874b76e0`.

Relevant added files:

- `solo13_benchmark.cc`, `run_benchmarks.py`, `run_benchmarks.sh`
- `kernel_sanitizer_smoke.cc`
- `facts_input/`: copies of the ten facts JSON inputs used by the suite
- `benchmark_smoke.json`, `profile_smoke.json`, `sanitizer_smoke.json`
- `pre-profile/`: source/test snapshot and both historical and rebuilt baseline
  records

Both benchmark binaries compiled. Only one `large_outputs` sample was run for
each binary as a smoke check; no 30-sample benchmark was started. The ordinary
sample reported 69.1 ms cold full, 51.3 ms run-only full, 3.0 ms parse plus
construction, 76.5 ms cold stop/resume, and 66.8 ms run-only stop/resume. This
single sample is diagnostic only. The separate profile-enabled stop/resume
smoke counted 14,094 adds, 1,614 subtracts, 9,723 multiplies, 3,995 divides,
3,923 compares, and 9 floor calls. Explicit counters for `u128` to rational
conversion and multiply-repeat were added afterward, so those counters are
unmeasured. Inclusive operation timers overlap; they are not wall time or a
throughput claim. The host reported AMD EPYC 9V74, 3 visible hardware threads,
and 5,136/6,032 KiB process peak RSS for normal/profile smoke.

Smoke binary hashes: normal benchmark
`3bfc899a6241c5004db68222461d66d30bb816e975449c975b1d802a281404f8`, profiled
benchmark `a8b6d575777d58c1dc08b4b422f6b1f9f052fbc66c0d312bee9a2259d6959bd6`,
and sanitizer smoke `0386d197f3fdfa4b6ed05d7dd5c20f728c50de31991b458a6b41e5535406a056`.
The two benchmark smoke binaries predate the later explicit conversion and
multiply-repeat profiler counters; the final source added those counters but
was not recompiled/re-run under the profile macro. The ordinary benchmark
source had compiled and run before that instrumentation-only extension.

`perf` was not invoked. The sanitizer binary compiled with AddressSanitizer and
UndefinedBehaviorSanitizer, but the smoke run stopped at the deliberately
asserted stop-clamp case: an enormous `2^200` stop overflows the fixed 256-bit
normalized rational first (`normalized rational exceeds 256-bit state slot`),
before reaching the intended checked-u128 repeat-count branch. No sanitizer
diagnostic was printed. Geometry aggregate overflow and zero-capacity checks
were reached before that assertion; the smoke JSON redirection is empty and
should be treated as a partial result. The separate full suite already covers
the exact 256-event fail-closed boundary.

No tests, benchmarks, or source edits should be inferred as having run after
this freeze point.
