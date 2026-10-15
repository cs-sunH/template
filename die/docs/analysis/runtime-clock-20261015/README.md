# Runtime clock checkpoint — 2026-10-15

This checkpoint archives the observed clock-translation regression, its old-library baseline, final probe outputs, build/test logs, the diagnostic K-readiness scenes, and five-ABI validation. The archived payloads are byte-for-byte copies compressed with gzip `mtime=0`; `manifest.json` records each original SHA-256 and compressed SHA-256. No executable, object, static/shared library, build tree, credential, or secret is included. Binary hashes are recorded for identity only.

Verify the compressed payloads from the repository root:

```sh
(cd die/docs/evidence/runtime-clock-20261015 && sha256sum -c SHA256SUMS)
```

`manifest.json` records the actual compile, K-readiness, five-ABI build, and config-test command arrays when they were captured. The original public-runtime probe build used GCC 14.2, `-std=c++17 -O2`, the current `die/include`, the Astra-Sim helper include, and `work/build-clock/libcoarse_die.a`. The probe source and the included readiness-driver source are archived under `raw/work/runtime-clock-replay-final/`.

## Findings

The old absolute-scalar clock is rejected by the regression: `clock-baseline-test-status.json` records exit `-6`, the expected relative-clock assertion, and `abi_consistent: true`. Its linked library was `work/build-k-readiness/libcoarse_die.a`; the corresponding old `Execution.hh` snapshot is archived. The baseline middle-pair output shows M=16,N=1,K=64 at 43,521 ns from starts 0 and 50,000,000 ns, but 43,521 ns at 1,000,000,000,000 ns, then 43,520 ns at 2^53−1 ns and 2^63 ns. M=1,N=64,K=64 is 166,400 ns at start 0 and 166,401 ns at 10^12 ns and 2^53−1 ns, returning to 166,400 ns at 2^63 ns. The final integer-origin implementation reports 43,521 ns and 166,400 ns respectively at all five tested origins.

The final isolated-pair probe compares 4 GEMM classes × 5 absolute origins × 32 repeated nonfinal middle pairs. For each class, the complete 32-cycle relative record arrays match exactly across all five origins, including event offsets, finish times, rates, alphas, and serialized busy counters. The within-origin `all_cycle_records_equal` flag is false for every origin: small per-cycle busy-counter tail differences remain. This evidence establishes origin translation for these tested records; it does not establish a constant record across cycles, justify skipping cycles, or certify a production K granularity.

The corrected translation regression reports 72 translated real-fact runs and covers partial import, cancel/capacity, legacy fractional serialization, and the `u64` edge. Two earlier fixture attempts are retained as failures: the first supplied a non-object `Plan.task`, and the second treated the legacy fractional clock value incorrectly. They are followed by the passing corrected run. The v2/v3 snapshot wire representation remains unchanged.

The full build log reaches 100%, and CTest records 33/33 passed, zero failures, zero skips, in 15.53 seconds. The nine diagnostic K-readiness scenes are three shapes, each with standalone, simultaneous two-owner, and 50 ms-offset two-owner comparisons; the captured reports mark those comparisons passed. They explicitly mark production granularity validation `not_run`, use diagnostic stage Plans, and do not execute the compiled baseline GEMM plan. Treat them as diagnostic comparisons only: there is no production E5 certificate and no target-fit claim.

All five ABI variants built successfully. Their config suites report 20 + 15 + 18 + 17 + 13 = 83 passing tests, exit code 0 for every suite, and no skips in the captured test logs. The canonical source digest is `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b` across the five repositories.

The measured toolchain was GCC 14.2 on x86_64 with a 64-bit `long double` significand (x87 extended precision), CMake/CTest 3.31.10, and Python 3.12.14. Results do not extrapolate to other compiler, floating-point ABI, or platform combinations.

## Replaying from a checkout

From the repository root, rebuild the tested library and run CTest. The defaults use this workspace’s bundled CMake/CTest; set `CMAKE=cmake CTEST=ctest` when those tools are on `PATH` in another checkout.

```sh
CMAKE=${CMAKE:-work/tool-env/bin/cmake}
CTEST=${CTEST:-work/tool-env/bin/ctest}
"$CMAKE" -S die -B work/build-clock \
  -DCMAKE_BUILD_TYPE=Release -DCOARSE_DIE_BUILD_TESTS=ON
"$CMAKE" --build work/build-clock --parallel 2
"$CTEST" --test-dir work/build-clock --output-on-failure
```

To run only the clock-translation regression after building:

```sh
"$CTEST" --test-dir work/build-clock \
  -R '^coarse_runtime_clock_translation_test$' --output-on-failure
```

The portable public-Runtime pair replay is the repository script used for this checkpoint. It rebuilds the small probe against the selected library and writes the results under `work/runtime-clock-replay/`; it applies no cycle skipping.

```sh
python3 die/docs/analysis/runtime-clock-20261015/replay_pair.py \
  --build-dir work/build-clock --output-dir work/runtime-clock-replay
```

The archived result is `raw/work/runtime-clock-replay-final/result.json.gz`; the exact recorded compile command and its hashes are in `manifest.json`. This is an isolated load/compute numerical probe, not a production lifecycle validation. Compare serialized long-double busy counters only under the recorded toolchain ABI.

## Evidence map

`die/docs/evidence/runtime-clock-20261015/manifest.json` inventories every archived input, its original path, raw size, raw hash, and deterministic gzip hash. `raw/work/` contains the baseline/final pair results, fixture and regression logs, CTest/build logs, all three readiness raw/grouped reports and manifests, and five-ABI build/config logs and result manifests. `raw/die/tests/` contains source snapshots needed to identify the regression and diagnostic driver. The final probe executable, temporary build outputs, and linked libraries are intentionally not archived.

The final 72-case source was also compiled against the matching old header and old static library again: `clock-final-source-old-lib.json` records source SHA-256 and the exact command, with expected exit -6. The first standalone compile omitted the test helper ROOT macro and failed to compile; its log is retained, followed by the corrected successful build and failing old-clock run.
