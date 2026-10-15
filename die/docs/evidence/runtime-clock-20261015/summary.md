# Runtime clock evidence summary

This archive is a read-only checkpoint of existing outputs. The compressed payloads are byte-for-byte source copies in deterministic gzip form; `manifest.json` has original and gzip SHA-256 values, exact captured command arrays, source hashes, and build-library hashes. No executable or build library is included.

## Clock regression and final probe

| GEMM shape (M,N,K) | Old absolute-clock observation | Final integer-origin period at all five origins |
|---|---|---:|
| (16,64,64) | At 2^63 ns, compute relative finish changed from 828,438 to 828,437 ns | 828,438 ns |
| (16,1,64) | 43,521 ns at 0 and 50 ms; 43,521 ns at 10^12 ns, then 43,520 ns at 2^53−1 ns and 2^63 ns | 43,521 ns |
| (1,64,64) | 166,400 ns at 0; 166,401 ns at 10^12 ns and 2^53−1 ns, returning to 166,400 ns at 2^63 ns | 166,400 ns |
| (1,1,64) | At 2^63 ns, compute relative finish changed from 810 to 809 ns | 5,120 ns |

The tested origins are 0, 50,000,000, 1,000,000,000,000, 2^53−1, and 2^63 ns. `result.json` records 32 cycles per origin for each of four classes. Exact array comparison shows each class's full relative cycle records match across all five origins (640 records total). Each origin's `all_cycle_records_equal` flag remains false because serialized per-cycle busy-counter tails vary slightly. There was no cycle skipping, and these measurements are not a production E5 certificate or a target-fit result.

The baseline translation regression aborts on its expected “relative completion/progress/busy/served must not depend on absolute clock” check. Its status records `exit_code: -6` and `abi_consistent: true`. The archived old-header snapshot and linked-library path identify the baseline setup. The final regression log reports 72 translated real-fact runs, partial import/cancel/capacity, legacy fractional wire, and the `u64` edge. Legacy v2/v3 wire representation is unchanged.

Two earlier translation-test fixture failures are retained: `Plan.task: expected object`, then `legacy fractional clock roundtrip`. The corrected test log follows those failures and passes.

## Other captured verification

- Full build: `clock-all-build.log` reaches 100%.
- CTest: 33/33 passed, zero failed, 15.53 seconds; see `clock-ctest.log`.
- K-readiness: 3 shapes × standalone/simultaneous/50 ms-offset scenes; the 9 captured diagnostic comparisons are marked passed. Reports say `production_granularity_validation: not_run`, and explicitly exclude compiled-plan execution and production certification.
- ABI/config: all five variants built; 83/83 config tests passed (20, 15, 18, 17, 13), exit 0, no skips in the captured logs. The common canonical source digest is `23160c512203596611d5c534104f5dd2015be2ef75a2665c88ffe1cfddcc7e9b`.

Toolchain: GCC 14.2.0, x86_64, `long double` significand 64 bits, CMake/CTest 3.31.10, Python 3.12.14. Do not extrapolate floating-point details across platforms or ABIs.

## Key raw artifacts

- `raw/work/middle-pair-results.json.gz` and `raw/work/middle-pair-clock-final-results.json.gz`: baseline and final middle-pair outputs.
- `raw/work/runtime-clock-replay-final/result.json.gz`, `identity.json.gz`, and the two `.cc.gz` files: final 4 × 5 × 32 probe and its build identity/source.
- `raw/work/clock-baseline-test-status.json.gz` and `clock-baseline-test.log.gz`: old-header regression failure and ABI status.
- `raw/work/clock-translation-test*.log.gz`: both preserved fixture failures and corrected pass.
- `raw/work/clock-ctest.log.gz` and `clock-all-build.log.gz`: complete CTest and full-build outputs.
- `raw/work/clock-k-readiness/`: three raw runs, three grouped reports, run manifest, and build/config logs.
- `raw/work/clock-five-abi/`: five build/config manifests, per-variant logs, and all test logs.

The final 72-case source was also compiled against the matching old header and old static library again: `clock-final-source-old-lib.json` records source SHA-256 and the exact command, with expected exit -6. The first standalone compile omitted the test helper ROOT macro and failed to compile; its log is retained, followed by the corrected successful build and failing old-clock run.
