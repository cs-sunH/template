# Current-Core Runtime A+B: fresh joint checkpoint

This is continued development, not a stop handoff, SDK release, or engineering
acceptance. The working branch is `codex/die-runtime-continue-20261016`.

## Recovery and source

All remote heads were fetched. Main was `94d197820ad92d7671913e353e1977b0d54a99e2`,
the saved recovery branch was `2750e2413b030524a56466da5e8c9b02a9d029c2`, and
their actual trees differed only in Execution.cc, its private overload, and
the tiny-credit test. A normal merge retained both histories at `18de6809`.
Every remote head was an ancestor of this merge. The initial workspace was
clean but stale (`9fdcad1`); no reset, force push, or old architecture restoration
was used. The unchanged authoritative DOCX SHA is `b4382bc7…b4cae4ad3`.

The saved A+B source/header exactly match the previous stop archive:

- Execution.cc: `6fcc637e13c876d240d92d8677e8f9d2e323b325aea46176848063ffe93e710e`
- Execution.hh: `4ae56348aab0f39c6e67ecc28fe157cf6aaffd9478ac107af4b2f1d3003f0d6a`

The existing tiny test was registered. The final archived split-owner C ABI
test and Core32 fixture were copied byte for byte and registered. No model,
time/progress threshold, physical work, fairness, or existing guard was changed
in this checkpoint. B retains one fixed tautological-row reconstruction, the
original one-ULP match and strict tangent guard, and the shared 256-pivot bound.

## Actual current-source results

Evidence: [public-ab](../../evidence/runtime-continue-20261016/public-ab/).
`artifact-manifest.json` binds original text and losslessly compressed copies;
no ELF, archive library, object, build cache, or package binary is committed.

- Clean Release all-source build succeeded (GCC14.2, CMake3.31.10).
  Complete CTest **53/53 passed, 29.65s**. This includes the DOCX source check.
- An independently extracted and freshly compiled main baseline was used to
  compile the same two new tests. Both actually failed: tiny unit2^-60 has
  four failed subcases, and split-owner import and actual fork both reject
  the original approximately five-ULP reconstruction disagreement. All logs,
  fresh build/source/library/loader identities and failures are retained.
- Exact public capture: two hardware configurations ×28 inputs, eight real
  competition/cancel/capacity traces. Every static result and every ordinary
  trace state agrees field for field; seven existing successful fork paths
  agree too. The sole behavioral change is the engineering split-owner fork
  now succeeding and copying the exact source snapshot. All28 C ABI exported
  function symbols are identical. `compare_api.py` was actually executed.
- Separate old-library source/new-library import replay passed at execution
  counts2/4/6/8, with full controlled/uncontrolled event continuation and seven
  malicious edits rejected. Its unchanged archived script's report contains
  historical mixed-object wording; the new command recorder and fresh build
  identities define this actual run. No old ELF or mixed objects were used.
- Immutable exact-Fraction matrices: coupled42/48 and asymmetric36/36,
  hence **78/84 passed, six failed**. All84 observations now reach their
  intended snapshot/cancel/completion stages. The six split cases retain
  strict exact-equality failures from ordinary LP binary64 rounding; they
  are not relabeled passes. The unchanged independent layered assessor finds
  zero exact binary64 physical/PRIVATE/inventory/tangent violations, zero
  machine-ledger failures, and no observed metric above10%; maximum relative
  rate/progress/byte error is1.7763568394002505e-15. This is synthetic evidence,
  not target calibration. Original projected-sum discrepancies remain recorded.
- Fixed-witness guards passed: a merely feasible four-ULP inward rate without
  the eligible fixed row still rejects; >32 owners and >64 extra rows reject
  atomically. The registered test also rejects an outward one-ULP tangent.

## Normal performance remains failed

Both normal30 runs were executed once, after all builds and other probes had
ended. Every raw sample and threshold is retained; neither run added a cgroup
throttle. No fastest rerun was selected and no gate was changed.

| Configuration | Failed fixed gate | P95 |
|---|---|---:|
| example | 128³ cold compile | 6.728323ms |
| example | residual cold / hot cost | 2.591824ms /70.956us |
| example | heterogeneous J8 update | 100.779us |
| engineering | 128³ cold compile | 2.343980ms |
| engineering | SwiGLU cold / hot cost | 2.602414ms /53.267us |
| engineering | compiled OS J8 update | 69.194us |

These observations do not establish a unique host cause or performance
acceptance. Compound fixtures and other metrics remain in the complete reports.

## Reproduce

Use a fresh output/build directory. The scripts deliberately refuse overwriting
evidence. Environment setup is not assumed to survive an account change.

```sh
python3 -m venv work/tool-env
work/tool-env/bin/pip install cmake==3.31.10 ninja==1.13.2
work/tool-env/bin/cmake -S die -B work/build-runtime-ab-release -DCMAKE_BUILD_TYPE=Release
work/tool-env/bin/cmake --build work/build-runtime-ab-release -j2
work/tool-env/bin/ctest --test-dir work/build-runtime-ab-release --output-on-failure
work/build-runtime-ab-release/coarse_service_curve_benchmark --samples 30 --output work/budget-example.json
work/build-runtime-ab-release/coarse_service_curve_benchmark --samples 30 --config die/configs/tx8_compute_matched_hardware.json --output work/budget-tx8.json
python3 -B die/docs/analysis/runtime-continue-20261016/run_matrix.py --build work/build-runtime-ab-release --output work/new-runtime-matrix
```

For the main red comparison, extract only `die/{src,include,configs,CMakeLists.txt}`
and the JSON helper from the pinned `94d1978` commit into
`work/runtime-baseline-source`, configure `COARSE_DIE_BUILD_TESTS=OFF` in a new
`work/build-runtime-baseline-release`, build it, then run `run_baseline_red.py`.
`run_command.py` captures unfiltered stdout/stderr, actual sources/libraries,
command, clocks and cgroup stats. See archived identities for exact commands.
`capture_storage_candidate.py` accepts explicit library/source/output paths;
run it once per library and pass its two captures to `compare_api.py`.

## Continue

Five actual SDK/consumer trees were copied into `work/runtime-ab-sdk` and frozen
before validation:33 public files, digest
`25c51345c5ddcedb45b21c8b4dc89805c1f9ea2166d49792ddaf639ccbf91013`.
Five real protoc generations preceded the4493-file manifest. The formal vendor
SDKs remain canonical24/digest23160c…7e9b; no publisher ran. New affected ABI and
consumer verification is in progress, not covered by old migration evidence.
The fresh environment has no sudo; Protobuf compiler/development dependencies
were unpacked under work/, with URLs/package hashes preserved, and Python
protobuf6.33.5 installed in the isolated tool venv.

The complete seven-epoch current-Core Norm fixture has separately reproduced
the same capacity1 deadlock in both fresh baseline and A+B. It also exposes a
real small negative physical output credit hidden by boundary snapping. That
ongoing isolated work is not included in A+B's passing public scope. Do not
broaden progress closure, erase unpaid work, adopt old failed numerical
patches, or claim Norm/subepsilon/R>1/original-busy accuracy is solved.
Engineering remains **NOT ACCEPTED**; independent target data remain unavailable.
