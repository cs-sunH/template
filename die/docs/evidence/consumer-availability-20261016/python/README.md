# Python consumer availability checks

Source base: `9bb8c0ac74ad3b78edb0004d7ebbc61f00942dda`, plus the six Python files listed in `verification.json`. Only FACE/FACE-LRU/JOINT `online/die_model_oracle.py` and their corresponding tests were edited. Scorers, the JOINT sh30 guard, C++ adapters, public `die/` source and vendor SDKs are outside this subtask's edits.

Coarse full/cost responses now require a string **top-level** `status`. A `not_available` row becomes `None` with a diagnostic in the same `last_errors` position before latency validation or seconds conversion. Valid siblings and native per-row errors retain their ordering. A malformed status raises `RuntimeError`; the result allocation is already freed by the existing `finally` block. Nested calibration/heldout statuses do not control prediction availability. Non-coarse behavior remains unchanged.

FACE's existing scorer rejects `None` with `RuntimeError` and propagates oracle exceptions. Its legacy arithmetic is entered only when no cost oracle was supplied, so unavailable rows do not trigger a legacy fallback. The optional absence of the coarse cost endpoint still chooses the **full coarse response**, where the same check applies. Regression tests exercise both choices without editing the scorer. JOINT's existing coarse projection guard remains intact.

Evidence chronology and actual results:

- `red-regression/`: FACE and FACE-LRU each ran 7 tests and exposed genuine semantic failures. JOINT failed module import because the first isolated staging omitted `sh_test_mesh/config_resolver.py`; this is **not** a semantic red result.
- `red-regression-complete/`: after staging the real shared Python dependencies, all three pre-fix consumers ran 7 focused tests each and failed as expected. Numeric `not_available` rows were still accepted; malformed status and conversion-order failures are retained in full stderr logs.
- `green-regression/`: after the consumer patch, all 21 focused tests passed, no skips.
- `green-all/`: FACE 70 and FACE-LRU 63 tests ran, but each had 5 staging-fixture errors (missing CSV/hardware JSON); JOINT 15 passed. These raw failures are preserved and are not production regressions or passes.
- `green-all-complete/`: after copying the real static fixtures, FACE **70/70**, FACE-LRU **63/63**, JOINT **15/15** passed; **148 total, no skips**. Python production/test source bytes did not change between the two green-all attempts.

The new regressions mock the underlying batch response and require no new SDK API or native library. They cover mixed good/unavailable/native-error rows, full/cost/optional-full fallback, malformed or missing status, rejection before latency conversion, `estimate_one`, buffer release, unchanged non-coarse behavior, and FACE scorer rejection/exception propagation. The JOINT existing valid-response fake was updated to include the status already returned by the real old SDK.

The wider FACE suites also use two pre-existing native libraries, without rebuilding either:

- Legacy: `work/native-dependency-check/elementwise-preview/build/lib/libdie_model_cabi.so`, SHA256 `fe16553813ac00874dfc5e1d0ea098236ea4fc06d1707567926b57b58137684b`.
- Coarse: `work/build-die-continue/lib/libcoarse_die_cabi.so`, SHA256 `38cba336f98c9c2f920326b0e0a11b9a7628123248b131add4c417dba2aa5d2d`.

Each run's JSON binds the exact staged source/fixture hashes, command, environment, library hashes and exit code. The `source/` folders preserve the three consumers and corresponding tests as actually staged in that run. The pre-fix consumers match Git `9bb8c0a`; added regression tests are deliberately already present in red. No model-accuracy, C++ integration, Online workload, CPU-budget, or formal SDK-release result is claimed here.

Re-run current relevant suites from the repository root using existing libraries:

```sh
source /workspace/cloud-setup/activate.sh
python3 die/docs/evidence/consumer-availability-20261016/python/run_checks.py green \
  --scope all --run-label green-all-rerun \
  --legacy-library /workspace/template/work/native-dependency-check/elementwise-preview/build/lib/libdie_model_cabi.so \
  --coarse-library /workspace/template/work/build-die-continue/lib/libcoarse_die_cabi.so
```

Use `--scope regression` without library arguments for SDK-independent focused checks. The runner requires a fresh `--run-label` and never overwrites previous evidence. To repeat red, stage the preserved pre-fix oracle bytes from `red-regression-complete/source/` with the corresponding saved tests; `run_checks.py red` labels the supplied checkout as pre-fix and does not reset the checkout.
