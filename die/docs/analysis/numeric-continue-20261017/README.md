# Numerical continuation, 2026-10-17

Recovered the actual latest `9c464ca` branch, not only main or the older
`2750e24` Runtime handoff. Merge `931855f` preserves main's additional shared
pointer. Verified the stop archive and all 499 restored text-file hashes.
Production remains A+B; no historical candidate was installed.

## C2: actual mixed-readiness rejection

Baseline and C2 libraries were each built from all source files in fresh Release
directories, without old ELF/object reuse. The original six-case review driver
had duplicate activity ID1 in its final two-epoch clock fixture. Only the second
ID was corrected to2, preserving the original assertions. An independent seventh
control keeps a curved second epoch active after a predecessor finishes just
before1ns, while a legacy activity is queued until exact integer1ns.

C2 passes the six corrected original controls but **fails the new mixed control**:
the legacy activity has already received service at its exact ready tick. Its
per-Plan epsilon readiness still starts legacy service before the external ready
boundary. Baseline passes both ready controls but fails four time/progress
controls; these failures remain in the raw output. C2 is rejected, not installed.
These are synthetic correctness tests, not calibration or accuracy certificates.

Commands and exact outputs are in
`../../evidence/numeric-continue-20261017/c2-review/manifest.json` and `summary.json`.
Compressed files decompress byte-for-byte to the original path/hash. No binary
or build cache is committed. Fresh build commands are in the recorder identities.

```sh
source /workspace/cloud-setup/activate.sh
python3 die/docs/analysis/numeric-continue-20261017/run_probe.py \
  --source die/docs/analysis/numeric-continue-20261017/time_only_review_driver.cc \
  --output work/a-new-review-output \
  --libraries baseline=work/build-numeric-baseline-20261017/libcoarse_die.a \
              c2=work/build-numeric-c2-20261017/libcoarse_die.a
```

`run_probe.py` preserves nonzero results and refuses an existing output directory.
Do not reuse its sample build paths without actually building identified sources.
The archived production recorder hashes the production tree; candidate identity
also requires the explicit candidate source/preparation hashes and actual library.

C3 preparation is isolated in `work/`: exact integer readiness for every Plan and
checking each selected physical credit participant, retaining the original credit,
allocation and closure rules. It has no adoption or whole-project acceptance yet.
Five-repository native continuation uses a separate fresh freeze; formal pinned
canonical24 SDKs are unchanged. Capacity1 Norm, structural busy error, R>1
readiness, speed budgets and target calibration remain unresolved.
