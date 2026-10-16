# Archived scratch experiment, not adopted

The README summarizes actual results. Evidence is the sibling docs/evidence/
core-resume-20261016/runtime-subunit tree. Original identities retain absolute
work paths and source hashes; artifact-manifest.json maps them to these copies.
Full large driver stdout files use lossless gzip, with original and compressed
SHA256 recorded. No ELF, library, object, cache or previous account artifact is
included. Production code/config/tests and formal vendors were not changed.

Current fixture drivers are under tests/, oracles and runners under runners/.
asymmetric_credit_driver.cc includes coupled_credit_driver.cc from the same
tests directory only to reuse input/observation helpers. Fraction expectations
are independent explicit formulas. For the original runner layout, restore
these files under work/runtime-lab, original tiny/sync inputs under
work/norm-audit/current-baseline, and use a genuinely fresh current-source public
build at work/build-core-release. Do not treat missing libraries as permission
to reuse old ELF. The checked source/headers/configs live in the repository;
their exact hashes are retained in each identity. No restoration was performed
by this archival script.

The source in coupled-attempt-1 and layered-attempt-1 is retained verbatim for
the original failed/mistaken test interpretation. It is not the final oracle.
Evidence/baseline/README.md is an older preparation note that said synchronous
was unrun. Its actual later sync-identity.json and full logs show build0/run0;
the actual result is pass/non-reproducer, as the current README states.

runner-source-status.json explicitly identifies unsaved executed wrappers and
the corrected future public-relink runner. No missing old source was invented.
The missing-ctest launch127 was not a test execution. The resumed51/51 used the
candidate C ABI and candidate static library, verified by member/ELF/loader
identities, but is a relink regression using this session's unchanged fresh
objects, not a final clean CMake build. Source-only/Python tests are identified
in public-relinked-identity.json and are not claimed to execute Runtime.

Combined candidate exact matrix:78/84 passed,6 failed; do not rewrite to84/84.
The six failed split imports remain not_run for cancel/completion. Separate
synthetic relative-error diagnostics meet10% by a very large margin on observed
samples, without waiving exact assertions, snapshot errors or correctness.
Original machine-defined snapshot/terminal ledger checks pass with no new
guard. Three public projected sums are non-additive after separate binary64
rounding; original mistaken assessment retained before correcting that method.

Open risks: arbitrary unit invariance is not proved; extreme subnormal/large
units not_run; relative guards may still erase tiny physical remainder. The
candidate leaves positive-event/time epsilon/progress thresholds unchanged,
does not address capacity1 Norm or saved-allocation snapshot reconstruction,
and has no final clean source build/API comparison/performance budget. It is
not target hardware calibration, SDK publication or engineering acceptance.
