# Tiny curved-credit unit experiment — not adopted

`work/runtime-lab/` contains the active numerical experiment; source and full
text evidence are now also preserved in the two `runtime-subunit/` archives
under `die/docs/analysis/core-resume-20261016/` and
`die/docs/evidence/core-resume-20261016/`. Production source,
configs, formal vendors and the fresh public51 baseline library were unchanged.
No budget was run. The current hardware migration and its performance decisions
are separate from this experiment.

## Actual red, reduction and green scope

- `../norm-audit/current-baseline/identity.json` binds the unchanged archived
  tiny-credit test (SHA609e3567733737cb7f796b69239cc7900a75ac32ca43c3d8b507cd138afe1c48)
  to current Core headers/configs and this session's fresh static library
  e12c029d3a574b234a2bbcd52139d7ab94dccb040e281cfe71ab85fe31afa599.
  Actual baseline build0/run1, with the500ms boundary assertion failing.
- The independent three-activity reduction from the historical Norm epoch4 was
  actually compiled and passed the unchanged Runtime. Its full logs/identity are
  `../norm-audit/current-baseline/sync-*`. It is not a reproducer of the original
  full Norm deadlock. No constants were changed to manufacture a red.
- `focused-identity.json` binds candidate Execution SHA66f34d699d79b3c6dcffc5af04002799f3b1bfa4d4dcb569165825fc6a3f655a
  and its actual compiled object to11 successful serial functional runs:
  original tiny, unchanged synchronous reduction, new scaled-credit snapshot
  inventory/capacity guard, plus8 existing Runtime/allocator tests. This first
  focused phase was a static-object override, not C ABI coverage.
- `public-relinked-identity.json` then binds a separatePIC compilation of the
  same candidate source to an independently assembled static archive/C ABI,
  relinked public targets and their actual loader paths. Public51/51 passed in
  56.45s. Original tiny and scaled snapshot guard were additionally relinked to
  thatPIC archive and passed. Three registered Python/source-only tests do not
  execute Runtime; the CLI and owner-oracle driver use candidate artifacts.

This does not solve capacity1 Norm terminal residuals, the sub-epsilon dropped
event, full R>1 readiness, busy accuracy, or the separately observed5ULP saved-
allocation import rejection. None was changed or claimed fixed here.

## Minimal source change

`subunit-curved-credit.patch` applies only to Execution.cc.
`baseline/Execution.cc` and `candidate/Execution.cc` are complete source copies;
`manifest.json` records exact identities and scope.

The old guard `2e-10*max(1,capacity)` can exceed an entire small buffer. The
independent red has capacity4*unit, initialunit/3unit, signed credit2*unit and
one trillion operations over1s. At unit2^-60 the old absolute floor erases
positive inventory/headroom, even though it represents0.5s of physical work.

The new helper only narrows guards when the current epoch has a cumulative
credit curve and0<capacity<1. It uses `2e-10*capacity` there. Every caller still
evaluates its original tolerance expression; the helper returns that exact
original value for capacity>=1, capacity0 and all legacy affine epochs. No
threshold is enlarged. Snapshot capacity/inventory guards and terminal-closure
feasibility are narrowed in the same coordinates, so an invalid import cannot
erase the inventory that the allocator now retains.

The consistent helper is used for initial buffer validation, terminal residual
checks, symmetry boundary masks, singleton/signed-row constraints, credit
recomputation, advance/next-event boundary selection and snapshot credit/tangent
validation. The actual credit accumulation, double diagnostic representation,
fixed time epsilon, positive-event filter, progress closure threshold, physical
work, Q/alpha, PRIVATE, fairness and solver operations are untouched.

The two historical time/credit guard normalizations were deliberately preserved:
ordinary event selection still uses its original2e-10 threshold, while the
other ordinary-capacity guards retain their original max(1,capacity) factor.
The experiment does not claim to unify every pre-existing numerical decision.
Its demonstrated correction is the subunit coordinate floor in the actual red.

## Public regression artifact binding

`public-build/libcoarse_die.a` SHA:
81e3c232a0c76d3171b94b13e75ff2a66caddea408e855239a30606c0bc17d57.

`public-build/lib/libcoarse_die_cabi.so` SHA:
b48cf7288e87faca9ebdadb1b55c23d141d389ef88b632a7bbbfe2163092a342.

The archive was copied from this session's fresh library, its original
Execution.cc.o was explicitly deleted, and the newPIC Execution.cc.o inserted.
ar member hashes prove exactly one Execution member and byte-identical other
four members. The unchanged fresh Capi.cc.o was linked against this archive.
Every public executable was relinked from its real fresh link.txt; its static
archive, candidate C ABI path and rpath are recorded. ldd checks reject loading
the baseline C ABI. The copiedCTest registration points to these new artifacts;
CLI --library and the Python owner-LP driver's argument also point here.

These are valid affected relink regressions using this session's fresh objects,
not a claim that the candidate was rebuilt by a fresh final-source CMake tree.
Final adoption still requires that independent build, complete API/trace and
normal budget checks at the actual final production source.

## Preserved tooling failure

The first publicCTest launch had noctest on PATH and did not start any test.
It is recorded127/not_run_missing_executable with its empty original logs.
The real configuredCTest absolute path was used in a separate resumed step,
with separate complete logs `public-ctest51-resumed.*`. All relinks were already
complete and were not rerun or relabeled after that launch error.

`run_public_relinked.py` was subsequently corrected for future use to resolve
CTest from its configured CMake directory and recordOSError. The identity keeps
the SHA of the runner version used for the completed initial relink attempt;
the corrected file is not represented as a past executed source.

## Remaining numerical risks and next checks

The guard becomes more strict, but a relative tolerance can still discard small
remaining physical work near a boundary. This experiment is not a universal
positive-slack or arbitrary-unit proof. Double tolerance underflow at extreme
subnormal capacities becomes an exact-zero guard; that stricter domain was not
independently certified. Correctly scaled guards also do not make an arbitrarily
small time/progress increment representable.

Keep current red/green evidence, broaden meaningful unit-scale/coupled-owner
coverage, then address any exact-credit representation separately. A future
positive-event change must detect whether both clock and intended boundary state
can represent the step, maintain true signed tangent feasibility and bound the
number of internal transitions. It must not skip a physical event, close unpaid
work, widen a guard, or replace this analytical Runtime with K/tile/wave replay.
The archived sub-epsilon patch was not used. More detailed risks are in
`../norm-audit/sub-epsilon-review.md`.

## Coupled, split-owner and layered assessment

- Same PIC candidate archive as above, unchanged: balanced shared-buffer
  L/C/S 36/36 exact Fraction checks; same-owner split 6/12 passed, six failed
  snapshot imports (J2/J8 at all three units). Thus the main matrix remains
  **42/48**, not all passed. Fresh unchanged baseline with identical driver is
  **32/48**; ordinary-unit J2/J8 snapshot errors are byte-identical. Original
  snapshot/import and exact physical assertions are not weakened.
- Added genuinely asymmetric signed coupling: L/S have maximum rate2, C1.
  Empty requires (L,C,S)=(2,1,1)/J, full=(1,1,2)/J, interior=(2,1,2)/J.
  All36 cases (owners1/2/8/32, units1/2^-10/2^-60) passed exact Fraction
  initial/quarter/half/complete/cancel, true signed tangents, original union/row/
  shared busy and snapshot continuation checks. No tolerance was fitted.
- Exact physical observations across all three units match in28 fixture groups,
  including both failed-import prefixes. Those prefixes do not certify the
  missing cancel/completion. Baseline/candidate ordinary-unit physical fields
  match exactly for all32 cases at units1/2^-10.
- The independent layered assessment leaves original verdicts intact. For84
  candidate cases' actually observed samples, all13 resource capacities,
  PRIVATE, credit inventory and boundary tangents have no positive exact
  binary64 Fraction excess. Maximum relative rate/progress/byte error is
  1.7763568394002505e-15; rate error maximum8ULP,2.220446049250313e-16/s.
  Available busy and completion/cancel timestamps have zero error. No observed
  metric exceeds10%; this synthetic statement does not waive snapshot failures
  or imply target hardware accuracy/calibration.
- Snapshot max_digits10 machine values and terminal integer ledgers conserve
  exactly in this fixture domain. Three displayed sums differ because summing
  separately rounded binary64 values does not commute with the existing final
  cast of a long-double total. They are retained as projection diagnostics.
  The initial assessment that mislabeled these as ledger failures is preserved
  in `layered-attempt-1/`, alongside its original source and logs.
- `coupled-attempt-1/` preserves an earlier fixture/oracle expectation error:
  without the shared SRAM bound, L/S legitimately receive idle HBM service after
  C's gate bottleneck. Actual39/48 is retained, not called a Runtime defect.
  The later fixture explicitly declares SRAM work/capacity and has a separate
  identity; it does not rewrite that old failure or tune credit constants.

No new build or performance run follows this archive. Extreme subnormal/large
units remain not_run. Capacity1 Norm, positive sub-epsilon representation,
R>1 readiness, target accuracy and the saved-allocation snapshot bug remain
outside this candidate. Parent review and a final clean source build/API/budget
are still required before any production adoption.
