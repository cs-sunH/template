# Pre-Core Norm/Runtime archive review

Read-only production review; no old numerical patch or Norm mapping was adopted,
compiled or executed. The later parent task superseded preparation of test copies,
so no current-tests/manual-reproducer source was created.

## Provenance verification

Source archive: `die/docs/analysis/pre-core-migration-20261016/`.

- unadopted-artifacts.tar.gz SHA256:
  `975e7a574f3a6ce322f340f2824c4ce12ebcaa35b0879d67e8d5612f0dee589e`.
- unadopted-tracked.patch SHA256:
  `6adf5e284706d8515e0567ef191bc861ce72b4c02379fa5633e70feea319e686`.
- All 656 archived files match the outer manifest. The nested Runtime manifest
  has 133/133 matching files. No missing or mismatched file was found.
- `manifest-verification.json` stores outer verification. Extraction root is
  this directory; original repo-relative archive paths follow below.
- Current production Execution.cc SHA256 during review was
  `3c9d58a0ed8c6b64450b86cac1059b3a1dba7b604caf67b31f00034c673e3043`,
  exactly the recorded unchanged Runtime baseline. This is source identity,
  not a fresh run of either archived regression.

## Independent Runtime red

`die/tests/runtime_credit_slack_test.cc:8-39` inside this extraction uses current
test_support helpers with synthetic manual work. It does not inject core_type.
Use units 1, 2^-10, 2^-60 in both emptying/filling directions, capacity4*unit,
initial1*unit or3*unit, signed delta -2*unit or+2*unit, private1s and 1e12 eqops.

The unchanged assertions are next_event500ms; at250ms progress0.25 and250e9
eqops; exact snapshot restore; at500ms progress0.5/nonterminal/no next event;
at750ms exactly500e9 eqops and500ms compute_busy; continuation snapshots equal;
cancel retains only actual service. Archived actual red is
`die/docs/evidence/norm-long-validation-20261016/slack-red-test.log`, failing
the first500ms next-event assertion.

## Synchronous Norm red and exact stalled state

`die/tests/norm_long_row_test.cc:219-247` compiles RMSNorm hidden65537,4 Core,
one BF16 input row, FP32 parameters/output; two distinct fair owners arrive at0.
Every loop must expose a next event, total<=64; each owner pays8*65537 eqops and
finishes no earlier than5.12s. Its line221 old core_type injection must be removed
when adapting to core32. The synthetic P_vec/rate override and fingerprint must
remain visibly synthetic. This test also depends on restoring the long-row
builder; an equivalent minimal manual curved-epoch regression can isolate the
Runtime issue first, but one was not constructed or claimed verified here.

Archived actual failure is `die/docs/evidence/norm-long-validation-20261016/
sync-liveness-red-test.log`: synchronous owners lose their last positive slack.
The full diagnostic snapshot in
`work/norm-review/runtime-archive-20261016/strict-projection.stderr.log` shows both
owners at apply-middle epoch4, now5130029297, Load13/Store15 done, Compute14 not
done with exact progress9.999999999997531472231e-01 and rate0, credit[0,0].
Remaining progress2.468527769e-13 exceeds progress-close2e-13. True input is
255*(1-p)=6.29474581095e-11, output=-254*(1-p)=-6.27006053326e-11. Original
2e-10 credit snapping erases both and removes the final consumer's event.

Current source relevant sites: Execution.cc:1889-1890 snaps credit;1262-1269,
1345-1349 and1401-1403 classify boundaries;1924/2044 use the double diagnostic
credit for next events;399-401 drops intervals<=1e-15ns.

## 92-case evidence actually inspected

`die/docs/evidence/norm-long-validation-20261016/mixed-initial/results/` has
92raw gzip files,91report gzip files and92driver stderr files. The only empty
raw and nonempty driver stderr is
`rmsnorm-h65537-balanced-c4-r1-bf16-fp32-fp32`: nonadvancing Runtime.

Recounting the retained reports gives455 completion comparisons, all<=10%,
max2.548221%;455first-output comparisons,6failures;273joint scenarios,258failed.
The six first-output misses are offset second-owner BF16/FP32/FP32 at full
consumer cohorts c1/r1,c2/r2,c4/r4: RMS12.99795%,Layer15.16932%. This is original
synthetic evidence, not a current Core validation or target-hardware calibration.
The later zero-reference correction was not rerun across the full matrix.

## Numerical proposals rejected or unverified

Read `work/norm-review/runtime-archive-20261016/RUNTIME-README.md`,
`archive-status.json`, individual Execution-*.patch and *-scratch-summary.json.

- strict-projection: original deadlock remained.
- global positive-slack: original driver completed, but legacy also changed and
  general regressions were not run; diagnostic only.
- curved-positive-slack: credit-scale input=-0.00551778488233889 and Norm
  output=-0.11475801280347738; rejected.
- curved-roundoff-slack:7/8 focused targets passed, same Norm output violation.
- curved-exact-credit:7/8passed, fourth mixed Norm tuple input=1.0622468484693504
  exceeded capacity1; rejected.
- curved-sub-epsilon: object compiled only; driver/tests/event amplification/
  budgets not_run. It is an unverified proposal, not the next proven solution.

`boundary-diagnostic.stderr.log` demonstrates output credit0.78024483619937,
netrate-223750712.2507/s and3.487116659ns integration to an actual residual
3.7079714299e-17. Its true remaining boundary interval1.6571886599e-16ns was
dropped by min_positive, so the public-tick remainder0.5128833408ns crossed to
-0.114758. Merely preserving positive credit or clamping arithmetic uncertainty
does not address a dropped real event. The exact candidate's gamma32 normal-
arithmetic estimate was not certified for subnormals or all segment extremes.

## Suggested dependency order

1. Bring the independent manual tiny-credit regression to current Core first;
   preserve its red and all physical-work/busy/snapshot assertions.
2. Isolate synchronous curve behavior independently or adapt long-row fixtures
   without relabeling old hardware evidence. Unify internal exact-credit boundary
   classification, arbitration, events and saved-rate tangent validation while
   retaining public wire, all original illegal-boundary checks and fairness.
3. Retaining every real positive event requires a proof/check of represented
   time/progress advancement and bounded events. Do not enlarge tolerances or
   complete unpaid work to avoid the red.
4. Adapt long-row BF16/FP32 eight-role combinations, per-Core capacity/prefix
   routes, pause/cancel/snapshot, nonzero startup; retain R>1/rows>Core and INT4
   unavailability. INT4 scale capacity/traffic remains absent from this scope.
5. Only then evaluate `work/norm-statistics-variant/`: fixed4statistics epochs
   plus5apply epochs,15physical descriptions or16withlaunch. Its36ledger,
   144FractionPRIVATE and112small-offline checks are static; no C++Runtime
   comparison, model improvement or budget was run. Hard-coded old epoch indices
   need adapting; old busy failures remain failures without new evidence.

No previous numerical patch is safe to apply wholesale.
