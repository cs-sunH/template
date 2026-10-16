# Current Core snapshot allocation audit (2026-10-16)

Read-only diagnosis against the fresh public migration build. No production,
SDK, feasibility tolerance, ULP policy or performance result was changed.
`identity.json` binds actual commands, diagnostic sources, binaries, original
capture, production source/library hashes and complete output hashes. The
baseline capture is not relabeled as a successful snapshot test.

## Reproduction and exact result

`failed-snapshot.json` is extracted from the tx8 engineering configuration,
8-execution split-owner trace in the original `api-baseline.json.gz`. All 8
executions use only the original first two tasks; adjacent GEMM/Norm executions
share one of four fair owner keys, request IDs remain distinct. No workload,
shape, geometry, precision, owner, timing or Plan was reduced to get a pass.
The public events are 1000 ns and 1165 ns.

`reproduce.py` performs (1) direct C ABI import of the captured snapshot,
(2) fresh compilation/submission and replay through those two real events,
(3) import and fork of that newly created world. The newly generated snapshot
is exactly equal to the original captured snapshot. All three original
import/fork operations return status 1 and a zero output handle. The live
parent remains exactly unchanged. The error is:

```
id=4 saved_rate=7.862645685395645909011e+06
     rebuilt_rate=7.862645685395650565624e+06
saved_alpha=rebuilt_alpha=6.878048780487804325645e-01
```

`count_probe.py` retains the original two-task/paired-owner arrangement and
checks counts 2,4,6,8. The first three fork successfully; 8 fails. This is only a
bounded count reduction, not a proof of a globally minimal failing workload.

Two initial probe-tool errors are preserved: v1 used the wrong shared-library
path and failed before loading it; v2 tried destroying an active diagnostic
runtime and hit the correct busy guard. The final tool drains diagnostic/live
worlds before destruction. Neither error is classified as a simulator failure.

## Root cause

The last real allocation was **signed LP: 8 variables, 16 rows, 9 pivots,
1 feasibility rounding**. Each Norm starts its current epoch with input
credit=capacity=1. Its sole active derivative is -1, so the high-boundary
constraint added by `Runtime::allocate()` is `-rate <= 0`. This row is
mathematically redundant because rate is already nonnegative. Four Norms add
four such rows. They nevertheless change solver dispatch from nonnegative
packing to signed LP.

The Runtime then integrates the remaining fraction to public integer tick1165
without a real event or membership change. It correctly retains its published
allocation for this stable interval. Norm credit is now 0.9966302947062587, so
those four high-boundary flags are no longer set. Snapshot import reconstructs
only the current state, omits the four rows and selects **packing: 8 variables,
12 rows, 0 pivots**. All Plan fields, owner weights, segment derivatives and
physical/PRIVATE constraints remain equal. There is no true feasible-set or
credit-tangent change.

The old signed LP produces these binary64 values:

- handle1: `0x1.dfe5d6bdd85b3p+22`;
- handles2..8: `0x1.dfe5d6bdd85b9p+22`.

Current packing rebuilds all eight to `0x1.dfe5d6bdd85b8p+22`. The first differs
by 5 ULP, the other seven by 1 ULP; every alpha is unchanged. The importer
rejects before reaching its existing strict service/tangent checks.

`trace_allocation.cc` links the unchanged fresh static library. Its only
private-access experiment changes Norm credit to the high boundary in a
**disposable, inconsistent diagnostic Runtime**, calls the real allocator,
and obtains every original rate bit-for-bit. It does not import or accept that
edited state. `allocation-paths.json` records current packing, original signed
LP and forced-context diagnostic separately. A second diagnostic adds all
one-sided inward input boundary rows, including the GEMM rows. For this fixture
20-row signed LP still gives the exact original rates. This does not prove
that redundant rows can never affect LP rounding in other cases.

## Strict feasibility of the originally saved allocation

`certify_saved_allocation.py` independently uses exact Python Fractions of the
captured binary64 work, capacity, PRIVATE time, current segment derivative and
saved rate. It does not use a numerical feasibility epsilon. All 13 physical
rows, all PRIVATE bounds, and every current finite-credit boundary tangent
are feasible. The COMPUTE_GATE usage is strictly below 1; its exact positive
margin is:

```
849068037310433 / 162259276829213363391578010288128
= 5.23278578521043751691591177727234990332571653898381669467454457844723947346210479736328125e-18
```

Every active input credit is strictly inside its buffer. Every output buffer is
at zero with identically zero current output derivative. Thus no outward
finite-credit tangent is hidden by the 5-ULP mismatch. This certificate proves
feasibility for this concrete saved allocation; feasibility alone does not
prove max-min fairness and is not proposed as sufficient import acceptance.

## Minimal repair choices (design only)

Do not enlarge `same_or_one_ulp`, replace the strict tangent check with a
relative tolerance, discard saved rates, or normalize only the exported copy.
The latter changes continuation relative to the live source. The current
original snapshot must remain a red regression until an implemented repair
passes the original restore and exact continuation checks.

A. **Record allocation context for new snapshots.** Persist the ordered
per-execution/buffer boundary context used by the last actual allocation, not
just boundary classification at export time. On import, use it only to restore
missing rows that are proved redundant under rate>=0: RHS exactly zero, every
coefficient <=0, at least one nonzero coefficient. Keep all current
nonredundant physical, PRIVATE and credit rows. Reject unknown handles,
segments, buffers, orientations, inconsistent duplicate rows or a missing
context row with any positive coefficient. Never write the claimed context
into actual credit. Reconstruct the same stable ordering/solver path, compare
rate/alpha with the unchanged one-ULP policy, then restore published values and
run all existing strict certificates. Split owners here are not compressed;
other owner-symmetry cases require context-sensitive compression order to be
specified and tested too. A row-only context is not yet a general proof for
all historical allocation/compression transitions.

This repairs future snapshots if original allocation context is actually
recorded, with bounded metadata proportional to the active set. **Old v3
snapshots lack it**. A alone must continue to reject old mismatching snapshots
explicitly; silently inventing authoritative old context would be wrong. It
can optionally combine with the narrow, separately identified B fallback.

B. **One fixed redundant-row reconstruction attempt after canonical mismatch.**
Construct exactly one deterministic row set from public Plan/current-segment
facts, not from saved rates or a search over phases/subsets. An especially
narrow set sufficient for this real fixture is: for each current execution
and buffer, if the epoch's exact initial credit equals capacity and every
current active derivative is <=0, add its missing high-boundary row; if initial
credit is exactly zero and every derivative is >=0, analogously add its missing
low-boundary row. Skip an all-zero row. Existing rows are not duplicated. Use
stable handle/buffer/orientation order from the existing allocator. These rows
are proved redundant because their coefficient vector is componentwise <=0
and progress rates are componentwise >=0. They alter neither the mathematical
feasible set, owner-Q/alpha definition, lexicographic fairness nor physical or
PRIVATE limits. This concrete fixture adds exactly the four old Norm rows and
reproduces the original LP/rates. Adding every current one-sided inward row
also happens to reproduce this fixture, but the initial-boundary subset has a
smaller stated scope and row count.

The fallback must be a scratch allocation of the **actual current world** with
these extra rows, not a credit mutation or an import with its marker removed.
It must not replace or omit any current signed row, active owner, mirror,
activity, physical domain, PRIVATE bound, zero-rate exclusion or owner-Q
recomputation. Apply the original one-ULP rate/alpha comparison to its result;
if it does not match, reject. If it matches, apply all existing strict current
13-domain/PRIVATE/finite-credit certificates before restoring exact saved
values. A nonredundant missing historical row or a different numerical case is
outside this fallback and can still fail explicitly. There is no guarantee of
repairing every old snapshot.

Both designs need an explicit import-only bound. Preserve the active-variable
bound 96 and enforce/document the requested <=32 fair-owner envelope in the
fallback rather than treating the current 96-variable check as proof of owner
count. Source currently checks active variables <=96, not a separate unique
owner count. There are at most 2 nonzero one-sided extra rows per active
execution; at most 96 active executions implies <=192 possible extra rows.
A smaller fixed fallback limit (e.g.64 extra rows) can fail closed above its
scope and comfortably covers this 4-row case. Do not increase general solver
pivot/description limits to make a snapshot pass. Canonical and fallback LP
calls need a stated total pivot/work bound (e.g. original global256 shared or
an explicit bounded import-only allowance), not an unbounded retry loop.
Performance cost is one extra bounded allocator attempt only on mismatch;
new-context snapshots can use a single reconstruction. CPU cost is **not yet
measured**, and no budget compliance is claimed.

For this particular failure B is simpler than a new wire/schema/context
contract. It provides an existing-LP numerical fairness witness while the
strict exact current tangent check preserves safety. It still requires a
focused implementation/review before adoption; this report contains no fix.

## Necessary tests before adoption

1. The original frozen snapshot imports/forks with exact full snapshot equality
   and source-vs-restored continuation, including the original public events,
   cancellation, capacity changes and final served/busy/counter values.
2. Original 2/4/6/8 count controls, mixed owners, same-owner split executions,
   symmetry/mirrors, all-current nonnegative vs signed/ratio paths, zero PRIVATE
   service and zero-rate exclusions; verify unchanged owner-Q and alphas.
3. Both low/high inward rows, strict interior vs exact boundary, mixed-sign
   credit rows. Deliberately outward saved tangents must still fail with no
   epsilon, even if rates are close to an alternate LP solution.
4. Tampered rates/alpha, invented context, unknown buffer/handle/segment,
   duplicated rows, and old snapshots with/without published markers must
   preserve atomic rejection. Add fixtures that fail the fallback match.
5. Work/owner/row/pivot-limit controls and stable source/fork equality at large
   clocks and sub-ns progress. Run existing snapshot/credit/clock suites and
   affected public/full-API exact comparisons.
6. Measure import/fork CPU/allocation overhead separately after functional
   correctness; do not infer hot-update or five-consumer CPU acceptance from
   this diagnostic.
