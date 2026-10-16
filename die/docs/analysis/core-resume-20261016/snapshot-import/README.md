# Snapshot import redundant-row witness — scratch candidate

This is an independent, bounded **partial repair candidate**, not production
adoption. Production Execution.cc/header and the formal five-repository SDK
were not changed. Parent integration/review and a final fresh build remain
necessary. No CPU budget or hardware accuracy result was measured here.

The preserved original failure, strict rational feasibility certificate and
A/B design review are in `baseline-audit-report.md`; original raw identities
and all diagnostic outputs are in the corresponding `evidence/.../snapshot-import/`
directory. Original failed import/fork evidence retains its original status.

## Candidate and mathematical boundary

`import-redundant-row-witness.patch` applies only to baseline Execution.cc SHA
`3c9d58a0ed8c6b64450b86cac1059b3a1dba7b604caf67b31f00034c673e3043`
and header SHA
`e9c3228f1afc4e8ae6daf5069636b9e7cc987179b8017e256f6a30ad772000ea`.
The actual two modified scratch files are also retained under `candidate/`.
The patch was checked with `git apply --check` and was not applied to production.

Ordinary `allocate()` still uses its original row set, compression, solver
selection, arithmetic and 256-pivot limit. A new private overload takes an
import-only witness flag and a shared pivot counter. It adds no data members,
changes no public signature or C ABI, and retains the old private `allocate()`
symbol. The experimental shared libraries export identical `die_coarse_*`
symbol sets; see `candidate/abi-symbols.json.gz` in evidence. This is an
experimental reuse of unchanged objects compiled from the original header,
not certification of every C++ ABI consumer or a replacement for a fresh build.

Import first reconstructs the ordinary allocation. Only if the complete saved
rate/alpha vector fails the original same-or-one-ULP test does it make **one**
extra reconstruction. In stable handle/buffer/low-then-high order, it adds a
missing row only when:

- current epoch initial credit is exactly zero and every current active slope
  for that buffer is >=0, for a low-boundary row; or
- current epoch initial credit equals its capacity exactly and every current
  active slope is <=0, for a high-boundary row.

All-zero rows are omitted. Existing rows are not duplicated; current physical,
PRIVATE, signed credit, owner-Q, alpha, zero exclusion and mirror constraints
are retained. These added rows have exactly zero RHS and every coefficient
<=0; rates are >=0, so each row is a tautology. The mathematical feasible set
and lexicographic max-min objective are therefore unchanged. Epoch initial
credit supplies a fixed row-selection rule, not a claim about actual past
state. No actual credit or progress is changed to request an allocator path.
There is no phase, subset, context, period or activity replay search.

The fallback is limited to <=32 distinct fair owner keys, <=96 active
variables, and <=64 extra rows. Canonical and fallback allocations share the
existing total256 simplex pivot budget. Successful import restores generation
and allocation counters exactly as before; allocate changes no event count,
and settled event count is checked before reconstruction. All rates/alphas must
still match the chosen witness reconstruction within the original one ULP,
then the original physical, PRIVATE and strict finite-credit tangent checks
run before saved values are restored. A failed candidate lives only inside
import and cannot alter a parent Runtime.

This intentionally fixes only a subset of old snapshots. A different LP
rounding disagreement with no eligible fixed row, a nonredundant missing row,
a compression transition not reproduced by this row set, an exceeded bound,
or a nonmatching alternate result still rejects. No promise of general old-v3
snapshot recovery is made. Snapshots without a published-allocation marker
retain their original canonical-only behavior.

## Actual verification

The unchanged fresh baseline objects were copied from this session's
`work/build-core-release`; only scratch Execution.cc was newly compiled,
single-process, with the public Release flags. It was archived/relinked with
those unchanged objects and the actual fresh C API object. Full identities and
actual commands are in `evidence/.../snapshot-import/experiment-identity.json`.
The only compiler warning is the existing aggregate Ref::mirrors missing
initializer warning. This mixed-object experiment is explicitly **not a final
fresh all-source build**.

`probe.py` uses both actual baseline and candidate C ABI libraries:

- Real captured engineering hardware/two-task/paired-owner counts2/4/6/8:
  candidate ordinary source snapshots equal the original-library source;
  import and fork snapshots preserve all fields exactly. The count8 input
  snapshot equals the originally archived failed snapshot exactly.
- All four counts continue through real events to completion both without
  future controls and with real cancellation plus HBM capacity reduction.
  Original-library source, candidate ordinary source, imported and forked
  worlds agree in full snapshots, including finish/busy/served/counters.
- Seven malicious edits of the original count8 fixture reject atomically with
  zero output handle: 1%rate increase, structural zero replacement, 1%alpha
  change, inactive-rate injection, inconsistent credit, capacity above hardware,
  and wrong fingerprint. The live parent is exactly unchanged.

`guards.cc` independently verifies that a merely feasible saved rate4ULP below
canonical still rejects when no initial-boundary row qualifies, and that the
32-owner and64-extra-row fallback bounds fail closed. Parent worlds remain
unchanged. It does not treat feasibility alone as sufficient fairness proof.

Five original tests were relinked from their actual fresh test objects against
the scratch archive and all passed: service_curve_runtime, clock_translation,
owner_symmetry, runtime_semantics, runtime_counters. The unchanged service-curve
test includes structural-zero injection and an outward one-ULP large finite
credit tangent; those remain rejected. Its extreme absolute-clock, snapshot,
cancel, repeat, and terminal-credit assertions remain present and executed.
These five executable tests are not relabeled as a complete51-item rerun.

## Portable C ABI replay

After constructing an independent library from the candidate patch, run from
repository root (supply actual fresh-library paths):

```sh
python3 -B die/docs/analysis/core-resume-20261016/snapshot-import/probe.py \
  --repo-root "$PWD" \
  --library work/snapshot-lab/libcoarse_die_cabi.so \
  --baseline-library work/build-core-release/lib/libcoarse_die_cabi.so \
  --fixture-dir die/docs/analysis/core-resume-20261016/snapshot-import/fixtures \
  --output work/snapshot-replay
```

The baseline-audit tools and `run_focused.py` are retained byte-for-byte with
the actual historical work-directory invocation recorded in identities. The
portable probe above accepts explicit paths; final integration should use a
new clean build and repository CTest registration rather than reuse these
objects. No ELF, object, cache, SDK replacement or secret is included in the
committable artifact directories.

## Supplemental synthetic same-owner split fixture

A subsequent immutable fixture from the independent tiny-credit audit exposed
the same original import refusal at J2/J8 under credit units1,2^-10,2^-60.
The driver was copied byte-for-byte and linked **only against this B archive**;
no tiny-credit change was included. In these cases the load's initial-zero
input and compute's initial-zero output supply two inward redundant rows.
The existing fixed rule suffices without widening B: all six original
snapshot errors become exact roundtrips and exact source/import continuation.
All48 driver cases finish observation.

The unchanged independent exact-Fraction oracle still reports **32/48 passed,
16 failed**. Original source rates differ by a few ULP from its exact1/J ledger
in the split cases, and the tiny-credit unit problem is not fixed by B. Thus
48 completed observations are **not48 numeric/precision passes**. Full new raw,
original baseline raw/identity, immutable driver/oracle hashes, original failure
messages and separate status are retained in `supplemental-coupled-summary.json`
and the evidence subdirectories. The other agent's source and evidence were
read/copied only and were not modified.

The supplemental strict binary64 certificate is now separately recorded in
`supplemental-exact-certificate.json.gz`, using actual Plan PRIVATE times and
Fraction operations on actual published binary64 rates/work/capacities. All
six cases at all four stages satisfy all13 domains, PRIVATE, inventory bounds,
and tangent sign at both exact boundaries and the original Runtime boundary
masks; no tangent epsilon or new physical tolerance is used. The184 ordinary
observation objects shared with the unchanged baseline are exactly equal.

Per-metric absolute and relative discrepancies distinguish scope. At units1
and2^-10, J2 maximum rate error is2.220446049250313e-16 (relative4.44e-16),
maximum busy error9.5367431640625e-7ns (relative4.77e-16). J8 maximum rate error
is2.220446049250313e-16 (relative1.78e-15), maximum busy error
1.430511474609375e-5ns (relative1.79e-15). Finish error is0. These are original
LP rounding residuals under a strict exact-Fraction expectation, **not >10%
hardware-model errors**. The original exact-oracle status is still retained.
At2^-60, this B-only candidate retains a real existing credit-unit defect:
rate relative error1, busy relative error0.5/0.125, credit relative error up
to7 and some expected terminal times still absent. Those numeric failures
require the separate tiny-credit repair and are not classified as harmless
ULP residuals. The analysis tool's initial missing affine-slope and live-null
finish handling errors are preserved as tool failures, then explicitly handled
without changing the simulator or original oracle.
