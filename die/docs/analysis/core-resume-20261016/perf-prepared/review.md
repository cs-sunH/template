# Current Core performance audit and unadopted scratch candidates

Status: source inspection and preparation only. No candidate was compiled, run,
adopted, benchmarked, published, or committed. Production files were not edited.
The baseline is HEAD `9fc4a6a88c03f8e4ca6ce318968f152d18732a9a`; exact source,
candidate and independent patch hashes are in `manifest.json`.

## Current evidence, with its actual paths

Read `die/docs/evidence/core-resume-20261016/public-checkpoint-summary.json`,
`budget-example.json` and `budget-tx8.json`. Current final migration is fresh
51/51 passed and 192 resource comparisons passed; both budgets failed.
No current numerical or speed acceptance is inferred from old runs.

| Metric | example | user-authorized TX8 engineering point |
|---|---:|---:|
| 128-cubed cold compile P95 | 2.587118 ms | 2.568263 ms |
| OS J1 update median / P95 | 6.434 / 9.085 us | 13.499 / 18.954 us |
| OS J8 update median / P95 | 18.366 / 68.344 us | 28.791 / 104.229 us |
| OS J32 update median / P95 | 28.681 / 33.253 us | 41.873 / 277.039 us |

Every one of the 30 example OS samples, at J1/J8/J32, used packing, three
compressed activities, 14 constraint rows and zero pivots. Every TX8 OS sample
used ratio, three compressed activities, 15 rows and zero pivots. The older
signed-LP candidate diagnosis does not describe these particular update cases.

Example J8's two largest raw samples were 68.344 and 86.250 us; the next largest
was 41.588 us. Example J32 had a 56.449-us maximum despite its passing P95.
TX8 J8's two largest were 104.229 and 115.798 us, with 54.374 us next. TX8 J32's
two largest were 277.039 and 382.125 us, with 91.116 us next. All remain in raw;
neither outlier removal nor choosing a faster rerun is proposed. Both identity
records report zero cgroup throttle-count delta; the raw data alone does not
establish a unique cause for the tail spikes.

Historical diagnostic scope: `compiler-ranking-20261016/phase-final.json` had
28 candidates for 128-cubed, internal enumeration 5038 C++ new calls / 306653
requested bytes, candidate Runtime 7293 / 631716, and complete compile 14447 /
1111097. Thread-CPU medians were approximately 0.379 / 0.710 / 1.340 ms for
those phases. This was old hardware, with instrumentation overhead; these
figures identify remaining mechanisms, not a current performance certificate.

`j32-update-20261016/checkpoint-summary.json` and README place old J8/J32
collect medians at 10.972 / 17.021 us and symmetry at 1.092 / 7.616 us. The
rejected two-traversal fusion did not reduce new counts or supported CPU time.
It is not included in any candidate here.

## Independent candidate 01: bounded symmetry scratch and stable Ref compaction

Files: `candidate/Execution.cc`, `01-stable-ref-compaction.patch`.

Production `Execution.cc:1236-1320` currently allocates blocks, representative
indices, merged bits and an entire second Ref vector on every multi-owner
allocation. The second vector is allocated even when no owners are merged.

The candidate changes only these storage choices:

- The existing preceding `refs.size() <= 96` rejection bounds block count,
  representative count and merged indices. Three call-local arrays of 96
  entries replace their vectors. Only active prefixes are read.
- Original collection, owner-index sorting, split-owner detection, boundary
  masks, exact block comparisons, representative selection and mirror insertion
  remain in the same order with the same expressions.
- After every block-index read is complete, one forward stable compaction moves
  each surviving Ref into its retained position and resizes the original vector.
  Self-move assignment is avoided. The surviving order, multiplicity and each
  mirror sequence remain unchanged; original comparison/tie arithmetic is intact.

The rate/alpha reset and collection traversals remain separate. The candidate
adds no persistent state, cached numeric decision, snapshot field or model rule.
It avoids at least the blocks/merged/reduced allocations and representative
vector growth. The fixed arrays use about 3 KiB of additional call-local stack
storage on the observed ABI; this is a bounded scratch tradeoff, not zero cost.

## Independent candidate 02: internal OS trial diagnostic storage

Files: `candidate/Mapping.cc`, `02-gemm-trial-audit.patch`.

Only the existing compile-only GEMM `ranking_only` flag is propagated through
`add_gemm_epoch` to `append_os_service_curve`. Its default is false. Existing
other callers, public enumeration and the full winner builder remain false.

`Mapping.cc:998-1013` currently constructs 21 audit fields for every internal OS
candidate. Internal downstream uses were traced:

- `finalize_summary:1319-1331` reads the curve model and computes the actual
  `service_description_count` from Activity credit curves.
- `apply_region_ready_overheads:3303-3311` reads epoch_begin and description_count,
  updates description_count/budget and adds its ready-gate description.
- `Compiler.cc:602-632` reads task/hardware identity, prediction status and rank
  summary keys, not the omitted OS diagnostic fields.
- `estimate_candidate_latency_ns` still invokes Plan structure validation,
  Runtime construction, complete submit validation and all actual event/service
  integration. `Plan.cc` structure checks actual epochs and curves; its only
  summary requirement is an object. Runtime does not use OS audit JSON for timing.

The compact audit retains model, epoch_begin, epoch_count, description_count
and description_budget. Actual credit curves, all Flow/ledger calculations,
checked endpoint representability and primitive-description checks are untouched.
The two checked conversions within the old audit initializer, first-output HBM
bytes via as_u64 and first-compute time via as_double, are executed explicitly
in their original relative order even when their JSON values are not retained.
All other checked arithmetic, EqWork, status, resource and capacity paths remain.
The existing final full builder reconstructs every original field, and its full
initializer expressions remain unchanged. Thus final JSON string-capacity/cache
accounting must still be compared against baseline; no equality is claimed yet.

This is the candidate requiring the closest review. Its safety depends on the
private-trial consumer trace above and the full-materialization tests. Do not
extend this change to public enumeration or other operator builders by analogy.

## Independent candidate 03: packing waterfill temporary reuse

Files: `candidate/PackingAllocation.hh`, `03-packing-scratch-reuse.patch`.

`PackingAllocation.hh:83-86` constructs slopes and increments inside every
waterfill iteration. The candidate declares those two vectors outside the loop
and assigns the same zero/infinity values at the start of each iteration.
Capacity is retained only within that call; no allocation is introduced when
the active count is zero before entering the loop. Every row/activity traversal,
floating operation, comparison, saturation, fairness weight, fallback, final
certification and conservative rounding remains unchanged. `saturated` is left
untouched to keep this independent candidate small.

This affects the current example packing path; TX8 OS uses ratio and should not
be assumed to benefit. It also applies to packing calls inside cold candidate
estimation without reducing or skipping candidates.

## Additional suggestions, not implemented

- `Execution.cc:1430-1432` deep-copies each RatioRow coefficient vector into an
  otherwise identical PackingRow before a read-only packing call. A private
  templated implementation with the existing public internal wrapper retained,
  or a read-only row view, can avoid that copy while keeping every original
  nonnegative/finite/final feasibility check. Interface scope was deliberately
  excluded from the prepared candidates. Example has 14 such rows per OS call.
- `Mapping.cc:1275-1276` and `1893-1894` assign template/dataflow strings twice
  for each GEMM candidate. Internal-only final-value emission can reduce those
  string allocations. The full path's string-capacity behavior must remain
  unchanged. Not included to avoid stacking a fourth code change.
- Do not use shared Plan pointer equality to skip definition comparisons unless
  immutability/provenance across submissions is actually enforced; Runtime owns
  copied LiveActivity definitions, and a mutable external alias could otherwise
  invalidate that assumption.
- Do not skip per-candidate validation, mutate numerical tolerances, increase
  ratio eligibility, replace original fairness or revive rejected traversal fusion.

## Validation to be done by the parent

Each patch is independent against the recorded baseline. `git apply --check`
passed for all three; this only checks patch applicability. No C++ syntax check,
compilation, test run or performance sample was executed here.

1. Build baseline and each candidate independently from fresh directories and
   record complete production/header/test and ELF identities.
2. Run `compiler_summary_test`: preserve all 64 tasks / 1092 current-hardware
   candidate count/order, materialized full wire/hash and errors, plus mutated
   rank-key atomic rejection. Compare baseline/public enumeration with
   `compiler-ranking-20261016/capture_candidates.cc` across separate binaries.
3. Use `hotpath-allocation-20261016/candidate_paths.cc` for complete per-candidate
   finish/paths/pivots/tangent trajectories and complete 11-object C ABI captures.
   Include complete cache/accounted bytes and `capture_retained.cc`; the earlier
   10-byte string-capacity mismatch is a known reason hashes alone are insufficient.
4. Run focused symmetry/owner split, endpoint, packing/ratio/LP, readiness,
   cancel/import/fork and source-snapshot continuation checks, then current public
   suite. Exercise no-merge, partial merge, split owner and maximum96 refs; compare
   all survivor/mirror rates/alpha and diagnostics against baseline.
5. Only after equality, collect normal30 budgets for both actual configurations
   serially without concurrent build/CPU work, keeping every sample and failed
   attempt. Diagnose current packing/ratio paths; do not extrapolate old LP costs.

The prepared changes do not claim a tail-latency cure or acceptance.
