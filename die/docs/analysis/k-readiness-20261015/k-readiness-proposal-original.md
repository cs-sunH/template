# OS GEMM K-readiness repair: minimal design study

Read-only proposal based on the E5 axis evidence, the 2026-10-14 granularity
audit, and the current `Mapping.cc` / `Execution.cc` implementation. No new
timing model or production code was changed for this note.

## Finding

The main gap is the unit used to schedule the work. `add_gemm_epoch()` builds
aggregate Load, Compute, and Store flows from exact OS work totals, then calls
`append_epoch()` with `rounds = ceil(output_tiles / cores)`; the plan separately
reports `rounds * Tk` consumption waves. For the one-core probes, K4096 and
K4097 both have 10 output rounds, but 640 and 650 K consumptions. The
`append_epoch()` interior therefore represents a whole K sequence per output
round as one progress coordinate. Its three activities share a continuous
progress window and its credit moves by `W-2` units over that window. That
preserves totals but removes each K-read completion boundary and spreads
final-output writes over the interior.

The critical code paths are:

- [`append_epoch()`](../../die/src/Mapping.cc#L803) divides aggregate flows by
  output-round count, then creates one `interior_repeat` activity set for
  `W-2` rounds. `repeat_count` is audit metadata; the runtime does not replay a
  per-round body.
- [`add_gemm_epoch()`](../../die/src/Mapping.cc#L878) checks exact OS aggregate
  bytes/ops and SRAM/LOCAL ledgers, then constructs one Load, one Compute, and
  one Store flow for all K work and passes output `rounds` to the macro plan.
  `GemmOSGeometry` already reports `R*Tk`, but the lifecycle does not consume
  that wave sequence.
- [`Runtime::settle()`](../../die/src/Execution.cc#L958) completes an activity
  only at progress 1 and advances at epoch boundaries. It has no K-read-ready
  token or final-K output-ready token.
- [`Runtime::allocate()`](../../die/src/Execution.cc#L1030) correctly applies
  the current owner-Q/alpha LP-A, all 13 resource-capacity rows, each
  activity's private `L` limit (`private_time_s`), and signed credit-boundary
  rows to whatever activities are eligible. This service allocator is not the
  defect and should stay the allocator.
- [`Runtime::integrate()`](../../die/src/Execution.cc#L1499) counts HBM busy
  when either HBM row has positive service, and shared busy when two owners
  serve the same row. It integrates the work vectors on each activity's one
  progress coordinate. HBM payload can conserve exactly while the interval
  during which HBM is occupied changes substantially.

The E5 reports support a topology error rather than a missing-work diagnosis.
On M80/N128, K4096 and K4097 select the same OS 16×64×64 mapping and both have
about 202.275% solo HBM-busy error; changing one K tail element does not remove
that gap. At K4097, changing to M65/N65 keeps the same tile but changes solo
HBM-busy error to 124.951%, while offset-first completion error rises to
10.782%. All grouped comparisons remain failed. The serialized work residuals
are at most about `3e-11` ops or `5e-13` bytes; they cannot explain 100–200%
busy-time gaps. K128 is excluded from this OS comparison because the compiler
reselects Weight Resident 1×64×64.

The runtime time error has a direct mechanism: the large Load activity bundles
HBM and its route/link work behind one scalar progress rate; the large Compute
activity bundles all PE, SRAM, and LOCAL work behind another; and the large
Store activity can progress during the interior even though each output is
not ready until its final K. The 13-domain LP can therefore keep HBM service
active for a different interval and expose a different set of active owners
to Q/alpha than the discrete K-wave schedule. The ledgers stay conserved,
while finish and HBM-union intervals move.

## Smallest plan representation

Keep the selected mapping, tile sizes, hardware rates, and all work formulas.
Replace the OS use of aggregate `append_epoch()` with an optional, versioned
`GemmOSReadiness` descriptor. Leave ordinary `Epoch` and its audit-only
`repeat_count` semantics unchanged for existing plans.

The descriptor is bounded by four M/N output-shape classes times three K
boundary classes. It needs only:

```text
GemmOSReadiness {
  row_patterns[<=2] = {m_extent, row_count, n_full_count, optional n_tail};
  facts keyed by (M_extent_class, N_extent_class, K_boundary_class);
  input_slots = 2; output_slots = 2;
  for each implied M/N class (at most 4):
    first_k:      {Load, Compute, Store facts};
    middle_k:     {Load, Compute, Store facts}, middle_count;
    final_k:      {Load, Compute, Store facts};
}
```

For `Tk == 1`, use one `first_final` fact instead of duplicating first and
final. For `Tk > 1`, the middle count is `max(Tk-2, 0)`. A tail changes the
final K extent to `K-(Tk-1)*tk`; an exact multiple uses `tk` for final K.
The per-consumption facts come from `GemmOSAccessCounter::count()` and
`gemm_os_single_core_resource_facts()`, rather than another hand-coded dtype
ledger. The existing helper returns the three actual resource activities and
documents that a nonfinal K has an empty Store. It intentionally rejects
multi-core inputs until per-core cohort semantics are supplied.

For class `(a,b)`, let `Q_ab` be its output-tile count and `F_ab,k` the
integer per-K activity ledger. Define `F_ab,Kseq` as `F_ab,first_final` when
`Tk == 1`; otherwise define it as
`F_ab,first + middle_count*F_ab,middle + F_ab,final`. For every integer field
in the OS access ledger, the conservation invariant is:

```text
Ledger = executions × Σ_(a,b) Q_ab × F_ab,Kseq
```

Use checked `u128`/the existing exact equivalent-work type for byte and op
sums, then compare them to the current Plan summary before accepting the
descriptor. The 13 resource-service entries continue to be derived by the
current `Flow`/resource builders and sealed with the current numeric types;
this proposal does not invent alternate rates or promise exact binary sums
for those doubles. For strict FP32 the simple inputs to this counter are
`read_bytes = 4*k_i*(m_i+n_j)`, `source_ops = 2*m_i*n_j*k_i`, and
`write_bytes = 4*m_i*n_j` only for final K. Other storage, PE, accumulation,
conversion, and cast costs stay with the existing access counter. In
particular, compute SRAM writes/accumulator updates occur on each K; the final
cast is charged to final-K Compute; output SRAM-read/HBM-write work exists
only in final Store.

Preserve row-major output order: a full-M row pattern consists of its N-full
run followed by its optional N-tail tile, repeated for the number of full M
rows; the optional M-tail row pattern is a final suffix. This is the compact
form of `WholeRowClasswordFact`, and avoids incorrectly sorting all equal
M/N classes into one run.

This is fixed-size metadata, not one Plan activity per output tile or per K
wave. It is a new lifecycle variant instead of overloading `Epoch.repeat_count`
or raising the 64-epoch/three-activities limits. Keep the existing `epochs`
representation for every non-OS path and for old serialized plans.

## Readiness and service rules

Execution needs a compact cursor and two small slot queues, not a K-sized
vector. A slot token contains only the output-class run, output ordinal, K
ordinal, and `first/final` flags. The event rules are:

1. Start a KRead only when an input slot is available. Its exact K token enters
   the data-ready queue only when that Read activity reaches terminal progress.
2. Start PE only for the next K ordinal of that output tile, after its own
   Read token is data-ready and the preceding K's PE activity has completed.
   A later K may be prefetched into the second input slot, but cannot become
   PE-eligible or execute out of order. PE completion releases that input
   slot.
3. On final-K PE start, reserve one output slot. On PE terminal, mark that
   output ready. Start output Store only from this ready queue; release the
   output slot when Store reaches terminal progress.
4. Nonfinal K has no output Store activity. Its vector/accumulator actions
   remain on Compute as the access counter specifies. A final cast is only on
   final-K Compute, and the output HBM Store is only after that Compute ends.

Use the same signed buffer deltas already understood by `allocate()`:
Load `(+1,0)`, nonfinal Compute `(-1,0)`, final Compute `(-1,+1)`, and Store
`(0,-1)`, with capacities `(2,2)`. Gate a Load start on one free input slot;
gate a final-Compute start on one free output slot. Only one activity per
stage is active for an execution, so admission reserves that slot until the
activity advances and no second start can claim it. Keep a lifecycle credit
base; when one activity terminals, absorb its completed signed delta into the
base and remove that activity's progress term. Current LP credit remains
`base + Σ(active_delta × active_progress)`, so the existing LP rows and
`advance_to()` credit-boundary events still constrain service within the
stage. Readiness tokens add the missing exact dependency on top of those
continuous limits.

Keep at most one active Load, one Compute, and one Store activity per
execution. They are the only activities submitted to the existing LP at that
moment; the token queues choose which bounded class fact feeds each role.
On every stage admission/completion, call the same `settle()`/`allocate()`
path. Do not replace it with `max(total_work/capacity)`, a private per-stage
bandwidth calculation, or equal-progress sharing. Each active role retains
its real 13-entry `work` vector, `canonical_q`, `private_time_s`, and current
credit limits. Preserve LP-A as implemented: owner `Q` is the sum of active
canonical weights, `alpha_i = q_i/Q`, physical rows constrain all 13 work
domains, and the private `L_i` bound remains `x_i <= q_i/private_time_s` in
canonical service. The readiness cursor only determines which stage ref is
eligible; it must not change Q, alpha, private limits, or the work counter.

Keep the existing time integration and busy definitions. This repair changes
when exact per-wave activities are eligible, not the resource capacities or
the union/shared HBM accounting. A failed cycle proof must use the bounded
event path or fail closed; it must not move a later K into the ready queue or
let a final output Store run early.

## Prefix / steady / suffix compression

For a run of identical middle-K facts, first execute the finite startup until
the joint runtime state returns to the same cycle boundary. The state key
must include all active owners and stages, the K/output class cursor modulo
the run, input/output slot counts, the LP allocation region/rates, resource
capacities, and pending external arrivals. If one complete cycle returns to
that same state with the same ordered stage completions, its time and ledgers
are reusable:

```text
T         = T_prefix + n*T_period + T_suffix
Busy_row  = B_prefix + n*B_period + B_suffix
Shared_HBM= S_prefix + n*S_period + S_suffix
Work[r]   = W_prefix[r] + n*W_period[r] + W_suffix[r]   (13 resources)
```

`B_period` and `S_period` are measured from the actual integrated busy
indicators over the half-open period, not derived from payload bytes. `T_period`
and `W_period` are produced by the current LP allocation and resource service;
only checked count multiplication is used to fast-forward. Split before a
50 ms arrival, capacity update, output-class boundary, final K, or any LP
region/active-set change. Those are exact prefix/suffix boundaries.

The state is joint across requests. If the stage pattern, LP region, or
credit/slot state does not return, there is no repeat formula to apply. In
that case retain the existing event loop up to its existing guard and report
that this plan did not admit a bounded cycle. Do not increase or reset the
1,000,000 advance-event guard, the existing 256-pivot LP budget, or any
per-call solver/event count. A skipped cycle must checked-add its logical
event and completion counts to the counters; it may not start a fresh budget.
This gives fixed work only for a proved stable run, not an unconditional
constant-time promise for arbitrary owner phases.

For the currently evidenced one-core workload, `GemmOSGeometry` already
provides at-most-12 M/N/K access classes and row-major class-run bounds. Plan
construction is therefore `O(1)` in `R*Tk`; cursor memory is `O(1)` and at
most two input plus two output slots are live. Runtime work is
`O(prefix + suffix + external boundaries + cycle witnesses)` for a certified
steady segment, independent of its repeat count. Without a witness it remains
`O(events)` and must not claim the fixed bound. Multi-core use requires a
separate bounded cohort-fact builder; do not pass the single-core facts to an
active-core aggregate or infer a slowest-core map from total work.

The event-level completion corrections in `settle()` should remain the source
of exact per-activity HBM/Vec conservation. The E5 probe found tiny nonzero
Decimal residuals in cumulative serialized long-double totals despite exact
planned work. Keep strict plan/runtime work equality separate from timing
acceptance; a separate exact shadow for completed integer bytes may be needed
if the Runtime API promises globally exact integer totals. That counter work
is much smaller in magnitude than, and must not be presented as the cause of,
the E5 busy-time failure.

## Independent E8/E16 validation

Validate E8 and E16 as separate compiled/runtime variants against the same
independent grouped owner-Q/alpha K-wave oracle; neither variant is the other's
oracle. Do not change the selected candidate or ranking to make a case fit.
For each variant:

- Check exact equality of integer byte/op counters with the current plan
  summary, including first/middle/final counts and one final output write per
  output tile. Independently reduce each descriptor's 13-resource vector
  through the same resource builders and compare the sealed plan values.
- Instrument a negative test proving PE for K=`i+1` cannot start before its
  KRead terminal, and Store cannot start before final-K PE terminal.
- Run M80/N128/K4096 (no tail), M80/N128/K4097 (tail), and M65/N65/K4097
  (M/N edges), each solo, synchronous two-owner, and 50 ms offset. Recheck the
  actual selected tiles for every run. The 3% solo / 5% concurrent completion
  and 5% HBM/shared-HBM limits remain unchanged; all comparisons must pass
  independently for E8 and E16. Keep raw failed reports if either fails.
- Run a large-repeat workload to check that descriptor size and expanded
  event work do not scale with `R*Tk` when a cycle is certified. Also force a
  50 ms arrival or LP-boundary crossing inside the candidate period and
  require an exact split or fail-closed result.

K128 is not part of this OS readiness proof while the compiler selects Weight
Resident. Its raw counters remain useful for the mapping-selection axis, but
they cannot validate OS K readiness. No expected monotonic improvement from
E8 to E16 should be assumed: both must meet the same independent oracle and
invariants separately.

## Evidence basis and limit

See [`os-kernel-e5-granularity-audit-20261014.md`](../../die/docs/analysis/os-kernel-e5-granularity-audit-20261014.md)
and [`E5 axis evidence`](../../die/docs/evidence/gemm-axis-20261014/README.md).
The present proposal identifies a concrete readiness and final-store seam; it
does not claim the prefix/period/suffix map is already proven for asynchronous
Q/alpha owners or that the proposed implementation would pass 3%/5%.
