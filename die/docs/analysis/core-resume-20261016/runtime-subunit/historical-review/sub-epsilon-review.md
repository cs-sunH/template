# Review of the unrun sub-epsilon proposal

This is source review only. The archived candidate was not adopted or executed.
The actual current tiny-credit regression red is separately recorded under
`current-baseline/`; it is not a test of this candidate.

## What the historical evidence does establish

The archived boundary trace contains a real interval1.657188659916e-16ns. Dropping
it because it is below1e-15ns led to a finite-buffer violation after0.51288334ns
of extra service. The interval is larger than the approximate long-double ULP
near a fractional nanosecond on the recorded x87 platform, so it is plausible
that this particular boundary can be represented. This does not establish that
every legal positive event interval can be represented or that the final
candidate reaches a correct terminal.

The proposal changes curved min-positive filtering to candidate>0 and changes
the internal rejection from next_delta<=1e-15ns to next_delta<=0. Its exact-credit
helper also changes boundary classification from binary64 diagnostic credits to
recomputed long-double credits with a gamma32 clamp. These are several coupled
numerical changes; the previous 7/8 result does not validate them together.

## Representation and bounded-event risks

1. `integrate` adds delta to `exact_ns_`, a long-double fraction in[0,1). For a
   sufficiently small positive delta, the rounded fraction can remain unchanged.
   Independently, rate*delta/1e9 may be positive but too small to change an
   activity's long-double progress. The proposal rejects neither situation. It
   can repeatedly choose the same boundary until the million-event failure,
   or advance service/busy counters without represented clock/progress movement.
   The existing progress-boundary clamp can close an exact terminal event, but
   a credit-boundary event generally does not receive that clamp.
2. Very small headroom is not always spurious arithmetic residue. Conservative
   downward-rounded tangent rates can create real residual headroom. A fixed
   positive-time floor discards that physics; merely removing the floor can
   instead create repeated tiny events. A bounded local boundary treatment needs
   explicit physical feasibility and progress guarantees, not a smaller magic
   epsilon or an unlimited retry loop.
3. `next_event` still publicly ceils to integer ns with its existing epsilon and
   minimum1ns return increment. Multiple true internal events can occur under one
   public call. That is allowed, but fixed external inputs and amplified work
   still must not produce tile/K/repeat-linear internal events.
4. `ceil_exact_time`, ready comparisons and UINT64-tail checks retain their old
   epsilon behavior. Enabling previously filtered sub-epsilon internal intervals
   requires checking their interaction with fractional time near0/1 and the last
   uint64 tick, not only ordinary-time fixtures. Mixed curved/legacy executions
   also share those internal integration intervals despite per-epoch filtering.

## Credit-unit scaling and uncertainty risks

5. The true event ratio credit/credit_rate is invariant when every quantity in
   one credit channel is multiplied by a positive unit scale. The current red's
   scales1,2^-10,2^-60 prove that small credit is not necessarily small remaining
   physical work. Test filling/emptying, two-channel coupled rows and both
   channels independently; use normal powers of two first to separate arithmetic
   rounding from changed physical inputs.
6. The candidate's gamma32 expression scales with contribution magnitude and
   capacity in ordinary arithmetic, but its cap and original illegal-credit
   threshold still contain max(1,capacity). For capacities far below1, that
   original threshold can exceed the entire physical capacity by many orders.
   In particular the candidate clamps any accepted negative credit or accepted
   above-capacity credit, even when it exceeds the tighter gamma uncertainty.
   Preserving the historical rejection threshold alone does not certify physical
   work conservation under unit rescaling. Any accepted clamp must be justified
   by arithmetic uncertainty, while never enlarging the original rejection scope
   to hide an actual finite-buffer crossing.
7. Cancellation between large cumulative contributions can make the uncertainty
   bound as large as the whole small buffer. Underflow/subnormal interpolation,
   derivative overflow or loss and changing-segment endpoints are outside the
   archive's normal-arithmetic gamma32 argument. Source comments explicitly say
   this is not a proof for all legal inputs. Do not promote it to one.

## Suggested next checks before selecting any numeric fix

- Keep the exact current tiny-credit red and historical mixed-case failures as
  separate source-bound fixtures. Establish a current-Core synchronous reproducer
  without importing/relabeling old hardware snapshots.
- For an internal event, check whether exact-origin/fraction, progress and credit
  boundary state can represent the intended advancement. A nonrepresentable
  event needs a bounded mathematically justified boundary operation or an honest
  numerical-unavailable rejection. Do not round time upward across a buffer
  boundary, round progress to terminal without paying work, or discard a real
  event because it is inconveniently small.
- Validate all boundary decisions from one consistent numerical credit view,
  including singleton/packing/ratio/LP, event selection, integration and imported
  saved-rate tangent checks. Retain original owner fairness, physical ledgers,
  PRIVATE, cancellation and exact snapshot continuation.
- Exercise scale-invariant normal powers of two, then explicitly scoped extreme
  units; near-segment boundaries, one-ULP headroom, fixed absolute time shifts,
  different long-double precision formats and UINT64 endpoints. Report unsupported
  representability honestly rather than using old BF16 successes as universal
  evidence.
- Once correctness holds, measure event counts under work amplification before
  normal performance budgets. No periodic proof, big-integer production solver
  or tile/K/wave replay is recommended.
