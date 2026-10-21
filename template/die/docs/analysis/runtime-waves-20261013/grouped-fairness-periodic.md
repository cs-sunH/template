# Grouped fairness and exact periodic detector scratch

This scratch adds the design's grouped `Q/alpha` rule to a tiny exact-rational
K-full stream model and compares it with a bounded exact-state cycle detector.
It changes no production source or design document. The executable output is
in [`grouped-fairness-periodic.json`](grouped-fairness-periodic.json), produced
by [`grouped-fairness-periodic.py`](grouped-fairness-periodic.py).

## Model and allocator

There are two identical owners, each with at most one active load `L` and one
active compute `C`. The model holds two input slots per owner: a load reserves
a slot at start, compute consumes only a completed full-K input, and a new
load starts when a slot becomes free. This is the `K_FULL_NONFINAL` class, so
it has no output store, K-final transition, or retirement. The model therefore
has at most four live activities, below the production limit of three per
owner. The fixed stage work is load payload 2 on HBM0 and compute-gate work 10;
the two compute-Q profiles are `Q_C=1` and `Q_C=LOCAL_payload=8` (the latter
also charges 8 LOCAL work against a deliberately nonbinding capacity 1000).
The capacities are HBM0=1, COMPUTE_GATE=1, LOCAL=1000.

For each allocation epoch, `Q_L=2` and `Q_C` are summed over the owner's
currently feasible activities. Thus `alpha_a=Q_a/sum_owner(Q)`. The solver
raises `y_a/alpha_a=Q_a*v_a/alpha_a=sum_owner(Q)*v_a` together, freezes
positive-rate users of the first saturated resource, then continues the
remaining tier with the same `alpha` values. It recomputes the owner sum only
after a discrete membership event. Load and compute masks are integrated from
the actual positive-rate activities; shared HBM time is the interval union
where both owners' load rates are positive.

The fill solver is checked at every event against a second exact-rational
small LP oracle. The oracle builds the tier LP directly: with frozen rates
subtracted, each resource gives `slope[r] * z <= residual[r]`; the primal
optimum is the minimum exact `residual/slope`. A binding resource supplies an
exact dual multiplier `1/slope`, and the code verifies matching primal/dual
objectives and every resource inequality. For a four-active overlap the
exported certificates are:

| Compute Q | First LP tier | Second LP tier | Rates in the overlap |
|---|---|---|---|
| 1 | COMPUTE_GATE binds at `z=3/20`; freeze both C activities | HBM0 binds at `z=3/4`; freeze both L activities | `v_C=1/20`, then `v_L=1/4` |
| 8 | COMPUTE_GATE binds at `z=1/2`; freeze both C activities | HBM0 binds at `z=5/2`; freeze both L activities | `v_C=1/20`, then `v_L=1/4` |

For `Q_C=8`, LOCAL usage is `16/20=4/5` work per second in the overlap, well
below its capacity. The owner denominator remains 3 or 10 across the two LP
tiers. The identical progress rates do not imply identical fairness floors:
the canonical unit changes the floor, while the frozen owner's weight is not
renormalized during the HBM tier.

## Exact recurrence and offset comparison

The cycle detector keys on the capacity generation and exact capacities/Q
policy, owner identity, full-K category, progress of every current `L`/`C`,
ready-slot counts, active rate map and exact tier basis. It omits absolute
wave counts and accumulated counters, which are instead stored as per-cycle
deltas. A repeat is accepted only when the entire key compares equal as exact
`Fraction` values; no tolerance or approximate phase equality is used. A
repeat records exact period, busy/work integrals, and owner completion deltas,
then skips `floor(remaining_horizon / period)` periods. It checks the
`10^12` wave cap before applying the skip and simulates the remaining prefix
event by event.

The arrival of B is an external prefix boundary: the detector clears its
`seen` map there, retains the exact prefix counters and current A state, admits
B, recomputes the rates, and begins cycle discovery from the new joint state.
For production cancellation or capacity/profile changes, the same rule must
be applied after advancing to the exact event time and processing same-tick
events in design order: preserve the already-integrated prefix, apply the
change, increment the generation, discard prior repeat candidates, and resume
with the new membership/capacity. No skip may cross a known external event.
Those cancellation/capacity cases are algorithm guidance only; this scratch
does not execute them.

The handpicked set has 11 rational arrival offsets per Q, including
`1/7`, `1/13`, `3/2`, `7/3`, `101/10`, and `1000/7`. A further deterministic
stress set uses 120 seeded rational offsets per Q with denominators from
97 through 65521. Across all 131 offsets per Q (262 Q-offset combinations):

- All runs find an exact joint repeat with period 20 seconds. The largest
  number of post-arrival probe intervals before a repeat is 9, well below the
  256-probe fail-closed limit.
- All accelerated runs finish the 500-second post-arrival horizon, so the
  unadvanced horizon gap is 0. There are no fail-closed cases in these samples.
- The event-by-event and accelerated exact `Fraction` counters, owner wave
  counts, and final state keys match for every handpicked case; the seeded
  stress set also reports exact final matches. Exact LP checks run at every
  event interval in both paths.
- A separate branch check temporarily lowers the probe limit to 1. It returns
  `fail_closed_no_exact_repeat_within_1` after exactly one probe, with the
  exact event-by-event prefix and a reported remaining horizon of `3474/7`;
  the 256-probe production limit is unchanged in the generated run.
- Phase snapshots are not constant before recurrence: among handpicked
  nonzero offsets, the number of distinct recorded within-wave phase
  differences is 2–4. For example, at offset `1/7` with `Q_C=1`, the exact
  repeat state is `A:(C=0,L=0,ready=0)` and
  `B:(C=34/35,L=inactive,ready=1)`, with period 20; the phase trace changes
  before that repeat. This demonstrates why a fixed phase bucket or one
  resource's counter alone is not a safe cycle key.
- In the repeating 20-second segment at offset `1/7`, `Q_C=1`, the exact
  positive-rate integrals are compute busy/work 20, HBM busy/work 4, and
  shared HBM busy `68/21`. The `Q_C=8` case gives shared HBM `118/35` while
  retaining compute busy/work 20 and HBM busy/work 4. These are mask-derived
  integrals, not extrapolated from average rates.

These are finite fixture results, not a proof for every rational offset or
the full production LP. The sample shows that exact repeat detection succeeds
quickly in this fixed K-full, fixed-capacity, two-owner domain. A general
implementation still must preserve all service-mask/counter categories and
fail closed after 256 exact probes if no joint state repeats.

## Support and limits

Supported by this scratch: two identical owners; rational arrival offset;
fixed rational capacities; at most two live activities per owner; one
non-final K-full transition; exact `Q/alpha` allocation over HBM0,
COMPUTE_GATE, and nonbinding LOCAL; event-by-event vs cycle-skipped
integration; positive-rate busy masks; exact completion and work counters.

Not covered: any K-final output generation, output-slot retirement, store
activity, cancellation, capacity change, heterogeneous owners, signed credit
constraints, private-resource bounds, the other seven full/tail transition
classes, production numerical tolerances, or a proof that every rational
offset recurs within 256 probes. A 256-probe exhaustion must report the exact
prefix and unadvanced gap as unsupported/fail-closed; it must never invent a
repeat by rounding phase.

Run and verify with:

```sh
python3 -m py_compile work/audit-busy/grouped-fairness-periodic.py
python3 work/audit-busy/grouped-fairness-periodic.py > work/audit-busy/grouped-fairness-periodic.json
```

The generated JSON labels the result `exploratory_not_certificate`.
