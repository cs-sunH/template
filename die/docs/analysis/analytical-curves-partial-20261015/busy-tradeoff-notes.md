# Conditional HBM-busy lower bound for continuous K-credit service

This is a scoped mathematical result about the existing **positive-service** busy metric. It does not change that metric, infer a universal impossibility for piecewise curves, or validate the v3 implementation. The optional numeric data below come from a retained **v1 historical raw** file and are arithmetic illustration only.

## Model and assumptions

Consider one output-stationary execution during a compute interval of length `T`. Throughout that interval:

- Compute runs without a stall and consumes input K-credit at a constant rate `c > 0`; hence it consumes `N = c T` units.
- The unified input-credit inventory is `x(t) in [0,B]`, where `B = 2 * active` (the stated two-slot capacity). `x` may begin full, so the bound permits the maximum possible prefetch before compute starts.
- Let `u(t) >= 0` be the K-credit supply rate from the corresponding L/load service. Outside intervals in which this L service is active, it supplies no input credit; the credit law is `x' = u - c` whenever interior, and feasibility at the boundaries keeps `x` in `[0,B]`. The active L service has nonzero physical traffic on at least one HBM row.
- Across this compute interval, L has at most `J` connected **HBM-positive-service** intervals. `J` counts actual on-runs, including any hidden LP/tangent switches; it is not automatically implied by a serialized knot count unless the implementation's switch budget also bounds those on-runs. If L progresses with zero input-credit slope, count it as active too; that can only increase measured HBM busy.

The result also applies with a lower bound `c_min` on compute consumption in place of constant `c`, by replacing each `B/c` below with `B/c_min` and using the actual interval length `T`.

## Theorem

Whenever L has no HBM-positive service, it supplies no input credit, so inventory drains at rate `c`. Since inventory never exceeds `B`, each connected no-load-service gap has length at most `B/c`. At most `J` positive-service intervals divide `[0,T]` into at most `J+1` such gaps. Therefore

```text
measure({t in [0,T] : L has positive HBM service}) >= max(0, T - (J+1) B/c)
                                               = T * max(0, 1 - (J+1) B/N).
```

The proof allows a full buffer at `t=0`; initial prefetch does not weaken the result. The left-hand set is exactly time with positive L service, and L has nonzero HBM work. Under the preserved metric (“busy” iff any positive service exists),

```text
actual_HBM_busy >= max(0, T - (J+1) B/c).
```

This is a support-duration bound, not an utilization or byte-weighted statistic. The stronger “L sits on the empty tangent and receives service at C's consumption rate” case is included: that is one continuous positive-service interval, so `J=1`, and the bound becomes `T - 2B/c`. Allowing up to 16 on-runs gives the more conservative `T - 17B/c` bound.

If the budget is expressed as `S` binary on/off transitions instead of an on-run count, use the corresponding upper bound on `J` (at most `floor(S/2)+1` on-runs, depending on the initial state). Keeping `J` explicit avoids confusing curve knots, LP basis changes, and actual positive-service intervals.

## Comparison with the discrete busy target

Let `B_ref` be the discrete oracle's same positive-service busy result. A direct sufficient condition for an unavoidable relative-error failure at tolerance `epsilon` is

```text
max(0, T - (J+1) B/c) > (1 + epsilon) * B_ref.
```

This uses the independently computed `B_ref`; it makes no universal assumption about discrete HBM utilization. In the simple one-row case where the reference HBM phases saturate capacity `H` whenever any row service is present, the reference busy time is exactly `Q/H` for total payload `Q`. More generally conservation only gives `B_ref >= Q/H`; equality or an upper bound on `B_ref` must be established before using it to prove a mismatch.

For an especially transparent load-only case, suppose each consumed K-credit unit corresponds to `q` bytes and the discrete HBM service rate is `H`, with `mu = H/q`. If its positive-service phases saturate HBM, `B_ref = N/mu`, so

```text
actual_HBM_busy / B_ref >= (mu/c) * (1 - (J+1) B/N).
```

Thus the unchanged relative tolerance `epsilon` is necessarily missed whenever

```text
(mu/c) * (1 - (J+1) B/N) > 1 + epsilon.
```

For large K (`N >> (J+1)B`) and a fast loader (`mu/c >> 1`), the bounded-switch curve's busy time approaches the compute duration while the discrete busy remains near total payload divided by HBM capacity. This conclusion is limited to the stated continuous-compute, bounded-on-run, physically coupled L/HBM case.

## Computable numerical illustration

Take `N=640` K-credit units, `B=2`, and `J=16`. At most `(J+1)B = 34` units can be consumed during all no-load gaps, so L must receive positive service for at least `606/640 = 94.6875%` of the compute interval.

The retained historical file `work/analytical-curves/os-80-128-4096.raw.json` reports the v1 fixture's one-Core compute busy as `T = 0.5301997269555547 s`, capacity `B=2`, reference HBM busy as `B_ref = 0.13148160000000145 s`, and `N=640` input units. Substituting these values gives:

```text
c                          = 1207.092285156249 K-credit/s
maximum all-gap time       = 17 * 2 / c = 0.028166860495 s
bounded-switch busy lower  = T - 17 * 2 / c = 0.502032866461 s
lower bound / reference    = 3.818274697
relative-error lower bound = 281.827%
```

This comfortably exceeds the existing 5% HBM-busy check in that *illustrative parameter regime*. The raw also records actual v1 `hbm_busy_s = 0.5304454869555543` and discrete `hbm_busy_s = 0.13148160000000145`, but those observed v1 numbers are not evidence about v3 or its final source/build identity. The calculation only shows how the conditional bound can be evaluated and how a long-K, compute-slow/load-fast regime can make the tradeoff decisive.

## What the bound does not prove

This does **not** show that every finite piecewise-affine representation fails. The bound becomes vacuous if `N <= (J+1)B`; a scheduler with enough on/off intervals can fill in bursts (roughly one useful refill every `B` units); a larger buffer can change the scale; compute stalls or variable/zero consumption change the premises; and L may fail to generate HBM work in some positive-progress segment. Multiple L activities or epochs must be counted in the aggregate `J` over the compute interval. A curve knot count alone is insufficient evidence that `J` is small if hidden allocator changes can create more service intervals. Finally, a discrete oracle that does not saturate HBM needs its actual computed `B_ref` (or a separate valid upper bound) in the comparison.
