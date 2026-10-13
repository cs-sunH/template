# Exact affine-cycle check for the Qlocal trace

This is a scratch proof for one exact state segment. It is not a production
certificate and does not cover arbitrary owner offsets, M/N-tail transitions,
cancellation, capacity changes, or other Runtime allocation bases.

The inputs are the 520 selected intervals in
[`affine-trace-complete-cycle.txt`](affine-trace-complete-cycle.txt), from
joint probes 5 to 9, plus exact boundary states at probes 5, 9, and 13 in
[`scale-r1e6-cycle-trace.txt`](scale-r1e6-cycle-trace.txt). The geometry is
`M=15985, N=63937, K=4097`, with `tm=16, tn=64, tk=64`: 1000 M tiles, 1000 N
tiles, and 65 K waves per output. The owners arrive at `0` and `1/20`.

## Exact cycle map

Let `x` be owner 1's compute progress at a joint-cycle boundary, `u` owner 0's
compute progress, and `v` owner 1's load progress. The exact boundary samples
give candidate lines

```text
u = (125/209) x + 415317/3800000
v = (800000/217547) x
```

The exact 520-event symbolic composition validates those lines, rather than
assuming that three samples prove closure. For each logged interval it treats
the independently checked rate as constant, forms every deadline
`D_i(x)=(1-p_i(x))/rate_i`, advances all progress by the selected affine
deadline, removes completed activities, and starts newly enabled activities
at progress zero. It checks the event-time sum against the exact endpoint
clock, every selected activity reaches progress one identically, and the
composed endpoint returns to both boundary lines. The resulting recurrence is

```text
x' = (721607775625/4648818956544) x
     + 1648756992419734626143/17871299754010214400000
u' = (125/209) x' + 415317/3800000
v' = (800000/217547) x'
```

Thus the 3-coordinate boundary state lies on an exactly invariant affine line
for this event trace. Its contraction coefficient is
`a=721607775625/4648818956544`, and its fixed point is
`x*=1645692165044367/15069183193600000`. The initial state is
`x0=317921483913953891933450891003571855541/
1629703480392930105404480315562393600000`.

The exact event-order region is
`[7514576096093/71249638400000, 244491137/1235600000]`; it contains the closed
orbit segment `[x*, x0]`. Across all active stages and all 520 intervals, the
checker evaluates each affine progress at both endpoints and verifies it is
in `[0,1]`. It checks 1310 deadline inequalities. Every nonselected deadline
margin is strictly positive at both endpoints (the smallest is
`14060678425108148397248290804887552497659/5453085325217179744957762253468769177600000000`),
and every exact identity tie is included in the selected set. Selected `dt` is
strictly positive at both endpoints; the minimum is `1/312500` seconds. Since
these expressions are affine and the orbit stays between `x*` and `x0`, the
same selected event sequence remains valid for every repeated cycle in this
segment.

This proof input contains 520 event intervals for one 2-output cycle. It is a
saved event-only trace, not evidence that a flat 256-probe detector can
discover the cycle. A bounded implementation still needs a sound way to
compose this event family through a shorter inner recurrence, or it must
fail-closed when that composition is not proved.

The exact cycle duration is affine too:

```text
T(x) = (-10493508275415568/9382517447874157125) x
       + 43498643214866702774924171/204936926558414922900000000 seconds
T(x*) = 524416/2472125 seconds
```

For 497 repetitions, the checker asserts with `Fraction` comparisons that
`a^497 < 2^-1335`, `|x0-x*| < 1/8`, the time-tail prefactor is `< 2^-13`
seconds, and the exact accumulated-time limit error is `< 2^-1348` seconds.
The resulting state/time rationals already have about 21,000-bit
numerators/denominators (`x_497`: 21038/21041 bits; elapsed time: 21059/21053
bits). This is a concrete warning that materializing exact GMP fractions for
many repetitions grows with the count; the affine formula avoids event
expansion but a fixed-size rational implementation still needs a bounded
representation or fail-closed policy.

## Independent grouped allocator and lifecycle checks

For every one of the 520 interval states, the checker reconstructs wave sizes
from the fixture dimensions and recomputes Q and physical work. Load/store Q
is the payload byte sum. Compute Q is
`4*k*(m+n) + 4*m*n + (k_index>0 ? 4*m*n : 0)`. Compute work and private time
are `vec_ops/158216000`; load/store work is assigned to the two HBM rows.
It then solves the exact grouped max-min tier LP from those values, with
capacities `COMPUTE_GATE=1`, `HBM0=50,000,000 bytes/s`, and
`HBM1=50,000,000 bytes/s`. Recomputed Q, rates, and basis match all logged
values exactly. The first state has active Q values `24576`, `20480`, and
`28672`, owner active-Q sums `24576` and `49152`.
At that first state the grouped coordinate `Q_owner * activity_rate` is
`19,777,000` for owner 0's compute and owner 1's compute, and `240,000,000`
for owner 1's load; the compute pair shares its exact compute-gate tier while
the load reaches its HBM tier separately.

The lifecycle replay starts from the bounded probe-5 state: owner 0 has C195
active and loaded wave 196; owner 1 has L135 and C134 active. It checks all
520 transitions and the endpoint, including K-final-only creation of a
computed output, bounded FIFO queues, input/output slot occupancy, stage
membership, and zero progress for each newly started activity. It never
exceeds three active activities per owner or two occupied slots of either
kind. From probe 5 to probe 9, each owner advances its load cursor by 130
waves and retires two outputs; active and queued waves shift by 130 while
their stage/slot roles remain identical.

The repeated segment stays within full M and full N categories. K-full and
K-tail both occur in its fixed 65-wave pattern, and K-final store/retirement
is included in the 520-event cycle. The repeated cycle advances each owner by
two outputs. Starting at owner 0's `next_load=197`, 497 applications end at
`64807` (still full N); the next application would cross to wave `64937`, in
the N-tail output, so it is outside the proved skip range. No M/N-tail
transition is skipped.

## Scope and source note

The strict endpoint checks certify only the recorded offset-`1/20`, full
M/N-category trace and its exact contraction orbit. They do not prove that all
offsets have an exact joint recurrence, that a 256-probe detector always
finds one, or that a production numeric representation can safely hold these
rationals. The 497-cycle result relies on this exact affine trace and stops
before the N-tail boundary.

The frozen C++ trace clone's `make_wave` prefix currently has
`mt*(tn*tm + N)` where the generic complete-M-row prefix is
`mt*(Ntiles*tm + N)`. This report does not modify or regenerate the frozen
trace. For this FP32 fixture, every byte payload and row prefix is divisible
by four, so the ordered row split is exactly half in either expression; the
trace's per-row work and the independent LP rates are unchanged. The clone
must not be described as a generic dtype row-split implementation.

The checker and compact machine-readable results are
[`derive_affine_cycle.py`](derive_affine_cycle.py) and
[`affine-cycle-proof.json`](affine-cycle-proof.json). The checker uses exact
Python `Fraction` arithmetic and does not compile or run the C++ prototype.
