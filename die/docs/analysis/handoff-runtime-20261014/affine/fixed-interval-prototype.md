# Fixed-precision affine powering prototype

This scratch prototype adds a C++17 fixed-format interval evaluator for the exact half-output map emitted by `nested_k_output.py`. It uses one continuous boundary coordinate, ten additive ledger values, and 32 template event steps per output segment. It never expands K waves, output segments, or event instances while powering the map.

Run the whole reproduction from the repository root:

```sh
work/affine-compose/run_fixed_interval_check.sh
```

The command regenerates `nested-k-output.affine`, compiles with `-frounding-math -fno-fast-math -ffp-contract=off`, runs the C++ evaluator, then checks its emitted intervals against an independent Fraction binary-powering reference for counts through 995 and the existing Q128 interval implementation through `10^12`.

## Interval arithmetic and failure rules

The runtime format is the host's IEC 60559 binary `long double` with a 64-bit significand. The executable checks that format and sets round-to-nearest. Every primitive add, subtract, multiply, and divide is first evaluated in that format and widened by one adjacent representable value with `nextafter`; interval multiplication/division examines all four endpoint pairs. Each primitive is forced through a stored `long double`, and the build disables fast math and contraction. Rational descriptor values are parsed into eight 32-bit limbs (256 bits), converted through outward-rounded Horner evaluation, then divided as intervals. A 257-bit integer, nonfinite operation, invalid interval, denominator interval containing zero, or ambiguous stop comparison returns failure.

Under these runtime/build assumptions, a rounded finite result and its adjacent representable values enclose the exact real result of that primitive. Induction over the interval expression tree therefore encloses the exact rational affine coefficients, each binary composition, and the final state/ledger outputs. This is a bounded-precision numerical argument; it does not rely on approximate cycle equality. The Python checker additionally compares the C++ decimal endpoints as exact rationals, so output formatting cannot conceal an inward endpoint.

## Domain and lifecycle checks

For the fixture map, the exact recurrence is `x' = (849475/2156112)x + 548564055014789/8288669491200000`. The C++ program verifies `0 < a < 1`, positive `b`, computes an outward interval for `b/(1-a)`, checks that fixed point and `x0` lie strictly inside the exact event-order interval, and checks `f(x0) < x0`. Thus the exact orbit from `x0` stays between its fixed point and `x0`, inside the checked domain for every bounded repeat. The domain itself and its strict deadline margins come from the grouped exact event-order proof in `nested_k_output.py`; this C++ program consumes and checks that boundary rather than rediscovering event templates.

The cursor arithmetic checks the first non-full wave `64935`, K length `65`, and initial cursors `197` and `136`. Their per-owner full-category bounds are 995 and 996, so the common bound is 995; one additional output segment crosses owner 0's first non-full wave. The external stop is the integer-nanosecond value `13,099,016,821 ns`, selected strictly between exact cumulative elapsed times for 123 and 124 segments. The C++ binary search accepts a count only when the outward upper bound at `n` is no later than the stop and the outward lower bound at `n+1` is strictly later. Any overlap fails closed.

The exact event-order interval is approximately `[0.10546826994272858, 0.1978723996438977]`; the contraction orbit is approximately `[0.1092091153117911, 0.19507934280001743]`. The upstream exact trace proof gives a minimum selected duration of `1/312500 s` and a minimum unselected-event margin of about `2.57848 μs` over this orbit. C++ checks the orbit enclosure against the event-order interval. The independently checked 995-segment elapsed interval has width about `3.747e-16 s`, or `3.747e-7 ns`, below the required `1 ns` bound. The exact Fraction checker verifies containment at every reported bounded scale, including all ten ledgers.

The external stop check returns 123 complete segments: the safe elapsed time is about `13.045983903328914 s`, the stop is exactly `13.099016821 s`, and the next elapsed time is about `13.152049737378567 s`. The Fraction checker verifies both strict inequalities using exact values.

## Fixed complexity versus repeat scale

The measured affine composition counts are 1, 5, 9, 16, 26, and 52 for repeat counts 1, 10, 100, 995, `10^6`, and `10^12`. Every run has one state coordinate, ten ledger coordinates, 32 stored event-template steps per segment, and zero expanded runtime events. Counts above 995 are labeled `numeric-only`: they show that powering complexity is logarithmic and do not extend the fixture's full-tile lifecycle proof beyond its first tail.

This is only a fixed-offset, fixed-fixture evaluator. It is not production code, a generic lifecycle template compiler, or evidence for arbitrary offsets. The C++ program does not independently solve grouped LOCAL-Q LPs or derive event-order cells; it is a separate fixed-precision execution of coefficients and bounds supplied by the exact nested verifier.
