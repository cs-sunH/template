# Generated guarded-return interval lowering

Run from the repository root:

    python3 work/generated-interval-lowering/run_generated_lowering.py

The runner reads the first generated_returns entry in
work/symbolic-lifecycle-discovery/generated-K4097.json, verifies it matches
first-return-certificate.json, writes a fixed-format descriptor, builds the
standard-library-only C++17 evaluator, and checks its output with Python
Fraction. The C++ evaluator is under
work/generated-interval-lowering/fixed_interval_generated.cpp. It consumes
one state coordinate and a descriptor-sized ledger vector; the current record
provides eleven ledgers. It accepts negative ledger slopes and does not require
non-time ledger increments to be positive. A scratch probe also confirms that
an exactly zero non-time ledger remains valid.

## Guard and lifecycle checks

The offline checker reads all 297 pulled-back affine guards. It recomputes the
exact rational intersection and both endpoint strictness flags, then checks
each ge, gt, and eq guard over the full interval between the exact initial
state and exact fixed point. The current record has 202 ge, 95 gt, and zero
eq guards. The smallest strict margin is

    14060678425108148397248290804887552497659/5453085325217179744957762253468769177600000000

or about 2.57848128e-6. A separate checker probe adds 0 = 0 to exercise the
equality path and confirms 0 = 1 is rejected. The C++ evaluator receives
only the recomputed strict interval domain and checks the contraction orbit
enclosure against it.

The independent lifecycle scan includes every owner's next_load, active
waves, loaded waves, and computed waves. Their maximum initial references are
192 and 132. With translations [65, 65] and first non-full wave 64935, the
strict bound is min over owners of floor((64935 - 1 - max_reference) / shift),
which gives 996 initial-state returns: after 996 the maximum reference is 64932;
the 997th return reaches 64997 and crosses the tail. From the recorded end
state, the independently derived additional bound is 995 returns, matching the
discovery field. This distinction matters because 995 is measured from the
end state, not the initial state.

The runner also recomputes the geometry bound from M=15985, N=63937,
K=4097, and tile sizes 16/64/64. This gives 65 K waves per return and the
same first non-full wave, 64935. The scratch result uses one state coordinate,
eleven ledger values, 36 representative template steps, 297 guard records,
seven lifecycle reference slots at each boundary, and zero expanded runtime
events.

## Numeric format and event limits

The C++17 evaluator requires the host's IEC 60559 binary long double with a
64-bit significand (the checked x87 format). It compiles with
-frounding-math -fno-fast-math -ffp-contract=off, forces each primitive
through a stored long double, and widens every result with nextafter.
Rational descriptor inputs use eight 32-bit limbs; an input above 256 bits,
non-finite operation, ambiguous comparison, or unsupported floating format
fails closed. No external numeric library is used.

Measured affine composition counts are 1, 5, 9, and 15 for repeat counts
1, 10, 100, and the lifecycle bound 996; they are 26 and 52 at 10^6 and
10^12. The latter two are labeled numeric-only and do not extend the
lifecycle proof. The external-stop bitwise search uses 20 compositions.

The selected stop is exactly 13,099,179,518 ns. Exact Fraction arithmetic
places it strictly between cumulative times for 123 and 124 returns. The
interval evaluator proves the upper bound at 123 is no later than the stop and
the lower bound at 124 is later than the stop. It reports the remaining partial
duration for the ordinary event integrator; it does not apply the return map
to that residual. At 996 returns, the cumulative elapsed-time interval width
is 3.955169525227120175741e-16 s, below the required 1 ns.

The runner verifies exact Fraction containment for the final state and every
ledger at repeats 1, 10, 100, and 996. These are scratch lowering results for
the generated symbolic return. The overall discovery remains
failed_closed at its 256-event cap, and the source record itself says geometry
category lifetime and runtime lowering remain outstanding. This evaluator
does not turn that discovery result into a production certificate.
