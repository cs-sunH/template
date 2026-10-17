# Integer/dyadic split proposal: static feasibility review only

No v3 code, compile, or experiment was started. This reviews the parent's
proposed bounded arithmetic follow-up; it is not a correctness certificate or
permission to change production.

## Exact products without long-double fma

For one affine channel require `d_i=C_i*2^k`, with exact signed integers
`|C_i|<=2^53`. Zero coefficients need no product. The common power and integer
coefficients must be mechanically certified by exact reconstruction of the
actual binary64 Plan coefficients, never by a tolerance or rounding a nearly
integer coefficient. A shared power may not exist within this integer bound
when coefficients have widely separated exponents; reject that domain rather
than enlarging integers or claiming every binary64 coefficient set qualifies.

On the observed64-bit-significand ABI, a nonzero represented progress has

```
p = M * 2^(e-64),   M = M_hi*2^32 + M_lo
C = C_hi*2^26 + C_lo
```

`frexp`/`ldexp` can extract M exactly before an unsigned64 conversion, including
p=1; M is in `[2^63,2^64)` for nonzero normal progress. Each M chunk has at most32
bits. For `|C|<2^53`, the C chunks have at most27 and26 bits; the inclusive
endpoint `|C|=2^53` makes the high chunk28 bits, which still keeps its product
within60 bits. Preserve sign separately.

The four unsigned integer partial products are exact in long double:

```
C_hi*M_hi * 2^(k+e-6)
C_hi*M_lo * 2^(k+e-38)
C_lo*M_hi * 2^(k+e-32)
C_lo*M_lo * 2^(k+e-64)
```

Their unscaled integer products require at most60 significant bits. Scaling
by an exactly representable power of two preserves exactness only when no
nonzero low term is lost to underflow and no term overflows. Explicit exponent
checks are therefore necessary even when the final product would be finite.
Zero/subnormal progress requires separate certification or unsupported status;
do not infer safety from a rounded zero partial product. Reject other
long-double significand sizes unless they receive their own decomposition and
proof; a32/32 split discards information on113-bit progress.

## Fixed finite expansion and sign

At most three products create at most12 partial terms, plus the exact initial
constant. Headroom must start from capacity and negative initial as exact
terms, or use error-free subtraction; a rounded `capacity-initial` is not
automatically an exact constant in a broader domain. Headroom then uses
negative coefficients directly, not `capacity-rounded_inventory`.

A fixed-capacity TwoSum expansion can retain these terms without variable-size
bignums, replay, or new simulation state. Its finite bounds and sign rule need
an explicit proof/validation: the largest component's sign is sufficient only
after nonoverlap/order or an equivalent domination invariant is established.
Do not substitute one final rounded sum or rely on the observed v2 signs.
Check rounding mode, overflow and residual representability; retain every
nonzero residual rather than applying the model credit epsilon to arithmetic.

A cheap normal-arithmetic interval is a useful fast path when q/headroom is
provably positive or provably negative. An interval containing zero establishes
neither zero nor availability. Invoke the certified finite expansion only for
uncertain signs. The v2 dataset has815 such scalar intervals under its frozen
bound, including true negative balances; treating them as zero repeats the
original physical-inventory mistake.

## Necessary limits and validation

The first domain should remain actual affine `(0,0)->(1,d)` channels with at most
three activities. Arbitrary piecewise slopes can introduce division/constant
errors not covered by this integer-product proof. Inventory and headroom must
share the same source progress, coefficients and exact constant semantics.
Outside the certified domain, disclose unsupported/old-path scope explicitly;
a fallback to a path with known physical-credit defects is not a safe result.

Keep the same1952 frozen cases and Fraction products, including signed boundary
neighbors and all421 negative-inventory/headroom cases. Validate product
decomposition, expansion identity, sign, and any cheap-bound exclusion
independently. Then measure actual fast-path and fallback frequencies/costs;
fixed item count does not guarantee the50us J32 update budget. Stable tangent
states may trigger uncertain signs often, so do not assume the slow path is
rare from ordinary interior samples alone.

Even a successful arithmetic result would not prove multi-role integration,
clock/progress representability, one-ULP historical allocation reconstruction,
the long-row builder, or five-repository consumers. Those gates remain intact.
