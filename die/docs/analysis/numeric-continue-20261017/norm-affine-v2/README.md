# Frozen-input FMA affine arithmetic checkpoint v2

2026-10-17, Asia/Shanghai. This builds only a standalone arithmetic probe. No
Runtime/Plan compiler is linked, built, executed, or modified. No production
model, test reference, physical balance, or snapshot rule is changed.

## Frozen scope

The input is the identical 1952-case v1 file, SHA256
`000cebcb1a2d8027bc8bb353f04a95076929ea53320333a0898f1086a43e4700`.
All old expression outputs are required to remain bit-identical, and the
unchanged copied Python Fraction oracle independently rechecks their old
summary. Given-progress source coefficients/progress, full/tail provenance,
the old wrong signs, and all true negative balances remain intact.

The new evaluator computes each affine product as

```
high = coefficient * progress
low  = fma(coefficient, progress, -high)
```

It applies fixed-size Neumaier summation to the initial constant and the at most
six high/low terms. Headroom uses its own constant `capacity-initial` and
negated coefficients; it is never recovered by subtracting rounded inventory.
The executable verifies FE_TONEAREST and 64-bit long-double precision. No fast
math or contraction is enabled; only the explicit fma supplies fused products.

The finite/normal domain checks are conservative. For every nonzero product,
the bound `ilogb(a)+ilogb(b)-2*(digits-1)` must be at least the minimum normal
exponent. This bounds the possible exact product quantum and avoids assuming
that a zero FMA residual proves an error-free product when underflow could have
erased it. Product/sum/residual/correction nonfinite and subnormal results are
also checked. These conditions cover lossless product splitting in the tested
normal domain; they do not prove that a single compensated final sum is always
correctly rounded for arbitrary inputs. Sticky FE_UNDERFLOW flags are not used
as an arithmetic proof or claimed measured here.

## Actual arithmetic results

On all 1952 q values and all 1952 independently evaluated h values:

- zero nonfinite results;
- zero incorrect high+low product decompositions against Fraction;
- zero incorrect point signs;
- zero incorrectly rounded final values against independently rounded Fraction.

The last three facts also hold empirically on samples outside the conservative
normal domain; those observations do not certify an underflow-capable domain.
1879 cases per side meet the new normal-product/summation checks; 73 per side
are explicitly unsupported by them. The old normal-bound policy remains
unchanged: only 1836 per side are graded, while all original116 subnormal-neighbor
cases remain ungraded. The43 extra cases satisfy the new arithmetic checks
because their subnormal progress multiplies zero; this does not retroactively
change the old bound scope.

| Group | FMA maximum inventory error | FMA maximum headroom error | Incorrect signs / incorrect rounding |
|---|---:|---:|---:|
| Saved internal222 | 1.273938e-17 | 2.168404e-19 | 0 / 0 |
| Saved public148 | 1.262069e-17 | 5.421011e-20 | 0 / 0 |
| Full/tail coefficient neighbors1562 | 2.775558e-17 | 2.775558e-17 | 0 / 0 |
| Dyadic capacity1 controls20 | 0 | 0 | 0 / 0 |

In case18 the prior structural anchor produced positive headroom
`6.9388939039072283776e-18` for a true balance
`-3.5508679940084338089e-18`. FMA product residuals restore the correctly rounded
negative value. All four wrong internal headroom signs are corrected; no
previously correct structural-anchor sign regresses. The224 true negative
inventory and197 true negative headroom cases remain negative, including the
real saved baseline output debt. No balance is clipped or declared legal.

These are given-progress arithmetic results. Correctly rounded inventory may
still equal512 when exact inventory is512 plus a tiny positive residual; the
separately evaluated headroom preserves its negative sign. The retained old
trajectories are not repaired or rebranded as valid integration.

## Signed intervals remain unresolved

No bound is narrowed to obtain a pass. The v1 direct-expression
`gamma8 * term-L1` bound and the same triangle-transfer rule are used for the
new point estimate. Every graded interval still contains Fraction, but504 q
and311 h intervals remain sign-uncertain. In the saved internal data the eight
negative headroom observations remain interval-uncertain. A correct observed
raw sign is not substituted for a certified boundary sign.

The single next step is to certify a tighter final-summation remainder for
this fixed set of at most seven high/low terms, for example using error-free
TwoSum accumulation with an explicit fixed-size residual bound, and verify it
on these unchanged cases. That is an offline arithmetic dependency before any
Runtime physical-view experiment. This checkpoint proves neither multi-role
integration, snapshot historical allocation context, Norm availability, nor
hardware calibration. The existing one-ULP/work/credit guards are unchanged.

## Cost, only for this tiny evaluator

Each method evaluates the same1879 normal-domain samples, both scalar q/h,
over five batches of100 rounds after one warmup. Parsing, JSON, Fraction,
Runtime work and model compilation are excluded. Original per-batch averages
are retained in `benchmark.json`; no sample is discarded.

Median batch-average nanoseconds per scalar expression are:

- direct:8.82ns (range8.72–8.95);
- structural anchor with compensation:21.95ns (range21.69–104.02);
- FMA product residual plus compensation and its implemented safety checks:
  764.23ns (range668.00–954.78).

The FMA cost includes per-product finite/normal/exponent checks. This is a
local arithmetic micro-cost, not a Runtime/API timing budget or end-to-end
performance result. It does not justify installing the slower path globally.
At32 owners, two channels and separate inventory/headroom would require128
scalar dots per update; simply multiplying the observed median gives about
97.8us before arbitration or other work. That suggests a material risk to the
unchanged50us arbitration budget. This estimate is not a measured Runtime
failure or a reason to change the budget.

## Review of the proposed integer-split follow-up

No v3 was started. `INTEGER-SPLIT-REVIEW.md` records a feasible narrower route:
for certified affine coefficients `d_i=C_i*2^k`, integer `|C_i|<=2^53`, split
64-bit progress significands32/32 and coefficient integers26/27. Each partial
product has at most60 significant bits and is exact on the observed64-bit
long-double ABI when its scaled value is finite and normal. A fixed-size
TwoSum expansion can then preserve the finite sum without software `fmal`.

This requires exact round-trip coefficient certification, exact progress
decomposition, representable partial products and residuals, FE_TONEAREST,
validated expansion/sign invariants, and explicit unsupported behavior outside
the domain. It does not establish performance by itself. A cheap proven normal
bound may bypass that work only when it actually certifies the required sign;
the815 uncertain intervals here cannot be treated as zero. The original
integration and snapshot-context gates still apply.

## Reproduce

Restore v1 at `work/norm-affine-evaluator-20261017/` using its saved source and
evidence. Place v2 files in a fresh `work/norm-affine-evaluator-20261017-v2/`
directory and decompress its frozen input there. `oracle_base.py` is the exact
unchanged v1 runner, loaded only for its Fraction functions and checks.

```
python3 work/norm-affine-evaluator-20261017-v2/run.py
```

The runner refuses to overwrite a completed identity. All commands, input and
source hashes, executable identity, old-output identity checks, and full raw
results are saved. No binary or object is required in the Git evidence archive.
