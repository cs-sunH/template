# Standalone affine credit arithmetic checkpoint

2026-10-17, Asia/Shanghai. No Runtime or Plan compiler was built or executed.
No production model, reference result, inventory, progress, or guard was changed.
This evaluates credit at **given represented progress**, not integration,
allocation-context validation, Norm availability, or hardware accuracy.

The successful standalone probe is `evaluator.cc`; `run.py` extracts immutable
saved observations, compiles only that probe, and checks it with independent
Python `Fraction`. Commands/source/input/output/ELF hashes are in `identity.json`.
The initial wrong JSON include path failed compilation; its exact source,
runner, input, compiler stderr and hashes remain in `attempt1-compile-failed/`.
No Runtime experiment was started to repair that tool failure.

## Data and methods

1952 scalar-channel cases contain:

- 222 cases from all 111 saved current-Core synthetic internal recompute records;
- 148 saved public-state cases from the actual A+B, interior-only and rejected
  physical-enclosure runs, retaining their original provenance;
- 1562 boundary/adjacent-progress cases using the frozen long-row builder's
  actual affine coefficient formulas, for H=513/768/1024/65536/65537/131072/131073;
- 20 dyadic capacity1 input/headroom controls.

The full/tail coefficients are **source-derived**, not newly compiled Plans.
H=65537 is checked against the saved real seven-epoch Plan coefficients. The
frozen builder hash and exact formulas are recorded. These credit coefficients
are independent of the supported BF16/FP32 dtype timing/byte coefficients; this
experiment does not claim mixed-dtype execution coverage or restore the builder.
The saved 456-case/912-channel enclosure implementation was not rerun.

Both inventory and headroom are evaluated directly. For affine initial I,
capacity H, coefficients d_i and endpoint E=I+sum(d_i), an anchor p_j gives

```
q = I*(1-p_j) + E*p_j + sum(i != j, d_i*(p_i-p_j))
h = (H-I)*(1-p_j) + (H-E)*p_j - sum(i != j, d_i*(p_i-p_j))
```

The predeclared structural q pivot is the most negative coefficient; the h
pivot is the most positive coefficient. Selection uses coefficients only,
never oracle error or reference latency. All other anchors are also retained
in raw output but are not cherry-picked for the summary.

Each direct and anchor expression is evaluated with ordinary summation and
fixed-size Neumaier summation. Constants, C++ binary64 parsing and represented
long-double progress (including neighbor generation) are independently checked
from hexadecimal output against exact Python rounding. The observed ABI has
64-bit long-double precision; conclusions are not extrapolated to other ABIs.

## Actual result

| Group | Direct maximum inventory error | Anchor maximum inventory error | Inventory improved / equal / worse |
|---|---:|---:|---:|
| Saved internal | 2.771763e-16 | 1.501620e-17 | 44 / 178 / 0 |
| Saved public | 2.771763e-16 | 1.262069e-17 | 38 / 108 / 2 |
| Full/tail source-formula neighbors | 7.105373e-15 | 3.810971e-17 | 529 / 1017 / 16 |
| Exact dyadic control | 0 | 0 | 0 / 20 / 0 |

There is a real cancellation improvement, but the fixed structural anchor is
**not uniformly better or safe for raw-sign boundary classification**.
On saved internal headroom, 50 cases improve, 152 tie and 20 worsen; four raw
anchor headroom signs are wrong. Case18 (stats stream, Load/Compute coefficients
65281/-65537) has exact headroom `-3.5508679940084338089e-18`; direct headroom
is `-1.3877787807814456755e-17`, while the producer-anchor gives positive
`6.9388939039072283776e-18`. Plain Neumaier summation does not restore the
lost product bits; for actual saved inventory it gives no improvement over
ordinary direct summation.

The original illegal balances are retained: 224 cases have true negative
inventory and 197 have true negative headroom, totalling 421 outside-capacity
cases across the data groups. The saved baseline's negative output inventory
remains negative. No terminal residual or reference was clipped to zero.

For plain expressions we also check a simple normal-arithmetic bound:
`gamma8 * sum(abs(exact input expression terms))`, where u=2^-64 and
gamma8=8u/(1-8u). Exact endpoint/headroom constants are verified before this
bound is used. Compensated results receive a conservative triangle-transfer
bound from their plain counterpart, not an asserted sharper theorem. All
tested eligible bounds contain the exact reference. 116 cases involving
subnormal adjacent progress remain explicitly ungraded for this normal-error
bound; their actual errors and signs are still retained. Near-zero uncertain
bounds are not called exact boundaries. In particular the four wrong raw
headroom signs remain covered by uncertain intervals and are not certified
positive headroom.

## Single next step

Keep this evaluator isolated. Add one fixed-size **FMA product-residual plus
compensated-dot** variant on these same affine cases, checking the exact source
products and sign behavior against the same Fraction oracle. This addresses
the demonstrated product-rounding loss before changing Runtime classification
or inventing an allocation-context solution. Do not install the present
consumer/producer anchor rule just because its maximum inventory error is
smaller. The unchanged one-ULP snapshot and physical-work guards remain separate
future gates.

## Reproduce

Restore the current-Core numeric archive to `work/norm-runtime-current/` as
described by the existing Runtime handoff. Place these source files at
`work/norm-affine-evaluator-20261017/` in a fresh checkout (the runner refuses to
overwrite a completed identity), then run:

```
python3 work/norm-affine-evaluator-20261017/run.py
```

No simulation library is linked. Reproduction requires GCC-compatible C++17,
Python3, and the repository's existing JSON header. Preserve any new failed
attempt under a new result label/directory rather than replacing this evidence.
