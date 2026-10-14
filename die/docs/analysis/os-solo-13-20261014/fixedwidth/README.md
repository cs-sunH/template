# Fixed-width feasibility audit for the 13-row Solo study

This scratch study is isolated from the Runtime and from the frozen
`work/os-single-owner-13-20261014/current/` directory. It replays the captured
class facts through the existing exact Python solver, instruments every
`Fraction` arithmetic result/comparison, and separately exercises the newer
exact stop/resume solver. The C++ rational below is a standalone arithmetic
prototype; it is not connected to `Mapping.cc`, `Runtime`, CMake, or any public
interface.

Reproduce the arithmetic differential test and operation audit from the
repository root:

```bash
python3 work/os-fixedwidth-13-20261014/check_rational256.py
python3 work/os-fixedwidth-13-20261014/bitwidth_audit.py
python3 work/os-fixedwidth-13-20261014/check_bitwidth_audit.py
```

`bitwidth_audit.json` records per-run and per-phase normalized numerator and
denominator widths, conservative unreduced scratch widths, integer geometry
and skip-repeat widths, every source file/facts hash, and stop/resume outcomes.
`rational256_test_results.json` records the C++ source hashes, compiler command,
and Python `Fraction` differential result. `run.log` and
`rational256_test.log` are the captured command output.

The exact solver's normalized rational values fit within 256 bits in every
captured trace. Across all 11 geometry runs and eight pause/resume runs, the
largest normalized numerator was 213 bits and the largest denominator was 153
bits; fact loading/conversion values peaked at 114/153 bits, and assembled
class facts peaked at 100/153 bits. The audit observed more than
one million rational operation/result records, including allocation tiers,
deadline comparisons, active-stage progress, resource/typed/HBM ledgers,
exact cycle identity comparisons, per-step skip ledger deltas, and skip repeat
multiplication. Dense geometry was also expanded without skips for 1,310
events. The large-K and large-output facts were powered without per-wave
expansion. The 1,310-event dense row uses an offline 5,000-event allowance
solely to measure the no-skip arithmetic reference; it is not the bounded
production event budget.

| Captured trace | Events | Exact skips | Max normalized numerator / denominator | Max inferred scratch numerator / denominator |
| --- | ---: | ---: | ---: | ---: |
| Strict FP32 small | 28 | 0 | 209 / 151 | 263 / 196 |
| Native INT4 small | 26 | 0 | 213 / 153 | 267 / 200 |
| Fractional converted INT4 small | 28 | 0 | 212 / 151 | 271 / 204 |
| Converted INT4 storage small | 28 | 0 | 213 / 153 | 266 / 200 |
| Dense 65×65×4097, powered | 72 | 7 descriptors | 213 / 151 | 271 / 204 |
| Dense 65×65×4097, expanded | 1,310 | 0 | 213 / 151 | 271 / 199 |
| Large K, 65×65×64,000,001 | 72 | 7 descriptors | 213 / 151 | 271 / 204 |
| Large output grid | 72 | 7 descriptors | 213 / 151 | 271 / 204 |

Stop/resume widths include the paused candidate plus its completed resumed
execution:

| Stop fixture | Final expanded events | Max normalized numerator / denominator | Max inferred scratch numerator / denominator |
| --- | ---: | ---: | ---: |
| Strict FP32 small | 31 | 209 / 151 | 263 / 202 |
| Native INT4 small | 29 | 213 / 153 | 267 / 206 |
| Fractional converted INT4 small | 31 | 212 / 151 | 271 / 210 |
| Converted INT4 storage small | 31 | 213 / 153 | 266 / 207 |
| Dense strict | 105 | 213 / 151 | 271 / 210 |
| K=1 two-output-slot fixture | 23 | 210 / 151 | 263 / 199 |
| Large K | 102 | 213 / 151 | 271 / 204 |
| Large output grid | 100 | 213 / 151 | 271 / 218 |

This is a concrete failure of a naïve 256-bit scratch implementation: exact
allocator comparisons need conservative raw cross-products up to 271 bits.
The exact large-output stop/resume trace also observes a 259-bit raw numerator
addition. A 544-bit scratch area contains all measured conservative scratch
widths, but 768 bits gives a cleaner checked contract when 256-bit rational
operands interact with a 128-bit skip count. Any implementation must check
every operation and fail closed when a normalized result exceeds its 256-bit
slot; these measurements do not prove that all future facts/geometries stay
inside that contract.

The exact identity state is naturally represented by normalized sign,
numerator, and positive denominator. `Fraction` reduces every value, so exact
progress identity is a direct canonical-pair equality; it does not use a
tolerance or floating-point equality. Comparisons that choose progressive-fill
limits use exact cross-products. The largest observed cross-product exceeds
256 bits, while both normalized operands/results remain bounded. The C++ test
also compares two approximately 255-bit fractions whose raw comparison
products are over 256 bits, adds a fraction to itself with over-256-bit
unreduced products, and multiplies by a large reciprocal pair that reduces to
one.

The measured integer repeat count reaches 1,000,002 (20 bits) in the
large-output geometry; its translated-wave count reaches 130,000,260 (27
bits). The pause/resume run keeps the same execution-wide event cap: dense
finishes in 105 expanded events after capacity invalidation, while large K and
large outputs finish in 102 and 100 events. The measured 17 ns stop is encoded
as the exact rational `17/1,000,000,000`; production should accept the external
clock as a bounded integer and convert it exactly, without epsilon equality.

The stop/resume audit includes six small/dense fixtures with a 17 ns absolute
stop, exact `finish/3` and `finish/2` stops, pause/resume, a 7/8 HBM capacity
change, rejection of a backwards stop, and final completion. The capacity
change clears saved identity returns but preserves the existing global event
count. The large-K and large-output cases stop at `finish/3` and `2*finish/3`,
resume to completion, and match their no-stop powered ledger with 102 and 100
total expanded events respectively. These checks exercise exact rational
clamping and resume state; they do not change or reinterpret the Runtime's
existing 2 ns public timing bound. The earlier 3 ns converted-storage and
6 ns requested pressure deviations remain failures under that bound.

## Bounded C++ arithmetic prototype

[`checked_rational256.hh`](checked_rational256.hh) stores each canonical
numerator and denominator in eight 32-bit limbs (equivalent to a 256-bit slot)
and performs add/subtract/multiply/divide/compare/normalization with 24
32-bit-limb (768-bit) scratch. It checks parser, intermediate, state-result,
and 128-bit repeat-count limits; it has no Boost/GMP dependency. Exact binary64
conversion records the input as a rational and rejects non-finite values and
values whose exact denominator exceeds the bounded representation. This
includes subnormals and the minimum normal binary64.

`check_rational256.py` compiles with `-std=c++17 -Wall -Wextra -Werror
-pedantic -O2`, runs the C++ self-tests, then checks 2,247 operations against
Python `Fraction`. Coverage includes signed/zero normalization, all four
arithmetic operations, ordering/equality, cross-products above 256 bits,
normalization after wide operations, large repeat multiplication, 257-bit
state/repeat overflow, zero denominator/divisor, over-budget decimal input,
exact binary64 conversion, and subnormal/non-finite rejection.

## Fixed-width design candidate and proof boundary

A production-bound candidate is a 256-bit normalized rational with a checked
768-bit scratch integer; exact progress coordinates and branch comparisons
stay rational. The 768-bit width covers the operation shapes under explicit
input limits: two 256-bit components can make a 512-bit cross-product; adding
two such products needs at most 513 bits; multiplying one 256-bit numerator
by a 128-bit skip count then adding it to a 256-bit ledger can need at most
641 numerator bits. A checked 128-bit geometry/skip count and a checked 256-bit
result slot make overflow fail closed. For an absolute unsigned 64-bit
nanosecond stop, conversion contributes at most a 64-bit numerator and a
30-bit denominator. Against a 256-bit stored clock, raw `stop - now` needs at
most 321 numerator / 286 denominator bits; dividing that by a 256-bit period
needs at most 577 / 542 scratch bits. Both fit within 768. The stop-clamp
quotient is checked against u128 and clipped by the geometry-derived repeat
bound before it updates lifecycle counts. This is a width derivation, not a
worst-case stop-time test: if any normalized stop delta, progress, or ledger
subresult exceeds 256 bits, that operation must fail closed rather than round
or resume from an approximate identity state. The captured integer-stop run
uses 17 ns; the other captured stop boundaries are exact fractions.

The old interval evaluator in
`die/docs/analysis/generated-lifecycle-20261014/interval/fixed_interval_generated.cpp`
already has an eight-limb 256-bit rational descriptor parser and outward
`long double` interval arithmetic. Its generated proof is for one scalar
affine return and eleven ledgers over 996 returns; it demonstrates a useful
numeric enclosure kernel, not a 13-row event allocator or an OS proof. It may
be reused after a separate exact branch/guard derivation, with ambiguous
interval comparisons failing closed. It cannot choose the progressive-fill
active set or prove the current exact identity return by itself. The old host
format restriction (`long double` with a 64-bit significand) and compiler
rounding flags also remain requirements for that numeric kernel.

For a production descriptor, class facts should remain a fixed cache of at
most 32 geometry-derived records. Runtime state is fixed to three active
stage slots, two input/two output queue slots, 13 service/busy rows, and a
global 256-event discovery budget. Identity keys compare canonical rational
pairs, class IDs, queue positions, and checked wave offsets. Capacity changes
invalidate the saved identity cache while retaining the execution-wide event
counter. Integer-nanosecond cancellation clamps the next exact interval before
applying a repeated return; capacity/stop changes never reset the cap. All of
this can remain internal to Runtime and preserve the public C ABI. It still
needs a production-owned proof of event/guard bounds, exact input width
validation, interval error bounds at the existing 2 ns limit, and integration
review before any production use.

The audit is limited to one owner, the captured synthetic capacities, and the
actual class facts identified by hashes in the JSON report. It makes no
production timing, multi-owner, silicon, descriptor-certification, or public
ABI claim. No tracked/public source was changed.
