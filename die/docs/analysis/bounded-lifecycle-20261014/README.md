# Bounded lifecycle research: exact cycles and affine contraction

This directory contains independent scratch research, not production code or a
granularity certificate. The public Runtime and its production K lifecycle did
not change in this checkpoint. GMP remains a scratch dependency.

The previous v1 only skipped nonfinal K within each output and remained O(R).
`output-cycle-v2/` adds exact whole-output/store/retire recurrence detection and
uses the real strict-FP32 GEMM compute Q=LOCAL bytes. The q=1 branch is a
pure-compute toy. The default 650-wave, 1-Core fixture was rebuilt by the root
and run for seven arrival scenarios and all three policies. Independent
equal-progress and grouped LOCAL-Q references each matched the corresponding
seven event/cycle results; exact LP point checks also passed. Commands and
root outputs/checkers are in `root-verification/`.

Scale experiments show both progress and limits. At R=10⁶ with Ntiles=2, solo
finishes with 72 executed intervals and skips 999994 output retirements;
closed-form byte/ops/LOCAL/retirement totals agree, without expanding the huge
wave oracle. At R=1000 and offset 1/20, the exact-repeat detector fails closed
at 256 outer probes. Its progress numerator/denominator grow to about 16k bits;
a fixed 128/256-bit rational replacement is unsafe. The corresponding long
expanded event attempt was interrupted and is not a passing result. Absolute
N-column state can also obstruct a repeat despite identical full-column work.

`derive_affine_cycle.py` offers a different representation for one fixed
offset/full-MN segment: 520 independently checked event intervals compose a
one-dimensional affine contraction. It validates all LP rates/Q/bases, 521
slot/queue/lifecycle states, invariant-line closure, progress bounds, strict
event order and identity ties. The entire orbit from x0 to its fixed point lies
inside the same feasible trace interval. Geometric power/error bounds are
checked with exact Fractions and explicit seconds. Initial invariant lines
were conjectured from samples and then proved by symbolic event composition;
three-point fitting alone is not the proof.

This proof has 520 logged intervals and does not establish discovery under a
flat 256-probe limit. It does not cover arbitrary offsets, tail transitions,
external capacity changes/cancellation, J=32 or the thirteen-domain production
solver. The saved C++ row-prefix formula has a known typo; all current FP32
payloads/prefixes are divisible by four, so each row gets exactly half and the
tested results are unaffected. The independent checker uses the correct
prefix. No generic dtype claim is made.

`allocator-eight/` and `allocator-q-priority/` retain twelve actual Runtime
zero-credit allocation queries and their inputs/source. The root's independent
checker validates rate/alpha and physical capacity inequalities. These are
allocator checks, not compiled lifecycle evidence. `workspace-implementation-design.md`
is a read-only interface proposal and has not been implemented or tested.

Reproduce the small root checks from the repository root after installing GMP:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic die/docs/analysis/bounded-lifecycle-20261014/output-cycle-v2/bounded_lifecycle.cpp -lgmpxx -lgmp -o work/root-bounded-v2
work/root-bounded-v2 1/7 1/97 14/101 91/97 > work/root-bounded-output.txt
python3 die/docs/analysis/bounded-lifecycle-20261014/root-verification/check_bounded_grouped.py --raw die/docs/analysis/bounded-lifecycle-20261014/output-cycle-v2/raw-fixture.json --cpp-output work/root-bounded-output.txt --output work/root-bounded-grouped-check.json
python3 die/docs/analysis/bounded-lifecycle-20261014/output-cycle-v2/derive_affine_cycle.py
```

The last command recomputes the adjacent proof report; run a scratch copy when
preserving the archived report byte for byte. Scale command provenance is saved
separately; scratch binaries and dependencies are not committed.
