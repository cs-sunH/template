# Output-cycle v2 with GEMM canonical Q (scratch only)

This is a separate scratch copy of the hierarchical v2 lifecycle prototype.
It does not edit production sources, the root design, the archived v1, or the
existing v2 experiment. The fixture is strict 1-Core GEMM `65x65x4097` with
`tm=16, tn=64, tk=64`; it keeps two input slots, two output slots, fixed
queues, at most one live L/C/S activity per owner, and generates each wave
from an integer cursor rather than allocating a wave or event array.

Three allocator policies are emitted. `equal_progress` is the legacy
event-vs-independent-wave-oracle comparison. `grouped_q_alpha_q1` is retained
as a pure-compute toy. `grouped_q_alpha_qlocal` uses the GEMM compute canonical
weight implied by the mapping: for a wave with dimensions `m,n,k`,

```text
Q_C = 4*k*(m+n) + 4*m*n + (k_index > 0 ? 4*m*n : 0)
```

The first term is A/B local reads; the second is the accumulator write; the
last term is the prior accumulator read for noninitial K waves. Load/store Q
remains the sum of that activity's physical HBM payload bytes. `Q_C` is
recorded as canonical work and used in each owner's active-Q sum for alpha.
LOCAL is not added as a fourth physical capacity in this three-resource
probe; its positive, nonbinding work is also accounted in the independent
ledger. Thus this corrects the q=1 weight for the GEMM scenario but does not
claim a complete Runtime or production allocator proof.

The local byte ledger is checked against the independent compiled-wave oracle
for event and accelerated paths. For this fixture it is
`9,636,640` bytes per owner: all A/B reads plus one accumulator write per
wave and one prior-accumulator read for each noninitial K wave. The exact
Fraction LP checker separately validates the C++ tier rates for both the q=1
toy state and the qlocal GEMM-weighted state at one four-activity point. The
three physical LP domains are COMPUTE_GATE, HBM0, and HBM1; local bytes affect
canonical Q and the ledger only. Neither point check establishes all Runtime
LP bases.

The recurrence detector is unchanged from the hierarchical v2 design: exact
GMP rational keys, inner full-K and outer whole-output/store/retire levels,
each capped at 256 probes. It preserves K-final production, slot occupancy,
store retirement, positive-rate HBM/compute union integrals, and the fixed
eight M/N/K full/tail classes. No epsilon phase equality is used; a missing
repeat fails closed. The tested offsets in `cpp-output-v2-local.txt` and
`oracle-check-v2-local.json` are solo, 0, 0.05, 1/7, 1/97, 14/101, and 91/97.
All three policies completed and matched event-vs-cycle in all seven samples;
qlocal peaked at 208 inner probes and 19 outer samples, below each 256 cap.
The exact-Fraction point LP agreed for both canonical-Q branches. These finite
samples are evidence for these states only. Strict-contraction exact-key limits,
GMP bit-growth limits, O(256^2) bounded discovery work, and unsupported
external cancellation/capacity-change handling remain the v2 limitations.

Build and run only in this scratch directory:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic bounded_lifecycle.cpp -lgmpxx -lgmp -o bounded_lifecycle-v2-local
./bounded_lifecycle-v2-local 1/7 1/97 14/101 91/97 > cpp-output-v2-local.txt
python3 check_oracle.py cpp-output-v2-local.txt --offset 1/7 --offset 1/97 --offset 14/101 --offset 91/97 > oracle-check-v2-local.json
python3 check_grouped_lp.py cpp-output-v2-local.txt > grouped-lp-check-v2-local.json
```

GMP is a scratch-only dependency and is not part of production.

## Exact affine trace check

[`affine-cycle-proof.md`](affine-cycle-proof.md) documents a separate exact
520-interval symbolic proof for the offset-`1/20` Qlocal full-M/N category
cycle, including an independent per-event grouped LP reconstruction, strict
deadline-region checks, and bounded lifecycle/slot replay. The compact exact
results are in [`affine-cycle-proof.json`](affine-cycle-proof.json), produced
by [`derive_affine_cycle.py`](derive_affine_cycle.py). This establishes only
that trace segment and its 497-cycle prefix before N-tail; it does not extend
the finite-sample prototype result to arbitrary offsets or production.

The frozen C++ trace clone's generic row-prefix expression has a tile-count
typo (`tn*tm` instead of `Ntiles*tm`). It does not change this FP32 fixture's
row split because all payloads and prefixes are divisible by four, so both
expressions yield an exact half split. The clone remains unmodified and is
not a general dtype row-split proof.
