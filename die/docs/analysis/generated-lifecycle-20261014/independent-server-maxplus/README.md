# Single-owner OS K-class prototype

This scratch prototype studies the one-Core, one-owner, output-stationary GEMM case requested in the 2026-10-14 handoff. It reads no saved trace. It does not modify `die/src/`, archived evidence, or production descriptors.

The derivation follows design-document §6.2 and §8.1–8.6: an OS output tile is one ordered K stream, only its K-final compute creates an output, store completion retires that output and releases the slot, and input/output queues stay bounded at two slots. The single-owner specialization is stronger than the two-owner fixed-offset fixture: it needs no event-order fit or trace-derived map.

## Exact transition

For a wave index `i`, let `A_i` be its load completion, `C_i` its compute completion, and `R_o` the store retirement of output `o`. Each stage has a fixed effective service duration for its declared K class. With two input slots, a load can start after the prior load and the compute completion two waves earlier. A new output's first C wave waits for the oldest of its two output credits; later K waves of that output cannot increase output occupancy before its K-final C finishes, so the credit check at K-first is sufficient. The resulting completion recurrences are:

```text
A_i = max(A_(i-1), C_(i-2)) + L_i
C_i = max(A_i, C_(i-1), R_(o-2) when this is K-first) + C_i_work
S_start_o = max(C_(o,Ktiles-1), R_(o-1))
R_o = S_start_o + S_o
```

The output slot is held from K-final compute completion through `R_o`; a C-stage completion cannot free it. The state is the seven exact clocks

```text
(A_(i-1), C_(i-1), C_(i-2), R_(o-1), R_(o-2),
 S_start_(o-1), S_start_(o-2))
```

Each L/C wave is a fixed 7×7 max-plus matrix. A K macro is K-first, `Ktiles−2` copies of K-middle, then K-final plus store. The middle matrix is raised by binary composition, so K-middle takes `O(log Ktiles)` matrix operations and allocates no per-K state. Full and tail M/N output classes have separate maps; row-major geometry becomes a few category runs and repeated row maps. Repeated output maps close when the normalized seven-clock state repeats by a common rational time translation. This prototype proves that cycle for the fixed rational stage profile and fails closed at a declared 4096-boundary discovery cap.

The geometry helper accepts any positive M/N/K and tm/tn/tk. It emits full/tail M/N classes, a K-first/middle/final table, exact FP32 or row-packed INT4 payload bytes, matrix work, INT4 conversion/add work, and final output bytes. INT4 output defaults to FP32 (four bytes/element) because this model keeps an FP32 accumulator; `--output-bytes` changes the final storage width explicitly.

## Busy and work ledgers

Per-stage byte/operation work and busy seconds add exactly across K classes and output repeats. L and S intervals are each serial and therefore non-overlapping within their own stage. Their union is

```text
busy_L_or_S = sum(L durations) + sum(S durations)
              − sum(all L/S interval intersections)
```

The intersection calculation does not enumerate K: for a fixed prior store interval, binary search the monotone K-middle load start/end clocks, count fully covered middle loads, and evaluate the two partial endpoints. The output repeat cycle powers this overlap ledger along with the state. `busy_any_stage_union` is elapsed time: under the stated work-conserving lifecycle, whenever work remains and no stage is active, either a load can start or a queued/output-credit condition makes store runnable. The independent small oracle checks that union identity.

The timing assumption is precise and restrictive: L, C, and S have fixed per-stage rates and act as independent servers. `busy_hbm_union_s` is the exact time union of the modeled L/S intervals. If L and S draw from the same physical HBM capacity, their rates must change when they overlap; this prototype does not model that shared-domain LP, so its time and HBM-union values cannot certify such a profile. A future bounded descriptor must split the exact active-set LP categories or fail closed.

## Tails and external boundaries

K tail work is its own K-final class and store still starts only after that final compute. M/N tail output classes break the repeated full-tile run: compose the full rows up to the first tail class, then use actual tail dimensions and INT4 packing in a separate map. No output resets an arbitrary repeat counter or fixed-width cap.

A capacity/profile change or external stop ends the current descriptor epoch. The evaluator must not apply old-rate maps across that event. The live residual needed to resume is absolute time, exact owner/output/K cursors, ordered loaded/computed queues, active L/C/S identities with fractional progress, slot occupancy, remaining work, and cumulative ledgers. The seven max-plus clocks are sufficient for scheduled endpoint composition, but they are **not alone a live scheduler snapshot**: next-output L/C may be ahead of the latest S retirement. This prototype records that boundary format and deliberately does not claim arbitrary-time snapshot reconstruction. A runtime event integrator (or a separately proved state extractor) must own the partial interval and apply the design's same-tick completion/credit/cancel/capacity ordering.

## Verification

Run the exact-rational analytic path against the independent explicit slot/event oracle for small cases:

```sh
python3 work/solo-k-classes/solo_k_classes.py --selfcheck
```

The oracle expands only its small fixtures. It checks completion time, work/byte totals, stage busy time, L/S interval-union busy, all-stage busy union, and both two-slot bounds. Selfcheck includes FP32 and odd-row INT4, K=1/2/many, M/N tails, load-limited and compute-limited profiles, a million K tiles with no middle-wave materialization, and a million-output row repeat with an exact normalized-state cycle.

Example direct geometry invocation:

```sh
python3 work/solo-k-classes/solo_k_classes.py --geometry 4096 8192 16385 64 64 256 int4 1000000000 500000000 --store-Bps 1000000000
```

The capacities are explicit effective L bytes/s, C equivalent-ops/s, and S bytes/s. They are inputs to this specialization, not architectural calibration claims.

## Not established

This result is not a production certificate. It excludes shared L/S HBM capacity, multi-owner fairness, multi-Core allocation, WR, cancellation, live capacity-change snapshot extraction, and arbitrary symbolic deadline/LP guard generation. Cycle discovery is for one fixed rational profile with a fixed cap; it does not prove a global cap over all rates. It does establish a reusable exact single-owner K-middle representation and an independent discrete cross-check for the stated assumptions.
