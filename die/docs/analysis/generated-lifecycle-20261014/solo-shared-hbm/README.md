# Solo shared-HBM lifecycle descriptor prototype

This is an offline, standard-library Python prototype for a single owner of a
strict-FP32 output-stationary GEMM. It models Load and Store as competing
activities on the same two HBM rows, Compute on the compute gate, two input
credits, two output credits, final-K store production, and store retirement.
It does not modify `die/src/`, and is not a production or hardware certificate.

Run the exact descriptor model and the independent small-shape cross-checks
from the repository root:

```sh
python3 work/solo-shared-hbm-v2/solo_shared_hbm.py --m 65 --n 65 --k 4097
python3 work/solo-shared-hbm-v2/selfcheck.py
```

The generated [selfcheck.json](selfcheck.json) comes from the second command.
It invokes the existing `coarse_compiled_gemm_oracle_driver` for its selected
tile/hardware facts, derives a small shape from that plan, rebuilds its wave
work with `compiled_gemm_oracle.make_waves`, and compares the exact
equal-progress event path with `compiled_gemm_oracle.discrete_oracle`. This
checks lifecycle, credit, busy-union, and wave-ledger handling independently.
The weighted grouped-Q policy is checked against its own fully expanded exact
Fraction event path on a small case and the full `65x65x4097` fixture; the
compiled oracle's equal-progress policy is not mislabeled as a grouped-Q
allocator check.

## Rate and ledger derivation

For each active stage `i`, let `b[i] = (compute, HBM0, HBM1)` be its work per
unit progress and `Q[i]` its canonical weight. Load and Store use their HBM
payload as `Q`; Compute uses the local-bus work. At each event, the owner
weight is `Q = sum(Q[i])` over active stages. The exact progressive fill raises
every unfrozen progress rate by the same `delta / Q`. A physical resource `r`
limits that increment by

```text
(capacity[r] - sum(b[i][r] * rate[i]))
------------------------------------------------
           sum(b[i][r] / Q) over free i
```

and a private Compute ceiling limits it by
`(1 / compute_time - rate[compute]) * Q`. The first exact limiter freezes all
free activities that use it; the at-most-three active stages require at most
three fill tiers. This handles HBM-bound Load/Store and compute-bound Compute
with the same rule. Every stage completion recomputes the allocation, so
concurrent Load and Store consume the same finite HBM capacities.

The next event is the minimum exact completion deadline `(1-progress)/rate`.
For each interval, resource busy time is charged once per active-resource
union. `hbm_busy` is incremented once if either HBM row has an active user; it
is never the sum of row durations. The strict-FP32 row split is symmetric, so
the tested solo cases also satisfy `hbm_busy == hbm0_busy == hbm1_busy`.

Credit occupancy follows the full lifecycle. Input occupancy is loaded waves
plus active Load and Compute. Output occupancy is completed outputs plus active
Store plus an active final-K Compute. A final-K output slot is reserved when
that Compute starts and is released only when its Store retires. Thus a
K-final store cannot free the slot early, and the small `K=1` check reaches two
occupied output slots while matching the expanded schedule.

## What can be powered

The event trace discovers exact returns from the live lifecycle state, not
from the old 520-event fixture. A return requires unchanged stage topology,
exactly translated wave/queue positions, identical live progress fractions,
and matching stage work. This is a zero-dimensional exact identity proof for
one fixed solo state; no affine offset parameter or expanding/converging map is
assumed.

Two bounded forms are implemented:

* `Kmiddle_identity` translates by interior full-K waves within one output.
  First-K accumulator initialization and final-K store/retire stay explicit.
  The arithmetic repeat bound stops before K-final.
* `output_classword_identity` translates by whole K-output periods. The
  compiler proves either a constant tile-shape run or an N-row-aligned
  row-major category word. The word distinguishes full and tail N tiles; the
  row bound stops before a short M row. The K-first, K-middle, K-tail, store,
  and retire behavior all remain in the repeated template.

The C++ lowering direction is a fixed descriptor record with: geometry-class
key (`M_full/M_tail`, `N_full/N_tail`, `K_first/K_middle/K_tail`); normalized
active stage/queue coordinates; exact rational start progress and per-event
rates; tied completion set and nonnegative deadline margins; wave/output
translation; category-word schema and arithmetic repeat bound; exact ledger
deltas for elapsed, HBM rows, operations, local bytes, and busy unions; and a
global discovery counter. The solo prototype's guard is exact equality of all
progress coordinates plus class-word invariance. A production lowering must
also include arrival/external-stop and capacity-change deadlines in the
minimum repeat bound and fail closed when a guard does not hold.

The 65x65x4097 result uses 72 expanded events under a global 256-event cap,
with 617 K waves and four output tiles skipped. The independent unaccelerated
Fraction event path uses 1,310 events and has identical finish time and every
ledger. A much larger K case (`K=64,000,001`) and an output-heavy case with
2,000,010 output tiles both finish under the same 256-event cap: the observed
event count stays 72, while the repeat count grows arithmetically. The
load/store-limited and compute-limited capacity settings also match their
unaccelerated exact paths.

## Boundaries

This descriptor applies only while the owner remains alone, capacities and
work facts remain fixed, and no cancellation, admission, timeout, or other
external stop occurs. A shared-runtime caller must truncate repetition at the
next external event boundary and resume ordinary event simulation there.
This prototype does not support arrivals or changing capacities.

The fact producer currently implements strict-FP32 payload geometry. INT4
packing and its exact per-row/per-tile tail rules need a separate facts source
before this return guard can be used for INT4. If an exact identity or the
restricted category word is not found within the global cap, the model fails
closed; it does not reset that cap for each output. Root's two-owner shared
HBM work remains a separate problem. No fixed independent-server extension is
promoted here as a shared-HBM model.
