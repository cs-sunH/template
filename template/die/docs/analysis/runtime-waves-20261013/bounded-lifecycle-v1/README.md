# Bounded lifecycle prototype (scratch only)

This directory contains an isolated C++17 model for the strict 1-Core GEMM
fixture `M=65, N=65, K=4097`, selected tile `16x64x64`. It does not alter the
production simulator or its current failed status. The model keeps two owners,
two fixed input-slot IDs and two fixed output-slot IDs per owner, and at most
one active `L`, `C`, and `S` activity per owner. Waves are generated from an
integer index on demand; no array scales with `R*Tk`.

The lifecycle is explicit. A load completion makes that input slot available
to compute. Only a K-final compute completion creates an output item and
reserves its output slot. Store completion writes the output bytes and retires
that slot. Request completion is recorded only after all ten output stores
complete. The fixed eight wave classes are the M-tail, N-tail, and K-tail bit
combination; the detector also includes each active, queued, and next wave's
exact dimensions and row-split payload in its key, so class equality alone
cannot certify a repeat.

The detector uses GMP-backed exact rationals and has a 256-transition probe
cap per stable class segment. Its key includes owner membership, bounded slot
queues, active stage and wave signatures, exact progress fractions, exact
stage work and Q, rates, positive-rate masks, resource tier basis, and the
fixed capacity generation `g0`. A repeat is skippable only when all shifted
live/queued wave IDs line up, no K-final work or output retirement occurs in
the cycle, and the skip stops before any owner reaches its next K-final wave.
It also caps skips at a pending arrival; admission clears the old prefix and
restarts detection from the new joint state. No epsilon matching is used. If
no exact repeat is found within the cap, the accelerated path reports
`fail_closed`.

This is a bounded-sample execution prototype, not a general periodicity proof.
In this v1, detector probes reset at wave-class changes, and a skip is rejected
if its cycle completes a store or retires an output. The skip cap stops before
the next K-final wave. Thus fixed `Tk` does not make total work bounded in
output count `R`: K-final load/compute/store and retirement still run for each
output, and the local 256-probe cap can be re-applied at every output. The
fixture has only `R=10`; its successful acceleration is not evidence of a
bounded representation for large `R`. `output-cycle-v2/` holds separate
research source for recurrence across complete output cycles. Cancellation
and capacity-change events are not implemented in either executable. To
extend either model, they must enter the next-event scheduler, cap a skip at
their exact event time, then apply the event, clear the detector prefix,
update the capacity generation where applicable, and recompute rates. They
must never be hidden inside a skipped period.

## Independent comparison

`check_oracle.py` compares the C++ `equal_progress` event and cycle paths with
the independent per-wave implementation in
`die/validation/compiled_gemm_oracle.py`, including per-owner read/write
bytes, operations, stage counts, finish times, HBM union/shared time, and row
masks. The `grouped_q_alpha` event and cycle paths are compared with each
other; that allocator differs from the legacy per-activity equal-progress
oracle, so the checker does not present the latter as a grouped-policy oracle.
The earlier grouped max-min LP scratch analysis remains separate evidence.

After a permitted run, build and check with:

```sh
g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic bounded_lifecycle.cpp -lgmpxx -lgmp -o bounded_lifecycle
./bounded_lifecycle > cpp-output.txt
python3 check_oracle.py cpp-output.txt
```

GMP is a scratch-only exact-arithmetic dependency. It is not a production
library dependency. The `pre-gmp-failure.txt` file retains the earlier fixed-width failure and are not evidence for the current GMP
source. Current source changes still require a new compile and fresh
independent comparison before any prototype result can be reported.

## Archived validation

The archived source was rebuilt and executed by the parent, and the archived checker was run against its fresh output: compile succeeded without diagnostics; the independent equal-progress fixture check passed. The checker locates the repository by ancestor search so it also runs from this saved directory. `validation.json` records source/output/binary hashes. Grouped allocation is only checked event-versus-cycle here; the separate two-stage LP study does not certify this three-stage allocator. GMP rational numerator/denominator size can grow with events; no production memory/CPU bound is proved.
