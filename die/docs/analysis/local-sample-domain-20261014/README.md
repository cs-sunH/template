# Actual Sys local sample to native Domain

`NativeCoarseDieDomain::import_local_sample()` accepts a strict
`coarse_die_local_sample_v1`, an expected rank, the same immutable Hardware,
and an explicit complete handle-to-leg map. It preserves the actual Owner,
Plan and partial Runtime progress. It does not invent graph or ledger ids.
Canonical Runtime import validates the original state; terminal handles are
retired only in the private copy through canonical `destroy_execution()`.
Their original progress is retained with one pending completion. Equal-finish
pending completions retain handle order even when external leg ids sort in
the opposite order. The existing Domain wire importer validates the result.

This is a rank-local physical bridge. It does not restore graph readiness,
workspace leases, external callbacks, network, pool or the production ledger.
The sh30 guard and `live_world_import_supported=false` remain in place.

## Actual results and limits

Final native/window CTest: 4/4 passed. Final real-library Python coarse
projection suite: 4/4 passed, no skip. Unit cases cover partial Plan plus
endpoint continuation, endpoint registries, pending terminal reclamation and
one-shot completion, tied terminal order, complete/unique bindings and
malformed rank/Hardware/Owner/time/lifecycle rejection without input mutation.

The replay consumes the actual Sys JSON saved in the preceding
[Sys sample checkpoint](../sys-local-sample-20261014/README.md). The original
Sys queue was continued independently after sampling. Imported COMP finishes
at 2408 ns and local HBM transfer at 303318 ns, exactly matching that queue.
The same-tick terminal sample yields one pending completion at 1 ns. Final
served counters, handle/endpoint/transaction registries, capacities, clock and
integer event/generation fields are equal to the saved drained Sys Runtime.

**Strict final Runtime wire equality is false.** Four cumulative floating
timing strings differ after the sample copy split integration at 1 ns while
the untouched live Runtime continued from 0 ns. Exact Decimal differences
(replay minus original, in ns) are:

| Field | Difference |
| --- | --- |
| compute_busy_ns | +2.23e-16 |
| hbm_shared_ns | -1.110e-16 |
| row_shared_ns[0] | -1.110e-16 |
| row_shared_ns[1] | -1.110e-16 |

The first strict comparison exited 2 and remains in the evidence. The final
runner grades completion timing, exact-once behavior and all Runtime fields
except these four explicitly ungraded accumulators. It reports the strict
failure and exact decimal differences; it applies no numeric tolerance. Its
zero exit is not a bit-identical busy-counter, target-accuracy or full-world
restoration certificate. Production timing and calibration were not changed.

## Reproduction

Activate the configured C++17/CMake/Protobuf environment. Configure the
tracked aggregate described in the preceding native snapshot README in a
fresh build directory, using Release, `BUILDTARGET=congestion_aware` and
`NETWORK_BACKEND_BUILD_AS_LIBRARY=ON`. Then run:

```sh
BUILD_DIR=/absolute/aggregate-build ./die/docs/analysis/local-sample-domain-20261014/run_replay.sh
```

The runner builds the real Domain target, links the checked-in replay source
to its actual native/public libraries, and saves the result under `work/`.
It reads the committed real samples without modifying them. Existing build
caches must retain their original CMake source directory.

The corresponding checkpoint also hardens the two WSCLLM adapters with the
same exact-ledger/bytes/raw-task preflight already validated in new JOINT.
Each existing real Sys lifecycle target passed from a separate clean build,
including prior blocked/retry/cancel flows. That scope and its exact
source/binary identities are in `evidence/ws-lease-hardening-20261014/`.
FACE admission/bridge work is separate and is not certified by this package.
