# Read-only production local samples and authoritative workspace leases

This checkpoint connects the actual new JOINT Sys/Workload state to a typed
read-only sample. It does not enable the sh30 coarse provider or claim a full
production world import.

`CoarseDieRuntime::sample_snapshot_at(now)` requires the current Sys decision
tick and rejects sampling during callback dispatch. It advances a private
Runtime copy, exports Runtime v3 state, and retains every real completion
binding, including a copy which is terminal before its live callback runs.
The existing mutating `export_snapshot()` remains for compatibility.

The adapter catalog reads actual active COMP records and the graph's immutable
node metadata. The HBM catalog now retains issue-time completion rank/node
and sink identity, and exports each real typed endpoint. It does not read the
legacy scalar `job_snapshot()` queue, which excludes coarse jobs. The unused
empty callback object in `CoarseJob` was removed; actual callbacks remain in
their original Runtime closures.

`Workload::sample_coarse_local_state_at(now)` checks that COMP and endpoint
owners form a bijection with Runtime callback handles, and that unique COMP
lease counts/bytes equal the authoritative workspace ledger. It also copies
capacity revisions and local network/HBM join flags without exporting raw
callback pointers. These local flags do not recreate network or pool state.
Unbound direct Runtime callbacks fail closed at the Workload aggregation seam.

Lease validation now checks the actual ledger owner pointer, live key/id and
canonical Plan bytes. Prepared input shares the immutable raw task reference;
preflight checks it against the current node without another compilation or
normalization model. Workload runs preflight before graph take, COMP occupancy
or statistics publication. The adapter repeats validation before submission;
workspace acquisition rejects inconsistent Prepared amounts before mutation.

## Actual verification

The final source built the real Sys fixture, three existing DIE/HBM lifecycle
fixtures and Online. The four fixtures plus the four native/window fixtures
passed: 8/8 CTests. The updated tiny legacy Online completed with C++/Python
exit 0, one accepted/completed request and 192/192 successful terminal nodes.
Public mapping/timing/calibration sources did not change; canonical23 matches
all five active repositories. No target-accuracy or host-budget pass is claimed.

The Sys fixture verifies positive partial physical service while live progress,
counters, cache, event queue, callbacks, workspace and COMP occupancy remain
unchanged. A same-tick one-byte endpoint sample becomes terminal only in the
private copy; the real queue later delivers its callback exactly once. It also
tests foreign same-rank tokens with identical public keys, wrong lease/Prepared
bytes and stale task payloads before graph take/occupancy, plus read-only owner
catalog rejection of inconsistent copied metadata.

The evidence includes actual emitted JSON, not reconstructed examples:

- `partial-workload-sample.json`: tick 1, one typed attention COMP and one
  registered COMM_READ endpoint, 40960 workspace bytes held.
- `drained-workload-sample.json`: actual tick 303318, no execution bindings or
  workspace holds, with cumulative physical counters preserved.
- `terminal-local-runtime-sample.json`: sampled tick 1 while the live Runtime
  remains at tick 0; the copy's terminal callback is still pending externally.
- `terminal-ticks.json`: actual COMP completion at 2408 ns and transfer
  completion at 303318 ns. These are same-model integration facts.

## Reproduction

Use the aggregate scaffold and toolchain prerequisites from the preceding
[native snapshot package](../native-world-snapshot-20261014/README.md). Configure
the tracked aggregate in a fresh work build directory, with Release,
`BUILDTARGET=congestion_aware` and `NETWORK_BACKEND_BUILD_AS_LIBRARY=ON`. Build:

```sh
cmake --build "$BUILD_DIR" -j2 --target \
  AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest \
  AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest \
  AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest
COARSE_DIE_SAMPLE_EVIDENCE_DIR="$REPO_ROOT/work/sys-local-sample-20261014/samples" \
  ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R '^AstraSim_Analytical_Congestion_Aware_(CoarseWorkspaceRuntime|DieModelAdapter|DieExecutionLifecycle|LocalHbmDieMemory)Test$'
```

Set the evidence variable only for optional JSON output. The fixture otherwise
uses isolated temporary configuration files and removes them normally. The
actual final run also executed all four native/window CTests in the same build.
Online reproduction uses the existing tiny Online runner with an isolated
`JOINT_REPRO_WORKDIR`; its newly relinked executable/source identity is recorded
in the evidence.

Next, bind these actual local samples to native Domain leg identities and
compare imported physical continuation with the untouched Sys queue. Full
graph/ledger/endpoint-join/cancel/capacity/network/pool import remains open.
The two WSCLLM branches need the same lease hardening; FACE branches first
need the workspace admission flow itself. No JOINT catalog is automatically
copied into another repository.
