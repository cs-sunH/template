# LRU C++ workspace-pressure diagnostic

`Workload` keeps bounded evidence from the most recent issue pass that observed coarse Plans denied by workspace admission: the number of blocked Plan observations and the last observed Plan's node id, generation, and requested bytes. `NodeStoreGraphSource` traverses the full ascending free-node snapshot; a callback return skips that node only, so a pass may record multiple blocked Plans. The count is for that last issue pass; it is neither a live count of currently blocked nodes nor a cumulative count. Matching issue or terminal paths invalidate the evidence to avoid stale labels. The Online dead-end census prints these fields as `last_issue_pass_blocked_plan_count` and `last_observed_*`, alongside the capacity authority bit, current and snapshot revisions, capacity, model weight, resident KV, reserved request bytes, and active workspace bytes. The census reads existing state only and never compiles a Plan, mutates admission, or schedules a retry.

Case K retains its single-Plan capacity retry/cancellation path and adds a separate rank-2 fixture with two independent positive-workspace Plans. The test inserts IDs 902 then 901, verifies the full pass observes both in ascending order and leaves node 902 as the last observation, with both nodes free, GPU occupancy and leases at zero, and no queued or deferred work. It drains a real no-op event and checks the state remains blocked and quiescent. The rank-2 fixture avoids replacing `Sys::all_sys[0]`, which supplies the shared simulation clock used by the rest of this lifecycle test. This verifies the blocked-plus-quiescent predicate using real workload, ledger, node store, and event queue objects. It does not execute the full Online closed-input fatal branch, and production Online E2E was not rerun for this diagnostic change.

## Build and test

Run from `/workspace/template` with the configured cloud environment and apt protobuf mode:

```bash
source /workspace/cloud-setup/activate.sh
unset PROTOBUF_FROM_SOURCE
cmake -S work/five-os-20261014/aggregate -B work/lru-cpp-liveness-pressure-20261014/two-plan-confirmation/build/aggregate \
  -DASTRA_SIM_ROOT=/workspace/template/astra-sim-wscllm-LRU \
  -DBUILDTARGET=congestion_aware -DBUILD_TESTING=ON \
  -DASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST=ON -DCMAKE_BUILD_TYPE=Release
cmake --build work/lru-cpp-liveness-pressure-20261014/two-plan-confirmation/build/aggregate --parallel 2 --target \
  AstraSim_Analytical_WorkspaceAdmissionTest \
  AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest \
  AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest \
  AstraSim_Analytical_Congestion_Aware_ParsedBatchTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest \
  AstraSim_Analytical_Congestion_Aware_LocalHbmModelTest \
  AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest \
  AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
  AstraSim_Analytical_Congestion_Aware_BridgeLoopbackTest \
  AstraSim_Analytical_Congestion_Aware_Online coarse_die_cabi die_model_cabi
ctest --test-dir work/lru-cpp-liveness-pressure-20261014/two-plan-confirmation/build/aggregate \
  -R 'DieExecutionLifecycleTest' --output-on-failure
ctest --test-dir work/lru-cpp-liveness-pressure-20261014/two-plan-confirmation/build/aggregate \
  -R '(WorkspaceAdmissionTest|DecisionMailboxTest|GraphBatchCommitterTest|ParsedBatchTest|LocalHbmBandwidthModelTest|LocalHbmModelTest|DieModelAdapterTest|DieExecutionLifecycleTest|BridgeLoopbackTest)' \
  --output-on-failure
```

The final lifecycle test passed 1/1 and all nine focused tests passed 9/9. Logs are in `two-plan-confirmation/logs/configure.log`, `build-final.log`, `lifecycle-focused.log`, and `focused-9.log`. The initial rank-0 two-Plan attempt failed a later service assertion because it replaced the test's rank-0 clock; its log is preserved as `lifecycle-focused-attempt1-failed.log`, and its source/build logs are labeled `attempt1-rank0`. The corrected rank-2 fixture passes. The accepted pre-two-Plan binaries and identity remain preserved under `pre-two-plan-fixture/`.

## Identity

The final build used HEAD `928f05309722869a8339918340aa953e3ea7c327` on branch `codex/die-workspace-envelope-20261014`. The five edited LRU files and local coarse C++ source/CMake inputs are listed and hashed under `two-plan-confirmation/source-identity/`; the source manifest checks pass. The C ABI target compiles from `astra-sim-wscllm-LRU/extern/die_model/coarse`, not the concurrent public `die/` working tree.

Final binary SHA-256 values are recorded in `two-plan-confirmation/source-identity/production-binaries.sha256` and `focused-test-binaries.sha256`. Online SHA: `13431b68fe4a62c67f9a52a93f0b94539b762b733b0a98dc88f7d3b4388da63a`; coarse C ABI SHA: `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b`; die-model C ABI SHA: `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`; lifecycle test SHA: `c801f1525784199f717b5ed5bc4e5641eeed5596a481ec5646ffafb1139976a6`.
