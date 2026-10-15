# C++ consumer boundary regression source

Base commit: `9bb8c0ac74ad3b78edb0004d7ebbc61f00942dda`. This subtask edits only four `execution_driven/tests/die_execution_lifecycle_test.cc` files (FACE, FACE-LRU, WSCLLM, WSCLLM-LRU) and active JOINT's `execution_driven/tests/coarse_workspace_runtime_test.cc`. JOINT's legacy-only lifecycle test, production code, public SDK, and Native Domain tests are unchanged by this subtask.

The new `test_incomplete_coarse_prediction_boundaries` exercises both `prediction_status` and `alone_estimate_status` independently. Each iteration starts with a legal R1 Plan from the actual compiler and adapter preparation, copies the Plan, marks only the selected status unavailable (the other remains `ok_assumed`), and recomputes its canonical fingerprint. A separate public diagnostic Runtime accepts the copied Plan and its wire round-trips, establishing that prediction completeness is the intended rejection reason rather than malformed work, resources, hardware, owner, or hash.

The actual Sys clock advances to tick 17 while the untouched production Runtime remains at 0. Tests then assert:

1. `CoarseDieRuntime::submit_plan` throws the specific `prediction_not_available` error naming the selected field. Entire Runtime counters, Sys tick, callbacks, graph ready set, GPU occupancy, and terminal observations remain unchanged.
2. A manually assembled `PreparedCoarsePlan` cannot bypass `try_acquire_workspace`; lease count, active bytes and revision remain unchanged.
3. The const-reference `validate_prepared_coarse_plan` rejects that Plan while preserving the caller's valid canonical lease, ownership, bytes, count and revision. The caller explicitly releases that lease once.
4. Actual `Workload::issue_prepared_coarse` rejects the manual bypass before graph take, GPU occupancy, Runtime submission or terminal publication. This by-value call releases its transferred lease during rejection, as required by the existing RAII protocol.
5. Fixture teardown leaves no deferred completion callback.

FACE and FACE-LRU reuse their existing `coarse_system_latency.json` fixture. WSCLLM uses its existing `coarse_system.json`; WSCLLM-LRU refreshes that temporary configuration using its existing `write_coarse_system_config`. Their test-local task is a materialized BF16 FFN (`m=64`, hidden 128, FFN 256), explicitly one layer instance/weight pass. JOINT reuses its existing materialized attention `task_json`, `coarse_node`, Sys, ledger, and terminal-observer seam. No production emitter task, workload count, or calibration value is changed.

The old SDK24 cannot naturally produce the new top-level unavailable marker. These tests therefore cover old-SDK compatibility and manual Plan/Prepared bypass protection. They intentionally do not mutate immutable cached Plans or introduce test seams. A genuine new-SDK R3 rejection through `prepare_coarse_plan` belongs to the root agent's separate isolated integration validation.

Build and test status here is **not_run** by explicit task instruction: this agent did not compile, run CTest, or execute a CPU/performance benchmark. `git diff --check` passed. The root agent owns actual builds/runs and should bind their resulting binaries and logs separately.

Existing targets/CTest entries required:

- FACE, FACE-LRU, WSCLLM, WSCLLM-LRU: `AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest`.
- Active JOINT: `AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest`.

All added code uses APIs already present in SDK24 (`Plan`, `plan_to_json`, `plan_from_json`, `Runtime::submit`), without depending on the unpublished public Prediction header. `source-identity.json` records the frozen test hashes and observed consumer/header hashes; these are source identities, not evidence that an executable was built from them.

Root validation after this source-only subtask: all five actual old-SDK consumer builds and **10/10 CTest entries passed**. The agent's frozen source-identity/patch above remains historical. Root corrected the four non-JOINT tests to the actual `lookup_ptr` API after a retained FACE compile failure. FACE-LRU's Workload prepared seam is private; that test now exercises its existing public adapter issue method, after a retained compile failure, without widening the production API. The other four retain their existing public Workload seam. The current tested source hashes, complete assertions, binaries, formal SDK identities and commands are in `../checkpoint-summary-final.json`, `../old-face-joint/`, `../old-three/`, and `../old-lru-public-seam/`. The original old-three matrix exit1 reflects the preserved first FACE-LRU failure; its later dedicated build/test passed. New-SDK real R3 ready-scan evidence is separate (`../preview-repeat-face/` currently), and is not a claim that repeated-region execution is supported.
