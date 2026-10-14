# C++ workspace admission validation — 2026-10-14

Scope: `astra-sim-wscllm` C++ workspace lease/admission, explicit bridge v2 parsing and serialization, snapshot preflight/install/wakeup, and coarse COMP lifecycle. No Python or public `die` source was changed in this workstream. Nothing was committed or pushed.

The aggregate `astra-sim-wscllm` build completed for `AstraSim_Analytical_Congestion_Aware_Online`, `WorkspaceAdmissionTest`, `DecisionMailboxTest`, `ParsedBatchTest`, `GraphBatchCommitterTest`, `DieModelAdapterTest`, `DieExecutionLifecycleTest`, and `LocalHbmDieMemoryTest` (CMake `-j2`, protobuf source install environment from the root coordinator). The final focused CTest run passed 7/7:

- `AstraSim_Analytical_WorkspaceAdmissionTest`
- `AstraSim_Analytical_Congestion_Aware_DecisionMailboxTest`
- `AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest`
- `AstraSim_Analytical_Congestion_Aware_ParsedBatchTest`
- `AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest`
- `AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest`
- `AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest`

The lifecycle fixture verifies a real materialized FFN Plan with positive exact workspace: it remains in the free set when blocked, acquires before issue after a snapshot, advances to positive elapsed time and typed served work, records cancellation utilization, releases its GPU slot and lease once, marks `die_cancelled_comp_count=1` in the rank service report, terminalizes with the existing `Skipped` watch semantics, and lets a real dependent successor issue. Existing shutdown cancellation remains callback-suppressed and is covered by lifecycle case F. Completion lease release and normal service metrics are covered by case G.

The parser and committer tests cover v2 full-rank ordered tables, exact integer typing/overflow, source-sequence and revision binding, atomic preflight rejection, initial snapshot installation before issue, and capacity-only changed-rank wakeup. DecisionMailbox tests cover v1 field compatibility and v2 full-rank serialization with the existing single delivery sequence.

`git diff --check` passed. Test output is in [cpp-workspace-ctest-final-20261014.log](/workspace/template/work/cpp-workspace-ctest-final-20261014.log). An earlier lifecycle attempt used a GEMM Plan whose exact workspace is zero; the fixture was corrected to use a real materialized FFN Plan, after which the final suite passed. No design checks were weakened.

Limit: this validation is focused C++ build/CTest coverage and does not include a complete long-running Online simulation exchanging batches with the Python scheduler. The cold-performance agent was notified that build/tests had released the CPU and is running the requested final N30 measurement separately.
