# Five-repository coarse runtime integration audit

Audit date: 2026-10-13. Scope: current `CoarseDieRuntime` and the five
`DieModelAdapter` / `Workload` consumers, compared with design-document §§12.3,
12.3.1, 12.4–12.8, and 14.5. No sync was run.

## Concrete gaps

1. **Plan workspace never enters capacity admission or release.** The compiler
   emits `Plan::summary["hbm_workspace_peak_bytes"]` (for example,
   `die/src/Compiler.cc` and `die/src/Mapping.cc`). `CoarseDieRuntime::compile`
   only caches/returns that plan; `submit_plan` submits it to `Runtime`. The
   adapters read plan work for utilization, then call node completion, but no
   facade, adapter, or `Workload` path reads the workspace field. A search of
   all five source trees found no consumer of `hbm_workspace_peak_bytes`.
   This conflicts with design §12.3.1/§12.4 and §12.7, which require the
   workspace increment to be admitted against local HBM alongside existing
   weights/KV and held for the COMP lifetime, without charging original
   tensors twice.

   The adapter is too late to perform a *deferrable* admission by itself:
   online `Workload::issue` first calls `graph_source_->take_node`, occupies
   the hardware slot, and starts statistics (`Workload.cc:447–450`), then
   dispatches to `issue_die_comp` and the adapter. A workspace check there
   would need a new rollback/requeue contract. The pre-issue ready/admission
   owner must therefore see the plan workspace (or a stable reservation
   prepared by it) before the node is taken; the adapter then owns the
   lifetime token and returns it at completion/cancel.

   The existing owners differ and should remain distinct: wscllm uses its
   `SessionKVCacheManager` capacity/used ledger and fixed decode reservations;
   wscllm-LRU adds remote-pool and eviction/migration ledgers; FACE uses its
   own candidate/score policy and capacity admission; FACE-LRU combines that
   policy with tiered KV state; JOINT has per-rank KV reservation and pipeline
   ledgers. The coarse adapter currently joins none of those ledgers. The
   smallest later implementation is a per-device ephemeral workspace
   reservation token acquired after compile and before submit, included in
   the existing capacity decision as a separate increment, and released or
   rolled back once on success, cancellation, or submit failure. It must not
   become an HBM transfer/endpoint charge or duplicate weight/KV residency.
   FACE must preserve candidate ordering/scoring when capacity rejects a
   candidate; LRU/JOINT must keep their current restore, pool, and reservation
   accounting paths intact.

2. **Normal COMP completion is single-shot in the four non-JOINT consumers.**
   `CoarseDieRuntime::dispatch_terminal` removes a terminal handle from its
   callback map before invoking the callback and retires the execution first.
   Zero-work plans are deferred to the next Sys tick, avoiding a synchronous
   callback inside issue. The four adapters erase their `coarse_active_`
   entry and funnel successful completion through that repository's
   `finish_generic_node`; those finish paths claim the online terminal before
   releasing the `HwResource`, stats/metrics, and graph dependency. Duplicate
   online completion fails closed at `mark_online_terminal_or_fail`.

   JOINT is currently a hard integration break: its adapter calls
   `workload_->finish_generic_node(node_id)` at
   `astra-sim-joint/astra-sim/workload/DieModelAdapter.cc:147`, but JOINT's
   `Workload` has no such method. Its terminal API is
   `finish_general_node(node_id, EventType)` at
   `astra-sim-joint/astra-sim/workload/Workload.cc:1684`. That API has JOINT-only
   behavior the generic four-repo exit cannot replace: a
   `RemoteReadPipelineAnchor` must not release the ordinary hardware slot;
   completion/release ledger transitions and `RemoteReadExecutor` notification
   occur there. Route the coarse completion through the JOINT terminal API
   with the correct event, then fixture both an ordinary COMP and a pipeline
   anchor. The adapter does retain a readiness check for
   `PipelineKernelContract`, but the coarse branch only checks non-null and
   does not pass that contract into the coarse execution; the legacy branch
   does pass it to `DieExecution`. Confirm whether coarse-v2 intentionally
   supersedes the contract's legacy timing behavior before claiming JOINT
   pipeline compatibility.

3. **Callback exceptions can strand same-tick completions.**
   `dispatch_terminal` sets `dispatching_`, erases all ready callbacks into a
   local vector, and resets `dispatching_` only at the normal tail. If one
   callback throws, later callbacks in that vector have already disappeared
   from `completion_` but have not been retired or delivered, and the facade
   remains permanently marked as dispatching. Valid-work callbacks should
   succeed, but a deliberately failing callback/runtime invariant fixture is
   needed before treating the dispatch loop as exception-safe.

4. **Live cancellation is not a completion path today.** Production call
   sites found for coarse COMP use `cancel(..., false)` during teardown;
   no caller uses notification cancellation. If `cancel(..., true)` is used,
   the adapter callback sees `state == "cancelled"` and returns without a
   graph terminal, hardware release, or workspace release. That is coherent
   only for teardown, where the workload is being destroyed; online live
   cancellation needs an explicit cancelled terminal/rollback path before it
   can satisfy design §14.5.

## Independent fixture set

- **Common COMP lifecycle fixture, per adapter flavor:** one zero-work plan,
  one multi-event plan, and two plans ending on the same Sys tick. Assert one
  graph terminal, one hardware-slot release, one online-statistics/metrics
  completion, one dependency release, and `has_active_die_executions()==false`
  after drain. Include a duplicate/stale callback injection and a callback
  exception case for facade retirement.
- **Workspace admission/lifetime fixture:** seed resident weights/KV and
  committed reservations, then submit plans whose workspace is exactly the
  remaining capacity and one byte above it. Assert exact-fit admission,
  over-capacity deferral/rejection under that repo's existing policy,
  additive accounting for two overlapping COMP plans, and one release per
  terminal/cancel. Check that workspace does not alter HBM transfer bytes.
- **LRU-specific fixture:** run restore plus COMP plus remote-pool activity
  with workspace pressure; assert eviction/migration/restore counters and
  endpoint bytes remain unchanged except for the intended capacity admission.
  Use for wscllm-LRU and FACE-LRU.
- **FACE-specific fixture:** exercise a candidate set with a capacity-blocked
  top-ranked candidate and a feasible next candidate; prove workspace gating
  preserves the selector's legacy score/tie policy and only changes
  feasibility.
- **JOINT-specific fixture:** use a frozen pipeline contract and a
  `RemoteReadPipelineAnchor`, then an ordinary COMP. Assert the anchor's
  non-release ownership, completion/release ledger transitions, remote-read
  notification, and exactly one GraphSource terminal; also test a missing
  contract fails before issue.
- **Cancellation/drain fixture:** destroy a workload with active plans and
  verify suppressed callbacks and reclaimed handles; separately define and
  verify the intended online live-cancel terminal and reservation rollback.

## Build preparation

The root CMake files retain `die_model` and link `coarse_die` in all five
trees; wscllm already had that link before this pass, and the other four root
files were updated. The current vendor copies have no
`extern/die_model/coarse/CMakeLists.txt`, so the coarse target and
`<coarse_die/...>` include path are not present yet. The aggregate driver
`build_all.sh` is ready for use after canonical freeze/sync. It uses the
portable Protobuf prefix in `work/deps/protobuf`, separate build trees/logs per
repo, and refuses to configure any selected repo if the coarse vendor target
is absent. It does not synchronize or fetch dependencies.
