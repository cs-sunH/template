# LRU source-release gate audit and scratch prototype

Date: 2026-10-14

## Scope and repository state

This task stayed within audit and scratch prototyping. At start, `HEAD` was
`fecf81c5574755209c5d58ecdb982843ff349042` on
`codex/die-workspace-envelope-20261014`. The shared worktree already had
uncommitted workspace-envelope, LRU manager, C++ lifecycle, and other-variant
work. The target `graph_batch_builder.py`, `generate_wsc_llm_trace.py`, and
`online_service.py` were clean at that point. No tracked production file was
changed by this task; the patch and copied fixtures live under this directory.

`die/docs/HANDOFF.md` says cold-store physical release ordering is still a
blocker for prospective workspace admission. The LRU design note says the
source bytes remain resident through direct `mem_store` completion or the
source-side ACK receive, and that gating only the first COMP is too late when a
restore or migration writes KV first.

## Audited behavior

- `GraphBatchBuilder._emit_side_branch` explicitly restores each rank's main
  frontier after store emission, so the store remains a no-join side branch
  (`graph_batch_builder.py:762-795`). Preserve this behavior for legacy mode.
- `pending_store_tails` is a separate per-session pool-write ledger containing
  `(edge_rank, edge_store_node_id)` (`graph_batch_builder.py:797-813`). It feeds
  the established same-session store-before-restore path
  (`graph_batch_builder.py:815-855`); it cannot represent source HBM residency
  across sessions.
- The generator already labels a direct-edge store's release point as
  `mem_store_completion` and retains that node as `edge_store_node_id`
  (`generate_wsc_llm_trace.py:1042-1063`). On the non-edge path it emits the
  source `ack_from_edge` receive after the edge `mem_store`, but discards that
  receive node ID (`1110-1127`). The edge `mem_store` ID still has a different
  purpose: pool-write ordering.
- The target write is `local_hbm_kv_restore` at
  `generate_wsc_llm_trace.py:1211-1219`. For NoC migration, the destination
  `recv_kv` node is the data-arrival/write point (`966-978`). These are the
  earliest transfer nodes that can allocate the restored destination KV.
- The C++ committer accepts cross-batch parent IDs and resolves them through
  its committed ID map (`GraphBatchCommitter.hh:55-59`). If a source-release
  node has already completed and NodeStore GC erased it, the committer skips
  that edge as a satisfied dependency when `node_gc` is on
  (`GraphBatchCommitter.cc:1906-1916`); `NodeStore::add_dependency` also treats
  an already-finished parent as a no-op (`NodeStore.cc:61-74`). A live,
  unfinished parent remains protected by its child dependency and is collected
  after the child finishes (`NodeStore.cc:245-268`). This supports retaining
  source-release IDs across batches without pinning completed nodes. This GC
  point was audited in source; the scratch run did not execute the C++ runtime.

## Minimal contract proposed

Use the existing `workspace_admission == "required"` selection as the
activation gate. The scratch `GraphBatchBuilder` constructor adds an explicit
`source_release_capacity_gates` boolean, default `False`; `online_service.py`
sets it only when workspace admission is required. If the prospective capacity
feature later gets a distinct policy switch, pass that switch here instead.

Each `remote_store` shard carries its existing `source_release_dependency` and
`source_rank`, plus a common `source_release_node_id` interpreted in that
source rank's node-ID space:

- direct edge: `source_release_node_id == edge_store_node_id`;
- non-edge: `source_release_node_id == ack_from_edge` receive ID, while
  `edge_store_node_id` remains the pool-write tail.

When enabled, the builder tracks those IDs in a separate
`pending_source_release_tails[source_rank]` ledger. It aggregates releases from
different sessions and stores. It consumes them at the next same-rank COMP, KV
restore write, or migration destination receive. A direct same-rank parent edge
is attached to the consumer; the old store branch is not joined to the current
main chain. The first ordinary compute consumes its rank's tails; timer gates
and control markers do not. Destination KV transfer nodes receive the tails
before the write/arrival node, not at the pool `mem_load` or a later compute.

`pending_store_tails[session_id]` remains unchanged and continues to order
same-session pool writes before pool reads. The source-release ledger is not
retired by session terminalization: release obligations belong to the rank's
freed capacity and may be consumed by a different session. Already-satisfied
IDs remain valid no-op parents under the existing NodeStore GC contract.

## Scratch artifact and checks

- `prototype.patch` contains the three-file prototype delta; `git apply --check`
  passed against the current repository files. It has not been applied.
- `test_source_release_capacity_gate.py` is a real `GraphBatchBuilder` plus
  `_emit_kv_transfer` fixture. It checks direct-edge release metadata,
  non-edge ACK release metadata, different-session reuse, repeated stores,
  multi-rank tails, remote restore, NoC migration, a `P segment end -> D store`
  trigger, and acyclicity after adding every tag-matched send-to-receive edge.
  The P-triggered store and D consumer are emitted in one GraphBatch in the
  cycle case; the fixture also keeps a direct-edge release tail across batches
  until a later P-session compute consumes it.
- `run_prototype.sh` reruns the baseline and prototype versions of the existing
  `test_store_restore_ordering.py`, `test_eviction_side_branch_structure.py`,
  and `test_kv_transfer_emission.py`, then runs the four new fixture tests and
  compares OFF-mode GraphBatch JSON byte-for-byte. It sources
  `/workspace/cloud-setup/activate.sh` and both copied project roots use the
  official generated protobuf file from
  `astra-sim-wscllm-LRU/work/chakra-python-root`.
- Actual test results: baseline and prototype each passed **10 + 13 + 18**
  existing tests; the new fixture passed **5/5**; Python `py_compile` passed;
  `git apply --check` passed. OFF baseline and prototype JSON had identical
  SHA-256 `64945939abc1ef398486bf8823f87e1770a8ddae29fc4ad41e63790e46a099d0`.

The fixture generates actual builder nodes and transfer edges; it does not
fake workspace capacity, edit the KV manager, or alter admission policy. It
does not claim a C++ GraphBatchCommitter/NodeStore replay, full Online service
run, workspace-floor correctness, or production acceptance. Those remain
separate integration checks before required-mode activation.
