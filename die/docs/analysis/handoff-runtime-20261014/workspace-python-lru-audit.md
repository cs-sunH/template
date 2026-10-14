# Read-only LRU workspace bridge audit

Date: 2026-10-14

Scope: read-only comparison of the Python paths in
astra-sim-wscllm-LRU against the no-LRU workspace implementation. No LRU
production or test files were modified.

## Reusable pieces

- The v2 file-bridge shape is independent of the KV replacement policy:
  schema version 2, request delivery_sequence, response
  source_delivery_sequence, and the existing ack fields can be ported. LRU
  currently has no workspace_state, capacity snapshot, or admission-mode
  hooks in its Python bridge/scheduler files.
- Rank validation, absolute revisions, idempotent delivery/ack behavior,
  stale-view rejection, and capacity_epoch wakeups can follow the no-LRU
  bridge contract, once connected to the LRU manager's own invariants.
- Workspace accounting should remain an external fit margin. LRU's remote
  pool account and journal's remote column represent KV stored remotely; they
  must not contain C++ transient workspace bytes.

## LRU behavior that needs independent implementation and tests

- LRU fit uses whole-session reclamation. The candidate set is completed,
  inactive, fully-local sessions for the target instance, ordered by
  (last_completion_ns, session_id); selected sessions are moved to the remote
  pool as remote_store transfers. Its feasibility margin already accounts
  for active/protected local residency and reserved request bytes. A workspace
  hold must be subtracted in this same margin and the existing victim order
  retained. The no-LRU cold-delete test does not validate LRU transfer or
  pool accounting.
- Permanent empty-instance overflow differs: LRU ensure_physical_fit
  raises ValueError; no-LRU returns a permanently_infeasible result. Keep
  the LRU contract unless a separate design decision changes it.
- LRU prepare_history has four paths: no history, local same-instance hit,
  local cross-instance NoC migration, and remote restore. Every path calls
  fit first. A workspace block during remote restore must leave the session
  remote and keep its remote-pool debit unchanged; after a capacity revision
  releases the hold, retry must restore once and debit the exact remote shard
  bytes. Verify the same-instance, migration, and new-history paths also leave
  their source/target state unchanged on a workspace wait.
- LRU transactions snapshot/restore manager ranks, sessions, reservations,
  event tails, remote pool balances, and invariant contribution ledgers. The
  external workspace view must remain external across transaction rollback,
  while invariant checks enforce base KV plus workspace within capacity.
- LRU scheduler adds store/restore transfer emission, pending completion
  gates, and ordering checks. Independently test capacity_epoch restore retry
  through the actual scheduler and ensure blocked restore emits no transfer;
  release retry must satisfy existing store-before-restore ordering and
  consume the restore/pool ledger once. No-LRU admission retry tests do not
  cover these LRU transfer lifetimes.

## Existing LRU tests to extend or run independently

- sh_test_mesh/workload/llama2_7b_inference/test_tiered_eviction_sequence.py:
  whole-session LRU order, protected/active/reservation fit margins, and
  remote restore paths.
- online/test_kv_transfer_emission.py and
  online/test_store_restore_ordering.py: remote-store/restore graph
  transfers, completion gates, and ordering.
- test_kv_delta_journal.py and
  test_wscllm_kv_incremental_invariants.py: pool and journal conservation,
  rollback-relevant manager invariants.
- online/test_admission_net_credit.py: scheduler reservation/retry
  interaction.

Read-only source inspection covered the LRU manager's candidate selection,
ensure_physical_fit, transaction rollback, and prepare_history branches,
plus the listed tests and scheduler capacity_epoch/restore flow.
