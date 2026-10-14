# LRU coarse workspace liveness pressure: read-only audit and test proposal

Recorded 2026-10-14 against source identity `source-identity/metrics-port/` (HEAD `e2dd5af7b61895a85b122c200606017bb741a004`, branch `codex/die-lru-workspace-v2-20261014`). No tracked source or tests were changed while preparing this note.

## Observed control flow

`astra-sim-wscllm-LRU/astra-sim/workload/Workload.cc:399-451` first checks hardware availability, prepares the immutable coarse Plan, and calls workspace admission. On `AcquireStatus::Blocked`, it returns before issuing/taking the node; the node stays in the dep-free set, with no GPU occupancy and no lease. The comment names the two retry causes: lease release or changed capacity snapshot.

The completion hook schedules the per-rank deferred issue pass after terminal events (`main_online.cc:1537-1565`); this is the retry path after a lease holder terminates. A schema-v2 commit installs the validated capacity snapshot and merges changed ranks with touched ranks for an issue pass (`GraphBatchCommitter.cc:1981-2033`); this retries a still-free node when KV/reservation capacity changes.

When the EventQueue has no main or deferred events, the Online loop processes pending mailbox work, then a finished service, then checks closed input. If input is closed while the service is unfinished, it dumps a read-only dead-end census and calls `online_fatal("lost-wakeup dead end...")` (`main_online.cc:1925-2050`). Only the open-input branch calls `wait_for_work[_until]`. Therefore a closed finite run does not wait indefinitely in this exact shape: it fails loudly. An open IDLE producer may still provide later work and is meant to park.

The current dead-end census (`main_online.cc:551-640`) shows unfinished nodes, unresolved-parent counts, `issued`, GPU occupancy, endpoint joins, HBM job state, active DIE executions and unfired watches. It does not print the workspace ledger's active bytes/capacity/revision, workspace demand for a dep-free blocked COMP, or whether that node is dep-free. A pressure dead end could therefore be reported as a generic lost wakeup without naming workspace admission as the reason.

## Narrow pressure reproduction proposal

The best starting point is lifecycle Case K in `astra-sim-wscllm-LRU/astra-sim/workload/execution_driven/tests/die_execution_lifecycle_test.cc:946-1060`, which already compiles a positive-workspace `ffn_region`, installs a capacity snapshot that leaves one byte less than demand, and verifies the node remains free with no lease/GPU occupancy. Case H at lines 788-818 is only a partially served marker teardown case; its useful reusable pattern is the harmless `no_op_event` scheduled at tick 101 and drained with `EventQueue::proceed()`.

A follow-up unit case can reuse Case K's blocked Plan and Case H's no-op event pattern: after the blocked issue pass, enqueue/drain the no-op event, then assert the COMP is still in the free set, GPU occupancy is zero, lease count and active bytes are zero, `EventQueue::finished()` is true, and no deferred work remains. This isolates the exact ready-Plan-blocked + GPU-free + event-queue-empty state.

The unit test should not call the production `online_fatal` path in-process because it aborts. If a branch-level assertion is needed, it can validate the existing close/unfinished/quiescent classification separately or use a subprocess harness. No retry should be added solely on finding the node blocked: the production retry sources remain lease completion and an authoritative changed/touched-rank commit. Any later policy should be forecast-driven rather than a blind repeated issue pass.

This focused unit may use a ledger capacity snapshot to create pressure. The official Online E2E used no capacity override and remains separate evidence.
