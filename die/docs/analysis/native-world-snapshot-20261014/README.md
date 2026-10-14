# Native local-domain and session wire recovery

This extends the fresh-domain bridge in `../native-local-projection-20261014/`.
It adds recovery of an already running **native session with the same frozen
input**, not import of a production Sys/WorkLedger/network/pool world. The
sh30 provider guard and `live_world_import_supported=false` remain. Inspection
advertises the separate `native_session_snapshot_schema`.

## Contract and implementation

`NativeCoarseDieDomain::export_snapshot()` writes
`native_coarse_die_domain_snapshot_v1`: canonical Runtime v3 state, immutable
Plan chains and cursor, Owner/handle bindings, endpoint/transaction registries,
retained terminal progress and the one-shot completion queue.
`import_snapshot(resource_key, hardware, wire)` builds a new domain through
the public Runtime importer. It checks hardware/Plan identity, live-handle
bijection, chain progression, endpoint ownership/refcounts and completion
bindings. Raw Runtime-only snapshots cannot substitute for this wrapper wire.
The existing cancelled-chain inspect convention is preserved: successful
prefixes are in `completed_plan_progress`; the cancelled current Plan remains
in `last_progress`.

`NativeProjectionWorldSession::operation` adds `export_snapshot` and
`import_snapshot`. The `native_coarse_projection_world_snapshot_v1` wire binds
the frozen input and kernel contract, full domain snapshots, ordered native
events and active timers, per-leg state, gate queues/history, barrier arrivals,
clock/counters and session controls. Import constructs a disposable world,
validates it completely, and publishes it only after inspection succeeds.
Work completion and dependency counts are reconstructed from the graph.

One processed-prefix counter distinguishes a consumed forecast from an event
still queued at the current tick after controlled integration or a budget
cut. Remaining forecast events must match the original `(time, sequence)`
suffix exactly. Native deadlines use simple duration/readiness formulas;
ungated starts must equal the graph's readiness floor. No Python timing or
workspace formulas were added. Python forwards the two operations and mirrors
controls; barrier views now read the actual native
`signallers`/`passed_at_ns` fields and retain the older aliases.

Three reachable scheduling defects exposed by recovery tests are fixed only
in coarse worlds: a fixed timer integrated to zero remaining service still
settles at that tick; release of another acquire gate retries existing waiter
queues; a zero-service predecessor cannot cause initial scanning to activate
its typed child twice. Gate queues retain their saved order. The common gate
fit comparison also avoids unsigned addition overflow.

These are structural compatibility and continuation checks, not authentication
of an arbitrary edited execution history. The wire does not recreate unknown
traffic, network or pool state, external callbacks, workspace leases or
production ledger identities. Domain cancellation is covered; native-world
cancel/capacity event import and production Sys live import remain separate
work.

## Verification and reproduction

With CMake, a C++17 compiler and the repository's protobuf dependencies
available, run `bash die/docs/analysis/native-world-snapshot-20261014/run_checks.sh`.
The runner derives the checkout root, uses the existing aggregate CMake
project, and accepts `BUILD_DIR` and `BUILD_JOBS`. It builds the actual libraries,
runs the four focused CTests and the four Python coarse-wire/facade tests
against explicitly selected libraries, then checks the five source manifests.

Actual final-source checks recorded in the corresponding evidence directory:

- Four focused native/window CTests passed. Domain cases cover fractional
  progress, chain boundaries, shared endpoint transaction ownership,
  pending/consumed/cancelled completions and malformed wire rejection.
- World cases compare an uninterrupted source's next step and drain with
  imported continuations, including initial, partial integration, known
  endpoint arrivals, intermediate chains, gate/barrier states, forecast budget
  prefixes, ready/latency timers and terminals. Negative imports leave both
  inspection and full source snapshot unchanged.
- Python legacy protocol plus coarse facade suite: 18/18, zero skips, with all
  three actual ABI paths set. The legacy protocol test used the real generated
  legacy configuration already materialized by the configuration suite; the
  portable focused runner needs no such generated fixture.
- The final Online executable was relinked and the existing real tiny legacy
  runner completed with C++/Python exit 0, one accepted/completed request,
  three GraphBatch commits, 192/192 successful terminal nodes and zero
  incomplete requests. It uses the same 54-rank tiny topology as the previous
  checkpoint. This is legacy compatibility evidence, not coarse Online
  acceptance. Its ledger report still discloses 48 `stale_amount_works`; no
  full live-ledger acceptance is inferred.
- Canonical23 remains byte-identical across the five active repositories,
  digest `45b7bf3d8df9969b8155323c79eecf1a81f8d1c7499177e811b6aedf2c4c3264`.

Online reproduction reuses
`../native-local-projection-20261014/online-reproduction/run_tiny_online.sh`;
set `JOINT_REPRO_WORKDIR` to an isolated work directory and use `legacy`.
Build `AstraSim_Analytical_Congestion_Aware_Online` and `joint_ledger_cabi`
first. The earlier coarse-intended guard failure is retained in its original
evidence package; the guard source is unchanged and was not bypassed.

No public mapping/timing or calibration source changed in this checkpoint.
E5 precision remains failed, E8/E16 and target measurements remain unavailable,
and no new host-overhead certificate is claimed.
