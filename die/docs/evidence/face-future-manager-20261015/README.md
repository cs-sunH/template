# FACE future workspace manager checkpoint — 2026-10-15

Scope: opt-in immutable per-rank future workspace floor in the two FACE KV managers. Default construction preserves the active-only protocol and existing KV snapshots/journal. Explicit floors use capacity snapshot protocol 2, fit uses max(actual, future), and a request exceeding empty-instance capacity after the fixed floor is infeasible before reclaiming cold KV.

Real exact-source Python checks used generated protobuf and both actual C ABIs: FACE 39/39 and FACE-LRU 34/34, zero skips. Modules: online.test_future_workspace_manager, online.test_workspace_capacity_manager, test_face_scheduler, test_face_kv_incremental_invariants. Commands, all staged source hashes, schemas and library identities are preserved per variant. `verified-source.json` binds the two files delivered by this checkpoint to those measured sources.

This additive manager checkpoint does not enable future floors in production. C++ protocol-2 parsing, finite-manifest C++ capacity certificates, service startup and physical source-release gates remain in progress and are not included in this commit. The unchanged default path does not emit protocol 2. No Online pressure acceptance, model calibration or E5 accuracy claim is made here.

The earlier test-fixture correction (events_since returns a list) changed only the new test assertion; the raw failed run remains in work/face-workspace-port-20261014/future-manager-*.log. The final logs here are the passing rerun. Reproduction uses the tracked analysis/face-workspace-20261015/run_python_checks.py entry point with explicit actual ABI paths.
