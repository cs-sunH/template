# Final-source wscllm Online E2E baseline

Run date: 2026-10-14 UTC. This fresh run used branch `codex/die-os-lifecycle-20261014` at source commit `fba3b8e544b65728be384b43ecfeb355fddd028f` (commit tree `2138882858cb90f00806f0b77713164a56c8c37e`). Outputs, adapted scripts, logs, generated configs, raw `strace`, protocol captures, and reports are confined to `work/online-baseline-20261014/`.

## Build and focused verification

The public Die library and aggregate wscllm Online executable were built in independent paths. Public CABI and aggregate CABI SHA-256 values match (`49e596bc661b410ac3ef7499fc7f68af4065e3946be2dbd2b36d224f859ef3cd`). The Online executable SHA-256 is `a35d0ef0dcf1fd039cdd2534b4f08a23f9fa37cc07ceef5e58cb3919358719ec`. Source file identities from committed HEAD are in `source-identity.json`; selected-source manifest SHA-256 is `728e4e069d253f94a33b39886e0355e6aa9608882b0e61d4f070b19f22af40ca`.
The shared working tree acquired later concurrent edits after these artifacts were built: the public Mapping object is timestamped 07:14 UTC, the Online binary 07:18 UTC, while current `die/src/Mapping.cc` and `die/CMakeLists.txt` mtimes are 07:21 and 07:22 UTC. The E2E used the already-built, hash-recorded binaries.

The initial aggregate configure records a failed config-mode lookup when `PROTOBUF_FROM_SOURCE=True` was set. The activated Cloud environment installs protoc/libprotobuf 3.21.12 in its sysroot, without a protobuf CMake config package. Reconfiguring with that variable unset uses CMake's installed `FindProtobuf` module and succeeds. Python protobuf remained at the activated environment's actual version, 6.33.5.

The Online and four teardown/lifecycle targets built successfully. Focused CTest passed 4/4: lifecycle, adapter, local HBM memory, and local HBM bandwidth. Exact commands and complete output are retained in `commands.txt` and `logs/`.

## Production Online run

The fixture was freshly generated through the official trace materializer, config loader, and plan materializer. It contains one request (prefill 64, decode 1) on two ranks: prefill on rank 0 and decode on rank 1. Workspace admission was `auto`; first-token split was enabled for this supported single-token decode fixture.

The final runner exited 0 with C++ and Python exit 0. The strace-based wire audit reports schema valid, no protocol errors, matched queue provenance, 3 requests / 3 responses / 3 successful schema-2 acknowledgments, and an observed `REQUEST_COMPLETE`. All response capacity checks fit. The responses contain 20 typed COMP specs, and all 20 recompiled with the independent public C ABI; the two positive-workspace plans are a 65,536-byte rank-0 prefill FFN and a 1,024-byte rank-1 decode FFN.

Workspace revisions start at 0 on both ranks. After prefill, rank 0 is at revision 20 with zero active workspace. At request completion, both ranks are at revision 20 with zero active workspace. Per-rank capacity snapshots report 144,000,000,000 bytes; all captured responses satisfy model weights + resident KV + reserved request + active workspace <= capacity.

The request completed with first token at 36,185 ns, prefill 25,294 ns, decode 10,890 ns, and E2E completion at 36,185 ns. The aggregate summary reports one unique completed request, zero incomplete requests, p50/p95/p99 36,185 ns, and simulation end 36,186 ns. The final Online audit reports accepted=1, completed=1, active=0, pending alarm=0, and 31 successful terminals with zero skipped/other terminals and no residual watches or issue passes. The marker DIE category recorded exactly 2 read bytes total and zero typed INT4/vector work; rank 0 recorded zero DIE bytes, rank 1 recorded 2 read bytes.

## Evidence paths

- `result.json` — machine-readable source, binaries, protocol, workspace, and metrics summary.
- `source-identity.json` — commit/tree hashes and selected source content hashes from committed HEAD.
- `e2e/run/provenance.txt` — runtime binary/source/config hashes and child exit statuses.
- `e2e/run/wire_audit/audit.json` and `compiled_plans.json` — protocol/workspace checks and replay compilation.
- `e2e/run/cpp.log`, `python.strace`, `cpp.strace`, `wire_audit/` — raw runtime and captured wire records.
- `logs/` — CMake configure/build, focused CTest, fixture preparation, and wrapper logs.

The previous run directory was preserved by the wrapper at `e2e/run-attempt-20261014T072054Z/`; the final validated run is `e2e/run/`.
