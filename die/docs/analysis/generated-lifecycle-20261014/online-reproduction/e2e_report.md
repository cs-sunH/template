# wscllm workspace Online E2E checkpoint

## Latest result

The production Python↔C++ two-rank Online run passed on 2026-10-14. The
reproducible command was:

```bash
PYTHON=python3 bash work/wscllm-workspace-e2e/run_online_e2e_preserve.sh
```

The wrapper exited 0; the recorded child statuses are `cpp_exit=0` and
`python_exit=0` in `run/exit_codes.txt`. It archived the previous failed
attempt to `run-attempt-20261014T061315Z/` before starting. The older
pre-provenance attempt remains at `run-attempt-01-missing-provenance/`.

This run used the built `AstraSim_Analytical_Congestion_Aware_Online`, the
production `online_service.py`, the official config loader and
`plan_materializer.py`, the official
`derive_20_first_30_seconds.py` request-queue materializer, and
`strace`-captured FIFO bridge writes. Python was 3.12.14 with protobuf 4.25.8.
The Python and integrated C++ coarse C ABI files had the same SHA-256 digest
(`49e596bc661b410ac3ef7499fc7f68af4065e3946be2dbd2b36d224f859ef3cd`). The
only workload experiment enabled was the supported `SH_FIRST_TOKEN_SPLIT=1`
path for this one-token decode request. No Python manager, captured response,
or workspace snapshot was patched or fabricated.

`run/wire_audit/audit.json` reports `protocol_schema_valid=true`, no protocol
errors, matched queue provenance, three successful schema-2 acknowledgments,
and an observed `REQUEST_COMPLETE`. The wire contained three request,
response, and ack exchanges, all with bound delivery sequence and batch IDs.
The sole request was queue index 0, session
`session_workspace_e2e_s0`, request
`session_workspace_e2e_s0_request_0`, with a 64-token prefill and one-token
decode. The Python queue provenance records one row in the official 30-second
window.

The workspace checks cover both ranks and preserve revision echoes. The
ARRIVAL state starts at revision 0 with zero active bytes on ranks 0 and 1.
Each response has full per-rank capacity, model-weight, resident-KV, and
reserved-request fields. The audit recomputes `base_used_bytes` as model
weights plus resident KV plus reserved request bytes, adds active workspace,
and verifies it fits capacity for every rank and response. After prefill,
rank 0 is at revision 20 with zero active workspace; after decode completion,
both ranks are at revision 20 with zero active workspace. This matches ten
typed COMP Plan acquisitions and releases per stage.

The actual production responses contained 20 typed COMP specs. Recompiling
those captured specs with the same public C ABI and resolved hardware
descriptor succeeded for all 20 Plans. Two emitted Plans had positive HBM
workspace: the rank-0 prefill FFN used 65,536 bytes and the rank-1 decode FFN
used 1,024 bytes. Both legal fused region kinds, `attention_region` and
`ffn_region`, were present. Response 1 contained a first-token node and the
two control-marker nodes (join and first-token); response 2 recorded
`DECODE_COMPLETION` and `REQUEST_COMPLETE`, with completed prefill/decode
groups.

The final request metric reports `completed=true`, `first_token_ns=36185`,
`completion_ns=36185`, `prefill_ns=25294`, and `decode_ns=10890`. Runtime
service records show positive typed COMP service on both ranks and complete
accounting history. The rank-1 DIE category records exactly two physical
read bytes from the two control-marker endpoints, with zero DIE int4/vector
ops; COMM traffic remains in its own categories. The Online end audit reports
31 successful terminals, zero skipped/other terminals, no outstanding
watches or issue passes, and service counters `accepted=1 completed=1
active=0 pending_alarm=0`. Metrics recorded a successful consistency audit.

The integrated C++ build used the configured CMake/protobuf environment and
covered Online, DieExecutionLifecycle, LocalHbmDieMemory,
LocalHbmBandwidthModel, `coarse_die_cabi`, and `die_model_cabi`:

```bash
env PATH="$PWD/work/tooling/cmake/cmake/data/bin:$PWD/work/deps/protobuf/bin:$PATH" \
  CMAKE_PREFIX_PATH="$PWD/work/deps/protobuf" PROTOBUF_FROM_SOURCE=True \
  cmake --build work/integration-build/build/astra-sim-wscllm \
    --target AstraSim_Analytical_Congestion_Aware_Online \
    AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
    AstraSim_Analytical_Congestion_Aware_LocalHbmDieMemoryTest \
    AstraSim_Analytical_Congestion_Aware_LocalHbmBandwidthModelTest \
    coarse_die_cabi die_model_cabi -j2
```

It exited 0. The lifecycle, local-HBM-memory, local-HBM-bandwidth, and
DieModelAdapter test binaries also exited 0; the
adapter regression retains the legacy HBM marker completion at issue plus
100 ns plus one drain nanosecond. Lifecycle Case I is a direct-dispatch unit
test of shared Runtime behavior while a typed positive-workspace Plan is live;
it explicitly does not model the production single-Compute availability
gate. Case L uses the normal issue gate to run the marker concurrently with
a real COMM HBM endpoint and confirms the shared Runtime and separate byte
categories.

## Failed attempt retained

The immediately preceding real run completed two schema-2 exchanges, then
aborted when a coarse control marker entered the legacy scalar COMP HBM queue.
Its request did not reach completion. The saved logs, wire records, and
compiled-plan evidence are under
`run-attempt-20261014T061315Z/`; the root cause was fixed with a dedicated
one-byte DIE-owned canonical endpoint carrying the configured initial delay.
The earlier missing-provenance setup failure is preserved separately under
`run-attempt-01-missing-provenance/`.

## Reproduction and limits

Repeat with:

```bash
PYTHON=python3 bash work/wscllm-workspace-e2e/run_online_e2e_preserve.sh
```

The wrapper preserves the current result before each rerun. It performs a
protocol and lifecycle check for this single controlled request. `strace`
adds wall-clock overhead, so the run is not a performance measurement and
does not establish broader throughput or multi-request statistical
behavior.
