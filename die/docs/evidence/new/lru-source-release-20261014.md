# LRU source-release gates evidence — 2026-10-14

Base repository revision: `ad865f11b6466dcc1d07a75be442740b8db20d01`.
The hashes below identify the current working-tree source snapshot; no commit
was created. The focused test runs recorded here predate the one-time
certificate-file persistence added to `online_service.py`; they validate the
capacity derivation and response path, not service startup's file write.

| File | SHA-256 |
|---|---|
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/generate_wsc_llm_trace.py` | `9229133144078ef41ee486d1e954f9a201b646639a085c926a76b040ed2b39dd` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/graph_batch_builder.py` | `6c5a24eb0342de77ab8f5c3e3863df355bd7d91489d8b00cc91554624bce58b0` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/online_service.py` | `3b1651bc5e45f6bce4a91d2e76697493c1c31e6aa86e1338ba72ebc282aedbe0` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/test_source_release_capacity_gate.py` | `3c5024954069477f1640ce52cc0cb62e248a70e1184f5420a8bb0258912cdac3` |
| `astra-sim-wscllm-LRU/astra-sim/workload/execution_driven/tests/graph_batch_committer_test.cc` | `b821d15b23e984890321003654d1adb38c13eed6729bb054fe765e6a5fcbfd51` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/manifest_workspace_envelope.py` | `2a0fadbf413a6014944b282ad73bad22c45f7ea83659f71e33b8e44a8182d24d` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/die_workspace_envelope.py` | `50b84b58faf3b6265d720d022b816e4d07a218f83170b798e189c9cf76f96b74` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/wsc_llm_online_scheduler.py` | `22087e082ba16efc023c7a9ff8613b179a964f2f1110a9735a4263f529beeb3e` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/session_kv_manager.py` | `73291b4dded31be361d853ce3e28376fa724bfe90577b77826c56154078809f4` |
| `astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference/online/test_manifest_workspace_envelope.py` | `e1d7cf8752352803b3098f59b1c64c83cc5a09e5dd19ebef81b4e8b5849381f2` |
| `astra-sim-wscllm-LRU/astra-sim/workload/WorkspaceAdmission.hh` | `9cc9e0a8df794f48a8bce0bb61e1219b5821fd52fd3d8fb4ed02cebfc14fd125` |
| `astra-sim-wscllm-LRU/astra-sim/workload/execution_driven/tests/workspace_admission_test.cc` | `c0d51b1cd9036a78cfea51309897ab24888e8e88594dd17cc1e59325257e8ed4` |

Python validation, run from
`astra-sim-wscllm-LRU/sh_test_mesh/workload/llama2_7b_inference` after
`source /workspace/template/work/activate.sh`:

| Command | Result |
|---|---|
| `python3 online/test_source_release_capacity_gate.py` | 7 tests passed |
| `python3 online/test_store_restore_ordering.py` | 10 tests passed |
| `python3 online/test_eviction_side_branch_structure.py` | 13 tests passed |
| `python3 online/test_kv_transfer_emission.py` | 18 tests passed |
| `python3 online/test_transfer_frontier_neutral.py` | 6 tests passed |
| `python3 online/test_decode_eviction_serialization.py` | 10 tests passed |
| `python3 -m pytest -q online/test_manifest_workspace_envelope.py online/test_source_release_capacity_gate.py online/test_workspace_admission_v2.py` | 18 tests passed (2 protobuf deprecation warnings) |
| `python3 -m py_compile generate_wsc_llm_trace.py online/graph_batch_builder.py online/online_service.py online/test_source_release_capacity_gate.py` | passed |

The byte-compatibility check was:

```sh
source /workspace/template/work/activate.sh && python3 online/test_source_release_capacity_gate.py --dump-off > /workspace/template/work/lru-source-release/current-off.json
cmp /workspace/template/work/lru-source-release/current-off.json /workspace/template/die/docs/analysis/workspace-envelope-20261014/research/store-release/baseline-off.json
```

`cmp` passed; the emitted OFF snapshot SHA-256 is
`64945939abc1ef398486bf8823f87e1770a8ddae29fc4ad41e63790e46a099d0`.

The C++ fixture uses the existing target (no new target registration):

```sh
source /workspace/template/work/activate.sh
cmake --build /workspace/template/work/build-lru --target AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest -j2
/workspace/template/work/build-lru/bin/AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest
```

Build succeeded and the binary exited 0 with
`[graph_batch_committer_test] ALL PASS`. Part Q covers P-triggered remote-store
send/recv/ACK nodes, live cross-batch ACK dependencies on a destination HBM
restore and COMP, terminal release, and GC followed by a later edge to the
collected ACK. Its test-local Kahn oracle includes parent edges and matched
send-to-recv edges; a deliberately cyclic P-trigger shape is rejected by this
oracle. The production C++ preflight still checks parent-edge cycles per rank;
the Python source-release test checks the generated combined graph.

The subsequent actual Online run and workspace E integration are recorded in
[`../lru-future-workspace-20261014/README.md`](../lru-future-workspace-20261014/README.md);
this note records focused source and fixture evidence only.
The manifest certificate proof is finite and specific to the request-aggregated
LRU P/D producer and the tested four-request fixture. The recorded tests do not
establish arbitrary producer families, hardware layouts, or a real Online
startup/bridge run with the saved certificate file.
