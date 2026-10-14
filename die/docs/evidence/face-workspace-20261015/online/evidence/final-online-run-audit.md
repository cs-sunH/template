# FACE Online reproduction audit

This work-only checkpoint contains six successful Online runs: one request per legacy/coarse profile for `astra-sim-face` and `astra-sim-face-LRU`, plus one trace-guided coarse workspace-overlap run for each variant. The runtime used the final batch-ID-gated Online binaries supplied by the C++ build agent. It did not rebuild or modify tracked C++ sources.

## Inputs and identity

The tiny baseline uses one real FACE queue row (prefill 4, decode 2), the matching materialized plan/manifest/model, and the legal 54-rank 9x6 topology. Legacy resolves to `mesh_die_v1` with 64,000,000,000-byte capacity. Coarse resolves independently to the actual `coarse-die` hardware descriptor, `compute-cost-model=die`, `die-model-profile=coarse_die_v2`, and 144,000,000,000-byte capacity from the coarse C ABI. FACE-LRU also passes the materializer-produced `remote_memory.json`; FACE does not receive a fabricated remote-memory config.

Each variant's real `sh_test_mesh` source stage and proto are copied into `work/`; `protoc 3.21.12` generates the staged `et_def_pb2.py`. The staged proto SHA-256 is `46b67392c6192561c146c14718cae5336c644d43cf5f96a1466028a89f3ab890`, and generated pb2 SHA-256 is `4608144d0aab16cde4d259e8ff56c0c7b947e181c722f7b8a50d07441487ab1b`. Exact copied-file hashes and C ABI identities are recorded in `inputs/{face,face-lru}/source-stage-identity.json` and copied into each run.

All runs use source HEAD `ca604f5e49dc6df387888a33c75b0134642d42f3`. The per-variant C++ source-tree hash and exact Online binary hash are recorded before and after every run; all six runs report unchanged source and binary identities during execution:

| Variant | C++ source-tree SHA-256 | Online binary SHA-256 |
| --- | --- | --- |
| `astra-sim-face` | `483399503ebf0b370c7847933f0664bbef729eff7747a02f61bf1017069ecaa5` | `6f9670f3fa2b615b70ef9f86ef6a93f7fe467dfec0809d2dfd6dcde977e9cfe6` |
| `astra-sim-face-LRU` | `679fe05f94de88fba9d7706185f4b3be46fde79477f6d5de84af7d21b58bca7c` | `e266ae04017a9be935f17ffde13996f6eb98189ee66afd0d3a733fef41657323` |

## Successful runs

All four tiny baselines completed three real delivery/ACK exchanges, completed their one input request, and ended with 186 successful terminals (phase-5 `total_nodes=186`; no skipped or other terminals). The two coarse baselines carry the expected disclosure: `composition_profile=face_legacy_selector`, `approximate_predictor=true`, `local_plan_profile=coarse_die_v2`, `representative_tp_rank=0`, `collectives_included=false`, `explicit_queue_delay_included=false`, `d2d_cost_profile=fixed_latency_addend`.

| Variant / profile | Run artifact | Deliveries / ACKs | Terminals |
| --- | --- | ---: | ---: |
| FACE legacy tiny | `runs/astra-sim-face/legacy-tiny-20261014T221325Z-55044/` | 3 / 3 | 186 successful |
| FACE coarse tiny | `runs/astra-sim-face/coarse-tiny-20261014T221328Z-55091/` | 3 / 3 | 186 successful |
| FACE coarse overlap | `runs/astra-sim-face/coarse-workspace-overlap-20261014T221353Z-55442/` | 6 / 6 | 378 successful |
| FACE-LRU legacy tiny | `runs/astra-sim-face-LRU/legacy-tiny-20261014T220951Z-53033/` | 3 / 3 | 186 successful |
| FACE-LRU coarse tiny | `runs/astra-sim-face-LRU/coarse-tiny-20261014T220957Z-53087/` | 3 / 3 | 186 successful |
| FACE-LRU coarse overlap | `runs/astra-sim-face-LRU/coarse-workspace-overlap-20261014T221401Z-55514/` | 6 / 6 | 378 successful |

Each run directory contains `inputs/run-input-identity.json` and `inputs/completion-identity.json`, the exact queue/config/manifest/model inputs, C++/Python logs, request journal, Online decision log and statistics. For overlap runs, trace-only `system.json` is recorded and hashed, and debug logs are kept in `trace-logs/`.

## Workspace-overlap evidence

Both overlap inputs contain two real sessions: request 0 has P32/D2 and starts at 0 ns; request 1 has P4/D2 and arrives at 150,000 ns. Native `trace-enabled` issue/callback logs show request 0's first `all_layers_ffn_region` issued at tick 122,710 and completed at tick 238,744 on ranks 20, 21, 26, 27, 32 and 33. At request 1's actual `ARRIVAL` journal boundary (tick 150,000), both variants report those six ranks holding 16,384 bytes each, for 98,304 active workspace bytes total; request 0's prefill had not completed. This is a positive live-workspace boundary sample, not a capacity-pressure or all-scenarios claim.

Earlier probes remain preserved with their original configs and logs: FACE coarse at 1,000 ns (`runs/astra-sim-face/coarse-workspace-overlap-20261014T221335Z-55163/`) and FACE-LRU coarse at 1,000 ns and 20,000 ns (`runs/astra-sim-face-LRU/coarse-workspace-overlap-20261014T221002Z-53180/` and `runs/astra-sim-face-LRU/coarse-workspace-overlap-20261014T221203Z-54325/`) completed simulation but sampled zero active workspace at the second-arrival boundary. The native FFN timing justified moving the arrival to 150,000 ns. These probes are not counted as positive overlap results.

## Preserved setup and protocol failures

- The pre-fix coarse-loader evidence is retained under `evidence/coarse-loader-pre-fix/`; it records the original typed-hardware-to-FACE loader failure. The successful runs above use the refreshed Python stage and actual `coarse_die_v2` materializer output.
- The first coarse ACK mismatch run remains at `runs/astra-sim-face/coarse-20261014T215210Z-45606/`. The successful runs above use the C++ parser/ACK-fixed binaries with required v2 batch identity.
- The first FACE-LRU CLI attempt remains at `runs/astra-sim-face-LRU/legacy-tiny-20261014T220808Z-51995/`; it failed because the runner omitted LRU's required remote-memory option. The work-only runner now passes the actual materializer-produced `remote_memory.json` for LRU only, and the subsequent legacy/coarse runs passed.

## Terminal-count comparison

The final FACE legacy and coarse single-request runs both have 186 terminals and 186 phase-5 nodes. The previously cited 192-terminal run belongs to `work/joint-reproduction/runs/legacy-20261014T161352Z-10833/cpp.log` (`repo_variant=astra-sim-joint`), not to FACE. FACE's existing audit records the different second-delivery sizes (FACE 90, JOINT 96) and explains why the removed successful response IR does not support a node-kind attribution: `evidence/terminal-count-audit.md`.

## Reproduction commands

From the repository root, after activating the supplied runtime environment:

```bash
source /workspace/cloud-setup/activate.sh
bash work/face-workspace-port-20261014/online-reproduction/run_tiny_online.sh face legacy /workspace/template/work/face-workspace-port-20261014/cpp-face/build/bin/AstraSim_Analytical_Congestion_Aware_Online
bash work/face-workspace-port-20261014/online-reproduction/run_tiny_online.sh face coarse /workspace/template/work/face-workspace-port-20261014/cpp-face/build/bin/AstraSim_Analytical_Congestion_Aware_Online
FACE_TRACE_ENABLED=1 bash work/face-workspace-port-20261014/online-reproduction/run_tiny_online.sh face coarse /workspace/template/work/face-workspace-port-20261014/cpp-face/build/bin/AstraSim_Analytical_Congestion_Aware_Online workspace-overlap 150000
bash work/face-workspace-port-20261014/online-reproduction/run_tiny_online.sh face-lru legacy /workspace/template/work/face-workspace-port-20261014/cpp-face-lru/build/bin/AstraSim_Analytical_Congestion_Aware_Online
bash work/face-workspace-port-20261014/online-reproduction/run_tiny_online.sh face-lru coarse /workspace/template/work/face-workspace-port-20261014/cpp-face-lru/build/bin/AstraSim_Analytical_Congestion_Aware_Online
FACE_TRACE_ENABLED=1 bash work/face-workspace-port-20261014/online-reproduction/run_tiny_online.sh face-lru coarse /workspace/template/work/face-workspace-port-20261014/cpp-face-lru/build/bin/AstraSim_Analytical_Congestion_Aware_Online workspace-overlap 150000
```

For trace-guided overlap reproduction, prefix the overlap command with `FACE_TRACE_ENABLED=1`. The completed successful overlap artifacts above used this setting; regular tiny baselines leave it off.
