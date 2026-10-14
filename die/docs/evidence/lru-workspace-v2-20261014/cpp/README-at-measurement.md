# LRU C++ workspace-v2 and metrics implementation

This worktree contains the C++ LRU adapter/system/protocol/workload integration for workspace v2 plus the coarse DIE metric projection and LRU metric compatibility fix. It preserves the existing LRU remote-store/history/anchor paths and emits coarse typed/runtime fields only for coarse Runtime ranks. The coarse `local_hbm` record retains the old `hbm_shared_definition` key using the coarse distinct-owner definition, keeps `shared_definition` as an alias, and restores `local_hbm_restore_bytes_issued` from the original issued-byte metric counter. The non-coarse LRU record retains its original equal-split semantics and key set.

The tracked source remains uncommitted on branch `codex/die-lru-workspace-v2-20261014`, HEAD `e2dd5af7b61895a85b122c200606017bb741a004`. The full source and binary identity is saved at [`source-identity/metrics-alias-fix/`](source-identity/metrics-alias-fix/). Its manifest checks clean against the current workspace. The selected patch includes the pre-existing workspace-v2 WIP plus the metrics changes; the separate initial workspace patch remains at `source-identity/lru-cpp-workspace-v2.patch`.

The current Online executable SHA-256 is `dab727d8b664d2785f86718d36874ad6295b5110618d3e86e4db9cb1c2dbc1ae`. CABI hashes are coarse `d2f06c80e6112f1215f6b124f4cb333041d2c592b84c90a81127e0faeb5ee17b` and die model `7b961c60505ce8e37be3520fe3f363d1cadbe7fb8cf99cced50daa6285690d68`. `MetricCollector.cc` SHA-256 is `686293e14a0fce9e397e69b3e945baa5ae7c3d57f5575d386c83b38588b4c6ef`; the lifecycle focused test SHA-256 is `f4b892a016074a5185b31a117b8a47682dcda6e257ec0dc9bf43fe84e6235bf2`.

## Rebuild and test

Use the activated cloud environment in apt protobuf mode:

```bash
source /workspace/cloud-setup/activate.sh
unset PROTOBUF_FROM_SOURCE
cmake -S work/five-os-20261014/aggregate \
  -B work/lru-cpp-workspace-20261014/build/aggregate \
  -DASTRA_SIM_ROOT=/workspace/template/astra-sim-wscllm-LRU \
  -DBUILDTARGET=congestion_aware \
  -DBUILD_TESTING=ON \
  -DASTRA_SIM_BUILD_OFFLINE_GREEDY_SCHEDULE_TEST=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build work/lru-cpp-workspace-20261014/build/aggregate \
  --parallel 2 \
  --target AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
           AstraSim_Analytical_Congestion_Aware_Online \
           coarse_die_cabi die_model_cabi
ctest --test-dir work/lru-cpp-workspace-20261014/build/aggregate \
  -R '(WorkspaceAdmission|DecisionMailbox|BridgeLoopback|GraphBatchCommitter|ParsedBatch|LocalHbmBandwidthModel|LocalHbmModel|DieModelAdapter|DieExecutionLifecycle)Test' \
  --output-on-failure
```

The final focused run passed 9/9. Full build output is in [`logs/metrics-alias-fix-build.log`](logs/metrics-alias-fix-build.log); exact CTest output is in [`logs/metrics-alias-fix-nine-ctest.log`](logs/metrics-alias-fix-nine-ctest.log). The source identity directory also records the exact command lines in `provenance.txt`.

## Online evidence boundary

The earlier positive-workspace official 8-rank run passed with Online SHA `ccb00ff06e74753a1385a43e0c42f6e1668f7b68bfbd5a4765f8f11dd016d116`: exit 0, three schema-v2 exchanges, 124/124 terminals successful, positive typed workspace on all ranks, and ten balanced lease acquisitions/releases per rank. That run predates the two metric-key compatibility additions; its full source/binary identity is preserved under `source-identity/metrics-port/`.

The final `metrics-alias-fix` binary also passed the official positive-workspace 8-rank E2E and acceptance audit at `work/lru-online-workspace-20261014/e2e-large-workspace-probe/run/`. C++ and Python both exited 0; all three acknowledgements were schema 2; the audit observed 124/124 successful terminals and no protocol errors. The plan audit found positive workspace on all eight participating ranks: 65,536 bytes on P ranks 0–3 and 1,024 bytes on D ranks 4–7. Each rank had ten balanced lease acquisitions/releases and ended with zero active workspace. The LRU metric-schema audit found no missing legacy `local_hbm` keys after the compatibility fix; `hbm_watermark_summary` retains two unmapped baseline-only window bounds because the LRU flat-zero record has no planner replay bounds. The exact run provenance binds to the current Online SHA recorded above.

No pressure retry/census source changes have been made; the read-only liveness proposal is in `liveness-pressure-readonly-proposal.md` pending the parent checkpoint.
