# Reproduce FACE fixed workspace checks

See ../../evidence/face-future-workspace-20261015/README.md for exact scope, results and the open Runtime failure. The production modules and tests live in each FACE repository; the thin client reuses the already validated WS-LRU client and records its source SHA.

Activate the environment, build the two actual C ABIs and matching Online binary for the selected source variant. The previous face-workspace-20261015/aggregate/CMakeLists.txt remains the build entry point; pass ASTRA_SIM_ROOT explicitly. Use ../face-workspace-20261015/run_python_checks.py with explicit absolute --legacy-library/--coarse-library paths. Final module lists, source hashes, ABI identities and commands are preserved in the evidence Python identities. The new modules are online.test_future_workspace_manager, online.test_manifest_workspace_envelope and online.test_source_release_capacity; LRU also exercises existing store/restore, frontier and eviction suites.

Copy online-reproduction/ or pressure-reproduction/ into an isolated repository work/ subdirectory before execution: the helpers create stage/inputs/runs beside themselves. Set DIE_MODEL_CABI_PATH and DIE_COARSE_CABI_PATH explicitly to actual built libraries and pass the matching Online binary as argument 3. The original fallback paths are historical conveniences, not packaged build dependencies.

Run online-reproduction/run_tiny_online.sh face|face-lru legacy|coarse ONLINE_BIN [tiny|workspace-overlap] [150000]. Coarse overlap uses workspace-overlap 150000. The cap/split case uses SH_TRAIN_MAX_ITER=2 SH_FIRST_TOKEN_SPLIT=1 and tiny. Keep variants isolated during staging; cases for one variant run sequentially.

The pressure helper additionally accepts capacity-pressure as scenario. It uses real loader/materializer/resolver and a large synthetic model on the unchanged SKU. Both observed runs failed in canonical Runtime, so this is currently a reproducer, not a passing benchmark. For a minimal replay independent of app staging:

    python replay_cabi_task.py --library ABS_LIB --hardware ../../evidence/face-future-workspace-20261015/runtime-reproducer/hardware.json --task ../../evidence/face-future-workspace-20261015/runtime-reproducer/minimal-logits-spec.json --output ABS_WORK/response.json

Resolve evidence paths from the repository root or use absolute paths. With the measured old library this exits 1 and reports status 11. It must succeed through the same shared Runtime after repair; do not widen credit tolerances to turn the error into a pass.

Raw archives preserve their measured bytes, including CRLF CSVs and shell invocation trailing spaces. Large raw files use deterministic gzip; RAW_ARTIFACTS.json records both uncompressed and stored SHA256/size. Exclude raw evidence from source whitespace linting. No build tree, binary, dependency cache, FIFO or generated stage is shipped.
