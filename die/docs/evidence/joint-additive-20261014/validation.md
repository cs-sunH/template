# JOINT coarse adapter validation checkpoint

This is scratch validation evidence for the additive JOINT adapter/runtime seam. It is not a production business-parity claim.

## Source/runtime identity

- `python3 die/tools/sync_repositories.py --check` passed for all five variants, 23 canonical shared files, digest `9d025e25699e44b708997da5e4bc9c45e60c75b02166be931282f20921c0c4f4`.
- Generated JOINT's ignored Python Chakra module with the configured `protoc` 3.21.12 using the source-relative JOINT paths. Generated module SHA-256: `4608144d0aab16cde4d259e8ff56c0c7b947e181c722f7b8a50d07441487ab1b`.
- JOINT aggregate `libcoarse_die_cabi.so` SHA-256: `c678d268a7f9e624d5389d5a11a830258f5b2a6bdbc4d6157c2a647cf231b0f1` (matches the actual artifact hash, not inferred from another variant). JOINT `libdie_model_cabi.so` SHA-256: `fe16553813ac00874dfc5e1d0ea098236ea4fc06d1707567926b57b58137684b`.
- Runtime-test source SHA-256: `c353a90c47ae96b53d3458e52cded3955a62235dee5a92f6a292e283234e0514`.
- Runtime-test executable SHA-256: `111b115de6632457b9cae6eb60319d86784dfdc150958c29fe653f8bf23ab1b4`.
- Other focused test source/executable identities are listed in `sha256sums.txt`.

The official ignored Python protobuf was generated with:

```sh
source /workspace/template/work/activate.sh
PROTO_DIR="$PWD/astra-sim-joint/extern/graph_frontend/chakra/schema/protobuf"
PY_OUT="$PWD/astra-sim-joint/work/chakra-python-root/extern/graph_frontend/chakra/schema/protobuf"
mkdir -p "$PY_OUT"
protoc --proto_path="$PROTO_DIR" --python_out="$PY_OUT" "$PROTO_DIR/et_def.proto"
```

## Exact commands and outcomes

The aggregate build was configured with:

```sh
source /workspace/template/work/activate.sh
cmake -S /workspace/template/work/integration-aggregate \
  -B /workspace/template/work/build-joint \
  -DASTRA_SIM_ROOT=/workspace/template/astra-sim-joint
```

Then the production sources, focused test executables, and both JOINT ABI libraries were built with:

```sh
source /workspace/template/work/activate.sh
cmake --build /workspace/template/work/build-joint --target \
  AstraSim_Analytical_Congestion_Aware_CoarseWorkspaceRuntimeTest \
  AstraSim_Analytical_Congestion_Aware_GraphBatchCommitterTest \
  AstraSim_Analytical_Congestion_Aware_ParsedBatchTest \
  AstraSim_Analytical_Congestion_Aware_WorkLedgerBridgeTest \
  AstraSim_Analytical_Congestion_Aware_DieModelAdapterTest \
  AstraSim_Analytical_Congestion_Aware_DieExecutionLifecycleTest \
  coarse_die_cabi die_model_cabi -j4
```

The build completed successfully. Its full output is in `cpp-build.log`. The six test executables were run directly; each returned zero and reported `ALL PASS`. Full command/output transcript: `cpp-tests.log`. Coverage includes unchanged v3 legacy request bytes, required coarse snapshots, atomic invalid-snapshot rejection/revision reissue, workspace lease acquire/block/retry/release and cancellation, shared COMP/endpoint Runtime lifetime, and legacy adapter/lifecycle behavior.

The actual JOINT Python builder/client/workspace suites were run against both JOINT ABI libraries and the generated protobuf module:

```sh
source /workspace/template/work/activate.sh
cd /workspace/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference
PYTHONPATH="/workspace/template/astra-sim-joint/sh_test_mesh:/workspace/template/astra-sim-joint/work/chakra-python-root:$PYTHONPATH" \
  DIE_MODEL_CABI_PATH=/workspace/template/work/build-joint/lib/libdie_model_cabi.so \
  DIE_COARSE_CABI_PATH=/workspace/template/work/build-joint/lib/libcoarse_die_cabi.so \
  python -m pytest -q -rs \
    online/test_die_model_adapter.py \
    online/test_coarse_die_adapter.py \
    online/test_coarse_die_workspace.py
```

Result: **45 passed, zero skipped**: 31 legacy DieModelAdapter tests plus 14 coarse builder/workspace tests. Output: `python-focused.log`.

As a direct ABI smoke, the JOINT `build_coarse_die_task` builder emitted every one of its 14 advertised `op_kind` values and submitted the batch through JOINT's real `libcoarse_die_cabi.so`:

```sh
source /workspace/template/work/activate.sh
PYTHONPATH="/workspace/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference/online:/workspace/template/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference:/workspace/template/astra-sim-joint/work/chakra-python-root:$PYTHONPATH" \
  DIE_COARSE_CABI_PATH=/workspace/template/work/build-joint/lib/libcoarse_die_cabi.so \
  python /workspace/template/work/joint-adapter/real_coarse_14_ops.py
```

Result: **14 estimates, zero errors**. The generated protobuf module imported and resolved `NodeType.COMP_NODE`. Exact task/ABI output: `real-coarse-14-ops.log`; runnable script: `real_coarse_14_ops.py`. These estimates check task parsing and ABI reachability only; they do not validate native-projection parity or workload correctness.

## Initial failures retained

- The earlier whole-build attempt failed because Chakra's `et_def.pb.h` was absent from the old source glob. The preserved original log is `initial-build-failure.log` (scratch original: `work/joint-build-baseline.log`). The JOINT CMake change now generates protobuf C++ into the build tree and the final build above passes.
- The first runtime fixture used `fusion_policy="none"`; the compiler selected a fused attention Plan with zero workspace, so the positive lease fixture failed its `peak > 0` assumption. The test now explicitly requests `materialized`, a valid Plan with scratch; the lease tests pass without weakening assertions.
- Early focused-test compiles caught missing `AstraSim::WorkspaceRankState` namespace imports in parser/bridge test sources. The imports are fixed and the final tests compile/pass.
- The first frozen-wire assertion encoded object keys in construction order, while `nlohmann::json::dump()` emits this project's canonical map order. The expected bytes were corrected to the exact original v3 payload order. Legacy fields and values were unchanged; the exact byte assertion now passes.

## Remaining integration gate

The frozen native projection guard remains enabled. JOINT's sh30 coarse task builder still cannot be reached by the production scheduler until real native-world parity is established. Therefore these checks establish an additive typed-task/ABI/runtime/admission seam and legacy bridge compatibility, not a completed production sh30 coarse integration.
