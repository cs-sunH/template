# JOINT Online reproduction

This package builds the active `astra-sim-joint` checkout and prepares one
small, real Online request without writing generated files into the source
tree. The default build and generated-input locations are under the repository
`work/` directory. Set `BUILD_DIR` or `JOINT_REPRO_WORKDIR` to use another
location.

The `legacy` profile exercises the current C++ Online binary, Python service,
bridge, GraphBatch and NodeStore path. `coarse_intended` reaches the current
coarse provider guard and is expected to exit nonzero. The latter is a
fail-closed integration probe, not a coarse Online success or performance
result. Current `online_service.py` only exposes strategy mode and creates
`Sh30OnlineScheduler`; admission calls `_joint_cost_model` before policy
selection, so face/load-first policy choices also reach this guard. There is
no supported non-sh30 Online route in this source state.

## Build

Run from any directory inside the repository:

```sh
REPO_ROOT=$(git rev-parse --show-toplevel)
ASTRA_SIM_ROOT="$REPO_ROOT/astra-sim-joint"
BUILD_DIR=${BUILD_DIR:-"$REPO_ROOT/work/build-joint"}
REPRO_DIR="$REPO_ROOT/die/docs/analysis/native-local-projection-20261014/online-reproduction"
source /workspace/cloud-setup/activate.sh
cmake -S "$REPRO_DIR/aggregate" \
  -B "$BUILD_DIR" \
  -DASTRA_SIM_ROOT="$ASTRA_SIM_ROOT" \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILDTARGET=congestion_aware \
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build "$BUILD_DIR" --target \
  AstraSim_Analytical_Congestion_Aware_Online \
  joint_projection coarse_die_cabi die_model_cabi joint_ledger_cabi \
  NativeProjectionCApiTest NativeCoarseProjectionTest NativeCoarseDieDomainTest \
  -j2
ctest --test-dir "$BUILD_DIR" --output-on-failure \
  -R 'NativeProjectionCApiTest|NativeCoarseProjectionTest|NativeCoarseDieDomainTest'
```

The aggregate CMake scaffold discovers the repository root from Git, so the
same command works if this package is copied elsewhere inside the checkout.
The JOINT top-level CMake selects its Protobuf config package only when
`PROTOBUF_FROM_SOURCE=True`; with the supplied activation environment that
variable is unset and CMake uses the installed Protobuf module package. C++
Chakra output stays in the build tree. The Python staging helper invokes the
active `protoc` into the isolated stage.

## Focused ABI checks

Both libraries are passed explicitly to the 13-case config resolver suite:

```sh
env DIE_MODEL_CABI_PATH="$BUILD_DIR/lib/libdie_model_cabi.so" \
  DIE_COARSE_CABI_PATH="$BUILD_DIR/lib/libcoarse_die_cabi.so" \
  python3 -m unittest discover \
    -s "$ASTRA_SIM_ROOT/sh_test_mesh/tests" \
    -p 'test_config_resolver*.py' -v
```

The current 45-test JOINT die adapter/workspace set is:

```sh
cd "$ASTRA_SIM_ROOT/sh_test_mesh/workload/llama2_7b_inference"
env DIE_MODEL_CABI_PATH="$BUILD_DIR/lib/libdie_model_cabi.so" \
  DIE_COARSE_CABI_PATH="$BUILD_DIR/lib/libcoarse_die_cabi.so" \
  pytest -q online/test_die_model_adapter.py \
    online/test_coarse_die_adapter.py online/test_coarse_die_workspace.py
```

The native-to-Python coarse projection test loads the actual JOINT projection
and coarse Die ABIs:

```sh
env NATIVE_COARSE_PROJECTION_LIBRARY="$BUILD_DIR/lib/libjoint_projection.so" \
  DIE_COARSE_CABI_PATH="$BUILD_DIR/lib/libcoarse_die_cabi.so" \
  python3 -m unittest joint.test_native_coarse_projection -v
```

## Prepare and run Online

```sh
source /workspace/cloud-setup/activate.sh
BUILD_DIR="$BUILD_DIR" bash \
  "$REPRO_DIR/run_tiny_online.sh" \
  legacy
BUILD_DIR="$BUILD_DIR" bash \
  "$REPRO_DIR/run_tiny_online.sh" \
  coarse_intended
```

The single request has prefill=4, decode=2, one layer, hidden=384, FFN=768,
12 heads, vocab=384, and TP=6. Nine configured instances cover all 54 FACE
ranks so the real resolver does not create empty shards. The helper stages
source/config inputs under `work/joint-reproduction/stage`, writes a request
queue provenance sidecar, and stores each run's raw C++ and Python logs under
`work/joint-reproduction/runs/`. It copies the required JOINT ledger and
projection libraries into the isolated stage so the service imports the
actual build products.

The `coarse_intended` input reuses the valid FACE mesh resolver output and
substitutes the checked-in coarse example descriptor to reach the production
provider guard. It deliberately retains the guard failure and nonzero exit;
it does not monkeypatch the provider or claim a completed coarse Online run.
Captured run logs and binary/source identity are in the sibling evidence
directory `die/docs/evidence/native-local-projection-20261014/online/`.
