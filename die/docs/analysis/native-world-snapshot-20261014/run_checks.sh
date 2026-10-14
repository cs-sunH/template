#!/usr/bin/env bash
set -euo pipefail

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(git -C "${HERE}" rev-parse --show-toplevel)
BUILD_DIR=${BUILD_DIR:-${ROOT}/work/build-native-world-snapshot}
AGGREGATE=${HERE}/../native-local-projection-20261014/online-reproduction/aggregate

cmake -S "${AGGREGATE}" -B "${BUILD_DIR}" \
  -DREPOSITORY_ROOT="${ROOT}" -DCMAKE_BUILD_TYPE=Release \
  -DBUILDTARGET=congestion_aware -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON
cmake --build "${BUILD_DIR}" -j "${BUILD_JOBS:-2}" --target \
  NativeCoarseDieDomainTest NativeCoarseProjectionTest \
  NativeProjectionCApiTest JointPipelineWindowTest \
  joint_projection die_model_cabi coarse_die_cabi
ctest --test-dir "${BUILD_DIR}" --output-on-failure \
  -R '^(Native(CoarseDieDomain|CoarseProjection|ProjectionCApi)Test|JointPipelineWindowTest)$'

export DIE_MODEL_CABI_PATH="${BUILD_DIR}/lib/libdie_model_cabi.so"
export DIE_COARSE_CABI_PATH="${BUILD_DIR}/lib/libcoarse_die_cabi.so"
export NATIVE_COARSE_PROJECTION_LIBRARY="${BUILD_DIR}/lib/libjoint_projection.so"
export PYTHONPATH="${ROOT}/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest joint.test_native_coarse_projection -v
python3 "${ROOT}/die/tools/sync_repositories.py" --check
