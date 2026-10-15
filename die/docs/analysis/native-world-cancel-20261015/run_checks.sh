#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
REPO_ROOT=$(git -C "${SCRIPT_DIR}" rev-parse --show-toplevel)
AGGREGATE=${REPO_ROOT}/die/docs/analysis/native-local-projection-20261014/online-reproduction/aggregate
BUILD_DIR=${BUILD_DIR:-${REPO_ROOT}/work/build-native-world-cancel-20261015}
mkdir -p "${BUILD_DIR}"
BUILD_DIR=$(cd -- "${BUILD_DIR}" && pwd -P)
PROTOBUF_PREFIX=${PROTOBUF_PREFIX:-${REPO_ROOT}/work/protobuf-root}
if [[ -n "${CMAKE_BIN:-}" ]]; then
  CMAKE=${CMAKE_BIN}
elif [[ -x "${REPO_ROOT}/work/tool-env/bin/cmake" ]]; then
  CMAKE=${REPO_ROOT}/work/tool-env/bin/cmake
else
  CMAKE=cmake
fi
if [[ -n "${CTEST_BIN:-}" ]]; then
  CTEST=${CTEST_BIN}
elif [[ -x "${REPO_ROOT}/work/tool-env/bin/ctest" ]]; then
  CTEST=${REPO_ROOT}/work/tool-env/bin/ctest
else
  CTEST=ctest
fi

protobuf_args=()
if [[ -d "${PROTOBUF_PREFIX}/usr/include/google/protobuf" ]]; then
  protobuf_args+=(
    "-DProtobuf_INCLUDE_DIR=${PROTOBUF_PREFIX}/usr/include"
    "-DProtobuf_LIBRARY=${PROTOBUF_PREFIX}/usr/lib/x86_64-linux-gnu/libprotobuf.so"
    "-DProtobuf_PROTOC_EXECUTABLE=${PROTOBUF_PREFIX}/usr/bin/protoc"
    "-DProtobuf_PROTOC_LIBRARY=${PROTOBUF_PREFIX}/usr/lib/x86_64-linux-gnu/libprotoc.so"
  )
  export LD_LIBRARY_PATH="${PROTOBUF_PREFIX}/usr/lib/x86_64-linux-gnu:${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-/lib/x86_64-linux-gnu}"
else
  export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-/lib/x86_64-linux-gnu}"
fi

"${CMAKE}" -S "${AGGREGATE}" -B "${BUILD_DIR}" \
  -DREPOSITORY_ROOT="${REPO_ROOT}" \
  -DNETWORK_BACKEND_BUILD_AS_LIBRARY=ON \
  -DCMAKE_BUILD_TYPE=Release \
  "${protobuf_args[@]}"

"${CMAKE}" --build "${BUILD_DIR}" -j "${BUILD_JOBS:-2}" --target \
  NativeCoarseDieDomainTest NativeCoarseProjectionTest \
  NativeProjectionCApiTest JointPipelineWindowTest \
  joint_projection die_model_cabi coarse_die_cabi

"${CTEST}" --test-dir "${BUILD_DIR}" --output-on-failure \
  -R '^(NativeCoarseDieDomainTest|NativeCoarseProjectionTest|NativeProjectionCApiTest|JointPipelineWindowTest)$'

export NATIVE_COARSE_PROJECTION_LIBRARY="${BUILD_DIR}/lib/libjoint_projection.so"
export DIE_COARSE_CABI_PATH="${BUILD_DIR}/lib/libcoarse_die_cabi.so"
export PYTHONPATH="${REPO_ROOT}/astra-sim-joint/sh_test_mesh/workload/llama2_7b_inference"
export PYTHONDONTWRITEBYTECODE=1
python3 -m unittest joint.test_native_coarse_projection -v
