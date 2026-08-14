#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SH_TEST_DIR=$(realpath "${SCRIPT_DIR}/..")
PROJECT_DIR=$(realpath "${SH_TEST_DIR}/..")
BUILD_ROOT="${PROJECT_DIR}/build/astra_analytical"
BUILD_DIR="${BUILD_ROOT}/build_congestion_aware"
CHAKRA_ET_DIR="${PROJECT_DIR}/extern/graph_frontend/chakra/schema/protobuf"
CACHE_FILE="${BUILD_DIR}/CMakeCache.txt"
YAML_CPP_SOURCE_DIR="${YAML_CPP_SOURCE_DIR:-}"

BUILD_TARGET=congestion_aware

if [[ $# -ne 0 ]]; then
  echo "This script uses a fixed build target. Edit sh_test_mesh/run_scripts/build_analytical_aware.sh instead of passing arguments." >&2
  exit 1
fi

print_protobuf_install_hint() {
  echo "[sh_test] Missing protobuf build dependency." >&2
  if [[ -f /etc/debian_version ]]; then
    echo "[sh_test] Install with: sudo apt update && sudo apt install -y libprotobuf-dev protobuf-compiler" >&2
  else
    echo "[sh_test] Install the protobuf C++ development package and protoc for your distribution." >&2
  fi
}

validate_protobuf_dependency() {
  local probe_dir
  local probe_build_dir
  probe_dir=$(mktemp -d)
  probe_build_dir="${probe_dir}/build"

  cat > "${probe_dir}/CMakeLists.txt" <<'EOF'
cmake_minimum_required(VERSION 3.22)
project(ProtobufProbe)
if(DEFINED ENV{PROTOBUF_FROM_SOURCE} AND "$ENV{PROTOBUF_FROM_SOURCE}" STREQUAL "True")
  find_package(protobuf CONFIG REQUIRED)
else()
  find_package(Protobuf REQUIRED)
endif()
EOF

  if ! cmake -S "${probe_dir}" -B "${probe_build_dir}" >"${probe_dir}/cmake.log" 2>&1; then
    print_protobuf_install_hint
    echo "[sh_test] CMake protobuf probe failed:" >&2
    sed 's/^/[sh_test]   /' "${probe_dir}/cmake.log" >&2
    rm -rf "${probe_dir:?}"
    exit 1
  fi

  rm -rf "${probe_dir:?}"
}

compile_chakra_et() {
  if [[ ! -f "${CHAKRA_ET_DIR}/et_def.pb.h" || ! -f "${CHAKRA_ET_DIR}/et_def.pb.cc" ]]; then
    if ! command -v protoc >/dev/null 2>&1; then
      print_protobuf_install_hint
      exit 1
    fi
    protoc "${CHAKRA_ET_DIR}/et_def.proto" \
      --proto_path="${CHAKRA_ET_DIR}" \
      --cpp_out="${CHAKRA_ET_DIR}"
  fi

  if [[ ! -f "${CHAKRA_ET_DIR}/et_def_pb2.py" ]]; then
    if ! command -v protoc >/dev/null 2>&1; then
      print_protobuf_install_hint
      exit 1
    fi
    protoc "${CHAKRA_ET_DIR}/et_def.proto" \
      --proto_path="${CHAKRA_ET_DIR}" \
      --python_out="${CHAKRA_ET_DIR}"
  fi
}

reset_build_dir_if_relocated() {
  if [[ ! -f "${CACHE_FILE}" ]]; then
    return
  fi

  local cached_home_dir
  local cached_cache_dir

  cached_home_dir=$(sed -n 's/^CMAKE_HOME_DIRECTORY:INTERNAL=//p' "${CACHE_FILE}" | head -n 1)
  cached_cache_dir=$(sed -n 's/^CMAKE_CACHEFILE_DIR:INTERNAL=//p' "${CACHE_FILE}" | head -n 1)

  if [[ "${cached_home_dir}" == "${BUILD_ROOT}" && "${cached_cache_dir}" == "${BUILD_DIR}" ]]; then
    return
  fi

  echo "[sh_test] Detected relocated repository or stale CMake cache."
  echo "[sh_test] Recreating build directory: ${BUILD_DIR}"
  rm -rf "${BUILD_DIR:?}"
}

resolve_yaml_cpp_source_dir() {
  if [[ -n "${YAML_CPP_SOURCE_DIR}" ]]; then
    if [[ ! -f "${YAML_CPP_SOURCE_DIR}/CMakeLists.txt" ]]; then
      echo "[sh_test] YAML_CPP_SOURCE_DIR is not a yaml-cpp source tree: ${YAML_CPP_SOURCE_DIR}" >&2
      exit 1
    fi
    return
  fi

  local candidate
  for candidate in "${BUILD_ROOT}/build/_deps/yaml-cpp-src"; do
    if [[ -f "${candidate}/CMakeLists.txt" ]]; then
      YAML_CPP_SOURCE_DIR="${candidate}"
      return
    fi
  done
}

NUM_THREADS=$(nproc)
if [[ ${NUM_THREADS} -ge 16 ]]; then
  NUM_THREADS=16
fi

mkdir -p "${BUILD_DIR}"
validate_protobuf_dependency
compile_chakra_et
reset_build_dir_if_relocated
mkdir -p "${BUILD_DIR}"
resolve_yaml_cpp_source_dir

echo "[sh_test] Building ASTRA-sim analytical backend: ${BUILD_TARGET}"
echo "[sh_test] Build directory: ${BUILD_DIR}"
cmake_args=(-DBUILDTARGET="${BUILD_TARGET}")
if [[ -n "${YAML_CPP_SOURCE_DIR}" ]]; then
  echo "[sh_test] Reusing cached yaml-cpp source: ${YAML_CPP_SOURCE_DIR}"
  cmake_args+=("-DFETCHCONTENT_SOURCE_DIR_YAML-CPP:PATH=${YAML_CPP_SOURCE_DIR}")
fi
cmake -S "${BUILD_ROOT}" -B "${BUILD_DIR}" "${cmake_args[@]}"
cmake --build "${BUILD_DIR}" -j "${NUM_THREADS}"
