#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SH_TEST_DIR=$(realpath "${SCRIPT_DIR}/..")

# Trace-generation parameters are defined in the CSV beside the Python generator.
TRACE_GENERATOR="${SH_TEST_DIR}/workload/llama2_7b_inference/generate_trace.py"

select_python_bin() {
  local candidate
  for candidate in python3 python python.exe; do
    if command -v "${candidate}" >/dev/null 2>&1 && \
      "${candidate}" - <<'PY' >/dev/null 2>&1; then
import google.protobuf
PY
      printf "%s\n" "${candidate}"
      return
    fi
  done

  echo "Cannot find python3/python/python.exe with google.protobuf installed." >&2
  echo "Install the Python protobuf runtime for the interpreter in PATH." >&2
  echo "On Ubuntu, run: sudo apt-get install -y python3-protobuf" >&2
  echo "Then verify with: python3 -c 'import google.protobuf; print(google.protobuf.__file__)'" >&2
  echo "If you use a virtualenv or conda env, activate it before running this script." >&2
  exit 1
}

PYTHON_BIN=$(select_python_bin)
TRACE_GENERATOR_ARG="${TRACE_GENERATOR}"
if [[ "${PYTHON_BIN}" == *.exe ]] && command -v wslpath >/dev/null 2>&1; then
  TRACE_GENERATOR_ARG=$(wslpath -m "${TRACE_GENERATOR}")
fi

TRACE_GENERATOR_ARGS=()
case $# in
  0)
    ;;
  1)
    case "$1" in
      -h|--help)
        cat <<'EOF'
Usage: bash sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS|--jobs JOBS]

Generate the default Chakra ET workload.  JOBS is the number of CPU worker
processes used for rank-local ET construction and serialization.  With no
explicit option, TRACE_GEN_JOBS is honored; otherwise all available CPUs are
used.  Use --jobs 1 for the serial baseline.
EOF
        exit 0
        ;;
      --jobs=*)
        TRACE_GENERATOR_ARGS=("$1")
        ;;
      *)
        echo "Usage: bash sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS|--jobs JOBS]" >&2
        exit 1
        ;;
    esac
    ;;
  2)
    if [[ "$1" == "-j" || "$1" == "--jobs" ]]; then
      TRACE_GENERATOR_ARGS=("$1" "$2")
    else
      echo "Usage: bash sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS|--jobs JOBS]" >&2
      exit 1
    fi
    ;;
  *)
    echo "Usage: bash sh_test_mesh/run_scripts/generate_trace.sh [-j JOBS|--jobs JOBS]" >&2
    exit 1
    ;;
esac

echo "[sh_test] Generating Chakra ET trace..."
"${PYTHON_BIN}" "${TRACE_GENERATOR_ARG}" "${TRACE_GENERATOR_ARGS[@]}"
