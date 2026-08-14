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

if [[ $# -ne 0 ]]; then
  echo "This script uses the default CSV next to generate_trace.py. Edit sh_test_mesh/workload/llama2_7b_inference/trace_config.csv instead of passing arguments." >&2
  exit 1
fi

echo "[sh_test] Generating Chakra ET trace..."
"${PYTHON_BIN}" "${TRACE_GENERATOR_ARG}"
