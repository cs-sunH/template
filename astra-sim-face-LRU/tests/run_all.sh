#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_DIR}/extern/graph_frontend${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$0] Running the FACE scheduler regression suite (the only suite"
echo "[$0] wired here; other suites live under sh_test_mesh/tests and"
echo "[$0] sh_test_mesh/slo_tools/tests)..."

echo "[$0] Running FACE scheduler tests..."
(cd "${PROJECT_DIR}" && python3 -m unittest sh_test_mesh/workload/llama2_7b_inference/test_face_scheduler.py) || (echo "Failed." ; exit 1)

echo "[$0] Finished the FACE scheduler regression suite."
