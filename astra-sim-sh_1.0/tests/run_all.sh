#!/bin/bash
set -e

# Path
SCRIPT_DIR=$(dirname "$(realpath $0)")
PROJECT_DIR=$(realpath "${SCRIPT_DIR}/..")
export PYTHONPATH="${PROJECT_DIR}/extern/graph_frontend${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$0] Running all regression tests..."

echo "[$0] Running rt_template..."
${SCRIPT_DIR}/rt_template/run.sh || (echo "Failed." ; exit 1)

echo "[$0] Finished all regression tests."
