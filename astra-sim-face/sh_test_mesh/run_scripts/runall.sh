#!/usr/bin/env bash
set -euo pipefail

START_SECONDS=${SECONDS}

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)

run_step() {
  local step_name=$1
  local script=$2
  shift 2
  echo "[runall] === Running ${step_name}: ${script} ==="
  bash "${SCRIPT_DIR}/${script}" "$@"
}

run_step "clean" "clean_history.sh"
run_step "build" "build_analytical_aware.sh"
run_step "generate" "generate_trace.sh"
run_step "run" "run_sh_test_aware.sh"

RESULT_DIR=$(cd -- "${SCRIPT_DIR}/../results" && pwd -P)
RUN_LOG_DIR="${RESULT_DIR}/run_logs"
RUN_LOGS=()
while IFS= read -r log; do
  RUN_LOGS+=("${log}")
done < <(find "${RUN_LOG_DIR}" -maxdepth 1 -type f -name '*.log' | sort)

if [[ ${#RUN_LOGS[@]} -eq 0 ]]; then
  echo "[runall] No run logs found under ${RUN_LOG_DIR}; cannot run metrics postprocess." >&2
  exit 1
fi

run_step "postprocess" "run_metrics_postprocess.sh" "${RUN_LOGS[@]}" \
  --out-raw="${RESULT_DIR}/raw_metrics.csv" \
  --out-normalized="${RESULT_DIR}/normalized_metrics.csv"

elapsed=$((SECONDS - START_SECONDS))
hours=$((elapsed / 3600))
minutes=$(((elapsed % 3600) / 60))
seconds=$((elapsed % 60))

printf "[runall] 整个流程共花费: %02d:%02d:%02d (%d seconds)\n" \
  "${hours}" "${minutes}" "${seconds}" "${elapsed}"
