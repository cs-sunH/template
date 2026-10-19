#!/usr/bin/env bash

set -Eeuo pipefail

repo_root=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$repo_root"

python_bin=${PYTHON:-python}
if ! command -v "$python_bin" >/dev/null 2>&1; then
  printf 'Python executable not found: %s\n' "$python_bin" >&2
  exit 1
fi

for executable in simulator/build/bin/test_dram pyta/thirdparty/hotspot/hotspot; do
  if [[ ! -x "$executable" ]]; then
    printf 'Required executable is missing: %s\n' "$executable" >&2
    exit 1
  fi
done

mkdir -p results
start_seconds=$SECONDS

run_ae() {
  local name=$1
  local description=$2
  shift 2

  local log_path="results/${name}.log"
  {
    printf '\n===== %s =====\n' "$description"
    printf '[LOG] %s\n' "$log_path"
    MPLBACKEND=${MPLBACKEND:-Agg} "$python_bin" -u "tests/${name}.py" "$@"
  } 2>&1 | tee "$log_path"
}

run_ae dram_dse_cloud "Cloud DRAM DSE (Figures 10 and 12)" \
  --experiment all \
  --action all

run_ae dram_dse_edge "Edge DRAM DSE (Figure 20)" \
  --experiment all \
  --action all

run_ae chip_dse_cloud "Cloud chip DSE (Figures 13-19)" \
  --experiment all \
  --action all \
  --num-workers 1 \
  --no-cache

run_ae chip_dse_edge "Edge chip DSE (Figure 21)" \
  --experiment all \
  --action all \
  --num-workers 1 \
  --no-cache

printf '\n[TIMER] All experiments elapsed: %ss\n' "$((SECONDS - start_seconds))"
