#!/usr/bin/env bash
# Cross-run metrics post-processing wrapper (implementation doc sec.11).
# Collects [METRIC] JSON records from one or more run logs — service runs
# from run_sh_test_aware.sh and/or microbenchmark runs from
# run_metric_microbench.sh — and emits raw_metrics.csv plus
# normalized_metrics.csv.  Works with whatever the logs contain: service
# logs alone, microbenchmark logs alone, or a mix.
#
# Usage:
#   bash run_metrics_postprocess.sh <run.log> [more.log ...] [options]
# All arguments are forwarded to metrics_postprocess.py:
#   --out-raw=PATH            raw CSV (default: raw_metrics.csv)
#   --out-normalized=PATH     normalized CSV (default: normalized_metrics.csv)
#   --normalization=group_max|ratio_to_baseline   (default: group_max)
#   --baseline=<run_id>       baseline run for ratio_to_baseline
#   --group-config=PATH       JSON comparison group config

set -euo pipefail

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SH_TEST_DIR=$(realpath "${SCRIPT_DIR}/..")

POSTPROCESS="${SH_TEST_DIR}/workload/llama2_7b_inference/metrics_postprocess.py"

if [[ ! -f "${POSTPROCESS}" ]]; then
  echo "Missing postprocess script: ${POSTPROCESS}" >&2
  exit 1
fi

if [[ $# -eq 0 ]]; then
  echo "Usage: bash run_metrics_postprocess.sh <run.log> [more.log ...] [metrics_postprocess.py options]" >&2
  exit 1
fi

exec python3 "${POSTPROCESS}" "$@"
