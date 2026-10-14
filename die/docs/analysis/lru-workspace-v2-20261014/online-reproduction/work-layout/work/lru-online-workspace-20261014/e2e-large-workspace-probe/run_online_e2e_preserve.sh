#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../../.." && pwd -P)
WORK=${ROOT}/work/lru-online-workspace-20261014/e2e-large-workspace-probe
RUN_DIR=${WORK}/run

if [[ -e "${RUN_DIR}" ]]; then
  ARCHIVE=${WORK}/run-attempt-$(date -u +%Y%m%dT%H%M%SZ)
  if [[ -e "${ARCHIVE}" ]]; then
    echo "refusing to overwrite existing run archive: ${ARCHIVE}" >&2
    exit 2
  fi
  mv -- "${RUN_DIR}" "${ARCHIVE}"
  echo "preserved previous run at ${ARCHIVE}"
fi

exec "${WORK}/run_online_e2e.sh" "$@"
