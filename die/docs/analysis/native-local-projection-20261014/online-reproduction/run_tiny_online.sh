#!/usr/bin/env bash
set -euo pipefail

PROFILE=${1:?usage: run_tiny_online.sh legacy|coarse_intended}
if [[ "${PROFILE}" != "legacy" && "${PROFILE}" != "coarse_intended" ]]; then
  echo "profile must be legacy or coarse_intended" >&2
  exit 2
fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=${JOINT_REPRO_REPO_ROOT:-$(git -C "${HERE}" rev-parse --show-toplevel)}
ASTRA_SIM_ROOT=${ASTRA_SIM_ROOT:-${ROOT}/astra-sim-joint}
BUILD_DIR=${BUILD_DIR:-${ROOT}/work/build-joint}
JOINT_REPRO_WORKDIR=${JOINT_REPRO_WORKDIR:-${ROOT}/work/joint-reproduction}
STAGE=${JOINT_REPRO_WORKDIR}/stage
MANIFEST=${JOINT_REPRO_WORKDIR}/tiny-inputs.json
BIN=${BUILD_DIR}/bin/AstraSim_Analytical_Congestion_Aware_Online
RUN_DIR=${JOINT_REPRO_WORKDIR}/runs/${PROFILE}-$(date -u +%Y%m%dT%H%M%SZ)-$$

if [[ ! -x "${BIN}" ]]; then
  echo "missing built Online binary; see README.md" >&2
  exit 2
fi

mkdir -p "${JOINT_REPRO_WORKDIR}/runs"
export JOINT_REPRO_REPO_ROOT="${ROOT}"
export JOINT_REPRO_WORKDIR ASTRA_SIM_ROOT BUILD_DIR
export DIE_MODEL_CABI_PATH="${BUILD_DIR}/lib/libdie_model_cabi.so"
export DIE_COARSE_CABI_PATH="${BUILD_DIR}/lib/libcoarse_die_cabi.so"
python "${HERE}/prepare_tiny_inputs.py" --die-cabi "${DIE_MODEL_CABI_PATH}" \
  > "${JOINT_REPRO_WORKDIR}/prepare-last-run.log"

# The staged Python import root resolves the repository's fixed build
# locations. Keep the real receiver ABI inside that isolated stage tree.
LEDGER_STAGE=${STAGE}/build_congestion_aware/lib
mkdir -p "${LEDGER_STAGE}"
cp "${BUILD_DIR}/lib/libjoint_ledger_cabi.so" \
  "${LEDGER_STAGE}/libjoint_ledger_cabi.so"
cp "${BUILD_DIR}/lib/libjoint_projection.so" \
  "${LEDGER_STAGE}/libjoint_projection.so"

eval "$(python - "${MANIFEST}" "${PROFILE}" <<'PY'
import json, shlex, sys
data = json.load(open(sys.argv[1], encoding="utf-8"))[sys.argv[2]]
for key in ("plan_dir", "runtime_config_dir", "trace_config", "active_runtime_config_dir"):
    print(f"{key.upper()}={shlex.quote(data[key])}")
PY
)"

RC_SOURCE=${RUNTIME_CONFIG_DIR}
RC_ACTIVE=${ACTIVE_RUNTIME_CONFIG_DIR}
if [[ ! -d "${RC_SOURCE}" || ! -d "$(dirname "${RC_ACTIVE}")" ]]; then
  echo "missing runtime config input: ${RC_SOURCE}" >&2
  exit 2
fi
REQUEST_CSV=${STAGE}/sh_test_mesh/workload/llama2_7b_inference/joint_tiny_requests.csv
ONLINE=${STAGE}/sh_test_mesh/workload/llama2_7b_inference/online/online_service.py
mkdir -p "${JOINT_REPRO_WORKDIR}/runs" "${RUN_DIR}/bridge" "${RC_ACTIVE}"
cp "${RUNTIME_CONFIG_DIR}/comm_group.json" "${RC_ACTIVE}/comm_group.json"
cp "${RUNTIME_CONFIG_DIR}/network.yml" "${RC_ACTIVE}/network.yml"
cp "${RUNTIME_CONFIG_DIR}/remote_memory.json" "${RC_ACTIVE}/remote_memory.json"
cp "${RUNTIME_CONFIG_DIR}/system.json" "${RC_ACTIVE}/system.json"

METRICS_DETAIL=${SH_METRICS_DETAIL:-summary}
BRIDGE_TIMEOUT_MS=30000
if [[ "${PROFILE}" == "coarse_intended" ]]; then
  BRIDGE_TIMEOUT_MS=10000
fi

"${BIN}" \
  --online-mode strategy \
  --bridge-dir "${RUN_DIR}/bridge" \
  --online-validate 1 \
  --bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}" \
  --request-queue-csv "${REQUEST_CSV}" \
  --close-input \
  --workload-configuration="${PLAN_DIR}/llama2_7b_inference" \
  --comm-group-configuration="${RC_ACTIVE}/comm_group.json" \
  --system-configuration="${RC_ACTIVE}/system.json" \
  --remote-memory-configuration="${RC_ACTIVE}/remote_memory.json" \
  --network-configuration="${RC_ACTIVE}/network.yml" \
  --logging-folder=off \
  --metrics-configuration="${PLAN_DIR}/metrics_manifest.json" \
  --metrics-detail="${METRICS_DETAIL}" \
  > "${RUN_DIR}/cpp.log" 2>&1 &
CPP_PID=$!
cleanup() {
  if kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

deadline=$(( $(date +%s) + 90 ))
until [[ -p "${RUN_DIR}/bridge/req_notify.fifo" && -p "${RUN_DIR}/bridge/resp_notify.fifo" ]]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "C++ Online exited before bridge setup (run=${RUN_DIR})" >&2
    tail -40 "${RUN_DIR}/cpp.log" >&2 || true
    exit 1
  fi
  if (( $(date +%s) > deadline )); then
    echo "bridge setup timed out (run=${RUN_DIR})" >&2
    exit 1
  fi
  sleep 0.2
done

set +e
(
  cd "${STAGE}/sh_test_mesh/workload/llama2_7b_inference"
  PYTHONPATH="${STAGE}:${STAGE}/sh_test_mesh:${STAGE}/sh_test_mesh/workload/llama2_7b_inference" \
    python -u "${ONLINE}" \
      --bridge-dir "${RUN_DIR}/bridge" \
      --mode strategy \
      --plan-dir "${PLAN_DIR}" \
      --config "${TRACE_CONFIG}"
) > "${RUN_DIR}/python.log" 2>&1
PY_EXIT=$?
wait "${CPP_PID}"
CPP_EXIT=$?
set -e
CPP_PID=
trap - EXIT

printf 'profile=%s cpp_exit=%s python_exit=%s run_dir=%s\n' \
  "${PROFILE}" "${CPP_EXIT}" "${PY_EXIT}" "${RUN_DIR}"
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 ]]; then
  tail -40 "${RUN_DIR}/cpp.log" >&2 || true
  tail -40 "${RUN_DIR}/python.log" >&2 || true
  exit 1
fi
