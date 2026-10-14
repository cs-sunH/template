#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd -P)
WORK=${ROOT}/work/wscllm-workspace-e2e
PLAN_MATCH=("${WORK}"/generated/llama2_7b_wsc_llm_inference_54npus_plan_*)
if [[ ${#PLAN_MATCH[@]} -ne 1 || ! -d ${PLAN_MATCH[0]} ]]; then
  echo "expected exactly one official plan directory under ${WORK}/generated; found: ${PLAN_MATCH[*]}" >&2
  exit 2
fi
PLAN_DIR=${PLAN_MATCH[0]}
RUNTIME=${WORK}/generated/generated/runtime_config/workspace_e2e_coarse_2rank__workspace-e2e-coarse__no_memory_expansion
REQUEST_QUEUE=${WORK}/workspace_e2e_request_queue_recompute.csv
SOURCE_CSV=${WORK}/workspace_e2e_source.csv
CANONICAL_SIDECAR=${WORK}/workspace_e2e_canonical_sidecar.csv
TRACE_MATERIALIZER=${ROOT}/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/traces/derive_20_first_30_seconds.py
PLAN_MATERIALIZER=${ROOT}/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/plan_materializer.py
HARDWARE=${RUNTIME}/resolved_die_model.json
PYTHON=${PYTHON:-python3}
BIN=${ROOT}/work/integration-build/build/astra-sim-wscllm/bin/AstraSim_Analytical_Congestion_Aware_Online
PY_CABI=${ROOT}/work/build-die/lib/libcoarse_die_cabi.so
CPP_CABI=${ROOT}/work/integration-build/build/astra-sim-wscllm/lib/libcoarse_die_cabi.so
RUN_DIR=${WORK}/run

for required in \
  "${BIN}" \
  "${PY_CABI}" \
  "${CPP_CABI}" \
  "${REQUEST_QUEUE}" \
  "${REQUEST_QUEUE}.provenance.json" \
  "${SOURCE_CSV}" \
  "${CANONICAL_SIDECAR}" \
  "${TRACE_MATERIALIZER}" \
  "${PLAN_MATERIALIZER}" \
  "${HARDWARE}" \
  "${WORK}/trace_config.csv" \
  "${PLAN_DIR}/manifest.json" \
  "${PLAN_DIR}/metrics_manifest.json" \
  "${RUNTIME}/system.json" \
  "${RUNTIME}/network.yml" \
  "${RUNTIME}/comm_group.json"; do
  [[ -f "${required}" ]] || { echo "missing required input: ${required}" >&2; exit 2; }
done
[[ -x "${BIN}" ]] || { echo "Online binary is not executable: ${BIN}" >&2; exit 2; }
command -v strace >/dev/null || { echo "strace is required to retain bridge wire records" >&2; exit 2; }

mkdir -p "${WORK}"
rm -rf -- "${RUN_DIR}"
mkdir -p "${RUN_DIR}/bridge" "${RUN_DIR}/wire_audit"

export PYTHONPATH="${ROOT}/work/deps/python-protobuf${PYTHONPATH:+:${PYTHONPATH}}"
export DIE_COARSE_CABI_PATH="${PY_CABI}"
export ASTRA_LINK_OBSERVER=0
export SH_ONLINE_VALIDATE=0
export SH_WORKSPACE_ADMISSION=auto
# Explicitly exercise the supported first-token marker path for this one-token
# decode request; the default production profile leaves this experiment off.
export SH_FIRST_TOKEN_SPLIT=1
export SH_PROFILE_JSONL=1

{
  printf 'root=%s\n' "${ROOT}"
  printf 'head=%s\n' "$(git -C "${ROOT}" rev-parse HEAD)"
  printf 'online_binary=%s\n' "${BIN}"
  printf 'online_binary_sha256=%s\n' "$(sha256sum "${BIN}" | awk '{print $1}')"
  printf 'online_binary_mtime=%s\n' "$(stat -c '%y' "${BIN}")"
  printf 'python=%s\n' "$("${PYTHON}" --version 2>&1)"
  printf 'protobuf=%s\n' "$("${PYTHON}" -c 'import google.protobuf; print(google.protobuf.__version__)')"
  printf 'python_cabi_sha256=%s\n' "$(sha256sum "${PY_CABI}" | awk '{print $1}')"
  printf 'cpp_cabi_sha256=%s\n' "$(sha256sum "${CPP_CABI}" | awk '{print $1}')"
  printf 'request_queue_sha256=%s\n' "$(sha256sum "${REQUEST_QUEUE}" | awk '{print $1}')"
  printf 'request_queue_provenance_sha256=%s\n' "$(sha256sum "${REQUEST_QUEUE}.provenance.json" | awk '{print $1}')"
  printf 'request_source_sha256=%s\n' "$(sha256sum "${SOURCE_CSV}" | awk '{print $1}')"
  printf 'canonical_sidecar_sha256=%s\n' "$(sha256sum "${CANONICAL_SIDECAR}" | awk '{print $1}')"
  printf 'trace_config_sha256=%s\n' "$(sha256sum "${WORK}/trace_config.csv" | awk '{print $1}')"
  printf 'trace_materializer_sha256=%s\n' "$(sha256sum "${TRACE_MATERIALIZER}" | awk '{print $1}')"
  printf 'plan_materializer_sha256=%s\n' "$(sha256sum "${PLAN_MATERIALIZER}" | awk '{print $1}')"
  printf 'resolved_hardware_sha256=%s\n' "$(sha256sum "${HARDWARE}" | awk '{print $1}')"
  printf 'manifest_sha256=%s\n' "$(sha256sum "${PLAN_DIR}/manifest.json" | awk '{print $1}')"
  printf 'metrics_manifest_sha256=%s\n' "$(sha256sum "${PLAN_DIR}/metrics_manifest.json" | awk '{print $1}')"
  printf 'plan_dir=%s\n' "${PLAN_DIR}"
  printf 'runtime_config_dir=%s\n' "${RUNTIME}"
  printf 'workspace_admission=%s\n' "${SH_WORKSPACE_ADMISSION}"
  printf 'first_token_split=%s\n' "${SH_FIRST_TOKEN_SPLIT}"
} > "${RUN_DIR}/provenance.txt"

CPP_EXIT=0
PY_EXIT=0
strace -f -qq -s 1048576 -yy \
  -e trace=openat,close,read,write,writev,rename,renameat,renameat2,unlink,unlinkat \
  -o "${RUN_DIR}/cpp.strace" \
  "${BIN}" \
    --online-mode strategy \
    --bridge-dir "${RUN_DIR}/bridge" \
    --online-validate "${SH_ONLINE_VALIDATE}" \
    --bridge-timeout-ms 120000 \
    --request-queue-csv "${REQUEST_QUEUE}" \
    --close-input \
    --workload-configuration="${PLAN_DIR}/llama2_7b_wsc_llm_inference" \
    --comm-group-configuration="${RUNTIME}/comm_group.json" \
    --system-configuration="${RUNTIME}/system.json" \
    --network-configuration="${RUNTIME}/network.yml" \
    --logging-folder=off \
    --metrics-configuration="${PLAN_DIR}/metrics_manifest.json" \
    --metrics-detail=full \
    > "${RUN_DIR}/cpp.log" 2>&1 &
CPP_PID=$!

cleanup() {
  if kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

FIFO_WAIT_S=${SH_FIFO_WAIT_S:-60}
deadline=$(( $(date +%s) + FIFO_WAIT_S ))
until [[ -p "${RUN_DIR}/bridge/req_notify.fifo" && -p "${RUN_DIR}/bridge/resp_notify.fifo" ]]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    set +e
    wait "${CPP_PID}"
    CPP_EXIT=$?
    set -e
    break
  fi
  if (( $(date +%s) >= deadline )); then
    echo "FIFOs were not ready within ${FIFO_WAIT_S}s" >&2
    tail -30 "${RUN_DIR}/cpp.log" >&2 || true
    exit 1
  fi
  sleep 0.1
done

if [[ -p "${RUN_DIR}/bridge/req_notify.fifo" && -p "${RUN_DIR}/bridge/resp_notify.fifo" ]]; then
  set +e
  strace -f -qq -s 1048576 -yy \
    -e trace=openat,close,read,write,writev,rename,renameat,renameat2,unlink,unlinkat \
    -o "${RUN_DIR}/python.strace" \
    "${PYTHON}" -u "${ROOT}/astra-sim-wscllm/sh_test_mesh/workload/llama2_7b_inference/online/online_service.py" \
      --bridge-dir "${RUN_DIR}/bridge" \
      --mode strategy \
      --plan-dir "${PLAN_DIR}" \
      --config "${WORK}/trace_config.csv" \
    > "${RUN_DIR}/python.log" 2>&1
  PY_EXIT=$?
  wait "${CPP_PID}"
  CPP_EXIT=$?
  set -e
fi

printf 'cpp_exit=%s\npython_exit=%s\n' "${CPP_EXIT}" "${PY_EXIT}" | tee "${RUN_DIR}/exit_codes.txt"
if (( CPP_EXIT != 0 )); then tail -50 "${RUN_DIR}/cpp.log" >&2 || true; fi
if (( PY_EXIT != 0 )); then tail -50 "${RUN_DIR}/python.log" >&2 || true; fi

if [[ -s "${RUN_DIR}/cpp.strace" && -s "${RUN_DIR}/python.strace" ]]; then
  if ! "${PYTHON}" "${WORK}/audit_wire.py" "${RUN_DIR}" \
      > "${RUN_DIR}/wire_audit/audit.stdout.json"; then
    echo "wire audit extraction failed; see ${RUN_DIR}/wire_audit" >&2
  fi
  if [[ -s "${RUN_DIR}/wire_audit/response_0.json" ]]; then
    if ! "${PYTHON}" "${WORK}/inspect_captured_plans.py" \
        > "${RUN_DIR}/wire_audit/compiled_plans.stdout.json"; then
      echo "captured Plan inspection failed; see ${RUN_DIR}/wire_audit" >&2
    fi
  fi
fi

echo "runner artifacts: ${RUN_DIR}"
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]]
