#!/usr/bin/env bash
set -euo pipefail

START_SECONDS=${SECONDS}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SH_TEST_DIR=$(realpath "${SCRIPT_DIR}/..")
PROJECT_DIR=$(realpath "${SH_TEST_DIR}/..")

TRACE_CONFIG="${TRACE_CONFIG:-${SH_TEST_DIR}/workload/llama2_7b_inference/trace_config.csv}"

TRACE_GENERATOR="${SH_TEST_DIR}/workload/llama2_7b_inference/generate_trace.py"

load_trace_config() {
  python3 "${TRACE_GENERATOR}" --print-shell-config "${TRACE_CONFIG}"
}

TRACE_CONFIG_ASSIGNMENTS=$(load_trace_config)
eval "${TRACE_CONFIG_ASSIGNMENTS}"
RESULT_DIR="${SH_TEST_DIR}/results"
RUN_OUTPUT_LOG_DIR="${RESULT_DIR}/run_logs"
RUN_OUTPUT_LOG_TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
RESULT_LOG="${RUN_OUTPUT_LOG_DIR}/run_aware_face_${TRACE_LABEL}_${RUN_OUTPUT_LOG_TIMESTAMP}.log"

ENABLE_ASTRA_INTERNAL_DEBUG_LOG=0
ASTRA_INTERNAL_LOG_DIR="${RESULT_DIR}/astra_internal_log_aware"

ASTRA_SIM="${PROJECT_DIR}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware"
SYSTEM="${SYSTEM_CONFIG}"
NETWORK="${NETWORK_CONFIG}"
COMM_GROUP="${COMM_GROUP_CONFIG}"
WORKLOAD="${TRACE_DIR}/${TRACE_PREFIX}"

mkdir -p "${RUN_OUTPUT_LOG_DIR}"
exec > >(tee "${RESULT_LOG}") 2>&1

print_elapsed_time() {
  local exit_code=$?
  local elapsed=$((SECONDS - START_SECONDS))
  local hours=$((elapsed / 3600))
  local minutes=$(((elapsed % 3600) / 60))
  local seconds=$((elapsed % 60))

  printf "[sh_test] 本次仿真花的时间: %02d:%02d:%02d (%d seconds)\n" \
    "${hours}" "${minutes}" "${seconds}" "${elapsed}"
  printf "[sh_test] 本次运行终端输出日志: %s\n" "${RESULT_LOG}"
  exit "${exit_code}"
}

trap print_elapsed_time EXIT

if [[ $# -ne 0 ]]; then
  echo "This script uses sh_test_mesh/workload/llama2_7b_inference/trace_config.csv instead of command-line arguments." >&2
  exit 1
fi

if [[ ! -x "${ASTRA_SIM}" ]]; then
  echo "Missing simulator binary: ${ASTRA_SIM}" >&2
  echo "Run: bash sh_test_mesh/run_scripts/build_analytical_aware.sh" >&2
  exit 1
fi

for ((rank = 0; rank < NPUS_COUNT; rank++)); do
  if [[ ! -f "${WORKLOAD}.${rank}.et" ]]; then
    echo "Missing generated trace file: ${WORKLOAD}.${rank}.et" >&2
    echo "Run: bash sh_test_mesh/run_scripts/generate_trace.sh" >&2
    exit 1
  fi
done

if [[ ! -f "${TRACE_DIR}/manifest.json" ]]; then
  echo "Missing generated trace files under: ${TRACE_DIR}" >&2
  echo "Run: bash sh_test_mesh/run_scripts/generate_trace.sh" >&2
  exit 1
fi

if [[ ! -f "${COMM_GROUP}" ]]; then
  echo "Missing communicator group configuration: ${COMM_GROUP}" >&2
  exit 1
fi

if [[ ! -f "${REMOTE_MEMORY}" ]]; then
  echo "Missing remote-memory configuration: ${REMOTE_MEMORY}" >&2
  exit 1
fi

# Optional metrics collection (implementation doc sec.10).  When enabled, the
# simulator reads the generator-written sidecar metrics_manifest.json; the
# trace_digest is verified against the current .et files before launch so a
# stale manifest can never be paired with new traces.
ENABLE_METRICS=${ENABLE_METRICS:-1}
METRICS_DETAIL=${METRICS_DETAIL:-full}

METRICS_ARGS=()
if [[ "${ENABLE_METRICS}" == "1" && "${METRICS_DETAIL}" != "off" ]]; then
  case "${METRICS_DETAIL}" in
    summary|full)
      ;;
    *)
      echo "METRICS_DETAIL must be off, summary or full, got: ${METRICS_DETAIL}" >&2
      exit 1
      ;;
  esac
  METRICS_MANIFEST="${TRACE_DIR}/metrics_manifest.json"
  if [[ ! -f "${METRICS_MANIFEST}" ]]; then
    echo "Missing metrics manifest: ${METRICS_MANIFEST}" >&2
    echo "Regenerate the trace (metrics are on by default): bash sh_test_mesh/run_scripts/generate_trace.sh" >&2
    exit 1
  fi
  python3 - "${METRICS_MANIFEST}" "${TRACE_DIR}" "${TRACE_PREFIX}" "${NPUS_COUNT}" <<'PY'
import hashlib
import json
import sys

manifest_path, trace_dir, trace_prefix, npus_count = (
    sys.argv[1],
    sys.argv[2],
    sys.argv[3],
    int(sys.argv[4]),
)
with open(manifest_path, encoding="utf-8") as source:
    manifest = json.load(source)
schema_version = manifest.get("schema_version")
if schema_version != 1:
    sys.exit(
        f"Unsupported metrics manifest schema_version {schema_version!r} "
        f"in {manifest_path} (this script supports 1)"
    )
per_rank = []
for rank in range(npus_count):
    et_path = f"{trace_dir}/{trace_prefix}.{rank}.et"
    with open(et_path, "rb") as source:
        per_rank.append(f"{rank}:{hashlib.sha256(source.read()).hexdigest()}")
actual = hashlib.sha256("\n".join(per_rank).encode("utf-8")).hexdigest()
expected = manifest.get("trace_digest")
if actual != expected:
    sys.exit(
        "metrics manifest trace_digest does not match the current ET files "
        f"(manifest={expected}, actual={actual}); regenerate the trace with "
        "bash sh_test_mesh/run_scripts/generate_trace.sh"
    )
print(f"[sh_test] metrics manifest verified: schema_version=1 trace_digest={actual}")
PY
  METRICS_ARGS=(
    --metrics-configuration="${METRICS_MANIFEST}"
    --metrics-detail="${METRICS_DETAIL}"
  )
fi

case "${ENABLE_ASTRA_INTERNAL_DEBUG_LOG}" in
  0)
    ASTRA_LOGGING_ARGS=(--logging-folder=off)
    ;;
  1)
    ASTRA_LOGGING_ARGS=(--logging-folder="${ASTRA_INTERNAL_LOG_DIR}")
    ;;
  *)
    echo "ENABLE_ASTRA_INTERNAL_DEBUG_LOG must be 0 or 1." >&2
    exit 1
    ;;
esac

echo "[sh_test] Running ASTRA-sim analytical congestion-aware simulation..."
echo "[sh_test] Scenario: ${MODEL_NAME} (${MLP_VARIANT}), ${HARDWARE_LABEL}, FACE mapping on ${NPUS_COUNT} NPUs in a ${MESH_SHAPE} mesh with ${INSTANCE_COUNT} unified TP=${TP_DEGREE} instances; p_chunk=${PREFILL_CHUNK_SIZE}; trace granularity=${TRACE_GRANULARITY}; Prefill keeps the existing remaining-chunk/last-arrival policy, Decode keeps the weighted Instance_map and minimum per-die LUT delta, while node-local HBM tracks model/KV usage and completed-session KV uses FIFO offload plus nearest-edge remote restore; ${REQUEST_COUNT} requests across ${SESSION_COUNT} sessions, prefill ${PREFILL_RANGE}, decode ${DECODE_RANGE}."
"${ASTRA_SIM}" \
  --workload-configuration="${WORKLOAD}" \
  --comm-group-configuration="${COMM_GROUP}" \
  --system-configuration="${SYSTEM}" \
  --remote-memory-configuration="${REMOTE_MEMORY}" \
  --network-configuration="${NETWORK}" \
  "${ASTRA_LOGGING_ARGS[@]}" \
  "${METRICS_ARGS[@]}"

echo "[sh_test] Finished."
