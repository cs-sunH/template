#!/usr/bin/env bash
# Generate microbenchmark ETs and simulate each point with metrics collection
# (implementation doc sec.8/10).  The per-point metrics manifest is verified
# against the current .et files before every launch, exactly like
# run_sh_test_aware.sh does for service traces.
#
# Usage:
#   bash run_metric_microbench.sh [generator subset args...]
# Subset args (forwarded to generate_metric_microbench.py):
#   --tp-degrees=1,2 --prefill-chunks=128 --decode-batches=1,4 \
#   --kv-lengths=128,1024 --repeats=1
# Switches:
#   --dry-run        enumerate points only, no generation, no simulation
#   --skip-generate  reuse the existing MICROBENCH_DIR contents
# Env:
#   MICROBENCH_DIR   output root (default: sh_test_mesh/generated/metric_microbench)
#   METRICS_DETAIL   simulator metrics detail (default: full; iteration
#                    boundary records are emitted at full detail only)

set -euo pipefail

START_SECONDS=${SECONDS}

SCRIPT_DIR=$(dirname "$(realpath "$0")")
SH_TEST_DIR=$(realpath "${SCRIPT_DIR}/..")
PROJECT_DIR=$(realpath "${SH_TEST_DIR}/..")

GENERATOR="${SH_TEST_DIR}/workload/llama2_7b_inference/generate_metric_microbench.py"
ASTRA_SIM="${PROJECT_DIR}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware"

MICROBENCH_DIR="${MICROBENCH_DIR:-${SH_TEST_DIR}/generated/metric_microbench}"
METRICS_DETAIL=${METRICS_DETAIL:-full}

DRY_RUN=0
SKIP_GENERATE=0
GENERATOR_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run)
      DRY_RUN=1
      ;;
    --skip-generate)
      SKIP_GENERATE=1
      ;;
    *)
      GENERATOR_ARGS+=("$1")
      ;;
  esac
  shift
done

if [[ "${DRY_RUN}" == "1" ]]; then
  python3 "${GENERATOR}" --dry-run --output-dir="${MICROBENCH_DIR}" ${GENERATOR_ARGS[@]+"${GENERATOR_ARGS[@]}"}
  exit 0
fi

if [[ "${SKIP_GENERATE}" == "0" ]]; then
  echo "[microbench] Generating microbenchmark ETs under ${MICROBENCH_DIR} ..."
  python3 "${GENERATOR}" --output-dir="${MICROBENCH_DIR}" ${GENERATOR_ARGS[@]+"${GENERATOR_ARGS[@]}"}
fi

INDEX="${MICROBENCH_DIR}/microbench_index.json"
if [[ ! -f "${INDEX}" ]]; then
  echo "Missing microbenchmark index: ${INDEX}" >&2
  exit 1
fi
if [[ ! -x "${ASTRA_SIM}" ]]; then
  echo "Missing simulator binary: ${ASTRA_SIM}" >&2
  echo "Run: bash sh_test_mesh/run_scripts/build_analytical_aware.sh" >&2
  exit 1
fi

LOG_DIR="${MICROBENCH_DIR}/logs"
mkdir -p "${LOG_DIR}"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
AGGREGATE_LOG="${LOG_DIR}/microbench_run_${TIMESTAMP}.log"

# Per-point pre-launch verification (doc sec.10): manifest exists,
# schema_version supported, trace_digest matches the current ET bytes.
verify_point() {
  local manifest="$1" prefix="$2" npus="$3"
  python3 - "${manifest}" "${prefix}" "${npus}" <<'PY'
import hashlib
import json
import sys

manifest_path, workload_prefix, npus_count = sys.argv[1], sys.argv[2], int(sys.argv[3])
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
    with open(f"{workload_prefix}.{rank}.et", "rb") as source:
        per_rank.append(f"{rank}:{hashlib.sha256(source.read()).hexdigest()}")
actual = hashlib.sha256("\n".join(per_rank).encode("utf-8")).hexdigest()
expected = manifest.get("trace_digest")
if actual != expected:
    sys.exit(
        f"trace_digest mismatch in {manifest_path} "
        f"(manifest={expected}, actual={actual}); regenerate the microbenchmark ETs"
    )
PY
}

POINT_COUNT=$(python3 -c "import json; print(len(json.load(open('${INDEX}'))['points']))")
echo "[microbench] Running ${POINT_COUNT} point(s); aggregate log: ${AGGREGATE_LOG}"

for ((point_index = 0; point_index < POINT_COUNT; point_index++)); do
  POINT_JSON=$(python3 -c "
import json
index = json.load(open('${INDEX}'))
print(json.dumps(index['points'][${point_index}]))
")
  POINT_ID=$(python3 -c "import json; print(json.loads('''${POINT_JSON}''')['benchmark_point_id'])")
  WORKLOAD=$(python3 -c "import json; print(json.loads('''${POINT_JSON}''')['workload_prefix'])")
  MANIFEST=$(python3 -c "import json; print(json.loads('''${POINT_JSON}''')['metrics_manifest'])")
  SYSTEM=$(python3 -c "import json; print(json.load(open('${INDEX}'))['system_config'])")
  NETWORK=$(python3 -c "import json; print(json.load(open('${INDEX}'))['network_config'])")
  REMOTE_MEMORY=$(python3 -c "import json; print(json.load(open('${INDEX}'))['remote_memory_config'])")
  COMM_GROUP=$(python3 -c "import json; print(json.load(open('${INDEX}'))['comm_group_config'])")
  NPUS=$(python3 -c "import json; print(json.load(open('${INDEX}'))['npus_count'])")

  if [[ ! -f "${MANIFEST}" ]]; then
    echo "Missing metrics manifest for point ${POINT_ID}: ${MANIFEST}" >&2
    exit 1
  fi
  if ! verify_point "${MANIFEST}" "${WORKLOAD}" "${NPUS}"; then
    echo "Pre-launch metrics verification failed for point ${POINT_ID}" >&2
    exit 1
  fi

  POINT_LOG="${LOG_DIR}/point_$(printf '%06d' "${POINT_ID}")_${TIMESTAMP}.log"
  {
    echo "[microbench] point ${POINT_ID} ($(python3 -c "import json; d=json.loads('''${POINT_JSON}'''); print(d['phase'], 'tp', d['tp_degree'], 'chunk', d['prefill_chunk'], 'batch', d['decode_batch'], 'kv', d['kv_length'], 'rep', d['repeat_index'])"))"
    "${ASTRA_SIM}" \
      --workload-configuration="${WORKLOAD}" \
      --comm-group-configuration="${COMM_GROUP}" \
      --system-configuration="${SYSTEM}" \
      --remote-memory-configuration="${REMOTE_MEMORY}" \
      --network-configuration="${NETWORK}" \
      --logging-folder=off \
      --metrics-configuration="${MANIFEST}" \
      --metrics-detail="${METRICS_DETAIL}"
  } 2>&1 | tee "${POINT_LOG}" >> "${AGGREGATE_LOG}"
done

elapsed=$((SECONDS - START_SECONDS))
echo "[microbench] Done ${POINT_COUNT} point(s) in ${elapsed}s; aggregate log: ${AGGREGATE_LOG}"
