#!/usr/bin/env bash
set -euo pipefail

usage() {
  echo "usage: run_tiny_online.sh face|face-lru legacy|coarse [ONLINE_BIN] [tiny|workspace-overlap] [OVERLAP_ARRIVAL_NS]" >&2
}

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  usage
  exit 0
fi
SOURCE_SHORT=${1:-}
PROFILE=${2:-}
SCENARIO=${4:-tiny}
OVERLAP_ARRIVAL_NS=${5:-${FACE_OVERLAP_ARRIVAL_NS:-1000}}
if [[ "${SOURCE_SHORT}" != "face" && "${SOURCE_SHORT}" != "face-lru" ]]; then
  usage
  exit 2
fi
if [[ "${PROFILE}" != "legacy" && "${PROFILE}" != "coarse" ]]; then
  usage
  exit 2
fi
if [[ "${SCENARIO}" != "tiny" && "${SCENARIO}" != "workspace-overlap" ]]; then
  usage
  exit 2
fi

HERE=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
ROOT=$(git -C "${HERE}" rev-parse --show-toplevel)
case "${SOURCE_SHORT}" in
  face)
    SOURCE_VARIANT=astra-sim-face
    ABI_BUILD_NAME=build-abi-astra-sim-face
    ;;
  face-lru)
    SOURCE_VARIANT=astra-sim-face-LRU
    ABI_BUILD_NAME=build-abi-astra-sim-face-LRU
    ;;
esac

REQUESTED_BIN=${3:-${ONLINE_BIN:-}}
if [[ -n "${BUILD_DIR:-}" ]]; then
  BUILD_DIR=$(cd -- "${BUILD_DIR}" && pwd -P)
elif [[ -n "${REQUESTED_BIN}" ]]; then
  BUILD_DIR=$(cd -- "$(dirname -- "${REQUESTED_BIN}")/.." && pwd -P)
else
  BUILD_DIR=${ROOT}/work/build-online-${SOURCE_SHORT}
fi
ONLINE_BIN=${REQUESTED_BIN:-${BUILD_DIR}/bin/AstraSim_Analytical_Congestion_Aware_Online}
FALLBACK_ABI_DIR=${ROOT}/work/five-abi-validation-20261014/${ABI_BUILD_NAME}/lib
DIE_CABI=${DIE_MODEL_CABI_PATH:-${BUILD_DIR}/lib/libdie_model_cabi.so}
COARSE_CABI=${DIE_COARSE_CABI_PATH:-${BUILD_DIR}/lib/libcoarse_die_cabi.so}
if [[ ! -f "${DIE_CABI}" ]]; then
  DIE_CABI=${FALLBACK_ABI_DIR}/libdie_model_cabi.so
fi
if [[ ! -f "${COARSE_CABI}" ]]; then
  COARSE_CABI=${FALLBACK_ABI_DIR}/libcoarse_die_cabi.so
fi
if [[ ! -x "${ONLINE_BIN}" ]]; then
  echo "missing ${SOURCE_VARIANT} Online binary: ${ONLINE_BIN}" >&2
  echo "Set ONLINE_BIN (or BUILD_DIR) to the matching build for this source variant." >&2
  exit 2
fi
if [[ ! -f "${DIE_CABI}" || ! -f "${COARSE_CABI}" ]]; then
  echo "missing FACE C ABI input(s) under ${FALLBACK_ABI_DIR}" >&2
  exit 2
fi

WORK=${HERE}
STAGE=${WORK}/stage/${SOURCE_SHORT}
RUN_ID=$(date -u +%Y%m%dT%H%M%SZ)-$$
RUN_DIR=${WORK}/runs/${SOURCE_VARIANT}/${PROFILE}-${SCENARIO}-${RUN_ID}
mkdir -p "${RUN_DIR}/bridge"
{
  printf 'command='
  printf '%q ' "$0" "$@"
  printf '\nresolved_online_binary=%s\n' "${ONLINE_BIN}"
  printf 'source_head='
  git -C "${ROOT}" rev-parse HEAD
  printf 'scenario=%s\n' "${SCENARIO}"
  printf 'overlap_arrival_ns=%s\n' "${OVERLAP_ARRIVAL_NS}"
  printf 'build_dir=%s\n' "${BUILD_DIR}"
  printf 'bridge_timeout_ms=%s\n' "${BRIDGE_TIMEOUT_MS:-120000}"
  printf 'online_validate=%s\n' "${SH_ONLINE_VALIDATE:-1}"
  printf 'metrics_detail=%s\n' "${SH_METRICS_DETAIL:-summary}"
  printf 'workspace_admission=%s\n' "${SH_WORKSPACE_ADMISSION:-auto}"
  printf 'link_observer=%s\n' "${ASTRA_LINK_OBSERVER:-0}"
printf 'trace_enabled=%s\n' "${FACE_TRACE_ENABLED:-0}"
} > "${RUN_DIR}/invocation.txt"

export DIE_MODEL_CABI_PATH=${DIE_CABI}
export DIE_COARSE_CABI_PATH=${COARSE_CABI}
export LD_LIBRARY_PATH="${BUILD_DIR}/lib:${LD_LIBRARY_PATH:-}"
export PYTHONUNBUFFERED=1
export ASTRA_LINK_OBSERVER=${ASTRA_LINK_OBSERVER:-0}
export SH_WORKSPACE_ADMISSION=${SH_WORKSPACE_ADMISSION:-auto}
export SH_METRICS_DETAIL=${SH_METRICS_DETAIL:-summary}

python3 "${HERE}/prepare_tiny_inputs.py" \
  --source-variant "${SOURCE_SHORT}" \
  --profile "${PROFILE}" \
  --scenario "${SCENARIO}" \
  --overlap-arrival-ns "${OVERLAP_ARRIVAL_NS}" \
  --die-cabi "${DIE_CABI}" \
  --coarse-die-cabi "${COARSE_CABI}" \
  > "${RUN_DIR}/prepare.log"

INPUTS=${WORK}/inputs/${SOURCE_SHORT}/tiny-inputs.json
eval "$(python3 - "${INPUTS}" "${PROFILE}" <<'PY'
import json
import shlex
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
entry = payload["profiles"][sys.argv[2]]
values = {
    "PLAN_DIR": entry["plan_dir"],
    "TRACE_CONFIG": entry["trace_config"],
    "REQUEST_CSV": entry["request_queue"],
    "RUNTIME_CONFIG_DIR": entry["runtime_config_dir"],
    "MANIFEST": entry["manifest"],
    "METRICS_MANIFEST": entry["metrics_manifest"],
}
for key, value in values.items():
    print(f"{key}={shlex.quote(value)}")
PY
)"

REMOTE_MEMORY_CONFIG=
if [[ "${SOURCE_SHORT}" == "face-lru" ]]; then
  REMOTE_MEMORY_CONFIG=${RUNTIME_CONFIG_DIR}/remote_memory.json
  if [[ ! -f "${REMOTE_MEMORY_CONFIG}" ]]; then
    echo "FACE-LRU materializer did not produce remote_memory.json: ${REMOTE_MEMORY_CONFIG}" >&2
    exit 1
  fi
fi

if [[ ! -d "${PLAN_DIR}" || ! -d "${RUNTIME_CONFIG_DIR}" ]]; then
  echo "prepared plan or runtime config is missing (see ${RUN_DIR}/prepare.log)" >&2
  exit 1
fi

TRACE_LOGGING_FOLDER=off
if [[ "${FACE_TRACE_ENABLED:-0}" == "1" ]]; then
  # This changes only the generated work config and the per-run logging sink.
  # Keeping the exact system input and trace logs makes this diagnostic run
  # distinguishable from the default reproduction inputs.
  python3 - "${RUNTIME_CONFIG_DIR}/system.json" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
payload["trace-enabled"] = 1
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
  TRACE_LOGGING_FOLDER=${RUN_DIR}/trace-logs
fi
if [[ ! -f "${MANIFEST}" || ! -f "${METRICS_MANIFEST}" || ! -f "${REQUEST_CSV}" ]]; then
  echo "prepared queue/manifest input is incomplete (see ${RUN_DIR}/prepare.log)" >&2
  exit 1
fi

mkdir -p "${RUN_DIR}/inputs"
mkdir -p "${RUN_DIR}/source-stage"
cp "${INPUTS}" "${RUN_DIR}/inputs/tiny-inputs.json"
cp "${WORK}/inputs/${SOURCE_SHORT}/source-stage-identity.json" "${RUN_DIR}/inputs/source-stage-identity.json"
cp -a "${STAGE}/." "${RUN_DIR}/source-stage/"
mkdir -p "${RUN_DIR}/inputs/protobuf"
cp "${STAGE}/extern/graph_frontend/chakra/schema/protobuf/et_def.proto" "${RUN_DIR}/inputs/protobuf/et_def.proto"
cp "${STAGE}/extern/graph_frontend/chakra/schema/protobuf/et_def_pb2.py" "${RUN_DIR}/inputs/protobuf/et_def_pb2.py"
cp "${TRACE_CONFIG}" "${RUN_DIR}/inputs/trace_config.csv"
cp "${REQUEST_CSV}" "${RUN_DIR}/inputs/request_queue.csv"
cp "${REQUEST_CSV}.provenance.json" "${RUN_DIR}/inputs/request_queue.csv.provenance.json"
cp "${MANIFEST}" "${RUN_DIR}/inputs/manifest.json"
cp "${METRICS_MANIFEST}" "${RUN_DIR}/inputs/metrics_manifest.json"
cp "${RUNTIME_CONFIG_DIR}/system.json" "${RUN_DIR}/inputs/system.json"
cp "${RUNTIME_CONFIG_DIR}/network.yml" "${RUN_DIR}/inputs/network.yml"
cp "${RUNTIME_CONFIG_DIR}/comm_group.json" "${RUN_DIR}/inputs/comm_group.json"
if [[ -n "${REMOTE_MEMORY_CONFIG}" ]]; then
  cp "${REMOTE_MEMORY_CONFIG}" "${RUN_DIR}/inputs/remote_memory.json"
fi

capture_cpp_source_identity() {
  python3 - "$1" "${ROOT}/${SOURCE_VARIANT}" <<'PY'
import hashlib
import json
import subprocess
import sys
from pathlib import Path

destination = Path(sys.argv[1])
variant_root = Path(sys.argv[2]).resolve()
source_root = variant_root
if not source_root.is_dir():
    raise SystemExit(f"missing C++ source tree: {source_root}")
files = {}
for path in sorted(source_root.rglob("*")):
    if not path.is_file():
        continue
    relative = path.relative_to(source_root)
    if any(
        part in {".git", "build", "generated", "runs", "__pycache__", ".pytest_cache"}
        for part in relative.parts
    ):
        continue
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    files[relative.as_posix()] = digest
tree_hash = hashlib.sha256()
for name, digest in files.items():
    tree_hash.update(name.encode("utf-8"))
    tree_hash.update(b"\0")
    tree_hash.update(bytes.fromhex(digest))
payload = {
    "source_variant": variant_root.name,
    "source_head": subprocess.check_output(
        ["git", "-C", str(variant_root), "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_tree": str(source_root),
    "source_tree_sha256": tree_hash.hexdigest(),
    "source_file_count": len(files),
    "source_file_sha256": files,
}
destination.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY
}

CPP_SOURCE_ID=${RUN_DIR}/inputs/cpp-source-identity.json
capture_cpp_source_identity "${CPP_SOURCE_ID}"

python3 - "${RUN_DIR}/inputs/run-input-identity.json" "${SOURCE_VARIANT}" "${PROFILE}" "${SCENARIO}" \
  "${OVERLAP_ARRIVAL_NS}" \
  "${ONLINE_BIN}" "${DIE_CABI}" "${COARSE_CABI}" "${MANIFEST}" "${REQUEST_CSV}.provenance.json" \
  "${CPP_SOURCE_ID}" "${RUN_DIR}/inputs/source-stage-identity.json" "${REMOTE_MEMORY_CONFIG}" \
  "${RUN_DIR}/inputs/system.json" "${FACE_TRACE_ENABLED:-0}" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

destination, source_variant, profile, scenario, overlap_arrival_ns, online, die, coarse, manifest, sidecar, cpp_source_identity, stage_identity_path, remote_memory, system_path, trace_enabled = sys.argv[1:]
cpp_source_path = Path(cpp_source_identity)
cpp_source = json.loads(cpp_source_path.read_text(encoding="utf-8"))
stage_path = Path(stage_identity_path)
stage = json.loads(stage_path.read_text(encoding="utf-8"))
payload = {
    "source_variant": source_variant,
    "profile": profile,
    "scenario": scenario,
    "workspace_overlap_arrival_ns": (
        int(overlap_arrival_ns) if scenario == "workspace-overlap" else None
    ),
    "online_binary": {"path": str(Path(online).resolve()), "sha256": sha(online)},
    "die_cabi": {"path": str(Path(die).resolve()), "sha256": sha(die)},
    "coarse_die_cabi": {"path": str(Path(coarse).resolve()), "sha256": sha(coarse)},
    "materialized_manifest_sha256": sha(manifest),
    "queue_provenance_sha256": sha(sidecar),
    "remote_memory_config": (
        {"path": str(Path(remote_memory).resolve()), "sha256": sha(remote_memory)}
        if remote_memory else None
    ),
    "system_config": {
        "path": str(Path(system_path).resolve()),
        "sha256": sha(system_path),
        "trace_enabled": trace_enabled == "1",
    },
    "python_source_stage_identity": {
        "path": str(stage_path.resolve()),
        "sha256": sha(stage_path),
        "source_head": stage["source_head"],
        "source_worktree_copy_sha256": stage["source_worktree_copy_sha256"],
        "et_def_proto_sha256": stage["et_def_proto_sha256"],
        "generated_et_def_pb2_sha256": stage["generated_et_def_pb2_sha256"],
    },
    "cpp_source_identity": {
        "path": str(cpp_source_path.resolve()),
        "sha256": sha(cpp_source_path),
        "source_head": cpp_source["source_head"],
        "source_tree": cpp_source["source_tree"],
        "source_tree_sha256": cpp_source["source_tree_sha256"],
        "source_file_count": cpp_source["source_file_count"],
    },
}
Path(destination).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
PY

BRIDGE_TIMEOUT_MS=${BRIDGE_TIMEOUT_MS:-120000}
ONLINE_VALIDATE=${SH_ONLINE_VALIDATE:-1}
BIN=${ONLINE_BIN}
CPP_PID=
cleanup_cpp() {
  if [[ -n "${CPP_PID:-}" ]] && kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
    local i
    for i in $(seq 1 30); do
      kill -0 "${CPP_PID}" 2>/dev/null || return 0
      sleep 0.1
    done
    kill -9 "${CPP_PID}" 2>/dev/null || true
  fi
}
trap cleanup_cpp EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

REMOTE_MEMORY_ARG=()
if [[ -n "${REMOTE_MEMORY_CONFIG}" ]]; then
  REMOTE_MEMORY_ARG=(--remote-memory-configuration="${REMOTE_MEMORY_CONFIG}")
fi

"${BIN}" \
  --online-mode strategy \
  --bridge-dir "${RUN_DIR}/bridge" \
  --online-validate "${ONLINE_VALIDATE}" \
  --bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}" \
  --request-queue-csv "${REQUEST_CSV}" \
  --close-input \
  --workload-configuration="${PLAN_DIR}/llama2_7b_inference" \
  --comm-group-configuration="${RUNTIME_CONFIG_DIR}/comm_group.json" \
  --system-configuration="${RUNTIME_CONFIG_DIR}/system.json" \
  "${REMOTE_MEMORY_ARG[@]}" \
  --network-configuration="${RUNTIME_CONFIG_DIR}/network.yml" \
  --logging-folder=off \
  --metrics-configuration="${METRICS_MANIFEST}" \
  --metrics-detail="${SH_METRICS_DETAIL}" \
  --logging-folder="${TRACE_LOGGING_FOLDER}" \
  > "${RUN_DIR}/cpp.log" 2>&1 &
CPP_PID=$!

deadline=$(( $(date +%s) + 120 ))
until [[ -p "${RUN_DIR}/bridge/req_notify.fifo" && -p "${RUN_DIR}/bridge/resp_notify.fifo" ]]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "C++ Online exited before bridge setup (run=${RUN_DIR})" >&2
    tail -50 "${RUN_DIR}/cpp.log" >&2 || true
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
    python3 -u "${STAGE}/sh_test_mesh/workload/llama2_7b_inference/online/online_service.py" \
      --bridge-dir "${RUN_DIR}/bridge" \
      --mode strategy \
      --plan-dir "${PLAN_DIR}" \
      --config "${TRACE_CONFIG}"
) > "${RUN_DIR}/python.log" 2>&1
PY_EXIT=$?
wait "${CPP_PID}"
CPP_EXIT=$?
CPP_PID=
trap - EXIT INT TERM
set -e

CPP_SOURCE_ID_POST=${RUN_DIR}/inputs/cpp-source-identity-post.json
capture_cpp_source_identity "${CPP_SOURCE_ID_POST}"
set +e
python3 - "${RUN_DIR}/inputs/run-input-identity.json" "${CPP_SOURCE_ID}" \
  "${CPP_SOURCE_ID_POST}" "${ONLINE_BIN}" "${RUN_DIR}/inputs/completion-identity.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

pre_identity_path, cpp_pre_path, cpp_post_path, online_path, destination = sys.argv[1:]
pre_identity = json.loads(Path(pre_identity_path).read_text(encoding="utf-8"))
cpp_pre = json.loads(Path(cpp_pre_path).read_text(encoding="utf-8"))
cpp_post = json.loads(Path(cpp_post_path).read_text(encoding="utf-8"))
digest = hashlib.sha256(Path(online_path).read_bytes()).hexdigest()
binary_before = pre_identity["online_binary"]["sha256"]
source_stable = (
    cpp_pre["source_head"] == cpp_post["source_head"]
    and cpp_pre["source_tree_sha256"] == cpp_post["source_tree_sha256"]
)
binary_stable = binary_before == digest
payload = {
    "source_head_before": cpp_pre["source_head"],
    "source_head_after": cpp_post["source_head"],
    "cpp_source_tree_sha256_before": cpp_pre["source_tree_sha256"],
    "cpp_source_tree_sha256_after": cpp_post["source_tree_sha256"],
    "cpp_source_tree_stable": source_stable,
    "online_binary_sha256_before": binary_before,
    "online_binary_sha256_after": digest,
    "online_binary_stable": binary_stable,
}
Path(destination).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
print(json.dumps(payload, sort_keys=True))
if not source_stable or not binary_stable:
    raise SystemExit("C++ source or Online binary changed during the run")
PY
IDENTITY_EXIT=$?
set -e

printf 'source=%s profile=%s cpp_exit=%s python_exit=%s run_dir=%s\n' \
  "${SOURCE_VARIANT}" "${PROFILE}" "${CPP_EXIT}" "${PY_EXIT}" "${RUN_DIR}"
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 || ${IDENTITY_EXIT} -ne 0 ]]; then
  tail -60 "${RUN_DIR}/cpp.log" >&2 || true
  tail -60 "${RUN_DIR}/python.log" >&2 || true
  exit 1
fi

python3 - "${RUN_DIR}/bridge/online_stats.jsonl" "${PROFILE}" "${SCENARIO}" "${RUN_DIR}/bridge/request_journal.jsonl" <<'PY'
import json
import sys
from pathlib import Path

stats_path = Path(sys.argv[1])
profile = sys.argv[2]
scenario = sys.argv[3]
journal_path = Path(sys.argv[4])
if not stats_path.is_file():
    raise SystemExit(f"missing Online summary: {stats_path}")
rows = [json.loads(line) for line in stats_path.read_text(encoding="utf-8").splitlines() if line.strip()]
summaries = [row for row in rows if row.get("summary") is True]
if len(summaries) != 1:
    raise SystemExit(f"expected one online_stats summary row, found {len(summaries)}")
summary = summaries[0]
if summary.get("delivery_count", 0) < 1 or summary.get("ack_count", 0) < 1:
    raise SystemExit(f"Online did not complete a real decision/ack exchange: {summary}")
disclosure = summary.get("die_predictor_disclosure")
expected = {
    "composition_profile": "face_legacy_selector",
    "approximate_predictor": True,
    "local_plan_profile": "coarse_die_v2",
    "representative_tp_rank": 0,
    "collectives_included": False,
    "explicit_queue_delay_included": False,
    "d2d_cost_profile": "fixed_latency_addend",
}
if profile == "coarse":
    if disclosure != expected:
        raise SystemExit(f"coarse predictor disclosure mismatch: {disclosure!r}")
elif disclosure is not None:
    raise SystemExit(f"legacy run must not carry coarse predictor metadata: {disclosure!r}")
result = {
    "delivery_count": summary["delivery_count"],
    "ack_count": summary["ack_count"],
    "die_predictor_disclosure": disclosure,
}
if scenario == "workspace-overlap":
    if not journal_path.is_file():
        raise SystemExit(f"missing Online request journal: {journal_path}")
    events = [
        json.loads(line)
        for line in journal_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    second_request = "face_overlap_request_1"
    first_request = "face_overlap_request_0"
    target = None
    first_prefill_completed_before_event = False
    for event in events:
        arrivals = event.get("arrivals", [])
        completed_groups = event.get("completed_groups", [])
        first_prefill_completed_in_event = any(
            row.get("request_id") == first_request and row.get("stage") == "prefill"
            for row in completed_groups
        )
        if "ARRIVAL" not in event.get("reasons", []):
            first_prefill_completed_before_event = (
                first_prefill_completed_before_event or first_prefill_completed_in_event
            )
            continue
        if not any(row.get("request_id") == second_request for row in arrivals):
            first_prefill_completed_before_event = (
                first_prefill_completed_before_event or first_prefill_completed_in_event
            )
            continue
        completed_prefill = (
            first_prefill_completed_before_event or first_prefill_completed_in_event
        )
        active = [
            row for row in event.get("workspace_state", {}).get("ranks", [])
            if row.get("active_workspace_bytes", 0) > 0
        ]
        target = {
            "tick": event.get("tick"),
            "active_rank_count": len(active),
            "active_workspace_bytes_total": sum(
                row["active_workspace_bytes"] for row in active
            ),
            "first_prefill_completed_at_arrival": completed_prefill,
        }
        if active and not completed_prefill:
            break
        first_prefill_completed_before_event = (
            first_prefill_completed_before_event or first_prefill_completed_in_event
        )
    if target is None:
        raise SystemExit(f"journal has no arrival event for {second_request}")
    if target["active_rank_count"] < 1 or target["first_prefill_completed_at_arrival"]:
        raise SystemExit(
            "workspace-overlap did not observe a live lease while the first prefill "
            f"was still open: {target}"
        )
    result["workspace_overlap"] = target
print(json.dumps(result, sort_keys=True))
PY

echo "cpp_log=${RUN_DIR}/cpp.log"
echo "python_log=${RUN_DIR}/python.log"
