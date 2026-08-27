#!/usr/bin/env bash
# face 阶段 7 §10.6 官方在线 runner:legacy 第二变体(strategy 模式,真实
# 策略实时物理)。与 run_online_strategy.sh 同构,差异:
#   - online_service 显式传 --config trace_config_legacy.csv(kv_cache_policy
#     =legacy,显式不依赖代码默认值;总改造计划 §9.4);
#   - C++ 侧 workload 前缀 / metrics_manifest 指向调用方物化的 legacy 基线
#     目录(<legacy_gen>,与运行同源对照物;物化规则见 traces/derive_20_first_30_seconds.py);
#   - 归档 kv_event_payload_legacy.json(legacy allocator run-end 终值,
#     run-end 审计件,随 results/ 归档)。
# Usage: bash run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=$1
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按 traces/derive_20_first_30_seconds.py 物化输入后显式传入;其 stdout 即权威 provenance 记录)"}

LEGACY_GEN=${3:?"legacy_gen 必填(request-neutral:请按 traces/derive_20_first_30_seconds.py 物化 legacy 基线目录后显式传入)"}
ET_PREFIX=${LEGACY_GEN}/llama2_7b_inference
ET_DIR=$(dirname "${ET_PREFIX}")
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__no_memory_expansion
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh
CONFIG=${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config_legacy.csv

# --metrics-detail 解析（B1/WP0）：优先级 env SH_METRICS_DETAIL > 本仓
# sh_test_mesh/workload/llama2_7b_inference/metrics_config.json 的
# detail_level；json 缺失/非法/值不在 off|summary|full → 立即报错退出
# （fail-closed，不引入新依赖，python3 -c json 解析）。
if [[ -n "${SH_METRICS_DETAIL:-}" ]]; then
  DETAIL="${SH_METRICS_DETAIL}"
else
  DETAIL=$(python3 -c '
import json, sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as source:
        config = json.load(source)
except (OSError, ValueError) as error:
    sys.exit("cannot read metrics detail from %s: %s" % (path, error))
value = config.get("detail_level") if isinstance(config, dict) else None
if not isinstance(value, str) or value not in ("off", "summary", "full"):
    sys.exit(
        "metrics_config.json detail_level must be one of off|summary|full, "
        "got %r" % (value,))
print(value)
' "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/metrics_config.json") || {
    echo "[runner] FAIL: cannot resolve --metrics-detail (env SH_METRICS_DETAIL unset/unreadable or metrics_config.json detail_level missing/invalid)" >&2
    exit 1
  }
fi
if [[ "${DETAIL}" != "off" && "${DETAIL}" != "summary" && "${DETAIL}" != "full" ]]; then
  echo "[runner] FAIL: invalid SH_METRICS_DETAIL='${DETAIL}' (expected off|summary|full)" >&2
  exit 1
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
cd "${PROJECT}"

# C++ first: creates the bridge dir + FIFOs (decision_bridge contract).
"${BIN}" \
  --online-mode strategy \
  --bridge-dir "${RUN_DIR}/bridge" \
  --request-queue-csv "${REQUEST_CSV}" \
  --close-input \
  --workload-configuration="${ET_PREFIX}" \
  --comm-group-configuration="${RC}/comm_group.json" \
  --system-configuration="${RC}/system.json" \
  --remote-memory-configuration="${RC}/remote_memory.json" \
  --network-configuration="${RC}/network.yml" \
  --logging-folder=off \
  --metrics-configuration="${ET_DIR}/metrics_manifest.json" \
  --metrics-detail="${DETAIL}" \
  > "${RUN_DIR}/cpp.log" 2>&1 &
CPP_PID=$!

until [ -p "${RUN_DIR}/bridge/req_notify.fifo" ] && [ -p "${RUN_DIR}/bridge/resp_notify.fifo" ]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[run_online_strategy_legacy] C++ exited before FIFO ready (see cpp.log)" >&2
    exit 1
  fi
  sleep 0.3
done

cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
PY_EXIT=0
python3 -u online/online_service.py \
  --bridge-dir "${RUN_DIR}/bridge" \
  --mode strategy \
  --plan-dir "${ET_DIR}" \
  --config "${CONFIG}" \
  > "${RUN_DIR}/python.log" 2>&1 || PY_EXIT=$?

wait "${CPP_PID}"; CPP_EXIT=$?
echo "[run_online_strategy_legacy] cpp_exit=${CPP_EXIT} python_exit=${PY_EXIT}"
if [[ ${CPP_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/cpp.log" >&2
fi
if [[ ${PY_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/python.log" >&2
fi
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 ]]; then
  echo "[run_online_strategy_legacy] FAIL: bridge retained for debugging (request=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l), jsonl=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)); next run rm -rf clears it" >&2
fi
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]] || exit 1

# 后处理:复用 run_metrics_postprocess.sh 的能力([METRIC] 行已在 cpp.log)。
if grep -q '\[METRIC\]' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-requests="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy_legacy] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
  if [ -f "${RUN_DIR}/request_metrics.csv" ]; then
    echo "[run_online_strategy_legacy] request metrics: ${RUN_DIR}/request_metrics.csv"
  else
    echo "[run_online_strategy_legacy] request metrics: none (detail=${DETAIL}; see postprocess.log)"
  fi
else
  echo "[run_online_strategy_legacy] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
fi
# 中间产物生命周期:同 run_online_strategy.sh(结果 / 检查点 / 临时产物
# 分目录;失败保留 bridge 为调试证据)。
mkdir -p "${RUN_DIR}/results"
ARCHIVED=0
for j in online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
# 阶段 7 §10.6:legacy allocator run-end 终值(online_service 写
# bridge/kv_event_payload_legacy.json)。
if [ -f "${RUN_DIR}/bridge/kv_event_payload_legacy.json" ]; then
  mv "${RUN_DIR}/bridge/kv_event_payload_legacy.json" \
     "${RUN_DIR}/results/kv_event_payload_legacy.json"
  ARCHIVED=$((ARCHIVED + 1))
fi
CP_COUNT=$(find "${RUN_DIR}/bridge/checkpoints" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
# Backport 2026-08-16 (对比报告 §5.3): ls with a >2e4-entry glob exceeds
# ARG_MAX (E2BIG, exit 126 under set -e) -- count via find instead.
REQ_COUNT=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l)
echo "[run_online_strategy_legacy] artifacts: ${ARCHIVED} archived -> results/; checkpoints=${CP_COUNT}; request retained=${REQ_COUNT}"
echo "[run_online_strategy_legacy] PASS: ${RUN_DIR}"
