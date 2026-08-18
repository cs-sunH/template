#!/usr/bin/env bash
# face 阶段 7 §10.6 官方在线 runner:legacy 第二变体(strategy 模式,真实
# 策略实时物理)。与 run_online_strategy.sh 同构,差异:
#   - online_service 显式传 --config trace_config_legacy.csv(kv_cache_policy
#     =legacy,显式不依赖代码默认值;总改造计划 §9.4);
#   - C++ 侧 workload 前缀 / metrics_manifest 指向调用方物化的 legacy 基线
#     目录(<legacy_gen>,与运行同源对照物;物化规则见方案文档 §3 步骤 0-1);
#   - 归档 kv_event_payload_legacy.json(legacy allocator run-end 终值,
#     tier_b_compare legacy 层对照输入)。
# Usage: bash run_online_strategy_legacy.sh <run_dir> <request_csv> <legacy_gen>
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=$1
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按方案文档 face仓库改造详细执行方案.md §3 步骤 0-1 物化输入后显式传入)"}

LEGACY_GEN=${3:?"legacy_gen 必填(request-neutral:请按方案文档 §3 步骤 0-1 物化 legacy 基线目录后显式传入)"}
ET_PREFIX=${LEGACY_GEN}/llama2_7b_inference
ET_DIR=$(dirname "${ET_PREFIX}")
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__no_memory_expansion
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh
CONFIG=${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config_legacy.csv

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
  --metrics-detail=summary \
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
  echo "[run_online_strategy_legacy] FAIL: bridge retained for debugging (request=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l), jsonl=$(ls "${RUN_DIR}/bridge/"*.jsonl 2>/dev/null | wc -l)); next run rm -rf clears it" >&2
fi
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]] || exit 1

# 后处理:复用 run_metrics_postprocess.sh 的能力([METRIC] 行已在 cpp.log)。
if grep -q '\[METRIC\]' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy_legacy] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
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
CP_COUNT=$(ls "${RUN_DIR}/bridge/checkpoints/"*.json 2>/dev/null | wc -l)
# Backport fix (2026-08-16, sh_2.0测试 §5.3): ls with a >2e4-entry glob
# exceeds ARG_MAX (E2BIG -> exit 126 under set -e -o pipefail); count
# via find -maxdepth 1 instead (same diagnostic value).
REQ_COUNT=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l)
echo "[run_online_strategy_legacy] artifacts: ${ARCHIVED} archived -> results/; checkpoints=${CP_COUNT}; request retained=${REQ_COUNT}"
echo "[run_online_strategy_legacy] PASS: ${RUN_DIR}"
