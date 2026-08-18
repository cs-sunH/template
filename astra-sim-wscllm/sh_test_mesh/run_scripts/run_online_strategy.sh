#!/usr/bin/env bash
# wscllm phase-1 step-1-10 官方在线 runner:strategy 模式(真实策略,实时物理)。
# Usage: bash run_online_strategy.sh <run_dir> <request_csv>
# request-neutral(裸仓库):仓库不预置输入队列;request_csv 由调用方按方案文档
# §3 步骤 0-1 物化后必填传入(缺失即 fail-closed)。
# 流程同 run_online_replay.sh(C++ 先起建桥,Python 服务后起,等退出码,
# 收日志,[METRIC] 行经 run_metrics_postprocess.sh 后处理)。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=$1
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按方案文档 wscllm仓库改造详细执行方案.md §3 步骤 0-1 物化输入后显式传入)"}

ET_PREFIX=${PROJECT}/sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_face_case5_config_c_wsc_llm_pd_6p3d_9inst_tp6_112sess_1177req_pc512_p66-169395_d1-13812_grequest_aggregated_q23936bbc_c38c794e6/llama2_7b_wsc_llm_inference
ET_DIR=$(dirname "${ET_PREFIX}")
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__no_memory_expansion
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh

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
    echo "[run_online_strategy] C++ exited before FIFO ready (see cpp.log)" >&2
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
  > "${RUN_DIR}/python.log" 2>&1 || PY_EXIT=$?

wait "${CPP_PID}"; CPP_EXIT=$?
echo "[run_online_strategy] cpp_exit=${CPP_EXIT} python_exit=${PY_EXIT}"
if [[ ${CPP_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/cpp.log" >&2
fi
if [[ ${PY_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/python.log" >&2
fi
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 ]]; then
  echo "[run_online_strategy] FAIL: bridge retained for debugging (request=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l), jsonl=$(ls "${RUN_DIR}/bridge/"*.jsonl 2>/dev/null | wc -l)); next run rm -rf clears it" >&2
fi
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]] || exit 1

# 后处理:复用 run_metrics_postprocess.sh 的能力([METRIC] 行已在 cpp.log)。
if grep -q '\[METRIC\]' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
else
  echo "[run_online_strategy] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
fi
# 阶段 7 §10.5 中间产物生命周期: 最终结果 / 检查点 / 临时产物分目录。
#  - results/    : 最终结果(审计 jsonl,Python 决策侧写出)。保留规则:每
#    run 一份,run 目录即版本,不轮转;run 脚本开头 rm -rf 保证有界。
#  - checkpoints/: C++ 窗口位置检查点(bridge/checkpoints/,run 结束原子写)。
#  - 临时产物    : response/ack 在协议层消费即删(阶段 7 §10.3);bridge/
#    保留 request_*.json(幂等重放输入 + 决策序列审计证据)与 fifo。
#  - 失败清理    : 失败时 bridge 保留为调试证据(不自动删),打印残留计数,
#    下次运行 rm -rf "${RUN_DIR}" 全量清理。
mkdir -p "${RUN_DIR}/results"
ARCHIVED=0
for j in online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
CP_COUNT=$(ls "${RUN_DIR}/bridge/checkpoints/"*.json 2>/dev/null | wc -l)
# Backport fix (2026-08-16, sh_2.0测试 §5.3): ls with a >2e4-entry glob
# exceeds ARG_MAX (E2BIG -> exit 126 under set -e -o pipefail); count
# via find -maxdepth 1 instead (same diagnostic value).
REQ_COUNT=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l)
echo "[run_online_strategy] artifacts: ${ARCHIVED} jsonl archived -> results/; checkpoints=${CP_COUNT}; request retained=${REQ_COUNT}"
echo "[run_online_strategy] PASS: ${RUN_DIR}"
