#!/usr/bin/env bash
# face phase-3 官方在线 runner:strategy 感知模式(真实策略 + 感知开关)。
# Usage: bash run_online_strategy_sensing.sh <run_dir> <request_csv>
# request-neutral(裸仓库):request_csv 由调用方按 traces/derive_20_first_30_seconds.py
# 物化后必填传入(缺失即 fail-closed)。
# 与 run_online_strategy.sh 完全同构,唯一区别:
#   C++   追加 --sensing-enabled(阶段 3 感知 feature flag,默认关;
#         开启后每交付批次携带 per-rank injected-unfinished 账本摘要);
#   Python 追加 --sensing(分层账本最小子集 + 两层剩余负载查询)。
# 感知口径(方案 §6.1):感知数据 = C++ 真实执行事实驱动的查询/审计输入,
# 不进策略判据(红线 §0.4),故本 runner 的决策序列与关感知逐字节一致
# (差异报告预期全部"无差异")。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=$1
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按 traces/derive_20_first_30_seconds.py 物化输入后显式传入;其 stdout 即权威 provenance 记录)"}

# ET 基线目录 = GEN_MATCH 动态解析(五仓统一口径)——恰好一个 llama2_7b_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_inference"
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__no_memory_expansion
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online

# 长跑楔死看门狗（2026-08-22 楔死诊断建议，可选）：设 BRIDGE_TIMEOUT_MS
# 为正毫秒数时向 C++ 桥传 --bridge-timeout-ms——Python 决策侧停滞超时即
# fail-closed abort（cpp.log 出现 Python side died 行），替代"永等+外部盲杀"。
# 值必须大于本负载最慢单决策耗时，且大于 Python 服务启动的 FIFO 开启
# 等待（实际下界是秒级，建议 ≥10000=10s——过小会在启动窗口 abort C++，
# Python 侧将阻塞在 fifo open）；缺省不设 = 永等（冻结默认）。
# 监督纪律：终止长跑用 SIGTERM（kill <pid>），勿用 SIGINT/Ctrl-C——后者会
# 冻结健康瞬态造成"楔死"伪影（2026-08-22 诊断结论）；疑似楔死时先保留
# bridge 目录盘态与双方 /proc/<pid>/stack 再清理。
BRIDGE_TIMEOUT_ARGS=()
if [[ -n "${BRIDGE_TIMEOUT_MS:-}" ]]; then
  BRIDGE_TIMEOUT_ARGS=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}")
fi
POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh

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
  --sensing-enabled \
  --bridge-dir "${RUN_DIR}/bridge" \
  "${BRIDGE_TIMEOUT_ARGS[@]}" \
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
    echo "[run_online_strategy_sensing] C++ exited before FIFO ready (see cpp.log)" >&2
    exit 1
  fi
  sleep 0.3
done

cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
PY_EXIT=0
python3 -u online/online_service.py \
  --bridge-dir "${RUN_DIR}/bridge" \
  --mode strategy \
  --sensing \
  --plan-dir "${ET_DIR}" \
  > "${RUN_DIR}/python.log" 2>&1 || PY_EXIT=$?

wait "${CPP_PID}"; CPP_EXIT=$?
echo "[run_online_strategy_sensing] cpp_exit=${CPP_EXIT} python_exit=${PY_EXIT}"
if [[ ${CPP_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/cpp.log" >&2
fi
if [[ ${PY_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/python.log" >&2
fi
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 ]]; then
  echo "[run_online_strategy_sensing] FAIL: bridge retained for debugging (request=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l), jsonl=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)); next run rm -rf clears it" >&2
fi
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]] || exit 1

# 后处理:复用 run_metrics_postprocess.sh 的能力([METRIC] 行已在 cpp.log)。
if grep -q '\[METRIC\]' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-requests="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy_sensing] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
  if [ -f "${RUN_DIR}/request_metrics.csv" ]; then
    echo "[run_online_strategy_sensing] request metrics: ${RUN_DIR}/request_metrics.csv"
  else
    echo "[run_online_strategy_sensing] request metrics: none (detail=${DETAIL}; see postprocess.log)"
  fi
else
  echo "[run_online_strategy_sensing] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
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
for j in online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log train_ledger; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
CP_COUNT=$(find "${RUN_DIR}/bridge/checkpoints" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
# Backport 2026-08-16 (对比报告 §5.3): ls with a >2e4-entry glob exceeds
# ARG_MAX (E2BIG, exit 126 under set -e) -- count via find instead.
REQ_COUNT=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l)
echo "[run_online_strategy_sensing] artifacts: ${ARCHIVED} jsonl archived -> results/; checkpoints=${CP_COUNT}; request retained=${REQ_COUNT}"
echo "[run_online_strategy_sensing] PASS: ${RUN_DIR}"
