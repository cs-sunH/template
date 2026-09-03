#!/usr/bin/env bash
# sh_1.0 阶段 4 官方在线 runner:strategy 感知模式(真实策略 + 感知开关)。
# Usage: bash run_online_strategy_sensing.sh <run_dir> <request_csv>
# request-neutral(裸仓库):request_csv 由调用方按
# traces/derive_20_first_30_seconds.py 物化后必填传入(缺失即
# fail-closed)。
# 与 run_online_strategy.sh 完全同构,区别仅三处:
#   C++   追加 --sensing-enabled(阶段 4 感知 feature flag,默认关;开启后
#         每交付批次携带 per-rank injected-unfinished 账本摘要,并逐交付
#         写 remote_fifo_ledger.jsonl 快照——本仓 remote FIFO 实账本层);
#   Python 追加 --sensing(分层账本最小子集 + 两层剩余负载查询);
#   归档清单追加 remote_fifo_ledger.jsonl。
# 感知口径(方案 §6.1):感知数据 = C++ 真实执行事实驱动的查询/审计输入,
# 不进策略判据(红线 §0.4),故本 runner 的决策序列与关感知 strategy
# 逐字节一致(差异报告预期全部"无差异")。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=$1
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按 traces/derive_20_first_30_seconds.py 物化输入后显式传入;其 stdout 即权威 provenance 记录)"}

# Bare-repo restore (2026-08-16, phase 7): the baseline/20_30s archive is
# deleted; resolve the single generated dir dynamically (the dir name
# encodes this window's sess/req/p/d ranges and the trace-config digest, so
# a hardcoded path breaks whenever the config bytes change -- same fix as
# the sh_2.0 backport). Regenerate via plan_materializer.py after
# materializing the input (traces/derive_20_first_30_seconds.py; its
# stdout is the authoritative provenance record).
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (regenerate via plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_inference"
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__edge_remote_memory_pool
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online

# 桥侧看门狗（2026-08-22 引入；P0-2 2026-08-31 常态化+定位修正；收尾批
# 2026-09-01 与主 runner 统一武装——sensing 变体此前保留旧"缺省不设=永等"
# 形态）：BRIDGE_TIMEOUT_MS 三态语义（与 run_online_strategy.sh 同款）：
#   未设/空  -> 缺省 120000（campaign 常态；值必须大于本负载最慢单决策耗时，
#               且大于 Python 服务启动的 FIFO 开启等待——实际下界秒级，
#               建议 >=10000=10s，过小会在启动窗口 abort C++）；
#   显式 =0  -> 永等（冻结旧默认的逃生口：桥侧 poll 不设超时）；
#   正值     -> 该值（毫秒）。
# 定位：只武装 C++ 桥的两处 response poll（覆盖"Python 侧单次决策交换停滞"
# 族，含 Python 启动即死形态）；管不到 wait_for_work() 停泊族（本仓已无
# input-open 死端 fail-loud 分支，对齐 wscllm 合同 §2.1——由
# --idle-watchdog-s 墙钟看门狗兜住）。
# 监督纪律：终止长跑用 SIGTERM（kill <pid>），勿用 SIGINT/Ctrl-C——后者会
# 冻结健康瞬态造成"楔死"伪影（2026-08-22 诊断结论）；疑似楔死时先保留
# bridge 目录盘态与双方 /proc/<pid>/{wchan,syscall,stack} 再清理。
BRIDGE_TIMEOUT_MS="${BRIDGE_TIMEOUT_MS:-120000}"
BRIDGE_TIMEOUT_ARGS=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}")

# FP3 (2026-09-01, sync-A16 batch P; 合同 §2.3/P3)：可选透传
# SH_REQUEST_WINDOW_ROWS——与 run_online_strategy.sh 同款。缺省不传（C++
# 侧 --request-window-rows 缺省 128）。四仓对齐 wscllm（sync-A16 批次4）：
# 本仓窗口为 advisory（calendar reader 按 arrival 序提交，窗口值不改变任何
# 行为；无 C++ 启动 span 预检——不设 A1 拒绝门，合同 §3.1），0/正数照原
# token 透传仅为 CLI/checkpoint 兼容口径统一；非法 token 原样交给 C++ 统一
# fail-closed，runner 不自行吞掉。
WINDOW_ROWS_ARGS=()
if [[ -n "${SH_REQUEST_WINDOW_ROWS:-}" ]]; then
  WINDOW_ROWS_ARGS=(--request-window-rows "${SH_REQUEST_WINDOW_ROWS}")
fi
POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh

# WP0 (SLO B1, 2026-08-26): --metrics-detail 不再硬编码 summary。优先级
# env SH_METRICS_DETAIL > 本仓 sh_test_mesh/workload/llama2_7b_inference/
# metrics_config.json 的 detail_level 字段；两边取值都必须是
# off|summary|full；走 json 回退时文件缺失/非法/取值非法一律 fail-closed
# 报错退出（python3 -c 解析，不引入 jq 依赖）。与 run_online_strategy.sh
# 同构。
METRICS_CONFIG="${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/metrics_config.json"
if [[ -n "${SH_METRICS_DETAIL:-}" ]]; then
  DETAIL="${SH_METRICS_DETAIL}"
  if [[ "${DETAIL}" != "off" && "${DETAIL}" != "summary" && "${DETAIL}" != "full" ]]; then
    echo "[run_online_strategy_sensing] SH_METRICS_DETAIL must be off|summary|full, got '${DETAIL}'" >&2
    exit 1
  fi
  DETAIL_SOURCE="env SH_METRICS_DETAIL"
else
  DETAIL=$(python3 -c '
import json, sys
try:
    with open(sys.argv[1], encoding="utf-8") as f:
        cfg = json.load(f)
except Exception as exc:
    print("cannot read metrics_config.json: %s" % exc, file=sys.stderr)
    sys.exit(1)
detail = cfg.get("detail_level") if isinstance(cfg, dict) else None
if detail not in ("off", "summary", "full"):
    print("metrics_config.json detail_level must be off|summary|full, got %r" % (detail,), file=sys.stderr)
    sys.exit(1)
print(detail)
' "${METRICS_CONFIG}") || exit 1
  DETAIL_SOURCE="json ${METRICS_CONFIG}"
fi
echo "[run_online_strategy_sensing] metrics detail=${DETAIL} (source: ${DETAIL_SOURCE})"

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"
cd "${PROJECT}"

# P2(2026-08-28):per-request manifest 拷入 run_dir 根——run_dir 自包含、
# 与仓还原状态解耦(slo_common.load_request_manifest 优先级:run_dir/
# metrics_manifest.json > cpp.log init 行指向的 generated/ 副本;裸仓还原
# 会删 generated/,且仓内副本每仿真点重写、跨点不可复现)。
cp "${ET_DIR}/metrics_manifest.json" "${RUN_DIR}/metrics_manifest.json"
if [ -f "${ET_DIR}/manifest.json" ]; then
  cp "${ET_DIR}/manifest.json" "${RUN_DIR}/manifest.json"
fi

# A1/C1/D3(2026-08-28)降耗开关(runner 默认值,env 可覆盖;同
# run_online_strategy.sh)。
export ASTRA_LINK_OBSERVER="${ASTRA_LINK_OBSERVER:-0}"
# PYTHONUNBUFFERED（收尾批 2026-09-01 补，与主 runner 对齐）：python3 -u 已
# 在启动行，此处防御性冗余——确保 Python 侧异常/楔死痕迹及时落 python.log。
export PYTHONUNBUFFERED=1

# C++ first: creates the bridge dir + FIFOs (decision_bridge contract).
"${BIN}" \
  --online-mode strategy \
  --sensing-enabled \
  --bridge-dir "${RUN_DIR}/bridge" \
  --online-node-gc "${SH_ONLINE_NODE_GC:-1}" \
  --online-validate "${SH_ONLINE_VALIDATE:-0}" \
  "${BRIDGE_TIMEOUT_ARGS[@]}" \
  "${WINDOW_ROWS_ARGS[@]}" \
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
    --out-request="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy_sensing] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
else
  echo "[run_online_strategy_sensing] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
fi
# 阶段 7 §10.5 中间产物生命周期:同 run_online_strategy.sh;本 runner 追加
# remote_fifo_ledger.jsonl(remote FIFO 实账本层逐交付快照,C++ 侧写出)。
mkdir -p "${RUN_DIR}/results"
ARCHIVED=0
for j in request_journal online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log remote_fifo_ledger train_ledger; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
CP_COUNT=$(find "${RUN_DIR}/bridge/checkpoints" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
echo "[run_online_strategy_sensing] artifacts: ${ARCHIVED} jsonl archived -> results/; checkpoints=${CP_COUNT}; request_journal=results/request_journal.jsonl"
# P3(2026-08-28):仿真成功后自动 SLO 指标提取(postprocess 成功之后、
# archive_run_outputs.sh 之前:cpp.log 未压缩、manifest 已拷入、results/
# 已归位,输入全齐;产物随保留集常驻,最细粒度纪律见脚本头注)。
# SH_SLO_POSTPROCESS: 1=默认 warn(子命令失败只写 slo_postprocess.FAIL,
# 不推翻仿真结果); 0=整步跳过; strict=失败即本 runner 非零退出。
SLO_MODE="${SH_SLO_POSTPROCESS:-1}"
if [[ "${SLO_MODE}" != "0" ]]; then
  if ! bash "${SCRIPT_DIR}/run_slo_postprocess.sh" "${RUN_DIR}"; then
    if [[ "${SLO_MODE}" == "strict" ]]; then
      echo "[run_online_strategy_sensing] FAIL: SLO postprocess failed (SH_SLO_POSTPROCESS=strict)" >&2
      exit 1
    fi
    echo "[run_online_strategy_sensing] WARN: SLO postprocess failures flagged in ${RUN_DIR}/slo_postprocess.FAIL (warn mode)" >&2
  fi
fi
# D1(2026-08-28):成功后产物瘦身归档(SH_ARCHIVE_RUN=0 关闭)。
if [[ "${SH_ARCHIVE_RUN:-1}" != "0" ]]; then
  bash "${SCRIPT_DIR}/archive_run_outputs.sh" "${RUN_DIR}" || exit 1
fi
echo "[run_online_strategy_sensing] PASS: ${RUN_DIR}"
