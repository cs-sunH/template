#!/usr/bin/env bash
# wscllm relevant_distributed 第三变体官方在线 runner(strategy 模式,真实
# 策略实时物理)。与 run_online_strategy.sh 同构(GEN_MATCH/看门狗/后处理/
# SLO 提取/归档全部照抄),差异仅两处:
#   - policy 传递沿用 legacy 脚本机制(online_service 显式 --config,不依赖
#     代码默认值;裁决 #21):本脚本从当前 trace_config.csv 派生 run 内配置
#     ${RUN_DIR}/trace_config_relevant.csv——kv_cache_policy 一律改写为
#     relevant_distributed(缺省 csv 的 session_lru_recompute 与代码 fallback
#     "legacy" 两处均不动),request_queue_csv 等其余行原样保留(相对路径由
#     loader 按 sh_test_mesh/workload 解析,与配置文件所在位置无关);
#   - KV_REMOTE_READ 环境变量透传新 optional 键 kv_remote_read
#     (physical|ideal_masked,缺省 physical):physical=发射 3300 读边+裁
#     远程 KV 分量;ideal_masked=不发 3300、不裁分量(现状字节口径)——
#     两者构成严格 A/B 对照(总文档 §3.2/裁决 #11,H28)。
# 归档差异:额外归档 kv_event_payload_relevant.json(若调度器产出;裁决 #23,
# 产物名若与 B3 实现有出入待 B4b 对齐,缺失不是错误)。
#   - SH_RUNTIME_CONFIG_DIR(缺省不设):覆盖 C++ runtime_config 四小件目录
#     (H28 d2d/hbm ratio 扫描的派生变体目录;缺省 = 主 hardware json 的
#     face_case5_config_c__validation-160gib__no_memory_expansion,行为不变)。
# Usage: bash run_online_strategy_relevant.sh <run_dir> <request_csv>
# request-neutral(裸仓库):request_csv 由调用方按 traces/derive_20_first_30_seconds.py
# 物化后必填传入(缺失即 fail-closed)。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=$1
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按 traces/derive_20_first_30_seconds.py 物化输入后显式传入;其 stdout 即权威 provenance 记录)"}

# KV_REMOTE_READ 三态语义(缺省 physical):
#   physical     = 3300 远程读边照发 + 列车体裁远程 KV 分量(策略真实行为);
#   ideal_masked = 不发 3300、不裁分量(理想对照臂,与 physical 构成严格 A/B)。
# 非法值 fail-closed(与 generate_wsc_llm_trace 的键值域校验同款)。
KV_REMOTE_READ="${KV_REMOTE_READ:-physical}"
if [[ "${KV_REMOTE_READ}" != "physical" && "${KV_REMOTE_READ}" != "ideal_masked" ]]; then
  echo "[run_online_strategy_relevant] FAIL: invalid KV_REMOTE_READ='${KV_REMOTE_READ}' (expected physical|ideal_masked)" >&2
  exit 1
fi

# ET 基线目录 = GEN_MATCH 动态解析(五仓统一口径)——恰好一个 llama2_7b_wsc_llm_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_wsc_llm_inference"
# RC = C++ 四小件 runtime_config 目录。缺省 = 主 hardware json 派生目录
# (config_resolver.materialize_runtime_configs 的 slug 命名)。H28 掩盖
# 边界扫描(总文档 §4 裁决 #22/#28:掩盖边界走 hardware json 派生 + ratio
# 扫描,不进冻结参数)用 SH_RUNTIME_CONFIG_DIR 覆盖为派生变体目录——变体
# hardware json 放独立目录、不改主 json;缺省不设 = 行为与既有 runner
# 逐字节一致。目录必须含四小件,缺失由 C++ fail-closed。
RC="${SH_RUNTIME_CONFIG_DIR:-${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__no_memory_expansion}"
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
WL_CONFIG=${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv
POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh

# 桥侧看门狗（2026-08-22 引入；P0-2 2026-08-31 常态化+定位修正）：向 C++ 桥传
# --bridge-timeout-ms——Python 决策侧单次交换停滞超时即 fail-closed abort
# （cpp.log 出现 Python side died 行），替代"永等+外部盲杀"。
# BRIDGE_TIMEOUT_MS 三态语义（与 run_online_strategy.sh 同款）：
#   未设/空  -> 缺省 120000（campaign 常态；值必须大于本负载最慢单决策耗时，
#               且大于 Python 服务启动的 FIFO 开启等待——实际下界是秒级，
#               建议 >=10000=10s，过小会在启动窗口 abort C++）；
#   显式 =0  -> 永等（冻结旧默认的逃生口：桥侧 poll 不设超时）；
#   正值     -> 该值（毫秒）。
# 定位：本看门狗只武装 C++ 桥的两处 response poll（DecisionBridge.cc），
# 覆盖"Python 侧单次决策交换停滞"族；它管不到 C++ 主循环 wait_for_work()
# 停泊族——后者由 --idle-watchdog-s 墙钟停泊看门狗兜住（缺省关）。
# 监督纪律：终止长跑用 SIGTERM（kill <pid>），勿用 SIGINT/Ctrl-C——后者会
# 冻结健康瞬态造成"楔死"伪影（2026-08-22 诊断结论）；疑似楔死时先保留
# bridge 目录盘态与双方 /proc/<pid>/{wchan,syscall,stack} 再清理。
BRIDGE_TIMEOUT_MS="${BRIDGE_TIMEOUT_MS:-120000}"
BRIDGE_TIMEOUT_ARGS=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}")

# P0-2 (2026-08-31)：可选透传 SH_REQUEST_WINDOW_ROWS——缺省不传（C++ 侧
# --request-window-rows 缺省 128）。wscllm：本仓窗口为 advisory（calendar
# reader 按 arrival 序提交，窗口值不改变任何行为；无 C++ 启动 span 预检），
# 0/正数照原 token 透传仅为 CLI/checkpoint 兼容口径统一；非法 token 原样
# 交给 C++ 统一 fail-closed，runner 不自行吞掉。
WINDOW_ROWS_ARGS=()
if [[ -n "${SH_REQUEST_WINDOW_ROWS:-}" ]]; then
  WINDOW_ROWS_ARGS=(--request-window-rows "${SH_REQUEST_WINDOW_ROWS}")
fi

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

# policy 派生配置（legacy 脚本的 --config 显式传递机制 + 裁决 #21：缺省 csv
# 与代码 fallback 两处不动）：从当前 trace_config.csv（已由调用方指向物化
# 队列）逐行派生——kv_cache_policy 值改写 relevant_distributed（恰一行，
# fail-closed）、剔除既有 kv_remote_read 行、追加 kv_remote_read=${KV_REMOTE_READ}
# 行（loader 对行序无要求、重复键 fail-closed）。产物落 RUN_DIR（调用方
# 自包含，随 run 保留；不引入 clean 脚本枚举外的仓内产物位置）。
DERIVED_CONFIG="${RUN_DIR}/trace_config_relevant.csv"
python3 - "${WL_CONFIG}" "${DERIVED_CONFIG}" "${KV_REMOTE_READ}" <<'PY'
import sys

src_path, dst_path, remote_read = sys.argv[1], sys.argv[2], sys.argv[3]
with open(src_path, encoding="utf-8") as source:
    lines = source.read().splitlines(keepends=True)
out = []
policy_rows = 0
for line in lines:
    stripped = line.rstrip("\n").rstrip("\r")
    if stripped.startswith("config,kv_cache_policy,"):
        parts = stripped.split(",")
        if len(parts) < 3:
            sys.exit("malformed kv_cache_policy row: %r" % stripped)
        parts[2] = "relevant_distributed"
        line = ",".join(parts) + ("\n" if line.endswith("\n") else "")
        policy_rows += 1
        out.append(line)
        continue
    if stripped.startswith("config,kv_remote_read,"):
        continue  # 单值键：派生行在文件尾统一重发（loader 拒重复键）
    out.append(line)
if policy_rows != 1:
    sys.exit(
        "expected exactly one config,kv_cache_policy row in %s, found %d"
        % (src_path, policy_rows))
out.append(
    "config,kv_remote_read,%s,,,,,derived by run_online_strategy_relevant.sh "
    "(relevant_distributed A/B; default physical)\n" % remote_read)
with open(dst_path, "w", encoding="utf-8") as sink:
    sink.write("".join(out))
PY
echo "[run_online_strategy_relevant] derived config: ${DERIVED_CONFIG} (kv_cache_policy=relevant_distributed, kv_remote_read=${KV_REMOTE_READ})"

# P2(2026-08-28):per-request manifest 拷入 run_dir 根——run_dir 自包含、
# 与仓还原状态解耦(slo_common.load_request_manifest 优先级:run_dir/
# metrics_manifest.json > cpp.log init 行指向的 generated/ 副本;裸仓还原
# 会删 generated/,且仓内副本每仿真点重写、跨点不可复现)。
cp "${ET_DIR}/metrics_manifest.json" "${RUN_DIR}/metrics_manifest.json"
if [ -f "${ET_DIR}/manifest.json" ]; then
  cp "${ET_DIR}/manifest.json" "${RUN_DIR}/manifest.json"
fi

# A1/C1/D3(2026-08-28)三个降耗开关(runner 默认值,env 可覆盖;同
# run_online_strategy.sh)。relevant 变体 Python 侧装配口径同 session_lru
# (全 sink + journal recorder,online_service 分发分支)。
export ASTRA_LINK_OBSERVER="${ASTRA_LINK_OBSERVER:-0}"
export PYTHONUNBUFFERED=1

# C++ first: creates the bridge dir + FIFOs (decision_bridge contract).
"${BIN}" \
  --online-mode strategy \
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
    echo "[run_online_strategy_relevant] C++ exited before FIFO ready (see cpp.log)" >&2
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
  --config "${DERIVED_CONFIG}" \
  > "${RUN_DIR}/python.log" 2>&1 || PY_EXIT=$?

wait "${CPP_PID}"; CPP_EXIT=$?
echo "[run_online_strategy_relevant] cpp_exit=${CPP_EXIT} python_exit=${PY_EXIT}"
if [[ ${CPP_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/cpp.log" >&2
fi
if [[ ${PY_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/python.log" >&2
fi
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 ]]; then
  echo "[run_online_strategy_relevant] FAIL: bridge retained for debugging (request=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l), jsonl=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)); next run rm -rf clears it" >&2
fi
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]] || exit 1

# 后处理:复用 run_metrics_postprocess.sh 的能力([METRIC] 行已在 cpp.log)。
if grep -q '\[METRIC\]' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-requests="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy_relevant] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
  if [ -f "${RUN_DIR}/request_metrics.csv" ]; then
    echo "[run_online_strategy_relevant] request metrics: ${RUN_DIR}/request_metrics.csv"
  else
    echo "[run_online_strategy_relevant] request metrics: none (detail=${DETAIL}; see postprocess.log)"
  fi
else
  echo "[run_online_strategy_relevant] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
fi
# 中间产物生命周期(同 run_online_strategy.sh:结果/检查点/临时产物分目录;
# 失败保留 bridge 为调试证据)。
mkdir -p "${RUN_DIR}/results"
ARCHIVED=0
for j in request_journal online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log train_ledger kv_delta_journal; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
# kv delta journal 的 run 末 checksum 门产物(journal 开关 off 时不存在,
# 缺文件不是错误;relevant 变体装配口径同 session_lru,KV_DELTA_JOURNAL=on
# 时产出)。
if [ -f "${RUN_DIR}/bridge/kv_delta_journal_checksum.json" ]; then
  mv "${RUN_DIR}/bridge/kv_delta_journal_checksum.json" \
     "${RUN_DIR}/results/kv_delta_journal_checksum.json"
fi
# relevant 变体 run-end 终值(裁决 #23;若 B3 实际产物名有出入待 B4b 对齐,
# 缺失不是错误)。
if [ -f "${RUN_DIR}/bridge/kv_event_payload_relevant.json" ]; then
  mv "${RUN_DIR}/bridge/kv_event_payload_relevant.json" \
     "${RUN_DIR}/results/kv_event_payload_relevant.json"
  ARCHIVED=$((ARCHIVED + 1))
fi
# Backport 2026-08-16 (对比报告 §5.3): ls with a >2e4-entry glob exceeds
# ARG_MAX (E2BIG, exit 126 under set -e) -- count via find instead.
CP_COUNT=$(find "${RUN_DIR}/bridge/checkpoints" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
echo "[run_online_strategy_relevant] artifacts: ${ARCHIVED} jsonl archived -> results/; checkpoints=${CP_COUNT}; request_journal=results/request_journal.jsonl"
# P3(2026-08-28):仿真成功后自动 SLO 指标提取(postprocess 成功之后、
# archive_run_outputs.sh 之前:cpp.log 未压缩、manifest 已拷入、results/
# 已归位,输入全齐;产物随保留集常驻,最细粒度纪律见脚本头注)。
# SH_SLO_POSTPROCESS: 1=默认 warn(子命令失败只写 slo_postprocess.FAIL,
# 不推翻仿真结果); 0=整步跳过; strict=失败即本 runner 非零退出。
SLO_MODE="${SH_SLO_POSTPROCESS:-1}"
if [[ "${SLO_MODE}" != "0" ]]; then
  if ! bash "${SCRIPT_DIR}/run_slo_postprocess.sh" "${RUN_DIR}"; then
    if [[ "${SLO_MODE}" == "strict" ]]; then
      echo "[run_online_strategy_relevant] FAIL: SLO postprocess failed (SH_SLO_POSTPROCESS=strict)" >&2
      exit 1
    fi
    echo "[run_online_strategy_relevant] WARN: SLO postprocess failures flagged in ${RUN_DIR}/slo_postprocess.FAIL (warn mode)" >&2
  fi
fi
# D1(2026-08-28):成功后产物瘦身归档(失败路径早已 exit 1 全量保留)。
# SH_ARCHIVE_RUN=0 关闭(调试/对拍需要散装 bridge 文件时)。
if [[ "${SH_ARCHIVE_RUN:-1}" != "0" ]]; then
  bash "${SCRIPT_DIR}/archive_run_outputs.sh" "${RUN_DIR}" || exit 1
fi
echo "[run_online_strategy_relevant] PASS: ${RUN_DIR}"
