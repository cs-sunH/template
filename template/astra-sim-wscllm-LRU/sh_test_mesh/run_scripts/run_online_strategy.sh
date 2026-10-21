#!/usr/bin/env bash
# wscllm phase-1 step-1-10 官方在线 runner:strategy 模式(真实策略,实时物理)。
# Usage: bash run_online_strategy.sh <run_dir> <request_csv>
# request-neutral(裸仓库):仓库不预置输入队列;request_csv 由调用方按
# sh_test_mesh/workload/llama2_7b_inference/traces/derive_20_first_30_seconds.py 物化后必填传入(缺失即 fail-closed)。
# 流程(C++ 先起建桥,Python 服务后起,等退出码,
# 收日志,[METRIC] 行经 run_metrics_postprocess.sh 后处理)。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")

RUN_DIR=${1:?"usage: bash run_online_strategy.sh <run_dir> <request_csv>"}
# infra-E3：RUN_DIR 规范化为绝对路径——本脚本启动 Python 服务前会 cd 进
# workload 目录且不回退，相对路径 RUN_DIR 在 cd 后会解析到 workload 目录下的
# 另一位置（bridge FIFO、cpp/python 日志与全部后处理产物随之错位）。
RUN_DIR=$(realpath -m "${RUN_DIR}")
# infra-F1：RUN_DIR 完全来自调用方 $1，realpath -m 不校验危险性——下方
# rm -rf 前拒绝顶级系统目录、$HOME、仓根及其祖先、以及仓内路径（campaign
# run_dir 的正当落点在仓外 ../runs/）。
case ":/bin:/boot:/dev:/etc:/home:/lib:/lib32:/lib64:/media:/mnt:/opt:/proc:/root:/run:/sbin:/srv:/sys:/tmp:/usr:/var:" in
  *:"${RUN_DIR}":*)
    echo "[runner] refusing dangerous RUN_DIR: ${RUN_DIR}" >&2
    exit 1
    ;;
esac
if [[ -z "${RUN_DIR}" || "${RUN_DIR}" == "/" || "${RUN_DIR}" == "${HOME}" \
      || "${PROJECT}" == "${RUN_DIR}" || "${PROJECT}" == "${RUN_DIR}/"* \
      || "${RUN_DIR}" == "${PROJECT}"/* ]]; then
  echo "[runner] refusing dangerous RUN_DIR: ${RUN_DIR}" >&2
  exit 1
fi
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按 sh_test_mesh/workload/llama2_7b_inference/traces/derive_20_first_30_seconds.py 物化输入后显式传入;其 stdout 即权威 provenance 记录)"}
# R3-1(2026-09-30):绝对化,使下方 D2 同一性闸与 C++ 侧 --request-queue-csv
# 在任何调用 CWD 下解析到同一文件(旧形态透传原始串,相对路径在 cd 后错位)。
REQUEST_CSV=$(realpath -m "${REQUEST_CSV}")

# D2 双源同一性闸(2026-09-30 审查):C++ 侧吃本 runner 第二参
# (--request-queue-csv),Python 侧经 trace_config.csv 的 request_queue_csv
# 指针产出 manifest(h_hist/h_pre/h_fin/interval 均由其派生)——同一事实
# 的两份拷贝无交叉校验时,id 相交而内容不同会静默分叉(到达时刻取 C++
# 文件、派生量取 Python manifest,SLO e2e 的到达模型互相矛盾)。此处
# 用仓内 loader 解析指针,归一化后与实参不同路径即 fail-closed。
# R2-5(2026-09-30):if ! 形态——set -e 下"先赋值后查 $?"的旧形态会在命令
# 替换失败时直接退出,丢失定制报错(退出码仍 1,fail-closed 不破)。
if ! PY_CSV_PTR=$(cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference" && python3 -c '
import sys
sys.path.insert(0, ".")
from generate_wsc_llm_trace import load_wsc_llm_trace_config
print(load_wsc_llm_trace_config().request_queue_csv.resolve())'); then
  echo "[runner] cannot resolve trace_config.csv request_queue_csv pointer (loader fail-closed; loader error printed above)" >&2
  exit 1
fi
if [[ "$(realpath -m "${REQUEST_CSV}")" != "$(realpath -m "${PY_CSV_PTR}")" ]]; then
  echo "[runner] request csv dual-source mismatch: argv=${REQUEST_CSV} vs trace_config.csv=${PY_CSV_PTR} — same fact must come from one file (fail-closed)" >&2
  exit 1
fi

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_wsc_llm_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 sh_test_mesh/workload/llama2_7b_inference/traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_wsc_llm_inference"
# RC = runtime_config 目录动态解析（sync-py-E11，2026-09-30 face-LRU 同款
# 移植，原为硬编码目录名）：目录名由配置装载器生成为 "<hardware slug>__<
# capacity_profile>__<remote-memory 标签>"（generate_wsc_llm_trace ->
# config_resolver.materialize_runtime_configs；本仓 hardware json 的
# remote-memory.npu-selection=mesh-boundary → 标签恒 edge_remote_memory_
# pool）。capacity_profile 权威同源 = trace_config.csv 的 config,local_
# hbm_capacity_profile 行（物化即按它落目录名，与 generate_wsc_llm_trace
# 消费同一文件）。按该 profile 过滤 generated/runtime_config/ 并要求恰好
# 一个（与上方 GEN_MATCH 的 exactly-one 口径一致），缺失/歧义 fail-closed
# ——旧 profile 残留时硬编码名会静默加载错误容量档。
RC_PROFILE=$(python3 -c '
import csv, sys
path = sys.argv[1]
with open(path, newline="", encoding="utf-8-sig") as source:
    rows = list(csv.reader(source))
hits = [row[2].strip() for row in rows
        if len(row) >= 3 and row[0].strip() == "config"
        and row[1].strip() == "local_hbm_capacity_profile"]
if len(hits) != 1 or not hits[0]:
    sys.exit("expected exactly one config,local_hbm_capacity_profile row in %s" % path)
print(hits[0])
' "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv") || {
  echo "[runner] FAIL: cannot read local_hbm_capacity_profile from trace_config.csv (sync-py-E11)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__edge_remote_memory_pool)
# RC 显式存在性闸(对齐上方 GEN_MATCH 闸;2026-09-26 引入,sync-py-E11 起含
# 四小件校验):runtime_config 四小件由 plan_materializer.py 经
# config_resolver 物化于 generated/runtime_config/(裸仓还原会删
# generated/)——缺失/歧义即在此报错,不等 C++ 打开 system.json 才失败。
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" \
      || ! -f "${RC_MATCH[0]}/remote_memory.json" ]]; then
  echo "[runner] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json/remote_memory.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online

# 桥侧看门狗（2026-08-22 引入；P0-2 2026-08-31 常态化+定位修正）：向 C++ 桥传
# --bridge-timeout-ms——Python 决策侧单次交换停滞超时即 fail-closed abort
# （cpp.log 出现 Python side died 行），替代"永等+外部盲杀"。
# BRIDGE_TIMEOUT_MS 三态语义：
#   未设/空  -> 缺省 120000（campaign 常态；值必须大于本负载最慢单决策耗时，
#               且大于 Python 服务启动的 FIFO 开启等待——实际下界是秒级，
#               建议 >=10000=10s，过小会在启动窗口 abort C++）；
#   显式 =0  -> 永等（冻结旧默认的逃生口：桥侧 poll 不设超时）；
#   正值     -> 该值（毫秒）。
# 定位（修正 2026-08-22 楔死报告 §6.2 的覆盖面误标）：本看门狗只武装 C++ 桥的
# 两处 response poll（DecisionBridge.cc），覆盖"Python 侧单次决策交换停滞"族；
# 它管不到 C++ 主循环 wait_for_work() 停泊族（2026-08-31 实锤的 41 分钟静默
# 楔死形态）——后者由 --idle-watchdog-s 墙钟停泊看门狗兜住（wscllm 无
# input-open 死端分支，合同 §2.1；2026-09-05 起缺省 1s 武装——官方
# CSV run 的到达全量预排为队列事件，健康运行不停车、不受影响；静默楔死将在
# ~1s 转为 fail-closed abort），不在本变量职责内。
# 监督纪律：终止长跑用 SIGTERM（kill <pid>），勿用 SIGINT/Ctrl-C——后者会
# 冻结健康瞬态造成"楔死"伪影（2026-08-22 诊断结论）；疑似楔死时先保留
# bridge 目录盘态与双方 /proc/<pid>/{wchan,syscall,stack} 再清理。
BRIDGE_TIMEOUT_MS="${BRIDGE_TIMEOUT_MS:-120000}"
BRIDGE_TIMEOUT_ARGS=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}")

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

# P2(2026-08-28):per-request manifest 拷入 run_dir 根——run_dir 自包含、
# 与仓还原状态解耦(slo_common.load_request_manifest 优先级:run_dir/
# metrics_manifest.json > cpp.log init 行指向的 generated/ 副本;裸仓还原
# 会删 generated/,且仓内副本每仿真点重写、跨点不可复现)。
cp "${ET_DIR}/metrics_manifest.json" "${RUN_DIR}/metrics_manifest.json"
if [ -f "${ET_DIR}/manifest.json" ]; then
  cp "${ET_DIR}/manifest.json" "${RUN_DIR}/manifest.json"
fi
# P4(2026-10-01):trace_config.csv 运行时快照拷入 run_dir 根——load_imbalance
# 的 CV 分母经 count_configured_instances 只认 run_dir 本地副本、不回退仓内
# 默认(hbm_watermark.resolve_trace_config_path 的 allow_repo_default=False,
# METRIC-01 防改布局 run 的分母静默膨胀);runner 此前不落这份副本,官方链路
# 的 load_imbalance 步必然 fail-closed。命名用 B0 归档名
# trace_config.csv.snapshot;拷贝点在 D2 闸之后,快照即经闸验证的
# "argv==指针"那份配置,与仓还原状态解耦(同上方 P2 的 run_dir 自包含
# rationale;归档步不触碰 run_dir 根级文件,快照常驻)。
cp "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv" \
   "${RUN_DIR}/trace_config.csv.snapshot"

# A1/C1/D3(2026-08-28)降耗开关(runner 默认值,env 可覆盖;B.3 清除 2026-09-05
#   移除 --online-node-gc 旗标——M2 节点 GC 摊销回收恒开,不可禁用):
#   --online-validate  默认 0(生产关全量校验;冒烟/对拍显式 SH_ONLINE_VALIDATE=1);
#   ASTRA_LINK_OBSERVER 默认 0(在线模式 link_bucket/link_total 行无消费者;
#                       metrics_postprocess 不读,hopbytes 走 decision log;
#                       =1 恢复发射)。
export ASTRA_LINK_OBSERVER="${ASTRA_LINK_OBSERVER:-0}"

# P0-2 (2026-08-31, 总文档 §4 P0-2.5): Python 侧无缓冲输出——下方启动行已是
# python3 -u，此处再 export PYTHONUNBUFFERED=1 属防御性冗余：确保本 runner
# 后续拉起的任何非 -u python 子进程（诊断/后处理脚本）也不再 block-buffer
# stdout。2026-08-31 楔死现场 python.log 0 字节即 block-buffered stdout 吞掉
# 了 Python 侧全部痕迹，楔死归因被迫全靠盘态法证。
export PYTHONUNBUFFERED=1

# C++ first: creates the bridge dir + FIFOs (decision_bridge contract).
"${BIN}" \
  --online-mode strategy \
  --bridge-dir "${RUN_DIR}/bridge" \
  --online-validate "${SH_ONLINE_VALIDATE:-0}" \
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

# FIFO 就绪等待带墙钟预算（infra-F2）：上面的 kill -0 只防 C++ 进程退出，
# 防不了"进程存活但迟迟不建 FIFO"的窗口；超时打印 cpp.log 尾部后
# fail-closed（SH_FIFO_WAIT_S 可覆盖，缺省 60s）。
fifo_deadline=$(( $(date +%s) + ${SH_FIFO_WAIT_S:-60} ))
until [ -p "${RUN_DIR}/bridge/req_notify.fifo" ] && [ -p "${RUN_DIR}/bridge/resp_notify.fifo" ]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[run_online_strategy] C++ exited before FIFO ready (see cpp.log)" >&2
    exit 1
  fi
  if (( $(date +%s) > fifo_deadline )); then
    echo "[run_online_strategy] FIFOs not ready within ${SH_FIFO_WAIT_S:-60}s (see cpp.log tail)" >&2
    tail -5 "${RUN_DIR}/cpp.log" >&2 || true
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

# 成功路径 wait 不触发 ||，须先置 0（与上方 PY_EXIT 同款），
# 否则 set -u 下引用未绑定 CPP_EXIT 直接终止脚本。
CPP_EXIT=0
wait "${CPP_PID}" || CPP_EXIT=$?
echo "[run_online_strategy] cpp_exit=${CPP_EXIT} python_exit=${PY_EXIT}"
if [[ ${CPP_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/cpp.log" >&2
fi
if [[ ${PY_EXIT} -ne 0 ]]; then
  tail -5 "${RUN_DIR}/python.log" >&2
fi
if [[ ${CPP_EXIT} -ne 0 || ${PY_EXIT} -ne 0 ]]; then
  echo "[run_online_strategy] FAIL: bridge retained for debugging (request=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' 2>/dev/null | wc -l), jsonl=$(find "${RUN_DIR}/bridge" -maxdepth 1 -name '*.jsonl' 2>/dev/null | wc -l)); next run rm -rf clears it" >&2
fi
[[ ${CPP_EXIT} -eq 0 && ${PY_EXIT} -eq 0 ]] || exit 1

# 后处理:复用 run_metrics_postprocess.sh 的能力([METRIC] 行已在 cpp.log)。
# 锚定探测（2026-10-01 对齐）：C++ 的 [METRIC][ERROR]/[METRIC][FATAL]
# 诊断行不是指标记录——非锚定 grep 会被其误命中；与归档/SLO 层的
# '^\[METRIC\] ' 同款锚定。
if grep -q '^\[METRIC\] ' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-requests="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
  if [ -f "${RUN_DIR}/request_metrics.csv" ]; then
    echo "[run_online_strategy] request metrics: ${RUN_DIR}/request_metrics.csv"
  else
    echo "[run_online_strategy] request metrics: none (detail=${DETAIL}; see postprocess.log)"
  fi
else
  # off 档无 [METRIC] 行属正常设计(与 run_slo_postprocess.sh 的 by-design
  # 口径一致,归档步 archive_run_outputs.sh 同样豁免);summary/full 档无行
  # 才是异常,保留 WARNING(归档步会 fail-closed)。
  if [ "${DETAIL}" = "off" ]; then
    echo "[run_online_strategy] no [METRIC] lines (metrics detail=off, by design); postprocess skipped"
  else
    echo "[run_online_strategy] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
  fi
fi
# 准入活性审计(2026-10-20 方案二):离线扫描 online_decision_log.jsonl,
# 判定"实例 KV 空闲足够却连续阻塞"的死端形态(fail-closed)。W/WL 决策
# 日志无 admission_blocked 行时应报 PASS no admission_blocked rows;审计
# 不通过即本 runner 非零退出(位于 metrics 后处理之后、归档之前,输入
# 取 bridge/ 或 results/ 下的决策日志,两阶段均可解析)。
if ! python3 "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/online/verify/admission_liveness_audit.py" "${RUN_DIR}"; then
  echo "[run_online_strategy] FAIL: admission liveness audit rejected ${RUN_DIR} (free-capacity deadlock streak)" >&2
  exit 1
fi
# 阶段 7 §10.5 中间产物生命周期: 最终结果 / 检查点 / 临时产物分目录。
#  - results/    : 最终结果(审计 jsonl,Python 决策侧写出)。保留规则:每
#    run 一份,run 目录即版本,不轮转;run 脚本开头 rm -rf 保证有界。
#  - 临时产物    : response/ack 消费即删;request 散装文件按 256 条批量
#    并入 request_journal.jsonl 后删除,成功结束仅保留单一顺序审计流与 fifo。
#  - 失败清理    : 失败时 bridge 保留为调试证据(不自动删),打印残留计数,
#    下次运行 rm -rf "${RUN_DIR}" 全量清理。
mkdir -p "${RUN_DIR}/results"
ARCHIVED=0
# remote_memory_transactions 为 C++ 桥侧逐事务明细(SerDes 并发化改造
# 2026-09-24):仅 --sensing-enabled 开启时写出,本 runner(sensing 关)通常
# 无此文件,-f 守卫为 no-op;与 run_online_strategy_sensing.sh 成对同改
# (W2 教训:成对脚本曾单边演进)。
for j in request_journal online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log train_ledger kv_delta_journal remote_memory_transactions; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
# P1(2026-08-30)kv delta journal 的 run 末 checksum 门产物(journal 开关
# off 时不存在,缺文件不是错误)。
if [ -f "${RUN_DIR}/bridge/kv_delta_journal_checksum.json" ]; then
  mv "${RUN_DIR}/bridge/kv_delta_journal_checksum.json" \
     "${RUN_DIR}/results/kv_delta_journal_checksum.json"
fi
echo "[run_online_strategy] artifacts: ${ARCHIVED} jsonl archived -> results/; request_journal=results/request_journal.jsonl"
# P3(2026-08-28):仿真成功后自动 SLO 指标提取(postprocess 成功之后、
# archive_run_outputs.sh 之前:cpp.log 未压缩、manifest 已拷入、results/
# 已归位,输入全齐;产物随保留集常驻,最细粒度纪律见脚本头注)。
# SH_SLO_POSTPROCESS: 1=默认 warn(子命令失败只写 slo_postprocess.FAIL,
# 不推翻仿真结果); 0=整步跳过; strict=失败即本 runner 非零退出。
SLO_MODE="${SH_SLO_POSTPROCESS:-1}"
if [[ "${SLO_MODE}" != "0" ]]; then
  if ! bash "${SCRIPT_DIR}/run_slo_postprocess.sh" "${RUN_DIR}"; then
    if [[ "${SLO_MODE}" == "strict" ]]; then
      echo "[run_online_strategy] FAIL: SLO postprocess failed (SH_SLO_POSTPROCESS=strict)" >&2
      exit 1
    fi
    echo "[run_online_strategy] WARN: SLO postprocess failures flagged in ${RUN_DIR}/slo_postprocess.FAIL (warn mode)" >&2
  fi
fi
# D1(2026-08-28):成功后产物瘦身归档(失败路径早已 exit 1 全量保留)。
# 传 DETAIL 档位:off 档归档豁免 [METRIC] 检查(2026-09-26 方案 A,与
# run_slo_postprocess.sh 口径对齐),summary/full 档无 [METRIC] 行仍 FAIL。
# SH_ARCHIVE_RUN=0 关闭(调试/对拍需要散装 bridge 文件时)。
if [[ "${SH_ARCHIVE_RUN:-1}" != "0" ]]; then
  bash "${SCRIPT_DIR}/archive_run_outputs.sh" "${RUN_DIR}" "${DETAIL}" || exit 1
fi
echo "[run_online_strategy] PASS: ${RUN_DIR}"
