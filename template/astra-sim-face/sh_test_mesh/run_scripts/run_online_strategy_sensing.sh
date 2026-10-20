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

# 相对入参锚定调用者 cwd 预先解析为绝对路径：下方 rm/mkdir 在
# cd "${PROJECT}" 之前执行，而 cp/bridge-dir/cpp.log/queue-csv 等引用全部
# 在其之后——放任相对形式会让 rm/mkdir 以调用者 cwd 为基准、其余引用改以
# 仓根为基准，cp 因目标父目录不存在而失败退出，并在调用者 cwd 留下空目录。
case "${RUN_DIR}" in /*) ;; *) RUN_DIR="$(pwd -P)/${RUN_DIR}" ;; esac
case "${REQUEST_CSV}" in /*) ;; *) REQUEST_CSV="$(pwd -P)/${REQUEST_CSV}" ;; esac

# rm -rf 护栏（2026-09-28，与 run_online_strategy.sh 同款）：RUN_DIR 已
# 绝对锚定非空，但 `.` 入参会锚成 "$(pwd)/."——GNU rm 拒删本体却会清空
# 其内容。拒绝解析为 /、仓根、调用者 cwd 本体的 RUN_DIR。
RUN_DIR_ABS=$(realpath -m -- "${RUN_DIR}")
if [ "${RUN_DIR_ABS}" = "/" ] || [ "${RUN_DIR_ABS}" = "${PROJECT}" ] \
   || [ "${RUN_DIR_ABS}" = "$(pwd -P)" ]; then
  echo "[run_online_strategy_sensing] FAIL: RUN_DIR='${RUN_DIR}' resolves to '/' / repo root / caller cwd — refusing rm -rf" >&2
  exit 1
fi

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_inference"
# RC = runtime_config 目录动态解析（2026-09-29，原为硬编码目录名）：目录名
# 由配置装载器生成为 "<hardware slug>__<capacity_profile>__no_memory_expansion"
# （generate_face_trace.py 的 _resolve + config_resolver.materialize_runtime_configs），
# 随 trace_config.csv 的 hardware_config/local_hbm_capacity_profile 变化；plan
# 目录的 manifest.json 记录本次物化的 capacity_profile。旧 profile 残留时硬编码
# 名会静默加载错误容量档——改为按 manifest 的 profile 过滤并要求恰好一个
# （与上方 ET_DIR 的 exactly-one 口径一致），缺失/歧义即 fail-closed。
RC_PROFILE=$(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as source:
    manifest = json.load(source)
value = manifest.get("hardware", {}).get("capacity_profile")
if not isinstance(value, str) or not value:
    sys.exit("manifest.json hardware.capacity_profile missing/invalid")
print(value)
' "${ET_DIR}/manifest.json") || {
  echo "[runner] FAIL: cannot read hardware.capacity_profile from ${ET_DIR}/manifest.json (stale plan dir? re-run plan_materializer.py)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__no_memory_expansion)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" ]]; then
  echo "[runner] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
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

# SH_SLO_POSTPROCESS / SH_ARCHIVE_RUN 值域校验（2026-09-29 fail-closed，与
# run_online_strategy.sh 同款）：白名单以脚本内实际消费逻辑为准——
# SH_SLO_POSTPROCESS: 1=默认 warn / 0=整步跳过 / strict=失败即本 runner
# 非零退出（语义见下方 P3 段）；SH_ARCHIVE_RUN: 1=默认归档 / 0=关闭
# （调试/对拍需要散装 bridge 文件时，见下方 D1 段）。取值域外的值会被
# 下方 !=0 与 ==strict 判断静默吞掉（如 "true"/"STRICT" 之类拼写偏差会把
# strict 门静默弱化为 warn、"false" 会让归档关不掉），故入口即拒绝。
SLO_MODE="${SH_SLO_POSTPROCESS:-1}"
case "${SLO_MODE}" in
  0|1|strict) ;;
  *)
    echo "[run_online_strategy_sensing] FAIL: invalid SH_SLO_POSTPROCESS='${SLO_MODE}' (expected 0|1|strict)" >&2
    exit 1
    ;;
esac
ARCHIVE_RUN="${SH_ARCHIVE_RUN:-1}"
case "${ARCHIVE_RUN}" in
  0|1) ;;
  *)
    echo "[run_online_strategy_sensing] FAIL: invalid SH_ARCHIVE_RUN='${ARCHIVE_RUN}' (expected 0|1)" >&2
    exit 1
    ;;
esac

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
  --online-validate "${SH_ONLINE_VALIDATE:-0}" \
  "${BRIDGE_TIMEOUT_ARGS[@]}" \
  --request-queue-csv "${REQUEST_CSV}" \
  --close-input \
  --workload-configuration="${ET_PREFIX}" \
  --comm-group-configuration="${RC}/comm_group.json" \
  --system-configuration="${RC}/system.json" \
  --network-configuration="${RC}/network.yml" \
  --logging-folder=off \
  --metrics-configuration="${ET_DIR}/metrics_manifest.json" \
  --metrics-detail="${DETAIL}" \
  > "${RUN_DIR}/cpp.log" 2>&1 &
CPP_PID=$!

# M20（自 joint 移植，2026-09-29，与 run_online_strategy.sh 同款）：中断/失败
# 路径 fail-closed 清场。此前无 trap：早期 exit 1（或 INT/TERM 半途打断）会把
# 已启动的 C++ 孤儿进程留在后台继续吃资源。现统一在 EXIT/INT/TERM 清掉尚存活
# 的 C++ 子进程：先 SIGTERM，宽限 ~3s 后 SIGKILL 兜底；正常路径均为 no-op。
cleanup_cpp() {
  if [[ -n "${CPP_PID:-}" ]] && kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
    local _i
    for _i in $(seq 1 30); do
      kill -0 "${CPP_PID}" 2>/dev/null || return 0
      sleep 0.1
    done
    kill -9 "${CPP_PID}" 2>/dev/null || true
  fi
}
trap cleanup_cpp EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# FIFO 就绪等待（2026-09-28 有界化，与 run_online_strategy.sh 同款）：
# 上限 120s（400×0.3s，与 BRIDGE_TIMEOUT_MS 缺省量级一致；固定值，不新增
# env 开关）；超时按监督纪律 SIGTERM C++ 后 fail-closed 退出。
FIFO_WAITED=0
until [ -p "${RUN_DIR}/bridge/req_notify.fifo" ] && [ -p "${RUN_DIR}/bridge/resp_notify.fifo" ]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[run_online_strategy_sensing] C++ exited before FIFO ready (see cpp.log)" >&2
    exit 1
  fi
  if [ "${FIFO_WAITED}" -ge 400 ]; then
    echo "[run_online_strategy_sensing] FAIL: bridge FIFOs not ready after 120s (cpp alive but startup hung; see cpp.log)" >&2
    kill "${CPP_PID}" 2>/dev/null || true
    exit 1
  fi
  sleep 0.3
  FIFO_WAITED=$((FIFO_WAITED + 1))
done

cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
PY_EXIT=0
python3 -u online/online_service.py \
  --bridge-dir "${RUN_DIR}/bridge" \
  --mode strategy \
  --sensing \
  --plan-dir "${ET_DIR}" \
  > "${RUN_DIR}/python.log" 2>&1 || PY_EXIT=$?

CPP_EXIT=0; wait "${CPP_PID}" || CPP_EXIT=$?
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
# 2026-09-28:rc 显式捕获(与 run_online_strategy.sh 同款)——原裸调用被
# set -e 静默吞。后处理失败≠仿真失败,只加可见性,成功路径输出原样。
# 锚定行首匹配（与 run_slo_postprocess.sh / archive_run_outputs.sh 同口径）：
# cpp.log 合并了 stderr，非锚定匹配会被 [METRIC][ERROR] 诊断行误触发。
if grep -q '^\[METRIC\] ' "${RUN_DIR}/cpp.log"; then
  PP_RC=0
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-requests="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1 || PP_RC=$?
  if [ "${PP_RC}" -ne 0 ]; then
    echo "[run_online_strategy_sensing] FAIL: metrics postprocess (exit=${PP_RC}) — simulation result NOT overturned; postprocess.log tail:" >&2
    tail -5 "${RUN_DIR}/postprocess.log" >&2
  else
    echo "[run_online_strategy_sensing] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv")"
    if [ -f "${RUN_DIR}/request_metrics.csv" ]; then
      echo "[run_online_strategy_sensing] request metrics: ${RUN_DIR}/request_metrics.csv"
    else
      echo "[run_online_strategy_sensing] request metrics: none (detail=${DETAIL}; see postprocess.log)"
    fi
  fi
else
  echo "[run_online_strategy_sensing] WARNING: no [METRIC] lines in cpp.log; postprocess skipped" >&2
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
for j in request_journal online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log train_ledger kv_delta_journal; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
# P1(2026-08-30)kv delta journal 的 run 末 checksum 门产物。本仓(face)为
# 消费侧、无写入方——写入方在 W 仓 KV_DELTA_JOURNAL 开关与 joint 仓
# online_service(侧车链)，故正常 face run 恒缺此二件、缺文件不是错误；
# 二件在场时(跨仓产物归位)hbm_watermark certified 层的 exit-3 正式判决
# 依赖 checksum，与 journal 同搬，见 archive_run_outputs.sh 头注；无
# journal 时 hbm 降级 upper_bound_only 上界口径(超限只诊断、exit 0)。
if [ -f "${RUN_DIR}/bridge/kv_delta_journal_checksum.json" ]; then
  mv "${RUN_DIR}/bridge/kv_delta_journal_checksum.json" \
     "${RUN_DIR}/results/kv_delta_journal_checksum.json"
fi
echo "[run_online_strategy_sensing] artifacts: ${ARCHIVED} jsonl archived -> results/; request_journal=results/request_journal.jsonl"
# P3(2026-08-28):仿真成功后自动 SLO 指标提取(postprocess 成功之后、
# archive_run_outputs.sh 之前:cpp.log 未压缩、manifest 已拷入、results/
# 已归位,输入全齐;产物随保留集常驻,最细粒度纪律见脚本头注)。
# SH_SLO_POSTPROCESS: 1=默认 warn(子命令失败只写 slo_postprocess.FAIL,
# 不推翻仿真结果); 0=整步跳过; strict=失败即本 runner 非零退出。
# warn/strict 边界(2026-09-26 A5-1/infra-4,与 run_online_strategy.sh
# 同构):run_slo_postprocess.sh 的 exit 3=hbm_watermark certified 层逐
# rank 容量违规正式判决,本层保码上抛(strict 下以同码 3 退出,不折叠为
# 1);默认 warn 下仿真结论不变,但码不丢——WARN 行明示 (exit=3) 判决。
# SLO_MODE 值域已在入口白名单校验（0|1|strict）。
if [[ "${SLO_MODE}" != "0" ]]; then
  # 本层 set -euo pipefail 生效中:rc 必须在条件上下文捕获,否则失败会
  # 立刻终止 runner,下方 warn/strict 分支不可达(review 2026-09-26 F1)。
  SLO_RC=0
  bash "${SCRIPT_DIR}/run_slo_postprocess.sh" "${RUN_DIR}" || SLO_RC=$?
  if [[ "${SLO_RC}" -eq 3 ]]; then
    if [[ "${SLO_MODE}" == "strict" ]]; then
      echo "[run_online_strategy_sensing] FAIL: SLO per-rank capacity verdict exit=3 (SH_SLO_POSTPROCESS=strict)" >&2
      exit 3
    fi
    echo "[run_online_strategy_sensing] WARN: SLO per-rank capacity verdict FAIL (exit=3), flagged in ${RUN_DIR}/slo_postprocess.FAIL (warn mode; code preserved)" >&2
  elif [[ "${SLO_RC}" -ne 0 ]]; then
    if [[ "${SLO_MODE}" == "strict" ]]; then
      echo "[run_online_strategy_sensing] FAIL: SLO postprocess failed (SH_SLO_POSTPROCESS=strict)" >&2
      exit 1
    fi
    echo "[run_online_strategy_sensing] WARN: SLO postprocess failures flagged in ${RUN_DIR}/slo_postprocess.FAIL (warn mode)" >&2
  fi
fi
# D1(2026-08-28):成功后产物瘦身归档(SH_ARCHIVE_RUN=0 关闭;值域已在入口
# 白名单校验)。
# 指标档位透传(参数 2):off 档归档步跳过①③(by design,与 SLO 步语义一致,
# 见 archive_run_outputs.sh 头注)。
if [[ "${ARCHIVE_RUN}" != "0" ]]; then
  bash "${SCRIPT_DIR}/archive_run_outputs.sh" "${RUN_DIR}" "${DETAIL}" || exit 1
fi
echo "[run_online_strategy_sensing] PASS: ${RUN_DIR}"
