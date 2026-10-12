#!/usr/bin/env bash
# wscllm phase-3 官方在线 runner:strategy 感知模式(真实策略 + 感知开关)。
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
# 相对 run_dir 归一(realpath -m,与 fixture 同款):rm -rf/mkdir 在调用方 cwd
# 执行、此后 cd PROJECT,不归一则后续 ${RUN_DIR} 引用相对 PROJECT 解析,
# 传相对路径且调用目录非仓根时必然 No such file or directory。
RUN_DIR=$(realpath -m "${RUN_DIR}")
# P3-3.9 E2(2026-09-28)rm -rf 护栏(fail-closed,与 GEN_MATCH/RC 闸同风格):
# RUN_DIR 由调用方任意传入,realpath -m 归一不改危险语义——传 `.`(cwd=仓根)
# 归一即仓根、传 `..` 归一即仓根祖先,下方 `rm -rf "${RUN_DIR}"` 会删掉整个
# 仓库或其上层目录。显式拒绝:空串、`/`、仓根本身、仓根的祖先目录、仓根
# 内部路径(runner 合同 run_dir 一律仓外,正式纪律 run_dir 一律 /tmp,
# 见 README §2);合法 run_dir 不受影响。
case "${RUN_DIR}" in
  ""|"/")
    echo "[run_online_strategy_sensing] FAIL: refusing dangerous RUN_DIR='${RUN_DIR}' (empty or filesystem root)" >&2
    exit 1 ;;
esac
if [[ "${RUN_DIR}" == "${PROJECT}" || "${RUN_DIR}" == "${PROJECT}"/* || "${PROJECT}" == "${RUN_DIR}"/* ]]; then
  echo "[run_online_strategy_sensing] FAIL: refusing RUN_DIR='${RUN_DIR}' (repo root, its ancestor, or inside the repo; use a dedicated dir outside the repo, e.g. /tmp/...)" >&2
  exit 1
fi
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:请按 traces/derive_20_first_30_seconds.py 物化输入后显式传入;其 stdout 即权威 provenance 记录)"}

# ET 基线目录 = GEN_MATCH 动态解析(五仓统一口径)——恰好一个 llama2_7b_wsc_llm_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_wsc_llm_inference"
# RC = runtime_config 目录动态恰一解析（crosscut-F1，2026-10-01，原为硬编码
# 目录名，照 face-LRU §16d 形态移植，与 run_online_strategy.sh 同款）：
# 目录名由物化器生成为 "<hardware slug>__<capacity_profile>__<remote-memory
# 标签>"（generate_wsc_llm_trace -> config_resolver.materialize_runtime_configs；
# 本仓标签恒为 no_memory_expansion）。capacity_profile 的权威同源 =
# trace_config.csv 的 config,local_hbm_capacity_profile 行（plan_materializer
# 按同一文件落目录名）。按该 profile 过滤 generated/runtime_config/ 并要求
# 恰好一个（与上方 GEN_MATCH 的 exactly-one 口径一致），缺失/歧义
# fail-closed——换 profile 后硬编码名会找不到目录或静默加载错误容量档；
# 现网唯一合法档位下动态解析与旧硬编码命中同一目录，健康路径零行为变化。
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
  echo "[runner] FAIL: cannot read local_hbm_capacity_profile from trace_config.csv" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/face_case5_config_c__*"${RC_PROFILE}"__no_memory_expansion)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -d "${RC_MATCH[0]}" \
      || ! -f "${RC_MATCH[0]}/system.json" || ! -f "${RC_MATCH[0]}/network.yml" \
      || ! -f "${RC_MATCH[0]}/comm_group.json" ]]; then
  echo "[runner] expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json (materialized under sh_test_mesh/generated/runtime_config/ by plan_materializer.py; bare-repo restore deletes generated/), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
# BIN 显式存在性闸（crosscut-F3，2026-10-01，与上方 GEN_MATCH/RC 闸对称，
# 与 run_online_strategy.sh 同款）：裸仓未编译时原报错路径是 FIFO 等待环
# 里的 "C++ exited before FIFO ready"，诊断晚一步且不指名缺什么；缺失即
# 在此报错。
if [[ ! -x "${BIN}" ]]; then
  echo "[runner] bin missing: ${BIN} (build first: cmake --build build/astra_analytical/build_congestion_aware)" >&2
  exit 1
fi

# 桥侧看门狗（2026-08-22 引入；P0-2 2026-08-31 常态化+定位修正；收尾批
# 2026-09-01 与主 runner 统一武装——sensing 变体此前保留旧"缺省不设=永等"
# 形态）：BRIDGE_TIMEOUT_MS 三态语义（与 run_online_strategy.sh 同款）：
#   未设/空  -> 缺省 120000（campaign 常态；值必须大于本负载最慢单决策耗时，
#               且大于 Python 服务启动的 FIFO 开启等待——实际下界是秒级，
#               建议 >=10000=10s，过小会在启动窗口 abort C++）；
#   显式 =0  -> 永等（冻结旧默认的逃生口：桥侧 poll 不设超时）；
#   正值     -> 该值（毫秒）。
# 定位：只武装 C++ 桥的两处 response poll（覆盖"Python 侧单次决策交换停滞"
# 族，含 Python 启动即死形态）；管不到 wait_for_work() 停泊族（wscllm 无
# input-open 死端 fail-loud 分支，合同 §2.1——由 --idle-watchdog-s 墙钟
# 看门狗兜住）。
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

# P3-3.9 E2(2026-09-28)孤儿 C++ 清理(对齐 run_online_idle_fixture.sh 的
# CPP_PID+cleanup+trap EXIT 模式,与 run_online_strategy.sh 同款):下方 C++
# 后台启动,脚本若在任何 wait 之前被终止(监督纪律 kill <pid>),C++ 成孤儿、
# 继续向 RUN_DIR/bridge 写文件至桥看门狗超时,与下一次运行的 rm -rf 重建
# 互踩。正常路径 C++ 已被 wait 收尸,cleanup 的 kill -0 落空、零扰动;
# 失败/被杀路径 kill 残留 C++。
CPP_PID=""
cleanup() {
  if [[ -n "${CPP_PID}" ]] && kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT
# 裸 EXIT trap 在 bash 被致命信号直接杀死时不执行;监督纪律以 SIGTERM 终止
# 长跑,必须显式转成 exit 才能进入上方清理。INT/HUP 同理。
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

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
# run-time 配置随 run_dir 冻结（hbm_watermark.load_trace_config 首选
# run_dir/trace_config.csv）：后处理重跑旧 run_dir 不再回退仓内现行文件
# （campaign 换 capacity profile/模型行后该文件字节会变）。
cp "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv" \
   "${RUN_DIR}/trace_config.csv"

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

# FIFO 就绪等待带墙钟预算（crosscut-F2，2026-10-01，照 wscllm-LRU 同款，
# 与 run_online_strategy.sh 同款）：上面的 kill -0 只防 C++ 进程退出，
# 防不了"进程存活但迟迟不建 FIFO"的窗口；超时打印 cpp.log 尾部后
# fail-closed（SH_FIFO_WAIT_S 可覆盖，缺省 60s；健康运行 FIFO 秒级就绪，
# 预算不会误伤）。
fifo_deadline=$(( $(date +%s) + ${SH_FIFO_WAIT_S:-60} ))
until [ -p "${RUN_DIR}/bridge/req_notify.fifo" ] && [ -p "${RUN_DIR}/bridge/resp_notify.fifo" ]; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[run_online_strategy_sensing] C++ exited before FIFO ready (see cpp.log)" >&2
    exit 1
  fi
  if (( $(date +%s) > fifo_deadline )); then
    echo "[run_online_strategy_sensing] FIFOs not ready within ${SH_FIFO_WAIT_S:-60}s (see cpp.log tail)" >&2
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
  --sensing \
  --plan-dir "${ET_DIR}" \
  > "${RUN_DIR}/python.log" 2>&1 || PY_EXIT=$?

# C++ 非零退出(桥超时/楔死 abort)时 wait 返回非零;set -e 下裸 wait 会
# 立即终止脚本、跳过下方退出码回显与诊断。|| 接住退出码走失败诊断路径;
# 成功路径 || 右侧不执行,先显式置 0(set -u 下未初始化引用即死)。
CPP_EXIT=0
wait "${CPP_PID}" || CPP_EXIT=$?
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
  # R4-blind-F3（2026-10-01）：postprocess 失败原先在 set -e 下静默死退出
  # （输出全进 postprocess.log，控制台零诊断）；显式捕获并报错，与其余
  # 失败出口同款响亮。
  if ! bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-requests="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1; then
    echo "[run_online_strategy_sensing] FAIL: metrics postprocess exited non-zero; see ${RUN_DIR}/postprocess.log" >&2
    exit 1
  fi
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
# P1(2026-08-30)kv delta journal 的 run 末 checksum 门产物(journal 开关
# off 时不存在,缺文件不是错误)。
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
  bash "${SCRIPT_DIR}/archive_run_outputs.sh" "${RUN_DIR}" "${DETAIL}" || exit 1
fi
echo "[run_online_strategy_sensing] PASS: ${RUN_DIR}"
