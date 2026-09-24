#!/usr/bin/env bash
# sh_3.0 phase-3 官方在线 runner:strategy 感知模式(真实策略 + 感知开关)。
# Usage: bash run_online_strategy_sensing.sh <run_dir> <request_csv>
# Env: SH_METRICS_DETAIL=off|summary|full 覆盖指标明细档;缺省读
# sh_test_mesh/workload/llama2_7b_inference/metrics_config.json 的
# detail_level(env > json;两者非法均 fail-closed)。full 档后处理额外产
# request_metrics.csv(逐请求时序)。
# request-neutral(裸仓库):request_csv 由调用方按 traces/materialize_20_30s.py
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

RUN_DIR=$(realpath -m -- "${1:?run_dir 必填}")
REQUEST_CSV=${2:?"request_csv 必填(request-neutral:materialize the 20.csv first-30s input via traces/materialize_20_30s.py 后显式传入)"}
REQUEST_CSV=$(realpath -- "${REQUEST_CSV}")

# 与主 runner、矩阵和容量压力夹具共用仓级锁。必须先锁再读 generated
# 或清理 run_dir；嵌套入口沿用同一 open-file description，避免自锁。
SINGLE_SIMULATION_LOCK_PATH="${PROJECT}/sh_test_mesh/runs/.single_simulation.lock"
SINGLE_SIMULATION_LOCK_FD=""
acquire_single_simulation_lock() {
  mkdir -p "${PROJECT}/sh_test_mesh/runs"
  local inherited_fd="${SH_SINGLE_SIMULATION_LOCK_FD:-}"
  if [[ -n "${inherited_fd}" ]]; then
    if [[ ! "${inherited_fd}" =~ ^[0-9]+$ ]]; then
      echo "[run_online_strategy_sensing] invalid SH_SINGLE_SIMULATION_LOCK_FD=${inherited_fd@Q}" >&2
      return 1
    fi
    local fd_identity lock_identity
    fd_identity=$(stat -Lc '%d:%i' "/proc/${BASHPID}/fd/${inherited_fd}" 2>/dev/null) || {
      echo "[run_online_strategy_sensing] inherited simulation lock FD ${inherited_fd} is not open" >&2
      return 1
    }
    lock_identity=$(stat -Lc '%d:%i' "${SINGLE_SIMULATION_LOCK_PATH}") || return 1
    if [[ "${fd_identity}" != "${lock_identity}" ]] || ! flock -n "${inherited_fd}"; then
      echo "[run_online_strategy_sensing] inherited simulation lock FD does not hold ${SINGLE_SIMULATION_LOCK_PATH}" >&2
      return 1
    fi
    SINGLE_SIMULATION_LOCK_FD="${inherited_fd}"
  else
    exec {SINGLE_SIMULATION_LOCK_FD}>>"${SINGLE_SIMULATION_LOCK_PATH}"
    if ! flock -n "${SINGLE_SIMULATION_LOCK_FD}"; then
      exec {SINGLE_SIMULATION_LOCK_FD}>&-
      echo "[run_online_strategy_sensing] another simulation holds ${SINGLE_SIMULATION_LOCK_PATH}; refusing to start" >&2
      return 1
    fi
  fi
  export SH_SINGLE_SIMULATION_LOCK_FD="${SINGLE_SIMULATION_LOCK_FD}"
}
acquire_single_simulation_lock

# Backport 2026-08-16 (four-tier 3-min comparison test adaptation 5a):
# resolve the single generated dir dynamically -- the dir name encodes
# this window's sess/req/p/d ranges and the trace-config digest, so a
# hardcoded path breaks whenever the config bytes change.
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[run_online_strategy_sensing] expected exactly one generated dir under sh_test_mesh/generated, found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_inference"
# 与主入口共用运行时硬件目录选择。受控硬件档在 materializer 后通过
# SH_RUNTIME_RC_DIR 指向实际四件目录；快照中的源 JSON 与执行档才可核对。
RC="${SH_RUNTIME_RC_DIR:-${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__edge_remote_memory_pool}"
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
# 族，含 Python 启动即死形态）；管不到 wait_for_work() 停泊族（由
# --idle-watchdog-s 墙钟看门狗兜住；本仓已无 input-open 死端分支，对齐
# wscllm 合同 §2.1）。
# 监督纪律：终止长跑用 SIGTERM（kill <pid>），勿用 SIGINT/Ctrl-C——后者会
# 冻结健康瞬态造成"楔死"伪影（2026-08-22 诊断结论）；疑似楔死时先保留
# bridge 目录盘态与双方 /proc/<pid>/{wchan,syscall,stack} 再清理。
BRIDGE_TIMEOUT_MS="${BRIDGE_TIMEOUT_MS:-120000}"
BRIDGE_TIMEOUT_ARGS=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS}")

POSTPROCESS=${SCRIPT_DIR}/run_metrics_postprocess.sh

# B1/WP0 (SLO 指标改造): --metrics-detail 不再硬编码 summary——优先取
# env SH_METRICS_DETAIL，缺省回落 workload metrics_config.json 的
# detail_level（优先级 env > json）。json 缺失/非法/值不在
# {off,summary,full} 或 env 值非法均 fail-closed 退出。
METRICS_CONFIG_JSON="${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/metrics_config.json"
if [[ -n "${SH_METRICS_DETAIL:-}" ]]; then
  DETAIL="${SH_METRICS_DETAIL}"
else
  DETAIL=$(python3 - "${METRICS_CONFIG_JSON}" <<'PY' || exit 1
import json, sys
path = sys.argv[1]
try:
    with open(path, encoding="utf-8") as source:
        config = json.load(source)
except (OSError, ValueError) as error:
    sys.stderr.write("[run_online_strategy_sensing] cannot read %s: %s\n" % (path, error))
    sys.exit(1)
detail = config.get("detail_level") if isinstance(config, dict) else None
if detail not in ("off", "summary", "full"):
    sys.stderr.write(
        "[run_online_strategy_sensing] invalid detail_level %r in %s "
        "(expected off|summary|full)\n" % (detail, path))
    sys.exit(1)
print(detail)
PY
)
fi
if [[ "${DETAIL}" != "off" && "${DETAIL}" != "summary" && "${DETAIL}" != "full" ]]; then
  echo "[run_online_strategy_sensing] invalid SH_METRICS_DETAIL='${DETAIL}' (expected off|summary|full)" >&2
  exit 1
fi

rm -rf "${RUN_DIR}"
mkdir -p "${RUN_DIR}"

# 归档本次真正读取的 trace 与硬件原始字节，供 joint 域指标离线复算。
# 与主 runner 同口径：旧 run 若无快照只给 NA，不借当前 checkout 补值。
TRACE_CONFIG_SOURCE="${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv"
python3 - "${TRACE_CONFIG_SOURCE}" "${PROJECT}/sh_test_mesh" "${RUN_DIR}" <<'PY'
import csv
import shutil
import sys
from pathlib import Path

trace_path, sh_test_mesh, run_dir = map(Path, sys.argv[1:4])
if not trace_path.is_file():
    raise SystemExit(f"[run_online_strategy_sensing] missing trace config: {trace_path}")
hardware_declared = None
try:
    with trace_path.open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if ((row.get("kind") or "").strip().lower() == "config"
                    and (row.get("key") or "").strip() == "hardware_config"):
                hardware_declared = (row.get("value") or "").strip()
                break
except (OSError, csv.Error) as error:
    raise SystemExit(
        f"[run_online_strategy_sensing] cannot parse {trace_path}: {error}")
if not hardware_declared:
    raise SystemExit(
        f"[run_online_strategy_sensing] trace config has no hardware_config row: "
        f"{trace_path}")
hardware_path = Path(hardware_declared)
if not hardware_path.is_absolute():
    hardware_path = sh_test_mesh / hardware_path
if not hardware_path.is_file():
    raise SystemExit(
        f"[run_online_strategy_sensing] missing configured hardware JSON: "
        f"{hardware_path}")
shutil.copyfile(trace_path, run_dir / "trace_config.csv.snapshot")
shutil.copyfile(hardware_path, run_dir / "hardware_config.json.snapshot")
print("[run_online_strategy_sensing] run inputs snapshotted: "
      f"{run_dir / 'trace_config.csv.snapshot'}, "
      f"{run_dir / 'hardware_config.json.snapshot'}")
PY
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
# K7 注记(2026-09-23 外部审计):本脚本是 sensing 演示旁路发射路径——
# 无 F7 耦合断言/无 SH_LINK_TELEMETRY 注入链(那是 run_online_strategy.sh
# 公共发射路径的职责);quota 实验一律走 runner/主路径,勿以本脚本为
# 闭环证据。
# L6 fail-closed(2026-09-23 复核审计):注记只劝不拦——aimd 在本旁路
# 起跑会静默空转(observer 缺省 0 ⇒ link_telemetry[] 恒空),直接拒绝
# 启动并指回主路径。
if [[ "${JOINT_QUOTA_MODE:-off}" == "aimd" ]]; then
  echo "[run_online_strategy_sensing] FAIL: JOINT_QUOTA_MODE=aimd is not supported on the sensing demo path (no telemetry injection here — AIMD would idle silently); use run_online_strategy.sh / joint_runner.py --quota aimd" >&2
  exit 1
fi
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

# L11（2026-09-23 深挖审计，与主 runner L10 同款）：set -e 下裸 wait 在
# C++ 非零退出时直接终止本脚本，下方失败诊断尾段（log tail /
# "bridge retained" 提示）不可达——改为 || 捕获退出码（与上方 PY_EXIT
# 同手法），诊断可达后仍经下方判定 fail-closed 非零退出。
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
# B1/WP1: full 档额外产 request_metrics.csv(逐请求时序,manifest fail-closed
# 连接);summary/off 档不产(说明写 postprocess.log)。
if grep -q '\[METRIC\]' "${RUN_DIR}/cpp.log"; then
  bash "${POSTPROCESS}" "${RUN_DIR}/cpp.log" \
    --out-raw="${RUN_DIR}/raw_metrics.csv" \
    --out-normalized="${RUN_DIR}/normalized_metrics.csv" \
    --out-request="${RUN_DIR}/request_metrics.csv" \
    > "${RUN_DIR}/postprocess.log" 2>&1
  echo "[run_online_strategy_sensing] metrics postprocess: raw=$(ls "${RUN_DIR}/raw_metrics.csv") normalized=$(ls "${RUN_DIR}/normalized_metrics.csv") request=$(ls "${RUN_DIR}/request_metrics.csv" 2>/dev/null || echo 'SKIPPED(detail!=full)')"
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
for j in request_journal online_decision_log graph_batch_digests ledger online_stats profile sensing_query_log train_ledger; do
  if [ -f "${RUN_DIR}/bridge/${j}.jsonl" ]; then
    mv "${RUN_DIR}/bridge/${j}.jsonl" "${RUN_DIR}/results/${j}.jsonl"
    ARCHIVED=$((ARCHIVED + 1))
  fi
done
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
  bash "${SCRIPT_DIR}/archive_run_outputs.sh" "${RUN_DIR}" || exit 1
fi
echo "[run_online_strategy_sensing] PASS: ${RUN_DIR}"
