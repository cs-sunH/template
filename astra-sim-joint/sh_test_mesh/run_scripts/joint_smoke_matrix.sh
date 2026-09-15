#!/usr/bin/env bash
# joint_smoke_matrix.sh -- §7-K 复审修复批的冒烟矩阵脚本化复现入口
#（PROVENANCE §7-14 冒烟证据保全；2026-09-14 新增，kimi 终审-中5 修订）。
#
# 目的：把"kimi 复审报告无法独立复验冒烟矩阵"的交付完整性缺口补上——
# 冒烟矩阵从临时目录 + 手工命令改为脚本一键复现，产物落仓外**持久**
# 固定目录并逐配置留存全套证据（run.log/cpp.log/python.log/决策日志/
# verify 摘要/退出码/env 快照）。
#
# 用法：
#   bash sh_test_mesh/run_scripts/joint_smoke_matrix.sh [窗口ns] [证据根目录]
# 缺省：窗口 2000000000（2s，Agents.md 冒烟纪律最小窗），证据根目录
#   /home/sunhao/joint_smoke_evidence（家目录持久路径——/tmp 易失，
#   kimi 终审建议；PROVENANCE 落指针）。
# 前置：C++ 二进制已构建
#   （build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_
#    Congestion_Aware_Online）；裸仓交付态须先按 README §4 拼装构建。
# 覆盖：八组合 + TJE+remote-off + 影子验证跑（SH_ADMIT_GATE_VERIFY=1 +
#   SH_SNAPSHOT_VERIFY=1 + SH_ONLINE_VALIDATE=1，TJE）= 10 配置。
# 已知边界：SIGKILL 不还原 trace_config 指针（不可捕获信号）；SIGINT
#（Ctrl-C）在子进程前台期存在竞态窗口同样可能不还原——两态均以裸仓
# 终检 clean_test_records.sh 兜底（四审-低4 补登）。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
WINDOW_NS=${1:-2000000000}
EVIDENCE_ROOT=${2:-/home/sunhao/joint_smoke_evidence}
SRC_CSV=${SH_SMOKE_SOURCE_CSV:-/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv}

if [[ ! -f "${SRC_CSV}" ]]; then
  echo "[smoke-matrix] source csv missing: ${SRC_CSV}" >&2
  exit 1
fi
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
if [[ ! -x "${BIN}" ]]; then
  echo "[smoke-matrix] binary missing: ${BIN}（裸仓交付态——先按 README §4 构建）" >&2
  exit 1
fi

mkdir -p "${EVIDENCE_ROOT}"
# 物化到固定路径：request_queue_csv 路径进 trace-config digest，换路径
# 会物化出第二个 plan 目录、撞 run_online_strategy 的"恰好一个"断言。
MATERIALIZED="${EVIDENCE_ROOT}/smoke_input"
mkdir -p "${MATERIALIZED}"
python3 "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/traces/materialize_20_30s.py" \
  "${SRC_CSV}" "${MATERIALIZED}/requests.csv" "${WINDOW_NS}" \
  > "${EVIDENCE_ROOT}/materialize.log" 2>&1
REQUEST_CSV=$(realpath "${MATERIALIZED}/requests.csv")

# trace_config 指针切换（→ 本次物化输入）+ plan 物化
# （run_online_strategy.sh 要求恰好一个 generated plan 目录；物化前清
# 历史残留，固定输入路径 → digest 稳定 → 单目录）。
rm -rf "${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_plan_* 2>/dev/null || true
TRACE_CONFIG="${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv"

set_pointer() {  # set_pointer <值>
  python3 - "${TRACE_CONFIG}" "$1" <<'PYEOF'
import sys
path, value = sys.argv[1], sys.argv[2]
lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
out = []
for line in lines:
    fields = line.rstrip("\n").split(",")
    if len(fields) > 2 and fields[1] == "request_queue_csv":
        fields[2] = value
        line = ",".join(fields) + "\n"
    out.append(line)
open(path, "w", encoding="utf-8").writelines(out)
PYEOF
}

read_pointer() {
  python3 - "${TRACE_CONFIG}" <<'PYEOF'
import sys
for line in open(sys.argv[1], encoding="utf-8"):
    fields = line.rstrip("\n").split(",")
    if len(fields) > 2 and fields[1] == "request_queue_csv":
        print(fields[2])
        break
PYEOF
}

# 终审-中5：还原**进入前原值**（不恒还原占位常量——预先配置的非占位
# 指针不得被吞）；指针在全部配置运行期间保持指向物化输入（Python 服务
# 运行期实时读该指针）；EXIT trap 收尾还原。SIGKILL 不可捕获（头部已
# 登记边界）。
POINTER_ORIGINAL=$(read_pointer)
restore_pointer() { set_pointer "${POINTER_ORIGINAL}"; }
trap restore_pointer EXIT
set_pointer "${REQUEST_CSV}"
( cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference" \
    && python3 plan_materializer.py ) \
  > "${EVIDENCE_ROOT}/plan_materialize.log" 2>&1

run_one() {
  local label=$1; shift
  local run_dir="${EVIDENCE_ROOT}/${label}"
  # 终审-中5：run.log 先落在 run_dir **外**（runner 起手 rm -rf RUN_DIR
  # 会把重定向文件解联成无名 inode——证据链断裂），runner 结束后再移入。
  local runner_log="${EVIDENCE_ROOT}/${label}.runner.log"
  rm -rf "${run_dir}" "${runner_log}"; mkdir -p "${run_dir}"
  local rc=0
  env "$@" \
    bash "${SCRIPT_DIR}/run_online_strategy.sh" "${run_dir}" "${REQUEST_CSV}" \
    > "${runner_log}" 2>&1 || rc=$?
  mv "${runner_log}" "${run_dir}/run.log"
  # §9-低7：逐配置 env 快照落盘（影子验证跑可自证开关状态）。
  env "$@" env > "${run_dir}/env.txt" 2>&1 || true
  echo "${rc}" > "${run_dir}/exit_code"
  if [[ ${rc} -eq 0 ]]; then
    echo "[smoke-matrix] ${label}: PASS"
  else
    echo "[smoke-matrix] ${label}: FAIL (exit ${rc})——tail run.log:"
    tail -5 "${run_dir}/run.log" || true
  fi
  # 同一请求集 → 同一 plan 目录全配置复用（物化一次；勿删——
  # run_online_strategy.sh 要求恰好一个 generated plan 目录）。
}

for combo in none T J E TJ TE JE TJE; do
  run_one "combo_${combo}" JOINT_ABLATION_COMBO="${combo}"
done
run_one "TJE_remote_off" JOINT_ABLATION_COMBO=TJE JOINT_REMOTE_ACTIONS=off
run_one "TJE_shadow_verify" JOINT_ABLATION_COMBO=TJE \
  SH_ADMIT_GATE_VERIFY=1 SH_SNAPSHOT_VERIFY=1 SH_ONLINE_VALIDATE=1

echo "[smoke-matrix] evidence root: ${EVIDENCE_ROOT}"
echo "[smoke-matrix] summary:"
rc_all=0
for f in "${EVIDENCE_ROOT}"/combo_*/exit_code "${EVIDENCE_ROOT}"/TJE_*/exit_code; do
  label=$(basename "$(dirname "${f}")")
  code=$(cat "${f}")
  printf '  %-22s exit=%s\n' "${label}" "${code}"
  [[ ${code} -ne 0 ]] && rc_all=1
done
exit ${rc_all}
