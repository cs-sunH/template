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
#   SH_SNAPSHOT_VERIFY=1 + SH_ONLINE_VALIDATE=1，TJE）= 10 配置；
#   C19（总收口）扩 3 臂 = 13 配置——TJE_quota_static（JOINT_QUOTA_MODE=
#   static）/ TJE_quota_aimd（=aimd，F7 自动注入链全程生效）/ TJE_face_
#   static（--scheduler face_static 对照臂，quota 强制 off 语义入
#   manifest 注记）。三新臂经 joint_runner.py CLI 面起跑（--quota /
#   --scheduler），aimd 臂不显式设 SH_LINK_TELEMETRY——由 runner 置位、
#   run_online_strategy.sh 消费追加 C++ 旗标（注入事实落 invocation.json
#   的 link_telemetry_injected）。
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

# 仓级共享仿真锁：必须先锁再写 evidence、物化输入、清 generated 或切换
# trace_config。内层 .sh 与 joint_runner 通过此 FD 复用同一 flock OFD，
# 避免嵌套自锁；只认当前 runs 锁文件的同一 inode，伪造/失效握手 fail-closed。
SINGLE_SIMULATION_LOCK_PATH="${PROJECT}/sh_test_mesh/runs/.single_simulation.lock"
SINGLE_SIMULATION_LOCK_FD=""
SINGLE_SIMULATION_LOCK_OWNED=0
acquire_single_simulation_lock() {
  mkdir -p "${PROJECT}/sh_test_mesh/runs"
  local inherited_fd="${SH_SINGLE_SIMULATION_LOCK_FD:-}"
  if [[ -n "${inherited_fd}" ]]; then
    if [[ ! "${inherited_fd}" =~ ^[0-9]+$ ]]; then
      echo "[smoke-matrix] invalid SH_SINGLE_SIMULATION_LOCK_FD=${inherited_fd@Q}" >&2
      return 1
    fi
    local fd_identity lock_identity
    fd_identity=$(stat -Lc '%d:%i' "/proc/${BASHPID}/fd/${inherited_fd}" 2>/dev/null) || {
      echo "[smoke-matrix] inherited simulation lock FD ${inherited_fd} is not open" >&2
      return 1
    }
    lock_identity=$(stat -Lc '%d:%i' "${SINGLE_SIMULATION_LOCK_PATH}") || return 1
    if [[ "${fd_identity}" != "${lock_identity}" ]] || ! flock -n "${inherited_fd}"; then
      echo "[smoke-matrix] inherited simulation lock FD does not hold ${SINGLE_SIMULATION_LOCK_PATH}" >&2
      return 1
    fi
    SINGLE_SIMULATION_LOCK_FD="${inherited_fd}"
  else
    exec {SINGLE_SIMULATION_LOCK_FD}>>"${SINGLE_SIMULATION_LOCK_PATH}"
    if ! flock -n "${SINGLE_SIMULATION_LOCK_FD}"; then
      exec {SINGLE_SIMULATION_LOCK_FD}>&-
      echo "[smoke-matrix] another simulation holds ${SINGLE_SIMULATION_LOCK_PATH}; refusing to mutate run inputs" >&2
      return 1
    fi
    SINGLE_SIMULATION_LOCK_OWNED=1
  fi
  export SH_SINGLE_SIMULATION_LOCK_FD="${SINGLE_SIMULATION_LOCK_FD}"
}
acquire_single_simulation_lock

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
finish_smoke_matrix() {
  local exit_status=$?
  if ! restore_pointer; then
    echo "[smoke-matrix] failed to restore trace_config pointer" >&2
    [[ ${exit_status} -ne 0 ]] || exit_status=1
  fi
  # Close the owning descriptor only after the EXIT restoration trap. Do not
  # LOCK_UN: a surviving child inherited the same open-file description and
  # must keep the repository protected until it exits too.
  if [[ ${SINGLE_SIMULATION_LOCK_OWNED} -eq 1 ]]; then
    exec {SINGLE_SIMULATION_LOCK_FD}>&-
  fi
  return "${exit_status}"
}
trap finish_smoke_matrix EXIT
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

# run_one_runner（C19）：与 run_one 同款证据链，但经 joint_runner.py
# （D14 推荐入口：仓内 flock/二进制 sha256/invocation.json/env 清洗）
# 起跑——三新臂走 runner CLI 面（--quota/--scheduler）；本函数无 env
# 前缀参数，开关证据 = <run_dir>/invocation.json 的 joint_switches +
# quota_mode_env + link_telemetry_injected（F7 注入事实），env.txt 快照
# 父进程继承环境备查。
run_one_runner() {
  local label=$1; shift
  local run_dir="${EVIDENCE_ROOT}/${label}"
  local runner_log="${EVIDENCE_ROOT}/${label}.runner.log"
  rm -rf "${run_dir}" "${runner_log}"; mkdir -p "${run_dir}"
  local rc=0
  python3 "${SCRIPT_DIR}/joint_runner.py" "${run_dir}" "${REQUEST_CSV}" "$@" \
    > "${runner_log}" 2>&1 || rc=$?
  mv "${runner_log}" "${run_dir}/run.log"
  env > "${run_dir}/env.txt" 2>&1 || true
  echo "${rc}" > "${run_dir}/exit_code"
  if [[ ${rc} -eq 0 ]]; then
    echo "[smoke-matrix] ${label}: PASS"
  else
    echo "[smoke-matrix] ${label}: FAIL (exit ${rc})——tail run.log:"
    tail -5 "${run_dir}/run.log" || true
  fi
}

for combo in none T J E TJ TE JE TJE; do
  run_one "combo_${combo}" JOINT_ABLATION_COMBO="${combo}"
done
run_one "TJE_remote_off" JOINT_ABLATION_COMBO=TJE JOINT_REMOTE_ACTIONS=off
run_one "TJE_shadow_verify" JOINT_ABLATION_COMBO=TJE \
  SH_ADMIT_GATE_VERIFY=1 SH_SNAPSHOT_VERIFY=1 SH_ONLINE_VALIDATE=1
# C19 三新臂（13 配置；runner CLI 面）：
# - TJE_quota_static：JOINT_QUOTA_MODE=static（固定预算配额门，off 臂
#   决策序列零漂移的对称对照——本臂 port_snapshot 实测值/quota 行键）；
# - TJE_quota_aimd：=aimd——F7 自动注入链全程生效（runner 置位
#   SH_LINK_TELEMETRY=1 → .sh 追加 --link-telemetry + K7/P2-9 缺省置
#   ASTRA_LINK_OBSERVER=1 → C++ 遥测差分有数据源 → AIMD 真闭环；
#   2026-09-23 前该臂数组恒空 = 无信号空转全绿——现默认真信号路径，
#   刻意跑 no-signal 对照须显式 ASTRA_LINK_OBSERVER=0）；
# - TJE_face_static：--scheduler face_static（policy variant，不进八
#   组合）+ --quota static（演示强制 quota-off：joint_config 显式覆盖
#   为 off 并 manifest 注记 quota_forced_off——对照臂语义的实证臂）。
run_one_runner "TJE_quota_static" --combo TJE --quota static
run_one_runner "TJE_quota_aimd"   --combo TJE --quota aimd
run_one_runner "TJE_face_static"  --scheduler face_static --quota static

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
