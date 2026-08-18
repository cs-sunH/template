#!/usr/bin/env bash
# run_online_wakeup_guard_fixture.sh -- 缺陷 C 回归 fixture(2026-08-16,
# face主动测试错误分析.md 缺陷 C:空队列分支唤醒条件)。
#
# 场景 1 f7-blueprint(F7 末态蓝图健康形态):
#   注入 r0 @T -> ARRIVAL 批携带 [同步完成控制节点(同 tick 里程碑)+
#   未到期 future alarm(r1 @T+1000)] -> 里程碑与 r1 ARRIVAL 在 T+1000
#   同一 epoch 交付 -> r0/r1 全部完成 -> CloseInput -> 双进程 exit 0。
#   断言:completed=2/accepted=1(future-alarm 到达不计 accepted)、
#   no_decision=0、future_alarm 调度留痕、无死锁。
#   ——证明修复后的空队列分支在健康蓝图形态上不误触发 fail-closed。
#
# 场景 2 defer-dead-end(唤醒真空死端):
#   注入 r0 @T -> 空批 defer(active=1、队列空、mailbox 空)-> 注入
#   CloseInput ->
#     EXPECT_MODE=fixed(缺省):C++ 立即 fail-closed abort,cpp.log 含
#       "lost-wakeup dead end" 且 active=1 计数入消息(替代 50 分钟静默
#       挂死/busy-spin);
#     EXPECT_MODE=legacy-stall:预修复二进制不退出(RED 基线采集用;
#       runner 断言 N 秒无退出后 SIGTERM 记录现场)。
#
# 用法:
#   bash run_online_wakeup_guard_fixture.sh [run_root]
# 环境变量:
#   BIN           在线二进制(缺省本仓 build 产物)
#   EXPECT_MODE   fixed | legacy-stall(缺省 fixed)
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
RUN_ROOT=${1:-/tmp/face_wakeup_guard}

ET_PREFIX=${PROJECT}/sh_test_mesh/generated/llama2_7b_inference_54npus_face_case5_config_c_9inst_tp6_112sess_1177req_pc512_p66-169395_d1-13812_grequest_aggregated_q23936bbc_cd6682e3b/llama2_7b_inference
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__no_memory_expansion
BIN=${BIN:-${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online}
SVC=${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/online/verify/wakeup_guard_fixture_service.py
EXPECT_MODE=${EXPECT_MODE:-fixed}

T_NS=3000000000

CPP_PID=""
PY_PID=""

cleanup() {
  if [[ -n "${CPP_PID}" ]] && kill -0 "${CPP_PID}" 2>/dev/null; then
    kill -9 "${CPP_PID}" 2>/dev/null || true
  fi
  if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    kill -9 "${PY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

start_online() {  # $1=run_dir $2=scenario
  local run_dir=$1 scenario=$2
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}/bridge"
  mkfifo "${run_dir}/cmd.fifo"
  cd "${PROJECT}"
  "${BIN}" \
    --online-mode strategy \
    --bridge-dir "${run_dir}/bridge" \
    --command-fifo "${run_dir}/cmd.fifo" \
    --workload-configuration="${ET_PREFIX}" \
    --comm-group-configuration="${RC}/comm_group.json" \
    --system-configuration="${RC}/system.json" \
    --remote-memory-configuration="${RC}/remote_memory.json" \
    --network-configuration="${RC}/network.yml" \
    --logging-folder=off \
    > "${run_dir}/cpp.log" 2>&1 &
  CPP_PID=$!
  until [ -p "${run_dir}/bridge/req_notify.fifo" ] && [ -p "${run_dir}/bridge/resp_notify.fifo" ]; do
    if ! kill -0 "${CPP_PID}" 2>/dev/null; then
      echo "[wakeup_guard] FAIL: C++ exited before FIFO ready" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    sleep 0.3
  done
  cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
  python3 online/verify/wakeup_guard_fixture_service.py \
    --bridge-dir "${run_dir}/bridge" \
    --scenario "${scenario}" \
    > "${run_dir}/python.log" 2>&1 &
  PY_PID=$!
  sleep 0.5
  cd - >/dev/null
}

wait_cpp_exit() {  # $1=timeout_s -> 0 exited, 1 still running
  local deadline=$(( $(date +%s%N) / 1000000000 + $1 ))
  while kill -0 "${CPP_PID}" 2>/dev/null; do
    if [[ $(date +%s) -ge ${deadline} ]]; then
      return 1
    fi
    sleep 0.2
  done
  return 0
}

echo "[wakeup_guard] run root: ${RUN_ROOT}  BIN=${BIN}  EXPECT_MODE=${EXPECT_MODE}"

# ============================ 场景 1:f7-blueprint ============================
run_dir="${RUN_ROOT}/f7_blueprint"
echo "[wakeup_guard] scenario 1 (f7-blueprint): undue future alarm + same-tick milestone"
start_online "${run_dir}" f7-blueprint || exit 1
sleep 1

exec 3>"${run_dir}/cmd.fifo"
echo '{"kind":"Submit","session_id":"wg_s0","turn_index":0,"request_id":"wg_r0","prefill_length":4096,"decode_length":128,"arrival_world_ns":'"${T_NS}"',"inter_request_interval_ns":0}' >&3

# 等两请求全部完成(ACTIVE -> IDLE),上限 60s。
waited=0
while ! grep -q "ACTIVE -> IDLE" "${run_dir}/cpp.log"; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[wakeup_guard] FAIL(场景1): C++ died before completion" >&2
    tail -12 "${run_dir}/cpp.log" >&2; tail -12 "${run_dir}/python.log" >&2
    exec 3>&- 2>/dev/null || true; exit 1
  fi
  sleep 0.3; waited=$((waited + 1))
  if [[ ${waited} -gt 200 ]]; then
    echo "[wakeup_guard] FAIL(场景1): 60s 未完成(死锁?)" >&2
    tail -12 "${run_dir}/cpp.log" >&2; tail -12 "${run_dir}/python.log" >&2
    exec 3>&- 2>/dev/null || true; exit 1
  fi
done
echo '{"kind":"CloseInput"}' >&3
exec 3>&-

py_exit=0; wait "${PY_PID}" || py_exit=$?; PY_PID=""
cpp_exit=0; wait "${CPP_PID}" || cpp_exit=$?; CPP_PID=""
echo "[wakeup_guard] 场景1: cpp_exit=${cpp_exit} python_exit=${py_exit}"
if [[ ${cpp_exit} -ne 0 || ${py_exit} -ne 0 ]]; then
  echo "[wakeup_guard] FAIL(场景1): 非零退出" >&2
  tail -12 "${run_dir}/cpp.log" >&2; tail -12 "${run_dir}/python.log" >&2
  exit 1
fi
# 断言:future_alarm 留痕 + 两请求完成 + r1 到达为 future-alarm 路径
# (accepted=1,completed=2)+ 无事件丢失。
grep -q "\[fixture\] future_alarm scheduled: wg_r01 @ tick+1000" \
  "${run_dir}/python.log" || {
  echo "[wakeup_guard] FAIL(场景1): 未到期 future alarm 未调度" >&2; exit 1; }
grep -q '\[online\] service counters: accepted=1 completed=2 active=0 pending_alarm=0' \
  "${run_dir}/cpp.log" || {
  echo "[wakeup_guard] FAIL(场景1): 服务计数不符(期望 accepted=1 completed=2)" >&2
  grep '\[online\] service counters' "${run_dir}/cpp.log" >&2; exit 1; }
grep -q 'no_decision_python_callback_count=0' "${run_dir}/cpp.log" || {
  echo "[wakeup_guard] FAIL(场景1): no_decision != 0" >&2; exit 1; }
echo "[wakeup_guard] 场景1 PASS: 蓝图形态(未到期 alarm+同 tick 里程碑)健康完成,无误触发"

# ========================== 场景 2:defer-dead-end ==========================
run_dir="${RUN_ROOT}/defer_dead_end"
echo "[wakeup_guard] scenario 2 (defer-dead-end): deferred request + CloseInput"
start_online "${run_dir}" defer-dead-end || exit 1
sleep 1

exec 3>"${run_dir}/cmd.fifo"
echo '{"kind":"Submit","session_id":"wg_s1","turn_index":0,"request_id":"wg_def_r0","prefill_length":4096,"decode_length":128,"arrival_world_ns":'"${T_NS}"',"inter_request_interval_ns":0}' >&3
# 等 ARRIVAL 交付完成(defer 空批已 commit),再注入 CloseInput。
sleep 2
if ! kill -0 "${CPP_PID}" 2>/dev/null; then
  echo "[wakeup_guard] FAIL(场景2): C++ died before CloseInput" >&2
  tail -12 "${run_dir}/cpp.log" >&2; exit 1
fi
echo '{"kind":"CloseInput"}' >&3
exec 3>&-

if [[ "${EXPECT_MODE}" == "legacy-stall" ]]; then
  # RED 基线:预修复二进制应陷入无进展(不退出)。10s 观察窗。
  if wait_cpp_exit 10; then
    echo "[wakeup_guard] legacy-stall: C++ 已退出(未复现挂死;退出码 $( \
      grep -c . /dev/null; echo '?'))——检查是否误用了修复后二进制" >&2
    exit 1
  fi
  echo "[wakeup_guard] legacy-stall REPRODUCED: 预修复二进制 CloseInput 后 10s+ 无退出"
  echo "[wakeup_guard]   (busy-spin/挂死现场保留于 ${run_dir};由 trap SIGTERM 清理)"
  exit 0
fi

# fixed:C++ 应在数秒内 fail-closed abort(lost-wakeup dead end 诊断)。
if ! wait_cpp_exit 15; then
  echo "[wakeup_guard] FAIL(场景2): C++ 15s 未退出(仍是挂死/spin?)" >&2
  tail -12 "${run_dir}/cpp.log" >&2; exit 1
fi
cpp_exit=0; wait "${CPP_PID}" || cpp_exit=$?; CPP_PID=""
py_exit=0; wait "${PY_PID}" 2>/dev/null || py_exit=$?; PY_PID=""
echo "[wakeup_guard] 场景2: cpp_exit=${cpp_exit} python_exit=${py_exit}"
grep -q "lost-wakeup dead end" "${run_dir}/cpp.log" || {
  echo "[wakeup_guard] FAIL(场景2): cpp.log 缺 lost-wakeup dead end 诊断" >&2
  tail -12 "${run_dir}/cpp.log" >&2; exit 1; }
grep -q "lost-wakeup dead end.*active=1" "${run_dir}/cpp.log" || {
  echo "[wakeup_guard] FAIL(场景2): 诊断未携带 active=1 计数" >&2
  grep "lost-wakeup" "${run_dir}/cpp.log" >&2; exit 1; }
echo "[wakeup_guard] 场景2 PASS: 唤醒真空死端 fail-closed 诊断(替代静默挂死)"
echo "[wakeup_guard] ALL PASS (EXPECT_MODE=${EXPECT_MODE})"
exit 0
