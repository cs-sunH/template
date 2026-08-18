#!/usr/bin/env bash
# sh_1.0 phase-1 step-1-10 IDLE/注入 lifecycle fixture(合同② 目标 5)。
#
# 五态迁移:IDLE --(注入 2 request,指定世界 tick T1/T2)--> ACTIVE --(完成)-->
#   IDLE --(注入关闭)--> DRAINING --> FINISHED(退出 0),全程记录
#   "[online] lifecycle:" 迁移日志与时间(cpp.log 的 t= 与 wall_ms=)。
#
# Scenario 1(无请求启动):不提供 --request-queue-csv -> IDLE 保持 2 秒不退出
#   (request-neutral 默认:无 source 参数必须保持 IDLE,不得回退到任何预置
#   队列);经 --command-fifo 注入关闭 -> DRAINING -> FINISHED。
# Scenario 2(注入):同样 IDLE 2 秒 -> 经 --command-fifo 注入 2 个 request
#   (arrival_world_ns = T1/T2 指定世界 tick)-> ACTIVE -> 完成 -> IDLE ->
#   注入关闭 -> DRAINING -> FINISHED。
#
# 注入通道:外部 Producer 线程读 --command-fifo,只写线程安全有界 command
#   queue(合同② 线程合同);决策桥接仍是纯决策通道(步骤 1-7 红线)。
# 默认队列 stub 不出现在任何正式 runner 中(fail-closed 已由步骤 0-1 保证)。
#
# Usage: bash run_online_idle_fixture.sh [run_root]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
RUN_ROOT=${1:-/tmp/sh10_idle_fixture}

# ET 目录动态解析(阶段 7 官方 runner 同款;2026-08-16 缺陷修复2同步轮补齐——
# 本 fixture runner 曾硬编码已删除的 baseline/20_30s 归档路径)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[fixture] expected exactly one generated dir under sh_test_mesh/generated (regenerate via generate_trace.sh after materializing the input; see traces/PROVENANCE.md), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_inference"
RC=${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c__validation-160gib__edge_remote_memory_pool
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
FIXTURE_SVC=${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/online/verify/lifecycle_fixture_service.py

# 指定世界 tick(纳秒,事件时钟从 0 起):注入在 wall ~2s,到达在事件 3.0s。
# 两个 request 用同一指定 tick:同 tick 到达 -> 单次 ACTIVE 周期(合同② 冻结
# 五态迁移 IDLE->ACTIVE->IDLE->DRAINING->FINISHED);若用不同 tick,前一
# request 完成(1ns 节点)后服务已回 IDLE,后一 request 会再拉一次
# ACTIVE->IDLE,状态序列变 7 段,不再是"五态迁移"。
T1_NS=3000000000
T2_NS=3000000000

CPP_PID=""
PY_PID=""

cleanup() {
  if [[ -n "${CPP_PID}" ]] && kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
  fi
  if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    kill "${PY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# cpp.log 的 "[online] lifecycle:" 行 -> 状态序列("IDLE ACTIVE ...")
# 注:C++ 的 "command-fifo: EOF" 消息不带结尾换行,后续 lifecycle 行会拼接在
# 同一物理行上,故用 grep -o 做行内提取(不以 ^ 锚定行首)。
lifecycle_states() {
  grep -o '\[online\] lifecycle: .*' "$1" | sed 's/^\[online\] lifecycle: //' | awk '
    /^start / { print $2; next }
    { print $3 }
  ' | tr '\n' ' ' | sed 's/ $//'
}

start_online() {  # $1=run_dir; sets CPP_PID / PY_PID
  local run_dir=$1
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
      echo "[fixture] C++ exited before FIFO ready (see ${run_dir}/cpp.log)" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    sleep 0.3
  done
  cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
  python3 online/verify/lifecycle_fixture_service.py \
    --bridge-dir "${run_dir}/bridge" \
    > "${run_dir}/python.log" 2>&1 &
  PY_PID=$!
  sleep 0.5
}

assert_processes_alive() {  # $1=run_dir $2=scenario
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[fixture] FAIL: C++ exited during the IDLE hold (${2})" >&2
    tail -5 "$1/cpp.log" >&2
    exit 1
  fi
  if ! kill -0 "${PY_PID}" 2>/dev/null; then
    echo "[fixture] FAIL: Python exited during the IDLE hold (${2})" >&2
    tail -5 "$1/python.log" >&2
    exit 1
  fi
}

wait_for_exits() {  # $1=run_dir $2=scenario
  local run_dir=$1 name=$2
  local cpp_exit=0 py_exit=0
  wait "${CPP_PID}" || cpp_exit=$?
  wait "${PY_PID}" || py_exit=$?
  CPP_PID=""
  PY_PID=""
  echo "[fixture] ${name}: cpp_exit=${cpp_exit} python_exit=${py_exit}"
  if [[ ${cpp_exit} -ne 0 ]]; then
    tail -8 "${run_dir}/cpp.log" >&2
  fi
  if [[ ${py_exit} -ne 0 ]]; then
    tail -8 "${run_dir}/python.log" >&2
  fi
  if [[ ${cpp_exit} -ne 0 || ${py_exit} -ne 0 ]]; then
    echo "[fixture] FAIL: ${name} exits non-zero" >&2
    exit 1
  fi
}

scenario1_idle_hold_close_only() {
  local run_dir="${RUN_ROOT}/scenario1"
  echo "[fixture] scenario 1: 无请求启动(无 --request-queue-csv)-> IDLE 保持 2s -> 注入关闭 -> DRAINING -> FINISHED"
  start_online "${run_dir}" || exit 1
  local t_before t_after
  t_before=$(date +%s%3N)
  sleep 2
  t_after=$(date +%s%3N)
  assert_processes_alive "${run_dir}" scenario1
  local states
  states=$(lifecycle_states "${run_dir}/cpp.log")
  if [[ "${states}" != "IDLE" ]]; then
    echo "[fixture] FAIL: after 2s hold expected state sequence 'IDLE', got '${states}'" >&2
    exit 1
  fi
  echo "[fixture] scenario 1: IDLE held $((t_after - t_before))ms (processes alive, state=IDLE, no preset queue)"

  exec 3>"${run_dir}/cmd.fifo"
  echo '{"kind":"CloseInput"}' >&3
  exec 3>&-
  wait_for_exits "${run_dir}" scenario1

  states=$(lifecycle_states "${run_dir}/cpp.log")
  echo "[fixture] scenario 1 transitions: ${states}"
  if [[ "${states}" != "IDLE DRAINING FINISHED" ]]; then
    echo "[fixture] FAIL: scenario 1 expected 'IDLE DRAINING FINISHED'" >&2
    exit 1
  fi
  echo "[fixture] scenario 1 PASS"
}

scenario2_inject_two() {
  local run_dir="${RUN_ROOT}/scenario2"
  echo "[fixture] scenario 2: 无请求启动 -> IDLE 2s -> 注入 2 request(世界 tick ${T1_NS}) -> ACTIVE -> 完成 -> IDLE -> 注入关闭 -> DRAINING -> FINISHED"
  start_online "${run_dir}" || exit 1
  sleep 2
  assert_processes_alive "${run_dir}" scenario2
  local states
  states=$(lifecycle_states "${run_dir}/cpp.log")
  if [[ "${states}" != "IDLE" ]]; then
    echo "[fixture] FAIL: after 2s hold expected state sequence 'IDLE', got '${states}'" >&2
    exit 1
  fi
  echo "[fixture] scenario 2: IDLE held 2s (processes alive, state=IDLE); injecting 2 requests"

  local inject_ms
  inject_ms=$(date +%s%3N)
  exec 3>"${run_dir}/cmd.fifo"
  echo '{"kind":"Submit","session_id":"fixture_s0","turn_index":0,"request_id":"fixture_s0_r0","prefill_length":4096,"decode_length":128,"arrival_world_ns":'"${T1_NS}"',"inter_request_interval_ns":0}' >&3
  echo '{"kind":"Submit","session_id":"fixture_s1","turn_index":0,"request_id":"fixture_s1_r0","prefill_length":2048,"decode_length":256,"arrival_world_ns":'"${T2_NS}"',"inter_request_interval_ns":0}' >&3

  # 等待两个 request 完成(服务回到 IDLE)再注入关闭:轮询 cpp.log,上限 60s
  local waited=0
  while ! grep -q "ACTIVE -> IDLE" "${run_dir}/cpp.log"; do
    if ! kill -0 "${CPP_PID}" 2>/dev/null; then
      echo "[fixture] FAIL: C++ died while waiting for completion" >&2
      tail -8 "${run_dir}/cpp.log" >&2
      exec 3>&- 2>/dev/null || true
      exit 1
    fi
    sleep 0.3
    waited=$((waited + 1))
    if [[ ${waited} -gt 200 ]]; then
      echo "[fixture] FAIL: timeout waiting for 'ACTIVE -> IDLE' (requests never completed)" >&2
      tail -8 "${run_dir}/cpp.log" >&2
      exec 3>&- 2>/dev/null || true
      exit 1
    fi
  done
  local complete_ms
  complete_ms=$(date +%s%3N)
  echo "[fixture] scenario 2: both requests completed at +$((complete_ms - inject_ms))ms; injecting CloseInput"

  echo '{"kind":"CloseInput"}' >&3
  exec 3>&-
  wait_for_exits "${run_dir}" scenario2

  states=$(lifecycle_states "${run_dir}/cpp.log")
  echo "[fixture] scenario 2 transitions: ${states}"
  if [[ "${states}" != "IDLE ACTIVE IDLE DRAINING FINISHED" ]]; then
    echo "[fixture] FAIL: scenario 2 expected 'IDLE ACTIVE IDLE DRAINING FINISHED'" >&2
    exit 1
  fi
  # 指定世界 tick:python.log 必须记录两个 request 的 arrival_world_ns 均恰好
  # 为 T1_NS。同 tick 到达在单次 delivery 内合批(seq=0 一行含两个 arrival),
  # 所以按"出现次数"数(每 arrival 一次),不按行数。
  local arrival_hits
  arrival_hits=$(grep -o '"arrival_world_ns": '"${T1_NS}" "${run_dir}/python.log" | wc -l)
  if [[ ${arrival_hits} -ne 2 ]]; then
    echo "[fixture] FAIL: expected exactly 2 arrivals at world tick ${T1_NS} (got ${arrival_hits})" >&2
    exit 1
  fi
  echo "[fixture] scenario 2 PASS (both arrivals at exactly ${T1_NS} ns)"
}

echo "[fixture] run root: ${RUN_ROOT}"
scenario1_idle_hold_close_only
scenario2_inject_two

echo "[fixture] lifecycle timeline (scenario 1):"
grep "\[online\] lifecycle:" "${RUN_ROOT}/scenario1/cpp.log"
echo "[fixture] lifecycle timeline (scenario 2):"
grep "\[online\] lifecycle:" "${RUN_ROOT}/scenario2/cpp.log"
echo "[fixture] ALL PASS: 五态迁移 IDLE/ACTIVE/IDLE/DRAINING/FINISHED 日志与时间齐全"
