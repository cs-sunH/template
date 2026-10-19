#!/usr/bin/env bash
# run_online_wakeup_guard_fixture.sh -- 缺陷 C 回归 fixture(2026-08-16,
# face主动测试错误分析.md 缺陷 C:空队列分支唤醒条件)。
#
# 场景 1 f7-blueprint(F7 末态蓝图健康形态):
#   注入 r0 @T -> ARRIVAL 批携带 [同步完成控制节点(同 tick 里程碑)+
#   未到期 future alarm(r1 @T+1000)] -> 里程碑与 r1 ARRIVAL 在 T+1000
#   同一 epoch 交付 -> r0/r1 全部完成 -> CloseInput -> 双进程 exit 0。
#   断言:completed=2/accepted=1(future-alarm 到达不计 accepted)、
#   无事件丢失(gate counters delivery_count>=1)、future_alarm 调度留痕、无死锁。
#   ——证明修复后的空队列分支在健康蓝图形态上不误触发 fail-closed。
#
# 场景 2 defer-dead-end(唤醒真空死端):
#   注入 r0 @T -> 空批 defer(active=1、队列空、mailbox 空)-> 注入
#   CloseInput ->
#     EXPECT_MODE=fixed(缺省):C++ 立即 fail-closed abort,cpp.log 含
#       "lost-wakeup dead end" 且 active=1 计数入消息(替代 50 分钟静默
#       挂死/busy-spin);
#     EXPECT_MODE=legacy-stall:预修复二进制不退出(RED 基线采集用;
#       runner 断言 10s 无退出后打印 REPRODUCED 并以 0 退出;现场目录
#       留给调用方检查,由 EXIT trap kill -9(SIGKILL) 清理进程)。
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
# 相对 run_root 锚定调用者 cwd 预先解析为绝对路径（与 run_online_strategy.sh
# 同款，2026-09-29）：start_online 会 cd 到仓根/工作负载目录，放任相对形式
# 会让 rm/mkdir 与日志重定向的相对基准随 cd 漂移——中途崩溃且产物散落。
case "${RUN_ROOT}" in /*) ;; *) RUN_ROOT="$(pwd -P)/${RUN_ROOT}" ;; esac

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[wakeup_guard] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
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
  echo "[wakeup_guard] FAIL: cannot read hardware.capacity_profile from ${ET_DIR}/manifest.json (stale plan dir? re-run plan_materializer.py)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__no_memory_expansion)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" ]]; then
  echo "[wakeup_guard] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${BIN:-${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online}
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
  # --idle-watchdog-s 0 (2026-09-05): 看门狗默认已武装为 1s；本 fixture 的契约是刻意
  # IDLE 停车（request-neutral 无输入必须保持 IDLE 不退出），显式 0 恢复无界停车契约。
  "${BIN}" \
    --online-mode strategy \
    --bridge-dir "${run_dir}/bridge" \
    --online-validate 1 \
    --idle-watchdog-s 0 \
    --command-fifo "${run_dir}/cmd.fifo" \
    --workload-configuration="${ET_PREFIX}" \
    --comm-group-configuration="${RC}/comm_group.json" \
    --system-configuration="${RC}/system.json" \
    --network-configuration="${RC}/network.yml" \
    --logging-folder=off \
    > "${run_dir}/cpp.log" 2>&1 &
  CPP_PID=$!
  # FIFO 就绪等待有界化（2026-09-29，与 run_online_strategy.sh 同款）：原循环
  # 只防"C++ 退出"，C++ 存活但启动期挂死（既不建 FIFO 也不退出）时无界自旋。
  # 上限 120s（400×0.3s）；超时按监督纪律 SIGTERM C++ 后 fail-closed 退出。
  local fifo_waited=0
  until [ -p "${run_dir}/bridge/req_notify.fifo" ] && [ -p "${run_dir}/bridge/resp_notify.fifo" ]; do
    if ! kill -0 "${CPP_PID}" 2>/dev/null; then
      echo "[wakeup_guard] FAIL: C++ exited before FIFO ready" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    if [[ ${fifo_waited} -ge 400 ]]; then
      echo "[wakeup_guard] FAIL: ${run_dir}/bridge/{req_notify,resp_notify}.fifo not ready after $((fifo_waited / 10 * 3))s (${fifo_waited}x0.3s; cpp alive but startup hung; see cpp.log)" >&2
      kill "${CPP_PID}" 2>/dev/null || true
      return 1
    fi
    sleep 0.3
    fifo_waited=$((fifo_waited + 1))
  done
  cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
  python3 -u online/verify/wakeup_guard_fixture_service.py \
    --bridge-dir "${run_dir}/bridge" \
    --scenario "${scenario}" \
    > "${run_dir}/python.log" 2>&1 &
  PY_PID=$!
  sleep 0.5
  cd - >/dev/null
}

wait_cpp_exit() {  # $1=timeout_s -> 0 exited, 1 still running
  # monotonic 时钟（与 idle fixture 的 monotonic_ms 同因）：uutils date 0.8.0
  # 对 %N 系列有缺陷，宿主 date 实现差异会让 deadline 算术走错分支。
  local deadline=$(( $(python3 -c 'import time; print(time.monotonic_ns() // 1_000_000_000)') + $1 ))
  while kill -0 "${CPP_PID}" 2>/dev/null; do
    if [[ $(python3 -c 'import time; print(time.monotonic_ns() // 1_000_000_000)') -ge ${deadline} ]]; then
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
# 2026-09-30: no_decision_python_callback_count 计数器已随死代码清理删除
# (其自增钩子早已不存在,恒 0);"无事件丢失"的等价证据改为 gate counters 行
# 的 delivery_count(每个交付 epoch 都已计数)。
grep -q '\[online\] gate counters: ' "${run_dir}/cpp.log" || {
  echo "[wakeup_guard] FAIL(场景1): gate counters 行缺失" >&2; exit 1; }
grep '\[online\] gate counters: ' "${run_dir}/cpp.log" | grep -q 'delivery_count=[1-9]' || {
  echo "[wakeup_guard] FAIL(场景1): delivery_count 为 0(事件丢失)" >&2; exit 1; }
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
    echo "[wakeup_guard] legacy-stall: C++ 已退出(未复现挂死)——检查是否误用了修复后二进制" >&2
    exit 1
  fi
  echo "[wakeup_guard] legacy-stall REPRODUCED: 预修复二进制 CloseInput 后 10s+ 无退出"
  echo "[wakeup_guard]   (busy-spin/挂死现场保留于 ${run_dir};进程由 EXIT trap kill -9 清理)"
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
