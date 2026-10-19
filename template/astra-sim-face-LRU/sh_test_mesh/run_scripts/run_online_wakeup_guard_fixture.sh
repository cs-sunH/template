#!/usr/bin/env bash
# run_online_wakeup_guard_fixture.sh -- 缺陷 C 回归 fixture(2026-08-16,
# face主动测试错误分析.md 缺陷 C:空队列分支唤醒条件;
# "face主动测试错误分析.md"为工作区根文档,不在本仓内)。
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
# (2026-10-01) -e 补齐:与另两个 fixture 同级防护——exec 3> FIFO
# 打开失败等未防护命令此前被静默跳过,依赖后续 grep 断言间接兜底。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
# T5(face-LRU 审计,2026-09-29):run root 绝对化——start_online 内部两
# 次 cd 后仍以相对路径解析日志重定向/mkdir/tail 会劈叉(set -e 必失败)。
RUN_ROOT=$(realpath -m "${1:-/tmp/face_wakeup_guard}")

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[wakeup_guard] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_inference"
# RC = runtime_config 目录动态解析（sync-py-E11，原为硬编码目录名）：目录名由
# 配置装载器生成为 "<hardware slug>__<capacity_profile>__<remote-memory 标签>"
# （本仓命名规则尾段恒为 edge_remote_memory_pool）。face 仓从 plan 目录
# manifest.json 的 hardware.capacity_profile 取档位；本仓 manifest.json 不写
# hardware 节（字段冻结），权威同源退路 = trace_config.csv 的
# config,local_hbm_capacity_profile 行。按 profile 过滤并要求恰好一个，
# 缺失/歧义 fail-closed（与 GEN_MATCH 口径一致）。
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
  echo "[wakeup_guard] FAIL: cannot read local_hbm_capacity_profile from trace_config.csv (sync-py-E11)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__edge_remote_memory_pool)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" \
      || ! -f "${RC_MATCH[0]}/remote_memory.json" ]]; then
  echo "[wakeup_guard] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json/remote_memory.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${BIN:-${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online}
EXPECT_MODE=${EXPECT_MODE:-fixed}

T_NS=3000000000

CPP_PID=""
PY_PID=""

cleanup() {
  # L21(2026-09-29 审查修复):先 SIGTERM 后 SIGKILL——此前一律 -9 剥夺了
  # C++/Python 的信号处理与日志/core 取证窗口(本脚本 legacy-stall 模式的
  # 现场保留语义恰依赖 SIGTERM,见下方注释)。宽限 ~3s 后 -9 兜底。
  if [[ -n "${CPP_PID}" ]] && kill -0 "${CPP_PID}" 2>/dev/null; then
    kill "${CPP_PID}" 2>/dev/null || true
    local _i
    for _i in $(seq 1 30); do
      kill -0 "${CPP_PID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -9 "${CPP_PID}" 2>/dev/null || true
  fi
  if [[ -n "${PY_PID}" ]] && kill -0 "${PY_PID}" 2>/dev/null; then
    kill "${PY_PID}" 2>/dev/null || true
    for _i in $(seq 1 30); do
      kill -0 "${PY_PID}" 2>/dev/null || break
      sleep 0.1
    done
    kill -9 "${PY_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

# M20(2026-09-29 审查修复):有界等待原语——$1=pid $2=timeout_s;0=已退出,
# 1=超时仍在运行。计数式轮询(sleep 0.3，约 3⅓ 次/秒，实际等待上界≈名义值×1.2（方向保守）),不依赖 date/%3N 语义。
wait_pid_bounded() {
  local pid=$1 waited=0 max_waited=$(( $2 * 4 ))
  while kill -0 "${pid}" 2>/dev/null; do
    if (( waited >= max_waited )); then return 1; fi
    sleep 0.3
    waited=$((waited + 1))
  done
  return 0
}

start_online() {  # $1=run_dir $2=scenario
  local run_dir=$1 scenario=$2
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}/bridge"
  mkfifo "${run_dir}/cmd.fifo"
  cd "${PROJECT}"
  # --idle-watchdog-s 0 (2026-09-05): 看门狗默认已武装为 1s；本 fixture 的契约是刻意
  # IDLE 停车（request-neutral 无输入必须保持 IDLE 不退出），显式 0 恢复无界停车契约。
  # M20(2026-09-29 审查修复):桥侧 response-poll 看门狗补武装(此前缺省
  # 不传=0=永等)。仅 EXPECT_MODE=fixed(修复后二进制)注入;legacy-stall 采
  # RED 基线用的预修复二进制可能早于 2026-08-22 该旗标引入,保持旧调用
  # 字节不变。120000ms 为宽松有界值(C++ 语义:毫秒;正常夹具决策毫秒级,
  # 永不触发,语义不变)。infra-E5(2026-09-30):env 名双兼容——README 文件化
  # 的 BRIDGE_TIMEOUT_MS 优先,既有 SH_BRIDGE_TIMEOUT_MS 作为旧别名保留
  # (二者均未设时缺省 120000;显式 0=恢复永等)。
  local bridge_timeout_args=()
  if [[ "${EXPECT_MODE}" != "legacy-stall" ]]; then
    bridge_timeout_args=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS:-${SH_BRIDGE_TIMEOUT_MS:-120000}}")
  fi
  "${BIN}" \
    --online-mode strategy \
    --bridge-dir "${run_dir}/bridge" \
    --online-validate 1 \
    --idle-watchdog-s 0 \
    "${bridge_timeout_args[@]}" \
    --command-fifo "${run_dir}/cmd.fifo" \
    --workload-configuration="${ET_PREFIX}" \
    --comm-group-configuration="${RC}/comm_group.json" \
    --system-configuration="${RC}/system.json" \
    --remote-memory-configuration="${RC}/remote_memory.json" \
    --network-configuration="${RC}/network.yml" \
    --logging-folder=off \
    > "${run_dir}/cpp.log" 2>&1 &
  CPP_PID=$!
  # FIFO 就绪等待带墙钟预算（infra-F2，SH_FIFO_WAIT_S 可覆盖，缺省 60s；
  # 同 run_online_strategy.sh 注）。infra-E10：截止时钟用本文件既有的
  # monotonic_ms 原语（墙钟跳变不漂移，见下方 uutils date 语义裁定）。
  local fifo_deadline=$(( $(monotonic_ms) + ${SH_FIFO_WAIT_S:-60} * 1000 ))
  until [ -p "${run_dir}/bridge/req_notify.fifo" ] && [ -p "${run_dir}/bridge/resp_notify.fifo" ]; do
    if ! kill -0 "${CPP_PID}" 2>/dev/null; then
      echo "[wakeup_guard] FAIL: C++ exited before FIFO ready" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    if (( $(monotonic_ms) > fifo_deadline )); then
      echo "[wakeup_guard] FAIL: FIFOs not ready within ${SH_FIFO_WAIT_S:-60}s (see ${run_dir}/cpp.log tail)" >&2
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

# uutils date 0.8.0 treats %3N as an untruncated nanosecond field, which makes
# elapsed-millisecond arithmetic overflow or report absurd values; wall-clock
# deadlines also jump with system time changes.  Use the same monotonic clock
# as run_online_idle_fixture.sh / run_online_same_tick_milestone.sh.
monotonic_ms() {
  python3 -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

wait_cpp_exit() {  # $1=timeout_s -> 0 exited, 1 still running
  local deadline=$(( $(monotonic_ms) + $1 * 1000 ))
  while kill -0 "${CPP_PID}" 2>/dev/null; do
    if [[ $(monotonic_ms) -ge ${deadline} ]]; then
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

# M20(2026-09-29 审查修复):场景1 收尾的裸 wait 无界(Python 服务或 C++ 挂死
# 时回归被掩盖为挂起)——先有界轮询(SH_EXIT_WAIT_S,缺省 120s),超时
# fail-closed(非零退出触发 EXIT trap 清场);正常路径立即通过,退出码仍由
# wait 收割,语义不变。
py_exit=0
if ! wait_pid_bounded "${PY_PID}" "${SH_EXIT_WAIT_S:-120}"; then
  echo "[wakeup_guard] FAIL(场景1): Python ${SH_EXIT_WAIT_S:-120}s 未退出(挂死?)" >&2
  tail -12 "${run_dir}/python.log" >&2; exit 1
fi
wait "${PY_PID}" || py_exit=$?; PY_PID=""
cpp_exit=0
if ! wait_pid_bounded "${CPP_PID}" "${SH_EXIT_WAIT_S:-120}"; then
  echo "[wakeup_guard] FAIL(场景1): C++ ${SH_EXIT_WAIT_S:-120}s 未退出(挂死?)" >&2
  tail -12 "${run_dir}/cpp.log" >&2; exit 1
fi
wait "${CPP_PID}" || cpp_exit=$?; CPP_PID=""
echo "[wakeup_guard] 场景1: cpp_exit=${cpp_exit} python_exit=${py_exit}"
if [[ ${cpp_exit} -ne 0 || ${py_exit} -ne 0 ]]; then
  echo "[wakeup_guard] FAIL(场景1): 非零退出" >&2
  tail -12 "${run_dir}/cpp.log" >&2; tail -12 "${run_dir}/python.log" >&2
  exit 1
fi
# 断言:future_alarm 留痕 + 两请求完成 + r1 到达为 future-alarm 路径
# (accepted=1,completed=2)+ run-end 门禁通过(cpp 退出码非 0 即失败)。
grep -q "\[fixture\] future_alarm scheduled: wg_r01 @ tick+1000" \
  "${run_dir}/python.log" || {
  echo "[wakeup_guard] FAIL(场景1): 未到期 future alarm 未调度" >&2; exit 1; }
grep -q '\[online\] service counters: accepted=1 completed=2 active=0 pending_alarm=0' \
  "${run_dir}/cpp.log" || {
  echo "[wakeup_guard] FAIL(场景1): 服务计数不符(期望 accepted=1 completed=2)" >&2
  grep '\[online\] service counters' "${run_dir}/cpp.log" >&2; exit 1; }
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
