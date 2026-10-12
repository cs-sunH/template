#!/usr/bin/env bash
# face phase-1 step-1-11(CORE)same-tick milestone fixture(方案 step 1-11
# / 仿真加速分析.md §4.3 最高优先级专门 fixture;
# "仿真加速分析.md"为工作区根文档,不在本仓内)。
#
# 场景:一个 request 的 prefill 阶段末尾含一个真实同步完成的控制节点
# (METADATA_NODE,type=1,正常 issue 路径——不是零时长 compute 节点,
# Workload::issue_replay 会把 runtime_ns==0 强制为 1,完成 tick 不再是 T),
# 使其在 commit 的同一 tick 内经 deferred drain 立即完成,触发
# PREFILL_DRAIN。deferred drain 结束后主队列为空但 mailbox 非空 -> 主循环
# 必须排一个显式的下一决策边界(schedule_event(T+1, delivery wakeup))并在
# StateDelta 记录 T->T+1 延后(deferred_from_tick)——否则 wait_for_work 会
# 永久阻塞、milestone 永不交付。
#
# 断言(方案 step 1-11 的 a/b/c/d):
#   (a) PREFILL_DRAIN 不在本次 Python 调用内被处理:ARRIVAL delivery
#       (seq=0) 的 reasons 恰为 ["ARRIVAL"],且没有任何 delivery 同时含
#       ARRIVAL 与 PREFILL_DRAIN(单 tick 单次 delivery,Python 不可重入);
#   (b) 下一 delivery 的唤醒机制显式存在:seq=1 的 delivery 为 tick=T+1
#       且 deferred_from_tick=T(StateDelta 显式记录 T->T+1 延后);
#   (c) milestone 在下一 delivery epoch 交付,decode 图随后正常提交:
#       seq=2 的 delivery reasons 恰为 ["DECODE_COMPLETION",
#       "REQUEST_COMPLETE"],cpp 侧 completed=1,双进程 exit 0;
#   (d) 无事件丢失(cpp run-end gate_ok 审计,退出码非 0 即失败)、
#       无死锁(轮询有界),request 最终完成。
#
# Usage: bash run_online_same_tick_milestone.sh [run_root]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
# T5(face-LRU 审计,2026-09-29):run root 绝对化——start_online 内部两
# 次 cd 后仍以相对路径解析日志重定向/mkdir/tail 会劈叉(set -e 必失败)。
RUN_ROOT=$(realpath -m "${1:-/tmp/face_same_tick_milestone}")

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[fixture] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
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
  echo "[fixture] FAIL: cannot read local_hbm_capacity_profile from trace_config.csv (sync-py-E11)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__edge_remote_memory_pool)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" \
      || ! -f "${RC_MATCH[0]}/remote_memory.json" ]]; then
  echo "[fixture] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json/remote_memory.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online

# 指定世界 tick(纳秒,事件时钟从 0 起):注入在 wall ~1s,到达在事件 3.0s。
T_NS=3000000000
T_PLUS1=$((T_NS + 1))
T_PLUS2=$((T_NS + 2))

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

# uutils date 0.8.0 treats %3N as an untruncated nanosecond field, which makes
# elapsed-millisecond arithmetic overflow or report absurd values.  Use a
# monotonic clock with explicit integer conversion on every supported host
# (same as run_online_idle_fixture.sh).
monotonic_ms() {
  python3 -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

# M20(2026-09-29 审查修复):有界等待原语——$1=pid $2=timeout_s;0=已退出,
# 1=超时仍在运行。计数式轮询(sleep 0.3，约 3⅓ 次/秒，实际等待上界≈名义值×1.2（方向保守）,与下方 ACTIVE->IDLE 等待同
# idiom),不依赖 date/%3N 语义。
wait_pid_bounded() {
  local pid=$1 waited=0 max_waited=$(( $2 * 4 ))
  while kill -0 "${pid}" 2>/dev/null; do
    if (( waited >= max_waited )); then return 1; fi
    sleep 0.3
    waited=$((waited + 1))
  done
  return 0
}

start_online() {  # $1=run_dir; sets CPP_PID / PY_PID
  local run_dir=$1
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}/bridge"
  mkfifo "${run_dir}/cmd.fifo"
  cd "${PROJECT}"
  # --idle-watchdog-s 0 (2026-09-05): 看门狗默认已武装为 1s；本 fixture 的契约是刻意
  # IDLE 停车（request-neutral 无输入必须保持 IDLE 不退出），显式 0 恢复无界停车契约。
  # M20(2026-09-29 审查修复):桥侧 response-poll 看门狗补武装(此前缺省
  # 不传=0=永等)——Python 夹具服务单次决策交换停滞时 fail-closed abort 而
  # 非无界挂死。120000ms 为宽松有界值(C++ 语义:毫秒,须大于最慢单决策
  # 耗时;本夹具决策为毫秒级,正常路径永不触发,语义不变)。
  # infra-E5(2026-09-30):env 名双兼容——README 文件化的 BRIDGE_TIMEOUT_MS
  # 优先,既有 SH_BRIDGE_TIMEOUT_MS 作为旧别名保留(二者均未设时缺省
  # 120000;显式 0=恢复永等逃生口)。
  "${BIN}" \
    --online-mode strategy \
    --bridge-dir "${run_dir}/bridge" \
    --online-validate 1 \
    --idle-watchdog-s 0 \
    --bridge-timeout-ms "${BRIDGE_TIMEOUT_MS:-${SH_BRIDGE_TIMEOUT_MS:-120000}}" \
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
  # monotonic_ms 原语（墙钟跳变不漂移，见上方 uutils date 语义裁定）。
  local fifo_deadline=$(( $(monotonic_ms) + ${SH_FIFO_WAIT_S:-60} * 1000 ))
  until [ -p "${run_dir}/bridge/req_notify.fifo" ] && [ -p "${run_dir}/bridge/resp_notify.fifo" ]; do
    if ! kill -0 "${CPP_PID}" 2>/dev/null; then
      echo "[fixture] C++ exited before FIFO ready (see ${run_dir}/cpp.log)" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    if (( $(monotonic_ms) > fifo_deadline )); then
      echo "[fixture] FIFOs not ready within ${SH_FIFO_WAIT_S:-60}s (see ${run_dir}/cpp.log tail)" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    sleep 0.3
  done
  cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
  python3 online/verify/same_tick_milestone_fixture_service.py \
    --bridge-dir "${run_dir}/bridge" \
    > "${run_dir}/python.log" 2>&1 &
  PY_PID=$!
  sleep 0.5
}

wait_for_exits() {  # $1=run_dir $2=name
  local run_dir=$1 name=$2
  local cpp_exit=0 py_exit=0
  # M20(2026-09-29 审查修复):裸 wait 无界——CloseInput 丢失/夹具服务回归时
  # 本 fixture 挂死而非失败。改为有界轮询(SH_EXIT_WAIT_S,缺省 120s)后再
  # wait 收割退出码;超时 fail-closed(非零退出触发 EXIT trap 清场)。正常
  # 路径进程自然退出,轮询立即返回,退出码语义不变。
  if ! wait_pid_bounded "${CPP_PID}" "${SH_EXIT_WAIT_S:-120}"; then
    echo "[fixture] FAIL: ${name} C++ did not exit within ${SH_EXIT_WAIT_S:-120}s (CloseInput lost?)" >&2
    tail -8 "${run_dir}/cpp.log" >&2
    exit 1
  fi
  if ! wait_pid_bounded "${PY_PID}" "${SH_EXIT_WAIT_S:-120}"; then
    echo "[fixture] FAIL: ${name} Python did not exit within ${SH_EXIT_WAIT_S:-120}s" >&2
    tail -8 "${run_dir}/python.log" >&2
    exit 1
  fi
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

echo "[fixture] run root: ${RUN_ROOT}"
echo "[fixture] scenario: 注入 1 request(世界 tick ${T_NS})-> prefill=[同步完成"
echo "[fixture]   控制节点 METADATA]-> 同 tick 形成 PREFILL_DRAIN -> 显式 T+1"
echo "[fixture]   唤醒交付(deferred_from_tick=${T_NS} 记录)-> decode 提交 -> 完成"

run_dir="${RUN_ROOT}/scenario1"
start_online "${run_dir}" || exit 1
sleep 1
if ! kill -0 "${CPP_PID}" 2>/dev/null; then
  echo "[fixture] FAIL: C++ exited during the IDLE hold" >&2
  tail -8 "${run_dir}/cpp.log" >&2
  exit 1
fi
if ! kill -0 "${PY_PID}" 2>/dev/null; then
  echo "[fixture] FAIL: Python exited during the IDLE hold" >&2
  tail -8 "${run_dir}/python.log" >&2
  exit 1
fi
echo "[fixture] IDLE held ~1s (processes alive); injecting 1 request"

inject_ms=$(monotonic_ms)
exec 3>"${run_dir}/cmd.fifo"
echo '{"kind":"Submit","session_id":"stm_s0","turn_index":0,"request_id":"stm_s0_r0","prefill_length":4096,"decode_length":128,"arrival_world_ns":'"${T_NS}"',"inter_request_interval_ns":0}' >&3

# 等待 request 完成(服务 ACTIVE -> IDLE)再注入关闭:轮询 cpp.log,上限 60s
waited=0
while ! grep -q "ACTIVE -> IDLE" "${run_dir}/cpp.log"; do
  if ! kill -0 "${CPP_PID}" 2>/dev/null; then
    echo "[fixture] FAIL: C++ died while waiting for completion" >&2
    tail -12 "${run_dir}/cpp.log" >&2
    exec 3>&- 2>/dev/null || true
    exit 1
  fi
  sleep 0.3
  waited=$((waited + 1))
  if [[ ${waited} -gt 200 ]]; then
    echo "[fixture] FAIL: timeout waiting for 'ACTIVE -> IDLE' "
         "(milestone never delivered / deadlock?)" >&2
    tail -12 "${run_dir}/cpp.log" >&2
    echo "--- python.log ---" >&2
    tail -12 "${run_dir}/python.log" >&2
    exec 3>&- 2>/dev/null || true
    exit 1
  fi
done
echo "[fixture] request completed at +$(( $(monotonic_ms) - inject_ms ))ms; injecting CloseInput"
echo '{"kind":"CloseInput"}' >&3
exec 3>&-
wait_for_exits "${run_dir}" scenario1

# ---- 断言 (a)/(b)/(c):python.log 的 delivery 序列 ----
log="${run_dir}/python.log"
grep "\[fixture\] delivery" "${log}" > "${run_dir}/deliveries.txt" || true
cat "${run_dir}/deliveries.txt"

# (a) ARRIVAL delivery(seq=0):tick=T,deferred_from_tick=0,reasons 恰为
#     ["ARRIVAL"] —— PREFILL_DRAIN 不在本次 Python 调用内(单 tick 单次
#     delivery,Python 不可重入)。
if ! grep -q 'seq=0 tick='"${T_NS}"' deferred_from_tick=0 reasons=\["ARRIVAL"\]' "${log}"; then
  echo "[fixture] FAIL (a): ARRIVAL delivery(seq=0)缺失或内容不符" >&2
  exit 1
fi
# (b) PREFILL_DRAIN delivery(seq=1):tick=T+1 且 deferred_from_tick=T ——
#     显式 T+1 下一决策边界唤醒,StateDelta 记录 T->T+1 延后。
if ! grep -q 'seq=1 tick='"${T_PLUS1}"' deferred_from_tick='"${T_NS}"' reasons=\["PREFILL_DRAIN"\]' "${log}"; then
  echo "[fixture] FAIL (b): T+1 唤醒交付缺失或未显式记录延后" >&2
  exit 1
fi
# (c) decode 提交 delivery(seq=2):reasons 恰为
#     ["DECODE_COMPLETION", "REQUEST_COMPLETE"](json.dumps 的 ", " 分隔)。
if ! grep -q 'seq=2 tick='"${T_PLUS2}"' deferred_from_tick=0 reasons=\["DECODE_COMPLETION", "REQUEST_COMPLETE"\]' "${log}"; then
  echo "[fixture] FAIL (c): decode 提交 delivery(seq=2)缺失或内容不符" >&2
  exit 1
fi
# 恰好 3 次 delivery(无额外 epoch、无事件丢失)。
# L21 同族修复(2026-09-29 审查):grep -c 零命中时退出码 1,pipefail 使本
# 赋值非零,set -e 先杀脚本,"got 0" 的 FAIL 诊断永远到不了——|| true 保住。
n_deliveries=$(grep -c "\[fixture\] delivery" "${log}" || true)
if [[ ${n_deliveries} -ne 3 ]]; then
  echo "[fixture] FAIL: expected exactly 3 deliveries, got ${n_deliveries}" >&2
  exit 1
fi
# 没有任何 delivery 同时含 ARRIVAL 与 PREFILL_DRAIN(同一 Python 调用内
# 处理两个 epoch 即重入违约)。
if grep -q 'reasons=\[.*ARRIVAL.*PREFILL_DRAIN' "${log}"; then
  echo "[fixture] FAIL (a): ARRIVAL 与 PREFILL_DRAIN 出现在同一 delivery" >&2
  exit 1
fi

# (d) cpp 侧审计:服务计数 accepted=1 completed=1 active=0
#     (事件丢失由 run-end gate_ok 审计覆盖,cpp 退出码非 0 即失败)。
if ! grep -q '\[online\] service counters: accepted=1 completed=1 active=0 pending_alarm=0' "${run_dir}/cpp.log"; then
  echo "[fixture] FAIL (d): 服务计数不符(期望 accepted=1 completed=1)" >&2
  grep "\[online\] service counters" "${run_dir}/cpp.log" >&2 || true
  exit 1
fi

echo "[fixture] ALL PASS:"
echo "[fixture]   (a) PREFILL_DRAIN 不在 ARRIVAL 的同一 Python 调用内(单 tick 单次 delivery)"
echo "[fixture]   (b) 显式 T+1 下一决策边界唤醒:seq=1 tick=${T_PLUS1} deferred_from_tick=${T_NS}"
echo "[fixture]   (c) milestone 在下一 epoch 交付,decode 提交,completed=1,双进程 exit 0"
echo "[fixture]   (d) 无事件丢失(run-end gate_ok 审计)、无死锁"
