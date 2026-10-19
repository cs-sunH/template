#!/usr/bin/env bash
# face phase-1 step-1-11(CORE)same-tick milestone fixture(方案 step 1-11
# / 仿真加速分析.md §4.3 最高优先级专门 fixture)。
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
#   (d) 无事件丢失(gate counters 行 delivery_count>=1;2026-09-30 起口径)、
#       无死锁(轮询有界),request 最终完成。
#
# Usage: bash run_online_same_tick_milestone.sh [run_root]
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
RUN_ROOT=${1:-/tmp/face_same_tick_milestone}
# 相对 run_root 锚定调用者 cwd 预先解析为绝对路径（与 run_online_strategy.sh
# 同款，2026-09-29）：start_online 会 cd 到仓根/工作负载目录，放任相对形式
# 会让 rm/mkdir 与日志重定向的相对基准随 cd 漂移——中途崩溃且产物散落。
case "${RUN_ROOT}" in /*) ;; *) RUN_ROOT="$(pwd -P)/${RUN_ROOT}" ;; esac

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[fixture] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
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
  echo "[fixture] FAIL: cannot read hardware.capacity_profile from ${ET_DIR}/manifest.json (stale plan dir? re-run plan_materializer.py)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__no_memory_expansion)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" ]]; then
  echo "[fixture] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
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
# monotonic clock with explicit integer conversion on every supported host.
monotonic_ms() {
  python3 -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

start_online() {  # $1=run_dir; sets CPP_PID / PY_PID
  local run_dir=$1
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
      echo "[fixture] C++ exited before FIFO ready (see ${run_dir}/cpp.log)" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    if [[ ${fifo_waited} -ge 400 ]]; then
      echo "[fixture] FAIL: ${run_dir}/bridge/{req_notify,resp_notify}.fifo not ready after $((fifo_waited / 10 * 3))s (${fifo_waited}x0.3s; cpp alive but startup hung; see cpp.log)" >&2
      kill "${CPP_PID}" 2>/dev/null || true
      return 1
    fi
    sleep 0.3
    fifo_waited=$((fifo_waited + 1))
  done
  cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference"
  python3 -u online/verify/same_tick_milestone_fixture_service.py \
    --bridge-dir "${run_dir}/bridge" \
    > "${run_dir}/python.log" 2>&1 &
  PY_PID=$!
  sleep 0.5
}

wait_for_exits() {  # $1=run_dir $2=name
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
    echo "[fixture] FAIL: timeout waiting for 'ACTIVE -> IDLE' " \
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
n_deliveries=$(grep -c "\[fixture\] delivery" "${log}")
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

# (d) cpp 侧审计:服务计数 accepted=1 completed=1 active=0;
#     无事件丢失(2026-09-30:恒 0 的 no_decision_python_callback_count 已随
#     死代码清理删除,等价证据改为 gate counters 行的 delivery_count)。
if ! grep -q '\[online\] service counters: accepted=1 completed=1 active=0 pending_alarm=0' "${run_dir}/cpp.log"; then
  echo "[fixture] FAIL (d): 服务计数不符(期望 accepted=1 completed=1)" >&2
  grep "\[online\] service counters" "${run_dir}/cpp.log" >&2 || true
  exit 1
fi
if ! grep '\[online\] gate counters: ' "${run_dir}/cpp.log" | grep -q 'delivery_count=[1-9]'; then
  echo "[fixture] FAIL (d): delivery_count 为 0(事件丢失)" >&2
  exit 1
fi

echo "[fixture] ALL PASS:"
echo "[fixture]   (a) PREFILL_DRAIN 不在 ARRIVAL 的同一 Python 调用内(单 tick 单次 delivery)"
echo "[fixture]   (b) 显式 T+1 下一决策边界唤醒:seq=1 tick=${T_PLUS1} deferred_from_tick=${T_NS}"
echo "[fixture]   (c) milestone 在下一 epoch 交付,decode 提交,completed=1,双进程 exit 0"
echo "[fixture]   (d) 无事件丢失(gate counters delivery_count>=1)、无死锁"
