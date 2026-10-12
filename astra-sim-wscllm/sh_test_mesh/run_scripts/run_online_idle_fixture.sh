#!/usr/bin/env bash
# wscllm phase-1 step-1-10 IDLE/注入 lifecycle fixture(合同② 目标 5)。
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
RUN_ROOT=${1:-${TMPDIR:-/tmp}/wscllm_idle_fixture}
# 注入通道（cmd.fifo）与 bridge 目录会被 mkfifo/启动端/注入端三方按各自 cwd
# 解析，必须先把 run root 归一为绝对路径，否则传相对 RUN_ROOT 时 C++ 端
# （cd PROJECT 后启动）与注入端（exec 3>）指向不同文件。
RUN_ROOT=$(realpath -m "${RUN_ROOT}")
# P3-3.9 E2-1 同款 rm -rf 护栏(F1-CPP C1-7 2026-09-29 移植,fail-closed):
# RUN_ROOT 由调用方任意传入,realpath -m 归一不改危险语义——传 `.`(cwd=仓根)
# 归一即仓根、传 `..` 归一即仓根祖先,下方 start_online 的 `rm -rf
# "${run_dir}"` 会删掉整个仓库或其上层目录。显式拒绝:空串、`/`、仓根本身、
# 仓根的祖先目录、仓根内部路径(fixture 合同 run root 一律仓外,正式纪律
# 一律 /tmp);合法 run root 不受影响。
case "${RUN_ROOT}" in
  ""|"/")
    echo "[fixture] FAIL: refusing dangerous RUN_ROOT='${RUN_ROOT}' (empty or filesystem root)" >&2
    exit 1 ;;
esac
if [[ "${RUN_ROOT}" == "${PROJECT}" || "${RUN_ROOT}" == "${PROJECT}"/* || "${PROJECT}" == "${RUN_ROOT}"/* ]]; then
  echo "[fixture] FAIL: refusing RUN_ROOT='${RUN_ROOT}' (repo root, its ancestor, or inside the repo; use a dedicated dir outside the repo, e.g. /tmp/...)" >&2
  exit 1
fi

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
# 目录名，照 face-LRU §16d 形态移植，与官方 runner 同款）：目录名由物化器
# 生成为 "<hardware slug>__<capacity_profile>__<remote-memory 标签>"
# （generate_wsc_llm_trace -> config_resolver.materialize_runtime_configs；
# 本仓标签恒为 no_memory_expansion）。capacity_profile 的权威同源 =
# trace_config.csv 的 config,local_hbm_capacity_profile 行。按该 profile
# 过滤 generated/runtime_config/ 并要求恰好一个（与上方 GEN_MATCH 的
# exactly-one 口径一致），缺失/歧义 fail-closed——此前 fixture 无 RC 闸，
# 目录缺失要等 C++ 打开 system.json 才炸；现网唯一合法档位下动态解析与
# 旧硬编码命中同一目录，健康路径零行为变化。
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
  echo "[fixture] FAIL: cannot read local_hbm_capacity_profile from trace_config.csv" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/face_case5_config_c__*"${RC_PROFILE}"__no_memory_expansion)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -d "${RC_MATCH[0]}" \
      || ! -f "${RC_MATCH[0]}/system.json" || ! -f "${RC_MATCH[0]}/network.yml" \
      || ! -f "${RC_MATCH[0]}/comm_group.json" ]]; then
  echo "[fixture] expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json (materialized under sh_test_mesh/generated/runtime_config/ by plan_materializer.py; bare-repo restore deletes generated/), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
# BIN 显式存在性闸（crosscut-F3，2026-10-01，与 RC/GEN_MATCH 闸及
# wakeup_guard fixture 同款）：缺失即在此报错，不再走"C++ 先于 FIFO
# 就绪退出"的晚诊断路径。
if [[ ! -x "${BIN}" ]]; then
  echo "[idle_fixture] bin missing: ${BIN} (build first)" >&2
  exit 1
fi

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
# P3-3.9 E2-2 同款信号转发(F1-CPP C1-7 2026-09-29 移植):裸 EXIT trap 在
# bash 被致命信号直接杀死时不执行;监督纪律以 SIGTERM 终止长跑,必须显式
# 转成 exit 才能进入上方 cleanup——否则 C++ 成孤儿,且本 fixture 的 C++
# 带 --idle-watchdog-s 0(契约性无界停车),孤儿在 wait_for_work 永久停泊
# 并持有 bridge 目录。INT/HUP 同理。
trap 'exit 143' TERM
trap 'exit 130' INT
trap 'exit 129' HUP

# uutils date 0.8.0 treats %3N as an untruncated nanosecond field, which makes
# elapsed-millisecond arithmetic overflow or report absurd values.  Use a
# monotonic clock with explicit integer conversion on every supported host.
monotonic_ms() {
  python3 -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

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
  # FIFO 就绪等待带墙钟预算（crosscut-F2，2026-10-01，照 wscllm-LRU/face-LRU
  # 同款）：kill -0 只防 C++ 进程退出，防不了"进程存活但迟迟不建 FIFO"的
  # 窗口；超时打印 cpp.log 尾部后 fail-closed（SH_FIFO_WAIT_S 可覆盖，
  # 缺省 60s）。
  local fifo_deadline=$(( $(date +%s) + ${SH_FIFO_WAIT_S:-60} ))
  until [ -p "${run_dir}/bridge/req_notify.fifo" ] && [ -p "${run_dir}/bridge/resp_notify.fifo" ]; do
    if ! kill -0 "${CPP_PID}" 2>/dev/null; then
      echo "[fixture] C++ exited before FIFO ready (see ${run_dir}/cpp.log)" >&2
      tail -5 "${run_dir}/cpp.log" >&2
      return 1
    fi
    if (( $(date +%s) > fifo_deadline )); then
      echo "[fixture] FIFOs not ready within ${SH_FIFO_WAIT_S:-60}s (see ${run_dir}/cpp.log tail)" >&2
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
  t_before=$(monotonic_ms)
  sleep 2
  t_after=$(monotonic_ms)
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
  inject_ms=$(monotonic_ms)
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
  complete_ms=$(monotonic_ms)
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
  # || true:grep 零命中时管道按 pipefail 返回 1,set -e 会直接退脚本、
  # FAIL 提示永远打不出;靠下方 -ne 2 计数断言报错。
  arrival_hits=$(grep -o '"arrival_world_ns": '"${T1_NS}" "${run_dir}/python.log" | wc -l || true)
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
