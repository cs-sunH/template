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
RUN_ROOT=${1:-/tmp/wscllm_idle_fixture}
# infra-E3：RUN_ROOT 规范化为绝对路径（与 run_online_strategy.sh 同款）——
# fixture 内部 cd 进 PROJECT/workload 目录，相对路径 run_root 在 cd 后解析错位。
RUN_ROOT=$(realpath -m "${RUN_ROOT}")

# infra-F1 危险路径闸（2026-10-01 补齐，与 run_online_strategy.sh /
# run_bridge_cpp_death_fixture.sh 的 R2-5 同款方向：危险命中→拒绝）——
# 本 fixture 内部对 RUN_ROOT 子目录执行 rm -rf，同样不得接受 /、$HOME、
# 仓内路径；缺省 /tmp 落点与常规自定义路径正常放行。
# infra-F1 顶级系统目录清单（R5 补齐，对齐 run_online_strategy.sh 的
# 完整闸深）：RUN_ROOT 命中系统目录本身即拒绝（其子目录 rm -rf 同样
# 不可接受）。
case ":/bin:/boot:/dev:/etc:/home:/lib:/lib32:/lib64:/media:/mnt:/opt:/proc:/root:/run:/sbin:/srv:/sys:/tmp:/usr:/var:" in
  *:"${RUN_ROOT}":*)
    echo "[fixture] refusing dangerous RUN_ROOT (system dir): ${RUN_ROOT}" >&2
    exit 1
    ;;
esac
if [[ -z "${RUN_ROOT}" || "${RUN_ROOT}" == / || "${RUN_ROOT}" == "${HOME}"       || "${PROJECT}" == "${RUN_ROOT}" || "${PROJECT}" == "${RUN_ROOT}"/*       || "${RUN_ROOT}" == "${PROJECT}"/* ]]; then
  echo "[fixture] refusing dangerous RUN_ROOT: ${RUN_ROOT}" >&2
  exit 1
fi

# ET 基线目录 = GEN_MATCH 动态解析(四仓统一口径)——恰好一个 llama2_7b_wsc_llm_inference_54npus_* 目录(plan_materializer 产出,
# 输入由 sh_test_mesh/workload/llama2_7b_inference/traces/derive_20_first_30_seconds.py 物化,其 stdout 即权威 provenance 记录)。
GEN_MATCH=("${PROJECT}"/sh_test_mesh/generated/llama2_7b_wsc_llm_inference_54npus_*)
if [[ ${#GEN_MATCH[@]} -ne 1 || ! -d "${GEN_MATCH[0]}" ]]; then
  echo "[runner] expected exactly one generated dir under sh_test_mesh/generated (run plan_materializer.py after the traces/ materializer; its stdout is the authoritative provenance record), found: ${GEN_MATCH[*]}" >&2
  exit 1
fi
ET_DIR=${GEN_MATCH[0]}
ET_PREFIX="${ET_DIR}/llama2_7b_wsc_llm_inference"
# RC = runtime_config 目录动态解析（sync-py-E11，2026-09-30 face-LRU 同款
# 移植，原为硬编码目录名；口径与主 runner 一致）：目录名由配置装载器生成
# 为 "<hardware slug>__<capacity_profile>__<remote-memory 标签>"
# （generate_wsc_llm_trace -> config_resolver.materialize_runtime_configs；
# 本仓 hardware json 的 remote-memory.npu-selection=mesh-boundary → 标签恒
# edge_remote_memory_pool）。capacity_profile 权威同源 = trace_config.csv
# 的 config,local_hbm_capacity_profile 行。按该 profile 过滤
# generated/runtime_config/ 并要求恰好一个，缺失/歧义 fail-closed。
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
  echo "[runner] FAIL: cannot read local_hbm_capacity_profile from trace_config.csv (sync-py-E11)" >&2
  exit 1
}
RC_MATCH=("${PROJECT}"/sh_test_mesh/generated/runtime_config/*__"${RC_PROFILE}"__edge_remote_memory_pool)
if [[ ${#RC_MATCH[@]} -ne 1 || ! -f "${RC_MATCH[0]}/system.json" \
      || ! -f "${RC_MATCH[0]}/network.yml" || ! -f "${RC_MATCH[0]}/comm_group.json" \
      || ! -f "${RC_MATCH[0]}/remote_memory.json" ]]; then
  echo "[runner] FAIL: expected exactly one runtime_config dir for capacity_profile '${RC_PROFILE}' carrying system.json/network.yml/comm_group.json/remote_memory.json (re-run plan_materializer.py), found: ${RC_MATCH[*]}" >&2
  exit 1
fi
RC=${RC_MATCH[0]}
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online

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

# uutils date 0.8.0 treats %3N as an untruncated nanosecond field, which makes
# elapsed-millisecond arithmetic overflow or report absurd values.  Use a
# monotonic clock with explicit integer conversion on every supported host.
monotonic_ms() {
  python3 -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

# M20(2026-09-29 审查修复,face-LRU 同款移植):有界等待原语——$1=pid
# $2=timeout_s;0=已退出,1=超时仍在运行。计数式轮询(sleep 0.3，约 3⅓ 次/秒，
# 实际等待上界≈名义值×1.2（方向保守）,与下方 ACTIVE->IDLE 等待同
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
  # M20(2026-09-29 审查修复,face-LRU 同款移植):桥侧 response-poll 看门狗补武装
  # (此前缺省不传=0=永等)——Python 夹具服务单次决策交换停滞时 fail-closed
  # abort 而非无界挂死。120000ms 为宽松有界值(C++ 语义:毫秒,须大于最慢单决策
  # 耗时;本夹具决策为毫秒级,正常路径永不触发,语义不变)。
  # infra-E5(2026-09-30,face-LRU 同款移植,env 名按本仓实况适配):本仓 README
  # 与 run_online_strategy.sh 仅文件化 BRIDGE_TIMEOUT_MS 单一 env 名(face 侧
  # 的 SH_BRIDGE_TIMEOUT_MS 旧别名系其仓内历史遗留,本仓无此沿革,不引入);
  # 未设时缺省 120000,显式 0=恢复永等逃生口。
  "${BIN}" \
    --online-mode strategy \
    --bridge-dir "${run_dir}/bridge" \
    --online-validate 1 \
    --idle-watchdog-s 0 \
    --bridge-timeout-ms "${BRIDGE_TIMEOUT_MS:-120000}" \
    --command-fifo "${run_dir}/cmd.fifo" \
    --workload-configuration="${ET_PREFIX}" \
    --comm-group-configuration="${RC}/comm_group.json" \
    --system-configuration="${RC}/system.json" \
    --remote-memory-configuration="${RC}/remote_memory.json" \
    --network-configuration="${RC}/network.yml" \
    --logging-folder=off \
    > "${run_dir}/cpp.log" 2>&1 &
  CPP_PID=$!
  # FIFO 就绪等待带墙钟预算(infra-F2,face-LRU 同款移植,SH_FIFO_WAIT_S 可覆盖,
  # 缺省 60s;同 run_online_strategy.sh 注)。infra-E10:截止时钟用本文件既有的
  # monotonic_ms 原语(墙钟跳变不漂移,见上方 uutils date 语义裁定)。
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
  # M20(2026-09-29 审查修复,face-LRU 同款移植):裸 wait 无界——CloseInput
  # 丢失/夹具服务回归时本 fixture 挂死而非失败。改为有界轮询(SH_EXIT_WAIT_S,
  # 缺省 120s)后再 wait 收割退出码;超时 fail-closed(非零退出触发 EXIT trap
  # 清场)。正常路径进程自然退出,轮询立即返回,退出码语义不变。
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
  # L21(2026-09-29 审查修复,face-LRU 同款移植):grep 计数为 0 时 pipefail
  # 使本赋值返回非零,set -e 会先杀掉脚本,下方 FAIL 诊断永远到不了——
  # || true 保住 FAIL 分支(零命中时 states 为空串,比较失败走诊断出口)。
  states=$(lifecycle_states "${run_dir}/cpp.log" || true)
  if [[ "${states}" != "IDLE" ]]; then
    echo "[fixture] FAIL: after 2s hold expected state sequence 'IDLE', got '${states}'" >&2
    exit 1
  fi
  echo "[fixture] scenario 1: IDLE held $((t_after - t_before))ms (processes alive, state=IDLE, no preset queue)"

  exec 3>"${run_dir}/cmd.fifo"
  echo '{"kind":"CloseInput"}' >&3
  exec 3>&-
  wait_for_exits "${run_dir}" scenario1

  # L21(2026-09-29 审查修复,face-LRU 同款移植):同上——零命中时 pipefail 使
  # 本赋值非零,set -e 先杀脚本,下方 FAIL 诊断永远到不了——|| true 保住。
  states=$(lifecycle_states "${run_dir}/cpp.log" || true)
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
  # L21(2026-09-29 审查修复,face-LRU 同款移植):同上,零命中时保住下方 FAIL
  # 诊断分支。
  states=$(lifecycle_states "${run_dir}/cpp.log" || true)
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

  # L21(2026-09-29 审查修复,face-LRU 同款移植):同上——零命中时保住下方
  # FAIL 诊断分支。
  states=$(lifecycle_states "${run_dir}/cpp.log" || true)
  echo "[fixture] scenario 2 transitions: ${states}"
  if [[ "${states}" != "IDLE ACTIVE IDLE DRAINING FINISHED" ]]; then
    echo "[fixture] FAIL: scenario 2 expected 'IDLE ACTIVE IDLE DRAINING FINISHED'" >&2
    exit 1
  fi
  # 指定世界 tick:python.log 必须记录两个 request 的 arrival_world_ns 均恰好
  # 为 T1_NS。同 tick 到达在单次 delivery 内合批(seq=0 一行含两个 arrival),
  # 所以按"出现次数"数(每 arrival 一次),不按行数。
  local arrival_hits
  # R2-5(2026-09-30):|| true 护栏——pipefail 下 grep 0 命中会使整条管道
  # 非零、set -e 直接退出,丢失下方 FAIL 定制消息(退出码仍非 0,但 0
  # 命中恰是本断言要报的场景)。
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
