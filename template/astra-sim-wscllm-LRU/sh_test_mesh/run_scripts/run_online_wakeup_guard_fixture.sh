#!/usr/bin/env bash
# run_online_wakeup_guard_fixture.sh -- 缺陷 C 回归 fixture(2026-08-16,
# face主动测试错误分析.md 缺陷 C:空队列分支唤醒条件;自 face 0049ef5 同步
# 移植,ET/RUN_ROOT 适配本仓 generated 目录)。
#
# 场景 1 f7-blueprint(F7 末态蓝图健康形态):
#   注入 r0 @T -> ARRIVAL 批携带 [同步完成控制节点(同 tick 里程碑)+
#   未到期 future alarm(r1 @T+1000)] -> 里程碑与 r1 ARRIVAL 在 T+1000
#   同一 epoch 交付 -> r0/r1 全部完成 -> CloseInput -> 双进程 exit 0。
#   断言:completed=2/accepted=1(future-alarm 到达不计 accepted)、
#   future_alarm 调度留痕、无死锁(cpp 服务计数审计)。
#   ——证明修复后的空队列分支在健康蓝图形态上不误触发 fail-closed。
#
# 场景 2 defer-dead-end(唤醒真空死端):
#   注入 r0 @T -> 空批 defer(active=1、队列空、mailbox 空)-> 注入
#   CloseInput ->
#     EXPECT_MODE=fixed(缺省):C++ 立即 fail-closed abort,cpp.log 含
#       "lost-wakeup dead end" 且 active=1 计数入消息(替代 50 分钟静默
#       挂死/busy-spin;本仓蓝本缺陷:wait_for_work 谓词含 !input_open_,
#       input 关而未 finished 时立即返回 -> 主循环忙转,sh_2.0 发现);
#     EXPECT_MODE=legacy-stall:预修复二进制不退出(RED 基线采集用;
#       runner 断言 10s 无退出后记录现场(双方 /proc 状态入 run_dir),
#       清理按 SIGTERM 纪律执行(先 TERM 留优雅终止痕迹,-9 兜底)。
#
# 用法:
#   bash run_online_wakeup_guard_fixture.sh [run_root]
# 环境变量:
#   BIN           在线二进制(缺省本仓 build 产物)
#   EXPECT_MODE   fixed | legacy-stall(缺省 fixed)
# 注:裸仓库态需先运行 sh_test_mesh/workload/llama2_7b_inference/traces/derive_20_first_30_seconds.py 物化 20.csv 前 30s 输入,
#   再由 plan_materializer.py 生成 ET 目录;fixture 经 GEN_MATCH 通配动态解析
#   唯一 ET 目录(与 run_online_strategy.sh 同款),不引用固定 ET 前缀。
# (2026-10-01,face-LRU 同款移植) -e 补齐:与另两个 fixture 同级防护——
#   exec 3> FIFO 打开失败等未防护命令此前被静默跳过,依赖后续 grep 断言
#   间接兜底。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
RUN_ROOT=${1:-/tmp/wscllm_wakeup_guard}
# infra-E3：RUN_ROOT 规范化为绝对路径——start_online 内部 cd 进 PROJECT/
# workload 目录，相对路径 run_root 在 cd 后会解析到错误位置（bridge FIFO 与
# 顶层 shell 的 cmd.fifo 重定向随之错位）。
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
BIN=${BIN:-${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online}
EXPECT_MODE=${EXPECT_MODE:-fixed}

T_NS=3000000000

CPP_PID=""
PY_PID=""

# 现场取证(2026-09-26,对齐 run_online_strategy.sh 监督纪律:疑似楔死时
# 先保留 bridge 目录盘态与双方 /proc/<pid>/{wchan,syscall,stack} 再清理)。
record_scene() {
  local dir=$1 pid
  mkdir -p "${dir}/forensics" 2>/dev/null || return 0
  for pid in "${CPP_PID}" "${PY_PID}"; do
    [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null || continue
    for f in wchan syscall stack; do
      cp "/proc/${pid}/${f}" "${dir}/forensics/$(basename "${dir}").pid${pid}.${f}" 2>/dev/null || true
    done
  done
}

cleanup() {
  # 终止纪律(与 run_online_strategy.sh 同款):先 SIGTERM 让被监督进程
  # 留下优雅终止痕迹;2s 宽限后仍存活才 -9 兜底(不作为首选)。
  local pid
  for pid in "${CPP_PID}" "${PY_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill "${pid}" 2>/dev/null || true
    fi
  done
  local waited=0
  while [[ ${waited} -lt 20 ]]; do
    local alive=0
    for pid in "${CPP_PID}" "${PY_PID}"; do
      if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
        alive=1
      fi
    done
    [[ ${alive} -eq 1 ]] || break
    sleep 0.1
    waited=$((waited + 1))
  done
  for pid in "${CPP_PID}" "${PY_PID}"; do
    if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
      kill -9 "${pid}" 2>/dev/null || true
    fi
  done
}
trap cleanup EXIT
# 监督纪律:终止本 runner 用 SIGTERM(kill <pid>)——收到即先记录现场再
# 走 EXIT 清理,勿用 SIGINT/Ctrl-C(会冻结健康瞬态造成"楔死"伪影)。
trap 'record_scene "${RUN_ROOT}"; exit 143' TERM

# M20(2026-09-29 审查修复,face-LRU 同款移植):有界等待原语——$1=pid
# $2=timeout_s;0=已退出,1=超时仍在运行。计数式轮询(sleep 0.3，约 3⅓ 次/秒，
# 实际等待上界≈名义值×1.2（方向保守）),不依赖 date/%3N 语义。
wait_pid_bounded() {
  local pid=$1 waited=0 max_waited=$(( $2 * 4 ))
  while kill -0 "${pid}" 2>/dev/null; do
    if (( waited >= max_waited )); then return 1; fi
    sleep 0.3
    waited=$((waited + 1))
  done
  return 0
}

# uutils date 0.8.0 treats %3N as an untruncated nanosecond field, which makes
# elapsed-millisecond arithmetic overflow or report absurd values; wall-clock
# deadlines also jump with system time changes.  Use the same monotonic clock
# as run_online_idle_fixture.sh / run_online_same_tick_milestone.sh.
# (infra-E10,face-LRU 同款移植。)
monotonic_ms() {
  python3 -c 'import time; print(time.monotonic_ns() // 1_000_000)'
}

start_online() {  # $1=run_dir $2=scenario
  local run_dir=$1 scenario=$2
  rm -rf "${run_dir}"
  mkdir -p "${run_dir}/bridge"
  mkfifo "${run_dir}/cmd.fifo"
  cd "${PROJECT}"
  # --idle-watchdog-s 0 (2026-09-05): 看门狗默认已武装为 1s；本 fixture 的契约是刻意
  # IDLE 停车（request-neutral 无输入必须保持 IDLE 不退出），显式 0 恢复无界停车契约。
  # M20(2026-09-29 审查修复,face-LRU 同款移植):桥侧 response-poll 看门狗补武装
  # (此前缺省不传=0=永等)。仅 EXPECT_MODE=fixed(修复后二进制)注入;legacy-stall 采
  # RED 基线用的预修复二进制可能早于 2026-08-22 该旗标引入,保持旧调用
  # 字节不变。120000ms 为宽松有界值(C++ 语义:毫秒;正常夹具决策毫秒级,
  # 永不触发,语义不变)。infra-E5(2026-09-30,face-LRU 同款移植,env 名按本仓
  # 实况适配):本仓 README 与 run_online_strategy.sh 仅文件化
  # BRIDGE_TIMEOUT_MS 单一 env 名(face 侧的 SH_BRIDGE_TIMEOUT_MS 旧别名系
  # 其仓内历史遗留,本仓无此沿革,不引入);未设时缺省 120000,显式 0=恢复永等。
  local bridge_timeout_args=()
  if [[ "${EXPECT_MODE}" != "legacy-stall" ]]; then
    bridge_timeout_args=(--bridge-timeout-ms "${BRIDGE_TIMEOUT_MS:-120000}")
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
  # FIFO 就绪等待带墙钟预算(infra-F2,face-LRU 同款移植,SH_FIFO_WAIT_S 可覆盖,
  # 缺省 60s;同 run_online_strategy.sh 注)。infra-E10:截止时钟用本文件既有的
  # monotonic_ms 原语(墙钟跳变不漂移,见上方 uutils date 语义裁定)。
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

wait_cpp_exit() {  # $1=timeout_s -> 0 exited, 1 still running
  # infra-E10(face-LRU 同款移植):截止时钟由 wall date 改 monotonic_ms——
  # 墙钟跳变不漂移(原语语义见上方 monotonic_ms 注)。
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

# M20(2026-09-29 审查修复,face-LRU 同款移植):场景1 收尾的裸 wait 无界
# (Python 服务或 C++ 挂死时回归被掩盖为挂起)——先有界轮询(SH_EXIT_WAIT_S,
# 缺省 120s),超时 fail-closed(非零退出触发 EXIT trap 清场);正常路径立即
# 通过,退出码仍由 wait 收割,语义不变。
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
# (accepted=1,completed=2)+ 无事件丢失(服务计数审计;no_decision 结束
# 审计行已删)。
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
  record_scene "${RUN_ROOT}"
  echo "[wakeup_guard]   (busy-spin/挂死现场保留于 ${run_dir},/proc 取证见"
  echo "[wakeup_guard]    ${RUN_ROOT}/forensics/;清理走 EXIT trap 的 SIGTERM 纪律)"
  exit 0
fi

# fixed:C++ 应在数秒内 fail-closed abort(lost-wakeup dead end 诊断)。
if ! wait_cpp_exit 15; then
  echo "[wakeup_guard] FAIL(场景2): C++ 15s 未退出(仍是挂死/spin?)" >&2
  tail -12 "${run_dir}/cpp.log" >&2; exit 1
fi
cpp_exit=0; wait "${CPP_PID}" || cpp_exit=$?; CPP_PID=""
# R5-A-S1(2026-10-01 审查修复):此处原为裸 wait——三 fixture 四件套移植时
# 枚举漏网的第三处无界等待(C++ 侧有 wait_cpp_exit 15s 有界前驱,PY 侧裸
# wait)。正常路径 Python 服务在桥 EOF 后自然退出,挂死需 Python 侧回归,
# 恰是 M20 要防的形态。补与场景1同款 wait_pid_bounded 有界前驱(SH_EXIT_WAIT_S,
# 缺省 120s),超时 fail-closed(非零退出触发 EXIT trap 清场);正常路径立即
# 通过,退出码仍由 wait 收割(2>/dev/null 保留:抑制 C++ fail-closed abort
# 时 shell 对 PY 进程组的 job 控制噪声),语义不变。
# 登记分叉:face-LRU 同名脚本同位点仍是裸 wait(本仓单侧加固,与 face 形成
# 分叉,供后续两仓同步参考)。
py_exit=0
if ! wait_pid_bounded "${PY_PID}" "${SH_EXIT_WAIT_S:-120}"; then
  echo "[wakeup_guard] FAIL(场景2): Python ${SH_EXIT_WAIT_S:-120}s 未退出(挂死?)" >&2
  tail -12 "${run_dir}/python.log" >&2; exit 1
fi
wait "${PY_PID}" 2>/dev/null || py_exit=$?; PY_PID=""
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
