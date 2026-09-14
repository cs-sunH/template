#!/usr/bin/env bash
# run_slo_postprocess.sh —— 仿真后自动 SLO 指标提取（P3，2026-08-28）。
#
# Usage: bash run_slo_postprocess.sh <run_dir>
#
# 职责：对单个 run_dir 依次调用 slo_tools 的单 run 自包含子命令，产物落
# run_dir 根（各工具缺省输出名）；runner 在 postprocess 成功后、
# archive_run_outputs.sh 之前挂接本脚本（此时 cpp.log 未压缩、per-request
# manifest 已拷入 run_dir、results/ jsonl 已归位，输入全齐）。也可对归档后
# 的 run_dir 幂等重跑（slo_common 走 cpp.log → metrics.log → cpp.log.gz
# 统一回退）。
#
# granularity 纪律：本层只产最细粒度（逐请求/逐事件），一律不传分桶/聚合
# 参数（backlog 不传 --bucket-ns）；任何粗化（分桶、p90-负载曲线换粒度）
# 留给下游画图脚本，保证后续任意再加工不需要重跑仿真。
#
# 自动提取清单（输入全部 run_dir 本地可得；A4/2026-08-29 起由
# slo_tools/slo_postprocess_driver.py 单进程单遍执行——9 步的产物集/
# 行序/公式/slo_postprocess.log 与逐工具串行逐字节一致，读放大收敛：
# request_metrics.csv 4 读→1、decision log 4 流→1、[METRIC] init 探测
# 4→1、解释器启动 9→1；工具 CLI 保持可独立调用）：
#   1. slo_stats.py e2e-stats --extra-pct 90,95  → slo_e2e_stats.csv
#   2. slo_stats.py backlog（不传 --bucket-ns）  → slo_backlog.csv（逐事件）
#   3. slo_stats.py session                      → slo_session.csv
#   4. slo_stats.py warmup                       → slo_warmup.json
#   5. kv_cache_adapter.py                       → cache_events.csv +
#                                                 kv_hit_states.csv
#   6. load_imbalance.py                         → slo_load_imbalance.csv
#   7. restore_decomposition.py（仅 full 档）    → slo_restore_decomposition.csv
#   8. hopbytes.py                               → slo_hopbytes_total.csv +
#                                                 slo_hopbytes_per_request.csv
#   9. hbm_watermark.py                          → slo_hbm_intervals.csv
#                                                 (权威 RLE 变点区间)
#                                                 + slo_hbm_plot_series.csv
#                                                 (行预算约束绘图产物；旧
#                                                 slo_hbm_watermark_series.
#                                                 csv 已退役)
#                                                 + slo_hbm_watermark_
#                                                 instances.csv
#
# hbm_watermark 四层可信度（P1，2026-08-30）：run_dir 含 results/
# kv_delta_journal.jsonl 时走 journal 权威重放（含 checksum 证书时
# per_rank_total_hbm_certified——正式逐 rank 容量判决，违规 exit 3；
# 缺证书降级 resident_kv_exact/lifecycle_replay_exact 仅报告）；
# 缺 journal 的旧 run 为 upper_bound_only 上界口径（超限只诊断、exit 0）。
#
# 不纳入自动提取（输入跨 run_dir 或依赖外部表，留 campaign 阶段）：
#   slo_stats.py violation / bucket-stats（要 T_isolated 表）、
#   slo_stats.py normalized / scan-export（多 run_dir）。
#
# 失败语义（文档 P；A4 后由 driver 逐复刻）：默认 warn——某步失败打印
# 告警并写 ${RUN_DIR}/slo_postprocess.FAIL 标记（含失败步骤与退出码），
# 不推翻仿真结果（runner 退出码不因 warn 改变）；SH_SLO_POSTPROCESS=strict
# 时失败即 runner 非零退出（由调用方 runner 实现）。本脚本自身退出码：
# 全过=0，任一步失败=1。
#
# 内存护栏（2026-08-30 爆内存根治 §3.5/§3.6）：driver 进程包 cgroup v2
# 内存上限（默认 16G，SH_SLO_MEM_LIMIT 覆盖，值直写 MemoryMax/memory.max，
# 如 "16G"/"8G"/字节数），memory.swap.max=0 成对设置（本机有 swap：只限
# max 会退化成换页而不杀进程）。语义：超限只 OOM-kill 本步（driver 非
# 零退出 → 走既有 FAIL 标记+warn 路径，不拖垮整机/不推翻仿真结果）。
# 主路径 systemd-run --user --scope（无 polkit、任意 shell 位置可用、
# scope 退出自动析构）；用户管理器不可达时降级为手工委派 cgroup
# （user@<uid>.service 下建 slo-mem-guard.<pid>，需调用 shell 在用户
# slice 内）；再不可达则告警后裸跑（降级不阻断）。监控补盲（§3.6）：
# driver 存活期间每 60s 从本脚本 stderr 输出一行 [slo-sample]（driver
# RSS / 整机 MemAvailable / 护栏组内存与 oom_kill 计数）——事故当年的
# 84 分钟内存爬升全程无进程级采样，本行补盲。护栏输出一律走 stderr：
# slo_postprocess.log 处于 test_driver_parity 的旧链/新链字节对拍契约
# 内，run_dir 内也不得新增文件。
#
# 非 full 档预期行为：request_metrics.csv 不存在（summary/off 档不产逐请求
# 文件）时，依赖它的 1-4 与 7 跳过并在 slo_postprocess.log 写说明（属正常
# 设计，不算失败）；metrics=off 无任何 [METRIC] 行时整步跳过。train_ledger
# 缺失时跳过 load_imbalance，同样写说明、不算失败。
set -uo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
SLO_TOOLS=$(cd -- "${SCRIPT_DIR}/../slo_tools" && pwd -P)

RUN_DIR=${1:?"usage: run_slo_postprocess.sh <run_dir>"}
if [ ! -d "${RUN_DIR}" ]; then
  echo "[slo-postprocess] FAIL: run dir not found: ${RUN_DIR}" >&2
  exit 1
fi

FAIL_MARKER=${RUN_DIR}/slo_postprocess.FAIL
LOG=${RUN_DIR}/slo_postprocess.log
rm -f "${FAIL_MARKER}"
: > "${LOG}"
SLO_FAIL=0

log() { echo "[slo-postprocess] $*" >> "${LOG}"; }
warn() { echo "[slo-postprocess] $*" | tee -a "${LOG}" >&2; }

# run_step（A4 前 9 子命令的执行壳）已由 slo_postprocess_driver.py 在
# 进程内逐复刻（"run:/ok:/FAIL:" 行、.FAIL 条目、失败不拦后续步）。
# 本层对 driver 退出码的处置：
#   0 = 全部步骤成功；1 = 有步骤失败（driver 已按步写 FAIL 行与 .FAIL
#       条目，此处只置 SLO_FAIL，不重复记条目）；其余 = driver 裸崩
#       （argparse/解释器级），按原 run_step 语义兜底记一条。

# [METRIC] 行来源探测（与 slo_common.resolve_cpp_metric_log 同序）。
if [ -f "${RUN_DIR}/cpp.log" ]; then
  METRIC_SRC="${RUN_DIR}/cpp.log"
elif [ -f "${RUN_DIR}/metrics.log" ]; then
  METRIC_SRC="${RUN_DIR}/metrics.log"
elif [ -f "${RUN_DIR}/cpp.log.gz" ]; then
  METRIC_SRC="${RUN_DIR}/cpp.log.gz"
else
  warn "FAIL: no cpp.log/metrics.log/cpp.log.gz under ${RUN_DIR}"
  printf '%s (exit=%s)\n' "input-resolve: no cpp.log/metrics.log/cpp.log.gz" 1 >> "${FAIL_MARKER}"
  exit 1
fi

# metrics=off：无任何 [METRIC] 行 → 整步跳过（正常设计，不算失败）。
if [ "${METRIC_SRC##*.}" = "gz" ]; then
  METRIC_LINES=$(zgrep -c '^\[METRIC\] ' "${METRIC_SRC}" || true)
else
  METRIC_LINES=$(grep -c '^\[METRIC\] ' "${METRIC_SRC}" || true)
fi
if [ "${METRIC_LINES:-0}" -eq 0 ]; then
  warn "no [METRIC] lines (metrics detail=off?) — SLO extraction skipped (by design)"
  exit 0
fi
log "metric source: ${METRIC_SRC} (${METRIC_LINES} [METRIC] lines)"

# 非 full 档：request_metrics.csv 不存在 → 依赖它的子命令跳过并写说明。
RM=${RUN_DIR}/request_metrics.csv
if [ -f "${RM}" ]; then
  HAVE_RM=1
else
  HAVE_RM=0
  warn "request_metrics.csv absent (metrics detail=summary/off) — e2e-stats/backlog/session/warmup/restore_decomposition skipped (by design)"
fi

# 输入探测（任务3 2026-08-28 口径，A.5/2026-09-05 随变体清除改写）：
# load_imbalance 需要 results/train_ledger.jsonl（列车台账），缺失 → 跳过。
TRAIN_LEDGER=${RUN_DIR}/results/train_ledger.jsonl
if [ -f "${TRAIN_LEDGER}" ]; then
  HAVE_TL=1
else
  HAVE_TL=0
  warn "results/train_ledger.jsonl absent — load_imbalance.py skipped (by design)"
fi
# A.5/2026-09-05：legacy 与 relevant 两个历史变体已清除，session-KV
# 水位重放对全部产物适用；SKIP_HBM 仅由显式 env（SH_SLO_SKIP_HBM）置位。
SKIP_HBM=${SH_SLO_SKIP_HBM:-0}

# A4（2026-08-29）：9 个子命令收敛为单遍 driver。门控探测结果经 env
# 传入（driver 裸调用时自探测并复刻同款说明行，双保险）；driver 的
# stdout+stderr 全量进 LOG——其内部逐复刻每步的 "run:/ok:" 行与工具
# stderr，块序=原链序，故不额外包 run_step（避免多一对 run/ok 行）。
#
# 内存护栏（2026-08-30 爆内存根治 §3.5）：driver 后台启动 + wait 取
# 退出码（FAIL 判定逻辑原样保留；被护栏 OOM-kill 时 rc=137 同样走
# 非零退出的 FAIL 路径）。两条护栏路径按可达性择优：
#   systemd —— systemd-run --user --scope -p MemoryMax -p MemorySwapMax=0
#              （主路径：无需 polkit，任意 shell 位置可用——含 init.scope
#              直启 shell；scope 随进程退出自动析构，无残留清理负担）；
#   cgroup  —— user@<uid>.service 委派目录下手工建 slo-mem-guard.<pid>
#              子 cgroup：memory.max + memory.swap.max=0 + oom.group=1，
#              driver 子 shell 自迁移（$BASHPID）后 exec。init.scope 下
#              cgroup.procs 写入被 nsdelegate 拒绝（EACCES），探测失败
#              自动降级；两条路径都必须 swap.max=0 成对（否则超限只换页
#              不死）。
# 护栏相关输出（glog/[slo-sample]）一律走本脚本 stderr，不写
# slo_postprocess.log——该文件处于 test_driver_parity 的旧链/新链字节
# 对拍契约内，且 run_dir 内不得新增任何文件。
# GUARD_ERR_FD：固化本脚本原始 stderr。glog 会在 run_driver_guarded 的
# ``>> LOG 2>&1`` 重定向子 shell 内被调用，直接 >&2 会污染 LOG，必须
# 走保存的 fd。
exec {GUARD_ERR_FD}>&2
glog() { echo "[slo-postprocess] $*" >&"${GUARD_ERR_FD}"; }
GUARD_MEM_LIMIT="${SH_SLO_MEM_LIMIT:-16G}"
GUARD_BASE="/sys/fs/cgroup/user.slice/user-$(id -u).slice/user-$(id -u).service"
GUARD_CG="${GUARD_BASE}/slo-mem-guard.$$"
GUARD_MODE=none
GUARD_TELE_PATH=""   # 护栏遥测（memory.peak/memory.events）读取的组路径

# systemd-run --user 需要连上 user bus；init.scope 直启 shell 常缺
# XDG_RUNTIME_DIR，按 uid 推导（socket 在场才设）。
if [ -z "${XDG_RUNTIME_DIR:-}" ] && [ -S "/run/user/$(id -u)/bus" ]; then
  XDG_RUNTIME_DIR="/run/user/$(id -u)"
  export XDG_RUNTIME_DIR
fi

if command -v systemd-run >/dev/null 2>&1 \
   && systemd-run --user --scope --collect -p MemoryMax="${GUARD_MEM_LIMIT}" \
        -p MemorySwapMax=0 -- true >/dev/null 2>&1; then
  GUARD_MODE=systemd
  glog "memory guardrail: systemd-run --user --scope (MemoryMax=${GUARD_MEM_LIMIT}, MemorySwapMax=0)"
elif [ -d "${GUARD_BASE}" ] && mkdir -p "${GUARD_CG}" 2>/dev/null \
   && printf '%s\n' "${GUARD_MEM_LIMIT}" > "${GUARD_CG}/memory.max" 2>/dev/null \
   && echo 0 > "${GUARD_CG}/memory.swap.max" 2>/dev/null; then
  echo 1 > "${GUARD_CG}/memory.oom.group" 2>/dev/null  # 失败容忍
  # 探测本 shell 能否自迁移并自证落位（init.scope 下 EACCES → 本路径
  # 不可用，清场降级）。写入成功 ≠ 已落位：必须回读 /proc/<pid>/cgroup
  # 确认确实在 slo-mem-guard 组内，不得"以为在护栏里"。
  if ( echo "${BASHPID}" > "${GUARD_CG}/cgroup.procs" 2>/dev/null \
       && grep -q slo-mem-guard "/proc/${BASHPID}/cgroup" 2>/dev/null )
  then
    GUARD_MODE=cgroup
    glog "memory guardrail: cgroup ${GUARD_CG} (memory.max=${GUARD_MEM_LIMIT}, swap.max=0, oom.group=1)"
  else
    rmdir "${GUARD_CG}" 2>/dev/null
    GUARD_CG=""
  fi
fi
if [ "${GUARD_MODE}" = none ]; then
  glog "WARN: cgroup guardrail unavailable — running WITHOUT memory cap"
fi

# driver 启动（三模式）。三种模式都保证 $! 即真 driver 进程本身：
#   systemd —— systemd-run --scope 对命令做 exec（无常驻包装层），
#              直接后台 systemd-run，$! = driver；
#   cgroup  —— 子 shell 自迁移（二次自证）后 exec env → python3，同 pid；
#   none    —— 子 shell exec env → python3，同 pid。
# （不依赖 /proc/<pid>/children：本 WSL2 内核无该文件；也不给 systemd-run
#  加函数包装层——那会让 $! 指向包装 shell，遥测解析退化为猜。）
if [ "${GUARD_MODE}" = systemd ]; then
  # --collect 不加（遥测优先）：失败单元立即卸载会销毁 cgroup，
  # 收尾读 memory.peak/oom_kill 的取证无从谈起；滞留的 failed 单元
  # 由收尾 systemctl --user reset-failed 卫生清理。
  systemd-run --user --scope -q \
    -p MemoryMax="${GUARD_MEM_LIMIT}" -p MemorySwapMax=0 -- \
    env SH_SLO_HAVE_RM=${HAVE_RM} SH_SLO_HAVE_TL=${HAVE_TL} \
      SH_SLO_SKIP_HBM=${SKIP_HBM} \
      python3 "${SLO_TOOLS}/slo_postprocess_driver.py" "${RUN_DIR}" \
    >> "${LOG}" 2>&1 &
  DRIVER_PID=$!
else
  run_driver_guarded_nonsystemd() {
    if [ "${GUARD_MODE}" = cgroup ]; then
      # 启动时二次自证：迁移写失败或未落位 → 记 WARN（走脚本 stderr）后
      # 裸跑兜底，绝不"以为在护栏里"。
      if echo "${BASHPID}" > "${GUARD_CG}/cgroup.procs" 2>/dev/null \
         && grep -q slo-mem-guard "/proc/${BASHPID}/cgroup" 2>/dev/null
      then
        exec env SH_SLO_HAVE_RM=${HAVE_RM} SH_SLO_HAVE_TL=${HAVE_TL} \
          SH_SLO_SKIP_HBM=${SKIP_HBM} \
          python3 "${SLO_TOOLS}/slo_postprocess_driver.py" "${RUN_DIR}"
      else
        glog "WARN: driver cgroup migration failed at start — running WITHOUT memory cap"
        exec env SH_SLO_HAVE_RM=${HAVE_RM} SH_SLO_HAVE_TL=${HAVE_TL} \
          SH_SLO_SKIP_HBM=${SKIP_HBM} \
          python3 "${SLO_TOOLS}/slo_postprocess_driver.py" "${RUN_DIR}"
      fi
    else
      exec env SH_SLO_HAVE_RM=${HAVE_RM} SH_SLO_HAVE_TL=${HAVE_TL} \
        SH_SLO_SKIP_HBM=${SKIP_HBM} \
        python3 "${SLO_TOOLS}/slo_postprocess_driver.py" "${RUN_DIR}"
    fi
  }
  run_driver_guarded_nonsystemd >> "${LOG}" 2>&1 &
  DRIVER_PID=$!
fi
# 生效层级记录（run 行）：probe 行见上方 guardrail 探测输出。
glog "memory guardrail engaged: mode=${GUARD_MODE} for driver pid=${DRIVER_PID} (MemoryMax=${GUARD_MEM_LIMIT}, MemorySwapMax=0)"

# 启动后校验护栏真实生效 + 解析遥测路径。DRIVER_PID 即 driver 进程，
# 其 cgroup 就是护栏组：systemd 模式下 scope 迁移是异步的（实测 ~0.5s
# 内从调用方组迁入 run-*.scope），带重试轮询其 /proc/<pid>/cgroup。
# 模式只认 run-*.scope（systemd-run 瞬态单元名）与 slo-mem-guard：裸
# *.scope 会把 /init.scope 误当护栏组（init.scope 也以 .scope 结尾），
# 遥测读成整机 init.scope、oom.group 误写报 Permission denied——V5
# 首跑实测踩中；/proc children 兜底已删（本内核无该文件，V5 二跑踩中）。
resolve_guard_path() {
  local rel i
  for i in 1 2 3 4 5 6 7 8 9 10; do
    rel=$(grep '^0::' "/proc/${1}/cgroup" 2>/dev/null | head -1 | cut -d: -f3)
    case "${rel}" in
      *slo-mem-guard*|*run-*.scope*)
        echo "/sys/fs/cgroup${rel}" ; return 0 ;;
    esac
    kill -0 "${1}" 2>/dev/null || break
    sleep 0.3
  done
  return 1
}
if [ "${GUARD_MODE}" = cgroup ]; then
  GUARD_TELE_PATH="${GUARD_CG}"
elif [ "${GUARD_MODE}" = systemd ]; then
  GUARD_TELE_PATH=$(resolve_guard_path "${DRIVER_PID}")
  # scope 组文件为 root 属主，oom.group 通常写不进（best-effort，静默）
  if [ -n "${GUARD_TELE_PATH}" ]; then
    { echo 1 > "${GUARD_TELE_PATH}/memory.oom.group"; } 2>/dev/null || true
  fi
  [ -n "${GUARD_TELE_PATH}" ] \
    && glog "memory guardrail: driver cgroup ${GUARD_TELE_PATH}"
fi
if [ "${GUARD_MODE}" != none ] && [ -z "${GUARD_TELE_PATH}" ]; then
  glog "WARN: memory guardrail engaged=${GUARD_MODE} but driver cgroup NOT resolved — guardrail unverified"
fi

# 监控补盲（§3.6）：driver 存活期间每 60s 采一行（进程消失时字段 NA）。
# 输出走本脚本 stderr（见上方 glog 说明），不进 slo_postprocess.log。
# rpid 即 pid 本身：三模式启动结构都保证 DRIVER_PID 就是真 driver 进程
# （systemd-run exec / 子 shell exec env），无需 children 查找。
slo_sample_loop() {
  local pid="$1" rpid rss avail cur peak ok
  while kill -0 "${pid}" 2>/dev/null; do
    rpid="${pid}"
    rss=$(awk '/^VmRSS:/{print $2}' "/proc/${rpid}/status" 2>/dev/null)
    avail=$(awk '/^MemAvailable:/{print $2}' /proc/meminfo 2>/dev/null)
    cur=NA; peak=NA; ok=NA
    if [ -n "${GUARD_TELE_PATH}" ] \
       && [ -r "${GUARD_TELE_PATH}/memory.current" ]; then
      cur=$(awk '{print $1}' "${GUARD_TELE_PATH}/memory.current" 2>/dev/null)
      peak=$(awk '{print $1}' "${GUARD_TELE_PATH}/memory.peak" 2>/dev/null)
      ok=$(awk '/^oom_kill /{print $2}' \
           "${GUARD_TELE_PATH}/memory.events" 2>/dev/null)
    fi
    echo "[slo-sample] $(date '+%F %T') driver_rss=${rss:-NA}kB mem_available=${avail:-NA}kB guard_cur=${cur}B guard_peak=${peak}B guard_oom_kill=${ok}" >&2
    sleep 60
  done
}
slo_sample_loop "${DRIVER_PID}" &
SAMPLE_PID=$!

wait "${DRIVER_PID}"
DRIVER_RC=$?
# 遥测抢读（必须第一步做）：scope 单元在 driver 退出后被 user manager
# 异步析构（实测 <0.5s），晚一步 cgroup 即消失；手工 cgroup 组由本脚本
# 持有至收尾，无此竞态。抢读失败再依赖末次 [slo-sample] 行。
if [ -n "${GUARD_TELE_PATH}" ] \
   && [ -r "${GUARD_TELE_PATH}/memory.peak" ]; then
  GUARD_PEAK=$(awk '{print $1}' "${GUARD_TELE_PATH}/memory.peak" 2>/dev/null)
  GUARD_OOM=$(awk '/^oom /{print $2}' \
    "${GUARD_TELE_PATH}/memory.events" 2>/dev/null)
  GUARD_OOM_KILL=$(awk '/^oom_kill /{print $2}' \
    "${GUARD_TELE_PATH}/memory.events" 2>/dev/null)
else
  GUARD_PEAK=""
  GUARD_OOM=""
  GUARD_OOM_KILL=""
fi
kill "${SAMPLE_PID}" 2>/dev/null
wait "${SAMPLE_PID}" 2>/dev/null

# 护栏遥测回显（值取自 wait 后的抢读）+ 清理（在 FAIL 判定前，确保
# 每次运行都留证；手工 cgroup 组由本脚本 rmdir，systemd scope 随进程
# 退出自动析构）。
if [ "${GUARD_MODE}" = cgroup ] && [ -n "${GUARD_CG}" ]; then
  glog "guardrail memory.peak=${GUARD_PEAK:-NA} oom=${GUARD_OOM:-NA} oom_kill=${GUARD_OOM_KILL:-NA} (cgroup ${GUARD_CG})"
  if [ "${GUARD_OOM_KILL:-0}" -gt 0 ] 2>/dev/null; then
    glog "WARN: guardrail OOM-killed driver (oom_kill=${GUARD_OOM_KILL}, rc=${DRIVER_RC})"
  fi
  rmdir "${GUARD_CG}" 2>/dev/null \
    || glog "note: rmdir ${GUARD_CG} failed (leftover kept for inspection)"
elif [ "${GUARD_MODE}" = systemd ]; then
  if [ -n "${GUARD_PEAK}" ]; then
    glog "guardrail memory.peak=${GUARD_PEAK} oom=${GUARD_OOM:-NA} oom_kill=${GUARD_OOM_KILL:-0}"
    if [ "${GUARD_OOM_KILL:-0}" -gt 0 ] 2>/dev/null; then
      glog "WARN: guardrail OOM-killed driver (oom_kill=${GUARD_OOM_KILL}, rc=${DRIVER_RC})"
    fi
  else
    glog "guardrail scope reaped before telemetry read (see last [slo-sample] line)"
  fi
  # 卫生：清掉可能滞留 user manager 的 failed scope 单元（--collect 未加
  # 以保事后取证；新 scope 名唯一、不清也不影响后续运行，reset-failed
  # 仅清列表项/计数）。
  systemctl --user reset-failed 2>/dev/null || true
fi

if [ "${DRIVER_RC}" -ne 0 ]; then
  if [ "${DRIVER_RC}" -ne 1 ]; then
    warn "FAIL: slo_postprocess_driver.py (exit=${DRIVER_RC}) — marked in slo_postprocess.FAIL (simulation result NOT overturned)"
    printf '%s (exit=%s)\n' "slo_postprocess_driver.py" "${DRIVER_RC}" >> "${FAIL_MARKER}"
  fi
  SLO_FAIL=1
fi

if [ "${SLO_FAIL}" -ne 0 ]; then
  warn "done WITH FAILURES (default=warn; see slo_postprocess.FAIL)"
  exit 1
fi
log "done: all steps passed"
echo "[slo-postprocess] done: ${RUN_DIR}"
exit 0
