#!/usr/bin/env bash
# archive_run_outputs.sh —— D1 成功运行产物瘦身归档(2026-08-28)。
#
# Usage: bash archive_run_outputs.sh <run_dir> [metrics_detail]
#   metrics_detail = 调用方 runner 实际使用的 --metrics-detail 档位
#     (off|summary|full);缺省按未知档处理——无 [METRIC] 行即 FAIL(fail-closed)。
#
# 前提:仅由 run_online_strategy*.sh 在整条链路(C++/Python/postprocess)
# 全部成功后调用;失败 run 不归档、全量保留供排查(调用方保证)。
#
# 动作(各步骤独立、可幂等重跑):
#   ① 抽取 cpp.log 的 [METRIC] 行 → <run_dir>/metrics.log(常驻,唯一指标
#      数据源,不压缩以便 grep);
#   ② 兼容旧运行:若仍有 bridge/request_*.json,归入
#      <run_dir>/bridge_requests.tar.gz 后删散装;新运行使用单一
#      results/request_journal.jsonl,不再执行百万 inode 归档；
#   ③ cpp.log → cpp.log.gz(pigz 优先,缺省 gzip;-c 走 tmp+mv 原子替换);
#   ④ results/ 非必需 jsonl → <run_dir>/results_extra.tar.gz 后删散装。
#
# 常驻保留集(指标必需,不压缩):
#   raw_metrics.csv normalized_metrics.csv request_metrics.csv metrics.log
#   python.log postprocess.log
#   results/request_journal.jsonl results/online_decision_log.jsonl
#   results/graph_batch_digests.jsonl results/online_stats.jsonl
#   results/profile.jsonl(存在时) results/train_ledger.jsonl
#   results/ledger.jsonl results/sensing_query_log.jsonl(仅 sensing 跑产生,
#     对账/差异报告输入;strategy 跑无此二件,保留集不受影响)
#   results/remote_memory_transactions.jsonl(SerDes 片外链路并发化改造
#     2026-09-24 新增:仅 --sensing-enabled 开启时 C++ 向 bridge_dir 流式
#     写出的逐事务明细,runner 成功后移入 results/;sensing 关跑无此文件,
#     保留集不受影响。shared_busy_ns 复算/完成分布对账输入,常驻不压缩)
#   results/kv_delta_journal.jsonl + results/kv_delta_journal_checksum.json
#     (P1 2026-08-30 权威 HBM delta journal 与 run 末 checksum 门产物,
#     journal 开关 off 时不存在;审计/重放输入,常驻不压缩)
#   campaign_provenance.json(存在时)
#   SLO 自动提取产物(P3,2026-08-28,由 run_slo_postprocess.sh 写在 run_dir
#   根,本脚本只新增 metrics.log/cpp.log.gz 并删除 cpp.log（run_dir 根级）——其余 run_dir 根级文件不触碰,此处显式登记为常驻):slo_*.csv、
#   slo_*.json、cache_events.csv、kv_hit_states.csv、slo_postprocess.log、
#   slo_postprocess.FAIL(若有)、metrics_manifest.json/manifest.json(P2
#   拷入的 per-request manifest,run_dir 自包含的关键件)、
#   trace_config.csv.snapshot(P4 2026-10-01 runner 拷入的运行时
#   trace_config 快照,load_imbalance CV 分母的 run 本地权威来源)
set -uo pipefail

RUN_DIR=${1:?"usage: archive_run_outputs.sh <run_dir> [metrics_detail]"}
if [ ! -d "${RUN_DIR}" ]; then
  echo "[archive] FAIL: run dir not found: ${RUN_DIR}" >&2
  exit 1
fi

# metrics_detail 档位校验(off|summary|full;缺省=未知档,见 ① 的豁免语义)。
METRICS_DETAIL=${2:-}
case "${METRICS_DETAIL}" in
  ""|off|summary|full) ;;
  *) echo "[archive] FAIL: invalid metrics_detail '${METRICS_DETAIL}' (expected off|summary|full)" >&2; exit 1 ;;
esac

if command -v pigz >/dev/null 2>&1; then
  COMPRESS=(pigz)
else
  COMPRESS=(gzip)
fi

# ① [METRIC] 行抽取(cpp.log 尚在时;已归档重跑则跳过)。
#    off 档豁免(2026-09-26,方案 A):--metrics-detail=off 是受支持档位,
#    C++ 端合法地不落任何 [METRIC] 行(与 run_slo_postprocess.sh 的
#    "by design 整步跳过"同口径)——此时无 [METRIC] 行不算失败,清掉空
#    metrics.log 后照常归档。summary/full 档(及未传档位的裸调用)无
#    [METRIC] 行仍 FAIL:采集异常不得被静默归档掩盖。
if [ -f "${RUN_DIR}/cpp.log" ]; then
  grep '^\[METRIC\] ' "${RUN_DIR}/cpp.log" > "${RUN_DIR}/metrics.log" || true
  METRIC_LINES=$(wc -l < "${RUN_DIR}/metrics.log")
  if [ "${METRIC_LINES}" -eq 0 ]; then
    rm -f "${RUN_DIR}/metrics.log"
    if [ "${METRICS_DETAIL}" != "off" ]; then
      echo "[archive] FAIL: no [METRIC] lines in cpp.log (refusing to archive; metrics detail=${METRICS_DETAIL:-unknown})" >&2
      exit 1
    fi
    echo "[archive] no [METRIC] lines (metrics detail=off, by design) — metrics.log skipped"
  else
    echo "[archive] metrics.log: ${METRIC_LINES} [METRIC] lines retained"
  fi
fi

# ② bridge/request_*.json 散装 → 单 tar.gz。
if compgen -G "${RUN_DIR}/bridge/request_*.json" > /dev/null; then
  LIST=$(mktemp /tmp/archive_bridge_files.XXXXXX) || { echo "[archive] FAIL: mktemp for bridge list" >&2; exit 1; }
  find "${RUN_DIR}/bridge" -maxdepth 1 -name 'request_*.json' -printf '%f\n' \
    > "${LIST}"
  tar -C "${RUN_DIR}/bridge" --null --files-from \
    <(tr '\n' '\0' < "${LIST}") -I "${COMPRESS[0]}" \
    -cf "${RUN_DIR}/bridge_requests.tar.gz" \
    || { rm -f "${LIST}"; echo "[archive] FAIL: bridge tar" >&2; exit 1; }
  while IFS= read -r name; do
    rm -f "${RUN_DIR}/bridge/${name}"
  done < "${LIST}"
  rm -f "${LIST}"
  echo "[archive] bridge_requests.tar.gz: \
$(tar -tf "${RUN_DIR}/bridge_requests.tar.gz" | wc -l) request files"
fi

# ③ cpp.log → cpp.log.gz(-c 走 tmp+mv,成功才删原文件)。
if [ -f "${RUN_DIR}/cpp.log" ]; then
  "${COMPRESS[@]}" -c "${RUN_DIR}/cpp.log" > "${RUN_DIR}/cpp.log.gz.tmp" \
    && mv "${RUN_DIR}/cpp.log.gz.tmp" "${RUN_DIR}/cpp.log.gz" \
    && rm -f "${RUN_DIR}/cpp.log" \
    || { rm -f "${RUN_DIR}/cpp.log.gz.tmp"; echo "[archive] FAIL: cpp.log gz" >&2; exit 1; }
  echo "[archive] cpp.log.gz: $(du -h "${RUN_DIR}/cpp.log.gz" | cut -f1)"
fi

# ④ results/ 非必需 jsonl → results_extra.tar.gz(常驻集见头部注释)。
if [ -d "${RUN_DIR}/results" ]; then
  EXTRA_LIST=$(mktemp /tmp/archive_results_extra.XXXXXX) || { echo "[archive] FAIL: mktemp for results-extra list" >&2; exit 1; }
  find "${RUN_DIR}/results" -maxdepth 1 -type f \
    ! -name 'request_journal.jsonl' \
    ! -name 'online_decision_log.jsonl' \
    ! -name 'graph_batch_digests.jsonl' \
    ! -name 'online_stats.jsonl' \
    ! -name 'profile.jsonl' \
    ! -name 'train_ledger.jsonl' \
    ! -name 'ledger.jsonl' \
    ! -name 'sensing_query_log.jsonl' \
    ! -name 'remote_memory_transactions.jsonl' \
    ! -name 'campaign_provenance.json' \
    ! -name 'kv_delta_journal.jsonl' \
    ! -name 'kv_delta_journal_checksum.json' \
    -printf '%f\n' > "${EXTRA_LIST}"
  if [ -s "${EXTRA_LIST}" ]; then
    tar -C "${RUN_DIR}/results" --null --files-from \
      <(tr '\n' '\0' < "${EXTRA_LIST}") -I "${COMPRESS[0]}" \
      -cf "${RUN_DIR}/results_extra.tar.gz" \
      || { rm -f "${EXTRA_LIST}"; echo "[archive] FAIL: results tar" >&2; exit 1; }
    while IFS= read -r name; do
      rm -f "${RUN_DIR}/results/${name}"
    done < "${EXTRA_LIST}"
    echo "[archive] results_extra.tar.gz: $(wc -l < "${EXTRA_LIST}") files"
  fi
  rm -f "${EXTRA_LIST}"
fi

echo "[archive] done: ${RUN_DIR}"
