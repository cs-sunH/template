#!/usr/bin/env bash
# joint_capacity_stress_fixture.sh -- R16-4-7 容量压力夹具（验证阶梯 L2）
#（joint第二次修改方案.md §6.2：决定性验证缺口——PARTIAL 是规模压力产物，
#  100 请求冒烟在原理上覆盖不到，不可再省略压力档）。
#
# 目的：小 HBM（stress 硬件档）+ 足量并发使 E 逐出真实发生 → PARTIAL 形成
# → 跨实例 copy 至少命中一次且全程 GREEN。RED→GREEN 纪律：未修复代码上
# 本夹具应复现 D2-1 崩溃（_noc_transfer 全驻留守卫 raise → 桥 fail-closed →
# C++ SIGABRT rc=134）；R16 修复后转绿。
#
# 用法：
#   bash sh_test_mesh/run_scripts/joint_capacity_stress_fixture.sh \
#        [window_ns] [capacity_profile] [evidence_root] [combo]
# 缺省：window 10000000000（10s，Agents.md 冒烟纪律上限档——2s 窗无逐出
#   压力，压力夹具按"满足测试需求的最小规模"取 10s）、profile stress-28gib
#  （最大单会话 ~11.7 GiB/rank（524288 B/token 全实例 @TP=6）留充分余量；
#   压力来自 sticky-home 倾斜 + copy 工作副本双驻留，非聚合不足）、
#   证据根 /home/sunhao/joint_r16_stress_evidence（持久路径）、combo TJE。
# 判据（全过才 PASS）：
#   1) run_online_strategy.sh 退出码 0；
#   2) PARTIAL×copy 命中 > 0：决策日志 prefill 行 joint_action=="copy" 且
#      noc_migrate 腿 layer_end < L（R16 复合前缀腿的真实证据，非绕行）；
#   3) deep_gap 台账空（硬门禁，D2 残留探测器；键名 = deep_gap_events，
#      与侧车导出一致——旧版误读 deep_gap_records 恒 None 空转；侧车
#      joint_kv_ledgers.json 缺失 = fail-closed：正常结束必导出，缺失
#      即证据链断裂。两处均外部审查处置批修正）；merge_degrade 计数
#      落盘披露
#      （judge_summary.json）不门禁——R4 设计内计价行为，且 10s 窗内
#      PARTIAL 形成通道与其同链不可分（外部审查 kimi P1 处置
#      2026-09-15：此前"两台账空"硬门禁使缺省 28gib 档必然 FAIL，与
#      夹具自身的 RED→GREEN 目标自相矛盾）。中量程参考：150s ×
#      stress-128gib 档 merge_degrade 披露值亦为 0。
#   4) R17 判据 4（仅 stress-96gib 有逐出档生效；其他档跳过并披露）：
#      链内 SLO 管线软标记升级为硬门 + 非空门（kimi B1，方案 v4 §5c）——
#      runner P3 块（run_online_strategy.sh）缺省 warn 档链内已跑
#      run_slo_postprocess.sh（SR-8 定谳：mtime 铁证 slo 日志早于
#      run.log 73ms），本判据不新增调用，只把链内结果升级为夹具硬门：
#      a) run_dir 无 slo_postprocess.FAIL（SLO 管线 rc=0，含
#         hbm_watermark 对 kv_eviction 新 kind 的消费）；
#      b) hbm 三件套存在（slo_hbm_intervals / plot_series /
#         watermark_instances）；
#      c) 非空门：决策日志 kind=kv_eviction 行 ≥1（R17-1b 披露面；该档
#         修复前在盘实测 23 条披露逐出 + 18 stall + 15 failed——28gib
#         缺省档旧词表零披露逐出会使本判据空转假全绿（kimi 终审 P2
#         措辞订正 2026-09-17：R17 起通道 2 亦披露，10s 窗 28gib 实测
#         1 条 kv_eviction、非空门虽会过但披露面远薄于 96gib×150s），
#         故判据 4 绑定 96gib 档；压力带窄
#         注记：44gib rc=134 崩溃、128gib 两台账皆净，可用窗口居中）。
#         **窗长勘误（R17 施工批实证，2026-09-17）：判据 4 需 150s 窗**
#         （第一参数 150000000000）——10s 窗负载填不满 96 GiB/rank，
#         实测 270 请求零逐出零 PARTIAL×copy 命中（判据 2/4 双空转）；
#         150s 窗 = 在盘 L3 档同口径（1918 请求，修复后 run 复现
#         23 披露逐出条目 + 18 stall + 15 failed 逐位一致 + kv_eviction
#         新增披露 ≥1；R17 首跑实测 1 条）。
#      R17-1b/2b 同批时序约束（kimi B2）：水印对未知 kind 在 kind 门
#      fail-closed，故判据 4 只有在仿真侧与工具侧改动同批落地后才可
#      启用（缺一即本判据必红）。
# 边界：SIGKILL 不还原 trace_config（不可捕获信号；沿 joint_smoke_matrix.sh
#   §9-低4 登记）；裸仓终检 clean_test_records.sh 兜底。
# 前置：C++ 二进制已构建（README §4）；源 CSV =
#   /home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv。
set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)
PROJECT=$(realpath "${SCRIPT_DIR}/../..")
WINDOW_NS=${1:-10000000000}
PROFILE=${2:-stress-28gib}
EVIDENCE_ROOT=${3:-/home/sunhao/joint_r16_stress_evidence}
COMBO=${4:-TJE}
SRC_CSV=${SH_SMOKE_SOURCE_CSV:-/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv}
# N8（2026-09-23 复核审计8）：SH_STRESS_JSON 可覆盖——C18 三孪生档位
# 冒烟入口（hardware/face_case5_config_c_d2d_x05.json = 2025 / _d2d_x2
# = 8100 / _d2d_sub = 1200 GB/s 任一）；缺省 stress JSON 不变（K 批
# stress 语义零漂移）。
STRESS_JSON="${SH_STRESS_JSON:-hardware/face_case5_config_c_stress.json}"

if [[ ! -f "${SRC_CSV}" ]]; then
  echo "[stress-fixture] source csv missing: ${SRC_CSV}" >&2
  exit 1
fi
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
if [[ ! -x "${BIN}" ]]; then
  echo "[stress-fixture] binary missing: ${BIN}（裸仓交付态——先按 README §4 构建）" >&2
  exit 1
fi

# 与 joint_runner 共用仓级单仿真锁。压力入口会物化队列、切换 trace_config
# 并清理 generated；这些副作用都必须发生在锁内。内层 runner 继承 FD 复用
# 同一 open-file description，不重复抢锁。
SINGLE_SIMULATION_LOCK_PATH="${PROJECT}/sh_test_mesh/runs/.single_simulation.lock"
SINGLE_SIMULATION_LOCK_FD=""
SINGLE_SIMULATION_LOCK_OWNED=0
acquire_single_simulation_lock() {
  mkdir -p "${PROJECT}/sh_test_mesh/runs"
  local inherited_fd="${SH_SINGLE_SIMULATION_LOCK_FD:-}"
  if [[ -n "${inherited_fd}" ]]; then
    if [[ ! "${inherited_fd}" =~ ^[0-9]+$ ]]; then
      echo "[stress-fixture] invalid SH_SINGLE_SIMULATION_LOCK_FD=${inherited_fd@Q}" >&2
      return 1
    fi
    local fd_identity lock_identity
    fd_identity=$(stat -Lc '%d:%i' "/proc/${BASHPID}/fd/${inherited_fd}" 2>/dev/null) || {
      echo "[stress-fixture] inherited simulation lock FD ${inherited_fd} is not open" >&2
      return 1
    }
    lock_identity=$(stat -Lc '%d:%i' "${SINGLE_SIMULATION_LOCK_PATH}") || return 1
    if [[ "${fd_identity}" != "${lock_identity}" ]] || ! flock -n "${inherited_fd}"; then
      echo "[stress-fixture] inherited simulation lock FD does not hold ${SINGLE_SIMULATION_LOCK_PATH}" >&2
      return 1
    fi
    SINGLE_SIMULATION_LOCK_FD="${inherited_fd}"
  else
    # 上一轮夹具的后台后代（C++/Python 继承锁 fd，见 EXIT trap 注释）可能
    # 比本壳晚退出：紧邻的下一次夹具调用（stress-r1 紧随门禁内同名夹具、
    # stress-verify 紧随 stress-rN）会在瞬时持锁期被拒。先做有界等待
    # （SINGLE_SIMULATION_LOCK_WAIT_S，默认 120s；轮询 2s）——等待发生在
    # 任何变更（物化/切配置/清 generated）之前，锁真实空闲才继续，超时
    # 仍被持有则维持原 fail-closed 拒绝，互斥语义不变。
    local wait_deadline=$(( $(date +%s) + ${SINGLE_SIMULATION_LOCK_WAIT_S:-120} ))
    while :; do
      exec {SINGLE_SIMULATION_LOCK_FD}>>"${SINGLE_SIMULATION_LOCK_PATH}"
      if flock -n "${SINGLE_SIMULATION_LOCK_FD}"; then
        SINGLE_SIMULATION_LOCK_OWNED=1
        break
      fi
      exec {SINGLE_SIMULATION_LOCK_FD}>&-
      if [ "$(date +%s)" -ge "${wait_deadline}" ]; then
        echo "[stress-fixture] another simulation holds ${SINGLE_SIMULATION_LOCK_PATH}; refusing to mutate run inputs" >&2
        return 1
      fi
      sleep 2
    done
  fi
  export SH_SINGLE_SIMULATION_LOCK_FD="${SINGLE_SIMULATION_LOCK_FD}"
}
acquire_single_simulation_lock

mkdir -p "${EVIDENCE_ROOT}"
# 物化到固定路径（request_queue_csv 路径进 trace-config digest，换路径会
# 物化出第二个 plan 目录、撞 run_online_strategy 的"恰好一个"断言）。
MATERIALIZED="${EVIDENCE_ROOT}/stress_input"
mkdir -p "${MATERIALIZED}"
python3 "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/traces/materialize_20_30s.py" \
  "${SRC_CSV}" "${MATERIALIZED}/requests.csv" "${WINDOW_NS}" \
  > "${EVIDENCE_ROOT}/materialize.log" 2>&1
REQUEST_CSV=$(realpath "${MATERIALIZED}/requests.csv")

TRACE_CONFIG="${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/trace_config.csv"
BACKUP="${EVIDENCE_ROOT}/trace_config.backup.csv"
cp "${TRACE_CONFIG}" "${BACKUP}"

set_config_row() {  # set_config_row <key> <value>
  python3 - "${TRACE_CONFIG}" "$1" "$2" <<'PYEOF'
import sys
path, key, value = sys.argv[1], sys.argv[2], sys.argv[3]
lines = open(path, encoding="utf-8").read().splitlines(keepends=True)
out, hit = [], 0
for line in lines:
    fields = line.rstrip("\n").split(",")
    if len(fields) > 2 and fields[1] == key:
        fields[2] = value
        line = ",".join(fields) + "\n"
        hit += 1
    out.append(line)
if hit != 1:
    raise SystemExit(f"config row {key!r}: expected exactly 1 hit, got {hit}")
open(path, "w", encoding="utf-8").writelines(out)
PYEOF
}

restore_config() { cp "${BACKUP}" "${TRACE_CONFIG}"; }
finish_stress_fixture() {
  local exit_status=$?
  if ! restore_config; then
    echo "[stress-fixture] failed to restore trace_config" >&2
    [[ ${exit_status} -ne 0 ]] || exit_status=1
  fi
  # Restore the shared trace pointer before closing our lock FD. Let inherited
  # descendants retain the flock if one outlives this shell.
  if [[ ${SINGLE_SIMULATION_LOCK_OWNED} -eq 1 ]]; then
    exec {SINGLE_SIMULATION_LOCK_FD}>&-
  fi
  return "${exit_status}"
}
trap finish_stress_fixture EXIT

# 三行切换：输入队列 → 物化 CSV；硬件源 → stress 孪生；容量档 → 压力档。
set_config_row request_queue_csv "${REQUEST_CSV}"
set_config_row hardware_config "${STRESS_JSON}"
set_config_row local_hbm_capacity_profile "${PROFILE}"

# plan 物化（digest 变化 → 新 plan 目录；物化前清历史残留保"恰好一个"）。
rm -rf "${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_plan_* 2>/dev/null || true
( cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference" \
    && python3 plan_materializer.py ) \
  > "${EVIDENCE_ROOT}/plan_materialize.log" 2>&1

# N8：RC 目录名从所选 JSON 基名派生（SH_STRESS_JSON 覆盖孪生档时不再
# 落 stress 基名；容量档 PROFILE 须在该 JSON 的 capacity-profiles 内
#——孪生 JSON 继承 base 的 paper-64gib/validation-160gib，无 28gib
# 压力档——C18 孪生冒烟用 paper-64gib）。
STRESS_BASENAME=$(basename "${STRESS_JSON}" .json)
RC_DIR="${PROJECT}/sh_test_mesh/generated/runtime_config/${STRESS_BASENAME}__${PROFILE}__edge_remote_memory_pool"
if [[ ! -d "${RC_DIR}" ]]; then
  echo "[stress-fixture] runtime config dir missing: ${RC_DIR}" >&2
  ls "${PROJECT}/sh_test_mesh/generated/runtime_config/" >&2 || true
  exit 1
fi

RUN_DIR="${EVIDENCE_ROOT}/${COMBO}_${PROFILE}"
RUNNER_LOG="${EVIDENCE_ROOT}/${COMBO}_${PROFILE}.runner.log"
# 证据保全（报告01 教训）：既有 run 目录归档留档而非 rm——失败现场一律
# 保留（RED 崩溃 run 与 GREEN run 同名共存于此根下时靠时间戳区分）。
if [[ -d "${RUN_DIR}" ]]; then
  mv "${RUN_DIR}" "${RUN_DIR}.prev.$(date +%Y%m%d%H%M%S)"
fi
rm -f "${RUNNER_LOG}"
mkdir -p "${RUN_DIR}"
rc=0
env JOINT_ABLATION_COMBO="${COMBO}" SH_RUNTIME_RC_DIR="${RC_DIR}" \
  bash "${SCRIPT_DIR}/run_online_strategy.sh" "${RUN_DIR}" "${REQUEST_CSV}" \
  > "${RUNNER_LOG}" 2>&1 || rc=$?
mv "${RUNNER_LOG}" "${RUN_DIR}/run.log" 2>/dev/null || true
echo "${rc}" > "${RUN_DIR}/exit_code"

# ---- 判据 2/3/4：决策日志、台账侧车与链内 SLO 结果。----
MODEL_LAYERS=$(awk -F',' '$2 == "layers" {print $3}' "${BACKUP}" | tr -d ' \r')
judge_rc=0
python3 - "${RUN_DIR}" "${MODEL_LAYERS}" "${PROFILE}" <<'PYEOF' || judge_rc=$?
import json
import sys
from pathlib import Path

run_dir, layers, profile = (
    Path(sys.argv[1]), int(sys.argv[2]), sys.argv[3])
log = run_dir / "results" / "online_decision_log.jsonl"
hits, copies, kv_eviction_rows = 0, 0, 0
if log.is_file():
    for line in log.open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
        if record.get("kind") == "kv_eviction":
            kv_eviction_rows += 1
        decision = record.get("decision") or {}
        if record.get("kind") != "prefill":
            continue
        transfers = decision.get("history_transfers") or []
        noc_legs = [t for t in transfers if t.get("kind") == "noc_migrate"]
        if not noc_legs:
            continue
        if decision.get("joint_action") == "copy":
            copies += 1
            # PARTIAL×copy 判据配方（D-F4-e）：noc 腿 layer_end < L。
            if any(isinstance(t.get("layer_end"), int)
                   and 0 < t["layer_end"] < layers for t in noc_legs):
                hits += 1
ledgers_path = run_dir / "bridge" / "joint_kv_ledgers.json"
degrade = deep = None
export_errors = []
sidecar_present = ledgers_path.is_file()
if sidecar_present:
    ledgers = json.loads(ledgers_path.read_text(encoding="utf-8"))
    # A13'（H1，2026-09-22）：G3 逐键哨兵（<key>_export_error）在场 =
    # 对应键导出失败——deep/degrade 会落 None，"not deep" 恒真 ⇒ 假
    # GREEN（证据链断裂）。哨兵扫描进硬门禁，恢复 G3 之前"任一生产
    # 器失败 ⇒ FAIL"的 fail-closed 强度，且不连坐逐键导出的其余收益。
    export_errors = sorted(
        key for key in ledgers if key.endswith("_export_error"))
    degrade = ledgers.get("merge_degrade_events")
    deep = ledgers.get("deep_gap_events")
degrade_count = len(degrade) if isinstance(degrade, list) else degrade
# ---- 判据 4（R17）：仅 stress-96gib 有逐出档生效（28gib 零逐出会
# 空转假全绿——kimi B1 非空门因此绑定有逐出档）。----
slo_fail = run_dir / "slo_postprocess.FAIL"
hbm_triple = [
    (run_dir / name).is_file()
    for name in ("slo_hbm_intervals.csv", "slo_hbm_plot_series.csv",
                 "slo_hbm_watermark_instances.csv")]
judge4_active = profile == "stress-96gib"
judge4 = {
    "active": judge4_active,
    "profile": profile,
    "slo_postprocess_fail_present": slo_fail.is_file(),
    "hbm_triple_present": all(hbm_triple),
    "kv_eviction_rows": kv_eviction_rows,
    "nonempty_gate": (
        kv_eviction_rows >= 1 if judge4_active else None),
}
judge4_ok = (
    (not slo_fail.is_file() and all(hbm_triple) and kv_eviction_rows >= 1)
    if judge4_active else True)
summary = {
    "partial_copy_hits": hits,
    "copy_prefill_rows": copies,
    "model_layers": layers,
    "ledger_sidecar_present": sidecar_present,
    "ledger_export_errors": export_errors,
    "merge_degrade_events": degrade,
    "merge_degrade_disclosure": {
        "count": degrade_count,
        "gated": False,
        "reason": "R4 in-design pricing behavior, chain-coupled with "
                  "PARTIAL formation in the 10 s window (kimi P1, "
                  "2026-09-15)",
    },
    "deep_gap_events": deep,
    "judge4_kv_eviction_disclosure": judge4,
}
(run_dir / "judge_summary.json").write_text(
    json.dumps(summary, indent=1) + "\n", encoding="utf-8")
print(json.dumps(summary))
# 硬门禁 = 命中>0 且侧车在且无逐键导出哨兵且 deep_gap 空 +（启用时）
# 判据 4 全过；merge_degrade 仅落盘披露（kimi P1）。侧车缺失 =
# fail-closed（三轮深审补强：此前缺失时判据 3 空转通过——None 恒过，
# 证据链断裂应 FAIL）；侧车在场但 <key>_export_error 哨兵在场 = 同
# fail-closed（A13'/H1：G3 逐键导出后单键失败不再拆整个侧车，哨兵即
# 证据链断裂信号——deep_gap 空判据失去证据基础，不得 GREEN）。
sys.exit(0 if (hits > 0 and sidecar_present and not export_errors
               and not deep and judge4_ok)
         else 2)
PYEOF

echo "[stress-fixture] combo=${COMBO} profile=${PROFILE} window=${WINDOW_NS}"
echo "[stress-fixture] run exit=${rc}; judge exit=${judge_rc}"
if [[ ${rc} -eq 0 && ${judge_rc} -eq 0 ]]; then
  echo "[stress-fixture] PASS（GREEN + PARTIAL×copy>0 + deep_gap 空；merge_degrade/judge4 披露见 ${RUN_DIR}/judge_summary.json）"
  exit 0
fi
echo "[stress-fixture] FAIL——tail run.log:" >&2
tail -15 "${RUN_DIR}/run.log" >&2 || true
exit 1
