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
STRESS_JSON="hardware/face_case5_config_c_stress.json"

if [[ ! -f "${SRC_CSV}" ]]; then
  echo "[stress-fixture] source csv missing: ${SRC_CSV}" >&2
  exit 1
fi
BIN=${PROJECT}/build/astra_analytical/build_congestion_aware/bin/AstraSim_Analytical_Congestion_Aware_Online
if [[ ! -x "${BIN}" ]]; then
  echo "[stress-fixture] binary missing: ${BIN}（裸仓交付态——先按 README §4 构建）" >&2
  exit 1
fi

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
trap restore_config EXIT

# 三行切换：输入队列 → 物化 CSV；硬件源 → stress 孪生；容量档 → 压力档。
set_config_row request_queue_csv "${REQUEST_CSV}"
set_config_row hardware_config "${STRESS_JSON}"
set_config_row local_hbm_capacity_profile "${PROFILE}"

# plan 物化（digest 变化 → 新 plan 目录；物化前清历史残留保"恰好一个"）。
rm -rf "${PROJECT}"/sh_test_mesh/generated/llama2_7b_inference_54npus_plan_* 2>/dev/null || true
( cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference" \
    && python3 plan_materializer.py ) \
  > "${EVIDENCE_ROOT}/plan_materialize.log" 2>&1

RC_DIR="${PROJECT}/sh_test_mesh/generated/runtime_config/face_case5_config_c_stress__${PROFILE}__edge_remote_memory_pool"
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

# ---- 判据 2/3：决策日志与台账侧车。----
MODEL_LAYERS=$(awk -F',' '$2 == "layers" {print $3}' "${BACKUP}" | tr -d ' \r')
judge_rc=0
python3 - "${RUN_DIR}" "${MODEL_LAYERS}" <<'PYEOF' || judge_rc=$?
import json
import sys
from pathlib import Path

run_dir, layers = Path(sys.argv[1]), int(sys.argv[2])
log = run_dir / "results" / "online_decision_log.jsonl"
hits, copies = 0, 0
if log.is_file():
    for line in log.open(encoding="utf-8"):
        try:
            record = json.loads(line)
        except ValueError:
            continue
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
sidecar_present = ledgers_path.is_file()
if sidecar_present:
    ledgers = json.loads(ledgers_path.read_text(encoding="utf-8"))
    degrade = ledgers.get("merge_degrade_events")
    deep = ledgers.get("deep_gap_events")
degrade_count = len(degrade) if isinstance(degrade, list) else degrade
summary = {
    "partial_copy_hits": hits,
    "copy_prefill_rows": copies,
    "model_layers": layers,
    "ledger_sidecar_present": sidecar_present,
    "merge_degrade_events": degrade,
    "merge_degrade_disclosure": {
        "count": degrade_count,
        "gated": False,
        "reason": "R4 in-design pricing behavior, chain-coupled with "
                  "PARTIAL formation in the 10 s window (kimi P1, "
                  "2026-09-15)",
    },
    "deep_gap_events": deep,
}
(run_dir / "judge_summary.json").write_text(
    json.dumps(summary, indent=1) + "\n", encoding="utf-8")
print(json.dumps(summary))
# 硬门禁 = 命中>0 且侧车在且 deep_gap 空；merge_degrade 仅落盘披露
# （kimi P1）。侧车缺失 = fail-closed（三轮深审补强：此前缺失时判据 3
# 空转通过——None 恒过，证据链断裂应 FAIL）。
sys.exit(0 if (hits > 0 and sidecar_present and not deep) else 2)
PYEOF

echo "[stress-fixture] combo=${COMBO} profile=${PROFILE} window=${WINDOW_NS}"
echo "[stress-fixture] run exit=${rc}; judge exit=${judge_rc}"
if [[ ${rc} -eq 0 && ${judge_rc} -eq 0 ]]; then
  echo "[stress-fixture] PASS（GREEN + PARTIAL×copy>0 + deep_gap 空；merge_degrade 披露见 ${RUN_DIR}/judge_summary.json）"
  exit 0
fi
echo "[stress-fixture] FAIL——tail run.log:" >&2
tail -15 "${RUN_DIR}/run.log" >&2 || true
exit 1
