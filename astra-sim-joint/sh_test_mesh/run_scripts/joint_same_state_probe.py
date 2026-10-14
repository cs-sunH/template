#!/usr/bin/env python3
"""joint_same_state_probe.py -- 式(24) 选择损失的最小可用同状态对照 harness（C7b-2）。

背景（审查报告 2026-09-27 §"第四阶段仍缺的验收"第 1 条）：既有冒烟证据只有
预测器输出的候选代价（``measured`` 恒 null），无法计算时延误差、排序错误与
式(24)选择损失。本工具补"同状态对照"的测量编排面：

1. ``generate``：给定一次基线 run（读其 ``results/online_decision_log.jsonl``
   取各请求边界决策的适用动作与预测代价），为 请求×适用动作 生成逐条**串行**
   仿真命令——同一 trace、同参数，只换强制动作 env。两族臂（修复轮二
   stageB 引入；修复轮三，2026-09-27 对齐调度器互斥合同）：
   - **entry 臂**（P 入口）：``JOINT_FORCE_BOUNDARY_REQUEST=<rid>`` +
     ``JOINT_FORCE_ENTRY_ACTION=<A>``，**不设**
     ``JOINT_FORCE_BOUNDARY_ACTION``（两 ACTION 钩子互斥武装，同设 =
     调度器 fail-closed 启动失败）——逐 P 入口适用动作枚举，entry 钩子
     只强制 P 准入选择，D 边界照常自由决策；
   - **boundary 臂**（P→D 边界同状态对照）：``JOINT_FORCE_BOUNDARY_REQUEST
     =<rid>`` + ``JOINT_FORCE_BOUNDARY_ACTION=<D_action>``，**不设**
     ``JOINT_FORCE_ENTRY_ACTION``（同上互斥）——逐 D 边界适用动作枚举。
     P 面同状态前提由基线的 P 选择承载（``entry_pin`` 披露字段；
     boundary 臂按互斥合同不强制 P 入口，summarize 的
     ``entry_pin_match`` 逐臂核对，P 选择漂移即如实降级披露）。
   契约（调度器侧并行任务施工，本工具不等待其落地）：钩子缺省 off；只把
   指定请求的边界决策强制为指定动作，不改状态构造、不改计价——属测量
   仪器，非双路径开关。本工具**不执行**任何仿真，只产出
   ``probe_plan.json`` / ``probe_commands.sh`` / ``probe_plan.md``。
   另产一个**固定动作回放臂**（``JOINT_SCHEDULER_MODE=load-first``，无强制
   钩子；从透传 env 剔除互斥的 ``JOINT_ABLATION_COMBO``）供宿主开销对照。
   D17 世代核验（修复轮三，fail-early）：generate 先比对 baseline 的
   ``model_semantics_version``（``bridge/joint_mechanism_manifest.json``，
   回落决策行同键）与当前 ``joint_config.MODEL_SEMANTICS_VERSION``——
   不一致 stderr 告警照常生成（旧代 baseline 只宜粗对照：适用面可能
   漂移，臂真跑时可能 fail-closed 如 ``no history to copy``）；
   ``--strict-version`` 直接拒绝生成。
2. ``summarize``：解析各候选 run 的决策日志 + 完成事件，抽各候选实测剩余
   时间，联表输出 预测排序 vs 实测排序、排序错误（成对逆序数 + top-1 错）、
   逐候选误差（预测−实测）与式(24)选择损失 ``actual(argmin 预测) −
   min(actual)``，落 ``probe_summary.md`` + ``probe_summary.jsonl``；boundary
   臂额外核对 entry pin 是否守住（P 入口选择 == pin，否则同状态前提被破坏，
   如实降级披露）。
3. **宿主开销报告项**（新增，v2.0 §19.3 验收纪律第 3 类**待验证预算**）：
   同请求同 trace 下，"两阶段决策开启（joint 模式，含 forced 臂——强制只
   换选择、决策计算照常全量跑）的决策行产生墙时" vs "固定动作回放
   （load-first 臂；缺失时回退 forced 臂并如实披露口径不构成预算证据）"，
   给出逐行墙时比值与"≤10% 预算是否达成"的如实披露（达成/未达成/数据
   不足都照实报——这是待验证预算不是结论）。测量口径（写进输出）：决策侧
   = ``results/online_stats.jsonl`` 汇总行 ``predictor_cost.decision_wall_ns.
   total_ns``（sh30 既有计时字段直接消费，本工具不改调度器加字段）÷ 决策
   日志总行数；该文件缺失时回落决策日志 ``joint_decision_metrics`` 收尾行
   （仅 P 面合计，口径降级披露）。

生产 baseline 的常态路径只在强制 probe hook 命中时写入完整
``causal_prestate``。因此 summarize 在原 baseline 缺该字段时，会严格把
生成计划中 baseline 已选的 instance×action forced 臂作为
``forced_selected_reference``：先核对候选面、tick、epoch/version 等所有
baseline 可观测字段，再要求该臂自身的完整 causal-prestate shape；缺任一项
仍标记无证书，绝不以 synthetic fixture 或空字段补齐。

实测剩余时间口径（防御式，来源逐条披露）：
- 优先取候选 run 决策日志中该动作 ``candidates[].measured``（并行任务可能
  填充；当前样例全 null）；
- 缺失则取 **settle 事件**：``settle_done`` 行 ``settle_done_ns − 边界
  tick``（W5 settle 事件已就绪；consolidate 零载荷时 settle==service 合法
  并显式记录 zero_payload_settle）；
- 再缺失（旧世代日志）回落 ``max(completion.tick, merge_done.tick) −
  边界 tick``（source=legacy_max_completion_merge 如实披露；distributed
  缺 settle 事件**不得**回退 completion=settle——预留 W9 分支）。

W8 扩展（方案 §11.3/§12.1/§3.2 末段）：
- **instance×action 精确强制**：``--granularity instance-action`` 逐适用
  (instance, action) 候选各一臂（``JOINT_FORCE_BOUNDARY_TARGET`` /
  ``JOINT_FORCE_ENTRY_TARGET`` = ``<instance>:<action>``——指定组合不在
  适用面 = 调度器显式 fail-closed，不静默重选实例；旧 per-action env 对
  保留向后兼容但为弃用方向）；summarize 双维面（键 ``i<inst>:<action>``）；
- **状态一致硬证书**：重放前缀在目标边界逐字段核验（tick/seq/冻结候选
  面/stage_epoch/state_versions/义务/stage_work/等待类别/语义版本；P 面
  另含选择与 service_factors）——任一不匹配 ⇒ 该分支 ``state_mismatch``
  **不纳入同状态损失样本**（原 predicted-drift note 软降级退役）；
- **双指标**：``actual_service_done`` 与 ``actual_settle_done`` 分列（各减
  common_decision_tick）；排序/式(24)损失以 **settle** 目标为准；
- **missing / fixed_policy_arm / P 参考接续分离**：未执行/未完成候选标
  missing（不以预测值填补）；replay/one-shot whole-run 臂独立 ``arm_class``
  标记（不与同状态数据混算）；P 参考接续（P 行 decode_instance/
  decode_action）与实际边界 D 分列披露。

决策日志 schema 以现有代码/样例为准（2026-09-27 核验
/home/sunhao/joint_smoke_evidence_v20_b4）：每行
``{kind, request_id, seq, tick, priority, decision}``；
- P 边界 = ``kind=="joint_admission"``：``decision.candidates[]``（action/
  applicable/cost_ns/measured/inapplicable_reason/instance_index）、
  ``decision.applicable_actions``、``decision.selected_action``/``joint_action``；
- D 边界 = ``kind=="decode"`` 且 ``decision.boundary_decision`` 在位：
  ``boundary_decision.candidates[]``（逐 (instance, action) 一条，W8 起双维
  全采集——不再按动作折叠最低成本实例）、``chosen_action``/
  ``chosen_instance_index``、``state_versions``；
- 完成事件 = ``kind=="completion"``（``service_done_ns`` 键）/
  ``kind=="merge_done"`` / ``kind=="settle_done"``（settle_done_ns/
  service_done_ns/merge_payload_bytes/settlement_policy/settle_state）。

字段缺失/形态不符：逐条报清楚（``status="error"`` + reason），不崩溃、不
静默跳过。输出目录带运行标识（缺省 ``probe_<UTC时间戳>``），已存在即拒
绝（防覆盖）。

用法示例：
    python3 sh_test_mesh/run_scripts/joint_same_state_probe.py generate \
        --baseline-run /home/sunhao/joint_smoke_evidence_v20_b4/TJE_quota_static \
        --dry-run
    python3 sh_test_mesh/run_scripts/joint_same_state_probe.py selftest
    # 真跑（主控执行，非本工具）：
    #   bash <evidence>/probe/<run_id>/probe_commands.sh
    # 然后：
    python3 sh_test_mesh/run_scripts/joint_same_state_probe.py summarize \
        --plan <evidence>/probe/<run_id>/probe_plan.json
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
import tempfile
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNNER_SH = _REPO_ROOT / "sh_test_mesh" / "run_scripts" / "run_online_strategy.sh"
_MATERIALIZE_PY = (
    _REPO_ROOT / "sh_test_mesh" / "workload" / "llama2_7b_inference" / "traces"
    / "materialize_20_30s.py")
_PLAN_MATERIALIZER_CWD = (
    _REPO_ROOT / "sh_test_mesh" / "workload" / "llama2_7b_inference")
_TRACE_CONFIG = (
    _PLAN_MATERIALIZER_CWD / "trace_config.csv")
_GENERATED_PLAN_GLOB = (
    _REPO_ROOT / "sh_test_mesh" / "generated" / "llama2_7b_inference_54npus_plan_*")

# 契约常量（调度器侧并行任务施工的验证专用强制钩子；缺省 off）。
FORCE_REQUEST_ENV = "JOINT_FORCE_BOUNDARY_REQUEST"
FORCE_ACTION_ENV = "JOINT_FORCE_BOUNDARY_ACTION"
# P 入口强制钩子（调度器侧修复轮二已落地）：ENTRY 钉 P 准入选择、
# BOUNDARY 钉 P→D 边界选择。两 ACTION env 互斥武装（同设 = 调度器启动
# 失败，fail-closed）——probe 两族臂各只武装其一：entry 臂不设
# BOUNDARY_ACTION，boundary 臂不设 ENTRY_ACTION。boundary 臂的 P 面
# 同状态前提由基线 P 选择承载（entry_pin），summarize 的 entry_pin_match
# 如实核对，未守住即降级披露，不静默当作有效同状态对照。
FORCE_ENTRY_ACTION_ENV = "JOINT_FORCE_ENTRY_ACTION"
# W8（方案 §11.3）：instance×action 精确强制钩子——值形如
# "<instance_index>:<action>"，与共享 REQUEST env 成对武装。强制**精确**
# (request, instance, action)：指定实例不在候选面/组合不适用 ⇒ 调度器
# 显式 fail-closed（不静默重选实例）——全部 (instance,action) 候选的反
# 事实真值前提。旧 per-action 变量保留向后兼容但为弃用方向（同族
# TARGET 与 ACTION 同设 = fail-closed）。
FORCE_BOUNDARY_TARGET_ENV = "JOINT_FORCE_BOUNDARY_TARGET"
FORCE_ENTRY_TARGET_ENV = "JOINT_FORCE_ENTRY_TARGET"

# 固定动作回放臂（宿主开销对照）用的模式开关；与 JOINT_ABLATION_COMBO
# 互斥（joint_config fail-closed），故回放臂从透传 env 剔除该键。
REPLAY_MODE_ENV = "JOINT_SCHEDULER_MODE"
REPLAY_MODE_VALUE = "load-first"
_COMBO_ENV = "JOINT_ABLATION_COMBO"

# 一次映射对照臂（2026-09-28，验收纪律第 2 类"一次映射 vs 两阶段闭环
# 对照"）：run 级策略 env（非逐请求钩子）——JOINT_BOUNDARY_POLICY=
# p-continuation 时 D 边界不做二次全候选重估，直接按 P 准入默认延续
# 形态提交。one-shot 臂族生成**成对**两 run（同 trace 同参数，仅差该
# env）：两阶段参照（不设 env）× 一次映射（设 env）。
ONESHOT_POLICY_ENV = "JOINT_BOUNDARY_POLICY"
ONESHOT_POLICY_VALUE = "p-continuation"

# 与 joint/joint_config.ALL_ACTIONS 同源的规范序（run_scripts 工具不 import
# workload 代码，保持独立可运行；若全集扩容以 joint_config 为准）。
ALL_ACTIONS = ("stay", "recompute", "copy")

DEFAULT_EVIDENCE_ROOT = Path("/home/sunhao/joint_fix_round_20260927/probe")
DEFAULT_SOURCE_CSV = Path(
    "/home/sunhao/wsc-simulator/agent-traces/tracelab/astra_compute_20.csv")
DEFAULT_WINDOW_NS = 2000000000  # 2s 冒烟纪律最小窗（Agents.md；同冒烟矩阵缺省）
DEFAULT_MAX_REQUESTS = 3
# 修复轮二（stageB）：缺省双族臂（entry + boundary）+ 1 回放臂——
# 2 请求 × (P 面 2-3 动作 + D 面 2-4 动作) + 1 常见量级 ≤ 24，仍 fail-closed。
DEFAULT_MAX_COMMANDS = 24

# 宿主开销预算（v2.0 §19.3 验收纪律第 3 类；**待验证预算不是结论**——
# 达成/未达成都如实披露）。
OVERHEAD_BUDGET_RATIO = 1.10

HOOK_CONTRACT_NOTE = {
    "request_env": FORCE_REQUEST_ENV,
    "boundary_action_env": FORCE_ACTION_ENV,
    "entry_action_env": FORCE_ENTRY_ACTION_ENV,
    "boundary_target_env": FORCE_BOUNDARY_TARGET_ENV,
    "entry_target_env": FORCE_ENTRY_TARGET_ENV,
    "default": "off",
    "mutually_exclusive": True,
    "semantics": (
        "只把指定请求的对应边界决策强制为指定候选；不改状态构造、不改"
        "计价；属测量仪器，非双路径开关。两 ACTION 钩子互斥武装（同设 = "
        "调度器 fail-closed 启动失败），单次运行只武装其一：entry 臂 = "
        "REQUEST + ENTRY(A)（只强制 P 准入选择）；boundary 臂 = REQUEST "
        "+ BOUNDARY(逐 D 适用动作)（只强制 P→D 边界选择，不设 ENTRY——"
        "P 面同状态前提由基线 P 选择承载为 entry_pin）。W8 起另有精确"
        "身份钩子：REQUEST + TARGET(<instance>:<action>)——强制恰好的 "
        "(instance, action) 候选，指定组合不在适用面 = 显式失败（不静默"
        "重选实例）；TARGET 与同族 ACTION 同设 = fail-closed。旧 "
        "per-action ACTION 变量为弃用方向（deprecated: use *_TARGET）。"
        "summarize 以 forced_selected_match / entry_pin_match / 状态一致"
        "证书显式核对，未守住即降级或 state_mismatch 披露，不静默当作有"
        "效对照"),
}

# W8 状态一致硬证书口径（方案 §11.3"一致性是硬门，不是 warning"）。
CERT_RNG_BASIS = "deterministic-replay-no-host-rng"
CERT_COVERAGE_NOTE = (
    "已核验维度（全部为强制前观测量）：tick（D 决策 tick）、P 面 seq（P 行"
    "决策时点发射，前缀序号稳定）、冻结候选面（适用性/代价/继续标记/理由"
    "）、stage_epoch（预约纪元）、state_versions（kv_ledger_epoch/"
    "directory_version/stage_epochs——目录与预约版本，W8 起决策行披露）、"
    "inherited_obligations（义务身份）、decode_steps（共同冻结剩余前向）、"
    "model_semantics_version、causal_prestate（强制前队列/剩余工作/资源/"
    "目录/容量/历史事实）；缺少必需证据也拒绝签发。P 面另含 joint_instance_index/joint_action"
    "（entry pin 时）、service_factors 摘要（历史估计器状态）。**排除维"
    "度（选择派生观测量，非同状态判据——2026-09-29 受控运行实证）**：D 行"
    " seq（发射时点序号，被选择依赖旁行移位）、boundary_wait_kind（由 "
    "f(selected, i_P) 计算）、stage_work（携带所选动作的 rebuild 腿）。随"
    "机状态维度：调度器决策路径无宿主 RNG（确定性回放协议，rng_basis="
    + CERT_RNG_BASIS + "）——前缀等价由 tick/P-seq/事件序与上述全部字段"
    "相等承载，不另行哈希。逐字段原值比较（摘要仅披露形态）；任一字段不"
    "匹配 ⇒ status=state_mismatch，不纳入同状态损失样本。")

# 宿主开销测量口径（写进输出，防口径漂移）。
OVERHEAD_CALIBER_NOTE = (
    "决策侧 = 决策行产生墙时：results/online_stats.jsonl 汇总行 "
    "predictor_cost.decision_wall_ns.total_ns（P+D 两阶段合计；sh30 既有"
    "predictor 计时字段直接消费，本工具未改调度器加字段）÷ 决策日志总行"
    "数；回放侧同口径。该文件缺失时回落决策日志 joint_decision_metrics "
    "收尾行（仅 P 面合计 admission_decision_wall_ns.total_ns，口径降级"
    "如实披露）。宿主全程墙时（host_wall_ns）含 I/O 等噪声，只作伴随披"
    "露，不进比值。")


class ProbeError(Exception):
    """工具级失败（参数/输入不符），打印后以非零码退出。"""


# ---------------------------------------------------------------------------
# 决策日志解析（防御式：字段缺失逐条报清楚，不崩）
# ---------------------------------------------------------------------------

def _iter_log_records(path: Path, errors: list[str], max_errors: int = 5):
    """逐行解析 jsonl；坏行收集进 errors（限量），好行照常产出。"""
    with open(path, encoding="utf-8") as source:
        for lineno, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except ValueError as exc:
                if len(errors) < max_errors:
                    errors.append(f"{path}:{lineno}: bad JSON: {exc}")
                continue
            if not isinstance(record, dict):
                if len(errors) < max_errors:
                    errors.append(f"{path}:{lineno}: record is not an object")
                continue
            yield record


def decision_log_path(run_dir: Path) -> Path:
    """run_online_strategy.sh 归档位：results/online_decision_log.jsonl。"""
    return Path(run_dir) / "results" / "online_decision_log.jsonl"


def find_boundary_record(run_dir: Path, request_id: str, stage: str):
    """在 run 的决策日志中定位指定请求的边界决策记录。

    stage="P" -> 第一条 kind==joint_admission（P→服务边界，预测器比较面）。
    stage="D" -> 第一条含 decision.boundary_decision 的 kind==decode 记录
    （P→D 边界）。返回 (record, dup_count, parse_errors)。
    """
    path = decision_log_path(run_dir)
    if not path.is_file():
        raise ProbeError(
            f"decision log missing: {path} (run 未完成或产物已归档移走?)")
    errors: list[str] = []
    found = None
    dup = 0
    for record in _iter_log_records(path, errors):
        kind = record.get("kind")
        decision = record.get("decision")
        if not isinstance(decision, dict):
            continue  # 无 decision 体的行（如纯指标行）不计
        if record.get("request_id") != request_id:
            continue
        if stage == "P" and kind == "joint_admission":
            if found is None:
                found = record
            else:
                dup += 1
        elif (stage == "D" and kind == "decode"
                and isinstance(decision.get("boundary_decision"), dict)):
            if found is None:
                found = record
            else:
                dup += 1
    if found is None:
        raise ProbeError(
            f"no {stage}-boundary record for request {request_id!r} in "
            f"{path} (kinds seen for this request may differ; check "
            f"request_id spelling)")
    return found, dup, errors


def _require(record: dict, dotted_key: str):
    """按点路径取字段，缺失即 ProbeError（带定位信息）。"""
    value = record
    walked = []
    for part in dotted_key.split("."):
        walked.append(part)
        if not isinstance(value, dict) or part not in value:
            raise ProbeError(
                f"decision log field {dotted_key!r} missing "
                f"(stopped at {'.'.join(walked)!r}); record kind="
                f"{record.get('kind')!r} request_id="
                f"{record.get('request_id')!r} seq={record.get('seq')!r}")
        value = value[part]
    return value


def extract_candidates(record: dict, stage: str) -> list[dict]:
    """取边界决策的候选列表（规范化 action/applicable/cost_ns/measured）。"""
    if stage == "P":
        raw = _require(record, "decision.candidates")
        chosen = record.get("decision", {}).get("selected_action")
        if chosen is None:
            chosen = record.get("decision", {}).get("joint_action")
    else:
        raw = _require(record, "decision.boundary_decision.candidates")
        chosen = record.get("decision", {}).get("boundary_decision", {}).get(
            "chosen_action")
        if chosen is None:
            chosen = record.get("decision", {}).get("joint_action")
    if not isinstance(raw, list) or not raw:
        raise ProbeError(
            f"candidates list empty/malformed for request "
            f"{record.get('request_id')!r} seq={record.get('seq')!r}")
    out = []
    for index, cand in enumerate(raw):
        if not isinstance(cand, dict) or "action" not in cand:
            raise ProbeError(
                f"candidate #{index} malformed (no 'action') for request "
                f"{record.get('request_id')!r} seq={record.get('seq')!r}")
        out.append({
            "action": cand.get("action"),
            "applicable": bool(cand.get("applicable")),
            "cost_ns": cand.get("cost_ns"),
            "measured": cand.get("measured"),
            "instance_index": cand.get("instance_index"),
            "is_continuation": bool(cand.get("is_continuation")),
            "inapplicable_reason": cand.get("inapplicable_reason"),
        })
    return out


def aggregate_by_instance_action(candidates: list[dict]) -> dict:
    """W8 双维聚合（方案 §11.3"不能只按 action 汇总最便宜 instance"）：
    逐 **(instance_index, action)** 候选全量采集——不再每动作折叠到最低
    预测成本实例。值 = {applicable, predicted_ns, measured_ns,
    inapplicable_reason, is_continuation}。"""
    surface: dict[tuple[int, str], dict] = {}
    for cand in candidates:
        key = (cand.get("instance_index"), cand["action"])
        slot = surface.setdefault(key, {
            "applicable": False, "predicted_ns": None, "measured_ns": None,
            "inapplicable_reason": cand.get("inapplicable_reason"),
            "is_continuation": bool(cand.get("is_continuation")),
            "cost_errors": []})
        if cand["applicable"]:
            slot["applicable"] = True
            cost = cand["cost_ns"]
            if isinstance(cost, (int, float)):
                slot["predicted_ns"] = cost
            else:
                slot["cost_errors"].append(
                    "applicable candidate lacks numeric cost_ns")
            measured = cand["measured"]
            if isinstance(measured, (int, float)):
                slot["measured_ns"] = measured
    return surface


def rollup_actions(surface: dict) -> dict[str, dict]:
    """双维面 → 动作级回卷（展示/旧 plan 兼容层）：适用 = 任一实例适用；
    预测 = 适用实例最小；measured = 适用实例最小非空数值。"""
    actions: dict[str, dict] = {}
    for (_instance, action), slot in surface.items():
        rolled = actions.setdefault(
            action,
            {"applicable": False, "predicted_ns": None, "measured_ns": None})
        if slot["applicable"]:
            rolled["applicable"] = True
            cost = slot["predicted_ns"]
            if isinstance(cost, (int, float)):
                if (rolled["predicted_ns"] is None
                        or cost < rolled["predicted_ns"]):
                    rolled["predicted_ns"] = cost
            if slot.get("cost_errors"):
                rolled.setdefault(
                    "cost_errors", []).extend(slot["cost_errors"])
            measured = slot["measured_ns"]
            if isinstance(measured, (int, float)):
                if (rolled["measured_ns"] is None
                        or measured < rolled["measured_ns"]):
                    rolled["measured_ns"] = measured
    return actions


def candidate_key(instance_index, action: str) -> str:
    """双维候选的复合键（predictions/actuals/rank 面用）。"""
    return "i{}:{}".format(instance_index, action)


def selected_action_of(record: dict, stage: str):
    """该边界决策实际选中的动作（强校验用；字段缺失返回 None 并给原因。"""
    decision = record.get("decision") or {}
    if stage == "P":
        return decision.get("selected_action", decision.get("joint_action"))
    boundary = decision.get("boundary_decision") or {}
    return boundary.get("chosen_action", decision.get("joint_action"))


def selected_instance_of(record: dict, stage: str):
    """该边界决策实际选中的实例（W8 强制身份核对用）。"""
    decision = record.get("decision") or {}
    if stage == "P":
        return decision.get("joint_instance_index")
    boundary = decision.get("boundary_decision") or {}
    return boundary.get("chosen_instance_index", decision.get(
        "decode_instance_index"))


# ---------------------------------------------------------------------------
# W8 状态一致硬证书（方案 §11.3：重放前缀在目标边界逐项匹配才发证；
# 任一字段不匹配 ⇒ state_mismatch，不纳入同状态损失样本）
# ---------------------------------------------------------------------------

def _canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False)


def _digest16(value) -> str:
    import hashlib
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()[:16]


def _candidates_surface(candidates: list[dict]) -> list:
    """冻结候选面的规范化（排序消序）：逐候选 [instance, action,
    applicable, cost_ns, is_continuation, inapplicable_reason]。"""
    surface = []
    for cand in candidates:
        surface.append([
            cand.get("instance_index"), cand.get("action"),
            bool(cand.get("applicable")), cand.get("cost_ns"),
            bool(cand.get("is_continuation")),
            cand.get("inapplicable_reason")])
    surface.sort(key=lambda row: _canonical(row))
    return surface


_CAUSAL_PRESTATE_FIELDS = (
    "schema_version", "stage", "request_id", "decision_snapshot",
    "work_remaining", "directory", "capacity", "arbitration", "history",
    "versions")
_CAUSAL_PRESTATE_VERSION_FIELDS = (
    "kv_ledger_epoch", "directory_version", "stage_epochs", "length",
    "arrivals", "efficiency")

_DECISION_SNAPSHOT_FIELDS = (
    "schema_version", "snapshot_id", "now_ns", "delivery_sequence",
    "event_watermark", "versions", "observed_tick", "observation_lag_ns",
    "arrivals", "request_queues", "work_index", "resource_state",
    "existing_unknown_work", "capacity_view", "projection_log")
_DECISION_SNAPSHOT_VERSION_FIELDS = (
    "runtime", "layout", "reservation", "history", "hardware", "model",
    "ledger")
_DECISION_SNAPSHOT_QUEUE_FIELDS = ("instances", "exposed_decode_steps")
_DECISION_SNAPSHOT_INDEX_FIELDS = (
    "by_link", "by_hbm_rank", "by_pool_port", "by_rank_nodes")
_DECISION_SNAPSHOT_RESOURCE_FIELDS = (
    "arbitration", "throughput_observation", "slot_gate",
    "physical_capacity")
_WORK_REMAINING_FIELDS = (
    "instances", "pending_admissions", "arrival_heap", "in_flight",
    "runtimes", "arrived_request_count", "completed_requests")
_DIRECTORY_FIELDS = (
    "present", "version", "entries", "inflight_copies")
_DIRECTORY_PRESENT_FIELDS = (
    "schema_version", "model_version", "tp_degree")
_DIRECTORY_ENTRY_FIELDS = (
    "entry_id", "session_id", "model_version", "layer_start", "layer_end",
    "token_start", "token_end", "tp_rank", "holder", "role",
    "bytes_by_tp_rank", "padding_by_tp_rank", "generation", "state",
    "pending_acks", "release_condition", "read_refs", "exec_pins")
_DIRECTORY_COPY_FIELDS = (
    "copy_id", "session_id", "source_holder", "target_instance",
    "layer_start", "layer_end", "token_start", "token_end",
    "bytes_by_tp_rank", "ack_id")
_CAPACITY_FIELDS = (
    "present", "ledgers_by_instance", "summary_version",
    "summary_error_sources", "summary", "maintenance", "stage_epochs")
_CAPACITY_SUMMARY_FIELDS = (
    "present", "snapshot_version", "error_sources", "instances")
_ARBITRATION_FIELDS = (
    "noc_flows", "noc_owners", "pool_ports", "hbm_ports", "hbm_owners",
    "telemetry", "quota")
_ARBITRATION_POOL_FIELDS = ("counts", "owners")
_ARBITRATION_TELEMETRY_FIELDS = (
    "rates", "flow_counts", "epoch_count", "sample_count", "last_tick_ns",
    "window_broken", "last_window_bounds_ns")
_HISTORY_FIELDS = (
    "length", "arrivals", "efficiency", "service_samples", "horizon",
    "epoch_log")
_HISTORY_LENGTH_FIELDS = (
    "version", "unit", "run_completed_samples", "run_completed_sum",
    "run_max_observed", "run_censored_count", "run_censored_progress_sum",
    "histogram_bucket_width", "histogram_bucket_count",
    "histogram_overflow_tail_count", "min_samples_per_level",
    "cold_start_prior_remainder", "cold_start_tail_distribution")
_HISTORY_ARRIVAL_FIELDS = (
    "version", "arrival_bucket_delta_ns", "open_bucket_rule",
    "observed_until_ns", "open_bucket_index", "categories",
    "excluded_event_counts")
_HISTORY_EFFICIENCY_FIELDS = (
    "efficiency_factors", "updates", "rejected", "anomalous", "contended",
    "duplicate_samples", "stale_samples", "residuals", "hardware_versions",
    "ewma_weight_source")
_HISTORY_SERVICE_FIELDS = (
    "version", "efficiency_factors", "updates", "rejected", "anomalous",
    "contended", "residuals", "hardware_versions", "pending")
_HISTORY_HORIZON_FIELDS = ("version",)
_HISTORY_EPOCH_FIELDS = ("version", "reset_rule", "quarantine_ns", "epochs")


def _shape_mapping_fields(value, prefix: str, required, errors: set[str]):
    """Record missing keys without accepting an empty envelope as evidence."""
    if not isinstance(value, dict):
        errors.add(prefix)
        return False
    for name in required:
        if name not in value:
            errors.add(f"{prefix}.{name}")
    return True


def _shape_list(value, prefix: str, errors: set[str]) -> bool:
    if not isinstance(value, list):
        errors.add(prefix)
        return False
    return True


def _shape_integer(value, prefix: str, errors: set[str], *, nonnegative=False):
    if type(value) is not int or (nonnegative and value < 0):
        errors.add(prefix)


def _shape_directory_entries(value, prefix: str, required, errors: set[str]):
    if not _shape_list(value, prefix, errors):
        return
    for index, item in enumerate(value):
        _shape_mapping_fields(item, f"{prefix}[{index}]", required, errors)


def _causal_prestate_shape_errors(
        value, *, expected_stage: str | None = None,
        expected_request_id: str | None = None) -> set[str]:
    """Require the production causal-prestate envelope and its version view.

    Empty mappings are valid *values* inside a real empty world, but a view
    itself must retain its production keys.  This prevents a synthetic or
    malformed certificate from passing merely because both sides contain the
    same empty envelope.
    """
    errors = set()
    if not isinstance(value, dict):
        return {"causal_prestate"}
    for name in _CAUSAL_PRESTATE_FIELDS:
        if name not in value:
            errors.add(f"causal_prestate.{name}")
    if value.get("schema_version") != "joint-causal-prestate-v1":
        errors.add("causal_prestate.schema_version")
    if type(value.get("stage")) is not str or not value.get("stage"):
        errors.add("causal_prestate.stage")
    elif value["stage"] not in ("P", "D"):
        errors.add("causal_prestate.stage")
    elif expected_stage is not None and value["stage"] != expected_stage:
        errors.add("causal_prestate.stage_match")
    if type(value.get("request_id")) is not str or not value.get("request_id"):
        errors.add("causal_prestate.request_id")
    elif (expected_request_id is not None
          and value["request_id"] != expected_request_id):
        errors.add("causal_prestate.request_id_match")

    snapshot = value.get("decision_snapshot")
    if _shape_mapping_fields(
            snapshot, "causal_prestate.decision_snapshot",
            _DECISION_SNAPSHOT_FIELDS, errors):
        if (not isinstance(snapshot.get("schema_version"), str)
                or not snapshot["schema_version"]):
            errors.add("causal_prestate.decision_snapshot.schema_version")
        if (not isinstance(snapshot.get("snapshot_id"), str)
                or not snapshot["snapshot_id"]):
            errors.add("causal_prestate.decision_snapshot.snapshot_id")
        for name in ("now_ns", "delivery_sequence", "event_watermark",
                     "observed_tick", "observation_lag_ns"):
            _shape_integer(snapshot.get(name),
                           f"causal_prestate.decision_snapshot.{name}",
                           errors)
        versions = snapshot.get("versions")
        if _shape_mapping_fields(
                versions, "causal_prestate.decision_snapshot.versions",
                _DECISION_SNAPSHOT_VERSION_FIELDS, errors):
            pass
        _shape_list(snapshot.get("arrivals"),
                    "causal_prestate.decision_snapshot.arrivals", errors)
        for name in ("request_queues", "work_index", "resource_state",
                     "existing_unknown_work"):
            if not isinstance(snapshot.get(name), dict):
                errors.add(f"causal_prestate.decision_snapshot.{name}")
        if isinstance(snapshot.get("request_queues"), dict):
            _shape_mapping_fields(
                snapshot["request_queues"],
                "causal_prestate.decision_snapshot.request_queues",
                _DECISION_SNAPSHOT_QUEUE_FIELDS, errors)
        if isinstance(snapshot.get("work_index"), dict):
            _shape_mapping_fields(
                snapshot["work_index"],
                "causal_prestate.decision_snapshot.work_index",
                _DECISION_SNAPSHOT_INDEX_FIELDS, errors)
        if isinstance(snapshot.get("resource_state"), dict):
            _shape_mapping_fields(
                snapshot["resource_state"],
                "causal_prestate.decision_snapshot.resource_state",
                _DECISION_SNAPSHOT_RESOURCE_FIELDS, errors)
        capacity_view = snapshot.get("capacity_view")
        if capacity_view is not None and not isinstance(capacity_view, dict):
            errors.add("causal_prestate.decision_snapshot.capacity_view")
        _shape_list(snapshot.get("projection_log"),
                    "causal_prestate.decision_snapshot.projection_log", errors)

    work = value.get("work_remaining")
    if _shape_mapping_fields(
            work, "causal_prestate.work_remaining", _WORK_REMAINING_FIELDS,
            errors):
        for name in ("instances", "pending_admissions", "arrival_heap",
                     "runtimes"):
            _shape_list(work.get(name), f"causal_prestate.work_remaining.{name}",
                        errors)
        if not isinstance(work.get("in_flight"), dict):
            errors.add("causal_prestate.work_remaining.in_flight")
        _shape_integer(work.get("arrived_request_count"),
                       "causal_prestate.work_remaining.arrived_request_count",
                       errors, nonnegative=True)
        _shape_integer(work.get("completed_requests"),
                       "causal_prestate.work_remaining.completed_requests",
                       errors, nonnegative=True)

    directory = value.get("directory")
    if _shape_mapping_fields(
            directory, "causal_prestate.directory", _DIRECTORY_FIELDS, errors):
        present = directory.get("present")
        if type(present) is not bool:
            errors.add("causal_prestate.directory.present")
        _shape_list(directory.get("entries"),
                    "causal_prestate.directory.entries", errors)
        _shape_list(directory.get("inflight_copies"),
                    "causal_prestate.directory.inflight_copies", errors)
        if present is True:
            _shape_mapping_fields(
                directory, "causal_prestate.directory",
                _DIRECTORY_FIELDS + _DIRECTORY_PRESENT_FIELDS, errors)
            if not isinstance(directory.get("schema_version"), str) \
                    or not directory["schema_version"]:
                errors.add("causal_prestate.directory.schema_version")
            if not isinstance(directory.get("model_version"), str) \
                    or not directory["model_version"]:
                errors.add("causal_prestate.directory.model_version")
            _shape_integer(directory.get("tp_degree"),
                           "causal_prestate.directory.tp_degree", errors,
                           nonnegative=True)
            _shape_integer(directory.get("version"),
                           "causal_prestate.directory.version", errors,
                           nonnegative=True)
        elif present is False and (
                directory.get("entries") or directory.get("inflight_copies")):
            errors.add("causal_prestate.directory.absent_payload")
        _shape_directory_entries(
            directory.get("entries"), "causal_prestate.directory.entries",
            _DIRECTORY_ENTRY_FIELDS, errors)
        _shape_directory_entries(
            directory.get("inflight_copies"),
            "causal_prestate.directory.inflight_copies", _DIRECTORY_COPY_FIELDS,
            errors)

    capacity = value.get("capacity")
    if _shape_mapping_fields(
            capacity, "causal_prestate.capacity", _CAPACITY_FIELDS, errors):
        if capacity.get("present") is not True:
            errors.add("causal_prestate.capacity.present")
        if not isinstance(capacity.get("ledgers_by_instance"), dict):
            errors.add("causal_prestate.capacity.ledgers_by_instance")
        if not isinstance(capacity.get("summary_error_sources"), list):
            errors.add("causal_prestate.capacity.summary_error_sources")
        _shape_list(capacity.get("stage_epochs"),
                    "causal_prestate.capacity.stage_epochs", errors)
        summary = capacity.get("summary")
        if _shape_mapping_fields(
                summary, "causal_prestate.capacity.summary",
                _CAPACITY_SUMMARY_FIELDS, errors):
            if summary.get("present") is not True:
                errors.add("causal_prestate.capacity.summary.present")
            if not isinstance(summary.get("error_sources"), list):
                errors.add("causal_prestate.capacity.summary.error_sources")
            if not isinstance(summary.get("instances"), dict):
                errors.add("causal_prestate.capacity.summary.instances")

    arbitration = value.get("arbitration")
    if _shape_mapping_fields(
            arbitration, "causal_prestate.arbitration", _ARBITRATION_FIELDS,
            errors):
        pool_ports = arbitration.get("pool_ports")
        if isinstance(pool_ports, dict):
            _shape_mapping_fields(
                pool_ports, "causal_prestate.arbitration.pool_ports",
                _ARBITRATION_POOL_FIELDS, errors)
        elif pool_ports is not None:
            errors.add("causal_prestate.arbitration.pool_ports")
        telemetry = arbitration.get("telemetry")
        if isinstance(telemetry, dict):
            _shape_mapping_fields(
                telemetry, "causal_prestate.arbitration.telemetry",
                _ARBITRATION_TELEMETRY_FIELDS, errors)
        elif telemetry is not None:
            errors.add("causal_prestate.arbitration.telemetry")

    history = value.get("history")
    if _shape_mapping_fields(
            history, "causal_prestate.history", _HISTORY_FIELDS, errors):
        for name, required in (
                ("length", _HISTORY_LENGTH_FIELDS),
                ("arrivals", _HISTORY_ARRIVAL_FIELDS),
                ("efficiency", _HISTORY_EFFICIENCY_FIELDS),
                ("service_samples", _HISTORY_SERVICE_FIELDS),
                ("horizon", _HISTORY_HORIZON_FIELDS),
                ("epoch_log", _HISTORY_EPOCH_FIELDS)):
            _shape_mapping_fields(
                history.get(name), f"causal_prestate.history.{name}",
                required, errors)
    versions = value.get("versions")
    if _shape_mapping_fields(
            versions, "causal_prestate.versions",
            _CAUSAL_PRESTATE_VERSION_FIELDS, errors):
        for name in ("kv_ledger_epoch", "directory_version", "length",
                     "arrivals", "efficiency"):
            if (name in versions and versions[name] is not None
                    and type(versions[name]) is not int):
                errors.add(f"causal_prestate.versions.{name}")
        stage_epochs = versions.get("stage_epochs")
        if isinstance(stage_epochs, list):
            for index, pair in enumerate(stage_epochs):
                if (not isinstance(pair, list) or len(pair) != 2
                        or not isinstance(pair[0], str)
                        or (pair[1] is not None and type(pair[1]) is not int)):
                    errors.add(f"causal_prestate.versions.stage_epochs[{index}]")
    return errors


def boundary_state_cert(record: dict, stage: str) -> dict:
    """目标边界决策行的状态摘要证书（基线与强制 run 同式计算）。

    只含**强制前状态字段**（不含 chosen 与选择派生观测量——2026-09-29
    受控运行实证：``boundary_wait_kind`` 由 ``f(selected, i_P)`` 计算、
    ``stage_work`` 携带所选动作的 rebuild 腿、D 行 ``seq`` 为发射时点序号
    且被选择依赖的旁行（如配额配对披露行）移位，三者均随强制分支
    变化，非同状态前提的判据——不入证书）；逐字段原值可比较，digest
    仅披露形态。必需字段缺失时不能签发证书，双方同缺也不构成证据。"""
    decision = record.get("decision") or {}
    if stage == "P":
        # P 行在决策时点即发射（无选择后旁行）——seq 为稳定的前缀观测量。
        fields = {
            "tick": record.get("tick"),
            "seq": record.get("seq"),
            "candidates_surface": _candidates_surface(
                extract_candidates(record, stage)),
            "service_factors": decision.get("service_factors"),
            "model_semantics_version": decision.get(
                "model_semantics_version"),
            "causal_prestate": decision.get("causal_prestate"),
        }
    else:
        boundary = decision.get("boundary_decision") or {}
        fields = {
            "tick": record.get("tick"),
            "candidates_surface": _candidates_surface(
                extract_candidates(record, stage)),
            "stage_epoch": boundary.get("stage_epoch"),
            "state_versions": boundary.get("state_versions"),
            "obligations": boundary.get("inherited_obligations"),
            "decode_steps": boundary.get("decode_steps"),
            "model_semantics_version": decision.get(
                "model_semantics_version"),
            "causal_prestate": boundary.get("causal_prestate"),
        }
    missing_fields = {name for name, value in fields.items() if value is None}
    for name in ("tick", "seq", "stage_epoch", "decode_steps"):
        if name in fields and not (type(fields[name]) is int and fields[name] >= 0):
            missing_fields.add(name)
    if not fields["candidates_surface"]:
        missing_fields.add("candidates_surface")
    if not isinstance(fields["model_semantics_version"], str) or not fields["model_semantics_version"]:
        missing_fields.add("model_semantics_version")
    missing_fields.update(_causal_prestate_shape_errors(
        fields["causal_prestate"], expected_stage=stage,
        expected_request_id=record.get("request_id")))
    if stage == "P":
        if not isinstance(fields["service_factors"], dict) or not fields["service_factors"]:
            missing_fields.add("service_factors")
    else:
        versions = fields["state_versions"]
        if not isinstance(versions, dict) or any(
                versions.get(name) is None for name in (
                    "kv_ledger_epoch", "directory_version", "stage_epochs")):
            missing_fields.add("state_versions")
    per_field_digest = {
        name: _digest16(value) for name, value in fields.items()}
    return {
        "stage": stage,
        "fields": fields,
        "field_digests": per_field_digest,
        "digest": _digest16(fields),
        "rng_basis": CERT_RNG_BASIS,
        "coverage": CERT_COVERAGE_NOTE,
        "missing_fields": sorted(missing_fields),
    }


def cert_mismatches(baseline_cert: dict, run_cert: dict) -> list[str]:
    """逐字段比较（原值比较，摘要仅披露）；返回不匹配字段名列表。"""
    base = baseline_cert["fields"]
    mine = run_cert["fields"]
    mismatches = []
    for name in sorted(set(baseline_cert.get("missing_fields", ())) |
                       set(run_cert.get("missing_fields", ()))):
        mismatches.append(f"missing:{name}")
    for name in sorted(set(base) | set(mine)):
        if base.get(name, "<absent>") != mine.get(name, "<absent>"):
            mismatches.append(name)
    return mismatches


# ---------------------------------------------------------------------------
# W8 service/settle 分采（service_done_ns 与 settle_done_ns 分列；
# 原 service_end_tick 的 max(completion, merge_done) 混称口径退役为
# legacy 回落——仅旧世代日志使用，source 如实披露）
# ---------------------------------------------------------------------------

def service_settle_ticks(run_dir: Path, request_id: str,
                         errors: list[str]) -> dict:
    """请求 service/settle 双终点分列采集（W5 settle 事件已就绪）。

    返回 {service_done_ns, service_source, settle_done_ns, settle_source,
    merge_done_ns, zero_payload_settle, settlement_policy,
    settle_missing_reason}：
    - service_done_ns：settle 事件 echo 的 service_done_ns → 回落
      completion 行 service_done_ns 键 → 回落 completion tick；
    - settle_done_ns：settle_done 事件；**consolidate 无 merge 载荷时
      settle==service 合法**（zero_payload_settle=True 显式记录）；无
      settle 事件的旧世代日志回落 max(completion, merge_done)（source=
      legacy_max_completion_merge——旧口径如实披露，不冒称 settle 真值）；
    - distributed（W9 后）：settlement_policy=distributed 而缺 settle 事件
      ⇒ **不得回退 completion=settle**，settle_missing_reason=
      settle_missing_policy_distributed（预留分支），settle_done_ns=None；
    - completion 缺失 ⇒ 双指标 None（missing，不以预测值填补）。
    """
    path = decision_log_path(run_dir)
    completion = None
    completion_service_key = None
    merge_done = None
    settle = None
    for record in _iter_log_records(path, errors):
        if record.get("request_id") != request_id:
            continue
        tick = record.get("tick")
        if type(tick) is not int or tick < 0:
            errors.append(
                f"{path}: kind={record.get('kind')!r} request {request_id!r} "
                f"tick missing/non-numeric (seq={record.get('seq')!r})")
            continue
        kind = record.get("kind")
        if kind == "completion":
            completion = tick if completion is None else max(completion, tick)
            service_key = (record.get("decision") or {}).get("service_done_ns")
            if type(service_key) is int and service_key >= 0:
                completion_service_key = service_key
        elif kind == "merge_done":
            merge_done = (tick if merge_done is None
                          else max(merge_done, tick))
        elif kind == "settle_done":
            settle = record.get("decision") or {}
    result = {
        "service_done_ns": None, "service_source": "missing",
        "settle_done_ns": None, "settle_source": "missing",
        "merge_done_ns": merge_done,
        "zero_payload_settle": False,
        "settlement_policy": None,
        "settle_missing_reason": None,
    }
    if completion is None:
        errors.append(
            f"{path}: no completion event for request {request_id!r} "
            f"(run 未跑完/请求未完成/日志缺事件)")
        return result
    # service：settle echo → completion 行显式键 → completion tick。
    if settle is not None and type(settle.get("service_done_ns")) is int:
        result["service_done_ns"] = settle["service_done_ns"]
        result["service_source"] = "settle_event_service_done_ns"
    elif completion_service_key is not None:
        result["service_done_ns"] = completion_service_key
        result["service_source"] = "completion_service_done_ns"
    else:
        result["service_done_ns"] = completion
        result["service_source"] = "completion_tick"
    manifest_path = Path(run_dir) / _BASELINE_MANIFEST_RELPATH
    manifest_prediction = None
    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as source:
                manifest = json.load(source)
            # Production online_service writes the prediction manifest under
            # this exact run-level path/key.  A legacy prediction_config key
            # is deliberately not accepted as a settlement certificate.
            prediction = manifest.get("prediction")
            if not isinstance(prediction, dict):
                errors.append(
                    f"{manifest_path}: missing manifest['prediction'] "
                    "settlement policy")
                result["settle_missing_reason"] = (
                    "settlement_policy_unavailable")
                return result
            manifest_prediction = prediction.get("settlement_policy")
            if manifest_prediction not in ("consolidate", "distributed"):
                errors.append(
                    f"{manifest_path}: manifest['prediction'] has no valid "
                    "settlement_policy")
                result["settle_missing_reason"] = (
                    "settlement_policy_unavailable")
                return result
        except (OSError, ValueError, AttributeError) as exc:
            errors.append(f"{manifest_path}: cannot determine settlement policy: {exc}")
            result["settle_missing_reason"] = "settlement_policy_unavailable"
            return result
    elif settle is not None:
        # A settle event without the real run manifest is insufficient to
        # establish which protocol was active; do not let a synthetic event
        # self-certify its own policy.
        errors.append(
            f"{manifest_path}: missing manifest['prediction'] for settle "
            f"event of request {request_id!r}")
        result["settle_missing_reason"] = "settlement_policy_unavailable"
        return result
    policy = manifest_prediction
    event_policy = (settle or {}).get("settlement_policy")
    if event_policy is not None and policy is not None and event_policy != policy:
        errors.append(
            f"{path}: settle event policy {event_policy!r} != "
            f"manifest prediction policy {policy!r}")
        result["settle_missing_reason"] = "settlement_policy_mismatch"
        return result
    result["settlement_policy"] = policy
    if settle is not None and type(settle.get("settle_done_ns")) is int:
        if (result["service_done_ns"] < 0 or
                settle["settle_done_ns"] < result["service_done_ns"]):
            errors.append(f"{path}: invalid service/settle ordering for {request_id!r}")
            result["settle_missing_reason"] = "invalid_settle_event"
            return result
        result["settle_done_ns"] = settle["settle_done_ns"]
        result["settle_source"] = "settle_event"
        payload = settle.get("merge_payload_bytes")
        result["zero_payload_settle"] = bool(
            (payload in (None, 0)) and settle.get("merge_done_ns") is None)
        if result["zero_payload_settle"]:
            result["settle_source"] = "settle_event_zero_payload"
    elif settle is not None:
        result["settle_missing_reason"] = "invalid_settle_event"
    elif policy in ("consolidate", "distributed"):
        # 新运行的两种策略都必须提供真实结算事件。
        result["settle_missing_reason"] = f"settle_missing_policy_{policy}"
    else:
        # 旧世代日志（W5 settle 事件之前）：回落旧混称口径并如实披露。
        result["settle_done_ns"] = max(
            completion, merge_done if merge_done is not None else completion)
        result["settle_source"] = "legacy_max_completion_merge"
    return result


# ---------------------------------------------------------------------------
# generate：请求×适用动作 -> 串行命令计划（不执行仿真）
# ---------------------------------------------------------------------------

def _sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)[:96] or "anon"


def _shell_quote(value) -> str:
    return "'" + str(value).replace("'", "'\"'\"'") + "'"


def _utc_run_id() -> str:
    return "probe_" + _dt.datetime.now(_dt.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")


# D17 世代核验（修复轮三 stageR7，2026-09-27）：baseline 的
# model_semantics_version 与当前代码比对——适用动作集/entry pin 语义随
# 世代漂移，旧代 baseline 生成的臂真跑时可能 fail-closed（实测：v3 代
# baseline 的 D__copy 臂在新代码下 no history to copy，probe_
# 20260927T213113Z），generate 期就 fail-early 披露；--strict-version
# 升级为拒绝生成。
_JOINT_CONFIG_PATH = (
    _REPO_ROOT / "sh_test_mesh" / "workload" / "llama2_7b_inference"
    / "joint" / "joint_config.py")
_BASELINE_MANIFEST_RELPATH = (
    Path("bridge") / "joint_mechanism_manifest.json")
_MODEL_SEMANTICS_VERSION_RE = re.compile(
    r'^MODEL_SEMANTICS_VERSION\s*=\s*"([^"]+)"', re.MULTILINE)


def current_model_semantics_version():
    """读当前代码的 MODEL_SEMANTICS_VERSION（joint_config.py 单一裁决点）。

    与 ALL_ACTIONS 同纪律：run_scripts 工具不 import workload 代码（保持
    独立可运行），以源码常量正则解析代之；文件缺失/解析失败返回 None
    （核验跳过并告警，不猜值）。
    """
    try:
        text = _JOINT_CONFIG_PATH.read_text(encoding="utf-8")
    except OSError:
        return None
    match = _MODEL_SEMANTICS_VERSION_RE.search(text)
    return match.group(1) if match else None


def baseline_model_semantics_version(baseline_run: Path):
    """读 baseline run 的 model_semantics_version 披露（防御式）。

    优先 bridge/joint_mechanism_manifest.json（固定配置 manifest，
    joint_config.manifest_dict 落盘）→ 回落决策行 model_semantics_version
    键（P 准入/边界行逐行披露，D17 schema）。都缺失返回 (None, "missing")。
    """
    manifest_path = Path(baseline_run) / _BASELINE_MANIFEST_RELPATH
    if manifest_path.is_file():
        try:
            with open(manifest_path, encoding="utf-8") as source:
                manifest = json.load(source)
        except (OSError, ValueError):
            manifest = None
        if isinstance(manifest, dict):
            value = manifest.get("model_semantics_version")
            if isinstance(value, str) and value:
                return value, "bridge/joint_mechanism_manifest.json"
    log_path = decision_log_path(baseline_run)
    if log_path.is_file():
        for record in _iter_log_records(log_path, []):
            decision = record.get("decision")
            holders = [record] + ([decision]
                                  if isinstance(decision, dict) else [])
            for holder in holders:
                value = holder.get("model_semantics_version")
                if isinstance(value, str) and value:
                    return value, "decision_log"
    return None, "missing"


def _check_baseline_generation(baseline_run: Path, strict: bool) -> dict:
    """generate 期世代核验（fail-early）：baseline 世代 vs 当前代码。

    不一致 → stderr 明确告警但照常生成（不阻断——旧代 baseline 仍可用于
    粗对照，由使用者判断）；``--strict-version`` 时升级为 ProbeError
    （fail-closed 拒绝生成）。无法核验（baseline 版本披露缺失/当前常量
    解析失败）同口径：告警，strict 下拒绝。返回披露 dict（写入 plan 的
    ``baseline_generation`` 键，provenance 面如实落盘）。
    """
    current = current_model_semantics_version()
    baseline, source = baseline_model_semantics_version(baseline_run)
    strict_note = "（--strict-version）" if strict else ""
    if current is None:
        if strict:
            raise ProbeError(
                f"cannot verify generation: MODEL_SEMANTICS_VERSION not "
                f"parseable from {_JOINT_CONFIG_PATH} (--strict-version "
                "拒绝生成)")
        print(f"[probe][warn] 无法从 {_JOINT_CONFIG_PATH} 解析当前 "
              f"MODEL_SEMANTICS_VERSION——baseline 世代核验跳过"
              f"{strict_note}（baseline 披露={baseline!r} source={source}）",
              file=sys.stderr)
    elif baseline is None:
        if strict:
            raise ProbeError(
                f"baseline run has no model_semantics_version disclosure "
                f"(source={source})——无法核验世代 (--strict-version 拒绝"
                "生成)")
        print(f"[probe][warn] baseline 无 model_semantics_version 披露"
              f"（source={source}）——无法核验世代，适用面可能漂移；建议"
              f"用同代 baseline{strict_note}", file=sys.stderr)
    elif baseline != current:
        if strict:
            raise ProbeError(
                f"baseline 世代 {baseline} ≠ 当前 {current}——适用面可能"
                "漂移，臂可能 fail-closed；建议用同代 baseline "
                "(--strict-version 拒绝生成)")
        print(f"[probe][warn] baseline 世代 {baseline} ≠ 当前 {current}"
              "——适用面可能漂移，臂可能 fail-closed；建议用同代 baseline",
              file=sys.stderr)
    return {
        "current": current,
        "baseline": baseline,
        "baseline_source": source,
        "match": bool(current is not None and baseline is not None
                      and current == baseline),
        "strict": bool(strict),
    }


def _scan_baseline_faces(baseline_run: Path):
    """一次扫描基线决策日志，逐请求抽 P/D 两面边界记录摘要。

    返回 (per_request, order)：per_request[rid] = {"faces": {stage: 摘要},
    "dup": {stage: n}}；order = 首见顺序（自动选目标的确定性遍历序）。
    """
    path = decision_log_path(baseline_run)
    if not path.is_file():
        raise ProbeError(f"baseline decision log missing: {path}")
    errors: list[str] = []
    per_request: dict[str, dict] = {}
    order: list[str] = []
    for record in _iter_log_records(path, errors):
        kind = record.get("kind")
        decision = record.get("decision")
        if not isinstance(decision, dict):
            continue
        rid = record.get("request_id")
        if not isinstance(rid, str) or not rid:
            continue
        if kind == "joint_admission":
            stage = "P"
        elif (kind == "decode"
                and isinstance(decision.get("boundary_decision"), dict)):
            stage = "D"
        else:
            continue
        info = per_request.setdefault(rid, {
            "request_id": rid, "faces": {}, "dup": {"P": 0, "D": 0}})
        if rid not in order:
            order.append(rid)
        if stage in info["faces"]:  # 多次边界（重入场等）：取首见，其余记 dup
            info["dup"][stage] += 1
            continue
        try:
            candidates = extract_candidates(record, stage)
        except ProbeError as exc:
            errors.append(f"baseline {rid}/{stage}: {exc}")
            continue
        # W8 双维：逐 (instance, action) 候选全量采集（动作级回卷仅供
        # 旧 --granularity action 展示兼容层）。
        surface = aggregate_by_instance_action(candidates)
        actions = rollup_actions(surface)
        applicable = [
            a for a in ALL_ACTIONS if actions.get(a, {}).get("applicable")]
        applicable += sorted(
            a for a in actions if a not in ALL_ACTIONS
            and actions[a].get("applicable"))
        instance_actions = sorted(
            (inst, action) for (inst, action), slot in surface.items()
            if slot["applicable"])
        face_info = {
            "tick": record.get("tick"),
            "seq": record.get("seq"),
            "actions": actions,
            "applicable": applicable,
            "instance_actions": [
                list(pair) for pair in instance_actions],
            "surface": {
                candidate_key(inst, action): {
                    "predicted_ns": slot["predicted_ns"],
                    "is_continuation": slot["is_continuation"]}
                for (inst, action), slot in surface.items()},
            "selected": selected_action_of(record, stage),
            "selected_instance": selected_instance_of(record, stage),
        }
        if stage == "P":
            # W8（阶段C.5）：P 行披露的默认延续形态 = P 估价内的"参考 D"
            #（decode_instance/decode_action）——与 D 边界实际选择分列。
            face_info["reference_continuation"] = {
                "instance": decision.get("decode_instance"),
                "action": decision.get("decode_action"),
            }
        info["faces"][stage] = face_info
    if errors:
        for line in errors[:5]:
            print(f"[probe][warn] {line}", file=sys.stderr)
    return per_request, order


def _face_candidate_list(face: dict, granularity: str) -> list[dict]:
    """臂枚举的候选清单（W8 --granularity）。

    - ``action``（旧口径）：逐**动作**一臂（多实例时调度器按动作重选实
      例——弃用方向，仅向后兼容旧 plan/脚本）；
    - ``instance-action``：逐**适用 (instance, action)** 候选各一臂
      （TARGET env 精确强制——全部候选的反事实真值，方案 §11.3）。
    """
    if granularity == "instance-action":
        out = []
        for inst, action in face["instance_actions"]:
            key = candidate_key(inst, action)
            out.append({
                "instance_index": inst,
                "action": action,
                "key": key,
                "predicted_ns": face["surface"][key]["predicted_ns"],
            })
        return out
    return [{
        "instance_index": None,
        "action": action,
        "key": action,
        "predicted_ns": face["actions"][action]["predicted_ns"],
    } for action in face["applicable"]]


def _face_requirement(arm_families: list[str]) -> dict:
    """所选臂族对各面的需求：entry 枚举 P 面（>=2 适用动作）；boundary
    枚举 D 面（>=2）且需 P 面在位（entry pin 取基线 P 选择，动作数不设
    下限——单动作面同样可钉）。"""
    need = {"present": {"P": False, "D": False},
            "min_actions": {}}
    if "entry" in arm_families:
        need["present"]["P"] = True
        need["min_actions"]["P"] = 2
    if "boundary" in arm_families:
        need["present"]["P"] = True
        need["present"]["D"] = True
        need["min_actions"]["D"] = 2
    return need


def _face_gap_reasons(info: dict, need: dict) -> list[str]:
    """该请求对所选臂族的面缺口描述（空列表 = 合格）。"""
    reasons = []
    for stage in ("P", "D"):
        face = info["faces"].get(stage)
        if need["present"][stage] or stage in need["min_actions"]:
            if face is None:
                reasons.append(f"{stage} 边界记录缺失")
            elif (stage in need["min_actions"]
                    and len(face["applicable"])
                    < need["min_actions"][stage]):
                reasons.append(
                    f"{stage} 面适用动作 {len(face['applicable'])} < "
                    f"{need['min_actions'][stage]}")
    return reasons


def _pick_targets(baseline_run: Path, requested: list[str],
                  max_requests: int, arm_families: list[str]):
    """从基线决策日志选对照目标：显式 --request 优先；否则自动选。

    自动选法（确定性，供 harnessCommands 复述）：按日志顺序取**所选臂族
    面合格**的请求（entry：P 面 >=2 适用动作；boundary：D 面 >=2 且 P 面
    在位；both：两者同时），跳过 (P 适用集, D 适用集) 与已选重复者，最多
    max_requests 个——排序对照在 >=2 适用动作时才有意义，集合去重保证
    动作面/状态面多样。
    """
    per_request, order = _scan_baseline_faces(baseline_run)
    need = _face_requirement(arm_families)

    def gap_reasons(info):
        return _face_gap_reasons(info, need)

    targets = []
    if requested:
        missing = [r for r in requested if r not in per_request]
        if missing:
            raise ProbeError(
                f"--request not found in baseline boundary records: "
                f"{missing} (available: {order[:20]}"
                f"{'...' if len(order) > 20 else ''})")
        for rid in requested:
            reasons = gap_reasons(per_request[rid])
            if reasons:
                raise ProbeError(
                    f"--request {rid!r} lacks faces for arm families "
                    f"{arm_families}: {reasons}")
            targets.append(per_request[rid])
    else:
        selected_ids = []
        seen_keys = set()
        skipped = []
        for rid in order:
            info = per_request[rid]
            reasons = gap_reasons(info)
            if reasons:
                skipped.append(f"{rid}({';'.join(reasons)})")
                continue
            p = info["faces"].get("P")
            d = info["faces"].get("D")
            key_p = tuple(p["applicable"]) if p else None
            key_d = tuple(d["applicable"]) if d else None
            key = (key_p, key_d) if "boundary" in arm_families else (key_p,)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            selected_ids.append(rid)
            if len(selected_ids) >= max_requests:
                break
        if skipped:
            print(f"[probe][warn] auto-selection skipped (face gaps): "
                  f"{'; '.join(skipped[:8])}"
                  f"{'...' if len(skipped) > 8 else ''}", file=sys.stderr)
        if not selected_ids:
            raise ProbeError(
                "auto-selection found no request with the required faces "
                f"for arm families {arm_families} "
                f"(need: present={need['present']}, "
                f"min_actions={need['min_actions']}) in "
                f"{decision_log_path(baseline_run)}")
        targets = [per_request[rid] for rid in selected_ids]
    return targets, order


def _render_commands_sh(plan: dict) -> str:
    """按 joint_smoke_matrix.sh 既有模式渲染串行执行脚本（物化+指针+逐臂）。"""
    project = _shell_quote(plan["project_root"])
    lines = [
        "#!/usr/bin/env bash",
        "# auto-generated by joint_same_state_probe.py (C7b-2) -- 同状态对照",
        "# probe 串行仿真脚本：同一 trace 同参数，只换强制动作 env。",
        "# 仿 joint_smoke_matrix.sh 的物化/指针切换/逐臂证据链模式；",
        "# 本脚本由主控执行（工具本身不执行仿真）。",
        "set -euo pipefail",
        "",
        f"PROJECT={project}",
        'SCRIPT_DIR="${PROJECT}/sh_test_mesh/run_scripts"',
        f"EVIDENCE_ROOT={_shell_quote(plan['evidence_root'])}",
        f"RUN_ID={_shell_quote(plan['run_id'])}",
        f"WINDOW_NS={_shell_quote(plan['window_ns'])}",
        f"SRC_CSV={_shell_quote(plan['source_csv'])}",
        "",
        'if [[ ! -f "${SRC_CSV}" ]]; then',
        '  echo "[probe] source csv missing: ${SRC_CSV}" >&2; exit 1; fi',
        'BIN="${PROJECT}/build/astra_analytical/build_congestion_aware/bin/'
        'AstraSim_Analytical_Congestion_Aware_Online"',
        'if [[ ! -x "${BIN}" ]]; then',
        '  echo "[probe] binary missing: ${BIN}（先按 README §4 构建）" >&2; '
        'exit 1; fi',
        "",
        "# —— 仓级共享仿真锁：先锁再物化/清 generated/切指针（同冒烟矩阵；",
        "# 内层 runner 经 SH_SINGLE_SIMULATION_LOCK_FD 复用同一 OFD 不自锁，",
        "# 避免逐臂独立抢锁被前臂后处理采样器的残留 fd 挡住）——",
        'SINGLE_SIMULATION_LOCK_PATH="${PROJECT}/sh_test_mesh/runs/'
        '.single_simulation.lock"',
        'mkdir -p "${PROJECT}/sh_test_mesh/runs"',
        'exec {PROBE_SIM_LOCK_FD}>>"${SINGLE_SIMULATION_LOCK_PATH}"',
        'if ! flock -n "${PROBE_SIM_LOCK_FD}"; then',
        '  exec {PROBE_SIM_LOCK_FD}>&-',
        '  echo "[probe] another simulation holds '
        '${SINGLE_SIMULATION_LOCK_PATH}; refusing to start" >&2',
        '  exit 1',
        'fi',
        'export SH_SINGLE_SIMULATION_LOCK_FD="${PROBE_SIM_LOCK_FD}"',
        "",
        "# —— 输入物化 + trace_config 指针切换（同冒烟矩阵；EXIT 还原）——",
        'MATERIALIZED="${EVIDENCE_ROOT}/probe_input"',
        'mkdir -p "${MATERIALIZED}"',
        'python3 "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/'
        'traces/materialize_20_30s.py" \\\n'
        '  "${SRC_CSV}" "${MATERIALIZED}/requests.csv" "${WINDOW_NS}"',
        'REQUEST_CSV=$(realpath "${MATERIALIZED}/requests.csv")',
        'rm -rf "${PROJECT}"/sh_test_mesh/generated/'
        'llama2_7b_inference_54npus_plan_* 2>/dev/null || true',
        'TRACE_CONFIG="${PROJECT}/sh_test_mesh/workload/llama2_7b_inference/'
        'trace_config.csv"',
        "",
        "read_pointer() {",
        "  python3 - \"${TRACE_CONFIG}\" <<'PYEOF'",
        "import sys",
        "for line in open(sys.argv[1], encoding=\"utf-8\"):",
        "    fields = line.rstrip(\"\\n\").split(\",\")",
        "    if len(fields) > 2 and fields[1] == \"request_queue_csv\":",
        "        print(fields[2])",
        "        break",
        "PYEOF",
        "}",
        "set_pointer() {  # set_pointer <值>",
        "  python3 - \"${TRACE_CONFIG}\" \"$1\" <<'PYEOF'",
        "import sys",
        "path, value = sys.argv[1], sys.argv[2]",
        "lines = open(path, encoding=\"utf-8\").read().splitlines("
        "keepends=True)",
        "out = []",
        "for line in lines:",
        "    fields = line.rstrip(\"\\n\").split(\",\")",
        "    if len(fields) > 2 and fields[1] == \"request_queue_csv\":",
        "        fields[2] = value",
        "        line = \",\".join(fields) + \"\\n\"",
        "    out.append(line)",
        "open(path, \"w\", encoding=\"utf-8\").writelines(out)",
        "PYEOF",
        "}",
        'POINTER_ORIGINAL=$(read_pointer)',
        "restore_pointer() { set_pointer \"${POINTER_ORIGINAL}\"; }",
        "trap restore_pointer EXIT",
        'set_pointer "${REQUEST_CSV}"',
        '( cd "${PROJECT}/sh_test_mesh/workload/llama2_7b_inference" \\',
        '    && python3 plan_materializer.py )',
        "",
        "# —— 逐臂串行执行（同 run_one 证据链：日志先落外再移入/退出码/env）——",
        "run_probe_arm() {  # run_probe_arm <run_dir> [ENV=VALUE ...]",
        '  local run_dir=$1; shift',
        '  local runner_log="${run_dir}.runner.log"',
        '  rm -rf "${run_dir}" "${runner_log}"; mkdir -p "${run_dir}"',
        "  local rc=0",
        '  env "$@" bash "${SCRIPT_DIR}/run_online_strategy.sh" \\\n'
        '    "${run_dir}" "${REQUEST_CSV}" > "${runner_log}" 2>&1 || rc=$?',
        '  mv "${runner_log}" "${run_dir}/run.log"',
        '  env "$@" env > "${run_dir}/env.txt" 2>&1 || true',
        '  echo "${rc}" > "${run_dir}/exit_code"',
        '  if [[ ${rc} -eq 0 ]]; then',
        '    echo "[probe] $(basename "${run_dir}"): PASS"',
        "  else",
        '    echo "[probe] $(basename "${run_dir}"): FAIL (exit ${rc})'
        '——tail run.log:"',
        '    tail -5 "${run_dir}/run.log" || true',
        "  fi",
        "}",
        "",
    ]
    for target in plan["targets"]:
        for arm in _plan_arm_list(plan, target):
            lines.append(
                f"# target request {target['request_id']} "
                f"arm={arm.get('arm_family', 'entry')} "
                f"(stage={arm.get('stage', plan.get('stage', 'P'))}"
                + (f", entry_pin={arm.get('entry_pin')}"
                   if arm.get("entry_pin") is not None else "")
                + f", boundary_tick={_arm_boundary_tick(plan, target, arm)})")
            for run in arm.get("runs", []):
                env_pairs = " ".join(
                    _shell_quote(f"{k}={v}") for k, v in run["env"].items())
                lines.append(
                    f'run_probe_arm "{run["run_dir"]}" \\\n  {env_pairs}')
            lines.append("")
    one_shot_pair = plan.get("one_shot_pair")
    if one_shot_pair:
        lines.append(
            "# 一次映射对照臂（成对两 run，验收纪律第 2 类『一次映射 vs "
            "两阶段闭环对照』）：同 trace 同参数，仅差 "
            f"{ONESHOT_POLICY_ENV}——两阶段参照不设该 env，一次映射臂 "
            f"{ONESHOT_POLICY_ENV}={one_shot_pair['oneshot_run']['policy']}；"
            "无逐请求强制钩子")
        for key in ("twophase_run", "oneshot_run"):
            run = one_shot_pair[key]
            env_pairs = " ".join(
                _shell_quote(f"{k}={v}") for k, v in run["env"].items())
            suffix = " \\\n  " + env_pairs if env_pairs else ""
            lines.append(
                f'run_probe_arm "{run["run_dir"]}"{suffix}')
        lines.append("")
    replay = plan.get("replay_arm")
    if replay:
        lines.append(
            "# 固定动作回放臂（宿主开销对照，v2.0 §19.3 待验证预算）：无强"
            f"制钩子，{REPLAY_MODE_ENV}={replay['mode']}")
        env_pairs = " ".join(
            _shell_quote(f"{k}={v}") for k, v in replay["env"].items())
        lines.append(
            f'run_probe_arm "{replay["run_dir"]}" \\\n  {env_pairs}')
        lines.append("")
    lines += [
        'echo "[probe] all candidate runs done; aggregate with:"',
        'echo "[probe]   python3 ${SCRIPT_DIR}/joint_same_state_probe.py '
        'summarize --plan '
        f"{_shell_quote(plan['plan_path'])}\"",
        "rc_all=0",
        'for f in "${EVIDENCE_ROOT}/${RUN_ID}"/runs/*/exit_code; do',
        '  code=$(cat "${f}")',
        '  printf \'  %-60s exit=%s\\n\' "$(basename "$(dirname "${f}")")" '
        '"${code}"',
        '  [[ ${code} -ne 0 ]] && rc_all=1',
        "done",
        "exit ${rc_all}",
        "",
    ]
    return "\n".join(lines)


def _render_plan_md(plan: dict) -> str:
    lines = [
        "# 同状态对照 probe 计划（式(24) 选择损失 harness）",
        "",
        f"- run_id: `{plan['run_id']}`",
        f"- baseline_run: `{plan['baseline_run']}`",
        f"- arm_families: {plan.get('arm_families')}"
        "（entry=P 入口枚举；boundary=固定 P 动作 + D 边界枚举）",
        f"- granularity: {plan.get('granularity')}"
        "（action=旧口径逐动作臂；instance-action=逐适用 (instance,action)"
        " 候选 TARGET 精确强制——W8 双维反事实真值）",
        f"- window_ns: {plan['window_ns']}（2s 冒烟纪律最小窗）",
        f"- source_csv: `{plan['source_csv']}`",
        f"- 钩子契约: `{FORCE_REQUEST_ENV}` + `{FORCE_ENTRY_ACTION_ENV}`"
        f"（P 入口）或 `{FORCE_ACTION_ENV}`（P→D 边界）——两 ACTION env "
        "互斥武装（同设 = 调度器启动失败），单臂只设其一；W8 精确身份："
        f"`{FORCE_ENTRY_TARGET_ENV}` / `{FORCE_BOUNDARY_TARGET_ENV}` = "
        "<instance>:<action>（与同族 ACTION 同设 = fail-closed；ACTION 对"
        "为弃用方向）。缺省 off，只强制指定请求的对应边界决策（测量仪器"
        "语义）",
        f"- 附加 env（各臂透传）: {json.dumps(plan['extra_env'])}",
        f"- 总命令数: {plan['total_commands']}"
        + (f"（另 +1 回放臂）" if plan.get("replay_arm") else ""),
        "",
        "| request | stage | boundary_tick | baseline_selected | "
        "applicable | instance_action_candidates |",
        "|---|---|---|---|---|---|",
    ]
    for target in plan["targets"]:
        for stage in ("P", "D"):
            face = (target.get("boundaries") or {}).get(stage)
            if face:
                pairs = face.get("instance_actions") or []
                lines.append(
                    f"| {target['request_id']} | {stage} | "
                    f"{face['tick']} | {face['selected']} | "
                    f"{','.join(face['applicable'])} | "
                    f"{','.join('i{}:{}'.format(i, a) for i, a in pairs)}"
                    " |")
    lines += ["", "## 命令（逐条串行，同 trace 同参数只换强制动作）", ""]
    for target in plan["targets"]:
        for arm in _plan_arm_list(plan, target):
            pin_note = (f"（entry pin={arm.get('entry_pin')}）"
                        if arm.get("entry_pin") is not None else "")
            lines.append(f"### {target['request_id']} — "
                         f"{arm.get('arm_family', 'entry')} 臂"
                         f"（stage={arm.get('stage')}）{pin_note}")
            for run in arm.get("runs", []):
                env_txt = " ".join(
                    f"{k}={v}" for k, v in run["env"].items())
                lines.append("```bash")
                lines.append(f"env {env_txt} \\")
                lines.append(f"  bash {_RUNNER_SH} \\")
                lines.append(
                    f'    "{run["run_dir"]}" "${{REQUEST_CSV}}"')
                lines.append("```")
    one_shot_pair = plan.get("one_shot_pair")
    if one_shot_pair:
        lines += [
            "",
            "### 一次映射对照臂（成对两 run；验收纪律第 2 类『一次映射 vs "
            "两阶段闭环对照』）",
            "",
            f"- 用途: {one_shot_pair['purpose']}",
        ]
        for key in ("twophase_run", "oneshot_run"):
            run = one_shot_pair[key]
            env_txt = " ".join(
                f"{k}={v}" for k, v in run["env"].items())
            lines += [
                f"- {run['label']}:",
                "```bash",
                f"env {env_txt} \\" if env_txt else "env \\",
                f"  bash {_RUNNER_SH} \"{run['run_dir']}\" "
                "\"${REQUEST_CSV}\"",
                "```",
            ]
    replay = plan.get("replay_arm")
    if replay:
        lines += [
            "",
            "### 固定动作回放臂（宿主开销对照；无强制钩子）",
            "",
            "```bash",
            "env " + " ".join(
                f"{k}={v}" for k, v in replay["env"].items()) + " \\",
            f'  bash {_RUNNER_SH} "{replay["run_dir"]}" "${{REQUEST_CSV}}"',
            "```",
            "",
            f"- 剔除的互斥/冲突 env: {replay['dropped_env']}",
            f"- 用途: {replay['purpose']}",
        ]
    lines += [
        "",
        "完整物化/指针切换/串行执行见 `probe_commands.sh`（工具不执行仿真）。",
        "",
    ]
    return "\n".join(lines)


def cmd_generate(args) -> int:
    baseline_run = Path(args.baseline_run).resolve()
    # D17 世代核验（fail-early，先于任何面扫描/目录创建）：baseline 与
    # 当前代码的 MODEL_SEMANTICS_VERSION 比对——不一致缺省 stderr 告警
    # 照常生成（旧代 baseline 仍可用于粗对照，由使用者判断），
    # --strict-version 直接 fail-closed 拒绝生成（详
    # _check_baseline_generation）。
    generation = _check_baseline_generation(baseline_run,
                                            args.strict_version)
    # 臂族解析：--stage 为旧接口（P→entry / D→boundary，后者语义升级为带
    # entry pin 的同状态对照）；缺省走 --arms（both = entry + boundary）；
    # one-shot = 一次映射对照臂族（成对两 run，无逐请求强制钩子）。
    if args.stage is not None:
        arm_families = ["entry"] if args.stage == "P" else ["boundary"]
        print(f"[probe][deprecated] --stage {args.stage} 映射为 --arms "
              f"{arm_families[0]}（boundary 臂现已带 entry pin，同状态语"
              "义升级）", file=sys.stderr)
    elif args.arms == "both":
        arm_families = ["entry", "boundary"]
    elif args.arms == "one-shot":
        arm_families = ["one-shot"]
    else:
        arm_families = [args.arms]

    if "one-shot" in arm_families:
        # 一次映射对照臂族：run 级策略（非逐请求钩子）——无需选点目标，
        # 生成**成对**两 run（同 trace 同参数，仅差 JOINT_BOUNDARY_POLICY）。
        targets, available = [], []
    else:
        targets, available = _pick_targets(
            baseline_run, args.request, args.max_requests, arm_families)

    evidence_root = Path(args.evidence_root).resolve()
    run_id = args.run_id or _utc_run_id()
    out_dir = evidence_root / run_id
    plan_path = out_dir / "probe_plan.json"

    extra_env = {}
    for entry in args.env or []:
        name, sep, value = entry.partition("=")
        if not name or not sep:
            raise ProbeError(f"--env expects NAME=VALUE, got {entry!r}")
        extra_env[name] = value

    plan = {
        "tool": "joint_same_state_probe.py",
        "hook_contract": HOOK_CONTRACT_NOTE,
        "run_id": run_id,
        "created_utc": _dt.datetime.now(_dt.timezone.utc).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "baseline_run": str(baseline_run),
        "baseline_generation": generation,
        "arm_families": list(arm_families),
        "granularity": args.granularity,
        # stage 字段保留为信息性兼容键（单族=该面；双族/一次映射族
        # ="P+D"）。
        "stage": ({"entry": "P", "boundary": "D",
                   "one-shot": "P+D"}[arm_families[0]]
                  if len(arm_families) == 1 else "P+D"),
        "window_ns": args.window_ns,
        "source_csv": str(Path(args.source_csv).resolve()),
        "evidence_root": str(evidence_root),
        "project_root": str(_REPO_ROOT),
        "extra_env": extra_env,
        "available_requests": available,
        "targets": [],
        "plan_path": str(plan_path),
    }

    total_commands = 0
    for target in targets:
        rid = target["request_id"]
        faces = target["faces"]
        p_face = faces.get("P")
        d_face = faces.get("D")
        arms = []
        if "entry" in arm_families:
            runs = []
            for cand in _face_candidate_list(p_face, args.granularity):
                # entry 臂：只武装 P 入口钩子——D 边界照常自由决策。
                # granularity=instance-action 用 TARGET 精确强制；action
                # 用旧 per-action env（弃用方向，向后兼容）。
                if cand["instance_index"] is None:
                    run_env = {FORCE_REQUEST_ENV: rid,
                               FORCE_ENTRY_ACTION_ENV: cand["action"]}
                    run_name = "{}__{}__{}".format(
                        _sanitize(rid), "P", _sanitize(cand["action"]))
                else:
                    run_env = {
                        FORCE_REQUEST_ENV: rid,
                        FORCE_ENTRY_TARGET_ENV: "{}:{}".format(
                            cand["instance_index"], cand["action"])}
                    run_name = "{}__{}__{}".format(
                        _sanitize(rid), "P",
                        "i{}__{}".format(cand["instance_index"],
                                         _sanitize(cand["action"])))
                run_env.update(extra_env)
                runs.append({
                    "action": cand["action"],
                    "instance_index": cand["instance_index"],
                    "key": cand["key"],
                    "run_dir": str(out_dir / "runs" / run_name),
                    "env": run_env,
                    "predicted_ns": cand["predicted_ns"],
                })
            arms.append({
                "arm_family": "entry",
                "stage": "P",
                "granularity": args.granularity,
                "arm_class": "same_state_forced_branch",
                "baseline_selected": p_face["selected"],
                "predictions": {run["key"]: run["predicted_ns"]
                                for run in runs},
                "runs": runs,
            })
        if "boundary" in arm_families:
            pin = p_face["selected"]
            if not isinstance(pin, str) or not pin:
                raise ProbeError(
                    f"request {rid!r}: baseline P-boundary selected action "
                    f"missing ({pin!r}) — 无法钉 entry pin（boundary 臂的"
                    "同状态前提），拒绝非受控生成")
            runs = []
            for cand in _face_candidate_list(d_face, args.granularity):
                # boundary 臂：只武装边界钩子（互斥合同禁止与 ENTRY 同
                # 设）；P 面同状态前提由基线 P 选择承载为 entry_pin。
                if cand["instance_index"] is None:
                    run_env = {FORCE_REQUEST_ENV: rid,
                               FORCE_ACTION_ENV: cand["action"]}
                    run_name = "{}__{}__{}".format(
                        _sanitize(rid), "D", _sanitize(cand["action"]))
                else:
                    run_env = {
                        FORCE_REQUEST_ENV: rid,
                        FORCE_BOUNDARY_TARGET_ENV: "{}:{}".format(
                            cand["instance_index"], cand["action"])}
                    run_name = "{}__{}__{}".format(
                        _sanitize(rid), "D",
                        "i{}__{}".format(cand["instance_index"],
                                         _sanitize(cand["action"])))
                run_env.update(extra_env)
                runs.append({
                    "action": cand["action"],
                    "instance_index": cand["instance_index"],
                    "key": cand["key"],
                    "run_dir": str(out_dir / "runs" / run_name),
                    "env": run_env,
                    "predicted_ns": cand["predicted_ns"],
                })
            arms.append({
                "arm_family": "boundary",
                "stage": "D",
                "granularity": args.granularity,
                "arm_class": "same_state_forced_branch",
                "entry_pin": pin,
                "entry_pin_instance": p_face.get("selected_instance"),
                "baseline_selected": d_face["selected"],
                "predictions": {run["key"]: run["predicted_ns"]
                                for run in runs},
                "runs": runs,
            })
        boundaries = {}
        for stage, face in (("P", p_face), ("D", d_face)):
            boundaries[stage] = None if face is None else {
                "tick": face["tick"],
                "seq": face["seq"],
                "selected": face["selected"],
                "selected_instance": face.get("selected_instance"),
                "applicable": list(face["applicable"]),
                "instance_actions": [
                    list(pair) for pair in face.get("instance_actions", [])],
                # 双维预测面（键 = "i<inst>:<action>"；值 = 预测 ns，不适
                # 用候选为 None——summarize 按数值过滤）。
                "predictions": {
                    key: slot["predicted_ns"]
                    for key, slot in face.get("surface", {}).items()},
                "continuations": sorted(
                    key for key, slot in face.get("surface", {}).items()
                    if slot.get("is_continuation")),
                **({"reference_continuation": face["reference_continuation"]}
                   if stage == "P" and "reference_continuation" in face
                   else {}),
            }
        plan["targets"].append({
            "request_id": rid,
            "boundaries": boundaries,
            # 旧字段别名（P 面；兼容既有读者/旧版 summarize 形态）——键
            # 保持动作名（与 applicable 别名配对；双维面见
            # boundaries[P].predictions / instance_actions）。
            "boundary_tick": boundaries["P"]["tick"] if p_face else None,
            "boundary_seq": boundaries["P"]["seq"] if p_face else None,
            "baseline_selected": (p_face["selected"] if p_face else None),
            "applicable": list(p_face["applicable"]) if p_face else [],
            "predictions": (
                {a: p_face["actions"][a]["predicted_ns"]
                 for a in p_face["applicable"]} if p_face else {}),
            "arms": arms,
        })
        total_commands += sum(len(arm["runs"]) for arm in arms)

    # 一次映射对照臂（验收纪律第 2 类"一次映射 vs 两阶段闭环对照"）：
    # 成对两 run——两阶段参照（不设 JOINT_BOUNDARY_POLICY）× 一次映射
    # （p-continuation）。同 trace 同参数仅差该 env；无逐请求强制钩子
    # （run 级策略）。与强制钩子互斥是调度器侧合同（同设 fail-closed
    # 启动失败），故本臂族不注入任何 JOINT_FORCE_* env。
    one_shot_pair = None
    if "one-shot" in arm_families:
        oneshot_env = dict(extra_env)
        oneshot_env[ONESHOT_POLICY_ENV] = ONESHOT_POLICY_VALUE
        twophase_env = dict(extra_env)
        one_shot_pair = {
            "family": "one-shot",
            # W8（阶段C.4）：固定策略 whole-run 臂——不与同状态数据混算。
            "arm_class": "oneshot_policy_whole_run",
            "twophase_run": {
                "label": "twophase_reference",
                "run_dir": str(out_dir / "runs" / "twophase_reference"),
                "env": twophase_env,
            },
            "oneshot_run": {
                "label": "oneshot_p_continuation",
                "run_dir": str(out_dir / "runs" / "oneshot_p_continuation"),
                "env": oneshot_env,
                "policy": ONESHOT_POLICY_VALUE,
            },
            "purpose": (
                "一次映射 vs 两阶段闭环对照（验收纪律第 2 类）：两 run 同 "
                "trace 同参数，仅差 JOINT_BOUNDARY_POLICY——两阶段参照不"
                "设该 env（D 边界恰好一次全候选重估），一次映射臂 "
                f"{ONESHOT_POLICY_ENV}={ONESHOT_POLICY_VALUE}（D 边界直接"
                "提交 P 准入默认延续形态，测量仪器仅供对照实验）"),
        }
    plan["one_shot_pair"] = one_shot_pair

    # 固定动作回放臂（宿主开销对照）：无强制钩子；load-first 与
    # JOINT_ABLATION_COMBO 互斥（joint_config fail-closed），从透传 env
    # 剔除互斥键并在 plan 披露。
    replay_arm = None
    if args.replay_arm == "load-first":
        dropped = sorted(
            key for key in extra_env
            if key in (REPLAY_MODE_ENV, _COMBO_ENV))
        replay_env = {REPLAY_MODE_ENV: REPLAY_MODE_VALUE}
        replay_env.update(
            {k: v for k, v in extra_env.items() if k not in dropped})
        replay_arm = {
            "family": "replay",
            "mode": REPLAY_MODE_VALUE,
            # W8（阶段C.4）：固定动作 whole-run 臂（fixed_policy_arm）——
            # 消融语义，独立标记，不与同状态 forced 分支数据混算。
            "arm_class": "fixed_policy_whole_run",
            "run_dir": str(out_dir / "runs" / "replay_load_first"),
            "env": replay_env,
            "dropped_env": dropped,
            "purpose": (
                "宿主开销对照（v2.0 §19.3 验收纪律第 3 类待验证预算）："
                "固定动作回放臂——无强制钩子，同 trace 同请求集，"
                f"{REPLAY_MODE_ENV}={REPLAY_MODE_VALUE}"),
        }
    plan["replay_arm"] = replay_arm
    replay_count = 1 if replay_arm else 0
    oneshot_count = 2 if one_shot_pair else 0
    if (total_commands + replay_count + oneshot_count
            > args.max_commands):
        raise ProbeError(
            f"generated {total_commands} candidate commands (+"
            f"{replay_count} replay, +{oneshot_count} one-shot pair) > "
            f"--max-commands {args.max_commands}; "
            "reduce --max-requests or pass explicit --request subset")
    plan["total_commands"] = total_commands

    script = _render_commands_sh(plan)
    markdown = _render_plan_md(plan)

    if args.dry_run:
        print("[probe][dry-run] no files written; plan follows")
        print(json.dumps(plan, indent=1, ensure_ascii=False))
        print("[probe][dry-run] ---- probe_commands.sh (head) ----")
        print("\n".join(script.splitlines()[:40]))
        print(f"[probe][dry-run] ... ({len(script.splitlines())} lines total)")
        return 0

    if out_dir.exists():
        raise ProbeError(
            f"output dir already exists (no-overwrite discipline): {out_dir}; "
            "pass --run-id to pick a fresh identifier")
    (out_dir / "runs").mkdir(parents=True, exist_ok=False)
    with open(plan_path, "w", encoding="utf-8") as sink:
        json.dump(plan, sink, indent=1, ensure_ascii=False)
    sh_path = out_dir / "probe_commands.sh"
    with open(sh_path, "w", encoding="utf-8") as sink:
        sink.write(script)
    os.chmod(sh_path, 0o755)
    with open(out_dir / "probe_plan.md", "w", encoding="utf-8") as sink:
        sink.write(markdown)
    print(f"[probe] plan written: {plan_path}")
    print(f"[probe] commands script: {sh_path} (not executed by this tool)")
    print(f"[probe] targets={len(plan['targets'])} "
          f"families={arm_families} commands={total_commands}"
          f"+{replay_count}replay+{oneshot_count}oneshot")
    return 0


# ---------------------------------------------------------------------------
# summarize：预测排序 vs 实测排序 + 式(24) 选择损失 + 宿主开销报告
# ---------------------------------------------------------------------------

def _plan_arm_list(plan: dict, target: dict) -> list[dict]:
    """归一化 target 的臂列表；旧版 plan（无 arms 键）合成单 entry 臂
    （stage 取 plan 级旧字段），保持旧 plan 可直接 summarize。"""
    arms = target.get("arms")
    if arms:
        return arms
    return [{
        "arm_family": "entry",
        "stage": plan.get("stage", "P"),
        "granularity": plan.get("granularity", "action"),
        "entry_pin": None,
        "baseline_selected": target.get("baseline_selected"),
        "predictions": target.get("predictions", {}),
        "runs": target.get("runs", []),
    }]


def _arm_boundary_tick(plan: dict, target: dict, arm: dict):
    """臂所属边界的 tick（新 plan 取 boundaries；旧 plan 取 target 旧字段）。"""
    stage = arm.get("stage") or plan.get("stage", "P")
    face = (target.get("boundaries") or {}).get(stage)
    if face is not None:
        return face.get("tick")
    return target.get("boundary_tick")


def _run_key(run: dict) -> str:
    """run 的候选键（W8 双维复合键；旧 plan 无 key 字段回退动作名）。"""
    if run.get("key") is not None:
        return run["key"]
    if run.get("instance_index") is not None:
        return candidate_key(run["instance_index"], run["action"])
    return run["action"]


def _rank(actions, key):
    """按 key 升序排名；None 值不参与（防御：实测缺失的动作剔除并披露）。"""
    present = [(a, key[a]) for a in actions if key.get(a) is not None]
    present.sort(key=lambda item: (item[1], ALL_ACTIONS.index(item[0].split(
        ":")[-1]) if item[0].split(":")[-1] in ALL_ACTIONS else 99))
    return [a for a, _ in present]


def _pairwise_errors(rank_a: list[str], rank_b: list[str]) -> tuple[int, int]:
    """共同动作集上的成对逆序数 / 总对数（排序错误度量）。"""
    common = [a for a in rank_a if a in rank_b]
    pos_b = {a: i for i, a in enumerate(rank_b)}
    discordant = 0
    total = 0
    for i in range(len(common)):
        for j in range(i + 1, len(common)):
            total += 1
            if pos_b[common[j]] < pos_b[common[i]]:
                discordant += 1
    return discordant, total


def _instrumented_reference_run(plan: dict, target: dict, stage: str):
    """Find the forced arm that repeats the baseline's selected candidate.

    Production strategy rows intentionally omit the expensive causal prestate
    unless a probe hook is armed.  The generated all-candidate plan already
    contains one forced run for the baseline-selected instance/action, so that
    run is the bounded instrumentation reference.  It is used only when the
    original baseline lacks the causal field; no synthetic object is accepted
    as a substitute.
    """
    face = (target.get("boundaries") or {}).get(stage) or {}
    selected_action = face.get("selected")
    selected_instance = face.get("selected_instance")
    family = "entry" if stage == "P" else "boundary"
    for arm in _plan_arm_list(plan, target):
        if arm.get("arm_family") != family:
            continue
        for run in arm.get("runs", []):
            if run.get("action") != selected_action:
                continue
            # action-granularity runs have no fixed instance.  The scheduler's
            # deterministic tie/order rule is checked below against the row;
            # instance-action runs carry the exact selected instance here.
            if (run.get("instance_index") is not None and
                    run.get("instance_index") != selected_instance):
                continue
            return Path(run["run_dir"])
    return None


def _baseline_reference_cert(plan: dict, target: dict, stage: str,
                             baseline_record: dict | None):
    """Return a complete baseline cert and its provenance.

    A normal run is authoritative when its causal prestate is complete.  For
    the current production path, the prestate is captured only by the existing
    forced-choice instrument.  In that case the selected forced arm becomes a
    reviewable reference after checking that its selected candidate, tick,
    candidate face, epochs, versions, and all other baseline-observable fields
    agree with the ordinary baseline row.  Missing or conflicting fields keep
    the certificate unavailable and therefore cannot certify a probe arm.
    """
    ordinary = (boundary_state_cert(baseline_record, stage)
                if baseline_record is not None else None)
    if ordinary is not None and not ordinary["missing_fields"]:
        ordinary["_evidence_source"] = "baseline_run"
        return ordinary

    reference_run = _instrumented_reference_run(plan, target, stage)
    if reference_run is None or not reference_run.is_dir():
        return ordinary
    try:
        reference_record, _dup, _errors = find_boundary_record(
            reference_run, target["request_id"], stage)
    except ProbeError:
        return ordinary
    reference = boundary_state_cert(reference_record, stage)
    if reference["missing_fields"]:
        return ordinary

    face = (target.get("boundaries") or {}).get(stage) or {}
    if (selected_action_of(reference_record, stage) != face.get("selected")
            or selected_instance_of(reference_record, stage)
            != face.get("selected_instance")):
        return ordinary

    # The forced reference may supply the missing causal prestate only.  Every
    # ordinary field that was present must still agree; this prevents a
    # synthetic/full-shape fixture or a drifted forced run from certifying a
    # different baseline face.
    if ordinary is not None:
        ordinary_fields = ordinary["fields"]
        for name, value in ordinary_fields.items():
            if name == "causal_prestate":
                continue
            if value is None or reference["fields"].get(name) != value:
                return ordinary
        non_causal_missing = [
            name for name in ordinary["missing_fields"]
            if not name.startswith("causal_prestate")]
        if non_causal_missing:
            return ordinary

    reference["_evidence_source"] = (
        "forced_selected_reference:" + str(reference_run))
    return reference


def _baseline_cert_cache(plan: dict):
    """逐 (rid, stage) 的基线边界证书缓存（summarize 一次扫描基线日志）。

    Returns a complete ordinary certificate when available.  Modern online
    baselines often omit ``causal_prestate`` by design; in that case the
    already-planned forced selected arm is used as a bounded, reviewable
    instrumented reference after strict face/epoch checks.  A missing or
    partial reference remains uncertified and cannot become an ``ok`` arm.
    """
    cache: dict = {}
    baseline_run = plan.get("baseline_run")
    if not baseline_run:
        return cache
    for target in plan.get("targets", []):
        rid = target["request_id"]
        for stage in ("P", "D"):
            baseline_record = None
            try:
                baseline_record, _dup, _errors = find_boundary_record(
                    Path(baseline_run), rid, stage)
            except ProbeError:
                pass
            cache[(rid, stage)] = _baseline_reference_cert(
                plan, target, stage, baseline_record)
    return cache


def _summarize_arm(plan: dict, target: dict, arm: dict,
                   baseline_certs: dict | None = None) -> dict:
    """单 (请求, 臂) 的式(24) 选择损失聚合（entry/boundary 同格式）。

    W8 口径（方案 §11.3）：
    - 实测 = **settle 目标**（actual_settle_done − common_decision_tick），
      service 指标（actual_service_done − common tick）并列分列；
    - 状态一致**硬证书**：逐 run 与基线证书逐字段比对（tick/seq/候选面/
      stage_epoch/state_versions/义务/stage_work/…）——任一不匹配 ⇒ 该
      run state_mismatch，**不纳入**同状态损失样本（原 predicted-drift
      note 软降级退役：预测漂移属于候选面字段不匹配）；
    - 未执行（exit_code 缺）/未完成（无 completion/settle）候选 ⇒ missing
      （实测 None，不以预测值填补）；
    - 损失/排序只在证书匹配 + 强制生效 + 实测齐全的子集上计算，覆盖率
      如实披露（全子集才称"该候选集的最优损失"，子集 = "已测子集参照"）。
    """
    if baseline_certs is None:
        baseline_certs = _baseline_cert_cache(plan)
    rid = target["request_id"]
    stage = arm.get("stage") or plan.get("stage", "P")
    family = arm.get("arm_family", "entry")
    granularity = arm.get("granularity", plan.get("granularity", "action"))
    pin = arm.get("entry_pin")
    pin_instance = arm.get("entry_pin_instance")
    if pin_instance is None:
        pin_instance = ((target.get("boundaries") or {}).get("P") or {}).get(
            "selected_instance")
    baseline_cert_source = None
    if baseline_certs.get((rid, stage)) is not None:
        baseline_cert_source = baseline_certs[(rid, stage)].get(
            "_evidence_source", "baseline_run")
    notes = []
    errors: list[str] = []
    actuals = {}          # key -> settle 目标实测（排序/损失主口径）
    actuals_service = {}  # key -> service 实测（分列披露）
    actual_sources = {}
    forced_match = {}
    cert_status = {}
    cert_mismatch_fields = {}
    missing_runs = []
    mismatch_runs = []
    exit_codes = {}
    pin_match = {}
    run_states = {}
    for run in arm.get("runs", []):
        run_errors_start = len(errors)
        key = _run_key(run)
        action = run["action"]
        forced_instance = run.get("instance_index")
        run_dir = Path(run["run_dir"])
        exit_code_path = run_dir / "exit_code"
        exit_code = None
        if exit_code_path.is_file():
            try:
                exit_code = int(
                    exit_code_path.read_text(encoding="utf-8").strip())
            except ValueError:
                errors.append(
                    f"{exit_code_path}: non-integer exit code content")
        else:
            errors.append(
                f"{exit_code_path} missing (probe_commands.sh 未跑或中断)")
        exit_codes[key] = exit_code
        # —— 决策日志：强校验 + 证书 + 实测抽取 ——
        try:
            record, dup, parse_errors = find_boundary_record(
                run_dir, rid, stage)
            errors.extend(parse_errors)
            if dup:
                notes.append(
                    f"{key}: {dup} extra {stage}-boundary records "
                    "(first used)")
            selected = selected_action_of(record, stage)
            selected_instance = selected_instance_of(record, stage)
            if forced_instance is None:
                forced_match[key] = (selected == action)
            else:
                forced_match[key] = (
                    selected == action and selected_instance == forced_instance)
            if forced_match[key] is not True:
                notes.append(
                    f"{key}: forced identity (instance={forced_instance!r}, "
                    f"action={action!r}) NOT selected at boundary "
                    f"(selected=({selected_instance!r}, {selected!r})) — "
                    "强制钩子未生效或被拒，该臂实测不构成有效同状态对照")
            # —— W8 状态一致硬证书（逐字段 vs 基线）——
            run_cert = boundary_state_cert(record, stage)
            baseline_cert = baseline_certs.get((rid, stage))
            if baseline_cert is None:
                cert_status[key] = "baseline_unavailable"
                notes.append(
                    f"{key}: 基线 {stage} 边界记录不可用——证书无法签发，"
                    "该 run 不得 status=ok（缺证书硬门）")
            else:
                mismatched = cert_mismatches(baseline_cert, run_cert)
                if mismatched:
                    cert_status[key] = "state_mismatch"
                    cert_mismatch_fields[key] = mismatched
                    mismatch_runs.append(key)
                    notes.append(
                        f"{key}: 状态证书不匹配字段 {mismatched} — 重放前缀"
                        "在目标边界与基线状态不一致，该分支不纳入同状态"
                        "损失样本")
                else:
                    cert_status[key] = "match"
            # —— 实测抽取（service/settle 分列；settle 为主口径）——
            surface = aggregate_by_instance_action(
                extract_candidates(record, stage))
            slot = surface.get(
                (forced_instance if forced_instance is not None
                 else selected_instance, action), {})
            measured = slot.get("measured_ns")
            boundary_tick = record.get("tick")
            ticks = service_settle_ticks(run_dir, rid, errors)
            if measured is not None and not (
                    type(measured) in (int, float) and math.isfinite(measured)
                    and measured >= 0):
                errors.append(f"{key}: invalid measured settle duration {measured!r}")
            if (type(measured) in (int, float) and math.isfinite(measured)
                    and measured >= 0):
                actuals[key] = measured
                actual_sources[key] = "measured"
                if isinstance(ticks["service_done_ns"], (int, float)) and \
                        isinstance(boundary_tick, (int, float)):
                    actuals_service[key] = (
                        ticks["service_done_ns"] - boundary_tick)
            elif (ticks["settle_done_ns"] is not None
                    and isinstance(boundary_tick, (int, float))):
                actuals[key] = ticks["settle_done_ns"] - boundary_tick
                actual_sources[key] = ticks["settle_source"]
                if ticks["service_done_ns"] is not None:
                    actuals_service[key] = (
                        ticks["service_done_ns"] - boundary_tick)
            else:
                if ticks.get("settle_missing_reason"):
                    notes.append(
                        f"{key}: settle 真值缺失（"
                        f"{ticks['settle_missing_reason']}）——不以预测值"
                        "填补，标 missing")
                else:
                    errors.append(
                        f"{key}: cannot compute settle delta (settle_done="
                        f"{ticks['settle_done_ns']}, boundary_tick="
                        f"{boundary_tick})")
        except ProbeError as exc:
            errors.append(f"{key}: {exc}")
            forced_match[key] = None
            cert_status[key] = "record_unavailable"
        # —— boundary 臂专属：entry pin 守恒核对（P 侧证书 + 选择）——
        if family == "boundary" and pin is not None:
            try:
                p_record, _, p_errors = find_boundary_record(
                    run_dir, rid, "P")
                errors.extend(p_errors)
                p_selected = selected_action_of(p_record, "P")
                p_instance = selected_instance_of(p_record, "P")
                pin_match[key] = (p_selected == pin and (
                    pin_instance is None or p_instance == pin_instance))
                if pin_match[key] is not True:
                    notes.append(
                        f"{key}: P 入口选择 ({p_instance!r}, {p_selected!r}) "
                        f"!= entry pin ({pin_instance!r}, {pin!r}) — "
                        "D 边界同状态前提被破坏（boundary 臂按"
                        "互斥合同不强制 P 入口，P 选择靠同 trace 复现基线"
                        "；漂移即对照降级）")
                p_baseline_cert = baseline_certs.get((rid, "P"))
                if p_baseline_cert is not None:
                    p_cert = boundary_state_cert(p_record, "P")
                    p_mismatch = cert_mismatches(p_baseline_cert, p_cert)
                    if p_mismatch:
                        cert_status[key] = "state_mismatch"
                        cert_mismatch_fields.setdefault(
                            key, []).extend(
                                f"P:{name}" for name in p_mismatch)
                        if key not in mismatch_runs:
                            mismatch_runs.append(key)
                        notes.append(
                            f"{key}: P 面证书不匹配字段 {p_mismatch} — "
                            "P action pin 之外的状态维度也不一致（§11.3："
                            "P action pin 单独不足以证明状态一致）")
                elif cert_status.get(key) == "match":
                    cert_status[key] = "baseline_unavailable"
                    notes.append(
                        f"{key}: 基线 P 边界记录不可用——P 面证书无法签发")
            except ProbeError as exc:
                pin_match[key] = None
                notes.append(
                    f"{key}: 无法核对 entry pin（{exc}）— 该臂同状态"
                    "前提未验证，对照降级")
        # —— run 级状态归类（missing 优先级仅次于解析 error）——
        if key in actuals and actuals[key] < 0:
            errors.append(f"{key}: observed settle precedes its decision tick")
        if exit_code is None:
            run_states[key] = "missing"
            missing_runs.append(key)
        elif exit_code != 0:
            run_states[key] = "missing"
            missing_runs.append(key)
            notes.append(
                f"{key}: run 非零退出（exit={exit_code}）——未完成候选标 "
                "missing，不以预测值填补")
        elif key not in actuals:
            run_states[key] = "missing"
            missing_runs.append(key)
        elif len(errors) > run_errors_start:
            run_states[key] = "error"
        elif cert_status.get(key) == "state_mismatch":
            run_states[key] = "state_mismatch"
        elif forced_match.get(key) is not True:
            run_states[key] = "degraded"
        elif cert_status.get(key) != "match":
            run_states[key] = "degraded"
        elif family == "boundary" and pin is not None and pin_match.get(key) is not True:
            run_states[key] = "degraded"
        else:
            run_states[key] = "ok"

    predictions = {
        k: v for k, v in (arm.get("predictions")
                          or target.get("predictions", {})).items()
        if type(v) in (int, float) and math.isfinite(v) and v >= 0}
    # 同状态样本 = 证书匹配 + 强制生效 + settle 实测在位（missing /
    # state_mismatch / degraded 剔除——损失与排序只在该子集上计算）。
    same_state_keys = [
        key for key, state in run_states.items() if state == "ok"]
    sample_predictions = {k: predictions[k] for k in same_state_keys
                          if k in predictions}
    sample_actuals = {k: actuals[k] for k in same_state_keys
                      if k in actuals}
    predicted_rank = _rank(list(sample_predictions), sample_predictions)
    actual_rank = _rank(list(sample_actuals), sample_actuals)
    discordant, total_pairs = _pairwise_errors(predicted_rank, actual_rank)
    top1_match = (
        bool(predicted_rank) and bool(actual_rank)
        and predicted_rank[0] == actual_rank[0])
    selection_loss = None
    if predicted_rank and actual_rank:
        chosen = predicted_rank[0]
        if chosen in sample_actuals:
            selection_loss = (
                sample_actuals[chosen] - min(sample_actuals.values()))
    # 逐候选误差（预测 − 实测；两侧齐全才报，否则缺席披露）。
    errors_ns = {
        k: predictions[k] - sample_actuals[k]
        for k in predictions if k in sample_actuals}
    baseline_chosen = arm.get("baseline_selected")
    if baseline_chosen is None:
        face = (target.get("boundaries") or {}).get(stage)
        baseline_chosen = (face or {}).get("selected")
    argmin_matches_chosen = (
        bool(predicted_rank) and baseline_chosen is not None
        and predicted_rank[0].split(":")[-1] == baseline_chosen)

    run_keys = [_run_key(r) for r in arm.get("runs", [])]
    n_ok = len(same_state_keys)
    n_total = len(run_keys)
    # —— 状态裁决（§11.3 硬门：缺证书/漂移 ⇒ 不得 ok）——
    if errors and not run_keys:
        status = "error"
    elif any(state == "state_mismatch"
             for state in run_states.values()):
        status = "state_mismatch"
    elif n_ok < n_total:
        # 部分缺失/降级：全部在测候选 ok 才称 ok；子集如实披露。
        status = "degraded" if n_ok else (
            "missing" if all(state == "missing"
                             for state in run_states.values()) else "error")
    elif not all(cert_status.get(k) == "match" for k in run_keys):
        status = "degraded"  # 证书不可签发（基线缺失等）——不 ok
    elif (family == "boundary" and pin is not None
          and not all(pin_match.get(k) is True for k in run_keys)):
        status = "degraded"
    else:
        status = "ok"

    # —— W8（阶段C.5）：P 参考接续 vs 实际两阶段效果分离 ——
    p_reference = None
    actual_boundary = None
    if family == "boundary":
        p_face = (target.get("boundaries") or {}).get("P")
        d_face = (target.get("boundaries") or {}).get("D")
        if p_face is not None and p_face.get("reference_continuation"):
            p_reference = p_face["reference_continuation"]
        if d_face is not None:
            actual_boundary = {
                "instance": d_face.get("selected_instance"),
                "action": d_face.get("selected"),
            }

    return {
        "record_kind": "arm_summary",
        "run_id": plan.get("run_id"),
        "request_id": rid,
        "arm_family": family,
        "arm_class": arm.get("arm_class", "same_state_forced_branch"),
        "granularity": granularity,
        "stage": stage,
        "boundary_tick": _arm_boundary_tick(plan, target, arm),
        "baseline_selected": baseline_chosen,
        "entry_pin": pin if family == "boundary" else None,
        "entry_pin_instance": pin_instance if family == "boundary" else None,
        "entry_pin_match": (pin_match if family == "boundary" else None),
        # P 参考接续（P 估价内的"参考 D"）与实际边界 D 分列——接续近似
        # 误差与资源误差分开归因（§11.3 P 测量区分"估价的固定参考 D"
        # 与"实际边界独立 D"）。
        "p_reference_continuation": p_reference,
        "actual_boundary_choice": actual_boundary,
        "predictions": predictions,
        "predicted_rank": predicted_rank,
        "actuals": sample_actuals,
        "actuals_all": actuals,
        "actuals_service": actuals_service,
        "actual_sources": actual_sources,
        "actual_rank": actual_rank,
        "errors_ns": errors_ns,
        "pairwise_discordant": discordant,
        "pairwise_total": total_pairs,
        "top1_match": top1_match,
        "selection_loss_ns": selection_loss,
        "selection_loss_target": "settle",
        "argmin_matches_baseline_chosen": argmin_matches_chosen,
        "forced_selected_match": forced_match,
        "exit_codes": exit_codes,
        "state_cert": {
            "status_by_run": cert_status,
            "mismatch_fields_by_run": cert_mismatch_fields,
            "baseline_source": baseline_cert_source,
            "rng_basis": CERT_RNG_BASIS,
            "coverage": CERT_COVERAGE_NOTE,
        },
        "run_states": run_states,
        "missing_candidates": sorted(set(missing_runs)),
        "state_mismatch_candidates": sorted(set(mismatch_runs)),
        "same_state_sample_coverage": {
            "ok_runs": n_ok,
            "total_runs": n_total,
            "caliber": ("同状态全候选集" if n_ok == n_total and n_total
                        else f"已测子集参照（{n_ok}/{n_total}）"),
        },
        "status": status,
        "errors": errors,
        "notes": notes,
    }


# —— 宿主开销报告（v2.0 §19.3 验收纪律第 3 类：待验证预算，非结论）——

def _read_run_decision_timing(run_dir: Path) -> dict:
    """读一个 run 的决策侧计时（只消费既有产物字段，不改调度器）。

    优先 results/online_stats.jsonl 汇总行 predictor_cost（P/D 分列 +
    总计 + 摘要维护 + 宿主墙时）；缺失回落决策日志 kind==
    joint_decision_metrics 收尾行（仅 P 面合计——口径降级如实披露）；
    再缺失 source="missing"。决策日志行数独立清点（逐行墙时分母）。
    """
    run_dir = Path(run_dir)
    metric = {
        "decision_log_rows": None,
        "decision_wall_total_ns": None,
        "decision_stage_total_ns": {},
        "summary_maintenance_total_ns": None,
        "host_wall_ns": None,
        "host_share_ppm": None,
        "per_row_ns": None,
        "source": "missing",
    }
    log_path = decision_log_path(run_dir)
    if log_path.is_file():
        metric["decision_log_rows"] = sum(
            1 for _ in _iter_log_records(log_path, []))
    stats_path = run_dir / "results" / "online_stats.jsonl"
    if stats_path.is_file():
        for record in _iter_log_records(stats_path, []):
            if not record.get("summary"):
                continue
            cost = record.get("predictor_cost")
            if not isinstance(cost, dict):
                continue
            total = (cost.get("decision_wall_ns") or {}).get("total_ns")
            if isinstance(total, (int, float)):
                metric["decision_wall_total_ns"] = total
                metric["source"] = "online_stats.predictor_cost"
            stages = cost.get("decision_stages")
            if isinstance(stages, dict):
                for stage in ("P", "D"):
                    block = stages.get(stage)
                    if isinstance(block, dict):
                        metric["decision_stage_total_ns"][stage] = (
                            block.get("total_ns"))
            summary_block = cost.get("summary_maintenance")
            if isinstance(summary_block, dict):
                metric["summary_maintenance_total_ns"] = (
                    summary_block.get("total_ns"))
            metric["host_wall_ns"] = cost.get("host_wall_ns")
            metric["host_share_ppm"] = cost.get("host_share_ppm")
            break
    if (metric["source"] == "missing" and log_path.is_file()):
        # 回落：决策日志收尾指标行（C11 埋点；仅 P 面合计，无 D 面分列）。
        for record in _iter_log_records(log_path, []):
            if record.get("kind") != "joint_decision_metrics":
                continue
            decision = record.get("decision") or {}
            adm = decision.get("admission_decision_wall_ns")
            if isinstance(adm, dict) and isinstance(
                    adm.get("total_ns"), (int, float)):
                metric["decision_wall_total_ns"] = adm["total_ns"]
                metric["source"] = (
                    "decision_log.joint_decision_metrics(P-only)")
            break
    total = metric["decision_wall_total_ns"]
    rows = metric["decision_log_rows"]
    if isinstance(total, (int, float)) and isinstance(rows, int) and rows > 0:
        metric["per_row_ns"] = round(total / rows, 3)
    return metric


def _host_overhead_report(plan: dict) -> dict:
    """宿主开销报告：两阶段决策(joint/forced) vs 固定动作回放(load-first)。

    比值 = 决策侧逐行墙时均值 ÷ 回放侧逐行墙时；预算 ≤1.10（额外开销
    ≤10%）。回放侧缺 load-first 臂时回退 forced 臂并如实披露"该口径不
    构成预算证据"（forced 臂仍全量跑两阶段决策计算）→ 比值置 None。
    """
    entries = []
    candidates = []
    baseline = plan.get("baseline_run")
    if baseline:
        candidates.append(("baseline", Path(baseline), "joint"))
    replay = plan.get("replay_arm")
    if replay and replay.get("run_dir"):
        candidates.append(("replay_load_first", Path(replay["run_dir"]),
                           "load-first"))
    for target in plan.get("targets", []):
        for arm in _plan_arm_list(plan, target):
            for run in arm.get("runs", []):
                candidates.append((
                    "{}::{}::{}".format(
                        target.get("request_id"),
                        arm.get("arm_family", "entry"),
                        _run_key(run)),
                    Path(run["run_dir"]), "joint+forced"))
    for label, run_dir, mode in candidates:
        timing = _read_run_decision_timing(run_dir)
        entries.append({"label": label, "run_dir": str(run_dir),
                        "mode": mode, **timing})

    def _mean(values):
        return round(sum(values) / len(values), 3) if values else None

    joint_entries = [e for e in entries
                     if e["mode"] in ("joint", "joint+forced")
                     and e.get("per_row_ns") is not None]
    replay_entries = [e for e in entries if e["mode"] == "load-first"
                      and e.get("per_row_ns") is not None]
    replay_basis = None
    if replay_entries:
        replay_basis = "load-first 臂"
        replay_side = {
            "per_row_ns_mean": _mean(
                [e["per_row_ns"] for e in replay_entries]),
            "labels": [e["label"] for e in replay_entries],
            "basis": replay_basis,
        }
    else:
        fallback = [e for e in entries if e["mode"] == "joint+forced"
                    and e.get("per_row_ns") is not None]
        replay_side = {
            "per_row_ns_mean": _mean(
                [e["per_row_ns"] for e in fallback]) if fallback else None,
            "labels": [e["label"] for e in fallback],
            "basis": ("forced 臂回退（load-first 臂缺失/缺测；forced 臂仍"
                      "全量跑两阶段决策计算——该回退口径不构成预算证据，"
                      "比值置 None）") if fallback else "无可用回放臂",
        }
    joint_side = {
        "per_row_ns_mean": _mean(
            [e["per_row_ns"] for e in joint_entries]),
        "labels": [e["label"] for e in joint_entries],
        "note": ("joint 模式两阶段决策开启（基线 + forced 臂——强制只换"
                 "选择，决策计算照常全量跑）"),
    } if joint_entries else None

    ratio = None
    delta = None
    if (joint_side and replay_side
            and joint_side["per_row_ns_mean"] is not None
            and replay_side["per_row_ns_mean"] is not None
            and replay_side["per_row_ns_mean"] > 0
            and replay_basis == "load-first 臂"):
        ratio = round(
            joint_side["per_row_ns_mean"]
            / replay_side["per_row_ns_mean"], 4)
        delta = round(
            joint_side["per_row_ns_mean"]
            - replay_side["per_row_ns_mean"], 3)
    if ratio is None:
        budget_met = None
        verdict = "insufficient_data"
    elif ratio <= OVERHEAD_BUDGET_RATIO:
        budget_met = True
        verdict = "met"
    else:
        budget_met = False
        verdict = "not_met"
    return {
        "record_kind": "host_overhead",
        "budget_ratio": OVERHEAD_BUDGET_RATIO,
        "metric_caliber": OVERHEAD_CALIBER_NOTE,
        "runs": entries,
        "joint_side": joint_side,
        "replay_side": replay_side,
        "ratio_decision_wall_per_row": ratio,
        "delta_ns_per_row": delta,
        "budget_met": budget_met,
        "verdict": verdict,
        "verdict_note": (
            "待验证预算（v2.0 §19.3 验收纪律第 3 类），非结论——达成/"
            "未达成/数据不足均如实披露；宿主 Python 墙时不外推片上决策"
            "时延（§19.3 末段）"),
    }


# —— 一次映射 vs 两阶段闭环对照（验收纪律第 2 类，2026-09-28）——
#
# 口径纪律：逐指标两 run 值 + 差；来源逐行披露（决策日志 / request_
# metrics.csv / online_stats.jsonl / bridge kv_ledgers——全部真实产物）；
# 未执行/不可观测项真值 NA 并注明，禁止用预测值冒充。

ONE_SHOT_CALIBER_NOTE = (
    "两 run 同 trace 同参数，仅差 JOINT_BOUNDARY_POLICY（两阶段参照不设、"
    "一次映射臂 p-continuation）。指标来源：D 改点率/同址改动作率/交接字"
    "节 = 两阶段参照 run 决策日志逐请求派生（D 行选择经 P 行 "
    "joint_instance_index 联表、延续形态 = (P实例, stay)——三动作下 p-"
    "continuation 恒本地延续）；E2E/完成率/首 token 间隔 = 各 run "
    "request_metrics.csv 真实事件字段；控制开销 = results/online_stats."
    "jsonl predictor_cost（P/D 分列）。现行日志无第二 token 事件——次 "
    "token 间隔真值 NA，不以均值/预测冒充。")


def _read_request_metrics(run_dir: Path):
    """读 run 根 request_metrics.csv（防御式）：逐请求真实事件字段。

    返回 dict[request_id, row]（row 值为字符串，数值列按需转换）；文件
    缺失返回 None（调用方如实降级为 NA）。"""
    path = Path(run_dir) / "request_metrics.csv"
    if not path.is_file():
        return None
    import csv as _csv
    rows = {}
    with open(path, encoding="utf-8", newline="") as source:
        for row in _csv.DictReader(source):
            rid = row.get("request_id")
            if rid:
                rows[rid] = row
    return rows


def _csv_number(row, key):
    """csv 行字段 → 数值（NA/空/非数值 → None）。"""
    if row is None:
        return None
    raw = (row.get(key) or "").strip()
    if not raw or raw.upper() == "NA":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def _read_handoff_bytes(run_dir: Path):
    """C13 copy handoff 交接字节聚合（bridge/joint_kv_ledgers.json 的
    copy_handoff_events[].d_handoff_bytes 求和；文件缺失 → (None, 缺因)）。"""
    path = Path(run_dir) / "bridge" / "joint_kv_ledgers.json"
    if not path.is_file():
        return None, f"missing {path}"
    try:
        with open(path, encoding="utf-8") as source:
            ledgers = json.load(source)
    except (OSError, ValueError) as exc:
        return None, f"unreadable {path}: {exc}"
    events = ledgers.get("copy_handoff_events")
    if not isinstance(events, list):
        return 0, None
    total = 0
    for event in events:
        value = (event.get("d_handoff_bytes")
                 if isinstance(event, dict) else None)
        if isinstance(value, (int, float)):
            total += value
    return total, None


def _collect_boundary_faces(run_dir: Path, errors: list[str]):
    """逐请求 P/D 边界行摘要（一次映射对照的派生基底）。

    返回 (faces, order)：faces[rid] = {"p_instance", "p_action",
    "d": {...} or None}；d 摘要含 tick/chosen_instance/chosen_action/
    is_continuation/policy/handoff_bytes（P→D 迁移腿 shards 聚合）。"""
    path = decision_log_path(run_dir)
    if not path.is_file():
        raise ProbeError(f"decision log missing: {path}")
    faces: dict[str, dict] = {}
    order: list[str] = []
    for record in _iter_log_records(path, errors):
        kind = record.get("kind")
        decision = record.get("decision")
        if not isinstance(decision, dict):
            continue
        rid = record.get("request_id")
        if not isinstance(rid, str) or not rid:
            continue
        if kind == "joint_admission":
            face = faces.setdefault(
                rid, {"p_instance": None, "p_action": None, "d": None})
            if face["p_instance"] is None:
                face["p_instance"] = decision.get("joint_instance_index")
            if face["p_action"] is None:
                face["p_action"] = decision.get(
                    "selected_action", decision.get("joint_action"))
            if rid not in order:
                order.append(rid)
        elif kind == "decode" and isinstance(
                decision.get("boundary_decision"), dict):
            face = faces.setdefault(
                rid, {"p_instance": None, "p_action": None, "d": None})
            boundary = decision["boundary_decision"]
            transfer = decision.get("prefill_decode_transfer")
            shards = (transfer or {}).get("shards") or []
            handoff = 0
            for shard in shards:
                value = (shard or {}).get("bytes")
                if isinstance(value, (int, float)):
                    handoff += value
            if face["d"] is None:
                face["d"] = {
                    "tick": record.get("tick"),
                    "chosen_instance": boundary.get("chosen_instance_index"),
                    "chosen_action": boundary.get("chosen_action"),
                    "is_continuation": boundary.get("is_continuation"),
                    "policy": boundary.get("boundary_policy"),
                    "handoff_bytes": handoff,
                }
            if rid not in order:
                order.append(rid)
    return faces, order


def _one_shot_comparison(plan: dict):
    """一次映射 vs 两阶段闭环对照汇总（plan 含 one_shot_pair 时输出）。"""
    pair = plan.get("one_shot_pair")
    if not pair:
        return None
    errors: list[str] = []
    notes: list[str] = []
    twophase_dir = Path(pair["twophase_run"]["run_dir"])
    oneshot_dir = Path(pair["oneshot_run"]["run_dir"])

    twophase_faces, order = _collect_boundary_faces(twophase_dir, errors)
    oneshot_faces, _ = _collect_boundary_faces(oneshot_dir, errors)

    # —— D 改点率 / 同址改动作率 / 交接字节（两阶段参照派生）——
    d_requests = 0
    re_point = 0
    same_instance_action_change = 0
    handoff_migration = 0
    for rid in order:
        face = twophase_faces.get(rid) or {}
        d = face.get("d")
        if d is None:
            continue
        d_requests += 1
        handoff_migration += d.get("handoff_bytes") or 0
        p_instance = face.get("p_instance")
        p_action = face.get("p_action")
        if p_instance is None or p_action is None:
            errors.append(
                f"{rid}: P 边界行缺 joint_instance_index/selected_action"
                "——延续形态无法派生")
            continue
        # 三动作下 p-continuation 恒本地延续（stay）。
        continuation_action = "stay"
        changed = (d.get("chosen_instance") != p_instance
                   or d.get("chosen_action") != continuation_action)
        if changed:
            re_point += 1
            if d.get("chosen_instance") == p_instance:
                same_instance_action_change += 1

    # —— 一次映射臂不变量核对（实例位/动作位均按实测记录对账，
    # 非"按构造 0"标签——动作位偏差场景如实披露）——
    oneshot_d_rows = 0
    oneshot_policy_rows = 0
    oneshot_at_p = 0
    oneshot_handoff_migration = 0
    p_face_drift = []
    oneshot_action_match = 0
    oneshot_action_verifiable = 0
    oneshot_continuation_match = 0
    oneshot_same_inst_action_change = 0
    for rid in order:
        twophase_face = twophase_faces.get(rid) or {}
        oneshot_face = oneshot_faces.get(rid) or {}
        d = oneshot_face.get("d")
        if d is None:
            continue
        oneshot_d_rows += 1
        if d.get("policy") == ONESHOT_POLICY_VALUE:
            oneshot_policy_rows += 1
        else:
            errors.append(
                f"{rid}: 一次映射臂 D 行缺 boundary_policy="
                f"{ONESHOT_POLICY_VALUE} 披露键（got {d.get('policy')!r}）")
        p_instance = oneshot_face.get("p_instance")
        p_action = oneshot_face.get("p_action")
        instance_ok = (d.get("chosen_instance") is not None
                       and d.get("chosen_instance") == p_instance)
        if instance_ok:
            oneshot_at_p += 1
        oneshot_handoff_migration += d.get("handoff_bytes") or 0
        # P 面同状态前提：policy 只改 D 边界，P 选择应与两阶段参照一致。
        if (twophase_face.get("p_instance") != p_instance
                or twophase_face.get("p_action") != p_action):
            p_face_drift.append(rid)
        # 动作位对账（与实例位同款：实测 chosen_action vs 派生延续
        # 形态，不用构造性断言替代）。
        action_ok = False
        if d.get("chosen_action") is not None and p_action is not None:
            oneshot_action_verifiable += 1
            # 三动作下延续形态恒 stay（同上）。
            continuation_action = "stay"
            action_ok = d.get("chosen_action") == continuation_action
            if action_ok:
                oneshot_action_match += 1
            elif instance_ok:
                oneshot_same_inst_action_change += 1
        if instance_ok and action_ok:
            oneshot_continuation_match += 1
    if oneshot_d_rows and oneshot_at_p != oneshot_d_rows:
        errors.append(
            f"一次映射臂 D 实例 ≠ P 实例的请求 {oneshot_d_rows - oneshot_at_p}"
            f"/{oneshot_d_rows}（一次映射语义被破坏）")
    oneshot_re_point = oneshot_d_rows - oneshot_continuation_match
    if oneshot_action_verifiable and (
            oneshot_action_match != oneshot_action_verifiable):
        notes.append(
            "一次映射臂 D 动作位实测对账："
            f"{oneshot_action_verifiable - oneshot_action_match}"
            f"/{oneshot_action_verifiable} 行 chosen_action ≠ 派生延续"
            "形态（动作位偏差场景如实披露，非按构造断言 0）")
    if p_face_drift:
        notes.append(
            "P 面选择漂移（对照降级披露，不影响 D 面语义核对）: "
            f"{p_face_drift}")

    def _rate(numerator, denominator):
        if not denominator:
            return None
        return round(numerator / denominator, 6)

    # —— E2E / 完成率 / 首 token 间隔（request_metrics.csv 真实事件）——
    def _run_completion_metrics(run_dir, run_label):
        rows = _read_request_metrics(run_dir)
        if rows is None:
            notes.append(
                f"{run_label}: request_metrics.csv 缺失——E2E/完成率/首 "
                "token 间隔真值 NA（文件级缺失，不猜值）")
            return {"requests": None, "completed": None,
                    "completion_rate": None, "e2e_mean_ns": None,
                    "first_token_interval_mean_ns": None}
        completed = 0
        e2e_values = []
        ttft_values = []
        for row in rows.values():
            if (row.get("terminal_status") or "").strip() == "completed":
                completed += 1
            e2e = _csv_number(row, "e2e_ns")
            if e2e is not None:
                e2e_values.append(e2e)
            first_token = _csv_number(row, "first_token_ns")
            arrival = _csv_number(row, "arrival_ns")
            if first_token is not None and arrival is not None:
                ttft_values.append(first_token - arrival)
        count = len(rows)
        return {
            "requests": count,
            "completed": completed,
            "completion_rate": _rate(completed, count),
            "e2e_mean_ns": (
                round(sum(e2e_values) / len(e2e_values), 3)
                if e2e_values else None),
            "first_token_interval_mean_ns": (
                round(sum(ttft_values) / len(ttft_values), 3)
                if ttft_values else None),
        }

    twophase_completion = _run_completion_metrics(
        twophase_dir, "twophase_reference")
    oneshot_completion = _run_completion_metrics(
        oneshot_dir, "oneshot_p_continuation")
    if (twophase_completion["first_token_interval_mean_ns"] is None
            and twophase_completion["requests"] is not None):
        notes.append(
            "首 token 间隔真值 NA：request_metrics.csv 的 first_token_ns "
            "无真实事件（no code-8 first-token event）——不以预测值冒充")

    # —— 控制开销（predictor_cost 账本；P/D 分列）——
    twophase_timing = _read_run_decision_timing(twophase_dir)
    oneshot_timing = _read_run_decision_timing(oneshot_dir)

    def _fmt(value):
        return "NA" if value is None else str(value)

    def _delta(a, b):
        if a is None or b is None:
            return "NA"
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            return round(b - a, 6)
        return "NA"

    re_point_rate = _rate(re_point, d_requests)
    same_inst_rate = _rate(same_instance_action_change, d_requests)
    c2p, c1s = twophase_completion, oneshot_completion
    h2, h2_err = _read_handoff_bytes(twophase_dir)
    h1, h1_err = _read_handoff_bytes(oneshot_dir)
    if h2_err:
        notes.append(f"两阶段参照 C13 handoff 聚合不可用: {h2_err}")
    if h1_err:
        notes.append(f"一次映射臂 C13 handoff 聚合不可用: {h1_err}")

    # 一次映射臂改点指标列：实测核对结果（非构造性标签）——分子为
    # 实测偏差行数，括注披露核对基准（一致行/可核对行）。
    oneshot_re_point_label = (
        f"{oneshot_re_point}（实测核对 {oneshot_continuation_match}/"
        f"{oneshot_d_rows} 一致）" if oneshot_d_rows else "NA")
    oneshot_action_label = (
        f"{oneshot_same_inst_action_change}（实测核对动作位 "
        f"{oneshot_action_match}/{oneshot_action_verifiable} 一致）"
        if oneshot_action_verifiable else "NA")
    metrics_rows = [
        ("D 改点率（最终 (instance,action) ≠ 延续形态）",
         _fmt(re_point_rate) if d_requests else "NA",
         oneshot_re_point_label, "NA"),
        ("同址改动作率（instance 不变 action 变）",
         _fmt(same_inst_rate) if d_requests else "NA",
         oneshot_action_label, "NA"),
        ("D 边界改点请求数 / D 边界请求数",
         f"{re_point}/{d_requests}" if d_requests else "NA",
         f"{oneshot_re_point}/{oneshot_d_rows}"
         if oneshot_d_rows else "NA", "NA"),
        ("交接字节（P→D 迁移腿，决策日志 shards 聚合）",
         _fmt(handoff_migration), _fmt(oneshot_handoff_migration),
         _delta(handoff_migration, oneshot_handoff_migration)),
        ("交接字节（C13 copy_handoff d_handoff_bytes 聚合）",
         _fmt(h2), _fmt(h1), _delta(h2, h1)),
        ("E2E 均值 ns（request_metrics.csv）",
         _fmt(c2p["e2e_mean_ns"]), _fmt(c1s["e2e_mean_ns"]),
         _delta(c2p["e2e_mean_ns"], c1s["e2e_mean_ns"])),
        ("完成率（terminal_status=completed）",
         _fmt(c2p["completion_rate"]), _fmt(c1s["completion_rate"]),
         _delta(c2p["completion_rate"], c1s["completion_rate"])),
        ("完成请求数",
         _fmt(c2p["completed"]), _fmt(c1s["completed"]),
         _delta(c2p["completed"], c1s["completed"])),
        ("首 token 间隔均值 ns（first_token_ns−arrival_ns）",
         _fmt(c2p["first_token_interval_mean_ns"]),
         _fmt(c1s["first_token_interval_mean_ns"]),
         _delta(c2p["first_token_interval_mean_ns"],
                c1s["first_token_interval_mean_ns"])),
        ("次 token 间隔",
         "NA（现行日志无第二 token 事件）",
         "NA（现行日志无第二 token 事件）", "NA"),
        ("控制开销 decision_wall_total_ns（predictor_cost）",
         _fmt(twophase_timing["decision_wall_total_ns"]),
         _fmt(oneshot_timing["decision_wall_total_ns"]),
         _delta(twophase_timing["decision_wall_total_ns"],
                oneshot_timing["decision_wall_total_ns"])),
        ("控制开销 D 阶段 total_ns",
         _fmt(twophase_timing["decision_stage_total_ns"].get("D")),
         _fmt(oneshot_timing["decision_stage_total_ns"].get("D")),
         _delta(twophase_timing["decision_stage_total_ns"].get("D"),
                oneshot_timing["decision_stage_total_ns"].get("D"))),
    ]
    status = "error" if errors else ("ok" if oneshot_d_rows else "degraded")
    if oneshot_d_rows == 0 and not errors:
        notes.append("一次映射臂无 D 边界行——对照无从核对，降级披露")
    return {
        "record_kind": "one_shot_comparison",
        "run_id": plan.get("run_id"),
        "metric_caliber": ONE_SHOT_CALIBER_NOTE,
        "twophase_run": pair["twophase_run"]["run_dir"],
        "oneshot_run": pair["oneshot_run"]["run_dir"],
        "policy_env": ONESHOT_POLICY_ENV,
        "policy_value": ONESHOT_POLICY_VALUE,
        "d_requests": d_requests,
        "re_point_requests": re_point,
        "re_point_rate": re_point_rate,
        "same_instance_action_change": same_instance_action_change,
        "same_instance_action_change_rate": same_inst_rate,
        "handoff_migration_bytes": {
            "twophase": handoff_migration,
            "oneshot": oneshot_handoff_migration,
        },
        "handoff_c13_bytes": {"twophase": h2, "oneshot": h1},
        "completion": {"twophase": c2p, "oneshot": c1s},
        "decision_timing": {"twophase": twophase_timing,
                            "oneshot": oneshot_timing},
        "oneshot_disclosure_rows": oneshot_policy_rows,
        "oneshot_d_rows": oneshot_d_rows,
        # 一次映射臂动作位实测对账（标签括注的结构化底数）。
        "oneshot_action_match": oneshot_action_match,
        "oneshot_action_verifiable": oneshot_action_verifiable,
        "oneshot_re_point_requests": oneshot_re_point,
        "oneshot_same_instance_action_change": (
            oneshot_same_inst_action_change),
        "p_face_drift": p_face_drift,
        "metrics": [
            {"metric": name, "twophase": a, "oneshot": b, "delta": d}
            for name, a, b, d in metrics_rows],
        "status": status,
        "errors": errors,
        "notes": notes,
    }


def summarize_plan(plan: dict, dry_run: bool) -> int:
    overall_status = "ok"
    summary_records = []
    baseline_certs = _baseline_cert_cache(plan)
    for target in plan.get("targets", []):
        for arm in _plan_arm_list(plan, target):
            record = _summarize_arm(plan, target, arm, baseline_certs)
            summary_records.append(record)
            if record["status"] == "error":
                overall_status = "error"
            elif (record["status"] in ("degraded", "state_mismatch",
                                       "missing")
                  and overall_status == "ok"):
                overall_status = "degraded"
    # 一次映射 vs 两阶段闭环对照（验收纪律第 2 类；plan 含 one_shot_pair
    # 时输出——状态同口径并入 overall）。
    one_shot = _one_shot_comparison(plan)
    if one_shot is not None:
        summary_records.append(one_shot)
        if one_shot["status"] == "error":
            overall_status = "error"
        elif one_shot["status"] == "degraded" and overall_status == "ok":
            overall_status = "degraded"
    overhead = _host_overhead_report(plan)
    summary_records.append(overhead)

    # —— 渲染 markdown ——
    md = [
        "# 同状态对照 probe 汇总（式(24) 选择损失）",
        "",
        f"- run_id: `{plan.get('run_id')}`",
        f"- baseline_run: `{plan.get('baseline_run')}`",
        f"- arm_families: {plan.get('arm_families', ['entry'])}"
        "（entry=P 入口；boundary=固定 P 动作 + D 边界枚举）",
        f"- granularity: {plan.get('granularity', 'action')}"
        "（action=旧口径逐动作；instance-action=逐 (instance,action) 候选"
        "TARGET 精确强制）",
        f"- 钩子契约: {FORCE_REQUEST_ENV}/{FORCE_ENTRY_ACTION_ENV}/"
        f"{FORCE_ACTION_ENV}/{FORCE_ENTRY_TARGET_ENV}/"
        f"{FORCE_BOUNDARY_TARGET_ENV}（缺省 off；ACTION 对为弃用方向）",
        f"- 状态一致硬证书口径: {CERT_COVERAGE_NOTE}",
        f"- 损失目标: settle（actual_settle_done − common_decision_tick；"
        "service 指标分列）",
        "",
    ]
    for rec in summary_records:
        # 仅 arm_summary 走通用臂渲染器；host_overhead 与
        # one_shot_comparison 各有专用段（后者无 request_id/predictions
        # 面，误入通用渲染器即 KeyError）。
        if rec.get("record_kind") != "arm_summary":
            continue
        pin_line = ""
        if rec.get("arm_family") == "boundary":
            pin_line = (f"，entry_pin=`{rec.get('entry_pin')}`"
                        f"（match={rec.get('entry_pin_match')}）")
        coverage = rec.get("same_state_sample_coverage") or {}
        md.append(f"## request `{rec['request_id']}` — "
                  f"{rec['arm_family']} 臂（stage={rec['stage']}"
                  f"，granularity={rec.get('granularity')}"
                  f"，status={rec['status']}）{pin_line}")
        md.append("")
        md.append(
            "| candidate | predicted_ns | settle_actual_ns | "
            "service_actual_ns | error_ns | actual_source | forced | "
            "cert | exit_code |")
        md.append("|---|---|---|---|---|---|---|---|---|")
        md.append(
            "（settle_actual_ns 为原始实测；cert 列标记该 run 是否进入同状"
            "态样本——state_mismatch/missing 的行不入损失/排序，实测值仅披"
            "露）")
        cert_by_run = (rec.get("state_cert") or {}).get(
            "status_by_run", {})
        for run_key in list(rec["predictions"]) + [
                k for k in rec["actuals_all"] if k not in rec["predictions"]]:
            error_ns = rec["errors_ns"].get(run_key)
            error_txt = ("-" if error_ns is None else str(error_ns))
            md.append(
                f"| {run_key} | {rec['predictions'].get(run_key)} | "
                f"{rec['actuals_all'].get(run_key)} | "
                f"{rec.get('actuals_service', {}).get(run_key)} | "
                f"{error_txt} | {rec['actual_sources'].get(run_key)} | "
                f"{rec['forced_selected_match'].get(run_key)} | "
                f"{cert_by_run.get(run_key)} | "
                f"{rec['exit_codes'].get(run_key)} |")
        md.append("")
        md.append(f"- 预测排序: {rec['predicted_rank']}")
        md.append(f"- 实测排序（settle 目标）: {rec['actual_rank']}")
        md.append(
            f"- 排序错误: 成对逆序 {rec['pairwise_discordant']}/"
            f"{rec['pairwise_total']}；top-1 "
            f"{'一致' if rec['top1_match'] else '不一致'}")
        md.append(
            f"- 式(24)选择损失 actual(argmin 预测) − min(actual) = "
            f"{rec['selection_loss_ns']} ns（目标=settle；样本口径："
            f"{coverage.get('caliber')}）")
        md.append(
            f"- argmin(预测) == 基线所选({rec['baseline_selected']}): "
            f"{rec['argmin_matches_baseline_chosen']}")
        if rec.get("p_reference_continuation") is not None:
            md.append(
                f"- P 参考接续（P 估价内的参考 D）: "
                f"{rec['p_reference_continuation']}；实际边界 D: "
                f"{rec.get('actual_boundary_choice')}（接续近似误差与资源"
                "误差分开归因）")
        if rec.get("missing_candidates"):
            md.append(
                f"- missing 候选（未执行/未完成，不以预测值填补）: "
                f"{rec['missing_candidates']}")
        if rec.get("state_mismatch_candidates"):
            md.append(
                f"- state_mismatch 候选（证书不匹配，不入同状态样本）: "
                f"{rec['state_mismatch_candidates']}（"
                f"{(rec.get('state_cert') or {}).get('mismatch_fields_by_run')}"
                "）")
        if rec["errors"]:
            md.append(f"- errors: {rec['errors']}")
        if rec["notes"]:
            md.append(f"- notes: {rec['notes']}")
        md.append("")

    if one_shot is not None:
        md += [
            "## 一次映射 vs 两阶段闭环对照（验收纪律第 2 类；"
            f"{one_shot['policy_env']}={one_shot['policy_value']}）",
            "",
            f"- 口径: {one_shot['metric_caliber']}",
            f"- 两阶段参照: `{one_shot['twophase_run']}`",
            f"- 一次映射臂: `{one_shot['oneshot_run']}`",
            f"- arm_class: oneshot_policy_whole_run（固定策略 whole-run "
            "臂——独立标记，不与同状态 forced 分支数据混算）",
            f"- 一次映射臂 D 行 {one_shot['oneshot_d_rows']}"
            f"（boundary_policy 披露键命中 "
            f"{one_shot['oneshot_disclosure_rows']}）；两阶段 D 边界请求 "
            f"{one_shot['d_requests']}，其中改点 "
            f"{one_shot['re_point_requests']}",
            "",
            "| 指标 | 两阶段参照 | 一次映射臂 | Δ |",
            "|---|---|---|---|",
        ]
        for row in one_shot["metrics"]:
            md.append(
                f"| {row['metric']} | {row['twophase']} | "
                f"{row['oneshot']} | {row['delta']} |")
        md.append("")
        if one_shot.get("p_face_drift"):
            md.append(
                "- P 面选择漂移请求: "
                f"{one_shot['p_face_drift']}（对照降级披露）")
        if one_shot["errors"]:
            md.append(f"- errors: {one_shot['errors']}")
        if one_shot["notes"]:
            md.append(f"- notes: {one_shot['notes']}")
        md.append("")

    md += [
        "## 宿主开销：两阶段决策 vs 固定动作回放"
        f"（待验证预算 ≤{int((OVERHEAD_BUDGET_RATIO - 1) * 100)}%）",
        "",
        f"- 口径: {overhead['metric_caliber']}",
        f"- arm_class: fixed_policy_whole_run（replay 臂——固定动作 "
        "whole-run 消融，独立标记，不与同状态 forced 分支数据混算）",
        "",
        "| run | mode | rows | decision_wall_total_ns | per_row_ns | "
        "source |",
        "|---|---|---|---|---|---|",
    ]
    for entry in overhead["runs"]:
        md.append(
            f"| {entry['label']} | {entry['mode']} | "
            f"{entry['decision_log_rows']} | "
            f"{entry['decision_wall_total_ns']} | "
            f"{entry['per_row_ns']} | {entry['source']} |")
    joint_side = overhead.get("joint_side") or {}
    replay_side = overhead.get("replay_side") or {}
    md += [
        "",
        f"- 决策侧（joint 模式，两阶段决策开启）: "
        f"{joint_side.get('per_row_ns_mean')} ns/行"
        f"（{len(joint_side.get('labels', []))} runs）",
        f"- 回放侧: {replay_side.get('per_row_ns_mean')} ns/行"
        f"（basis: {replay_side.get('basis')}）",
        f"- 比值（决策侧/回放侧）: {overhead['ratio_decision_wall_per_row']}"
        f"（差值 {overhead['delta_ns_per_row']} ns/行）",
        f"- 预算 ≤{OVERHEAD_BUDGET_RATIO}: "
        f"{overhead['verdict']}（budget_met={overhead['budget_met']}）——"
        f"{overhead['verdict_note']}",
        "",
    ]

    text = "\n".join(md)
    if dry_run:
        print("[probe][dry-run] no files written; summary follows")
        print(text)
        print("[probe][dry-run] jsonl records:")
        for rec in summary_records:
            print(json.dumps(rec, ensure_ascii=False))
        return 0 if overall_status == "ok" else 1

    plan_dir = Path(plan["plan_path"]).parent
    md_path = plan_dir / "probe_summary.md"
    jsonl_path = plan_dir / "probe_summary.jsonl"
    with open(md_path, "w", encoding="utf-8") as sink:
        sink.write(text)
    with open(jsonl_path, "w", encoding="utf-8") as sink:
        for rec in summary_records:
            sink.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[probe] summary written: {md_path}")
    print(f"[probe] summary written: {jsonl_path}")
    print(f"[probe] overall status: {overall_status}")
    return 0 if overall_status == "ok" else 1


def cmd_summarize(args) -> int:
    plan_path = Path(args.plan).resolve()
    if not plan_path.is_file():
        raise ProbeError(f"plan file missing: {plan_path}")
    with open(plan_path, encoding="utf-8") as source:
        plan = json.load(source)
    if not isinstance(plan, dict) or "targets" not in plan:
        raise ProbeError(f"plan file malformed (no 'targets'): {plan_path}")
    return summarize_plan(plan, args.dry_run)


# ---------------------------------------------------------------------------
# selftest：/tmp 小 fixture 端到端验证（命令生成 + 日志解析 + 损失计算
# + 边界同状态臂 + 宿主开销聚合 + D17 世代核验；不跑真实仿真）
# ---------------------------------------------------------------------------

def _write_log(run_dir: Path, records: list):
    """把记录列表写成最小决策日志 run（含 exit_code=0）。

    允许一层嵌套列表（便捷拼装 completion/merge_done 记录组）。
    """
    results = run_dir / "results"
    results.mkdir(parents=True, exist_ok=True)
    flat = []
    for record in records:
        if isinstance(record, list):
            flat.extend(record)
        else:
            flat.append(record)
    with open(results / "online_decision_log.jsonl", "w",
              encoding="utf-8") as sink:
        for record in flat:
            sink.write(json.dumps(record) + "\n")
    # Modern synthetic runs that carry settle events get the same manifest
    # shape as production; legacy completion-only fixtures intentionally keep
    # the old fallback path.  Settlement evidence is accepted only through
    # bridge/...['prediction'].
    has_settle = any(
        isinstance(record, dict) and record.get("kind") == "settle_done"
        for record in flat)
    if has_settle:
        manifest_dir = run_dir / "bridge"
        manifest_dir.mkdir(parents=True, exist_ok=True)
        (manifest_dir / "joint_mechanism_manifest.json").write_text(
            json.dumps({"prediction": {
                "settlement_policy": "consolidate"}}),
            encoding="utf-8")
    (run_dir / "exit_code").write_text("0\n", encoding="utf-8")


def _boundary_record(request_id: str, tick: int, candidates: list[dict],
                     selected: str, stage: str = "P",
                     p_instance: int | None = None,
                     chosen_instance: int = 0,
                     boundary_policy: str | None = None,
                     transfer_shards: list | None = None) -> dict:
    """合成一条边界决策记录（schema 同实证样例的最小子集）。

    可选参数（缺省缺席 = 既有场景零漂移）：p_instance = P 行
    joint_instance_index 披露；chosen_instance/boundary_policy/
    transfer_shards 仅 D 行消费（chosen_instance_index/boundary_policy/
    prefill_decode_transfer.shards——one-shot 场景夹具用）。
    """
    causal_prestate = {
        "schema_version": "joint-causal-prestate-v1",
        "stage": stage,
        "request_id": request_id,
        "decision_snapshot": {
            "schema_version": "snapshot-v1-20260929",
            "snapshot_id": f"fixture:{request_id}:{stage}:{tick}",
            "now_ns": tick,
            "delivery_sequence": 1,
            "event_watermark": 0,
            "versions": {
                "runtime": [],
                "layout": 0,
                "reservation": [],
                "history": [0, 0],
                "hardware": "fixture-hardware-v1",
                "model": current_model_semantics_version(),
                "ledger": {"version": 0, "event_watermark": 0},
            },
            "observed_tick": tick,
            "observation_lag_ns": 0,
            "arrivals": [],
            "request_queues": {
                "instances": {},
                "exposed_decode_steps": {},
            },
            "work_index": {
                "by_link": {},
                "by_hbm_rank": {},
                "by_pool_port": {},
                "by_rank_nodes": {},
            },
            "resource_state": {
                "arbitration": {},
                "throughput_observation": None,
                "slot_gate": {},
                "physical_capacity": {},
            },
            "existing_unknown_work": {},
            "capacity_view": {"instances": {}},
            "projection_log": [],
        },
        "work_remaining": {
            "instances": [],
            "pending_admissions": [],
            "arrival_heap": [],
            "in_flight": {},
            "runtimes": [],
            "arrived_request_count": 0,
            "completed_requests": 0,
        },
        "directory": {
            "present": True,
            "schema_version": "fixture-directory-v1",
            "model_version": "fixture-model-v1",
            "tp_degree": 1,
            "version": 1,
            "entries": [],
            "inflight_copies": [],
        },
        "capacity": {
            "present": True,
            "ledgers_by_instance": {},
            "summary_version": 1,
            "summary_error_sources": [],
            "summary": {
                "present": True,
                "snapshot_version": 1,
                "error_sources": [],
                "instances": {},
            },
            "maintenance": {},
            "stage_epochs": [],
        },
        "arbitration": {
            "noc_flows": [],
            "noc_owners": {},
            "pool_ports": {"counts": {}, "owners": {}},
            "hbm_ports": [],
            "hbm_owners": {},
            "telemetry": {
                "rates": {},
                "flow_counts": {},
                "epoch_count": 0,
                "sample_count": 0,
                "last_tick_ns": 0,
                "window_broken": False,
                "last_window_bounds_ns": None,
            },
            "quota": {},
        },
        "history": {
            "length": {
                "version": 0,
                "unit": "decode_forward_steps",
                "run_completed_samples": 0,
                "run_completed_sum": 0,
                "run_max_observed": 0,
                "run_censored_count": 0,
                "run_censored_progress_sum": 0,
                "histogram_bucket_width": 1,
                "histogram_bucket_count": 1,
                "histogram_overflow_tail_count": 0,
                "min_samples_per_level": 1,
                "cold_start_prior_remainder": 32.0,
                "cold_start_tail_distribution": "geometric",
            },
            "arrivals": {
                "version": 0,
                "arrival_bucket_delta_ns": 1,
                "open_bucket_rule": "full_buckets_only",
                "observed_until_ns": None,
                "open_bucket_index": None,
                "categories": {},
                "excluded_event_counts": {},
            },
            "efficiency": {
                "efficiency_factors": {},
                "updates": {},
                "rejected": {},
                "anomalous": {},
                "contended": {},
                "duplicate_samples": 0,
                "stale_samples": 0,
                "residuals": {},
                "hardware_versions": [],
                "ewma_weight_source": "params.service_efficiency_ewma_weight",
            },
            "service_samples": {
                "version": 0,
                "efficiency_factors": {},
                "updates": {},
                "rejected": {},
                "anomalous": {},
                "contended": {},
                "residuals": {},
                "hardware_versions": [],
                "pending": {},
            },
            "horizon": {"version": 1},
            "epoch_log": {
                "version": 0,
                "reset_rule": "no_reset_quarantine",
                "quarantine_ns": 0,
                "epochs": [],
            },
        },
        "versions": {
            "kv_ledger_epoch": 1,
            "directory_version": 1,
            "stage_epochs": [],
            "length": 1,
            "arrivals": 1,
            "efficiency": 1,
        },
    }
    decision = {
        "allowed_actions": "all",
        "applicable_actions": [c["action"] for c in candidates
                               if c.get("applicable")],
        "candidates": candidates,
        "selected_action": selected,
        "joint_action": selected,
        "decision_stage": stage,
        "decision_epoch": 0,
        "model_semantics_version": current_model_semantics_version(),
        "service_factors": {"prefill": 1.0, "decode": 1.0},
        "causal_prestate": causal_prestate,
    }
    if p_instance is not None:
        decision["joint_instance_index"] = p_instance
    if stage == "D":
        boundary = {
            "boundary_wait_kind": "fixture",
            "candidates": candidates,
            "chosen_action": selected,
            "chosen_instance_index": chosen_instance,
            "stage_epoch": 0,
            "state_versions": {"kv_ledger_epoch": 0,
                               "directory_version": 0, "stage_epochs": []},
            "inherited_obligations": [],
            "decode_steps": 1,
            "causal_prestate": causal_prestate,
        }
        if boundary_policy is not None:
            boundary["boundary_policy"] = boundary_policy
        decision = {
            "joint_action": selected,
            "boundary_decision": boundary,
            "model_semantics_version": current_model_semantics_version(),
        }
        if transfer_shards is not None:
            decision["prefill_decode_transfer"] = {
                "shards": transfer_shards}
    return {
        "kind": "joint_admission" if stage == "P" else "decode",
        "request_id": request_id,
        "seq": 1,
        "tick": tick,
        "priority": 0,
        "decision": decision,
    }


def _completion_records(request_id: str, completion_tick,
                        merge_done_tick=None) -> list[dict]:
    records = []
    if completion_tick is not None:
        records.append({
            "kind": "completion", "request_id": request_id, "seq": 2,
            "tick": completion_tick, "priority": 0, "decision": {}})
    if merge_done_tick is not None:
        records.append({
            "kind": "merge_done", "request_id": request_id, "seq": 3,
            "tick": merge_done_tick, "priority": 0, "decision": {}})
    return records


def _settle_records(request_id: str, tick: int, *, service_done_ns,
                    settle_done_ns, merge_done_ns=None,
                    merge_payload_bytes=0,
                    settlement_policy="consolidate") -> list[dict]:
    """W5 settle 事件夹具（service/settle 分采钉测用）。零载荷形态 =
    merge_payload_bytes=0 且 merge_done_ns=None（settle==service 合法）。"""
    return [{
        "kind": "settle_done", "request_id": request_id, "seq": 4,
        "tick": settle_done_ns, "priority": 0,
        "decision": {
            "settle_done_ns": settle_done_ns,
            "service_done_ns": service_done_ns,
            "settle_committed_ns": service_done_ns,
            "merge_done_ns": merge_done_ns,
            "merge_payload_bytes": merge_payload_bytes,
            "settlement_policy": settlement_policy,
            "settle_state": "SETTLE_DONE",
        },
    }]


def _face_candidates_from_plan(target: dict, stage: str) -> list[dict]:
    """从 plan 的 boundaries[stage] 双维面重建候选列表（selftest 夹具用：
    强制 run 写入与基线**同一冻结候选面**——W8 状态一致硬证书的前提，
    旧夹具的"instance_index 恒 0"退化面会在证书比对下显式 mismatch）。"""
    face = (target.get("boundaries") or {}).get(stage) or {}
    out = []
    for inst, action in face.get("instance_actions", []):
        key = candidate_key(inst, action)
        out.append({
            "action": action, "applicable": True,
            "cost_ns": face.get("predictions", {}).get(key),
            "measured": None, "instance_index": inst})
        out[-1]["is_continuation"] = key in face.get("continuations", ())
    return out


def _write_stats(run_dir: Path, per_row_ns: float):
    """合成 results/online_stats.jsonl 汇总行（predictor_cost 计时字段）。

    total_ns = per_row_ns × 决策日志实发行数（分母与产物一致，比值确定）。
    """
    log_path = decision_log_path(run_dir)
    rows = sum(1 for _ in _iter_log_records(log_path, []))
    total = int(round(per_row_ns * rows))
    run_dir.joinpath("results").mkdir(parents=True, exist_ok=True)
    row = {
        "summary": True,
        "mode": "fixture",
        "predictor_cost": {
            "decision_stages": {
                "P": {"count": max(1, rows // 2),
                      "total_ns": int(total * 0.6)},
                "D": {"count": max(1, rows - rows // 2),
                      "total_ns": total - int(total * 0.6)},
            },
            "decision_wall_ns": {"count": rows, "total_ns": total},
            "summary_maintenance": {"count": 1, "total_ns": 7,
                                    "max_ns": 7},
            "predictor_total_ns": total + 7,
            "host_wall_ns": 1000000,
            "host_share_ppm": (total + 7) * 1000000 // 1000000,
            "peak_rss": {"bytes": 1, "source": "fixture"},
            "decision_log_bytes": 999,
            "backend_events": {"batches": 1},
        },
    }
    with open(run_dir / "results" / "online_stats.jsonl", "w",
              encoding="utf-8") as sink:
        sink.write(json.dumps(row) + "\n")


def _summarize_capture(plan: dict) -> list[dict]:
    """selftest 内部复用 summarize_plan 的记录（不打文件）。"""
    records = []
    import io
    import contextlib
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        summarize_plan(plan, dry_run=True)
    for line in buf.getvalue().splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                records.append(json.loads(line))
            except ValueError:
                # 非 JSON 行静默跳过——selftest 夹具容错面，非生产路径。
                pass
    return records


def cmd_selftest(args) -> int:
    passed = {"n": 0}

    def ck(cond, label):
        assert cond, label
        passed["n"] += 1

    with tempfile.TemporaryDirectory(
            prefix="joint_probe_selftest_", dir="/tmp") as tmp:
        tmp = Path(tmp)

        # ============================================================
        # 场景 1：legacy P-only 基线 + --arms entry（旧路径回归）
        # ============================================================
        # 预测排序与实测故意相反 → 排序错误 1 对、top-1 不一致、
        # 式(24)选择损失 = actual(argmin 预测) − min(actual) = 60 − 50 = 10ns
        #（merge_done 口径修正后 copy=55 → 损失 5，见下）。
        baseline = tmp / "baseline_run"
        _write_log(baseline, [
            _boundary_record("session_9_request_9", 1000, [
                {"action": "stay", "applicable": True, "cost_ns": 10,
                 "measured": None, "instance_index": 0},
                {"action": "copy", "applicable": True, "cost_ns": 20,
                 "measured": None, "instance_index": 1}],
                selected="stay"),
            _completion_records("session_9_request_9", 1060),  # actual 60
            _boundary_record("session_9_request_8", 2000, [
                {"action": "recompute", "applicable": True, "cost_ns": 5,
                 "measured": None, "instance_index": 0},
                {"action": "copy", "applicable": True, "cost_ns": 9,
                 "measured": None, "instance_index": 0}],
                selected="recompute"),
            _completion_records("session_9_request_8", 2010),  # actual 10
        ])

        # 缺省 both 对 P-only 基线 fail-closed（D 面缺失，防误生成）。
        rc = main([
            "generate", "--baseline-run", str(baseline),
            "--evidence-root", str(tmp / "probe"), "--run-id", "selftest0",
            "--dry-run"])
        ck(rc == 2, f"P-only baseline + default arms should fail-closed, "
                    f"got {rc}")

        rc = main([
            "generate", "--baseline-run", str(baseline),
            "--evidence-root", str(tmp / "probe"), "--run-id", "selftest",
            "--arms", "entry", "--dry-run"])
        ck(rc == 0, "generate --arms entry --dry-run failed")
        rc = main([
            "generate", "--baseline-run", str(baseline),
            "--evidence-root", str(tmp / "probe"), "--run-id", "selftest",
            "--arms", "entry"])
        ck(rc == 0, "generate --arms entry failed")
        plan_path = tmp / "probe" / "selftest" / "probe_plan.json"
        with open(plan_path, encoding="utf-8") as source:
            plan = json.load(source)
        ck(plan["total_commands"] == 4, plan["total_commands"])
        ck(plan["arm_families"] == ["entry"], plan["arm_families"])
        ck(len(plan["targets"][0]["arms"]) == 1
           and plan["targets"][0]["arms"][0]["arm_family"] == "entry",
           plan["targets"][0]["arms"])
        env0 = plan["targets"][0]["arms"][0]["runs"][0]["env"]
        ck(env0[FORCE_REQUEST_ENV] == "session_9_request_9", env0)
        ck(env0[FORCE_ENTRY_ACTION_ENV] in ("stay", "copy"), env0)
        # 修复轮三互斥合同钉测：entry 臂 env 恰不含 BOUNDARY_ACTION。
        ck(FORCE_ACTION_ENV not in env0, env0)
        script_text = (plan_path.parent / "probe_commands.sh").read_text(
            encoding="utf-8")
        ck(f"{FORCE_REQUEST_ENV}=session_9_request_9" in script_text,
           script_text[:400])
        ck(f"{FORCE_ENTRY_ACTION_ENV}=copy" in script_text, script_text[:400])
        # 互斥合同脚本级钉测：entry-only 计划不出现 BOUNDARY_ACTION=。
        ck(f"{FORCE_ACTION_ENV}=" not in script_text, script_text[:400])
        ck("run_online_strategy.sh" in script_text, script_text[:400])
        ck("run_probe_arm" in script_text and "restore_pointer" in (
            script_text), script_text[:400])

        # —— 合成各候选 run 的强制结果（同状态：同边界 tick，只换选择）——
        actual_map = {
            ("session_9_request_9", "stay"): 60,
            ("session_9_request_9", "copy"): 50,  # merge_done 晚于 completion
            ("session_9_request_8", "recompute"): 10,
            ("session_9_request_8", "copy"): 30,
        }
        for target in plan["targets"]:
            for arm in target["arms"]:
                for run in arm["runs"]:
                    rid = target["request_id"]
                    action = run["action"]
                    delta = actual_map[(rid, action)]
                    _write_log(Path(run["run_dir"]), [
                        # 同状态（W8 硬证书前提）：冻结候选面与基线逐位
                        # 相同（instance/cost），只换最终选择。
                        _boundary_record(
                            rid, target["boundary_tick"],
                            _face_candidates_from_plan(target, "P"),
                            selected=action),  # 强制动作生效
                        _completion_records(
                            rid, target["boundary_tick"] + delta,
                            merge_done_tick=(
                                target["boundary_tick"] + delta + 5
                                if action == "copy"
                                else None)),
                    ])

        rc = main(["summarize", "--plan", str(plan_path), "--dry-run"])
        # 退出码口径：数据有效（强制生效+实测齐全）即 0——排序错误是测量
        # 发现不是工具失败；degraded/error（强制未生效/解析失败）才非零。
        ck(rc == 0, f"expected 0 for healthy fixture, got {rc}")
        records = [r for r in _summarize_capture(plan)
                   if r.get("record_kind") == "arm_summary"]
        by_rid = {r["request_id"]: r for r in records}
        r9 = by_rid["session_9_request_9"]
        # copy 实测 = max(completion, merge_done) − 边界 = 50+5 = 55
        #（completion_delta 口径取 merge_done（晚 5ns）→ copy=55）。
        ck(r9["actuals"]["copy"] == 55, r9["actuals"])
        ck(r9["selection_loss_ns"] == 5, r9)  # actual(stay)=60 − min=55
        ck(r9["pairwise_discordant"] == 1 and r9["pairwise_total"] == 1, r9)
        ck(r9["top1_match"] is False, r9)
        ck(r9["status"] == "ok", r9["status"])
        ck(r9["actual_sources"]["copy"] == "legacy_max_completion_merge",
           r9["actual_sources"])
        ck(r9["state_cert"]["status_by_run"]
           == {"stay": "match", "copy": "match"}, r9["state_cert"])
        ck(r9["errors_ns"] == {"stay": 10 - 60, "copy": 20 - 55},
           r9["errors_ns"])
        r8 = by_rid["session_9_request_8"]
        ck(r8["selection_loss_ns"] == 0, r8)  # argmin 预测 = 实测最优
        ck(r8["top1_match"] is True, r8)
        ck(r8["arm_family"] == "entry" and r8["entry_pin"] is None, r8)

        # —— 旧版 plan 形态（无 arms 键）仍可 summarize（兼容钉测）——
        legacy_plan = dict(plan)
        legacy_plan["stage"] = "P"
        legacy_plan["targets"] = [{
            "request_id": t["request_id"],
            "boundary_tick": t["boundary_tick"],
            "boundary_seq": t["boundary_seq"],
            "baseline_selected": t["baseline_selected"],
            "applicable": t["applicable"],
            "predictions": t["predictions"],
            "runs": t["arms"][0]["runs"],
        } for t in plan["targets"]]
        legacy_path = tmp / "probe" / "selftest" / "legacy_plan.json"
        legacy_plan["plan_path"] = str(legacy_path)
        with open(legacy_path, "w", encoding="utf-8") as sink:
            json.dump(legacy_plan, sink, indent=1)
        rc = main(["summarize", "--plan", str(legacy_path), "--dry-run"])
        ck(rc == 0, f"legacy plan summarize rc={rc}")
        legacy_records = [
            r for r in _summarize_capture(legacy_plan)
            if r.get("record_kind") == "arm_summary"]
        ck(len(legacy_records) == 2
           and all(r["arm_family"] == "entry" for r in legacy_records),
           legacy_records)
        ck({r["request_id"] for r in legacy_records}
           == {"session_9_request_9", "session_9_request_8"},
           legacy_records)

        # —— 防御式：强制未生效 + completion 缺失 → error 且报清楚 ——
        # runs[0] = stay 臂：写入 selected=copy（钩子未生效形态），并整
        # 日志缺 completion 事件。
        bad_dir = Path(plan["targets"][0]["arms"][0]["runs"][0]["run_dir"])
        _write_log(bad_dir, [
            _boundary_record("session_9_request_9", 1000, [
                {"action": "stay", "applicable": True, "cost_ns": 10,
                 "measured": None}, {"action": "copy", "applicable": True,
                                     "cost_ns": 20, "measured": None}],
                selected="copy"),
        ])
        records = [r for r in _summarize_capture(plan)
                   if r.get("record_kind") == "arm_summary"]
        r9 = records[0]
        # W8 口径：强制未生效（选择≠强制身份）+ completion 缺失 ⇒ 该候选
        # 无受控真值——run 标 missing（不以预测值填补）、臂降级 degraded；
        # 错误面（no completion）与注记面（强制未生效）均逐条报清楚。
        ck(r9["status"] == "degraded" and r9["errors"], r9)
        ck("stay" in r9["missing_candidates"], r9["missing_candidates"])
        ck(any("forced identity" in n and "NOT selected" in n
               for n in r9["notes"]), r9["notes"])
        ck(any("no completion event" in e or "cannot compute settle" in e
               for e in r9["errors"]), r9)
        print("[probe][selftest] scenario 1 PASS (legacy entry arms + "
              "old-plan compat + defensive mismatch & missing-field)")

        # ============================================================
        # 场景 2：双面基线（P+D）—— 缺省 both：entry + boundary 臂 +
        # 回放臂 + 宿主开销聚合
        # ============================================================
        # r3: P@500 preds stay=20/copy=10 → 基线选 copy（=entry pin）；
        #     D@900 preds stay=40/copy=30 → 基线选 copy。
        # r5: P@700 preds recompute=5/copy=9 → 选 recompute（pin）；
        #     D@1100 preds copy=8/recompute=12 → 选 copy。
        baseline2 = tmp / "baseline2_run"

        def _faces(rid, p_tick, p_cands, p_sel, d_tick, d_cands, d_sel):
            return [
                _boundary_record(rid, p_tick, p_cands, selected=p_sel,
                                 stage="P"),
                _completion_records(rid, p_tick + 40),
                _boundary_record(rid, d_tick, d_cands, selected=d_sel,
                                 stage="D"),
                _completion_records(rid, d_tick + 50),
            ]

        _write_log(baseline2, _faces(
            "session_7_request_3", 500,
            [{"action": "stay", "applicable": True, "cost_ns": 20,
              "measured": None, "instance_index": 0},
             {"action": "copy", "applicable": True, "cost_ns": 10,
              "measured": None, "instance_index": 1}],
            "copy",
            900,
            [{"action": "stay", "applicable": True, "cost_ns": 40,
              "measured": None, "instance_index": 0},
             {"action": "copy", "applicable": True, "cost_ns": 30,
              "measured": None, "instance_index": 2}],
            "copy") + _faces(
            "session_7_request_5", 700,
            [{"action": "recompute", "applicable": True, "cost_ns": 5,
              "measured": None, "instance_index": 0},
             {"action": "copy", "applicable": True, "cost_ns": 9,
              "measured": None, "instance_index": 0}],
            "recompute",
            1100,
            [{"action": "copy", "applicable": True, "cost_ns": 8,
              "measured": None, "instance_index": 1},
             {"action": "recompute", "applicable": True, "cost_ns": 12,
              "measured": None, "instance_index": 0}],
            "copy"))

        # 旧接口映射：--stage D → boundary 臂（带 entry pin）。
        rc = main([
            "generate", "--baseline-run", str(baseline2),
            "--evidence-root", str(tmp / "probe2"), "--run-id", "stagemap",
            "--stage", "D", "--replay-arm", "none", "--dry-run"])
        ck(rc == 0, "--stage D mapping dry-run failed")

        rc = main([
            "generate", "--baseline-run", str(baseline2),
            "--evidence-root", str(tmp / "probe2"), "--run-id", "selftest2",
            "--env", "JOINT_ABLATION_COMBO=TJE"])
        ck(rc == 0, "generate both arms failed")
        plan2_path = tmp / "probe2" / "selftest2" / "probe_plan.json"
        with open(plan2_path, encoding="utf-8") as source:
            plan2 = json.load(source)
        ck(plan2["arm_families"] == ["entry", "boundary"],
           plan2["arm_families"])
        ck(plan2["total_commands"] == 8, plan2["total_commands"])
        t3 = plan2["targets"][0]
        ck(t3["boundaries"]["P"]["selected"] == "copy", t3["boundaries"])
        ck(t3["boundaries"]["D"]["applicable"] == ["stay", "copy"],
           t3["boundaries"])
        arms_by_family = {a["arm_family"]: a for a in t3["arms"]}
        ck(arms_by_family["entry"]["stage"] == "P", arms_by_family)
        ck(arms_by_family["boundary"]["entry_pin"] == "copy",
           arms_by_family["boundary"])
        b_stay = arms_by_family["boundary"]["runs"][0]
        ck(b_stay["env"][FORCE_REQUEST_ENV] == "session_7_request_3",
           b_stay["env"])
        ck(b_stay["env"][FORCE_ACTION_ENV] == "stay", b_stay["env"])
        # 修复轮三互斥合同钉测：boundary 臂 env 恰不含 ENTRY_ACTION。
        ck(FORCE_ENTRY_ACTION_ENV not in b_stay["env"], b_stay["env"])
        ck(b_stay["env"]["JOINT_ABLATION_COMBO"] == "TJE", b_stay["env"])
        ck(b_stay["run_dir"].endswith("session_7_request_3__D__stay"),
           b_stay["run_dir"])
        e_stay = arms_by_family["entry"]["runs"][0]
        ck(e_stay["run_dir"].endswith("session_7_request_3__P__stay"),
           e_stay["run_dir"])
        ck(e_stay["env"][FORCE_ENTRY_ACTION_ENV] == "stay", e_stay["env"])
        # 互斥合同：entry 臂 env 恰不含 BOUNDARY_ACTION。
        ck(FORCE_ACTION_ENV not in e_stay["env"], e_stay["env"])
        # 互斥模板全量钉测（两族全部 run）：entry 臂恰含 ENTRY、恰不含
        # BOUNDARY_ACTION；boundary 臂恰含 BOUNDARY_ACTION、恰不含 ENTRY。
        for tgt in plan2["targets"]:
            for arm in tgt["arms"]:
                for run in arm["runs"]:
                    env = run["env"]
                    if arm["arm_family"] == "entry":
                        ck(FORCE_ENTRY_ACTION_ENV in env
                           and FORCE_ACTION_ENV not in env, env)
                    else:
                        ck(FORCE_ACTION_ENV in env
                           and FORCE_ENTRY_ACTION_ENV not in env, env)
        replay = plan2["replay_arm"]
        ck(replay is not None and replay["mode"] == "load-first", replay)
        ck(replay["env"][REPLAY_MODE_ENV] == "load-first", replay["env"])
        ck(_COMBO_ENV not in replay["env"], replay["env"])
        ck(replay["dropped_env"] == [_COMBO_ENV], replay["dropped_env"])
        ck(not any(k.startswith("JOINT_FORCE")
                   for k in replay["env"]), replay["env"])
        script2 = (plan2_path.parent / "probe_commands.sh").read_text(
            encoding="utf-8")
        ck(f"{FORCE_ENTRY_ACTION_ENV}=copy" in script2, script2)
        ck(f"{FORCE_ACTION_ENV}=copy" in script2, script2)
        ck("replay_load_first" in script2
           and f"{REPLAY_MODE_ENV}=load-first" in script2, script2)
        # 互斥合同脚本级钉测：任一逐臂调用行不得同时出现两 ACTION env
        # （每臂 env 对渲染为单行，行级检查即逐臂检查）。
        for line in script2.splitlines():
            if f"{FORCE_ENTRY_ACTION_ENV}=" in line:
                ck(f"{FORCE_ACTION_ENV}=" not in line, line)
            if f"{FORCE_ACTION_ENV}=" in line:
                ck(f"{FORCE_ENTRY_ACTION_ENV}=" not in line, line)

        # —— 合成各臂 run 日志 ——
        # 下面两表是 completion 偏移（delta）；merge 动作（copy）另有
        # merge_done = delta+5，completion_delta 口径实测值 = delta+5
        # （同场景 1 的 merge_done 口径钉测）。
        # entry 臂实测：r3 stay=60/copy=75；r5 recompute=10/copy=35。
        # boundary 臂实测：r3 stay=60/copy=75；r5 copy=90/recompute=110。
        entry_deltas = {
            ("session_7_request_3", "stay"): 60,
            ("session_7_request_3", "copy"): 70,
            ("session_7_request_5", "recompute"): 10,
            ("session_7_request_5", "copy"): 30,
        }
        boundary_deltas = {
            ("session_7_request_3", "stay"): 60,
            ("session_7_request_3", "copy"): 70,
            ("session_7_request_5", "copy"): 85,
            ("session_7_request_5", "recompute"): 110,
        }
        for target in plan2["targets"]:
            rid = target["request_id"]
            p_face = target["boundaries"]["P"]
            d_face = target["boundaries"]["D"]
            for arm in target["arms"]:
                for run in arm["runs"]:
                    action = run["action"]
                    if arm["arm_family"] == "entry":
                        base_tick = p_face["tick"]
                        delta = entry_deltas[(rid, action)]
                        records = [
                            # 同状态（W8 硬证书前提）：候选面与基线逐位同。
                            _boundary_record(
                                rid, base_tick,
                                _face_candidates_from_plan(target, "P"),
                                selected=action),
                            _completion_records(
                                rid, base_tick + delta,
                                merge_done_tick=(
                                    base_tick + delta + 5
                                    if action == "copy"
                                    else None))]
                    else:
                        base_tick = d_face["tick"]
                        pin = arm["entry_pin"]
                        delta = boundary_deltas[(rid, action)]
                        records = [
                            # P 入口选择 = pin（boundary 臂按互斥合同不
                            # 强制 P，同 trace 复现基线选择——同状态前提
                            # 成立形态；候选面同基线）
                            _boundary_record(
                                rid, p_face["tick"],
                                _face_candidates_from_plan(target, "P"),
                                selected=pin),
                            _boundary_record(
                                rid, base_tick,
                                _face_candidates_from_plan(target, "D"),
                                selected=action, stage="D"),
                            _completion_records(
                                rid, base_tick + delta,
                                merge_done_tick=(
                                    base_tick + delta + 5
                                    if action == "copy"
                                    else None))]
                    _write_log(Path(run["run_dir"]), records)
        # 回放臂 + 基线：最小日志（completion 齐全即可计时）。
        _write_log(Path(replay["run_dir"]), [
            _boundary_record("session_7_request_3", 500, [
                {"action": "stay", "applicable": True, "cost_ns": 20,
                 "measured": None, "instance_index": 0}],
                selected="stay"),
            _completion_records("session_7_request_3", 560),
        ])
        # 宿主开销计时：joint 侧（基线 + forced 臂）100 ns/行，
        # 回放臂 100 ns/行 → 比值 1.0（预算达成）。
        _write_stats(Path(plan2["baseline_run"]), 100)
        for target in plan2["targets"]:
            for arm in target["arms"]:
                for run in arm["runs"]:
                    _write_stats(Path(run["run_dir"]), 100)
        _write_stats(Path(replay["run_dir"]), 100)

        rc = main(["summarize", "--plan", str(plan2_path), "--dry-run"])
        ck(rc == 0, f"both-arms summarize rc={rc}")
        recs2 = _summarize_capture(plan2)
        arm_recs = [r for r in recs2
                    if r.get("record_kind") == "arm_summary"]
        ck(len(arm_recs) == 4, len(arm_recs))
        by_key = {(r["request_id"], r["arm_family"]): r for r in arm_recs}
        b3 = by_key[("session_7_request_3", "boundary")]
        ck(b3["stage"] == "D" and b3["entry_pin"] == "copy", b3)
        ck(b3["entry_pin_match"] == {"stay": True, "copy": True},
           b3["entry_pin_match"])
        ck(b3["actuals"] == {"stay": 60, "copy": 75}, b3["actuals"])
        # D preds stay=40 / copy=30 → argmin copy；actual(copy)=75，
        # min(actual)=60 → 损失 15；排序 1 对全逆；top-1 不一致。
        ck(b3["selection_loss_ns"] == 15, b3)
        ck(b3["pairwise_discordant"] == 1 and b3["pairwise_total"] == 1, b3)
        ck(b3["top1_match"] is False, b3)
        ck(b3["errors_ns"] == {"stay": 40 - 60, "copy": 30 - 75},
           b3["errors_ns"])
        ck(b3["argmin_matches_baseline_chosen"] is True, b3)
        ck(b3["status"] == "ok", b3["status"])
        b5 = by_key[("session_7_request_5", "boundary")]
        ck(b5["selection_loss_ns"] == 0 and b5["top1_match"] is True, b5)
        e3 = by_key[("session_7_request_3", "entry")]
        ck(e3["stage"] == "P" and e3["entry_pin"] is None, e3)
        ck(e3["actuals"] == {"stay": 60, "copy": 75}, e3["actuals"])
        # P preds stay=20 / copy=10 → argmin copy；actual(copy)=75，
        # min=60 → 损失 15。
        ck(e3["selection_loss_ns"] == 15, e3)

        overhead = [r for r in recs2
                    if r.get("record_kind") == "host_overhead"][0]
        ck(overhead["replay_side"]["basis"] == "load-first 臂", overhead)
        ck(abs(overhead["joint_side"]["per_row_ns_mean"] - 100.0) < 1e-6,
           overhead["joint_side"])
        ck(abs(overhead["replay_side"]["per_row_ns_mean"] - 100.0) < 1e-6,
           overhead["replay_side"])
        ck(abs(overhead["ratio_decision_wall_per_row"] - 1.0) < 1e-6,
           overhead)
        ck(overhead["budget_met"] is True and overhead["verdict"] == "met",
           overhead)
        ck(overhead["budget_ratio"] == OVERHEAD_BUDGET_RATIO, overhead)
        modes = {e["label"]: e["mode"] for e in overhead["runs"]}
        ck(modes.get("baseline") == "joint"
           and modes.get("replay_load_first") == "load-first"
           and modes.get("session_7_request_3::boundary::stay")
           == "joint+forced", modes)

        # —— 开销：回放臂更慢/更快翻转预算判定（如实披露钉测）——
        _write_stats(Path(replay["run_dir"]), 80)  # 回放 80 → 比值 1.25
        recs2 = _summarize_capture(plan2)
        overhead = [r for r in recs2
                    if r.get("record_kind") == "host_overhead"][0]
        ck(abs(overhead["ratio_decision_wall_per_row"] - 1.25) < 1e-6,
           overhead)
        ck(overhead["budget_met"] is False
           and overhead["verdict"] == "not_met", overhead)

        # —— 开销：回放臂计时缺失 → forced 回退 + 比值 None（口径披露）——
        (Path(replay["run_dir"]) / "results" / "online_stats.jsonl").unlink()
        recs2 = _summarize_capture(plan2)
        overhead = [r for r in recs2
                    if r.get("record_kind") == "host_overhead"][0]
        replay_entry = [e for e in overhead["runs"]
                        if e["label"] == "replay_load_first"][0]
        ck(replay_entry["source"] == "missing", replay_entry)
        ck(overhead["ratio_decision_wall_per_row"] is None, overhead)
        ck(overhead["verdict"] == "insufficient_data", overhead)
        ck("forced 臂回退" in overhead["replay_side"]["basis"],
           overhead["replay_side"])

        # —— 开销：单臂 online_stats 缺失 → joint_decision_metrics 回落 ——
        fallback_dir = Path(
            by_key_label_run_dir(plan2, "session_7_request_3", "entry",
                                 "stay"))
        (fallback_dir / "results" / "online_stats.jsonl").unlink()
        with open(decision_log_path(fallback_dir), "a",
                  encoding="utf-8") as sink:
            sink.write(json.dumps({
                "kind": "joint_decision_metrics", "request_id": "",
                "seq": 99, "tick": 9999, "priority": 0,
                "decision": {"admission_decision_wall_ns": {
                    "count": 1, "total_ns": 500, "avg_ns": 500,
                    "max_ns": 500}}}) + "\n")
        recs2 = _summarize_capture(plan2)
        overhead = [r for r in recs2
                    if r.get("record_kind") == "host_overhead"][0]
        fb_entry = [e for e in overhead["runs"] if e["run_dir"]
                    == str(fallback_dir)][0]
        ck(fb_entry["source"] == "decision_log.joint_decision_metrics"
           "(P-only)", fb_entry)
        rows = fb_entry["decision_log_rows"]
        ck(fb_entry["per_row_ns"] == round(500 / rows, 3), fb_entry)

        # —— 防御式（boundary）：entry pin 未守住 → degraded + 注记 ——
        # boundary 臂按互斥合同不强制 P 入口（P 面靠同 trace 复现基线
        # 选择）——若 P 选择漂移（pin=copy 却选了 stay），同状态前提
        # 破坏即降级。
        pinbreak_dir = Path(
            by_key_label_run_dir(plan2, "session_7_request_3", "boundary",
                                 "stay"))
        d_tick = 900
        _write_log(pinbreak_dir, [
            _boundary_record("session_7_request_3", 500, [
                {"action": "stay", "applicable": True, "cost_ns": 20,
                 "measured": None, "instance_index": 0},
                {"action": "copy", "applicable": True, "cost_ns": 10,
                 "measured": None, "instance_index": 1}],
                selected="stay"),  # != pin "copy"
            _boundary_record("session_7_request_3", d_tick, [
                {"action": "stay", "applicable": True, "cost_ns": 40,
                 "measured": None, "instance_index": 0},
                {"action": "copy", "applicable": True, "cost_ns": 30,
                 "measured": None, "instance_index": 2}],
                selected="stay", stage="D"),
            _completion_records("session_7_request_3", d_tick + 60),
        ])
        recs2 = _summarize_capture(plan2)
        arm_recs = [r for r in recs2
                    if r.get("record_kind") == "arm_summary"]
        b3 = [r for r in arm_recs
              if r["request_id"] == "session_7_request_3"
              and r["arm_family"] == "boundary"][0]
        ck(b3["status"] == "degraded", b3)
        ck(b3["entry_pin_match"].get("stay") is False, b3)
        ck(any("entry pin" in n for n in b3["notes"]), b3["notes"])
        rc = main(["summarize", "--plan", str(plan2_path), "--dry-run"])
        ck(rc == 1, f"pin-break summarize rc={rc} (expect 1=degraded)")

        print("[probe][selftest] scenario 2 PASS (boundary same-state arms "
              "+ entry pin + replay arm + host overhead met/not_met/"
              "insufficient + P-only fallback + pin-break degradation)")

        # ============================================================
        # 场景 3：D17 世代核验（stageR7）——一致零告警 / 不一致告警文案 /
        # --strict-version 拒绝生成 / 决策行回落源 / 披露缺失
        # ============================================================
        import contextlib
        import io
        current_ver = current_model_semantics_version()
        ck(current_ver is not None, current_ver)  # 仓内 joint_config 可解析

        def _gen_baseline(name, manifest_version, row_version=None):
            """世代核验夹具：双面基线 + 可选 manifest/决策行版本披露。"""
            base = tmp / name
            records = [
                _boundary_record("session_5_request_1", 300, [
                    {"action": "stay", "applicable": True, "cost_ns": 20,
                     "measured": None, "instance_index": 0},
                    {"action": "copy", "applicable": True, "cost_ns": 10,
                     "measured": None, "instance_index": 1}],
                    selected="copy"),
                _completion_records("session_5_request_1", 340),
                _boundary_record("session_5_request_1", 380, [
                    {"action": "stay", "applicable": True, "cost_ns": 40,
                     "measured": None, "instance_index": 0},
                    {"action": "copy", "applicable": True,
                     "cost_ns": 30, "measured": None, "instance_index": 2}],
                    selected="copy", stage="D"),
                _completion_records("session_5_request_1", 430),
            ]
            if row_version is not None:
                flat = []
                for record in records:
                    if isinstance(record, list):
                        flat.extend(record)
                    else:
                        flat.append(record)
                for record in flat:
                    if record.get("kind") in ("joint_admission", "decode"):
                        record["decision"]["model_semantics_version"] = (
                            row_version)
            else:
                # This is the deliberate missing-disclosure negative fixture
                # used by scenario 3e.  _boundary_record normally mirrors a
                # real producer and fills the current version; remove that
                # optional provenance key here so the warning/strict reject
                # path is exercised with genuinely absent evidence.
                flat = []
                for record in records:
                    if isinstance(record, list):
                        flat.extend(record)
                    else:
                        flat.append(record)
                for record in flat:
                    if record.get("kind") in ("joint_admission", "decode"):
                        record.get("decision", {}).pop(
                            "model_semantics_version", None)
            _write_log(base, records)
            if manifest_version is not None:
                bridge = base / "bridge"
                bridge.mkdir(parents=True, exist_ok=True)
                with open(bridge / "joint_mechanism_manifest.json", "w",
                          encoding="utf-8") as sink:
                    json.dump({"model_semantics_version": manifest_version,
                               "fixture": True}, sink)
            return base

        def _gen_argv(run, run_id, *extra):
            return ["generate", "--baseline-run", str(run),
                    "--evidence-root", str(tmp / "probe3"),
                    "--run-id", run_id, *extra]

        def _gen_run(argv):
            """跑 generate 并捕获 stderr（告警面；返回 (rc, stderr)）。"""
            buf = io.StringIO()
            with contextlib.redirect_stderr(buf):
                rc = main(argv)
            return rc, buf.getvalue()

        # —— 3a. 版本一致：零告警 + plan 披露 match=True ——
        same_gen = _gen_baseline("baseline_same_gen",
                                 manifest_version=current_ver)
        rc, err = _gen_run(_gen_argv(same_gen, "gen_same", "--dry-run"))
        ck(rc == 0, rc)
        ck("[probe][warn]" not in err, err)
        rc, err = _gen_run(_gen_argv(same_gen, "gen_same_real"))
        ck(rc == 0, rc)
        ck("[probe][warn]" not in err, err)
        with open(tmp / "probe3" / "gen_same_real" / "probe_plan.json",
                  encoding="utf-8") as source:
            plan3 = json.load(source)
        ck(plan3["baseline_generation"]["match"] is True,
           plan3["baseline_generation"])
        ck(plan3["baseline_generation"]["baseline"] == current_ver
           and plan3["baseline_generation"]["current"] == current_ver,
           plan3["baseline_generation"])
        ck(plan3["baseline_generation"]["baseline_source"]
           == "bridge/joint_mechanism_manifest.json",
           plan3["baseline_generation"])

        # —— 3b. 不一致（manifest 源）：stderr 明确告警 + 照常生成 ——
        stale_ver = "v0-legacy-20260101"
        stale_gen = _gen_baseline("baseline_stale_gen",
                                  manifest_version=stale_ver)
        rc, err = _gen_run(_gen_argv(stale_gen, "gen_stale", "--dry-run"))
        ck(rc == 0, rc)  # 告警不阻断
        ck(f"baseline 世代 {stale_ver} ≠ 当前 {current_ver}" in err, err)
        ck("适用面可能漂移" in err and "建议用同代 baseline" in err, err)

        # —— 3c. 不一致 + --strict-version：fail-closed 拒绝（无产物目录）——
        rc, err = _gen_run(_gen_argv(stale_gen, "gen_stale_strict",
                                     "--strict-version"))
        ck(rc == 2, rc)
        ck(f"baseline 世代 {stale_ver} ≠ 当前 {current_ver}" in err, err)
        ck(not (tmp / "probe3" / "gen_stale_strict").exists(),
           "strict reject must not create output dir")

        # —— 3d. 决策行回落源（无 bridge manifest）：同告警/同拒绝 ——
        row_ver = "v2-single-stage-20260101"
        row_gen = _gen_baseline("baseline_row_gen", manifest_version=None,
                                row_version=row_ver)
        rc, err = _gen_run(_gen_argv(row_gen, "gen_row", "--dry-run"))
        ck(rc == 0, rc)
        ck(f"baseline 世代 {row_ver} ≠ 当前 {current_ver}" in err, err)
        rc, err = _gen_run(_gen_argv(row_gen, "gen_row_strict",
                                     "--strict-version"))
        ck(rc == 2, rc)
        ck(f"baseline 世代 {row_ver} ≠ 当前 {current_ver}" in err, err)

        # —— 3e. 披露缺失：无法核验 → 告警；strict 下拒绝 ——
        bare_gen = _gen_baseline("baseline_bare_gen", manifest_version=None)
        rc, err = _gen_run(_gen_argv(bare_gen, "gen_bare", "--dry-run"))
        ck(rc == 0, rc)
        ck("无法核验世代" in err, err)
        rc, err = _gen_run(_gen_argv(bare_gen, "gen_bare_strict",
                                     "--strict-version"))
        ck(rc == 2, rc)
        ck("无法核验" in err, err)

        print("[probe][selftest] scenario 3 PASS (D17 generation check: "
              "match zero-warn + mismatch warn text + strict reject + "
              "decision-log fallback + missing disclosure)")

        # ============================================================
        # 场景 4：one_shot_pair（--arms one-shot）——两阶段参照 × 一次
        # 映射 p-continuation 成对臂：plan 生成含 one-shot 臂 +
        # _one_shot_comparison 指标行（D 改点率/同址改动作率/交接字节/
        # E2E/完成率/控制开销）+ 动作位实测对账披露 + 专用 markdown 段
        # + request_metrics 缺失降级 NA。
        # ============================================================
        baseline4 = tmp / "baseline4_run"
        _write_log(baseline4, [
            _boundary_record("session_4_request_0", 50,
                             [{"action": "stay", "applicable": True,
                               "cost_ns": 20, "measured": None}],
                             selected="stay"),
            _completion_records("session_4_request_0", 90),
        ])

        def _write_pair_run(run_dir: Path, rows: list[tuple],
                            policy: str | None):
            """成对 run 夹具：同 P 面（p_instance 披露）、D 面逐行给定。"""
            records = []
            for (rid, p_tick, p_sel, p_inst, d_tick, d_sel, d_inst,
                    shards) in rows:
                records.append(_boundary_record(
                    rid, p_tick,
                    [{"action": p_sel, "applicable": True, "cost_ns": 1,
                      "measured": None}], selected=p_sel,
                    p_instance=p_inst))
                records.append(_completion_records(rid, p_tick + 40))
                records.append(_boundary_record(
                    rid, d_tick,
                    [{"action": d_sel, "applicable": True, "cost_ns": 1,
                      "measured": None}], selected=d_sel, stage="D",
                    chosen_instance=d_inst, boundary_policy=policy,
                    transfer_shards=shards))
                records.append(_completion_records(rid, d_tick + 50))
            _write_log(run_dir, records)

        def _write_request_metrics(run_dir: Path, rows: list[tuple]):
            path = Path(run_dir) / "request_metrics.csv"
            lines = ["request_id,terminal_status,e2e_ns,first_token_ns,"
                     "arrival_ns"]
            lines += [f"{rid},{status},{e2e},{ttft},0"
                      for rid, status, e2e, ttft in rows]
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")

        rc = main(["generate", "--baseline-run", str(baseline4),
                   "--evidence-root", str(tmp / "probe4"),
                   "--run-id", "selftest4", "--arms", "one-shot"])
        ck(rc == 0, "generate --arms one-shot failed")
        plan4_path = tmp / "probe4" / "selftest4" / "probe_plan.json"
        with open(plan4_path, encoding="utf-8") as source:
            plan4 = json.load(source)
        ck(plan4["arm_families"] == ["one-shot"], plan4["arm_families"])
        # one-shot 族无逐请求目标（total_commands 只计 targets 内命令；
        # 成对两 run 记在 one_shot_pair）。
        ck(plan4["total_commands"] == 0, plan4["total_commands"])
        pair = plan4["one_shot_pair"]
        ck(pair["twophase_run"]["label"] == "twophase_reference", pair)
        ck(pair["oneshot_run"]["label"] == "oneshot_p_continuation", pair)
        ck(pair["twophase_run"]["run_dir"].endswith(
            "runs/twophase_reference"), pair)
        ck(pair["oneshot_run"]["run_dir"].endswith(
            "runs/oneshot_p_continuation"), pair)
        ck(pair["oneshot_run"]["env"].get(ONESHOT_POLICY_ENV)
           == ONESHOT_POLICY_VALUE, pair)
        ck(ONESHOT_POLICY_ENV not in pair["twophase_run"]["env"], pair)

        # —— 成对 run 夹具：同 P 面、D 面分叉（两阶段真选择 vs 一次映射
        # 延续形态；r3 一次映射臂动作位偏差形态，实测披露不报错）——
        # (rid, p_tick, p_sel, p_inst, d_tick, d_sel, d_inst, shards)
        twophase_rows = [
            ("session_4_request_1", 100, "copy", 1, 500, "copy", 0,
             [{"bytes": 128}]),  # 两阶段改点：跨实例 copy ≠ 延续 stay@1
            ("session_4_request_2", 200, "copy", 0, 600, "copy", 0,
             [{"bytes": 64}]),  # 同址动作位改点：copy ≠ 延续 stay@0
            ("session_4_request_3", 300, "stay", 0, 700, "stay", 0,
             None),  # 按延续形态提交（stay@0）
        ]
        oneshot_rows = [
            ("session_4_request_1", 100, "copy", 1, 500, "stay", 1,
             None),  # 延续形态 stay@1
            ("session_4_request_2", 200, "copy", 0, 600, "stay", 0,
             None),  # 延续形态 stay@0
            ("session_4_request_3", 300, "stay", 0, 700, "copy", 0,
             None),  # 动作位偏差形态
        ]
        _write_pair_run(Path(pair["twophase_run"]["run_dir"]),
                        twophase_rows, policy=None)
        _write_pair_run(Path(pair["oneshot_run"]["run_dir"]),
                        oneshot_rows, policy=ONESHOT_POLICY_VALUE)
        _write_request_metrics(Path(pair["twophase_run"]["run_dir"]), [
            ("session_4_request_1", "completed", 1000, 300),
            ("session_4_request_2", "completed", 2000, 600),
            ("session_4_request_3", "completed", 3000, 900)])
        _write_request_metrics(Path(pair["oneshot_run"]["run_dir"]), [
            ("session_4_request_1", "completed", 900, 250),
            ("session_4_request_2", "completed", 1800, 500),
            ("session_4_request_3", "completed", 2700, 750)])
        _write_stats(Path(pair["twophase_run"]["run_dir"]), 100)
        _write_stats(Path(pair["oneshot_run"]["run_dir"]), 100)

        recs4 = _summarize_capture(plan4)
        one_shot = [r for r in recs4
                    if r.get("record_kind") == "one_shot_comparison"][0]
        ck(one_shot["status"] == "ok", one_shot)
        ck(one_shot["oneshot_d_rows"] == 3
           and one_shot["oneshot_disclosure_rows"] == 3, one_shot)
        ck(one_shot["d_requests"] == 3
           and one_shot["re_point_requests"] == 2, one_shot)
        # 动作位实测对账底数：2/3 行 chosen_action 与派生延续形态一致
        #（r3 动作位偏差如实计数，不报错）。
        ck(one_shot["oneshot_action_match"] == 2
           and one_shot["oneshot_action_verifiable"] == 3, one_shot)
        ck(one_shot["oneshot_re_point_requests"] == 1, one_shot)
        ck(one_shot["oneshot_same_instance_action_change"] == 1, one_shot)
        metrics4 = {row["metric"]: row for row in one_shot["metrics"]}
        r1 = metrics4["D 改点率（最终 (instance,action) ≠ 延续形态）"]
        ck(r1["twophase"] == "0.666667"
           and r1["oneshot"] == "1（实测核对 2/3 一致）", r1)
        r2 = metrics4["同址改动作率（instance 不变 action 变）"]
        ck(r2["twophase"] == "0.333333"
           and r2["oneshot"] == "1（实测核对动作位 2/3 一致）", r2)
        r3 = metrics4["D 边界改点请求数 / D 边界请求数"]
        ck(r3["twophase"] == "2/3" and r3["oneshot"] == "1/3", r3)
        rh = metrics4["交接字节（P→D 迁移腿，决策日志 shards 聚合）"]
        ck(rh["twophase"] == "192" and rh["oneshot"] == "0"
           and rh["delta"] == -192, rh)
        re2e = metrics4["E2E 均值 ns（request_metrics.csv）"]
        ck(re2e["twophase"] == "2000.0" and re2e["oneshot"] == "1800.0"
           and re2e["delta"] == -200.0, re2e)
        rcr = metrics4["完成率（terminal_status=completed）"]
        ck(rcr["twophase"] == "1.0" and rcr["oneshot"] == "1.0", rcr)
        rcs = metrics4["控制开销 D 阶段 total_ns"]
        ck(rcs["twophase"] == "480" and rcs["oneshot"] == "480"
           and rcs["delta"] == 0, rcs)
        ck(any("动作位实测对账" in n and "1/3" in n
               for n in one_shot["notes"]), one_shot["notes"])

        # —— 专用 markdown 段：表行渲染非空且携带实测核对括注 ——
        rc = main(["summarize", "--plan", str(plan4_path)])
        ck(rc == 0, f"one-shot summarize rc={rc}")
        summary_md = (plan4_path.parent / "probe_summary.md").read_text(
            encoding="utf-8")
        ck("## 一次映射 vs 两阶段闭环对照" in summary_md, summary_md[:400])
        ck("twophase_reference" in summary_md
           and "oneshot_p_continuation" in summary_md, summary_md[:800])
        ck("| D 改点率（最终 (instance,action) ≠ 延续形态） | 0.666667 |"
           " 1（实测核对 2/3 一致） | NA |" in summary_md, summary_md)
        ck("动作位实测对账" in summary_md, summary_md)

        # —— request_metrics.csv 缺失：E2E/完成率降级 NA + 注记 ——
        (Path(pair["oneshot_run"]["run_dir"])
         / "request_metrics.csv").unlink()
        recs4 = _summarize_capture(plan4)
        one_shot = [r for r in recs4
                    if r.get("record_kind") == "one_shot_comparison"][0]
        ck(one_shot["status"] == "ok", one_shot)
        ck(one_shot["completion"]["oneshot"]["requests"] is None
           and one_shot["completion"]["oneshot"]["e2e_mean_ns"] is None,
           one_shot["completion"])
        metrics4 = {row["metric"]: row for row in one_shot["metrics"]}
        ck(metrics4["E2E 均值 ns（request_metrics.csv）"]["oneshot"]
           == "NA", metrics4)
        ck(metrics4["完成率（terminal_status=completed）"]["oneshot"]
           == "NA", metrics4)
        ck(any("request_metrics.csv 缺失" in n and "oneshot_p_continuation"
               in n for n in one_shot["notes"]), one_shot["notes"])

        print("[probe][selftest] scenario 4 PASS (one-shot pair arms: "
              "plan generation + metric rows + measured action "
              "reconciliation + markdown section + missing-metrics NA "
              "degrade)")

        # ============================================================
        # 场景 5（W8）：instance-action 粒度 + TARGET env + settle 事件
        # 分采（service/settle 双指标、零载荷形态）+ 状态一致硬证书
        # （全匹配 ok / 单字段漂移 state_mismatch 不入样本）+ missing 标记
        # + P 参考接续分列 + fixed_policy_arm 标记。
        # ============================================================
        baseline5 = tmp / "baseline5_run"
        # D 面：stay@0（cost 40，is_continuation=True）/ copy@1（cost 30）/
        # copy@0（cost 35）——同动作双实例（copy 最低成本实例是 1，i0:copy
        # 即"非最低成本实例"反证面）；P 面 stay@0(20)/copy@1(10)，基线选
        # copy@1（entry pin）。
        _write_log(baseline5, [
            _boundary_record("session_3_request_1", 500, [
                {"action": "stay", "applicable": True, "cost_ns": 20,
                 "measured": None, "instance_index": 0},
                {"action": "copy", "applicable": True, "cost_ns": 10,
                 "measured": None, "instance_index": 1}],
                selected="copy", p_instance=1),
            _completion_records("session_3_request_1", 540),
            _settle_records("session_3_request_1", 545,
                            service_done_ns=540, settle_done_ns=545),
            _boundary_record("session_3_request_1", 900, [
                {"action": "stay", "applicable": True, "cost_ns": 40,
                 "measured": None, "instance_index": 0,
                 "is_continuation": True},
                {"action": "copy", "applicable": True, "cost_ns": 35,
                 "measured": None, "instance_index": 0},
                {"action": "copy", "applicable": True, "cost_ns": 30,
                 "measured": None, "instance_index": 1}],
                selected="copy", stage="D", chosen_instance=1),
            _completion_records("session_3_request_1", 960),
            _settle_records("session_3_request_1", 970,
                            service_done_ns=960, settle_done_ns=970,
                            merge_done_ns=965, merge_payload_bytes=131072),
        ])
        rc = main([
            "generate", "--baseline-run", str(baseline5),
            "--evidence-root", str(tmp / "probe5"), "--run-id", "selftest5",
            "--arms", "boundary", "--granularity", "instance-action",
            "--replay-arm", "none"])
        ck(rc == 0, "generate boundary instance-action failed")
        plan5_path = tmp / "probe5" / "selftest5" / "probe_plan.json"
        with open(plan5_path, encoding="utf-8") as source:
            plan5 = json.load(source)
        ck(plan5["granularity"] == "instance-action", plan5["granularity"])
        t5 = plan5["targets"][0]
        b_arm = t5["arms"][0]
        ck(b_arm["granularity"] == "instance-action", b_arm)
        ck(b_arm["arm_class"] == "same_state_forced_branch", b_arm)
        ck(b_arm["entry_pin"] == "copy", b_arm)
        ck(len(b_arm["runs"]) == 3, b_arm["runs"])
        runs5 = {run["key"]: run for run in b_arm["runs"]}
        ck(set(runs5) == {"i0:stay", "i0:copy", "i1:copy"}, set(runs5))
        ck(runs5["i0:copy"]["env"][FORCE_BOUNDARY_TARGET_ENV] == "0:copy",
           runs5["i0:copy"]["env"])
        ck(runs5["i1:copy"]["env"][FORCE_BOUNDARY_TARGET_ENV] == "1:copy",
           runs5["i1:copy"]["env"])
        ck(FORCE_ENTRY_TARGET_ENV not in runs5["i0:stay"]["env"]
           and FORCE_ACTION_ENV not in runs5["i0:stay"]["env"],
           runs5["i0:stay"]["env"])
        ck(runs5["i0:copy"]["run_dir"].endswith(
            "session_3_request_1__D__i0__copy"), runs5["i0:copy"])
        ck(b_arm["predictions"] == {"i0:stay": 40, "i0:copy": 35,
                                    "i1:copy": 30}, b_arm["predictions"])
        # P 参考接续披露（阶段C.5）：P 行 decode_instance/decode_action。
        # 注意 _boundary_record 夹具只落 selected；reference_continuation
        # 从基线 P 行 decision.decode_instance/decode_action 派生——夹具
        # 未落该两键时为 None/None（如实缺席，不猜值）。
        ck(t5["boundaries"]["P"].get("reference_continuation")
           == {"instance": None, "action": None},
           t5["boundaries"]["P"].get("reference_continuation"))

        # —— 合成三个强制 run（settle 事件分采夹具）——
        # i0:stay  = service 960/settle 960（零载荷 settle==service 合法）
        # i0:copy  = service 960/settle 985（merge 载荷 25ns）
        # i1:copy  = service 960/settle 975
        settle_map = {
            "i0:stay": (960, 960, 0),
            "i0:copy": (960, 985, 262144),
            "i1:copy": (960, 975, 131072),
        }
        for key, run in runs5.items():
            service_done, settle_done, payload = settle_map[key]
            instance, action = key[1:].split(":", 1)
            _write_log(Path(run["run_dir"]), [
                _boundary_record(
                    "session_3_request_1", 500,
                    [{"action": "stay", "applicable": True, "cost_ns": 20,
                      "measured": None, "instance_index": 0},
                     {"action": "copy", "applicable": True, "cost_ns": 10,
                      "measured": None, "instance_index": 1}],
                    selected="copy", p_instance=1),
                _boundary_record(
                    "session_3_request_1", 900,
                    _face_candidates_from_plan(t5, "D"),
                    selected=action, stage="D", chosen_instance=int(instance)),
                _completion_records("session_3_request_1", service_done),
                _settle_records(
                    "session_3_request_1", settle_done,
                    service_done_ns=service_done, settle_done_ns=settle_done,
                    merge_done_ns=(settle_done if payload else None),
                    merge_payload_bytes=payload),
            ])
        rc = main(["summarize", "--plan", str(plan5_path), "--dry-run"])
        ck(rc == 0, f"instance-action summarize rc={rc}")
        recs5 = _summarize_capture(plan5)
        a5 = [r for r in recs5
              if r.get("record_kind") == "arm_summary"][0]
        ck(a5["status"] == "ok", a5["status"])
        ck(a5["granularity"] == "instance-action", a5)
        # 双指标：settle 主口径（960/985/975 − 900）；service 并列（60）。
        ck(a5["actuals"] == {"i0:stay": 60, "i0:copy": 85, "i1:copy": 75},
           a5["actuals"])
        ck(a5["actuals_service"] == {"i0:stay": 60, "i0:copy": 60,
                                     "i1:copy": 60}, a5["actuals_service"])
        # 零载荷形态：i0:stay settle==service 显式记录（source 带 zero_
        # payload 标）；载荷形态 source=settle_event。
        ck(a5["actual_sources"]["i0:stay"] == "settle_event_zero_payload",
           a5["actual_sources"])
        ck(a5["actual_sources"]["i1:copy"] == "settle_event",
           a5["actual_sources"])
        # 证书全匹配 + 排序/损失以 settle 为准：argmin 预测 = i1:copy(30)
        # → actual 75；min(actual)=60（i0:stay）→ 损失 15；排序 3 对。
        ck(a5["state_cert"]["status_by_run"]
           == {"i0:stay": "match", "i0:copy": "match", "i1:copy": "match"},
           a5["state_cert"]["status_by_run"])
        ck(a5["selection_loss_ns"] == 15, a5)
        ck(a5["predicted_rank"] == ["i1:copy", "i0:copy", "i0:stay"],
           a5["predicted_rank"])
        ck(a5["actual_rank"] == ["i0:stay", "i1:copy", "i0:copy"],
           a5["actual_rank"])
        ck(a5["errors_ns"] == {"i0:stay": 40 - 60, "i0:copy": 35 - 85,
                               "i1:copy": 30 - 75}, a5["errors_ns"])
        ck(a5["same_state_sample_coverage"]["caliber"] == "同状态全候选集",
           a5["same_state_sample_coverage"])
        # P 参考接续 vs 实际边界 D 分列（阶段C.5）。
        ck(a5["p_reference_continuation"] == {"instance": None,
                                              "action": None},
           a5["p_reference_continuation"])
        ck(a5["actual_boundary_choice"] == {"instance": 1, "action": "copy"},
           a5["actual_boundary_choice"])
        # forced 身份核对含实例位（i0:copy 次优实例被强制，非最低成本）。
        ck(a5["forced_selected_match"]
           == {"i0:stay": True, "i0:copy": True, "i1:copy": True},
           a5["forced_selected_match"])

        # —— 状态证书硬门：单字段漂移（tick 或 cost）⇒ state_mismatch 且
        # 不入同状态样本（升级原 predicted-drift note 软降级）——
        drift_dir = Path(runs5["i1:copy"]["run_dir"])
        _write_log(drift_dir, [
            _boundary_record(
                "session_3_request_1", 500,
                [{"action": "stay", "applicable": True, "cost_ns": 20,
                  "measured": None, "instance_index": 0},
                 {"action": "copy", "applicable": True, "cost_ns": 10,
                  "measured": None, "instance_index": 1}],
                selected="copy"),
            _boundary_record(
                "session_3_request_1", 900,
                [{"action": "stay", "applicable": True,
                  "cost_ns": 40,  # = 基线
                  "measured": None, "instance_index": 0,
                  "is_continuation": True},
                 {"action": "copy", "applicable": True,
                  "cost_ns": 33,  # ≠ 基线 35（预测漂移——原 note 语义）
                  "measured": None, "instance_index": 0},
                 {"action": "copy", "applicable": True, "cost_ns": 30,
                  "measured": None, "instance_index": 1}],
                selected="copy", stage="D", chosen_instance=1),
            _completion_records("session_3_request_1", 960),
            _settle_records("session_3_request_1", 975,
                            service_done_ns=960, settle_done_ns=975,
                            merge_done_ns=975, merge_payload_bytes=131072),
        ])
        recs5 = _summarize_capture(plan5)
        a5 = [r for r in recs5
              if r.get("record_kind") == "arm_summary"][0]
        ck(a5["status"] == "state_mismatch", a5["status"])
        ck(a5["state_cert"]["status_by_run"]["i1:copy"] == "state_mismatch",
           a5["state_cert"]["status_by_run"])
        ck(a5["state_cert"]["mismatch_fields_by_run"]["i1:copy"]
           == ["candidates_surface"],
           a5["state_cert"]["mismatch_fields_by_run"])
        ck(a5["state_mismatch_candidates"] == ["i1:copy"],
           a5["state_mismatch_candidates"])
        # 漂移分支不入样本：损失/排序只在 {i0:stay, i0:copy} 上计算。
        ck(set(a5["actuals"]) == {"i0:stay", "i0:copy"}, a5["actuals"])
        ck(a5["selection_loss_ns"] == 25, a5)  # argmin i0:copy=85 − 60
        ck(a5["same_state_sample_coverage"]["caliber"]
           == "已测子集参照（2/3）", a5["same_state_sample_coverage"])
        ck(any("状态证书不匹配字段" in n for n in a5["notes"]),
           a5["notes"])

        # —— missing：未执行候选（exit_code 缺）标 missing，不以预测值
        # 填补（阶段C.4）——
        (Path(runs5["i0:copy"]["run_dir"]) / "exit_code").unlink()
        recs5 = _summarize_capture(plan5)
        a5 = [r for r in recs5
              if r.get("record_kind") == "arm_summary"][0]
        ck(a5["missing_candidates"] == ["i0:copy"],
           a5["missing_candidates"])
        ck("i0:copy" not in a5["actuals"], a5["actuals"])
        ck(a5["run_states"]["i0:copy"] == "missing", a5["run_states"])
        ck(a5["same_state_sample_coverage"]["caliber"]
           == "已测子集参照（1/3）", a5["same_state_sample_coverage"])

        # —— fixed_policy_arm / oneshot 标记（阶段C.4：whole-run 臂独立
        # 标记，不与同状态数据混算）——
        ck(plan2["replay_arm"]["arm_class"] == "fixed_policy_whole_run",
           plan2["replay_arm"])
        ck(plan4["one_shot_pair"]["arm_class"] == "oneshot_policy_whole_run",
           plan4["one_shot_pair"])

        # —— P 参考接续真值形态：P 行落 decode_instance/decode_action 时
        # reference_continuation 携带真值（接续近似误差可归因）——
        baseline6 = tmp / "baseline6_run"

        def _p_row_with_reference(rid, tick, p_inst, ref_instance,
                                  ref_action):
            record = _boundary_record(
                rid, tick,
                [{"action": "stay", "applicable": True, "cost_ns": 20,
                  "measured": None, "instance_index": 0}],
                selected="stay", p_instance=p_inst)
            record["decision"]["decode_instance"] = ref_instance
            record["decision"]["decode_action"] = ref_action
            return record

        # 直接验证 _scan_baseline_faces 的 reference_continuation 派生
        #（P 行披露参考 D (0, stay)；D 边界实际选择 stay@0——两阶段一致
        # 形态；分列字段让接续近似误差与资源误差分开归因）。
        _write_log(baseline6, [
            _p_row_with_reference("session_3_request_2", 500, 0, 0, "stay"),
            _completion_records("session_3_request_2", 540),
            _boundary_record(
                "session_3_request_2", 900,
                [{"action": "stay", "applicable": True, "cost_ns": 40,
                  "measured": None, "instance_index": 0,
                  "is_continuation": True}],
                selected="stay", stage="D", chosen_instance=0),
            _completion_records("session_3_request_2", 960),
        ])
        per_request6, _order6 = _scan_baseline_faces(baseline6)
        ck(per_request6["session_3_request_2"]["faces"]["P"][
            "reference_continuation"] == {"instance": 0, "action": "stay"},
           per_request6["session_3_request_2"]["faces"]["P"])

        uncertified = _summarize_arm(plan5, t5, b_arm, baseline_certs={})
        ck(uncertified["actuals"] == {}, uncertified)
        ck(uncertified["selection_loss_ns"] is None, uncertified)

        # Production baselines omit causal_prestate unless a force hook hits.
        # The selected forced arm is a valid bounded reference only after its
        # complete shape and every ordinary face field agree; a missing field
        # in both records must stay uncertified.
        reference_root = tmp / "forced_selected_reference"
        reference_base = _boundary_record(
            "reference_request", 120,
            [{"action": "stay", "applicable": True, "cost_ns": 4,
              "measured": None, "instance_index": 0}],
            "stay", p_instance=0)
        reference_forced = json.loads(json.dumps(reference_base))
        reference_forced["decision"]["forced_by_probe"] = {
            "stage": "P", "hook": "entry-target", "action": "stay",
            "instance_index": 0}
        reference_base["decision"].pop("causal_prestate")
        _write_log(reference_root / "baseline", [reference_base])
        _write_log(reference_root / "forced", [reference_forced])
        reference_plan = {
            "baseline_run": str(reference_root / "baseline"),
            "targets": [{
                "request_id": "reference_request",
                "boundaries": {"P": {"selected": "stay",
                                      "selected_instance": 0}},
                "arms": [{
                    "arm_family": "entry", "stage": "P",
                    "runs": [{"action": "stay", "instance_index": 0,
                              "run_dir": str(reference_root / "forced")}],
                }],
            }],
        }
        reference_cert = _baseline_cert_cache(reference_plan)[
            ("reference_request", "P")]
        ck(reference_cert["_evidence_source"].startswith(
            "forced_selected_reference:"), reference_cert)
        ck(not reference_cert["missing_fields"], reference_cert)

        distributed_incomplete = tmp / "distributed_incomplete"
        _write_log(distributed_incomplete, _completion_records("r", 100))
        (distributed_incomplete / "bridge").mkdir()
        (distributed_incomplete / _BASELINE_MANIFEST_RELPATH).write_text(
            json.dumps({"prediction": {"settlement_policy": "distributed"}}),
            encoding="utf-8")
        missing_settle = service_settle_ticks(distributed_incomplete, "r", [])
        ck(missing_settle["service_done_ns"] == 100, missing_settle)
        ck(missing_settle["settle_done_ns"] is None, missing_settle)
        ck(missing_settle["settle_missing_reason"] ==
           "settle_missing_policy_distributed", missing_settle)

        (distributed_incomplete / _BASELINE_MANIFEST_RELPATH).write_text(
            json.dumps({"prediction": {"settlement_policy": "consolidate"}}),
            encoding="utf-8")
        missing_consolidation = service_settle_ticks(distributed_incomplete, "r", [])
        ck(missing_consolidation["settle_done_ns"] is None, missing_consolidation)
        ck(missing_consolidation["settle_missing_reason"] ==
           "settle_missing_policy_consolidate", missing_consolidation)

        (distributed_incomplete / _BASELINE_MANIFEST_RELPATH).write_text(
            json.dumps({"prediction_config": {
                "settlement_policy": "consolidate"}}),
            encoding="utf-8")
        legacy_manifest = service_settle_ticks(
            distributed_incomplete, "r", [])
        ck(legacy_manifest["settle_done_ns"] is None, legacy_manifest)
        ck(legacy_manifest["settle_missing_reason"] ==
           "settlement_policy_unavailable", legacy_manifest)

        incomplete_record = _boundary_record(
            "r", 100,
            [{"action": "stay", "applicable": True, "cost_ns": 1,
              "measured": None, "instance_index": 0}],
            "stay")
        incomplete_record["decision"].pop("causal_prestate")
        incomplete_cert = boundary_state_cert(incomplete_record, "P")
        ck("missing:causal_prestate" in cert_mismatches(
            incomplete_cert, incomplete_cert), incomplete_cert)

        # An empty envelope is not a complete empty world.  Keep this as one
        # compact negative assertion so future fixture edits cannot turn
        # missing production views into a certifiable synthetic baseline.
        empty_record = json.loads(json.dumps(reference_forced))
        empty_prestate = {
            "schema_version": "joint-causal-prestate-v1",
            "stage": "P",
            "request_id": "reference_request",
            **{name: {} for name in _CAUSAL_PRESTATE_FIELDS
               if name not in ("schema_version", "stage", "request_id")},
        }
        empty_record["decision"]["causal_prestate"] = empty_prestate
        empty_cert = boundary_state_cert(empty_record, "P")
        empty_missing = set(empty_cert["missing_fields"])
        ck({
            "decision_snapshot", "work_remaining", "directory", "capacity",
            "arbitration", "history", "versions",
        } <= {
            name.split(".", 2)[1]
            for name in empty_missing
            if name.startswith("causal_prestate.")
            and name.count(".") >= 2
        }, empty_cert)

        print("[probe][selftest] scenario 5 PASS (W8 instance-action arms "
              "+ TARGET env + settle/service split with zero-payload form "
              "+ hard state certificate ok/mismatch + missing marking + "
              "P-reference-continuation separation + fixed_policy_arm "
              "labels)")

        print(f"[probe][selftest] PASS ({passed['n']} assertions)")
    return 0


def by_key_label_run_dir(plan: dict, rid: str, family: str,
                         action: str) -> str:
    """selftest 辅助：取 (请求, 臂族, 动作) 的 run_dir。"""
    for target in plan["targets"]:
        if target["request_id"] != rid:
            continue
        for arm in target["arms"]:
            if arm["arm_family"] != family:
                continue
            for run in arm["runs"]:
                if run["action"] == action:
                    return run["run_dir"]
    raise AssertionError(f"no run for {rid}/{family}/{action}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="同状态对照 probe harness（式(24) 选择损失 + 宿主开销"
                    "待验证预算；generate 只生成命令不执行仿真）")
    sub = parser.add_subparsers(dest="command", required=True)

    gen = sub.add_parser(
        "generate", help="从基线 run 生成 请求×适用动作 的串行命令计划")
    gen.add_argument("--baseline-run", required=True,
                     help="基线 run 目录（含 results/online_decision_log"
                          ".jsonl；取预测代价与适用动作）")
    gen.add_argument("--strict-version", action="store_true",
                     help="baseline 世代（model_semantics_version）与当前"
                          "代码不一致（或无法核验）时 fail-closed 拒绝生成"
                          "（缺省只 stderr 告警照常生成——旧代 baseline 仍"
                          "可用于粗对照，由使用者判断）")
    gen.add_argument("--arms",
                     choices=("entry", "boundary", "both", "one-shot"),
                     default="both",
                     help="臂族：entry=P 入口枚举；boundary=固定 P 动作"
                          "（entry pin）+ D 边界枚举；both=两者（缺省）；"
                          "one-shot=一次映射对照成对臂（twophase_reference"
                          "×oneshot_p_continuation，同 trace 同参数仅差 "
                          "JOINT_BOUNDARY_POLICY，无需强制钩子）")
    gen.add_argument("--stage", choices=("P", "D"), default=None,
                     help="[旧接口] P→--arms entry；D→--arms boundary"
                          "（boundary 臂现已带 entry pin，同状态语义升级）")
    gen.add_argument("--granularity",
                     choices=("action", "instance-action"),
                     default="action",
                     help="W8 臂枚举粒度：action=逐动作一臂（旧口径，多实"
                          "例时调度器按动作重选实例——弃用方向，向后兼容）；"
                          "instance-action=逐适用 (instance,action) 候选各"
                          "一臂（JOINT_FORCE_*_TARGET 精确强制，全部候选的"
                          "反事实真值，方案 §11.3）")
    gen.add_argument("--request", action="append", default=[],
                     help="显式指定请求 id（可重复；缺省自动选面合格且"
                          "动作集不重复的前 N 个）")
    gen.add_argument("--max-requests", type=int, default=DEFAULT_MAX_REQUESTS)
    gen.add_argument("--max-commands", type=int, default=DEFAULT_MAX_COMMANDS,
                     help="命令总数上限（含回放臂；超出即 fail-closed，"
                          "防误生成大矩阵）")
    gen.add_argument("--replay-arm", choices=("load-first", "none"),
                     default="load-first",
                     help="固定动作回放臂（宿主开销对照；none=不生成）")
    gen.add_argument("--window-ns", default=str(DEFAULT_WINDOW_NS),
                     help="物化窗口 ns（缺省 2e9 = 2s 冒烟最小窗）")
    gen.add_argument("--source-csv", default=str(DEFAULT_SOURCE_CSV))
    gen.add_argument("--evidence-root", default=str(DEFAULT_EVIDENCE_ROOT))
    gen.add_argument("--run-id", default=None,
                     help="运行标识（缺省 probe_<UTC 时间戳>；已存在即拒绝）")
    gen.add_argument("--env", action="append", default=[],
                     help="各臂透传的额外 env（NAME=VALUE，可重复；如 "
                          "JOINT_ABLATION_COMBO=TJE 对齐基线臂开关；回放"
                          "臂自动剔除与 load-first 互斥的键）")
    gen.add_argument("--dry-run", action="store_true",
                     help="只打印计划与脚本内容，不写任何文件")
    gen.set_defaults(func=cmd_generate)

    summ = sub.add_parser("summarize", help="汇总各候选 run 并计算式(24)损失")
    summ.add_argument("--plan", required=True, help="probe_plan.json 路径")
    summ.add_argument("--dry-run", action="store_true",
                      help="只打印汇总，不写 probe_summary.*")
    summ.set_defaults(func=cmd_summarize)

    self = sub.add_parser(
        "selftest", help="/tmp 小 fixture 端到端自验（不触仿真/仓库）")
    self.set_defaults(func=cmd_selftest)

    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except ProbeError as exc:
        print(f"[probe] FAIL: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
