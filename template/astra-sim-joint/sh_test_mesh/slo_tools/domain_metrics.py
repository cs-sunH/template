#!/usr/bin/env python3
"""domain_metrics —— 局部统一内存域三口径离线度量（C16/WP6b，2026-09-22；
2026-10-14 远读动作退役重收缩：挑战者动作重指派为 copy；2026-10-15
recompute 动作退役：替代族收缩为 stay-only，二动作化）。

依据：设计文档《joint机制改造方案_局部统一内存域》§5.1-§5.3（域只描述
挑战者条件、三类观测必须分开、边界检验）、§1.2（d≤ρ 简化解析锚点仅历史
扫描项平价）；执行计划 §5 C16 卡；C5 冻结 schema（PROVENANCE §19）；
C10 δ_adm 终冻报告（维持 0）+ A3' 下游指令（离线敏感性重定价
δ∈{σ̂,2σ̂,4σ̂}，不留 δ=0 空转对拍臂）。

三类观测必须分开（§5.2，F13）——本工具的三张口径（二动作化后挑战者 =
copy，替代族 = stay）：

1. 预测供给集合 D_feed(q,p,s)：决策日志中 copy 候选**配额前结构可行**
   的实例集合（M2/N6：applicable ∨ quota_ 前缀拒且配额前成本在——
   供给条件已判过；实际配额准入域 = challenger_applicable 列，两域
   分列）；逐成员标注本地参考（min-cost stay）与性能容忍度
   （C_copy−C_stay_ref）。
2. 预测经济集合 D_econ(q,s)：C_alt* = min C(e,a) 遍历**全部实例的适用
   非挑战者动作（stay）**；D_econ = {e: C_copy(e) ≤ C_alt*+δ}。
   主臂 δ=δ_adm=0（C10 终冻）；敏感性臂 δ∈{σ̂,2σ̂,4σ̂} 仅离线诊断。
   同位置动作比较（copy vs 同实例最优非 copy）另列、口径单独标注。
3. 实际选择与受控 measured 集合：在线日志只含最终 (e,a)（selected_
   action/joint_instance_index）；两动作选中计数（stay/copy）单列
   （§19.2，两口径不得混报）。
   独立同状态执行的 measured 集合**不在本产物**（C4b 受控测量三表）——
   如实标 NA，不与预测混合；预测/实测差仅作 §5.2 诊断分列。

派生指标（§5.3）：|D|、距离分位数（hop）、方向分布（分方向半径方差）、
瓶颈资源（breakdown 五段 argmax）、约束不满足原因、预测/实测差异；
双域同图数据（D_econ 等值线 vs 配额准入域实线——两域之差即配额作用量；
N6 口径消歧：D_econ/D_feed 为**配额前**口径（quota_admissible
列 = 移除配额后的结构可行域），实际配额准入域 =
challenger_applicable/applicable 列；配额前成本等值线输入 = instances CSV
challenger_cost_pre_quota_ns 列；run 级配额作用量 = quota_counterfactual_
flips）；内等值线（C_copy=C_stay_ref）/
外等值线（C_copy=C_alt*，替代族 = stay、含池恢复形态）逐 (请求,实例)
导出。

边界对拍纪律（全部落输出注释）：
* 挑战者/local 分界 vs d≤ρ 简化解析锚点——**仅历史扫描项平价、不保证
  完整请求平价**；ρ = B_D2D/B_HBM 逐配置硬件派生（trace_config
  hardware_config → hardware json）。
* **无 n* = d′/(d−ρ) 式**（v2.2 已删除）；对拍是诊断不是验收——不要求
  逐点重合，偏差本身是模型诊断结果。

叙事纪律内置：饱和下域由配额关闭而非 argmin 涌现——"涌现域"表述限定
配额界内；域是解释与可选准入分析对象，**不裁剪任何动作/实例候选**
（全候选 argmin 照常，本工具只读不反向参与调度）。

数据源（全部 run_dir 本地、先实测登记不猜测）：
* results/online_decision_log.jsonl —— kind=joint_admission（C5 schema：
  candidates[]{action,instance_index,applicable,cost_ns,hops,
  inapplicable_reason,breakdown 字段} + selected_action/applicable_
  actions/load_view/flow_snapshot/port_snapshot）
  与 kind=completion（C3b 披露：merge_direction/home_flipped_to/
  merge_transferred_bytes/kv_instance_after_completion）。
  （2026-10-15 recompute 退役：决策行 recompute_selection 键随 D5 删除；
  旧日志含该键的行照常读取、该面按缺失口径 NA，不解析。）
* per-request manifest（history_tokens_before/final_context_tokens）——
  逐请求 KV 字节组件（H/I）。
* trace_config.csv（inference_group ranks + layers/hidden_size/
  bytes_per_elem + hardware_config）与 hardware json（mesh 行列、
  d2d/local-hbm 带宽 → ρ、实例锚点坐标 → 方向分布）——任一环缺失 →
  对应派生量 NA 并注明，不代拟。
* bridge/joint_kv_ledgers.json 的 kv_delta_journal 键 —— home 迁移轨迹
  结算时刻源（C14 逐请求合并披露、F3 序列化落盘；决策时刻 decision-log
  与结算时刻 journal 不混，四层可信度分级读取见 _kv_delta_trajectory；
  results/kv_delta_journal.jsonl 的 wscllm 式 checksum 权威层本仓不产
  出——C14 §20.3 README 注记）。

非 joint 仓/无 joint_admission 行的 run_dir：本工具**完全静默跳过**
（不写产物、不打日志行）——四仓共链的字节对拍契约（test_driver_
parity 的 sh_1.0 fixture）要求共享后处理链对非 joint 输入零扰动；
joint run 恒有决策行，步骤行/失败语义与链上其他步骤一致。

纯离线、纯标准库、只读输入；B 类参数（slo_params_manifest.json）
fail-closed：domain_delta_adm_ns / domain_delta_sensitivity_multipliers
遇 null/缺失/非法即退出码 2 并指明参数名。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))

from slo_common import (  # noqa: E402
    DECISION_LOG_RELPATH, NA, SloToolError, default_manifest_path,
    emit_json, fail, fmt_ratio, iter_jsonl, load_request_manifest,
    load_slo_manifest, nearest_rank_percentile, nearest_rank_percentile_many,
    open_output, require_param, require_param_int, run_main, write_csv,
)

REQUESTS_CSV_NAME = "slo_domain_requests.csv"
INSTANCES_CSV_NAME = "slo_domain_instances.csv"
SUMMARY_JSON_NAME = "slo_domain_summary.json"

# C5 schema（PROVENANCE §19.1）selected_action 枚举（F10 二动作范围，
# 2026-10-15 recompute 退役缩二）。
ALL_ACTIONS = ("stay", "copy")
# 域口径的动作分族（2026-10-14 远读退役重收缩挑战者 = copy；2026-10-15
# recompute 退役：替代族收缩 stay-only——C_alt* 在替代族上取 min，
# D_feed/D_econ 按挑战者候选重建）。
CHALLENGER_ACTION = "copy"
ALT_ACTIONS = ("stay",)
# 候选落盘序 = SH ACTION_ORDER（stay, copy；§19.1
# applicable_actions 序）。K7（P2-10，2026-09-23 外部审计）：
# δ=0 谓词重放的平局裁决**镜像在线 argmin 全键序**（cost, instance,
# ACTION_ORDER 偏好）——原 (cost, ACTION_ORDER, instance) 与在线
# (cost_ns, (instance_index, action_priority)) 平局分歧（同 cost 时
# 在线取小实例、离线取优动作），"replay_mismatch 应为 0"的声称被
# 三向平局证伪。O8①（2026-09-23）：族间比较（挑战者 min vs C_alt*，
# _selection_under_delta）补同款全键序——"平局裁决镜像在线 argmin
# 全键序"自此对族内（_min_cost）与族间均属实，跨族同价平局不再
# 假翻挑战者（replay_mismatch/quota_counterfactual_flips 假阳性消除）。
# O8① v9 续注（2026-10-20，轮转感知基准）：在线 joint 主臂跨实例平局
# 自 v9 起按实例编号环序轮转分摊（语义单一源 = joint/
# joint_scheduler.joint_tie_rotation_pick）——固定全键序不再是选择
# 预测，"与在线实际选择比对"的各指标（replay_mismatch_at_delta0 /
# δ 分档 flips_vs_actual / quota_counterfactual_flips /
# displacement_hop）比较基准自 actual 改为 actual_ref：带轮转三键且
# rotated 的 P 行 actual_ref = 由本行候选表现算的轮转公式期望选择
# （_joint_tie_rotation_pick_mirror——单一源的本地镜像 + 对账钉测，
# slo_tools 不 import joint 模块的既有纪律），纯轮转位移不计翻转/
# 不污染位移分位，actual != expected 计 replay_mismatch（真发散）；
# 键缺席（旧 run）⇒ actual_ref = actual，输出与旧口径逐位一致。
ACTION_ORDER = ("stay", "copy")

# v9（2026-10-20）在线 P 准入决策行的轮转披露三键（joint_admission 行
# decision 级；joint_admission_wait 折叠行与旧 run 无这三键——解析缺省
# 即旧行为。joint_admission_failed 首行在线也披露，但失败行不进本工具
# 的 P 样本（DomainScan 只消费 joint_admission 行），此处不解析）。
_TIE_DISCLOSURE_KEYS = (
    "joint_tie_cursor", "joint_tie_instances", "joint_tie_rotated")
_ACTION_RANK = {action: rank for rank, action in enumerate(ACTION_ORDER)}

# breakdown 五段时间段（瓶颈资源 argmax 的候选集）。字段序 =
# _JOINT_BREAKDOWN_LOG_FIELDS 冻结序的子集（远读段随 D07/D7 退役删除）。
BREAKDOWN_SEGMENTS = (
    "target_wait_ns", "history_prep_ns", "eviction_wait_ns",
    "compute_ns", "merge_ns",
)

KV_DELTA_JOURNAL_RELPATH = Path("results") / "kv_delta_journal.jsonl"

# kv_delta_journal 消费的四层可信度（F3；语义镜像 hbm_watermark 的四层
# 可信度纪律——tier 由 run_dir 内容自动判定，status 如实标注层级）。
# 水印 certified 层依赖的守恒证书（results/kv_delta_journal.jsonl +
# kv_delta_journal_checksum.json，wscllm 式归档权威层）**本仓不产出**
# （C14 §20.3 README 注记）——顶层按可得信号降级实现并披露，不冒认
# certified。四层（证据强度降序）：
#   settlement_full_join     sidecar 键在场非空、行链自洽（seq 严格增）、
#                            全部行联上 completion tick（驻留时序完备）；
#   settlement_partial_join  行在案且链自洽，但存在联不上 tick 的行
#                            （结算事实完整——行内字段自证；仅驻留时序
#                            近似降级，rows_unjoined/dwell_unjoined 计数）；
#   settlement_empty         sidecar 在场但键缺席/空列表（零结算 run——
#                            journal 行存在 ⇔ 结算完成，或 F3 前旧 sidecar
#                            schema）——home 迁移轨迹仅剩决策时刻披露；
#   decision_log_only        sidecar 文件缺席/不可解析——同上退决策时刻
#                            披露，且无法区分"零结算"与"F3 前未序列化"。
# 行链断裂（非对象行 / seq 非严格增）fail-closed 退出码 2——账本破损
# 不得静默降级（水印同款纪律）。
KV_DELTA_TIER_FULL = "settlement_full_join"
KV_DELTA_TIER_PARTIAL = "settlement_partial_join"
KV_DELTA_TIER_EMPTY = "settlement_empty"
KV_DELTA_TIER_DECISION_ONLY = "decision_log_only"
KV_DELTA_TIER_NOTE = (
    "四层可信度语义镜像 hbm_watermark（tier 由 run_dir 内容自动判定）；"
    "水印 certified 层的守恒证书（results/kv_delta_journal.jsonl + "
    "checksum，wscllm 式归档权威层）本仓不产出（C14 §20.3 README 注记）"
    "——顶层按可得信号（sidecar 行 + 行链自洽 + completion 联 join）"
    "降级实现并如实标注，不冒认 certified")

REQUESTS_COLUMNS = (
    "request_id", "tick_ns", "n_instances", "reference_instance",
    "reference_source", "selected_instance", "selected_action",
    "selected_hop",
    "d_feed_size", "d_feed_members",
    "c_stay_ref_ns", "c_alt_star_ns", "c_alt_star_action",
    "d_econ_size_delta0", "d_econ_members_delta0",
    "same_position_challenger_wins",
    "observed_inner_boundary_hop", "observed_feed_boundary_hop",
    "rho_anchor", "boundary_deviation_hop",
    "n_active_links", "pred_ns", "measured_ns", "measured_service_ns", "pred_measured_err_ns",
    "h_bytes_per_rank", "i_bytes_per_rank", "f_min_bytes_at_selected",
)

INSTANCES_COLUMNS = (
    "request_id", "instance_index", "anchor_row", "anchor_col",
    "direction", "hops",
    "challenger_applicable", "in_d_feed", "challenger_cost_ns",
    "challenger_cost_pre_quota_ns",
    "feed_margin_ns", "feed_margin_ratio",
    "stay_cost_ns", "copy_cost_ns",
    "same_pos_best_alt_ns", "same_pos_challenger_pref",
    "c_challenger_minus_stay_ref_ns", "c_challenger_minus_alt_star_ns",
    "in_d_econ_delta0", "in_d_econ_m1", "in_d_econ_m2", "in_d_econ_m4",
    "port_u_port_total", "quota_admissible",
    "challenger_bottleneck_component", "challenger_inapplicable_reason",
)


# ---------------------------------------------------------------------------
# 拓扑 / 硬件锚点（ρ、实例锚点坐标；任一环缺失 → NA 降级不 fail）
# ---------------------------------------------------------------------------

def _read_trace_config_rows(path: Path) -> dict:
    """trace_config.csv → {config 值表, groups: [(name, ranks)]}。"""
    config: dict[str, str] = {}
    groups: list[tuple[str, list[int]]] = []
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            kind = (row.get("kind") or "").strip().lower()
            if kind == "config":
                key = (row.get("key") or "").strip()
                if key:
                    config[key] = (row.get("value") or "").strip()
            elif kind == "inference_group":
                name = (row.get("group_name") or "").strip()
                ranks = _parse_rank_spec(row.get("ranks") or "")
                groups.append((name, ranks))
    return {"config": config, "groups": groups}


def _parse_rank_spec(spec: str) -> list[int]:
    ranks: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            ranks.extend(range(int(lo), int(hi) + 1))
        else:
            ranks.append(int(chunk))
    return sorted(set(ranks))


def _load_hardware_json(path: Path) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        fail(f"hardware 配置必须是对象：{path}")
    return data


def resolve_topology(run_dir: Path, trace_config_arg: Optional[str],
                     hardware_config_arg: Optional[str],
                     warnings: list[str]) -> dict:
    """ρ 与实例锚点坐标解析（NA 降级链，警告收集到 warnings 延迟输出）。

    trace 优先级：显式 ``--trace-config`` > run_dir/trace_config.csv.snapshot
    > run_dir/trace_config.csv。hardware 优先级：显式 ``--hardware-config``
    > run_dir/hardware_config.json.snapshot > run_dir/hardware.json.snapshot
    > run_dir 中的旧硬件文件。历史 run 没有随 run 归档的输入时保留 NA
    并警告；不得用当前 checkout 的配置替历史 run 伪造 ρ。
    """
    run_dir = Path(run_dir)
    trace_path: Optional[Path] = None
    if trace_config_arg:
        trace_path = Path(trace_config_arg)
        if not trace_path.is_file():
            fail(f"--trace-config 指定的文件不存在：{trace_path}")
    else:
        for candidate in (run_dir / "trace_config.csv.snapshot",
                          run_dir / "trace_config.csv"):
            if candidate.is_file():
                trace_path = candidate
                break
    result: dict[str, Any] = {
        "trace_config_path": str(trace_path) if trace_path else NA,
        "hardware_path": NA, "rho": None, "rho_source": NA,
        "anchors": {}, "mesh_rows": None, "mesh_cols": None,
        "coef_bytes_per_token_per_npu": None,
    }
    if trace_path is None:
        warnings.append(
            "[domain-metrics] 警告：run_dir 缺 trace_config 快照（或旧 run "
            "本地配置）——ρ/实例锚点/KV 字节系数保持 NA；请显式传入 "
            "--trace-config，未回退当前 checkout")
        trace = {"config": {}, "groups": []}
        config = trace["config"]
    else:
        trace = _read_trace_config_rows(trace_path)
        config = trace["config"]

        # KV 每_token 每 NPU 字节系数（2·layers·hidden·bytes_per_elem；与
        # hbm_watermark 同式）——逐请求 KV 字节组件（H/I）用。
        try:
            coef = (2 * int(config["layers"]) * int(config["hidden_size"])
                    * int(config["bytes_per_elem"]))
            result["coef_bytes_per_token_per_npu"] = coef
        except (KeyError, ValueError):
            warnings.append(
                "[domain-metrics] 警告：trace_config 缺 layers/hidden_size/"
                "bytes_per_elem——H/I 字节组件 = NA")

    hardware_path: Optional[Path] = None
    if hardware_config_arg:
        hardware_path = Path(hardware_config_arg)
        if not hardware_path.is_file():
            fail(f"--hardware-config 指定的文件不存在：{hardware_path}")
    else:
        for candidate in (
                run_dir / "hardware_config.json.snapshot",
                run_dir / "hardware.json.snapshot",
                # 早期自包含 run 目录使用过这两个普通文件名；它们仍是
                # run 本地证据，可读但不会再从当前 checkout 猜测替代。
                run_dir / "hardware_config.json",
                run_dir / "hardware.json"):
            if candidate.is_file():
                hardware_path = candidate
                break
    if hardware_path is None:
        warnings.append(
            "[domain-metrics] 警告：run_dir 缺 hardware JSON 快照（或旧 run "
            "本地文件）——ρ/网格锚点保持 NA；请显式传入 "
            "--hardware-config，未回退当前 checkout")
        return result
    hardware = _load_hardware_json(hardware_path)
    result["hardware_path"] = str(hardware_path)

    d2d = (hardware.get("d2d") or {}).get("bandwidth-gbps")
    hbm = (hardware.get("local-hbm") or {}).get("bandwidth-gbps")
    if isinstance(d2d, (int, float)) and isinstance(hbm, (int, float)) \
            and hbm > 0:
        result["rho"] = float(d2d) / float(hbm)
        result["rho_source"] = (
            f"{hardware_path}#d2d.bandwidth-gbps={d2d} / "
            f"local-hbm.bandwidth-gbps={hbm}")
    else:
        warnings.append(
            "[domain-metrics] 警告：hardware json 缺 d2d/local-hbm 带宽"
            "键——ρ 锚点 = NA")

    mesh = hardware.get("mesh") or {}
    rows, cols = mesh.get("rows"), mesh.get("columns")
    if isinstance(rows, int) and isinstance(cols, int) and rows > 0 \
            and cols > 0:
        result["mesh_rows"], result["mesh_cols"] = rows, cols
        anchors: dict[int, tuple[int, int]] = {}
        for index, (_name, ranks) in enumerate(trace["groups"]):
            if not ranks:
                continue
            anchor = min(ranks)
            anchors[index] = divmod(anchor, cols)
        if anchors:
            result["anchors"] = anchors
        else:
            warnings.append(
                "[domain-metrics] 警告：trace_config 无 inference_group 行"
                "——实例锚点坐标 = NA（方向分布降级）")
    else:
        warnings.append(
            "[domain-metrics] 警告：hardware json 缺 mesh.rows/columns——"
            "实例锚点坐标 = NA（方向分布降级）")
    return result


def direction_bucket(source: tuple[int, int],
                     target: tuple[int, int]) -> str:
    """实例锚点坐标差的罗盘方向（3×3 实例网格的 8 向 + local）。

    距离只是描述量（§5.3）——方向分布用于观察各向异性，不要求域为
    圆形或连通区域。
    """
    drow = target[0] - source[0]
    dcol = target[1] - source[1]
    if drow == 0 and dcol == 0:
        return "local"
    ns = "N" if drow < 0 else ("S" if drow > 0 else "")
    ew = "W" if dcol < 0 else ("E" if dcol > 0 else "")
    return ns + ew


# ---------------------------------------------------------------------------
# 单遍扫描（消费 joint_admission / completion 行；拓扑无关的原始累积）
# ---------------------------------------------------------------------------

def _int_or_none(value: Any) -> Optional[int]:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def _tick_or_none(value: Any) -> Optional[int]:
    value = _int_or_none(value)
    return value if value is not None and value >= 0 else None


class DomainScan:
    """decision log 单遍消费者（prepare/consume/finish 三段中的扫描段）。

    只累积与派生相关 的原始字段（逐候选
    cost/hops/reason/六段 breakdown、决策级 selected/load_view/port/
    flow、completion 披露）——方向/分位/敏感重定价等拓扑与参数相关的
    计算全部推迟到 finish（参数 fail-closed 与警告输出也推迟，保证非
    joint run 的完全静默）。窗口尺度 run（2s/10s）内存 O(决策数×候选数)。
    """

    def __init__(self) -> None:
        self.joint_rows = 0
        self.decisions: list[dict] = []
        # cfg1（2026-09-30，§10.3 完成预测成熟报告）：kind="decode" 行的
        # D 边界决策投影（boundary_decision.candidates + 执行候选身份）
        # ——P/D 分列误差报告的第二样本源（P 样本 = joint_admission 行）。
        self.d_decisions: list[dict] = []
        self.completions: dict[str, dict] = {}
        # M3：kind="merge_done" 披露行索引（request_id → merge_done
        # tick）——σ̂ 同终点配对用（C3b 分报行）。
        self.merge_done: dict[str, int] = {}
        # W5（2026-09-29，§7.4/§9.3 settle_done 统一语义）：kind=
        # "settle_done" 披露行索引（request_id → settle_done tick）
        # ——settle 终点的**主**配对源；旧日志的 merge_done 行仅在
        # consolidate 已核实等价（settle_done_ns == merge_done_ns，
        # 同一 watch 交付事件）时映射（兼容层见 _settle_tick_of）。
        self.settle_done: dict[str, int] = {}
        self.invalid_settle: set[str] = set()
        self.last_tick: Optional[int] = None

    def consume(self, record: dict) -> None:
        kind = record.get("kind")
        tick = _tick_or_none(record.get("tick"))
        if tick is not None:
            self.last_tick = tick if self.last_tick is None \
                else max(self.last_tick, tick)
        request_id = record.get("request_id")
        decision = record.get("decision")
        if not isinstance(decision, dict):
            return
        if kind == "joint_admission":
            self.joint_rows += 1
            self.decisions.append(self._scan_decision(
                record, request_id, decision))
        elif kind == "decode":
            # cfg1：D 边界决策行投影（无 boundary_decision 的旧日志/一次
            # 映射臂行照读、投影为 None——样本计数如实披露，不造样本）。
            scanned = self._scan_d_decision(record, request_id, decision)
            if scanned is not None:
                self.d_decisions.append(scanned)
        elif kind == "completion":
            self.completions[str(request_id)] = {
                "tick": tick,
                "merge_direction": decision.get("merge_direction"),
                "home_flipped_to": decision.get("home_flipped_to"),
                "merge_transferred_bytes":
                    _int_or_none(decision.get("merge_transferred_bytes")),
                "kv_instance_after_completion":
                    decision.get("kv_instance_after_completion"),
                "joint_action": decision.get("joint_action"),
            }
        elif kind == "settle_done":
            settle_tick = (_tick_or_none(decision["settle_done_ns"])
                           if "settle_done_ns" in decision else tick)
            service_tick = (_tick_or_none(decision["service_done_ns"])
                            if "service_done_ns" in decision else None)
            key = str(request_id)
            if (settle_tick is None
                    or ("service_done_ns" in decision and service_tick is None)
                    or (service_tick is not None and settle_tick < service_tick)
                    or (key in self.settle_done
                        and self.settle_done[key] != settle_tick)):
                self.invalid_settle.add(key)
            else:
                self.settle_done[key] = settle_tick
        elif kind == "merge_done":
            merge_tick = (_tick_or_none(decision["merge_done_ns"])
                          if "merge_done_ns" in decision else tick)
            if merge_tick is not None:
                self.merge_done[str(request_id)] = merge_tick

    @staticmethod
    def _candidate_slot(cand: dict) -> dict:
        """逐候选披露槽（P/D 两扫描器共用；cfg1 扩完成预测里程碑与
        归因面——approximations/notes 原样保留供七分归因映射）。"""
        breakdown = cand.get("breakdown")
        segments: dict[str, Optional[int]] = {
            name: None for name in BREAKDOWN_SEGMENTS}
        predicted: dict[str, Optional[int]] = {
            "predicted_service_done": None,
            "predicted_merge_done": None,
            "predicted_next_token_done": None,
        }
        approximations: list = []
        notes: list = []
        if isinstance(breakdown, dict):
            for name in BREAKDOWN_SEGMENTS:
                segments[name] = _int_or_none(breakdown.get(name))
            for name in predicted:
                predicted[name] = _int_or_none(breakdown.get(name))
            if isinstance(breakdown.get("approximations"), list):
                approximations = [
                    str(item) for item in breakdown["approximations"]]
            if isinstance(breakdown.get("notes"), list):
                notes = [str(item) for item in breakdown["notes"]]
        slot = {
            "applicable": bool(cand.get("applicable")),
            "cost_ns": _int_or_none(cand.get("cost_ns")),
            "hops": _int_or_none(cand.get("hops")),
            "reason": cand.get("inapplicable_reason"),
            "segments": segments,
        }
        slot.update(predicted)
        slot["approximations"] = approximations
        slot["notes"] = notes
        return slot

    @staticmethod
    def _scan_decision(record: dict, request_id: Any,
                       decision: dict) -> dict:
        candidates_raw = decision.get("candidates")
        if not isinstance(candidates_raw, list):
            fail(f"joint_admission 行缺 candidates 数组：{request_id}")
        candidates: dict[int, dict[str, dict]] = {}
        for cand in candidates_raw:
            if not isinstance(cand, dict):
                continue
            instance = _int_or_none(cand.get("instance_index"))
            action = cand.get("action")
            if instance is None or action not in ALL_ACTIONS:
                fail(f"joint_admission 候选 instance_index/action 非法："
                     f"{request_id}（{action!r}@{instance!r}）")
            slot = DomainScan._candidate_slot(cand)
            candidates.setdefault(instance, {})[str(action)] = slot
        load_view: dict[int, dict] = {}
        raw_load = decision.get("load_view")
        if isinstance(raw_load, list):
            for entry in raw_load:
                if not isinstance(entry, dict):
                    continue
                instance = _int_or_none(entry.get("instance_index"))
                if instance is None:
                    continue
                remaining = entry.get("hbm_remaining_bytes_by_tp_rank")
                load_view[instance] = {
                    "hbm_remaining_min":
                        min((r for r in remaining
                             if isinstance(r, int) and r >= 0), default=None)
                        if isinstance(remaining, list) else None,
                    # cfg1 负载阶段分组输入（决策行 load_view 的队/在跑
                    # 量字段；缺席 = None → 阶段 NA，不造）。
                    "queued_task_load_ns":
                        _int_or_none(entry.get("queued_task_load_ns")),
                    "active_decode_task_load_ns":
                        _int_or_none(
                            entry.get("active_decode_task_load_ns")),
                }
        port_rows: dict[int, dict] = {}
        raw_port = decision.get("port_snapshot")
        if isinstance(raw_port, dict) \
                and isinstance(raw_port.get("instances"), list):
            for entry in raw_port["instances"]:
                if not isinstance(entry, dict):
                    continue
                instance = _int_or_none(entry.get("instance_index"))
                if instance is None:
                    continue
                port_rows[instance] = {
                    "u_port_total": entry.get("u_port_total"),
                    "parity_gate_headroom": entry.get("parity_gate_headroom"),
                    "na": all(
                        isinstance(entry.get(key), str)
                        for key in ("u_port_total", "parity_gate_headroom")),
                }
        flow = decision.get("flow_snapshot")
        active_links = 0
        if isinstance(flow, dict):
            for value in flow.values():
                if isinstance(value, int) and value > 0:
                    active_links += 1
        selected_action = decision.get("selected_action")
        if selected_action is not None and selected_action not in ALL_ACTIONS:
            fail(f"selected_action 枚举非法：{request_id}"
                 f"（{selected_action!r}，F10 二动作范围）")
        joint_action = decision.get("joint_action")
        if selected_action is not None and joint_action is not None \
                and selected_action != joint_action:
            fail(f"selected_action 与 joint_action 不同值：{request_id}"
                 f"（{selected_action!r} vs {joint_action!r}，§19.1 冗余"
                 f"断言位）")
        # 2026-10-15 recompute 退役：recompute_selection 决策行键随 D5
        # 删除；旧日志行含该键不解析（该面按缺失口径 NA）。
        # v9（2026-10-20）轮转三键（decision 级 _TIE_DISCLOSURE_KEYS；
        # wait 折叠行与旧 run 无这三键）：缺省 None/None/False——
        # joint_tie_instances 解析值仅披露对照，重放裁决以候选表现算值
        # 为准（derive_entries 的轮转感知重放不消费解析值）。
        tie_keys_present = any(
            key in decision for key in _TIE_DISCLOSURE_KEYS)
        tie_instances_raw = decision.get("joint_tie_instances")
        tie_instances: Optional[tuple] = None
        if isinstance(tie_instances_raw, list):
            parsed_instances = [
                _int_or_none(item) for item in tie_instances_raw]
            if all(item is not None for item in parsed_instances):
                tie_instances = tuple(parsed_instances)
        return {
            "request_id": str(request_id),
            "tick": _tick_or_none(record.get("tick")),
            "candidates": candidates,
            "selected_action": selected_action,
            "selected_instance": _int_or_none(
                decision.get("joint_instance_index")),
            "applicable_actions": decision.get("applicable_actions"),
            "load_view": load_view,
            "port_rows": port_rows,
            "active_links": active_links,
            "origin_home_instance": _int_or_none(
                decision.get("origin_home_instance")),
            "cost_ns": _int_or_none(decision.get("joint_cost_ns")),
            "tie_keys_present": tie_keys_present,
            "tie_cursor": _int_or_none(decision.get("joint_tie_cursor")),
            "tie_instances": tie_instances,
            "tie_rotated": decision.get("joint_tie_rotated") is True,
        }

    @staticmethod
    def _scan_d_decision(record: dict, request_id: Any,
                         decision: dict) -> Optional[dict]:
        """cfg1：kind="decode" 行的 D 边界决策投影（completion_error_
        report 的 D 样本源）。无 boundary_decision（旧日志/一次映射臂
        boundary 键为 None）→ None（不造样本）。

        执行候选身份解析（回退形态与 sh30 登记侧同判据）：非回退 =
        (chosen_instance_index, selected_action_face)；回退行 chosen_* 已是
        执行继续形态（sh30 侧 boundary["chosen_action"] = 执行面动作）。"""
        boundary = decision.get("boundary_decision")
        if not isinstance(boundary, dict):
            return None
        candidates_raw = boundary.get("candidates")
        candidates: dict[int, dict[str, dict]] = {}
        executed_slot: Optional[dict] = None
        if isinstance(candidates_raw, list):
            chosen_instance = _int_or_none(
                boundary.get("chosen_instance_index"))
            face = boundary.get("selected_action_face") or \
                boundary.get("chosen_action")
            for cand in candidates_raw:
                if not isinstance(cand, dict):
                    continue
                instance = _int_or_none(cand.get("instance_index"))
                action = cand.get("action")
                if instance is None or not isinstance(action, str):
                    continue
                slot = DomainScan._candidate_slot(cand)
                candidates.setdefault(instance, {})[action] = slot
                if (executed_slot is None and chosen_instance is not None
                        and instance == chosen_instance
                        and isinstance(face, str) and action == face):
                    executed_slot = slot
        inherited = boundary.get("inherited_obligations")
        return {
            "request_id": str(request_id),
            "tick": _tick_or_none(record.get("tick")),
            "candidates": candidates,
            "selected_instance": _int_or_none(
                boundary.get("chosen_instance_index")),
            "selected_action": boundary.get("chosen_action"),
            "selected_action_face": boundary.get("selected_action_face"),
            "is_continuation": bool(boundary.get("is_continuation")),
            "fallback": bool(boundary.get("fallback")),
            "inherited_obligation_count": (
                len(inherited) if isinstance(inherited, list) else 0),
            "executed_slot": executed_slot,
            "has_boundary_candidates": isinstance(candidates_raw, list),
        }


# ---------------------------------------------------------------------------
# finish 侧派生（参数/拓扑在此引入）
# ---------------------------------------------------------------------------

def _quota_deferred(cand: dict) -> bool:
    """M2（2026-09-23 验收审计）：候选被配额判据拒绝（quota_ 前缀理由）
    ——其 cost_ns 为配额前预测成本（SH 侧 M2 修复后保留），重建双域时
    应纳入"配额移除后可行"的经济域。"""
    return str(cand.get("reason") or "").startswith("quota_")


def _min_cost(candidates: dict[int, dict[str, dict]], actions: tuple[str, ...],
              *, include_quota_deferred: bool = True,
              ) -> Optional[tuple[int, int, str]]:
    """(cost, instance, action) 最小（平局镜像在线 argmin 键序：实例小者
    先、再 ACTION_ORDER 偏好——K7/P2-10，sh _quota_filter_candidates 的
    (cost_ns, order_key) 全键同构）。M2：纳入条件 = applicable ∨
    quota_ 前缀拒（配额拒候选保留的配额前成本入经济域——quota-on run
    的 D_econ 不被配额自身裁掉）。N6：include_quota_deferred=False =
    实际可行集口径（在线选择的真实约束——replay 一致性重放用，与
    配额前口径分开）。v9 注（2026-10-20）：本函数返回值是经济域参考
    锚（等价并列内取哪个成员不影响集合口径），非选择预测——选择预测
    面在轮转感知重放（_rotation_aware_selection / actual_ref）。"""
    best: Optional[tuple[int, int, int, str]] = None
    for instance in sorted(candidates):
        for action in actions:
            cand = candidates[instance].get(action)
            if not cand:
                continue
            if not cand["applicable"] and (
                    not include_quota_deferred
                    or not _quota_deferred(cand)):
                continue
            cost = cand["cost_ns"]
            if cost is None:
                continue
            key = (cost, instance, _ACTION_RANK[action], action)
            if best is None or key < best:
                best = key
    if best is None:
        return None
    return (best[0], best[1], best[3])


def _selection_under_delta(delta: int,
                           challenger_min: Optional[tuple[int, int]],
                           alt_star: Optional[tuple[int, int, str]],
                           ) -> tuple[str, Optional[int]]:
    """δ 带下的离线重定价选择（C10 A3' 指令的决策路径口径）。

    谓词 = challenger_flips 同构：挑战者（copy，远读退役后重指派——
    2026-10-14）须以 ≥δ 胜出替代动作族（C_copy + δ ≤ C_alt*——严格
    小于挑战者胜；等号成立按下方全键序裁决）。D_econ 集合口径
    （≤ C_alt*+δ）另计，两者不混。

    O8①（2026-09-23）：跨族平局镜像在线 argmin 全键序（cost_ns,
    (instance_index, action_priority)，joint_scheduler order_key）——
    比较键从裸 cost 扩为 (cost+δ, instance, ACTION_ORDER 秩；未知动作
    秩取 len(ACTION_ORDER)，与 order_key().get 缺省同形)。δ=0 同价时
    实例号小者胜：stay@0=copy@1 在线取 stay@0，重放不再恒翻 copy。
    修前裸 cost 比较在 C_copy==C_alt* 恒翻挑战者，replay_mismatch/
    quota_counterfactual_flips 各计一次假翻转（配额作用量指标被污染）。
    """
    if alt_star is None:
        return ("NA", None)
    if challenger_min is not None:
        challenger_key = (
            challenger_min[0] + delta, challenger_min[1],
            _ACTION_RANK.get(CHALLENGER_ACTION, len(ACTION_ORDER)))
        alt_key = (alt_star[0], alt_star[1],
                   _ACTION_RANK.get(alt_star[2], len(ACTION_ORDER)))
        if challenger_key <= alt_key:
            return (CHALLENGER_ACTION, challenger_min[1])
    return (alt_star[2], alt_star[1])


def _joint_tie_rotation_pick_mirror(
        applicable_triples: list[tuple[int, str, int]],
        tie_cursor: int,
        instance_universe: list[int],
) -> tuple[Optional[str], Optional[int], tuple[int, ...]]:
    """v9（2026-10-20）joint 主臂跨实例平局轮转裁决的离线镜像。

    **语义单一源 = joint/joint_scheduler.py 的模块级纯函数
    ``joint_tie_rotation_pick``**——slo_tools 沿"不 import joint 模块"
    的既有独立常量纪律（运行上下文 sys.path 无 workload 侧包；对账钉测
    ast 锁字面值的既有机制），此处按单一源算法逐句镜像，不得另立漂移
    公式；**镜像义务**：在线函数任何改动必须同步本镜像，跨包对账钉测
    （tests/test_domain_metrics.py::TieRotationMirrorParityTests——ast
    提取在线源函数直接执行对拍）单侧漂移即测试红。输入形态适配：在线
    吃 ActionCandidate 对象序列（调用方已滤 applicable 且 cost_ns 非
    None），本镜像吃等价投影 (instance, action, cost_ns) 三元组；
    order_key 投影 = ACTION_ORDER 秩（joint_cost_model.order_key 的
    (instance, action_priority) 在实例锁定后退化为动作秩）。

    返回 ``(action, instance, tie_instances)``：tie_instances = 适用候选
    中最优 cost_ns（整数 ns 完全相等，无容差）并列实例的去重升序元组；
    |T| >= 2 时自 tie_cursor 起沿 instance_universe 升序环序（越最大值
    回绕；游标不在全集内自首个大于游标的元素起）取首个在集实例，返回
    该实例上最优价并列候选中 ACTION_ORDER 秩最小者的 (action,
    instance)；|T| < 2 返回 (None, None, tie_instances)——单实例平局/
    唯一最优不轮转（与旧确定序恒等）。并列集与全集无交的防御分支取
    T 首元素（本仓调用形态下不可达，如实镜像在线防御分支）。
    """
    best_cost = min(triple[2] for triple in applicable_triples)
    tie_instances = tuple(sorted({
        triple[0] for triple in applicable_triples
        if triple[2] == best_cost}))
    if len(tie_instances) < 2:
        return None, None, tie_instances
    tie_set = set(tie_instances)
    universe = sorted(set(instance_universe))
    start = 0
    for position, instance_index in enumerate(universe):
        if instance_index >= tie_cursor:
            start = position
            break
    # 游标大于全集最大值：环序回绕到首元素（start 保持 0）。
    picked_instance = tie_instances[0]  # 防御默认（并列集 ⊆ 全集时不可达）
    for offset in range(len(universe)):
        candidate_instance = universe[(start + offset) % len(universe)]
        if candidate_instance in tie_set:
            picked_instance = candidate_instance
            break
    action = min(
        (triple[1] for triple in applicable_triples
         if triple[0] == picked_instance and triple[2] == best_cost),
        key=lambda action: _ACTION_RANK.get(action, len(ACTION_ORDER)))
    return action, picked_instance, tie_instances


def _rotation_aware_selection(candidates: dict[int, dict[str, dict]],
                              delta: int, tie_cursor: int, *,
                              include_quota_deferred: bool,
                              ) -> tuple[str, Optional[int]]:
    """v9（2026-10-20）轮转感知重放选择（rotated 行专用；无跨行状态
    ——游标 = 本行 joint_tie_cursor 披露值，不重建调度器游标推进序列）。

    在线 joint 主臂选择规则在 δ 重定价下的镜像：适用候选（M2/N6 双
    口径——include_quota_deferred 选实际可行集/配额前集，与
    _selection_under_delta 的 challenger_min[_actual]/alt_star[_actual]
    同判据）中挑战者（copy）计 cost+δ（C10 A3' 离线诊断口径），取最优
    重定价成本；跨实例并列 |T|>=2 按编号环序轮转裁决（游标 = 本行披露
    值，_joint_tie_rotation_pick_mirror），实例内并列由 ACTION_ORDER
    终裁；|T|<2 走旧确定序全键序 argmin（(cost, instance, ACTION_ORDER
    秩)——与在线单实例平局/唯一最优分支恒等）。无适用候选返回
    ("NA", None)。键缺席行不走本函数（_selection_under_delta 旧路径
    ——输出与旧行为逐位一致的硬性兼容判据；本函数仅在三键在场且
    rotated 且游标非 None 的行进入）。"""
    triples: list[tuple[int, str, int]] = []
    for instance, per_instance in candidates.items():
        for action, cand in per_instance.items():
            if not cand["applicable"] and (
                    not include_quota_deferred
                    or not _quota_deferred(cand)):
                continue
            cost = cand["cost_ns"]
            if cost is None:
                continue
            triples.append((
                instance, action,
                cost + (delta if action == CHALLENGER_ACTION else 0)))
    if not triples:
        return ("NA", None)
    action, instance, _tie_instances = _joint_tie_rotation_pick_mirror(
        triples, int(tie_cursor), sorted(candidates))
    if instance is None:
        best = min(
            triples,
            key=lambda triple: (
                triple[2], triple[0],
                _ACTION_RANK.get(triple[1], len(ACTION_ORDER))))
        return (best[1], best[0])
    return (action, instance)


def _variance(values: list[int]) -> Optional[float]:
    if len(values) < 2:
        return None
    mean = sum(values) / len(values)
    return sum((v - mean) ** 2 for v in values) / len(values)


def _quantiles_or_na(values: list[int]) -> dict:
    if not values:
        return {"n": 0, "p25": NA, "p50": NA, "p75": NA, "p90": NA,
                "min": NA, "max": NA}
    # O8⑤：5 次 nearest_rank_percentile 各带一次全量 sorted——改共享
    # 单排序的 many 版（语义逐值一致：同公式、同 fail 行为）。
    p25, p50, p75, p90 = nearest_rank_percentile_many(
        values, (0.25, 0.50, 0.75, 0.90))
    return {
        "n": len(values),
        "p25": p25, "p50": p50, "p75": p75, "p90": p90,
        "min": min(values), "max": max(values),
    }


def _size_stats(sizes: list[int]) -> dict:
    if not sizes:
        return {"n": 0, "min": NA, "p50": NA, "max": NA, "mean": NA}
    ordered = sorted(sizes)
    return {
        "n": len(sizes), "min": ordered[0],
        "p50": nearest_rank_percentile(sizes, 0.50),
        "max": ordered[-1],
        "mean": round(sum(sizes) / len(sizes), 6),
    }


def _direction_distribution(pairs: list[tuple[str, int]]) -> dict:
    buckets: dict[str, list[int]] = {}
    for direction, hops in pairs:
        buckets.setdefault(direction, []).append(hops)
    return {
        direction: {
            "count": len(hops_list),
            "hop_variance": _variance(hops_list),
        }
        for direction, hops_list in sorted(buckets.items())
    }


def _bottleneck_component(segments: dict[str, Optional[int]]) -> str:
    present = {name: value for name, value in segments.items()
               if value is not None}
    if not present:
        return NA
    top = max(present.values())
    winners = [name for name, value in present.items() if value == top]
    return "+".join(winners)


def _format_members(members: list[tuple[int, int]]) -> str:
    return "|".join(f"{instance}:{hops}" for instance, hops in members)


def _requests_csv_row(entry: dict, topology: dict,
                      tokens: Optional[dict]) -> tuple:
    decision = entry["decision"]
    derived = entry["derived"]
    selected_instance = decision["selected_instance"]
    rho = topology["rho"]
    observed = derived["inner_boundary_hop"]
    coef = topology["coef_bytes_per_token_per_npu"]
    h_bytes = i_bytes = NA
    if coef is not None and tokens is not None:
        history = _int_or_none(tokens.get("history_tokens_before"))
        final = _int_or_none(tokens.get("final_context_tokens"))
        if history is not None:
            h_bytes = coef * history
        if history is not None and final is not None \
                and final >= history:
            i_bytes = coef * (final - history)
    f_min = NA
    if selected_instance is not None:
        f_min = (decision["load_view"].get(selected_instance)
                 or {}).get("hbm_remaining_min", NA)
        if f_min is None:
            f_min = NA
    return (
        decision["request_id"],
        decision["tick"] if decision["tick"] is not None else NA,
        derived["n_instances"],
        derived["reference_instance"] if derived["reference_instance"]
        is not None else NA,
        derived["reference_source"],
        selected_instance if selected_instance is not None else NA,
        decision["selected_action"] or NA,
        derived["selected_hop"] if derived["selected_hop"] is not None
        else NA,
        len(derived["d_feed"]),
        _format_members(derived["d_feed"]),
        derived["c_stay_ref"] if derived["c_stay_ref"] is not None else NA,
        derived["alt_star"][0] if derived["alt_star"] else NA,
        derived["alt_star"][2] if derived["alt_star"] else NA,
        len(derived["d_econ_delta0"]),
        _format_members(derived["d_econ_delta0"]),
        derived["same_position_wins"],
        observed if observed is not None else NA,
        derived["feed_boundary_hop"] if derived["feed_boundary_hop"]
        is not None else NA,
        fmt_ratio(rho) if rho is not None else NA,
        (fmt_ratio(observed - rho)
         if observed is not None and rho is not None else NA),
        decision["active_links"],
        decision["cost_ns"] if decision["cost_ns"] is not None else NA,
        entry["measured_ns"] if entry["measured_ns"] is not None else NA,
        entry.get("measured_service_ns")
        if entry.get("measured_service_ns") is not None else NA,
        entry["err_ns"] if entry["err_ns"] is not None else NA,
        h_bytes, i_bytes, f_min,
    )


def _quota_admissible_flag(per_instance: dict) -> int:
    """K7（P2-11）：quota_admissible = 配额移除后的结构可行域——候选
    applicable 或不可行理由为 quota_ 前缀（C11 后判据可从
    inapplicable_reason 导出；WP3 前恒 NA 注记废止）。C19"双域同图"：
    本列 = 配额侧域，admissible/applied 列 = 实际域，两域之差即配额
    作用量。quota-off run 无 quota_ 理由 ⇒ 退化为结构可行动作存在性。
    """
    return int(any(
        (cand and (cand["applicable"]
                   or str(cand.get("reason") or "").startswith("quota_")))
        for cand in per_instance.values()))


def _sigma_tier_available(sigma_tiers: dict, index: int) -> bool:
    """O8③：敏感档列门控 = 档在 ∧ σ̂ 可估计。σ̂=None 时 derive_entries
    的 tier delta 全为 None（d_econ_tier_sets 落空集）——列如实 NA
    （与 summary 侧 sensitivity_tiers_membership 的 delta_ns=NA 同
    口径；修前只看档数 len(multipliers)>idx，σ̂=None 输出 0，同一
    产物自相矛盾——M2 系列修复后第四处 NA 残留收口）。"""
    return sigma_tiers.get("sigma_hat_ns") is not None \
        and len(sigma_tiers.get("multipliers", [])) > index


def _instances_csv_rows(entry: dict, topology: dict, sigma_tiers: dict):
    decision = entry["decision"]
    derived = entry["derived"]
    anchors = topology["anchors"]
    reference = derived["reference_instance"]
    reference_anchor = anchors.get(reference) if reference is not None \
        else None
    port_rows = decision["port_rows"]
    for instance in sorted(decision["candidates"]):
        per_instance = decision["candidates"][instance]
        challenger = per_instance.get(CHALLENGER_ACTION)
        anchor = anchors.get(instance)
        if reference_anchor is not None and anchor is not None:
            direction = direction_bucket(reference_anchor, anchor)
        else:
            direction = NA
        challenger_applicable = bool(challenger and challenger["applicable"])
        challenger_cost = (challenger["cost_ns"]
                           if challenger_applicable else None)
        c_stay = derived["c_stay_ref"]
        alt_star_cost = derived["alt_star"][0] if derived["alt_star"] \
            else None
        stay = per_instance.get("stay")
        copy = per_instance.get("copy")
        # 二动作下同位置非挑战者 = stay-only（2026-10-15 recompute 退役）。
        same_pos_alts = [
            cand["cost_ns"] for cand in (stay,)
            if cand and cand["applicable"] and cand["cost_ns"] is not None]
        same_pos_best = min(same_pos_alts) if same_pos_alts else None
        port = port_rows.get(instance) or {}
        # N6（2026-09-23 复核审计6）：配额前成本单列——被配额拒（quota_
        # 前缀）的候选保留配额前预测成本（M2 修复后决策行携带），实例
        # CSV 此前列 applicable-only 置 NA ⇒ quota-on run 画不出配额前
        # 成本等值线（计划文档 §5.4"双域同图"的 D_econ 侧输入）。
        challenger_pre_quota_cost = (
            challenger["cost_ns"]
            if challenger and challenger["cost_ns"] is not None
            and (challenger["applicable"] or _quota_deferred(challenger))
            else None)
        # D_feed 与等值线展示的是移除 quota gate 后的结构/经济供给面；
        # 对 quota_ 拒绝项用保留的配额前预测成本。challenger_cost_ns 与
        # same_pos_challenger_pref 仍是实际 applicable-only 口径。
        margin = (challenger_pre_quota_cost - c_stay
                  if challenger_pre_quota_cost is not None
                  and c_stay is not None
                  else None)
        inner_margin = (
            challenger_pre_quota_cost - c_stay
            if challenger_pre_quota_cost is not None
            and c_stay is not None
            else None)
        outer_margin = (
            challenger_pre_quota_cost - alt_star_cost
            if challenger_pre_quota_cost is not None
            and alt_star_cost is not None
            else None)
        quota_admissible = _quota_admissible_flag(per_instance)
        yield (
            decision["request_id"], instance,
            anchor[0] if anchor else NA, anchor[1] if anchor else NA,
            direction,
            challenger["hops"] if challenger
            and challenger["hops"] is not None else NA,
            int(challenger_applicable),
            int(instance in derived["d_feed_set"]),
            challenger_cost if challenger_cost is not None else NA,
            challenger_pre_quota_cost
            if challenger_pre_quota_cost is not None else NA,
            margin if margin is not None else NA,
            fmt_ratio(margin / c_stay)
            if margin is not None and c_stay not in (None, 0) else NA,
            stay["cost_ns"] if stay and stay["applicable"]
            and stay["cost_ns"] is not None else NA,
            copy["cost_ns"] if copy and copy["applicable"]
            and copy["cost_ns"] is not None else NA,
            same_pos_best if same_pos_best is not None else NA,
            (int(challenger_cost <= same_pos_best)
             if challenger_cost is not None and same_pos_best is not None
             else NA),
            inner_margin if inner_margin is not None else NA,
            outer_margin if outer_margin is not None else NA,
            int(instance in derived["d_econ_delta0_set"]),
            int(instance in derived["d_econ_tier_sets"][0])
            if _sigma_tier_available(sigma_tiers, 0) else NA,
            int(instance in derived["d_econ_tier_sets"][1])
            if _sigma_tier_available(sigma_tiers, 1) else NA,
            int(instance in derived["d_econ_tier_sets"][2])
            if _sigma_tier_available(sigma_tiers, 2) else NA,
            port.get("u_port_total", NA),
            quota_admissible,
            (_bottleneck_component(challenger["segments"])
             if challenger_applicable else NA),
            (challenger["reason"] if challenger
             and not challenger["applicable"]
             and challenger["reason"] is not None else NA),
        )


# ---------------------------------------------------------------------------
# σ̂ 估计链（A3'：在线预测噪声尺度；不新建在线估计器，全部离线只读）
# ---------------------------------------------------------------------------

def estimate_sigma_hat(entries: list[dict],
                       warnings: list[str]) -> dict:
    """σ̂ 三级链：A 显式误差披露字段（C15 落地后）→ B 观测 |pred−measured|
    分布 → C 冷启动锚点（I_merge_leg/B_eff 量级 = breakdown.merge_ns 分布）。

    B 级口径警示（输出注释落字）：pred = joint_cost_ns（merge_done 终点
    预测，F11）；measured 配对**同终点**（M3，2026-09-23 验收审计）：
    有 kind="merge_done" 披露行的请求用 merge_done tick−准入 tick（有
    merge 流的形态），否则 completion 行 tick−准入 tick（service_done
    终点——无 merge 流时两者同刻；异终点配对会产生系统性 −M 偏差，
    M=merge 尾段时长）。观测误差含排队/状态演化等全候选共同项，是
    相关噪声的上偏估计，不等于候选间差分噪声。
    """
    result: dict[str, Any] = {
        "sigma_hat_ns": None, "source": NA,
        "observational": None, "merge_leg_anchor": None,
        "identity_note": NA,
    }
    # A 级：C5/C15 schema 已有误差披露字段（现阶段无——列出已核查键名，
    # C15 落地后此处接入，不新建在线估计器）。
    checked_keys = ("prediction_error_ns", "service_factor_error_ns",
                    "horizon_error_ns")
    # B 级：观测配对。
    errs = [entry["err_ns"] for entry in entries
            if entry["err_ns"] is not None]
    if errs:
        abs_errs = sorted(abs(e) for e in errs)
        result["observational"] = {
            "n": len(errs),
            "signed": {"min": min(errs),
                       "p50": nearest_rank_percentile(errs, 0.50),
                       "p90": nearest_rank_percentile(errs, 0.90),
                       "max": max(errs)},
            "abs": {"p50": nearest_rank_percentile(abs_errs, 0.50),
                    "p90": nearest_rank_percentile(abs_errs, 0.90)},
            "measured_anchor": "同终点配对（M3）：merge_done 披露行在"
                               "场 ⇒ merge_done tick − 准入 tick；否则 "
                               "completion 行 tick − 准入 tick（无 merge"
                               " 流时两者同刻）",
        }
        result["sigma_hat_ns"] = nearest_rank_percentile(abs_errs, 0.50)
        result["source"] = "observational_pred_vs_measured_p50_abs"
        result["identity_note"] = (
            "观测 |pred−measured| p50（A3' 授权的决策日志估计路径）；"
            "含排队/状态演化共同项，上偏——分位报告见 "
            "prediction_vs_measured，差分噪声量级待 C4b 受控测量")
    else:
        warnings.append(
            "[domain-metrics] 警告：无 pred/measured 配对样本（无 "
            "completion 行）——σ̂ 走 C 级冷启动锚点")
    # C 级：冷启动锚点（C10 指定被否决来源量级 I_merge_leg/B_eff 的替身，
    # joint_cost_model.py 前向合并腿传输时间量级族）。
    merge_ns_values = sorted(
        cand["segments"]["merge_ns"]
        for entry in entries
        for per_instance in entry["decision"]["candidates"].values()
        for cand in per_instance.values()
        if cand["applicable"]
        and cand["segments"].get("merge_ns") is not None
        and cand["segments"]["merge_ns"] > 0)
    if merge_ns_values:
        anchor = nearest_rank_percentile(merge_ns_values, 0.50)
        result["merge_leg_anchor"] = {
            "n": len(merge_ns_values), "p50": anchor,
            "max": merge_ns_values[-1],
            "identity": "breakdown.merge_ns p50（前向合并腿传输时间量级"
                        "族，JCM:903-911；C10 冷启动替身锚点）",
        }
        if result["sigma_hat_ns"] is None:
            result["sigma_hat_ns"] = anchor
            result["source"] = "merge_leg_cold_start_anchor"
            result["identity_note"] = (
                "C10 冷启动替身：I_merge_leg/B_eff 量级（上界锚点身份"
                "披露）；配对样本出现后自动回到 B 级")
    if result["sigma_hat_ns"] is None:
        warnings.append(
            "[domain-metrics] 警告：σ̂ 不可估计（无配对样本且无 merge 腿"
            "样本）——δ 敏感性重定价降级 NA")
    result["checked_schema_keys"] = list(checked_keys)
    return result


# ---------------------------------------------------------------------------
# cfg1（2026-09-30，§10.3 误差成熟与归因）：P/D 完成预测分组误差报告
# ---------------------------------------------------------------------------

#: 实际预测距离档（§10.3"按实际预测距离"分组的冻结界；与预测参数
#: Δ=100 ms 预设族同源推导的工程档位：near < 1×Δs、mid < 4×Δs、
#: far < 16×Δs、其余 beyond——离线报告预注册，非在线机制）。
#: 对账钉测：前两档与在线 prediction_config.params.
#: forecast_distance_buckets_ns 锁等（slo_tools/tests/test_domain_metrics.
#: py::OnlineOfflineLedgerReconciliationTests，ast 提在线源字面值比对、
#: 不 import joint——任一侧单改即报错）。
PREDICTION_DISTANCE_BANDS_NS = (100_000_000, 400_000_000, 1_600_000_000)
PREDICTION_DISTANCE_BAND_LABELS = ("near", "mid", "far", "beyond")

#: 归一化分母规则（与在线成熟账本 joint/forecast_validation.
#: CompletionMaturityLedger.DENOMINATOR_RULE 同源同字串——离线侧
#: 独立常量登记，slo_tools 不 import joint 模块）。对账钉测：
#: test_domain_metrics.OnlineOfflineLedgerReconciliationTests（ast 提
#: 在线源字面值比对，任一侧单改即报错）。
ERROR_DENOMINATOR_RULE = "(actual-predicted)/max(1,|predicted|) ns"

#: 七分归因的关键词规则（§10.3"误差来源拆为快照陈旧/缺账、物理服务近
#: 似、容量摘要偏差、未知长度、新到达、布局/协议、取整"）。映射对象 =
#: 执行候选 breakdown 的 approximations + notes 词面（小写包含匹配）；
#: 诊断性启发映射——非精确单因归因，一样本可多标签；无命中单列
#: unattributed，不强行归入"历史预测误差"。
ERROR_SOURCE_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("snapshot_stale_or_missing", (
        "ledger_source", "inferred_unknown_eta", "snapshot",
        "summary_maintenance", "absent", "registry", "stale")),
    ("physical_service_approx", (
        "next_token_single_step_service", "decode_aggregate",
        "no_backpressure", "single_step")),
    ("capacity_summary_deviation", (
        "capacity_wait_uncertain", "capacity_summary", "reclaimable",
        "theta")),
    ("unknown_length", (
        "cold_start", "preset", "decode_steps_source",
        "estimated_decode", "length")),
    ("new_arrivals", ("forecast_",)),
    ("layout_protocol", (
        "frozen_diagnostic", "pool_port_rank_proxy", "kv_location_unknown",
        "placement", "layout", "double_buffer", "staging")),
    ("rounding", ("ceil", "round", "tile_align")),
)


def _distance_band(realized_ns: Optional[int]) -> str:
    """实际预测距离档（realized = 决策→settle 实际跨度；None = NA）。"""
    if realized_ns is None:
        return "NA"
    band = 0
    for bound in PREDICTION_DISTANCE_BANDS_NS:
        if realized_ns >= bound:
            band += 1
    return PREDICTION_DISTANCE_BAND_LABELS[band]


def _load_phase_of(decision: dict) -> str:
    """负载阶段（决策行 load_view 的全系统冻结规则；P 行承载，D 行无
    负载快照恒 NA）：任一实例 decode 在跑量 > 0 = decode_active；否则
    任一实例 prefill 队列量 > 0 = prefill_queued；否则 idle；量缺席
    （旧日志）= NA。"""
    phases = [
        entry for entry in decision.get("load_view", {}).values()
        if isinstance(entry, dict)]
    if not phases:
        return "NA"
    if any((entry.get("active_decode_task_load_ns") or 0) > 0
           for entry in phases):
        return "decode_active"
    if any((entry.get("queued_task_load_ns") or 0) > 0
           for entry in phases):
        return "prefill_queued"
    if all(entry.get("active_decode_task_load_ns") is None
           and entry.get("queued_task_load_ns") is None
           for entry in phases):
        return "NA"
    return "idle"


def _settle_tick_for(scan: DomainScan, request_id: str, policy: str,
                     completion: Optional[dict],
                     require_explicit_settle: bool = False) -> Optional[int]:
    """settle 终点解析（与 derive_entries 的 M3/W5 配对规则同判据）：
    现代 run 只消费有效 settle_done；仅旧 consolidate 日志允许等价的
    merge_done/零载荷 service 映射。损坏或先于 service 的结算证据拒绝
    配对，不能靠兼容回退重新变成样本。"""
    if request_id in scan.invalid_settle:
        return None
    settle_tick = scan.settle_done.get(request_id)
    if settle_tick is None and policy != "distributed" and not require_explicit_settle:
        settle_tick = scan.merge_done.get(request_id)
    if (settle_tick is None and policy != "distributed"
            and not require_explicit_settle
            and completion is not None
            and not (completion.get("merge_transferred_bytes") or 0)):
        settle_tick = completion.get("tick")
    if (settle_tick is not None and completion is not None
            and completion.get("tick") is not None
            and settle_tick < completion["tick"]):
        return None
    return settle_tick


def _error_block(samples: list[dict], endpoint: str) -> dict:
    """单组误差聚合（绝对/带方向/分位/归一化分母——§10.3 报告口径；
    空组 = {} 如实空形态）。"""
    errs: list[int] = []
    normalized_signed: list[float] = []
    normalized_abs: list[float] = []
    unpaired = 0
    for sample in samples:
        error = sample["errors"].get(endpoint)
        predicted = sample["predicted"].get(endpoint)
        if error is None:
            unpaired += 1
            continue
        errs.append(int(error))
        denominator = max(1, abs(int(predicted))) if predicted is not None \
            else 1
        normalized_signed.append(error / denominator)
        normalized_abs.append(abs(error) / denominator)
    if not errs and not unpaired:
        return {}
    block: dict = {}
    if errs:
        signed = sorted(errs)
        abs_sorted = sorted(abs(err) for err in errs)
        block["n"] = len(errs)
        block["signed"] = {
            "min": signed[0],
            "p50": nearest_rank_percentile(signed, 0.50),
            "p90": nearest_rank_percentile(signed, 0.90),
            "max": signed[-1],
        }
        block["abs"] = {
            "p50": nearest_rank_percentile(abs_sorted, 0.50),
            "p90": nearest_rank_percentile(abs_sorted, 0.90),
        }
        block["normalized_mean_signed"] = (
            sum(normalized_signed) / len(normalized_signed))
        block["normalized_mean_abs"] = (
            sum(normalized_abs) / len(normalized_abs))
    if unpaired:
        block["unpaired"] = unpaired
    return block


def _grouped_blocks(samples: list[dict], keyfn, endpoint: str) -> dict:
    """按预注册分组键输出误差块（组键排序确定性；NA 组照落——缺输入
    如实披露）。"""
    groups: dict[str, list[dict]] = {}
    for sample in samples:
        groups.setdefault(str(keyfn(sample)), []).append(sample)
    return {key: _error_block(members, endpoint)
            for key, members in sorted(groups.items())}


def _attribution_tags(slot: dict) -> list[str]:
    """七分归因标签（关键词规则映射；一样本多标签；无命中 = 空 →
    上层单列 unattributed）。"""
    tokens = [str(item).lower()
              for item in (slot.get("approximations", [])
                           + slot.get("notes", []))]
    tags = []
    for category, keywords in ERROR_SOURCE_RULES:
        if any(keyword in token for token in tokens
               for keyword in keywords):
            tags.append(category)
    return tags


def completion_error_report(scan: DomainScan, entries: list[dict],
                            settlement_policy: str,
                            future_demand_mode: str,
                            warnings: list[str],
                            require_explicit_settle: bool = False) -> dict:
    """§10.3 完成时间预测的 P/D 分列分组误差报告（cfg1 离线半面）。

    样本 = 实际执行候选的决策（§10.3"在线只有实际执行候选能产生该次
    预测的直接真值"；未执行候选不进样本——missing 不造）：

    * P 样本 = joint_admission 行选中候选，配对 derive_entries 已算的
      measured_ns（settle 终点）/ measured_service_ns（service 终点）；
    * D 样本 = decode 行 boundary_decision 的执行候选（回退/混合身份
      与 sh30 登记侧同判据），配对同款 settle 终点解析。

    分组（settle 终点为主轴；service 终点在 stage_endpoint 主块分列）：
    阶段×终点、实际预测距离档、动作、瓶颈段、负载阶段（P 行承载，D
    行 NA）、已知承诺代理（P=在册已提交通信在途/D=继承义务在册——
    代理口径如实登记）、历史模式（run 级 manifest）。七分归因按
    breakdown approximations/notes 关键词规则（启发式映射，规则随
    披露；无命中单列 unattributed）。
    """
    p_samples: list[dict] = []
    service_unpaired_p = 0
    for entry in entries:
        decision = entry["decision"]
        instance = decision.get("selected_instance")
        action = decision.get("selected_action")
        slot = (decision.get("candidates", {}).get(instance, {})
                .get(action or ""))
        if slot is None:
            continue
        predicted_settle = slot.get("predicted_merge_done")
        predicted_service = slot.get("predicted_service_done")
        measured_settle = entry.get("measured_ns")
        measured_service = entry.get("measured_service_ns")
        if measured_service is None:
            service_unpaired_p += 1
        p_samples.append({
            "request_id": decision["request_id"],
            "errors": {
                "settle": (measured_settle - predicted_settle
                           if measured_settle is not None
                           and predicted_settle is not None else None),
                "service": (measured_service - predicted_service
                            if measured_service is not None
                            and predicted_service is not None else None),
            },
            "predicted": {"settle": predicted_settle,
                          "service": predicted_service},
            "realized_distance_ns": measured_settle,
            "action": action,
            "bottleneck": _bottleneck_component(slot["segments"]),
            "load_phase": _load_phase_of(decision),
            "known_commit": (
                "with_submitted_comms"
                if decision.get("active_links") else "without"),
            "approximations": slot.get("approximations", []),
            "notes": slot.get("notes", []),
        })
    d_samples: list[dict] = []
    skipped_d = 0
    for ddec in scan.d_decisions:
        slot = ddec.get("executed_slot")
        if slot is None:
            # 一次映射臂（无 D 计价）/身份未匹配：单列不造样本。
            skipped_d += 1
            continue
        completion = scan.completions.get(ddec["request_id"])
        tick = ddec.get("tick")
        measured_service = (
            completion["tick"] - tick
            if completion is not None and completion.get("tick") is not None
            and tick is not None and completion["tick"] >= tick else None)
        settle_tick = _settle_tick_for(
            scan, ddec["request_id"], settlement_policy, completion,
            require_explicit_settle)
        if settle_tick is None and completion is not None and (
                completion.get("merge_transferred_bytes") or 0) > 0:
            message = (
                "settle endpoint row missing for D decision of request {} "
                "(endpoint pairing unreliable; D settle error -> NA)"
                .format(ddec["request_id"]))
            if message not in warnings:
                warnings.append(message)
        measured_settle = (
            settle_tick - tick if settle_tick is not None
            and tick is not None and settle_tick >= tick else None)
        predicted_settle = slot.get("predicted_merge_done")
        predicted_service = slot.get("predicted_service_done")
        face = ddec.get("selected_action_face") or ddec.get(
            "selected_action")
        d_samples.append({
            "request_id": ddec["request_id"],
            "errors": {
                "settle": (measured_settle - predicted_settle
                           if measured_settle is not None
                           and predicted_settle is not None else None),
                "service": (measured_service - predicted_service
                            if measured_service is not None
                            and predicted_service is not None else None),
            },
            "predicted": {"settle": predicted_settle,
                          "service": predicted_service},
            "realized_distance_ns": measured_settle,
            "action": face,
            "bottleneck": _bottleneck_component(slot["segments"]),
            "load_phase": "NA",
            "known_commit": (
                "with_inherited_obligations"
                if ddec.get("inherited_obligation_count") else "without"),
            "approximations": slot.get("approximations", []),
            "notes": slot.get("notes", []),
        })

    def _attribution(samples: list[dict]) -> dict:
        counts: dict[str, int] = {}
        attributed = 0
        for sample in samples:
            tags = _attribution_tags(sample)
            if tags:
                attributed += 1
            for tag in tags:
                counts[tag] = counts.get(tag, 0) + 1
        counts["unattributed"] = len(samples) - attributed
        return dict(sorted(counts.items()))

    by_stage_endpoint: dict[str, dict] = {}
    for stage, samples in (("P", p_samples), ("D", d_samples)):
        by_stage_endpoint[stage] = {
            "registered": len(samples),
            "settle": _error_block(samples, "settle"),
            "service": _error_block(samples, "service"),
        }
    by_stage_endpoint["P"]["service_unpaired"] = service_unpaired_p
    return {
        "caliber": {
            "denominator_rule": ERROR_DENOMINATOR_RULE,
            "sample_rule": "仅实际执行候选进样本（未执行候选保持 "
                           "missing）；误差 = 实际 − 预测（决策相对轴）",
            "settle_endpoint_source": (
                "settle_done 行主源；consolidate 旧日志 merge_done 等价"
                "映射；零载荷 consolidate 与 service 同刻；distributed "
                "缺 settle_done 行恒 NA（§9.3）"),
            "distance_bands_ns": list(PREDICTION_DISTANCE_BANDS_NS),
            "distance_band_note": "实际预测距离 = 决策→settle 实际跨度"
                                  "（realized），档界为离线预注册工程"
                                  "档（Δ=100ms 预设族同源）",
            "load_phase_rule": "任一实例 decode 在跑量>0 = decode_active"
                               "；否则任一 prefill 队列量>0 = "
                               "prefill_queued；否则 idle；P 行承载，"
                               "D 行无负载快照恒 NA",
            "known_commit_proxy": "已知承诺占比的代理口径：P = 决策时刻"
                                  "在册已提交通信在途（flow_snapshot）；"
                                  "D = 继承义务在册（inherited_"
                                  "obligations）——二值代理，非连续占比"
                                  "（行内无连续承诺占比披露，如实登记）",
            "grouping_endpoint_note": "分组维度以 settle 终点误差为主轴；"
                                      "service 终点在 stage_endpoint "
                                      "主块 P/D 分列",
        },
        "future_demand_mode": future_demand_mode,
        "samples": {
            "P": len(p_samples),
            "D": len(d_samples),
            "D_skipped_no_executed_breakdown": skipped_d,
        },
        "stage_endpoint": by_stage_endpoint,
        "by_realized_distance_band": {
            "P": _grouped_blocks(
                p_samples, lambda s: _distance_band(
                    s["realized_distance_ns"]), "settle"),
            "D": _grouped_blocks(
                d_samples, lambda s: _distance_band(
                    s["realized_distance_ns"]), "settle"),
        },
        "by_action": {
            "P": _grouped_blocks(p_samples, lambda s: s["action"],
                                 "settle"),
            "D": _grouped_blocks(d_samples, lambda s: s["action"],
                                 "settle"),
        },
        "by_bottleneck_segment": {
            "P": _grouped_blocks(p_samples, lambda s: s["bottleneck"],
                                 "settle"),
            "D": _grouped_blocks(d_samples, lambda s: s["bottleneck"],
                                 "settle"),
        },
        "by_load_phase": {
            "P": _grouped_blocks(p_samples, lambda s: s["load_phase"],
                                 "settle"),
            "D": _grouped_blocks(d_samples, lambda s: s["load_phase"],
                                 "settle"),
        },
        "by_known_commitment": {
            "P": _grouped_blocks(p_samples, lambda s: s["known_commit"],
                                 "settle"),
            "D": _grouped_blocks(d_samples, lambda s: s["known_commit"],
                                 "settle"),
        },
        "by_history_mode": {
            str(future_demand_mode): {
                "P": _error_block(p_samples, "settle"),
                "D": _error_block(d_samples, "settle"),
            },
        },
        "error_source_attribution": {
            "rules": {
                category: list(keywords)
                for category, keywords in ERROR_SOURCE_RULES},
            "rule_note": "诊断性关键词启发映射（breakdown "
                         "approximations/notes 词面）——非精确单因归因，"
                         "一样本可多标签；unattributed = 无命中样本数，"
                         "不强行归入历史预测误差",
            "P": _attribution(p_samples),
            "D": _attribution(d_samples),
        },
    }


def _resolve_future_demand_mode(run_dir, warnings: list[str]) -> str:
    """cfg1：run 级 joint_mechanism_manifest.json prediction 节的
    future_demand_mode（历史模式分组输入；缺失/不可读 = 缺省 none，
    与 _resolve_settlement_policy 同款读取面与降级姿态）。"""
    prediction = _read_prediction_manifest(run_dir)
    if prediction is None:
        return "none"
    mode = prediction.get("future_demand_mode", "none")
    if mode not in ("none", "recent", "reverting"):
        fail("joint prediction manifest has an invalid future_demand_mode")
    return mode


# ---------------------------------------------------------------------------
# home 迁移轨迹（决策时刻 completion 披露 + 结算时刻 kv_delta_journal）
# ---------------------------------------------------------------------------

def home_migration_block(scan: DomainScan, entries: list[dict],
                         run_dir: Path) -> dict:
    return {
        "decision_log_completion": _completion_trajectory(scan, entries),
        "kv_delta_journal": _kv_delta_trajectory(scan, run_dir),
    }


def _kv_delta_trajectory(scan: DomainScan, run_dir: Path) -> dict:
    """kv_delta_journal 消费面（C14 结算时刻逐请求合并披露；F3 接通）。

    数据通路（F3 收口，C14 §20.6-2 / C16 §22.7-1 移交义务履行）：C14 在
    FS 侧产出 KVCacheManager.kv_delta_journal（行字段含 direction/
    zero_byte_flip/home_before/home_after/home_migration/transferred_bytes/
    两侧实际保留量/staging_return_bytes），F3 起 run 级 sidecar 序列化落
    bridge/joint_kv_ledgers.json 的 "kv_delta_journal" 键（dump_joint_
    kv_ledgers 第四键，行源 = kv_delta_journal_rows）。读取按四层可信度
    分级（常量区注记）：status = tier 名（非 ok/TODO 二值），顶层缺口
    （无守恒证书）如实披露。

    驻留时间口径：journal 行无 tick——以 trigger_request_id 联
    completion 行 tick 的近似（披露），不可联时 NA（计入 rows_unjoined）。
    """
    sidecar = run_dir / "bridge" / "joint_kv_ledgers.json"
    if not sidecar.is_file():
        return {
            "status": KV_DELTA_TIER_DECISION_ONLY,
            "rows": 0,
            "note": "sidecar 缺席（bridge/joint_kv_ledgers.json）——home "
                    "迁移轨迹仅决策时刻 completion 披露；无法区分"
                    "「零结算 run」与「F3 前未序列化 run」（证据最弱层）",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    try:
        data = json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {
            "status": KV_DELTA_TIER_DECISION_ONLY,
            "rows": 0,
            "note": "sidecar 在场但不可解析（损坏/半截 JSON）——同 "
                    "decision_log_only 退决策时刻披露；dump 原子写"
                    "（tmp + os.replace）下半截不应出现，建议复核落盘日志",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    if not isinstance(data, dict):
        # A12'：顶层结构非对象 = schema 损坏，非"零结算 run"——按最弱
        # 层退决策时刻披露，不得虚标 settlement_empty 证据等级。
        return {
            "status": KV_DELTA_TIER_DECISION_ONLY,
            "rows": 0,
            "note": "sidecar 顶层结构非对象（schema 损坏）——按最弱层退"
                    "决策时刻披露；dump 原子写（tmp + os.replace）下不应"
                    "出现，建议复核落盘日志",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    export_error = data.get("kv_delta_journal_export_error")
    if isinstance(export_error, str):
        # A12'：生产端逐键哨兵——kv_delta_journal 导出失败（seq 链
        # fail-closed 等）显式落盘，其余键已按各自通道落盘消费。
        return {
            "status": KV_DELTA_TIER_DECISION_ONLY,
            "rows": 0,
            "note": "kv_delta_journal 导出失败（生产端 fail-closed 触发）："
                    + export_error
                    + " ——本通道退决策时刻披露",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    if "kv_delta_journal" not in data:
        return {
            "status": KV_DELTA_TIER_EMPTY,
            "rows": 0,
            "sidecar_keys": sorted(data),
            "note": "sidecar 在场但 kv_delta_journal 键缺席——F3 前旧 "
                    "sidecar schema；home 迁移轨迹仅决策时刻 completion "
                    "披露",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    rows = data["kv_delta_journal"]
    if not isinstance(rows, list):
        # A14'（H7，2026-09-22 第三轮复审）：键在而值非列表 = 值级
        # schema 损坏——顶层 A12' 原则（损坏不得虚标 settlement_empty
        # 证据等级）在值级贯彻，按最弱层退决策时刻披露。
        return {
            "status": KV_DELTA_TIER_DECISION_ONLY,
            "rows": 0,
            "sidecar_keys": sorted(data),
            "note": "kv_delta_journal 键值非列表（schema 损坏）——按最弱"
                    "层退决策时刻披露；dump 原子写下不应出现，建议复核"
                    "落盘日志",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    if not rows:
        return {
            "status": KV_DELTA_TIER_EMPTY,
            "rows": 0,
            "sidecar_keys": sorted(data),
            "note": "sidecar 在场但 kv_delta_journal 空列表——"
                    "零结算 run（journal 行存在 ⇔ 结算完成，C14 §20.3）；"
                    "home 迁移轨迹仅决策时刻 completion 披露",
            "tier_note": KV_DELTA_TIER_NOTE,
        }
    completions = scan.completions
    direction_counts: dict[str, int] = {}
    zero_byte_flips = 0
    migrations: list[dict] = []
    session_homes: dict[str, list] = {}
    transferred_total = 0
    rows_unjoined = 0
    previous_seq: Optional[int] = None
    for row in rows:
        if not isinstance(row, dict):
            fail(f"kv_delta_journal 行必须是对象：{sidecar}")
        seq = row.get("seq")
        if (not isinstance(seq, int)
                or (previous_seq is not None and seq <= previous_seq)):
            fail(f"kv_delta_journal seq 链断裂（须 int 严格增）：{sidecar}")
        previous_seq = seq
        direction = str(row.get("direction"))
        direction_counts[direction] = direction_counts.get(direction, 0) + 1
        if row.get("zero_byte_flip"):
            zero_byte_flips += 1
        nbytes = _int_or_none(row.get("transferred_bytes"))
        if nbytes:
            transferred_total += nbytes
        request_id = str(row.get("trigger_request_id"))
        tick = (completions.get(request_id) or {}).get("tick")
        if tick is None:
            rows_unjoined += 1
        migrated = bool(row.get("home_migration")) or (
            row.get("home_before") is not None
            and row.get("home_after") is not None
            and row.get("home_before") != row.get("home_after"))
        migrations.append({
            "request_id": request_id,
            "tick": tick,
            "direction": direction,
            "home_before": row.get("home_before"),
            "home_after": row.get("home_after"),
            "home_migration": migrated,
            "transferred_bytes": nbytes,
            "home_side_retained_bytes":
                row.get("home_side_retained_bytes"),
            "exec_side_retained_bytes":
                row.get("exec_side_retained_bytes"),
            "staging_return_bytes": row.get("staging_return_bytes"),
        })
        session = str(row.get("session_id") or
                      request_id.rsplit("_request_", 1)[0])
        session_homes.setdefault(session, []).append(
            {"tick": tick, "home": row.get("home_after")})
    dwell_samples: list[int] = []
    unjoined = 0
    for series in session_homes.values():
        series.sort(key=lambda item: (item["tick"] is None,
                                      item["tick"] or 0))
        for previous, current in zip(series, series[1:]):
            if previous["tick"] is not None and current["tick"] is not None:
                dwell_samples.append(current["tick"] - previous["tick"])
            else:
                unjoined += 1
    block: dict[str, Any] = {}
    block.update({
        "status": (
            KV_DELTA_TIER_FULL if rows_unjoined == 0
            else KV_DELTA_TIER_PARTIAL),
        "rows": len(rows),
        "rows_unjoined": rows_unjoined,
        "direction_counts": direction_counts,
        "zero_byte_flip_count": zero_byte_flips,
        "home_migration_count": sum(1 for m in migrations if m["home_migration"]),
        "transferred_bytes_total": transferred_total,
        "sessions": len(session_homes),
        "sessions_with_multiple_homes": sum(
            1 for series in session_homes.values()
            # A12'：过滤 None home（与 _completion_trajectory 的
            # is not None 口径对齐——生产端 stay 行已落 home_before，
            # 此处为对旧 sidecar/异常形态的防御）。
            if len({item["home"] for item in series
                    if item["home"] is not None}) > 1),
        "home_dwell_ns": _quantiles_or_na(dwell_samples),
        "dwell_unjoined": unjoined,
        "dwell_caliber": "journal 行无 tick——trigger_request_id 联 "
                         "completion 行 tick 的近似",
        "per_request": migrations,
        "tier_note": KV_DELTA_TIER_NOTE,
        "ping_pong_note": "是否减少乒乓由结果判断（§5.3），不由少并多"
                          "规则直接保证",
    })
    return block


def _completion_trajectory(scan: DomainScan, entries: list[dict]) -> dict:
    by_request = {entry["decision"]["request_id"]: entry
                  for entry in entries}
    merges: dict[str, int] = {}
    home_changes = 0
    migrated_bytes = 0
    session_homes: dict[str, set] = {}
    session_series: dict[str, list] = {}
    for request_id, completion in sorted(scan.completions.items(),
                                         key=lambda kv: kv[1]["tick"]
                                         or 0):
        entry = by_request.get(request_id)
        session = request_id.rsplit("_request_", 1)[0]
        direction = completion.get("merge_direction")
        if isinstance(direction, str):
            merges[direction] = merges.get(direction, 0) + 1
        flipped = _int_or_none(completion.get("home_flipped_to"))
        origin = entry["decision"]["origin_home_instance"] if entry else None
        after = _int_or_none(completion.get("kv_instance_after_completion"))
        if flipped is not None:
            home_changes += 1
        nbytes = completion.get("merge_transferred_bytes")
        if nbytes:
            migrated_bytes += nbytes
        if after is not None:
            session_homes.setdefault(session, set()).add(after)
            session_series.setdefault(session, []).append(
                {"tick": completion["tick"], "home": after,
                 "home_flipped_to": flipped,
                 "origin_home_at_admission": origin})
    multi_home = sorted(s for s, homes in session_homes.items()
                        if len(homes) > 1)
    return {
        "note": "决策时刻披露（C3b completion 行）——merge 方向/字节为"
                "决策日志口径；零字节结算与逐请求合并细账以 kv_delta_"
                "journal 为准（C14）",
        "merge_direction_counts": merges,
        "home_migration_count": home_changes,
        "merge_transferred_bytes_total": migrated_bytes,
        "sessions": len(session_homes),
        "sessions_with_multiple_homes": len(multi_home),
        "sessions_with_multiple_homes_ids": multi_home,
        "ping_pong_note": "是否减少乒乓由结果判断（§5.3），不由少并多"
                          "规则直接保证——多 home 会话清单已列",
        "session_trajectories": session_series,
    }


# ---------------------------------------------------------------------------
# 主派生：三口径 + 全部派生指标（一次遍历 entries）
# ---------------------------------------------------------------------------

def _read_prediction_manifest(run_dir) -> Optional[dict]:
    """读取生产 bridge 侧车；仅旧归档位置允许回落到 run 根。"""
    if not run_dir:
        return None
    root = Path(run_dir)
    path = root / "bridge" / "joint_mechanism_manifest.json"
    if not path.is_file():
        path = root / "joint_mechanism_manifest.json"
    if not path.is_file():
        return None
    try:
        with path.open(encoding="utf-8") as source:
            payload = json.load(source)
    except (OSError, ValueError) as exc:
        fail(f"cannot read joint prediction manifest {path}: {exc}")
    if not isinstance(payload, dict):
        fail(f"joint manifest must be an object: {path}")
    prediction = payload.get("prediction")
    if prediction is not None and not isinstance(prediction, dict):
        fail(f"joint prediction manifest must be an object: {path}")
    return prediction


def _resolve_settlement_policy(run_dir, warnings) -> str:
    """W5：run 级 joint_mechanism_manifest.json 的 prediction.settlement_
    policy（W5 起随 manifest 落盘）；缺失 = 旧 run（consolidate 缺省）。
    """
    prediction = _read_prediction_manifest(run_dir)
    if prediction is None:
        return "consolidate"
    policy = prediction.get("settlement_policy", "consolidate")
    if policy not in ("consolidate", "distributed"):
        fail("joint prediction manifest has an invalid settlement_policy")
    return policy


def derive_entries(scan: DomainScan, topology: dict, sigma_hat_ns,
                   multipliers: list,
                   warnings: Optional[list[str]] = None,
                   settlement_policy: str = "consolidate",
                   require_explicit_settle: bool = False,
                   ) -> list[dict]:
    """W5（§7.4/§9.3）：``settlement_policy`` 来自 run 级 manifest
    （joint_mechanism_manifest.json prediction.settlement_policy；缺省
    consolidate = 旧 run 兼容）。distributed 下 settle 终点只认
    settle_done 行——禁止 completion=settle 回退。"""
    entries: list[dict] = []
    for decision in scan.decisions:
        candidates = decision["candidates"]
        n_instances = len(candidates)
        alt_star = _min_cost(candidates, ALT_ACTIONS)
        stay_best = _min_cost(candidates, ("stay",))
        challenger_candidates = [
            (instance, cand)
            for instance, per_instance in candidates.items()
            for action, cand in per_instance.items()
            # M2：D_feed/D_econ 纳入配额拒候选（quota_ 前缀理由 +
            # 保留的配额前成本）——quota-on run 的经济域可重建。
            if action == CHALLENGER_ACTION
            and (cand["applicable"] or _quota_deferred(cand))
            and cand["cost_ns"] is not None]
        challenger_min = min(
            ((cand["cost_ns"], instance)
             for instance, cand in challenger_candidates), default=None)
        # N6：实际可行集口径（applicable-only——在线选择的真实约束）。
        # 配额前口径（challenger_min/alt_star 供 D_econ/双域重建）与实际
        # 口径（供 replay 一致性重放）分列，replay_mismatch 不再混入
        # 配额作用量（配额拒挑战者更便宜时：实际域重放与在线同选 ⇒ 不
        # 计 mismatch；差异进 quota_counterfactual_flips）。
        challenger_min_actual = min(
            ((cand["cost_ns"], instance)
             for instance, per_instance in candidates.items()
             for action, cand in per_instance.items()
             if action == CHALLENGER_ACTION and cand["applicable"]
             and cand["cost_ns"] is not None), default=None)
        alt_star_actual = _min_cost(
            candidates, ALT_ACTIONS,
            include_quota_deferred=False)
        if stay_best is not None:
            reference_instance = stay_best[1]
            reference_source = "min_cost_stay"
            c_stay_ref = stay_best[0]
        elif alt_star is not None:
            reference_instance = alt_star[1]
            reference_source = "min_cost_alt(no_stay_applicable)"
            c_stay_ref = None
        else:
            reference_instance = decision["selected_instance"]
            reference_source = "selected(no_alt_applicable)"
            c_stay_ref = None
        d_feed = sorted(
            (instance, cand["hops"] if cand["hops"] is not None else -1)
            for instance, cand in challenger_candidates)
        d_feed_set = {instance for instance, _ in d_feed}
        d_econ_delta0: list[tuple[int, int]] = []
        if alt_star is not None:
            d_econ_delta0 = sorted(
                (instance, hops) for instance, hops in d_feed
                if challenger_cost_of(candidates, instance) <= alt_star[0])
        tier_deltas = []
        for multiplier in multipliers:
            delta = (int(sigma_hat_ns * multiplier)
                     if sigma_hat_ns is not None else None)
            tier_deltas.append(delta)
            # D_econ 集合口径（≤ C_alt*+δ）——仅记录，行级成员在 CSV。
        d_econ_tier_sets = []
        for delta in tier_deltas:
            if alt_star is None or delta is None:
                d_econ_tier_sets.append(set())
            else:
                d_econ_tier_sets.append({
                    instance for instance, _ in d_feed
                    if challenger_cost_of(candidates, instance)
                    <= alt_star[0] + delta})
        same_position_wins = 0
        for instance, per_instance in candidates.items():
            challenger = per_instance.get(CHALLENGER_ACTION)
            if not challenger or not challenger["applicable"] \
                    or challenger["cost_ns"] is None:
                continue
            same_alts = [
                cand["cost_ns"] for action, cand in per_instance.items()
                if action != CHALLENGER_ACTION and cand["applicable"]
                and cand["cost_ns"] is not None]
            if same_alts and challenger["cost_ns"] <= min(same_alts):
                same_position_wins += 1
        inner_boundary_hop = None
        if c_stay_ref is not None:
            hops_in = [hops for instance, hops in d_feed
                       if challenger_cost_of(candidates, instance)
                       <= c_stay_ref
                       and hops >= 0]
            if hops_in:
                inner_boundary_hop = max(hops_in)
        feed_hops = [hops for _, hops in d_feed if hops >= 0]
        selected_hop = None
        if decision["selected_instance"] is not None:
            cand = (candidates.get(decision["selected_instance"], {})
                    .get(decision["selected_action"] or ""))
            if cand and cand["hops"] is not None:
                selected_hop = cand["hops"]
        # v9（2026-10-20）轮转感知重放（逐行、无跨行状态）：三键在场且
        # rotated 且游标非 None 时 actual_ref = 由本行候选表现算的轮转
        # 公式期望选择（适用候选口径——U = 候选表实例全集升序、T = 适用
        # 候选中最优价并列实例升序），actual != expected 即真发散
        # （build_summary 计 replay_mismatch）；否则 actual_ref = actual
        # ——键缺席（旧 run）/未轮转行与旧口径逐位一致（硬性兼容判据）。
        rotation_active = bool(
            decision.get("tie_keys_present")
            and decision.get("tie_rotated")
            and decision.get("tie_cursor") is not None)
        actual_ref_action = decision["selected_action"]
        actual_ref_instance = decision["selected_instance"]
        rotation_consistency_mismatch = False
        cursor_anomaly = False
        if rotation_active:
            if decision["tie_cursor"] not in candidates:
                # 防御：游标不在本行候选表实例全集（U）内——轮转公式全
                # 定义不崩（自首个大于游标的元素起/回绕），计数告警
                # （summary 的 rotation_aware_replay 披露）。
                cursor_anomaly = True
            expected_action, expected_instance = _rotation_aware_selection(
                candidates, 0, decision["tie_cursor"],
                include_quota_deferred=False)
            actual_ref_action = expected_action
            actual_ref_instance = expected_instance
            rotation_consistency_mismatch = (
                expected_action != decision["selected_action"]
                or expected_instance != decision["selected_instance"])
        # 比较基准的位移锚点（build_summary 的 displacement_hop 消费）：
        # actual_ref 候选的 hops——键缺席行与 selected_hop 同值同查表。
        selected_hop_ref = None
        if actual_ref_instance is not None:
            ref_cand = (candidates.get(actual_ref_instance, {})
                        .get(actual_ref_action or ""))
            if ref_cand and ref_cand["hops"] is not None:
                selected_hop_ref = ref_cand["hops"]
        completion = scan.completions.get(decision["request_id"])
        # M3（2026-09-23 验收审计）：双终点分列——pred（joint_cost_ns）
        # 是 merge_done 终点预测（F11 冻结），completion 行 tick 是
        # service_done 终点（C3b 新合同在 merge 前）：有 merge_done
        # 披露行的请求以 merge 终点配对（同终点），否则 service 终点
        # （无 merge 流时两者同刻）——异终点配对会产生系统性 −M 偏差
        # （M = merge 尾段时长）污染 σ̂ 与 δ 敏感性。
        # W5（§7.4/§9.3 settle_done 统一语义）：settle 终点主源 =
        # settle_done 行；consolidate 旧日志缺该行时按已核实等价映射
        # merge_done（同一 watch 交付事件，settle_done_ns ==
        # merge_done_ns）；distributed 策略**不得**回退 completion 行
        # （§9.3"distributed 可以没有 merge 事件而仍有 settle"——缺
        # settle_done 行即 NA + 告警，不把 service 终点冒充 settle 终点）。
        request_settle_policy = settlement_policy
        settle_tick = _settle_tick_for(
            scan, decision["request_id"], request_settle_policy, completion,
            require_explicit_settle)
        merge_done_tick = scan.merge_done.get(decision["request_id"])
        measured_service_ns = measured_ns = err_ns = None
        if completion and completion["tick"] is not None \
                and decision["tick"] is not None \
                and completion["tick"] >= decision["tick"]:
            measured_service_ns = completion["tick"] - decision["tick"]
        # N9（2026-09-23 复核审计·M3 缺行分支）：completion 行显示 merge
        # 字节（merge_transferred_bytes > 0 ⇒ 有 merge 义务）而 settle/
        # merge 行缺失（丢行/被过滤的日志）——service 终点早于 settle
        # 完成，配对会引入系统性 −M 偏差：该请求 measured_ns 置 NA +
        # 告警，不静默退 service 终点。W5 起判据键改 settle 终点行族
        # （settle_done 主源 + 旧 merge_done 兼容映射）。
        settle_row_missing = bool(
            completion and (completion.get("merge_transferred_bytes") or 0) > 0
            and settle_tick is None and merge_done_tick is None)
        if settle_row_missing and warnings is not None:
            message = (
                "settle endpoint row missing for request {} with merge bytes "
                "(endpoint pairing unreliable; measured_ns -> NA)".format(
                    decision["request_id"]))
            if message not in warnings:
                warnings.append(message)
        if decision["tick"] is not None and decision["cost_ns"] is not None:
            if settle_tick is not None and settle_tick >= decision["tick"]:
                measured_ns = settle_tick - decision["tick"]
            elif ((request_settle_policy == "distributed" or require_explicit_settle)
                    and settle_tick is None and warnings is not None):
                message = (
                    "{} settlement without settle_done row for "
                    "request {}: no completion=settle fallback (sec.9.3); "
                    "measured_ns -> NA".format(
                        request_settle_policy, decision["request_id"]))
                if message not in warnings:
                    warnings.append(message)
            if measured_ns is not None:
                err_ns = measured_ns - decision["cost_ns"]
        entries.append({
            "decision": decision,
            "measured_ns": measured_ns,
            "measured_service_ns": measured_service_ns,
            "err_ns": err_ns,
            "derived": {
                "n_instances": n_instances,
                "alt_star": alt_star,
                "challenger_min_actual": challenger_min_actual,
                "alt_star_actual": alt_star_actual,
                "c_stay_ref": c_stay_ref,
                "reference_instance": reference_instance,
                "reference_source": reference_source,
                "challenger_min": challenger_min,
                "d_feed": d_feed,
                "d_feed_set": d_feed_set,
                "d_econ_delta0": d_econ_delta0,
                "d_econ_delta0_set": {i for i, _ in d_econ_delta0},
                "d_econ_tier_sets": d_econ_tier_sets,
                "tier_deltas": tier_deltas,
                "same_position_wins": same_position_wins,
                "inner_boundary_hop": inner_boundary_hop,
                "feed_boundary_hop": max(feed_hops) if feed_hops else None,
                "selected_hop": selected_hop,
                # v9 轮转感知重放的逐行承载面（build_summary 消费）。
                "tie_keys_present": bool(decision.get("tie_keys_present")),
                "rotation_active": rotation_active,
                "actual_ref_action": actual_ref_action,
                "actual_ref_instance": actual_ref_instance,
                "selected_hop_ref": selected_hop_ref,
                "rotation_consistency_mismatch":
                    rotation_consistency_mismatch,
                "cursor_anomaly": cursor_anomaly,
            },
        })
    return entries


def challenger_cost_of(candidates: dict[int, dict[str, dict]],
                       instance: int) -> int:
    """挑战者（copy）候选成本；缺候选/缺成本 = +inf（域谓词恒假）。"""
    cand = candidates[instance].get(CHALLENGER_ACTION)
    return cand["cost_ns"] if cand and cand["cost_ns"] is not None \
        else math.inf


def build_summary(scan: DomainScan, entries: list[dict], topology: dict,
                  repo_variant: str, delta_adm: int, multipliers: list,
                  sigma: dict, warnings: list[str]) -> dict:
    anchors = topology["anchors"]
    # -- 实际选择：两动作选中计数（stay/copy 单列，§19.2；2026-10-15
    #    recompute 退役后无 elected/forced 分面——旧键随 D5 删除） --
    action_counts = {action: 0 for action in ALL_ACTIONS}
    for entry in entries:
        decision = entry["decision"]
        action = decision["selected_action"]
        if action is None:
            continue
        action_counts[action] += 1

    # -- 三口径集合统计 --
    feed_sizes = [len(e["derived"]["d_feed"]) for e in entries]
    econ_sizes = [len(e["derived"]["d_econ_delta0"]) for e in entries]
    feed_hop_pairs: list[tuple[str, int]] = []
    econ_hop_pairs: list[tuple[str, int]] = []
    feed_hops: list[int] = []
    econ_hops: list[int] = []
    for entry in entries:
        reference = entry["derived"]["reference_instance"]
        anchor = anchors.get(reference) if reference is not None else None
        for member, hops in entry["derived"]["d_feed"]:
            # O8②：hops=None → -1 哨兵（derive_entries 存储形态）不进
            # 聚合——与边界计算（内边界 / feed 边界 同
            # hops>=0 口径），修前 -1 直接进 hop_quantiles/direction_
            # distribution（及 econ 侧同名聚合）。
            if hops < 0:
                continue
            feed_hops.append(hops)
            if anchor is not None and member in anchors:
                feed_hop_pairs.append(
                    (direction_bucket(anchor, anchors[member]), hops))
        for member, hops in entry["derived"]["d_econ_delta0"]:
            if hops < 0:
                continue
            econ_hops.append(hops)
            if anchor is not None and member in anchors:
                econ_hop_pairs.append(
                    (direction_bucket(anchor, anchors[member]), hops))

    # -- 瓶颈资源 / 约束不满足原因 / 适用计数 --
    bottleneck_feed: dict[str, int] = {}
    bottleneck_selected: dict[str, int] = {}
    reasons: dict[str, dict[str, int]] = {}
    applicable_counts: dict[str, int] = {a: 0 for a in ALL_ACTIONS}
    applicable_mismatch = 0
    for entry in entries:
        decision = entry["decision"]
        feed_set = entry["derived"]["d_feed_set"]
        for instance, per_instance in decision["candidates"].items():
            for action, cand in per_instance.items():
                if cand["applicable"]:
                    applicable_counts[action] += 1
                else:
                    reason = str(cand["reason"])
                    reasons.setdefault(action, {})
                    reasons[action][reason] = \
                        reasons[action].get(reason, 0) + 1
                if action == CHALLENGER_ACTION and cand["applicable"]:
                    key = _bottleneck_component(cand["segments"])
                    if instance in feed_set:
                        bottleneck_feed[key] = \
                            bottleneck_feed.get(key, 0) + 1
                if instance == decision["selected_instance"] \
                        and action == decision["selected_action"] \
                        and cand["applicable"]:
                    key = _bottleneck_component(cand["segments"])
                    bottleneck_selected[key] = \
                        bottleneck_selected.get(key, 0) + 1
        declared = decision["applicable_actions"]
        if isinstance(declared, list):
            observed = sorted({
                action for per_instance in decision["candidates"].values()
                for action, cand in per_instance.items() if cand["applicable"]})
            if sorted(declared) != observed:
                applicable_mismatch += 1

    # -- δ 敏感性重定价（A3'：C10 终冻=0 后撤销 δ=0 空转对拍臂） --
    sigma_hat_ns = sigma["sigma_hat_ns"]
    # v9（2026-10-20）轮转感知基准：rotated 行的 δ 分档重放走轮转镜像
    # （配额前集口径——镜像旧路径 challenger_min/alt_star 的 caliber），
    # 比较基准/位移锚点一律 actual_ref/selected_hop_ref——纯轮转位移
    # 不计翻转/不污染位移分位，真成本驱动翻转照计；键缺席行
    # actual_ref = actual（与旧口径逐位一致）。
    flip_tiers: dict[str, dict] = {}
    if sigma_hat_ns is not None:
        for index, multiplier in enumerate(multipliers):
            delta = int(sigma_hat_ns * multiplier)
            flips = 0
            displacements: list[int] = []
            for entry in entries:
                decision = entry["decision"]
                derived = entry["derived"]
                if derived["rotation_active"]:
                    pick_action, pick_instance = _rotation_aware_selection(
                        decision["candidates"], delta,
                        decision["tie_cursor"],
                        include_quota_deferred=True)
                else:
                    pick_action, pick_instance = _selection_under_delta(
                        delta, derived["challenger_min"],
                        derived["alt_star"])
                if pick_action == derived["actual_ref_action"] \
                        and pick_instance == derived["actual_ref_instance"]:
                    continue
                flips += 1
                if pick_instance is not None \
                        and derived["selected_hop_ref"] is not None:
                    hop = None
                    cand = (decision["candidates"]
                            .get(pick_instance, {}).get(pick_action))
                    if cand and cand["hops"] is not None:
                        hop = cand["hops"]
                    if hop is not None:
                        displacements.append(
                            hop - derived["selected_hop_ref"])
            flip_tiers[str(multiplier)] = {
                "delta_ns": delta,
                "flips_vs_actual": flips,
                "displacement_hop": _quantiles_or_na(displacements),
            }
    # δ=0 主臂重放一致性（谓词重放 vs 在线选择；tie 序差异 → 计数披露）。
    # N6（2026-09-23 复核审计6）：重放集改**实际可行集**（applicable-only
    # 的 challenger_min_actual/alt_star_actual——与在线选择同约束，纯
    # 谓词一致性）；配额前集重放与在线选择的差单列 quota_counterfactual_
    # flips（配额作用量 run 级，计划文档 §5.4"两域之差"落地）——修前被拒
    # 挑战者更便宜时配额效应被计入 replay_mismatch。
    replay_mismatch = 0
    quota_counterfactual_flips = 0
    # v9（2026-10-20）轮转披露计数（sensitivity_scan.flip_repricing.
    # rotation_aware_replay 披露；键缺席 run 四计数恒 0）。
    rows_with_tie_keys = 0
    rotated_rows = 0
    rotation_consistency_mismatches = 0
    cursor_anomalies = 0
    for entry in entries:
        decision = entry["decision"]
        derived = entry["derived"]
        if derived["tie_keys_present"]:
            rows_with_tie_keys += 1
        if derived["rotation_active"]:
            rotated_rows += 1
            if derived["cursor_anomaly"]:
                cursor_anomalies += 1
            if derived["rotation_consistency_mismatch"]:
                rotation_consistency_mismatches += 1
            # 轮转路径：δ=0 实际可行集重放与轮转公式期望是同一计算
            #（actual_ref 本身）——谓词侧恒自洽，replay_mismatch 的唯一
            # 来源 = 一致性校验（在线 actual != expected，真发散）。
            if derived["rotation_consistency_mismatch"]:
                replay_mismatch += 1
            cf_action, cf_instance = _rotation_aware_selection(
                decision["candidates"], 0, decision["tie_cursor"],
                include_quota_deferred=True)
        else:
            pick_action, pick_instance = _selection_under_delta(
                0, derived["challenger_min_actual"],
                derived["alt_star_actual"])
            if pick_action != derived["actual_ref_action"] \
                    or pick_instance != derived["actual_ref_instance"]:
                replay_mismatch += 1
            cf_action, cf_instance = _selection_under_delta(
                0, derived["challenger_min"], derived["alt_star"])
        if cf_action != derived["actual_ref_action"] \
                or cf_instance != derived["actual_ref_instance"]:
            quota_counterfactual_flips += 1

    # -- 边界对拍 --
    rho = topology["rho"]
    observed_boundaries = [e["derived"]["inner_boundary_hop"]
                           for e in entries
                           if e["derived"]["inner_boundary_hop"] is not None]
    deviations = [observed - rho for observed in observed_boundaries
                  if rho is not None]

    window = [scan.decisions[0]["tick"] if scan.decisions else None,
              scan.last_tick]

    return {
        "command": "domain_metrics",
        "repo_variant": repo_variant,
        "joint_admission_rows": scan.joint_rows,
        "calibers": {
            "d_feed": {
                "definition": "D_feed(q,p,s) = 决策日志中 copy 挑战者候选"
                              " 配额前结构可行的实例集合（M2/N6：applicable"
                              " ∨ quota_ 前缀拒且配额前成本在——配额拒候选"
                              " 的供给条件已判过、经济域可重建）；本地参考与"
                              "性能容忍度逐成员标注（instances CSV feed_"
                              "margin_* 与 challenger_cost_pre_quota_ns）",
                "size": _size_stats(feed_sizes),
                "hop_quantiles": _quantiles_or_na(feed_hops),
                "direction_distribution": _direction_distribution(
                    feed_hop_pairs),
                "note": "距离只是描述量，不要求域为圆形或连通区域（§5.3）"
                        if anchors else
                       "实例锚点坐标 NA（trace_config/hardware 缺）——"
                       "方向分布不可得",
            },
            "d_econ": {
                "definition": "D_econ(q,s) = {e: C_copy(e) ≤ C_alt*+δ}；"
                              "C_alt* = min C(e,a) 遍历全部实例的适用替"
                              "代动作（stay；F13）",
                "delta_main_ns": delta_adm,
                "size_delta0": _size_stats(econ_sizes),
                "hop_quantiles_delta0": _quantiles_or_na(econ_hops),
                "direction_distribution_delta0": _direction_distribution(
                    econ_hop_pairs),
                "same_position_comparison": {
                    "caliber": "同实例 copy 挑战者 vs 同实例最优非 "
                               "挑战者（stay）——同位置动作比较"
                               "另报（§5.2），与全实例 C_alt* 口径不混",
                    "total_wins": sum(
                        e["derived"]["same_position_wins"] for e in entries),
                },
                "sensitivity_tiers_membership": {
                    str(multiplier): {
                        "delta_ns": (int(sigma_hat_ns * multiplier)
                                     if sigma_hat_ns is not None else NA),
                        "total_members": sum(
                            len(e["derived"]["d_econ_tier_sets"][index])
                            for e in entries),
                    }
                    for index, multiplier in enumerate(multipliers)
                },
            },
            "actual_selection": {
                "definition": "在线仅记录最终 (e,a)（selected_action/"
                              "joint_instance_index）；单次实际选择不代表"
                              "该请求的完整域（§5.2）",
                "two_action_counts": action_counts,
                "applicable_actions_mismatch_rows": applicable_mismatch,
                "aggregation_window_ns": window,
                "note": "不同请求的集合统计聚合窗口 = 本决策日志首准入 "
                        "tick 至末事件 tick（如上），不解释为瞬时能力",
            },
            "measured": {
                "status": NA,
                "note": "独立同状态执行的 measured 经济/性能集合不在本产"
                        "物（C4b 同状态受控测量三表）——不与预测混"
                        "合；预测/实测差仅作诊断分列（见 "
                        "derived.prediction_vs_measured）",
            },
        },
        "derived": {
            "bottleneck_resource": {
                "caliber": "breakdown 五段（target_wait/history_prep/"
                           "eviction_wait/compute/merge）argmax；并列以 "
                           "+ 联结",
                "d_feed_challenger_members": bottleneck_feed,
                "selected_candidates": bottleneck_selected,
            },
            "constraint_unsatisfied_reasons": reasons,
            "applicable_candidate_counts": applicable_counts,
            "prediction_vs_measured": sigma.get("observational"),
            "dual_domain_chart": {
                "data": "instances CSV：in_d_econ_*（D_econ 等值线侧）vs "
                        "quota_admissible/port_*（配额准入域侧）——同图导"
                        "出；两域之差即配额作用量",
                "quota_domain_status": (
                    "derived_from_candidates" if any(
                        _quota_deferred(cand)
                        for decision in scan.decisions
                        for per_instance in (
                            decision["candidates"].values())
                        for cand in per_instance.values())
                    else "structural_only"),
                "quota_domain_note": "配额域数据源 = 候选 quota_ 前缀理由"
                                     "派生（M2，2026-09-23：实例 CSV "
                                     "quota_admissible 已派生 0/1——status "
                                     "与列一致；quota-off run 无 quota_ 理"
                                     "由 ⇒ structural_only）。"
                                     "port_* 逐字段 = port_snapshot 实测"
                                     "值（C11 已接通：quota-on run 携带"
                                     "实测、quota-off 态如实 NA——原“WP3 "
                                     "前 C5 冻结占位”注记系 N13 订正遗漏"
                                     "落进摘要产物，O8④ 补正）",
            },
            "contours": {
                "inner": "C_copy = C_stay_ref（列 c_challenger_minus_"
                         "stay_ref_ns 的零穿越）",
                "outer": "C_copy = C_alt*（列 c_challenger_minus_alt_star_ns"
                         " 的零穿越；替代族 = stay，含池恢复形态）",
            },
            "boundary_cross_check": {
                "rho_anchor": fmt_ratio(rho) if rho is not None else NA,
                "rho_source": topology["rho_source"],
                "observed_challenger_local_boundary_hop":
                    _quantiles_or_na(observed_boundaries),
                "boundary_deviation_hop":
                    _quantiles_or_na([int(round(d)) for d in deviations])
                    if deviations else {"n": 0},
                "caliber": "观测分界 = max{hops(e): C_copy(e) ≤ C_stay_"
                           "ref}（内等值线半径的 hop 投影）",
                "disclaimers": [
                    "d≤ρ 仅历史扫描项平价、不保证完整请求平价（§1.2）",
                    "对拍是诊断不是验收：不要求逐点重合，偏差本身是模型"
                    "诊断结果（§5.3）",
                    "无 n* = d'/(d-ρ) 式（v2.2 已删除）",
                ],
            },
        },
        "delta_epsilon_registry": {
            "delta_adm_main_ns": delta_adm,
            "delta_unit": "ns",
            "delta_source": "C10 δ_adm 终冻 = 维持 0（三选一路径三；"
                            "被否决来源 I/有效带宽=重复计费已登记）",
            "delta_semantics": "D_econ 集合口径 ≤ C_alt*+δ；决策路径口径 "
                               "challenger C_copy+δ ≤ C_alt*——两口径分列"
                               "不混",
            "sensitivity_scan": {
                "multipliers": multipliers,
                "sigma_hat_ns": sigma_hat_ns,
                "sigma_hat_source": sigma["source"],
                "sigma_hat_identity": sigma["identity_note"],
                "merge_leg_cold_start_anchor":
                    sigma.get("merge_leg_anchor"),
                "flip_repricing": {
                    "predicate": "C_copy(e)+δ ≤ C_alt*（C10 challenger_"
                                 "flips 同构；等号成立按在线 argmin 全键"
                                 "序 (cost+δ, instance, ACTION_ORDER 秩) "
                                 "裁决——O8① 起 tie 序镜像在线，残余 "
                                 "replay 差异非平局 artifact）",
                    "per_tier": flip_tiers,
                    "replay_mismatch_at_delta0": replay_mismatch,
                    "quota_counterfactual_flips": quota_counterfactual_flips,
                    "rotation_aware_replay": {
                        "caliber": "v9（2026-10-20）起 δ=0/δ 分档的比较"
                                   "基准 = 轮转感知 actual_ref：三键"
                                   "（joint_tie_cursor/joint_tie_"
                                   "instances/joint_tie_rotated）在场且"
                                   " rotated 的 P 行，actual_ref = 由本"
                                   "行候选表现算的轮转公式期望选择（语义"
                                   "单一源 joint_scheduler.joint_tie_"
                                   "rotation_pick 的本地镜像 _joint_tie_"
                                   "rotation_pick_mirror + 对账钉测；"
                                   "joint_tie_instances 解析值仅披露对照、"
                                   "裁决以现算值为准）；纯轮转位移不计"
                                   "翻转/不污染位移分位，actual != "
                                   "expected 计 replay_mismatch（真发"
                                   "散）。键缺席（旧 run）⇒ actual_ref = "
                                   "actual，输出与旧口径逐位一致",
                        "rows_with_tie_keys": rows_with_tie_keys,
                        "rotated_rows": rotated_rows,
                        "rotation_consistency_mismatches":
                            rotation_consistency_mismatches,
                        "cursor_anomalies": cursor_anomalies,
                    },
                    "replay_calibers_note": (
                        "N6 双口径：replay_mismatch 重放集=实际可行集"
                        "（applicable-only，与在线选择同约束——纯谓词一"
                        "致性）；quota_counterfactual_flips=配额前集重放"
                        "与在线选择的差（配额作用量 run 级，计划 两"
                        "域之差）"),
                    "interpretation": "翻转计数=0 的读法：σ̂（观测口径，含"
                                      "排队/状态演化共同项、上偏）量级超"
                                      "过候选间成本差 → 该 δ 下挑战者无"
                                      "翻转是敏感性结果而非空转对照；差分"
                                      "噪声的真实量级归 C4b 受控测量",
                },
                "note": "A3' 强制修订：原 δ_adm=0 空转对拍臂已撤销；三档"
                        "仅离线诊断，不进任何在线决策路径，不据其修改 "
                        "δ_adm 终冻值，不据结果扩大平局带",
            },
            "epsilon": {
                "value": "见 manifest epsilon（0.01，ratio）",
                "note": "ε = SLO violation 口径（QoServe 容量门槛族），"
                        "本工具计算不使用——F13『ε/δ 含义、单位和来源实"
                        "验前固定』登记口径",
            },
        },
        "home_migration": None,  # 由调用方填充（含 kv_delta 轮询结果）
        "narrative_discipline": "饱和下域由配额关闭而非 argmin 涌现——"
                                "'涌现域'表述限定配额界内；域是解释与可"
                                "选准入分析对象，不裁剪任何动作/实例候选"
                                "（全候选 argmin 照常，本工具只读）",
        "warnings": warnings,
        "products": {
            "requests_csv": REQUESTS_CSV_NAME,
            "instances_csv": INSTANCES_CSV_NAME,
            "summary_json": SUMMARY_JSON_NAME,
        },
    }


# ---------------------------------------------------------------------------
# prepare / consume / emit（driver Sink 拆分面 + 独立 CLI 同一逻辑）
# ---------------------------------------------------------------------------

def _load_tokens_map(run_dir: Path, request_manifest_arg: Optional[str],
                     warnings: list[str]) -> dict[str, dict]:
    """逐请求 KV 字节组件（H/I）的 token 来源。

    物化 plan manifest（run_dir/manifest.json，含 history_tokens_before/
    final_context_tokens——与 hbm_watermark 的 token 链同源）优先；
    metrics_manifest（load_request_manifest）具备同名字段时合并补齐。
    任一缺失 → 组件 NA（警告），不编造。
    """
    tokens_map: dict[str, dict] = {}
    plan_manifest = run_dir / "manifest.json"
    sources: list[tuple[str, object]] = []
    if plan_manifest.is_file():
        sources.append(("plan_manifest", plan_manifest))
    try:
        sources.append(("metrics_manifest",
                        load_request_manifest(run_dir,
                                              request_manifest_arg)))
    except SloToolError:
        pass
    for _tag, source in sources:
        try:
            if isinstance(source, Path):
                data = json.loads(source.read_text(encoding="utf-8"))
            else:
                data = source
            rows = data.get("requests") if isinstance(data, dict) else None
            if not isinstance(rows, list):
                continue
            for row in rows:
                if not isinstance(row, dict):
                    continue
                request_id = row.get("request_id")
                if not request_id:
                    continue
                slot = tokens_map.setdefault(str(request_id), {})
                for key in ("history_tokens_before", "final_context_tokens",
                            "decode_length"):
                    if row.get(key) is not None:
                        slot[key] = row.get(key)
        except (OSError, json.JSONDecodeError, SloToolError):
            continue
    if not any("history_tokens_before" in slot
               for slot in tokens_map.values()):
        warnings.append(
            "[domain-metrics] 警告：history_tokens_before 不可得（plan "
            "manifest 与 metrics manifest 均无）——逐请求 KV 字节"
            "组件 H/I = NA")
    return tokens_map


def domain_prepare() -> DomainScan:
    """A4 driver 用：空扫描态（参数/拓扑解析全部推迟到 emit——非 joint
    run 完全静默；joint run 的 fail-closed 与警告在 emit 时点发生）。"""
    return DomainScan()


def domain_consume(scan: DomainScan, record: dict) -> None:
    scan.consume(record)


def domain_emit(args: argparse.Namespace, repo_variant: str,
                scan: DomainScan) -> int:
    if scan.joint_rows == 0:
        # 非 joint run（其他仓 / 无 joint_admission 行）：完全静默跳过——
        # 四仓共链字节对拍契约（test_driver_parity）要求零扰动。
        if not getattr(args, "quiet_when_empty", False):
            print("[domain-metrics] 无 joint_admission 决策行——局部统一"
                  "内存域三口径不适用，跳过（by design）", file=sys.stderr)
        return 0

    warnings: list[str] = []
    manifest = load_slo_manifest(args.manifest or default_manifest_path())
    delta_adm = require_param_int(manifest, "domain_delta_adm_ns")
    if delta_adm < 0:
        fail("domain_delta_adm_ns 必须为非负整数（ns）")
    multipliers_raw = require_param(
        manifest, "domain_delta_sensitivity_multipliers")
    if not isinstance(multipliers_raw, list) or not multipliers_raw \
            or not all(isinstance(m, (int, float)) and not isinstance(
                m, bool) and m > 0 for m in multipliers_raw):
        fail("domain_delta_sensitivity_multipliers 必须为正数列表"
             f"（实得 {multipliers_raw!r}）")
    multipliers = [float(m) for m in multipliers_raw]

    topology = resolve_topology(args.run_dir, args.trace_config,
                                args.hardware_config, warnings)
    tokens_map: dict[str, dict] = {}
    if topology["coef_bytes_per_token_per_npu"] is not None:
        try:
            tokens_map = _load_tokens_map(args.run_dir,
                                          args.request_manifest, warnings)
        except Exception:  # noqa: BLE001 —— 组件缺失 → NA 降级不 fail
            warnings.append(
                "[domain-metrics] 警告：per-request manifest 不可得——"
                "逐请求 KV 字节组件 H/I = NA")

    # W5（§7.4/§9.3）：结算策略从 run 级 manifest 读（prediction 节；
    # 缺失/不可读 = 旧 run 兼容缺省 consolidate，告警登记）。
    settlement_policy = _resolve_settlement_policy(args.run_dir, warnings)
    require_explicit_settle = _read_prediction_manifest(args.run_dir) is not None
    # cfg1（§10.3）：历史模式（future_demand_mode）同面读取——分组
    # 报告的 run 级维度输入。
    future_demand_mode = _resolve_future_demand_mode(
        args.run_dir, warnings)
    entries = derive_entries(scan, topology, None, multipliers, warnings,
                             settlement_policy=settlement_policy,
                             require_explicit_settle=require_explicit_settle)
    sigma = estimate_sigma_hat(entries, warnings)
    entries = derive_entries(scan, topology, sigma["sigma_hat_ns"],
                             multipliers, warnings,
                             settlement_policy=settlement_policy,
                             require_explicit_settle=require_explicit_settle)

    summary = build_summary(scan, entries, topology, repo_variant,
                            delta_adm, multipliers, sigma, warnings)
    summary["home_migration"] = home_migration_block(scan, entries,
                                                    args.run_dir)
    # cfg1（2026-09-30，§10.3 误差成熟与归因）：P/D 完成预测分组误差
    # 报告（距离/动作/瓶颈/负载阶段/承诺代理/历史模式分组 + 七分归因
    # ——此前离线侧仅有 estimate_sigma_hat 单一聚合）。
    summary["derived"]["completion_error_report"] = completion_error_report(
        scan, entries, settlement_policy, future_demand_mode, warnings,
        require_explicit_settle=require_explicit_settle)

    requests_rows = [
        _requests_csv_row(entry, topology,
                          tokens_map.get(entry["decision"]["request_id"]))
        for entry in entries]
    stream, close = open_output(args.output, REQUESTS_CSV_NAME, args.run_dir)
    try:
        write_csv(stream, REQUESTS_COLUMNS, requests_rows)
    finally:
        if close:
            stream.close()
    inst_stream, inst_close = open_output(
        args.instances_csv, INSTANCES_CSV_NAME, args.run_dir)
    try:
        row_iterator = (
            row
            for entry in entries
            for row in _instances_csv_rows(entry, topology,
                                           {"multipliers": multipliers,
                                            "sigma_hat_ns": sigma[
                                                "sigma_hat_ns"]}))
        write_csv(inst_stream, INSTANCES_COLUMNS, row_iterator)
    finally:
        if inst_close:
            inst_stream.close()
    for warning in warnings:
        print(warning, file=sys.stderr)
    emit_json(sys.stderr, {
        "command": "domain_metrics",
        "joint_admission_rows": scan.joint_rows,
        "two_action_counts":
            summary["calibers"]["actual_selection"]["two_action_counts"],
        "d_feed_sizes": summary["calibers"]["d_feed"]["size"],
        "d_econ_sizes_delta0": summary["calibers"]["d_econ"]["size_delta0"],
        "rho_anchor": summary["derived"]["boundary_cross_check"]["rho_anchor"],
        "sigma_hat_ns": sigma["sigma_hat_ns"],
        "sigma_hat_source": sigma["source"],
        # M2：打印块 status 与 derived.dual_domain_chart 同源派生
        #（第三处固定 NA 残留订正——与实例 CSV 0/1 一致）。
        "quota_domain_status": summary["derived"]["dual_domain_chart"][
            "quota_domain_status"],
    })
    if args.json:
        jstream, jclose = open_output(args.json, SUMMARY_JSON_NAME,
                                      args.run_dir)
        try:
            emit_json(jstream, summary)
        finally:
            if jclose:
                jstream.close()
    else:
        # 摘要 JSON 是主产物（口径登记/叙事纪律/δ 登记承载面）——缺省
        # 恒写 run_dir/slo_domain_summary.json（'-' 走 open_output 的
        # stdout 分支；与 hopbytes 的可选 --json 不同属有意为之）。
        jstream, jclose = open_output("", SUMMARY_JSON_NAME, args.run_dir)
        try:
            emit_json(jstream, summary)
        finally:
            if jclose:
                jstream.close()
    return 0


def cmd_domain(args: argparse.Namespace) -> int:
    from slo_common import detect_repo_variant
    repo_variant = detect_repo_variant(args.run_dir, args.repo_variant)
    scan = domain_prepare()
    for record in iter_jsonl(args.run_dir / DECISION_LOG_RELPATH):
        domain_consume(scan, record)
    return domain_emit(args, repo_variant, scan)


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="domain_metrics.py",
        description="局部统一内存域三口径离线度量（D_feed/D_econ/实际选择"
                    "分列；双域同图数据/内外等值线/边界对拍/δ 敏感性重定"
                    "价/home 迁移轨迹；设计文档 §5.2-§5.3 + C5 schema）")
    parser.add_argument("run_dir", type=Path,
                        help="运行目录（含 results/online_decision_log."
                             "jsonl）")
    parser.add_argument("-o", "--output", default="",
                        help="逐请求三口径 CSV（'-'=stdout；缺省写 run_dir/"
                             "slo_domain_requests.csv）")
    parser.add_argument("--instances-csv", default="",
                        help="逐 (请求,实例) 双域/等值线 CSV（缺省写 "
                             "run_dir/slo_domain_instances.csv）")
    parser.add_argument("--json", default="",
                        help="汇总 JSON 输出路径（缺省写 run_dir/"
                             "slo_domain_summary.json）")
    parser.add_argument("--manifest", default=None,
                        help="slo_params_manifest.json 显式路径")
    parser.add_argument("--request-manifest", default=None,
                        help="per-request manifest 显式路径")
    parser.add_argument("--trace-config", default=None,
                        help="trace_config.csv 显式路径（ρ/锚点/字节系数）")
    parser.add_argument("--hardware-config", default=None,
                        help="hardware json 显式路径（ρ/网格）")
    parser.add_argument("--repo-variant", default=None,
                        help="显式指定 repo_variant（默认读 cpp.log init 行）")
    args = parser.parse_args()
    args.quiet_when_empty = False
    return int(cmd_domain(args) or 0)


if __name__ == "__main__":
    sys.exit(run_main(main))
