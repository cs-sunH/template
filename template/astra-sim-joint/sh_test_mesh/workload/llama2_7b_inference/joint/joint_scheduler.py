"""joint_scheduler.py -- J：发射点选择（joint / load-first / affinity-first
/ face_static）。

设计依据：《三机制联合策略_template仓库设计方案》§1（J 语义）、§7.1
（J-off 顺序参照）与实验总纲 §13.3（内部策略替代）；C17 卡（face_static
静态距离对照臂）。

四模式（同一候选枚举与计费，只换选择规则）：

* ``joint``（J-on）：同一队列发射决策点比较全部 instance ×
  stay/copy 的 ``cost``（到 merge_done 终点的
  预测完成时间——F11 口径：预测终点恒 merge_done，service_done 作
  服务指标另行分报），取最小。候选评估只读——不为每个候选实际驱逐。
* ``load-first``（J-off，八组合固定参照）：先在**全部 instance**（无
  容量/边缘/距离掩码——HBM 容量只影响动作计费中的驱逐等待，不做准
  入过滤）按在线任务负载（ns 服务台账）选 instance，再在该位置对二
  动作取最小代价。
* ``affinity-first``（J 边际比较第二参照，不进八组合）：有可复用历史
  时优先选择其逻辑 home；无历史或无唯一 home 时回退 load-first。选定
  实例后的动作计费、驱逐和生命周期保持共同规则（亲和是选择偏好，
  不是容量/距离掩码）。
* ``face_static``（C17 静态距离对照臂：policy variant，不进八组合、
  非 J 主臂）：实例按 ``hop ≤ floor(rho)`` 静态距离球掩码（rho =
  B_D2D/B_HBM 逐配置硬件派生，与 ``joint.link_quota.derive_rho_eff``
  同源；base 配置 4050/1640 ≈ 2.47 → 半径 2），球内逐实例
  ``_best_at_instance`` 按 ``(cost_ns, order_key)`` 择优。锚 = session
  驻留实例（无驻留回退逻辑 home——与 cost_model 路由同锚）；ρ<1 退化
  仅锚实例（半径 0，无 max(1,·) 保底——如实记录）；turn-0 无历史无锚
  时球退化为空约束（全部实例照常参与，如实记录，不 fail-closed）。
  **身份纪律（设计文档 §5.3/§1.3）**：本臂是本文内部的静态距离变体，
  命名与 manifest 必须明确这一身份，**不得冒称完整 FACE**；在完成
  路由、执行、共享和端点假设对齐推导前，不宣称静态距离球是本文动态
  供给模型的零负载特例（身份声明同步落 joint_config manifest 的
  ``face_static_identity`` 键）。该臂强制 quota-off（F7；覆盖与注记在
  joint_config 构造点），以量静态掩码自身的两侧误差。**joint 主臂
  永不掩码**——本臂是本模块唯一的实例掩码分支。

平局与 tie 序：固定 ``(instance_index, ACTION_ORDER)`` 确定序（事前
定义，不按事后输赢调整）。
动作掩码 ``JOINT_ALLOWED_ACTIONS``（M2，2026-09-26 消融矩阵前置）是
**候选集上限不是强制**：命中掩码的动作在进入本模块前已由 JCM
``applicable_actions`` 尾部剔除（掩码命中且原本可用 ⇒ "action
excluded by JOINT_ALLOWED_ACTIONS mask"），本模块枚举循环与
candidates 形状不变、四模式共用同一上限——掩码剔除致零适用时维持
fail-closed（下方各 raise，非容量拒绝）。

零适用行为按模式分述（D2，2026-10-15 recompute 退役后——原
"recompute 物理面恒有定义"前提不复存在；stay/copy 双不适用唯一
编码点 = W9 distributed 分裂布局锚实例，当前无自然生产者的防御
姿态编码）：joint 主臂 = argmin 天然忽略不适用候选（全局零适用
raise）；face_static = ``_best_at_instance`` 返回 None 即跳过该实例
（既有分支，语义不变；球内全零适用 raise）；sequential
（load-first/affinity-first）= 固定实例零适用即 raise（无跳过分支，
保留原 fail-closed 语义）。全局零适用一律 ``JointSchedulerError``
协议缺口归因，与容量类拒绝（``KVPhysicalInfeasibleError``）分立。

本模块只做选择与记录；发射/驱逐提交/合并事务由在线调度器与
KVCacheManager 执行（§7 模块边界：scheduler 管选择与事务提交，
cost_model 管纯预测）。

D18（2026-09-26，两阶段决策）：本模块现提供两个选择入口——
``select_instance_and_action``（P 发射点，式 1a：每候选估
P + 默认 D 延续 π0 的整轮，**不做全 D 搜索**）与
``select_decode_target_and_action``（P→D 边界，式 1b/1c：真实快照
上恰好一次全遍历，原地继续恒为候选且平局优先，只有严格更低成本才
切换；M2 掩码剔除延续动作时除外——掩码是候选集上限不是强制，红线
掩码感知裁定见该入口 docstring）。每请求恰两次全候选遍历（P/D），
遍历结果即最终配置精度面——算出来是多少就是多少（2026-10-16 E0
精确复核仲裁退役后无第二次重询），D 提交后无第三次位置决策
（v2.0 §2/§16/§17）。

D01（2026-09-26，阶段扩展收口）：上述两入口即估价器"阶段参数与
当前布局参数"的落点（P：布局 = 到达时 SessionKVView；D：布局 =
PDBoundaryView→BoundaryKVLayout）；初始 argmin 与初始平局序
``(cost, (instance_index, ACTION_ORDER))`` **保留不变**（C07 现状
即无组合搜索，不引入新优化器），D 边界单独用式(1c) 平局键；两阶段
SelectionRecord 的 ``diagnostics["decision_stage"]``（"P"/"D"）与
候选 notes 的 ``decision_stage=`` 键自描述决策阶段——两个时刻的
全部实例×动作审计项逐候选可辨阶段（v2.0 §22 验收）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional, Sequence

from joint.joint_config import (
    SCHEDULER_MODE_AFFINITY,
    SCHEDULER_MODE_FACE_STATIC,
    SCHEDULER_MODE_JOINT,
    SCHEDULER_MODE_LOAD_FIRST,
)
from joint.joint_cost_model import (
    ACTION_ORDER,
    ACTION_STAY,
    ActionCandidate,
    JointCostModel,
    PDBoundaryView,
    SessionKVView,
)
from joint.link_quota import derive_rho_eff

_SCHEDULER_MODES = (
    SCHEDULER_MODE_JOINT, SCHEDULER_MODE_LOAD_FIRST,
    SCHEDULER_MODE_AFFINITY, SCHEDULER_MODE_FACE_STATIC,
)


class JointSchedulerError(ValueError):
    """fail-closed：非法模式或状态不一致。"""


def _candidate_id(candidate: ActionCandidate) -> str:
    """候选稳定标识（L3 导出器 acceptance_projection._candidate_rows
    消费的模块级工具）。"""
    return f"{candidate.instance_index}:{candidate.action}"


def _release_projection_contexts(candidates) -> None:
    """决策收尾释放逐候选 native 投影上下文（D1，2026-10-16 E0
    复核仲裁退役后从仲裁 finally 迁至两选择入口）。

    共享上下文经 ``WorldContext`` 缓存持有，所有候选的
    ``projection_context`` 指向同一对象，close 幂等；不可适用候选
    提前 return 不挂 graph，字段缺省 None——仅对非 None 者调用。
    """
    for item in candidates:
        context = getattr(item, "projection_context", None)
        if context is not None:
            context.close()


@dataclass
class SelectionRecord:
    """一次发射决策的完整记录（决策日志逐项落盘，§7/§13.2）。"""

    mode: str
    chosen: ActionCandidate
    candidates: tuple[ActionCandidate, ...]
    instance_rule_note: str
    tie_break_note: str = ""
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "chosen": {
                "instance_index": self.chosen.instance_index,
                "action": self.chosen.action,
                "cost_ns": self.chosen.cost_ns,
            },
            "instance_rule_note": self.instance_rule_note,
            "tie_break_note": self.tie_break_note,
            "candidates": [
                {
                    "instance_index": candidate.instance_index,
                    "action": candidate.action,
                    "applicable": candidate.applicable,
                    "cost_ns": candidate.cost_ns,
                    "inapplicable_reason": candidate.inapplicable_reason,
                }
                for candidate in self.candidates
            ],
            "diagnostics": dict(self.diagnostics),
        }


def _best_at_instance(
    candidates: Sequence[ActionCandidate],
    instance_index: int,
) -> tuple[Optional[ActionCandidate], str]:
    """单实例内二动作择优（确定 tie 序：ACTION_ORDER）。"""
    applicable = [
        candidate for candidate in candidates
        if candidate.instance_index == instance_index
        and candidate.applicable and candidate.cost_ns is not None
    ]
    if not applicable:
        return None, "no applicable action at instance"
    best = min(
        applicable,
        key=lambda candidate: (
            candidate.cost_ns, candidate.order_key()))
    ties = [
        candidate for candidate in applicable
        if candidate.cost_ns == best.cost_ns
        and candidate.order_key() != best.order_key()
    ]
    note = (
        f"tie among {len(ties) + 1} at cost {best.cost_ns}ns"
        if ties else "")
    return best, note


def _static_ball_radius(cost_model: JointCostModel) -> tuple[float, int]:
    """静态距离球半径 = floor(rho_eff)（C17；逐配置硬件派生）。

    rho_eff = B_D2D/B_HBM，与 ``joint.link_quota.derive_rho_eff`` 同源
    （单一裁决点，不另立公式）。半径**无** ``max(1, .)`` 保底——结构性
    防死锁下限是配额 ``Q_init`` 的专有语义（F2），静态球在 rho < 1 配置
    下如实退化为半径 0（仅锚实例），由调用方如实记录。
    """
    rho_eff = derive_rho_eff(
        cost_model.rates.noc_link_bytes_per_ns,
        cost_model.rates.local_hbm_bytes_per_ns)
    return rho_eff, int(math.floor(rho_eff))


def _static_ball_hops(cost_model: JointCostModel, source: int,
                      target: int) -> int:
    """静态距离球的逐实例跳数（锚语义与 cost_model 路由一致：source ==
    target 恒 0 跳、不调 route_fn；跨实例由注入的 route_fn 给
    ``(path, hops)``）。"""
    if source == target:
        return 0
    _path, hops = cost_model.route_fn(source, target)
    return int(hops)


def select_instance_and_action(
    *,
    mode: str,
    cost_model: JointCostModel,
    session: SessionKVView,
    request,
    prep_frontier_by_instance=None,
) -> SelectionRecord:
    """发射点选择：枚举全候选 -> 按模式选择 -> 返回完整记录。

    ``request`` 为 RequestView（无 oracle 输入；结构上不接受
    decode_length/final_context_tokens）。枚举阶段对每个 (instance,
    action) 调用 ``estimate_P_with_continuation``（只读，不修改任何账本或事件队列）。

    D01（2026-09-26）：全实例×二动作枚举与初始平局序
    ``(cost, (instance_index, ACTION_ORDER))`` 保留不变（joint 主臂
    argmin，C07 现状）；记录 ``diagnostics["decision_stage"]="P"``。

    D03 修复轮二（2026-09-27）：``prep_frontier_by_instance`` = 逐目标
    实例的准备腿最早发射时刻（决策相对 ns；缺省 None = 全部空闲 0，
    行为与旧口径逐字节一致）——调度器按构图事实（准入链接续 rank
    frontier）注入，见 sh30 ``_prep_frontier_by_instance``。
    """
    if mode not in _SCHEDULER_MODES:
        raise JointSchedulerError(f"unknown scheduler mode: {mode!r}")
    instance_indices = sorted(cost_model.loads.keys())
    if not instance_indices:
        raise JointSchedulerError("no instance load views available")

    candidates: list[ActionCandidate] = []
    for instance_index in instance_indices:
        for action in ACTION_ORDER:
            candidates.append(cost_model.estimate_P_with_continuation(
                session=session,
                request=request,
                instance_index=instance_index,
                action=action,
                prep_frontier_ns=int(
                    (prep_frontier_by_instance or {}).get(
                        instance_index, 0)),
            ))

# D1（2026-10-16 E0 复核仲裁退役）：候选 native 投影上下文的释
# 放义务迁至选择入口——枚举完成后 try/finally 包住全部选择逻辑
# （含 joint 早退、sequential/face_static 分支与 raise 路径）。
    try:

        if mode == SCHEDULER_MODE_JOINT:
            # D2 模式分述：joint 主臂 argmin 天然忽略不适用候选；全局零适用
            # （含掩码剔空允许面）= 协议缺口 fail-closed，与容量类拒绝分立。
            applicable = [
                candidate for candidate in candidates
                if candidate.applicable and candidate.cost_ns is not None
            ]
            if not applicable:
                raise JointSchedulerError(
                    "joint selection found no applicable candidate "
                    "(execution-protocol gap — incl. JOINT_ALLOWED_ACTIONS "
                    "mask excluding every remaining action, M2; not a "
                    "capacity rejection)")
            best = min(
                applicable,
                key=lambda candidate: (
                    candidate.cost_ns, candidate.order_key()))
            ties = [
                candidate for candidate in applicable
                if candidate.cost_ns == best.cost_ns
                and candidate.order_key() != best.order_key()
            ]
            return SelectionRecord(
                mode=mode,
                chosen=best,
                candidates=tuple(candidates),
                instance_rule_note="joint instance x action argmin",
                tie_break_note=(
                    f"tie among {len(ties) + 1} at cost {best.cost_ns}ns -> "
                    "(instance_index, action order)" if ties else ""),
                # D01（2026-09-26）：决策阶段自描述（P 初始时刻；与 D 边界
                # 记录的 "D" 对称）。
                diagnostics={"decision_stage": "P"},
            )

        # ---- 逐模式 instance 规则（顺序参照 + face_static 对照臂）：先定
        # instance，再在该位置选动作。joint 主臂已在上方全候选 argmin 提前
        # 返回——**永不掩码**；face_static 是本模块唯一实例掩码分支。 ----
        extra_diagnostics: dict = {}
        if mode == SCHEDULER_MODE_LOAD_FIRST:
            chosen_instance = min(
                instance_indices,
                key=lambda index: (
                    cost_model.loads[index].total_task_load_ns, index))
            note = "load-first: min task-load instance (all instances, no mask)"
        elif mode == SCHEDULER_MODE_FACE_STATIC:
            # 静态距离球对照臂（C17，policy variant——身份纪律见模块
            # docstring：本文内部的静态距离变体，非完整 FACE）。掩码是
            # 本臂唯一策略偏差：球内选择规则与 joint 同判据（逐实例
            # _best_at_instance 后按 (cost_ns, order_key) 择优），以隔离
            # "静态距离准入"这一被测策略量。
            anchor = (
                session.resident_instance
                if session.resident_instance is not None
                else session.home_instance)
            rho_eff, radius_hops = _static_ball_radius(cost_model)
            if anchor is None:
                # turn-0 无历史：无距离锚，球退化为空约束（全部实例照常
                # 参与）——如实记录，不 fail-closed（turn-0 的 stay 本地
                # 建立在任何实例恒适用，D2(a)；JOINT_ALLOWED_ACTIONS 掩码
                # 剔除 stay 时全实例零适用 ⇒ 下方防御 raise，M2）。
                masked = list(instance_indices)
                note = (
                    "face_static: no KV anchor (fresh session), static ball "
                    "vacuous, all instances admitted")
                extra_diagnostics["ball_vacuous"] = True
            else:
                masked = [
                    index for index in instance_indices
                    if _static_ball_hops(cost_model, anchor, index)
                    <= radius_hops]
                note = (
                    f"face_static: static ball hop <= {radius_hops} around "
                    f"anchor {anchor} (rho_eff={rho_eff:.6g}; "
                    f"{len(masked)}/{len(instance_indices)} instances in "
                    "ball)")
                if radius_hops == 0:
                    note += "; rho<1 degenerate: anchor instance only"
            if not masked:
                # 锚在视野内则 hop=0 恒入球，此处不可达；防御性 fail-closed
                # （锚为视野外陈旧索引 = 状态不一致，非容量拒绝）。
                raise JointSchedulerError(
                    f"face_static ball is empty (anchor={anchor!r}, "
                    f"radius={radius_hops}); anchor must be an in-view "
                    "instance at hop 0")
            # 掩码内 _best_at_instance：逐实例取最优动作，再按
            # (cost_ns, order_key) 择优实例（与 joint 同 tie 序）。
            chosen_instance = None
            chosen_key: Optional[tuple] = None
            for index in masked:
                best_at_index, _per_instance_note = _best_at_instance(
                    candidates, index)
                if best_at_index is None:
                    # 实例零适用即跳过（既有分支，语义不变，D2）：M2 掩码
                    # 剔除后本实例可能零适用动作，或（防御姿态编码，当前
                    # 无自然生产者——W9 分裂布局锚）物理双不适用——实例
                    # 不因动作面为空而出球。
                    continue
                key = (best_at_index.cost_ns, best_at_index.order_key())
                if chosen_key is None or key < chosen_key:
                    chosen_instance = index
                    chosen_key = key
            if chosen_instance is None:
                # 球内全部实例零适用（全局零适用防御 raise，D2）：M2 掩码把
                # 球内动作剔空、或球内仅含双不适用实例（防御姿态编码——
                # W9 分裂布局锚，当前无自然生产者）时可达——fail-closed
                # （协议缺口归因，候选集上限不是强制，非容量拒绝）。
                raise JointSchedulerError(
                    f"face_static found no applicable action inside the "
                    f"static ball of radius {radius_hops} around "
                    f"{anchor!r}")
            masked_set = set(masked)
            extra_diagnostics.update({
                "static_ball_anchor": anchor,
                "rho_eff": rho_eff,
                "radius_hops": radius_hops,
                "ball_instances": tuple(masked),
                # 掩出侧清单（量静态掩码两侧误差的被排除侧披露）。
                "masked_out_instances": tuple(
                    index for index in instance_indices
                    if index not in masked_set),
            })
        else:  # affinity-first
            if session.home_instance is not None and (
                    session.location in ("local_hbm", "partial_hbm_remote",
                                         "remote_memory")):
                chosen_instance = session.home_instance
                note = (
                    "affinity-first: session home "
                    f"{session.home_instance} (preference, not a mask)")
            else:
                chosen_instance = min(
                    instance_indices,
                    key=lambda index: (
                        cost_model.loads[index].total_task_load_ns, index))
                note = (
                    "affinity-first: no reusable home -> load-first fallback")

        best, action_note = _best_at_instance(candidates, chosen_instance)
        if best is None:
            # sequential 模式实例级 fail-closed（无跳过分支，保留原语义，
            # D2）：固定实例零适用即 raise——M2 掩码剔除本实例全部适用
            # 动作、或双不适用实例（防御姿态编码——W9 分裂布局锚，当前
            # 无自然生产者）被负载/亲和规则选中时可达。协议缺口归因，
            # 候选集上限不是强制，非容量拒绝。
            raise JointSchedulerError(
                f"sequential mode {mode!r} found no applicable action at "
                f"instance {chosen_instance}")
        return SelectionRecord(
            mode=mode,
            chosen=best,
            candidates=tuple(candidates),
            instance_rule_note=note,
            tie_break_note=action_note,
            diagnostics={"decision_stage": "P",
                         "sequential_instance": chosen_instance,
                         **extra_diagnostics},
        )
    finally:
        _release_projection_contexts(candidates)


def select_decode_target_and_action(
    *,
    mode: str,
    cost_model: JointCostModel,
    boundary: PDBoundaryView,
    decision_epoch: int = 0,
    inherited_obligation_ids: Sequence[str] = (),
) -> SelectionRecord:
    """P→D 边界二次选择（D18，v2.0 §2.3/§2.5/§16.2/§17）。

    D17（2026-09-27，v2.0 §18/D17）：``decision_epoch``（边界阶段
    epoch）与 ``inherited_obligation_ids``（继承义务 ID）为纯披露通
    道，逐候选透传进 breakdown schema——不计入计价、不改变任何候选
    成本与选择。

    **恰好一次全遍历**：全部 instance × 二动作调 ``estimate_D_remaining``
    （真实边界快照 S_B、共同 n̂_B、继承义务在负载视图内），随后
    ``boundary_order_key`` 一次 argmin——平局键 = ``(cost, 非继续标记,
    instance_index, action_order)``，**式(1c)：只有改变方案的完整预测
    剩余成本严格小于原地继续时才切换；成本相同优先继续**。原地继续
    x0 归并在对应的 (i_P, action) 记录上（``is_continuation=True``），
    不作第四动作（v2.0 §2.4）。

    与初始遍历（``select_instance_and_action``）的关系：每请求正常恰
    两次全候选遍历（§17"成功路径通常只进行 P、D 两次全候选遍历"）；
    D 提交后无第三次位置决策（验收红线）。本入口是协议级边界规则，
    不随调度模式变体（J-off 臂的差异只在初始选择规则——消融合同）；
    ``mode`` 仅进记录披露。

    已知终止的请求不进入本函数（调度器在 P_COMPUTE_DONE 直接结算，
    不做无意义的 D 选点——v2.0 §2.3/§16.4）。

    M2 掩码与 continuation 红线（掩码感知裁定：掩码是候选集上限不是
    强制）：原地继续 x0 归并在 (i_P, 默认延续动作 stay) 记录上——
    P 末工作副本恒为完整型布局（增量型布局族已随 remote-read 动作
    退役删除，默认延续恒 stay@i_P）。延续动作属允许面（无掩码或掩码
    放行）时，该记录缺席 = 边界视图与候选面不一致（结构性缺陷，
    fail-closed 不变）；延续动作被掩码剔除时，其缺席是掩码上限的合
    法结果——在允许面内正常选择，平局键退化为
    ``(cost, instance_index, action_order)``（无继续候选可优先），
    决策行 diagnostics 披露 ``continuation_masked_out``；掩码把允许
    面整体剔空时抛明确的 no-applicable-under-mask 错误（不许空序列
    ``min()``）——归因掩码，非容量拒绝。
    """
    if mode not in _SCHEDULER_MODES:
        raise JointSchedulerError(f"unknown scheduler mode: {mode!r}")
    instance_indices = sorted(cost_model.loads.keys())
    if not instance_indices:
        raise JointSchedulerError("no instance load views available")

    candidates: list[ActionCandidate] = []
    for instance_index in instance_indices:
        for action in ACTION_ORDER:
            candidates.append(cost_model.estimate_D_remaining(
                boundary=boundary,
                instance_index=instance_index,
                action=action,
                decision_epoch=decision_epoch,
                inherited_obligation_ids=inherited_obligation_ids,
            ))
# D1（2026-10-16 E0 复核仲裁退役）：同 P 入口——枚举完成后
# try/finally 包住全部边界选择逻辑（含掩码分支与 raise 路径）。
    try:
        applicable = [
            candidate for candidate in candidates
            if candidate.applicable and candidate.cost_ns is not None
        ]
        # 掩码感知红线判据：延续动作（恒 stay@i_P——完整型布局单族）是否
        # 被 JOINT_ALLOWED_ACTIONS 剔除（``getattr`` 兼容选择器存根——真实
        # JCM 恒有该属性，None=无掩码）。
        continuation_action = ACTION_STAY
        allowed_actions = getattr(cost_model, "allowed_actions", None)
        continuation_masked = (
            allowed_actions is not None
            and continuation_action not in allowed_actions)
        if not applicable:
            if continuation_masked:
                # 掩码把允许面整体剔空：fail-closed（明确归因掩码、非容量
                # 拒绝；不许空序列 min()）。
                raise JointSchedulerError(
                    "boundary D selection found no applicable candidate "
                    "under JOINT_ALLOWED_ACTIONS mask (continuation "
                    f"{continuation_action}@i_P is mask-excluded too; "
                    f"p_instance={boundary.p_instance_index}, "
                    f"p_action={boundary.p_action!r})")
            # 原地继续恒为候选（验收红线）：x0 记录缺失 = 边界视图与候选
            # 面不一致（结构性缺陷，非容量拒绝）——fail-closed。
            raise JointSchedulerError(
                "boundary traversal lost the in-place continuation record "
                f"(p_instance={boundary.p_instance_index}, "
                f"p_action={boundary.p_action!r})")
        continuation_records = [
            candidate for candidate in applicable if candidate.is_continuation]
        if continuation_masked:
            if continuation_records:
                # 掩码剔除延续动作后其记录仍现身 = 候选面掩码过滤失效
                # （状态不一致，非容量拒绝）——fail-closed。
                raise JointSchedulerError(
                    "mask-excluded continuation record is still applicable "
                    f"({continuation_action}@i_P; p_action="
                    f"{boundary.p_action!r})")
            best = min(applicable, key=lambda candidate: (
                candidate.cost_ns, candidate.order_key()))
            ties = [
                candidate for candidate in applicable
                if candidate.cost_ns == best.cost_ns
                and candidate.order_key() != best.order_key()
            ]
            decision_note = (
                f"continuation ({continuation_action}@i_P) excluded by "
                "JOINT_ALLOWED_ACTIONS mask; selected within allowed face")
            tie_note = (
                f"tie among {len(ties) + 1} at cost {best.cost_ns}ns -> "
                "(cost, instance, action order)" if ties else decision_note)
            return SelectionRecord(
                mode=mode,
                chosen=best,
                candidates=tuple(candidates),
                instance_rule_note=(
                    "boundary instance x action argmin; continuation x0 "
                    "merged into (i_P, action) records; continuation "
                    "mask-excluded (M2 upper bound, not a mandate)"),
                tie_break_note=tie_note,
                diagnostics={
                    "decision_stage": "D",
                    "decision_epoch": int(decision_epoch),
                    "p_instance_index": boundary.p_instance_index,
                    "p_action": boundary.p_action,
                    "decode_steps": int(boundary.decode_steps),
                    "is_continuation": bool(best.is_continuation),
                    "continuation_action": continuation_action,
                    "continuation_cost_ns": None,
                    "continuation_masked_out": True,
                },
            )
        if not continuation_records:
            # 原地继续恒为候选（验收红线）：x0 记录缺失 = 边界视图与候选
            # 面不一致（结构性缺陷，非容量拒绝）——fail-closed。
            raise JointSchedulerError(
                "boundary traversal lost the in-place continuation record "
                f"(p_instance={boundary.p_instance_index}, "
                f"p_action={boundary.p_action!r})")
        best = min(applicable, key=lambda candidate: (
            candidate.cost_ns, candidate.boundary_order_key()))
        ties = [
            candidate for candidate in applicable
            if candidate.cost_ns == best.cost_ns
            and candidate.boundary_order_key() != best.boundary_order_key()
        ]
        if best.is_continuation:
            decision_note = "in-place continuation (equal-cost preferred)"
        else:
            continuation = continuation_records[0]
            decision_note = (
                "switch from continuation "
                f"({continuation.cost_ns}ns -> {best.cost_ns}ns, strict "
                "improvement per Eq.1c)")
        return SelectionRecord(
            mode=mode,
            chosen=best,
            candidates=tuple(candidates),
            instance_rule_note=(
                "boundary instance x action argmin; continuation x0 merged "
                "into (i_P, action) records"),
            tie_break_note=(
                f"tie among {len(ties) + 1} at cost {best.cost_ns}ns -> "
                "(cost, non-continuation, instance, action order)"
                if ties else decision_note),
            diagnostics={
                "decision_stage": "D",
                "decision_epoch": int(decision_epoch),
                "p_instance_index": boundary.p_instance_index,
                "p_action": boundary.p_action,
                "decode_steps": int(boundary.decode_steps),
                "is_continuation": bool(best.is_continuation),
                "continuation_cost_ns": continuation_records[0].cost_ns,
            },
        )
    finally:
        _release_projection_contexts(candidates)
