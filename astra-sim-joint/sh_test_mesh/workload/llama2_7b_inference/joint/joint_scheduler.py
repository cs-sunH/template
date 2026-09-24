"""joint_scheduler.py -- J：发射点选择（joint / load-first / affinity-first
/ face_static）。

设计依据：《三机制联合策略_template仓库设计方案》§1（J 语义）、§7.1
（J-off 顺序参照）与实验总纲 §13.3（内部策略替代）；C17 卡（face_static
静态距离对照臂）。

四模式（同一候选枚举与计费，只换选择规则）：

* ``joint``（J-on）：同一队列发射决策点比较全部 instance ×
  stay/recompute/copy/remote-read 的 ``cost``（到 merge_done 终点的
  预测完成时间——F11 口径：预测终点恒 merge_done，service_done 作
  服务指标另行分报），取最小。候选评估只读——不为每个候选实际驱逐。
* ``load-first``（J-off，八组合固定参照）：先在**全部 instance**（无
  容量/边缘/距离掩码——HBM 容量只影响动作计费中的驱逐等待，不做准
  入过滤）按在线任务负载（ns 服务台账）选 instance，再在该位置对四
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
定义，不按事后输赢调整）。remote on/off 是正交能力开关：off 仅从候
选动作集中移除 remote-read，全部 instance、其余动作与生命周期不变。

本模块只做选择与记录；发射/驱逐提交/合并事务由在线调度器与
KVCacheManager 执行（§7 模块边界：scheduler 管选择与事务提交，
cost_model 管纯预测）。
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
    ActionCandidate,
    JointCostModel,
    SessionKVView,
)
from joint.link_quota import derive_rho_eff

_SCHEDULER_MODES = (
    SCHEDULER_MODE_JOINT, SCHEDULER_MODE_LOAD_FIRST,
    SCHEDULER_MODE_AFFINITY, SCHEDULER_MODE_FACE_STATIC,
)


class JointSchedulerError(ValueError):
    """fail-closed：非法模式或状态不一致。"""


@dataclass
class SelectionRecord:
    """一次发射决策的完整记录（决策日志逐项落盘，§7/§13.2）。"""

    mode: str
    chosen: ActionCandidate
    candidates: tuple[ActionCandidate, ...]
    instance_rule_note: str
    remote_enabled: bool
    tie_break_note: str = ""
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "remote_enabled": self.remote_enabled,
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
    """单实例内四动作择优（确定 tie 序：ACTION_ORDER）。"""
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
    remote_enabled: bool,
) -> SelectionRecord:
    """发射点选择：枚举全候选 -> 按模式选择 -> 返回完整记录。

    ``request`` 为 RequestView（无 oracle 输入；结构上不接受
    decode_length/final_context_tokens）。枚举阶段对每个 (instance,
    action) 调用 ``estimate_action``（只读，不修改任何账本或事件队列）。
    """
    if mode not in _SCHEDULER_MODES:
        raise JointSchedulerError(f"unknown scheduler mode: {mode!r}")
    instance_indices = sorted(cost_model.loads.keys())
    if not instance_indices:
        raise JointSchedulerError("no instance load views available")

    candidates: list[ActionCandidate] = []
    for instance_index in instance_indices:
        for action in ACTION_ORDER:
            candidates.append(cost_model.estimate_action(
                session=session,
                request=request,
                instance_index=instance_index,
                action=action,
                remote_enabled=remote_enabled,
            ))

    if mode == SCHEDULER_MODE_JOINT:
        applicable = [
            candidate for candidate in candidates
            if candidate.applicable and candidate.cost_ns is not None
        ]
        if not applicable:
            raise JointSchedulerError(
                "joint selection found no applicable candidate "
                "(execution-protocol gap; not a capacity rejection)")
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
            remote_enabled=remote_enabled,
            tie_break_note=(
                f"tie among {len(ties) + 1} at cost {best.cost_ns}ns -> "
                "(instance_index, action order)" if ties else ""),
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
            # 参与）——如实记录，不 fail-closed（recompute 恒适用）。
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
                continue  # recompute 恒适用，实际不可达
            key = (best_at_index.cost_ns, best_at_index.order_key())
            if chosen_key is None or key < chosen_key:
                chosen_instance = index
                chosen_key = key
        if chosen_instance is None:  # pragma: no cover - 防御性 fail-closed
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
        # recompute 在任何 instance 恒适用，此处不可达；防御性 fail-closed。
        raise JointSchedulerError(
            f"sequential mode {mode!r} found no applicable action at "
            f"instance {chosen_instance}")
    return SelectionRecord(
        mode=mode,
        chosen=best,
        candidates=tuple(candidates),
        instance_rule_note=note,
        remote_enabled=remote_enabled,
        tie_break_note=action_note,
        diagnostics={"sequential_instance": chosen_instance,
                     **extra_diagnostics},
    )
