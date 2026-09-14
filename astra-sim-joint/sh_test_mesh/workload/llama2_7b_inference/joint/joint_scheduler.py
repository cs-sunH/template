"""joint_scheduler.py -- J：发射点选择（joint / load-first / affinity-first）。

设计依据：《三机制联合策略_template仓库设计方案》§1（J 语义）、§7.1
（J-off 顺序参照）与实验总纲 §13.3（内部策略替代）。

三模式（同一候选枚举与计费，只换选择规则）：

* ``joint``（J-on）：同一队列发射决策点比较全部 instance ×
  stay/recompute/copy/remote-read 的 ``cost``（到 service_done 边界的
  预测完成时间），取最小。候选评估只读——不为每个候选实际驱逐。
* ``load-first``（J-off，八组合固定参照）：先在**全部 instance**（无
  容量/边缘/距离掩码——HBM 容量只影响动作计费中的驱逐等待，不做准
  入过滤）按在线任务负载（ns 服务台账）选 instance，再在该位置对四
  动作取最小代价。
* ``affinity-first``（J 边际比较第二参照，不进八组合）：有可复用历史
  时优先选择其逻辑 home；无历史或无唯一 home 时回退 load-first。选定
  实例后的动作计费、驱逐和生命周期保持共同规则（亲和是选择偏好，
  不是容量/距离掩码）。

平局与 tie 序：固定 ``(instance_index, ACTION_ORDER)`` 确定序（事前
定义，不按事后输赢调整）。remote on/off 是正交能力开关：off 仅从候
选动作集中移除 remote-read，全部 instance、其余动作与生命周期不变。

本模块只做选择与记录；发射/驱逐提交/合并事务由在线调度器与
KVCacheManager 执行（§7 模块边界：scheduler 管选择与事务提交，
cost_model 管纯预测）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from joint.joint_config import (
    SCHEDULER_MODE_AFFINITY,
    SCHEDULER_MODE_JOINT,
    SCHEDULER_MODE_LOAD_FIRST,
)
from joint.joint_cost_model import (
    ACTION_ORDER,
    ActionCandidate,
    JointCostModel,
    SessionKVView,
)

_SCHEDULER_MODES = (
    SCHEDULER_MODE_JOINT, SCHEDULER_MODE_LOAD_FIRST,
    SCHEDULER_MODE_AFFINITY,
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

    # ---- 顺序参照：先定 instance，再在该位置选动作。 ----
    if mode == SCHEDULER_MODE_LOAD_FIRST:
        chosen_instance = min(
            instance_indices,
            key=lambda index: (
                cost_model.loads[index].total_task_load_ns, index))
        note = "load-first: min task-load instance (all instances, no mask)"
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
        diagnostics={"sequential_instance": chosen_instance},
    )
