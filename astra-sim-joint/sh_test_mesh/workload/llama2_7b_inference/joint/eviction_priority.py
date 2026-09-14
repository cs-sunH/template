"""eviction_priority.py -- T：victim 类别优先级（策略纯函数，不写账本）。

设计依据：《三机制联合策略_template仓库设计方案》§4 与 §7.1。

T 只决定"同一待释放物理资源上合法 victim 的类别/对象优先级"，不决定
每次释放几层（层数由 E 的 layer_eviction_policy 决定），也不越过在用
保护（protected/active/未完成 session 不属于合法 victim）。

两种模式：

* ``typed``（T-on，SH 严格类别顺序）：合法 victim 先 human 类、后 tool
  类；类内沿用 ``(last_completion_ns, session_id)`` FIFO 顺序；human 类
  内可释放层未耗尽且缺口未满足时**不得**转向 tool。类别是决策时已知的
  等待状态：``next_request_type == "tool"`` 归工具类，缺失及其他值按
  human 优先级（SH fallback 裁定），但日志保留原始值与 fallback 标记。
* ``lru``（T-off）：同一合法 victim 集合上的类型无关 LRU——单一遍历，
  不分类，其余流程（两阶段结构、层数策略、保护规则、排序键）不变。

本模块不持有可变状态、不读取环境；KVCacheManager 在每次释放规划时以
只读方式调用。E 的层数选择不由本模块代定（§4 末条）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

CATEGORY_TYPED = "typed"
CATEGORY_LRU = "lru"

HUMAN_CLASS = "human"
TOOL_CLASS = "tool"


class EvictionPriorityError(ValueError):
    """fail-closed：非法模式或非法类别。"""


@dataclass(frozen=True)
class EvictionClassPass:
    """一次类别遍历：``trigger_type=None`` 表示类型无关全集合遍历。"""

    trigger_type: Optional[str]

    def describe(self) -> str:
        return self.trigger_type if self.trigger_type is not None else "all"


def eviction_class_order(category_mode: str) -> tuple[EvictionClassPass, ...]:
    """返回类别遍历序（typed = human→tool 两遍；lru = 单遍全集合）。

    typed 序保证严格类别优先：human 遍内"可合法释放的 human 层"未耗尽
    且缺口未满足时，调用方不得提前进入 tool 遍（§4）。lru 序不改变合法
    victim 集合与类内排序键，只去掉类别维度（T-off 语义）。
    """
    if category_mode == CATEGORY_TYPED:
        return (
            EvictionClassPass(trigger_type=HUMAN_CLASS),
            EvictionClassPass(trigger_type=TOOL_CLASS),
        )
    if category_mode == CATEGORY_LRU:
        return (EvictionClassPass(trigger_type=None),)
    raise EvictionPriorityError(
        f"unknown category_mode: {category_mode!r}")


def classify_session_class(next_request_type: Optional[str]) -> str:
    """session 的逐出类别：仅显式 ``tool`` 归工具类，其余（含 human、
    None、未知值）按 human 优先级（SH fallback 裁定，§4）。

    注意：这不是把未知语义断言为真实人类返回——调用方须用
    :func:`classify_with_fallback` 保留原始值与 fallback 标记入日志。
    """
    return TOOL_CLASS if next_request_type == "tool" else HUMAN_CLASS


@dataclass(frozen=True)
class SessionClassRecord:
    """类别判定结果（含 fallback 披露，供决策/事件日志使用）。"""

    eviction_class: str
    raw_next_request_type: Optional[str]
    fallback: bool

    def as_dict(self) -> dict:
        return {
            "eviction_class": self.eviction_class,
            "raw_next_request_type": self.raw_next_request_type,
            "fallback": self.fallback,
        }


def classify_with_fallback(
    next_request_type: Optional[str],
) -> SessionClassRecord:
    """类别判定 + fallback 标记：原始值非 ``tool``/``human``/``None``
    时 fallback=True（未知值按 human 优先级排序并披露原值）。"""
    if next_request_type in (None, "human", "tool"):
        return SessionClassRecord(
            eviction_class=(
                TOOL_CLASS if next_request_type == "tool" else HUMAN_CLASS),
            raw_next_request_type=next_request_type,
            fallback=False,
        )
    return SessionClassRecord(
        eviction_class=HUMAN_CLASS,
        raw_next_request_type=next_request_type,
        fallback=True,
    )


def victim_sort_key(session):
    """类内确定序：``(last_completion_ns, session_id)``（SH 原序）。

    两模式共用同一排序键；``last_completion_ns`` 为 None 的 session 不是
    合法 victim（未完成），调用方应先行过滤。
    """
    completion = session.last_completion_ns
    if completion is None:
        raise EvictionPriorityError(
            f"session {session.session_id!r} has no completion time; "
            "it is not a legal eviction victim")
    return (int(completion), session.session_id)
