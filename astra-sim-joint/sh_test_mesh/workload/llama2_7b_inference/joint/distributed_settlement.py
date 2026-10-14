"""distributed_settlement.py -- W9（§7.4）：目录提交型结算与区间级留存。

`settlement_policy=distributed` 的执行面子协议（纯逻辑层，不 import
face_scheduler——被 KVCacheManager/调度器单向消费，避免环）：

* **目录提交型结算**：settle = 提交完整有效目录（区间条目权威化、生成
  版本推进）+ 执行 pin 转可管理驻留引用（结算即释放——不永久 pin）+
  释放 staging/临时预约 + 闭合必要旧义务（未完 ack/在途访问）；**不执行
  物理归并**（零 merge 载荷；settle 事件仍产生——W5 状态机零载荷形态）。
* **跨轮留存段（通道 E）**：distributed settle 后仍驻留片上的旧基础段/
  边界来源段——目录条目集合无三位置上限（§7.4），逐段携带
  (holder, layer_range, token_range, bytes, role, state)；驱逐经目录
  ``evict_to_pool``（owner 记录保留、唯一内容在池），恢复经
  ``restore_from_pool`` → RESTORING → READY。
* **下一轮 P 构图查询面**：来源面全量枚举（``enumerate_source_surface``：
  位置数与区间数分别统计、跨轮增长 ≤ +2 无拆分披露），池区间另列。
* **开销计量**（§10）：目录操作计数 + host 累计墙时 + 条目数/来源数
  时间线（与 capacity_summary 的 ACC 维护账同模式——计量不改变行为）。
* **会话终局清理审计**：悬挂引用/未完 ack/在途 copy fail-closed。

纯 Python 无外部依赖；schema 版本引用 ``joint.schema_registry``。
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

from .kv_location_directory import (
    ENTRY_STATE_EVICTED_TO_POOL,
    ENTRY_STATE_READY,
    ENTRY_STATE_RESTORING,
    POOL_HOLDER,
    ROLE_BASE,
    ROLE_BOUNDARY_SOURCE,
    ROLE_RESIDENT,
    KvLocationDirectory,
)
from .schema_registry import KV_DIRECTORY_SCHEMA_VERSION  # noqa: F401

__all__ = [
    "KV_DIRECTORY_SCHEMA_VERSION",
    "DistributedSettleError",
    "RetainedSegment",
    "DistributedSettleOutcome",
    "SourceSurface",
    "DirectoryOverheadMeter",
    "enumerate_source_surface",
    "cell_coverage_complete",
    "assert_full_content_locatable",
    "assert_no_dangling_session_state",
    "assert_round_growth_within_budget",
]

#: 无其他拆分时单轮位置数增量的合法上界（§7.4：初始 m 个位置本轮最多
#: +2——P/D 实例；超出必须来自显式拆分，由调用方披露而非静默放行）。
DEFAULT_ROUND_GROWTH_BUDGET = 2


class DistributedSettleError(RuntimeError):
    """distributed 结算协议违规（fail-closed：覆盖断言/版本键/悬挂态）。"""


@dataclass(frozen=True)
class RetainedSegment:
    """跨轮留存段（目录通道 E 的执行面载体）。

    分布式结算后仍驻留片上（或池内）的旧基础/边界来源区间：内容键
    ``(session, holder, layer_range, token_range)``；state 与目录条目状态
    机同词表（READY / EVICTED_TO_POOL / RESTORING——holder=池时 state 恒
    EVICTED_TO_POOL/RESTORING 语义由 target_instance 承载）。生成版本在
    每次目录提交时 +1（区间条目权威化）。
    """

    session_id: str
    instance_index: int
    layer_start: int
    layer_end: int
    token_start: int
    token_end: int
    bytes_by_tp_rank: tuple[int, ...]
    role: str
    state: str = ENTRY_STATE_READY
    generation: int = 1
    #: RESTORING 态的目标实例（恢复发射在途；READY/EVICTED 时 None）。
    restoring_to: Optional[int] = None
    #: 审计链（驱逐/恢复/结算来源——owner 记录保留的披露面）。
    history: tuple[str, ...] = ()

    @property
    def key(self) -> tuple[int, int, int, int, int]:
        return (
            self.instance_index, self.layer_start, self.layer_end,
            self.token_start, self.token_end)

    def is_on_chip(self) -> bool:
        return self.state == ENTRY_STATE_READY

    def total_bytes(self) -> int:
        return sum(self.bytes_by_tp_rank)


@dataclass(frozen=True)
class DistributedSettleOutcome:
    """一次目录提交型结算的披露快照（settle 事件行/kv_delta 消费）。"""

    session_id: str
    trigger_request_id: str
    settlement_policy: str = "distributed"
    #: 结算家族（zero_byte_flip：exec 持完整并集 / stay_local 等）。
    family: str = ""
    final_tokens: int = 0
    merge_payload_bytes: int = 0
    #: 同轮口径：本轮若走 consolidate 将发生的合并载荷估计（base 侧驻留
    #: 字节——仅同轮对照披露，**不得**报告为长期收益；E2E 对照按完整
    #: 会话 service+settle 分列）。
    same_round_merge_saved_bytes: int = 0
    #: 提交后的目录事实（来源面 = enumerate_source_surface 输出）。
    source_count: int = 0
    interval_count: int = 0
    retained_keys: tuple[tuple[int, int, int, int, int], ...] = ()
    directory_version: int = 0
    commit_ns: int = 0

    def as_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "trigger_request_id": self.trigger_request_id,
            "settlement_policy": self.settlement_policy,
            "family": self.family,
            "final_tokens": self.final_tokens,
            "merge_payload_bytes": self.merge_payload_bytes,
            "same_round_merge_saved_bytes": self.same_round_merge_saved_bytes,
            "source_count": self.source_count,
            "interval_count": self.interval_count,
            "retained_key_count": len(self.retained_keys),
            "directory_version": self.directory_version,
            "commit_ns": self.commit_ns,
        }


@dataclass(frozen=True)
class SourceSurface:
    """下一轮 P 构图的目录全量来源面（§7.4：区间数与位置数分别统计）。"""

    session_id: str
    #: 片上来源条目（READY/WRITING；holder ≠ 池）。逐条 (holder,
    #: layer_range, token_range, bytes, state, role)。
    on_chip: tuple[tuple[int, tuple[int, int], tuple[int, int], int, str, str], ...]
    #: 池区间条目（EVICTED_TO_POOL——恢复另列资源）。
    pooled: tuple[tuple[int, tuple[int, int], tuple[int, int], int, str, str], ...]
    #: 恢复在途条目（RESTORING@target）。
    restoring: tuple[tuple[int, tuple[int, int], tuple[int, int], int, str, str], ...]

    @property
    def source_count(self) -> int:
        """位置数 = 片上来源的不同 holder 数（§7.4 分别统计的位置维）。"""
        return len({item[0] for item in self.on_chip})

    @property
    def interval_count(self) -> int:
        """区间数 = 片上有效条目数（目录条目逐区间记账，§7.4 区间维）。"""
        return len(self.on_chip)

    def as_dict(self) -> dict:
        def _entry(item):
            holder, layer_range, token_range, total, state, role = item
            return {
                "holder": holder,
                "layer_range": list(layer_range),
                "token_range": list(token_range),
                "bytes": total,
                "state": state,
                "role": role,
            }

        return {
            "session_id": self.session_id,
            "source_count": self.source_count,
            "interval_count": self.interval_count,
            "on_chip": [_entry(item) for item in self.on_chip],
            "pooled": [_entry(item) for item in self.pooled],
            "restoring": [_entry(item) for item in self.restoring],
        }


def enumerate_source_surface(
    directory: KvLocationDirectory,
    session_id: str,
    *,
    exclude_holder: Optional[int] = None,
) -> SourceSurface:
    """目录全量来源面枚举（W4 执行面 query_sources 的会话级全量形态）。

    与 ``query_sources`` 的差别：不做目标实例排除（下一轮 P 构图对全部
    候选目标共用同一来源面，逐目标缺失由 ``query_missing_intervals``
    另查）；池/恢复在途条目分列（§6.2 分列披露）。
    """
    on_chip: list[tuple[int, tuple[int, int], tuple[int, int], int, str, str]] = []
    pooled: list[tuple[int, tuple[int, int], tuple[int, int], int, str, str]] = []
    restoring: list[tuple[int, tuple[int, int], tuple[int, int], int, str, str]] = []
    for entry in directory.entries_for_session(session_id, valid_only=False):
        if not entry.is_valid():
            continue
        key = entry.content_key
        item = (
            int(entry.holder), key.layer_range, key.token_range,
            entry.total_bytes, entry.state, entry.role,
        )
        if entry.state == ENTRY_STATE_RESTORING:
            restoring.append(item)
        elif entry.is_pool_backed():
            pooled.append(item)
        elif entry.holder != exclude_holder:
            on_chip.append(item)
    return SourceSurface(
        session_id=session_id,
        on_chip=tuple(on_chip),
        pooled=tuple(pooled),
        restoring=tuple(restoring),
    )


def cell_coverage_complete(
    directory: KvLocationDirectory,
    session_id: str,
    *,
    history_tokens: int,
    model_layers: int,
    holder: Optional[int] = None,
) -> tuple[bool, int]:
    """目录区间**单元格级**并集覆盖审计（distributed 跨轮重叠区间）。

    W4 的 ``effective_resident_bytes`` 现按**单元格并集**去重（与本函
    数同一口径）；历史上它按区间身份去重——distributed 跨轮留存段造
    出的跨持有者**部分重叠**区间（如 [0,10)@A 与 [0,12)@B）会被计两
    遍。本函数把全部条目边界切成初等单元格
    （token × layer），逐格判定覆盖（任一条目覆盖即算唯一内容可定位），
    返回 (complete, covered_area)：

    * complete = 期望矩形 [0, history) × [0, L) 的每个初等格都有 ≥1
      条有效条目覆盖（片上或池——唯一内容可定位，§7.4 驱逐不清 owner
      后唯一内容在池等合法位置）；
    * covered_area = 覆盖格的 token×layer 面积之和（字节按本仓 KV 线
      性模型随面积精确比例，面积完备 ⇔ 字节完备——避免整数除法精度）。
    """
    entries = [
        entry for entry in directory.entries_for_session(session_id)
        if entry.is_valid()
        and (holder is None or entry.holder == holder)]
    token_bounds = {0, int(history_tokens)}
    layer_bounds = {0, int(model_layers)}
    for entry in entries:
        key = entry.content_key
        token_bounds.update((key.token_start, key.token_end))
        layer_bounds.update((key.layer_start, key.layer_end))
    token_cells = sorted(b for b in token_bounds if 0 <= b <= history_tokens)
    layer_cells = sorted(b for b in layer_bounds if 0 <= b <= model_layers)
    covered_area = 0
    complete = True
    for t_index in range(len(token_cells) - 1):
        t0, t1 = token_cells[t_index], token_cells[t_index + 1]
        if t1 <= t0:
            continue
        for l_index in range(len(layer_cells) - 1):
            l0, l1 = layer_cells[l_index], layer_cells[l_index + 1]
            if l1 <= l0:
                continue
            covered = any(
                entry.content_key.token_start <= t0
                and t1 <= entry.content_key.token_end
                and entry.content_key.layer_start <= l0
                and l1 <= entry.content_key.layer_end
                for entry in entries)
            if covered:
                covered_area += (t1 - t0) * (l1 - l0)
            else:
                complete = False
    return complete, covered_area


def assert_full_content_locatable(
    directory: KvLocationDirectory,
    session_id: str,
    *,
    history_tokens: int,
    model_layers: int,
) -> int:
    """G-DIST"全部有效区间可定位"断言（fail-closed；返回覆盖面积）。"""
    complete, covered_area = cell_coverage_complete(
        directory, session_id,
        history_tokens=history_tokens, model_layers=model_layers)
    expected_area = int(history_tokens) * int(model_layers)
    if not complete or covered_area != expected_area:
        raise DistributedSettleError(
            f"session {session_id} unique content not fully locatable: "
            f"coverage complete={complete}, covered_area={covered_area} != "
            f"expected {expected_area} (history={history_tokens} tokens × "
            f"{model_layers} layers)")
    return covered_area


def assert_round_growth_within_budget(
    previous_source_count: int,
    surface: SourceSurface,
    *,
    budget: int = DEFAULT_ROUND_GROWTH_BUDGET,
    explicit_splits: int = 0,
) -> int:
    """§7.4 增长规则断言：无显式拆分时单轮位置增量 ≤ budget（P/D 实例）。

    返回本轮增长量（披露用）；超限即 ``DistributedSettleError``（fail-
    closed——超限增长必须来自登记过的显式拆分，不得静默放行）。
    """
    growth = surface.source_count - previous_source_count
    allowed = budget + explicit_splits
    if growth > allowed:
        raise DistributedSettleError(
            f"session {surface.session_id} on-chip source positions grew by "
            f"{growth} in one round (previous {previous_source_count} -> "
            f"{surface.source_count}); budget={budget}+{explicit_splits} "
            "explicit splits (sec.7.4: at most +P/D positions per round "
            "without explicit splits)")
    return growth


def assert_no_dangling_session_state(
    directory: KvLocationDirectory,
    session_id: str,
) -> None:
    """会话终局清理审计：悬挂引用/未完 ack/在途 copy fail-closed。

    在 ``forget_session`` 前调用：任何未闭合义务（read_refs>0 / exec pin /
    pending acks / 在途 copy 记录）都是清理合同违规——调用方必须先闭合
    （结算/取消）再清理（§7.4"会话真正结束后清理仍须与未完访问/ack 协调"）。
    """
    for entry in directory.entries_for_session(session_id, valid_only=False):
        # 执行 pin 先于裸引用检查（pin 是引用的显式标记位——先报强信号）。
        if directory.is_exec_pinned(entry.entry_id):
            raise DistributedSettleError(
                f"session {session_id} entry {entry.entry_id} still holds "
                "an execution pin at terminal cleanup (dangling pin)")
        if directory.read_refs(entry.entry_id) > 0:
            raise DistributedSettleError(
                f"session {session_id} entry {entry.entry_id} still holds "
                f"{directory.read_refs(entry.entry_id)} live read "
                "references at terminal cleanup (dangling access)")
        if entry.pending_acks:
            raise DistributedSettleError(
                f"session {session_id} entry {entry.entry_id} still has "
                f"pending ack obligations {entry.pending_acks} at terminal "
                "cleanup (dangling ack)")
    inflight = directory.inflight_copies(session_id)
    if inflight:
        raise DistributedSettleError(
            f"session {session_id} still has {len(inflight)} inflight copy "
            "records at terminal cleanup (dangling in-flight access)")


class DirectoryOverheadMeter:
    """W9 开销计量（§10）：目录操作计数 + host 累计墙时 + 规模时间线。

    与 capacity_summary 的 ACC 维护账同模式：计量不改行为；调用方在
    run 末把 ``snapshot`` 并入 manifest / 决策统计通道。纯计数器实现
    （无锁——仿真单线程决策路径）。
    """

    def __init__(self) -> None:
        self.op_counts: dict[str, int] = {}
        self.op_total_ns: dict[str, int] = {}
        self.entry_count_timeline: tuple[tuple[int, int], ...] = ()
        self._entry_timeline_list: list[tuple[int, int]] = []
        self.session_rounds: dict[str, list[dict[str, int]]] = {}
        self.peak_entry_count = 0

    def record_op(self, op: str, *, elapsed_ns: int = 0) -> None:
        self.op_counts[op] = self.op_counts.get(op, 0) + 1
        if elapsed_ns:
            self.op_total_ns[op] = self.op_total_ns.get(op, 0) + int(elapsed_ns)

    def timed_op(self, op: str):
        """上下文管理器形态（with meter.timed_op("query_sources"): ...）。"""
        import contextlib

        start = time.perf_counter_ns()

        @contextlib.contextmanager
        def _ctx():
            try:
                yield
            finally:
                self.record_op(op, elapsed_ns=time.perf_counter_ns() - start)

        return _ctx()

    def record_directory_commit(
        self,
        directory: KvLocationDirectory,
        session_id: str,
        outcome: DistributedSettleOutcome,
    ) -> None:
        """结算提交点计量：条目数时间线 + 逐会话逐轮来源/区间/增长。"""
        self.peak_entry_count = max(
            self.peak_entry_count, directory.entry_count)
        self._entry_timeline_list.append(
            (directory.version, directory.entry_count))
        self.entry_count_timeline = tuple(self._entry_timeline_list)
        rounds = self.session_rounds.setdefault(session_id, [])
        previous = rounds[-1]["source_count"] if rounds else 0
        rounds.append({
            "round": len(rounds) + 1,
            "source_count": outcome.source_count,
            "interval_count": outcome.interval_count,
            "growth": outcome.source_count - previous,
            "retained_keys": len(outcome.retained_keys),
            "directory_version": outcome.directory_version,
        })

    def snapshot(self) -> dict:
        """run 末披露快照（manifest `distributed_directory_stats` 节）。"""
        return {
            "op_counts": dict(sorted(self.op_counts.items())),
            "op_total_ns": dict(sorted(self.op_total_ns.items())),
            "peak_entry_count": self.peak_entry_count,
            "entry_count_timeline_length": len(self.entry_count_timeline),
            "session_rounds": {
                session_id: list(rounds)
                for session_id, rounds in sorted(
                    self.session_rounds.items())},
        }


# (RETAINED_ROLE_TO_DIRECTORY_ROLE was removed with the 2026-10-01
# dead-code audit: the identity map had no consumer; the channel-E role
# history is carried by the entry role/history fields themselves.)
