"""kv_location_directory.py -- 唯一权威 KV 区间目录（W4，§6.1）。

内容键 ``(session_id, model_version, layer_range, token_range, tp_rank)``；
物理条目携带 owner/replica 角色、所在 instance 或池、实际字节/padding、
生成版本、ready/ack 义务、读取引用、执行 pin、驱逐/恢复状态与释放条件。
目录、KV 管理器与 journal 共享一次事务事实，不能各自维护互相猜测的
home（``logical_home`` 只作逻辑锚点；``execution_instance`` 只是阶段执行
位置；两者都不替代本目录）。

守恒不变量（``assert_conservation`` 单一裁决点，fail-closed）：

1. **同 (内容, 持有者) 唯一**：同一内容键在同一 holder 至多一条未释放
   条目（重复副本按条目分立，不合并字节）；
2. **唯一有效内容按区间并集只计一次**：会话的有效保留量 V
   （merge 目标排序口径，式(21)）= 全部未释放条目区间并集的字节，
   重复副本不增加 V（§15.1）；
3. **未释放物理副本都占容量**：``capacity_projection`` 把每条实例驻留
   条目的逐 rank 字节直和入持有实例（池条目不占实例容量）；
4. **引用>0 / 执行 pin 的条目不可驱逐**（读取引用守恒：acquire/release
   配对，悬空 release fail-closed）；
5. **驱逐不清 owner 记录**：逐出转 ``EVICTED_TO_POOL``（holder=池），
   唯一内容保留在池等合法位置，owner/角色/生成版本不动（§7.4）。

W9 预留（结构不写死，本包不启用）：条目集合无三位置上限；
``query_sources``/``query_missing_intervals`` 对任意来源数迭代；
跨轮 distributed 结算 = 目录提交型（本包只提供引用/义务位）。

缺失/来源推导口径（§6.1/§6.3，V6/W4）：按**初等单元格（token×layer）
并集**去重——目标已持有或已在途的内容不重复搬运/重复读；同内容多副
本按冻结确定性顺序（``ROLE_ORDER`` → 层/token → holder）只选读一条
权威副本，部分重叠条目只补差集片段（字节按面积线性精确比例）。

纯 Python 无外部依赖；不 import face_scheduler（被其镜像层单向消费，
避免环）。schema 版本从 ``joint.schema_registry`` 引用（KV_DIRECTORY_
SCHEMA_VERSION，W0 冻结——语义改变必须升级该常量）。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterable, Optional, Sequence

from .schema_registry import KV_DIRECTORY_SCHEMA_VERSION  # noqa: F401

__all__ = [
    "KV_DIRECTORY_SCHEMA_VERSION",
    "KvContentKey",
    "KvDirectoryEntry",
    "KvLocationDirectory",
    "KvDirectoryError",
    "POOL_HOLDER",
    "ENTRY_STATE_WRITING",
    "ENTRY_STATE_READY",
    "ENTRY_STATE_EVICTED_TO_POOL",
    "ENTRY_STATE_RESTORING",
    "ENTRY_STATE_RELEASED",
    "ROLE_RESIDENT",
    "ROLE_WORKING",
    "ROLE_BASE",
    "ROLE_BOUNDARY_SOURCE",
    "ROLE_POOL_BACKED",
    "ROLE_ORDER",
]

#: 池持有者哨兵（片外共享内存池；区别于任意实例编号）。
POOL_HOLDER = -1

# ---- 条目状态机（§6.1 物理条目字段：生成版本、ready/ack 义务、驱逐/恢复状态） ----
#: 物理化在途（copy 交接 / 重算增长 / 池恢复发射后未就绪）。
ENTRY_STATE_WRITING = "WRITING"
#: 权威可消费（ready 义务闭合、ack 齐备）。
ENTRY_STATE_READY = "READY"
#: 逐出到池（owner 记录保留；唯一内容在池）。
ENTRY_STATE_EVICTED_TO_POOL = "EVICTED_TO_POOL"
#: 池恢复发射在途（目标实例物化未就绪）。
ENTRY_STATE_RESTORING = "RESTORING"
#: 物理副本已释放（终态墓碑；不占容量、不计 V）。
ENTRY_STATE_RELEASED = "RELEASED"

#: 合法状态转移表（fail-closed：表外转移 = 目录事务违规）。
_ENTRY_TRANSITIONS: dict[str, frozenset[str]] = {
    ENTRY_STATE_WRITING: frozenset({
        ENTRY_STATE_READY, ENTRY_STATE_RELEASED}),
    ENTRY_STATE_READY: frozenset({
        ENTRY_STATE_EVICTED_TO_POOL, ENTRY_STATE_RELEASED}),
    ENTRY_STATE_EVICTED_TO_POOL: frozenset({
        ENTRY_STATE_RESTORING, ENTRY_STATE_RELEASED}),
    ENTRY_STATE_RESTORING: frozenset({
        ENTRY_STATE_READY, ENTRY_STATE_EVICTED_TO_POOL,
        ENTRY_STATE_RELEASED}),
    ENTRY_STATE_RELEASED: frozenset(),
}

#: 实例驻留（占容量）状态集。
_ON_CHIP_STATES = frozenset({
    ENTRY_STATE_WRITING, ENTRY_STATE_READY, ENTRY_STATE_RESTORING})
#: 有效（计入并集 V）状态集——RELEASED 墓碑除外。
_VALID_STATES = frozenset({
    ENTRY_STATE_WRITING, ENTRY_STATE_READY, ENTRY_STATE_EVICTED_TO_POOL,
    ENTRY_STATE_RESTORING})

# ---- 角色（owner/replica 语义；W4 镜像层与现有三槽对齐，可自由扩展） ----
ROLE_RESIDENT = "resident"            # 主驻留（非工作副本期）
ROLE_WORKING = "working"              # 跨实例执行工作副本
ROLE_BASE = "base"                    # 权威基础历史（base_* 槽）
ROLE_BOUNDARY_SOURCE = "boundary-source"  # 边界第二来源（i_P 旧副本）
ROLE_POOL_BACKED = "pool-backed"      # 唯一内容仅在池

#: 冻结的确定性来源顺序（§6.1"多个可选副本按冻结的确定性顺序选读"）：
#: base 先于 boundary-source（与 credit plan 现行 home→边界源序一致）。
ROLE_ORDER: dict[str, int] = {
    ROLE_BASE: 0,
    ROLE_BOUNDARY_SOURCE: 1,
    ROLE_RESIDENT: 2,
    ROLE_WORKING: 3,
    ROLE_POOL_BACKED: 4,
}


class KvDirectoryError(RuntimeError):
    """目录事务违规（fail-closed：状态机/守恒/引用守恒破坏）。"""


@dataclass(frozen=True)
class KvContentKey:
    """内容键 ``(session, model/version, layer_range, token_range, TP_rank)``。

    ``tp_rank`` 为 None 表示整 TP 带（条目登记粒度）；``shard_key`` 生成
    逐 rank 内容键（逐 rank 物理身份——同带条目的逐 rank 字节向量是
    一组 per-rank 内容的联合记账）。
    """

    session_id: str
    model_version: str
    layer_start: int
    layer_end: int
    token_start: int
    token_end: int
    tp_rank: Optional[int] = None

    def shard_key(self, tp_rank: int) -> "KvContentKey":
        return replace(self, tp_rank=int(tp_rank))

    @property
    def layer_range(self) -> tuple[int, int]:
        return (self.layer_start, self.layer_end)

    @property
    def token_range(self) -> tuple[int, int]:
        return (self.token_start, self.token_end)

    def covers_same_content(self, other: "KvContentKey") -> bool:
        """同物理内容（忽略 tp_rank 维的分带/逐 rank 表示差）。"""
        return (
            self.session_id == other.session_id
            and self.model_version == other.model_version
            and self.layer_range == other.layer_range
            and self.token_range == other.token_range
        )


@dataclass(frozen=True)
class KvDirectoryEntry:
    """物理条目（§6.1：owner/replica、instance/池、实际字节/padding、
    生成版本、ready/ack 义务、读取引用、执行 pin、驱逐/恢复状态、释放
    条件）。

    frozen + ``replace`` 转移（每次目录事务整体换新，旧对象即不可变
    历史）；可变计数（读取引用）由目录统一保管（条目不可变，引用守恒
    的单一事实源在目录）。
    """

    entry_id: str
    content_key: KvContentKey
    #: 持有者：实例编号或 POOL_HOLDER（池）。
    holder: int
    #: owner/replica 角色（ROLE_*；owner 记录在驱逐时保留，§7.4）。
    role: str
    #: 实际字节（逐 TP rank 向量——rank 维真值）。
    bytes_by_tp_rank: tuple[int, ...]
    #: padding 字节（逐 TP rank；对齐填充，占容量但非有效载荷）。
    padding_by_tp_rank: tuple[int, ...] = ()
    #: 生成版本（同区间内容的重写代数；目录事务内单调）。
    generation: int = 0
    state: str = ENTRY_STATE_WRITING
    #: ready/ack 义务位（journal 协议 tag；非空 = 尚未 ready）。
    pending_acks: tuple[str, ...] = ()
    #: 释放条件描述（审计披露：journal 边界/结算点/引用归零）。
    release_condition: str = ""

    @property
    def total_bytes(self) -> int:
        return sum(self.bytes_by_tp_rank)

    def is_on_chip(self) -> bool:
        return self.holder != POOL_HOLDER

    def is_pool_backed(self) -> bool:
        return self.holder == POOL_HOLDER

    def is_valid(self) -> bool:
        """有效内容（计入会话保留量 V 的并集；RELEASED 除外）。"""
        return self.state in _VALID_STATES


@dataclass(frozen=True)
class _InflightCopy:
    """在途 copy 记录（去重：已在途的同身份 copy 不重发，§6.1）。"""

    copy_id: str
    session_id: str
    source_holder: int
    target_instance: int
    layer_start: int
    layer_end: int
    token_start: int
    token_end: int
    bytes_by_tp_rank: tuple[int, ...]
    ack_id: str


class KvLocationDirectory:
    """唯一权威区间目录（事务版本单调递增；每操作 = 一次目录事务）。

    本类**不**维护另一套 KV 容量真值：实例容量守恒由持有方
    （``KVCacheManager._rank_states``）裁决，本目录提供
    ``capacity_projection`` 供逐 rank 对账（§3.3"禁止维护另一套不同步
    的 KV 容量真值"——投影是条目直和，不是独立账本）。
    """

    def __init__(
        self,
        *,
        tp_degree: int,
        model_version: str,
    ) -> None:
        if tp_degree <= 0:
            raise ValueError("tp_degree must be positive")
        if not model_version:
            raise ValueError("model_version must be a non-empty string")
        self.tp_degree = int(tp_degree)
        self.model_version = str(model_version)
        #: schema 版本（schema_registry 单一登记处引用）。
        self.schema_version = KV_DIRECTORY_SCHEMA_VERSION
        #: 目录事务版本（单调递增；每次成功事务 +1）。
        self.version = 0
        self._entries: dict[str, KvDirectoryEntry] = {}
        #: (session_id, holder) → entry_id 集合（容量投影/同步差量索引）。
        self._by_holder: dict[tuple[str, int], set[str]] = {}
        #: 会话在途 copy 记录（去重键 = (session, source, target, 区间)）。
        self._inflight_copies: dict[str, _InflightCopy] = {}
        #: 逐条目读取引用计数（区间级引用守恒，§7.4"执行 pin 转可管理
        #: 驻留引用"）；执行 pin = 引用的特例（布尔位单列披露）。
        self._read_refs: dict[str, int] = {}
        self._exec_pins: dict[str, int] = {}
        self._entry_seq = 0

    # ------------------------------------------------------ 基础查询 --

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def entries_for_session(
        self, session_id: str, *, valid_only: bool = True,
    ) -> tuple[KvDirectoryEntry, ...]:
        """会话全部条目（确定性序：角色 → 层 → token → holder → entry_id）。"""
        entries = [
            entry for entry in self._entries.values()
            if entry.content_key.session_id == session_id
            and (entry.is_valid() or not valid_only)
        ]
        return tuple(sorted(entries, key=self._entry_sort_key))

    @staticmethod
    def _entry_sort_key(entry: KvDirectoryEntry) -> tuple:
        key = entry.content_key
        return (
            ROLE_ORDER.get(entry.role, len(ROLE_ORDER)),
            key.layer_start, key.layer_end,
            key.token_start, key.token_end,
            entry.holder, entry.entry_id,
        )

    def entry(self, entry_id: str) -> KvDirectoryEntry:
        try:
            return self._entries[entry_id]
        except KeyError:
            raise KvDirectoryError(
                f"unknown directory entry {entry_id!r}") from None

    def read_refs(self, entry_id: str) -> int:
        return self._read_refs.get(entry_id, 0)

    def is_exec_pinned(self, entry_id: str) -> bool:
        return self._exec_pins.get(entry_id, 0) > 0

    # ------------------------------------------------------ 目录事务 --

    def _bump(self) -> None:
        self.version += 1

    def _validate_shards(self, bytes_by_tp_rank: Sequence[int]) -> tuple[int, ...]:
        shards = tuple(int(value) for value in bytes_by_tp_rank)
        if len(shards) != self.tp_degree:
            raise KvDirectoryError(
                f"entry byte vector must have exactly tp_degree="
                f"{self.tp_degree} ranks (got {len(shards)})")
        if any(value < 0 for value in shards):
            raise KvDirectoryError("entry bytes must be non-negative")
        return shards

    @staticmethod
    def _validate_pending_acks(
        pending_acks: Sequence[str],
    ) -> tuple[str, ...]:
        """Normalize an obligation list without allowing silent duplication."""
        normalized = tuple(str(ack_id) for ack_id in pending_acks)
        if len(set(normalized)) != len(normalized):
            raise KvDirectoryError(
                "an entry cannot carry duplicate pending ack obligations")
        if any(not ack_id for ack_id in normalized):
            raise KvDirectoryError(
                "pending ack obligations must have non-empty identities")
        return normalized

    def _assert_no_duplicate(
        self, session_id: str, holder: int, key: KvContentKey,
        *, exclude_entry_id: Optional[str] = None,
    ) -> None:
        for entry_id in self._by_holder.get((session_id, holder), ()):  # noqa: B007
            if entry_id == exclude_entry_id:
                continue
            other = self._entries[entry_id]
            if other.is_valid() and other.content_key.covers_same_content(key):
                raise KvDirectoryError(
                    f"duplicate physical replica for session {session_id} "
                    f"at holder {holder}: content {key.token_range}×"
                    f"{key.layer_range} already held by entry "
                    f"{entry_id} (state {other.state})")

    def register_entry(
        self,
        *,
        session_id: str,
        holder: int,
        role: str,
        layer_start: int,
        layer_end: int,
        token_start: int,
        token_end: int,
        bytes_by_tp_rank: Sequence[int],
        padding_by_tp_rank: Sequence[int] = (),
        state: str = ENTRY_STATE_WRITING,
        pending_acks: Sequence[str] = (),
        release_condition: str = "",
        generation: int = 0,
    ) -> KvDirectoryEntry:
        """登记物理条目（同 (内容, 持有者) 重复登记 fail-closed）。"""
        if role not in ROLE_ORDER:
            raise KvDirectoryError(f"unknown directory entry role {role!r}")
        if state not in (
            ENTRY_STATE_WRITING, ENTRY_STATE_READY,
            ENTRY_STATE_EVICTED_TO_POOL,
        ):
            raise KvDirectoryError(
                f"register_entry requires WRITING/READY/EVICTED_TO_POOL "
                f"initial state (got {state!r}; RESTORING 只经 "
                "restore_from_pool 发射)")
        if layer_start < 0 or layer_end <= layer_start:
            raise KvDirectoryError("layer range must satisfy 0 <= start < end")
        if token_start < 0 or token_end < token_start:
            raise KvDirectoryError("token range must satisfy 0 <= start, end >= start")
        pending = self._validate_pending_acks(pending_acks)
        if state == ENTRY_STATE_READY and pending:
            raise KvDirectoryError(
                "a READY entry cannot carry pending ack obligations")
        if state == ENTRY_STATE_EVICTED_TO_POOL:
            if holder != POOL_HOLDER:
                raise KvDirectoryError(
                    "an EVICTED_TO_POOL entry must be held by the pool")
            if role != ROLE_POOL_BACKED:
                raise KvDirectoryError(
                    "an EVICTED_TO_POOL entry must use the pool-backed role")
            if pending:
                raise KvDirectoryError(
                    "an EVICTED_TO_POOL entry cannot carry pending acks")
        key = KvContentKey(
            session_id=session_id,
            model_version=self.model_version,
            layer_start=int(layer_start),
            layer_end=int(layer_end),
            token_start=int(token_start),
            token_end=int(token_end),
        )
        self._assert_no_duplicate(session_id, holder, key)
        self._entry_seq += 1
        entry = KvDirectoryEntry(
            entry_id=(
                f"{session_id}#{role}#L{layer_start}-{layer_end}"
                f"#T{token_start}-{token_end}#{self._entry_seq}"),
            content_key=key,
            holder=int(holder),
            role=role,
            bytes_by_tp_rank=self._validate_shards(bytes_by_tp_rank),
            padding_by_tp_rank=(
                self._validate_shards(padding_by_tp_rank)
                if padding_by_tp_rank else ()),
            generation=int(generation),
            state=state,
            pending_acks=pending,
            release_condition=release_condition,
        )
        self._entries[entry.entry_id] = entry
        self._by_holder.setdefault((session_id, holder), set()).add(
            entry.entry_id)
        self._bump()
        return entry

    def _transition(
        self,
        entry_id: str,
        state: str,
        **overrides,
    ) -> KvDirectoryEntry:
        entry = self.entry(entry_id)
        if state not in _ENTRY_TRANSITIONS.get(entry.state, frozenset()):
            raise KvDirectoryError(
                f"illegal directory state transition for {entry_id}: "
                f"{entry.state} → {state}")
        updated = replace(entry, state=state, **overrides)
        pending = self._validate_pending_acks(updated.pending_acks)
        if pending != updated.pending_acks:
            updated = replace(updated, pending_acks=pending)
        removed = set(entry.pending_acks) - set(pending)
        if removed:
            raise KvDirectoryError(
                f"state transition for {entry_id} attempted to drop "
                f"unacknowledged obligations {sorted(removed)}")
        if state == ENTRY_STATE_READY and pending:
            raise KvDirectoryError(
                f"entry {entry_id} cannot become READY with pending ack "
                f"obligations {pending}")
        if state == ENTRY_STATE_EVICTED_TO_POOL:
            if updated.holder != POOL_HOLDER:
                raise KvDirectoryError(
                    f"entry {entry_id} cannot become EVICTED_TO_POOL "
                    "while held by an instance")
            if pending:
                raise KvDirectoryError(
                    f"entry {entry_id} cannot become EVICTED_TO_POOL with "
                    f"pending ack obligations {pending}")
        self._entries[entry_id] = updated
        holder_key = (updated.content_key.session_id, updated.holder)
        if holder_key not in self._by_holder:
            self._by_holder.setdefault(holder_key, set()).add(entry_id)
        self._bump()
        return updated

    # (_drop was removed with the 2026-10-01 dead-code audit: it had no
    # caller anywhere -- retire_entry/evict/restore/merge all reach
    # _release_index directly, and the RETIRING semantics it implied are
    # owned by those explicit transition entry points.)

    def _release_index(self, entry: KvDirectoryEntry) -> None:
        self._entries.pop(entry.entry_id, None)
        self._read_refs.pop(entry.entry_id, None)
        self._exec_pins.pop(entry.entry_id, None)
        holder_ids = self._by_holder.get(
            (entry.content_key.session_id, entry.holder))
        if holder_ids is not None:
            holder_ids.discard(entry.entry_id)
            if not holder_ids:
                self._by_holder.pop(
                    (entry.content_key.session_id, entry.holder), None)

    # ------------------------------------------- ready / ack 义务协议 --

    def add_ack_obligation(self, entry_id: str, ack_id: str) -> None:
        """挂 ready/ack 义务位（journal 协议 tag：copy 块/恢复组身份）。"""
        entry = self.entry(entry_id)
        if entry.state not in (ENTRY_STATE_WRITING, ENTRY_STATE_RESTORING):
            raise KvDirectoryError(
                f"entry {entry_id} in state {entry.state} cannot acquire "
                "a new pending ack obligation")
        ack_id = str(ack_id)
        if not ack_id:
            raise KvDirectoryError("ack obligation identity must be non-empty")
        if ack_id in entry.pending_acks:
            raise KvDirectoryError(
                f"ack obligation {ack_id!r} already registered on "
                f"{entry_id}")
        updated = replace(
            entry, pending_acks=tuple(entry.pending_acks) + (ack_id,))
        self._entries[entry_id] = updated
        self._bump()

    def acknowledge(self, entry_id: str, ack_id: str) -> None:
        """ack 一个义务位（未知/重复 ack fail-closed）。"""
        entry = self.entry(entry_id)
        if ack_id not in entry.pending_acks:
            raise KvDirectoryError(
                f"ack {ack_id!r} is not pending on entry {entry_id} "
                f"(pending: {entry.pending_acks})")
        updated = replace(
            entry,
            pending_acks=tuple(
                item for item in entry.pending_acks if item != ack_id))
        self._entries[entry_id] = updated
        self._bump()

    def mark_ready(self, entry_id: str) -> KvDirectoryEntry:
        """WRITING/RESTORING → READY（ready 义务位必须已闭合）。"""
        entry = self.entry(entry_id)
        if entry.pending_acks:
            raise KvDirectoryError(
                f"entry {entry_id} cannot become READY with pending ack "
                f"obligations {entry.pending_acks}")
        return self._transition(entry_id, ENTRY_STATE_READY)

    # ------------------------------------------------ 区间级引用守恒 --

    def acquire_entry_reference(
        self, entry_id: str, *, exec_pin: bool = False,
    ) -> int:
        """区间级读取引用（+1；执行 pin = 引用的显式标记位）。"""
        self.entry(entry_id)  # 未知条目 fail-closed
        self._read_refs[entry_id] = self._read_refs.get(entry_id, 0) + 1
        if exec_pin:
            self._exec_pins[entry_id] = self._exec_pins.get(entry_id, 0) + 1
        self._bump()
        return self._read_refs[entry_id]

    def release_entry_reference(
        self, entry_id: str, *, exec_pin: bool = False,
    ) -> int:
        """释放读取引用（-1；悬空 release / 越权 unpin fail-closed）。"""
        if entry_id not in self._entries:
            raise KvDirectoryError(
                f"cannot release a reference on unknown entry {entry_id!r}")
        if not exec_pin and self._exec_pins.get(entry_id, 0) > 0:
            # 执行 pin 是引用的组成部分：pin 在持时不得裸放引用
            # （先放引用后放 pin 会造出 pins > refs 的守恒破坏）。
            raise KvDirectoryError(
                f"entry {entry_id} still holds an execution pin; release "
                "the pin (exec_pin=True) before dropping the reference")
        if exec_pin:
            if self._exec_pins.get(entry_id, 0) <= 0:
                raise KvDirectoryError(
                    f"entry {entry_id} has no execution pin to release")
            self._exec_pins[entry_id] -= 1
        current = self._read_refs.get(entry_id, 0)
        if current <= 0:
            raise KvDirectoryError(
                f"unbalanced reference release for entry {entry_id} "
                "(no live read reference)")
        self._read_refs[entry_id] = current - 1
        self._bump()
        return current - 1

    # ------------------------------------------------- 驱逐 / 池恢复 --

    def evict_to_pool(
        self,
        entry_id: str,
        *,
        padding_by_tp_rank: Sequence[int] = (),
    ) -> KvDirectoryEntry:
        """逐出到池：READY → EVICTED_TO_POOL（owner 记录保留，§7.4）。

        引用>0 / 执行 pin 的条目不可驱逐（fail-closed）。逐出后唯一内
        容在池（不占任何实例容量），holder 翻为 POOL_HOLDER、role 换
        ROLE_POOL_BACKED——原角色记入 release_condition 审计披露。
        """
        entry = self.entry(entry_id)
        if entry.holder == POOL_HOLDER:
            raise KvDirectoryError(
                f"entry {entry_id} is already pool-backed")
        if entry.state != ENTRY_STATE_READY:
            raise KvDirectoryError(
                f"entry {entry_id} is not READY and cannot be evicted "
                f"(state={entry.state})")
        if entry.pending_acks:
            raise KvDirectoryError(
                f"entry {entry_id} has pending ack obligations and cannot "
                "be evicted")
        if self._read_refs.get(entry_id, 0) > 0:
            raise KvDirectoryError(
                f"entry {entry_id} has live read references and cannot "
                "be evicted (reference conservation)")
        key = entry.content_key
        self._assert_no_duplicate(
            key.session_id, POOL_HOLDER, key, exclude_entry_id=entry_id)
        self._release_index(entry)
        pooled = KvDirectoryEntry(
            entry_id=entry.entry_id,
            content_key=replace(key, tp_rank=None),
            holder=POOL_HOLDER,
            role=ROLE_POOL_BACKED,
            bytes_by_tp_rank=entry.bytes_by_tp_rank,
            padding_by_tp_rank=(
                self._validate_shards(padding_by_tp_rank)
                if padding_by_tp_rank else entry.padding_by_tp_rank),
            generation=entry.generation + 1,
            state=ENTRY_STATE_EVICTED_TO_POOL,
            pending_acks=(),
            release_condition=(
                f"evicted_from:{entry.holder};role_was:{entry.role};"
                f"{entry.release_condition}"),
        )
        self._entries[pooled.entry_id] = pooled
        self._by_holder.setdefault(
            (key.session_id, POOL_HOLDER), set()).add(pooled.entry_id)
        self._bump()
        return pooled

    def restore_from_pool(
        self, entry_id: str, *, target_instance: int,
    ) -> KvDirectoryEntry:
        """池恢复发射：EVICTED_TO_POOL → RESTORING@target_instance。"""
        entry = self.entry(entry_id)
        if entry.holder != POOL_HOLDER:
            raise KvDirectoryError(
                f"entry {entry_id} is not pool-backed (holder="
                f"{entry.holder})")
        if entry.state != ENTRY_STATE_EVICTED_TO_POOL:
            raise KvDirectoryError(
                f"entry {entry_id} is not in EVICTED_TO_POOL state "
                f"(state={entry.state})")
        if entry.pending_acks:
            raise KvDirectoryError(
                f"pool entry {entry_id} has pending ack obligations")
        if int(target_instance) == POOL_HOLDER:
            raise KvDirectoryError(
                "restore target must be an instance, not the pool holder")
        key = entry.content_key
        self._assert_no_duplicate(
            key.session_id, target_instance, key, exclude_entry_id=entry_id)
        self._release_index(entry)
        restoring = replace(
            entry,
            holder=int(target_instance),
            generation=entry.generation + 1,
            state=ENTRY_STATE_RESTORING,
            release_condition=f"restoring_to:{target_instance}",
        )
        self._entries[restoring.entry_id] = restoring
        self._by_holder.setdefault(
            (key.session_id, target_instance), set()).add(
                restoring.entry_id)
        self._bump()
        return restoring

    # -------------------------------------------- 在途 copy 去重（§6.1）--

    def begin_inflight_copy(
        self,
        *,
        session_id: str,
        source_holder: int,
        target_instance: int,
        layer_start: int,
        layer_end: int,
        token_start: int,
        token_end: int,
        bytes_by_tp_rank: Sequence[int],
        ack_id: str,
    ) -> _InflightCopy:
        """登记在途 copy（同 (来源, 目标, 区间) 在途时不重复登记）。"""
        copy_id = (
            f"{session_id}#inflight#{source_holder}->{target_instance}"
            f"#L{layer_start}-{layer_end}#T{token_start}-{token_end}")
        existing = self._inflight_copies.get(copy_id)
        if existing is not None:
            return existing
        record = _InflightCopy(
            copy_id=copy_id,
            session_id=session_id,
            source_holder=int(source_holder),
            target_instance=int(target_instance),
            layer_start=int(layer_start),
            layer_end=int(layer_end),
            token_start=int(token_start),
            token_end=int(token_end),
            bytes_by_tp_rank=self._validate_shards(bytes_by_tp_rank),
            ack_id=ack_id,
        )
        self._inflight_copies[copy_id] = record
        self._bump()
        return record

    def end_inflight_copy(self, copy_id: str) -> _InflightCopy:
        """在途 copy 闭合（未知 id fail-closed）。"""
        record = self._inflight_copies.pop(copy_id, None)
        if record is None:
            raise KvDirectoryError(f"unknown inflight copy {copy_id!r}")
        self._bump()
        return record

    def inflight_copies(
        self, session_id: Optional[str] = None,
    ) -> tuple[_InflightCopy, ...]:
        records = [
            record for record in self._inflight_copies.values()
            if session_id is None or record.session_id == session_id
        ]
        return tuple(
            sorted(records, key=lambda record: record.copy_id))

    def _intervals_overlap(
        self, a: tuple[int, int], b: tuple[int, int],
    ) -> bool:
        return a[0] < b[1] and b[0] < a[1]

    # ------------------------------------------------------ 目录查询 --

    @staticmethod
    def _rect_difference(
        rect: tuple[tuple[int, int], tuple[int, int]],
        covered: Iterable[tuple[tuple[int, int], tuple[int, int]]],
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """矩形差集：``rect`` 减去全部 ``covered`` 矩形（轴对齐精确切割）。

        返回差集矩形列表（至多每 cover 切出 4 块）；空差集 = ``rect``
        已被覆盖面完全包含（§6.1 不重复搬运/重复读的面）。
        """
        layer_range, token_range = rect
        fragments: list[tuple[tuple[int, int], tuple[int, int]]] = [
            (tuple(layer_range), tuple(token_range))]
        for cover_layer, cover_token in covered:
            next_fragments: list[
                tuple[tuple[int, int], tuple[int, int]]] = []
            for layer, token in fragments:
                l0 = max(layer[0], cover_layer[0])
                l1 = min(layer[1], cover_layer[1])
                t0 = max(token[0], cover_token[0])
                t1 = min(token[1], cover_token[1])
                if l0 >= l1 or t0 >= t1:
                    next_fragments.append((layer, token))   # 不相交
                    continue
                if token[0] < t0:                # token 前段（层带全宽）
                    next_fragments.append((layer, (token[0], t0)))
                if t1 < token[1]:                # token 后段（层带全宽）
                    next_fragments.append((layer, (t1, token[1])))
                if layer[0] < l0:                # 交带内层维前段
                    next_fragments.append(((layer[0], l0), (t0, t1)))
                if l1 < layer[1]:                # 交带内层维后段
                    next_fragments.append(((l1, layer[1]), (t0, t1)))
            fragments = next_fragments
            if not fragments:
                break
        return fragments

    def _fragment_view(
        self,
        entry: KvDirectoryEntry,
        layer_range: tuple[int, int],
        token_range: tuple[int, int],
    ) -> KvDirectoryEntry:
        """条目的片段投影视图（区间裁剪 + 字节线性比例；§6.1 并集格）。

        本仓 KV 字节严格随 token×layer 面积线性（逐 rank 常数率），片段
        字节 = 条目字节 × 片段面积 / 条目面积——整数精确（不可整除 =
        字节模型破坏，fail-closed）。片段视图不是目录实际条目：区间/
        字节为权威搬运单元的裁剪值，holder/role/entry_id/state 沿用源
        条目（引用/pin 仍锚定真实条目）。
        """
        key = entry.content_key
        if (key.layer_range == tuple(layer_range)
                and key.token_range == tuple(token_range)):
            return entry
        entry_area = (
            (key.layer_end - key.layer_start)
            * (key.token_end - key.token_start))
        frag_area = (
            (int(layer_range[1]) - int(layer_range[0]))
            * (int(token_range[1]) - int(token_range[0])))
        scaled: list[int] = []
        for value in entry.bytes_by_tp_rank:
            numer = int(value) * frag_area
            if entry_area <= 0 or numer % entry_area:
                raise KvDirectoryError(
                    f"fragment byte scaling is not exact for entry "
                    f"{entry.entry_id}: {value}*{frag_area}/{entry_area} "
                    "(KV bytes must be integral-linear in token×layer "
                    "area)")
            scaled.append(numer // entry_area)
        return replace(
            entry,
            content_key=replace(
                key,
                layer_start=int(layer_range[0]),
                layer_end=int(layer_range[1]),
                token_start=int(token_range[0]),
                token_end=int(token_range[1])),
            bytes_by_tp_rank=tuple(scaled),
            padding_by_tp_rank=(),
        )

    def _authoritative_fragments(
        self,
        candidates: Sequence[KvDirectoryEntry],
        covered: list[tuple[tuple[int, int], tuple[int, int]]],
    ) -> list[KvDirectoryEntry]:
        """候选条目按冻结序的**单元格并集去重**（§6.1 权威选读贪心）。

        逐候选取其对已覆盖面 ``covered`` 的差集：差集空 = 同内容已由
        冻结序更早的权威副本/目标已持/在途覆盖（不重复搬运/重复读）；
        差集非空 = 生成片段视图条目（部分重叠只补缺失格）并把候选整条
        矩形并入已覆盖面。``covered`` 就地扩展；调用方须先按冻结确定
        序（``_entry_sort_key``）排列候选。
        """
        fragments: list[KvDirectoryEntry] = []
        for entry in candidates:
            key = entry.content_key
            diff = self._rect_difference(
                (key.layer_range, key.token_range), covered)
            if not diff:
                continue
            for layer_range, token_range in diff:
                fragments.append(self._fragment_view(
                    entry, layer_range, token_range))
            covered.append((key.layer_range, key.token_range))
        return fragments

    def _target_covered_rectangles(
        self, session_id: str, target_instance: int,
    ) -> list[tuple[tuple[int, int], tuple[int, int]]]:
        """目标已持/已在途的覆盖矩形（§6.1 不重复搬运/重复读的面）。

        目标自身在途条目（READY/WRITING/RESTORING@target）+ 指向目标
        的在途 copy 记录——两者都是"目标已持有或已在途"的内容。
        """
        covered: list[tuple[tuple[int, int], tuple[int, int]]] = []
        for entry in self.entries_for_session(session_id):
            if (entry.holder == target_instance
                    and entry.state in _ON_CHIP_STATES):
                covered.append((
                    entry.content_key.layer_range,
                    entry.content_key.token_range))
        for record in self.inflight_copies(session_id):
            if record.target_instance == target_instance:
                covered.append((
                    (record.layer_start, record.layer_end),
                    (record.token_start, record.token_end)))
        return covered

    def query_missing_intervals(
        self,
        session_id: str,
        target_instance: int,
        *,
        layer_range: Optional[tuple[int, int]] = None,
    ) -> tuple[KvDirectoryEntry, ...]:
        """目标实例缺失的有效内容**并集**（权威搬运片段序列，§6.1）。

        单元格并集去重（V6/W4 口径）：目标已持有或已在途的内容不返回；
        同内容多副本按冻结确定性顺序（``_entry_sort_key``：角色 → 层 →
        token → holder → entry_id）只选读一条权威副本——部分重叠条目
        只补差集片段（初等单元格 token×layer 级，字节按面积线性精确比
        例）。返回片段视图条目（``_fragment_view``：区间/字节为权威搬
        运单元的裁剪值，其余字段沿用源条目——条目直和即缺失并集字节，
        copy 工作字节据此计）。返回序 = token_start 升序（区间消费序）。
        """
        entries = self.entries_for_session(session_id)
        if not entries:
            return ()
        covered = self._target_covered_rectangles(
            session_id, target_instance)
        candidates = []
        for entry in entries:
            if not entry.is_valid() or entry.holder == target_instance:
                continue
            if (layer_range is not None
                    and not self._intervals_overlap(
                        entry.content_key.layer_range, layer_range)):
                continue
            candidates.append(entry)
        # 权威选择序 = 冻结确定性序（§6.1"多个可选副本按冻结的确定性
        # 顺序选读"）；返回序 = token 消费序。
        candidates.sort(key=self._entry_sort_key)
        missing = self._authoritative_fragments(candidates, covered)
        return tuple(sorted(
            missing,
            key=lambda entry: (
                entry.content_key.token_start,
                entry.content_key.layer_start,
                ROLE_ORDER.get(entry.role, len(ROLE_ORDER)),
                entry.holder,
                entry.entry_id)))

    def query_sources(
        self,
        session_id: str,
        target_instance: int,
    ) -> tuple[KvDirectoryEntry, ...]:
        """枚举目标的片上来源**并集**（权威读流片段序列，§6.1/§6.3）。

        单元格并集去重：目标已持有/已在途覆盖的格不进读流；
        同内容多副本按冻结确定性顺序（角色 → 层
        → token → holder → entry_id，§6.1"多个可选副本按冻结的确定性
        顺序选读"）只选读一条权威副本，部分重叠条目只补差集片段。返
        回片段视图条目（``_fragment_view``：区间/字节为权威读流单元的
        裁剪值，entry_id 沿用源条目——来源引用 pin 锚定真实条目）。
        池条目不进片上来源集（池恢复走 ``query_pool_intervals`` 另列，
        §6.2 分列披露）。
        """
        covered = self._target_covered_rectangles(
            session_id, target_instance)
        candidates = [
            entry for entry in self.entries_for_session(session_id)
            if entry.is_valid()
            and entry.holder != POOL_HOLDER
            and entry.holder != target_instance
            and entry.state in (ENTRY_STATE_READY, ENTRY_STATE_WRITING)
            and any(entry.bytes_by_tp_rank)
        ]
        candidates.sort(key=self._entry_sort_key)
        return tuple(sorted(
            self._authoritative_fragments(candidates, covered),
            key=self._entry_sort_key))

    def query_pool_intervals(
        self, session_id: str,
    ) -> tuple[KvDirectoryEntry, ...]:
        """池恢复区间（EVICTED_TO_POOL 条目；确定性序）。"""
        pooled = [
            entry for entry in self.entries_for_session(session_id)
            if entry.is_pool_backed() and entry.is_valid()
        ]
        return tuple(sorted(pooled, key=self._entry_sort_key))

    def query_unique_protected_sources(
        self, session_id: str,
    ) -> tuple[KvDirectoryEntry, ...]:
        """唯一保护来源：某内容区间仅存一条有效条目时该条目受保护
        （不可驱逐至零副本——§6.1"唯一保护来源"）。"""
        by_interval: dict[
            tuple[tuple[int, int], tuple[int, int]], list[KvDirectoryEntry]
        ] = {}
        for entry in self.entries_for_session(session_id):
            key = entry.content_key
            by_interval.setdefault(
                (key.layer_range, key.token_range), []).append(entry)
        unique = [
            entries[0]
            for _interval, entries in sorted(by_interval.items())
            if len(entries) == 1
        ]
        return tuple(unique)

    # ------------------------------------------------ 元数据合并（§7.4）--

    def merge_adjacent_metadata(
        self, entry_id_a: str, entry_id_b: str,
    ) -> KvDirectoryEntry:
        """相邻区间元数据合并（仅元数据；不免费物理搬运）。

        合法性（全部满足，否则 fail-closed）：同会话/同 holder/同角色/
        token 区间恰相邻（a.end == b.start）/层区间一致/物理布局一致
        （padding 同零或同形）/就绪保护状态兼容（state 相同且 ack 均
        闭合或均按序挂接）/引用计数为零（合并期间无在途读取）。
        """
        a = self.entry(entry_id_a)
        b = self.entry(entry_id_b)
        if (a.content_key.session_id != b.content_key.session_id
                or a.holder != b.holder
                or a.role != b.role
                or a.state != b.state):
            raise KvDirectoryError(
                f"metadata merge requires same session/holder/role/state "
                f"({a.entry_id} vs {b.entry_id})")
        if a.content_key.layer_range != b.content_key.layer_range:
            raise KvDirectoryError(
                f"metadata merge requires identical layer ranges "
                f"({a.content_key.layer_range} vs "
                f"{b.content_key.layer_range})")
        if a.content_key.token_end != b.content_key.token_start:
            raise KvDirectoryError(
                f"metadata merge requires adjacent token ranges "
                f"({a.content_key.token_range} vs "
                f"{b.content_key.token_range})")
        if bool(a.padding_by_tp_rank) != bool(b.padding_by_tp_rank):
            raise KvDirectoryError(
                "metadata merge requires compatible physical layout "
                "(padding presence mismatch)")
        if self._read_refs.get(a.entry_id, 0) or self._read_refs.get(
                b.entry_id, 0):
            raise KvDirectoryError(
                "metadata merge requires zero live read references")
        merged = KvDirectoryEntry(
            entry_id=(
                f"{a.content_key.session_id}#{a.role}#L"
                f"{a.content_key.layer_start}-{a.content_key.layer_end}"
                f"#T{a.content_key.token_start}-{b.content_key.token_end}"
                f"#m{self._entry_seq + 1}"),
            content_key=replace(
                a.content_key, token_end=b.content_key.token_end),
            holder=a.holder,
            role=a.role,
            bytes_by_tp_rank=tuple(
                x + y for x, y in zip(
                    a.bytes_by_tp_rank, b.bytes_by_tp_rank)),
            padding_by_tp_rank=(
                tuple(
                    x + y for x, y in zip(
                        a.padding_by_tp_rank, b.padding_by_tp_rank))
                if a.padding_by_tp_rank else ()),
            generation=max(a.generation, b.generation) + 1,
            state=a.state,
            pending_acks=a.pending_acks + b.pending_acks,
            release_condition=(
                f"metadata_merge:{a.entry_id}+{b.entry_id};"
                f"{a.release_condition}"),
        )
        self._release_index(a)
        self._release_index(b)
        self._entry_seq += 1
        self._entries[merged.entry_id] = merged
        self._by_holder.setdefault(
            (merged.content_key.session_id, merged.holder),
            set()).add(merged.entry_id)
        self._bump()
        return merged

    # ---------------------------------------------------- 守恒投影 --

    def capacity_projection(
        self, holder: int,
    ) -> tuple[int, ...]:
        """逐 holder 字节并集去重投影（占容量的物理副本直和）。

        只计实例驻留条目（WRITING/READY/RESTORING）；同 (内容, 持有者)
        唯一性由守恒断言保证——并集去重 = 每条目恰计一次。池条目不占
        实例容量（池容量另列）。
        """
        total = [0] * self.tp_degree
        for entry in self._entries.values():
            if entry.holder != holder or entry.state not in _ON_CHIP_STATES:
                continue
            for rank, value in enumerate(entry.bytes_by_tp_rank):
                total[rank] += value
        return tuple(total)

    def effective_resident_bytes(
        self, session_id: str,
    ) -> tuple[int, ...]:
        """会话有效保留量 V（唯一内容按**单元格并集**只计一次，§15.1/式(21)）。

        每个初等单元格（token×layer）只按冻结确定性序（``_entry_sort_
        key``）最早的覆盖条目计一次（重复副本/跨持有者部分重叠区间均
        不增加 V——与 ``distributed_settlement.cell_coverage_complete``
        同一单元格并集口径，distributed 跨轮留存段的重叠区间不再计两
        遍）；RELEASED 条目不计。
        """
        total = [0] * self.tp_degree
        covered: list[tuple[tuple[int, int], tuple[int, int]]] = []
        candidates = [
            entry for entry in self.entries_for_session(session_id)
            if entry.is_valid() and any(entry.bytes_by_tp_rank)]
        candidates.sort(key=self._entry_sort_key)
        for fragment in self._authoritative_fragments(candidates, covered):
            for rank, value in enumerate(fragment.bytes_by_tp_rank):
                total[rank] += int(value)
        return tuple(total)

    def assert_conservation(self) -> None:
        """守恒不变量全量断言（fail-closed；O(条目数)）。"""
        seen_content_holders: dict[
            tuple[str, int, tuple[int, int], tuple[int, int]], str] = {}
        for entry in self._entries.values():
            key = entry.content_key
            # 不变量 1：同 (内容, 持有者) 唯一。
            index_key = (
                key.session_id, entry.holder,
                key.layer_range, key.token_range)
            duplicate = seen_content_holders.get(index_key)
            if duplicate is not None:
                raise KvDirectoryError(
                    f"conservation failure: entries {duplicate} and "
                    f"{entry.entry_id} both hold session {key.session_id} "
                    f"content {key.token_range}×{key.layer_range} at "
                    f"holder {entry.holder}")
            seen_content_holders[index_key] = entry.entry_id
            if len(entry.bytes_by_tp_rank) != self.tp_degree:
                raise KvDirectoryError(
                    f"entry {entry.entry_id} byte vector width drifted "
                    f"from tp_degree={self.tp_degree}")
            pending = self._validate_pending_acks(entry.pending_acks)
            if pending != entry.pending_acks:
                raise KvDirectoryError(
                    f"entry {entry.entry_id} has non-normalized ack "
                    "obligations")
            if entry.state == ENTRY_STATE_READY and pending:
                raise KvDirectoryError(
                    f"READY entry {entry.entry_id} retains pending ack "
                    f"obligations {pending}")
            if entry.state == ENTRY_STATE_EVICTED_TO_POOL:
                if entry.holder != POOL_HOLDER:
                    raise KvDirectoryError(
                        f"EVICTED_TO_POOL entry {entry.entry_id} is not "
                        "held by the pool")
                if entry.role != ROLE_POOL_BACKED:
                    raise KvDirectoryError(
                        f"EVICTED_TO_POOL entry {entry.entry_id} does not "
                        "use the pool-backed role")
                if pending:
                    raise KvDirectoryError(
                        f"pool entry {entry.entry_id} retains pending ack "
                        f"obligations {pending}")
            # 不变量 4：引用守恒的计数基础（负计数不可达——配对强制）。
            if self._read_refs.get(entry.entry_id, 0) < 0:
                raise KvDirectoryError(
                    f"entry {entry.entry_id} has negative read references")
            if entry.state == ENTRY_STATE_RELEASED:
                raise KvDirectoryError(
                    f"RELEASED tombstone {entry.entry_id} leaked into the "
                    "live entry table")
        for entry_id, refs in self._read_refs.items():
            if refs < 0:
                raise KvDirectoryError(
                    f"entry {entry_id} has negative read references")
            if refs > 0 and entry_id not in self._entries:
                raise KvDirectoryError(
                    f"read references leaked for dropped entry {entry_id}")
        for entry_id, pins in self._exec_pins.items():
            if pins < 0:
                raise KvDirectoryError(
                    f"entry {entry_id} has negative execution pins")
            if pins > 0 and entry_id not in self._entries:
                raise KvDirectoryError(
                    f"execution pins leaked for dropped entry {entry_id}")
            if pins > self._read_refs.get(entry_id, 0):
                raise KvDirectoryError(
                    f"execution pins exceed read references for "
                    f"{entry_id}")
        # 不变量 3（实例侧）：投影对账由持有方 rank 台账执行（见
        # KVCacheManager._kv_directory_audit——目录不自带第二套容量真值）。

    # ------------------------------------------------ 镜像同步原语 --

    def retire_entry(self, entry_id: str, *, reason: str) -> None:
        """显式释放条目（终局：内容被其他权威副本取代/会话终态清理）。

        引用>0 fail-closed；释放 = 索引移除（RELEASED 墓碑不进活跃表，
        守恒断言保证无泄漏）。
        """
        entry = self.entry(entry_id)
        if self._read_refs.get(entry_id, 0) > 0:
            raise KvDirectoryError(
                f"cannot retire entry {entry_id} with live read "
                f"references ({reason})")
        if self._exec_pins.get(entry_id, 0) > 0:
            raise KvDirectoryError(
                f"cannot retire entry {entry_id} with a live execution pin "
                f"({reason})")
        if entry.pending_acks:
            raise KvDirectoryError(
                f"cannot retire entry {entry_id} with pending ack "
                f"obligations {entry.pending_acks} ({reason})")
        self._release_index(entry)
        self._bump()

    def sync_entries(
        self,
        session_id: str,
        target_entries: Iterable[KvDirectoryEntry],
        *,
        acknowledged_acks: Iterable[str] = (),
        allow_restore_issue: bool = False,
    ) -> None:
        """镜像同步（W4 适配层专用）：把会话条目集合收敛到目标形态。

        目标集合由旧字段重投影给出（同事务更新）；本方法按
        (holder, role, layer_range, token_range) 匹配现存条目并应用差量：
        字节/状态/义务位更新优先原位改写（保 entry_id 稳定，供引用/
        去重持续生效），多余条目显式退休、缺失条目登记。不匹配的非法
        状态迁移 fail-closed（如 READY → WRITING）。
        """
        targets = list(target_entries)
        acknowledged = {
            str(ack_id) for ack_id in acknowledged_acks
        }
        if any(not ack_id for ack_id in acknowledged):
            raise KvDirectoryError(
                "acknowledged obligation identities must be non-empty")
        existing = {
            (entry.holder, entry.role,
             entry.content_key.layer_range,
             entry.content_key.token_range): entry
            for entry in self.entries_for_session(
                session_id, valid_only=False)
            if entry.entry_id in self._entries
        }
        if len(existing) != len(tuple(
                entry for entry in self.entries_for_session(
                    session_id, valid_only=False)
                if entry.entry_id in self._entries)):
            raise KvDirectoryError(
                f"session {session_id} has duplicate mirror match keys")
        live_obligations = {
            ack_id for entry in existing.values()
            for ack_id in entry.pending_acks
        }
        unknown_acknowledgements = acknowledged - live_obligations
        if unknown_acknowledgements:
            raise KvDirectoryError(
                f"mirror_sync acknowledged unknown obligations "
                f"{sorted(unknown_acknowledgements)} for session {session_id}")
        matched: dict[tuple, KvDirectoryEntry] = {}
        for target in targets:
            match_key = (
                target.holder, target.role,
                target.content_key.layer_range,
                target.content_key.token_range)
            if match_key in matched:
                raise KvDirectoryError(
                    f"sync target for session {session_id} contains duplicate "
                    f"match key {match_key}")
            current = existing.get(match_key)
            if current is not None:
                matched[match_key] = current

        target_keys = set(matched)
        retiring = {
            entry.entry_id for match_key, entry in existing.items()
            if match_key not in target_keys
        }
        target_obligations = {
            ack_id for target in targets
            for ack_id in target.pending_acks
        }

        def _validate_target(target: KvDirectoryEntry) -> None:
            key = target.content_key
            if key.session_id != session_id:
                raise KvDirectoryError(
                    f"sync target belongs to session {key.session_id!r}, "
                    f"not {session_id!r}")
            if key.model_version != self.model_version:
                raise KvDirectoryError(
                    f"sync target for session {session_id} has model version "
                    f"{key.model_version!r}, expected {self.model_version!r}")
            if target.role not in ROLE_ORDER:
                raise KvDirectoryError(
                    f"sync target for session {session_id} has unknown role "
                    f"{target.role!r}")
            if target.state not in _VALID_STATES:
                raise KvDirectoryError(
                    f"sync target for session {session_id} has unsupported "
                    f"state {target.state!r}")
            self._validate_shards(target.bytes_by_tp_rank)
            if target.padding_by_tp_rank:
                self._validate_shards(target.padding_by_tp_rank)
            pending = self._validate_pending_acks(target.pending_acks)
            if pending != target.pending_acks:
                raise KvDirectoryError(
                    f"sync target {target.entry_id} has non-normalized "
                    "pending ack obligations")
            if target.state == ENTRY_STATE_READY and pending:
                raise KvDirectoryError(
                    f"sync target {target.entry_id} cannot be READY with "
                    f"pending acks {pending}")
            if target.state == ENTRY_STATE_EVICTED_TO_POOL:
                if target.holder != POOL_HOLDER:
                    raise KvDirectoryError(
                        f"sync target {target.entry_id} marked "
                        "EVICTED_TO_POOL outside the pool")
                if target.role != ROLE_POOL_BACKED:
                    raise KvDirectoryError(
                        f"sync target {target.entry_id} marked "
                        "EVICTED_TO_POOL without pool-backed role")
                if pending:
                    raise KvDirectoryError(
                        f"sync target {target.entry_id} cannot carry pending "
                        "acks after eviction")

        for target in targets:
            if target.state == ENTRY_STATE_RELEASED:
                # (2026-10-01 R2 加固) 退休以**缺席**表达——RELEASED 墓
                # 碑不得进活跃表（旧行为把它写入后由守恒断言以误导性的
                # "tombstone leaked" 拒绝，归因错位）。
                raise KvDirectoryError(
                    f"sync target for session {session_id} carries a "
                    "RELEASED state (retirement is expressed by omitting "
                    f"the entry, not by a tombstone): holder={target.holder} "
                    f"key={target.content_key}")
            _validate_target(target)

        # A mirror target set is one transaction.  Match keys include role so
        # a role change can retire the old entry, but two *new* roles at the
        # same holder/content would still collide when the second entry is
        # registered.  Reject that collision before retiring or registering
        # anything so a failed sync leaves every directory index untouched.
        target_content_holders: dict[
            tuple[int, tuple[int, int], tuple[int, int]], KvDirectoryEntry
        ] = {}
        for target in targets:
            key = target.content_key
            content_holder = (
                int(target.holder), key.layer_range, key.token_range)
            previous = target_content_holders.get(content_holder)
            if previous is not None:
                raise KvDirectoryError(
                    f"sync target for session {session_id} contains duplicate "
                    f"physical holder/content {content_holder} "
                    f"({previous.entry_id} and {target.entry_id})")
            target_content_holders[content_holder] = target

        # Validate every retirement before mutating any one entry.  A mirror
        # update is one directory transaction: a later blocked retirement
        # must not leave an earlier entry silently removed.
        for entry in existing.values():
            if entry.entry_id in retiring:
                if self._read_refs.get(entry.entry_id, 0) > 0:
                    raise KvDirectoryError(
                        f"cannot retire entry {entry.entry_id} with live "
                        "read references during mirror_sync")
                if self._exec_pins.get(entry.entry_id, 0) > 0:
                    raise KvDirectoryError(
                        f"cannot retire entry {entry.entry_id} with a live "
                        "execution pin during mirror_sync")
                if entry.pending_acks:
                    # A changing token/layer projection may replace one
                    # physical mirror rectangle with another while the same
                    # copy chunk is still in flight.  Carry the obligation to
                    # the target entry; only obligations absent from the
                    # entire target set require an explicit acknowledgement.
                    transferred = (
                        set(entry.pending_acks) & target_obligations)
                    unclosed = set(entry.pending_acks) - transferred
                    if not unclosed <= acknowledged:
                        raise KvDirectoryError(
                            f"mirror_sync would drop unacknowledged obligations "
                            f"{sorted(unclosed)} from {entry.entry_id}")

        # For matched entries, only explicitly acknowledged obligations may
        # disappear.  New obligations may be added by the source journal.
        for target in targets:
            match_key = (
                target.holder, target.role,
                target.content_key.layer_range,
                target.content_key.token_range)
            current = existing.get(match_key)
            if current is None:
                if (target.state == ENTRY_STATE_RESTORING
                        and not (
                            allow_restore_issue
                            and target.holder == POOL_HOLDER
                            and target.role == ROLE_POOL_BACKED)):
                    raise KvDirectoryError(
                        f"sync cannot create RESTORING entry {target.entry_id} "
                        "without a prior pool entry")
                continue
            removed = set(current.pending_acks) - set(target.pending_acks)
            if removed and not removed <= acknowledged:
                raise KvDirectoryError(
                    f"mirror_sync would drop unacknowledged obligations "
                    f"{sorted(removed)} from {current.entry_id}")
            if current.state != target.state:
                if target.state not in _ENTRY_TRANSITIONS.get(
                        current.state, frozenset()):
                    raise KvDirectoryError(
                        f"sync cannot transition {current.entry_id} "
                        f"{current.state} → {target.state} (illegal "
                        "directory state machine edge)")
                if target.state == ENTRY_STATE_RESTORING and (
                        current.state != ENTRY_STATE_EVICTED_TO_POOL
                        or current.holder != POOL_HOLDER):
                    raise KvDirectoryError(
                        f"sync can enter RESTORING only from a pool entry "
                        f"({current.entry_id} is {current.state}@"
                        f"{current.holder})")

        # Check target duplicates against entries that will survive.  Role
        # replacement is allowed because its old entry is in ``retiring``.
        for target in targets:
            match_key = (
                target.holder, target.role,
                target.content_key.layer_range,
                target.content_key.token_range)
            current = existing.get(match_key)
            if current is None:
                for entry in existing.values():
                    if entry.entry_id in retiring:
                        continue
                    if (entry.holder == target.holder
                            and entry.content_key.covers_same_content(
                                target.content_key)):
                        raise KvDirectoryError(
                            f"sync target {target.entry_id} duplicates live "
                            f"entry {entry.entry_id} at holder {entry.holder}")

        # 先退休未匹配条目、后登记新条目——同内容同持有者的角色/形态
        # 切换（如 resident → working 同址）不会撞同 (内容, 持有者) 唯
        # 一性断言（旧形态先离场）。所有失败条件均已在上面预检。
        for match_key, entry in existing.items():
            if match_key not in matched:
                self._release_index(entry)
        for target in targets:
            match_key = (
                target.holder, target.role,
                target.content_key.layer_range,
                target.content_key.token_range)
            current = matched.get(match_key)
            if current is None:
                self._assert_no_duplicate(
                    session_id, target.holder, target.content_key)
                self._entry_seq += 1
                registered = replace(
                    target, entry_id=f"{target.entry_id}#{self._entry_seq}")
                self._entries[registered.entry_id] = registered
                self._by_holder.setdefault(
                    (session_id, registered.holder),
                    set()).add(registered.entry_id)
                continue
            updated = replace(
                current,
                bytes_by_tp_rank=target.bytes_by_tp_rank,
                padding_by_tp_rank=target.padding_by_tp_rank,
                generation=max(current.generation, target.generation),
                pending_acks=target.pending_acks,
                release_condition=target.release_condition,
            )
            # (2026-10-01 审查修复) 状态差必须参与外层判据：目标条目与现
            # 存条目仅 state 不同（其余字段逐位相同）时 updated == current，
            # 旧实现在此静默跳过——合法迁移既不应用也不报错（实证：仅
            # state=READY 的目标同步到 WRITING 条目后状态仍 WRITING）。
            if updated != current or current.state != target.state:
                if current.state != target.state:
                    updated = replace(updated, state=target.state)
                self._entries[current.entry_id] = updated
        self._bump()
        self.assert_conservation()

    def forget_session(self, session_id: str, *, reason: str) -> None:
        """会话终局清理（retire_terminal_session 镜像；引用>0 fail-closed）。"""
        for entry in self.entries_for_session(session_id, valid_only=False):
            live = self._entries.get(entry.entry_id)
            if live is not None:
                self.retire_entry(entry.entry_id, reason=reason)
        for record in self.inflight_copies(session_id):
            self.end_inflight_copy(record.copy_id)
        self._bump()
