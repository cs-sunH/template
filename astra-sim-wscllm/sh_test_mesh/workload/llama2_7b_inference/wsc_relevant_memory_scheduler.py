#!/usr/bin/env python3
"""relevant_distributed 变体的纯策略 KV 分配器（总文档 §2.4 模块行 / 裁决 #18）。

本模块实现《wscllm要补充的选择分析方案总文档.md》为第三变体
``kv_cache_policy = relevant_distributed`` 规定的分布式 KV 存放策略核心：

- 逐 NPU 全局账本（每 rank 一条 ``(capacity, model_weight_shard,
  resident_kv, staging_scratch)``，容量口径 = capacity −
  model_weight_shard_bytes_by_tp_rank[rel_rank]；总文档 §2.2 / 裁决 #1、#3）；
- 准入三条件一次性预分配 ``try_place``（③ 钉 D decode 段 → ② P 整段暂存
  检查 → ① 全序贪心散布 prefill 段；空域档 ValueError / 当前档 None 两档
  背压；总文档 §3.1 / 裁决 #6、#7、#16、#33、#34）；
- 文档优先序全序 ``(tier, distance_to_decode, -min_rank剩余token,
  instance_index)``，tier：D=0 / 本请求 P=1 / 中间 die≥2；D′ 严格 = 选定
  路径实例集合（废 legacy sibling 扩域；裁决 #4、#31、#32——与 legacy 的
  ``(2,1,0,3,4)`` 序不同是有意设计，legacy 测试不动）；
- P 账本拆分：own piece 记 resident、散布出部分记 staging scratch，两者
  精确合覆盖整个 prefill 段、无重复计账（裁决 #17）；
- KV 事件流三类（placement / release / history_pull，``KVCacheEvent``
  16 列结构沿用）与 journal 行经注入的 ``MemoryActionRecorder``（四字段
  schema 零改动；不装 recorder 时零行为，总文档裁决 #23）；
- 增量不变量校验（``SH_STRICT_KV_INVARIANTS`` 同款语义：环境变量开时
  每次变更后追加全量守恒审计）。

本模块是纯库：无调度循环、不 import online/ 任何模块，也不经过
``SessionKVCacheManager``（裁决 #16）。``session_kv_manager.py`` /
``wsc_llm_scheduler.py`` 为只读红线文件，本模块只 import 使用其公式与
数据类，不改其行为。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence

from session_kv_manager import (  # noqa: E402  只读红线：只 import 使用
    KVCacheEvent,
    kv_cache_shard_bytes_for_tokens,
    model_weight_shard_bytes_by_tp_rank,
)
from wsc_llm_scheduler import (  # noqa: E402  只读红线：只 import 使用
    DECODE_ROLE,
    PREFILL_ROLE,
    StaticPdRoute,
    WscLlmTopology,
)


# KVPiece.tier 值域（总文档 §2.2）：decode 段钉 D 本地 / prefill 段留在 P /
# prefill 段散布到 D 剩余或中间 die。
TIER_DECODE_LOCAL = "decode_local"
TIER_PREFILL_STAY = "prefill_stay"
TIER_SCATTER_REMOTE = "scatter_remote"
PIECE_TIERS = frozenset((TIER_DECODE_LOCAL, TIER_PREFILL_STAY, TIER_SCATTER_REMOTE))

# KV 事件流三类（总文档裁决 #23）。
EVENT_PLACEMENT = "placement"
EVENT_RELEASE = "release"
EVENT_HISTORY_PULL = "history_pull"

# journal 行 cause（四字段 schema 零改动；pieces→resident、scratch→reserved，
# 总文档 §3.1 journal 记账条）。释放行沿用同类 cause 的负 delta（scratch）
# 或对称的 relevant_release（resident），使按 cause 聚合在生命周期闭环后
# 净值为零（守恒链可直读）。
JOURNAL_CAUSE_PLACEMENT = "relevant_placement"
JOURNAL_CAUSE_RELEASE = "relevant_release"
JOURNAL_CAUSE_STAGING = "staging_scratch"
JOURNAL_CAUSE_WEIGHT_PRELOAD = "model_weight_preload"


def _strict_kv_invariants_from_environment() -> bool:
    """与 session_kv_manager 同款语义的严格不变量开关（总文档裁决 #23）。"""

    return os.environ.get("SH_STRICT_KV_INVARIANTS", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _require_index(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")


@dataclass(frozen=True)
class KVPiece:
    """一段 KV 的存放位置（总文档 §2.2 verbatim；执行文档 §4.1 契约）。

    半开区间 ``[token_start, token_end)`` 对"拉回历史 + 本轮新 token 合并
    后的连续上下文"编址；逐 piece 字节不入对象，发射/记账时按
    ``kv_cache_shard_bytes_for_tokens(model, piece_tokens, tp)`` 现算。
    ``path`` = 该实例到 D 的静态路径（实例→D 方向，元数据；物理路由仍走
    C++ 维序，裁决 #12——注意 legacy 分配器的 transfer_path 是 D→实例的
    反向约定，本字段按总文档 §2.2 注释"该实例到 D"的字面方向）。
    """

    instance_index: int
    token_start: int
    token_end: int
    tier: str
    distance_to_decode: int
    path: tuple[int, ...]

    def __post_init__(self) -> None:
        _require_index(self.instance_index, "instance_index")
        _require_index(self.token_start, "token_start")
        _require_index(self.token_end, "token_end")
        if self.token_start > self.token_end:
            raise ValueError("KVPiece token range must satisfy start <= end")
        if self.tier not in PIECE_TIERS:
            raise ValueError(f"unsupported KVPiece tier: {self.tier!r}")
        if not self.path or self.path[0] != self.instance_index:
            raise ValueError("KVPiece path must start at its own instance")
        if self.distance_to_decode != len(self.path) - 1:
            raise ValueError("KVPiece distance must match its path hop count")


@dataclass(frozen=True)
class KVPlacement:
    """一次准入冻结的 page table（总文档 §2.2 verbatim；裁决 #8）。

    请求生命周期内不变；``staging_shard_bytes`` 为 P 上将散布至他实例的
    piece 字节（逐 P 相对 rank），prefill drain 时释放。
    """

    request_id: str
    session_id: str
    turn_index: int
    prefill_instance_index: int
    decode_instance_index: int
    static_route: tuple[int, ...]
    total_tokens: int
    pieces: tuple[KVPiece, ...]
    staging_shard_bytes: tuple[int, ...]

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("KVPlacement request_id must be non-empty")
        if not self.session_id:
            raise ValueError("KVPlacement session_id must be non-empty")
        _require_index(self.turn_index, "turn_index")
        _require_index(self.prefill_instance_index, "prefill_instance_index")
        _require_index(self.decode_instance_index, "decode_instance_index")
        _require_index(self.total_tokens, "total_tokens")
        if not self.static_route:
            raise ValueError("KVPlacement static_route must be non-empty")
        for piece in self.pieces:
            # pieces ⊆ 静态路径实例集合（D′ 域不变量，裁决 #32）。
            if piece.instance_index not in self.static_route:
                raise ValueError(
                    "KVPlacement piece outside the static-route D' set: "
                    f"instance {piece.instance_index} not in {self.static_route}"
                )
        for value in self.staging_shard_bytes:
            _require_index(value, "staging_shard_bytes entry")

    @property
    def prefill_context_tokens(self) -> int:
        """decode 段起点 = prefill 上下文长度（无 decode piece 时 = 总 token）。"""

        for piece in self.pieces:
            if piece.tier == TIER_DECODE_LOCAL:
                return piece.token_start
        return self.total_tokens

    def piece_tokens(self, instance_index: int) -> int:
        """指定实例上承载的 token 总量（守恒/读边元数据辅助）。"""

        return sum(
            piece.token_end - piece.token_start
            for piece in self.pieces
            if piece.instance_index == instance_index
        )


@dataclass(frozen=True)
class RelevantKvRequest:
    """try_place 的准入请求视图（总文档 §3.1 伪代码的 request）。

    ``prefill_context_tokens``：t>0 时 = 拉回历史 + 本轮新 token（连续
    编址的 prefill 段长度）；``decode_tokens``：decode 段长度；final =
    两者之和（准入按 final 一次性预分配，裁决 #6）。
    """

    request_id: str
    session_id: str
    turn_index: int
    prefill_context_tokens: int
    decode_tokens: int

    def __post_init__(self) -> None:
        if not self.request_id:
            raise ValueError("request_id must be non-empty")
        if not self.session_id:
            raise ValueError("session_id must be non-empty")
        _require_index(self.turn_index, "turn_index")
        _require_index(self.prefill_context_tokens, "prefill_context_tokens")
        _require_index(self.decode_tokens, "decode_tokens")

    @property
    def final_context_tokens(self) -> int:
        return self.prefill_context_tokens + self.decode_tokens


@dataclass(frozen=True)
class HistoryPullSource:
    """turn>0 历史拉回的单源聚合记录（总文档 §3.4；1000 族每源一条边）。

    ``shard_bytes`` 为该源全部 token 段的逐相对 rank 字节之和（同实例可
    能持有多段——如 D 同时持有 prefill 散布段与 decode 段，中间 token 在
    他实例——按源聚合成一条 1000 边，``[token_start, token_end)`` 为首末
    段端点、中间可有空洞）；``local_hit`` 表示源 == 新 P（零边，LOCAL_HIT
    语义）。
    """

    source_instance_index: int
    token_start: int
    token_end: int
    distance_to_decode: int
    path: tuple[int, ...]
    shard_bytes: tuple[int, ...]
    local_hit: bool


@dataclass(frozen=True)
class RankKvLedgerSnapshot:
    """逐 NPU 账本行的只读快照。"""

    rank: int
    instance_index: int
    capacity_bytes: int
    model_weight_bytes: int
    resident_kv_bytes: int
    staging_scratch_bytes: int

    @property
    def kv_free_bytes(self) -> int:
        """容量口径 = capacity − model_weight − resident − staging（§2.2）。"""

        return (
            self.capacity_bytes
            - self.model_weight_bytes
            - self.resident_kv_bytes
            - self.staging_scratch_bytes
        )


@dataclass
class _RankKvLedger:
    """逐 NPU 账本行（可变内部状态；快照对外只读）。"""

    rank: int
    instance_index: int
    capacity_bytes: int
    model_weight_bytes: int
    resident_kv_bytes: int = 0
    staging_scratch_bytes: int = 0

    @property
    def available_bytes(self) -> int:
        return (
            self.capacity_bytes
            - self.model_weight_bytes
            - self.resident_kv_bytes
            - self.staging_scratch_bytes
        )

    def snapshot(self) -> RankKvLedgerSnapshot:
        return RankKvLedgerSnapshot(
            rank=self.rank,
            instance_index=self.instance_index,
            capacity_bytes=self.capacity_bytes,
            model_weight_bytes=self.model_weight_bytes,
            resident_kv_bytes=self.resident_kv_bytes,
            staging_scratch_bytes=self.staging_scratch_bytes,
        )


class WscDistributedKvAllocator:
    """relevant_distributed 的逐 NPU 账本 + 三条件准入分配器（纯库）。

    与 legacy ``WscRelevantKvAllocator``（实例级账本、sibling 扩域、旧优先
    序）的差异有总文档裁决 #4/#26 背书，legacy 类与其测试保持原样。

    用法（B3 调度器粘合；本类无调度循环）::

        allocator = WscDistributedKvAllocator(topology, model)
        placement = allocator.try_place(request, route)     # None → FCFS 阻塞
        ...
        allocator.release_staging(placement)                # prefill drain
        allocator.release(placement)                        # 终轮/下一轮到达
    """

    def __init__(
        self,
        topology: WscLlmTopology,
        model: Any,
        *,
        rank_kv_free_bytes: Optional[Mapping[int, int]] = None,
        strict_invariants: Optional[bool] = None,
        recorder: Any = None,
    ) -> None:
        self.topology = topology
        self.model = model
        if not topology.instances:
            raise ValueError("at least one WSC-LLM instance is required")
        if len({instance.size for instance in topology.instances}) != 1:
            raise ValueError(
                "relevant_distributed per-rank ledger requires equal-size instances"
            )
        self.tp_degree = topology.instances[0].size

        # 每 token 逐相对 rank 字节（整头分片线性于 token，可整除折算剩余
        # token 数；公式来自只读红线的 session_kv_manager）。
        self._bytes_per_token_by_relative_rank = kv_cache_shard_bytes_for_tokens(
            model, 1, self.tp_degree
        )

        self._rank_ledgers: dict[int, _RankKvLedger] = {}
        for instance in topology.instances:
            for relative_rank, rank in enumerate(instance.ranks):
                if rank in self._rank_ledgers:
                    raise ValueError(f"rank {rank} belongs to multiple instances")
                if rank_kv_free_bytes is None:
                    capacity = topology.hardware.local_hbm_capacity_bytes
                    weight = model_weight_shard_bytes_by_tp_rank(
                        model, self.tp_degree
                    )[relative_rank]
                else:
                    # 逐 rank 折算构造账本（总文档 §7 算例 B 的 fixture 口
                    # 径）：覆盖值即净空容量，权重不入账本（weight=0）。
                    if rank not in rank_kv_free_bytes:
                        raise ValueError(
                            f"rank_kv_free_bytes must cover rank {rank}"
                        )
                    capacity = rank_kv_free_bytes[rank]
                    _require_index(capacity, f"rank_kv_free_bytes[{rank}]")
                    weight = 0
                if capacity - weight < 0:
                    raise ValueError(
                        f"model does not fit on rank {rank}: needs {weight}, "
                        f"has {capacity}"
                    )
                self._rank_ledgers[rank] = _RankKvLedger(
                    rank=rank,
                    instance_index=instance.index,
                    capacity_bytes=capacity,
                    model_weight_bytes=weight,
                )
        if rank_kv_free_bytes is not None:
            extra = set(rank_kv_free_bytes) - set(self._rank_ledgers)
            if extra:
                raise ValueError(
                    f"rank_kv_free_bytes has unknown ranks: {sorted(extra)}"
                )

        # 活跃 placement 登记表：request_id -> (placement, staging 是否已 drain)。
        self._placements: dict[str, tuple[KVPlacement, bool]] = {}
        self._events: list[KVCacheEvent] = []
        self._event_index_next = 0
        # 增量不变量的第二账本（与 _rank_ledgers 经不同代码路径维护，交叉
        # 验证防漂移；语义对齐 session_kv_manager 的 incremental invariants）。
        self._expected_resident_by_rank: dict[int, int] = {
            rank: 0 for rank in self._rank_ledgers
        }
        self._expected_staging_by_rank: dict[int, int] = {
            rank: 0 for rank in self._rank_ledgers
        }
        self._strict_kv_invariants = (
            _strict_kv_invariants_from_environment()
            if strict_invariants is None
            else strict_invariants
        )

        # 注入式 recorder：不装时零行为。装上时本类是该变体唯一的 rank
        # 账本持有者，故权重预载行由本类落 journal（transaction 0、
        # planner_time_ns=0，先于任何事务；run 末守恒门 physical=weight 由
        # 此成立，口径对齐 SessionKVCacheManager 的权重预载行）。
        self._recorder = recorder
        if recorder is not None:
            for rank in sorted(self._rank_ledgers):
                ledger = self._rank_ledgers[rank]
                recorder.initialize_rank(rank, ledger.capacity_bytes)
                if ledger.model_weight_bytes:
                    recorder.record(
                        planner_time_ns=0,
                        anchor_kind="tick_zero",
                        request_id=None,
                        session_id=None,
                        rank=rank,
                        instance_index=ledger.instance_index,
                        allocation_key=f"weight:{rank}",
                        weight_delta_bytes=ledger.model_weight_bytes,
                        cause=JOURNAL_CAUSE_WEIGHT_PRELOAD,
                    )

    # ------------------------------------------------------------ 校验 --

    def _validate_route(self, route: StaticPdRoute) -> None:
        if not route.path or len(set(route.path)) != len(route.path):
            raise ValueError("static P-to-D route must be a non-empty simple path")
        if route.path[0] != route.prefill_instance_index:
            raise ValueError("static route must start at its Prefill instance")
        if route.path[-1] != route.decode_instance_index:
            raise ValueError("static route must end at its Decode instance")
        if (
            self.topology.instance(route.prefill_instance_index).phase_role
            != PREFILL_ROLE
        ):
            raise ValueError("static route source is not a Prefill instance")
        if (
            self.topology.instance(route.decode_instance_index).phase_role
            != DECODE_ROLE
        ):
            raise ValueError("static route destination is not a Decode instance")
        for source, target in zip(route.path, route.path[1:]):
            if target not in self.topology.adjacency[source]:
                raise ValueError(f"invalid static route edge: {source}->{target}")

    def _shards_for_tokens(self, tokens: int) -> tuple[int, ...]:
        return kv_cache_shard_bytes_for_tokens(self.model, tokens, self.tp_degree)

    def _instance_ranks(self, instance_index: int) -> tuple[int, ...]:
        return self.topology.instance(instance_index).ranks

    def _insufficient(
        self,
        instance_index: int,
        required_shards: Sequence[int],
    ) -> tuple[int, ...]:
        """逐 rank 当前余量不足的绝对 rank 列表（对齐 manager._insufficient）。"""

        ranks = self._instance_ranks(instance_index)
        if len(required_shards) != len(ranks):
            raise ValueError("required KV shards must match the target TP degree")
        return tuple(
            rank
            for rank, required in zip(ranks, required_shards)
            if self._rank_ledgers[rank].available_bytes < required
        )

    def _fit_tokens(self, instance_index: int) -> int:
        """实例当前可整装容纳的 token 数 = 逐 rank 余量整除的最小值。

        整头分片下 piece 字节线性于 token，故整除折算精确无近似。
        """

        return min(
            self._rank_ledgers[rank].available_bytes
            // self._bytes_per_token_by_relative_rank[relative_rank]
            for relative_rank, rank in enumerate(self._instance_ranks(instance_index))
        )

    def _ordered_locations(
        self,
        route: StaticPdRoute,
    ) -> list[tuple[int, int, int, tuple[int, ...]]]:
        """D′ 内实例的全序位置表（裁决 #4/#5：每次准入从当前账本现算）。

        返回 ``(instance_index, tier, distance_to_decode, path)``，按比较键
        ``(tier, distance_to_decode, -fit_tokens, instance_index)`` 升序；
        tier：D=0 / 本请求 P=1 / 中间 die=2。D′ 严格 = 路径实例集合（不并
        入 sibling 扩域，裁决 #32）。
        """

        last = len(route.path) - 1
        entries: list[tuple[int, int, int, tuple[int, ...]]] = []
        for position, instance_index in enumerate(route.path):
            if instance_index == route.decode_instance_index:
                tier = 0
            elif instance_index == route.prefill_instance_index:
                tier = 1
            else:
                tier = 2
            entries.append(
                (instance_index, tier, last - position, route.path[position:])
            )
        entries.sort(
            key=lambda entry: (
                entry[1],
                entry[2],
                -self._fit_tokens(entry[0]),
                entry[0],
            )
        )
        return entries

    # ------------------------------------------------------- 准入/释放 --

    def try_place(
        self,
        request: RelevantKvRequest,
        route: StaticPdRoute,
        *,
        now_ns: int = 0,
    ) -> Optional[KVPlacement]:
        """三条件一次性预分配（总文档 §3.1 伪代码逐行实现）。

        返回值三档：
        - ``KVPlacement``：准入成功（账本已扣、journal/事件已落）；
        - ``None``：当前容量不足（③/②当前档）→ 零写入即回滚 → FCFS 队头
          阻塞，待释放事件后 frontier 重查（裁决 #19）；
        - ``ValueError``：空域不可行（③/②/①空域档，对空账本仍不可行）
          → 配置非法（裁决 #16）。

        顺序钉死：③ 钉 D decode 段 → ② P 整段暂存检查 → ① 全序贪心散布
        prefill 段。条件①（空域档与当前档）被②③结构性蕴含——P、D ∈ D′
        且 piece 按 token 可细分，②通过即保证贪心总能以"P 兜底"完成放置；
        ①保留为防御性断言（总文档 §3.1 结构性注记），不构造其单独触发
        用例、回归断言其永不独立触发。
        """

        _require_index(now_ns, "now_ns")
        self._validate_route(route)
        if request.request_id in self._placements:
            raise RuntimeError(
                f"request {request.request_id} already has a live placement; "
                "release it before re-placing"
            )

        decode_tokens = request.decode_tokens
        prefill_tokens = request.prefill_context_tokens
        decode_shards = self._shards_for_tokens(decode_tokens)
        prefill_shards = self._shards_for_tokens(prefill_tokens)
        final_shards = self._shards_for_tokens(request.final_context_tokens)

        # ---- 空域档（对空账本仍不可行 → ValueError，配置非法）----
        self._check_empty_domain(request, route, decode_shards, prefill_shards, final_shards)

        # ---- 当前档（检查先行，任何失败路径零写入 = 无需回滚）----
        # ③ 当前档：decode 段钉 D 逐 rank 检查。
        if decode_tokens and self._insufficient(route.decode_instance_index, decode_shards):
            return None
        # ② 当前档：P 独自暂存整个 prefill 段（own resident 与散布 scratch
        # 共用 P 的同一可用池）逐 rank 检查。
        if prefill_tokens and self._insufficient(route.prefill_instance_index, prefill_shards):
            return None

        with self._journal_transaction():
            placement = self._place_locked(
                request, route, decode_shards, prefill_shards, final_shards, now_ns
            )
        return placement

    def _check_empty_domain(
        self,
        request: RelevantKvRequest,
        route: StaticPdRoute,
        decode_shards: tuple[int, ...],
        prefill_shards: tuple[int, ...],
        final_shards: tuple[int, ...],
    ) -> None:
        """空域三条件（③→②→①顺序；违反 = 配置非法 ValueError）。

        口径全部逐相对 rank（裁决 #1：容量/字节记账逐 NPU）：③ 对 D 单实
        例空账本、② 对 P 单实例空账本、① 对 D′ 全域空账本逐 rank 求和。
        """

        def empty_available(instance_index: int) -> tuple[int, ...]:
            return tuple(
                ledger.capacity_bytes - ledger.model_weight_bytes
                for ledger in (
                    self._rank_ledgers[rank]
                    for rank in self._instance_ranks(instance_index)
                )
            )

        decode_instance = route.decode_instance_index
        prefill_instance = route.prefill_instance_index
        # ③ 空域：decode 段 ≤ D 空账本逐 rank。
        for relative_rank, (available, required) in enumerate(
            zip(empty_available(decode_instance), decode_shards)
        ):
            if available < required:
                raise ValueError(
                    f"relevant_distributed request {request.request_id} decode "
                    f"segment exceeds the empty-domain capacity of Decode "
                    f"instance {decode_instance} on relative rank "
                    f"{relative_rank}: needs {required}, has {available}; the "
                    "configured request can never be admitted on this route"
                )
        # ② 空域：prefill 段 ≤ P 空账本逐 rank。
        for relative_rank, (available, required) in enumerate(
            zip(empty_available(prefill_instance), prefill_shards)
        ):
            if available < required:
                raise ValueError(
                    f"relevant_distributed request {request.request_id} prefill "
                    f"segment exceeds the empty-domain staging capacity of "
                    f"Prefill instance {prefill_instance} on relative rank "
                    f"{relative_rank}: needs {required}, has {available}; the "
                    "configured request can never be admitted on this route"
                )
        # ① 空域：final ≤ D′ 空账本逐 rank 总量（结构性蕴含的防御档，与
        # legacy empty-domain 档同构，防账本漂移）。
        domain_totals = [0] * self.tp_degree
        for instance_index in route.path:
            for relative_rank, available in enumerate(empty_available(instance_index)):
                domain_totals[relative_rank] += available
        for relative_rank, (available, required) in enumerate(
            zip(domain_totals, final_shards)
        ):
            if available < required:
                raise ValueError(
                    f"relevant_distributed request {request.request_id} final "
                    f"context exceeds the empty-domain capacity of the static "
                    f"route {route.path} on relative rank {relative_rank}: "
                    f"needs {required}, has {available}"
                )

    def _place_locked(
        self,
        request: RelevantKvRequest,
        route: StaticPdRoute,
        decode_shards: tuple[int, ...],
        prefill_shards: tuple[int, ...],
        final_shards: tuple[int, ...],
        now_ns: int,
    ) -> KVPlacement:
        """③→②(已检查)→① 贪心落账（仅在检查全过后进入，零回滚路径）。"""

        prefill_instance = route.prefill_instance_index
        decode_instance = route.decode_instance_index
        prefill_tokens = request.prefill_context_tokens
        decode_tokens = request.decode_tokens
        before = self._remaining(prefill_instance)

        pieces: list[KVPiece] = []
        # ③ 钉 D：decode 段 [prefill_ctx, final) 硬钉 D 本地（裁决 #7）。
        if decode_tokens:
            for rank, delta in zip(self._instance_ranks(decode_instance), decode_shards):
                if delta:
                    self._rank_ledgers[rank].resident_kv_bytes += delta
            pieces.append(
                KVPiece(
                    instance_index=decode_instance,
                    token_start=prefill_tokens,
                    token_end=prefill_tokens + decode_tokens,
                    tier=TIER_DECODE_LOCAL,
                    distance_to_decode=0,
                    path=(decode_instance,),
                )
            )

        # ① 全序贪心散布 prefill 段：每轮从当前账本现算最优位置（裁决
        # #5），吸收至该实例逐 rank 余量整除最小值为止；每实例至多一轮。
        #
        # P 的吸收预算 = P 当前余量 − 整个 prefill 段字节（②的暂存义务之
        # 外的裕量，总文档 §7 算例 B：P0=260、段 200 → own piece 恰为 60 B
        # 裕量）：段内每个 token 要么留在 P（own resident）要么散布出去
        # （drain 前仍以 scratch 记在 P 账上），故 P 的总占用恒等于整段字
        # 节，own piece 只动用裕量、不与暂存义务重复计账（裁决 #17）。
        # 裕量算式对贪心全程稳定（scratch_so_far + remaining 随放置相互转
        # 化，差值不变），开场一次折算即可。
        remaining = prefill_tokens
        used: set[int] = set()
        staging = [0] * self.tp_degree
        prefill_ranks = self._instance_ranks(prefill_instance)
        p_budget_tokens = min(
            (
                self._rank_ledgers[rank].available_bytes - prefill_shards[relative_rank]
            )
            // self._bytes_per_token_by_relative_rank[relative_rank]
            for relative_rank, rank in enumerate(prefill_ranks)
        )
        p_entry: Optional[tuple[int, int, tuple[int, ...]]] = None
        while remaining:
            candidates = [
                entry
                for entry in self._ordered_locations(route)
                if entry[0] not in used
            ]
            if not candidates:
                # P 兜底（总文档 §3.1 结构性注记："②通过即保证贪心总能以
                # P 兜底完成放置"）——其他实例吸不完的剩余 token 全部由 P
                # 以 own piece 承接（②已保证 P 容量覆盖整段）。P 已在轮次
                # 中用过则追加第二段 piece（token 不连续，字节精确）。
                if p_entry is None:
                    raise RuntimeError(
                        f"relevant_distributed greedy scatter failed for "
                        f"request {request.request_id} with {remaining} "
                        "prefill tokens left; admission invariant (P "
                        "fallback) was violated"
                    )
                instance_index, distance, path = p_entry
                take = remaining
            else:
                instance_index, _tier, distance, path = candidates[0]
                if instance_index == prefill_instance:
                    p_entry = (instance_index, distance, path)
                    take = min(remaining, max(0, p_budget_tokens))
                else:
                    take = min(remaining, self._fit_tokens(instance_index))
            used.add(instance_index)
            if not take:
                continue
            token_start = prefill_tokens - remaining
            pieces.append(
                KVPiece(
                    instance_index=instance_index,
                    token_start=token_start,
                    token_end=token_start + take,
                    tier=(
                        TIER_PREFILL_STAY
                        if instance_index == prefill_instance
                        else TIER_SCATTER_REMOTE
                    ),
                    distance_to_decode=distance,
                    path=path,
                )
            )
            piece_shards = self._shards_for_tokens(take)
            for rank, delta in zip(self._instance_ranks(instance_index), piece_shards):
                if delta:
                    self._rank_ledgers[rank].resident_kv_bytes += delta
            if instance_index != prefill_instance:
                # 散布出的 piece 在 P 侧同时记 staging scratch（暂存期物理
                # 上仍占 P 的 HBM；own piece resident + scratch 精确合覆盖
                # 整个 prefill 段、无重复计账，裁决 #17）。
                for relative_rank, delta in enumerate(piece_shards):
                    staging[relative_rank] += delta
                    if delta:
                        self._rank_ledgers[
                            prefill_ranks[relative_rank]
                        ].staging_scratch_bytes += delta
            remaining -= take

        placement = KVPlacement(
            request_id=request.request_id,
            session_id=request.session_id,
            turn_index=request.turn_index,
            prefill_instance_index=prefill_instance,
            decode_instance_index=decode_instance,
            static_route=route.path,
            total_tokens=prefill_tokens + decode_tokens,
            pieces=tuple(pieces),
            staging_shard_bytes=tuple(staging),
        )
        self._placements[request.request_id] = (placement, False)
        touched = self._apply_expected(placement, drained=False)

        # journal 行：pieces → resident（cause=relevant_placement，逐 piece
        # rank 行）；scratch → reserved（cause=staging_scratch，P 的 rank 行）。
        if self._recorder is not None:
            for piece in pieces:
                piece_shards = self._shards_for_tokens(
                    piece.token_end - piece.token_start
                )
                for rank, delta in zip(
                    self._instance_ranks(piece.instance_index), piece_shards
                ):
                    if not delta:
                        continue
                    self._recorder.record(
                        planner_time_ns=now_ns,
                        anchor_kind="prefill_start",
                        request_id=request.request_id,
                        session_id=request.session_id,
                        rank=rank,
                        instance_index=piece.instance_index,
                        allocation_key=(
                            f"relevant:{request.request_id}:"
                            f"instance{piece.instance_index}"
                        ),
                        resident_kv_delta_bytes=int(delta),
                        cause=JOURNAL_CAUSE_PLACEMENT,
                    )
            prefill_ranks = self._instance_ranks(prefill_instance)
            for relative_rank, delta in enumerate(staging):
                if not delta:
                    continue
                self._recorder.record(
                    planner_time_ns=now_ns,
                    anchor_kind="prefill_start",
                    request_id=request.request_id,
                    session_id=request.session_id,
                    rank=prefill_ranks[relative_rank],
                    instance_index=prefill_instance,
                    allocation_key=f"relevant:{request.request_id}:staging",
                    reserved_kv_delta_bytes=int(delta),
                    cause=JOURNAL_CAUSE_STAGING,
                )

        # KV 事件（placement 类）：before/after 取 P 的逐 rank 余量（② 是
        # 唯一实质暂存约束所在），口径与 manager 事件一致。
        self._event(
            now_ns=now_ns,
            phase="prefill_admission",
            event_type=EVENT_PLACEMENT,
            reason="admission_three_conditions",
            trigger_request_id=request.request_id,
            session_id=request.session_id,
            source_instance_index=prefill_instance,
            target_instance_index=decode_instance,
            context_tokens=prefill_tokens + decode_tokens,
            total_bytes=sum(final_shards),
            shard_bytes=final_shards,
            instance_index=prefill_instance,
            before=before,
        )
        self._check_invariants_after_mutation(touched)
        return placement

    def release_staging(self, placement: KVPlacement, *, now_ns: int = 0) -> None:
        """prefill drain：释放 P 的 staging scratch（逐 rank 一次减法）。

        own piece 的 resident 不在此释放（准入时已记，终轮 release 才回加）；
        对同一 placement 二次 drain 直接 fail-closed。
        """

        _require_index(now_ns, "now_ns")
        registered = self._placements.get(placement.request_id)
        if registered is None or registered[0] is not placement:
            raise RuntimeError(
                f"unknown or stale placement for request {placement.request_id}"
            )
        if registered[1]:
            raise RuntimeError(
                f"staging scratch for request {placement.request_id} was "
                "already released at prefill drain"
            )
        prefill_ranks = self._instance_ranks(placement.prefill_instance_index)
        before = self._remaining(placement.prefill_instance_index)
        with self._journal_transaction():
            for rank, delta in zip(prefill_ranks, placement.staging_shard_bytes):
                ledger = self._rank_ledgers[rank]
                if ledger.staging_scratch_bytes < delta:
                    raise RuntimeError(
                        f"staging scratch underflow on rank {rank}: "
                        f"{ledger.staging_scratch_bytes} < {delta}"
                    )
                ledger.staging_scratch_bytes -= delta
                self._expected_staging_by_rank[rank] -= delta
            self._placements[placement.request_id] = (placement, True)
            if self._recorder is not None:
                for relative_rank, delta in enumerate(placement.staging_shard_bytes):
                    if not delta:
                        continue
                    self._recorder.record(
                        planner_time_ns=now_ns,
                        anchor_kind="transfer_complete",
                        request_id=placement.request_id,
                        session_id=placement.session_id,
                        rank=prefill_ranks[relative_rank],
                        instance_index=placement.prefill_instance_index,
                        allocation_key=f"relevant:{placement.request_id}:staging",
                        reserved_kv_delta_bytes=-int(delta),
                        cause=JOURNAL_CAUSE_STAGING,
                    )
            self._event(
                now_ns=now_ns,
                phase="prefill_decode",
                event_type=EVENT_RELEASE,
                reason="staging_scratch_drain",
                trigger_request_id=placement.request_id,
                session_id=placement.session_id,
                source_instance_index=placement.prefill_instance_index,
                target_instance_index=placement.prefill_instance_index,
                context_tokens=0,
                total_bytes=sum(placement.staging_shard_bytes),
                shard_bytes=placement.staging_shard_bytes,
                instance_index=placement.prefill_instance_index,
                before=before,
            )
            self._check_invariants_after_mutation(set(prefill_ranks))

    def release(self, placement: KVPlacement, *, now_ns: int = 0) -> None:
        """释放整个 KVPlacement：逐 rank 精确还原到放置前。

        scratch 已在 drain 释放过则不再重复回加；未知/重复释放 fail-closed。
        """

        _require_index(now_ns, "now_ns")
        registered = self._placements.get(placement.request_id)
        if registered is None or registered[0] is not placement:
            raise RuntimeError(
                f"unknown or stale placement for request {placement.request_id}"
            )
        placement, staging_drained = registered
        before = self._remaining(placement.prefill_instance_index)
        with self._journal_transaction():
            touched: set[int] = set()
            for piece in placement.pieces:
                piece_shards = self._shards_for_tokens(
                    piece.token_end - piece.token_start
                )
                for rank, delta in zip(
                    self._instance_ranks(piece.instance_index), piece_shards
                ):
                    ledger = self._rank_ledgers[rank]
                    if ledger.resident_kv_bytes < delta:
                        raise RuntimeError(
                            f"resident KV underflow on rank {rank}: "
                            f"{ledger.resident_kv_bytes} < {delta}"
                        )
                    ledger.resident_kv_bytes -= delta
                    self._expected_resident_by_rank[rank] -= delta
                    if delta:
                        touched.add(rank)
            if not staging_drained:
                prefill_ranks = self._instance_ranks(placement.prefill_instance_index)
                for rank, delta in zip(prefill_ranks, placement.staging_shard_bytes):
                    ledger = self._rank_ledgers[rank]
                    if ledger.staging_scratch_bytes < delta:
                        raise RuntimeError(
                            f"staging scratch underflow on rank {rank}: "
                            f"{ledger.staging_scratch_bytes} < {delta}"
                        )
                    ledger.staging_scratch_bytes -= delta
                    self._expected_staging_by_rank[rank] -= delta
                    if delta:
                        touched.add(rank)
            del self._placements[placement.request_id]

            if self._recorder is not None:
                for piece in placement.pieces:
                    piece_shards = self._shards_for_tokens(
                        piece.token_end - piece.token_start
                    )
                    for rank, delta in zip(
                        self._instance_ranks(piece.instance_index), piece_shards
                    ):
                        if not delta:
                            continue
                        self._recorder.record(
                            planner_time_ns=now_ns,
                            anchor_kind="completion",
                            request_id=placement.request_id,
                            session_id=placement.session_id,
                            rank=rank,
                            instance_index=piece.instance_index,
                            allocation_key=(
                                f"relevant:{placement.request_id}:"
                                f"instance{piece.instance_index}"
                            ),
                            resident_kv_delta_bytes=-int(delta),
                            cause=JOURNAL_CAUSE_RELEASE,
                        )
                if not staging_drained:
                    prefill_ranks = self._instance_ranks(
                        placement.prefill_instance_index
                    )
                    for relative_rank, delta in enumerate(placement.staging_shard_bytes):
                        if not delta:
                            continue
                        self._recorder.record(
                            planner_time_ns=now_ns,
                            anchor_kind="completion",
                            request_id=placement.request_id,
                            session_id=placement.session_id,
                            rank=prefill_ranks[relative_rank],
                            instance_index=placement.prefill_instance_index,
                            allocation_key=f"relevant:{placement.request_id}:staging",
                            reserved_kv_delta_bytes=-int(delta),
                            cause=JOURNAL_CAUSE_STAGING,
                        )

            self._event(
                now_ns=now_ns,
                phase="completion",
                event_type=EVENT_RELEASE,
                reason="placement_retire",
                trigger_request_id=placement.request_id,
                session_id=placement.session_id,
                source_instance_index=placement.prefill_instance_index,
                target_instance_index=placement.decode_instance_index,
                context_tokens=placement.total_tokens,
                total_bytes=sum(self._shards_for_tokens(placement.total_tokens)),
                shard_bytes=self._shards_for_tokens(placement.total_tokens),
                instance_index=placement.prefill_instance_index,
                before=before,
            )
            self._check_invariants_after_mutation(touched)

    # ------------------------------------------------------ 历史拉回 --

    def plan_history_pull(
        self,
        placement: KVPlacement,
        *,
        history_tokens: int,
        target_instance_index: int,
        now_ns: int = 0,
    ) -> tuple[HistoryPullSource, ...]:
        """规划 turn>0 的多源历史拉回（总文档 §3.4；纯规划，零账本变更）。

        对旧 placement 的 pieces 与 ``[0, history_tokens)`` 求交、按源实例
        合并相邻区间，返回逐源聚合记录（源 == 新 P 的部分 local_hit=True、
        零边 = LOCAL_HIT 语义）；总字节 = kv(history_tokens)。源侧 KV 旧账
        已在 release 时回填，不建模清刷流量（总文档 §3.4）。本方法只落
        history_pull 类 KV 事件，不落 journal 行（journal 只记 planner 账
        本变更；1000 族的物理 HBM 计费走发射层 hbm_charge）。
        """

        _require_index(history_tokens, "history_tokens")
        _require_index(target_instance_index, "target_instance_index")
        if history_tokens > placement.total_tokens:
            raise ValueError(
                "history_tokens exceeds the placement's total tokens: "
                f"{history_tokens} > {placement.total_tokens}"
            )
        spans: dict[int, list[tuple[int, int]]] = {}
        metadata: dict[int, KVPiece] = {}
        for piece in placement.pieces:
            end = min(piece.token_end, history_tokens)
            if piece.token_start >= end:
                continue
            spans.setdefault(piece.instance_index, []).append(
                (piece.token_start, end)
            )
            metadata[piece.instance_index] = piece
        sources: list[HistoryPullSource] = []
        for instance_index in sorted(spans):
            ranges = sorted(spans[instance_index])
            token_start, token_end = ranges[0]
            span_tokens = 0
            for start, end in ranges:
                # 同实例多段（如 D 同时持有 prefill 散布段与 decode 段，中
                # 间 token 在他实例）按源聚合为一条 1000 边：字节 = 各段字
                # 节之和（边只关心逐 rank 字节），起止 = 首末段端点。
                token_end = max(token_end, end)
                span_tokens += end - start
            piece = metadata[instance_index]
            shard_bytes = self._shards_for_tokens(span_tokens)
            local_hit = instance_index == target_instance_index
            sources.append(
                HistoryPullSource(
                    source_instance_index=instance_index,
                    token_start=token_start,
                    token_end=token_end,
                    distance_to_decode=piece.distance_to_decode,
                    path=piece.path,
                    shard_bytes=shard_bytes,
                    local_hit=local_hit,
                )
            )
            self._event(
                now_ns=now_ns,
                phase="history",
                event_type=EVENT_HISTORY_PULL,
                reason="turn_repull_local_hit" if local_hit else "turn_repull_remote",
                trigger_request_id=placement.request_id,
                session_id=placement.session_id,
                source_instance_index=instance_index,
                target_instance_index=target_instance_index,
                context_tokens=token_end - token_start,
                total_bytes=sum(shard_bytes),
                shard_bytes=shard_bytes,
                instance_index=placement.prefill_instance_index,
            )
        return tuple(sources)

    # -------------------------------------------------------- 观测 --

    def _remaining(self, instance_index: int) -> tuple[int, ...]:
        return tuple(
            self._rank_ledgers[rank].available_bytes
            for rank in self._instance_ranks(instance_index)
        )

    def available_shard_bytes(self, route: StaticPdRoute) -> tuple[int, ...]:
        """D′ 全域当前可用字节的逐相对 rank 视图（裁决 #1 逐 NPU 口径）。"""

        self._validate_route(route)
        totals = [0] * self.tp_degree
        for instance_index in route.path:
            for relative_rank, available in enumerate(self._remaining(instance_index)):
                totals[relative_rank] += available
        return tuple(totals)

    def available_tokens(self, route: StaticPdRoute) -> int:
        """D′ 全域当前还能整装的 token 上限（① 口径的观测值）。

        min over rank 的 (D′ 逐 rank 可用和 // 每 token 字节)；不含 ②③ 的
        单实例约束，仅作容量观测，不替代 try_place 的判定。
        """

        totals = self.available_shard_bytes(route)
        return min(
            total // per_token
            for total, per_token in zip(
                totals, self._bytes_per_token_by_relative_rank
            )
        )

    def empty_domain_shard_bytes(self, route: StaticPdRoute) -> tuple[int, ...]:
        """D′ 全域空账本可用字节的逐相对 rank 视图（空域档口径）。"""

        self._validate_route(route)
        totals = [0] * self.tp_degree
        for instance_index in route.path:
            for relative_rank, rank in enumerate(self._instance_ranks(instance_index)):
                ledger = self._rank_ledgers[rank]
                totals[relative_rank] += (
                    ledger.capacity_bytes - ledger.model_weight_bytes
                )
        return tuple(totals)

    def rank_ledger_snapshots(
        self, instance_index: Optional[int] = None
    ) -> tuple[RankKvLedgerSnapshot, ...]:
        """逐 NPU 账本快照（全部 rank 按 id 排序，或单实例）。"""

        if instance_index is None:
            ranks = tuple(sorted(self._rank_ledgers))
        else:
            ranks = self._instance_ranks(instance_index)
        return tuple(self._rank_ledgers[rank].snapshot() for rank in ranks)

    @property
    def events(self) -> tuple[KVCacheEvent, ...]:
        """KV 事件流（placement / release / history_pull 三类）。"""

        return tuple(self._events)

    @property
    def active_placement_count(self) -> int:
        return len(self._placements)

    # ------------------------------------------------------ 事件/不变量 --

    def _event(
        self,
        *,
        now_ns: int,
        phase: str,
        event_type: str,
        reason: str,
        trigger_request_id: str,
        session_id: Optional[str] = None,
        source_instance_index: Optional[int] = None,
        target_instance_index: Optional[int] = None,
        context_tokens: int = 0,
        total_bytes: int = 0,
        shard_bytes: Sequence[int] = (),
        instance_index: Optional[int] = None,
        before: Optional[Sequence[int]] = None,
    ) -> None:
        if before is None:
            remaining = (
                self._remaining(instance_index) if instance_index is not None else ()
            )
        else:
            remaining = tuple(int(value) for value in before)
        self._events.append(
            KVCacheEvent(
                event_index=self._event_index_next,
                planner_time_ns=now_ns,
                phase=phase,
                event_type=event_type,
                reason=reason,
                trigger_request_id=trigger_request_id,
                session_id=session_id,
                source_instance_index=source_instance_index,
                target_instance_index=target_instance_index,
                context_tokens=context_tokens,
                total_bytes=total_bytes,
                shard_bytes=tuple(int(value) for value in shard_bytes),
                last_completion_ns=None,
                instance_remaining_before_bytes=remaining,
                instance_remaining_after_bytes=(
                    self._remaining(instance_index)
                    if instance_index is not None
                    else ()
                ),
                insufficient_ranks=(),
            )
        )
        self._event_index_next += 1

    def _apply_expected(self, placement: KVPlacement, *, drained: bool) -> set[int]:
        """放置后更新增量不变量第二账本，返回受影响 rank 集合。"""

        touched: set[int] = set()
        for piece in placement.pieces:
            piece_shards = self._shards_for_tokens(
                piece.token_end - piece.token_start
            )
            for rank, delta in zip(
                self._instance_ranks(piece.instance_index), piece_shards
            ):
                self._expected_resident_by_rank[rank] += delta
                touched.add(rank)
        if not drained:
            prefill_ranks = self._instance_ranks(placement.prefill_instance_index)
            for relative_rank, delta in enumerate(placement.staging_shard_bytes):
                rank = prefill_ranks[relative_rank]
                self._expected_staging_by_rank[rank] += delta
                touched.add(rank)
        return touched

    class _JournalTransaction:
        """journal 模式下的公开 mutation 事务包裹（对齐 manager ③/④ 语义）。

        正常返回 = commit 点，随后做 ④ 逐 rank 对账（allocator 账本 vs
        journal 记账）；异常路径放弃事务登记不対账（避免二次异常掩盖原始
        错误）。非 journal 模式（含未装 recorder）透传、零开销。
        """

        def __init__(self, allocator: "WscDistributedKvAllocator") -> None:
            self._allocator = allocator
            self._active = False

        def __enter__(self) -> "WscDistributedKvAllocator._JournalTransaction":
            recorder = self._allocator._recorder
            if recorder is not None and recorder.journal_enabled:
                recorder.begin_transaction()
                self._active = True
            return self

        def __exit__(self, exc_type, exc, tb) -> bool:
            recorder = self._allocator._recorder
            if not self._active or recorder is None:
                return False
            if exc_type is not None:
                recorder.abort_transaction()
                return False
            _transaction_id, ranks = recorder.finish_transaction()
            self._allocator._journal_reconcile(ranks)
            return False

    def _journal_transaction(self) -> "_JournalTransaction":
        return self._JournalTransaction(self)

    def _journal_reconcile(self, ranks: "set[int] | frozenset[int]") -> None:
        """④ 对账：journal 侧运行终态 vs allocator 账本（守恒链交叉验证）。"""

        recorder = self._recorder
        if recorder is None or not recorder.journal_enabled:
            return
        for rank in sorted(ranks):
            ledger = self._rank_ledgers[rank]
            weight, resident, reserved = recorder.rank_totals(rank)
            if (weight, resident, reserved) != (
                ledger.model_weight_bytes,
                ledger.resident_kv_bytes,
                ledger.staging_scratch_bytes,
            ):
                raise RuntimeError(
                    f"kv delta journal reconciliation mismatch on rank {rank}: "
                    f"journal={(weight, resident, reserved)}, "
                    f"ledger={(ledger.model_weight_bytes, ledger.resident_kv_bytes, ledger.staging_scratch_bytes)}"
                )

    def _check_invariants_after_mutation(self, affected_ranks: "set[int]") -> None:
        """增量不变量：只校验本次变更触及的 rank（对齐 manager 同名语义）。"""

        for rank in sorted(affected_ranks):
            ledger = self._rank_ledgers[rank]
            if ledger.resident_kv_bytes < 0 or ledger.staging_scratch_bytes < 0:
                raise RuntimeError("negative HBM accounting")
            if ledger.available_bytes < 0:
                raise RuntimeError(
                    f"HBM capacity exceeded on rank {rank}: used="
                    f"{ledger.capacity_bytes - ledger.available_bytes}, "
                    f"capacity={ledger.capacity_bytes}"
                )
            if ledger.resident_kv_bytes != self._expected_resident_by_rank[rank]:
                raise RuntimeError(
                    f"placement/rank KV accounting mismatch for rank {rank}: "
                    f"states={ledger.resident_kv_bytes}, placements="
                    f"{self._expected_resident_by_rank[rank]}"
                )
            if ledger.staging_scratch_bytes != self._expected_staging_by_rank[rank]:
                raise RuntimeError(
                    f"placement/rank staging accounting mismatch for rank "
                    f"{rank}: states={ledger.staging_scratch_bytes}, "
                    f"placements={self._expected_staging_by_rank[rank]}"
                )
        if self._strict_kv_invariants:
            self._check_invariants()

    def _check_invariants(self) -> None:
        """全量守恒审计：账本 vs 活跃 placement 登记表重算 + 位置域不变量。"""

        resident: dict[int, int] = {rank: 0 for rank in self._rank_ledgers}
        staging: dict[int, int] = {rank: 0 for rank in self._rank_ledgers}
        for placement, drained in self._placements.values():
            route_set = set(placement.static_route)
            covered_tokens = 0
            for piece in placement.pieces:
                if piece.instance_index not in route_set:
                    raise RuntimeError(
                        f"placement piece outside D' for request "
                        f"{placement.request_id}: instance "
                        f"{piece.instance_index}"
                    )
                covered_tokens += piece.token_end - piece.token_start
                piece_shards = self._shards_for_tokens(
                    piece.token_end - piece.token_start
                )
                for rank, delta in zip(
                    self._instance_ranks(piece.instance_index), piece_shards
                ):
                    resident[rank] += delta
            if covered_tokens != placement.total_tokens:
                raise RuntimeError(
                    f"placement pieces do not cover the context of request "
                    f"{placement.request_id}: {covered_tokens} != "
                    f"{placement.total_tokens}"
                )
            if not drained:
                prefill_ranks = self._instance_ranks(placement.prefill_instance_index)
                for relative_rank, delta in enumerate(placement.staging_shard_bytes):
                    staging[prefill_ranks[relative_rank]] += delta
        for rank in sorted(self._rank_ledgers):
            ledger = self._rank_ledgers[rank]
            if (
                ledger.resident_kv_bytes != resident[rank]
                or ledger.staging_scratch_bytes != staging[rank]
                or ledger.resident_kv_bytes != self._expected_resident_by_rank[rank]
                or ledger.staging_scratch_bytes != self._expected_staging_by_rank[rank]
            ):
                raise RuntimeError(
                    f"placement/rank KV accounting mismatch for rank {rank}: "
                    f"states=({ledger.resident_kv_bytes}, "
                    f"{ledger.staging_scratch_bytes}), placements="
                    f"({resident[rank]}, {staging[rank]})"
                )
            if ledger.available_bytes < 0:
                raise RuntimeError(
                    f"HBM capacity exceeded on rank {rank}: used="
                    f"{ledger.capacity_bytes - ledger.available_bytes}, "
                    f"capacity={ledger.capacity_bytes}"
                )

    def assert_final_state(self) -> None:
        """run 末审计：全部 placement 已释放、账本归零（resident=staging=0）。"""

        if self._placements:
            pending = ", ".join(sorted(self._placements))
            raise RuntimeError(f"placements still active at final state: {pending}")
        self._check_invariants()
        for rank in sorted(self._rank_ledgers):
            ledger = self._rank_ledgers[rank]
            if ledger.resident_kv_bytes or ledger.staging_scratch_bytes:
                raise RuntimeError(
                    f"rank {rank} did not return to a drained ledger: "
                    f"resident={ledger.resident_kv_bytes}, "
                    f"staging={ledger.staging_scratch_bytes}"
                )


__all__ = [
    "EVENT_HISTORY_PULL",
    "EVENT_PLACEMENT",
    "EVENT_RELEASE",
    "HistoryPullSource",
    "KVPiece",
    "KVPlacement",
    "PIECE_TIERS",
    "RankKvLedgerSnapshot",
    "RelevantKvRequest",
    "TIER_DECODE_LOCAL",
    "TIER_PREFILL_STAY",
    "TIER_SCATTER_REMOTE",
    "WscDistributedKvAllocator",
]
