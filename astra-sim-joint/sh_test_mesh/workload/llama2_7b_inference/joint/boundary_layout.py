"""boundary_layout.py -- D19：D 候选专用边界视图（纯状态/计价接口）。

v2.0 §5.3（区间账本与选读规则）/§6.5（必须能正确表达的 D 边界状态）/
§13.1（两个决策点分别计算哪些工作）；joint 修改
文档 D19。D18 的 ``PDBoundaryView``（joint_cost_model）是边界的**计价
摘要**（布局的逐 rank 字节向量）；本模块在其上细化出 D 候选专用的
**区间级因果视图**：

* ``BoundaryKVLayout``：H_B、按 token/层/holder 的有限来源区间
  （v2.0 §1/§5.3：D 新目标若不在既有两位置，片上位置 ≤3）、有效副本/
  权威归属、首输出 pending token、终止/RNG 状态、继承义务表。派生函数
  回答每个候选目标 j 的口径——**copy 只补 j 缺少部分**（不按 ``H_B``
  全量收费）。
* ``StageWork``：决策点阶段工作（v2.0 §13.1 工作表）——本阶段新执行的
  prefill（D 边界恒 0：只做必要状态准备）、静默重建基数（D recompute =
  H_B，不输出/不采样/不推进 RNG）、**剩余前向步数**（长度入口归一为
  n̂_B，式 4a）、首 token 消费一次契约。
* 等待接口拆分：**原地接续**（in_place_continuation——请求已在 i_P
  在服，不重排队：等待基 = 在服流 running+active，不含新到达排队积压）
  与 **新接纳**（new_admission——跨实例切换是目标实例的新成员，不免费
  插队：等待基 = 目标全部在账负载 queued+running+active）。§19.1 P→D
  专项"原地接续不重排队"；D19"同一已有请求不能被移到新请求队尾来制造
  迁移收益，也不能给异地免费 batch 接入"。

边界纪律（与 joint_cost_model.estimate_D_remaining 同源）：
* 字节真值 = KV 账本逐 rank 向量（D1 裁决），区间只补身份元数据
  （token/层区间、holder、权威位）；本模块不逐 token 建对象（§5.3
  "预测器只遍历区间摘要"）。
* 本模块**不导入** joint_cost_model（避免环导入）；``from_boundary``
  以鸭子类型读取 PDBoundaryView 的字段。跨阶段预约事务/阶段引用原语
  已随 D20 交付；C4（2026-09-27 执行面接通）起边界基材料化 copy
  与 D 静默重建均在运行时可选面内（计价 =
  ``joint_cost_model.estimate_D_remaining``、执行 = KV 账本边界事务
  ）——本模块提供其消费的状态与计价口径。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple


def _require_nonnegative_int(value, name: str) -> int:
    """Validate a serialized count before any lossy ``int`` conversion."""
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _tokens_from_bytes(
        values: Tuple[int, ...], kv_token_bytes: int, name: str) -> int:
    """Validate token-aligned shards and use the widest rank as span.

    TP head sharding can give ranks different byte counts for the same logical
    token interval (uneven heads and replicated/padding ranks).  The byte
    vector remains authoritative; only its token alignment is required.  The
    largest per-rank count is the logical span used for interval metadata.
    """
    token_counts = []
    for index, value in enumerate(values):
        if value % kv_token_bytes:
            raise ValueError(
                f"{name}[{index}] is not an integral token byte count")
        token_counts.append(value // kv_token_bytes)
    return max(token_counts, default=0)


def _tokens_from_rank_bytes(
        values: Tuple[int, ...], bases: Tuple[int, ...], name: str) -> int:
    """反推完整层区间的 token span，使用各 TP rank 自己的字节基准。"""
    if len(values) != len(bases) or not bases:
        raise ValueError(f"{name} and per-rank token bases must share TP width")
    counts = []
    for index, (value, basis) in enumerate(zip(values, bases)):
        if type(basis) is not int or basis <= 0:
            raise ValueError(
                f"kv_token_bytes_by_tp_rank[{index}] must be positive integer")
        if value % basis:
            raise ValueError(
                f"{name}[{index}] is not aligned to its TP token-byte basis")
        counts.append(value // basis)
    if len(set(counts)) != 1:
        raise ValueError(
            f"{name} has inconsistent token spans across TP ranks: {counts}")
    return counts[0]

# ---- 等待接口（D19：原地接续 vs 新接纳）--------------------------------

#: 原地接续：目标 = P 实例自身，请求已在服（P 刚在该实例 drain）——
#: 不重排队。等待基 = 在服流（在飞列车 running + 同批 decode 成员
#: active）；排队中的**新到达** prefill 积压不重压到已有请求头上
#: （§19.1"原地接续不重排队"；decode 成员随下一列车同批行进，不串行
#: 排在新到达之后）。
WAIT_IN_PLACE_CONTINUATION = "in_place_continuation"

#: 新接纳：目标 ≠ P 实例，请求以 decode 成员身份新进入目标实例——
#: 不免费插队。等待基 = 目标全部在账负载（queued + running + active），
#: 目标真实占用（含其排队积压）如实计价；执行侧经 KV 就绪栅栏
#: （pending_decode_ready → 下一列车）进入，受在飞列车 busy 门约束，
#: 不跳过目标既有工作。
WAIT_NEW_ADMISSION = "new_admission"


def boundary_wait_kind(instance_index: int, p_instance_index: int) -> str:
    """候选目标的等待接口类别（单一判据：是否原地）。"""
    return (
        WAIT_IN_PLACE_CONTINUATION
        if instance_index == p_instance_index
        else WAIT_NEW_ADMISSION
    )


def boundary_target_wait_ns(load, wait_kind: str) -> int:
    """按等待接口类别取目标等待基（ns）。

    ``load`` 鸭子类型读取 InstanceLoadView 的三分量字段。原地接续不付
    重排队（排除 queued 新到达积压）；新接纳付目标全部在账负载（无免费
    batch 接入）。未知类别 fail-closed。
    """
    running = _require_nonnegative_int(
        load.running_task_load_ns, "running_task_load_ns")
    active = _require_nonnegative_int(
        load.active_decode_task_load_ns, "active_decode_task_load_ns")
    if wait_kind == WAIT_IN_PLACE_CONTINUATION:
        return running + active
    if wait_kind == WAIT_NEW_ADMISSION:
        queued = _require_nonnegative_int(
            load.queued_task_load_ns, "queued_task_load_ns")
        return running + active + queued
    raise ValueError(f"unknown boundary wait kind: {wait_kind!r}")


# ---- 区间账本（v2.0 §5.3 位置记录的最小形态）---------------------------

#: 完整型布局的单区间角色：H_B 全量工作副本（P ∈ {stay,copy,recompute}
#: 后的合并上下文）。
INTERVAL_ROLE_CONTEXT = "context"
#: 基础区间角色：源端（home）仍保留的基础历史（三位置 merge 投影的
#: base 锚）。
INTERVAL_ROLE_BASE = "base"
# Merge projection appends newly generated decode KV as a distinct interval;
# it shares the same identity validation even though it is not a D-entry role.
INTERVAL_ROLE_GROWTH = "growth"

#: 已承诺传输类义务：区间已在向确定接收端发送（§5.3"已在向 j 传输的
#: 内容不重复发送"——从该接收端的 missing/copy 推导中排除）。
OBLIGATION_COMMITTED_TRANSFER = "committed_transfer"

#: 继承义务表登记的类别（D19 视图项"继承义务表"；生产构造点见
#: sh30_online_scheduler._joint_boundary_stage_view，全部来自真实
#: 账本对象，不做推测性登记）。
OBLIGATION_KINDS = (
    OBLIGATION_COMMITTED_TRANSFER,
    "capacity_reservation",
    "copy_handoff",
    "restore_journal",
)


@dataclass(frozen=True)
class BoundaryKvInterval:
    """边界基础上下文的一个有限来源区间。

    ``bytes_by_tp_rank`` 是账本真值（计价口径）；token/层区间与
    holder/权威位是身份元数据（§5.3 位置记录：区间、holder、owner、
    replica_id——ready/release 事件在账本侧，预测器不建对象）。
    """

    replica_id: str                     # 稳定副本 ID（§5.3 平局判据键）
    role: str                           # context / base / growth
    holder_instance: int                # 当前片上持有位置
    authority: bool                     # 权威归属（有效副本 vs 普通副本）
    token_start: int
    token_end: int
    layer_start: int
    layer_end: int
    bytes_by_tp_rank: Tuple[int, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.replica_id, str) or not self.replica_id.strip():
            raise ValueError("boundary interval replica_id must be non-empty")
        if self.role not in (
                INTERVAL_ROLE_CONTEXT, INTERVAL_ROLE_BASE,
                INTERVAL_ROLE_GROWTH):
            raise ValueError(f"unknown boundary interval role: {self.role!r}")
        _require_nonnegative_int(self.holder_instance, "holder_instance")
        if type(self.authority) is not bool:
            raise ValueError("boundary interval authority must be boolean")
        token_start = _require_nonnegative_int(
            self.token_start, "token_start")
        token_end = _require_nonnegative_int(self.token_end, "token_end")
        if token_end < token_start:
            raise ValueError("boundary interval token_end precedes token_start")
        layer_start = _require_nonnegative_int(
            self.layer_start, "layer_start")
        layer_end = _require_nonnegative_int(self.layer_end, "layer_end")
        if layer_end < layer_start:
            raise ValueError("boundary interval layer_end precedes layer_start")
        if not isinstance(self.bytes_by_tp_rank, tuple) or not self.bytes_by_tp_rank:
            raise ValueError(
                "boundary interval bytes_by_tp_rank must be a non-empty tuple")
        for index, value in enumerate(self.bytes_by_tp_rank):
            _require_nonnegative_int(value, f"bytes_by_tp_rank[{index}]")

    @property
    def tokens(self) -> int:
        return max(0, self.token_end - self.token_start)

    def total_bytes(self) -> int:
        return sum(int(value) for value in self.bytes_by_tp_rank)


@dataclass(frozen=True)
class BoundaryFirstTokenState:
    """首输出 pending token 与采样/RNG 状态（D19 视图项）。

    P 末 chunk 以正常 prefill 采样生成本轮首 token 恰一次（RNG 推进
    一次，D10 阶段语义：P ∈ {stay,copy} 采样、P recompute 重放
    末 chunk 采样——suppress 恒 False）；``issued`` = 已发出待首个 D
    前向消费。**首 token 只消费一次**：``consume_once`` 二次调用
    fail-closed。D 边界静默重建（``rebuild_sampling_suppressed``）不
    重复输出、不重新采样、不推进 RNG（v2.0 §6.2）。
    """

    issued: bool = False
    consumed: bool = False
    rng_advances: int = 0               # 已发生的采样/RNG 推进次数
    rebuild_sampling_suppressed: bool = False

    def __post_init__(self) -> None:
        for name in ("issued", "consumed", "rebuild_sampling_suppressed"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"boundary first-token {name} must be boolean")
        if type(self.rng_advances) is not int:
            raise ValueError("boundary first-token RNG advances must be integer")
        if self.rng_advances not in (0, 1):
            raise ValueError(
                "boundary first-token RNG advances must be 0 or 1, got "
                f"{self.rng_advances}")
        if self.consumed and not self.issued:
            raise ValueError("consumed first token must have been issued")
        if self.consumed and self.rng_advances != 1:
            raise ValueError(
                "consumed first token requires exactly one prior sampling "
                f"advance, got {self.rng_advances}")

    def consume_once(self) -> "BoundaryFirstTokenState":
        """首个 D 前向消费 pending 首 token（恰一次；重复消费
        fail-closed——"正常第一步仍只消费已发首 token 一次"）。"""
        if not self.issued:
            raise ValueError("no pending first token to consume")
        if self.consumed:
            raise ValueError(
                "first token already consumed: the normal first D forward "
                "consumes the issued first token exactly once")
        return replace(self, consumed=True)


@dataclass(frozen=True)
class InheritedObligation:
    """继承义务表条目（§5.3 义务表；已发出未闭合的已提交义务）。

    仅登记真实账本对象（构造点校验存在性）；``receiver_instance`` =
    已承诺接收端（None = 无固定接收端）；``replica_id`` 命中
    ``OBLIGATION_COMMITTED_TRANSFER`` 时从该接收端的缺失推导中排除
    （不重复发送）。
    """

    obligation_id: str
    kind: str
    receiver_instance: Optional[int] = None
    replica_id: Optional[str] = None
    summary: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.obligation_id, str) or not self.obligation_id.strip():
            raise ValueError("inherited obligation id must be non-empty")
        if self.kind not in OBLIGATION_KINDS:
            raise ValueError(
                f"unknown inherited obligation kind: {self.kind!r}")
        if self.receiver_instance is not None:
            _require_nonnegative_int(
                self.receiver_instance, "obligation receiver_instance")
        if self.replica_id is not None and (
                not isinstance(self.replica_id, str)
                or not self.replica_id.strip()):
            raise ValueError("inherited obligation replica_id must be non-empty")
        if not isinstance(self.summary, str):
            raise ValueError("inherited obligation summary must be a string")


@dataclass(frozen=True)
class StageWork:
    """决策点阶段工作（v2.0 §13.1 工作表的接口形态）。

    * P 初始：``prefill_tokens`` = U（recompute 重放 = H+U），D 延续按
      默认方案估——P 入口（estimate_P_with_continuation）口径；
    * D 边界 stay/copy：``prefill_tokens`` = 0（没有新的
      U-token prefill，只做必要状态准备）+ ``remaining_forward_steps``
      = n̂_B 次正常 decode；
    * D 边界 recompute：``rebuild_tokens`` = H_B 静默重建
      （``rebuild_silent``：不向用户输出、不重新采样、不推进 RNG）+
      n̂_B 次正常 decode；已发生 P 是沉没时间，本次重建是新工作。

    ``remaining_forward_steps`` = 长度入口归一（剩余 decode 前向，式
    4a 下界 1；已知终止 0）。不得携带：已完成 P 再收费、未知目标的
    提前传输（本结构只描述本阶段新工作）。
    """

    stage: str                          # "P" / "D"
    prefill_tokens: int                 # 本阶段新执行的 U-token prefill
    rebuild_tokens: int                 # 静默重建基数（D recompute = H_B）
    rebuild_silent: bool                # 重建不输出/不采样/不推进 RNG
    remaining_forward_steps: int        # n̂_B（剩余前向——长度入口归一）
    first_token: BoundaryFirstTokenState = field(
        default_factory=BoundaryFirstTokenState)

    def __post_init__(self) -> None:
        if self.stage not in ("P", "D"):
            raise ValueError(f"stage must be 'P' or 'D', got {self.stage!r}")
        _require_nonnegative_int(self.prefill_tokens, "prefill_tokens")
        _require_nonnegative_int(self.rebuild_tokens, "rebuild_tokens")
        _require_nonnegative_int(
            self.remaining_forward_steps, "remaining_forward_steps")
        if self.rebuild_tokens and not self.rebuild_silent:
            raise ValueError(
                "boundary rebuild work is silent by definition (v2.0 "
                "sec.6.2): no output, no re-sampling, no RNG advance")
        if self.rebuild_silent and not self.rebuild_tokens:
            raise ValueError(
                "rebuild_sampling_suppressed without rebuild work is "
                "inconsistent")
        if self.first_token.rebuild_sampling_suppressed != (
                self.stage == "D" and self.rebuild_silent):
            raise ValueError(
                "first-token suppression flag must mirror the stage work: "
                "silent D rebuild suppresses sampling, P (and non-rebuild "
                "D) does not")


def boundary_stage_work(*, decode_steps: int,
                        first_token: BoundaryFirstTokenState) -> StageWork:
    """D 边界 stay/copy 的工作模板：零新 prefill、零重建、n̂_B
    次剩余前向（§13.1 第二行）。"""
    decode_steps = _require_nonnegative_int(decode_steps, "decode_steps")
    return StageWork(
        stage="D", prefill_tokens=0, rebuild_tokens=0,
        rebuild_silent=False,
        remaining_forward_steps=decode_steps,
        first_token=first_token)


def boundary_recompute_stage_work(
        *, history_tokens: int, decode_steps: int,
        first_token: BoundaryFirstTokenState) -> StageWork:
    """D 边界 recompute 的工作模板：H_B 静默重建（不输出/不采样/不
    推进 RNG）+ n̂_B 次正常前向（§13.1 第四行）；首 token 只消费一次。
    """
    history_tokens = _require_nonnegative_int(
        history_tokens, "history_tokens")
    decode_steps = _require_nonnegative_int(decode_steps, "decode_steps")
    return StageWork(
        stage="D",
        prefill_tokens=0,
        rebuild_tokens=history_tokens,
        rebuild_silent=True,
        remaining_forward_steps=decode_steps,
        first_token=replace(
            first_token, rebuild_sampling_suppressed=True))


# ---- D 候选专用边界布局 --------------------------------------------------

# ---- D19 轮二（2026-09-27）：计价边界视图的运行时状态增补 --------------

class BoundaryRuntimeStateEnvelope:
    """计价边界视图（``PDBoundaryView`` 鸭子类型）的轻包装——把 D19 的
    三项运行时状态与真实层覆盖以**可选属性面**带进 ``from_boundary``。

    为什么是包装而不是改视图：``PDBoundaryView`` 是 frozen dataclass
    （joint_cost_model，本修复轮的共享文件——不得改动它的字段面）；
    ``from_boundary`` 以 ``getattr(boundary, ..., default)`` 拾取增补项，
    直接构造的计价视图（无这些属性）缺省回退——terminated=False、首
    token 全缺省、义务表空、层覆盖 0/0（旧夹具/旧审计面逐位兼容）。

    透传纪律：未知属性一律委托被包装视图（``__getattr__``），计价/
    选择器读取的 PDBoundaryView 字段面不变；本类只增补
    ``terminated``/``first_token``/``obligations``/``working_layers``/
    ``base_layers`` 五个可选属性并在构造点做存在性/类型断言
    （fail-closed——不接收推测性登记）。"""

    __slots__ = ("_view", "terminated", "first_token", "obligations",
                 "working_layers", "base_layers")

    def __init__(self, boundary, *, terminated: bool = False,
                 first_token: Optional[BoundaryFirstTokenState] = None,
                 obligations: Tuple[InheritedObligation, ...] = (),
                 working_layers: int = 0,
                 base_layers: int = 0) -> None:
        self._view = boundary
        if type(terminated) is not bool:
            raise ValueError("boundary runtime envelope terminated must be boolean")
        self.terminated = terminated
        if first_token is None:
            first_token = BoundaryFirstTokenState()
        if not isinstance(first_token, BoundaryFirstTokenState):
            raise ValueError(
                "boundary runtime envelope first_token must be a "
                f"BoundaryFirstTokenState, got {type(first_token)!r}")
        self.first_token = first_token
        items = tuple(obligations)
        for obligation in items:
            if not isinstance(obligation, InheritedObligation):
                raise ValueError(
                    "boundary runtime envelope obligations must be "
                    f"InheritedObligation instances, got {type(obligation)!r}")
        self.obligations = items
        self.working_layers = _require_nonnegative_int(
            working_layers, "working_layers")
        self.base_layers = _require_nonnegative_int(
            base_layers, "base_layers")

    def __getattr__(self, name: str):
        # 只拦截本类 __slots__ 之外的属性——全部委托被包装的计价视图。
        return getattr(self._view, name)


@dataclass(frozen=True)
class BoundaryKVLayout:
    """D 候选专用边界区间布局（D19 核心视图）。

    从 D18 计价边界视图（``PDBoundaryView``，鸭子类型读取）派生的
    有限来源区间账本：单一完整型 context 区间（H_B 全量在 i_P）。
    片上位置数 ≤3（v2.0 §1/§5.3：home、i_P、D 新目标 j）。

    适用前提（W4/W9 登记，方案附录 A13）：本 ≤3 界只约束单轮 D 边界
    视图（构造 S ⊆ {base 锚, i_P}）；跨轮任意多来源/多持有者的
    权威状态由 `kv_location_directory` 一般目录承载（W9 起无位置
    上限）。distributed 跨轮 >2 持有者会话的逐源 D 边界视图升级为
    登记遗留（W10 EVIDENCE §6.3），不得以本界回退目录。

    派生口径（对任一候选目标 j）：
    * ``copy_plan_for`` / ``copy_bytes_by_rank_for``：j 缺少的区间，
      按权威 holder 逐源补齐（"回 home 仅补 U"、"第三目标补 H+U"）；
    * C4 执行面：copy（逐区间补齐后本地读）在候选动作层——缺失集
      推导即本账本的 ``missing_intervals_for``。
    """

    session_id: str
    home_instance: Optional[int]
    p_instance_index: int
    history_tokens: int                 # H_B（边界基础上下文 token 数）
    kv_token_bytes: int                 # 每 rank 每 token 全层 KV 字节
    intervals: Tuple[BoundaryKvInterval, ...]
    decode_steps: int = 0               # n̂_B（剩余前向——长度入口归一）
    terminated: bool = False            # 终止状态（已知终止不进 D 选点）
    first_token: BoundaryFirstTokenState = field(
        default_factory=BoundaryFirstTokenState)
    obligations: Tuple[InheritedObligation, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id.strip():
            raise ValueError("boundary layout session_id must be non-empty")
        if self.home_instance is not None:
            _require_nonnegative_int(self.home_instance, "home_instance")
        _require_nonnegative_int(self.p_instance_index, "p_instance_index")
        _require_nonnegative_int(self.history_tokens, "history_tokens")
        if type(self.kv_token_bytes) is not int or self.kv_token_bytes <= 0:
            raise ValueError("boundary layout kv_token_bytes must be positive integer")
        if not isinstance(self.intervals, tuple):
            raise ValueError("boundary layout intervals must be a tuple")
        seen_replica_ids = set()
        for interval in self.intervals:
            if not isinstance(interval, BoundaryKvInterval):
                raise ValueError(
                    "boundary layout intervals must contain BoundaryKvInterval")
            if interval.replica_id in seen_replica_ids:
                raise ValueError(
                    f"duplicate boundary interval replica_id {interval.replica_id!r}")
            seen_replica_ids.add(interval.replica_id)
        tp_widths = {len(interval.bytes_by_tp_rank)
                     for interval in self.intervals}
        if len(tp_widths) > 1:
            raise ValueError(
                "boundary layout intervals must share one TP byte-vector width")
        _require_nonnegative_int(self.decode_steps, "decode_steps")
        if type(self.terminated) is not bool:
            raise ValueError("boundary layout terminated must be boolean")
        if not isinstance(self.first_token, BoundaryFirstTokenState):
            raise ValueError(
                "boundary layout first_token must be BoundaryFirstTokenState")
        if not isinstance(self.obligations, tuple):
            raise ValueError("boundary layout obligations must be a tuple")
        for obligation in self.obligations:
            if not isinstance(obligation, InheritedObligation):
                raise ValueError(
                    "boundary layout obligations must contain "
                    "InheritedObligation")

    # ---- 构造 ----

    @classmethod
    def from_boundary(cls, boundary) -> "BoundaryKVLayout":
        """从 D18 计价边界视图派生（鸭子类型；不导入避免环依赖）。

        单一完整型布局：单区间 [0, H_B) 全层 @ i_P，权威，
        role=context，字节 = working 向量（H_B 全量的账本真值）。
        生产边界优先提供显式矩形；无矩形时使用逐 rank token 字节基准
        反推完整层区间，旧夹具才回退 kv_token_bytes 标量。

        D19 轮二（执行面填充接通）：三项运行时状态与层覆盖经鸭子类型
        可选属性进场——``terminated``（完成/终止判定；已知终止不进
        D 选点，正常 D 路径恒 False）、``first_token``（StageWork 的
        BoundaryFirstTokenState：P 末 chunk 已采样恰一次、待首个 D 前
        向消费）、``obligations``（继承义务表：t_B 在册已提交义务）、
        ``working_layers``（真实层覆盖：完整型 working = 全模型层数）。
        属性缺席（直接构造的 PDBoundaryView/旧夹具）回退：
        terminated=False、首 token 缺省、义务表空、层覆盖 0/0——
        **层字段是披露面，字节向量仍是账本真值（计价口径），缺省 0 =
        未披露、不改任何计价行为**。
        """
        session_id = getattr(boundary, "session_id", None)
        if not isinstance(session_id, str) or not session_id.strip():
            raise ValueError("boundary layout requires a non-empty session_id")
        kv_token_bytes = boundary.kv_token_bytes
        if type(kv_token_bytes) is not int or kv_token_bytes <= 0:
            raise ValueError(
                "boundary layout requires a positive kv_token_bytes basis")
        raw_working = boundary.working_bytes_by_tp_rank
        if not isinstance(raw_working, (tuple, list)) or not raw_working:
            raise ValueError(
                "boundary working_bytes_by_tp_rank must be a non-empty vector")
        working = tuple(
            _require_nonnegative_int(value, f"working_bytes_by_tp_rank[{index}]")
            for index, value in enumerate(raw_working))
        raw_rank_bases = getattr(boundary, "kv_token_bytes_by_tp_rank", None)
        rank_bases = None
        if raw_rank_bases is not None:
            if not isinstance(raw_rank_bases, (tuple, list)):
                raise ValueError(
                    "kv_token_bytes_by_tp_rank must be a vector when supplied")
            rank_bases = tuple(
                _require_nonnegative_int(
                    value, f"kv_token_bytes_by_tp_rank[{index}]")
                for index, value in enumerate(raw_rank_bases))
            if any(value <= 0 for value in rank_bases):
                raise ValueError(
                    "kv_token_bytes_by_tp_rank must contain positive values")
            if len(rank_bases) != len(working):
                raise ValueError(
                    "kv_token_bytes_by_tp_rank must match working TP width")
        working_intervals = getattr(boundary, "working_intervals", ()) or ()
        if not isinstance(working_intervals, (tuple, list)):
            raise ValueError("working_intervals must be a tuple or list")
        working_intervals = tuple(working_intervals)
        for interval in working_intervals:
            if not isinstance(interval, BoundaryKvInterval):
                raise ValueError(
                    "working_intervals must contain BoundaryKvInterval")
            if len(interval.bytes_by_tp_rank) != len(working):
                raise ValueError(
                    "working_intervals must match working TP width")
        if working_intervals:
            interval_totals = tuple(
                sum(interval.bytes_by_tp_rank[index]
                    for interval in working_intervals)
                for index in range(len(working)))
            if interval_totals != working:
                raise ValueError(
                    "working_intervals bytes must conserve working byte vector")
            working_tokens = max(
                (interval.tokens for interval in working_intervals), default=0)
        elif rank_bases is not None:
            working_tokens = _tokens_from_rank_bytes(
                working, rank_bases, "working_bytes_by_tp_rank")
        else:
            working_tokens = _tokens_from_bytes(
                working, kv_token_bytes, "working_bytes_by_tp_rank")
        p_index = _require_nonnegative_int(
            boundary.p_instance_index, "p_instance_index")
        history_tokens = _require_nonnegative_int(
            boundary.history_tokens, "history_tokens")
        raw_home = boundary.home_instance
        if raw_home is not None:
            _require_nonnegative_int(raw_home, "home_instance")
        # D19 轮二：可选属性拾取（getattr 缺省回退；生产侧由
        # BoundaryRuntimeStateEnvelope 携带真实值进场）。
        terminated = getattr(boundary, "terminated", False)
        if type(terminated) is not bool:
            raise ValueError("boundary terminated must be boolean")
        first_token_state = getattr(boundary, "first_token", None)
        if first_token_state is None:
            first_token_state = BoundaryFirstTokenState()
        if not isinstance(first_token_state, BoundaryFirstTokenState):
            raise ValueError(
                "boundary first_token must be a BoundaryFirstTokenState, "
                f"got {type(first_token_state)!r}")
        obligations = tuple(getattr(boundary, "obligations", ()) or ())
        for obligation in obligations:
            if not isinstance(obligation, InheritedObligation):
                raise ValueError(
                    "boundary obligations must be InheritedObligation "
                    f"instances, got {type(obligation)!r}")
        working_layers = _require_nonnegative_int(
            getattr(boundary, "working_layers", 0), "working_layers")
        raw_base = getattr(boundary, "base_bytes_by_tp_rank", ())
        if raw_base is None:
            raw_base = ()
        if not isinstance(raw_base, (tuple, list)):
            raise ValueError("complete boundary base byte vector must be a vector")
        if len(raw_base) not in (0, len(working)):
            raise ValueError(
                "complete boundary base byte vector must be empty or match "
                "the working TP width")
        for index, value in enumerate(raw_base):
            _require_nonnegative_int(
                value, f"base_bytes_by_tp_rank[{index}]")
        if any(raw_base):
            raise ValueError(
                "complete boundary base byte vector must be empty or all zero")
        intervals = working_intervals
        if not intervals and any(value > 0 for value in working):
            intervals = (BoundaryKvInterval(
                replica_id="boundary_context",
                role=INTERVAL_ROLE_CONTEXT,
                holder_instance=p_index,
                authority=True,
                token_start=0,
                token_end=history_tokens,
                layer_start=0,
                # 完整型 = 全模型层数（披露面；缺省 0 = 未披露）。
                layer_end=working_layers,
                bytes_by_tp_rank=working,
            ),)
        return cls(
            session_id=session_id,
            home_instance=(
                None if boundary.home_instance is None
                else boundary.home_instance),
            p_instance_index=p_index,
            history_tokens=history_tokens,
            kv_token_bytes=kv_token_bytes,
            intervals=intervals,
            decode_steps=_require_nonnegative_int(
                boundary.decode_steps, "decode_steps"),
            terminated=terminated,
            first_token=first_token_state,
            obligations=obligations,
        )

    # ---- 布局不变量 ----

    def on_chip_locations(self) -> Tuple[int, ...]:
        """持有基础上下文区间的片上位置（升序去重；v2.0 §1/§5.3 的
        ≤3 界——home、i_P、D 新目标；单轮边界视图适用前提见类注，
        跨轮多来源以目录为准）。"""
        return tuple(sorted({
            interval.holder_instance for interval in self.intervals}))

    def _committed_transfer_replica_ids(self, target: int) -> frozenset:
        """已在向 target 传输的区间 ID（继承义务表消费；§5.3"已在向 j
        传输的内容不重复发送"）。"""
        return frozenset({
            obligation.replica_id
            for obligation in self.obligations
            if obligation.kind == OBLIGATION_COMMITTED_TRANSFER
            and obligation.receiver_instance == int(target)
            and obligation.replica_id is not None
        })

    # ---- 逐目标派生（D copy 只数目标缺少部分）----

    def missing_intervals_for(self, target: int) -> Tuple[BoundaryKvInterval, ...]:
        """target 缺少的基础上下文区间（按权威 holder 排序：token 起点
        升序——确定性消费序，§6.4 多来源按确定 owner 顺序）。

        * target 已持有有效副本的区间不算缺失（本地复用）；
        * 已在向 target 传输的区间不算缺失（不重复发送）。
        """
        target = int(target)
        committed = self._committed_transfer_replica_ids(target)
        missing = []
        for interval in sorted(self.intervals, key=lambda item: (
                item.token_start, item.replica_id)):
            if interval.holder_instance == target and interval.authority:
                continue    # 目标已有有效副本（权威在本目标）——本地复用
            if interval.replica_id in committed:
                continue    # 已在向 target 传输——不重复发送
            missing.append(interval)
        return tuple(missing)

    def copy_plan_for(
            self, target: int) -> Tuple[Tuple[int, BoundaryKvInterval], ...]:
        """D copy 计划：(权威 holder, 区间) 序列——只补 target 缺少
        部分（§6.3"从当前合法片上来源迁入目标尚缺的有效区间"；"回
        home copy 仅补 U"与"第三目标 copy 补 H+U"由此逐源给出）。"""
        return tuple(
            (interval.holder_instance, interval)
            for interval in self.missing_intervals_for(target))

    def copy_bytes_by_rank_for(self, target: int,
                               tp_size: int) -> Tuple[int, ...]:
        """D copy 的逐 rank 字节（缺失区间并集；区间 token/层身份互不
        重叠——完整型单区间，逐 rank 直和去重）。"""
        tp_size = int(tp_size)
        if tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {tp_size}")
        totals = [0] * tp_size
        for _holder, interval in self.copy_plan_for(target):
            for rank, value in enumerate(interval.bytes_by_tp_rank):
                if rank >= tp_size:
                    raise ValueError(
                        "boundary interval byte vector exceeds instance "
                        f"tp size {tp_size}")
                totals[rank] += int(value)
        return tuple(totals)
