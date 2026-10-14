"""resource_projection.py -- W2：公共资源事件前推内核（2026-09-29）。

设计依据：《joint仓库当下+历史修改方案》§4.1（ProjectionResult 契约）、
§5.1/§5.1a（正式算法与逐资源服务语义）、§5.2–§5.6、§11.2（手算锚）、
§12.1 W2 段；盘点报告 inv_w23.md §3/§6.1/§6.2。

本模块是**独立内核**（正式接线归 W3；W1 桥接前以显式受控 fixture 开发
——方案 W2 段"可在桥接未完成时用显式受控 fixture 实施"）。核心语义：

* **事件队列推进**（§5.1）：维护 ready/active/blocked 工作集合 + 资源仲裁
  状态 + 按模拟时刻排序的事件队列；每步取最近事件（服务完成/工作就绪/
  latency 到期入流集合/门释放/仲裁变化/预测到达占位），对过去一段按
  分段恒速**积分服务量**，同刻事件按 ``(tick, priority, producer,
  local_sequence)`` 确定序处理，再重算受影响资源与依赖闭包的速率，推进
  至焦点工作 settle。
* **逐资源服务规则**（§5.1a 权威，等权 N-way、无全局 max-min 再分配）：
  - NoC 链路：路径速率 = 各链路 ``capacity/active_flow_count`` 的最小值
    （多链路路由 = min over links）；
  - HBM：每 rank 单 aggregate bus、读写共用带宽、N-way 等分、job 有
    latency（latency 后才入流集合）；计算 ops 与 memory 并行，保留
    roofline 较晚完成条件（提炼 ERP ``ComputeLayerSegment`` 的
    ``max(γ·base, port_done)`` 骨架；memory 字节 drained 即离开分母）；
  - 池端口：每端口 latency 后入流集合、N-way 共享、join/completion 改
    份额；
  - TP：barrier 逐层同步（层完成 = 跨 rank max），腿阻塞沿依赖边传播；
  - 发射/通信槽：``slot`` 门为**准入门**（可发射时间/槽位状态/release），
    不是带宽分母。
* **候选反作用**（§5.1 红线）：候选插入后背景流完成时刻随活跃集变化
  重算——绝不先求"无候选背景结束时刻"固定住再插候选；每背景工作输出
  with/without 候选完成时刻对照（``BackgroundComparison``）。
* **committed-not-issued**（锚2）：已提交但依赖/就绪未满足的腿以
  ready_at/deps 入账（timer 事件激活后进入带宽分母），只导出 active 流
  会错估。
* **候选隔离与闭包**（§5.4）：输入为冻结 dataclass，每候选经
  ``ProjectionInput.with_candidate`` 叠加后独立重建世界（ERP
  ``ResourceSnapshot.with_committed`` 的增量副本语义）；候选枚举换序结果
  不变；影响闭包由 native 投影内核在 context 创建时自行计算。
  下游阻塞/buffer/容量释放依赖扩展至固定点。
* **有限缓冲跨段状态**（§5.3）：buffer 取得/填充/消费/释放全部事件化；
  解析段加速算子（``BlockPipelineState`` + 闭式递推）仅在"协议相同、块
  服务时间相同、区间内仲裁/数据/容量门不变"前提下可用，遇流加入/完成/
  到达/反压/容量变化即截断并结转（已服务字节 + 双 buffer 状态 + 已付
  首块启动标记——分段不重复首块启动）。
* **守恒与取整**：积分守恒校验（提炼 ERP ``_assert_conservation`` 语义，
  raise 级 ``ProjectionConservationError``）；完成时刻统一向上取整
  （ceil），小字节传输不为零时长。
* **预算与降级**（§5.6）：事件数/闭包资源数预算**只**从
  ``joint.prediction_config.PredictionParams`` 消费（构造时传入，本模块
  不另设常数）；超限 → 状态 = 降级/模型不可用 + 原因，**绝不**返回截断
  的"已完成"时刻。
* **复杂度披露**：每候选事件数/资源数/工作数/闭包规模经
  ``ProjectionResult`` 与 native 投影内核的 cost disclosure 输出
  （W3 接 predictor_cost_accounting 的挂钩点）。

纯 Python、无 I/O、无环境读取；不 import 交付物（E 专属的 k 枚举/剪枝/
期限钉死不搬——W2 不扩大 E 搜索范围）。
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field, replace
from enum import IntEnum
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from joint.prediction_config import PredictionParams

# ------------------------------------------------------------------ 状态码 --

#: 焦点工作 settle 推进成功（预测时刻可信）。
STATUS_OK = "ok"
#: 合法等待：horizon 到期但事件仍在推进（有进展路径）——显式区别于零进展。
STATUS_WAIT_PROGRESS = "wait_progress"
#: 不可进展：事件队列抽干、焦点未完成（零速率/门死锁）。
STATUS_BLOCKED_NO_PROGRESS = "blocked_no_progress"
#: 降级：前推事件数超 ``projection_event_budget``（绝不返回截断完成时刻）。
STATUS_DEGRADED = "degraded"
#: 模型不可用：闭包资源数超 ``closure_resource_budget``，无法证明候选
#: 影响边界独立（§5.4"不能静默忽略外部流"）。
STATUS_MODEL_UNAVAILABLE = "model_unavailable"

_DEGRADED_EVENT_BUDGET = "event_budget_exceeded"
_MODEL_UNAVAILABLE_CLOSURE = "closure_resource_budget_exceeded"

# 守恒校验容差（相对区间前总量）与 dust 界（完成边界 1 ulp 邻域零头）——
# 沿 ERP 数值纪律（A11'/A14'）。
_CONSERVATION_CHECK = True
_CONSERVATION_TOLERANCE = 1e-6
_DUST = 1e-9


class ResourceProjectionError(ValueError):
    """fail-closed：非法输入（未知资源/依赖环/字段矛盾）或内部不变量破损。"""


class ProjectionConservationError(ResourceProjectionError):
    """积分守恒不变量破损（raise 级，python -O 免疫；参考 ERP 同名语义）。"""


# ------------------------------------------------------------------ 事件序 --

class EventKind:
    """堆事件类型（服务完成由每步边界求值处理——与入堆等价）。"""

    LATENCY_EXPIRE = "latency_expire"        # job latency 到期 → 入流集合
    TASK_READY = "task_ready"                # 已提交未就绪工作 ready_at 到期
    FORECAST_ARRIVAL = "forecast_arrival"    # 预测到达占位（W7 接线钩子）


class EventPriority(IntEnum):
    """同刻处理序（小者先）：先结算完成释放份额，再处理 latency/ready。"""

    COMPLETION = 0
    LATENCY_EXPIRE = 1
    TASK_READY = 2
    FORECAST_ARRIVAL = 3


def _ceil_ns(value: float) -> int:
    """完成时刻统一向上取整；正时长至少 1 ns（小字节传输不为零时长）。"""
    if value <= 0.0:
        return 0
    # Do not subtract a fixed epsilon here.  A real value just above an
    # integer is still a positive amount of elapsed time and must round up;
    # floating-point residue is absorbed when the event is integrated and
    # completed, rather than by changing the public completion boundary.
    return max(1, int(math.ceil(value)))


# ------------------------------------------------------------------ 效率因子 --


@dataclass(frozen=True)
class ProjectionEfficiency:
    """η（传输资源）/ γ（计算组）只读承载（提炼 ERP ``EfficiencyFactors``）。

    键匹配先精确后类前缀（``link``/``hbm``/``pool``），缺省 1.0。非有限或
    ≤0 直接 fail-closed。
    """

    eta: Mapping[str, float] = field(default_factory=dict)
    gamma: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for kind, table in (("eta", self.eta), ("gamma", self.gamma)):
            for key, value in table.items():
                if not math.isfinite(value) or value <= 0:
                    raise ResourceProjectionError(
                        f"{kind} factor for {key!r} must be finite positive")

    def eta_of(self, resource_key: str) -> float:
        if resource_key in self.eta:
            return float(self.eta[resource_key])
        prefix = resource_key.split(":", 1)[0]
        return float(self.eta.get(prefix, 1.0))

    def gamma_of(self, group: str) -> float:
        return float(self.gamma.get(group, 1.0))


# ------------------------------------------------------------------ 输入面 --

#: 资源种类（§5.1a 逐资源服务规则的登记面；仲裁统一为活跃集 N-way 等分）。
RESOURCE_KIND_LINK = "link"      # NoC 链路：路径速率 = 各链路 capacity/n 的 min
RESOURCE_KIND_HBM = "hbm"        # 每 rank 单 aggregate bus，读写共用
RESOURCE_KIND_POOL = "pool"      # 池端口：latency 后入流集合
RESOURCE_KINDS = (RESOURCE_KIND_LINK, RESOURCE_KIND_HBM, RESOURCE_KIND_POOL)

#: 门种类：全部是**准入门**（激活前必须取得），不是带宽分母（§5.1a 槽位行）。
GATE_KIND_BUFFER = "buffer"      # 有限共享缓冲（块计数）
GATE_KIND_CAPACITY = "capacity"  # 容量（单元计数；取得/释放事件化）
# GATE_KIND_SLOT：发射/通信槽（接口先立，fixture 提供状态）。生产图暂不
# 构造 slot/credit 门——准入/发射次序由调度器注入的 roofline 标量
# frontier 近似承担（prep_issue_frontier 披露），槽位/迭代状态与真实
# release 的入图登记为缺口（§5.1a/§5.3；sh30 _prep_frontier_by_instance
# docstring 登记近似边界），非能力声明。
GATE_KIND_SLOT = "slot"
GATE_KINDS = (GATE_KIND_BUFFER, GATE_KIND_CAPACITY, GATE_KIND_SLOT)

#: 腿（任务）类型。
TASK_KIND_STREAM = "stream"    # 字节流：route 多资源、rate = min over route
TASK_KIND_FIXED = "fixed"      # 固定时长（纯 latency 腿/固定服务腿）
TASK_KIND_COMPUTE = "compute"  # roofline：memory 字节流 ∥ γ·base 时长，完成 = max；
                               # die_plan 在场时 = A3 执行协议驱动的计划腿（§9.6）

_WORK_NODE_PREFIX = "work:"


@dataclass(frozen=True)
class ResourceSpec:
    """流资源（链路/HBM 总线/池端口）峰值速率。"""

    key: str
    kind: str
    capacity_per_ns: float

    def __post_init__(self) -> None:
        if not isinstance(self.key, str) or not self.key.strip():
            raise ResourceProjectionError("resource key must be non-empty")
        if self.kind not in RESOURCE_KINDS:
            raise ResourceProjectionError(
                f"resource kind must be one of {RESOURCE_KINDS}, got "
                f"{self.kind!r}")
        expected = {
            RESOURCE_KIND_LINK: "link",
            RESOURCE_KIND_HBM: "hbm",
            RESOURCE_KIND_POOL: "pool",
        }[self.kind]
        if self.key.split(":", 1)[0] != expected:
            raise ResourceProjectionError(
                f"resource key {self.key!r} must use prefix {expected!r} "
                f"for kind {self.kind!r}")
        if not math.isfinite(self.capacity_per_ns) or (
                self.capacity_per_ns <= 0):
            raise ResourceProjectionError(
                f"resource {self.key!r} capacity must be finite positive")


@dataclass(frozen=True)
class GateSpec:
    """准入门（buffer/capacity/slot）：计数信号量，激活前取得、完成时释放。

    ``initial_occupied`` = 快照时刻已被占用的单元数（W1 账本事实：如驻留
    KV 占的容量、在途流持有的缓冲块/槽位）——释放它们必须经某腿的
    ``releases``（已承诺释放事件），不由时间自动归还。
    """

    gate_id: str
    kind: str
    capacity_units: int
    initial_occupied: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.gate_id, str) or not self.gate_id.strip():
            raise ResourceProjectionError("gate id must be non-empty")
        if self.kind not in GATE_KINDS:
            raise ResourceProjectionError(
                f"gate kind must be one of {GATE_KINDS}, got {self.kind!r}")
        if (type(self.capacity_units) is not int
                or self.capacity_units < 1):
            raise ResourceProjectionError(
                f"gate {self.gate_id!r} capacity must be int >= 1")
        if (type(self.initial_occupied) is not int
                or not 0 <= self.initial_occupied <= self.capacity_units):
            raise ResourceProjectionError(
                f"gate {self.gate_id!r} initial_occupied must be int in "
                f"[0, capacity], got {self.initial_occupied!r}")


@dataclass(frozen=True)
class GateLease:
    """一次门取得/释放（units 计数；如双缓冲一块 = 1 unit）。"""

    gate_id: str
    units: int = 1

    def __post_init__(self) -> None:
        if not isinstance(self.gate_id, str) or not self.gate_id.strip():
            raise ResourceProjectionError("gate lease id must be non-empty")
        if type(self.units) is not int or self.units < 1:
            raise ResourceProjectionError(
                f"gate lease on {self.gate_id!r} must be int units >= 1")


@dataclass(frozen=True)
class BarrierSpec:
    """TP 逐层同步 barrier：全部 signaller 腿完成后通过（时刻 = max 完成）。

    依赖边可引用 barrier id：腿阻塞传播——候选推迟某条 TP 腿 → barrier
    推迟 → 其他 rank 的后继腿/容量释放被连带推迟（§5.4 闭包用例）。
    """

    barrier_id: str
    signallers: Tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.barrier_id, str) or not self.barrier_id.strip():
            raise ResourceProjectionError("barrier id must be non-empty")
        if not self.signallers:
            raise ResourceProjectionError(
                f"barrier {self.barrier_id!r} needs at least one signaller "
                "leg (layer completion = cross-rank max)")
        if any(not isinstance(leg_id, str) or not leg_id.strip()
               for leg_id in self.signallers):
            raise ResourceProjectionError(
                f"barrier {self.barrier_id!r} signallers must be non-empty ids")
        if len(set(self.signallers)) != len(self.signallers):
            raise ResourceProjectionError(
                f"barrier {self.barrier_id!r} has duplicate signallers")


@dataclass(frozen=True)
class TaskSpec:
    """一条资源腿（W1 账本的最小构造单位；fixture 直接构造）。

    * ``kind=stream``：``route`` 上以 ``min_e(peak·η/n_e)`` 恒速消耗
      ``bytes_amount``；``latency_ns`` 为入流集合前的 job latency（HBM/
      池端口语义：latency 后入流集合，latency 期间**不占**带宽分母）。
    * ``kind=fixed``：``duration_ns`` 固定时长（纯 latency 腿/协议腿）。
    * ``kind=compute``：roofline 腿——``route``/``bytes_amount`` 为 memory
      字节流（HBM 端口），``base_ns``×γ 为计算时长，两者并行，完成 =
      max；memory 字节 drained 即离开 HBM 分母（带宽随字节完成释放）。
    * 依赖：``deps`` 可引用腿 id、``work:<id>`` 伪节点（整工作完成）或
      barrier id；激活时刻 = max(自身 ready_at, 全部 deps 完成)。
    * 门：``acquires`` 门控激活（准入门）；``releases`` 在完成时点释放
      （如消费腿释放填充腿取得的缓冲块）。
    * W1 账本部分进度：``remaining_bytes``/``remaining_duration_ns`` 缺省
      = 全量；``in_service=True`` 表示腿已在服务（立即入活跃集、latency
      已付、已占门计数照记）。committed-not-issued 用 ready_at/deps 表达
      （锚2）。
    * ``order``：同刻结算序首键；缺省按 ``leg_id`` 字典序（输入列举顺序
      换序不改变结果）。
    """

    leg_id: str
    kind: str = TASK_KIND_STREAM
    route: Tuple[str, ...] = ()
    bytes_amount: int = 0
    duration_ns: float = 0.0
    base_ns: float = 0.0
    group: str = "prefill"
    latency_ns: int = 0
    ready_at_ns: Optional[int] = None
    deps: Tuple[str, ...] = ()
    acquires: Tuple[GateLease, ...] = ()
    releases: Tuple[GateLease, ...] = ()
    signals_barrier: Optional[str] = None
    remaining_bytes: Optional[int] = None
    remaining_duration_ns: Optional[float] = None
    in_service: bool = False
    order: Optional[int] = None
    # Original WorkItem registration position.  This is independent of
    # ``order`` (the same-tick completion tie-break key) and is populated when
    # ordinary legs share a WorkItem with algebraic pipeline legs.
    registration_ordinal: Optional[int] = None
    # §9.6 第 2/3 条（joint 二期）：die 计划腿——在场时本腿由 A3 执行
    # 协议推进（timer 阶段 = 内部时长、memory 阶段 = service_units 按
    # ProjectionWorld 的分享速率排空），不再携带聚合 (bytes, base_ns)
    # 等效量（退役纪律：不把 service 时间伪造成字节、不按 pure-time
    # 比例分摊）。值为 joint.die_plan_projection.DiePlanDescriptor（鸭子
    # 类型：specs/route/create_execution）。
    die_plan: Optional[object] = None
    # Canonical external local access. The native coarse Runtime owns its
    # service and typed counters; scalar bytes/base/duration stay zero.
    coarse_endpoint: Optional[Mapping] = None

    def __post_init__(self) -> None:
        if not isinstance(self.leg_id, str) or not self.leg_id.strip():
            raise ResourceProjectionError("leg id must be non-empty")
        if self.coarse_endpoint is not None:
            if (not isinstance(self.coarse_endpoint, Mapping)
                    or self.die_plan is not None or self.kind != TASK_KIND_STREAM
                    or len(self.route) != 1
                    or self.bytes_amount or self.base_ns or self.duration_ns
                    or self.latency_ns or self.in_service
                    or self.remaining_bytes is not None
                    or self.remaining_duration_ns is not None):
                raise ResourceProjectionError(
                    "coarse endpoint requires a fresh stream leg, one local "
                    "domain and no scalar work aliases")
        if self.kind not in (
                TASK_KIND_STREAM, TASK_KIND_FIXED, TASK_KIND_COMPUTE):
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} kind must be stream/fixed/compute, "
                f"got {self.kind!r}")
        for name, value in (("bytes_amount", self.bytes_amount),
                            ("latency_ns", self.latency_ns)):
            if (not isinstance(value, int) or isinstance(value, bool)
                    or value < 0):
                raise ResourceProjectionError(
                    f"leg {self.leg_id!r} {name} must be int >= 0")
        for name, value in (
                ("duration_ns", self.duration_ns), ("base_ns", self.base_ns),
                ("remaining_duration_ns",
                 self.remaining_duration_ns
                 if self.remaining_duration_ns is not None else 0.0)):
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or value < 0):
                raise ResourceProjectionError(
                    f"leg {self.leg_id!r} {name} must be finite >= 0")
        if self.ready_at_ns is not None and (
                isinstance(self.ready_at_ns, bool)
                or not isinstance(self.ready_at_ns, int)
                or self.ready_at_ns < 0):
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} ready_at_ns must be int >= 0")
        if self.registration_ordinal is not None and (
                type(self.registration_ordinal) is not int
                or not 0 <= self.registration_ordinal <= (1 << 64) - 1):
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} registration_ordinal must be a uint64 or null")
        if self.remaining_bytes is not None and (
                type(self.remaining_bytes) is not int
                or self.remaining_bytes < 0
                or self.remaining_bytes > self.bytes_amount):
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} remaining_bytes must be 0.."
                f"bytes_amount (W1 ledger partial progress)")
        if self.kind in (TASK_KIND_STREAM, TASK_KIND_COMPUTE):
            if any(not isinstance(resource, str) or not resource.strip()
                   for resource in self.route):
                raise ResourceProjectionError(
                    f"leg {self.leg_id!r} route entries must be non-empty ids")
            if len(set(self.route)) != len(self.route):
                raise ResourceProjectionError(
                    f"leg {self.leg_id!r} route contains duplicate resources")
            if self.bytes_amount > 0 and not self.route:
                raise ResourceProjectionError(
                    f"leg {self.leg_id!r} has bytes but no route (missing "
                    "hardware facts must fail closed, not zero cost)")
        if self.kind == TASK_KIND_FIXED and self.route:
            raise ResourceProjectionError(
                f"fixed leg {self.leg_id!r} must not occupy stream resources")
        if self.kind == TASK_KIND_FIXED and (
                self.bytes_amount != 0
                or (self.remaining_bytes is not None
                    and self.remaining_bytes != 0)):
            raise ResourceProjectionError(
                f"fixed leg {self.leg_id!r} must not carry byte work")
        if not isinstance(self.group, str):
            raise ResourceProjectionError(
                f"compute leg {self.leg_id!r} group must be a string")
        if self.kind == TASK_KIND_COMPUTE and not self.group.strip():
            raise ResourceProjectionError(
                f"compute leg {self.leg_id!r} needs a gamma group name")
        if any(not isinstance(dep, str) or not dep.strip()
               for dep in self.deps):
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} deps must be non-empty ids")
        if len(set(self.deps)) != len(self.deps):
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} has duplicate dependencies")
        if self.leg_id in self.deps:
            raise ResourceProjectionError(
                f"leg {self.leg_id!r} cannot depend on itself")
        if self.in_service:
            if self.deps or self.ready_at_ns is not None:
                raise ResourceProjectionError(
                    f"in-service leg {self.leg_id!r} must not carry deps/"
                    "ready_at (already running; ledger state, not "
                    "scheduling state)")
            if self.remaining_bytes is None and (
                    self.kind in (TASK_KIND_STREAM, TASK_KIND_COMPUTE)):
                raise ResourceProjectionError(
                    f"in-service leg {self.leg_id!r} must declare "
                    "remaining_bytes (partial progress from the W1 ledger)")
            if (self.remaining_duration_ns is None
                    and self.kind == TASK_KIND_FIXED):
                raise ResourceProjectionError(
                    f"in-service leg {self.leg_id!r} must declare "
                    "remaining_duration_ns (partial progress)")
        # §9.6 退役纪律：die 计划腿不携带聚合等效量（bytes/base_ns/
        # duration 是 legacy roofline 双侧表示；die 侧阶段由协议交付），
        # 也不进入 W1 在账部分进度面（die 腿 = 候选侧预测，非在账腿）。
        if self.die_plan is not None:
            if self.kind != TASK_KIND_COMPUTE:
                raise ResourceProjectionError(
                    f"die plan leg {self.leg_id!r} must be kind=compute")
            if (self.bytes_amount != 0 or self.base_ns != 0.0
                    or self.duration_ns != 0.0
                    or self.remaining_bytes is not None
                    or self.remaining_duration_ns is not None):
                raise ResourceProjectionError(
                    f"die plan leg {self.leg_id!r} must not carry the "
                    "retired aggregate (bytes/base_ns/duration) surface "
                    "(section 9.6: no service-time-as-bytes fabrication)")
            if self.in_service:
                raise ResourceProjectionError(
                    f"die plan leg {self.leg_id!r} cannot be in_service "
                    "(candidate-side projection, not a W1 ledger leg)")
            if not getattr(self.die_plan, "route", None):
                raise ResourceProjectionError(
                    f"die plan leg {self.leg_id!r} descriptor has no HBM "
                    "port route")
            if tuple(self.route) != tuple(self.die_plan.route):
                raise ResourceProjectionError(
                    f"die plan leg {self.leg_id!r} route must match the "
                    "descriptor HBM port route (the share set is decided "
                    "by the native projection kernel on the leg route)")


@dataclass(frozen=True)
class WorkItem:
    """一个工作（若干腿）；committed 背景 / 候选 / 焦点共用同一构造面。

    ``ready_at_ns``/``depends_on_work`` 为工作级缺省门：**无显式 deps 且未
    显式给 ready_at 的腿**继承它们（已链化的后继腿只看自身 deps；链化腿
    的就绪 = max(工作级 ready_at, 自身 deps 完成)）。
    ``service_leg_ids`` 指定 service_done 的腿集合（缺省全部腿）；
    settle_done = 全部腿完成（W5 接正式结算协议前的 fixture 口径）。
    """

    work_id: str
    legs: Tuple[TaskSpec, ...]
    ready_at_ns: int = 0
    depends_on_work: Tuple[str, ...] = ()
    is_candidate: bool = False
    service_leg_ids: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if not isinstance(self.work_id, str) or not self.work_id.strip():
            raise ResourceProjectionError("work id must be non-empty")
        if (isinstance(self.ready_at_ns, bool)
                or not isinstance(self.ready_at_ns, int)
                or self.ready_at_ns < 0):
            raise ResourceProjectionError(
                f"work {self.work_id!r} ready_at_ns must be int >= 0")
        if (not isinstance(self.legs, tuple) or not all(
                isinstance(leg, TaskSpec) for leg in self.legs)):
            raise ResourceProjectionError(
                f"work {self.work_id!r} legs must be a tuple of TaskSpec "
                "(单腿也须尾逗号成组——裸 TaskSpec 迭代会以 TypeError 逃逸)")
        if not self.legs:
            raise ResourceProjectionError(
                f"work {self.work_id!r} must have a leg")
        seen: Set[str] = set()
        for leg in self.legs:
            if leg.leg_id in seen:
                raise ResourceProjectionError(
                    f"work {self.work_id!r} duplicate leg {leg.leg_id!r}")
            seen.add(leg.leg_id)
        if any(not isinstance(dep, str) or not dep.strip()
               for dep in self.depends_on_work):
            raise ResourceProjectionError(
                f"work {self.work_id!r} dependencies must be non-empty ids")
        if len(set(self.depends_on_work)) != len(self.depends_on_work):
            raise ResourceProjectionError(
                f"work {self.work_id!r} has duplicate work dependencies")
        if self.work_id in self.depends_on_work:
            raise ResourceProjectionError(
                f"work {self.work_id!r} cannot depend on itself")
        if self.service_leg_ids is not None:
            if any(not isinstance(leg_id, str) or not leg_id.strip()
                   for leg_id in self.service_leg_ids):
                raise ResourceProjectionError(
                    f"work {self.work_id!r} service legs must be ids")
            if len(set(self.service_leg_ids)) != len(self.service_leg_ids):
                raise ResourceProjectionError(
                    f"work {self.work_id!r} has duplicate service legs")
            unknown = set(self.service_leg_ids) - seen
            if unknown:
                raise ResourceProjectionError(
                    f"work {self.work_id!r} service_leg_ids unknown: "
                    f"{sorted(unknown)}")


@dataclass(frozen=True)
class BlockPipelineSpec:
    """共享有限缓冲的顺序块管线（§5.3 解析段加速算子的协议常数）。

    协议（与逐步事件推进同构）：串行 fill（来源边界排空、确定顺序）×
    串行 consume × ``buffer_slots`` 块上限；fill 腿在激活时取得缓冲块、
    consume 腿在完成时释放；``startup_ns`` 为首块启动延迟（只付一次——
    ``startup_paid`` 结转后分段不重复收取）。
    """

    pipeline_id: str
    fill_resource: str
    block_bytes: int
    num_blocks: int
    consume_duration_ns: float
    buffer_gate_id: str
    buffer_slots: int = 2
    startup_ns: int = 0

    def __post_init__(self) -> None:
        if not isinstance(self.block_bytes, int) or self.block_bytes <= 0:
            raise ResourceProjectionError(
                f"pipeline {self.pipeline_id!r} block_bytes must be int > 0")
        if not isinstance(self.num_blocks, int) or self.num_blocks <= 0:
            raise ResourceProjectionError(
                f"pipeline {self.pipeline_id!r} num_blocks must be int > 0")
        if not math.isfinite(self.consume_duration_ns) or (
                self.consume_duration_ns < 0):
            raise ResourceProjectionError(
                f"pipeline {self.pipeline_id!r} consume duration must be "
                "finite >= 0")
        if not isinstance(self.buffer_slots, int) or self.buffer_slots < 1:
            raise ResourceProjectionError(
                f"pipeline {self.pipeline_id!r} buffer_slots must be int >= 1")
        if not isinstance(self.startup_ns, int) or self.startup_ns < 0:
            raise ResourceProjectionError(
                f"pipeline {self.pipeline_id!r} startup must be int >= 0")


@dataclass(frozen=True)
class BlockPipelineState:
    """块管线的可结转状态（已服务字节 + 双 buffer 占位 + 启动标记）。

    不变式：在充块恰一块（``blocks_remaining`` ≥ 1 时 ``in_fill_served`` <
    ``block_bytes``）；占用中槽位 = (1 if consuming) + buffered_full +
    (1 if blocks_remaining ≥ 1) ≤ buffer_slots。
    """

    now_ns: float
    rate_bns: float
    blocks_remaining: int
    in_fill_served_bytes: float = 0.0
    buffered_full_blocks: int = 0
    consuming_remaining_ns: float = 0.0
    startup_paid: bool = False

    def __post_init__(self) -> None:
        if self.blocks_remaining < 0 or self.buffered_full_blocks < 0:
            raise ResourceProjectionError("pipeline block counts must be >= 0")
        if not math.isfinite(self.rate_bns) or self.rate_bns <= 0:
            raise ResourceProjectionError(
                "pipeline rate must be finite positive, got "
                f"{self.rate_bns!r}")
        if self.in_fill_served_bytes < 0:
            raise ResourceProjectionError("in-fill served bytes must be >= 0")
        if self.consuming_remaining_ns < 0:
            raise ResourceProjectionError(
                "consuming remaining must be >= 0")


@dataclass(frozen=True)
class CoarseDieDomainSpec:
    """Frozen canonical local hardware; no Python capacity or timing formulas."""
    resource_key: str
    hardware_config: Mapping

    def __post_init__(self) -> None:
        if (not isinstance(self.resource_key, str)
                or not self.resource_key.startswith("hbm:")
                or not isinstance(self.hardware_config, Mapping)
                or self.hardware_config.get("schema") != "coarse_die_v2"):
            raise ResourceProjectionError(
                "coarse domain requires an HBM resource and coarse_die_v2 hardware")


@dataclass(frozen=True)
class ProjectionInput:
    """冻结前推输入（背景账本 + 硬件 + 焦点；候选经 ``with_candidate`` 叠加）。

    ``background_works`` 为决策快照时刻的真实账本（含 in-service 部分进度
    与 committed-not-issued 的 ready_at/deps 腿）；``focus_work_id`` 为本
    次预测目标（无候选运行 = 背景焦点，如参考运行）。
    """

    now_ns: int
    resources: Tuple[ResourceSpec, ...]
    background_works: Tuple[WorkItem, ...]
    focus_work_id: str
    gates: Tuple[GateSpec, ...] = ()
    barriers: Tuple[BarrierSpec, ...] = ()
    efficiency: ProjectionEfficiency = field(
        default_factory=ProjectionEfficiency)
    pipelines: Tuple[BlockPipelineSpec, ...] = ()
    forecast_arrival_placeholders_ns: Tuple[int, ...] = ()
    # v5 共享快照形态：background_works 中预测背景（forecast works）的起始
    # 下标；None = 无 forecast 段（共享上下文的私有 overlay extras 拼接到
    # 末尾）。仅共享上下文序列化使用；旧全量逐候选快照不设置该字段。
    forecast_start_index: Optional[int] = None
    coarse_die_domains: Tuple[CoarseDieDomainSpec, ...] = ()

    def __post_init__(self) -> None:
        if (isinstance(self.now_ns, bool)
                or not isinstance(self.now_ns, (int, float))
                or not math.isfinite(self.now_ns)
                or self.now_ns < 0):
            raise ResourceProjectionError(
                "now_ns must be a finite non-negative time")
        resource_keys = set()
        for resource in self.resources:
            if resource.key in resource_keys:
                raise ResourceProjectionError(
                    f"duplicate resource {resource.key!r}")
            resource_keys.add(resource.key)
        domain_keys = set()
        for domain in self.coarse_die_domains:
            if (not isinstance(domain, CoarseDieDomainSpec)
                    or domain.resource_key not in resource_keys
                    or domain.resource_key in domain_keys):
                raise ResourceProjectionError(
                    "coarse domains must uniquely name registered physical resources")
            domain_keys.add(domain.resource_key)
        gate_capacity: Dict[str, int] = {}
        for gate in self.gates:
            if gate.gate_id in gate_capacity:
                raise ResourceProjectionError(
                    f"duplicate gate {gate.gate_id!r}")
            gate_capacity[gate.gate_id] = gate.capacity_units
        work_ids = set()
        for work in self.background_works:
            if work.work_id in work_ids:
                raise ResourceProjectionError(
                    f"duplicate work {work.work_id!r}")
            work_ids.add(work.work_id)
        barrier_ids = set()
        signaller_ids = set()
        for barrier in self.barriers:
            if barrier.barrier_id in barrier_ids:
                raise ResourceProjectionError(
                    f"duplicate barrier {barrier.barrier_id!r}")
            barrier_ids.add(barrier.barrier_id)
            for signaller in barrier.signallers:
                if signaller in signaller_ids:
                    raise ResourceProjectionError(
                        f"leg {signaller!r} cannot signal two barriers")
                signaller_ids.add(signaller)
        # 共享快照形态（forecast_start_index 在场）延迟 focus 校验到
        # 原生 overlay 应用时重算——共享底座可为空背景（锚与否取决于
        # 逐调用 overlay 的 extras），focus 为占位符。
        if self.forecast_start_index is None and (
                self.focus_work_id not in work_ids):
            raise ResourceProjectionError(
                f"focus work {self.focus_work_id!r} not in background_works")
        for pipeline in self.pipelines:
            if pipeline.fill_resource not in resource_keys:
                raise ResourceProjectionError(
                    f"pipeline {pipeline.pipeline_id!r} fill resource "
                    f"{pipeline.fill_resource!r} unknown")
            if pipeline.buffer_gate_id not in gate_capacity:
                raise ResourceProjectionError(
                    f"pipeline {pipeline.pipeline_id!r} buffer gate "
                    f"{pipeline.buffer_gate_id!r} unknown")
            if pipeline.buffer_slots > gate_capacity[
                    pipeline.buffer_gate_id]:
                raise ResourceProjectionError(
                    f"pipeline {pipeline.pipeline_id!r} declares more buffer "
                    "slots than the gate capacity")
        for tick in self.forecast_arrival_placeholders_ns:
            if (type(tick) is not int or tick < self.now_ns):
                raise ResourceProjectionError(
                    "forecast arrival placeholders must be integer >= now_ns")

    def with_candidate(
        self, candidates: Sequence[WorkItem],
    ) -> "ProjectionInput":
        """候选增量副本（ERP ``with_committed`` 语义：冻结输入叠加候选）。

        每候选独立世界从本副本重建——移除候选后背景参考还原，不存在跨候
        选污染（§5.4）。焦点缺省改为首个候选工作（J 预测目标 = 候选本
        轮完成时间）。
        """
        candidates = tuple(candidates)
        for candidate in candidates:
            if not candidate.is_candidate:
                raise ResourceProjectionError(
                    f"candidate work {candidate.work_id!r} must be flagged "
                    "is_candidate=True")
            if candidate.work_id in {w.work_id for w in self.background_works}:
                raise ResourceProjectionError(
                    f"candidate {candidate.work_id!r} duplicates a background "
                    "work id (commit order dedup, §4.4)")
        focus = candidates[0].work_id if candidates else self.focus_work_id
        return ProjectionInput(
            now_ns=self.now_ns,
            resources=self.resources,
            background_works=self.background_works + candidates,
            focus_work_id=focus,
            gates=self.gates,
            barriers=self.barriers,
            efficiency=self.efficiency,
            pipelines=self.pipelines,
            forecast_arrival_placeholders_ns=(
                self.forecast_arrival_placeholders_ns),
            coarse_die_domains=self.coarse_die_domains,
        )



# 该模块的 Python 事件执行循环（含解析段算子）与模块级 project() 入口
# 已按 C5 退役（补丁 0001-retire-python-projection-loop）；typed DTO
# 面、世界源与 NativeProjectionWorld facade 保留。原循环的仓外冻结
# 脚本见证据根 packages/python-impl-20261006/c5-prep/frozen-tests/。
# ------------------------------------------------------------------ 结果 --


@dataclass(frozen=True)
class BackgroundComparison:
    """候选反作用的显式输出面：每背景工作 with/without 候选完成时刻。"""

    work_id: str
    done_without_candidate_ns: Optional[int]
    done_with_candidate_ns: Optional[int]

    @property
    def delay_ns(self) -> Optional[int]:
        if (self.done_without_candidate_ns is None
                or self.done_with_candidate_ns is None):
            return None
        return self.done_with_candidate_ns - self.done_without_candidate_ns


@dataclass(frozen=True)
class ProjectionResult:
    """§4.1 契约的 W2 终稿（W3 接线时由此填 breakdown）。"""

    status: str
    predicted_service_done_ns: Optional[int]
    predicted_settle_done_ns: Optional[int]
    reason: str = ""
    dependency_attribution: Tuple[str, ...] = ()
    focus_bytes: int = 0
    bytes_served_total: int = 0
    event_count: int = 0
    resource_count: int = 0
    work_count: int = 0
    closure_resource_count: int = 0
    closure_work_count: int = 0
    approximations: Tuple[str, ...] = ()
    fallback: Optional[str] = None
    background_comparisons: Tuple[BackgroundComparison, ...] = ()
    # 墙时分段(2026-10-05 埋点,perf_counter_ns,仅 project() 粗相级——
    # 事件循环 _step 内零计时):影响闭包构建 / 候选世界 run / 背景参考
    # 世界 run 三段 + 背景参考运行计数(0/1——project 单次调用内至多
    # 一次)。纯披露,不进任何计价语义。
    closure_wall_ns: int = 0
    candidate_world_wall_ns: int = 0
    reference_world_wall_ns: int = 0
    reference_world_runs: int = 0
    # Native compact-window quality/disclosure fields are kept separate from
    # the legacy result surface; the Python reference leaves this None.
    native_disclosure: Optional[Mapping[str, object]] = None

    def as_dict(self) -> dict:
        result = {
            "status": self.status,
            "predicted_service_done_ns": self.predicted_service_done_ns,
            "predicted_settle_done_ns": self.predicted_settle_done_ns,
            "reason": self.reason,
            "dependency_attribution": list(self.dependency_attribution),
            "focus_bytes": self.focus_bytes,
            "bytes_served_total": self.bytes_served_total,
            "event_count": self.event_count,
            "resource_count": self.resource_count,
            "work_count": self.work_count,
            "closure_resource_count": self.closure_resource_count,
            "closure_work_count": self.closure_work_count,
            # 注意:墙时分段字段(closure/candidate_world/reference_world_
            # wall_ns、reference_world_runs)是宿主墙时观测值,**不进
            # as_dict**——本方法是候选枚举换序不变性等确定性契约的
            # 比对面(见 test_candidate_enumeration_order_independent),
            # 埋点值只经属性与 _projection_delivery_stats 披露。
            "approximations": list(self.approximations),
            "fallback": self.fallback,
            "background_comparisons": [
                {
                    "work_id": item.work_id,
                    "done_without_candidate_ns": (
                        item.done_without_candidate_ns),
                    "done_with_candidate_ns": item.done_with_candidate_ns,
                    "delay_ns": item.delay_ns,
                }
                for item in self.background_comparisons],
        }
        if self.native_disclosure is not None:
            result.update(self.native_disclosure)
        return result

# 模块级 project() 旧入口已随循环退役（C5）；生产候选前推唯一路径
# = action_work_graph.CandidateGraph.project_against 的 native 投影。
