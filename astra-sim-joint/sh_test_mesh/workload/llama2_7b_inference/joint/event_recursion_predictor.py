"""event_recursion_predictor.py -- E：运行期事件递推预测器（C15，2026-09-22）。

设计依据：《三机制联合策略_template仓库设计方案》§5.1/§5.2/§5.4/§5.5/
§5.6/§5.7；设计文档《joint机制改造方案_局部统一内存域》§9（E 的运行期
事件递推为实现缺项）；F12（递推预测器 = ``adaptive`` 的正式在线实现，
替换现行解析在线模型，同公式同因果边界；不新增开关、不静默回退）。

本模块在同一只读资源快照上做小型事件递推，**写回与恢复两方向同推演
仲裁**（§5.2 递推段）：

* 召回侧：加入候选冷层恢复流（逐组、消费顺序、逐 rank 串行链——与
  构图器 GB 的逐组恢复支链同构），推进可观测依赖与资源仲裁，输出逐
  层恢复完成时刻 ``R̂_ℓ(k)`` 与消费期限 ``D̂_ℓ``。恢复写腿与计算
  memory 腿在同一端口仲裁内竞争（§5.4"恢复写入造成的 HBM 竞争不能
  遗漏"）——候选自致的计算减速单列 ``compute_slowdown_ns`` 记账、
  不回移期限。
* 逐出侧：加入候选写回流（源 HBM 读/路径传输/池端口写），输出空间
  可用时刻；同一缺口多 victim 按**对象顺序逐个决策**，每定一个 victim
  其写回流即登记进共享快照供后续 victim 推演——同批 victim 不得各自
  按独享带宽估算空间可用时刻。
* 两方向合流（``predict_release_and_recall``）：恢复流的就绪门槛
  ``q̂`` 取"首块启动等待"与"写回流空间可用时刻"的关键路径最大值，
  恢复流在含已定写回流的共享快照上递推。

资源仲裁口径与 C1 ``LinkFlowRegistry.divisor_multi`` / C2
``HbmPortFlowRegistry.divisor`` 同源（冻结形状注入，不 import 交付物）：
等权 N-way 均分——资源上共享流数 n_e（已提交在册 + 活跃候选）决定份额
``peak_e·η_e/n_e``，流启动/完成/已提交 ETA 到期时重算（§5.4）。

``k̂_hide`` 枚举与剪枝口径（钉死，§5.2 剪枝段）：

1. **解析下界（唯一剪枝依据）**：逐 rank 松弛——各 rank 以自身恢复
   路径的独享峰值服务率（路径各资源 ``η·peak`` 取 min）按消费顺序串行
   累计 ``Σ b_{j,r}/B_r^excl``，跨 rank 取 max、加首块启动 ``q̂``；不计
   中间层启动与 rank 间互斥。该量 ≤ 事件递推的 ``R̂``。**不得**以 §5.4
   的 ``r̂_j`` 逐层近似求和 ``Σ_j max_r`` 代替（跨层 rank 并行下它可能
   高于递推结果，按它剪枝会误剪可行 k）。
2. **期限钉死**：``D̂_ℓ`` = 无候选恢复流量（reference 递推）的消费完
   成时刻；候选引起的计算减速单列成本记账、不回移/不放松期限。
3. 该口径下解析不可行 ⇒ 递推必不可行（无漏剪）——枚举自解析最小
   可行 k 起向上；解析不可用（字节/速率缺失或未给出可行 k）则自 0
   全枚举并标注状态 ``analytic_unavailable``。解析可行不保证递推可
   行，仍须递推校验。**不二分、不假设共享资源下单调**。
4. 多通道、staging 阻塞与共享 HBM/collective 存在时，递推内部 R 随 k
   如实变化，不假定各层恢复时间固定；无法完成、未知释放 ETA、遥测
   覆盖不足 → 状态标注（``unknown_release_eta`` / ``coverage_*``），
   不当零代价、不删实例候选。

因果边界（§5.5）：预测只读快照内已提交工作的可见剩余字节与已知请求
可推导的算子，不读仿真事件队列预存的未来真值；未来 CSV 行、未来输出
长度、真实返回时间不得进入本模块（调用方保证；估计器只在
``observe_*`` 显式调用时更新——改变未来信息、保持当前可见状态相同，
E 输出与估计器状态不变）。

纯函数 + 状态范式（``eviction_priority.py`` 同款）：无 I/O、无环境读
取、不改 KV 账本/事件队列；``adaptive`` 的在线身份由调用方
（face_scheduler）接线，``legacy_half``/``minimal_layer_groups`` 身份
不变。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

# 预测器来源标识（决策披露用）：递推 = adaptive 正式在线实现。
PREDICTOR_SOURCE_RECURSION = "recursion"
PREDICTOR_SOURCE_ANALYTIC_FALLBACK = "analytic_unavailable_full_enumeration"

# 状态标注（§5.2：不当零代价、不删实例候选）。
STATUS_UNKNOWN_RELEASE_ETA = "unknown_release_eta"
STATUS_ANALYTIC_UNAVAILABLE = "analytic_unavailable"
STATUS_COVERAGE_NO_COLLECTIVE = "coverage_no_collective_telemetry"
STATUS_COVERAGE_NO_LINK_MODEL = "coverage_no_link_model"
STATUS_COVERAGE_PORT_REGISTRY = "coverage_port_registry_executor"

# 统一数值精度规则：整数 ns 严格比较，不附加按 workload 调优的安全带。
_FEASIBILITY_EPSILON_NS = 0.0

# step() 积分守恒校验开关（F2/A11'）：分段恒速积分不变量的步级复核
# ——① Σremaining 减少量 == Σ区间服务字节；② Σ区间服务字节 == Σ活跃
# 流份额速率 × 推进时长（时刻-服务联动）。raise 级
# ConservationViolationError（EventRecursionError 子类），默认开，
# python -O 免疫；O(|active|) 求和开销敏感的热路径可置 False 关闭
# （仅去校验，数值行为不变——见 ``UnifiedTimeline.step`` docstring）。
_CONSERVATION_CHECK = True
# 守恒容差（相对 pre 总量）：完成流移除时残留的浮点零头由其吸收；
# finishes = now + remaining/rate 在大时间基座（now ≳ 1e12 ns）下的
# catastrophic cancellation 残差另由 Σrate×ulp(now) 量化松弛吸收
# （A11'——1e12 ns 基座实证误杀修正，见 _assert_conservation）。
_CONSERVATION_TOLERANCE = 1e-6

# predict_k_hide(analytic_min_k=...) 的"缺省 = 自算解析下界"哨兵（与
# 调用方显式 None = "解析校验不可用"区分）。
_AUTO_ANALYTIC = object()


class EventRecursionError(ValueError):
    """fail-closed：非法快照/腿参数或递推内部不变量破损。"""


class ConservationViolationError(EventRecursionError):
    """A11'（§4.3 补遗）：step() 积分守恒不变量破损。

    EventRecursionError 子类——face 调用方既有降级通道（保守保留全
    层 + recursion_error 披露）直接接住，消除"守恒违规走裸 assert
    击穿调度器 / python -O 静默剥离"的通道分裂；raise 级检查对
    解释器旗标免疫。"""


# ============================================================ 只读快照 ==


@dataclass(frozen=True)
class CommittedFlow:
    """快照内一条已提交（在册）流。

    ``resources`` = 该流占用的资源键序列；``release_eta_ns`` = 已知完
    成时刻（None = 未知释放 ETA——流持续占用仲裁份额，状态标注，不当
    零代价）。
    """

    flow_id: str
    resources: Tuple[str, ...]
    release_eta_ns: Optional[int]


@dataclass(frozen=True)
class ResourceSnapshot:
    """只读资源快照：峰值速率表 + 在册流 + 覆盖披露。

    资源键约定（字符串前缀分类）：``link:<src>-><dst>``（有向链路）、
    ``port:<rank>``（实例 HBM 端口）、``pool:<edge_rank>``（池边缘端
    口）。腿路径上出现 ``peak_bytes_per_ns`` 未含的资源键 → 接线
    fail-closed（缺峰值 = 缺硬件事实，不能静默按无限速率）。
    """

    peak_bytes_per_ns: Mapping[str, float]
    committed: Tuple[CommittedFlow, ...] = ()
    collective_coverage: bool = False
    port_telemetry: bool = False
    link_model: bool = False
    now_ns: int = 0

    def __post_init__(self) -> None:
        # L2（P2-2，2026-09-23 复核审计）：now_ns 非有限拒收——NaN 进
        # step() 后 `start <= NaN` 恒 False 流永不激活、
        # `max(NaN, min(starts))` 保持 NaN ⇒ run() 死循环（K8 同族
        # 教义：入口 fail-closed，与 peak/字节字段同一验证面）。
        if not math.isfinite(self.now_ns):
            raise EventRecursionError(
                f"snapshot now_ns must be finite, got {self.now_ns!r}")
        for key, rate in self.peak_bytes_per_ns.items():
            if not math.isfinite(rate) or rate <= 0:
                raise EventRecursionError(
                    f"resource {key!r} peak rate must be finite positive")
        for flow in self.committed:
            if not flow.resources:
                raise EventRecursionError(
                    f"committed flow {flow.flow_id!r} occupies no resource")
            # L2（P2-2）：release_eta_ns 非有限拒收——NaN 的
            # `eta > now` 恒 False ⇒ 在册流被当已过期、竞争份额被
            # 静默忽略（乐观方向，K8 自称封死的同族形态）；负值合法
            # （已过期语义）。
            if (flow.release_eta_ns is not None
                    and not math.isfinite(flow.release_eta_ns)):
                raise EventRecursionError(
                    f"committed flow {flow.flow_id!r} release ETA must "
                    f"be finite, got {flow.release_eta_ns!r}")
            for key in flow.resources:
                if key not in self.peak_bytes_per_ns:
                    raise EventRecursionError(
                        f"committed flow {flow.flow_id!r} occupies unknown "
                        f"resource {key!r} (missing peak rate)")

    def with_committed(
        self, extra: Sequence[CommittedFlow],
    ) -> "ResourceSnapshot":
        return ResourceSnapshot(
            peak_bytes_per_ns=self.peak_bytes_per_ns,
            committed=self.committed + tuple(extra),
            collective_coverage=self.collective_coverage,
            port_telemetry=self.port_telemetry,
            link_model=self.link_model,
            now_ns=self.now_ns,
        )

    def coverage_statuses(self) -> Tuple[str, ...]:
        statuses = []
        if not self.collective_coverage:
            statuses.append(STATUS_COVERAGE_NO_COLLECTIVE)
        if not self.link_model:
            statuses.append(STATUS_COVERAGE_NO_LINK_MODEL)
        if not self.port_telemetry:
            statuses.append(STATUS_COVERAGE_PORT_REGISTRY)
        return tuple(statuses)

    def unknown_eta_flow_ids(self) -> Tuple[str, ...]:
        return tuple(
            flow.flow_id for flow in self.committed
            if flow.release_eta_ns is None)


@dataclass(frozen=True)
class EfficiencyFactors:
    """在线效率因子（§5.4）：η（传输资源）与 γ（计算组）。

    键匹配：先精确资源键/组名，再取 ``:`` 前的类前缀（``pool``/
    ``link``/``port``），缺省 1.0（冷启动，调用方标记）。效率样本的
    observed_ratio **分母**必须用不含修正因子的基础模型（避免循环校
    正）——由调用方保证，本结构只承载现值。
    """

    eta: Mapping[str, float] = field(default_factory=dict)
    gamma: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # A14'（H4，2026-09-22 第三轮复审）：η/γ 与 ResourceSnapshot 对
        # peak 的验证同教义——非有限或 ≤0 直接 fail-closed（η=0 会在
        # finishes 计算处除零且逃出 EventRecursionError 族击穿 face 降
        # 级通道、η=inf 零代价瞬时完成且 quant_slack=∞ 掩蔽两条守恒
        # 式；生产路径 EWMA 恒正有限，仅直连误用可触发）。
        for kind, table in (("eta", self.eta), ("gamma", self.gamma)):
            for key, value in table.items():
                if not math.isfinite(value) or value <= 0:
                    raise EventRecursionError(
                        f"{kind} factor for {key!r} must be finite "
                        "positive")

    def eta_of(self, resource_key: str) -> float:
        if resource_key in self.eta:
            return float(self.eta[resource_key])
        prefix = resource_key.split(":", 1)[0]
        return float(self.eta.get(prefix, 1.0))

    def gamma_of(self, group: str) -> float:
        return float(self.gamma.get(group, 1.0))


# ============================================================ 递推腿 ==


@dataclass(frozen=True)
class RestoreGroupLeg:
    """候选冷层恢复流的一条腿（覆盖 0-based 层区间 [layer_start, layer_end)）。

    逐 rank：``bytes_by_rank[r]`` 缺失字节数；``path_by_rank[r]`` = 该
    rank 恢复路径的资源键序列（池端口 → 链路… → 目标端口）。腿按消费
    顺序（layer_start 递增）排列；rank 内串行链（腿 g+1 等腿 g 完成）、
    rank 间并行——与 GB 逐组恢复支链的发射拓扑同构。
    """

    layer_start: int
    layer_end: int
    bytes_by_rank: Tuple[int, ...]
    path_by_rank: Tuple[Tuple[str, ...], ...]
    startup_ns: int = 0

    def __post_init__(self) -> None:
        if not 0 <= self.layer_start < self.layer_end:
            raise EventRecursionError(
                f"restore leg layer range [{self.layer_start}, "
                f"{self.layer_end}) is empty")
        if len(self.bytes_by_rank) != len(self.path_by_rank):
            raise EventRecursionError(
                "restore leg bytes/path rank vectors must align")
        # K8（2026-09-23 外部审计）：NaN/inf 经 ``value < 0`` 检查（NaN
        # 比较恒 False）存活到 int() 转换，以裸 ValueError 逃出 face 的
        # except EventRecursionError 降级通道——数值字段统一有限性设防。
        if any(not math.isfinite(value) for value in self.bytes_by_rank):
            raise EventRecursionError(
                "restore leg bytes must be finite numbers")
        if any(value < 0 for value in self.bytes_by_rank):
            raise EventRecursionError("restore leg bytes must be >= 0")
        if not math.isfinite(self.startup_ns) or self.startup_ns < 0:
            raise EventRecursionError("restore leg startup must be >= 0")
        for rank_index, (rank_bytes, path) in enumerate(
                zip(self.bytes_by_rank, self.path_by_rank)):
            if rank_bytes > 0 and not path:
                raise EventRecursionError(
                    f"restore leg [{self.layer_start}, {self.layer_end}) "
                    f"rank {rank_index} has bytes but no restore path "
                    "(missing model facts must fail closed, not zero cost)")


@dataclass(frozen=True)
class ComputeLayerSegment:
    """计算侧逐层段（1-based 层号；消费顺序 = 层序；TP 逐层同步）。

    ``base_ns_by_rank`` = 基础模型逐层服务（roofline F/P 腿，不含 γ——
    γ 在预测侧显式乘入，样本分母则用本基础值）；``memory_bytes_by_rank``
    = 该层段内存读字节（HBM 端口腿，与恢复写腿同端口仲裁）；
    ``group`` = γ 分组（事前定义，不按事后输赢分组）。
    """

    layer: int
    base_ns_by_rank: Tuple[float, ...]
    memory_bytes_by_rank: Tuple[int, ...]
    port_by_rank: Tuple[str, ...]
    group: str = "prefill"

    def __post_init__(self) -> None:
        if not (
            len(self.base_ns_by_rank)
            == len(self.memory_bytes_by_rank)
            == len(self.port_by_rank)
        ):
            raise EventRecursionError(
                "compute segment rank vectors must align")
        if any(value < 0 for value in self.base_ns_by_rank):
            raise EventRecursionError("compute base ns must be >= 0")
        # K8：NaN/inf 设防（同 RestoreGroupLeg——int()/比较链逃逸通道封死）。
        if any(not math.isfinite(value)
               for value in self.base_ns_by_rank):
            raise EventRecursionError(
                "compute base ns must be finite numbers")
        if any(not math.isfinite(value)
               for value in self.memory_bytes_by_rank):
            raise EventRecursionError(
                "compute memory bytes must be finite numbers")
        if any(value < 0 for value in self.memory_bytes_by_rank):
            raise EventRecursionError("compute memory bytes must be >= 0")


@dataclass(frozen=True)
class WritebackVictim:
    """逐出侧候选写回流（一个 victim 一条腿；对象顺序由调用方/T 给定）。

    ``path_by_rank[r]`` = 源端口 → 链路… → 池端口。空间可用时刻 = 写回
    流完成时刻（§5.6：写回未完成不算已释放）。
    """

    victim_id: str
    bytes_by_rank: Tuple[int, ...]
    path_by_rank: Tuple[Tuple[str, ...], ...]
    startup_ns: int = 0

    def __post_init__(self) -> None:
        if len(self.bytes_by_rank) != len(self.path_by_rank):
            raise EventRecursionError(
                "writeback bytes/path rank vectors must align")
        # K8：NaN/inf 设防（同上两类腿）。
        if any(not math.isfinite(value) for value in self.bytes_by_rank):
            raise EventRecursionError(
                "writeback bytes must be finite numbers")
        if any(value < 0 for value in self.bytes_by_rank):
            raise EventRecursionError("writeback bytes must be >= 0")
        if not math.isfinite(self.startup_ns) or self.startup_ns < 0:
            raise EventRecursionError("writeback startup must be >= 0")


# ============================================================ 递推引擎 ==


class UnifiedTimeline:
    """等权 N-way 均分流体的确定性统一时间线（恢复/计算/写回共用）。

    时间线元素 = 候选流（字节按当前瓶颈份额 ``min_e peak_e·η_e/n_e``
    消耗）＋已提交流（已知 ETA 占用份额至 ETA；未知 ETA 永久占用——状
    态标注、不当零代价）。速率重算时机 = 任何事件（流激活/完成、ETA
    到期）之后；同刻事件按 flow_id 确定序处理。``step()`` 处理恰好一
    个事件时刻并返回是否有进展——依赖驱动循环（恢复 rank 链、计算段
    依赖）在步间调度后继流，R/D 随 k 变化如实推演。
    """

    def __init__(
        self,
        snapshot: ResourceSnapshot,
        efficiency: Optional[EfficiencyFactors] = None,
    ) -> None:
        self.snapshot = snapshot
        self.efficiency = efficiency or EfficiencyFactors()
        self.now_ns = float(snapshot.now_ns)
        self.active: List = []           # _ActiveFlow
        self._pending: Dict[str, Tuple[Tuple[str, ...], int, float]] = {}
        self.completions: Dict[str, int] = {}

    # ------------------------------------------------------------ 录入 --
    def add_flow(
        self, flow_id: str, resources: Sequence[str], remaining_bytes: int,
        start_ns: int,
    ) -> None:
        """登记候选流（``start_ns`` ≥ 当前时间；激活由 step 驱动）。"""
        if (flow_id in self._pending or flow_id in self.completions
                or any(flow.flow_id == flow_id for flow in self.active)):
            # A14'（H4）：重复 id 检查补在册 active——原只查 pending/
            # completions，同 id 二次添加产生双份份额条目且 next() 只
            # 结第一份、第二份永久滞留。
            raise EventRecursionError(f"duplicate flow id {flow_id!r}")
        if remaining_bytes < 0:
            raise EventRecursionError("flow bytes must be >= 0")
        if not resources:
            raise EventRecursionError(
                f"flow {flow_id!r} must occupy at least one resource")
        for key in resources:
            if key not in self.snapshot.peak_bytes_per_ns:
                raise EventRecursionError(
                    f"flow {flow_id!r} uses unknown resource {key!r}")
        start_value = float(start_ns)
        if not math.isfinite(start_value):
            # A14'（H4）：NaN 永不激活且空活跃分支 max(now, min([nan]))
            # = now 不前进 ⇒ run() 死循环；inf 直接把时刻推到无穷——
            # 均拒绝（docstring"≥ 当前时间"为调用方契约，不作硬校验）。
            raise EventRecursionError(
                f"flow {flow_id!r} start_ns must be finite")
        if remaining_bytes == 0:
            self.completions[flow_id] = max(
                int(start_value), int(self.now_ns))
            return
        self._pending[flow_id] = (
            tuple(resources), int(remaining_bytes), start_value)

    def pending_flow_ids(self) -> Tuple[str, ...]:
        return tuple(sorted(self._pending))

    def has_work(self) -> bool:
        return bool(self.active or self._pending)

    # ------------------------------------------------------------ 内部 --
    def _committed_on(self, resource: str, time_ns: float) -> int:
        count = 0
        for flow in self.snapshot.committed:
            until = flow.release_eta_ns
            if until is None or until > time_ns:
                if resource in flow.resources:
                    count += 1
        return count

    def _rate_of(self, flow, time_ns: float) -> float:
        rate = math.inf
        for resource in flow.resources:
            peak = float(self.snapshot.peak_bytes_per_ns[resource])
            share_count = (
                self._committed_on(resource, time_ns)
                + sum(1 for other in self.active
                      if resource in other.resources))
            effective = peak * self.efficiency.eta_of(resource)
            rate = min(rate, effective / max(1, share_count))
        return rate

    def _next_committed_expiry(self) -> Optional[float]:
        earliest = None
        for flow in self.snapshot.committed:
            if flow.release_eta_ns is not None:
                eta = float(flow.release_eta_ns)
                if eta > self.now_ns and (earliest is None or eta < earliest):
                    earliest = eta
        return earliest

    def _settle_completions(
        self, finishes: Dict[str, float], event_ns: float,
    ) -> None:
        """结算 event_ns 时刻：finish ≤ event 的流按 flow_id 确定序先到
        先结（移出活跃、落 completion = ceil(event)）；随后复核存活流未
        被抽干穿透零（积分不变量破损即 fail-closed）。"""
        done_ids = sorted(
            flow_id for flow_id, finish in finishes.items()
            if finish <= event_ns)
        for flow_id in done_ids:
            flow = next(f for f in self.active if f.flow_id == flow_id)
            self.active.remove(flow)
            self.completions[flow_id] = int(math.ceil(event_ns))
        for flow in self.active:
            if flow.remaining <= -1e-9:
                raise EventRecursionError(
                    f"flow {flow.flow_id!r} drained below zero without "
                    "completing (integration invariant broken)")

    def _assert_conservation(
        self,
        pre_remaining: float,
        served_total: float,
        rates: Mapping[str, float],
        interval_start_ns: float,
    ) -> None:
        """积分守恒校验（A11'：raise 级 ConservationViolationError，
        EventRecursionError 子类——python -O 免疫，face 调用方既有
        降级通道直接接住）：
        ① 字节账目——本步 Σremaining 减少量 == Σ区间服务字节；
        ② 时刻-服务联动——Σ区间服务字节 == Σ活跃流份额速率 × 推进时
        长。只推时刻不扣字节（如 ETA 截断分支丢弃服务量）或字节凭空
        增删都在此被抓；完成流移除时的浮点零头残差由相对容差吸收，
        finishes = now + remaining/rate 在大时间基座下的 catastrophic
        cancellation 残差（rate×ulp(now) 量级）由量化松弛吸收。"""
        if not _CONSERVATION_CHECK:
            return
        post_remaining = sum(flow.remaining for flow in self.active)
        # A11' 量化松弛：finishes 绝对时刻量化到 ulp(now)——完成流的
        # rate×elapsed 扣减与其 remaining 全额移除之差以 Σrate×
        # ulp(now) 为上界（1e12 ns 基座实证误杀修正）。
        quant_slack = sum(rates.values()) * math.ulp(
            max(1.0, float(self.now_ns)))
        if abs(pre_remaining - post_remaining - served_total) > (
                _CONSERVATION_TOLERANCE * max(1.0, pre_remaining)
                + quant_slack):
            raise ConservationViolationError(
                "step() byte conservation violated (remaining decrease != "
                f"served): pre={pre_remaining!r} post={post_remaining!r} "
                f"served={served_total!r}")
        elapsed = self.now_ns - interval_start_ns
        expected = sum(rate * elapsed for rate in rates.values())
        if abs(served_total - expected) > (
                _CONSERVATION_TOLERANCE * max(1.0, abs(expected))
                + quant_slack):
            raise ConservationViolationError(
                "step() time-service linkage violated (advanced time "
                f"without integrating bytes): served={served_total!r} "
                f"expected={expected!r} elapsed={elapsed!r}")

    # ------------------------------------------------------------ 推进 --
    def step(self) -> bool:
        """处理恰好一个事件时刻；返回是否推进。

        分段恒速积分不变量（F2 钉死）：时刻推进与字节扣减必须成对发生
        ——任何推进区间（到最早完成，或截断到已提交流 ETA 到期 / 下一条
        pending 流激活的边界）内，活跃流按该区间恒定份额速率同步扣减
        剩余字节。步末守恒校验（``_CONSERVATION_CHECK`` 默认开，raise
        级 ConservationViolationError——python -O 免疫、调用方降级通道
        接得住）两条：① Σremaining
        减少量 == Σ区间服务字节（字节账目）；② Σ区间服务字节 == Σ活跃
        流份额速率 × 推进时长（时刻-服务联动——只推时刻不扣字节在此
        被抓）。
        """
        # 1) 激活就绪 pending 流。
        for flow_id in sorted(self._pending):
            _resources, _remaining, start = self._pending[flow_id]
            if start <= self.now_ns:
                resources, remaining, _start = self._pending.pop(flow_id)
                self.active.append(_ActiveFlow(
                    flow_id, resources, remaining, self.now_ns))
        if not self.active:
            starts = [start for _r, _b, start in self._pending.values()]
            if not starts:
                return False
            self.now_ns = max(self.now_ns, min(starts))
            return True
        pre_remaining = (
            sum(flow.remaining for flow in self.active)
            if _CONSERVATION_CHECK else 0.0)
        interval_start_ns = self.now_ns
        served_total = 0.0
        # 2) 当前速率下的最早完成；区间边界 = min(最早完成, 已提交流
        #    ETA 到期, 下一条 pending 流激活)——ETA 到期与流激活都是
        #    份额重算事件，截断到边界后按与正常完成分支**同一分段恒速
        #    积分式**扣减活跃流剩余字节（只推时刻不扣字节会让该区间
        #    服务量凭空消失、完成时刻系统性偏晚，步末守恒校验拦截）。
        finishes = {}
        rates: Dict[str, float] = {}
        for flow in self.active:
            rate = self._rate_of(flow, self.now_ns)
            rates[flow.flow_id] = rate
            finishes[flow.flow_id] = self.now_ns + flow.remaining / rate
        next_time = min(finishes.values())
        expiry = self._next_committed_expiry()
        if expiry is not None and expiry < next_time:
            next_time = expiry
        # A11'：pending 激活事件切割积分区间——区间内部激活的流到边界
        # 才加入份额竞争（消除在册流"提前完成"的可行性乐观偏差，兑现
        # 类 docstring"任何事件重算速率"承诺）。
        pending_start = min(
            (start for _r, _b, start in self._pending.values()
             if start > self.now_ns),
            default=None)
        if pending_start is not None and pending_start < next_time:
            next_time = pending_start
        # 3) 推进到边界：区间内速率恒定，按各流当前速率精确扣减剩余
        #    字节（分段恒速积分），随后按 flow_id 确定序处理同刻完成
        #    （浮点触底护栏——精确算术下非完成边界早于 min(finishes)，
        #    不会有流在此完成；判定式与完成分支同口径、同确定序）。
        elapsed = next_time - self.now_ns
        if elapsed > 0:
            for flow in self.active:
                served = rates[flow.flow_id] * elapsed
                flow.remaining -= served
                if flow.remaining < 0.0:
                    # A14'（H4，2026-09-22 第三轮复审）：完成边界 1 ulp
                    # 邻域内的幸存流负尘埃零化——不零化则下一步
                    # finishes = now + 负值/rate < now 出现时刻回退，且
                    # 大时间基座下尘埃可达 rate×ulp(now) ≫ 1e-9、误炸
                    # _settle_completions 的 drained-below-zero（根因
                    # 被误标为积分不变量破损）。真破损（超出尘埃界）仍
                    # 由该阈值拦截 raise；零化只影响幸存流的 post 侧
                    # （完成流在 post 求和前已移除），①式残差只会更小。
                    dust = 1e-9 + (
                        rates[flow.flow_id]
                        * math.ulp(max(1.0, float(next_time))))
                    if flow.remaining >= -dust:
                        flow.remaining = 0.0
                served_total += served
        self.now_ns = next_time
        self._settle_completions(finishes, next_time)
        self._assert_conservation(
            pre_remaining, served_total, rates, interval_start_ns)
        return True

    def run(self) -> None:
        """推进到无待启动流且无活跃流（写回方向整段推演用）。"""
        while self.step():
            pass

    def run_until_complete(self, flow_ids: Iterable[str]) -> None:
        """推进到指定流全部完成（或时间线无事件——依赖驱动循环用）。"""
        wanted = set(flow_ids)
        while wanted - set(self.completions):
            if not self.step():
                break


class _ActiveFlow:
    """时间线上一条活跃流：在所有占用资源上以瓶颈份额同步消耗字节。"""

    __slots__ = ("flow_id", "resources", "remaining", "started_ns")

    def __init__(self, flow_id, resources, remaining, started_ns):
        self.flow_id = flow_id
        self.resources = tuple(resources)
        self.remaining = float(remaining)
        self.started_ns = float(started_ns)


# ------------------------------------------------------------ k 递推 --


@dataclass(frozen=True)
class LayerTiming:
    """单层递推结果（ns；ℓ 为 1-based 层号）。"""

    layer: int
    restore_complete_ns: Optional[int]   # R̂_ℓ(k)（热层 = None）
    deadline_ns: int                     # D̂_ℓ（钉死：无候选恢复流量）
    consumption_ns: int                  # 该层消费完成（含候选减速，如实）


@dataclass(frozen=True)
class KTrace:
    """单个 k 的递推记录（一致性抽检/日志用）。"""

    k: int
    feasible: bool
    binding_layer: Optional[int]
    exposed_stall_ns: int                # Ŝ(k) = max(0, max(R̂−D̂))
    compute_slowdown_ns: int             # 候选自致计算减速（单列成本）
    timings: Tuple[LayerTiming, ...]


@dataclass(frozen=True)
class KHidePrediction:
    """k̂_hide 递推预测结果（决策披露面）。"""

    k_hide: int
    model_layers: int
    source: str                          # recursion / analytic fallback 状态
    analytic_min_k: Optional[int]        # 解析最小可行 k（None = 不可用）
    statuses: Tuple[str, ...]
    selected_trace: KTrace
    all_traces: Tuple[KTrace, ...]       # 枚举过的 k（抽检/审计）

    def as_dict(self) -> dict:
        return {
            "k_hide": self.k_hide,
            "model_layers": self.model_layers,
            "source": self.source,
            "analytic_min_k": self.analytic_min_k,
            "statuses": list(self.statuses),
            "selected_trace": {
                "k": self.selected_trace.k,
                "feasible": self.selected_trace.feasible,
                "binding_layer": self.selected_trace.binding_layer,
                "exposed_stall_ns": self.selected_trace.exposed_stall_ns,
                "compute_slowdown_ns": (
                    self.selected_trace.compute_slowdown_ns),
            },
        }


class LayerRecursionPredictor:
    """恢复方向的逐层事件递推 + 写回方向合流（§5.2 正式预测器）。

    ``predict_k_hide`` 按钉死剪枝口径枚举；``prune_from_analytic=False``
    供"剪枝枚举 vs 全枚举"一致性抽检（选层结果必须一致——不一致即剪枝
    口径缺陷）。
    """

    def __init__(
        self,
        snapshot: ResourceSnapshot,
        restore_legs: Sequence[RestoreGroupLeg],
        compute_segments: Sequence[ComputeLayerSegment],
        first_block_wait_ns: int = 0,
        efficiency: Optional[EfficiencyFactors] = None,
    ) -> None:
        self.snapshot = snapshot
        self.efficiency = efficiency or EfficiencyFactors()
        self.legs = tuple(restore_legs)
        self.segments = tuple(compute_segments)
        # L2（P2-2）：先校验再 int()——NaN/inf 的 int() 裸 ValueError
        # 逃出调用方只 catch EventRecursionError 的降级通道
        # （face_scheduler 递推入口）；负等待无物理意义同拒。
        if (not math.isfinite(first_block_wait_ns)
                or first_block_wait_ns < 0):
            raise EventRecursionError(
                "first_block_wait_ns must be finite non-negative, got "
                f"{first_block_wait_ns!r}")
        self.first_block_wait_ns = int(first_block_wait_ns)
        if not self.legs and not self.segments:
            raise EventRecursionError(
                "predictor requires restore legs or compute segments")
        if [leg.layer_start for leg in self.legs] != sorted(
                leg.layer_start for leg in self.legs):
            raise EventRecursionError(
                "restore legs must be in consumption order (layer asc)")
        if [seg.layer for seg in self.segments] != sorted(
                seg.layer for seg in self.segments):
            raise EventRecursionError(
                "compute segments must be in layer order")
        # L2（P3 小项）：腿间 rank 数等长——_run_dependency_driven 以
        # leg0 定 rank_count、按 rank_index 索引各腿 bytes_by_rank，
        # 跨腿不一致会裸 IndexError（生产腿恒同 TP 组不可达，防御
        # 纵深与下方 legs-vs-segments 对齐检查同面）。
        if any(len(leg.bytes_by_rank) != len(self.legs[0].bytes_by_rank)
               for leg in self.legs):
            raise EventRecursionError(
                "restore legs must share a common rank count")
        for leg in self.legs:
            for path in leg.path_by_rank:
                for key in path:
                    if key not in snapshot.peak_bytes_per_ns:
                        raise EventRecursionError(
                            f"restore path uses unknown resource {key!r}")
        for seg in self.segments:
            for key in seg.port_by_rank:
                if key not in snapshot.peak_bytes_per_ns:
                    raise EventRecursionError(
                        f"compute port {key!r} missing from snapshot")
        if self.legs and self.segments:
            if len(self.legs[0].bytes_by_rank) != len(
                    self.segments[0].base_ns_by_rank):
                raise EventRecursionError(
                    "compute/restore rank counts must align")

    # ---------------------------------------------------- 依赖驱动递推 --
    def _run_dependency_driven(
        self, cold_legs: Sequence[RestoreGroupLeg],
    ) -> Tuple[Dict[int, int], Dict[int, Dict[int, int]], Dict[int, int]]:
        """统一时间线递推（写腿×计算 memory 腿同仲裁）。

        ``cold_legs=[]`` → reference 递推（无候选恢复流量；输出 = 钉死
        D̂ 的消费完成时刻）。返回 ``(consumption_by_layer,
        restore_done_by_layer_rank, restore_done_by_layer)``。

        依赖结构：恢复腿 rank 内串行链（消费顺序）；计算段 rank-r 腿启
        动 = max(前层 TP 同步, 该 rank 该层所在恢复组完成)；段完成 =
        max(计算腿 γ·base 终点, 端口 memory 流完成)；层消费 = 跨 rank
        max（TP 逐层同步）。
        """
        timeline = UnifiedTimeline(self.snapshot, self.efficiency)
        rank_count = (
            len(self.segments[0].base_ns_by_rank) if self.segments else (
                len(cold_legs[0].bytes_by_rank) if cold_legs else 0))
        base_now = int(self.snapshot.now_ns)

        # ---- 恢复侧状态。gated 链：组 0 起点固定，组 g+1 由前驱完成
        #      事件解锁（步间调度）。
        restore_done: Dict[Tuple[int, int], int] = {}
        by_layer_rank: Dict[int, Dict[int, int]] = {}
        scheduled: Set[Tuple[int, int]] = set()

        def leg_of_layer(layer: int) -> Optional[int]:
            """层 ℓ（1-based）所在冷腿序号；热层 None。"""
            for leg_index, leg in enumerate(cold_legs):
                if leg.layer_start <= layer - 1 < leg.layer_end:
                    return leg_index
            return None

        def schedule_restore(leg_index: int, rank_index: int,
                             start_ns: int) -> str:
            leg = cold_legs[leg_index]
            flow_id = f"restore:g{leg_index}:r{rank_index}"
            timeline.add_flow(
                flow_id, leg.path_by_rank[rank_index],
                int(leg.bytes_by_rank[rank_index]),
                start_ns=start_ns + int(leg.startup_ns))
            scheduled.add((leg_index, rank_index))
            return flow_id

        def drain_restore_completions() -> None:
            """消化已完成恢复流：记 done + 解锁同 rank 后继组。"""
            progressed = True
            while progressed:
                progressed = False
                for leg_index, leg in enumerate(cold_legs):
                    for rank_index in range(len(leg.bytes_by_rank)):
                        key = (leg_index, rank_index)
                        if key in restore_done:
                            continue
                        if leg.bytes_by_rank[rank_index] <= 0:
                            # 零字节腿即时完成（q̂ 时刻就绪）。K4
                            # （P1-⑥，2026-09-23 外部审计）：rank 内
                            # 串行链（消费顺序）对零字节腿同样成立——
                            # 前驱腿未完成则不得先行就绪（否则用该时刻
                            # 解锁后继组，R̂ 乐观低估；g0>0/g1=0/g2>0
                            # 形态实证低估 50%）。
                            if (leg_index > 0
                                    and (leg_index - 1, rank_index)
                                    not in restore_done):
                                continue
                            done = max(
                                int(restore_done[(
                                    leg_index - 1, rank_index)])
                                if leg_index > 0 else base_now,
                                base_now + self.first_block_wait_ns)
                        else:
                            if key not in scheduled:
                                continue
                            done = timeline.completions.get(
                                f"restore:g{leg_index}:r{rank_index}")
                            if done is None:
                                continue
                        restore_done[key] = int(done)
                        for layer in range(leg.layer_start, leg.layer_end):
                            by_layer_rank.setdefault(layer + 1, {})[
                                rank_index] = int(done)
                        nxt = leg_index + 1
                        if (nxt < len(cold_legs)
                                and (nxt, rank_index) not in scheduled
                                and (nxt, rank_index) not in restore_done):
                            if cold_legs[nxt].bytes_by_rank[rank_index] > 0:
                                schedule_restore(
                                    nxt, rank_index,
                                    max(int(done), base_now))
                        progressed = True

        if cold_legs:
            leg0 = cold_legs[0]
            for rank_index in range(len(leg0.bytes_by_rank)):
                if leg0.bytes_by_rank[rank_index] > 0:
                    schedule_restore(
                        0, rank_index,
                        base_now + self.first_block_wait_ns)
        drain_restore_completions()

        # ---- 计算侧：逐层解析（层序 = 消费顺序）。
        consumption: Dict[int, int] = {}
        prev_sync = base_now
        for segment in self.segments:
            layer = segment.layer
            leg_index = leg_of_layer(layer)
            gamma = self.efficiency.gamma_of(segment.group)
            # 各 rank 腿启动 = max(前层同步, 该 rank 恢复门)。
            leg_gate_flow_ids = []
            rank_starts = []
            for rank_index in range(rank_count):
                gate_ns = None
                if leg_index is not None:
                    # 等该 rank 该组恢复完成（步进时间线直到可见）。
                    while (leg_index, rank_index) not in restore_done:
                        drain_restore_completions()
                        if (leg_index, rank_index) in restore_done:
                            break
                        if not timeline.step():
                            raise EventRecursionError(
                                "compute dependency stalled waiting for "
                                f"restore group {leg_index} rank "
                                f"{rank_index} (layer {layer})")
                        drain_restore_completions()
                    gate_ns = restore_done[(leg_index, rank_index)]
                start = prev_sync if gate_ns is None else max(
                    prev_sync, int(gate_ns))
                rank_starts.append(int(start))
                mem_bytes = int(segment.memory_bytes_by_rank[rank_index])
                if mem_bytes > 0:
                    flow_id = f"compute:L{layer}:r{rank_index}"
                    timeline.add_flow(
                        flow_id, (segment.port_by_rank[rank_index],),
                        mem_bytes, start_ns=start)
                    leg_gate_flow_ids.append(flow_id)
            # 推进到本层端口流全部完成（恢复/他流事件沿途如实处理）。
            timeline.run_until_complete(leg_gate_flow_ids)
            drain_restore_completions()
            layer_done = []
            for rank_index in range(rank_count):
                compute_leg_end = (
                    rank_starts[rank_index]
                    + gamma * float(segment.base_ns_by_rank[rank_index]))
                mem_bytes = int(segment.memory_bytes_by_rank[rank_index])
                port_done = (
                    timeline.completions.get(
                        f"compute:L{layer}:r{rank_index}")
                    if mem_bytes > 0 else None)
                assert port_done is not None or mem_bytes == 0
                layer_done.append(int(math.ceil(max(
                    compute_leg_end,
                    port_done if port_done is not None else 0.0,
                    rank_starts[rank_index]))))
            consumption[layer] = max(layer_done)
            prev_sync = consumption[layer]

        # 时间线残余恢复流（比最后计算层更晚完成的冷层）继续推完，供
        # R̂ 全量披露（消费已过——迟到如实，作断供入账）。
        timeline.run()
        drain_restore_completions()
        by_layer: Dict[int, int] = {}
        for layer, per_rank in by_layer_rank.items():
            by_layer[layer] = max(per_rank.values())
        return consumption, by_layer_rank, by_layer

    # ---------------------------------------------------- 解析下界 --
    def exclusive_rank_serial_bound_ns(
        self, k: int, layer: int,
    ) -> Optional[float]:
        """钉死下界（§5.2 剪枝段其一）：max_r [q̂ + Σ_{j=k+1..ℓ} b_{j,r}/B_{j,r}^excl]。

        ``B_{j,r}^excl`` = rank r 恢复层 j 那条腿的路径各资源 ``η·peak``
        取 min（**逐腿**独享峰值服务率——rank 的恢复路径可逐层不同，按
        全腿全局取 min 会在快腿上高估、破坏"该量 ≤ 事件递推的 R̂"的剪
        枝健全性）；不计中间层启动与 rank 间互斥。字节/速率不可用 →
        None（解析不可用，不剪枝）。**不是** ``Σ_j max_r r̂_j`` 逐层近似。
        """
        cold_legs = [leg for leg in self.legs if leg.layer_start >= k]
        if not cold_legs:
            return float(self.first_block_wait_ns)
        rank_count = len(cold_legs[0].bytes_by_rank)
        worst = None
        for rank_index in range(rank_count):
            elapsed = float(self.first_block_wait_ns)
            saw_leg = False
            for leg in cold_legs:
                if leg.layer_start > layer - 1:
                    continue
                path = leg.path_by_rank[rank_index]
                if not path:
                    continue
                saw_leg = True
                exclusive_rate = min(
                    float(self.snapshot.peak_bytes_per_ns[key])
                    * self.efficiency.eta_of(key)
                    for key in path)
                span = leg.layer_end - leg.layer_start
                per_layer_bytes = float(
                    leg.bytes_by_rank[rank_index]) / span
                layers_in_leg = min(leg.layer_end, layer) - leg.layer_start
                elapsed += per_layer_bytes * layers_in_leg / exclusive_rate
            if not saw_leg:
                return None
            worst = elapsed if worst is None else max(worst, elapsed)
        return worst

    # ---------------------------------------------------- 主入口 --
    def reference_deadlines(self) -> Tuple[Dict[int, int], Dict[int, int]]:
        """D̂ 钉死口径：无候选恢复流量的消费期限（§5.2 剪枝段其二）。

        D̂_ℓ = 该层**开始**消费的时刻 = reference 递推中前层完成时刻
        （``D̂_ℓ = Σ_{j<ℓ} ĉ_j`` 的递推化：保守以该层开始为期限；
        层 1 的期限 = 快照 now）。同时返回 reference 消费完成时刻（候
        选自致计算减速的对比锚——减速单列、不回移期限）。
        """
        consumption, _rank, _layer = self._run_dependency_driven([])
        base = int(self.snapshot.now_ns)
        deadlines: Dict[int, int] = {}
        previous = base
        for layer in sorted(consumption):
            deadlines[layer] = previous
            previous = int(consumption[layer])
        return deadlines, dict(consumption)

    def predict_k_hide(
        self, *, prune_from_analytic: bool = True,
        analytic_min_k=_AUTO_ANALYTIC,
    ) -> KHidePrediction:
        """枚举 k̂_hide（钉死剪枝口径）。

        ``prune_from_analytic=False`` = 全枚举（一致性抽检）。
        ``analytic_min_k=None`` = 调用方声明解析校验不可用/未给出可行
        k——不剪枝、自 0 全枚举并标注 ``analytic_unavailable``（来源降
        级 fallback；不当零代价、不删实例候选）。缺省 = 自算解析下界。
        """
        statuses = list(self.snapshot.coverage_statuses())
        if self.snapshot.unknown_eta_flow_ids():
            statuses.append(STATUS_UNKNOWN_RELEASE_ETA)
        deadlines, reference_consumption = self.reference_deadlines()
        model_layers = (
            self.segments[-1].layer if self.segments else
            (self.legs[-1].layer_end if self.legs else 0))

        def lower_bound_feasible(k: int) -> Optional[bool]:
            for layer in range(k + 1, model_layers + 1):
                bound = self.exclusive_rank_serial_bound_ns(k, layer)
                if bound is None:
                    return None
                deadline = deadlines.get(layer)
                if deadline is None:
                    return None
                if bound > deadline + _FEASIBILITY_EPSILON_NS:
                    return False
            return True

        analytic_usable = True
        if analytic_min_k is _AUTO_ANALYTIC:
            self_computed: Optional[int] = None
            for k in range(0, model_layers + 1):
                feasible = lower_bound_feasible(k)
                if feasible is None:
                    self_computed = None
                    break
                if feasible:
                    self_computed = k
                    break
            analytic_min_k = self_computed
            analytic_usable = self_computed is not None
        elif analytic_min_k is None:
            # 调用方显式声明解析校验不可用（缺省自算路径不含此态——
            # 构造期已 fail-closed 拒绝缺速率/空路径腿）。
            analytic_usable = False
        else:
            analytic_min_k = int(analytic_min_k)
            analytic_usable = True
        if not analytic_usable:
            analytic_min_k = None
            if prune_from_analytic:
                statuses.append(STATUS_ANALYTIC_UNAVAILABLE)
        start_k = (
            int(analytic_min_k)
            if (prune_from_analytic and analytic_min_k is not None) else 0)
        traces: List[KTrace] = []
        selected: Optional[KTrace] = None
        for k in range(start_k, model_layers + 1):
            trace = self._trace_for_k(k, deadlines, reference_consumption)
            traces.append(trace)
            if trace.feasible:
                selected = trace
                break
        if selected is None:
            raise EventRecursionError(
                "k enumeration failed to terminate (k=L must be feasible)")
        source = (
            PREDICTOR_SOURCE_RECURSION if analytic_usable
            else PREDICTOR_SOURCE_ANALYTIC_FALLBACK)
        return KHidePrediction(
            k_hide=selected.k,
            model_layers=model_layers,
            source=source,
            analytic_min_k=analytic_min_k,
            statuses=tuple(dict.fromkeys(statuses)),
            selected_trace=selected,
            all_traces=tuple(traces),
        )

    def _trace_for_k(
        self, k: int, deadlines: Mapping[int, int],
        reference_consumption: Mapping[int, int],
    ) -> KTrace:
        cold_legs = [leg for leg in self.legs if leg.layer_start >= k]
        consumption, _rank, by_layer = self._run_dependency_driven(cold_legs)
        # 候选自致计算减速 = 含候选恢复流的消费完成 − reference 消费完成
        # （逐层单列成本记账，不回移/不放松期限）。
        timings: List[LayerTiming] = []
        worst_over = 0
        binding = None
        slowdown = 0
        for layer in sorted(set(deadlines) | set(consumption)):
            restore_done = by_layer.get(layer)
            deadline = int(deadlines.get(layer, 0))
            timings.append(LayerTiming(
                layer=layer,
                restore_complete_ns=restore_done,
                deadline_ns=deadline,
                consumption_ns=int(consumption.get(layer, deadline)),
            ))
            if restore_done is not None:
                over = restore_done - deadline
                if over > worst_over:
                    worst_over = over
                    binding = layer
            actual = consumption.get(layer)
            reference = reference_consumption.get(layer)
            if actual is not None and reference is not None:
                slowdown += max(0, actual - reference)
        return KTrace(
            k=k,
            feasible=worst_over <= 0,
            binding_layer=binding,
            exposed_stall_ns=max(0, worst_over),
            compute_slowdown_ns=slowdown,
            timings=tuple(timings),
        )


# ==================================================== 写回方向（逐出侧） ==


@dataclass(frozen=True)
class VictimSpacePrediction:
    """单 victim 的写回递推结果。"""

    victim_id: str
    space_available_ns: int               # 写回流完成（写回未完成不算释放）
    per_rank_complete_ns: Tuple[int, ...]
    writeback_bytes_by_rank: Tuple[int, ...]


@dataclass(frozen=True)
class ReleaseSpacePrediction:
    """逐对象顺序的写回方向预测（同批 victim 共享快照推演）。"""

    victims: Tuple[VictimSpacePrediction, ...]
    statuses: Tuple[str, ...]

    def as_dict(self) -> dict:
        return {
            "victims": [
                {
                    "victim_id": v.victim_id,
                    "space_available_ns": v.space_available_ns,
                    "per_rank_complete_ns": list(v.per_rank_complete_ns),
                }
                for v in self.victims],
            "statuses": list(self.statuses),
        }


def predict_space_available(
    snapshot: ResourceSnapshot,
    victims: Sequence[WritebackVictim],
    efficiency: Optional[EfficiencyFactors] = None,
) -> ReleaseSpacePrediction:
    """写回方向递推：对象顺序逐 victim 决策，已定写回流登记进共享快照。

    同批 victim **不得**各自按独享带宽估算——victim v 的推演快照包含
    v 之前全部已定 victim 的写回流（在册、ETA = 其预测完成时刻），共
    享链路/端口在统一仲裁内竞争。
    """
    efficiency = efficiency or EfficiencyFactors()
    # 共享快照：仅累积**已定 victim 的候选写回流**（在册、ETA = 其预
    # 测完成时刻）——原快照的在册流经 with_committed 追加，不得重复拼。
    committed_extra: List[CommittedFlow] = []
    statuses = list(snapshot.coverage_statuses())
    if snapshot.unknown_eta_flow_ids():
        statuses.append(STATUS_UNKNOWN_RELEASE_ETA)
    results: List[VictimSpacePrediction] = []
    for victim in victims:
        for path in victim.path_by_rank:
            for key in path:
                if key not in snapshot.peak_bytes_per_ns:
                    raise EventRecursionError(
                        f"writeback path uses unknown resource {key!r}")
        working = snapshot.with_committed(committed_extra)
        timeline = UnifiedTimeline(working, efficiency)
        for rank_index, path in enumerate(victim.path_by_rank):
            timeline.add_flow(
                f"writeback:{victim.victim_id}:r{rank_index}",
                path, int(victim.bytes_by_rank[rank_index]),
                start_ns=int(snapshot.now_ns) + int(victim.startup_ns))
        timeline.run()
        per_rank = tuple(
            timeline.completions.get(
                f"writeback:{victim.victim_id}:r{index}",
                int(snapshot.now_ns) + int(victim.startup_ns))
            for index in range(len(victim.path_by_rank)))
        results.append(VictimSpacePrediction(
            victim_id=victim.victim_id,
            space_available_ns=max(per_rank, default=int(snapshot.now_ns)),
            per_rank_complete_ns=per_rank,
            writeback_bytes_by_rank=tuple(victim.bytes_by_rank),
        ))
        for rank_index, path in enumerate(victim.path_by_rank):
            committed_extra.append(CommittedFlow(
                flow_id=f"writeback:{victim.victim_id}:r{rank_index}",
                resources=path,
                release_eta_ns=int(per_rank[rank_index]),
            ))
    return ReleaseSpacePrediction(
        victims=tuple(results), statuses=tuple(dict.fromkeys(statuses)))


# ============================================== 写回×恢复合流（同推演） ==


@dataclass(frozen=True)
class ReleaseRecallPrediction:
    """两方向合流结果：写回侧空间可用 + 恢复侧 k̂_hide（同一仲裁内）。"""

    space: ReleaseSpacePrediction
    k_hide: KHidePrediction
    first_block_wait_ns: int              # 合流后的 q̂（关键路径 max）


def predict_release_and_recall(
    snapshot: ResourceSnapshot,
    victims: Sequence[WritebackVictim],
    restore_legs: Sequence[RestoreGroupLeg],
    compute_segments: Sequence[ComputeLayerSegment],
    *,
    first_block_wait_ns: int = 0,
    efficiency: Optional[EfficiencyFactors] = None,
    prune_from_analytic: bool = True,
) -> ReleaseRecallPrediction:
    """写回与恢复两方向同推演仲裁（§5.2 递推段的合流入口）。

    先按对象顺序定 victim 写回流（空间可用时刻），再把已定写回流登记
    进共享快照、以 ``q̂ = max(首块基础等待, 空间可用时刻)`` 的接收空间
    依赖跑恢复方向 k 递推——恢复流与写回流在同一资源仲裁内竞争。
    """
    efficiency = efficiency or EfficiencyFactors()
    # L2（P2-2）：合流入口与构造器同一先校验后转换纪律——int(NaN)
    # 裸 ValueError 逃出 EventRecursionError 降级通道。
    if (not math.isfinite(first_block_wait_ns)
            or first_block_wait_ns < 0):
        raise EventRecursionError(
            "first_block_wait_ns must be finite non-negative, got "
            f"{first_block_wait_ns!r}")
    space = predict_space_available(snapshot, victims, efficiency)
    committed_extra: List[CommittedFlow] = []
    for victim in space.victims:
        victim_spec = next(
            v for v in victims if v.victim_id == victim.victim_id)
        for rank_index, eta in enumerate(victim.per_rank_complete_ns):
            committed_extra.append(CommittedFlow(
                flow_id=f"writeback:{victim.victim_id}:r{rank_index}",
                resources=victim_spec.path_by_rank[rank_index],
                release_eta_ns=int(eta),
            ))
    merged_wait = int(first_block_wait_ns)
    if space.victims:
        merged_wait = max(
            merged_wait,
            max(v.space_available_ns for v in space.victims))
    working = snapshot.with_committed(committed_extra)
    predictor = LayerRecursionPredictor(
        working, restore_legs, compute_segments,
        first_block_wait_ns=merged_wait,
        efficiency=efficiency)
    k_result = predictor.predict_k_hide(
        prune_from_analytic=prune_from_analytic)
    return ReleaseRecallPrediction(
        space=space, k_hide=k_result,
        first_block_wait_ns=merged_wait)


# ================================================== 重试复核（§5.5/§5.6） ==


RESUBMIT_SAME = "resubmit_same"
INVALIDATE_LAYER_IN_USE = "invalidate_layer_in_use"
INVALIDATE_INTERVAL_RELEASED = "invalidate_interval_released"


@dataclass(frozen=True)
class CommittedPlanView:
    """已提交逐出拆分的只读复核视图（§5.6：提交后仅保护/合法性复核）。"""

    victim_id: str
    layer_start: int                     # 0-based，含
    layer_end: int                       # 0-based，不含

    def covers_layer(self, layer: int) -> bool:
        return self.layer_start <= layer < self.layer_end


def revalidate_committed_plan(
    plan_steps: Sequence[CommittedPlanView],
    *,
    layers_now_in_use_or_inflight: Iterable[int],
    still_valid_interval: bool,
) -> str:
    """已提交计划的重试复核断言口径（§5.5 末段/§5.6）。

    已提交逐出拆分不因流量观测重开（决策落定）；重试重交**同一拆分**，
    仅做保护/合法性复核：相关层是否转为在用/在途、区间是否仍有效。
    仅当相关层转在用/在途或区间失效才作废（按触发点重新规划 = 新决
    策）。快照新鲜度复核只存在于未提交计划的首次提交前（调用方保证
    ——本函数不重推演、不读流量观测）。
    """
    steps = tuple(plan_steps)
    if not steps:
        return RESUBMIT_SAME
    if not still_valid_interval:
        return INVALIDATE_INTERVAL_RELEASED
    locked = set(int(layer) for layer in layers_now_in_use_or_inflight)
    for step in steps:
        for layer in range(step.layer_start, step.layer_end):
            if layer in locked:
                return INVALIDATE_LAYER_IN_USE
    return RESUBMIT_SAME


# ================================================== 在线效率状态（η/γ） ==


class ServiceFactorGroup:
    """η/γ 估计组集合（§5.4 因果更新；LEP EwmaServiceFactor 的组容器）。

    每组一个 ``EwmaServiceFactor``（同更新规则：θ、α=1−exp(−Δt/τ)、τ=
    上一有效样本正服务时长、首样本初始化、无效样本拒绝并记录）。有效
    服务样本剔除排队与冷 KV 等待（``observe_valid_service`` 只接收可分
    离样本；受混合等待污染的墙钟**不得**入样——调用方用
    ``mark_unobservable`` 记录并保持原估计）。逐扫描点重置 = 新实例；
    同 run 负载升降持续更新、不重置。
    """

    def __init__(self, initial: float = 1.0) -> None:
        self._initial = float(initial)
        self._factors: dict = {}
        self.unobservable_samples: dict = {}
        self.cold_start = True

    def _group(self, name: str):
        from joint.layer_eviction_policy import EwmaServiceFactor
        if name not in self._factors:
            self._factors[name] = EwmaServiceFactor(self._initial)
        return self._factors[name]

    def observe_valid_service(
        self,
        group: str,
        *,
        observed_ratio: float,
        completion_ns: int,
        service_duration_ns: int,
    ) -> float:
        """一个可分离的有效服务样本（比值 = 实际量/基础模型预测量——
        预测分母不含修正因子，避免循环校正）。K8（2026-09-23 外部
        审计）：cold_start 只在样本**被接受**后翻转（组内拒绝不 raise
        而早退返回——以样本计数判接受；原实现在 update 前翻转，被拒
        样本会提前结束冷启动态）。"""
        factor = self._group(group)
        samples_before = factor.samples
        value = factor.update(
            observed_ratio=observed_ratio,
            completion_ns=completion_ns,
            service_duration_ns=service_duration_ns)
        if factor.samples > samples_before:
            self.cold_start = False
        return value

    def mark_unobservable(self, group: str, reason: str = "mixed_wait") -> None:
        """缺可分离样本：保持原估计并标记不可观测（§5.4）。"""
        self.unobservable_samples[group] = (
            self.unobservable_samples.get(group, 0) + 1)

    def value(self, group: str) -> float:
        factor = self._factors.get(group)
        return factor.value if factor is not None else self._initial

    def rejected_updates(self, group: str) -> Tuple[str, ...]:
        factor = self._factors.get(group)
        return tuple(factor.rejected_updates) if factor else ()

    def snapshot(self) -> dict:
        return {
            "cold_start": self.cold_start,
            "groups": {
                name: {
                    "value": self.value(name),
                    "samples": (
                        self._factors[name].samples
                        if name in self._factors else 0),
                    "unobservable_samples": (
                        self.unobservable_samples.get(name, 0)),
                }
                for name in sorted(
                    set(self._factors) | set(self.unobservable_samples))},
        }
