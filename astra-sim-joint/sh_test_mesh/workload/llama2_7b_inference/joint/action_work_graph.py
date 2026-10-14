"""action_work_graph.py -- W3：三动作预测工作图构造（2026-09-29）。

设计依据：《joint仓库当下+历史修改方案》§5.2（工作图构造与预测范围）、
§5.1a（逐资源服务语义——经 W2 ``resource_projection`` 内核兑现）、
§6.2（动作共同构造规则）、§6.3（A/B/C 例字节核账）、§7.1（两决策点
独立）、§7.2（旧义务真实依赖）、§7.3（consolidate 结算）；W3 段
（方案 §6.4）。盘点 inv_w23.md §2/§6.3/§6.4/§6.5。

职责：从（目录投影布局 = SessionKVView / PDBoundaryView→BoundaryKVLayout）
+（背景世界源 WorldSource）+（因果长度接口供给的剩余步数 n̂）构造
stay/recompute/copy 三动作与参考接续的**预测工作图**，输出
``resource_projection`` 的 WorkItem/TaskSpec 形态。**不提交真实图、
不改真实 home/预约**（§7.1：P 参考接续只在预测副本；选中后仅提交
P 必需工作）。

WorldSource 抽象（§5.2"所有候选共享快照"；§4.4"每次决策构造一个只读
背景结构"）：

* ``LoadsWorldSource``（正式过渡源）：现有 ``InstanceLoadView`` 服务台账
  + 链路/HBM/池流注册表 → 背景工作，``progress_source="inferred"``——
  注册表是计数不是每流剩余字节账本（方案 §4.4：计数只能作带不确定性
  标记的近似），在册流按 unknown-ETA 永久持有语义入背景（W2 内核以
  in-service 长腿承载；ERP ``unknown_release_eta`` 同族）。
* ``LedgerWorldSource``（占位）：W1 ``work_ledger/runtime_snapshot`` 接线
  后由主 Agent 替换——本包只留接口与 fail-closed 降级（**绝不**静默
  返回空背景：无剩余量账本的乐观计价比显式失败更危险）。

与旧解析路径的数值关系（对拍基线）：不受控夹具（空注册表/宽裕容量）下
事件图与旧闭式逐位一致——传输腿 = 单 stream 腿 route=(逐跳 link…,
hbm:src, hbm:dst)（W2 ``min_e(peak/n_e)`` ≡ 旧三腿 ``max(noc, home_read,
exec_write)`` 同字节过全资源）、``latency_ns`` ≡ 旧 λ（逐跳/池/启动，
latency 期不占分母）、ceil 出口同向；计算腿同时携带纯 FLOP 下限与 HBM
字节工作，后者随事件活跃集动态重仲裁；Q = 计算腿 ready_at
（式 13 F_0 = Q 的事件形态）；A（容量摘要 θ 估计，§5.5 估计义务）+
prep_frontier → 定长 marker 腿门控写腿；TP = 逐层 ``BarrierSpec``。

字节守恒锚（§6.3）：``decode_context_weighted_tokens``（式 18b）保留为
核账闭式；``copy_effective_bytes = Σmissing`` 经 audits 披露——这是
有效载荷核账，不是完成时间公式。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from joint.prediction_config import (
    PROJECTION_MODE_FROZEN_DIAGNOSTIC,
    SETTLEMENT_DISTRIBUTED,
    PredictionParams,
)
from joint.resource_projection import (
    BarrierSpec,
    GateLease,
    GateSpec,
    ProjectionEfficiency,
    ProjectionInput,
    ProjectionResult,
    ResourceSpec,
    RESOURCE_KIND_HBM,
    RESOURCE_KIND_LINK,
    RESOURCE_KIND_POOL,
    TASK_KIND_COMPUTE,
    TASK_KIND_FIXED,
    TASK_KIND_STREAM,
    WorkItem,
)
from joint.work_ledger import CommittedWork, WorkAmount, WorkIdentity


class ActionWorkGraphError(ValueError):
    """fail-closed：图构造非法输入/未实现结算策略/世界源降级拒绝。"""


#: unknown-ETA 在册流的永久持有哨兵（ns）。**语义常量非调参常数**：注册
#: 表只给计数不给剩余量（§4.4），在册流在整个预测时域内不离开资源活跃
#: 集——与旧"冻结除数整段"同向保守；W1 账本接线后由真实 remaining_bytes
#: 取代（LedgerWorldSource）。
UNKNOWN_ETA_HOLD_NS = 10 ** 15

#: 背景进度来源标签（§4.4 不确定性标记）。
PROGRESS_SOURCE_INFERRED = "inferred"
PROGRESS_SOURCE_LEDGER = "ledger"
PROGRESS_SOURCE_DEGRADED = "degraded"

#: 结算策略注记（W9，§7.4）：distributed settle 腿零载荷的 notes 披露键。
SETTLEMENT_DISTRIBUTED_NOTE = (
    "settlement=distributed:directory-commit-no-merge-payload")


def link_resource_key(source_rank: int, target_rank: int) -> str:
    """有向链路资源键（W2 kind 前缀对齐：``link:{src}->{dst}``）。"""
    return f"link:{int(source_rank)}->{int(target_rank)}"


def hbm_resource_key(rank: int) -> str:
    """单 rank aggregate HBM 总线资源键（读写共用分母，§5.1a）。"""
    return f"hbm:{int(rank)}"


def pool_resource_key(port: int) -> str:
    """池边缘端口资源键（latency 后入流集合、N-way 共享）。"""
    return f"pool:{int(port)}"


def noc_route_keys(path: Sequence[int]) -> Tuple[str, ...]:
    """路径的逐跳有向链路键序列（store-and-forward 跳数线性时延同源）。"""
    ranks = [int(rank) for rank in path]
    return tuple(
        link_resource_key(ranks[index], ranks[index + 1])
        for index in range(len(ranks) - 1))


# ---------------------------------------------------------------- 义务面 --


@dataclass(frozen=True)
class ObligationView:
    """§7.2 继承义务的最小解析面（ID → 图节点参竞争）。

    ``done`` = 已完成 P 义务（零腿，不再收费）；未完义务以
    ``remaining_bytes_by_tp_rank`` 参与背景竞争，``gate_settle`` 时其完成
    还门控候选 settle（"即使不门控 D 计算，也可能门控释放或 settle"）。
    生产来源 = sh30 边界义务表（``InheritedObligation`` 账本对象）；无
    ledger 时调用方不提供解析器 ⇒ 降级披露（不虚构义务腿）。
    """

    obligation_id: str
    kind: str
    source_instance: Optional[int] = None
    target_instance: Optional[int] = None
    remaining_bytes_by_tp_rank: Tuple[int, ...] = ()
    gate_settle: bool = True
    done: bool = False

    def __post_init__(self) -> None:
        if not str(self.obligation_id).strip():
            raise ActionWorkGraphError(
                "obligation id must be non-empty")


# -------------------------------------------------------------- 世界源 --


@dataclass(frozen=True)
class WorldContext:
    """决策级只读背景结构（§4.4：一次构造、候选间共享、增量副本隔离）。

    ``approximations`` 携带背景构造的不确定性标记（inferred 计数、
    unknown-ETA 持有等）；``member_count`` 供 frozen_diagnostic A 臂读取
    决策 tick 活跃集基数（§9.1 冻结份额规则）。

    W7（future_demand 消费钩子，最小面）：``forecast_works`` /
    ``forecast_arrival_placeholders_ns`` / ``forecast_barriers`` 携带
    新增需求预测注入的背景模板（``joint.background_forecast`` 构造；
    所有候选共享同一情景）。三者缺省空 ⇒ ``projection_input`` 输出与
    W6 世代逐字段一致（none 档零漂移锚）。``forecast_scenario_key`` 进
    approximations 披露并作 JCM 快照缓存键分量（模式+统计版本变化 ⇒
    重建，§8.6）。
    """

    now_ns: int
    resources: Tuple[ResourceSpec, ...]
    background_works: Tuple[WorkItem, ...]
    efficiency: ProjectionEfficiency
    approximations: Tuple[str, ...]
    progress_source: str
    forecast_works: Tuple[WorkItem, ...] = ()
    forecast_arrival_placeholders_ns: Tuple[int, ...] = ()
    forecast_barriers: Tuple[BarrierSpec, ...] = ()
    forecast_scenario_key: str = ""

    def _member_counts(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for work in self.background_works:
            for leg in work.legs:
                for key in leg.route:
                    counts[key] = counts.get(key, 0) + 1
        return counts

    def projection_input(
        self,
        candidate: "CandidateGraph",
        *,
        frozen_resource_keys: Optional[Mapping[str, float]] = None,
    ) -> ProjectionInput:
        """叠加候选（含其私有资源/门/barrier）构造前推输入。

        ``frozen_resource_keys``（A 臂）：决策 tick 活跃集冻结的份额实现
        ——本层预冻结（W2 内核零侵入）：背景工作移出、命中的资源峰值
        按冻结除数折减（候选在该资源上的速率恒 = peak/n_e，不随事件重
        分配——per_action 时域，FrozenDiagnosticSpec 冻结规则）。

        W7：受控图验证可将预测背景模板并入 ``background_works``（同真实
        背景参与资源竞争与候选反作用——候选腿占用拖慢预测腿、预测腿拖慢
        候选，FORECAST_ARRIVAL 到达占位触发重仲裁）；其 barrier（模板 TP
        同步）与到达占位（重仲裁 tick）随输入传递。生产
        ``ForecastAugmentedWorldSource`` 对未生成候选物理状态恒走
        ``physical_world_error`` shadow 路径，不把这些模板作为正式 future
        price 世界启用。

        §8.4 名实边界（2026-09-30 修复订正，勿再据本接线宣称已满足）：
        "模板在注入时读取**该候选预测世界内**的目录/容量/已发生驱逐，
        再按同一因果规则生成访问/恢复"**尚未实现**——当前注入物为逐决
        策冻结的共享计划（放置实例按内源背景腿数一次选定、恢复量 =
        preset 常数、池路由 = rank 代理），对所有候选逐字节固定，仅以
        uncertainty 标签披露（RESTORE_PRESET/KV_LOCATION_UNKNOWN/
        POOL_PORT_PROXY）；本层实现的是共享资源竞争耦合，非候选级目录派生。
        生产候选级只读面（KV 目录投影/容量摘要/驱逐状态的候选内重派放置
        与恢复量）落地前，A18 的"依其预测目录生成合法恢复/访问"不得声称
        达成，正式 future price 继续 shadow。
        """
        resources: Dict[str, ResourceSpec] = {
            spec.key: spec for spec in self.resources}
        for spec in candidate.resources:
            resources.setdefault(spec.key, spec)
        background = list(self.background_works)
        if frozen_resource_keys:
            frozen = set(frozen_resource_keys)
            resources = {
                key: (
                    ResourceSpec(
                        key=key, kind=spec.kind,
                        capacity_per_ns=spec.capacity_per_ns
                        / max(1.0, float(frozen_resource_keys[key])))
                    if key in frozen else spec)
                for key, spec in resources.items()}
            # 冻结份额规则 = 背景不随事件重分配 ⇒ 背景工作不再入图
            # （其占用已折入资源峰值折减；候选反作用面随之关闭——A 臂
            # 诊断语义，§9.1"A 对照须在 manifest 明确冻结哪些竞争集合"）。
            background = [
                work for work in background
                if not any(
                    key in frozen
                    for leg in work.legs for key in leg.route)]
        background.extend(candidate.extra_background_works)
        background.extend(self.forecast_works)
        if not background:
            # 空背景锚：focus 必须在背景集合内（W2 ProjectionInput 契约）
            # ——零时长 fixed 腿，t=0 完成，不产生任何资源交互。
            background.append(WorkItem(
                work_id="world:anchor",
                legs=(
                    _fixed_leg("world:anchor#anchor", 0.0),),
            ))
        gates = tuple(candidate.gates)
        barriers = tuple(candidate.barriers) + tuple(self.forecast_barriers)
        return ProjectionInput(
            now_ns=self.now_ns,
            resources=tuple(resources.values()),
            background_works=tuple(background),
            focus_work_id=background[0].work_id,
            gates=gates,
            barriers=barriers,
            efficiency=self.efficiency,
            forecast_arrival_placeholders_ns=(
                self.forecast_arrival_placeholders_ns),
        )

    # --------------------------------------------- v5 共享快照形态 --

    def shared_projection_input(self) -> ProjectionInput:
        """决策级共享前推输入（v5 P 主线：一次构造、候选间共享）。

        与 :meth:`projection_input` 的关系：本方法产出**不含任何候选私有
        附加**的共享底座（资源/背景/forecast barrier/效率表），候选的
        resources/gates/barriers/义务背景经 :meth:`candidate_overlay`
        以逐调用 ``private_overlay`` 传入原生侧，由原生按与旧逐候选快照
        逐字段等价的规则合并（见 NativeProjection.cc
        apply_projection_overlay）。空背景锚规则同样移入 overlay
        （共享背景与 extras 皆空时原生补锚）。
        """
        background = list(self.background_works) + list(self.forecast_works)
        return ProjectionInput(
            now_ns=self.now_ns,
            resources=tuple(self.resources),
            background_works=tuple(background),
            focus_work_id=(background[0].work_id if background
                           else "world:anchor"),
            gates=(),
            barriers=tuple(self.forecast_barriers),
            efficiency=self.efficiency,
            forecast_arrival_placeholders_ns=(
                self.forecast_arrival_placeholders_ns),
            forecast_start_index=len(self.background_works),
        )

    def candidate_overlay(self, candidate: "CandidateGraph") -> Dict:
        """候选私有 overlay（v5 共享快照形态的逐调用增量）。

        与 projection_input 的候选附加逐字段对应：setdefault 语义的资源
        增量（只携带共享底座中不存在的键，先来先占序保留）、候选 gates、
        候选 barriers（原生侧排在共享 forecast barriers 之前）、§7.2 义
        务背景 works、以及"共享背景与 extras 皆空时补 world:anchor 锚"
        的旗标。
        """
        shared_keys = {spec.key for spec in self.resources}
        added_resources = tuple(
            spec for spec in candidate.resources
            if spec.key not in shared_keys)
        needs_anchor = (
            not self.background_works
            and not self.forecast_works
            and not candidate.extra_background_works)
        return {
            "resources": added_resources,
            "gates": tuple(candidate.gates),
            "barriers": tuple(candidate.barriers),
            "extra_background_works": tuple(
                candidate.extra_background_works),
            "append_background_anchor": needs_anchor,
        }

    def shared_native_projection_context(self, native_library,
                                         kernel_contract):
        """懒构造并缓存决策级共享原生投影上下文（v5 P 主线）。

        生命周期：随本 WorldContext（决策级）存活；调度侧在决策收尾
        （``_projection_arbitrate`` finally，经候选 ``projection_context``
        引用）统一 close。E0 K=1 复询复用同一上下文（不同 options 进
        入不同 reference 缓存键）。
        """
        context = getattr(self, "_shared_native_projection_context", None)
        if context is None or getattr(context, "_closed", False):
            # 决策收尾 close 后同一 WorldContext 若被再次使用（验收
            # harness 的跨帧复用形态），按当次输入重建——输入本身不可变，
            # 重建结果逐字节一致。
            context = native_library.create_context(
                self.shared_projection_input(), kernel_contract)
            object.__setattr__(
                self, "_shared_native_projection_context", context)
        return context


class WorldSource:
    """背景世界源抽象（§5.2 共享快照；W1 接线前 LoadsWorldSource 过渡）。"""

    progress_source = "abstract"

    def build_context(self, model) -> WorldContext:
        raise NotImplementedError

    def scenario_key(self) -> str:
        """快照缓存键分量（W7：模式+统计版本变化 ⇒ 世界重建；缺省空）。"""
        return ""


def _world_projection_efficiency(model) -> ProjectionEfficiency:
    """Return the formal stream efficiency for a model world.

    ``ServiceFactors.transfer_factor`` is an observed transfer time multiplier
    (``actual / baseline``), while :class:`ProjectionEfficiency` stores the
    corresponding rate multiplier ``eta``.  Keep that conversion at the
    WorldContext boundary so every candidate and every background stream uses
    the same resource service rule.  Prefill/decode factors are deliberately
    absent here: ``_GraphBuilder`` already applies them to the pure compute
    floor, and putting them in ``gamma`` would charge them twice.

    The telemetry sample is a link service observation, so it applies to NoC
    link resources only.  HBM and pool resources retain their own physical
    peak rates until a resource-specific efficiency sample exists.
    """
    factors = getattr(model, "service_factors", None)
    transfer_factor = getattr(factors, "transfer_factor", 1.0)
    try:
        transfer_factor = float(transfer_factor)
    except (TypeError, ValueError) as exc:
        raise ActionWorkGraphError(
            "service_factors.transfer_factor must be finite positive"
        ) from exc
    if not math.isfinite(transfer_factor) or transfer_factor <= 0:
        raise ActionWorkGraphError(
            "service_factors.transfer_factor must be finite positive")
    eta = 1.0 / transfer_factor
    if math.isclose(eta, 1.0, rel_tol=0.0, abs_tol=0.0):
        return ProjectionEfficiency()
    return ProjectionEfficiency(eta={RESOURCE_KIND_LINK: eta})


class LoadsWorldSource(WorldSource):
    """现有视图/注册表 → 背景工作（progress_source=inferred，§4.4）。

    * 链路：``_noc_registry``（遥测流数口径 max(在册, 遥测)，D13/M1）
      每有向链路 n 条在册流 ⇒ n 条 unknown-ETA in-service 腿；
    * 池端口：``pool_edge_flows_fn(edge)`` 在册在途计数同上；
    * HBM rank：``hbm_port_registry.divisor(rank)``（活跃 decode 消费流
      + 在册传输流，不含候选）同上——load ns 台账**不**再造 HBM
      等价字节流（与 registry 双计；其计算阻塞由 Q 门 = 计算腿 ready_at
      承载，D19 等待接口拆分同口径）。
    """

    progress_source = PROGRESS_SOURCE_INFERRED

    def build_context(self, model) -> WorldContext:
        rates = model.rates
        resources: List[ResourceSpec] = []
        works: List[WorkItem] = []
        approximations: List[str] = []

        # -- HBM rank：全部实例的 TP rank（instance_ranks_fn 或相对序）。
        hbm_counts: Dict[int, int] = {}
        for instance_index in sorted(model.loads):
            for rank in model._exec_ranks(instance_index):
                divisor = model._hbm_background_divisor(rank)
                if divisor > 0:
                    hbm_counts[int(rank)] = max(
                        hbm_counts.get(int(rank), 0), int(divisor))
        for rank in sorted(hbm_counts):
            resources.append(ResourceSpec(
                key=hbm_resource_key(rank), kind=RESOURCE_KIND_HBM,
                capacity_per_ns=float(rates.local_hbm_bytes_per_ns)))
            hold_bytes = _unknown_eta_bytes(
                float(rates.local_hbm_bytes_per_ns))
            works.append(WorkItem(
                work_id=f"bg:hbm:{rank}",
                legs=tuple(
                    _stream_leg(
                        f"bg:hbm:{rank}#f{index}",
                        route=(hbm_resource_key(rank),),
                        bytes_amount=hold_bytes,
                        remaining_bytes=hold_bytes, in_service=True)
                    for index in range(hbm_counts[rank]))))
        if hbm_counts:
            approximations.append(
                "hbm_registry_counts_inferred_unknown_eta")

        # -- 链路：注册表在册/遥测流数（有向）。
        link_counts: Dict[Tuple[int, int], int] = {}
        for link in model._background_links():
            count = model._link_background_members(link)
            if count > 0:
                link_counts[(int(link[0]), int(link[1]))] = int(count)
        for (src, dst) in sorted(link_counts):
            resources.append(ResourceSpec(
                key=link_resource_key(src, dst), kind=RESOURCE_KIND_LINK,
                capacity_per_ns=float(rates.noc_link_bytes_per_ns)))
            hold_link = _unknown_eta_bytes(
                float(rates.noc_link_bytes_per_ns))
            works.append(WorkItem(
                work_id=f"bg:link:{src}->{dst}",
                legs=tuple(
                    _stream_leg(
                        f"bg:link:{src}->{dst}#f{index}",
                        route=(link_resource_key(src, dst),),
                        bytes_amount=hold_link,
                        remaining_bytes=hold_link, in_service=True)
                    for index in range(link_counts[(src, dst)]))))
        if link_counts:
            # 不确定性标记（§4.4：计数只能作带不确定性标记的近似）：遥测
            # 视图的 max(在册, 遥测) 口径除数 = 窗口时间加权均值合成——
            # "窗口均值 ≠ 决策瞬时"的不确定性进标签（A03 条件保留的对端
            # 披露义务），与纯注册表计数口径分列。
            if getattr(
                    getattr(model, "_noc_registry", None),
                    "divisor_uncertainty_active", False):
                approximations.append(
                    "link_registry_telemetry_max_window_mean"
                    "_inferred_unknown_eta")
            else:
                approximations.append(
                    "link_registry_counts_inferred_unknown_eta")

        # -- 池端口：在册在途计数。
        pool_counts: Dict[int, int] = {}
        for port in model._background_pool_ports():
            count = model._pool_background_members(port)
            if count > 0:
                pool_counts[int(port)] = int(count)
        for port in sorted(pool_counts):
            resources.append(ResourceSpec(
                key=pool_resource_key(port), kind=RESOURCE_KIND_POOL,
                capacity_per_ns=float(rates.pool_port_bytes_per_ns)))
            hold_pool = _unknown_eta_bytes(
                float(rates.pool_port_bytes_per_ns))
            works.append(WorkItem(
                work_id=f"bg:pool:{port}",
                legs=tuple(
                    _stream_leg(
                        f"bg:pool:{port}#f{index}",
                        route=(pool_resource_key(port),),
                        bytes_amount=hold_pool,
                        remaining_bytes=hold_pool, in_service=True)
                    for index in range(pool_counts[port]))))
        if pool_counts:
            approximations.append(
                "pool_registry_counts_inferred_unknown_eta")

        return WorldContext(
            now_ns=0,
            resources=tuple(resources),
            background_works=tuple(works),
            efficiency=_world_projection_efficiency(model),
            approximations=tuple(approximations),
            progress_source=self.progress_source,
        )


class LedgerWorldSource(WorldSource):
    """W1 账本源（W5 接线：真实每流剩余量/依赖的背景投影）。

    数据面（W3 EVIDENCE §7 接线点）：

    * ``ledger``（``joint.work_ledger.WorkLedger``）：活跃 ``CommittedWork``
      的权威 Python 侧状态——``leg`` 携带资源腿（``hbm:{KIND}`` /
      ``pool:{port}:{seq}`` / ``noc:{flow}``）。``remaining`` 对
      progress_source=authoritative 的腿是后端积分真值（full 快照在场腿
      与 delta commit 携带 remaining_* 的腿）；delta 交付中**未携带权威
      remaining 的中途首见腿**以总量上界占位（progress_source=stale/
      inferred，§4.3:174：不得把最初总字节当精确剩余）——本源把这些腿
      的过估入 ``approximations`` 披露（``ledger_leg_remaining_
      unverified=N``），不静默冒充权威观测；
    * ``snapshot``（``DecisionSnapshot``，可选）：决策级快照（work_index
      交叉对账 + 版本面披露），缺省 None = 仅账本通道；
    * ``link_id_map_fn``（可选）：C++ 链路 id → (src, dst) rank 对的换算
      （生产 = sh30 ``_telemetry_link_id_map`` 同源表；None/未命中 = 该流
      记入 approximation 披露，不虚构链路键）。

    投影规则（V2 资源前推）：

    * in-service 资源腿 → ``TaskSpec`` in-service 流腿（账本
      ``remaining_bytes``，不假设 ETA）；
    * issued-not-streaming 腿 → 非 in-service 流腿（激活即竞争带宽；
      ``ready_at=now``——发射已发生，未到流式段）；
    * committed-not-issued（节点级、无资源腿）→ 不虚构资源腿（Q 门/
      依赖语义由候选自身图承担）；计数入 ``approximations`` 披露
      （``ledger_committed_not_issued=N``）；
    * **逐资源键降级**（不静默空背景）：账本在键 ``hbm:{r}`` /
      ``pool:{p}`` / ``link:{a}->{b}`` 上没有任何权威腿时，该键回退
      LoadsWorldSource 的注册表计数腿（unknown-ETA，approximation
      ``ledger_leg_absent_registry_counts``）；键上有权威腿则**只**用
      账本腿（避免账本/注册表同键双计）。HBM 腿当前无字节量纲
      （对端 commit 未携带 remaining_*，连 total 也不可得），逐键恒走
      注册表回退——bridge-v1 对端补齐剩余量携带前，该降级与 stale 腿
      过估同为已披露常态；随对端补齐，权威路径逐键接管。
    """

    progress_source = PROGRESS_SOURCE_LEDGER

    def __init__(self, ledger=None, snapshot=None,
                 link_id_map_fn=None) -> None:
        self._ledger = ledger
        self._snapshot = snapshot
        self._link_id_map_fn = link_id_map_fn

    def build_context(self, model) -> WorldContext:
        if self._ledger is None:
            # 无账本的裸构造仍显式失败（乐观计价比显式失败危险，§5.6）；
            # 生产装配（sh30 ``_joint_cost_model``）恒带账本或显式降级到
            # LoadsWorldSource（缺席标记在装配点披露）。
            raise ActionWorkGraphError(
                "LedgerWorldSource requires the causal work ledger "
                "(joint.work_ledger.WorkLedger); refusing to price with "
                "an empty background")
        ledger_keys: set[str] = set()
        resources: List[ResourceSpec] = []
        works: List[WorkItem] = []
        approximations: List[str] = []
        committed_not_issued = 0
        unmapped_link_flows = 0
        unverified_keys: Dict[str, int] = {}
        rates = model.rates
        projection = getattr(self._snapshot, "ledger_projection", None)
        if projection is not None:
            # The hot ledger view freezes only its incrementally maintained
            # current pricing frontier. Node-only and committed rows are
            # summarized in the maintained count, without a full ledger scan.
            committed_not_issued = projection.committed_not_issued
            active_works = list(projection.pricing_rows)
        else:
            # Compatibility for direct/test callers without a production
            # DecisionSnapshot.
            committed_not_issued = 0
            active_works = list(self._ledger.iter_active_works())

        # Protocol-v3 active_resources is the same authoritative resource
        # frontier as W1's resource legs, but it also covers pipeline-issued
        # endpoints whose callbacks are represented by compact descriptors.
        # Overlay matching physical leg identities in-place and append only
        # genuinely additional legs.  In particular, request_id is not a
        # filter: cross-request NoC flows and same-request pipeline flows all
        # compete on the shared link keys, exactly as in the W1 snapshot.
        pipeline_state = getattr(self._snapshot, "pipeline_ledger_state", None)
        pipeline_rows = (
            pipeline_state.get("active_resources", ())
            if isinstance(pipeline_state, Mapping) else ())
        if pipeline_rows:
            replacements = {}
            for row in pipeline_rows:
                identity = WorkIdentity(
                    request_id=row["request_id"], stage=row["stage"],
                    generation=row["generation"], rank=row["rank"],
                    node_id=row["node_id"], operation=row["operation"],
                    resource_legs=(row["leg"],))
                resource_state = row["state"]
                if row["resource_type"] == "hbm_job":
                    state = resource_state
                elif row["resource_type"] == "pool_job":
                    state = {
                        "LatencyWaiting": "issued",
                        "Streaming": "in_service",
                        "FluidCompleteAwaitingCallback": "completed",
                    }[resource_state]
                else:
                    state = ("in_service" if resource_state == "Active"
                             else "completed")
                work = CommittedWork(
                    identity=identity, state=state,
                    total=WorkAmount(**row["total"]),
                    served=WorkAmount(),
                    remaining=WorkAmount(**row["remaining"]),
                    progress_source="authoritative",
                    service_unit=row["service_unit"],
                    route=tuple(row["route"]), leg=row["leg"],
                    issued_tick=row["issued_tick"],
                    as_of_tick=row["as_of_tick"])
                replacements[work.work_id] = work
            if len(replacements) != len(pipeline_rows):
                raise ActionWorkGraphError(
                    "protocol-v3 active_resources repeats a physical leg identity")
            replaced_ids = set()
            merged = []
            for work in active_works:
                replacement = replacements.get(work.work_id)
                if replacement is not None:
                    merged.append(replacement)
                    replaced_ids.add(work.work_id)
                else:
                    merged.append(work)
            merged.extend(
                work for work_id, work in replacements.items()
                if work_id not in replaced_ids)
            active_works = merged
        for work in active_works:
            if projection is None and work.state in ("committed", "ready"):
                committed_not_issued += 1
                continue
            leg = work.leg
            if not leg:
                continue  # 节点级 issued 工作：无资源腿，不虚构
            keys = (("",),)
            capacity = 0.0
            kind = ""
            if leg.startswith("hbm:"):
                keys = ((hbm_resource_key(work.identity.rank),),)
                capacity = float(rates.local_hbm_bytes_per_ns)
                kind = RESOURCE_KIND_HBM
            elif leg.startswith("pool:"):
                try:
                    port = int(leg.split(":")[1])
                except (IndexError, ValueError):
                    continue
                keys = ((pool_resource_key(port),),)
                capacity = float(rates.pool_port_bytes_per_ns)
                kind = RESOURCE_KIND_POOL
            elif leg.startswith("noc:"):
                route = work.route or ()
                link_keys: List[str] = []
                if self._link_id_map_fn is not None:
                    for link_id in route:
                        mapped = self._link_id_map_fn(int(link_id))
                        if mapped is None:
                            unmapped_link_flows += 1
                            link_keys = []
                            break
                        link_keys.append(link_resource_key(*mapped))
                else:
                    unmapped_link_flows += 1
                if not link_keys:
                    continue
                keys = (tuple(dict.fromkeys(link_keys)),)
                capacity = float(rates.noc_link_bytes_per_ns)
                kind = RESOURCE_KIND_LINK
            else:
                continue
            remaining = work.remaining.bytes
            if remaining is None:
                total = work.total.bytes
                if not total:
                    # 无任何量纲（如 HBM 腿在 commit 无 total 时）——该键
                    # 交由注册表回退路径披露（ledger_leg_absent_registry_
                    # counts），这里不计入过估。
                    continue
                remaining = total
            if remaining <= 0:
                continue  # 已服务完的腿（等待 completion 事件清账）
            unverified = work.progress_source != "authoritative"
            in_service = work.state == "in_service"
            # A multi-hop ledger work is one end-to-end flow.  Keep the whole
            # route on one WorkItem so every hop competes concurrently while
            # the payload is drained exactly once.  Splitting it into one
            # WorkItem per link both duplicated the work id and charged the
            # same remaining bytes once per hop.
            for key_group in keys:
                route = tuple(key_group)
                if not route:
                    continue
                for key in route:
                    first_for_key = key not in ledger_keys
                    ledger_keys.add(key)
                    if unverified:
                        # 非权威剩余量 = 总量上界占位（首见前已服务量不
                        # 可恢复，§4.3:174）——过估**逐键**显式登记降级
                        # （决策投影披露面；与 absent_registry_counts
                        # 同形态），不冒充精确剩余。
                        unverified_keys[key] = (
                            unverified_keys.get(key, 0) + 1)
                    if first_for_key:
                        resources.append(ResourceSpec(
                            key=key, kind=kind,
                            capacity_per_ns=capacity))
                works.append(WorkItem(
                    work_id="bg-ledger:" + work.work_id,
                    legs=(_stream_leg(
                        "bg-ledger:{}#{}".format(
                            work.work_id, "|".join(route)),
                        route=route,
                        bytes_amount=int(remaining),
                        remaining_bytes=int(remaining),
                        in_service=in_service),),
                ))
        if committed_not_issued:
            approximations.append(
                "ledger_committed_not_issued={}".format(
                    committed_not_issued))
        if unmapped_link_flows:
            approximations.append(
                "ledger_noc_link_id_unmapped={}".format(
                    unmapped_link_flows))
        for key in sorted(unverified_keys):
            approximations.append(
                "ledger_leg_remaining_unverified:{}={}".format(
                    key, unverified_keys[key]))
        if self._snapshot is not None:
            approximations.append(
                "ledger_snapshot_watermark={}".format(
                    int(self._snapshot.event_watermark)))
        # 逐键降级：账本无权威腿的键回退注册表计数（LoadsWorldSource
        # 同源构造），不在账本已接管的键上双计。
        fallback = LoadsWorldSource().build_context(model)
        for spec in fallback.resources:
            if spec.key in ledger_keys:
                continue
            resources.append(spec)
            approximations.append(
                "ledger_leg_absent_registry_counts:{}".format(spec.key))
        for work in fallback.background_works:
            if any(key in ledger_keys
                   for leg in work.legs for key in leg.route):
                continue
            works.append(work)
        approximations.append("ledger_source=ledger_v1")
        return WorldContext(
            now_ns=0,
            resources=tuple(resources),
            background_works=tuple(works),
            efficiency=_world_projection_efficiency(model),
            approximations=tuple(approximations),
            progress_source=self.progress_source,
        )


# ------------------------------------------------------------- 腿基元 --


def _unknown_eta_bytes(capacity_per_ns: float) -> int:
    """unknown-ETA 持有哨兵字节（capacity × hold；语义见模块 docstring）。"""
    return max(1, int(float(capacity_per_ns) * UNKNOWN_ETA_HOLD_NS))


def _fixed_leg(leg_id: str, duration_ns: float, *,
               deps: Sequence[str] = (), ready_at_ns: Optional[int] = None,
               signals_barrier: Optional[str] = None,
               releases: Sequence["GateLease"] = (),
               order: Optional[int] = None,
               registration_ordinal: Optional[int] = None) -> "TaskSpecLike":
    from joint.resource_projection import TaskSpec
    return TaskSpec(
        leg_id=leg_id, kind=TASK_KIND_FIXED,
        duration_ns=float(duration_ns),
        deps=tuple(deps), ready_at_ns=ready_at_ns,
        signals_barrier=signals_barrier, releases=tuple(releases),
        order=order, registration_ordinal=registration_ordinal)


def _compute_leg(leg_id: str, *, route: Sequence[str],
                 bytes_amount: int, base_ns: float,
                 group: str = "prefill",
                 deps: Sequence[str] = (),
                 latency_ns: int = 0,
                 ready_at_ns: Optional[int] = None,
                 signals_barrier: Optional[str] = None,
                 releases: Sequence["GateLease"] = (),
                 order: Optional[int] = None,
                 registration_ordinal: Optional[int] = None,
                 die_plan: object = None) -> "TaskSpecLike":
    """构造 compute 腿（legacy 双侧聚合 / §9.6 die 计划腿）。

    ``die_plan`` 在场 = A3 执行协议推进的计划腿：bytes/base_ns 必须为
    0（聚合等效面退役，§9.6 第 3 条），route = 描述符的 HBM 端口路由
    （分享集合/除数仍由 native 投影内核裁决）。
    """
    from joint.resource_projection import TaskSpec
    return TaskSpec(
        leg_id=leg_id, kind=TASK_KIND_COMPUTE,
        route=tuple(route), bytes_amount=int(bytes_amount),
        base_ns=float(base_ns), group=group, deps=tuple(deps),
        latency_ns=int(latency_ns),
        ready_at_ns=ready_at_ns, signals_barrier=signals_barrier,
        releases=tuple(releases), order=order,
        registration_ordinal=registration_ordinal, die_plan=die_plan)


def _stream_leg(leg_id: str, *, route: Sequence[str],
                bytes_amount: int, latency_ns: int = 0,
                deps: Sequence[str] = (), ready_at_ns: Optional[int] = None,
                acquires: Sequence[GateLease] = (),
                releases: Sequence[GateLease] = (),
                remaining_bytes: Optional[int] = None,
                in_service: bool = False,
                order: Optional[int] = None,
                registration_ordinal: Optional[int] = None) -> "TaskSpecLike":
    from joint.resource_projection import TaskSpec
    return TaskSpec(
        leg_id=leg_id, kind=TASK_KIND_STREAM,
        route=tuple(route), bytes_amount=int(bytes_amount),
        latency_ns=int(latency_ns), deps=tuple(deps),
        ready_at_ns=ready_at_ns,
        acquires=tuple(acquires), releases=tuple(releases),
        remaining_bytes=remaining_bytes,
        in_service=in_service, order=order,
        registration_ordinal=registration_ordinal)


# ------------------------------------------------------------ 候选图 --


@dataclass
class CandidateGraph:
    """一个 (instance, action) 候选的预测工作图 + 审计披露束。"""

    work: WorkItem
    resources: Tuple[ResourceSpec, ...] = ()
    gates: Tuple[GateSpec, ...] = ()
    barriers: Tuple[BarrierSpec, ...] = ()
    audits: Mapping[str, object] = field(default_factory=dict)
    approximations: Tuple[str, ...] = ()
    service_leg_ids: Tuple[str, ...] = ()
    # §7.2 未完 P 义务（参竞争；与候选共享冻结背景——只读、无跨候选
    # 状态污染）。
    extra_background_works: Tuple[WorkItem, ...] = ()

    def project_against(
        self,
        world: WorldContext,
        params: PredictionParams,
        *,
        frozen_diagnostic=None,
        native_library=None,
        native_context=None,
        kernel_contract=None,
        effective_window_tile_limit: Optional[int] = None,
    ) -> ProjectionResult:
        """前推（known_work 正式路径 / frozen_diagnostic A 臂）。

        v5 共享快照形态（frozen_diagnostic 为 None 的正式路径）：上下文 =
        决策级共享底座（WorldContext.shared_native_projection_context，
        全体候选一次 jproj_create），候选私有附加经 ``private_overlay``
        逐调用传入；原生侧背景参考运行按 (overlay 内容, 运行选项) 记忆
        化——同决策内 overlay 相同的候选共享一次参考运行。

        A 臂（frozen_diagnostic 非 None）：份额折减随候选自身腿数变化，
        无法共享——保留旧逐候选全量快照路径（诊断臂，非生产热路径）。
        """
        frozen_keys: Optional[Dict[str, float]] = None
        approximations = list(self.approximations)
        if frozen_diagnostic is not None:
            member_counts = world._member_counts()
            own_counts: Dict[str, int] = {}
            for leg in self.work.legs:
                for key in leg.route:
                    own_counts[key] = own_counts.get(key, 0) + 1
            frozen_keys = {
                key: float(member_counts.get(key, 0)
                           + own_counts.get(key, 0))
                for key in own_counts}
            approximations.append(
                "frozen_diagnostic:"
                + str(frozen_diagnostic.frozen_horizon_mode))
        if native_library is None:
            if native_context is not None:
                raise ActionWorkGraphError(
                    "native context reuse requires its library provenance")
            from joint.native_projection import load_production_projection_library
            native_library = load_production_projection_library()
        if kernel_contract is None and params.window_tile_limit != 1:
            raise ActionWorkGraphError(
                "a direct projection without a frozen run contract is exact-only")
        context = native_context
        created = context is None
        if frozen_diagnostic is not None:
            # A 臂：候选私有份额折减已折入全量输入，走旧逐候选上下文。
            input_data = world.projection_input(
                self, frozen_resource_keys=frozen_keys)
            if created:
                context = native_library.create_context(
                    input_data, kernel_contract)
            try:
                result = context.project(
                    (self.work,), params,
                    analytic_segments=True,
                    effective_window_tile_limit=(
                        params.window_tile_limit
                        if effective_window_tile_limit is None
                        else effective_window_tile_limit),
                )
            except Exception:
                if created:
                    context.close()
                raise
        else:
            # 正式路径：决策级共享上下文 + 候选私有 overlay。
            if created:
                context = world.shared_native_projection_context(
                    native_library, kernel_contract)
            try:
                result = context.project(
                    (self.work,), params,
                    analytic_segments=True,
                    effective_window_tile_limit=(
                        params.window_tile_limit
                        if effective_window_tile_limit is None
                        else effective_window_tile_limit),
                    private_overlay=world.candidate_overlay(self),
                )
            except Exception:
                # 共享上下文不可因单候选失败被关闭（其余候选仍要用）；
                # 它由决策收尾统一 close。
                raise
        # E0 精确复询与决策收尾 close 都经候选的 projection_context 引用；
        # 共享上下文对所有候选是同一对象（close 幂等）。
        self._native_projection_context = context
        result = ProjectionResult(
            status=result.status,
            predicted_service_done_ns=result.predicted_service_done_ns,
            predicted_settle_done_ns=result.predicted_settle_done_ns,
            reason=result.reason,
            dependency_attribution=result.dependency_attribution,
            focus_bytes=result.focus_bytes,
            bytes_served_total=result.bytes_served_total,
            event_count=result.event_count,
            resource_count=result.resource_count,
            work_count=result.work_count,
            closure_resource_count=result.closure_resource_count,
            closure_work_count=result.closure_work_count,
            approximations=tuple(approximations) + tuple(
                result.approximations),
            fallback=result.fallback,
            background_comparisons=result.background_comparisons,
            closure_wall_ns=result.closure_wall_ns,
            candidate_world_wall_ns=result.candidate_world_wall_ns,
            reference_world_wall_ns=result.reference_world_wall_ns,
            reference_world_runs=result.reference_world_runs,
            native_disclosure=result.native_disclosure,
        )
        return result


# ------------------------------------------------------------ 图构造器 --

#: 审计字典键（JCM breakdown/notes 消费；C5 冻结 schema 的 notes 面）。
AUDIT_NOTES = "notes"
AUDIT_TARGET_WAIT = "target_wait_ns"
AUDIT_HISTORY_PREP = "history_prep_ns"
AUDIT_COMPUTE = "compute_ns"
AUDIT_MERGE = "merge_ns"
AUDIT_FIRST_FORWARD = "prefill_first_forward_ns"


def _link_route(model, paths_by_rank, rank_index: int,
                *, include_endpoints: bool = True) -> Tuple[str, ...]:
    """单 shard 传输腿资源集：逐跳链路 + 双端 HBM（三腿 max 的统一形态）。

    ``paths_by_rank[rank_index]`` 为该 rank 的路径序列（``_shard_paths``
    口径：每 rank 至多一条代表路径——多路径时取首条，与
    ``_shard_endpoint_divisors`` 的端点取径同源）。W2 内核
    ``min_e(peak/n_e)`` ≡ 旧 ``max(noc_stream, home_read, exec_write)``
    （同字节过全资源）；``include_endpoints=False`` 剥离 HBM 端点（纯
    NoC 段）。
    """
    rank_paths = paths_by_rank[rank_index] if paths_by_rank else ()
    path = tuple(rank_paths[0]) if rank_paths else ()
    route: List[str] = []
    for key in noc_route_keys(path):
        if key not in route:
            route.append(key)
    if include_endpoints and len(path) >= 2:
        for rank in (path[0], path[-1]):
            key = hbm_resource_key(rank)
            if key not in route:
                route.append(key)
    return tuple(route)


def _hop_latency(model, paths_by_rank, rank_index: int) -> int:
    rank_paths = paths_by_rank[rank_index] if paths_by_rank else ()
    path = tuple(rank_paths[0]) if rank_paths else ()
    return int(model.rates.d2d_latency_ns * max(0, len(path) - 1))


def _pool_restore_route(model, paths_by_rank, rank_index: int,
                        target_rank: int) -> Tuple[str, ...]:
    """池恢复方向资源集：池端口 + edge→rank 链路 + 目标 HBM 写（式 7）。

    注意 ``_pool_shard_inputs`` 的 ``paths[rank]`` 即该 rank 的路径本体
    （非 ``_shard_paths`` 的路径序列包裹形态）。
    """
    path = tuple(
        paths_by_rank[rank_index]) if paths_by_rank else (target_rank,)
    route: List[str] = [pool_resource_key(path[0])]
    for key in noc_route_keys(path):
        if key not in route:
            route.append(key)
    # 目标端 HBM 写腿无条件在场（式 7 契约：edge==端点 rank 的零距离
    # 事务仍真实写端点 HBM，只是无 NoC 段）。
    key = hbm_resource_key(path[-1] if len(path) >= 2 else target_rank)
    if key not in route:
        route.append(key)
    return tuple(route)


def _layer_split(total_ns: int, layer_count: int) -> Tuple[int, ...]:
    """逐层均分（余数前置，总和守恒——``_split_ns_by_layer`` 同式）。"""
    if layer_count <= 0:
        raise ActionWorkGraphError("layer count must be positive")
    total_ns = int(total_ns)
    base, remainder = divmod(max(0, total_ns), int(layer_count))
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(int(layer_count)))


def _decode_step_total_ns(model, *, base_context_tokens: int, steps: int) -> float:
    """剩余 n̂ 步 decode 基础服务 = **逐步精确** Σ_k τ(z_k)（W3：K=8 区段
    中点折叠退役——§5.3 不再固定 8 段；z_k = base + k − 1，式 18b 口径）。

    模板在场时逐步求值（生产 τ 有 memo——重复上下文跨候选命中）；标量
    回退 = n̂ × decode_ns_per_token 恒等（零漂移锚）。返回 pre-factor 浮
    点量（消费方 × decode_factor 后 int 落盘——旧口径同纪律）。
    """
    steps = max(0, int(steps))
    if steps == 0:
        return 0.0
    provider = getattr(model, "die_plan_provider", None)
    if provider is not None:
        # §9.6-1：die 后端的名义 service = 同源聚合计划在 B/s 服务率下
        # 的推进量（非内存资源时间不随除数/因子缩放）。
        return float(provider.decode_horizon_nominal_ns(
            int(base_context_tokens), steps))
    if model.decode_task_load_ns_fn is None:
        return float(steps) * float(model.decode_ns_per_token)
    base = int(base_context_tokens)
    total = 0.0
    for offset in range(steps):
        total += float(model.decode_task_load_ns_fn(
            base + offset, hbm_share_divisor=1))
    return total


def _decode_step_profile(
    model, *, base_context_tokens: int, steps: int,
    service_total: Optional[float] = None,
) -> Tuple[int, int]:
    """Return (pure compute ns, HBM bytes) for the decode horizon.

    §9.6（joint 二期）：die 后端（``model.die_plan_provider`` 在场）逐
    步取同源 DiePlan 的 ``(非内存 ns, service_units)``——第二个量是
    库 CostResult 的内存服务量（等分仲裁的排空量），**不是**
    ``service×bandwidth`` 伪造的字节；图以 ``decode_hbm_field=
    service_units`` 披露该口径切换。
    """
    steps = max(0, int(steps))
    if steps == 0:
        return 0, 0
    provider = getattr(model, "die_plan_provider", None)
    if provider is not None:
        pure, units = provider.decode_horizon_profile(
            int(base_context_tokens), steps)
        return int(pure), int(math.ceil(units))
    profile = getattr(model, "decode_compute_profile", None)
    # A JCM always exposes the public profile method, including direct
    # fixtures whose profile falls back to ``decode_task_load_ns_fn``.  In
    # that fallback shape, calling the profile and then the exact service
    # ledger would evaluate the template twice per step (and distort the
    # causal call-count contract).  Only invoke the decomposition path when
    # at least one explicit pure/HBM callback is present; otherwise the
    # already computed exact service total is the shared fallback.
    explicit_profile = (
        getattr(model, "decode_pure_compute_ns_fn", None) is not None
        or getattr(model, "decode_hbm_bytes_fn", None) is not None)
    if callable(profile) and explicit_profile:
        pure = 0
        hbm = 0
        for offset in range(steps):
            step_pure, step_hbm = profile(
                int(base_context_tokens) + offset)
            pure += int(step_pure)
            hbm += int(step_hbm)
        return pure, hbm
    service = (service_total if service_total is not None else
               _decode_step_total_ns(
                   model, base_context_tokens=base_context_tokens,
                   steps=steps))
    pure = max(0, int(math.ceil(service)))
    hbm = max(0, int(math.ceil(
        service * float(model.rates.local_hbm_bytes_per_ns))))
    return pure, hbm


def _decode_step_latency_profile(
    model, *, base_context_tokens: int, steps: int,
) -> Tuple[int, int]:
    """Sum per-forward ``(HBM startup, controller)`` latency profiles."""
    steps = max(0, int(steps))
    if steps == 0:
        return 0, 0
    profile = getattr(model, "decode_latency_profile", None)
    if not callable(profile):
        return 0, 0
    startup = 0
    controller = 0
    for offset in range(steps):
        step_startup, step_controller = profile(
            int(base_context_tokens) + offset)
        startup += int(step_startup)
        controller += int(step_controller)
    return startup, controller


def _die_decode_descriptors(
        model, *, steps: int, base_context_tokens: int, exec_ranks,
        service_factor: float, kv_identity: str):
    """§9.6 第 2 条：decode 地平线的逐 rank DiePlanDescriptor。

    无供给器（legacy）或供给器声明聚合规景（decode_horizon_descriptor
    返回 None——逐 step 计划的 tile 级协议展开不可行，见供给器
    docstring）= None = 聚合 compute 腿（bytes 场 = service_units、
    base = 非内存 ns，两量同源估值）。
    """
    provider = getattr(model, "die_plan_provider", None)
    if provider is None:
        return None
    descriptors = {}
    for rank_index in range(len(exec_ranks)):
        descriptor = provider.decode_horizon_descriptor(
            decode_steps=max(1, int(steps)),
            base_context_tokens=int(base_context_tokens),
            tensor_parallel_rank=rank_index,
            service_factor=float(service_factor),
            kv_identity=kv_identity,
            route=(hbm_resource_key(exec_ranks[rank_index]),))
        if descriptor is None:
            return None
        descriptors[rank_index] = descriptor
    return descriptors


def _die_prefill_layer_descriptors(
        model, *, input_tokens: int, history_tokens: int, exec_ranks,
        service_factor: float, kv_identity: str):
    """§9.6 第 2/3 条：prefill **逐层段**的逐 rank DiePlanDescriptor。

    供给器声明聚合规景（描述符 None——tile 级协议展开超预算，见供给器
    docstring）或 legacy 无供给器 = None：逐层量走供给器的逐层 DieOpSpec
    估值（``prefill_layer_profile``，见消费点）。
    """
    provider = getattr(model, "die_plan_provider", None)
    if provider is None:
        return None
    descriptors = {}
    for rank_index in range(len(exec_ranks)):
        descriptor = provider.prefill_layer_descriptor(
            input_tokens=int(input_tokens),
            history_tokens=int(history_tokens),
            tensor_parallel_rank=rank_index,
            service_factor=float(service_factor),
            kv_identity=kv_identity,
            route=(hbm_resource_key(exec_ranks[rank_index]),))
        if descriptor is None:
            return None
        descriptors[rank_index] = descriptor
    return descriptors


def _die_rebuild_descriptors(
        model, *, rebuild_tokens: int, exec_ranks,
        service_factor: float):
    """§9.6：D 边界静默重建（H_B 整体重放，span 基 0）的逐 rank
    DiePlanDescriptor（供给器聚合规景/legacy = None → 聚合双量腿）。"""
    provider = getattr(model, "die_plan_provider", None)
    if provider is None:
        return None
    descriptors = {}
    for rank_index in range(len(exec_ranks)):
        descriptor = provider.rebuild_descriptor(
            rebuild_tokens=int(rebuild_tokens),
            tensor_parallel_rank=rank_index,
            service_factor=float(service_factor),
            route=(hbm_resource_key(exec_ranks[rank_index]),))
        if descriptor is None:
            return None
        descriptors[rank_index] = descriptor
    return descriptors


def _mark_aggregate_profile_if_needed(
    builder: "_GraphBuilder", model, stage: str, profile_fn,
) -> None:
    """Disclose the default aggregate-to-HBM profile approximation.

    The public profile methods keep the stable ``(compute_ns, hbm_bytes)``
    contract. A model with both decomposition callbacks supplies a physical
    split; the default method derives HBM work from aggregate service and
    therefore needs an explicit approximation marker. Custom profile
    implementations own their disclosure and are left untouched here.
    """
    if getattr(model, "die_plan_provider", None) is not None:
        # §9.6 第 3 条：die 后端的 profile 双量来自同源 DiePlan（非内存
        # ns + service_units）——不适用"聚合等效 HBM profile"近似标记。
        return
    if not callable(profile_fn):
        marker = "aggregate_service_equivalent_hbm_profile"
        if marker not in builder.approximations:
            builder.approximations.append(marker)
        return
    try:
        from joint.joint_cost_model import JointCostModel
        expected = getattr(
            JointCostModel, f"{stage}_compute_profile")
        uses_default = (
            getattr(profile_fn, "__func__", profile_fn) is expected)
    except (ImportError, AttributeError):
        uses_default = False
    if not uses_default:
        return
    callback_names = (
        f"{stage}_pure_compute_ns_fn", f"{stage}_hbm_bytes_fn")
    if all(getattr(model, name, None) is not None
           for name in callback_names):
        return
    marker = "aggregate_service_equivalent_hbm_profile"
    if marker not in builder.approximations:
        builder.approximations.append(marker)


def _latency_profile(model, stage: str, *args) -> Tuple[int, int]:
    """Read one aggregate/per-step ``(startup, controller)`` profile."""
    profile = getattr(model, f"{stage}_latency_profile", None)
    if not callable(profile):
        return 0, 0
    value = profile(*args)
    if (isinstance(value, (str, bytes))
            or not isinstance(value, (tuple, list))
            or len(value) != 2):
        raise ActionWorkGraphError(
            f"{stage} latency profile must return (startup_ns, "
            "controller_ns)")
    result = []
    for name, item in zip(("startup", "controller"), value):
        if isinstance(item, bool):
            raise ActionWorkGraphError(
                f"{stage} latency {name} must be finite and >= 0")
        try:
            number = float(item)
        except (TypeError, ValueError) as exc:
            raise ActionWorkGraphError(
                f"{stage} latency {name} must be finite and >= 0") from exc
        if not math.isfinite(number) or number < 0:
            raise ActionWorkGraphError(
                f"{stage} latency {name} must be finite and >= 0")
        result.append(int(math.ceil(number)))
    return result[0], result[1]


def _mark_latency_profile_if_needed(
    builder: "_GraphBuilder", model, stage: str,
) -> None:
    """Disclose missing latency decomposition or aggregate phase folding."""
    marker = (
        "aggregate_phase_latency_folded"
        if getattr(model, f"{stage}_latency_profile_fn", None) is not None
        else "profile_latency_unavailable")
    if marker not in builder.approximations:
        builder.approximations.append(marker)


def _split_bytes_by_layer(total_bytes: int, layer_count: int) -> Tuple[int, ...]:
    """Conservation-preserving integer split, with remainder up front."""
    if int(total_bytes) < 0 or int(layer_count) <= 0:
        raise ActionWorkGraphError(
            "compute byte split needs non-negative bytes and positive layers")
    base, remainder = divmod(int(total_bytes), int(layer_count))
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(int(layer_count)))


def _split_bytes_by_weights(
    total_bytes: int, weights: Sequence[int],
) -> Tuple[int, ...]:
    """Split bytes by pure-compute weights without losing integer remainder.

    The fallback profile represents one aggregate service interval as an
    equivalent HBM workload.  Equal byte chunks would make a short final
    layer's memory side exceed its rounded compute floor and add a spurious
    nanosecond at every TP barrier.  Proportional largest-remainder splitting
    preserves the aggregate workload and keeps each layer's isolated roofline
    completion at its already rounded compute share.
    """
    total_bytes = int(total_bytes)
    if total_bytes < 0 or not weights or any(int(weight) < 0 for weight in weights):
        raise ActionWorkGraphError(
            "weighted compute byte split needs non-negative bytes and weights")
    weights = tuple(int(weight) for weight in weights)
    weight_total = sum(weights)
    if weight_total <= 0:
        return _split_bytes_by_layer(total_bytes, len(weights))
    bases = []
    remainders = []
    assigned = 0
    for index, weight in enumerate(weights):
        numerator = total_bytes * weight
        base, remainder = divmod(numerator, weight_total)
        bases.append(base)
        remainders.append((remainder, index))
        assigned += base
    for _remainder, index in sorted(
            remainders, key=lambda item: (-item[0], item[1]))[
                :total_bytes - assigned]:
        bases[index] += 1
    return tuple(bases)


def _pool_restore_leg_specs(
    model,
    instance_index: int,
    bytes_by_rank: Sequence[int],
    *,
    leg_prefix: str,
    deps: Sequence[str] = (),
) -> Tuple[List["TaskSpecLike"], List[ResourceSpec], Dict[int, str]]:
    """池恢复腿族（式 7 恢复方向：池端口 + edge→rank 链路 + 目标 HBM 写）。

    返回 (legs, resources, per-rank leg id)——每 rank 一腿（λ = 池时延 +
    逐跳时延；带宽项 = min(pool, noc, hbm)），与
    ``_pool_transfer_ns_by_rank`` 的 uncontended 值同式。零字节 rank 无腿
    （F1 契约）。事件世界实际份额 = 资源活跃集动态计数。
    """
    edges, paths = model._pool_shard_inputs(
        instance_index, bytes_by_rank, restore=True)
    legs: List["TaskSpecLike"] = []
    resources: Dict[str, ResourceSpec] = {}
    per_rank: Dict[int, str] = {}
    for rank_index, shard_bytes in enumerate(bytes_by_rank):
        shard_bytes = int(shard_bytes)
        if shard_bytes <= 0:
            continue
        route = _pool_restore_route(
            model, paths, rank_index, edges[rank_index])
        for key in route:
            prefix = key.split(":", 1)[0]
            resources.setdefault(key, ResourceSpec(
                key=key,
                kind={
                    "link": RESOURCE_KIND_LINK,
                    "hbm": RESOURCE_KIND_HBM,
                    "pool": RESOURCE_KIND_POOL,
                }[prefix],
                capacity_per_ns=float({
                    "link": model.rates.noc_link_bytes_per_ns,
                    "hbm": model.rates.local_hbm_bytes_per_ns,
                    "pool": model.rates.pool_port_bytes_per_ns,
                }[prefix])))
        path = tuple(paths[rank_index]) if paths else (
            (edges[rank_index],))
        hop_latency = int(model.rates.d2d_latency_ns) * max(
            0, len(path) - 1)
        leg = _stream_leg(
            f"{leg_prefix}#R{rank_index}",
            route=route, bytes_amount=shard_bytes,
            latency_ns=int(model.rates.pool_latency_ns) + hop_latency,
            deps=tuple(deps))
        legs.append(leg)
        per_rank[rank_index] = leg.leg_id
    return legs, list(resources.values()), per_rank


class _GraphBuilder:
    """单候选图构造器（P/D 共用腿装配原语；audit 与事件图同源）。"""

    def __init__(self, model, work_id: str):
        self.model = model
        self.work_id = work_id
        self.legs: List["TaskSpecLike"] = []
        # Logical WorkItem insertion position, kept separate from the
        # completion tie-break ``TaskSpec.order``.
        self._next_registration_ordinal = 0
        self.resources: Dict[str, ResourceSpec] = {}
        self.gates: Dict[str, GateSpec] = {}
        self.barriers: List[BarrierSpec] = []
        self.notes: List[str] = []
        self.audits: Dict[str, object] = {}
        self.approximations: List[str] = []

    # ------------------------------------------------------------ 原语 --
    def note(self, text: str) -> None:
        self.notes.append(text)

    def add_resources(self, specs: Sequence[ResourceSpec]) -> None:
        for spec in specs:
            self.resources.setdefault(spec.key, spec)

    def _capacity_of(self, key: str) -> float:
        prefix = key.split(":", 1)[0]
        return float({
            "link": self.model.rates.noc_link_bytes_per_ns,
            "hbm": self.model.rates.local_hbm_bytes_per_ns,
            "pool": self.model.rates.pool_port_bytes_per_ns,
        }[prefix])

    def _register_route(self, route: Sequence[str]) -> None:
        for key in route:
            self.resources.setdefault(key, ResourceSpec(
                key=key,
                kind={
                    "link": RESOURCE_KIND_LINK,
                    "hbm": RESOURCE_KIND_HBM,
                    "pool": RESOURCE_KIND_POOL,
                }[key.split(":", 1)[0]],
                capacity_per_ns=self._capacity_of(key)))

    def reserve_registration_ordinals(self, count: int) -> int:
        """Reserve a contiguous original WorkItem insertion-order interval."""
        if type(count) is not int or count < 0:
            raise ValueError("registration ordinal reservation must be >= 0")
        first = self._next_registration_ordinal
        self._next_registration_ordinal += count
        if self._next_registration_ordinal > (1 << 64):
            raise OverflowError("WorkItem registration ordinal overflow")
        return first

    def append_existing_leg(self, leg: "TaskSpecLike") -> str:
        """Append a prebuilt ordinary leg at this builder's original position."""
        ordinal = self.reserve_registration_ordinals(1)
        self.legs.append(replace(leg, registration_ordinal=ordinal))
        return leg.leg_id

    def fixed_leg(self, tag: str, duration_ns: float, *,
                  deps: Sequence[str] = (),
                  ready_at_ns: Optional[int] = None,
                  signals_barrier: Optional[str] = None,
                  releases: Sequence[GateLease] = ()) -> str:
        leg = _fixed_leg(
            f"{self.work_id}#{tag}", duration_ns, deps=deps,
            ready_at_ns=ready_at_ns, signals_barrier=signals_barrier,
            releases=releases,
            registration_ordinal=self.reserve_registration_ordinals(1))
        self.legs.append(leg)
        return leg.leg_id

    def compute_leg(self, tag: str, *, route: Sequence[str],
                    bytes_amount: int, base_ns: float,
                    group: str = "prefill",
                    deps: Sequence[str] = (),
                    latency_ns: int = 0,
                    ready_at_ns: Optional[int] = None,
                    signals_barrier: Optional[str] = None,
                    releases: Sequence[GateLease] = (),
                    die_plan: object = None) -> str:
        self._register_route(route)
        leg = _compute_leg(
            f"{self.work_id}#{tag}", route=route,
            bytes_amount=bytes_amount, base_ns=base_ns, group=group,
            deps=deps, latency_ns=int(latency_ns), ready_at_ns=ready_at_ns,
            signals_barrier=signals_barrier, releases=releases,
            die_plan=die_plan,
            registration_ordinal=self.reserve_registration_ordinals(1))
        self.legs.append(leg)
        return leg.leg_id

    def stream_leg(self, tag: str, *, route: Sequence[str],
                   bytes_amount: int, latency_ns: int = 0,
                   deps: Sequence[str] = (),
                   ready_at_ns: Optional[int] = None,
                   acquires: Sequence[GateLease] = (),
                   releases: Sequence[GateLease] = ()) -> str:
        self._register_route(route)
        leg = _stream_leg(
            f"{self.work_id}#{tag}", route=route,
            bytes_amount=bytes_amount, latency_ns=latency_ns,
            deps=deps, ready_at_ns=ready_at_ns,
            acquires=acquires, releases=releases,
            registration_ordinal=self.reserve_registration_ordinals(1))
        self.legs.append(leg)
        return leg.leg_id

    def finish(self, *, service_excluded: Sequence[str] = ()) -> CandidateGraph:
        service_ids = tuple(
            leg.leg_id for leg in self.legs
            if leg.leg_id not in set(service_excluded))
        work = WorkItem(
            work_id=self.work_id,
            legs=tuple(self.legs),
            is_candidate=True,
            service_leg_ids=service_ids)
        self.audits[AUDIT_NOTES] = tuple(self.notes)
        # (2026-10-01 R2 补正) merge_plan（MergePlan dataclass）与
        # obligation_works 同理不留在 CandidateGraph.audits（结构化对象
        # 混入标量审计映射；消费方 build_p_candidate 在 finish 前已读回）。
        self.audits.pop("merge_plan", None)
        # (2026-10-01 审查修复) 先 pop 后拷贝：旧实现在 dict(self.audits)
        # 之后才 pop，CandidateGraph.audits 残留 obligation_works 的
        # tuple[WorkItem]（本应全为标量/字符串元组的审计映射——任何整
        # 体序列化/落盘会撞上不可序列化对象）。
        obligation_works = self.audits.pop("obligation_works", ())
        return CandidateGraph(
            work=work,
            resources=tuple(self.resources.values()),
            gates=tuple(self.gates.values()),
            barriers=tuple(self.barriers),
            audits=dict(self.audits),
            approximations=tuple(self.approximations),
            service_leg_ids=service_ids,
            extra_background_works=tuple(obligation_works),
        )

    # ---------------------------------------------------- 传输腿族构造 --
    def leg_ids(self) -> List[str]:
        return [leg.leg_id for leg in self.legs]

    def transfer_legs(
        self,
        tag: str,
        paths_by_rank,
        bytes_by_rank: Sequence[int],
        *,
        deps: Sequence[str] = (),
        prior_chain: Optional[Mapping[int, str]] = None,
        ready_at_ns: Optional[int] = None,
    ) -> Dict[int, str]:
        """逐 shard NoC 传输腿（三腿 max 统一形态：links + 双端 HBM）。

        ``prior_chain`` = 该 rank 在通道内的前一腿（copy handoff chunk
        串行链 / D 边界逐区间迁移：R_k = max(R_{k-1}, A_dest) + X_k 的
        依赖边形态）；缺省 = TP 并行（一腿/rank，共享 ``deps``）。返回
        per-rank leg id（零字节 rank 无腿——F1 契约）。
        """
        per_rank: Dict[int, str] = {}
        for rank_index, shard_bytes in enumerate(bytes_by_rank):
            shard_bytes = int(shard_bytes)
            if shard_bytes <= 0:
                continue
            route = _link_route(
                self.model, paths_by_rank, rank_index,
                include_endpoints=True)
            leg_deps = list(deps)
            if prior_chain and rank_index in prior_chain:
                leg_deps.append(prior_chain[rank_index])
            per_rank[rank_index] = self.stream_leg(
                f"{tag}#R{rank_index}", route=route,
                bytes_amount=shard_bytes,
                latency_ns=_hop_latency(
                    self.model, paths_by_rank, rank_index),
                deps=leg_deps, ready_at_ns=ready_at_ns)
        return per_rank

    def settle_legs_from_plan(
        self,
        plan,
        *,
        service_deps: Sequence[str],
    ) -> List[str]:
        """结算腿（§7.3 consolidate：service_done 后物理 merge 载荷）。

        目标规则/逐来源缺失区间/通道/引用义务/回退序全部来自
        ``merge_from_layout``（保留的确定性规则，m_end 位置无上限）；
        本处只把每 ``MergeLegPlan`` 物化为逐 rank stream 腿（源读＋NoC＋
        目标写三腿统一形态），**串行殿后**（来源升序链式——旧
        payload_ns = Σ transfer 的图形态）；目标空间容量等待 =
        ``target_space_wait_ns`` 定长 marker 腿（merge 控制点发射的
        ``_space_ready_ns_estimate`` 估计义务）。
        """
        settle_ids: List[str] = []
        if plan is None or plan.target_instance is None:
            return settle_ids
        wait_marker = None
        if int(plan.target_space_wait_ns) > 0:
            wait_marker = self.fixed_leg(
                "mgwait", float(int(plan.target_space_wait_ns)),
                deps=list(service_deps))
        chain_deps = list(service_deps) + (
            [wait_marker] if wait_marker is not None else [])
        for source_index, merge_leg in enumerate(plan.legs):
            leg_deps = list(chain_deps) if source_index == 0 else (
                list(previous_legs))
            previous_legs = []
            for rank_index, shard_bytes in enumerate(
                    merge_leg.bytes_by_rank):
                shard_bytes = int(shard_bytes)
                if shard_bytes <= 0:
                    continue
                route = _link_route(
                    self.model, merge_leg.paths_by_rank, rank_index,
                    include_endpoints=True)
                leg_id = self.stream_leg(
                    f"mg#S{source_index}#R{rank_index}",
                    route=route, bytes_amount=shard_bytes,
                    latency_ns=_hop_latency(
                        self.model, merge_leg.paths_by_rank, rank_index),
                    deps=leg_deps)
                settle_ids.append(leg_id)
                previous_legs.append(leg_id)
            if not previous_legs:
                previous_legs = list(leg_deps)
        if wait_marker is not None:
            settle_ids.append(wait_marker)
        return settle_ids


# ------------------------------------------------------------ P 图构造 --


def _settlement_is_distributed(prediction_config) -> bool:
    """§7.4（W9）：distributed = 目录提交型结算——预测侧 settle 腿零载荷
    （无物理 merge 腿/无 merge 空间等待；settle_done = service_done 同刻
    闭合，与执行侧零载荷形态一致）。consolidate 路径不受影响。"""
    return prediction_config is not None and (
        prediction_config.settlement_policy == SETTLEMENT_DISTRIBUTED)


def _flowing_path_family_public(paths_by_rank, bytes_by_rank):
    """``_flowing_path_family`` 同式局部副本（audit 视图构造，避免模块环）。"""
    return [
        tuple(path)
        for paths, shard_bytes in zip(paths_by_rank, bytes_by_rank)
        if int(shard_bytes) > 0 for path in paths]


def _chunk_chain_audit(model, stage_share, noc_share, copy_paths_by_rank,
                       chunk_bytes, upto_chunk: int, prep_anchor: int,
                       tp: int) -> List[int]:
    """copy handoff 通道内串行链审计（旧式逐 chunk 累计：R_k = max(
    R_{k-1}, A_dest) + X_k；返回 upto_chunk（含）的逐 rank 链值）。"""
    from joint.joint_cost_model import _transfer_ns_shards
    chain = [prep_anchor] * tp
    for vector in chunk_bytes[:upto_chunk + 1]:
        for rank_index in range(tp):
            shard_bytes = int(vector[rank_index])
            if shard_bytes <= 0:
                continue
            chain[rank_index] = (
                max(chain[rank_index], prep_anchor)
                + _transfer_ns_shards(
                    model.rates, model._noc_registry, stage_share,
                    paths_by_rank=(copy_paths_by_rank[rank_index],),
                    bytes_by_rank=(shard_bytes,),
                    include_self=True, kind="copy", noc_share=noc_share))
    return chain


def _contention_divisor_audit(model, noc_share, candidate_paths) -> int:
    """breakdown 除数披露位（旧式：主计价腿服务视图最坏路径除数）。"""
    if noc_share is not None:
        divisor = noc_share.max_path_divisor()
    else:
        divisor = max(1, model._noc_registry.divisor_multi(
            candidate_paths, include_self=True))
    if not isinstance(divisor, int):
        divisor = int(math.ceil(divisor))
    return divisor


def build_p_candidate(
    model,
    *,
    session,
    request,
    instance_index: int,
    action: str,
    prep_frontier_ns: int = 0,
    prediction_config=None,
) -> CandidateGraph:
    """P 阶段单候选预测工作图（§5.2：P 必要工作 + 默认 D 延续 π0 的
    参考接续 + consolidate 结算；**不提交真实图**）。

    参考接续（D18 语义的事件形态）：P 完成后按其预测布局原地运行 D——
    copy/recompute 已物化不重搬（decode 腿本地读）。结算 = service_done
    后物理 merge 载荷（``merge_from_layout`` 目标规则复用；m_end-1 回退）。

    审计口径（C5 冻结 schema/notes 面）与旧解析路径同源：传输审计值经
    保留原语（``_transfer_ns_shards`` 族）求值；cost 与里程碑自事件世界
    （``CandidateGraph.project_against``）。
    """
    from joint.joint_cost_model import (
        ACTION_COPY,
        ACTION_RECOMPUTE,
        ACTION_STAY,
        CandidatePathNocShare,
        _copy_handoff_chunk_bytes,
        _copy_handoff_layer_ranges,
        _split_ns_by_layer,
        _transfer_ns_shards,
        decode_context_weighted_tokens,
        estimate_first_forward,
    )
    from joint.capacity_summary import PhaseFootprint
    from joint.hbm_port_flow_registry import CandidateStageHbmShare

    distributed_settle = _settlement_is_distributed(prediction_config)
    load = model.loads.get(instance_index)
    if load is None:
        raise ActionWorkGraphError(
            f"no load view for instance {instance_index}")
    tp = int(model.instance_tp_size)
    layers = int(model.model_layers)
    exec_ranks = model._exec_ranks(instance_index)
    builder = _GraphBuilder(model, f"P:{instance_index}:{action}")
    builder.note("decision_stage=P")

    # ---- 1. 目标等待 Q（观测字段；计算门 = 首层计算腿 ready_at）。 ----
    target_wait = int(load.total_task_load_ns)

    # ---- 2. 空间准备（D11 分阶段足迹；A = θ 份额估计义务，§5.5）。 ----
    decode_growth_ranks = model._decode_growth_by_tp_rank(request)
    space_final = tuple(
        int(base) + int(growth)
        for base, growth in zip(
            model._space_footprint_by_tp_rank(
                session, request, instance_index, action),
            decode_growth_ranks))
    prefill_peak = tuple(
        max(0, total - growth)
        for total, growth in zip(space_final, decode_growth_ranks))
    has_pool_restore_footprint = (
        (action == ACTION_STAY and any(session.missing_bytes_by_tp_rank))
        or (action == ACTION_COPY and (
            session.location == "remote_memory"
            or any(session.missing_bytes_by_tp_rank))))
    prepare_peak = tuple(
        int(missing) if has_pool_restore_footprint else 0
        for missing in session.missing_bytes_by_tp_rank)
    footprint = PhaseFootprint(
        prepare=prepare_peak, prefill=prefill_peak,
        decode=space_final, merge=space_final)
    # §5.5（2026-09-30 修复）：θ 预测结构化接走（空摘要/零缺口 = 空
    # dict，不计入 audits——与 notes 文本注记同一在场语义）。
    theta_prediction: Dict[str, object] = {}
    space_ready = model._space_ready_ns_estimate(
        load, footprint, builder.notes, instance_index=instance_index,
        prediction_out=theta_prediction)
    if theta_prediction:
        builder.audits["capacity_theta_prediction"] = theta_prediction
    builder.note(f"space_ready_ns={space_ready}")
    prep_frontier_ns = max(0, int(prep_frontier_ns))
    prep_anchor = max(int(space_ready), prep_frontier_ns)
    if prep_frontier_ns > 0:
        builder.note(f"prep_frontier_ns={prep_frontier_ns}")
    space_leg = builder.fixed_leg("space", float(prep_anchor))

    # ---- 2a. 候选阶段 HBM 份额：静态视图**仅审计披露**（成本路径事件
    # 化——端点腿实际份额 = HBM 资源活跃集动态计数；冻结份额构造段退役，
    # 类留作静态诊断视图，inv_w23 §6.4）。 ----
    has_copy_write_leg = (
        action == ACTION_COPY
        and session.location != "remote_memory"
        and any(session.history_bytes_by_tp_rank))
    has_pool_restore_leg = has_pool_restore_footprint
    exec_stage_flows = (
        1 + int(has_copy_write_leg) + int(has_pool_restore_leg))
    stage_share = CandidateStageHbmShare(
        model.hbm_port_registry,
        {rank: exec_stage_flows for rank in exec_ranks})
    compute_share_divisor = max(
        (stage_share.stage_denominator(rank) for rank in exec_ranks),
        default=1)
    builder.note(
        "hbm_stage_share=exec_flows={}(compute+copy{}/pool{}"
        "),compute_denom={}".format(
            exec_stage_flows, int(has_copy_write_leg),
            int(has_pool_restore_leg),
            compute_share_divisor))

    # ---- 3. 历史准备腿（依动作而异；§6.2 表——仅缺失区间的源读→路由
    # →目标写；不允许重传已有副本/提前释放源端）。 ----
    history_prep = 0
    rebuild_context_tokens = 0
    copy_pool_suffix_ns = 0
    noc_share = None
    data_gate: Dict[Tuple[int, int], str] = {}
    ready_by_layer: List[List[int]] = [[0] * tp for _ in range(layers)]
    route_path, hops = model._route(instance_index, session)
    candidate_paths = model._route_paths(instance_index, session)

    def _emit_pool_legs(bytes_by_rank, tag: str) -> Dict[int, str]:
        legs_pool, _res, per_rank = _pool_restore_leg_specs(
            model, instance_index, bytes_by_rank,
            leg_prefix=f"{builder.work_id}#{tag}", deps=(space_leg,))
        for leg in legs_pool:
            builder.append_existing_leg(leg)
            builder._register_route(leg.route)
        return per_rank

    def _gate_layers(first_layer: int, per_rank_leg: Mapping[int, str],
                     audit_ready_ns: int) -> None:
        for layer_index in range(first_layer, layers):
            for rank_index, leg_id in per_rank_leg.items():
                data_gate[(layer_index, rank_index)] = leg_id
            ready_by_layer[layer_index] = [
                int(audit_ready_ns) for _ in range(tp)]

    if action == ACTION_STAY:
        missing = sum(session.missing_bytes_by_tp_rank)
        if missing:
            history_prep = model._pool_transfer_ns_by_rank(
                instance_index, session.missing_bytes_by_tp_rank,
                restore=True, hbm_port_registry=stage_share)
            builder.note("partial_suffix_pool_restore")
            per_rank = _emit_pool_legs(
                session.missing_bytes_by_tp_rank, "poolrestore")
            first_missing = min(
                max(int(session.resident_prefix_layers), 0), layers)
            _gate_layers(first_missing, per_rank,
                         prep_anchor + history_prep)
    elif action == ACTION_COPY:
        missing = sum(session.missing_bytes_by_tp_rank)
        if session.location == "remote_memory":
            full_vectors = tuple(
                int(miss) + int(resident)
                for miss, resident in zip(
                    session.missing_bytes_by_tp_rank,
                    session.history_bytes_by_tp_rank))
            history_prep = model._pool_transfer_ns_by_rank(
                instance_index, full_vectors, restore=True,
                hbm_port_registry=stage_share)
            builder.note("pool_restore_to_target")
            per_rank = _emit_pool_legs(full_vectors, "poolrestore")
            _gate_layers(0, per_rank, prep_anchor + history_prep)
        else:
            copy_paths_by_rank = model._shard_paths(
                candidate_paths, session.history_bytes_by_tp_rank)
            noc_share = CandidatePathNocShare(
                model._noc_registry,
                _flowing_path_family_public(
                    copy_paths_by_rank, session.history_bytes_by_tp_rank),
                include_self=True)
            history_prep = _transfer_ns_shards(
                model.rates, model._noc_registry, stage_share,
                paths_by_rank=copy_paths_by_rank,
                bytes_by_rank=session.history_bytes_by_tp_rank,
                include_self=True, kind="copy", noc_share=noc_share)
            model._append_u_port_note(builder.notes, copy_paths_by_rank)
            prefix_layers = min(
                max(int(session.resident_prefix_layers), 0), layers)
            if missing:
                copy_pool_suffix_ns = model._pool_transfer_ns_by_rank(
                    instance_index, session.missing_bytes_by_tp_rank,
                    restore=True, hbm_port_registry=stage_share)
                history_prep += copy_pool_suffix_ns
                per_rank = _emit_pool_legs(
                    session.missing_bytes_by_tp_rank, "poolsuffix")
                _gate_layers(prefix_layers, per_rank,
                             prep_anchor + copy_pool_suffix_ns)
            chunk_ranges = _copy_handoff_layer_ranges(prefix_layers)
            chunk_bytes = _copy_handoff_chunk_bytes(
                session.history_bytes_by_tp_rank, prefix_layers,
                chunk_ranges)
            if chunk_bytes and any(session.history_bytes_by_tp_rank):
                chunk_walls: List[int] = []
                prior_chain: Dict[int, str] = {}
                for chunk_index, vector in enumerate(chunk_bytes):
                    per_rank = builder.transfer_legs(
                        f"chunk{chunk_index}", copy_paths_by_rank,
                        vector, deps=(space_leg,),
                        prior_chain=prior_chain)
                    prior_chain.update(per_rank)
                    chain_values = _chunk_chain_audit(
                        model, stage_share, noc_share, copy_paths_by_rank,
                        chunk_bytes, chunk_index, prep_anchor, tp)
                    chunk_walls.append(max(chain_values))
                    range_start, range_end = chunk_ranges[chunk_index]
                    for layer_index in range(range_start, range_end):
                        for rank_index, leg_id in per_rank.items():
                            data_gate[(layer_index, rank_index)] = leg_id
                        ready_by_layer[layer_index] = [
                            int(chain_values[rank_idx])
                            for rank_idx in range(tp)]
                builder.note(f"copy_handoff_chunks={len(chunk_bytes)}")
                builder.note(
                    f"copy_handoff_first_chunk_ns={chunk_walls[0]}")
                builder.note(
                    "copy_handoff_chunk_walls="
                    + ",".join(str(wall) for wall in chunk_walls))
            builder.note("noc_prefix+pool_suffix_restore")
    elif action == ACTION_RECOMPUTE:
        rebuild_context_tokens = model.rebuild_context_tokens(session)
        builder.note("recompute_history_in_compute")
        builder.note(f"rebuild_context_tokens={rebuild_context_tokens}")
        builder.note("rebuild_stage=prefill_replay")
        builder.note("rebuild_suppress_sampling=False")
        builder.note("rebuild_suppress_output=False")
    else:  # pragma: no cover - ACTION_ORDER 封闭
        raise ActionWorkGraphError(f"unknown action {action!r}")

    # ---- 4. 计算段（roofline 基础 × 在线因子）。计算腿显式携带纯
    # compute floor + HBM byte work；native 投影内核会在每个资源事件
    # 边界重算 HBM 份额，不能把 aggregate service 固定成 fixed 腿。 ----
    prefill_tokens = int(request.input_tokens)
    if action == ACTION_RECOMPUTE:
        prefill_tokens += rebuild_context_tokens
    history_base = (
        0 if action == ACTION_RECOMPUTE
        else int(session.history_tokens))
    profile_fn = getattr(model, "prefill_compute_profile", None)
    if callable(profile_fn):
        prefill_pure_base_ns, prefill_hbm_bytes = profile_fn(
            prefill_tokens, history_base)
        prefill_base_ns = int(prefill_pure_base_ns)
    else:
        if model.prefill_task_load_ns_fn is not None:
            try:
                prefill_base_ns = int(model.prefill_task_load_ns_fn(
                    prefill_tokens, history_base,
                    hbm_share_divisor=1))
            except TypeError:
                prefill_base_ns = int(model.prefill_task_load_ns_fn(
                    prefill_tokens, history_base))
        else:
            prefill_base_ns = int(
                prefill_tokens * model.prefill_ns_per_token)
        prefill_hbm_bytes = int(math.ceil(
            max(0, prefill_base_ns)
            * float(model.rates.local_hbm_bytes_per_ns)))
    _mark_aggregate_profile_if_needed(
        builder, model, "prefill", profile_fn)
    prefill_startup_ns, prefill_controller_ns = _latency_profile(
        model, "prefill", prefill_tokens, history_base)
    _mark_latency_profile_if_needed(builder, model, "prefill")
    service_factor = float(model.service_factors.prefill_factor)
    # §9.6 第 4 条（joint 二期）：die 后端的 service factor 只缩放预测
    # service_units（描述符/聚合服务场内已 ×factor）；非内存资源时间
    # 不再 ×factor。legacy roofline 路径保持"pure ×factor"旧口径。
    _die_provider = getattr(model, "die_plan_provider", None)
    if _die_provider is not None:
        builder.note("die_backend=1")
        builder.note("hbm_profile_field=service_units")
        prefill_compute_ns = int(max(0, prefill_base_ns))
        prefill_service_field = int(math.ceil(
            max(0, prefill_hbm_bytes) * service_factor))
    else:
        prefill_compute_ns = int(math.ceil(
            max(0, prefill_base_ns) * service_factor))
        prefill_service_field = int(prefill_hbm_bytes)
    decode_ctx_base = (
        int(session.history_tokens) + int(request.input_tokens))
    decode_steps_count = max(0, int(request.estimated_decode_tokens))
    decode_base_service_ns = _decode_step_total_ns(
        model, base_context_tokens=decode_ctx_base,
        steps=decode_steps_count)
    decode_pure_base_ns, decode_hbm_bytes = _decode_step_profile(
        model, base_context_tokens=decode_ctx_base,
        steps=decode_steps_count, service_total=decode_base_service_ns)
    if decode_steps_count > 0:
        _mark_aggregate_profile_if_needed(
            builder, model, "decode",
            getattr(model, "decode_compute_profile", None))
    decode_startup_ns, decode_controller_ns = _decode_step_latency_profile(
        model, base_context_tokens=decode_ctx_base,
        steps=decode_steps_count)
    if decode_steps_count > 0:
        _mark_latency_profile_if_needed(builder, model, "decode")
    builder.note(
        "profile_latency=prefill_startup:{},prefill_controller:{},"
        "decode_startup:{},decode_controller:{}".format(
            prefill_startup_ns, prefill_controller_ns,
            decode_startup_ns, decode_controller_ns))
    decode_factor = float(model.service_factors.decode_factor)
    if _die_provider is not None:
        decode_compute_ns = int(max(0, decode_pure_base_ns))
        decode_service_field = int(math.ceil(
            max(0, decode_hbm_bytes) * decode_factor))
    else:
        decode_compute_ns = int(math.ceil(
            max(0, decode_pure_base_ns) * decode_factor))
        decode_service_field = int(decode_hbm_bytes)
    compute_ns = int(
        prefill_compute_ns + decode_compute_ns)
    builder.note(
        "decode_ctx_exact=base={},steps={},ns={}".format(
            decode_ctx_base, decode_steps_count,
            int(decode_base_service_ns)))
    if model.decode_task_load_ns_fn is not None:
        builder.note(
            "decode_ctx_weighted_tokens={}".format(
                decode_context_weighted_tokens(
                    decode_ctx_base, decode_steps_count)))

    # ---- 4a. prefill 逐层计算腿 + TP barrier（F_g = max(F_{g-1}, A,
    # R_g) + C 的依赖边形态；首层 ready_at = Q——式 13 F_0 = Q）。 ----
    # §9.6 第 2/3 条：die 后端的逐层腿优先携带该层的 DiePlanDescriptor
    # （协议推进；聚合 bytes/base_ns 退役为 0）；供给器声明聚合规景时
    # 逐层量走**逐层 DieOpSpec 估值**（prefill_layer_profile——每层
    # layer_count=1 的同源估算，不按 pure-time 比例分摊聚合量）。
    pf_die_plans = _die_prefill_layer_descriptors(
        model, input_tokens=prefill_tokens, history_tokens=history_base,
        exec_ranks=exec_ranks, service_factor=service_factor,
        kv_identity="local")
    if _die_provider is not None and pf_die_plans is None:
        pure_by_layer, units_by_layer = _die_provider.prefill_layer_profile(
            prefill_tokens, history_base)
        shares = tuple(int(value) for value in pure_by_layer)
        hbm_shares = tuple(
            int(math.ceil(value * service_factor))
            for value in units_by_layer)
        builder.note("die_prefill_leg=per_layer_spec_estimate")
    else:
        shares = _split_ns_by_layer(prefill_compute_ns, layers)
        hbm_shares = _split_bytes_by_weights(prefill_service_field, shares)
    # The profile's HBM startup is one linear+attention phase pair for the
    # aggregate prefill span.  It belongs on the first compute event; putting
    # a slice on every sequential layer would multiply the physical startup
    # by ``layers``.
    startup_shares = tuple(
        int(prefill_startup_ns) if layer_index == 0 else 0
        for layer_index in range(layers))
    barrier_ids: List[str] = []
    for layer_index in range(layers):
        layer_leg_ids: List[str] = []
        for rank_index in range(tp):
            deps = [space_leg]
            if layer_index:
                deps.append(barrier_ids[layer_index - 1])
            gate_leg = data_gate.get((layer_index, rank_index))
            if gate_leg is not None:
                deps.append(gate_leg)
            layer_leg_ids.append(builder.compute_leg(
                f"pf#L{layer_index}R{rank_index}",
                route=(hbm_resource_key(exec_ranks[rank_index]),),
                bytes_amount=(
                    0 if pf_die_plans is not None
                    else hbm_shares[layer_index]),
                base_ns=(
                    0.0 if pf_die_plans is not None
                    else float(shares[layer_index])),
                group="prefill", deps=deps,
                latency_ns=startup_shares[layer_index],
                ready_at_ns=(target_wait if layer_index == 0 else None),
                signals_barrier=f"{builder.work_id}:tp:L{layer_index}",
                die_plan=(
                    None if pf_die_plans is None
                    else pf_die_plans[rank_index])))
        barrier_id = f"{builder.work_id}:tp:L{layer_index}"
        builder.barriers.append(BarrierSpec(
            barrier_id, tuple(layer_leg_ids)))
        barrier_ids.append(barrier_id)
    last_barrier = barrier_ids[-1]
    # (2026-10-01 审查修复) 审计里程碑与事件世界同锚：首层计算腿经
    # space_leg（终点 = prep_anchor = max(space_ready, prep_frontier_ns)）
    # 门控，A^work 审计的 ready_space 旧值未并 prep_frontier——准备锚
    # 约束首层时 predicted_next_token_done 相对事件世界少报（D03 修复
    # 轮二口径：P 准备腿起点 = max(A, prep_frontier_ns)）。
    audit_first_forward = estimate_first_forward(
        layer_count=layers,
        pre_ns_by_layer=(0,) * layers,
        attn_ns_by_layer=shares,
        post_ns_by_layer=(0,) * layers,
        ready_data_by_layer=[tuple(row) for row in ready_by_layer],
        ready_space_ns=int(prep_anchor),
        start_ns=int(target_wait))
    builder.note(f"prefill_first_forward_ns={audit_first_forward}")
    decode_service_deps = [last_barrier, space_leg]
    decode_compute_deps = list(decode_service_deps)

    # ---- 4b. decode 服务腿（本地 KV 读：裸 decode 计算链）。 ----
    # §9.6 第 2 条：die 后端的 dec 腿携带地平线 DiePlanDescriptor（协议
    # 推进；本地 KV = local——与执行侧发射同区分）。
    dec_die_plans = _die_decode_descriptors(
        model, steps=decode_steps_count,
        base_context_tokens=decode_ctx_base,
        exec_ranks=model._exec_ranks(instance_index),
        service_factor=decode_factor, kv_identity="local")
    for rank_index in range(tp):
        builder.compute_leg(
            f"dec#R{rank_index}",
            route=(hbm_resource_key(
                model._exec_ranks(instance_index)[rank_index]),),
            bytes_amount=(
                0 if dec_die_plans is not None
                else decode_hbm_bytes),
            base_ns=(
                0.0 if dec_die_plans is not None
                else float(decode_compute_ns)),
            group="decode",
            latency_ns=decode_startup_ns,
            deps=decode_compute_deps,
            die_plan=(
                None if dec_die_plans is None
                else dec_die_plans[rank_index]))

    service_deps_for_settle = [
        leg_id for leg_id in builder.leg_ids()
        if leg_id.startswith(f"{builder.work_id}#dec")]
    controller_leg = builder.fixed_leg(
        "controller", float(prefill_controller_ns + decode_controller_ns),
        deps=service_deps_for_settle or [last_barrier])
    service_deps_for_settle.append(controller_leg)
    builder.note(
        f"controller_post_latency_ns={prefill_controller_ns + decode_controller_ns}")

    # ---- 5. 结算（consolidate：service_done 后物理 merge 载荷——
    # project_end_state/merge_from_layout 确定性目标规则复用；
    # distributed（W9 §7.4）：目录提交型——零 merge 载荷，settle 腿空）。 ----
    merge_ns = 0
    plan = None
    if distributed_settle:
        builder.note(SETTLEMENT_DISTRIBUTED_NOTE)
    elif session.home_instance is not None and (
            session.home_instance != instance_index
            or session.location == "remote_memory"):
        if session.location == "remote_memory":
            builder.note("merge_in_place")
        else:
            _settle_p_merge(
                builder, model, session, request, instance_index,
                decode_growth_ranks)
            plan = builder.audits.get("merge_plan")
            merge_ns = int(plan.merge_ns) if plan is not None else 0

    # ---- 6. 审计装配（cost/里程碑自事件世界——见 JCM 两入口）。 ----
    settle_ids = builder.settle_legs_from_plan(
        plan, service_deps=service_deps_for_settle or [last_barrier])
    builder.note(
        "prep_issue_frontier=P_prep_at_target_frontier"
        f"(+{prep_frontier_ns}ns)"
        if prep_frontier_ns > 0 else
        "prep_issue_frontier=P_prep_at_decision_epoch")
    divisor = _contention_divisor_audit(model, noc_share, candidate_paths)
    builder.audits.update({
        AUDIT_TARGET_WAIT: target_wait,
        AUDIT_HISTORY_PREP: history_prep,
        AUDIT_COMPUTE: compute_ns,
        AUDIT_MERGE: merge_ns,
        "eviction_wait_ns": int(space_ready),
        "contention_divisor": divisor,
        "hops": int(hops),
        AUDIT_FIRST_FORWARD: int(audit_first_forward),
        "decode_leg_span_ns": int(decode_compute_ns),
        "phase_space_bytes": tuple(
            tuple(int(value) for value in phase)
            for phase in (footprint.prepare, footprint.prefill,
                          footprint.decode, footprint.merge)),
        "interval_layout": _session_layout_audit(session, request),
        "inherited_obligation_ids": (),
        "estimated_remaining_decode_steps": decode_steps_count,
        "prefill_instance": int(instance_index),
        "decode_instance": int(instance_index),
        "prefill_action": action,
        "decode_action": ACTION_STAY,
        "is_continuation": False,
        "decision_stage": "P",
        "decision_epoch": 0,
        "copy_pool_suffix_ns": copy_pool_suffix_ns,
        "prefill_hbm_latency_ns": int(prefill_startup_ns),
        "prefill_controller_latency_ns": int(prefill_controller_ns),
        "decode_hbm_latency_ns": int(decode_startup_ns),
        "decode_controller_latency_ns": int(decode_controller_ns),
        "control_latency_ns": int(
            prefill_controller_ns + decode_controller_ns),
        "copy_effective_bytes": (
            sum(int(v) for v in session.missing_bytes_by_tp_rank)
            if action == ACTION_COPY else 0),
    })
    return builder.finish(service_excluded=settle_ids)


def _settle_p_merge(builder, model, session, request, instance_index,
                    decode_growth_ranks) -> None:
    """P 入口轮末布局投影 + merge 计划（旧 end_state 构造同式）。"""
    from joint.boundary_layout import (
        INTERVAL_ROLE_BASE,
        INTERVAL_ROLE_CONTEXT,
        BoundaryKvInterval,
    )
    from joint.joint_cost_model import project_end_state
    increments_by_rank = tuple(
        int(input_bytes) + int(growth)
        for input_bytes, growth in zip(
            request.input_kv_bytes_by_tp_rank, decode_growth_ranks))
    logic_home = int(session.home_instance)
    history_tokens = max(0, int(session.history_tokens))
    union_by_rank = tuple(
        int(base) + int(increment)
        for base, increment in zip(
            session.history_bytes_by_tp_rank, increments_by_rank))
    end_state = project_end_state(
        intervals=(
            BoundaryKvInterval(
                replica_id="p_merge_base", role=INTERVAL_ROLE_BASE,
                holder_instance=logic_home, authority=False,
                token_start=0,
                token_end=max(0, int(session.history_tokens)),
                layer_start=0, layer_end=0,
                bytes_by_tp_rank=tuple(
                    int(value) for value
                    in session.history_bytes_by_tp_rank)),
            BoundaryKvInterval(
                replica_id="p_merge_union", role=INTERVAL_ROLE_CONTEXT,
                holder_instance=int(instance_index), authority=True,
                token_start=0,
                token_end=(max(0, int(session.history_tokens))
                           + max(0, int(request.input_tokens))
                           + max(0, int(
                               request.estimated_decode_tokens))),
                layer_start=0, layer_end=0,
                bytes_by_tp_rank=union_by_rank),
        ),
        decode_instance=int(instance_index), logic_home=logic_home)
    plan = model.merge_from_layout(end_state, notes=builder.notes)
    builder.audits["merge_plan"] = plan


def _session_layout_audit(session, request) -> tuple:
    """P 入口区间布局摘要（D17 schema 审计透传）。"""
    from joint.joint_cost_model import _session_interval_layout
    return _session_interval_layout(session, request)



# ------------------------------------------------------------ D 图构造 --


def build_d_candidate(
    model,
    *,
    boundary,
    instance_index: int,
    action: str,
    inherited_obligation_ids: Sequence[str] = (),
    obligation_resolver: Optional[Callable[[str], ObligationView]] = None,
    prediction_config=None,
) -> CandidateGraph:
    """D 边界单候选剩余预测工作图（式 1b：Ĉ_B(j,a) 自 t_B 起）。

    计费纪律（v2.0 §2.3/§16.3 验收红线）：

    * **已完成 P 不重复收费**——P 计算量/传输/已就绪数据全部沉没
      （continuation 零准备腿）；未完的**已提交义务**经实例负载视图与
      继承义务解析照常占资源（§7.2：未完 P 义务参竞争、门控 settle；
      已完 P 零腿）。
    * **D 新迁移不回填 P 期重叠**——迁移腿自 train_join = max(Q, A) 起
      （随加入列车发射），与 D 等待不重叠。
    * 控制延迟（§7.1）：执行图目前只承诺披露的零延迟控制节点；变更新增
      交接（handoff_control_rule）仍为空并登记，不能在预测图中虚构执行。
    """
    from joint.boundary_layout import (
        BoundaryKVLayout,
        boundary_target_wait_ns,
        boundary_wait_kind,
    )
    from joint.joint_cost_model import (
        ACTION_COPY,
        ACTION_RECOMPUTE,
        ACTION_STAY,
        CandidatePathNocShare,
        _transfer_ns_shards,
        decode_context_weighted_tokens,
    )
    from joint.capacity_summary import PhaseFootprint

    distributed_settle = _settlement_is_distributed(prediction_config)
    load = model.loads.get(instance_index)
    if load is None:
        raise ActionWorkGraphError(
            f"no load view for instance {instance_index}")
    tp = int(model.instance_tp_size)
    layers = int(model.model_layers)
    exec_ranks = model._exec_ranks(instance_index)
    at_p = instance_index == boundary.p_instance_index
    # 增量型边界布局族已退役（布局恒完整型）：P 实例上的原地接续动作
    # 恒为 stay。
    is_continuation = at_p and action == ACTION_STAY
    layout = BoundaryKVLayout.from_boundary(boundary)
    missing = layout.missing_intervals_for(instance_index)
    has_missing = any(interval.total_bytes() > 0 for interval in missing)
    wait_kind = boundary_wait_kind(instance_index, boundary.p_instance_index)
    builder = _GraphBuilder(model, f"D:{instance_index}:{action}")
    builder.note("boundary_stage=D")
    builder.note("decision_stage=D")
    builder.note(f"p_instance_index={boundary.p_instance_index}")
    builder.note(f"p_action={boundary.p_action}")
    builder.note(f"is_continuation={is_continuation}")
    steps = max(1, int(boundary.decode_steps))
    builder.note(f"decode_steps={steps}")
    builder.note(f"boundary_wait_kind={wait_kind}")
    builder.note(
        "boundary_on_chip_locations="
        + ",".join(str(index) for index in layout.on_chip_locations()))
    source = boundary.p_instance_index
    _path, hops = model.route_fn(source, instance_index)

    # ---- Q（D19 等待接口拆分：原地接续 running+active / 新接纳全账）
    # + n̂_B 步 decode 服务（逐步精确——K=8 折叠退役）。 ----
    target_wait = boundary_target_wait_ns(load, wait_kind)
    decode_base_service_ns = _decode_step_total_ns(
        model, base_context_tokens=int(boundary.history_tokens),
        steps=steps)
    decode_pure_base_ns, decode_hbm_bytes = _decode_step_profile(
        model, base_context_tokens=int(boundary.history_tokens),
        steps=steps, service_total=decode_base_service_ns)
    _mark_aggregate_profile_if_needed(
        builder, model, "decode",
        getattr(model, "decode_compute_profile", None))
    decode_startup_ns, decode_controller_ns = _decode_step_latency_profile(
        model, base_context_tokens=int(boundary.history_tokens),
        steps=steps)
    _mark_latency_profile_if_needed(builder, model, "decode")
    decode_factor = float(model.service_factors.decode_factor)
    # §9.6 第 4 条：die 后端 service factor 只缩放 service_units（描述符
    # 与聚合服务场内已 ×factor），非内存资源时间不 ×factor。
    _die_provider = getattr(model, "die_plan_provider", None)
    if _die_provider is not None:
        builder.note("die_backend=1")
        builder.note("hbm_profile_field=service_units")
        decode_compute_ns = int(max(0, decode_pure_base_ns))
        decode_service_field = int(math.ceil(
            max(0, decode_hbm_bytes) * decode_factor))
    else:
        decode_compute_ns = int(math.ceil(
            max(0, decode_pure_base_ns) * decode_factor))
        decode_service_field = int(decode_hbm_bytes)
    builder.note(
        "decode_ctx_exact=base={},steps={},ns={}".format(
            int(boundary.history_tokens), steps,
            int(decode_base_service_ns)))
    builder.note(
        "profile_latency=decode_startup:{},decode_controller:{}".format(
            decode_startup_ns, decode_controller_ns))
    if model.decode_task_load_ns_fn is not None:
        builder.note(
            "decode_ctx_weighted_tokens={}".format(
                decode_context_weighted_tokens(
                    int(boundary.history_tokens), steps)))

    kv_token_bytes = model._boundary_kv_token_bytes(boundary)
    rank_bases = getattr(boundary, "kv_token_bytes_by_tp_rank", None)
    if rank_bases is not None:
        rank_bases = tuple(int(value) for value in rank_bases)
        if len(rank_bases) != tp or any(value <= 0 for value in rank_bases):
            raise ActionWorkGraphError(
                "boundary kv_token_bytes_by_tp_rank must match execution TP "
                "width and contain positive values")
        growth_ranks = tuple(value * steps for value in rank_bases)
    else:
        growth_ranks = tuple(
            kv_token_bytes * steps for _ in range(tp))

    # ---- 空间准备（C4 足迹按动作族逐区间派生；A = θ 估计义务）。 ----
    if action == ACTION_RECOMPUTE:
        hb_by_rank = boundary.working_context_bytes()
        retained_totals = [0] * len(hb_by_rank)
        for interval in layout.intervals:
            if interval in missing:
                continue
            for rank, value in enumerate(interval.bytes_by_tp_rank):
                retained_totals[rank] += int(value)
        prepare_peak = tuple(
            max(0, int(hb) - retained)
            for hb, retained in zip(hb_by_rank, retained_totals))
    elif action == ACTION_COPY and has_missing:
        copy_bytes = layout.copy_bytes_by_rank_for(
            instance_index, tp)
        prepare_peak = tuple(int(value) for value in copy_bytes)
    elif not at_p:
        working = boundary.working_bytes_by_tp_rank
        prepare_peak = tuple(int(work) for work in working)
    else:
        prepare_peak = tuple(0 for _ in growth_ranks)
    decode_peak = tuple(
        prepare + growth for prepare, growth in zip(
            prepare_peak, growth_ranks))
    footprint = PhaseFootprint(
        prepare=prepare_peak, prefill=prepare_peak,
        decode=decode_peak, merge=decode_peak)
    # §5.5（2026-09-30 修复）：θ 预测结构化接走（同 P 入口语义）。
    theta_prediction: Dict[str, object] = {}
    space_ready = model._space_ready_ns_estimate(
        load, footprint, builder.notes, instance_index=instance_index,
        prediction_out=theta_prediction)
    if theta_prediction:
        builder.audits["capacity_theta_prediction"] = theta_prediction
    builder.note(f"space_ready_ns={space_ready}")
    train_join_ns = max(int(target_wait), int(space_ready))
    builder.note("prep_issue_frontier=D_prep_at_train_join,"
                 "D_evict_at_decision_epoch")

    # ---- 控制延迟节点：旧零值夹具仍合法；生产 profile 把 controller
    # latency 放在候选工作完成后的独立 fixed leg，不伪造纯 FLOP floor。 ----
    control_latency_ns = int(decode_controller_ns)
    rebuild_startup_ns = 0
    rebuild_controller_ns = 0
    ctrl_leg = builder.fixed_leg("ctrl", 0.0)
    join_leg = builder.fixed_leg(
        "trainjoin", float(train_join_ns), deps=[ctrl_leg])

    # ---- 新准备腿（t_B 后新工作；依动作族）。 ----
    history_prep = 0
    rebuild_ns = 0
    noc_shares: List[CandidatePathNocShare] = []
    move_done_legs: List[str] = [join_leg]
    if action == ACTION_COPY and has_missing:
        copy_leg_count = 0
        prior_chain: Dict[int, str] = {}
        for holder, interval in layout.copy_plan_for(instance_index):
            leg_bytes = interval.bytes_by_tp_rank
            if not any(int(value) for value in leg_bytes):
                continue
            copy_leg_count += 1
            leg_paths = model._shard_paths(
                model._route_paths_pair(holder, instance_index), leg_bytes)
            leg_share = CandidatePathNocShare(
                model._noc_registry,
                _flowing_path_family_public(leg_paths, leg_bytes),
                include_self=True)
            noc_shares.append(leg_share)
            history_prep += _transfer_ns_shards(
                model.rates, model._noc_registry,
                model.hbm_port_registry,
                paths_by_rank=leg_paths,
                bytes_by_rank=leg_bytes,
                include_self=True, kind="copy", noc_share=leg_share)
            per_rank = builder.transfer_legs(
                f"mvS{copy_leg_count - 1}", leg_paths, leg_bytes,
                prior_chain=prior_chain if copy_leg_count > 1 else {
                    rank_index: join_leg for rank_index in range(tp)},
                )
            # 逐区间串行（旧 Σ legs）：区间 k 的腿依赖区间 k-1 的全 rank 腿。
            prior_chain = dict(per_rank)
            move_done_legs = list(per_rank.values()) or move_done_legs
        builder.note(
            "boundary_move_bytes={}".format(
                sum(layout.copy_bytes_by_rank_for(instance_index, tp))))
        builder.note(f"boundary_copy_legs={copy_leg_count}")
        builder.note(
            "boundary_copy_sources="
            + ",".join(str(holder) for holder, _interval
                       in layout.copy_plan_for(instance_index)))
    elif action == ACTION_RECOMPUTE:
        rebuild_tokens = max(0, int(boundary.history_tokens))
        if rebuild_tokens > 0:
            profile_fn = getattr(model, "prefill_compute_profile", None)
            if callable(profile_fn):
                rebuild_pure_ns, rebuild_hbm_bytes = profile_fn(
                    rebuild_tokens, 0)
                rebuild_base_ns = int(rebuild_pure_ns)
            else:
                if model.prefill_task_load_ns_fn is not None:
                    try:
                        rebuild_base_ns = int(model.prefill_task_load_ns_fn(
                            rebuild_tokens, 0, hbm_share_divisor=1))
                    except TypeError:
                        rebuild_base_ns = int(model.prefill_task_load_ns_fn(
                            rebuild_tokens, 0))
                else:
                    rebuild_base_ns = (
                        rebuild_tokens * model.prefill_ns_per_token)
                rebuild_hbm_bytes = int(math.ceil(
                    max(0, rebuild_base_ns)
                    * float(model.rates.local_hbm_bytes_per_ns)))
            _mark_aggregate_profile_if_needed(
                builder, model, "prefill", profile_fn)
            rebuild_startup_ns, rebuild_controller_ns = _latency_profile(
                model, "prefill", rebuild_tokens, 0)
            _mark_latency_profile_if_needed(builder, model, "prefill")
            control_latency_ns += rebuild_controller_ns
            rebuild_factor = float(model.service_factors.prefill_factor)
            if _die_provider is not None:
                rebuild_ns = int(max(0, rebuild_base_ns))
                rebuild_hbm_field = int(math.ceil(
                    max(0, rebuild_hbm_bytes) * rebuild_factor))
            else:
                rebuild_ns = int(math.ceil(
                    max(0, rebuild_base_ns) * rebuild_factor))
                rebuild_hbm_field = int(rebuild_hbm_bytes)
            rebuild_die_plans = _die_rebuild_descriptors(
                model, rebuild_tokens=rebuild_tokens, exec_ranks=exec_ranks,
                service_factor=rebuild_factor)
            for rank_index in range(tp):
                move_done_legs.append(builder.compute_leg(
                    f"rebuild#R{rank_index}",
                    route=(hbm_resource_key(exec_ranks[rank_index]),),
                    bytes_amount=(
                        0 if rebuild_die_plans is not None
                        else rebuild_hbm_field),
                    base_ns=(
                        0.0 if rebuild_die_plans is not None
                        else float(rebuild_ns)), group="prefill",
                    latency_ns=rebuild_startup_ns,
                    deps=[join_leg],
                    die_plan=(
                        None if rebuild_die_plans is None
                        else rebuild_die_plans[rank_index])))
        builder.note(f"boundary_rebuild_tokens={rebuild_tokens}")
        builder.note("boundary_rebuild_silent=1")
        builder.note("boundary_rebuild_history_base=0")

    # Decode compute events wait for all preparation/rebuild work.  The
    # aggregate controller fixed leg is emitted after the service legs below;
    # this preserves FACE's post-attention controller term without charging it
    # once per layer/tile.
    decode_compute_deps = list(move_done_legs)

    # ---- decode 服务腿（本地 KV 读：裸 n̂_B 步计算链）。 ----
    # §9.6 第 2 条：die 后端的 dec 腿携带地平线 DiePlanDescriptor（协议
    # 推进；本地 KV = local——与执行侧发射同区分）。
    dec_die_plans_local = _die_decode_descriptors(
        model, steps=steps,
        base_context_tokens=int(boundary.history_tokens),
        exec_ranks=exec_ranks, service_factor=decode_factor,
        kv_identity="local")
    if _die_provider is not None and dec_die_plans_local is None:
        builder.note("die_decode_leg=aggregate_service_estimate")
    if action == ACTION_RECOMPUTE:
        for rank_index in range(tp):
            builder.compute_leg(
                f"dec#R{rank_index}",
                route=(hbm_resource_key(exec_ranks[rank_index]),),
                bytes_amount=(
                    0 if dec_die_plans_local is not None
                    else decode_service_field),
                base_ns=(
                    0.0 if dec_die_plans_local is not None
                    else float(decode_compute_ns)),
                group="decode",
                latency_ns=decode_startup_ns,
                deps=move_done_legs,
                die_plan=(
                    None if dec_die_plans_local is None
                    else dec_die_plans_local[rank_index]))
        builder.note(f"boundary_rebuild_ns={rebuild_ns}")
    else:
        for rank_index in range(tp):
            builder.compute_leg(
                f"dec#R{rank_index}",
                route=(hbm_resource_key(exec_ranks[rank_index]),),
                bytes_amount=(
                    0 if dec_die_plans_local is not None
                    else decode_service_field),
                base_ns=(
                    0.0 if dec_die_plans_local is not None
                    else float(decode_compute_ns)),
                group="decode",
                latency_ns=decode_startup_ns,
                deps=move_done_legs,
                die_plan=(
                    None if dec_die_plans_local is None
                    else dec_die_plans_local[rank_index]))
    decode_service_deps_for_controller = [
        leg_id for leg_id in builder.leg_ids()
        if leg_id.startswith(f"{builder.work_id}#dec")]
    controller_leg = builder.fixed_leg(
        "controller", float(control_latency_ns),
        deps=(decode_service_deps_for_controller or move_done_legs))
    builder.note(f"controller_post_latency_ns={control_latency_ns}")
    # ---- 结算（consolidate：轮末布局确定性合并，migrated_ids 语义；
    # distributed（W9 §7.4）：目录提交型——零 merge 载荷，settle 腿空）。 ----
    merge_ns = 0
    plan = None
    home = boundary.home_instance
    settle_extra_deps: List[str] = []
    if distributed_settle:
        builder.note(SETTLEMENT_DISTRIBUTED_NOTE)
    elif home is not None and home != instance_index:
        logic_home = (
            int(boundary.logic_home_instance)
            if boundary.logic_home_instance is not None
            else int(home))
        migrated_ids: Tuple[str, ...] = ()
        if action == ACTION_COPY:
            migrated_ids = tuple(
                interval.replica_id for interval in missing)
        elif action == ACTION_RECOMPUTE:
            migrated_ids = tuple(
                interval.replica_id for interval in layout.intervals)
        end_state = _project_d_end_state(
            layout, instance_index, growth_ranks, steps,
            int(boundary.history_tokens), migrated_ids, logic_home)
        plan = model.merge_from_layout(end_state, notes=builder.notes)
        merge_ns = int(plan.merge_ns)

    # ---- §7.2 继承义务：未完 P 义务解析为图节点（参竞争/门控 settle）。 ----
    approximations = builder.approximations
    obligation_works: List[WorkItem] = []
    # (2026-10-01 审查修复) 路径缓存按 (src, dst) 取键：旧实现只在第一条
    # 端点非空义务上建立路由族、此后所有义务复用——端点不同的义务腿
    # 全部按第一条义务的端点计价（资源键/跳数错）。
    _ob_paths_by_endpoint: dict = {}
    for obligation_id in inherited_obligation_ids:
        view = (
            obligation_resolver(str(obligation_id))
            if obligation_resolver is not None else None)
        if view is None:
            approximations.append(
                f"inherited_obligation_unresolved:{obligation_id}")
            continue
        if view.done:
            continue          # 已完 P 义务零腿（不重付沉没费用）
        legs_ob: List["TaskSpecLike"] = []
        for rank_index, shard_bytes in enumerate(
                view.remaining_bytes_by_tp_rank):
            if int(shard_bytes) <= 0:
                continue
            src = view.source_instance
            dst = view.target_instance
            if src is None or dst is None:
                continue
            _ob_cache_key = (int(src), int(dst))
            if _ob_cache_key not in _ob_paths_by_endpoint:
                _ob_paths_by_endpoint[_ob_cache_key] = model._shard_paths(
                    model._route_paths_pair(int(src), int(dst)),
                    view.remaining_bytes_by_tp_rank)
            leg_paths = _ob_paths_by_endpoint[_ob_cache_key]
            legs_ob.append(_stream_leg(
                f"obl:{obligation_id}#R{rank_index}",
                route=_link_route(model, leg_paths, rank_index,
                                  include_endpoints=True),
                bytes_amount=int(shard_bytes),
                latency_ns=_hop_latency(model, leg_paths, rank_index),
                in_service=True,
                remaining_bytes=int(shard_bytes)))
        if legs_ob:
            for leg in legs_ob:
                builder._register_route(leg.route)
            obligation_works.append(WorkItem(
                work_id=f"obl:{obligation_id}",
                legs=tuple(legs_ob)))
            if view.gate_settle:
                settle_extra_deps.extend(
                    f"work:obl:{obligation_id}" for _ in (0,))

    service_deps_for_settle = [
        leg_id for leg_id in builder.leg_ids()
        if leg_id.startswith(f"{builder.work_id}#dec")] or move_done_legs
    service_deps_for_settle.append(controller_leg)
    settle_ids = builder.settle_legs_from_plan(
        plan,
        service_deps=service_deps_for_settle + settle_extra_deps)
    divisor = max(
        (share.max_path_divisor() for share in noc_shares),
        default=max(1, model._noc_registry.divisor_multi(
            model._route_paths_pair(source, instance_index),
            include_self=True)))
    if not isinstance(divisor, int):
        divisor = int(math.ceil(divisor))
    single_step_service_ns = int(
        _decode_step_total_ns(
            model, base_context_tokens=int(boundary.history_tokens),
            steps=1) * model.service_factors.decode_factor)
    builder.audits.update({
        AUDIT_TARGET_WAIT: int(target_wait),
        AUDIT_HISTORY_PREP: history_prep,
        AUDIT_COMPUTE: int(decode_compute_ns + rebuild_ns),
        AUDIT_MERGE: merge_ns,
        "eviction_wait_ns": int(space_ready),
        "contention_divisor": divisor,
        "hops": int(hops),
        "interval_layout": tuple(
            (interval.replica_id, interval.role,
             int(interval.holder_instance), bool(interval.authority),
             int(interval.token_start), int(interval.token_end),
             tuple(int(value) for value in interval.bytes_by_tp_rank))
            for interval in layout.intervals),
        "inherited_obligation_ids": tuple(
            str(obligation_id) for obligation_id
            in inherited_obligation_ids),
        "phase_space_bytes": tuple(
            tuple(int(value) for value in phase)
            for phase in (footprint.prepare, footprint.prefill,
                          footprint.decode, footprint.merge)),
        "is_continuation": bool(is_continuation),
        "estimated_remaining_decode_steps": int(steps),
        "prefill_instance": int(boundary.p_instance_index),
        "decode_instance": int(instance_index),
        "prefill_action": boundary.p_action,
        "decode_action": action,
        "decision_stage": "D",
        "move_done_ns": int(train_join_ns),
        "single_step_service_ns": single_step_service_ns,
        "control_latency_ns": int(control_latency_ns),
        "decode_hbm_latency_ns": int(decode_startup_ns),
        "decode_controller_latency_ns": int(decode_controller_ns),
        "rebuild_hbm_latency_ns": int(rebuild_startup_ns),
        "rebuild_controller_latency_ns": int(rebuild_controller_ns),
        "copy_effective_bytes": (
            sum(int(v) for v in layout.copy_bytes_by_rank_for(
                instance_index, tp))
            if action == ACTION_COPY else 0),
        "obligation_works": tuple(obligation_works),
    })
    return builder.finish(service_excluded=settle_ids)


def _project_d_end_state(layout, instance_index, growth_ranks, steps,
                         history_tokens, migrated_ids, logic_home):
    """D 入口轮末布局投影（project_end_state 包装——m_end 位置无上限）。"""
    from joint.joint_cost_model import project_end_state
    return project_end_state(
        intervals=layout.intervals,
        decode_instance=int(instance_index),
        decode_growth_by_rank=growth_ranks,
        decode_growth_tokens=steps if any(growth_ranks) else 0,
        history_tokens=history_tokens,
        migrated_interval_ids=migrated_ids,
        logic_home=logic_home)
