"""因果新增需求的 shadow 预测、成熟账和受控资源模板。

类别为 new_session_first/session_return × 已揭示类型；recent 外推
λ_fast，reverting 使用 λ_slow + exp(-h/τ)·(λ_fast-λ_slow)。固定时间
桶积分后按冻结序号阈值生成完整代表请求，模板使用纯计算与 HBM 工作量
的共同 profile、TP 屏障和池恢复腿，预算在展开前检查，无 RNG。

当前有限模板的放置、恢复量和返回下界仍在共同快照上冻结，没有实现
《joint仓库当下+历史修改方案》§5.4/§8.4 的可延长共同到达生成器及候选
目录/容量/驱逐内物化。因此生产 ForecastAugmentedWorldSource 始终将
模板隔离为 shadow，并披露 template_uncovered；独立验证登记不能替代
缺失的物理能力。已知账本背景仍完整前推，不受预测登记时域截断。
RequestWorkTemplate/build_plan 的物化接口用于受控图验证，不能宣称
生产 C/D 计价已经实现。none 档不生成新增需求。

桶预测以 [start,end) 的真实到达计数恰一次成熟，迟交付不扩大真值窗口。
回退链和独立验证质量门由 forecast_validation.py 统一裁决；窗口、预算、
先验与容差均来自 PredictionParams，不按当前 run 的误差自评转正。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field, replace
from types import MappingProxyType
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from joint.action_work_graph import (
    WorldContext,
    WorldSource,
    hbm_resource_key,
    link_resource_key,
    pool_resource_key,
)
from joint.forecast_validation import (
    DistanceAdmissionLedger,
    ForecastMaturityDriver,
    ForecastOverheadLedger,
    distance_band_for_offset,
    run_fallback_chain,
)
from joint.history_estimator import (
    ARRIVAL_CLASS_NEW_SESSION_FIRST,
    ARRIVAL_CLASS_SESSION_RETURN,
    ConditionalRemainingLengthEstimator,
    EpochLog,
    ArrivalStatistics,
    FallbackRecord,
    FallbackCode,
    ForecastMaturityLedger,
    NOT_NEW_ARRIVAL_EXCLUSIONS,
)
from joint.prediction_config import (
    FUTURE_DEMAND_NONE,
    FUTURE_DEMAND_RECENT,
    FUTURE_DEMAND_REVERTING,
    HISTORY_USE_SHADOW,
    JointPredictionConfig,
    PredictionParams,
)
from joint.resource_projection import (
    BarrierSpec,
    ResourceSpec,
    RESOURCE_KIND_HBM,
    RESOURCE_KIND_LINK,
    RESOURCE_KIND_POOL,
    TASK_KIND_COMPUTE,
    TASK_KIND_FIXED,
    TASK_KIND_STREAM,
    TaskSpec,
    WorkItem,
)

__all__ = [
    "FORECAST_REVEALED_TYPES",
    "ForecastFacts",
    "ForecastFactsError",
    "TemplateParams",
    "RequestWorkTemplate",
    "ForecastWorkRecord",
    "ForecastInjection",
    "ForecastPlan",
    "BackgroundForecastEngine",
    "ForecastAugmentedWorldSource",
    "default_facts_from_model",
    "count_instance_background_legs",
]

#: 类别键的已揭示类型白名单（sh30 ``_joint_revealed_category`` 同源取值
#: 域：None/human/tool——W6 白名单通道，不从 manifest 读隐藏真值）。
FORECAST_REVEALED_TYPES: Tuple[Optional[str], ...] = (None, "human", "tool")

#: 背景模板腿的 decode 计算组名（ProjectionEfficiency.gamma 键，缺省 1.0）。
_TEMPLATE_DECODE_GROUP = "decode"
_TEMPLATE_PREFILL_GROUP = "prefill"

#: 注入不确定标记（record.uncertainty 与世界 approximations 共用标签）。
UNCERTAINTY_PLACEMENT = "forecast_placement_load_approx"
UNCERTAINTY_POOL_PORT_PROXY = "forecast_pool_port_rank_proxy"
UNCERTAINTY_DECODE_AGGREGATE = "forecast_decode_aggregate"
UNCERTAINTY_KV_LOCATION_UNKNOWN = "forecast_kv_location_unknown"
UNCERTAINTY_PREFILL_CONTEXT_PRESET = "forecast_prefill_context_preset"
UNCERTAINTY_RESTORE_PRESET = "forecast_restore_preset"
UNCERTAINTY_CLOSED_LOOP_DEFER = "forecast_closed_loop_floor_deferred"

# 生产缺少候选状态内物化与可按需延长的共同到达生成器；有限模板只能
# 用于 shadow/受控图验证，验证登记不能补齐物理模型能力。
_FORECAST_PHYSICAL_WORLD_ERROR = "missing_candidate_state_and_extensible_arrivals"


class ForecastFactsError(ValueError):
    """fail-closed：模板事实缺失（→ template_uncovered 回退，不虚构）。"""


# ------------------------------------------------------------ 模板事实 --


@dataclass(frozen=True)
class ForecastFacts:
    """模板物化所需的硬件/模型事实（同量硬件口径，不从被测 trace 反推）。

    * ``instance_ranks``：实例 → TP rank 序（放置候选集）；
    * ``pool_edge_by_rank``：rank → 池边缘端口 rank（缺省空 = 未知——
      池恢复腿用 rank 代理端口、无 NoC 跳，approx 登记，不虚构链路）；
    * ``kv_token_bytes_per_rank``：每 token 每 rank KV 字节（HBM 腿量）；
    * ``*_compute_ns``/``*_hbm_bytes_per_rank``：代表上下文下的纯计算
      时间与 HBM 总工作量；两者交给共同资源前推，不预先冻结带宽份额；
    * ``*_bytes_per_ns``：资源峰值速率（ResourceSpec 构造）。
    """

    model_layers: int
    instance_tp_size: int
    kv_token_bytes_per_rank: int
    prefill_compute_ns: float
    prefill_hbm_bytes_per_rank: int
    decode_step_compute_ns: float
    decode_step_hbm_bytes_per_rank: int
    local_hbm_bytes_per_ns: float
    noc_link_bytes_per_ns: float
    pool_port_bytes_per_ns: float
    instance_ranks: Mapping[int, Tuple[int, ...]]
    pool_edge_by_rank: Mapping[int, int] = field(default_factory=dict)
    pool_latency_ns: int = 0
    prefill_hbm_latency_ns: int = 0
    prefill_controller_latency_ns: int = 0
    decode_step_hbm_latency_ns: int = 0
    decode_step_controller_latency_ns: int = 0

    def __post_init__(self) -> None:
        for name in ("model_layers", "instance_tp_size",
                     "kv_token_bytes_per_rank"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ForecastFactsError(f"{name} must be a positive integer")
        for name in ("prefill_hbm_bytes_per_rank",
                     "decode_step_hbm_bytes_per_rank", "pool_latency_ns",
                     "prefill_hbm_latency_ns", "prefill_controller_latency_ns",
                     "decode_step_hbm_latency_ns", "decode_step_controller_latency_ns"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ForecastFactsError(f"{name} must be an integer >= 0")
        for name in ("prefill_compute_ns", "decode_step_compute_ns",
                     "local_hbm_bytes_per_ns", "noc_link_bytes_per_ns",
                     "pool_port_bytes_per_ns"):
            value = getattr(self, name)
            minimum_ok = value >= 0 if isinstance(value, (int, float)) else False
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or not minimum_ok
                    or (name.endswith("bytes_per_ns") and value == 0)):
                raise ForecastFactsError(
                    f"{name} has an invalid resource value: {value!r}")
        if not isinstance(self.instance_ranks, Mapping) or not self.instance_ranks:
            raise ForecastFactsError(
                "instance_ranks must be non-empty (no placement candidate)")
        normalized_ranks = {}
        for instance, ranks in self.instance_ranks.items():
            if (type(instance) is not int or instance < 0
                    or not isinstance(ranks, (tuple, list))
                    or len(ranks) != self.instance_tp_size
                    or any(type(rank) is not int or rank < 0 for rank in ranks)
                    or len(set(ranks)) != len(ranks)):
                raise ForecastFactsError(
                    f"instance {instance!r} has an invalid TP rank vector")
            normalized_ranks[instance] = tuple(ranks)
        if (not isinstance(self.pool_edge_by_rank, Mapping)
                or any(type(rank) is not int or rank < 0
                       or type(edge) is not int or edge < 0
                       for rank, edge in self.pool_edge_by_rank.items())):
            raise ForecastFactsError("pool_edge_by_rank must map nonnegative ranks")
        object.__setattr__(self, "instance_ranks", MappingProxyType(normalized_ranks))
        object.__setattr__(self, "pool_edge_by_rank",
                           MappingProxyType(dict(self.pool_edge_by_rank)))


def default_facts_from_model(model) -> ForecastFacts:
    """从 JCM（或同形 duck-typed 夹具）提取模板事实。

    缺关键事实（如 ``remote_read_kv_token_bytes`` 未注入）→
    ``ForecastFactsError`` → 调用方按 template_uncovered 回退（缺项不
    假设，§9.2）。纯计算与 HBM 工作量直接消费共同工作图的 profile
    接口，不能把已经含 HBM 时间的服务模板再次当纯计算时间。
    """
    rates = getattr(model, "rates", None)
    if rates is None:
        raise ForecastFactsError("model.rates missing (hardware facts)")
    context_tokens = int(
        model._resolved_prediction_config().params.forecast_context_tokens
        if hasattr(model, "_resolved_prediction_config") else
        PredictionParams().forecast_context_tokens)
    kv_bytes = getattr(model, "remote_read_kv_token_bytes", None)
    if not kv_bytes:
        raise ForecastFactsError(
            "kv_token_bytes_per_rank unavailable: model.remote_read_"
            "kv_token_bytes not injected (template uncovered)")
    decode_fn = getattr(model, "decode_compute_profile", None)
    prefill_fn = getattr(model, "prefill_compute_profile", None)
    if not callable(decode_fn) or not callable(prefill_fn):
        raise ForecastFactsError("pure-compute/HBM profile interfaces missing")
    decode_step_ns, decode_step_bytes = decode_fn(context_tokens)
    prefill_ns, prefill_bytes = prefill_fn(context_tokens, 0)
    prefill_latency_fn = getattr(model, "prefill_latency_profile", None)
    decode_latency_fn = getattr(model, "decode_latency_profile", None)
    if not callable(prefill_latency_fn) or not callable(decode_latency_fn):
        raise ForecastFactsError("hardware latency profile interfaces missing")
    prefill_hbm_latency, prefill_controller_latency = prefill_latency_fn(context_tokens, 0)
    decode_hbm_latency, decode_controller_latency = decode_latency_fn(context_tokens)
    instance_ranks: Dict[int, Tuple[int, ...]] = {}
    exec_ranks_fn = getattr(model, "_exec_ranks", None)
    loads = getattr(model, "loads", None) or {}
    for instance_index in sorted(loads):
        if exec_ranks_fn is not None:
            instance_ranks[int(instance_index)] = tuple(
                int(rank) for rank in exec_ranks_fn(int(instance_index)))
        else:
            instance_ranks[int(instance_index)] = tuple(
                range(int(getattr(model, "instance_tp_size", 1))))
    return ForecastFacts(
        model_layers=int(getattr(model, "model_layers", 0)),
        instance_tp_size=int(getattr(model, "instance_tp_size", 0)),
        kv_token_bytes_per_rank=int(kv_bytes),
        prefill_compute_ns=prefill_ns,
        prefill_hbm_bytes_per_rank=prefill_bytes,
        decode_step_compute_ns=decode_step_ns,
        decode_step_hbm_bytes_per_rank=decode_step_bytes,
        local_hbm_bytes_per_ns=float(rates.local_hbm_bytes_per_ns),
        noc_link_bytes_per_ns=float(rates.noc_link_bytes_per_ns),
        pool_port_bytes_per_ns=float(rates.pool_port_bytes_per_ns),
        instance_ranks=instance_ranks,
        pool_latency_ns=int(getattr(rates, "pool_latency_ns", 0) or 0),
        prefill_hbm_latency_ns=prefill_hbm_latency,
        prefill_controller_latency_ns=prefill_controller_latency,
        decode_step_hbm_latency_ns=decode_hbm_latency,
        decode_step_controller_latency_ns=decode_controller_latency,
    )


# ------------------------------------------------------------ 模板参数 --


@dataclass(frozen=True)
class TemplateParams:
    """一份代表请求的模板参数（全部带来源/不确定标记）。

    * ``decode_forward_steps``：外生长度代表值 L̂（类别均值 → run 级 →
      冷启动先验；来源与样本量随 uncertainty 携带）；
    * ``context_tokens``：prefill 上下文代表值（preset——W6 无该统计，
      标记不确定，不冒充统计）；
    * ``pool_restore_bytes_per_rank``：session_return 冷层恢复字节
      （preset × kv_token_bytes_per_rank；new_session_first 恒 0——新
      session 无冷层历史）。
    """

    decode_forward_steps: int
    decode_steps_source: str
    decode_steps_samples: int
    context_tokens: int
    pool_restore_bytes_per_rank: int

    def __post_init__(self) -> None:
        for name in ("decode_forward_steps", "decode_steps_samples",
                     "context_tokens", "pool_restore_bytes_per_rank"):
            value = getattr(self, name)
            if (type(value) is not int or value < 0
                    or (name == "decode_forward_steps" and value == 0)):
                raise ForecastFactsError(f"{name} has an invalid template amount")
        if not isinstance(self.decode_steps_source, str) or not self.decode_steps_source:
            raise ForecastFactsError("decode_steps_source must be a nonempty string")

    def uncertainty(self) -> Dict[str, str]:
        out = {
            "decode_steps_source": self.decode_steps_source,
            "decode_steps_samples": str(self.decode_steps_samples),
            "prefill_context": "preset",
            "pool_restore": "preset",
        }
        return out


def derive_template_params(
        *, arrival_class: str, revealed_type: Optional[str],
        length_estimator: Optional[ConditionalRemainingLengthEstimator],
        facts: ForecastFacts, params: PredictionParams,
) -> TemplateParams:
    """模板参数派生（§8.4"从已揭示统计派生，缺样本回退冷启动 preset"）。

    decode 步数 = W6 条件剩余估计器在 d=0 的类别级均值（类别键 =
    revealed_type；命中 run 级/先验时来源标签随行），下限 1（正尾保
    证——仍活跃请求不估零步，§8.2 同纪律）。
    """
    if length_estimator is not None:
        estimate = length_estimator.estimate_conditional(
            session_id="__forecast_aggregate__",
            exposed_progress_steps=0,
            category=revealed_type)
        steps = max(1, int(estimate.remaining_forward_steps))
        source = estimate.source
        samples = int(estimate.level_samples)
    else:
        steps = max(1, int(params.cold_start_forward_steps))
        source = "cold_start_prior"
        samples = 0
    restore_tokens = (
        int(params.forecast_return_restore_tokens)
        if arrival_class == ARRIVAL_CLASS_SESSION_RETURN else 0)
    return TemplateParams(
        decode_forward_steps=steps,
        decode_steps_source=source,
        decode_steps_samples=samples,
        context_tokens=int(params.forecast_context_tokens),
        pool_restore_bytes_per_rank=(
            restore_tokens * int(facts.kv_token_bytes_per_rank)),
    )


# -------------------------------------------------------------- 模板 --


def _split_layers(total_amount: float, layer_count: int) -> Tuple[int, ...]:
    """先向上取整总量再逐层均分，时间与字节都保持整数总量守恒。"""
    total = max(0, int(math.ceil(total_amount)))
    base, remainder = divmod(total, int(layer_count))
    return tuple(
        base + (1 if index < remainder else 0)
        for index in range(int(layer_count)))


@dataclass(frozen=True)
class RequestWorkTemplate:
    """类别级资源工作模板（compute/HBM/NoC/TP/池腿的构造面）。

    ``materialize`` 把模板放在给定 rank 集上生成一份完整代表请求的
    WorkItem（prefill 逐层 compute/HBM 腿 + 逐层 TP barrier + decode 聚合
    compute/HBM 腿 + decode TP barrier +（返回类）池
    恢复 stream 腿）。适用条件 = 类别键匹配 + facts 完备；不确定标记
    经 ``uncertainty_tags`` 随记录与世界披露。
    """

    arrival_class: str
    revealed_type: Optional[str]
    template_params: TemplateParams

    @property
    def category(self) -> Tuple[str, Optional[str]]:
        return (self.arrival_class, self.revealed_type)

    def uncertainty_tags(self, pool_edge_known: bool) -> Tuple[str, ...]:
        tags = [
            UNCERTAINTY_KV_LOCATION_UNKNOWN,
            UNCERTAINTY_DECODE_AGGREGATE,
            UNCERTAINTY_PREFILL_CONTEXT_PRESET,
            "aggregate_phase_latency_folded",
        ]
        if self.template_params.pool_restore_bytes_per_rank > 0:
            tags.append(UNCERTAINTY_RESTORE_PRESET)
            if not pool_edge_known:
                tags.append(UNCERTAINTY_POOL_PORT_PROXY)
        return tuple(tags)

    def materialize(
            self, *, work_id: str, arrival_ns: int, ranks: Tuple[int, ...],
            facts: ForecastFacts,
    ) -> Tuple[WorkItem, Tuple[BarrierSpec, ...], Tuple[ResourceSpec, ...]]:
        """物化一份代表请求（确定性：同输入同输出，无 RNG）。

        ``arrival_ns`` 为世界相对时刻（工作级 ready_at——无显式 deps 的
        腿继承；未来到达经 W2 TIMER/TASK_READY 事件在其时刻激活）。
        """
        if not isinstance(work_id, str) or not work_id:
            raise ForecastFactsError("work_id must be a nonempty string")
        if type(arrival_ns) is not int or arrival_ns < 0:
            raise ForecastFactsError("arrival_ns must be an integer >= 0")
        if (len(ranks) != facts.instance_tp_size
                or any(type(rank) is not int or rank < 0 for rank in ranks)
                or len(set(ranks)) != len(ranks)):
            raise ForecastFactsError("materialization needs a complete TP rank vector")
        layers = int(facts.model_layers)
        params = self.template_params
        legs: List[TaskSpec] = []
        barriers: List[BarrierSpec] = []
        resources: Dict[str, ResourceSpec] = {}

        def _need_hbm(rank: int) -> None:
            key = hbm_resource_key(rank)
            resources.setdefault(key, ResourceSpec(
                key=key, kind=RESOURCE_KIND_HBM,
                capacity_per_ns=facts.local_hbm_bytes_per_ns))

        # -- 1. 池恢复腿（session_return 冷层；new_session_first 无） --
        pool_leg_by_rank: Dict[int, str] = {}
        restore_bytes = int(params.pool_restore_bytes_per_rank)
        if restore_bytes > 0:
            for rank in ranks:
                edge = int(facts.pool_edge_by_rank.get(rank, rank))
                route: List[str] = [pool_resource_key(edge)]
                resources.setdefault(route[0], ResourceSpec(
                    key=route[0], kind=RESOURCE_KIND_POOL,
                    capacity_per_ns=facts.pool_port_bytes_per_ns))
                if edge != rank:
                    link_key = link_resource_key(edge, rank)
                    route.append(link_key)
                    resources.setdefault(link_key, ResourceSpec(
                        key=link_key, kind=RESOURCE_KIND_LINK,
                        capacity_per_ns=facts.noc_link_bytes_per_ns))
                _need_hbm(rank)
                route.append(hbm_resource_key(rank))
                leg_id = f"{work_id}#restore:r{rank}"
                legs.append(TaskSpec(
                    leg_id=leg_id, kind=TASK_KIND_STREAM,
                    route=tuple(route), bytes_amount=restore_bytes,
                    latency_ns=int(facts.pool_latency_ns)))
                pool_leg_by_rank[rank] = leg_id

        # -- 2. prefill 逐层计算腿 + 逐层 TP barrier（§6.2 图形态同源） --
        shares = _split_layers(facts.prefill_compute_ns, layers)
        hbm_shares = _split_layers(facts.prefill_hbm_bytes_per_rank, layers)
        # The aggregate phase startup precedes the first layer, matching the
        # physical profile placement in action_work_graph.py.
        latency_shares = tuple(
            facts.prefill_hbm_latency_ns if index == 0 else 0
            for index in range(layers))
        last_barrier: Optional[str] = None
        for layer_index in range(layers):
            layer_ids: List[str] = []
            barrier_id = f"{work_id}:tp:L{layer_index}"
            for rank in ranks:
                deps: List[str] = []
                if layer_index == 0 and rank in pool_leg_by_rank:
                    deps.append(pool_leg_by_rank[rank])
                if last_barrier is not None:
                    deps.append(last_barrier)
                _need_hbm(rank)
                leg_id = f"{work_id}#pf:L{layer_index}r{rank}"
                legs.append(TaskSpec(
                    leg_id=leg_id, kind=TASK_KIND_COMPUTE,
                    route=(hbm_resource_key(rank),),
                    bytes_amount=hbm_shares[layer_index],
                    base_ns=float(shares[layer_index]),
                    latency_ns=latency_shares[layer_index],
                    group=_TEMPLATE_PREFILL_GROUP,
                    deps=tuple(deps),
                    signals_barrier=barrier_id))
                layer_ids.append(leg_id)
            barriers.append(BarrierSpec(barrier_id, tuple(layer_ids)))
            last_barrier = barrier_id

        prefill_done = last_barrier
        if facts.prefill_controller_latency_ns:
            prefill_done = f"{work_id}#pf:controller"
            legs.append(TaskSpec(
                leg_id=prefill_done, kind=TASK_KIND_FIXED,
                duration_ns=facts.prefill_controller_latency_ns,
                deps=((last_barrier,) if last_barrier is not None else ())))

        # -- 3. decode 聚合 compute 腿（L̂ 步 × 每步纯计算/HBM profile） --
        decode_ids: List[str] = []
        steps = int(params.decode_forward_steps)
        decode_bytes = steps * facts.decode_step_hbm_bytes_per_rank
        decode_barrier = f"{work_id}:tp:decode"
        for rank in ranks:
            _need_hbm(rank)
            leg_id = f"{work_id}#dec:r{rank}"
            legs.append(TaskSpec(
                leg_id=leg_id, kind=TASK_KIND_COMPUTE,
                route=(hbm_resource_key(rank),),
                bytes_amount=decode_bytes,
                base_ns=float(steps) * float(facts.decode_step_compute_ns),
                latency_ns=steps * facts.decode_step_hbm_latency_ns,
                group=_TEMPLATE_DECODE_GROUP,
                deps=((prefill_done,) if prefill_done is not None else ()),
                signals_barrier=decode_barrier))
            decode_ids.append(leg_id)
        # decode TP barrier（聚合——跨 rank max；approx 登记）。
        barriers.append(BarrierSpec(decode_barrier, tuple(decode_ids)))
        if facts.decode_step_controller_latency_ns:
            legs.append(TaskSpec(
                leg_id=f"{work_id}#dec:controller", kind=TASK_KIND_FIXED,
                duration_ns=steps * facts.decode_step_controller_latency_ns,
                deps=(decode_barrier,)))

        work = WorkItem(
            work_id=work_id, legs=tuple(legs), ready_at_ns=int(arrival_ns))
        return work, tuple(barriers), tuple(resources.values())


# ------------------------------------------------------------- 记录 --


@dataclass(frozen=True)
class ForecastWorkRecord:
    """一份预测工作的 forecast_origin/类别/排除集合登记（§8.3）。"""

    forecast_origin: str
    category: Tuple[str, Optional[str]]
    arrival_tick_ns: int                 # 绝对时刻（成熟账目/披露口径）
    distance_band: str
    admitted: bool
    exclusion_set: Tuple[str, ...]
    placement_instance: int
    uncertainty: Mapping[str, str]
    closed_loop_deferred: bool = False

    def as_dict(self) -> Dict[str, Any]:
        return {
            "forecast_origin": self.forecast_origin,
            "category": [self.category[0], self.category[1]],
            "arrival_tick_ns": self.arrival_tick_ns,
            "distance_band": self.distance_band,
            "admitted": self.admitted,
            "exclusion_set": list(self.exclusion_set),
            "placement_instance": self.placement_instance,
            "uncertainty": dict(self.uncertainty),
            "closed_loop_deferred": self.closed_loop_deferred,
        }


@dataclass(frozen=True)
class ForecastInjection:
    """一条已物化的注入（记录 + 世界内容腿）。"""

    record: ForecastWorkRecord
    work: WorkItem
    barriers: Tuple[BarrierSpec, ...]
    resources: Tuple[ResourceSpec, ...]
    arrival_offset_ns: int               # 世界相对时刻（世界 now 基）


@dataclass(frozen=True)
class ForecastPlan:
    """一次决策的完整预测计划（所有候选共享——冻结快照）。"""

    mode: str                            # 配置臂（recent/reverting/none）
    priced_mode: str                     # 回退/降级后的实际计价档
    injections: Tuple[ForecastInjection, ...]     # 进正式计价的注入
    shadow_records: Tuple[ForecastWorkRecord, ...]  # 降级 shadow（只产误差）
    placeholders_ns: Tuple[int, ...]     # 世界相对到达偏移（重仲裁 tick）
    scenario_key: str
    fallbacks: Tuple[FallbackRecord, ...]
    accounting: Mapping[str, Any]

    @property
    def zero_behaviour(self) -> bool:
        """B 等价（零注入：none 档或致命回退）。"""
        return not self.injections


# ------------------------------------------------------------- 引擎 --


class BackgroundForecastEngine:
    """背景预测引擎（§8.3/§8.4 落地；每决策一次 build_plan）。

    组件（构造注入或缺省自建；全部 PredictionParams 单源）：

    * ``arrival_stats``：W6 ArrivalStatistics（只读查询——λ 反映已闭合
      桶，open bucket = full_buckets_only 规则不进预测）；
    * ``length_estimator``：W6 条件剩余估计器（类别均值派生）；
    * ``epoch_log`` / ``maturity_ledger`` / ``distance_ledger`` /
      ``overhead_ledger``：§8.5/§8.7 组件（forecast_validation）。
    """

    def __init__(
            self,
            *,
            arrival_stats: ArrivalStatistics,
            prediction_config: JointPredictionConfig,
            length_estimator: Optional[
                ConditionalRemainingLengthEstimator] = None,
            epoch_log: Optional[EpochLog] = None,
            maturity_ledger: Optional[ForecastMaturityLedger] = None,
            distance_ledger: Optional[DistanceAdmissionLedger] = None,
            overhead_ledger: Optional[ForecastOverheadLedger] = None,
            revealed_types: Sequence[Optional[str]] = (
                FORECAST_REVEALED_TYPES),
    ) -> None:
        self._arrival_stats = arrival_stats
        self._config = prediction_config
        self._params = prediction_config.params
        self._length_estimator = length_estimator
        self._epoch_log = epoch_log if epoch_log is not None else EpochLog(
            self._params)
        self._maturity_ledger = (
            maturity_ledger if maturity_ledger is not None
            else ForecastMaturityLedger(self._params))
        self._distance_ledger = (
            distance_ledger if distance_ledger is not None
            else DistanceAdmissionLedger(
                self._params, validation_registry={
                    tuple(key.split("|")): record
                    for key, record in prediction_config.validation_registry.items()}))
        self._overhead = (
            overhead_ledger if overhead_ledger is not None
            else ForecastOverheadLedger())
        self._revealed_types = tuple(revealed_types)
        self._maturity_driver = ForecastMaturityDriver(
            arrival_stats=arrival_stats,
            maturity_ledger=self._maturity_ledger,
            distance_ledger=self._distance_ledger)
        self._decision_seq = 0
        self._last_plan_disclosure: Dict[str, Any] = {
            "priced_mode": FUTURE_DEMAND_NONE,
            "priced_templates": 0, "shadow_records": 0, "fallbacks": [],
        }

    # -------------------------------------------------------- 组件 --
    @property
    def overhead_ledger(self) -> ForecastOverheadLedger:
        return self._overhead

    @property
    def distance_ledger(self) -> DistanceAdmissionLedger:
        return self._distance_ledger

    @property
    def maturity_ledger(self) -> ForecastMaturityLedger:
        return self._maturity_ledger

    def scenario_key(self) -> str:
        """决策前可计算的快照缓存键（模式 + 统计版本 + 准入版本）。"""
        length_version = (
            int(self._length_estimator.version)
            if self._length_estimator is not None else -1)
        return "|".join((
            "fc",
            self._config.future_demand_mode,
            self._config.history_use,
            f"a{int(self._arrival_stats.version)}",
            f"l{length_version}",
            f"e{len(self._epoch_log.epochs)}",
            f"d{int(self._distance_ledger.version)}",
        ))

    def mature(self, now_ns: int):
        """桶预测成熟驱动（决策批挂点调用；差分真值，恰一次）。"""
        return self._maturity_driver.mature(now_ns)

    def disclosure(self) -> Dict[str, Any]:
        return {
            "mode": self._config.future_demand_mode,
            "history_use": self._config.history_use,
            "priced_mode": self._last_plan_disclosure["priced_mode"],
            "last_plan": dict(self._last_plan_disclosure),
            "scenario_key": self.scenario_key(),
            "maturity": self._maturity_driver.as_dict(),
            "overhead": self._overhead.as_dict(),
            "arrival_exclusions": self._arrival_stats.exclusion_records(),
        }

    # -------------------------------------------------------- 计划 --
    def build_plan(
            self,
            *,
            now_ns: int,
            facts: ForecastFacts,
            session_return_floor_offset_ns: int = 0,
            instance_background_legs: Optional[Mapping[int, int]] = None,
            facts_provider_error: Optional[str] = None,
            physical_world_error: Optional[str] = None,
    ) -> ForecastPlan:
        """构造一次决策的背景预测计划（确定性：同输入同输出）。

        ``session_return_floor_offset_ns``：闭环返回约束下界（相对 now），
        由调度侧按"当前处理中 session 的最早 service_done 估计"供给
        （共享等待假设）；session_return 注入时刻低于下界时推迟锚点
        （保持期望计数——聚合返回模型，§8.3）。``facts_provider_error``
        = 事实提取失败原因（→ template_uncovered）。

        ``physical_world_error`` 由生产世界适配器提供能力缺口；有缺口时
        即使独立验证登记通过也只记录 shadow，桶预测照常成熟。

        选择隔离（§8.7 先 shadow 门槛）：``history_use=shadow`` 档全部
        crossing 记 shadow_records（零 injections、``priced_mode=
        shadow_only``）——历史可以生成预测和成熟误差，但不影响正式选
        择；正式计价只经 ``history_use=validated`` ＋ 显式验证登记
        （无登记 (mode, band) 按 UNVALIDATED_DISTANCE 降级 shadow）。
        """
        started_ns = time.monotonic_ns()
        self._decision_seq += 1
        mode = self._config.future_demand_mode
        key = self.scenario_key()
        zero = self._zero_plan(mode, key)

        # none 档 = 零行为（缺省；现状态逐字节不变——不含任何世界内容）。
        if mode == FUTURE_DEMAND_NONE:
            self._account(zero, started_ns, ())
            return zero

        observed_until = self._arrival_stats.observed_until_ns
        if observed_until is not None and observed_until > now_ns:
            raise ValueError("forecast decision precedes its observed arrival history")
        floor = max(0, int(session_return_floor_offset_ns))

        categories = self._forecast_categories()
        # ---- 跨桶积分：逐类别逐桶 μ_b 与序号阈值跨阈（确定性） ----
        intensity_values = self._intensity_probe_values(
            now_ns=now_ns, mode=mode, categories=categories)
        if any(not math.isfinite(value) or value < 0.0
               for value in intensity_values):
            crossings, requested_templates = [], 0
        else:
            crossings, requested_templates = self._intensity_crossings(
                now_ns=now_ns, mode=mode, categories=categories)

        # ---- 回退链（§8.5 固定顺序；forecast_validation 单一裁决） ----
        chain = run_fallback_chain(
            config=self._config,
            arrival_stats=self._arrival_stats,
            epoch_log=self._epoch_log,
            now_ns=now_ns,
            facts_available=(facts_provider_error is None
                             and facts is not None),
            requested_templates=requested_templates,
            bands_used=[distance_band_for_offset(
                max(tick, floor) if cat[0] == ARRIVAL_CLASS_SESSION_RETURN
                else tick, self._params)
                for tick, cat, _band in crossings],
            distance_ledger=self._distance_ledger,
            intensity_values=intensity_values,
            params=self._params)
        fallbacks: List[FallbackRecord] = []
        if chain.fatal is not None:
            plan = self._fallback_plan(mode, key, chain.fatal)
            self._account(plan, started_ns, (chain.fatal.code.value,))
            return plan
        if chain.unvalidated_record is not None:
            fallbacks.append(chain.unvalidated_record)
        if physical_world_error is not None:
            fallbacks.append(FallbackRecord(
                code=FallbackCode.TEMPLATE_UNCOVERED, at_ns=now_ns,
                origin=f"future_demand:{mode}",
                detail={"physical_world_error": physical_world_error,
                        "effective_mode": "shadow_only"}))

        # ---- 物化（placement = 背景腿数最少实例，平局小序号；冻结） ----
        legs_count = instance_background_legs or {}
        placement_instance = min(
            sorted(int(i) for i in facts.instance_ranks),
            key=lambda i: (int(legs_count.get(i, 0)), i))
        ranks = tuple(facts.instance_ranks[placement_instance])
        pool_edge_known = any(
            rank in facts.pool_edge_by_rank for rank in ranks)
        unvalidated = set(chain.unvalidated_bands)
        # §8.7"先 shadow"门槛（选择隔离）：``history_use=shadow`` 档下
        # 全部 crossing 只发桶预测、只产成熟误差（shadow 记录），零
        # injections——不进 J 正式计价、不影响候选选择。显式研究臂经
        # ``history_use=validated`` ＋ 显式验证登记表达（§9.1：标实验
        # 用途；无登记自动降级 shadow_only）。
        shadow_isolation = (
            self._config.history_use == HISTORY_USE_SHADOW
            or physical_world_error is not None)

        injections: List[ForecastInjection] = []
        shadow: List[ForecastWorkRecord] = []
        templates = [
            RequestWorkTemplate(
                arrival_class=arrival_class,
                revealed_type=revealed,
                template_params=derive_template_params(
                    arrival_class=arrival_class, revealed_type=revealed,
                    length_estimator=self._length_estimator,
                    facts=facts, params=self._params))
            for (arrival_class, revealed) in categories]
        template_by_category = {
            template.category: template for template in templates}

        for seq, (tick_offset, category, band) in enumerate(crossings):
            template = template_by_category[category]
            deferred = False
            offset = int(tick_offset)
            if (category[0] == ARRIVAL_CLASS_SESSION_RETURN
                    and offset < floor):
                offset = int(floor)     # 闭环返回约束：不早于 service_done
                deferred = True
            band = distance_band_for_offset(offset, self._params)
            admitted = (
                not shadow_isolation and band not in unvalidated)
            record = ForecastWorkRecord(
                forecast_origin=(
                    f"{mode}:{category[0]}|{category[1] or ''}"
                    f"@{now_ns}#{seq}"),
                category=category,
                arrival_tick_ns=int(now_ns) + offset,
                distance_band=band,
                admitted=admitted,
                exclusion_set=tuple(NOT_NEW_ARRIVAL_EXCLUSIONS),
                placement_instance=placement_instance,
                uncertainty=(
                    dict(
                        template.template_params.uncertainty(),
                        placement="load_approx",
                        # (2026-10-01 审查修复) 闭环推迟的不确定标记接
                        # 线：UNCERTAINTY_CLOSED_LOOP_DEFER 声明
                        # "record.uncertainty 与世界 approximations 共用
                        # 标签"，但 build_plan 的推迟分支从未把它写进
                        # record.uncertainty——超阈值推迟此前只能靠布尔
                        # 字段单独辨认。此处补齐标签面。
                        **({UNCERTAINTY_CLOSED_LOOP_DEFER:
                            "arrival_clamped_to_service_done_floor"}
                           if deferred else {}))),
                closed_loop_deferred=deferred)
            if not admitted:
                # shadow 档（选择隔离）/距离未验证 → 只产误差，不进正式
                # 计价（§8.7 先 shadow 门槛）。
                shadow.append(record)
                continue
            work_id = f"fc:{self._decision_seq}:{seq}:{category[0]}|{(
                category[1] or '')}"
            work, barriers, resources = template.materialize(
                work_id=work_id, arrival_ns=offset, ranks=ranks,
                facts=facts)
            injections.append(ForecastInjection(
                record=record, work=work, barriers=barriers,
                resources=resources, arrival_offset_ns=offset))

        # ---- 桶强度预测成熟登记（issued_at/matures_at 全记录） ----
        issued = self._issue_bucket_forecasts(now_ns, mode, categories)

        priced_mode = mode if injections else (
            "none" if not shadow else "shadow_only")
        plan = ForecastPlan(
            mode=mode,
            priced_mode=priced_mode,
            injections=tuple(injections),
            shadow_records=tuple(shadow),
            placeholders_ns=tuple(sorted({
                injection.arrival_offset_ns
                for injection in injections})),
            scenario_key=key,
            fallbacks=tuple(fallbacks),
            accounting={
                "decision_seq": self._decision_seq,
                "requested_templates": requested_templates,
                "priced_templates": len(injections),
                "shadow_records": len(shadow),
                "legs": sum(len(injection.work.legs)
                            for injection in injections),
                "barriers": sum(len(injection.barriers)
                                for injection in injections),
                "bucket_forecasts_issued": issued,
                "placement_instance": placement_instance,
                "uncertainty_tags": sorted({
                    tag for injection in injections
                    for tag in template_by_category[
                        injection.record.category].uncertainty_tags(
                            pool_edge_known)}),
            })
        self._account(
            plan, started_ns,
            [record.code.value for record in fallbacks])
        return plan

    # ------------------------------------------------------------ 内部 --
    def _forecast_categories(self) -> List[Tuple[str, Optional[str]]]:
        """预测类别枚举（确定性序：类 × 已揭示类型白名单字典序）。"""
        return [
            (arrival_class, revealed)
            for arrival_class in sorted((
                ARRIVAL_CLASS_NEW_SESSION_FIRST,
                ARRIVAL_CLASS_SESSION_RETURN))
            for revealed in self._revealed_types
        ]

    def _bucket_count(self) -> int:
        delta = int(self._params.arrival_bucket_delta_ns)
        return (int(self._params.forecast_horizon_ns) + delta - 1) // delta

    def _bucket_span(self, bucket: int) -> Tuple[int, int]:
        """有限登记窗口的末桶只覆盖剩余时间，不遗漏也不越界。"""
        delta = int(self._params.arrival_bucket_delta_ns)
        start = bucket * delta
        return start, min(delta, int(self._params.forecast_horizon_ns) - start)

    def _intensity(
            self, category: Tuple[str, Optional[str]], *,
            mode: str, horizon_ns: float) -> float:
        if mode == FUTURE_DEMAND_RECENT:
            return float(self._arrival_stats.intensity_recent(category))
        if mode == FUTURE_DEMAND_REVERTING:
            return float(self._arrival_stats.intensity_reverting(
                category, horizon_ns))
        raise ValueError(f"unsupported forecast mode {mode!r}")

    def _intensity_probe_values(
            self, *, now_ns: int, mode: str,
            categories: Sequence[Tuple[str, Optional[str]]]
    ) -> List[float]:
        """回退链 anomaly 检查用的强度值序列（首桶中点口径）。"""
        first_duration = min(int(self._params.arrival_bucket_delta_ns),
                             int(self._params.forecast_horizon_ns))
        probe: List[float] = []
        for category in categories:
            probe.append(self._intensity(
                category, mode=mode, horizon_ns=first_duration * 0.5))
        return probe

    def _intensity_crossings(
            self, *, now_ns: int, mode: str,
            categories: Sequence[Tuple[str, Optional[str]]],
    ) -> Tuple[List[Tuple[int, Tuple[str, Optional[str]], str]], int]:
        """逐类别逐桶积分 + 序号阈值跨阈（§8.4 确定性第一版）。

        返回 (跨阈列表, 所需模板数)。模板超预算时只计数、不展开请求，
        交给固定回退链整档处理。桶内位相规则在合法预算内保持一致。
        """
        theta = float(self._params.forecast_ordinal_threshold)
        budget = int(self._params.forecast_template_budget)
        requested = 0
        results: List[Tuple[int, Tuple[str, Optional[str]], str]] = []
        for category in categories:
            accumulator = 0.0
            for bucket in range(self._bucket_count()):
                start, duration = self._bucket_span(bucket)
                midpoint_ns = start + duration * 0.5
                intensity = self._intensity(
                    category, mode=mode, horizon_ns=midpoint_ns)
                mu = max(0.0, intensity) * duration
                before = accumulator
                accumulator += mu
                if not math.isfinite(accumulator / theta):
                    return [], budget + 1
                crossings = int(math.floor(accumulator / theta) - math.floor(
                    before / theta))
                requested += crossings
                if requested > budget:
                    continue
                for j in range(crossings):
                    phase = int(((j + 0.5) / crossings) * duration)
                    offset = start + phase
                    results.append((
                        offset, category,
                        distance_band_for_offset(
                            offset, self._params)))
        results.sort(key=lambda item: (item[0], repr(item[1])))
        return (results if requested <= budget else []), requested

    def _issue_bucket_forecasts(
            self, now_ns: int, mode: str,
            categories: Sequence[Tuple[str, Optional[str]]]) -> int:
        """桶强度预测登记（成熟账本：issued_at/matures_at 全记录）。"""
        issued = 0
        for category in categories:
            for bucket in range(self._bucket_count()):
                start, duration = self._bucket_span(bucket)
                midpoint_ns = start + duration * 0.5
                intensity = self._intensity(
                    category, mode=mode, horizon_ns=midpoint_ns)
                mu = max(0.0, intensity) * duration
                forecast_id = (
                    f"fc:{self._decision_seq}:{bucket}:{category[0]}|{(
                        category[1] or '')}")
                if self._maturity_ledger.entry(forecast_id) is not None:
                    continue
                self._maturity_ledger.issue(
                    forecast_id=forecast_id,
                    issued_at_ns=int(now_ns),
                    matures_at_ns=int(now_ns) + start + duration,
                    origin=f"forecast_bucket:{mode}",
                    predicted=float(mu),
                    category=category,
                    exclusion_set=tuple(NOT_NEW_ARRIVAL_EXCLUSIONS))
                self._maturity_driver.note_issued(
                    forecast_id=forecast_id, mode=mode,
                    starts_at_ns=int(now_ns) + start,
                    band=distance_band_for_offset(
                        midpoint_ns, self._params))
                issued += 1
        return issued

    def _zero_plan(self, mode: str, key: str) -> ForecastPlan:
        return ForecastPlan(
            mode=mode, priced_mode=FUTURE_DEMAND_NONE, injections=(),
            shadow_records=(), placeholders_ns=(), scenario_key=key,
            fallbacks=(), accounting={
                "decision_seq": self._decision_seq,
                "requested_templates": 0, "priced_templates": 0,
                "shadow_records": 0, "legs": 0, "barriers": 0,
                "bucket_forecasts_issued": 0,
                "placement_instance": None, "uncertainty_tags": [],
            })

    def _fallback_plan(
            self, mode: str, key: str, record: FallbackRecord) -> ForecastPlan:
        plan = self._zero_plan(mode, key)
        return ForecastPlan(
            mode=mode, priced_mode=FUTURE_DEMAND_NONE,
            injections=(), shadow_records=(), placeholders_ns=(),
            scenario_key=key, fallbacks=(record,),
            accounting=dict(plan.accounting,
                            fallback_code=record.code.value,
                            fallback_detail=dict(record.detail)))

    def _account(self, plan: ForecastPlan, started_ns: int,
                 fallback_codes: Sequence[str]) -> None:
        self._last_plan_disclosure = {
            "priced_mode": plan.priced_mode,
            "priced_templates": len(plan.injections),
            "shadow_records": len(plan.shadow_records),
            "bucket_forecasts_issued": plan.accounting.get("bucket_forecasts_issued", 0),
            "fallbacks": [{"code": record.code.value,
                           "at_ns": record.at_ns, "origin": record.origin,
                           "detail": dict(record.detail)} for record in plan.fallbacks],
        }
        self._overhead.note_decision(
            templates_injected=int(
                plan.accounting.get("priced_templates", 0)),
            shadow_records=len(plan.shadow_records),
            legs_generated=int(plan.accounting.get("legs", 0)),
            bucket_forecasts_issued=int(
                plan.accounting.get("bucket_forecasts_issued", 0)),
            host_time_ns=int(time.monotonic_ns() - started_ns),
            fallback_codes=tuple(fallback_codes))


# ------------------------------------------------------------- 世界源 --


def count_instance_background_legs(
        context: WorldContext,
        facts: ForecastFacts) -> Dict[int, int]:
    """各实例 TP rank 上的背景腿计数（因果放置规则的输入，§8.4）。

    只数内源背景工作（真实在册/账本腿）；候选自身与预测模板不计
    （放置决定对所有候选共享一次——同一情景）。
    """
    counts = {
        instance: sum(
            1 for work in context.background_works
            for leg in work.legs
            for rank in ranks
            if hbm_resource_key(int(rank)) in leg.route)
        for instance, ranks in facts.instance_ranks.items()}
    return counts


class ForecastAugmentedWorldSource(WorldSource):
    """生产预测的 shadow 适配器，保留已知工作世界的完整前推。

    当前模板缺少候选目录/容量内物化及可延长到达生成器，故生产不把
    有限模板注入正式计价；仍生成 shadow 与成熟账，并披露有效模式。
    ``scenario_key()`` 供 JCM 快照缓存，统计版本变化时重新观测。
    """

    progress_source = "forecast_augmented"

    def __init__(
            self,
            inner: WorldSource,
            *,
            engine: BackgroundForecastEngine,
            now_ns: int,
            facts_provider: Optional[
                Callable[[Any], ForecastFacts]] = None,
            session_return_floor_offset_ns: int = 0,
    ) -> None:
        self._inner = inner
        self._engine = engine
        self._now_ns = int(now_ns)
        self._facts_provider = (
            facts_provider if facts_provider is not None
            else default_facts_from_model)
        self._floor_offset_ns = int(session_return_floor_offset_ns)

    def scenario_key(self) -> str:
        return self._engine.scenario_key()

    def build_context(self, model) -> WorldContext:
        inner_context = self._inner.build_context(model)
        facts_error: Optional[str] = None
        facts: Optional[ForecastFacts] = None
        try:
            facts = self._facts_provider(model)
        except ForecastFactsError as exc:
            facts_error = str(exc)
        plan = self._engine.build_plan(
            now_ns=self._now_ns,
            facts=facts if facts is not None else _EMPTY_FACTS,
            session_return_floor_offset_ns=self._floor_offset_ns,
            instance_background_legs=(
                count_instance_background_legs(inner_context, facts)
                if facts is not None else None),
            facts_provider_error=facts_error,
            physical_world_error=_FORECAST_PHYSICAL_WORLD_ERROR)
        approximations = list(inner_context.approximations) + [
            f"forecast_scenario:{plan.scenario_key}",
            f"forecast_priced_mode:{plan.priced_mode}",
        ] + list(plan.accounting.get("uncertainty_tags", ()))
        if plan.fallbacks:
            approximations.extend(
                f"forecast_fallback:{record.code.value}"
                for record in plan.fallbacks)
        return replace(
            inner_context,
            approximations=tuple(approximations),
            forecast_scenario_key=plan.scenario_key)


#: 事实提取失败时的占位（build_plan 的 facts_available=False 分支只读
#: 空映射；不进物化路径——template_uncovered 回退先短路）。
_EMPTY_FACTS = ForecastFacts(
    model_layers=1, instance_tp_size=1, kv_token_bytes_per_rank=1,
    prefill_compute_ns=0.0, prefill_hbm_bytes_per_rank=0,
    decode_step_compute_ns=0.0, decode_step_hbm_bytes_per_rank=0,
    local_hbm_bytes_per_ns=1.0, noc_link_bytes_per_ns=1.0,
    pool_port_bytes_per_ns=1.0, instance_ranks={0: (0,)},
)
