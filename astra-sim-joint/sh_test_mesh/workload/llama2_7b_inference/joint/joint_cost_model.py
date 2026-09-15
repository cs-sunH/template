"""joint_cost_model.py -- J：无 oracle 的 (instance × action) 完成时间预测。

设计依据：《三机制联合策略_template仓库设计方案》§6、§9.2（实验 joint
选择性移植的六项缺口修正）与实验总纲 §13.2。

统一预测目标：``cost(instance, action)`` 预测到同一 ``service_done``
边界的完成时间——包括目标等待、历史准备、执行期增长、remote 路径、
compute_done 后的 home 释放与合并（§2.3 三时刻合同）。重叠工作按依赖
关键路径处理，不机械相加，不把同一拥塞重复计费。

单位合同（v2 教训，§6）：后端链路带宽换算为十进制 SI，``1 GB/s ==
1 B/ns``（数值恒等）；``from_gbps`` 是唯一构造路径。bit/s 与 byte/s
显式换算；全部时间为 ns 整数（速率用 float 只作除法中间量）。

无 oracle（§9.2 缺口 1，结构性排除）：

* trace 的 ``decode_length`` / ``final_context_tokens`` 不是本模块的
  输入（签名不接受）；执行期 decode 工作量用因果时域估计器
  （session 已完成轮次在线均值 → 本 run 均值 → 冷启动模型缺省并
  标记），且决策日志必须披露冷启动/估计误差状态；
* 当前请求输入在真实到达后可用；已生成 decode 长度仅在实际生成后
  可用；未来 CSV 行、未来返回时间一律不进入。

争用模型（§6 + phase5 v3 教训）：有效速率 = 链路峰值带宽 / 该有向
链路上的仲裁份额除数（含候选自身流与已知在途流，含自身多跳路线的
自重叠）；不用 ``B_peak × (1−utilization)``。合并流同样登记。跨
collective 的观测缺口由调用方以覆盖标记披露（本模型不假定其为零）。

状态（§5.6 表，R15 后 2026-09-14）：本模块按"已实现（解析近似 +
在线因子）"交付，且在线反馈通道**已接线**——``LinkFlowRegistry`` 由
调度器在五类传输发射时逐 shard 全路径登记、完成事件注销（F-B 聚合
除数）；``ServiceFactors`` 三观测入口挂列车完成事件（P3 α 公式；
transfer 因子保留接口位——节点级传输完成遥测未交付，updates=0 披
露，传输争用在线修正由除数通道承担）；池端口除数注入（含 E 内核
r_j，R15-3）。``collective_coverage`` 仍由调用方按遥测覆盖披露。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Optional, Sequence

ACTION_STAY = "stay"
ACTION_RECOMPUTE = "recompute"
ACTION_COPY = "copy"
ACTION_REMOTE = "remote-read"

#: 确定平局序：动作按此序、实例按索引（§13.1 不按事后输赢调 tie）。
ACTION_ORDER = (ACTION_STAY, ACTION_RECOMPUTE, ACTION_COPY, ACTION_REMOTE)
_ACTION_PRIORITY = {name: index for index, name in enumerate(ACTION_ORDER)}


class JointCostError(ValueError):
    """fail-closed：非法单位、非法状态视图或 oracle 输入。"""


# ============================================================ 硬件速率 ==


@dataclass(frozen=True)
class JointHardwareRates:
    """配置派生物理速率（bytes/ns）与时延（ns）。

    来源：run 的硬件配置（face_case5_config_c.json 派生的
    FaceHardware 字段）。``from_gbps`` 是唯一构造路径（1 GB/s == 1
    B/ns，数值恒等——v1 曾把 GB/s 乘 1e9 导致全部传输低估 1e9 倍）。
    """

    noc_link_bytes_per_ns: float       # D2D 单链路带宽
    pool_port_bytes_per_ns: float      # 片外池单端口带宽
    local_hbm_bytes_per_ns: float      # 单 rank 本地 HBM 带宽
    d2d_latency_ns: int                # 单跳链路时延
    pool_latency_ns: int               # 池端口事务时延

    @classmethod
    def from_gbps(
        cls,
        *,
        noc_link_gbps: float,
        pool_port_gbps: float,
        local_hbm_gbps: float,
        d2d_latency_ns: int,
        pool_latency_ns: int,
    ) -> "JointHardwareRates":
        rates = cls(
            noc_link_bytes_per_ns=float(noc_link_gbps),
            pool_port_bytes_per_ns=float(pool_port_gbps),
            local_hbm_bytes_per_ns=float(local_hbm_gbps),
            d2d_latency_ns=int(d2d_latency_ns),
            pool_latency_ns=int(pool_latency_ns),
        )
        return rates.validate()

    def validate(self) -> "JointHardwareRates":
        for name in ("noc_link_bytes_per_ns", "pool_port_bytes_per_ns",
                     "local_hbm_bytes_per_ns"):
            value = getattr(self, name)
            if not (value > 0) or not math.isfinite(value):
                raise JointCostError(f"{name} must be a positive finite rate")
        if self.d2d_latency_ns < 0 or self.pool_latency_ns < 0:
            raise JointCostError("latencies must be non-negative")
        return self

    def as_dict(self) -> dict:
        return {
            "noc_link_bytes_per_ns": self.noc_link_bytes_per_ns,
            "pool_port_bytes_per_ns": self.pool_port_bytes_per_ns,
            "local_hbm_bytes_per_ns": self.local_hbm_bytes_per_ns,
            "d2d_latency_ns": self.d2d_latency_ns,
            "pool_latency_ns": self.pool_latency_ns,
        }


# ============================================================ 在线估计 ==


class CausalHorizonEstimator:
    """decode 长度因果时域（§13.2 预测时域；phase5 v2 教训）。

    session 已完成轮次的 decode 长度在线均值 → 本 run 已完成轮次均值
    → 冷启动缺省（调用方注入，须有物理来源并披露）。``observe`` 仅在
    请求真实完成后可调；``estimate`` 返回 (tokens, source) 供决策日志
    披露冷启动与估计来源。
    """

    def __init__(self, cold_start_default_tokens: int) -> None:
        if cold_start_default_tokens <= 0:
            raise JointCostError("cold start horizon must be positive")
        self._cold_default = int(cold_start_default_tokens)
        self._session_totals: dict[str, list[int]] = {}
        self._run_count = 0
        self._run_sum = 0
        # M3（kimi 复审）：在线估计器版本——task-load 快照的隐藏输入，
        # 供快照缓存失效判定（估计器更新不 bump 实例纪元，漏查会让
        # 跨实例缓存陈旧 + SH_SNAPSHOT_VERIFY 影子断言假阳性）。
        self.version = 0

    def observe_completed(self, session_id: str, decode_tokens: int) -> None:
        if decode_tokens < 0:
            raise JointCostError("completed decode length must be >= 0")
        self._session_totals.setdefault(session_id, []).append(decode_tokens)
        self._run_count += 1
        self._run_sum += decode_tokens
        self.version += 1

    def estimate(self, session_id: str) -> tuple[int, str]:
        samples = self._session_totals.get(session_id)
        if samples:
            mean = self._run_mean_of(samples)
            return mean, "session_online_mean"
        if self._run_count:
            return self._run_sum // self._run_count, "run_online_mean"
        return self._cold_default, "cold_start_default"

    @staticmethod
    def _run_mean_of(samples: Sequence[int]) -> int:
        return sum(samples) // len(samples)


@dataclass
class ServiceFactors:
    """在线服务效率因子（§5.4 同规则：EWMA、因果、无拟合 clamp）。

    R15-2/P3（2026-09-14）：``α_n = 1 − exp(−Δt_n / τ_n)`` 时间衰减公式
    （设计方案 §5.4 原文；τ_n = 本组上一有效样本的**正服务时长**；Δt_n
    = 距上一有效样本的决策 tick 间隔；首样本直接初始化）。同时刻样本先
    按实际量/基础量汇总再更新（Σactual/Σbase，一条更新）；零时长/零
    分母/非有限样本不更新并计数披露。样本纯净性约束：``actual_ns`` 必
    须是**纯服务段**实测（排队/传输/merge 尾段不得入分子——调用方保证
    收集口径；本模块只拒绝非正样本）。更新仅由已完成事件驱动，因果
    合规；首次观测前为 1.0（roofline 冷启动，标记乐观可能）。
    """

    prefill_factor: float = 1.0
    decode_factor: float = 1.0
    transfer_factor: float = 1.0
    updates: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)
    # P3 时间衰减状态（组内私有）：上一有效样本的 tick 与正服务时长。
    _last_tick_ns: dict = field(default_factory=dict)
    _last_service_ns: dict = field(default_factory=dict)
    # 同时刻样本汇总缓冲（group -> [tick, Σactual, Σbase, Σservice]）。
    _pending: dict = field(default_factory=dict)

    # ------------------------------------------------------------ 观测 --
    def observe_prefill(
        self, *, actual_ns: int, base_ns: int, service_ns: int,
        tick_ns: int,
    ) -> None:
        self._record("prefill_factor", actual_ns, base_ns, service_ns,
                     tick_ns)

    def observe_decode(
        self, *, actual_ns: int, base_ns: int, service_ns: int,
        tick_ns: int,
    ) -> None:
        self._record("decode_factor", actual_ns, base_ns, service_ns,
                     tick_ns)

    def observe_transfer(
        self, *, actual_ns: int, base_ns: int, service_ns: int,
        tick_ns: int,
    ) -> None:
        self._record("transfer_factor", actual_ns, base_ns, service_ns,
                     tick_ns)

    def flush(self) -> None:
        """把同时刻汇总缓冲落进因子（决策时刻读因子前调用；幂等）。"""
        for name in tuple(self._pending):
            self._apply_pending(name)

    # ------------------------------------------------------------ 内部 --
    def _record(
        self, name: str, actual_ns: int, base_ns: int, service_ns: int,
        tick_ns: int,
    ) -> None:
        if (
            not math.isfinite(actual_ns) or not math.isfinite(base_ns)
            or actual_ns <= 0 or base_ns <= 0 or service_ns <= 0
        ):
            # 零时长/零分母/非有限计量：不更新，计数披露（§5.4）。
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return
        pending = self._pending.get(name)
        if pending is None:
            self._pending[name] = [tick_ns, actual_ns, base_ns, service_ns]
            return
        if pending[0] != tick_ns:
            # 时刻推进：先落上一时刻的汇总样本。
            self._apply_pending(name)
            self._pending[name] = [tick_ns, actual_ns, base_ns, service_ns]
            return
        pending[1] += actual_ns
        pending[2] += base_ns
        pending[3] += service_ns

    def _apply_pending(self, name: str) -> None:
        pending = self._pending.pop(name, None)
        if pending is None:
            return
        _tick, actual_sum, base_sum, service_sum = pending
        ratio = actual_sum / base_sum
        if not math.isfinite(ratio) or ratio <= 0:
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return
        current = getattr(self, name)
        if self.updates.get(name, 0) == 0:
            # 首样本直接初始化（§5.4）。
            setattr(self, name, float(ratio))
        else:
            delta_t = max(0, _tick - self._last_tick_ns.get(name, _tick))
            tau = self._last_service_ns.get(name, service_sum)
            alpha = 1.0 - math.exp(-(delta_t / tau) if tau > 0 else 1.0)
            setattr(self, name, (1 - alpha) * current + alpha * ratio)
        self._last_tick_ns[name] = _tick
        self._last_service_ns[name] = service_sum
        self.updates[name] = self.updates.get(name, 0) + 1

    def as_dict(self) -> dict:
        return {
            "prefill_factor": self.prefill_factor,
            "decode_factor": self.decode_factor,
            "transfer_factor": self.transfer_factor,
            "updates": dict(self.updates),
            "rejected": dict(self.rejected),
        }


# ======================================================== 链路流登记表 ==


class LinkFlowRegistry:
    """有向链路在途流登记表（聚合争用除数的因果来源，§6）。

    键 = (source_rank, destination_rank) 有向链路；值 = 该链路上已登记
    的在途流数（含 KV 迁移、恢复、逐出写回与合并流；候选自身流加入时
    以自身占用数计）。跨 collective 的互扰若未被遥测覆盖，调用方保留
    ``collective_coverage=False`` 披露——本模型不把未覆盖流量当作零。

    F-B（2026-09-14，八审）：登记按**逐 shard 流的完整链路序列**——
    TP 并行的 tp_degree 条路径全部登记（非代表对）；除数按候选**全部
    TP 并行流链路并集**取瓶颈（共享链路争用取 max，``include_self``
    按全部自身流计）。按"代表 route_path 登记"字面施工会漏计非代表
    路径上的自身与他流争用（最多 tp_degree−1 倍）；phase5 v3 的
    sorted-rank 分支聚合除数即同问题解法。

    生命周期（R15-1）：``register_path`` 在流发射时登记（携带 owner
    归属键），``release_owner`` 在该归属的完成事件（列车核销 / merge
    尾 watch / 实例空闲清扫）注销——登记/注销全部由已观测完成事件
    驱动，决策时刻快照因果可见。
    """

    def __init__(self) -> None:
        self._flows: dict[tuple[int, int], int] = {}
        self._flow_links: dict[int, list[tuple[int, int]]] = {}
        self._owner_flows: dict[str, list[int]] = {}
        self._next_flow_id = 0
        self.collective_coverage = False
        self.has_registrations = False

    # ------------------------------------------------------------ 原语 --
    def register(self, source_rank: int, destination_rank: int) -> None:
        key = (source_rank, destination_rank)
        self._flows[key] = self._flows.get(key, 0) + 1
        self.has_registrations = True

    def unregister(self, source_rank: int, destination_rank: int) -> None:
        key = (source_rank, destination_rank)
        count = self._flows.get(key, 0)
        if count <= 0:
            raise JointCostError(
                f"unregister on empty link flow {key}: double release")
        if count == 1:
            del self._flows[key]
        else:
            self._flows[key] = count - 1

    # -------------------------------------------------------- 流级登记 --
    def register_path(
        self, path_ranks: Sequence[int], *, owner: str,
    ) -> Optional[int]:
        """登记一条逐 shard 流的完整链路序列；返回流 id（空路径 None）。"""
        if len(path_ranks) < 2:
            return None
        links = [
            (path_ranks[index], path_ranks[index + 1])
            for index in range(len(path_ranks) - 1)
        ]
        flow_id = self._next_flow_id
        self._next_flow_id += 1
        self._flow_links[flow_id] = links
        for link in links:
            self.register(*link)
        self._owner_flows.setdefault(owner, []).append(flow_id)
        return flow_id

    def release_flow(self, flow_id: int) -> None:
        links = self._flow_links.pop(flow_id, None)
        if links is None:
            raise JointCostError(
                f"release on unknown link flow {flow_id}: double release")
        for link in links:
            self.unregister(*link)

    def release_owner(self, owner: str) -> int:
        """注销该归属键的全部在途流（完成事件驱动）；返回注销条数。"""
        flow_ids = self._owner_flows.pop(owner, None)
        if not flow_ids:
            return 0
        for flow_id in flow_ids:
            links = self._flow_links.pop(flow_id, None)
            if links is None:
                continue
            for link in links:
                self.unregister(*link)
        return len(flow_ids)

    # ------------------------------------------------------------ 除数 --
    def divisor(self, path_ranks: Sequence[int], *, include_self: bool,
                self_overlap: int = 1) -> int:
        """单路线仲裁除数（有向链路 = 已登记流数 + 候选自身占用数；
        多跳路线自重叠按经过次数；结果取路线瓶颈链路最大值）。"""
        if len(path_ranks) < 2:
            return 1
        worst = 1
        for index in range(len(path_ranks) - 1):
            key = (path_ranks[index], path_ranks[index + 1])
            flows = self._flows.get(key, 0)
            share = flows + (self_overlap if include_self else 0)
            worst = max(worst, max(1, share))
        return worst

    def divisor_multi(
        self, paths: Sequence[Sequence[int]], *, include_self: bool = True,
    ) -> int:
        """F-B 聚合除数：候选全部 TP 并行流链路并集取瓶颈。

        每条链路上的自身份额 = 候选自身路径经过该链路的次数（TP 并行
        各 shard 路径共享同一链路时按全部自身流计）；共享 = 已登记流数
        + 自身次数；结果取并集内瓶颈链路最大值。
        """
        if not paths:
            return 1
        self_counts: dict[tuple[int, int], int] = {}
        for path in paths:
            if len(path) < 2:
                continue
            for index in range(len(path) - 1):
                key = (path[index], path[index + 1])
                self_counts[key] = self_counts.get(key, 0) + 1
        if not self_counts:
            return 1
        worst = 1
        for key, self_share in self_counts.items():
            flows = self._flows.get(key, 0)
            share = flows + (self_share if include_self else 0)
            worst = max(worst, max(1, share))
        return worst

    def snapshot(self) -> dict:
        return {f"{src}->{dst}": count for (src, dst), count
                in sorted(self._flows.items())}


# ============================================================ 决策视图 ==


@dataclass(frozen=True)
class InstanceLoadView:
    """单实例只读负载视图（ns 计的 roofline 服务台账，非队列长度）。

    R3'（2026-09-14）：``reclaimable_bytes_by_tp_rank`` = 逐出全部
    inactive 会话后的可用字节（KV 账本派生）——驱逐等待定价区分
    "逐出可解"（写回搬运时间）与"深缺口"（等活跃完成，取活跃剩余
    负载峰值），不再一律按池写回计价。
    """

    instance_index: int
    queued_task_load_ns: int
    running_task_load_ns: int
    active_decode_task_load_ns: int
    # 目标增长与 home 合并两处空间压力分别标记（§7 容量诊断口径）。
    hbm_remaining_bytes_by_tp_rank: tuple[int, ...]
    reclaimable_bytes_by_tp_rank: tuple[int, ...] = ()

    @property
    def total_task_load_ns(self) -> int:
        return (
            self.queued_task_load_ns
            + self.running_task_load_ns
            + self.active_decode_task_load_ns)


@dataclass(frozen=True)
class SessionKVView:
    """session KV 的决策时点只读视图（来自 KV 账本，无未来信息）。"""

    session_id: str
    home_instance: Optional[int]        # 逻辑 home（§2.1；逐出不清除）
    resident_instance: Optional[int]    # 当前驻留实例（LOCAL/PARTIAL）
    location: str                       # local_hbm/partial_hbm_remote/remote_memory/none
    history_tokens: int
    resident_prefix_layers: int
    # 逐 rank 历史 KV 字节（resident 部分；partial 时仅前缀层）。
    history_bytes_by_tp_rank: tuple[int, ...]
    # 缺失后缀逐 rank 字节（PARTIAL 时 = 后缀层；REMOTE 时 = 全量）。
    missing_bytes_by_tp_rank: tuple[int, ...]


@dataclass(frozen=True)
class RequestView:
    """已到达请求的已知信息（oracle 结构性排除：无 decode_length）。"""

    request_id: str
    session_id: str
    input_tokens: int                   # 本轮新增输入（已到达，可用）
    history_tokens_before: int
    estimated_decode_tokens: int        # 因果时域估计（来源另行披露）
    horizon_source: str                 # session_online_mean/run_online_mean/cold_start_default
    # 已知 prefill 输入的逐 rank KV 字节（增量，物理派生）。
    input_kv_bytes_by_tp_rank: tuple[int, ...]


@dataclass(frozen=True)
class ActionCostBreakdown:
    """预测分解（审计字段；决策日志逐项落盘，§13.2 代价审计）。"""

    target_wait_ns: int
    history_prep_ns: int
    eviction_wait_ns: int
    compute_ns: int
    remote_read_ns: int
    merge_ns: int
    contention_divisor: int
    hops: int
    notes: tuple = ()


@dataclass(frozen=True)
class ActionCandidate:
    """一个 (instance, action) 候选的完整评估结果。"""

    instance_index: int
    action: str
    applicable: bool
    inapplicable_reason: Optional[str]
    cost_ns: Optional[int]
    breakdown: Optional[ActionCostBreakdown] = None

    def order_key(self) -> tuple:
        return (self.instance_index, _ACTION_PRIORITY.get(
            self.action, len(ACTION_ORDER)))


# ============================================================ 代价估计 ==


def _transfer_ns(
    *,
    total_bytes: int,
    path_hops: int,
    divisor: int,
    rates: JointHardwareRates,
    per_hop_latency_ns: Optional[int],
    startup_ns: int,
) -> int:
    """单服务区间传输时间：startup + bytes / (B_link / divisor)。

    多跳逐跳累计时延按跳数线性计（store-and-forward 物理模型）；除数
    为路线瓶颈链路的仲裁份额。除数恒 >= 1（无登记流 = 独占，而非无
    带宽——利用率 100% 不意味着候选流拿不到带宽，§5.4）。
    """
    if total_bytes < 0:
        raise JointCostError("transfer bytes must be >= 0")
    effective = rates.noc_link_bytes_per_ns / max(1, divisor)
    latency = per_hop_latency_ns if per_hop_latency_ns is not None else (
        rates.d2d_latency_ns * max(0, path_hops))
    data_ns = int(total_bytes / effective) if total_bytes else 0
    return int(startup_ns) + int(latency) + data_ns


def _pool_transfer_ns(
    *, total_bytes: int, divisor: int, rates: JointHardwareRates,
) -> int:
    """池端口事务：端口时延 + bytes / (B_port / divisor)。"""
    effective = rates.pool_port_bytes_per_ns / max(1, divisor)
    data_ns = int(total_bytes / effective) if total_bytes else 0
    return rates.pool_latency_ns + data_ns


@dataclass
class JointCostModel:
    """决策时点代价模型（只读快照 -> 每候选完成时间预测）。

    输入均为因果可见量：硬件速率（配置派生）、实例负载视图（服务台账
    ns）、session KV 视图（账本）、请求已知输入、在线因子与链路流登
    记表。``route_fn`` 给出 (path_ranks, hops)（真实拓扑 XY 路由的
    确定性函数，由调用方注入以避免对 face_scheduler 的导入依赖）。
    """

    rates: JointHardwareRates
    loads: Mapping[int, InstanceLoadView]
    flow_registry: LinkFlowRegistry
    service_factors: ServiceFactors
    # roofline 服务函数（ns/token，事前定义、不按 workload 拟合）。
    prefill_ns_per_token: float
    decode_ns_per_token: float
    model_layers: int
    instance_tp_size: int
    route_fn: object  # Callable[[int, int], tuple[Sequence[int], int]]
    # R15-1/F-B：逐 shard 全路径路由（source, target) -> 全部 TP 并行
    # 路径；None 时退回 route_fn 代表路径（单测/离线口径）。
    route_paths_fn: object = None
    # R15-3：池端口仲裁份额（instance_index -> 除数，含自身 +1）；
    # None 时单流全带宽（冷启动）。
    pool_divisor_fn: object = None

    def __post_init__(self) -> None:
        if self.prefill_ns_per_token <= 0 or self.decode_ns_per_token <= 0:
            raise JointCostError("service rates must be positive")
        if self.model_layers <= 0 or self.instance_tp_size <= 0:
            raise JointCostError("layout parameters must be positive")
        # 决策时刻先落盘同时刻汇总样本（P3：读因子前 flush）。
        self.service_factors.flush()

    # ------------------------------------------------------------ 动作 --
    def applicable_actions(
        self,
        session: SessionKVView,
        request: RequestView,
        instance_index: int,
        *,
        remote_enabled: bool,
    ) -> tuple[tuple[bool, Optional[str]], ...]:
        """四动作在 (session, instance) 的适用性（§13.1：某动作不适用
        不等于删除该 instance，其余动作仍参与比较）。

        N1 裁定 (a)（2026-09-14）：remote-read 要求基础历史**全层驻留**
        home HBM（LOCAL）——PARTIAL 的池后缀不存在"前缀 home + 后缀池"
        混合读流原语（后端能力边界，非 joint 理论排除；partial 会话跨
        实例服务走 copy——前缀 NoC + 后缀池恢复，已实现且正确；实现/
        测试锚点（R16，2026-09-15）：物化 face_scheduler._noc_transfer
        层区间化 + copy 分支 [0, base_prefix)；计价本函数 ACTION_COPY
        两腿 noc_prefix+pool_suffix_restore；单测
        joint/test_joint_review3_fixes.py 22 用例 + 容量压力夹具
        joint_capacity_stress_fixture.sh）。LOCAL 会话 remote-read 不受
        影响；README 动作适用性表按此口径披露。
        """
        results = []
        is_home = (
            session.home_instance == instance_index
            and session.location in ("local_hbm", "partial_hbm_remote"))
        # stay：相应历史在目标本地可用（LOCAL 或 PARTIAL 驻留在目标）；
        # 无历史（turn-0 新会话）在任何 instance 均 stay-型（本地建立）。
        stay_ok = (
            session.history_tokens == 0
            or (
                session.resident_instance == instance_index
                and session.location in ("local_hbm", "partial_hbm_remote")))
        results.append((
            stay_ok,
            None if stay_ok else "history not resident at target"))
        # recompute：任何 instance 恒适用（R13 后重算的是缺失区间——
        # 目标未驻留则整份、驻留则仅池后缀，恒有定义）。
        results.append((True, None))
        # copy：存在需要搬运的历史（别处驻留或池 backing）；基础历史
        # 已驻留目标实例时退化为 stay（不构成第二份工作副本）。
        copy_ok = (
            session.location != "none"
            and session.history_tokens > 0
            and not (
                session.resident_instance == instance_index
                and session.location in ("local_hbm", "partial_hbm_remote")))
        results.append((
            copy_ok,
            None if copy_ok else "no history to copy"))
        # remote-read：基础历史全层驻留在别的 instance（直接远读路径，
        # 读流自 home HBM 全上下文），且 remote 能力开关开启。
        remote_ok = (
            remote_enabled
            and session.location == "local_hbm"
            and session.resident_instance is not None
            and session.resident_instance != instance_index)
        results.append((
            remote_ok,
            None if remote_ok else (
                "remote actions disabled"
                if not remote_enabled
                else "suffix not directly readable at home"
                if session.location == "partial_hbm_remote"
                else "no resident remote history")))
        del is_home  # 保留参数语义：is_home 影响 merge 计费而非适用性
        return tuple(results)

    def estimate_action(
        self,
        *,
        session: SessionKVView,
        request: RequestView,
        instance_index: int,
        action: str,
        remote_enabled: bool,
    ) -> ActionCandidate:
        """单候选代价：到 service_done 边界的完成时间预测（关键路径，
        非机械相加：可重叠段取 max，串行依赖段相加）。

        R3'（2026-09-14）：空间准备按动作感知逐 rank 足迹（与 R1' 预约
        足迹同源——stay/recompute@驻留 = final−resident、copy/recompute
        @异地 = 整份、remote-read = input 增量；一律叠加因果 decode 增长
        估计），深缺口按"活跃剩余负载峰值"计价而非一律池写回。
        R15-1/F-B：NoC 除数按候选全部 TP 并行流链路并集取瓶颈；
        R15-3：池路径除数挂池端口仲裁份额。N11：remote-read 计价基数
        补 input + 在线 decode 增长（步数仍为因果估计，因果边界）。
        """
        applicability = dict(zip(
            ACTION_ORDER,
            self.applicable_actions(
                session, request, instance_index,
                remote_enabled=remote_enabled),
        ))
        ok, reason = applicability[action]
        if not ok:
            return ActionCandidate(
                instance_index=instance_index,
                action=action,
                applicable=False,
                inapplicable_reason=reason,
                cost_ns=None)

        load = self.loads.get(instance_index)
        if load is None:
            raise JointCostError(f"no load view for instance {instance_index}")

        notes: list[str] = []
        # ---- 1. 目标等待：实例已有服务台账（queued/running/active 三
        # 分量已并行重叠在台账口径中，取总量作为本请求可发射前的等待）。
        target_wait = load.total_task_load_ns

        # ---- 2. 空间准备（动作感知逐 rank 足迹，与 R1' 预约同源）。----
        decode_growth_ranks = self._decode_growth_by_tp_rank(request)
        space_needed = self._space_footprint_by_tp_rank(
            session, request, instance_index, action)
        space_needed = tuple(
            base + growth
            for base, growth in zip(space_needed, decode_growth_ranks))
        eviction_wait = self._eviction_wait_estimate(
            load, space_needed, notes, instance_index=instance_index)

        # ---- 3. 历史准备（依动作而异）。----
        history_prep = 0
        remote_read_ns = 0
        route_path, hops = self._route(instance_index, session)
        candidate_paths = self._route_paths(instance_index, session)
        pool_divisor = self._pool_divisor(instance_index)
        if action == ACTION_STAY:
            # LOCAL：无传输；PARTIAL：缺失后缀按池路径恢复（工作区间）。
            missing = sum(session.missing_bytes_by_tp_rank)
            if missing:
                history_prep = _pool_transfer_ns(
                    total_bytes=missing, divisor=pool_divisor,
                    rates=self.rates)
                notes.append("partial_suffix_pool_restore")
        elif action == ACTION_COPY:
            # LOCAL/PARTIAL 别处：复合两腿——驻留前缀经 NoC 复制、缺失
            # 后缀经池端口恢复（R16-3/D2-3，2026-09-15：与物化
            # face_scheduler.prepare_prefill 的 noc_migrate[0,prefix) +
            # remote_load[prefix,L) 同源；旧口径整份全按 NoC 速率对
            # PARTIAL 基系统性低估）；REMOTE：从池恢复必要历史到目标。
            resident_bytes = sum(session.history_bytes_by_tp_rank)
            missing = sum(session.missing_bytes_by_tp_rank)
            if session.location == "remote_memory":
                history_prep = _pool_transfer_ns(
                    total_bytes=missing + resident_bytes,
                    divisor=pool_divisor, rates=self.rates)
                notes.append("pool_restore_to_target")
            else:
                # 后缀腿必须带 missing 守卫：_pool_transfer_ns 对零字节
                # 仍返回端口时延（pool_latency_ns），无守卫会击穿 LOCAL
                # 基与旧值逐位一致（stay 分支 if missing 即先例）。
                divisor = self.flow_registry.divisor_multi(
                    candidate_paths, include_self=True)
                history_prep = _transfer_ns(
                    total_bytes=resident_bytes,
                    path_hops=hops, divisor=divisor,
                    rates=self.rates, per_hop_latency_ns=None,
                    startup_ns=0)
                if missing:
                    history_prep += _pool_transfer_ns(
                        total_bytes=missing, divisor=pool_divisor,
                        rates=self.rates)
                notes.append("noc_prefix+pool_suffix_restore")
        elif action == ACTION_RECOMPUTE:
            # 无历史搬运；重算缺失区间的时间计入计算段（R13：@驻留仅
            # 缺失后缀折算 token，@异地整份历史；§13.2 复用收益体现为
            # 少做的重算或传输）。
            notes.append("recompute_history_in_compute")
        elif action == ACTION_REMOTE:
            # 执行期间按消费远读历史：dense attention 的重复读取按
            # 估计 decode 步数计次（§13.2 remote）。N11：计价基数补
            # input 增量 + 在线 decode 增长（执行侧读终态上下文为后端
            # 真值，不进决策；此处为因果可见口径）。
            read_base = (
                sum(session.history_bytes_by_tp_rank)
                + sum(session.missing_bytes_by_tp_rank)
                + sum(request.input_kv_bytes_by_tp_rank)
                + sum(decode_growth_ranks))
            divisor = self.flow_registry.divisor_multi(
                candidate_paths, include_self=True)
            read_passes = max(
                1, request.estimated_decode_tokens)
            remote_read_ns = _transfer_ns(
                total_bytes=int(read_base) * read_passes,
                path_hops=hops, divisor=divisor,
                rates=self.rates, per_hop_latency_ns=None,
                startup_ns=0)
            notes.append(f"remote_read_passes={read_passes}")
        else:  # pragma: no cover - ACTION_ORDER 封闭
            raise JointCostError(f"unknown action {action!r}")

        # ---- 4. 计算（roofline 基础 × 在线因子；recompute 追加重算
        # 缺失区间的工作量——R13：@驻留目标按池后缀折算 token、@异地
        # 按整份历史；不含排队——排队在 target_wait/eviction 段）。
        prefill_tokens = request.input_tokens
        if action == ACTION_RECOMPUTE:
            prefill_tokens += self._recompute_missing_tokens(
                session, instance_index)
        compute_ns = int(
            prefill_tokens * self.prefill_ns_per_token
            * self.service_factors.prefill_factor
            + request.estimated_decode_tokens
            * self.decode_ns_per_token
            * self.service_factors.decode_factor)

        # ---- 5. compute_done 后的合并（§2.2/§3.2：增量回 origin_home；
        # 执行位置等于 home 的本地提交不生成虚构流量；新会话（home 待
        # 建立 = 执行实例）同 stay 本地提交）。终审-中3 残角（kimi 三审）：
        # REMOTE 基 + exec==home 的工作副本组合（copy@home 退化池恢复 /
        # recompute@home 于 REMOTE 基）执行侧 merge_back 恒走**池写**
        # （face_scheduler REMOTE 归并分支先于 N9 home==exec 防御），计价
        # 不得因 home==exec 免单——门控扩为 home != exec **或** REMOTE 基。
        merge_ns = 0
        if session.home_instance is not None and (
                session.home_instance != instance_index
                or session.location == "remote_memory"):
            # K4（kimi 复审，2026-09-14）+ 自查 C（2026-09-15）+
            # R16-3（GLM 三审，2026-09-15）：merge 段计价按**基础位置**
            # 分流——LOCAL/PARTIAL 基 = 增量口径（input + 因果 decode
            # 增长）按基础驻留前缀**逐 rank 分裂**：前缀增量经 NoC 回
            # home + 后缀增量写池（执行端池端口）+ home 侧空间准备等待
            # （与物化 merge_back 的两笔拆分同源；旧"整份增量回传 NoC"
            # 对 PARTIAL 基是系统性**低估**——池端口单字节速率仅 NoC
            # 的 ~1/7.9，方向经实配速率核实，非代码注释旧称的"保守
            # 上界"）；REMOTE 基 = 执行侧 merge_back 走**池写**
            # （_increment_pool_store_transfer，经执行端池端口），无
            # home 空间准备（home 不持有基础）。此前统一 NoC→home 口径
            # 对 REMOTE 基是路由错配（K4 只修了字节口径）。执行端空间
            # 准备段（space_needed）保持 R3'.2 整份口径不变。
            merge_increments_by_rank = tuple(
                input_bytes + growth
                for input_bytes, growth in zip(
                    request.input_kv_bytes_by_tp_rank, decode_growth_ranks))
            increments = sum(merge_increments_by_rank)
            if session.location == "remote_memory":
                merge_ns = _pool_transfer_ns(
                    total_bytes=increments,
                    divisor=self._pool_divisor(instance_index),
                    rates=self.rates)
                notes.append("merge_to_pool_backing")
            else:
                # 拆分公式形态钉死（GLM 四审 P2-2）：
                # prefix = inc × p // L; suffix = inc − prefix——两腿之和
                # 恒等于 inc（守恒）；LOCAL（p == L）下 prefix = inc、
                # suffix = 0 与旧值逐位一致；inc // L × p 会在 L∤inc 的
                # 合成视图下破坏 LOCAL 一致性，两腿独立取整则丢守恒。
                layers = self.model_layers
                base_prefix = min(
                    max(session.resident_prefix_layers, 0), layers)
                prefix_increments_by_rank = tuple(
                    increment * base_prefix // layers
                    for increment in merge_increments_by_rank)
                prefix_increments = sum(prefix_increments_by_rank)
                suffix_increments = increments - prefix_increments
                merge_route, merge_hops = self._route_pair(
                    instance_index, session.home_instance)
                merge_divisor = self.flow_registry.divisor_multi(
                    self._route_paths_pair(
                        instance_index, session.home_instance),
                    include_self=True)
                merge_ns = _transfer_ns(
                    total_bytes=prefix_increments, path_hops=merge_hops,
                    divisor=merge_divisor, rates=self.rates,
                    per_hop_latency_ns=None, startup_ns=0)
                if suffix_increments > 0:
                    # 后缀池腿守卫与 copy 段同因：_pool_transfer_ns 零
                    # 字节时延陷阱；除数挂执行端池端口仲裁（与 REMOTE
                    # 基分支同源同口径）。
                    merge_ns += _pool_transfer_ns(
                        total_bytes=suffix_increments,
                        divisor=self._pool_divisor(instance_index),
                        rates=self.rates)
                # home 侧空间准备：按 home 当前占用估计释放等待（§3.2 不
                # 认为"写回免费"）；需求 = 前缀增量逐 rank——物化侧
                # _ensure_capacity(home, prefix_shards) 本就只备前缀，
                # 同源自洽（后缀实际写池不占 home）。
                home_load = self.loads.get(session.home_instance)
                if home_load is not None:
                    merge_ns += self._eviction_wait_estimate(
                        home_load, prefix_increments_by_rank, notes,
                        instance_index=session.home_instance,
                        prefix="home_merge")
                notes.append(f"merge_to_home={session.home_instance}")

        # ---- 关键路径合成：目标等待 -> [历史准备 ∥ 空间准备（争用同
        # 链路，保守串行）] -> 计算（remote 与计算重叠推进、取增量） ->
        # 合并。可重叠段取 max：remote 读流与计算相互重叠（§5.2 恢复
        # 争用使计算变慢的口径：以 remote_read 延长计算的保守合成）。
        effective_compute = compute_ns + remote_read_ns
        prep = max(history_prep, eviction_wait)
        cost = target_wait + prep + effective_compute + merge_ns

        divisor = self.flow_registry.divisor_multi(
            candidate_paths, include_self=True)
        return ActionCandidate(
            instance_index=instance_index,
            action=action,
            applicable=True,
            inapplicable_reason=None,
            cost_ns=int(cost),
            breakdown=ActionCostBreakdown(
                target_wait_ns=target_wait,
                history_prep_ns=history_prep,
                eviction_wait_ns=eviction_wait,
                compute_ns=compute_ns,
                remote_read_ns=remote_read_ns,
                merge_ns=merge_ns,
                contention_divisor=divisor,
                hops=hops,
                notes=tuple(notes),
            ),
        )

    # ------------------------------------------------------------ 辅助 --
    def _route(self, instance_index: int, session: SessionKVView):
        source_instance = (
            session.resident_instance
            if session.resident_instance is not None
            else session.home_instance)
        if source_instance is None:
            return ((), 0)
        return self._route_pair(source_instance, instance_index)

    def _route_pair(self, source_instance: int, target_instance: int):
        if source_instance == target_instance:
            return ((source_instance,), 0)
        path, hops = self.route_fn(source_instance, target_instance)
        return tuple(path), int(hops)

    def _route_paths(
        self, instance_index: int, session: SessionKVView,
    ) -> tuple[Sequence[int], ...]:
        """候选自身 TP 并行流的全部路径（F-B 并集除数用）。"""
        source_instance = (
            session.resident_instance
            if session.resident_instance is not None
            else session.home_instance)
        if source_instance is None or source_instance == instance_index:
            return ()
        return self._route_paths_pair(source_instance, instance_index)

    def _route_paths_pair(
        self, source_instance: int, target_instance: int,
    ) -> tuple[Sequence[int], ...]:
        if source_instance == target_instance:
            return ()
        if self.route_paths_fn is None:
            path, _hops = self.route_fn(source_instance, target_instance)
            return (tuple(path),)
        paths = self.route_paths_fn(source_instance, target_instance)
        return tuple(tuple(path) for path in paths)

    def _pool_divisor(self, instance_index: int) -> int:
        """R15-3：池端口仲裁份额（未注入 fn 时单流全带宽）。"""
        if self.pool_divisor_fn is None:
            return 1
        return max(1, int(self.pool_divisor_fn(instance_index)))

    def _decode_growth_by_tp_rank(
        self, request: RequestView,
    ) -> tuple[int, ...]:
        """因果 decode 增长估计的逐 rank KV 字节（比例随 input 分片）。"""
        if request.input_tokens <= 0:
            return tuple(0 for _ in request.input_kv_bytes_by_tp_rank)
        return tuple(
            bytes_by_rank * request.estimated_decode_tokens
            // request.input_tokens
            for bytes_by_rank in request.input_kv_bytes_by_tp_rank)

    def _space_footprint_by_tp_rank(
        self,
        session: SessionKVView,
        request: RequestView,
        instance_index: int,
        action: str,
    ) -> tuple[int, ...]:
        """R3'：动作感知空间足迹（与 R1' 预约足迹同源，逐 rank 精确）。

        stay / recompute@驻留目标：final − resident（驻留前缀复用）；
        copy@异地 / recompute@异地 / REMOTE 基础：整份 final；
        remote-read：仅 input 增量。
        """
        size = self.instance_tp_size
        final_total = sum(session.history_bytes_by_tp_rank) + sum(
            session.missing_bytes_by_tp_rank) + sum(
                request.input_kv_bytes_by_tp_rank)
        final_ranks = self._per_rank(final_total)
        resident_here = (
            session.resident_instance == instance_index
            and session.location in ("local_hbm", "partial_hbm_remote"))
        if action == ACTION_REMOTE:
            return tuple(request.input_kv_bytes_by_tp_rank)
        if resident_here and action in (ACTION_STAY, ACTION_RECOMPUTE):
            return tuple(
                max(0, final - resident)
                for final, resident in zip(
                    final_ranks, session.history_bytes_by_tp_rank))
        return final_ranks

    def _recompute_missing_tokens(
        self,
        session: SessionKVView,
        instance_index: int,
    ) -> int:
        """R13：recompute 需重算的历史 token 折算量。

        @驻留目标（含 home）：仅缺失后缀层折算 ceil(H×(L−prefix)/L)；
        @异地 / REMOTE 基础：整份历史 H（工作副本从零物化）。
        """
        layers = self.model_layers
        history = session.history_tokens
        if history <= 0:
            return 0
        resident_here = (
            session.resident_instance == instance_index
            and session.location == "local_hbm")
        if resident_here and session.resident_prefix_layers >= layers:
            return 0
        if resident_here and session.location == "partial_hbm_remote":
            missing_layers = layers - session.resident_prefix_layers
            return (history * missing_layers + layers - 1) // layers
        return history

    def _eviction_wait_estimate(
        self,
        load: InstanceLoadView,
        needed_bytes_by_tp_rank: Sequence[int],
        notes: list,
        *,
        instance_index: int = -1,
        prefix: str = "execution_growth",
    ) -> int:
        """空间缺口 -> 时间换算（不把无单位容量风险加到延迟上）。

        R3'（2026-09-14）：逐 rank 精确缺口（调用方给逐 rank 足迹）。
        缺口可由逐出 inactive 解决（needed ≤ reclaimable）→ 池写回
        搬运时间（除数 = 池端口仲裁份额）；深缺口（超出 reclaimable，
        只有活跃完成才能释放）→ ``max(写回估计, 活跃任务剩余负载)``，
        并注记 ``deep_gap_unresolved``（docstring 承诺的观测落盘）——
        §13.1"驱逐压力换算成时间"逐字对齐，因果可观测、无 oracle。
        """
        worst_gap = 0
        beyond_reclaimable = 0
        reclaimable = load.reclaimable_bytes_by_tp_rank
        for rank_index, needed in enumerate(needed_bytes_by_tp_rank):
            remaining = (
                load.hbm_remaining_bytes_by_tp_rank[rank_index]
                if rank_index < len(load.hbm_remaining_bytes_by_tp_rank)
                else 0)
            gap = max(0, needed - remaining)
            worst_gap = max(worst_gap, gap)
            if gap > 0 and reclaimable:
                reclaim = (
                    reclaimable[rank_index]
                    if rank_index < len(reclaimable) else 0)
                beyond_reclaimable = max(
                    beyond_reclaimable, max(0, needed - reclaim))
        if worst_gap <= 0:
            return 0
        pool_divisor = self._pool_divisor(instance_index)
        writeback = _pool_transfer_ns(
            total_bytes=worst_gap,
            divisor=pool_divisor,
            rates=self.rates)
        wait = writeback
        notes.append(f"{prefix}_eviction_writeback_est")
        if beyond_reclaimable > 0 and (
                load.running_task_load_ns or load.active_decode_task_load_ns):
            # 深缺口：等活跃任务完成释放——剩余负载峰值（TP 并行下逐
            # rank 同账）为因果可见的等待下界。
            active_remaining = (
                load.running_task_load_ns + load.active_decode_task_load_ns)
            wait = max(wait, active_remaining)
            notes.append(f"{prefix}_deep_gap_unresolved")
        return wait

    def _per_rank(self, total_bytes: int) -> tuple[int, ...]:
        """总量按 TP 均分的近似视图（调用方有逐 rank 字节时应直接给
        逐 rank 输入；本辅助仅用于粗粒度缺口语境并显式注明）。"""
        size = self.instance_tp_size
        base = total_bytes // size
        remainder = total_bytes - base * size
        return tuple(
            base + (1 if index < remainder else 0)
            for index in range(size))
