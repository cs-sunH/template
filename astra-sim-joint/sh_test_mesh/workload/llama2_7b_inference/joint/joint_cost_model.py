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

状态（§5.6 表）：本模块按"已实现（解析近似 + 在线因子）"交付；正式
预测器的事件递推（共享资源份额随流加入/离开的逐事件重算）为接口位
``LinkFlowRegistry``——调用方可注入真实在途流登记表，缺省为空表
（冷启动，标记 ``contention_coverage=none``）。
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

    def observe_completed(self, session_id: str, decode_tokens: int) -> None:
        if decode_tokens < 0:
            raise JointCostError("completed decode length must be >= 0")
        self._session_totals.setdefault(session_id, []).append(decode_tokens)
        self._run_count += 1
        self._run_sum += decode_tokens

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

    ``recompute`` 复用 J 层换算：factor = 实际服务时长 / 基础模型预测。
    首次观测前为 1.0（配置/roofline 冷启动，标记乐观可能）。更新仅由
    已完成事件驱动；无效样本拒绝（复用 layer_eviction_policy 的拒绝
    记账语义，此处以计数披露）。
    """

    prefill_factor: float = 1.0
    decode_factor: float = 1.0
    transfer_factor: float = 1.0
    updates: dict = field(default_factory=dict)
    rejected: dict = field(default_factory=dict)

    def observe_prefill(self, ratio: float) -> None:
        self._blend("prefill_factor", ratio)

    def observe_decode(self, ratio: float) -> None:
        self._blend("decode_factor", ratio)

    def observe_transfer(self, ratio: float) -> None:
        self._blend("transfer_factor", ratio)

    def _blend(self, name: str, ratio: float) -> None:
        if not math.isfinite(ratio) or ratio <= 0:
            self.rejected[name] = self.rejected.get(name, 0) + 1
            return
        # 固定平滑（α=0.25）：事前定义的确定性规则，非按 workload 调参；
        # 首样本直接初始化。
        current = getattr(self, name)
        if self.updates.get(name, 0) == 0:
            setattr(self, name, float(ratio))
        else:
            alpha = 0.25
            setattr(self, name, (1 - alpha) * current + alpha * ratio)
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
    以 ``+1`` 计）。跨 collective 的互扰若未被遥测覆盖，调用方保留
    ``collective_coverage=False`` 披露——本模型不把未覆盖流量当作零。
    """

    def __init__(self) -> None:
        self._flows: dict[tuple[int, int], int] = {}
        self.collective_coverage = False

    def register(self, source_rank: int, destination_rank: int) -> None:
        key = (source_rank, destination_rank)
        self._flows[key] = self._flows.get(key, 0) + 1

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

    def divisor(self, path_ranks: Sequence[int], *, include_self: bool,
                self_overlap: int = 1) -> int:
        """候选流沿 ``path_ranks`` 路线的仲裁除数。

        有向链路除数 = 已登记流数 + 候选自身占用数（多跳路线自重叠时
        为经过次数）；结果取路线瓶颈链路的最大值。``include_self``
        为 False 时仅报告既有争用（用于只读探测）。
        """
        if len(path_ranks) < 2:
            return 1
        worst = 1
        for index in range(len(path_ranks) - 1):
            key = (path_ranks[index], path_ranks[index + 1])
            flows = self._flows.get(key, 0)
            share = flows + (self_overlap if include_self else 0)
            worst = max(worst, max(1, share))
        return worst

    def snapshot(self) -> dict:
        return {f"{src}->{dst}": count for (src, dst), count
                in sorted(self._flows.items())}


# ============================================================ 决策视图 ==


@dataclass(frozen=True)
class InstanceLoadView:
    """单实例只读负载视图（ns 计的 roofline 服务台账，非队列长度）。"""

    instance_index: int
    queued_task_load_ns: int
    running_task_load_ns: int
    active_decode_task_load_ns: int
    # 目标增长与 home 合并两处空间压力分别标记（§7 容量诊断口径）。
    hbm_remaining_bytes_by_tp_rank: tuple[int, ...]

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

    def __post_init__(self) -> None:
        if self.prefill_ns_per_token <= 0 or self.decode_ns_per_token <= 0:
            raise JointCostError("service rates must be positive")
        if self.model_layers <= 0 or self.instance_tp_size <= 0:
            raise JointCostError("layout parameters must be positive")

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
        不等于删除该 instance，其余动作仍参与比较）。"""
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
        # recompute：任何 instance 恒适用（真实重算必要历史）。
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
        # remote-read：历史驻留在别的 instance（直接远读路径），且
        # remote 能力开关开启（正交能力消融，§7.1）。
        remote_ok = (
            remote_enabled
            and session.location in ("local_hbm", "partial_hbm_remote")
            and session.resident_instance is not None
            and session.resident_instance != instance_index)
        results.append((
            remote_ok,
            None if remote_ok else (
                "remote actions disabled"
                if not remote_enabled
                else "no resident remote history"))
        )
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
        非机械相加：可重叠段取 max，串行依赖段相加）。"""
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

        # ---- 2. 空间准备（执行 instance 的新增 KV 增长，§3.1）。
        input_bytes = sum(request.input_kv_bytes_by_tp_rank)
        decode_estimate_bytes = (
            sum(request.input_kv_bytes_by_tp_rank)
            * request.estimated_decode_tokens
            // max(1, request.input_tokens))
        growth_bytes = input_bytes + decode_estimate_bytes
        eviction_wait = self._eviction_wait_estimate(
            load, growth_bytes, notes)

        # ---- 3. 历史准备（依动作而异）。
        history_prep = 0
        remote_read_ns = 0
        route_path, hops = self._route(instance_index, session)
        if action == ACTION_STAY:
            # LOCAL：无传输；PARTIAL：缺失后缀按池路径恢复（工作区间）。
            missing = sum(session.missing_bytes_by_tp_rank)
            if missing:
                history_prep = _pool_transfer_ns(
                    total_bytes=missing, divisor=1,
                    rates=self.rates)
                notes.append("partial_suffix_pool_restore")
        elif action == ACTION_COPY:
            # LOCAL/PARTIAL 别处：整份驻留历史经 NoC 复制为工作副本；
            # REMOTE：从池恢复必要历史到目标。
            resident_bytes = sum(session.history_bytes_by_tp_rank)
            if session.location == "remote_memory":
                history_prep = _pool_transfer_ns(
                    total_bytes=sum(session.missing_bytes_by_tp_rank)
                    + resident_bytes,
                    divisor=1, rates=self.rates)
                notes.append("pool_restore_to_target")
            else:
                divisor = self.flow_registry.divisor(
                    route_path, include_self=True)
                history_prep = _transfer_ns(
                    total_bytes=resident_bytes
                    + sum(session.missing_bytes_by_tp_rank),
                    path_hops=hops, divisor=divisor,
                    rates=self.rates, per_hop_latency_ns=None,
                    startup_ns=0)
                notes.append("noc_working_copy")
        elif action == ACTION_RECOMPUTE:
            # 无历史搬运；重算必要历史的时间计入计算段（§13.2 驱逐与
            # 复用：复用收益体现为少做的重算或传输）。
            notes.append("recompute_history_in_compute")
        elif action == ACTION_REMOTE:
            # 执行期间按消费远读历史：dense attention 的重复读取按
            # 估计 decode 步数计次（§13.2 remote：多次读取按所选时域
            # 计算，不只按总字节计一次）。
            resident_bytes = sum(session.history_bytes_by_tp_rank) + sum(
                session.missing_bytes_by_tp_rank)
            divisor = self.flow_registry.divisor(
                route_path, include_self=True)
            read_passes = max(
                1, request.estimated_decode_tokens)
            remote_read_ns = _transfer_ns(
                total_bytes=resident_bytes * read_passes,
                path_hops=hops, divisor=divisor,
                rates=self.rates, per_hop_latency_ns=None,
                startup_ns=0)
            notes.append(f"remote_read_passes={read_passes}")
        else:  # pragma: no cover - ACTION_ORDER 封闭
            raise JointCostError(f"unknown action {action!r}")

        # ---- 4. 计算（roofline 基础 × 在线因子；recompute 追加重算
        # 历史的工作量；不含排队——排队在 target_wait/eviction 段）。
        prefill_tokens = request.input_tokens
        if action == ACTION_RECOMPUTE:
            prefill_tokens += request.history_tokens_before
        compute_ns = int(
            prefill_tokens * self.prefill_ns_per_token
            * self.service_factors.prefill_factor
            + request.estimated_decode_tokens
            * self.decode_ns_per_token
            * self.service_factors.decode_factor)

        # ---- 5. compute_done 后的合并（§2.2/§3.2：增量回 origin_home；
        # 执行位置等于 home 的本地提交不生成虚构流量；新会话（home 待
        # 建立 = 执行实例）同 stay 本地提交）。
        merge_ns = 0
        if (
            session.home_instance is not None
            and session.home_instance != instance_index
        ):
            increments = growth_bytes
            merge_route, merge_hops = self._route_pair(
                instance_index, session.home_instance)
            merge_divisor = self.flow_registry.divisor(
                merge_route, include_self=True)
            merge_ns = _transfer_ns(
                total_bytes=increments, path_hops=merge_hops,
                divisor=merge_divisor, rates=self.rates,
                per_hop_latency_ns=None, startup_ns=0)
            # home 侧空间准备：按 home 当前占用估计释放等待（§3.2 不
            # 认为"写回免费"）。
            home_load = self.loads.get(session.home_instance)
            if home_load is not None:
                merge_ns += self._eviction_wait_estimate(
                    home_load, increments, notes, prefix="home_merge")
            notes.append(f"merge_to_home={session.home_instance}")

        # ---- 关键路径合成：目标等待 -> [历史准备 ∥ 空间准备（争用同
        # 链路，保守串行）] -> 计算（remote 与计算重叠推进、取增量） ->
        # 合并。可重叠段取 max：remote 读流与计算相互重叠（§5.2 恢复
        # 争用使计算变慢的口径：以 remote_read 延长计算的保守合成）。
        effective_compute = compute_ns + remote_read_ns
        prep = max(history_prep, eviction_wait)
        cost = target_wait + prep + effective_compute + merge_ns

        divisor = self.flow_registry.divisor(
            route_path, include_self=True)
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

    def _eviction_wait_estimate(
        self,
        load: InstanceLoadView,
        needed_bytes: int,
        notes: list,
        *,
        prefix: str = "execution_growth",
    ) -> int:
        """空间缺口 -> 写回时间的换算（不把无单位容量风险加到延迟上）。

        逐 rank 检查（§3.3.1）：缺口 = max_r(0, needed_r − remaining_r)。
        无缺口返回 0；有缺口按池写回路径估计（除数含在途流）。深缺口
        （无可释放对象）不在本模型内解决——返回该估计并注明
        ``deep_gap_unresolved``，由共同生命周期进入等待协议。
        """
        worst_gap = 0
        for needed, remaining in zip(
                self._per_rank(needed_bytes), load.hbm_remaining_bytes_by_tp_rank):
            worst_gap = max(worst_gap, max(0, needed - remaining))
        if worst_gap <= 0:
            return 0
        writeback = _pool_transfer_ns(
            total_bytes=worst_gap,
            divisor=1,  # 在途写回流未单列时的冷启动份额
            rates=self.rates)
        notes.append(f"{prefix}_eviction_writeback_est")
        return writeback

    def _per_rank(self, total_bytes: int) -> tuple[int, ...]:
        """总量按 TP 均分的近似视图（调用方有逐 rank 字节时应直接给
        逐 rank 输入；本辅助仅用于粗粒度缺口语境并显式注明）。"""
        size = self.instance_tp_size
        base = total_bytes // size
        remainder = total_bytes - base * size
        return tuple(
            base + (1 if index < remainder else 0)
            for index in range(size))
