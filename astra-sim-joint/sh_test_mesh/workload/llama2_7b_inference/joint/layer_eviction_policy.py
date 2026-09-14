"""layer_eviction_policy.py -- E：消费期限驱动的动态热前缀保留。

设计依据：《三机制联合策略_template仓库设计方案》§5。E 决定"每次释放
几层、保留哪个连续热前缀"；victim 的类别/对象优先级由 T
（eviction_priority）决定，两者正交（§4 末条：T 只决定类别/对象优先
级，不决定每次释放几层；E 不得通过收益打分越过 T 的严格优先级）。

三种模式（§5.6 表）：

* ``adaptive``（E-on）：按层消费期限公式计算最小热前缀 ``k_hide``
  作为**软保留目标**；释放时同一优先级组内先扫各对象的目标外后缀
  ``k_hide+1..h``，需求未满足再扫目标内仍合法的后缀；只释放满足逐
  rank 缺口所需的完整层组，满足即停。目标可被实际缺口突破；无空间
  需求不主动驱逐（§5.5）。
* ``minimal_layer_groups``（E-off）：不计算隐藏目标，同一合法对象排
  序下从高层号向低层号释放最少完整层组直到逐 rank 缺口满足。
* ``legacy_half``：原半层逻辑的隔离适配（保留
  ``L - L//2`` 前缀，先逐 victim 释放后半层、再整份释放），仅作回
  归/辅助对照，不是 E-off 参照。

期限公式（§5.2，串行恢复简化模型，全部 ns；正式预测器的事件递推另
行接入，本模块的公式枚举 k=0..L、不假设单调）：

    R̂_ℓ(k) = q̂ + Σ_{j=k+1..ℓ} r̂_j          （冷层 ℓ 恢复完成时刻）
    D̂_ℓ    = Σ_{j=1..ℓ-1} ĉ_j              （无冷 KV 等待的消费期限）
    k̂_hide = min{k ∈ 0..L : R̂_ℓ(k) ≤ D̂_ℓ, ∀ℓ>k}
    Ŝ(k)   = max(0, max_{ℓ>k}[R̂_ℓ(k) − D̂_ℓ]),  Ŝ(L)=0

必须检查所有冷层（不能只查第一冷层或比较总量）；无缺失层时条件为
空（k=L 是公式边界，不保证物理内存容得下）；无预取且首层需要历史
时 k=0 通常不能完全隐藏恢复。

在线估计（§5.4/§5.5，全部因果：只用已完成轮次的在线统计，禁止预读
CSV/未来输出长度；改变未来输入行、未来真实输出长度及返回时间，保
持当前可见状态相同，E 输出与估计器状态必须不变）：

* 输入长度估计：session 已完成请求的新增输入长度在线均值；无本
  session 样本用本 run 已完成请求均值；连 run 样本也没有时保留目标
  为 L 并标记 ``cold_start_unknown_input``。
* 服务效率因子：``θ_n = (1−α_n)θ_{n−1} + α_n x_n``，
  ``α_n = 1−exp(−Δt_n/τ_n)``，``τ_n`` 取本组上一有效样本的正服务时
  长；首样本直接初始化；零时长/零分母/不完整计量样本不更新并记录
  原因。不做逐 workload 拟合的 clamp。

本模块为纯策略层：``plan_release`` 只读（无资源副作用、不修改 KV 账
本/事件队列），仅输出计划；提交前由调用方复核版本与保护（§5.6）。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Mapping, Optional, Sequence

LAYER_POLICY_ADAPTIVE = "adaptive"
LAYER_POLICY_LEGACY_HALF = "legacy_half"
LAYER_POLICY_MINIMAL = "minimal_layer_groups"

LAYER_POLICIES = (
    LAYER_POLICY_ADAPTIVE, LAYER_POLICY_LEGACY_HALF, LAYER_POLICY_MINIMAL,
)


class LayerEvictionError(ValueError):
    """fail-closed：非法模式、非法层参数或非法字节函数。"""


# ============================================================ §5.2 公式 ==


@dataclass(frozen=True)
class KHideResult:
    """期限公式的解析结果（全部 ns 整数；k=L 为公式边界）。"""

    k_hide: int
    model_layers: int
    exposed_stall_ns: int          # Ŝ(k_hide)
    binding_layer: Optional[int]   # 约束最紧冷层（全余量 <0 或 k=L 时 None）
    first_block_wait_ns: int


def serial_restore_completion_ns(
    k: int,
    ell: int,
    restore_ns_by_layer: Sequence[int],
    first_block_wait_ns: int,
) -> int:
    """R̂_ℓ(k) = q̂ + Σ_{j=k+1..ℓ} r̂_j（逐层串行恢复；ell 为 1-based 层号）。"""
    layers = len(restore_ns_by_layer)
    if not 0 <= k <= layers:
        raise LayerEvictionError(f"k={k} outside 0..{layers}")
    if not 1 <= ell <= layers:
        raise LayerEvictionError(f"ell={ell} outside 1..{layers}")
    if ell <= k:
        raise LayerEvictionError(
            f"cold-layer check requires ell>k (ell={ell}, k={k})")
    total = first_block_wait_ns
    for j in range(k + 1, ell + 1):
        total += int(restore_ns_by_layer[j - 1])
    return total


def layer_consumption_deadline_ns(
    ell: int,
    compute_ns_by_layer: Sequence[int],
) -> int:
    """D̂_ell = Σ_{j=1..ell-1} ĉ_j（默认以该层开始为保守消费期限，§5.2）。"""
    layers = len(compute_ns_by_layer)
    if not 1 <= ell <= layers:
        raise LayerEvictionError(f"ell={ell} outside 1..{layers}")
    return sum(int(compute_ns_by_layer[j - 1]) for j in range(1, ell))


def exposed_stall_ns(
    k: int,
    compute_ns_by_layer: Sequence[int],
    restore_ns_by_layer: Sequence[int],
    first_block_wait_ns: int,
) -> int:
    """Ŝ(k) = max(0, max_{ℓ>k}[R̂_ℓ(k) − D̂_ℓ])；Ŝ(L)=0。"""
    layers = len(compute_ns_by_layer)
    if k == layers:
        return 0
    worst = 0
    for ell in range(k + 1, layers + 1):
        restore = serial_restore_completion_ns(
            k, ell, restore_ns_by_layer, first_block_wait_ns)
        deadline = layer_consumption_deadline_ns(ell, compute_ns_by_layer)
        worst = max(worst, restore - deadline)
    return worst


def k_hide_deadline(
    compute_ns_by_layer: Sequence[int],
    restore_ns_by_layer: Sequence[int],
    first_block_wait_ns: int = 0,
) -> KHideResult:
    """枚举 k=0..L 求 ``k̂_hide``（不假设共享资源下单调，不二分，§5.2）。

    检查所有冷层 ℓ>k；无缺失层（k=L）条件为空。浮点近平局只按整数
    ns 精确比较（本模型全部整数量），不附加安全百分比。
    """
    layers = len(compute_ns_by_layer)
    if len(restore_ns_by_layer) != layers:
        raise LayerEvictionError(
            "compute/restore per-layer vectors must have equal length")
    if any(value < 0 for value in compute_ns_by_layer) or any(
            value < 0 for value in restore_ns_by_layer):
        raise LayerEvictionError("per-layer times must be non-negative")
    if first_block_wait_ns < 0:
        raise LayerEvictionError("first_block_wait_ns must be non-negative")

    deadlines = [
        layer_consumption_deadline_ns(ell, compute_ns_by_layer)
        for ell in range(1, layers + 1)
    ]

    def margins(k: int) -> tuple[int, int]:
        """返回 (max 超出量, 绑定层)；k=L 时 (0, None)。"""
        if k == layers:
            return 0, None
        worst = 0
        binding = None
        restore = first_block_wait_ns
        # 逐层累加恢复时间，避免每层重算前缀和。
        for ell in range(k + 1, layers + 1):
            restore += int(restore_ns_by_layer[ell - 1])
            over = restore - deadlines[ell - 1]
            if over > worst:
                worst = over
                binding = ell
        return worst, binding

    for k in range(0, layers + 1):
        worst, binding = margins(k)
        if worst <= 0:
            # 全部冷层满足 R<=D；worst==0 也可能出现在无冷层（k=L）。
            return KHideResult(
                k_hide=k,
                model_layers=layers,
                exposed_stall_ns=max(0, worst),
                binding_layer=binding,
                first_block_wait_ns=first_block_wait_ns,
            )
    # 数学上不可达：k=L 时条件恒空。
    raise LayerEvictionError("k_hide enumeration failed to terminate")


# ==================================================== §5.4/§5.5 在线估计 ==


class OnlineMean:
    """在线均值：``mean += (observed − mean) / count``（§5.5，禁预读 CSV）。"""

    def __init__(self) -> None:
        self.count = 0
        self.mean: Optional[float] = None

    def update(self, observed: float) -> float:
        if observed < 0:
            raise LayerEvictionError("online mean rejects negative samples")
        self.count += 1
        if self.mean is None:
            self.mean = float(observed)
        else:
            self.mean += (float(observed) - self.mean) / self.count
        return self.mean

    @property
    def available(self) -> bool:
        return self.count > 0


class InputLengthEstimator:
    """等待 session 下一轮输入长度的因果估计（§5.5 调用 2）。

    优先级：本 session 已完成请求的新增输入长度在线均值 → 本 run 已
    完成请求均值 → 无样本（返回 None，调用方保留目标 = L 并标
    ``cold_start_unknown_input``）。不预测真实返回时刻；返回后按真实
    请求重算。
    """

    def __init__(self) -> None:
        self._session_means: dict[str, OnlineMean] = {}
        self._run_mean = OnlineMean()

    def observe_completed_turn(
        self, session_id: str, input_tokens: int,
    ) -> None:
        """仅已完成轮次可入样（真实到达并完成后的新增输入长度）。"""
        if input_tokens < 0:
            raise LayerEvictionError("input length sample must be >= 0")
        if session_id not in self._session_means:
            self._session_means[session_id] = OnlineMean()
        self._session_means[session_id].update(input_tokens)
        self._run_mean.update(input_tokens)

    def estimate_next_input(self, session_id: str) -> Optional[float]:
        session_mean = self._session_means.get(session_id)
        if session_mean is not None and session_mean.available:
            return session_mean.mean
        if self._run_mean.available:
            return self._run_mean.mean
        return None

    def session_sample_count(self, session_id: str) -> int:
        mean = self._session_means.get(session_id)
        return mean.count if mean is not None else 0


class EwmaServiceFactor:
    """确定性因果效率更新（§5.4）：θ、γ 共用的更新规则。

    ``θ_n = (1−α_n)θ_{n−1} + α_n x_n``，``α_n = 1−exp(−Δt_n/τ_n)``；
    ``τ_n`` 默认取本组上一有效样本的正服务时长；首个有效样本直接初始
    化；零时长/零分母/不完整计量样本不更新并记录原因（不把受混合等待
    污染的墙钟当纯服务样本）。不沿用无依据的数值 clamp；物理合法范围
    （因子 > 0）由调用方在物理量检查中保持。
    """

    def __init__(self, initial: float = 1.0) -> None:
        self._value = float(initial)
        self._last_completion_ns: Optional[int] = None
        self._last_service_duration_ns: Optional[int] = None
        self.samples = 0
        self.rejected_updates: list[str] = []

    @property
    def value(self) -> float:
        return self._value

    def update(
        self,
        *,
        observed_ratio: float,
        completion_ns: int,
        service_duration_ns: Optional[int],
    ) -> float:
        """一个独立估计组的一次观测。

        ``observed_ratio`` = 实际量 / 同区间基础模型预测量（预测分母不
        含本修正因子，避免循环校正）；``service_duration_ns`` 为本组上
        一有效样本的正服务时长（τ）；``completion_ns`` 为相邻观测完成
        时刻。无效样本（ratio<=0、非有限值、缺 τ、Δt<0）拒绝并记录。
        """
        if not math.isfinite(observed_ratio) or observed_ratio <= 0:
            self.rejected_updates.append("invalid_ratio")
            return self._value
        if service_duration_ns is not None and service_duration_ns <= 0:
            self.rejected_updates.append("nonpositive_service_duration")
            return self._value
        if self._last_completion_ns is None:
            # 首个有效样本直接初始化（§5.4）。
            self._value = float(observed_ratio)
            self._last_completion_ns = int(completion_ns)
            self._last_service_duration_ns = (
                int(service_duration_ns)
                if service_duration_ns is not None else None)
            self.samples = 1
            return self._value
        delta_ns = int(completion_ns) - self._last_completion_ns
        if delta_ns < 0:
            self.rejected_updates.append("negative_delta_t")
            return self._value
        tau_ns = self._last_service_duration_ns
        if tau_ns is None or tau_ns <= 0:
            self.rejected_updates.append("missing_tau")
            return self._value
        alpha = 1.0 - math.exp(-delta_ns / tau_ns)
        self._value = (1.0 - alpha) * self._value + alpha * observed_ratio
        if not math.isfinite(self._value) or self._value <= 0:
            raise LayerEvictionError(
                "EWMA update produced a non-physical factor")
        self._last_completion_ns = int(completion_ns)
        if service_duration_ns is not None:
            self._last_service_duration_ns = int(service_duration_ns)
        self.samples += 1
        return self._value


# ======================================================= §5.6 释放计划 ==


@dataclass(frozen=True)
class VictimView:
    """合法 victim 的只读视图（T 已定序；E 只看层数与字节）。

    ``layer_group_bytes_fn(layer_start, layer_end) -> per-rank bytes``
    由调用方（KVCacheManager）注入，保证与账本同一字节口径；受保护
    的 victim 不应出现在列表中（保护过滤在共同生命周期完成）。
    """

    session_id: str
    resident_prefix_layers: int      # 当前热前缀层数 h（0..L）
    layer_group_bytes_fn: Callable[[int, int], tuple[int, ...]]
    retention_target_layers: int     # 本策略给出的软保留目标（0..L）
    next_request_type: Optional[str] = None
    last_completion_ns: Optional[int] = None


@dataclass(frozen=True)
class LayerReleaseStep:
    """对单一 victim 的一次完整层组释放（区间 [layer_start, before_h)。"""

    session_id: str
    layer_start: int
    before_layers: int               # 释放前的驻留层数
    bytes_by_tp_rank: tuple[int, ...]

    @property
    def layer_end(self) -> int:
        return self.before_layers


@dataclass
class EvictionPlan:
    """plan_release 的只读输出（无资源副作用；仅选中后由调用方提交）。"""

    mode: str
    steps: tuple[LayerReleaseStep, ...]
    satisfied: bool
    # 逐 rank 记账：缺口 vs 实际释放字节（§3.3.1 不用汇总掩盖单 rank）。
    gap_bytes_by_tp_rank: tuple[int, ...]
    released_bytes_by_tp_rank: tuple[int, ...]
    # 目标突破披露：释放进入软目标内时记录（§5.6 输出要求）。
    target_breach_sessions: tuple[str, ...] = ()
    diagnostics: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "mode": self.mode,
            "satisfied": self.satisfied,
            "steps": [
                {
                    "session_id": step.session_id,
                    "layer_start": step.layer_start,
                    "layer_end": step.layer_end,
                    "bytes_by_tp_rank": list(step.bytes_by_tp_rank),
                }
                for step in self.steps
            ],
            "gap_bytes_by_tp_rank": list(self.gap_bytes_by_tp_rank),
            "released_bytes_by_tp_rank": list(self.released_bytes_by_tp_rank),
            "target_breach_sessions": list(self.target_breach_sessions),
            "diagnostics": dict(self.diagnostics),
        }


def _check_gap_met(
    gap: Sequence[int], released: Sequence[int],
) -> bool:
    return all(r >= g for r, g in zip(released, gap))


def coalesce_steps(
    steps: Sequence[LayerReleaseStep],
) -> tuple[LayerReleaseStep, ...]:
    """把同一 victim 的连续单层组步合并为整段释放区间（逐字节恒等）。

    释放循环逐完整层组推进（每步后重查逐 rank 缺口）；物化层把同一
    victim 的连续区间合成一笔 KVTransfer。步序即 victim 序 × 高层号
    递减序，同 victim 的连续步区间天然连续。
    """
    merged: list[LayerReleaseStep] = []
    for step in steps:
        if (
            merged
            and merged[-1].session_id == step.session_id
            and merged[-1].layer_start == step.before_layers
        ):
            previous = merged[-1]
            merged[-1] = LayerReleaseStep(
                session_id=previous.session_id,
                layer_start=step.layer_start,
                before_layers=previous.before_layers,
                bytes_by_tp_rank=tuple(
                    a + b for a, b in zip(
                        previous.bytes_by_tp_rank, step.bytes_by_tp_rank)),
            )
        else:
            merged.append(step)
    return tuple(merged)


def _release_prefix_groups(
    *,
    victim: VictimView,
    low_bound: int,
    gap: Sequence[int],
    released: list[int],
    steps: list[LayerReleaseStep],
    breach_sessions: list[str],
    soft_target: int,
    current_prefix: dict[str, int],
) -> None:
    """从 victim 的**当前**驻留顶端向下释放完整层组，直到缺口满足。

    ``low_bound`` 为本轮扫描允许释放到的下界（不含）：adaptive 第一轮
    = max(0, soft_target)，第二轮 = 0；minimal 第一轮 = 0。跨轮调用时
    通过 ``current_prefix`` 跟踪该 victim 的实际剩余层数（victim 视图
    是进入 plan_release 时的只读快照，不得重复释放已释放区间）。每步
    恰好一个完整层组（保持热前缀连续，不制造非连续前缀），满足即停。
    """
    layer = current_prefix[victim.session_id]
    while layer > low_bound and not _check_gap_met(gap, released):
        group_bytes = victim.layer_group_bytes_fn(layer - 1, layer)
        if len(group_bytes) != len(gap):
            raise LayerEvictionError(
                "layer group bytes vector length mismatch with gap")
        steps.append(LayerReleaseStep(
            session_id=victim.session_id,
            layer_start=layer - 1,
            before_layers=layer,
            bytes_by_tp_rank=tuple(int(value) for value in group_bytes),
        ))
        for rank_index, value in enumerate(group_bytes):
            released[rank_index] += int(value)
        layer -= 1
        current_prefix[victim.session_id] = layer
        if layer < soft_target:
            # 释放已进入软目标内（目标被实际缺口突破），披露（§5.6）。
            if victim.session_id not in breach_sessions:
                breach_sessions.append(victim.session_id)


class LayerEvictionPolicy:
    """E 的策略入口：软保留目标 + 释放计划生成（只读）。

    构造参数 ``model_layers`` / ``tp_degree`` 来自模型与硬件配置（静态
    派生，§5.3 表）；运行态（缺口、victim、字节函数、目标）经
    :meth:`plan_release` 的 context 注入。
    """

    def __init__(self, mode: str, model_layers: int) -> None:
        if mode not in LAYER_POLICIES:
            raise LayerEvictionError(f"unknown layer policy: {mode!r}")
        if model_layers <= 0:
            raise LayerEvictionError("model_layers must be positive")
        self.mode = mode
        self.model_layers = model_layers
        # legacy_half 的固定保留前缀 = L - L//2（SH 原语义：奇数层保留
        # 较大的一半），逐出从该常量下界开始（§5.6 回归适配）。
        self.legacy_resident_prefix = model_layers - model_layers // 2

    # ------------------------------------------------------------ 目标 --
    def retention_target(
        self,
        victims: Sequence[VictimView],
    ) -> dict[str, int]:
        """各 victim 的软保留目标层数（只读计算，不搬数据）。

        adaptive：调用方通过 ``VictimView.retention_target_layers`` 注入
        k_hide 目标（等待 session 的保留目标预测，非未来严格最小值，
        §5.5）；legacy_half：固定 ``L − L//2``；minimal：0（不计算隐藏
        目标——所有驻留层都在可释放后缀内）。当前已经少于目标层数时不
        主动补回；目标下降也不在没有空间需求时主动驱逐（§5.5，由调用
        方保证：plan_release 只在释放规划点被调用）。
        """
        targets: dict[str, int] = {}
        for victim in victims:
            current = victim.resident_prefix_layers
            if self.mode == LAYER_POLICY_LEGACY_HALF:
                target = min(current, self.legacy_resident_prefix)
            elif self.mode == LAYER_POLICY_MINIMAL:
                target = 0
            else:
                target = max(0, min(current, victim.retention_target_layers))
            targets[victim.session_id] = target
        return targets

    # ------------------------------------------------------------ 计划 --
    def plan_release(
        self,
        *,
        gap_bytes_by_tp_rank: Sequence[int],
        victims: Sequence[VictimView],
    ) -> EvictionPlan:
        """按模式生成释放计划（只读；不修改任何账本/事件队列）。

        victim 顺序由调用方（T）给定；本函数在同一优先级组内工作。
        逐 rank 缺口满足即停；类间推进（human→tool）由调用方按
        eviction_class_order 驱动。legacy_half 保持原两段式：第一段
        每 victim 只释放固定半层后缀（逐 victim 一步、重查缺口），不足
        再第二段从高层号整组释放到 0——与 SH 历史 Release 顺序一致。
        """
        if not gap_bytes_by_tp_rank or any(
                value < 0 for value in gap_bytes_by_tp_rank):
            raise LayerEvictionError("gap vector must be non-empty and >=0")
        gap = [int(value) for value in gap_bytes_by_tp_rank]
        released = [0] * len(gap)
        steps: list[LayerReleaseStep] = []
        breach_sessions: list[str] = []
        diagnostics: dict = {"scan_rounds": []}
        # 跨轮实际剩余层数（victim 视图为入口快照；两轮扫描不得重复
        # 释放第一轮已释放的区间）。
        current_prefix = {
            victim.session_id: victim.resident_prefix_layers
            for victim in victims
        }

        if _check_gap_met(gap, released):
            return EvictionPlan(
                mode=self.mode,
                steps=(),
                satisfied=True,
                gap_bytes_by_tp_rank=tuple(gap),
                released_bytes_by_tp_rank=(),
                diagnostics=diagnostics,
            )

        if self.mode == LAYER_POLICY_LEGACY_HALF:
            # 第一段：每个**全本地**（prefix == L）victim 恰好一步——释放
            # 固定半层后缀（SH 原序：suffix 逐出只作用于 full-local；
            # partial 已在后缀内，直接进第二段），victim 间按序推进，每
            # 步后重查缺口（不能整份清空）。legacy 不做步合并——两段式
            # 的逐笔传输序列即回归语义（与 SH 原实现逐字节等价）。
            for victim in victims:
                if _check_gap_met(gap, released):
                    break
                prefix = current_prefix[victim.session_id]
                if prefix != self.model_layers:
                    continue
                low = self.legacy_resident_prefix
                if prefix <= low:
                    continue
                group_bytes = victim.layer_group_bytes_fn(low, prefix)
                steps.append(LayerReleaseStep(
                    session_id=victim.session_id,
                    layer_start=low,
                    before_layers=prefix,
                    bytes_by_tp_rank=tuple(
                        int(value) for value in group_bytes),
                ))
                for rank_index, value in enumerate(group_bytes):
                    released[rank_index] += int(value)
                current_prefix[victim.session_id] = low
            diagnostics["scan_rounds"].append("legacy_half_suffix")
            # 第二段：仍不足时**整份外迁**（SH 原序：每 victim 一步释放
            # 全部剩余驻留前缀 [0, current)，组粒度逐组停止是 minimal/
            # adaptive 的语义，legacy 保持全有或全无）。
            if not _check_gap_met(gap, released):
                for victim in victims:
                    if _check_gap_met(gap, released):
                        break
                    prefix = current_prefix[victim.session_id]
                    if prefix <= 0:
                        continue
                    group_bytes = victim.layer_group_bytes_fn(0, prefix)
                    steps.append(LayerReleaseStep(
                        session_id=victim.session_id,
                        layer_start=0,
                        before_layers=prefix,
                        bytes_by_tp_rank=tuple(
                            int(value) for value in group_bytes),
                    ))
                    for rank_index, value in enumerate(group_bytes):
                        released[rank_index] += int(value)
                    current_prefix[victim.session_id] = 0
                diagnostics["scan_rounds"].append("legacy_full_fallback")

        elif self.mode == LAYER_POLICY_MINIMAL:
            # E-off：无隐藏目标，victim 序从高到低释放最少完整层组。
            for victim in victims:
                if _check_gap_met(gap, released):
                    break
                _release_prefix_groups(
                    victim=victim,
                    low_bound=0,
                    gap=gap,
                    released=released,
                    steps=steps,
                    breach_sessions=breach_sessions,
                    soft_target=0,
                    current_prefix=current_prefix,
                )
            diagnostics["scan_rounds"].append("minimal_groups")

        else:  # adaptive：两次扫描（§5.6）
            targets = self.retention_target(victims)
            # 第一轮：各对象的目标外后缀 k_hide+1..h（高→低，完整层组）。
            for victim in victims:
                if _check_gap_met(gap, released):
                    break
                soft_target = targets[victim.session_id]
                _release_prefix_groups(
                    victim=victim,
                    low_bound=soft_target,
                    gap=gap,
                    released=released,
                    steps=steps,
                    breach_sessions=breach_sessions,
                    soft_target=soft_target,
                    current_prefix=current_prefix,
                )
            diagnostics["scan_rounds"].append("target_exterior_suffix")
            # 第二轮：需求未满足再扫目标内仍合法的后缀（目标可被突破）。
            if not _check_gap_met(gap, released):
                for victim in victims:
                    if _check_gap_met(gap, released):
                        break
                    soft_target = targets[victim.session_id]
                    _release_prefix_groups(
                        victim=victim,
                        low_bound=0,
                        gap=gap,
                        released=released,
                        steps=steps,
                        breach_sessions=breach_sessions,
                        soft_target=soft_target,
                        current_prefix=current_prefix,
                    )
                diagnostics["scan_rounds"].append("target_interior_suffix")

        return EvictionPlan(
            mode=self.mode,
            steps=(
                coalesce_steps(steps)
                if self.mode != LAYER_POLICY_LEGACY_HALF
                else tuple(steps)),
            satisfied=_check_gap_met(gap, released),
            gap_bytes_by_tp_rank=tuple(gap),
            released_bytes_by_tp_rank=tuple(released),
            target_breach_sessions=tuple(breach_sessions),
            diagnostics=diagnostics,
        )
