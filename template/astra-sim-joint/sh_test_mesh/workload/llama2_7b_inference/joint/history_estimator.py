"""history_estimator.py -- W6：因果历史统计与未知工作量接口（先 shadow）。

《joint仓库当下+历史修改方案》§8（因果历史模块）+ W6 段与主文
《当下+历史分析建模方法》§3.2/§5.1-§5.3/§8-9 的落地。W6 交付四件事：

1. ``ConditionalRemainingLengthEstimator``（§8.2）——已有请求的**条件剩余
   长度**正式量 ``E[N_total - d | N_total > d, F_t]``（主文 §3.2），不是
   "历史均值 − d 后截零"；有界直方图 + 生存尾计数 + 活跃请求删失显式记录；
   回退链 session → 已知类型类别 → run → 冷启动正尾先验。
2. ``ArrivalStatistics``（§8.3/§8.4）——互斥到达类别（新 session 首请求 /
   已存在 session 后续返回）+ 固定模拟时间桶的快慢双尺度强度 + 空桶批量
   衰减 + 回归预测公式。
3. ``ServiceSampleRecorder``（§8.1）——携带分解字段（busy/queue/data/
   capacity wait、仲裁参与、硬件版本）的服务样本账；仅纯服务段进效率
   EWMA（与现有 ``ServiceFactors`` 同公式同纪律），不可分离样本保硬件
   模型 + 残差 + 原因，不吸收成"有效带宽"。
4. ``ForecastMaturityLedger`` / ``EpochLog`` / ``FallbackCode``（§8.5）——
   预测成熟（issued_at/matures_at，恰一次）、过期、回退原因码（可复现
   字段）与分布变化 epoch 标记骨架，供 W7 消费。

**单位契约（与现有 CausalHorizonEstimator 同源）**：长度量全部是 decode
**前向步**——完成观测直收 ``runtime.decode_length``（不含 debut 首 token
步，首 token 归 P 末前向；见 joint_cost_model.CausalHorizonEstimator
docstring 与 joint/test_d14_forward_step_semantics.py 的消费者一致性对账
表）。本模块不做任何 ±1 换算。

**参数纪律（W0 冻结）**：全部数值参数从 ``joint.prediction_config.
PredictionParams`` 注入（构造缺省 = ``PredictionParams()``），本模块**零
平行常数**：桶宽/半衰期/τ/冷启动先验/直方图界/EWMA 权重/成熟门槛/有效
期/隔离窗均单一来源于 PredictionParams；α 只读其派生属性
（``arrival_alpha_fast/slow`` = 1 − 2^(−Δ/L)），绝不另行配置或重算出不
一致 alpha。服务效率 EWMA 权重（1/16，逐完成服务样本）**不得**挪用为新
增请求强度的时间衰减——到达衰减由 Lf/Ls 经 α 派生，两处来源不同
（prediction_config 数值来源登记；本模块按类隔离：ServiceSampleRecorder
不读 arrival_* 字段，ArrivalStatistics 不读 service_* 字段）。

**输入白名单（防"顺引用读未来"，§8.2/§8.3）**：本模块只接受真实结束
（``observe_completed``）与当前已暴露进度（``observe_censored`` 的
exposed_progress_steps——调用方应取 ``runtime.decode_tokens_consumed``）。
构造与观测接口**不设** ``decode_length`` / ``final_context_tokens`` /
``prefill_length`` / ``inter_request_interval`` 等隐藏真值形参（结构上
杜绝；传入即 TypeError）。``next_request`` 对象引用、future_alarms
envelope、真实 interval 均在调度器侧，本模块不 import 调度器、不存在顺
引用路径。

接线位于 ``online/sh30_online_scheduler.py``：条件剩余长度和统一纯服务
效率用于正式计价；新增需求预测的 shadow/validated 隔离由
``joint/background_forecast.py`` 与 ``joint/forecast_validation.py`` 实施。
``CausalHorizonEstimator`` 保留为诊断与先验来源；``ServiceFactors``
消费统一效率估计器的发布值。
"""
from __future__ import annotations

import enum
import heapq
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Tuple

from joint.prediction_config import (
    COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1,
    PredictionParams,
)

__all__ = [
    "COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1",
    "ARRIVAL_CLASS_NEW_SESSION_FIRST",
    "ARRIVAL_CLASS_SESSION_RETURN",
    "ARRIVAL_CLASSES",
    "NOT_NEW_ARRIVAL_EXCLUSIONS",
    "OPEN_BUCKET_RULE_FULL_BUCKETS_ONLY",
    "EPOCH_RESET_RULE_NO_RESET_QUARANTINE",
    "ConditionalRemainingEstimate",
    "ConditionalRemainingLengthEstimator",
    "ArrivalCountWindow",
    "ArrivalStatistics",
    "ServiceWorkSample",
    "ServiceSampleRecorder",
    "FallbackCode",
    "FallbackRecord",
    "ForecastEntry",
    "ForecastMaturityLedger",
    "EpochLog",
]

#: W6 实现的冷启动尾部分布 ID（PredictionParams.cold_start_tail_distribution
#: 合法值；其他 ID 一律 fail-closed，不用静默别名）。语义 = **条件剩余步数**
#: 分布（非总长）：均值 = PredictionParams.cold_start_forward_steps 的几何
#: 尾（正支撑 ⇒ P(R>0)=1，正尾保证；先验点估计与 d 无关——登记的工程
#: 候选，G-H 前不得按测试胜负调整）。

#: 互斥到达类别（§8.3 最低划分；可再按已揭示类型分组）。
ARRIVAL_CLASS_NEW_SESSION_FIRST = "new_session_first"
ARRIVAL_CLASS_SESSION_RETURN = "session_return"
ARRIVAL_CLASSES = (ARRIVAL_CLASS_NEW_SESSION_FIRST, ARRIVAL_CLASS_SESSION_RETURN)

#: 不是新增到达的事件族（§8.3 互斥边界，冻结元组披露）：当前队列、当前请
#: 求剩余 D、已提交通信。``ArrivalStatistics.note_excluded_event`` 只落披
#: 露计数，绝不更新任何 λ（W7 生成背景时的 forecast 排除集合素材）。
NOT_NEW_ARRIVAL_EXCLUSIONS = (
    "current_queue",
    "current_request_remaining_decode",
    "submitted_communications",
)

#: 未闭合当前桶规则（§8.4"两规则选一并登记"）：W6 冻结 = 仅完整桶——开放
#: 桶的部分计数不进 λ，直至该桶闭合；开放桶计数进披露（曝光时长校正为
#: 未实现的登记变体，不与本规则静默并存）。
OPEN_BUCKET_RULE_FULL_BUCKETS_ONLY = "full_buckets_only"

#: epoch/分布变化的"是否重置"规则位（§8.5"冻结是否重置或降级的规则"）：
#: W6 冻结 = **不重置**统计，改进隔离窗（history_drift_quarantine_ns 内
#: 不启用 validated，回退 distribution_shift）。防止"结果变差时临时换窗
#: 口/偷偷重置"。
EPOCH_RESET_RULE_NO_RESET_QUARANTINE = "no_reset_quarantine"


# ============================================================ §8.2 条件剩余 ==


@dataclass(frozen=True)
class ConditionalRemainingEstimate:
    """``estimate_conditional`` 的返回（值 + 来源 + 删失披露）。

    ``remaining_forward_steps`` 单位 = decode 前向步（与现有估计器同单位），
    下限 1：条件于 ``N_total > d``（请求未结束）⇒ 至少还要一次前向——同
    源于现有 ``estimate_boundary`` 的式(4a) ``max(1, n̂)`` 语义。
    """

    #: 条件剩余步数估计 E[N_total − d | N_total > d]（下限 1）。
    remaining_forward_steps: int
    #: 来源："session_conditional" / "category_conditional" / "run_conditional"
    #: / "cold_start_prior"。
    source: str
    #: 命中级已完成样本数（先验时为到达的最高级样本数）。
    level_samples: int
    #: 命中级删失（活跃未完成）计数——删失程度披露（§8.2"披露删失程度"）。
    censored_count: int
    #: 先验回退原因（命中级为 None）：
    #: no_samples / insufficient_mature_samples / d_beyond_observed_max /
    #: unresolvable_tail_bucket。
    fallback_reason: Optional[str]
    #: 复现披露（直方图界、尾桶、先验分布 ID、各级样本量等原始类型字段）。
    disclosure: Mapping[str, Any]


class _LevelStats:
    """单级（session/类别/run）的有界统计存储。

    * ``(count, sum)``：常数空间在线均值（与 CausalHorizonEstimator 的
      D14 sum/count 语义同源——保留作诊断与先验来源）。
    * ``max_observed``：已完成样本最大值（精确标量）——"d 超过完成样本
      最大值 ⇒ 回正尾先验"的判据（§8.2）。
    * 有界直方图 ``bucket_counts``：bucket i 覆盖 [i·W, (i+1)·W)，i < B−1；
      **末桶为开口溢出尾桶**（覆盖 [(B−1)·W, ∞)），溢出样本归并进尾桶并
      经 ``tail_count/tail_sum`` 保住尾桶均值——溢出归并不截尾（§8.2）。
    * 删失显式记录：``censored_count`` / ``censored_progress_sum``——活跃
      请求未结束即删失，不入完成样本、不进直方图/max_observed（首版仅披
      露，不做因果生存修正；§8.2"披露删失程度，必要时单列变体"）。
    """

    __slots__ = ("count", "total_sum", "max_observed", "bucket_counts",
                 "tail_count", "tail_sum", "censored_count",
                 "censored_progress_sum")

    def __init__(self, bucket_count: int) -> None:
        self.count = 0
        self.total_sum = 0
        self.max_observed = -1
        self.bucket_counts = [0] * bucket_count
        self.tail_count = 0
        self.tail_sum = 0
        self.censored_count = 0
        self.censored_progress_sum = 0

    def observe_completed(self, steps: int, width: int,
                          bucket_count: int) -> None:
        self.count += 1
        self.total_sum += steps
        if steps > self.max_observed:
            self.max_observed = steps
        index = steps // width
        if index >= bucket_count - 1:
            # 溢出归并尾桶（开口，保 count+sum 不截尾）。
            self.bucket_counts[bucket_count - 1] += 1
            self.tail_count += 1
            self.tail_sum += steps
        else:
            self.bucket_counts[index] += 1

    def observe_censored(self, exposed_steps: int) -> None:
        self.censored_count += 1
        self.censored_progress_sum += max(0, exposed_steps)

    def survival_count_above(self, steps: int, width: int,
                             bucket_count: int) -> int:
        """生存尾计数：已完成样本中（按桶表示）大于 ``steps`` 的样本数。

        保守口径：有限桶整桶在 ``steps`` 之上才计入（桶起点 > steps）；
        尾桶按其开口起点 [(B−1)·W, ∞) 计入。精确逐样本计数不保存（有界
        内存），删失披露见 ``censored_count``。
        """
        total = 0
        for i in range(bucket_count):
            if self.bucket_counts[i] and i * width > steps:
                total += self.bucket_counts[i]
        return total

    def conditional_remainder(self, d: int, width: int,
                              bucket_count: int) -> Optional[float]:
        """E[N − d | N > d] 的桶化估计；无法解析出正质量时返回 None。

        有限桶内均匀近似（整数支撑 [i·W, i·W+W) 的均值 = i·W + (W−1)/2）；
        跨 d 的桶在整数支撑上解析，严格排除 N=d；尾桶用归并样本的
        精确均值。返回 None 的情形 =
        尾桶均值 ≤ d（有界内存分辨不出尾桶内超过 d 的质量）——调用方按
        回退链降级。
        """
        num = 0.0   # Σ (N_j − d)^+
        den = 0.0   # #{N_j > d}
        for i in range(bucket_count):
            c = self.bucket_counts[i]
            if not c:
                continue
            lo = i * width
            if i == bucket_count - 1:
                tail_mean = self.tail_sum / c
                if tail_mean <= d:
                    return None
                num += c * (tail_mean - d)
                den += c
                continue
            hi = lo + width
            if lo > d:
                bucket_mean = lo + (width - 1) / 2.0
                num += c * (bucket_mean - d)
                den += c
            elif hi - 1 > d:
                survivors = hi - 1 - d
                frac = survivors / float(width)
                den += c * frac
                num += c * frac * ((survivors + 1) / 2.0)
        if den <= 0.0:
            return None
        return num / den


class ConditionalRemainingLengthEstimator:
    """已有请求的条件剩余长度（§8.2；主文 §3.2 正式量）。

    输入白名单：只接受真实结束（``observe_completed``——请求真实完成后，
    P 完成不是完成样本，同现有估计器契约）与当前已暴露进度
    （``observe_censored`` 的 exposed_progress_steps，调用方取
    ``runtime.decode_tokens_consumed`` 一类执行侧进度字段）。构造与观测
    接口不设 ``decode_length``/``final_context_tokens`` 形参——隐藏真值
    在结构上无处可传。

    回退链（§8.2）：session → 已知类型类别（category，调用方已揭示的类型
    键）→ run → 冷启动正尾先验；逐级门槛 = ``PredictionParams.
    history_min_mature_samples``，且该级须可解析（d < 该级 max_observed、
    尾桶均值 > d）。样本不足或 d 超完成样本最大值 ⇒ 带正尾的固定先验
    （geometric-remainder-mean32-v1，条件剩余口径、均值 =
    cold_start_forward_steps），**不把仍活跃请求估为零步**。

    ``version``：每次观测（完成或删失）+1——快照缓存失效键（同现有
    CausalHorizonEstimator 模式；task-load 快照隐藏输入先例见
    sh30_online_scheduler.py 的 _task_load_snapshot）。
    P/D 同比较的所有候选查询同一实例（同规则同版本）；D 边界新暴露信息经
    observe_censored / observe_completed 进入。
    """

    SOURCE_SESSION = "session_conditional"
    SOURCE_CATEGORY = "category_conditional"
    SOURCE_RUN = "run_conditional"
    SOURCE_PRIOR = "cold_start_prior"

    _REASON_NO_SAMPLES = "no_samples"
    _REASON_INSUFFICIENT = "insufficient_mature_samples"
    _REASON_D_BEYOND_MAX = "d_beyond_observed_max"
    _REASON_UNRESOLVABLE_TAIL = "unresolvable_tail_bucket"

    def __init__(self, params: PredictionParams = PredictionParams()) -> None:
        if (params.cold_start_tail_distribution
                != COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1):
            # fail-closed：未知尾部分布不用静默别名（§9.3 纪律）。
            raise ValueError(
                "cold_start_tail_distribution must be "
                f"{COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1!r} "
                "(the only W6-registered conditional-remainder tail); got "
                f"{params.cold_start_tail_distribution!r}")
        self._params = params
        self._width = int(params.length_histogram_bucket_steps)
        self._bucket_count = int(params.length_histogram_max_buckets)
        self._min_samples = int(params.history_min_mature_samples)
        self._session_stats: Dict[str, _LevelStats] = {}
        self._category_stats: Dict[str, _LevelStats] = {}
        self._run_stats = _LevelStats(self._bucket_count)
        self.version = 0

    # ------------------------------------------------------------ 观测 --
    def observe_completed(
            self, *, session_id: str, completed_forward_steps: int,
            category: Optional[str] = None,
            tick_ns: Optional[int] = None) -> None:
        """真实完成的请求进完成样本（前向步 = ``runtime.decode_length`` 口径）。

        ``tick_ns`` 仅为披露/调试保留（可缺省）；不参与估计。
        """
        if type(completed_forward_steps) is not int or completed_forward_steps < 0:
            raise ValueError(
                "completed_forward_steps must be an integer >= 0, got "
                f"{completed_forward_steps!r}")
        self._session_level(session_id).observe_completed(
            completed_forward_steps, self._width, self._bucket_count)
        if category is not None:
            self._category_level(category).observe_completed(
                completed_forward_steps, self._width, self._bucket_count)
        self._run_stats.observe_completed(
            completed_forward_steps, self._width, self._bucket_count)
        self.version += 1

    def observe_censored(
            self, *, session_id: str, exposed_progress_steps: int,
            category: Optional[str] = None,
            tick_ns: Optional[int] = None) -> None:
        """活跃请求删失登记：已知存活至 d、尚未结束（§8.2 显式记录）。

        只递增删失计数/进度和——不入完成样本、不进直方图/max_observed，
        不影响回退链的样本量门槛（删失不是成熟样本）。D 边界新暴露的进度
        由此通道进入。
        """
        if type(exposed_progress_steps) is not int or exposed_progress_steps < 0:
            raise ValueError(
                "exposed_progress_steps must be an integer >= 0, got "
                f"{exposed_progress_steps!r}")
        self._session_level(session_id).observe_censored(exposed_progress_steps)
        if category is not None:
            self._category_level(category).observe_censored(
                exposed_progress_steps)
        self._run_stats.observe_censored(exposed_progress_steps)
        self.version += 1

    # ------------------------------------------------------------ 估计 --
    def prior_remainder(self) -> int:
        """冷启动条件剩余先验（几何尾均值；与 d 无关，正支撑）。"""
        return int(self._params.cold_start_forward_steps)

    def estimate_conditional(
            self, *, session_id: str,
            exposed_progress_steps: int,
            category: Optional[str] = None,
    ) -> ConditionalRemainingEstimate:
        """E[N_total − d | N_total > d]（只读：不改状态、不 bump version）。

        回退链逐级尝试；返回值下限 1（条件于 N>d ⇒ 至少一步，同源式(4a)
        ``max(1, n̂)``）。同输入确定性（纯函数式桶化算术，无 RNG）。
        """
        if type(exposed_progress_steps) is not int or exposed_progress_steps < 0:
            raise ValueError(
                "exposed_progress_steps must be an integer >= 0, got "
                f"{exposed_progress_steps!r}")
        d = exposed_progress_steps
        chain = (
            (self.SOURCE_SESSION, self._session_stats.get(session_id)),
            (self.SOURCE_CATEGORY, self._category_stats.get(category))
            if category is not None else (self.SOURCE_CATEGORY, None),
            (self.SOURCE_RUN, self._run_stats),
        )
        for source, stats in chain:
            stats = stats if stats is not None else _LevelStats(
                self._bucket_count)
            value = self._level_remainder(stats, d)
            if value is not None:
                return ConditionalRemainingEstimate(
                    remaining_forward_steps=max(1, int(math.floor(value))),
                    source=source,
                    level_samples=stats.count,
                    censored_count=stats.censored_count,
                    fallback_reason=None,
                    disclosure=self._disclosure(d, stats, source, None))
        reason = self._prior_reason(d)
        return ConditionalRemainingEstimate(
            remaining_forward_steps=self.prior_remainder(),
            source=self.SOURCE_PRIOR,
            level_samples=self._run_stats.count,
            censored_count=self._run_stats.censored_count,
            fallback_reason=reason,
            disclosure=self._disclosure(d, self._run_stats,
                                        self.SOURCE_PRIOR, reason))

    def survival_count_above(self, steps: int, *, session_id: Optional[str] = None,
                             category: Optional[str] = None) -> int:
        """生存尾计数披露（run 级，或指定 session/category 级）。"""
        if session_id is not None:
            stats = self._session_stats.get(session_id)
        elif category is not None:
            stats = self._category_stats.get(category)
        else:
            stats = self._run_stats
        if stats is None:
            return 0
        return stats.survival_count_above(
            steps, self._width, self._bucket_count)

    def as_dict(self) -> Dict[str, Any]:
        """run 级披露（session/category 逐查询披露，避免无界枚举）。"""
        run = self._run_stats
        return {
            "version": self.version,
            "unit": "decode_forward_steps",
            "run_completed_samples": run.count,
            "run_completed_sum": run.total_sum,
            "run_max_observed": max(0, run.max_observed),
            "run_censored_count": run.censored_count,
            "run_censored_progress_sum": run.censored_progress_sum,
            "histogram_bucket_width": self._width,
            "histogram_bucket_count": self._bucket_count,
            "histogram_overflow_tail_count": run.tail_count,
            "min_samples_per_level": self._min_samples,
            "cold_start_prior_remainder": self.prior_remainder(),
            "cold_start_tail_distribution":
                self._params.cold_start_tail_distribution,
        }

    # ------------------------------------------------------------ 内部 --
    def _session_level(self, session_id: str) -> _LevelStats:
        stats = self._session_stats.get(session_id)
        if stats is None:
            stats = _LevelStats(self._bucket_count)
            self._session_stats[session_id] = stats
        return stats

    def _category_level(self, category: str) -> _LevelStats:
        stats = self._category_stats.get(category)
        if stats is None:
            stats = _LevelStats(self._bucket_count)
            self._category_stats[category] = stats
        return stats

    def _level_remainder(self, stats: _LevelStats,
                         d: int) -> Optional[float]:
        """该级可用则返回条件剩余，否则 None（样本量/超界/尾桶不可解析）。"""
        if stats.count < self._min_samples:
            return None
        if d >= stats.max_observed:
            return None
        return stats.conditional_remainder(d, self._width, self._bucket_count)

    def _prior_reason(self, d: int) -> str:
        """先验回退原因（按 run 级判据分类，可复现：披露带 d/max/count）。"""
        run = self._run_stats
        if run.count == 0:
            return self._REASON_NO_SAMPLES
        if run.count < self._min_samples:
            return self._REASON_INSUFFICIENT
        if d >= run.max_observed:
            return self._REASON_D_BEYOND_MAX
        # run 级有足量样本且 d < max，唯一剩余失败模式 = 尾桶均值 ≤ d。
        return self._REASON_UNRESOLVABLE_TAIL

    def _disclosure(self, d: int, stats: _LevelStats, source: str,
                    reason: Optional[str]) -> Dict[str, Any]:
        out = self.as_dict()
        out.update({
            "d_exposed_progress_steps": d,
            "source": source,
            "level_completed_samples": stats.count,
            "level_max_observed": max(0, stats.max_observed),
            "level_censored_count": stats.censored_count,
            "fallback_reason": reason,
        })
        return out


# ==================================================== §8.3/§8.4 到达统计 ==


@dataclass
class _CategoryState:
    """单个互斥类别的快慢强度与开放桶计数（内部可变存储单元）。"""
    lambda_fast: float = 0.0
    lambda_slow: float = 0.0
    open_bucket_count: int = 0
    total_arrivals: int = 0


@dataclass
class ArrivalCountWindow:
    """已登记未来区间的真实到达计数；边界计数不保留逐到达历史。"""
    category: Tuple[str, Optional[str]]
    starts_at_ns: int
    ends_at_ns: int
    start_total: Optional[int] = None
    end_total: Optional[int] = None

    @property
    def actual_count(self) -> Optional[int]:
        if self.start_total is None or self.end_total is None:
            return None
        return self.end_total - self.start_total


class ArrivalStatistics:
    """互斥到达类别的时间桶强度统计（§8.3/§8.4；主文 §5.2/§5.3）。

    * **互斥类别**：``new_session_first``（真实 session 首请求）与
      ``session_return``（已存在 session 的闭环后继返回），可再按已揭示
      类型分组（类别键 = (class, revealed_type)）。每个真实到达恰更新一
      个类别（``classify_arrival`` 唯一裁决点：turn_index==0 ⇒ 首请求、
      >0 ⇒ 返回）。当前队列、当前请求剩余 D、已提交通信**不是**新到达
      （``NOT_NEW_ARRIVAL_EXCLUSIONS``；``note_excluded_event`` 只落披露
      计数，绝不更新 λ）。无隐式"总到达率"类别——聚合 = 类别求和，由消
      费方显式做（避免总率+类别率双计，§8.3）。
    * **时间桶**：Δ = ``arrival_bucket_delta_ns``；闭合桶更新
      ``λ_next = (1−α)·λ + α·n/Δ``，α 只读 ``PredictionParams`` 派生属性
      ``arrival_alpha_fast/slow``（= 1 − 2^(−Δ/L)，不另行配置 alpha）。
      **空桶必须衰减**：``advance_to(now)`` 幂等推进，连续空桶按等价闭式
      ``λ ← λ·(1−α)^g`` 批量推进（中间桶按构造必空——计数只落开放桶），
      不依赖"下一次请求来才减一次"。
    * **未闭合桶规则**（两规则选一并登记，缺省冻结）：``full_buckets_only``
      ——开放桶部分计数不进 λ 直至闭合；开放桶计数进披露。
    * **预测**（主文 §5.3）：近期持续 = λ_fast；回归 =
      ``λ_slow + exp(−h/τ)·(λ_fast − λ_slow)``，τ = ``revert_tau_ns``
      （独立预测参数，不宣称平台切换必然回归）。查询只读不推进。
    * **因果**：``observed_until_ns`` 单调；``now`` 倒退 ValueError
      （fail-closed）。统计只依赖 event_time ≤ observed_until 的事件
      （双实例重放等价可验）。
    * **与效率衰减严格分离**：本类不读 ``service_*`` 参数；λ 的时间衰减
      只来自 Lf/Ls 的派生 α，绝不复用 1/16 服务 EWMA 权重。
    """

    OPEN_BUCKET_RULE = OPEN_BUCKET_RULE_FULL_BUCKETS_ONLY

    def __init__(self, params: PredictionParams = PredictionParams()) -> None:
        self._params = params
        self._delta = int(params.arrival_bucket_delta_ns)
        # 只读派生 α（不重算、不另配）。
        self._alpha_fast = params.arrival_alpha_fast
        self._alpha_slow = params.arrival_alpha_slow
        self._tau = float(params.revert_tau_ns)
        self._states: Dict[Tuple[str, Optional[str]], _CategoryState] = {}
        self._open_index: Optional[int] = None
        self._excluded_counts: Dict[str, int] = {}
        self._excluded_last_ns: Dict[str, int] = {}
        self.observed_until_ns: Optional[int] = None
        self._count_boundaries: list = []
        self._count_boundary_seq = 0
        self.version = 0

    def watch_count_window(
            self, *, category: Tuple[str, Optional[str]],
            starts_at_ns: int, ends_at_ns: int) -> ArrivalCountWindow:
        """登记 [start, end) 计数，只捕获未来边界的类别累计量。

        当前已观测到达不纳入预测真值；同 tick 后续到达仍可纳入。
        边界在到达入账之前捕获，因此右端到达属于下一桶。
        """
        frontier = self.observed_until_ns
        if (starts_at_ns < 0 or ends_at_ns <= starts_at_ns
                or (frontier is not None and starts_at_ns < frontier)):
            raise ValueError("arrival count window must be a future non-empty interval")
        if category[0] not in ARRIVAL_CLASSES:
            raise ValueError(f"unregistered arrival class {category[0]!r}")
        window = ArrivalCountWindow(category, starts_at_ns, ends_at_ns)
        if frontier == starts_at_ns:
            window.start_total = self._total_for(category)
        else:
            self._push_count_boundary(starts_at_ns, window, "start_total")
        self._push_count_boundary(ends_at_ns, window, "end_total")
        return window

    def _total_for(self, category: Tuple[str, Optional[str]]) -> int:
        state = self._states.get(category)
        return 0 if state is None else state.total_arrivals

    def _push_count_boundary(self, tick: int, window: ArrivalCountWindow,
                             field_name: str) -> None:
        self._count_boundary_seq += 1
        heapq.heappush(self._count_boundaries, (
            tick, self._count_boundary_seq, window, field_name))

    def _capture_count_boundaries(self, now_ns: int) -> None:
        while self._count_boundaries and self._count_boundaries[0][0] <= now_ns:
            _tick, _sequence, window, field_name = heapq.heappop(
                self._count_boundaries)
            setattr(window, field_name, self._total_for(window.category))

    # ------------------------------------------------------------ 裁决 --
    @staticmethod
    def classify_arrival(
            *, turn_index: int,
            revealed_type: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """真实到达的唯一互斥裁决点（§8.3）。

        turn_index == 0 ⇒ 新 session 首请求；> 0 ⇒ 已存在 session 的后续
        返回（其 alarm 已由上一轮 completion 排定，到达事件本身仍是真实
        到达）。返回类别键 (arrival_class, revealed_type)。**只裁决真实到
        达**：当前队列/当前请求剩余 D/已提交通信不进本裁决，走
        ``note_excluded_event``。
        """
        if turn_index < 0:
            raise ValueError(
                f"turn_index must be >= 0, got {turn_index!r}")
        arrival_class = (
            ARRIVAL_CLASS_NEW_SESSION_FIRST if turn_index == 0
            else ARRIVAL_CLASS_SESSION_RETURN)
        return arrival_class, revealed_type

    # ------------------------------------------------------------ 观测 --
    def observe_arrival(
            self, *, now_ns: int, turn_index: int,
            revealed_type: Optional[str] = None,
    ) -> Tuple[str, Optional[str]]:
        """classify + 推进 + 计数（真实到达事件的标准入口）。"""
        category = self.classify_arrival(
            turn_index=turn_index, revealed_type=revealed_type)
        self.record_arrival(now_ns=now_ns, category=category)
        return category

    def record_arrival(self, *, now_ns: int,
                       category: Tuple[str, Optional[str]]) -> None:
        """低层入口：把一个真实到达计入恰一个互斥类别。"""
        arrival_class, revealed = category
        if arrival_class not in ARRIVAL_CLASSES:
            raise ValueError(
                f"arrival class must be one of {ARRIVAL_CLASSES}, got "
                f"{arrival_class!r}")
        self.advance_to(now_ns)
        state = self._state_for(category)
        state.open_bucket_count += 1
        state.total_arrivals += 1
        self.observed_until_ns = now_ns
        self.version += 1

    def note_excluded_event(
            self, *, kind: str, now_ns: Optional[int] = None,
            session_id: Optional[str] = None) -> None:
        """非新到达事件的披露记录（不改任何 λ / 计数）。

        kind ∈ ``NOT_NEW_ARRIVAL_EXCLUSIONS``（当前队列 / 当前请求剩余 D /
        已提交通信）。W7 生成背景时的排除集合素材；``session_id`` 仅随披
        露记录，不参与统计。
        """
        if kind not in NOT_NEW_ARRIVAL_EXCLUSIONS:
            raise ValueError(
                f"excluded event kind must be one of "
                f"{NOT_NEW_ARRIVAL_EXCLUSIONS}, got {kind!r}")
        self._excluded_counts[kind] = (
            self._excluded_counts.get(kind, 0) + 1)
        if now_ns is not None:
            self._advance_time_only(now_ns)
            self._excluded_last_ns[kind] = now_ns
        self.version += 1

    # ------------------------------------------------------------ 推进 --
    def advance_to(self, now_ns: int) -> None:
        """幂等推进到 now_ns：闭合沿途桶（空桶批量闭式衰减）。

        先单独闭合当前开放桶（其 n 可为 0——空桶也更新 λ，即衰减），其后
        到新开放桶之间的 g 个桶按构造必为空，按等价闭式
        ``λ ← λ·(1−α)^g`` 批量推进（与逐桶推进数学等价；浮点上差异在
        最后若干 ulp 内，测试按容差对拍）。
        """
        self._advance_time_only(now_ns)

    def _advance_time_only(self, now_ns: int) -> None:
        if now_ns < 0:
            raise ValueError(f"now_ns must be >= 0, got {now_ns!r}")
        if self.observed_until_ns is not None and now_ns < (
                self.observed_until_ns):
            raise ValueError(
                f"now_ns must be monotonic: got {now_ns} < observed_until "
                f"{self.observed_until_ns} (fail-closed)")
        self._capture_count_boundaries(now_ns)
        if self._open_index is None:
            # 首次观测建立 epoch 起点（之前的桶无数据，λ 从 0 起也无可衰减）。
            self._open_index = now_ns // self._delta
            self.observed_until_ns = now_ns
            return
        target_index = now_ns // self._delta
        if target_index <= self._open_index:
            self.observed_until_ns = max(self.observed_until_ns or now_ns,
                                         now_ns)
            return
        # 1) 闭合当前开放桶（含其计数；空桶 ⇒ 纯衰减一次）。
        self._close_open_bucket()
        # 2) 中间 (target − open − 1) 个桶必空：等价闭式批量衰减。
        gap = target_index - self._open_index - 1
        if gap > 0:
            decay_fast = (1.0 - self._alpha_fast) ** gap
            decay_slow = (1.0 - self._alpha_slow) ** gap
            for state in self._states.values():
                state.lambda_fast *= decay_fast
                state.lambda_slow *= decay_slow
            self.version += 1
        self._open_index = target_index
        for state in self._states.values():
            state.open_bucket_count = 0
        self.observed_until_ns = max(self.observed_until_ns or now_ns,
                                     now_ns)

    def _close_open_bucket(self) -> None:
        for state in self._states.values():
            rate = state.open_bucket_count / float(self._delta)
            state.lambda_fast = (
                (1.0 - self._alpha_fast) * state.lambda_fast
                + self._alpha_fast * rate)
            state.lambda_slow = (
                (1.0 - self._alpha_slow) * state.lambda_slow
                + self._alpha_slow * rate)
            state.open_bucket_count = 0
        self.version += 1

    # ------------------------------------------------------------ 查询 --
    def lambdas(self, category: Tuple[str, Optional[str]],
                ) -> Tuple[float, float]:
        """(λ_fast, λ_slow)（request/ns；只读——只反映已闭合桶）。"""
        state = self._states.get(category)
        if state is None:
            return (0.0, 0.0)
        return (state.lambda_fast, state.lambda_slow)

    def intensity_recent(
            self, category: Tuple[str, Optional[str]]) -> float:
        """近期持续预测：λ̂(t+h) = λ_fast(t)（主文 §5.3）。"""
        return self.lambdas(category)[0]

    def intensity_reverting(
            self, category: Tuple[str, Optional[str]],
            horizon_ns: float) -> float:
        """向局部基线回归：λ̂(t+h) = λ_slow + e^(−h/τ)·(λ_fast − λ_slow)。

        τ = ``PredictionParams.revert_tau_ns``（独立预测参数）。回归形式
        不是既定物理规律（主文 §5.3）；τ 不宣称平台切换必然回归。
        """
        lambda_fast, lambda_slow = self.lambdas(category)
        return lambda_slow + math.exp(-horizon_ns / self._tau) * (
            lambda_fast - lambda_slow)

    def snapshot(self) -> Dict[str, Any]:
        """当前统计快照（含 observed_until 与开放桶披露；只读副本）。"""
        return {
            "version": self.version,
            "arrival_bucket_delta_ns": self._delta,
            "open_bucket_rule": self.OPEN_BUCKET_RULE,
            "observed_until_ns": self.observed_until_ns,
            "open_bucket_index": self._open_index,
            "categories": {
                f"{cls}|{typ if typ is not None else ''}": {
                    "lambda_fast": st.lambda_fast,
                    "lambda_slow": st.lambda_slow,
                    "open_bucket_count": st.open_bucket_count,
                    "total_arrivals": st.total_arrivals,
                }
                for (cls, typ), st in sorted(
                    self._states.items(), key=lambda kv: repr(kv[0]))
            },
            "excluded_event_counts": dict(sorted(
                self._excluded_counts.items())),
        }

    def exclusion_records(self) -> Dict[str, Any]:
        """排除集合披露（非新到达事件族 + 最后记录时刻）。"""
        return {
            "not_new_arrival_reasons": list(NOT_NEW_ARRIVAL_EXCLUSIONS),
            "counts": dict(sorted(self._excluded_counts.items())),
            "last_ns": dict(sorted(self._excluded_last_ns.items())),
        }

    # ------------------------------------------------------------ 内部 --
    def _state_for(self, category: Tuple[str, Optional[str]]
                   ) -> _CategoryState:
        state = self._states.get(category)
        if state is None:
            state = _CategoryState()
            self._states[category] = state
        return state


# ==================================================== §8.1 服务样本账本 ==


@dataclass(frozen=True)
class ServiceWorkSample:
    """§8.1 服务样本字段终稿（新样本至少携带的字段全集）。

    * ``busy_service_ns``：纯资源忙碌服务段实测（效率 η_sample 的分母侧
      素材）；``baseline_work_ns``：同内容硬件 roofline 基准。
    * 等待分解：``queue_wait_ns`` / ``data_wait_ns`` / ``capacity_wait_ns``
      ——非零即"该样本的端到端时长不可分离为纯服务"（§5.1 校准必须区分）。
    * ``foreign_participants``：观测窗口内共享仲裁资源上**他请求**的参与
      数（0 = 纯窗口；>0 = 争用样本，同现有 ``reject_contended`` 语义）。
    * ``hardware_version``：硬件/模型版本标识（版本混变更应走 EpochLog，
      不把版本差吸收进 η）。
    """
    work_id: str
    category: str
    observation_tick_ns: int
    baseline_work_ns: int
    busy_service_ns: int
    queue_wait_ns: int = 0
    data_wait_ns: int = 0
    capacity_wait_ns: int = 0
    foreign_participants: int = 0
    hardware_version: str = ""

    def __post_init__(self) -> None:
        for name in ("work_id", "category"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if not isinstance(self.hardware_version, str):
            raise ValueError("hardware_version must be a string")
        for name in ("observation_tick_ns", "baseline_work_ns", "busy_service_ns",
                     "queue_wait_ns", "data_wait_ns", "capacity_wait_ns",
                     "foreign_participants"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")


class ServiceSampleRecorder:
    """服务样本账本 + 统一效率估计器（§8.1；与现有 ServiceFactors 同源）。

    **纯度门**（复用现有语义：仅纯服务段样本进效率估计；争用样本不灌入
    带宽衰减——争用已由资源内核/除数通道计价，再乘一个吸收拥塞的降速
    因子即双计）：

    * ``queue_wait_ns``/``data_wait_ns``/``capacity_wait_ns`` 全 0 且
      ``foreign_participants`` == 0 ⇒ 纯样本，进效率 EWMA；
    * ``foreign_participants`` > 0 ⇒ 争用拒收（``contended`` 计数，同
      ``ServiceFactors.reject_contended`` 语义）；
    * 等待分解非零 ⇒ 不可分离：保留硬件模型 + 残差记录 + 不可识别原因
      （``residuals``），**不吸收成"有效带宽"**（§5.1）。

    **效率 EWMA（与 ServiceFactors 逐点同公式，唯一差异 = 权重/初值经
    PredictionParams 注入，默认值与现有冻结常数相等）**：同时刻同样本先
    Σactual/Σbase 汇总再取一次 η_sample（不对逐条 η 求算术平均）；
    η_sample > 1（亚 roofline 记量异常）→ 计数跳过（不钳位）；零/非有
    限 → rejected；η_new = (1−w)·η_old + w·η_sample，因子 = 1/η_new，
    flush 幂等；首样本同规则自初值出发。

    **与到达衰减严格分离（钉死）**：``service_efficiency_ewma_weight``
    （1/16，逐完成服务样本）只在本类使用；新增请求强度的时间衰减在
    ``ArrivalStatistics`` 由 Lf/Ls 派生 α 承担——本类不读任何 ``arrival_*``
    参数，``ArrivalStatistics`` 不读任何 ``service_*`` 参数；两处来源不同，
    禁止配置出互相挪用的系数（prediction_config 数值来源登记）。

    T/J/E 共享同一服务样本时只更新本统一估计器一次，再发布只读版本
    （``as_dict`` / ``efficiency_factor``）——2026-09-30 修复起生产
    落地：纯列车服务样本经 sh30 ``_observe_unified_service_sample``
    **只**入本账本；J 计价 ``ServiceFactors``（构造注入
    ``unified_source``）与 face γ/k_hide（``PublishedServiceFactorGroup``）
    在决策时点读本估计器的只读发布值，不再并联更新。
    """

    #: 残差原因（不可分离时的不可识别原因登记）。
    RESIDUAL_UNSEPARABLE_WAITS = "unseparable_waits"
    RESIDUAL_CONTENDED = "contended"

    def __init__(
        self,
        params: PredictionParams = PredictionParams(),
        epoch_log: Optional["EpochLog"] = None,
    ) -> None:
        self._params = params
        # §8.5 epoch/分布变化登记（生产由 sh30 注入共享 EpochLog）：样本
        # 硬件/模型版本分叉 = 分布变化（ServiceWorkSample 契约——版本混
        # 变更应走 EpochLog，不把版本差吸收进 η）。
        self._epoch_log = epoch_log
        # 唯一权重来源：PredictionParams.service_efficiency_ewma_weight。
        # 断言钉死"不得挪用为到达衰减"：本类不持有任何 arrival 半衰期/
        # alpha 状态；到达衰减在 ArrivalStatistics（其构造亦不读本字段）。
        self._ewma_weight = float(params.service_efficiency_ewma_weight)
        eta_init = float(params.service_efficiency_init)
        self._eta: Dict[str, float] = {}
        self._eta_init = eta_init
        self.updates: Dict[str, int] = {}
        self.rejected: Dict[str, int] = {}
        self.anomalous: Dict[str, int] = {}
        self.contended: Dict[str, int] = {}
        # (category, tick) -> [Σactual, Σbase]（同时刻汇总缓冲，同现有
        # ServiceFactors._pending 纪律；tick 推进先落上一时刻）。
        self._pending: Dict[Tuple[str, int], list] = {}
        self._pending_tick: Dict[str, int] = {}
        # (category, reason) -> {"count", "sum_actual_ns", "sum_baseline_ns"}
        self.residuals: Dict[Tuple[str, str], Dict[str, int]] = {}
        self._hardware_versions: Dict[str, None] = {}
        # 完成回调按模拟时刻单调交付；仅保留当前时刻的身份，避免为整次
        # trace 保留所有 train。旧时刻的重放不会再更新估计器。
        self._observation_frontier_ns: Optional[int] = None
        self._observed_samples: Dict[Tuple[str, str], ServiceWorkSample] = {}
        self.duplicate_samples = 0
        self.stale_samples = 0
        self.version = 0

    # ------------------------------------------------------------ 观测 --
    def observe(self, sample: ServiceWorkSample) -> bool:
        """一个完成/观测样本入账（纯度门内置于本裁决点）。"""
        tick = sample.observation_tick_ns
        frontier = self._observation_frontier_ns
        if frontier is not None and tick < frontier:
            self.stale_samples += 1
            return False
        if frontier is None or tick > frontier:
            self._observation_frontier_ns = tick
            self._observed_samples.clear()
        identity = (sample.category, sample.work_id)
        previous = self._observed_samples.get(identity)
        if previous is not None:
            if previous != sample:
                raise ValueError("conflicting service sample for the same work identity")
            self.duplicate_samples += 1
            return False
        self._observed_samples[identity] = sample
        # §8.5：硬件/模型版本分叉 = 分布变化——登记 epoch（在纯度裁决
        # 前记版本：争用/不可分离样本同样携带版本证据）。
        self._note_hardware_version(sample)
        if sample.foreign_participants > 0:
            self._record_residual(sample, self.RESIDUAL_CONTENDED)
            self.contended[sample.category] = (
                self.contended.get(sample.category, 0) + 1)
            self.version += 1
            return True
        if (sample.queue_wait_ns or sample.data_wait_ns
                or sample.capacity_wait_ns):
            self._record_residual(sample, self.RESIDUAL_UNSEPARABLE_WAITS)
            self.version += 1
            return True
        self._record_efficiency(sample)
        return True

    def _note_hardware_version(self, sample: ServiceWorkSample) -> None:
        """样本硬件/模型版本登记 + 分叉检测（§8.5；本 run 唯一运行期
        可分叉源——配置/策略/硬件摘要均在装配期一次读取 fail-closed）。
        首版本只登记；与已登记集合不同的新版本 = 版本混分叉，恰一次
        mark_epoch_change(kind="hardware")，不重置统计（no_reset_
        quarantine 规则在 EpochLog 侧）。"""
        version = sample.hardware_version
        if not version:
            return
        fork = (
            bool(self._hardware_versions)
            and version not in self._hardware_versions)
        self._hardware_versions[version] = None
        if fork and self._epoch_log is not None:
            self._epoch_log.mark_epoch_change(
                kind="hardware", at_ns=int(sample.observation_tick_ns),
                detail={"source": "service_sample_recorder",
                        "hardware_version": version,
                        "known_versions": sorted(self._hardware_versions)})

    def flush(self) -> None:
        """落同时刻汇总缓冲（读因子前调用；幂等——同现有纪律）。"""
        for key in tuple(self._pending):
            self._apply_pending(key)

    # ------------------------------------------------------------ 查询 --
    def efficiency_factor(self, category: str) -> float:
        """效率因子 = 1/η（时间倍数 ≥1；只读，读前应已 ``flush``）。"""
        self.flush()
        eta = self._eta.get(category, self._eta_init)
        return 1.0 / eta

    def as_dict(self) -> Dict[str, Any]:
        self.flush()
        return {
            "efficiency_factors": {
                cat: 1.0 / eta for cat, eta in sorted(self._eta.items())},
            "updates": dict(sorted(self.updates.items())),
            "rejected": dict(sorted(self.rejected.items())),
            "anomalous": dict(sorted(self.anomalous.items())),
            "contended": dict(sorted(self.contended.items())),
            "duplicate_samples": self.duplicate_samples,
            "stale_samples": self.stale_samples,
            "residuals": {
                f"{cat}|{reason}": dict(aggr)
                for (cat, reason), aggr in sorted(
                    self.residuals.items(), key=lambda kv: repr(kv[0]))},
            "hardware_versions": sorted(self._hardware_versions),
            "ewma_weight_source":
                "params.service_efficiency_ewma_weight",
        }

    # ------------------------------------------------------------ 内部 --
    def _record_residual(self, sample: ServiceWorkSample,
                         reason: str) -> None:
        key = (sample.category, reason)
        aggr = self.residuals.setdefault(
            key, {"count": 0, "sum_actual_ns": 0, "sum_baseline_ns": 0})
        aggr["count"] += 1
        aggr["sum_actual_ns"] += int(sample.busy_service_ns)
        aggr["sum_baseline_ns"] += int(sample.baseline_work_ns)

    def _record_efficiency(self, sample: ServiceWorkSample) -> None:
        actual = float(sample.busy_service_ns)
        base = float(sample.baseline_work_ns)
        if (not math.isfinite(actual) or not math.isfinite(base)
                or actual <= 0 or base <= 0):
            self.rejected[sample.category] = (
                self.rejected.get(sample.category, 0) + 1)
            self.version += 1
            return
        category = sample.category
        tick = int(sample.observation_tick_ns)
        last_tick = self._pending_tick.get(category)
        if last_tick is not None and last_tick != tick:
            # 时刻推进：先落上一时刻的汇总样本。
            self._apply_pending((category, last_tick))
        key = (category, tick)
        pending = self._pending.get(key)
        if pending is None:
            self._pending[key] = [actual, base]
        else:
            pending[0] += actual
            pending[1] += base
        self._pending_tick[category] = tick

    def _apply_pending(self, key: Tuple[str, int]) -> None:
        pending = self._pending.pop(key, None)
        if pending is None:
            return
        category = key[0]
        actual_sum, base_sum = pending
        eta_sample = base_sum / actual_sum
        if not math.isfinite(eta_sample) or eta_sample <= 0:
            self.rejected[category] = (
                self.rejected.get(category, 0) + 1)
            return
        if eta_sample > 1.0:
            # 亚 roofline = 记量异常：计数跳过，不钳位（同现有纪律）。
            self.anomalous[category] = (
                self.anomalous.get(category, 0) + 1)
            return
        eta_old = self._eta.get(category, self._eta_init)
        eta_new = (
            (1.0 - self._ewma_weight) * eta_old
            + self._ewma_weight * eta_sample)
        self._eta[category] = eta_new
        self.updates[category] = self.updates.get(category, 0) + 1
        self.version += 1


# ============================================== §8.5 成熟/回退/epoch 骨架 ==


class FallbackCode(enum.Enum):
    """模式回退原因码（§8.5；W7 消费）。每个码配 ``FallbackRecord`` 的
    可复现字段（detail 全部原始类型，``reproduce_key`` 确定性）。"""

    NO_SAMPLES = "no_samples"
    ANOMALY = "anomaly"
    EXPIRED = "expired"
    TEMPLATE_UNCOVERED = "template_uncovered"
    BUDGET_EXCEEDED = "budget_exceeded"
    DISTRIBUTION_SHIFT = "distribution_shift"
    UNVALIDATED_DISTANCE = "unvalidated_distance"


@dataclass(frozen=True)
class FallbackRecord:
    """一次回退的可复现记录（code + 时刻 + origin + 原始类型 detail）。"""
    code: FallbackCode
    at_ns: int
    origin: str
    detail: Mapping[str, Any] = field(default_factory=dict)

    def reproduce_key(self) -> Tuple[Any, ...]:
        """确定性复现键（同记录恒等；detail 按键排序展开）。"""
        return (self.code.value, self.at_ns, self.origin,
                tuple(sorted((str(k), repr(v))
                             for k, v in self.detail.items())))


@dataclass
class ForecastEntry:
    """一份预测的成熟账目（§8.5：issued_at_ns / matures_at_ns + 逐预测
    forecast_origin/类别/会话/排除集合登记，§8.3）。"""
    forecast_id: str
    issued_at_ns: int
    matures_at_ns: int
    origin: str
    predicted: float
    category: Optional[Tuple[str, Optional[str]]] = None
    session_id: Optional[str] = None
    exclusion_set: Tuple[str, ...] = ()
    matured: bool = False
    error: Optional[float] = None
    expired: bool = False
    fallback: Optional[FallbackRecord] = None


class ForecastMaturityLedger:
    """预测成熟账本（§8.5）：区间走完/事件完成后才可更新误差，恰一次。

    * ``issue``：登记预测（issued_at/matures_at + origin/类别/会话/排除
      集合）。matures_at ≥ issued_at（fail-closed）。
    * ``mature_entries(now_ns, actuals)``：仅当 ``matures_at_ns ≤ now``
      **且** 调用方已提供该 forecast_id 的真值（= 对应事件已完成）才结
      算误差并置 matured——**恰一次**（重复调用幂等：已 matured 的不再
      更新）；未成熟或缺真值的条目保持 pending，不进任何误差统计（未成
      熟不入模式选择）。
    * 过期：成熟时 ``issued_at_ns + history_sample_validity_ns < now`` ⇒
      标记 expired + ``FallbackCode.EXPIRED`` 记录（不产生误差更新）。
    * 内存：条目随 run 增长（单 run 有界，同现有 session 统计字典现
      状）；如需显式上界须先扩 PredictionParams，本模块不造平行常数。
    """

    def __init__(self, params: PredictionParams = PredictionParams()) -> None:
        self._params = params
        self._validity_ns = int(params.history_sample_validity_ns)
        self._entries: Dict[str, ForecastEntry] = {}
        self._order: list = []
        self._last_now: Optional[int] = None
        self.fallbacks: list = []
        self.version = 0

    # ------------------------------------------------------------ 登记 --
    def issue(self, *, forecast_id: str, issued_at_ns: int,
              matures_at_ns: int, origin: str, predicted: float,
              category: Optional[Tuple[str, Optional[str]]] = None,
              session_id: Optional[str] = None,
              exclusion_set: Tuple[str, ...] = (),
    ) -> ForecastEntry:
        if forecast_id in self._entries:
            raise ValueError(
                f"forecast_id {forecast_id!r} already issued (fail-closed; "
                "ids are unique)")
        if matures_at_ns < issued_at_ns:
            raise ValueError(
                f"matures_at_ns ({matures_at_ns}) must be >= issued_at_ns "
                f"({issued_at_ns}) for forecast {forecast_id!r}")
        for kind in exclusion_set:
            if kind not in NOT_NEW_ARRIVAL_EXCLUSIONS:
                raise ValueError(
                    f"exclusion_set entry {kind!r} is not a registered "
                    f"not-new-arrival reason {NOT_NEW_ARRIVAL_EXCLUSIONS}")
        entry = ForecastEntry(
            forecast_id=forecast_id, issued_at_ns=issued_at_ns,
            matures_at_ns=matures_at_ns, origin=origin,
            predicted=float(predicted), category=category,
            session_id=session_id, exclusion_set=tuple(exclusion_set))
        self._entries[forecast_id] = entry
        self._order.append(forecast_id)
        self.version += 1
        return entry

    # ------------------------------------------------------------ 成熟 --
    def mature_entries(
            self, now_ns: int,
            actuals: Mapping[str, float],
    ) -> list:
        """结算所有已到期且真值已提供的预测（恰一次；幂等）。

        返回本次新成熟的条目列表（调用方可用于误差披露）。``now_ns`` 单
        调（倒退 ValueError，同 ArrivalStatistics fail-closed 纪律）。
        """
        if self._last_now is not None and now_ns < self._last_now:
            raise ValueError(
                f"now_ns must be monotonic: got {now_ns} < {self._last_now}")
        self._last_now = now_ns
        newly = []
        for forecast_id in tuple(self._order):
            entry = self._entries[forecast_id]
            if entry.matured or entry.expired:
                continue
            if entry.matures_at_ns > now_ns:
                continue  # 未成熟：不入误差、不入模式选择。
            if forecast_id not in actuals:
                continue  # 区间走完但对应事件未完成（真值未揭示）。
            if entry.issued_at_ns + self._validity_ns < now_ns:
                entry.expired = True
                record = FallbackRecord(
                    code=FallbackCode.EXPIRED, at_ns=now_ns,
                    origin=entry.origin,
                    detail={
                        "forecast_id": forecast_id,
                        "issued_at_ns": entry.issued_at_ns,
                        "matures_at_ns": entry.matures_at_ns,
                        "validity_ns": self._validity_ns,
                    })
                entry.fallback = record
                self.fallbacks.append(record)
                self.version += 1
                continue
            entry.error = float(entry.predicted) - float(
                actuals[forecast_id])
            entry.matured = True
            newly.append(entry)
            self.version += 1
        return newly

    # ------------------------------------------------------------ 查询 --
    def entry(self, forecast_id: str) -> Optional[ForecastEntry]:
        return self._entries.get(forecast_id)

    def matured_entries(self) -> list:
        """已成熟条目（误差可用的唯一来源；未成熟/过期不在内）。"""
        return [self._entries[fid] for fid in self._order
                if self._entries[fid].matured]

    def pending_entries(self) -> list:
        return [self._entries[fid] for fid in self._order
                if not self._entries[fid].matured
                and not self._entries[fid].expired]

    def matured_count(self) -> int:
        """已成熟误差样本数（W7 与 history_min_mature_samples 比较用）。"""
        return sum(1 for e in self._entries.values() if e.matured)

    def mean_error(self) -> Optional[float]:
        matured = self.matured_entries()
        if not matured:
            return None
        return sum(e.error for e in matured) / len(matured)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "issued": len(self._entries),
            "matured": self.matured_count(),
            "pending": len(self.pending_entries()),
            "expired": sum(1 for e in self._entries.values() if e.expired),
            "mean_error": self.mean_error(),
            "validity_ns": self._validity_ns,
            "fallbacks": [
                {"code": r.code.value, "at_ns": r.at_ns, "origin": r.origin,
                 "detail": dict(r.detail)}
                for r in self.fallbacks],
        }

    # ------------------------------------------------------------ 内部 --


class EpochLog:
    """epoch/分布变化标记接口（§8.5：配置/策略/路由/硬件变化时调用）。

    冻结规则位 ``EPOCH_RESET_RULE = "no_reset_quarantine"``：变化**不重置**
    统计（防"结果变差时偷偷换窗口"），改进隔离窗——变化时刻起
    ``history_drift_quarantine_ns`` 内 ``in_quarantine`` 为真（W7 据此回退
    distribution_shift、不启用 validated），窗外恢复。
    """

    EPOCH_RESET_RULE = EPOCH_RESET_RULE_NO_RESET_QUARANTINE

    def __init__(self, params: PredictionParams = PredictionParams()) -> None:
        self._quarantine_ns = int(params.history_drift_quarantine_ns)
        self.epochs: list = []
        self.version = 0

    def mark_epoch_change(self, *, kind: str, at_ns: int,
                          detail: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """登记一次分布变化（kind ∈ config/policy/route/hardware/...）。"""
        if not kind.strip():
            raise ValueError("epoch kind must be a non-empty label")
        if self.epochs and at_ns < self.epochs[-1]["at_ns"]:
            raise ValueError(
                f"epoch times must be monotonic: got {at_ns} < "
                f"{self.epochs[-1]['at_ns']}")
        self.epochs.append({
            "kind": kind, "at_ns": at_ns,
            "reset_rule": self.EPOCH_RESET_RULE,
            "detail": dict(detail or {}),
        })
        self.version += 1

    def last_epoch(self) -> Optional[Mapping[str, Any]]:
        return self.epochs[-1] if self.epochs else None

    def quarantine_until_ns(self) -> Optional[int]:
        last = self.last_epoch()
        if last is None:
            return None
        return last["at_ns"] + self._quarantine_ns

    def in_quarantine(self, now_ns: int) -> bool:
        until = self.quarantine_until_ns()
        return until is not None and now_ns < until

    def fallback_if_blocked(self, now_ns: int,
    ) -> Optional[FallbackRecord]:
        """隔离期内给出可复现的 distribution_shift 回退记录（W7 消费）。"""
        last = self.last_epoch()
        if last is None or not self.in_quarantine(now_ns):
            return None
        return FallbackRecord(
            code=FallbackCode.DISTRIBUTION_SHIFT, at_ns=now_ns,
            origin="epoch_log",
            detail={"kind": last["kind"], "epoch_at_ns": last["at_ns"],
                    "quarantine_ns": self._quarantine_ns,
                    "reset_rule": self.EPOCH_RESET_RULE})

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "reset_rule": self.EPOCH_RESET_RULE,
            "quarantine_ns": self._quarantine_ns,
            "epochs": list(self.epochs),
        }
