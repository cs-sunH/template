"""forecast_validation.py -- W7：预测成熟、按距离准入、回退链与开销计量。

《joint仓库当下+历史修改方案》§8.5（训练、冷启动、异常与漂移）、§8.7
（历史正式启用门槛）、W7 段与主文 §8-9 的落地。三件事：

1. **预测成熟**（§8.5"只有预测区间走完/对应事件完成后才能更新误差"）：
   复用 W6 ``ForecastMaturityLedger`` 语义——``issued_at_ns``/
   ``matures_at_ns`` 全记录、成熟误差更新恰一次（幂等）、过期显式回退。
   本模块补"到达窗桶预测的真值回填"通道：桶强度预测在桶末端过后用
   ``ArrivalStatistics`` 的类别累计到达差分作真值（调用方
   ``ForecastMaturityDriver.mature``）。
2. **按距离准入分组**（§8.7"按预测距离、运行阶段、已知承诺占比、样本
   质量等预注册分组；不强制所有距离启用同一种历史模式"）：
   ``DistanceAdmissionLedger`` 逐 (mode, distance_band) 维护成熟误差与
   验证登记。转正唯一依据 = 显式验证登记（独立验证前缀/W8 同状态实验
   提供）；run 内在线成熟误差只作 shadow 披露——同 run 误差自评不得
   自证转正。``history_use=validated`` 运行时核验：无验证登记的
   (mode, band) → ``UNVALIDATED_DISTANCE`` 回退 shadow（该距离桶的注入
   不进正式计价，只产误差），**不 fail 整个 run**；``shadow`` 档零注
   入（全部 crossing 只产桶预测与成熟误差，不影响选择，§8.7 先
   shadow 门槛）。
3. **固定回退链**（§8.5 + 方案 W7 段）：``run_fallback_chain`` 按冻结
   顺序 no_samples → anomaly → expired → template_uncovered →
   budget_exceeded → distribution_shift → unvalidated_distance 逐项
   检查，首个命中返回可复现 ``FallbackRecord``（W6 类型，detail 全原始
   类型、``reproduce_key`` 确定性）；回退 = 退到 B（零背景注入），
   **不删除统计、不伪造未来空闲**。epoch/分布变化隔离窗 =
   ``params.history_drift_quarantine_ns``（W6 ``EpochLog.
   no_reset_quarantine`` 规则位）。
4. **开销计量**（§8.4/§10"报告触发率"）：``ForecastOverheadLedger``
   逐决策记录注入模板数/腿数/事件数/宿主耗时/回退计数（挂钩点，快照
   披露；宿主 wall time 与目标控制时间分别报告——W0 纪律）。

数值单一来源 ``PredictionParams``（W7 preset 段：距离桶界/容差/预算/
阈值；零平行常数）。
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from joint.history_estimator import (
    ArrivalCountWindow,
    ArrivalStatistics,
    EpochLog,
    FallbackCode,
    FallbackRecord,
    ForecastEntry,
    ForecastMaturityLedger,
)
from joint.prediction_config import (
    HISTORY_USE_SHADOW,
    HISTORY_USE_VALIDATED,
    JointPredictionConfig,
    PredictionParams,
)

__all__ = [
    "DISTANCE_BAND_NEAR",
    "DISTANCE_BAND_MID",
    "DISTANCE_BAND_FAR",
    "distance_band_for_offset",
    "FallbackChainResult",
    "run_fallback_chain",
    "DistanceAdmissionLedger",
    "ForecastOverheadLedger",
    "ForecastMaturityDriver",
    "CompletionMaturityLedger",
]

#: 预测距离档标签（桶界 = params.forecast_distance_buckets_ns；预注册
#: 分组——§8.7"按预测距离分组"，桶界 preset 登记 EVIDENCE）。
DISTANCE_BAND_NEAR = "near"
DISTANCE_BAND_MID = "mid"
DISTANCE_BAND_FAR = "far"


def distance_band_for_offset(
        offset_ns: float, params: PredictionParams) -> str:
    """预测距离 |注入时刻 − now| 落入的距离档（冻结界 = preset 桶界）。

    界数 2（缺省 (1×Δ, 4×Δ)）⇒ 档 near/mid/far：offset < b0 ⇒ near、
    offset < b1 ⇒ mid、其余 far。更多界时第 3 档起并入 far（档数封顶
    3，登记冻结）。
    """
    offset = float(offset_ns)
    bounds = tuple(params.forecast_distance_buckets_ns)
    index = sum(1 for bound in bounds if offset >= float(bound))
    return (DISTANCE_BAND_NEAR, DISTANCE_BAND_MID, DISTANCE_BAND_FAR)[
        min(index, 2)]


# ------------------------------------------------------------- 回退链 --


@dataclass(frozen=True)
class FallbackChainResult:
    """回退链一次评估的输出（首个命中 + 距离准入拒绝集）。

    * ``fatal``：no_samples/anomaly/expired/template_uncovered/
      budget_exceeded/distribution_shift 之一——整档回退 B（零背景注入）。
    * ``unvalidated_bands``：``history_use=validated`` 下无验证登记的
      (mode, band) 集合——该距离桶注入降级 shadow（不进正式计价），
      run 继续；``unvalidated_record`` 为可复现回退记录（无拒绝为
      None）。二者互斥于 fatal（fatal 时不评距离准入）。
    """

    fatal: Optional[FallbackRecord]
    unvalidated_bands: Tuple[str, ...] = ()
    unvalidated_record: Optional[FallbackRecord] = None

    @property
    def fell_back_to_none(self) -> bool:
        return self.fatal is not None


def run_fallback_chain(
        *,
        config: JointPredictionConfig,
        arrival_stats: ArrivalStatistics,
        epoch_log: Optional[EpochLog],
        now_ns: int,
        facts_available: bool,
        requested_templates: int,
        bands_used: Sequence[str],
        distance_ledger: "DistanceAdmissionLedger",
        intensity_values: Sequence[float],
        params: Optional[PredictionParams] = None,
) -> FallbackChainResult:
    """§8.5/W7 冻结回退链（顺序固定，首个命中短路）。

    顺序（任务/方案 §8.5 固定，登记 PLAN §3）：
    ``no_samples → anomaly → expired → template_uncovered →
    budget_exceeded → distribution_shift → unvalidated_distance``。
    全部命中均返回可复现 ``FallbackRecord``（detail 原始类型）。
    """
    params = params if params is not None else config.params
    mode = config.future_demand_mode
    origin = f"future_demand:{mode}"

    # 1. no_samples：无任何真实到达观测（λ 全零且无类别历史）。
    snapshot = arrival_stats.snapshot()
    categories = snapshot.get("categories", {})
    total_arrivals = sum(
        int(entry.get("total_arrivals", 0)) for entry in categories.values())
    if total_arrivals == 0:
        return FallbackChainResult(fatal=FallbackRecord(
            code=FallbackCode.NO_SAMPLES, at_ns=now_ns, origin=origin,
            detail={
                "observed_until_ns": snapshot.get("observed_until_ns"),
                "category_count": len(categories),
                "min_mature_samples": params.history_min_mature_samples,
            }))

    # 2. anomaly：数值异常（非有限/负强度——记量异常不外推）。
    for index, value in enumerate(intensity_values):
        if not math.isfinite(value) or value < 0.0:
            return FallbackChainResult(fatal=FallbackRecord(
                code=FallbackCode.ANOMALY, at_ns=now_ns, origin=origin,
                detail={
                    "intensity_index": index,
                    "intensity_value": None if not math.isfinite(value)
                    else float(value),
                    "non_finite": not math.isfinite(value),
                }))

    # 3. expired：统计状态过期（观测前沿 + 有效期 < now）。
    observed_until = snapshot.get("observed_until_ns")
    if observed_until is not None and (
            int(observed_until) + int(params.history_sample_validity_ns)
            < int(now_ns)):
        return FallbackChainResult(fatal=FallbackRecord(
            code=FallbackCode.EXPIRED, at_ns=now_ns, origin=origin,
            detail={
                "observed_until_ns": int(observed_until),
                "validity_ns": int(params.history_sample_validity_ns),
                "now_ns": int(now_ns),
            }))

    # 4. template_uncovered：模板不覆盖当前资源/类型（事实缺失或无可
    #    用放置实例——缺项不假设无限资源，§9.2）。
    if not facts_available:
        return FallbackChainResult(fatal=FallbackRecord(
            code=FallbackCode.TEMPLATE_UNCOVERED, at_ns=now_ns,
            origin=origin, detail={"facts_available": False}))

    # 5. budget_exceeded：预测注入超预算（回退整档，不截断冒充）。
    if requested_templates > int(params.forecast_template_budget):
        return FallbackChainResult(fatal=FallbackRecord(
            code=FallbackCode.BUDGET_EXCEEDED, at_ns=now_ns, origin=origin,
            detail={
                "requested_templates": int(requested_templates),
                "template_budget": int(params.forecast_template_budget),
            }))

    # 6. distribution_shift：epoch/分布变化隔离窗内（W6 no_reset_
    #    quarantine 规则位——不重置统计，只降级）。
    if epoch_log is not None:
        quarantine = epoch_log.fallback_if_blocked(now_ns)
        if quarantine is not None:
            return FallbackChainResult(fatal=quarantine)

    # 7. unvalidated_distance：validated 档运行时核验——无验证登记的
    #    (mode, band) 注入降级 shadow（不 fail run，披露继续）。
    if config.history_use == HISTORY_USE_VALIDATED:
        rejected = tuple(sorted({
            band for band in set(bands_used)
            if not distance_ledger.is_validated(mode, band)[0]}))
        if rejected:
            return FallbackChainResult(
                fatal=None,
                unvalidated_bands=rejected,
                unvalidated_record=FallbackRecord(
                    code=FallbackCode.UNVALIDATED_DISTANCE, at_ns=now_ns,
                    origin=origin,
                    detail={
                        "mode": mode,
                        "rejected_bands": list(rejected),
                        "history_use": config.history_use,
                        "downgraded_to": HISTORY_USE_SHADOW,
                    }))
    return FallbackChainResult(fatal=None)


# ------------------------------------------------------ 距离准入账本 --


@dataclass
class _BandStats:
    """单 (mode, band) 的在线成熟误差聚合（有界标量）。"""
    samples: int = 0
    sum_signed_normalized_error: float = 0.0
    sum_abs_normalized_error: float = 0.0


class DistanceAdmissionLedger:
    """按 (mode, distance_band) 的准入账本（§8.7 预注册分组）。

    转正门槛（§8.7"先 shadow"）：唯一判 validated 的依据是**显式验证
    登记**（``validation_registry``，构造注入）——独立开发/验证前缀产
    出的 (mode, band) 到达绝对误差、同状态选择损失与来源记录；登记须
    匹配当前模型世代及参数 preset，不能用正负误差抵消证明准确。
    （登记由 W8 同状态实验显式提供；首版交付无预置登记——没有历史收
    益是合法研究结果，不得自证）。run 内在线成熟误差（``record_
    matured``）**只作 shadow 披露**：同 run 误差自评不构成转正依据
    （§8.7"不能凭未执行候选的估计自证排序正确"；正式计价转正还须
    §8.7 的受控选择损失证据，由登记流程一并把关）。

    ``is_validated`` 返回 (bool, reason dict)——reason 可复现披露（登记
    来源/样本量/均值/在线聚合）。
    """

    def __init__(
            self,
            params: PredictionParams = PredictionParams(),
            *,
            validation_registry: Optional[
                Mapping[Tuple[str, str], Mapping[str, float]]] = None,
    ) -> None:
        self._params = params
        self._registry: Dict[Tuple[str, str], Mapping[str, float]] = {
            (str(mode), str(band)): dict(record)
            for (mode, band), record in (validation_registry or {}).items()}
        self._online: Dict[Tuple[str, str], _BandStats] = {}
        self.version = 0

    def record_matured(self, *, mode: str, band: str,
                       signed_normalized_error: float) -> None:
        """一条成熟误差入账（符号保留——偏差方向披露，§10.3；仅披露，
        不改变 is_validated）。"""
        key = (str(mode), str(band))
        stats = self._online.setdefault(key, _BandStats())
        stats.samples += 1
        stats.sum_signed_normalized_error += float(
            signed_normalized_error)
        stats.sum_abs_normalized_error += abs(
            float(signed_normalized_error))
        self.version += 1

    def is_validated(
            self, mode: str, band: str,
    ) -> Tuple[bool, Mapping[str, Any]]:
        """(mode, band) 是否持**显式验证登记**（§8.7 转正唯一依据）。

        无登记恒 False（reason 携带在线成熟聚合作 shadow 披露——在线
        样本量/均值不参与判定，不存在 run 内误差自评转正路径）。"""
        key = (str(mode), str(band))
        if key in self._registry:
            record = self._registry[key]
            samples = record.get("samples")
            mean_error = record.get("mean_abs_normalized_error")
            valid_samples = (type(samples) is int and
                             samples >= self._params.history_min_mature_samples)
            valid_error = (type(mean_error) in (int, float) and
                           math.isfinite(mean_error) and
                           0 <= mean_error <= self._params.forecast_distance_tolerance)
            loss_samples = record.get("selection_loss_samples")
            loss = record.get("mean_selection_loss_ns")
            baseline_loss = record.get("baseline_mean_selection_loss_ns")
            valid_loss = (
                type(loss_samples) is int and
                loss_samples >= self._params.history_min_mature_samples and
                type(loss) in (int, float) and math.isfinite(loss) and loss >= 0 and
                type(baseline_loss) in (int, float) and
                math.isfinite(baseline_loss) and baseline_loss >= loss)
            from joint.joint_config import MODEL_SEMANTICS_VERSION
            registered_params = record.get("params")
            valid_params = False
            if isinstance(registered_params, Mapping):
                normalized_params = dict(registered_params)
                buckets = normalized_params.get("forecast_distance_buckets_ns")
                if isinstance(buckets, (list, tuple)):
                    normalized_params["forecast_distance_buckets_ns"] = list(buckets)
                valid_params = normalized_params == self._params.params_manifest()
            valid_provenance = (
                isinstance(record.get("validation_source"), str) and
                bool(record["validation_source"].strip()) and
                record.get("validation_scope") == "independent_prefix" and
                record.get("model_semantics_version") == MODEL_SEMANTICS_VERSION and
                valid_params)
            admitted = valid_samples and valid_error and valid_loss and valid_provenance
            return admitted, {
                "source": ("registered_validation" if admitted else
                           "registered_validation_rejected"),
                "samples": samples if type(samples) is int else None,
                "mean_abs_normalized_error": (
                    mean_error if type(mean_error) in (int, float) and
                    math.isfinite(mean_error) else None),
                "quality_passed": admitted,
                "arrival_quality_passed": valid_samples and valid_error,
                "selection_loss_passed": valid_loss,
                "provenance_passed": valid_provenance,
            }
        stats = self._online.get(key)
        return False, {
            "source": "unregistered",
            "online_matured_samples": (
                0 if stats is None else stats.samples),
            "online_mean_normalized_error": (
                stats.sum_signed_normalized_error / stats.samples
                if stats is not None and stats.samples else None),
            "online_disclosure_only": True,
            "admission_requires": "validation_registry",
        }

    def as_dict(self) -> Dict[str, Any]:
        online: Dict[str, Any] = {}
        for (mode, band), stats in sorted(self._online.items()):
            mean = (stats.sum_signed_normalized_error / stats.samples
                    if stats.samples else None)
            online[f"{mode}|{band}"] = {
                "samples": stats.samples,
                "mean_normalized_error": mean,
                "mean_abs_normalized_error": (
                    stats.sum_abs_normalized_error / stats.samples
                    if stats.samples else None),
                # 在线成熟样本只作 shadow 披露——转正唯一依据 = 显式验
                # 证登记（§8.7；同 run 误差自评不构成 validated）。
                "counts_toward_admission": False,
            }
        return {
            "version": self.version,
            "registered_bands": sorted(
                f"{mode}|{band}" for mode, band in self._registry),
            "online": online,
        }


# ------------------------------------------------------------ 开销账本 --


class ForecastOverheadLedger:
    """W7 开销计量（§8.4/§10：注入模板数/事件数/宿主耗时挂钩点）。

    恒开、无开关（验收必需项）；快照只披露测量口径，宿主 Python 墙时
    ≠ 片上控制时延（W0 纪律）。回退触发率 = fallbacks_by_code / max(1,
    decisions)。
    """

    def __init__(self) -> None:
        self.decisions = 0
        self.templates_injected = 0
        self.shadow_records = 0
        self.legs_generated = 0
        self.bucket_forecasts_issued = 0
        self.host_time_ns = 0
        self.fallbacks_by_code: Dict[str, int] = {}
        self.version = 0

    def note_decision(
            self,
            *,
            templates_injected: int,
            shadow_records: int,
            legs_generated: int,
            bucket_forecasts_issued: int,
            host_time_ns: int,
            fallback_codes: Sequence[str] = (),
    ) -> None:
        self.decisions += 1
        self.templates_injected += int(templates_injected)
        self.shadow_records += int(shadow_records)
        self.legs_generated += int(legs_generated)
        self.bucket_forecasts_issued += int(bucket_forecasts_issued)
        self.host_time_ns += int(host_time_ns)
        for code in fallback_codes:
            self.fallbacks_by_code[code] = (
                self.fallbacks_by_code.get(code, 0) + 1)
        self.version += 1

    def as_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "decisions": self.decisions,
            "templates_injected": self.templates_injected,
            "shadow_records": self.shadow_records,
            "legs_generated": self.legs_generated,
            "bucket_forecasts_issued": self.bucket_forecasts_issued,
            "host_time_ns": self.host_time_ns,
            "fallbacks_by_code": dict(sorted(
                self.fallbacks_by_code.items())),
            "fallback_trigger_rate": (
                sum(self.fallbacks_by_code.values())
                / max(1, self.decisions)),
        }


# ------------------------------------------------- 成熟真值回填驱动器 --


class ForecastMaturityDriver:
    """到达窗桶预测的成熟驱动器（§8.5 恰一次 + 真值差分回填）。

    * ``note_issued``：登记每个预测桶的左右边界；真值只包含该桶内
      的真实到达，迟交付不会纳入桶外事件。
    * ``mature(now_ns)``：桶末端已过的预测用差分真值经 W6
      ``ForecastMaturityLedger.mature_entries`` 结算（幂等、恰一次），
      成熟误差按 (mode, band) 入 ``DistanceAdmissionLedger``
      （normalized = (predicted − actual)/max(1, predicted)）。
    """

    def __init__(
            self,
            *,
            arrival_stats: ArrivalStatistics,
            maturity_ledger: ForecastMaturityLedger,
            distance_ledger: DistanceAdmissionLedger,
    ) -> None:
        self._arrival_stats = arrival_stats
        self._ledger = maturity_ledger
        self._distance_ledger = distance_ledger
        self._windows: Dict[str, ArrivalCountWindow] = {}
        self._bands: Dict[str, Tuple[str, str]] = {}
        self.last_matured: List[ForecastEntry] = []

    def note_issued(
            self, *, forecast_id: str, mode: str, band: str,
            starts_at_ns: int) -> None:
        """登记桶预测的真实区间与 (mode, band)（issue 之后调用）。"""
        entry = self._ledger.entry(forecast_id)
        if entry is None or entry.category is None:
            raise ValueError("bucket forecast must be issued with an arrival category")
        self._windows[forecast_id] = self._arrival_stats.watch_count_window(
            category=tuple(entry.category), starts_at_ns=starts_at_ns,
            ends_at_ns=entry.matures_at_ns)
        self._bands[forecast_id] = (str(mode), str(band))

    def mature(self, now_ns: int) -> List[ForecastEntry]:
        """结算所有已到期桶预测（差分真值；恰一次、幂等）。"""
        self._arrival_stats.advance_to(now_ns)
        actuals: Dict[str, float] = {}
        for forecast_id in list(self._bands):
            entry = self._ledger.entry(forecast_id)
            if entry is None or entry.matured or entry.expired:
                continue
            if entry.matures_at_ns > now_ns:
                continue  # 未成熟：不入误差、不入准入（§8.5）。
            if entry.category is None:
                continue
            actual_count = self._windows[forecast_id].actual_count
            if actual_count is not None:
                actuals[forecast_id] = float(actual_count)
        newly = self._ledger.mature_entries(now_ns, actuals)
        for entry in newly:
            band_info = self._bands.get(entry.forecast_id)
            if band_info is None:
                continue
            mode, band = band_info
            # error = predicted − actual（W6 口径）；normalized 分母 =
            # max(1, predicted)——预注册容差定义（preset 登记）。
            actual = float(entry.predicted) - float(entry.error or 0.0)
            normalized = (
                (float(entry.predicted) - actual)
                / max(1.0, float(entry.predicted)))
            self._distance_ledger.record_matured(
                mode=mode, band=band,
                signed_normalized_error=normalized)
        self.last_matured = list(newly)
        for forecast_id in list(self._bands):
            entry = self._ledger.entry(forecast_id)
            if entry is None or entry.matured or entry.expired:
                self._bands.pop(forecast_id)
                self._windows.pop(forecast_id)
        return newly

    def as_dict(self) -> Dict[str, Any]:
        return {
            "tracked_bucket_forecasts": len(self._bands),
            "maturity_ledger": self._ledger.as_dict(),
            "distance_admission": self._distance_ledger.as_dict(),
        }


# -------------------------------------------- 完成时间预测成熟账本（cfg1） --

#: 决策阶段词表（与 D17 schema 决策行 decision_stage 对齐——P 准入 /
#: D 边界；其余值 fail-closed 拒绝，不静默归并）。
COMPLETION_DECISION_STAGES: Tuple[str, ...] = ("P", "D")


@dataclass
class _CompletionEntry:
    """一条已执行候选的完成时间预测登记（§10.3 直接真值配对单元）。"""
    request_id: str
    decision_stage: str
    decision_tick_ns: int
    action: str
    predicted_next_token_done_ns: Optional[int]
    predicted_service_done_ns: Optional[int]
    predicted_settle_done_ns: Optional[int]
    matured: bool = False
    actual_service_ns: Optional[int] = None
    actual_settle_ns: Optional[int] = None


def _nearest_rank_int(sorted_values: Sequence[int], fraction: float) -> int:
    """最近秩分位（升序样本、1 基 ceil 秩；空集调用方保证不达）。"""
    count = len(sorted_values)
    rank = -(-int(fraction * 100) * count // 100)
    rank = min(max(rank, 1), count)
    return int(sorted_values[rank - 1])


class CompletionMaturityLedger:
    """P/D 完成时间预测的在线成熟账本（§10.3，cfg1 2026-09-30）。

    文档条款："在线只有实际执行候选能产生该次预测的直接真值。未执行
    候选保持 missing…完成时间预测在相应服务/结算完成后成熟；未完成/
    超时样本单列，不能丢掉以美化尾部"；"分别报告 P 与 D 的绝对误差、
    偏差方向、分位误差、归一化分母"。

    * ``note_decision``：调度侧在**实际执行候选**的决策行发射点登记
      （每 (request, decision_stage) 恰一次；重复登记 fail-closed——
      配对账目破损当场报，不静默覆盖）。未执行候选不登记（决策行
      measured 槽位恒 None 的既有口径不变——本账本不接收估计冒充）。
    * ``mature``：结算闭合点（settle_done 事件）按 request 配对实测——
      实际 service/settle = 事件 tick − 决策 tick（与 predicted_* 同
      决策相对轴）；误差 = 实际 − 预测（带方向）；归一化分母 =
      max(1, |predicted|)（与 DistanceAdmissionLedger 同族口径，规则
      字串随披露）。幂等（恰一次：重复 mature 对已成熟条目空放）。
      service_done 缺席的条目计入 ``service_unpaired`` 单列（不丢）。
    * ``as_dict``：按 (stage, endpoint) 聚合——登记/成熟/未决（超时/
      未完成单列）计数、带符号分位（min/p50/p90/max）、绝对分位
      （p50/p90）、归一化均值（带方向与绝对）与分母规则；动作分布
      最小面（分组细化由离线验证报告按决策日志展开——slo_tools/
      domain_metrics 的 completion_error_report）。
    """

    #: 归一化分母规则（披露字串；与 distance ledger 的 max(1, predicted)
    #: 同族，completion 预测恒非负故 |·| 仅防御）。离线对账钉测：
    #: slo_tools/tests/test_domain_metrics.py::
    #: OnlineOfflineLedgerReconciliationTests（与离线 ERROR_DENOMINATOR_RULE
    #: 同字串锁定——任一侧单改即在该钉测报错）。
    DENOMINATOR_RULE = "(actual-predicted)/max(1,|predicted|) ns"

    def __init__(self) -> None:
        self._entries: Dict[Tuple[str, str], _CompletionEntry] = {}
        self.version = 0

    def note_decision(
            self, *, request_id: str, decision_stage: str,
            decision_tick_ns: int, action: str,
            predicted_next_token_done_ns: Optional[int],
            predicted_service_done_ns: Optional[int],
            predicted_settle_done_ns: Optional[int]) -> None:
        """登记一条已执行候选的完成预测（每 (request, stage) 恰一次）。"""
        stage = str(decision_stage)
        if stage not in COMPLETION_DECISION_STAGES:
            raise ValueError(
                f"unknown completion decision stage {stage!r} "
                f"(expected one of {COMPLETION_DECISION_STAGES})")
        key = (str(request_id), stage)
        if key in self._entries:
            raise ValueError(
                "completion maturity double registration for "
                f"({request_id!r}, {stage!r})—pairing ledger corrupted")
        self._entries[key] = _CompletionEntry(
            request_id=str(request_id),
            decision_stage=stage,
            decision_tick_ns=int(decision_tick_ns),
            action=str(action),
            predicted_next_token_done_ns=(
                None if predicted_next_token_done_ns is None
                else int(predicted_next_token_done_ns)),
            predicted_service_done_ns=(
                None if predicted_service_done_ns is None
                else int(predicted_service_done_ns)),
            predicted_settle_done_ns=(
                None if predicted_settle_done_ns is None
                else int(predicted_settle_done_ns)))
        self.version += 1

    def mature(self, *, request_id: str, service_done_ns: Optional[int],
               settle_done_ns: Optional[int]) -> int:
        """按 request 配对实测（幂等）；返回本次新成熟条数。

        settle_done 缺席（None）不成熟任何条目——结算未闭合即无真值
        （未完成样本保持登记单列，不丢不放）。"""
        if settle_done_ns is None:
            return 0
        newly = 0
        for entry_request, entry in self._entries.items():
            if entry_request[0] != str(request_id) or entry.matured:
                continue
            entry.matured = True
            newly += 1
            entry.actual_settle_ns = (
                int(settle_done_ns) - entry.decision_tick_ns)
            entry.actual_service_ns = (
                int(service_done_ns) - entry.decision_tick_ns
                if service_done_ns is not None else None)
        if newly:
            self.version += 1
        return newly

    def has_entries(self) -> bool:
        return bool(self._entries)

    def _stage_block(self, stage: str) -> Dict[str, Any]:
        entries = [entry for entry in self._entries.values()
                   if entry.decision_stage == stage]
        if not entries:
            return {}
        matured = [entry for entry in entries if entry.matured]
        pending = [entry for entry in entries if not entry.matured]

        def _error_block(endpoint: str) -> Dict[str, Any]:
            """endpoint ∈ {service, settle} 的成熟误差聚合（None = 无
            配对样本——如实空形态，不造零）。"""
            errs: List[int] = []
            normalized_signed: List[float] = []
            normalized_abs: List[float] = []
            unpaired = 0
            for entry in matured:
                actual = getattr(entry, f"actual_{endpoint}_ns")
                predicted = getattr(entry, f"predicted_{endpoint}_done_ns")
                if predicted is None:
                    unpaired += 1
                    continue
                if actual is None:
                    # settle 恒有（mature 门控）；service_done 缺席 =
                    # 未完成/超时单列（§10.3 不丢尾部）。
                    unpaired += 1
                    continue
                error = int(actual) - int(predicted)
                errs.append(error)
                denominator = max(1, abs(int(predicted)))
                normalized_signed.append(error / denominator)
                normalized_abs.append(abs(error) / denominator)
            if not errs and not unpaired:
                return {}
            block: Dict[str, Any] = {}
            if errs:
                signed = sorted(errs)
                abs_sorted = sorted(abs(err) for err in errs)
                block["n"] = len(errs)
                block["signed"] = {
                    "min": signed[0],
                    "p50": _nearest_rank_int(signed, 0.50),
                    "p90": _nearest_rank_int(signed, 0.90),
                    "max": signed[-1],
                }
                block["abs"] = {
                    "p50": _nearest_rank_int(abs_sorted, 0.50),
                    "p90": _nearest_rank_int(abs_sorted, 0.90),
                }
                block["normalized_mean_signed"] = (
                    sum(normalized_signed) / len(normalized_signed))
                block["normalized_mean_abs"] = (
                    sum(normalized_abs) / len(normalized_abs))
            if unpaired:
                block["unpaired"] = unpaired
            return block

        actions: Dict[str, int] = {}
        for entry in entries:
            actions[entry.action] = actions.get(entry.action, 0) + 1
        block = {
            "registered": len(entries),
            "matured": len(matured),
            "pending": len(pending),
            "actions": dict(sorted(actions.items())),
            "settle_error_ns": _error_block("settle"),
            "service_error_ns": _error_block("service"),
        }
        return block

    def as_dict(self) -> Dict[str, Any]:
        stages: Dict[str, Any] = {}
        for stage in COMPLETION_DECISION_STAGES:
            block = self._stage_block(stage)
            if block:
                stages[stage] = block
        return {
            "version": self.version,
            "denominator_rule": self.DENOMINATOR_RULE,
            "stages": stages,
            "note": (
                "成熟 = settle_done 事件配对（P/D 分列；误差 = 实际 − "
                "预测，决策相对轴；未执行候选不登记保持 missing——"
                "measured 槽位恒 None 口径不变）；pending = 已登记未成"
                "熟（未完成/超时单列）；分组细化（距离/动作/瓶颈/负载/"
                "承诺/历史模式）见 slo_tools/domain_metrics 的 "
                "completion_error_report"),
        }
