"""prediction_config.py -- 版本化预测配置：一个入口、正交研究轴与参数 preset。

W0 冻结（2026-09-29，《joint仓库当下+历史修改方案》§9.1/§9.2）。目标：

* **一个配置入口**：环境变量 ``JOINT_PREDICTION_CONFIG`` 指向一个版本化
  JSON 文件，承载全部预测轴与常数参数；不为每个常数新增环境变量
  （方案 §9.1 实施建议的落地，本模块为单一裁决点）。
* **正交研究轴**（§9.1 表）：预测模型 / 新增需求 / 历史启用 / 阶段策略 /
  结算策略相互独立，runner 校验不支持组合并记录
  生效值与来源（不能只记录用户输入而遗漏回退后的实际模式——回退由
  W7 forecast_validation 运行期执行，本模块只冻结取值域与组合约束）。
* **fail-closed**：未知键、未知版本、非法值、矛盾组合一律
  ``PredictionConfigError``；未知必需字段不允许缺省补零（§9.3）。

轴语义（§9.1 表逐条）：

* ``projection_mode``
  - ``known_work``（缺省）：正式路径 = 当前真实账本 → 候选加入后的资源
    依赖事件前推 → 因果历史补充 → 本轮 settle 完成时间（B 模式）。
  - ``frozen_diagnostic``：A 对照臂——共用新物理契约（同因果快照、真实
    目录、容量/保护条件、三动作与结算策略），但把决策 tick 的竞争份额/
    服务估计按显式登记的冻结规则外推，不随事件重分配。必须附带
    ``frozen_diagnostic`` 冻结规格（冻结哪些竞争集合/服务估计、冻结时
    域、资源规则），缺规格即启动失败。不能调用整套旧冻结计价实现。
* ``future_demand_mode``
  - ``none``（缺省）：B 模式不预测未来新增请求（明示条件近似，不是性
    能上下界声明）；已有请求的条件剩余长度估计仍存在。
  - ``recent``：C 臂——近期持续 ``lambda_fast`` 外推。
  - ``reverting``：D 臂——``lambda_slow + exp(-h/tau)*(lambda_fast -
    lambda_slow)`` 回归形式。tau 是独立预测参数，不宣称平台切换必然
    回归（§8.4）。
  - （``scenario`` 为条件增强，未实现时配置即拒绝，不用静默别名。）
* ``history_use``
  - ``shadow``（缺省）：历史统计可生成预测与成熟误差，但不影响正式
    选择（§8.7 shadow 门槛）。
  - ``validated``：仅当该预测距离分组已有独立因果验证记录时才允许配置；
    运行期由 forecast_validation 核验验证登记，无记录即回退 shadow 并
    披露（不在本模块静态裁决，本模块保证缺省安全）。
* ``stage_policy``
  - ``two_stage``（缺省）：P/D 两阶段闭环（D18 主语义）。
  - ``once_mapping``：一次映射对照臂 = 现有 ``boundary_policy=
    p-continuation`` 的同一机制（单一机制两个名字的统一裁决：本轴为
    正名，``JOINT_BOUNDARY_POLICY`` 是同一轴的既有 env 入口；两入口
    矛盾 fail-closed，见 ``reconcile_stage_policy``）。一次映射臂保留
    初始阶段方案，不承担虚构的第二次选优费用。
  - （``causal_lookahead`` 为条件增强，登记未实现。）
* ``settlement_policy``
  - ``consolidate``（缺省）：单位置收敛（物理 merge 属结算子协议）。
  - ``distributed``：目录提交/引用交接/暂存归还/旧义务闭合型结算，不
    强制物理归并。一次 run 固定，同次候选比较不能变（§7.3/§7.4）。

与 ``JointMechanismConfig`` 的组合约束（``cross_validate`` 单一裁决点）：

* J-off（scheduler_mode != joint）：预测轴中仅影响 J 计价的部分
  （projection_mode / future_demand_mode / history_use）记 inactive，
  不输出"历史已改善 J"式的生效声明；结算策略等共同底座轴
  仍然适用（§9.1"J-off 仍使用其映射策略，执行正确性、目录、结算和因
  果观测等共同底座仍适用"）。

§9.2 参数表：``PredictionParams`` 是可运行、可校验的具体 preset。所有
数值的来源在字段注释与 preset JSON 内登记；主文未指定通用最优窗口，
这些值是**按服务尺度推导并事前冻结的工程候选**（G-H 前不得按测试胜负
调整），不是从主文推导的必做实现（A09 裁决）。
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field, fields, replace
from types import MappingProxyType
from typing import Any, Mapping, Optional, Tuple

from joint.joint_config import MODEL_SEMANTICS_VERSION

#: 预测配置 schema 版本（与模型语义版本、快照 schema、bridge protocol、
#: 目录 schema、日志 schema 分开管理——§9.3）。语义改变必须升级本版本。
PREDICTION_CONFIG_SCHEMA_VERSION = "predcfg-v4-20261014"

#: 唯一配置入口环境变量（路径 → 版本化 JSON；W0 冻结命名，登记开关清单）。
ENV_JOINT_PREDICTION_CONFIG = "JOINT_PREDICTION_CONFIG"

# ---------------------------------------------------------------- 取值域 --
PROJECTION_MODE_KNOWN_WORK = "known_work"
PROJECTION_MODE_FROZEN_DIAGNOSTIC = "frozen_diagnostic"
PROJECTION_MODES = (
    PROJECTION_MODE_KNOWN_WORK,
    PROJECTION_MODE_FROZEN_DIAGNOSTIC,
)

FUTURE_DEMAND_NONE = "none"
FUTURE_DEMAND_RECENT = "recent"
FUTURE_DEMAND_REVERTING = "reverting"
FUTURE_DEMAND_MODES = (
    FUTURE_DEMAND_NONE,
    FUTURE_DEMAND_RECENT,
    FUTURE_DEMAND_REVERTING,
)

HISTORY_USE_SHADOW = "shadow"
HISTORY_USE_VALIDATED = "validated"
HISTORY_USE_MODES = (HISTORY_USE_SHADOW, HISTORY_USE_VALIDATED)
COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1 = "geometric-remainder-mean32-v1"

STAGE_POLICY_TWO_STAGE = "two_stage"
STAGE_POLICY_ONCE_MAPPING = "once_mapping"
STAGE_POLICIES = (STAGE_POLICY_TWO_STAGE, STAGE_POLICY_ONCE_MAPPING)

SETTLEMENT_CONSOLIDATE = "consolidate"
SETTLEMENT_DISTRIBUTED = "distributed"
SETTLEMENT_POLICIES = (SETTLEMENT_CONSOLIDATE, SETTLEMENT_DISTRIBUTED)

#: A 对照冻结规格的冻结时域取值域：``per_action`` = 把决策 tick 的活跃
#: 集合/份额冻结到整个候选动作完成（旧阶段冻结份额语义的显式登记形
#: 式）；``horizon_ns`` = 冻结到显式 ns 时域。A 臂共用新物理契约，仅
#: 替换"份额随事件重分配"这一预测假设（A02 裁决：A 是主文要求的诊断
#: 对照，不是新正式机制的隐藏回退）。
FROZEN_HORIZON_MODES = ("per_action", "horizon_ns")


class PredictionConfigError(ValueError):
    """fail-closed 预测配置错误（未知键/版本/值、矛盾组合、缺规格）。"""


# ------------------------------------------------------------ §9.2 参数表 --
# 数值来源登记（W0 冻结；G-H 前不得按测试胜负调整——A09/A20 裁决）：
#
# * 到达桶 Delta / 半衰期 / tau：按本仓服务尺度推导——单步 decode 约
#   ms 级、整轮 settle 约 10^2 ms 级（prefill 数千 token + 数十步
#   decode @ ms 级步长）、冒烟窗口 2 s；Delta=100 ms 可分辨该尺度内的
#   突发，Lf=1 s / Ls=30 s / tau=5 s 给出"轮内持续、跨轮回归"的可检验
#   候选。均为工程候选 preset，非主文指定常数。
# * 冷启动长度 32 forward steps：沿用现有 CausalHorizonEstimator 先验
#   （A.4 历史来源），保持对照一致。
# * 效率初值 1.0 / EWMA 1/16：沿用现有 ServiceFactors（A09：可保留的
#   预先固定实现方案；该系数是逐完成服务样本的效率更新权重，不得挪用
#   为到达强度的时间衰减——到达衰减由 Lf/Ls 经 alpha=1-2^(-Delta/L)
#   派生，两处来源不同、禁止配置不一致的 alpha）。
# * 控制延迟缺省 0 ns：**零值也须披露为假设**（§9.2 末行）——manifest
#   恒披露 control_latency_ns 及其 source 标签 "assumed-zero"。非零值
#   在执行侧定时提交事件实现前被配置门拒绝（§7.1"执行图/调度事件也
#   必须兑现该延迟"）。当前仅接受 0；登记标记不能替代真实延迟执行。
# * 预算：事件/闭包预算是超限行为显式化的上界（超限 → 已登记降级/
#   模型不可用，不允许截断成本冒充完成，§5.6）；宿主 wall time 与目标
#   控制时间默认分别报告，无映射模型。
# * W7 新增需求预测 preset（2026-09-29 登记）：forecast_horizon_ns=2s
#   按服务尺度推导（整轮 settle ~10^2 ms 级——覆盖当前轮+下一轮起点，
#   与 Delta=100 ms/Lf=1 s/Ls=30 s 同一推导族）；forecast_template_
#   budget=64 为有界事件预算工程候选（超限回退不截断）；forecast_
#   ordinal_threshold=1.0 为 §8.4 第一版"累计强度跨整数阈值生成完整
#   代表请求"的冻结值；forecast_distance_buckets_ns=(1×Δ, 4×Δ) 距离
#   桶界按 Δ=100 ms 推导；forecast_distance_tolerance=0.5 为预注册
#   归一化平均误差容差工程候选；forecast_context_tokens=2048 /
#   forecast_return_restore_tokens=128 为无 W6 统计来源的冷启动
#   preset（模板不确定标记携带，不冒充统计）。


@dataclass(frozen=True)
class FrozenDiagnosticSpec:
    """A 对照（projection_mode=frozen_diagnostic）的显式冻结规格。

    A 臂必须在 manifest 明确冻结哪些竞争集合/服务估计、冻结时域以及资
    源规则（§9.1）。缺任一必要字段即启动失败（fail-closed，不静默给
    默认冻结规则）。
    """

    #: 冻结时域模式：per_action（决策 tick 份额外推到候选完成）或
    #: horizon_ns（显式 ns 时域，配 frozen_horizon_ns > 0）。
    frozen_horizon_mode: str
    #: horizon_ns 模式下的显式时域（ns；per_action 模式必须为 None）。
    frozen_horizon_ns: Optional[int] = None
    #: 冻结的资源规则描述（机读摘要，登记进 manifest；例如
    #: "decision-tick active-set shares held fixed; HBM N-way shares,
    #: NoC per-link capacity/active-flow and pool port N-way from the
    #: shared physical contract"）。
    resource_rule: str = ""
    #: 冻结的服务估计描述（哪些服务估计不随事件更新）。
    service_estimate_rule: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.frozen_horizon_mode, str):
            raise PredictionConfigError("frozen_horizon_mode must be a string")
        if self.frozen_horizon_mode not in FROZEN_HORIZON_MODES:
            raise PredictionConfigError(
                f"frozen_horizon_mode must be one of {FROZEN_HORIZON_MODES},"
                f" got {self.frozen_horizon_mode!r}")
        if self.frozen_horizon_mode == "horizon_ns":
            if type(self.frozen_horizon_ns) is not int or (
                    self.frozen_horizon_ns <= 0):
                raise PredictionConfigError(
                    "frozen_horizon_mode=horizon_ns requires a positive "
                    f"integer frozen_horizon_ns, got {self.frozen_horizon_ns!r}")
        elif self.frozen_horizon_ns is not None:
            raise PredictionConfigError(
                "frozen_horizon_ns must be absent when "
                f"frozen_horizon_mode={self.frozen_horizon_mode!r}")
        if not isinstance(self.resource_rule, str) or not self.resource_rule.strip():
            raise PredictionConfigError(
                "frozen_diagnostic requires a non-empty resource_rule "
                "(which resource rules are frozen; manifest disclosure)")
        if (not isinstance(self.service_estimate_rule, str)
                or not self.service_estimate_rule.strip()):
            raise PredictionConfigError(
                "frozen_diagnostic requires a non-empty "
                "service_estimate_rule (which service estimates are frozen)")


@dataclass(frozen=True)
class PredictionParams:
    """§9.2 参数表的可运行 preset（字段级来源见模块 docstring 尾节）。"""

    # -- 到达统计（W6/W7 消费；单位统一 ns / request） --
    #: 到达统计时间桶 Delta（ns；正数，不得小于不可区分的观测粒度）。
    arrival_bucket_delta_ns: int = 100_000_000
    #: 快半衰期 Lf（ns；0 < Lf < Ls；alpha_f = 1-2^(-Delta/Lf) 派生，
    #: 不单独配置 alpha，杜绝同时配置不一致的 alpha——§9.2 表）。
    arrival_half_life_fast_ns: int = 1_000_000_000
    #: 慢半衰期 Ls（ns）。
    arrival_half_life_slow_ns: int = 30_000_000_000
    #: 回归时间尺度 tau（ns；独立预测参数；reverting 臂使用）。
    revert_tau_ns: int = 5_000_000_000
    # -- 未知长度（W6 消费） --
    #: 冷启动条件剩余长度先验（forward steps；沿用现有 32 步先验来源）。
    cold_start_forward_steps: int = 32
    #: 冷启动尾部分布 ID + 版本（冻结的工程候选：均值 32 的几何尾；
    #: 区分总长/剩余口径——先验给的是条件剩余步数分布，非总长）。
    cold_start_tail_distribution: str = COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1
    #: 剩余长度直方图桶上限（有界内存；溢出归并到尾桶，不截尾冒充短样本）。
    length_histogram_max_buckets: int = 256
    #: 直方图桶宽（forward steps）。
    length_histogram_bucket_steps: int = 8
    # -- 服务效率（W6 消费；与现有 ServiceFactors 同源） --
    #: 无争用效率初值（沿用 1.0）。
    service_efficiency_init: float = 1.0
    #: 逐完成纯服务样本的 EWMA 权重（沿用 1/16；**不得**挪用为到达衰减）。
    service_efficiency_ewma_weight: float = 1.0 / 16.0
    # -- 历史准入/成熟（W7 消费） --
    #: 最小成熟样本数（低于此保持 shadow/回退，不更新准入判定）。
    history_min_mature_samples: int = 16
    #: 预测样本有效期（ns；过期样本不进入在线统计）。
    history_sample_validity_ns: int = 600_000_000_000
    #: 分布变化（epoch）后的降级观察窗（ns；窗内不启用 validated）。
    history_drift_quarantine_ns: int = 2_000_000_000_000
    # -- 情景（条件增强；未启用也声明确定性规则——§9.2 表） --
    #: 情景数（1 = 确定性背景；>1 需要 scenario 模式已实现）。
    scenario_budget: int = 1
    #: 情景随机种子（候选间一致；不按输赢临时增加）。
    scenario_seed: int = 0
    # -- 前推预算（W2/W3 消费；超限 → 显式降级/模型不可用，不截断） --
    #: 单候选前推事件数预算。
    projection_event_budget: int = 262_144
    #: 加速窗口最大 tile 数（run contract 同时冻结该上限；精确调用可
    #: 在单候选 options 中收紧为 1，不改变全局配置摘要）。发布默认
    #: = F 链验收档（2026-10-09 tune 平局带选出 + holdout 123 例独立
    #: 验证通过，gates/G-F0 done 4fa53baf；approximation_accepted=True）。
    window_tile_limit: int = 2
    #: 局部前缀误差预算，整数 ppm（同上验收档）。
    window_local_error_ppm: int = 0
    #: 候选精确细化敏感带，整数 ppm。
    window_refine_margin_ppm: int = 20_000
    #: 竞争闭包资源数预算。
    closure_resource_budget: int = 512
    #: 每决策宿主 wall time 预算（秒；仅报告与告警，不自动截断——预算
    #: 来源须按 §9.2 显式声明，缺省为报告型）。
    host_decision_budget_seconds: float = 0.0
    # -- 控制与交接（W5 消费；零值也披露为假设） --
    #: 第二次共同评估的目标系统控制延迟（ns；缺省 0 = assumed-zero 假设，
    #: manifest 恒披露；执行图/调度事件必须兑现非零值——§7.1）。
    control_latency_ns: int = 0
    #: 变更方案新增交接的控制/协议工作描述（机读摘要；缺省空 = 未建模，
    #: 显式登记）。
    handoff_control_rule: str = ""
    # -- W7 新增需求预测（消费方 joint/background_forecast.py 与
    #    joint/forecast_validation.py；§9.2 表"桶/直方图/模板上限"与
    #    "历史准入的预测距离桶/容差/开销预算"两行的落地。数值来源登记
    #    见模块 docstring 尾节 W7 段；G-H 前不得按测试胜负调整） --
    #: 预测时域 H（ns；决策时点向未来展开背景到达的窗口）。
    forecast_horizon_ns: int = 2_000_000_000
    #: 单决策注入背景模板数上限（超限 → budget_exceeded 回退整档，
    #: 不截断冒充——§8.5"预测超预算"行）。
    forecast_template_budget: int = 64
    #: 确定性注入序号阈值 θ（累计预测强度每跨过一个 θ 整数倍生成一份
    #: 完整代表请求；候选间一致——§8.4 第一版冻结）。
    forecast_ordinal_threshold: float = 1.0
    #: 预测距离桶界（ns，严格递增正数）：偏移 < b0 = near、< b1 = mid、
    #: 其余 far（按距离准入分组的预注册分组，§8.7）。离线对账钉测：
    #: slo_tools/tests/test_domain_metrics.py::
    #: OnlineOfflineLedgerReconciliationTests（前两档与离线 §10.3 实距
    #: 档位锁等；语义轴不同——在线按预测 offset、离线按 realized 决策
    #: →settle 跨度且多 beyond 档——任一侧单改即在该钉测报错）。
    forecast_distance_buckets_ns: Tuple[int, ...] = (
        100_000_000, 400_000_000)
    #: validated 准入容差：|mean((predicted−actual)/max(1,predicted))| ≤
    #: tol 且成熟样本 ≥ history_min_mature_samples 才判该 (mode, band)
    #: 已验证（预注册误差定义，§8.7/§9.2）。
    forecast_distance_tolerance: float = 0.5
    #: 模板 prefill 上下文代表值（token）。冷启动 preset：W6 无类别级
    #: prefill 上下文统计来源（到达时输入长度已揭示但 W6 未入账）——
    #: 登记为不确定 preset，不冒充统计。
    forecast_context_tokens: int = 2048
    #: 会话返回冷层恢复代表 token 数/TP rank（同上：无统计来源的
    #: preset，模板参数不确定标记携带）。
    forecast_return_restore_tokens: int = 128

    def __post_init__(self) -> None:
        # Validate units before consumers convert values to int or use them
        # as divisors. Booleans, fractional counts and non-finite values cannot
        # represent the declared physical parameters.
        for spec in fields(self):
            value = getattr(self, spec.name)
            if spec.type == "int" and type(value) is not int:
                raise PredictionConfigError(f"{spec.name} must be an integer")
            if spec.type == "float" and (
                    type(value) not in (int, float) or not math.isfinite(value)):
                raise PredictionConfigError(f"{spec.name} must be a finite number")
            if spec.type == "bool" and type(value) is not bool:
                raise PredictionConfigError(f"{spec.name} must be a boolean")
            if spec.type == "str" and not isinstance(value, str):
                raise PredictionConfigError(f"{spec.name} must be a string")
        if self.arrival_bucket_delta_ns <= 0:
            raise PredictionConfigError(
                "arrival_bucket_delta_ns must be positive, got "
                f"{self.arrival_bucket_delta_ns}")
        if not (0 < self.arrival_half_life_fast_ns
                < self.arrival_half_life_slow_ns):
            raise PredictionConfigError(
                "arrival half-lives must satisfy 0 < Lf < Ls, got "
                f"Lf={self.arrival_half_life_fast_ns}, "
                f"Ls={self.arrival_half_life_slow_ns}")
        if self.revert_tau_ns <= 0:
            raise PredictionConfigError(
                f"revert_tau_ns must be positive, got {self.revert_tau_ns}")
        if self.cold_start_forward_steps <= 0:
            raise PredictionConfigError(
                "cold_start_forward_steps must be positive, got "
                f"{self.cold_start_forward_steps}")
        if self.cold_start_tail_distribution != COLD_START_TAIL_GEOMETRIC_REMAINDER_MEAN32_V1:
            raise PredictionConfigError(
                "cold_start_tail_distribution must name the implemented geometric prior")
        if self.length_histogram_max_buckets <= 0 or (
                self.length_histogram_bucket_steps <= 0):
            raise PredictionConfigError(
                "length histogram bounds must be positive, got buckets="
                f"{self.length_histogram_max_buckets}, width="
                f"{self.length_histogram_bucket_steps}")
        if not (0.0 < self.service_efficiency_init <= 1.0):
            raise PredictionConfigError(
                "service_efficiency_init must be in (0, 1], got "
                f"{self.service_efficiency_init}")
        if not (0.0 < self.service_efficiency_ewma_weight <= 1.0):
            raise PredictionConfigError(
                "service_efficiency_ewma_weight must be in (0, 1], got "
                f"{self.service_efficiency_ewma_weight}")
        if self.history_min_mature_samples <= 0:
            raise PredictionConfigError(
                "history_min_mature_samples must be positive, got "
                f"{self.history_min_mature_samples}")
        if self.history_sample_validity_ns <= 0 or (
                self.history_drift_quarantine_ns <= 0):
            raise PredictionConfigError(
                "history validity/quarantine windows must be positive")
        if self.scenario_budget != 1:
            # W11 死配置面收口（2026-09-29）：scenario 模式属条件增强
            # 未实现——>1 曾被接受但零消费（静默死值），改为配置即拒
            # （与 future_demand_mode=scenario 同一 fail-closed 姿态）。
            raise PredictionConfigError(
                "scenario_budget must be 1 (deterministic background); "
                "scenario mode is an unimplemented conditional "
                "enhancement and configuration is rejected, got "
                f"{self.scenario_budget}")
        if self.projection_event_budget <= 0 or (
                self.closure_resource_budget <= 0):
            raise PredictionConfigError(
                "projection budgets must be positive")
        if self.window_tile_limit not in (1, 2, 4, 8):
            raise PredictionConfigError(
                "window_tile_limit must be one of (1, 2, 4, 8)")
        if not 0 <= self.window_local_error_ppm <= 10_000:
            raise PredictionConfigError(
                "window_local_error_ppm must be an integer in [0,10000]")
        if not 0 <= self.window_refine_margin_ppm <= 1_000_000:
            raise PredictionConfigError(
                "window_refine_margin_ppm must be an integer in [0,1000000]")
        if self.host_decision_budget_seconds < 0:
            raise PredictionConfigError(
                "host_decision_budget_seconds must be non-negative "
                "(0 = report-only), got "
                f"{self.host_decision_budget_seconds}")
        if self.control_latency_ns < 0:
            raise PredictionConfigError(
                "control_latency_ns must be non-negative (0 = disclosed "
                f"assumed-zero), got {self.control_latency_ns}")
        if self.control_latency_ns > 0:
            raise PredictionConfigError(
                "control_latency_ns>0 requires the execution graph/"
                "scheduling events to honor the delay (sec.7.1); the "
                "execution side does not yet implement deferred commit "
                "events, so a non-zero value cannot be accepted. Use 0 "
                f"(assumed-zero), got control_latency_ns="
                f"{self.control_latency_ns}")
        # -- W7 新增需求预测 preset 校验（fail-closed）--
        if self.forecast_horizon_ns <= 0:
            raise PredictionConfigError(
                "forecast_horizon_ns must be positive, got "
                f"{self.forecast_horizon_ns}")
        if self.forecast_template_budget <= 0:
            raise PredictionConfigError(
                "forecast_template_budget must be positive, got "
                f"{self.forecast_template_budget}")
        if not (self.forecast_ordinal_threshold > 0.0):
            raise PredictionConfigError(
                "forecast_ordinal_threshold must be positive, got "
                f"{self.forecast_ordinal_threshold}")
        if not isinstance(self.forecast_distance_buckets_ns, (tuple, list)):
            raise PredictionConfigError("forecast_distance_buckets_ns must be a sequence")
        bounds = tuple(self.forecast_distance_buckets_ns)
        if len(bounds) != 2 or any(
                (type(b) is not int or b <= 0) for b in bounds):
            raise PredictionConfigError(
                "forecast_distance_buckets_ns requires two positive integer "
                f"boundaries for near/mid/far, got {bounds!r}")
        if any(bounds[i] >= bounds[i + 1] for i in range(len(bounds) - 1)):
            raise PredictionConfigError(
                "forecast_distance_buckets_ns must be strictly increasing,"
                f" got {bounds!r}")
        if not (self.forecast_distance_tolerance > 0.0):
            raise PredictionConfigError(
                "forecast_distance_tolerance must be positive, got "
                f"{self.forecast_distance_tolerance}")
        if self.forecast_context_tokens <= 0:
            raise PredictionConfigError(
                "forecast_context_tokens must be positive, got "
                f"{self.forecast_context_tokens}")
        if self.forecast_return_restore_tokens < 0:
            raise PredictionConfigError(
                "forecast_return_restore_tokens must be >= 0, got "
                f"{self.forecast_return_restore_tokens}")

    @property
    def arrival_alpha_fast(self) -> float:
        """快桶 EWMA 系数 alpha_f = 1 - 2^(-Delta/Lf)（派生，不配置）。"""
        return 1.0 - 2.0 ** (
            -self.arrival_bucket_delta_ns / self.arrival_half_life_fast_ns)

    @property
    def arrival_alpha_slow(self) -> float:
        """慢桶 EWMA 系数 alpha_s = 1 - 2^(-Delta/Ls)（派生，不配置）。"""
        return 1.0 - 2.0 ** (
            -self.arrival_bucket_delta_ns / self.arrival_half_life_slow_ns)

    def params_manifest(self) -> dict[str, Any]:
        """参数 preset 的 manifest 披露（含派生 alpha 与来源标签）。"""
        return {
            "arrival_bucket_delta_ns": self.arrival_bucket_delta_ns,
            "arrival_half_life_fast_ns": self.arrival_half_life_fast_ns,
            "arrival_half_life_slow_ns": self.arrival_half_life_slow_ns,
            "arrival_alpha_fast_derived": self.arrival_alpha_fast,
            "arrival_alpha_slow_derived": self.arrival_alpha_slow,
            "revert_tau_ns": self.revert_tau_ns,
            "cold_start_forward_steps": self.cold_start_forward_steps,
            "cold_start_tail_distribution": self.cold_start_tail_distribution,
            "length_histogram_max_buckets":
                self.length_histogram_max_buckets,
            "length_histogram_bucket_steps":
                self.length_histogram_bucket_steps,
            "service_efficiency_init": self.service_efficiency_init,
            "service_efficiency_ewma_weight":
                self.service_efficiency_ewma_weight,
            "history_min_mature_samples": self.history_min_mature_samples,
            "history_sample_validity_ns": self.history_sample_validity_ns,
            "history_drift_quarantine_ns": self.history_drift_quarantine_ns,
            "scenario_budget": self.scenario_budget,
            "scenario_seed": self.scenario_seed,
            "projection_event_budget": self.projection_event_budget,
            "window_tile_limit": self.window_tile_limit,
            "window_local_error_ppm": self.window_local_error_ppm,
            "window_refine_margin_ppm": self.window_refine_margin_ppm,
            "closure_resource_budget": self.closure_resource_budget,
            "host_decision_budget_seconds":
                self.host_decision_budget_seconds,
            "control_latency_ns": self.control_latency_ns,
            "control_latency_source": "assumed-zero",
            "handoff_control_rule": self.handoff_control_rule or None,
            "forecast_horizon_ns": self.forecast_horizon_ns,
            "forecast_template_budget": self.forecast_template_budget,
            "forecast_ordinal_threshold":
                self.forecast_ordinal_threshold,
            "forecast_distance_buckets_ns":
                list(self.forecast_distance_buckets_ns),
            "forecast_distance_tolerance":
                self.forecast_distance_tolerance,
            "forecast_context_tokens": self.forecast_context_tokens,
            "forecast_return_restore_tokens":
                self.forecast_return_restore_tokens,
        }


# ---------------------------------------------------------------- 主配置 --
def _freeze_validation_value(value):
    if isinstance(value, Mapping):
        if any(not isinstance(key, str) for key in value):
            raise PredictionConfigError("validation evidence keys must be strings")
        return MappingProxyType({key: _freeze_validation_value(item)
                                 for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_validation_value(item) for item in value)
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float and math.isfinite(value):
        return value
    raise PredictionConfigError("validation evidence must contain finite JSON values")


def _validation_manifest_value(value):
    if isinstance(value, Mapping):
        return {key: _validation_manifest_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_validation_manifest_value(item) for item in value]
    return value


@dataclass(frozen=True)
class JointPredictionConfig:
    """已解析的预测配置（五正交轴 + 参数 preset + A 臂冻结规格）。

    直接构造用于测试；运行期经 ``load_prediction_config``（唯一 env 入口
    ``JOINT_PREDICTION_CONFIG``）或 ``from_mapping``（显式字典）获得。
    ``sources`` 记录每个生效值的来源（env 路径 / 文件路径 / 缺省），
    manifest 披露回退后的实际模式（§9.1）。
    """

    projection_mode: str = PROJECTION_MODE_KNOWN_WORK
    future_demand_mode: str = FUTURE_DEMAND_NONE
    history_use: str = HISTORY_USE_SHADOW
    stage_policy: str = STAGE_POLICY_TWO_STAGE
    settlement_policy: str = SETTLEMENT_CONSOLIDATE
    params: PredictionParams = field(default_factory=PredictionParams)
    frozen_diagnostic: Optional[FrozenDiagnosticSpec] = None
    # 独立验证前缀的登记，键为 mode|distance_band；不由当前 run 的
    # 在线误差自动填充。质量门在 forecast_validation.py 统一裁决。
    validation_registry: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.params, PredictionParams):
            raise PredictionConfigError("params must be PredictionParams")
        if self.frozen_diagnostic is not None and not isinstance(
                self.frozen_diagnostic, FrozenDiagnosticSpec):
            raise PredictionConfigError("frozen_diagnostic must be FrozenDiagnosticSpec")
        if not isinstance(self.validation_registry, Mapping):
            raise PredictionConfigError("validation_registry must be a mapping")
        registry = {}
        allowed_record_fields = {
            "samples", "mean_abs_normalized_error", "selection_loss_samples",
            "mean_selection_loss_ns", "baseline_mean_selection_loss_ns",
            "validation_source", "validation_scope", "model_semantics_version",
            "params",
        }
        for key, record in self.validation_registry.items():
            if key not in {f"{mode}|{band}" for mode in ("recent", "reverting")
                           for band in ("near", "mid", "far")}:
                raise PredictionConfigError(f"invalid validation_registry key {key!r}")
            if not isinstance(record, Mapping) or set(record) - allowed_record_fields:
                raise PredictionConfigError(f"invalid validation_registry record {key!r}")
            copied = dict(record)
            for name, value in copied.items():
                if name in ("samples", "selection_loss_samples"):
                    if type(value) is not int or value < 0:
                        raise PredictionConfigError(f"validation {name} must be a non-negative integer")
                elif name in ("mean_abs_normalized_error", "mean_selection_loss_ns",
                              "baseline_mean_selection_loss_ns"):
                    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
                        raise PredictionConfigError(f"validation {name} must be finite and non-negative")
                elif name != "params" and not isinstance(value, str):
                    raise PredictionConfigError(f"validation {name} must be a string")
            if "params" in copied:
                if not isinstance(copied["params"], Mapping):
                    raise PredictionConfigError("validation params must be a mapping")
            registry[key] = _freeze_validation_value(copied)
        object.__setattr__(self, "validation_registry", MappingProxyType(registry))
        if self.projection_mode not in PROJECTION_MODES:
            raise PredictionConfigError(
                f"projection_mode must be one of {PROJECTION_MODES}, got "
                f"{self.projection_mode!r}")
        if self.projection_mode == PROJECTION_MODE_FROZEN_DIAGNOSTIC and (
                self.frozen_diagnostic is None):
            raise PredictionConfigError(
                "projection_mode=frozen_diagnostic requires an explicit "
                "frozen_diagnostic spec (frozen competition sets, horizon "
                "and resource rules); refusing to start with an implicit "
                "default freeze")
        if (self.projection_mode != PROJECTION_MODE_FROZEN_DIAGNOSTIC
                and self.frozen_diagnostic is not None):
            raise PredictionConfigError(
                "frozen_diagnostic spec is only valid with "
                f"projection_mode={PROJECTION_MODE_FROZEN_DIAGNOSTIC!r}")
        if self.future_demand_mode not in FUTURE_DEMAND_MODES:
            raise PredictionConfigError(
                f"future_demand_mode must be one of {FUTURE_DEMAND_MODES} "
                f"(scenario is a conditional enhancement, not yet "
                f"implemented), got {self.future_demand_mode!r}")
        if self.history_use not in HISTORY_USE_MODES:
            raise PredictionConfigError(
                f"history_use must be one of {HISTORY_USE_MODES}, got "
                f"{self.history_use!r}")
        if self.stage_policy not in STAGE_POLICIES:
            raise PredictionConfigError(
                f"stage_policy must be one of {STAGE_POLICIES} "
                f"(causal_lookahead is a conditional enhancement), got "
                f"{self.stage_policy!r}")
        if self.settlement_policy not in SETTLEMENT_POLICIES:
            raise PredictionConfigError(
                f"settlement_policy must be one of {SETTLEMENT_POLICIES}, "
                f"got {self.settlement_policy!r}")
        # 组合约束（§9.1）：新增需求预测依赖历史统计。静态层不做额外
        # 门——history_use 已在上方强制属于 HISTORY_USE_MODES（缺省与
        # 显式同值，此层无从判别"显式"），recent/reverting 对历史轴的
        # 完整性要求由 W7 运行期核验登记。

    # ------------------------------------------------------------ 派生 --
    def j_pricing_axes_active(self, scheduler_mode: str) -> bool:
        """预测轴是否实际影响 J 正式计价（J-off 时 False）。"""
        return scheduler_mode == "joint"

    def manifest_dict(self) -> dict[str, Any]:
        """预测配置的 manifest 披露节（含 schema 版本与来源）。"""
        return {
            "prediction_config_schema_version":
                PREDICTION_CONFIG_SCHEMA_VERSION,
            "projection_mode": self.projection_mode,
            "future_demand_mode": self.future_demand_mode,
            "history_use": self.history_use,
            "stage_policy": self.stage_policy,
            "settlement_policy": self.settlement_policy,
            "frozen_diagnostic": (
                {
                    "frozen_horizon_mode":
                        self.frozen_diagnostic.frozen_horizon_mode,
                    "frozen_horizon_ns":
                        self.frozen_diagnostic.frozen_horizon_ns,
                    "resource_rule": self.frozen_diagnostic.resource_rule,
                    "service_estimate_rule":
                        self.frozen_diagnostic.service_estimate_rule,
                }
                if self.frozen_diagnostic is not None else None),
            "params": self.params.params_manifest(),
            "validation_registry": {
                key: _validation_manifest_value(record)
                for key, record in self.validation_registry.items()},
            "sources": dict(self.sources),
        }

    def semantic_dict(self) -> dict[str, Any]:
        """Normalized complete semantic configuration, excluding provenance.

        Unlike ``manifest_dict`` this includes every configured value exactly
        once and omits source paths/derived disclosures. It is the stable
        digest input shared by Python and the execution contract.
        """
        frozen = None
        if self.frozen_diagnostic is not None:
            frozen = {
                name: getattr(self.frozen_diagnostic, name)
                for name in self.frozen_diagnostic.__dataclass_fields__}
        return {
            "version": PREDICTION_CONFIG_SCHEMA_VERSION,
            "projection_mode": self.projection_mode,
            "future_demand_mode": self.future_demand_mode,
            "history_use": self.history_use,
            "stage_policy": self.stage_policy,
            "settlement_policy": self.settlement_policy,
            "params": {
                name: getattr(self.params, name)
                for name in self.params.__dataclass_fields__},
            "frozen_diagnostic": frozen,
            "validation_registry": {
                key: _validation_manifest_value(value)
                for key, value in self.validation_registry.items()},
        }

    def with_sources(self, sources: Mapping[str, str]) -> "JointPredictionConfig":
        return replace(self, sources=dict(sources))


# ------------------------------------------------------- stage_policy 合一 --
def reconcile_stage_policy(
    boundary_policy: str,
    stage_policy: str,
) -> str:
    """``stage_policy`` 与既有 ``boundary_policy`` 的单一合一裁决点。

    两名字同一机制（once_mapping == p-continuation、two_stage == off）。
    返回合一后的 ``stage_policy``；矛盾组合 fail-closed：

    * boundary=p-continuation 而 stage=two_stage：一次映射 env 已把边界
      固定为延续，stage 轴再声称两阶段 = 二义，拒绝；
    * boundary=off 而 stage=once_mapping：同理拒绝（要一次映射应设
      boundary 或由预测配置一致表达）。
    """
    env_once = boundary_policy == "p-continuation"
    file_once = stage_policy == STAGE_POLICY_ONCE_MAPPING
    if env_once != file_once:
        raise PredictionConfigError(
            f"stage_policy={stage_policy!r} conflicts with "
            f"boundary_policy={boundary_policy!r}: they are the same axis "
            "(once_mapping == p-continuation); specify them consistently")
    return stage_policy


def cross_validate(
    mech_config: Any,
    prediction: JointPredictionConfig,
) -> dict[str, Any]:
    """与 ``JointMechanismConfig`` 的组合校验（调度器装配处调用）。

    返回组合披露节（inactive 轴清单等）；矛盾组合 fail-closed。J-off 时
    预测轴记 inactive 而不是拒绝（§9.1：共同底座轴仍适用）。
    """
    scheduler_mode = getattr(mech_config, "scheduler_mode", None)
    if scheduler_mode is None:
        raise PredictionConfigError(
            "cross_validate expects a JointMechanismConfig-like object "
            f"with scheduler_mode, got {type(mech_config).__name__}")
    reconcile_stage_policy(
        getattr(mech_config, "boundary_policy", "off"),
        prediction.stage_policy)
    j_active = prediction.j_pricing_axes_active(scheduler_mode)
    disclosure: dict[str, Any] = {
        "j_pricing_active": j_active,
        "inactive_axes_under_j_off": (
            [] if j_active else
            ["projection_mode", "future_demand_mode", "history_use"]),
    }
    if not j_active:
        disclosure["inactive_note"] = (
            "scheduler_mode != joint: predictor-ablation axes are "
            "recorded but inactive; settlement/common-base "
            "axes still apply")
    return disclosure


# ------------------------------------------------------------------ 加载 --
def from_mapping(payload: Mapping[str, Any]) -> JointPredictionConfig:
    """从显式字典构造（文件加载与测试共用；未知键 fail-closed）。"""
    known_top = {
        "version", "projection_mode", "future_demand_mode", "history_use",
        "stage_policy", "settlement_policy",
        "params", "frozen_diagnostic", "validation_registry",
    }
    unknown = set(payload) - known_top
    if unknown:
        raise PredictionConfigError(
            f"unknown prediction config keys: {sorted(unknown)}; known "
            f"keys: {sorted(known_top)}")
    version = payload.get("version")
    if version != PREDICTION_CONFIG_SCHEMA_VERSION:
        raise PredictionConfigError(
            f"prediction config version mismatch: file declares "
            f"{version!r}, runtime requires "
            f"{PREDICTION_CONFIG_SCHEMA_VERSION!r} (fail-closed; no "
            "auto-migration)")
    params_payload = payload.get("params", {})
    if not isinstance(params_payload, Mapping):
        raise PredictionConfigError(
            f"params must be a mapping, got {type(params_payload).__name__}")
    params_fields = {
        f for f in PredictionParams.__dataclass_fields__}
    unknown_params = set(params_payload) - params_fields
    if unknown_params:
        raise PredictionConfigError(
            f"unknown prediction params keys: {sorted(unknown_params)}")
    params = PredictionParams(**params_payload)
    frozen_payload = payload.get("frozen_diagnostic")
    # (2026-10-01 审查修复) 子字典键域先校验再构造：裸 **dict() 对未知
    # 键抛 TypeError，逃出模块声明的 PredictionConfigError fail-closed
    # 单一通道（与顶层/params 层同款守卫）。
    frozen = None
    if frozen_payload is not None:
        if not isinstance(frozen_payload, dict):
            raise PredictionConfigError(
                "frozen_diagnostic must be a mapping (got "
                f"{type(frozen_payload).__name__})")
        unknown_frozen = set(frozen_payload) - {
            field_name for field_name in FrozenDiagnosticSpec.__dataclass_fields__}
        if unknown_frozen:
            raise PredictionConfigError(
                f"unknown frozen_diagnostic keys: {sorted(unknown_frozen)}")
        if "frozen_horizon_mode" not in frozen_payload:
            raise PredictionConfigError("frozen_diagnostic requires frozen_horizon_mode")
        frozen = FrozenDiagnosticSpec(**dict(frozen_payload))
    return JointPredictionConfig(
        projection_mode=payload.get(
            "projection_mode", PROJECTION_MODE_KNOWN_WORK),
        future_demand_mode=payload.get(
            "future_demand_mode", FUTURE_DEMAND_NONE),
        history_use=payload.get("history_use", HISTORY_USE_SHADOW),
        stage_policy=payload.get(
            "stage_policy", STAGE_POLICY_TWO_STAGE),
        settlement_policy=payload.get(
            "settlement_policy", SETTLEMENT_CONSOLIDATE),
        params=params,
        frozen_diagnostic=frozen,
        validation_registry=payload.get("validation_registry", {}),
    )


def load_prediction_config(
    env: Optional[Mapping[str, str]] = None,
) -> JointPredictionConfig:
    """唯一 env 入口 ``JOINT_PREDICTION_CONFIG`` 的加载裁决点。

    * 未设置：返回缺省配置（known_work / none / shadow / two_stage /
      consolidate + 缺省 preset），来源记 "default"；
    * 设置：读取 JSON 文件，``from_mapping`` 解析（版本/未知键/值域
      fail-closed），来源记文件路径。
    """
    lookup = os.environ if env is None else env
    raw = lookup.get(ENV_JOINT_PREDICTION_CONFIG)
    if raw is None or raw == "":
        return JointPredictionConfig().with_sources({"*": "default"})
    path = raw.strip()
    if path != raw or not path:
        raise PredictionConfigError(
            f"{ENV_JOINT_PREDICTION_CONFIG} must be a non-empty path "
            f"without surrounding whitespace, got {raw!r}")
    try:
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle, object_pairs_hook=_unique_json_object)
    except OSError as exc:
        raise PredictionConfigError(
            f"{ENV_JOINT_PREDICTION_CONFIG} path {path!r} cannot be read: "
            f"{exc}") from exc
    except json.JSONDecodeError as exc:
        raise PredictionConfigError(
            f"{ENV_JOINT_PREDICTION_CONFIG} path {path!r} is not valid "
            f"JSON: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise PredictionConfigError(
            f"{ENV_JOINT_PREDICTION_CONFIG} path {path!r} must contain a "
            f"JSON object, got {type(payload).__name__}")
    config = from_mapping(payload)
    return config.with_sources({"*": path})


def _unique_json_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise PredictionConfigError(f"duplicate prediction JSON key {key!r}")
        result[key] = value
    return result


def require_prediction_config(value: Any) -> JointPredictionConfig:
    """类型守卫（调度器装配处 fail-closed）。"""
    if not isinstance(value, JointPredictionConfig):
        raise PredictionConfigError(
            f"expected JointPredictionConfig, got {type(value).__name__}")
    return value


def make_pipeline_kernel_contract(
        config: JointPredictionConfig) -> dict[str, Any]:
    """Build the immutable schema-2 GraphBatch kernel contract.

    ``prediction_config_sha256`` hashes normalized parsed semantic values,
    not the original file bytes or provenance path. This keeps run copies
    equivalent while making defaults part of the contract.
    """
    import hashlib

    config = require_prediction_config(config)
    semantic = json.dumps(
        config.semantic_dict(), sort_keys=True, separators=(",", ":"),
        ensure_ascii=False, allow_nan=False)
    params = config.params
    return {
        "projection_abi_version": 1,
        "descriptor_version": 1,
        "pipeline_protocol_version": 1,
        # 模型语义版本随 joint_config 单一定义源派生（2026-10-14 远读
        # 退役升 v7；C++ 侧 PipelineKernelContract::kModelVersion 同串）。
        "model_version": MODEL_SEMANTICS_VERSION,
        "timer_whitelist_version": "timer-whitelist-v1-20261005",
        "window_tile_limit": params.window_tile_limit,
        "window_local_error_ppm": params.window_local_error_ppm,
        "window_refine_margin_ppm": params.window_refine_margin_ppm,
        "prediction_config_sha256": hashlib.sha256(
            semantic.encode("utf-8")).hexdigest(),
    }
