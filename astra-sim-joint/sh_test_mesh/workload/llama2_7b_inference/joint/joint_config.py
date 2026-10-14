"""joint_config.py -- 三机制独立开关、八组合消融预设与运行 manifest。

设计依据：《三机制联合策略_template仓库设计方案》§7.1（2026-09-13 裁定的
八组合固定映射）与实验总纲第四节。硬性要求：T/J/E 三机制在同一套共同实现
内独立开/关，运行期只通过配置切换，不维护分叉代码；每个开关只改变其指定
分支，KV 物理布局、新 KV 写回/合并语义、硬件配置、初始状态、在线估计与
更新规则、动作计费在八组合间严格一致。

开关语义（off 的共同替代）：

* category_mode（T 开关）
  - ``typed``：SH 严格类别逐出顺序——同一待释放资源的合法 victim 中先
    human 类后 tool 类，类内沿用 ``(last_completion_ns, session_id)``。
  - ``lru``  ：T-off——同一合法 victim 集合上的类型无关 LRU；层数策略
    （E 模式）、保护规则、调度与合并流程不变。
* scheduler_mode（J 开关）
  - ``joint``        ：全 instance 候选，联合选择 instance ×
    stay/recompute/copy。
  - ``load-first``   ：J-off 顺序参照（八组合固定参照，2026-09-13 主控
    裁定）——先按在线负载排序选 instance，再在该位置选择 KV 动作。
  - ``affinity-first``：J 边际比较的第二顺序参照（有 home 时选 home，
    否则回退 load-first）——不进八组合，供 T+J+E vs T+E、T+J vs T 的
    边际比较并列运行。
  - ``face_static``：**静态距离对照臂**（C17，policy variant——不进
    八组合、非 J 主臂；joint 主臂永不掩码）。实例按 ``hop ≤
    floor(rho)`` 静态距离球掩码（rho = B_D2D/B_HBM 逐配置硬件派生，
    base 4050/1640 → 2；rho < 1 退化仅锚实例，如实记录；掩码半径
    **无** ``max(1, .)`` 保底——结构性防死锁下限是配额 Q_init 专有
    语义，静态球如实退化），球内逐实例择优（见 joint_scheduler）。
    **身份纪律（设计文档 §5.3/§1.3）**：本臂是本文内部的静态距离
    变体，命名与 manifest 必须明确这一身份，**不得冒称完整 FACE**；
    在完成路由、执行、共享和端点假设对齐推导前，不宣称静态距离球
    是本文动态供给模型的零负载特例（身份声明落 manifest
    ``face_static_identity`` 键与 joint_scheduler 模块 docstring）。
    **quota 强制耦合（F7）**：``face_static`` 强制 ``quota_mode=off``
    ——env 配了非 off 值时显式覆盖为 off 并注记 ``face_static forces
    quota-off``（manifest ``quota_forced_off``/``quota_forced_off_note``
    键），**不 fail-closed 拒绝启动**（对照臂语义：量静态掩码自身的
    两侧误差，不被配额准入层混杂；quota-on 静态变体如需另行预登记）。
* layer_policy（E 开关）
  - ``adaptive``           ：E-on——按层消费期限计算最小热前缀 k_hide 的
    软目标式逐出（§5）。
  - ``minimal_layer_groups``：E-off——同一合法对象排序下按逐 rank 缺口
    释放最少完整层组，不计算隐藏目标。
  - ``legacy_half``        ：原半层逻辑的隔离适配，仅作回归/辅助对照。
* allowed_actions（M2 动作掩码，2026-09-26 消融矩阵前置；与三机制/
  quota 开关正交记录）
  - ``all``（缺省）：无剔除——三动作全集参与 J 比较；
  - 逗号分隔的 ``ALL_ACTIONS`` 子集（如 ``recompute``、
    ``copy,recompute``）：命中掩码外且原本可用的动作在
    ``applicable_actions`` 尾部统一剔除（拒绝理由 ``action excluded
    by JOINT_ALLOWED_ACTIONS mask``）。**语义裁定：掩码是候选集上限
    不是强制**；臂预设恒含 recompute（recompute-only=``{recompute}``
    +load-first；always-copy=``{copy,recompute}``）；掩码剔除
    recompute 致零适用时维持 fail-closed（joint_scheduler 的
    no-applicable-candidate raise，非容量拒绝）。
* quota_mode（通用流量治理/配额开关，与三机制正交记录——仓库设计
  方案 §7；实现模块 = ``joint.link_quota``，设计语义权威 = 设计文档
  §4.2/§4.4）
  - ``off``（缺省，F7）：配额门全部旁路（动作级准入不设链路/端口
    配额判据；link_quota 簿记语义见模块 docstring）；
  - ``static``：固定预算——链路 ``Q_init = max(1, floor(rho_eff))``
    逐配置硬件派生、端口平价门 ``B_HBM/(u_port+1) >= r_hat_KV`` 与
    merge bulk 名额 ``N_bulk = Q_init`` 常开（流类别三分与 merge
    预留语义见 ``link_quota`` 模块）；
  - ``aimd``：static 基础上加链路 Q 的 AIMD 动态调整律（控制律本体
    见 C10；``aimd`` 发射时自动注入 ``--link-telemetry`` 的耦合规则
    F7 在发射层落地 = C11，非本解析处职责——本模块只解析与记录）。

  quota 正交性的唯一耦合例外（F7）：``scheduler_mode=face_static``
  强制 ``quota_mode=off``（对照臂语义，见 scheduler_mode face_static
  节）——非 off 配置显式覆盖为 off 并 manifest 注记，不拒绝启动。
* boundary_policy（D 边界策略，测量仪器开关——验收纪律第 2 类
  "一次映射 vs 两阶段闭环对照"的对照面，2026-09-28）
  - ``off``（缺省）：两阶段闭环（D18 主语义）——P 准入提交 P；真实
    P→D 边界以真实快照做恰好一次全候选遍历重估（``select_decode_
    target_and_action``，式 1b/1c）。
  - ``p-continuation``：**一次映射对照臂**（唯一合法非缺省值）——D
    边界不做二次全候选重估，直接按 P 准入时
    ``estimate_P_with_continuation`` 披露的默认延续形态提交：
    decode_instance = prefill_instance、decode_action = 默认延续动作
    （stay@i_P；P 末工作副本恒为完整型布局——增量型布局族已随
    remote-read 动作退役删除）。语义要点：延续 =
    原地继续、无迁移腿，D 提交后的完成归属/构图器/merge 链路与两阶段
    的延续分支完全同路（同一 commit 路径，不新造平行路径）；掩码交互
    ——P 准入已按掩码过滤（P 动作在掩码内），延续动作 stay 由本模块
    ``__post_init__`` 钉在掩码内（见下 fail-closed 裁决），故无需
    边界重查。
    **仪器纪律**：仅供对照实验（测量仪器模式，仿 JOINT_FORCE_* 钩子
    先例——但本开关是 run 级策略而非逐请求钩子），生产 run 不设；
    生效时 D 决策行以专用披露键 ``boundary_policy`` 披露（off 时键
    缺席 = 行为零漂移）；与 probe 强制钩子
    （JOINT_FORCE_BOUNDARY_ACTION / JOINT_FORCE_ENTRY_ACTION）互斥
    武装（同设 = 仪器二义，调度器启动失败——sh30 消费侧裁决）。
    fail-closed 裁决（本模块单一裁决点）：p-continuation 要求掩码
    放行 stay——P 动作（stay/copy/recompute）的默认延续恒为
    stay，掩码剔除 stay 时一次映射将在边界提交掩码外动作，违反
    "掩码是候选集上限"合同，该组合启动即拒。


八组合固定映射（§7.1 表）：

  combo | category_mode | scheduler_mode | layer_policy
  ------+---------------+----------------+----------------------
  none  | lru           | load-first     | minimal_layer_groups
  T     | typed         | load-first     | minimal_layer_groups
  J     | lru           | joint          | minimal_layer_groups
  E     | lru           | load-first     | adaptive
  TJ    | typed         | joint          | minimal_layer_groups
  TE    | typed         | load-first     | adaptive
  JE    | lru           | joint          | adaptive
  TJE   | typed         | joint          | adaptive

环境变量（一次读取，fail-closed；与仓内 SH30_ABLATION 同一风格）：

* ``JOINT_ABLATION_COMBO``  -- 八组合预设名（none/T/J/E/TJ/TE/JE/TJE）；
* ``JOINT_CATEGORY_MODE``   -- 显式 T 开关（typed|lru）；
* ``JOINT_SCHEDULER_MODE``  -- 显式 J 开关（joint|load-first|
                               affinity-first|face_static——末者为
                               C17 静态距离对照臂，非八组合成员）；
* ``JOINT_LAYER_POLICY``    -- 显式 E 开关（adaptive|legacy_half|
                               minimal_layer_groups）；
* ``JOINT_ALLOWED_ACTIONS`` -- M2 动作掩码（all|逗号分隔的
                               ALL_ACTIONS 子集，缺省 all；见上文
                               allowed_actions 节）；
* ``JOINT_QUOTA_MODE``      -- 通用流量治理/配额开关（off|static|aimd，
                               缺省 off（F7）；与三机制正交）。
* ``JOINT_BOUNDARY_POLICY`` -- D 边界策略（off|p-continuation，缺省
                               off；测量仪器开关——验收纪律第 2 类
                               "一次映射 vs 两阶段闭环对照"对照臂，
                               见上文 boundary_policy 节；生产 run
                               不设）。
* ``SH_TRAIN_MAX_ITER``     -- 列车步数上限 T_max（治理旋钮；解析 =
                               ``parse_train_max_iter`` 单一裁决点。
                               缺省/未设置按协议推导合法值（交付默认
                               8）；**显式 ≤0 fail-closed 拒绝**——
                               0=无界诊断配置违反 v2.0 §17.1 有界性
                               （D16：无界 T_max 会预建整轮节点），
                               非整数/带空白同样拒绝）。

预设与显式开关互斥：同一开关被两处指定即启动失败（fail-closed），不做
静默合并。所有取值大小写敏感、禁空白（同 SH30_ABLATION 数值合同）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

CATEGORY_MODES = ("typed", "lru")
#: 调度模式取值域（face_static = C17 静态距离对照臂：policy variant，
#: 不进八组合——COMBO_PRESETS 的 scheduler 档恒为 joint/load-first）。
SCHEDULER_MODES = ("joint", "load-first", "affinity-first", "face_static")
LAYER_POLICIES = ("adaptive", "legacy_half", "minimal_layer_groups")
#: 三动作全集与规范序（M2 动作掩码定义源，2026-09-26：joint_cost_model
#: .ACTION_ORDER 从此派生——定义源放本模块避免环导入）。掩码
#: JOINT_ALLOWED_ACTIONS 是候选集上限不是强制，不改此序。
#: 2026-10-14 远读退役：第四元 ``remote-read`` 及其执行/计价/配置
#: 全链路删除，动作空间收缩为 stay/recompute/copy 三动作（MODEL_
#: SEMANTICS_VERSION v7 世代）。
ALL_ACTIONS = ("stay", "recompute", "copy")
#: ``allowed_actions`` 的全集哨兵值（= ALL_ACTIONS，无剔除）。
ALLOWED_ACTIONS_ALL = "all"
#: 通用流量治理/配额模式（F7：缺省 off；static/aimd 语义见模块
#: docstring quota_mode 节与 joint.link_quota）。
QUOTA_MODES = ("off", "static", "aimd")
#: D 边界策略取值域（测量仪器开关；缺省 off = 两阶段闭环 D18 主语义，
#: 唯一合法非缺省值 p-continuation = 一次映射对照臂——语义与 fail-closed
#: 裁决见模块 docstring boundary_policy 节）。
BOUNDARY_POLICY_OFF = "off"
BOUNDARY_POLICY_P_CONTINUATION = "p-continuation"
BOUNDARY_POLICIES = (BOUNDARY_POLICY_OFF, BOUNDARY_POLICY_P_CONTINUATION)

# 模式常量（调度器/策略模块引用；与上表同源）。
CATEGORY_TYPED = "typed"
CATEGORY_LRU = "lru"
SCHEDULER_MODE_JOINT = "joint"
SCHEDULER_MODE_LOAD_FIRST = "load-first"
SCHEDULER_MODE_AFFINITY = "affinity-first"
#: 静态距离对照臂常量（C17；身份纪律见模块 docstring scheduler_mode
#: face_static 节——本文内部的静态距离变体，非完整 FACE）。
SCHEDULER_MODE_FACE_STATIC = "face_static"
LAYER_POLICY_ADAPTIVE = "adaptive"
LAYER_POLICY_LEGACY_HALF = "legacy_half"
LAYER_POLICY_MINIMAL = "minimal_layer_groups"
#: 配额模式常量（与 QUOTA_MODES 同源；link_quota 模块按同值字符串消费）。
QUOTA_OFF = "off"
QUOTA_STATIC = "static"
QUOTA_AIMD = "aimd"

#: 列车步数上限 T_max 的缺省合法值（D16 有界性，v2.0 §17.1）——
#: 协议推导的交付默认（§3.2.8/§7.4 A2 对拍裁决，sh_1.0 母本统一）。
#: 未设置 SH_TRAIN_MAX_ITER 时按本值生效；显式 ≤0 一律拒绝（见
#: ``parse_train_max_iter`` 单一裁决点）。
TRAIN_MAX_ITER_DEFAULT = 8

#: D17（v2.0 §18/D17，2026-09-27）：**模型语义版本号**——代价模型公式、
#: 字节/时间口径与执行协议（两阶段 P→D 边界、有限字节双缓冲、重算语义、
#: 每路径份额等）整体的机读语义身份，随 run manifest（``manifest_dict``
#: 键 ``model_semantics_version``）与逐条决策日志落盘。裁定（v2.0 §18
#: 末段）：**语义（公式、字节口径、协议）改变时必须升级本版本号，旧实验
#: 不自动代表新模型；不能只修改注释而把旧数值当作新物理语义。** 本值
#: "v3.2-2phase-20260928" 标识两阶段边界协议世代的 D06 式(6) 收口修订
#: （前代 "v3.1-2phase-20260927" = 修复轮二；首版 "v3-2phase-20260927"，
#: 2026-09-26 前为单阶段同址估价 + 步数 credit 世代）——本修订：D 边界
#: remote 候选的 decode credit 暂存腿（双缓冲闭式供给基元/tile x）改走
#: decode 阶段冻结 HBM 份额（k_e=2，与同端口同阶段并发的 decode 计算
#: 读同 m_e+2 分母；修复前裸注册表 m_e+1 偏快 ⇒ remote D 边界候选偏
#: 乐观），D 边界 remote 候选腿时延相对前代变长——旧记录持旧值时不得
#: 重命名为新协议结果、不得自动充当新模型的证据。
#: W3（2026-09-29）升版 v4-projection：四动作 J 正式计价切换为事件前推
#: （Q 冻结/K=8 折叠/阶段冻结份额退役为事件重算——joint_cost_model 模块
#: docstring 的 W3 世代段）。旧实验不自动代表新模型。
#: 2026-10-02 升版 v5-projection：计算腿分解纯计算与动态 HBM 工作量，
#: 修复多跳背景身份、剩余进度与混合 KV 字节口径；未完成的新增需求
#: 候选世界契约强制 shadow。对应旧版本验证登记不得自动沿用。
#: 2026-10-14 升版 v7-three-actions：remote-read 动作及其执行/计价/
#: 配置全链路退役——联合决策动作空间从四动作收缩为 stay/recompute/
#: copy 三动作（W10 hybrid 随之退役；增量型 P→D 边界布局族一步删除，
#: 恒完整型单族；远读审计/决策日志 schema 字段删除并升版）。v7 起
#: 与 v6 及以前旧实验数据不可比，正式数据生产须按新版本重跑。
MODEL_SEMANTICS_VERSION = "v7-three-actions-20261014"

#: 八组合预设名（固定映射，2026-09-13 裁定；无 J 组合一律 load-first）。
COMBO_NAMES = ("none", "T", "J", "E", "TJ", "TE", "JE", "TJE")

#: 八组合 -> (category_mode, scheduler_mode, layer_policy) 固定映射表。
COMBO_PRESETS: dict[str, tuple[str, str, str]] = {
    "none": ("lru", "load-first", "minimal_layer_groups"),
    "T": ("typed", "load-first", "minimal_layer_groups"),
    "J": ("lru", "joint", "minimal_layer_groups"),
    "E": ("lru", "load-first", "adaptive"),
    "TJ": ("typed", "joint", "minimal_layer_groups"),
    "TE": ("typed", "load-first", "adaptive"),
    "JE": ("lru", "joint", "adaptive"),
    "TJE": ("typed", "joint", "adaptive"),
}

_ENV_COMBO = "JOINT_ABLATION_COMBO"
_ENV_CATEGORY = "JOINT_CATEGORY_MODE"
_ENV_SCHEDULER = "JOINT_SCHEDULER_MODE"
_ENV_LAYER = "JOINT_LAYER_POLICY"
#: 退役环境变量（任何显式值 fail-closed，同 SH30_ABLATION 姿态——
#: 陈旧脚本不得静默假装旧机制仍在生效）：
#: * JOINT_REMOTE_CREDIT_ITERS：D09 步数块大小 K 随字节 tile 化消解；
#: * JOINT_REMOTE_ACTIONS / JOINT_REMOTE_READ_TILE_BYTES /
#:   JOINT_REMOTE_READ_PARTIAL：2026-10-14 远读机制整体退役
#:   （remote-read 动作、tile/双缓冲执行链与 PARTIAL 适用面消融全链路
#:   删除——三动作化后不再有开关语义可配置）。
_RETIRED_ENV_KEYS = (
    "JOINT_REMOTE_CREDIT_ITERS",
    "JOINT_REMOTE_ACTIONS",
    "JOINT_REMOTE_READ_TILE_BYTES",
    "JOINT_REMOTE_READ_PARTIAL",
)
_ENV_ALLOWED_ACTIONS = "JOINT_ALLOWED_ACTIONS"
_ENV_QUOTA_MODE = "JOINT_QUOTA_MODE"
_ENV_BOUNDARY_POLICY = "JOINT_BOUNDARY_POLICY"


class JointConfigError(ValueError):
    """fail-closed 配置错误（非法值 / 预设与显式开关冲突）。"""


@dataclass(frozen=True)
class JointMechanismConfig:
    """三机制开关的已解析运行配置。

    属性即三开关生效值；``combo`` 在经八组合预设指定时记录预设名（显式
    开关路径为 None）。manifest 记录三开关、预设名（若有）与来源环境
    变量，日志标签与之一致（§7.1）。
    """

    category_mode: str
    scheduler_mode: str
    layer_policy: str
    # M2 动作掩码（all|逗号分隔的 ALL_ACTIONS 子集，缺省 all——见模块
    # docstring allowed_actions 节）。掩码是候选集上限不是强制：命中
    # 掩码外且原本可用的动作由消费侧（JCM applicable_actions 尾部过滤
    # 与 SH _classify_physical_feasibility 第二过滤点）统一剔除；掩码
    # 剔除 recompute 致零适用维持 fail-closed。
    allowed_actions: str = ALLOWED_ACTIONS_ALL
    # 通用流量治理/配额模式（off|static|aimd，缺省 off——F7；与三机制
    # 正交。SH 侧接线与 aimd⇒--link-telemetry 注入归 C11）。
    # face_static 耦合例外见下 quota_forced_off（构造点统一强制，非仅
    # env 解析路径——直接构造同样生效）。
    quota_mode: str = QUOTA_OFF
    # D 边界策略（off|p-continuation，缺省 off——测量仪器开关，验收
    # 纪律第 2 类"一次映射 vs 两阶段闭环对照"对照臂；语义/fail-closed
    # 裁决见模块 docstring boundary_policy 节。生产 run 不设）。
    boundary_policy: str = BOUNDARY_POLICY_OFF
    # 派生旗标（__post_init__ 重算，非构造参数语义）：True = 构造时的
    # quota_mode 为非 off 值、因 scheduler_mode=face_static 被强制覆盖
    # 为 off（F7；manifest 注记 "face_static forces quota-off"）。
    quota_forced_off: bool = False
    combo: Optional[str] = None
    # 非构造参数：来源环境变量清单（parse_joint_config 注入；直接构造时
    # 为空 dict，manifest_dict 报空）。
    _sources: dict = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        if self._sources is None:
            object.__setattr__(self, "_sources", {})
        if self.category_mode not in CATEGORY_MODES:
            raise JointConfigError(
                f"category_mode must be one of {CATEGORY_MODES}, got "
                f"{self.category_mode!r}")
        if self.scheduler_mode not in SCHEDULER_MODES:
            raise JointConfigError(
                f"scheduler_mode must be one of {SCHEDULER_MODES}, got "
                f"{self.scheduler_mode!r}")
        if self.layer_policy not in LAYER_POLICIES:
            raise JointConfigError(
                f"layer_policy must be one of {LAYER_POLICIES}, got "
                f"{self.layer_policy!r}")
        # M2 动作掩码（fail-closed：未知动作名/空 token/空白一律拒绝；
        # 与 env 解析共用 _parse_allowed_actions 单一裁决点）。
        if self.allowed_actions != ALLOWED_ACTIONS_ALL:
            _parse_allowed_actions(self.allowed_actions)
        if self.quota_mode not in QUOTA_MODES:
            raise JointConfigError(
                f"quota_mode must be one of {QUOTA_MODES}, got "
                f"{self.quota_mode!r}")
        # D 边界策略（测量仪器开关；取值域校验 + 与掩码的一致性裁决，
        # 单一裁决点——env 路径与直接构造同此）。
        if self.boundary_policy not in BOUNDARY_POLICIES:
            raise JointConfigError(
                f"boundary_policy must be one of {BOUNDARY_POLICIES}, got "
                f"{self.boundary_policy!r}")
        if (self.boundary_policy == BOUNDARY_POLICY_P_CONTINUATION
                and "stay" not in _parse_allowed_actions(
                    self.allowed_actions)):
            # 一次映射对照臂 × 掩码一致性（fail-closed）：P 动作
            # （stay/copy/recompute）的默认延续恒为 stay@i_P——掩码剔除
            # stay 时一次映射将在 D 边界提交掩码外动作，违反"掩码是候选
            # 集上限"合同，该组合启动即拒。
            raise JointConfigError(
                f"boundary_policy={BOUNDARY_POLICY_P_CONTINUATION!r} "
                "requires the allowed_actions mask to admit 'stay': the "
                "default continuation for every complete-layout P action "
                "(stay/copy/recompute) is stay@i_P, and a mask excluding "
                "stay would make the one-shot boundary commit an action "
                "outside the candidate ceiling")
        # face_static ⇒ quota-off（F7 唯一耦合例外；C17）：非 off 配置
        # 显式覆盖为 off 并注记，**不 fail-closed 拒绝启动**（对照臂
        # 语义——量静态掩码自身两侧误差，不被配额准入层混杂）。在
        # __post_init__ 强制 = 单一裁决点，直接构造（非 env 路径）同
        # 样生效。
        forced_off = (
            self.scheduler_mode == SCHEDULER_MODE_FACE_STATIC
            and self.quota_mode != QUOTA_OFF)
        object.__setattr__(self, "quota_forced_off", forced_off)
        if forced_off:
            object.__setattr__(self, "quota_mode", QUOTA_OFF)
        if self.combo is not None and self.combo not in COMBO_NAMES:
            raise JointConfigError(
                f"combo must be one of {COMBO_NAMES} or None, got "
                f"{self.combo!r}")
        if self.combo is not None:
            expected = COMBO_PRESETS[self.combo]
            actual = (
                self.category_mode, self.scheduler_mode, self.layer_policy)
            if actual != expected:
                raise JointConfigError(
                    f"combo {self.combo!r} requires {expected}, got "
                    f"{actual}")

    # ------------------------------------------------------------ 派生 --
    @property
    def t_enabled(self) -> bool:
        return self.category_mode == "typed"

    @property
    def j_enabled(self) -> bool:
        return self.scheduler_mode == "joint"

    @property
    def e_enabled(self) -> bool:
        return self.layer_policy == "adaptive"

    @property
    def allowed_action_set(self) -> frozenset:
        """M2 掩码放行动作集（JCM ``allowed_actions`` 注入源；"all" 派
        生全集、子集派生解析集——``_parse_allowed_actions`` 单一裁决点
        同源）。候选集上限不是强制。"""
        return _parse_allowed_actions(self.allowed_actions)

    @property
    def quota_enabled(self) -> bool:
        return self.quota_mode != QUOTA_OFF

    def manifest_dict(self) -> dict[str, Any]:
        """run manifest 的 joint 开关节（含 off 替代语义披露，§7.1）。"""
        return {
            # 合并语义版本披露键（2026-09-17《部分层逐出kv管理改造方案
            # 分析方案》§4.5/裁定④）：恒单值 "v2"——非开关（无候选档位，
            # 旧机制已物理删除），仅为重放工具/数据归档提供机读语义版
            # 本号；改动合并语义时同步此键并登记 PROVENANCE。
            "merge_semantics": "v2",
            # D17（v2.0 §18/D17）：模型语义版本号（固定配置 manifest 的
            # 版本身份键；语义改变必须升级，旧实验不自动代表新模型——
            # 常量定义处 MODEL_SEMANTICS_VERSION 注记为单一裁决点）。
            "model_semantics_version": MODEL_SEMANTICS_VERSION,
            "category_mode": self.category_mode,
            "scheduler_mode": self.scheduler_mode,
            "layer_policy": self.layer_policy,
            # M2 动作掩码（候选集上限不是强制；"all" = 无剔除）。
            "allowed_actions": self.allowed_actions,
            # 通用流量治理/配额开关（与三机制正交；aimd 的
            # --link-telemetry 注入事实由发射层另行记入 manifest = C11）。
            "quota_mode": self.quota_mode,
            "quota_enabled": self.quota_enabled,
            # D 边界策略（测量仪器开关；off = 两阶段闭环缺省零漂移，
            # p-continuation = 一次映射对照臂——见模块 docstring
            # boundary_policy 节）。
            "boundary_policy": self.boundary_policy,
            # face_static 对照臂注记（C17，仿 quota 键模式）：身份纪律
            # （设计文档 §5.3/§1.3）+ quota 强制覆盖事实（F7 唯一耦合
            # 例外——被覆盖时 source_env 仍保留 env 原值，差异即注记
            # 所指）。
            "face_static_identity": (
                "static-distance variant internal to this work "
                "(hop <= floor(rho) ball around the session KV anchor); "
                "NOT full FACE; until routing/execution/sharing/endpoint "
                "assumptions are aligned, the static distance ball is "
                "NOT claimed as the zero-load special case of the "
                "dynamic supply model (design doc 5.3/1.3)"
                if self.scheduler_mode == SCHEDULER_MODE_FACE_STATIC
                else None),
            "quota_forced_off": self.quota_forced_off,
            "quota_forced_off_note": (
                "face_static forces quota-off" if self.quota_forced_off
                else None),
            "combo": self.combo,
            "t_enabled": self.t_enabled,
            "j_enabled": self.j_enabled,
            "e_enabled": self.e_enabled,
            "off_substitutions": {
                "T-off": "type-agnostic LRU on the same legal victim set",
                "J-off": (
                    "sequential reference: fixed instance rule "
                    "(load-first in the eight-combo mapping) then KV "
                    "action at that instance"),
                "E-off": (
                    "minimal_layer_groups: release the fewest complete "
                    "layer groups per per-rank gap"),
            },
            "source_env": dict(getattr(self, "_sources", None) or {}),

        }

    def with_sources(self, sources: Mapping[str, str]) -> "JointMechanismConfig":
        object.__setattr__(self, "_sources", dict(sources))
        return self


def _read_env(lookup: Mapping[str, str], name: str) -> Optional[str]:
    raw = lookup.get(name)
    if raw is None:
        return None
    if raw != raw.strip() or not raw:
        # 数值合同：禁止空白/空值（同 SH30_ABLATION fail-closed 风格）。
        raise JointConfigError(
            f"{name} must be a non-empty value without surrounding "
            f"whitespace, got {raw!r}")
    return raw


def parse_train_max_iter(raw: Optional[str]) -> int:
    """列车步数上限 T_max 解析的单一裁决点（D16 有界性，v2.0 §17.1）。

    取值合同（调度器 ``_train_max_iter`` 的唯一供给源）：

    * ``raw is None``（缺省/未设置）→ 按协议推导合法值
      ``TRAIN_MAX_ITER_DEFAULT``（=8）；
    * 显式设置必须为正整数字符串——**显式 ≤0 fail-closed 拒绝**：
      0=无界诊断配置违反 v2.0 §17.1 有界性（无界 T_max 的列车折叠/
      预建面会扩张到整轮节点数），负值同拒；非整数/带空白/空值同拒
      （数值合同，同 ``_read_env`` 姿态）。

    只拒显式 ≤0：缺省路径（None → 协议推导值）不受影响——测试替身
    直接改 ``_train_max_iter`` 属性不经本函数，语义测试的"自然边界"
    夹具不受本合同约束。
    """
    if raw is None:
        return TRAIN_MAX_ITER_DEFAULT
    message = (
        f"SH_TRAIN_MAX_ITER must be a positive integer train-length cap "
        f"(default {TRAIN_MAX_ITER_DEFAULT} when unset); got {raw!r}: an "
        "explicit non-positive cap (0 = unbounded diagnostic) violates "
        "the boundedness contract (v2.0 sec 17.1) and is rejected "
        "fail-closed")
    # (2026-10-01 审查修复) ASCII 数字词法：裸 isdigit()/int() 放行
    # "1_0"/"+8"/上标数字（"²²" 在 int() 处抛裸 ValueError 而非
    # JointConfigError 通道）——显式拒绝，同 _read_env 姿态。
    if (raw != raw.strip() or not raw
            or not (raw.isascii() and raw.isdigit())):
        raise JointConfigError(message)
    try:
        value = int(raw, 10)
    except ValueError:
        raise JointConfigError(message) from None
    if value <= 0:
        raise JointConfigError(message)
    return value


def _parse_allowed_actions(value: str) -> frozenset:
    """M2 动作掩码取值校验 + 放行集派生（``__post_init__`` 与 env 解析
    共用单一裁决点，fail-closed）。

    ``"all"`` = 全集（无剔除）；否则逗号分隔的 ``ALL_ACTIONS`` 子集：
    未知动作名 / 空 token（含逗号悬挂）/ 带空白 token 一律
    JointConfigError；重复 token 容忍（frozenset 集合语义去重）。
    """
    if value == ALLOWED_ACTIONS_ALL:
        return frozenset(ALL_ACTIONS)
    tokens = value.split(",")
    for token in tokens:
        if token not in ALL_ACTIONS:
            raise JointConfigError(
                f"allowed_actions must be {ALLOWED_ACTIONS_ALL!r} or a "
                f"comma-separated subset of {ALL_ACTIONS}, got "
                f"{value!r} (offending token {token!r})")
    return frozenset(tokens)


def parse_joint_config(
    env: Optional[Mapping[str, str]] = None,
) -> "JointMechanismConfig":
    """解析运行配置（默认 = 完整三机制 T+J+E）。

    ``env`` 缺省读 ``os.environ``（一次读取；测试可注入）。预设与显式
    开关指定同一机制即 fail-closed。
    """
    lookup = os.environ if env is None else env

    # 退役开关扫描（同 SH30_ABLATION 姿态）：任何显式值即启动失败——
    # 陈旧脚本不得静默假装旧机制/旧粒度参数仍在生效。
    for retired in _RETIRED_ENV_KEYS:
        if lookup.get(retired) is not None:
            raise JointConfigError(
                f"{retired} is retired in astra-sim-joint (any explicit "
                "value fails closed; the mechanism it configured has "
                "been removed)")

    combo_raw = _read_env(lookup, _ENV_COMBO)
    category_raw = _read_env(lookup, _ENV_CATEGORY)
    scheduler_raw = _read_env(lookup, _ENV_SCHEDULER)
    layer_raw = _read_env(lookup, _ENV_LAYER)
    allowed_actions_raw = _read_env(lookup, _ENV_ALLOWED_ACTIONS)
    quota_mode_raw = _read_env(lookup, _ENV_QUOTA_MODE)
    boundary_policy_raw = _read_env(lookup, _ENV_BOUNDARY_POLICY)

    if combo_raw is not None and combo_raw not in COMBO_NAMES:
        raise JointConfigError(
            f"{_ENV_COMBO} must be one of {COMBO_NAMES}, got "
            f"{combo_raw!r}")

    combo = combo_raw
    sources: dict[str, str] = {}
    if combo is not None:
        sources[_ENV_COMBO] = combo
        category_mode, scheduler_mode, layer_policy = COMBO_PRESETS[combo]
    else:
        category_mode = category_raw or "typed"
        scheduler_mode = scheduler_raw or "joint"
        layer_policy = layer_raw or "adaptive"
        for name, value in (
            (_ENV_CATEGORY, category_raw),
            (_ENV_SCHEDULER, scheduler_raw),
            (_ENV_LAYER, layer_raw),
        ):
            if value is not None:
                sources[name] = value

    # 预设与显式开关互斥：同一机制被两处指定即失败（含与预设同值的
    # 重复指定），不静默合并。
    if combo is not None:
        for name, value in (
            (_ENV_CATEGORY, category_raw),
            (_ENV_SCHEDULER, scheduler_raw),
            (_ENV_LAYER, layer_raw),
        ):
            if value is not None:
                raise JointConfigError(
                    f"{name} conflicts with {_ENV_COMBO}={combo!r}; "
                    "specify either the combo preset or explicit switches, "
                    "not both")

    # M2 动作掩码（all|逗号分隔的 ALL_ACTIONS 子集，缺省 all；与三机制/
    # combo 预设及 quota 开关正交，不参与预设互斥）。语义裁定：掩码是
    # 候选集上限不是强制；臂预设恒含 recompute（recompute-only=
    # {recompute}+load-first；always-copy={copy,recompute}）；掩码剔除
    # recompute 且零适用 ⇒ 上层 fail-closed（非容量拒绝）。
    allowed_actions = allowed_actions_raw or ALLOWED_ACTIONS_ALL
    if allowed_actions != ALLOWED_ACTIONS_ALL:
        _parse_allowed_actions(allowed_actions)
    if allowed_actions_raw is not None:
        sources[_ENV_ALLOWED_ACTIONS] = allowed_actions

    # 通用流量治理/配额开关（off|static|aimd，缺省 off——F7；与三机制/
    # combo 预设正交，不参与预设互斥）。aimd ⇒ 发射层自动注入
    # --link-telemetry 的耦合规则（F7）在 joint_runner/
    # run_online_strategy 落地 = C11，本解析处只取值与记录。
    quota_mode = quota_mode_raw or QUOTA_OFF
    if quota_mode not in QUOTA_MODES:
        raise JointConfigError(
            f"{_ENV_QUOTA_MODE} must be one of {QUOTA_MODES}, got "
            f"{quota_mode!r}")
    if quota_mode_raw is not None:
        sources[_ENV_QUOTA_MODE] = quota_mode

    # D 边界策略（off|p-continuation，缺省 off——测量仪器开关，与三
    # 机制/combo 预设及 quota/掩码开关正交，不参与预设互斥；掩码一致
    # 性裁决在 __post_init__ 单一裁决点，env 与直接构造同受）。
    # 生产 run 不设。
    boundary_policy = boundary_policy_raw or BOUNDARY_POLICY_OFF
    if boundary_policy not in BOUNDARY_POLICIES:
        raise JointConfigError(
            f"{_ENV_BOUNDARY_POLICY} must be one of {BOUNDARY_POLICIES}, "
            f"got {boundary_policy!r}")
    if boundary_policy_raw is not None:
        sources[_ENV_BOUNDARY_POLICY] = boundary_policy

    if category_mode not in CATEGORY_MODES:
        raise JointConfigError(
            f"{_ENV_CATEGORY} must be one of {CATEGORY_MODES}, got "
            f"{category_mode!r}")
    if scheduler_mode not in SCHEDULER_MODES:
        raise JointConfigError(
            f"{_ENV_SCHEDULER} must be one of {SCHEDULER_MODES}, got "
            f"{scheduler_mode!r}")
    if layer_policy not in LAYER_POLICIES:
        raise JointConfigError(
            f"{_ENV_LAYER} must be one of {LAYER_POLICIES}, got "
            f"{layer_policy!r}")

    return JointMechanismConfig(
        category_mode=category_mode,
        scheduler_mode=scheduler_mode,
        layer_policy=layer_policy,
        allowed_actions=allowed_actions,
        quota_mode=quota_mode,
        boundary_policy=boundary_policy,
        combo=combo,
    ).with_sources(sources)


def require_joint_config(value: Any) -> JointMechanismConfig:
    """类型守卫（调度器装配处 fail-closed）。"""
    if not isinstance(value, JointMechanismConfig):
        raise JointConfigError(
            f"expected JointMechanismConfig, got {type(value).__name__}")
    return value
