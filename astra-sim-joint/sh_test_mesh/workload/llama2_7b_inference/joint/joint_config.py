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
    stay/recompute/copy/remote-read。
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
* remote_actions（与三机制正交的能力消融开关，单独记录）
  - ``on`` / ``off``：off 时 remote-read 动作从候选动作集中移除，全部
    instance、其余动作、驱逐、合并与生命周期不变。
* remote-read 执行口径 = 逐 credit 交错流（唯一机制，2026-09-17 用户
  裁定：旧 v1 批量读流 + readiness barrier 串行口径**删除**，不作为
  开关可选项保留）。读流按列车切片、切片内按 K 迭代分块，credit 块
  只栅栏对应计算体块，读与 decode 计算重叠；决策计价同形态 =
  first_credit_ns + max(remaining_stream_ns, compute_ns)。字节语义
  与旧口径零变更（均匀终态上下文 + I1 字节守恒）。
* remote_credit_iters（credit 块大小 K——唯一机制的粒度参数）
  - ``auto``（缺省）：自适应 K = max(1, ceil(S_j / 8))——每列车每成员
    切片块数 M_j 上限 8，节点膨胀与 S 解耦（D1/D3）；
  - 正整数：显式 K（K=1 仅验证配置；K >= S_j 时单块切片、块 1 走
    原 pd_transfer 发射路径 = 逐字节等价锚）。
* remote_read_partial（PARTIAL 基 remote-read 适用面消融，2026-09-17
  《部分层逐出kv管理改造分析方案》需求①）
  - ``on``（缺省）：PARTIAL 基（前缀驻留 home＋后缀在池）可进
    remote-read 候选——混合形态：准入相后缀池恢复物化（热 KV）＋
    decode 相前缀层 [0, p) credit 读流；
  - ``off``：PARTIAL 基照旧不进 remote-read 候选（拒绝理由
    ``remote-read for partial sessions disabled``——N1(a) 旧理由
    ``suffix not directly readable at home`` 已随混合形态解除退役）。注意：这不是旧机制
    回归档——copy/recompute 是 J 常驻比较候选（on 档下仍可被选中），
    off 档只把 PARTIAL 基的 remote-read 移出候选集，与
    ``JOINT_REMOTE_ACTIONS`` 消融开关同一模式（裁定④边界澄清）。
* quota_mode（通用流量治理/配额开关，与三机制及 remote on/off 正交
  记录——仓库设计方案 §7；实现模块 = ``joint.link_quota``，设计语义
  权威 = 设计文档 §4.2/§4.4）
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
* ``JOINT_REMOTE_ACTIONS``  -- remote 能力开关（on|off，缺省 on）；
* ``JOINT_REMOTE_CREDIT_ITERS`` -- remote-read credit 块大小 K
                               （auto|正整数，缺省 auto；唯一执行机制
                               的粒度参数）；
* ``JOINT_REMOTE_READ_PARTIAL`` -- PARTIAL 基 remote-read 适用面消融
                               （on|off，缺省 on；见上文 remote_read_
                               partial 节）；
* ``JOINT_QUOTA_MODE``      -- 通用流量治理/配额开关（off|static|aimd，
                               缺省 off（F7）；与三机制及 remote 开关
                               正交，解析模式同 remote_actions）。

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
REMOTE_ACTION_MODES = ("on", "off")
REMOTE_CREDIT_ITERS_MODES = ("auto",)  # 或正整数字符串（见解析处校验）
REMOTE_READ_PARTIAL_MODES = ("on", "off")
#: 通用流量治理/配额模式（F7：缺省 off；static/aimd 语义见模块
#: docstring quota_mode 节与 joint.link_quota）。
QUOTA_MODES = ("off", "static", "aimd")

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
REMOTE_ON = "on"
REMOTE_OFF = "off"
REMOTE_CREDIT_AUTO = "auto"
#: 配额模式常量（与 QUOTA_MODES 同源；link_quota 模块按同值字符串消费）。
QUOTA_OFF = "off"
QUOTA_STATIC = "static"
QUOTA_AIMD = "aimd"
#: 自适应 K 的切片块数上限 M（D1 裁定 8~16 取保守下沿；K = ceil(S/M)
#: 使 M_j ≤ 8，节点膨胀按列车有界、与 S 解耦——README/D3 披露）。
REMOTE_CREDIT_ADAPTIVE_MAX_BLOCKS = 8

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
_ENV_REMOTE = "JOINT_REMOTE_ACTIONS"
_ENV_REMOTE_CREDIT_ITERS = "JOINT_REMOTE_CREDIT_ITERS"
_ENV_REMOTE_READ_PARTIAL = "JOINT_REMOTE_READ_PARTIAL"
_ENV_QUOTA_MODE = "JOINT_QUOTA_MODE"


class JointConfigError(ValueError):
    """fail-closed 配置错误（非法值 / 预设与显式开关冲突）。"""


@dataclass(frozen=True)
class JointMechanismConfig:
    """三机制 + remote 能力开关的已解析运行配置。

    属性即四开关生效值；``combo`` 在经八组合预设指定时记录预设名（显式
    开关路径为 None）。manifest 记录四开关、预设名（若有）与来源环境
    变量，日志标签与之一致（§7.1）。
    """

    category_mode: str
    scheduler_mode: str
    layer_policy: str
    remote_actions: str
    # remote-read credit 块大小 K（auto=自适应 | 显式正整数）。执行口径
    # 本身为逐 credit 交错流唯一机制（2026-09-17 用户裁定，旧 v1 批量
    # 串行口径已删除、无开关可选项）。
    remote_credit_iters: str = REMOTE_CREDIT_AUTO
    # PARTIAL 基 remote-read 适用面（on|off；需求①消融开关，缺省 on——
    # 见模块 docstring remote_read_partial 节，非旧机制回归档）。
    remote_read_partial: str = REMOTE_ON
    # 通用流量治理/配额模式（off|static|aimd，缺省 off——F7；与三机制
    # 及 remote 开关正交。SH 侧接线与 aimd⇒--link-telemetry 注入归 C11）。
    # face_static 耦合例外见下 quota_forced_off（构造点统一强制，非仅
    # env 解析路径——直接构造同样生效）。
    quota_mode: str = QUOTA_OFF
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
        if self.remote_actions not in REMOTE_ACTION_MODES:
            raise JointConfigError(
                f"remote_actions must be one of {REMOTE_ACTION_MODES}, got "
                f"{self.remote_actions!r}")
        if self.remote_credit_iters != REMOTE_CREDIT_AUTO:
            if not self.remote_credit_iters.isdigit() or (
                    int(self.remote_credit_iters) <= 0):
                raise JointConfigError(
                    f"remote_credit_iters must be {REMOTE_CREDIT_AUTO!r} or "
                    f"a positive integer, got {self.remote_credit_iters!r}")
        if self.remote_read_partial not in REMOTE_READ_PARTIAL_MODES:
            raise JointConfigError(
                f"remote_read_partial must be one of "
                f"{REMOTE_READ_PARTIAL_MODES}, got "
                f"{self.remote_read_partial!r}")
        if self.quota_mode not in QUOTA_MODES:
            raise JointConfigError(
                f"quota_mode must be one of {QUOTA_MODES}, got "
                f"{self.quota_mode!r}")
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
    def remote_enabled(self) -> bool:
        return self.remote_actions == "on"

    @property
    def remote_read_partial_enabled(self) -> bool:
        return self.remote_read_partial == "on"

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
            "category_mode": self.category_mode,
            "scheduler_mode": self.scheduler_mode,
            "layer_policy": self.layer_policy,
            "remote_actions": self.remote_actions,
            "remote_credit_iters": self.remote_credit_iters,
            "remote_read_partial": self.remote_read_partial,
            # 通用流量治理/配额开关（与三机制及 remote 正交；aimd 的
            # --link-telemetry 注入事实由发射层另行记入 manifest = C11）。
            "quota_mode": self.quota_mode,
            "quota_enabled": self.quota_enabled,
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


def parse_joint_config(
    env: Optional[Mapping[str, str]] = None,
) -> JointMechanismConfig:
    """解析运行配置（默认 = 完整三机制 T+J+E、remote on）。

    ``env`` 缺省读 ``os.environ``（一次读取；测试可注入）。预设与显式
    开关指定同一机制即 fail-closed。
    """
    lookup = os.environ if env is None else env

    combo_raw = _read_env(lookup, _ENV_COMBO)
    category_raw = _read_env(lookup, _ENV_CATEGORY)
    scheduler_raw = _read_env(lookup, _ENV_SCHEDULER)
    layer_raw = _read_env(lookup, _ENV_LAYER)
    remote_raw = _read_env(lookup, _ENV_REMOTE)
    remote_credit_iters_raw = _read_env(lookup, _ENV_REMOTE_CREDIT_ITERS)
    remote_read_partial_raw = _read_env(lookup, _ENV_REMOTE_READ_PARTIAL)
    quota_mode_raw = _read_env(lookup, _ENV_QUOTA_MODE)

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

    remote_actions = remote_raw or "on"
    if remote_actions not in REMOTE_ACTION_MODES:
        raise JointConfigError(
            f"{_ENV_REMOTE} must be one of {REMOTE_ACTION_MODES}, got "
            f"{remote_actions!r}")
    if remote_raw is not None:
        sources[_ENV_REMOTE] = remote_actions

    # remote-read credit 块大小 K（与 combo/三机制开关正交，同
    # remote_actions 的独立解析；auto|正整数，非法值 fail-closed）。执行
    # 口径本身无开关（逐 credit 交错流唯一机制，2026-09-17 用户裁定）。
    remote_credit_iters = (
        remote_credit_iters_raw or REMOTE_CREDIT_AUTO)
    if remote_credit_iters != REMOTE_CREDIT_AUTO:
        if (not remote_credit_iters.isdigit()
                or int(remote_credit_iters) <= 0):
            raise JointConfigError(
                f"{_ENV_REMOTE_CREDIT_ITERS} must be {REMOTE_CREDIT_AUTO!r} "
                f"or a positive integer, got {remote_credit_iters!r}")
    if remote_credit_iters_raw is not None:
        sources[_ENV_REMOTE_CREDIT_ITERS] = remote_credit_iters

    # PARTIAL 基 remote-read 适用面消融（与 remote_actions 正交：off 档
    # 只把 PARTIAL 基的 remote-read 移出候选集，copy/recompute 照常）。
    remote_read_partial = remote_read_partial_raw or REMOTE_ON
    if remote_read_partial not in REMOTE_READ_PARTIAL_MODES:
        raise JointConfigError(
            f"{_ENV_REMOTE_READ_PARTIAL} must be one of "
            f"{REMOTE_READ_PARTIAL_MODES}, got {remote_read_partial!r}")
    if remote_read_partial_raw is not None:
        sources[_ENV_REMOTE_READ_PARTIAL] = remote_read_partial

    # 通用流量治理/配额开关（off|static|aimd，缺省 off——F7；与三机制/
    # combo 预设及 remote 开关正交，同 remote_actions 的独立解析模式，
    # 不参与预设互斥）。aimd ⇒ 发射层自动注入 --link-telemetry 的耦合
    # 规则（F7）在 joint_runner/run_online_strategy 落地 = C11，本解析
    # 处只取值与记录。
    quota_mode = quota_mode_raw or QUOTA_OFF
    if quota_mode not in QUOTA_MODES:
        raise JointConfigError(
            f"{_ENV_QUOTA_MODE} must be one of {QUOTA_MODES}, got "
            f"{quota_mode!r}")
    if quota_mode_raw is not None:
        sources[_ENV_QUOTA_MODE] = quota_mode

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
        remote_actions=remote_actions,
        remote_credit_iters=remote_credit_iters,
        remote_read_partial=remote_read_partial,
        quota_mode=quota_mode,
        combo=combo,
    ).with_sources(sources)


def remote_credit_block_size(remote_credit_iters: str, steps: int) -> int:
    """credit 块大小 K 的单一裁决点（调度器切片与决策计价同源，§4.3.1）。

    ``steps`` = 本列车参与步数 S_j（执行侧）或因果时域估计步数（决策
    侧）。auto：K = max(1, ceil(S_j / M 上限))——块数 M_j = ceil(S_j/K)
    ≤ 上限，节点膨胀按列车有界（D1）；显式 K 原样返回（K=1 仅验证配置，
    K ≥ S_j 时单块 = v1 等价锚 I3a）。
    """
    if steps <= 0:
        raise JointConfigError(
            f"credit block size requires positive steps, got {steps}")
    if remote_credit_iters != REMOTE_CREDIT_AUTO:
        explicit = int(remote_credit_iters)
        return min(explicit, max(1, steps))
    return max(1, -(-steps // REMOTE_CREDIT_ADAPTIVE_MAX_BLOCKS))


def require_joint_config(value: Any) -> JointMechanismConfig:
    """类型守卫（调度器装配处 fail-closed）。"""
    if not isinstance(value, JointMechanismConfig):
        raise JointConfigError(
            f"expected JointMechanismConfig, got {type(value).__name__}")
    return value
