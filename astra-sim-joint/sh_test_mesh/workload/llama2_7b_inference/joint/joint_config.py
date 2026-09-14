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
* layer_policy（E 开关）
  - ``adaptive``           ：E-on——按层消费期限计算最小热前缀 k_hide 的
    软目标式逐出（§5）。
  - ``minimal_layer_groups``：E-off——同一合法对象排序下按逐 rank 缺口
    释放最少完整层组，不计算隐藏目标。
  - ``legacy_half``        ：原半层逻辑的隔离适配，仅作回归/辅助对照。
* remote_actions（与三机制正交的能力消融开关，单独记录）
  - ``on`` / ``off``：off 时 remote-read 动作从候选动作集中移除，全部
    instance、其余动作、驱逐、合并与生命周期不变。

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
* ``JOINT_SCHEDULER_MODE``  -- 显式 J 开关（joint|load-first|affinity-first）；
* ``JOINT_LAYER_POLICY``    -- 显式 E 开关（adaptive|legacy_half|
                               minimal_layer_groups）；
* ``JOINT_REMOTE_ACTIONS``  -- remote 能力开关（on|off，缺省 on）。

预设与显式开关互斥：同一开关被两处指定即启动失败（fail-closed），不做
静默合并。所有取值大小写敏感、禁空白（同 SH30_ABLATION 数值合同）。
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Mapping, Optional

CATEGORY_MODES = ("typed", "lru")
SCHEDULER_MODES = ("joint", "load-first", "affinity-first")
LAYER_POLICIES = ("adaptive", "legacy_half", "minimal_layer_groups")
REMOTE_ACTION_MODES = ("on", "off")

# 模式常量（调度器/策略模块引用；与上表同源）。
CATEGORY_TYPED = "typed"
CATEGORY_LRU = "lru"
SCHEDULER_MODE_JOINT = "joint"
SCHEDULER_MODE_LOAD_FIRST = "load-first"
SCHEDULER_MODE_AFFINITY = "affinity-first"
LAYER_POLICY_ADAPTIVE = "adaptive"
LAYER_POLICY_LEGACY_HALF = "legacy_half"
LAYER_POLICY_MINIMAL = "minimal_layer_groups"
REMOTE_ON = "on"
REMOTE_OFF = "off"

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

    def manifest_dict(self) -> dict[str, Any]:
        """run manifest 的 joint 开关节（含 off 替代语义披露，§7.1）。"""
        return {
            "category_mode": self.category_mode,
            "scheduler_mode": self.scheduler_mode,
            "layer_policy": self.layer_policy,
            "remote_actions": self.remote_actions,
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
        combo=combo,
    ).with_sources(sources)


def require_joint_config(value: Any) -> JointMechanismConfig:
    """类型守卫（调度器装配处 fail-closed）。"""
    if not isinstance(value, JointMechanismConfig):
        raise JointConfigError(
            f"expected JointMechanismConfig, got {type(value).__name__}")
    return value
